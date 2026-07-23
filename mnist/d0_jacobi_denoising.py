"""Exact two-cell Jacobi tools for the D0 Eulerian denoising gate.

This module is deliberately independent of experiment orchestration and of
the historical Euler--Maruyama sampler.  It fixes the orientation convention

``x = mass_at_head / (mass_at_tail + mass_at_head)``

and exposes the four disjoint torus matchings used by a palindromic Strang
composition.  For ``alpha=1`` the conditional two-cell transition density,
relative to ``Beta(1, 1)``, is evaluated with the Legendre expansion

``sum_n (2n+1) exp(-n(n+1)u) P_n(2x-1) P_n(2y-1)``.

The spectral evaluator never clamps a density, CDF, or score.  It accompanies
every result with explicit geometric tail bounds and fails closed when the
requested tolerance cannot be certified within the mode cap.

The exact Wright--Fisher ancestral-count sampler is intentionally *not*
implemented here.  A certified numerical inverse-CDF sampler is provided for
transition-law controls, but it does not expose the latent ``(M, L)`` needed
for the DDPM-like label.  Callers must not silently substitute it when a
latent-label cache is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


JACOBI_DENOISING_SCHEMA = "experiment12-d0-jacobi-denoising"
JACOBI_DENOISING_SCHEMA_VERSION = 1
JACOBI_ORIENTATION = "head-fraction"
JACOBI_ALPHA1_SPECTRAL_VERSION = "alpha1-legendre-certified-v1"
JACOBI_STRANG_VERSION = "four-color-palindromic-strang-v1"


__all__ = [
    "JACOBI_DENOISING_SCHEMA",
    "JACOBI_DENOISING_SCHEMA_VERSION",
    "JACOBI_ORIENTATION",
    "JACOBI_ALPHA1_SPECTRAL_VERSION",
    "JACOBI_STRANG_VERSION",
    "EdgeMatching",
    "StrangPhase",
    "Alpha1SpectralConfig",
    "Alpha1SpectralDiagnostics",
    "Alpha1SpectralEvaluation",
    "TorchAlpha1SpectralEvaluation",
    "SpectralConvergenceError",
    "SpectralInverseCDFConfig",
    "SpectralInverseCDFDiagnostics",
    "SpectralInverseCDFSamples",
    "build_four_color_matchings",
    "validate_four_color_matchings",
    "palindromic_strang_plan",
    "apply_matching_head_fractions",
    "jacobi_phase_exposure",
    "evaluate_alpha1_spectral",
    "evaluate_alpha1_spectral_torch_fixed_modes",
    "sample_alpha1_spectral_inverse_cdf",
    "jacobi_latent_label",
    "jacobi_component_relative_score",
    "linear_teacher_relative_density",
    "linear_teacher_arrival_score",
    "linear_teacher_denoising_mean",
    "denoising_mean_to_mass_flux",
]


@dataclass(frozen=True)
class EdgeMatching:
    """One perfect matching of oriented nearest-neighbour torus edges."""

    index: int
    name: str
    direction: str
    parity: int
    tails: np.ndarray
    heads: np.ndarray
    flux_indices: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(self.tails.size)


@dataclass(frozen=True)
class StrangPhase:
    """One phase in the fixed palindromic four-colour Strang sweep."""

    phase_index: int
    matching_index: int
    matching_name: str
    duration_fraction: float


def build_four_color_matchings(grid_size: int) -> tuple[EdgeMatching, ...]:
    """Return the four exact perfect matchings of an even periodic grid.

    Horizontal edges are oriented to the right and vertical edges downward.
    ``flux_indices`` match the historical layout: all horizontal edges first,
    followed by all vertical edges.
    """

    n = int(grid_size)
    if n <= 0 or n % 2:
        raise ValueError("grid_size must be a positive even integer")
    records: list[list[tuple[int, int, int]]] = [[], [], [], []]
    for row in range(n):
        for col in range(n):
            tail = row * n + col
            horizontal_head = row * n + ((col + 1) % n)
            vertical_head = ((row + 1) % n) * n + col
            records[col % 2].append((tail, horizontal_head, tail))
            records[2 + row % 2].append(
                (tail, vertical_head, n * n + tail)
            )
    names = (
        ("horizontal_even", "horizontal", 0),
        ("horizontal_odd", "horizontal", 1),
        ("vertical_even", "vertical", 0),
        ("vertical_odd", "vertical", 1),
    )
    result: list[EdgeMatching] = []
    for index, (name, direction, parity) in enumerate(names):
        edges = records[index]
        result.append(
            EdgeMatching(
                index=index,
                name=name,
                direction=direction,
                parity=parity,
                tails=np.asarray([edge[0] for edge in edges], dtype=np.int64),
                heads=np.asarray([edge[1] for edge in edges], dtype=np.int64),
                flux_indices=np.asarray(
                    [edge[2] for edge in edges], dtype=np.int64
                ),
            )
        )
    matchings = tuple(result)
    validate_four_color_matchings(n, matchings)
    return matchings


def validate_four_color_matchings(
    grid_size: int, matchings: Sequence[EdgeMatching]
) -> None:
    """Fail if matchings are not a disjoint four-colour edge partition."""

    n = int(grid_size)
    pixels = n * n
    if n <= 0 or n % 2:
        raise ValueError("grid_size must be a positive even integer")
    if len(matchings) != 4:
        raise ValueError("exactly four edge matchings are required")
    observed: set[tuple[int, int]] = set()
    observed_flux: set[int] = set()
    expected_names = (
        ("horizontal_even", "horizontal", 0),
        ("horizontal_odd", "horizontal", 1),
        ("vertical_even", "vertical", 0),
        ("vertical_odd", "vertical", 1),
    )
    expected_classes: list[set[tuple[int, int]]] = [set() for _ in range(4)]
    for row in range(n):
        for col in range(n):
            tail = row * n + col
            expected_classes[col % 2].add(
                (tail, row * n + ((col + 1) % n))
            )
            expected_classes[2 + row % 2].add(
                (tail, ((row + 1) % n) * n + col)
            )
    for expected_index, matching in enumerate(matchings):
        if int(matching.index) != expected_index:
            raise ValueError("matching indices must be ordered 0,1,2,3")
        expected_name, expected_direction, expected_parity = expected_names[
            expected_index
        ]
        if (
            matching.name != expected_name
            or matching.direction != expected_direction
            or int(matching.parity) != expected_parity
        ):
            raise ValueError("matching name, direction, or parity is incompatible")
        tails = np.asarray(matching.tails, dtype=np.int64)
        heads = np.asarray(matching.heads, dtype=np.int64)
        flux = np.asarray(matching.flux_indices, dtype=np.int64)
        if tails.shape != heads.shape or tails.shape != flux.shape:
            raise ValueError("matching tail/head/flux arrays must agree")
        if tails.size != pixels // 2:
            raise ValueError("each matching must contain N/2 edges")
        if np.any(tails < 0) or np.any(tails >= pixels):
            raise ValueError("tail index outside the grid")
        if np.any(heads < 0) or np.any(heads >= pixels):
            raise ValueError("head index outside the grid")
        if np.any(flux < 0) or np.any(flux >= 2 * pixels):
            raise ValueError("flux index outside the two-channel layout")
        incident = np.concatenate([tails, heads])
        if np.unique(incident).size != pixels:
            raise ValueError("a matching must touch every cell exactly once")
        matching_edges = set(
            zip(tails.tolist(), heads.tolist(), strict=True)
        )
        if matching_edges != expected_classes[expected_index]:
            raise ValueError("edge membership does not match the named colour")
        for tail, head in zip(tails.tolist(), heads.tolist(), strict=True):
            edge = (int(tail), int(head))
            if edge in observed:
                raise ValueError("an oriented edge occurs in multiple matchings")
            observed.add(edge)
        for flux_index in flux.tolist():
            if int(flux_index) in observed_flux:
                raise ValueError("a flux index occurs in multiple matchings")
            observed_flux.add(int(flux_index))
    expected = set()
    for row in range(n):
        for col in range(n):
            tail = row * n + col
            expected.add((tail, row * n + ((col + 1) % n)))
            expected.add((tail, ((row + 1) % n) * n + col))
    if observed != expected:
        raise ValueError("matchings do not partition the oriented grid edges")
    if observed_flux != set(range(2 * pixels)):
        raise ValueError("matchings do not partition the two-channel flux layout")


def palindromic_strang_plan() -> tuple[StrangPhase, ...]:
    """Return the immutable H0,H1,V0,V1,V0,H1,H0 phase plan."""

    indices = (0, 1, 2, 3, 2, 1, 0)
    names = (
        "horizontal_even",
        "horizontal_odd",
        "vertical_even",
        "vertical_odd",
    )
    fractions = (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
    return tuple(
        StrangPhase(
            phase_index=phase_index,
            matching_index=matching_index,
            matching_name=names[matching_index],
            duration_fraction=fractions[phase_index],
        )
        for phase_index, matching_index in enumerate(indices)
    )


def apply_matching_head_fractions(
    states: np.ndarray,
    matching: EdgeMatching,
    head_fractions: np.ndarray,
) -> np.ndarray:
    """Apply later head fractions while preserving every pair total exactly."""

    values = np.asarray(states)
    if values.ndim < 1:
        raise ValueError("states must have a final cell dimension")
    tails = np.asarray(matching.tails, dtype=np.int64)
    heads = np.asarray(matching.heads, dtype=np.int64)
    if values.shape[-1] <= int(max(tails.max(), heads.max())):
        raise ValueError("state cell dimension is incompatible with matching")
    y = np.asarray(head_fractions, dtype=values.dtype)
    expected = values.shape[:-1] + (matching.edge_count,)
    if y.shape != expected:
        raise ValueError(f"head_fractions must have shape {expected}")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("head fractions must be finite and lie in [0,1]")
    pair_total = values[..., tails] + values[..., heads]
    out = values.copy()
    out[..., tails] = pair_total * (1.0 - y)
    out[..., heads] = pair_total * y
    return out


def jacobi_phase_exposure(
    pair_total: np.ndarray | float,
    integrated_schedule_time: np.ndarray | float,
    *,
    alpha: float = 1.0,
    grid_spacing: float,
) -> np.ndarray:
    """Convert physical phase time to the dimensionless Jacobi exposure.

    ``integrated_schedule_time`` is the integral of the scalar schedule over
    the phase.  The returned exposure multiplies
    ``x(1-x)d_xx + alpha(1-2x)d_x``.
    """

    r, schedule_time = np.broadcast_arrays(
        np.asarray(pair_total, dtype=np.float64),
        np.asarray(integrated_schedule_time, dtype=np.float64),
    )
    alpha_value = float(alpha)
    h = float(grid_spacing)
    if not math.isfinite(alpha_value) or alpha_value <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("grid_spacing must be finite and positive")
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("pair_total must be finite and positive")
    if not np.all(np.isfinite(schedule_time)) or np.any(schedule_time < 0.0):
        raise ValueError("integrated_schedule_time must be finite and nonnegative")
    return (
        (2.0 * alpha_value + 1.0)
        * schedule_time
        / (alpha_value * h * h * r)
    )


@dataclass(frozen=True)
class Alpha1SpectralConfig:
    """Numerical contract for the certified alpha=1 Legendre expansion."""

    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-10
    max_modes: int = 4096

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.absolute_tolerance)) or not (
            float(self.absolute_tolerance) > 0.0
        ):
            raise ValueError("absolute_tolerance must be finite and positive")
        if not math.isfinite(float(self.relative_tolerance)) or not (
            float(self.relative_tolerance) >= 0.0
        ):
            raise ValueError("relative_tolerance must be finite and nonnegative")
        if int(self.max_modes) < 2:
            raise ValueError("max_modes must be at least two")


@dataclass(frozen=True)
class Alpha1SpectralDiagnostics:
    converged: bool
    modes_used: int
    max_modes: int
    max_density_tail_bound: float
    max_cdf_tail_bound: float
    max_derivative_tail_bound: float
    max_arrival_score_error_bound: float
    minimum_density_lower_bound: float
    minimum_density: float
    minimum_cdf: float
    maximum_cdf: float
    endpoint_cdf_count: int
    nonfinite_count: int
    negative_density_count: int
    reason: str
    orientation: str = JACOBI_ORIENTATION
    evaluator_version: str = JACOBI_ALPHA1_SPECTRAL_VERSION


@dataclass(frozen=True)
class Alpha1SpectralEvaluation:
    density: np.ndarray
    cdf: np.ndarray
    arrival_score: np.ndarray
    diagnostics: Alpha1SpectralDiagnostics


@dataclass(frozen=True)
class TorchAlpha1SpectralEvaluation:
    density: Tensor
    cdf: Tensor
    arrival_score: Tensor
    modes_used: int


class SpectralConvergenceError(RuntimeError):
    """Raised when a spectral value cannot be certified without correction."""

    def __init__(self, diagnostics: Alpha1SpectralDiagnostics):
        super().__init__(
            "alpha=1 spectral evaluation failed closed: " + diagnostics.reason
        )
        self.diagnostics = diagnostics


def _geometric_tail_bounds(
    first_omitted: int, exposure: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous uniform tails for density, CDF, and y derivative series."""

    m = int(first_omitted)
    if m < 1:
        raise ValueError("first omitted mode must be positive")
    u = np.asarray(exposure, dtype=np.float64)
    decay = np.exp(-float(m * (m + 1)) * u)

    density_ratio = (
        float(2 * m + 3) / float(2 * m + 1)
    ) * np.exp(-2.0 * float(m + 1) * u)
    density_first = float(2 * m + 1) * decay
    density_tail = np.where(
        density_ratio < 1.0,
        density_first / (1.0 - density_ratio),
        np.inf,
    )

    cdf_ratio = np.exp(-2.0 * float(m + 1) * u)
    cdf_tail = decay / (1.0 - cdf_ratio)

    derivative_ratio = (
        float(2 * m + 3)
        / float(2 * m + 1)
        * float(m + 2)
        / float(m)
        * np.exp(-2.0 * float(m + 1) * u)
    )
    derivative_first = float((2 * m + 1) * m * (m + 1)) * decay
    derivative_tail = np.where(
        derivative_ratio < 1.0,
        derivative_first / (1.0 - derivative_ratio),
        np.inf,
    )
    return density_tail, cdf_tail, derivative_tail


def _spectral_diagnostics(
    *,
    converged: bool,
    modes_used: int,
    config: Alpha1SpectralConfig,
    density: np.ndarray,
    cdf: np.ndarray,
    arrival_score: np.ndarray,
    density_tail: np.ndarray,
    cdf_tail: np.ndarray,
    derivative_tail: np.ndarray,
    score_error: np.ndarray,
    density_lower: np.ndarray,
    endpoint_mask: np.ndarray,
    reason: str,
) -> Alpha1SpectralDiagnostics:
    finite_mask = (
        np.isfinite(density) & np.isfinite(cdf) & np.isfinite(arrival_score)
    )
    return Alpha1SpectralDiagnostics(
        converged=bool(converged),
        modes_used=int(modes_used),
        max_modes=int(config.max_modes),
        max_density_tail_bound=float(np.max(density_tail)),
        max_cdf_tail_bound=float(np.max(cdf_tail)),
        max_derivative_tail_bound=float(np.max(derivative_tail)),
        max_arrival_score_error_bound=float(np.max(score_error)),
        minimum_density_lower_bound=float(np.min(density_lower)),
        minimum_density=float(np.min(density)),
        minimum_cdf=float(np.min(cdf)),
        maximum_cdf=float(np.max(cdf)),
        endpoint_cdf_count=int(np.count_nonzero(endpoint_mask)),
        nonfinite_count=int(finite_mask.size - np.count_nonzero(finite_mask)),
        negative_density_count=int(np.count_nonzero(density < 0.0)),
        reason=str(reason),
    )


def evaluate_alpha1_spectral(
    x: np.ndarray | float,
    y: np.ndarray | float,
    exposure: np.ndarray | float,
    *,
    config: Alpha1SpectralConfig | None = None,
    strict: bool = True,
) -> Alpha1SpectralEvaluation:
    """Evaluate the alpha=1 Jacobi density, CDF, and arrival score.

    The density is relative to the invariant uniform law.  ``arrival_score``
    is ``partial_y log k_u(y|x)`` under the head-fraction convention.
    Certification is uniform over the broadcast input array.  With ``strict``
    (the default), a cap hit, nonpositive certified density lower bound,
    negative density, or nonfinite value raises :class:`SpectralConvergenceError`.
    """

    numerical = config or Alpha1SpectralConfig()
    x_array, y_array, u_array = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(exposure, dtype=np.float64),
    )
    if not np.all(np.isfinite(x_array)) or np.any(x_array < 0.0) or np.any(
        x_array > 1.0
    ):
        raise ValueError("x must be finite and lie in [0,1]")
    if not np.all(np.isfinite(y_array)) or np.any(y_array < 0.0) or np.any(
        y_array > 1.0
    ):
        raise ValueError("y must be finite and lie in [0,1]")
    if not np.all(np.isfinite(u_array)) or np.any(u_array <= 0.0):
        raise ValueError("exposure must be finite and strictly positive")

    zx = 2.0 * x_array - 1.0
    zy = 2.0 * y_array - 1.0
    endpoint_mask = (y_array == 0.0) | (y_array == 1.0)

    density = np.ones_like(zx)
    cdf = y_array.copy()
    derivative = np.zeros_like(zx)

    px_previous = np.ones_like(zx)
    px_current = zx.copy()
    py_previous = np.ones_like(zy)
    py_current = zy.copy()
    dpy_previous = np.zeros_like(zy)
    dpy_current = np.ones_like(zy)

    density_tail = np.full_like(zx, np.inf)
    cdf_tail = np.full_like(zx, np.inf)
    derivative_tail = np.full_like(zx, np.inf)
    score_error = np.full_like(zx, np.inf)
    density_lower = np.full_like(zx, -np.inf)
    converged = False
    modes_used = 1

    for n in range(1, int(numerical.max_modes)):
        py_next = (
            float(2 * n + 1) * zy * py_current - float(n) * py_previous
        ) / float(n + 1)
        dpy_next = (
            float(2 * n + 1) * (py_current + zy * dpy_current)
            - float(n) * dpy_previous
        ) / float(n + 1)

        coefficient = float(2 * n + 1) * np.exp(
            -float(n * (n + 1)) * u_array
        )
        density = density + coefficient * px_current * py_current
        derivative = derivative + coefficient * px_current * (2.0 * dpy_current)
        cdf = cdf + (
            0.5
            * np.exp(-float(n * (n + 1)) * u_array)
            * px_current
            * (py_next - py_previous)
        )
        modes_used = n + 1

        first_omitted = n + 1
        density_tail, cdf_tail_raw, derivative_tail = _geometric_tail_bounds(
            first_omitted, u_array
        )
        cdf_tail = np.where(endpoint_mask, 0.0, cdf_tail_raw)
        density_lower = density - density_tail
        with np.errstate(divide="ignore", invalid="ignore"):
            score = derivative / density
            score_error = (
                derivative_tail * np.abs(density)
                + np.abs(derivative) * density_tail
            ) / (density_lower * np.abs(density))
        score_error = np.where(density_lower > 0.0, score_error, np.inf)
        density_tolerance = float(numerical.absolute_tolerance) + float(
            numerical.relative_tolerance
        ) * np.abs(density)
        cdf_tolerance = float(numerical.absolute_tolerance) + float(
            numerical.relative_tolerance
        ) * np.maximum(1.0, np.abs(cdf))
        score_tolerance = float(numerical.absolute_tolerance) + float(
            numerical.relative_tolerance
        ) * np.maximum(1.0, np.abs(score))
        converged_mask = (
            (density_tail <= density_tolerance)
            & (cdf_tail <= cdf_tolerance)
            & (score_error <= score_tolerance)
            & (density_lower > 0.0)
            & np.isfinite(score)
        )
        if bool(np.all(converged_mask)):
            converged = True
            break

        px_next = (
            float(2 * n + 1) * zx * px_current - float(n) * px_previous
        ) / float(n + 1)
        px_previous, px_current = px_current, px_next
        py_previous, py_current = py_current, py_next
        dpy_previous, dpy_current = dpy_current, dpy_next

    # CDF endpoint values are analytic identities, not numerical clipping.
    cdf = np.where(y_array == 0.0, 0.0, np.where(y_array == 1.0, 1.0, cdf))
    with np.errstate(divide="ignore", invalid="ignore"):
        arrival_score = derivative / density
    finite = (
        np.all(np.isfinite(density))
        and np.all(np.isfinite(cdf))
        and np.all(np.isfinite(arrival_score))
    )
    positive = bool(np.all(density > 0.0) and np.all(density_lower > 0.0))
    passed = bool(converged and finite and positive)
    if not converged:
        reason = "mode cap reached before density/CDF/score certification"
    elif not finite:
        reason = "nonfinite spectral density, CDF, or arrival score"
    elif not positive:
        reason = "density is not certified strictly positive"
    else:
        reason = "certified"
    diagnostics = _spectral_diagnostics(
        converged=passed,
        modes_used=modes_used,
        config=numerical,
        density=density,
        cdf=cdf,
        arrival_score=arrival_score,
        density_tail=density_tail,
        cdf_tail=cdf_tail,
        derivative_tail=derivative_tail,
        score_error=score_error,
        density_lower=density_lower,
        endpoint_mask=endpoint_mask,
        reason=reason,
    )
    result = Alpha1SpectralEvaluation(
        density=density,
        cdf=cdf,
        arrival_score=arrival_score,
        diagnostics=diagnostics,
    )
    if strict and not diagnostics.converged:
        raise SpectralConvergenceError(diagnostics)
    return result


def evaluate_alpha1_spectral_torch_fixed_modes(
    x: Tensor,
    y: Tensor,
    exposure: Tensor,
    *,
    modes: int,
) -> TorchAlpha1SpectralEvaluation:
    """Evaluate a fixed, externally certified Legendre truncation in Torch.

    This helper intentionally does not choose a truncation on the GPU.  Use
    :func:`evaluate_alpha1_spectral` over the same support to certify a mode
    count, then compare/evaluate that exact truncation on the target device.
    """

    if int(modes) < 2:
        raise ValueError("modes must be at least two")
    x_value, y_value, u_value = torch.broadcast_tensors(x, y, exposure)
    if not x_value.is_floating_point():
        raise ValueError("x, y, and exposure must use a floating dtype")
    if y_value.dtype != x_value.dtype or u_value.dtype != x_value.dtype:
        raise ValueError("x, y, and exposure must have the same dtype")
    if bool((~torch.isfinite(x_value)).any()) or bool(
        ((x_value < 0.0) | (x_value > 1.0)).any()
    ):
        raise ValueError("x must be finite and lie in [0,1]")
    if bool((~torch.isfinite(y_value)).any()) or bool(
        ((y_value < 0.0) | (y_value > 1.0)).any()
    ):
        raise ValueError("y must be finite and lie in [0,1]")
    if bool((~torch.isfinite(u_value)).any()) or bool((u_value <= 0.0).any()):
        raise ValueError("exposure must be finite and strictly positive")

    zx = 2.0 * x_value - 1.0
    zy = 2.0 * y_value - 1.0
    density = torch.ones_like(zx)
    cdf = y_value.clone()
    derivative = torch.zeros_like(zx)

    px_previous = torch.ones_like(zx)
    px_current = zx
    py_previous = torch.ones_like(zy)
    py_current = zy
    dpy_previous = torch.zeros_like(zy)
    dpy_current = torch.ones_like(zy)

    for n in range(1, int(modes)):
        py_next = (
            float(2 * n + 1) * zy * py_current - float(n) * py_previous
        ) / float(n + 1)
        dpy_next = (
            float(2 * n + 1) * (py_current + zy * dpy_current)
            - float(n) * dpy_previous
        ) / float(n + 1)
        decay = torch.exp(-float(n * (n + 1)) * u_value)
        coefficient = float(2 * n + 1) * decay
        density = density + coefficient * px_current * py_current
        derivative = derivative + coefficient * px_current * (2.0 * dpy_current)
        cdf = cdf + 0.5 * decay * px_current * (py_next - py_previous)

        px_next = (
            float(2 * n + 1) * zx * px_current - float(n) * px_previous
        ) / float(n + 1)
        px_previous, px_current = px_current, px_next
        py_previous, py_current = py_current, py_next
        dpy_previous, dpy_current = dpy_current, dpy_next

    cdf = torch.where(
        y_value == 0.0,
        torch.zeros_like(cdf),
        torch.where(y_value == 1.0, torch.ones_like(cdf), cdf),
    )
    return TorchAlpha1SpectralEvaluation(
        density=density,
        cdf=cdf,
        arrival_score=derivative / density,
        modes_used=int(modes),
    )


@dataclass(frozen=True)
class SpectralInverseCDFConfig:
    spectral: Alpha1SpectralConfig = field(
        default_factory=lambda: Alpha1SpectralConfig(
            absolute_tolerance=1e-13,
            relative_tolerance=1e-12,
            max_modes=4096,
        )
    )
    cdf_residual_tolerance: float = 1e-9
    y_tolerance: float = 1e-10
    max_iterations: int = 96

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.cdf_residual_tolerance)) or not (
            float(self.cdf_residual_tolerance) > 0.0
        ):
            raise ValueError("cdf_residual_tolerance must be positive")
        if not math.isfinite(float(self.y_tolerance)) or not (
            float(self.y_tolerance) > 0.0
        ):
            raise ValueError("y_tolerance must be positive")
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class SpectralInverseCDFDiagnostics:
    certified: bool
    sample_count: int
    maximum_iterations_used: int
    maximum_modes_used: int
    maximum_cdf_residual_bound: float
    maximum_final_bracket_width: float
    method: str = "certified-alpha1-spectral-inverse-cdf-v1"
    latent_label_available: bool = False


@dataclass(frozen=True)
class SpectralInverseCDFSamples:
    samples: np.ndarray
    uniforms: np.ndarray
    diagnostics: SpectralInverseCDFDiagnostics


def sample_alpha1_spectral_inverse_cdf(
    x: np.ndarray | float,
    exposure: np.ndarray | float,
    *,
    rng: np.random.Generator,
    config: SpectralInverseCDFConfig | None = None,
) -> SpectralInverseCDFSamples:
    """Draw a numerically certified alpha=1 transition sample.

    Certification bounds ``|F(sample)-U|`` by the reported residual.  This is
    suitable for transition-law validation.  It is not an ancestral-mixture
    sampler and therefore cannot produce the exact latent denoising label Z.
    """

    numerical = config or SpectralInverseCDFConfig()
    x_array, u_array = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(exposure, dtype=np.float64)
    )
    if not np.all(np.isfinite(x_array)) or np.any(x_array < 0.0) or np.any(
        x_array > 1.0
    ):
        raise ValueError("x must be finite and lie in [0,1]")
    if not np.all(np.isfinite(u_array)) or np.any(u_array <= 0.0):
        raise ValueError("exposure must be finite and strictly positive")
    uniforms = rng.random(x_array.shape)
    samples = np.empty_like(x_array)
    maximum_iterations = 0
    maximum_modes = 0
    maximum_residual = 0.0
    maximum_width = 0.0

    for index in np.ndindex(x_array.shape):
        target = float(uniforms[index])
        left = 0.0
        right = 1.0
        final_evaluation: Alpha1SpectralEvaluation | None = None
        midpoint = 0.5
        used_iterations = 0
        for iteration in range(1, int(numerical.max_iterations) + 1):
            midpoint = 0.5 * (left + right)
            final_evaluation = evaluate_alpha1_spectral(
                float(x_array[index]),
                midpoint,
                float(u_array[index]),
                config=numerical.spectral,
                strict=True,
            )
            estimate = float(final_evaluation.cdf)
            residual_bound = (
                abs(estimate - target)
                + final_evaluation.diagnostics.max_cdf_tail_bound
            )
            used_iterations = iteration
            if residual_bound <= float(numerical.cdf_residual_tolerance):
                break
            if estimate < target:
                left = midpoint
            else:
                right = midpoint
            if right - left <= float(numerical.y_tolerance):
                break
        if final_evaluation is None:
            raise RuntimeError("inverse-CDF loop did not evaluate the CDF")
        estimate = float(final_evaluation.cdf)
        residual_bound = (
            abs(estimate - target)
            + final_evaluation.diagnostics.max_cdf_tail_bound
        )
        width = right - left
        if residual_bound > float(numerical.cdf_residual_tolerance):
            raise SpectralConvergenceError(
                Alpha1SpectralDiagnostics(
                    **{
                        **final_evaluation.diagnostics.__dict__,
                        "converged": False,
                        "reason": (
                            "inverse-CDF residual exceeded its configured "
                            "certificate tolerance"
                        ),
                    }
                )
            )
        samples[index] = midpoint
        maximum_iterations = max(maximum_iterations, used_iterations)
        maximum_modes = max(
            maximum_modes, final_evaluation.diagnostics.modes_used
        )
        maximum_residual = max(maximum_residual, residual_bound)
        maximum_width = max(maximum_width, width)

    diagnostics = SpectralInverseCDFDiagnostics(
        certified=True,
        sample_count=int(samples.size),
        maximum_iterations_used=maximum_iterations,
        maximum_modes_used=maximum_modes,
        maximum_cdf_residual_bound=maximum_residual,
        maximum_final_bracket_width=maximum_width,
    )
    return SpectralInverseCDFSamples(
        samples=samples, uniforms=uniforms, diagnostics=diagnostics
    )


def jacobi_latent_label(
    ancestral_count: np.ndarray | int,
    selected_count: np.ndarray | int,
    later_head_fraction: np.ndarray | float,
) -> np.ndarray:
    """Return the stable DDPM-like component score ``Z = L - M Y``."""

    m, selected, y = np.broadcast_arrays(
        np.asarray(ancestral_count),
        np.asarray(selected_count),
        np.asarray(later_head_fraction, dtype=np.float64),
    )
    if not np.issubdtype(m.dtype, np.integer) or not np.issubdtype(
        selected.dtype, np.integer
    ):
        raise ValueError("ancestral_count and selected_count must be integers")
    if np.any(m < 0) or np.any(selected < 0) or np.any(selected > m):
        raise ValueError("counts must satisfy 0 <= L <= M")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError("later_head_fraction must be finite and lie in [0,1]")
    return selected.astype(np.float64) - m.astype(np.float64) * y


def jacobi_component_relative_score(
    ancestral_count: np.ndarray | int,
    selected_count: np.ndarray | int,
    later_head_fraction: np.ndarray | float,
) -> np.ndarray:
    """Return the component arrival score relative to ``Beta(alpha,alpha)``.

    The result is ``Z / (Y(1-Y))`` and is intentionally undefined at a
    boundary when ``Z`` is nonzero; no denominator floor is introduced.
    """

    y = np.asarray(later_head_fraction, dtype=np.float64)
    z = jacobi_latent_label(ancestral_count, selected_count, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return z / (y * (1.0 - y))


def linear_teacher_relative_density(
    y: np.ndarray | float,
    exposure: np.ndarray | float,
    *,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Positive-time density for ``v0(x)=1+c(2x-1)`` at alpha=1."""

    y_value, u_value = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64), np.asarray(exposure, dtype=np.float64)
    )
    c = float(amplitude)
    if not math.isfinite(c) or abs(c) >= 1.0:
        raise ValueError("amplitude must be finite and have magnitude below one")
    if not np.all(np.isfinite(y_value)) or np.any(y_value < 0.0) or np.any(
        y_value > 1.0
    ):
        raise ValueError("y must be finite and lie in [0,1]")
    if not np.all(np.isfinite(u_value)) or np.any(u_value < 0.0):
        raise ValueError("exposure must be finite and nonnegative")
    return 1.0 + c * np.exp(-2.0 * u_value) * (2.0 * y_value - 1.0)


def linear_teacher_arrival_score(
    y: np.ndarray | float,
    exposure: np.ndarray | float,
    *,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Exact arrival relative score of the bounded linear teacher."""

    y_value, u_value = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64), np.asarray(exposure, dtype=np.float64)
    )
    density = linear_teacher_relative_density(
        y_value, u_value, amplitude=amplitude
    )
    return 2.0 * float(amplitude) * np.exp(-2.0 * u_value) / density


def linear_teacher_denoising_mean(
    y: np.ndarray | float,
    exposure: np.ndarray | float,
    *,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Exact conditional mean of Z for the bounded linear teacher."""

    y_value = np.asarray(y, dtype=np.float64)
    return y_value * (1.0 - y_value) * linear_teacher_arrival_score(
        y_value, exposure, amplitude=amplitude
    )


def denoising_mean_to_mass_flux(
    denoising_mean: np.ndarray | float,
    *,
    alpha: float = 1.0,
    grid_spacing: float,
    schedule_value: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Map the head-oriented conditional mean of Z to physical Doob flux."""

    mean, schedule = np.broadcast_arrays(
        np.asarray(denoising_mean, dtype=np.float64),
        np.asarray(schedule_value, dtype=np.float64),
    )
    alpha_value = float(alpha)
    h = float(grid_spacing)
    if not math.isfinite(alpha_value) or alpha_value <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("grid_spacing must be finite and positive")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(schedule)):
        raise ValueError("mean and schedule must be finite")
    if np.any(schedule < 0.0):
        raise ValueError("schedule must be nonnegative")
    return (
        2.0
        * (2.0 * alpha_value + 1.0)
        * schedule
        * mean
        / (alpha_value * h * h)
    )
