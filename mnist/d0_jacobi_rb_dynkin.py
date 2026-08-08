"""Exact phasewise Dynkin observables for Jacobi RB Strang controls.

This module is deliberately additive.  It observes the immutable certified
Jacobi sampler used by :mod:`mnist.d0_jacobi_rb_strang_refinement` without
changing its transition inputs, outputs, state-update order, or hashes.

For the frozen Fourier, quadratic, and cubic observables the conditional
one-phase drift is analytic.  Accumulating those conditional drifts gives

    A_t = f(S_0) + sum_j (P_j f(S_j) - f(S_j)),

which has the same expectation as the raw endpoint observable while removing
the phase martingale innovations.  No coefficient is fitted and no future
state is used by the observer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import operator
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist import d0_jacobi_rb_cuda_controls as _controls
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    GRID_SIZE,
    GRID_SPACING,
    PATH_STATE_SIZE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    REFINEMENT_SHARD_STEPS,
    TAU_EFF,
    RefinementObservableSpec,
    RefinementShardResult,
    RefinementTransitionIDProvider,
    evaluate_refinement_observables,
    refinement_observable_spec,
    refinement_phase_exposure,
    run_refinement_shard,
)


DYNKIN_ESTIMATOR_VERSION = "jacobi-rb-exact-phase-dynkin-v1"
DYNKIN_EXPONENTIAL_VERSION = "binary64-ball-exp24-scale256-v1"
OBSERVABLE_COUNT = 10
FOURIER_OBSERVABLE_COUNT = 8

# Each elementary binary64 operation is widened by a deliberately loose
# 2^-52 relative allowance.  This is twice the usual round-to-nearest unit
# roundoff and also covers the final centre compression used below.
_ROUNDING_ALLOWANCE = float.fromhex("0x1.0p-52")
_MIN_SUBNORMAL = float.fromhex("0x0.0000000000001p-1022")
_EXP_REDUCTION_SQUARES = 8
_EXP_REDUCTION_FACTOR = 1 << _EXP_REDUCTION_SQUARES
_EXP_TAYLOR_DEGREE = 24


def _sha256_arrays(*values: np.ndarray) -> str:
    return _controls._digest_arrays(*values)


def _freeze_numpy(value: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_edge_tensor(
    value: Tensor,
    name: str,
    *,
    shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if not isinstance(value, Tensor) or value.dtype != torch.float64 or value.ndim != 2:
        raise TypeError(f"{name} must be a rank-two float64 torch tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    observed = (int(value.shape[0]), int(value.shape[1]))
    if shape is not None and observed != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if observed[1] != EDGES_PER_PHASE:
        raise ValueError(f"{name} must contain {EDGES_PER_PHASE} matching edges")
    if value.device.type == "cpu" and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return observed


def _matching_indices(
    matching_index: int, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    try:
        index = operator.index(matching_index)
    except TypeError as exc:
        raise TypeError("matching_index must be an integer") from exc
    if not 0 <= index < 4:
        raise ValueError("matching_index must lie in 0..3")
    tails, heads = _controls._matching_arrays()[index]
    return (
        torch.as_tensor(tails, dtype=torch.int64, device=device).contiguous(),
        torch.as_tensor(heads, dtype=torch.int64, device=device).contiguous(),
    )


def _coefficient_ball(coefficient: int, factorial: int) -> tuple[float, float]:
    """Return a binary64 centre and outward radius for coefficient/factorial."""

    from fractions import Fraction

    exact = Fraction(int(coefficient), int(factorial))
    centre = float(exact)
    difference = abs(Fraction.from_float(centre) - exact)
    radius = float(difference)
    if Fraction.from_float(radius) < difference:
        radius = math.nextafter(radius, math.inf)
    return centre, radius


_EXP_COEFFICIENTS: tuple[tuple[float, float], ...] = tuple(
    _coefficient_ball(-1 if degree % 2 else 1, math.factorial(degree))
    for degree in range(_EXP_TAYLOR_DEGREE + 1)
)
_EXP_TAYLOR_REMAINDER = math.nextafter(
    float(
        # The alternating exponential series has decreasing terms on
        # [0,1/4], so the first omitted term is an absolute bound.
        (1.0 / 4.0) ** (_EXP_TAYLOR_DEGREE + 1)
        / math.factorial(_EXP_TAYLOR_DEGREE + 1)
    ),
    math.inf,
)
_EXP_NEGATIVE_64_UPPER = math.nextafter(math.exp(-64.0), math.inf)


@dataclass(frozen=True)
class _BallTensor:
    center: Tensor
    radius: Tensor


def _rounding_radius(scale: Tensor) -> Tensor:
    value = (
        torch.abs(scale) * _ROUNDING_ALLOWANCE
        + torch.as_tensor(_MIN_SUBNORMAL, dtype=torch.float64, device=scale.device)
    )
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _upward(value: Tensor) -> Tensor:
    """Round a nonnegative binary64 bound toward positive infinity."""

    return torch.nextafter(value, torch.full_like(value, math.inf))


def _upward_add(left: Tensor, right: Tensor) -> Tensor:
    return _upward(left + right)


def _upward_mul(left: Tensor, right: Tensor) -> Tensor:
    return _upward(left * right)


def _ball_point(center: Tensor, radius: Tensor | None = None) -> _BallTensor:
    return _BallTensor(
        center=center,
        radius=(torch.zeros_like(center) if radius is None else radius),
    )


def _ball_add(left: _BallTensor, right: _BallTensor) -> _BallTensor:
    center = left.center + right.center
    scale = _upward_add(torch.abs(left.center), torch.abs(right.center))
    radius = _upward_add(left.radius, right.radius)
    radius = _upward_add(radius, _rounding_radius(scale))
    return _BallTensor(center=center, radius=radius)


def _ball_sub(left: _BallTensor, right: _BallTensor) -> _BallTensor:
    return _ball_add(
        left,
        _BallTensor(center=-right.center, radius=right.radius),
    )


def _ball_mul(left: _BallTensor, right: _BallTensor) -> _BallTensor:
    center = left.center * right.center
    radius = _upward_mul(torch.abs(left.center), right.radius)
    radius = _upward_add(
        radius, _upward_mul(torch.abs(right.center), left.radius)
    )
    radius = _upward_add(radius, _upward_mul(left.radius, right.radius))
    scale = _upward_mul(torch.abs(left.center), torch.abs(right.center))
    radius = _upward_add(radius, _rounding_radius(scale))
    return _BallTensor(center=center, radius=radius)


def _ball_scale(ball: _BallTensor, scalar: float | Tensor) -> _BallTensor:
    represented = torch.as_tensor(
        scalar, dtype=torch.float64, device=ball.center.device
    )
    return _ball_mul(ball, _ball_point(represented))


def _ball_div_positive(ball: _BallTensor, denominator: Tensor) -> _BallTensor:
    if denominator.device != ball.center.device or denominator.dtype != torch.float64:
        raise TypeError("ball denominator must be an aligned float64 tensor")
    if denominator.device.type == "cpu" and not bool((denominator > 0.0).all()):
        raise ValueError("ball denominator must be strictly positive")
    center = ball.center / denominator
    radius = _upward(ball.radius / denominator)
    scale = _upward(
        (torch.abs(ball.center) + ball.radius) / denominator
    )
    radius = _upward_add(radius, _rounding_radius(scale))
    return _BallTensor(center=center, radius=radius)


def _ball_sum(ball: _BallTensor, *, dim: int, keepdim: bool = False) -> _BallTensor:
    term_count = int(ball.center.shape[dim])
    gamma = (
        term_count * _ROUNDING_ALLOWANCE
    ) / (1.0 - term_count * _ROUNDING_ALLOWANCE)
    center = torch.sum(ball.center, dim=dim, keepdim=keepdim)
    absolute_sum_rounded = torch.sum(
        torch.abs(ball.center), dim=dim, keepdim=keepdim
    )
    radius_sum_rounded = torch.sum(ball.radius, dim=dim, keepdim=keepdim)
    denominator = torch.full_like(
        absolute_sum_rounded, 1.0 - gamma
    )
    # Both nonnegative reductions can round down.  Dividing their rounded
    # values by ``1-gamma_n`` is an outward upper bound; multiplying a
    # rounded-down sum by ``1+gamma_n`` is not sufficient.
    absolute_sum_upper = _upward(absolute_sum_rounded / denominator)
    input_radius_upper = _upward(radius_sum_rounded / denominator)
    reduction_radius = _upward_mul(
        absolute_sum_upper, torch.full_like(absolute_sum_upper, gamma)
    )
    radius = _upward_add(input_radius_upper, reduction_radius)
    return _BallTensor(center=center, radius=radius)


def _ball_mul_exact(ball: _BallTensor, scalar: Tensor) -> _BallTensor:
    return _ball_mul(ball, _ball_point(scalar))


def _ball_add_constant(
    ball: _BallTensor, center_value: float, radius_value: float
) -> _BallTensor:
    constant_center = torch.as_tensor(
        center_value, dtype=torch.float64, device=ball.center.device
    )
    constant_radius = torch.as_tensor(
        radius_value, dtype=torch.float64, device=ball.center.device
    )
    return _ball_add(
        ball,
        _BallTensor(center=constant_center, radius=constant_radius),
    )


def _ball_square(ball: _BallTensor) -> _BallTensor:
    return _ball_mul(ball, ball)


def _certified_negative_expm1(arguments: Tensor) -> _BallTensor:
    """Enclose ``expm1(-arguments)`` for nonnegative binary64 arguments.

    The evaluator uses a degree-24 alternating Taylor ball on ``x/256`` and
    eight certified squarings.  Arguments above 64 are enclosed structurally
    by ``0 < exp(-x) <= exp(-64)``.  The calculation uses only separate torch
    binary64 operations; it never authorizes a libdevice transcendental.
    """

    if (
        not isinstance(arguments, Tensor)
        or arguments.dtype != torch.float64
        or not arguments.is_contiguous()
    ):
        raise TypeError("arguments must be a contiguous float64 tensor")
    if arguments.device.type == "cpu":
        if not bool(torch.isfinite(arguments).all()) or not bool(
            (arguments >= 0.0).all()
        ):
            raise ValueError("exponential arguments must be finite and nonnegative")

    exact_zero = arguments == 0.0
    large = arguments > 64.0
    reduced = torch.where(
        large,
        torch.zeros_like(arguments),
        arguments / float(_EXP_REDUCTION_FACTOR),
    )
    highest_center, highest_radius = _EXP_COEFFICIENTS[-1]
    ball = _BallTensor(
        center=torch.full_like(arguments, highest_center),
        radius=torch.full_like(arguments, highest_radius),
    )
    for center_value, radius_value in reversed(_EXP_COEFFICIENTS[:-1]):
        ball = _ball_mul_exact(ball, reduced)
        ball = _ball_add_constant(ball, center_value, radius_value)
    ball = _BallTensor(
        center=ball.center,
        radius=ball.radius
        + torch.as_tensor(
            _EXP_TAYLOR_REMAINDER,
            dtype=torch.float64,
            device=arguments.device,
        ),
    )
    for _ in range(_EXP_REDUCTION_SQUARES):
        ball = _ball_square(ball)

    exp_center = torch.where(large, torch.zeros_like(arguments), ball.center)
    exp_radius = torch.where(
        large,
        torch.full_like(arguments, _EXP_NEGATIVE_64_UPPER),
        ball.radius,
    )
    expm1_center = exp_center - 1.0
    expm1_radius = exp_radius + _rounding_radius(
        torch.abs(exp_center) + 1.0
    )
    return _BallTensor(
        center=torch.where(exact_zero, torch.zeros_like(arguments), expm1_center),
        radius=torch.where(exact_zero, torch.zeros_like(arguments), expm1_radius),
    )


@dataclass(frozen=True)
class DynkinPhaseDriftBatch:
    """Ten exact conditional observable drifts and numerical enclosures."""

    center: Tensor
    error_radius: Tensor
    diagnostics: Mapping[str, Any]
    certificate_mask: Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.center, Tensor)
            or not isinstance(self.error_radius, Tensor)
            or self.center.dtype != torch.float64
            or self.error_radius.dtype != torch.float64
            or self.center.shape != self.error_radius.shape
            or self.center.ndim != 2
            or self.center.shape[1] != OBSERVABLE_COUNT
            or self.center.device != self.error_radius.device
        ):
            raise TypeError("Dynkin drift tensors must be aligned float64 [P,10]")
        if self.certificate_mask is not None and (
            not isinstance(self.certificate_mask, Tensor)
            or self.certificate_mask.dtype != torch.bool
            or self.certificate_mask.shape[0] != self.center.shape[0]
            or self.certificate_mask.device != self.center.device
        ):
            raise TypeError("Dynkin certificate_mask must be a path-aligned bool tensor")

    @property
    def lower(self) -> Tensor:
        return torch.nextafter(
            self.center - self.error_radius,
            torch.full_like(self.center, -math.inf),
        )

    @property
    def upper(self) -> Tensor:
        return torch.nextafter(
            self.center + self.error_radius,
            torch.full_like(self.center, math.inf),
        )


def compute_dynkin_phase_drift(
    pair_total: Tensor,
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    matching_index: int,
    spec: RefinementObservableSpec | None = None,
    standardized: bool = False,
    pair_total_error_radius: Tensor | None = None,
    cuda_profile: JacobiRBCudaProfile | None = None,
) -> DynkinPhaseDriftBatch:
    """Return the analytic one-phase drift for the frozen ten observables.

    ``pair_total``, ``head_fraction``, and ``exposure`` have shape ``[P,392]``
    in the parent scheduler's head-oriented convention.  The result is raw by
    default; standardization divides each drift and enclosure by the exact
    frozen Dirichlet standard deviation without recentering a drift.
    """

    shape = _validate_edge_tensor(pair_total, "pair_total")
    _validate_edge_tensor(head_fraction, "head_fraction", shape=shape)
    _validate_edge_tensor(exposure, "exposure", shape=shape)
    if any(
        value.device != pair_total.device
        for value in (head_fraction, exposure)
    ):
        raise ValueError("phase tensors must share one device")
    if pair_total.device.type == "cpu":
        if not bool((pair_total >= 0.0).all()):
            raise ValueError("pair_total must be nonnegative")
        if not bool(((head_fraction >= 0.0) & (head_fraction <= 1.0)).all()):
            raise ValueError("head_fraction must lie in [0,1]")
        if not bool((exposure >= 0.0).all()):
            raise ValueError("exposure must be nonnegative")

    selected = spec or refinement_observable_spec(GRID_SIZE)
    if int(selected.grid_size) != GRID_SIZE:
        raise ValueError("the Dynkin gate is frozen to the 28x28 observable basis")
    tails, heads = _matching_indices(matching_index, device=pair_total.device)
    weights = torch.as_tensor(
        np.array(selected.fourier_weights, copy=True),
        dtype=torch.float64,
        device=pair_total.device,
    )
    weight_delta_ball = _ball_sub(
        _ball_point(weights.index_select(1, heads)),
        _ball_point(weights.index_select(1, tails)),
    )

    if pair_total_error_radius is None:
        r_radius = torch.zeros_like(pair_total)
    else:
        _validate_edge_tensor(
            pair_total_error_radius, "pair_total_error_radius", shape=shape
        )
        if pair_total_error_radius.device != pair_total.device:
            raise ValueError("pair_total_error_radius must share the phase device")
        if pair_total.device.type == "cpu" and not bool(
            (pair_total_error_radius >= 0.0).all()
        ):
            raise ValueError("pair_total_error_radius must be nonnegative")
        r_radius = pair_total_error_radius

    r_ball = _ball_point(pair_total, r_radius)
    x_ball = _ball_point(head_fraction)
    z_ball = _ball_add_constant(_ball_scale(x_ball, 2.0), -1.0, 0.0)
    p2_ball = _ball_scale(
        _ball_add_constant(
            _ball_scale(_ball_square(z_ball), 3.0),
            -1.0,
            0.0,
        ),
        0.5,
    )
    cuda_decay_diagnostics: Mapping[str, Any] = {}
    decay_certificate_mask: Tensor | None = None
    if exposure.is_cuda:
        from mnist.d0_jacobi_rb_dynkin_cuda import (
            certified_dynkin_decay_batch_cuda,
        )

        selected_profile = cuda_profile or JacobiRBCudaProfile()
        flat_exposure = exposure.reshape(-1)
        cuda_chunks = []
        for offset in range(0, int(flat_exposure.numel()), 4096):
            cuda_chunks.append(
                certified_dynkin_decay_batch_cuda(
                    flat_exposure[offset : offset + 4096]
                    .reshape(1, -1)
                    .contiguous(),
                    compile_flags=tuple(selected_profile.compile_flags),
                    threads_per_block=int(selected_profile.threads_per_block),
                )
            )
        exp2_center = torch.cat(
            [value.expm1_2_center.reshape(-1) for value in cuda_chunks]
        ).reshape_as(exposure)
        exp2_radius = torch.cat(
            [value.expm1_2_radius.reshape(-1) for value in cuda_chunks]
        ).reshape_as(exposure)
        exp6_center = torch.cat(
            [value.expm1_6_center.reshape(-1) for value in cuda_chunks]
        ).reshape_as(exposure)
        exp6_radius = torch.cat(
            [value.expm1_6_radius.reshape(-1) for value in cuda_chunks]
        ).reshape_as(exposure)
        decay_certificate_mask = torch.cat(
            [value.valid_mask.reshape(-1) for value in cuda_chunks]
        ).reshape_as(exposure)
        source_hashes = {
            str(value.diagnostics["source_sha256"]) for value in cuda_chunks
        }
        binary_hashes = {
            str(value.diagnostics["binary_sha256"]) for value in cuda_chunks
        }
        cuda_decay_diagnostics = {
            **dict(cuda_chunks[0].diagnostics),
            "lane_count": int(exposure.numel()),
            "launch_count": len(cuda_chunks),
            "maximum_launch_lanes": min(4096, int(exposure.numel())),
            "single_source_hash": int(len(source_hashes) == 1),
            "single_binary_hash": int(len(binary_hashes) == 1),
        }
    else:
        decays = _certified_negative_expm1(
            torch.stack((2.0 * exposure, 6.0 * exposure)).contiguous()
        )
        exp2_center, exp6_center = decays.center.unbind(0)
        exp2_radius, exp6_radius = decays.radius.unbind(0)
    # The exponential backends enclose their represented argument.  Widen by
    # the multiplication rounding needed to interpret the requested exact
    # products 2*u and 6*u.  On the nonnegative support exp(-x)-1 is
    # one-Lipschitz, so the argument radius is also an output radius.
    exp2_argument_radius = _rounding_radius(torch.abs(2.0 * exposure))
    exp6_argument_radius = _rounding_radius(torch.abs(6.0 * exposure))
    exp2_ball = _BallTensor(
        exp2_center,
        _upward_add(exp2_radius, exp2_argument_radius),
    )
    exp6_ball = _BallTensor(
        exp6_center,
        _upward_add(exp6_radius, exp6_argument_radius),
    )

    rz_half_ball = _ball_scale(_ball_mul(r_ball, z_ball), 0.5)
    linear_edge_ball = _ball_mul(rz_half_ball, exp2_ball)
    linear_edge_broadcast = _BallTensor(
        linear_edge_ball.center[:, None, :],
        linear_edge_ball.radius[:, None, :],
    )
    weight_delta_broadcast = _BallTensor(
        weight_delta_ball.center[None, :, :],
        weight_delta_ball.radius[None, :, :],
    )
    linear_terms_ball = _ball_mul(
        linear_edge_broadcast, weight_delta_broadcast
    )

    r2_ball = _ball_square(r_ball)
    r3_ball = _ball_mul(r2_ball, r_ball)
    q_prefactor_ball = _ball_div_positive(
        _ball_mul(r2_ball, p2_ball),
        torch.full_like(pair_total, 3.0),
    )
    c_prefactor_ball = _ball_scale(
        _ball_mul(r3_ball, p2_ball),
        0.5,
    )
    q_edge_ball = _ball_mul(q_prefactor_ball, exp6_ball)
    c_edge_ball = _ball_mul(c_prefactor_ball, exp6_ball)
    active = (pair_total > 0.0) & (exposure > 0.0)
    linear_terms_ball = _BallTensor(
        torch.where(
            active[:, None, :],
            linear_terms_ball.center,
            torch.zeros_like(linear_terms_ball.center),
        ),
        torch.where(
            active[:, None, :],
            linear_terms_ball.radius,
            torch.zeros_like(linear_terms_ball.radius),
        ),
    )
    q_edge_ball = _BallTensor(
        torch.where(
            active, q_edge_ball.center, torch.zeros_like(q_edge_ball.center)
        ),
        torch.where(
            active, q_edge_ball.radius, torch.zeros_like(q_edge_ball.radius)
        ),
    )
    c_edge_ball = _BallTensor(
        torch.where(
            active, c_edge_ball.center, torch.zeros_like(c_edge_ball.center)
        ),
        torch.where(
            active, c_edge_ball.radius, torch.zeros_like(c_edge_ball.radius)
        ),
    )
    linear_ball = _ball_sum(linear_terms_ball, dim=2)
    q_ball = _ball_sum(q_edge_ball, dim=1, keepdim=True)
    c_ball = _ball_sum(c_edge_ball, dim=1, keepdim=True)
    result_ball = _BallTensor(
        center=torch.cat(
            (linear_ball.center, q_ball.center, c_ball.center), dim=1
        ).contiguous(),
        radius=torch.cat(
            (linear_ball.radius, q_ball.radius, c_ball.radius), dim=1
        ).contiguous(),
    )
    path_active = torch.any(active, dim=1, keepdim=True)
    result_ball = _BallTensor(
        torch.where(
            path_active, result_ball.center, torch.zeros_like(result_ball.center)
        ),
        torch.where(
            path_active, result_ball.radius, torch.zeros_like(result_ball.radius)
        ),
    )

    if standardized:
        scales = torch.as_tensor(
            selected.standard_deviations,
            dtype=torch.float64,
            device=pair_total.device,
        )
        result_ball = _ball_div_positive(result_ball, scales)
    center = result_ball.center.contiguous()
    radius = result_ball.radius.contiguous()

    diagnostics: dict[str, Any] = {
        "version": DYNKIN_ESTIMATOR_VERSION,
        "exponential_version": DYNKIN_EXPONENTIAL_VERSION,
        "matching_index": int(matching_index),
        "path_count": int(shape[0]),
        "edge_count": int(shape[0] * shape[1]),
        "observable_count": OBSERVABLE_COUNT,
        "standardized": int(bool(standardized)),
        "uses_future_state": 0,
        "fitted_coefficient_count": 0,
        "transcendental_library_call_count": 0,
        "certified_polynomial_degree": _EXP_TAYLOR_DEGREE,
        "certified_scaling_squares": _EXP_REDUCTION_SQUARES,
        "pair_total_reconstruction_enclosed": int(
            pair_total_error_radius is not None
        ),
        "full_arithmetic_ball_propagation": 1,
        "authorizing_exponential_enclosure": 1,
        "cuda_exponential_certificate": dict(cuda_decay_diagnostics),
    }
    return DynkinPhaseDriftBatch(
        center=center,
        error_radius=radius,
        diagnostics=diagnostics,
        certificate_mask=decay_certificate_mask,
    )


@dataclass(frozen=True)
class DynkinAccumulatorState:
    """Device-resident raw-observable state persisted between shards."""

    center: Tensor
    compensation: Tensor
    error_radius: Tensor

    def __post_init__(self) -> None:
        tensors = (self.center, self.compensation, self.error_radius)
        if any(not isinstance(value, Tensor) for value in tensors):
            raise TypeError("Dynkin accumulator fields must be torch tensors")
        if any(value.dtype != torch.float64 for value in tensors):
            raise TypeError("Dynkin accumulator fields must be float64")
        if (
            self.center.ndim != 2
            or self.center.shape[1] != OBSERVABLE_COUNT
            or self.compensation.shape != self.center.shape
            or self.error_radius.shape != self.center.shape
            or any(value.device != self.center.device for value in tensors)
            or any(not value.is_contiguous() for value in tensors)
        ):
            raise ValueError("Dynkin accumulator fields must align as contiguous [P,10]")

    def clone(self) -> "DynkinAccumulatorState":
        return DynkinAccumulatorState(
            center=self.center.detach().clone(),
            compensation=self.compensation.detach().clone(),
            error_radius=self.error_radius.detach().clone(),
        )


def _initial_observable_ball(
    states: Tensor,
    *,
    spec: RefinementObservableSpec,
) -> _BallTensor:
    """Enclose the represented initial observable evaluation.

    The public observable evaluator remains the source of the stored centre,
    preserving the parent scheduler's numerical convention.  A separate
    elementwise ball evaluation encloses the exact real expression on the
    represented state and weight inputs; the final radius also encloses the
    difference between that ball centre and the stored BLAS/reduction centre.
    """

    raw = evaluate_refinement_observables(
        states, spec=spec, standardized=False
    )
    assert isinstance(raw, Tensor)
    weights = torch.as_tensor(
        np.array(spec.fourier_weights, copy=True),
        dtype=torch.float64,
        device=states.device,
    )
    state_ball = _ball_point(states)
    state_for_linear = _BallTensor(
        state_ball.center[:, None, :],
        state_ball.radius[:, None, :],
    )
    weight_for_linear = _BallTensor(
        weights[None, :, :],
        torch.zeros_like(weights)[None, :, :],
    )
    linear_ball = _ball_sum(
        _ball_mul(state_for_linear, weight_for_linear),
        dim=2,
    )
    square_ball = _ball_square(state_ball)
    quadratic_ball = _ball_sum(square_ball, dim=1, keepdim=True)
    cubic_ball = _ball_sum(
        _ball_mul(square_ball, state_ball),
        dim=1,
        keepdim=True,
    )
    independent = _BallTensor(
        center=torch.cat(
            (linear_ball.center, quadratic_ball.center, cubic_ball.center),
            dim=1,
        ),
        radius=torch.cat(
            (linear_ball.radius, quadratic_ball.radius, cubic_ball.radius),
            dim=1,
        ),
    )
    difference = torch.abs(raw - independent.center)
    difference_upper = _upward(
        difference
        / torch.full_like(difference, 1.0 - _ROUNDING_ALLOWANCE)
    )
    radius = _upward_add(independent.radius, difference_upper)
    return _BallTensor(
        center=raw.contiguous(),
        radius=radius.contiguous(),
    )


class CompensatedDynkinAccumulator:
    """Deterministic Kahan accumulator with a separately additive enclosure."""

    def __init__(self, state: DynkinAccumulatorState) -> None:
        self._center = state.center.detach().clone()
        self._compensation = state.compensation.detach().clone()
        self._error_radius = state.error_radius.detach().clone()

    @classmethod
    def from_initial_observables(
        cls,
        raw_values: Tensor,
        *,
        initial_error_radius: Tensor | None = None,
    ) -> "CompensatedDynkinAccumulator":
        if (
            not isinstance(raw_values, Tensor)
            or raw_values.dtype != torch.float64
            or raw_values.ndim != 2
            or raw_values.shape[1] != OBSERVABLE_COUNT
            or not raw_values.is_contiguous()
        ):
            raise TypeError("raw_values must be contiguous float64 [P,10]")
        zeros = torch.zeros_like(raw_values)
        radius = zeros.clone() if initial_error_radius is None else initial_error_radius
        if (
            not isinstance(radius, Tensor)
            or radius.dtype != torch.float64
            or radius.shape != raw_values.shape
            or radius.device != raw_values.device
            or not radius.is_contiguous()
        ):
            raise TypeError(
                "initial_error_radius must be contiguous float64 aligned with raw_values"
            )
        if radius.device.type == "cpu" and not bool((radius >= 0.0).all()):
            raise ValueError("initial_error_radius must be nonnegative")
        return cls(
            DynkinAccumulatorState(
                center=raw_values,
                compensation=zeros,
                error_radius=radius,
            )
        )

    def add_(self, drift: DynkinPhaseDriftBatch) -> None:
        if (
            drift.center.shape != self._center.shape
            or drift.center.device != self._center.device
        ):
            raise ValueError("Dynkin drift does not match its accumulator")
        previous = self._center
        corrected = drift.center - self._compensation
        updated = previous + corrected
        self._compensation = (updated - previous) - corrected
        self._center = updated
        # Enclose Kahan's stored centre relative to the exact real addition
        # ``previous + drift.center``.  This accounts for every compensation
        # operation through the observed centre displacement, without assuming
        # the compensation itself is exact.
        naive = previous + drift.center
        displacement = torch.abs(updated - naive)
        displacement_upper = _upward(
            displacement
            / torch.full_like(displacement, 1.0 - _ROUNDING_ALLOWANCE)
        )
        addition_rounding = _rounding_radius(
            _upward_add(torch.abs(previous), torch.abs(drift.center))
        )
        radius = _upward_add(self._error_radius, drift.error_radius)
        radius = _upward_add(radius, displacement_upper)
        self._error_radius = _upward_add(radius, addition_rounding)

    def state(self) -> DynkinAccumulatorState:
        return DynkinAccumulatorState(
            center=self._center.contiguous(),
            compensation=self._compensation.contiguous(),
            error_radius=self._error_radius.contiguous(),
        )


@dataclass(frozen=True)
class DynkinObservableCheckpoint:
    completed_step: int
    time_fraction: float
    path_ids: tuple[int, ...]
    raw_values: np.ndarray = field(repr=False, compare=False)
    dynkin_values: np.ndarray = field(repr=False, compare=False)
    dynkin_error_radius: np.ndarray = field(repr=False, compare=False)
    raw_values_sha256: str
    dynkin_values_sha256: str
    dynkin_error_radius_sha256: str

    def __post_init__(self) -> None:
        expected = (len(self.path_ids), OBSERVABLE_COUNT)
        for name in ("raw_values", "dynkin_values", "dynkin_error_radius"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            object.__setattr__(self, name, _freeze_numpy(value))

    def to_record(self, *, include_values: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "completed_step": int(self.completed_step),
            "time_fraction": float(self.time_fraction),
            "path_ids": list(self.path_ids),
            "raw_values_sha256": self.raw_values_sha256,
            "dynkin_values_sha256": self.dynkin_values_sha256,
            "dynkin_error_radius_sha256": self.dynkin_error_radius_sha256,
        }
        if include_values:
            result.update(
                raw_values=self.raw_values.tolist(),
                dynkin_values=self.dynkin_values.tolist(),
                dynkin_error_radius=self.dynkin_error_radius.tolist(),
            )
        return result


@dataclass(frozen=True)
class DynkinShardResult:
    """An immutable refinement shard plus exact Dynkin observer evidence."""

    base_shard: RefinementShardResult
    accumulator_state: DynkinAccumulatorState
    committed_accumulator_center: np.ndarray = field(repr=False, compare=False)
    committed_accumulator_compensation: np.ndarray = field(
        repr=False, compare=False
    )
    committed_accumulator_error_radius: np.ndarray = field(
        repr=False, compare=False
    )
    observable_checkpoints: tuple[DynkinObservableCheckpoint, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = tuple(int(value) for value in self.accumulator_state.center.shape)
        for name in (
            "committed_accumulator_center",
            "committed_accumulator_compensation",
            "committed_accumulator_error_radius",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            object.__setattr__(self, name, _freeze_numpy(value))

    @property
    def final_states(self) -> Tensor:
        return self.base_shard.final_states

    @property
    def committed_final_states(self) -> np.ndarray:
        return self.base_shard.committed_final_states

    @property
    def path_records(self) -> Any:
        return self.base_shard.path_records

    @property
    def phase_state_records(self) -> Any:
        return self.base_shard.phase_state_records

    @property
    def batch_output_sha256(self) -> str:
        return self.base_shard.batch_output_sha256

    @property
    def batch_final_state_sha256(self) -> str:
        return self.base_shard.batch_final_state_sha256

    @property
    def batch_certificate_sha256(self) -> str:
        return self.base_shard.batch_certificate_sha256

    @property
    def checkpoint_states(self) -> tuple[DynkinObservableCheckpoint, ...]:
        return self.observable_checkpoints

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": DYNKIN_ESTIMATOR_VERSION + "-shard",
            "schema_version": 1,
            "base_shard": self.base_shard.to_record(),
            "observable_checkpoints": [
                value.to_record() for value in self.observable_checkpoints
            ],
            "accumulator": {
                "shape": list(self.accumulator_state.center.shape),
                "center_sha256": _sha256_arrays(
                    self.committed_accumulator_center
                ),
                "compensation_sha256": _sha256_arrays(
                    self.committed_accumulator_compensation
                ),
                "error_radius_sha256": _sha256_arrays(
                    self.committed_accumulator_error_radius
                ),
            },
            "diagnostics": dict(self.diagnostics),
        }


class _DynkinSamplerObserver:
    def __init__(
        self,
        *,
        sampler: Callable[..., Any],
        accumulator: CompensatedDynkinAccumulator,
        sample_steps: int,
        start_step: int,
        checkpoint_steps: Sequence[int],
        spec: RefinementObservableSpec,
        profile: JacobiRBCudaProfile,
    ) -> None:
        self.sampler = sampler
        self.accumulator = accumulator
        self.sample_steps = int(sample_steps)
        self.start_step = int(start_step)
        self.checkpoint_steps = frozenset(int(value) for value in checkpoint_steps)
        self.spec = spec
        self.profile = profile
        self.call_count = 0
        self.snapshots: dict[int, DynkinAccumulatorState] = {}
        self.maximum_phase_error = torch.zeros(
            (), dtype=torch.float64, device=accumulator.state().center.device
        )
        self.exponential_invalid_count = torch.zeros(
            (), dtype=torch.int64, device=accumulator.state().center.device
        )

    def __call__(self, x: Tensor, exposure: Tensor, **kwargs: Any) -> Any:
        phase = self.call_count % len(PHASE_MATCHINGS)
        local_step = self.call_count // len(PHASE_MATCHINGS)
        if local_step >= REFINEMENT_SHARD_STEPS:
            raise RuntimeError("Dynkin observer received too many phase calls")
        duration = PHASE_DURATIONS[phase]
        numerator = (
            3.0
            * (TAU_EFF / float(self.sample_steps))
            * float(duration)
            / (GRID_SPACING * GRID_SPACING)
        )
        exposure_matrix = exposure.reshape(-1, EDGES_PER_PHASE)
        fraction_matrix = x.reshape_as(exposure_matrix)
        positive = exposure_matrix > 0.0
        reconstructed_total = torch.where(
            positive,
            torch.as_tensor(
                numerator, dtype=torch.float64, device=exposure.device
            )
            / exposure_matrix,
            torch.zeros_like(exposure_matrix),
        ).contiguous()
        # fl(n/fl(n/r)) encloses the original represented r within a loose
        # four-rounding relative envelope.
        reconstructed_radius = (
            4.0 * _ROUNDING_ALLOWANCE * torch.abs(reconstructed_total)
            + _MIN_SUBNORMAL
        ).contiguous()
        drift = compute_dynkin_phase_drift(
            reconstructed_total,
            fraction_matrix.contiguous(),
            exposure_matrix.contiguous(),
            matching_index=PHASE_MATCHINGS[phase],
            spec=self.spec,
            standardized=False,
            pair_total_error_radius=reconstructed_radius,
            cuda_profile=self.profile,
        )
        self.accumulator.add_(drift)
        self.maximum_phase_error = torch.maximum(
            self.maximum_phase_error, torch.max(drift.error_radius)
        )
        if drift.certificate_mask is not None:
            self.exponential_invalid_count += torch.sum(
                ~drift.certificate_mask, dtype=torch.int64
            )
        completed_step = self.start_step + local_step + 1
        if phase == len(PHASE_MATCHINGS) - 1 and completed_step in self.checkpoint_steps:
            self.snapshots[completed_step] = self.accumulator.state().clone()
        self.call_count += 1
        # The observer is strictly pre-transition and forwards the exact
        # original call without changing a keyword or tensor.
        return self.sampler(x, exposure, **kwargs)


def _validated_accumulator(
    states: Tensor,
    *,
    start_step: int,
    accumulator_state: DynkinAccumulatorState | None,
    spec: RefinementObservableSpec,
) -> CompensatedDynkinAccumulator:
    if accumulator_state is None:
        if int(start_step) != 0:
            raise ValueError("resumed Dynkin shards require accumulator_state")
        initial = _initial_observable_ball(states, spec=spec)
        return CompensatedDynkinAccumulator.from_initial_observables(
            initial.center.contiguous(),
            initial_error_radius=initial.radius.contiguous(),
        )
    if int(start_step) == 0:
        raise ValueError("the initial Dynkin shard must infer f(S0)")
    if (
        accumulator_state.center.shape != (int(states.shape[0]), OBSERVABLE_COUNT)
        or accumulator_state.center.device != states.device
    ):
        raise ValueError("accumulator_state does not match the resumed state batch")
    return CompensatedDynkinAccumulator(accumulator_state)


def run_dynkin_refinement_shard(
    states: Tensor,
    *,
    path_ids: Sequence[int],
    sample_steps: int,
    start_step: int,
    root_seed: int,
    panel_namespace: str,
    profile: JacobiRBCudaProfile,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
    checkpoint_steps: Sequence[int] = (),
    transition_id_provider: RefinementTransitionIDProvider | None = None,
    rng_key_override: Any | None = None,
    capture_phase_state_trace: bool = False,
    accumulator_state: DynkinAccumulatorState | None = None,
) -> DynkinShardResult:
    """Run one unchanged refinement shard with a pre-transition Dynkin observer."""

    if sampler is sample_alpha1_rb_transition_batch_cuda and (
        not isinstance(states, Tensor) or not states.is_cuda
    ):
        raise ValueError("the production Dynkin scheduler requires CUDA states")
    spec = refinement_observable_spec(GRID_SIZE)
    accumulator = _validated_accumulator(
        states,
        start_step=int(start_step),
        accumulator_state=accumulator_state,
        spec=spec,
    )
    requested = tuple(sorted({operator.index(value) for value in checkpoint_steps}))
    observer = _DynkinSamplerObserver(
        sampler=sampler,
        accumulator=accumulator,
        sample_steps=int(sample_steps),
        start_step=int(start_step),
        checkpoint_steps=requested,
        spec=spec,
        profile=profile,
    )
    base = run_refinement_shard(
        states,
        path_ids=path_ids,
        sample_steps=sample_steps,
        start_step=start_step,
        root_seed=root_seed,
        panel_namespace=panel_namespace,
        profile=profile,
        sampler=observer,
        checkpoint_steps=requested,
        transition_id_provider=transition_id_provider,
        rng_key_override=rng_key_override,
        capture_phase_state_trace=capture_phase_state_trace,
    )
    expected_calls = REFINEMENT_SHARD_STEPS * len(PHASE_MATCHINGS)
    if observer.call_count != expected_calls:
        raise _controls.RigorousCudaControlError(
            "Dynkin observer did not see every refinement phase"
        )
    canonical_paths = tuple(sorted(int(value) for value in path_ids))
    path_order = [list(path_ids).index(value) for value in canonical_paths]
    raw_by_step = {
        value.completed_step: value for value in base.observable_checkpoints
    }
    scales = torch.as_tensor(
        spec.standard_deviations,
        dtype=torch.float64,
        device=states.device,
    )
    means = torch.as_tensor(
        spec.means, dtype=torch.float64, device=states.device
    )
    checkpoint_device: list[Tensor] = []
    checkpoint_steps_present: list[int] = []
    for step in requested:
        if step not in observer.snapshots or step not in raw_by_step:
            continue
        snapshot = observer.snapshots[step]
        standardized_snapshot = _ball_div_positive(
            _ball_sub(
                _BallTensor(snapshot.center, snapshot.error_radius),
                _ball_point(means),
            ),
            scales,
        )
        checkpoint_device.extend(
            (
                standardized_snapshot.center.index_select(
                    0,
                    torch.as_tensor(
                        path_order, dtype=torch.int64, device=states.device
                    ),
                ),
                standardized_snapshot.radius.index_select(
                    0,
                    torch.as_tensor(
                        path_order, dtype=torch.int64, device=states.device
                    ),
                ),
            )
        )
        checkpoint_steps_present.append(step)
    final_accumulator = observer.accumulator.state()
    final_standardized = _ball_div_positive(
        _ball_sub(
            _BallTensor(
                final_accumulator.center,
                final_accumulator.error_radius,
            ),
            _ball_point(means),
        ),
        scales,
    )
    summary_device = [
        observer.exponential_invalid_count.to(
            dtype=torch.float64, device=states.device
        ).reshape(1),
        torch.max(final_accumulator.error_radius).reshape(1),
        torch.max(final_standardized.radius).reshape(1),
        final_accumulator.center.reshape(-1),
        final_accumulator.compensation.reshape(-1),
        final_accumulator.error_radius.reshape(-1),
        *(value.reshape(-1) for value in checkpoint_device),
    ]
    # The additive observer has exactly one host synchronization, at the
    # immutable base shard's existing commit boundary.
    packed = torch.cat(summary_device).detach().cpu().numpy()
    exponential_invalid_count = int(packed[0])
    if exponential_invalid_count:
        raise _controls.RigorousCudaControlError(
            "Dynkin exponential certificate failed for "
            f"{exponential_invalid_count} phase lanes"
        )
    max_error = float(packed[1])
    max_standardized_error = float(packed[2])
    offset = 3
    accumulator_count = int(states.shape[0]) * OBSERVABLE_COUNT
    committed_accumulator_center = packed[
        offset : offset + accumulator_count
    ].reshape(int(states.shape[0]), OBSERVABLE_COUNT)
    offset += accumulator_count
    committed_accumulator_compensation = packed[
        offset : offset + accumulator_count
    ].reshape(int(states.shape[0]), OBSERVABLE_COUNT)
    offset += accumulator_count
    committed_accumulator_error_radius = packed[
        offset : offset + accumulator_count
    ].reshape(int(states.shape[0]), OBSERVABLE_COUNT)
    offset += accumulator_count
    checkpoints: list[DynkinObservableCheckpoint] = []
    for step in checkpoint_steps_present:
        count = len(canonical_paths) * OBSERVABLE_COUNT
        dynkin = packed[offset : offset + count].reshape(
            len(canonical_paths), OBSERVABLE_COUNT
        )
        offset += count
        radius = packed[offset : offset + count].reshape(
            len(canonical_paths), OBSERVABLE_COUNT
        )
        offset += count
        raw = raw_by_step[step].values
        checkpoints.append(
            DynkinObservableCheckpoint(
                completed_step=step,
                time_fraction=step / float(sample_steps),
                path_ids=canonical_paths,
                raw_values=raw,
                dynkin_values=dynkin,
                dynkin_error_radius=radius,
                raw_values_sha256=_sha256_arrays(raw),
                dynkin_values_sha256=_sha256_arrays(dynkin),
                dynkin_error_radius_sha256=_sha256_arrays(radius),
            )
        )
    if offset != packed.size:
        raise AssertionError("Dynkin checkpoint summary was not fully decoded")
    diagnostics = {
        **dict(base.diagnostics),
        "dynkin_version": DYNKIN_ESTIMATOR_VERSION,
        "dynkin_exponential_version": DYNKIN_EXPONENTIAL_VERSION,
        "dynkin_phase_observer_call_count": observer.call_count,
        "dynkin_exponential_invalid_count": exponential_invalid_count,
        "dynkin_checkpoint_steps": checkpoint_steps_present,
        "dynkin_maximum_raw_error_radius": max_error,
        "dynkin_maximum_cumulative_standardized_error_radius": (
            max_standardized_error
        ),
        "dynkin_checkpoint_maximum_standardized_error_radius": {
            str(value.completed_step): float(
                np.max(value.dynkin_error_radius, initial=0.0)
            )
            for value in checkpoints
        },
        "dynkin_observer_uses_future_state": 0,
        "dynkin_fitted_coefficient_count": 0,
        "dynkin_transition_hash_preserved": 1,
        "dynkin_state_hash_preserved": 1,
        "dynkin_accumulator_device_resident_until_commit": 1,
        # The immutable base runner commits its transition/state summary once;
        # this additive observer commits only its ten-value accumulator.
        "dynkin_summary_device_to_host_transfer_count": 1,
    }
    return DynkinShardResult(
        base_shard=base,
        accumulator_state=final_accumulator,
        committed_accumulator_center=committed_accumulator_center,
        committed_accumulator_compensation=committed_accumulator_compensation,
        committed_accumulator_error_radius=committed_accumulator_error_radius,
        observable_checkpoints=tuple(checkpoints),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class DynkinTowerPhaseResult:
    final_states: Tensor
    raw_before_values: Tensor
    raw_after_values: Tensor
    drift_center: Tensor
    drift_error_radius: Tensor
    standardized_residual: Tensor
    transition_result: Any = field(repr=False, compare=False)
    transition_output_sha256: str
    final_state_sha256: str
    diagnostics: Mapping[str, Any]

    @property
    def before_values(self) -> Tensor:
        return self.raw_before_values

    @property
    def after_values(self) -> Tensor:
        return self.raw_after_values

    @property
    def residual(self) -> Tensor:
        return self.standardized_residual

    @property
    def error_radius(self) -> Tensor:
        return self.drift_error_radius

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": DYNKIN_ESTIMATOR_VERSION + "-tower-phase",
            "schema_version": 1,
            "transition_output_sha256": self.transition_output_sha256,
            "final_state_sha256": self.final_state_sha256,
            "raw_before_values_sha256": _sha256_arrays(
                self.raw_before_values.detach().cpu().numpy()
            ),
            "raw_after_values_sha256": _sha256_arrays(
                self.raw_after_values.detach().cpu().numpy()
            ),
            "drift_center_sha256": _sha256_arrays(
                self.drift_center.detach().cpu().numpy()
            ),
            "drift_error_radius_sha256": _sha256_arrays(
                self.drift_error_radius.detach().cpu().numpy()
            ),
            "standardized_residual_sha256": _sha256_arrays(
                self.standardized_residual.detach().cpu().numpy()
            ),
            "diagnostics": dict(self.diagnostics),
        }


def run_dynkin_tower_phase(
    states: Tensor,
    *,
    matching_index: int,
    duration_fraction: float,
    sample_steps: int,
    rng_key: Any,
    transition_ids: Tensor,
    profile: JacobiRBCudaProfile,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
    standardized: bool = True,
) -> DynkinTowerPhaseResult:
    """Apply one certified matching phase and return actual-minus-drift residuals."""

    started = time.perf_counter()
    if (
        not isinstance(states, Tensor)
        or states.dtype != torch.float64
        or states.ndim != 2
        or states.shape[1] != PATH_STATE_SIZE
        or not states.is_contiguous()
    ):
        raise ValueError("states must be contiguous float64 [P,784]")
    if sampler is sample_alpha1_rb_transition_batch_cuda and not states.is_cuda:
        raise ValueError("the production tower control requires CUDA states")
    tails, heads = _matching_indices(matching_index, device=states.device)
    tail_mass = states.index_select(1, tails)
    head_mass = states.index_select(1, heads)
    pair_total = (tail_mass + head_mass).contiguous()
    positive = pair_total > 0.0
    fraction = torch.where(
        positive,
        head_mass / torch.where(positive, pair_total, torch.ones_like(pair_total)),
        torch.zeros_like(pair_total),
    ).contiguous()
    exposure = refinement_phase_exposure(
        pair_total,
        sample_steps=int(sample_steps),
        duration_fraction=float(duration_fraction),
    )
    drift = compute_dynkin_phase_drift(
        pair_total,
        fraction,
        exposure,
        matching_index=matching_index,
        standardized=False,
        cuda_profile=profile,
    )
    if drift.certificate_mask is not None:
        invalid_decay_count = int(
            torch.sum(~drift.certificate_mask, dtype=torch.int64)
            .detach()
            .cpu()
            .item()
        )
        if invalid_decay_count:
            raise _controls.RigorousCudaControlError(
                "tower phase contained uncertified Dynkin exponential lanes"
            )
    else:
        invalid_decay_count = 0
    flat_ids = transition_ids.reshape(-1)
    if (
        transition_ids.dtype != torch.uint64
        or transition_ids.device != states.device
        or transition_ids.numel() != fraction.numel()
        or not transition_ids.is_contiguous()
    ):
        raise ValueError("transition_ids must be contiguous device uint64")
    flat_fraction = fraction.reshape(-1)
    flat_exposure = exposure.reshape(-1)
    transition_results: list[Any] = []
    later_chunks: list[Tensor] = []
    target_chunks: list[Tensor] = []
    code_chunks: list[Tensor] = []
    for offset in range(0, int(flat_fraction.numel()), 4096):
        result = _controls._call_sampler(
            flat_fraction[offset : offset + 4096].contiguous(),
            flat_exposure[offset : offset + 4096].contiguous(),
            profile=profile,
            rng_key=rng_key,
            transition_offset=0,
            transition_ids=flat_ids[offset : offset + 4096].contiguous(),
            sampler=sampler,
        )
        transition_results.append(result)
        later_chunks.append(
            _controls._field(
                result, "later_head_fraction", "later", "y"
            ).reshape(-1).to(dtype=torch.float64)
        )
        target_chunks.append(
            _controls._field(
                result, "denoising_target", "target", "z"
            ).reshape(-1).to(dtype=torch.float64)
        )
        code_chunks.append(
            _controls._field(
                result, "certificate_codes", "certificate_code"
            ).reshape(-1).to(dtype=torch.uint8)
        )
    later = torch.cat(later_chunks).reshape_as(fraction)
    target = torch.cat(target_chunks).reshape_as(fraction)
    codes = torch.cat(code_chunks).reshape_as(fraction)
    final_states = states.detach().clone()
    final_states[:, tails] = pair_total * (1.0 - later)
    final_states[:, heads] = pair_total * later
    spec = refinement_observable_spec(GRID_SIZE)
    raw_before = evaluate_refinement_observables(
        states, spec=spec, standardized=False
    )
    raw_after = evaluate_refinement_observables(
        final_states, spec=spec, standardized=False
    )
    assert isinstance(raw_before, Tensor) and isinstance(raw_after, Tensor)
    scales = torch.as_tensor(
        spec.standard_deviations, dtype=torch.float64, device=states.device
    )
    standardized_residual = (
        raw_after - raw_before - drift.center
    ) / scales
    pair_error = torch.max(
        torch.abs(
            final_states.index_select(1, tails)
            + final_states.index_select(1, heads)
            - pair_total
        )
    )
    global_error = torch.max(
        torch.abs(torch.sum(final_states, dim=1) - torch.sum(states, dim=1))
    )
    packed_host = (
        torch.cat(
            (
                later.reshape(-1),
                target.reshape(-1),
                codes.reshape(-1).to(torch.float64),
                final_states.reshape(-1),
                pair_error.reshape(1),
                global_error.reshape(1),
            )
        )
        .detach()
        .cpu()
        .numpy()
    )
    # Decode through known sizes only to keep the hash contract independent of
    # a caller's tensor views.
    edge_count = int(later.numel())
    state_count = int(final_states.numel())
    later_np = packed_host[:edge_count]
    target_np = packed_host[edge_count : 2 * edge_count]
    codes_np = packed_host[2 * edge_count : 3 * edge_count].astype(
        np.uint8, copy=False
    )
    final_np = packed_host[
        3 * edge_count : 3 * edge_count + state_count
    ].reshape(tuple(final_states.shape))
    pair_mass_error = float(packed_host[3 * edge_count + state_count])
    global_mass_error = float(packed_host[3 * edge_count + state_count + 1])
    transition_hash = _sha256_arrays(later_np, target_np, codes_np)
    final_hash = _sha256_arrays(final_np)
    certified_count = int(np.count_nonzero((codes_np & 0xF) == 0xF))

    def diagnostic_sum(name: str, default: int | float = 0) -> int | float:
        values = [
            _controls._diagnostic_scalar(value, name, default)
            for value in transition_results
        ]
        if isinstance(default, int):
            return sum(int(value) for value in values)
        return sum(float(value) for value in values)

    fallback_elapsed = float(
        diagnostic_sum("arb_fallback_elapsed_seconds", 0.0)
    )
    wall_elapsed = time.perf_counter() - started
    diagnostics = {
        "version": DYNKIN_ESTIMATOR_VERSION,
        "matching_index": int(matching_index),
        "duration_fraction": float(duration_fraction),
        "sample_steps": int(sample_steps),
        "transition_count": edge_count,
        "backend_call_count": len(transition_results),
        "maximum_backend_call_size": min(4096, edge_count),
        "certified_count": certified_count,
        "uncertified_count": edge_count - certified_count,
        "fallback_count": int(diagnostic_sum("fallback_count", 0)),
        "fallback_elapsed_seconds": fallback_elapsed,
        "elapsed_seconds": wall_elapsed,
        "wall_elapsed_seconds": wall_elapsed,
        "maximum_pair_total_error": pair_mass_error,
        "maximum_global_simplex_error": global_mass_error,
        "dynkin_exponential_invalid_count": invalid_decay_count,
        "uses_future_state": 0,
        "fitted_coefficient_count": 0,
        **{
            name: int(diagnostic_sum(name, 0))
            for name in (
                "resource_cap_count",
                "invalid_density_count",
                "approximation_count",
                "correction_count",
                "floor_count",
                "limiter_count",
                "renormalization_count",
                "nonfinite_count",
            )
        },
        # The matching update is exact and never invokes a projection.
        "projection_count": 0,
    }
    return DynkinTowerPhaseResult(
        final_states=final_states,
        raw_before_values=raw_before,
        raw_after_values=raw_after,
        drift_center=drift.center,
        drift_error_radius=drift.error_radius,
        standardized_residual=standardized_residual,
        transition_result=tuple(transition_results),
        transition_output_sha256=transition_hash,
        final_state_sha256=final_hash,
        diagnostics=diagnostics,
    )


__all__ = [
    "DYNKIN_ESTIMATOR_VERSION",
    "DYNKIN_EXPONENTIAL_VERSION",
    "OBSERVABLE_COUNT",
    "FOURIER_OBSERVABLE_COUNT",
    "DynkinPhaseDriftBatch",
    "DynkinAccumulatorState",
    "CompensatedDynkinAccumulator",
    "DynkinObservableCheckpoint",
    "DynkinShardResult",
    "DynkinTowerPhaseResult",
    "compute_dynkin_phase_drift",
    "run_dynkin_refinement_shard",
    "run_dynkin_tower_phase",
]
