"""Certified phase-local observations for the Jacobi Dynkin controls.

The original tower control formed an observed phase increment by evaluating a
global observable before and after a matching update.  That is mathematically
correct, but two independent reductions leave roundoff residues in Fourier
modes that a matching preserves exactly.

This additive observer evaluates the same increment edge by edge.  Endpoint
weights that are bit-identical produce structural zeroes, with both centre and
radius equal to binary64 ``+0.0``.  No tolerance snapping or change to the
Jacobi transition is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_dynkin import (
    DynkinPhaseDriftBatch,
    _BallTensor,
    _ball_add,
    _ball_div_positive,
    _ball_mul,
    _ball_point,
    _ball_scale,
    _ball_sub,
    _matching_indices,
    _rounding_radius,
    _upward,
    _upward_add,
    _validate_edge_tensor,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    GRID_SIZE,
    PATH_STATE_SIZE,
    RefinementObservableSpec,
    evaluate_refinement_observables,
    refinement_observable_spec,
)


PHASE_OBSERVER_VERSION = "jacobi-rb-dynkin-phase-local-observer-v1"
OBSERVABLE_COUNT = 10
FOURIER_OBSERVABLE_COUNT = 8


def _tensor_sha256(value: Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _fixed_pairwise_sum(ball: _BallTensor) -> _BallTensor:
    """Reduce the final dimension with one fixed adjacent-pair tree."""

    center = ball.center
    radius = ball.radius
    while center.shape[-1] > 1:
        pair_count = int(center.shape[-1]) // 2
        paired = _ball_add(
            _BallTensor(
                center[..., : 2 * pair_count : 2],
                radius[..., : 2 * pair_count : 2],
            ),
            _BallTensor(
                center[..., 1 : 2 * pair_count : 2],
                radius[..., 1 : 2 * pair_count : 2],
            ),
        )
        if center.shape[-1] % 2:
            center = torch.cat((paired.center, center[..., -1:]), dim=-1)
            radius = torch.cat((paired.radius, radius[..., -1:]), dim=-1)
        else:
            center = paired.center
            radius = paired.radius
    return _BallTensor(center=center.squeeze(-1), radius=radius.squeeze(-1))


def _represented_difference_ball(left: Tensor, right: Tensor) -> _BallTensor:
    """Enclose the exact difference of represented binary64 operands."""

    center = left - right
    exact_equal = left.view(torch.int64) == right.view(torch.int64)
    radius = _rounding_radius(torch.abs(left) + torch.abs(right))
    return _BallTensor(
        center=torch.where(exact_equal, torch.zeros_like(center), center),
        radius=torch.where(exact_equal, torch.zeros_like(radius), radius),
    )


def _structural_zero_mask(
    spec: RefinementObservableSpec, matching_index: int
) -> np.ndarray:
    """Derive exact matching invariants from represented endpoint weights."""

    tails, heads = _matching_indices(
        matching_index, device=torch.device("cpu")
    )
    weights = np.ascontiguousarray(spec.fourier_weights, dtype=np.float64)
    represented = weights.view(np.uint64)
    tails_np = tails.numpy()
    heads_np = heads.numpy()
    mask = np.zeros(OBSERVABLE_COUNT, dtype=np.bool_)
    mask[:FOURIER_OBSERVABLE_COUNT] = np.all(
        represented[:, tails_np] == represented[:, heads_np], axis=1
    )
    return mask


def _validate_phase_inputs(
    pair_total: Tensor,
    earlier: Tensor,
    later: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> tuple[int, int, Tensor]:
    shape = _validate_edge_tensor(pair_total, "pair_total")
    for value, name in (
        (earlier, "earlier_head_fraction"),
        (later, "later_head_fraction"),
        (lower, "quantile_lower"),
        (upper, "quantile_upper"),
    ):
        _validate_edge_tensor(value, name, shape=shape)
        if value.device != pair_total.device:
            raise ValueError("phase observer tensors must share one device")
    valid = (
        torch.isfinite(pair_total)
        & torch.isfinite(earlier)
        & torch.isfinite(later)
        & torch.isfinite(lower)
        & torch.isfinite(upper)
        & (pair_total >= 0.0)
        & (earlier >= 0.0)
        & (earlier <= 1.0)
        & (lower >= 0.0)
        & (lower <= later)
        & (later <= upper)
        & (upper <= 1.0)
    )
    if not bool(torch.all(valid).detach().cpu().item()):
        raise ValueError(
            "phase observer inputs or quantile enclosures are invalid"
        )
    return shape[0], shape[1], valid


@dataclass(frozen=True)
class DynkinPhaseIncrementBatch:
    """Certified observed increments for the frozen ten observables."""

    center: Tensor
    error_radius: Tensor
    structural_zero_mask: Tensor
    quantile_enclosure_valid: Tensor
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.center, Tensor)
            or not isinstance(self.error_radius, Tensor)
            or self.center.dtype != torch.float64
            or self.error_radius.dtype != torch.float64
            or self.center.ndim != 2
            or self.center.shape[1] != OBSERVABLE_COUNT
            or self.center.shape != self.error_radius.shape
            or self.center.device != self.error_radius.device
            or not self.center.is_contiguous()
            or not self.error_radius.is_contiguous()
        ):
            raise TypeError(
                "phase increment tensors must be contiguous float64 [P,10]"
            )
        if (
            not isinstance(self.structural_zero_mask, Tensor)
            or self.structural_zero_mask.dtype != torch.bool
            or self.structural_zero_mask.shape != (OBSERVABLE_COUNT,)
            or self.structural_zero_mask.device != self.center.device
        ):
            raise TypeError("structural_zero_mask must be device bool [10]")
        if (
            not isinstance(self.quantile_enclosure_valid, Tensor)
            or self.quantile_enclosure_valid.dtype != torch.bool
            or self.quantile_enclosure_valid.ndim != 2
            or self.quantile_enclosure_valid.shape[0] != self.center.shape[0]
            or self.quantile_enclosure_valid.device != self.center.device
        ):
            raise TypeError(
                "quantile_enclosure_valid must be a path-aligned bool tensor"
            )

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

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PHASE_OBSERVER_VERSION + "-increment",
            "schema_version": 1,
            "center_sha256": _tensor_sha256(self.center),
            "error_radius_sha256": _tensor_sha256(self.error_radius),
            "structural_zero_mask": [
                int(value)
                for value in self.structural_zero_mask.detach().cpu().tolist()
            ],
            "quantile_enclosure_valid_count": int(
                torch.count_nonzero(self.quantile_enclosure_valid)
                .detach()
                .cpu()
                .item()
            ),
            "maximum_error_radius": float(
                torch.max(self.error_radius).detach().cpu().item()
            ),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class DynkinPhaseResidualBatch:
    """Observed-minus-conditional-drift residual and complete enclosure."""

    center: Tensor
    error_radius: Tensor
    observed_center: Tensor
    observed_error_radius: Tensor
    drift_center: Tensor
    drift_error_radius: Tensor
    structural_zero_mask: Tensor
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        tensors = (
            self.center,
            self.error_radius,
            self.observed_center,
            self.observed_error_radius,
            self.drift_center,
            self.drift_error_radius,
        )
        if any(
            not isinstance(value, Tensor)
            or value.dtype != torch.float64
            or value.shape != self.center.shape
            or value.device != self.center.device
            for value in tensors
        ):
            raise TypeError("phase residual tensors must be aligned float64")
        if self.center.ndim != 2 or self.center.shape[1] != OBSERVABLE_COUNT:
            raise ValueError("phase residual tensors must have shape [P,10]")
        if (
            self.structural_zero_mask.dtype != torch.bool
            or self.structural_zero_mask.shape != (OBSERVABLE_COUNT,)
            or self.structural_zero_mask.device != self.center.device
        ):
            raise TypeError("phase residual structural mask must be device bool [10]")

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

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PHASE_OBSERVER_VERSION + "-residual",
            "schema_version": 1,
            "center_sha256": _tensor_sha256(self.center),
            "error_radius_sha256": _tensor_sha256(self.error_radius),
            "observed_center_sha256": _tensor_sha256(self.observed_center),
            "drift_center_sha256": _tensor_sha256(self.drift_center),
            "maximum_error_radius": float(
                torch.max(self.error_radius).detach().cpu().item()
            ),
            "structural_zero_mask": [
                int(value)
                for value in self.structural_zero_mask.detach().cpu().tolist()
            ],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class AdvisoryGlobalPhaseIncrement:
    """The former global-subtraction observer, retained as a diagnostic."""

    center: Tensor
    error_radius: Tensor
    diagnostics: Mapping[str, Any]

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


def compute_dynkin_phase_observed_increment(
    pair_total: Tensor,
    earlier_head_fraction: Tensor,
    later_head_fraction: Tensor,
    *,
    matching_index: int,
    quantile_lower: Tensor,
    quantile_upper: Tensor,
    profile: JacobiRBCudaProfile | None = None,
    spec: RefinementObservableSpec | None = None,
    duration_fraction: float = 1.0,
) -> DynkinPhaseIncrementBatch:
    """Compute the exact phase-local realized observable increment.

    Quantile bounds are propagated into an outward enclosure of every
    increment.  A zero duration, zero pair mass, or exactly deterministic
    no-motion transition contributes bitwise ``+0.0``.
    """

    path_count, edge_count, valid = _validate_phase_inputs(
        pair_total,
        earlier_head_fraction,
        later_head_fraction,
        quantile_lower,
        quantile_upper,
    )
    duration = float(duration_fraction)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_fraction must be finite and nonnegative")
    if profile is not None and not isinstance(profile, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")
    selected = spec or refinement_observable_spec(GRID_SIZE)
    if int(selected.grid_size) != GRID_SIZE:
        raise ValueError("phase observer is frozen to the 28x28 basis")

    tails, heads = _matching_indices(
        matching_index, device=pair_total.device
    )
    structural_np = _structural_zero_mask(selected, matching_index)
    structural = torch.as_tensor(
        structural_np, dtype=torch.bool, device=pair_total.device
    )

    exact_quantile = (
        (quantile_lower == later_head_fraction)
        & (later_head_fraction == quantile_upper)
    )
    quantile_radius = _upward(
        torch.maximum(
            later_head_fraction - quantile_lower,
            quantile_upper - later_head_fraction,
        )
    )
    quantile_radius = torch.where(
        exact_quantile, torch.zeros_like(quantile_radius), quantile_radius
    )
    y_ball = _BallTensor(later_head_fraction, quantile_radius)
    x_ball = _ball_point(earlier_head_fraction)
    d_ball = _ball_sub(y_ball, x_ball)
    exact_no_motion = (
        exact_quantile
        & (later_head_fraction.view(torch.int64)
           == earlier_head_fraction.view(torch.int64))
    )
    d_ball = _BallTensor(
        torch.where(
            exact_no_motion, torch.zeros_like(d_ball.center), d_ball.center
        ),
        torch.where(
            exact_no_motion, torch.zeros_like(d_ball.radius), d_ball.radius
        ),
    )
    c_ball = _ball_add(
        _ball_add(x_ball, y_ball),
        _ball_point(torch.full_like(pair_total, -1.0)),
    )
    r_ball = _ball_point(pair_total)

    weights = torch.as_tensor(
        np.array(selected.fourier_weights, copy=True),
        dtype=torch.float64,
        device=pair_total.device,
    )
    weight_delta = _represented_difference_ball(
        weights.index_select(1, heads),
        weights.index_select(1, tails),
    )
    invariant_fourier = structural[:FOURIER_OBSERVABLE_COUNT, None]
    weight_delta = _BallTensor(
        torch.where(
            invariant_fourier,
            torch.zeros_like(weight_delta.center),
            weight_delta.center,
        ),
        torch.where(
            invariant_fourier,
            torch.zeros_like(weight_delta.radius),
            weight_delta.radius,
        ),
    )

    rd_ball = _ball_mul(r_ball, d_ball)
    linear_terms = _ball_mul(
        _BallTensor(
            rd_ball.center[:, None, :],
            rd_ball.radius[:, None, :],
        ),
        _BallTensor(
            weight_delta.center[None, :, :],
            weight_delta.radius[None, :, :],
        ),
    )
    r2_ball = _ball_mul(r_ball, r_ball)
    r3_ball = _ball_mul(r2_ball, r_ball)
    common = _ball_mul(d_ball, c_ball)
    q_terms = _ball_scale(_ball_mul(r2_ball, common), 2.0)
    c_terms = _ball_scale(_ball_mul(r3_ball, common), 3.0)

    edge_active = (pair_total > 0.0) & (~exact_no_motion)
    if duration == 0.0:
        edge_active = torch.zeros_like(edge_active)

    def mask_edges(ball: _BallTensor, active: Tensor) -> _BallTensor:
        return _BallTensor(
            torch.where(active, ball.center, torch.zeros_like(ball.center)),
            torch.where(active, ball.radius, torch.zeros_like(ball.radius)),
        )

    linear_terms = mask_edges(linear_terms, edge_active[:, None, :])
    q_terms = mask_edges(q_terms, edge_active)
    c_terms = mask_edges(c_terms, edge_active)
    linear = _fixed_pairwise_sum(linear_terms)
    quadratic_sum = _fixed_pairwise_sum(q_terms)
    cubic_sum = _fixed_pairwise_sum(c_terms)
    quadratic = _BallTensor(
        quadratic_sum.center[:, None], quadratic_sum.radius[:, None]
    )
    cubic = _BallTensor(cubic_sum.center[:, None], cubic_sum.radius[:, None])
    center = torch.cat(
        (linear.center, quadratic.center, cubic.center), dim=1
    ).contiguous()
    radius = torch.cat(
        (linear.radius, quadratic.radius, cubic.radius), dim=1
    ).contiguous()

    center = torch.where(
        structural[None, :], torch.zeros_like(center), center
    ).contiguous()
    radius = torch.where(
        structural[None, :], torch.zeros_like(radius), radius
    ).contiguous()
    path_active = torch.any(edge_active, dim=1, keepdim=True)
    center = torch.where(
        path_active, center, torch.zeros_like(center)
    ).contiguous()
    radius = torch.where(
        path_active, radius, torch.zeros_like(radius)
    ).contiguous()
    if duration == 0.0:
        center.zero_()
        radius.zero_()

    diagnostics = {
        "version": PHASE_OBSERVER_VERSION,
        "matching_index": int(matching_index),
        "duration_fraction": duration,
        "path_count": path_count,
        "edge_count": path_count * edge_count,
        "observable_count": OBSERVABLE_COUNT,
        "structural_zero_count": int(np.count_nonzero(structural_np)),
        "structural_zero_names": [
            selected.names[index]
            for index, enabled in enumerate(structural_np)
            if enabled
        ],
        "quantile_enclosure_valid_count": int(
            torch.count_nonzero(valid).detach().cpu().item()
        ),
        "deterministic_pairwise_reduction": 1,
        "tolerance_zeroing_used": 0,
        "future_state_model_input": 0,
        "authorizing_observer_uses_later_state": 1,
        "zero_duration": int(duration == 0.0),
        "profile_bound": int(profile is not None),
    }
    return DynkinPhaseIncrementBatch(
        center=center,
        error_radius=radius,
        structural_zero_mask=structural.contiguous(),
        quantile_enclosure_valid=valid.contiguous(),
        diagnostics=diagnostics,
    )


def compute_dynkin_phase_observed_increment_from_states(
    states_before: Tensor,
    states_after: Tensor,
    *,
    matching_index: int,
    quantile_lower: Tensor,
    quantile_upper: Tensor,
    later_head_fraction: Tensor | None = None,
    profile: JacobiRBCudaProfile | None = None,
    spec: RefinementObservableSpec | None = None,
    duration_fraction: float = 1.0,
) -> DynkinPhaseIncrementBatch:
    """Gather one matching from full states and call the phase-local observer."""

    if (
        not isinstance(states_before, Tensor)
        or not isinstance(states_after, Tensor)
        or states_before.dtype != torch.float64
        or states_after.dtype != torch.float64
        or states_before.ndim != 2
        or states_before.shape != states_after.shape
        or states_before.shape[1] != PATH_STATE_SIZE
        or states_before.device != states_after.device
        or not states_before.is_contiguous()
        or not states_after.is_contiguous()
    ):
        raise TypeError("states must be aligned contiguous float64 [P,784]")
    if not bool(
        (
            torch.isfinite(states_before).all()
            & torch.isfinite(states_after).all()
            & (states_before >= 0.0).all()
            & (states_after >= 0.0).all()
        )
        .detach()
        .cpu()
        .item()
    ):
        raise ValueError("states must be finite and nonnegative")
    tails, heads = _matching_indices(
        matching_index, device=states_before.device
    )
    tail_mass = states_before.index_select(1, tails)
    head_mass = states_before.index_select(1, heads)
    pair_total = (tail_mass + head_mass).contiguous()
    positive = pair_total > 0.0
    denominator = torch.where(
        positive, pair_total, torch.ones_like(pair_total)
    )
    earlier = torch.where(
        positive, head_mass / denominator, torch.zeros_like(pair_total)
    ).contiguous()
    if later_head_fraction is None:
        later = torch.where(
            positive,
            states_after.index_select(1, heads) / denominator,
            torch.zeros_like(pair_total),
        ).contiguous()
    else:
        _validate_edge_tensor(
            later_head_fraction,
            "later_head_fraction",
            shape=(int(states_before.shape[0]), int(tails.numel())),
        )
        if later_head_fraction.device != states_before.device:
            raise ValueError("later_head_fraction must share the state device")
        later = later_head_fraction
    return compute_dynkin_phase_observed_increment(
        pair_total,
        earlier,
        later,
        matching_index=matching_index,
        quantile_lower=quantile_lower,
        quantile_upper=quantile_upper,
        profile=profile,
        spec=spec,
        duration_fraction=duration_fraction,
    )


def combine_dynkin_phase_residual(
    observed: DynkinPhaseIncrementBatch,
    drift: DynkinPhaseDriftBatch,
    *,
    spec: RefinementObservableSpec | None = None,
    standardized: bool = True,
) -> DynkinPhaseResidualBatch:
    """Combine observed and conditional increments with outward error bounds."""

    if (
        drift.center.shape != observed.center.shape
        or drift.center.device != observed.center.device
    ):
        raise ValueError("observed increment and drift must align")
    structural = observed.structural_zero_mask[None, :]
    drift_center = torch.where(
        structural, torch.zeros_like(drift.center), drift.center
    )
    drift_radius = torch.where(
        structural,
        torch.zeros_like(drift.error_radius),
        drift.error_radius,
    )
    residual = _ball_sub(
        _BallTensor(observed.center, observed.error_radius),
        _BallTensor(drift_center, drift_radius),
    )
    if standardized:
        selected = spec or refinement_observable_spec(GRID_SIZE)
        scales = torch.as_tensor(
            selected.standard_deviations,
            dtype=torch.float64,
            device=observed.center.device,
        )
        residual = _ball_div_positive(residual, scales)
    center = torch.where(
        structural, torch.zeros_like(residual.center), residual.center
    ).contiguous()
    radius = torch.where(
        structural, torch.zeros_like(residual.radius), residual.radius
    ).contiguous()
    return DynkinPhaseResidualBatch(
        center=center,
        error_radius=radius,
        observed_center=observed.center,
        observed_error_radius=observed.error_radius,
        drift_center=drift_center.contiguous(),
        drift_error_radius=drift_radius.contiguous(),
        structural_zero_mask=observed.structural_zero_mask,
        diagnostics={
            "version": PHASE_OBSERVER_VERSION,
            "standardized": int(bool(standardized)),
            "observed_quantile_enclosure_included": 1,
            "analytic_drift_enclosure_included": 1,
            "standardization_rounding_included": int(bool(standardized)),
            "quantile_enclosure_valid_count": int(
                torch.count_nonzero(observed.quantile_enclosure_valid)
                .detach()
                .cpu()
                .item()
            ),
            "drift_certificate_valid_count": (
                int(drift.center.shape[0])
                if drift.certificate_mask is None
                else int(
                    torch.count_nonzero(
                        torch.all(
                            drift.certificate_mask.reshape(
                                int(drift.center.shape[0]), -1
                            ),
                            dim=1,
                        )
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            ),
            "maximum_standardized_error_radius": float(
                torch.max(radius).detach().cpu().item()
            ),
            "structural_drift_zero_count": int(
                torch.count_nonzero(observed.structural_zero_mask)
                .detach()
                .cpu()
                .item()
            ),
        },
    )


def _global_observable_ball(
    states: Tensor, spec: RefinementObservableSpec
) -> _BallTensor:
    weights = torch.as_tensor(
        np.array(spec.fourier_weights, copy=True),
        dtype=torch.float64,
        device=states.device,
    )
    state_ball = _ball_point(states)
    fourier_terms = _ball_mul(
        _BallTensor(
            state_ball.center[:, None, :],
            state_ball.radius[:, None, :],
        ),
        _ball_point(weights[None, :, :]),
    )
    fourier = _fixed_pairwise_sum(fourier_terms)
    square = _ball_mul(state_ball, state_ball)
    cube = _ball_mul(square, state_ball)
    quadratic = _fixed_pairwise_sum(square)
    cubic = _fixed_pairwise_sum(cube)
    return _BallTensor(
        torch.cat(
            (fourier.center, quadratic.center[:, None], cubic.center[:, None]),
            dim=1,
        ),
        torch.cat(
            (fourier.radius, quadratic.radius[:, None], cubic.radius[:, None]),
            dim=1,
        ),
    )


def compute_advisory_global_phase_increment(
    states_before: Tensor,
    states_after: Tensor,
    *,
    spec: RefinementObservableSpec | None = None,
) -> AdvisoryGlobalPhaseIncrement:
    """Reproduce the old global subtraction and attach an error enclosure."""

    if (
        states_before.dtype != torch.float64
        or states_after.dtype != torch.float64
        or states_before.shape != states_after.shape
        or states_before.ndim != 2
        or states_before.shape[1] != PATH_STATE_SIZE
        or states_before.device != states_after.device
    ):
        raise TypeError("states must be aligned float64 [P,784]")
    selected = spec or refinement_observable_spec(GRID_SIZE)
    raw_before = evaluate_refinement_observables(
        states_before, spec=selected, standardized=False
    )
    raw_after = evaluate_refinement_observables(
        states_after, spec=selected, standardized=False
    )
    assert isinstance(raw_before, Tensor) and isinstance(raw_after, Tensor)
    before_ball = _global_observable_ball(states_before, selected)
    after_ball = _global_observable_ball(states_after, selected)
    difference_ball = _ball_sub(after_ball, before_ball)
    return AdvisoryGlobalPhaseIncrement(
        center=(raw_after - raw_before).contiguous(),
        error_radius=difference_ball.radius.contiguous(),
        diagnostics={
            "version": PHASE_OBSERVER_VERSION,
            "advisory_only": 1,
            "independent_global_reduction_count": 2,
            "authorizing": 0,
        },
    )


__all__ = [
    "PHASE_OBSERVER_VERSION",
    "DynkinPhaseIncrementBatch",
    "DynkinPhaseResidualBatch",
    "AdvisoryGlobalPhaseIncrement",
    "compute_dynkin_phase_observed_increment",
    "compute_dynkin_phase_observed_increment_from_states",
    "combine_dynkin_phase_residual",
    "compute_advisory_global_phase_increment",
]
