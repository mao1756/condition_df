"""Additive fused/deferred boundary-tangent execution adapter.

This module contains the exploratory rollout-only protocol, telemetry, and
duplicate-ID fused phase integration.  The provenance-bound historical
`d0_jacobi_rb_boundary_tangent` module remains byte-identical to its sealed
version; scientific extensions live here and reuse its public geometry and
contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

from mnist.d0_jacobi_rb_boundary_tangent import *  # noqa: F403
from mnist.d0_jacobi_rb_boundary_tangent import BoundaryTangentContractError
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
    matching_indices,
)
from mnist import d0_jacobi_rb_reverse_controller as _reverse_controller


@runtime_checkable
class TangentScoreController(Protocol):
    """Structural interface required by the tangent phase integrator.

    The protocol intentionally describes only the permitted score call.  It
    allows independently implemented predictors to use the already-audited
    reference/control/reference composition without inheriting from the
    historical :class:`BoundaryTangentPredictor`.
    """

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        """Return a finite tangent-score coefficient for every matched edge."""


@dataclass(frozen=True)
class TangentControlledPhaseResult(_reverse_controller.ControlledPhaseResult):
    """Controlled phase result with additive tangent-flow telemetry.

    The base fields and their meanings are unchanged, so existing callers that
    consume :class:`ControlledPhaseResult` remain compatible.  Squared sums
    and counts are sufficient to combine phase records and recover RMS values
    without retaining per-edge diagnostics.
    """

    reference_fraction_displacement_squared_sum: float = 0.0
    reference_fraction_displacement_count: int = 0
    reference_fraction_displacement_maximum_absolute: float = 0.0
    control_fraction_displacement_squared_sum: float = 0.0
    control_fraction_displacement_count: int = 0
    control_fraction_displacement_maximum_absolute: float = 0.0
    score_squared_sum: float = 0.0
    score_count: int = 0
    score_maximum_absolute: float = 0.0
    logistic_shift_squared_sum: float = 0.0
    logistic_shift_count: int = 0
    logistic_shift_maximum_absolute: float = 0.0
    boundary_fraction_count: int = 0


@dataclass(frozen=True)
class TangentPhaseDeviceTelemetry:
    """Per-row phase reductions retained on the state device.

    No method on this record converts a tensor to a Python scalar.  The fused
    scheduler combines these records for an eight-step shard and performs one
    packed validation transfer at the commit boundary.
    """

    sums: Mapping[str, Tensor]
    maxima: Mapping[str, Tensor]
    failure_flags: Mapping[str, Tensor]


@dataclass(frozen=True)
class FusedTangentPhaseResult:
    """Device-resident result of one duplicate-ID fused tangent phase."""

    state: Tensor
    midpoint_reverse_times: tuple[float, ...]
    transition_count: int
    telemetry: TangentPhaseDeviceTelemetry



def _frozen_score_logistic_fraction_device(
    y: Tensor, score: Tensor, exposure: Tensor
) -> Tensor:
    """Shared exact tensor math; callers choose their validation boundary."""

    interior = (y > 0.0) & (y < 1.0) & (exposure != 0.0) & (score != 0.0)
    shift = 2.0 * score * exposure
    positive = interior & (shift >= 0.0)
    negative = interior & (shift < 0.0)
    # Mask before exponentiation so inactive lanes cannot overflow.  Dense
    # ``where`` algebra is deliberate: CUDA boolean advanced indexing invokes
    # a dynamic nonzero path which may synchronize the host.
    positive_shift = torch.where(positive, shift, torch.zeros_like(shift))
    exp_negative = torch.exp(-positive_shift)
    positive_result = y / (
        y + (1.0 - y) * exp_negative
    )
    negative_shift = torch.where(negative, shift, torch.zeros_like(shift))
    exp_positive = torch.exp(negative_shift)
    negative_result = y * exp_positive / (
        (1.0 - y) + y * exp_positive
    )
    return torch.where(
        positive,
        positive_result,
        torch.where(negative, negative_result, y),
    )


def frozen_score_logistic_fraction(
    head_fraction: Tensor, frozen_score: Tensor, delta_u: Tensor | float
) -> Tensor:
    """Exact flow of ``dy/du=2*y*(1-y)*q`` for frozen finite ``q``."""

    if (
        not isinstance(head_fraction, Tensor)
        or not isinstance(frozen_score, Tensor)
        or head_fraction.shape != frozen_score.shape
        or not head_fraction.dtype.is_floating_point
        or not frozen_score.dtype.is_floating_point
        or head_fraction.device != frozen_score.device
    ):
        raise BoundaryTangentContractError("logistic-flow tensors are malformed")
    y = head_fraction.to(dtype=torch.float64)
    score = frozen_score.to(dtype=torch.float64)
    exposure = torch.as_tensor(delta_u, dtype=torch.float64, device=y.device)
    try:
        exposure = torch.broadcast_to(exposure, y.shape)
    except RuntimeError as exc:
        raise BoundaryTangentContractError("delta_u is not broadcastable") from exc
    if (
        not bool(torch.isfinite(y).all())
        or not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(exposure).all())
        or bool(torch.any((y < 0.0) | (y > 1.0)))
        or bool(torch.any(exposure < 0.0))
    ):
        raise BoundaryTangentContractError("logistic-flow values are invalid")
    result = _frozen_score_logistic_fraction_device(y, score, exposure)
    if not bool(torch.isfinite(result).all()) or bool(
        torch.any((result < 0.0) | (result > 1.0))
    ):
        raise BoundaryTangentContractError("logistic flow produced an invalid fraction")
    return result


def _scatter_pair_fraction_device(
    states: Tensor,
    tails: Tensor,
    heads: Tensor,
    pair_mass: Tensor,
    fraction: Tensor,
) -> Tensor:
    """Shared pair-mass preserving state update with no host predicate."""

    active = pair_mass > 0.0
    next_head = pair_mass * fraction
    next_tail = pair_mass - next_head
    tail_values = torch.where(active, next_tail, states[:, tails])
    head_values = torch.where(active, next_head, states[:, heads])
    # Matching indices are fixed-size integer tensors.  index_copy therefore
    # avoids the dynamic masked-scatter/nonzero path used by boolean indexing.
    output = torch.index_copy(states, 1, tails, tail_values)
    output = torch.index_copy(output, 1, heads, head_values)
    return output


def _frozen_score_logistic_flow_device(
    states: Tensor,
    tails: Tensor,
    heads: Tensor,
    score: Tensor,
    exposure: Tensor,
) -> Tensor:
    tail = states[:, tails]
    head = states[:, heads]
    pair = tail + head
    active = pair > 0.0
    fraction = torch.where(
        active, head / torch.where(active, pair, torch.ones_like(pair)), torch.zeros_like(pair)
    )
    next_fraction = _frozen_score_logistic_fraction_device(
        fraction, score.to(dtype=torch.float64), exposure
    )
    return _scatter_pair_fraction_device(states, tails, heads, pair, next_fraction)


def frozen_score_logistic_flow(
    state: Tensor,
    matching: int | tuple[Tensor, Tensor],
    frozen_score: Tensor,
    delta_u: Tensor | float,
) -> Tensor:
    """Apply the exact frozen-score logistic flow while preserving pair mass."""

    if not isinstance(state, Tensor) or state.dtype != torch.float64:
        raise BoundaryTangentContractError("state must be float64")
    squeezed = state.ndim == 1
    states = state.unsqueeze(0) if squeezed else state
    if states.ndim != 2 or states.shape[1] != STATE_SIZE:
        raise BoundaryTangentContractError("state must have shape [P,784]")
    if not bool(torch.isfinite(states).all()) or bool(torch.any(states < 0.0)):
        raise BoundaryTangentContractError("state is not finite and nonnegative")
    if isinstance(matching, tuple):
        tails, heads = matching
        tails = tails.to(device=states.device, dtype=torch.long).reshape(-1)
        heads = heads.to(device=states.device, dtype=torch.long).reshape(-1)
    else:
        index = int(matching)
        if not 0 <= index < 4:
            raise BoundaryTangentContractError("matching is outside [0,4)")
        all_tails, all_heads = matching_indices(device=states.device)
        tails, heads = all_tails[index], all_heads[index]
    if tails.shape != (EDGES_PER_PHASE,) or heads.shape != tails.shape:
        raise BoundaryTangentContractError("matching has the wrong shape")
    score = frozen_score.unsqueeze(0) if frozen_score.ndim == 1 else frozen_score
    if score.shape != (states.shape[0], EDGES_PER_PHASE):
        raise BoundaryTangentContractError("frozen score must have shape [P,392]")
    exposure = torch.as_tensor(delta_u, dtype=torch.float64, device=states.device)
    try:
        exposure = torch.broadcast_to(exposure, score.shape)
    except RuntimeError as exc:
        raise BoundaryTangentContractError("delta_u is not broadcastable") from exc
    if (
        not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(exposure).all())
        or bool(torch.any(exposure < 0.0))
    ):
        raise BoundaryTangentContractError("logistic-flow values are invalid")
    output = _frozen_score_logistic_flow_device(
        states, tails, heads, score.to(dtype=torch.float64), exposure
    )
    if not bool(torch.isfinite(output).all()) or bool(torch.any(output < 0.0)):
        raise BoundaryTangentContractError("logistic flow produced an invalid state")
    return output[0] if squeezed else output



def controlled_reverse_phase_tangent(
    state: Tensor,
    k: Any,
    phase: Any,
    M: Any,
    transition_namespace: str,
    *,
    controller: TangentScoreController,
    reference_transition: _reverse_controller.ReferenceTransition,
    path_ids: Sequence[int],
    label: int | Tensor,
) -> _reverse_controller.ControlledPhaseResult:
    """Exact-reference/tangent-control/exact-reference phase composition.

    This mirrors the frozen controller's split order and transition IDs.  The
    only changed operation is the learned subflow: the model returns its
    direct finite ``q`` coefficient, which is advanced by the exact logistic
    flow instead of an affine fraction step.
    """

    if not isinstance(controller, TangentScoreController) or not callable(
        getattr(controller, "score_prediction", None)
    ):
        raise BoundaryTangentContractError(
            "controller must implement score_prediction(ModelInputs)"
        )
    step = _reverse_controller._index(k, "k")  # noqa: SLF001
    occurrence = _reverse_controller._index(phase, "phase")  # noqa: SLF001
    microsteps = _reverse_controller._index(M, "M")  # noqa: SLF001
    if microsteps not in _reverse_controller.REFINEMENT_CONTROL_MICROSTEPS:
        raise BoundaryTangentContractError("M must be one of the frozen {2,4,8}")
    if transition_namespace != _reverse_controller.NAMESPACE_VERSION:
        raise BoundaryTangentContractError("transition namespace changed")
    if not callable(reference_transition):
        raise BoundaryTangentContractError("certified reference callback is missing")
    states, squeezed = _reverse_controller._batched_state(state)  # noqa: SLF001
    paths = tuple(_reverse_controller._index(item, "path_id") for item in path_ids)  # noqa: SLF001
    if len(paths) != states.shape[0] or len(set(paths)) != len(paths):
        raise BoundaryTangentContractError("path IDs must uniquely identify each state")
    color = PHASE_MATCHINGS[occurrence]
    duration = PHASE_DURATIONS[occurrence]
    tails, heads = _reverse_controller._matching_tensors(  # noqa: SLF001
        color, device=states.device
    )
    initial_total = torch.sum(states, dim=1)
    pair_mass = states[:, tails] + states[:, heads]
    full_exposure = _reverse_controller.phase_exposure(pair_mass, duration)
    delta_u = full_exposure / float(microsteps)
    midpoint_times: list[float] = []
    maximum_pair_error = 0.0
    maximum_simplex_error = 0.0
    reference_displacement_squared_sum = 0.0
    reference_displacement_count = 0
    reference_displacement_maximum = 0.0
    control_displacement_squared_sum = 0.0
    control_displacement_count = 0
    control_displacement_maximum = 0.0
    score_squared_sum = 0.0
    score_count = 0
    score_maximum = 0.0
    logistic_shift_squared_sum = 0.0
    logistic_shift_count = 0
    logistic_shift_maximum = 0.0
    boundary_fraction_count = 0

    for reverse_index, j in enumerate(range(microsteps, 0, -1)):
        for side in ("pre", "post"):
            role = f"reverse_reference_{side}_control_M{microsteps}"
            head_fraction = torch.zeros_like(pair_mass)
            active = pair_mass > 0.0
            head_fraction[active] = states[:, heads][active] / pair_mass[active]
            ids = _reverse_controller.controller_transition_ids(
                paths,
                outer_step=step,
                phase=occurrence,
                reverse_microstep=reverse_index,
                role=role,
                device=states.device,
            )
            result = reference_transition(
                head_fraction=head_fraction,
                exposure=delta_u / 2.0,
                transition_ids=ids,
                role=role,
            )
            fraction = _reverse_controller._reference_fraction(  # noqa: SLF001
                result, tuple(pair_mass.shape)
            ).to(device=states.device, dtype=torch.float64)
            if not bool(torch.isfinite(fraction).all()) or bool(
                torch.any((fraction < 0.0) | (fraction > 1.0))
            ):
                raise BoundaryTangentContractError(
                    "certified reference fraction lies outside [0,1]"
                )
            reference_displacement = fraction[active] - head_fraction[active]
            if reference_displacement.numel():
                reference_displacement_squared_sum += float(
                    torch.sum(reference_displacement.square()).item()
                )
                reference_displacement_count += int(reference_displacement.numel())
                reference_displacement_maximum = max(
                    reference_displacement_maximum,
                    float(torch.max(torch.abs(reference_displacement)).item()),
                )
            states = _scatter_pair_fraction_device(
                states, tails, heads, pair_mass, fraction
            )
            current_pair = states[:, tails] + states[:, heads]
            maximum_pair_error = max(
                maximum_pair_error,
                float(torch.max(torch.abs(current_pair - pair_mass)).item()),
            )
            maximum_simplex_error = max(
                maximum_simplex_error,
                float(
                    torch.max(
                        torch.abs(torch.sum(states, dim=1) - initial_total)
                    ).item()
                ),
            )
            if side == "pre":
                q_mid = (j - 0.5) / float(microsteps)
                reverse_time = _reverse_controller.internal_reverse_time(
                    step, occurrence, q_mid
                )
                midpoint_times.append(reverse_time)
                labels = (
                    label.to(device=states.device, dtype=torch.long).reshape(-1)
                    if isinstance(label, Tensor)
                    else torch.full(
                        (states.shape[0],),
                        int(label),
                        dtype=torch.long,
                        device=states.device,
                    )
                )
                if labels.shape != (states.shape[0],):
                    raise BoundaryTangentContractError("label must be scalar or [P]")
                inputs = ModelInputs(
                    later_full_state=states.to(dtype=torch.float32),
                    reverse_time=torch.full(
                        (states.shape[0],),
                        reverse_time,
                        dtype=torch.float64,
                        device=states.device,
                    ),
                    phase=torch.full(
                        (states.shape[0],),
                        occurrence,
                        dtype=torch.long,
                        device=states.device,
                    ),
                    color=torch.full(
                        (states.shape[0],),
                        color,
                        dtype=torch.long,
                        device=states.device,
                    ),
                    duration=torch.full(
                        (states.shape[0],),
                        duration,
                        dtype=torch.float32,
                        device=states.device,
                    ),
                    label=labels,
                )
                if type(inputs) is not ModelInputs:
                    raise BoundaryTangentContractError(
                        "controller input must be an exact ModelInputs object"
                    )
                score = controller.score_prediction(inputs)
                if (
                    not isinstance(score, Tensor)
                    or score.shape != pair_mass.shape
                    or not score.dtype.is_floating_point
                    or score.device != states.device
                    or not bool(torch.isfinite(score).all())
                ):
                    raise BoundaryTangentContractError(
                        "controller score must be a finite floating [P,392] tensor "
                        "on the state device"
                    )
                control_input_fraction = fraction
                active_score = score[active].to(dtype=torch.float64)
                active_shift = (
                    2.0 * score.to(dtype=torch.float64) * delta_u
                )[active]
                if active_score.numel():
                    if not bool(torch.isfinite(active_shift).all()):
                        raise BoundaryTangentContractError(
                            "controller logistic shift must be finite"
                        )
                    score_squared_sum += float(torch.sum(active_score.square()).item())
                    score_count += int(active_score.numel())
                    score_maximum = max(
                        score_maximum,
                        float(torch.max(torch.abs(active_score)).item()),
                    )
                    logistic_shift_squared_sum += float(
                        torch.sum(active_shift.square()).item()
                    )
                    logistic_shift_count += int(active_shift.numel())
                    logistic_shift_maximum = max(
                        logistic_shift_maximum,
                        float(torch.max(torch.abs(active_shift)).item()),
                    )
                    boundary_fraction_count += int(
                        torch.count_nonzero(
                            active
                            & (
                                (control_input_fraction == 0.0)
                                | (control_input_fraction == 1.0)
                            )
                        ).item()
                    )
                states = frozen_score_logistic_flow(
                    states, (tails, heads), score, delta_u
                )
                control_output_fraction = torch.zeros_like(pair_mass)
                control_output_fraction[active] = (
                    states[:, heads][active] / pair_mass[active]
                )
                control_displacement = (
                    control_output_fraction[active] - control_input_fraction[active]
                )
                if control_displacement.numel():
                    control_displacement_squared_sum += float(
                        torch.sum(control_displacement.square()).item()
                    )
                    control_displacement_count += int(control_displacement.numel())
                    control_displacement_maximum = max(
                        control_displacement_maximum,
                        float(torch.max(torch.abs(control_displacement)).item()),
                    )
                current_pair = states[:, tails] + states[:, heads]
                maximum_pair_error = max(
                    maximum_pair_error,
                    float(torch.max(torch.abs(current_pair - pair_mass)).item()),
                )
                maximum_simplex_error = max(
                    maximum_simplex_error,
                    float(
                        torch.max(
                            torch.abs(torch.sum(states, dim=1) - initial_total)
                        ).item()
                    ),
                )

    if maximum_pair_error > 2.0e-12 or maximum_simplex_error > 2.0e-12:
        raise BoundaryTangentContractError("tangent phase violated simplex mass")
    return TangentControlledPhaseResult(
        state=states[0] if squeezed else states,
        midpoint_reverse_times=tuple(midpoint_times),
        transition_count=2 * microsteps * len(paths) * EDGES_PER_PHASE,
        maximum_pair_mass_error=maximum_pair_error,
        maximum_simplex_mass_error=maximum_simplex_error,
        reference_fraction_displacement_squared_sum=(
            reference_displacement_squared_sum
        ),
        reference_fraction_displacement_count=reference_displacement_count,
        reference_fraction_displacement_maximum_absolute=(
            reference_displacement_maximum
        ),
        control_fraction_displacement_squared_sum=control_displacement_squared_sum,
        control_fraction_displacement_count=control_displacement_count,
        control_fraction_displacement_maximum_absolute=control_displacement_maximum,
        score_squared_sum=score_squared_sum,
        score_count=score_count,
        score_maximum_absolute=score_maximum,
        logistic_shift_squared_sum=logistic_shift_squared_sum,
        logistic_shift_count=logistic_shift_count,
        logistic_shift_maximum_absolute=logistic_shift_maximum,
        boundary_fraction_count=boundary_fraction_count,
    )



def controlled_reverse_phase_tangent_fused(
    state: Tensor,
    k: Any,
    phase: Any,
    M: Any,
    transition_namespace: str,
    *,
    controller_bank: TangentScoreController,
    reference_transition: _reverse_controller.ReferenceTransition,
    row_keys: Sequence[str],
    canonical_path_ids: Sequence[int],
    label: int | Tensor,
    prebuilt_transition_ids: Tensor,
    prebuilt_matching_tails: Tensor,
    prebuilt_matching_heads: Tensor,
) -> FusedTangentPhaseResult:
    """Duplicate-ID fused phase with device-resident validation telemetry.

    ``row_keys`` identify independent scientific rows.  The canonical path
    IDs intentionally need not be unique: equal IDs produce equal exact
    transition IDs and therefore common random bits.  This entry point has no
    tensor-to-host predicate in its phase loop.  Its caller must validate the
    returned device flags at the atomic shard boundary before committing state.
    """

    if not isinstance(controller_bank, TangentScoreController) or not callable(
        getattr(controller_bank, "score_prediction", None)
    ):
        raise BoundaryTangentContractError(
            "controller bank must implement score_prediction(ModelInputs)"
        )
    step = _reverse_controller._index(k, "k")  # noqa: SLF001
    occurrence = _reverse_controller._index(phase, "phase")  # noqa: SLF001
    microsteps = _reverse_controller._index(M, "M")  # noqa: SLF001
    if microsteps not in _reverse_controller.REFINEMENT_CONTROL_MICROSTEPS:
        raise BoundaryTangentContractError("M must be one of the frozen {2,4,8}")
    if transition_namespace != _reverse_controller.NAMESPACE_VERSION:
        raise BoundaryTangentContractError("transition namespace changed")
    if not callable(reference_transition):
        raise BoundaryTangentContractError("certified reference callback is missing")
    if not isinstance(state, Tensor) or state.dtype != torch.float64:
        raise BoundaryTangentContractError("fused state must be float64")
    states = state.unsqueeze(0) if state.ndim == 1 else state
    if states.ndim != 2 or states.shape[1] != STATE_SIZE:
        raise BoundaryTangentContractError("fused state must have shape [R,784]")
    rows = int(states.shape[0])
    keys = tuple(row_keys)
    if (
        len(keys) != rows
        or len(set(keys)) != rows
        or any(not isinstance(value, str) or not value for value in keys)
    ):
        raise BoundaryTangentContractError("row keys must uniquely identify each row")
    paths = tuple(
        _reverse_controller._index(item, "canonical_path_id")  # noqa: SLF001
        for item in canonical_path_ids
    )
    if len(paths) != rows:
        raise BoundaryTangentContractError(
            "canonical path IDs must match fused state rows"
        )
    if (
        not isinstance(prebuilt_matching_tails, Tensor)
        or not isinstance(prebuilt_matching_heads, Tensor)
        or prebuilt_matching_tails.shape != (4, EDGES_PER_PHASE)
        or prebuilt_matching_heads.shape != prebuilt_matching_tails.shape
        or prebuilt_matching_tails.dtype != torch.long
        or prebuilt_matching_heads.dtype != torch.long
        or prebuilt_matching_tails.device != states.device
        or prebuilt_matching_heads.device != states.device
        or not prebuilt_matching_tails.is_contiguous()
        or not prebuilt_matching_heads.is_contiguous()
    ):
        raise BoundaryTangentContractError(
            "prebuilt matching tables must be contiguous [4,392] int64 tensors "
            "on the state device"
        )
    color = PHASE_MATCHINGS[occurrence]
    duration = PHASE_DURATIONS[occurrence]
    tails = prebuilt_matching_tails[color]
    heads = prebuilt_matching_heads[color]
    initial_total = torch.sum(states, dim=1)
    pair_mass = states[:, tails] + states[:, heads]
    active = pair_mass > 0.0
    coefficient = (
        (2.0 * float(_reverse_controller.ALPHA) + 1.0)
        * float(_reverse_controller.MACROSTEP_SCHEDULE_INTEGRAL)
        / (
            float(_reverse_controller.ALPHA)
            * float(_reverse_controller.GRID_SPACING) ** 2
        )
    )
    # Match ``phase_exposure``'s binary64 operation order exactly while using
    # dense masking.  Keeping the numerator as a device tensor avoids scalar
    # constant folding that differs by one ulp on some lanes.
    safe_pair_mass = torch.where(active, pair_mass, torch.ones_like(pair_mass))
    numerator = coefficient * torch.full_like(pair_mass, float(duration))
    full_exposure = torch.where(
        active, numerator / safe_pair_mass, torch.zeros_like(pair_mass)
    )
    delta_u = full_exposure / float(microsteps)

    if (
        not isinstance(prebuilt_transition_ids, Tensor)
        or prebuilt_transition_ids.shape
        != (microsteps, 2, rows, EDGES_PER_PHASE)
        or prebuilt_transition_ids.device != states.device
        or prebuilt_transition_ids.dtype not in (torch.int64, torch.uint64)
        or not prebuilt_transition_ids.is_contiguous()
    ):
        raise BoundaryTangentContractError(
            "prebuilt transition IDs must be contiguous [M,2,R,392] integers "
            "on the state device"
        )

    float_sums = {
        name: torch.zeros(rows, dtype=torch.float64, device=states.device)
        for name in (
            "reference_fraction_displacement_squared_sum",
            "control_fraction_displacement_squared_sum",
            "score_squared_sum",
            "logistic_shift_squared_sum",
        )
    }
    int_sums = {
        name: torch.zeros(rows, dtype=torch.int64, device=states.device)
        for name in (
            "reference_fraction_displacement_count",
            "control_fraction_displacement_count",
            "score_count",
            "logistic_shift_count",
            "boundary_fraction_count",
            "transition_count",
        )
    }
    maxima = {
        name: torch.zeros(rows, dtype=torch.float64, device=states.device)
        for name in (
            "reference_fraction_displacement_maximum_absolute",
            "control_fraction_displacement_maximum_absolute",
            "score_maximum_absolute",
            "logistic_shift_maximum_absolute",
            "maximum_pair_mass_error",
            "maximum_simplex_mass_error",
        )
    }
    failure_flags = {
        name: torch.zeros(rows, dtype=torch.bool, device=states.device)
        for name in (
            "input_invalid",
            "reference_fraction_invalid",
            "score_invalid",
            "logistic_shift_invalid",
            "state_invalid",
            "mass_invalid",
            "metadata_invalid",
        )
    }
    failure_flags["input_invalid"] |= (~torch.isfinite(states)).any(dim=1) | (
        states < 0.0
    ).any(dim=1)
    failure_flags["input_invalid"] |= (
        torch.abs(initial_total - 1.0) > 2.0e-12
    )

    labels = (
        label.to(device=states.device, dtype=torch.long).reshape(-1)
        if isinstance(label, Tensor)
        else torch.full(
            (rows,), int(label), dtype=torch.long, device=states.device
        )
    )
    if labels.shape != (rows,):
        raise BoundaryTangentContractError("label must be scalar or [R]")
    failure_flags["metadata_invalid"] |= (labels < 0) | (labels >= 10)
    midpoint_times: list[float] = []

    def accumulate(
        prefix: str, values: Tensor, mask: Tensor
    ) -> None:
        safe = torch.where(mask & torch.isfinite(values), values, torch.zeros_like(values))
        float_sums[f"{prefix}_squared_sum"] += torch.sum(
            safe.square(), dim=1, dtype=torch.float64
        )
        int_sums[f"{prefix}_count"] += torch.sum(mask, dim=1, dtype=torch.int64)
        maxima[f"{prefix}_maximum_absolute"] = torch.maximum(
            maxima[f"{prefix}_maximum_absolute"],
            torch.amax(torch.where(mask, torch.abs(safe), torch.zeros_like(safe)), dim=1),
        )

    for reverse_index, j in enumerate(range(microsteps, 0, -1)):
        for side_index, side in enumerate(("pre", "post")):
            role = f"reverse_reference_{side}_control_M{microsteps}"
            current_head = states[:, heads]
            head_fraction = torch.where(
                active,
                current_head
                / torch.where(active, pair_mass, torch.ones_like(pair_mass)),
                torch.zeros_like(pair_mass),
            )
            ids = prebuilt_transition_ids[reverse_index, side_index]
            result = reference_transition(
                head_fraction=head_fraction,
                exposure=(delta_u / 2.0).contiguous(),
                transition_ids=ids,
                role=role,
            )
            fraction = _reverse_controller._reference_fraction(  # noqa: SLF001
                result, tuple(pair_mass.shape)
            ).to(device=states.device, dtype=torch.float64)
            valid_fraction = torch.isfinite(fraction) & (fraction >= 0.0) & (
                fraction <= 1.0
            )
            failure_flags["reference_fraction_invalid"] |= (
                active & ~valid_fraction
            ).any(dim=1)
            reference_displacement = fraction - head_fraction
            accumulate(
                "reference_fraction_displacement",
                reference_displacement,
                active & valid_fraction,
            )
            states = _scatter_pair_fraction_device(
                states, tails, heads, pair_mass, fraction
            )
            current_pair = states[:, tails] + states[:, heads]
            pair_error = torch.amax(torch.abs(current_pair - pair_mass), dim=1)
            simplex_error = torch.abs(torch.sum(states, dim=1) - initial_total)
            maxima["maximum_pair_mass_error"] = torch.maximum(
                maxima["maximum_pair_mass_error"], pair_error
            )
            maxima["maximum_simplex_mass_error"] = torch.maximum(
                maxima["maximum_simplex_mass_error"], simplex_error
            )

            if side == "pre":
                q_mid = (j - 0.5) / float(microsteps)
                reverse_time = _reverse_controller.internal_reverse_time(
                    step, occurrence, q_mid
                )
                midpoint_times.append(reverse_time)
                inputs = ModelInputs(
                    later_full_state=states.to(dtype=torch.float32),
                    reverse_time=torch.full(
                        (rows,), reverse_time, dtype=torch.float64, device=states.device
                    ),
                    phase=torch.full(
                        (rows,), occurrence, dtype=torch.long, device=states.device
                    ),
                    color=torch.full(
                        (rows,), color, dtype=torch.long, device=states.device
                    ),
                    duration=torch.full(
                        (rows,), duration, dtype=torch.float32, device=states.device
                    ),
                    label=labels,
                )
                score = controller_bank.score_prediction(inputs)
                if (
                    not isinstance(score, Tensor)
                    or score.shape != pair_mass.shape
                    or not score.dtype.is_floating_point
                    or score.device != states.device
                ):
                    raise BoundaryTangentContractError(
                        "controller bank score must be floating [R,392] on device"
                    )
                score64 = score.to(dtype=torch.float64)
                valid_score = torch.isfinite(score64)
                shift = 2.0 * score64 * delta_u
                valid_shift = torch.isfinite(shift)
                failure_flags["score_invalid"] |= (active & ~valid_score).any(dim=1)
                failure_flags["logistic_shift_invalid"] |= (
                    active & ~valid_shift
                ).any(dim=1)
                accumulate("score", score64, active & valid_score)
                accumulate("logistic_shift", shift, active & valid_shift)
                int_sums["boundary_fraction_count"] += torch.sum(
                    active & ((fraction == 0.0) | (fraction == 1.0)),
                    dim=1,
                    dtype=torch.int64,
                )
                before_control = fraction
                states = _frozen_score_logistic_flow_device(
                    states, tails, heads, score64, delta_u
                )
                after_head = states[:, heads]
                after_fraction = torch.where(
                    active,
                    after_head
                    / torch.where(active, pair_mass, torch.ones_like(pair_mass)),
                    torch.zeros_like(pair_mass),
                )
                valid_after = torch.isfinite(after_fraction) & (
                    after_fraction >= 0.0
                ) & (after_fraction <= 1.0)
                accumulate(
                    "control_fraction_displacement",
                    after_fraction - before_control,
                    active & valid_after,
                )
                failure_flags["state_invalid"] |= (
                    (~torch.isfinite(states)).any(dim=1)
                    | (states < 0.0).any(dim=1)
                    | (active & ~valid_after).any(dim=1)
                )
                current_pair = states[:, tails] + states[:, heads]
                pair_error = torch.amax(torch.abs(current_pair - pair_mass), dim=1)
                simplex_error = torch.abs(torch.sum(states, dim=1) - initial_total)
                maxima["maximum_pair_mass_error"] = torch.maximum(
                    maxima["maximum_pair_mass_error"], pair_error
                )
                maxima["maximum_simplex_mass_error"] = torch.maximum(
                    maxima["maximum_simplex_mass_error"], simplex_error
                )

            int_sums["transition_count"] += EDGES_PER_PHASE

    failure_flags["state_invalid"] |= (~torch.isfinite(states)).any(dim=1) | (
        states < 0.0
    ).any(dim=1)
    failure_flags["mass_invalid"] |= (
        maxima["maximum_pair_mass_error"] > 2.0e-12
    ) | (maxima["maximum_simplex_mass_error"] > 2.0e-12)
    return FusedTangentPhaseResult(
        state=states,
        midpoint_reverse_times=tuple(midpoint_times),
        transition_count=2 * microsteps * rows * EDGES_PER_PHASE,
        telemetry=TangentPhaseDeviceTelemetry(
            sums={**float_sums, **int_sums},
            maxima=maxima,
            failure_flags=failure_flags,
        ),
    )




__all__ = [
    "FusedTangentPhaseResult",
    "TangentControlledPhaseResult",
    "TangentPhaseDeviceTelemetry",
    "TangentScoreController",
    "controlled_reverse_phase_tangent",
    "controlled_reverse_phase_tangent_fused",
    "frozen_score_logistic_flow",
    "frozen_score_logistic_fraction",
]

