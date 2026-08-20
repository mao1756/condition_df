"""Approximate-candidate CUDA adapter for bounded K=128/K=512 experiments.

This module is deliberately narrow.  It changes only the transition proposal
used by the immutable Eulerian Jacobi DDPM core: every active edge is sampled
by the existing 128-mode, 56-bisection CUDA candidate kernel.  The resulting
targets are approximate-candidate Rao--Blackwell targets; they are neither
certified transitions nor exact reverse scores.

Phase work stays device-resident.  Candidate masks, numerical health, and
conservation diagnostics are materialized and checked once at each enclosing
outer-step boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import time
from types import MappingProxyType
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import numpy as np
import torch

from mnist import eulerian_jacobi_ddpm as core
from mnist.d0_jacobi_rb_boundary_tangent import frozen_score_logistic_flow
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_deferred import (
    CandidateRBCudaBatch,
    PreparedCandidateRBCudaBackend,
    PreparedDeferredRBCudaSeed,
    enqueue_alpha1_rb_transition_batch_cuda_candidate,
    prepare_alpha1_rb_transition_batch_cuda_candidate,
    prepare_alpha1_rb_transition_cuda_rng_seed,
)
from mnist.d0_jacobi_rb_learnability import ModelInputs, matching_indices
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    canonical_refinement_transition_ids,
    refinement_phase_exposure,
)


CANDIDATE_PILOT_VERSION = "eulerian-jacobi-ddpm-candidate-k128-k512-v2"
CANDIDATE_BACKEND_NAME = "cuda-approximate-candidate-128m-56b"
CANDIDATE_TARGET_SEMANTICS = "approximate-candidate Rao--Blackwell target"

_DEFAULT_SAMPLE_STEPS = 128
_SUPPORTED_SAMPLE_STEPS = (128, 512)
_MAX_PATHS = 8
_MAX_LANES = 4096
_MASS_TOLERANCE = 2e-12
_COUNTER_FIELDS = (
    "candidate_active_count",
    "candidate_structural_noop_count",
    "candidate_approximation_count",
    "candidate_invalid_input_count",
    "candidate_invalid_output_count",
    "candidate_nonfinite_count",
    "candidate_negative_bracket_width_count",
    "candidate_bracket_order_invalid_count",
    "candidate_kernel_launch_count",
    "candidate_correction_count",
    "candidate_clipping_count",
    "candidate_floor_count",
    "candidate_limiter_count",
    "candidate_projection_count",
    "candidate_renormalization_count",
    "candidate_invalid_lane_count",
    "candidate_approximation_mismatch_count",
    "state_nonfinite_count",
    "state_negative_count",
    "controller_nonfinite_count",
)
_MAXIMUM_FIELDS = (
    "candidate_maximum_bracket_width",
    "maximum_mass_error",
    "maximum_pair_total_error",
)
_MUST_BE_ZERO = (
    "candidate_invalid_input_count",
    "candidate_invalid_output_count",
    "candidate_nonfinite_count",
    "candidate_negative_bracket_width_count",
    "candidate_bracket_order_invalid_count",
    "candidate_correction_count",
    "candidate_clipping_count",
    "candidate_floor_count",
    "candidate_limiter_count",
    "candidate_projection_count",
    "candidate_renormalization_count",
    "candidate_invalid_lane_count",
    "candidate_approximation_mismatch_count",
    "state_nonfinite_count",
    "state_negative_count",
    "controller_nonfinite_count",
)


class CandidatePilotError(core.EulerianJacobiDDPMError):
    """The approximate-candidate pilot contract was violated."""


@dataclass(frozen=True)
class CandidateRuntime:
    device: torch.device
    profile: JacobiRBCudaProfile
    prepared: PreparedCandidateRBCudaBackend
    prepared_seeds: Mapping[tuple[Any, ...], PreparedDeferredRBCudaSeed]
    candidate_binary_sha256: str


def _require_sample_steps(sample_steps: int) -> int:
    steps = int(sample_steps)
    if steps not in _SUPPORTED_SAMPLE_STEPS:
        raise CandidatePilotError(
            f"candidate runtime supports sample_steps in {_SUPPORTED_SAMPLE_STEPS}"
        )
    return steps


def _require_profile(profile: JacobiRBCudaProfile) -> None:
    if (
        int(profile.candidate_modes) != 128
        or int(profile.candidate_bisection_steps) != 56
        or int(profile.threads_per_block) != 128
    ):
        raise CandidatePilotError(
            "candidate profile must use 128 modes, 56 bisections, and 128 threads"
        )


def _path_ids(path_ids: Sequence[int], row_count: int) -> tuple[int, ...]:
    ids_list: list[int] = []
    for value in path_ids:
        if isinstance(value, bool):
            raise CandidatePilotError("candidate path IDs must be integers")
        try:
            path_id = operator.index(value)
        except TypeError as exc:
            raise CandidatePilotError("candidate path IDs must be integers") from exc
        if not 0 <= path_id < (1 << 20):
            raise CandidatePilotError("candidate path IDs must fit the 20-bit field")
        ids_list.append(path_id)
    ids = tuple(ids_list)
    if len(ids) != int(row_count) or not 1 <= len(ids) <= _MAX_PATHS:
        raise CandidatePilotError("candidate cohort must contain 1..8 aligned paths")
    if len(set(ids)) != len(ids):
        raise CandidatePilotError("candidate path IDs must be unique within a cohort")
    return ids


def _require_state_static(state: torch.Tensor, runtime: CandidateRuntime) -> None:
    if (
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.float64
        or state.ndim != 2
        or int(state.shape[1]) != core.STATE_SIZE
    ):
        raise CandidatePilotError("state must be float64 [P,784]")
    if state.device != runtime.device:
        raise CandidatePilotError("state and candidate runtime devices differ")
    if not 1 <= int(state.shape[0]) <= _MAX_PATHS:
        raise CandidatePilotError("candidate state must contain 1..8 paths")


def _score_logistic_flow_prevalidated(
    state: torch.Tensor,
    tails: torch.Tensor,
    heads: torch.Tensor,
    score: torch.Tensor,
    delta_u: torch.Tensor | float,
) -> torch.Tensor:
    """Apply the frozen-score flow without materializing a device predicate.

    The enclosing candidate outer step owns all value checks.  This helper
    therefore validates only static tensor metadata and expresses the stable
    logistic solution entirely as device tensor operations.
    """

    if (
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.float64
        or state.ndim != 2
        or state.shape[1] != core.STATE_SIZE
    ):
        raise CandidatePilotError("prevalidated logistic state is malformed")
    if (
        not isinstance(tails, torch.Tensor)
        or not isinstance(heads, torch.Tensor)
        or tails.shape != (EDGES_PER_PHASE,)
        or heads.shape != tails.shape
        or tails.device != state.device
        or heads.device != state.device
    ):
        raise CandidatePilotError("prevalidated logistic matching is malformed")
    if (
        not isinstance(score, torch.Tensor)
        or score.shape != (state.shape[0], EDGES_PER_PHASE)
        or score.device != state.device
        or not score.dtype.is_floating_point
    ):
        raise CandidatePilotError("prevalidated logistic score is malformed")
    exposure = torch.as_tensor(delta_u, dtype=torch.float64, device=state.device)
    try:
        exposure = torch.broadcast_to(exposure, score.shape)
    except RuntimeError as exc:
        raise CandidatePilotError("prevalidated logistic exposure is malformed") from exc

    tail = state[:, tails]
    head = state[:, heads]
    pair = tail + head
    active = pair > 0.0
    fraction = torch.where(active, head / torch.where(active, pair, torch.ones_like(pair)), torch.zeros_like(pair))
    score64 = score.to(dtype=torch.float64)
    interior = (
        (fraction > 0.0)
        & (fraction < 1.0)
        & (exposure != 0.0)
        & (score64 != 0.0)
    )
    shift = 2.0 * score64 * exposure
    positive = interior & (shift >= 0.0)
    negative = interior & (shift < 0.0)
    positive_shift = torch.where(positive, shift, torch.zeros_like(shift))
    negative_shift = torch.where(negative, shift, torch.zeros_like(shift))
    exp_negative = torch.exp(-positive_shift)
    positive_fraction = fraction / (
        fraction + (1.0 - fraction) * exp_negative
    )
    exp_positive = torch.exp(negative_shift)
    negative_fraction = fraction * exp_positive / (
        (1.0 - fraction) + fraction * exp_positive
    )
    next_fraction = torch.where(
        positive,
        positive_fraction,
        torch.where(negative, negative_fraction, fraction),
    )

    output = state.clone()
    moved = next_fraction != fraction
    next_head = pair * next_fraction
    output[:, heads] = torch.where(moved, next_head, head)
    output[:, tails] = torch.where(moved, pair - next_head, tail)
    return output


def _prepared_seed(
    runtime: CandidateRuntime, key: tuple[Any, ...]
) -> PreparedDeferredRBCudaSeed:
    try:
        return runtime.prepared_seeds[key]
    except KeyError as exc:
        raise CandidatePilotError(
            f"candidate RNG key was not prepared before the hot loop: {key!r}"
        ) from exc


def prepare_candidate_runtime(
    *,
    device: str | torch.device,
    rng_keys: Sequence[tuple[Any, ...]],
    profile: JacobiRBCudaProfile | None = None,
) -> CandidateRuntime:
    """Prepare the proposal kernel and the complete finite RNG-key bank."""

    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise CandidatePilotError("candidate production runtime requires CUDA")
    profile = JacobiRBCudaProfile() if profile is None else profile
    _require_profile(profile)
    prepared = prepare_alpha1_rb_transition_batch_cuda_candidate(
        device=selected_device, profile=profile
    )
    selected_device = torch.device(prepared.device)
    keys = tuple(tuple(key) for key in rng_keys)
    if not keys or len(set(keys)) != len(keys):
        raise CandidatePilotError("candidate RNG keys must be nonempty and unique")
    seeds = {
        key: prepare_alpha1_rb_transition_cuda_rng_seed(
            rng_key=key, prepared=prepared
        )
        for key in keys
    }
    binary_sha256 = str(prepared.candidate_binary_sha256)
    if len(binary_sha256) != 64:
        raise CandidatePilotError("candidate binary SHA-256 is malformed")
    return CandidateRuntime(
        device=selected_device,
        profile=profile,
        prepared=prepared,
        prepared_seeds=MappingProxyType(seeds),
        candidate_binary_sha256=binary_sha256,
    )


def _device_maximum(values: torch.Tensor) -> torch.Tensor:
    return torch.amax(values) if values.numel() else torch.zeros((), dtype=torch.float64, device=values.device)


def _phase_health(
    *,
    state_before: torch.Tensor,
    state_after: torch.Tensor,
    pair_before: torch.Tensor,
    tails: torch.Tensor,
    heads: torch.Tensor,
    batch: CandidateRBCudaBatch,
    runtime: CandidateRuntime,
) -> Mapping[str, Any]:
    diagnostics = batch.device_diagnostics
    return {
        "backend": CANDIDATE_BACKEND_NAME,
        "candidate_target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "candidate_modes": int(runtime.profile.candidate_modes),
        "candidate_bisection_steps": int(runtime.profile.candidate_bisection_steps),
        "candidate_binary_sha256": runtime.candidate_binary_sha256,
        "candidate_active_count": diagnostics["active_count"],
        "candidate_structural_noop_count": diagnostics["structural_noop_count"],
        "candidate_approximation_count": diagnostics["approximation_count"],
        "candidate_invalid_input_count": diagnostics["invalid_input_count"],
        "candidate_invalid_output_count": diagnostics["invalid_output_count"],
        "candidate_nonfinite_count": diagnostics["nonfinite_count"],
        "candidate_negative_bracket_width_count": diagnostics[
            "negative_bracket_width_count"
        ],
        "candidate_bracket_order_invalid_count": diagnostics[
            "bracket_order_invalid_count"
        ],
        "candidate_maximum_bracket_width": diagnostics[
            "maximum_candidate_bracket_width"
        ],
        "candidate_kernel_launch_count": diagnostics[
            "candidate_kernel_launch_count"
        ],
        "candidate_correction_count": diagnostics["correction_count"],
        "candidate_clipping_count": diagnostics["clipping_count"],
        "candidate_floor_count": diagnostics["floor_count"],
        "candidate_limiter_count": diagnostics["limiter_count"],
        "candidate_projection_count": diagnostics["projection_count"],
        "candidate_renormalization_count": diagnostics[
            "renormalization_count"
        ],
        "candidate_invalid_lane_count": (~batch.valid_mask).sum(dtype=torch.int64),
        "candidate_approximation_mismatch_count": (
            batch.approximation_mask != batch.active_mask
        ).sum(dtype=torch.int64),
        "state_nonfinite_count": (
            (~torch.isfinite(state_before)).sum(dtype=torch.int64)
            + (~torch.isfinite(state_after)).sum(dtype=torch.int64)
        ),
        "state_negative_count": (
            (state_before < 0.0).sum(dtype=torch.int64)
            + (state_after < 0.0).sum(dtype=torch.int64)
        ),
        "controller_nonfinite_count": torch.zeros(
            (), dtype=torch.int64, device=state_before.device
        ),
        "maximum_mass_error": torch.maximum(
            _device_maximum(torch.abs(state_before.sum(1) - 1.0)),
            _device_maximum(torch.abs(state_after.sum(1) - 1.0)),
        ),
        "maximum_pair_total_error": _device_maximum(
            torch.abs(state_after[:, tails] + state_after[:, heads] - pair_before)
        ),
    }


def candidate_forward_phase(
    state: torch.Tensor,
    path_ids: Sequence[int],
    *,
    outer_step: int,
    phase: int,
    root_seed: int,
    sample_steps: int,
    runtime: CandidateRuntime,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """Enqueue one forward phase without materializing device diagnostics."""

    steps = _require_sample_steps(sample_steps)
    _require_profile(runtime.profile)
    _require_state_static(state, runtime)
    ids = _path_ids(path_ids, int(state.shape[0]))
    step = int(outer_step)
    phase_index = int(phase)
    if not 0 <= step < steps or not 0 <= phase_index < len(PHASE_MATCHINGS):
        raise CandidatePilotError("candidate forward phase index is out of range")
    lane_count = int(state.shape[0]) * EDGES_PER_PHASE
    if lane_count > _MAX_LANES:
        raise CandidatePilotError("candidate enqueue exceeds the 4096-lane cap")

    tails_all, heads_all = matching_indices(device=state.device)
    color = int(PHASE_MATCHINGS[phase_index])
    tails, heads = tails_all[color], heads_all[color]
    pair = state[:, tails] + state[:, heads]
    fraction = torch.zeros_like(pair)
    active = pair > 0.0
    fraction[active] = state[:, heads][active] / pair[active]
    exposure = refinement_phase_exposure(
        pair,
        sample_steps=steps,
        duration_fraction=float(PHASE_DURATIONS[phase_index]),
    )
    transition_ids = canonical_refinement_transition_ids(
        ids,
        sample_steps=steps,
        outer_step=step,
        phase=phase_index,
        device=state.device,
    ).reshape_as(fraction)
    key = (int(root_seed), "forward")
    batch = enqueue_alpha1_rb_transition_batch_cuda_candidate(
        fraction,
        exposure,
        rng_key=key,
        transition_ids=transition_ids,
        prepared=runtime.prepared,
        prepared_rng_seed=_prepared_seed(runtime, key),
    )
    if not isinstance(batch, CandidateRBCudaBatch):
        raise CandidatePilotError("candidate dispatch returned a non-candidate batch")
    later_fraction = batch.later_head_fraction.to(dtype=torch.float64)
    output = state.clone()
    output[:, heads] = pair * later_fraction
    output[:, tails] = pair * (1.0 - later_fraction)
    health = _phase_health(
        state_before=state,
        state_after=output,
        pair_before=pair,
        tails=tails,
        heads=heads,
        batch=batch,
        runtime=runtime,
    )
    return output, batch.denoising_target.to(dtype=torch.float64), health


def candidate_forward_phase_prefixes(
    state: torch.Tensor,
    path_ids: Sequence[int],
    *,
    outer_step: int,
    phase: int,
    root_seed: int,
    sample_steps: int,
    prefix_fractions: Sequence[float] = tuple((2 * index + 1) / 16 for index in range(8)),
    runtime: CandidateRuntime,
) -> tuple[tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]], ...]:
    """Evaluate eager within-phase prefixes without advancing ``state``.

    Every prefix reuses the same RNG key and canonical transition IDs as the
    full phase.  The candidate inverse-CDF kernel therefore sees the same
    underlying uniform stream at each exposure; only the exposure duration is
    changed.  This mirrors the eager-prefix evidence contract while keeping
    the production forward transition itself untouched.
    """

    steps = _require_sample_steps(sample_steps)
    _require_profile(runtime.profile)
    _require_state_static(state, runtime)
    ids = _path_ids(path_ids, int(state.shape[0]))
    step = int(outer_step)
    phase_index = int(phase)
    if not 0 <= step < steps or not 0 <= phase_index < len(PHASE_MATCHINGS):
        raise CandidatePilotError("candidate prefix phase index is out of range")
    fractions = tuple(float(value) for value in prefix_fractions)
    if (
        not fractions
        or len(set(fractions)) != len(fractions)
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in fractions)
        or tuple(sorted(fractions)) != fractions
    ):
        raise CandidatePilotError(
            "prefix_fractions must be finite, unique, increasing, and in (0,1)"
        )

    tails_all, heads_all = matching_indices(device=state.device)
    color = int(PHASE_MATCHINGS[phase_index])
    tails, heads = tails_all[color], heads_all[color]
    pair = state[:, tails] + state[:, heads]
    earlier_fraction = torch.zeros_like(pair)
    active = pair > 0.0
    earlier_fraction[active] = state[:, heads][active] / pair[active]
    full_exposure = refinement_phase_exposure(
        pair,
        sample_steps=steps,
        duration_fraction=float(PHASE_DURATIONS[phase_index]),
    )
    transition_ids = canonical_refinement_transition_ids(
        ids,
        sample_steps=steps,
        outer_step=step,
        phase=phase_index,
        device=state.device,
    ).reshape_as(earlier_fraction)
    key = (int(root_seed), "forward")
    prepared_seed = _prepared_seed(runtime, key)
    results: list[tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]] = []
    for prefix_fraction in fractions:
        batch = enqueue_alpha1_rb_transition_batch_cuda_candidate(
            earlier_fraction,
            full_exposure * prefix_fraction,
            rng_key=key,
            transition_ids=transition_ids,
            prepared=runtime.prepared,
            prepared_rng_seed=prepared_seed,
        )
        if not isinstance(batch, CandidateRBCudaBatch):
            raise CandidatePilotError("candidate prefix dispatch returned a non-candidate batch")
        later_fraction = batch.later_head_fraction.to(dtype=torch.float64)
        output = state.clone()
        output[:, heads] = pair * later_fraction
        output[:, tails] = pair * (1.0 - later_fraction)
        health = _phase_health(
            state_before=state,
            state_after=output,
            pair_before=pair,
            tails=tails,
            heads=heads,
            batch=batch,
            runtime=runtime,
        )
        results.append(
            (output, batch.denoising_target.to(dtype=torch.float64), health)
        )
    return tuple(results)


def _aggregate_phase_health(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise CandidatePilotError("an outer step has no candidate phase diagnostics")
    first = parts[0]
    result: dict[str, Any] = {
        "backend": first["backend"],
        "candidate_target_semantics": first["candidate_target_semantics"],
        "candidate_modes": first["candidate_modes"],
        "candidate_bisection_steps": first["candidate_bisection_steps"],
        "candidate_binary_sha256": first["candidate_binary_sha256"],
    }
    for name in _COUNTER_FIELDS:
        result[name] = torch.stack([part[name] for part in parts]).sum()
    for name in _MAXIMUM_FIELDS:
        result[name] = torch.stack([part[name] for part in parts]).amax()
    return result


def _finish_outer_step(
    parts: Sequence[Mapping[str, Any]],
    *,
    runtime: CandidateRuntime,
    direction: str,
    outer_step: int,
    elapsed_started: float,
) -> dict[str, Any]:
    aggregate = _aggregate_phase_health(parts)
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    record: dict[str, Any] = {
        name: int(aggregate[name].item()) for name in _COUNTER_FIELDS
    }
    record.update(
        {name: float(aggregate[name].item()) for name in _MAXIMUM_FIELDS}
    )
    record.update(
        backend=aggregate["backend"],
        candidate_target_semantics=aggregate["candidate_target_semantics"],
        candidate_modes=int(aggregate["candidate_modes"]),
        candidate_bisection_steps=int(aggregate["candidate_bisection_steps"]),
        candidate_binary_sha256=str(aggregate["candidate_binary_sha256"]),
        direction=str(direction),
        outer_step=int(outer_step),
        outer_step_seconds=float(time.perf_counter() - elapsed_started),
    )
    bad = {name: record[name] for name in _MUST_BE_ZERO if record[name] != 0}
    if bad:
        raise CandidatePilotError(f"candidate outer step failed closed: {bad}")
    if (
        not math.isfinite(record["candidate_maximum_bracket_width"])
        or record["candidate_maximum_bracket_width"] < 0.0
        or not math.isfinite(record["maximum_mass_error"])
        or not math.isfinite(record["maximum_pair_total_error"])
        or record["maximum_mass_error"] > _MASS_TOLERANCE
        or record["maximum_pair_total_error"] > _MASS_TOLERANCE
    ):
        raise CandidatePilotError("candidate outer step violated numerical health")
    if record["candidate_active_count"] != record["candidate_approximation_count"]:
        raise CandidatePilotError("not every active candidate lane is marked approximate")
    return record


def finish_candidate_outer_step(
    parts: Sequence[Mapping[str, Any]],
    *,
    runtime: CandidateRuntime,
    direction: str,
    outer_step: int,
    elapsed_started: float,
) -> dict[str, Any]:
    """Materialize and validate one outer step after deferred phase work."""

    return _finish_outer_step(
        parts,
        runtime=runtime,
        direction=direction,
        outer_step=outer_step,
        elapsed_started=elapsed_started,
    )


def _unit_mass_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != core.STATE_SIZE or not rows.shape[0]:
        raise CandidatePilotError(f"{name} must have shape [P,784]")
    if not np.isfinite(rows).all() or np.any(rows < 0.0):
        raise CandidatePilotError(f"{name} must be finite and nonnegative")
    if np.any(np.abs(rows.sum(1) - 1.0) > _MASS_TOLERANCE):
        raise CandidatePilotError(f"{name} must be on the simplex")
    return np.ascontiguousarray(rows)


def _forward_records_one_cohort(
    initial_states: np.ndarray,
    labels: np.ndarray,
    path_ids: Sequence[int],
    *,
    root_seed: int,
    runtime: CandidateRuntime,
    sample_steps: int,
    record_outer_steps: Sequence[int],
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None,
) -> core.ForwardRecordDataset:
    steps = _require_sample_steps(sample_steps)
    rows = _unit_mass_rows(initial_states, name="initial states")
    label_values = np.asarray(labels, dtype=np.int64).reshape(-1)
    ids = _path_ids(path_ids, rows.shape[0])
    if label_values.size != len(ids):
        raise CandidatePilotError("initial states, labels, and path IDs are misaligned")
    selected = tuple(int(value) for value in record_outer_steps)
    if len(set(selected)) != len(selected) or any(not 0 <= value < steps for value in selected):
        raise CandidatePilotError("candidate record steps are invalid")
    state = torch.as_tensor(rows, dtype=torch.float64, device=runtime.device)
    outputs: list[tuple[np.ndarray, float, int, int, float, int, np.ndarray, int, int]] = []
    for outer_step in range(steps):
        started = time.perf_counter()
        health_parts: list[Mapping[str, Any]] = []
        pending: list[
            tuple[torch.Tensor, float, int, int, float, int, torch.Tensor, int, int]
        ] = []
        for phase in range(len(PHASE_MATCHINGS)):
            state, target, health = candidate_forward_phase(
                state,
                ids,
                outer_step=outer_step,
                phase=phase,
                root_seed=root_seed,
                sample_steps=steps,
                runtime=runtime,
            )
            health_parts.append(health)
            if outer_step in selected:
                quartile = selected.index(outer_step)
                for row, path_id in enumerate(ids):
                    if phase == (row + quartile) % len(PHASE_MATCHINGS):
                        pending.append(
                            (
                                state[row].detach().clone(),
                                1.0
                                - (len(PHASE_MATCHINGS) * outer_step + phase + 1)
                                / (len(PHASE_MATCHINGS) * steps),
                                phase,
                                int(PHASE_MATCHINGS[phase]),
                                float(PHASE_DURATIONS[phase]),
                                int(label_values[row]),
                                target[row].detach().clone(),
                                int(path_id),
                                outer_step,
                            )
                        )
        record = _finish_outer_step(
            health_parts,
            runtime=runtime,
            direction="forward-record",
            outer_step=outer_step,
            elapsed_started=started,
        )
        outputs.extend(
            (
                later.detach().cpu().numpy().astype(np.float32),
                reverse_time,
                phase,
                color,
                duration,
                label,
                target.detach().cpu().numpy().astype(np.float32),
                path_id,
                selected_outer_step,
            )
            for (
                later,
                reverse_time,
                phase,
                color,
                duration,
                label,
                target,
                path_id,
                selected_outer_step,
            ) in pending
        )
        record["outer_step_seconds"] = float(time.perf_counter() - started)
        if outer_step_callback is not None:
            outer_step_callback(record)
    if len(outputs) != len(ids) * len(selected):
        raise CandidatePilotError("candidate forward record selection is incomplete")
    return core.ForwardRecordDataset(
        later_states=np.stack([row[0] for row in outputs]),
        reverse_time=np.asarray([row[1] for row in outputs], dtype=np.float32),
        phase=np.asarray([row[2] for row in outputs], dtype=np.int64),
        color=np.asarray([row[3] for row in outputs], dtype=np.int64),
        duration=np.asarray([row[4] for row in outputs], dtype=np.float32),
        labels=np.asarray([row[5] for row in outputs], dtype=np.int64),
        targets=np.stack([row[6] for row in outputs]),
        path_ids=np.asarray([row[7] for row in outputs], dtype=np.int64),
        outer_steps=np.asarray([row[8] for row in outputs], dtype=np.int64),
    )


def iter_forward_record_batches_candidate(
    initial_states: np.ndarray,
    labels: np.ndarray,
    path_ids: Sequence[int],
    *,
    root_seed: int,
    runtime: CandidateRuntime,
    sample_steps: int = 128,
    record_outer_steps: Sequence[int] = (15, 47, 79, 111),
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Iterator[core.ForwardRecordDataset]:
    """Yield bounded forward-record cohorts with the requested records per path."""

    rows = np.asarray(initial_states)
    label_values = np.asarray(labels).reshape(-1)
    ids = tuple(int(value) for value in path_ids)
    if rows.ndim != 2 or rows.shape[0] != label_values.size or rows.shape[0] != len(ids):
        raise CandidatePilotError("initial states, labels, and path IDs are misaligned")
    for start in range(0, len(ids), _MAX_PATHS):
        yield _forward_records_one_cohort(
            rows[start : start + _MAX_PATHS],
            label_values[start : start + _MAX_PATHS],
            ids[start : start + _MAX_PATHS],
            root_seed=root_seed,
            runtime=runtime,
            sample_steps=sample_steps,
            record_outer_steps=record_outer_steps,
            outer_step_callback=outer_step_callback,
        )


def forward_terminal_states_candidate(
    initial_states: np.ndarray,
    path_ids: Sequence[int],
    *,
    root_seed: int,
    runtime: CandidateRuntime,
    sample_steps: int = 128,
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Run one bounded candidate forward cohort to its configured terminal state."""

    steps = _require_sample_steps(sample_steps)
    rows = _unit_mass_rows(initial_states, name="forward starts")
    ids = _path_ids(path_ids, rows.shape[0])
    state = torch.as_tensor(rows, dtype=torch.float64, device=runtime.device)
    total_counts = {name: 0 for name in _COUNTER_FIELDS}
    maxima = {name: 0.0 for name in _MAXIMUM_FIELDS}
    durations: list[float] = []
    for outer_step in range(steps):
        started = time.perf_counter()
        health_parts: list[Mapping[str, Any]] = []
        for phase in range(len(PHASE_MATCHINGS)):
            state, _, health = candidate_forward_phase(
                state,
                ids,
                outer_step=outer_step,
                phase=phase,
                root_seed=root_seed,
                sample_steps=steps,
                runtime=runtime,
            )
            health_parts.append(health)
        record = _finish_outer_step(
            health_parts,
            runtime=runtime,
            direction="forward-terminal",
            outer_step=outer_step,
            elapsed_started=started,
        )
        durations.append(float(record["outer_step_seconds"]))
        for name in _COUNTER_FIELDS:
            total_counts[name] += int(record[name])
        for name in _MAXIMUM_FIELDS:
            maxima[name] = max(maxima[name], float(record[name]))
        if outer_step_callback is not None:
            outer_step_callback(record)
    telemetry: dict[str, Any] = {
        "backend": CANDIDATE_BACKEND_NAME,
        "candidate_target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "candidate_modes": 128,
        "candidate_bisection_steps": 56,
        "candidate_binary_sha256": runtime.candidate_binary_sha256,
        "outer_steps": steps,
        "outer_step_seconds": durations,
        **total_counts,
        **maxima,
    }
    return (
        np.ascontiguousarray(state.detach().cpu().numpy(), dtype=np.float64),
        telemetry,
    )


def _reverse_candidate_half(
    state: torch.Tensor,
    pair: torch.Tensor,
    ids: tuple[int, ...],
    *,
    outer_step: int,
    phase: int,
    micro: int,
    side: str,
    exposure: torch.Tensor,
    sample_steps: int,
    root_seed: int,
    runtime: CandidateRuntime,
    tails: torch.Tensor,
    heads: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    if int(state.shape[0]) * EDGES_PER_PHASE > _MAX_LANES:
        raise CandidatePilotError("candidate enqueue exceeds the 4096-lane cap")
    fraction = torch.zeros_like(pair)
    active = pair > 0.0
    fraction[active] = state[:, heads][active] / pair[active]
    transition_ids = canonical_refinement_transition_ids(
        ids,
        sample_steps=sample_steps,
        outer_step=outer_step,
        phase=phase,
        device=state.device,
    ).reshape_as(fraction)
    key = (int(root_seed), "reverse", int(micro), str(side))
    batch = enqueue_alpha1_rb_transition_batch_cuda_candidate(
        fraction,
        exposure,
        rng_key=key,
        transition_ids=transition_ids,
        prepared=runtime.prepared,
        prepared_rng_seed=_prepared_seed(runtime, key),
    )
    if not isinstance(batch, CandidateRBCudaBatch):
        raise CandidatePilotError("candidate dispatch returned a non-candidate batch")
    next_fraction = batch.later_head_fraction.to(dtype=torch.float64)
    output = state.clone()
    output[:, heads] = pair * next_fraction
    output[:, tails] = pair * (1.0 - next_fraction)
    health = _phase_health(
        state_before=state,
        state_after=output,
        pair_before=pair,
        tails=tails,
        heads=heads,
        batch=batch,
        runtime=runtime,
    )
    return output, next_fraction - fraction, health


def candidate_reverse_outer_step(
    state: torch.Tensor,
    labels: torch.Tensor,
    path_ids: Sequence[int],
    *,
    outer_step: int,
    controller: Literal["null", "learned", "oracle"],
    root_seed: int,
    runtime: CandidateRuntime,
    model: core.EulerianJacobiDDPMModel | None = None,
    oracle_targets: torch.Tensor | None = None,
    sample_steps: int = _DEFAULT_SAMPLE_STEPS,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Apply one complete reverse outer step and materialize health once."""

    _require_state_static(state, runtime)
    ids = _path_ids(path_ids, int(state.shape[0]))
    steps = _require_sample_steps(sample_steps)
    step = int(outer_step)
    if not 0 <= step < steps:
        raise CandidatePilotError("reverse outer step is outside the configured chain")
    if controller not in {"null", "learned", "oracle"}:
        raise CandidatePilotError("controller must be null, learned, or oracle")
    if labels.device != state.device or labels.ndim != 1 or labels.shape[0] != state.shape[0]:
        raise CandidatePilotError("labels must be a device-aligned vector")
    if controller == "learned" and model is None:
        raise CandidatePilotError("learned controller requires a model")
    if controller == "oracle":
        if (
            oracle_targets is None
            or oracle_targets.dtype != torch.float64
            or oracle_targets.shape != state.shape
            or oracle_targets.device != state.device
        ):
            raise CandidatePilotError("oracle targets must match the reverse state")

    started = time.perf_counter()
    health_parts: list[Mapping[str, Any]] = []
    tails_all, heads_all = matching_indices(device=state.device)
    quarter = min(3, (4 * step) // steps)
    score_squares = torch.zeros((), dtype=torch.float64, device=state.device)
    score_count = 0
    maximum_score = torch.zeros((), dtype=torch.float64, device=state.device)
    reference_squares = torch.zeros((), dtype=torch.float64, device=state.device)
    reference_count = 0
    control_squares = torch.zeros((), dtype=torch.float64, device=state.device)
    control_count = 0
    maximum_logit_increment = torch.zeros((), dtype=torch.float64, device=state.device)

    for phase in range(len(PHASE_MATCHINGS) - 1, -1, -1):
        color = int(PHASE_MATCHINGS[phase])
        tails, heads = tails_all[color], heads_all[color]
        pair = state[:, tails] + state[:, heads]
        full_exposure = refinement_phase_exposure(
            pair,
            sample_steps=steps,
            duration_fraction=float(PHASE_DURATIONS[phase]),
        )
        delta = full_exposure / core.CONTROLLER_MICROSTEPS
        for micro in range(core.CONTROLLER_MICROSTEPS):
            for side in ("pre", "post"):
                state, reference_displacement, health = _reverse_candidate_half(
                    state,
                    pair,
                    ids,
                    outer_step=step,
                    phase=phase,
                    micro=micro,
                    side=side,
                    exposure=delta / 2.0,
                    sample_steps=steps,
                    root_seed=root_seed,
                    runtime=runtime,
                    tails=tails,
                    heads=heads,
                )
                reference_squares = reference_squares + reference_displacement.square().sum()
                reference_count += int(reference_displacement.numel())
                if side != "pre":
                    health_parts.append(health)
                    continue
                if controller == "null":
                    score = torch.zeros_like(pair)
                elif controller == "learned":
                    assert model is not None
                    reverse_time = core.reverse_midpoint_time(
                        step, phase, micro, sample_steps=steps
                    )
                    inputs = ModelInputs(
                        later_full_state=state.to(torch.float32),
                        reverse_time=torch.full(
                            (len(ids),), reverse_time, dtype=torch.float32, device=state.device
                        ),
                        phase=torch.full(
                            (len(ids),), phase, dtype=torch.long, device=state.device
                        ),
                        color=torch.full(
                            (len(ids),), color, dtype=torch.long, device=state.device
                        ),
                        duration=torch.full(
                            (len(ids),),
                            float(PHASE_DURATIONS[phase]),
                            dtype=torch.float32,
                            device=state.device,
                        ),
                        label=labels,
                    )
                    with torch.no_grad():
                        score = model.predictor.score_prediction_prevalidated(inputs).to(
                            torch.float64
                        )
                else:
                    assert oracle_targets is not None
                    current = torch.zeros_like(pair)
                    active = pair > 0.0
                    current[active] = state[:, heads][active] / pair[active]
                    target_pair = oracle_targets[:, tails] + oracle_targets[:, heads]
                    target_fraction = torch.zeros_like(pair)
                    reachable = active & (target_pair > 0.0)
                    target_fraction[reachable] = (
                        oracle_targets[:, heads][reachable] / target_pair[reachable]
                    )
                    interior = (
                        reachable
                        & (current > 0.0)
                        & (current < 1.0)
                        & (target_fraction > 0.0)
                        & (target_fraction < 1.0)
                        & (delta > 0.0)
                    )
                    score = torch.zeros_like(pair)
                    score[interior] = (
                        torch.logit(target_fraction[interior])
                        - torch.logit(current[interior])
                    ) / (2.0 * delta[interior])
                health = dict(health)
                health["controller_nonfinite_count"] = (
                    ~torch.isfinite(score)
                ).sum(dtype=torch.int64)
                before_control = torch.zeros_like(pair)
                active = pair > 0.0
                before_control[active] = state[:, heads][active] / pair[active]
                state = _score_logistic_flow_prevalidated(
                    state, tails, heads, score, delta
                )
                after_control = torch.zeros_like(pair)
                after_control[active] = state[:, heads][active] / pair[active]
                displacement = after_control - before_control
                score_squares = score_squares + score.square().sum()
                score_count += int(score.numel())
                maximum_score = torch.maximum(maximum_score, torch.amax(torch.abs(score)))
                control_squares = control_squares + displacement.square().sum()
                control_count += int(displacement.numel())
                maximum_logit_increment = torch.maximum(
                    maximum_logit_increment, torch.amax(torch.abs(2.0 * score * delta))
                )
                health_parts.append(health)

    record = _finish_outer_step(
        health_parts,
        runtime=runtime,
        direction="reverse",
        outer_step=step,
        elapsed_started=started,
    )
    score_square_sum = float(score_squares.item())
    reference_square_sum = float(reference_squares.item())
    control_square_sum = float(control_squares.item())
    record.update(
        controller=controller,
        time_quarter=quarter,
        score_count=score_count,
        score_square_sum=score_square_sum,
        controller_rms=math.sqrt(score_square_sum / score_count) if score_count else 0.0,
        reference_count=reference_count,
        reference_square_sum=reference_square_sum,
        reference_fraction_displacement_rms=(
            math.sqrt(reference_square_sum / reference_count) if reference_count else 0.0
        ),
        control_count=control_count,
        control_square_sum=control_square_sum,
        control_fraction_displacement_rms=(
            math.sqrt(control_square_sum / control_count) if control_count else 0.0
        ),
        maximum_absolute_q=float(maximum_score.item()),
        maximum_absolute_logit_increment=float(maximum_logit_increment.item()),
    )
    return state, record


def reverse_sample_candidate(
    starts: np.ndarray,
    labels: np.ndarray,
    path_ids: Sequence[int],
    *,
    controller: Literal["null", "learned", "oracle"],
    root_seed: int,
    runtime: CandidateRuntime,
    model: core.EulerianJacobiDDPMModel | None = None,
    oracle_targets: np.ndarray | None = None,
    anchors: Sequence[int] = (0, 32, 64, 96, 128),
    sample_steps: int = 128,
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> core.SamplingResult:
    """Run the immutable reverse composition with candidate reference halves."""

    steps = _require_sample_steps(sample_steps)
    rows = _unit_mass_rows(starts, name="reverse starts")
    label_values = np.asarray(labels, dtype=np.int64).reshape(-1)
    ids = _path_ids(path_ids, rows.shape[0])
    if label_values.size != len(ids):
        raise CandidatePilotError("reverse starts, labels, and path IDs are misaligned")
    target_rows = None
    if controller == "oracle":
        if oracle_targets is None:
            raise CandidatePilotError("oracle targets are required")
        target_rows = _unit_mass_rows(oracle_targets, name="oracle targets")
        if target_rows.shape != rows.shape:
            raise CandidatePilotError("oracle target shape changed")
    if controller == "learned" and model is None:
        raise CandidatePilotError("learned controller requires a model")
    anchor_set = {int(value) for value in anchors}
    if 0 not in anchor_set or steps not in anchor_set or any(
        not 0 <= value <= steps for value in anchor_set
    ):
        raise CandidatePilotError("anchors must lie in the configured interval and include both ends")

    state = torch.as_tensor(rows, dtype=torch.float64, device=runtime.device)
    labels_tensor = torch.as_tensor(label_values, dtype=torch.long, device=runtime.device)
    target_tensor = (
        torch.as_tensor(target_rows, dtype=torch.float64, device=runtime.device)
        if target_rows is not None
        else None
    )
    if model is not None:
        model = model.to(runtime.device).eval()
    saved: dict[int, np.ndarray] = {0: rows.copy()}
    step_records: list[Mapping[str, Any]] = []
    completed = 0
    for outer_step in range(steps - 1, -1, -1):
        state, record = candidate_reverse_outer_step(
            state,
            labels_tensor,
            ids,
            outer_step=outer_step,
            controller=controller,
            root_seed=root_seed,
            runtime=runtime,
            model=model,
            oracle_targets=target_tensor,
            sample_steps=steps,
        )
        completed += 1
        if completed in anchor_set:
            saved[completed] = state.detach().cpu().numpy().copy()
        step_records.append(record)
        if outer_step_callback is not None:
            outer_step_callback(record)

    final = np.ascontiguousarray(state.detach().cpu().numpy(), dtype=np.float64)
    by_time_quarter: list[dict[str, Any]] = []
    for quarter in range(4):
        selected = [row for row in step_records if int(row["time_quarter"]) == quarter]
        score_count = sum(int(row["score_count"]) for row in selected)
        reference_count = sum(int(row["reference_count"]) for row in selected)
        control_count = sum(int(row["control_count"]) for row in selected)
        score_squares = sum(float(row["score_square_sum"]) for row in selected)
        reference_squares = sum(float(row["reference_square_sum"]) for row in selected)
        control_squares = sum(float(row["control_square_sum"]) for row in selected)
        by_time_quarter.append(
            {
                "quarter": quarter,
                "time_quarter": quarter,
                "score_count": score_count,
                "score_rms": math.sqrt(score_squares / score_count) if score_count else 0.0,
                "controller_rms": math.sqrt(score_squares / score_count) if score_count else 0.0,
                "reference_fraction_displacement_rms": (
                    math.sqrt(reference_squares / reference_count) if reference_count else 0.0
                ),
                "control_fraction_displacement_rms": (
                    math.sqrt(control_squares / control_count) if control_count else 0.0
                ),
                "maximum_absolute_q": max(
                    (float(row["maximum_absolute_q"]) for row in selected), default=0.0
                ),
                "maximum_absolute_logit_increment": max(
                    (
                        float(row["maximum_absolute_logit_increment"])
                        for row in selected
                    ),
                    default=0.0,
                ),
                "maximum_mass_error": max(
                    (float(row["maximum_mass_error"]) for row in selected), default=0.0
                ),
                "maximum_pair_total_error": max(
                    (float(row["maximum_pair_total_error"]) for row in selected),
                    default=0.0,
                ),
            }
        )

    score_count = sum(int(row["score_count"]) for row in step_records)
    score_squares = sum(float(row["score_square_sum"]) for row in step_records)
    telemetry: dict[str, Any] = {
        "controller": controller,
        "controller_rms": math.sqrt(score_squares / score_count) if score_count else 0.0,
        "maximum_absolute_q": max(
            (float(row["maximum_absolute_q"]) for row in step_records), default=0.0
        ),
        "maximum_mass_error": max(
            (float(row["maximum_mass_error"]) for row in step_records), default=0.0
        ),
        "maximum_pair_total_error": max(
            (float(row["maximum_pair_total_error"]) for row in step_records), default=0.0
        ),
        "exact_facet_count": int(np.count_nonzero(final == 0.0)),
        "finite": int(np.isfinite(final).all()),
        "nonnegative": int(np.all(final >= 0.0)),
        "microsteps": core.CONTROLLER_MICROSTEPS,
        "by_time_quarter": by_time_quarter,
        "backend": CANDIDATE_BACKEND_NAME,
        "candidate_target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "candidate_modes": 128,
        "candidate_bisection_steps": 56,
        "candidate_binary_sha256": runtime.candidate_binary_sha256,
        "outer_step_seconds": [float(row["outer_step_seconds"]) for row in step_records],
    }
    for name in _COUNTER_FIELDS:
        telemetry[name] = sum(int(row[name]) for row in step_records)
    telemetry["candidate_maximum_bracket_width"] = max(
        (float(row["candidate_maximum_bracket_width"]) for row in step_records),
        default=0.0,
    )
    return core.SamplingResult(
        starts=np.ascontiguousarray(rows),
        final_states=final,
        anchors=saved,
        telemetry=telemetry,
    )


def prior_sample_candidate(
    path_ids: Sequence[int],
    labels: np.ndarray,
    *,
    start_seed: int,
    controller: Literal["null", "learned", "oracle"],
    root_seed: int,
    runtime: CandidateRuntime,
    model: core.EulerianJacobiDDPMModel | None = None,
    oracle_targets: np.ndarray | None = None,
    anchors: Sequence[int] = (0, 32, 64, 96, 128),
    sample_steps: int = 128,
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> core.SamplingResult:
    starts = core.sample_dirichlet_starts(path_ids, root_seed=int(start_seed))
    return reverse_sample_candidate(
        starts,
        labels,
        path_ids,
        controller=controller,
        root_seed=root_seed,
        runtime=runtime,
        model=model,
        oracle_targets=oracle_targets,
        anchors=anchors,
        sample_steps=sample_steps,
        outer_step_callback=outer_step_callback,
    )


def forward_terminal_sample_candidate(
    initial_states: np.ndarray,
    labels: np.ndarray,
    path_ids: Sequence[int],
    *,
    forward_seed: int,
    controller: Literal["null", "learned", "oracle"],
    root_seed: int,
    runtime: CandidateRuntime,
    model: core.EulerianJacobiDDPMModel | None = None,
    oracle_targets: np.ndarray | None = None,
    anchors: Sequence[int] = (0, 32, 64, 96, 128),
    sample_steps: int = 128,
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> core.SamplingResult:
    terminal, _ = forward_terminal_states_candidate(
        initial_states,
        path_ids,
        root_seed=forward_seed,
        runtime=runtime,
        sample_steps=sample_steps,
        outer_step_callback=outer_step_callback,
    )
    return reverse_sample_candidate(
        terminal,
        labels,
        path_ids,
        controller=controller,
        root_seed=root_seed,
        runtime=runtime,
        model=model,
        oracle_targets=oracle_targets,
        anchors=anchors,
        sample_steps=sample_steps,
        outer_step_callback=outer_step_callback,
    )


def oracle_sample_candidate(
    starts: np.ndarray,
    labels: np.ndarray,
    path_ids: Sequence[int],
    targets: np.ndarray,
    *,
    root_seed: int,
    runtime: CandidateRuntime,
    anchors: Sequence[int] = (0, 32, 64, 96, 128),
    sample_steps: int = 128,
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> core.SamplingResult:
    return reverse_sample_candidate(
        starts,
        labels,
        path_ids,
        controller="oracle",
        root_seed=root_seed,
        runtime=runtime,
        oracle_targets=targets,
        anchors=anchors,
        sample_steps=sample_steps,
        outer_step_callback=outer_step_callback,
    )


__all__ = [
    "CANDIDATE_BACKEND_NAME",
    "CANDIDATE_PILOT_VERSION",
    "CANDIDATE_TARGET_SEMANTICS",
    "CandidatePilotError",
    "CandidateRuntime",
    "candidate_forward_phase",
    "candidate_forward_phase_prefixes",
    "candidate_reverse_outer_step",
    "finish_candidate_outer_step",
    "forward_terminal_sample_candidate",
    "forward_terminal_states_candidate",
    "iter_forward_record_batches_candidate",
    "oracle_sample_candidate",
    "prepare_candidate_runtime",
    "prior_sample_candidate",
    "reverse_sample_candidate",
]
