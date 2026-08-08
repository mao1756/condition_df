"""Exact fused-lane scheduling primitives for the boundary-tangent pilot.

This module changes execution packing only.  Canonical Jacobi transitions keep
their existing state-dependent exposures, stateless transition identifiers,
and certified sampler.  Midpoint branches are flattened in the frozen
``(midpoint, path, edge)`` order and split into contiguous launches of at most
4,096 lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    BOUNDARY_TANGENT_CACHE_VERSION,
    FORBIDDEN_DIAGNOSTICS,
    MIDPOINT_COUNT,
    MIDPOINT_FRACTIONS,
    MidpointBranchBatch,
    phase_base_exposure,
)
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_MATCHINGS,
    STATE_SIZE,
    matching_indices,
)
from mnist.d0_jacobi_rb_reverse_controller import controller_transition_ids


SCHEDULE_VERSION = "d0-jacobi-rb-boundary-tangent-fused-schedule-v1"
ROOT_SEED = 261_321
MAXIMUM_LAUNCH_LANES = 4_096
SHARD_STEPS = 8
WINDOW_START_STEPS = (0, 128, 256, 384)
WINDOW_STEP_COUNT = 16
WINDOW_OUTER_STEP_COUNT = len(WINDOW_START_STEPS) * WINDOW_STEP_COUNT
PILOT_REPEAT_COUNT = 3

TRAIN_PATH_IDS = tuple(range(0xEC100, 0xEC140))
VALIDATION_PATH_IDS = tuple(range(0xEC200, 0xEC220))
CONFIRMATION_PATH_IDS = tuple(range(0xED000, 0xED040))
TRAIN_VALIDATION_COHORT_SIZES = (10,) * 9 + (6,)
CONFIRMATION_COHORT_SIZES = (10,) * 6 + (4,)

PROFILE_CACHE_P10 = "cache_p10"
PROFILE_CACHE_P6 = "cache_p6"
PROFILE_STREAM_P10 = "stream_p10"
PROFILE_STREAM_P4 = "stream_p4"
PILOT_PROFILE_NAMES = (
    PROFILE_CACHE_P10,
    PROFILE_CACHE_P6,
    PROFILE_STREAM_P10,
    PROFILE_STREAM_P4,
)
PROFILE_PATH_COUNTS = {
    PROFILE_CACHE_P10: 10,
    PROFILE_CACHE_P6: 6,
    PROFILE_STREAM_P10: 10,
    PROFILE_STREAM_P4: 4,
}
PROFILE_PROJECTION_MULTIPLICITIES = {
    PROFILE_CACHE_P10: 9,
    PROFILE_CACHE_P6: 1,
    PROFILE_STREAM_P10: 6,
    PROFILE_STREAM_P4: 1,
}
PROFILE_PATH_IDS = {
    PROFILE_CACHE_P10: tuple(range(0xEE000, 0xEE00A)),
    PROFILE_CACHE_P6: tuple(range(0xEE010, 0xEE016)),
    PROFILE_STREAM_P10: tuple(range(0xEE100, 0xEE10A)),
    PROFILE_STREAM_P4: tuple(range(0xEE110, 0xEE114)),
}
WARMUP_PATH_IDS = tuple(range(0xEE200, 0xEE20A))

BASE_TRANSITIONS_PER_PILOT_PATH = (
    WINDOW_OUTER_STEP_COUNT * PHASE_COUNT * EDGES_PER_PHASE
)
MIDPOINT_TRANSITIONS_PER_PILOT_PATH = (
    len(WINDOW_START_STEPS) * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
)
TOTAL_TRANSITIONS_PER_PILOT_PATH = (
    BASE_TRANSITIONS_PER_PILOT_PATH + MIDPOINT_TRANSITIONS_PER_PILOT_PATH
)
PROJECTED_BASE_TRANSITIONS = 224_788_480
PROJECTED_MIDPOINT_TRANSITIONS = 112_394_240
PROJECTED_TOTAL_TRANSITIONS = 337_182_720
PROJECTION_FACTOR = OUTER_STEPS // WINDOW_OUTER_STEP_COUNT

MAXIMUM_PROJECTED_SECONDS = 30.0 * 60.0 * 60.0
MINIMUM_EFFECTIVE_PROJECTED_RATE = (
    PROJECTED_TOTAL_TRANSITIONS / MAXIMUM_PROJECTED_SECONDS
)
MINIMUM_PROFILE_RATE = 1_300.0
MAXIMUM_FALLBACK_FRACTION = 1.0e-4
MAXIMUM_FALLBACK_TIME_FRACTION = 0.10
MAXIMUM_MASS_ERROR = 2.0e-12
MAXIMUM_MEMORY_FRACTION = 0.80
MAXIMUM_PERSISTENCE_BYTES = int(1.25 * (1024**3))


class BoundaryTangentScheduleError(ValueError):
    """Raised when the frozen fused scheduling contract is violated."""


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise BoundaryTangentScheduleError(f"{name} must be integral")
    return int(value)


def _sha256(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64:
        raise BoundaryTangentScheduleError(f"{name} is not a SHA-256 digest")
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise BoundaryTangentScheduleError(
            f"{name} is not a SHA-256 digest"
        ) from exc
    return text.lower()


def frozen_repeat_order(repeat_index: int) -> tuple[str, ...]:
    """Return the cyclic profile order for one deterministic pilot repeat."""

    repeat = _integer(repeat_index, "repeat_index")
    if not 0 <= repeat < PILOT_REPEAT_COUNT:
        raise BoundaryTangentScheduleError("repeat index lies outside [0,3)")
    shift = repeat % len(PILOT_PROFILE_NAMES)
    return PILOT_PROFILE_NAMES[shift:] + PILOT_PROFILE_NAMES[:shift]


def frozen_path_plan() -> dict[str, Any]:
    """Validate and return the collision-free benchmark namespace plan."""

    roles = {**PROFILE_PATH_IDS, "warmup": WARMUP_PATH_IDS}
    seen: set[int] = set()
    for role, values in roles.items():
        if len(values) != len(set(values)) or any(
            not 0 <= value < (1 << 20) for value in values
        ):
            raise BoundaryTangentScheduleError(f"{role} path IDs are malformed")
        if seen.intersection(values):
            raise BoundaryTangentScheduleError(f"{role} path IDs collide")
        seen.update(values)
    return {
        "schema": SCHEDULE_VERSION + "-path-plan",
        "root_seed": ROOT_SEED,
        "roles": {key: list(value) for key, value in roles.items()},
        "collision_free": 1,
    }


def _partition(values: Sequence[int], sizes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    canonical = tuple(_integer(value, "path_id") for value in values)
    groups = tuple(_integer(size, "cohort_size") for size in sizes)
    if any(size < 1 or size > 10 for size in groups) or sum(groups) != len(canonical):
        raise BoundaryTangentScheduleError("cohort sizes do not partition the paths")
    output: list[tuple[int, ...]] = []
    cursor = 0
    for size in groups:
        output.append(canonical[cursor : cursor + size])
        cursor += size
    return tuple(output)


def frozen_production_cohort_plan() -> dict[str, Any]:
    """Return the exact future 96-path and 64-path cohort decomposition."""

    train_validation = TRAIN_PATH_IDS + VALIDATION_PATH_IDS
    if (
        len(set(train_validation + CONFIRMATION_PATH_IDS))
        != len(train_validation) + len(CONFIRMATION_PATH_IDS)
    ):
        raise BoundaryTangentScheduleError("production evidence path roles collide")
    tv_groups = _partition(train_validation, TRAIN_VALIDATION_COHORT_SIZES)
    confirmation_groups = _partition(
        CONFIRMATION_PATH_IDS, CONFIRMATION_COHORT_SIZES
    )
    # A cohort may share a CUDA call; its artifact rows retain their immutable
    # path role.  This map is the firewall consumed by downstream commit code.
    path_roles = {
        **{str(path): "train" for path in TRAIN_PATH_IDS},
        **{str(path): "validation" for path in VALIDATION_PATH_IDS},
        **{str(path): "confirmation" for path in CONFIRMATION_PATH_IDS},
    }
    return {
        "schema": SCHEDULE_VERSION + "-production-cohorts",
        "train_validation": [list(group) for group in tv_groups],
        "confirmation": [list(group) for group in confirmation_groups],
        "train_validation_sizes": list(TRAIN_VALIDATION_COHORT_SIZES),
        "confirmation_sizes": list(CONFIRMATION_COHORT_SIZES),
        "path_roles": path_roles,
        "cross_role_artifact_commit": 0,
    }


def split_co_scheduled_payload_by_role(
    path_ids: Sequence[int], payload: Mapping[str, Tensor | np.ndarray]
) -> dict[str, dict[str, Tensor | np.ndarray]]:
    """Physically separate co-scheduled evidence before artifact commit.

    The CUDA scheduler may share a call across the train/validation boundary,
    but no downstream payload can retain a mixed leading path dimension.
    Returned arrays are new contiguous values grouped by immutable path role.
    """

    role_by_path = {
        **{path: "train" for path in TRAIN_PATH_IDS},
        **{path: "validation" for path in VALIDATION_PATH_IDS},
        **{path: "confirmation" for path in CONFIRMATION_PATH_IDS},
    }
    paths = tuple(_integer(value, "path_id") for value in path_ids)
    if any(path not in role_by_path for path in paths):
        raise BoundaryTangentScheduleError("co-scheduled payload contains an unknown path")
    return split_payload_by_path_roles(
        paths,
        tuple(role_by_path[path] for path in paths),
        payload,
    )


def split_payload_by_path_roles(
    path_ids: Sequence[int],
    path_roles: Sequence[str],
    payload: Mapping[str, Tensor | np.ndarray],
) -> dict[str, dict[str, Tensor | np.ndarray]]:
    """Physically split a path-major payload using an explicit role map.

    This is the generic form used by additive workflows with fresh path-ID
    allocations.  The historical wrapper above deliberately retains the
    frozen v2 role lookup.
    """

    paths = tuple(_integer(value, "path_id") for value in path_ids)
    roles = tuple(str(value) for value in path_roles)
    if not paths or len(paths) != len(set(paths)):
        raise BoundaryTangentScheduleError("co-scheduled path IDs are empty/duplicated")
    if len(roles) != len(paths) or any(not role for role in roles):
        raise BoundaryTangentScheduleError("path roles do not align with path IDs")
    indices: dict[str, list[int]] = {}
    for index, role in enumerate(roles):
        indices.setdefault(role, []).append(index)
    output: dict[str, dict[str, Tensor | np.ndarray]] = {
        role: {} for role in indices
    }
    for name, value in payload.items():
        if isinstance(value, Tensor):
            if value.ndim < 1 or int(value.shape[0]) != len(paths):
                raise BoundaryTangentScheduleError(
                    f"co-scheduled tensor {name} has wrong leading dimension"
                )
            for role, positions in indices.items():
                selector = torch.as_tensor(
                    positions, dtype=torch.int64, device=value.device
                )
                output[role][name] = value.index_select(0, selector).contiguous()
        elif isinstance(value, np.ndarray):
            if value.ndim < 1 or int(value.shape[0]) != len(paths):
                raise BoundaryTangentScheduleError(
                    f"co-scheduled array {name} has wrong leading dimension"
                )
            for role, positions in indices.items():
                output[role][name] = np.ascontiguousarray(value[positions]).copy()
        else:
            raise BoundaryTangentScheduleError(
                f"co-scheduled payload {name} is not a tensor/array"
            )
    return output


@dataclass(frozen=True)
class FusedLaunchPlan:
    """Contiguous canonical-lane partition for one midpoint branch call."""

    path_count: int
    total_lanes: int
    chunk_ranges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        paths = _integer(self.path_count, "path_count")
        total = _integer(self.total_lanes, "total_lanes")
        expected = MIDPOINT_COUNT * paths * EDGES_PER_PHASE
        if paths < 1 or paths > 10 or total != expected:
            raise BoundaryTangentScheduleError("fused launch dimensions are invalid")
        cursor = 0
        for raw_start, raw_stop in self.chunk_ranges:
            start = _integer(raw_start, "chunk_start")
            stop = _integer(raw_stop, "chunk_stop")
            if start != cursor or stop <= start or stop - start > MAXIMUM_LAUNCH_LANES:
                raise BoundaryTangentScheduleError(
                    "fused launch chunks are not a contiguous capped partition"
                )
            cursor = stop
        if cursor != total:
            raise BoundaryTangentScheduleError("fused launch chunks are incomplete")

    @property
    def maximum_chunk_lanes(self) -> int:
        return max(stop - start for start, stop in self.chunk_ranges)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULE_VERSION + "-fused-launch-plan",
            "canonical_order": ["midpoint", "path", "edge"],
            "path_count": self.path_count,
            "midpoint_count": MIDPOINT_COUNT,
            "edges_per_phase": EDGES_PER_PHASE,
            "total_lanes": self.total_lanes,
            "maximum_launch_lanes": MAXIMUM_LAUNCH_LANES,
            "maximum_observed_chunk_lanes": self.maximum_chunk_lanes,
            "chunk_ranges": [list(value) for value in self.chunk_ranges],
        }


def build_fused_launch_plan(path_count: int) -> FusedLaunchPlan:
    paths = _integer(path_count, "path_count")
    if not 1 <= paths <= 10:
        raise BoundaryTangentScheduleError("fused branch cohorts contain 1..10 paths")
    total = MIDPOINT_COUNT * paths * EDGES_PER_PHASE
    chunks = tuple(
        (start, min(total, start + MAXIMUM_LAUNCH_LANES))
        for start in range(0, total, MAXIMUM_LAUNCH_LANES)
    )
    return FusedLaunchPlan(paths, total, chunks)


def _field(result: Any, *names: str) -> Tensor:
    for name in names:
        value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
        if isinstance(value, Tensor):
            return value
    raise BoundaryTangentScheduleError(f"certified sampler omitted {names[0]}")


def _optional_tensor(
    result: Any,
    name: str,
    *,
    lanes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
    if value is None:
        return torch.zeros(lanes, dtype=dtype, device=device)
    if not isinstance(value, Tensor) or value.device != device or value.numel() != lanes:
        raise BoundaryTangentScheduleError(f"sampler field {name} has wrong lanes/device")
    return value.reshape(-1).to(dtype=dtype).contiguous()


def _scalar_diagnostic(result: Any, name: str, default: float = 0.0) -> float:
    diagnostics = (
        result.get("diagnostics")
        if isinstance(result, Mapping)
        else getattr(result, "diagnostics", None)
    )
    value = diagnostics.get(name, default) if isinstance(diagnostics, Mapping) else default
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise BoundaryTangentScheduleError(f"diagnostic {name} is not scalar")
        value = value.detach().cpu().item()
    output = float(value)
    if not math.isfinite(output) or output < 0.0:
        raise BoundaryTangentScheduleError(f"diagnostic {name} is invalid")
    return output


@dataclass(frozen=True)
class FusedMidpointBranchBatch:
    """Fused result with the same scientific tensor layout as the v1 cache."""

    batch: MidpointBranchBatch
    launch_plan: FusedLaunchPlan
    launch_count: int
    fallback_reason_codes: Tensor
    candidate_elapsed_seconds: float
    reported_authorizer_launch_count: int
    reported_maximum_launch_lanes: int

    def __post_init__(self) -> None:
        if len(self.batch.path_ids) != self.launch_plan.path_count:
            raise BoundaryTangentScheduleError("branch batch and launch plan disagree")
        if self.batch.transition_count != self.launch_plan.total_lanes:
            raise BoundaryTangentScheduleError("branch transition count changed")
        if self.launch_count != len(self.launch_plan.chunk_ranges):
            raise BoundaryTangentScheduleError("branch launch count changed")
        expected = (MIDPOINT_COUNT, self.launch_plan.path_count, EDGES_PER_PHASE)
        if (
            not isinstance(self.fallback_reason_codes, Tensor)
            or self.fallback_reason_codes.shape != expected
            or self.fallback_reason_codes.dtype != torch.uint8
            or self.fallback_reason_codes.device != self.batch.denoising_target.device
        ):
            raise BoundaryTangentScheduleError(
                "branch fallback-reason tensor is malformed"
            )
        if (
            not math.isfinite(float(self.candidate_elapsed_seconds))
            or float(self.candidate_elapsed_seconds) < 0.0
            or _integer(
                self.reported_authorizer_launch_count,
                "reported_authorizer_launch_count",
            )
            < self.launch_count
            or not 1
            <= _integer(
                self.reported_maximum_launch_lanes,
                "reported_maximum_launch_lanes",
            )
            <= MAXIMUM_LAUNCH_LANES
        ):
            raise BoundaryTangentScheduleError(
                "branch authorizer diagnostics are malformed"
            )

    @property
    def later_full_state(self) -> Tensor:
        return self.batch.later_full_state

    @property
    def denoising_target(self) -> Tensor:
        return self.batch.denoising_target

    @property
    def certificate_codes(self) -> Tensor:
        return self.batch.certificate_codes

    def output_sha256(self) -> str:
        return self.batch.output_sha256()

    def to_record(self) -> dict[str, Any]:
        def histogram(value: Tensor) -> dict[str, int]:
            unique, counts = np.unique(value.detach().cpu().numpy(), return_counts=True)
            return {
                str(int(item)): int(count)
                for item, count in zip(unique.tolist(), counts.tolist(), strict=True)
            }

        return {
            "schema": SCHEDULE_VERSION + "-fused-midpoint-batch",
            "path_ids": list(self.batch.path_ids),
            "outer_step": self.batch.outer_step,
            "phase": self.batch.phase,
            "transition_count": self.batch.transition_count,
            "certified_count": self.batch.certified_count,
            "launch_count": self.launch_count,
            "reported_authorizer_launch_count": self.reported_authorizer_launch_count,
            "maximum_launch_lanes": self.launch_plan.maximum_chunk_lanes,
            "reported_maximum_launch_lanes": self.reported_maximum_launch_lanes,
            "fallback_count": int(self.batch.fallback_mask.sum().item()),
            "strengthened_count": int(self.batch.strengthened_mask.sum().item()),
            "mode_count_histogram": histogram(self.batch.mode_counts),
            "prefix_bit_histogram": histogram(self.batch.prefix_bits),
            "fallback_reason_histogram": histogram(self.fallback_reason_codes),
            "forbidden_counts": dict(self.batch.forbidden_counts),
            "fallback_elapsed_seconds": self.batch.fallback_elapsed_seconds,
            "backend_elapsed_seconds": self.batch.backend_elapsed_seconds,
            "candidate_elapsed_seconds": self.candidate_elapsed_seconds,
            "output_sha256": self.output_sha256(),
            "target_transformed": 0,
            "approximate_transition_used": 0,
        }


def sample_fused_midpoint_branches(
    pre_phase_states: Tensor,
    *,
    path_ids: Sequence[int],
    outer_step: int,
    phase: int,
    root_seed: int = ROOT_SEED,
    profile: JacobiRBCudaProfile | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> FusedMidpointBranchBatch:
    """Sample exact M8 branches through capped fused CUDA launches.

    Input lanes, exposures, and transition IDs are assembled in exact
    ``(midpoint, path, edge)`` order.  Chunking is a contiguous view of that
    sequence and therefore cannot change any lane's stateless randomness.
    """

    if not isinstance(pre_phase_states, Tensor):
        raise TypeError("pre_phase_states must be a torch.Tensor")
    if (
        pre_phase_states.dtype != torch.float64
        or pre_phase_states.ndim != 2
        or pre_phase_states.shape[1] != STATE_SIZE
        or not pre_phase_states.is_contiguous()
    ):
        raise BoundaryTangentScheduleError(
            "pre-phase states must be contiguous float64 [P,784]"
        )
    if sampler is sample_alpha1_rb_transition_batch_cuda and not pre_phase_states.is_cuda:
        raise BoundaryTangentScheduleError("production fused sampling requires CUDA")
    paths = tuple(_integer(value, "path_id") for value in path_ids)
    if len(paths) != pre_phase_states.shape[0] or len(paths) != len(set(paths)):
        raise BoundaryTangentScheduleError("path IDs must be unique and match states")
    if any(not 0 <= value < (1 << 20) for value in paths):
        raise BoundaryTangentScheduleError("path ID lies outside the 20-bit plan")
    step = _integer(outer_step, "outer_step")
    occurrence = _integer(phase, "phase")
    if not 0 <= step < OUTER_STEPS or not 0 <= occurrence < PHASE_COUNT:
        raise BoundaryTangentScheduleError("branch split coordinate is invalid")

    device = pre_phase_states.device
    snapshot = pre_phase_states.detach().clone()
    tails_all, heads_all = matching_indices(device=device)
    matching = int(PHASE_MATCHINGS[occurrence])
    tails = tails_all[matching]
    heads = heads_all[matching]
    tail_mass = pre_phase_states.index_select(1, tails)
    head_mass = pre_phase_states.index_select(1, heads)
    pair_mass = tail_mass + head_mass
    positive = pair_mass > 0.0
    safe = torch.where(positive, pair_mass, torch.ones_like(pair_mass))
    x = torch.where(positive, head_mass / safe, torch.zeros_like(pair_mass))
    full_exposure = phase_base_exposure(pair_mass, occurrence)

    fractions = torch.as_tensor(
        MIDPOINT_FRACTIONS, dtype=torch.float64, device=device
    ).reshape(MIDPOINT_COUNT, 1, 1)
    x_lanes = x.unsqueeze(0).expand(MIDPOINT_COUNT, -1, -1).contiguous().reshape(-1)
    exposure_lanes = (full_exposure.unsqueeze(0) * fractions).contiguous().reshape(-1)
    id_lanes = torch.stack(
        tuple(
            controller_transition_ids(
                paths,
                outer_step=step,
                phase=occurrence,
                reverse_microstep=midpoint,
                role="partial_phase_target_prefix",
                device=device,
            )
            for midpoint in range(MIDPOINT_COUNT)
        )
    ).contiguous().reshape(-1)
    plan = build_fused_launch_plan(len(paths))
    if any(value.numel() != plan.total_lanes for value in (x_lanes, exposure_lanes, id_lanes)):
        raise BoundaryTangentScheduleError("canonical fused lane assembly is incomplete")

    later_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    code_parts: list[Tensor] = []
    mode_parts: list[Tensor] = []
    prefix_parts: list[Tensor] = []
    fallback_parts: list[Tensor] = []
    strengthened_parts: list[Tensor] = []
    fallback_reason_parts: list[Tensor] = []
    forbidden = {name: 0 for name in FORBIDDEN_DIAGNOSTICS}
    fallback_elapsed = 0.0
    backend_elapsed = 0.0
    candidate_elapsed = 0.0
    reported_launch_count = 0
    reported_maximum_lanes = 0
    active_profile = profile or JacobiRBCudaProfile()
    for start, stop in plan.chunk_ranges:
        lanes = stop - start
        result = sampler(
            x_lanes[start:stop].contiguous(),
            exposure_lanes[start:stop].contiguous(),
            rng_key=(
                int(root_seed),
                BOUNDARY_TANGENT_CACHE_VERSION,
                "partial-phase-target-prefix",
            ),
            transition_ids=id_lanes[start:stop].contiguous(),
            profile=active_profile,
        )
        later = _field(result, "later_head_fraction", "later", "y")
        target = _field(result, "denoising_target", "target", "z")
        codes = _field(result, "certificate_codes", "certificate_code")
        if any(
            value.device != device or value.numel() != lanes
            for value in (later, target, codes)
        ):
            raise BoundaryTangentScheduleError("fused sampler output has wrong lanes/device")
        later = later.reshape(-1).to(torch.float64).contiguous()
        target = target.reshape(-1).to(torch.float64).contiguous()
        codes = codes.reshape(-1).to(torch.uint8).contiguous()
        if (
            not bool(torch.isfinite(later).all())
            or not bool(torch.isfinite(target).all())
            or bool(torch.any((later < 0.0) | (later > 1.0)))
            or not bool(torch.all((codes & 0b1111) == 0b1111))
        ):
            raise BoundaryTangentScheduleError("fused sampler returned invalid output")
        later_parts.append(later)
        target_parts.append(target)
        code_parts.append(codes)
        mode_parts.append(
            _optional_tensor(
                result, "mode_counts", lanes=lanes, dtype=torch.int32, device=device
            )
        )
        prefix_parts.append(
            _optional_tensor(
                result, "prefix_bits", lanes=lanes, dtype=torch.int32, device=device
            )
        )
        fallback_parts.append(
            _optional_tensor(
                result, "fallback_mask", lanes=lanes, dtype=torch.bool, device=device
            )
        )
        strengthened_parts.append(
            _optional_tensor(
                result,
                "strengthened_mask",
                lanes=lanes,
                dtype=torch.bool,
                device=device,
            )
        )
        fallback_reason_parts.append(
            _optional_tensor(
                result,
                "arb_fallback_reason_codes",
                lanes=lanes,
                dtype=torch.uint8,
                device=device,
            )
        )
        for name in FORBIDDEN_DIAGNOSTICS:
            forbidden[name] += int(_scalar_diagnostic(result, name))
        fallback_elapsed += _scalar_diagnostic(result, "arb_fallback_elapsed_seconds")
        backend_elapsed += _scalar_diagnostic(result, "fused_authorizer_elapsed_seconds")
        candidate_elapsed += _scalar_diagnostic(result, "candidate_elapsed_seconds")
        launch_value = _scalar_diagnostic(
            result, "fused_authorizer_launch_count", default=1.0
        )
        maximum_lanes_value = _scalar_diagnostic(
            result, "maximum_cuda_launch_lanes", default=float(lanes)
        )
        if not launch_value.is_integer() or not maximum_lanes_value.is_integer():
            raise BoundaryTangentScheduleError(
                "authorizer launch diagnostics must be integral"
            )
        reported_launch_count += int(launch_value)
        reported_maximum_lanes = max(reported_maximum_lanes, int(maximum_lanes_value))
        if reported_maximum_lanes > MAXIMUM_LAUNCH_LANES:
            raise BoundaryTangentScheduleError("authorizer reported a lane-cap breach")

    shape = (MIDPOINT_COUNT, len(paths), EDGES_PER_PHASE)
    later_all = torch.cat(later_parts).reshape(shape)
    target_all = torch.cat(target_parts).reshape(shape)
    code_all = torch.cat(code_parts).reshape(shape)
    mode_all = torch.cat(mode_parts).reshape(shape)
    prefix_all = torch.cat(prefix_parts).reshape(shape)
    fallback_all = torch.cat(fallback_parts).reshape(shape)
    strengthened_all = torch.cat(strengthened_parts).reshape(shape)
    fallback_reason_all = torch.cat(fallback_reason_parts).reshape(shape)
    state_all = pre_phase_states.unsqueeze(0).expand(MIDPOINT_COUNT, -1, -1).clone()
    state_all[:, :, tails] = pair_mass.unsqueeze(0) * (1.0 - later_all)
    state_all[:, :, heads] = pair_mass.unsqueeze(0) * later_all
    batch = MidpointBranchBatch(
        path_ids=paths,
        outer_step=step,
        phase=occurrence,
        midpoint_fractions=MIDPOINT_FRACTIONS,
        later_full_state=state_all,
        later_head_fraction=later_all,
        denoising_target=target_all,
        certificate_codes=code_all,
        mode_counts=mode_all,
        prefix_bits=prefix_all,
        fallback_mask=fallback_all,
        strengthened_mask=strengthened_all,
        transition_count=plan.total_lanes,
        forbidden_counts=forbidden,
        fallback_elapsed_seconds=fallback_elapsed,
        backend_elapsed_seconds=backend_elapsed,
    )
    if not torch.equal(pre_phase_states, snapshot):
        raise BoundaryTangentScheduleError("fused observer mutated canonical states")
    return FusedMidpointBranchBatch(
        batch=batch,
        launch_plan=plan,
        launch_count=len(plan.chunk_ranges),
        fallback_reason_codes=fallback_reason_all,
        candidate_elapsed_seconds=candidate_elapsed,
        reported_authorizer_launch_count=reported_launch_count,
        reported_maximum_launch_lanes=reported_maximum_lanes,
    )


def expected_profile_transition_counts(profile_name: str) -> tuple[int, int, int]:
    if profile_name not in PROFILE_PATH_COUNTS:
        raise BoundaryTangentScheduleError("pilot profile is not frozen")
    paths = PROFILE_PATH_COUNTS[profile_name]
    base = paths * BASE_TRANSITIONS_PER_PILOT_PATH
    midpoint = paths * MIDPOINT_TRANSITIONS_PER_PILOT_PATH
    return base, midpoint, base + midpoint


@dataclass(frozen=True)
class PilotRepeatRecord:
    """One complete 64-step, complete-pipeline profile measurement."""

    profile_name: str
    repeat_index: int
    execution_order_index: int
    elapsed_seconds: float
    base_transition_count: int
    midpoint_transition_count: int
    certified_count: int
    fallback_count: int
    fallback_elapsed_seconds: float
    maximum_mass_error: float
    peak_memory_fraction: float
    committed_bytes: int
    maximum_launch_lanes: int
    output_sha256: str
    final_state_sha256: str
    forbidden_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.profile_name not in PILOT_PROFILE_NAMES:
            raise BoundaryTangentScheduleError("repeat profile is not frozen")
        repeat = _integer(self.repeat_index, "repeat_index")
        order = _integer(self.execution_order_index, "execution_order_index")
        expected_order = frozen_repeat_order(repeat).index(self.profile_name)
        if order != expected_order:
            raise BoundaryTangentScheduleError("repeat execution order changed")
        base, midpoint, total = expected_profile_transition_counts(self.profile_name)
        if (
            _integer(self.base_transition_count, "base_transition_count") != base
            or _integer(self.midpoint_transition_count, "midpoint_transition_count") != midpoint
            or _integer(self.certified_count, "certified_count") != total
        ):
            raise BoundaryTangentScheduleError("repeat transition counts changed")
        if _integer(self.fallback_count, "fallback_count") < 0:
            raise BoundaryTangentScheduleError("fallback count is negative")
        if _integer(self.committed_bytes, "committed_bytes") < 0:
            raise BoundaryTangentScheduleError("committed bytes is negative")
        lanes = _integer(self.maximum_launch_lanes, "maximum_launch_lanes")
        if lanes < 1 or lanes > MAXIMUM_LAUNCH_LANES:
            raise BoundaryTangentScheduleError("repeat exceeded the lane cap")
        numeric = (
            float(self.elapsed_seconds),
            float(self.fallback_elapsed_seconds),
            float(self.maximum_mass_error),
            float(self.peak_memory_fraction),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric) or numeric[0] <= 0.0:
            raise BoundaryTangentScheduleError("repeat diagnostics are invalid")
        if set(self.forbidden_counts) != set(FORBIDDEN_DIAGNOSTICS) or any(
            _integer(value, name) < 0 for name, value in self.forbidden_counts.items()
        ):
            raise BoundaryTangentScheduleError("repeat forbidden counts are malformed")
        _sha256(self.output_sha256, "output_sha256")
        _sha256(self.final_state_sha256, "final_state_sha256")

    @property
    def transition_count(self) -> int:
        return self.base_transition_count + self.midpoint_transition_count

    @property
    def transitions_per_second(self) -> float:
        return self.transition_count / self.elapsed_seconds

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULE_VERSION + "-pilot-repeat",
            "profile_name": self.profile_name,
            "repeat_index": self.repeat_index,
            "execution_order_index": self.execution_order_index,
            "elapsed_seconds": self.elapsed_seconds,
            "base_transition_count": self.base_transition_count,
            "midpoint_transition_count": self.midpoint_transition_count,
            "transition_count": self.transition_count,
            "certified_count": self.certified_count,
            "fallback_count": self.fallback_count,
            "fallback_elapsed_seconds": self.fallback_elapsed_seconds,
            "maximum_mass_error": self.maximum_mass_error,
            "peak_memory_fraction": self.peak_memory_fraction,
            "committed_bytes": self.committed_bytes,
            "maximum_launch_lanes": self.maximum_launch_lanes,
            "transitions_per_second": self.transitions_per_second,
            "output_sha256": self.output_sha256,
            "final_state_sha256": self.final_state_sha256,
            "forbidden_counts": dict(self.forbidden_counts),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PilotRepeatRecord":
        if record.get("schema") != SCHEDULE_VERSION + "-pilot-repeat":
            raise BoundaryTangentScheduleError("pilot repeat schema changed")
        value = cls(
            profile_name=str(record.get("profile_name")),
            repeat_index=record.get("repeat_index"),
            execution_order_index=record.get("execution_order_index"),
            elapsed_seconds=record.get("elapsed_seconds"),
            base_transition_count=record.get("base_transition_count"),
            midpoint_transition_count=record.get("midpoint_transition_count"),
            certified_count=record.get("certified_count"),
            fallback_count=record.get("fallback_count"),
            fallback_elapsed_seconds=record.get("fallback_elapsed_seconds"),
            maximum_mass_error=record.get("maximum_mass_error"),
            peak_memory_fraction=record.get("peak_memory_fraction"),
            committed_bytes=record.get("committed_bytes"),
            maximum_launch_lanes=record.get("maximum_launch_lanes"),
            output_sha256=str(record.get("output_sha256")),
            final_state_sha256=str(record.get("final_state_sha256")),
            forbidden_counts=record.get("forbidden_counts", {}),
        )
        expected_rate = value.transitions_per_second
        if (
            _integer(record.get("transition_count"), "transition_count")
            != value.transition_count
            or not math.isclose(
                float(record.get("transitions_per_second")),
                expected_rate,
                rel_tol=1.0e-15,
                abs_tol=0.0,
            )
        ):
            raise BoundaryTangentScheduleError("pilot repeat derived fields changed")
        return value


def validate_repeat_records(
    records: Sequence[PilotRepeatRecord],
) -> dict[str, tuple[PilotRepeatRecord, ...]]:
    """Validate a complete sealed 4-profile by 3-repeat pilot panel."""

    values = tuple(records)
    if len(values) != len(PILOT_PROFILE_NAMES) * PILOT_REPEAT_COUNT:
        raise BoundaryTangentScheduleError("pilot repeat panel is incomplete")
    grouped: dict[str, list[PilotRepeatRecord]] = {
        name: [] for name in PILOT_PROFILE_NAMES
    }
    identities: set[tuple[str, int]] = set()
    for value in values:
        if not isinstance(value, PilotRepeatRecord):
            raise BoundaryTangentScheduleError("pilot record has wrong type")
        identity = (value.profile_name, value.repeat_index)
        if identity in identities:
            raise BoundaryTangentScheduleError("pilot repeat identity is duplicated")
        identities.add(identity)
        grouped[value.profile_name].append(value)
    output: dict[str, tuple[PilotRepeatRecord, ...]] = {}
    for profile_name, profile_records in grouped.items():
        ordered = tuple(sorted(profile_records, key=lambda value: value.repeat_index))
        if tuple(value.repeat_index for value in ordered) != tuple(range(PILOT_REPEAT_COUNT)):
            raise BoundaryTangentScheduleError("pilot repeat sequence is incomplete")
        if len({value.output_sha256 for value in ordered}) != 1 or len(
            {value.final_state_sha256 for value in ordered}
        ) != 1:
            raise BoundaryTangentScheduleError("pilot repeat hashes changed")
        output[profile_name] = ordered
    return output


@dataclass(frozen=True)
class ScheduleProjection:
    slowest_profile_seconds: Mapping[str, float]
    slowest_profile_rates: Mapping[str, float]
    projected_seconds: float
    projected_effective_rate: float
    projected_persistence_bytes: int
    maximum_peak_memory_fraction: float
    fallback_fraction: float
    fallback_time_fraction: float
    maximum_mass_error: float
    forbidden_total: int
    passed: bool
    failed_checks: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": SCHEDULE_VERSION + "-projection",
            "slowest_profile_seconds": dict(self.slowest_profile_seconds),
            "slowest_profile_rates": dict(self.slowest_profile_rates),
            "projected_seconds": self.projected_seconds,
            "projected_hours": self.projected_seconds / 3600.0,
            "projected_effective_rate": self.projected_effective_rate,
            "projected_base_transitions": PROJECTED_BASE_TRANSITIONS,
            "projected_midpoint_transitions": PROJECTED_MIDPOINT_TRANSITIONS,
            "projected_total_transitions": PROJECTED_TOTAL_TRANSITIONS,
            "projected_persistence_bytes": self.projected_persistence_bytes,
            "maximum_peak_memory_fraction": self.maximum_peak_memory_fraction,
            "fallback_fraction": self.fallback_fraction,
            "fallback_time_fraction": self.fallback_time_fraction,
            "maximum_mass_error": self.maximum_mass_error,
            "forbidden_total": self.forbidden_total,
            "passed": int(self.passed),
            "failed_checks": list(self.failed_checks),
            "maximum_projected_exact_cache_hours": 30.0,
        }


def project_frozen_schedule(records: Sequence[PilotRepeatRecord]) -> ScheduleProjection:
    """Apply the exact slowest-repeat production projection and frozen gates."""

    grouped = validate_repeat_records(records)
    slowest_records = {
        profile: max(values, key=lambda value: value.elapsed_seconds)
        for profile, values in grouped.items()
    }
    slowest_seconds = {
        profile: value.elapsed_seconds for profile, value in slowest_records.items()
    }
    slowest_rates = {
        profile: value.transitions_per_second
        for profile, value in slowest_records.items()
    }
    projected_seconds = PROJECTION_FACTOR * sum(
        PROFILE_PROJECTION_MULTIPLICITIES[profile] * seconds
        for profile, seconds in slowest_seconds.items()
    )
    projected_rate = PROJECTED_TOTAL_TRANSITIONS / projected_seconds
    # Cache persistence is conservatively based on the largest committed byte
    # count seen in each cache profile, not on the selected timing repeat.
    persistence = PROJECTION_FACTOR * (
        9 * max(value.committed_bytes for value in grouped[PROFILE_CACHE_P10])
        + max(value.committed_bytes for value in grouped[PROFILE_CACHE_P6])
    )
    total_transitions = sum(value.transition_count for value in records)
    total_fallback = sum(value.fallback_count for value in records)
    total_elapsed = sum(value.elapsed_seconds for value in records)
    total_fallback_elapsed = sum(value.fallback_elapsed_seconds for value in records)
    fallback_fraction = total_fallback / total_transitions
    fallback_time_fraction = total_fallback_elapsed / total_elapsed
    maximum_memory = max(value.peak_memory_fraction for value in records)
    maximum_mass = max(value.maximum_mass_error for value in records)
    forbidden_total = sum(
        sum(value.forbidden_counts.values()) for value in records
    )
    failed: list[str] = []
    if projected_seconds > MAXIMUM_PROJECTED_SECONDS:
        failed.append("projected_seconds")
    if projected_rate < MINIMUM_EFFECTIVE_PROJECTED_RATE:
        failed.append("projected_effective_rate")
    if any(rate < MINIMUM_PROFILE_RATE for rate in slowest_rates.values()):
        failed.append("profile_rate")
    if fallback_fraction > MAXIMUM_FALLBACK_FRACTION:
        failed.append("fallback_fraction")
    if fallback_time_fraction > MAXIMUM_FALLBACK_TIME_FRACTION:
        failed.append("fallback_time_fraction")
    if maximum_mass > MAXIMUM_MASS_ERROR:
        failed.append("mass_error")
    if maximum_memory > MAXIMUM_MEMORY_FRACTION:
        failed.append("memory_fraction")
    if persistence > MAXIMUM_PERSISTENCE_BYTES:
        failed.append("persistence_bytes")
    if forbidden_total:
        failed.append("forbidden_events")
    return ScheduleProjection(
        slowest_profile_seconds=slowest_seconds,
        slowest_profile_rates=slowest_rates,
        projected_seconds=projected_seconds,
        projected_effective_rate=projected_rate,
        projected_persistence_bytes=persistence,
        maximum_peak_memory_fraction=maximum_memory,
        fallback_fraction=fallback_fraction,
        fallback_time_fraction=fallback_time_fraction,
        maximum_mass_error=maximum_mass,
        forbidden_total=forbidden_total,
        passed=not failed,
        failed_checks=tuple(failed),
    )


def _validate_module_constants() -> None:
    if PROJECTION_FACTOR != 8:
        raise AssertionError("pilot projection factor changed")
    weighted_paths = sum(
        PROFILE_PATH_COUNTS[name] * PROFILE_PROJECTION_MULTIPLICITIES[name]
        for name in PILOT_PROFILE_NAMES
    )
    if weighted_paths != 160:
        raise AssertionError("frozen production path schedule changed")
    if (
        PROJECTION_FACTOR * weighted_paths * BASE_TRANSITIONS_PER_PILOT_PATH
        != PROJECTED_BASE_TRANSITIONS
        or PROJECTION_FACTOR * weighted_paths * MIDPOINT_TRANSITIONS_PER_PILOT_PATH
        != PROJECTED_MIDPOINT_TRANSITIONS
        or PROJECTED_BASE_TRANSITIONS + PROJECTED_MIDPOINT_TRANSITIONS
        != PROJECTED_TOTAL_TRANSITIONS
    ):
        raise AssertionError("frozen transition projection arithmetic changed")
    frozen_path_plan()
    frozen_production_cohort_plan()


_validate_module_constants()


__all__ = [
    "BASE_TRANSITIONS_PER_PILOT_PATH",
    "BoundaryTangentScheduleError",
    "CONFIRMATION_COHORT_SIZES",
    "FusedLaunchPlan",
    "FusedMidpointBranchBatch",
    "MAXIMUM_LAUNCH_LANES",
    "MAXIMUM_PROJECTED_SECONDS",
    "MIDPOINT_TRANSITIONS_PER_PILOT_PATH",
    "MINIMUM_EFFECTIVE_PROJECTED_RATE",
    "PILOT_PROFILE_NAMES",
    "PILOT_REPEAT_COUNT",
    "PROFILE_CACHE_P10",
    "PROFILE_CACHE_P6",
    "PROFILE_PATH_COUNTS",
    "PROFILE_PATH_IDS",
    "PROFILE_PROJECTION_MULTIPLICITIES",
    "PROFILE_STREAM_P10",
    "PROFILE_STREAM_P4",
    "PROJECTED_BASE_TRANSITIONS",
    "PROJECTED_MIDPOINT_TRANSITIONS",
    "PROJECTED_TOTAL_TRANSITIONS",
    "PilotRepeatRecord",
    "ROOT_SEED",
    "SCHEDULE_VERSION",
    "SHARD_STEPS",
    "ScheduleProjection",
    "TOTAL_TRANSITIONS_PER_PILOT_PATH",
    "TRAIN_VALIDATION_COHORT_SIZES",
    "WARMUP_PATH_IDS",
    "WINDOW_START_STEPS",
    "WINDOW_STEP_COUNT",
    "build_fused_launch_plan",
    "expected_profile_transition_counts",
    "frozen_path_plan",
    "frozen_production_cohort_plan",
    "frozen_repeat_order",
    "project_frozen_schedule",
    "sample_fused_midpoint_branches",
    "split_co_scheduled_payload_by_role",
    "validate_repeat_records",
]
