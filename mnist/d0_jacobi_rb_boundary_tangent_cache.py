"""Exact time-local cache helpers for the boundary-tangent RB controller.

The canonical K=512 path is advanced by the existing certified multipath
scheduler.  This module only creates independent prefix branches from a saved
pre-phase state.  A branch is diagnostic/training evidence and is never fed
back into the canonical path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    matching_indices,
)
from mnist.d0_jacobi_rb_reverse_controller import (
    controller_transition_ids,
    internal_reverse_time,
)


BOUNDARY_TANGENT_CACHE_VERSION = "d0-jacobi-rb-boundary-tangent-cache-v1"
MIDPOINT_FRACTIONS = tuple((2 * index + 1) / 16.0 for index in range(8))
MIDPOINT_COUNT = len(MIDPOINT_FRACTIONS)
SELECTED_OUTER_STEPS = tuple(range(15, OUTER_STEPS, 16))
ROOT_SEED = 261_311
FORBIDDEN_DIAGNOSTICS = (
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "renormalization_count",
    "nonfinite_count",
)


class BoundaryTangentCacheError(ValueError):
    """Raised when an exact time-local cache contract is violated."""


def midpoint_sample_key(
    path_id: int, outer_step: int, phase: int, midpoint_index: int
) -> int:
    """Pack a unique signed-int64-safe time-local sample key."""

    values = (path_id, outer_step, phase, midpoint_index)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise BoundaryTangentCacheError("sample-key coordinates must be integers")
    path = int(path_id)
    step = int(outer_step)
    occurrence = int(phase)
    midpoint = int(midpoint_index)
    if not 0 <= path < (1 << 20):
        raise BoundaryTangentCacheError("path ID lies outside the 20-bit plan")
    if not 0 <= step < OUTER_STEPS:
        raise BoundaryTangentCacheError("outer step lies outside K=512")
    if not 0 <= occurrence < PHASE_COUNT:
        raise BoundaryTangentCacheError("phase occurrence lies outside [0,7)")
    if not 0 <= midpoint < MIDPOINT_COUNT:
        raise BoundaryTangentCacheError("midpoint index lies outside [0,8)")
    # 3 midpoint bits, 3 phase bits, 9 outer-step bits and 20 path bits.
    return (path << 15) | (step << 6) | (occurrence << 3) | midpoint


def midpoint_fraction(index: int) -> float:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise BoundaryTangentCacheError("midpoint index must be integral")
    value = int(index)
    if not 0 <= value < MIDPOINT_COUNT:
        raise BoundaryTangentCacheError("midpoint index lies outside [0,8)")
    return MIDPOINT_FRACTIONS[value]


def phase_base_exposure(pair_mass: Tensor, phase: int) -> Tensor:
    """Return the exact full-phase exposure under the frozen convention."""

    if not isinstance(pair_mass, Tensor) or pair_mass.dtype != torch.float64:
        raise BoundaryTangentCacheError("pair mass must be a float64 tensor")
    if isinstance(phase, bool) or not isinstance(phase, (int, np.integer)):
        raise BoundaryTangentCacheError("phase must be integral")
    occurrence = int(phase)
    if not 0 <= occurrence < PHASE_COUNT:
        raise BoundaryTangentCacheError("phase occurrence lies outside [0,7)")
    positive = pair_mass > 0.0
    safe = torch.where(positive, pair_mass, torch.ones_like(pair_mass))
    coefficient = (
        3.0
        * (5.0e-5 / float(OUTER_STEPS))
        * float(PHASE_DURATIONS[occurrence])
        / (1.0 / 28.0) ** 2
    )
    return torch.where(positive, coefficient / safe, torch.zeros_like(pair_mass))


def _field(result: Any, *names: str) -> Tensor:
    for name in names:
        if isinstance(result, Mapping) and isinstance(result.get(name), Tensor):
            return result[name]
        value = getattr(result, name, None)
        if isinstance(value, Tensor):
            return value
    raise BoundaryTangentCacheError(f"certified sampler omitted {names[0]}")


def _optional_field(
    result: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
    if value is None:
        return torch.zeros(shape, dtype=dtype, device=device)
    if not isinstance(value, Tensor) or value.numel() != math.prod(shape):
        raise BoundaryTangentCacheError(f"certified sampler field {name} has wrong shape")
    return value.reshape(shape).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class MidpointBranchBatch:
    path_ids: tuple[int, ...]
    outer_step: int
    phase: int
    midpoint_fractions: tuple[float, ...]
    later_full_state: Tensor
    later_head_fraction: Tensor
    denoising_target: Tensor
    certificate_codes: Tensor
    mode_counts: Tensor
    prefix_bits: Tensor
    fallback_mask: Tensor
    strengthened_mask: Tensor
    transition_count: int
    forbidden_counts: Mapping[str, int]
    fallback_elapsed_seconds: float
    backend_elapsed_seconds: float

    def __post_init__(self) -> None:
        paths = len(self.path_ids)
        expected_edge = (MIDPOINT_COUNT, paths, EDGES_PER_PHASE)
        if self.later_full_state.shape != (MIDPOINT_COUNT, paths, STATE_SIZE):
            raise BoundaryTangentCacheError("branch full-state tensor has wrong shape")
        for name in (
            "later_head_fraction",
            "denoising_target",
            "certificate_codes",
            "mode_counts",
            "prefix_bits",
            "fallback_mask",
            "strengthened_mask",
        ):
            if getattr(self, name).shape != expected_edge:
                raise BoundaryTangentCacheError(f"branch {name} tensor has wrong shape")
        if self.transition_count != MIDPOINT_COUNT * paths * EDGES_PER_PHASE:
            raise BoundaryTangentCacheError("branch transition count is inconsistent")
        if set(self.forbidden_counts) != set(FORBIDDEN_DIAGNOSTICS) or any(
            isinstance(value, bool) or int(value) < 0
            for value in self.forbidden_counts.values()
        ):
            raise BoundaryTangentCacheError("branch forbidden diagnostics are malformed")
        if (
            not math.isfinite(float(self.fallback_elapsed_seconds))
            or float(self.fallback_elapsed_seconds) < 0.0
            or not math.isfinite(float(self.backend_elapsed_seconds))
            or float(self.backend_elapsed_seconds) < 0.0
        ):
            raise BoundaryTangentCacheError("branch timing diagnostics are malformed")

    @property
    def certified_count(self) -> int:
        return int(((self.certificate_codes & 0b1111) == 0b1111).sum().item())

    def output_sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.later_full_state,
            self.denoising_target,
            self.certificate_codes,
        ):
            array = np.ascontiguousarray(value.detach().cpu().numpy())
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()


def sample_midpoint_branches(
    pre_phase_states: Tensor,
    *,
    path_ids: Sequence[int],
    outer_step: int,
    phase: int,
    root_seed: int = ROOT_SEED,
    profile: JacobiRBCudaProfile | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> MidpointBranchBatch:
    """Sample all M8 prefix branches without mutating ``pre_phase_states``."""

    if not isinstance(pre_phase_states, Tensor):
        raise TypeError("pre_phase_states must be a torch.Tensor")
    if (
        pre_phase_states.dtype != torch.float64
        or pre_phase_states.ndim != 2
        or pre_phase_states.shape[1] != STATE_SIZE
        or not pre_phase_states.is_contiguous()
    ):
        raise BoundaryTangentCacheError(
            "pre-phase states must be contiguous float64 [P,784]"
        )
    if sampler is sample_alpha1_rb_transition_batch_cuda and not pre_phase_states.is_cuda:
        raise BoundaryTangentCacheError("production branch sampling requires CUDA states")
    paths = tuple(int(value) for value in path_ids)
    if len(paths) != int(pre_phase_states.shape[0]) or len(set(paths)) != len(paths):
        raise BoundaryTangentCacheError("path IDs must be unique and match the batch")
    if any(not 0 <= value < (1 << 20) for value in paths):
        raise BoundaryTangentCacheError("path ID lies outside the 20-bit plan")
    if not 0 <= int(outer_step) < OUTER_STEPS or not 0 <= int(phase) < PHASE_COUNT:
        raise BoundaryTangentCacheError("branch split coordinate is invalid")
    device = pre_phase_states.device
    input_snapshot = pre_phase_states.detach().clone()
    tails_all, heads_all = matching_indices(device=device)
    matching = int(PHASE_MATCHINGS[int(phase)])
    tails = tails_all[matching]
    heads = heads_all[matching]
    tail_mass = pre_phase_states.index_select(1, tails)
    head_mass = pre_phase_states.index_select(1, heads)
    pair_mass = tail_mass + head_mass
    positive = pair_mass > 0.0
    safe = torch.where(positive, pair_mass, torch.ones_like(pair_mass))
    x = torch.where(positive, head_mass / safe, torch.zeros_like(pair_mass))
    full_exposure = phase_base_exposure(pair_mass, int(phase))
    active_profile = profile or JacobiRBCudaProfile()
    later_blocks: list[Tensor] = []
    target_blocks: list[Tensor] = []
    code_blocks: list[Tensor] = []
    state_blocks: list[Tensor] = []
    mode_blocks: list[Tensor] = []
    prefix_blocks: list[Tensor] = []
    fallback_blocks: list[Tensor] = []
    strengthened_blocks: list[Tensor] = []
    forbidden_counts = {name: 0 for name in FORBIDDEN_DIAGNOSTICS}
    fallback_elapsed_seconds = 0.0
    backend_elapsed_seconds = 0.0
    shape = (len(paths), EDGES_PER_PHASE)
    for midpoint_index, fraction in enumerate(MIDPOINT_FRACTIONS):
        ids = controller_transition_ids(
            paths,
            outer_step=int(outer_step),
            phase=int(phase),
            reverse_microstep=midpoint_index,
            role="partial_phase_target_prefix",
            device=device,
        )
        result = sampler(
            x.reshape(-1).contiguous(),
            (full_exposure * fraction).reshape(-1).contiguous(),
            rng_key=(
                int(root_seed),
                BOUNDARY_TANGENT_CACHE_VERSION,
                "partial-phase-target-prefix",
            ),
            transition_ids=ids.reshape(-1).contiguous(),
            profile=active_profile,
        )
        later = _field(result, "later_head_fraction", "later", "y").reshape(shape)
        target = _field(result, "denoising_target", "target", "z").reshape(shape)
        codes = _field(result, "certificate_codes", "certificate_code").reshape(shape)
        if any(value.device != device for value in (later, target, codes)):
            raise BoundaryTangentCacheError("branch sampler output left its device")
        later = later.to(torch.float64)
        target = target.to(torch.float64)
        codes = codes.to(torch.uint8)
        certified = (codes & 0b1111) == 0b1111
        if not bool(torch.all(certified)):
            raise BoundaryTangentCacheError("midpoint branch contains uncertified transitions")
        if not bool(torch.isfinite(later).all()) or not bool(torch.isfinite(target).all()):
            raise BoundaryTangentCacheError("midpoint branch contains nonfinite output")
        if bool(torch.any((later < 0.0) | (later > 1.0))):
            raise BoundaryTangentCacheError("midpoint branch fraction leaves [0,1]")
        branch_state = pre_phase_states.clone()
        branch_state[:, tails] = pair_mass * (1.0 - later)
        branch_state[:, heads] = pair_mass * later
        later_blocks.append(later)
        target_blocks.append(target)
        code_blocks.append(codes)
        state_blocks.append(branch_state)
        mode_blocks.append(
            _optional_field(result, "mode_counts", shape=shape, dtype=torch.int32, device=device)
        )
        prefix_blocks.append(
            _optional_field(result, "prefix_bits", shape=shape, dtype=torch.int32, device=device)
        )
        fallback_blocks.append(
            _optional_field(result, "fallback_mask", shape=shape, dtype=torch.bool, device=device)
        )
        strengthened_blocks.append(
            _optional_field(result, "strengthened_mask", shape=shape, dtype=torch.bool, device=device)
        )
        diagnostics = (
            result.get("diagnostics")
            if isinstance(result, Mapping)
            else getattr(result, "diagnostics", None)
        )
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        for name in FORBIDDEN_DIAGNOSTICS:
            raw = diagnostics.get(name, 0)
            if isinstance(raw, Tensor):
                if raw.numel() != 1:
                    raise BoundaryTangentCacheError(
                        f"branch diagnostic {name} is not scalar"
                    )
                raw = raw.detach().cpu().item()
            forbidden_counts[name] += int(raw)
        for name, target_name in (
            ("arb_fallback_elapsed_seconds", "fallback"),
            ("fused_authorizer_elapsed_seconds", "backend"),
        ):
            raw = diagnostics.get(name, 0.0)
            if isinstance(raw, Tensor):
                if raw.numel() != 1:
                    raise BoundaryTangentCacheError(
                        f"branch diagnostic {name} is not scalar"
                    )
                raw = raw.detach().cpu().item()
            if target_name == "fallback":
                fallback_elapsed_seconds += float(raw)
            else:
                backend_elapsed_seconds += float(raw)
    output = MidpointBranchBatch(
        path_ids=paths,
        outer_step=int(outer_step),
        phase=int(phase),
        midpoint_fractions=MIDPOINT_FRACTIONS,
        later_full_state=torch.stack(state_blocks),
        later_head_fraction=torch.stack(later_blocks),
        denoising_target=torch.stack(target_blocks),
        certificate_codes=torch.stack(code_blocks),
        mode_counts=torch.stack(mode_blocks),
        prefix_bits=torch.stack(prefix_blocks),
        fallback_mask=torch.stack(fallback_blocks),
        strengthened_mask=torch.stack(strengthened_blocks),
        transition_count=MIDPOINT_COUNT * len(paths) * EDGES_PER_PHASE,
        forbidden_counts=forbidden_counts,
        fallback_elapsed_seconds=fallback_elapsed_seconds,
        backend_elapsed_seconds=backend_elapsed_seconds,
    )
    # The observer must never alter the canonical input state.
    if not torch.equal(pre_phase_states, input_snapshot):
        raise BoundaryTangentCacheError("midpoint observer mutated the canonical state")
    return output


def flatten_midpoint_batches(
    batches: Sequence[MidpointBranchBatch],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Create separated permitted-input and label/audit cache arrays."""

    rows: list[tuple[int, int, int, int, float, np.ndarray, np.ndarray, np.ndarray]] = []
    for batch in batches:
        states = batch.later_full_state.detach().cpu().numpy()
        targets = batch.denoising_target.detach().cpu().numpy()
        codes = batch.certificate_codes.detach().cpu().numpy()
        for midpoint_index, fraction in enumerate(batch.midpoint_fractions):
            for path_index, path_id in enumerate(batch.path_ids):
                rows.append(
                    (
                        path_id,
                        batch.outer_step,
                        batch.phase,
                        midpoint_index,
                        fraction,
                        np.ascontiguousarray(states[midpoint_index, path_index]),
                        np.ascontiguousarray(targets[midpoint_index, path_index]),
                        np.ascontiguousarray(codes[midpoint_index, path_index]),
                    )
                )
    rows.sort(key=lambda row: row[:4])
    keys = np.asarray(
        [midpoint_sample_key(*row[:4]) for row in rows], dtype=np.int64
    )
    path_id = np.asarray([row[0] for row in rows], dtype=np.int64)
    outer_step = np.asarray([row[1] for row in rows], dtype=np.int16)
    phase = np.asarray([row[2] for row in rows], dtype=np.int8)
    midpoint_index = np.asarray([row[3] for row in rows], dtype=np.int8)
    midpoint = np.asarray([row[4] for row in rows], dtype=np.float64)
    reverse_time = np.asarray(
        [internal_reverse_time(row[1], row[2], row[4]) for row in rows],
        dtype=np.float64,
    )
    inputs = {
        "sample_key": keys,
        "later_full_state": np.stack([row[5] for row in rows]).astype(np.float64),
        "reverse_time": reverse_time,
        "phase": phase.copy(),
        "color": np.asarray([PHASE_MATCHINGS[int(value)] for value in phase], dtype=np.int8),
        "duration": np.asarray([PHASE_DURATIONS[int(value)] for value in phase], dtype=np.float64),
        "label": np.full(len(rows), 3, dtype=np.int64),
    }
    audit = {
        "sample_key": keys.copy(),
        "path_id": path_id,
        "outer_step": outer_step,
        "phase": phase.copy(),
        "midpoint_index": midpoint_index,
        "midpoint_fraction": midpoint,
        "denoising_target": np.stack([row[6] for row in rows]).astype(np.float64),
        "certificate_codes": np.stack([row[7] for row in rows]).astype(np.uint8),
    }
    if len(np.unique(keys)) != len(keys):
        raise BoundaryTangentCacheError("time-local sample keys collide")
    return inputs, audit


__all__ = [
    "BOUNDARY_TANGENT_CACHE_VERSION",
    "BoundaryTangentCacheError",
    "FORBIDDEN_DIAGNOSTICS",
    "MIDPOINT_COUNT",
    "MIDPOINT_FRACTIONS",
    "MidpointBranchBatch",
    "ROOT_SEED",
    "SELECTED_OUTER_STEPS",
    "flatten_midpoint_batches",
    "midpoint_fraction",
    "midpoint_sample_key",
    "phase_base_exposure",
    "sample_midpoint_branches",
]
