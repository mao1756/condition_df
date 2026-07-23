"""Exact same-phase multi-path scheduler for the certified Jacobi RB CUDA API.

The certified transition law is unchanged.  This module only packs independent
paths into larger CUDA calls.  Within every path, the seven matching phases are
still evaluated serially and each phase updates the device-resident state before
the next phase begins.  The flattened lane order is always path-major, then
edge-major, and random prefixes remain functions of the canonical
``(path, outer_step, phase, edge)`` transition ID rather than batch position.

The frozen projected cache consists of 64 paths in six ten-path groups and one
four-path group.  Ten paths occupy ``10 * 392 == 3920`` lanes, below the
immutable 4096-lane backend limit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
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


MULTIPATH_SCHEDULER_VERSION = "jacobi-rb-cuda-exact-multipath-v1"
MAX_PATHS_PER_GROUP = 10
SHARD_STEPS = 8
PATH_STATE_SIZE = 28 * 28
EDGES_PER_PHASE = 392
PHASE_MATCHINGS = (0, 1, 2, 3, 2, 1, 0)
PHASE_DURATIONS = (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
FROZEN_VALIDATION_GROUP_SIZES = (10, 4)
FROZEN_PROJECTION_GROUP_SIZES = (10, 10, 10, 10, 10, 10, 4)
FROZEN_PROJECTION_PATH_COUNT = sum(FROZEN_PROJECTION_GROUP_SIZES)

_FORBIDDEN_DIAGNOSTICS = (
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "renormalization_count",
    "nonfinite_count",
)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExactMultipathPathRecord:
    """Deterministic per-path evidence extracted at the shard boundary."""

    path_id: int
    transition_count: int
    certified_count: int
    fallback_count: int
    strengthened_count: int
    maximum_mode_count: int
    maximum_prefix_bits: int
    certificate_code_counts: Mapping[str, int]
    mode_count_counts: Mapping[str, int]
    prefix_bit_counts: Mapping[str, int]
    arb_fallback_reason_code_counts: Mapping[str, int]
    input_state_sha256: str
    output_sha256: str
    final_state_sha256: str
    certificate_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactMultipathPhaseStateRecord:
    """Post-phase state commitments, canonically ordered by path ID."""

    outer_step: int
    phase: int
    path_state_sha256_by_id: tuple[tuple[int, str], ...]
    batch_state_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactMultipathShardResult:
    """One exact eight-step shard and its restart/certificate evidence."""

    final_states: Tensor
    committed_final_states: np.ndarray = field(repr=False, compare=False)
    path_records: tuple[ExactMultipathPathRecord, ...]
    phase_state_records: tuple[ExactMultipathPhaseStateRecord, ...]
    batch_output_sha256: str
    batch_final_state_sha256: str
    batch_certificate_sha256: str
    diagnostics: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-safe evidence; the device tensor stays separate."""

        return {
            "schema": MULTIPATH_SCHEDULER_VERSION + "-shard",
            "schema_version": 1,
            "path_records": [record.to_dict() for record in self.path_records],
            "phase_state_records": [
                record.to_dict() for record in self.phase_state_records
            ],
            "batch_output_sha256": self.batch_output_sha256,
            "batch_final_state_sha256": self.batch_final_state_sha256,
            "batch_certificate_sha256": self.batch_certificate_sha256,
            "diagnostics": dict(self.diagnostics),
        }


def _path_id_tuple(path_ids: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in path_ids:
        if isinstance(value, bool):
            raise TypeError("path IDs must be integers, not bool")
        try:
            integer = operator.index(value)
        except TypeError as exc:
            raise TypeError("path IDs must be integers") from exc
        if not 0 <= integer < (1 << 20):
            raise ValueError("path IDs must fit the canonical 20-bit field")
        result.append(integer)
    if not result:
        raise ValueError("at least one path is required")
    if len(set(result)) != len(result):
        raise ValueError("path IDs must be unique within a multipath shard")
    return tuple(result)


def _group_tuple(group_sizes: Sequence[int] | None, path_count: int) -> tuple[int, ...]:
    if group_sizes is None:
        if path_count > MAX_PATHS_PER_GROUP:
            raise ValueError("more than ten paths require an explicit group schedule")
        return (path_count,)
    try:
        groups = tuple(operator.index(value) for value in group_sizes)
    except TypeError as exc:
        raise TypeError("path group sizes must be integers") from exc
    if (
        not groups
        or any(value <= 0 or value > MAX_PATHS_PER_GROUP for value in groups)
        or sum(groups) != int(path_count)
    ):
        raise ValueError("path groups must partition all paths in groups of at most ten")
    return groups


def canonical_same_phase_transition_ids(
    path_ids: Sequence[int],
    *,
    outer_step: int,
    phase: int,
    device: torch.device,
) -> Tensor:
    """Return contiguous path-major canonical IDs for one matching phase."""

    paths = _path_id_tuple(path_ids)
    rows = [
        _controls.canonical_transition_ids(
            path=path_id,
            outer_step=int(outer_step),
            phase=int(phase),
            edge_start=0,
            count=EDGES_PER_PHASE,
            device=device,
        )
        for path_id in paths
    ]
    return torch.stack(rows, dim=0).reshape(-1).contiguous()


def _result_tensor(result: Any, *names: str, shape: tuple[int, int]) -> Tensor:
    value = _controls._field(result, *names)
    if not isinstance(value, Tensor):
        raise _controls.RigorousCudaControlError(
            "multipath scheduler requires device-resident tensor outputs"
        )
    if value.numel() != math.prod(shape):
        raise _controls.RigorousCudaControlError(
            "multipath CUDA output has the wrong lane count"
        )
    return value.reshape(shape)


def _optional_result_tensor(
    result: Any,
    name: str,
    *,
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    value = _controls._optional_field(result, name, None)
    if value is None:
        return torch.zeros(shape, dtype=dtype, device=device)
    if not isinstance(value, Tensor):
        raise _controls.RigorousCudaControlError(
            f"multipath CUDA output {name} is not device resident"
        )
    if value.device != device:
        raise _controls.RigorousCudaControlError(
            f"multipath CUDA output {name} left the selected device"
        )
    if value.numel() != math.prod(shape):
        raise _controls.RigorousCudaControlError(
            f"multipath CUDA output {name} has the wrong lane count"
        )
    return value.reshape(shape).to(device=device, dtype=dtype)


def _integer_counts(values: np.ndarray) -> dict[str, int]:
    """Return a stable, JSON-safe exact histogram for an integer array."""

    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }


def _diagnostic_tensor(
    result: Any, name: str, *, dtype: torch.dtype, device: torch.device
) -> Tensor:
    diagnostics = _controls._optional_field(result, "diagnostics", {})
    value = diagnostics.get(name) if isinstance(diagnostics, Mapping) else None
    if value is None:
        return torch.zeros((), dtype=dtype, device=device)
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise _controls.RigorousCudaControlError(
                f"CUDA diagnostic {name} is not scalar"
            )
        return value.reshape(()).to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device).reshape(())


def run_exact_multipath_shard(
    states: Tensor,
    *,
    path_ids: Sequence[int],
    start_step: int,
    root_seed: int,
    profile: JacobiRBCudaProfile,
    group_sizes: Sequence[int] | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
    step_count: int = SHARD_STEPS,
    capture_phase_state_trace: bool = False,
) -> ExactMultipathShardResult:
    """Advance independent paths through one exact, phase-serial shard.

    Each backend call contains at most ten paths.  Calls are flattened in
    path-major order, and the state remains on its input device throughout the
    eight steps.  Hashing and scalar diagnostics are materialized only after
    the complete shard.
    """

    if not isinstance(states, Tensor):
        raise TypeError("states must be a torch.Tensor")
    if states.dtype != torch.float64 or states.ndim != 2:
        raise TypeError("states must be a float64 tensor with shape [paths, 784]")
    if states.shape[1] != PATH_STATE_SIZE or not states.is_contiguous():
        raise ValueError("states must be contiguous with shape [paths, 784]")
    if sampler is sample_alpha1_rb_transition_batch_cuda and not states.is_cuda:
        raise ValueError("the production multipath scheduler requires CUDA states")
    if int(step_count) != SHARD_STEPS:
        raise ValueError("exact restart shards contain exactly eight steps")
    if not 0 <= int(start_step) <= 512 - SHARD_STEPS:
        raise ValueError("start_step must begin a complete K=512 eight-step shard")
    if int(start_step) % SHARD_STEPS:
        raise ValueError("start_step must lie on an eight-step restart boundary")
    if not isinstance(profile, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")

    paths = _path_id_tuple(path_ids)
    if len(paths) != int(states.shape[0]):
        raise ValueError("path_ids must match the leading state dimension")
    groups = _group_tuple(group_sizes, len(paths))
    initial_mass = states.sum(dim=1)
    validation_flags = torch.stack(
        (
            torch.isfinite(states).all(),
            (states >= 0.0).all(),
            (initial_mass > 0.0).all(),
        )
    ).detach().cpu().tolist()
    if not bool(validation_flags[0]) or not bool(validation_flags[1]):
        raise ValueError("states must be finite and nonnegative")
    if not bool(validation_flags[2]):
        raise ValueError("every path state must have positive mass")

    device = states.device
    initial_states = states.detach().clone()
    values = states.detach().clone()
    matching_arrays = tuple(
        (
            torch.as_tensor(tails, dtype=torch.int64, device=device).contiguous(),
            torch.as_tensor(heads, dtype=torch.int64, device=device).contiguous(),
        )
        for tails, heads in _controls._matching_arrays()
    )
    group_ranges: list[tuple[int, int]] = []
    group_start = 0
    for size in groups:
        group_ranges.append((group_start, group_start + size))
        group_start += size

    path_count = len(paths)
    per_path_certified = torch.zeros(path_count, dtype=torch.int64, device=device)
    per_path_fallback = torch.zeros(path_count, dtype=torch.int64, device=device)
    per_path_strengthened = torch.zeros(path_count, dtype=torch.int64, device=device)
    per_path_maximum_mode = torch.zeros(path_count, dtype=torch.int32, device=device)
    per_path_maximum_prefix = torch.zeros(path_count, dtype=torch.int32, device=device)
    maximum_cuda_launch_lanes = torch.zeros((), dtype=torch.int64, device=device)
    fused_authorizer_launch_count = torch.zeros((), dtype=torch.int64, device=device)
    fallback_seconds = torch.zeros((), dtype=torch.float64, device=device)
    fused_seconds = torch.zeros((), dtype=torch.float64, device=device)
    candidate_seconds = torch.zeros((), dtype=torch.float64, device=device)
    forbidden = {
        name: torch.zeros((), dtype=torch.int64, device=device)
        for name in _FORBIDDEN_DIAGNOSTICS
    }
    later_blocks: list[Tensor] = []
    target_blocks: list[Tensor] = []
    code_blocks: list[Tensor] = []
    mode_blocks: list[Tensor] = []
    prefix_blocks: list[Tensor] = []
    fallback_reason_blocks: list[Tensor] = []
    phase_state_blocks: list[Tensor] = []

    started = time.perf_counter()
    for local_step in range(SHARD_STEPS):
        outer_step = int(start_step) + local_step
        for phase, (matching_index, duration) in enumerate(
            zip(PHASE_MATCHINGS, PHASE_DURATIONS, strict=True)
        ):
            tails, heads = matching_arrays[matching_index]
            phase_later = torch.empty(
                (path_count, EDGES_PER_PHASE), dtype=torch.float64, device=device
            )
            phase_target = torch.empty_like(phase_later)
            phase_codes = torch.empty(
                (path_count, EDGES_PER_PHASE), dtype=torch.uint8, device=device
            )
            phase_modes = torch.empty(
                (path_count, EDGES_PER_PHASE), dtype=torch.int32, device=device
            )
            phase_prefixes = torch.empty_like(phase_modes)
            phase_fallback_reasons = torch.empty(
                (path_count, EDGES_PER_PHASE), dtype=torch.uint8, device=device
            )
            for first, last in group_ranges:
                group_values = values.narrow(0, first, last - first)
                tail_mass = group_values.index_select(1, tails)
                head_mass = group_values.index_select(1, heads)
                pair_total = tail_mass + head_mass
                positive = pair_total > 0.0
                safe_pair_total = torch.where(
                    positive, pair_total, torch.ones_like(pair_total)
                )
                current = torch.where(
                    positive,
                    head_mass / safe_pair_total,
                    torch.zeros_like(pair_total),
                ).contiguous()
                exposure = torch.where(
                    positive,
                    torch.as_tensor(
                        3.0 * (5.0e-5 / 512.0) * duration / (1.0 / 28.0) ** 2,
                        dtype=torch.float64,
                        device=device,
                    )
                    / safe_pair_total,
                    torch.zeros_like(pair_total),
                ).contiguous()
                ids = canonical_same_phase_transition_ids(
                    paths[first:last],
                    outer_step=outer_step,
                    phase=phase,
                    device=device,
                )
                lane_count = (last - first) * EDGES_PER_PHASE
                result = _controls._call_sampler(
                    current.reshape(-1).contiguous(),
                    exposure.reshape(-1).contiguous(),
                    profile=profile,
                    rng_key=(int(root_seed), "full-path-v2"),
                    transition_offset=0,
                    transition_ids=ids,
                    sampler=sampler,
                )
                shape = (last - first, EDGES_PER_PHASE)
                later = _result_tensor(
                    result, "later_head_fraction", "later", "y", shape=shape
                )
                target = _result_tensor(
                    result, "denoising_target", "target", "z", shape=shape
                )
                codes = _result_tensor(
                    result, "certificate_codes", "certificate_code", shape=shape
                )
                if any(value.device != device for value in (later, target, codes)):
                    raise _controls.RigorousCudaControlError(
                        "multipath CUDA output left the selected device"
                    )
                later = later.to(dtype=torch.float64)
                target = target.to(dtype=torch.float64)
                codes = codes.to(dtype=torch.uint8)
                fallback_mask = _optional_result_tensor(
                    result,
                    "fallback_mask",
                    shape=shape,
                    dtype=torch.bool,
                    device=device,
                )
                strengthened_mask = _optional_result_tensor(
                    result,
                    "strengthened_mask",
                    shape=shape,
                    dtype=torch.bool,
                    device=device,
                )
                mode_counts = _optional_result_tensor(
                    result,
                    "mode_counts",
                    shape=shape,
                    dtype=torch.int32,
                    device=device,
                )
                prefix_bits = _optional_result_tensor(
                    result,
                    "prefix_bits",
                    shape=shape,
                    dtype=torch.int32,
                    device=device,
                )
                fallback_reason_codes = _optional_result_tensor(
                    result,
                    "arb_fallback_reason_codes",
                    shape=shape,
                    dtype=torch.uint8,
                    device=device,
                )
                certified = (codes & 0b1111) == 0b1111
                per_path_certified[first:last] += certified.sum(dim=1, dtype=torch.int64)
                per_path_fallback[first:last] += fallback_mask.sum(
                    dim=1, dtype=torch.int64
                )
                per_path_strengthened[first:last] += strengthened_mask.sum(
                    dim=1, dtype=torch.int64
                )
                per_path_maximum_mode[first:last] = torch.maximum(
                    per_path_maximum_mode[first:last], mode_counts.max(dim=1).values
                )
                per_path_maximum_prefix[first:last] = torch.maximum(
                    per_path_maximum_prefix[first:last], prefix_bits.max(dim=1).values
                )
                fallback_seconds += _diagnostic_tensor(
                    result,
                    "arb_fallback_elapsed_seconds",
                    dtype=torch.float64,
                    device=device,
                )
                fused_seconds += _diagnostic_tensor(
                    result,
                    "fused_authorizer_elapsed_seconds",
                    dtype=torch.float64,
                    device=device,
                )
                candidate_seconds += _diagnostic_tensor(
                    result,
                    "candidate_elapsed_seconds",
                    dtype=torch.float64,
                    device=device,
                )
                maximum_cuda_launch_lanes = torch.maximum(
                    maximum_cuda_launch_lanes,
                    _diagnostic_tensor(
                        result,
                        "maximum_cuda_launch_lanes",
                        dtype=torch.int64,
                        device=device,
                    ),
                )
                fused_authorizer_launch_count += _diagnostic_tensor(
                    result,
                    "fused_authorizer_launch_count",
                    dtype=torch.int64,
                    device=device,
                )
                for name in _FORBIDDEN_DIAGNOSTICS:
                    forbidden[name] += _diagnostic_tensor(
                        result, name, dtype=torch.int64, device=device
                    )
                phase_later[first:last] = later
                phase_target[first:last] = target
                phase_codes[first:last] = codes
                phase_modes[first:last] = mode_counts
                phase_prefixes[first:last] = prefix_bits
                phase_fallback_reasons[first:last] = fallback_reason_codes
                group_values[:, tails] = pair_total * (1.0 - later)
                group_values[:, heads] = pair_total * later
                if lane_count > 4096:  # Defensive: the public API also rejects it.
                    raise AssertionError("multipath group exceeded the CUDA lane cap")
            later_blocks.append(phase_later.detach())
            target_blocks.append(phase_target.detach())
            code_blocks.append(phase_codes.detach())
            mode_blocks.append(phase_modes.detach())
            prefix_blocks.append(phase_prefixes.detach())
            fallback_reason_blocks.append(phase_fallback_reasons.detach())
            if capture_phase_state_trace:
                phase_state_blocks.append(values.detach().clone())

    # All materialization occurs after the complete eight-step shard.  Pack
    # every hash/certificate input into one float64 buffer so the summary is a
    # single physical device-to-host transfer as well as one logical boundary.
    # Small integer fields are exactly representable in float64 and are cast
    # back to their original dtypes before hashing or histogramming.  The
    # returned final_states tensor itself remains on the input device.
    phase_shape = (len(later_blocks), path_count, EDGES_PER_PHASE)
    state_shape = (path_count, PATH_STATE_SIZE)
    path_shape = (path_count,)
    later_device = torch.stack(later_blocks)
    target_device = torch.stack(target_blocks)
    codes_device = torch.stack(code_blocks)
    modes_device = torch.stack(mode_blocks)
    prefixes_device = torch.stack(prefix_blocks)
    fallback_reasons_device = torch.stack(fallback_reason_blocks)
    phase_states_device = (
        torch.stack(phase_state_blocks)
        if capture_phase_state_trace
        else torch.empty(
            (0, path_count, PATH_STATE_SIZE),
            dtype=torch.float64,
            device=device,
        )
    )
    scalar_device = torch.stack(
        [
            maximum_cuda_launch_lanes.to(torch.float64),
            fused_authorizer_launch_count.to(torch.float64),
            fallback_seconds,
            fused_seconds,
            candidate_seconds,
            *[forbidden[name].to(torch.float64) for name in _FORBIDDEN_DIAGNOSTICS],
        ]
    )
    packed_host = torch.cat(
        (
            later_device.reshape(-1),
            target_device.reshape(-1),
            codes_device.reshape(-1).to(torch.float64),
            modes_device.reshape(-1).to(torch.float64),
            prefixes_device.reshape(-1).to(torch.float64),
            fallback_reasons_device.reshape(-1).to(torch.float64),
            phase_states_device.reshape(-1),
            initial_states.reshape(-1),
            values.reshape(-1),
            per_path_certified.to(torch.float64),
            per_path_fallback.to(torch.float64),
            per_path_strengthened.to(torch.float64),
            per_path_maximum_mode.to(torch.float64),
            per_path_maximum_prefix.to(torch.float64),
            scalar_device,
        )
    ).detach().cpu().numpy()
    elapsed = time.perf_counter() - started

    packed_offset = 0

    def unpack(shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
        nonlocal packed_offset
        count = math.prod(shape)
        result = packed_host[packed_offset : packed_offset + count].reshape(shape)
        packed_offset += count
        return result.astype(dtype, copy=False)

    later_host = unpack(phase_shape, np.dtype(np.float64))
    target_host = unpack(phase_shape, np.dtype(np.float64))
    codes_host = unpack(phase_shape, np.dtype(np.uint8))
    modes_host = unpack(phase_shape, np.dtype(np.int32))
    prefixes_host = unpack(phase_shape, np.dtype(np.int32))
    fallback_reasons_host = unpack(phase_shape, np.dtype(np.uint8))
    phase_states_host = unpack(
        (int(phase_states_device.shape[0]), path_count, PATH_STATE_SIZE),
        np.dtype(np.float64),
    )
    initial_host = unpack(state_shape, np.dtype(np.float64))
    final_host = unpack(state_shape, np.dtype(np.float64))
    final_host.setflags(write=False)
    certified_host = unpack(path_shape, np.dtype(np.int64))
    fallback_host = unpack(path_shape, np.dtype(np.int64))
    strengthened_host = unpack(path_shape, np.dtype(np.int64))
    mode_host = unpack(path_shape, np.dtype(np.int32))
    prefix_host = unpack(path_shape, np.dtype(np.int32))
    scalar_values = unpack((int(scalar_device.numel()),), np.dtype(np.float64))
    if packed_offset != int(packed_host.size):
        raise AssertionError("multipath summary buffer was not fully decoded")

    mass_errors = np.abs(final_host.sum(axis=1) - initial_host.sum(axis=1))
    maximum_mass_error = float(np.max(mass_errors))
    if not np.isfinite(final_host).all() or np.any(final_host < 0.0):
        raise _controls.RigorousCudaControlError(
            "multipath shard produced an invalid final state"
        )
    if maximum_mass_error > 2.0e-12:
        raise _controls.RigorousCudaControlError(
            "multipath shard failed per-path mass conservation"
        )

    transitions_per_path = SHARD_STEPS * len(PHASE_MATCHINGS) * EDGES_PER_PHASE
    path_records: list[ExactMultipathPathRecord] = []
    for path_index, path_id in enumerate(paths):
        output_digest = hashlib.sha256()
        for block in range(len(later_blocks)):
            output_digest.update(
                bytes.fromhex(
                    _controls._digest_arrays(
                        later_host[block, path_index],
                        target_host[block, path_index],
                        codes_host[block, path_index],
                    )
                )
            )
        input_hash = _controls._digest_arrays(initial_host[path_index])
        output_hash = output_digest.hexdigest()
        final_hash = _controls._digest_arrays(final_host[path_index])
        certificate_code_counts = _integer_counts(codes_host[:, path_index, :])
        mode_count_counts = _integer_counts(modes_host[:, path_index, :])
        prefix_bit_counts = _integer_counts(prefixes_host[:, path_index, :])
        fallback_reason_counts = _integer_counts(
            fallback_reasons_host[:, path_index, :]
        )
        certificate_hash = _fingerprint(
            {
                "version": MULTIPATH_SCHEDULER_VERSION,
                "path_id": path_id,
                "start_step": int(start_step),
                "step_count": SHARD_STEPS,
                "input_state_sha256": input_hash,
                "output_sha256": output_hash,
                "final_state_sha256": final_hash,
                "certified_count": int(certified_host[path_index]),
                "fallback_count": int(fallback_host[path_index]),
                "certificate_code_counts": certificate_code_counts,
                "mode_count_counts": mode_count_counts,
                "prefix_bit_counts": prefix_bit_counts,
                "arb_fallback_reason_code_counts": fallback_reason_counts,
            }
        )
        path_records.append(
            ExactMultipathPathRecord(
                path_id=path_id,
                transition_count=transitions_per_path,
                certified_count=int(certified_host[path_index]),
                fallback_count=int(fallback_host[path_index]),
                strengthened_count=int(strengthened_host[path_index]),
                maximum_mode_count=int(mode_host[path_index]),
                maximum_prefix_bits=int(prefix_host[path_index]),
                certificate_code_counts=certificate_code_counts,
                mode_count_counts=mode_count_counts,
                prefix_bit_counts=prefix_bit_counts,
                arb_fallback_reason_code_counts=fallback_reason_counts,
                input_state_sha256=input_hash,
                output_sha256=output_hash,
                final_state_sha256=final_hash,
                certificate_sha256=certificate_hash,
            )
        )

    canonical_indices = sorted(range(path_count), key=lambda index: paths[index])
    canonical_records = sorted(path_records, key=lambda record: record.path_id)
    phase_state_records: list[ExactMultipathPhaseStateRecord] = []
    for block_index, phase_states in enumerate(phase_states_host):
        local_step, phase = divmod(block_index, len(PHASE_MATCHINGS))
        per_path_hashes = tuple(
            (
                paths[index],
                _controls._digest_arrays(phase_states[index]),
            )
            for index in canonical_indices
        )
        phase_state_records.append(
            ExactMultipathPhaseStateRecord(
                outer_step=int(start_step) + local_step,
                phase=phase,
                path_state_sha256_by_id=per_path_hashes,
                batch_state_sha256=_controls._digest_arrays(
                    phase_states[canonical_indices]
                ),
            )
        )
    batch_output_hash = _fingerprint(
        [[record.path_id, record.output_sha256] for record in canonical_records]
    )
    batch_final_hash = _controls._digest_arrays(final_host[canonical_indices])
    batch_certificate_hash = _fingerprint(
        {
            "version": MULTIPATH_SCHEDULER_VERSION,
            "start_step": int(start_step),
            "step_count": SHARD_STEPS,
            "group_sizes": list(groups),
            "path_certificates": [
                [record.path_id, record.certificate_sha256]
                for record in canonical_records
            ],
            "batch_output_sha256": batch_output_hash,
            "batch_final_state_sha256": batch_final_hash,
        }
    )
    transition_count = path_count * transitions_per_path
    scalar_names = (
        "maximum_cuda_launch_lanes",
        "fused_authorizer_launch_count",
        "fallback_elapsed_seconds",
        "fused_authorizer_elapsed_seconds",
        "candidate_elapsed_seconds",
        *_FORBIDDEN_DIAGNOSTICS,
    )
    scalars = dict(zip(scalar_names, scalar_values.tolist(), strict=True))
    diagnostics: dict[str, Any] = {
        "version": MULTIPATH_SCHEDULER_VERSION,
        "path_count": path_count,
        "path_ids": list(paths),
        "group_sizes": list(groups),
        "start_step": int(start_step),
        "step_count": SHARD_STEPS,
        "phase_count": SHARD_STEPS * len(PHASE_MATCHINGS),
        "backend_call_count": SHARD_STEPS * len(PHASE_MATCHINGS) * len(groups),
        "transition_count": transition_count,
        "maximum_backend_call_size": max(groups) * EDGES_PER_PHASE,
        "maximum_cuda_launch_lanes": int(scalars["maximum_cuda_launch_lanes"]),
        "fused_authorizer_launch_count": int(
            scalars["fused_authorizer_launch_count"]
        ),
        "certified_count": int(np.sum(certified_host)),
        "uncertified_count": transition_count - int(np.sum(certified_host)),
        "fallback_count": int(np.sum(fallback_host)),
        "strengthened_count": int(np.sum(strengthened_host)),
        "maximum_mode_count": int(np.max(mode_host)),
        "maximum_prefix_bits": int(np.max(prefix_host)),
        "certificate_code_counts": _integer_counts(codes_host),
        "mode_count_counts": _integer_counts(modes_host),
        "prefix_bit_counts": _integer_counts(prefixes_host),
        "arb_fallback_reason_code_counts": _integer_counts(
            fallback_reasons_host
        ),
        "fallback_elapsed_seconds": float(scalars["fallback_elapsed_seconds"]),
        "fused_authorizer_elapsed_seconds": float(
            scalars["fused_authorizer_elapsed_seconds"]
        ),
        "candidate_elapsed_seconds": float(scalars["candidate_elapsed_seconds"]),
        "maximum_mass_error": maximum_mass_error,
        "state_updates_device_resident": 1,
        "in_shard_host_roundtrip_count": 0,
        "evolving_state_host_roundtrip_count": 0,
        "phase_state_trace_enabled": int(bool(capture_phase_state_trace)),
        "phase_state_trace_record_count": len(phase_state_records),
        "pre_shard_input_validation_synchronization_count": 1,
        "shard_summary_logical_boundary_count": 1,
        "shard_summary_device_to_host_transfer_count": 1,
        "shard_summary_synchronization_count": 1,
        "synchronization_count_scope": (
            "one packed post-shard summary transfer; excludes one pre-shard "
            "input-validation synchronization and backend-internal instrumentation"
        ),
        "elapsed_seconds": elapsed,
        "transitions_per_second": (
            transition_count / elapsed if elapsed > 0.0 else math.inf
        ),
        **{
            name: int(scalars[name])
            for name in _FORBIDDEN_DIAGNOSTICS
        },
    }
    return ExactMultipathShardResult(
        final_states=values,
        committed_final_states=final_host,
        path_records=tuple(path_records),
        phase_state_records=tuple(phase_state_records),
        batch_output_sha256=batch_output_hash,
        batch_final_state_sha256=batch_final_hash,
        batch_certificate_sha256=batch_certificate_hash,
        diagnostics=diagnostics,
    )


def run_frozen_projection_shard(
    states: Tensor,
    *,
    start_step: int,
    root_seed: int,
    profile: JacobiRBCudaProfile,
    path_ids: Sequence[int] = tuple(range(FROZEN_PROJECTION_PATH_COUNT)),
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> ExactMultipathShardResult:
    """Run the frozen 64-path ``[10,10,10,10,10,10,4]`` schedule."""

    if int(states.shape[0]) != FROZEN_PROJECTION_PATH_COUNT:
        raise ValueError("the frozen projection shard requires exactly 64 paths")
    return run_exact_multipath_shard(
        states,
        path_ids=path_ids,
        start_step=start_step,
        root_seed=root_seed,
        profile=profile,
        group_sizes=FROZEN_PROJECTION_GROUP_SIZES,
        sampler=sampler,
        step_count=SHARD_STEPS,
    )


def run_stateful_multipath_shard(
    states: Tensor,
    path_ids: Sequence[int],
    *,
    start_step: int,
    step_count: int,
    root_seed: int,
    profile: JacobiRBCudaProfile,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> ExactMultipathShardResult:
    """Public plan-facing wrapper for one same-phase cohort.

    The production scheduler advances cohorts of at most ten paths.  Keeping
    this small wrapper separate from ``run_exact_multipath_shard`` makes that
    contract explicit while retaining the latter's test-only regrouping hook.
    """

    if int(states.shape[0]) > MAX_PATHS_PER_GROUP:
        raise ValueError("a stateful multipath cohort contains at most ten paths")
    return run_exact_multipath_shard(
        states,
        path_ids=path_ids,
        start_step=start_step,
        step_count=step_count,
        root_seed=root_seed,
        profile=profile,
        group_sizes=(int(states.shape[0]),),
        sampler=sampler,
    )


__all__ = [
    "MULTIPATH_SCHEDULER_VERSION",
    "MAX_PATHS_PER_GROUP",
    "SHARD_STEPS",
    "FROZEN_VALIDATION_GROUP_SIZES",
    "FROZEN_PROJECTION_GROUP_SIZES",
    "FROZEN_PROJECTION_PATH_COUNT",
    "ExactMultipathPathRecord",
    "ExactMultipathPhaseStateRecord",
    "ExactMultipathShardResult",
    "canonical_same_phase_transition_ids",
    "run_exact_multipath_shard",
    "run_stateful_multipath_shard",
    "run_frozen_projection_shard",
]
