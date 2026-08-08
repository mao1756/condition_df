"""Resumable eager-prefix execution and physical boundary-tangent caches.

The scientific transition is delegated to the existing exact multipath and
fused-midpoint schedulers.  This module owns only the production cohort walk,
eight-step restart commits, physical role separation, and aggregate evidence.
Confirmation uses :func:`iter_eager_shards` directly; only the
train/validation consumer is allowed to persist raw midpoint labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_artifacts import (
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    FORBIDDEN_DIAGNOSTICS,
    MIDPOINT_COUNT,
    MIDPOINT_FRACTIONS,
    ROOT_SEED,
    SELECTED_OUTER_STEPS,
    MidpointBranchBatch,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_prefix_fallback import (
    sample_alpha1_rb_transition_batch_cuda_eager,
)
from mnist.d0_jacobi_rb_boundary_tangent_prefix_schedule import (
    eager_prefix_contract,
    eager_prefix_profile,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule import (
    CONFIRMATION_COHORT_SIZES,
    EDGES_PER_PHASE,
    MAXIMUM_LAUNCH_LANES,
    PHASE_COUNT,
    PROJECTED_BASE_TRANSITIONS,
    PROJECTED_MIDPOINT_TRANSITIONS,
    PROJECTED_TOTAL_TRANSITIONS,
    SHARD_STEPS,
    TRAIN_VALIDATION_COHORT_SIZES,
    FusedMidpointBranchBatch,
    build_fused_launch_plan,
    frozen_production_cohort_plan,
    sample_fused_midpoint_branches,
    split_payload_by_path_roles,
)
from mnist.d0_jacobi_rb_cuda_multipath import run_exact_multipath_shard
from mnist.d0_jacobi_rb_learnability import (
    OUTER_STEPS,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
)
from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time


EAGER_CACHE_VERSION = "d0-jacobi-rb-boundary-tangent-eager-cache-v1"
COHORT_KINDS = ("train_validation", "confirmation")
CACHE_ROLES = ("train", "validation")


class EagerCacheError(ValueError):
    """The frozen eager execution or cache contract was violated."""


def _callable_name(value: Callable[..., Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _profile_record(profile: Any) -> dict[str, Any]:
    if hasattr(profile, "to_dict"):
        value = profile.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise EagerCacheError("the eager profile has no JSON record")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return config_fingerprint(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )


def _json(path: Path) -> dict[str, Any]:
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EagerCacheError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise EagerCacheError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                **{
                    str(name): np.ascontiguousarray(value)
                    for name, value in arrays.items()
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": path,
        "size": int(path.stat().st_size),
        "sha256": file_fingerprint(path),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise EagerCacheError(f"cannot read NPZ artifact: {path}") from exc


def _semantic_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _validate_semantic_record(value: Mapping[str, Any]) -> None:
    expected = value.get("semantic_sha256")
    body = {key: item for key, item in value.items() if key != "semantic_sha256"}
    if expected != config_fingerprint(body):
        raise EagerCacheError("artifact semantic fingerprint changed")


@dataclass(frozen=True)
class EagerCohort:
    kind: str
    index: int
    path_ids: tuple[int, ...]
    path_roles: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "index": self.index,
            "path_ids": list(self.path_ids),
            "path_roles": list(self.path_roles),
            "size": len(self.path_ids),
        }


@dataclass(frozen=True)
class EagerShardIdentity:
    cohort_kind: str
    cohort_index: int
    start_step: int
    step_count: int = SHARD_STEPS

    def __post_init__(self) -> None:
        if self.cohort_kind not in COHORT_KINDS:
            raise EagerCacheError("unknown eager cohort kind")
        if self.cohort_index < 0:
            raise EagerCacheError("cohort index must be nonnegative")
        if (
            self.step_count != SHARD_STEPS
            or self.start_step < 0
            or self.start_step % SHARD_STEPS
            or self.start_step + SHARD_STEPS > OUTER_STEPS
        ):
            raise EagerCacheError("restart identity is not an exact eight-step shard")

    def to_record(self) -> dict[str, Any]:
        return {
            "cohort_kind": self.cohort_kind,
            "cohort_index": self.cohort_index,
            "start_step": self.start_step,
            "step_count": self.step_count,
        }


@dataclass(frozen=True)
class EagerBranchExecution:
    outer_step: int
    phase: int
    pre_phase_states: Tensor = field(repr=False, compare=False)
    batch: Any = field(repr=False, compare=False)
    record: Mapping[str, Any]
    maximum_mass_error: float


@dataclass(frozen=True)
class EagerShardExecution:
    identity: EagerShardIdentity
    path_ids: tuple[int, ...]
    path_roles: tuple[str, ...]
    selected_step: int | None
    input_state_sha256: str
    final_states: Tensor = field(repr=False, compare=False)
    committed_final_states: np.ndarray = field(repr=False, compare=False)
    base_record: Mapping[str, Any]
    branches: tuple[EagerBranchExecution, ...] = field(repr=False, compare=False)
    diagnostics: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": EAGER_CACHE_VERSION + "-execution",
            "schema_version": 1,
            "identity": self.identity.to_record(),
            "path_ids": list(self.path_ids),
            "path_roles": list(self.path_roles),
            "selected_step": self.selected_step,
            "input_state_sha256": self.input_state_sha256,
            "base_record": dict(self.base_record),
            "branch_records": [dict(item.record) for item in self.branches],
            "diagnostics": dict(self.diagnostics),
            "raw_payload_persisted": 0,
        }


def frozen_cache_cohorts(kind: str) -> tuple[EagerCohort, ...]:
    """Return the exact ``[10x9,6]`` or ``[10x6,4]`` production plan."""

    if kind not in COHORT_KINDS:
        raise EagerCacheError("cohort kind must be train_validation or confirmation")
    plan = frozen_production_cohort_plan()
    groups = tuple(tuple(int(path) for path in group) for group in plan[kind])
    expected = (
        TRAIN_VALIDATION_COHORT_SIZES
        if kind == "train_validation"
        else CONFIRMATION_COHORT_SIZES
    )
    if tuple(map(len, groups)) != tuple(expected):
        raise EagerCacheError("frozen production cohort sizes changed")
    roles = plan["path_roles"]
    return tuple(
        EagerCohort(
            kind=kind,
            index=index,
            path_ids=group,
            path_roles=tuple(str(roles[str(path)]) for path in group),
        )
        for index, group in enumerate(groups)
    )


def frozen_eager_cache_plan() -> dict[str, Any]:
    """Return both immutable production plans and their persistence policy."""

    return _semantic_record(
        {
            "schema": EAGER_CACHE_VERSION + "-plan",
            "schema_version": 1,
            "train_validation": [
                cohort.to_record() for cohort in frozen_cache_cohorts("train_validation")
            ],
            "confirmation": [
                cohort.to_record() for cohort in frozen_cache_cohorts("confirmation")
            ],
            "train_validation_sizes": list(TRAIN_VALIDATION_COHORT_SIZES),
            "confirmation_sizes": list(CONFIRMATION_COHORT_SIZES),
            "shard_steps": SHARD_STEPS,
            "confirmation_execution_mode": "streaming_only",
            "confirmation_raw_label_persistence": 0,
            "cross_role_artifact_commit": 0,
        }
    )


def explicit_eager_cache_plan(
    cohorts: Sequence[EagerCohort],
) -> dict[str, Any]:
    """Return the canonical semantic plan for an explicit cohort sequence."""

    values = tuple(cohorts)
    if not values:
        raise EagerCacheError("an explicit cohort plan cannot be empty")
    kinds = {str(value.kind) for value in values}
    if len(kinds) != 1 or not kinds.issubset(COHORT_KINDS):
        raise EagerCacheError("explicit cohorts must share one supported kind")
    if tuple(value.index for value in values) != tuple(range(len(values))):
        raise EagerCacheError("explicit cohort indices are not canonical")
    seen: set[int] = set()
    role_counts: Counter[str] = Counter()
    for cohort in values:
        if not 1 <= len(cohort.path_ids) <= 10:
            raise EagerCacheError("explicit cohort size lies outside [1,10]")
        if len(cohort.path_roles) != len(cohort.path_ids):
            raise EagerCacheError("explicit path roles do not align")
        if any(not isinstance(path, int) or isinstance(path, bool) for path in cohort.path_ids):
            raise EagerCacheError("explicit path IDs must be integers")
        if any(path < 0 or path >= (1 << 20) for path in cohort.path_ids):
            raise EagerCacheError("explicit path ID lies outside 20 bits")
        if len(set(cohort.path_ids)) != len(cohort.path_ids) or seen.intersection(
            cohort.path_ids
        ):
            raise EagerCacheError("explicit cohort path IDs collide")
        if any(not str(role) for role in cohort.path_roles):
            raise EagerCacheError("explicit path role is empty")
        if cohort.kind == "train_validation" and not set(cohort.path_roles).issubset(
            CACHE_ROLES
        ):
            raise EagerCacheError("train/validation cohort has another role")
        if cohort.kind == "confirmation" and set(cohort.path_roles) != {
            "confirmation"
        }:
            raise EagerCacheError("confirmation cohort role changed")
        seen.update(cohort.path_ids)
        role_counts.update(cohort.path_roles)
    return _semantic_record(
        {
            "schema": EAGER_CACHE_VERSION + "-explicit-plan",
            "schema_version": 1,
            "cohort_kind": values[0].kind,
            "cohorts": [value.to_record() for value in values],
            "role_counts": dict(sorted(role_counts.items())),
            "cohort_sizes": [len(value.path_ids) for value in values],
            "path_count": len(seen),
            "cross_role_artifact_commit": 0,
        }
    )


def _validated_explicit_cohorts(
    cohorts: Sequence[EagerCohort], cohort_plan_sha256: str
) -> tuple[EagerCohort, ...]:
    values = tuple(cohorts)
    plan = explicit_eager_cache_plan(values)
    if str(cohort_plan_sha256) != str(plan["semantic_sha256"]):
        raise EagerCacheError("explicit cohort-plan fingerprint changed")
    return values


def eager_execution_contract_for_cohorts(
    *,
    cohorts: Sequence[EagerCohort],
    cohort_plan_sha256: str,
    root_seed: int = ROOT_SEED,
    outer_steps: int = OUTER_STEPS,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    profile: Any | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda_eager,
    shard_runner: Callable[..., Any] = run_exact_multipath_shard,
    branch_runner: Callable[..., Any] = sample_fused_midpoint_branches,
) -> dict[str, Any]:
    """Bind execution to a caller-supplied, hash-validated cohort plan."""

    _validated_explicit_cohorts(cohorts, cohort_plan_sha256)
    active_profile = eager_prefix_profile() if profile is None else profile
    steps = _validated_steps(outer_steps, selected_steps)
    return _semantic_record(
        {
            "schema": EAGER_CACHE_VERSION + "-execution-contract",
            "schema_version": 1,
            "root_seed": int(root_seed),
            "outer_steps": int(outer_steps),
            "selected_outer_steps": list(steps),
            "shard_steps": SHARD_STEPS,
            "profile": _profile_record(active_profile),
            "eager_prefix_contract": eager_prefix_contract(),
            "sampler": _callable_name(sampler),
            "shard_runner": _callable_name(shard_runner),
            "branch_runner": _callable_name(branch_runner),
            "cohort_plan_sha256": str(cohort_plan_sha256),
            "base_transition_law_changed": 0,
            "midpoint_transition_law_changed": 0,
        }
    )


def eager_execution_contract(
    *,
    root_seed: int = ROOT_SEED,
    outer_steps: int = OUTER_STEPS,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    profile: Any | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda_eager,
    shard_runner: Callable[..., Any] = run_exact_multipath_shard,
    branch_runner: Callable[..., Any] = sample_fused_midpoint_branches,
) -> dict[str, Any]:
    active_profile = eager_prefix_profile() if profile is None else profile
    steps = _validated_steps(outer_steps, selected_steps)
    return _semantic_record(
        {
            "schema": EAGER_CACHE_VERSION + "-execution-contract",
            "schema_version": 1,
            "root_seed": int(root_seed),
            "outer_steps": int(outer_steps),
            "selected_outer_steps": list(steps),
            "shard_steps": SHARD_STEPS,
            "profile": _profile_record(active_profile),
            "eager_prefix_contract": eager_prefix_contract(),
            "sampler": _callable_name(sampler),
            "shard_runner": _callable_name(shard_runner),
            "branch_runner": _callable_name(branch_runner),
            "cohort_plan_sha256": frozen_eager_cache_plan()["semantic_sha256"],
            "base_transition_law_changed": 0,
            "midpoint_transition_law_changed": 0,
        }
    )


def _validated_steps(
    outer_steps: int, selected_steps: Sequence[int]
) -> tuple[int, ...]:
    if (
        isinstance(outer_steps, bool)
        or int(outer_steps) != outer_steps
        or int(outer_steps) < SHARD_STEPS
        or int(outer_steps) > OUTER_STEPS
        or int(outer_steps) % SHARD_STEPS
    ):
        raise EagerCacheError("outer steps must be an eight-step prefix of K=512")
    raw_steps = tuple(selected_steps)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        for value in raw_steps
    ):
        raise EagerCacheError("selected outer steps must be integers")
    steps = tuple(int(value) for value in raw_steps if int(value) < int(outer_steps))
    if len(steps) != len(set(steps)) or tuple(sorted(steps)) != steps or any(
        value < 0 for value in steps
    ):
        raise EagerCacheError("selected outer steps are malformed")
    for start in range(0, int(outer_steps), SHARD_STEPS):
        if sum(start <= value < start + SHARD_STEPS for value in steps) > 1:
            raise EagerCacheError("an eight-step shard contains multiple observations")
    return steps


def _selected_step(
    start_step: int, selected_steps: Sequence[int]
) -> int | None:
    values = [
        value
        for value in selected_steps
        if start_step <= int(value) < start_step + SHARD_STEPS
    ]
    if len(values) > 1:
        raise EagerCacheError("an eight-step shard contains multiple observations")
    return None if not values else int(values[0])


def _result_record(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_record"):
        record = value.to_record()
        if isinstance(record, Mapping):
            return dict(record)
    raise EagerCacheError("scheduler result has no JSON evidence record")


def _branch_batch(value: Any) -> Any:
    return getattr(value, "batch", value)


def _capture_pre_phase_states(
    result: Any,
    *,
    input_states: Tensor,
    path_ids: tuple[int, ...],
    selected_step: int,
    start_step: int,
) -> tuple[np.ndarray, ...]:
    capture = getattr(result, "capture_payload", None)
    if capture is None:
        raise EagerCacheError("selected shard omitted its phase-state trace")
    if tuple(int(value) for value in capture.path_ids) != path_ids:
        raise EagerCacheError("captured path order changed")
    trace = np.asarray(capture.post_phase_states, dtype=np.float64)
    expected_shape = (SHARD_STEPS * PHASE_COUNT, len(path_ids), STATE_SIZE)
    if trace.shape != expected_shape:
        raise EagerCacheError("captured phase-state trace has the wrong shape")
    local_step = selected_step - start_step
    initial = input_states.detach().cpu().numpy()
    output: list[np.ndarray] = []
    for phase in range(PHASE_COUNT):
        if phase:
            value = trace[local_step * PHASE_COUNT + phase - 1]
        elif local_step:
            value = trace[local_step * PHASE_COUNT - 1]
        else:
            value = initial
        output.append(np.array(value, dtype=np.float64, order="C", copy=True))
    return tuple(output)


def _merge_histogram(target: Counter[int], raw: Mapping[str, Any]) -> None:
    for key, value in raw.items():
        target[int(key)] += int(value)


def _execution_diagnostics(
    *,
    identity: EagerShardIdentity,
    path_count: int,
    base_record: Mapping[str, Any],
    branches: Sequence[EagerBranchExecution],
    elapsed_seconds: float,
    device: torch.device,
) -> dict[str, Any]:
    base = base_record.get("diagnostics")
    if not isinstance(base, Mapping):
        raise EagerCacheError("base scheduler diagnostics are missing")
    expected_base = path_count * SHARD_STEPS * PHASE_COUNT * EDGES_PER_PHASE
    base_count = int(base.get("transition_count", -1))
    if base_count != expected_base:
        raise EagerCacheError("base scheduler transition count changed")
    expected_branch = path_count * MIDPOINT_COUNT * EDGES_PER_PHASE
    branch_count = 0
    branch_certified = 0
    branch_fallback = 0
    branch_strengthened = 0
    branch_fallback_seconds = 0.0
    branch_backend_seconds = 0.0
    branch_candidate_seconds = 0.0
    forbidden = Counter(
        {name: int(base.get(name, 0)) for name in FORBIDDEN_DIAGNOSTICS}
    )
    prefix_histogram: Counter[int] = Counter()
    _merge_histogram(prefix_histogram, base.get("prefix_bit_counts", {}))
    maximum_lanes = int(base.get("maximum_cuda_launch_lanes", 0))
    maximum_mass_error = float(base.get("maximum_mass_error", math.inf))
    for branch in branches:
        record = branch.record
        transitions = int(record.get("transition_count", -1))
        if transitions != expected_branch:
            raise EagerCacheError("fused midpoint transition count changed")
        branch_count += transitions
        branch_certified += int(record.get("certified_count", 0))
        branch_fallback += int(record.get("fallback_count", 0))
        branch_strengthened += int(record.get("strengthened_count", 0))
        branch_fallback_seconds += float(record.get("fallback_elapsed_seconds", 0.0))
        branch_backend_seconds += float(record.get("backend_elapsed_seconds", 0.0))
        branch_candidate_seconds += float(record.get("candidate_elapsed_seconds", 0.0))
        maximum_lanes = max(maximum_lanes, int(record.get("maximum_launch_lanes", 0)))
        maximum_mass_error = max(maximum_mass_error, branch.maximum_mass_error)
        _merge_histogram(prefix_histogram, record.get("prefix_bit_histogram", {}))
        for name, value in record.get("forbidden_counts", {}).items():
            forbidden[str(name)] += int(value)
    expected_midpoint = (
        path_count * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
        if branches
        else 0
    )
    if len(branches) not in (0, PHASE_COUNT) or branch_count != expected_midpoint:
        raise EagerCacheError("selected shard does not contain exactly seven branches")
    base_certified = int(base.get("certified_count", 0))
    total = base_count + branch_count
    certified = base_certified + branch_certified
    if certified != total:
        raise EagerCacheError("eager shard contains uncertified transitions")
    if not math.isfinite(maximum_mass_error) or maximum_mass_error > 2.0e-12:
        raise EagerCacheError("eager shard failed mass conservation")
    if maximum_lanes < 1 or maximum_lanes > MAXIMUM_LAUNCH_LANES:
        raise EagerCacheError("eager shard exceeded the launch cap")
    if device.type == "cuda":
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        peak_bytes = 0
        total_bytes = 1
    return {
        "schema": EAGER_CACHE_VERSION + "-shard-diagnostics",
        "identity": identity.to_record(),
        "path_count": path_count,
        "base_transition_count": base_count,
        "midpoint_transition_count": branch_count,
        "transition_count": total,
        "certified_count": certified,
        "uncertified_count": total - certified,
        "fallback_count": int(base.get("fallback_count", 0)) + branch_fallback,
        "strengthened_count": int(base.get("strengthened_count", 0))
        + branch_strengthened,
        "fallback_elapsed_seconds": float(base.get("fallback_elapsed_seconds", 0.0))
        + branch_fallback_seconds,
        "backend_elapsed_seconds": float(
            base.get("fused_authorizer_elapsed_seconds", 0.0)
        )
        + branch_backend_seconds,
        "candidate_elapsed_seconds": float(base.get("candidate_elapsed_seconds", 0.0))
        + branch_candidate_seconds,
        "complete_pipeline_elapsed_seconds": float(elapsed_seconds),
        "maximum_mass_error": maximum_mass_error,
        "maximum_launch_lanes": maximum_lanes,
        "maximum_peak_memory_bytes": peak_bytes,
        "device_total_memory_bytes": total_bytes,
        "peak_memory_fraction": peak_bytes / total_bytes,
        "prefix_bit_histogram": {
            str(key): value for key, value in sorted(prefix_histogram.items())
        },
        "forbidden_counts": {
            name: int(forbidden[name]) for name in FORBIDDEN_DIAGNOSTICS
        },
        "eager_sampler_injected": 1,
        "approximate_transition_used": 0,
    }


def execute_eager_shard(
    states: Tensor,
    *,
    cohort: EagerCohort,
    start_step: int,
    root_seed: int = ROOT_SEED,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    profile: Any | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda_eager,
    shard_runner: Callable[..., Any] = run_exact_multipath_shard,
    branch_runner: Callable[..., Any] = sample_fused_midpoint_branches,
) -> EagerShardExecution:
    """Execute one exact eager eight-step shard without writing artifacts."""

    identity = EagerShardIdentity(cohort.kind, cohort.index, int(start_step))
    if (
        not isinstance(states, Tensor)
        or states.dtype != torch.float64
        or states.shape != (len(cohort.path_ids), STATE_SIZE)
        or not states.is_contiguous()
    ):
        raise EagerCacheError("cohort states must be contiguous float64 [P,784]")
    active_profile = eager_prefix_profile() if profile is None else profile
    selected = _selected_step(identity.start_step, selected_steps)
    input_snapshot = states.detach().clone()
    started = time.perf_counter()
    result = shard_runner(
        states,
        path_ids=cohort.path_ids,
        start_step=identity.start_step,
        root_seed=int(root_seed),
        profile=active_profile,
        group_sizes=(len(cohort.path_ids),),
        sampler=sampler,
        step_count=SHARD_STEPS,
        capture_phase_state_trace=selected is not None,
        capture_training_payload=selected is not None,
    )
    committed = np.ascontiguousarray(
        getattr(result, "committed_final_states"), dtype=np.float64
    )
    final_states = getattr(result, "final_states", None)
    if (
        committed.shape != states.shape
        or not isinstance(final_states, Tensor)
        or final_states.shape != states.shape
        or final_states.dtype != torch.float64
        or final_states.device != states.device
        or not final_states.is_contiguous()
    ):
        raise EagerCacheError("base scheduler returned malformed continuation states")
    device_committed = np.ascontiguousarray(
        final_states.detach().cpu().numpy(), dtype=np.float64
    )
    if (
        not np.isfinite(committed).all()
        or not np.isfinite(device_committed).all()
        or not np.array_equal(device_committed, committed)
    ):
        raise EagerCacheError(
            "base scheduler device and committed continuation states differ"
        )
    branches: list[EagerBranchExecution] = []
    if selected is not None:
        pre_states = _capture_pre_phase_states(
            result,
            input_states=input_snapshot,
            path_ids=cohort.path_ids,
            selected_step=selected,
            start_step=identity.start_step,
        )
        for phase, pre_array in enumerate(pre_states):
            pre = torch.as_tensor(
                pre_array, dtype=torch.float64, device=states.device
            ).contiguous()
            branch_started = time.perf_counter()
            value = branch_runner(
                pre,
                path_ids=cohort.path_ids,
                outer_step=selected,
                phase=phase,
                root_seed=int(root_seed),
                profile=active_profile,
                sampler=sampler,
            )
            record = _result_record(value)
            record.setdefault(
                "complete_call_elapsed_seconds", time.perf_counter() - branch_started
            )
            batch = _branch_batch(value)
            later = np.asarray(batch.later_full_state.detach().cpu().numpy())
            mass_error = float(
                np.max(
                    np.abs(
                        later.sum(axis=2)
                        - np.asarray(pre_array).sum(axis=1)[None, :]
                    )
                )
            )
            branches.append(
                EagerBranchExecution(
                    outer_step=selected,
                    phase=phase,
                    pre_phase_states=pre,
                    batch=value,
                    record=record,
                    maximum_mass_error=mass_error,
                )
            )
    base_record = _result_record(result)
    diagnostics = _execution_diagnostics(
        identity=identity,
        path_count=len(cohort.path_ids),
        base_record=base_record,
        branches=branches,
        elapsed_seconds=time.perf_counter() - started,
        device=states.device,
    )
    return EagerShardExecution(
        identity=identity,
        path_ids=cohort.path_ids,
        path_roles=cohort.path_roles,
        selected_step=selected,
        input_state_sha256=_array_sha256(input_snapshot.detach().cpu().numpy()),
        final_states=final_states,
        committed_final_states=committed,
        base_record=base_record,
        branches=tuple(branches),
        diagnostics=diagnostics,
    )


def _cohort_initial_states(
    initial_state: np.ndarray | Tensor,
    path_count: int,
    device: torch.device,
) -> Tensor:
    if isinstance(initial_state, Tensor):
        source = initial_state.detach().to(device=device, dtype=torch.float64)
    else:
        source = torch.as_tensor(initial_state, dtype=torch.float64, device=device)
    if source.shape != (STATE_SIZE,):
        raise EagerCacheError("initial state must have shape [784]")
    if not bool(torch.isfinite(source).all()) or not bool((source >= 0.0).all()):
        raise EagerCacheError("initial state must be finite and nonnegative")
    if not bool(source.sum() > 0.0):
        raise EagerCacheError("initial state must have positive mass")
    return source.reshape(1, STATE_SIZE).repeat(path_count, 1).contiguous()


def iter_eager_shards(
    initial_state: np.ndarray | Tensor,
    *,
    cohort_kind: str,
    device: str | torch.device,
    root_seed: int = ROOT_SEED,
    outer_steps: int = OUTER_STEPS,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    cohort_indices: Sequence[int] | None = None,
    profile: Any | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda_eager,
    shard_runner: Callable[..., Any] = run_exact_multipath_shard,
    branch_runner: Callable[..., Any] = sample_fused_midpoint_branches,
) -> Iterator[EagerShardExecution]:
    """Stream frozen cohort shards without creating any filesystem artifact."""

    selected = _validated_steps(outer_steps, selected_steps)
    cohorts = frozen_cache_cohorts(cohort_kind)
    indices = (
        tuple(range(len(cohorts)))
        if cohort_indices is None
        else tuple(int(value) for value in cohort_indices)
    )
    if len(indices) != len(set(indices)) or any(
        value < 0 or value >= len(cohorts) for value in indices
    ):
        raise EagerCacheError("cohort indices are malformed")
    active_device = torch.device(device)
    if active_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(active_device)
    for cohort_index in indices:
        cohort = cohorts[cohort_index]
        states = _cohort_initial_states(initial_state, len(cohort.path_ids), active_device)
        for start_step in range(0, int(outer_steps), SHARD_STEPS):
            execution = execute_eager_shard(
                states,
                cohort=cohort,
                start_step=start_step,
                root_seed=root_seed,
                selected_steps=selected,
                profile=profile,
                sampler=sampler,
                shard_runner=shard_runner,
                branch_runner=branch_runner,
            )
            next_states = execution.final_states.detach().clone().contiguous()
            yield execution
            states = next_states


def iter_eager_shards_for_cohorts(
    initial_state: np.ndarray | Tensor,
    *,
    cohorts: Sequence[EagerCohort],
    cohort_plan_sha256: str,
    device: str | torch.device,
    root_seed: int = ROOT_SEED,
    outer_steps: int = OUTER_STEPS,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    cohort_indices: Sequence[int] | None = None,
    profile: Any | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda_eager,
    shard_runner: Callable[..., Any] = run_exact_multipath_shard,
    branch_runner: Callable[..., Any] = sample_fused_midpoint_branches,
) -> Iterator[EagerShardExecution]:
    """Stream a caller-supplied, hash-bound cohort plan."""

    selected = _validated_steps(outer_steps, selected_steps)
    values = _validated_explicit_cohorts(cohorts, cohort_plan_sha256)
    indices = (
        tuple(range(len(values)))
        if cohort_indices is None
        else tuple(int(value) for value in cohort_indices)
    )
    if len(indices) != len(set(indices)) or any(
        value < 0 or value >= len(values) for value in indices
    ):
        raise EagerCacheError("cohort indices are malformed")
    active_device = torch.device(device)
    if active_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(active_device)
    for cohort_index in indices:
        cohort = values[cohort_index]
        states = _cohort_initial_states(
            initial_state, len(cohort.path_ids), active_device
        )
        for start_step in range(0, int(outer_steps), SHARD_STEPS):
            execution = execute_eager_shard(
                states,
                cohort=cohort,
                start_step=start_step,
                root_seed=root_seed,
                selected_steps=selected,
                profile=profile,
                sampler=sampler,
                shard_runner=shard_runner,
                branch_runner=branch_runner,
            )
            yield execution
            states = execution.final_states.detach().clone().contiguous()


def _branch_arrays(
    execution: EagerShardExecution,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if execution.selected_step is None or len(execution.branches) != PHASE_COUNT:
        raise EagerCacheError("non-selected shard has no branch arrays")
    paths = execution.path_ids
    path_count = len(paths)
    later_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    code_blocks: list[np.ndarray] = []
    for phase, branch in enumerate(execution.branches):
        if branch.phase != phase:
            raise EagerCacheError("branch phase order changed")
        batch = _branch_batch(branch.batch)
        later_blocks.append(batch.later_full_state.detach().cpu().numpy())
        target_blocks.append(batch.denoising_target.detach().cpu().numpy())
        code_blocks.append(batch.certificate_codes.detach().cpu().numpy())
    # Scheduler layout is [phase, midpoint, path, ...].  Persistence layout is
    # [path, phase, midpoint, ...], so the role splitter always sees paths first.
    later = np.stack(later_blocks).transpose(2, 0, 1, 3)
    target = np.stack(target_blocks).transpose(2, 0, 1, 3)
    codes = np.stack(code_blocks).transpose(2, 0, 1, 3)
    grid_shape = (path_count, PHASE_COUNT, MIDPOINT_COUNT)
    path_grid = np.broadcast_to(
        np.asarray(paths, dtype=np.int64)[:, None, None], grid_shape
    ).copy()
    step_grid = np.full(grid_shape, execution.selected_step, dtype=np.int16)
    phase_grid = np.broadcast_to(
        np.arange(PHASE_COUNT, dtype=np.int8)[None, :, None], grid_shape
    ).copy()
    midpoint_grid = np.broadcast_to(
        np.arange(MIDPOINT_COUNT, dtype=np.int8)[None, None, :], grid_shape
    ).copy()
    fraction_grid = np.broadcast_to(
        np.asarray(MIDPOINT_FRACTIONS, dtype=np.float64)[None, None, :], grid_shape
    ).copy()
    keys = np.empty(grid_shape, dtype=np.int64)
    reverse_time = np.empty(grid_shape, dtype=np.float64)
    for path_index, path_id in enumerate(paths):
        for phase in range(PHASE_COUNT):
            for midpoint_index, fraction in enumerate(MIDPOINT_FRACTIONS):
                keys[path_index, phase, midpoint_index] = midpoint_sample_key(
                    path_id, execution.selected_step, phase, midpoint_index
                )
                reverse_time[path_index, phase, midpoint_index] = internal_reverse_time(
                    execution.selected_step, phase, fraction
                )
    identity = {
        "sample_key": keys,
        "path_id": path_grid,
        "outer_step": step_grid,
        "phase": phase_grid,
        "midpoint_index": midpoint_grid,
        "midpoint_fraction": fraction_grid,
    }
    inputs = {
        **identity,
        "later_full_state": np.ascontiguousarray(later, dtype=np.float32),
        "reverse_time": reverse_time,
        "color": np.asarray(
            [PHASE_MATCHINGS[int(value)] for value in phase_grid.reshape(-1)],
            dtype=np.int8,
        ).reshape(grid_shape),
        "duration": np.asarray(
            [PHASE_DURATIONS[int(value)] for value in phase_grid.reshape(-1)],
            dtype=np.float64,
        ).reshape(grid_shape),
        "label": np.full(grid_shape, 3, dtype=np.int64),
    }
    labels = {
        **identity,
        "denoising_target": np.ascontiguousarray(target, dtype=np.float64),
        "certificate_codes": np.ascontiguousarray(codes, dtype=np.uint8),
    }
    return inputs, labels


def _shard_dir(run_dir: Path, identity: EagerShardIdentity) -> Path:
    return (
        run_dir
        / "eager_cache"
        / identity.cohort_kind
        / f"cohort-{identity.cohort_index:03d}"
        / f"shard-{identity.start_step:06d}"
    )


def _metadata_path(run_dir: Path, identity: EagerShardIdentity) -> Path:
    return _shard_dir(run_dir, identity) / "metadata.json"


def _relative_artifact(run_dir: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(artifact["path"])
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "size": int(artifact["size"]),
        "sha256": str(artifact["sha256"]),
    }


def persist_eager_shard(
    run_dir: str | Path,
    execution: EagerShardExecution,
    *,
    execution_contract_sha256: str,
) -> dict[str, Any]:
    """Commit one train/validation shard; metadata is the atomic commit point."""

    if execution.identity.cohort_kind != "train_validation":
        raise EagerCacheError("confirmation raw labels are streaming-only")
    root = Path(run_dir).resolve()
    shard_dir = _shard_dir(root, execution.identity)
    persistence_started = time.perf_counter()
    state_payload = split_payload_by_path_roles(
        execution.path_ids,
        execution.path_roles,
        {
            "path_ids": np.asarray(execution.path_ids, dtype=np.int64),
            "final_states": execution.committed_final_states,
        },
    )
    input_payload: dict[str, dict[str, Tensor | np.ndarray]] = {}
    label_payload: dict[str, dict[str, Tensor | np.ndarray]] = {}
    if execution.selected_step is not None:
        inputs, labels = _branch_arrays(execution)
        input_payload = split_payload_by_path_roles(
            execution.path_ids, execution.path_roles, inputs
        )
        label_payload = split_payload_by_path_roles(
            execution.path_ids, execution.path_roles, labels
        )
    role_artifacts: dict[str, Any] = {}
    for role, state_arrays in state_payload.items():
        if role not in CACHE_ROLES:
            raise EagerCacheError("train/validation shard contains another role")
        role_dir = shard_dir / role
        state_artifact = _relative_artifact(
            root,
            _atomic_npz(
                role_dir / "continuation_state.npz",
                {name: np.asarray(value) for name, value in state_arrays.items()},
            ),
        )
        input_artifact = None
        label_artifact = None
        if execution.selected_step is not None:
            input_artifact = _relative_artifact(
                root,
                _atomic_npz(
                    role_dir / "branch_inputs.npz",
                    {name: np.asarray(value) for name, value in input_payload[role].items()},
                ),
            )
            label_artifact = _relative_artifact(
                root,
                _atomic_npz(
                    role_dir / "branch_labels.npz",
                    {name: np.asarray(value) for name, value in label_payload[role].items()},
                ),
            )
        role_paths = [
            path
            for path, path_role in zip(
                execution.path_ids, execution.path_roles, strict=True
            )
            if path_role == role
        ]
        role_artifacts[role] = {
            "path_ids": role_paths,
            "path_count": len(role_paths),
            "continuation_state": state_artifact,
            "branch_inputs": input_artifact,
            "branch_labels": label_artifact,
            "physical_role": role,
        }
    payload_bytes = sum(
        int(artifact["size"])
        for role in role_artifacts.values()
        for artifact in (
            role["continuation_state"],
            role["branch_inputs"],
            role["branch_labels"],
        )
        if artifact is not None
    )
    record = _semantic_record(
        {
            "schema": EAGER_CACHE_VERSION + "-atomic-shard",
            "schema_version": 1,
            "identity": execution.identity.to_record(),
            "path_ids": list(execution.path_ids),
            "path_roles": list(execution.path_roles),
            "selected_step": execution.selected_step,
            "input_state_sha256": execution.input_state_sha256,
            "final_state_sha256": _array_sha256(execution.committed_final_states),
            "execution_contract_sha256": str(execution_contract_sha256),
            "base_record": dict(execution.base_record),
            "branch_records": [dict(branch.record) for branch in execution.branches],
            "diagnostics": dict(execution.diagnostics),
            "role_artifacts": role_artifacts,
            "payload_bytes": payload_bytes,
            "persistence_elapsed_seconds": time.perf_counter()
            - persistence_started,
            "continuation_state_role_separated": 1,
            "branch_input_label_separated": 1,
            "cross_role_artifact_commit": 0,
            "committed": 1,
        }
    )
    # This write is deliberately last.  Payloads without this valid metadata
    # are orphans and are overwritten when the cohort tail is recomputed.
    atomic_write_json(_metadata_path(root, execution.identity), record)
    return record


def _verify_artifact(run_dir: Path, artifact: Mapping[str, Any]) -> Path:
    path = run_dir / str(artifact["path"])
    if (
        not path.is_file()
        or int(path.stat().st_size) != int(artifact["size"])
        or file_fingerprint(path) != artifact["sha256"]
    ):
        raise EagerCacheError(f"cache artifact changed: {path}")
    return path


def _load_valid_shard(
    run_dir: Path,
    *,
    cohort: EagerCohort,
    start_step: int,
    selected_step: int | None,
    current_states: np.ndarray,
    execution_contract_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    identity = EagerShardIdentity(cohort.kind, cohort.index, start_step)
    metadata_path = _metadata_path(run_dir, identity)
    try:
        record = _json(metadata_path)
        _validate_semantic_record(record)
        if (
            record.get("schema") != EAGER_CACHE_VERSION + "-atomic-shard"
            or int(record.get("committed", 0)) != 1
            or record.get("identity") != identity.to_record()
            or tuple(record.get("path_ids", ())) != cohort.path_ids
            or tuple(record.get("path_roles", ())) != cohort.path_roles
            or record.get("selected_step") != selected_step
            or record.get("input_state_sha256") != _array_sha256(current_states)
            or record.get("execution_contract_sha256")
            != execution_contract_sha256
        ):
            raise EagerCacheError("atomic shard identity or predecessor changed")
        role_artifacts = record.get("role_artifacts")
        expected_roles = set(cohort.path_roles)
        if not isinstance(role_artifacts, Mapping) or set(role_artifacts) != expected_roles:
            raise EagerCacheError("atomic shard role artifacts changed")
        states_by_path: dict[int, np.ndarray] = {}
        for role, value in role_artifacts.items():
            if not isinstance(value, Mapping) or value.get("physical_role") != role:
                raise EagerCacheError("cache artifact lost its physical role")
            state_arrays = _load_npz(
                _verify_artifact(run_dir, value["continuation_state"])
            )
            role_paths = tuple(int(path) for path in value["path_ids"])
            if (
                tuple(int(path) for path in state_arrays["path_ids"]) != role_paths
                or state_arrays["final_states"].shape
                != (len(role_paths), STATE_SIZE)
            ):
                raise EagerCacheError("continuation state payload changed")
            for path, state in zip(
                role_paths, state_arrays["final_states"], strict=True
            ):
                states_by_path[path] = np.asarray(state, dtype=np.float64)
            input_artifact = value.get("branch_inputs")
            label_artifact = value.get("branch_labels")
            if selected_step is None:
                if input_artifact is not None or label_artifact is not None:
                    raise EagerCacheError("non-selected shard contains branch payload")
            else:
                # Resume checks the label commitment but deliberately does not
                # decode raw labels.  Only load_eager_role_labels crosses that
                # explicit firewall.
                arrays = _load_npz(_verify_artifact(run_dir, input_artifact))
                _verify_artifact(run_dir, label_artifact)
                if arrays["path_id"].shape != (
                    len(role_paths),
                    PHASE_COUNT,
                    MIDPOINT_COUNT,
                ) or set(np.unique(arrays["path_id"]).tolist()) != set(role_paths):
                    raise EagerCacheError("branch payload crossed a physical role")
        final = np.stack([states_by_path[path] for path in cohort.path_ids]).astype(
            np.float64, copy=False
        )
        if record.get("final_state_sha256") != _array_sha256(final):
            raise EagerCacheError("reassembled continuation state changed")
        return np.ascontiguousarray(final), record
    except (EagerCacheError, KeyError, TypeError, ValueError):
        return None


def _metadata_artifact(run_dir: Path, identity: EagerShardIdentity) -> dict[str, Any]:
    path = _metadata_path(run_dir, identity)
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "size": int(path.stat().st_size),
        "sha256": file_fingerprint(path),
    }


def _role_index_path(run_dir: Path, role: str) -> Path:
    return run_dir / "eager_cache" / f"{role}_index.json"


def _write_role_indexes(
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    outer_steps: int,
    selected_steps: Sequence[int],
    execution_contract_sha256: str,
) -> dict[str, dict[str, Any]]:
    indexes: dict[str, dict[str, Any]] = {}
    for role in CACHE_ROLES:
        entries: list[dict[str, Any]] = []
        paths: set[int] = set()
        base_count = 0
        midpoint_count = 0
        row_count = 0
        for record in records:
            role_artifacts = record["role_artifacts"]
            if role not in role_artifacts:
                continue
            value = role_artifacts[role]
            role_paths = tuple(int(path) for path in value["path_ids"])
            paths.update(role_paths)
            identity = EagerShardIdentity(**record["identity"])
            base_count += len(role_paths) * SHARD_STEPS * PHASE_COUNT * EDGES_PER_PHASE
            if record["selected_step"] is not None:
                midpoint_count += (
                    len(role_paths) * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
                )
                row_count += len(role_paths) * PHASE_COUNT * MIDPOINT_COUNT
            entries.append(
                {
                    "identity": identity.to_record(),
                    "selected_step": record["selected_step"],
                    "path_ids": list(role_paths),
                    "metadata": _metadata_artifact(run_dir, identity),
                    "continuation_state": dict(value["continuation_state"]),
                    "branch_inputs": None
                    if value["branch_inputs"] is None
                    else dict(value["branch_inputs"]),
                    "branch_labels": None
                    if value["branch_labels"] is None
                    else dict(value["branch_labels"]),
                }
            )
        if not entries:
            continue
        entries.sort(
            key=lambda value: (
                int(value["identity"]["cohort_index"]),
                int(value["identity"]["start_step"]),
            )
        )
        index = _semantic_record(
            {
                "schema": EAGER_CACHE_VERSION + "-role-index",
                "schema_version": 1,
                "role": role,
                "cohort_kind": "train_validation",
                "path_ids": sorted(paths),
                "path_count": len(paths),
                "outer_steps": int(outer_steps),
                "selected_outer_steps": list(selected_steps),
                "execution_contract_sha256": execution_contract_sha256,
                "entries": entries,
                "base_transition_count": base_count,
                "midpoint_transition_count": midpoint_count,
                "transition_count": base_count + midpoint_count,
                "input_row_count": row_count,
                "label_row_count": row_count,
                "branch_input_label_separated": 1,
                "continuation_state_role_separated": 1,
                "cross_role_artifact_commit": 0,
            }
        )
        atomic_write_json(_role_index_path(run_dir, role), index)
        indexes[role] = index
    return indexes


def _record_view(value: EagerShardExecution | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, EagerShardExecution):
        return {
            "identity": value.identity.to_record(),
            "path_ids": list(value.path_ids),
            "selected_step": value.selected_step,
            "base_record": dict(value.base_record),
            "branch_records": [dict(branch.record) for branch in value.branches],
            "diagnostics": dict(value.diagnostics),
        }
    return dict(value)


def _aggregate_eager_diagnostics_with_cohorts(
    values: Iterable[EagerShardExecution | Mapping[str, Any]],
    *,
    cohorts: Sequence[EagerCohort],
    cohort_kind: str,
    outer_steps: int,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    cohort_indices: Sequence[int] | None = None,
    persisted_bytes: int = 0,
) -> dict[str, Any]:
    """Aggregate exact transition counts and resource evidence."""

    selected = _validated_steps(outer_steps, selected_steps)
    cohorts = tuple(cohorts)
    indices = (
        tuple(range(len(cohorts)))
        if cohort_indices is None
        else tuple(int(value) for value in cohort_indices)
    )
    records = [_record_view(value) for value in values]
    identities = [
        (
            str(record["identity"]["cohort_kind"]),
            int(record["identity"]["cohort_index"]),
            int(record["identity"]["start_step"]),
        )
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise EagerCacheError("aggregate contains duplicate shard identities")
    expected_identities = {
        (cohort_kind, index, start)
        for index in indices
        for start in range(0, int(outer_steps), SHARD_STEPS)
    }
    if set(identities) != expected_identities:
        raise EagerCacheError("aggregate does not cover the requested shard plan")
    role_transition_counts: Counter[str] = Counter()
    role_row_counts: Counter[str] = Counter()
    role_elapsed_seconds: Counter[str] = Counter()
    for record in records:
        identity = record["identity"]
        cohort = cohorts[int(identity["cohort_index"])]
        if tuple(int(path) for path in record["path_ids"]) != cohort.path_ids:
            raise EagerCacheError("aggregate shard paths changed")
        elapsed_value = float(
            record["diagnostics"].get("complete_pipeline_elapsed_seconds", 0.0)
        )
        for role in set(cohort.path_roles):
            count = cohort.path_roles.count(role)
            role_transition_counts[role] += (
                count * SHARD_STEPS * PHASE_COUNT * EDGES_PER_PHASE
            )
            if record.get("selected_step") is not None:
                role_transition_counts[role] += (
                    count * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
                )
                role_row_counts[role] += count * PHASE_COUNT * MIDPOINT_COUNT
            # Charging the whole co-scheduled call to every participating role
            # is conservative and avoids inventing an unmeasured time split.
            role_elapsed_seconds[role] += elapsed_value
    path_count = sum(len(cohorts[index].path_ids) for index in indices)
    expected_base = path_count * int(outer_steps) * PHASE_COUNT * EDGES_PER_PHASE
    expected_midpoint = (
        path_count * len(selected) * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
    )
    base_count = sum(int(record["diagnostics"]["base_transition_count"]) for record in records)
    midpoint_count = sum(
        int(record["diagnostics"]["midpoint_transition_count"]) for record in records
    )
    certified_count = sum(
        int(record["diagnostics"]["certified_count"]) for record in records
    )
    if (base_count, midpoint_count) != (expected_base, expected_midpoint):
        raise EagerCacheError("aggregate transition count changed")
    total = base_count + midpoint_count
    if certified_count != total:
        raise EagerCacheError("aggregate contains uncertified transitions")
    forbidden = {
        name: sum(
            int(record["diagnostics"].get("forbidden_counts", {}).get(name, 0))
            for record in records
        )
        for name in FORBIDDEN_DIAGNOSTICS
    }
    elapsed = sum(
        float(record["diagnostics"].get("complete_pipeline_elapsed_seconds", 0.0))
        for record in records
    )
    role_rates = {
        role: (
            role_transition_counts[role] / role_elapsed_seconds[role]
            if role_elapsed_seconds[role] > 0.0
            else None
        )
        for role in sorted(role_transition_counts)
    }
    finite_role_rates = [value for value in role_rates.values() if value is not None]
    fallback_count = sum(
        int(record["diagnostics"].get("fallback_count", 0))
        for record in records
    )
    fallback_elapsed = sum(
        float(record["diagnostics"].get("fallback_elapsed_seconds", 0.0))
        for record in records
    )
    maximum_peak_fraction = max(
        float(record["diagnostics"].get("peak_memory_fraction", 0.0))
        for record in records
    )
    approximate_transition_used = max(
        int(record["diagnostics"].get("approximate_transition_used", 0))
        for record in records
    )
    forbidden_event_count = (
        sum(forbidden.values())
        + (total - certified_count)
        + approximate_transition_used
    )
    complete_plan = set(indices) == set(range(len(cohorts)))
    return _semantic_record(
        {
            "schema": EAGER_CACHE_VERSION + "-aggregate",
            "schema_version": 1,
            "cohort_kind": cohort_kind,
            "cohort_indices": list(indices),
            "cohort_sizes": [len(cohorts[index].path_ids) for index in indices],
            "path_ids": [
                path for index in indices for path in cohorts[index].path_ids
            ],
            "path_count": path_count,
            "shard_count": len(records),
            "outer_steps": int(outer_steps),
            "selected_outer_steps": list(selected),
            "base_transition_count": base_count,
            "midpoint_transition_count": midpoint_count,
            "transition_count": total,
            "certified_count": certified_count,
            "uncertified_count": 0,
            "branch_row_count": path_count
            * len(selected)
            * PHASE_COUNT
            * MIDPOINT_COUNT,
            "role_transition_counts": dict(sorted(role_transition_counts.items())),
            "role_branch_row_counts": dict(sorted(role_row_counts.items())),
            "role_complete_pipeline_elapsed_seconds": dict(
                sorted(role_elapsed_seconds.items())
            ),
            "role_transitions_per_second": role_rates,
            "minimum_role_rate": min(finite_role_rates) if finite_role_rates else None,
            "certificate_fraction": certified_count / total,
            "fallback_count": fallback_count,
            "fallback_fraction": fallback_count / total,
            "fallback_elapsed_seconds": fallback_elapsed,
            "fallback_time_fraction": fallback_elapsed / elapsed if elapsed > 0.0 else None,
            "complete_pipeline_elapsed_seconds": elapsed,
            "transitions_per_second": total / elapsed if elapsed > 0.0 else None,
            "maximum_mass_error": max(
                float(record["diagnostics"]["maximum_mass_error"])
                for record in records
            ),
            "maximum_launch_lanes": max(
                int(record["diagnostics"]["maximum_launch_lanes"])
                for record in records
            ),
            "maximum_peak_memory_bytes": max(
                int(record["diagnostics"].get("maximum_peak_memory_bytes", 0))
                for record in records
            ),
            "maximum_peak_memory_fraction": maximum_peak_fraction,
            "peak_memory_fraction": maximum_peak_fraction,
            "persisted_bytes": int(persisted_bytes),
            "total_persisted_cache_bytes": int(persisted_bytes),
            "forbidden_counts": forbidden,
            "forbidden_event_count": forbidden_event_count,
            "approximate_transition_used": approximate_transition_used,
            "complete_frozen_cohort_plan": int(complete_plan),
            "exact_count_checks_passed": 1,
            "raw_label_persistence": int(cohort_kind == "train_validation"),
        }
    )


def aggregate_eager_diagnostics(
    values: Iterable[EagerShardExecution | Mapping[str, Any]],
    *,
    cohort_kind: str,
    outer_steps: int,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    cohort_indices: Sequence[int] | None = None,
    persisted_bytes: int = 0,
) -> dict[str, Any]:
    """Historical v2 aggregate over the frozen cohort plan."""

    return _aggregate_eager_diagnostics_with_cohorts(
        values,
        cohorts=frozen_cache_cohorts(cohort_kind),
        cohort_kind=cohort_kind,
        outer_steps=outer_steps,
        selected_steps=selected_steps,
        cohort_indices=cohort_indices,
        persisted_bytes=persisted_bytes,
    )


def aggregate_eager_diagnostics_for_cohorts(
    values: Iterable[EagerShardExecution | Mapping[str, Any]],
    *,
    cohorts: Sequence[EagerCohort],
    cohort_plan_sha256: str,
    outer_steps: int,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    cohort_indices: Sequence[int] | None = None,
    persisted_bytes: int = 0,
) -> dict[str, Any]:
    """Aggregate an explicit, hash-bound cohort execution."""

    values_cohorts = _validated_explicit_cohorts(cohorts, cohort_plan_sha256)
    return _aggregate_eager_diagnostics_with_cohorts(
        values,
        cohorts=values_cohorts,
        cohort_kind=values_cohorts[0].kind,
        outer_steps=outer_steps,
        selected_steps=selected_steps,
        cohort_indices=cohort_indices,
        persisted_bytes=persisted_bytes,
    )


@dataclass
class EagerDiagnosticsAccumulator:
    """Incremental aggregation for the streaming confirmation consumer."""

    cohort_kind: str
    outer_steps: int = OUTER_STEPS
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS
    cohort_indices: Sequence[int] | None = None
    _records: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def add(self, value: EagerShardExecution | Mapping[str, Any]) -> None:
        self._records.append(_record_view(value))

    def to_record(self, *, persisted_bytes: int = 0) -> dict[str, Any]:
        return aggregate_eager_diagnostics(
            self._records,
            cohort_kind=self.cohort_kind,
            outer_steps=self.outer_steps,
            selected_steps=self.selected_steps,
            cohort_indices=self.cohort_indices,
            persisted_bytes=persisted_bytes,
        )


def combine_eager_metrics(*records: Mapping[str, Any]) -> dict[str, Any]:
    """Combine train/validation and confirmation resource/count records."""

    if not records:
        raise EagerCacheError("at least one aggregate is required")
    base = sum(int(record["base_transition_count"]) for record in records)
    midpoint = sum(int(record["midpoint_transition_count"]) for record in records)
    total = base + midpoint
    kinds = {str(record["cohort_kind"]) for record in records}
    full_production = (
        kinds == set(COHORT_KINDS)
        and all(int(record.get("complete_frozen_cohort_plan", 0)) == 1 for record in records)
        and all(int(record.get("outer_steps", 0)) == OUTER_STEPS for record in records)
        and all(
            tuple(record.get("selected_outer_steps", ())) == SELECTED_OUTER_STEPS
            for record in records
        )
    )
    if full_production and (
        base != PROJECTED_BASE_TRANSITIONS
        or midpoint != PROJECTED_MIDPOINT_TRANSITIONS
        or total != PROJECTED_TOTAL_TRANSITIONS
    ):
        raise EagerCacheError("full production transition projection changed")
    names = set().union(
        *(record.get("forbidden_counts", {}).keys() for record in records)
    )
    elapsed = sum(float(record["complete_pipeline_elapsed_seconds"]) for record in records)
    certified = sum(int(record["certified_count"]) for record in records)
    fallback = sum(int(record["fallback_count"]) for record in records)
    fallback_elapsed = sum(
        float(record["fallback_elapsed_seconds"]) for record in records
    )
    forbidden = {
        name: sum(
            int(record.get("forbidden_counts", {}).get(name, 0))
            for record in records
        )
        for name in sorted(names)
    }
    peak_fraction = max(
        float(record["maximum_peak_memory_fraction"]) for record in records
    )
    persisted = sum(int(record.get("persisted_bytes", 0)) for record in records)
    role_rates = [
        float(value)
        for record in records
        for value in record.get("role_transitions_per_second", {}).values()
        if value is not None
    ]
    approximate = max(
        int(record.get("approximate_transition_used", 0)) for record in records
    )
    return _semantic_record(
        {
            "schema": EAGER_CACHE_VERSION + "-combined-metrics",
            "schema_version": 1,
            "cohort_kinds": sorted(kinds),
            "base_transition_count": base,
            "midpoint_transition_count": midpoint,
            "transition_count": total,
            "certified_count": certified,
            "uncertified_count": total - certified,
            "certificate_fraction": certified / total,
            "fallback_count": fallback,
            "fallback_fraction": fallback / total,
            "fallback_elapsed_seconds": fallback_elapsed,
            "fallback_time_fraction": fallback_elapsed / elapsed if elapsed > 0.0 else None,
            "complete_pipeline_elapsed_seconds": elapsed,
            "transitions_per_second": total / elapsed if elapsed > 0.0 else None,
            "maximum_mass_error": max(float(record["maximum_mass_error"]) for record in records),
            "maximum_launch_lanes": max(int(record["maximum_launch_lanes"]) for record in records),
            "minimum_role_rate": min(role_rates) if role_rates else None,
            "maximum_peak_memory_fraction": peak_fraction,
            "peak_memory_fraction": peak_fraction,
            "persisted_bytes": persisted,
            "total_persisted_cache_bytes": persisted,
            "forbidden_counts": forbidden,
            "forbidden_event_count": sum(forbidden.values())
            + (total - certified)
            + approximate,
            "approximate_transition_used": approximate,
            "full_production_projection": int(full_production),
            "exact_projected_counts_passed": int(full_production),
        }
    )


def _generate_eager_cache_with_cohorts(
    run_dir: str | Path,
    initial_state: np.ndarray | Tensor,
    *,
    cohorts: Sequence[EagerCohort],
    contract: Mapping[str, Any],
    device: str | torch.device,
    root_seed: int = ROOT_SEED,
    outer_steps: int = OUTER_STEPS,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    cohort_indices: Sequence[int] | None = None,
    profile: Any | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda_eager,
    shard_runner: Callable[..., Any] = run_exact_multipath_shard,
    branch_runner: Callable[..., Any] = sample_fused_midpoint_branches,
    progress: Callable[[EagerShardIdentity, str], None] | None = None,
) -> dict[str, Any]:
    """Generate or resume the physically split train/validation cache."""

    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = _validated_steps(outer_steps, selected_steps)
    cohorts = tuple(cohorts)
    if not cohorts or any(cohort.kind != "train_validation" for cohort in cohorts):
        raise EagerCacheError("persistent eager cache requires train/validation cohorts")
    indices = (
        tuple(range(len(cohorts)))
        if cohort_indices is None
        else tuple(int(value) for value in cohort_indices)
    )
    if len(indices) != len(set(indices)) or any(
        value < 0 or value >= len(cohorts) for value in indices
    ):
        raise EagerCacheError("cohort indices are malformed")
    active_profile = eager_prefix_profile() if profile is None else profile
    contract_path = root / "eager_cache" / "execution_contract.json"
    contract = dict(contract)
    _validate_semantic_record(contract)
    atomic_write_json(contract_path, contract)
    active_device = torch.device(device)
    if active_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(active_device)
    records: list[dict[str, Any]] = []
    recomputed = 0
    reused = 0
    for cohort_index in indices:
        cohort = cohorts[cohort_index]
        states = _cohort_initial_states(initial_state, len(cohort.path_ids), active_device)
        current = np.ascontiguousarray(states.detach().cpu().numpy(), dtype=np.float64)
        recompute_tail = False
        for start_step in range(0, int(outer_steps), SHARD_STEPS):
            selected_step = _selected_step(start_step, selected)
            cached = None
            if not recompute_tail:
                cached = _load_valid_shard(
                    root,
                    cohort=cohort,
                    start_step=start_step,
                    selected_step=selected_step,
                    current_states=current,
                    execution_contract_sha256=contract["semantic_sha256"],
                )
            identity = EagerShardIdentity("train_validation", cohort_index, start_step)
            if cached is not None:
                current, record = cached
                states = torch.as_tensor(
                    current, dtype=torch.float64, device=active_device
                ).contiguous()
                records.append(record)
                reused += 1
                if progress is not None:
                    progress(identity, "reused")
                continue
            recompute_tail = True
            execution = execute_eager_shard(
                states,
                cohort=cohort,
                start_step=start_step,
                root_seed=root_seed,
                selected_steps=selected,
                profile=active_profile,
                sampler=sampler,
                shard_runner=shard_runner,
                branch_runner=branch_runner,
            )
            record = persist_eager_shard(
                root,
                execution,
                execution_contract_sha256=contract["semantic_sha256"],
            )
            current = np.ascontiguousarray(execution.committed_final_states)
            states = execution.final_states.detach().clone().contiguous()
            records.append(record)
            recomputed += 1
            if progress is not None:
                progress(identity, "committed")
    indexes = _write_role_indexes(
        root,
        records,
        outer_steps=outer_steps,
        selected_steps=selected,
        execution_contract_sha256=contract["semantic_sha256"],
    )
    persisted_bytes = int(contract_path.stat().st_size)
    persisted_bytes += sum(
        int(record.get("payload_bytes", 0))
        + int(_metadata_path(root, EagerShardIdentity(**record["identity"])).stat().st_size)
        for record in records
    )
    persisted_bytes += sum(
        int(_role_index_path(root, role).stat().st_size) for role in indexes
    )
    metrics = _aggregate_eager_diagnostics_with_cohorts(
        records,
        cohorts=cohorts,
        cohort_kind="train_validation",
        outer_steps=outer_steps,
        selected_steps=selected,
        cohort_indices=indices,
        persisted_bytes=persisted_bytes,
    )
    metrics = _semantic_record(
        {
            key: value
            for key, value in metrics.items()
            if key != "semantic_sha256"
        }
        | {
            "recomputed_shard_count": recomputed,
            "reused_shard_count": reused,
        }
    )
    atomic_write_json(root / "eager_cache" / "train_validation_metrics.json", metrics)
    return {
        "schema": EAGER_CACHE_VERSION + "-generation-result",
        "execution_contract": contract,
        "metrics": metrics,
        "role_indexes": indexes,
        "recomputed_shard_count": recomputed,
        "reused_shard_count": reused,
    }


def generate_eager_cache(
    run_dir: str | Path,
    initial_state: np.ndarray | Tensor,
    *,
    device: str | torch.device,
    root_seed: int = ROOT_SEED,
    outer_steps: int = OUTER_STEPS,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    cohort_indices: Sequence[int] | None = None,
    profile: Any | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda_eager,
    shard_runner: Callable[..., Any] = run_exact_multipath_shard,
    branch_runner: Callable[..., Any] = sample_fused_midpoint_branches,
    progress: Callable[[EagerShardIdentity, str], None] | None = None,
) -> dict[str, Any]:
    """Historical v2 cache generator over the frozen cohort plan."""

    contract = eager_execution_contract(
        root_seed=root_seed,
        outer_steps=outer_steps,
        selected_steps=selected_steps,
        profile=profile,
        sampler=sampler,
        shard_runner=shard_runner,
        branch_runner=branch_runner,
    )
    return _generate_eager_cache_with_cohorts(
        run_dir,
        initial_state,
        cohorts=frozen_cache_cohorts("train_validation"),
        contract=contract,
        device=device,
        root_seed=root_seed,
        outer_steps=outer_steps,
        selected_steps=selected_steps,
        cohort_indices=cohort_indices,
        profile=profile,
        sampler=sampler,
        shard_runner=shard_runner,
        branch_runner=branch_runner,
        progress=progress,
    )


def generate_eager_cache_for_cohorts(
    run_dir: str | Path,
    initial_state: np.ndarray | Tensor,
    *,
    cohorts: Sequence[EagerCohort],
    cohort_plan_sha256: str,
    device: str | torch.device,
    root_seed: int = ROOT_SEED,
    outer_steps: int = OUTER_STEPS,
    selected_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    cohort_indices: Sequence[int] | None = None,
    profile: Any | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda_eager,
    shard_runner: Callable[..., Any] = run_exact_multipath_shard,
    branch_runner: Callable[..., Any] = sample_fused_midpoint_branches,
    progress: Callable[[EagerShardIdentity, str], None] | None = None,
) -> dict[str, Any]:
    """Generate or resume a hash-bound explicit train/validation cache."""

    values = _validated_explicit_cohorts(cohorts, cohort_plan_sha256)
    contract = eager_execution_contract_for_cohorts(
        cohorts=values,
        cohort_plan_sha256=cohort_plan_sha256,
        root_seed=root_seed,
        outer_steps=outer_steps,
        selected_steps=selected_steps,
        profile=profile,
        sampler=sampler,
        shard_runner=shard_runner,
        branch_runner=branch_runner,
    )
    return _generate_eager_cache_with_cohorts(
        run_dir,
        initial_state,
        cohorts=values,
        contract=contract,
        device=device,
        root_seed=root_seed,
        outer_steps=outer_steps,
        selected_steps=selected_steps,
        cohort_indices=cohort_indices,
        profile=profile,
        sampler=sampler,
        shard_runner=shard_runner,
        branch_runner=branch_runner,
        progress=progress,
    )


def _load_role_index(run_dir: Path, role: str) -> dict[str, Any]:
    if role not in CACHE_ROLES:
        raise EagerCacheError("cache role must be train or validation")
    value = _json(_role_index_path(run_dir, role))
    _validate_semantic_record(value)
    if value.get("schema") != EAGER_CACHE_VERSION + "-role-index" or value.get("role") != role:
        raise EagerCacheError("role index is incompatible")
    return value


def _flatten_artifact_arrays(
    arrays: Mapping[str, np.ndarray], *, vector_fields: set[str]
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if name in vector_fields:
            output[name] = np.ascontiguousarray(array.reshape(-1, array.shape[-1]))
        else:
            output[name] = np.ascontiguousarray(array.reshape(-1))
    return output


def _concat_sorted_rows(
    blocks: Sequence[Mapping[str, np.ndarray]], *, vector_fields: set[str]
) -> dict[str, np.ndarray]:
    if not blocks:
        return {}
    names = set(blocks[0])
    if any(set(block) != names for block in blocks):
        raise EagerCacheError("role artifacts have inconsistent fields")
    output = {
        name: np.concatenate([block[name] for block in blocks], axis=0)
        for name in sorted(names)
    }
    keys = np.asarray(output["sample_key"], dtype=np.int64)
    if len(np.unique(keys)) != len(keys):
        raise EagerCacheError("role sample keys collide")
    order = np.argsort(keys, kind="stable")
    return {
        name: np.ascontiguousarray(value[order])
        for name, value in output.items()
    }


def load_eager_role_inputs(
    run_dir: str | Path, role: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load permitted model inputs without opening any raw-label file."""

    root = Path(run_dir).resolve()
    index = _load_role_index(root, role)
    blocks: list[dict[str, np.ndarray]] = []
    for entry in index["entries"]:
        artifact = entry.get("branch_inputs")
        if artifact is None:
            continue
        arrays = _load_npz(_verify_artifact(root, artifact))
        blocks.append(
            _flatten_artifact_arrays(arrays, vector_fields={"later_full_state"})
        )
    output = _concat_sorted_rows(blocks, vector_fields={"later_full_state"})
    if len(output.get("sample_key", ())) != int(index["input_row_count"]):
        raise EagerCacheError("role input row count changed")
    return output, index


def load_eager_role_labels(
    run_dir: str | Path, role: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Open the physically separate labels only after the caller authorizes it."""

    root = Path(run_dir).resolve()
    index = _load_role_index(root, role)
    blocks: list[dict[str, np.ndarray]] = []
    for entry in index["entries"]:
        artifact = entry.get("branch_labels")
        if artifact is None:
            continue
        arrays = _load_npz(_verify_artifact(root, artifact))
        blocks.append(
            _flatten_artifact_arrays(
                arrays,
                vector_fields={"denoising_target", "certificate_codes"},
            )
        )
    output = _concat_sorted_rows(
        blocks, vector_fields={"denoising_target", "certificate_codes"}
    )
    if len(output.get("sample_key", ())) != int(index["label_row_count"]):
        raise EagerCacheError("role label row count changed")
    return output, index


def load_eager_role_final_states(
    run_dir: str | Path, role: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return each role path's latest physically separated continuation state."""

    root = Path(run_dir).resolve()
    index = _load_role_index(root, role)
    latest: dict[int, tuple[int, np.ndarray]] = {}
    for entry in index["entries"]:
        arrays = _load_npz(_verify_artifact(root, entry["continuation_state"]))
        start = int(entry["identity"]["start_step"])
        for path_id, state in zip(
            arrays["path_ids"], arrays["final_states"], strict=True
        ):
            path = int(path_id)
            if path not in latest or start > latest[path][0]:
                latest[path] = (start, np.asarray(state, dtype=np.float64))
    paths = np.asarray(sorted(latest), dtype=np.int64)
    states = np.stack([latest[int(path)][1] for path in paths]).astype(
        np.float64, copy=False
    )
    return paths, np.ascontiguousarray(states)


@dataclass(frozen=True)
class _DeterministicCapture:
    path_ids: tuple[int, ...]
    start_step: int
    post_phase_states: np.ndarray


@dataclass(frozen=True)
class _DeterministicShardResult:
    final_states: Tensor
    committed_final_states: np.ndarray
    diagnostics: Mapping[str, Any]
    capture_payload: _DeterministicCapture | None

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": EAGER_CACHE_VERSION + "-deterministic-test-shard",
            "diagnostics": dict(self.diagnostics),
        }


def deterministic_test_shard_runner(
    states: Tensor,
    *,
    path_ids: Sequence[int],
    start_step: int,
    root_seed: int,
    profile: Any,
    group_sizes: Sequence[int] | None = None,
    sampler: Callable[..., Any],
    step_count: int = SHARD_STEPS,
    capture_phase_state_trace: bool = False,
    capture_training_payload: bool = False,
) -> _DeterministicShardResult:
    """Fast CPU test double with production-compatible exact counts."""

    del root_seed, profile, group_sizes, sampler
    if step_count != SHARD_STEPS:
        raise EagerCacheError("test shard must contain eight steps")
    paths = tuple(int(value) for value in path_ids)
    final = torch.roll(states.detach().clone(), shifts=1, dims=1).contiguous()
    committed = np.ascontiguousarray(final.detach().cpu().numpy(), dtype=np.float64)
    transition_count = len(paths) * SHARD_STEPS * PHASE_COUNT * EDGES_PER_PHASE
    capture = None
    if capture_phase_state_trace or capture_training_payload:
        trace = np.repeat(
            states.detach().cpu().numpy()[None, :, :],
            SHARD_STEPS * PHASE_COUNT,
            axis=0,
        )
        capture = _DeterministicCapture(paths, int(start_step), trace)
    diagnostics = {
        "path_count": len(paths),
        "transition_count": transition_count,
        "certified_count": transition_count,
        "uncertified_count": 0,
        "fallback_count": 0,
        "strengthened_count": transition_count,
        "maximum_cuda_launch_lanes": len(paths) * EDGES_PER_PHASE,
        "maximum_mass_error": 0.0,
        "prefix_bit_counts": {"128": transition_count},
        "fallback_elapsed_seconds": 0.0,
        "fused_authorizer_elapsed_seconds": 0.0,
        "candidate_elapsed_seconds": 0.0,
        "elapsed_seconds": 0.001,
        **{name: 0 for name in FORBIDDEN_DIAGNOSTICS},
    }
    return _DeterministicShardResult(final, committed, diagnostics, capture)


def deterministic_test_branch_runner(
    pre_phase_states: Tensor,
    *,
    path_ids: Sequence[int],
    outer_step: int,
    phase: int,
    root_seed: int,
    profile: Any,
    sampler: Callable[..., Any],
) -> FusedMidpointBranchBatch:
    """Fast fused-midpoint test double with certified eager metadata."""

    del root_seed, profile, sampler
    paths = tuple(int(value) for value in path_ids)
    path_count = len(paths)
    edge_shape = (MIDPOINT_COUNT, path_count, EDGES_PER_PHASE)
    later_state = pre_phase_states.unsqueeze(0).repeat(MIDPOINT_COUNT, 1, 1)
    target = torch.zeros(edge_shape, dtype=torch.float64, device=pre_phase_states.device)
    codes = torch.full(edge_shape, 15, dtype=torch.uint8, device=pre_phase_states.device)
    modes = torch.ones(edge_shape, dtype=torch.int32, device=pre_phase_states.device)
    prefixes = torch.full(edge_shape, 128, dtype=torch.int32, device=pre_phase_states.device)
    flags = torch.zeros(edge_shape, dtype=torch.bool, device=pre_phase_states.device)
    batch = MidpointBranchBatch(
        path_ids=paths,
        outer_step=int(outer_step),
        phase=int(phase),
        midpoint_fractions=MIDPOINT_FRACTIONS,
        later_full_state=later_state,
        later_head_fraction=torch.zeros_like(target),
        denoising_target=target,
        certificate_codes=codes,
        mode_counts=modes,
        prefix_bits=prefixes,
        fallback_mask=flags,
        strengthened_mask=torch.ones_like(flags),
        transition_count=MIDPOINT_COUNT * path_count * EDGES_PER_PHASE,
        forbidden_counts={name: 0 for name in FORBIDDEN_DIAGNOSTICS},
        fallback_elapsed_seconds=0.0,
        backend_elapsed_seconds=0.0,
    )
    plan = build_fused_launch_plan(path_count)
    return FusedMidpointBranchBatch(
        batch=batch,
        launch_plan=plan,
        launch_count=len(plan.chunk_ranges),
        fallback_reason_codes=torch.zeros(
            edge_shape, dtype=torch.uint8, device=pre_phase_states.device
        ),
        candidate_elapsed_seconds=0.0,
        reported_authorizer_launch_count=len(plan.chunk_ranges),
        reported_maximum_launch_lanes=plan.maximum_chunk_lanes,
    )


__all__ = [
    "CACHE_ROLES",
    "COHORT_KINDS",
    "EAGER_CACHE_VERSION",
    "EagerBranchExecution",
    "EagerCacheError",
    "EagerCohort",
    "EagerDiagnosticsAccumulator",
    "EagerShardExecution",
    "EagerShardIdentity",
    "aggregate_eager_diagnostics",
    "combine_eager_metrics",
    "deterministic_test_branch_runner",
    "deterministic_test_shard_runner",
    "eager_execution_contract",
    "execute_eager_shard",
    "frozen_cache_cohorts",
    "frozen_eager_cache_plan",
    "generate_eager_cache",
    "iter_eager_shards",
    "load_eager_role_final_states",
    "load_eager_role_inputs",
    "load_eager_role_labels",
    "persist_eager_shard",
]
