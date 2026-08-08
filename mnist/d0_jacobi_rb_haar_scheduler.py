r"""Exact hierarchical scheduling for certified Haar-coupled Jacobi shards.

This module is deliberately additive.  It does not change the variable-\(K\)
Strang scheduler, the Dynkin observer, or the Jacobi transition authorizer.
Instead it supplies the existing Dynkin runner with a sampler whose uniforms
come from a certified Haar tree.

There are two supported schedules:

``nested_haar_single_arm``
    One common Haar tree for either the main levels
    ``(128, 256, 512, 1024)`` or the independent reference levels
    ``(512, 1024, 2048)``.

``pairwise_haar_antithetic``
    One pair-local tree for an adjacent pair ``(K, 2K)``.  The coarse branch
    is run once and the two fine branches use opposite signs for the single
    refinement-detail level.  Consequently both fine pairs aggregate to the
    same coarse normal while retaining exact fine-level marginals.

An aligned scheduler shard spans eight steps of the coarsest level.  Finer
levels therefore execute the corresponding number of immutable eight-step
Dynkin shards.  State, exact Rao--Blackwell labels, raw observables, Dynkin
observables, and certificates all remain owned by the unchanged parent
runners.

The production normal transform in ``d0_jacobi_rb_haar`` uses a fused CUDA
double-double certificate with transition-local Arb escalation.  Production
entry points still fail *before* state evolution unless that builder
advertises the frozen contract, and they verify its measured fallback and
certificate diagnostics on every shard.  Tests may exercise scheduling with
explicitly nonauthorizing injected doubles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_dynkin import (
    DynkinAccumulatorState,
    run_dynkin_refinement_shard,
)
from mnist.d0_jacobi_rb_haar import (
    HaarCouplingProfile,
    HaarEventIdentity,
    build_certified_haar_uniform_batch,
    canonical_haar_transition_id,
    validate_role_path_id,
)
from mnist.d0_jacobi_rb_haar_cuda import (
    sample_alpha1_rb_transition_batch_cuda_from_uniform_cells,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    MAX_REFINEMENT_PATHS_PER_GROUP,
    PHASE_MATCHINGS,
    REFINEMENT_SHARD_STEPS,
)


HAAR_SCHEDULER_VERSION = "d0-jacobi-rb-certified-haar-scheduler-v1"
HAAR_SHARD_SCHEMA = HAAR_SCHEDULER_VERSION + "-shard"
NESTED_PROFILE_NAME = "nested_haar_single_arm"
ANTITHETIC_PROFILE_NAME = "pairwise_haar_antithetic"
NESTED_MAIN_LEVELS = (128, 256, 512, 1024)
NESTED_REFERENCE_LEVELS = (512, 1024, 2048)
ADJACENT_LEVEL_PAIRS = (
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
)
MAX_NORMAL_FALLBACK_FRACTION = 1.0e-4
MAX_NORMAL_FALLBACK_TIME_FRACTION = 0.10
OBSERVABLE_COUNT = 10


class HaarSchedulerError(RuntimeError):
    """A scheduler, backend, or immutable checkpoint contract failed."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        failure_domain: str = "hierarchical_scheduler",
        **diagnostics: Any,
    ) -> None:
        super().__init__(message)
        self.failure_code = str(failure_code)
        self.failure_domain = str(failure_domain)
        self.diagnostics = dict(diagnostics)


def _normalize_interval_adapter_tensor(
    value: Tensor | None,
    *,
    name: str,
    target: Tensor,
    prefix_bits: bool = False,
) -> Tensor | None:
    """Normalize one Haar certificate tensor for the flattened authorizer.

    The certified Haar builder owns path-major ``[P,392]`` records, whereas
    the Dynkin runner presents the Jacobi authorizer with a flattened
    transition tensor.  This boundary is the only permitted shape adapter:
    it preserves logical path-major order, dtype, and device while making the
    authorizer contract explicit and fail-closed.
    """

    if value is None:
        return None

    def fail(reason: str, **diagnostics: Any) -> None:
        raise HaarSchedulerError(
            f"{name} cannot be adapted to the Jacobi transition shape: {reason}",
            failure_code="hierarchical_interval_adapter_shape_invalid",
            failure_domain="scheduler_execution",
            certificate_tensor=name,
            target_shape=list(target.shape),
            target_numel=int(target.numel()),
            **diagnostics,
        )

    if not isinstance(value, Tensor):
        fail("the certificate value is not a torch tensor")
    if not isinstance(target, Tensor) or not target.is_cuda:
        fail(
            "the target is not a CUDA tensor",
            target_device=(
                None
                if not isinstance(target, Tensor)
                else str(target.device)
            ),
        )
    if not value.is_cuda or value.device != target.device:
        fail(
            "the certificate and target are not on the common CUDA device",
            certificate_device=str(value.device),
            target_device=str(target.device),
        )
    if int(value.numel()) != int(target.numel()):
        fail(
            "the certificate element count differs from the target",
            certificate_shape=list(value.shape),
            certificate_numel=int(value.numel()),
        )
    if prefix_bits:
        if value.dtype not in {torch.int32, torch.int64}:
            fail(
                "prefix counts must use int32 or int64",
                certificate_dtype=str(value.dtype),
            )
        if bool((value < 1).any()) or bool((value > 1024).any()):
            fail(
                "prefix counts must lie in [1,1024]",
                certificate_min=int(value.min().item()),
                certificate_max=int(value.max().item()),
            )

    return value.reshape(target.shape).contiguous()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("scheduler records must be canonical JSON values") from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_hash(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value.__index__())
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    return result


def _validated_paths(path_ids: Sequence[int], role: str) -> tuple[int, ...]:
    paths = tuple(_integer(value, "path_id") for value in path_ids)
    if not 1 <= len(paths) <= MAX_REFINEMENT_PATHS_PER_GROUP:
        raise ValueError(
            "one exact hierarchical cohort must contain between one and "
            f"{MAX_REFINEMENT_PATHS_PER_GROUP} paths"
        )
    if tuple(sorted(paths)) != paths or len(set(paths)) != len(paths):
        raise ValueError("path_ids must be unique and in canonical increasing order")
    for path_id in paths:
        validate_role_path_id(role, path_id)
    return paths


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class HaarBackendContract:
    """Static production prerequisites for the normal and Jacobi backends."""

    fused_cuda_normal_authorizer: bool
    normal_cuda_authorizing: bool
    arbitrary_uniform_jacobi_authorizer: bool
    normal_fallback_fraction_upper_bound: float
    normal_fallback_time_fraction_upper_bound: float
    source: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def production_ready(self) -> bool:
        return bool(
            self.fused_cuda_normal_authorizer
            and self.normal_cuda_authorizing
            and self.arbitrary_uniform_jacobi_authorizer
            and math.isfinite(self.normal_fallback_fraction_upper_bound)
            and self.normal_fallback_fraction_upper_bound
            <= MAX_NORMAL_FALLBACK_FRACTION
            and math.isfinite(self.normal_fallback_time_fraction_upper_bound)
            and self.normal_fallback_time_fraction_upper_bound
            <= MAX_NORMAL_FALLBACK_TIME_FRACTION
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": HAAR_SCHEDULER_VERSION + "-backend-contract",
            "fused_cuda_normal_authorizer": int(
                self.fused_cuda_normal_authorizer
            ),
            "normal_cuda_authorizing": int(self.normal_cuda_authorizing),
            "arbitrary_uniform_jacobi_authorizer": int(
                self.arbitrary_uniform_jacobi_authorizer
            ),
            "normal_fallback_fraction_upper_bound": float(
                self.normal_fallback_fraction_upper_bound
            ),
            "normal_fallback_time_fraction_upper_bound": float(
                self.normal_fallback_time_fraction_upper_bound
            ),
            "production_ready": int(self.production_ready),
            "source": self.source,
            "details": dict(self.details),
        }


def inspect_haar_backend_contract(
    *,
    uniform_builder: Callable[..., Any] = build_certified_haar_uniform_batch,
    interval_authorizer: Callable[..., Any] = (
        sample_alpha1_rb_transition_batch_cuda_from_uniform_cells
    ),
) -> HaarBackendContract:
    """Return a fail-closed static backend report.

    A future fused builder may expose ``haar_backend_contract`` as either a
    :class:`HaarBackendContract` or a mapping with the corresponding fields.
    Merely being callable is not evidence of directed-rounding certification.
    """

    advertised = getattr(uniform_builder, "haar_backend_contract", None)
    if isinstance(advertised, HaarBackendContract):
        normal = advertised
    elif isinstance(advertised, Mapping):
        normal = HaarBackendContract(
            fused_cuda_normal_authorizer=bool(
                advertised.get("fused_cuda_normal_authorizer", False)
            ),
            normal_cuda_authorizing=bool(
                advertised.get("normal_cuda_authorizing", False)
            ),
            arbitrary_uniform_jacobi_authorizer=False,
            normal_fallback_fraction_upper_bound=float(
                advertised.get("normal_fallback_fraction_upper_bound", math.inf)
            ),
            normal_fallback_time_fraction_upper_bound=float(
                advertised.get(
                    "normal_fallback_time_fraction_upper_bound", math.inf
                )
            ),
            source=str(advertised.get("source", "callable-advertisement")),
            details=dict(advertised),
        )
    elif uniform_builder is build_certified_haar_uniform_batch:
        normal = HaarBackendContract(
            fused_cuda_normal_authorizer=False,
            normal_cuda_authorizing=False,
            arbitrary_uniform_jacobi_authorizer=False,
            normal_fallback_fraction_upper_bound=1.0,
            normal_fallback_time_fraction_upper_bound=1.0,
            source="portable-python-flint-arb",
            details={
                "reason": (
                    "the installed certified Haar builder authorizes every "
                    "normal transform with Arb and has no fused CUDA proof"
                ),
                "exact_but_production_ineligible": 1,
            },
        )
    else:
        normal = HaarBackendContract(
            fused_cuda_normal_authorizer=False,
            normal_cuda_authorizing=False,
            arbitrary_uniform_jacobi_authorizer=False,
            normal_fallback_fraction_upper_bound=math.inf,
            normal_fallback_time_fraction_upper_bound=math.inf,
            source="unverified-custom-normal-builder",
            details={"reason": "no immutable backend contract was advertised"},
        )

    interval_advertised = getattr(
        interval_authorizer, "haar_interval_authorizer_contract", None
    )
    interval_exact = bool(
        interval_authorizer
        is sample_alpha1_rb_transition_batch_cuda_from_uniform_cells
        or (
            isinstance(interval_advertised, Mapping)
            and interval_advertised.get(
                "arbitrary_uniform_jacobi_authorizer", False
            )
        )
    )
    return replace(
        normal,
        arbitrary_uniform_jacobi_authorizer=interval_exact,
        details={
            **dict(normal.details),
            "interval_authorizer": getattr(
                interval_authorizer, "__qualname__", type(interval_authorizer).__name__
            ),
            "interval_authorizer_verified": int(interval_exact),
        },
    )


def require_production_haar_backend(
    *,
    uniform_builder: Callable[..., Any] = build_certified_haar_uniform_batch,
    interval_authorizer: Callable[..., Any] = (
        sample_alpha1_rb_transition_batch_cuda_from_uniform_cells
    ),
) -> HaarBackendContract:
    contract = inspect_haar_backend_contract(
        uniform_builder=uniform_builder,
        interval_authorizer=interval_authorizer,
    )
    if not contract.production_ready:
        raise HaarSchedulerError(
            "the certified Haar normal backend cannot satisfy production "
            "fallback and CUDA-authorizer prerequisites",
            failure_code="fused_cuda_normal_backend_unavailable",
            failure_domain="normal_transform_backend",
            backend_contract=contract.to_record(),
        )
    return contract


def canonical_haar_scheduler_transition_ids(
    *,
    role: str,
    path_ids: Sequence[int],
    sample_steps: int,
    outer_step: int,
    phase: int,
    detail_sign: int,
    tree_root_steps: int,
    device: str | torch.device,
) -> Tensor:
    """Build the structural IDs consumed by one scheduled phase.

    ``tree_root_steps`` is explicit so adjacent pairwise schedules cannot
    accidentally inherit the 128-step root of the nested main profile.
    """

    paths = _validated_paths(path_ids, role)
    sign = _integer(detail_sign, "detail_sign")
    if sign not in {-1, 0, 1}:
        raise ValueError("detail_sign must be -1, 0, or 1")
    root = _integer(tree_root_steps, "tree_root_steps")
    values = np.asarray(
        [
            canonical_haar_transition_id(
                HaarEventIdentity(
                    role=role,
                    path_id=path_id,
                    sample_steps=int(sample_steps),
                    outer_step=int(outer_step),
                    phase=int(phase),
                    edge_id=edge,
                    arm=sign,
                    tree_root_steps=root,
                )
            )
            for path_id in paths
            for edge in range(EDGES_PER_PHASE)
        ],
        dtype=np.uint64,
    )
    return torch.from_numpy(values.copy()).to(device=torch.device(device)).contiguous()


@dataclass(frozen=True)
class NestedHaarSchedule:
    pool: str
    role: str
    levels: tuple[int, ...] | None = None
    profile_name: str = NESTED_PROFILE_NAME

    def __post_init__(self) -> None:
        if self.pool not in {"main", "reference"}:
            raise ValueError("nested pool must be main or reference")
        if self.role not in {"nested_a", "nested_b"}:
            raise ValueError("nested schedule requires a nested A/B role")
        frozen = (
            NESTED_MAIN_LEVELS if self.pool == "main" else NESTED_REFERENCE_LEVELS
        )
        selected = frozen if self.levels is None else tuple(self.levels)
        if tuple(selected) != tuple(frozen):
            raise ValueError("nested temporal levels are frozen by pool")
        object.__setattr__(self, "levels", tuple(int(value) for value in selected))
        if self.profile_name != NESTED_PROFILE_NAME:
            raise ValueError("unsupported nested profile name")

    @property
    def coarsest_steps(self) -> int:
        assert self.levels is not None
        return int(self.levels[0])

    @property
    def finest_steps(self) -> int:
        assert self.levels is not None
        return int(self.levels[-1])

    def to_record(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "pool": self.pool,
            "role": self.role,
            "levels": list(self.levels or ()),
            "coarsest_steps": self.coarsest_steps,
            "finest_steps": self.finest_steps,
            "single_arm": 1,
        }


@dataclass(frozen=True)
class PairwiseHaarAntitheticSchedule:
    coarse_steps: int
    fine_steps: int
    role: str
    profile_name: str = ANTITHETIC_PROFILE_NAME

    def __post_init__(self) -> None:
        pair = (int(self.coarse_steps), int(self.fine_steps))
        if pair not in ADJACENT_LEVEL_PAIRS:
            raise ValueError("antithetic schedule requires a frozen adjacent pair")
        if self.role not in {"antithetic_a", "antithetic_b"}:
            raise ValueError("antithetic schedule requires an antithetic A/B role")
        if self.profile_name != ANTITHETIC_PROFILE_NAME:
            raise ValueError("unsupported antithetic profile name")

    @property
    def levels(self) -> tuple[int, int]:
        return int(self.coarse_steps), int(self.fine_steps)

    def to_record(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "role": self.role,
            "coarse_steps": int(self.coarse_steps),
            "fine_steps": int(self.fine_steps),
            "pair_local_tree": 1,
            "fine_arms": [-1, 1],
        }


HaarSchedule = NestedHaarSchedule | PairwiseHaarAntitheticSchedule


@dataclass(frozen=True)
class HaarShardIdentity:
    """Immutable identity for one aligned eight-coarse-step shard."""

    schedule: HaarSchedule
    path_ids: tuple[int, ...]
    coarsest_start_step: int
    root_seed: int
    panel_namespace: str
    scheduler_version: str = HAAR_SCHEDULER_VERSION

    def __post_init__(self) -> None:
        paths = _validated_paths(self.path_ids, self.schedule.role)
        object.__setattr__(self, "path_ids", paths)
        first = _integer(self.coarsest_start_step, "coarsest_start_step")
        coarse = (
            self.schedule.coarsest_steps
            if isinstance(self.schedule, NestedHaarSchedule)
            else int(self.schedule.coarse_steps)
        )
        if (
            first < 0
            or first % REFINEMENT_SHARD_STEPS
            or first + REFINEMENT_SHARD_STEPS > coarse
        ):
            raise ValueError(
                "coarsest_start_step must identify a complete eight-step shard"
            )
        object.__setattr__(self, "coarsest_start_step", first)
        seed = _integer(self.root_seed, "root_seed")
        if not 0 <= seed < (1 << 64):
            raise ValueError("root_seed must fit uint64")
        object.__setattr__(self, "root_seed", seed)
        if (
            not isinstance(self.panel_namespace, str)
            or not self.panel_namespace.strip()
            or len(self.panel_namespace.encode("utf-8")) > 128
        ):
            raise ValueError("panel_namespace must contain 1..128 UTF-8 bytes")
        if self.scheduler_version != HAAR_SCHEDULER_VERSION:
            raise ValueError("unsupported Haar scheduler version")

    def to_record(self) -> dict[str, Any]:
        return {
            "scheduler_version": self.scheduler_version,
            "schedule": self.schedule.to_record(),
            "path_ids": list(self.path_ids),
            "coarsest_start_step": int(self.coarsest_start_step),
            "coarsest_step_count": REFINEMENT_SHARD_STEPS,
            "root_seed": int(self.root_seed),
            "panel_namespace": self.panel_namespace,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_record())


@dataclass(frozen=True)
class HaarBranchResult:
    branch: str
    sample_steps: int
    detail_sign: int
    completed_steps: tuple[int, ...]
    final_states: Tensor = field(repr=False, compare=False)
    accumulator_state: DynkinAccumulatorState = field(repr=False, compare=False)
    committed_final_states: np.ndarray = field(repr=False, compare=False)
    committed_accumulator_center: np.ndarray = field(repr=False, compare=False)
    committed_accumulator_compensation: np.ndarray = field(
        repr=False, compare=False
    )
    committed_accumulator_error_radius: np.ndarray = field(
        repr=False, compare=False
    )
    raw_observables: np.ndarray = field(repr=False, compare=False)
    dynkin_observables: np.ndarray = field(repr=False, compare=False)
    dynkin_error_radius: np.ndarray = field(repr=False, compare=False)
    base_output_hashes: tuple[str, ...]
    base_state_hashes: tuple[str, ...]
    output_sha256: str
    diagnostics: Mapping[str, Any]
    shard_results: tuple[Any, ...] = field(repr=False, compare=False)

    def to_record(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "sample_steps": int(self.sample_steps),
            "detail_sign": int(self.detail_sign),
            "completed_steps": list(self.completed_steps),
            "state_shape": list(self.committed_final_states.shape),
            "raw_observable_shape": list(self.raw_observables.shape),
            "dynkin_observable_shape": list(self.dynkin_observables.shape),
            "final_state_sha256": _array_hash(self.committed_final_states),
            "accumulator_center_sha256": _array_hash(
                self.committed_accumulator_center
            ),
            "accumulator_compensation_sha256": _array_hash(
                self.committed_accumulator_compensation
            ),
            "accumulator_error_radius_sha256": _array_hash(
                self.committed_accumulator_error_radius
            ),
            "raw_observables_sha256": _array_hash(self.raw_observables),
            "dynkin_observables_sha256": _array_hash(self.dynkin_observables),
            "dynkin_error_radius_sha256": _array_hash(
                self.dynkin_error_radius
            ),
            "base_output_hashes": list(self.base_output_hashes),
            "base_state_hashes": list(self.base_state_hashes),
            "output_sha256": self.output_sha256,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class HaarHierarchicalShardResult:
    identity: HaarShardIdentity
    branches: Mapping[str, HaarBranchResult]
    backend_contract: HaarBackendContract
    input_sha256: str
    output_sha256: str
    diagnostics: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": HAAR_SHARD_SCHEMA,
            "schema_version": 1,
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.fingerprint,
            "backend_contract": self.backend_contract.to_record(),
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "branches": {
                name: self.branches[name].to_record()
                for name in sorted(self.branches)
            },
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class HaarShardResumeState:
    identity: HaarShardIdentity
    states: Mapping[str, Tensor]
    accumulators: Mapping[str, DynkinAccumulatorState]
    raw_observables: Mapping[str, np.ndarray]
    dynkin_observables: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


class _CertifiedHaarSampler:
    """Per-level sampler hook consumed by the unchanged Dynkin runner."""

    def __init__(
        self,
        *,
        root_seed: int,
        role: str,
        path_ids: tuple[int, ...],
        sample_steps: int,
        start_step: int,
        detail_sign: int,
        pair_coarse_steps: int | None,
        haar_profile: HaarCouplingProfile,
        jacobi_profile: JacobiRBCudaProfile,
        uniform_builder: Callable[..., Any],
        interval_authorizer: Callable[..., Any],
        enforce_runtime_contract: bool,
    ) -> None:
        self.root_seed = int(root_seed)
        self.role = role
        self.path_ids = path_ids
        self.sample_steps = int(sample_steps)
        self.start_step = int(start_step)
        self.detail_sign = int(detail_sign)
        self.pair_coarse_steps = (
            None if pair_coarse_steps is None else int(pair_coarse_steps)
        )
        self.haar_profile = haar_profile
        self.jacobi_profile = jacobi_profile
        self.uniform_builder = uniform_builder
        self.interval_authorizer = interval_authorizer
        self.enforce_runtime_contract = bool(enforce_runtime_contract)
        self.call_count = 0
        self.uniform_seconds = 0.0
        self.jacobi_seconds = 0.0
        self.uniform_fallback_count = 0
        self.uniform_sample_count = 0
        self.uniform_fallback_seconds = 0.0
        self._last_uniform_runtime: dict[str, Any] = {}
        self._last_jacobi_runtime: dict[str, Any] = {}

    def _ids(
        self,
        *,
        sample_steps: int,
        outer_step: int,
        phase: int,
        device: torch.device,
    ) -> Tensor:
        return canonical_haar_scheduler_transition_ids(
            role=self.role,
            path_ids=self.path_ids,
            sample_steps=int(sample_steps),
            outer_step=int(outer_step),
            phase=int(phase),
            detail_sign=(
                self.detail_sign if self.pair_coarse_steps is not None else 0
            ),
            tree_root_steps=(
                int(self.pair_coarse_steps)
                if self.pair_coarse_steps is not None
                else int(self.haar_profile.coarsest_steps)
            ),
            device=device,
        )

    def transition_id_provider(
        self,
        path_ids: Sequence[int],
        *,
        sample_steps: int,
        outer_step: int,
        phase: int,
        device: torch.device,
    ) -> Tensor:
        if tuple(path_ids) != self.path_ids or int(sample_steps) != self.sample_steps:
            raise HaarSchedulerError(
                "the parent runner changed the frozen Haar path/level identity",
                failure_code="hierarchical_transition_identity_invalid",
            )
        return self._ids(
            sample_steps=sample_steps,
            outer_step=outer_step,
            phase=phase,
            device=device,
        )

    def _verify_runtime(self, runtime: Mapping[str, Any]) -> None:
        if not self.enforce_runtime_contract:
            return
        if not bool(runtime.get("fused_cuda_authorizer_available", False)):
            raise HaarSchedulerError(
                "normal builder did not execute a fused CUDA authorizer",
                failure_code="fused_cuda_normal_backend_unavailable",
                failure_domain="normal_transform_backend",
                runtime_report=dict(runtime),
            )
        if int(runtime.get("device_resident_certified_output", 0)) != 1:
            raise HaarSchedulerError(
                "normal builder materialized certified production lanes on "
                "the host",
                failure_code="hierarchical_device_residency_invalid",
                failure_domain="normal_transform_backend",
                runtime_report=dict(runtime),
            )
        fallback = float(runtime.get("arb_fallback_fraction", math.inf))
        fallback_time = float(
            runtime.get("arb_fallback_time_fraction", math.inf)
        )
        if (
            not math.isfinite(fallback)
            or fallback > MAX_NORMAL_FALLBACK_FRACTION
            or not math.isfinite(fallback_time)
            or fallback_time > MAX_NORMAL_FALLBACK_TIME_FRACTION
        ):
            raise HaarSchedulerError(
                "normal builder exceeded the frozen fallback contract",
                failure_code="hierarchical_normal_fallback_excessive",
                failure_domain="normal_transform_backend",
                fallback_fraction=fallback,
                fallback_time_fraction=fallback_time,
            )

    def __call__(
        self,
        x: Tensor,
        exposure: Tensor,
        *,
        rng_key: Any,
        transition_ids: Tensor,
        profile: JacobiRBCudaProfile,
    ) -> Any:
        del rng_key
        if profile != self.jacobi_profile:
            raise HaarSchedulerError(
                "Jacobi profile changed inside the exact Haar shard",
                failure_code="hierarchical_jacobi_profile_mismatch",
            )
        local_step, phase = divmod(self.call_count, len(PHASE_MATCHINGS))
        if local_step >= REFINEMENT_SHARD_STEPS:
            raise HaarSchedulerError(
                "Haar sampler received too many phase calls",
                failure_code="hierarchical_phase_schedule_invalid",
            )
        outer_step = self.start_step + local_step
        expected_ids = self._ids(
            sample_steps=self.sample_steps,
            outer_step=outer_step,
            phase=phase,
            device=x.device,
        )
        if not torch.equal(transition_ids.reshape(-1), expected_ids):
            raise HaarSchedulerError(
                "parent transition IDs differ from the Haar identity plan",
                failure_code="hierarchical_transition_identity_invalid",
            )

        _sync(x.device)
        started = time.perf_counter()
        uniforms = self.uniform_builder(
            root_seed=self.root_seed,
            role=self.role,
            path_ids=self.path_ids,
            sample_steps=self.sample_steps,
            outer_step=outer_step,
            phase=phase,
            edge_ids=range(EDGES_PER_PHASE),
            profile=self.haar_profile,
            detail_sign=self.detail_sign,
            pair_coarse_steps=self.pair_coarse_steps,
            device=x.device,
        )
        _sync(x.device)
        uniform_elapsed = time.perf_counter() - started
        runtime = dict(getattr(uniforms, "runtime_report", {}))
        self._verify_runtime(runtime)
        generated_ids = getattr(uniforms, "transition_ids", None)
        if generated_ids is None:
            if self.enforce_runtime_contract:
                raise HaarSchedulerError(
                    "normal builder omitted structural transition IDs",
                    failure_code="hierarchical_transition_identity_invalid",
                )
        elif not torch.equal(
            generated_ids.reshape(-1).to(
                device=expected_ids.device, dtype=torch.uint64
            ),
            expected_ids,
        ):
            raise HaarSchedulerError(
                "normal builder used a different structural transition ID",
                failure_code="hierarchical_transition_identity_invalid",
            )
        uniform_certified = getattr(uniforms, "certificate_mask", None)
        if self.enforce_runtime_contract and (
            not isinstance(uniform_certified, Tensor)
            or not bool(uniform_certified.all())
        ):
            raise HaarSchedulerError(
                "normal builder returned an uncertified uniform enclosure",
                failure_code="certified_normal_transform_invalid",
                failure_domain="normal_transform_backend",
            )
        self.uniform_seconds += uniform_elapsed
        self.uniform_sample_count += int(np.prod(getattr(uniforms, "shape", x.shape)))
        diagnostics = getattr(uniforms, "diagnostics", {})
        if isinstance(diagnostics, Mapping):
            self.uniform_fallback_count += int(
                diagnostics.get("arb_fallback_count", 0)
            )
        self.uniform_fallback_seconds += float(
            runtime.get("arb_fallback_elapsed_seconds", 0.0)
        )
        self._last_uniform_runtime = runtime

        lower = uniforms.uniform_lower.reshape(x.shape).contiguous()
        upper = uniforms.uniform_upper.reshape(x.shape).contiguous()
        refinement_callback = getattr(uniforms, "refinement_callback", None)
        _sync(x.device)
        started = time.perf_counter()
        interval_kwargs: dict[str, Any] = {
            "transition_ids": expected_ids.reshape(x.shape).contiguous(),
            "refinement_callback": refinement_callback,
            "profile": profile,
        }
        using_installed_interval_authorizer = (
            self.interval_authorizer
            is sample_alpha1_rb_transition_batch_cuda_from_uniform_cells
        )
        if using_installed_interval_authorizer:
            interval_kwargs.update(
                uniform_center_hi=_normalize_interval_adapter_tensor(
                    getattr(uniforms, "uniform_center_hi", None),
                    name="uniform_center_hi",
                    target=x,
                ),
                uniform_center_lo=_normalize_interval_adapter_tensor(
                    getattr(uniforms, "uniform_center_lo", None),
                    name="uniform_center_lo",
                    target=x,
                ),
                uniform_radius=_normalize_interval_adapter_tensor(
                    getattr(uniforms, "uniform_radius", None),
                    name="uniform_radius",
                    target=x,
                ),
                source_prefix_bits=_normalize_interval_adapter_tensor(
                    getattr(uniforms, "prefix_bits", None),
                    name="prefix_bits",
                    target=x,
                    prefix_bits=True,
                ),
            )
        try:
            result = self.interval_authorizer(
                x, exposure, lower, upper, **interval_kwargs
            )
        except HaarSchedulerError:
            raise
        except (TypeError, ValueError) as exc:
            if not using_installed_interval_authorizer:
                raise
            raise HaarSchedulerError(
                "the installed arbitrary-uniform Jacobi authorizer rejected "
                "the normalized scheduler inputs",
                failure_code="hierarchical_interval_adapter_shape_invalid",
                failure_domain="scheduler_execution",
                authorizer_type=type(exc).__name__,
                authorizer_message=str(exc),
                transition_shape=list(x.shape),
                transition_numel=int(x.numel()),
            ) from exc
        _sync(x.device)
        self.jacobi_seconds += time.perf_counter() - started
        self._last_jacobi_runtime = dict(
            getattr(result, "runtime_report", {})
        )
        if self.enforce_runtime_contract:
            codes = getattr(result, "certificate_codes", None)
            active = exposure > 0.0
            if (
                not isinstance(codes, Tensor)
                or codes.shape != exposure.shape
                or bool((active & ((codes.to(torch.uint8) & 0xF) != 0xF)).any())
            ):
                raise HaarSchedulerError(
                    "arbitrary-uniform Jacobi authorizer returned an "
                    "uncertified active transition",
                    failure_code="arbitrary_uniform_jacobi_certificate_invalid",
                    failure_domain="jacobi_authorizer",
                )
        self.call_count += 1
        return result

    def diagnostics(self) -> dict[str, Any]:
        fallback_fraction = (
            self.uniform_fallback_count / self.uniform_sample_count
            if self.uniform_sample_count
            else 0.0
        )
        elapsed = self.uniform_seconds + self.jacobi_seconds
        return {
            "haar_sampler_call_count": self.call_count,
            "normal_transform_seconds": self.uniform_seconds,
            "jacobi_authorizer_seconds": self.jacobi_seconds,
            "normal_fallback_seconds": self.uniform_fallback_seconds,
            "normal_sample_count": self.uniform_sample_count,
            "normal_fallback_count": self.uniform_fallback_count,
            "normal_fallback_fraction": fallback_fraction,
            "normal_fallback_time_fraction": (
                self.uniform_fallback_seconds / elapsed if elapsed > 0.0 else 0.0
            ),
            "last_normal_runtime": self._last_uniform_runtime,
            "last_jacobi_runtime": self._last_jacobi_runtime,
        }


def _state_array(value: Any, *, name: str, paths: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (paths, 784) or not np.all(np.isfinite(array)):
        raise HaarSchedulerError(
            f"{name} returned an invalid committed state",
            failure_code="hierarchical_shard_state_invalid",
        )
    return np.array(array, copy=True, order="C")


def _checkpoint_arrays(
    result: Any, *, paths: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoints = tuple(
        getattr(result, "observable_checkpoints", ())
        or getattr(result, "checkpoint_states", ())
    )
    if not checkpoints:
        raise HaarSchedulerError(
            "Dynkin runner omitted the requested raw/Dynkin checkpoint",
            failure_code="hierarchical_observable_checkpoint_missing",
        )
    checkpoint = checkpoints[-1]
    arrays = tuple(
        np.asarray(getattr(checkpoint, name), dtype=np.float64)
        for name in (
            "raw_values",
            "dynkin_values",
            "dynkin_error_radius",
        )
    )
    if any(value.shape != (paths, OBSERVABLE_COUNT) for value in arrays):
        raise HaarSchedulerError(
            "Dynkin checkpoint has the wrong observable shape",
            failure_code="hierarchical_observable_checkpoint_invalid",
        )
    return tuple(np.array(value, copy=True, order="C") for value in arrays)  # type: ignore[return-value]


def _accumulator_arrays(
    result: Any, *, paths: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = tuple(
        np.asarray(getattr(result, name), dtype=np.float64)
        for name in (
            "committed_accumulator_center",
            "committed_accumulator_compensation",
            "committed_accumulator_error_radius",
        )
    )
    if any(value.shape != (paths, OBSERVABLE_COUNT) for value in values):
        raise HaarSchedulerError(
            "Dynkin runner returned an invalid accumulator",
            failure_code="hierarchical_dynkin_accumulator_invalid",
        )
    return tuple(np.array(value, copy=True, order="C") for value in values)  # type: ignore[return-value]


def _run_branch(
    *,
    branch: str,
    states: Tensor,
    accumulator_state: DynkinAccumulatorState | None,
    sample_steps: int,
    coarsest_steps: int,
    coarsest_start_step: int,
    detail_sign: int,
    pair_coarse_steps: int | None,
    identity: HaarShardIdentity,
    haar_profile: HaarCouplingProfile,
    jacobi_profile: JacobiRBCudaProfile,
    level_runner: Callable[..., Any],
    uniform_builder: Callable[..., Any],
    interval_authorizer: Callable[..., Any],
    enforce_runtime_contract: bool,
) -> HaarBranchResult:
    ratio = int(sample_steps) // int(coarsest_steps)
    if ratio < 1 or ratio & (ratio - 1):
        raise ValueError("branch level is not dyadic over the coarsest level")
    if (
        not isinstance(states, Tensor)
        or states.dtype != torch.float64
        or states.ndim != 2
        or states.shape != (len(identity.path_ids), 784)
        or not states.is_contiguous()
    ):
        raise ValueError(f"{branch} states must be contiguous float64 [P,784]")
    if enforce_runtime_contract and not states.is_cuda:
        raise HaarSchedulerError(
            "production Haar scheduling requires CUDA states",
            failure_code="hierarchical_cuda_state_required",
            failure_domain="runtime_backend",
        )

    values = states
    accumulator = accumulator_state
    raw_blocks: list[np.ndarray] = []
    dynkin_blocks: list[np.ndarray] = []
    radius_blocks: list[np.ndarray] = []
    completed_steps: list[int] = []
    base_output_hashes: list[str] = []
    base_state_hashes: list[str] = []
    sampler_reports: list[dict[str, Any]] = []
    base_reports: list[dict[str, Any]] = []
    shard_results: list[Any] = []
    runner_elapsed = 0.0
    fine_start = int(coarsest_start_step) * ratio

    for subshard in range(ratio):
        start_step = fine_start + subshard * REFINEMENT_SHARD_STEPS
        sampler = _CertifiedHaarSampler(
            root_seed=identity.root_seed,
            role=identity.schedule.role,
            path_ids=identity.path_ids,
            sample_steps=int(sample_steps),
            start_step=start_step,
            detail_sign=detail_sign,
            pair_coarse_steps=pair_coarse_steps,
            haar_profile=haar_profile,
            jacobi_profile=jacobi_profile,
            uniform_builder=uniform_builder,
            interval_authorizer=interval_authorizer,
            enforce_runtime_contract=enforce_runtime_contract,
        )
        _sync(values.device)
        started = time.perf_counter()
        try:
            result = level_runner(
                values,
                path_ids=identity.path_ids,
                sample_steps=int(sample_steps),
                start_step=start_step,
                root_seed=identity.root_seed,
                panel_namespace=(
                    f"{identity.panel_namespace}:{identity.schedule.profile_name}:"
                    f"{branch}"
                ),
                profile=jacobi_profile,
                sampler=sampler,
                checkpoint_steps=(start_step + REFINEMENT_SHARD_STEPS,),
                transition_id_provider=sampler.transition_id_provider,
                accumulator_state=accumulator,
            )
        except Exception as exc:
            cause: BaseException | None = exc
            while cause is not None and not isinstance(cause, HaarSchedulerError):
                cause = cause.__cause__
            if isinstance(cause, HaarSchedulerError):
                raise cause
            raise
        _sync(values.device)
        runner_elapsed += time.perf_counter() - started
        if sampler.call_count != REFINEMENT_SHARD_STEPS * len(PHASE_MATCHINGS):
            raise HaarSchedulerError(
                "Dynkin runner did not execute every exact Haar phase",
                failure_code="hierarchical_phase_schedule_invalid",
            )
        sampler_reports.append(sampler.diagnostics())
        shard_results.append(result)
        base_diagnostics = dict(getattr(result, "diagnostics", {}))
        base_reports.append(base_diagnostics)
        if enforce_runtime_contract:
            if float(base_diagnostics.get("certificate_fraction", 0.0)) != 1.0:
                raise HaarSchedulerError(
                    "exact Dynkin shard contains uncertified transitions",
                    failure_code="hierarchical_jacobi_certificate_invalid",
                    failure_domain="jacobi_authorizer",
                )
            if float(base_diagnostics.get("fallback_fraction", math.inf)) > 1.0e-4:
                raise HaarSchedulerError(
                    "exact Dynkin shard exceeded the Jacobi fallback limit",
                    failure_code="hierarchical_jacobi_fallback_excessive",
                    failure_domain="jacobi_authorizer",
                )
            for forbidden_name in (
                "resource_cap_count",
                "invalid_density_count",
                "approximation_count",
                "correction_count",
                "floor_count",
                "limiter_count",
                "projection_count",
                "renormalization_count",
                "nonfinite_count",
            ):
                if int(base_diagnostics.get(forbidden_name, 0)):
                    raise HaarSchedulerError(
                        "exact Dynkin shard recorded a forbidden numerical event",
                        failure_code="hierarchical_forbidden_event",
                        failure_domain="jacobi_authorizer",
                        diagnostic=forbidden_name,
                        count=int(base_diagnostics[forbidden_name]),
                    )
            required_diagnostics = (
                "transition_count",
                "certified_count",
                "uncertified_count",
                "fallback_count",
                "fallback_elapsed_seconds",
                "maximum_global_simplex_error",
                "state_updates_device_resident",
                "resource_cap_count",
                "invalid_density_count",
                "approximation_count",
                "correction_count",
                "floor_count",
                "limiter_count",
                "projection_count",
                "renormalization_count",
                "nonfinite_count",
            )
            missing = [
                name for name in required_diagnostics if name not in base_diagnostics
            ]
            if missing:
                raise HaarSchedulerError(
                    "exact Dynkin shard omitted required diagnostics",
                    failure_code="hierarchical_shard_diagnostics_invalid",
                    missing=missing,
                )
        values = result.final_states
        accumulator = result.accumulator_state
        raw, dynkin, radius = _checkpoint_arrays(
            result, paths=len(identity.path_ids)
        )
        raw_blocks.append(raw)
        dynkin_blocks.append(dynkin)
        radius_blocks.append(radius)
        completed_steps.append(start_step + REFINEMENT_SHARD_STEPS)
        base_output_hashes.append(str(result.batch_output_sha256))
        base_state_hashes.append(str(result.batch_final_state_sha256))

    if accumulator is None or not shard_results:
        raise AssertionError("hierarchical branch executed no exact shards")
    final_result = shard_results[-1]
    final_host = _state_array(
        final_result.committed_final_states,
        name=branch,
        paths=len(identity.path_ids),
    )
    acc_center, acc_compensation, acc_radius = _accumulator_arrays(
        final_result, paths=len(identity.path_ids)
    )
    raw_array = np.stack(raw_blocks)
    dynkin_array = np.stack(dynkin_blocks)
    radius_array = np.stack(radius_blocks)
    normal_seconds = sum(
        float(value["normal_transform_seconds"]) for value in sampler_reports
    )
    jacobi_seconds = sum(
        float(value["jacobi_authorizer_seconds"]) for value in sampler_reports
    )
    normal_fallback_count = sum(
        int(value["normal_fallback_count"]) for value in sampler_reports
    )
    normal_sample_count = sum(
        int(value["normal_sample_count"]) for value in sampler_reports
    )
    base_transition_count = sum(
        int(value.get("transition_count", 0)) for value in base_reports
    )
    base_certified_count = sum(
        int(value.get("certified_count", 0)) for value in base_reports
    )
    base_fallback_count = sum(
        int(value.get("fallback_count", 0)) for value in base_reports
    )
    base_fallback_seconds = sum(
        float(value.get("fallback_elapsed_seconds", 0.0))
        for value in base_reports
    )
    forbidden_names = (
        "uncertified_count",
        "resource_cap_count",
        "invalid_density_count",
        "approximation_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
    )
    output_hash = _fingerprint(
        {
            "scheduler_version": HAAR_SCHEDULER_VERSION,
            "branch": branch,
            "sample_steps": int(sample_steps),
            "detail_sign": int(detail_sign),
            "completed_steps": completed_steps,
            "final_state_sha256": _array_hash(final_host),
            "accumulator_center_sha256": _array_hash(acc_center),
            "accumulator_compensation_sha256": _array_hash(acc_compensation),
            "accumulator_error_radius_sha256": _array_hash(acc_radius),
            "raw_observables_sha256": _array_hash(raw_array),
            "dynkin_observables_sha256": _array_hash(dynkin_array),
            "dynkin_error_radius_sha256": _array_hash(radius_array),
            "base_output_hashes": base_output_hashes,
            "base_state_hashes": base_state_hashes,
        }
    )
    diagnostics = {
        "underlying_dynkin_shard_count": ratio,
        "underlying_step_count": ratio * REFINEMENT_SHARD_STEPS,
        "phase_count": ratio * REFINEMENT_SHARD_STEPS * len(PHASE_MATCHINGS),
        "transition_count": (
            len(identity.path_ids)
            * ratio
            * REFINEMENT_SHARD_STEPS
            * len(PHASE_MATCHINGS)
            * EDGES_PER_PHASE
        ),
        "normal_transform_seconds": normal_seconds,
        "jacobi_authorizer_seconds": jacobi_seconds,
        "dynkin_runner_complete_seconds": runner_elapsed,
        "state_and_observer_seconds": max(
            0.0, runner_elapsed - normal_seconds - jacobi_seconds
        ),
        "normal_fallback_count": normal_fallback_count,
        "normal_sample_count": normal_sample_count,
        "normal_fallback_fraction": (
            normal_fallback_count / normal_sample_count
            if normal_sample_count
            else 0.0
        ),
        "jacobi_transition_count": base_transition_count,
        "jacobi_certified_count": base_certified_count,
        "jacobi_certificate_fraction": (
            base_certified_count / base_transition_count
            if base_transition_count
            else 0.0
        ),
        "jacobi_fallback_count": base_fallback_count,
        "jacobi_fallback_seconds": base_fallback_seconds,
        "jacobi_fallback_fraction": (
            base_fallback_count / base_transition_count
            if base_transition_count
            else 0.0
        ),
        "mass_error": max(
            (
                float(
                    value.get(
                        "mass_error",
                        value.get(
                            "maximum_global_simplex_error",
                            (
                                math.inf
                                if enforce_runtime_contract
                                else float(
                                    np.max(
                                        np.abs(
                                            final_host.sum(axis=1) - 1.0
                                        )
                                    )
                                )
                            ),
                        ),
                    )
                )
                for value in base_reports
            ),
            default=math.inf,
        ),
        "peak_memory_fraction": max(
            (
                float(value.get("peak_memory_fraction", 0.0))
                for value in base_reports
            ),
            default=0.0,
        ),
        "state_updates_device_resident_pass": int(
            bool(base_reports)
            and all(
                int(
                    value.get(
                        "state_updates_device_resident_pass",
                        value.get(
                            "state_updates_device_resident",
                            0 if enforce_runtime_contract else 1,
                        ),
                    )
                )
                == 1
                for value in base_reports
            )
        ),
        **{
            name: sum(
                int(value.get(name, -1 if enforce_runtime_contract else 0))
                for value in base_reports
            )
            for name in forbidden_names
        },
        "sampler_reports": sampler_reports,
    }
    return HaarBranchResult(
        branch=branch,
        sample_steps=int(sample_steps),
        detail_sign=int(detail_sign),
        completed_steps=tuple(completed_steps),
        final_states=values,
        accumulator_state=accumulator,
        committed_final_states=final_host,
        committed_accumulator_center=acc_center,
        committed_accumulator_compensation=acc_compensation,
        committed_accumulator_error_radius=acc_radius,
        raw_observables=raw_array,
        dynkin_observables=dynkin_array,
        dynkin_error_radius=radius_array,
        base_output_hashes=tuple(base_output_hashes),
        base_state_hashes=tuple(base_state_hashes),
        output_sha256=output_hash,
        diagnostics=diagnostics,
        shard_results=tuple(shard_results),
    )


def _runtime_mode(
    *,
    production_authorizing: bool,
    nonauthorizing_test_only: bool,
    uniform_builder: Callable[..., Any],
    interval_authorizer: Callable[..., Any],
) -> tuple[HaarBackendContract, bool]:
    if production_authorizing and nonauthorizing_test_only:
        raise ValueError(
            "production_authorizing and nonauthorizing_test_only are exclusive"
        )
    if not production_authorizing and not nonauthorizing_test_only:
        raise HaarSchedulerError(
            "disabling production checks requires the explicit test-only flag",
            failure_code="nonauthorizing_mode_not_explicit",
        )
    contract = inspect_haar_backend_contract(
        uniform_builder=uniform_builder,
        interval_authorizer=interval_authorizer,
    )
    if production_authorizing:
        contract = require_production_haar_backend(
            uniform_builder=uniform_builder,
            interval_authorizer=interval_authorizer,
        )
    return contract, production_authorizing


def _input_hash(
    identity: HaarShardIdentity,
    states: Mapping[str, Tensor],
    accumulators: Mapping[str, DynkinAccumulatorState | None],
    jacobi_profile: JacobiRBCudaProfile,
) -> str:
    branch_records: list[dict[str, Any]] = []
    for branch in sorted(states):
        state = states[branch].detach().cpu().numpy()
        accumulator = accumulators.get(branch)
        branch_records.append(
            {
                "branch": branch,
                "state_sha256": _array_hash(state),
                "accumulator_sha256": (
                    None
                    if accumulator is None
                    else _array_hash(
                        accumulator.center.detach().cpu().numpy(),
                        accumulator.compensation.detach().cpu().numpy(),
                        accumulator.error_radius.detach().cpu().numpy(),
                    )
                ),
            }
        )
    return _fingerprint(
        {
            "identity": identity.to_record(),
            "branches": branch_records,
            "jacobi_profile": asdict(jacobi_profile),
        }
    )


def expected_haar_shard_input_sha256(
    identity: HaarShardIdentity,
    states: Mapping[str, Tensor],
    accumulators: Mapping[str, DynkinAccumulatorState | None],
    jacobi_profile: JacobiRBCudaProfile,
) -> str:
    """Return the exact input binding used by a committed Haar shard.

    High-level panel orchestration uses this before accepting a resumed shard,
    so a valid archive from a different predecessor cannot be spliced into a
    chain merely because its immutable shard identity has the same coordinates.
    """

    return _input_hash(identity, states, accumulators, jacobi_profile)


def _finish_result(
    *,
    identity: HaarShardIdentity,
    branches: Mapping[str, HaarBranchResult],
    backend_contract: HaarBackendContract,
    input_sha256: str,
    elapsed_seconds: float,
    production_authorizing: bool,
) -> HaarHierarchicalShardResult:
    ordered = {
        name: branches[name] for name in sorted(branches)
    }
    output_hash = _fingerprint(
        {
            "identity_sha256": identity.fingerprint,
            "input_sha256": input_sha256,
            "branches": [
                [name, ordered[name].output_sha256] for name in ordered
            ],
        }
    )
    transition_count = sum(
        int(value.diagnostics["transition_count"]) for value in ordered.values()
    )
    normal_seconds = sum(
        float(value.diagnostics["normal_transform_seconds"])
        for value in ordered.values()
    )
    jacobi_seconds = sum(
        float(value.diagnostics["jacobi_authorizer_seconds"])
        for value in ordered.values()
    )
    fallback_count = sum(
        int(value.diagnostics["normal_fallback_count"])
        for value in ordered.values()
    )
    normal_count = sum(
        int(value.diagnostics["normal_sample_count"])
        for value in ordered.values()
    )
    jacobi_count = sum(
        int(value.diagnostics["jacobi_transition_count"])
        for value in ordered.values()
    )
    jacobi_certified = sum(
        int(value.diagnostics["jacobi_certified_count"])
        for value in ordered.values()
    )
    jacobi_fallback = sum(
        int(value.diagnostics["jacobi_fallback_count"])
        for value in ordered.values()
    )
    jacobi_fallback_seconds = sum(
        float(value.diagnostics["jacobi_fallback_seconds"])
        for value in ordered.values()
    )
    forbidden_names = (
        "uncertified_count",
        "resource_cap_count",
        "invalid_density_count",
        "approximation_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
    )
    diagnostics = {
        "scheduler_version": HAAR_SCHEDULER_VERSION,
        "profile_name": identity.schedule.profile_name,
        "branch_order": list(ordered),
        "branch_count": len(ordered),
        "transition_count": transition_count,
        "normal_transform_seconds": normal_seconds,
        "jacobi_authorizer_seconds": jacobi_seconds,
        "complete_pipeline_before_file_commit_seconds": elapsed_seconds,
        "complete_pipeline_transitions_per_second": (
            transition_count / elapsed_seconds
            if elapsed_seconds > 0.0
            else math.inf
        ),
        "normal_fallback_count": fallback_count,
        "normal_sample_count": normal_count,
        "normal_fallback_fraction": (
            fallback_count / normal_count if normal_count else 0.0
        ),
        "jacobi_transition_count": jacobi_count,
        "jacobi_certified_count": jacobi_certified,
        "jacobi_certificate_fraction": (
            jacobi_certified / jacobi_count if jacobi_count else 0.0
        ),
        "jacobi_fallback_count": jacobi_fallback,
        "jacobi_fallback_seconds": jacobi_fallback_seconds,
        "jacobi_fallback_fraction": (
            jacobi_fallback / jacobi_count if jacobi_count else 0.0
        ),
        "fallback_count": fallback_count + jacobi_fallback,
        "fallback_fraction": (
            (fallback_count + jacobi_fallback) / (normal_count + jacobi_count)
            if normal_count + jacobi_count
            else 0.0
        ),
        "fallback_elapsed_seconds": sum(
            float(value.diagnostics.get("normal_fallback_seconds", 0.0))
            for value in ordered.values()
        )
        + jacobi_fallback_seconds,
        "certificate_fraction": (
            jacobi_certified / jacobi_count if jacobi_count else 0.0
        ),
        "mass_error": max(
            (float(value.diagnostics["mass_error"]) for value in ordered.values()),
            default=math.inf,
        ),
        "peak_memory_fraction": max(
            (
                float(value.diagnostics["peak_memory_fraction"])
                for value in ordered.values()
            ),
            default=math.inf,
        ),
        "state_updates_device_resident_pass": int(
            bool(ordered)
            and all(
                int(value.diagnostics["state_updates_device_resident_pass"]) == 1
                for value in ordered.values()
            )
        ),
        **{
            name: sum(
                int(value.diagnostics.get(name, -1))
                for value in ordered.values()
            )
            for name in forbidden_names
        },
        "raw_observables_recorded": 1,
        "dynkin_observables_recorded": 1,
        "production_authorizing": int(production_authorizing),
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }
    return HaarHierarchicalShardResult(
        identity=identity,
        branches=ordered,
        backend_contract=backend_contract,
        input_sha256=input_sha256,
        output_sha256=output_hash,
        diagnostics=diagnostics,
    )


def run_nested_haar_shard(
    states_by_level: Mapping[int, Tensor],
    *,
    identity: HaarShardIdentity,
    jacobi_profile: JacobiRBCudaProfile,
    accumulators_by_level: Mapping[int, DynkinAccumulatorState | None] | None = None,
    level_runner: Callable[..., Any] = run_dynkin_refinement_shard,
    uniform_builder: Callable[..., Any] = build_certified_haar_uniform_batch,
    interval_authorizer: Callable[..., Any] = (
        sample_alpha1_rb_transition_batch_cuda_from_uniform_cells
    ),
    production_authorizing: bool = True,
    nonauthorizing_test_only: bool = False,
) -> HaarHierarchicalShardResult:
    """Run one aligned exact nested-Haar main or reference shard."""

    if not isinstance(identity.schedule, NestedHaarSchedule):
        raise TypeError("identity must contain a NestedHaarSchedule")
    levels = tuple(identity.schedule.levels or ())
    if set(states_by_level) != set(levels):
        raise ValueError("states_by_level must contain exactly the frozen levels")
    accumulators = dict(accumulators_by_level or {})
    if set(accumulators) - set(levels):
        raise ValueError("accumulator levels are not part of this nested schedule")
    contract, enforce = _runtime_mode(
        production_authorizing=production_authorizing,
        nonauthorizing_test_only=nonauthorizing_test_only,
        uniform_builder=uniform_builder,
        interval_authorizer=interval_authorizer,
    )
    profile = HaarCouplingProfile(
        coarsest_steps=identity.schedule.coarsest_steps,
        finest_steps=identity.schedule.finest_steps,
    )
    named_states = {f"k{level}": states_by_level[level] for level in levels}
    named_accumulators = {
        f"k{level}": accumulators.get(level) for level in levels
    }
    input_sha = _input_hash(
        identity, named_states, named_accumulators, jacobi_profile
    )
    first_device = next(iter(named_states.values())).device
    _sync(first_device)
    started = time.perf_counter()
    branches: dict[str, HaarBranchResult] = {}
    for level in levels:
        branch = f"k{level}"
        branches[branch] = _run_branch(
            branch=branch,
            states=states_by_level[level],
            accumulator_state=accumulators.get(level),
            sample_steps=level,
            coarsest_steps=identity.schedule.coarsest_steps,
            coarsest_start_step=identity.coarsest_start_step,
            detail_sign=1,
            pair_coarse_steps=None,
            identity=identity,
            haar_profile=profile,
            jacobi_profile=jacobi_profile,
            level_runner=level_runner,
            uniform_builder=uniform_builder,
            interval_authorizer=interval_authorizer,
            enforce_runtime_contract=enforce,
        )
    _sync(first_device)
    elapsed = time.perf_counter() - started
    return _finish_result(
        identity=identity,
        branches=branches,
        backend_contract=contract,
        input_sha256=input_sha,
        elapsed_seconds=elapsed,
        production_authorizing=enforce,
    )


def run_pairwise_haar_antithetic_shard(
    *,
    coarse_state: Tensor,
    fine_plus_state: Tensor,
    fine_minus_state: Tensor,
    identity: HaarShardIdentity,
    jacobi_profile: JacobiRBCudaProfile,
    coarse_accumulator: DynkinAccumulatorState | None = None,
    fine_plus_accumulator: DynkinAccumulatorState | None = None,
    fine_minus_accumulator: DynkinAccumulatorState | None = None,
    level_runner: Callable[..., Any] = run_dynkin_refinement_shard,
    uniform_builder: Callable[..., Any] = build_certified_haar_uniform_batch,
    interval_authorizer: Callable[..., Any] = (
        sample_alpha1_rb_transition_batch_cuda_from_uniform_cells
    ),
    production_authorizing: bool = True,
    nonauthorizing_test_only: bool = False,
) -> HaarHierarchicalShardResult:
    """Run one coarse branch and its two exact antithetic fine branches."""

    if not isinstance(identity.schedule, PairwiseHaarAntitheticSchedule):
        raise TypeError("identity must contain a pairwise antithetic schedule")
    contract, enforce = _runtime_mode(
        production_authorizing=production_authorizing,
        nonauthorizing_test_only=nonauthorizing_test_only,
        uniform_builder=uniform_builder,
        interval_authorizer=interval_authorizer,
    )
    schedule = identity.schedule
    # A pair-local depth-one tree is essential: flipping all details in a
    # deeper global tree would not leave each fine pair anchored to the same
    # coarse driver.
    profile = HaarCouplingProfile(
        coarsest_steps=int(schedule.coarse_steps),
        finest_steps=int(schedule.fine_steps),
    )
    states = {
        "coarse": coarse_state,
        "fine_minus": fine_minus_state,
        "fine_plus": fine_plus_state,
    }
    accumulators: dict[str, DynkinAccumulatorState | None] = {
        "coarse": coarse_accumulator,
        "fine_minus": fine_minus_accumulator,
        "fine_plus": fine_plus_accumulator,
    }
    input_sha = _input_hash(identity, states, accumulators, jacobi_profile)
    _sync(coarse_state.device)
    started = time.perf_counter()
    branches = {
        "coarse": _run_branch(
            branch="coarse",
            states=coarse_state,
            accumulator_state=coarse_accumulator,
            sample_steps=int(schedule.coarse_steps),
            coarsest_steps=int(schedule.coarse_steps),
            coarsest_start_step=identity.coarsest_start_step,
            detail_sign=1,
            pair_coarse_steps=int(schedule.coarse_steps),
            identity=identity,
            haar_profile=profile,
            jacobi_profile=jacobi_profile,
            level_runner=level_runner,
            uniform_builder=uniform_builder,
            interval_authorizer=interval_authorizer,
            enforce_runtime_contract=enforce,
        ),
        "fine_plus": _run_branch(
            branch="fine_plus",
            states=fine_plus_state,
            accumulator_state=fine_plus_accumulator,
            sample_steps=int(schedule.fine_steps),
            coarsest_steps=int(schedule.coarse_steps),
            coarsest_start_step=identity.coarsest_start_step,
            detail_sign=1,
            pair_coarse_steps=int(schedule.coarse_steps),
            identity=identity,
            haar_profile=profile,
            jacobi_profile=jacobi_profile,
            level_runner=level_runner,
            uniform_builder=uniform_builder,
            interval_authorizer=interval_authorizer,
            enforce_runtime_contract=enforce,
        ),
        "fine_minus": _run_branch(
            branch="fine_minus",
            states=fine_minus_state,
            accumulator_state=fine_minus_accumulator,
            sample_steps=int(schedule.fine_steps),
            coarsest_steps=int(schedule.coarse_steps),
            coarsest_start_step=identity.coarsest_start_step,
            detail_sign=-1,
            pair_coarse_steps=int(schedule.coarse_steps),
            identity=identity,
            haar_profile=profile,
            jacobi_profile=jacobi_profile,
            level_runner=level_runner,
            uniform_builder=uniform_builder,
            interval_authorizer=interval_authorizer,
            enforce_runtime_contract=enforce,
        ),
    }
    _sync(coarse_state.device)
    elapsed = time.perf_counter() - started
    result = _finish_result(
        identity=identity,
        branches=branches,
        backend_contract=contract,
        input_sha256=input_sha,
        elapsed_seconds=elapsed,
        production_authorizing=enforce,
    )
    return replace(
        result,
        diagnostics={
            **dict(result.diagnostics),
            "pair_local_tree": 1,
            "coarse_executed_once": 1,
            "fine_observable_arithmetic_mean_required": 1,
            "antithetic_detail_signs": [-1, 1],
        },
    )


def initialize_nested_branch_states(
    initial_states: Tensor, schedule: NestedHaarSchedule
) -> dict[int, Tensor]:
    """Clone one common initial panel for every nested temporal level."""

    return {
        int(level): initial_states.detach().clone().contiguous()
        for level in schedule.levels or ()
    }


def initialize_antithetic_branch_states(
    initial_states: Tensor,
) -> dict[str, Tensor]:
    """Clone one common initial panel for coarse, plus, and minus branches."""

    return {
        name: initial_states.detach().clone().contiguous()
        for name in ("coarse", "fine_plus", "fine_minus")
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = _canonical_json(value) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def commit_haar_shard(
    result: HaarHierarchicalShardResult,
    directory: str | Path,
) -> dict[str, Any]:
    """Atomically commit state, observer, identity, and complete-path timing."""

    if not isinstance(result, HaarHierarchicalShardResult):
        raise TypeError("result must be a HaarHierarchicalShardResult")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stem = result.identity.fingerprint
    npz_path = root / f"{stem}.npz"
    metadata_path = root / f"{stem}.json"
    if metadata_path.exists() or npz_path.exists():
        if not (metadata_path.exists() and npz_path.exists()):
            raise HaarSchedulerError(
                "hierarchical shard has an incomplete prior commit",
                failure_code="hierarchical_shard_commit_incomplete",
            )
        existing = load_committed_haar_shard(
            root, expected_identity=result.identity, device="cpu"
        ).metadata
        if existing.get("output_sha256") != result.output_sha256:
            raise HaarSchedulerError(
                "an existing shard commit has different exact output",
                failure_code="hierarchical_shard_commit_conflict",
            )
        return {**dict(existing), "reused_existing_commit": 1}

    arrays: dict[str, np.ndarray] = {}
    array_hashes: dict[str, str] = {}
    branch_keys: dict[str, dict[str, str]] = {}
    for index, name in enumerate(sorted(result.branches)):
        branch = result.branches[name]
        prefix = f"b{index}"
        values = {
            "state": branch.committed_final_states,
            "acc_center": branch.committed_accumulator_center,
            "acc_compensation": branch.committed_accumulator_compensation,
            "acc_radius": branch.committed_accumulator_error_radius,
            "raw": branch.raw_observables,
            "dynkin": branch.dynkin_observables,
            "dynkin_radius": branch.dynkin_error_radius,
        }
        branch_keys[name] = {}
        for suffix, value in values.items():
            key = f"{prefix}_{suffix}"
            array = np.ascontiguousarray(value, dtype=np.float64)
            arrays[key] = array
            array_hashes[key] = _array_hash(array)
            branch_keys[name][suffix] = key

    io_started = time.perf_counter()
    _atomic_npz(npz_path, arrays)
    state_io_seconds = time.perf_counter() - io_started
    npz_sha = _file_hash(npz_path)
    metadata = {
        **result.to_record(),
        "state_archive": {
            "path": npz_path.name,
            "sha256": npz_sha,
            "array_hashes": array_hashes,
            "branch_keys": branch_keys,
        },
        "timing": {
            "complete_pipeline_before_file_commit_seconds": float(
                result.diagnostics[
                    "complete_pipeline_before_file_commit_seconds"
                ]
            ),
            "state_shard_io_seconds": state_io_seconds,
            "complete_pipeline_including_state_shard_io_seconds": (
                float(
                    result.diagnostics[
                        "complete_pipeline_before_file_commit_seconds"
                    ]
                )
                + state_io_seconds
            ),
            "metadata_control_plane_write_excluded": 1,
        },
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def _schedule_from_record(value: Mapping[str, Any]) -> HaarSchedule:
    name = value.get("profile_name")
    if name == NESTED_PROFILE_NAME:
        return NestedHaarSchedule(
            pool=str(value["pool"]),
            role=str(value["role"]),
            levels=tuple(int(item) for item in value["levels"]),
        )
    if name == ANTITHETIC_PROFILE_NAME:
        return PairwiseHaarAntitheticSchedule(
            coarse_steps=int(value["coarse_steps"]),
            fine_steps=int(value["fine_steps"]),
            role=str(value["role"]),
        )
    raise HaarSchedulerError(
        "committed shard has an unknown schedule",
        failure_code="hierarchical_shard_identity_invalid",
    )


def _identity_from_record(value: Mapping[str, Any]) -> HaarShardIdentity:
    return HaarShardIdentity(
        schedule=_schedule_from_record(value["schedule"]),
        path_ids=tuple(int(item) for item in value["path_ids"]),
        coarsest_start_step=int(value["coarsest_start_step"]),
        root_seed=int(value["root_seed"]),
        panel_namespace=str(value["panel_namespace"]),
        scheduler_version=str(value["scheduler_version"]),
    )


def load_committed_haar_shard(
    directory: str | Path,
    *,
    expected_identity: HaarShardIdentity,
    device: str | torch.device,
) -> HaarShardResumeState:
    """Load and verify one exact restart point without trusting filenames."""

    root = Path(directory)
    stem = expected_identity.fingerprint
    metadata_path = root / f"{stem}.json"
    if not metadata_path.exists():
        raise HaarSchedulerError(
            "hierarchical shard metadata does not exist",
            failure_code="hierarchical_shard_missing",
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema") != HAAR_SHARD_SCHEMA:
        raise HaarSchedulerError(
            "hierarchical shard schema is incompatible",
            failure_code="hierarchical_shard_schema_invalid",
        )
    actual_identity = _identity_from_record(metadata["identity"])
    if (
        actual_identity.fingerprint != expected_identity.fingerprint
        or metadata.get("identity_sha256") != expected_identity.fingerprint
    ):
        raise HaarSchedulerError(
            "hierarchical shard identity changed",
            failure_code="hierarchical_shard_identity_invalid",
        )
    archive = metadata.get("state_archive", {})
    npz_path = root / str(archive.get("path", ""))
    if (
        not npz_path.is_file()
        or _file_hash(npz_path) != archive.get("sha256")
    ):
        raise HaarSchedulerError(
            "hierarchical shard state archive failed its hash",
            failure_code="hierarchical_shard_archive_corrupt",
        )
    target_device = torch.device(device)
    states: dict[str, Tensor] = {}
    accumulators: dict[str, DynkinAccumulatorState] = {}
    raw: dict[str, np.ndarray] = {}
    dynkin: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as values:
        expected_keys = set(archive.get("array_hashes", {}))
        if set(values.files) != expected_keys:
            raise HaarSchedulerError(
                "hierarchical shard archive keys changed",
                failure_code="hierarchical_shard_archive_corrupt",
            )
        arrays: dict[str, np.ndarray] = {}
        for key in values.files:
            array = np.array(values[key], dtype=np.float64, copy=True, order="C")
            if _array_hash(array) != archive["array_hashes"].get(key):
                raise HaarSchedulerError(
                    "hierarchical shard array failed its hash",
                    failure_code="hierarchical_shard_archive_corrupt",
                    array_key=key,
                )
            arrays[key] = array
    recomputed_branch_outputs: dict[str, str] = {}
    branch_records = metadata.get("branches", {})
    for branch, keys in archive.get("branch_keys", {}).items():
        record = branch_records.get(branch)
        if not isinstance(record, Mapping):
            raise HaarSchedulerError(
                "hierarchical shard branch metadata is missing",
                failure_code="hierarchical_shard_archive_corrupt",
                branch=branch,
            )
        recomputed = _fingerprint(
            {
                "scheduler_version": HAAR_SCHEDULER_VERSION,
                "branch": branch,
                "sample_steps": int(record["sample_steps"]),
                "detail_sign": int(record["detail_sign"]),
                "completed_steps": [
                    int(value) for value in record["completed_steps"]
                ],
                "final_state_sha256": _array_hash(arrays[keys["state"]]),
                "accumulator_center_sha256": _array_hash(
                    arrays[keys["acc_center"]]
                ),
                "accumulator_compensation_sha256": _array_hash(
                    arrays[keys["acc_compensation"]]
                ),
                "accumulator_error_radius_sha256": _array_hash(
                    arrays[keys["acc_radius"]]
                ),
                "raw_observables_sha256": _array_hash(arrays[keys["raw"]]),
                "dynkin_observables_sha256": _array_hash(
                    arrays[keys["dynkin"]]
                ),
                "dynkin_error_radius_sha256": _array_hash(
                    arrays[keys["dynkin_radius"]]
                ),
                "base_output_hashes": list(record["base_output_hashes"]),
                "base_state_hashes": list(record["base_state_hashes"]),
            }
        )
        if recomputed != record.get("output_sha256"):
            raise HaarSchedulerError(
                "hierarchical branch output binding failed",
                failure_code="hierarchical_shard_archive_corrupt",
                branch=branch,
            )
        recomputed_branch_outputs[branch] = recomputed
    recomputed_output = _fingerprint(
        {
            "identity_sha256": expected_identity.fingerprint,
            "input_sha256": metadata.get("input_sha256"),
            "branches": [
                [name, recomputed_branch_outputs[name]]
                for name in sorted(recomputed_branch_outputs)
            ],
        }
    )
    if recomputed_output != metadata.get("output_sha256"):
        raise HaarSchedulerError(
            "hierarchical shard output hash failed",
            failure_code="hierarchical_shard_archive_corrupt",
        )
    for branch, keys in archive.get("branch_keys", {}).items():
        state = torch.as_tensor(
            arrays[keys["state"]], dtype=torch.float64, device=target_device
        ).contiguous()
        center = torch.as_tensor(
            arrays[keys["acc_center"]],
            dtype=torch.float64,
            device=target_device,
        ).contiguous()
        compensation = torch.as_tensor(
            arrays[keys["acc_compensation"]],
            dtype=torch.float64,
            device=target_device,
        ).contiguous()
        radius = torch.as_tensor(
            arrays[keys["acc_radius"]],
            dtype=torch.float64,
            device=target_device,
        ).contiguous()
        states[branch] = state
        accumulators[branch] = DynkinAccumulatorState(
            center=center,
            compensation=compensation,
            error_radius=radius,
        )
        raw[branch] = arrays[keys["raw"]]
        dynkin[branch] = arrays[keys["dynkin"]]
    return HaarShardResumeState(
        identity=actual_identity,
        states=states,
        accumulators=accumulators,
        raw_observables=raw,
        dynkin_observables=dynkin,
        metadata=metadata,
    )


def exact_pairwise_fine_observable_mean(
    result: HaarHierarchicalShardResult,
    *,
    use_dynkin: bool = False,
) -> np.ndarray:
    """Return the arithmetic mean of the two exact fine marginal branches."""

    if not isinstance(result.identity.schedule, PairwiseHaarAntitheticSchedule):
        raise TypeError("result is not pairwise antithetic")
    name = "dynkin_observables" if use_dynkin else "raw_observables"
    plus = getattr(result.branches["fine_plus"], name)
    minus = getattr(result.branches["fine_minus"], name)
    if plus.shape != minus.shape:
        raise HaarSchedulerError(
            "antithetic fine branches have inconsistent observables",
            failure_code="hierarchical_antithetic_shape_invalid",
        )
    return 0.5 * (plus + minus)


__all__ = [
    "ADJACENT_LEVEL_PAIRS",
    "ANTITHETIC_PROFILE_NAME",
    "HAAR_SCHEDULER_VERSION",
    "HAAR_SHARD_SCHEMA",
    "MAX_NORMAL_FALLBACK_FRACTION",
    "MAX_NORMAL_FALLBACK_TIME_FRACTION",
    "NESTED_MAIN_LEVELS",
    "NESTED_PROFILE_NAME",
    "NESTED_REFERENCE_LEVELS",
    "HaarBackendContract",
    "HaarBranchResult",
    "HaarHierarchicalShardResult",
    "HaarSchedule",
    "HaarSchedulerError",
    "HaarShardIdentity",
    "HaarShardResumeState",
    "NestedHaarSchedule",
    "PairwiseHaarAntitheticSchedule",
    "commit_haar_shard",
    "canonical_haar_scheduler_transition_ids",
    "exact_pairwise_fine_observable_mean",
    "expected_haar_shard_input_sha256",
    "initialize_antithetic_branch_states",
    "initialize_nested_branch_states",
    "inspect_haar_backend_contract",
    "load_committed_haar_shard",
    "require_production_haar_backend",
    "run_nested_haar_shard",
    "run_pairwise_haar_antithetic_shard",
]
