"""Paired-mixture streams for the D0 density-ratio controls.

This module is deliberately additive.  The first density-ratio workflow uses
independent positive/reference draws.  The versioned stream below instead
couples the bounded teacher and its Dirichlet reference through common Gamma
variables and supports deterministic gradient accumulation.  It contains no
filesystem orchestration, experiment gate, physical-score training, or
sampler code.

For one matched-time teacher cluster, let ``G_i ~ Gamma(1)``,
``E ~ Gamma(1)``, and ``J ~ w(tau)`` be independent.  The two states are

``S0 = G / sum(G)`` and ``SJ = (G + E e_J) / (sum(G) + E)``.

Thus ``S0`` is exactly ``Dirichlet(1)`` and, conditional on ``J=j``, ``SJ``
is exactly ``Dirichlet(1 + e_j)``.  The unbiased per-cluster objective is

``.5(1-eps) softplus(-l(S0)) + .5 eps softplus(-l(SJ))``
``+ .5 softplus(l(S0))``.

The stationary null uses two independent ``Dirichlet(1)`` draws at the same
time anchor.  A stateless swap assigns the pooled pair to positive/reference
roles, so no class-specific sampling namespace is present.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mnist.d0_score_boundary_controls import (
    BOUNDED_TEACHER_VERSION,
    bounded_teacher_anchor_indices,
    bounded_teacher_weights,
)
from mnist.d0_score_optimizer_scale import parameter_gradient_l2_norm


PAIRED_MIXTURE_SCHEMA = "experiment12-d0-score-density-ratio-paired-mixture"
PAIRED_MIXTURE_SCHEMA_VERSION = 1
PAIRED_MIXTURE_STREAM_VERSION = "d0-density-ratio-paired-mixture-stream-v1"
PAIRED_MIXTURE_OBJECTIVE_VERSION = (
    "d0-density-ratio-paired-mixture-weighted-softplus-v1"
)
PAIRED_MIXTURE_ACCUMULATION_VERSION = (
    "d0-density-ratio-deterministic-gradient-accumulation-v1"
)
PAIRED_MIXTURE_PRODUCTION_ROOT_SEED = 260851
PAIRED_MIXTURE_CLUSTER_BIN_COUNTS = (4, 4, 4, 4, 16)
PAIRED_MIXTURE_MICROBATCH_CLUSTERS = 32
PAIRED_MIXTURE_ACCUMULATION_LEVELS = (2, 4, 8)
PAIRED_MIXTURE_TASKS = ("bounded_teacher", "dirichlet_null")


__all__ = [
    "PAIRED_MIXTURE_SCHEMA",
    "PAIRED_MIXTURE_SCHEMA_VERSION",
    "PAIRED_MIXTURE_STREAM_VERSION",
    "PAIRED_MIXTURE_OBJECTIVE_VERSION",
    "PAIRED_MIXTURE_ACCUMULATION_VERSION",
    "PAIRED_MIXTURE_PRODUCTION_ROOT_SEED",
    "PAIRED_MIXTURE_CLUSTER_BIN_COUNTS",
    "PAIRED_MIXTURE_MICROBATCH_CLUSTERS",
    "PAIRED_MIXTURE_ACCUMULATION_LEVELS",
    "PairedMixtureStreamPlan",
    "PairedMixtureMicrobatch",
    "AccumulatedPairedMixtureStream",
    "AccumulatedGradientDiagnostics",
    "build_paired_mixture_stream_plan",
    "paired_mixture_stream_plan_record",
    "derive_paired_mixture_seed",
    "generate_paired_mixture_microbatch",
    "generate_accumulated_paired_stream",
    "paired_mixture_replay_record",
    "verify_paired_mixture_replay",
    "weighted_paired_softplus_components",
    "weighted_paired_softplus_loss",
    "backward_accumulated_objective",
    "accumulation_diagnostics",
]


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_fingerprint(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_positive(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


def _task(value: str) -> str:
    task = str(value)
    if task not in PAIRED_MIXTURE_TASKS:
        raise ValueError(f"unsupported paired-mixture task {task!r}")
    return task


def _plan_payload(plan: "PairedMixtureStreamPlan") -> dict[str, Any]:
    value = asdict(plan)
    value.update(
        {
            "cluster_bin_counts": list(plan.cluster_bin_counts),
            "accumulation_levels": list(plan.accumulation_levels),
            "effective_cluster_counts": [
                int(plan.microbatch_clusters) * int(level)
                for level in plan.accumulation_levels
            ],
            "teacher_version": BOUNDED_TEACHER_VERSION,
            "stream_version": PAIRED_MIXTURE_STREAM_VERSION,
            "objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
            "accumulation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
            "production_root_seed": PAIRED_MIXTURE_PRODUCTION_ROOT_SEED,
            "production_defaults_match": int(
                int(plan.root_seed) == PAIRED_MIXTURE_PRODUCTION_ROOT_SEED
                and tuple(plan.cluster_bin_counts)
                == PAIRED_MIXTURE_CLUSTER_BIN_COUNTS
                and int(plan.microbatch_clusters)
                == PAIRED_MIXTURE_MICROBATCH_CLUSTERS
                and tuple(plan.accumulation_levels)
                == PAIRED_MIXTURE_ACCUMULATION_LEVELS
                and math.isclose(
                    float(plan.teacher_epsilon), 0.5, rel_tol=0.0, abs_tol=0.0
                )
            ),
        }
    )
    return value


@dataclass(frozen=True)
class PairedMixtureStreamPlan:
    """Immutable scientific plan for paired-mixture training streams."""

    root_seed: int
    grid_size: int
    horizon: float
    label: int = 3
    teacher_epsilon: float = 0.5
    cluster_bin_counts: tuple[int, ...] = PAIRED_MIXTURE_CLUSTER_BIN_COUNTS
    microbatch_clusters: int = PAIRED_MIXTURE_MICROBATCH_CLUSTERS
    accumulation_levels: tuple[int, ...] = PAIRED_MIXTURE_ACCUMULATION_LEVELS
    schema: str = PAIRED_MIXTURE_SCHEMA + "-stream-plan"
    schema_version: int = PAIRED_MIXTURE_SCHEMA_VERSION
    derivation_version: str = PAIRED_MIXTURE_STREAM_VERSION

    def __post_init__(self) -> None:
        if self.schema != PAIRED_MIXTURE_SCHEMA + "-stream-plan":
            raise ValueError("incompatible paired-mixture stream-plan schema")
        if int(self.schema_version) != PAIRED_MIXTURE_SCHEMA_VERSION:
            raise ValueError("incompatible paired-mixture schema version")
        if self.derivation_version != PAIRED_MIXTURE_STREAM_VERSION:
            raise ValueError("incompatible paired-mixture derivation version")
        if int(self.grid_size) <= 0 or int(self.grid_size) % 4:
            raise ValueError("grid_size must be a positive multiple of four")
        if not _finite_positive(self.horizon):
            raise ValueError("horizon must be finite and positive")
        if int(self.label) < 0:
            raise ValueError("label must be nonnegative")
        if not math.isfinite(float(self.teacher_epsilon)) or not (
            0.0 < float(self.teacher_epsilon) < 1.0
        ):
            raise ValueError("teacher_epsilon must lie strictly inside (0,1)")
        if tuple(int(value) for value in self.cluster_bin_counts) != (
            PAIRED_MIXTURE_CLUSTER_BIN_COUNTS
        ):
            raise ValueError("paired-mixture v1 freezes bin counts at 4,4,4,4,16")
        if int(self.microbatch_clusters) != PAIRED_MIXTURE_MICROBATCH_CLUSTERS:
            raise ValueError("paired-mixture v1 freezes microbatches at 32 clusters")
        if tuple(int(value) for value in self.accumulation_levels) != (
            PAIRED_MIXTURE_ACCUMULATION_LEVELS
        ):
            raise ValueError("paired-mixture v1 freezes accumulation at 2,4,8")

    @property
    def pixels(self) -> int:
        return int(self.grid_size) ** 2

    @property
    def fingerprint(self) -> str:
        return _canonical_fingerprint(_plan_payload(self))


def build_paired_mixture_stream_plan(
    *,
    grid_size: int,
    horizon: float,
    root_seed: int = PAIRED_MIXTURE_PRODUCTION_ROOT_SEED,
    label: int = 3,
    teacher_epsilon: float = 0.5,
) -> PairedMixtureStreamPlan:
    return PairedMixtureStreamPlan(
        root_seed=int(root_seed),
        grid_size=int(grid_size),
        horizon=float(horizon),
        label=int(label),
        teacher_epsilon=float(teacher_epsilon),
    )


def paired_mixture_stream_plan_record(
    plan: PairedMixtureStreamPlan,
) -> dict[str, Any]:
    return {
        **_plan_payload(plan),
        "fingerprint": plan.fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def derive_paired_mixture_seed(
    plan_or_root_seed: PairedMixtureStreamPlan | int,
    phase: str,
    task: str,
    optimizer_step: int,
    microbatch_index: int,
    namespace: str,
) -> int:
    root_seed = (
        int(plan_or_root_seed.root_seed)
        if isinstance(plan_or_root_seed, PairedMixtureStreamPlan)
        else int(plan_or_root_seed)
    )
    phase_value = str(phase)
    task_value = _task(task)
    namespace_value = str(namespace)
    if not phase_value or not namespace_value:
        raise ValueError("phase and namespace must be nonempty")
    if int(optimizer_step) < 0 or int(microbatch_index) < 0:
        raise ValueError("optimizer_step and microbatch_index must be nonnegative")
    payload = json.dumps(
        [
            PAIRED_MIXTURE_STREAM_VERSION,
            root_seed,
            phase_value,
            task_value,
            int(optimizer_step),
            int(microbatch_index),
            namespace_value,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _time_template(dtype: torch.dtype) -> tuple[Tensor, np.ndarray, np.ndarray]:
    fractions: list[float] = []
    strata: list[int] = []
    anchors: list[int] = []
    anchor = 0
    for bin_index, count in enumerate(PAIRED_MIXTURE_CLUSTER_BIN_COUNTS):
        for offset in range(int(count)):
            fractions.append(
                (float(bin_index) + (float(offset) + 0.5) / float(count)) / 5.0
            )
            strata.append(bin_index)
            anchors.append(anchor)
            anchor += 1
    return (
        torch.tensor(fractions, dtype=dtype),
        np.asarray(strata, dtype=np.int64),
        np.asarray(anchors, dtype=np.int64),
    )


def _gamma_ones(
    shape: Sequence[int], *, seed: int, dtype: torch.dtype
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch._standard_gamma(
        torch.ones(tuple(int(value) for value in shape), dtype=dtype),
        generator=generator,
    )


def _microbatch_payload(
    *,
    reference_states: Tensor,
    component_states: Tensor,
    tau_fraction: Tensor,
    component_indices: Tensor,
    swap_bits: Tensor,
    base_gamma_sums: Tensor,
    tilt_increments: Tensor,
    strata: np.ndarray,
    anchor_ids: np.ndarray,
    plan: PairedMixtureStreamPlan,
    phase: str,
    task: str,
    optimizer_step: int,
    microbatch_index: int,
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema": PAIRED_MIXTURE_SCHEMA + "-microbatch",
        "schema_version": PAIRED_MIXTURE_SCHEMA_VERSION,
        "derivation_version": PAIRED_MIXTURE_STREAM_VERSION,
        "phase": str(phase),
        "task": str(task),
        "optimizer_step": int(optimizer_step),
        "microbatch_index": int(microbatch_index),
        "clusters": int(reference_states.shape[0]),
        "reference_state_sha256": _tensor_fingerprint(reference_states),
        "component_state_sha256": _tensor_fingerprint(component_states),
        "tau_fraction_sha256": _tensor_fingerprint(tau_fraction),
        "component_indices_sha256": _tensor_fingerprint(component_indices),
        "swap_bits_sha256": _tensor_fingerprint(swap_bits),
        "base_gamma_sums_sha256": _tensor_fingerprint(base_gamma_sums),
        "tilt_increments_sha256": _tensor_fingerprint(tilt_increments),
        "strata_sha256": _array_fingerprint(strata),
        "anchor_ids_sha256": _array_fingerprint(anchor_ids),
        "seeds": {str(key): int(value) for key, value in seeds.items()},
        "plan_fingerprint": plan.fingerprint,
        "objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
    }


@dataclass(frozen=True)
class PairedMixtureMicrobatch:
    """One stateless 32-cluster paired-mixture microbatch."""

    reference_states: Tensor
    component_states: Tensor
    tau: Tensor
    tau_fraction: Tensor
    labels: Tensor
    component_indices: Tensor
    swap_bits: Tensor
    base_gamma_sums: Tensor
    tilt_increments: Tensor
    strata: np.ndarray
    anchor_ids: np.ndarray
    phase: str
    task: str
    optimizer_step: int
    microbatch_index: int
    grid_size: int
    teacher_epsilon: float
    seeds: Mapping[str, int]
    plan_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        rows = int(self.reference_states.shape[0])
        pixels = int(self.grid_size) ** 2
        if self.reference_states.shape != (32, pixels):
            raise ValueError("reference states must have shape (32, grid_size^2)")
        if self.component_states.shape != self.reference_states.shape:
            raise ValueError("paired component/reference state shapes disagree")
        if any(
            value.shape != (rows,)
            for value in (
                self.tau,
                self.tau_fraction,
                self.labels,
                self.component_indices,
                self.swap_bits,
                self.base_gamma_sums,
                self.tilt_increments,
            )
        ):
            raise ValueError("paired-mixture metadata tensor axes disagree")
        if any(
            np.asarray(value).shape != (rows,)
            for value in (self.strata, self.anchor_ids)
        ):
            raise ValueError("paired-mixture NumPy metadata axes disagree")
        task = _task(self.task)
        if not self.phase or int(self.optimizer_step) < 0 or int(self.microbatch_index) < 0:
            raise ValueError("invalid phase or stream cursor")
        for states in (self.reference_states, self.component_states):
            if not bool(torch.isfinite(states).all() and (states > 0.0).all()):
                raise ValueError("paired-mixture states must be finite and positive")
            tolerance = 2e-12 if states.dtype == torch.float64 else 2e-6
            if float((states.sum(1) - 1.0).abs().max().detach().cpu()) > tolerance:
                raise ValueError("paired-mixture states are not simplex-valued")
        counts = tuple(
            int(np.count_nonzero(np.asarray(self.strata) == index))
            for index in range(5)
        )
        if counts != PAIRED_MIXTURE_CLUSTER_BIN_COUNTS:
            raise ValueError("paired-mixture time-bin counts are invalid")
        if set(np.asarray(self.anchor_ids).tolist()) != set(range(32)):
            raise ValueError("paired-mixture anchor IDs must be a permutation of 0..31")
        if task == "bounded_teacher":
            if not bool(torch.all((self.component_indices >= 0) & (self.component_indices < 4))):
                raise ValueError("teacher component indices must lie in 0..3")
            if not bool(torch.all(self.swap_bits == -1)):
                raise ValueError("teacher batches must not carry null swaps")
            if not bool(
                torch.isfinite(self.base_gamma_sums).all()
                and torch.isfinite(self.tilt_increments).all()
                and (self.base_gamma_sums > 0.0).all()
                and (self.tilt_increments > 0.0).all()
            ):
                raise ValueError("teacher Gamma diagnostics must be positive")
            anchors = bounded_teacher_anchor_indices(
                int(self.grid_size), device=self.reference_states.device
            )[self.component_indices]
            gamma = self.reference_states * self.base_gamma_sums[:, None]
            gamma = gamma.clone()
            gamma[torch.arange(rows, device=gamma.device), anchors] += (
                self.tilt_increments
            )
            reconstructed = gamma / (
                self.base_gamma_sums + self.tilt_increments
            )[:, None]
            tolerance = 2e-12 if gamma.dtype == torch.float64 else 2e-6
            if float((reconstructed - self.component_states).abs().max().detach().cpu()) > tolerance:
                raise ValueError("teacher states do not satisfy the common-Gamma identity")
        else:
            if not bool(torch.all(self.component_indices == -1)):
                raise ValueError("null batches must not carry teacher components")
            if not bool(torch.all((self.swap_bits == 0) | (self.swap_bits == 1))):
                raise ValueError("null swap bits must be binary")
            if not bool(
                torch.all(self.base_gamma_sums == 0.0)
                and torch.all(self.tilt_increments == 0.0)
            ):
                raise ValueError("null batches must not carry teacher Gamma diagnostics")

    @property
    def clusters(self) -> int:
        return int(self.reference_states.shape[0])

    def record(self) -> dict[str, Any]:
        component = self.component_indices.detach().cpu().numpy()
        swaps = self.swap_bits.detach().cpu().numpy()
        return {
            "schema": PAIRED_MIXTURE_SCHEMA + "-microbatch",
            "schema_version": PAIRED_MIXTURE_SCHEMA_VERSION,
            "derivation_version": PAIRED_MIXTURE_STREAM_VERSION,
            "phase": self.phase,
            "task": self.task,
            "optimizer_step": int(self.optimizer_step),
            "microbatch_index": int(self.microbatch_index),
            "clusters": self.clusters,
            "grid_size": int(self.grid_size),
            "teacher_epsilon": float(self.teacher_epsilon),
            "time_bin_counts": [
                int(np.count_nonzero(self.strata == index)) for index in range(5)
            ],
            "component_counts": [
                int(np.count_nonzero(component == index)) for index in range(4)
            ],
            "swap_counts": [
                int(np.count_nonzero(swaps == index)) for index in range(2)
            ],
            "reference_state_sha256": _tensor_fingerprint(self.reference_states),
            "component_state_sha256": _tensor_fingerprint(self.component_states),
            "tau_fraction_sha256": _tensor_fingerprint(self.tau_fraction),
            "component_indices_sha256": _tensor_fingerprint(self.component_indices),
            "swap_bits_sha256": _tensor_fingerprint(self.swap_bits),
            "base_gamma_sums_sha256": _tensor_fingerprint(self.base_gamma_sums),
            "tilt_increments_sha256": _tensor_fingerprint(self.tilt_increments),
            "strata_sha256": _array_fingerprint(self.strata),
            "anchor_ids_sha256": _array_fingerprint(self.anchor_ids),
            "seeds": {str(key): int(value) for key, value in self.seeds.items()},
            "plan_fingerprint": self.plan_fingerprint,
            "fingerprint": self.fingerprint,
            "common_gamma_teacher": int(self.task == "bounded_teacher"),
            "null_pooled_stateless_swaps": int(self.task == "dirichlet_null"),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }


def generate_paired_mixture_microbatch(
    plan: PairedMixtureStreamPlan,
    *,
    phase: str,
    task: str,
    optimizer_step: int,
    microbatch_index: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PairedMixtureMicrobatch:
    """Generate one exact, stateless paired-mixture microbatch on CPU."""

    if dtype not in (torch.float32, torch.float64):
        raise ValueError("paired-mixture dtype must be float32 or float64")
    phase_value = str(phase)
    task_value = _task(task)
    step_value = int(optimizer_step)
    micro_value = int(microbatch_index)
    if not phase_value or step_value < 0 or micro_value < 0:
        raise ValueError("invalid paired-mixture stream cursor")

    fractions, strata, anchor_ids = _time_template(dtype)
    rows, pixels = int(plan.microbatch_clusters), int(plan.pixels)
    common_names = ("cluster-permutation",)
    teacher_names = ("common-base-gamma", "tilt-increment", "mixture-choice")
    null_names = ("null-pool", "null-swaps")
    names = common_names + (teacher_names if task_value == "bounded_teacher" else null_names)
    seeds = {
        name: derive_paired_mixture_seed(
            plan, phase_value, task_value, step_value, micro_value, name
        )
        for name in names
    }

    if task_value == "bounded_teacher":
        gamma = _gamma_ones(
            (rows, pixels), seed=seeds["common-base-gamma"], dtype=dtype
        )
        increments = _gamma_ones(
            (rows,), seed=seeds["tilt-increment"], dtype=dtype
        )
        base_sums = gamma.sum(dim=1)
        reference = gamma / base_sums[:, None]
        choice_generator = torch.Generator(device="cpu").manual_seed(
            int(seeds["mixture-choice"])
        )
        uniforms = torch.rand((rows,), generator=choice_generator, dtype=dtype)
        weights = bounded_teacher_weights(fractions).to(dtype=dtype)
        choices = (
            uniforms[:, None] > weights.cumsum(dim=1)
        ).sum(dim=1).clamp_max(3).to(torch.long)
        selected = bounded_teacher_anchor_indices(int(plan.grid_size))[choices]
        tilted_gamma = gamma.clone()
        tilted_gamma[torch.arange(rows), selected] += increments
        component = tilted_gamma / (base_sums + increments)[:, None]
        swaps = torch.full((rows,), -1, dtype=torch.long)
    else:
        pooled_gamma = _gamma_ones(
            (2 * rows, pixels), seed=seeds["null-pool"], dtype=dtype
        )
        pooled = (
            pooled_gamma / pooled_gamma.sum(dim=1, keepdim=True)
        ).reshape(rows, 2, pixels)
        swap_generator = torch.Generator(device="cpu").manual_seed(
            int(seeds["null-swaps"])
        )
        swaps = torch.randint(
            0, 2, (rows,), generator=swap_generator, dtype=torch.long
        )
        row_index = torch.arange(rows)
        component = pooled[row_index, swaps]
        reference = pooled[row_index, 1 - swaps]
        choices = torch.full((rows,), -1, dtype=torch.long)
        base_sums = torch.zeros((rows,), dtype=dtype)
        increments = torch.zeros((rows,), dtype=dtype)

    permutation_generator = torch.Generator(device="cpu").manual_seed(
        int(seeds["cluster-permutation"])
    )
    permutation = torch.randperm(rows, generator=permutation_generator)
    permutation_np = permutation.numpy()
    reference = reference.index_select(0, permutation).contiguous()
    component = component.index_select(0, permutation).contiguous()
    fractions = fractions.index_select(0, permutation).contiguous()
    choices = choices.index_select(0, permutation).contiguous()
    swaps = swaps.index_select(0, permutation).contiguous()
    base_sums = base_sums.index_select(0, permutation).contiguous()
    increments = increments.index_select(0, permutation).contiguous()
    strata = strata[permutation_np]
    anchor_ids = anchor_ids[permutation_np]

    payload = _microbatch_payload(
        reference_states=reference,
        component_states=component,
        tau_fraction=fractions,
        component_indices=choices,
        swap_bits=swaps,
        base_gamma_sums=base_sums,
        tilt_increments=increments,
        strata=strata,
        anchor_ids=anchor_ids,
        plan=plan,
        phase=phase_value,
        task=task_value,
        optimizer_step=step_value,
        microbatch_index=micro_value,
        seeds=seeds,
    )
    target = torch.device(device)
    return PairedMixtureMicrobatch(
        reference_states=reference.to(target),
        component_states=component.to(target),
        tau=(fractions * float(plan.horizon)).to(target),
        tau_fraction=fractions.to(target),
        labels=torch.full(
            (rows,), int(plan.label), device=target, dtype=torch.long
        ),
        component_indices=choices.to(target),
        swap_bits=swaps.to(target),
        base_gamma_sums=base_sums.to(target),
        tilt_increments=increments.to(target),
        strata=strata,
        anchor_ids=anchor_ids,
        phase=phase_value,
        task=task_value,
        optimizer_step=step_value,
        microbatch_index=micro_value,
        grid_size=int(plan.grid_size),
        teacher_epsilon=float(plan.teacher_epsilon),
        seeds=seeds,
        plan_fingerprint=plan.fingerprint,
        fingerprint=_canonical_fingerprint(payload),
    )


def _accumulated_payload(
    *,
    plan: PairedMixtureStreamPlan,
    phase: str,
    task: str,
    optimizer_step: int,
    accumulation_level: int,
    canonical_microbatches: Sequence[PairedMixtureMicrobatch],
) -> dict[str, Any]:
    return {
        "schema": PAIRED_MIXTURE_SCHEMA + "-accumulated-stream",
        "schema_version": PAIRED_MIXTURE_SCHEMA_VERSION,
        "derivation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
        "phase": str(phase),
        "task": str(task),
        "optimizer_step": int(optimizer_step),
        "accumulation_level": int(accumulation_level),
        "microbatch_clusters": int(plan.microbatch_clusters),
        "effective_clusters": int(accumulation_level) * int(plan.microbatch_clusters),
        "canonical_microbatch_fingerprints": [
            value.fingerprint for value in canonical_microbatches
        ],
        "plan_fingerprint": plan.fingerprint,
        "objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
    }


@dataclass(frozen=True)
class AccumulatedPairedMixtureStream:
    """One optimizer step's order-independent set of microbatches."""

    microbatches: tuple[PairedMixtureMicrobatch, ...]
    requested_order: tuple[int, ...]
    phase: str
    task: str
    optimizer_step: int
    accumulation_level: int
    microbatch_clusters: int
    plan_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        level = int(self.accumulation_level)
        if level not in PAIRED_MIXTURE_ACCUMULATION_LEVELS:
            raise ValueError("accumulation_level must be one of 2,4,8")
        if len(self.microbatches) != level:
            raise ValueError("accumulated stream has the wrong microbatch count")
        if tuple(sorted(self.requested_order)) != tuple(range(level)):
            raise ValueError("requested_order must be a complete permutation")
        indices = tuple(value.microbatch_index for value in self.microbatches)
        if indices != self.requested_order:
            raise ValueError("microbatch tuple does not follow requested_order")
        if any(
            value.phase != self.phase
            or value.task != self.task
            or int(value.optimizer_step) != int(self.optimizer_step)
            or value.plan_fingerprint != self.plan_fingerprint
            for value in self.microbatches
        ):
            raise ValueError("accumulated microbatch binding mismatch")

    @property
    def effective_clusters(self) -> int:
        return int(self.accumulation_level) * int(self.microbatch_clusters)

    @property
    def canonical_microbatches(self) -> tuple[PairedMixtureMicrobatch, ...]:
        return tuple(sorted(self.microbatches, key=lambda value: value.microbatch_index))

    def record(self) -> dict[str, Any]:
        return {
            "schema": PAIRED_MIXTURE_SCHEMA + "-accumulated-stream",
            "schema_version": PAIRED_MIXTURE_SCHEMA_VERSION,
            "derivation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
            "phase": self.phase,
            "task": self.task,
            "optimizer_step": int(self.optimizer_step),
            "accumulation_level": int(self.accumulation_level),
            "microbatch_clusters": int(self.microbatch_clusters),
            "effective_clusters": self.effective_clusters,
            "canonical_microbatches": [
                value.record() for value in self.canonical_microbatches
            ],
            "canonical_microbatch_fingerprints": [
                value.fingerprint for value in self.canonical_microbatches
            ],
            # The execution order is diagnostic only and is intentionally not
            # part of the scientific fingerprint.
            "requested_order": list(self.requested_order),
            "plan_fingerprint": self.plan_fingerprint,
            "fingerprint": self.fingerprint,
            "order_invariant_fingerprint": 1,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }


def generate_accumulated_paired_stream(
    plan: PairedMixtureStreamPlan,
    *,
    phase: str,
    task: str,
    optimizer_step: int,
    accumulation_level: int,
    microbatch_order: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AccumulatedPairedMixtureStream:
    level = int(accumulation_level)
    if level not in tuple(plan.accumulation_levels):
        raise ValueError("accumulation_level is not present in the frozen plan")
    order = tuple(range(level)) if microbatch_order is None else tuple(
        int(value) for value in microbatch_order
    )
    if tuple(sorted(order)) != tuple(range(level)):
        raise ValueError("microbatch_order must be a permutation of 0..level-1")
    by_index = {
        index: generate_paired_mixture_microbatch(
            plan,
            phase=str(phase),
            task=str(task),
            optimizer_step=int(optimizer_step),
            microbatch_index=index,
            device=device,
            dtype=dtype,
        )
        for index in range(level)
    }
    canonical = tuple(by_index[index] for index in range(level))
    payload = _accumulated_payload(
        plan=plan,
        phase=str(phase),
        task=str(task),
        optimizer_step=int(optimizer_step),
        accumulation_level=level,
        canonical_microbatches=canonical,
    )
    return AccumulatedPairedMixtureStream(
        microbatches=tuple(by_index[index] for index in order),
        requested_order=order,
        phase=str(phase),
        task=_task(task),
        optimizer_step=int(optimizer_step),
        accumulation_level=level,
        microbatch_clusters=int(plan.microbatch_clusters),
        plan_fingerprint=plan.fingerprint,
        fingerprint=_canonical_fingerprint(payload),
    )


def paired_mixture_replay_record(
    plan: PairedMixtureStreamPlan,
    *,
    phase: str,
    task: str,
    optimizer_step: int,
    accumulation_level: int,
) -> dict[str, Any]:
    stream = generate_accumulated_paired_stream(
        plan,
        phase=str(phase),
        task=str(task),
        optimizer_step=int(optimizer_step),
        accumulation_level=int(accumulation_level),
        device="cpu",
        dtype=torch.float32,
    )
    record = {
        "schema": PAIRED_MIXTURE_SCHEMA + "-replay",
        "schema_version": PAIRED_MIXTURE_SCHEMA_VERSION,
        "derivation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
        "plan_fingerprint": plan.fingerprint,
        "phase": str(phase),
        "task": str(task),
        "optimizer_step": int(optimizer_step),
        "accumulation_level": int(accumulation_level),
        "stream_fingerprint": stream.fingerprint,
        "canonical_microbatch_fingerprints": [
            value.fingerprint for value in stream.canonical_microbatches
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    record["fingerprint"] = _canonical_fingerprint(record)
    return record


def verify_paired_mixture_replay(
    plan: PairedMixtureStreamPlan, record: Mapping[str, Any]
) -> dict[str, Any]:
    reason: str | None = None
    actual: str | None = None
    expected_fingerprint: str | None = None
    try:
        expected = paired_mixture_replay_record(
            plan,
            phase=str(record["phase"]),
            task=str(record["task"]),
            optimizer_step=int(record["optimizer_step"]),
            accumulation_level=int(record["accumulation_level"]),
        )
        expected_fingerprint = str(expected["fingerprint"])
        actual = _canonical_fingerprint(
            {key: value for key, value in dict(record).items() if key != "fingerprint"}
        )
        passed = int(
            actual == expected_fingerprint
            and str(record.get("fingerprint", "")) == expected_fingerprint
            and str(record.get("plan_fingerprint", "")) == plan.fingerprint
        )
        if not passed:
            reason = "paired-mixture replay fingerprint differs"
    except Exception as exc:
        passed = 0
        reason = f"{type(exc).__name__}: {exc}"
    return {
        "passed": int(passed),
        "reason": reason,
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual,
        "plan_fingerprint": plan.fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _logit_vectors(
    reference_logits: Tensor, component_logits: Tensor
) -> tuple[Tensor, Tensor]:
    reference = reference_logits.reshape(-1)
    component = component_logits.reshape(-1)
    if reference.shape != component.shape or reference.numel() <= 0:
        raise ValueError("paired logits must have the same nonempty shape")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(component).all()):
        raise FloatingPointError("paired logits must be finite")
    return reference, component


def weighted_paired_softplus_components(
    reference_logits: Tensor,
    component_logits: Tensor,
    *,
    task: str,
    teacher_epsilon: float = 0.5,
) -> dict[str, Tensor]:
    """Return elementwise unbiased objective components.

    For the teacher, ``reference_logits`` are logits at ``S0`` and
    ``component_logits`` are logits at the sampled ``SJ``.  For the null they
    are, respectively, the negative and positive pooled/swapped states.
    """

    reference, component = _logit_vectors(reference_logits, component_logits)
    task_value = _task(task)
    epsilon = float(teacher_epsilon)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError("teacher_epsilon must lie strictly inside (0,1)")
    if task_value == "bounded_teacher":
        base_positive = 0.5 * (1.0 - epsilon) * F.softplus(-reference)
        mixture_positive = 0.5 * epsilon * F.softplus(-component)
        reference_negative = 0.5 * F.softplus(reference)
    else:
        base_positive = torch.zeros_like(reference)
        mixture_positive = 0.5 * F.softplus(-component)
        reference_negative = 0.5 * F.softplus(reference)
    return {
        "base_positive": base_positive,
        "mixture_positive": mixture_positive,
        "reference_negative": reference_negative,
        "total": base_positive + mixture_positive + reference_negative,
    }


def weighted_paired_softplus_loss(
    reference_logits: Tensor,
    component_logits: Tensor,
    *,
    task: str,
    teacher_epsilon: float = 0.5,
    reduction: str = "mean",
) -> Tensor:
    components = weighted_paired_softplus_components(
        reference_logits,
        component_logits,
        task=task,
        teacher_epsilon=teacher_epsilon,
    )
    total = components["total"]
    if reduction == "none":
        return total
    if reduction == "mean":
        return total.mean()
    if reduction == "sum":
        return total.sum()
    raise ValueError("reduction must be none, mean, or sum")


@dataclass(frozen=True)
class AccumulatedGradientDiagnostics:
    """Diagnostics for one correctly weighted accumulated backward pass."""

    microbatch_count: int
    cluster_count: int
    expected_microbatches: int
    expected_clusters: int
    unscaled_objective: float
    scaled_objective: float
    loss_scale: float
    raw_gradient_norm: float
    scaled_gradient_norm: float
    weight_sum: float
    finite: int
    accumulation_version: str = PAIRED_MIXTURE_ACCUMULATION_VERSION
    physical_training_performed: int = 0
    sampling_performed: int = 0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def backward_accumulated_objective(
    objective_batches: Iterable[tuple[Tensor, int]],
    parameters: Iterable[nn.Parameter],
    *,
    expected_microbatches: int,
    expected_clusters: int,
    loss_scale: float = 1.0,
) -> AccumulatedGradientDiagnostics:
    """Backward the exact cluster-weighted mean without retaining all graphs.

    The caller must clear gradients first.  Each yielded scalar is the mean
    unscaled loss over ``cluster_count`` clusters.  Weighting by
    ``cluster_count / expected_clusters`` makes the accumulated gradient equal
    to a single backward pass through the concatenated objective, up to normal
    floating-point accumulation order.
    """

    expected_batches = int(expected_microbatches)
    expected_count = int(expected_clusters)
    scale = float(loss_scale)
    if expected_batches <= 0 or expected_count <= 0:
        raise ValueError("expected accumulation sizes must be positive")
    if not _finite_positive(scale):
        raise ValueError("loss_scale must be finite and positive")
    trainable = [value for value in parameters if value.requires_grad]
    if not trainable:
        raise ValueError("parameters contains no trainable parameters")
    if any(
        value.grad is not None and bool(torch.any(value.grad.detach() != 0.0))
        for value in trainable
    ):
        raise ValueError("gradients must be clear before accumulation")

    seen_batches = 0
    seen_clusters = 0
    objective = 0.0
    weight_sum = 0.0
    for loss, cluster_count_value in objective_batches:
        cluster_count = int(cluster_count_value)
        if cluster_count <= 0:
            raise ValueError("microbatch cluster counts must be positive")
        if not isinstance(loss, Tensor) or loss.numel() != 1:
            raise TypeError("each accumulated loss must be a scalar tensor")
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("accumulated objective is non-finite")
        if seen_batches >= expected_batches or seen_clusters + cluster_count > expected_count:
            raise ValueError("accumulated objective exceeds its declared size")
        weight = float(cluster_count) / float(expected_count)
        (loss * (scale * weight)).backward()
        objective += float(loss.detach().cpu()) * weight
        weight_sum += weight
        seen_batches += 1
        seen_clusters += cluster_count

    if seen_batches != expected_batches or seen_clusters != expected_count:
        raise ValueError(
            "accumulated objective did not yield the declared microbatches/clusters"
        )
    scaled_norm = float(parameter_gradient_l2_norm(trainable).detach().cpu())
    raw_norm = scaled_norm / scale
    finite = int(
        math.isfinite(objective)
        and math.isfinite(scaled_norm)
        and math.isfinite(raw_norm)
        and math.isclose(weight_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    )
    if not finite:
        raise FloatingPointError("accumulated gradient diagnostics are non-finite")
    return AccumulatedGradientDiagnostics(
        microbatch_count=seen_batches,
        cluster_count=seen_clusters,
        expected_microbatches=expected_batches,
        expected_clusters=expected_count,
        unscaled_objective=float(objective),
        scaled_objective=float(objective * scale),
        loss_scale=scale,
        raw_gradient_norm=float(raw_norm),
        scaled_gradient_norm=float(scaled_norm),
        weight_sum=float(weight_sum),
        finite=finite,
    )


def _quantiles(value: Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.0, 0.5, 0.9, 0.99, 1.0], dtype=torch.float64
    )
    numbers = value.detach().abs().double().cpu().reshape(-1)
    return {
        name: float(result)
        for name, result in zip(
            ("q00", "q50", "q90", "q99", "q100"),
            torch.quantile(numbers, probabilities).tolist(),
        )
    }


def accumulation_diagnostics(
    stream: AccumulatedPairedMixtureStream,
) -> dict[str, Any]:
    """Summarize coupling, strata, and stateless accumulation health."""

    canonical = stream.canonical_microbatches
    reference = torch.cat([value.reference_states.detach().cpu() for value in canonical])
    component = torch.cat([value.component_states.detach().cpu() for value in canonical])
    difference = component - reference
    strata = np.concatenate([value.strata for value in canonical])
    component_indices = torch.cat(
        [value.component_indices.detach().cpu() for value in canonical]
    )
    swaps = torch.cat([value.swap_bits.detach().cpu() for value in canonical])
    common_gamma_error = 0.0
    if stream.task == "bounded_teacher":
        errors: list[float] = []
        for value in canonical:
            anchors = bounded_teacher_anchor_indices(int(value.grid_size))[
                value.component_indices.detach().cpu()
            ]
            gamma = (
                value.reference_states.detach().cpu()
                * value.base_gamma_sums.detach().cpu()[:, None]
            )
            gamma[torch.arange(value.clusters), anchors] += (
                value.tilt_increments.detach().cpu()
            )
            reconstructed = gamma / (
                value.base_gamma_sums.detach().cpu()
                + value.tilt_increments.detach().cpu()
            )[:, None]
            errors.append(
                float(
                    (
                        reconstructed - value.component_states.detach().cpu()
                    ).abs().max()
                )
            )
        common_gamma_error = max(errors, default=0.0)
    simplex_error = max(
        float((reference.sum(1) - 1.0).abs().max()),
        float((component.sum(1) - 1.0).abs().max()),
    )
    return {
        "schema": PAIRED_MIXTURE_SCHEMA + "-accumulation-diagnostics",
        "schema_version": PAIRED_MIXTURE_SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "finite": int(
            bool(torch.isfinite(reference).all())
            and bool(torch.isfinite(component).all())
        ),
        "task": stream.task,
        "optimizer_step": int(stream.optimizer_step),
        "accumulation_level": int(stream.accumulation_level),
        "microbatch_clusters": int(stream.microbatch_clusters),
        "effective_clusters": int(stream.effective_clusters),
        "effective_state_evaluations": 2 * int(stream.effective_clusters),
        "time_bin_counts": [
            int(np.count_nonzero(strata == index)) for index in range(5)
        ],
        "component_counts": [
            int(torch.count_nonzero(component_indices == index)) for index in range(4)
        ],
        "swap_counts": [
            int(torch.count_nonzero(swaps == index)) for index in range(2)
        ],
        "common_gamma_reconstruction_max_error": common_gamma_error,
        "simplex_max_error": simplex_error,
        "paired_state_l1_quantiles": _quantiles(difference.abs().sum(dim=1)),
        "paired_state_element_delta_quantiles": _quantiles(difference),
        "canonical_microbatch_fingerprints": [
            value.fingerprint for value in canonical
        ],
        "stream_fingerprint": stream.fingerprint,
        "order_invariant_fingerprint": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
