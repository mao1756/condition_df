"""Exact density-ratio controls for the boundary-admissible D0 score model.

This module is intentionally controls-only.  It provides a deterministic
binary classification problem whose Bayes logit is the bounded teacher's log
density ratio with respect to ``Dirichlet(1)``.  It contains no physical score
training orchestration and no sampler.

The positive and reference examples in every batch use the same reverse-time
anchors and equal class priors at each anchor.  Consequently the Bayes logit
is exactly ``log(p_tau / nu)`` without a time-dependent prior correction.  A
Dirichlet-null task draws both classes from ``nu`` and has the analytic zero
logit as its population optimum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .d0_dirichlet_score import (
    edge_difference_channels,
    harmonic_mobility_exact,
    physical_flux_from_edge_score,
)
from .d0_score_boundary_controls import (
    BOUNDED_TEACHER_VERSION,
    BOUNDARY_SMOOTH_MODEL_VERSION,
    bounded_teacher_edge_score,
    bounded_teacher_log_relative_potential,
    sample_bounded_teacher_mixture,
)
from .d0_score_optimizer_scale import (
    LossScaleCalibration,
    calibrate_initial_loss_scale,
)
from .eulerian_flux_mnist import DirectFluxMNISTConfig


DENSITY_RATIO_SCHEMA = "experiment12-d0-score-density-ratio"
DENSITY_RATIO_SCHEMA_VERSION = 1
DENSITY_RATIO_STREAM_VERSION = "d0-density-ratio-balanced-stream-v1"
DENSITY_RATIO_PANEL_VERSION = "d0-density-ratio-panel-v1"
DENSITY_RATIO_OBJECTIVE_VERSION = "d0-balanced-raw-logit-bce-v1"

FROZEN_CLASS_BIN_COUNTS = (4, 4, 4, 4, 16)
FROZEN_BATCH_BIN_COUNTS = (8, 8, 8, 8, 32)
SUPPORTED_DENSITY_RATIO_TASKS = ("bounded_teacher", "dirichlet_null")


__all__ = [
    "DENSITY_RATIO_SCHEMA",
    "DENSITY_RATIO_SCHEMA_VERSION",
    "DENSITY_RATIO_STREAM_VERSION",
    "DENSITY_RATIO_PANEL_VERSION",
    "DENSITY_RATIO_OBJECTIVE_VERSION",
    "FROZEN_CLASS_BIN_COUNTS",
    "FROZEN_BATCH_BIN_COUNTS",
    "SUPPORTED_DENSITY_RATIO_TASKS",
    "DensityRatioStreamPlan",
    "DensityRatioBatch",
    "DensityRatioPanel",
    "build_density_ratio_stream_plan",
    "stream_plan_record",
    "derive_density_ratio_seed",
    "generate_density_ratio_batch",
    "density_ratio_replay_record",
    "verify_density_ratio_replay",
    "stream_replay_record",
    "verify_stream_replay",
    "build_density_ratio_panel",
    "save_density_ratio_panel",
    "load_density_ratio_panel",
    "panel_identity",
    "verify_panel_identity",
    "panel_disjointness_record",
    "equal_prior_bayes_logit",
    "correct_logit_for_class_prior",
    "class_posterior_from_log_ratio",
    "classification_loss",
    "scaled_classification_loss",
    "evaluate_classification_risk",
    "evaluate_classification_panel",
    "analytic_teacher_metrics",
    "calibrate_density_ratio_loss_scale",
]


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_fingerprint(value: Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    array = tensor.numpy()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_positive(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


def _validate_task(task: str) -> str:
    value = str(task)
    if value not in SUPPORTED_DENSITY_RATIO_TASKS:
        raise ValueError(f"unsupported density-ratio task {value!r}")
    return value


@dataclass(frozen=True)
class DensityRatioStreamPlan:
    """Frozen derivation plan for balanced teacher/reference streams."""

    root_seed: int
    grid_size: int
    horizon: float
    label: int = 3
    bin_counts: tuple[int, ...] = FROZEN_CLASS_BIN_COUNTS
    teacher_epsilon: float = 0.5
    schema: str = DENSITY_RATIO_SCHEMA + "-stream-plan"
    schema_version: int = DENSITY_RATIO_SCHEMA_VERSION
    derivation_version: str = DENSITY_RATIO_STREAM_VERSION

    def __post_init__(self) -> None:
        if self.schema != DENSITY_RATIO_SCHEMA + "-stream-plan":
            raise ValueError("incompatible density-ratio stream schema")
        if int(self.schema_version) != DENSITY_RATIO_SCHEMA_VERSION:
            raise ValueError("incompatible density-ratio stream schema version")
        if self.derivation_version != DENSITY_RATIO_STREAM_VERSION:
            raise ValueError("incompatible density-ratio stream derivation")
        if int(self.grid_size) <= 0 or int(self.grid_size) % 4:
            raise ValueError("grid_size must be a positive multiple of four")
        if not _finite_positive(self.horizon):
            raise ValueError("horizon must be finite and positive")
        if int(self.label) < 0:
            raise ValueError("label must be nonnegative")
        if tuple(int(value) for value in self.bin_counts) != FROZEN_CLASS_BIN_COUNTS:
            raise ValueError("schema v1 requires class-bin counts (4,4,4,4,16)")
        if not math.isfinite(float(self.teacher_epsilon)) or not (
            0.0 < float(self.teacher_epsilon) < 1.0
        ):
            raise ValueError("teacher_epsilon must lie strictly between zero and one")

    @property
    def examples_per_class(self) -> int:
        return int(sum(int(value) for value in self.bin_counts))

    @property
    def batch_size(self) -> int:
        return 2 * self.examples_per_class

    @property
    def fingerprint(self) -> str:
        return _canonical_fingerprint(_stream_plan_payload(self))


def _stream_plan_payload(plan: DensityRatioStreamPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload.update(
        {
            "bin_counts": list(plan.bin_counts),
            "examples_per_class": int(plan.examples_per_class),
            "batch_size": int(plan.batch_size),
            "batch_bin_counts": list(FROZEN_BATCH_BIN_COUNTS),
            "teacher_version": BOUNDED_TEACHER_VERSION,
            "model_version": BOUNDARY_SMOOTH_MODEL_VERSION,
            "objective_version": DENSITY_RATIO_OBJECTIVE_VERSION,
        }
    )
    return payload


def build_density_ratio_stream_plan(
    *,
    root_seed: int,
    grid_size: int,
    horizon: float,
    label: int = 3,
    bin_counts: Sequence[int] = FROZEN_CLASS_BIN_COUNTS,
    teacher_epsilon: float = 0.5,
) -> DensityRatioStreamPlan:
    """Build the immutable version-one density-ratio stream plan."""

    return DensityRatioStreamPlan(
        root_seed=int(root_seed),
        grid_size=int(grid_size),
        horizon=float(horizon),
        label=int(label),
        bin_counts=tuple(int(value) for value in bin_counts),
        teacher_epsilon=float(teacher_epsilon),
    )


def stream_plan_record(plan: DensityRatioStreamPlan) -> dict[str, Any]:
    return {
        **_stream_plan_payload(plan),
        "fingerprint": plan.fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def derive_density_ratio_seed(
    plan_or_root_seed: DensityRatioStreamPlan | int,
    phase: str,
    task: str,
    step: int,
    namespace: str,
) -> int:
    """Derive a device-independent seed without consulting global RNG state."""

    root_seed = (
        int(plan_or_root_seed.root_seed)
        if isinstance(plan_or_root_seed, DensityRatioStreamPlan)
        else int(plan_or_root_seed)
    )
    phase_value = str(phase)
    task_value = _validate_task(task)
    namespace_value = str(namespace)
    if not phase_value or not namespace_value:
        raise ValueError("phase and namespace must be nonempty")
    if int(step) < 0:
        raise ValueError("step must be nonnegative")
    payload = json.dumps(
        [
            DENSITY_RATIO_STREAM_VERSION,
            root_seed,
            phase_value,
            task_value,
            int(step),
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
    for bin_index, count in enumerate(FROZEN_CLASS_BIN_COUNTS):
        for offset in range(int(count)):
            fractions.append(
                (float(bin_index) + (float(offset) + 0.5) / float(count)) / 5.0
            )
            strata.append(int(bin_index))
            anchors.append(anchor)
            anchor += 1
    return (
        torch.tensor(fractions, dtype=dtype),
        np.asarray(strata, dtype=np.int64),
        np.asarray(anchors, dtype=np.int64),
    )


def _sample_dirichlet_one(
    rows: int, pixels: int, *, seed: int, dtype: torch.dtype
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    concentration = torch.ones((int(rows), int(pixels)), dtype=dtype)
    draws = torch._standard_gamma(concentration, generator=generator)
    return draws / draws.sum(dim=1, keepdim=True)


@dataclass(frozen=True)
class DensityRatioBatch:
    """One replayable, equal-prior binary classification batch."""

    states: Tensor
    tau: Tensor
    tau_fraction: Tensor
    labels: Tensor
    class_targets: Tensor
    path_ids: np.ndarray
    strata: np.ndarray
    anchor_ids: np.ndarray
    phase: str
    task: str
    step: int
    seeds: Mapping[str, int]
    plan_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        rows = int(self.states.shape[0]) if self.states.ndim == 2 else -1
        if rows != 64:
            raise ValueError("density-ratio batches must have shape (64, pixels)")
        if any(
            value.shape != (rows,)
            for value in (self.tau, self.tau_fraction, self.labels, self.class_targets)
        ):
            raise ValueError("density-ratio batch tensor axes disagree")
        if any(
            np.asarray(value).shape != (rows,)
            for value in (self.path_ids, self.strata, self.anchor_ids)
        ):
            raise ValueError("density-ratio batch metadata axes disagree")
        _validate_task(self.task)
        if not self.phase or int(self.step) < 0:
            raise ValueError("phase must be nonempty and step must be nonnegative")
        if not bool(torch.isfinite(self.states).all() and (self.states > 0.0).all()):
            raise ValueError("density-ratio states must be finite and strictly positive")
        tolerance = 2e-12 if self.states.dtype == torch.float64 else 2e-6
        if float((self.states.sum(1) - 1.0).abs().max().detach().cpu()) > tolerance:
            raise ValueError("density-ratio states are not simplex-valued")
        targets = self.class_targets.detach().cpu()
        if not bool(torch.all((targets == 0.0) | (targets == 1.0))):
            raise ValueError("class targets must be binary")
        if int(targets.sum()) != rows // 2:
            raise ValueError("density-ratio batches must have equal class priors")
        strata = np.asarray(self.strata, dtype=np.int64)
        targets_np = targets.numpy().astype(np.int64, copy=False)
        for class_value in (0, 1):
            counts = tuple(
                int(((strata == index) & (targets_np == class_value)).sum())
                for index in range(5)
            )
            if counts != FROZEN_CLASS_BIN_COUNTS:
                raise ValueError("class-conditional time strata are not frozen/balanced")
        fractions = self.tau_fraction.detach().cpu().numpy()
        anchors = np.asarray(self.anchor_ids, dtype=np.int64)
        for anchor in range(32):
            mask = anchors == anchor
            if int(mask.sum()) != 2 or set(targets_np[mask].tolist()) != {0, 1}:
                raise ValueError("each time anchor must contain exactly one example per class")
            if not np.all(fractions[mask] == fractions[mask][0]):
                raise ValueError("paired class examples do not share an exact time anchor")

    def record(self) -> dict[str, Any]:
        targets = self.class_targets.detach().cpu().numpy().astype(np.int64)
        return {
            "schema": DENSITY_RATIO_SCHEMA + "-batch",
            "schema_version": DENSITY_RATIO_SCHEMA_VERSION,
            "derivation_version": DENSITY_RATIO_STREAM_VERSION,
            "phase": self.phase,
            "task": self.task,
            "step": int(self.step),
            "rows": int(self.states.shape[0]),
            "pixels": int(self.states.shape[1]),
            "class_counts": [int((targets == value).sum()) for value in (0, 1)],
            "class_bin_counts": {
                str(value): [
                    int(((targets == value) & (self.strata == index)).sum())
                    for index in range(5)
                ]
                for value in (0, 1)
            },
            "state_sha256": _tensor_fingerprint(self.states),
            "tau_fraction_sha256": _tensor_fingerprint(self.tau_fraction),
            "class_targets_sha256": _tensor_fingerprint(self.class_targets),
            "path_ids_sha256": _array_fingerprint(self.path_ids),
            "strata_sha256": _array_fingerprint(self.strata),
            "anchor_ids_sha256": _array_fingerprint(self.anchor_ids),
            "seeds": {str(key): int(value) for key, value in self.seeds.items()},
            "plan_fingerprint": self.plan_fingerprint,
            "fingerprint": self.fingerprint,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }


def _batch_fingerprint_payload(
    *,
    states: Tensor,
    fractions: Tensor,
    class_targets: Tensor,
    path_ids: np.ndarray,
    strata: np.ndarray,
    anchor_ids: np.ndarray,
    plan: DensityRatioStreamPlan,
    phase: str,
    task: str,
    step: int,
    seeds: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema": DENSITY_RATIO_SCHEMA + "-batch",
        "schema_version": DENSITY_RATIO_SCHEMA_VERSION,
        "derivation_version": DENSITY_RATIO_STREAM_VERSION,
        "phase": str(phase),
        "task": str(task),
        "step": int(step),
        "state_sha256": _tensor_fingerprint(states),
        "tau_fraction_sha256": _tensor_fingerprint(fractions),
        "class_targets_sha256": _tensor_fingerprint(class_targets),
        "path_ids_sha256": _array_fingerprint(path_ids),
        "strata_sha256": _array_fingerprint(strata),
        "anchor_ids_sha256": _array_fingerprint(anchor_ids),
        "seeds": {str(key): int(value) for key, value in seeds.items()},
        "plan_fingerprint": plan.fingerprint,
    }


def generate_density_ratio_batch(
    plan: DensityRatioStreamPlan,
    *,
    phase: str,
    task: str,
    step: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DensityRatioBatch:
    """Generate one fresh, stateless, time-matched binary batch.

    Rows are generated on CPU and transferred only after the scientific
    fingerprint is fixed, making CPU and CUDA runs share the same byte stream.
    For the null, all 64 states come from one Dirichlet sampler namespace and
    one state from every identical-time pair is assigned to each class by a
    stateless swap.  Thus labels cannot reveal a class-specific sampler.
    """

    if dtype not in (torch.float32, torch.float64):
        raise ValueError("density-ratio stream dtype must be float32 or float64")
    phase_value = str(phase)
    task_value = _validate_task(task)
    step_value = int(step)
    if not phase_value or step_value < 0:
        raise ValueError("phase must be nonempty and step must be nonnegative")

    fractions_one, strata_one, anchors_one = _time_template(dtype)
    rows_per_class = int(plan.examples_per_class)
    pixels = int(plan.grid_size) ** 2
    seeds = {
        namespace: derive_density_ratio_seed(
            plan, phase_value, task_value, step_value, namespace
        )
        for namespace in (
            "positive-states",
            "reference-states",
            "null-pool",
            "null-swaps",
            "row-permutation",
            "path-id",
        )
    }

    if task_value == "bounded_teacher":
        positive = sample_bounded_teacher_mixture(
            fractions_one,
            int(plan.grid_size),
            seed=int(seeds["positive-states"]),
            device="cpu",
            dtype=dtype,
            epsilon=float(plan.teacher_epsilon),
        )
        reference = _sample_dirichlet_one(
            rows_per_class,
            pixels,
            seed=int(seeds["reference-states"]),
            dtype=dtype,
        )
    else:
        pooled = _sample_dirichlet_one(
            2 * rows_per_class,
            pixels,
            seed=int(seeds["null-pool"]),
            dtype=dtype,
        ).reshape(rows_per_class, 2, pixels)
        swap_generator = torch.Generator(device="cpu").manual_seed(
            int(seeds["null-swaps"])
        )
        swaps = torch.randint(
            0, 2, (rows_per_class,), generator=swap_generator, dtype=torch.long
        )
        row_index = torch.arange(rows_per_class)
        positive = pooled[row_index, swaps]
        reference = pooled[row_index, 1 - swaps]

    states_cpu = torch.cat([positive, reference], dim=0).contiguous()
    fractions_cpu = fractions_one.repeat(2).contiguous()
    targets_cpu = torch.cat(
        [
            torch.ones(rows_per_class, dtype=dtype),
            torch.zeros(rows_per_class, dtype=dtype),
        ]
    )
    strata = np.tile(strata_one, 2)
    anchor_ids = np.tile(anchors_one, 2)
    path_id = int(seeds["path-id"])
    path_ids = np.full((2 * rows_per_class,), path_id, dtype=np.int64)

    permutation_generator = torch.Generator(device="cpu").manual_seed(
        int(seeds["row-permutation"])
    )
    permutation = torch.randperm(2 * rows_per_class, generator=permutation_generator)
    permutation_np = permutation.numpy()
    states_cpu = states_cpu.index_select(0, permutation)
    fractions_cpu = fractions_cpu.index_select(0, permutation)
    targets_cpu = targets_cpu.index_select(0, permutation)
    path_ids = path_ids[permutation_np]
    strata = strata[permutation_np]
    anchor_ids = anchor_ids[permutation_np]

    payload = _batch_fingerprint_payload(
        states=states_cpu,
        fractions=fractions_cpu,
        class_targets=targets_cpu,
        path_ids=path_ids,
        strata=strata,
        anchor_ids=anchor_ids,
        plan=plan,
        phase=phase_value,
        task=task_value,
        step=step_value,
        seeds=seeds,
    )
    target_device = torch.device(device)
    states = states_cpu.to(target_device)
    fractions = fractions_cpu.to(target_device)
    return DensityRatioBatch(
        states=states,
        tau=(fractions * float(plan.horizon)).contiguous(),
        tau_fraction=fractions.contiguous(),
        labels=torch.full(
            (int(plan.batch_size),),
            int(plan.label),
            device=target_device,
            dtype=torch.long,
        ),
        class_targets=targets_cpu.to(target_device),
        path_ids=path_ids,
        strata=strata,
        anchor_ids=anchor_ids,
        phase=phase_value,
        task=task_value,
        step=step_value,
        seeds=seeds,
        plan_fingerprint=plan.fingerprint,
        fingerprint=_canonical_fingerprint(payload),
    )


def density_ratio_replay_record(
    plan: DensityRatioStreamPlan,
    *,
    phase: str,
    task: str,
    step: int,
) -> dict[str, Any]:
    """Materialize the canonical CPU-float32 replay certificate."""

    batch = generate_density_ratio_batch(
        plan,
        phase=str(phase),
        task=str(task),
        step=int(step),
        device="cpu",
        dtype=torch.float32,
    )
    record = {
        "schema": DENSITY_RATIO_SCHEMA + "-replay",
        "schema_version": DENSITY_RATIO_SCHEMA_VERSION,
        "derivation_version": DENSITY_RATIO_STREAM_VERSION,
        "plan_fingerprint": plan.fingerprint,
        "phase": str(phase),
        "task": str(task),
        "step": int(step),
        "batch": batch.record(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    record["fingerprint"] = _canonical_fingerprint(record)
    return record


def verify_density_ratio_replay(
    plan: DensityRatioStreamPlan, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Regenerate a replay record and compare it fail-closed."""

    reason: str | None = None
    actual: str | None = None
    try:
        expected = density_ratio_replay_record(
            plan,
            phase=str(record["phase"]),
            task=str(record["task"]),
            step=int(record["step"]),
        )
        actual = _canonical_fingerprint(
            {key: value for key, value in dict(record).items() if key != "fingerprint"}
        )
        expected_fingerprint = str(expected["fingerprint"])
        recorded_fingerprint = str(record.get("fingerprint", ""))
        passed = int(
            actual == expected_fingerprint
            and recorded_fingerprint == expected_fingerprint
            and str(record.get("plan_fingerprint", "")) == plan.fingerprint
        )
        if not passed:
            reason = "replayed density-ratio stream fingerprint differs"
    except Exception as error:  # fail-closed artifact verifier
        expected_fingerprint = None
        passed = 0
        reason = f"{type(error).__name__}: {error}"
    return {
        "passed": int(passed),
        "reason": reason,
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual,
        "plan_fingerprint": plan.fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


# Short aliases are convenient for a dedicated CLI importing this module only.
stream_replay_record = density_ratio_replay_record
verify_stream_replay = verify_density_ratio_replay


@dataclass(frozen=True)
class DensityRatioPanel:
    """A fixed collection of independent paired clusters for selection/audit."""

    states: Tensor
    tau: Tensor
    tau_fraction: Tensor
    labels: Tensor
    class_targets: Tensor
    path_ids: np.ndarray
    strata: np.ndarray
    anchor_ids: np.ndarray
    phase: str
    role: str
    task: str
    path_count: int
    plan_fingerprint: str
    stream_fingerprints: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        rows = int(self.states.shape[0]) if self.states.ndim == 2 else -1
        expected_rows = 64 * int(self.path_count)
        if int(self.path_count) <= 0 or rows != expected_rows:
            raise ValueError("panel rows must equal 64 times its positive path count")
        if any(
            value.shape != (rows,)
            for value in (self.tau, self.tau_fraction, self.labels, self.class_targets)
        ):
            raise ValueError("panel tensor axes disagree")
        if any(
            np.asarray(value).shape != (rows,)
            for value in (self.path_ids, self.strata, self.anchor_ids)
        ):
            raise ValueError("panel metadata axes disagree")
        if len(self.stream_fingerprints) != int(self.path_count):
            raise ValueError("panel stream fingerprint count disagrees with paths")
        if not self.phase or not self.role:
            raise ValueError("panel phase and role must be nonempty")
        _validate_task(self.task)
        if len(set(np.asarray(self.path_ids, dtype=np.int64).tolist())) != int(
            self.path_count
        ):
            raise ValueError("panel path IDs are not whole-cluster isolated")
        targets = self.class_targets.detach().cpu().numpy().astype(np.int64)
        strata = np.asarray(self.strata, dtype=np.int64)
        for class_value in (0, 1):
            counts = tuple(
                int(((targets == class_value) & (strata == index)).sum())
                for index in range(5)
            )
            expected = tuple(int(self.path_count) * value for value in FROZEN_CLASS_BIN_COUNTS)
            if counts != expected:
                raise ValueError("panel class/time balance is invalid")
        if self.fingerprint != _panel_fingerprint(self):
            raise ValueError("panel fingerprint does not match its contents")

    def identity(self) -> dict[str, Any]:
        return panel_identity(self)


def _panel_fingerprint_payload(panel: DensityRatioPanel) -> dict[str, Any]:
    return {
        "schema": DENSITY_RATIO_SCHEMA + "-panel",
        "schema_version": DENSITY_RATIO_SCHEMA_VERSION,
        "panel_version": DENSITY_RATIO_PANEL_VERSION,
        "phase": panel.phase,
        "role": panel.role,
        "task": panel.task,
        "path_count": int(panel.path_count),
        "rows": int(panel.states.shape[0]),
        "pixels": int(panel.states.shape[1]),
        "state_sha256": _tensor_fingerprint(panel.states),
        "tau_fraction_sha256": _tensor_fingerprint(panel.tau_fraction),
        "labels_sha256": _tensor_fingerprint(panel.labels),
        "class_targets_sha256": _tensor_fingerprint(panel.class_targets),
        "path_ids_sha256": _array_fingerprint(panel.path_ids),
        "strata_sha256": _array_fingerprint(panel.strata),
        "anchor_ids_sha256": _array_fingerprint(panel.anchor_ids),
        "stream_fingerprints": list(panel.stream_fingerprints),
        "plan_fingerprint": panel.plan_fingerprint,
    }


def _panel_fingerprint(panel: DensityRatioPanel) -> str:
    return _canonical_fingerprint(_panel_fingerprint_payload(panel))


def build_density_ratio_panel(
    plan: DensityRatioStreamPlan,
    *,
    phase: str,
    role: str,
    task: str,
    path_count: int,
    start_step: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DensityRatioPanel:
    """Build a fixed stateless panel in a role-specific seed namespace."""

    count = int(path_count)
    start = int(start_step)
    phase_value, role_value = str(phase), str(role)
    if count <= 0 or start < 0:
        raise ValueError("path_count must be positive and start_step nonnegative")
    if not phase_value or not role_value:
        raise ValueError("phase and role must be nonempty")
    task_value = _validate_task(task)
    stream_phase = f"panel:{phase_value}:{role_value}"
    batches = [
        generate_density_ratio_batch(
            plan,
            phase=stream_phase,
            task=task_value,
            step=start + offset,
            device="cpu",
            dtype=dtype,
        )
        for offset in range(count)
    ]
    target = torch.device(device)
    provisional = DensityRatioPanel.__new__(DensityRatioPanel)
    values = {
        "states": torch.cat([batch.states for batch in batches]).to(target),
        "tau": torch.cat([batch.tau for batch in batches]).to(target),
        "tau_fraction": torch.cat([batch.tau_fraction for batch in batches]).to(target),
        "labels": torch.cat([batch.labels for batch in batches]).to(target),
        "class_targets": torch.cat([batch.class_targets for batch in batches]).to(target),
        "path_ids": np.concatenate([batch.path_ids for batch in batches]),
        "strata": np.concatenate([batch.strata for batch in batches]),
        # Anchor IDs need only identify the 32 positions within each path.
        "anchor_ids": np.concatenate([batch.anchor_ids for batch in batches]),
        "phase": phase_value,
        "role": role_value,
        "task": task_value,
        "path_count": count,
        "plan_fingerprint": plan.fingerprint,
        "stream_fingerprints": tuple(batch.fingerprint for batch in batches),
    }
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    object.__setattr__(provisional, "fingerprint", _panel_fingerprint(provisional))
    # Construct normally so every invariant is checked.
    return DensityRatioPanel(
        **values,
        fingerprint=provisional.fingerprint,
    )


def panel_identity(panel: DensityRatioPanel) -> dict[str, Any]:
    """Return the complete, JSON-safe identity of a fixed panel."""

    targets = panel.class_targets.detach().cpu().numpy().astype(np.int64)
    record = _panel_fingerprint_payload(panel)
    record.update(
        {
            "fingerprint": panel.fingerprint,
            "class_counts": [int((targets == value).sum()) for value in (0, 1)],
            "unique_path_count": int(np.unique(panel.path_ids).size),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    )
    return record


def verify_panel_identity(
    panel: DensityRatioPanel, expected: Mapping[str, Any]
) -> dict[str, Any]:
    actual = panel_identity(panel)
    expected_fingerprint = str(expected.get("fingerprint", ""))
    passed = int(
        expected_fingerprint == panel.fingerprint
        and str(expected.get("plan_fingerprint", "")) == panel.plan_fingerprint
        and int(expected.get("rows", -1)) == int(panel.states.shape[0])
        and int(expected.get("path_count", -1)) == int(panel.path_count)
    )
    return {
        "passed": passed,
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual["fingerprint"],
        "reason": None if passed else "panel identity differs",
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def save_density_ratio_panel(
    path: str | Path,
    panel: DensityRatioPanel,
    binding: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a panel as a tensors-and-primitives checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": DENSITY_RATIO_SCHEMA + "-panel-checkpoint",
        "schema_version": DENSITY_RATIO_SCHEMA_VERSION,
        "panel_version": DENSITY_RATIO_PANEL_VERSION,
        "states": panel.states.detach().cpu(),
        "tau": panel.tau.detach().cpu(),
        "tau_fraction": panel.tau_fraction.detach().cpu(),
        "labels": panel.labels.detach().cpu(),
        "class_targets": panel.class_targets.detach().cpu(),
        "path_ids": torch.from_numpy(np.asarray(panel.path_ids, dtype=np.int64)),
        "strata": torch.from_numpy(np.asarray(panel.strata, dtype=np.int64)),
        "anchor_ids": torch.from_numpy(np.asarray(panel.anchor_ids, dtype=np.int64)),
        "phase": panel.phase,
        "role": panel.role,
        "task": panel.task,
        "path_count": int(panel.path_count),
        "plan_fingerprint": panel.plan_fingerprint,
        "stream_fingerprints": list(panel.stream_fingerprints),
        "fingerprint": panel.fingerprint,
        "identity": panel_identity(panel),
        "binding": {} if binding is None else dict(binding),
        "binding_fingerprint": _canonical_fingerprint(
            {} if binding is None else dict(binding)
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_density_ratio_panel(
    path: str | Path,
    binding: Mapping[str, Any] | None = None,
    *,
    device: torch.device | str = "cpu",
    expected_plan_fingerprint: str | None = None,
    expected_role: str | None = None,
    expected_task: str | None = None,
) -> DensityRatioPanel:
    """Load and hash-verify a density-ratio panel."""

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("density-ratio panel payload must be a mapping")
    if payload.get("schema") != DENSITY_RATIO_SCHEMA + "-panel-checkpoint":
        raise ValueError("incompatible density-ratio panel checkpoint schema")
    if int(payload.get("schema_version", -1)) != DENSITY_RATIO_SCHEMA_VERSION:
        raise ValueError("incompatible density-ratio panel checkpoint version")
    if payload.get("panel_version") != DENSITY_RATIO_PANEL_VERSION:
        raise ValueError("incompatible density-ratio panel derivation")
    if binding is not None:
        expected_binding = dict(binding)
        if dict(payload.get("binding", {})) != expected_binding or str(
            payload.get("binding_fingerprint", "")
        ) != _canonical_fingerprint(expected_binding):
            raise ValueError("density-ratio panel binding differs")
    plan_fingerprint = str(payload["plan_fingerprint"])
    role, task = str(payload["role"]), str(payload["task"])
    if expected_plan_fingerprint is not None and plan_fingerprint != str(
        expected_plan_fingerprint
    ):
        raise ValueError("density-ratio panel plan fingerprint differs")
    if expected_role is not None and role != str(expected_role):
        raise ValueError("density-ratio panel role differs")
    if expected_task is not None and task != str(expected_task):
        raise ValueError("density-ratio panel task differs")
    target = torch.device(device)
    panel = DensityRatioPanel(
        states=payload["states"].to(target),
        tau=payload["tau"].to(target),
        tau_fraction=payload["tau_fraction"].to(target),
        labels=payload["labels"].to(target),
        class_targets=payload["class_targets"].to(target),
        path_ids=payload["path_ids"].cpu().numpy().astype(np.int64, copy=False),
        strata=payload["strata"].cpu().numpy().astype(np.int64, copy=False),
        anchor_ids=payload["anchor_ids"].cpu().numpy().astype(np.int64, copy=False),
        phase=str(payload["phase"]),
        role=role,
        task=task,
        path_count=int(payload["path_count"]),
        plan_fingerprint=plan_fingerprint,
        stream_fingerprints=tuple(str(value) for value in payload["stream_fingerprints"]),
        fingerprint=str(payload["fingerprint"]),
    )
    verification = verify_panel_identity(panel, dict(payload["identity"]))
    if not verification["passed"]:
        raise ValueError(str(verification["reason"]))
    return panel


def panel_disjointness_record(
    panels: Sequence[DensityRatioPanel],
) -> dict[str, Any]:
    """Check path, stream, and scientific-state disjointness across panels."""

    values = list(panels)
    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(values):
        for right_index in range(left_index + 1, len(values)):
            right = values[right_index]
            path_overlap = sorted(
                set(np.asarray(left.path_ids, dtype=np.int64).tolist())
                & set(np.asarray(right.path_ids, dtype=np.int64).tolist())
            )
            stream_overlap = sorted(
                set(left.stream_fingerprints) & set(right.stream_fingerprints)
            )
            state_equal = int(
                _tensor_fingerprint(left.states) == _tensor_fingerprint(right.states)
            )
            if path_overlap or stream_overlap or state_equal:
                overlaps.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "left_role": left.role,
                        "right_role": right.role,
                        "path_overlap": path_overlap,
                        "stream_fingerprint_overlap": stream_overlap,
                        "identical_state_hash": state_equal,
                    }
                )
    return {
        "schema": DENSITY_RATIO_SCHEMA + "-panel-disjointness",
        "schema_version": DENSITY_RATIO_SCHEMA_VERSION,
        "panel_count": len(values),
        "passed": int(not overlaps),
        "overlaps": overlaps,
        "panel_fingerprints": [value.fingerprint for value in values],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def equal_prior_bayes_logit(
    states: Tensor,
    tau_fraction: Tensor | float,
    *,
    task: str = "bounded_teacher",
    epsilon: float = 0.5,
) -> Tensor:
    """Return the exact equal-prior Bayes logit for a synthetic task."""

    task_value = _validate_task(task)
    if task_value == "dirichlet_null":
        if states.ndim != 2:
            raise ValueError("states must have shape (B, pixels)")
        return states.new_zeros((states.shape[0],))
    return bounded_teacher_log_relative_potential(
        states, tau_fraction, epsilon=float(epsilon)
    )


def _validate_prior(positive_prior: float) -> float:
    prior = float(positive_prior)
    if not math.isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("positive_prior must lie strictly between zero and one")
    return prior


def correct_logit_for_class_prior(
    raw_logit: Tensor, *, positive_prior: float
) -> Tensor:
    """Remove known case-control log odds from a raw classifier logit."""

    prior = _validate_prior(positive_prior)
    log_odds = math.log(prior) - math.log1p(-prior)
    return raw_logit - raw_logit.new_tensor(log_odds)


def class_posterior_from_log_ratio(
    log_ratio: Tensor, *, positive_prior: float = 0.5
) -> Tensor:
    """Convert a log density ratio to the corresponding class posterior."""

    prior = _validate_prior(positive_prior)
    log_odds = math.log(prior) - math.log1p(-prior)
    return torch.sigmoid(log_ratio + log_ratio.new_tensor(log_odds))


def _binary_vectors(raw_logits: Tensor, class_targets: Tensor) -> tuple[Tensor, Tensor]:
    logits = raw_logits.reshape(-1)
    targets = class_targets.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if logits.shape != targets.shape or logits.numel() <= 0:
        raise ValueError("raw_logits and class_targets must be nonempty matching vectors")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(targets).all()):
        raise FloatingPointError("classification logits/targets must be finite")
    if not bool(torch.all((targets == 0.0) | (targets == 1.0))):
        raise ValueError("class_targets must be binary")
    return logits, targets


def classification_loss(
    raw_logits: Tensor,
    class_targets: Tensor,
    *,
    reduction: str = "mean",
    require_balanced: bool = True,
) -> Tensor:
    """Evaluate binary cross entropy directly from raw logits.

    The function intentionally accepts no probabilities and applies no class
    weights or label smoothing.  With balanced targets its population optimum
    is the unshifted log density ratio.
    """

    logits, targets = _binary_vectors(raw_logits, class_targets)
    reduction_value = str(reduction)
    if reduction_value not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if require_balanced and int(targets.sum().detach().cpu()) * 2 != targets.numel():
        raise ValueError("equal-prior BCE requires exactly balanced class targets")
    return F.binary_cross_entropy_with_logits(
        logits, targets, reduction=reduction_value
    )


def scaled_classification_loss(
    raw_logits: Tensor,
    class_targets: Tensor,
    *,
    loss_scale: float,
) -> tuple[Tensor, Tensor]:
    """Return unscaled scientific BCE and the scaled optimizer objective."""

    scale = float(loss_scale)
    if not _finite_positive(scale) or scale > 1.0:
        raise ValueError("loss_scale must be finite in (0, 1]")
    unscaled = classification_loss(raw_logits, class_targets)
    return unscaled, unscaled * scale


def _risk_scope(raw_logits: Tensor, class_targets: Tensor) -> dict[str, Any]:
    logits, targets = _binary_vectors(raw_logits, class_targets)
    if int(targets.sum().detach().cpu()) * 2 != targets.numel():
        raise ValueError("classification risk scopes must have equal class priors")
    losses = classification_loss(logits, targets, reduction="none")
    baseline = math.log(2.0)
    probabilities = torch.sigmoid(logits)
    positive = targets == 1.0
    negative = ~positive
    risk = float(losses.mean().detach().cpu())
    return {
        "state_count": int(logits.numel()),
        "positive_count": int(positive.sum().detach().cpu()),
        "negative_count": int(negative.sum().detach().cpu()),
        "risk": risk,
        "zero_logit_risk": baseline,
        "objective_improvement": baseline - risk,
        "positive_risk": float(losses[positive].mean().detach().cpu()),
        "negative_risk": float(losses[negative].mean().detach().cpu()),
        "accuracy": float(((logits >= 0.0) == positive).float().mean().detach().cpu()),
        "brier": float((probabilities - targets).square().mean().detach().cpu()),
        "mean_logit": float(logits.mean().detach().cpu()),
        "logit_rms": float(logits.square().mean().sqrt().detach().cpu()),
        "finite_fraction": float(torch.isfinite(losses).float().mean().detach().cpu()),
    }


def evaluate_classification_risk(
    raw_logits: Tensor,
    class_targets: Tensor,
    *,
    strata: np.ndarray | Tensor | None = None,
    path_ids: np.ndarray | Tensor | None = None,
) -> dict[str, Any]:
    """Summarize balanced BCE overall, by time bin, and by paired path."""

    logits, targets = _binary_vectors(raw_logits, class_targets)
    rows = int(logits.numel())
    overall = _risk_scope(logits, targets)
    result: dict[str, Any] = {
        "overall": overall,
        "risk": overall["risk"],
        "zero_logit_risk": overall["zero_logit_risk"],
        "objective_improvement": overall["objective_improvement"],
        "finite": int(overall["finite_fraction"] == 1.0),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    if strata is not None:
        strata_np = (
            strata.detach().cpu().numpy()
            if isinstance(strata, Tensor)
            else np.asarray(strata)
        ).astype(np.int64, copy=False)
        if strata_np.shape != (rows,):
            raise ValueError("strata must match logits")
        bins: list[dict[str, Any]] = []
        for index in range(5):
            ids = torch.as_tensor(
                np.flatnonzero(strata_np == index), device=logits.device, dtype=torch.long
            )
            bins.append(
                {
                    "time_bin": index,
                    **_risk_scope(logits.index_select(0, ids), targets.index_select(0, ids)),
                }
            )
        result["time_bins"] = bins
        result["data_end"] = bins[4]
    if path_ids is not None:
        paths_np = (
            path_ids.detach().cpu().numpy()
            if isinstance(path_ids, Tensor)
            else np.asarray(path_ids)
        ).astype(np.int64, copy=False)
        if paths_np.shape != (rows,):
            raise ValueError("path_ids must match logits")
        losses = classification_loss(logits, targets, reduction="none").detach().cpu().numpy()
        baseline = math.log(2.0)
        path_rows = []
        for path_id in sorted(np.unique(paths_np).tolist()):
            mask = paths_np == path_id
            path_rows.append(
                {
                    "path_id": int(path_id),
                    "state_count": int(mask.sum()),
                    "objective_improvement_vs_zero": float(
                        np.mean(baseline - losses[mask])
                    ),
                }
            )
        result["per_path"] = path_rows
    return result


def _model_logits(
    model: nn.Module,
    panel: DensityRatioPanel,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    values: list[Tensor] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, int(panel.states.shape[0]), int(batch_size)):
                end = min(start + int(batch_size), int(panel.states.shape[0]))
                logits = model(
                    panel.tau[start:end].to(device),
                    panel.states[start:end].to(device),
                    panel.labels[start:end].to(device),
                ).reshape(-1)
                if logits.shape != (end - start,):
                    raise ValueError("density-ratio model must return one raw logit per state")
                values.append(logits.detach().cpu())
    finally:
        model.train(was_training)
    return torch.cat(values)


def evaluate_classification_panel(
    model: nn.Module,
    panel: DensityRatioPanel,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 128,
    return_logits: bool = False,
) -> dict[str, Any]:
    """Evaluate deterministic raw-logit risk on one fixed panel."""

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    logits = _model_logits(
        model, panel, device=torch.device(device), batch_size=int(batch_size)
    )
    metrics = evaluate_classification_risk(
        logits,
        panel.class_targets.detach().cpu(),
        strata=panel.strata,
        path_ids=panel.path_ids,
    )
    metrics.update(
        {
            "panel_fingerprint": panel.fingerprint,
            "task": panel.task,
            "role": panel.role,
            "evaluation_status": "evaluated",
        }
    )
    if return_logits:
        metrics["raw_logits"] = logits
    return metrics


def _scope_analytic_metrics(
    *,
    states: Tensor,
    predicted_logits: Tensor,
    target_logits: Tensor,
    predicted_cell_gradient: Tensor,
    predicted_edge_score: Tensor,
    target_edge_score: Tensor,
    theta: Tensor,
    strata: Tensor,
    config: DirectFluxMNISTConfig,
    ids: Tensor,
) -> dict[str, Any]:
    states_scope = states.index_select(0, ids)
    predicted_scope = predicted_edge_score.index_select(0, ids)
    target_scope = target_edge_score.index_select(0, ids)
    weights = theta.index_select(0, ids)
    model_mse = float(
        (weights * (predicted_scope - target_scope).square()).mean().detach().cpu()
    )
    zero_mse = float((weights * target_scope.square()).mean().detach().cpu())
    target_flux = physical_flux_from_edge_score(target_scope, states_scope, config)
    predicted_flux = physical_flux_from_edge_score(predicted_scope, states_scope, config)
    flat_target = target_flux.reshape(-1).double()
    flat_prediction = predicted_flux.reshape(-1).double()
    target_norm = torch.linalg.vector_norm(flat_target)
    prediction_norm = torch.linalg.vector_norm(flat_prediction)
    cosine = float(
        (
            (flat_target @ flat_prediction)
            / (target_norm * prediction_norm).clamp_min(1e-30)
        )
        .detach()
        .cpu()
    )
    relative = float(
        (
            torch.linalg.vector_norm(flat_prediction - flat_target)
            / target_norm.clamp_min(1e-30)
        )
        .detach()
        .cpu()
    )
    predicted_logit_scope = predicted_logits.index_select(0, ids)
    target_logit_scope = target_logits.index_select(0, ids)
    predicted_gradient_scope = predicted_cell_gradient.index_select(0, ids)
    strata_scope = strata.index_select(0, ids)
    logit_mse = float(
        (predicted_logit_scope - target_logit_scope).square().mean().detach().cpu()
    )
    zero_logit_mse = float(target_logit_scope.square().mean().detach().cpu())
    centered_prediction = predicted_logit_scope - predicted_logit_scope.mean()
    centered_target = target_logit_scope - target_logit_scope.mean()
    centered_mse = float(
        (centered_prediction - centered_target).square().mean().detach().cpu()
    )

    def correlation(first: Tensor, second: Tensor) -> float:
        first_flat = first.reshape(-1).double()
        second_flat = second.reshape(-1).double()
        first_centered = first_flat - first_flat.mean()
        second_centered = second_flat - second_flat.mean()
        denominator = torch.linalg.vector_norm(first_centered) * torch.linalg.vector_norm(
            second_centered
        )
        if float(denominator.detach().cpu()) <= 1e-30:
            return 0.0
        return float(
            ((first_centered @ second_centered) / denominator).detach().cpu()
        )

    time_centered_prediction = torch.empty_like(predicted_logit_scope)
    time_centered_target = torch.empty_like(target_logit_scope)
    for bin_index in torch.unique(strata_scope, sorted=True).tolist():
        bin_mask = strata_scope == int(bin_index)
        predicted_bin = predicted_logit_scope[bin_mask]
        target_bin = target_logit_scope[bin_mask]
        time_centered_prediction[bin_mask] = predicted_bin - predicted_bin.mean()
        time_centered_target[bin_mask] = target_bin - target_bin.mean()
    time_bin_centered_mse = float(
        (time_centered_prediction - time_centered_target)
        .square()
        .mean()
        .detach()
        .cpu()
    )

    def absolute_quantiles(values: Tensor) -> dict[str, float]:
        probabilities = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        names = ("q00", "q10", "q50", "q90", "q99", "q100")
        flat = values.detach().abs().reshape(-1).double()
        if flat.numel() <= 0 or not bool(torch.isfinite(flat).all()):
            raise FloatingPointError("analytic field quantiles require finite values")
        quantile_values = torch.quantile(
            flat,
            torch.tensor(probabilities, device=flat.device, dtype=flat.dtype),
        )
        return {
            name: float(value)
            for name, value in zip(names, quantile_values.detach().cpu().tolist())
        }

    return {
        "state_count": int(ids.numel()),
        "target_mse": zero_mse,
        "model_mse": model_mse,
        "score_gain": 1.0 - model_mse / max(zero_mse, 1e-30),
        "flux_cosine": cosine,
        "flux_relative_l2": relative,
        "logit_mse": logit_mse,
        "zero_logit_mse": zero_logit_mse,
        "logit_gain": 1.0 - logit_mse / max(zero_logit_mse, 1e-30),
        "centered_logit_mse": centered_mse,
        "raw_logit_correlation": correlation(
            predicted_logit_scope, target_logit_scope
        ),
        "time_bin_centered_logit_mse": time_bin_centered_mse,
        "time_bin_centered_logit_correlation": correlation(
            time_centered_prediction, time_centered_target
        ),
        "predicted_cell_gradient_abs_quantiles": absolute_quantiles(
            predicted_gradient_scope
        ),
        "predicted_edge_score_abs_quantiles": absolute_quantiles(predicted_scope),
        "predicted_physical_flux_abs_quantiles": absolute_quantiles(
            predicted_flux
        ),
        "mean_logit_offset": float(
            (predicted_logit_scope - target_logit_scope).mean().detach().cpu()
        ),
    }


def analytic_teacher_metrics(
    model: nn.Module,
    panel: DensityRatioPanel,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    evaluate_class_target: int = 1,
    epsilon: float = 0.5,
) -> dict[str, Any]:
    """Evaluate analytic bounded-teacher logit, edge-score, and flux metrics.

    By default metrics are evaluated on the positive teacher states, matching
    the established boundary-control gates.  Passing ``evaluate_class_target=0``
    gives an advisory reference-law evaluation of the same analytic field.
    """

    if panel.task != "bounded_teacher":
        raise ValueError("analytic teacher metrics require a bounded-teacher panel")
    class_value = int(evaluate_class_target)
    if class_value not in (0, 1):
        raise ValueError("evaluate_class_target must be zero or one")
    if int(config.grid_size) ** 2 != int(panel.states.shape[1]):
        raise ValueError("panel and dynamics grid sizes disagree")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    mask = panel.class_targets.detach().cpu().numpy().astype(np.int64) == class_value
    selected_indices = np.flatnonzero(mask)
    target_device = torch.device(device)
    states = panel.states.index_select(
        0, torch.as_tensor(selected_indices, device=panel.states.device)
    ).to(target_device)
    fractions = panel.tau_fraction.index_select(
        0, torch.as_tensor(selected_indices, device=panel.tau_fraction.device)
    ).to(target_device)
    tau = panel.tau.index_select(
        0, torch.as_tensor(selected_indices, device=panel.tau.device)
    ).to(target_device)
    labels = panel.labels.index_select(
        0, torch.as_tensor(selected_indices, device=panel.labels.device)
    ).to(target_device)
    strata = np.asarray(panel.strata, dtype=np.int64)[selected_indices]

    predicted_logits: list[Tensor] = []
    predicted_gradients: list[Tensor] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, int(states.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(states.shape[0]))
            states_batch = states[start:end].detach().clone().requires_grad_(True)
            logits = model(tau[start:end], states_batch, labels[start:end]).reshape(-1)
            if logits.shape != (end - start,):
                raise ValueError("density-ratio model must return one raw logit per state")
            gradient = torch.autograd.grad(logits.sum(), states_batch)[0]
            predicted_logits.append(logits.detach())
            predicted_gradients.append(gradient.detach())
    finally:
        model.train(was_training)

    prediction_logit = torch.cat(predicted_logits)
    prediction_gradient = torch.cat(predicted_gradients)
    prediction_edge = edge_difference_channels(
        prediction_gradient, int(config.grid_size)
    )
    target_logit = equal_prior_bayes_logit(
        states,
        fractions,
        task="bounded_teacher",
        epsilon=float(epsilon),
    )
    target_edge = bounded_teacher_edge_score(
        states, fractions, epsilon=float(epsilon)
    )
    theta = harmonic_mobility_exact(states, config)
    strata_tensor = torch.as_tensor(strata, device=target_device, dtype=torch.long)

    def scope(mask_value: np.ndarray) -> dict[str, Any]:
        ids = torch.as_tensor(
            np.flatnonzero(mask_value), device=target_device, dtype=torch.long
        )
        return _scope_analytic_metrics(
            states=states,
            predicted_logits=prediction_logit,
            target_logits=target_logit,
            predicted_cell_gradient=prediction_gradient,
            predicted_edge_score=prediction_edge,
            target_edge_score=target_edge,
            theta=theta,
            strata=strata_tensor,
            config=config,
            ids=ids,
        )

    overall = scope(np.ones(states.shape[0], dtype=bool))
    data_end = scope(strata == 4)
    bins = [scope(strata == index) for index in range(5)]
    finite = bool(
        torch.isfinite(prediction_logit).all()
        and torch.isfinite(prediction_gradient).all()
        and torch.isfinite(prediction_edge).all()
    )
    return {
        "complete": 1,
        "finite": int(finite),
        "class_target": class_value,
        "audit_overall_score_gain": overall["score_gain"],
        "audit_data_end_score_gain": data_end["score_gain"],
        "overall_flux_cosine": overall["flux_cosine"],
        "time_bin_flux_cosines": [value["flux_cosine"] for value in bins],
        "overall_relative_flux_l2": overall["flux_relative_l2"],
        "time_bin_relative_flux_l2": [value["flux_relative_l2"] for value in bins],
        "raw_logit_correlation": overall["raw_logit_correlation"],
        "time_bin_centered_logit_mse": overall["time_bin_centered_logit_mse"],
        "time_bin_centered_logit_correlation": overall[
            "time_bin_centered_logit_correlation"
        ],
        "predicted_cell_gradient_abs_quantiles": overall[
            "predicted_cell_gradient_abs_quantiles"
        ],
        "predicted_edge_score_abs_quantiles": overall[
            "predicted_edge_score_abs_quantiles"
        ],
        "predicted_physical_flux_abs_quantiles": overall[
            "predicted_physical_flux_abs_quantiles"
        ],
        "overall": overall,
        "data_end": data_end,
        "time_bins": bins,
        "panel_fingerprint": panel.fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def calibrate_density_ratio_loss_scale(
    model: nn.Module,
    panel: DensityRatioPanel,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    target_initial_gradient_norm: float = 0.10,
    binding: Mapping[str, Any] | None = None,
) -> LossScaleCalibration:
    """Calibrate raw-logit BCE optimizer units on a training-only panel."""

    role = panel.role.lower()
    if "train" not in role and "calibration" not in role:
        raise ValueError("density-ratio loss calibration requires a training-only panel")
    if int(batch_size) <= 0 or int(batch_size) % 64:
        raise ValueError("calibration batch_size must be a positive multiple of 64")
    target_device = torch.device(device)

    def objective_batches():
        for start in range(0, int(panel.states.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(panel.states.shape[0]))
            logits = model(
                panel.tau[start:end].to(target_device),
                panel.states[start:end].to(target_device),
                panel.labels[start:end].to(target_device),
            ).reshape(-1)
            yield classification_loss(
                logits, panel.class_targets[start:end].to(target_device)
            ), end - start

    calibration_binding = {
        "objective_version": DENSITY_RATIO_OBJECTIVE_VERSION,
        "panel_fingerprint": panel.fingerprint,
        "plan_fingerprint": panel.plan_fingerprint,
        "task": panel.task,
        "role": panel.role,
        **({} if binding is None else dict(binding)),
    }
    return calibrate_initial_loss_scale(
        model,
        objective_batches,
        objective_kind="density_ratio_balanced_raw_logit_bce",
        calibration_state_sha256=_tensor_fingerprint(panel.states),
        binding=calibration_binding,
        target_initial_gradient_norm=float(target_initial_gradient_norm),
        calibration_state_count=int(panel.states.shape[0]),
        calibration_split="train",
    )
