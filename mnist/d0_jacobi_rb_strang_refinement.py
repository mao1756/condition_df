"""Exact variable-K scheduler for Jacobi RB Strang-refinement controls.

This module is additive: it does not change the certified Jacobi transition,
the seven-phase palindrome, or the historical K=512 multipath scheduler.  It
adds a finest-tick random-ID plan that couples the supported temporal levels
through the same stateless Philox quantiles while recomputing the exact
state-dependent pair exposure at every phase.

The scheduler is controls-only.  It contains neither a neural trainer nor a
reverse sampler, and it provides no approximate transition fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import operator
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist import d0_jacobi_rb_cuda_controls as _controls
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_cuda_multipath import (
    canonical_same_phase_transition_ids,
)


REFINEMENT_SCHEDULER_VERSION = "jacobi-rb-state-dependent-strang-refinement-v1"
REFINEMENT_ID_VERSION = "jacobi-rb-finest-tick-k2048-id-v1"
REFINEMENT_RNG_VERSION = "jacobi-rb-strang-common-quantile-v1"
REFINEMENT_OBSERVABLE_VERSION = "jacobi-rb-grid-observables-v1"
SUPPORTED_SAMPLE_STEPS = (128, 256, 512, 1024, 2048)
FINEST_SAMPLE_STEPS = 2048
REFINEMENT_SHARD_STEPS = 8
MAX_REFINEMENT_PATHS_PER_GROUP = 8
GRID_SIZE = 28
PATH_STATE_SIZE = GRID_SIZE * GRID_SIZE
EDGES_PER_PHASE = PATH_STATE_SIZE // 2
PHASE_MATCHINGS = (0, 1, 2, 3, 2, 1, 0)
PHASE_DURATIONS = (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
TAU_EFF = 5.0e-5
GRID_SPACING = 1.0 / GRID_SIZE

# The packed ID fields are, low to high: edge 9 bits, phase 3 bits,
# finest-tick 11 bits, path 20 bits.  They occupy only 43 uint64 bits.
_EDGE_BITS = 9
_PHASE_BITS = 3
_TICK_BITS = 11
_LOW_COORDINATE_BITS = _EDGE_BITS + _PHASE_BITS + _TICK_BITS
_MAX_PATH_ID = 1 << 20

_FORBIDDEN_DIAGNOSTICS = (
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


class RefinementTransitionIDProvider(Protocol):
    """Callable hook used for historical K=512 ID injection."""

    def __call__(
        self,
        path_ids: Sequence[int],
        *,
        sample_steps: int,
        outer_step: int,
        phase: int,
        device: torch.device,
    ) -> Tensor: ...


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(unique.tolist(), counts.tolist(), strict=True)
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
        if not 0 <= integer < _MAX_PATH_ID:
            raise ValueError("path IDs must fit the frozen 20-bit field")
        result.append(integer)
    if not result:
        raise ValueError("at least one path is required")
    if len(result) > MAX_REFINEMENT_PATHS_PER_GROUP:
        raise ValueError("a refinement cohort contains at most eight paths")
    if len(set(result)) != len(result):
        raise ValueError("path IDs must be unique within a refinement shard")
    return tuple(result)


def _sample_steps(value: int) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("sample_steps must be an integer") from exc
    if result not in SUPPORTED_SAMPLE_STEPS:
        raise ValueError(
            "sample_steps must be one of "
            + ",".join(str(item) for item in SUPPORTED_SAMPLE_STEPS)
        )
    return result


def _panel_namespace(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("panel_namespace must be text")
    result = value.strip()
    if not result or len(result.encode("utf-8")) > 128:
        raise ValueError("panel_namespace must contain 1..128 UTF-8 bytes")
    return result


@dataclass(frozen=True)
class RefinementRNGPlan:
    """Stateless common-quantile namespace for one immutable path panel."""

    root_seed: int
    panel_namespace: str
    id_version: str = REFINEMENT_ID_VERSION
    rng_version: str = REFINEMENT_RNG_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("root_seed must be an integer")
        if not 0 <= int(self.root_seed) < (1 << 64):
            raise ValueError("root_seed must lie in the uint64 range")
        object.__setattr__(self, "panel_namespace", _panel_namespace(self.panel_namespace))
        if self.id_version != REFINEMENT_ID_VERSION:
            raise ValueError("unsupported refinement transition-ID version")
        if self.rng_version != REFINEMENT_RNG_VERSION:
            raise ValueError("unsupported refinement RNG version")

    @property
    def rng_key(self) -> tuple[int, str, str]:
        return (
            int(self.root_seed),
            self.rng_version,
            self.panel_namespace,
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def finest_tick_for_step(sample_steps: int, outer_step: int) -> int:
    """Map a level step to its aligned right-endpoint K=2048 tick."""

    level = _sample_steps(sample_steps)
    try:
        step = operator.index(outer_step)
    except TypeError as exc:
        raise TypeError("outer_step must be an integer") from exc
    if not 0 <= step < level:
        raise ValueError("outer_step lies outside the selected temporal level")
    stride = FINEST_SAMPLE_STEPS // level
    return (step + 1) * stride - 1


def canonical_refinement_transition_ids(
    path_ids: Sequence[int],
    *,
    sample_steps: int,
    outer_step: int,
    phase: int,
    device: torch.device,
) -> Tensor:
    """Return path-major IDs coupled through the K=2048 finest-tick plan."""

    paths = _path_id_tuple(path_ids)
    tick = finest_tick_for_step(sample_steps, outer_step)
    try:
        phase_index = operator.index(phase)
    except TypeError as exc:
        raise TypeError("phase must be an integer") from exc
    if not 0 <= phase_index < len(PHASE_MATCHINGS):
        raise ValueError("phase must lie in the seven-phase palindrome")
    rows: list[Tensor] = []
    for path_id in paths:
        base = (
            (int(path_id) << _LOW_COORDINATE_BITS)
            | (int(tick) << (_PHASE_BITS + _EDGE_BITS))
            | (int(phase_index) << _EDGE_BITS)
        )
        rows.append(
            torch.arange(
                EDGES_PER_PHASE, dtype=torch.int64, device=device
            ).add_(base)
        )
    return torch.stack(rows).reshape(-1).to(torch.uint64).contiguous()


def legacy_k512_transition_ids(
    path_ids: Sequence[int],
    *,
    sample_steps: int,
    outer_step: int,
    phase: int,
    device: torch.device,
) -> Tensor:
    """Inject the immutable parent scheduler's historical K=512 IDs."""

    if _sample_steps(sample_steps) != 512:
        raise ValueError("legacy transition IDs are defined only for K=512")
    return canonical_same_phase_transition_ids(
        path_ids,
        outer_step=int(outer_step),
        phase=int(phase),
        device=device,
    )


def refinement_phase_exposure(
    pair_total: Tensor,
    *,
    sample_steps: int,
    duration_fraction: float,
    tau_eff: float = TAU_EFF,
    grid_spacing: float = GRID_SPACING,
) -> Tensor:
    """Compute the exact alpha=1 state-dependent Jacobi phase exposure."""

    if not isinstance(pair_total, Tensor) or pair_total.dtype != torch.float64:
        raise TypeError("pair_total must be a float64 torch tensor")
    level = _sample_steps(sample_steps)
    duration = float(duration_fraction)
    tau = float(tau_eff)
    h = float(grid_spacing)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_fraction must be finite and nonnegative")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau_eff must be finite and positive")
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("grid_spacing must be finite and positive")
    # The scheduler validates its state once before entering the shard.
    # Avoid a per-phase device synchronization here: certified matching
    # updates preserve finiteness and nonnegativity structurally.
    if pair_total.device.type == "cpu":
        if not bool(torch.isfinite(pair_total).all()):
            raise ValueError("pair_total must be finite")
        if not bool((pair_total >= 0.0).all()):
            raise ValueError("pair_total must be nonnegative")
    positive = pair_total > 0.0
    safe_total = torch.where(positive, pair_total, torch.ones_like(pair_total))
    numerator = torch.as_tensor(
        3.0 * (tau / float(level)) * duration / (h * h),
        dtype=torch.float64,
        device=pair_total.device,
    )
    return torch.where(
        positive, numerator / safe_total, torch.zeros_like(pair_total)
    ).contiguous()


@dataclass(frozen=True)
class DirichletObservableMoment:
    """Exact Dirichlet(1) raw moment and its binary64 representation."""

    name: str
    family: str
    mean_numerator: int
    mean_denominator: int
    second_moment_numerator: int
    second_moment_denominator: int
    variance_numerator: int
    variance_denominator: int

    @property
    def mean(self) -> float:
        return self.mean_numerator / self.mean_denominator

    @property
    def second_moment(self) -> float:
        return self.second_moment_numerator / self.second_moment_denominator

    @property
    def variance(self) -> float:
        return self.variance_numerator / self.variance_denominator

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "mean": self.mean,
            "second_moment": self.second_moment,
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
        }


@dataclass(frozen=True)
class RefinementObservableSpec:
    """The frozen eight Fourier plus quadratic/cubic observable basis."""

    grid_size: int
    names: tuple[str, ...]
    families: tuple[str, ...]
    moments: tuple[DirichletObservableMoment, ...]
    fourier_weights: np.ndarray = field(repr=False, compare=False)
    version: str = REFINEMENT_OBSERVABLE_VERSION

    def __post_init__(self) -> None:
        weights = np.asarray(self.fourier_weights, dtype=np.float64)
        expected = (8, int(self.grid_size) ** 2)
        if weights.shape != expected or not np.all(np.isfinite(weights)):
            raise ValueError(f"fourier_weights must have shape {expected}")
        frozen = np.array(weights, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "fourier_weights", frozen)
        if len(self.names) != 10 or len(self.families) != 10:
            raise ValueError("the refinement basis must contain ten observables")
        if len(self.moments) != 10:
            raise ValueError("every refinement observable requires an exact moment")

    @property
    def means(self) -> np.ndarray:
        return np.asarray([moment.mean for moment in self.moments], dtype=np.float64)

    @property
    def standard_deviations(self) -> np.ndarray:
        return np.asarray(
            [moment.standard_deviation for moment in self.moments],
            dtype=np.float64,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "grid_size": int(self.grid_size),
            "names": list(self.names),
            "families": list(self.families),
            "moments": [moment.to_record() for moment in self.moments],
            "fourier_weight_sha256": hashlib.sha256(
                self.fourier_weights.tobytes(order="C")
            ).hexdigest(),
        }


def _moment(
    name: str,
    family: str,
    mean: Fraction,
    second_moment: Fraction,
) -> DirichletObservableMoment:
    variance = second_moment - mean * mean
    if variance <= 0:
        raise AssertionError("observable variance must be positive")
    return DirichletObservableMoment(
        name=name,
        family=family,
        mean_numerator=mean.numerator,
        mean_denominator=mean.denominator,
        second_moment_numerator=second_moment.numerator,
        second_moment_denominator=second_moment.denominator,
        variance_numerator=variance.numerator,
        variance_denominator=variance.denominator,
    )


def exact_dirichlet_observable_moments(
    grid_size: int = GRID_SIZE,
) -> tuple[DirichletObservableMoment, ...]:
    """Return exact raw moments under Dirichlet(1) on the grid simplex."""

    n_grid = operator.index(grid_size)
    if n_grid <= 2 or n_grid % 2:
        raise ValueError("grid_size must be an even integer greater than two")
    count = n_grid * n_grid
    fourier_variance = Fraction(1, 2 * (count + 1))
    fourier = tuple(
        _moment(
            name,
            "linear",
            Fraction(0, 1),
            fourier_variance,
        )
        for name in (
            "x_frequency_1_cos",
            "x_frequency_1_sin",
            "x_frequency_2_cos",
            "x_frequency_2_sin",
            "y_frequency_1_cos",
            "y_frequency_1_sin",
            "y_frequency_2_cos",
            "y_frequency_2_sin",
        )
    )
    q_mean = Fraction(2, count + 1)
    q_second = Fraction(
        4 * (count + 5), (count + 1) * (count + 2) * (count + 3)
    )
    c_mean = Fraction(6, (count + 1) * (count + 2))
    c_second = Fraction(
        36 * (count + 19),
        (count + 1)
        * (count + 2)
        * (count + 3)
        * (count + 4)
        * (count + 5),
    )
    return fourier + (
        _moment("quadratic_mass", "quadratic", q_mean, q_second),
        _moment("cubic_mass", "cubic", c_mean, c_second),
    )


def refinement_observable_spec(
    grid_size: int = GRID_SIZE,
) -> RefinementObservableSpec:
    """Build the immutable periodic observable basis."""

    n = operator.index(grid_size)
    moments = exact_dirichlet_observable_moments(n)
    rows, cols = np.indices((n, n), dtype=np.float64)
    basis: list[np.ndarray] = []
    for coordinates in (cols, rows):
        for frequency in (1, 2):
            angle = 2.0 * math.pi * frequency * coordinates / float(n)
            basis.extend((np.cos(angle).reshape(-1), np.sin(angle).reshape(-1)))
    weights = np.stack(basis)
    # Remove only roundoff in the analytically zero sums.  This fixes the
    # represented finite-precision observable and preserves its exact
    # Dirichlet mean at zero.
    weights -= np.mean(weights, axis=1, keepdims=True)
    expected_norm = (n * n) / 2.0
    if not np.allclose(
        np.sum(weights * weights, axis=1),
        expected_norm,
        rtol=2.0e-14,
        atol=2.0e-14,
    ):
        raise AssertionError("periodic Fourier weights lost their exact norm")
    return RefinementObservableSpec(
        grid_size=n,
        names=tuple(moment.name for moment in moments),
        families=tuple(moment.family for moment in moments),
        moments=moments,
        fourier_weights=weights,
    )


def evaluate_refinement_observables(
    states: Tensor | np.ndarray,
    *,
    spec: RefinementObservableSpec | None = None,
    standardized: bool = True,
) -> Tensor | np.ndarray:
    """Evaluate the frozen observables, optionally centered to unit variance."""

    selected = spec or refinement_observable_spec(GRID_SIZE)
    count = int(selected.grid_size) ** 2
    if isinstance(states, Tensor):
        if states.dtype != torch.float64 or states.shape[-1] != count:
            raise TypeError(f"states must be float64 with final dimension {count}")
        if states.device.type == "cpu" and not bool(torch.isfinite(states).all()):
            raise ValueError("states must be finite")
        weights = torch.as_tensor(
            np.array(selected.fourier_weights, copy=True),
            dtype=torch.float64,
            device=states.device,
        )
        linear = states @ weights.t()
        raw = torch.cat(
            (
                linear,
                torch.sum(states * states, dim=-1, keepdim=True),
                torch.sum(states * states * states, dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        if not standardized:
            return raw
        means = torch.as_tensor(
            selected.means, dtype=torch.float64, device=states.device
        )
        scales = torch.as_tensor(
            selected.standard_deviations,
            dtype=torch.float64,
            device=states.device,
        )
        return (raw - means) / scales

    values = np.asarray(states)
    if values.dtype != np.float64 or values.shape[-1] != count:
        raise TypeError(f"states must be float64 with final dimension {count}")
    if not np.all(np.isfinite(values)):
        raise ValueError("states must be finite")
    linear = values @ selected.fourier_weights.T
    raw_numpy = np.concatenate(
        (
            linear,
            np.sum(values * values, axis=-1, keepdims=True),
            np.sum(values * values * values, axis=-1, keepdims=True),
        ),
        axis=-1,
    )
    if not standardized:
        return raw_numpy
    return (raw_numpy - selected.means) / selected.standard_deviations


@dataclass(frozen=True)
class RefinementObservableCheckpoint:
    """Post-step standardized observables committed at a shard boundary."""

    completed_step: int
    time_fraction: float
    path_ids: tuple[int, ...]
    values: np.ndarray = field(repr=False, compare=False)
    values_sha256: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        expected = (len(self.path_ids), 10)
        if values.shape != expected:
            raise ValueError(f"checkpoint values must have shape {expected}")
        frozen = np.array(values, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "values", frozen)

    def to_record(self, *, include_values: bool = True) -> dict[str, Any]:
        result = {
            "completed_step": int(self.completed_step),
            "time_fraction": float(self.time_fraction),
            "path_ids": list(self.path_ids),
            "values_sha256": self.values_sha256,
        }
        if include_values:
            result["values"] = self.values.tolist()
        return result


@dataclass(frozen=True)
class RefinementPathRecord:
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
class RefinementPhaseStateRecord:
    outer_step: int
    phase: int
    path_state_sha256_by_id: tuple[tuple[int, str], ...]
    batch_state_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RefinementShardResult:
    """One exact variable-K restart shard."""

    final_states: Tensor
    committed_final_states: np.ndarray = field(repr=False, compare=False)
    path_records: tuple[RefinementPathRecord, ...]
    phase_state_records: tuple[RefinementPhaseStateRecord, ...]
    observable_checkpoints: tuple[RefinementObservableCheckpoint, ...]
    batch_output_sha256: str
    batch_final_state_sha256: str
    batch_certificate_sha256: str
    diagnostics: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": REFINEMENT_SCHEDULER_VERSION + "-shard",
            "schema_version": 1,
            "path_records": [record.to_dict() for record in self.path_records],
            "phase_state_records": [
                record.to_dict() for record in self.phase_state_records
            ],
            "observable_checkpoints": [
                record.to_record() for record in self.observable_checkpoints
            ],
            "batch_output_sha256": self.batch_output_sha256,
            "batch_final_state_sha256": self.batch_final_state_sha256,
            "batch_certificate_sha256": self.batch_certificate_sha256,
            "diagnostics": dict(self.diagnostics),
        }


def _result_tensor(
    result: Any,
    *names: str,
    shape: tuple[int, int],
    dtype: torch.dtype | None = None,
) -> Tensor:
    value = _controls._field(result, *names)
    if not isinstance(value, Tensor) or value.numel() != math.prod(shape):
        raise _controls.RigorousCudaControlError(
            "refinement sampler returned an invalid device tensor"
        )
    reshaped = value.reshape(shape)
    return reshaped if dtype is None else reshaped.to(dtype=dtype)


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
    if not isinstance(value, Tensor) or value.numel() != math.prod(shape):
        raise _controls.RigorousCudaControlError(
            f"refinement sampler output {name} is invalid"
        )
    if value.device != device:
        raise _controls.RigorousCudaControlError(
            f"refinement sampler output {name} left the selected device"
        )
    return value.reshape(shape).to(dtype=dtype)


def _diagnostic_tensor(
    result: Any,
    name: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    diagnostics = _controls._optional_field(result, "diagnostics", {})
    value = diagnostics.get(name) if isinstance(diagnostics, Mapping) else None
    if value is None:
        return torch.zeros((), dtype=dtype, device=device)
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise _controls.RigorousCudaControlError(
                f"CUDA diagnostic {name} must be scalar"
            )
        return value.reshape(()).to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device).reshape(())


def _matching_arrays(device: torch.device) -> tuple[tuple[Tensor, Tensor], ...]:
    return tuple(
        (
            torch.as_tensor(tails, dtype=torch.int64, device=device).contiguous(),
            torch.as_tensor(heads, dtype=torch.int64, device=device).contiguous(),
        )
        for tails, heads in _controls._matching_arrays()
    )


def run_refinement_shard(
    states: Tensor,
    *,
    path_ids: Sequence[int],
    sample_steps: int,
    start_step: int,
    root_seed: int,
    panel_namespace: str,
    profile: JacobiRBCudaProfile,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
    checkpoint_steps: Sequence[int] = (),
    transition_id_provider: RefinementTransitionIDProvider | None = None,
    rng_key_override: Any | None = None,
    capture_phase_state_trace: bool = False,
) -> RefinementShardResult:
    """Advance at most eight paths through one exact eight-step shard.

    All evolving state, transition evidence, and observables remain on the
    input device until the one post-shard commitment transfer.  Passing
    ``legacy_k512_transition_ids`` with
    ``rng_key_override=(root_seed, "full-path-v2")`` replays the immutable
    historical K=512 random namespace without changing the default v1 plan.
    """

    if not isinstance(states, Tensor):
        raise TypeError("states must be a torch.Tensor")
    if (
        states.dtype != torch.float64
        or states.ndim != 2
        or states.shape[1] != PATH_STATE_SIZE
        or not states.is_contiguous()
    ):
        raise ValueError("states must be contiguous float64 with shape [P,784]")
    if sampler is sample_alpha1_rb_transition_batch_cuda and not states.is_cuda:
        raise ValueError("the production refinement scheduler requires CUDA states")
    if not isinstance(profile, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")
    level = _sample_steps(sample_steps)
    try:
        first_step = operator.index(start_step)
    except TypeError as exc:
        raise TypeError("start_step must be an integer") from exc
    if (
        first_step < 0
        or first_step % REFINEMENT_SHARD_STEPS
        or first_step + REFINEMENT_SHARD_STEPS > level
    ):
        raise ValueError(
            "start_step must begin a complete eight-step refinement shard"
        )
    paths = _path_id_tuple(path_ids)
    if len(paths) != int(states.shape[0]):
        raise ValueError("path_ids must match the leading state dimension")
    rng_plan = RefinementRNGPlan(
        root_seed=int(root_seed), panel_namespace=panel_namespace
    )
    provider = transition_id_provider or canonical_refinement_transition_ids
    requested_checkpoints = tuple(sorted({operator.index(v) for v in checkpoint_steps}))
    if any(value <= 0 or value > level for value in requested_checkpoints):
        raise ValueError("checkpoint steps must lie in 1..sample_steps")

    device = states.device
    initial_mass = states.sum(dim=1)
    validation = torch.stack(
        (
            torch.isfinite(states).all(),
            (states >= 0.0).all(),
            (torch.abs(initial_mass - 1.0) <= 2.0e-12).all(),
        )
    ).detach().cpu().tolist()
    if not bool(validation[0]) or not bool(validation[1]):
        raise ValueError("states must be finite and nonnegative")
    if not bool(validation[2]):
        raise ValueError("every refinement state must lie on the unit simplex")

    initial_states = states.detach().clone()
    values = states.detach().clone()
    matchings = _matching_arrays(device)
    path_count = len(paths)
    shape = (path_count, EDGES_PER_PHASE)
    per_path_certified = torch.zeros(path_count, dtype=torch.int64, device=device)
    per_path_fallback = torch.zeros(path_count, dtype=torch.int64, device=device)
    per_path_strengthened = torch.zeros(path_count, dtype=torch.int64, device=device)
    per_path_max_mode = torch.zeros(path_count, dtype=torch.int32, device=device)
    per_path_max_prefix = torch.zeros(path_count, dtype=torch.int32, device=device)
    maximum_pair_error = torch.zeros((), dtype=torch.float64, device=device)
    maximum_launch = torch.zeros((), dtype=torch.int64, device=device)
    fallback_seconds = torch.zeros((), dtype=torch.float64, device=device)
    fused_seconds = torch.zeros((), dtype=torch.float64, device=device)
    candidate_seconds = torch.zeros((), dtype=torch.float64, device=device)
    authorizer_launches = torch.zeros((), dtype=torch.int64, device=device)
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
    observable_steps: list[int] = []
    observable_blocks: list[Tensor] = []
    observable_spec = refinement_observable_spec(GRID_SIZE)

    started = time.perf_counter()
    for local_step in range(REFINEMENT_SHARD_STEPS):
        outer_step = first_step + local_step
        for phase, (matching_index, duration) in enumerate(
            zip(PHASE_MATCHINGS, PHASE_DURATIONS, strict=True)
        ):
            tails, heads = matchings[matching_index]
            tail_mass = values.index_select(1, tails)
            head_mass = values.index_select(1, heads)
            pair_total = tail_mass + head_mass
            positive = pair_total > 0.0
            safe_total = torch.where(
                positive, pair_total, torch.ones_like(pair_total)
            )
            current = torch.where(
                positive, head_mass / safe_total, torch.zeros_like(pair_total)
            ).contiguous()
            exposure = refinement_phase_exposure(
                pair_total,
                sample_steps=level,
                duration_fraction=duration,
            )
            ids = provider(
                paths,
                sample_steps=level,
                outer_step=outer_step,
                phase=phase,
                device=device,
            )
            if (
                not isinstance(ids, Tensor)
                or ids.dtype != torch.uint64
                or ids.device != device
                or ids.numel() != path_count * EDGES_PER_PHASE
                or not ids.is_contiguous()
            ):
                raise _controls.RigorousCudaControlError(
                    "transition-ID provider violated the uint64 device contract"
                )
            result = _controls._call_sampler(
                current.reshape(-1).contiguous(),
                exposure.reshape(-1).contiguous(),
                profile=profile,
                rng_key=(rng_plan.rng_key if rng_key_override is None else rng_key_override),
                transition_offset=0,
                transition_ids=ids,
                sampler=sampler,
            )
            later = _result_tensor(
                result,
                "later_head_fraction",
                "later",
                "y",
                shape=shape,
                dtype=torch.float64,
            )
            target = _result_tensor(
                result,
                "denoising_target",
                "target",
                "z",
                shape=shape,
                dtype=torch.float64,
            )
            codes = _result_tensor(
                result,
                "certificate_codes",
                "certificate_code",
                shape=shape,
                dtype=torch.uint8,
            )
            if any(item.device != device for item in (later, target, codes)):
                raise _controls.RigorousCudaControlError(
                    "refinement sampler output left the selected device"
                )
            fallback = _optional_result_tensor(
                result, "fallback_mask", shape=shape, dtype=torch.bool, device=device
            )
            strengthened = _optional_result_tensor(
                result,
                "strengthened_mask",
                shape=shape,
                dtype=torch.bool,
                device=device,
            )
            modes = _optional_result_tensor(
                result, "mode_counts", shape=shape, dtype=torch.int32, device=device
            )
            prefixes = _optional_result_tensor(
                result, "prefix_bits", shape=shape, dtype=torch.int32, device=device
            )
            fallback_reasons = _optional_result_tensor(
                result,
                "arb_fallback_reason_codes",
                shape=shape,
                dtype=torch.uint8,
                device=device,
            )
            certified = (codes & 0xF) == 0xF
            per_path_certified += certified.sum(dim=1, dtype=torch.int64)
            per_path_fallback += fallback.sum(dim=1, dtype=torch.int64)
            per_path_strengthened += strengthened.sum(dim=1, dtype=torch.int64)
            per_path_max_mode = torch.maximum(per_path_max_mode, modes.max(dim=1).values)
            per_path_max_prefix = torch.maximum(
                per_path_max_prefix, prefixes.max(dim=1).values
            )
            maximum_launch = torch.maximum(
                maximum_launch,
                _diagnostic_tensor(
                    result,
                    "maximum_cuda_launch_lanes",
                    dtype=torch.int64,
                    device=device,
                ),
            )
            authorizer_launches += _diagnostic_tensor(
                result,
                "fused_authorizer_launch_count",
                dtype=torch.int64,
                device=device,
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
            for name in _FORBIDDEN_DIAGNOSTICS:
                forbidden[name] += _diagnostic_tensor(
                    result, name, dtype=torch.int64, device=device
                )

            new_tail = pair_total * (1.0 - later)
            new_head = pair_total * later
            values[:, tails] = new_tail
            values[:, heads] = new_head
            maximum_pair_error = torch.maximum(
                maximum_pair_error,
                torch.max(torch.abs((new_tail + new_head) - pair_total)),
            )
            later_blocks.append(later.detach())
            target_blocks.append(target.detach())
            code_blocks.append(codes.detach())
            mode_blocks.append(modes.detach())
            prefix_blocks.append(prefixes.detach())
            fallback_reason_blocks.append(fallback_reasons.detach())
            if capture_phase_state_trace:
                phase_state_blocks.append(values.detach().clone())

        completed_step = outer_step + 1
        if completed_step in requested_checkpoints:
            observable_steps.append(completed_step)
            block = evaluate_refinement_observables(
                values, spec=observable_spec, standardized=True
            )
            assert isinstance(block, Tensor)
            observable_blocks.append(block.detach())

    phase_shape = (
        REFINEMENT_SHARD_STEPS * len(PHASE_MATCHINGS),
        path_count,
        EDGES_PER_PHASE,
    )
    state_shape = (path_count, PATH_STATE_SIZE)
    path_shape = (path_count,)
    phase_states_device = (
        torch.stack(phase_state_blocks)
        if capture_phase_state_trace
        else torch.empty((0, *state_shape), dtype=torch.float64, device=device)
    )
    observables_device = (
        torch.stack(observable_blocks)
        if observable_blocks
        else torch.empty((0, path_count, 10), dtype=torch.float64, device=device)
    )
    scalar_device = torch.stack(
        (
            maximum_pair_error,
            maximum_launch.to(torch.float64),
            authorizer_launches.to(torch.float64),
            fallback_seconds,
            fused_seconds,
            candidate_seconds,
            *[forbidden[name].to(torch.float64) for name in _FORBIDDEN_DIAGNOSTICS],
        )
    )
    packed_host = torch.cat(
        (
            torch.stack(later_blocks).reshape(-1),
            torch.stack(target_blocks).reshape(-1),
            torch.stack(code_blocks).reshape(-1).to(torch.float64),
            torch.stack(mode_blocks).reshape(-1).to(torch.float64),
            torch.stack(prefix_blocks).reshape(-1).to(torch.float64),
            torch.stack(fallback_reason_blocks).reshape(-1).to(torch.float64),
            phase_states_device.reshape(-1),
            observables_device.reshape(-1),
            initial_states.reshape(-1),
            values.reshape(-1),
            per_path_certified.to(torch.float64),
            per_path_fallback.to(torch.float64),
            per_path_strengthened.to(torch.float64),
            per_path_max_mode.to(torch.float64),
            per_path_max_prefix.to(torch.float64),
            scalar_device,
        )
    ).detach().cpu().numpy()
    elapsed = time.perf_counter() - started

    offset = 0

    def unpack(shape_: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
        nonlocal offset
        count = math.prod(shape_)
        result_ = packed_host[offset : offset + count].reshape(shape_)
        offset += count
        return result_.astype(dtype, copy=False)

    later_host = unpack(phase_shape, np.dtype(np.float64))
    target_host = unpack(phase_shape, np.dtype(np.float64))
    codes_host = unpack(phase_shape, np.dtype(np.uint8))
    modes_host = unpack(phase_shape, np.dtype(np.int32))
    prefixes_host = unpack(phase_shape, np.dtype(np.int32))
    fallback_reasons_host = unpack(phase_shape, np.dtype(np.uint8))
    phase_states_host = unpack(
        (int(phase_states_device.shape[0]), *state_shape), np.dtype(np.float64)
    )
    observables_host = unpack(
        (len(observable_steps), path_count, 10), np.dtype(np.float64)
    )
    initial_host = unpack(state_shape, np.dtype(np.float64))
    final_host = unpack(state_shape, np.dtype(np.float64))
    final_host.setflags(write=False)
    certified_host = unpack(path_shape, np.dtype(np.int64))
    fallback_host = unpack(path_shape, np.dtype(np.int64))
    strengthened_host = unpack(path_shape, np.dtype(np.int64))
    max_mode_host = unpack(path_shape, np.dtype(np.int32))
    max_prefix_host = unpack(path_shape, np.dtype(np.int32))
    scalar_values = unpack((int(scalar_device.numel()),), np.dtype(np.float64))
    if offset != packed_host.size:
        raise AssertionError("refinement shard summary was not fully decoded")

    if not np.all(np.isfinite(final_host)) or np.any(final_host < 0.0):
        raise _controls.RigorousCudaControlError(
            "refinement shard produced an invalid final state"
        )
    mass_error = float(np.max(np.abs(final_host.sum(axis=1) - 1.0)))
    if mass_error > 2.0e-12:
        raise _controls.RigorousCudaControlError(
            "refinement shard failed global simplex conservation"
        )

    transitions_per_path = (
        REFINEMENT_SHARD_STEPS * len(PHASE_MATCHINGS) * EDGES_PER_PHASE
    )
    path_records: list[RefinementPathRecord] = []
    for path_index, path_id in enumerate(paths):
        output_digest = hashlib.sha256()
        for block in range(phase_shape[0]):
            output_digest.update(
                bytes.fromhex(
                    _controls._digest_arrays(
                        later_host[block, path_index],
                        target_host[block, path_index],
                        codes_host[block, path_index],
                    )
                )
            )
        output_hash = output_digest.hexdigest()
        input_hash = _controls._digest_arrays(initial_host[path_index])
        final_hash = _controls._digest_arrays(final_host[path_index])
        certificate_hash = _fingerprint(
            {
                "version": REFINEMENT_SCHEDULER_VERSION,
                "sample_steps": level,
                "path_id": path_id,
                "start_step": first_step,
                "step_count": REFINEMENT_SHARD_STEPS,
                "input_state_sha256": input_hash,
                "output_sha256": output_hash,
                "final_state_sha256": final_hash,
                "certified_count": int(certified_host[path_index]),
                "fallback_count": int(fallback_host[path_index]),
                "certificate_code_counts": _integer_counts(
                    codes_host[:, path_index]
                ),
                "mode_count_counts": _integer_counts(modes_host[:, path_index]),
                "prefix_bit_counts": _integer_counts(
                    prefixes_host[:, path_index]
                ),
                "arb_fallback_reason_code_counts": _integer_counts(
                    fallback_reasons_host[:, path_index]
                ),
            }
        )
        path_records.append(
            RefinementPathRecord(
                path_id=path_id,
                transition_count=transitions_per_path,
                certified_count=int(certified_host[path_index]),
                fallback_count=int(fallback_host[path_index]),
                strengthened_count=int(strengthened_host[path_index]),
                maximum_mode_count=int(max_mode_host[path_index]),
                maximum_prefix_bits=int(max_prefix_host[path_index]),
                certificate_code_counts=_integer_counts(
                    codes_host[:, path_index]
                ),
                mode_count_counts=_integer_counts(modes_host[:, path_index]),
                prefix_bit_counts=_integer_counts(
                    prefixes_host[:, path_index]
                ),
                arb_fallback_reason_code_counts=_integer_counts(
                    fallback_reasons_host[:, path_index]
                ),
                input_state_sha256=input_hash,
                output_sha256=output_hash,
                final_state_sha256=final_hash,
                certificate_sha256=certificate_hash,
            )
        )

    canonical_indices = sorted(range(path_count), key=lambda index: paths[index])
    canonical_records = sorted(path_records, key=lambda record: record.path_id)
    phase_records: list[RefinementPhaseStateRecord] = []
    for block_index, phase_states in enumerate(phase_states_host):
        local_step, phase = divmod(block_index, len(PHASE_MATCHINGS))
        per_path_hashes = tuple(
            (
                paths[index],
                _controls._digest_arrays(phase_states[index]),
            )
            for index in canonical_indices
        )
        phase_records.append(
            RefinementPhaseStateRecord(
                outer_step=first_step + local_step,
                phase=phase,
                path_state_sha256_by_id=per_path_hashes,
                batch_state_sha256=_controls._digest_arrays(
                    phase_states[canonical_indices]
                ),
            )
        )
    checkpoints = tuple(
        RefinementObservableCheckpoint(
            completed_step=step,
            time_fraction=step / float(level),
            path_ids=tuple(paths[index] for index in canonical_indices),
            values=observables_host[checkpoint_index, canonical_indices],
            values_sha256=_controls._digest_arrays(
                observables_host[checkpoint_index, canonical_indices]
            ),
        )
        for checkpoint_index, step in enumerate(observable_steps)
    )
    batch_output_hash = _fingerprint(
        [[record.path_id, record.output_sha256] for record in canonical_records]
    )
    batch_final_hash = _controls._digest_arrays(final_host[canonical_indices])
    batch_certificate_hash = _fingerprint(
        {
            "version": REFINEMENT_SCHEDULER_VERSION,
            "sample_steps": level,
            "start_step": first_step,
            "step_count": REFINEMENT_SHARD_STEPS,
            "rng_plan": rng_plan.to_record(),
            "path_certificates": [
                [record.path_id, record.certificate_sha256]
                for record in canonical_records
            ],
            "batch_output_sha256": batch_output_hash,
            "batch_final_state_sha256": batch_final_hash,
        }
    )
    scalar_names = (
        "maximum_pair_mass_error",
        "maximum_cuda_launch_lanes",
        "fused_authorizer_launch_count",
        "fallback_elapsed_seconds",
        "fused_authorizer_elapsed_seconds",
        "candidate_elapsed_seconds",
        *_FORBIDDEN_DIAGNOSTICS,
    )
    scalars = dict(zip(scalar_names, scalar_values.tolist(), strict=True))
    transition_count = path_count * transitions_per_path
    certified_count = int(np.sum(certified_host))
    fallback_count = int(np.sum(fallback_host))
    diagnostics: dict[str, Any] = {
        "version": REFINEMENT_SCHEDULER_VERSION,
        "id_version": (
            REFINEMENT_ID_VERSION
            if transition_id_provider is None
            else getattr(provider, "__name__", type(provider).__name__)
        ),
        "sample_steps": level,
        "finest_sample_steps": FINEST_SAMPLE_STEPS,
        "path_count": path_count,
        "path_ids": list(paths),
        "start_step": first_step,
        "step_count": REFINEMENT_SHARD_STEPS,
        "phase_count": REFINEMENT_SHARD_STEPS * len(PHASE_MATCHINGS),
        "backend_call_count": REFINEMENT_SHARD_STEPS * len(PHASE_MATCHINGS),
        "maximum_backend_call_size": path_count * EDGES_PER_PHASE,
        "transition_count": transition_count,
        "certified_count": certified_count,
        "uncertified_count": transition_count - certified_count,
        "certificate_fraction": certified_count / transition_count,
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_count / transition_count,
        "strengthened_count": int(np.sum(strengthened_host)),
        "maximum_mode_count": int(np.max(max_mode_host)),
        "maximum_prefix_bits": int(np.max(max_prefix_host)),
        "certificate_code_counts": _integer_counts(codes_host),
        "mode_count_counts": _integer_counts(modes_host),
        "prefix_bit_counts": _integer_counts(prefixes_host),
        "arb_fallback_reason_code_counts": _integer_counts(
            fallback_reasons_host
        ),
        "maximum_pair_mass_error": float(scalars["maximum_pair_mass_error"]),
        "maximum_global_simplex_error": mass_error,
        "maximum_cuda_launch_lanes": int(scalars["maximum_cuda_launch_lanes"]),
        "fused_authorizer_launch_count": int(
            scalars["fused_authorizer_launch_count"]
        ),
        "fallback_elapsed_seconds": float(scalars["fallback_elapsed_seconds"]),
        "fused_authorizer_elapsed_seconds": float(
            scalars["fused_authorizer_elapsed_seconds"]
        ),
        "candidate_elapsed_seconds": float(
            scalars["candidate_elapsed_seconds"]
        ),
        "panel_namespace": rng_plan.panel_namespace,
        "rng_plan": rng_plan.to_record(),
        "checkpoint_steps": list(observable_steps),
        "observable_spec": observable_spec.to_record(),
        "state_updates_device_resident": 1,
        "diagnostics_device_resident_until_commit": 1,
        "in_shard_host_roundtrip_count": 0,
        "shard_summary_device_to_host_transfer_count": 1,
        "elapsed_seconds": elapsed,
        "transitions_per_second": (
            transition_count / elapsed if elapsed > 0.0 else math.inf
        ),
        **{name: int(scalars[name]) for name in _FORBIDDEN_DIAGNOSTICS},
    }
    return RefinementShardResult(
        final_states=values,
        committed_final_states=final_host,
        path_records=tuple(path_records),
        phase_state_records=tuple(phase_records),
        observable_checkpoints=checkpoints,
        batch_output_sha256=batch_output_hash,
        batch_final_state_sha256=batch_final_hash,
        batch_certificate_sha256=batch_certificate_hash,
        diagnostics=diagnostics,
    )


__all__ = [
    "REFINEMENT_SCHEDULER_VERSION",
    "REFINEMENT_ID_VERSION",
    "REFINEMENT_RNG_VERSION",
    "REFINEMENT_OBSERVABLE_VERSION",
    "SUPPORTED_SAMPLE_STEPS",
    "FINEST_SAMPLE_STEPS",
    "REFINEMENT_SHARD_STEPS",
    "MAX_REFINEMENT_PATHS_PER_GROUP",
    "GRID_SIZE",
    "PATH_STATE_SIZE",
    "EDGES_PER_PHASE",
    "PHASE_MATCHINGS",
    "PHASE_DURATIONS",
    "TAU_EFF",
    "GRID_SPACING",
    "RefinementTransitionIDProvider",
    "RefinementRNGPlan",
    "DirichletObservableMoment",
    "RefinementObservableSpec",
    "RefinementObservableCheckpoint",
    "RefinementPathRecord",
    "RefinementPhaseStateRecord",
    "RefinementShardResult",
    "finest_tick_for_step",
    "canonical_refinement_transition_ids",
    "legacy_k512_transition_ids",
    "refinement_phase_exposure",
    "exact_dirichlet_observable_moments",
    "refinement_observable_spec",
    "evaluate_refinement_observables",
    "run_refinement_shard",
]
