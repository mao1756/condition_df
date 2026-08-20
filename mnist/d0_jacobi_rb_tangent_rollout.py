"""Reusable core for exploratory exact Jacobi boundary-tangent rollouts.

The functions in this module assemble the already-audited Jacobi transition
and tangent logistic flow into restartable forward and reverse trajectories.
They deliberately contain no training, validation/confirmation traversal, or
experiment-CLI imports.  Scientific metrics always operate on raw float64
simplex states; rendering is a separate, fixed-scale presentation operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_artifacts import atomic_write_json
from mnist.d0_jacobi_rb_boundary_tangent_fused import (
    BoundaryTangentContractError,
    TangentScoreController,
    controlled_reverse_phase_tangent,
    edge_pair_geometry,
)
from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate import (
    FrequencyOneCoordinateZeroBaselinePredictor,
)
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_cuda_multipath import (
    SHARD_STEPS,
    run_exact_multipath_shard,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
    matching_indices,
    semantic_sha256,
    state_dict_sha256,
)
from mnist.d0_jacobi_rb_reverse_controller import (
    ALPHA,
    GRID_SPACING,
    MACROSTEP_SCHEDULE_INTEGRAL,
    NAMESPACE_VERSION,
    controller_transition_ids,
    phase_exposure,
)


TANGENT_ROLLOUT_VERSION = "d0-jacobi-rb-tangent-rollout-v1"
EXPLORATORY_REFERENCE_RNG_NAMESPACE = (
    "d0-jacobi-rb-frequency1-exploratory-reference-v1"
)
REVERSE_SHARD_OUTER_STEPS = 8
REVERSE_SHARD_PHASES = REVERSE_SHARD_OUTER_STEPS * PHASE_COUNT
REFERENCE_LANE_CAP = 4096
SIMPLEX_TOLERANCE = 2.0e-12
ORACLE_FRACTION_TOLERANCE = 2.0e-6

_FORBIDDEN_REFERENCE_COUNTS = (
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "clipping_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "projection_count",
    "renormalization_count",
    "nonfinite_count",
)

# These are the counters emitted authoritatively by the exact forward
# multipath scheduler.  ``clipping_count`` and ``projection_count`` belong to
# broader rollout health surfaces, but are not fields in that scheduler's
# committed diagnostic schema and therefore must not be silently invented by
# the strict forward-shard aggregate.
EXACT_FORWARD_FORBIDDEN_COUNTERS_VERSION = (
    "exact-forward-scheduler-forbidden-counts-v1"
)
EXACT_FORWARD_FORBIDDEN_COUNTERS = (
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "renormalization_count",
    "nonfinite_count",
)


class TangentRolloutContractError(ValueError):
    """A rollout input, numerical result, or restart artifact is invalid."""


class ExactForwardShardAggregateError(TangentRolloutContractError):
    """A strict exact-forward aggregate failed its typed evidence contract."""

    def __init__(self, message: str, *, failure_domain: str) -> None:
        if failure_domain not in {"implementation_contract", "numerical_integrity"}:
            raise ValueError("invalid exact-forward aggregate failure domain")
        super().__init__(message)
        self.failure_domain = failure_domain
        self.failure_code = (
            "exact_forward_shard_contract_invalid"
            if failure_domain == "implementation_contract"
            else "exact_forward_shard_numerical_integrity_invalid"
        )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _semantic_record(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result.pop("semantic_sha256", None)
    return {**result, "semantic_sha256": semantic_sha256(result)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray | Tensor) -> str:
    array = (
        value.detach().to(device="cpu").contiguous().numpy()
        if isinstance(value, Tensor)
        else np.ascontiguousarray(value)
    )
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_measure_sha256(value: np.ndarray) -> str:
    measured = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
    digest = hashlib.sha256()
    digest.update(str(measured.shape).encode("ascii"))
    digest.update(measured.tobytes(order="C"))
    return digest.hexdigest()


def source_measure_sha256(value: np.ndarray) -> str:
    """Public frozen float32 source-image semantic hash convention."""

    array = np.asarray(value)
    if array.shape != (STATE_SIZE,) or not np.isfinite(array).all():
        raise TangentRolloutContractError("source hash input must be finite [784]")
    return _source_measure_sha256(array)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TangentRolloutContractError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise TangentRolloutContractError(f"{path} is not a JSON object")
    return value


def _validate_semantic_record(value: Mapping[str, Any], name: str) -> None:
    body = dict(value)
    recorded = body.pop("semantic_sha256", None)
    if not isinstance(recorded, str) or semantic_sha256(body) != recorded:
        raise TangentRolloutContractError(f"{name} semantic hash changed")


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_state_npz(path: Path, *, expected_rows: int) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"state"}:
                raise TangentRolloutContractError("restart state archive schema changed")
            state = np.array(archive["state"], copy=True, order="C")
    except (OSError, ValueError, KeyError) as exc:
        raise TangentRolloutContractError(f"cannot read restart state {path}") from exc
    if state.dtype != np.float64 or state.shape != (expected_rows, STATE_SIZE):
        raise TangentRolloutContractError("restart state has wrong dtype or shape")
    if not np.isfinite(state).all() or np.any(state < 0.0):
        raise TangentRolloutContractError("restart state is not finite/nonnegative")
    mass_error = float(np.max(np.abs(np.sum(state, axis=1) - 1.0)))
    if mass_error > SIMPLEX_TOLERANCE:
        raise TangentRolloutContractError("restart state violates simplex mass")
    return state


def _batched_float64_state(
    value: np.ndarray | Tensor, *, device: torch.device | str | None = None
) -> tuple[Tensor, bool]:
    if isinstance(value, Tensor):
        tensor = value.detach()
        target_device = tensor.device if device is None else torch.device(device)
        tensor = tensor.to(device=target_device, dtype=torch.float64).contiguous()
    else:
        target_device = torch.device("cpu" if device is None else device)
        tensor = torch.as_tensor(
            np.array(value, dtype=np.float64, copy=True, order="C"),
            dtype=torch.float64,
            device=target_device,
        ).contiguous()
    squeezed = tensor.ndim == 1
    if squeezed:
        tensor = tensor.unsqueeze(0).contiguous()
    if tensor.ndim != 2 or tensor.shape[1] != STATE_SIZE:
        raise TangentRolloutContractError("state must have shape [784] or [P,784]")
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any(tensor < 0.0)):
        raise TangentRolloutContractError("state must be finite and nonnegative")
    mass_error = float(torch.max(torch.abs(torch.sum(tensor, dim=1) - 1.0)).item())
    if mass_error > SIMPLEX_TOLERANCE:
        raise TangentRolloutContractError("state violates simplex mass")
    return tensor, squeezed


def _path_ids(value: Sequence[int], rows: int) -> tuple[int, ...]:
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise TypeError("path IDs must be integers")
        path_id = int(item)
        if not 0 <= path_id < (1 << 20):
            raise TangentRolloutContractError("path ID exceeds the 20-bit contract")
        result.append(path_id)
    if len(result) != rows or len(set(result)) != len(result):
        raise TangentRolloutContractError("path IDs must uniquely match state rows")
    return tuple(result)


class ZeroTangentScoreController(nn.Module):
    """Exact zero-score baseline for the tangent control subflow."""

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError("zero controller requires exact ModelInputs")
        state = inputs.later_full_state
        if state.ndim != 2 or state.shape[1] != STATE_SIZE:
            raise TangentRolloutContractError("zero controller state is malformed")
        return torch.zeros(
            (state.shape[0], EDGES_PER_PHASE),
            dtype=torch.float64,
            device=state.device,
        )

    def score_prediction_deferred(self, inputs: ModelInputs) -> Tensor:
        """Device-only fused-dispatch path; zero needs no value predicate."""

        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError("zero controller requires exact ModelInputs")
        return torch.zeros(
            (inputs.batch_size, EDGES_PER_PHASE),
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )


class ScaledTangentScoreController(nn.Module):
    """Inference-only scalar wrapper around one tangent-score controller."""

    def __init__(self, base_controller: TangentScoreController, gain: float) -> None:
        super().__init__()
        if not isinstance(base_controller, TangentScoreController):
            raise TypeError("base controller must implement score_prediction")
        gain_value = float(gain)
        if not math.isfinite(gain_value) or gain_value < 0.0:
            raise TangentRolloutContractError("controller gain must be finite/nonnegative")
        # Registering an nn.Module keeps device movement conventional while no
        # parameter or buffer bytes are modified by this wrapper.
        if isinstance(base_controller, nn.Module):
            self.base_controller = base_controller
        else:
            object.__setattr__(self, "base_controller", base_controller)
        self.gain = gain_value
        self._unscaled_squared_sum = 0.0
        self._scaled_squared_sum = 0.0
        self._count = 0
        self._unscaled_maximum = 0.0
        self._scaled_maximum = 0.0

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError("scaled controller requires exact ModelInputs")
        with torch.inference_mode():
            value = self.base_controller.score_prediction(inputs)
        expected = (inputs.later_full_state.shape[0], EDGES_PER_PHASE)
        if (
            not isinstance(value, Tensor)
            or value.shape != expected
            or value.device != inputs.later_full_state.device
            or not value.dtype.is_floating_point
            or not bool(torch.isfinite(value).all())
        ):
            raise TangentRolloutContractError(
                "base controller output must be finite floating [P,392] on input device"
            )
        scaled = value * self.gain
        active = edge_pair_geometry(inputs).active
        base64 = value.detach().to(dtype=torch.float64)[active]
        scaled64 = scaled.detach().to(dtype=torch.float64)[active]
        self._unscaled_squared_sum += float(torch.sum(base64.square()).item())
        self._scaled_squared_sum += float(torch.sum(scaled64.square()).item())
        self._count += int(base64.numel())
        if base64.numel():
            self._unscaled_maximum = max(
                self._unscaled_maximum, float(torch.max(torch.abs(base64)).item())
            )
            self._scaled_maximum = max(
                self._scaled_maximum, float(torch.max(torch.abs(scaled64)).item())
            )
        return scaled

    def score_prediction_deferred(self, inputs: ModelInputs) -> Tensor:
        """One-row score path whose validity is checked at the shard boundary."""

        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError("scaled controller requires exact ModelInputs")
        method = getattr(self.base_controller, "score_prediction_prevalidated", None)
        if not callable(method):
            raise TangentRolloutContractError(
                "fused learned controller lacks a prevalidated score path"
            )
        with torch.inference_mode():
            value = method(inputs)
        expected = (inputs.batch_size, EDGES_PER_PHASE)
        if (
            not isinstance(value, Tensor)
            or value.shape != expected
            or value.device != inputs.later_full_state.device
            or not value.dtype.is_floating_point
        ):
            raise TangentRolloutContractError(
                "prevalidated controller output must be floating [P,392] on device"
            )
        unscaled = value.to(dtype=torch.float64)
        scaled = unscaled * self.gain
        # These tensors are intentionally retained on device.  The fused bank
        # consumes them before the next row call; no cumulative Python scalar
        # record is modified on this path.
        self._last_deferred_unscaled = unscaled  # type: ignore[attr-defined]
        return scaled

    def record(self) -> dict[str, Any]:
        return {
            "gain": self.gain,
            "score_count": self._count,
            "unscaled_score_squared_sum": self._unscaled_squared_sum,
            "scaled_score_squared_sum": self._scaled_squared_sum,
            "unscaled_score_rms": math.sqrt(
                self._unscaled_squared_sum / self._count
            ) if self._count else 0.0,
            "scaled_score_rms": math.sqrt(
                self._scaled_squared_sum / self._count
            ) if self._count else 0.0,
            "unscaled_score_maximum_absolute": self._unscaled_maximum,
            "scaled_score_maximum_absolute": self._scaled_maximum,
        }


class SignedDiagnosticTangentScoreController(nn.Module):
    """Diagnostic-only negative scalar wrapper for a tangent-score controller.

    This is deliberately separate from :class:`ScaledTangentScoreController`,
    whose finite, nonnegative gain contract remains the production contract.
    """

    def __init__(self, base_controller: TangentScoreController, gain: float) -> None:
        super().__init__()
        if not isinstance(base_controller, TangentScoreController):
            raise TypeError("base controller must implement score_prediction")
        gain_value = float(gain)
        if not math.isfinite(gain_value) or gain_value >= 0.0:
            raise TangentRolloutContractError(
                "signed diagnostic controller gain must be finite/negative"
            )
        if isinstance(base_controller, nn.Module):
            self.base_controller = base_controller
        else:
            object.__setattr__(self, "base_controller", base_controller)
        self.gain = gain_value
        self._unscaled_squared_sum = 0.0
        self._scaled_squared_sum = 0.0
        self._count = 0
        self._unscaled_maximum = 0.0
        self._scaled_maximum = 0.0

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError(
                "signed diagnostic controller requires exact ModelInputs"
            )
        with torch.inference_mode():
            value = self.base_controller.score_prediction(inputs)
        expected = (inputs.later_full_state.shape[0], EDGES_PER_PHASE)
        if (
            not isinstance(value, Tensor)
            or value.shape != expected
            or value.device != inputs.later_full_state.device
            or not value.dtype.is_floating_point
            or not bool(torch.isfinite(value).all())
        ):
            raise TangentRolloutContractError(
                "base controller output must be finite floating [P,392] on input device"
            )
        scaled = value * self.gain
        active = edge_pair_geometry(inputs).active
        base64 = value.detach().to(dtype=torch.float64)[active]
        scaled64 = scaled.detach().to(dtype=torch.float64)[active]
        self._unscaled_squared_sum += float(torch.sum(base64.square()).item())
        self._scaled_squared_sum += float(torch.sum(scaled64.square()).item())
        self._count += int(base64.numel())
        if base64.numel():
            self._unscaled_maximum = max(
                self._unscaled_maximum, float(torch.max(torch.abs(base64)).item())
            )
            self._scaled_maximum = max(
                self._scaled_maximum, float(torch.max(torch.abs(scaled64)).item())
            )
        return scaled

    def score_prediction_deferred(self, inputs: ModelInputs) -> Tensor:
        """Device-only diagnostic path checked at the fused shard boundary."""

        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError(
                "signed diagnostic controller requires exact ModelInputs"
            )
        method = getattr(self.base_controller, "score_prediction_prevalidated", None)
        if not callable(method):
            raise TangentRolloutContractError(
                "fused signed diagnostic controller lacks a prevalidated score path"
            )
        with torch.inference_mode():
            value = method(inputs)
        expected = (inputs.batch_size, EDGES_PER_PHASE)
        if (
            not isinstance(value, Tensor)
            or value.shape != expected
            or value.device != inputs.later_full_state.device
            or not value.dtype.is_floating_point
        ):
            raise TangentRolloutContractError(
                "prevalidated controller output must be floating [P,392] on device"
            )
        unscaled = value.to(dtype=torch.float64)
        scaled = unscaled * self.gain
        self._last_deferred_unscaled = unscaled  # type: ignore[attr-defined]
        return scaled

    def record(self) -> dict[str, Any]:
        return {
            "diagnostic_only": 1,
            "gain": self.gain,
            "score_count": self._count,
            "unscaled_score_squared_sum": self._unscaled_squared_sum,
            "scaled_score_squared_sum": self._scaled_squared_sum,
            "unscaled_score_rms": math.sqrt(
                self._unscaled_squared_sum / self._count
            ) if self._count else 0.0,
            "scaled_score_rms": math.sqrt(
                self._scaled_squared_sum / self._count
            ) if self._count else 0.0,
            "unscaled_score_maximum_absolute": self._unscaled_maximum,
            "scaled_score_maximum_absolute": self._scaled_maximum,
        }


class TargetFractionOracleController(nn.Module):
    """Source-informed diagnostic that pulls each movable pair to target fraction."""

    def __init__(self, target_state: np.ndarray | Tensor, microsteps: int) -> None:
        super().__init__()
        target, squeezed = _batched_float64_state(target_state, device="cpu")
        if not squeezed or target.shape != (1, STATE_SIZE):
            raise TangentRolloutContractError("oracle target must have shape [784]")
        if int(microsteps) not in {2, 4, 8}:
            raise TangentRolloutContractError("oracle microsteps must be 2, 4, or 8")
        self.register_buffer("target_state", target[0].clone(), persistent=True)
        self.microsteps = int(microsteps)
        self._call_count = 0
        self._lane_count = 0
        self._movable_count = 0
        self._already_equal_count = 0
        self._zero_pair_mass_count = 0
        self._zero_duration_count = 0
        self._unreachable_boundary_count = 0

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError("oracle requires exact ModelInputs")
        geometry = edge_pair_geometry(inputs)
        device = geometry.pair_mass.device
        rows = int(geometry.pair_mass.shape[0])
        tails_all, heads_all = matching_indices(device=device)
        colors = inputs.color.to(device=device, dtype=torch.long)
        tails = tails_all[colors]
        heads = heads_all[colors]
        target = self.target_state.to(device=device, dtype=torch.float64).expand(rows, -1)
        target_tail = target.gather(1, tails)
        target_head = target.gather(1, heads)
        target_pair = target_tail + target_head
        target_active = target_pair > 0.0
        target_fraction = torch.zeros_like(target_pair)
        target_fraction[target_active] = (
            target_head[target_active] / target_pair[target_active]
        )
        exposure = phase_exposure(
            geometry.pair_mass,
            inputs.duration.to(device=device, dtype=torch.float64)[:, None],
        ) / float(self.microsteps)
        positive_exposure = exposure > 0.0
        equal = geometry.active & target_active & (
            geometry.head_fraction == target_fraction
        )
        needs_move = geometry.active & target_active & positive_exposure & ~equal
        current_interior = (geometry.head_fraction > 0.0) & (
            geometry.head_fraction < 1.0
        )
        target_interior = (target_fraction > 0.0) & (target_fraction < 1.0)
        movable = needs_move & current_interior & target_interior
        unreachable = needs_move & ~movable
        score = torch.zeros_like(geometry.pair_mass, dtype=torch.float64)
        if bool(movable.any()):
            current = geometry.head_fraction[movable]
            wanted = target_fraction[movable]
            score[movable] = (
                torch.log(wanted) - torch.log1p(-wanted)
                - torch.log(current) + torch.log1p(-current)
            ) / (2.0 * exposure[movable])
        if not bool(torch.isfinite(score).all()):
            raise TangentRolloutContractError("target oracle produced a nonfinite score")
        self._call_count += 1
        self._lane_count += int(score.numel())
        self._movable_count += int(torch.count_nonzero(movable).item())
        self._already_equal_count += int(torch.count_nonzero(equal).item())
        self._zero_pair_mass_count += int(torch.count_nonzero(~geometry.active).item())
        self._zero_duration_count += int(
            torch.count_nonzero(geometry.active & ~positive_exposure).item()
        )
        self._unreachable_boundary_count += int(torch.count_nonzero(unreachable).item())
        return score

    def score_prediction_deferred(self, inputs: ModelInputs) -> Tensor:
        """Device-only oracle algebra with telemetry masks left on device."""

        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError("oracle requires exact ModelInputs")
        tails_all, heads_all = matching_indices(
            device=inputs.later_full_state.device
        )
        return self.score_prediction_deferred_prepared(
            inputs, tails_all=tails_all, heads_all=heads_all
        )

    def score_prediction_deferred_prepared(
        self,
        inputs: ModelInputs,
        *,
        tails_all: Tensor,
        heads_all: Tensor,
    ) -> Tensor:
        """Device-only oracle using scheduler-prebuilt matching tensors."""

        if type(inputs) is not ModelInputs:
            raise TangentRolloutContractError("oracle requires exact ModelInputs")
        state = inputs.later_full_state.to(dtype=torch.float64)
        device = state.device
        rows = inputs.batch_size
        if (
            not isinstance(tails_all, Tensor)
            or not isinstance(heads_all, Tensor)
            or tails_all.shape != (4, EDGES_PER_PHASE)
            or heads_all.shape != tails_all.shape
            or tails_all.dtype != torch.long
            or heads_all.dtype != torch.long
            or tails_all.device != device
            or heads_all.device != device
        ):
            raise TangentRolloutContractError(
                "prepared oracle matching tensors are invalid"
            )
        colors = inputs.color.to(device=device, dtype=torch.long)
        tails = tails_all[colors]
        heads = heads_all[colors]
        tail = state.gather(1, tails)
        head = state.gather(1, heads)
        pair = tail + head
        active = pair > 0.0
        current = torch.where(active, head / torch.where(active, pair, torch.ones_like(pair)), torch.zeros_like(pair))

        target = self.target_state.to(device=device, dtype=torch.float64).expand(rows, -1)
        target_tail = target.gather(1, tails)
        target_head = target.gather(1, heads)
        target_pair = target_tail + target_head
        target_active = target_pair > 0.0
        wanted = torch.where(
            target_active,
            target_head
            / torch.where(target_active, target_pair, torch.ones_like(target_pair)),
            torch.zeros_like(target_pair),
        )
        coefficient = (
            (2.0 * float(ALPHA) + 1.0)
            * float(MACROSTEP_SCHEDULE_INTEGRAL)
            / (float(ALPHA) * float(GRID_SPACING) ** 2)
        )
        exposure = torch.where(
            active,
            coefficient
            * inputs.duration.to(device=device, dtype=torch.float64)[:, None]
            / torch.where(active, pair, torch.ones_like(pair)),
            torch.zeros_like(pair),
        ) / float(self.microsteps)
        positive_exposure = exposure > 0.0
        equal = active & target_active & (current == wanted)
        needs_move = active & target_active & positive_exposure & ~equal
        current_interior = (current > 0.0) & (current < 1.0)
        target_interior = (wanted > 0.0) & (wanted < 1.0)
        movable = needs_move & current_interior & target_interior
        unreachable = needs_move & ~movable
        safe_current = torch.where(movable, current, torch.full_like(current, 0.5))
        safe_wanted = torch.where(movable, wanted, torch.full_like(wanted, 0.5))
        safe_exposure = torch.where(movable, exposure, torch.ones_like(exposure))
        candidate = (
            torch.log(safe_wanted)
            - torch.log1p(-safe_wanted)
            - torch.log(safe_current)
            + torch.log1p(-safe_current)
        ) / (2.0 * safe_exposure)
        score = torch.where(movable, candidate, torch.zeros_like(candidate))
        self._last_deferred_masks = {  # type: ignore[attr-defined]
            "movable": movable,
            "already_equal": equal,
            "zero_pair_mass": ~active,
            "zero_duration": active & ~positive_exposure,
            "target_oracle_unreachable_boundary": unreachable,
        }
        return score

    def record(self) -> dict[str, Any]:
        return {
            "call_count": self._call_count,
            "lane_count": self._lane_count,
            "movable_count": self._movable_count,
            "already_equal_count": self._already_equal_count,
            "zero_pair_mass_count": self._zero_pair_mass_count,
            "zero_duration_count": self._zero_duration_count,
            "target_oracle_unreachable_boundary_count": (
                self._unreachable_boundary_count
            ),
            "clipping_count": 0,
            "floor_count": 0,
            "projection_count": 0,
        }


def target_oracle_identity_control(*, microsteps: int = 2) -> dict[str, Any]:
    """Exercise one exact oracle control flow with an identity reference.

    Interior fractions are pulled to their fixed targets.  A second fixture
    proves that an exact boundary requiring movement is reported and remains
    unchanged rather than being clipped or floored.
    """

    if int(microsteps) not in {2, 4, 8}:
        raise TangentRolloutContractError("identity control microsteps are invalid")
    tails_all, heads_all = matching_indices(device="cpu")
    tails, heads = tails_all[0], heads_all[0]
    pair_mass = torch.full((EDGES_PER_PHASE,), 1.0 / EDGES_PER_PHASE, dtype=torch.float64)
    current_fraction = torch.full((EDGES_PER_PHASE,), 0.35, dtype=torch.float64)
    target_fraction = torch.full((EDGES_PER_PHASE,), 0.65, dtype=torch.float64)
    state = torch.empty(STATE_SIZE, dtype=torch.float64)
    target = torch.empty_like(state)
    state[tails] = pair_mass * (1.0 - current_fraction)
    state[heads] = pair_mass * current_fraction
    target[tails] = pair_mass * (1.0 - target_fraction)
    target[heads] = pair_mass * target_fraction
    oracle = TargetFractionOracleController(target.numpy(), int(microsteps))
    identity_calls: list[dict[str, Any]] = []

    def identity_reference(**kwargs: Any) -> dict[str, Tensor]:
        head_fraction = kwargs.get("head_fraction")
        transition_ids = kwargs.get("transition_ids")
        role = kwargs.get("role")
        if not isinstance(head_fraction, Tensor) or not isinstance(
            transition_ids, Tensor
        ):
            raise TangentRolloutContractError(
                "identity reference received malformed transition inputs"
            )
        identity_calls.append(
            {
                "role": str(role),
                "transition_ids": tuple(
                    int(value)
                    for value in transition_ids.detach().cpu().reshape(-1).tolist()
                ),
            }
        )
        return {"later_head_fraction": head_fraction.clone()}

    result = controlled_reverse_phase_tangent(
        state.unsqueeze(0),
        0,
        0,
        int(microsteps),
        NAMESPACE_VERSION,
        controller=oracle,
        reference_transition=identity_reference,
        path_ids=(0,),
        label=3,
    )
    output = result.state.reshape(1, STATE_SIZE).to(dtype=torch.float64)
    observed = output[:, heads] / (output[:, tails] + output[:, heads])
    maximum_interior_error = float(
        torch.max(torch.abs(observed - target_fraction[None, :])).item()
    )

    boundary_state = state.clone()
    first_tail, first_head = int(tails[0]), int(heads[0])
    boundary_state[first_tail] += boundary_state[first_head]
    boundary_state[first_head] = 0.0
    boundary_probe = TargetFractionOracleController(target.numpy(), int(microsteps))
    boundary_probe_inputs = ModelInputs(
        later_full_state=boundary_state.to(dtype=torch.float32).unsqueeze(0),
        reverse_time=torch.full((1,), 0.5, dtype=torch.float64),
        phase=torch.zeros(1, dtype=torch.long),
        color=torch.zeros(1, dtype=torch.long),
        duration=torch.full((1,), PHASE_DURATIONS[0], dtype=torch.float32),
        label=torch.full((1,), 3, dtype=torch.long),
    )
    boundary_probe_score = boundary_probe.score_prediction(boundary_probe_inputs)
    boundary_score_zero = bool(boundary_probe_score[0, 0].item() == 0.0)
    boundary_oracle = TargetFractionOracleController(target.numpy(), int(microsteps))
    boundary_calls: list[dict[str, Any]] = []

    def boundary_identity_reference(**kwargs: Any) -> dict[str, Tensor]:
        head_fraction = kwargs.get("head_fraction")
        transition_ids = kwargs.get("transition_ids")
        if not isinstance(head_fraction, Tensor) or not isinstance(
            transition_ids, Tensor
        ):
            raise TangentRolloutContractError(
                "boundary identity reference received malformed transition inputs"
            )
        boundary_calls.append(
            {
                "role": str(kwargs.get("role")),
                "transition_ids": tuple(
                    int(value)
                    for value in transition_ids.detach().cpu().reshape(-1).tolist()
                ),
            }
        )
        return {"later_head_fraction": head_fraction.clone()}

    boundary_result = controlled_reverse_phase_tangent(
        boundary_state.unsqueeze(0),
        0,
        0,
        int(microsteps),
        NAMESPACE_VERSION,
        controller=boundary_oracle,
        reference_transition=boundary_identity_reference,
        path_ids=(0,),
        label=3,
    )
    boundary_output = boundary_result.state.reshape(1, STATE_SIZE).to(dtype=torch.float64)
    boundary_unchanged = bool(
        boundary_output[0, first_head].item()
        == boundary_state[first_head].item()
    )
    boundary_record = boundary_oracle.record()
    expected_roles = tuple(
        f"reverse_reference_{side}_control_M{int(microsteps)}"
        for _ in range(int(microsteps))
        for side in ("pre", "post")
    )
    expected_calls: list[dict[str, Any]] = []
    for reverse_index in range(int(microsteps)):
        for side in ("pre", "post"):
            role = f"reverse_reference_{side}_control_M{int(microsteps)}"
            expected_ids = controller_transition_ids(
                (0,),
                outer_step=0,
                phase=0,
                reverse_microstep=reverse_index,
                role=role,
                device="cpu",
            )
            expected_calls.append(
                {
                    "role": role,
                    "transition_ids": tuple(
                        int(value)
                        for value in expected_ids.reshape(-1).tolist()
                    ),
                }
            )
    observed_roles = tuple(item["role"] for item in identity_calls)
    identity_ids_unique_per_call = all(
        len(item["transition_ids"]) == EDGES_PER_PHASE
        and len(set(item["transition_ids"])) == EDGES_PER_PHASE
        for item in identity_calls
    )
    all_identity_ids = tuple(
        value for item in identity_calls for value in item["transition_ids"]
    )
    identity_ids_globally_unique = (
        len(set(all_identity_ids)) == 2 * int(microsteps) * EDGES_PER_PHASE
    )
    reference_sequence_valid = bool(
        identity_calls == expected_calls
        and boundary_calls == expected_calls
        and observed_roles == expected_roles
        and identity_ids_unique_per_call
        and identity_ids_globally_unique
    )
    passed = int(
        maximum_interior_error <= ORACLE_FRACTION_TOLERANCE
        and boundary_score_zero
        and boundary_unchanged
        and int(boundary_record["target_oracle_unreachable_boundary_count"]) >= 1
        and int(boundary_record["clipping_count"]) == 0
        and int(boundary_record["floor_count"]) == 0
        and int(boundary_record["projection_count"]) == 0
        and reference_sequence_valid
    )
    return _semantic_record(
        {
            "schema": TANGENT_ROLLOUT_VERSION + "-target-oracle-identity-control",
            "schema_version": 1,
            "identity_reference": 1,
            "microsteps": int(microsteps),
            "interior_lane_count": EDGES_PER_PHASE,
            "maximum_interior_fraction_error": maximum_interior_error,
            "maximum_interior_fraction_error_threshold": (
                ORACLE_FRACTION_TOLERANCE
            ),
            "boundary_score_zero": int(boundary_score_zero),
            "boundary_fraction_unchanged": int(boundary_unchanged),
            "reference_call_count": len(identity_calls),
            "reference_roles": list(observed_roles),
            "reference_sequence_valid": int(reference_sequence_valid),
            "identity_transition_ids_unique_per_call": int(
                identity_ids_unique_per_call
            ),
            "identity_transition_ids_globally_unique": int(
                identity_ids_globally_unique
            ),
            "canonical_transition_ids_valid": int(
                identity_calls == expected_calls and boundary_calls == expected_calls
            ),
            "boundary_record": boundary_record,
            "passed": passed,
        }
    )


def exploratory_reference_rng_key(
    root_seed: int, stream_role: str, role: str
) -> tuple[int, str, str, str]:
    """Return the paired exact-reference key; no variant can enter this API."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)):
        raise TypeError("root seed must be an integer")
    if not isinstance(stream_role, str) or not stream_role:
        raise TangentRolloutContractError("stream role must be nonempty text")
    if not isinstance(role, str) or not role:
        raise TangentRolloutContractError("transition role must be nonempty text")
    return (
        int(root_seed),
        EXPLORATORY_REFERENCE_RNG_NAMESPACE,
        stream_role,
        role,
    )


class CertifiedExploratoryReference:
    """Exact certified CUDA reference adapter with paired RNG provenance."""

    def __init__(
        self,
        *,
        profile: JacobiRBCudaProfile,
        root_seed: int,
        stream_role: str,
        sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
        lane_cap: int = REFERENCE_LANE_CAP,
    ) -> None:
        if not isinstance(profile, JacobiRBCudaProfile):
            raise TypeError("profile must be a JacobiRBCudaProfile")
        if not callable(sampler):
            raise TypeError("sampler must be callable")
        if not 1 <= int(lane_cap) <= REFERENCE_LANE_CAP:
            raise TangentRolloutContractError("reference lane cap is invalid")
        exploratory_reference_rng_key(root_seed, stream_role, "contract-check")
        self.profile = profile
        self.root_seed = int(root_seed)
        self.stream_role = stream_role
        self.sampler = sampler
        self.lane_cap = int(lane_cap)
        self.transition_count = 0
        self.call_count = 0
        self.certified_count = 0
        self.fallback_count = 0
        self.fallback_seconds = 0.0
        self.elapsed_seconds = 0.0
        self.maximum_transition_count_per_call = 0
        self.maximum_cuda_memory_allocated = 0
        self.total_cuda_memory_bytes = 0
        self.forbidden_counts = {name: 0 for name in _FORBIDDEN_REFERENCE_COUNTS}

    @staticmethod
    def _scalar(value: Any) -> float:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise TangentRolloutContractError("reference diagnostic is not scalar")
            return float(value.detach().cpu().reshape(()).item())
        return float(value)

    def __call__(
        self,
        *,
        head_fraction: Tensor,
        exposure: Tensor,
        transition_ids: Tensor,
        role: str,
    ) -> Any:
        if not all(isinstance(item, Tensor) for item in (head_fraction, exposure, transition_ids)):
            raise TypeError("reference inputs must be tensors")
        if head_fraction.shape != exposure.shape or head_fraction.shape != transition_ids.shape:
            raise TangentRolloutContractError("reference inputs must have identical shapes")
        count = int(head_fraction.numel())
        if count > self.lane_cap:
            raise TangentRolloutContractError("reference launch exceeds lane cap")
        started = time.perf_counter()
        result = self.sampler(
            head_fraction.contiguous(),
            exposure.contiguous(),
            rng_key=exploratory_reference_rng_key(
                self.root_seed, self.stream_role, role
            ),
            transition_ids=transition_ids.contiguous(),
            profile=self.profile,
        )
        certified_mask = getattr(result, "certified_mask", None)
        if not isinstance(certified_mask, Tensor) or certified_mask.numel() != count:
            raise TangentRolloutContractError("reference result lacks a certificate mask")
        certified = int(torch.count_nonzero(certified_mask).detach().cpu().item())
        if certified != count:
            raise TangentRolloutContractError("reference transition was uncertified")
        later = getattr(result, "later_head_fraction", None)
        if (
            not isinstance(later, Tensor)
            or later.shape != head_fraction.shape
            or not bool(torch.isfinite(later).all())
            or bool(torch.any((later < 0.0) | (later > 1.0)))
        ):
            raise TangentRolloutContractError("reference fraction output is invalid")
        diagnostics = getattr(result, "diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            diagnostics = {}
        fallback_mask = getattr(result, "fallback_mask", None)
        fallback = (
            int(torch.count_nonzero(fallback_mask).detach().cpu().item())
            if isinstance(fallback_mask, Tensor)
            else int(self._scalar(diagnostics.get("fallback_count", 0)))
        )
        self.transition_count += count
        self.call_count += 1
        self.certified_count += certified
        self.fallback_count += fallback
        self.fallback_seconds += self._scalar(
            diagnostics.get("arb_fallback_elapsed_seconds", 0.0)
        )
        self.maximum_transition_count_per_call = max(
            self.maximum_transition_count_per_call, count
        )
        for name in self.forbidden_counts:
            self.forbidden_counts[name] += int(self._scalar(diagnostics.get(name, 0)))
        self.elapsed_seconds += time.perf_counter() - started
        if head_fraction.is_cuda:
            self.maximum_cuda_memory_allocated = max(
                self.maximum_cuda_memory_allocated,
                int(torch.cuda.max_memory_allocated(head_fraction.device)),
            )
            self.total_cuda_memory_bytes = int(
                torch.cuda.get_device_properties(head_fraction.device).total_memory
            )
        if any(self.forbidden_counts.values()):
            raise TangentRolloutContractError("reference backend reported a forbidden event")
        return result

    def record(self) -> dict[str, Any]:
        return {
            "schema": TANGENT_ROLLOUT_VERSION + "-certified-reference",
            "root_seed": self.root_seed,
            "rng_namespace": EXPLORATORY_REFERENCE_RNG_NAMESPACE,
            "stream_role": self.stream_role,
            "variant_in_rng_key": 0,
            "transition_count": self.transition_count,
            "call_count": self.call_count,
            "certified_count": self.certified_count,
            "certificate_fraction": (
                self.certified_count / self.transition_count
                if self.transition_count
                else 1.0
            ),
            "fallback_count": self.fallback_count,
            "fallback_fraction": (
                self.fallback_count / self.transition_count
                if self.transition_count
                else 0.0
            ),
            "fallback_seconds": self.fallback_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "transitions_per_second": (
                self.transition_count / self.elapsed_seconds
                if self.elapsed_seconds > 0.0
                else 0.0
            ),
            "fallback_time_fraction": (
                self.fallback_seconds / self.elapsed_seconds
                if self.elapsed_seconds > 0.0
                else 0.0
            ),
            "maximum_transition_count_per_call": self.maximum_transition_count_per_call,
            "maximum_cuda_memory_allocated": self.maximum_cuda_memory_allocated,
            "peak_cuda_memory_bytes": self.maximum_cuda_memory_allocated,
            "total_cuda_memory_bytes": self.total_cuda_memory_bytes,
            "forbidden_counts": dict(self.forbidden_counts),
        }


def reverse_suffix_sequence(anchor_step: int) -> tuple[tuple[int, int], ...]:
    """Return the exact phase-occurrence reverse suffix from ``anchor_step``."""

    if isinstance(anchor_step, bool) or not isinstance(anchor_step, (int, np.integer)):
        raise TypeError("anchor_step must be an integer")
    anchor = int(anchor_step)
    if not 0 <= anchor < OUTER_STEPS:
        raise TangentRolloutContractError("anchor_step lies outside K=512")
    return tuple(
        (step, phase)
        for step in range(anchor, -1, -1)
        for phase in range(PHASE_COUNT - 1, -1, -1)
    )


_TELEMETRY_SUM_FIELDS = (
    "reference_fraction_displacement_squared_sum",
    "reference_fraction_displacement_count",
    "control_fraction_displacement_squared_sum",
    "control_fraction_displacement_count",
    "score_squared_sum",
    "score_count",
    "logistic_shift_squared_sum",
    "logistic_shift_count",
    "boundary_fraction_count",
    "boundary_rejection_count",
    "clipping_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "projection_count",
    "renormalization_count",
    "transition_count",
)
_TELEMETRY_MAX_FIELDS = (
    "reference_fraction_displacement_maximum_absolute",
    "control_fraction_displacement_maximum_absolute",
    "score_maximum_absolute",
    "logistic_shift_maximum_absolute",
    "maximum_pair_mass_error",
    "maximum_simplex_mass_error",
)


@dataclass(frozen=True)
class ReverseShardResult:
    final_state: np.ndarray = field(repr=False, compare=False)
    sequence: tuple[tuple[int, int], ...]
    phase_diagnostics: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    controller_diagnostics: Mapping[str, Any]
    elapsed_seconds: float
    transition_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": TANGENT_ROLLOUT_VERSION + "-reverse-shard-result",
            "sequence": [list(item) for item in self.sequence],
            "phase_diagnostics": [dict(item) for item in self.phase_diagnostics],
            "diagnostics": dict(self.diagnostics),
            "controller_diagnostics": dict(self.controller_diagnostics),
            "elapsed_seconds": self.elapsed_seconds,
            "transition_count": self.transition_count,
            "final_state_sha256": _array_sha256(self.final_state),
        }


def _phase_telemetry(result: Any, *, outer_step: int, phase: int) -> dict[str, Any]:
    record: dict[str, Any] = {"outer_step": int(outer_step), "phase": int(phase)}
    for name in (*_TELEMETRY_SUM_FIELDS, *_TELEMETRY_MAX_FIELDS):
        default: int | float = 0 if name.endswith("count") or name == "transition_count" else 0.0
        value = getattr(result, name, default)
        record[name] = int(value) if isinstance(default, int) else float(value)
    return record


def _aggregate_phase_diagnostics(
    records: Sequence[Mapping[str, Any]], reference_record: Mapping[str, Any]
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for name in _TELEMETRY_SUM_FIELDS:
        values = [item.get(name, 0) for item in records]
        if name.endswith("count") or name == "transition_count":
            aggregate[name] = sum(int(value) for value in values)
        else:
            aggregate[name] = math.fsum(float(value) for value in values)
    for name in _TELEMETRY_MAX_FIELDS:
        aggregate[name] = max((float(item.get(name, 0.0)) for item in records), default=0.0)
    for prefix in ("reference_fraction_displacement", "control_fraction_displacement", "score", "logistic_shift"):
        count = int(aggregate[f"{prefix}_count"])
        squared_sum = float(aggregate[f"{prefix}_squared_sum"])
        aggregate[f"{prefix}_rms"] = math.sqrt(squared_sum / count) if count else 0.0
    reference_rms = float(aggregate["reference_fraction_displacement_rms"])
    aggregate["control_reference_displacement_ratio"] = (
        float(aggregate["control_fraction_displacement_rms"]) / reference_rms
        if reference_rms > 0.0
        else None
    )
    aggregate["reference"] = dict(reference_record)
    return aggregate


def _controller_record(controller: TangentScoreController) -> dict[str, Any]:
    method = getattr(controller, "record", None)
    if not callable(method):
        return {"controller_kind": type(controller).__name__}
    value = method()
    if not isinstance(value, Mapping):
        raise TangentRolloutContractError("controller record is not a mapping")
    result = dict(value)
    result.setdefault("controller_kind", type(controller).__name__)
    return result


def _controller_record_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"controller_kind": after.get("controller_kind", before.get("controller_kind", "recorded"))}
    for name, value in after.items():
        if name == "controller_kind":
            continue
        prior = before.get(name)
        if isinstance(value, bool):
            result[name] = int(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if name.endswith("_count") or name.endswith("_squared_sum"):
                old = float(prior) if isinstance(prior, (int, float)) else 0.0
                difference = float(value) - old
                result[name] = int(difference) if isinstance(value, int) else difference
            elif "maximum" in name or name.endswith("_max"):
                result[name] = float(value)
            else:
                result[name] = value
        else:
            result[name] = value
    # RMS fields are derived from shard-local sums and counts, never by
    # subtracting cumulative RMS values.
    for prefix in ("unscaled_score", "scaled_score"):
        squared = result.get(f"{prefix}_squared_sum")
        count = result.get("score_count")
        if isinstance(squared, (int, float)) and isinstance(count, int):
            result[f"{prefix}_rms"] = math.sqrt(float(squared) / count) if count else 0.0
    return result


def _aggregate_controller_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        return {}
    result: dict[str, Any] = {}
    keys = {key for record in records for key in record}
    for name in sorted(keys):
        values = [record[name] for record in records if name in record]
        if name == "controller_kind":
            unique = {str(value) for value in values}
            result[name] = next(iter(unique)) if len(unique) == 1 else sorted(unique)
        elif name.endswith("_count") or name.endswith("_squared_sum"):
            result[name] = math.fsum(float(value) for value in values)
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                result[name] = int(result[name])
        elif "maximum" in name or name.endswith("_max"):
            result[name] = max(float(value) for value in values)
        elif name.endswith("_rms"):
            continue
        elif all(value == values[0] for value in values):
            result[name] = values[0]
        else:
            result[name] = values[-1]
    for prefix in ("unscaled_score", "scaled_score"):
        squared = result.get(f"{prefix}_squared_sum")
        count = result.get("score_count")
        if isinstance(squared, (int, float)) and isinstance(count, int):
            result[f"{prefix}_rms"] = math.sqrt(float(squared) / count) if count else 0.0
    return result


def aggregate_trajectory_phase_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate mechanism telemetry overall, by outer quartile, and by phase."""

    normalized = tuple(dict(item) for item in records)
    for item in normalized:
        step = item.get("outer_step")
        phase = item.get("phase")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not 0 <= step < OUTER_STEPS
            or isinstance(phase, bool)
            or not isinstance(phase, int)
            or not 0 <= phase < PHASE_COUNT
        ):
            raise TangentRolloutContractError("phase diagnostic coordinate is invalid")
    return {
        "overall": _aggregate_phase_diagnostics(normalized, {}),
        "by_forward_outer_quartile": {
            str(quartile): _aggregate_phase_diagnostics(
                [item for item in normalized if int(item["outer_step"]) // 128 == quartile],
                {},
            )
            for quartile in range(4)
            if any(int(item["outer_step"]) // 128 == quartile for item in normalized)
        },
        "by_phase": {
            str(phase): _aggregate_phase_diagnostics(
                [item for item in normalized if int(item["phase"]) == phase], {}
            )
            for phase in range(PHASE_COUNT)
            if any(int(item["phase"]) == phase for item in normalized)
        },
    }


def run_reverse_shard(
    state: np.ndarray | Tensor,
    sequence: Sequence[tuple[int, int]],
    *,
    controller: TangentScoreController,
    reference_transition: Callable[..., Any],
    path_ids: Sequence[int],
    label: int | Tensor = 3,
    microsteps: int = 2,
    device: torch.device | str | None = None,
) -> ReverseShardResult:
    """Execute at most one eight-outer-step reverse shard without persistence."""

    tensor, _ = _batched_float64_state(state, device=device)
    paths = _path_ids(path_ids, int(tensor.shape[0]))
    if not isinstance(controller, TangentScoreController):
        raise TypeError("controller must implement score_prediction")
    if not callable(reference_transition):
        raise TypeError("reference transition must be callable")
    normalized = tuple((int(step), int(phase)) for step, phase in sequence)
    if not normalized or len(normalized) > REVERSE_SHARD_PHASES:
        raise TangentRolloutContractError("reverse shard must contain 1..56 phases")
    expected_anchor = normalized[0][0]
    expected = reverse_suffix_sequence(expected_anchor)[: len(normalized)]
    if normalized != expected:
        raise TangentRolloutContractError("reverse shard sequence is not contiguous")
    started = time.perf_counter()
    controller_before = _controller_record(controller)
    phase_records: list[dict[str, Any]] = []
    values = tensor
    with torch.inference_mode():
        for outer_step, phase in normalized:
            result = controlled_reverse_phase_tangent(
                values,
                outer_step,
                phase,
                microsteps,
                NAMESPACE_VERSION,
                controller=controller,
                reference_transition=reference_transition,
                path_ids=paths,
                label=label,
            )
            values = result.state
            if values.ndim == 1:
                values = values.unsqueeze(0)
            phase_records.append(
                _phase_telemetry(result, outer_step=outer_step, phase=phase)
            )
    final = np.ascontiguousarray(values.detach().cpu().numpy(), dtype=np.float64)
    reference_record = (
        reference_transition.record()
        if callable(getattr(reference_transition, "record", None))
        else {}
    )
    diagnostics = _aggregate_phase_diagnostics(phase_records, reference_record)
    controller_diagnostics = _controller_record_delta(
        controller_before, _controller_record(controller)
    )
    return ReverseShardResult(
        final_state=final,
        sequence=normalized,
        phase_diagnostics=tuple(phase_records),
        diagnostics=diagnostics,
        controller_diagnostics=controller_diagnostics,
        elapsed_seconds=float(time.perf_counter() - started),
        transition_count=int(diagnostics["transition_count"]),
    )


@dataclass(frozen=True)
class ReverseTrajectoryResult:
    final_state: np.ndarray = field(repr=False, compare=False)
    saved_states: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    diagnostics: Mapping[str, Any]
    elapsed_seconds: float
    transition_count: int
    shard_records: tuple[Mapping[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": TANGENT_ROLLOUT_VERSION + "-reverse-trajectory-result",
            "final_state_sha256": _array_sha256(self.final_state),
            "saved_state_sha256": {
                name: _array_sha256(value) for name, value in self.saved_states.items()
            },
            "diagnostics": dict(self.diagnostics),
            "elapsed_seconds": self.elapsed_seconds,
            "transition_count": self.transition_count,
            "shard_count": len(self.shard_records),
        }


def _safe_stem(value: str) -> str:
    if not isinstance(value, str) or not value or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in value
    ):
        raise TangentRolloutContractError("trajectory name is not a safe artifact stem")
    return value


def _callable_binding(value: Callable[..., Any]) -> str:
    return f"{getattr(value, '__module__', type(value).__module__)}:{getattr(value, '__qualname__', type(value).__qualname__)}"


def _valid_restart_record(
    record_path: Path,
    state_path: Path,
    *,
    binding: Mapping[str, Any],
    rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not record_path.exists():
        # State is written first.  An NPZ without its commit record is an
        # uncommitted orphan from interruption and is safely replaced by the
        # deterministic shard replay.  A JSON record without state, however,
        # falsely claims a commit and must fail closed.
        raise FileNotFoundError(record_path)
    if not state_path.exists():
        raise TangentRolloutContractError("committed restart record lacks its state")
    record = _load_json(record_path)
    _validate_semantic_record(record, "restart shard")
    for name, expected in binding.items():
        if record.get(name) != expected:
            raise TangentRolloutContractError(f"restart binding {name} changed")
    if record.get("state_file_sha256") != _file_sha256(state_path):
        raise TangentRolloutContractError("restart state file hash changed")
    state = _load_state_npz(state_path, expected_rows=rows)
    if record.get("output_state_sha256") != _array_sha256(state):
        raise TangentRolloutContractError("restart state array hash changed")
    return state, record


def run_reverse_trajectory(
    initial_state: np.ndarray | Tensor,
    *,
    anchor_step: int,
    output_dir: str | Path,
    trajectory_name: str,
    controller: TangentScoreController,
    reference_factory: Callable[[int], Callable[..., Any]],
    path_ids: Sequence[int],
    controller_binding: Mapping[str, Any],
    rng_binding: Mapping[str, Any],
    label: int | Tensor = 3,
    microsteps: int = 2,
    device: torch.device | str | None = None,
) -> ReverseTrajectoryResult:
    """Run or resume one exact reverse trajectory in eight-step shards."""

    name = _safe_stem(trajectory_name)
    tensor, squeezed = _batched_float64_state(initial_state, device=device)
    paths = _path_ids(path_ids, int(tensor.shape[0]))
    sequence = reverse_suffix_sequence(anchor_step)
    if len(sequence) % REVERSE_SHARD_PHASES:
        raise TangentRolloutContractError(
            "restartable reverse anchor must end on an eight-step boundary"
        )
    if not callable(reference_factory):
        raise TypeError("reference_factory must be callable")
    root = Path(output_dir) / "reverse_shards" / name
    root.mkdir(parents=True, exist_ok=True)
    controller_hash = semantic_sha256(dict(controller_binding))
    rng_hash = semantic_sha256(dict(rng_binding))
    state = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float64)
    initial_hash = _array_sha256(state)
    saved: dict[str, np.ndarray] = {"start": state.copy()}
    records: list[dict[str, Any]] = []
    elapsed = 0.0
    progress_steps = int(anchor_step) + 1
    capture_counts = {
        progress_steps // 4: "progress_25",
        progress_steps // 2: "progress_50",
        3 * progress_steps // 4: "progress_75",
        progress_steps: "final",
    }
    for shard_index, offset in enumerate(range(0, len(sequence), REVERSE_SHARD_PHASES)):
        shard_sequence = sequence[offset : offset + REVERSE_SHARD_PHASES]
        state_path = root / f"shard-{shard_index:04d}.npz"
        record_path = root / f"shard-{shard_index:04d}.json"
        input_hash = _array_sha256(state)
        binding = {
            "schema": TANGENT_ROLLOUT_VERSION + "-reverse-shard",
            "schema_version": 1,
            "trajectory_name": name,
            "shard_index": shard_index,
            "sequence_start": list(shard_sequence[0]),
            "sequence_end": list(shard_sequence[-1]),
            "sequence_sha256": semantic_sha256([list(item) for item in shard_sequence]),
            "path_ids": list(paths),
            "microsteps": int(microsteps),
            "label": int(label) if not isinstance(label, Tensor) else "tensor",
            "input_state_sha256": input_hash,
            "controller_binding_sha256": controller_hash,
            "rng_binding_sha256": rng_hash,
        }
        try:
            state, record = _valid_restart_record(
                record_path, state_path, binding=binding, rows=len(paths)
            )
        except FileNotFoundError:
            reference = reference_factory(shard_index)
            result = run_reverse_shard(
                torch.as_tensor(
                    np.array(state, copy=True, order="C"),
                    dtype=torch.float64,
                    device=tensor.device,
                ).contiguous(),
                shard_sequence,
                controller=controller,
                reference_transition=reference,
                path_ids=paths,
                label=label,
                microsteps=microsteps,
            )
            state = result.final_state
            _atomic_npz(state_path, {"state": state})
            record = _semantic_record(
                {
                    **binding,
                    "output_state_sha256": _array_sha256(state),
                    "state_file_sha256": _file_sha256(state_path),
                    "state_file_size": int(state_path.stat().st_size),
                    "elapsed_seconds": result.elapsed_seconds,
                    "transition_count": result.transition_count,
                    "phase_diagnostics": [dict(item) for item in result.phase_diagnostics],
                    "diagnostics": dict(result.diagnostics),
                    "controller_diagnostics": dict(result.controller_diagnostics),
                    "committed": 1,
                }
            )
            atomic_write_json(record_path, record)
        records.append(record)
        elapsed += float(record.get("elapsed_seconds", 0.0))
        completed_outer_steps = (shard_index + 1) * REVERSE_SHARD_OUTER_STEPS
        label_name = capture_counts.get(completed_outer_steps)
        if label_name is not None:
            saved[label_name] = state.copy()
    if "final" not in saved:
        saved["final"] = state.copy()
    transition_count = sum(int(item.get("transition_count", 0)) for item in records)
    phase_records = [
        phase_record
        for item in records
        for phase_record in item.get("phase_diagnostics", [])
    ]
    reference_records = [
        dict(item.get("diagnostics", {}).get("reference", {})) for item in records
    ]
    controller_records = [
        dict(item.get("controller_diagnostics", {})) for item in records
    ]
    aggregate = _aggregate_phase_diagnostics(phase_records, {})
    reference_transition_count = sum(
        int(item.get("transition_count", 0)) for item in reference_records
    )
    reference_certified_count = sum(
        int(item.get("certified_count", 0)) for item in reference_records
    )
    reference_fallback_count = sum(
        int(item.get("fallback_count", 0)) for item in reference_records
    )
    reference_fallback_seconds = math.fsum(
        float(item.get("fallback_seconds", 0.0)) for item in reference_records
    )
    reference_elapsed_seconds = math.fsum(
        float(item.get("elapsed_seconds", 0.0)) for item in reference_records
    )
    reference_forbidden = {
        key: sum(
            int(item.get("forbidden_counts", {}).get(key, 0))
            for item in reference_records
        )
        for key in _FORBIDDEN_REFERENCE_COUNTS
    }
    controller_aggregate = _aggregate_controller_diagnostics(controller_records)
    for name in ("clipping_count", "floor_count", "projection_count"):
        aggregate[name] = int(aggregate.get(name, 0)) + int(
            controller_aggregate.get(name, 0)
        )
    aggregate.update(
        initial_state_sha256=initial_hash,
        final_state_sha256=_array_sha256(state),
        shard_count=len(records),
        restart_chain_valid=1,
        reference_transition_count=reference_transition_count,
        reference_certified_count=reference_certified_count,
        reference_fallback_count=reference_fallback_count,
        reference_fallback_seconds=reference_fallback_seconds,
        reference_elapsed_seconds=reference_elapsed_seconds,
        peak_cuda_memory_allocated_bytes=max(
            (
                int(item.get("maximum_cuda_memory_allocated", 0))
                for item in reference_records
            ),
            default=0,
        ),
        peak_cuda_memory_bytes=max(
            (int(item.get("peak_cuda_memory_bytes", 0)) for item in reference_records),
            default=0,
        ),
        total_cuda_memory_bytes=max(
            (int(item.get("total_cuda_memory_bytes", 0)) for item in reference_records),
            default=0,
        ),
        reference_forbidden_counts=reference_forbidden,
        certificate_fraction=(
            reference_certified_count / reference_transition_count
            if reference_transition_count
            else 1.0
        ),
        fallback_fraction=(
            reference_fallback_count / reference_transition_count
            if reference_transition_count
            else 0.0
        ),
        fallback_time_fraction=(
            reference_fallback_seconds / reference_elapsed_seconds
            if reference_elapsed_seconds > 0.0
            else 0.0
        ),
        controller=controller_aggregate,
        target_oracle_unreachable_boundary_count=int(
            controller_aggregate.get("target_oracle_unreachable_boundary_count", 0)
        ),
        phase_aggregation=aggregate_trajectory_phase_diagnostics(phase_records),
    )
    aggregate["maximum_mass_error"] = max(
        float(aggregate["maximum_pair_mass_error"]),
        float(aggregate["maximum_simplex_mass_error"]),
    )
    aggregate["maximum_global_mass_error"] = float(
        aggregate["maximum_simplex_mass_error"]
    )
    aggregate["forbidden_event_count"] = sum(reference_forbidden.values()) + sum(
        int(aggregate.get(name, 0))
        for name in (
            "boundary_rejection_count",
            "clipping_count",
            "correction_count",
            "floor_count",
            "limiter_count",
            "projection_count",
            "renormalization_count",
        )
    )
    aggregate["elapsed_seconds"] = elapsed
    aggregate["transitions_per_second"] = (
        transition_count / elapsed if elapsed > 0.0 else 0.0
    )
    public_final = state[0] if squeezed else state
    public_saved = {
        key: (value[0] if squeezed else value) for key, value in saved.items()
    }
    return ReverseTrajectoryResult(
        final_state=np.ascontiguousarray(public_final),
        saved_states={
            key: np.ascontiguousarray(value) for key, value in public_saved.items()
        },
        diagnostics=aggregate,
        elapsed_seconds=elapsed,
        transition_count=transition_count,
        shard_records=tuple(records),
    )


def aggregate_exact_forward_shards(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_shard_count: int,
    expected_transition_count: int,
    expected_path_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Reconstruct exact health only from required committed shard fields.

    Missing or malformed evidence is an implementation-contract failure.
    Well-formed evidence that reports an invalid chain, authorization count,
    numerical state, conservation result, or forbidden event is a numerical-
    integrity failure.  This distinction lets an experiment CLI adjudicate a
    broken summary adapter separately from a failed exact transition.
    """

    def contract(message: str) -> None:
        raise ExactForwardShardAggregateError(
            message, failure_domain="implementation_contract"
        )

    def numerical(message: str) -> None:
        raise ExactForwardShardAggregateError(
            message, failure_domain="numerical_integrity"
        )

    def required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
        if key not in mapping:
            contract(f"{where} is missing required field {key}")
        return mapping[key]

    def exact_int(value: Any, where: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            contract(f"{where} must be an integer")
        return int(value)

    def finite_float(value: Any, where: str) -> float:
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            contract(f"{where} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            numerical(f"{where} is nonfinite")
        return result

    def digest(value: Any, where: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            contract(f"{where} must be a lowercase SHA-256 digest")
        return value

    if isinstance(expected_shard_count, bool) or not isinstance(
        expected_shard_count, (int, np.integer)
    ):
        raise TypeError("expected_shard_count must be an integer")
    if isinstance(expected_transition_count, bool) or not isinstance(
        expected_transition_count, (int, np.integer)
    ):
        raise TypeError("expected_transition_count must be an integer")
    shard_count_expected = int(expected_shard_count)
    transition_count_expected = int(expected_transition_count)
    if shard_count_expected <= 0 or transition_count_expected <= 0:
        raise ValueError("expected exact-forward counts must be positive")
    if type(expected_path_ids) is not tuple or not expected_path_ids:
        raise TypeError("expected_path_ids must be a nonempty tuple")
    paths_expected: tuple[int, ...] = tuple(
        exact_int(value, "expected_path_ids") for value in expected_path_ids
    )
    if len(set(paths_expected)) != len(paths_expected) or any(
        value < 0 or value >= (1 << 20) for value in paths_expected
    ):
        raise ValueError("expected_path_ids violate the canonical path contract")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    if len(records) != shard_count_expected:
        contract("exact-forward shard count changed")

    total_transitions = 0
    total_certified = 0
    total_uncertified = 0
    total_fallbacks = 0
    elapsed_parts: list[float] = []
    fallback_elapsed_parts: list[float] = []
    maximum_pair_mass_error = 0.0
    maximum_simplex_mass_error = 0.0
    maximum_peak_memory = 0
    total_memory_values: set[int] = set()
    merged_histogram: dict[int, int] = {}
    forbidden_counts = {name: 0 for name in EXACT_FORWARD_FORBIDDEN_COUNTERS}
    previous_output_sha256: str | None = None
    first_input_sha256: str | None = None
    final_output_sha256: str | None = None
    output_health_presence: bool | None = None
    output_nonfinite_count = 0
    output_negative_count = 0
    maximum_output_mass_error = 0.0

    for expected_index, raw_record in enumerate(records):
        where = f"forward shard {expected_index}"
        if not isinstance(raw_record, Mapping):
            contract(f"{where} must be a mapping")
        record = raw_record
        schema = required(record, "schema", where)
        schema_version = exact_int(
            required(record, "schema_version", where), f"{where}.schema_version"
        )
        if schema != TANGENT_ROLLOUT_VERSION + "-forward-shard" or schema_version not in {
            1,
            2,
        }:
            contract(f"{where} schema is incompatible")
        committed = exact_int(required(record, "committed", where), f"{where}.committed")
        if committed != 1:
            contract(f"{where} is not committed")
        shard_index = exact_int(
            required(record, "shard_index", where), f"{where}.shard_index"
        )
        start_step = exact_int(
            required(record, "start_step", where), f"{where}.start_step"
        )
        step_count = exact_int(
            required(record, "step_count", where), f"{where}.step_count"
        )
        if (
            shard_index != expected_index
            or start_step != expected_index * SHARD_STEPS
            or step_count != SHARD_STEPS
        ):
            numerical(f"{where} coordinate sequence is invalid")

        raw_paths = required(record, "path_ids", where)
        if not isinstance(raw_paths, (list, tuple)):
            contract(f"{where}.path_ids must be a sequence")
        paths = tuple(exact_int(item, f"{where}.path_ids") for item in raw_paths)
        if paths != paths_expected:
            numerical(f"{where} path IDs changed")

        input_sha256 = digest(
            required(record, "input_state_sha256", where),
            f"{where}.input_state_sha256",
        )
        output_sha256 = digest(
            required(record, "output_state_sha256", where),
            f"{where}.output_state_sha256",
        )
        digest(
            required(record, "state_file_sha256", where),
            f"{where}.state_file_sha256",
        )
        if expected_index == 0:
            first_input_sha256 = input_sha256
        elif input_sha256 != previous_output_sha256:
            numerical(f"{where} input/output state hash chain is broken")
        previous_output_sha256 = output_sha256
        final_output_sha256 = output_sha256

        elapsed = finite_float(
            required(record, "elapsed_seconds", where), f"{where}.elapsed_seconds"
        )
        if elapsed < 0.0:
            numerical(f"{where}.elapsed_seconds is negative")
        elapsed_parts.append(elapsed)
        transition_count = exact_int(
            required(record, "transition_count", where),
            f"{where}.transition_count",
        )
        if transition_count < 0:
            numerical(f"{where}.transition_count is negative")
        pair_error = finite_float(
            required(record, "maximum_pair_mass_error", where),
            f"{where}.maximum_pair_mass_error",
        )
        if pair_error < 0.0 or pair_error > SIMPLEX_TOLERANCE:
            numerical(f"{where} pair-mass conservation failed")
        maximum_pair_mass_error = max(maximum_pair_mass_error, pair_error)

        peak_memory = exact_int(
            required(record, "peak_cuda_memory_allocated_bytes", where),
            f"{where}.peak_cuda_memory_allocated_bytes",
        )
        total_memory = exact_int(
            required(record, "total_cuda_memory_bytes", where),
            f"{where}.total_cuda_memory_bytes",
        )
        if peak_memory < 0 or total_memory < 0 or (
            total_memory == 0 and peak_memory != 0
        ) or (total_memory > 0 and peak_memory > total_memory):
            numerical(f"{where} CUDA memory telemetry is invalid")
        maximum_peak_memory = max(maximum_peak_memory, peak_memory)
        total_memory_values.add(total_memory)

        scheduler = required(record, "scheduler_record", where)
        if not isinstance(scheduler, Mapping):
            contract(f"{where}.scheduler_record must be a mapping")
        diagnostics = required(scheduler, "diagnostics", f"{where}.scheduler_record")
        if not isinstance(diagnostics, Mapping):
            contract(f"{where}.scheduler_record.diagnostics must be a mapping")
        scheduler_transitions = exact_int(
            required(diagnostics, "transition_count", f"{where}.diagnostics"),
            f"{where}.diagnostics.transition_count",
        )
        certified = exact_int(
            required(diagnostics, "certified_count", f"{where}.diagnostics"),
            f"{where}.diagnostics.certified_count",
        )
        uncertified = exact_int(
            required(diagnostics, "uncertified_count", f"{where}.diagnostics"),
            f"{where}.diagnostics.uncertified_count",
        )
        fallback = exact_int(
            required(diagnostics, "fallback_count", f"{where}.diagnostics"),
            f"{where}.diagnostics.fallback_count",
        )
        fallback_elapsed = finite_float(
            required(
                diagnostics, "fallback_elapsed_seconds", f"{where}.diagnostics"
            ),
            f"{where}.diagnostics.fallback_elapsed_seconds",
        )
        simplex_error = finite_float(
            required(diagnostics, "maximum_mass_error", f"{where}.diagnostics"),
            f"{where}.diagnostics.maximum_mass_error",
        )
        if min(scheduler_transitions, certified, uncertified, fallback) < 0:
            numerical(f"{where} scheduler counts are negative")
        if fallback_elapsed < 0.0 or fallback_elapsed > elapsed:
            numerical(f"{where} fallback timing is invalid")
        if simplex_error < 0.0 or simplex_error > SIMPLEX_TOLERANCE:
            numerical(f"{where} simplex conservation failed")

        raw_histogram = required(
            diagnostics, "certificate_code_counts", f"{where}.diagnostics"
        )
        if not isinstance(raw_histogram, Mapping) or not raw_histogram:
            contract(f"{where}.diagnostics.certificate_code_counts is malformed")
        histogram: dict[int, int] = {}
        for raw_code, raw_count in raw_histogram.items():
            if isinstance(raw_code, bool):
                contract(f"{where} certificate code is malformed")
            try:
                code = int(raw_code)
            except (TypeError, ValueError) as exc:
                raise ExactForwardShardAggregateError(
                    f"{where} certificate code is malformed",
                    failure_domain="implementation_contract",
                ) from exc
            if str(code) != str(raw_code):
                contract(f"{where} certificate code is not canonical")
            count = exact_int(raw_count, f"{where}.certificate_code_counts[{code}]")
            if code < 0 or code > 255 or count < 0 or code in histogram:
                numerical(f"{where} certificate histogram is invalid")
            if code != 0 and code & 0xF != 0xF:
                numerical(f"{where} has an unauthorized certificate code {code}")
            histogram[code] = count
        active = sum(
            count for code, count in histogram.items() if code != 0 and code & 0xF == 0xF
        )
        structural_noop = histogram.get(0, 0)
        if (
            sum(histogram.values()) != transition_count
            or scheduler_transitions != transition_count
            or certified != active
            or uncertified != structural_noop
            or active + structural_noop != transition_count
            or certified + structural_noop != transition_count
            or fallback > active
        ):
            numerical(f"{where} authorization/count identities failed")

        for name in EXACT_FORWARD_FORBIDDEN_COUNTERS:
            count = exact_int(
                required(diagnostics, name, f"{where}.diagnostics"),
                f"{where}.diagnostics.{name}",
            )
            if count < 0:
                numerical(f"{where}.{name} is negative")
            forbidden_counts[name] += count
            if count != 0:
                numerical(f"{where} records forbidden exact-forward events")

        output_health_names = (
            "output_state_nonfinite_count",
            "output_state_negative_count",
            "maximum_output_state_mass_error",
        )
        present = tuple(name in record for name in output_health_names)
        if any(present) and not all(present):
            contract(f"{where} has a partial output-state health record")
        shard_has_output_health = all(present)
        if schema_version >= 2 and not shard_has_output_health:
            contract(f"{where} v2 record omits output-state health")
        if output_health_presence is None:
            output_health_presence = shard_has_output_health
        elif output_health_presence != shard_has_output_health:
            contract("exact-forward shards mix output-state health schemas")
        if shard_has_output_health:
            nonfinite_count = exact_int(
                record["output_state_nonfinite_count"],
                f"{where}.output_state_nonfinite_count",
            )
            negative_count = exact_int(
                record["output_state_negative_count"],
                f"{where}.output_state_negative_count",
            )
            output_mass_error = finite_float(
                record["maximum_output_state_mass_error"],
                f"{where}.maximum_output_state_mass_error",
            )
            if (
                nonfinite_count != 0
                or negative_count != 0
                or output_mass_error < 0.0
                or output_mass_error > SIMPLEX_TOLERANCE
            ):
                numerical(f"{where} output-state numerical health failed")
            output_nonfinite_count += nonfinite_count
            output_negative_count += negative_count
            maximum_output_mass_error = max(
                maximum_output_mass_error, output_mass_error
            )

        total_transitions += transition_count
        total_certified += certified
        total_uncertified += uncertified
        total_fallbacks += fallback
        fallback_elapsed_parts.append(fallback_elapsed)
        maximum_simplex_mass_error = max(maximum_simplex_mass_error, simplex_error)
        for code, count in histogram.items():
            merged_histogram[code] = merged_histogram.get(code, 0) + count

    if total_transitions != transition_count_expected:
        numerical("exact-forward aggregate transition count changed")
    if len(total_memory_values) != 1:
        numerical("exact-forward total CUDA memory telemetry changed between shards")
    active_count = sum(
        count
        for code, count in merged_histogram.items()
        if code != 0 and code & 0xF == 0xF
    )
    structural_noop_count = merged_histogram.get(0, 0)
    authorized_count = total_certified + structural_noop_count
    if (
        sum(merged_histogram.values()) != total_transitions
        or total_certified != active_count
        or total_uncertified != structural_noop_count
        or active_count + structural_noop_count != total_transitions
        or authorized_count != total_transitions
    ):
        numerical("exact-forward aggregate authorization identities failed")
    elapsed_seconds = math.fsum(elapsed_parts)
    fallback_elapsed_seconds = math.fsum(fallback_elapsed_parts)
    if elapsed_seconds <= 0.0 or fallback_elapsed_seconds > elapsed_seconds:
        numerical("exact-forward aggregate timing is invalid")
    total_cuda_memory_bytes = next(iter(total_memory_values))
    return _semantic_record(
        {
            "schema": TANGENT_ROLLOUT_VERSION + "-exact-forward-shard-aggregate",
            "schema_version": 1,
            "passed": 1,
            "restart_chain_valid": 1,
            "expected_shard_count": shard_count_expected,
            "shard_count": len(records),
            "expected_transition_count": transition_count_expected,
            "transition_count": total_transitions,
            "path_ids": list(paths_expected),
            "first_input_state_sha256": first_input_sha256,
            "final_output_state_sha256": final_output_sha256,
            "certificate_code_counts": {
                str(code): merged_histogram[code] for code in sorted(merged_histogram)
            },
            "active_count": active_count,
            "structural_noop_count": structural_noop_count,
            "authorized_count": authorized_count,
            "certified_count": total_certified,
            "uncertified_count": total_uncertified,
            "authorization_fraction": 1.0,
            "certificate_fraction": 1.0,
            "authorization_semantics": (
                "active-lanes-certified-plus-exact-structural-noops-v1"
            ),
            "certificate_code_authorization_semantics": (
                "active-low-nibble-0xf-plus-code-zero-structural-noop-v1"
            ),
            "fallback_count": total_fallbacks,
            "fallback_fraction": (
                total_fallbacks / active_count if active_count else 0.0
            ),
            "fallback_elapsed_seconds": fallback_elapsed_seconds,
            "fallback_seconds": fallback_elapsed_seconds,
            "fallback_time_fraction": fallback_elapsed_seconds / elapsed_seconds,
            "forbidden_counter_schema": EXACT_FORWARD_FORBIDDEN_COUNTERS_VERSION,
            "forbidden_counter_names": list(EXACT_FORWARD_FORBIDDEN_COUNTERS),
            "forbidden_counts": forbidden_counts,
            "forbidden_event_count": sum(forbidden_counts.values()),
            "maximum_pair_mass_error": maximum_pair_mass_error,
            "maximum_simplex_mass_error": maximum_simplex_mass_error,
            "maximum_mass_error": maximum_simplex_mass_error,
            "maximum_global_mass_error": maximum_simplex_mass_error,
            "output_state_health_recorded": int(bool(output_health_presence)),
            "output_state_nonfinite_count": (
                output_nonfinite_count if output_health_presence else None
            ),
            "output_state_negative_count": (
                output_negative_count if output_health_presence else None
            ),
            "maximum_output_state_mass_error": (
                maximum_output_mass_error if output_health_presence else None
            ),
            "elapsed_seconds": elapsed_seconds,
            "transitions_per_second": total_transitions / elapsed_seconds,
            "peak_cuda_memory_allocated_bytes": maximum_peak_memory,
            "peak_cuda_memory_bytes": maximum_peak_memory,
            "total_cuda_memory_bytes": total_cuda_memory_bytes,
        }
    )


@dataclass(frozen=True)
class ForwardTrajectoryResult:
    final_state: np.ndarray = field(repr=False, compare=False)
    initial_state: np.ndarray = field(repr=False, compare=False)
    anchors: Mapping[int, np.ndarray] = field(repr=False, compare=False)
    diagnostics: Mapping[str, Any]
    elapsed_seconds: float
    transition_count: int
    shard_records: tuple[Mapping[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": TANGENT_ROLLOUT_VERSION + "-forward-trajectory-result",
            "initial_state_sha256": _array_sha256(self.initial_state),
            "final_state_sha256": _array_sha256(self.final_state),
            "anchor_sha256": {
                str(step): _array_sha256(value) for step, value in self.anchors.items()
            },
            "diagnostics": dict(self.diagnostics),
            "elapsed_seconds": self.elapsed_seconds,
            "transition_count": self.transition_count,
            "shard_count": len(self.shard_records),
        }


def run_forward_trajectory(
    initial_state: np.ndarray | Tensor,
    *,
    anchor_steps: Sequence[int],
    output_dir: str | Path,
    trajectory_name: str,
    path_ids: Sequence[int],
    root_seed: int,
    profile: JacobiRBCudaProfile,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
    step_limit: int = OUTER_STEPS,
    device: torch.device | str | None = None,
) -> ForwardTrajectoryResult:
    """Run or resume the exact forward split chain in eight-step shards."""

    name = _safe_stem(trajectory_name)
    tensor, squeezed = _batched_float64_state(initial_state, device=device)
    paths = _path_ids(path_ids, int(tensor.shape[0]))
    if not isinstance(profile, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")
    limit = int(step_limit)
    if not SHARD_STEPS <= limit <= OUTER_STEPS or limit % SHARD_STEPS:
        raise TangentRolloutContractError("forward step_limit must be a multiple of eight")
    anchors_requested = tuple(sorted(set(int(item) for item in anchor_steps)))
    if any(item < 0 or item >= limit or (item + 1) % SHARD_STEPS for item in anchors_requested):
        raise TangentRolloutContractError("forward anchors must end eight-step shards")
    root = Path(output_dir) / "forward_shards" / name
    root.mkdir(parents=True, exist_ok=True)
    state = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float64)
    initial = state.copy()
    anchors: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    profile_hash = semantic_sha256(profile.to_dict())
    sampler_binding = _callable_binding(sampler)
    elapsed = 0.0
    for shard_index, start_step in enumerate(range(0, limit, SHARD_STEPS)):
        state_path = root / f"shard-{shard_index:04d}.npz"
        record_path = root / f"shard-{shard_index:04d}.json"
        binding = {
            "schema": TANGENT_ROLLOUT_VERSION + "-forward-shard",
            "schema_version": 2,
            "trajectory_name": name,
            "shard_index": shard_index,
            "start_step": start_step,
            "step_count": SHARD_STEPS,
            "path_ids": list(paths),
            "root_seed": int(root_seed),
            "profile_sha256": profile_hash,
            "sampler_binding": sampler_binding,
            "input_state_sha256": _array_sha256(state),
        }
        try:
            state, record = _valid_restart_record(
                record_path, state_path, binding=binding, rows=len(paths)
            )
        except FileNotFoundError:
            started = time.perf_counter()
            input_state = np.array(state, copy=True, order="C")
            result = run_exact_multipath_shard(
                torch.as_tensor(
                    input_state,
                    dtype=torch.float64,
                    device=tensor.device,
                ).contiguous(),
                path_ids=paths,
                start_step=start_step,
                step_count=SHARD_STEPS,
                root_seed=int(root_seed),
                profile=profile,
                group_sizes=(len(paths),),
                sampler=sampler,
                capture_training_payload=True,
            )
            payload = result.capture_payload
            if payload is None:
                raise TangentRolloutContractError(
                    "forward scheduler omitted the phase-state health trace"
                )
            canonical_order = np.argsort(np.asarray(paths, dtype=np.int64))
            previous = input_state[canonical_order]
            maximum_pair_mass_error = 0.0
            tails_all, heads_all = matching_indices(device="cpu")
            tails_np = tails_all.numpy()
            heads_np = heads_all.numpy()
            for block, post in enumerate(payload.post_phase_states):
                phase_index = block % PHASE_COUNT
                color = PHASE_MATCHINGS[phase_index]
                before_pair = previous[:, tails_np[color]] + previous[:, heads_np[color]]
                after_pair = post[:, tails_np[color]] + post[:, heads_np[color]]
                maximum_pair_mass_error = max(
                    maximum_pair_mass_error,
                    float(np.max(np.abs(after_pair - before_pair))),
                )
                previous = post
            state = np.ascontiguousarray(result.committed_final_states, dtype=np.float64)
            shard_elapsed = float(time.perf_counter() - started)
            peak_cuda_memory = (
                int(torch.cuda.max_memory_allocated(tensor.device))
                if tensor.is_cuda
                else 0
            )
            total_cuda_memory = (
                int(torch.cuda.get_device_properties(tensor.device).total_memory)
                if tensor.is_cuda
                else 0
            )
            _atomic_npz(state_path, {"state": state})
            try:
                with np.load(state_path, allow_pickle=False) as archive:
                    if set(archive.files) != {"state"}:
                        raise TangentRolloutContractError(
                            "committed forward state archive schema changed"
                        )
                    persisted_state = np.array(
                        archive["state"], dtype=np.float64, copy=True, order="C"
                    )
            except (OSError, ValueError, KeyError) as exc:
                raise TangentRolloutContractError(
                    "cannot verify committed forward state archive"
                ) from exc
            if persisted_state.shape != state.shape:
                raise TangentRolloutContractError(
                    "committed forward state archive shape changed"
                )
            output_state_nonfinite_count = int(
                np.count_nonzero(~np.isfinite(persisted_state))
            )
            output_state_negative_count = int(
                np.count_nonzero(np.isfinite(persisted_state) & (persisted_state < 0.0))
            )
            maximum_output_state_mass_error = (
                float(
                    np.max(
                        np.abs(np.sum(persisted_state, axis=1, dtype=np.float64) - 1.0)
                    )
                )
                if output_state_nonfinite_count == 0
                else 0.0
            )
            record = _semantic_record(
                {
                    **binding,
                    "output_state_sha256": _array_sha256(persisted_state),
                    "state_file_sha256": _file_sha256(state_path),
                    "state_file_size": int(state_path.stat().st_size),
                    "elapsed_seconds": shard_elapsed,
                    "transition_count": int(
                        result.diagnostics.get(
                            "transition_count", len(paths) * SHARD_STEPS * PHASE_COUNT * EDGES_PER_PHASE
                        )
                    ),
                    "maximum_pair_mass_error": maximum_pair_mass_error,
                    "peak_cuda_memory_allocated_bytes": peak_cuda_memory,
                    "peak_cuda_memory_bytes": peak_cuda_memory,
                    "total_cuda_memory_bytes": total_cuda_memory,
                    "output_state_nonfinite_count": output_state_nonfinite_count,
                    "output_state_negative_count": output_state_negative_count,
                    "maximum_output_state_mass_error": maximum_output_state_mass_error,
                    "scheduler_record": result.to_record(),
                    "committed": 1,
                }
            )
            atomic_write_json(record_path, record)
        records.append(record)
        elapsed += float(record.get("elapsed_seconds", 0.0))
        completed_step = start_step + SHARD_STEPS - 1
        if completed_step in anchors_requested:
            anchors[completed_step] = state.copy()
    if set(anchors) != set(anchors_requested):
        raise TangentRolloutContractError("forward trajectory did not commit all anchors")
    diagnostics = aggregate_exact_forward_shards(
        records,
        expected_shard_count=limit // SHARD_STEPS,
        expected_transition_count=(
            len(paths) * limit * PHASE_COUNT * EDGES_PER_PHASE
        ),
        expected_path_ids=paths,
    )
    elapsed = float(diagnostics["elapsed_seconds"])
    transition_count = int(diagnostics["transition_count"])
    return ForwardTrajectoryResult(
        final_state=np.ascontiguousarray(state[0] if squeezed else state),
        initial_state=np.ascontiguousarray(initial[0] if squeezed else initial),
        anchors={
            key: np.ascontiguousarray(value[0] if squeezed else value)
            for key, value in anchors.items()
        },
        diagnostics=diagnostics,
        elapsed_seconds=elapsed,
        transition_count=transition_count,
        shard_records=tuple(records),
    )


def benchmark_tangent_phase(
    state: np.ndarray | Tensor,
    *,
    controller: TangentScoreController,
    path_ids: Sequence[int],
    outer_step: int,
    phase: int,
    reference_factory: Callable[[int], Callable[..., Any]],
    label: int | Tensor = 3,
    microsteps: int = 2,
    repeats: int = 3,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Measure a complete tangent phase without averaging favorable repeats.

    Each repeat starts from identical state and obtains a fresh adapter from
    ``reference_factory``.  The slowest complete-repeat transition rate is the
    conservative value intended for resource projection.
    """

    tensor, _ = _batched_float64_state(state, device=device)
    paths = _path_ids(path_ids, int(tensor.shape[0]))
    repeat_count = int(repeats)
    if repeat_count < 1:
        raise TangentRolloutContractError("benchmark repeats must be positive")
    sequence = ((int(outer_step), int(phase)),)
    rows: list[dict[str, Any]] = []
    for repeat in range(repeat_count):
        reference = reference_factory(repeat)
        result = run_reverse_shard(
            tensor.clone(),
            sequence,
            controller=controller,
            reference_transition=reference,
            path_ids=paths,
            label=label,
            microsteps=microsteps,
        )
        elapsed = result.elapsed_seconds
        rate = result.transition_count / elapsed if elapsed > 0.0 else 0.0
        rows.append(
            {
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "transition_count": result.transition_count,
                "transitions_per_second": rate,
                "output_state_sha256": _array_sha256(result.final_state),
                "diagnostics": dict(result.diagnostics),
            }
        )
    hashes = {str(item["output_state_sha256"]) for item in rows}
    return _semantic_record(
        {
            "schema": TANGENT_ROLLOUT_VERSION + "-phase-benchmark",
            "schema_version": 1,
            "path_count": len(paths),
            "outer_step": int(outer_step),
            "phase": int(phase),
            "microsteps": int(microsteps),
            "repeats": rows,
            "repeat_output_hashes_identical": int(len(hashes) == 1),
            "slowest_complete_repeat_rate": min(
                float(item["transitions_per_second"]) for item in rows
            ),
            "slowest_complete_repeat_seconds": max(
                float(item["elapsed_seconds"]) for item in rows
            ),
        }
    )


@dataclass(frozen=True)
class VerifiedFrequencyOneCheckpoint:
    model: FrequencyOneCoordinateZeroBaselinePredictor = field(repr=False, compare=False)
    run_dir: Path
    checkpoint_path: Path
    seed: int
    update: int
    checkpoint_file_sha256: str
    state_sha256: str
    candidate_inventory_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "seed": self.seed,
            "update": self.update,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "state_sha256": self.state_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
        }


def load_verified_frequency1_checkpoint(
    run_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_seed: int = 261_372,
    expected_update: int = 3_700,
) -> VerifiedFrequencyOneCheckpoint:
    """Strict-load the frozen post-hoc checkpoint through its inventory binding."""

    root = Path(run_dir).resolve()
    inventory_path = root / "candidate_inventory.json"
    inventory = _load_json(inventory_path)
    _validate_semantic_record(inventory, "candidate inventory")
    checkpoints = inventory.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise TangentRolloutContractError("candidate inventory checkpoint list is invalid")
    matches = [
        item
        for item in checkpoints
        if isinstance(item, Mapping)
        and item.get("seed") == int(expected_seed)
        and item.get("update") == int(expected_update)
    ]
    if len(matches) != 1:
        raise TangentRolloutContractError("frozen checkpoint is absent or ambiguous")
    candidate = dict(matches[0])
    relative = candidate.get("checkpoint_path")
    if not isinstance(relative, str):
        raise TangentRolloutContractError("checkpoint path binding is invalid")
    checkpoint_path = (root / relative).resolve()
    try:
        checkpoint_path.relative_to(root)
    except ValueError as exc:
        raise TangentRolloutContractError("checkpoint path escapes its parent run") from exc
    if not checkpoint_path.is_file():
        raise TangentRolloutContractError("checkpoint file is missing")
    measured_file_hash = _file_sha256(checkpoint_path)
    if measured_file_hash != candidate.get("checkpoint_file_sha256"):
        raise TangentRolloutContractError("checkpoint file hash changed")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise TangentRolloutContractError("cannot load checkpoint") from exc
    if not isinstance(payload, Mapping):
        raise TangentRolloutContractError("checkpoint payload is invalid")
    if payload.get("seed") != int(expected_seed) or payload.get("update") != int(expected_update):
        raise TangentRolloutContractError("checkpoint seed/update binding changed")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not all(isinstance(value, Tensor) for value in state.values()):
        raise TangentRolloutContractError("checkpoint state dictionary is invalid")
    measured_state_hash = state_dict_sha256(state)
    if (
        measured_state_hash != candidate.get("state_sha256")
        or measured_state_hash != payload.get("state_sha256")
    ):
        raise TangentRolloutContractError("checkpoint state hash changed")
    model = FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=False)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise TangentRolloutContractError("checkpoint strict state load failed") from exc
    if state_dict_sha256(model.state_dict()) != measured_state_hash:
        raise TangentRolloutContractError("strict-loaded model state differs")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return VerifiedFrequencyOneCheckpoint(
        model=model,
        run_dir=root,
        checkpoint_path=checkpoint_path,
        seed=int(expected_seed),
        update=int(expected_update),
        checkpoint_file_sha256=measured_file_hash,
        state_sha256=measured_state_hash,
        candidate_inventory_sha256=_file_sha256(inventory_path),
    )


@dataclass(frozen=True)
class VerifiedSourceTarget:
    run_dir: Path
    metadata: Mapping[str, Any]
    source_image: np.ndarray = field(repr=False, compare=False)
    mixed_target: np.ndarray = field(repr=False, compare=False)
    source_json_sha256: str
    source_npz_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "metadata": dict(self.metadata),
            "source_image_sha256": _array_sha256(self.source_image),
            "mixed_target_array_sha256": _array_sha256(self.mixed_target),
            "source_json_sha256": self.source_json_sha256,
            "source_npz_sha256": self.source_npz_sha256,
        }


def load_verified_source_target(run_dir: str | Path) -> VerifiedSourceTarget:
    """Load and internally verify the frozen source and mixed target."""

    root = Path(run_dir).resolve()
    metadata_path = root / "source_image.json"
    archive_path = root / "source_image.npz"
    metadata = _load_json(metadata_path)
    if not archive_path.is_file():
        raise TangentRolloutContractError("source image archive is missing")
    archive_hash = _file_sha256(archive_path)
    if metadata.get("npz_sha256") != archive_hash or metadata.get("npz_size") != archive_path.stat().st_size:
        raise TangentRolloutContractError("source image archive binding changed")
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != {"image", "mixed_target"}:
                raise TangentRolloutContractError("source archive schema changed")
            image = np.array(archive["image"], copy=True, order="C")
            mixed = np.array(archive["mixed_target"], copy=True, order="C")
    except (OSError, ValueError, KeyError) as exc:
        raise TangentRolloutContractError("cannot load source image archive") from exc
    for name, value in (("source image", image), ("mixed target", mixed)):
        if value.dtype != np.float64 or value.shape != (STATE_SIZE,):
            raise TangentRolloutContractError(f"{name} has wrong dtype or shape")
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise TangentRolloutContractError(f"{name} is not finite/nonnegative")
        if abs(float(np.sum(value)) - 1.0) > SIMPLEX_TOLERANCE:
            raise TangentRolloutContractError(f"{name} violates simplex mass")
    if _source_measure_sha256(image) != metadata.get("image_sha256"):
        raise TangentRolloutContractError("source image semantic hash changed")
    if _source_measure_sha256(mixed) != metadata.get("mixed_target_sha256"):
        raise TangentRolloutContractError("mixed target semantic hash changed")
    mix = metadata.get("lambda_mix")
    if not isinstance(mix, (int, float)) or not 0.0 <= float(mix) < 1.0:
        raise TangentRolloutContractError("lambda_mix is invalid")
    expected_mixed = (1.0 - float(mix)) * image + float(mix) / STATE_SIZE
    if not np.allclose(mixed, expected_mixed, rtol=0.0, atol=5.0e-16):
        raise TangentRolloutContractError("mixed target does not match source/lambda")
    if metadata.get("label") != 3 or metadata.get("dataset_index") != 7:
        raise TangentRolloutContractError("source image identity changed")
    image.setflags(write=False)
    mixed.setflags(write=False)
    return VerifiedSourceTarget(
        run_dir=root,
        metadata=dict(metadata),
        source_image=image,
        mixed_target=mixed,
        source_json_sha256=_file_sha256(metadata_path),
        source_npz_sha256=archive_hash,
    )


@dataclass(frozen=True)
class RawStateMetrics:
    squared_l2_error: float
    l1_error: float
    total_variation_distance: float
    centered_contrast_correlation: float
    contrast_correlation_defined: int
    simplex_mass: float
    simplex_mass_error: float
    minimum_state_value: float
    maximum_state_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def raw_state_metrics(
    state: np.ndarray | Tensor, target: np.ndarray | Tensor
) -> RawStateMetrics:
    """Compute objective metrics directly on raw float64 simplex states."""

    def array(value: np.ndarray | Tensor, name: str) -> np.ndarray:
        result = (
            value.detach().to(device="cpu").contiguous().numpy()
            if isinstance(value, Tensor)
            else np.asarray(value)
        )
        if result.dtype != np.float64 or result.shape != (STATE_SIZE,):
            raise TangentRolloutContractError(f"{name} must be float64 [784]")
        if not np.isfinite(result).all():
            raise TangentRolloutContractError(f"{name} contains nonfinite values")
        return result

    measured = array(state, "state")
    wanted = array(target, "target")
    difference = measured - wanted
    l1 = float(np.sum(np.abs(difference), dtype=np.float64))
    uniform = 1.0 / STATE_SIZE
    left = measured - uniform
    right = wanted - uniform
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    defined = int(denominator > 0.0 and math.isfinite(denominator))
    correlation = float(np.dot(left, right) / denominator) if defined else 0.0
    mass = float(np.sum(measured, dtype=np.float64))
    return RawStateMetrics(
        squared_l2_error=float(np.dot(difference, difference)),
        l1_error=l1,
        total_variation_distance=0.5 * l1,
        centered_contrast_correlation=correlation,
        contrast_correlation_defined=defined,
        simplex_mass=mass,
        simplex_mass_error=abs(mass - 1.0),
        minimum_state_value=float(np.min(measured)),
        maximum_state_value=float(np.max(measured)),
    )


def paired_metric_improvement(
    candidate: RawStateMetrics, zero: RawStateMetrics
) -> dict[str, float | None]:
    improvement = zero.squared_l2_error - candidate.squared_l2_error
    relative = (
        improvement / zero.squared_l2_error
        if zero.squared_l2_error > 0.0
        else None
    )
    return {
        "paired_squared_l2_improvement_over_zero": improvement,
        "relative_paired_squared_l2_improvement_over_zero": relative,
    }


@dataclass(frozen=True)
class FixedRenderingScale:
    raw_density_scale: float
    background_mass_per_cell: float
    source_image_scale: float
    lambda_mix: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fixed_rendering_scale(
    source_image: np.ndarray,
    mixed_target: np.ndarray,
    lambda_mix: float,
) -> FixedRenderingScale:
    source = np.asarray(source_image)
    mixed = np.asarray(mixed_target)
    if source.dtype != np.float64 or mixed.dtype != np.float64:
        raise TangentRolloutContractError("rendering sources must be float64")
    if source.shape != (STATE_SIZE,) or mixed.shape != (STATE_SIZE,):
        raise TangentRolloutContractError("rendering sources must have shape [784]")
    if not np.isfinite(source).all() or not np.isfinite(mixed).all():
        raise TangentRolloutContractError("rendering sources contain nonfinite values")
    mix = float(lambda_mix)
    raw_scale = float(np.max(mixed))
    source_scale = float(np.max(source))
    if not 0.0 <= mix < 1.0 or raw_scale <= 0.0 or source_scale <= 0.0:
        raise TangentRolloutContractError("fixed rendering scale is invalid")
    return FixedRenderingScale(
        raw_density_scale=raw_scale,
        background_mass_per_cell=mix / STATE_SIZE,
        source_image_scale=source_scale,
        lambda_mix=mix,
    )


def _render_uint8(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return np.ascontiguousarray(np.rint(255.0 * clipped), dtype=np.uint8).reshape(28, 28)


def render_raw_density(
    state: np.ndarray | Tensor, scale: FixedRenderingScale
) -> np.ndarray:
    value = (
        state.detach().to(device="cpu").contiguous().numpy()
        if isinstance(state, Tensor)
        else np.asarray(state)
    )
    if value.dtype != np.float64 or value.shape != (STATE_SIZE,):
        raise TangentRolloutContractError("rendered state must be float64 [784]")
    if not np.isfinite(value).all():
        raise TangentRolloutContractError("rendered state contains nonfinite values")
    return _render_uint8(value / scale.raw_density_scale)


def render_background_demixed(
    state: np.ndarray | Tensor, scale: FixedRenderingScale
) -> np.ndarray:
    value = (
        state.detach().to(device="cpu").contiguous().numpy()
        if isinstance(state, Tensor)
        else np.asarray(state)
    )
    if value.dtype != np.float64 or value.shape != (STATE_SIZE,):
        raise TangentRolloutContractError("rendered state must be float64 [784]")
    if not np.isfinite(value).all():
        raise TangentRolloutContractError("rendered state contains nonfinite values")
    demixed = (value - scale.background_mass_per_cell) / (1.0 - scale.lambda_mix)
    return _render_uint8(demixed / scale.source_image_scale)


def render_source_image(
    source_image: np.ndarray | Tensor, scale: FixedRenderingScale
) -> np.ndarray:
    """Render the already-unmixed source with the same frozen source scale."""

    value = (
        source_image.detach().to(device="cpu").contiguous().numpy()
        if isinstance(source_image, Tensor)
        else np.asarray(source_image)
    )
    if value.dtype != np.float64 or value.shape != (STATE_SIZE,):
        raise TangentRolloutContractError("source rendering must be float64 [784]")
    if not np.isfinite(value).all():
        raise TangentRolloutContractError("source rendering contains nonfinite values")
    return _render_uint8(value / scale.source_image_scale)


def save_png(path: str | Path, image: np.ndarray) -> None:
    """Atomically save a two-dimensional uint8 rendering."""

    value = np.asarray(image)
    if value.dtype != np.uint8 or value.shape != (28, 28):
        raise TangentRolloutContractError("PNG payload must be uint8 [28,28]")
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - torchvision installs Pillow.
        raise TangentRolloutContractError("Pillow is required to save PNG artifacts") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(value, mode="L").save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


# Small public persistence/hash surface shared by the additive fused-family
# scheduler.  Keeping the implementations here prevents two subtly different
# restart hash or atomic-write conventions.
def rollout_array_sha256(value: np.ndarray | Tensor) -> str:
    return _array_sha256(value)


def rollout_file_sha256(path: str | Path) -> str:
    return _file_sha256(Path(path))


def rollout_semantic_record(body: Mapping[str, Any]) -> dict[str, Any]:
    return _semantic_record(body)


def atomic_rollout_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    _atomic_npz(Path(path), arrays)


def load_rollout_state_npz(path: str | Path, *, expected_rows: int) -> np.ndarray:
    return _load_state_npz(Path(path), expected_rows=expected_rows)


def batched_rollout_state(
    value: np.ndarray | Tensor, *, device: torch.device | str | None = None
) -> tuple[Tensor, bool]:
    return _batched_float64_state(value, device=device)


__all__ = [
    "CertifiedExploratoryReference",
    "EXACT_FORWARD_FORBIDDEN_COUNTERS",
    "EXACT_FORWARD_FORBIDDEN_COUNTERS_VERSION",
    "EXPLORATORY_REFERENCE_RNG_NAMESPACE",
    "ExactForwardShardAggregateError",
    "FixedRenderingScale",
    "ForwardTrajectoryResult",
    "REFERENCE_LANE_CAP",
    "REVERSE_SHARD_OUTER_STEPS",
    "REVERSE_SHARD_PHASES",
    "RawStateMetrics",
    "ReverseShardResult",
    "ReverseTrajectoryResult",
    "ScaledTangentScoreController",
    "SignedDiagnosticTangentScoreController",
    "TANGENT_ROLLOUT_VERSION",
    "TangentRolloutContractError",
    "TargetFractionOracleController",
    "VerifiedFrequencyOneCheckpoint",
    "VerifiedSourceTarget",
    "ZeroTangentScoreController",
    "aggregate_exact_forward_shards",
    "aggregate_trajectory_phase_diagnostics",
    "atomic_rollout_npz",
    "batched_rollout_state",
    "benchmark_tangent_phase",
    "exploratory_reference_rng_key",
    "fixed_rendering_scale",
    "load_verified_frequency1_checkpoint",
    "load_verified_source_target",
    "load_rollout_state_npz",
    "paired_metric_improvement",
    "raw_state_metrics",
    "render_background_demixed",
    "render_raw_density",
    "render_source_image",
    "reverse_suffix_sequence",
    "rollout_array_sha256",
    "rollout_file_sha256",
    "rollout_semantic_record",
    "run_forward_trajectory",
    "run_reverse_shard",
    "run_reverse_trajectory",
    "save_png",
    "source_measure_sha256",
    "target_oracle_identity_control",
]
