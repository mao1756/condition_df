"""Exact duplicate-ID fused scheduler for exploratory tangent rollouts.

The scientific row identity and the stateless Jacobi transition identity are
deliberately separate.  Rows are uniquely named for persistence and controller
dispatch, while paired rows may repeat a canonical path ID so the certified
sampler receives identical Philox transition IDs.

The fast CUDA path is speculative only with respect to certification timing:
device masks are inspected once at the eight-step shard boundary.  Any
unresolved, fallback, or invalid lane discards the entire speculative shard and
replays it from the last committed input through the existing synchronous exact
sampler.  No speculative state is ever committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_artifacts import atomic_write_json
from mnist.d0_jacobi_rb_boundary_tangent_fused import (
    FusedTangentPhaseResult,
    TangentScoreController,
    controlled_reverse_phase_tangent_fused,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    OUTER_STEPS,
    PHASE_COUNT,
    STATE_SIZE,
    ModelInputs,
    matching_indices,
    semantic_sha256,
)
from mnist.d0_jacobi_rb_reverse_controller import (
    NAMESPACE_VERSION,
    controller_transition_ids,
)
from mnist.d0_jacobi_rb_tangent_rollout import (
    CertifiedExploratoryReference,
    REFERENCE_LANE_CAP,
    REVERSE_SHARD_PHASES,
    SIMPLEX_TOLERANCE,
    ScaledTangentScoreController,
    SignedDiagnosticTangentScoreController,
    TargetFractionOracleController,
    TangentRolloutContractError,
    atomic_rollout_npz,
    batched_rollout_state,
    exploratory_reference_rng_key,
    load_rollout_state_npz,
    rollout_array_sha256,
    rollout_file_sha256,
    rollout_semantic_record,
)


FUSED_TANGENT_VERSION = "d0-jacobi-rb-tangent-fused-v1"
FUSED_SHARD_OUTER_STEPS = 8
FUSED_SHARD_PHASES = FUSED_SHARD_OUTER_STEPS * PHASE_COUNT
FusedControllerKind = Literal[
    "zero", "learned", "signed_diagnostic", "oracle"
]
FUSED_CONTROLLER_KINDS = frozenset(
    {"zero", "learned", "signed_diagnostic", "oracle"}
)
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
_FORBIDDEN_CANDIDATE_COUNTS = tuple(
    name for name in _FORBIDDEN_REFERENCE_COUNTS if name != "approximation_count"
)
ReferenceContract = Literal["certified_exact", "candidate_approximate"]
CANDIDATE_REFERENCE_CONTRACT = "candidate_approximate_v1"


class FusedTangentContractError(TangentRolloutContractError):
    """A fused-row, exact-health, or restart contract was violated."""


def _safe_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in value
    ):
        raise FusedTangentContractError(f"{name} is not safe nonempty text")
    return value


def _json_safe_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    try:
        json.dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FusedTangentContractError(f"{name} is not canonical JSON") from exc
    return result


@dataclass(frozen=True)
class FusedRowSpec:
    """Unique scientific row plus intentionally reusable RNG path identity."""

    row_key: str
    canonical_path_id: int
    controller_kind: FusedControllerKind
    variant: str
    horizon: str
    gain: float | None = None
    controller_binding: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _safe_text(self.row_key, "row key")
        if (
            isinstance(self.canonical_path_id, bool)
            or not isinstance(self.canonical_path_id, (int, np.integer))
            or not 0 <= int(self.canonical_path_id) < (1 << 20)
        ):
            raise FusedTangentContractError(
                "canonical path ID lies outside the 20-bit contract"
            )
        if self.controller_kind not in FUSED_CONTROLLER_KINDS:
            raise FusedTangentContractError("controller kind is not fused-row safe")
        _safe_text(self.variant, "variant")
        _safe_text(self.horizon, "horizon")
        if self.controller_kind == "learned":
            if self.gain is None or not math.isfinite(float(self.gain)) or float(self.gain) < 0.0:
                raise FusedTangentContractError("learned row gain is invalid")
        elif self.controller_kind == "signed_diagnostic":
            if (
                self.gain is None
                or not math.isfinite(float(self.gain))
                or float(self.gain) >= 0.0
            ):
                raise FusedTangentContractError(
                    "signed diagnostic row gain is invalid"
                )
        elif self.gain is not None:
            raise FusedTangentContractError(
                "only learned and signed diagnostic rows may bind a gain"
            )
        _json_safe_mapping(self.controller_binding, "controller binding")

    def to_record(self) -> dict[str, Any]:
        return {
            "row_key": self.row_key,
            "canonical_path_id": int(self.canonical_path_id),
            "controller_kind": self.controller_kind,
            "variant": self.variant,
            "horizon": self.horizon,
            "gain": None if self.gain is None else float(self.gain),
            "controller_binding": dict(self.controller_binding),
        }


def validate_fused_row_specs(
    row_specs: Sequence[FusedRowSpec], *, expected_rows: int | None = None
) -> tuple[FusedRowSpec, ...]:
    specs = tuple(row_specs)
    if not specs or any(not isinstance(item, FusedRowSpec) for item in specs):
        raise FusedTangentContractError("fused row table is empty or malformed")
    if expected_rows is not None and len(specs) != int(expected_rows):
        raise FusedTangentContractError("fused row table does not match state rows")
    keys = tuple(item.row_key for item in specs)
    if len(set(keys)) != len(keys):
        raise FusedTangentContractError("fused row keys must be unique")
    if len(specs) * EDGES_PER_PHASE > REFERENCE_LANE_CAP:
        raise FusedTangentContractError("fused row table exceeds the 4096-lane cap")
    return specs


def fused_transition_ids(
    row_specs: Sequence[FusedRowSpec],
    *,
    outer_step: int,
    phase: int,
    reverse_microstep: int,
    role: str,
    device: str | torch.device,
) -> Tensor:
    specs = validate_fused_row_specs(row_specs)
    return controller_transition_ids(
        tuple(int(item.canonical_path_id) for item in specs),
        outer_step=int(outer_step),
        phase=int(phase),
        reverse_microstep=int(reverse_microstep),
        role=role,
        device=device,
    )


@dataclass(frozen=True)
class FusedTransitionIdPlan:
    """One-transfer device plan for every exact transition ID in a shard."""

    sequence: tuple[tuple[int, int], ...]
    canonical_path_ids: tuple[int, ...]
    microsteps: int
    ids: Tensor = field(repr=False, compare=False)
    matching_tails: Tensor = field(repr=False, compare=False)
    matching_heads: Tensor = field(repr=False, compare=False)

    def phase_ids(self, sequence_index: int) -> Tensor:
        index = int(sequence_index)
        if not 0 <= index < len(self.sequence):
            raise FusedTangentContractError("transition-ID phase index is invalid")
        return self.ids[index]


def build_fused_transition_id_plan(
    row_specs: Sequence[FusedRowSpec],
    sequence: Sequence[tuple[int, int]],
    *,
    microsteps: int,
    device: str | torch.device,
) -> FusedTransitionIdPlan:
    """Build all shard IDs on CPU, then issue one device transfer.

    The tensor layout is ``[phase,microstep,side,row,edge]``.  Row keys,
    variants, horizons, gains, and chunking never enter the packed identity.
    """

    specs = validate_fused_row_specs(row_specs)
    normalized = _validate_reverse_sequence(sequence)
    count = int(microsteps)
    if count not in {2, 4, 8}:
        raise FusedTangentContractError("microsteps must be 2, 4, or 8")
    paths = tuple(int(item.canonical_path_id) for item in specs)
    phase_tensors: list[Tensor] = []
    for outer_step, phase in normalized:
        microstep_tensors: list[Tensor] = []
        for reverse_index in range(count):
            side_tensors = [
                controller_transition_ids(
                    paths,
                    outer_step=outer_step,
                    phase=phase,
                    reverse_microstep=reverse_index,
                    role=f"reverse_reference_{side}_control_M{count}",
                    device="cpu",
                )
                for side in ("pre", "post")
            ]
            microstep_tensors.append(torch.stack(side_tensors, dim=0))
        phase_tensors.append(torch.stack(microstep_tensors, dim=0))
    host_ids = torch.stack(phase_tensors, dim=0).contiguous()
    selected_device = torch.device(device)
    device_ids = host_ids.to(
        device=selected_device, non_blocking=False
    ).contiguous()
    host_tails, host_heads = matching_indices(device="cpu")
    matching_tails = host_tails.to(
        device=selected_device, non_blocking=False
    ).contiguous()
    matching_heads = host_heads.to(
        device=selected_device, non_blocking=False
    ).contiguous()
    return FusedTransitionIdPlan(
        sequence=normalized,
        canonical_path_ids=paths,
        microsteps=count,
        ids=device_ids,
        matching_tails=matching_tails,
        matching_heads=matching_heads,
    )


def _slice_inputs(inputs: ModelInputs, row: int) -> ModelInputs:
    selection = slice(int(row), int(row) + 1)
    return ModelInputs(
        later_full_state=inputs.later_full_state[selection],
        reverse_time=inputs.reverse_time[selection],
        phase=inputs.phase[selection],
        color=inputs.color[selection],
        duration=inputs.duration[selection],
        label=inputs.label[selection],
    )


class FusedTangentControllerBank:
    """Stable one-row dispatcher that never exposes row identity to a model."""

    def __init__(
        self,
        row_specs: Sequence[FusedRowSpec],
        controllers: Mapping[str, TangentScoreController],
    ) -> None:
        self.row_specs = validate_fused_row_specs(row_specs)
        if not isinstance(controllers, Mapping):
            raise TypeError("controllers must be a row-key mapping")
        expected = {
            item.row_key for item in self.row_specs if item.controller_kind != "zero"
        }
        if set(controllers) != expected:
            raise FusedTangentContractError(
                "controller mapping must bind exactly the nonzero fused rows"
            )
        self.controllers = dict(controllers)
        for spec in self.row_specs:
            if spec.controller_kind == "zero":
                continue
            controller = self.controllers[spec.row_key]
            if not isinstance(controller, TangentScoreController):
                raise TypeError("each fused controller must implement score_prediction")
            is_signed_diagnostic = isinstance(
                controller, SignedDiagnosticTangentScoreController
            )
            if spec.controller_kind == "signed_diagnostic" and not is_signed_diagnostic:
                raise FusedTangentContractError(
                    "signed diagnostic rows require the diagnostic-only wrapper"
                )
            if spec.controller_kind != "signed_diagnostic" and is_signed_diagnostic:
                raise FusedTangentContractError(
                    "signed diagnostic wrapper requires a signed diagnostic row"
                )
            if isinstance(
                controller,
                (ScaledTangentScoreController, SignedDiagnosticTangentScoreController),
            ):
                controller_gain = getattr(controller, "gain", None)
                if (
                    not isinstance(controller_gain, (int, float))
                    or float(controller_gain) != float(spec.gain)
                ):
                    raise FusedTangentContractError(
                        "fused row gain differs from its controller wrapper"
                    )
        self._device_record: dict[str, Tensor] = {}
        self._matching_tensors: tuple[Tensor, Tensor] | None = None

    def prepare_device(
        self,
        device: torch.device | str,
        *,
        matching_tensors: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        """Move controller state and matching indices before shard timing."""

        selected = torch.device(device)
        for controller in self.controllers.values():
            if isinstance(controller, torch.nn.Module):
                controller.to(device=selected)
        if matching_tensors is None:
            tails, heads = matching_indices(device=selected)
        else:
            tails, heads = matching_tensors
            if (
                tails.shape != (4, EDGES_PER_PHASE)
                or heads.shape != tails.shape
                or tails.dtype != torch.long
                or heads.dtype != torch.long
                or tails.device != selected
                or heads.device != selected
            ):
                raise FusedTangentContractError(
                    "prepared matching tensors are invalid"
                )
        self._matching_tensors = (tails, heads)
        self.reset_device_telemetry(selected)

    def reset_device_telemetry(self, device: torch.device) -> None:
        rows = len(self.row_specs)
        self._device_record = {
            name: torch.zeros(rows, dtype=dtype, device=device)
            for name, dtype in {
                "call_count": torch.int64,
                "lane_count": torch.int64,
                "score_count": torch.int64,
                "score_squared_sum": torch.float64,
                "score_maximum_absolute": torch.float64,
                "unscaled_score_squared_sum": torch.float64,
                "unscaled_score_maximum_absolute": torch.float64,
                "movable_count": torch.int64,
                "already_equal_count": torch.int64,
                "zero_pair_mass_count": torch.int64,
                "zero_duration_count": torch.int64,
                "target_oracle_unreachable_boundary_count": torch.int64,
                "clipping_count": torch.int64,
                "floor_count": torch.int64,
                "projection_count": torch.int64,
                "nonfinite_score_count": torch.int64,
            }.items()
        }

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs or inputs.batch_size != len(self.row_specs):
            raise FusedTangentContractError(
                "controller bank requires exact row-aligned ModelInputs"
            )
        if (
            not self._device_record
            or self._matching_tensors is None
            or self._matching_tensors[0].device
            != inputs.later_full_state.device
        ):
            raise FusedTangentContractError(
                "controller bank must be prepared before the fused phase"
            )
        outputs: list[Tensor] = []
        for row, spec in enumerate(self.row_specs):
            one = _slice_inputs(inputs, row)
            if spec.controller_kind == "zero":
                score = torch.zeros(
                    (1, EDGES_PER_PHASE),
                    dtype=torch.float64,
                    device=inputs.later_full_state.device,
                )
            else:
                controller = self.controllers[spec.row_key]
                method = getattr(controller, "score_prediction_deferred", None)
                kwargs: dict[str, Any] = {}
                if isinstance(controller, TargetFractionOracleController):
                    method = getattr(
                        controller, "score_prediction_deferred_prepared", None
                    )
                    assert self._matching_tensors is not None
                    kwargs = {
                        "tails_all": self._matching_tensors[0],
                        "heads_all": self._matching_tensors[1],
                    }
                if not callable(method):
                    raise FusedTangentContractError(
                        f"row {spec.row_key} lacks a device-only score path"
                    )
                score = method(one, **kwargs)
                if (
                    not isinstance(score, Tensor)
                    or score.shape != (1, EDGES_PER_PHASE)
                    or score.device != inputs.later_full_state.device
                    or not score.dtype.is_floating_point
                ):
                    raise FusedTangentContractError(
                        f"row {spec.row_key} returned a malformed score"
                    )
            score64 = score.to(dtype=torch.float64)
            # Geometry is reconstructed from permitted current inputs only.
            assert self._matching_tensors is not None
            tails_all, heads_all = self._matching_tensors
            color = one.color.to(dtype=torch.long)
            tails, heads = tails_all[color], heads_all[color]
            state64 = one.later_full_state.to(dtype=torch.float64)
            pair = state64.gather(1, tails) + state64.gather(1, heads)
            active = pair > 0.0
            finite = torch.isfinite(score64)
            safe = torch.where(active & finite, score64, torch.zeros_like(score64))
            self._device_record["call_count"][row] += int(
                spec.controller_kind != "zero"
            )
            self._device_record["lane_count"][row] += EDGES_PER_PHASE
            self._device_record["score_count"][row] += torch.sum(
                active, dtype=torch.int64
            )
            self._device_record["score_squared_sum"][row] += torch.sum(
                safe.square(), dtype=torch.float64
            )
            self._device_record["score_maximum_absolute"][row] = torch.maximum(
                self._device_record["score_maximum_absolute"][row],
                torch.amax(torch.abs(safe)),
            )
            self._device_record["nonfinite_score_count"][row] += torch.sum(
                active & ~finite, dtype=torch.int64
            )
            if isinstance(
                self.controllers.get(spec.row_key),
                (
                    ScaledTangentScoreController,
                    SignedDiagnosticTangentScoreController,
                ),
            ):
                base = getattr(
                    self.controllers[spec.row_key], "_last_deferred_unscaled", None
                )
                if isinstance(base, Tensor):
                    base_safe = torch.where(
                        active & torch.isfinite(base), base, torch.zeros_like(base)
                    )
                    self._device_record["unscaled_score_squared_sum"][row] += torch.sum(
                        base_safe.square(), dtype=torch.float64
                    )
                    self._device_record["unscaled_score_maximum_absolute"][row] = torch.maximum(
                        self._device_record["unscaled_score_maximum_absolute"][row],
                        torch.amax(torch.abs(base_safe)),
                    )
            controller = self.controllers.get(spec.row_key)
            if isinstance(controller, TargetFractionOracleController):
                masks = getattr(controller, "_last_deferred_masks", {})
                for source, destination in (
                    ("movable", "movable_count"),
                    ("already_equal", "already_equal_count"),
                    ("zero_pair_mass", "zero_pair_mass_count"),
                    ("zero_duration", "zero_duration_count"),
                    (
                        "target_oracle_unreachable_boundary",
                        "target_oracle_unreachable_boundary_count",
                    ),
                ):
                    mask = masks.get(source) if isinstance(masks, Mapping) else None
                    if isinstance(mask, Tensor):
                        self._device_record[destination][row] += torch.sum(
                            mask, dtype=torch.int64
                        )
            outputs.append(score64)
        return torch.cat(outputs, dim=0).contiguous()

    def device_record_tensors(self) -> Mapping[str, Tensor]:
        return dict(self._device_record)

    def row_records_from_host(
        self, host: Mapping[str, Sequence[int | float]]
    ) -> tuple[dict[str, Any], ...]:
        result: list[dict[str, Any]] = []
        for row, spec in enumerate(self.row_specs):
            record: dict[str, Any] = {
                "row_key": spec.row_key,
                "controller_kind": spec.controller_kind,
                "gain": spec.gain,
            }
            for name, values in host.items():
                if name in self._device_record and len(values) == len(self.row_specs):
                    value = values[row]
                    record[name] = (
                        int(value) if self._device_record[name].dtype == torch.int64 else float(value)
                    )
            count = int(record.get("score_count", 0))
            for prefix in ("score", "unscaled_score"):
                squared = float(record.get(f"{prefix}_squared_sum", 0.0))
                record[f"{prefix}_rms"] = math.sqrt(squared / count) if count else 0.0
            result.append(record)
        return tuple(result)


class _SynchronousFusedReference(CertifiedExploratoryReference):
    """Existing synchronous authorizer plus truthful per-row health."""

    def __init__(self, *args: Any, row_count: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fused_row_count = int(row_count)
        self._fused_per_row = [
            {
                "transition_count": 0,
                "active_count": 0,
                "certified_count": 0,
                "fallback_count": 0,
                "unauthorized_count": 0,
                "invalid_count": 0,
            }
            for _ in range(self._fused_row_count)
        ]

    def __call__(self, **kwargs: Any) -> Any:
        exposure = kwargs.get("exposure")
        result = super().__call__(**kwargs)
        later = getattr(result, "later_head_fraction", None)
        certified = getattr(result, "certified_mask", None)
        fallback = getattr(result, "fallback_mask", None)
        if not all(isinstance(value, Tensor) for value in (exposure, later, certified)):
            raise FusedTangentContractError(
                "synchronous replay lacks per-row certificate tensors"
            )
        assert isinstance(exposure, Tensor)
        assert isinstance(later, Tensor)
        assert isinstance(certified, Tensor)
        if later.shape != (self._fused_row_count, EDGES_PER_PHASE):
            raise FusedTangentContractError(
                "synchronous replay row shape changed"
            )
        active = exposure > 0.0
        fallback_mask = (
            fallback
            if isinstance(fallback, Tensor)
            else torch.zeros_like(active)
        )
        valid = torch.isfinite(later) & (later >= 0.0) & (later <= 1.0)
        packed = torch.stack(
            (
                torch.full(
                    (self._fused_row_count,),
                    EDGES_PER_PHASE,
                    dtype=torch.int64,
                    device=later.device,
                ),
                torch.sum(active, dim=1, dtype=torch.int64),
                torch.sum(active & certified, dim=1, dtype=torch.int64),
                torch.sum(fallback_mask, dim=1, dtype=torch.int64),
                torch.sum(active & ~certified, dim=1, dtype=torch.int64),
                torch.sum(~valid, dim=1, dtype=torch.int64),
            ),
            dim=1,
        ).detach().cpu().numpy()
        names = tuple(self._fused_per_row[0]) if self._fused_per_row else ()
        for row, values in enumerate(packed):
            for name, value in zip(names, values, strict=True):
                self._fused_per_row[row][name] += int(value)
        return result

    def record(self) -> dict[str, Any]:
        record = super().record()
        rows = []
        for value in self._fused_per_row:
            item = dict(value)
            item["structural_noop_count"] = (
                item["transition_count"] - item["active_count"]
            )
            item["certificate_fraction"] = (
                item["certified_count"] / item["active_count"]
                if item["active_count"]
                else 1.0
            )
            rows.append(item)
        aggregate_names = (
            "transition_count",
            "active_count",
            "structural_noop_count",
            "certified_count",
            "fallback_count",
            "unauthorized_count",
            "invalid_count",
        )
        aggregate = {
            name: sum(int(row[name]) for row in rows)
            for name in aggregate_names
        }
        if aggregate["transition_count"] != int(record["transition_count"]):
            raise FusedTangentContractError(
                "synchronous replay aggregate transition count changed"
            )
        record.update(aggregate)
        record["certificate_fraction"] = (
            aggregate["certified_count"] / aggregate["active_count"]
            if aggregate["active_count"]
            else 1.0
        )
        record["per_row"] = rows
        return record


DEFERRED_REFERENCE_RNG_ROLES = tuple(
    f"reverse_reference_{side}_control_M{microsteps}"
    for microsteps in (2, 4, 8)
    for side in ("pre", "post")
)


def prepare_deferred_reference_rng_seed_map(
    *,
    prepared_backend: Any,
    root_seed: int,
    stream_role: str,
) -> Mapping[str, Any]:
    """Build the finite CUDA RNG seed map before family/shard timing.

    Exactly six role seeds cover the frozen tangent refinement set M={2,4,8}.
    The returned mapping is immutable and may be reused by every shard
    reference sharing the same backend, root seed, and stream role.
    """

    if prepared_backend is None:
        raise FusedTangentContractError(
            "deferred seed preparation requires a warm CUDA backend"
        )
    exploratory_reference_rng_key(root_seed, stream_role, "contract-check")
    from mnist.d0_jacobi_rb_cuda_deferred import (
        prepare_alpha1_rb_transition_cuda_rng_seed,
    )

    values = {
        role: prepare_alpha1_rb_transition_cuda_rng_seed(
            rng_key=exploratory_reference_rng_key(root_seed, stream_role, role),
            prepared=prepared_backend,
        )
        for role in DEFERRED_REFERENCE_RNG_ROLES
    }
    return MappingProxyType(values)


class DeferredCertifiedFusedReference:
    """Enqueue-only exact CUDA reference with one shard-boundary transfer."""

    def __init__(
        self,
        *,
        profile: JacobiRBCudaProfile,
        root_seed: int,
        stream_role: str,
        sampler: Callable[..., Any] | None = None,
        synchronous_sampler: Callable[..., Any] | None = None,
        prepared_backend: Any | None = None,
        prepared_rng_seeds: Mapping[str, Any] | None = None,
        lane_cap: int = REFERENCE_LANE_CAP,
        row_chunk_size: int | None = None,
    ) -> None:
        if not isinstance(profile, JacobiRBCudaProfile):
            raise TypeError("profile must be a JacobiRBCudaProfile")
        if not 1 <= int(lane_cap) <= REFERENCE_LANE_CAP:
            raise FusedTangentContractError("reference lane cap is invalid")
        if row_chunk_size is not None and int(row_chunk_size) <= 0:
            raise FusedTangentContractError("row chunk size must be positive")
        default_sampler = sampler is None
        if default_sampler:
            from mnist import d0_jacobi_rb_cuda_deferred as cuda_backend

            sampler = getattr(
                cuda_backend,
                "enqueue_alpha1_rb_transition_batch_cuda_no_fallback",
                None,
            )
        if not callable(sampler):
            raise FusedTangentContractError(
                "deferred CUDA sampler is unavailable; synchronous substitution is forbidden"
            )
        if default_sampler and prepared_backend is None:
            raise FusedTangentContractError(
                "default deferred sampler requires a warm prepared backend"
            )
        if synchronous_sampler is None:
            from mnist.d0_jacobi_rb_cuda import sample_alpha1_rb_transition_batch_cuda

            synchronous_sampler = sample_alpha1_rb_transition_batch_cuda
        if not callable(synchronous_sampler):
            raise TypeError("synchronous sampler must be callable")
        exploratory_reference_rng_key(root_seed, stream_role, "contract-check")
        self.profile = profile
        self.root_seed = int(root_seed)
        self.stream_role = stream_role
        self.sampler = sampler
        self.prepared_backend = prepared_backend
        self.synchronous_sampler = synchronous_sampler
        self.lane_cap = int(lane_cap)
        self.row_chunk_size = None if row_chunk_size is None else int(row_chunk_size)
        if self.prepared_backend is None:
            if prepared_rng_seeds not in (None, {}):
                raise FusedTangentContractError(
                    "a prepared RNG seed map requires a prepared CUDA backend"
                )
            self._prepared_rng_seeds: Mapping[str, Any] = MappingProxyType({})
        else:
            from mnist.d0_jacobi_rb_cuda_deferred import (
                validate_prepared_alpha1_rb_transition_cuda_rng_seed,
            )

            if not isinstance(prepared_rng_seeds, Mapping):
                raise FusedTangentContractError(
                    "prepared CUDA backend requires a prebuilt RNG seed map"
                )
            supplied = dict(prepared_rng_seeds)
            if set(supplied) != set(DEFERRED_REFERENCE_RNG_ROLES):
                raise FusedTangentContractError(
                    "prebuilt RNG seed map has missing or unexpected roles"
                )
            for role in DEFERRED_REFERENCE_RNG_ROLES:
                try:
                    validate_prepared_alpha1_rb_transition_cuda_rng_seed(
                        prepared_seed=supplied[role],
                        rng_key=exploratory_reference_rng_key(
                            self.root_seed, self.stream_role, role
                        ),
                        prepared=self.prepared_backend,
                    )
                except (TypeError, ValueError) as exc:
                    raise FusedTangentContractError(
                        f"prebuilt CUDA RNG seed is invalid for role {role}"
                    ) from exc
            self._prepared_rng_seeds = MappingProxyType(supplied)
        self._batches: list[tuple[int, int, Any]] = []
        self._row_count: int | None = None
        self._started = time.perf_counter()
        self._record: dict[str, Any] | None = None

    @staticmethod
    def _result_tensor(result: Any, *names: str) -> Tensor | None:
        for name in names:
            value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
            if isinstance(value, Tensor):
                return value
        return None

    def __call__(
        self,
        *,
        head_fraction: Tensor,
        exposure: Tensor,
        transition_ids: Tensor,
        role: str,
    ) -> Any:
        if not all(isinstance(item, Tensor) for item in (head_fraction, exposure, transition_ids)):
            raise TypeError("fused reference inputs must be tensors")
        if head_fraction.shape != exposure.shape or head_fraction.shape != transition_ids.shape:
            raise FusedTangentContractError("fused reference shapes differ")
        if head_fraction.ndim != 2 or head_fraction.shape[1] != EDGES_PER_PHASE:
            raise FusedTangentContractError("fused reference requires [R,392]")
        if int(head_fraction.numel()) > self.lane_cap:
            raise FusedTangentContractError("reference launch exceeds lane cap")
        rows = int(head_fraction.shape[0])
        if self._row_count is None:
            self._row_count = rows
        elif self._row_count != rows:
            raise FusedTangentContractError(
                "fused reference row count changed within a shard"
            )
        chunk = rows if self.row_chunk_size is None else min(rows, self.row_chunk_size)
        outputs: list[Tensor] = []
        targets: list[Tensor] = []
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            kwargs = {
                "rng_key": exploratory_reference_rng_key(
                    self.root_seed, self.stream_role, role
                ),
                "transition_ids": transition_ids[start:stop].contiguous(),
            }
            if self.prepared_backend is None:
                kwargs["profile"] = self.profile
            else:
                kwargs["prepared"] = self.prepared_backend
                prepared_seed = self._prepared_rng_seeds.get(role)
                if prepared_seed is None:
                    raise FusedTangentContractError(
                        "deferred role lacks a prebuilt CUDA RNG seed"
                    )
                kwargs["prepared_rng_seed"] = prepared_seed
            result = self.sampler(
                head_fraction[start:stop].contiguous(),
                exposure[start:stop].contiguous(),
                **kwargs,
            )
            later = self._result_tensor(result, "later_head_fraction")
            if later is None or later.shape != head_fraction[start:stop].shape:
                raise FusedTangentContractError("deferred sampler lacks later fractions")
            target = self._result_tensor(result, "denoising_target")
            outputs.append(later)
            if target is not None:
                targets.append(target)
            self._batches.append((start, stop, result))
        return SimpleNamespace(
            later_head_fraction=torch.cat(outputs, dim=0).contiguous(),
            denoising_target=(
                torch.cat(targets, dim=0).contiguous() if len(targets) == len(outputs) else None
            ),
        )

    def make_synchronous_reference(self) -> CertifiedExploratoryReference:
        if self._row_count is None:
            raise FusedTangentContractError(
                "cannot replay an unstarted fused reference"
            )
        return _SynchronousFusedReference(
            profile=self.profile,
            root_seed=self.root_seed,
            stream_role=self.stream_role,
            sampler=self.synchronous_sampler,
            lane_cap=self.lane_cap,
            row_count=self._row_count,
        )

    def finalize_shard(
        self, extra_device_tensors: Mapping[str, Tensor]
    ) -> dict[str, Any]:
        if self._record is not None:
            raise FusedTangentContractError("reference shard was finalized twice")
        device = next(iter(extra_device_tensors.values())).device
        counters: dict[str, Tensor] = {
            "transition_count": torch.zeros((), dtype=torch.int64, device=device),
            "active_count": torch.zeros((), dtype=torch.int64, device=device),
            "structural_noop_count": torch.zeros(
                (), dtype=torch.int64, device=device
            ),
            "certified_count": torch.zeros((), dtype=torch.int64, device=device),
            "fallback_count": torch.zeros((), dtype=torch.int64, device=device),
            "unauthorized_count": torch.zeros((), dtype=torch.int64, device=device),
            "invalid_count": torch.zeros((), dtype=torch.int64, device=device),
        }
        for name in _FORBIDDEN_REFERENCE_COUNTS:
            counters[name] = torch.zeros((), dtype=torch.int64, device=device)
        row_count = int(self._row_count or 0)
        row_counters = {
            name: torch.zeros(row_count, dtype=torch.int64, device=device)
            for name in (
                "transition_count",
                "active_count",
                "structural_noop_count",
                "certified_count",
                "fallback_count",
                "unauthorized_count",
                "invalid_count",
            )
        }
        for start, stop, result in self._batches:
            later = self._result_tensor(result, "later_head_fraction")
            if later is None:
                raise FusedTangentContractError("deferred result lost its scientific output")
            active = self._result_tensor(result, "active_mask")
            if active is None:
                active = torch.ones_like(later, dtype=torch.bool)
            structural_noop = self._result_tensor(result, "structural_noop_mask")
            if structural_noop is None:
                structural_noop = ~active
            authorized = self._result_tensor(
                result, "authorized_mask", "certified_mask"
            )
            if authorized is None:
                raise FusedTangentContractError("deferred result lacks authorization mask")
            fallback = self._result_tensor(result, "fallback_mask")
            if fallback is None:
                fallback = active & ~authorized
            valid = self._result_tensor(result, "valid_mask", "validity_mask")
            if valid is None:
                valid = torch.isfinite(later) & (later >= 0.0) & (later <= 1.0)
            counters["transition_count"] += later.numel()
            counters["active_count"] += torch.sum(active, dtype=torch.int64)
            counters["structural_noop_count"] += torch.sum(
                structural_noop, dtype=torch.int64
            )
            counters["certified_count"] += torch.sum(
                active & authorized, dtype=torch.int64
            )
            counters["fallback_count"] += torch.sum(fallback, dtype=torch.int64)
            counters["unauthorized_count"] += torch.sum(
                active & ~authorized, dtype=torch.int64
            )
            counters["invalid_count"] += torch.sum(~valid, dtype=torch.int64)
            row_counters["transition_count"][start:stop] += torch.full(
                (stop - start,),
                int(later.shape[1]),
                dtype=torch.int64,
                device=device,
            )
            row_counters["active_count"][start:stop] += torch.sum(
                active, dim=1, dtype=torch.int64
            )
            row_counters["structural_noop_count"][start:stop] += torch.sum(
                structural_noop, dim=1, dtype=torch.int64
            )
            row_counters["certified_count"][start:stop] += torch.sum(
                active & authorized, dim=1, dtype=torch.int64
            )
            row_counters["fallback_count"][start:stop] += torch.sum(
                fallback, dim=1, dtype=torch.int64
            )
            row_counters["unauthorized_count"][start:stop] += torch.sum(
                active & ~authorized, dim=1, dtype=torch.int64
            )
            row_counters["invalid_count"][start:stop] += torch.sum(
                ~valid, dim=1, dtype=torch.int64
            )
            diagnostics = (
                result.get("device_diagnostics", result.get("diagnostics", {}))
                if isinstance(result, Mapping)
                else getattr(result, "device_diagnostics", getattr(result, "diagnostics", {}))
            )
            if isinstance(diagnostics, Mapping):
                for name in _FORBIDDEN_REFERENCE_COUNTS:
                    value = diagnostics.get(name)
                    if isinstance(value, Tensor):
                        counters[name] += value.to(device=device, dtype=torch.int64).sum()

        ordered: list[tuple[str, Tensor]] = []
        ordered.extend((f"reference.{name}", value.reshape(-1)) for name, value in counters.items())
        ordered.extend(
            (f"reference_row.{name}", value.reshape(-1))
            for name, value in row_counters.items()
        )
        ordered.extend(
            (f"extra.{name}", value.to(device=device).reshape(-1))
            for name, value in sorted(extra_device_tensors.items())
        )
        sizes = {name: int(value.numel()) for name, value in ordered}
        packed = torch.cat(
            [value.to(dtype=torch.float64) for _, value in ordered], dim=0
        ).contiguous()
        # Exactly one packed device-to-host validation transfer.  It is also
        # the shard's synchronization boundary.
        host_values = packed.detach().cpu().numpy()
        host: dict[str, list[float]] = {}
        offset = 0
        for name, _ in ordered:
            count = sizes[name]
            host[name] = [float(item) for item in host_values[offset : offset + count]]
            offset += count
        scalar = {name: int(host[f"reference.{name}"][0]) for name in counters}
        per_row = []
        for row in range(row_count):
            row_record = {
                name: int(host[f"reference_row.{name}"][row])
                for name in row_counters
            }
            row_record["certificate_fraction"] = (
                row_record["certified_count"] / row_record["active_count"]
                if row_record["active_count"]
                else 1.0
            )
            per_row.append(row_record)
        forbidden = {name: scalar[name] for name in _FORBIDDEN_REFERENCE_COUNTS}
        needs_replay = bool(
            scalar["fallback_count"]
            or scalar["unauthorized_count"]
            or scalar["invalid_count"]
            or any(forbidden.values())
        )
        elapsed = time.perf_counter() - self._started
        total_memory = 0
        peak_memory = 0
        if device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(device))
            total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        self._record = {
            "schema": FUSED_TANGENT_VERSION + "-deferred-reference-shard",
            "root_seed": self.root_seed,
            "rng_namespace": exploratory_reference_rng_key(
                self.root_seed, self.stream_role, "record"
            )[1],
            "stream_role": self.stream_role,
            "variant_in_rng_key": 0,
            **scalar,
            "certificate_fraction": (
                scalar["certified_count"] / scalar["active_count"]
                if scalar["active_count"]
                else 1.0
            ),
            "fallback_fraction": (
                scalar["fallback_count"] / scalar["transition_count"]
                if scalar["transition_count"]
                else 0.0
            ),
            "forbidden_counts": forbidden,
            "needs_synchronous_replay": int(needs_replay),
            "elapsed_seconds": float(elapsed),
            "maximum_transition_count_per_call": max(
                (
                    int(self._result_tensor(item, "later_head_fraction").numel())
                    for _, _, item in self._batches
                ),
                default=0,
            ),
            "maximum_cuda_memory_allocated": peak_memory,
            "peak_cuda_memory_bytes": peak_memory,
            "total_cuda_memory_bytes": total_memory,
            "packed_extra": {
                name.removeprefix("extra."): values
                for name, values in host.items()
                if name.startswith("extra.")
            },
            "per_row": per_row,
        }
        return dict(self._record)

    def record(self) -> dict[str, Any]:
        if self._record is None:
            raise FusedTangentContractError("reference shard is not finalized")
        return dict(self._record)


class CandidateApproximateFusedReference(DeferredCertifiedFusedReference):
    """Candidate-only fused reference with one shard-boundary host transfer.

    This class mirrors the exact deferred reference's row chunking and prepared
    seed-map contract.  Its active lanes are all explicitly approximate; it
    neither constructs an exact replay path nor emits authorizing certificate
    health.
    """

    reference_contract = CANDIDATE_REFERENCE_CONTRACT

    def __init__(
        self,
        *,
        profile: JacobiRBCudaProfile,
        root_seed: int,
        stream_role: str,
        sampler: Callable[..., Any] | None = None,
        prepared_backend: Any | None = None,
        prepared_rng_seeds: Mapping[str, Any] | None = None,
        lane_cap: int = REFERENCE_LANE_CAP,
        row_chunk_size: int | None = None,
    ) -> None:
        if not isinstance(profile, JacobiRBCudaProfile):
            raise TypeError("profile must be a JacobiRBCudaProfile")
        if (
            int(profile.candidate_modes) != 128
            or int(profile.candidate_bisection_steps) != 56
        ):
            raise FusedTangentContractError(
                "candidate approximate reference requires the frozen "
                "128-mode, 56-bisection profile"
            )
        default_sampler = sampler is None
        if default_sampler:
            from mnist.d0_jacobi_rb_cuda_deferred import (
                enqueue_alpha1_rb_transition_batch_cuda_candidate,
            )

            sampler = enqueue_alpha1_rb_transition_batch_cuda_candidate
        if default_sampler and prepared_backend is None:
            raise FusedTangentContractError(
                "default candidate sampler requires a warm prepared backend"
            )
        if not 1 <= int(lane_cap) <= REFERENCE_LANE_CAP:
            raise FusedTangentContractError("reference lane cap is invalid")
        if row_chunk_size is not None and int(row_chunk_size) <= 0:
            raise FusedTangentContractError("row chunk size must be positive")
        if not callable(sampler):
            raise FusedTangentContractError("candidate CUDA sampler is unavailable")
        exploratory_reference_rng_key(root_seed, stream_role, "contract-check")
        self.profile = profile
        self.root_seed = int(root_seed)
        self.stream_role = stream_role
        self.sampler = sampler
        self.prepared_backend = prepared_backend
        self.lane_cap = int(lane_cap)
        self.row_chunk_size = (
            None if row_chunk_size is None else int(row_chunk_size)
        )
        if self.prepared_backend is None:
            if prepared_rng_seeds not in (None, {}):
                raise FusedTangentContractError(
                    "a prepared RNG seed map requires a prepared CUDA backend"
                )
            self._prepared_rng_seeds = MappingProxyType({})
        else:
            from mnist.d0_jacobi_rb_cuda_deferred import (
                validate_prepared_alpha1_rb_transition_cuda_rng_seed,
            )

            if not isinstance(prepared_rng_seeds, Mapping):
                raise FusedTangentContractError(
                    "prepared CUDA backend requires a prebuilt RNG seed map"
                )
            supplied = dict(prepared_rng_seeds)
            if set(supplied) != set(DEFERRED_REFERENCE_RNG_ROLES):
                raise FusedTangentContractError(
                    "prebuilt RNG seed map has missing or unexpected roles"
                )
            for role in DEFERRED_REFERENCE_RNG_ROLES:
                try:
                    validate_prepared_alpha1_rb_transition_cuda_rng_seed(
                        prepared_seed=supplied[role],
                        rng_key=exploratory_reference_rng_key(
                            self.root_seed, self.stream_role, role
                        ),
                        prepared=self.prepared_backend,
                    )
                except (TypeError, ValueError) as exc:
                    raise FusedTangentContractError(
                        f"prebuilt CUDA RNG seed is invalid for role {role}"
                    ) from exc
            self._prepared_rng_seeds = MappingProxyType(supplied)
        self._batches = []
        self._row_count = None
        self._started = time.perf_counter()
        self._record = None
        self._candidate_expected: list[tuple[Tensor, Tensor, Tensor]] = []

    @staticmethod
    def _reject_authorizing_fields(result: Any) -> None:
        forbidden = {
            "authorized_mask",
            "certified_mask",
            "cuda_certified_mask",
            "certified_count",
            "cuda_authorized_count",
            "certificate_codes",
            "certificate_fraction",
        }
        names = set(result) if isinstance(result, Mapping) else {
            name for name in forbidden if hasattr(result, name)
        }
        diagnostics = (
            result.get("device_diagnostics", result.get("diagnostics", {}))
            if isinstance(result, Mapping)
            else getattr(
                result,
                "device_diagnostics",
                getattr(result, "diagnostics", {}),
            )
        )
        if isinstance(diagnostics, Mapping):
            names.update(diagnostics)
        overlap = sorted(forbidden.intersection(names))
        if overlap:
            raise FusedTangentContractError(
                "candidate result exposes authorizing fields: " + ", ".join(overlap)
            )

    def __call__(
        self,
        *,
        head_fraction: Tensor,
        exposure: Tensor,
        transition_ids: Tensor,
        role: str,
    ) -> Any:
        before = len(self._batches)
        output = super().__call__(
            head_fraction=head_fraction,
            exposure=exposure,
            transition_ids=transition_ids,
            role=role,
        )
        new_batches = self._batches[before:]
        for start, stop, result in new_batches:
            self._reject_authorizing_fields(result)
            self._candidate_expected.append(
                (
                    head_fraction[start:stop].contiguous(),
                    exposure[start:stop].contiguous(),
                    transition_ids[start:stop].contiguous(),
                )
            )
        return output

    def make_synchronous_reference(self) -> CertifiedExploratoryReference:
        raise FusedTangentContractError(
            "candidate approximate reference has no exact replay authority"
        )

    def finalize_shard(
        self, extra_device_tensors: Mapping[str, Tensor]
    ) -> dict[str, Any]:
        if self._record is not None:
            raise FusedTangentContractError("reference shard was finalized twice")
        if len(self._candidate_expected) != len(self._batches):
            raise FusedTangentContractError(
                "candidate reference input ledger is incomplete"
            )
        if extra_device_tensors:
            device = next(iter(extra_device_tensors.values())).device
        elif self._batches:
            later = self._result_tensor(self._batches[0][2], "later_head_fraction")
            if later is None:
                raise FusedTangentContractError(
                    "candidate result lost its scientific output"
                )
            device = later.device
        else:
            raise FusedTangentContractError("candidate reference has no device tensors")

        counter_names = (
            "transition_count",
            "active_count",
            "structural_noop_count",
            "approximation_count",
            "invalid_count",
        )
        counters = {
            name: torch.zeros((), dtype=torch.int64, device=device)
            for name in counter_names
        }
        for name in _FORBIDDEN_CANDIDATE_COUNTS:
            counters[name] = torch.zeros((), dtype=torch.int64, device=device)
        maximum_width = torch.zeros((), dtype=torch.float64, device=device)
        row_count = int(self._row_count or 0)
        row_counters = {
            name: torch.zeros(row_count, dtype=torch.int64, device=device)
            for name in counter_names
        }
        row_maximum_width = torch.zeros(
            row_count, dtype=torch.float64, device=device
        )

        for (start, stop, result), expected in zip(
            self._batches, self._candidate_expected, strict=True
        ):
            self._reject_authorizing_fields(result)
            later = self._result_tensor(result, "later_head_fraction")
            target = self._result_tensor(result, "denoising_target")
            earlier = self._result_tensor(result, "earlier_head_fraction")
            exposure = self._result_tensor(result, "exposure")
            transition_ids = self._result_tensor(result, "transition_ids")
            active = self._result_tensor(result, "active_mask")
            structural_noop = self._result_tensor(result, "structural_noop_mask")
            approximation = self._result_tensor(result, "approximation_mask")
            claimed_valid = self._result_tensor(result, "valid_mask", "validity_mask")
            lower = self._result_tensor(result, "candidate_lower")
            upper = self._result_tensor(result, "candidate_upper")
            required = (
                later,
                target,
                earlier,
                exposure,
                transition_ids,
                active,
                structural_noop,
                approximation,
                claimed_valid,
                lower,
                upper,
            )
            if any(value is None for value in required):
                raise FusedTangentContractError(
                    "candidate result omits an integrity field"
                )
            assert later is not None
            assert target is not None
            assert earlier is not None
            assert exposure is not None
            assert transition_ids is not None
            assert active is not None
            assert structural_noop is not None
            assert approximation is not None
            assert claimed_valid is not None
            assert lower is not None
            assert upper is not None
            expected_x, expected_u, expected_ids = expected
            if any(
                value.shape != expected_x.shape
                for value in (
                    later,
                    target,
                    earlier,
                    exposure,
                    transition_ids,
                    active,
                    structural_noop,
                    approximation,
                    claimed_valid,
                    lower,
                    upper,
                )
            ):
                raise FusedTangentContractError(
                    "candidate result shape differs from its launch"
                )
            input_valid = (
                torch.isfinite(expected_x)
                & torch.isfinite(expected_u)
                & (expected_x >= 0.0)
                & (expected_x <= 1.0)
                & (expected_u >= 0.0)
            )
            expected_active = input_valid & (expected_u > 0.0)
            expected_noop = input_valid & (expected_u == 0.0)
            width = upper - lower
            output_valid = (
                torch.isfinite(later)
                & torch.isfinite(target)
                & torch.isfinite(lower)
                & torch.isfinite(upper)
                & torch.isfinite(width)
                & (later >= 0.0)
                & (later <= 1.0)
                & (lower >= 0.0)
                & (lower <= later)
                & (later <= upper)
                & (upper <= 1.0)
                & (width >= 0.0)
            )
            noop_valid = (
                expected_noop
                & (later == expected_x)
                & (target == 0.0)
                & (lower == expected_x)
                & (upper == expected_x)
            )
            echo_valid = (
                (earlier == expected_x)
                & (exposure == expected_u)
                & (transition_ids == expected_ids)
            )
            mask_valid = (
                (active == expected_active)
                & (structural_noop == expected_noop)
                & (approximation == expected_active)
            )
            independently_valid = (
                input_valid
                & echo_valid
                & mask_valid
                & (noop_valid | (expected_active & output_valid))
            )
            valid = independently_valid & claimed_valid.bool()
            invalid = ~valid
            active_bool = active.bool()
            noop_bool = structural_noop.bool()
            approximation_bool = approximation.bool()

            lanes_per_row = int(later.shape[1])
            counters["transition_count"] += int(later.numel())
            counters["active_count"] += torch.sum(active_bool, dtype=torch.int64)
            counters["structural_noop_count"] += torch.sum(
                noop_bool, dtype=torch.int64
            )
            counters["approximation_count"] += torch.sum(
                approximation_bool, dtype=torch.int64
            )
            counters["invalid_count"] += torch.sum(invalid, dtype=torch.int64)
            row_counters["transition_count"][start:stop] += torch.full(
                (stop - start,), lanes_per_row, dtype=torch.int64, device=device
            )
            row_counters["active_count"][start:stop] += torch.sum(
                active_bool, dim=1, dtype=torch.int64
            )
            row_counters["structural_noop_count"][start:stop] += torch.sum(
                noop_bool, dim=1, dtype=torch.int64
            )
            row_counters["approximation_count"][start:stop] += torch.sum(
                approximation_bool, dim=1, dtype=torch.int64
            )
            row_counters["invalid_count"][start:stop] += torch.sum(
                invalid, dim=1, dtype=torch.int64
            )
            safe_width = torch.where(
                expected_active & output_valid, width, torch.zeros_like(width)
            )
            batch_row_maximum = torch.amax(safe_width, dim=1)
            row_maximum_width[start:stop] = torch.maximum(
                row_maximum_width[start:stop], batch_row_maximum
            )
            maximum_width = torch.maximum(maximum_width, torch.amax(safe_width))

            diagnostics = (
                result.get("device_diagnostics", result.get("diagnostics", {}))
                if isinstance(result, Mapping)
                else getattr(
                    result,
                    "device_diagnostics",
                    getattr(result, "diagnostics", {}),
                )
            )
            if not isinstance(diagnostics, Mapping):
                raise FusedTangentContractError(
                    "candidate result diagnostics are missing or malformed"
                )
            for name in _FORBIDDEN_CANDIDATE_COUNTS:
                if name not in diagnostics:
                    raise FusedTangentContractError(
                        f"candidate result diagnostics omit forbidden counter {name}"
                    )
                value = diagnostics[name]
                if (
                    not isinstance(value, Tensor)
                    or value.ndim != 0
                    or value.dtype != torch.int64
                    or value.device != device
                ):
                    raise FusedTangentContractError(
                        f"candidate forbidden counter {name} is not a scalar "
                        "int64 tensor on the shard device"
                    )
                counters[name] += value

        ordered: list[tuple[str, Tensor]] = []
        ordered.extend(
            (f"reference.{name}", value.reshape(-1))
            for name, value in counters.items()
        )
        ordered.append(("reference.maximum_candidate_bracket_width", maximum_width.reshape(-1)))
        ordered.extend(
            (f"reference_row.{name}", value.reshape(-1))
            for name, value in row_counters.items()
        )
        ordered.append(
            (
                "reference_row.maximum_candidate_bracket_width",
                row_maximum_width.reshape(-1),
            )
        )
        ordered.extend(
            (f"extra.{name}", value.to(device=device).reshape(-1))
            for name, value in sorted(extra_device_tensors.items())
        )
        sizes = {name: int(value.numel()) for name, value in ordered}
        packed = torch.cat(
            [value.to(dtype=torch.float64) for _, value in ordered], dim=0
        ).contiguous()
        # The sole device-to-host transfer for candidate integrity and phase
        # telemetry occurs at the explicit shard boundary.
        host_values = packed.detach().cpu().numpy()
        host: dict[str, list[float]] = {}
        offset = 0
        for name, _ in ordered:
            count = sizes[name]
            host[name] = [
                float(item) for item in host_values[offset : offset + count]
            ]
            offset += count
        scalar = {name: int(host[f"reference.{name}"][0]) for name in counters}
        per_row: list[dict[str, Any]] = []
        for row in range(row_count):
            item = {
                name: int(host[f"reference_row.{name}"][row])
                for name in row_counters
            }
            item["maximum_candidate_bracket_width"] = float(
                host["reference_row.maximum_candidate_bracket_width"][row]
            )
            item["certificate_fraction"] = "not_applicable"
            per_row.append(item)
        forbidden = {
            name: scalar[name] for name in _FORBIDDEN_CANDIDATE_COUNTS
        }
        elapsed = time.perf_counter() - self._started
        total_memory = 0
        peak_memory = 0
        if device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(device))
            total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        self._record = {
            "schema": FUSED_TANGENT_VERSION + "-candidate-reference-shard",
            "reference_contract": CANDIDATE_REFERENCE_CONTRACT,
            "approximation_label": (
                "frozen-128-profile-legendre-cuda-inverse-cdf-candidate;"
                "56-bisection;stateless-philox;no-correct-rounding-or-arb"
            ),
            "candidate_modes": 128,
            "candidate_bisection_steps": 56,
            "root_seed": self.root_seed,
            "rng_namespace": exploratory_reference_rng_key(
                self.root_seed, self.stream_role, "record"
            )[1],
            "stream_role": self.stream_role,
            "variant_in_rng_key": 0,
            **scalar,
            "maximum_candidate_bracket_width": float(
                host["reference.maximum_candidate_bracket_width"][0]
            ),
            "certificate_fraction": "not_applicable",
            "forbidden_counts": forbidden,
            "needs_synchronous_replay": 0,
            "elapsed_seconds": float(elapsed),
            "maximum_transition_count_per_call": max(
                (
                    int(self._result_tensor(item, "later_head_fraction").numel())
                    for _, _, item in self._batches
                ),
                default=0,
            ),
            "maximum_cuda_memory_allocated": peak_memory,
            "peak_cuda_memory_bytes": peak_memory,
            "total_cuda_memory_bytes": total_memory,
            "packed_extra": {
                name.removeprefix("extra."): values
                for name, values in host.items()
                if name.startswith("extra.")
            },
            "per_row": per_row,
        }
        return dict(self._record)


def _require_reference_contract(value: str) -> ReferenceContract:
    if value not in {"certified_exact", "candidate_approximate"}:
        raise FusedTangentContractError("reference contract is invalid")
    return value  # type: ignore[return-value]


def _validate_reverse_sequence(
    sequence: Sequence[tuple[int, int]], *, require_full_shards: bool = False
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(step), int(phase)) for step, phase in sequence)
    if not normalized:
        raise FusedTangentContractError("fused reverse sequence is empty")
    for step, phase in normalized:
        if not 0 <= step < OUTER_STEPS or not 0 <= phase < PHASE_COUNT:
            raise FusedTangentContractError("fused reverse coordinate is invalid")
    for previous, current in zip(normalized, normalized[1:]):
        step, phase = previous
        expected = (step, phase - 1) if phase > 0 else (step - 1, PHASE_COUNT - 1)
        if current != expected:
            raise FusedTangentContractError("fused reverse sequence is not contiguous")
    if require_full_shards and len(normalized) % FUSED_SHARD_PHASES:
        raise FusedTangentContractError(
            "restartable fused family must contain complete eight-step shards"
        )
    return normalized


def _combine_phase_device_telemetry(
    results: Sequence[FusedTangentPhaseResult], rows: int, device: torch.device
) -> dict[str, Tensor]:
    combined: dict[str, Tensor] = {}
    for result in results:
        for name, value in result.telemetry.sums.items():
            key = f"phase.sum.{name}"
            combined[key] = combined.get(key, torch.zeros(rows, dtype=value.dtype, device=device)) + value
        for name, value in result.telemetry.maxima.items():
            key = f"phase.max.{name}"
            combined[key] = torch.maximum(
                combined.get(key, torch.zeros(rows, dtype=value.dtype, device=device)),
                value,
            )
        for name, value in result.telemetry.failure_flags.items():
            key = f"phase.flag.{name}"
            combined[key] = combined.get(
                key, torch.zeros(rows, dtype=torch.bool, device=device)
            ) | value
    return combined


def _host_phase_records(
    packed: Mapping[str, Sequence[float]], specs: Sequence[FusedRowSpec]
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for row, spec in enumerate(specs):
        record: dict[str, Any] = {"row_key": spec.row_key}
        for name, values in packed.items():
            if not name.startswith("phase.") or len(values) != len(specs):
                continue
            short = name.split(".", 2)[2]
            value = values[row]
            record[short] = (
                int(value)
                if name.startswith("phase.flag.")
                or short.endswith("count")
                or short == "transition_count"
                else float(value)
            )
        for prefix in (
            "reference_fraction_displacement",
            "control_fraction_displacement",
            "score",
            "logistic_shift",
        ):
            count = int(record.get(f"{prefix}_count", 0))
            squared = float(record.get(f"{prefix}_squared_sum", 0.0))
            record[f"{prefix}_rms"] = math.sqrt(squared / count) if count else 0.0
        records.append(record)
    return tuple(records)


@dataclass(frozen=True)
class FusedReverseShardResult:
    final_state: np.ndarray = field(repr=False, compare=False)
    row_specs: tuple[FusedRowSpec, ...]
    sequence: tuple[tuple[int, int], ...]
    per_row_diagnostics: tuple[Mapping[str, Any], ...]
    controller_diagnostics: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    elapsed_seconds: float
    transition_count: int
    synchronous_replay_performed: int

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": FUSED_TANGENT_VERSION + "-reverse-shard-result",
            "row_table": [item.to_record() for item in self.row_specs],
            "sequence": [list(item) for item in self.sequence],
            "per_row_diagnostics": [dict(item) for item in self.per_row_diagnostics],
            "controller_diagnostics": [dict(item) for item in self.controller_diagnostics],
            "diagnostics": dict(self.diagnostics),
            "elapsed_seconds": float(self.elapsed_seconds),
            "transition_count": int(self.transition_count),
            "synchronous_replay_performed": int(self.synchronous_replay_performed),
            "final_state_sha256": rollout_array_sha256(self.final_state),
        }


def _execute_fused_shard_device(
    tensor: Tensor,
    sequence: tuple[tuple[int, int], ...],
    *,
    specs: tuple[FusedRowSpec, ...],
    controller_bank: FusedTangentControllerBank,
    reference_transition: Callable[..., Any],
    label: int | Tensor,
    microsteps: int,
    transition_id_plan: FusedTransitionIdPlan,
) -> tuple[Tensor, list[FusedTangentPhaseResult]]:
    if (
        transition_id_plan.sequence != sequence
        or transition_id_plan.microsteps != int(microsteps)
        or transition_id_plan.canonical_path_ids
        != tuple(int(item.canonical_path_id) for item in specs)
        or transition_id_plan.ids.device != tensor.device
        or transition_id_plan.matching_tails.device != tensor.device
        or transition_id_plan.matching_heads.device != tensor.device
    ):
        raise FusedTangentContractError("transition-ID plan does not bind this shard")
    values = tensor
    controller_bank.reset_device_telemetry(values.device)
    phase_results: list[FusedTangentPhaseResult] = []
    with torch.inference_mode():
        for sequence_index, (outer_step, phase) in enumerate(sequence):
            result = controlled_reverse_phase_tangent_fused(
                values,
                outer_step,
                phase,
                microsteps,
                NAMESPACE_VERSION,
                controller_bank=controller_bank,
                reference_transition=reference_transition,
                row_keys=tuple(item.row_key for item in specs),
                canonical_path_ids=tuple(item.canonical_path_id for item in specs),
                label=label,
                prebuilt_transition_ids=transition_id_plan.phase_ids(sequence_index),
                prebuilt_matching_tails=transition_id_plan.matching_tails,
                prebuilt_matching_heads=transition_id_plan.matching_heads,
            )
            values = result.state
            phase_results.append(result)
    return values, phase_results


def run_fused_reverse_shard(
    state: np.ndarray | Tensor,
    sequence: Sequence[tuple[int, int]],
    *,
    row_specs: Sequence[FusedRowSpec],
    controller_bank: FusedTangentControllerBank,
    reference_transition: Callable[..., Any],
    label: int | Tensor = 3,
    microsteps: int = 2,
    device: torch.device | str | None = None,
    transition_id_plan: FusedTransitionIdPlan | None = None,
    reference_contract: ReferenceContract = "certified_exact",
) -> FusedReverseShardResult:
    """Execute at most one fused eight-step shard and validate at its boundary."""

    contract = _require_reference_contract(reference_contract)
    tensor, _ = batched_rollout_state(state, device=device)
    specs = validate_fused_row_specs(row_specs, expected_rows=int(tensor.shape[0]))
    if tuple(controller_bank.row_specs) != specs:
        raise FusedTangentContractError("controller bank row table changed")
    normalized = _validate_reverse_sequence(sequence)
    if len(normalized) > FUSED_SHARD_PHASES:
        raise FusedTangentContractError("fused shard exceeds eight outer steps")
    if int(microsteps) not in {2, 4, 8}:
        raise FusedTangentContractError("microsteps must be 2, 4, or 8")
    id_plan = (
        build_fused_transition_id_plan(
            specs,
            normalized,
            microsteps=int(microsteps),
            device=tensor.device,
        )
        if transition_id_plan is None
        else transition_id_plan
    )
    controller_bank.prepare_device(
        tensor.device,
        matching_tensors=(id_plan.matching_tails, id_plan.matching_heads),
    )
    started = time.perf_counter()
    initial = tensor.clone()
    values, phase_results = _execute_fused_shard_device(
        tensor,
        normalized,
        specs=specs,
        controller_bank=controller_bank,
        reference_transition=reference_transition,
        label=label,
        microsteps=int(microsteps),
        transition_id_plan=id_plan,
    )
    phase_device = _combine_phase_device_telemetry(
        phase_results, len(specs), values.device
    )
    controller_device = {
        f"controller.{name}": value
        for name, value in controller_bank.device_record_tensors().items()
    }
    extras = {**phase_device, **controller_device}
    finalize = getattr(reference_transition, "finalize_shard", None)
    if callable(finalize):
        reference_record = finalize(extras)
        packed = dict(reference_record.get("packed_extra", {}))
    else:
        ordered = sorted(extras.items())
        packed_tensor = torch.cat(
            [value.to(dtype=torch.float64).reshape(-1) for _, value in ordered]
        )
        host_values = packed_tensor.detach().cpu().numpy()
        packed = {}
        offset = 0
        for name, value in ordered:
            count = int(value.numel())
            packed[name] = [float(item) for item in host_values[offset : offset + count]]
            offset += count
        record_method = getattr(reference_transition, "record", None)
        reference_record = dict(record_method()) if callable(record_method) else {
            "transition_count": sum(item.transition_count for item in phase_results),
            "certified_count": sum(item.transition_count for item in phase_results),
            "certificate_fraction": 1.0,
            "fallback_count": 0,
            "fallback_fraction": 0.0,
            "forbidden_counts": {name: 0 for name in _FORBIDDEN_REFERENCE_COUNTS},
            "needs_synchronous_replay": 0,
        }
    replayed = int(reference_record.get("needs_synchronous_replay", 0))
    if contract == "candidate_approximate" and replayed:
        raise FusedTangentContractError(
            "candidate approximate reference requested synchronous replay"
        )
    if replayed:
        maker = getattr(reference_transition, "make_synchronous_reference", None)
        if not callable(maker):
            raise FusedTangentContractError(
                "unresolved speculative shard lacks an exact replay path"
            )
        synchronous = maker()
        values, phase_results = _execute_fused_shard_device(
            initial,
            normalized,
            specs=specs,
            controller_bank=controller_bank,
            reference_transition=synchronous,
            label=label,
            microsteps=int(microsteps),
            transition_id_plan=id_plan,
        )
        phase_device = _combine_phase_device_telemetry(
            phase_results, len(specs), values.device
        )
        controller_device = {
            f"controller.{name}": value
            for name, value in controller_bank.device_record_tensors().items()
        }
        ordered = sorted({**phase_device, **controller_device}.items())
        packed_tensor = torch.cat(
            [value.to(dtype=torch.float64).reshape(-1) for _, value in ordered]
        )
        host_values = packed_tensor.detach().cpu().numpy()
        packed = {}
        offset = 0
        for name, value in ordered:
            count = int(value.numel())
            packed[name] = [float(item) for item in host_values[offset : offset + count]]
            offset += count
        reference_record = synchronous.record()
        reference_record = {
            **reference_record,
            "needs_synchronous_replay": 0,
            "speculative_attempt_discarded": 1,
        }

    phase_rows = _host_phase_records(packed, specs)
    reference_rows = reference_record.get("per_row", ())
    if not isinstance(reference_rows, Sequence) or len(reference_rows) != len(specs):
        if callable(finalize):
            raise FusedTangentContractError(
                "fused reference omitted per-row certificate health"
                if contract == "certified_exact"
                else "candidate reference omitted per-row integrity health"
            )
        per_row_transition_count = sum(
            item.transition_count for item in phase_results
        ) // len(specs)
        reference_rows = tuple(
            {
                "transition_count": int(per_row_transition_count),
                "active_count": int(per_row_transition_count),
                "certified_count": int(per_row_transition_count),
                "fallback_count": 0,
                "unauthorized_count": 0,
                "invalid_count": 0,
                "certificate_fraction": 1.0,
            }
            for _ in specs
        )
    merged_rows: list[dict[str, Any]] = []
    for phase_row, reference_row in zip(
        phase_rows, reference_rows, strict=True
    ):
        if not isinstance(reference_row, Mapping):
            raise FusedTangentContractError(
                "per-row reference health record is malformed"
            )
        merged = dict(phase_row)
        for name, value in reference_row.items():
            merged[f"reference_{name}"] = value
        merged_rows.append(merged)
    per_row = tuple(merged_rows)
    controller_host = {
        name.removeprefix("controller."): values
        for name, values in packed.items()
        if name.startswith("controller.")
    }
    controller_rows = controller_bank.row_records_from_host(controller_host)
    failed_rows = [
        item.row_key
        for item, record in zip(specs, per_row, strict=True)
        if any(
            int(value)
            for name, value in record.items()
            if name in {
                "input_invalid",
                "reference_fraction_invalid",
                "score_invalid",
                "logistic_shift_invalid",
                "state_invalid",
                "mass_invalid",
                "metadata_invalid",
            }
        )
    ]
    forbidden = dict(reference_record.get("forbidden_counts", {}))
    forbidden_names = (
        _FORBIDDEN_REFERENCE_COUNTS
        if contract == "certified_exact"
        else _FORBIDDEN_CANDIDATE_COUNTS
    )
    if failed_rows or any(int(forbidden.get(name, 0)) for name in forbidden_names):
        raise FusedTangentContractError(
            (
                f"fused shard failed exact health validation for rows {failed_rows}"
                if contract == "certified_exact"
                else f"candidate shard failed integrity validation for rows {failed_rows}"
            )
        )
    if contract == "certified_exact":
        if float(reference_record.get("certificate_fraction", 1.0)) != 1.0:
            raise FusedTangentContractError(
                "fused shard certificate fraction is not one"
            )
    else:
        transition_health = int(reference_record.get("transition_count", -1))
        active_health = int(reference_record.get("active_count", -1))
        noop_health = int(reference_record.get("structural_noop_count", -1))
        approximation_health = int(
            reference_record.get("approximation_count", -1)
        )
        invalid_health = int(reference_record.get("invalid_count", -1))
        width_health = float(
            reference_record.get("maximum_candidate_bracket_width", float("nan"))
        )
        expected_transition_health = sum(
            item.transition_count for item in phase_results
        )
        if reference_record.get("reference_contract") != CANDIDATE_REFERENCE_CONTRACT:
            raise FusedTangentContractError(
                "candidate reference contract label is missing or changed"
            )
        if reference_record.get("certificate_fraction") != "not_applicable":
            raise FusedTangentContractError(
                "candidate reference falsely exposes a certificate fraction"
            )
        if (
            transition_health != expected_transition_health
            or active_health < 0
            or noop_health < 0
            or approximation_health != active_health
            or active_health + noop_health != transition_health
            or invalid_health != 0
            or not math.isfinite(width_health)
            or width_health < 0.0
        ):
            raise FusedTangentContractError(
                "candidate reference integrity or approximation labeling failed"
            )
    final = np.ascontiguousarray(values.detach().cpu().numpy(), dtype=np.float64)
    if (
        not np.isfinite(final).all()
        or np.any(final < 0.0)
        or float(np.max(np.abs(np.sum(final, axis=1) - 1.0))) > SIMPLEX_TOLERANCE
    ):
        raise FusedTangentContractError("fused shard final state is invalid")
    transition_count = sum(item.transition_count for item in phase_results)
    aggregate = {
        "reference": reference_record,
        "row_count": len(specs),
        "maximum_launch_lanes": len(specs) * EDGES_PER_PHASE,
        "transition_count": int(transition_count),
        "certificate_fraction": (
            float(reference_record.get("certificate_fraction", 1.0))
            if contract == "certified_exact"
            else "not_applicable"
        ),
        "fallback_count": int(reference_record.get("fallback_count", 0)),
        "forbidden_counts": forbidden,
        "maximum_mass_error": max(
            (
                max(
                    float(record.get("maximum_pair_mass_error", 0.0)),
                    float(record.get("maximum_simplex_mass_error", 0.0)),
                )
                for record in per_row
            ),
            default=0.0,
        ),
    }
    if contract == "candidate_approximate":
        aggregate.update(
            reference_contract=CANDIDATE_REFERENCE_CONTRACT,
            approximation_count=int(reference_record["approximation_count"]),
            invalid_count=int(reference_record["invalid_count"]),
            maximum_candidate_bracket_width=float(
                reference_record["maximum_candidate_bracket_width"]
            ),
        )
    return FusedReverseShardResult(
        final_state=final,
        row_specs=specs,
        sequence=normalized,
        per_row_diagnostics=per_row,
        controller_diagnostics=controller_rows,
        diagnostics=aggregate,
        elapsed_seconds=float(time.perf_counter() - started),
        transition_count=int(transition_count),
        synchronous_replay_performed=replayed,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FusedTangentContractError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise FusedTangentContractError(f"{path} is not a JSON object")
    body = dict(value)
    recorded = body.pop("semantic_sha256", None)
    if not isinstance(recorded, str) or semantic_sha256(body) != recorded:
        raise FusedTangentContractError(f"{path} semantic hash changed")
    return value


@dataclass(frozen=True)
class FusedReverseFamilyResult:
    final_state: np.ndarray = field(repr=False, compare=False)
    saved_states: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    row_specs: tuple[FusedRowSpec, ...]
    per_row_diagnostics: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    elapsed_seconds: float
    transition_count: int
    shard_records: tuple[Mapping[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": FUSED_TANGENT_VERSION + "-reverse-family-result",
            "row_table": [item.to_record() for item in self.row_specs],
            "final_state_sha256": rollout_array_sha256(self.final_state),
            "saved_state_sha256": {
                name: rollout_array_sha256(value)
                for name, value in self.saved_states.items()
            },
            "per_row_diagnostics": [dict(item) for item in self.per_row_diagnostics],
            "diagnostics": dict(self.diagnostics),
            "elapsed_seconds": float(self.elapsed_seconds),
            "transition_count": int(self.transition_count),
            "shard_count": len(self.shard_records),
        }


@dataclass(frozen=True)
class FusedShardExecutionPlan:
    """Resource-accounting view emitted before an uncommitted shard."""

    shard_index: int
    sequence: tuple[tuple[int, int], ...]
    row_count: int
    transition_count: int
    input_state_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": FUSED_TANGENT_VERSION + "-shard-execution-plan",
            "shard_index": int(self.shard_index),
            "sequence": [list(item) for item in self.sequence],
            "row_count": int(self.row_count),
            "transition_count": int(self.transition_count),
            "input_state_sha256": self.input_state_sha256,
        }


def run_fused_reverse_family(
    initial_state: np.ndarray | Tensor,
    *,
    sequence: Sequence[tuple[int, int]],
    output_dir: str | Path,
    family_name: str,
    segment_name: str,
    row_specs: Sequence[FusedRowSpec],
    controller_bank: FusedTangentControllerBank,
    reference_factory: Callable[[int], Callable[..., Any]],
    controller_binding: Mapping[str, Any],
    rng_binding: Mapping[str, Any],
    label: int | Tensor = 3,
    microsteps: int = 2,
    device: torch.device | str | None = None,
    capture_coordinates: Mapping[tuple[int, int], str] | None = None,
    before_uncommitted_shard: Callable[[FusedShardExecutionPlan], None]
    | None = None,
    reference_contract: ReferenceContract = "certified_exact",
) -> FusedReverseFamilyResult:
    """Run/resume a fused family with NPZ-first, JSON-second shard commits."""

    contract = _require_reference_contract(reference_contract)
    family = _safe_text(family_name, "family name")
    segment = _safe_text(segment_name, "segment name")
    normalized = _validate_reverse_sequence(sequence, require_full_shards=True)
    tensor, _ = batched_rollout_state(initial_state, device=device)
    specs = validate_fused_row_specs(row_specs, expected_rows=int(tensor.shape[0]))
    if tuple(controller_bank.row_specs) != specs:
        raise FusedTangentContractError("controller bank row table changed")
    if not callable(reference_factory):
        raise TypeError("reference factory must be callable")
    if before_uncommitted_shard is not None and not callable(
        before_uncommitted_shard
    ):
        raise TypeError("before_uncommitted_shard must be callable")
    controller_hash = semantic_sha256(_json_safe_mapping(controller_binding, "controller binding"))
    rng_hash = semantic_sha256(_json_safe_mapping(rng_binding, "RNG binding"))
    root = Path(output_dir) / "fused_families" / family / segment
    root.mkdir(parents=True, exist_ok=True)
    state = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float64)
    initial_hash = rollout_array_sha256(state)
    captures = dict(capture_coordinates or {})
    for coordinate, name in captures.items():
        if coordinate not in normalized:
            raise FusedTangentContractError("capture coordinate is outside family sequence")
        _safe_text(name, "capture name")
    saved: dict[str, np.ndarray] = {"start": state.copy()}
    records: list[dict[str, Any]] = []

    for shard_index, offset in enumerate(range(0, len(normalized), FUSED_SHARD_PHASES)):
        shard_sequence = normalized[offset : offset + FUSED_SHARD_PHASES]
        state_path = root / f"shard-{shard_index:04d}.npz"
        record_path = root / f"shard-{shard_index:04d}.json"
        input_hash = rollout_array_sha256(state)
        binding = {
            "schema": FUSED_TANGENT_VERSION + "-reverse-shard",
            "schema_version": 1,
            "scheduler_version": FUSED_TANGENT_VERSION,
            "family_name": family,
            "segment_name": segment,
            "shard_index": shard_index,
            "sequence_start": list(shard_sequence[0]),
            "sequence_end": list(shard_sequence[-1]),
            "sequence_sha256": semantic_sha256([list(item) for item in shard_sequence]),
            "row_table": [item.to_record() for item in specs],
            "row_keys": [item.row_key for item in specs],
            "canonical_path_ids": [int(item.canonical_path_id) for item in specs],
            "microsteps": int(microsteps),
            "label": int(label) if not isinstance(label, Tensor) else "tensor",
            "input_state_sha256": input_hash,
            "controller_binding_sha256": controller_hash,
            "rng_binding_sha256": rng_hash,
            "variant_in_rng_key": 0,
        }
        if contract == "candidate_approximate":
            binding["reference_contract"] = CANDIDATE_REFERENCE_CONTRACT
        if record_path.exists():
            if not state_path.exists():
                raise FusedTangentContractError(
                    "committed fused shard lacks its state archive"
                )
            record = _load_json(record_path)
            for name, expected in binding.items():
                if record.get(name) != expected:
                    raise FusedTangentContractError(
                        f"fused restart binding {name} changed"
                    )
            if (
                contract == "certified_exact"
                and "reference_contract" in record
            ):
                raise FusedTangentContractError(
                    "candidate restart prefix cannot be opened as exact"
                )
            if int(record.get("committed", 0)) != 1:
                raise FusedTangentContractError("fused restart record is not committed")
            if record.get("state_file_sha256") != rollout_file_sha256(state_path):
                raise FusedTangentContractError("fused restart NPZ hash changed")
            state = load_rollout_state_npz(state_path, expected_rows=len(specs))
            if record.get("output_state_sha256") != rollout_array_sha256(state):
                raise FusedTangentContractError("fused restart state hash changed")
        else:
            planned_transition_count = (
                len(shard_sequence)
                * 2
                * int(microsteps)
                * len(specs)
                * EDGES_PER_PHASE
            )
            execution_plan = FusedShardExecutionPlan(
                shard_index=shard_index,
                sequence=shard_sequence,
                row_count=len(specs),
                transition_count=planned_transition_count,
                input_state_sha256=input_hash,
            )
            shard_end_to_end_started = time.perf_counter()
            try:
                if before_uncommitted_shard is not None:
                    before_uncommitted_shard(execution_plan)
                reference = reference_factory(shard_index)
                result = run_fused_reverse_shard(
                    torch.as_tensor(
                        np.array(state, copy=True, order="C"),
                        dtype=torch.float64,
                        device=tensor.device,
                    ).contiguous(),
                    shard_sequence,
                    row_specs=specs,
                    controller_bank=controller_bank,
                    reference_transition=reference,
                    label=label,
                    microsteps=microsteps,
                    reference_contract=contract,
                )
                if result.transition_count != planned_transition_count:
                    raise FusedTangentContractError(
                        "fused shard transition count differs from its resource plan"
                    )
            except Exception as exc:
                failure = rollout_semantic_record(
                    {
                        **binding,
                        "execution_plan": execution_plan.to_record(),
                        "committed": 0,
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    }
                )
                atomic_write_json(root / f"shard-{shard_index:04d}.failure.json", failure)
                raise
            state = result.final_state
            atomic_rollout_npz(state_path, {"state": state})
            elapsed_through_npz_commit = (
                time.perf_counter() - shard_end_to_end_started
            )
            record = rollout_semantic_record(
                {
                    **binding,
                    "execution_plan": execution_plan.to_record(),
                    "output_state_sha256": rollout_array_sha256(state),
                    "state_file_sha256": rollout_file_sha256(state_path),
                    "state_file_size": int(state_path.stat().st_size),
                    "execution_elapsed_seconds": result.elapsed_seconds,
                    "elapsed_seconds": elapsed_through_npz_commit,
                    "elapsed_scope": (
                        "pre-resource-callback-through-atomic-npz-commit"
                    ),
                    "transition_count": result.transition_count,
                    "per_row_diagnostics": [
                        dict(item) for item in result.per_row_diagnostics
                    ],
                    "controller_diagnostics": [
                        dict(item) for item in result.controller_diagnostics
                    ],
                    "diagnostics": dict(result.diagnostics),
                    "synchronous_replay_performed": result.synchronous_replay_performed,
                    "committed": 1,
                }
            )
            atomic_write_json(record_path, record)
        records.append(record)
        end_coordinate = tuple(shard_sequence[-1])
        capture_name = captures.get(end_coordinate)
        if capture_name is not None:
            saved[capture_name] = state.copy()
    saved.setdefault("final", state.copy())

    elapsed = math.fsum(float(item.get("elapsed_seconds", 0.0)) for item in records)
    transition_count = sum(int(item.get("transition_count", 0)) for item in records)
    row_records: list[dict[str, Any]] = []
    for row, spec in enumerate(specs):
        record: dict[str, Any] = {"row_key": spec.row_key}
        keys = {
            key
            for shard in records
            for key in shard.get("per_row_diagnostics", [])[row]
            if key != "row_key"
        }
        for key in keys:
            values = [
                shard["per_row_diagnostics"][row].get(key, 0) for shard in records
            ]
            if key.endswith("count") or key in {
                "transition_count",
                "input_invalid",
                "reference_fraction_invalid",
                "score_invalid",
                "logistic_shift_invalid",
                "state_invalid",
                "mass_invalid",
                "metadata_invalid",
            }:
                record[key] = sum(int(value) for value in values)
            elif "maximum" in key:
                record[key] = max(float(value) for value in values)
            elif key.endswith("squared_sum"):
                record[key] = math.fsum(float(value) for value in values)
        row_records.append(record)
    aggregate = {
        "initial_state_sha256": initial_hash,
        "final_state_sha256": rollout_array_sha256(state),
        "restart_chain_valid": 1,
        "shard_count": len(records),
        "row_count": len(specs),
        "transition_count": transition_count,
        "synchronous_replay_count": sum(
            int(item.get("synchronous_replay_performed", 0)) for item in records
        ),
        "maximum_mass_error": max(
            (
                float(item.get("diagnostics", {}).get("maximum_mass_error", 0.0))
                for item in records
            ),
            default=0.0,
        ),
        "certificate_fraction": (
            min(
                float(item.get("diagnostics", {}).get("certificate_fraction", 1.0))
                for item in records
            )
            if records and contract == "certified_exact"
            else (1.0 if contract == "certified_exact" else "not_applicable")
        ),
    }
    if contract == "candidate_approximate":
        aggregate.update(
            reference_contract=CANDIDATE_REFERENCE_CONTRACT,
            approximation_count=sum(
                int(
                    item.get("diagnostics", {})
                    .get("reference", {})
                    .get("approximation_count", 0)
                )
                for item in records
            ),
            invalid_count=sum(
                int(
                    item.get("diagnostics", {})
                    .get("reference", {})
                    .get("invalid_count", 0)
                )
                for item in records
            ),
            maximum_candidate_bracket_width=max(
                (
                    float(
                        item.get("diagnostics", {})
                        .get("reference", {})
                        .get("maximum_candidate_bracket_width", 0.0)
                    )
                    for item in records
                ),
                default=0.0,
            ),
        )
    return FusedReverseFamilyResult(
        final_state=state,
        saved_states=saved,
        row_specs=specs,
        per_row_diagnostics=tuple(row_records),
        diagnostics=aggregate,
        elapsed_seconds=elapsed,
        transition_count=transition_count,
        shard_records=tuple(records),
    )


@dataclass(frozen=True)
class FusedFamilyJoinResult:
    state: np.ndarray = field(repr=False, compare=False)
    row_specs: tuple[FusedRowSpec, ...]
    record: Mapping[str, Any]


def join_fused_family_rows(
    prefix_state: np.ndarray | Tensor,
    prefix_row_specs: Sequence[FusedRowSpec],
    append_state: np.ndarray | Tensor,
    append_row_specs: Sequence[FusedRowSpec],
    *,
    next_coordinate: tuple[int, int],
    bindings: Mapping[str, Any] | None = None,
) -> FusedFamilyJoinResult:
    """Append fresh suffix rows without changing evolved prefix-row bytes."""

    prefix_specs = validate_fused_row_specs(prefix_row_specs)
    append_specs = validate_fused_row_specs(append_row_specs)
    joined_specs = validate_fused_row_specs((*prefix_specs, *append_specs))
    prefix, _ = batched_rollout_state(prefix_state, device="cpu")
    append, _ = batched_rollout_state(append_state, device="cpu")
    if prefix.shape[0] != len(prefix_specs) or append.shape[0] != len(append_specs):
        raise FusedTangentContractError("join states do not match their row tables")
    coordinate = _validate_reverse_sequence((next_coordinate,))[0]
    joined = np.ascontiguousarray(
        torch.cat((prefix, append), dim=0).numpy(), dtype=np.float64
    )
    record = rollout_semantic_record(
        {
            "schema": FUSED_TANGENT_VERSION + "-family-join",
            "schema_version": 1,
            "prefix_row_table": [item.to_record() for item in prefix_specs],
            "append_row_table": [item.to_record() for item in append_specs],
            "joined_row_table": [item.to_record() for item in joined_specs],
            "prefix_state_sha256": rollout_array_sha256(prefix),
            "append_state_sha256": rollout_array_sha256(append),
            "joined_state_sha256": rollout_array_sha256(joined),
            "next_coordinate": list(coordinate),
            "bindings": _json_safe_mapping(bindings or {}, "join bindings"),
        }
    )
    return FusedFamilyJoinResult(joined, joined_specs, record)


def split_fused_family_diagnostics(
    result: FusedReverseFamilyResult,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(result, FusedReverseFamilyResult):
        raise TypeError("result must be a FusedReverseFamilyResult")
    return {
        spec.row_key: dict(record)
        for spec, record in zip(
            result.row_specs, result.per_row_diagnostics, strict=True
        )
    }


__all__ = [
    "build_fused_transition_id_plan",
    "CANDIDATE_REFERENCE_CONTRACT",
    "CandidateApproximateFusedReference",
    "DEFERRED_REFERENCE_RNG_ROLES",
    "DeferredCertifiedFusedReference",
    "FUSED_SHARD_OUTER_STEPS",
    "FUSED_SHARD_PHASES",
    "FUSED_TANGENT_VERSION",
    "FusedControllerKind",
    "FusedFamilyJoinResult",
    "FusedReverseFamilyResult",
    "FusedReverseShardResult",
    "FusedRowSpec",
    "FusedShardExecutionPlan",
    "FusedTangentContractError",
    "FusedTangentControllerBank",
    "FusedTransitionIdPlan",
    "ReferenceContract",
    "fused_transition_ids",
    "join_fused_family_rows",
    "prepare_deferred_reference_rng_seed_map",
    "run_fused_reverse_family",
    "run_fused_reverse_shard",
    "split_fused_family_diagnostics",
    "validate_fused_row_specs",
]
