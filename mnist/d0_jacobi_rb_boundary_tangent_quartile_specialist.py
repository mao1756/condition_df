"""Core contracts for the exact Jacobi/RB quartile-specialist learner.

This module is intentionally limited to model dispatch and training-only
selection arithmetic.  It performs no cache generation, optimization,
bootstrap inference, controller execution, or sampling.

The physical target remains the unmodified binary64 Rao--Blackwell label.
Each of four independent width-32 boundary-tangent experts is supported on
one forward-time quartile.  The only amplitude adjustment is the sealed,
training-only scalar gain for the q2/q3 experts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_boundary_tangent import (
    BoundaryTangentContractError,
    edge_pair_geometry,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    ModelInputs,
    call_model,
    semantic_sha256,
)
from mnist.d0_jacobi_rb_reverse_controller import fractional_coordinate


QUARTILE_SPECIALIST_VERSION = "d0-jacobi-rb-boundary-tangent-quartile-specialist-v1"
QUARTILE_SPECIALIST_MODEL_SCHEMA = QUARTILE_SPECIALIST_VERSION + "-model"
CANDIDATE_IDENTITY_SCHEMA = QUARTILE_SPECIALIST_VERSION + "-candidate"
GAIN_CALIBRATION_SCHEMA = QUARTILE_SPECIALIST_VERSION + "-gain-calibration"
TRAINING_RANK_SCHEMA = QUARTILE_SPECIALIST_VERSION + "-training-rank"
SELECTED_SYSTEM_SCHEMA = QUARTILE_SPECIALIST_VERSION + "-selected-system"

QUARTILE_COUNT = 4
MODEL_WIDTH = 32
CHECKPOINT_UPDATES = tuple(range(100, 4_001, 100))
UPDATE_ZERO = 0
MODEL_SEEDS_BY_QUARTILE = (
    (261_332, 261_333, 261_334),
    (261_335, 261_336, 261_337),
    (261_338, 261_339, 261_340),
    (261_341, 261_342, 261_343),
)
MINIMUM_POSITIVE_FINE_CELLS = 51
FINE_CELL_SHAPE = (7, 8)
Q1_SENTINEL = (4, 7)


class QuartileSpecialistContractError(ValueError):
    """A frozen quartile-specialist contract was violated."""


class NoEligibleQuartileCandidateError(QuartileSpecialistContractError):
    """A quartile has no candidate satisfying the training-only rank rule."""


def _index(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value.__index__())
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    return result


def _quartile(value: Any) -> int:
    result = _index(value, "quartile")
    if not 0 <= result < QUARTILE_COUNT:
        raise QuartileSpecialistContractError("quartile must lie in [0,4)")
    return result


def _sha256(value: Any, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise QuartileSpecialistContractError(f"{name} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True, order=True)
class CandidateIdentity:
    """Canonical identity of one frozen quartile/seed/update checkpoint."""

    quartile: int
    seed: int
    update: int
    schema: str = CANDIDATE_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        quartile = _quartile(self.quartile)
        seed = _index(self.seed, "seed")
        update = _index(self.update, "update")
        if seed not in MODEL_SEEDS_BY_QUARTILE[quartile]:
            raise QuartileSpecialistContractError(
                "candidate seed is not assigned to its quartile"
            )
        if update not in (UPDATE_ZERO,) + CHECKPOINT_UPDATES:
            raise QuartileSpecialistContractError(
                "candidate update is outside the frozen checkpoint grid"
            )
        if self.schema != CANDIDATE_IDENTITY_SCHEMA:
            raise QuartileSpecialistContractError("candidate schema changed")

    @property
    def key(self) -> str:
        return f"q{self.quartile}.seed{self.seed}.update{self.update:04d}"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "quartile": self.quartile,
            "seed": self.seed,
            "update": self.update,
            "key": self.key,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CandidateIdentity:
        if not isinstance(record, Mapping):
            raise QuartileSpecialistContractError("candidate record is malformed")
        candidate = cls(
            quartile=record.get("quartile"),
            seed=record.get("seed"),
            update=record.get("update"),
            schema=record.get("schema"),
        )
        if record.get("key") != candidate.key:
            raise QuartileSpecialistContractError("candidate key changed")
        return candidate


def candidate_identities(*, include_update_zero: bool = True) -> tuple[CandidateIdentity, ...]:
    """Return the canonical quartile, seed, update candidate order."""

    updates = ((UPDATE_ZERO,) if include_update_zero else ()) + CHECKPOINT_UPDATES
    return tuple(
        CandidateIdentity(quartile, seed, update)
        for quartile in range(QUARTILE_COUNT)
        for seed in MODEL_SEEDS_BY_QUARTILE[quartile]
        for update in updates
    )


NONZERO_CANDIDATE_IDENTITIES = candidate_identities(include_update_zero=False)
ALL_CANDIDATE_IDENTITIES = candidate_identities(include_update_zero=True)


def candidate_grid_record() -> dict[str, Any]:
    return {
        "schema": QUARTILE_SPECIALIST_VERSION + "-candidate-grid",
        "ordering": "quartile-seed-update",
        "quartile_count": QUARTILE_COUNT,
        "seeds_by_quartile": [list(values) for values in MODEL_SEEDS_BY_QUARTILE],
        "nonzero_updates": list(CHECKPOINT_UPDATES),
        "nonzero_candidate_count": len(NONZERO_CANDIDATE_IDENTITIES),
        "update_zero_control_count": len(ALL_CANDIDATE_IDENTITIES)
        - len(NONZERO_CANDIDATE_IDENTITIES),
        "candidate_keys": [candidate.key for candidate in ALL_CANDIDATE_IDENTITIES],
    }


CANDIDATE_GRID_SHA256 = semantic_sha256(candidate_grid_record())


def reconstruct_forward_outer_quartile(inputs: ModelInputs) -> Tensor:
    """Recover the forward quartile only through the public coordinate API."""

    if type(inputs) is not ModelInputs:
        raise QuartileSpecialistContractError(
            "quartile reconstruction accepts only exact ModelInputs"
        )
    coordinate = fractional_coordinate(inputs.reverse_time, inputs.phase)
    quartiles = coordinate.forward_outer_quartile.to(dtype=torch.long)
    if quartiles.shape != (inputs.batch_size,) or bool(
        torch.any((quartiles < 0) | (quartiles >= QUARTILE_COUNT))
    ):
        raise QuartileSpecialistContractError("reconstructed quartiles are malformed")
    return quartiles


def _validate_expert_independence(
    experts: Sequence[ZeroBaselineBoundaryTangentPredictor],
) -> None:
    if len(experts) != QUARTILE_COUNT or any(
        not isinstance(expert, ZeroBaselineBoundaryTangentPredictor)
        for expert in experts
    ):
        raise QuartileSpecialistContractError(
            "exactly four zero-baseline boundary-tangent experts are required"
        )
    if len({id(expert) for expert in experts}) != QUARTILE_COUNT:
        raise QuartileSpecialistContractError("quartile expert modules are shared")
    parameter_ids: set[int] = set()
    parameter_storage: set[int] = set()
    for expert in experts:
        for parameter in expert.parameters():
            if id(parameter) in parameter_ids or (
                parameter.numel() and parameter.data_ptr() in parameter_storage
            ):
                raise QuartileSpecialistContractError(
                    "quartile expert parameters are shared"
                )
            parameter_ids.add(id(parameter))
            if parameter.numel():
                parameter_storage.add(parameter.data_ptr())


def _sealed_gains(values: Sequence[float] | None, sealed: bool) -> tuple[float, ...]:
    if not sealed:
        raise QuartileSpecialistContractError("quartile gains are not sealed")
    if values is None or len(values) != QUARTILE_COUNT:
        raise QuartileSpecialistContractError("exactly four sealed gains are required")
    gains = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in gains):
        raise QuartileSpecialistContractError(
            "quartile gains must be finite and positive"
        )
    if gains[:2] != (1.0, 1.0):
        raise QuartileSpecialistContractError("q0/q1 gains must equal exactly one")
    return gains


class QuartileSpecialistBoundaryTangentPredictor(nn.Module):
    """Four independent boundary-tangent experts with disjoint time support."""

    def __init__(
        self,
        experts: Sequence[ZeroBaselineBoundaryTangentPredictor] | None = None,
        *,
        gains: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        gains_sealed: bool = True,
    ) -> None:
        super().__init__()
        active = tuple(
            experts
            if experts is not None
            else (
                ZeroBaselineBoundaryTangentPredictor(),
                ZeroBaselineBoundaryTangentPredictor(),
                ZeroBaselineBoundaryTangentPredictor(),
                ZeroBaselineBoundaryTangentPredictor(),
            )
        )
        _validate_expert_independence(active)
        sealed_gains = _sealed_gains(gains, gains_sealed)
        self.experts = nn.ModuleList(active)
        self.register_buffer(
            "_quartile_gains",
            torch.tensor(sealed_gains, dtype=torch.float64),
            persistent=True,
        )
        self.gains_sealed = True

    @property
    def gains(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._quartile_gains.tolist())

    @property
    def gains_sha256(self) -> str:
        return semantic_sha256(
            {
                "schema": QUARTILE_SPECIALIST_MODEL_SCHEMA,
                "gains": list(self.gains),
                "sealed": 1,
            }
        )

    def _dispatch(self, inputs: ModelInputs, *, apply_gains: bool) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise QuartileSpecialistContractError(
                "specialist accepts only exact permitted ModelInputs"
            )
        quartiles = reconstruct_forward_outer_quartile(inputs)
        output = torch.zeros(
            (inputs.batch_size, EDGES_PER_PHASE),
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )
        for quartile, expert in enumerate(self.experts):
            row_indices = torch.nonzero(quartiles == quartile, as_tuple=False).flatten()
            if row_indices.numel() == 0:
                continue
            prediction = call_model(expert, inputs.index_select(row_indices)).to(
                dtype=torch.float64
            )
            if apply_gains:
                prediction = prediction * self._quartile_gains[quartile]
            output.index_copy_(0, row_indices, prediction)
        mobility = edge_pair_geometry(inputs).mobility
        return torch.where(mobility == 0.0, torch.zeros_like(output), output)

    def raw_prediction(self, inputs: ModelInputs) -> Tensor:
        """Dispatch experts without applying q2/q3 training-only gains."""

        return self._dispatch(inputs, apply_gains=False)

    def forward(self, inputs: ModelInputs) -> Tensor:
        return self._dispatch(inputs, apply_gains=True)


def _target_array(target: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(target, Tensor):
        array = target.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(target, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != EDGES_PER_PHASE
        or array.shape[0] == 0
        or not np.isfinite(array).all()
    ):
        raise QuartileSpecialistContractError(
            "raw targets must be a finite nonempty [N,392] array"
        )
    return np.ascontiguousarray(array)


def _quartile_array(
    coordinates: ModelInputs | np.ndarray | Tensor, *, row_count: int
) -> np.ndarray:
    if type(coordinates) is ModelInputs:
        values = (
            reconstruct_forward_outer_quartile(coordinates)
            .detach()
            .to(device="cpu", dtype=torch.long)
            .numpy()
        )
    elif isinstance(coordinates, Tensor):
        values = coordinates.detach().to(device="cpu", dtype=torch.long).numpy()
    else:
        source = np.asarray(coordinates)
        if not np.issubdtype(source.dtype, np.integer):
            raise QuartileSpecialistContractError("quartile coordinates must be integral")
        values = source.astype(np.int64, copy=False)
    values = np.asarray(values, dtype=np.int64)
    if values.shape != (row_count,) or ((values < 0) | (values >= 4)).any():
        raise QuartileSpecialistContractError("quartile coordinates are malformed")
    return values


def exact_quartile_target_scale(
    target: np.ndarray | Tensor,
    coordinates: ModelInputs | np.ndarray | Tensor,
    quartile: int,
) -> float:
    """Return training-only RMS of raw targets in one forward quartile.

    Values are consumed in canonical C order and accumulated with
    :func:`math.fsum`.
    """

    targets = _target_array(target)
    quartiles = _quartile_array(coordinates, row_count=targets.shape[0])
    selected = targets[quartiles == _quartile(quartile)]
    if selected.size == 0:
        raise QuartileSpecialistContractError("quartile has no training targets")
    squared_sum = math.fsum(float(value) * float(value) for value in selected.flat)
    scale = math.sqrt(squared_sum / selected.size)
    if not math.isfinite(scale) or scale <= 0.0:
        raise QuartileSpecialistContractError(
            "quartile target scale must be finite and positive"
        )
    return scale


def scaled_raw_target_mse(
    prediction: Tensor, target: Tensor, target_scale: float
) -> tuple[Tensor, Tensor]:
    """Return constant-normalized optimizer loss and plain raw-target MSE."""

    scale = float(target_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise QuartileSpecialistContractError("target scale must be finite and positive")
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise QuartileSpecialistContractError("prediction and target shapes differ")
    difference = prediction.to(dtype=torch.float64) - target.to(dtype=torch.float64)
    if not bool(torch.isfinite(difference).all()):
        raise QuartileSpecialistContractError("raw-target MSE is nonfinite")
    raw_mse = torch.mean(difference.square())
    return raw_mse / (scale * scale), raw_mse


GAIN_ELIGIBLE = "eligible"
GAIN_FIXED_UNIT = "fixed_unit_gain"
GAIN_CROSS_TERM_NONFINITE = "cross_term_nonfinite"
GAIN_PREDICTION_ENERGY_NONFINITE = "prediction_energy_nonfinite"
GAIN_CROSS_TERM_NONPOSITIVE = "cross_term_nonpositive"
GAIN_PREDICTION_ENERGY_NONPOSITIVE = "prediction_energy_nonpositive"
GAIN_NONFINITE = "gain_nonfinite"
GAIN_OUTSIDE_OPEN_UNIT = "gain_outside_open_unit_interval"


@dataclass(frozen=True)
class GainCalibrationRecord:
    candidate: CandidateIdentity
    cross_term: float | None
    prediction_energy: float | None
    gain: float | None
    sample_count: int
    eligible: bool
    reason_code: str
    schema: str = GAIN_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateIdentity):
            raise QuartileSpecialistContractError("gain candidate is malformed")
        if _index(self.sample_count, "sample_count") < 0:
            raise QuartileSpecialistContractError("gain sample count is negative")
        if self.schema != GAIN_CALIBRATION_SCHEMA:
            raise QuartileSpecialistContractError("gain schema changed")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate": self.candidate.to_record(),
            "cross_term": self.cross_term,
            "prediction_energy": self.prediction_energy,
            "gain": self.gain,
            "sample_count": self.sample_count,
            "eligible": int(self.eligible),
            "reason_code": self.reason_code,
        }


def fixed_unit_gain_record(candidate: CandidateIdentity) -> GainCalibrationRecord:
    if candidate.quartile not in (0, 1):
        raise QuartileSpecialistContractError("only q0/q1 have fixed unit gain")
    return GainCalibrationRecord(
        candidate=candidate,
        cross_term=None,
        prediction_energy=None,
        gain=1.0,
        sample_count=0,
        eligible=True,
        reason_code=GAIN_FIXED_UNIT,
    )


def gain_record_from_moments(
    candidate: CandidateIdentity,
    *,
    cross_term: float,
    prediction_energy: float,
    sample_count: int,
) -> GainCalibrationRecord:
    """Validate the unique unconstrained scalar gain ``C/P`` without clipping."""

    if candidate.quartile not in (2, 3):
        raise QuartileSpecialistContractError("only q2/q3 gains may be calibrated")
    count = _index(sample_count, "sample_count")
    if count <= 0:
        raise QuartileSpecialistContractError("gain sample count must be positive")
    cross = float(cross_term)
    energy = float(prediction_energy)
    gain: float | None = None
    if not math.isfinite(cross):
        reason = GAIN_CROSS_TERM_NONFINITE
    elif not math.isfinite(energy):
        reason = GAIN_PREDICTION_ENERGY_NONFINITE
    elif cross <= 0.0:
        reason = GAIN_CROSS_TERM_NONPOSITIVE
    elif energy <= 0.0:
        reason = GAIN_PREDICTION_ENERGY_NONPOSITIVE
    else:
        gain = cross / energy
        if not math.isfinite(gain):
            reason = GAIN_NONFINITE
        elif not 0.0 < gain < 1.0:
            reason = GAIN_OUTSIDE_OPEN_UNIT
        else:
            reason = GAIN_ELIGIBLE
    return GainCalibrationRecord(
        candidate=candidate,
        cross_term=cross,
        prediction_energy=energy,
        gain=gain,
        sample_count=count,
        eligible=reason == GAIN_ELIGIBLE,
        reason_code=reason,
    )


def calibrate_training_only_gain(
    candidate: CandidateIdentity,
    raw_target: np.ndarray | Tensor,
    raw_prediction: np.ndarray | Tensor,
) -> GainCalibrationRecord:
    """Compute canonical binary64 ``C`` and ``P`` for a q2/q3 candidate."""

    targets = (
        raw_target.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(raw_target, Tensor)
        else np.asarray(raw_target, dtype=np.float64)
    )
    predictions = (
        raw_prediction.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(raw_prediction, Tensor)
        else np.asarray(raw_prediction, dtype=np.float64)
    )
    if targets.shape != predictions.shape or targets.size == 0:
        raise QuartileSpecialistContractError(
            "gain targets and predictions must have equal nonempty shapes"
        )
    targets = np.ascontiguousarray(targets)
    predictions = np.ascontiguousarray(predictions)
    cross = math.fsum(
        float(target) * float(prediction)
        for target, prediction in zip(targets.flat, predictions.flat, strict=True)
    ) / targets.size
    energy = (
        math.fsum(float(value) * float(value) for value in predictions.flat)
        / predictions.size
    )
    return gain_record_from_moments(
        candidate,
        cross_term=cross,
        prediction_energy=energy,
        sample_count=targets.size,
    )


RANK_ELIGIBLE = "eligible"
RANK_NONFINITE = "rank_metric_nonfinite"
RANK_GAIN_INELIGIBLE = "gain_ineligible"
RANK_POOLED_NONPOSITIVE = "pooled_improvement_nonpositive"
RANK_PHASE_NONPOSITIVE = "phase_marginal_nonpositive"
RANK_MIDPOINT_NONPOSITIVE = "midpoint_marginal_nonpositive"
RANK_FINE_CELLS_INSUFFICIENT = "positive_fine_cells_below_51"
RANK_Q1_SENTINEL_NONPOSITIVE = "q1_phase4_midpoint7_nonpositive"


@dataclass(frozen=True)
class TrainingRankRecord:
    candidate: CandidateIdentity
    gain_record: GainCalibrationRecord
    pooled_improvement: float
    phase_improvements: tuple[float, ...]
    midpoint_improvements: tuple[float, ...]
    fine_cell_improvements: tuple[tuple[float, ...], ...]
    positive_fine_cells: int
    eligible: bool
    reason_code: str
    schema: str = TRAINING_RANK_SCHEMA

    def __post_init__(self) -> None:
        if self.gain_record.candidate != self.candidate:
            raise QuartileSpecialistContractError("rank gain candidate changed")
        if len(self.phase_improvements) != 7 or len(self.midpoint_improvements) != 8:
            raise QuartileSpecialistContractError("rank marginals have wrong shape")
        if len(self.fine_cell_improvements) != 7 or any(
            len(row) != 8 for row in self.fine_cell_improvements
        ):
            raise QuartileSpecialistContractError("rank fine cells have wrong shape")
        if self.schema != TRAINING_RANK_SCHEMA:
            raise QuartileSpecialistContractError("training-rank schema changed")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate": self.candidate.to_record(),
            "gain": self.gain_record.to_record(),
            "pooled_improvement": self.pooled_improvement,
            "phase_improvements": list(self.phase_improvements),
            "midpoint_improvements": list(self.midpoint_improvements),
            "fine_cell_improvements": [
                list(row) for row in self.fine_cell_improvements
            ],
            "positive_fine_cells": self.positive_fine_cells,
            "eligible": int(self.eligible),
            "reason_code": self.reason_code,
        }


def build_training_rank_record(
    candidate: CandidateIdentity,
    gain_record: GainCalibrationRecord,
    *,
    pooled_improvement: float,
    phase_improvements: Sequence[float],
    midpoint_improvements: Sequence[float],
    fine_cell_improvements: Sequence[Sequence[float]],
) -> TrainingRankRecord:
    phases = tuple(float(value) for value in phase_improvements)
    midpoints = tuple(float(value) for value in midpoint_improvements)
    cells = tuple(tuple(float(value) for value in row) for row in fine_cell_improvements)
    values = (float(pooled_improvement),) + phases + midpoints + tuple(
        value for row in cells for value in row
    )
    if len(phases) != 7 or len(midpoints) != 8 or len(cells) != 7 or any(
        len(row) != 8 for row in cells
    ):
        raise QuartileSpecialistContractError("training-rank metrics have wrong shape")
    positive_cells = sum(value > 0.0 for row in cells for value in row)
    if not all(math.isfinite(value) for value in values):
        reason = RANK_NONFINITE
    elif not gain_record.eligible or gain_record.candidate != candidate:
        reason = RANK_GAIN_INELIGIBLE
    elif float(pooled_improvement) <= 0.0:
        reason = RANK_POOLED_NONPOSITIVE
    elif any(value <= 0.0 for value in phases):
        reason = RANK_PHASE_NONPOSITIVE
    elif any(value <= 0.0 for value in midpoints):
        reason = RANK_MIDPOINT_NONPOSITIVE
    elif positive_cells < MINIMUM_POSITIVE_FINE_CELLS:
        reason = RANK_FINE_CELLS_INSUFFICIENT
    elif candidate.quartile == 1 and cells[Q1_SENTINEL[0]][Q1_SENTINEL[1]] <= 0.0:
        reason = RANK_Q1_SENTINEL_NONPOSITIVE
    else:
        reason = RANK_ELIGIBLE
    return TrainingRankRecord(
        candidate=candidate,
        gain_record=gain_record,
        pooled_improvement=float(pooled_improvement),
        phase_improvements=phases,
        midpoint_improvements=midpoints,
        fine_cell_improvements=cells,
        positive_fine_cells=positive_cells,
        eligible=reason == RANK_ELIGIBLE,
        reason_code=reason,
    )


def select_training_rank_candidate(
    records: Sequence[TrainingRankRecord], quartile: int
) -> TrainingRankRecord:
    """Select one quartile independently: improvement, update, then seed."""

    active_quartile = _quartile(quartile)
    eligible = [
        record
        for record in records
        if isinstance(record, TrainingRankRecord)
        and record.candidate.quartile == active_quartile
        and record.eligible
    ]
    if not eligible:
        raise NoEligibleQuartileCandidateError(
            f"q{active_quartile} has no eligible training-only candidate"
        )
    return min(
        eligible,
        key=lambda record: (
            -record.pooled_improvement,
            record.candidate.update,
            record.candidate.seed,
        ),
    )


def select_training_rank_system(
    records: Sequence[TrainingRankRecord],
) -> tuple[TrainingRankRecord, ...]:
    """Apply the separable rank rule once per disjoint quartile support."""

    return tuple(
        select_training_rank_candidate(records, quartile)
        for quartile in range(QUARTILE_COUNT)
    )


@dataclass(frozen=True)
class HashBinding:
    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise QuartileSpecialistContractError("hash binding name is empty")
        _sha256(self.sha256, f"{self.name}.sha256")

    def to_record(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True)
class SelectedExpert:
    candidate: CandidateIdentity
    checkpoint_path: str
    checkpoint_sha256: str
    model_state_sha256: str
    target_scale: float
    gain: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateIdentity):
            raise QuartileSpecialistContractError("selected candidate is malformed")
        if self.candidate.update == 0:
            raise QuartileSpecialistContractError("selected expert must be nonzero")
        if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path:
            raise QuartileSpecialistContractError("selected checkpoint path is empty")
        _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _sha256(self.model_state_sha256, "model_state_sha256")
        scale = float(self.target_scale)
        gain = float(self.gain)
        if not math.isfinite(scale) or scale <= 0.0:
            raise QuartileSpecialistContractError("selected target scale is invalid")
        if self.candidate.quartile in (0, 1):
            if gain != 1.0:
                raise QuartileSpecialistContractError("selected q0/q1 gain changed")
        elif not math.isfinite(gain) or not 0.0 < gain < 1.0:
            raise QuartileSpecialistContractError("selected q2/q3 gain is inadmissible")

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_record(),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_state_sha256": self.model_state_sha256,
            "target_scale": float(self.target_scale),
            "gain": float(self.gain),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SelectedExpert:
        if not isinstance(record, Mapping):
            raise QuartileSpecialistContractError("selected expert record is malformed")
        return cls(
            candidate=CandidateIdentity.from_record(record.get("candidate", {})),
            checkpoint_path=record.get("checkpoint_path"),
            checkpoint_sha256=record.get("checkpoint_sha256"),
            model_state_sha256=record.get("model_state_sha256"),
            target_scale=record.get("target_scale"),
            gain=record.get("gain"),
        )


@dataclass(frozen=True)
class SelectedSystem:
    experts: tuple[SelectedExpert, ...]
    candidate_grid_sha256: str
    gain_table_sha256: str
    rank_table_sha256: str
    role_open_bindings: tuple[HashBinding, ...]
    schema: str = SELECTED_SYSTEM_SCHEMA

    def __post_init__(self) -> None:
        if len(self.experts) != QUARTILE_COUNT or tuple(
            expert.candidate.quartile for expert in self.experts
        ) != tuple(range(QUARTILE_COUNT)):
            raise QuartileSpecialistContractError(
                "selected experts must be ordered q0 through q3"
            )
        if self.candidate_grid_sha256 != CANDIDATE_GRID_SHA256:
            raise QuartileSpecialistContractError("candidate-grid hash changed")
        _sha256(self.gain_table_sha256, "gain_table_sha256")
        _sha256(self.rank_table_sha256, "rank_table_sha256")
        if not self.role_open_bindings:
            raise QuartileSpecialistContractError("role-open bindings are absent")
        names = tuple(binding.name for binding in self.role_open_bindings)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise QuartileSpecialistContractError(
                "role-open bindings must be unique and sorted"
            )
        if self.schema != SELECTED_SYSTEM_SCHEMA:
            raise QuartileSpecialistContractError("selected-system schema changed")

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experts": [expert.to_record() for expert in self.experts],
            "candidate_grid_sha256": self.candidate_grid_sha256,
            "gain_table_sha256": self.gain_table_sha256,
            "rank_table_sha256": self.rank_table_sha256,
            "role_open_bindings": [
                binding.to_record() for binding in self.role_open_bindings
            ],
        }

    @property
    def semantic_sha256(self) -> str:
        return semantic_sha256(self.semantic_payload())

    def to_record(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "semantic_sha256": self.semantic_sha256}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> SelectedSystem:
        if not isinstance(record, Mapping):
            raise QuartileSpecialistContractError("selected-system record is malformed")
        experts_raw = record.get("experts")
        bindings_raw = record.get("role_open_bindings")
        if not isinstance(experts_raw, list) or not isinstance(bindings_raw, list):
            raise QuartileSpecialistContractError("selected-system lists are malformed")
        system = cls(
            experts=tuple(SelectedExpert.from_record(value) for value in experts_raw),
            candidate_grid_sha256=record.get("candidate_grid_sha256"),
            gain_table_sha256=record.get("gain_table_sha256"),
            rank_table_sha256=record.get("rank_table_sha256"),
            role_open_bindings=tuple(
                HashBinding(value.get("name"), value.get("sha256"))
                for value in bindings_raw
                if isinstance(value, Mapping)
            ),
            schema=record.get("schema"),
        )
        if record.get("semantic_sha256") != system.semantic_sha256:
            raise QuartileSpecialistContractError(
                "selected-system semantic fingerprint changed"
            )
        return system


__all__ = [
    "ALL_CANDIDATE_IDENTITIES",
    "CANDIDATE_GRID_SHA256",
    "CHECKPOINT_UPDATES",
    "CandidateIdentity",
    "GAIN_CROSS_TERM_NONFINITE",
    "GAIN_CROSS_TERM_NONPOSITIVE",
    "GAIN_ELIGIBLE",
    "GAIN_FIXED_UNIT",
    "GAIN_NONFINITE",
    "GAIN_OUTSIDE_OPEN_UNIT",
    "GAIN_PREDICTION_ENERGY_NONFINITE",
    "GAIN_PREDICTION_ENERGY_NONPOSITIVE",
    "GainCalibrationRecord",
    "HashBinding",
    "MINIMUM_POSITIVE_FINE_CELLS",
    "MODEL_SEEDS_BY_QUARTILE",
    "NONZERO_CANDIDATE_IDENTITIES",
    "NoEligibleQuartileCandidateError",
    "QUARTILE_SPECIALIST_VERSION",
    "QuartileSpecialistBoundaryTangentPredictor",
    "QuartileSpecialistContractError",
    "RANK_ELIGIBLE",
    "RANK_FINE_CELLS_INSUFFICIENT",
    "RANK_GAIN_INELIGIBLE",
    "RANK_MIDPOINT_NONPOSITIVE",
    "RANK_NONFINITE",
    "RANK_PHASE_NONPOSITIVE",
    "RANK_POOLED_NONPOSITIVE",
    "RANK_Q1_SENTINEL_NONPOSITIVE",
    "SelectedExpert",
    "SelectedSystem",
    "TrainingRankRecord",
    "build_training_rank_record",
    "calibrate_training_only_gain",
    "candidate_grid_record",
    "candidate_identities",
    "exact_quartile_target_scale",
    "fixed_unit_gain_record",
    "gain_record_from_moments",
    "reconstruct_forward_outer_quartile",
    "scaled_raw_target_mse",
    "select_training_rank_candidate",
    "select_training_rank_system",
]
