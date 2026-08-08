"""Boundary-tangent parameterization for the exact Jacobi/RB target.

This module changes only the coordinates used to represent the conditional
mean ``m = E[L-MY | later state]``.  The model predicts a finite coefficient
``q`` and returns ``m = y(1-y) q``.  Training remains direct, unweighted MSE
against the unchanged binary64 Rao--Blackwell target.

There is deliberately no sampler or orchestration code here.  In particular,
no quotient target, clipping, floor, limiter, projection, or renormalization
is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_coarse_residual import zero_initialize_residual
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    JacobiRBPhasePredictor,
    ModelInputs,
    call_model,
    configure_exact_synthetic_teacher,
    matching_indices,
    semantic_sha256,
    synthetic_teacher_target,
)
from mnist.d0_jacobi_rb_reverse_controller import (
    ALLOWED_FRACTIONAL_COORDINATES,
    FRACTION_TOLERANCE,
    MIDPOINT_FRACTIONS,
    fractional_coordinate,
)
from mnist import d0_jacobi_rb_reverse_controller as _reverse_controller


BOUNDARY_TANGENT_VERSION = "d0-jacobi-rb-boundary-tangent-v1"
TANGENT_BASELINE_SCHEMA = BOUNDARY_TANGENT_VERSION + "-baseline"
TANGENT_BASELINE_FILE_SCHEMA = BOUNDARY_TANGENT_VERSION + "-baseline-file"
TANGENT_INTERPOLATION_RULE = "piecewise-linear-in-q-over-M8-midpoints-v1"
TANGENT_QUARTILE_RULE = "k//128-reconstructed-solely-from-reverse-time-and-phase-v1"

TIME_QUARTILES = 4
M8_KNOTS = tuple(float(value) for value in MIDPOINT_FRACTIONS[8])
TANGENT_BASELINE_SHAPE = (TIME_QUARTILES, PHASE_COUNT, 8, EDGES_PER_PHASE)


class BoundaryTangentContractError(ValueError):
    """A boundary-tangent mathematical or artifact contract was violated."""


def _float64_c_order_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype != np.float64:
        raise BoundaryTangentContractError("scientific hash requires float64 values")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(
    value: Any, *, dtype: np.dtype[Any], shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape or array.size == 0:
        raise BoundaryTangentContractError(
            f"{name} must have dtype {dtype.str} and shape {shape}"
        )
    if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
        raise BoundaryTangentContractError(f"{name} contains nonfinite values")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _hash_string(value: str, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise BoundaryTangentContractError(f"{name} is not a lowercase SHA-256")
    return value


def _tensor_bundle_sha256(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class EdgePairGeometry:
    """Head-oriented pair geometry extracted from permitted model inputs."""

    tail_mass: Tensor
    head_mass: Tensor
    pair_mass: Tensor
    head_fraction: Tensor
    mobility: Tensor
    active: Tensor


def edge_pair_geometry(inputs: ModelInputs) -> EdgePairGeometry:
    """Return ``r``, ``y`` and ``mu=y(1-y)`` for the active matching.

    The computation uses only ``later_full_state``, ``phase`` and ``color``.
    Zero-pair-mass edges receive exactly zero fraction and mobility.
    """

    if type(inputs) is not ModelInputs:
        raise BoundaryTangentContractError("geometry requires exact ModelInputs")
    state = inputs.later_full_state
    if not bool(torch.isfinite(state).all()) or bool(torch.any(state < 0.0)):
        raise BoundaryTangentContractError("later state is not finite and nonnegative")
    phases = inputs.phase.to(dtype=torch.long)
    colors = inputs.color.to(dtype=torch.long)
    if bool(torch.any((phases < 0) | (phases >= PHASE_COUNT))):
        raise BoundaryTangentContractError("phase is outside the split chain")
    expected_colors = torch.as_tensor(
        PHASE_MATCHINGS, dtype=torch.long, device=state.device
    )[phases]
    if not torch.equal(colors, expected_colors):
        raise BoundaryTangentContractError("color does not match phase")
    expected_durations = torch.as_tensor(
        PHASE_DURATIONS, dtype=inputs.duration.dtype, device=state.device
    )[phases]
    if not torch.equal(inputs.duration, expected_durations):
        raise BoundaryTangentContractError("duration does not match phase")

    tails, heads = matching_indices(device=state.device)
    active_tails = tails[colors]
    active_heads = heads[colors]
    state64 = state.to(dtype=torch.float64)
    tail = state64.gather(1, active_tails)
    head = state64.gather(1, active_heads)
    pair = tail + head
    active = pair > 0.0
    fraction = torch.zeros_like(pair)
    fraction[active] = head[active] / pair[active]
    mobility = torch.zeros_like(pair)
    mobility[active] = fraction[active] * (1.0 - fraction[active])
    return EdgePairGeometry(tail, head, pair, fraction, mobility, active)


def _cell_coordinates(inputs: ModelInputs, *, require_m8: bool) -> tuple[Tensor, Tensor, Tensor]:
    coordinate = fractional_coordinate(inputs.reverse_time, inputs.phase)
    fraction = coordinate.within_phase_fraction
    knots = torch.as_tensor(M8_KNOTS, dtype=torch.float64, device=fraction.device)
    error = torch.abs(fraction[:, None] - knots[None, :])
    knot_index = torch.argmin(error, dim=1)
    if require_m8 and bool(torch.any(torch.min(error, dim=1).values != 0.0)):
        raise BoundaryTangentContractError(
            "baseline fitting requires exact M8 midpoint coordinates"
        )
    return (
        coordinate.forward_outer_quartile.to(dtype=torch.long),
        inputs.phase.to(dtype=torch.long),
        knot_index.to(dtype=torch.long),
    )


@dataclass(frozen=True)
class TangentBaseline:
    """Training-only least-squares coefficient table ``q_B``."""

    q_values: np.ndarray = field(repr=False, compare=False)
    numerators: np.ndarray = field(repr=False, compare=False)
    denominators: np.ndarray = field(repr=False, compare=False)
    counts: np.ndarray = field(repr=False, compare=False)
    training_path_ids: np.ndarray = field(repr=False, compare=False)
    training_inputs_sha256: str
    training_targets_sha256: str
    training_row_path_ids_sha256: str
    interpolation_rule: str = TANGENT_INTERPOLATION_RULE
    schema: str = TANGENT_BASELINE_SCHEMA

    def __post_init__(self) -> None:
        q_values = _readonly(
            self.q_values,
            dtype=np.dtype(np.float64),
            shape=TANGENT_BASELINE_SHAPE,
            name="q_values",
        )
        numerators = _readonly(
            self.numerators,
            dtype=np.dtype(np.float64),
            shape=TANGENT_BASELINE_SHAPE,
            name="numerators",
        )
        denominators = _readonly(
            self.denominators,
            dtype=np.dtype(np.float64),
            shape=TANGENT_BASELINE_SHAPE,
            name="denominators",
        )
        counts = _readonly(
            self.counts,
            dtype=np.dtype(np.int64),
            shape=TANGENT_BASELINE_SHAPE,
            name="counts",
        )
        paths_raw = np.asarray(self.training_path_ids)
        if paths_raw.ndim != 1 or paths_raw.dtype != np.int64:
            raise BoundaryTangentContractError("training_path_ids must be int64")
        paths = np.ascontiguousarray(paths_raw)
        if (
            paths.size == 0
            or np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths))
            or ((paths < 0) | (paths >= (1 << 20))).any()
        ):
            raise BoundaryTangentContractError("training path IDs are not canonical")
        paths.setflags(write=False)
        if (denominators <= 0.0).any() or (counts <= 0).any():
            raise BoundaryTangentContractError(
                "every tangent-baseline denominator/count must be positive"
            )
        if not np.array_equal(
            counts,
            np.broadcast_to(counts[..., :1], TANGENT_BASELINE_SHAPE),
        ):
            raise BoundaryTangentContractError(
                "tangent-baseline counts must be constant across each edge cell"
            )
        expected = np.ascontiguousarray(numerators / denominators)
        if not np.array_equal(q_values, expected):
            raise BoundaryTangentContractError("q_B formula changed")
        if self.schema != TANGENT_BASELINE_SCHEMA:
            raise BoundaryTangentContractError("tangent-baseline schema changed")
        if self.interpolation_rule != TANGENT_INTERPOLATION_RULE:
            raise BoundaryTangentContractError("baseline interpolation rule changed")
        for name in (
            "training_inputs_sha256",
            "training_targets_sha256",
            "training_row_path_ids_sha256",
        ):
            _hash_string(str(getattr(self, name)), name)
        object.__setattr__(self, "q_values", q_values)
        object.__setattr__(self, "numerators", numerators)
        object.__setattr__(self, "denominators", denominators)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "training_path_ids", paths)

    @property
    def q_values_sha256(self) -> str:
        return _float64_c_order_sha256(self.q_values)

    def to_record(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "schema_version": 1,
            "shape": list(TANGENT_BASELINE_SHAPE),
            "dtype": self.q_values.dtype.str,
            "formula": "q_B=sum_train(mu*Zbar)/sum_train(mu^2)",
            "interpolation_rule": self.interpolation_rule,
            "quartile_rule": TANGENT_QUARTILE_RULE,
            "m8_knots": list(M8_KNOTS),
            "q_values_sha256": self.q_values_sha256,
            "numerators_sha256": _array_sha256(self.numerators),
            "denominators_sha256": _array_sha256(self.denominators),
            "counts_sha256": _array_sha256(self.counts),
            "training_path_ids": self.training_path_ids.tolist(),
            "training_row_count": int(np.sum(self.counts[..., 0], dtype=np.int64)),
            "training_inputs_sha256": self.training_inputs_sha256,
            "training_targets_sha256": self.training_targets_sha256,
            "training_row_path_ids_sha256": self.training_row_path_ids_sha256,
            "fit_role": "training_only",
            "raw_target": "unchanged exact Rao-Blackwell E[L-MY|X,Y,u]",
            "quotient_target_persisted": 0,
            "target_modified": 0,
        }
        return {**body, "semantic_sha256": semantic_sha256(body)}

    @property
    def fingerprint(self) -> str:
        return str(self.to_record()["semantic_sha256"])


def derive_tangent_baseline(
    inputs: ModelInputs,
    exact_target: Tensor,
    row_path_ids: Tensor | np.ndarray | Sequence[int],
) -> TangentBaseline:
    """Fit ``q_B`` by direct training-only least squares.

    No quotient target is formed.  Every one of the 4x7x8x392 denominator
    cells must be strictly positive.
    """

    if type(inputs) is not ModelInputs:
        raise BoundaryTangentContractError("baseline fit requires exact ModelInputs")
    if (
        not isinstance(exact_target, Tensor)
        or exact_target.dtype != torch.float64
        or exact_target.shape != (inputs.batch_size, EDGES_PER_PHASE)
        or exact_target.device != inputs.later_full_state.device
        or not bool(torch.isfinite(exact_target).all())
    ):
        raise BoundaryTangentContractError("exact target must be finite binary64 [B,392]")
    paths = (
        row_path_ids.detach().cpu().numpy()
        if isinstance(row_path_ids, Tensor)
        else np.asarray(row_path_ids)
    )
    if paths.shape != (inputs.batch_size,) or paths.dtype.kind not in "iu":
        raise BoundaryTangentContractError("row path IDs must align with training rows")
    paths64 = np.ascontiguousarray(paths, dtype=np.int64)
    if ((paths64 < 0) | (paths64 >= (1 << 20))).any():
        raise BoundaryTangentContractError("training path ID is outside 20 bits")

    quartile, phase, midpoint = _cell_coordinates(inputs, require_m8=True)
    mobility = edge_pair_geometry(inputs).mobility.detach().cpu().numpy()
    target = exact_target.detach().cpu().numpy()
    q_np = quartile.detach().cpu().numpy()
    p_np = phase.detach().cpu().numpy()
    m_np = midpoint.detach().cpu().numpy()
    numerator = np.zeros(TANGENT_BASELINE_SHAPE, dtype=np.float64)
    denominator = np.zeros(TANGENT_BASELINE_SHAPE, dtype=np.float64)
    counts = np.zeros(TANGENT_BASELINE_SHAPE, dtype=np.int64)
    for q_index in range(TIME_QUARTILES):
        for phase_index in range(PHASE_COUNT):
            for midpoint_index in range(8):
                selected = (
                    (q_np == q_index)
                    & (p_np == phase_index)
                    & (m_np == midpoint_index)
                )
                if not np.any(selected):
                    raise BoundaryTangentContractError(
                        "training rows do not populate every tangent-baseline cell"
                    )
                mu = np.ascontiguousarray(mobility[selected], dtype=np.float64)
                zbar = np.ascontiguousarray(target[selected], dtype=np.float64)
                numerator[q_index, phase_index, midpoint_index] = np.sum(
                    mu * zbar, axis=0, dtype=np.float64
                )
                denominator[q_index, phase_index, midpoint_index] = np.sum(
                    mu * mu, axis=0, dtype=np.float64
                )
                counts[q_index, phase_index, midpoint_index] = int(selected.sum())
    if (
        not np.isfinite(numerator).all()
        or not np.isfinite(denominator).all()
        or (denominator <= 0.0).any()
    ):
        raise BoundaryTangentContractError(
            "tangent-baseline denominator is nonpositive or nonfinite"
        )
    q_values = np.ascontiguousarray(numerator / denominator)
    input_hash = _tensor_bundle_sha256(
        {
            "later_full_state": inputs.later_full_state,
            "reverse_time": inputs.reverse_time,
            "phase": inputs.phase,
            "color": inputs.color,
            "duration": inputs.duration,
            "label": inputs.label,
        }
    )
    target_hash = _array_sha256(np.ascontiguousarray(target))
    path_hash = _array_sha256(paths64)
    return TangentBaseline(
        q_values=q_values,
        numerators=numerator,
        denominators=denominator,
        counts=counts,
        training_path_ids=np.unique(paths64),
        training_inputs_sha256=input_hash,
        training_targets_sha256=target_hash,
        training_row_path_ids_sha256=path_hash,
    )


def save_tangent_baseline(path: str | Path, baseline: TangentBaseline) -> dict[str, Any]:
    if not isinstance(baseline, TangentBaseline):
        raise BoundaryTangentContractError("baseline has the wrong type")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": TANGENT_BASELINE_FILE_SCHEMA,
        "schema_version": 1,
        "baseline_record": baseline.to_record(),
    }
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                q_values=baseline.q_values,
                numerators=baseline.numerators,
                denominators=baseline.denominators,
                counts=baseline.counts,
                training_path_ids=baseline.training_path_ids,
                metadata_json=np.frombuffer(encoded, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(target),
        "size": int(target.stat().st_size),
        "sha256": _file_sha256(target),
        "baseline_semantic_sha256": baseline.fingerprint,
        "q_values_sha256": baseline.q_values_sha256,
    }


def load_tangent_baseline(
    path: str | Path, *, expected_sha256: str | None = None
) -> TangentBaseline:
    source = Path(path)
    if expected_sha256 is not None and _file_sha256(source) != expected_sha256:
        raise BoundaryTangentContractError("baseline file fingerprint mismatch")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {
                "q_values",
                "numerators",
                "denominators",
                "counts",
                "training_path_ids",
                "metadata_json",
            }:
                raise BoundaryTangentContractError("baseline archive fields changed")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise BoundaryTangentContractError("cannot load tangent baseline") from exc
    encoded = arrays.pop("metadata_json")
    if encoded.dtype != np.uint8 or encoded.ndim != 1:
        raise BoundaryTangentContractError("baseline metadata encoding changed")
    try:
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BoundaryTangentContractError("baseline metadata is invalid") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != TANGENT_BASELINE_FILE_SCHEMA
        or not isinstance(metadata.get("baseline_record"), dict)
    ):
        raise BoundaryTangentContractError("baseline metadata schema changed")
    record = metadata["baseline_record"]
    result = TangentBaseline(
        q_values=arrays["q_values"],
        numerators=arrays["numerators"],
        denominators=arrays["denominators"],
        counts=arrays["counts"],
        training_path_ids=arrays["training_path_ids"],
        training_inputs_sha256=str(record.get("training_inputs_sha256", "")),
        training_targets_sha256=str(record.get("training_targets_sha256", "")),
        training_row_path_ids_sha256=str(
            record.get("training_row_path_ids_sha256", "")
        ),
        interpolation_rule=str(record.get("interpolation_rule", "")),
        schema=str(record.get("schema", "")),
    )
    if (
        record.get("q_values_sha256") != result.q_values_sha256
        or record.get("numerators_sha256") != _array_sha256(result.numerators)
        or record.get("denominators_sha256") != _array_sha256(result.denominators)
        or record.get("counts_sha256") != _array_sha256(result.counts)
        or record != result.to_record()
    ):
        raise BoundaryTangentContractError("baseline archive metadata was tampered")
    return result


def interpolate_tangent_baseline(baseline: TangentBaseline, inputs: ModelInputs) -> Tensor:
    """Piecewise-linearly interpolate ``q_B`` over the frozen M8 knots."""

    if not isinstance(baseline, TangentBaseline) or type(inputs) is not ModelInputs:
        raise BoundaryTangentContractError("baseline lookup arguments are invalid")
    values = torch.as_tensor(
        np.array(baseline.q_values, copy=True),
        dtype=torch.float64,
        device=inputs.reverse_time.device,
    )
    return _interpolate_q_values(values, inputs)


def _interpolate_q_values(values: Tensor, inputs: ModelInputs) -> Tensor:
    if values.shape != TANGENT_BASELINE_SHAPE or values.dtype != torch.float64:
        raise BoundaryTangentContractError("q_B tensor has the wrong shape or dtype")
    if type(inputs) is not ModelInputs:
        raise BoundaryTangentContractError("baseline lookup requires exact ModelInputs")
    times = inputs.reverse_time.to(dtype=torch.float64)
    phases = inputs.phase.to(dtype=torch.long)
    if not bool(torch.isfinite(times).all()) or bool(
        torch.any((phases < 0) | (phases >= PHASE_COUNT))
    ):
        raise BoundaryTangentContractError("baseline coordinates are malformed")
    scaled = (1.0 - times) * float(PHASE_COUNT * OUTER_STEPS) - phases.to(
        dtype=torch.float64
    )
    steps = torch.floor(scaled / float(PHASE_COUNT)).to(dtype=torch.long)
    fraction = scaled - float(PHASE_COUNT) * steps.to(dtype=torch.float64)
    if bool(torch.any((steps < 0) | (steps >= OUTER_STEPS))):
        raise BoundaryTangentContractError("baseline outer step lies outside K=512")
    if bool(
        torch.any(
            (fraction < M8_KNOTS[0] - FRACTION_TOLERANCE)
            | (fraction > M8_KNOTS[-1] + FRACTION_TOLERANCE)
        )
    ):
        raise BoundaryTangentContractError(
            "control midpoint lies outside the frozen M8 interpolation range"
        )
    scheduled = torch.as_tensor(
        ALLOWED_FRACTIONAL_COORDINATES,
        dtype=torch.float64,
        device=fraction.device,
    )
    distance = torch.abs(fraction[:, None] - scheduled[None, :])
    nearest_distance, nearest_index = torch.min(distance, dim=1)
    fraction = torch.where(
        nearest_distance <= FRACTION_TOLERANCE,
        scheduled[nearest_index],
        fraction,
    )
    lower_boundary = torch.as_tensor(
        M8_KNOTS[0], dtype=torch.float64, device=fraction.device
    )
    upper_boundary = torch.as_tensor(
        M8_KNOTS[-1], dtype=torch.float64, device=fraction.device
    )
    fraction = torch.where(fraction < lower_boundary, lower_boundary, fraction)
    fraction = torch.where(fraction > upper_boundary, upper_boundary, fraction)
    position = 8.0 * fraction - 0.5
    lower = torch.floor(position).to(dtype=torch.long)
    upper = torch.ceil(position).to(dtype=torch.long)
    weight = position - lower.to(dtype=torch.float64)
    values = values.to(device=fraction.device)
    rows = torch.div(steps, 128, rounding_mode="floor")
    left = values[rows, phases, lower]
    right = values[rows, phases, upper]
    interpolated = left + weight[:, None] * (right - left)
    return torch.where((weight == 0.0)[:, None], left, interpolated)


class BoundaryTangentPredictor(nn.Module):
    """Frozen tangent baseline plus an unchanged width-32 score network."""

    def __init__(
        self,
        baseline: TangentBaseline,
        residual_score: JacobiRBPhasePredictor | None = None,
        *,
        zero_residual: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(baseline, TangentBaseline):
            raise BoundaryTangentContractError("baseline has the wrong type")
        active = residual_score or JacobiRBPhasePredictor(width=32)
        if not isinstance(active, JacobiRBPhasePredictor) or active.width != 32:
            raise BoundaryTangentContractError(
                "residual score must be the unchanged width-32 JacobiRBPhasePredictor"
            )
        self.residual_score = active
        self.baseline_fingerprint = baseline.fingerprint
        self.interpolation_rule = baseline.interpolation_rule
        self.register_buffer(
            "_q_values",
            torch.from_numpy(np.array(baseline.q_values, copy=True)),
            persistent=True,
        )
        if zero_residual:
            zero_initialize_residual(self.residual_score)

    def baseline_score(self, inputs: ModelInputs) -> Tensor:
        return _interpolate_q_values(self._q_values, inputs)

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        """Return the finite coefficient ``q_B + q_theta`` directly."""

        if type(inputs) is not ModelInputs:
            raise BoundaryTangentContractError(
                "predictor accepts only exact permitted ModelInputs"
            )
        return self.baseline_score(inputs) + call_model(
            self.residual_score, inputs
        ).to(dtype=torch.float64)

    def baseline_prediction(self, inputs: ModelInputs) -> Tensor:
        geometry = edge_pair_geometry(inputs)
        return geometry.mobility * self.baseline_score(inputs)

    def forward(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise BoundaryTangentContractError(
                "predictor accepts only exact permitted ModelInputs"
            )
        geometry = edge_pair_geometry(inputs)
        score = self.score_prediction(inputs)
        if not bool(torch.isfinite(score).all()):
            raise BoundaryTangentContractError("predicted tangent score is nonfinite")
        prediction = geometry.mobility * score
        return torch.where(geometry.mobility == 0.0, torch.zeros_like(prediction), prediction)


def direct_raw_target_mse(
    prediction: Tensor, exact_target: Tensor, target_scale: float
) -> tuple[Tensor, Tensor]:
    """Direct normalized MSE against raw ``Zbar``; no quotient is formed."""

    if (
        not isinstance(prediction, Tensor)
        or not isinstance(exact_target, Tensor)
        or prediction.shape != exact_target.shape
        or prediction.ndim != 2
        or prediction.shape[1] != EDGES_PER_PHASE
    ):
        raise BoundaryTangentContractError("prediction/target shape is invalid")
    scale = float(target_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise BoundaryTangentContractError("target scale must be finite and positive")
    raw = torch.mean(
        (prediction.to(dtype=torch.float64) - exact_target.to(dtype=torch.float64)).square()
    )
    return raw / (scale * scale), raw


def synthetic_tangent_score(inputs: ModelInputs) -> Tensor:
    """Finite analytic score coefficient using only permitted inputs."""

    return synthetic_teacher_target(inputs)


def synthetic_tangent_target(inputs: ModelInputs) -> Tensor:
    """Analytic boundary-tangent target ``y(1-y) q_teacher``."""

    return edge_pair_geometry(inputs).mobility * synthetic_tangent_score(inputs)


def configure_exact_synthetic_tangent_teacher(model: BoundaryTangentPredictor) -> None:
    """Configure the residual score to the exact analytic teacher.

    The helper requires an all-zero baseline so the combined score equals the
    representable analytic teacher exactly.
    """

    if not isinstance(model, BoundaryTangentPredictor):
        raise BoundaryTangentContractError("synthetic teacher model has wrong type")
    if bool(torch.any(model._q_values != 0.0)):  # noqa: SLF001 - control helper
        raise BoundaryTangentContractError("synthetic teacher requires zero q_B")
    configure_exact_synthetic_teacher(model.residual_score)


def frozen_score_logistic_fraction(
    head_fraction: Tensor, frozen_score: Tensor, delta_u: Tensor | float
) -> Tensor:
    """Exact flow of ``dy/du=2*y*(1-y)*q`` for frozen finite ``q``."""

    if (
        not isinstance(head_fraction, Tensor)
        or not isinstance(frozen_score, Tensor)
        or head_fraction.shape != frozen_score.shape
        or not head_fraction.dtype.is_floating_point
        or not frozen_score.dtype.is_floating_point
        or head_fraction.device != frozen_score.device
    ):
        raise BoundaryTangentContractError("logistic-flow tensors are malformed")
    y = head_fraction.to(dtype=torch.float64)
    score = frozen_score.to(dtype=torch.float64)
    exposure = torch.as_tensor(delta_u, dtype=torch.float64, device=y.device)
    try:
        exposure = torch.broadcast_to(exposure, y.shape)
    except RuntimeError as exc:
        raise BoundaryTangentContractError("delta_u is not broadcastable") from exc
    if (
        not bool(torch.isfinite(y).all())
        or not bool(torch.isfinite(score).all())
        or not bool(torch.isfinite(exposure).all())
        or bool(torch.any((y < 0.0) | (y > 1.0)))
        or bool(torch.any(exposure < 0.0))
    ):
        raise BoundaryTangentContractError("logistic-flow values are invalid")
    result = y.clone()
    interior = (y > 0.0) & (y < 1.0) & (exposure != 0.0) & (score != 0.0)
    shift = 2.0 * score * exposure
    positive = interior & (shift >= 0.0)
    negative = interior & (shift < 0.0)
    exp_negative = torch.zeros_like(y)
    exp_negative[positive] = torch.exp(-shift[positive])
    result[positive] = y[positive] / (
        y[positive] + (1.0 - y[positive]) * exp_negative[positive]
    )
    exp_positive = torch.zeros_like(y)
    exp_positive[negative] = torch.exp(shift[negative])
    result[negative] = y[negative] * exp_positive[negative] / (
        (1.0 - y[negative]) + y[negative] * exp_positive[negative]
    )
    if not bool(torch.isfinite(result).all()) or bool(
        torch.any((result < 0.0) | (result > 1.0))
    ):
        raise BoundaryTangentContractError("logistic flow produced an invalid fraction")
    return result


def frozen_score_logistic_flow(
    state: Tensor,
    matching: int | tuple[Tensor, Tensor],
    frozen_score: Tensor,
    delta_u: Tensor | float,
) -> Tensor:
    """Apply the exact frozen-score logistic flow while preserving pair mass."""

    if not isinstance(state, Tensor) or state.dtype != torch.float64:
        raise BoundaryTangentContractError("state must be float64")
    squeezed = state.ndim == 1
    states = state.unsqueeze(0) if squeezed else state
    if states.ndim != 2 or states.shape[1] != STATE_SIZE:
        raise BoundaryTangentContractError("state must have shape [P,784]")
    if not bool(torch.isfinite(states).all()) or bool(torch.any(states < 0.0)):
        raise BoundaryTangentContractError("state is not finite and nonnegative")
    if isinstance(matching, tuple):
        tails, heads = matching
        tails = tails.to(device=states.device, dtype=torch.long).reshape(-1)
        heads = heads.to(device=states.device, dtype=torch.long).reshape(-1)
    else:
        index = int(matching)
        if not 0 <= index < 4:
            raise BoundaryTangentContractError("matching is outside [0,4)")
        all_tails, all_heads = matching_indices(device=states.device)
        tails, heads = all_tails[index], all_heads[index]
    if tails.shape != (EDGES_PER_PHASE,) or heads.shape != tails.shape:
        raise BoundaryTangentContractError("matching has the wrong shape")
    score = frozen_score.unsqueeze(0) if frozen_score.ndim == 1 else frozen_score
    if score.shape != (states.shape[0], EDGES_PER_PHASE):
        raise BoundaryTangentContractError("frozen score must have shape [P,392]")
    tail = states[:, tails]
    head = states[:, heads]
    pair = tail + head
    active = pair > 0.0
    fraction = torch.zeros_like(pair)
    fraction[active] = head[active] / pair[active]
    next_fraction = frozen_score_logistic_fraction(fraction, score, delta_u)
    output = states.clone()
    next_head = pair * next_fraction
    output[:, heads] = next_head
    output[:, tails] = pair - next_head
    if not bool(torch.isfinite(output).all()) or bool(torch.any(output < 0.0)):
        raise BoundaryTangentContractError("logistic flow produced an invalid state")
    return output[0] if squeezed else output


def controlled_reverse_phase_tangent(
    state: Tensor,
    k: Any,
    phase: Any,
    M: Any,
    transition_namespace: str,
    *,
    controller: BoundaryTangentPredictor,
    reference_transition: _reverse_controller.ReferenceTransition,
    path_ids: Sequence[int],
    label: int | Tensor,
) -> _reverse_controller.ControlledPhaseResult:
    """Exact-reference/tangent-control/exact-reference phase composition.

    This mirrors the frozen controller's split order and transition IDs.  The
    only changed operation is the learned subflow: the model returns its
    direct finite ``q`` coefficient, which is advanced by the exact logistic
    flow instead of an affine fraction step.
    """

    if not isinstance(controller, BoundaryTangentPredictor):
        raise BoundaryTangentContractError("controller has the wrong type")
    step = _reverse_controller._index(k, "k")  # noqa: SLF001
    occurrence = _reverse_controller._index(phase, "phase")  # noqa: SLF001
    microsteps = _reverse_controller._index(M, "M")  # noqa: SLF001
    if microsteps not in _reverse_controller.REFINEMENT_CONTROL_MICROSTEPS:
        raise BoundaryTangentContractError("M must be one of the frozen {2,4,8}")
    if transition_namespace != _reverse_controller.NAMESPACE_VERSION:
        raise BoundaryTangentContractError("transition namespace changed")
    if not callable(reference_transition):
        raise BoundaryTangentContractError("certified reference callback is missing")
    states, squeezed = _reverse_controller._batched_state(state)  # noqa: SLF001
    paths = tuple(_reverse_controller._index(item, "path_id") for item in path_ids)  # noqa: SLF001
    if len(paths) != states.shape[0] or len(set(paths)) != len(paths):
        raise BoundaryTangentContractError("path IDs must uniquely identify each state")
    color = PHASE_MATCHINGS[occurrence]
    duration = PHASE_DURATIONS[occurrence]
    tails, heads = _reverse_controller._matching_tensors(  # noqa: SLF001
        color, device=states.device
    )
    initial_total = torch.sum(states, dim=1)
    pair_mass = states[:, tails] + states[:, heads]
    full_exposure = _reverse_controller.phase_exposure(pair_mass, duration)
    delta_u = full_exposure / float(microsteps)
    midpoint_times: list[float] = []
    maximum_pair_error = 0.0
    maximum_simplex_error = 0.0

    for reverse_index, j in enumerate(range(microsteps, 0, -1)):
        for side in ("pre", "post"):
            role = f"reverse_reference_{side}_control_M{microsteps}"
            head_fraction = torch.zeros_like(pair_mass)
            active = pair_mass > 0.0
            head_fraction[active] = states[:, heads][active] / pair_mass[active]
            ids = _reverse_controller.controller_transition_ids(
                paths,
                outer_step=step,
                phase=occurrence,
                reverse_microstep=reverse_index,
                role=role,
                device=states.device,
            )
            result = reference_transition(
                head_fraction=head_fraction,
                exposure=delta_u / 2.0,
                transition_ids=ids,
                role=role,
            )
            fraction = _reverse_controller._reference_fraction(  # noqa: SLF001
                result, tuple(pair_mass.shape)
            ).to(device=states.device, dtype=torch.float64)
            states = _reverse_controller._scatter_fraction(  # noqa: SLF001
                states, tails, heads, pair_mass, fraction
            )
            current_pair = states[:, tails] + states[:, heads]
            maximum_pair_error = max(
                maximum_pair_error,
                float(torch.max(torch.abs(current_pair - pair_mass)).item()),
            )
            maximum_simplex_error = max(
                maximum_simplex_error,
                float(
                    torch.max(
                        torch.abs(torch.sum(states, dim=1) - initial_total)
                    ).item()
                ),
            )
            if side == "pre":
                q_mid = (j - 0.5) / float(microsteps)
                reverse_time = _reverse_controller.internal_reverse_time(
                    step, occurrence, q_mid
                )
                midpoint_times.append(reverse_time)
                labels = (
                    label.to(device=states.device, dtype=torch.long).reshape(-1)
                    if isinstance(label, Tensor)
                    else torch.full(
                        (states.shape[0],),
                        int(label),
                        dtype=torch.long,
                        device=states.device,
                    )
                )
                if labels.shape != (states.shape[0],):
                    raise BoundaryTangentContractError("label must be scalar or [P]")
                inputs = ModelInputs(
                    later_full_state=states.to(dtype=torch.float32),
                    reverse_time=torch.full(
                        (states.shape[0],),
                        reverse_time,
                        dtype=torch.float64,
                        device=states.device,
                    ),
                    phase=torch.full(
                        (states.shape[0],),
                        occurrence,
                        dtype=torch.long,
                        device=states.device,
                    ),
                    color=torch.full(
                        (states.shape[0],),
                        color,
                        dtype=torch.long,
                        device=states.device,
                    ),
                    duration=torch.full(
                        (states.shape[0],),
                        duration,
                        dtype=torch.float32,
                        device=states.device,
                    ),
                    label=labels,
                )
                score = controller.score_prediction(inputs)
                if score.shape != pair_mass.shape or not bool(
                    torch.isfinite(score).all()
                ):
                    raise BoundaryTangentContractError(
                        "controller score must be finite [P,392]"
                    )
                states = frozen_score_logistic_flow(
                    states, (tails, heads), score, delta_u
                )
                current_pair = states[:, tails] + states[:, heads]
                maximum_pair_error = max(
                    maximum_pair_error,
                    float(torch.max(torch.abs(current_pair - pair_mass)).item()),
                )
                maximum_simplex_error = max(
                    maximum_simplex_error,
                    float(
                        torch.max(
                            torch.abs(torch.sum(states, dim=1) - initial_total)
                        ).item()
                    ),
                )

    if maximum_pair_error > 2.0e-12 or maximum_simplex_error > 2.0e-12:
        raise BoundaryTangentContractError("tangent phase violated simplex mass")
    return _reverse_controller.ControlledPhaseResult(
        state=states[0] if squeezed else states,
        midpoint_reverse_times=tuple(midpoint_times),
        transition_count=2 * microsteps * len(paths) * EDGES_PER_PHASE,
        maximum_pair_mass_error=maximum_pair_error,
        maximum_simplex_mass_error=maximum_simplex_error,
    )


__all__ = [
    "BOUNDARY_TANGENT_VERSION",
    "BoundaryTangentContractError",
    "BoundaryTangentPredictor",
    "EdgePairGeometry",
    "M8_KNOTS",
    "TANGENT_BASELINE_FILE_SCHEMA",
    "TANGENT_BASELINE_SCHEMA",
    "TANGENT_BASELINE_SHAPE",
    "TANGENT_INTERPOLATION_RULE",
    "TANGENT_QUARTILE_RULE",
    "TangentBaseline",
    "configure_exact_synthetic_tangent_teacher",
    "controlled_reverse_phase_tangent",
    "derive_tangent_baseline",
    "direct_raw_target_mse",
    "edge_pair_geometry",
    "frozen_score_logistic_flow",
    "frozen_score_logistic_fraction",
    "interpolate_tangent_baseline",
    "load_tangent_baseline",
    "save_tangent_baseline",
    "synthetic_tangent_score",
    "synthetic_tangent_target",
]
