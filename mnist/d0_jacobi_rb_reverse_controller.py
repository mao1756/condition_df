"""Pure mathematics for the exact Jacobi/RB reverse-controller controls.

This module is intentionally additive.  It does not implement a forward cache,
an optimizer, a reconstruction sampler, or an approximate transition.  The
only stochastic operation accepted here is an injected *certified exact
Jacobi* reference-transition callback.  Between two such half transitions the
module applies the exact affine flow of the frozen learned vector field.

The learned quantity is the unchanged Rao--Blackwell conditional mean

``m = E[L - M Y | later full state, edge, time]``.

In Jacobi exposure time its contribution to the reverse fraction drift is
``2*m``.  Positive values therefore transfer mass from the frozen matching's
tail to its head.  An inadmissible affine step is rejected; it is never
clipped, projected, floored, limited, or renormalized.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_coarse_residual import (
    CoarseResidualPredictor,
    WITNESS_VALUES_SHA256,
    load_frozen_coarse_baseline,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    GRID_SIZE,
    MODEL_INPUT_FIELDS,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
    call_model,
    matching_indices,
    model_inputs_from_mapping,
    selected_reverse_time,
    state_dict_sha256,
)


REVERSE_CONTROLLER_VERSION = "d0-jacobi-rb-reverse-controller-control-v1"
NAMESPACE_VERSION = REVERSE_CONTROLLER_VERSION

ALPHA = 1.0
GRID_SPACING = 1.0 / GRID_SIZE
TAU_EFF = 5.0e-5
MACROSTEP_SCHEDULE_INTEGRAL = TAU_EFF / OUTER_STEPS

PRODUCTION_CONTROLLER_MICROSTEPS = 8
REFINEMENT_CONTROL_MICROSTEPS = (2, 4, 8)
MIDPOINT_FRACTIONS = {
    2: (1.0 / 4.0, 3.0 / 4.0),
    4: (1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0),
    8: tuple((2 * index + 1) / 16.0 for index in range(8)),
}
ALLOWED_FRACTIONAL_COORDINATES = tuple(
    sorted({1.0, *(item for values in MIDPOINT_FRACTIONS.values() for item in values)})
)
FRACTION_TOLERANCE = 2.0e-10

CONTROLLER_ROOT_SEED = 261_301
LOCAL_BOOTSTRAP_SEED = 261_302
TRAJECTORY_BOOTSTRAP_SEED = 261_303
ORACLE_ROOT_SEED = 261_304

PREFLIGHT_PATH_IDS = tuple(range(0xEA000, 0xEA008))
PHYSICAL_CONTROL_PATH_IDS = tuple(range(0xEB000, 0xEB040))
ORACLE_PATH_IDS = tuple(range(0xEE000, 0xEE020))
RESERVED_FUTURE_PATH_IDS = tuple(range(0xEC000, 0xEE000))
RESERVED_PRODUCTION_PATH_IDS = tuple(range(0xF0000, 0x100000))
PATH_ID_LIMIT = 1 << 20

TRANSITION_ROLES = (
    "partial_phase_target_prefix",
    "reverse_reference_pre_control_M2",
    "reverse_reference_post_control_M2",
    "reverse_reference_pre_control_M4",
    "reverse_reference_post_control_M4",
    "reverse_reference_pre_control_M8",
    "reverse_reference_post_control_M8",
    "analytic_teacher_exact_reverse",
)
_TRANSITION_ROLE_CODE = {name: index for index, name in enumerate(TRANSITION_ROLES)}
# A frozen 14-bit workflow tag occupies bits 49--62.  The remaining packed
# fields are injective, fit below bit 48, and leave the sign bit clear.  This
# makes tens of millions of production IDs cheap to construct while retaining
# an explicit, fingerprintable version namespace.
_NAMESPACE_TAG = int.from_bytes(
    hashlib.sha256(NAMESPACE_VERSION.encode("ascii")).digest()[:2], "little"
) & 0x3FFF
_EDGE_SHIFT = 0
_MICROSTEP_SHIFT = 9
_PHASE_SHIFT = 13
_OUTER_STEP_SHIFT = 16
_PATH_SHIFT = 25
_ROLE_SHIFT = 45
_NAMESPACE_SHIFT = 49

EXPECTED_PARENT_CHECKPOINT_SHA256 = (
    "24a0893daa31196815463a7396220542003e7dc2557689950ba4dd0eeaa9c914"
)
EXPECTED_PARENT_STATE_SHA256 = (
    "df479e979cf6dd99580bd918377405b665791a4608f45f6cae326cc10e5e6ad9"
)
EXPECTED_PARENT_SEED = 261_254
EXPECTED_PARENT_UPDATE = 3_000
EXPECTED_PARENT_DECISION = "exact_rb_coarse_residual_learnable"

LOCAL_RISK_FAMILY_SIZE = 228
LOCAL_RISK_CONFIDENCE = 0.995
LOCAL_RISK_BOOTSTRAP_REPLICATES = 50_000
TRAJECTORY_FAMILY_SIZE = 784
TRAJECTORY_COMPONENT_COUNT = TRAJECTORY_FAMILY_SIZE // 2
TRAJECTORY_CONFIDENCE = 0.995
TRAJECTORY_BOOTSTRAP_REPLICATES = 50_000
REVERSE_LAW_BIAS_MARGIN = 0.10
MICROSTEP_REFINEMENT_MARGIN = 0.05

CLAIM_FLAG_NAMES = (
    "reverse_sampling_authorized",
    "reconstruction_authorized",
    "known_prior_claim_authorized",
    "full_dataset_training_authorized",
    "unsplit_generator_claim_authorized",
    "spatial_dirichlet_ferguson_claim_authorized",
)


class ReverseControllerContractError(ValueError):
    """A reverse-controller mathematical or input contract was violated."""


class ControllerBoundaryStepRejected(ReverseControllerContractError):
    """The unmodified affine learned flow would leave an edge simplex."""

    failure_code = "controller_boundary_step_rejected"


def _index(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReverseControllerContractError(f"cannot read {path}") from exc
    if not isinstance(value, dict):
        raise ReverseControllerContractError(f"{path} is not a JSON object")
    return value


def internal_reverse_time(k: Any, phase: Any, q: Any) -> float:
    """Return the exact internal split coordinate ``1-(7k+p+q)/(7K)``."""

    step = _index(k, "k")
    occurrence = _index(phase, "phase")
    try:
        fraction = float(q)
    except (TypeError, ValueError) as exc:
        raise TypeError("q must be a real scalar") from exc
    if not 0 <= step < OUTER_STEPS:
        raise ReverseControllerContractError("k lies outside the K=512 chain")
    if not 0 <= occurrence < PHASE_COUNT:
        raise ReverseControllerContractError("phase occurrence lies outside [0,7)")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ReverseControllerContractError("q must be finite and lie in [0,1]")
    return 1.0 - (
        PHASE_COUNT * step + occurrence + fraction
    ) / float(PHASE_COUNT * OUTER_STEPS)


def reverse_execution_order() -> tuple[tuple[int, int], ...]:
    """Frozen reverse order, retaining repeated *occurrences*, not just colors."""

    return tuple(
        (step, phase)
        for step in range(OUTER_STEPS - 1, -1, -1)
        for phase in range(PHASE_COUNT - 1, -1, -1)
    )


@dataclass(frozen=True)
class FractionalCoordinate:
    outer_step: Tensor
    within_phase_fraction: Tensor
    forward_outer_quartile: Tensor
    reverse_quartile: Tensor
    reverse_start: Tensor


def fractional_coordinate(reverse_time: Tensor, phase: Tensor) -> FractionalCoordinate:
    """Recover ``k``, ``q`` and unambiguous quartiles from permitted inputs.

    Only the exact endpoint and the predeclared M=2/4/8 midpoints are accepted.
    The computation deliberately has no ``outer_step`` argument: the scheduler's
    audit coordinate cannot enter the model or coarse-table lookup.
    """

    if (
        not isinstance(reverse_time, Tensor)
        or not isinstance(phase, Tensor)
        or reverse_time.ndim != 1
        or phase.shape != reverse_time.shape
        or not reverse_time.dtype.is_floating_point
        or phase.dtype
        not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        or reverse_time.device != phase.device
    ):
        raise ReverseControllerContractError("fractional coordinates are malformed")
    if not bool(torch.isfinite(reverse_time).all()):
        raise ReverseControllerContractError("reverse time contains nonfinite values")
    phases = phase.to(dtype=torch.int64)
    if bool(torch.any((phases < 0) | (phases >= PHASE_COUNT))):
        raise ReverseControllerContractError("phase occurrence lies outside [0,7)")

    times = reverse_time.to(dtype=torch.float64)
    scaled = (1.0 - times) * float(PHASE_COUNT * OUTER_STEPS) - phases.to(
        dtype=torch.float64
    )
    steps = torch.floor(scaled / float(PHASE_COUNT)).to(dtype=torch.int64)
    fraction = scaled - float(PHASE_COUNT) * steps.to(dtype=torch.float64)
    if bool(torch.any((steps < 0) | (steps >= OUTER_STEPS))):
        raise ReverseControllerContractError("recovered outer step lies outside K=512")
    if bool(torch.any((fraction <= 0.0) | (fraction > 1.0 + FRACTION_TOLERANCE))):
        raise ReverseControllerContractError("internal phase fraction must lie in (0,1]")

    allowed = torch.as_tensor(
        ALLOWED_FRACTIONAL_COORDINATES,
        dtype=torch.float64,
        device=times.device,
    )
    nearest_error = torch.min(torch.abs(fraction[:, None] - allowed[None, :]), dim=1).values
    if bool(torch.any(nearest_error > FRACTION_TOLERANCE)):
        raise ReverseControllerContractError(
            "internal phase fraction is outside the frozen midpoint set"
        )
    # Snap only the recovered indexing coordinate, never model time or state.
    nearest = torch.argmin(torch.abs(fraction[:, None] - allowed[None, :]), dim=1)
    fraction = allowed[nearest]
    quartile = torch.div(steps, 128, rounding_mode="floor")
    return FractionalCoordinate(
        outer_step=steps,
        within_phase_fraction=fraction,
        forward_outer_quartile=quartile,
        reverse_quartile=3 - quartile,
        reverse_start=quartile == 3,
    )


class FractionalFrozenController(nn.Module):
    """Read-only adapter for the sealed baseline-plus-residual checkpoint."""

    def __init__(self, predictor: CoarseResidualPredictor) -> None:
        super().__init__()
        if not isinstance(predictor, CoarseResidualPredictor):
            raise ReverseControllerContractError(
                "fractional adapter requires the unchanged combined predictor"
            )
        self.predictor = predictor
        self.predictor.eval()
        self.predictor.requires_grad_(False)

    def baseline_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise ReverseControllerContractError(
                "controller accepts only the exact six-field ModelInputs"
            )
        coordinate = fractional_coordinate(inputs.reverse_time, inputs.phase)
        values = self.predictor._coarse_values.to(  # noqa: SLF001 - frozen adapter
            device=inputs.phase.device, dtype=torch.float64
        )
        return values[
            coordinate.forward_outer_quartile,
            inputs.phase.to(dtype=torch.int64),
        ]

    def residual_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise ReverseControllerContractError(
                "controller accepts only the exact six-field ModelInputs"
            )
        return call_model(self.predictor.residual, inputs)

    def forward(self, inputs: ModelInputs) -> Tensor:
        residual = self.residual_prediction(inputs)
        return self.baseline_prediction(inputs) + residual.to(dtype=torch.float64)


def load_frozen_controller(
    parent_run: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> FractionalFrozenController:
    """Load and verify the exact sealed checkpoint and frozen coarse table."""

    root = Path(parent_run).resolve()
    status = _load_json(root / "run_status.json")
    if (
        status.get("state") != "complete"
        or status.get("stage") != "confirm"
        or status.get("decision") != EXPECTED_PARENT_DECISION
        or int(status.get("reverse_sampling_performed", 1)) != 0
        or int(status.get("reconstruction_performed", 1)) != 0
    ):
        raise ReverseControllerContractError("parent terminal status is incompatible")

    selected = _load_json(root / "selected_model.json")
    candidate = selected.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ReverseControllerContractError("selected model candidate is missing")
    if (
        int(candidate.get("seed", -1)) != EXPECTED_PARENT_SEED
        or int(candidate.get("update", -1)) != EXPECTED_PARENT_UPDATE
        or candidate.get("state_sha256") != EXPECTED_PARENT_STATE_SHA256
        or selected.get("selected_model_sha256") != EXPECTED_PARENT_CHECKPOINT_SHA256
    ):
        raise ReverseControllerContractError("selected model identity changed")

    checkpoint_path = root / "selected_model.pt"
    if (
        not checkpoint_path.is_file()
        or _file_sha256(checkpoint_path) != EXPECTED_PARENT_CHECKPOINT_SHA256
    ):
        raise ReverseControllerContractError("selected checkpoint file changed")
    baseline = load_frozen_coarse_baseline(root / "frozen_coarse_baseline.npz")
    if baseline.values_sha256 != WITNESS_VALUES_SHA256:
        raise ReverseControllerContractError("frozen coarse table changed")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device(device), weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReverseControllerContractError("cannot load selected checkpoint") from exc
    if not isinstance(checkpoint, Mapping):
        raise ReverseControllerContractError("selected checkpoint is malformed")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or state_dict_sha256(state) != EXPECTED_PARENT_STATE_SHA256:
        raise ReverseControllerContractError("selected checkpoint state changed")
    predictor = CoarseResidualPredictor(baseline, zero_residual=True).to(device)
    predictor.load_state_dict(state, strict=True)
    if state_dict_sha256(predictor.state_dict()) != EXPECTED_PARENT_STATE_SHA256:
        raise ReverseControllerContractError("selected state replay hash changed")
    return FractionalFrozenController(predictor)


def frozen_fractional_prediction(
    controller: FractionalFrozenController,
    model_inputs: ModelInputs | Mapping[str, Tensor],
) -> Tensor:
    """Evaluate the frozen controller through the immutable six-field firewall."""

    if not isinstance(controller, FractionalFrozenController):
        raise ReverseControllerContractError("controller has the wrong type")
    inputs = (
        model_inputs_from_mapping(model_inputs)
        if isinstance(model_inputs, Mapping)
        else model_inputs
    )
    if type(inputs) is not ModelInputs:
        raise ReverseControllerContractError("model inputs must use the exact firewall")
    return controller(inputs)


def _validate_duration_numpy(duration: np.ndarray) -> None:
    if not np.isfinite(duration).all() or not np.isin(duration, (0.5, 1.0)).all():
        raise ReverseControllerContractError("duration must be a frozen half/full phase")


def phase_exposure(
    pair_mass: Any,
    duration: Any,
    *,
    h: float = GRID_SPACING,
    alpha: float = ALPHA,
    schedule_integral: float = MACROSTEP_SCHEDULE_INTEGRAL,
) -> Any:
    """Exact state-dependent Jacobi exposure, with exact zero-mass masking."""

    if (
        not math.isfinite(float(h))
        or float(h) <= 0.0
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0.0
        or not math.isfinite(float(schedule_integral))
        or float(schedule_integral) < 0.0
    ):
        raise ReverseControllerContractError("exposure constants are invalid")
    coefficient = (2.0 * float(alpha) + 1.0) * float(schedule_integral) / (
        float(alpha) * float(h) ** 2
    )
    if isinstance(pair_mass, Tensor) or isinstance(duration, Tensor):
        mass = pair_mass if isinstance(pair_mass, Tensor) else torch.as_tensor(pair_mass)
        dur = duration if isinstance(duration, Tensor) else torch.as_tensor(
            duration, device=mass.device
        )
        if mass.device != dur.device:
            raise ReverseControllerContractError("mass and duration must share a device")
        mass, dur = torch.broadcast_tensors(
            mass.to(dtype=torch.float64), dur.to(dtype=torch.float64)
        )
        if not bool(torch.isfinite(mass).all()) or bool(torch.any(mass < 0.0)):
            raise ReverseControllerContractError("pair mass must be finite and nonnegative")
        if not bool(torch.isfinite(dur).all()) or not bool(
            torch.isin(dur, torch.tensor((0.5, 1.0), dtype=dur.dtype, device=dur.device)).all()
        ):
            raise ReverseControllerContractError(
                "duration must be a frozen half/full phase"
            )
        result = torch.zeros_like(mass)
        active = mass > 0.0
        result[active] = coefficient * dur[active] / mass[active]
        return result

    mass_np, duration_np = np.broadcast_arrays(
        np.asarray(pair_mass, dtype=np.float64), np.asarray(duration, dtype=np.float64)
    )
    if not np.isfinite(mass_np).all() or (mass_np < 0.0).any():
        raise ReverseControllerContractError("pair mass must be finite and nonnegative")
    _validate_duration_numpy(duration_np)
    result = np.zeros_like(mass_np)
    active = mass_np > 0.0
    result[active] = coefficient * duration_np[active] / mass_np[active]
    return float(result) if result.ndim == 0 else np.ascontiguousarray(result)


def learned_mass_flux(
    prediction: Any,
    *,
    duration: float = 1.0,
    h: float = GRID_SPACING,
    alpha: float = ALPHA,
    schedule_integral: float = MACROSTEP_SCHEDULE_INTEGRAL,
) -> Any:
    """Integrated tail-to-head learned mass transfer for one phase interval.

    Pair mass is intentionally absent: it cancels between fraction drift and
    the conversion back to mass.  At alpha=1 this is
    ``6 * schedule_integral * duration * prediction / h**2``.
    """

    duration_value = float(duration)
    if duration_value not in PHASE_DURATIONS:
        raise ReverseControllerContractError("duration is not a frozen occurrence duration")
    if (
        not math.isfinite(float(h))
        or float(h) <= 0.0
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0.0
        or not math.isfinite(float(schedule_integral))
        or float(schedule_integral) < 0.0
    ):
        raise ReverseControllerContractError("learned flux constants are invalid")
    coefficient = (
        2.0
        * (2.0 * float(alpha) + 1.0)
        * float(schedule_integral)
        * duration_value
        / (float(alpha) * float(h) ** 2)
    )
    if not math.isfinite(coefficient):
        raise ReverseControllerContractError("learned flux coefficient is nonfinite")
    if isinstance(prediction, Tensor):
        if not bool(torch.isfinite(prediction).all()):
            raise ReverseControllerContractError("prediction is nonfinite")
        return prediction * coefficient
    values = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ReverseControllerContractError("prediction is nonfinite")
    result = values * coefficient
    return float(result) if result.ndim == 0 else np.ascontiguousarray(result)


def _matching_tensors(
    matching: int | tuple[Tensor, Tensor], *, device: torch.device
) -> tuple[Tensor, Tensor]:
    if isinstance(matching, tuple):
        if len(matching) != 2 or not all(isinstance(item, Tensor) for item in matching):
            raise ReverseControllerContractError("matching tuple is malformed")
        tails, heads = matching
        if tails.device != device or heads.device != device:
            raise ReverseControllerContractError("matching indices are on the wrong device")
        if tails.numel() != EDGES_PER_PHASE or heads.shape != tails.shape:
            raise ReverseControllerContractError("matching has the wrong edge count")
        return tails.to(torch.long).reshape(-1), heads.to(torch.long).reshape(-1)
    index = _index(matching, "matching")
    if not 0 <= index < 4:
        raise ReverseControllerContractError("matching color lies outside [0,4)")
    tails, heads = matching_indices(device=device)
    return tails[index], heads[index]


def _batched_state(state: Tensor) -> tuple[Tensor, bool]:
    if not isinstance(state, Tensor) or state.dtype != torch.float64:
        raise ReverseControllerContractError("controller state must be a float64 tensor")
    squeezed = state.ndim == 1
    active = state.unsqueeze(0) if squeezed else state
    if active.ndim != 2 or active.shape[1] != STATE_SIZE:
        raise ReverseControllerContractError("controller state must have shape [P,784]")
    if not bool(torch.isfinite(active).all()) or bool(torch.any(active < 0.0)):
        raise ReverseControllerContractError("controller state is not a finite simplex state")
    return active, squeezed


def frozen_control_half_flow(
    state: Tensor,
    matching: int | tuple[Tensor, Tensor],
    prediction: Tensor,
    delta_u: Tensor | float,
) -> Tensor:
    """Apply ``y <- y + 2*m*delta_u`` without clipping or projection."""

    states, squeezed = _batched_state(state)
    tails, heads = _matching_tensors(matching, device=states.device)
    batch = states.shape[0]
    expected = (batch, EDGES_PER_PHASE)
    if not isinstance(prediction, Tensor):
        raise ReverseControllerContractError("prediction must be a torch.Tensor")
    values = prediction.unsqueeze(0) if prediction.ndim == 1 else prediction
    if values.shape != expected or values.device != states.device:
        raise ReverseControllerContractError("prediction must have shape [P,392]")
    values = values.to(dtype=torch.float64)
    exposure = torch.as_tensor(delta_u, dtype=torch.float64, device=states.device)
    try:
        exposure = torch.broadcast_to(exposure, expected)
    except RuntimeError as exc:
        raise ReverseControllerContractError("delta_u is not edge-broadcastable") from exc
    if (
        not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(exposure).all())
        or bool(torch.any(exposure < 0.0))
    ):
        raise ReverseControllerContractError("control prediction/exposure is invalid")

    tail_mass = states[:, tails]
    head_mass = states[:, heads]
    pair_mass = tail_mass + head_mass
    active = pair_mass > 0.0
    fraction = torch.zeros_like(pair_mass)
    fraction[active] = head_mass[active] / pair_mass[active]
    increment = torch.zeros_like(pair_mass)
    increment[active] = 2.0 * values[active] * exposure[active]
    if bool(torch.all(increment == 0.0)):
        return state.clone()
    next_fraction = fraction + increment
    invalid = active & (
        ~torch.isfinite(next_fraction)
        | (next_fraction < 0.0)
        | (next_fraction > 1.0)
    )
    if bool(torch.any(invalid)):
        indices = torch.nonzero(invalid, as_tuple=False)[0].tolist()
        raise ControllerBoundaryStepRejected(
            f"controller affine flow leaves [0,1] at path/edge {indices}"
        )

    output = states.clone()
    changed = active & (increment != 0.0)
    next_head = pair_mass * next_fraction
    next_tail = pair_mass - next_head
    tail_values = output[:, tails]
    head_values = output[:, heads]
    tail_values[changed] = next_tail[changed]
    head_values[changed] = next_head[changed]
    output[:, tails] = tail_values
    output[:, heads] = head_values
    if not bool(torch.isfinite(output).all()) or bool(torch.any(output < 0.0)):
        raise ReverseControllerContractError("controller flow produced an invalid state")
    return output[0] if squeezed else output


def _scatter_fraction(
    state: Tensor,
    tails: Tensor,
    heads: Tensor,
    pair_mass: Tensor,
    fraction: Tensor,
) -> Tensor:
    if fraction.shape != pair_mass.shape:
        raise ReverseControllerContractError("reference fraction has the wrong shape")
    if not bool(torch.isfinite(fraction).all()) or bool(
        torch.any((fraction < 0.0) | (fraction > 1.0))
    ):
        raise ReverseControllerContractError("reference fraction lies outside [0,1]")
    output = state.clone()
    active = pair_mass > 0.0
    next_head = pair_mass * fraction
    next_tail = pair_mass - next_head
    tail_values = output[:, tails]
    head_values = output[:, heads]
    tail_values[active] = next_tail[active]
    head_values[active] = next_head[active]
    output[:, tails] = tail_values
    output[:, heads] = head_values
    return output


def controller_transition_id(
    *,
    path_id: Any,
    outer_step: Any,
    phase: Any,
    reverse_microstep: Any,
    edge: Any,
    role: str,
) -> int:
    """Stable uint64 ID with explicit workflow, occurrence, microstep, and role."""

    path = _index(path_id, "path_id")
    step = _index(outer_step, "outer_step")
    occurrence = _index(phase, "phase")
    microstep = _index(reverse_microstep, "reverse_microstep")
    edge_index = _index(edge, "edge")
    if not 0 <= path < PATH_ID_LIMIT:
        raise ReverseControllerContractError("path_id lies outside the 20-bit plan")
    if not 0 <= step < OUTER_STEPS or not 0 <= occurrence < PHASE_COUNT:
        raise ReverseControllerContractError("transition split coordinate is invalid")
    if microstep < 0 or not 0 <= edge_index < EDGES_PER_PHASE:
        raise ReverseControllerContractError("transition microstep/edge is invalid")
    if role not in TRANSITION_ROLES:
        raise ReverseControllerContractError("transition role is not frozen")
    if microstep >= 16:
        raise ReverseControllerContractError("reverse microstep exceeds packed ID field")
    return int(
        (_NAMESPACE_TAG << _NAMESPACE_SHIFT)
        | (_TRANSITION_ROLE_CODE[role] << _ROLE_SHIFT)
        | (path << _PATH_SHIFT)
        | (step << _OUTER_STEP_SHIFT)
        | (occurrence << _PHASE_SHIFT)
        | (microstep << _MICROSTEP_SHIFT)
        | (edge_index << _EDGE_SHIFT)
    )


def controller_transition_ids(
    path_ids: Sequence[int],
    *,
    outer_step: int,
    phase: int,
    reverse_microstep: int,
    role: str,
    device: str | torch.device,
) -> Tensor:
    # Validate once per path, then fill the edge field with one NumPy vector
    # operation.  No state or batching coordinate affects the packed identity.
    bases = np.asarray(
        [
            controller_transition_id(
                path_id=path,
                outer_step=outer_step,
                phase=phase,
                reverse_microstep=reverse_microstep,
                edge=0,
                role=role,
            )
            for path in path_ids
        ],
        dtype=np.uint64,
    )
    edges = np.arange(EDGES_PER_PHASE, dtype=np.uint64)
    rows = np.ascontiguousarray(bases[:, None] | edges[None, :])
    return torch.from_numpy(rows).to(device=device).contiguous()


def validate_controller_path_plan() -> dict[str, Any]:
    roles = {
        "preflight": PREFLIGHT_PATH_IDS,
        "physical_control": PHYSICAL_CONTROL_PATH_IDS,
        "oracle": ORACLE_PATH_IDS,
        "future_confirmation_reserved": RESERVED_FUTURE_PATH_IDS,
        "production_reserved": RESERVED_PRODUCTION_PATH_IDS,
    }
    seen: set[int] = set()
    for role, values in roles.items():
        if any(not 0 <= value < PATH_ID_LIMIT for value in values):
            raise ReverseControllerContractError(f"{role} path ID is out of bounds")
        if len(set(values)) != len(values) or seen.intersection(values):
            raise ReverseControllerContractError(f"{role} path IDs collide")
        seen.update(values)
    return {
        "schema": REVERSE_CONTROLLER_VERSION + "-path-plan",
        "namespace_version": NAMESPACE_VERSION,
        "roles": {name: list(values) for name, values in roles.items()},
        "collision_free": 1,
    }


def _reference_fraction(result: Any, shape: tuple[int, int]) -> Tensor:
    if isinstance(result, Tensor):
        value = result
    elif isinstance(result, Mapping):
        value = result.get("later_head_fraction")
    else:
        value = getattr(result, "later_head_fraction", None)
    if not isinstance(value, Tensor) or value.numel() != math.prod(shape):
        raise ReverseControllerContractError(
            "certified reference callback returned the wrong fraction payload"
        )
    return value.reshape(shape)


@dataclass(frozen=True)
class ControlledPhaseResult:
    state: Tensor
    midpoint_reverse_times: tuple[float, ...]
    transition_count: int
    maximum_pair_mass_error: float
    maximum_simplex_mass_error: float
    boundary_rejection_count: int = 0
    correction_count: int = 0
    floor_count: int = 0
    limiter_count: int = 0
    projection_count: int = 0
    renormalization_count: int = 0


ReferenceTransition = Callable[..., Any]


def controlled_reverse_phase(
    state: Tensor,
    k: Any,
    phase: Any,
    M: Any,
    transition_namespace: str,
    *,
    controller: FractionalFrozenController | nn.Module,
    reference_transition: ReferenceTransition,
    path_ids: Sequence[int],
    label: int | Tensor,
) -> ControlledPhaseResult:
    """Compose exact-reference/learned/exact-reference reverse microsteps.

    ``reference_transition`` is the only transition hook.  It is called with
    keyword arguments ``head_fraction``, ``exposure``, ``transition_ids``, and
    ``role`` and must return a Tensor, mapping, or object containing certified
    ``later_head_fraction``.  This function contains no Euler/Gaussian fallback.
    """

    step = _index(k, "k")
    occurrence = _index(phase, "phase")
    microsteps = _index(M, "M")
    if microsteps not in REFINEMENT_CONTROL_MICROSTEPS:
        raise ReverseControllerContractError("M must be one of the frozen {2,4,8}")
    if transition_namespace != NAMESPACE_VERSION:
        raise ReverseControllerContractError("transition namespace changed")
    if not callable(reference_transition):
        raise ReverseControllerContractError("certified reference callback is missing")
    states, squeezed = _batched_state(state)
    paths = tuple(_index(item, "path_id") for item in path_ids)
    if len(paths) != states.shape[0] or len(set(paths)) != len(paths):
        raise ReverseControllerContractError("path IDs must uniquely identify each state")
    color = PHASE_MATCHINGS[occurrence]
    duration = PHASE_DURATIONS[occurrence]
    tails, heads = _matching_tensors(color, device=states.device)
    initial_total = torch.sum(states, dim=1)
    pair_mass = states[:, tails] + states[:, heads]
    full_exposure = phase_exposure(pair_mass, duration)
    delta_u = full_exposure / float(microsteps)
    midpoint_times: list[float] = []
    maximum_pair_error = 0.0
    maximum_simplex_error = 0.0

    for reverse_index, j in enumerate(range(microsteps, 0, -1)):
        for side in ("pre", "post"):
            role = f"reverse_reference_{side}_control_M{microsteps}"
            head = torch.zeros_like(pair_mass)
            active = pair_mass > 0.0
            head[active] = states[:, heads][active] / pair_mass[active]
            ids = controller_transition_ids(
                paths,
                outer_step=step,
                phase=occurrence,
                reverse_microstep=reverse_index,
                role=role,
                device=states.device,
            )
            result = reference_transition(
                head_fraction=head,
                exposure=delta_u / 2.0,
                transition_ids=ids,
                role=role,
            )
            fraction = _reference_fraction(result, tuple(pair_mass.shape)).to(
                device=states.device, dtype=torch.float64
            )
            states = _scatter_fraction(states, tails, heads, pair_mass, fraction)
            current_pair = states[:, tails] + states[:, heads]
            maximum_pair_error = max(
                maximum_pair_error,
                float(torch.max(torch.abs(current_pair - pair_mass)).item()),
            )
            maximum_simplex_error = max(
                maximum_simplex_error,
                float(torch.max(torch.abs(torch.sum(states, dim=1) - initial_total)).item()),
            )
            if side == "pre":
                q_mid = (j - 0.5) / float(microsteps)
                reverse_time = internal_reverse_time(step, occurrence, q_mid)
                midpoint_times.append(reverse_time)
                labels = (
                    label.to(device=states.device, dtype=torch.long).reshape(-1)
                    if isinstance(label, Tensor)
                    else torch.full(
                        (states.shape[0],), int(label), dtype=torch.long, device=states.device
                    )
                )
                if labels.shape != (states.shape[0],):
                    raise ReverseControllerContractError("label must be scalar or [P]")
                inputs = ModelInputs(
                    later_full_state=states.to(dtype=torch.float32),
                    reverse_time=torch.full(
                        (states.shape[0],),
                        reverse_time,
                        dtype=torch.float64,
                        device=states.device,
                    ),
                    phase=torch.full(
                        (states.shape[0],), occurrence, dtype=torch.long, device=states.device
                    ),
                    color=torch.full(
                        (states.shape[0],), color, dtype=torch.long, device=states.device
                    ),
                    duration=torch.full(
                        (states.shape[0],), duration, dtype=torch.float32, device=states.device
                    ),
                    label=labels,
                )
                prediction = controller(inputs)
                if not isinstance(prediction, Tensor) or prediction.shape != pair_mass.shape:
                    raise ReverseControllerContractError(
                        "controller prediction must have shape [P,392]"
                    )
                states = frozen_control_half_flow(
                    states, (tails, heads), prediction, delta_u
                )
                current_pair = states[:, tails] + states[:, heads]
                maximum_pair_error = max(
                    maximum_pair_error,
                    float(torch.max(torch.abs(current_pair - pair_mass)).item()),
                )
                maximum_simplex_error = max(
                    maximum_simplex_error,
                    float(
                        torch.max(torch.abs(torch.sum(states, dim=1) - initial_total)).item()
                    ),
                )

    if maximum_pair_error > 2.0e-12 or maximum_simplex_error > 2.0e-12:
        raise ReverseControllerContractError("controller phase violated simplex mass")
    final_state = states[0] if squeezed else states
    return ControlledPhaseResult(
        state=final_state,
        midpoint_reverse_times=tuple(midpoint_times),
        transition_count=2 * microsteps * len(paths) * EDGES_PER_PHASE,
        maximum_pair_mass_error=maximum_pair_error,
        maximum_simplex_mass_error=maximum_simplex_error,
    )


FOURIER_WAVEVECTORS = ((1, 0), (0, 1), (1, 1), (1, -1))
OBSERVABLE_NAMES = tuple(
    name
    for a, b in FOURIER_WAVEVECTORS
    for name in (f"fourier_{a}_{b}_real", f"fourier_{a}_{b}_imag")
) + ("quadratic", "cubic", "matching_legendre_1", "matching_legendre_2", "matching_legendre_3", "matching_legendre_4")


def _legendre_values(z: np.ndarray) -> tuple[np.ndarray, ...]:
    z2 = z * z
    return (
        z,
        0.5 * (3.0 * z2 - 1.0),
        0.5 * (5.0 * z2 * z - 3.0 * z),
        0.125 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0),
    )


def _observable_values(states: np.ndarray, occurrence: int) -> np.ndarray:
    batch = states.shape[0]
    grid = states.reshape(batch, GRID_SIZE, GRID_SIZE)
    rows, columns = np.meshgrid(
        np.arange(GRID_SIZE, dtype=np.float64),
        np.arange(GRID_SIZE, dtype=np.float64),
        indexing="ij",
    )
    values: list[np.ndarray] = []
    for a, b in FOURIER_WAVEVECTORS:
        weight = np.exp(2j * np.pi * (a * columns + b * rows) / GRID_SIZE)
        coefficient = np.sum(grid * weight[None, :, :], axis=(1, 2))
        values.extend((coefficient.real, coefficient.imag))
    values.extend((np.sum(states**2, axis=1), np.sum(states**3, axis=1)))
    tails_all, heads_all = matching_indices(device="cpu")
    color = PHASE_MATCHINGS[occurrence]
    tails = tails_all[color].numpy()
    heads = heads_all[color].numpy()
    pair_mass = states[:, tails] + states[:, heads]
    fraction = np.zeros_like(pair_mass)
    active = pair_mass > 0.0
    fraction[active] = states[:, heads][active] / pair_mass[active]
    z = 2.0 * fraction - 1.0
    for polynomial in _legendre_values(z):
        values.append(np.sum(pair_mass * polynomial, axis=1))
    return np.ascontiguousarray(np.stack(values, axis=1), dtype=np.float64)


@dataclass(frozen=True)
class PairedObservableResult:
    names: tuple[str, ...]
    before: np.ndarray
    after: np.ndarray
    difference: np.ndarray
    structural_invariant: np.ndarray
    maximum_structural_pair_mass_error: float


def paired_observables(
    before: np.ndarray | Tensor,
    after: np.ndarray | Tensor,
    *,
    phase: Any,
    structural_phase_invariants: bool = True,
) -> PairedObservableResult:
    """Evaluate the frozen fourteen non-image observables on paired states."""

    occurrence = _index(phase, "phase")
    if not 0 <= occurrence < PHASE_COUNT:
        raise ReverseControllerContractError("phase occurrence lies outside [0,7)")
    before_np = (
        before.detach().cpu().numpy() if isinstance(before, Tensor) else np.asarray(before)
    )
    after_np = after.detach().cpu().numpy() if isinstance(after, Tensor) else np.asarray(after)
    if before_np.ndim == 1:
        before_np = before_np[None, :]
    if after_np.ndim == 1:
        after_np = after_np[None, :]
    before_np = np.asarray(before_np, dtype=np.float64)
    after_np = np.asarray(after_np, dtype=np.float64)
    if (
        before_np.shape != after_np.shape
        or before_np.ndim != 2
        or before_np.shape[1] != STATE_SIZE
        or not np.isfinite(before_np).all()
        or not np.isfinite(after_np).all()
    ):
        raise ReverseControllerContractError("paired observable states are malformed")
    before_values = _observable_values(before_np, occurrence)
    after_values = _observable_values(after_np, occurrence)
    difference = np.ascontiguousarray(after_values - before_values)
    if not isinstance(structural_phase_invariants, (bool, np.bool_)):
        raise TypeError("structural_phase_invariants must be boolean")
    structural = np.zeros(len(OBSERVABLE_NAMES), dtype=np.bool_)
    color = PHASE_MATCHINGS[occurrence]
    maximum_structural_pair_mass_error = 0.0
    if structural_phase_invariants:
        # A one-occurrence transition is allowed to declare the appropriate
        # Fourier components structural only after verifying its actual local
        # pair identity.  This prevents a bad scatter/scheduler from being
        # hidden by simply overwriting a noisy global subtraction with zero.
        tails_all, heads_all = matching_indices(device="cpu")
        tails = tails_all[color].numpy()
        heads = heads_all[color].numpy()
        pair_before = before_np[:, tails] + before_np[:, heads]
        pair_after = after_np[:, tails] + after_np[:, heads]
        maximum_structural_pair_mass_error = float(
            np.max(np.abs(pair_after - pair_before), initial=0.0)
        )
        scale = max(
            1.0,
            float(np.max(np.abs(pair_before), initial=0.0)),
            float(np.max(np.abs(pair_after), initial=0.0)),
        )
        roundoff_bound = 8.0 * np.finfo(np.float64).eps * scale
        if maximum_structural_pair_mass_error > roundoff_bound:
            raise ReverseControllerContractError(
                "structural Fourier identity exceeds its explicit roundoff bound"
            )
        # Horizontal matchings preserve row-only (0,1); vertical matchings
        # preserve column-only (1,0).  Having verified the local identity, emit
        # the structural result as an exact binary64 zero.
        structural[2:4] = color < 2
        structural[0:2] = color >= 2
        difference[:, structural] = 0.0
    return PairedObservableResult(
        names=OBSERVABLE_NAMES,
        before=np.ascontiguousarray(before_values),
        after=np.ascontiguousarray(after_values),
        difference=difference,
        structural_invariant=structural,
        maximum_structural_pair_mass_error=maximum_structural_pair_mass_error,
    )


def bounded_linear_teacher_score(
    y: Any, exposure_time: Any, *, c: float = 0.5
) -> Any:
    """Exact bounded alpha=1 teacher ``y(1-y) d_y log rho_s``."""

    if not math.isfinite(float(c)) or abs(float(c)) >= 1.0:
        raise ReverseControllerContractError("teacher coefficient must lie in (-1,1)")
    if isinstance(y, Tensor) or isinstance(exposure_time, Tensor):
        value = y if isinstance(y, Tensor) else torch.as_tensor(y)
        time = exposure_time if isinstance(exposure_time, Tensor) else torch.as_tensor(
            exposure_time, device=value.device
        )
        value, time = torch.broadcast_tensors(
            value.to(torch.float64), time.to(torch.float64)
        )
        if bool(torch.any((value < 0.0) | (value > 1.0) | (time < 0.0))) or not bool(
            torch.isfinite(value).all() and torch.isfinite(time).all()
        ):
            raise ReverseControllerContractError("teacher coordinate is invalid")
        amplitude = float(c) * torch.exp(-2.0 * time)
        return 2.0 * amplitude * value * (1.0 - value) / (
            1.0 + amplitude * (2.0 * value - 1.0)
        )
    value_np, time_np = np.broadcast_arrays(
        np.asarray(y, dtype=np.float64), np.asarray(exposure_time, dtype=np.float64)
    )
    if (
        not np.isfinite(value_np).all()
        or not np.isfinite(time_np).all()
        or ((value_np < 0.0) | (value_np > 1.0) | (time_np < 0.0)).any()
    ):
        raise ReverseControllerContractError("teacher coordinate is invalid")
    amplitude = float(c) * np.exp(-2.0 * time_np)
    result = 2.0 * amplitude * value_np * (1.0 - value_np) / (
        1.0 + amplitude * (2.0 * value_np - 1.0)
    )
    return float(result) if result.ndim == 0 else np.ascontiguousarray(result)


def assert_unambiguous_metric_schema(value: Any) -> None:
    """Reject the historical, directionally ambiguous ``data_end`` token."""

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if "data_end" in str(key):
                    raise ReverseControllerContractError(
                        "new metric schemas must not contain data_end"
                    )
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str) and "data_end" in item:
            raise ReverseControllerContractError(
                "new metric schemas must not contain data_end"
            )

    visit(value)


def local_risk_gate(lower_bounds: Sequence[float]) -> bool:
    values = np.asarray(lower_bounds, dtype=np.float64)
    if values.shape != (LOCAL_RISK_FAMILY_SIZE,) or not np.isfinite(values).all():
        raise ReverseControllerContractError("local-risk max-T family must have 228 values")
    return bool(np.all(values > 0.0))


def trajectory_gate(
    reverse_law_absolute_upper_bounds: Sequence[float],
    refinement_absolute_upper_bounds: Sequence[float],
) -> bool:
    bias = np.asarray(reverse_law_absolute_upper_bounds, dtype=np.float64)
    refinement = np.asarray(refinement_absolute_upper_bounds, dtype=np.float64)
    expected = (TRAJECTORY_COMPONENT_COUNT,)
    if (
        bias.shape != expected
        or refinement.shape != expected
        or not np.isfinite(bias).all()
        or not np.isfinite(refinement).all()
        or (bias < 0.0).any()
        or (refinement < 0.0).any()
    ):
        raise ReverseControllerContractError(
            "trajectory family must contain 392 finite nonnegative bounds per statistic"
        )
    return bool(
        np.all(bias <= REVERSE_LAW_BIAS_MARGIN)
        and np.all(refinement <= MICROSTEP_REFINEMENT_MARGIN)
    )


def claim_boundary(*, controlled: bool) -> dict[str, int]:
    result = {
        "one_image_reconstruction_planning_authorized": int(bool(controlled)),
        "controller_control_trajectory_performed": int(bool(controlled)),
        "maximum_control_trajectory_phase_count": 8 if controlled else 0,
        "full_reverse_path_performed": 0,
        "reverse_sampling_performed": 0,
        "image_sampling_performed": 0,
        "reconstruction_performed": 0,
    }
    result.update({name: 0 for name in CLAIM_FLAG_NAMES})
    return result


def validate_claim_boundary(record: Mapping[str, Any]) -> None:
    if any(int(record.get(name, 1)) != 0 for name in CLAIM_FLAG_NAMES):
        raise ReverseControllerContractError("controller controls cannot authorize sampling claims")
    if (
        int(record.get("full_reverse_path_performed", 1)) != 0
        or int(record.get("reverse_sampling_performed", 1)) != 0
        or int(record.get("image_sampling_performed", 1)) != 0
        or int(record.get("reconstruction_performed", 1)) != 0
        or int(record.get("maximum_control_trajectory_phase_count", 9)) > 8
    ):
        raise ReverseControllerContractError("claim boundary exceeds eight-phase controls")


__all__ = [
    "ALLOWED_FRACTIONAL_COORDINATES",
    "CLAIM_FLAG_NAMES",
    "CONTROLLER_ROOT_SEED",
    "ControllerBoundaryStepRejected",
    "ControlledPhaseResult",
    "EXPECTED_PARENT_CHECKPOINT_SHA256",
    "EXPECTED_PARENT_STATE_SHA256",
    "FRACTION_TOLERANCE",
    "FractionalCoordinate",
    "FractionalFrozenController",
    "LOCAL_BOOTSTRAP_SEED",
    "LOCAL_RISK_BOOTSTRAP_REPLICATES",
    "LOCAL_RISK_CONFIDENCE",
    "LOCAL_RISK_FAMILY_SIZE",
    "MACROSTEP_SCHEDULE_INTEGRAL",
    "MICROSTEP_REFINEMENT_MARGIN",
    "MIDPOINT_FRACTIONS",
    "NAMESPACE_VERSION",
    "OBSERVABLE_NAMES",
    "ORACLE_PATH_IDS",
    "ORACLE_ROOT_SEED",
    "PHYSICAL_CONTROL_PATH_IDS",
    "PREFLIGHT_PATH_IDS",
    "PRODUCTION_CONTROLLER_MICROSTEPS",
    "PairedObservableResult",
    "REFINEMENT_CONTROL_MICROSTEPS",
    "REVERSE_CONTROLLER_VERSION",
    "REVERSE_LAW_BIAS_MARGIN",
    "ReverseControllerContractError",
    "TRAJECTORY_BOOTSTRAP_REPLICATES",
    "TRAJECTORY_BOOTSTRAP_SEED",
    "TRAJECTORY_COMPONENT_COUNT",
    "TRAJECTORY_CONFIDENCE",
    "TRAJECTORY_FAMILY_SIZE",
    "TRANSITION_ROLES",
    "assert_unambiguous_metric_schema",
    "bounded_linear_teacher_score",
    "claim_boundary",
    "controlled_reverse_phase",
    "controller_transition_id",
    "controller_transition_ids",
    "fractional_coordinate",
    "frozen_control_half_flow",
    "frozen_fractional_prediction",
    "internal_reverse_time",
    "learned_mass_flux",
    "load_frozen_controller",
    "local_risk_gate",
    "paired_observables",
    "phase_exposure",
    "reverse_execution_order",
    "trajectory_gate",
    "validate_claim_boundary",
    "validate_controller_path_plan",
]
