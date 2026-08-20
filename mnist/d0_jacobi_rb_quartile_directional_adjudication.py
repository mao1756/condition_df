"""Read-only directional and representation diagnostics for Jacobi/RB experts.

The routines in this module evaluate frozen boundary-tangent checkpoints.  They
never fit parameters, alter the raw Rao--Blackwell target, allocate paths, or
invoke a transition/reverse sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from mnist.d0_jacobi_rb_boundary_tangent import edge_pair_geometry
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    GRID_SIZE,
    PHASE_COUNT,
    STATE_SIZE,
    ModelInputs,
)


SCHEMA = "experiment12-d0-jacobi-rb-quartile-directional-adjudication"
SCHEMA_VERSION = 1
COMPONENT_NAMES = ("full", "local_affine", "spatial_cnn")
MIDPOINT_COUNT = 8
RECOMPOSITION_TOLERANCE = 5.0e-15
MAXIMUM_FORWARD_BATCH = 32


class QuartileDirectionalAdjudicationError(ValueError):
    """A frozen diagnostic algebra or representation contract was violated."""


def _readonly(value: Any, *, dtype: np.dtype[Any], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or not np.isfinite(array).all():
        raise QuartileDirectionalAdjudicationError(
            f"{name} must be finite with dtype {dtype.str}"
        )
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def quadratic_improvement(cross_term: Any, prediction_energy: Any, gain: float) -> Any:
    """Return ``2*gain*C-gain**2*P`` without modifying either moment."""

    value = float(gain)
    if not math.isfinite(value) or value < 0.0:
        raise QuartileDirectionalAdjudicationError(
            "diagnostic positive-ray gain must be finite and nonnegative"
        )
    cross = np.asarray(cross_term, dtype=np.float64)
    energy = np.asarray(prediction_energy, dtype=np.float64)
    if cross.shape != energy.shape or not (
        np.isfinite(cross).all() and np.isfinite(energy).all()
    ):
        raise QuartileDirectionalAdjudicationError("quadratic moments are malformed")
    result = 2.0 * value * cross - value * value * energy
    return float(result) if result.ndim == 0 else result


def positive_ray_optimum(
    target_energy: float, cross_term: float, prediction_energy: float
) -> dict[str, float]:
    """Return the frozen positive-ray scalar, ceiling and scale-free alignment."""

    target = float(target_energy)
    cross = float(cross_term)
    energy = float(prediction_energy)
    if not all(math.isfinite(value) for value in (target, cross, energy)):
        raise QuartileDirectionalAdjudicationError("directional moments are nonfinite")
    if target < 0.0 or energy < 0.0:
        raise QuartileDirectionalAdjudicationError("moment energies must be nonnegative")
    if energy == 0.0:
        if cross != 0.0:
            raise QuartileDirectionalAdjudicationError("P=0 with C!=0 is impossible")
        return {"lambda_plus": 0.0, "D_plus": 0.0, "rho": 0.0}
    if target == 0.0 and cross != 0.0:
        raise QuartileDirectionalAdjudicationError("T=0 with C!=0 is impossible")
    rho = 0.0 if target == 0.0 else cross / math.sqrt(target * energy)
    if cross <= 0.0:
        return {"lambda_plus": 0.0, "D_plus": 0.0, "rho": rho}
    return {
        "lambda_plus": cross / energy,
        "D_plus": cross * cross / energy,
        "rho": rho,
    }


@dataclass(frozen=True)
class ComponentPredictions:
    """Exact frozen decomposition of one model forward pass."""

    full: Tensor
    local_affine: Tensor
    spatial_cnn: Tensor
    q_full64: Tensor = field(repr=False)
    q_local64: Tensor = field(repr=False)
    q_spatial64_exact: Tensor = field(repr=False)
    q_spatial64_direct: Tensor = field(repr=False)
    rounding_bound: Tensor = field(repr=False)
    maximum_prediction_recomposition_error: float
    maximum_spatial_rounding_error: float

    def as_mapping(self) -> Mapping[str, Tensor]:
        return {
            "full": self.full,
            "local_affine": self.local_affine,
            "spatial_cnn": self.spatial_cnn,
        }


def evaluate_frozen_components(
    model: ZeroBaselineBoundaryTangentPredictor,
    inputs: ModelInputs,
) -> ComponentPredictions:
    """Evaluate the exact full/local/spatial decomposition of a frozen model.

    The full score is added in the model dtype, exactly as in the production
    forward.  The diagnostic spatial score is then defined by float64
    subtraction so the full prediction is unchanged.
    """

    if not isinstance(model, ZeroBaselineBoundaryTangentPredictor):
        raise QuartileDirectionalAdjudicationError("model has the wrong class")
    if type(inputs) is not ModelInputs or inputs.batch_size > MAXIMUM_FORWARD_BATCH:
        raise QuartileDirectionalAdjudicationError("input batch violates the firewall")
    network = model.residual_score
    state = inputs.later_full_state
    dtype = network.conv1.weight.dtype
    state_model = state.to(dtype=dtype)
    metadata = network._validated_metadata(inputs, dtype)  # noqa: SLF001
    batch = inputs.batch_size
    density = state_model.reshape(batch, 1, GRID_SIZE, GRID_SIZE) * float(STATE_SIZE)
    planes = metadata[:, :, None, None].expand(
        batch, metadata.shape[1], GRID_SIZE, GRID_SIZE
    )
    hidden = F.silu(network.conv1(torch.cat([density, planes], dim=1)))
    hidden = F.silu(network.conv2(hidden))
    hidden = F.silu(network.conv3(hidden))
    spatial = network.spatial_output(hidden).reshape(batch, 4, STATE_SIZE)

    colors = inputs.color.to(dtype=torch.long)
    rows = torch.arange(batch, device=state.device)
    heads = network.head_indices[colors]
    tails = network.tail_indices[colors]
    active_spatial = spatial[rows, colors].gather(1, heads)
    head_mass = state_model.gather(1, heads) * float(STATE_SIZE)
    tail_mass = state_model.gather(1, tails) * float(STATE_SIZE)
    local_metadata = metadata[:, None, :].expand(
        batch, EDGES_PER_PHASE, metadata.shape[1]
    )
    local_features = torch.cat(
        (tail_mass[:, :, None], head_mass[:, :, None], local_metadata), dim=2
    )
    local_model = network.local_affine(local_features).squeeze(-1)
    q_full_model = active_spatial + local_model

    q_full64 = q_full_model.to(dtype=torch.float64)
    q_local64 = local_model.to(dtype=torch.float64)
    q_spatial64_direct = active_spatial.to(dtype=torch.float64)
    q_spatial64_exact = q_full64 - q_local64
    finfo = torch.finfo(dtype)
    gamma1 = float(finfo.eps) / (1.0 - float(finfo.eps))
    bound = gamma1 * (q_local64.abs() + q_spatial64_direct.abs()) + float(finfo.tiny)
    spatial_error = (q_spatial64_exact - q_spatial64_direct).abs()
    if not bool(torch.all(spatial_error <= bound)):
        raise QuartileDirectionalAdjudicationError(
            "model-dtype branch addition exceeded its exact rounding bound"
        )

    mobility = edge_pair_geometry(inputs).mobility
    full = mobility * q_full64
    local = mobility * q_local64
    spatial_exact = full - local
    full = torch.where(mobility == 0.0, torch.zeros_like(full), full)
    local = torch.where(mobility == 0.0, torch.zeros_like(local), local)
    spatial_exact = torch.where(
        mobility == 0.0, torch.zeros_like(spatial_exact), spatial_exact
    )
    production = model(inputs).to(dtype=torch.float64)
    production_error = float(torch.max(torch.abs(full - production)).item())
    recomposition_error = float(torch.max(torch.abs(full - (local + spatial_exact))).item())
    if production_error != 0.0 or recomposition_error > RECOMPOSITION_TOLERANCE:
        raise QuartileDirectionalAdjudicationError(
            "component decomposition changed the frozen model prediction"
        )
    if not all(
        bool(torch.isfinite(value).all())
        for value in (full, local, spatial_exact, q_full64, q_local64)
    ):
        raise QuartileDirectionalAdjudicationError("component prediction is nonfinite")
    return ComponentPredictions(
        full=full,
        local_affine=local,
        spatial_cnn=spatial_exact,
        q_full64=q_full64,
        q_local64=q_local64,
        q_spatial64_exact=q_spatial64_exact,
        q_spatial64_direct=q_spatial64_direct,
        rounding_bound=bound,
        maximum_prediction_recomposition_error=recomposition_error,
        maximum_spatial_rounding_error=float(torch.max(spatial_error).item()),
    )


@dataclass(frozen=True)
class ComponentMomentCube:
    """Per-path, phase, midpoint moments for one frozen candidate and role."""

    path_ids: np.ndarray
    target_energy: np.ndarray
    cross_terms: np.ndarray
    prediction_energies: np.ndarray
    local_spatial_cross: np.ndarray
    counts: np.ndarray
    maximum_recomposition_error: float = 0.0
    maximum_risk_identity_error: float = 0.0

    def __post_init__(self) -> None:
        paths = np.asarray(self.path_ids)
        if paths.ndim != 1 or paths.dtype != np.int64 or np.unique(paths).size != paths.size:
            raise QuartileDirectionalAdjudicationError("path IDs are not canonical")
        shape = (paths.size, PHASE_COUNT, MIDPOINT_COUNT)
        target = _readonly(self.target_energy, dtype=np.dtype(np.float64), name="T")
        cross = _readonly(self.cross_terms, dtype=np.dtype(np.float64), name="C")
        energy = _readonly(
            self.prediction_energies, dtype=np.dtype(np.float64), name="P"
        )
        branch = _readonly(
            self.local_spatial_cross, dtype=np.dtype(np.float64), name="Q"
        )
        counts = np.asarray(self.counts)
        if (
            target.shape != shape
            or cross.shape != (len(COMPONENT_NAMES), *shape)
            or energy.shape != cross.shape
            or branch.shape != shape
            or counts.shape != shape
            or counts.dtype != np.int64
            or np.any(counts <= 0)
            or np.any(target < 0.0)
            or np.any(energy < 0.0)
        ):
            raise QuartileDirectionalAdjudicationError("moment cube shape/content changed")
        counts = np.ascontiguousarray(counts)
        counts.setflags(write=False)
        paths = np.ascontiguousarray(paths)
        paths.setflags(write=False)
        for value in (self.maximum_recomposition_error, self.maximum_risk_identity_error):
            if not math.isfinite(float(value)) or float(value) > RECOMPOSITION_TOLERANCE:
                raise QuartileDirectionalAdjudicationError("moment algebra check failed")
        object.__setattr__(self, "path_ids", paths)
        object.__setattr__(self, "target_energy", target)
        object.__setattr__(self, "cross_terms", cross)
        object.__setattr__(self, "prediction_energies", energy)
        object.__setattr__(self, "local_spatial_cross", branch)
        object.__setattr__(self, "counts", counts)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "path_ids": self.path_ids,
            "target_energy": self.target_energy,
            "cross_terms": self.cross_terms,
            "prediction_energies": self.prediction_energies,
            "local_spatial_cross": self.local_spatial_cross,
            "counts": self.counts,
            "maximum_recomposition_error": np.asarray(
                self.maximum_recomposition_error, dtype=np.float64
            ),
            "maximum_risk_identity_error": np.asarray(
                self.maximum_risk_identity_error, dtype=np.float64
            ),
        }

    @classmethod
    def from_arrays(cls, values: Mapping[str, Any]) -> "ComponentMomentCube":
        return cls(**{name: values[name] for name in (
            "path_ids", "target_energy", "cross_terms", "prediction_energies",
            "local_spatial_cross", "counts", "maximum_recomposition_error",
            "maximum_risk_identity_error",
        )})


class ComponentMomentAccumulator:
    """Canonical float64/``math.fsum`` reduction for streamed predictions."""

    def __init__(self, path_ids: Sequence[int]) -> None:
        paths = np.asarray(path_ids, dtype=np.int64)
        if paths.ndim != 1 or paths.size == 0 or np.unique(paths).size != paths.size:
            raise QuartileDirectionalAdjudicationError("path plan is invalid")
        self.path_ids = np.ascontiguousarray(paths)
        self._lookup = {int(value): index for index, value in enumerate(paths)}
        self._shape = (paths.size, PHASE_COUNT, MIDPOINT_COUNT)
        cells = int(np.prod(self._shape))
        self._target: list[list[float]] = [[] for _ in range(cells)]
        self._cross: list[list[list[float]]] = [
            [[] for _ in range(cells)] for _ in COMPONENT_NAMES
        ]
        self._energy: list[list[list[float]]] = [
            [[] for _ in range(cells)] for _ in COMPONENT_NAMES
        ]
        self._branch: list[list[float]] = [[] for _ in range(cells)]
        self._counts = np.zeros(self._shape, dtype=np.int64)
        self._max_recomposition = 0.0
        self._max_identity = 0.0

    def add_batch(
        self,
        *,
        path_id: Sequence[int] | np.ndarray,
        phase: Sequence[int] | np.ndarray,
        midpoint: Sequence[int] | np.ndarray,
        target: Tensor,
        predictions: ComponentPredictions,
    ) -> None:
        n = int(target.shape[0])
        if target.shape != (n, EDGES_PER_PHASE):
            raise QuartileDirectionalAdjudicationError("target batch shape changed")
        paths = np.asarray(path_id, dtype=np.int64)
        phases = np.asarray(phase, dtype=np.int64)
        midpoints = np.asarray(midpoint, dtype=np.int64)
        if any(array.shape != (n,) for array in (paths, phases, midpoints)):
            raise QuartileDirectionalAdjudicationError("row identity shape changed")
        truth = target.detach().to(dtype=torch.float64).cpu().numpy()
        component_values = [
            predictions.as_mapping()[name].detach().cpu().numpy()
            for name in COMPONENT_NAMES
        ]
        if not np.isfinite(truth).all() or any(
            not np.isfinite(value).all() for value in component_values
        ):
            raise QuartileDirectionalAdjudicationError("streamed values are nonfinite")
        row_target = np.mean(truth * truth, axis=1, dtype=np.float64)
        row_cross = [np.mean(truth * value, axis=1, dtype=np.float64) for value in component_values]
        row_energy = [np.mean(value * value, axis=1, dtype=np.float64) for value in component_values]
        row_branch = np.mean(component_values[1] * component_values[2], axis=1, dtype=np.float64)
        direct = np.mean(truth * truth - (truth - component_values[0]) ** 2, axis=1, dtype=np.float64)
        reconstructed = 2.0 * row_cross[0] - row_energy[0]
        self._max_identity = max(
            self._max_identity, float(np.max(np.abs(direct - reconstructed)))
        )
        self._max_recomposition = max(
            self._max_recomposition,
            predictions.maximum_prediction_recomposition_error,
            float(np.max(np.abs(row_cross[0] - row_cross[1] - row_cross[2]))),
            float(
                np.max(
                    np.abs(
                        row_energy[0]
                        - row_energy[1]
                        - row_energy[2]
                        - 2.0 * row_branch
                    )
                )
            ),
        )
        for row in range(n):
            if int(paths[row]) not in self._lookup:
                raise QuartileDirectionalAdjudicationError("unexpected path ID")
            p = self._lookup[int(paths[row])]
            q = int(phases[row])
            m = int(midpoints[row])
            if not (0 <= q < PHASE_COUNT and 0 <= m < MIDPOINT_COUNT):
                raise QuartileDirectionalAdjudicationError("stratum index changed")
            flat = np.ravel_multi_index((p, q, m), self._shape)
            self._target[flat].append(float(row_target[row]))
            for component in range(len(COMPONENT_NAMES)):
                self._cross[component][flat].append(float(row_cross[component][row]))
                self._energy[component][flat].append(float(row_energy[component][row]))
            self._branch[flat].append(float(row_branch[row]))
            self._counts[p, q, m] += 1

    def finish(self) -> ComponentMomentCube:
        if np.any(self._counts <= 0):
            raise QuartileDirectionalAdjudicationError("one or more cells are empty")

        def finish_one(parts: Sequence[Sequence[float]]) -> np.ndarray:
            out = np.empty(self._shape, dtype=np.float64)
            for index, values in enumerate(parts):
                out.flat[index] = math.fsum(values) / len(values)
            return out

        target = finish_one(self._target)
        cross = np.stack([finish_one(parts) for parts in self._cross], axis=0)
        energy = np.stack([finish_one(parts) for parts in self._energy], axis=0)
        branch = finish_one(self._branch)
        return ComponentMomentCube(
            path_ids=self.path_ids,
            target_energy=target,
            cross_terms=cross,
            prediction_energies=energy,
            local_spatial_cross=branch,
            counts=self._counts,
            maximum_recomposition_error=self._max_recomposition,
            maximum_risk_identity_error=self._max_identity,
        )


def weighted_pooled(values: Any, counts: Any) -> float:
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(counts, dtype=np.int64)
    if value.shape != weight.shape or np.any(weight <= 0) or not np.isfinite(value).all():
        raise QuartileDirectionalAdjudicationError("weighted reduction is malformed")
    numerator = math.fsum(
        float(v) * int(w) for v, w in zip(value.flat, weight.flat, strict=True)
    )
    return numerator / int(np.sum(weight, dtype=np.int64))


def marginalize(values: Any, counts: Any) -> dict[str, np.ndarray | float]:
    """Return pooled, path, phase, midpoint and 7x8 weighted means."""

    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(counts, dtype=np.int64)
    if value.shape != weight.shape or value.ndim != 3:
        raise QuartileDirectionalAdjudicationError("marginal cube shape changed")

    def reduce_axes(axes: tuple[int, ...]) -> np.ndarray:
        numerator = np.sum(value * weight, axis=axes, dtype=np.float64)
        denominator = np.sum(weight, axis=axes, dtype=np.int64)
        if np.any(denominator <= 0):
            raise QuartileDirectionalAdjudicationError("empty marginal")
        return numerator / denominator

    return {
        "pooled": weighted_pooled(value, weight),
        "path": reduce_axes((1, 2)),
        "phase": reduce_axes((0, 2)),
        "midpoint": reduce_axes((0, 1)),
        "cell": reduce_axes((0,)),
    }


def component_summary(cube: ComponentMomentCube, component: str) -> dict[str, Any]:
    if component not in COMPONENT_NAMES:
        raise QuartileDirectionalAdjudicationError("unknown component")
    index = COMPONENT_NAMES.index(component)
    target = marginalize(cube.target_energy, cube.counts)
    cross = marginalize(cube.cross_terms[index], cube.counts)
    energy = marginalize(cube.prediction_energies[index], cube.counts)
    optimum = positive_ray_optimum(
        float(target["pooled"]), float(cross["pooled"]), float(energy["pooled"])
    )
    return {
        "component": component,
        "T": float(target["pooled"]),
        "C": float(cross["pooled"]),
        "P": float(energy["pooled"]),
        **optimum,
        "path_C": np.asarray(cross["path"], dtype=np.float64),
        "path_P": np.asarray(energy["path"], dtype=np.float64),
        "phase_C": np.asarray(cross["phase"], dtype=np.float64),
        "midpoint_C": np.asarray(cross["midpoint"], dtype=np.float64),
        "cell_C": np.asarray(cross["cell"], dtype=np.float64),
    }


def normalized_cosine(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise QuartileDirectionalAdjudicationError("cosine operands are malformed")
    aa = math.fsum(float(value) * float(value) for value in a)
    bb = math.fsum(float(value) * float(value) for value in b)
    if aa == 0.0 or bb == 0.0:
        return 1.0 if aa == bb == 0.0 else 0.0
    ab = math.fsum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    return ab / math.sqrt(aa * bb)


__all__ = [
    "COMPONENT_NAMES",
    "MAXIMUM_FORWARD_BATCH",
    "MIDPOINT_COUNT",
    "RECOMPOSITION_TOLERANCE",
    "ComponentMomentAccumulator",
    "ComponentMomentCube",
    "ComponentPredictions",
    "QuartileDirectionalAdjudicationError",
    "component_summary",
    "evaluate_frozen_components",
    "marginalize",
    "normalized_cosine",
    "positive_ray_optimum",
    "quadratic_improvement",
    "weighted_pooled",
]
