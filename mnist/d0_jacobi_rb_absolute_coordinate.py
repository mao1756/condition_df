"""Read-only absolute-coordinate diagnostics for the exact Jacobi/RB target.

The module contains only deterministic projection, cross-panel inference, and
architecture-symmetry helpers.  It does not generate transitions, train a
model, open a confirmation role, construct a controller, or sample.

The coordinate decomposition is defined independently on every occurrence of
the seven-phase split.  For each 392-edge matching it separates

``dc``
    the phasewise constant direction,
``frequency1``
    sine/cosine head-coordinate modes at periodic frequency one,
``frequency2``
    sine/cosine head-coordinate modes at periodic frequency two after
    orthogonalization against the lower-frequency space, and
``residual``
    the exact orthogonal complement in the represented binary64 lattice.

Panel A may seal a direction in each subspace.  Panel B is then used only
linearly, so the confirmatory path statistics do not optimize a direction or
its sign on panel B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    GRID_SIZE,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    ModelInputs,
    PhaseConditionedLocalAffineCNN,
    call_model,
    matching_indices,
    semantic_sha256,
)


ABSOLUTE_COORDINATE_VERSION = "d0-jacobi-rb-absolute-coordinate-v1"
ABSOLUTE_COORDINATE_SCHEMA = ABSOLUTE_COORDINATE_VERSION + "-lattice"
QUARTILE_COUNT = 4
COORDINATE_COMPONENTS = ("dc", "frequency1", "frequency2", "residual")
LOW_FREQUENCIES = (1, 2)
DEFAULT_CONFIDENCE = 0.99
DEFAULT_BOOTSTRAP_REPLICATES = 50_000
DEFAULT_BOOTSTRAP_SEED = 261_361
DEFAULT_BOOTSTRAP_NAMESPACE = 0x41425343
DEFAULT_BOOTSTRAP_CHUNK_SIZE = 1_000
MAXIMUM_FORWARD_BATCH = 32


class AbsoluteCoordinateError(ValueError):
    """An absolute-coordinate algebra or inference contract was violated."""


def _readonly(value: Any, *, dtype: np.dtype[Any], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or not np.isfinite(array).all():
        raise AbsoluteCoordinateError(f"{name} must be finite {dtype}")
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _readonly_integer(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise AbsoluteCoordinateError(f"{name} must be integral")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    result.setflags(write=False)
    return result


def _array_sha256(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def coordinate_family_names(
    components: Sequence[str] = COORDINATE_COMPONENTS,
) -> tuple[str, ...]:
    selected = tuple(str(value) for value in components)
    if not selected or len(set(selected)) != len(selected) or any(
        value not in COORDINATE_COMPONENTS for value in selected
    ):
        raise AbsoluteCoordinateError("coordinate component family is malformed")
    return tuple(
        f"q{quartile}.{component}"
        for quartile in range(QUARTILE_COUNT)
        for component in selected
    )


@dataclass(frozen=True)
class CoordinateLattice:
    """Frozen phase/matching lattice and nested orthonormal bases."""

    phase_colors: np.ndarray = field(repr=False, compare=False)
    tail_indices: np.ndarray = field(repr=False, compare=False)
    head_indices: np.ndarray = field(repr=False, compare=False)
    head_rows: np.ndarray = field(repr=False, compare=False)
    head_columns: np.ndarray = field(repr=False, compare=False)
    dc_basis: np.ndarray = field(repr=False, compare=False)
    frequency1_basis: np.ndarray = field(repr=False, compare=False)
    frequency2_basis: np.ndarray = field(repr=False, compare=False)
    maximum_gram_error: float
    version: str = ABSOLUTE_COORDINATE_VERSION

    def __post_init__(self) -> None:
        if self.version != ABSOLUTE_COORDINATE_VERSION:
            raise AbsoluteCoordinateError("coordinate lattice version changed")
        phase_colors = _readonly_integer(self.phase_colors, name="phase_colors")
        tails = _readonly_integer(self.tail_indices, name="tail_indices")
        heads = _readonly_integer(self.head_indices, name="head_indices")
        rows = _readonly_integer(self.head_rows, name="head_rows")
        columns = _readonly_integer(self.head_columns, name="head_columns")
        dc = _readonly(self.dc_basis, dtype=np.dtype(np.float64), name="dc_basis")
        f1 = _readonly(
            self.frequency1_basis,
            dtype=np.dtype(np.float64),
            name="frequency1_basis",
        )
        f2 = _readonly(
            self.frequency2_basis,
            dtype=np.dtype(np.float64),
            name="frequency2_basis",
        )
        if phase_colors.shape != (PHASE_COUNT,) or not np.array_equal(
            phase_colors, np.asarray(PHASE_MATCHINGS, dtype=np.int64)
        ):
            raise AbsoluteCoordinateError("phase colors changed")
        edge_shape = (PHASE_COUNT, EDGES_PER_PHASE)
        if any(value.shape != edge_shape for value in (tails, heads, rows, columns)):
            raise AbsoluteCoordinateError("coordinate lattice edge arrays are malformed")
        if dc.shape != edge_shape + (1,) or f1.shape != edge_shape + (4,) or f2.shape != edge_shape + (4,):
            raise AbsoluteCoordinateError("coordinate basis ranks changed")
        if (
            ((tails < 0) | (tails >= GRID_SIZE * GRID_SIZE)).any()
            or ((heads < 0) | (heads >= GRID_SIZE * GRID_SIZE)).any()
            or ((rows < 0) | (rows >= GRID_SIZE)).any()
            or ((columns < 0) | (columns >= GRID_SIZE)).any()
            or not np.array_equal(rows, heads // GRID_SIZE)
            or not np.array_equal(columns, heads % GRID_SIZE)
        ):
            raise AbsoluteCoordinateError("coordinate lattice indices are invalid")
        gram_error = float(self.maximum_gram_error)
        if not math.isfinite(gram_error) or gram_error < 0.0 or gram_error > 5e-14:
            raise AbsoluteCoordinateError("coordinate basis is not orthonormal")
        for name, value in (
            ("phase_colors", phase_colors),
            ("tail_indices", tails),
            ("head_indices", heads),
            ("head_rows", rows),
            ("head_columns", columns),
            ("dc_basis", dc),
            ("frequency1_basis", f1),
            ("frequency2_basis", f2),
        ):
            object.__setattr__(self, name, value)

    @property
    def basis_sha256(self) -> str:
        return _array_sha256(
            self.phase_colors,
            self.tail_indices,
            self.head_indices,
            self.dc_basis,
            self.frequency1_basis,
            self.frequency2_basis,
        )

    def basis(self, component: str) -> np.ndarray:
        name = str(component)
        if name == "dc":
            return self.dc_basis
        if name == "frequency1":
            return self.frequency1_basis
        if name == "frequency2":
            return self.frequency2_basis
        raise AbsoluteCoordinateError(f"{name} has no finite-rank stored basis")

    def to_record(self) -> dict[str, Any]:
        body = {
            "schema": ABSOLUTE_COORDINATE_SCHEMA,
            "schema_version": 1,
            "version": self.version,
            "grid_size": GRID_SIZE,
            "phase_count": PHASE_COUNT,
            "edges_per_phase": EDGES_PER_PHASE,
            "phase_colors": self.phase_colors.tolist(),
            "components": list(COORDINATE_COMPONENTS),
            "dc_rank": 1,
            "frequency1_rank": 4,
            "frequency2_rank": 4,
            "residual_rank": EDGES_PER_PHASE - 9,
            "maximum_gram_error": float(self.maximum_gram_error),
            "basis_sha256": self.basis_sha256,
        }
        return {**body, "semantic_sha256": semantic_sha256(body)}


def build_coordinate_lattice() -> CoordinateLattice:
    """Build the deterministic nested periodic-coordinate lattice."""

    tails_by_color, heads_by_color = matching_indices(device="cpu")
    tails_color = tails_by_color.detach().cpu().numpy().astype(np.int64, copy=False)
    heads_color = heads_by_color.detach().cpu().numpy().astype(np.int64, copy=False)
    colors = np.asarray(PHASE_MATCHINGS, dtype=np.int64)
    tails = np.ascontiguousarray(tails_color[colors])
    heads = np.ascontiguousarray(heads_color[colors])
    rows = heads // GRID_SIZE
    columns = heads % GRID_SIZE
    dc = np.empty((PHASE_COUNT, EDGES_PER_PHASE, 1), dtype=np.float64)
    f1 = np.empty((PHASE_COUNT, EDGES_PER_PHASE, 4), dtype=np.float64)
    f2 = np.empty_like(f1)
    maximum_error = 0.0
    for phase in range(PHASE_COUNT):
        raw_columns: list[np.ndarray] = [np.ones(EDGES_PER_PHASE, dtype=np.float64)]
        for frequency in LOW_FREQUENCIES:
            for coordinate in (rows[phase], columns[phase]):
                angle = (
                    2.0
                    * math.pi
                    * float(frequency)
                    * coordinate.astype(np.float64)
                    / float(GRID_SIZE)
                )
                raw_columns.extend((np.sin(angle), np.cos(angle)))
        raw = np.stack(raw_columns, axis=1)
        basis, triangular = np.linalg.qr(raw, mode="reduced")
        signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
        basis = np.ascontiguousarray(basis * signs[None, :], dtype=np.float64)
        gram = basis.T @ basis
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(gram - np.eye(gram.shape[0], dtype=np.float64)))),
        )
        dc[phase] = basis[:, 0:1]
        f1[phase] = basis[:, 1:5]
        f2[phase] = basis[:, 5:9]
    return CoordinateLattice(
        phase_colors=colors,
        tail_indices=tails,
        head_indices=heads,
        head_rows=rows,
        head_columns=columns,
        dc_basis=dc,
        frequency1_basis=f1,
        frequency2_basis=f2,
        maximum_gram_error=maximum_error,
    )


def _coordinate_values(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float64)
        or array.ndim < 2
        or array.shape[-2:] != (PHASE_COUNT, EDGES_PER_PHASE)
        or not np.isfinite(array).all()
    ):
        raise AbsoluteCoordinateError(
            f"{name} must be finite binary64 [...,{PHASE_COUNT},{EDGES_PER_PHASE}]"
        )
    return np.ascontiguousarray(array)


def _basis_projection(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    for phase in range(PHASE_COUNT):
        coefficients = values[..., phase, :] @ basis[phase]
        result[..., phase, :] = coefficients @ basis[phase].T
    return np.ascontiguousarray(result)


def project_coordinate_components(
    value: Any,
    *,
    lattice: CoordinateLattice | None = None,
) -> dict[str, np.ndarray]:
    """Return the exact represented DC/F1/F2/residual decomposition."""

    values = _coordinate_values(value, name="coordinate values")
    active = lattice or build_coordinate_lattice()
    dc = _basis_projection(values, active.dc_basis)
    frequency1 = _basis_projection(values, active.frequency1_basis)
    frequency2 = _basis_projection(values, active.frequency2_basis)
    residual = np.ascontiguousarray(values - dc - frequency1 - frequency2)
    return {
        "dc": dc,
        "frequency1": frequency1,
        "frequency2": frequency2,
        "residual": residual,
    }


def project_coordinate_component(
    value: Any,
    component: str,
    *,
    lattice: CoordinateLattice | None = None,
) -> np.ndarray:
    name = str(component)
    if name not in COORDINATE_COMPONENTS:
        raise AbsoluteCoordinateError("unknown coordinate component")
    return project_coordinate_components(value, lattice=lattice)[name]


@dataclass(frozen=True)
class CoordinatePanel:
    """One immutable whole-path panel of coarse cell means."""

    role: str
    path_ids: np.ndarray = field(repr=False, compare=False)
    cell_means: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        role = str(self.role)
        paths = _readonly_integer(self.path_ids, name=f"{role}.path_ids")
        cells = _readonly(
            self.cell_means,
            dtype=np.dtype(np.float64),
            name=f"{role}.cell_means",
        )
        if not role or paths.ndim != 1 or paths.size < 2:
            raise AbsoluteCoordinateError("coordinate panel role/paths are malformed")
        if (
            np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths, kind="stable"))
            or ((paths < 0) | (paths >= (1 << 20))).any()
        ):
            raise AbsoluteCoordinateError("coordinate panel paths are not canonical")
        expected = (paths.size, QUARTILE_COUNT, PHASE_COUNT, EDGES_PER_PHASE)
        if cells.shape != expected:
            raise AbsoluteCoordinateError(f"coordinate panel must have shape {expected}")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "path_ids", paths)
        object.__setattr__(self, "cell_means", cells)

    @property
    def fingerprint(self) -> str:
        return _array_sha256(self.path_ids, self.cell_means)


def _require_disjoint_panels(left: CoordinatePanel, right: CoordinatePanel) -> None:
    if not isinstance(left, CoordinatePanel) or not isinstance(right, CoordinatePanel):
        raise AbsoluteCoordinateError("cross-panel inputs have wrong type")
    if left.role == right.role or np.intersect1d(left.path_ids, right.path_ids).size:
        raise AbsoluteCoordinateError("cross-panel roles must be independent")


@dataclass(frozen=True)
class CrossPanelDecomposition:
    component_names: tuple[str, ...]
    left_path_ids: np.ndarray = field(repr=False, compare=False)
    right_path_ids: np.ndarray = field(repr=False, compare=False)
    component_kernels: np.ndarray = field(repr=False, compare=False)
    component_point_energies: np.ndarray = field(repr=False, compare=False)
    full_point_energies: np.ndarray = field(repr=False, compare=False)
    maximum_reconstruction_error: float

    def __post_init__(self) -> None:
        names = tuple(self.component_names)
        if names != COORDINATE_COMPONENTS:
            raise AbsoluteCoordinateError("cross-panel component order changed")
        left = _readonly_integer(self.left_path_ids, name="left_path_ids")
        right = _readonly_integer(self.right_path_ids, name="right_path_ids")
        kernels = _readonly(
            self.component_kernels,
            dtype=np.dtype(np.float64),
            name="component_kernels",
        )
        points = _readonly(
            self.component_point_energies,
            dtype=np.dtype(np.float64),
            name="component_point_energies",
        )
        full = _readonly(
            self.full_point_energies,
            dtype=np.dtype(np.float64),
            name="full_point_energies",
        )
        if kernels.shape != (
            len(names),
            QUARTILE_COUNT,
            left.size,
            right.size,
        ) or points.shape != (len(names), QUARTILE_COUNT) or full.shape != (
            QUARTILE_COUNT,
        ):
            raise AbsoluteCoordinateError("cross-panel decomposition shape changed")
        error = float(self.maximum_reconstruction_error)
        if not math.isfinite(error) or error < 0.0:
            raise AbsoluteCoordinateError("cross-panel reconstruction error is invalid")
        object.__setattr__(self, "left_path_ids", left)
        object.__setattr__(self, "right_path_ids", right)
        object.__setattr__(self, "component_kernels", kernels)
        object.__setattr__(self, "component_point_energies", points)
        object.__setattr__(self, "full_point_energies", full)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": ABSOLUTE_COORDINATE_VERSION + "-cross-panel-decomposition",
            "schema_version": 1,
            "component_names": list(self.component_names),
            "left_path_count": int(self.left_path_ids.size),
            "right_path_count": int(self.right_path_ids.size),
            "component_point_energies": self.component_point_energies.tolist(),
            "full_point_energies": self.full_point_energies.tolist(),
            "maximum_reconstruction_error": float(self.maximum_reconstruction_error),
            "negative_values_truncated": 0,
        }


def decompose_cross_panel_signal(
    left: CoordinatePanel,
    right: CoordinatePanel,
    *,
    lattice: CoordinateLattice | None = None,
) -> CrossPanelDecomposition:
    """Compute signed independent-panel energy in orthogonal subspaces."""

    _require_disjoint_panels(left, right)
    active = lattice or build_coordinate_lattice()
    left_parts = project_coordinate_components(left.cell_means, lattice=active)
    right_parts = project_coordinate_components(right.cell_means, lattice=active)
    denominator = float(PHASE_COUNT * EDGES_PER_PHASE)
    kernels = np.empty(
        (
            len(COORDINATE_COMPONENTS),
            QUARTILE_COUNT,
            left.path_ids.size,
            right.path_ids.size,
        ),
        dtype=np.float64,
    )
    for component_index, component in enumerate(COORDINATE_COMPONENTS):
        left_flat = left_parts[component].reshape(
            left.path_ids.size, QUARTILE_COUNT, -1
        ).transpose(1, 0, 2)
        right_flat = right_parts[component].reshape(
            right.path_ids.size, QUARTILE_COUNT, -1
        ).transpose(1, 0, 2)
        kernels[component_index] = (
            np.einsum("qid,qjd->qij", left_flat, right_flat, optimize=True)
            / denominator
        )
    points = np.mean(kernels, axis=(2, 3), dtype=np.float64)
    left_full = left.cell_means.reshape(left.path_ids.size, QUARTILE_COUNT, -1).transpose(1, 0, 2)
    right_full = right.cell_means.reshape(right.path_ids.size, QUARTILE_COUNT, -1).transpose(1, 0, 2)
    full_kernel = (
        np.einsum("qid,qjd->qij", left_full, right_full, optimize=True)
        / denominator
    )
    full_points = np.mean(full_kernel, axis=(1, 2), dtype=np.float64)
    reconstruction = np.sum(kernels, axis=0, dtype=np.float64)
    error = float(np.max(np.abs(full_kernel - reconstruction)))
    return CrossPanelDecomposition(
        component_names=COORDINATE_COMPONENTS,
        left_path_ids=left.path_ids,
        right_path_ids=right.path_ids,
        component_kernels=kernels,
        component_point_energies=points,
        full_point_energies=full_points,
        maximum_reconstruction_error=error,
    )


@dataclass(frozen=True)
class ADirectionSeal:
    """Panel-A-only normalized directions, frozen before panel B opens."""

    family_names: tuple[str, ...]
    components: tuple[str, ...]
    panel_a_role: str
    panel_a_path_ids: np.ndarray = field(repr=False, compare=False)
    directions: np.ndarray = field(repr=False, compare=False)
    direction_norms: np.ndarray = field(repr=False, compare=False)
    direction_active_mask: np.ndarray = field(repr=False, compare=False)
    basis_sha256: str
    panel_a_sha256: str

    def __post_init__(self) -> None:
        components = tuple(self.components)
        names = tuple(self.family_names)
        if names != coordinate_family_names(components):
            raise AbsoluteCoordinateError("A-direction family order changed")
        if not isinstance(self.panel_a_role, str) or not self.panel_a_role:
            raise AbsoluteCoordinateError("Panel-A role is malformed")
        paths = _readonly_integer(self.panel_a_path_ids, name="panel_a_path_ids")
        directions = _readonly(
            self.directions, dtype=np.dtype(np.float64), name="directions"
        )
        norms = _readonly(
            self.direction_norms,
            dtype=np.dtype(np.float64),
            name="direction_norms",
        )
        mask_raw = np.asarray(self.direction_active_mask)
        if mask_raw.dtype != np.dtype(np.bool_):
            raise AbsoluteCoordinateError("direction active mask must be boolean")
        mask = np.ascontiguousarray(mask_raw)
        mask.setflags(write=False)
        family_count = len(names)
        if (
            paths.ndim != 1
            or paths.size < 2
            or np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths, kind="stable"))
            or ((paths < 0) | (paths >= (1 << 20))).any()
            or directions.shape != (family_count, PHASE_COUNT, EDGES_PER_PHASE)
            or norms.shape != (family_count,)
            or mask.shape != (family_count,)
            or np.any(norms < 0.0)
            or np.any(mask != (norms > 0.0))
        ):
            raise AbsoluteCoordinateError("A-direction arrays are malformed")
        represented_norm = np.sqrt(
            np.mean(directions * directions, axis=(1, 2), dtype=np.float64)
        )
        if np.any(np.abs(represented_norm[mask] - 1.0) > 5e-14) or np.any(
            directions[~mask] != 0.0
        ):
            raise AbsoluteCoordinateError("A directions are not unit RMS/zero")
        for value, description in (
            (self.basis_sha256, "basis_sha256"),
            (self.panel_a_sha256, "panel_a_sha256"),
        ):
            if not (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            ):
                raise AbsoluteCoordinateError(f"{description} is malformed")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "family_names", names)
        object.__setattr__(self, "panel_a_path_ids", paths)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "direction_norms", norms)
        object.__setattr__(self, "direction_active_mask", mask)

    @property
    def directions_sha256(self) -> str:
        return _array_sha256(
            self.panel_a_path_ids,
            self.directions,
            self.direction_norms,
            self.direction_active_mask.astype(np.uint8),
        )

    @property
    def seal_sha256(self) -> str:
        return semantic_sha256(
            {
                "schema": ABSOLUTE_COORDINATE_VERSION + "-a-direction-seal",
                "family_names": list(self.family_names),
                "panel_a_role": self.panel_a_role,
                "panel_a_sha256": self.panel_a_sha256,
                "basis_sha256": self.basis_sha256,
                "directions_sha256": self.directions_sha256,
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": ABSOLUTE_COORDINATE_VERSION + "-a-direction-seal",
            "schema_version": 1,
            "family_names": list(self.family_names),
            "components": list(self.components),
            "panel_a_role": self.panel_a_role,
            "panel_a_path_ids": self.panel_a_path_ids.tolist(),
            "panel_a_sha256": self.panel_a_sha256,
            "basis_sha256": self.basis_sha256,
            "direction_norms": self.direction_norms.tolist(),
            "direction_active_mask": self.direction_active_mask.astype(np.int8).tolist(),
            "directions_sha256": self.directions_sha256,
            "seal_sha256": self.seal_sha256,
            "panel_b_opened": 0,
            "negative_directions_truncated": 0,
        }


def seal_panel_a_directions(
    panel_a: CoordinatePanel,
    *,
    lattice: CoordinateLattice | None = None,
    components: Sequence[str] = COORDINATE_COMPONENTS,
) -> ADirectionSeal:
    """Freeze normalized Panel-A means without inspecting Panel B."""

    if not isinstance(panel_a, CoordinatePanel):
        raise AbsoluteCoordinateError("panel A has wrong type")
    selected = tuple(str(value) for value in components)
    names = coordinate_family_names(selected)
    active = lattice or build_coordinate_lattice()
    projections = project_coordinate_components(panel_a.cell_means, lattice=active)
    directions = np.zeros((len(names), PHASE_COUNT, EDGES_PER_PHASE), dtype=np.float64)
    norms = np.zeros(len(names), dtype=np.float64)
    cursor = 0
    for quartile in range(QUARTILE_COUNT):
        for component in selected:
            mean_direction = np.mean(
                projections[component][:, quartile], axis=0, dtype=np.float64
            )
            norm = math.sqrt(float(np.mean(mean_direction * mean_direction)))
            if not math.isfinite(norm):
                raise AbsoluteCoordinateError("Panel-A direction norm is nonfinite")
            norms[cursor] = norm
            if norm > 0.0:
                directions[cursor] = mean_direction / norm
            cursor += 1
    return ADirectionSeal(
        family_names=names,
        components=selected,
        panel_a_role=panel_a.role,
        panel_a_path_ids=panel_a.path_ids,
        directions=directions,
        direction_norms=norms,
        direction_active_mask=norms > 0.0,
        basis_sha256=active.basis_sha256,
        panel_a_sha256=panel_a.fingerprint,
    )


@dataclass(frozen=True)
class BLinearEvidence:
    family_names: tuple[str, ...]
    panel_b_role: str
    path_ids: np.ndarray = field(repr=False, compare=False)
    path_values: np.ndarray = field(repr=False, compare=False)
    point_estimates: np.ndarray = field(repr=False, compare=False)
    direction_norms: np.ndarray = field(repr=False, compare=False)
    signed_cross_energies: np.ndarray = field(repr=False, compare=False)
    a_direction_seal_sha256: str

    def __post_init__(self) -> None:
        names = tuple(self.family_names)
        paths = _readonly_integer(self.path_ids, name="B path IDs")
        values = _readonly(
            self.path_values, dtype=np.dtype(np.float64), name="B path values"
        )
        points = _readonly(
            self.point_estimates,
            dtype=np.dtype(np.float64),
            name="B point estimates",
        )
        norms = _readonly(
            self.direction_norms,
            dtype=np.dtype(np.float64),
            name="A direction norms",
        )
        cross = _readonly(
            self.signed_cross_energies,
            dtype=np.dtype(np.float64),
            name="signed cross energies",
        )
        seal_hash_valid = (
            isinstance(self.a_direction_seal_sha256, str)
            and len(self.a_direction_seal_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in self.a_direction_seal_sha256
            )
        )
        if (
            not names
            or len(set(names)) != len(names)
            or not isinstance(self.panel_b_role, str)
            or not self.panel_b_role
            or paths.ndim != 1
            or paths.size < 2
            or np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths, kind="stable"))
            or ((paths < 0) | (paths >= (1 << 20))).any()
            or values.shape != (paths.size, len(names))
            or points.shape != (len(names),)
            or norms.shape != points.shape
            or cross.shape != points.shape
            or not np.array_equal(points, np.mean(values, axis=0, dtype=np.float64))
            or not np.allclose(cross, norms * points, rtol=0.0, atol=5e-15)
            or not seal_hash_valid
        ):
            raise AbsoluteCoordinateError("B-linear evidence is malformed")
        object.__setattr__(self, "family_names", names)
        object.__setattr__(self, "path_ids", paths)
        object.__setattr__(self, "path_values", values)
        object.__setattr__(self, "point_estimates", points)
        object.__setattr__(self, "direction_norms", norms)
        object.__setattr__(self, "signed_cross_energies", cross)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": ABSOLUTE_COORDINATE_VERSION + "-b-linear-evidence",
            "schema_version": 1,
            "family_names": list(self.family_names),
            "panel_b_role": self.panel_b_role,
            "panel_b_path_ids": self.path_ids.tolist(),
            "point_estimates": self.point_estimates.tolist(),
            "direction_norms": self.direction_norms.tolist(),
            "signed_cross_energies": self.signed_cross_energies.tolist(),
            "a_direction_seal_sha256": self.a_direction_seal_sha256,
            "negative_values_truncated": 0,
        }


def evaluate_panel_b_linear(
    seal: ADirectionSeal,
    panel_b: CoordinatePanel,
    *,
    lattice: CoordinateLattice | None = None,
) -> BLinearEvidence:
    """Evaluate the sealed Panel-A directions linearly on Panel B."""

    if not isinstance(seal, ADirectionSeal) or not isinstance(panel_b, CoordinatePanel):
        raise AbsoluteCoordinateError("A seal/B panel types are invalid")
    if seal.panel_a_role == panel_b.role or np.intersect1d(
        seal.panel_a_path_ids, panel_b.path_ids
    ).size:
        raise AbsoluteCoordinateError("Panel B is not independent from Panel A")
    active = lattice or build_coordinate_lattice()
    if seal.basis_sha256 != active.basis_sha256:
        raise AbsoluteCoordinateError("A-direction basis binding changed")
    values = np.empty((panel_b.path_ids.size, len(seal.family_names)), dtype=np.float64)
    denominator = float(PHASE_COUNT * EDGES_PER_PHASE)
    cursor = 0
    for quartile in range(QUARTILE_COUNT):
        for _component in seal.components:
            values[:, cursor] = np.einsum(
                "ped,ed->p",
                panel_b.cell_means[:, quartile],
                seal.directions[cursor],
                optimize=True,
            ) / denominator
            cursor += 1
    points = np.mean(values, axis=0, dtype=np.float64)
    return BLinearEvidence(
        family_names=seal.family_names,
        panel_b_role=panel_b.role,
        path_ids=panel_b.path_ids,
        path_values=values,
        point_estimates=points,
        direction_norms=seal.direction_norms,
        signed_cross_energies=seal.direction_norms * points,
        a_direction_seal_sha256=seal.seal_sha256,
    )


@dataclass(frozen=True)
class LinearMaxTResult:
    family_names: tuple[str, ...]
    path_ids: np.ndarray = field(repr=False, compare=False)
    point_estimates: np.ndarray = field(repr=False, compare=False)
    standard_errors: np.ndarray = field(repr=False, compare=False)
    lower_bounds: np.ndarray = field(repr=False, compare=False)
    analytic_constant_mask: np.ndarray = field(repr=False, compare=False)
    bootstrap_maxima: np.ndarray = field(repr=False, compare=False)
    critical_value: float
    confidence: float
    replicates: int
    seed: int
    namespace: int

    def __post_init__(self) -> None:
        names = tuple(self.family_names)
        paths = _readonly_integer(self.path_ids, name="max-T path IDs")
        point = _readonly(
            self.point_estimates, dtype=np.dtype(np.float64), name="max-T points"
        )
        error = _readonly(
            self.standard_errors, dtype=np.dtype(np.float64), name="max-T errors"
        )
        lower = _readonly(
            self.lower_bounds, dtype=np.dtype(np.float64), name="max-T bounds"
        )
        maxima = _readonly(
            self.bootstrap_maxima,
            dtype=np.dtype(np.float64),
            name="max-T maxima",
        )
        mask_raw = np.asarray(self.analytic_constant_mask)
        if mask_raw.dtype != np.dtype(np.bool_):
            raise AbsoluteCoordinateError("max-T constant mask must be boolean")
        mask = np.ascontiguousarray(mask_raw)
        mask.setflags(write=False)
        if (
            not names
            or len(set(names)) != len(names)
            or point.shape != (len(names),)
            or error.shape != point.shape
            or lower.shape != point.shape
            or mask.shape != point.shape
            or paths.size < 2
            or paths.ndim != 1
            or np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths, kind="stable"))
            or ((paths < 0) | (paths >= (1 << 20))).any()
            or maxima.shape != (int(self.replicates),)
            or np.any(error < 0.0)
            or np.any(mask != (error == 0.0))
            or not math.isfinite(float(self.critical_value))
            or float(self.critical_value) < 0.0
            or not 0.0 < float(self.confidence) < 1.0
            or isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or self.replicates <= 0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or isinstance(self.namespace, bool)
            or not isinstance(self.namespace, int)
        ):
            raise AbsoluteCoordinateError("max-T result is malformed")
        object.__setattr__(self, "family_names", names)
        object.__setattr__(self, "path_ids", paths)
        object.__setattr__(self, "point_estimates", point)
        object.__setattr__(self, "standard_errors", error)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "analytic_constant_mask", mask)
        object.__setattr__(self, "bootstrap_maxima", maxima)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": ABSOLUTE_COORDINATE_VERSION + "-b-linear-max-t",
            "schema_version": 1,
            "family_names": list(self.family_names),
            "path_ids": self.path_ids.tolist(),
            "point_estimates": self.point_estimates.tolist(),
            "standard_errors": self.standard_errors.tolist(),
            "lower_bounds": self.lower_bounds.tolist(),
            "analytic_constant_mask": self.analytic_constant_mask.astype(np.int8).tolist(),
            "critical_value": float(self.critical_value),
            "confidence": float(self.confidence),
            "replicates": int(self.replicates),
            "seed": int(self.seed),
            "namespace": int(self.namespace),
            "negative_values_truncated": 0,
        }


def one_sided_b_linear_max_t(
    evidence: BLinearEvidence,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    namespace: int = DEFAULT_BOOTSTRAP_NAMESPACE,
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK_SIZE,
) -> LinearMaxTResult:
    """Whole-path one-sided studentized max-T for sealed B-linear values."""

    if not isinstance(evidence, BLinearEvidence):
        raise AbsoluteCoordinateError("B-linear evidence has wrong type")
    if (
        not 0.0 < float(confidence) < 1.0
        or isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(namespace, bool)
        or not isinstance(namespace, int)
    ):
        raise AbsoluteCoordinateError("B-linear max-T configuration is invalid")
    values = evidence.path_values
    path_count = int(evidence.path_ids.size)
    point = np.mean(values, axis=0, dtype=np.float64)
    error = np.std(values, axis=0, ddof=1, dtype=np.float64) / math.sqrt(path_count)
    constants = error == 0.0
    stochastic = ~constants
    maxima = np.zeros(replicates, dtype=np.float64)
    generator = np.random.Generator(np.random.Philox([int(seed), int(namespace)]))
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        indices = generator.integers(
            0,
            path_count,
            size=(stop - start, path_count),
            dtype=np.int64,
        )
        sampled = values[indices]
        sampled_mean = np.mean(sampled, axis=1, dtype=np.float64)
        if np.any(stochastic):
            sampled_error = np.std(
                sampled[:, :, stochastic], axis=1, ddof=1, dtype=np.float64
            ) / math.sqrt(path_count)
            if not np.isfinite(sampled_error).all() or np.any(sampled_error <= 0.0):
                raise AbsoluteCoordinateError(
                    "bootstrap produced degenerate/nonfinite studentization"
                )
            centered = (
                sampled_mean[:, stochastic] - point[None, stochastic]
            ) / sampled_error
            maxima[start:stop] = np.maximum(0.0, np.max(centered, axis=1))
    critical = float(np.quantile(maxima, float(confidence), method="higher"))
    lower = point - critical * error
    lower[constants] = point[constants]
    return LinearMaxTResult(
        family_names=evidence.family_names,
        path_ids=evidence.path_ids,
        point_estimates=point,
        standard_errors=error,
        lower_bounds=lower,
        analytic_constant_mask=constants,
        bootstrap_maxima=maxima,
        critical_value=critical,
        confidence=float(confidence),
        replicates=int(replicates),
        seed=int(seed),
        namespace=int(namespace),
    )


def scaled_signed_cross_bounds(
    seal: ADirectionSeal, inference: LinearMaxTResult
) -> np.ndarray:
    """Convert unit-direction B bounds back to signed cross-energy units."""

    if tuple(inference.family_names) != tuple(seal.family_names):
        raise AbsoluteCoordinateError("A seal/max-T family mismatch")
    return np.ascontiguousarray(seal.direction_norms * inference.lower_bounds)


def phase_predictor_architecture_contract(
    model: PhaseConditionedLocalAffineCNN | None = None,
) -> dict[str, Any]:
    """Describe the frozen coordinate-free, translation-equivariant model."""

    active = model or PhaseConditionedLocalAffineCNN(width=32)
    if not isinstance(active, PhaseConditionedLocalAffineCNN):
        raise AbsoluteCoordinateError("architecture helper requires the phase predictor")
    convolution_layers = (active.conv1, active.conv2, active.conv3)
    circular = all(layer.padding_mode == "circular" for layer in convolution_layers)
    three_by_three = all(tuple(layer.kernel_size) == (3, 3) for layer in convolution_layers)
    coordinate_buffers = tuple(
        name
        for name, _value in active.named_buffers()
        if "coordinate" in name.lower() or "position" in name.lower()
    )
    metadata_channels = 1 + PHASE_COUNT + 4 + 1 + active.num_classes
    expected_input_channels = 1 + metadata_channels
    expected_local_features = 2 + metadata_channels
    checks = {
        "model_width_32": int(active.width == 32),
        "input_channels_without_coordinates": int(
            active.conv1.in_channels == expected_input_channels
        ),
        "local_features_without_coordinates": int(
            active.local_affine.in_features == expected_local_features
        ),
        "three_circular_three_by_three_convolutions": int(circular and three_by_three),
        "periodic_coordinate_buffer_absent": int(not coordinate_buffers),
        "four_matching_outputs": int(active.spatial_output.out_channels == 4),
    }
    passed = int(all(checks.values()))
    return {
        "schema": ABSOLUTE_COORDINATE_VERSION + "-architecture-contract",
        "schema_version": 1,
        "model_width": int(active.width),
        "input_channels": int(active.conv1.in_channels),
        "metadata_channels": int(metadata_channels),
        "local_affine_features": int(active.local_affine.in_features),
        "periodic_coordinate_buffers": list(coordinate_buffers),
        "effective_receptive_field": 7,
        "translation_equivariant_by_construction": passed,
        "checks": checks,
        "passed": passed,
    }


def edge_translation_permutation(
    color: int, row_shift: int, column_shift: int
) -> np.ndarray:
    """Return source-edge indices for a torus translation of one matching."""

    if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color < 4:
        raise AbsoluteCoordinateError("matching color must lie in [0,4)")
    if isinstance(row_shift, bool) or not isinstance(row_shift, int):
        raise AbsoluteCoordinateError("row shift must be integral")
    if isinstance(column_shift, bool) or not isinstance(column_shift, int):
        raise AbsoluteCoordinateError("column shift must be integral")
    _tails, heads = matching_indices(device="cpu")
    values = heads[color].detach().cpu().numpy().astype(np.int64, copy=False)
    coordinates = {(int(value // GRID_SIZE), int(value % GRID_SIZE)): index for index, value in enumerate(values)}
    permutation = np.empty(EDGES_PER_PHASE, dtype=np.int64)
    for destination, value in enumerate(values):
        row = int(value // GRID_SIZE)
        column = int(value % GRID_SIZE)
        source = (
            (row - row_shift) % GRID_SIZE,
            (column - column_shift) % GRID_SIZE,
        )
        if source not in coordinates:
            raise AbsoluteCoordinateError(
                "translation does not preserve the oriented matching"
            )
        permutation[destination] = coordinates[source]
    permutation.setflags(write=False)
    return permutation


def translate_model_inputs(
    inputs: ModelInputs, *, row_shift: int, column_shift: int
) -> ModelInputs:
    if type(inputs) is not ModelInputs:
        raise AbsoluteCoordinateError("translation requires exact ModelInputs")
    state = inputs.later_full_state.reshape(
        inputs.batch_size, GRID_SIZE, GRID_SIZE
    )
    translated = torch.roll(state, shifts=(row_shift, column_shift), dims=(1, 2))
    return ModelInputs(
        later_full_state=translated.reshape(inputs.batch_size, -1),
        reverse_time=inputs.reverse_time,
        phase=inputs.phase,
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )


def model_translation_equivariance_record(
    model: nn.Module,
    inputs: ModelInputs,
    *,
    row_shift: int = 2,
    column_shift: int = 2,
    tolerance: float = 2e-6,
) -> dict[str, Any]:
    """Evaluate exact matching-aware translation equivariance on permitted inputs."""

    if not isinstance(model, PhaseConditionedLocalAffineCNN):
        raise AbsoluteCoordinateError("equivariance helper requires phase predictor")
    if type(inputs) is not ModelInputs or inputs.batch_size > MAXIMUM_FORWARD_BATCH:
        raise AbsoluteCoordinateError("equivariance inputs violate the batch contract")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise AbsoluteCoordinateError("equivariance tolerance is invalid")
    translated = translate_model_inputs(
        inputs, row_shift=row_shift, column_shift=column_shift
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            original = call_model(model, inputs).to(dtype=torch.float64)
            observed = call_model(model, translated).to(dtype=torch.float64)
    finally:
        model.train(was_training)
    expected = torch.empty_like(original)
    colors = inputs.color.detach().to(device="cpu", dtype=torch.long).numpy()
    for row, color in enumerate(colors):
        permutation = edge_translation_permutation(
            int(color), row_shift, column_shift
        )
        indices = torch.as_tensor(
            np.array(permutation, copy=True),
            dtype=torch.long,
            device=original.device,
        )
        expected[row] = original[row].index_select(0, indices)
    error = float(torch.max(torch.abs(observed - expected)).item())
    return {
        "schema": ABSOLUTE_COORDINATE_VERSION + "-translation-equivariance",
        "schema_version": 1,
        "row_shift": int(row_shift),
        "column_shift": int(column_shift),
        "batch_size": inputs.batch_size,
        "maximum_translation_equivariance_error": error,
        "tolerance": float(tolerance),
        "passed": int(math.isfinite(error) and error <= float(tolerance)),
    }


def synthetic_model_inputs(
    *, batch_size: int = 7, seed: int = 261_362
) -> ModelInputs:
    """Build deterministic permitted inputs for symmetry regression fixtures."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= MAXIMUM_FORWARD_BATCH
    ):
        raise AbsoluteCoordinateError("synthetic batch size is invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    state = torch.rand(
        (batch_size, GRID_SIZE * GRID_SIZE),
        dtype=torch.float32,
        generator=generator,
    )
    state = state / torch.sum(state, dim=1, keepdim=True)
    phases = torch.arange(batch_size, dtype=torch.long) % PHASE_COUNT
    colors = torch.as_tensor(PHASE_MATCHINGS, dtype=torch.long)[phases]
    durations = torch.as_tensor(PHASE_DURATIONS, dtype=torch.float32)[phases]
    return ModelInputs(
        later_full_state=state,
        reverse_time=torch.linspace(0.05, 0.95, batch_size, dtype=torch.float32),
        phase=phases,
        color=colors,
        duration=durations,
        label=torch.full((batch_size,), 3, dtype=torch.long),
    )


@dataclass(frozen=True)
class SyntheticCoordinateFixture:
    left: CoordinatePanel
    right: CoordinatePanel
    signal: np.ndarray = field(repr=False, compare=False)
    component_amplitudes: Mapping[str, tuple[float, ...]]


def synthetic_coordinate_fixture(
    *,
    path_count: int = 16,
    noise_scale: float = 0.05,
    seed: int = 261_363,
    component_amplitudes: Mapping[str, Sequence[float]] | None = None,
    lattice: CoordinateLattice | None = None,
) -> SyntheticCoordinateFixture:
    """Create deterministic independent A/B panels with known subspace signal."""

    if isinstance(path_count, bool) or not isinstance(path_count, int) or path_count < 2:
        raise AbsoluteCoordinateError("synthetic path count is invalid")
    if not math.isfinite(float(noise_scale)) or float(noise_scale) < 0.0:
        raise AbsoluteCoordinateError("synthetic noise scale is invalid")
    active = lattice or build_coordinate_lattice()
    supplied = component_amplitudes or {
        "dc": (0.05, 0.04, 0.03, 0.02),
        "frequency1": (0.40, 0.30, 0.20, 0.10),
        "frequency2": (0.20, 0.15, 0.10, 0.05),
        "residual": (0.10, 0.08, 0.06, 0.04),
    }
    amplitudes: dict[str, tuple[float, ...]] = {}
    for component in COORDINATE_COMPONENTS:
        raw = tuple(float(value) for value in supplied.get(component, ()))
        if len(raw) != QUARTILE_COUNT or any(not math.isfinite(value) for value in raw):
            raise AbsoluteCoordinateError("synthetic component amplitudes are malformed")
        amplitudes[component] = raw
    directions: dict[str, np.ndarray] = {}
    for component in ("dc", "frequency1", "frequency2"):
        directions[component] = np.ascontiguousarray(
            active.basis(component)[:, :, 0] * math.sqrt(EDGES_PER_PHASE)
        )
    impulse = np.zeros((PHASE_COUNT, EDGES_PER_PHASE), dtype=np.float64)
    impulse[:, 0] = 1.0
    residual = project_coordinate_component(impulse, "residual", lattice=active)
    residual_norm = np.sqrt(np.mean(residual * residual, axis=1, keepdims=True))
    if np.any(residual_norm <= 0.0):
        raise AssertionError("synthetic residual direction vanished")
    directions["residual"] = residual / residual_norm
    signal = np.zeros(
        (QUARTILE_COUNT, PHASE_COUNT, EDGES_PER_PHASE), dtype=np.float64
    )
    for component in COORDINATE_COMPONENTS:
        for quartile in range(QUARTILE_COUNT):
            signal[quartile] += amplitudes[component][quartile] * directions[component]
    generator = np.random.Generator(np.random.Philox(int(seed)))
    left_values = signal[None] + float(noise_scale) * generator.standard_normal(
        (path_count,) + signal.shape
    )
    right_values = signal[None] + float(noise_scale) * generator.standard_normal(
        (path_count,) + signal.shape
    )
    left = CoordinatePanel(
        role="synthetic-panel-a",
        path_ids=np.arange(0x10000, 0x10000 + path_count, dtype=np.int64),
        cell_means=np.ascontiguousarray(left_values, dtype=np.float64),
    )
    right = CoordinatePanel(
        role="synthetic-panel-b",
        path_ids=np.arange(0x10100, 0x10100 + path_count, dtype=np.int64),
        cell_means=np.ascontiguousarray(right_values, dtype=np.float64),
    )
    frozen_signal = np.ascontiguousarray(signal)
    frozen_signal.setflags(write=False)
    return SyntheticCoordinateFixture(
        left=left,
        right=right,
        signal=frozen_signal,
        component_amplitudes=amplitudes,
    )


__all__ = [
    "ABSOLUTE_COORDINATE_SCHEMA",
    "ABSOLUTE_COORDINATE_VERSION",
    "ADirectionSeal",
    "AbsoluteCoordinateError",
    "BLinearEvidence",
    "COORDINATE_COMPONENTS",
    "CoordinateLattice",
    "CoordinatePanel",
    "CrossPanelDecomposition",
    "DEFAULT_BOOTSTRAP_CHUNK_SIZE",
    "DEFAULT_BOOTSTRAP_NAMESPACE",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE",
    "LinearMaxTResult",
    "SyntheticCoordinateFixture",
    "build_coordinate_lattice",
    "coordinate_family_names",
    "decompose_cross_panel_signal",
    "edge_translation_permutation",
    "evaluate_panel_b_linear",
    "model_translation_equivariance_record",
    "one_sided_b_linear_max_t",
    "phase_predictor_architecture_contract",
    "project_coordinate_component",
    "project_coordinate_components",
    "scaled_signed_cross_bounds",
    "seal_panel_a_directions",
    "synthetic_coordinate_fixture",
    "synthetic_model_inputs",
    "translate_model_inputs",
]
