"""Pure statistics for the exact-K=512 physical coarse-signal witness.

The witness partitions the permitted conditioning information into

    (time quartile, phase, oriented edge)

and estimates the squared energy of the corresponding conditional mean from
two independent panels.  Keeping the panels independent makes the cross-panel
product unbiased even though every panel-cell mean contains noisy exact
Rao--Blackwell labels.

This module contains no transition runner, trainer, checkpoint reader, or
sampler.  Orchestration is responsible for provenance, panel sealing, and
certified path generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_COUNT,
    SELECTED_OUTER_STEPS,
    semantic_sha256,
    stable_sum,
)


PHYSICAL_COARSE_SIGNAL_VERSION = "d0-jacobi-rb-physical-coarse-signal-v1"
ROOT_SEED = 261_241
BOOTSTRAP_SEED = 261_242
BOOTSTRAP_REPLICATES = 50_000
CONFIDENCE = 0.99
RESOLUTION_TARGET = 5.0e-4
PATHS_PER_PANEL = 64
OBSERVATIONS_PER_CELL = 8
TIME_QUARTILES = 4
COARSE_CELL_COUNT = TIME_QUARTILES * PHASE_COUNT * EDGES_PER_PHASE

PANEL_A_PATH_IDS = tuple(range(0xE5000, 0xE5040))
PANEL_B_PATH_IDS = tuple(range(0xE5100, 0xE5140))
PREFLIGHT_BENCHMARK_PATH_IDS = tuple(range(0xE5200, 0xE5208))

_BOOTSTRAP_LEFT_NAMESPACE = 0x50435741
_BOOTSTRAP_RIGHT_NAMESPACE = 0x50435742
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


class PhysicalCoarseSignalError(ValueError):
    """Raised when a coarse-signal statistic violates its frozen contract."""


def _finite_array(value: Any, *, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.size == 0:
        raise PhysicalCoarseSignalError(f"{name} must be nonempty")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise PhysicalCoarseSignalError(f"{name} contains nonfinite values")
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _validate_path_ids(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
    expected_count: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise PhysicalCoarseSignalError(f"{name} must be a nonempty vector")
    if raw.dtype.kind not in "iu":
        raise PhysicalCoarseSignalError(f"{name} must contain integers")
    result = np.asarray(raw, dtype=np.int64)
    if expected_count is not None and result.size != int(expected_count):
        raise PhysicalCoarseSignalError(
            f"{name} must contain exactly {expected_count} paths"
        )
    if ((result < 0) | (result >= (1 << 20))).any():
        raise PhysicalCoarseSignalError(f"{name} lies outside the 20-bit range")
    if np.unique(result).size != result.size:
        raise PhysicalCoarseSignalError(f"{name} contains duplicate paths")
    return np.ascontiguousarray(np.sort(result))


@dataclass(frozen=True)
class PhysicalCoarsePathPlan:
    """Frozen, disjoint path roles for the production witness."""

    panel_a: tuple[int, ...] = PANEL_A_PATH_IDS
    panel_b: tuple[int, ...] = PANEL_B_PATH_IDS
    preflight_benchmark: tuple[int, ...] = PREFLIGHT_BENCHMARK_PATH_IDS
    root_seed: int = ROOT_SEED
    version: str = PHYSICAL_COARSE_SIGNAL_VERSION + "-path-plan-v1"

    def __post_init__(self) -> None:
        roles = {
            "panel_a": _validate_path_ids(
                self.panel_a, name="panel_a", expected_count=PATHS_PER_PANEL
            ),
            "panel_b": _validate_path_ids(
                self.panel_b, name="panel_b", expected_count=PATHS_PER_PANEL
            ),
            "preflight_benchmark": _validate_path_ids(
                self.preflight_benchmark,
                name="preflight_benchmark",
                expected_count=8,
            ),
        }
        if int(self.root_seed) != ROOT_SEED or self.version != (
            PHYSICAL_COARSE_SIGNAL_VERSION + "-path-plan-v1"
        ):
            raise PhysicalCoarseSignalError("path-plan constants changed")
        names = tuple(roles)
        for index, left_name in enumerate(names):
            left = set(int(value) for value in roles[left_name])
            for right_name in names[index + 1 :]:
                if left.intersection(int(value) for value in roles[right_name]):
                    raise PhysicalCoarseSignalError("path-plan roles overlap")

    def to_record(self) -> dict[str, Any]:
        body = {
            "schema": self.version,
            "schema_version": 1,
            "version": self.version,
            "root_seed": int(self.root_seed),
            "roles": {
                "panel_a": list(self.panel_a),
                "panel_b": list(self.panel_b),
                "preflight_benchmark": list(self.preflight_benchmark),
            },
            "checks": {
                "integer_20_bit_pass": 1,
                "role_disjoint_pass": 1,
                "panel_sizes_frozen_pass": 1,
            },
            **NO_WORK,
        }
        return {**body, "path_plan_sha256": semantic_sha256(body)}

    @property
    def fingerprint(self) -> str:
        return str(self.to_record()["path_plan_sha256"])


@dataclass(frozen=True)
class PhysicalCoarseStatisticPlan:
    """Frozen definition of the authorizing cross-panel statistic."""

    selected_outer_steps: tuple[int, ...] = SELECTED_OUTER_STEPS
    time_quartiles: int = TIME_QUARTILES
    phase_count: int = PHASE_COUNT
    edges_per_phase: int = EDGES_PER_PHASE
    observations_per_cell: int = OBSERVATIONS_PER_CELL
    paths_per_panel: int = PATHS_PER_PANEL
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    bootstrap_seed: int = BOOTSTRAP_SEED
    confidence: float = CONFIDENCE
    resolution_target: float = RESOLUTION_TARGET
    version: str = PHYSICAL_COARSE_SIGNAL_VERSION + "-statistic-plan-v1"

    def __post_init__(self) -> None:
        if (
            tuple(self.selected_outer_steps) != SELECTED_OUTER_STEPS
            or int(self.time_quartiles) != TIME_QUARTILES
            or int(self.phase_count) != PHASE_COUNT
            or int(self.edges_per_phase) != EDGES_PER_PHASE
            or int(self.observations_per_cell) != OBSERVATIONS_PER_CELL
            or int(self.paths_per_panel) != PATHS_PER_PANEL
            or int(self.bootstrap_replicates) != BOOTSTRAP_REPLICATES
            or int(self.bootstrap_seed) != BOOTSTRAP_SEED
            or float(self.confidence) != CONFIDENCE
            or float(self.resolution_target) != RESOLUTION_TARGET
            or self.version
            != PHYSICAL_COARSE_SIGNAL_VERSION + "-statistic-plan-v1"
        ):
            raise PhysicalCoarseSignalError("statistic-plan constants changed")

    def to_record(self) -> dict[str, Any]:
        body = {
            **asdict(self),
            "selected_outer_steps": list(self.selected_outer_steps),
            "schema": self.version,
            "schema_version": 1,
            "coarse_cell_count": COARSE_CELL_COUNT,
            "estimand": (
                "mean_cell(square(E[exact_binary64_rb_label|"
                "time_quartile,phase,oriented_edge]))"
            ),
            "bootstrap_unit": "whole_path_independently_within_panel",
            "bootstrap_interval": "one_sided_99_percent_percentile_bounds",
            "secondary_interval": "one_sided_99_percent_welch_delta_bounds",
            "negative_estimates_or_intervals_truncated": 0,
            **NO_WORK,
        }
        return {**body, "statistic_plan_sha256": semantic_sha256(body)}

    @property
    def fingerprint(self) -> str:
        return str(self.to_record()["statistic_plan_sha256"])


def frozen_path_plan() -> PhysicalCoarsePathPlan:
    return PhysicalCoarsePathPlan()


def frozen_statistic_plan() -> PhysicalCoarseStatisticPlan:
    return PhysicalCoarseStatisticPlan()


@dataclass(frozen=True)
class PhysicalCoarsePanel:
    """Canonical path-level cell means for one independent panel."""

    role: str
    path_ids: np.ndarray = field(repr=False, compare=False)
    cell_means: np.ndarray = field(repr=False, compare=False)
    observations_per_cell: int = OBSERVATIONS_PER_CELL

    def __post_init__(self) -> None:
        role = str(self.role)
        if not role:
            raise PhysicalCoarseSignalError("panel role must be nonempty")
        raw_paths = np.asarray(self.path_ids)
        sorted_paths = _validate_path_ids(raw_paths, name=f"{role}.path_ids")
        if not np.array_equal(raw_paths.astype(np.int64, copy=False), sorted_paths):
            raise PhysicalCoarseSignalError("panel paths must be in canonical order")
        values = _finite_array(
            self.cell_means, name=f"{role}.cell_means", dtype=np.dtype(np.float64)
        )
        expected = (
            sorted_paths.size,
            TIME_QUARTILES,
            PHASE_COUNT,
            EDGES_PER_PHASE,
        )
        if values.shape != expected:
            raise PhysicalCoarseSignalError(
                f"panel cell means must have shape {expected}"
            )
        if int(self.observations_per_cell) != OBSERVATIONS_PER_CELL:
            raise PhysicalCoarseSignalError("observations-per-cell changed")
        paths = np.ascontiguousarray(sorted_paths)
        means = np.ascontiguousarray(values)
        paths.setflags(write=False)
        means.setflags(write=False)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "path_ids", paths)
        object.__setattr__(self, "cell_means", means)

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    @property
    def flattened(self) -> np.ndarray:
        return self.cell_means.reshape(self.path_count, COARSE_CELL_COUNT)

    @property
    def fingerprint(self) -> str:
        digest_body = {
            "schema": PHYSICAL_COARSE_SIGNAL_VERSION + "-panel-v1",
            "role": self.role,
            "path_ids": self.path_ids.tolist(),
            "shape": list(self.cell_means.shape),
            "dtype": self.cell_means.dtype.str,
            "observations_per_cell": self.observations_per_cell,
            "cell_means_sha256": _array_sha256(self.cell_means),
        }
        return semantic_sha256(digest_body)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_COARSE_SIGNAL_VERSION + "-panel-v1",
            "schema_version": 1,
            "role": self.role,
            "path_ids": self.path_ids.tolist(),
            "path_count": self.path_count,
            "shape": list(self.cell_means.shape),
            "coarse_cell_count": COARSE_CELL_COUNT,
            "observations_per_cell": self.observations_per_cell,
            "panel_sha256": self.fingerprint,
            **NO_WORK,
        }


def coarse_cell_path_means(
    denoising_target: Any,
    path_id: Any,
    outer_step: Any,
    phase: Any,
    *,
    role: str,
) -> PhysicalCoarsePanel:
    """Aggregate exact labels into the frozen path/quartile/phase/edge cells."""

    target = _finite_array(
        denoising_target, name="denoising_target", dtype=np.dtype(np.float64)
    )
    paths_raw = np.asarray(path_id)
    steps_raw = np.asarray(outer_step)
    phases_raw = np.asarray(phase)
    if (
        target.ndim != 2
        or target.shape[1] != EDGES_PER_PHASE
        or paths_raw.shape != (target.shape[0],)
        or steps_raw.shape != paths_raw.shape
        or phases_raw.shape != paths_raw.shape
        or paths_raw.dtype.kind not in "iu"
        or steps_raw.dtype.kind not in "iu"
        or phases_raw.dtype.kind not in "iu"
    ):
        raise PhysicalCoarseSignalError(
            "target metadata must have [N,392], [N], [N], [N] shapes"
        )
    paths = np.asarray(paths_raw, dtype=np.int64)
    steps = np.asarray(steps_raw, dtype=np.int64)
    phases = np.asarray(phases_raw, dtype=np.int64)
    unique_paths = _validate_path_ids(np.unique(paths), name=f"{role}.path_ids")
    selected = np.asarray(SELECTED_OUTER_STEPS, dtype=np.int64)
    if not np.isin(steps, selected).all():
        raise PhysicalCoarseSignalError("cache contains an unselected outer step")
    if ((phases < 0) | (phases >= PHASE_COUNT)).any():
        raise PhysicalCoarseSignalError("cache phase lies outside [0,7)")

    expected_rows = unique_paths.size * len(SELECTED_OUTER_STEPS) * PHASE_COUNT
    if target.shape[0] != expected_rows:
        raise PhysicalCoarseSignalError("cache row count is not the frozen design")
    result = np.empty(
        (unique_paths.size, TIME_QUARTILES, PHASE_COUNT, EDGES_PER_PHASE),
        dtype=np.float64,
    )
    for path_index, path_value in enumerate(unique_paths):
        path_mask = paths == path_value
        observed_keys = np.stack((steps[path_mask], phases[path_mask]), axis=1)
        if np.unique(observed_keys, axis=0).shape[0] != (
            len(SELECTED_OUTER_STEPS) * PHASE_COUNT
        ):
            raise PhysicalCoarseSignalError(
                "every path must contain each selected step/phase exactly once"
            )
        for quartile in range(TIME_QUARTILES):
            quartile_steps = selected[selected // 128 == quartile]
            if quartile_steps.size != OBSERVATIONS_PER_CELL:
                raise PhysicalCoarseSignalError("selected-step quartiles changed")
            for phase_index in range(PHASE_COUNT):
                mask = (
                    path_mask
                    & np.isin(steps, quartile_steps)
                    & (phases == phase_index)
                )
                if int(mask.sum()) != OBSERVATIONS_PER_CELL:
                    raise PhysicalCoarseSignalError(
                        "every path/cell must contain exactly eight observations"
                    )
                result[path_index, quartile, phase_index] = np.sum(
                    target[mask], axis=0, dtype=np.float64
                ) / float(OBSERVATIONS_PER_CELL)
    return PhysicalCoarsePanel(
        role=str(role),
        path_ids=unique_paths,
        cell_means=np.ascontiguousarray(result),
    )


def _validate_independent_panels(
    left: PhysicalCoarsePanel, right: PhysicalCoarsePanel
) -> None:
    if left.role == right.role:
        raise PhysicalCoarseSignalError("panel roles must differ")
    if set(left.path_ids.tolist()).intersection(right.path_ids.tolist()):
        raise PhysicalCoarseSignalError("panel path IDs must be disjoint")
    if left.path_count < 2 or right.path_count < 2:
        raise PhysicalCoarseSignalError("each panel requires at least two paths")


def cross_panel_path_kernel(
    left: PhysicalCoarsePanel, right: PhysicalCoarsePanel
) -> np.ndarray:
    """Return K_ij = mean_cell(A_i,c B_j,c)."""

    _validate_independent_panels(left, right)
    kernel = np.asarray(
        (left.flattened @ right.flattened.T) / float(COARSE_CELL_COUNT),
        dtype=np.float64,
    )
    if kernel.shape != (left.path_count, right.path_count):
        raise PhysicalCoarseSignalError("cross-panel kernel shape is invalid")
    if not np.isfinite(kernel).all():
        raise PhysicalCoarseSignalError("cross-panel kernel is nonfinite")
    return np.ascontiguousarray(kernel)


def cross_panel_point_estimate(
    left: PhysicalCoarsePanel, right: PhysicalCoarsePanel
) -> float:
    kernel = cross_panel_path_kernel(left, right)
    return stable_sum(kernel) / kernel.size


def _resample_count_matrix(
    generator: np.random.Generator, *, replicate_count: int, path_count: int
) -> np.ndarray:
    indices = generator.integers(
        0,
        int(path_count),
        size=(int(replicate_count), int(path_count)),
        dtype=np.int64,
    )
    offsets = (
        np.arange(int(replicate_count), dtype=np.int64)[:, None] * int(path_count)
    )
    counts = np.bincount(
        (indices + offsets).reshape(-1),
        minlength=int(replicate_count) * int(path_count),
    ).reshape(int(replicate_count), int(path_count))
    return np.asarray(counts, dtype=np.float64) / float(path_count)


def whole_path_cross_panel_bootstrap(
    left: PhysicalCoarsePanel,
    right: PhysicalCoarsePanel,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE,
    namespace: int = 0,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Independent whole-path percentile bootstrap for the cross-panel product."""

    _validate_independent_panels(left, right)
    if (
        int(replicates) <= 0
        or int(chunk_size) <= 0
        or not 0.0 < float(confidence) < 1.0
        or int(namespace) < 0
    ):
        raise PhysicalCoarseSignalError("bootstrap configuration is invalid")
    kernel = cross_panel_path_kernel(left, right)
    point = stable_sum(kernel) / kernel.size
    left_rng = np.random.Generator(
        np.random.Philox(
            [int(seed), int(namespace), _BOOTSTRAP_LEFT_NAMESPACE]
        )
    )
    right_rng = np.random.Generator(
        np.random.Philox(
            [int(seed), int(namespace), _BOOTSTRAP_RIGHT_NAMESPACE]
        )
    )
    estimates = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), int(chunk_size)):
        stop = min(int(replicates), start + int(chunk_size))
        count = stop - start
        left_weights = _resample_count_matrix(
            left_rng, replicate_count=count, path_count=left.path_count
        )
        right_weights = _resample_count_matrix(
            right_rng, replicate_count=count, path_count=right.path_count
        )
        estimates[start:stop] = np.einsum(
            "bi,ij,bj->b",
            left_weights,
            kernel,
            right_weights,
            optimize=True,
        )
    if not np.isfinite(estimates).all():
        raise PhysicalCoarseSignalError("bootstrap estimates are nonfinite")
    alpha = 1.0 - float(confidence)
    central_alpha = alpha / 2.0
    central_lower, lower, upper, central_upper = np.quantile(
        estimates,
        [central_alpha, alpha, float(confidence), 1.0 - central_alpha],
        method="linear",
    )
    return {
        "schema": PHYSICAL_COARSE_SIGNAL_VERSION + "-bootstrap-v1",
        "schema_version": 1,
        "method": "independent_whole_path_percentile_bootstrap",
        "left_role": left.role,
        "right_role": right.role,
        "left_path_count": left.path_count,
        "right_path_count": right.path_count,
        "coarse_cell_count": COARSE_CELL_COUNT,
        "point_estimate": float(point),
        "confidence": float(confidence),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "central_99_lower_bound": float(central_lower),
        "central_99_upper_bound": float(central_upper),
        "lower_quantile": float(alpha),
        "upper_quantile": float(confidence),
        "central_lower_quantile": float(central_alpha),
        "central_upper_quantile": float(1.0 - central_alpha),
        "replicates": int(replicates),
        "seed": int(seed),
        "namespace": int(namespace),
        "chunk_size": int(chunk_size),
        "bootstrap_unit": "whole_path_independently_within_panel",
        "negative_values_truncated": 0,
        **NO_WORK,
    }


def welch_delta_cross_panel_bounds(
    left: PhysicalCoarsePanel,
    right: PhysicalCoarsePanel,
    *,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    """Welch--Satterthwaite delta bound from the two path influences."""

    _validate_independent_panels(left, right)
    if not 0.0 < float(confidence) < 1.0:
        raise PhysicalCoarseSignalError("Welch confidence is invalid")
    kernel = cross_panel_path_kernel(left, right)
    point = stable_sum(kernel) / kernel.size
    left_projection = np.mean(kernel, axis=1, dtype=np.float64)
    right_projection = np.mean(kernel, axis=0, dtype=np.float64)
    left_influence = left_projection - point
    right_influence = right_projection - point
    left_component = float(np.var(left_projection, ddof=1) / left.path_count)
    right_component = float(np.var(right_projection, ddof=1) / right.path_count)
    variance = left_component + right_component
    if not math.isfinite(variance) or variance < 0.0:
        raise PhysicalCoarseSignalError("Welch variance is invalid")
    if variance == 0.0:
        standard_error = 0.0
        degrees_of_freedom = math.inf
        critical_value = 0.0
        central_critical_value = 0.0
        lower = point
        upper = point
        central_lower = point
        central_upper = point
    else:
        denominator = (
            left_component * left_component / (left.path_count - 1)
            + right_component * right_component / (right.path_count - 1)
        )
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise PhysicalCoarseSignalError("Welch degrees of freedom are invalid")
        degrees_of_freedom = variance * variance / denominator
        from scipy import stats

        critical_value = float(
            stats.t.ppf(float(confidence), degrees_of_freedom)
        )
        central_critical_value = float(
            stats.t.ppf(1.0 - (1.0 - float(confidence)) / 2.0, degrees_of_freedom)
        )
        standard_error = math.sqrt(variance)
        lower = point - critical_value * standard_error
        upper = point + critical_value * standard_error
        central_lower = point - central_critical_value * standard_error
        central_upper = point + central_critical_value * standard_error
    if not all(
        math.isfinite(value)
        for value in (
            point,
            standard_error,
            critical_value,
            central_critical_value,
            lower,
            upper,
            central_lower,
            central_upper,
        )
    ):
        raise PhysicalCoarseSignalError("Welch bounds are nonfinite")
    return {
        "schema": PHYSICAL_COARSE_SIGNAL_VERSION + "-welch-delta-v1",
        "schema_version": 1,
        "method": "two_sample_delta_welch_satterthwaite",
        "left_role": left.role,
        "right_role": right.role,
        "left_path_count": left.path_count,
        "right_path_count": right.path_count,
        "coarse_cell_count": COARSE_CELL_COUNT,
        "point_estimate": float(point),
        "confidence": float(confidence),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "central_99_lower_bound": float(central_lower),
        "central_99_upper_bound": float(central_upper),
        "standard_error": float(standard_error),
        "degrees_of_freedom": float(degrees_of_freedom),
        "critical_value": float(critical_value),
        "central_critical_value": float(central_critical_value),
        "left_variance_component": left_component,
        "right_variance_component": right_component,
        "left_path_ids": left.path_ids.tolist(),
        "right_path_ids": right.path_ids.tolist(),
        "left_influence": left_influence.tolist(),
        "right_influence": right_influence.tolist(),
        "negative_values_truncated": 0,
        **NO_WORK,
    }


def _valid_method_record(record: Mapping[str, Any]) -> bool:
    try:
        point = float(record["point_estimate"])
        lower = float(record["lower_bound"])
        upper = float(record["upper_bound"])
        confidence = float(record["confidence"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        all(math.isfinite(value) for value in (point, lower, upper, confidence))
        and 0.0 < confidence < 1.0
        and lower <= upper
    )


def classify_cross_panel_signal(
    bootstrap: Mapping[str, Any],
    welch: Mapping[str, Any],
    *,
    resolution_target: float = RESOLUTION_TARGET,
) -> dict[str, Any]:
    """Apply the frozen detection/resolution trichotomy."""

    if (
        not _valid_method_record(bootstrap)
        or not _valid_method_record(welch)
        or not math.isfinite(float(resolution_target))
        or float(resolution_target) <= 0.0
    ):
        decision = "physical_coarse_signal_estimator_invalid"
    else:
        point_error = abs(
            float(bootstrap["point_estimate"]) - float(welch["point_estimate"])
        )
        tolerance = 8.0 * math.ulp(
            max(
                1.0,
                abs(float(bootstrap["point_estimate"])),
                abs(float(welch["point_estimate"])),
            )
        )
        if point_error > tolerance:
            decision = "physical_coarse_signal_estimator_invalid"
        else:
            detected = (
                float(bootstrap["lower_bound"]) > 0.0,
                float(welch["lower_bound"]) > 0.0,
            )
            resolved_below = (
                float(bootstrap["upper_bound"]) <= float(resolution_target),
                float(welch["upper_bound"]) <= float(resolution_target),
            )
            if all(detected):
                decision = "exact_physical_coarse_signal_detected"
            elif not any(detected) and all(resolved_below):
                decision = "coarse_signal_below_preregistered_resolution"
            else:
                decision = "physical_coarse_signal_inconclusive"
    return {
        "schema": PHYSICAL_COARSE_SIGNAL_VERSION + "-classification-v1",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "decision": decision,
        "resolution_target": float(resolution_target),
        "bootstrap_lower_bound": bootstrap.get("lower_bound"),
        "bootstrap_upper_bound": bootstrap.get("upper_bound"),
        "welch_lower_bound": welch.get("lower_bound"),
        "welch_upper_bound": welch.get("upper_bound"),
        "negative_estimates_or_intervals_truncated": 0,
        "physical_training_authorized": 0,
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
        "reverse_sampling_authorized": 0,
        "reverse_sampling_performed": 0,
    }


def analyze_cross_panel_signal(
    left: PhysicalCoarsePanel,
    right: PhysicalCoarsePanel,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE,
    namespace: int = 0,
    chunk_size: int = 512,
    resolution_target: float = RESOLUTION_TARGET,
) -> dict[str, Any]:
    bootstrap = whole_path_cross_panel_bootstrap(
        left,
        right,
        seed=seed,
        replicates=replicates,
        confidence=confidence,
        namespace=namespace,
        chunk_size=chunk_size,
    )
    welch = welch_delta_cross_panel_bounds(left, right, confidence=confidence)
    classification = classify_cross_panel_signal(
        bootstrap, welch, resolution_target=resolution_target
    )
    return {
        "schema": PHYSICAL_COARSE_SIGNAL_VERSION + "-analysis-v1",
        "schema_version": 1,
        "estimand": (
            "mean_cell(square(E[exact_binary64_rb_label|"
            "time_quartile,phase,oriented_edge]))"
        ),
        "left_panel": left.to_record(),
        "right_panel": right.to_record(),
        "bootstrap": bootstrap,
        "welch_delta": welch,
        "classification": classification,
        "claim_scope": (
            "frozen selected-step design of the exact-K=512 split chain "
            "for one lambda-mixed image"
        ),
        "lower_bound_on_full_allowed_input_conditional_mean_energy": 1,
        "conditional_mean_identically_zero_proven": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


def bayes_control_split_pairs(
    panels: Mapping[str, PhysicalCoarsePanel],
) -> tuple[tuple[str, str, PhysicalCoarsePanel, PhysicalCoarsePanel], ...]:
    """Return the exact three unordered train/validation/confirmation pairs."""

    required = ("train", "validation", "confirmation")
    if set(panels) != set(required):
        raise PhysicalCoarseSignalError(
            "Bayes controls require exactly train, validation, confirmation"
        )
    return (
        ("train", "validation", panels["train"], panels["validation"]),
        ("train", "confirmation", panels["train"], panels["confirmation"]),
        (
            "validation",
            "confirmation",
            panels["validation"],
            panels["confirmation"],
        ),
    )


def evaluate_bayes_control_replay(
    *,
    teacher_panels: Mapping[str, PhysicalCoarsePanel],
    null_panels: Mapping[str, PhysicalCoarsePanel],
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = CONFIDENCE,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Exercise both inference methods on immutable noisy Bayes label caches."""

    rows: list[dict[str, Any]] = []
    for law_index, (law, panels) in enumerate(
        (("teacher", teacher_panels), ("null", null_panels))
    ):
        for pair_index, (left_name, right_name, left, right) in enumerate(
            bayes_control_split_pairs(panels)
        ):
            namespace = 100 + 10 * law_index + pair_index
            analysis = analyze_cross_panel_signal(
                left,
                right,
                seed=seed,
                replicates=replicates,
                confidence=confidence,
                namespace=namespace,
                chunk_size=chunk_size,
            )
            bootstrap = analysis["bootstrap"]
            welch = analysis["welch_delta"]
            if law == "teacher":
                passed = (
                    float(bootstrap["lower_bound"]) > 0.0
                    and float(welch["lower_bound"]) > 0.0
                )
            else:
                passed = (
                    float(bootstrap["central_99_lower_bound"])
                    <= 0.0
                    <= float(bootstrap["central_99_upper_bound"])
                    and float(welch["central_99_lower_bound"])
                    <= 0.0
                    <= float(welch["central_99_upper_bound"])
                )
            rows.append(
                {
                    "law": law,
                    "left_split": left_name,
                    "right_split": right_name,
                    "namespace": namespace,
                    "point_estimate": bootstrap["point_estimate"],
                    "bootstrap_lower_bound": bootstrap["lower_bound"],
                    "bootstrap_upper_bound": bootstrap["upper_bound"],
                    "welch_lower_bound": welch["lower_bound"],
                    "welch_upper_bound": welch["upper_bound"],
                    "bootstrap_central_99_lower_bound": bootstrap[
                        "central_99_lower_bound"
                    ],
                    "bootstrap_central_99_upper_bound": bootstrap[
                        "central_99_upper_bound"
                    ],
                    "welch_central_99_lower_bound": welch[
                        "central_99_lower_bound"
                    ],
                    "welch_central_99_upper_bound": welch[
                        "central_99_upper_bound"
                    ],
                    "passed": int(passed),
                }
            )
    passed = len(rows) == 6 and all(int(row["passed"]) == 1 for row in rows)
    return {
        "schema": PHYSICAL_COARSE_SIGNAL_VERSION + "-bayes-control-replay-v1",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": int(passed),
        "pair_count": len(rows),
        "teacher_pair_count": sum(row["law"] == "teacher" for row in rows),
        "null_pair_count": sum(row["law"] == "null" for row in rows),
        "rows": rows,
        "source": "immutable_noisy_label_caches_not_oracle_means",
        **NO_WORK,
        "bootstrap_unit": "whole_path_independently_within_split",
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "COARSE_CELL_COUNT",
    "CONFIDENCE",
    "OBSERVATIONS_PER_CELL",
    "PANEL_A_PATH_IDS",
    "PANEL_B_PATH_IDS",
    "PATHS_PER_PANEL",
    "PHYSICAL_COARSE_SIGNAL_VERSION",
    "PREFLIGHT_BENCHMARK_PATH_IDS",
    "PhysicalCoarsePanel",
    "PhysicalCoarsePathPlan",
    "PhysicalCoarseSignalError",
    "PhysicalCoarseStatisticPlan",
    "RESOLUTION_TARGET",
    "ROOT_SEED",
    "analyze_cross_panel_signal",
    "bayes_control_split_pairs",
    "classify_cross_panel_signal",
    "coarse_cell_path_means",
    "cross_panel_path_kernel",
    "cross_panel_point_estimate",
    "evaluate_bayes_control_replay",
    "frozen_path_plan",
    "frozen_statistic_plan",
    "welch_delta_cross_panel_bounds",
    "whole_path_cross_panel_bootstrap",
]
