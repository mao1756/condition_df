"""Fail-closed statistics and gates for exact Jacobi Strang refinement.

This module is deliberately independent of trainers and reverse samplers.  It
contains only small, deterministic statistical helpers and the closed
decision ladder consumed by the controls-only refinement CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "experiment12-d0-jacobi-rb-strang-refinement-gate"
SCHEMA_VERSION = 1
BOOTSTRAP_VERSION = "paired-whole-path-refinement-v1"
MAX_T_VERSION = "two-sided-whole-path-max-t-v1"
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


@dataclass(frozen=True)
class StrangRefinementThresholds:
    """Frozen production thresholds for the fixed-grid refinement claim."""

    parent_record_count: int = 891
    grid_size: int = 28
    cell_count: int = 784
    alpha: float = 1.0
    tau_eff: float = 5.0e-5
    levels: tuple[int, ...] = (128, 256, 512, 1024)
    reference_level: int = 2048
    observation_time_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    restart_steps_per_shard: int = 8
    production_group_size: int = 8
    preflight_panel_count: int = 2
    preflight_paths_per_panel: int = 128
    preflight_transitions_per_path: int = 32
    pilot_main_paths: int = 16
    pilot_reference_paths: int = 8
    candidate_main_paths: tuple[int, ...] = (32, 64)
    candidate_reference_paths: tuple[int, ...] = (16, 32)
    bootstrap_replicates: int = 20_000
    bootstrap_confidence: float = 0.99
    weak_order_confidence: float = 0.90
    minimum_observed_weak_order: float = 1.8
    minimum_weak_order_interval_lower: float = 1.5
    expected_weak_order: float = 2.0
    maximum_main_half_width: float = 0.0025
    maximum_reference_half_width: float = 0.005
    maximum_512_1024_discrepancy: float = 0.005
    maximum_512_reference_error: float = 0.01
    maximum_reference_instability: float = 0.005
    maximum_projected_hours: float = 48.0
    minimum_rate: float = 1_300.0
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_cost_fraction: float = 0.10
    maximum_peak_memory_fraction: float = 0.80
    maximum_local_generator_error: float = 1.0e-8
    maximum_cuda_mass_error: float = 2.0e-6
    required_image_sha256: str = (
        "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
    )

    def __post_init__(self) -> None:
        frozen: dict[str, Any] = {
            "parent_record_count": 891,
            "grid_size": 28,
            "cell_count": 784,
            "alpha": 1.0,
            "tau_eff": 5.0e-5,
            "levels": (128, 256, 512, 1024),
            "reference_level": 2048,
            "observation_time_fractions": (0.25, 0.5, 0.75, 1.0),
            "restart_steps_per_shard": 8,
            "production_group_size": 8,
            "preflight_panel_count": 2,
            "preflight_paths_per_panel": 128,
            "preflight_transitions_per_path": 32,
            "pilot_main_paths": 16,
            "pilot_reference_paths": 8,
            "candidate_main_paths": (32, 64),
            "candidate_reference_paths": (16, 32),
            "bootstrap_replicates": 20_000,
            "bootstrap_confidence": 0.99,
            "weak_order_confidence": 0.90,
            "minimum_observed_weak_order": 1.8,
            "minimum_weak_order_interval_lower": 1.5,
            "expected_weak_order": 2.0,
            "maximum_main_half_width": 0.0025,
            "maximum_reference_half_width": 0.005,
            "maximum_512_1024_discrepancy": 0.005,
            "maximum_512_reference_error": 0.01,
            "maximum_reference_instability": 0.005,
            "maximum_projected_hours": 48.0,
            "minimum_rate": 1_300.0,
            "maximum_fallback_fraction": 1.0e-4,
            "maximum_fallback_cost_fraction": 0.10,
            "maximum_peak_memory_fraction": 0.80,
            "maximum_local_generator_error": 1.0e-8,
            "maximum_cuda_mass_error": 2.0e-6,
            "required_image_sha256": (
                "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
            ),
        }
        for name, value in frozen.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is frozen at {value}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "levels",
            "observation_time_fractions",
            "candidate_main_paths",
            "candidate_reference_paths",
        ):
            result[name] = list(result[name])
        return result


class StrangRefinementDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    REFINEMENT_SCHEDULER_INVALID = "refinement_scheduler_invalid"
    JACOBI_STATIONARITY_CONTROL_INVALID = "jacobi_stationarity_control_invalid"
    REFINEMENT_KERNEL_NUMERICALLY_UNRESOLVED = (
        "refinement_kernel_numerically_unresolved"
    )
    REFINEMENT_POWER_INFEASIBLE = "refinement_power_infeasible"
    REFINEMENT_COMPUTATIONALLY_INFEASIBLE = (
        "refinement_computationally_infeasible"
    )
    REFINEMENT_REFERENCE_INCONCLUSIVE = "refinement_reference_inconclusive"
    STRANG_REFINEMENT_INCONCLUSIVE = "strang_refinement_inconclusive"
    STRANG_SPLIT_REFERENCE_INVALID = "strang_split_reference_invalid"
    EXACT_STATE_DEPENDENT_STRANG_REFINEMENT_FEASIBLE = (
        "exact_state_dependent_strang_refinement_feasible"
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int, np.bool_, np.integer)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int, np.bool_, np.integer)) and int(value) == 0


def _status(gate: Mapping[str, Any] | None) -> str:
    if not isinstance(gate, Mapping):
        return "not_evaluated"
    return str(gate.get("evaluation_status", "not_evaluated"))


def _passed(value: bool | int | Mapping[str, Any] | None) -> bool:
    if isinstance(value, Mapping):
        return _status(value) == "evaluated" and _one(value.get("passed"))
    return value is True or _one(value)


def _check(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _eq_one(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", 1, _one(value))


def _eq_zero(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", 0, _zero(value))


def _equal(metrics: Mapping[str, Any], name: str, expected: Any) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", expected, value == expected)


def _sequence_equal(
    metrics: Mapping[str, Any], name: str, expected: Sequence[Any]
) -> dict[str, Any]:
    value = metrics.get(name)
    valid = isinstance(value, (list, tuple)) and tuple(value) == tuple(expected)
    return _check(value, "==", list(expected), valid)


def _le(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value,
        "<=",
        threshold,
        _finite(value) and 0.0 <= float(value) <= float(threshold),
    )


def _ge(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value,
        ">=",
        threshold,
        _finite(value) and float(value) >= float(threshold),
    )


def _gate(
    name: str,
    claim_scope: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_status: str = "evaluated",
) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in checks.items()}
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": str(name),
        "claim_scope": str(claim_scope),
        "evaluation_status": str(evaluation_status),
        "subchecks": normalized,
        "passed": int(
            evaluation_status == "evaluated"
            and bool(normalized)
            and all(_one(value.get("passed")) for value in normalized.values())
        ),
        **NO_WORK,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    result = _gate(name, "not evaluated", {}, evaluation_status="not_evaluated")
    result["reason"] = str(reason)
    return result


def dirichlet_linear_moments(
    coefficients: Sequence[float] | np.ndarray, *, alpha: float = 1.0
) -> dict[str, float]:
    """Exact mean and variance of ``sum_i coefficients[i] * S_i``."""

    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("coefficients must be a finite one-dimensional vector")
    if not _finite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be finite and positive")
    a = float(alpha)
    total = a * values.size
    coefficient_sum = float(values.sum(dtype=np.float64))
    mean = coefficient_sum / values.size
    numerator = (
        a * float(np.dot(values, values))
        - (a / values.size) * coefficient_sum * coefficient_sum
    )
    variance = numerator / (total * (total + 1.0))
    variance = max(0.0, float(variance))
    return {"mean": float(mean), "variance": variance, "rms": math.sqrt(variance)}


def _rising(value: float, power: int) -> float:
    result = 1.0
    for offset in range(int(power)):
        result *= value + offset
    return result


def _dirichlet_power_sum_moments(
    cell_count: int, alpha: float, power: int
) -> dict[str, float]:
    count = int(cell_count)
    a = float(alpha)
    total = count * a
    mean = count * _rising(a, power) / _rising(total, power)
    second = (
        count * _rising(a, 2 * power)
        + count * (count - 1) * _rising(a, power) ** 2
    ) / _rising(total, 2 * power)
    variance = max(0.0, second - mean * mean)
    return {
        "mean": float(mean),
        "second_moment": float(second),
        "variance": float(variance),
        "rms": math.sqrt(variance),
    }


def dirichlet_observable_moments(
    *, cell_count: int = 784, alpha: float = 1.0
) -> dict[str, Any]:
    """Return exact symmetric-Dirichlet normalization for gated observables."""

    if int(cell_count) < 2:
        raise ValueError("cell_count must be at least two")
    if not _finite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be finite and positive")
    return {
        "schema": SCHEMA + "-dirichlet-observable-moments",
        "schema_version": SCHEMA_VERSION,
        "cell_count": int(cell_count),
        "alpha": float(alpha),
        "quadratic_power_sum": _dirichlet_power_sum_moments(
            int(cell_count), float(alpha), 2
        ),
        "cubic_power_sum": _dirichlet_power_sum_moments(
            int(cell_count), float(alpha), 3
        ),
    }


def center_scale_observable(
    values: Sequence[float] | np.ndarray, *, mean: float, rms: float
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("observable values must be finite")
    if not _finite(mean) or not _finite(rms) or float(rms) <= 0.0:
        raise ValueError("normalization mean/rms must be finite with positive rms")
    return (array - float(mean)) / float(rms)


def fit_weak_order(
    level_means: Mapping[int, float | Sequence[float] | np.ndarray],
    *,
    levels: Sequence[int] = (128, 256, 512),
) -> np.ndarray:
    """Fit ``log |mu_K-mu_2K|`` against ``log K`` feature by feature."""

    ordered = tuple(int(value) for value in levels)
    if len(ordered) < 2 or any(
        ordered[index + 1] != 2 * ordered[index]
        for index in range(len(ordered) - 1)
    ):
        raise ValueError("levels must be an increasing dyadic sequence")
    required = ordered + (2 * ordered[-1],)
    arrays = {
        level: np.asarray(level_means[level], dtype=np.float64)
        for level in required
    }
    shape = arrays[required[0]].shape
    if any(value.shape != shape or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("level means must be finite and have identical shapes")
    differences = np.stack(
        [np.abs(arrays[level] - arrays[2 * level]) for level in ordered], axis=0
    )
    if np.any(differences <= 0.0):
        raise ValueError("weak order is undefined for a zero successive difference")
    x = np.log(np.asarray(ordered, dtype=np.float64))
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    flattened = np.log(differences.reshape(len(ordered), -1))
    slopes = centered_x @ (flattened - flattened.mean(axis=0)) / denominator
    return (-slopes).reshape(shape)


def richardson_reference(
    mu_512: float | Sequence[float] | np.ndarray,
    mu_1024: float | Sequence[float] | np.ndarray,
    mu_2048: float | Sequence[float] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Second-order Richardson reference and its lower-resolution analogue."""

    v512 = np.asarray(mu_512, dtype=np.float64)
    v1024 = np.asarray(mu_1024, dtype=np.float64)
    v2048 = np.asarray(mu_2048, dtype=np.float64)
    if (
        v512.shape != v1024.shape
        or v512.shape != v2048.shape
        or not all(np.isfinite(value).all() for value in (v512, v1024, v2048))
    ):
        raise ValueError("Richardson inputs must be finite and shape-compatible")
    high = (4.0 * v2048 - v1024) / 3.0
    low = (4.0 * v1024 - v512) / 3.0
    return {
        "reference": high,
        "lower_reference": low,
        "stability_error": np.abs(high - low),
        "level_512_error": np.abs(v512 - high),
    }


def _quantile_higher(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, probability, interpolation="higher"))


def _quantile_lower(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="lower"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, probability, interpolation="lower"))


def whole_path_max_t_intervals(
    path_values: Mapping[str, Sequence[float] | np.ndarray],
    *,
    expected: Mapping[str, float] | None = None,
    seed: int,
    confidence: float = 0.99,
    reps: int = 20_000,
    standard_error_floor: float = 1.0e-15,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Deterministic two-sided simultaneous intervals over aligned path IDs."""

    if not path_values:
        raise ValueError("path_values must not be empty")
    if not 0.0 < float(confidence) < 1.0 or int(reps) <= 0:
        raise ValueError("invalid bootstrap confidence or replicate count")
    names = sorted(str(name) for name in path_values)
    columns = [np.asarray(path_values[name], dtype=np.float64) for name in names]
    if any(value.ndim != 1 for value in columns):
        raise ValueError("each max-T member must be a one-dimensional path vector")
    path_count = columns[0].size
    if path_count < 2 or any(value.size != path_count for value in columns):
        raise ValueError("max-T members must contain the same two or more paths")
    if any(not np.isfinite(value).all() for value in columns):
        raise ValueError("max-T path values must be finite")
    expected_map = {name: 0.0 for name in names}
    if expected is not None:
        expected_map.update({str(name): float(value) for name, value in expected.items()})
    matrix = np.column_stack(
        [value - expected_map[name] for name, value in zip(names, columns)]
    )
    means = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    se = sd / math.sqrt(path_count)
    exact_zero = np.count_nonzero(matrix, axis=0) == 0
    if np.any((se <= standard_error_floor) & ~exact_zero):
        raise ValueError("nonzero degenerate whole-path statistic")
    centered = matrix - means[None, :]
    rng = np.random.Generator(np.random.Philox(int(seed)))
    maxima = np.empty(int(reps), dtype=np.float64)
    for start in range(0, int(reps), int(chunk_size)):
        stop = min(int(reps), start + int(chunk_size))
        indices = rng.integers(
            0, path_count, size=(stop - start, path_count), endpoint=False
        )
        samples = centered[indices]
        boot_mean = samples.mean(axis=1)
        boot_sd = samples.std(axis=1, ddof=1)
        denominator = boot_sd / math.sqrt(path_count)
        t_values = np.divide(
            boot_mean,
            denominator,
            out=np.zeros_like(boot_mean),
            where=denominator > standard_error_floor,
        )
        maxima[start:stop] = np.max(np.abs(t_values), axis=1)
    if not np.isfinite(maxima).all():
        raise ValueError("max-T bootstrap produced nonfinite statistics")
    critical = _quantile_higher(maxima, float(confidence))
    members = []
    for index, name in enumerate(names):
        lower = float(means[index] - critical * se[index])
        upper = float(means[index] + critical * se[index])
        members.append(
            {
                "name": name,
                "expected": float(expected_map[name]),
                "path_count": int(path_count),
                "mean_minus_expected": float(means[index]),
                "standard_error": float(se[index]),
                "simultaneous_lower": lower,
                "simultaneous_upper": upper,
                "contains_zero": int(lower <= 0.0 <= upper),
            }
        )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                name: np.asarray(path_values[name], dtype=np.float64).tolist()
                for name in names
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": SCHEMA + "-whole-path-max-t",
        "schema_version": SCHEMA_VERSION,
        "version": MAX_T_VERSION,
        "evaluation_status": "evaluated",
        "valid": 1,
        "passed": int(all(_one(value["contains_zero"]) for value in members)),
        "seed": int(seed),
        "confidence": float(confidence),
        "bootstrap_replicates": int(reps),
        "path_count": int(path_count),
        "family_size": len(members),
        "critical_value": critical,
        "family_fingerprint": fingerprint,
        "members": members,
        **NO_WORK,
    }


def _as_path_feature_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite path-by-feature array")
    return array.reshape(array.shape[0], -1)


def whole_path_refinement_bootstrap(
    level_values: Mapping[int, Sequence[float] | np.ndarray],
    *,
    seed: int,
    reps: int = 20_000,
    confidence: float = 0.99,
    order_confidence: float = 0.90,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Joint whole-path bootstrap for order, discrepancies, and Richardson error.

    Levels 128--1024 must contain the same aligned main paths.  Level 2048 may
    contain a prefix subset; that same subset is used from levels 512 and 1024
    for Richardson calculations.
    """

    required = (128, 256, 512, 1024, 2048)
    if any(level not in level_values for level in required):
        raise ValueError(f"level_values must contain {required}")
    if int(reps) <= 0 or not 0.0 < float(confidence) < 1.0:
        raise ValueError("invalid bootstrap configuration")
    arrays = {
        level: _as_path_feature_array(level_values[level], f"level {level}")
        for level in required
    }
    main_paths, feature_count = arrays[128].shape
    if any(arrays[level].shape != (main_paths, feature_count) for level in required[:-1]):
        raise ValueError("levels 128--1024 must share aligned paths and features")
    reference_paths = arrays[2048].shape[0]
    if arrays[2048].shape[1] != feature_count or reference_paths > main_paths:
        raise ValueError("level 2048 must be an aligned prefix subset")

    means = {level: values.mean(axis=0) for level, values in arrays.items()}
    point_order = fit_weak_order(means, levels=(128, 256, 512))
    point_main = np.abs(means[512] - means[1024])
    point_richardson = richardson_reference(
        arrays[512][:reference_paths].mean(axis=0),
        arrays[1024][:reference_paths].mean(axis=0),
        means[2048],
    )

    # Separate statelessly-derived streams make the result invariant to the
    # implementation chunk size.  The reference stream is shared by every
    # level on the K=2048 subset; the second stream is used only for paths
    # that are absent from that subset.
    rng_reference = np.random.Generator(np.random.Philox(int(seed)))
    rng_remaining = np.random.Generator(
        np.random.Philox(int(seed) ^ 0x9E3779B97F4A7C15)
    )
    boot_order = np.empty((int(reps), feature_count), dtype=np.float64)
    boot_main = np.empty_like(boot_order)
    boot_reference = np.empty_like(boot_order)
    boot_stability = np.empty_like(boot_order)
    cursor = 0
    valid = True
    while cursor < int(reps):
        count = min(int(chunk_size), int(reps) - cursor)
        reference_indices = rng_reference.integers(
            0, reference_paths, size=(count, reference_paths), endpoint=False
        )
        # The K=2048 paths are the aligned prefix of the main panel.  Resample
        # that prefix once and reuse the same path IDs at every level.  Sample
        # the remaining main-only stratum independently, then concatenate the
        # two strata.  This keeps the reference sample size fixed while
        # preserving all cross-level covariance for every shared path.
        remaining_paths = main_paths - reference_paths
        if remaining_paths:
            remaining_indices = reference_paths + rng_remaining.integers(
                0,
                remaining_paths,
                size=(count, remaining_paths),
                endpoint=False,
            )
            main_indices = np.concatenate(
                (reference_indices, remaining_indices), axis=1
            )
        else:
            main_indices = reference_indices
        boot_means = {
            level: arrays[level][main_indices].mean(axis=1)
            for level in required[:-1]
        }
        deltas = np.stack(
            [
                np.abs(boot_means[level] - boot_means[2 * level])
                for level in (128, 256, 512)
            ],
            axis=1,
        )
        if np.any(deltas <= 0.0):
            valid = False
            break
        x = np.log(np.asarray((128, 256, 512), dtype=np.float64))
        centered_x = x - x.mean()
        log_delta = np.log(deltas)
        slope = np.einsum(
            "l,rlf->rf",
            centered_x,
            log_delta - log_delta.mean(axis=1, keepdims=True),
        ) / float(np.dot(centered_x, centered_x))
        boot_order[cursor : cursor + count] = -slope
        boot_main[cursor : cursor + count] = np.abs(
            boot_means[512] - boot_means[1024]
        )

        ref512 = arrays[512][:reference_paths][reference_indices].mean(axis=1)
        ref1024 = arrays[1024][:reference_paths][reference_indices].mean(axis=1)
        ref2048 = arrays[2048][reference_indices].mean(axis=1)
        high = (4.0 * ref2048 - ref1024) / 3.0
        low = (4.0 * ref1024 - ref512) / 3.0
        boot_reference[cursor : cursor + count] = np.abs(ref512 - high)
        boot_stability[cursor : cursor + count] = np.abs(high - low)
        cursor += count

    if not valid or not all(
        np.isfinite(value).all()
        for value in (boot_order, boot_main, boot_reference, boot_stability)
    ):
        return {
            "schema": SCHEMA + "-refinement-bootstrap",
            "schema_version": SCHEMA_VERSION,
            "version": BOOTSTRAP_VERSION,
            "evaluation_status": "evaluated",
            "valid": 0,
            "passed": 0,
            "reason": "zero or nonfinite bootstrapped successive difference",
            "seed": int(seed),
            "bootstrap_replicates": int(reps),
            **NO_WORK,
        }

    order_tail = (1.0 - float(order_confidence)) / 2.0
    feature_rows = []
    for feature in range(feature_count):
        order_lower = _quantile_lower(boot_order[:, feature], order_tail)
        order_upper = _quantile_higher(
            boot_order[:, feature], 1.0 - order_tail
        )
        feature_rows.append(
            {
                "feature": int(feature),
                "observed_weak_order": float(point_order[feature]),
                "weak_order_interval_lower": order_lower,
                "weak_order_interval_upper": order_upper,
                "weak_order_interval_contains_two": int(
                    order_lower <= 2.0 <= order_upper
                ),
                "observed_512_1024_discrepancy": float(point_main[feature]),
                "observed_512_reference_error": float(
                    point_richardson["level_512_error"][feature]
                ),
                "observed_reference_instability": float(
                    point_richardson["stability_error"][feature]
                ),
            }
        )
    simultaneous_main = _quantile_higher(
        np.max(boot_main, axis=1), float(confidence)
    )
    simultaneous_reference = _quantile_higher(
        np.max(boot_reference, axis=1), float(confidence)
    )
    simultaneous_stability = _quantile_higher(
        np.max(boot_stability, axis=1), float(confidence)
    )
    return {
        "schema": SCHEMA + "-refinement-bootstrap",
        "schema_version": SCHEMA_VERSION,
        "version": BOOTSTRAP_VERSION,
        "evaluation_status": "evaluated",
        "valid": 1,
        "seed": int(seed),
        "bootstrap_replicates": int(reps),
        "confidence": float(confidence),
        "order_confidence": float(order_confidence),
        "main_path_count": int(main_paths),
        "reference_path_count": int(reference_paths),
        "feature_count": int(feature_count),
        "feature_metrics": feature_rows,
        "simultaneous_512_1024_upper_bound": simultaneous_main,
        "simultaneous_512_reference_upper_bound": simultaneous_reference,
        "simultaneous_reference_stability_upper_bound": simultaneous_stability,
        **NO_WORK,
    }


def select_refinement_design(
    pilot_records: Sequence[Mapping[str, Any]],
    thresholds: StrangRefinementThresholds | None = None,
) -> dict[str, Any]:
    """Select the cheapest predeclared design using variance/timing only."""

    t = thresholds or StrangRefinementThresholds()
    expected = {
        (main, reference)
        for main in t.candidate_main_paths
        for reference in t.candidate_reference_paths
        if reference <= main
    }
    records: list[dict[str, Any]] = []
    observed: set[tuple[int, int]] = set()
    for raw in pilot_records:
        row = dict(raw)
        main = int(row.get("main_paths", -1))
        reference = int(row.get("reference_paths", -1))
        key = (main, reference)
        if key not in expected or key in observed:
            raise ValueError("pilot design rows must enumerate each frozen candidate once")
        observed.add(key)
        main_width = row.get("predicted_main_half_width")
        reference_width = row.get("predicted_reference_half_width")
        hours = row.get("projected_hours")
        finite = all(_finite(value) and float(value) >= 0.0 for value in (
            main_width, reference_width, hours
        ))
        eligible = (
            finite
            and float(main_width) <= t.maximum_main_half_width
            and float(reference_width) <= t.maximum_reference_half_width
            and float(hours) <= t.maximum_projected_hours
            and _one(row.get("variance_only_pass"))
            and _one(row.get("pilot_production_isolation_pass"))
            and _one(row.get("pilot_means_excluded_pass"))
        )
        row["eligible"] = int(eligible)
        records.append(row)
    if observed != expected:
        raise ValueError("pilot records do not contain the complete frozen design grid")
    eligible_rows = [row for row in records if _one(row["eligible"])]
    selected = min(
        eligible_rows,
        key=lambda row: (
            float(row["projected_hours"]),
            int(row["main_paths"]) + int(row["reference_paths"]),
            int(row["main_paths"]),
            int(row["reference_paths"]),
        ),
        default=None,
    )
    return {
        "schema": SCHEMA + "-selected-design",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "selection_status": "selected" if selected is not None else "no_eligible_design",
        "passed": int(selected is not None),
        "candidate_count": len(records),
        "eligible_candidate_count": len(eligible_rows),
        "candidates": sorted(
            records, key=lambda row: (int(row["main_paths"]), int(row["reference_paths"]))
        ),
        "selected": None if selected is None else dict(selected),
        "ranking": [
            "projected_hours",
            "total_paths",
            "main_paths",
            "reference_paths",
        ],
        "thresholds": {
            "maximum_main_half_width": t.maximum_main_half_width,
            "maximum_reference_half_width": t.maximum_reference_half_width,
            "maximum_projected_hours": t.maximum_projected_hours,
        },
        **NO_WORK,
    }


_FORBIDDEN_COUNTS = (
    "uncertified_count",
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


def evaluate_strang_preflight(
    metrics: Mapping[str, Any],
    thresholds: StrangRefinementThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or StrangRefinementThresholds()
    checks = {
        name: _eq_one(metrics, name)
        for name in (
            "control_provenance_pass",
            "parent_kernel_gate_pass",
            "parent_target_gate_pass",
            "parent_strang_authorized_pass",
            "eleven_parent_sources_immutable_pass",
            "parent_scientific_config_pass",
            "parent_no_work_pass",
            "variable_k_exposure_pass",
            "palindromic_phase_order_pass",
            "nested_id_uniqueness_pass",
            "nested_id_aliasing_exact_pass",
            "nested_id_marginal_law_pass",
            "nested_id_order_invariance_pass",
            "nested_id_resume_invariance_pass",
            "legacy_k512_replay_pass",
            "local_generator_mixed_digit_pass",
            "local_generator_interior_fixture_pass",
            "k1024_support_certificate_pass",
            "k2048_support_certificate_pass",
            "stationarity_panel_a_pass",
            "stationarity_panel_b_pass",
            "stationarity_joint_max_t_pass",
            "stationarity_panels_immutable_pass",
            "stationarity_panel_disjoint_pass",
        )
    }
    checks.update(
        {
            "parent_record_count": _equal(
                metrics, "parent_record_count", t.parent_record_count
            ),
            "grid_size": _equal(metrics, "grid_size", t.grid_size),
            "alpha": _equal(metrics, "alpha", t.alpha),
            "tau_eff": _equal(metrics, "tau_eff", t.tau_eff),
            "levels": _sequence_equal(
                metrics, "levels", (*t.levels, t.reference_level)
            ),
            "stationarity_panel_count": _equal(
                metrics, "stationarity_panel_count", t.preflight_panel_count
            ),
            "stationarity_paths_per_panel": _equal(
                metrics,
                "stationarity_paths_per_panel",
                t.preflight_paths_per_panel,
            ),
            "stationarity_transitions_per_path": _equal(
                metrics,
                "stationarity_transitions_per_path",
                t.preflight_transitions_per_path,
            ),
            "local_generator_max_error": _le(
                metrics,
                "local_generator_max_error",
                t.maximum_local_generator_error,
            ),
            "minimum_support_rate": _ge(
                metrics, "minimum_support_rate", t.minimum_rate
            ),
            "certificate_fraction": _equal(
                metrics, "certificate_fraction", 1.0
            ),
            "fallback_fraction": _le(
                metrics, "fallback_fraction", t.maximum_fallback_fraction
            ),
            "fallback_cost_fraction": _le(
                metrics,
                "fallback_cost_fraction",
                t.maximum_fallback_cost_fraction,
            ),
            "peak_memory_fraction": _le(
                metrics, "peak_memory_fraction", t.maximum_peak_memory_fraction
            ),
            **{name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS},
        }
    )
    result = _gate(
        "jacobi_rb_strang_preflight",
        "immutable parent, variable-K scheduler, and powered stationarity controls",
        checks,
    )
    result["scheduler_valid"] = int(
        all(
            checks[name]["passed"]
            for name in (
                "variable_k_exposure_pass",
                "palindromic_phase_order_pass",
                "nested_id_uniqueness_pass",
                "nested_id_aliasing_exact_pass",
                "nested_id_marginal_law_pass",
                "nested_id_order_invariance_pass",
                "nested_id_resume_invariance_pass",
                "legacy_k512_replay_pass",
            )
        )
    )
    result["stationarity_valid"] = int(
        all(
            checks[name]["passed"]
            for name in (
                "stationarity_panel_a_pass",
                "stationarity_panel_b_pass",
                "stationarity_joint_max_t_pass",
                "stationarity_panels_immutable_pass",
                "stationarity_panel_disjoint_pass",
                "stationarity_panel_count",
                "stationarity_paths_per_panel",
                "stationarity_transitions_per_path",
            )
        )
    )
    local_generator_names = {
        "local_generator_mixed_digit_pass",
        "local_generator_interior_fixture_pass",
        "local_generator_max_error",
    }
    result["local_generator_valid"] = int(
        all(checks[name]["passed"] for name in local_generator_names)
    )
    numerical_names = {
        "k1024_support_certificate_pass",
        "k2048_support_certificate_pass",
        "certificate_fraction",
        *set(_FORBIDDEN_COUNTS),
    }
    result["numerically_valid"] = int(
        all(checks[name]["passed"] for name in numerical_names)
    )
    resource_names = {
        "minimum_support_rate",
        "fallback_fraction",
        "fallback_cost_fraction",
        "peak_memory_fraction",
    }
    result["resource_valid"] = int(
        all(checks[name]["passed"] for name in resource_names)
    )
    result["thresholds"] = t.to_dict()
    return result


def evaluate_refinement_power(
    metrics: Mapping[str, Any],
    thresholds: StrangRefinementThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or StrangRefinementThresholds()
    checks = {
        name: _eq_one(metrics, name)
        for name in (
            "pilot_complete_pass",
            "pilot_finite_pass",
            "pilot_paths_disjoint_from_preflight_pass",
            "pilot_paths_disjoint_from_production_pass",
            "pilot_means_excluded_pass",
            "variance_only_selection_pass",
            "complete_candidate_grid_pass",
            "selected_design_frozen_pass",
            "selected_design_hash_pass",
            "pilot_certification_pass",
        )
    }
    checks.update(
        {
            "pilot_main_paths": _equal(
                metrics, "pilot_main_paths", t.pilot_main_paths
            ),
            "pilot_reference_paths": _equal(
                metrics, "pilot_reference_paths", t.pilot_reference_paths
            ),
            "candidate_main_paths": _sequence_equal(
                metrics, "candidate_main_paths", t.candidate_main_paths
            ),
            "candidate_reference_paths": _sequence_equal(
                metrics,
                "candidate_reference_paths",
                t.candidate_reference_paths,
            ),
            "selected_main_paths": _check(
                metrics.get("selected_main_paths"),
                "in",
                list(t.candidate_main_paths),
                metrics.get("selected_main_paths") in t.candidate_main_paths,
            ),
            "selected_reference_paths": _check(
                metrics.get("selected_reference_paths"),
                "in",
                list(t.candidate_reference_paths),
                metrics.get("selected_reference_paths")
                in t.candidate_reference_paths,
            ),
            "predicted_main_half_width": _le(
                metrics,
                "predicted_main_half_width",
                t.maximum_main_half_width,
            ),
            "predicted_reference_half_width": _le(
                metrics,
                "predicted_reference_half_width",
                t.maximum_reference_half_width,
            ),
            "projected_production_hours": _le(
                metrics, "projected_production_hours", t.maximum_projected_hours
            ),
            "certificate_fraction": _equal(
                metrics, "certificate_fraction", 1.0
            ),
            **{name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS},
        }
    )
    result = _gate(
        "jacobi_rb_strang_power",
        "disjoint variance-only pilot and frozen production design",
        checks,
    )
    result["numerically_valid"] = int(
        all(
            checks[name]["passed"]
            for name in (
                "pilot_complete_pass",
                "pilot_finite_pass",
                "pilot_certification_pass",
                "certificate_fraction",
                *_FORBIDDEN_COUNTS,
            )
        )
    )
    result["power_valid"] = int(
        all(
            checks[name]["passed"]
            for name in (
                "variance_only_selection_pass",
                "complete_candidate_grid_pass",
                "selected_design_frozen_pass",
                "selected_design_hash_pass",
                "selected_main_paths",
                "selected_reference_paths",
                "predicted_main_half_width",
                "predicted_reference_half_width",
            )
        )
    )
    result["resource_valid"] = int(
        checks["projected_production_hours"]["passed"]
    )
    result["thresholds"] = t.to_dict()
    return result


def evaluate_strang_refinement(
    metrics: Mapping[str, Any],
    thresholds: StrangRefinementThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or StrangRefinementThresholds()
    checks = {
        name: _eq_one(metrics, name)
        for name in (
            "selected_design_binding_pass",
            "production_pilot_isolation_pass",
            "production_complete_pass",
            "production_finite_pass",
            "all_levels_complete_pass",
            "reference_level_subset_pass",
            "observation_plan_pass",
            "observable_family_plan_pass",
            "dirichlet_normalization_pass",
            "paired_whole_path_bootstrap_pass",
            "linear_family_pass",
            "quadratic_family_pass",
            "cubic_family_pass",
            "pooled_family_pass",
            "stationarity_panel_a_pass",
            "stationarity_panel_b_pass",
            "stationarity_all_levels_pass",
            "stationarity_eight_sweep_k512_pass",
            "detailed_balance_max_t_pass",
            "mass_conservation_pass",
            "shard_chain_pass",
            "state_updates_device_resident_pass",
        )
    }
    checks.update(
        {
            "image_sha256": _equal(
                metrics, "image_sha256", t.required_image_sha256
            ),
            "levels": _sequence_equal(
                metrics, "levels", (*t.levels, t.reference_level)
            ),
            "observation_time_fractions": _sequence_equal(
                metrics,
                "observation_time_fractions",
                t.observation_time_fractions,
            ),
            "bootstrap_replicates": _equal(
                metrics, "bootstrap_replicates", t.bootstrap_replicates
            ),
            "bootstrap_confidence": _equal(
                metrics, "bootstrap_confidence", t.bootstrap_confidence
            ),
            "minimum_observed_weak_order": _ge(
                metrics,
                "minimum_observed_weak_order",
                t.minimum_observed_weak_order,
            ),
            "minimum_weak_order_interval_lower": _ge(
                metrics,
                "minimum_weak_order_interval_lower",
                t.minimum_weak_order_interval_lower,
            ),
            "weak_order_two_coverage_fraction": _equal(
                metrics, "weak_order_two_coverage_fraction", 1.0
            ),
            "maximum_512_1024_upper_bound": _le(
                metrics,
                "maximum_512_1024_upper_bound",
                t.maximum_512_1024_discrepancy,
            ),
            "maximum_512_reference_upper_bound": _le(
                metrics,
                "maximum_512_reference_upper_bound",
                t.maximum_512_reference_error,
            ),
            "maximum_reference_stability_upper_bound": _le(
                metrics,
                "maximum_reference_stability_upper_bound",
                t.maximum_reference_instability,
            ),
            "mass_error": _le(
                metrics, "mass_error", t.maximum_cuda_mass_error
            ),
            "certificate_fraction": _equal(
                metrics, "certificate_fraction", 1.0
            ),
            "projected_or_actual_hours": _le(
                metrics, "projected_or_actual_hours", t.maximum_projected_hours
            ),
            **{name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS},
        }
    )
    result = _gate(
        "jacobi_rb_strang_refinement",
        "state-dependent exact split refinement at fixed grid 28",
        checks,
    )
    scientific_names = {
        "linear_family_pass",
        "quadratic_family_pass",
        "cubic_family_pass",
        "pooled_family_pass",
        "minimum_observed_weak_order",
        "minimum_weak_order_interval_lower",
        "weak_order_two_coverage_fraction",
        "maximum_512_1024_upper_bound",
        "maximum_512_reference_upper_bound",
        "maximum_reference_stability_upper_bound",
    }
    result["reference_valid"] = int(
        all(
            checks[name]["passed"]
            for name in (
                "maximum_512_reference_upper_bound",
                "maximum_reference_stability_upper_bound",
                "reference_level_subset_pass",
            )
        )
    )
    result["refinement_valid"] = int(
        all(checks[name]["passed"] for name in scientific_names)
    )
    numerical_names = {
        "production_complete_pass",
        "production_finite_pass",
        "all_levels_complete_pass",
        "mass_conservation_pass",
        "shard_chain_pass",
        "certificate_fraction",
        "mass_error",
        *set(_FORBIDDEN_COUNTS),
    }
    result["numerically_valid"] = int(
        all(checks[name]["passed"] for name in numerical_names)
    )
    result["resource_valid"] = int(checks["projected_or_actual_hours"]["passed"])
    result["thresholds"] = t.to_dict()
    return result


def _failed_names(gate: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(gate, Mapping) or not isinstance(gate.get("subchecks"), Mapping):
        return set()
    return {
        str(name)
        for name, value in gate["subchecks"].items()
        if not isinstance(value, Mapping) or not _one(value.get("passed"))
    }


def decide_strang_refinement_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    power_gate: Mapping[str, Any] | None,
    refinement_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not _passed(provenance):
        decision = StrangRefinementDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 891-artifact parent binding"
    elif _status(preflight_gate) != "evaluated":
        decision = StrangRefinementDecision.REFINEMENT_SCHEDULER_INVALID
        action = "complete the exact variable-K preflight"
    elif not _passed(preflight_gate):
        failed = _failed_names(preflight_gate)
        if failed & {
            "control_provenance_pass",
            "parent_record_count",
            "parent_kernel_gate_pass",
            "parent_target_gate_pass",
            "parent_strang_authorized_pass",
            "eleven_parent_sources_immutable_pass",
            "parent_scientific_config_pass",
            "parent_no_work_pass",
        }:
            decision = StrangRefinementDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable parent/source/config binding"
        elif not _one(preflight_gate.get("scheduler_valid")):
            decision = StrangRefinementDecision.REFINEMENT_SCHEDULER_INVALID
            action = "repair variable-K exposure, coupling IDs, or legacy replay"
        elif not _one(preflight_gate.get("stationarity_valid")):
            decision = StrangRefinementDecision.JACOBI_STATIONARITY_CONTROL_INVALID
            action = "repair or re-examine the frozen powered stationarity controls"
        elif not _one(preflight_gate.get("local_generator_valid")):
            decision = StrangRefinementDecision.STRANG_SPLIT_REFERENCE_INVALID
            action = "repair the exact local Eulerian generator correspondence"
        elif not _one(preflight_gate.get("numerically_valid")):
            decision = (
                StrangRefinementDecision.REFINEMENT_KERNEL_NUMERICALLY_UNRESOLVED
            )
            action = "repair exact high-K certification or local generator algebra"
        else:
            decision = (
                StrangRefinementDecision.REFINEMENT_COMPUTATIONALLY_INFEASIBLE
            )
            action = "repair exact high-K execution scheduling"
    elif _status(power_gate) != "evaluated":
        decision = StrangRefinementDecision.REFINEMENT_POWER_INFEASIBLE
        action = "run the disjoint variance-only power pilot"
    elif not _passed(power_gate):
        if not _one(power_gate.get("numerically_valid")):
            decision = (
                StrangRefinementDecision.REFINEMENT_KERNEL_NUMERICALLY_UNRESOLVED
            )
            action = "repair exact pilot execution"
        elif not _one(power_gate.get("power_valid")):
            decision = StrangRefinementDecision.REFINEMENT_POWER_INFEASIBLE
            action = "retain evidence and redesign a predeclared powered experiment"
        else:
            decision = (
                StrangRefinementDecision.REFINEMENT_COMPUTATIONALLY_INFEASIBLE
            )
            action = "repair exact scheduling within the frozen 48-hour budget"
    elif _status(refinement_gate) != "evaluated":
        decision = StrangRefinementDecision.STRANG_REFINEMENT_INCONCLUSIVE
        action = "complete the frozen production refinement"
    elif not _passed(refinement_gate):
        failed = _failed_names(refinement_gate)
        if failed & {
            "stationarity_panel_a_pass",
            "stationarity_panel_b_pass",
            "stationarity_all_levels_pass",
            "stationarity_eight_sweep_k512_pass",
            "detailed_balance_max_t_pass",
            "mass_conservation_pass",
        }:
            decision = StrangRefinementDecision.STRANG_SPLIT_REFERENCE_INVALID
            action = "repair the exact split reference before model training"
        elif not _one(refinement_gate.get("numerically_valid")):
            decision = (
                StrangRefinementDecision.REFINEMENT_KERNEL_NUMERICALLY_UNRESOLVED
            )
            action = "repair exact production execution or artifact chains"
        elif not _one(refinement_gate.get("resource_valid")):
            decision = (
                StrangRefinementDecision.REFINEMENT_COMPUTATIONALLY_INFEASIBLE
            )
            action = "repair exact scheduling without changing the law or target"
        elif not _one(refinement_gate.get("reference_valid")):
            decision = StrangRefinementDecision.REFINEMENT_REFERENCE_INCONCLUSIVE
            action = "strengthen the exact high-K numerical reference"
        else:
            decision = StrangRefinementDecision.STRANG_REFINEMENT_INCONCLUSIVE
            action = "retain evidence and investigate the fixed split discretization"
    else:
        decision = (
            StrangRefinementDecision.EXACT_STATE_DEPENDENT_STRANG_REFINEMENT_FEASIBLE
        )
        action = "plan one-image phase-conditioned MSE training on exact RB labels"
    authorized = (
        decision
        is StrangRefinementDecision.EXACT_STATE_DEPENDENT_STRANG_REFINEMENT_FEASIBLE
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "decision": decision.value,
        "recommended_next_action": action,
        "one_image_phase_conditioned_training_patch_authorized": int(authorized),
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "closed_terminal_scientific_outcome": int(
            _status(refinement_gate) == "evaluated"
        ),
        **NO_WORK,
    }


__all__ = [
    "BOOTSTRAP_VERSION",
    "MAX_T_VERSION",
    "StrangRefinementDecision",
    "StrangRefinementThresholds",
    "center_scale_observable",
    "decide_strang_refinement_workflow",
    "dirichlet_linear_moments",
    "dirichlet_observable_moments",
    "evaluate_refinement_power",
    "evaluate_strang_preflight",
    "evaluate_strang_refinement",
    "fit_weak_order",
    "not_evaluated_gate",
    "richardson_reference",
    "select_refinement_design",
    "whole_path_max_t_intervals",
    "whole_path_refinement_bootstrap",
]
