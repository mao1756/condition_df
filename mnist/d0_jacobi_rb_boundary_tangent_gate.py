"""Pure gates for the boundary-tangent Jacobi/RB controller workflow.

This module contains no cache builder, trainer, transition kernel, or sampler.
It fixes the statistical families and fail-closed adjudication used by the
additive boundary-tangent workflow:

* a 228-component, one-sided, whole-path studentized max-T family;
* a 784-component, two-sided controller-law family;
* numerical-health and resource checks; and
* the closed workflow decision partition.

The confirmation family contains 224 internal-time combined-vs-zero cells
(``4 quartiles x 7 phases x 8 controller midpoints``) followed by four
quartile-pooled combined-vs-baseline contrasts.  The order is scientific
configuration, not presentation metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "d0-jacobi-rb-boundary-tangent-gate-v1"
SCHEMA_VERSION = 1
CLAIM_SCOPE = (
    "time-local boundary-tangent learnability and at-most-eight-phase "
    "reverse-controller controls for one frozen image under the exact "
    "certified K512 split chain"
)

TIME_QUARTILES = 4
PHASE_COUNT = 7
MIDPOINT_COUNT = 8
COMBINED_VS_ZERO_FAMILY_SIZE = TIME_QUARTILES * PHASE_COUNT * MIDPOINT_COUNT
COMBINED_VS_BASELINE_FAMILY_SIZE = TIME_QUARTILES
CONFIRMATION_FAMILY_SIZE = (
    COMBINED_VS_ZERO_FAMILY_SIZE + COMBINED_VS_BASELINE_FAMILY_SIZE
)
CONTROLLER_FAMILY_SIZE = 784
CONTROLLER_BIAS_FAMILY_SIZE = CONTROLLER_FAMILY_SIZE // 2
CONTROLLER_REFINEMENT_FAMILY_SIZE = CONTROLLER_FAMILY_SIZE // 2

DEFAULT_ROOT_SEED = 261_311
DEFAULT_BOOTSTRAP_SEED = 261_315
DEFAULT_CONTROLLER_BOOTSTRAP_SEED = 261_316
DEFAULT_BOOTSTRAP_REPLICATES = 50_000
DEFAULT_SIMULTANEOUS_CONFIDENCE = 0.995
_BOOTSTRAP_NAMESPACE = 0x42544754

NO_SAMPLING_OR_RECONSTRUCTION = {
    "reverse_sampling_authorized": 0,
    "sampling_authorized": 0,
    "reconstruction_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "known_prior_claim_authorized": 0,
    "full_dataset_training_authorized": 0,
    "unsplit_generator_claim_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
    "reverse_sampling_performed": 0,
    "sampling_performed": 0,
    "image_sampling_performed": 0,
    "reconstruction_performed": 0,
    "full_reverse_path_performed": 0,
}


class BoundaryTangentGateError(ValueError):
    """Raised when evidence violates a frozen gate contract."""


class BoundaryTangentDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    FAILED_CONTROLLER_ADJUDICATION_INVALID = (
        "failed_controller_adjudication_invalid"
    )
    BOUNDARY_TANGENT_REPRESENTATION_INVALID = (
        "boundary_tangent_representation_invalid"
    )
    BOUNDARY_TANGENT_DESIGN_INFEASIBLE = (
        "boundary_tangent_design_infeasible"
    )
    FRESH_EXACT_CACHE_INVALID = "fresh_exact_cache_invalid"
    BOUNDARY_TANGENT_BASELINE_INVALID = "boundary_tangent_baseline_invalid"
    BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID = (
        "boundary_tangent_optimization_pipeline_invalid"
    )
    BOUNDARY_TANGENT_BASELINE_ONLY_SIGNAL = (
        "boundary_tangent_baseline_only_signal"
    )
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_NOT_DETECTED = (
        "boundary_tangent_time_local_signal_not_detected"
    )
    BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE = (
        "boundary_tangent_audit_inconclusive"
    )
    PAIRED_RISK_INFERENCE_INVALID = "paired_risk_inference_invalid"
    BOUNDARY_TANGENT_CONTROLLER_NUMERICALLY_INVALID = (
        "boundary_tangent_controller_numerically_invalid"
    )
    REVERSE_CONTROLLER_WEAK_LAW_FAILED = "reverse_controller_weak_law_failed"
    REVERSE_CONTROLLER_MICROSTEP_REFINEMENT_FAILED = (
        "reverse_controller_microstep_refinement_failed"
    )
    EXACT_RB_BOUNDARY_TANGENT_CONTROLLER_CONTROLLED = (
        "exact_rb_boundary_tangent_controller_controlled"
    )


@dataclass(frozen=True)
class BoundaryTangentThresholds:
    """Frozen production thresholds for statistics and controller health."""

    confirmation_paths: int = 64
    controller_paths: int = 64
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    simultaneous_confidence: float = DEFAULT_SIMULTANEOUS_CONFIDENCE
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    controller_bootstrap_seed: int = DEFAULT_CONTROLLER_BOOTSTRAP_SEED
    maximum_weak_law_bias: float = 0.10
    maximum_microstep_refinement_error: float = 0.05
    maximum_mass_error: float = 2.0e-12
    minimum_transitions_per_second: float = 1300.0
    maximum_peak_memory_fraction: float = 0.80
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_time_fraction: float = 0.10
    maximum_persisted_bytes: int = 5 * 1024**3 // 4

    def __post_init__(self) -> None:
        integer_fields = (
            "confirmation_paths",
            "controller_paths",
            "bootstrap_replicates",
            "bootstrap_seed",
            "controller_bootstrap_seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            minimum = 8 if name in {"confirmation_paths", "controller_paths"} else 0
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
                or (name == "bootstrap_replicates" and value == 0)
            ):
                raise BoundaryTangentGateError(f"{name} is invalid")
        if not 0.5 < float(self.simultaneous_confidence) < 1.0:
            raise BoundaryTangentGateError(
                "simultaneous_confidence must lie in (0.5,1)"
            )
        positive = (
            "maximum_weak_law_bias",
            "maximum_microstep_refinement_error",
            "maximum_mass_error",
            "minimum_transitions_per_second",
            "maximum_fallback_fraction",
            "maximum_fallback_time_fraction",
        )
        if any(
            not math.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) <= 0.0
            for name in positive
        ):
            raise BoundaryTangentGateError("a positive threshold is invalid")
        if not 0.0 < float(self.maximum_peak_memory_fraction) <= 1.0:
            raise BoundaryTangentGateError("maximum_peak_memory_fraction is invalid")
        if (
            not isinstance(self.maximum_persisted_bytes, int)
            or isinstance(self.maximum_persisted_bytes, bool)
            or self.maximum_persisted_bytes <= 0
        ):
            raise BoundaryTangentGateError("maximum_persisted_bytes is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def confirmation_family_names() -> tuple[str, ...]:
    """Return the frozen 228-member family in canonical order."""

    zero = tuple(
        f"combined_vs_zero.q{quartile}.phase{phase}.midpoint{midpoint}"
        for quartile in range(TIME_QUARTILES)
        for phase in range(PHASE_COUNT)
        for midpoint in range(MIDPOINT_COUNT)
    )
    baseline = tuple(
        f"combined_vs_baseline.q{quartile}"
        for quartile in range(TIME_QUARTILES)
    )
    names = zero + baseline
    if len(names) != CONFIRMATION_FAMILY_SIZE or len(set(names)) != len(names):
        raise AssertionError("the frozen confirmation family is malformed")
    return names


CONFIRMATION_FAMILY_NAMES = confirmation_family_names()
COMBINED_VS_ZERO_NAMES = CONFIRMATION_FAMILY_NAMES[
    :COMBINED_VS_ZERO_FAMILY_SIZE
]
COMBINED_VS_BASELINE_NAMES = CONFIRMATION_FAMILY_NAMES[
    COMBINED_VS_ZERO_FAMILY_SIZE:
]


def claim_scope_flags(
    *,
    controlled: bool = False,
    physical_training_performed: bool = False,
    controller_control_trajectory_performed: bool = False,
) -> dict[str, Any]:
    """Return the claim firewall shared by every workflow artifact.

    ``controlled`` authorizes planning a later reconstruction-control patch;
    it never authorizes or records sampling or reconstruction itself.
    """

    return {
        "claim_scope": CLAIM_SCOPE,
        "boundary_tangent_controller_controlled": int(bool(controlled)),
        "one_image_reconstruction_control_planning_authorized": int(
            bool(controlled)
        ),
        "physical_training_performed": int(bool(physical_training_performed)),
        "controller_control_trajectory_performed": int(
            bool(controller_control_trajectory_performed)
        ),
        **NO_SAMPLING_OR_RECONSTRUCTION,
    }


def _canonical_path_table(
    values: Any,
    path_ids: Any | None,
    *,
    family_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values)
    if raw.dtype != np.dtype(np.float64):
        raise BoundaryTangentGateError("path values must be float64")
    if (
        raw.ndim != 2
        or raw.shape[0] < 8
        or raw.shape[1] != int(family_size)
        or not np.isfinite(raw).all()
    ):
        raise BoundaryTangentGateError(
            f"path values must be finite [at-least-8,{family_size}]"
        )
    if path_ids is None:
        paths = np.arange(raw.shape[0], dtype=np.int64)
    else:
        source = np.asarray(path_ids)
        if source.ndim != 1 or source.shape[0] != raw.shape[0] or source.dtype.kind not in "iu":
            raise BoundaryTangentGateError(
                "path_ids must be a one-dimensional integer path table"
            )
        paths = np.asarray(source, dtype=np.int64)
    if (
        np.any(paths < 0)
        or np.unique(paths).size != paths.size
    ):
        raise BoundaryTangentGateError("path_ids must be unique and nonnegative")
    order = np.argsort(paths, kind="stable")
    return (
        np.ascontiguousarray(paths[order]),
        np.ascontiguousarray(raw[order]),
    )


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    result = float(np.quantile(values, float(probability), method="higher"))
    if not math.isfinite(result):
        raise BoundaryTangentGateError("bootstrap critical value is nonfinite")
    return result


def one_sided_whole_path_max_t(
    values: Any,
    *,
    path_ids: Any | None = None,
    names: Sequence[str] = CONFIRMATION_FAMILY_NAMES,
    confidence: float = DEFAULT_SIMULTANEOUS_CONFIDENCE,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    namespace: int = 0,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Compute deterministic centered whole-path one-sided max-T bounds.

    Rows, never cells or edges, are resampled jointly across the full family.
    Path IDs are sorted before inference, so input row order cannot alter the
    result.  ``chunk_size`` affects memory only and not the random stream.
    """

    family_names = tuple(str(name) for name in names)
    if (
        len(family_names) != CONFIRMATION_FAMILY_SIZE
        or family_names != CONFIRMATION_FAMILY_NAMES
        or len(set(family_names)) != len(family_names)
    ):
        raise BoundaryTangentGateError("confirmation family names changed")
    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates <= 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or not isinstance(namespace, int)
        or isinstance(namespace, bool)
        or namespace < 0
        or not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
        or not 0.5 < float(confidence) < 1.0
    ):
        raise BoundaryTangentGateError("max-T configuration is invalid")
    paths, table = _canonical_path_table(
        values, path_ids, family_size=CONFIRMATION_FAMILY_SIZE
    )
    path_count = int(paths.size)
    point = np.mean(table, axis=0, dtype=np.float64)
    standard_error = np.std(table, axis=0, ddof=1) / math.sqrt(path_count)
    if (
        not np.isfinite(point).all()
        or not np.isfinite(standard_error).all()
        or np.any(standard_error <= 0.0)
    ):
        raise BoundaryTangentGateError(
            "confirmation family has degenerate/nonfinite studentization"
        )
    generator = np.random.Generator(
        np.random.Philox(
            [int(seed), int(namespace), _BOOTSTRAP_NAMESPACE]
        )
    )
    maxima = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), int(chunk_size)):
        stop = min(int(replicates), start + int(chunk_size))
        indices = generator.integers(
            0,
            path_count,
            size=(stop - start, path_count),
            dtype=np.int64,
        )
        sampled = table[indices]
        sampled_mean = np.mean(sampled, axis=1, dtype=np.float64)
        sampled_error = np.std(sampled, axis=1, ddof=1) / math.sqrt(path_count)
        if not np.isfinite(sampled_error).all() or np.any(sampled_error <= 0.0):
            raise BoundaryTangentGateError(
                "bootstrap produced degenerate/nonfinite studentization"
            )
        maxima[start:stop] = np.max(
            (sampled_mean - point[None, :]) / sampled_error,
            axis=1,
        )
    critical = _higher_quantile(maxima, float(confidence))
    lower = point - critical * standard_error
    passed = bool(np.all(lower > 0.0))
    return {
        "schema": SCHEMA + "-confirmation-max-t",
        "schema_version": SCHEMA_VERSION,
        "method": "centered_whole_path_studentized_max_t",
        "bootstrap_unit": "whole_path_jointly_across_family",
        "quantile_method": "higher",
        "family_names": list(family_names),
        "family_size": len(family_names),
        "point_estimates": {
            name: float(value)
            for name, value in zip(family_names, point, strict=True)
        },
        "standard_errors": {
            name: float(value)
            for name, value in zip(family_names, standard_error, strict=True)
        },
        "lower_bounds": {
            name: float(value)
            for name, value in zip(family_names, lower, strict=True)
        },
        "critical_value": critical,
        "path_ids": paths.tolist(),
        "path_count": path_count,
        "confidence": float(confidence),
        "replicates": int(replicates),
        "seed": int(seed),
        "namespace": int(namespace),
        "negative_values_truncated": 0,
        "combined_vs_zero_all_point_positive": int(
            np.all(point[:COMBINED_VS_ZERO_FAMILY_SIZE] > 0.0)
        ),
        "combined_vs_zero_all_lower_positive": int(
            np.all(lower[:COMBINED_VS_ZERO_FAMILY_SIZE] > 0.0)
        ),
        "combined_vs_baseline_all_point_positive": int(
            np.all(point[COMBINED_VS_ZERO_FAMILY_SIZE:] > 0.0)
        ),
        "combined_vs_baseline_all_lower_positive": int(
            np.all(lower[COMBINED_VS_ZERO_FAMILY_SIZE:] > 0.0)
        ),
        "passed": int(passed),
        **claim_scope_flags(),
    }


def _finite_mapping(
    record: Mapping[str, Any], key: str, names: Sequence[str]
) -> dict[str, float]:
    raw = record.get(key)
    if (
        not isinstance(raw, Mapping)
        or len(raw) != len(names)
        or set(raw) != set(names)
    ):
        raise BoundaryTangentGateError(f"{key} does not match the frozen family")
    try:
        result = {name: float(raw[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise BoundaryTangentGateError(f"{key} is invalid") from exc
    if not all(math.isfinite(value) for value in result.values()):
        raise BoundaryTangentGateError(f"{key} contains nonfinite values")
    return result


def validate_confirmation_family(
    record: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentThresholds | None = None,
) -> dict[str, Any]:
    """Validate and summarize a frozen 228-member confirmation record."""

    t = thresholds or BoundaryTangentThresholds()
    names = tuple(record.get("family_names", ()))
    if (
        names != CONFIRMATION_FAMILY_NAMES
        or int(record.get("family_size", -1)) != CONFIRMATION_FAMILY_SIZE
        or record.get("method") != "centered_whole_path_studentized_max_t"
        or record.get("bootstrap_unit")
        != "whole_path_jointly_across_family"
        or record.get("quantile_method") != "higher"
        or int(record.get("path_count", -1)) != t.confirmation_paths
        or float(record.get("confidence", -1.0)) != t.simultaneous_confidence
        or int(record.get("replicates", -1)) != t.bootstrap_replicates
        or int(record.get("seed", -1)) != t.bootstrap_seed
        or int(record.get("negative_values_truncated", -1)) != 0
    ):
        raise BoundaryTangentGateError(
            "confirmation max-T configuration does not match production"
        )
    critical = record.get("critical_value")
    if not isinstance(critical, (int, float)) or not math.isfinite(float(critical)):
        raise BoundaryTangentGateError("confirmation critical value is invalid")
    points = _finite_mapping(record, "point_estimates", names)
    errors = _finite_mapping(record, "standard_errors", names)
    lower = _finite_mapping(record, "lower_bounds", names)
    if any(value <= 0.0 for value in errors.values()):
        raise BoundaryTangentGateError("confirmation standard error is degenerate")
    zero_point = all(points[name] > 0.0 for name in COMBINED_VS_ZERO_NAMES)
    zero_lower = all(lower[name] > 0.0 for name in COMBINED_VS_ZERO_NAMES)
    baseline_point = all(
        points[name] > 0.0 for name in COMBINED_VS_BASELINE_NAMES
    )
    baseline_lower = all(
        lower[name] > 0.0 for name in COMBINED_VS_BASELINE_NAMES
    )
    expected_pass = zero_lower and baseline_lower
    if "passed" in record and int(record["passed"]) != int(expected_pass):
        raise BoundaryTangentGateError("confirmation pass flag is inconsistent")
    return {
        "paired_risk_inference_valid": 1,
        "family_size": CONFIRMATION_FAMILY_SIZE,
        "combined_vs_zero_point_positive": int(zero_point),
        "combined_vs_zero_replicated": int(zero_lower),
        "combined_vs_baseline_point_positive": int(baseline_point),
        "combined_vs_baseline_replicated": int(baseline_lower),
        "all_simultaneous_lower_bounds_positive": int(expected_pass),
    }


def _check(value: Any, comparator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "comparator": comparator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _gate(
    name: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    numerical_valid: bool | None = None,
    resource_valid: bool | None = None,
    physical_training_performed: bool = False,
    controller_control_trajectory_performed: bool = False,
    **evidence: Any,
) -> dict[str, Any]:
    passed = bool(checks) and all(int(item.get("passed", 0)) == 1 for item in checks.values())
    return {
        "schema": SCHEMA + f"-{name}-gate",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "gate": str(name),
        "checks": {str(key): dict(value) for key, value in checks.items()},
        "passed": int(passed),
        "numerically_valid": int(passed if numerical_valid is None else numerical_valid),
        "resource_valid": int(passed if resource_valid is None else resource_valid),
        **evidence,
        **claim_scope_flags(
            physical_training_performed=physical_training_performed,
            controller_control_trajectory_performed=(
                controller_control_trajectory_performed
            ),
        ),
    }


def evaluate_confirmation_gate(
    record: Mapping[str, Any],
    *,
    integrity_checks: Mapping[str, bool] | None = None,
    thresholds: BoundaryTangentThresholds | None = None,
) -> dict[str, Any]:
    """Fail-closed gate for fresh sealed 228-family confirmation evidence."""

    t = thresholds or BoundaryTangentThresholds()
    error: str | None = None
    try:
        summary = validate_confirmation_family(record, thresholds=t)
    except (BoundaryTangentGateError, TypeError, ValueError) as exc:
        error = str(exc)
        summary = {
            "paired_risk_inference_valid": 0,
            "family_size": int(record.get("family_size", -1))
            if isinstance(record, Mapping)
            else -1,
            "combined_vs_zero_point_positive": 0,
            "combined_vs_zero_replicated": 0,
            "combined_vs_baseline_point_positive": 0,
            "combined_vs_baseline_replicated": 0,
            "all_simultaneous_lower_bounds_positive": 0,
        }
    integrity = dict(integrity_checks or {})
    checks = {
        "paired_risk_inference_valid": _check(
            summary["paired_risk_inference_valid"], "==", 1,
            bool(summary["paired_risk_inference_valid"]),
        ),
        "confirmation_family_size": _check(
            summary["family_size"], "==", CONFIRMATION_FAMILY_SIZE,
            summary["family_size"] == CONFIRMATION_FAMILY_SIZE,
        ),
        "combined_vs_zero_all_simultaneous_lower_positive": _check(
            summary["combined_vs_zero_replicated"], "==", 1,
            bool(summary["combined_vs_zero_replicated"]),
        ),
        "combined_vs_baseline_all_simultaneous_lower_positive": _check(
            summary["combined_vs_baseline_replicated"], "==", 1,
            bool(summary["combined_vs_baseline_replicated"]),
        ),
    }
    checks.update(
        {
            str(name): _check(int(bool(value)), "==", 1, bool(value))
            for name, value in sorted(integrity.items())
        }
    )
    return _gate(
        "confirm",
        checks,
        numerical_valid=bool(summary["paired_risk_inference_valid"]),
        resource_valid=True,
        physical_training_performed=True,
        inference_error=error,
        max_t_record=dict(record),
        **summary,
    )


def validate_controller_family(
    record: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentThresholds | None = None,
) -> dict[str, Any]:
    """Validate a 784-component normalized two-sided controller result."""

    t = thresholds or BoundaryTangentThresholds()
    names = tuple(str(name) for name in record.get("family_names", ()))
    bias_names = tuple(name for name in names if name.endswith(".bias"))
    refinement_names = tuple(
        name for name in names if name.endswith(".M8_vs_M4")
    )
    path_ids_source = record.get("path_ids")
    path_ids = (
        np.asarray(path_ids_source)
        if isinstance(path_ids_source, (list, tuple, np.ndarray))
        else np.asarray([], dtype=np.int64)
    )
    path_ids_valid = bool(
        path_ids.ndim == 1
        and path_ids.dtype.kind in "iu"
        and path_ids.size == t.controller_paths
        and np.all(path_ids >= 0)
        and np.all(path_ids < 1 << 20)
        and np.unique(path_ids).size == path_ids.size
    )
    if (
        len(names) != CONTROLLER_FAMILY_SIZE
        or len(set(names)) != CONTROLLER_FAMILY_SIZE
        or len(bias_names) != CONTROLLER_BIAS_FAMILY_SIZE
        or len(refinement_names) != CONTROLLER_REFINEMENT_FAMILY_SIZE
        or set(bias_names).intersection(refinement_names)
        or set(bias_names).union(refinement_names) != set(names)
        or int(record.get("family_size", -1)) != CONTROLLER_FAMILY_SIZE
        or record.get("method")
        != "whole_path_rms_normalized_two_sided_studentized_max_t"
        or record.get("bootstrap_unit") != "whole_path"
        or int(record.get("denominator_recomputed_per_resample", -1)) != 1
        or record.get("quantile_method") != "higher"
        or int(record.get("path_count", -1)) != t.controller_paths
        or not path_ids_valid
        or float(record.get("confidence", -1.0)) != t.simultaneous_confidence
        or int(record.get("replicates", -1)) != t.bootstrap_replicates
        or int(record.get("seed", -1)) != t.controller_bootstrap_seed
        or int(record.get("negative_values_truncated", -1)) != 0
    ):
        raise BoundaryTangentGateError(
            "controller max-T configuration does not match production"
        )
    critical = record.get("critical_value")
    if not isinstance(critical, (int, float)) or not math.isfinite(float(critical)):
        raise BoundaryTangentGateError("controller critical value is invalid")
    points = _finite_mapping(record, "point_estimates", names)
    errors = _finite_mapping(record, "standard_errors", names)
    upper = _finite_mapping(record, "simultaneous_upper_absolute", names)
    if any(value <= 0.0 for value in errors.values()) or any(
        value < 0.0 for value in upper.values()
    ):
        raise BoundaryTangentGateError("controller max-T bounds are invalid")
    weak_law = all(
        upper[name] <= t.maximum_weak_law_bias for name in bias_names
    )
    refinement = all(
        upper[name] <= t.maximum_microstep_refinement_error
        for name in refinement_names
    )
    return {
        "controller_family_valid": 1,
        "trajectory_family_size": CONTROLLER_FAMILY_SIZE,
        "weak_law_controlled": int(weak_law),
        "microstep_refinement_controlled": int(refinement),
        "maximum_bias_upper_absolute": max(upper[name] for name in bias_names),
        "maximum_refinement_upper_absolute": max(
            upper[name] for name in refinement_names
        ),
        "point_estimates_finite": int(bool(points)),
    }


def _finite_number(record: Mapping[str, Any], name: str, default: float) -> float:
    value = record.get(name, default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _integer_equals(value: Any, expected: int) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and int(value) == int(expected)
    )


def evaluate_numerical_resource_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate exact-controller numerical health and execution resources."""

    t = thresholds or BoundaryTangentThresholds()
    pair_mass = _finite_number(metrics, "maximum_pair_mass_error", math.inf)
    simplex_mass = _finite_number(metrics, "maximum_simplex_mass_error", math.inf)
    certificate = _finite_number(metrics, "certificate_fraction", -math.inf)
    fallback = _finite_number(metrics, "fallback_fraction", math.inf)
    fallback_time = _finite_number(metrics, "fallback_time_fraction", math.inf)
    rate = _finite_number(metrics, "transitions_per_second", -math.inf)
    memory = _finite_number(metrics, "peak_device_memory_fraction", math.inf)
    persisted = metrics.get("total_persisted_bytes", metrics.get("persisted_bytes"))
    persisted_valid = (
        isinstance(persisted, int)
        and not isinstance(persisted, bool)
        and 0 <= persisted <= t.maximum_persisted_bytes
    )
    forbidden = metrics.get("forbidden_counts")
    controller_forbidden = metrics.get("controller_forbidden_counts")
    forbidden_valid = bool(forbidden) and isinstance(forbidden, Mapping) and all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and int(value) == 0
        for value in forbidden.values()
    )
    controller_forbidden_valid = (
        bool(controller_forbidden)
        and isinstance(controller_forbidden, Mapping)
        and all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, bool)
            and int(value) == 0
            for value in controller_forbidden.values()
        )
    )
    checks = {
        "certificate_fraction": _check(certificate, "==", 1.0, certificate == 1.0),
        "states_finite": _check(
            metrics.get("states_finite"), "==", 1,
            _integer_equals(metrics.get("states_finite"), 1),
        ),
        "states_nonnegative": _check(
            metrics.get("states_nonnegative"), "==", 1,
            _integer_equals(metrics.get("states_nonnegative"), 1),
        ),
        "pair_mass": _check(
            pair_mass,
            "<=",
            t.maximum_mass_error,
            pair_mass <= t.maximum_mass_error,
        ),
        "simplex_mass": _check(
            simplex_mass,
            "<=",
            t.maximum_mass_error,
            simplex_mass <= t.maximum_mass_error,
        ),
        "boundary_rejections": _check(
            metrics.get("boundary_rejection_count"), "==", 0,
            _integer_equals(metrics.get("boundary_rejection_count"), 0),
        ),
        "forbidden_counts": _check(forbidden, "all ==", 0, forbidden_valid),
        "controller_forbidden_counts": _check(
            controller_forbidden, "all ==", 0, controller_forbidden_valid
        ),
        "fallback_fraction": _check(
            fallback,
            "<=",
            t.maximum_fallback_fraction,
            fallback <= t.maximum_fallback_fraction,
        ),
        "fallback_time_fraction": _check(
            fallback_time,
            "<=",
            t.maximum_fallback_time_fraction,
            fallback_time <= t.maximum_fallback_time_fraction,
        ),
        "throughput": _check(
            rate,
            ">=",
            t.minimum_transitions_per_second,
            rate >= t.minimum_transitions_per_second,
        ),
        "peak_memory": _check(
            memory,
            "<=",
            t.maximum_peak_memory_fraction,
            memory <= t.maximum_peak_memory_fraction,
        ),
        "persisted_size": _check(
            persisted,
            "<=",
            t.maximum_persisted_bytes,
            persisted_valid,
        ),
    }
    numerical_names = (
        "certificate_fraction",
        "states_finite",
        "states_nonnegative",
        "pair_mass",
        "simplex_mass",
        "boundary_rejections",
        "forbidden_counts",
        "controller_forbidden_counts",
    )
    resource_names = (
        "fallback_fraction",
        "fallback_time_fraction",
        "throughput",
        "peak_memory",
        "persisted_size",
    )
    numerical = all(checks[name]["passed"] == 1 for name in numerical_names)
    resource = all(checks[name]["passed"] == 1 for name in resource_names)
    return _gate(
        "controller_health",
        checks,
        numerical_valid=numerical,
        resource_valid=resource,
        metrics=dict(metrics),
    )


def evaluate_controller_gate(
    trajectory: Mapping[str, Any],
    health: Mapping[str, Any],
    *,
    integrity_checks: Mapping[str, bool] | None = None,
    thresholds: BoundaryTangentThresholds | None = None,
) -> dict[str, Any]:
    """Combine the 784-family law result with exact numerical health."""

    t = thresholds or BoundaryTangentThresholds()
    error: str | None = None
    try:
        family = validate_controller_family(trajectory, thresholds=t)
    except (BoundaryTangentGateError, TypeError, ValueError) as exc:
        error = str(exc)
        family = {
            "controller_family_valid": 0,
            "trajectory_family_size": int(trajectory.get("family_size", -1)),
            "weak_law_controlled": 0,
            "microstep_refinement_controlled": 0,
            "maximum_bias_upper_absolute": math.inf,
            "maximum_refinement_upper_absolute": math.inf,
        }
    health_gate = evaluate_numerical_resource_gate(health, thresholds=t)
    checks = {
        "trajectory_family_valid": _check(
            family["controller_family_valid"], "==", 1,
            bool(family["controller_family_valid"]),
        ),
        "weak_law_controlled": _check(
            family["maximum_bias_upper_absolute"], "<=",
            t.maximum_weak_law_bias, bool(family["weak_law_controlled"]),
        ),
        "microstep_refinement_controlled": _check(
            family["maximum_refinement_upper_absolute"], "<=",
            t.maximum_microstep_refinement_error,
            bool(family["microstep_refinement_controlled"]),
        ),
        "numerical_health": _check(
            health_gate["numerically_valid"], "==", 1,
            bool(health_gate["numerically_valid"]),
        ),
        "resource_health": _check(
            health_gate["resource_valid"], "==", 1,
            bool(health_gate["resource_valid"]),
        ),
    }
    checks.update(
        {
            str(name): _check(int(bool(value)), "==", 1, bool(value))
            for name, value in sorted(dict(integrity_checks or {}).items())
        }
    )
    return _gate(
        "control",
        checks,
        numerical_valid=(
            bool(family["controller_family_valid"])
            and bool(health_gate["numerically_valid"])
        ),
        resource_valid=bool(health_gate["resource_valid"]),
        physical_training_performed=True,
        controller_control_trajectory_performed=True,
        trajectory_validation_error=error,
        trajectory_max_t=dict(trajectory),
        controller_health_gate=health_gate,
        **family,
    )


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA + f"-{name}-gate",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "not_evaluated",
        "gate": str(name),
        "reason": str(reason),
        "passed": 0,
        **claim_scope_flags(),
    }


def _status(gate: Mapping[str, Any] | None) -> str:
    if not isinstance(gate, Mapping):
        return "not_evaluated"
    return str(gate.get("evaluation_status", "not_evaluated"))


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return _status(gate) == "evaluated" and int((gate or {}).get("passed", 0)) == 1


def _flag(gate: Mapping[str, Any] | None, *names: str, default: bool = False) -> bool:
    if not isinstance(gate, Mapping):
        return bool(default)
    for name in names:
        if name in gate:
            try:
                return int(gate[name]) == 1
            except (TypeError, ValueError):
                return False
    return bool(default)


_ACTIONS = {
    BoundaryTangentDecision.CONTROL_PROVENANCE_INVALID.value:
        "repair immutable parent provenance",
    BoundaryTangentDecision.FAILED_CONTROLLER_ADJUDICATION_INVALID.value:
        "repair the read-only adjudication of the failed affine controller",
    BoundaryTangentDecision.BOUNDARY_TANGENT_REPRESENTATION_INVALID.value:
        "repair boundary-tangent parameterization controls",
    BoundaryTangentDecision.BOUNDARY_TANGENT_DESIGN_INFEASIBLE.value:
        "repair the fresh controls-only design without changing the target",
    BoundaryTangentDecision.FRESH_EXACT_CACHE_INVALID.value:
        "repair fresh exact certified cache generation",
    BoundaryTangentDecision.BOUNDARY_TANGENT_BASELINE_INVALID.value:
        "repair the frozen boundary-tangent baseline",
    BoundaryTangentDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID.value:
        "repair synthetic/null optimization controls",
    BoundaryTangentDecision.BOUNDARY_TANGENT_BASELINE_ONLY_SIGNAL.value:
        "retain the baseline; do not claim learned boundary-tangent signal",
    BoundaryTangentDecision.SELECTION_FALSE_DISCOVERY.value:
        "record that the selected residual did not beat its baseline",
    BoundaryTangentDecision.BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_NOT_DETECTED.value:
        "do not construct a reverse controller from undetected time-local signal",
    BoundaryTangentDecision.BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE.value:
        "retain the sealed audit without resizing or regenerating it",
    BoundaryTangentDecision.PAIRED_RISK_INFERENCE_INVALID.value:
        "repair report-only paired max-T inference",
    BoundaryTangentDecision.BOUNDARY_TANGENT_CONTROLLER_NUMERICALLY_INVALID.value:
        "repair controller numerical health without clipping or changing the target",
    BoundaryTangentDecision.REVERSE_CONTROLLER_WEAK_LAW_FAILED.value:
        "improve the time-local boundary-tangent predictor on fresh evidence",
    BoundaryTangentDecision.REVERSE_CONTROLLER_MICROSTEP_REFINEMENT_FAILED.value:
        "repair the controller integration while preserving the exact law",
    BoundaryTangentDecision.EXACT_RB_BOUNDARY_TANGENT_CONTROLLER_CONTROLLED.value:
        "plan a separate one-image conditional reconstruction control",
}


def decide_boundary_tangent_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    control_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the closed fail-closed boundary-tangent decision partition."""

    if _status(preflight_gate) == "not_evaluated":
        decision, action = "ready_for_preflight", "verify parents and controller algebra"
    elif not _passed(preflight_gate):
        failure_domain = str((preflight_gate or {}).get("failure_domain", ""))
        if failure_domain == "provenance" or not _flag(
            preflight_gate, "provenance_valid", "control_provenance_valid"
        ):
            decision = BoundaryTangentDecision.CONTROL_PROVENANCE_INVALID.value
        elif not _flag(
            preflight_gate,
            "failed_controller_adjudication_valid",
            "parent_adjudication_valid",
        ):
            decision = BoundaryTangentDecision.FAILED_CONTROLLER_ADJUDICATION_INVALID.value
        elif not _flag(
            preflight_gate,
            "boundary_tangent_representation_valid",
            "representation_valid",
        ):
            decision = BoundaryTangentDecision.BOUNDARY_TANGENT_REPRESENTATION_INVALID.value
        else:
            decision = BoundaryTangentDecision.BOUNDARY_TANGENT_DESIGN_INFEASIBLE.value
        action = _ACTIONS[decision]
    elif _status(cache_gate) == "not_evaluated":
        decision = "ready_for_cache"
        action = "generate fresh exact train and validation caches"
    elif not _passed(cache_gate):
        decision = BoundaryTangentDecision.FRESH_EXACT_CACHE_INVALID.value
        action = _ACTIONS[decision]
    elif _status(train_gate) == "not_evaluated":
        decision = "ready_for_train"
        action = "run baseline, synthetic, null, and physical selection tasks"
    elif not _flag(
        train_gate,
        "optimization_pipeline_valid",
        "boundary_tangent_optimization_pipeline_valid",
    ):
        decision = BoundaryTangentDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID.value
        action = _ACTIONS[decision]
    elif not _flag(train_gate, "boundary_tangent_baseline_valid", "baseline_valid"):
        decision = BoundaryTangentDecision.BOUNDARY_TANGENT_BASELINE_INVALID.value
        action = _ACTIONS[decision]
    elif _flag(train_gate, "boundary_tangent_baseline_only", "baseline_only"):
        decision = BoundaryTangentDecision.BOUNDARY_TANGENT_BASELINE_ONLY_SIGNAL.value
        action = _ACTIONS[decision]
    elif not _passed(train_gate):
        decision = BoundaryTangentDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID.value
        action = _ACTIONS[decision]
    elif _status(confirm_gate) == "not_evaluated":
        decision, action = "ready_for_confirm", "open the fresh sealed confirmation exactly once"
    elif not _flag(confirm_gate, "paired_risk_inference_valid"):
        decision = BoundaryTangentDecision.PAIRED_RISK_INFERENCE_INVALID.value
        action = _ACTIONS[decision]
    elif not _flag(confirm_gate, "combined_vs_baseline_point_positive"):
        decision = BoundaryTangentDecision.SELECTION_FALSE_DISCOVERY.value
        action = _ACTIONS[decision]
    elif not _flag(confirm_gate, "combined_vs_zero_point_positive"):
        decision = BoundaryTangentDecision.BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_NOT_DETECTED.value
        action = _ACTIONS[decision]
    elif not (
        _flag(confirm_gate, "combined_vs_zero_replicated")
        and _flag(confirm_gate, "combined_vs_baseline_replicated")
    ):
        decision = BoundaryTangentDecision.BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE.value
        action = _ACTIONS[decision]
    elif not _passed(confirm_gate):
        decision = BoundaryTangentDecision.PAIRED_RISK_INFERENCE_INVALID.value
        action = _ACTIONS[decision]
    elif _status(control_gate) == "not_evaluated":
        decision, action = "ready_for_control", "run at-most-eight-phase controller controls"
    elif not _flag(control_gate, "controller_family_valid", default=False) or not _flag(
        control_gate, "numerically_valid", default=False
    ) or not _flag(control_gate, "resource_valid", default=False):
        decision = BoundaryTangentDecision.BOUNDARY_TANGENT_CONTROLLER_NUMERICALLY_INVALID.value
        action = _ACTIONS[decision]
    elif not _flag(control_gate, "weak_law_controlled"):
        decision = BoundaryTangentDecision.REVERSE_CONTROLLER_WEAK_LAW_FAILED.value
        action = _ACTIONS[decision]
    elif not _flag(control_gate, "microstep_refinement_controlled"):
        decision = BoundaryTangentDecision.REVERSE_CONTROLLER_MICROSTEP_REFINEMENT_FAILED.value
        action = _ACTIONS[decision]
    elif not _passed(control_gate):
        decision = BoundaryTangentDecision.BOUNDARY_TANGENT_CONTROLLER_NUMERICALLY_INVALID.value
        action = _ACTIONS[decision]
    else:
        decision = BoundaryTangentDecision.EXACT_RB_BOUNDARY_TANGENT_CONTROLLER_CONTROLLED.value
        action = _ACTIONS[decision]
    controlled = (
        decision
        == BoundaryTangentDecision.EXACT_RB_BOUNDARY_TANGENT_CONTROLLER_CONTROLLED.value
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "decision": decision,
        "recommended_next_action": action,
        **claim_scope_flags(
            controlled=controlled,
            physical_training_performed=(
                _status(train_gate) == "evaluated"
                and _flag(train_gate, "physical_training_performed", default=True)
            ),
            controller_control_trajectory_performed=(
                _status(control_gate) == "evaluated"
                and _flag(
                    control_gate,
                    "controller_control_trajectory_performed",
                    default=True,
                )
            ),
        ),
    }


__all__ = [
    "BoundaryTangentDecision",
    "BoundaryTangentGateError",
    "BoundaryTangentThresholds",
    "CLAIM_SCOPE",
    "COMBINED_VS_BASELINE_FAMILY_SIZE",
    "COMBINED_VS_BASELINE_NAMES",
    "COMBINED_VS_ZERO_FAMILY_SIZE",
    "COMBINED_VS_ZERO_NAMES",
    "CONFIRMATION_FAMILY_NAMES",
    "CONFIRMATION_FAMILY_SIZE",
    "CONTROLLER_FAMILY_SIZE",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONTROLLER_BOOTSTRAP_SEED",
    "DEFAULT_ROOT_SEED",
    "DEFAULT_SIMULTANEOUS_CONFIDENCE",
    "NO_SAMPLING_OR_RECONSTRUCTION",
    "claim_scope_flags",
    "confirmation_family_names",
    "decide_boundary_tangent_workflow",
    "evaluate_confirmation_gate",
    "evaluate_controller_gate",
    "evaluate_numerical_resource_gate",
    "not_evaluated_gate",
    "one_sided_whole_path_max_t",
    "validate_confirmation_family",
    "validate_controller_family",
]
