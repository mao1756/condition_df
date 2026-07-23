"""Fail-closed gates for certified Rao--Blackwellized Jacobi denoising.

The gates in this module are deliberately pure.  They do not evaluate a
spectral series, touch the filesystem, train a model, or sample a path.  They
only adjudicate records produced by the additive exact-spectral workflow.

There are two distinctions worth preserving in the report layer:

* a bad or incomplete numerical certificate is a numerical failure;
* a completely certified implementation which misses the frozen wall-clock
  or memory budget is a resource failure.

Neither branch authorises an approximate Gaussian/Euler transition.  A
``not_evaluated`` gate is also always failing, so skipped work cannot acquire
success through empty ``all(...)`` semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


SCHEMA = "experiment12-d0-jacobi-rao-blackwell-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class JacobiRBThresholds:
    """Frozen numerical and resource contract for the production gate."""

    parent_record_count: int = 16
    projected_transition_count: int = 89_915_392
    full_path_transition_count: int = 1_404_928
    minimum_full_path_benchmark_repeats: int = 3
    minimum_slowest_transitions_per_second: float = 1_300.0
    maximum_projected_cache_hours: float = 20.0
    maximum_peak_memory_fraction: float = 0.80
    maximum_arb_fallback_fraction: float = 1.0e-4
    maximum_arb_cost_fraction: float = 0.10
    maximum_float64_mass_error: float = 1.0e-12
    maximum_cuda_mass_error: float = 2.0e-6
    maximum_float64_kernel_error: float = 1.0e-9
    maximum_cuda_kernel_error: float = 2.0e-6
    maximum_rb_identity_relative_error: float = 1.0e-8
    maximum_legacy_mixture_error: float = 1.0e-8
    maximum_cuda_rb_relative_error: float = 2.0e-5
    minimum_certificate_fraction: float = 1.0

    def __post_init__(self) -> None:
        frozen_ints = {
            "parent_record_count": (self.parent_record_count, 16),
            "projected_transition_count": (
                self.projected_transition_count,
                89_915_392,
            ),
            "full_path_transition_count": (
                self.full_path_transition_count,
                1_404_928,
            ),
            "minimum_full_path_benchmark_repeats": (
                self.minimum_full_path_benchmark_repeats,
                3,
            ),
        }
        for name, (actual, expected) in frozen_ints.items():
            if int(actual) != expected:
                raise ValueError(f"{name} is frozen at {expected}")
        frozen_floats = {
            "minimum_slowest_transitions_per_second": (
                self.minimum_slowest_transitions_per_second,
                1_300.0,
            ),
            "maximum_projected_cache_hours": (
                self.maximum_projected_cache_hours,
                20.0,
            ),
            "maximum_peak_memory_fraction": (
                self.maximum_peak_memory_fraction,
                0.80,
            ),
            "maximum_arb_fallback_fraction": (
                self.maximum_arb_fallback_fraction,
                1.0e-4,
            ),
            "maximum_arb_cost_fraction": (
                self.maximum_arb_cost_fraction,
                0.10,
            ),
            "maximum_float64_mass_error": (
                self.maximum_float64_mass_error,
                1.0e-12,
            ),
            "maximum_cuda_mass_error": (self.maximum_cuda_mass_error, 2.0e-6),
            "maximum_float64_kernel_error": (
                self.maximum_float64_kernel_error,
                1.0e-9,
            ),
            "maximum_cuda_kernel_error": (
                self.maximum_cuda_kernel_error,
                2.0e-6,
            ),
            "maximum_rb_identity_relative_error": (
                self.maximum_rb_identity_relative_error,
                1.0e-8,
            ),
            "maximum_cuda_rb_relative_error": (
                self.maximum_cuda_rb_relative_error,
                2.0e-5,
            ),
            "maximum_legacy_mixture_error": (
                self.maximum_legacy_mixture_error,
                1.0e-8,
            ),
            "minimum_certificate_fraction": (
                self.minimum_certificate_fraction,
                1.0,
            ),
        }
        for name, (actual, expected) in frozen_floats.items():
            if not math.isfinite(float(actual)) or float(actual) != expected:
                raise ValueError(f"{name} is frozen at {expected}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JacobiRBDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    CERTIFIED_BACKEND_UNAVAILABLE = "certified_backend_unavailable"
    RAO_BLACKWELL_IDENTITY_INVALID = "rao_blackwell_identity_invalid"
    SPECTRAL_CDF_ALGEBRA_INVALID = "spectral_cdf_algebra_invalid"
    SPECTRAL_INTERVAL_BACKEND_INVALID = "spectral_interval_backend_invalid"
    SPECTRAL_INVERSION_NUMERICALLY_UNRESOLVED = (
        "spectral_inversion_numerically_unresolved"
    )
    SPECTRAL_INVERSION_COMPUTATIONALLY_INFEASIBLE = (
        "spectral_inversion_computationally_infeasible"
    )
    JACOBI_RB_TARGET_INVALID = "jacobi_rb_target_invalid"
    EXACT_JACOBI_RB_KERNEL_FEASIBLE = "exact_jacobi_rb_kernel_feasible"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _status(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return "not_evaluated"
    raw = str(value.get("evaluation_status", "not_evaluated")).lower()
    return {
        "complete": "evaluated",
        "completed": "evaluated",
        "pending": "not_evaluated",
        "skipped": "not_evaluated",
        "incomplete": "not_evaluated",
    }.get(raw, raw)


def _passed(value: bool | int | Mapping[str, Any] | None) -> bool:
    if isinstance(value, Mapping):
        return _status(value) == "evaluated" and _one(
            value.get("passed", value.get("gate_pass", 0))
        )
    return value is True or (isinstance(value, int) and value == 1)


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
            and all(_one(item.get("passed", 0)) for item in normalized.values())
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    gate = _gate(name, "not evaluated", {}, evaluation_status="not_evaluated")
    gate["reason"] = str(reason)
    return gate


def evaluate_jacobi_rb_preflight(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBThresholds | None = None,
) -> dict[str, Any]:
    """Gate parent binding, interval arithmetic, and operator conventions."""

    t = thresholds or JacobiRBThresholds()
    checks = {
        "parent_provenance_pass": _eq_one(metrics, "parent_provenance_pass"),
        "parent_record_count": _check(
            metrics.get("parent_record_count"),
            "==",
            t.parent_record_count,
            _finite(metrics.get("parent_record_count"))
            and int(metrics["parent_record_count"]) == t.parent_record_count,
        ),
        "parent_reclassification_pass": _eq_one(
            metrics, "parent_reclassification_pass"
        ),
        "arb_backend_available": _eq_one(metrics, "arb_backend_available"),
        "python_flint_exact_version_pass": _eq_one(
            metrics, "python_flint_exact_version_pass"
        ),
        "arb_outward_rounding_pass": _eq_one(
            metrics, "arb_outward_rounding_pass"
        ),
        "gpu_interval_enclosure_pass": _eq_one(
            metrics, "gpu_interval_enclosure_pass"
        ),
        "alpha1_legendre_formula_pass": _eq_one(
            metrics, "alpha1_legendre_formula_pass"
        ),
        "jacobi_wf_clock_factor_pass": _eq_one(
            metrics, "jacobi_wf_clock_factor_pass"
        ),
        "head_fraction_orientation_pass": _eq_one(
            metrics, "head_fraction_orientation_pass"
        ),
        "stable_conormal_formula_pass": _eq_one(
            metrics, "stable_conormal_formula_pass"
        ),
        "lazy_dyadic_uniform_pass": _eq_one(
            metrics, "lazy_dyadic_uniform_pass"
        ),
        "rounding_cell_contract_pass": _eq_one(
            metrics, "rounding_cell_contract_pass"
        ),
        "cantelli_bracket_pass": _eq_one(metrics, "cantelli_bracket_pass"),
        "forbidden_approximation_count": _eq_zero(
            metrics, "forbidden_approximation_count"
        ),
        "nonfinite_count": _eq_zero(metrics, "nonfinite_count"),
    }
    result = _gate(
        "jacobi_rb_preflight",
        "immutable parent, exact alpha=1 operator, and certified arithmetic backend",
        checks,
    )
    result["thresholds"] = t.to_dict()
    result["backend_valid"] = int(
        checks["arb_backend_available"]["passed"]
        and checks["python_flint_exact_version_pass"]["passed"]
        and checks["arb_outward_rounding_pass"]["passed"]
        and checks["gpu_interval_enclosure_pass"]["passed"]
    )
    return result


_KERNEL_NUMERICAL_CHECKS = frozenset(
    {
        "adversarial_support_pass",
        "cdf_endpoint_certificate_pass",
        "spectral_tail_enclosure_pass",
        "roundoff_enclosure_pass",
        "law_control_pass",
        "moment_control_pass",
        "eigenmoment_control_pass",
        "stationarity_control_pass",
        "reversibility_control_pass",
        "support_case_count_pass",
        "cdf_monotonicity_pass",
        "normalization_pass",
        "semigroup_pass",
        "detailed_balance_pass",
        "precision_doubling_hash_pass",
        "float64_kernel_max_error",
        "cuda_kernel_max_error",
        "cuda_evaluated_pass",
        "quantile_certificate_fraction",
        "uncertified_draw_count",
        "resource_cap_count",
        "approximation_count",
        "gaussian_fallback_count",
        "euler_fallback_count",
        "finite_ancestral_proxy_count",
        "exposure_binning_count",
        "replay_y_bit_mismatch_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "renormalization_count",
        "negative_state_count",
        "nonfinite_count",
        "float64_pair_mass_error",
        "float64_simplex_error",
        "cuda_pair_mass_error",
        "cuda_simplex_error",
    }
)

_KERNEL_RESOURCE_CHECKS = frozenset(
    {
        "full_path_transition_count",
        "full_path_benchmark_repeats",
        "slowest_transitions_per_second",
        "projected_transition_count",
        "projected_cache_hours",
        "peak_memory_fraction",
        "arb_fallback_fraction",
        "arb_cost_fraction",
        "benchmark_output_hash_pass",
        "full_api_completed_pass",
    }
)


def evaluate_jacobi_rb_kernel(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBThresholds | None = None,
) -> dict[str, Any]:
    """Gate every-draw certification and end-to-end production throughput."""

    t = thresholds or JacobiRBThresholds()
    checks: dict[str, dict[str, Any]] = {
        name: _eq_one(metrics, name)
        for name in (
            "adversarial_support_pass",
            "support_case_count_pass",
            "cdf_endpoint_certificate_pass",
            "cdf_monotonicity_pass",
            "spectral_tail_enclosure_pass",
            "roundoff_enclosure_pass",
            "normalization_pass",
            "semigroup_pass",
            "detailed_balance_pass",
            "law_control_pass",
            "moment_control_pass",
            "eigenmoment_control_pass",
            "stationarity_control_pass",
            "reversibility_control_pass",
            "precision_doubling_hash_pass",
            "benchmark_output_hash_pass",
            "full_api_completed_pass",
            "cuda_evaluated_pass",
        )
    }
    checks.update(
        {
            "quantile_certificate_fraction": _check(
                metrics.get("quantile_certificate_fraction"),
                "==",
                t.minimum_certificate_fraction,
                _finite(metrics.get("quantile_certificate_fraction"))
                and float(metrics["quantile_certificate_fraction"])
                == t.minimum_certificate_fraction,
            ),
            **{
                name: _eq_zero(metrics, name)
                for name in (
                    "uncertified_draw_count",
                    "resource_cap_count",
                    "approximation_count",
                    "gaussian_fallback_count",
                    "euler_fallback_count",
                    "finite_ancestral_proxy_count",
                    "exposure_binning_count",
                    "replay_y_bit_mismatch_count",
                    "correction_count",
                    "floor_count",
                    "limiter_count",
                    "renormalization_count",
                    "negative_state_count",
                    "nonfinite_count",
                )
            },
            "float64_pair_mass_error": _le(
                metrics, "float64_pair_mass_error", t.maximum_float64_mass_error
            ),
            "float64_simplex_error": _le(
                metrics, "float64_simplex_error", t.maximum_float64_mass_error
            ),
            "cuda_pair_mass_error": _le(
                metrics, "cuda_pair_mass_error", t.maximum_cuda_mass_error
            ),
            "cuda_simplex_error": _le(
                metrics, "cuda_simplex_error", t.maximum_cuda_mass_error
            ),
            "float64_kernel_max_error": _le(
                metrics,
                "float64_kernel_max_error",
                t.maximum_float64_kernel_error,
            ),
            "cuda_kernel_max_error": _le(
                metrics,
                "cuda_kernel_max_error",
                t.maximum_cuda_kernel_error,
            ),
            "full_path_transition_count": _ge(
                metrics,
                "full_path_transition_count",
                float(t.full_path_transition_count),
            ),
            "full_path_benchmark_repeats": _ge(
                metrics,
                "full_path_benchmark_repeats",
                float(t.minimum_full_path_benchmark_repeats),
            ),
            "slowest_transitions_per_second": _ge(
                metrics,
                "slowest_transitions_per_second",
                t.minimum_slowest_transitions_per_second,
            ),
            "projected_transition_count": _check(
                metrics.get("projected_transition_count"),
                "==",
                t.projected_transition_count,
                _finite(metrics.get("projected_transition_count"))
                and int(metrics["projected_transition_count"])
                == t.projected_transition_count,
            ),
            "projected_cache_hours": _le(
                metrics,
                "projected_cache_hours",
                t.maximum_projected_cache_hours,
            ),
            "peak_memory_fraction": _le(
                metrics,
                "peak_memory_fraction",
                t.maximum_peak_memory_fraction,
            ),
            "arb_fallback_fraction": _le(
                metrics,
                "arb_fallback_fraction",
                t.maximum_arb_fallback_fraction,
            ),
            "arb_cost_fraction": _le(
                metrics,
                "arb_cost_fraction",
                t.maximum_arb_cost_fraction,
            ),
        }
    )
    result = _gate(
        "jacobi_rb_kernel",
        "correctly rounded exact spectral transition on the frozen grid-28 support",
        checks,
    )
    result["thresholds"] = t.to_dict()
    result["numerically_valid"] = int(
        all(checks[name]["passed"] for name in _KERNEL_NUMERICAL_CHECKS)
    )
    result["resource_valid"] = int(
        all(checks[name]["passed"] for name in _KERNEL_RESOURCE_CHECKS)
    )
    return result


def evaluate_jacobi_rb_target(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBThresholds | None = None,
) -> dict[str, Any]:
    """Gate the exact Rao--Blackwell target and no-leakage model contract."""

    t = thresholds or JacobiRBThresholds()
    checks: dict[str, dict[str, Any]] = {
        name: _eq_one(metrics, name)
        for name in (
            "rao_blackwell_identity_pass",
            "population_tower_identity_pass",
            "latent_mixture_equivalence_pass",
            "density_positive_certificate_pass",
            "target_unique_rounding_pass",
            "conormal_orientation_pass",
            "synthetic_teacher_pass",
            "stationary_null_pass",
            "all_phase_colors_pass",
            "half_full_duration_pass",
            "negative_fixtures_pass",
            "later_state_only_input_pass",
            "cuda_target_evaluated_pass",
        )
    }
    checks.update(
        {
            "target_certificate_fraction": _check(
                metrics.get("target_certificate_fraction"),
                "==",
                t.minimum_certificate_fraction,
                _finite(metrics.get("target_certificate_fraction"))
                and float(metrics["target_certificate_fraction"])
                == t.minimum_certificate_fraction,
            ),
            "rb_identity_relative_error": _le(
                metrics,
                "rb_identity_relative_error",
                t.maximum_rb_identity_relative_error,
            ),
            "cuda_rb_relative_error": _le(
                metrics,
                "cuda_rb_relative_error",
                t.maximum_cuda_rb_relative_error,
            ),
            "legacy_mixture_max_absolute_error": _le(
                metrics,
                "legacy_mixture_max_absolute_error",
                t.maximum_legacy_mixture_error,
            ),
            **{
                name: _eq_zero(metrics, name)
                for name in (
                    "target_uncertified_count",
                    "target_resource_cap_count",
                    "target_replay_bit_mismatch_count",
                    "target_nonfinite_count",
                    "earlier_state_input_count",
                    "latent_variable_input_count",
                    "classifier_target_count",
                    "value_target_count",
                    "h1_target_count",
                    "raw_euler_residual_target_count",
                    "gaussian_target_count",
                    "target_clip_count",
                )
            },
        }
    )
    result = _gate(
        "jacobi_rb_target",
        "exact DDPM-like Rao--Blackwellized Jacobi conormal-score target",
        checks,
    )
    result["thresholds"] = t.to_dict()
    return result


def _failed_names(gate: Mapping[str, Any]) -> set[str]:
    checks = gate.get("subchecks", {})
    if not isinstance(checks, Mapping):
        return set()
    return {
        str(name)
        for name, value in checks.items()
        if not isinstance(value, Mapping) or not _one(value.get("passed", 0))
    }


def decide_jacobi_rb_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    kernel_gate: Mapping[str, Any] | None,
    target_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return one closed outcome without treating skipped gates as passes."""

    if not _passed(provenance):
        decision = JacobiRBDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 16-record parent binding"
    elif _status(preflight_gate) != "evaluated":
        decision = JacobiRBDecision.SPECTRAL_CDF_ALGEBRA_INVALID
        action = "complete the exact spectral/backend preflight"
    elif not _passed(preflight_gate):
        failed = _failed_names(dict(preflight_gate or {}))
        if failed & {"arb_backend_available", "python_flint_exact_version_pass"}:
            decision = JacobiRBDecision.CERTIFIED_BACKEND_UNAVAILABLE
            action = "install and pin the required certified Arb backend"
        elif failed & {
            "arb_outward_rounding_pass",
            "gpu_interval_enclosure_pass",
            "rounding_cell_contract_pass",
        }:
            decision = JacobiRBDecision.SPECTRAL_INTERVAL_BACKEND_INVALID
            action = "repair outward-rounded GPU/Arb certification"
        else:
            decision = JacobiRBDecision.SPECTRAL_CDF_ALGEBRA_INVALID
            action = "repair the Jacobi clock, orientation, or inverse-CDF contract"
    elif _status(kernel_gate) != "evaluated":
        decision = JacobiRBDecision.SPECTRAL_INVERSION_NUMERICALLY_UNRESOLVED
        action = "complete the certified spectral inversion and full-path benchmark"
    elif not _passed(kernel_gate):
        numerical = _one(dict(kernel_gate or {}).get("numerically_valid", 0))
        resource = _one(dict(kernel_gate or {}).get("resource_valid", 0))
        if numerical and not resource:
            decision = (
                JacobiRBDecision.SPECTRAL_INVERSION_COMPUTATIONALLY_INFEASIBLE
            )
            action = "optimize the exact spectral implementation without changing its law"
        else:
            decision = JacobiRBDecision.SPECTRAL_INVERSION_NUMERICALLY_UNRESOLVED
            action = "repair per-draw certificates; do not emit an approximate transition"
    elif _status(target_gate) != "evaluated":
        decision = JacobiRBDecision.JACOBI_RB_TARGET_INVALID
        action = "complete exact target and no-input-leakage controls"
    elif not _passed(target_gate):
        identity_checks = {
            "rao_blackwell_identity_pass",
            "population_tower_identity_pass",
            "latent_mixture_equivalence_pass",
            "rb_identity_relative_error",
            "cuda_rb_relative_error",
            "legacy_mixture_max_absolute_error",
        }
        failed = _failed_names(target_gate)
        if failed and failed.issubset(identity_checks):
            decision = JacobiRBDecision.RAO_BLACKWELL_IDENTITY_INVALID
            action = "repair the exact Fisher/tower identity; do not change the target"
        else:
            decision = JacobiRBDecision.JACOBI_RB_TARGET_INVALID
            action = "repair target certification or the later-state-only contract"
    else:
        decision = JacobiRBDecision.EXACT_JACOBI_RB_KERNEL_FEASIBLE
        action = "run the separate actual Eulerian Strang-refinement gate"

    value = decision.value
    ready = value == JacobiRBDecision.EXACT_JACOBI_RB_KERNEL_FEASIBLE.value
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "decision": value,
        "closed_terminal_scientific_outcome": 1,
        "recommended_next_action": action,
        "strang_refinement_authorized": int(ready),
        "one_image_training_authorized": 0,
        "physical_training_authorized": 0,
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


def evaluate_jacobi_rb_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any],
    kernel_gate: Mapping[str, Any],
    target_gate: Mapping[str, Any],
    require_gate: str = "none",
    thresholds: JacobiRBThresholds | None = None,
) -> dict[str, Any]:
    """Compose the three gates and compute the requested fail-closed result."""

    t = thresholds or JacobiRBThresholds()
    if require_gate not in {"none", "preflight", "kernel", "target"}:
        raise ValueError("require_gate must be none, preflight, kernel, or target")
    required_pass = {
        "none": True,
        "preflight": _passed(provenance) and _passed(preflight_gate),
        "kernel": (
            _passed(provenance)
            and _passed(preflight_gate)
            and _passed(kernel_gate)
        ),
        "target": (
            _passed(provenance)
            and _passed(preflight_gate)
            and _passed(kernel_gate)
            and _passed(target_gate)
        ),
    }[require_gate]
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "components": {
            "provenance": (
                dict(provenance)
                if isinstance(provenance, Mapping)
                else int(_passed(provenance))
            ),
            "preflight": dict(preflight_gate),
            "kernel": dict(kernel_gate),
            "target": dict(target_gate),
        },
        "decision": decide_jacobi_rb_workflow(
            provenance=provenance,
            preflight_gate=preflight_gate,
            kernel_gate=kernel_gate,
            target_gate=target_gate,
        ),
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "thresholds": t.to_dict(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "JacobiRBDecision",
    "JacobiRBThresholds",
    "not_evaluated_gate",
    "evaluate_jacobi_rb_preflight",
    "evaluate_jacobi_rb_kernel",
    "evaluate_jacobi_rb_target",
    "decide_jacobi_rb_workflow",
    "evaluate_jacobi_rb_workflow",
]
