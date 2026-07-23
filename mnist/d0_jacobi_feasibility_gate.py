"""Fail-closed gates for the exact Jacobi denoising feasibility workflow.

The workflow deliberately separates mathematical validity from numerical and
computational feasibility.  In particular, a numerically uncertified
Wright--Fisher draw is never reclassified as an approximate success.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


GATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class JacobiFeasibilityThresholds:
    float64_relative_error: float = 1.0e-9
    cuda_relative_error: float = 2.0e-6
    float64_score_error: float = 1.0e-8
    cuda_score_error: float = 2.0e-5
    float64_mass_error: float = 1.0e-12
    cuda_mass_error: float = 2.0e-6
    minimum_weak_order: float = 1.8
    k512_k1024_relative_error: float = 5.0e-3
    k512_generator_relative_error: float = 1.0e-2
    maximum_projected_cache_hours: float = 24.0
    maximum_memory_fraction: float = 0.80
    deterministic_identity_error: float = 1.0e-8
    edge_generator_observable_error: float = 1.0e-8

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _gate(name: str, claim_scope: str, checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in checks.items()}
    return {
        "schema": "d0-jacobi-feasibility-gate",
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": name,
        "claim_scope": claim_scope,
        "evaluation_status": "evaluated",
        "subchecks": normalized,
        "passed": int(bool(normalized) and all(int(item.get("passed", 0)) == 1 for item in normalized.values())),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema": "d0-jacobi-feasibility-gate",
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": str(name),
        "claim_scope": "not evaluated",
        "evaluation_status": "not_evaluated",
        "reason": str(reason),
        "subchecks": {},
        "passed": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_jacobi_preflight(
    *,
    parent_registry_valid: bool,
    parent_record_count: int,
    parent_registry_sha256: str,
    expected_registry_sha256: str,
    parent_readjudication_valid: bool,
    matching_partition_valid: bool,
    matching_disjoint: bool,
    strang_palindromic: bool,
    convention_valid: bool,
) -> dict[str, Any]:
    checks = {
        "parent_registry_valid": _check(int(parent_registry_valid), "==", 1, parent_registry_valid),
        "parent_record_count": _check(int(parent_record_count), "==", 277, int(parent_record_count) == 277),
        "parent_registry_sha256": _check(
            str(parent_registry_sha256), "==", str(expected_registry_sha256),
            str(parent_registry_sha256) == str(expected_registry_sha256),
        ),
        "parent_readjudication_valid": _check(
            int(parent_readjudication_valid), "==", 1, parent_readjudication_valid
        ),
        "matching_partition_valid": _check(
            int(matching_partition_valid), "==", 1, matching_partition_valid
        ),
        "matching_disjoint": _check(int(matching_disjoint), "==", 1, matching_disjoint),
        "strang_palindromic": _check(int(strang_palindromic), "==", 1, strang_palindromic),
        "kernel_convention_valid": _check(int(convention_valid), "==", 1, convention_valid),
    }
    return _gate(
        "jacobi_preflight",
        "immutable parent provenance and fixed-grid Jacobi operator algebra",
        checks,
    )


def evaluate_jacobi_kernel(
    metrics: Mapping[str, Any],
    thresholds: JacobiFeasibilityThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or JacobiFeasibilityThresholds()

    def le(name: str, limit: float) -> dict[str, Any]:
        value = metrics.get(name)
        return _check(value, "<=", limit, _finite(value) and float(value) <= limit)

    checks = {
        "normalization_relative_error": le("normalization_relative_error", t.float64_relative_error),
        "cdf_endpoint_error": le("cdf_endpoint_error", t.float64_relative_error),
        "detailed_balance_relative_error": le("detailed_balance_relative_error", t.float64_relative_error),
        "semigroup_relative_error": le("semigroup_relative_error", t.float64_relative_error),
        "eigenmoment_relative_error": le("eigenmoment_relative_error", t.float64_relative_error),
        "arrival_score_relative_error": le("arrival_score_relative_error", t.float64_score_error),
        "cuda_relative_error": le("cuda_relative_error", t.cuda_relative_error),
        "cuda_score_relative_error": le("cuda_score_relative_error", t.cuda_score_error),
        "production_spectral_support_pass": _check(
            int(bool(metrics.get("production_spectral_support_pass", False))), "==", 1,
            bool(metrics.get("production_spectral_support_pass", False)),
        ),
        "uncertified_draw_count": _check(
            metrics.get("uncertified_draw_count"), "==", 0,
            int(metrics.get("uncertified_draw_count", -1)) == 0,
        ),
        "numerical_certification_failure_count": _check(
            metrics.get("numerical_certification_failure_count"), "==", 0,
            int(metrics.get("numerical_certification_failure_count", -1)) == 0,
        ),
        "resource_cap_count": _check(
            metrics.get("resource_cap_count"), "==", 0,
            int(metrics.get("resource_cap_count", -1)) == 0,
        ),
        "negative_density_count": _check(
            metrics.get("negative_density_count"), "==", 0,
            int(metrics.get("negative_density_count", -1)) == 0,
        ),
        "nonfinite_count": _check(
            metrics.get("nonfinite_count"), "==", 0,
            int(metrics.get("nonfinite_count", -1)) == 0,
        ),
        "correction_count": _check(
            metrics.get("correction_count"), "==", 0,
            int(metrics.get("correction_count", -1)) == 0,
        ),
        "floor_count": _check(
            metrics.get("floor_count"), "==", 0,
            int(metrics.get("floor_count", -1)) == 0,
        ),
        "limiter_count": _check(
            metrics.get("limiter_count"), "==", 0,
            int(metrics.get("limiter_count", -1)) == 0,
        ),
        "renormalization_count": _check(
            metrics.get("renormalization_count"), "==", 0,
            int(metrics.get("renormalization_count", -1)) == 0,
        ),
        "distribution_control_pass": _check(
            int(bool(metrics.get("distribution_control_pass", False))), "==", 1,
            bool(metrics.get("distribution_control_pass", False)),
        ),
        "projected_cache_hours": le(
            "projected_cache_hours", t.maximum_projected_cache_hours
        ),
        "peak_memory_fraction": le("peak_memory_fraction", t.maximum_memory_fraction),
    }
    gate = _gate(
        "jacobi_kernel",
        "certified exact two-cell transition on the frozen production support",
        checks,
    )
    gate["thresholds"] = t.to_dict()
    gate["numerically_valid"] = int(all(
        checks[key]["passed"] for key in checks
        if key not in {
            "uncertified_draw_count", "resource_cap_count",
            "projected_cache_hours", "peak_memory_fraction"
        }
    ))
    gate["computationally_feasible"] = int(
        checks["resource_cap_count"]["passed"]
        and checks["projected_cache_hours"]["passed"]
        and checks["peak_memory_fraction"]["passed"]
    )
    return gate


def evaluate_jacobi_controls(
    metrics: Mapping[str, Any],
    thresholds: JacobiFeasibilityThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or JacobiFeasibilityThresholds()

    def le(name: str, limit: float) -> dict[str, Any]:
        value = metrics.get(name)
        return _check(value, "<=", limit, _finite(value) and float(value) <= limit)

    order = metrics.get("observed_weak_order")
    checks = {
        "float64_pair_mass_error": le("float64_pair_mass_error", t.float64_mass_error),
        "float64_simplex_error": le("float64_simplex_error", t.float64_mass_error),
        "cuda_pair_mass_error": le("cuda_pair_mass_error", t.cuda_mass_error),
        "cuda_simplex_error": le("cuda_simplex_error", t.cuda_mass_error),
        "dirichlet_stationarity_pass": _check(
            int(bool(metrics.get("dirichlet_stationarity_pass", False))), "==", 1,
            bool(metrics.get("dirichlet_stationarity_pass", False)),
        ),
        "full_sweep_detailed_balance_pass": _check(
            int(bool(metrics.get("full_sweep_detailed_balance_pass", False))), "==", 1,
            bool(metrics.get("full_sweep_detailed_balance_pass", False)),
        ),
        "split_reference_evaluated": _check(
            int(bool(metrics.get("split_reference_evaluated", False))), "==", 1,
            bool(metrics.get("split_reference_evaluated", False)),
        ),
        "split_fixture": _check(
            metrics.get("split_fixture"), "==", "exact-state-dependent-jacobi-grid28",
            metrics.get("split_fixture") == "exact-state-dependent-jacobi-grid28",
        ),
        "observed_weak_order": _check(
            order, ">=", t.minimum_weak_order,
            _finite(order) and float(order) >= t.minimum_weak_order,
        ),
        "actual_eulerian_refinement_pass": _check(
            int(bool(metrics.get("actual_eulerian_refinement_pass", False))), "==", 1,
            bool(metrics.get("actual_eulerian_refinement_pass", False)),
        ),
        "edge_generator_observable_error": le(
            "edge_generator_observable_error", t.edge_generator_observable_error
        ),
        "k512_k1024_relative_error": le(
            "k512_k1024_relative_error", t.k512_k1024_relative_error
        ),
        "k512_generator_relative_error": le(
            "k512_generator_relative_error", t.k512_generator_relative_error
        ),
        "deterministic_identity_error": le(
            "deterministic_identity_error", t.deterministic_identity_error
        ),
        "monte_carlo_identity_pass": _check(
            int(bool(metrics.get("monte_carlo_identity_pass", False))), "==", 1,
            bool(metrics.get("monte_carlo_identity_pass", False)),
        ),
        "stationary_null_pass": _check(
            int(bool(metrics.get("stationary_null_pass", False))), "==", 1,
            bool(metrics.get("stationary_null_pass", False)),
        ),
        "orientation_fixtures_pass": _check(
            int(bool(metrics.get("orientation_fixtures_pass", False))), "==", 1,
            bool(metrics.get("orientation_fixtures_pass", False)),
        ),
        "intervention_count": _check(
            metrics.get("intervention_count"), "==", 0,
            int(metrics.get("intervention_count", -1)) == 0,
        ),
        "correction_count": _check(
            metrics.get("correction_count"), "==", 0,
            int(metrics.get("correction_count", -1)) == 0,
        ),
        "floor_count": _check(
            metrics.get("floor_count"), "==", 0,
            int(metrics.get("floor_count", -1)) == 0,
        ),
        "limiter_count": _check(
            metrics.get("limiter_count"), "==", 0,
            int(metrics.get("limiter_count", -1)) == 0,
        ),
        "renormalization_count": _check(
            metrics.get("renormalization_count"), "==", 0,
            int(metrics.get("renormalization_count", -1)) == 0,
        ),
        "nonfinite_count": _check(
            metrics.get("nonfinite_count"), "==", 0,
            int(metrics.get("nonfinite_count", -1)) == 0,
        ),
    }
    gate = _gate(
        "jacobi_controls",
        "symmetric split reference and exact Jacobi denoising identity",
        checks,
    )
    gate["thresholds"] = t.to_dict()
    return gate


def decide_jacobi_feasibility(
    *,
    provenance_valid: bool,
    adjudication_valid: bool,
    preflight_gate: Mapping[str, Any],
    kernel_gate: Mapping[str, Any] | None,
    controls_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not provenance_valid:
        decision = "control_provenance_invalid"
        action = "repair immutable parent registry or lineage binding"
    elif not adjudication_valid:
        decision = "parent_adjudication_invalid"
        action = "repair report-only parent evidence reconstruction"
    elif int(preflight_gate.get("passed", 0)) != 1:
        decision = "jacobi_kernel_algebra_invalid"
        action = "repair edge orientation, matching partition, or time convention"
    elif kernel_gate is None or kernel_gate.get("evaluation_status") != "evaluated":
        decision = "jacobi_kernel_numerically_unresolved"
        action = "complete the certified exact-kernel study"
    elif int(kernel_gate.get("numerically_valid", 0)) != 1:
        decision = "jacobi_kernel_numerically_unresolved"
        action = "repair the certified kernel; do not substitute an approximate transition"
    elif int(kernel_gate.get("computationally_feasible", 0)) != 1:
        decision = "jacobi_kernel_computationally_infeasible"
        action = "optimize the exact transition implementation without changing the target law"
    elif controls_gate is None or controls_gate.get("evaluation_status") != "evaluated":
        decision = "jacobi_split_reference_invalid"
        action = "complete split-reference and denoising-identity controls"
    elif int(controls_gate.get("passed", 0)) != 1:
        identity_names = {
            "deterministic_identity_error", "monte_carlo_identity_pass",
            "stationary_null_pass", "orientation_fixtures_pass",
        }
        failed = {
            name for name, value in dict(controls_gate.get("subchecks", {})).items()
            if int(value.get("passed", 0)) != 1
        }
        if failed and failed.issubset(identity_names):
            decision = "jacobi_denoising_identity_invalid"
            action = "repair the exact denoising label or invariant-measure convention"
        else:
            decision = "jacobi_split_reference_invalid"
            action = "repair conservation, stationarity, or Strang refinement"
    else:
        decision = "exact_jacobi_denoising_feasible"
        action = "plan a separate phase-conditioned one-image Jacobi denoising gate"
    return {
        "schema": "d0-jacobi-feasibility-decision",
        "schema_version": GATE_SCHEMA_VERSION,
        "decision": decision,
        "closed_terminal_scientific_outcome": 1,
        "recommended_next_action": action,
        "one_image_training_authorized": int(decision == "exact_jacobi_denoising_feasible"),
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }
