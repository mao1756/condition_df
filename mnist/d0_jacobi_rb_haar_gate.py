"""Pure fail-closed gates for certified Haar-coupled Jacobi refinement power.

This module has no sampler or trainer dependency.  It centralizes frozen
thresholds, raw endpoint/Richardson statistics, profile nomination, sealed
panel semantics, and the closed decision ladder used by the controls-only
workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "experiment12-d0-jacobi-rb-haar-coupling-gate"
SCHEMA_VERSION = 1
NESTED_HAAR_PROFILE = "nested_haar_single_arm"
ANTITHETIC_HAAR_PROFILE = "pairwise_haar_antithetic"
PROFILE_ORDER = (NESTED_HAAR_PROFILE, ANTITHETIC_HAAR_PROFILE)
STAGES = ("preflight", "coupling", "pilot", "report", "all")
REQUIRED_GATES = ("none", "preflight", "coupling", "pilot")
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


@dataclass(frozen=True)
class HaarCouplingThresholds:
    """Frozen production thresholds and profile definitions."""

    parent_record_count: int = 1061
    parent_source_count: int = 26
    root_seed: int = 261_181
    grid_size: int = 28
    alpha: float = 1.0
    tau_eff: float = 5.0e-5
    levels: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    observation_time_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    panel_clusters: int = 8
    nested_main_paths: tuple[int, ...] = (32, 64)
    nested_reference_paths: tuple[int, ...] = (16, 32)
    antithetic_main_paths: int = 16
    antithetic_reference_paths: int = 16
    maximum_main_half_width: float = 0.0025
    maximum_reference_half_width: float = 0.005
    maximum_projected_hours: float = 48.0
    minimum_rate: float = 1_300.0
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_cost_fraction: float = 0.10
    maximum_peak_memory_fraction: float = 0.80
    maximum_cuda_mass_error: float = 2.0e-6
    maximum_prefix_bits: int = 1024

    def __post_init__(self) -> None:
        frozen = {
            "parent_record_count": 1061,
            "parent_source_count": 26,
            "root_seed": 261_181,
            "grid_size": 28,
            "alpha": 1.0,
            "tau_eff": 5.0e-5,
            "levels": (128, 256, 512, 1024, 2048),
            "observation_time_fractions": (0.25, 0.5, 0.75, 1.0),
            "panel_clusters": 8,
            "nested_main_paths": (32, 64),
            "nested_reference_paths": (16, 32),
            "antithetic_main_paths": 16,
            "antithetic_reference_paths": 16,
            "maximum_main_half_width": 0.0025,
            "maximum_reference_half_width": 0.005,
            "maximum_projected_hours": 48.0,
            "minimum_rate": 1300.0,
            "maximum_fallback_fraction": 1.0e-4,
            "maximum_fallback_cost_fraction": 0.10,
            "maximum_peak_memory_fraction": 0.80,
            "maximum_cuda_mass_error": 2.0e-6,
            "maximum_prefix_bits": 1024,
        }
        for name, value in frozen.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is frozen at {value}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "levels",
            "observation_time_fractions",
            "nested_main_paths",
            "nested_reference_paths",
        ):
            result[name] = list(result[name])
        return result


class HaarCouplingDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    HIERARCHICAL_RNG_ALGEBRA_INVALID = "hierarchical_rng_algebra_invalid"
    CERTIFIED_NORMAL_TRANSFORM_INVALID = "certified_normal_transform_invalid"
    ARBITRARY_UNIFORM_JACOBI_CERTIFICATE_INVALID = (
        "arbitrary_uniform_jacobi_certificate_invalid"
    )
    HIERARCHICAL_MARGINAL_LAW_INVALID = "hierarchical_marginal_law_invalid"
    HIERARCHICAL_SCHEDULER_INVALID = "hierarchical_scheduler_invalid"
    HIERARCHICAL_COUPLING_COMPUTATIONALLY_INFEASIBLE = (
        "hierarchical_coupling_computationally_infeasible"
    )
    HIERARCHICAL_POWER_INFEASIBLE = "hierarchical_power_infeasible"
    HIERARCHICAL_PANELS_DISAGREE = "hierarchical_panels_disagree"
    EXACT_HAAR_HIERARCHICAL_REFINEMENT_COUPLING_FEASIBLE = (
        "exact_haar_hierarchical_refinement_coupling_feasible"
    )


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
    passed = isinstance(value, (list, tuple)) and tuple(value) == tuple(expected)
    return _check(value, "==", list(expected), passed)


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
            and all(_one(check.get("passed")) for check in normalized.values())
        ),
        **NO_WORK,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    result = _gate(name, "not evaluated", {}, evaluation_status="not_evaluated")
    result["reason"] = str(reason)
    return result


def evaluate_haar_preflight(
    metrics: Mapping[str, Any],
    thresholds: HaarCouplingThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate immutable provenance and algebraic preflight controls."""

    t = thresholds or HaarCouplingThresholds()
    flags = (
        "production_authorizing_pass",
        "control_provenance_pass",
        "parent_readjudication_pass",
        "parent_sources_immutable_pass",
        "parent_preflight_pass",
        "parent_pilot_numerically_valid_pass",
        "parent_pilot_resource_valid_pass",
        "parent_power_only_failure_pass",
        "normal_inverse_arb_enclosure_pass",
        "normal_cdf_arb_enclosure_pass",
        "normal_extreme_prefix_pass",
        "fused_cuda_normal_authorizer_pass",
        "haar_covariance_pass",
        "haar_within_level_independence_pass",
        "haar_parent_child_aggregation_pass",
        "antithetic_marginal_equality_pass",
        "state_independent_uniform_pass",
        "path_id_slot_plan_pass",
        "future_production_reserved_pass",
        "profile_panel_disjoint_pass",
        "path_id_uniqueness_pass",
        "intentional_haar_ancestry_only_pass",
        "order_chunk_resume_invariance_pass",
        "saved_prefix_jacobi_replay_pass",
        "arbitrary_uniform_cuda_authorizer_pass",
        "jacobi_marginal_cdf_pass",
        "jacobi_eigenmoment_pass",
        "jacobi_detailed_balance_pass",
        "rb_target_certificate_pass",
        "later_state_only_contract_pass",
        "all_colors_pass",
        "half_full_duration_pass",
        "facet_pass",
        "zero_mass_duration_pass",
        "phase_tower_identity_pass",
        "interruption_replay_pass",
        "deterministic_batching_pass",
        "candidate_under_48h_forecast_pass",
    )
    checks: dict[str, dict[str, Any]] = {
        name: _eq_one(metrics, name) for name in flags
    }
    checks.update(
        {
            "parent_record_count": _equal(
                metrics, "parent_record_count", t.parent_record_count
            ),
            "parent_source_count": _equal(
                metrics, "parent_source_count", t.parent_source_count
            ),
            "root_seed": _equal(metrics, "root_seed", t.root_seed),
            "grid_size": _equal(metrics, "grid_size", t.grid_size),
            "alpha": _equal(metrics, "alpha", t.alpha),
            "tau_eff": _equal(metrics, "tau_eff", t.tau_eff),
            "levels": _sequence_equal(metrics, "levels", t.levels),
            "maximum_prefix_bits": _le(
                metrics, "maximum_prefix_bits", t.maximum_prefix_bits
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
            "mass_error": _le(
                metrics, "mass_error", t.maximum_cuda_mass_error
            ),
            **{
                name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS
            },
        }
    )
    result = _gate(
        "haar_preflight",
        "provenance, certified normal/Haar algebra, and exact Jacobi marginals",
        checks,
    )
    result.update(
        {
            "provenance_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "control_provenance_pass",
                        "parent_readjudication_pass",
                        "parent_sources_immutable_pass",
                        "parent_preflight_pass",
                        "parent_pilot_numerically_valid_pass",
                        "parent_pilot_resource_valid_pass",
                        "parent_power_only_failure_pass",
                        "parent_record_count",
                        "parent_source_count",
                    )
                )
            ),
            "rng_algebra_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "haar_covariance_pass",
                        "haar_within_level_independence_pass",
                        "haar_parent_child_aggregation_pass",
                        "antithetic_marginal_equality_pass",
                        "state_independent_uniform_pass",
                    )
                )
            ),
            "normal_transform_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "normal_inverse_arb_enclosure_pass",
                        "normal_cdf_arb_enclosure_pass",
                        "normal_extreme_prefix_pass",
                        "fused_cuda_normal_authorizer_pass",
                        "maximum_prefix_bits",
                    )
                )
            ),
            "jacobi_certificate_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "saved_prefix_jacobi_replay_pass",
                        "arbitrary_uniform_cuda_authorizer_pass",
                        "rb_target_certificate_pass",
                        "certificate_fraction",
                    )
                )
            ),
            "marginal_law_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "jacobi_marginal_cdf_pass",
                        "jacobi_eigenmoment_pass",
                        "jacobi_detailed_balance_pass",
                        "later_state_only_contract_pass",
                        "phase_tower_identity_pass",
                    )
                )
            ),
            "scheduler_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "interruption_replay_pass",
                        "deterministic_batching_pass",
                        "path_id_slot_plan_pass",
                        "future_production_reserved_pass",
                        "profile_panel_disjoint_pass",
                        "path_id_uniqueness_pass",
                        "intentional_haar_ancestry_only_pass",
                        "order_chunk_resume_invariance_pass",
                        "all_colors_pass",
                        "half_full_duration_pass",
                        "facet_pass",
                        "zero_mass_duration_pass",
                    )
                )
            ),
            "resource_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "candidate_under_48h_forecast_pass",
                        "fallback_fraction",
                        "fallback_cost_fraction",
                        "peak_memory_fraction",
                    )
                )
            ),
        }
    )
    return result


def evaluate_haar_coupling(
    metrics: Mapping[str, Any],
    thresholds: HaarCouplingThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the measured certified coupling/marginal benchmark."""

    t = thresholds or HaarCouplingThresholds()
    flags = (
        "production_authorizing_pass",
        "nested_profile_complete_pass",
        "antithetic_profile_complete_pass",
        "normal_cells_certified_pass",
        "uniform_cells_certified_pass",
        "fused_cuda_normal_authorizer_pass",
        "jacobi_outputs_certified_pass",
        "arbitrary_uniform_cuda_authorizer_pass",
        "haar_covariance_pass",
        "within_level_independence_pass",
        "parent_child_aggregation_pass",
        "antithetic_marginal_pass",
        "state_independent_rng_pass",
        "id_uniqueness_pass",
        "intentional_ancestry_only_pass",
        "order_invariance_pass",
        "chunk_invariance_pass",
        "resume_invariance_pass",
        "marginal_cdf_pass",
        "marginal_eigenmoment_pass",
        "marginal_detailed_balance_pass",
        "conservation_pass",
        "target_contract_pass",
        "pipeline_runtime_projection_pass",
    )
    checks: dict[str, dict[str, Any]] = {
        name: _eq_one(metrics, name) for name in flags
    }
    checks.update(
        {
            "profile_order": _sequence_equal(
                metrics, "profile_order", PROFILE_ORDER
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
            "minimum_rate": _ge(metrics, "minimum_rate", t.minimum_rate),
            "minimum_projected_hours": _le(
                metrics, "minimum_projected_hours", t.maximum_projected_hours
            ),
            "peak_memory_fraction": _le(
                metrics, "peak_memory_fraction", t.maximum_peak_memory_fraction
            ),
            "mass_error": _le(
                metrics, "mass_error", t.maximum_cuda_mass_error
            ),
            **{
                name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS
            },
        }
    )
    result = _gate(
        "haar_coupling",
        "certified hierarchical-uniform execution and unchanged Jacobi marginals",
        checks,
    )
    result.update(
        {
            "rng_algebra_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "haar_covariance_pass",
                        "within_level_independence_pass",
                        "parent_child_aggregation_pass",
                        "antithetic_marginal_pass",
                        "state_independent_rng_pass",
                    )
                )
            ),
            "normal_transform_valid": int(
                checks["normal_cells_certified_pass"]["passed"]
                and checks["uniform_cells_certified_pass"]["passed"]
                and checks["fused_cuda_normal_authorizer_pass"]["passed"]
            ),
            "jacobi_certificate_valid": int(
                checks["jacobi_outputs_certified_pass"]["passed"]
                and checks["arbitrary_uniform_cuda_authorizer_pass"]["passed"]
                and checks["certificate_fraction"]["passed"]
            ),
            "marginal_law_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "marginal_cdf_pass",
                        "marginal_eigenmoment_pass",
                        "marginal_detailed_balance_pass",
                        "conservation_pass",
                        "target_contract_pass",
                    )
                )
            ),
            "scheduler_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "id_uniqueness_pass",
                        "intentional_ancestry_only_pass",
                        "order_invariance_pass",
                        "chunk_invariance_pass",
                        "resume_invariance_pass",
                        "profile_order",
                    )
                )
            ),
            "resource_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "pipeline_runtime_projection_pass",
                        "minimum_rate",
                        "minimum_projected_hours",
                        "fallback_fraction",
                        "fallback_cost_fraction",
                        "peak_memory_fraction",
                    )
                )
            ),
            "numerically_valid": int(
                all(
                    check["passed"]
                    for name, check in checks.items()
                    if name
                    not in {
                        "pipeline_runtime_projection_pass",
                        "minimum_rate",
                        "minimum_projected_hours",
                        "peak_memory_fraction",
                    }
                )
            ),
        }
    )
    return result


def _as_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with a nonempty path axis")
    return array


def antithetic_fine_mean(
    positive_detail: Sequence[float] | np.ndarray,
    negative_detail: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the exact arithmetic mean of the two antithetic fine arms."""

    positive = _as_array(positive_detail, "positive_detail")
    negative = _as_array(negative_detail, "negative_detail")
    if positive.shape != negative.shape:
        raise ValueError("antithetic arms must have identical shapes")
    return (positive + negative) * 0.5


def raw_successive_differences(
    level_observables: Mapping[int, Sequence[float] | np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute raw endpoint differences D1..D4 in the frozen orientation."""

    levels = (128, 256, 512, 1024, 2048)
    if set(level_observables) != set(levels):
        raise ValueError(f"level_observables must contain exactly {list(levels)}")
    arrays = {
        level: _as_array(level_observables[level], f"K={level}")
        for level in levels
    }
    shape = arrays[128].shape
    if any(array.shape != shape for array in arrays.values()):
        raise ValueError("all coupled level observables must have identical shapes")
    return {
        "D1": arrays[128] - arrays[256],
        "D2": arrays[256] - arrays[512],
        "D3": arrays[512] - arrays[1024],
        "D4": arrays[1024] - arrays[2048],
    }


def independent_pool_richardson_contrasts(
    main_d3: Sequence[float] | np.ndarray,
    reference_d4: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Return frozen Richardson means and independent-pool variances.

    ``main_d3`` and ``reference_d4`` are deliberately separate whole-cluster
    samples.  Consequently their covariance is exactly absent from the
    estimator variance rather than estimated from an artificial pairing.
    """

    d3 = _as_array(main_d3, "main_d3")
    d4 = _as_array(reference_d4, "reference_d4")
    if d3.shape[1:] != d4.shape[1:]:
        raise ValueError("main/reference feature shapes must agree")
    n3, n4 = d3.shape[0], d4.shape[0]
    mean3 = np.mean(d3, axis=0)
    mean4 = np.mean(d4, axis=0)
    var3 = np.var(d3, axis=0, ddof=1) if n3 > 1 else np.full_like(mean3, np.inf)
    var4 = np.var(d4, axis=0, ddof=1) if n4 > 1 else np.full_like(mean4, np.inf)
    generator_reference = mean3 + (4.0 / 3.0) * mean4
    reference_stability = (mean3 - 4.0 * mean4) / 3.0
    generator_variance = var3 / n3 + (16.0 / 9.0) * var4 / n4
    stability_variance = var3 / (9.0 * n3) + (16.0 / 9.0) * var4 / n4
    return {
        "generator_reference_contrast": generator_reference,
        "reference_stability_contrast": reference_stability,
        "generator_reference_variance": generator_variance,
        "reference_stability_variance": stability_variance,
        "main_path_count": int(n3),
        "reference_path_count": int(n4),
        "independent_pool_covariance": 0.0,
    }


def _candidate_paths_valid(
    profile: str, main_paths: Any, reference_paths: Any, t: HaarCouplingThresholds
) -> bool:
    if profile == NESTED_HAAR_PROFILE:
        return (
            main_paths in t.nested_main_paths
            and reference_paths in t.nested_reference_paths
        )
    if profile == ANTITHETIC_HAAR_PROFILE:
        return (
            main_paths == t.antithetic_main_paths
            and reference_paths == t.antithetic_reference_paths
        )
    return False


def nominate_haar_power_design(
    *,
    profile: str,
    panel_role: str,
    candidates: Sequence[Mapping[str, Any]],
    thresholds: HaarCouplingThresholds | None = None,
) -> dict[str, Any]:
    """Nominate the cheapest qualifying design from discovery panel A."""

    t = thresholds or HaarCouplingThresholds()
    if profile not in PROFILE_ORDER:
        raise ValueError(f"unknown Haar profile: {profile}")
    if panel_role != "a":
        raise ValueError("only sealed panel A may nominate a design")
    if not candidates:
        raise ValueError("candidate list cannot be empty")
    expected_count = 4 if profile == NESTED_HAAR_PROFILE else 1
    if len(candidates) != expected_count:
        raise ValueError(
            f"{profile} requires exactly {expected_count} candidate rows"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in candidates:
        row = dict(raw)
        main_paths = row.get("main_paths")
        reference_paths = row.get("reference_paths")
        if not _candidate_paths_valid(
            profile, main_paths, reference_paths, t
        ):
            raise ValueError("candidate path counts are outside the frozen profile")
        key = (int(main_paths), int(reference_paths))
        if key in seen:
            raise ValueError("duplicate candidate path counts")
        seen.add(key)
        required_flags = (
            "panel_complete_pass",
            "panel_finite_pass",
            "panel_certification_pass",
            "panel_numerical_health_pass",
            "mass_conservation_pass",
            "shard_chain_pass",
            "pilot_production_isolation_pass",
            "pilot_means_excluded_pass",
            "raw_endpoint_authorizing_pass",
            "dynkin_advisory_only_pass",
        )
        numerical = all(_one(row.get(name)) for name in required_flags)
        main_width = row.get("predicted_main_half_width")
        generator_width = row.get("predicted_generator_reference_half_width")
        stability_width = row.get("predicted_reference_stability_half_width")
        hours = row.get("projected_hours")
        rate = row.get("conservative_rate")
        eligible = (
            numerical
            and _finite(main_width)
            and float(main_width) <= t.maximum_main_half_width
            and _finite(generator_width)
            and float(generator_width) <= t.maximum_reference_half_width
            and _finite(stability_width)
            and float(stability_width) <= t.maximum_reference_half_width
            and _finite(hours)
            and 0.0 <= float(hours) <= t.maximum_projected_hours
            and _finite(rate)
            and float(rate) >= t.minimum_rate
        )
        row.update(
            {
                "profile": profile,
                "panel_role": panel_role,
                "eligible": int(eligible),
                "forecast_only": 1,
                **NO_WORK,
            }
        )
        normalized.append(row)
    if profile == NESTED_HAAR_PROFILE:
        expected = {
            (main, reference)
            for main in t.nested_main_paths
            for reference in t.nested_reference_paths
        }
        if seen != expected:
            raise ValueError("nested profile candidate grid is incomplete")
    eligible_rows = [row for row in normalized if _one(row["eligible"])]
    eligible_rows.sort(
        key=lambda row: (
            float(row["projected_hours"]),
            int(row["main_paths"]) + int(row["reference_paths"]),
            int(row["main_paths"]),
            int(row["reference_paths"]),
        )
    )
    selected = dict(eligible_rows[0]) if eligible_rows else None
    return {
        "schema": SCHEMA + "-panel-a-nomination",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "profile": profile,
        "panel_role": panel_role,
        "candidate_count": len(normalized),
        "candidates": normalized,
        "eligible_candidate_count": len(eligible_rows),
        "selected": selected,
        "selection_status": (
            "panel_a_design_nominated"
            if selected is not None
            else "panel_a_no_eligible_design"
        ),
        "passed": int(selected is not None),
        "ranking": [
            "projected_hours",
            "total_paths",
            "main_paths",
            "reference_paths",
        ],
        "thresholds": {
            "maximum_main_half_width": t.maximum_main_half_width,
            "maximum_generator_reference_half_width": (
                t.maximum_reference_half_width
            ),
            "maximum_reference_stability_half_width": (
                t.maximum_reference_half_width
            ),
            "maximum_projected_hours": t.maximum_projected_hours,
            "minimum_rate": t.minimum_rate,
        },
        **NO_WORK,
    }


def _panel_confirms(
    panel: Mapping[str, Any] | None,
    selected: Mapping[str, Any],
    t: HaarCouplingThresholds,
) -> bool:
    if not isinstance(panel, Mapping):
        return False
    return (
        _status(panel) == "evaluated"
        and panel.get("profile") == selected.get("profile")
        and panel.get("main_paths") == selected.get("main_paths")
        and panel.get("reference_paths") == selected.get("reference_paths")
        and _one(panel.get("complete_pass"))
        and _one(panel.get("finite_pass"))
        and _one(panel.get("certification_pass"))
        and _one(panel.get("numerical_health_pass"))
        and _one(panel.get("mass_conservation_pass"))
        and _one(panel.get("shard_chain_pass"))
        and _finite(panel.get("main_half_width"))
        and float(panel["main_half_width"]) <= t.maximum_main_half_width
        and _finite(panel.get("generator_reference_half_width"))
        and float(panel["generator_reference_half_width"])
        <= t.maximum_reference_half_width
        and _finite(panel.get("reference_stability_half_width"))
        and float(panel["reference_stability_half_width"])
        <= t.maximum_reference_half_width
        and _finite(panel.get("projected_hours"))
        and float(panel["projected_hours"]) <= t.maximum_projected_hours
        and _finite(panel.get("minimum_rate"))
        and float(panel["minimum_rate"]) >= t.minimum_rate
    )


def decide_sealed_profile_selection(
    *,
    nested_panel_a: Mapping[str, Any],
    nested_panel_b: Mapping[str, Any] | None = None,
    nested_combined: Mapping[str, Any] | None = None,
    antithetic_panel_a: Mapping[str, Any] | None = None,
    antithetic_panel_b: Mapping[str, Any] | None = None,
    antithetic_combined: Mapping[str, Any] | None = None,
    thresholds: HaarCouplingThresholds | None = None,
) -> dict[str, Any]:
    """Apply frozen profile order and fail-closed sealed-panel semantics."""

    t = thresholds or HaarCouplingThresholds()
    if nested_panel_a.get("profile") != NESTED_HAAR_PROFILE:
        raise ValueError("nested panel A has the wrong profile")
    nested_selected = nested_panel_a.get("selected")
    chosen_profile: str | None
    panel_a: Mapping[str, Any]
    panel_b: Mapping[str, Any] | None
    combined: Mapping[str, Any] | None
    if isinstance(nested_selected, Mapping):
        if any(
            value is not None
            for value in (
                antithetic_panel_a,
                antithetic_panel_b,
                antithetic_combined,
            )
        ):
            raise ValueError(
                "antithetic profile must remain unopened after nested nomination"
            )
        chosen_profile = NESTED_HAAR_PROFILE
        panel_a, panel_b, combined = (
            nested_panel_a,
            nested_panel_b,
            nested_combined,
        )
    else:
        if nested_panel_b is not None or nested_combined is not None:
            raise ValueError("nested panel B cannot open without a nomination")
        if antithetic_panel_a is None:
            return {
                "schema": SCHEMA + "-sealed-profile-selection",
                "schema_version": SCHEMA_VERSION,
                "evaluation_status": "evaluated",
                "selection_status": "antithetic_panel_a_not_evaluated",
                "passed": 0,
                "selected": None,
                "panel_a_nominated": 0,
                "panel_b_opened": 0,
                "panels_agree": 0,
                **NO_WORK,
            }
        if antithetic_panel_a.get("profile") != ANTITHETIC_HAAR_PROFILE:
            raise ValueError("antithetic panel A has the wrong profile")
        chosen_profile = ANTITHETIC_HAAR_PROFILE
        panel_a, panel_b, combined = (
            antithetic_panel_a,
            antithetic_panel_b,
            antithetic_combined,
        )
    selected = panel_a.get("selected")
    if not isinstance(selected, Mapping):
        if panel_b is not None or combined is not None:
            raise ValueError("panel B cannot open without a panel-A nomination")
        return {
            "schema": SCHEMA + "-sealed-profile-selection",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "selection_status": "all_profile_a_panels_underpowered",
            "passed": 0,
            "selected": None,
            "selected_profile": None,
            "panel_a_nominated": 0,
            "panel_b_opened": 0,
            "panels_agree": 0,
            **NO_WORK,
        }
    frozen_selected = dict(selected)
    if panel_b is None:
        return {
            "schema": SCHEMA + "-sealed-profile-selection",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "selection_status": "panel_a_nominated_panel_b_sealed",
            "passed": 0,
            "selected": frozen_selected,
            "selected_profile": chosen_profile,
            "panel_a_nominated": 1,
            "panel_b_opened": 0,
            "panels_agree": 0,
            **NO_WORK,
        }
    b_pass = _panel_confirms(panel_b, frozen_selected, t)
    combined_pass = _panel_confirms(combined, frozen_selected, t)
    passed = b_pass and combined_pass
    return {
        "schema": SCHEMA + "-sealed-profile-selection",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "selection_status": (
            "sealed_design_confirmed"
            if passed
            else "sealed_panel_b_or_combined_disagrees"
        ),
        "passed": int(passed),
        "selected": frozen_selected,
        "selected_profile": chosen_profile,
        "panel_a_nominated": 1,
        "panel_b_opened": 1,
        "panel_b_confirmation_pass": int(b_pass),
        "combined_confirmation_pass": int(combined_pass),
        "panels_agree": int(passed),
        "fallback_after_panel_b_permitted": 0,
        **NO_WORK,
    }


def evaluate_haar_pilot(
    metrics: Mapping[str, Any],
    thresholds: HaarCouplingThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the sealed power-pilot record produced by profile selection."""

    t = thresholds or HaarCouplingThresholds()
    flags = (
        "production_authorizing_pass",
        "plans_frozen_pass",
        "panels_disjoint_pass",
        "panel_nonregeneration_pass",
        "profile_order_pass",
        "no_fallback_after_panel_b_pass",
        "raw_endpoint_authorizing_pass",
        "dynkin_advisory_only_pass",
        "independent_pool_variance_pass",
        "richardson_formula_pass",
        "executed_panels_complete_pass",
        "executed_panels_numerically_valid_pass",
        "shard_chain_pass",
        "mass_conservation_pass",
        "pilot_production_isolation_pass",
    )
    checks: dict[str, dict[str, Any]] = {
        name: _eq_one(metrics, name) for name in flags
    }
    checks.update(
        {
            "selected_profile": _check(
                metrics.get("selected_profile"),
                "in",
                list(PROFILE_ORDER),
                metrics.get("selected_profile") in PROFILE_ORDER,
            ),
            "panel_a_clusters": _equal(
                metrics, "panel_a_clusters", t.panel_clusters
            ),
            "panel_b_clusters": _equal(
                metrics, "panel_b_clusters", t.panel_clusters
            ),
            "combined_clusters": _equal(
                metrics, "combined_clusters", 2 * t.panel_clusters
            ),
            "panel_a_nominated": _eq_one(metrics, "panel_a_nominated"),
            "panel_b_opened": _eq_one(metrics, "panel_b_opened"),
            "panels_agree": _eq_one(metrics, "panels_agree"),
            "combined_main_half_width": _le(
                metrics,
                "combined_main_half_width",
                t.maximum_main_half_width,
            ),
            "combined_generator_reference_half_width": _le(
                metrics,
                "combined_generator_reference_half_width",
                t.maximum_reference_half_width,
            ),
            "combined_reference_stability_half_width": _le(
                metrics,
                "combined_reference_stability_half_width",
                t.maximum_reference_half_width,
            ),
            "projected_hours": _le(
                metrics, "projected_hours", t.maximum_projected_hours
            ),
            "minimum_rate": _ge(metrics, "minimum_rate", t.minimum_rate),
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
            "mass_error": _le(
                metrics, "mass_error", t.maximum_cuda_mass_error
            ),
            **{
                name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS
            },
        }
    )
    result = _gate(
        "haar_power_pilot",
        "sealed A/B engineering power forecast using raw coupled endpoints",
        checks,
    )
    numerical_names = (
        "executed_panels_complete_pass",
        "executed_panels_numerically_valid_pass",
        "shard_chain_pass",
        "mass_conservation_pass",
        "certificate_fraction",
        *tuple(_FORBIDDEN_COUNTS),
    )
    resource_names = (
        "projected_hours",
        "minimum_rate",
        "fallback_fraction",
        "fallback_cost_fraction",
        "peak_memory_fraction",
    )
    power_names = (
        "panel_a_nominated",
        "panel_b_opened",
        "panels_agree",
        "combined_main_half_width",
        "combined_generator_reference_half_width",
        "combined_reference_stability_half_width",
    )
    result.update(
        {
            "scheduler_valid": int(
                all(
                    checks[name]["passed"]
                    for name in (
                        "plans_frozen_pass",
                        "panels_disjoint_pass",
                        "panel_nonregeneration_pass",
                        "profile_order_pass",
                        "no_fallback_after_panel_b_pass",
                        "raw_endpoint_authorizing_pass",
                        "dynkin_advisory_only_pass",
                        "independent_pool_variance_pass",
                        "richardson_formula_pass",
                        "pilot_production_isolation_pass",
                        "panel_a_clusters",
                        "panel_b_clusters",
                        "combined_clusters",
                    )
                )
            ),
            "numerically_valid": int(
                all(checks[name]["passed"] for name in numerical_names)
            ),
            "resource_valid": int(
                all(checks[name]["passed"] for name in resource_names)
            ),
            "power_valid": int(
                all(checks[name]["passed"] for name in power_names)
            ),
            "panel_a_nominated": int(checks["panel_a_nominated"]["passed"]),
            "panel_b_opened": int(checks["panel_b_opened"]["passed"]),
            "panels_agree": int(checks["panels_agree"]["passed"]),
        }
    )
    return result


def _explicit_provenance_invalid(
    provenance: bool | int | Mapping[str, Any],
) -> bool:
    if provenance is False or (
        isinstance(provenance, (int, np.integer)) and int(provenance) == 0
    ):
        return True
    return (
        isinstance(provenance, Mapping)
        and _status(provenance) == "evaluated"
        and _zero(provenance.get("passed"))
    )


def _failure_decision(
    gate: Mapping[str, Any],
) -> tuple[HaarCouplingDecision, str]:
    domain = str(gate.get("failure_domain", "")).lower()
    if "provenance" in domain:
        return (
            HaarCouplingDecision.CONTROL_PROVENANCE_INVALID,
            "repair the immutable parent binding",
        )
    if "rng" in domain or "haar" in domain or "algebra" in domain:
        return (
            HaarCouplingDecision.HIERARCHICAL_RNG_ALGEBRA_INVALID,
            "repair the certified Haar algebra",
        )
    if "normal" in domain:
        return (
            HaarCouplingDecision.CERTIFIED_NORMAL_TRANSFORM_INVALID,
            "repair the certified normal transform",
        )
    if "jacobi" in domain or "certificate" in domain:
        return (
            HaarCouplingDecision.ARBITRARY_UNIFORM_JACOBI_CERTIFICATE_INVALID,
            "repair arbitrary-uniform Jacobi certification",
        )
    if "marginal" in domain or "law" in domain:
        return (
            HaarCouplingDecision.HIERARCHICAL_MARGINAL_LAW_INVALID,
            "repair exact marginal-law controls",
        )
    if "scheduler" in domain or "config" in domain:
        return (
            HaarCouplingDecision.HIERARCHICAL_SCHEDULER_INVALID,
            "repair the hierarchical scheduler or namespace",
        )
    return (
        HaarCouplingDecision.HIERARCHICAL_COUPLING_COMPUTATIONALLY_INFEASIBLE,
        "repair exact coupling execution within frozen resources",
    )


def decide_haar_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    coupling_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return exactly one closed controls-only workflow decision."""

    if _explicit_provenance_invalid(provenance):
        decision = HaarCouplingDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 1,061-artifact parent binding"
    elif _status(preflight_gate) == "execution_failed":
        decision, action = _failure_decision(preflight_gate)
    elif not _passed(provenance):
        decision = HaarCouplingDecision.CONTROL_PROVENANCE_INVALID
        action = "obtain verified provenance evidence"
    elif _status(preflight_gate) != "evaluated":
        decision = HaarCouplingDecision.HIERARCHICAL_RNG_ALGEBRA_INVALID
        action = "complete the certified Haar preflight"
    elif not _passed(preflight_gate):
        if _zero(preflight_gate.get("provenance_valid")):
            decision = HaarCouplingDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable parent/source/config binding"
        elif _zero(preflight_gate.get("rng_algebra_valid")):
            decision = HaarCouplingDecision.HIERARCHICAL_RNG_ALGEBRA_INVALID
            action = "repair hierarchical normal/Haar algebra"
        elif _zero(preflight_gate.get("normal_transform_valid")):
            decision = HaarCouplingDecision.CERTIFIED_NORMAL_TRANSFORM_INVALID
            action = "repair certified normal inverse/CDF transforms"
        elif _zero(preflight_gate.get("jacobi_certificate_valid")):
            decision = (
                HaarCouplingDecision.ARBITRARY_UNIFORM_JACOBI_CERTIFICATE_INVALID
            )
            action = "repair arbitrary-uniform Jacobi authorization"
        elif _zero(preflight_gate.get("marginal_law_valid")):
            decision = HaarCouplingDecision.HIERARCHICAL_MARGINAL_LAW_INVALID
            action = "repair exact marginal-law controls"
        elif _zero(preflight_gate.get("scheduler_valid")):
            decision = HaarCouplingDecision.HIERARCHICAL_SCHEDULER_INVALID
            action = "repair coupling namespaces or deterministic replay"
        else:
            decision = (
                HaarCouplingDecision.HIERARCHICAL_COUPLING_COMPUTATIONALLY_INFEASIBLE
            )
            action = "repair exact coupling execution within resource limits"
    elif _status(coupling_gate) == "execution_failed":
        decision, action = _failure_decision(coupling_gate)
    elif _status(coupling_gate) != "evaluated":
        decision = HaarCouplingDecision.HIERARCHICAL_SCHEDULER_INVALID
        action = "run certified coupling controls and benchmarks"
    elif not _passed(coupling_gate):
        if _zero(coupling_gate.get("rng_algebra_valid")):
            decision = HaarCouplingDecision.HIERARCHICAL_RNG_ALGEBRA_INVALID
            action = "repair measured Haar algebra"
        elif _zero(coupling_gate.get("normal_transform_valid")):
            decision = HaarCouplingDecision.CERTIFIED_NORMAL_TRANSFORM_INVALID
            action = "repair certified normal cells"
        elif _zero(coupling_gate.get("jacobi_certificate_valid")):
            decision = (
                HaarCouplingDecision.ARBITRARY_UNIFORM_JACOBI_CERTIFICATE_INVALID
            )
            action = "repair arbitrary-uniform Jacobi certification"
        elif _zero(coupling_gate.get("marginal_law_valid")):
            decision = HaarCouplingDecision.HIERARCHICAL_MARGINAL_LAW_INVALID
            action = "repair hierarchical marginal-law evidence"
        elif _zero(coupling_gate.get("scheduler_valid")):
            decision = HaarCouplingDecision.HIERARCHICAL_SCHEDULER_INVALID
            action = "repair exact deterministic coupling scheduling"
        else:
            decision = (
                HaarCouplingDecision.HIERARCHICAL_COUPLING_COMPUTATIONALLY_INFEASIBLE
            )
            action = "repair exact coupling performance within 48 hours"
    elif _status(pilot_gate) == "execution_failed":
        decision, action = _failure_decision(pilot_gate)
    elif _status(pilot_gate) != "evaluated":
        decision = HaarCouplingDecision.HIERARCHICAL_POWER_INFEASIBLE
        action = "run the sealed hierarchical A/B power pilot"
    elif not _passed(pilot_gate):
        if _zero(pilot_gate.get("scheduler_valid")):
            decision = HaarCouplingDecision.HIERARCHICAL_SCHEDULER_INVALID
            action = "repair sealed profile order or raw-endpoint statistics"
        elif _zero(pilot_gate.get("numerically_valid")) or _zero(
            pilot_gate.get("resource_valid")
        ):
            decision = (
                HaarCouplingDecision.HIERARCHICAL_COUPLING_COMPUTATIONALLY_INFEASIBLE
            )
            action = "repair exact pilot execution within frozen resources"
        elif _zero(pilot_gate.get("panel_a_nominated")):
            decision = HaarCouplingDecision.HIERARCHICAL_POWER_INFEASIBLE
            action = "retain sealed evidence; do not weaken thresholds"
        else:
            decision = HaarCouplingDecision.HIERARCHICAL_PANELS_DISAGREE
            action = "retain both sealed panels and do not select a design"
    else:
        decision = (
            HaarCouplingDecision.EXACT_HAAR_HIERARCHICAL_REFINEMENT_COUPLING_FEASIBLE
        )
        action = "plan a fresh production Strang-refinement experiment"

    production_ready = (
        decision
        is HaarCouplingDecision.EXACT_HAAR_HIERARCHICAL_REFINEMENT_COUPLING_FEASIBLE
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "decision": decision.value,
        "recommended_next_action": action,
        "coupling_stage_authorized": int(
            _passed(preflight_gate) and _status(coupling_gate) != "evaluated"
        ),
        "sealed_power_pilot_authorized": int(
            _passed(preflight_gate)
            and _passed(coupling_gate)
            and _status(pilot_gate) != "evaluated"
        ),
        "production_refinement_patch_authorized": int(production_ready),
        "one_image_phase_conditioned_training_patch_authorized": 0,
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        **NO_WORK,
    }


def evaluate_haar_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    coupling_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    """Combine stage gates and apply the requested fail-closed prefix."""

    order = ("preflight", "coupling", "pilot")
    if require_gate not in REQUIRED_GATES:
        raise ValueError(f"unknown required gate: {require_gate}")
    components = {
        "preflight": dict(
            preflight_gate or not_evaluated_gate("preflight", "not run")
        ),
        "coupling": dict(
            coupling_gate or not_evaluated_gate("coupling", "not run")
        ),
        "pilot": dict(pilot_gate or not_evaluated_gate("pilot", "not run")),
    }
    required = (
        ()
        if require_gate == "none"
        else order[: order.index(require_gate) + 1]
    )
    passed = bool(
        _passed(provenance)
        and all(_passed(components[name]) for name in required)
    )
    decision = decide_haar_workflow(
        provenance=provenance,
        preflight_gate=components["preflight"],
        coupling_gate=components["coupling"],
        pilot_gate=components["pilot"],
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_components": list(required),
        "components": components,
        "passed": int(passed),
        "required_gate_pass": int(passed),
        "decision": decision,
        "thresholds": HaarCouplingThresholds().to_dict(),
        **NO_WORK,
    }


__all__ = [
    "ANTITHETIC_HAAR_PROFILE",
    "HaarCouplingDecision",
    "HaarCouplingThresholds",
    "NESTED_HAAR_PROFILE",
    "PROFILE_ORDER",
    "REQUIRED_GATES",
    "STAGES",
    "antithetic_fine_mean",
    "decide_haar_workflow",
    "decide_sealed_profile_selection",
    "evaluate_haar_coupling",
    "evaluate_haar_pilot",
    "evaluate_haar_preflight",
    "evaluate_haar_workflow",
    "independent_pool_richardson_contrasts",
    "nominate_haar_power_design",
    "not_evaluated_gate",
    "raw_successive_differences",
]
