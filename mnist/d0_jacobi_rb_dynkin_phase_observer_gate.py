"""Fail-closed gates for the exact phase-local Dynkin observer repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping

import numpy as np

from mnist.d0_jacobi_rb_dynkin_power_gate import (
    DynkinPowerThresholds,
    evaluate_dynkin_power,
)


SCHEMA = "experiment12-d0-jacobi-rb-dynkin-phase-observer-gate"
SCHEMA_VERSION = 1
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


@dataclass(frozen=True)
class PhaseObserverThresholds:
    """Frozen thresholds for the controls-only observer repair."""

    parent_record_count: int = 18
    parent_source_count: int = 21
    root_seed: int = 261_171
    grid_size: int = 28
    alpha: float = 1.0
    tau_eff: float = 5.0e-5
    tower_panel_count: int = 2
    tower_clusters_per_panel: int = 128
    tower_family_member_count: int = 80
    structural_zero_member_count: int = 32
    tower_bootstrap_replicates: int = 20_000
    tower_confidence: float = 0.99
    maximum_float64_observer_error: float = 1.0e-10
    maximum_cuda_observer_error: float = 2.0e-6
    maximum_cumulative_standardized_error: float = 1.0e-8
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_cost_fraction: float = 0.10
    maximum_peak_memory_fraction: float = 0.80
    maximum_cuda_mass_error: float = 2.0e-6

    def __post_init__(self) -> None:
        frozen = {
            "parent_record_count": 18,
            "parent_source_count": 21,
            "root_seed": 261_171,
            "grid_size": 28,
            "alpha": 1.0,
            "tau_eff": 5.0e-5,
            "tower_panel_count": 2,
            "tower_clusters_per_panel": 128,
            "tower_family_member_count": 80,
            "structural_zero_member_count": 32,
            "tower_bootstrap_replicates": 20_000,
            "tower_confidence": 0.99,
            "maximum_float64_observer_error": 1.0e-10,
            "maximum_cuda_observer_error": 2.0e-6,
            "maximum_cumulative_standardized_error": 1.0e-8,
            "maximum_fallback_fraction": 1.0e-4,
            "maximum_fallback_cost_fraction": 0.10,
            "maximum_peak_memory_fraction": 0.80,
            "maximum_cuda_mass_error": 2.0e-6,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen at {expected}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhaseObserverDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    PARENT_FAILURE_ADJUDICATION_INVALID = (
        "parent_failure_adjudication_invalid"
    )
    REFINEMENT_SCHEDULER_INVALID = "refinement_scheduler_invalid"
    PHASE_OBSERVER_ALGEBRA_INVALID = "phase_observer_algebra_invalid"
    PHASE_OBSERVER_NUMERICALLY_UNRESOLVED = (
        "phase_observer_numerically_unresolved"
    )
    DYNKIN_TOWER_IDENTITY_INVALID = "dynkin_tower_identity_invalid"
    DYNKIN_COMPUTATIONALLY_INFEASIBLE = "dynkin_computationally_infeasible"
    PHASE_LOCAL_DYNKIN_OBSERVER_REPAIRED = (
        "phase_local_dynkin_observer_repaired"
    )
    DYNKIN_POWER_INFEASIBLE = "dynkin_power_infeasible"
    DYNKIN_PANELS_DISAGREE = "dynkin_panels_disagree"
    EXACT_DYNKIN_REFINEMENT_ESTIMATOR_FEASIBLE = (
        "exact_dynkin_refinement_estimator_feasible"
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


def _explicit_provenance_invalid(
    value: bool | int | Mapping[str, Any],
) -> bool:
    if not isinstance(value, Mapping):
        return value is False or _zero(value)
    if _status(value) != "evaluated":
        return False
    if _zero(value.get("provenance_valid")):
        return True
    subchecks = value.get("subchecks")
    return isinstance(subchecks, Mapping) and any(
        (
            "provenance" in str(name)
            or "parent_source" in str(name)
            or "parent_record" in str(name)
        )
        and isinstance(check, Mapping)
        and _zero(check.get("passed"))
        and check.get("value") is not None
        for name, check in subchecks.items()
    )


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


def _equal(
    metrics: Mapping[str, Any],
    name: str,
    expected: Any,
) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", expected, value == expected)


def _le(
    metrics: Mapping[str, Any],
    name: str,
    threshold: float,
) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value,
        "<=",
        threshold,
        _finite(value) and float(value) <= float(threshold),
    )


def _gate(
    name: str,
    claim_scope: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_status: str = "evaluated",
) -> dict[str, Any]:
    passed = (
        evaluation_status == "evaluated"
        and bool(checks)
        and all(_one(record.get("passed")) for record in checks.values())
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "claim_scope": claim_scope,
        "evaluation_status": evaluation_status,
        "passed": int(passed),
        "subchecks": {str(name): dict(value) for name, value in checks.items()},
        **NO_WORK,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    result = _gate(name, "not evaluated", {}, evaluation_status="not_evaluated")
    result["reason"] = str(reason)
    return result


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

_FLAG_NAMES = (
    "production_authorizing_pass",
    "control_provenance_pass",
    "parent_failure_adjudication_pass",
    "twenty_one_parent_sources_immutable_pass",
    "parent_path_id_plan_pass",
    "parent_legacy_k512_replay_pass",
    "parent_phase_moment_oracle_pass",
    "parent_no_tower_pilot_work_pass",
    "path_id_plan_pass",
    "fresh_namespace_disjoint_pass",
    "legacy_k512_replay_pass",
    "transition_target_certificate_hash_invariance_pass",
    "observer_state_hash_invariance_pass",
    "global_subtraction_roundoff_reproduced_pass",
    "phase_local_fourier_formula_pass",
    "phase_local_quadratic_formula_pass",
    "phase_local_cubic_formula_pass",
    "phase_local_all_matchings_pass",
    "phase_local_half_full_duration_pass",
    "phase_local_facet_interior_pass",
    "phase_local_zero_mass_duration_pass",
    "structural_invariant_mask_pass",
    "structural_zero_center_pass",
    "structural_zero_radius_pass",
    "float64_arb_agreement_pass",
    "cuda_enclosure_pass",
    "quantile_enclosure_pass",
    "noninvariant_local_global_agreement_pass",
    "phase_moment_oracle_pass",
    "tower_panel_a_pass",
    "tower_panel_b_pass",
    "tower_joint_max_t_pass",
    "tower_panels_frozen_pass",
    "tower_panels_disjoint_pass",
    "tower_case_atomic_resume_pass",
    "tower_case_observer_input_hash_invariance_pass",
    "tower_case_structural_zero_center_pass",
    "tower_case_structural_zero_radius_pass",
    "tower_case_noninvariant_global_agreement_pass",
    "tower_authorizing_interval_radius_pass",
    "negative_orientation_fixture_pass",
    "negative_quadratic_factor_fixture_pass",
    "negative_cubic_factor_fixture_pass",
    "negative_pair_mass_fixture_pass",
    "negative_duration_fixture_pass",
    "negative_eigenvalue_fixture_pass",
    "negative_corrupt_enclosure_fixture_pass",
    "negative_post_state_fixture_pass",
)


def evaluate_phase_observer_preflight(
    metrics: Mapping[str, Any],
    thresholds: PhaseObserverThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate immutable provenance, local observer, and tower controls."""

    t = thresholds or PhaseObserverThresholds()
    checks = {name: _eq_one(metrics, name) for name in _FLAG_NAMES}
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
            "tower_panel_count": _equal(
                metrics, "tower_panel_count", t.tower_panel_count
            ),
            "tower_clusters_per_panel": _equal(
                metrics,
                "tower_clusters_per_panel",
                t.tower_clusters_per_panel,
            ),
            "tower_family_member_count": _equal(
                metrics,
                "tower_family_member_count",
                t.tower_family_member_count,
            ),
            "structural_zero_member_count": _equal(
                metrics,
                "structural_zero_member_count",
                t.structural_zero_member_count,
            ),
            "tower_bootstrap_replicates": _equal(
                metrics,
                "tower_bootstrap_replicates",
                t.tower_bootstrap_replicates,
            ),
            "tower_confidence": _equal(
                metrics, "tower_confidence", t.tower_confidence
            ),
            "maximum_float64_observer_error": _le(
                metrics,
                "maximum_float64_observer_error",
                t.maximum_float64_observer_error,
            ),
            "maximum_cuda_observer_error": _le(
                metrics,
                "maximum_cuda_observer_error",
                t.maximum_cuda_observer_error,
            ),
            "maximum_cumulative_standardized_error": _le(
                metrics,
                "maximum_cumulative_standardized_error",
                t.maximum_cumulative_standardized_error,
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
            **{name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS},
        }
    )
    result = _gate(
        "jacobi_rb_dynkin_phase_observer_preflight",
        "exact phase-local observer and two sealed tower-identity controls",
        checks,
    )
    provenance_names = {
        "control_provenance_pass",
        "twenty_one_parent_sources_immutable_pass",
        "parent_path_id_plan_pass",
        "parent_legacy_k512_replay_pass",
        "parent_phase_moment_oracle_pass",
        "parent_no_tower_pilot_work_pass",
        "parent_record_count",
        "parent_source_count",
    }
    adjudication_names = {
        "parent_failure_adjudication_pass",
        "global_subtraction_roundoff_reproduced_pass",
    }
    scheduler_names = {
        "path_id_plan_pass",
        "fresh_namespace_disjoint_pass",
        "root_seed",
    }
    algebra_names = {
        "phase_local_fourier_formula_pass",
        "phase_local_quadratic_formula_pass",
        "phase_local_cubic_formula_pass",
        "phase_local_all_matchings_pass",
        "phase_local_half_full_duration_pass",
        "phase_local_facet_interior_pass",
        "phase_local_zero_mass_duration_pass",
        "structural_invariant_mask_pass",
        "negative_orientation_fixture_pass",
        "negative_quadratic_factor_fixture_pass",
        "negative_cubic_factor_fixture_pass",
        "negative_pair_mass_fixture_pass",
        "negative_duration_fixture_pass",
        "negative_eigenvalue_fixture_pass",
        "negative_post_state_fixture_pass",
    }
    tower_names = {
        "tower_panel_a_pass",
        "tower_panel_b_pass",
        "tower_joint_max_t_pass",
        "tower_panels_frozen_pass",
        "tower_panels_disjoint_pass",
        "tower_case_atomic_resume_pass",
        "tower_authorizing_interval_radius_pass",
        "tower_panel_count",
        "tower_clusters_per_panel",
        "tower_family_member_count",
        "structural_zero_member_count",
        "tower_bootstrap_replicates",
        "tower_confidence",
    }
    numerical_names = {
        "legacy_k512_replay_pass",
        "transition_target_certificate_hash_invariance_pass",
        "observer_state_hash_invariance_pass",
        "structural_zero_center_pass",
        "structural_zero_radius_pass",
        "tower_case_structural_zero_center_pass",
        "tower_case_structural_zero_radius_pass",
        "tower_case_noninvariant_global_agreement_pass",
        "tower_case_observer_input_hash_invariance_pass",
        "float64_arb_agreement_pass",
        "cuda_enclosure_pass",
        "quantile_enclosure_pass",
        "noninvariant_local_global_agreement_pass",
        "phase_moment_oracle_pass",
        "negative_corrupt_enclosure_fixture_pass",
        "maximum_float64_observer_error",
        "maximum_cuda_observer_error",
        "maximum_cumulative_standardized_error",
        "certificate_fraction",
        "mass_error",
        *_FORBIDDEN_COUNTS,
    }
    resource_names = {
        "fallback_fraction",
        "fallback_cost_fraction",
        "peak_memory_fraction",
    }
    result["provenance_valid"] = int(
        all(checks[name]["passed"] for name in provenance_names)
    )
    result["parent_adjudication_valid"] = int(
        all(checks[name]["passed"] for name in adjudication_names)
    )
    result["scheduler_valid"] = int(
        all(checks[name]["passed"] for name in scheduler_names)
    )
    result["observer_algebra_valid"] = int(
        all(checks[name]["passed"] for name in algebra_names)
    )
    result["tower_identity_valid"] = int(
        all(checks[name]["passed"] for name in tower_names)
    )
    result["numerically_valid"] = int(
        all(checks[name]["passed"] for name in numerical_names)
    )
    result["resource_valid"] = int(
        all(checks[name]["passed"] for name in resource_names)
    )
    result["thresholds"] = t.to_dict()
    return result


def evaluate_phase_observer_power(
    metrics: Mapping[str, Any],
    thresholds: DynkinPowerThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the unchanged sealed Dynkin power pilot."""

    result = evaluate_dynkin_power(metrics, thresholds)
    result["schema"] = SCHEMA
    result["gate"] = "jacobi_rb_dynkin_phase_observer_power"
    result["claim_scope"] = (
        "unchanged sealed A/B engineering forecast after observer repair"
    )
    return result


def _execution_failure_decision(
    gate: Mapping[str, Any],
) -> tuple[PhaseObserverDecision, str]:
    domain = (
        str(gate.get("failure_domain", ""))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if "scheduler" in domain or "configuration" in domain or domain == "config":
        return (
            PhaseObserverDecision.REFINEMENT_SCHEDULER_INVALID,
            "repair the frozen phase-observer namespace or scheduler",
        )
    if "resource" in domain or "comput" in domain:
        return (
            PhaseObserverDecision.DYNKIN_COMPUTATIONALLY_INFEASIBLE,
            "repair exact observer execution within frozen resource limits",
        )
    if "algebra" in domain or "formula" in domain:
        return (
            PhaseObserverDecision.PHASE_OBSERVER_ALGEBRA_INVALID,
            "repair the exact phase-local observer formulas",
        )
    return (
        PhaseObserverDecision.PHASE_OBSERVER_NUMERICALLY_UNRESOLVED,
        "repair the failed exact phase-observer execution",
    )


def decide_phase_observer_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return one closed controls-only workflow decision."""

    if _explicit_provenance_invalid(provenance):
        decision = PhaseObserverDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 18-artifact parent binding"
    elif _status(preflight_gate) == "execution_failed":
        decision, action = _execution_failure_decision(preflight_gate)
    elif not _passed(provenance):
        decision = PhaseObserverDecision.PHASE_OBSERVER_NUMERICALLY_UNRESOLVED
        action = "obtain verified provenance evidence before adjudication"
    elif _status(preflight_gate) != "evaluated":
        decision = PhaseObserverDecision.PHASE_OBSERVER_ALGEBRA_INVALID
        action = "complete the exact phase-local observer preflight"
    elif not _passed(preflight_gate):
        if _zero(preflight_gate.get("provenance_valid")):
            decision = PhaseObserverDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable parent/source/config binding"
        elif _zero(preflight_gate.get("parent_adjudication_valid")):
            decision = PhaseObserverDecision.PARENT_FAILURE_ADJUDICATION_INVALID
            action = "repair the tower-observer parent re-adjudication"
        elif _zero(preflight_gate.get("scheduler_valid")):
            decision = PhaseObserverDecision.REFINEMENT_SCHEDULER_INVALID
            action = "repair the frozen path-ID plan or scheduler"
        elif _zero(preflight_gate.get("observer_algebra_valid")):
            decision = PhaseObserverDecision.PHASE_OBSERVER_ALGEBRA_INVALID
            action = "repair the exact phase-local increment formulas"
        elif _zero(preflight_gate.get("tower_identity_valid")):
            decision = PhaseObserverDecision.DYNKIN_TOWER_IDENTITY_INVALID
            action = "repair the phase-local tower-identity controls"
        elif _zero(preflight_gate.get("numerically_valid")):
            decision = PhaseObserverDecision.PHASE_OBSERVER_NUMERICALLY_UNRESOLVED
            action = "repair certified phase-local observation or accumulation"
        elif _zero(preflight_gate.get("resource_valid")):
            decision = PhaseObserverDecision.DYNKIN_COMPUTATIONALLY_INFEASIBLE
            action = "repair exact observer execution without changing the law"
        else:
            decision = PhaseObserverDecision.PHASE_OBSERVER_NUMERICALLY_UNRESOLVED
            action = "repair incomplete preflight evidence"
    elif _status(pilot_gate) == "execution_failed":
        decision, action = _execution_failure_decision(pilot_gate)
    elif _status(pilot_gate) != "evaluated":
        decision = PhaseObserverDecision.PHASE_LOCAL_DYNKIN_OBSERVER_REPAIRED
        action = "run the unchanged sealed Dynkin A/B power pilot"
    elif not _passed(pilot_gate):
        if _zero(pilot_gate.get("numerically_valid")):
            decision = PhaseObserverDecision.PHASE_OBSERVER_NUMERICALLY_UNRESOLVED
            action = "repair exact pilot execution or shard evidence"
        elif _zero(pilot_gate.get("resource_valid")):
            decision = PhaseObserverDecision.DYNKIN_COMPUTATIONALLY_INFEASIBLE
            action = "repair observer scheduling within the frozen 48-hour budget"
        elif _zero(pilot_gate.get("panel_a_nominated")):
            decision = PhaseObserverDecision.DYNKIN_POWER_INFEASIBLE
            action = "retain evidence and add exact-marginal hierarchical coupling"
        elif _zero(pilot_gate.get("panels_agree")):
            decision = PhaseObserverDecision.DYNKIN_PANELS_DISAGREE
            action = "retain both sealed panels and do not select a design"
        else:
            decision = PhaseObserverDecision.PHASE_OBSERVER_NUMERICALLY_UNRESOLVED
            action = "repair incomplete pilot evidence"
    else:
        decision = (
            PhaseObserverDecision.EXACT_DYNKIN_REFINEMENT_ESTIMATOR_FEASIBLE
        )
        action = "plan a fresh production refinement using the frozen design"

    observer_repaired = (
        decision is PhaseObserverDecision.PHASE_LOCAL_DYNKIN_OBSERVER_REPAIRED
    )
    production_ready = (
        decision
        is PhaseObserverDecision.EXACT_DYNKIN_REFINEMENT_ESTIMATOR_FEASIBLE
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "decision": decision.value,
        "recommended_next_action": action,
        "sealed_power_pilot_authorized": int(observer_repaired),
        "production_refinement_patch_authorized": int(production_ready),
        "one_image_phase_conditioned_training_patch_authorized": 0,
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "closed_terminal_scientific_outcome": 0,
        **NO_WORK,
    }


__all__ = [
    "PhaseObserverDecision",
    "PhaseObserverThresholds",
    "decide_phase_observer_workflow",
    "evaluate_phase_observer_power",
    "evaluate_phase_observer_preflight",
    "not_evaluated_gate",
]
