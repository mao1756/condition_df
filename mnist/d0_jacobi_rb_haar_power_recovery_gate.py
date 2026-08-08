"""Fail-closed gates for immutable Haar power-panel recovery.

This module contains no sampler, trainer, or filesystem dependency.  It
adjudicates three prefixes:

``preflight``
    Bind the immutable failed Haar run and its completed shard set.
``replay``
    Reconstruct nested panel A without changing or recomputing it.
``pilot``
    Execute the already-sealed pairwise-antithetic fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping


SCHEMA = "experiment12-d0-jacobi-rb-haar-power-recovery-gate"
SCHEMA_VERSION = 1
STAGES = ("preflight", "replay", "pilot", "report", "all")
REQUIRED_GATES = ("none", "preflight", "replay", "pilot")
NO_WORK = {
    "physical_training_performed": 0,
    "production_refinement_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


class HaarPowerRecoveryDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    PANEL_SCHEDULE_BINDING_INVALID = "panel_schedule_binding_invalid"
    NESTED_PANEL_REPLAY_INVALID = "nested_panel_replay_invalid"
    ANTITHETIC_SCHEDULER_INVALID = "antithetic_scheduler_invalid"
    ANTITHETIC_COUPLING_COMPUTATIONALLY_INFEASIBLE = (
        "antithetic_coupling_computationally_infeasible"
    )
    HIERARCHICAL_POWER_INFEASIBLE = "hierarchical_power_infeasible"
    HIERARCHICAL_PANELS_DISAGREE = "hierarchical_panels_disagree"
    EXACT_HAAR_HIERARCHICAL_REFINEMENT_COUPLING_FEASIBLE = (
        "exact_haar_hierarchical_refinement_coupling_feasible"
    )


@dataclass(frozen=True)
class HaarPowerRecoveryThresholds:
    root_seed: int = 261_181
    parent_registry_record_count: int = 197
    parent_source_count: int = 35
    parent_main_shards: int = 16
    parent_reference_shards: int = 64
    recovered_shards: int = 80
    recovered_transition_count: int = 120_823_808
    recovered_fallback_count: int = 38
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_cost_fraction: float = 0.10
    maximum_mass_error: float = 2.0e-6
    maximum_peak_memory_fraction: float = 0.80
    minimum_rate: float = 1_300.0
    maximum_main_half_width: float = 0.0025
    maximum_reference_half_width: float = 0.005
    maximum_projected_hours: float = 48.0
    expected_nested_candidates: int = 4
    expected_nested_eligible_candidates: int = 0
    antithetic_main_paths: int = 16
    antithetic_reference_paths: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _status(gate: Mapping[str, Any] | None) -> str:
    return (
        str(gate.get("evaluation_status", "not_evaluated"))
        if isinstance(gate, Mapping)
        else "not_evaluated"
    )


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return (
        isinstance(gate, Mapping)
        and _status(gate) == "evaluated"
        and _one(gate.get("passed"))
    )


def _check(
    value: Any,
    threshold: Any,
    operator: str,
) -> dict[str, Any]:
    if operator == "==":
        passed = value == threshold
    elif operator == "<=":
        passed = _finite(value) and float(value) <= float(threshold)
    elif operator == ">=":
        passed = _finite(value) and float(value) >= float(threshold)
    else:  # pragma: no cover - construction invariant
        raise ValueError(f"unknown gate operator: {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(passed),
    }


def _gate(
    name: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    claim_scope: str,
    **fields: Any,
) -> dict[str, Any]:
    passed = bool(checks) and all(_one(value.get("passed")) for value in checks.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "evaluation_status": "evaluated",
        "claim_scope": claim_scope,
        "subchecks": {str(key): dict(value) for key, value in checks.items()},
        "passed": int(passed),
        **fields,
        **NO_WORK,
    }


def not_evaluated_gate(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": str(stage),
        "evaluation_status": "not_evaluated",
        "reason": str(reason),
        "subchecks": {},
        "passed": 0,
        **NO_WORK,
    }


def execution_failed_gate(
    stage: str,
    *,
    failure_domain: str,
    failure_code: str,
    error_type: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": str(stage),
        "evaluation_status": "execution_failed",
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "failure_domain": str(failure_domain),
        "failure_code": str(failure_code),
        "error_type": str(error_type),
        "error": str(error),
        "subchecks": {},
        "passed": 0,
        **NO_WORK,
    }


def evaluate_recovery_preflight(metrics: Mapping[str, Any]) -> dict[str, Any]:
    t = HaarPowerRecoveryThresholds()
    expected_one = (
        "control_provenance_pass",
        "parent_registry_verified_pass",
        "parent_sources_immutable_pass",
        "parent_scientific_config_pass",
        "parent_preflight_pass",
        "parent_coupling_pass",
        "parent_pilot_execution_failure_pass",
        "parent_failure_code_pass",
        "parent_shard_layout_pass",
        "parent_shard_hashes_pass",
        "parent_shard_chains_pass",
        "parent_schedule_location_pass",
        "parent_antithetic_absent_pass",
        "parent_panel_b_absent_pass",
        "parent_selection_absent_pass",
        "parent_no_work_pass",
        "transitive_provenance_pass",
        "path_plan_frozen_pass",
    )
    checks = {name: _check(metrics.get(name), 1, "==") for name in expected_one}
    checks.update(
        {
            "root_seed": _check(metrics.get("root_seed"), t.root_seed, "=="),
            "parent_registry_record_count": _check(
                metrics.get("parent_registry_record_count"),
                t.parent_registry_record_count,
                "==",
            ),
            "parent_source_count": _check(
                metrics.get("parent_source_count"), t.parent_source_count, "=="
            ),
            "parent_main_shards": _check(
                metrics.get("parent_main_shards"), t.parent_main_shards, "=="
            ),
            "parent_reference_shards": _check(
                metrics.get("parent_reference_shards"),
                t.parent_reference_shards,
                "==",
            ),
        }
    )
    result = _gate(
        "preflight",
        checks,
        claim_scope="immutable parent, sealed path plan, and completed nested shards",
    )
    result.update(
        provenance_valid=int(
            all(
                checks[name]["passed"]
                for name in (
                    "control_provenance_pass",
                    "parent_registry_verified_pass",
                    "parent_sources_immutable_pass",
                    "transitive_provenance_pass",
                )
            )
        ),
        schedule_binding_valid=int(
            checks["parent_schedule_location_pass"]["passed"]
        ),
        shard_evidence_valid=int(
            checks["parent_shard_layout_pass"]["passed"]
            and checks["parent_shard_hashes_pass"]["passed"]
            and checks["parent_shard_chains_pass"]["passed"]
        ),
    )
    return result


def evaluate_nested_replay(metrics: Mapping[str, Any]) -> dict[str, Any]:
    t = HaarPowerRecoveryThresholds()
    expected_one = (
        "canonical_schedule_binding_pass",
        "parent_read_only_pass",
        "no_nested_gpu_recomputation_pass",
        "observable_replay_pass",
        "candidate_reconstruction_pass",
        "candidate_numerical_health_pass",
        "candidate_resource_health_pass",
        "frozen_no_nominee_pass",
        "raw_endpoint_authorizing_pass",
        "dynkin_advisory_only_pass",
        "shard_chain_pass",
        "state_updates_device_resident_pass",
        "antithetic_path_ids_untouched_pass",
    )
    checks = {name: _check(metrics.get(name), 1, "==") for name in expected_one}
    checks.update(
        {
            "shard_count": _check(
                metrics.get("shard_count"), t.recovered_shards, "=="
            ),
            "transition_count": _check(
                metrics.get("transition_count"),
                t.recovered_transition_count,
                "==",
            ),
            "certificate_fraction": _check(
                metrics.get("certificate_fraction"), 1.0, "=="
            ),
            "fallback_count": _check(
                metrics.get("fallback_count"), t.recovered_fallback_count, "=="
            ),
            "fallback_fraction": _check(
                metrics.get("fallback_fraction"),
                t.maximum_fallback_fraction,
                "<=",
            ),
            "fallback_cost_fraction": _check(
                metrics.get("fallback_cost_fraction"),
                t.maximum_fallback_cost_fraction,
                "<=",
            ),
            "mass_error": _check(
                metrics.get("mass_error"), t.maximum_mass_error, "<="
            ),
            "peak_memory_fraction": _check(
                metrics.get("peak_memory_fraction"),
                t.maximum_peak_memory_fraction,
                "<=",
            ),
            "conservative_rate": _check(
                metrics.get("conservative_rate"), t.minimum_rate, ">="
            ),
            "candidate_count": _check(
                metrics.get("candidate_count"),
                t.expected_nested_candidates,
                "==",
            ),
            "eligible_candidate_count": _check(
                metrics.get("eligible_candidate_count"),
                t.expected_nested_eligible_candidates,
                "==",
            ),
            "selection_status": _check(
                metrics.get("selection_status"),
                "panel_a_no_eligible_design",
                "==",
            ),
            "uncertified_count": _check(
                metrics.get("uncertified_count"), 0, "=="
            ),
            "forbidden_event_count": _check(
                metrics.get("forbidden_event_count"), 0, "=="
            ),
        }
    )
    result = _gate(
        "replay",
        checks,
        claim_scope="read-only reconstruction of sealed nested-Haar panel A",
    )
    result.update(
        replay_valid=int(result["passed"]),
        antithetic_panel_a_authorized=int(result["passed"]),
        nested_panel_a_nominated=0,
    )
    return result


def evaluate_antithetic_pilot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    t = HaarPowerRecoveryThresholds()
    expected_one = (
        "plans_frozen_pass",
        "profile_order_pass",
        "panel_nonregeneration_pass",
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
        "production_authorizing_pass",
        "antithetic_panel_a_nominated",
        "antithetic_panel_b_opened",
        "antithetic_panels_agree",
    )
    checks = {name: _check(metrics.get(name), 1, "==") for name in expected_one}
    checks.update(
        {
            "selected_profile": _check(
                metrics.get("selected_profile"),
                "pairwise_haar_antithetic",
                "==",
            ),
            "main_paths": _check(
                metrics.get("main_paths"), t.antithetic_main_paths, "=="
            ),
            "reference_paths": _check(
                metrics.get("reference_paths"),
                t.antithetic_reference_paths,
                "==",
            ),
            "combined_main_half_width": _check(
                metrics.get("combined_main_half_width"),
                t.maximum_main_half_width,
                "<=",
            ),
            "combined_generator_reference_half_width": _check(
                metrics.get("combined_generator_reference_half_width"),
                t.maximum_reference_half_width,
                "<=",
            ),
            "combined_reference_stability_half_width": _check(
                metrics.get("combined_reference_stability_half_width"),
                t.maximum_reference_half_width,
                "<=",
            ),
            "projected_hours": _check(
                metrics.get("projected_hours"),
                t.maximum_projected_hours,
                "<=",
            ),
            "minimum_rate": _check(
                metrics.get("minimum_rate"), t.minimum_rate, ">="
            ),
            "certificate_fraction": _check(
                metrics.get("certificate_fraction"), 1.0, "=="
            ),
            "fallback_fraction": _check(
                metrics.get("fallback_fraction"),
                t.maximum_fallback_fraction,
                "<=",
            ),
            "fallback_cost_fraction": _check(
                metrics.get("fallback_cost_fraction"),
                t.maximum_fallback_cost_fraction,
                "<=",
            ),
            "peak_memory_fraction": _check(
                metrics.get("peak_memory_fraction"),
                t.maximum_peak_memory_fraction,
                "<=",
            ),
            "forbidden_event_count": _check(
                metrics.get("forbidden_event_count"), 0, "=="
            ),
        }
    )
    result = _gate(
        "pilot",
        checks,
        claim_scope="sealed pairwise-antithetic refinement-power feasibility",
    )
    result.update(
        numerically_valid=int(
            checks["executed_panels_numerically_valid_pass"]["passed"]
            and checks["certificate_fraction"]["passed"]
            and checks["forbidden_event_count"]["passed"]
        ),
        resource_valid=int(
            checks["projected_hours"]["passed"]
            and checks["minimum_rate"]["passed"]
            and checks["peak_memory_fraction"]["passed"]
        ),
        power_valid=int(
            checks["combined_main_half_width"]["passed"]
            and checks["combined_generator_reference_half_width"]["passed"]
            and checks["combined_reference_stability_half_width"]["passed"]
        ),
        panel_a_nominated=int(
            checks["antithetic_panel_a_nominated"]["passed"]
        ),
        panel_b_opened=int(checks["antithetic_panel_b_opened"]["passed"]),
        panels_agree=int(checks["antithetic_panels_agree"]["passed"]),
    )
    return result


def _failure_decision(
    gate: Mapping[str, Any],
) -> tuple[HaarPowerRecoveryDecision, str]:
    domain = str(gate.get("failure_domain", "")).lower()
    stage = str(gate.get("gate", "")).lower()
    if "provenance" in domain:
        return (
            HaarPowerRecoveryDecision.CONTROL_PROVENANCE_INVALID,
            "repair the immutable 197-artifact parent binding",
        )
    if stage == "preflight" or "schedule_binding" in domain:
        return (
            HaarPowerRecoveryDecision.PANEL_SCHEDULE_BINDING_INVALID,
            "repair canonical parent schedule extraction",
        )
    if stage == "replay" or "replay" in domain:
        return (
            HaarPowerRecoveryDecision.NESTED_PANEL_REPLAY_INVALID,
            "repair exact read-only nested-panel reconstruction",
        )
    if "resource" in domain or "comput" in domain:
        return (
            HaarPowerRecoveryDecision.ANTITHETIC_COUPLING_COMPUTATIONALLY_INFEASIBLE,
            "repair exact antithetic execution within frozen resources",
        )
    return (
        HaarPowerRecoveryDecision.ANTITHETIC_SCHEDULER_INVALID,
        "repair the antithetic scheduler or sealed-panel orchestration",
    )


def decide_recovery_workflow(
    *,
    provenance: Mapping[str, Any] | bool,
    preflight_gate: Mapping[str, Any] | None,
    replay_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a closed outcome, or an explicit pending-stage record."""

    if provenance is False or (
        isinstance(provenance, Mapping)
        and _status(provenance) == "evaluated"
        and not _one(provenance.get("passed"))
    ):
        decision: HaarPowerRecoveryDecision | None = (
            HaarPowerRecoveryDecision.CONTROL_PROVENANCE_INVALID
        )
        action = "repair the immutable 197-artifact parent binding"
    elif _status(preflight_gate) == "execution_failed":
        decision, action = _failure_decision(preflight_gate)
    elif _status(preflight_gate) != "evaluated":
        decision = None
        action = "run immutable-parent preflight"
    elif not _passed(preflight_gate):
        decision = (
            HaarPowerRecoveryDecision.CONTROL_PROVENANCE_INVALID
            if not _one(preflight_gate.get("provenance_valid"))
            else HaarPowerRecoveryDecision.PANEL_SCHEDULE_BINDING_INVALID
        )
        action = "repair parent provenance or canonical schedule binding"
    elif _status(replay_gate) == "execution_failed":
        decision, action = _failure_decision(replay_gate)
    elif _status(replay_gate) != "evaluated":
        decision = None
        action = "replay the immutable nested panel A"
    elif not _passed(replay_gate):
        decision = HaarPowerRecoveryDecision.NESTED_PANEL_REPLAY_INVALID
        action = "repair exact read-only nested-panel reconstruction"
    elif _status(pilot_gate) == "execution_failed":
        decision, action = _failure_decision(pilot_gate)
    elif _status(pilot_gate) != "evaluated":
        decision = None
        action = "run sealed pairwise-antithetic panel A"
    elif _passed(pilot_gate):
        decision = (
            HaarPowerRecoveryDecision.EXACT_HAAR_HIERARCHICAL_REFINEMENT_COUPLING_FEASIBLE
        )
        action = "plan a fresh production Strang-refinement experiment"
    elif not _one(pilot_gate.get("numerically_valid", 0)):
        decision = HaarPowerRecoveryDecision.ANTITHETIC_SCHEDULER_INVALID
        action = "repair exact antithetic execution"
    elif not _one(pilot_gate.get("resource_valid", 0)):
        decision = (
            HaarPowerRecoveryDecision.ANTITHETIC_COUPLING_COMPUTATIONALLY_INFEASIBLE
        )
        action = "repair exact antithetic execution within frozen resources"
    elif not _one(pilot_gate.get("panel_a_nominated", 0)):
        decision = HaarPowerRecoveryDecision.HIERARCHICAL_POWER_INFEASIBLE
        action = "retain sealed evidence and do not weaken thresholds"
    else:
        decision = HaarPowerRecoveryDecision.HIERARCHICAL_PANELS_DISAGREE
        action = "retain both sealed panels and do not select a design"

    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated" if decision is not None else "pending",
        "decision": None if decision is None else decision.value,
        "recommended_next_action": action,
        "preflight_authorized": int(_status(preflight_gate) != "evaluated"),
        "replay_authorized": int(
            _passed(preflight_gate) and _status(replay_gate) != "evaluated"
        ),
        "antithetic_panel_a_authorized": int(
            _passed(replay_gate) and _status(pilot_gate) != "evaluated"
        ),
        "production_refinement_patch_authorized": int(
            decision
            is HaarPowerRecoveryDecision.EXACT_HAAR_HIERARCHICAL_REFINEMENT_COUPLING_FEASIBLE
        ),
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        **NO_WORK,
    }


def evaluate_recovery_workflow(
    *,
    provenance: Mapping[str, Any] | bool,
    preflight_gate: Mapping[str, Any] | None,
    replay_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    order = ("preflight", "replay", "pilot")
    if require_gate not in REQUIRED_GATES:
        raise ValueError(f"unknown required gate: {require_gate}")
    components = {
        "preflight": dict(
            preflight_gate or not_evaluated_gate("preflight", "not run")
        ),
        "replay": dict(replay_gate or not_evaluated_gate("replay", "not run")),
        "pilot": dict(pilot_gate or not_evaluated_gate("pilot", "not run")),
    }
    required = (
        ()
        if require_gate == "none"
        else order[: order.index(require_gate) + 1]
    )
    provenance_pass = (
        provenance is True
        or (
            isinstance(provenance, Mapping)
            and _status(provenance) == "evaluated"
            and _one(provenance.get("passed"))
        )
    )
    passed = bool(
        provenance_pass
        and all(_passed(components[name]) for name in required)
    )
    decision = decide_recovery_workflow(
        provenance=provenance,
        preflight_gate=components["preflight"],
        replay_gate=components["replay"],
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
        "thresholds": HaarPowerRecoveryThresholds().to_dict(),
        **NO_WORK,
    }


__all__ = [
    "HaarPowerRecoveryDecision",
    "HaarPowerRecoveryThresholds",
    "NO_WORK",
    "REQUIRED_GATES",
    "STAGES",
    "decide_recovery_workflow",
    "evaluate_antithetic_pilot",
    "evaluate_nested_replay",
    "evaluate_recovery_preflight",
    "evaluate_recovery_workflow",
    "execution_failed_gate",
    "not_evaluated_gate",
]
