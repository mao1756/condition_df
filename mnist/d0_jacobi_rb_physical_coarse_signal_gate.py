"""Fail-closed gates for the exact physical coarse-signal witness.

The gate validates evidence production.  A scientifically valid witness may
detect a signal, resolve it below the preregistered scale, or remain
inconclusive; none of those outcomes changes the exact Jacobi/RB target.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


GATE_SCHEMA = "d0-jacobi-rb-physical-coarse-signal-gate-v1"
RESOLUTION = 5.0e-4
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}
CLAIM_FLAGS = {
    "physical_coarse_signal_witness_authorized": 0,
    "neural_training_authorized": 0,
    "reverse_sampling_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "full_state_conditional_mean_zero_proven": 0,
}


def _check(
    value: Any,
    *,
    operator: str = "==",
    threshold: Any = 1,
) -> dict[str, Any]:
    if operator == "==":
        passed = value == threshold
    elif operator == "<=":
        passed = math.isfinite(float(value)) and float(value) <= float(threshold)
    elif operator == ">=":
        passed = math.isfinite(float(value)) and float(value) >= float(threshold)
    else:
        raise ValueError(f"unsupported gate operator {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _gate(name: str, checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passed = bool(checks) and all(
        int(item.get("passed", 0)) == 1 for item in checks.values()
    )
    return {
        "schema": GATE_SCHEMA,
        "schema_version": 1,
        "gate": str(name),
        "evaluation_status": "evaluated",
        "passed": int(passed),
        "subchecks": {key: dict(value) for key, value in checks.items()},
        **CLAIM_FLAGS,
        **NO_WORK,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema": GATE_SCHEMA,
        "schema_version": 1,
        "gate": str(name),
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "reason": str(reason),
        "subchecks": {},
        **CLAIM_FLAGS,
        **NO_WORK,
    }


def execution_failed_gate(
    name: str,
    *,
    failure_code: str,
    failure_domain: str,
    message: str,
) -> dict[str, Any]:
    return {
        "schema": GATE_SCHEMA,
        "schema_version": 1,
        "gate": str(name),
        "evaluation_status": "execution_failed",
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "passed": 0,
        "failure_code": str(failure_code),
        "failure_domain": str(failure_domain),
        "message": str(message),
        "subchecks": {},
        **CLAIM_FLAGS,
        **NO_WORK,
    }


def evaluate_preflight(metrics: Mapping[str, Any]) -> dict[str, Any]:
    equality = (
        "physical_parent_verified",
        "zero_signal_parent_verified",
        "bayes_parent_verified",
        "all_parent_registries_verified",
        "all_parent_sources_verified",
        "no_parent_artifact_reused_in_estimate",
        "path_plan_valid",
        "path_roles_disjoint",
        "statistic_plan_frozen",
        "bayes_teacher_all_pairs_detected",
        "bayes_null_all_pairs_cover_zero",
        "bayes_replay_whole_path_only",
        "old_physical_forecast_nonauthorizing",
        "benchmark_complete_capture_path",
        "benchmark_certificate_fraction_one",
        "benchmark_forbidden_events_zero",
        "benchmark_states_finite",
        "benchmark_target_finite",
        "benchmark_mass_conservation_pass",
        "benchmark_raw_targets_not_persisted",
    )
    checks: dict[str, dict[str, Any]] = {
        name: _check(metrics.get(name)) for name in equality
    }
    checks.update(
        {
            "projected_two_panel_hours": _check(
                metrics.get("projected_two_panel_hours", math.inf),
                operator="<=",
                threshold=24.0,
            ),
            "transitions_per_second": _check(
                metrics.get("transitions_per_second", -math.inf),
                operator=">=",
                threshold=1300.0,
            ),
            "peak_memory_fraction": _check(
                metrics.get("peak_memory_fraction", math.inf),
                operator="<=",
                threshold=0.80,
            ),
            "fallback_fraction": _check(
                metrics.get("fallback_fraction", math.inf),
                operator="<=",
                threshold=1.0e-4,
            ),
            "fallback_time_fraction": _check(
                metrics.get("fallback_time_fraction", math.inf),
                operator="<=",
                threshold=0.10,
            ),
            "maximum_mass_error": _check(
                metrics.get("maximum_mass_error", math.inf),
                operator="<=",
                threshold=2.0e-12,
            ),
        }
    )
    return _gate("preflight", checks)


def evaluate_panel(
    metrics: Mapping[str, Any],
    *,
    panel: str,
    prerequisite_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prerequisite_pass = int(
        prerequisite_gate is not None
        and prerequisite_gate.get("evaluation_status") == "evaluated"
        and int(prerequisite_gate.get("passed", 0)) == 1
    )
    equality = (
        "path_plan_binding_pass",
        "statistic_plan_binding_pass",
        "panel_role_isolated",
        "panel_sealed",
        "all_groups_complete",
        "shard_chains_valid",
        "resume_state_hashes_valid",
        "path_count_pass",
        "cell_shape_pass",
        "selected_step_coverage_pass",
        "eight_observations_per_cell_pass",
        "cell_means_finite",
        "cell_means_persistence_audit_pass",
        "certificate_fraction_one",
        "forbidden_events_zero",
        "state_updates_device_resident",
        "target_modification_count_zero",
        "raw_target_observations_not_persisted",
    )
    checks: dict[str, dict[str, Any]] = {
        "prerequisite_gate": _check(prerequisite_pass),
        **{name: _check(metrics.get(name)) for name in equality},
        "path_count": _check(metrics.get("path_count"), threshold=64),
        "transition_count": _check(
            metrics.get("transition_count"), threshold=89_915_392
        ),
        "maximum_mass_error": _check(
            metrics.get("maximum_mass_error", math.inf),
            operator="<=",
            threshold=2.0e-12,
        ),
        "fallback_fraction": _check(
            metrics.get("fallback_fraction", math.inf),
            operator="<=",
            threshold=1.0e-4,
        ),
        "fallback_time_fraction": _check(
            metrics.get("fallback_time_fraction", math.inf),
            operator="<=",
            threshold=0.10,
        ),
        "peak_memory_fraction": _check(
            metrics.get("peak_memory_fraction", math.inf),
            operator="<=",
            threshold=0.80,
        ),
        "transitions_per_second": _check(
            metrics.get("transitions_per_second", -math.inf),
            operator=">=",
            threshold=1300.0,
        ),
    }
    return _gate(panel, checks)


def evaluate_witness(
    metrics: Mapping[str, Any],
    *,
    panel_a_gate: Mapping[str, Any] | None,
    panel_b_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    def passed(gate: Mapping[str, Any] | None) -> int:
        return int(
            gate is not None
            and gate.get("evaluation_status") == "evaluated"
            and int(gate.get("passed", 0)) == 1
        )

    checks = {
        "panel_a_gate": _check(passed(panel_a_gate)),
        "panel_b_gate": _check(passed(panel_b_gate)),
        "joint_analysis_seal_valid": _check(
            metrics.get("joint_analysis_seal_valid")
        ),
        "panels_opened_once": _check(metrics.get("panels_opened_once")),
        "panel_hashes_unchanged": _check(metrics.get("panel_hashes_unchanged")),
        "panel_path_sets_disjoint": _check(
            metrics.get("panel_path_sets_disjoint")
        ),
        "cell_count": _check(metrics.get("cell_count"), threshold=10_976),
        "panel_a_path_count": _check(
            metrics.get("panel_a_path_count"), threshold=64
        ),
        "panel_b_path_count": _check(
            metrics.get("panel_b_path_count"), threshold=64
        ),
        "estimator_algebra_pass": _check(metrics.get("estimator_algebra_pass")),
        "bootstrap_whole_path_only": _check(
            metrics.get("bootstrap_whole_path_only")
        ),
        "bootstrap_replicates": _check(
            metrics.get("bootstrap_replicates"), threshold=50_000
        ),
        "bootstrap_finite": _check(metrics.get("bootstrap_finite")),
        "influence_components_finite": _check(
            metrics.get("influence_components_finite")
        ),
        "welch_bound_finite": _check(metrics.get("welch_bound_finite")),
        "one_sided_99_percent_bounds": _check(
            metrics.get("one_sided_99_percent_bounds")
        ),
        "negative_values_not_truncated": _check(
            metrics.get("negative_values_not_truncated")
        ),
        "decision_partition_pass": _check(
            metrics.get("decision_partition_pass")
        ),
        "old_physical_data_excluded": _check(
            metrics.get("old_physical_data_excluded")
        ),
        "no_training_performed": _check(metrics.get("no_training_performed")),
        "no_sampling_performed": _check(metrics.get("no_sampling_performed")),
    }
    return _gate("witness", checks)


def witness_decision(
    *,
    preflight_gate: Mapping[str, Any] | None,
    panel_a_gate: Mapping[str, Any] | None,
    panel_b_gate: Mapping[str, Any] | None,
    witness_gate: Mapping[str, Any] | None,
    scientific_outcome: str | None,
) -> dict[str, Any]:
    def evaluated(gate: Mapping[str, Any] | None) -> bool:
        return bool(gate and gate.get("evaluation_status") == "evaluated")

    def passed(gate: Mapping[str, Any] | None) -> bool:
        return bool(evaluated(gate) and int(gate.get("passed", 0)) == 1)

    def failed_checks(gate: Mapping[str, Any]) -> set[str]:
        return {
            name
            for name, check in gate.get("subchecks", {}).items()
            if int(check.get("passed", 0)) == 0
        }

    def panel_failure(
        gate: Mapping[str, Any], *, panel_name: str
    ) -> tuple[str, str]:
        failed = failed_checks(gate)
        if failed.intersection(
            {
                "transitions_per_second",
                "peak_memory_fraction",
                "fallback_time_fraction",
            }
        ):
            return (
                "physical_coarse_signal_computationally_infeasible",
                f"repair exact {panel_name} capture scheduling without changing it",
            )
        if failed.intersection(
            {
                "certificate_fraction_one",
                "forbidden_events_zero",
                "maximum_mass_error",
                "fallback_fraction",
                "state_updates_device_resident",
                "target_modification_count_zero",
            }
        ):
            return (
                "physical_coarse_signal_numerically_unresolved",
                f"repair exact {panel_name} numerical execution without regenerating it",
            )
        return (
            "physical_coarse_signal_panel_integrity_invalid",
            f"repair or exactly resume {panel_name} without resizing it",
        )

    execution_failure: tuple[str, Mapping[str, Any]] | None = next(
        (
            (name, gate)
            for name, gate in (
                ("preflight", preflight_gate),
                ("panel-a", panel_a_gate),
                ("panel-b", panel_b_gate),
                ("witness", witness_gate),
            )
            if gate is not None
            and gate.get("evaluation_status") == "execution_failed"
        ),
        None,
    )
    if execution_failure is not None:
        failed_stage, failed_gate = execution_failure
        domain = str(failed_gate.get("failure_domain", "workflow_execution"))
        if "provenance" in domain:
            decision = "control_provenance_invalid"
        elif "resource" in domain or "comput" in domain:
            decision = "physical_coarse_signal_computationally_infeasible"
        elif any(
            token in domain for token in ("numerical", "certificate", "transition")
        ):
            decision = "physical_coarse_signal_numerically_unresolved"
        elif failed_stage in {"panel-a", "panel-b"}:
            decision = "physical_coarse_signal_panel_integrity_invalid"
        elif failed_stage == "witness":
            decision = "physical_coarse_signal_estimator_invalid"
        else:
            decision = "physical_coarse_signal_preflight_invalid"
        action = (
            "repair the recorded execution failure before resuming: "
            f"{failed_gate.get('failure_code', 'unknown_failure')}"
        )
    elif not evaluated(preflight_gate):
        decision = "ready_for_preflight"
        action = "run the immutable-parent, estimator-control, and resource preflight"
    elif not passed(preflight_gate):
        failed = failed_checks(preflight_gate)
        if any(name.startswith("bayes_") for name in failed):
            decision = "coarse_signal_estimator_controls_invalid"
            action = "repair the estimator controls before generating physical panels"
        elif failed.intersection(
            {
                "projected_two_panel_hours",
                "transitions_per_second",
                "peak_memory_fraction",
                "fallback_time_fraction",
            }
        ):
            decision = "physical_coarse_signal_computationally_infeasible"
            action = "repair exact capture scheduling without changing the target"
        elif any("parent" in name for name in failed):
            decision = "control_provenance_invalid"
            action = "repair immutable parent provenance"
        elif failed.intersection(
            {
                "benchmark_certificate_fraction_one",
                "benchmark_forbidden_events_zero",
                "benchmark_states_finite",
                "benchmark_target_finite",
                "benchmark_mass_conservation_pass",
                "fallback_fraction",
                "maximum_mass_error",
            }
        ):
            decision = "physical_coarse_signal_numerically_unresolved"
            action = "repair exact benchmark numerics before generating panels"
        else:
            decision = "physical_coarse_signal_preflight_invalid"
            action = "repair preflight integrity before generating panels"
    elif not evaluated(panel_a_gate):
        decision = "ready_for_panel_a"
        action = "generate and seal fresh physical panel A"
    elif not passed(panel_a_gate):
        decision, action = panel_failure(panel_a_gate, panel_name="panel A")
    elif not evaluated(panel_b_gate):
        decision = "ready_for_panel_b"
        action = "generate and seal independent physical panel B"
    elif not passed(panel_b_gate):
        decision, action = panel_failure(panel_b_gate, panel_name="panel B")
    elif not evaluated(witness_gate):
        decision = "ready_for_witness_analysis"
        action = "open the sealed panels once and run the frozen joint analysis"
    elif not passed(witness_gate):
        decision = "physical_coarse_signal_estimator_invalid"
        action = "repair report-only inference without changing either panel"
    else:
        allowed = {
            "exact_physical_coarse_signal_detected",
            "coarse_signal_below_preregistered_resolution",
            "physical_coarse_signal_inconclusive",
        }
        if scientific_outcome not in allowed:
            decision = "physical_coarse_signal_estimator_invalid"
            action = "repair the scientific decision partition"
        else:
            decision = str(scientific_outcome)
            action = {
                "exact_physical_coarse_signal_detected": (
                    "plan a coarse-baseline plus exact-RB residual learner with "
                    "unweighted MSE against the unchanged exact label"
                ),
                "coarse_signal_below_preregistered_resolution": (
                    "record that only the preregistered coarse projection is "
                    "below resolution; do not infer a zero full conditional mean"
                ),
                "physical_coarse_signal_inconclusive": (
                    "retain both sealed panels and plan a separately powered "
                    "model-free witness without changing the exact target"
                ),
            }[str(scientific_outcome)]
    return {
        "schema": "d0-jacobi-rb-physical-coarse-signal-decision-v1",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "decision": decision,
        "recommended_next_action": action,
        "scientific_outcome": scientific_outcome,
        "resolution": RESOLUTION,
        "claim_scope": (
            "coarse conditional-mean witness for one frozen image under the "
            "exact K=512 certified Jacobi/Rao-Blackwell split chain"
        ),
        **CLAIM_FLAGS,
        **NO_WORK,
    }


__all__ = [
    "CLAIM_FLAGS",
    "GATE_SCHEMA",
    "NO_WORK",
    "RESOLUTION",
    "evaluate_panel",
    "evaluate_preflight",
    "evaluate_witness",
    "execution_failed_gate",
    "not_evaluated_gate",
    "witness_decision",
]
