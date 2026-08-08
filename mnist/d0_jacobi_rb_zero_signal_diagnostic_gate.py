"""Fail-closed gates for the sealed Jacobi/RB zero-signal diagnostic."""

from __future__ import annotations

import math
from typing import Any, Mapping


CLAIM_FLAGS = {
    "new_scientific_gate_authorized": 0,
    "full_dataset_training_authorized": 0,
    "larger_training_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "sampling_authorized": 0,
    "reverse_sampling_authorized": 0,
    "production_refinement_authorized": 0,
    "conditional_mean_identically_zero_proven": 0,
    "population_signal_absence_proven": 0,
}


def _check(value: Any, *, operator: str = "==", threshold: Any = 1) -> dict[str, Any]:
    if operator == "==":
        passed = value == threshold
    elif operator == "<=":
        passed = float(value) <= float(threshold)
    else:
        raise ValueError(f"unsupported gate operator {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _gate(name: str, checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passed = bool(checks) and all(int(item.get("passed", 0)) == 1 for item in checks.values())
    return {
        "schema": "d0-jacobi-rb-zero-signal-diagnostic-gate-v1",
        "schema_version": 1,
        "gate": name,
        "evaluation_status": "evaluated",
        "passed": int(passed),
        "subchecks": {key: dict(value) for key, value in checks.items()},
        **CLAIM_FLAGS,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "production_refinement_performed": 0,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema": "d0-jacobi-rb-zero-signal-diagnostic-gate-v1",
        "schema_version": 1,
        "gate": name,
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "reason": str(reason),
        "subchecks": {},
        **CLAIM_FLAGS,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "production_refinement_performed": 0,
    }


def evaluate_zero_signal_preflight(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        name: _check(metrics.get(name))
        for name in (
            "parent_registry_verified",
            "parent_terminal_scope_verified",
            "parent_negative_decision_verified",
            "parent_only_confirmation_failure_verified",
            "selected_model_binding_verified",
            "metadata_baseline_binding_verified",
            "all_three_cache_bindings_verified",
            "confirmation_opened_once_verified",
            "no_parent_mutation_planned",
            "no_new_data_planned",
            "no_training_planned",
            "no_tuning_planned",
            "no_sampling_planned",
            "posthoc_non_authorizing",
        )
    }
    return _gate("preflight", checks)


def evaluate_zero_signal_analysis(
    metrics: Mapping[str, Any],
    *,
    preflight_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {
        "preflight_gate": _check(
            int(
                preflight_gate is not None
                and preflight_gate.get("evaluation_status") == "evaluated"
                and int(preflight_gate.get("passed", 0)) == 1
            )
        ),
        "split_count": _check(metrics.get("split_count"), threshold=3),
        "train_path_count": _check(metrics.get("train_path_count"), threshold=8),
        "validation_path_count": _check(
            metrics.get("validation_path_count"), threshold=8
        ),
        "confirmation_path_count": _check(
            metrics.get("confirmation_path_count"), threshold=8
        ),
        "all_predictions_finite": _check(metrics.get("all_predictions_finite")),
        "all_metrics_finite": _check(metrics.get("all_metrics_finite")),
        "checkpoint_replay_verified": _check(
            metrics.get("checkpoint_replay_verified")
        ),
        "confirmation_metrics_reproduced": _check(
            metrics.get("confirmation_metrics_reproduced")
        ),
        "validation_metrics_reproduced": _check(
            metrics.get("validation_metrics_reproduced")
        ),
        "decomposition_identity_max_abs_error": _check(
            metrics.get("decomposition_identity_max_abs_error", math.inf),
            operator="<=",
            threshold=1.0e-12,
        ),
        "metadata_identity_max_abs_error": _check(
            metrics.get("metadata_identity_max_abs_error", math.inf),
            operator="<=",
            threshold=1.0e-12,
        ),
        "whole_path_only_bootstrap": _check(
            metrics.get("whole_path_only_bootstrap")
        ),
        "coarse_signal_pair_count": _check(
            metrics.get("coarse_signal_pair_count"), threshold=3
        ),
        "coarse_observations_per_split_cell": _check(
            metrics.get("coarse_observations_per_split_cell"), threshold=64
        ),
        "coarse_bootstrap_whole_path_only": _check(
            metrics.get("coarse_bootstrap_whole_path_only")
        ),
        "coarse_signal_all_finite": _check(
            metrics.get("coarse_signal_all_finite")
        ),
        "parent_artifacts_read_only": _check(metrics.get("parent_artifacts_read_only")),
        "no_new_data_generated": _check(metrics.get("no_new_data_generated")),
        "no_training_performed": _check(metrics.get("no_training_performed")),
        "no_tuning_performed": _check(metrics.get("no_tuning_performed")),
        "no_sampling_performed": _check(metrics.get("no_sampling_performed")),
        "posthoc_non_authorizing": _check(metrics.get("posthoc_non_authorizing")),
    }
    return _gate("analysis", checks)


def zero_signal_decision(
    *,
    preflight_gate: Mapping[str, Any] | None,
    analysis_gate: Mapping[str, Any] | None,
    conclusion: Mapping[str, Any] | None,
) -> dict[str, Any]:
    preflight_evaluated = bool(
        preflight_gate
        and preflight_gate.get("evaluation_status") == "evaluated"
    )
    preflight_passed = bool(
        preflight_gate
        and preflight_evaluated
        and int(preflight_gate.get("passed", 0)) == 1
    )
    analysis_evaluated = bool(
        analysis_gate and analysis_gate.get("evaluation_status") == "evaluated"
    )
    analysis_passed = bool(
        analysis_gate
        and analysis_evaluated
        and int(analysis_gate.get("passed", 0)) == 1
    )
    if not preflight_evaluated:
        decision = "diagnostic_not_evaluated"
        action = "run the immutable-parent diagnostic preflight"
    elif not preflight_passed:
        decision = "parent_scope_invalid"
        action = "repair the immutable parent binding before interpretation"
    elif not analysis_evaluated:
        decision = "ready_for_zero_signal_analysis"
        action = "run the report-only analysis on the frozen selected model"
    elif not analysis_passed:
        decision = "diagnostic_execution_invalid"
        action = "repair the report-only diagnostic without changing the parent experiment"
    else:
        decision = "zero_signal_diagnostic_complete"
        action = (
            "retain the sealed no-signal result; use this decomposition only to "
            "plan a separately preregistered theoretical or signal-identifiability study"
        )
    return {
        "schema": "d0-jacobi-rb-zero-signal-diagnostic-decision-v1",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "decision": decision,
        "recommended_next_action": action,
        "diagnostic_conclusion": (
            None if conclusion is None else conclusion.get("conclusion")
        ),
        "claim_scope": (
            "post-hoc quadratic-risk decomposition of the frozen selected model "
            "on the sealed exact-K=512 one-image caches"
        ),
        **CLAIM_FLAGS,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "production_refinement_performed": 0,
    }


__all__ = [
    "CLAIM_FLAGS",
    "evaluate_zero_signal_analysis",
    "evaluate_zero_signal_preflight",
    "not_evaluated_gate",
    "zero_signal_decision",
]
