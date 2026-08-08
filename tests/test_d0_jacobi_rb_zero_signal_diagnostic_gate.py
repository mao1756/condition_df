from __future__ import annotations

from mnist.d0_jacobi_rb_zero_signal_diagnostic_gate import (
    evaluate_zero_signal_analysis,
    evaluate_zero_signal_preflight,
    zero_signal_decision,
)


def _preflight_metrics() -> dict[str, int]:
    return {
        name: 1
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


def _analysis_metrics() -> dict[str, int | float]:
    return {
        "split_count": 3,
        "train_path_count": 8,
        "validation_path_count": 8,
        "confirmation_path_count": 8,
        "all_predictions_finite": 1,
        "all_metrics_finite": 1,
        "checkpoint_replay_verified": 1,
        "confirmation_metrics_reproduced": 1,
        "validation_metrics_reproduced": 1,
        "decomposition_identity_max_abs_error": 1.0e-15,
        "metadata_identity_max_abs_error": 1.0e-15,
        "whole_path_only_bootstrap": 1,
        "coarse_signal_pair_count": 3,
        "coarse_observations_per_split_cell": 64,
        "coarse_bootstrap_whole_path_only": 1,
        "coarse_signal_all_finite": 1,
        "parent_artifacts_read_only": 1,
        "no_new_data_generated": 1,
        "no_training_performed": 1,
        "no_tuning_performed": 1,
        "no_sampling_performed": 1,
        "posthoc_non_authorizing": 1,
    }


def test_diagnostic_gates_pass_without_authorizing_science() -> None:
    preflight = evaluate_zero_signal_preflight(_preflight_metrics())
    analysis = evaluate_zero_signal_analysis(
        _analysis_metrics(), preflight_gate=preflight
    )
    decision = zero_signal_decision(
        preflight_gate=preflight,
        analysis_gate=analysis,
        conclusion={"conclusion": "frozen_model_does_not_beat_zero"},
    )
    assert preflight["passed"] == 1
    assert analysis["passed"] == 1
    assert decision["decision"] == "zero_signal_diagnostic_complete"
    assert decision["new_scientific_gate_authorized"] == 0
    assert decision["conditional_mean_identically_zero_proven"] == 0
    assert decision["sampling_authorized"] == 0


def test_identity_error_fails_analysis_gate() -> None:
    preflight = evaluate_zero_signal_preflight(_preflight_metrics())
    metrics = _analysis_metrics()
    metrics["decomposition_identity_max_abs_error"] = 1.1e-12
    analysis = evaluate_zero_signal_analysis(metrics, preflight_gate=preflight)
    decision = zero_signal_decision(
        preflight_gate=preflight, analysis_gate=analysis, conclusion=None
    )
    assert analysis["passed"] == 0
    assert decision["decision"] == "diagnostic_execution_invalid"


def test_parent_scope_failure_is_distinct() -> None:
    metrics = _preflight_metrics()
    metrics["parent_registry_verified"] = 0
    preflight = evaluate_zero_signal_preflight(metrics)
    decision = zero_signal_decision(
        preflight_gate=preflight, analysis_gate=None, conclusion=None
    )
    assert preflight["passed"] == 0
    assert decision["decision"] == "parent_scope_invalid"


def test_passing_preflight_is_ready_not_execution_failure() -> None:
    preflight = evaluate_zero_signal_preflight(_preflight_metrics())
    decision = zero_signal_decision(
        preflight_gate=preflight, analysis_gate=None, conclusion=None
    )
    assert decision["decision"] == "ready_for_zero_signal_analysis"
