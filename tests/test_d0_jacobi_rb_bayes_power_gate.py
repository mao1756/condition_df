from __future__ import annotations

import copy

import pytest

from mnist.d0_jacobi_rb_bayes_power_gate import (
    BayesPowerDecision,
    BayesPowerThresholds,
    NO_CLAIM_AUTHORIZATION,
    decide_bayes_power_workflow,
    evaluate_bayes_cache,
    evaluate_bayes_cache_set,
    evaluate_bayes_confirmation,
    evaluate_bayes_power_workflow,
    evaluate_bayes_preflight,
    evaluate_bayes_train,
    execution_failed_gate,
)


def _preflight() -> dict[str, object]:
    t = BayesPowerThresholds()
    names = (
        "parent_provenance_pass",
        "parent_registry_pass",
        "parent_terminal_no_signal_pass",
        "parent_only_aggregate_zero_failure_pass",
        "parent_exact_cache_pass",
        "parent_teacher_pass",
        "parent_optimizer_pass",
        "parent_seal_pass",
        "parent_no_sampling_pass",
        "parent_label_firewall_pass",
        "parent_template_allowlist_pass",
        "source_binding_pass",
        "analytic_normalization_pass",
        "analytic_positive_time_density_pass",
        "analytic_score_pass",
        "analytic_bayes_mean_pass",
        "stationary_null_identity_pass",
        "path_plan_frozen_pass",
        "path_plan_disjoint_pass",
        "path_id_uniqueness_pass",
        "confirmation_absent_pass",
    )
    return {
        **{name: 1 for name in names},
        "root_seed": t.root_seed,
        "selected_outer_steps": list(t.selected_outer_steps),
        "model_seeds": list(t.model_seeds),
        "maximum_float64_identity_error": t.maximum_float64_identity_error,
        "maximum_cuda_identity_error": t.maximum_cuda_identity_error,
        "projected_transition_count": t.total_transition_count,
        "test_only_reduced_workload": 0,
    }


def _cache(law: str, split: str) -> dict[str, object]:
    t = BayesPowerThresholds()
    names = (
        "cache_complete_pass",
        "cache_replay_hash_pass",
        "states_finite_pass",
        "targets_finite_pass",
        "oracle_audit_finite_pass",
        "sample_key_join_pass",
        "role_isolation_pass",
        "model_input_schema_firewall_pass",
        "oracle_input_isolation_pass",
        "exact_jacobi_transition_pass",
        "exact_rb_target_pass",
        "whole_cluster_tower_identity_pass",
    )
    result: dict[str, object] = {
        **{name: 1 for name in names},
        "law": law,
        "split": split,
        "path_count": t.paths_per_role,
        "sample_count": t.samples_per_role,
        "transition_count": t.transitions_per_role,
        "selected_outer_steps": list(t.selected_outer_steps),
        "certificate_fraction": 1.0,
        **{
            name: 0
            for name in (
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
                "target_modification_count",
            )
        },
    }
    if split == "confirmation":
        result.update(
            confirmation_seal_pass=1,
            confirmation_opened_once_pass=1,
            confirmation_plan_unchanged_pass=1,
        )
    else:
        result["confirmation_absent_pass"] = 1
    return result


def _train() -> dict[str, object]:
    t = BayesPowerThresholds()
    names = (
        "teacher_training_complete_pass",
        "null_training_complete_pass",
        "all_six_tasks_complete_pass",
        "all_losses_finite_pass",
        "same_pipeline_pass",
        "training_only_scale_pass",
        "unweighted_mse_objective_pass",
        "no_target_modification_pass",
        "validation_only_selection_pass",
        "analytic_zero_candidate_pass",
        "teacher_nonzero_checkpoint_pass",
        "teacher_checkpoint_hash_pass",
        "null_checkpoint_hash_pass",
        "selected_candidates_frozen_pass",
        "confirmation_gate_definition_frozen_pass",
        "confirmation_absent_pass",
        "model_input_schema_firewall_pass",
        "oracle_input_isolation_pass",
    )
    return {
        **{name: 1 for name in names},
        "model_seed_count": len(t.model_seeds),
        "model_seeds": list(t.model_seeds),
        "validation_path_count_per_law": t.paths_per_role,
        "maximum_updates": t.maximum_updates,
        "teacher_target_scale": 0.25,
        "null_target_scale": 0.20,
    }


def _confirmation() -> dict[str, object]:
    names = (
        "predictions_finite_pass",
        "losses_finite_pass",
        "teacher_selected_model_hash_pass",
        "null_selected_model_hash_pass",
        "model_config_hash_pass",
        "metadata_baseline_hash_pass",
        "path_plan_hash_pass",
        "confirmation_opened_once_pass",
        "confirmation_paths_not_replaced_pass",
        "confirmation_paths_not_added_pass",
        "model_input_schema_firewall_pass",
        "oracle_input_isolation_pass",
    )
    return {
        **{name: 1 for name in names},
        "teacher_confirmation_path_count": 8,
        "null_confirmation_path_count": 8,
        "teacher_path_zero_minus_oracle_mse": [0.02] * 8,
        "teacher_path_metadata_minus_model_mse": [0.01] * 8,
        "null_path_metadata_minus_model_mse": [-0.01] * 8,
        "teacher_aggregate_zero_mse": 1.0,
        "teacher_aggregate_oracle_mse": 0.8,
        "teacher_aggregate_model_mse": 0.9,
        "null_aggregate_zero_mse": 1.0,
        "null_aggregate_model_mse": 1.0,
    }


def test_frozen_workload_cardinalities() -> None:
    t = BayesPowerThresholds()
    assert t.samples_per_role == 8 * 32 * 7
    assert t.transitions_per_role == 8 * 32 * 7 * 392
    assert t.total_transition_count == 6 * t.transitions_per_role
    assert t.model_seeds == (261201, 261202, 261203)


def test_preflight_and_cache_fail_closed() -> None:
    assert evaluate_bayes_preflight(_preflight())["passed"] == 1
    bad = _preflight()
    bad["parent_label_firewall_pass"] = 0
    assert evaluate_bayes_preflight(bad)["passed"] == 0
    gate = evaluate_bayes_cache(
        _cache("teacher", "train"), law="teacher", split="train"
    )
    assert gate["passed"] == 1
    bad_cache = _cache("teacher", "train")
    bad_cache["target_modification_count"] = 1
    assert (
        evaluate_bayes_cache(
            bad_cache, law="teacher", split="train"
        )["passed"]
        == 0
    )
    with pytest.raises(ValueError, match="unknown law"):
        evaluate_bayes_cache(_cache("teacher", "train"), law="physical", split="train")


def test_train_requires_nonzero_teacher_and_sealed_confirmation() -> None:
    assert evaluate_bayes_train(_train())["passed"] == 1
    for name in (
        "all_six_tasks_complete_pass",
        "analytic_zero_candidate_pass",
        "teacher_nonzero_checkpoint_pass",
        "confirmation_absent_pass",
    ):
        bad = _train()
        bad[name] = 0
        assert evaluate_bayes_train(bad)["passed"] == 0
    for name in ("teacher_target_scale", "null_target_scale"):
        bad = _train()
        bad[name] = 0.0
        assert evaluate_bayes_train(bad)["passed"] == 0


def test_cache_set_requires_exact_four_preconfirmation_roles() -> None:
    t = BayesPowerThresholds()
    gates = {
        f"{law}_{split}": evaluate_bayes_cache(
            _cache(law, split), law=law, split=split
        )
        for law in ("teacher", "null")
        for split in ("train", "validation")
    }
    metrics = {
        "transition_count": t.preconfirmation_transition_count,
        "confirmation_absent_pass": 1,
        "role_isolation_pass": 1,
        "training_only_scale_source_pass": 1,
    }
    assert evaluate_bayes_cache_set(metrics, cache_gates=gates)["passed"] == 1
    gates["teacher_confirmation"] = {
        "evaluation_status": "evaluated",
        "passed": 1,
    }
    assert evaluate_bayes_cache_set(metrics, cache_gates=gates)["passed"] == 0


def test_confirmation_recomputes_recovery_and_null_conjunction() -> None:
    teacher_cache = evaluate_bayes_cache(
        _cache("teacher", "confirmation"),
        law="teacher",
        split="confirmation",
    )
    null_cache = evaluate_bayes_cache(
        _cache("null", "confirmation"),
        law="null",
        split="confirmation",
    )
    gate = evaluate_bayes_confirmation(
        _confirmation(),
        teacher_cache_gate=teacher_cache,
        null_cache_gate=null_cache,
    )
    assert gate["passed"] == 1
    assert gate["oracle_relative_gain"] == pytest.approx(0.2)
    assert gate["oracle_gain_recovery"] == pytest.approx(0.5)
    assert gate["null_discovery_conjunction"] == 0
    assert gate["fresh_physical_witness_planning_authorized"] == 1

    false_null = _confirmation()
    false_null["null_path_metadata_minus_model_mse"] = [0.01] * 8
    false_null["null_aggregate_model_mse"] = 0.99
    gate = evaluate_bayes_confirmation(
        false_null,
        teacher_cache_gate=teacher_cache,
        null_cache_gate=null_cache,
    )
    assert gate["passed"] == 0
    assert gate["subchecks"]["null_no_false_discovery"]["passed"] == 0


def test_oracle_underpower_and_model_failure_are_distinct() -> None:
    teacher_cache = evaluate_bayes_cache(
        _cache("teacher", "confirmation"),
        law="teacher",
        split="confirmation",
    )
    null_cache = evaluate_bayes_cache(
        _cache("null", "confirmation"),
        law="null",
        split="confirmation",
    )
    underpowered = _confirmation()
    underpowered["teacher_aggregate_oracle_mse"] = 0.995
    oracle_gate = evaluate_bayes_confirmation(
        underpowered,
        teacher_cache_gate=teacher_cache,
        null_cache_gate=null_cache,
    )
    decision = decide_bayes_power_workflow(
        provenance=1,
        preflight_gate=evaluate_bayes_preflight(_preflight()),
        cache_gate={"evaluation_status": "evaluated", "passed": 1},
        train_gate=evaluate_bayes_train(_train()),
        confirmation_gate=oracle_gate,
    )
    assert decision["decision"] == BayesPowerDecision.ORACLE_PANEL_UNDERPOWERED.value

    weak_model = _confirmation()
    weak_model["teacher_aggregate_model_mse"] = 1.0
    model_gate = evaluate_bayes_confirmation(
        weak_model,
        teacher_cache_gate=teacher_cache,
        null_cache_gate=null_cache,
    )
    decision = decide_bayes_power_workflow(
        provenance=1,
        preflight_gate=evaluate_bayes_preflight(_preflight()),
        cache_gate={"evaluation_status": "evaluated", "passed": 1},
        train_gate=evaluate_bayes_train(_train()),
        confirmation_gate=model_gate,
    )
    assert (
        decision["decision"]
        == BayesPowerDecision.OPTIMIZATION_PIPELINE_INVALID.value
    )


def test_workflow_authorizes_only_fresh_witness_planning() -> None:
    cache_gate = {"evaluation_status": "evaluated", "passed": 1}
    teacher_cache = evaluate_bayes_cache(
        _cache("teacher", "confirmation"),
        law="teacher",
        split="confirmation",
    )
    null_cache = evaluate_bayes_cache(
        _cache("null", "confirmation"),
        law="null",
        split="confirmation",
    )
    confirmation = evaluate_bayes_confirmation(
        _confirmation(),
        teacher_cache_gate=teacher_cache,
        null_cache_gate=null_cache,
    )
    workflow = evaluate_bayes_power_workflow(
        provenance=1,
        preflight_gate=evaluate_bayes_preflight(_preflight()),
        cache_gate=cache_gate,
        train_gate=evaluate_bayes_train(_train()),
        confirmation_gate=confirmation,
        require_gate="controls",
    )
    assert workflow["passed"] == 1
    assert (
        workflow["decision"]["decision"]
        == BayesPowerDecision.NOISY_BAYES_DETECTION_PIPELINE_CALIBRATED.value
    )
    assert workflow["fresh_physical_witness_planning_authorized"] == 1
    for name in NO_CLAIM_AUTHORIZATION:
        assert workflow[name] == 0

    provenance_failure = copy.deepcopy(workflow)
    decision = decide_bayes_power_workflow(
        provenance=0,
        preflight_gate=provenance_failure["components"]["preflight"],
        cache_gate=cache_gate,
        train_gate=provenance_failure["components"]["train"],
        confirmation_gate=provenance_failure["components"]["controls"],
    )
    assert (
        decision["decision"]
        == BayesPowerDecision.CONTROL_PROVENANCE_INVALID.value
    )


def test_execution_failures_receive_closed_scientific_decisions() -> None:
    preflight = evaluate_bayes_preflight(_preflight())
    cache = {"evaluation_status": "evaluated", "passed": 1}
    train = evaluate_bayes_train(_train())
    cache_failure = execution_failed_gate(
        "cache",
        failure_domain="exact_control_cache",
        failure_code="fixture",
        error_type="RuntimeError",
        error="fixture",
    )
    decision = decide_bayes_power_workflow(
        provenance=1,
        preflight_gate=preflight,
        cache_gate=cache_failure,
        train_gate=None,
        confirmation_gate=None,
    )
    assert (
        decision["decision"]
        == BayesPowerDecision.EXACT_CONTROL_CACHE_INVALID.value
    )
    train_failure = execution_failed_gate(
        "train",
        failure_domain="optimization_pipeline",
        failure_code="fixture",
        error_type="RuntimeError",
        error="fixture",
    )
    decision = decide_bayes_power_workflow(
        provenance=1,
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train_failure,
        confirmation_gate=None,
    )
    assert (
        decision["decision"]
        == BayesPowerDecision.OPTIMIZATION_PIPELINE_INVALID.value
    )
