from __future__ import annotations

import math

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_v3_gate import (
    BoundaryTangentV3Decision,
    BoundaryTangentV3GateError,
    BoundaryTangentV3Thresholds,
    decide_workflow,
    evaluate_cache_gate,
    evaluate_confirm_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    evaluate_select_gate,
    evaluate_train_gate,
    not_evaluated_gate,
)


def _preflight() -> dict[str, object]:
    t = BoundaryTangentV3Thresholds()
    names = (
        "parent_v2_valid",
        "adjudication_valid",
        "adjudication_authority_valid",
        "complete_parent_registry_valid",
        "complete_adjudication_registry_valid",
        "parent_immutability_valid",
        "source_closure_valid",
        "path_plan_valid",
        "cohort_plan_valid",
        "path_collision_scan_valid",
        "zero_baseline_contract_valid",
        "baseline_artifacts_absent",
        "state_dict_baseline_free",
        "update_zero_exact",
        "certificate_semantics_comparator_valid",
        "preflight_complete",
        "scheduler_seam_valid",
        "exact_kernel_contract_valid",
        "cuda_determinism_valid",
        "source_image_binding_valid",
        "selected_step_contract_valid",
        "model_input_firewall_valid",
        "raw_target_contract_valid",
        "train_validation_confirmation_unopened",
        "inherited_resource_projection_valid",
    )
    return {
        **{name: 1 for name in names},
        "preflight_path_ids": list(t.preflight_path_ids),
        "preflight_path_count": 8,
        "certificate_fraction": 1.0,
        "maximum_mass_error": 0.0,
        "forbidden_event_count": 0,
        "transitions_per_second": 2_000.0,
        "peak_memory_fraction": 0.1,
    }


def _cache() -> dict[str, object]:
    t = BoundaryTangentV3Thresholds()
    names = (
        "cache_complete",
        "train_cache_complete",
        "validation_cache_complete",
        "atomic_shard_chains_valid",
        "resume_replay_valid",
        "selected_sample_cartesian_valid",
        "train_validation_indexes_disjoint",
        "artifact_role_isolation_valid",
        "mixed_cohort_split_before_commit",
        "raw_target_contract_valid",
        "input_field_contract_valid",
        "confirmation_absent",
        "confirmation_namespace_unopened",
        "baseline_artifacts_absent",
    )
    return {
        **{name: 1 for name in names},
        "train_path_count": 64,
        "validation_path_count": 32,
        "train_row_count": t.train_rows,
        "validation_row_count": t.validation_rows,
        "train_transition_count": t.train_transitions,
        "validation_transition_count": t.validation_transitions,
        "certificate_fraction": 1.0,
        "maximum_mass_error": 0.0,
        "forbidden_event_count": 0,
        "minimum_role_rate": 2_000.0,
        "fallback_fraction": 0.0,
        "fallback_time_fraction": 0.0,
        "peak_memory_fraction": 0.1,
        "total_persisted_cache_bytes": 1,
        "projected_cache_plus_confirmation_seconds": 100_000.0,
        "production_cache_generation_performed": 1,
        "physical_training_performed": 0,
        "confirmation_performed": 0,
    }


def _train() -> dict[str, object]:
    names = (
        "zero_initialization_control_passed",
        "synthetic_teacher_passed",
        "synthetic_every_validation_path_beats_zero",
        "exact_model_null_passed",
        "null_selected_update_zero",
        "null_parameters_bitwise_unchanged",
        "controls_before_training_label_open",
        "physical_training_complete",
        "all_physical_tasks_complete_finite",
        "training_labels_opened_after_controls",
        "validation_labels_opened_zero",
        "validation_inputs_unavailable_to_physical_trainer",
        "fixed_checkpoint_grid_complete",
        "candidate_grid_valid",
        "pointwise_checkpoint_selection_performed_zero",
        "physical_task_records_selection_free",
        "training_only_target_scale_valid",
        "baseline_artifacts_absent",
        "confirmation_absent",
    )
    return {
        **{name: 1 for name in names},
        "synthetic_relative_validation_mse": 0.001,
        "model_seed_count": 3,
        "checkpoint_count": 123,
        "nonzero_candidate_count": 120,
        "maximum_updates": 4_000,
        "physical_training_performed": 1,
        "validation_labels_opened": 0,
        "pointwise_checkpoint_selection_performed": 0,
    }


def _select(*, eligible: int = 1) -> dict[str, object]:
    names = (
        "selection_complete",
        "train_stage_seal_valid",
        "search_plan_committed_before_validation_labels",
        "bootstrap_counts_committed_before_validation_labels",
        "validation_labels_opened_once",
        "update_zero_control_valid",
        "all_candidate_commits_valid",
        "candidate_table_valid",
        "family_names_and_order_valid",
        "whole_path_shared_counts_valid",
        "studentization_valid",
        "bootstrap_restart_evidence_valid",
        "quantile_rule_valid",
        "confirmation_absent",
    )
    return {
        **{name: 1 for name in names},
        "path_count": 32,
        "candidate_count": 120,
        "component_count": 228,
        "search_family_size": 27_360,
        "bootstrap_replicates": 50_000,
        "bootstrap_shard_count": 50,
        "eligible_candidate_count": eligible,
        "selected_nonzero": int(eligible > 0),
        "logical_update_zero_selected": int(eligible == 0),
        "selected_minimum_lower_bound": 1e-4 if eligible else None,
    }


def _confirm(*, minimum: float = 1e-4) -> dict[str, object]:
    names = (
        "confirmation_complete",
        "confirmation_namespace_opened_once",
        "selection_sealed_before_namespace_open",
        "bootstrap_counts_committed_before_paths",
        "same_228_family_valid",
        "whole_path_shared_counts_valid",
        "studentization_valid",
        "atomic_shard_chains_valid",
        "resume_replay_valid",
        "raw_confirmation_inputs_not_persisted",
        "raw_confirmation_labels_not_persisted",
    )
    return {
        **{name: 1 for name in names},
        "confirmation_path_count": 64,
        "confirmation_row_count": 114_688,
        "confirmation_transition_count": 134_873_088,
        "component_count": 228,
        "bootstrap_replicates": 50_000,
        "minimum_lower_bound": minimum,
        "all_lower_bounds_strictly_positive": int(minimum > 0.0),
        "certificate_fraction": 1.0,
        "maximum_mass_error": 0.0,
        "forbidden_event_count": 0,
        "confirmation_transitions_per_second": 2_000.0,
        "fallback_fraction": 0.0,
        "fallback_time_fraction": 0.0,
        "peak_memory_fraction": 0.1,
        "actual_cache_plus_confirmation_seconds": 100_000.0,
        "confirmation_performed": 1,
    }


def _gates() -> tuple[dict[str, object], ...]:
    return (
        evaluate_preflight_gate(_preflight()),
        evaluate_cache_gate(_cache()),
        evaluate_train_gate(_train()),
        evaluate_select_gate(_select()),
        evaluate_confirm_gate(_confirm()),
    )


def test_frozen_thresholds_and_exact_family() -> None:
    t = BoundaryTangentV3Thresholds()
    assert t.search_family_size == 120 * 228 == 27_360
    assert t.preflight_path_ids == tuple(range(0xF0000, 0xF0008))
    assert t.training_path_ids == tuple(range(0xF1000, 0xF1040))
    assert t.validation_path_ids == tuple(range(0xF1100, 0xF1120))
    assert t.confirmation_path_ids == tuple(range(0xF2000, 0xF2040))
    with pytest.raises(BoundaryTangentV3GateError):
        BoundaryTangentV3Thresholds(candidate_count=119)


def test_stage_gates_pass_and_only_final_authorizes_planning() -> None:
    preflight, cache, train, select, confirm = _gates()
    assert all(gate["passed"] == 1 for gate in _gates())
    result = decide_workflow(
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train,
        select_gate=select,
        confirm_gate=confirm,
    )
    assert result["decision"] == (
        BoundaryTangentV3Decision.EXACT_RB_ZERO_BASELINE_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED
    )
    assert result["controller_control_planning_authorized"] == 1
    assert result["controller_control_trajectory_authorized"] == 0
    assert result["sampling_authorized"] == 0


def test_closed_decision_precedence() -> None:
    preflight, cache, train, select, confirm = _gates()
    broken = _preflight()
    broken["path_plan_valid"] = 0
    assert decide_workflow(
        preflight_gate=evaluate_preflight_gate(broken),
        cache_gate=cache,
        train_gate=train,
        select_gate=select,
        confirm_gate=confirm,
    )["decision"] == BoundaryTangentV3Decision.PROVENANCE_OR_PATH_PLAN_INVALID

    broken = _preflight()
    broken["zero_baseline_contract_valid"] = 0
    assert decide_workflow(
        preflight_gate=evaluate_preflight_gate(broken),
        cache_gate=cache,
        train_gate=train,
        select_gate=select,
        confirm_gate=confirm,
    )["decision"] == BoundaryTangentV3Decision.ZERO_BASELINE_CONTRACT_INVALID

    broken = _preflight()
    broken["certificate_semantics_comparator_valid"] = 0
    comparator = evaluate_preflight_gate(broken)
    assert comparator["failure_domain"] == "implementation_contract"
    assert comparator["stage_execution_valid"] == 1
    assert comparator["scientific_evidence_complete"] == 1
    assert comparator["numerically_valid"] == 1
    assert comparator["resource_valid"] == 1
    assert decide_workflow(
        preflight_gate=comparator,
        cache_gate=cache,
        train_gate=train,
        select_gate=select,
        confirm_gate=confirm,
    )["decision"] == (
        BoundaryTangentV3Decision.CERTIFICATE_SEMANTICS_COMPARATOR_INVALID
    )

    no_candidate = evaluate_select_gate(_select(eligible=0))
    assert no_candidate["passed"] == 0
    assert no_candidate["scientific_evidence_complete"] == 1
    assert decide_workflow(
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train,
        select_gate=no_candidate,
        confirm_gate=not_evaluated_gate("confirm", "forbidden"),
    )["decision"] == BoundaryTangentV3Decision.NO_VALIDATION_CANDIDATE

    failed_confirm = evaluate_confirm_gate(_confirm(minimum=0.0))
    assert decide_workflow(
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train,
        select_gate=select,
        confirm_gate=failed_confirm,
    )["decision"] == BoundaryTangentV3Decision.ZERO_BASELINE_V3_SIGNAL_NOT_CONFIRMED


def test_preflight_comparator_and_scientific_seam_failures_are_distinct() -> None:
    comparator_metrics = _preflight()
    comparator_metrics["certificate_semantics_comparator_valid"] = 0
    comparator = evaluate_preflight_gate(comparator_metrics)
    assert comparator["passed"] == 0
    assert comparator["failure_domain"] == "implementation_contract"
    assert comparator["certificate_semantics_comparator_valid"] == 0
    assert comparator["certificate_semantics_comparator_failure"] == 1
    assert comparator["stage_execution_valid"] == 1
    assert comparator["scientific_evidence_complete"] == 1
    assert comparator["numerically_valid"] == 1
    assert comparator["resource_valid"] == 1
    assert comparator["controller_control_planning_authorized"] == 0
    assert comparator["physical_training_performed"] == 0
    assert comparator["confirmation_performed"] == 0

    semantic_metrics = _preflight()
    semantic_metrics["scheduler_seam_valid"] = 0
    semantic = evaluate_preflight_gate(semantic_metrics)
    assert semantic["passed"] == 0
    assert semantic["failure_domain"] == "numerical"
    assert semantic["certificate_semantics_comparator_valid"] == 1
    assert semantic["certificate_semantics_comparator_failure"] == 0
    assert semantic["stage_execution_valid"] == 1
    assert semantic["scientific_evidence_complete"] == 1
    assert semantic["numerically_valid"] == 0
    assert semantic["resource_valid"] == 1
    assert decide_workflow(
        preflight_gate=semantic,
        cache_gate=None,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )["decision"] == BoundaryTangentV3Decision.EXACT_CACHE_INVALID


def test_genuine_numerical_failure_takes_precedence_over_comparator_flag() -> None:
    metrics = _preflight()
    metrics["certificate_semantics_comparator_valid"] = 0
    metrics["scheduler_seam_valid"] = 0
    gate = evaluate_preflight_gate(metrics)
    assert gate["failure_domain"] == "numerical"
    assert gate["certificate_semantics_comparator_valid"] == 0
    assert gate["certificate_semantics_comparator_failure"] == 0
    assert gate["numerically_valid"] == 0
    assert decide_workflow(
        preflight_gate=gate,
        cache_gate=None,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )["decision"] == BoundaryTangentV3Decision.EXACT_CACHE_INVALID


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("certificate_fraction", math.nextafter(1.0, 0.0)),
        ("maximum_mass_error", math.nextafter(2.0e-12, math.inf)),
        ("forbidden_event_count", 1),
    ),
)
def test_numerical_health_failure_is_not_masked_by_comparator_failure(
    field: str, value: float | int
) -> None:
    metrics = _preflight()
    metrics["certificate_semantics_comparator_valid"] = 0
    metrics[field] = value
    gate = evaluate_preflight_gate(metrics)
    assert gate["failure_domain"] == "numerical"
    assert gate["certificate_semantics_comparator_failure"] == 0
    assert gate["numerically_valid"] == 0
    assert gate["stage_execution_valid"] == 1
    assert gate["scientific_evidence_complete"] == 1
    assert decide_workflow(
        preflight_gate=gate,
        cache_gate=None,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )["decision"] == BoundaryTangentV3Decision.EXACT_CACHE_INVALID


def test_generic_preflight_execution_contract_failure_is_incomplete() -> None:
    metrics = _preflight()
    metrics["exact_kernel_contract_valid"] = 0
    gate = evaluate_preflight_gate(metrics)
    assert gate["failure_domain"] == "execution"
    assert gate["stage_execution_valid"] == 0
    assert gate["scientific_evidence_complete"] == 0
    assert gate["numerically_valid"] == 1
    assert gate["resource_valid"] == 1
    assert decide_workflow(
        preflight_gate=gate,
        cache_gate=None,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )["decision"] == BoundaryTangentV3Decision.EXACT_CACHE_INVALID


def test_select_and_confirm_are_strict_at_zero() -> None:
    no_candidate = evaluate_select_gate(_select(eligible=0))
    assert no_candidate["no_validation_candidate"] == 1
    invalid = _select()
    invalid["selected_minimum_lower_bound"] = 0.0
    assert evaluate_select_gate(invalid)["passed"] == 0
    assert evaluate_confirm_gate(_confirm(minimum=math.nextafter(0.0, 1.0)))["passed"] == 1
    assert evaluate_confirm_gate(_confirm(minimum=0.0))["passed"] == 0


def test_cache_and_confirmation_enforce_the_frozen_runtime_cap() -> None:
    t = BoundaryTangentV3Thresholds()
    cache = _cache()
    cache["projected_cache_plus_confirmation_seconds"] = t.maximum_projected_seconds
    assert evaluate_cache_gate(cache)["passed"] == 1
    cache["projected_cache_plus_confirmation_seconds"] = math.nextafter(
        t.maximum_projected_seconds, math.inf
    )
    failed_cache = evaluate_cache_gate(cache)
    assert failed_cache["passed"] == 0
    assert failed_cache["resource_valid"] == 0

    confirm = _confirm()
    confirm["actual_cache_plus_confirmation_seconds"] = t.maximum_projected_seconds
    assert evaluate_confirm_gate(confirm)["passed"] == 1
    confirm["actual_cache_plus_confirmation_seconds"] = math.nextafter(
        t.maximum_projected_seconds, math.inf
    )
    failed_confirm = evaluate_confirm_gate(confirm)
    assert failed_confirm["passed"] == 0
    assert failed_confirm["resource_valid"] == 0
    assert failed_confirm["scientific_evidence_complete"] == 1


def test_required_gate_fails_closed_and_unknown_name_rejected() -> None:
    preflight, cache, train, select, confirm = _gates()
    workflow = evaluate_required_gate(
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train,
        select_gate=select,
        confirm_gate=confirm,
        require_gate="confirm",
    )
    assert workflow["required_gate_pass"] == 1
    assert workflow["required_gate_exit_code"] == 0
    failed = evaluate_required_gate(
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train,
        select_gate=evaluate_select_gate(_select(eligible=0)),
        confirm_gate=None,
        require_gate="select",
    )
    assert failed["required_gate_pass"] == 0
    assert failed["required_gate_exit_code"] == 1
    with pytest.raises(BoundaryTangentV3GateError):
        evaluate_required_gate(
            preflight_gate=None,
            cache_gate=None,
            train_gate=None,
            select_gate=None,
            confirm_gate=None,
            require_gate="controller",
        )
