from __future__ import annotations

from mnist.d0_jacobi_rb_boundary_tangent_v3_gate import (
    CONFIRM_FLAGS,
    SELECT_FLAGS,
    TRAIN_CONTROL_FLAGS,
    TRAIN_PHYSICAL_FLAGS,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory_gate import (
    BoundaryTangentV3MemoryDecision,
    decide_workflow,
    evaluate_confirm_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    evaluate_select_gate,
    evaluate_train_gate,
    not_evaluated_gate,
)


_MEMORY_FLAGS = (
    "host_backed_batches_valid",
    "maximum_forward_batch_enforced",
    "full_cache_cuda_tensor_absent",
    "streaming_reducer_valid",
    "label_firewall_valid",
    "memory_diagnostics_complete",
)


def _memory_fields() -> dict[str, object]:
    return {
        **{name: 1 for name in _MEMORY_FLAGS},
        "maximum_observed_model_forward_batch_size": 32,
        "full_cache_cuda_tensor_count": 0,
        "peak_memory_fraction": 0.25,
    }


def _preflight() -> dict[str, object]:
    flags = (
        "failed_parent_valid",
        "corrected_parent_adjudication_valid",
        "complete_parent_registry_valid",
        "parent_preflight_and_cache_passed",
        "parent_immutability_valid",
        "downstream_evidence_absent",
        "confirmation_namespace_unopened",
        "immutable_cache_binding_valid",
        "cache_seal_valid",
        "cache_indexes_valid",
        "cache_read_only",
        "cache_not_copied_or_linked",
        "physical_labels_deserialized_during_binding_zero",
        "memory_contract_valid",
        "host_backed_input_store_valid",
        "host_backed_label_store_valid",
        "label_firewall_valid",
        "maximum_forward_batch_enforced",
        "full_cache_cuda_tensor_absent",
        "host_device_batch_equivalence_valid",
        "cuda_forward_backward_seam_valid",
        "streaming_reducer_valid",
        "automatic_batch_sizing_disabled",
        "allocator_workaround_disabled",
    )
    return {
        **{name: 1 for name in flags},
        "maximum_observed_model_forward_batch_size": 32,
        "full_cache_cuda_tensor_count": 0,
        "peak_memory_fraction": 0.25,
        "synthetic_scale_relative_error": 0.0,
    }


def _train() -> dict[str, object]:
    return {
        **{name: 1 for name in TRAIN_CONTROL_FLAGS + TRAIN_PHYSICAL_FLAGS},
        **_memory_fields(),
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
    return {
        **{name: 1 for name in SELECT_FLAGS},
        **_memory_fields(),
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


def _confirm(*, lower: float = 1e-4) -> dict[str, object]:
    return {
        **{name: 1 for name in CONFIRM_FLAGS},
        **_memory_fields(),
        "confirmation_path_count": 64,
        "confirmation_row_count": 114_688,
        "confirmation_transition_count": 134_873_088,
        "component_count": 228,
        "bootstrap_replicates": 50_000,
        "minimum_lower_bound": lower,
        "all_lower_bounds_strictly_positive": int(lower > 0),
        "certificate_fraction": 1.0,
        "maximum_mass_error": 0.0,
        "forbidden_event_count": 0,
        "confirmation_transitions_per_second": 2_000.0,
        "fallback_fraction": 0.0,
        "fallback_time_fraction": 0.0,
        "actual_cache_plus_confirmation_seconds": 90_000.0,
        "confirmation_performed": 1,
    }


def test_memory_preflight_passes_and_authorizes_train_only() -> None:
    gate = evaluate_preflight_gate(_preflight())
    assert gate["passed"] == 1
    decision = decide_workflow(
        preflight_gate=gate,
        train_gate=not_evaluated_gate("train", "not run"),
        select_gate=None,
        confirm_gate=None,
    )
    assert decision["decision"] == "ready_for_train"
    assert decision["physical_training_authorized"] == 1
    assert decision["confirmation_authorized"] == 0
    assert decision["sampling_authorized"] == 0


def test_batch_or_peak_memory_failures_have_distinct_decisions() -> None:
    bad_batch = _preflight()
    bad_batch["maximum_observed_model_forward_batch_size"] = 33
    gate = evaluate_preflight_gate(bad_batch)
    assert gate["failure_domain"] == "training_memory_schedule"
    assert decide_workflow(
        preflight_gate=gate,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )["decision"] == BoundaryTangentV3MemoryDecision.TRAINING_MEMORY_SCHEDULE_INVALID

    bad_peak = _preflight()
    bad_peak["peak_memory_fraction"] = 0.8000000000000002
    gate = evaluate_preflight_gate(bad_peak)
    assert gate["failure_domain"] == "training_memory_resource"
    assert gate["scientific_evidence_complete"] == 1
    assert decide_workflow(
        preflight_gate=gate,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )["decision"] == (
        BoundaryTangentV3MemoryDecision.TRAINING_MEMORY_RESOURCE_INFEASIBLE
    )


def test_preflight_execution_failure_preserves_provenance_domain() -> None:
    for domain, decision in (
        (
            "control_provenance",
            BoundaryTangentV3MemoryDecision.CONTROL_PROVENANCE_INVALID,
        ),
        (
            "immutable_cache_binding",
            BoundaryTangentV3MemoryDecision.IMMUTABLE_CACHE_BINDING_INVALID,
        ),
    ):
        gate = evaluate_preflight_gate(
            {
                "evaluation_status": "execution_failed",
                "failure_domain": domain,
                "failure_code": "fixture_invalid",
            }
        )
        assert gate["failure_domain"] == domain
        assert decide_workflow(
            preflight_gate=gate,
            train_gate=None,
            select_gate=None,
            confirm_gate=None,
        )["decision"] == decision


def test_downstream_binding_failure_keeps_closed_binding_decision() -> None:
    preflight = evaluate_preflight_gate(_preflight())
    failure = {
        "evaluation_status": "execution_failed",
        "passed": 0,
        "failure_domain": "immutable_cache_binding",
    }
    assert decide_workflow(
        preflight_gate=preflight,
        train_gate=failure,
        select_gate=None,
        confirm_gate=None,
    )["decision"] == BoundaryTangentV3MemoryDecision.IMMUTABLE_CACHE_BINDING_INVALID


def test_streamed_scientific_gates_preserve_v3_outcomes() -> None:
    preflight = evaluate_preflight_gate(_preflight())
    train = evaluate_train_gate(_train())
    select = evaluate_select_gate(_select())
    confirm = evaluate_confirm_gate(_confirm())
    assert all(gate["passed"] == 1 for gate in (train, select, confirm))
    decision = decide_workflow(
        preflight_gate=preflight,
        train_gate=train,
        select_gate=select,
        confirm_gate=confirm,
    )
    assert decision["decision"] == (
        BoundaryTangentV3MemoryDecision.EXACT_RB_ZERO_BASELINE_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED
    )
    assert decision["controller_control_planning_authorized"] == 1
    assert decision["controller_control_trajectory_authorized"] == 0
    assert decision["sampling_authorized"] == 0


def test_no_candidate_never_opens_confirmation_and_required_gate_fails_closed() -> None:
    preflight = evaluate_preflight_gate(_preflight())
    train = evaluate_train_gate(_train())
    select = evaluate_select_gate(_select(eligible=0))
    assert select["failure_domain"] == "no_validation_candidate"
    decision = decide_workflow(
        preflight_gate=preflight,
        train_gate=train,
        select_gate=select,
        confirm_gate=not_evaluated_gate("confirm", "forbidden"),
    )
    assert decision["decision"] == BoundaryTangentV3MemoryDecision.NO_VALIDATION_CANDIDATE
    assert decision["confirmation_authorized"] == 0

    workflow = evaluate_required_gate(
        preflight_gate=preflight,
        train_gate=train,
        select_gate=select,
        confirm_gate=None,
        require_gate="select",
    )
    assert workflow["required_gate_pass"] == 0
    assert workflow["required_gate_exit_code"] == 1
    assert workflow["artifacts_must_be_committed_before_required_gate_exit"] == 1


def test_train_oom_maps_to_memory_resource_not_scientific_controls() -> None:
    gate = evaluate_train_gate(
        {
            "evaluation_status": "execution_failed",
            "failure_code": "streaming_train_cuda_out_of_memory",
            "failure_domain": "training_memory_resource",
            "physical_training_performed": 0,
        }
    )
    assert gate["failure_domain"] == "training_memory_resource"
    assert gate["numerically_valid"] == 1
    assert gate["resource_valid"] == 0
    decision = decide_workflow(
        preflight_gate=evaluate_preflight_gate(_preflight()),
        train_gate=gate,
        select_gate=None,
        confirm_gate=None,
    )
    assert decision["decision"] == (
        BoundaryTangentV3MemoryDecision.TRAINING_MEMORY_RESOURCE_INFEASIBLE
    )
