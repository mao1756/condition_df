from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_eager_pipeline_gate import (
    BoundaryTangentEagerPipelineGateError,
    BoundaryTangentEagerPipelineThresholds,
    decide_eager_pipeline_workflow,
    evaluate_eager_pipeline_pilot,
    evaluate_eager_pipeline_preflight,
    evaluate_eager_pipeline_workflow,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import (
    BoundaryTangentScheduleThresholds,
)


_NO_WORK = {
    "production_cache_generation_performed": 0,
    "physical_training_performed": 0,
    "controller_control_trajectory_performed": 0,
    "reconstruction_performed": 0,
    "image_sampling_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "full_reverse_path_performed": 0,
}


def _preflight() -> dict[str, object]:
    flags = (
        "provenance_valid",
        "readjudication_valid",
        "parent_registry_valid",
        "parent_profile_gate_valid",
        "parent_resource_only_failure",
        "parent_stage_execution_valid",
        "parent_numerically_valid",
        "parent_scientific_evidence_complete",
        "only_runtime_checks_failed",
        "eager_profile_frozen",
        "pilot_namespaces_unopened",
        "path_plan_valid",
        "timing_plan_valid",
        "transition_counts_valid",
        "schedule_frozen",
        "cross_role_isolation_valid",
        "output_contract_valid",
        "resume_plan_valid",
        "runtime_contract_valid",
    )
    return {
        **{name: 1 for name in flags},
        **_NO_WORK,
        "parent_record_count": 33,
        "repeat_count": 3,
        "profile_count": 4,
        "projected_base_transitions": 224_788_480,
        "projected_midpoint_transitions": 112_394_240,
        "projected_total_transitions": 337_182_720,
        "maximum_launch_lanes": 4_096,
    }


def _pilot(*, seconds: float | None = None) -> dict[str, object]:
    t = BoundaryTangentScheduleThresholds()
    if seconds is None:
        seconds = 108_000.0 / (8.0 * 17.0)
    elapsed = {name: [seconds] * 3 for name in t.profile_transition_counts}
    projected = 8.0 * 17.0 * seconds
    flags = (
        "all_profiles_complete",
        "repeat_hashes_identical",
        "output_hashes_identical",
        "final_state_hashes_identical",
        "certificate_hashes_identical",
        "atomic_shard_chains_valid",
        "resume_replay_valid",
        "completed_repeat_skipping_valid",
        "permitted_input_conversion_valid",
        "raw_label_conversion_valid",
        "cache_commit_valid",
        "predictor_forward_valid",
        "gpu_risk_accumulation_valid",
        "stream_commit_valid",
        "cross_role_isolation_valid",
        "slowest_repeat_selection_valid",
        "repeat_averaging_not_used",
        "posthoc_allowance_not_used",
    )
    zero_counts = (
        "uncertified_count",
        "cap_count",
        "invalid_density_count",
        "approximation_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
        "boundary_rejection_count",
        "transition_id_collision_count",
        "repeat_hash_mismatch_count",
    )
    return {
        **{name: 1 for name in flags},
        **{name: 0 for name in zero_counts},
        **_NO_WORK,
        "certificate_fraction": 1.0,
        "fallback_fraction": 1.0e-4,
        "fallback_time_fraction": 0.10,
        "maximum_mass_error": 2.0e-12,
        "peak_memory_fraction": 0.80,
        "projected_persisted_bytes": 5 * 1024**3 // 4,
        "projected_transition_count": t.projected_transition_count,
        "base_transition_count": t.base_transition_count,
        "midpoint_transition_count": t.midpoint_transition_count,
        "profile_transition_counts": t.profile_transition_counts,
        "restart_outer_steps": 8,
        "pilot_repeats": 3,
        "completed_shard_count": 96,
        "pilot_total_executed_transition_count": 23_708_160,
        "maximum_observed_launch_lanes": 4_096,
        "production_cache_generated": 0,
        "profile_elapsed_seconds": elapsed,
        "projected_elapsed_seconds": projected,
        "projected_effective_transitions_per_second": 337_182_720 / projected,
        "eager_prefix_policy_applied": 1,
        "eager_base_prefix_schedule_valid": 1,
        "eager_branch_prefix_schedule_valid": 1,
        "candidate_modes": 128,
    }


def _decision(
    *,
    provenance: bool = True,
    preflight: dict[str, object] | None = None,
    pilot: dict[str, object] | None = None,
) -> str:
    return decide_eager_pipeline_workflow(
        provenance=provenance,
        preflight_gate=preflight or evaluate_eager_pipeline_preflight(_preflight()),
        pilot_gate=pilot or evaluate_eager_pipeline_pilot(_pilot()),
    )["decision"]


def test_thresholds_freeze_exact_workload_and_thirty_hour_boundary() -> None:
    t = BoundaryTangentEagerPipelineThresholds()
    assert t.parent_record_count == 33
    assert t.repeat_count == 3
    assert t.profile_count == 4
    assert t.projected_base_transitions == 224_788_480
    assert t.projected_midpoint_transitions == 112_394_240
    assert t.projected_total_transitions == 337_182_720
    assert t.maximum_projected_seconds == 108_000.0
    assert t.minimum_effective_rate == pytest.approx(3_122.0622222222223)
    with pytest.raises(BoundaryTangentEagerPipelineGateError):
        replace(t, maximum_projected_seconds=108_001.0)


def test_preflight_passes_and_separates_provenance_policy_and_design() -> None:
    assert evaluate_eager_pipeline_preflight(_preflight())["passed"] == 1

    metrics = _preflight()
    metrics["parent_registry_valid"] = 0
    gate = evaluate_eager_pipeline_preflight(metrics)
    assert gate["failure_domain"] == "provenance"
    assert _decision(preflight=gate) == "control_provenance_invalid"

    metrics = _preflight()
    metrics["eager_profile_frozen"] = 0
    gate = evaluate_eager_pipeline_preflight(metrics)
    assert gate["failure_domain"] == "prefix_policy"
    assert _decision(preflight=gate) == "eager_prefix_policy_invalid"

    metrics = _preflight()
    metrics["pilot_namespaces_unopened"] = 0
    gate = evaluate_eager_pipeline_preflight(metrics)
    assert gate["failure_domain"] == "design"
    assert _decision(preflight=gate) == "eager_pipeline_design_invalid"


def test_runtime_exact_boundary_passes_and_nextafter_fails_resource_only() -> None:
    exact = evaluate_eager_pipeline_pilot(_pilot())
    assert exact["passed"] == 1
    assert exact["resource_valid"] == 1
    assert exact["legacy_schedule_gate"]["passed"] == 1

    failed = evaluate_eager_pipeline_pilot(
        _pilot(
            seconds=math.nextafter(108_000.0 / (8.0 * 17.0), math.inf)
        )
    )
    assert failed["passed"] == 0
    assert failed["failure_domain"] == "resource_gate"
    assert failed["resource_only_failure"] == 1
    assert failed["numerically_valid"] == 1
    assert failed["scientific_evidence_complete"] == 1
    assert _decision(pilot=failed) == "eager_pipeline_computationally_infeasible"


def test_profile_rate_and_numerical_failures_do_not_masquerade_as_resource_only() -> None:
    metrics = _pilot()
    elapsed = dict(metrics["profile_elapsed_seconds"])
    elapsed["cache_p10"] = [
        2_634_240 / math.nextafter(1_300.0, 0.0)
    ] * 3
    metrics["profile_elapsed_seconds"] = elapsed
    slowest = {name: max(values) for name, values in elapsed.items()}
    projected = 8.0 * (
        9.0 * slowest["cache_p10"]
        + slowest["cache_p6"]
        + 6.0 * slowest["stream_p10"]
        + slowest["stream_p4"]
    )
    metrics["projected_elapsed_seconds"] = projected
    metrics["projected_effective_transitions_per_second"] = (
        337_182_720 / projected
    )
    resource = evaluate_eager_pipeline_pilot(metrics)
    assert resource["failure_domain"] == "resource_gate"
    assert resource["resource_only_failure"] == 1

    metrics = _pilot()
    metrics["certificate_fraction"] = math.nextafter(1.0, 0.0)
    numerical = evaluate_eager_pipeline_pilot(metrics)
    assert numerical["failure_domain"] == "numerical"
    assert numerical["resource_only_failure"] == 0
    assert numerical["numerically_valid"] == 0
    assert _decision(pilot=numerical) == "eager_pipeline_numerically_unresolved"


def test_eager_policy_slowest_repeat_and_projection_contract_fail_closed() -> None:
    metrics = _pilot()
    metrics["eager_branch_prefix_schedule_valid"] = 0
    gate = evaluate_eager_pipeline_pilot(metrics)
    assert gate["failure_domain"] == "prefix_policy"
    assert _decision(pilot=gate) == "eager_prefix_policy_invalid"

    metrics = _pilot()
    metrics["slowest_repeat_selection_valid"] = 0
    gate = evaluate_eager_pipeline_pilot(metrics)
    assert gate["failure_domain"] == "design"
    assert _decision(pilot=gate) == "eager_pipeline_design_invalid"

    metrics = _pilot()
    metrics["repeat_hashes_identical"] = 0
    gate = evaluate_eager_pipeline_pilot(metrics)
    assert gate["failure_domain"] == "numerical"
    assert _decision(pilot=gate) == "eager_pipeline_numerically_unresolved"


def test_execution_failure_and_no_work_claims_fail_closed() -> None:
    execution = evaluate_eager_pipeline_pilot(
        {
            "evaluation_status": "execution_failed",
            "failure_domain": "execution",
            "failure_code": "pilot_shard_failed",
        }
    )
    assert execution["stage_execution_valid"] == 0
    assert execution["scientific_evidence_complete"] == 0
    assert _decision(pilot=execution) == "eager_pipeline_execution_invalid"

    metrics = _pilot()
    metrics["physical_training_performed"] = 1
    gate = evaluate_eager_pipeline_pilot(metrics)
    assert gate["passed"] == 0
    assert gate["failure_domain"] == "execution"
    assert gate["resource_only_failure"] == 0


def test_fresh_pilot_does_not_require_a_parent_pilot_hash_match() -> None:
    metrics = _pilot()
    metrics["scientific_hashes_match_parent"] = 0
    gate = evaluate_eager_pipeline_pilot(metrics)
    assert gate["passed"] == 1
    assert "scientific_hashes_match_parent" not in gate["checks"]


def test_launch_limit_is_a_resource_only_failure() -> None:
    metrics = _pilot()
    metrics["maximum_observed_launch_lanes"] = 4_097
    gate = evaluate_eager_pipeline_pilot(metrics)
    assert gate["passed"] == 0
    assert gate["failure_domain"] == "resource_gate"
    assert gate["resource_only_failure"] == 1
    assert gate["scientific_evidence_complete"] == 1


def test_workflow_is_sequential_and_only_pilot_pass_authorizes_integration() -> None:
    preflight = evaluate_eager_pipeline_preflight(_preflight())
    pilot = evaluate_eager_pipeline_pilot(_pilot())
    waiting = decide_eager_pipeline_workflow(
        provenance=True,
        preflight_gate=preflight,
        pilot_gate=not_evaluated_gate("pilot", "not run"),
    )
    assert waiting["decision"] == "ready_for_pilot"
    assert waiting["schedule_integration_authorized"] == 0

    workflow = evaluate_eager_pipeline_workflow(
        provenance=True,
        preflight_gate=preflight,
        pilot_gate=pilot,
        require_gate="pilot",
    )
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == (
        "exact_boundary_tangent_eager_pipeline_feasible"
    )
    assert workflow["decision"]["schedule_integration_authorized"] == 1
    assert workflow["training_authorized"] == 0
    assert workflow["physical_training_performed"] == 0
    assert workflow["sampling_performed"] == 0

    with pytest.raises(BoundaryTangentEagerPipelineGateError):
        evaluate_eager_pipeline_workflow(
            provenance=True,
            preflight_gate=preflight,
            pilot_gate=pilot,
            require_gate="controls",
        )
