from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import (
    BoundaryTangentScheduleGateError,
    BoundaryTangentScheduleThresholds,
    decide_schedule_workflow,
    evaluate_schedule_pilot,
    evaluate_schedule_preflight,
    evaluate_schedule_workflow,
    not_evaluated_gate,
)


def _preflight_metrics() -> dict[str, object]:
    t = BoundaryTangentScheduleThresholds()
    result: dict[str, object] = {
        name: 1
        for name in (
            "provenance_valid",
            "readjudication_valid",
            "path_plan_valid",
            "cohort_plan_valid",
            "timing_plan_valid",
            "initial_states_valid",
            "launch_plan_valid",
            "path_collision_free",
            "base_equivalence_valid",
            "fused_branch_equivalence_valid",
            "cross_role_isolation_valid",
            "no_work_valid",
            "parent_sources_immutable",
            "canonical_id_uniqueness_valid",
            "canonical_id_order_invariance_valid",
            "p10_singleton_equivalence_valid",
            "path_permutation_invariance_valid",
            "chunk_invariance_valid",
            "repeat_rotation_valid",
            "atomic_commit_plan_valid",
            "transition_count_algebra",
        )
    }
    result.update(
        {
            "failed_parent_record_count": 14,
            "root_seed": 261_321,
            "cache_group_sizes": [10] * 9 + [6],
            "stream_group_sizes": [10] * 6 + [4],
            "timing_window_starts": [0, 128, 256, 384],
            "timing_branch_steps": [15, 143, 271, 399],
            "timing_window_outer_steps": 16,
            "pilot_repeats": 3,
            "restart_outer_steps": 8,
            "maximum_launch_lanes": 4096,
            "maximum_observed_launch_lanes": 4096,
            "profile_transition_counts": t.profile_transition_counts,
            "base_transition_count": 224_788_480,
            "midpoint_transition_count": 112_394_240,
            "projected_transition_count": 337_182_720,
            "maximum_projected_exact_cache_hours": 30.0,
            "scientific_target_changed": 0,
            "production_cache_generated": 0,
            "physical_training_performed": 0,
            "controller_control_trajectory_performed": 0,
            "full_reverse_path_performed": 0,
            "image_sampling_performed": 0,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        }
    )
    return result


def _pilot_metrics(*, seconds: float | None = None) -> dict[str, object]:
    t = BoundaryTangentScheduleThresholds()
    if seconds is None:
        seconds = 108_000.0 / (8.0 * 17.0)
    elapsed = {name: [seconds, seconds, seconds] for name in t.profile_transition_counts}
    projected = 8.0 * (9.0 + 1.0 + 6.0 + 1.0) * seconds
    effective = t.projected_transition_count / projected
    result: dict[str, object] = {
        name: 1
        for name in (
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
    }
    result.update(
        {
            name: 0
            for name in (
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
        }
    )
    result.update(
        {
            "certificate_fraction": 1.0,
            "maximum_mass_error": 2.0e-12,
            "fallback_fraction": 1.0e-4,
            "fallback_time_fraction": 0.10,
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
            "maximum_observed_launch_lanes": 4096,
            "production_cache_generated": 0,
            "physical_training_performed": 0,
            "controller_control_trajectory_performed": 0,
            "full_reverse_path_performed": 0,
            "image_sampling_performed": 0,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
            "profile_elapsed_seconds": elapsed,
            "projected_elapsed_seconds": projected,
            "projected_effective_transitions_per_second": effective,
        }
    )
    return result


def _decision(
    *,
    provenance: bool = True,
    preflight: dict | None = None,
    pilot: dict | None = None,
) -> str:
    return decide_schedule_workflow(
        provenance=provenance,
        preflight_gate=preflight or evaluate_schedule_preflight(_preflight_metrics()),
        pilot_gate=pilot or evaluate_schedule_pilot(_pilot_metrics()),
    )["decision"]


def test_frozen_thresholds_and_projection_counts() -> None:
    t = BoundaryTangentScheduleThresholds()
    assert t.maximum_projected_exact_cache_hours == 30.0
    assert t.maximum_projected_elapsed_seconds == 108_000.0
    assert t.minimum_projected_effective_transitions_per_second == pytest.approx(
        3_122.0622222222223
    )
    assert t.base_transition_count + t.midpoint_transition_count == (
        t.projected_transition_count
    )
    assert t.cache_group_sizes == (10,) * 9 + (6,)
    assert t.stream_group_sizes == (10,) * 6 + (4,)
    assert evaluate_schedule_preflight(_preflight_metrics())["passed"] == 1
    with pytest.raises(BoundaryTangentScheduleGateError):
        replace(t, maximum_projected_exact_cache_hours=30.0001)


def test_preflight_decisions_distinguish_provenance_algebra_and_equivalence() -> None:
    metrics = _preflight_metrics()
    metrics["failed_parent_record_count"] = 13
    assert _decision(preflight=evaluate_schedule_preflight(metrics)) == (
        "control_provenance_invalid"
    )
    metrics = _preflight_metrics()
    metrics["root_seed"] = 261_322
    assert _decision(preflight=evaluate_schedule_preflight(metrics)) == (
        "boundary_tangent_schedule_algebra_invalid"
    )
    metrics = _preflight_metrics()
    metrics["profile_transition_counts"] = {
        **BoundaryTangentScheduleThresholds().profile_transition_counts,
        "cache_p10": 2_634_239,
    }
    assert _decision(preflight=evaluate_schedule_preflight(metrics)) == (
        "boundary_tangent_schedule_algebra_invalid"
    )
    metrics = _preflight_metrics()
    metrics["provenance_valid"] = 0
    assert _decision(preflight=evaluate_schedule_preflight(metrics)) == (
        "control_provenance_invalid"
    )
    metrics = _preflight_metrics()
    metrics["initial_states_valid"] = 0
    gate = evaluate_schedule_preflight(metrics)
    assert gate["passed"] == 0
    assert gate["schedule_algebra_valid"] == 0
    assert _decision(preflight=gate) == (
        "boundary_tangent_schedule_algebra_invalid"
    )
    metrics = _preflight_metrics()
    metrics["transition_count_algebra"] = 0
    assert _decision(preflight=evaluate_schedule_preflight(metrics)) == (
        "boundary_tangent_schedule_algebra_invalid"
    )
    metrics = _preflight_metrics()
    metrics["fused_branch_equivalence_valid"] = 0
    assert _decision(preflight=evaluate_schedule_preflight(metrics)) == (
        "boundary_tangent_schedule_equivalence_invalid"
    )


@pytest.mark.parametrize("name", ["atomic_commit_plan_valid", "no_work_valid"])
def test_any_preflight_failure_invalidates_stage_execution(name: str) -> None:
    metrics = _preflight_metrics()
    metrics[name] = 0
    gate = evaluate_schedule_preflight(metrics)
    assert gate["passed"] == 0
    assert gate["stage_execution_valid"] == 0
    assert gate["scientific_evidence_complete"] == 0


def test_exact_30_hour_boundary_passes_and_nextafter_fails() -> None:
    metrics = _pilot_metrics()
    gate = evaluate_schedule_pilot(metrics)
    assert gate["passed"] == 1
    assert gate["computed_projected_elapsed_seconds"] == pytest.approx(108_000.0)
    assert gate["computed_projected_exact_cache_hours"] == pytest.approx(30.0)

    seconds = math.nextafter(108_000.0 / (8.0 * 17.0), math.inf)
    failed = evaluate_schedule_pilot(_pilot_metrics(seconds=seconds))
    assert failed["passed"] == 0
    assert failed["failure_domain"] == "resource_gate"
    assert failed["scientific_evidence_complete"] == 1
    assert failed["stage_execution_valid"] == 1
    assert failed["numerically_valid"] == 1
    assert failed["resource_valid"] == 0
    assert _decision(pilot=failed) == (
        "boundary_tangent_schedule_computationally_infeasible"
    )


def test_resource_only_failure_is_not_execution_or_scientific_failure() -> None:
    gate = evaluate_schedule_pilot(_pilot_metrics(seconds=800.0))
    assert gate["passed"] == 0
    assert gate["resource_only_failure"] == 1
    assert gate["failure_domain"] == "resource_gate"
    assert gate["scientific_evidence_complete"] == 1
    assert gate["numerically_valid"] == 1
    assert gate["resource_valid"] == 0

    metrics = _pilot_metrics()
    metrics["maximum_observed_launch_lanes"] = 4097
    gate = evaluate_schedule_pilot(metrics)
    assert gate["passed"] == 0
    assert gate["failure_domain"] == "execution"
    assert gate["stage_execution_valid"] == 1
    assert gate["scientific_evidence_complete"] == 0

    metrics = _pilot_metrics()
    metrics["production_cache_generated"] = 1
    gate = evaluate_schedule_pilot(metrics)
    assert gate["passed"] == 0
    assert gate["failure_domain"] == "execution"
    assert gate["scientific_evidence_complete"] == 0

    metrics = _pilot_metrics()
    metrics["uncertified_count"] = 1
    gate = evaluate_schedule_pilot(metrics)
    assert gate["passed"] == 0
    assert gate["resource_only_failure"] == 0
    assert gate["failure_domain"] == "execution"
    assert gate["scientific_evidence_complete"] == 0
    assert gate["numerically_valid"] == 0
    assert _decision(pilot=gate) == "boundary_tangent_schedule_execution_invalid"


def test_slowest_repeat_not_average_controls_projection() -> None:
    metrics = _pilot_metrics(seconds=700.0)
    elapsed = dict(metrics["profile_elapsed_seconds"])
    elapsed["stream_p4"] = [700.0, 800.0, 700.0]
    metrics["profile_elapsed_seconds"] = elapsed
    projected = 8.0 * (9 * 700.0 + 700.0 + 6 * 700.0 + 800.0)
    metrics["projected_elapsed_seconds"] = projected
    metrics["projected_effective_transitions_per_second"] = (
        337_182_720 / projected
    )
    gate = evaluate_schedule_pilot(metrics)
    assert gate["profile_slowest_elapsed_seconds"]["stream_p4"] == 800.0
    assert gate["computed_projected_elapsed_seconds"] == projected
    assert gate["passed"] == 1

    metrics["projected_elapsed_seconds"] = 8.0 * 17.0 * 700.0
    assert evaluate_schedule_pilot(metrics)["passed"] == 0


def test_execution_failed_gate_maps_fail_closed() -> None:
    gate = evaluate_schedule_pilot(
        {
            "evaluation_status": "execution_failed",
            "failure_code": "atomic_commit_failed",
        }
    )
    assert gate["evaluation_status"] == "execution_failed"
    assert gate["failure_domain"] == "execution"
    assert gate["scientific_evidence_complete"] == 0
    assert _decision(pilot=gate) == "boundary_tangent_schedule_execution_invalid"


def test_workflow_required_gates_and_no_work_scope() -> None:
    preflight = evaluate_schedule_preflight(_preflight_metrics())
    pilot = evaluate_schedule_pilot(_pilot_metrics())
    workflow = evaluate_schedule_workflow(
        provenance=True,
        preflight_gate=preflight,
        pilot_gate=pilot,
        require_gate="pilot",
    )
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == (
        "exact_boundary_tangent_schedule_feasible"
    )
    assert workflow["decision"]["schedule_integration_authorized"] == 1
    assert workflow["cache_generation_authorized"] == 0
    assert workflow["physical_training_performed"] == 0
    assert workflow["sampling_performed"] == 0

    waiting = decide_schedule_workflow(
        provenance=True,
        preflight_gate=preflight,
        pilot_gate=not_evaluated_gate("pilot", "not run"),
    )
    assert waiting["evaluation_status"] == "not_evaluated"
    assert waiting["decision"] == "ready_for_pilot"
    with pytest.raises(BoundaryTangentScheduleGateError):
        evaluate_schedule_workflow(
            provenance=True,
            preflight_gate=preflight,
            pilot_gate=pilot,
            require_gate="control",
        )
