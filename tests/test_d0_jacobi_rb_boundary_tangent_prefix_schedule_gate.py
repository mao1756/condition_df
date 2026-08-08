from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_prefix_schedule_gate import (
    BoundaryTangentPrefixScheduleGateError,
    BoundaryTangentPrefixScheduleThresholds,
    decide_prefix_schedule_workflow,
    evaluate_prefix_schedule_pilot,
    evaluate_prefix_schedule_preflight,
    evaluate_prefix_schedule_profile,
    evaluate_prefix_schedule_workflow,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import (
    BoundaryTangentScheduleThresholds,
)


_NO_WORK = {
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
        "parent_resource_only_failure",
        "parent_scientific_evidence_complete",
        "parent_registry_valid",
        "candidate_unchanged",
        "thread_geometry_unchanged",
        "same_philox_key_and_counter",
        "same_infinite_dyadic_uniform",
        "second_word_revealed_earlier_only",
        "eager_prefix_policy_observed",
        "prefix_interval_nesting_valid",
        "base_scientific_output_equivalent",
        "branch_scientific_output_equivalent",
        "final_state_hashes_equal",
        "canonical_ids_equal",
        "permutation_invariant",
        "chunk_invariant",
        "resume_invariant",
        "eager_certificate_fraction_one",
        "arb_oracle_valid",
        "eager_prefix_arb_fallback_valid",
        "facet_zero_mass_duration_valid",
        "path_plan_valid",
        "cohort_plan_valid",
        "timing_plan_valid",
        "path_collision_free",
        "initial_states_valid",
        "launch_plan_valid",
        "cross_role_isolation_valid",
    )
    return {
        **{name: 1 for name in flags},
        **_NO_WORK,
        "parent_record_count": 614,
        "candidate_modes": 128,
        "candidate_bisection_steps": 56,
        "threads_per_block": 128,
        "maximum_prefix_bits": 1024,
        "initial_eager_prefix_bits": 128,
        "forbidden_event_count": 0,
    }


def _profile(*, seconds: float = 108_000.0) -> dict[str, object]:
    return {
        **_NO_WORK,
        "profile_complete": 1,
        "profile_repeat_count": 3,
        "eager_prefix_policy_observed": 1,
        "scientific_outputs_equal": 1,
        "certificate_fraction": 1.0,
        "forbidden_event_count": 0,
        "projected_elapsed_seconds": seconds,
        "projected_effective_transitions_per_second": 337_182_720 / seconds,
    }


def _pilot(*, seconds: float | None = None) -> dict[str, object]:
    t = BoundaryTangentScheduleThresholds()
    if seconds is None:
        seconds = 108_000.0 / (8.0 * 17.0)
    elapsed = {name: [seconds] * 3 for name in t.profile_transition_counts}
    projected = 8.0 * 17.0 * seconds
    one_flags = (
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
        **{name: 1 for name in one_flags},
        **{name: 0 for name in zero_counts},
        **_NO_WORK,
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
        "profile_elapsed_seconds": elapsed,
        "projected_elapsed_seconds": projected,
        "projected_effective_transitions_per_second": 337_182_720 / projected,
        "eager_prefix_policy_applied": 1,
        "eager_base_prefix_schedule_valid": 1,
        "eager_branch_prefix_schedule_valid": 1,
        "candidate_modes": 128,
        "scientific_hashes_match_parent": 1,
    }


def _decision(
    *,
    provenance: bool = True,
    preflight: dict | None = None,
    profile: dict | None = None,
    pilot: dict | None = None,
) -> str:
    return decide_prefix_schedule_workflow(
        provenance=provenance,
        preflight_gate=preflight or evaluate_prefix_schedule_preflight(_preflight()),
        profile_gate=profile or evaluate_prefix_schedule_profile(_profile()),
        pilot_gate=pilot or evaluate_prefix_schedule_pilot(_pilot()),
    )["decision"]


def test_thresholds_are_frozen_at_exact_parent_gate() -> None:
    t = BoundaryTangentPrefixScheduleThresholds()
    assert t.parent_record_count == 614
    assert t.candidate_modes == 128
    assert t.candidate_bisection_steps == 56
    assert t.threads_per_block == 128
    assert t.initial_eager_prefix_bits == 128
    assert t.maximum_prefix_bits == 1024
    assert t.maximum_projected_seconds == 108_000.0
    assert t.minimum_effective_rate == pytest.approx(3_122.0622222222223)
    with pytest.raises(BoundaryTangentPrefixScheduleGateError):
        replace(t, threads_per_block=64)


def test_preflight_separates_provenance_rng_and_equivalence_failures() -> None:
    assert evaluate_prefix_schedule_preflight(_preflight())["passed"] == 1

    metrics = _preflight()
    metrics["provenance_valid"] = 0
    gate = evaluate_prefix_schedule_preflight(metrics)
    assert gate["failure_domain"] == "provenance"
    assert _decision(preflight=gate) == "control_provenance_invalid"

    metrics = _preflight()
    metrics["same_infinite_dyadic_uniform"] = 0
    gate = evaluate_prefix_schedule_preflight(metrics)
    assert gate["failure_domain"] == "rng_contract"
    assert _decision(preflight=gate) == "eager_prefix_rng_contract_invalid"

    metrics = _preflight()
    metrics["rb_target_bit_identical"] = 0
    # A required declared equivalence flag is what gates the actual replay.
    metrics["branch_scientific_output_equivalent"] = 0
    gate = evaluate_prefix_schedule_preflight(metrics)
    assert gate["failure_domain"] == "equivalence"
    assert _decision(preflight=gate) == "eager_prefix_equivalence_invalid"


def test_preflight_certificate_execution_failure_has_distinct_decision() -> None:
    gate = evaluate_prefix_schedule_preflight(
        {
            "evaluation_status": "execution_failed",
            "failure_domain": "certificate_execution",
            "failure_code": "eager_prefix_certificate_fallback_failed",
        }
    )
    assert gate["passed"] == 0
    assert gate["failure_domain"] == "certificate_execution"
    assert gate["failure_code"] == "eager_prefix_certificate_fallback_failed"
    assert _decision(preflight=gate) == "eager_prefix_certificate_invalid"


def test_profile_exact_runtime_boundary_passes_and_nextafter_fails_resource_only() -> None:
    exact = evaluate_prefix_schedule_profile(_profile())
    assert exact["passed"] == 1
    assert exact["resource_valid"] == 1

    seconds = math.nextafter(108_000.0, math.inf)
    failed = evaluate_prefix_schedule_profile(_profile(seconds=seconds))
    assert failed["passed"] == 0
    assert failed["failure_domain"] == "resource_gate"
    assert failed["resource_only_failure"] == 1
    assert failed["scientific_evidence_complete"] == 1
    assert failed["numerically_valid"] == 1
    assert failed["resource_valid"] == 0
    assert _decision(profile=failed) == (
        "eager_prefix_profile_computationally_infeasible"
    )


def test_profile_numerical_or_equivalence_failure_is_execution_failure() -> None:
    metrics = _profile()
    metrics["scientific_outputs_equal"] = 0
    gate = evaluate_prefix_schedule_profile(metrics)
    assert gate["passed"] == 0
    assert gate["failure_domain"] == "execution"
    assert gate["scientific_evidence_complete"] == 0
    assert gate["resource_only_failure"] == 0
    assert _decision(profile=gate) == "eager_prefix_schedule_execution_invalid"


def test_profile_gate_requires_observed_eager_prefix_policy() -> None:
    metrics = _profile()
    metrics["eager_prefix_policy_observed"] = 0
    gate = evaluate_prefix_schedule_profile(metrics)
    assert gate["passed"] == 0
    assert gate["checks"]["eager_prefix_policy_observed"]["passed"] == 0
    assert gate["failure_domain"] == "execution"
    assert gate["scientific_evidence_complete"] == 0
    assert _decision(profile=gate) == "eager_prefix_schedule_execution_invalid"


def test_full_pilot_preserves_exact_30_hour_gate_and_prefix_contract() -> None:
    gate = evaluate_prefix_schedule_pilot(_pilot())
    assert gate["passed"] == 1
    assert gate["legacy_schedule_gate"]["passed"] == 1

    too_slow = evaluate_prefix_schedule_pilot(
        _pilot(seconds=math.nextafter(108_000.0 / (8.0 * 17.0), math.inf))
    )
    assert too_slow["passed"] == 0
    assert too_slow["failure_domain"] == "resource_gate"
    assert too_slow["resource_only_failure"] == 1
    assert _decision(pilot=too_slow) == (
        "eager_prefix_schedule_computationally_infeasible"
    )

    changed = _pilot()
    changed["scientific_hashes_match_parent"] = 0
    invalid = evaluate_prefix_schedule_pilot(changed)
    assert invalid["passed"] == 0
    assert invalid["failure_domain"] == "execution"
    assert invalid["resource_only_failure"] == 0
    assert invalid["scientific_evidence_complete"] == 0
    assert _decision(pilot=invalid) == "eager_prefix_schedule_execution_invalid"


def test_workflow_is_sequential_and_only_final_pass_authorizes_integration() -> None:
    preflight = evaluate_prefix_schedule_preflight(_preflight())
    profile = evaluate_prefix_schedule_profile(_profile())
    pilot = evaluate_prefix_schedule_pilot(_pilot())

    waiting = decide_prefix_schedule_workflow(
        provenance=True,
        preflight_gate=preflight,
        profile_gate=not_evaluated_gate("profile", "not run"),
        pilot_gate=not_evaluated_gate("pilot", "not run"),
    )
    assert waiting["decision"] == "ready_for_profile"
    assert waiting["schedule_integration_authorized"] == 0

    workflow = evaluate_prefix_schedule_workflow(
        provenance=True,
        preflight_gate=preflight,
        profile_gate=profile,
        pilot_gate=pilot,
        require_gate="pilot",
    )
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == (
        "exact_boundary_tangent_eager_prefix_schedule_feasible"
    )
    assert workflow["decision"]["schedule_integration_authorized"] == 1
    assert workflow["training_authorized"] == 0
    assert workflow["physical_training_performed"] == 0
    assert workflow["sampling_performed"] == 0

    with pytest.raises(BoundaryTangentPrefixScheduleGateError):
        evaluate_prefix_schedule_workflow(
            provenance=True,
            preflight_gate=preflight,
            profile_gate=profile,
            pilot_gate=pilot,
            require_gate="controls",
        )
