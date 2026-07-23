from __future__ import annotations

import pytest

from mnist.d0_jacobi_rb_cuda_multipath_gate import (
    JacobiRBMultipathThresholds,
    decide_multipath_workflow,
    evaluate_multipath_kernel,
    evaluate_multipath_pilot,
    evaluate_multipath_preflight,
    evaluate_multipath_target,
    not_evaluated_gate,
)


def _preflight_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {
        name: 1
        for name in (
            "control_provenance_pass",
            "parent_certificate_pass",
            "parent_kernel_numerically_valid_pass",
            "parent_single_path_resource_failure_pass",
            "parent_target_not_evaluated_pass",
            "seven_parent_sources_immutable_pass",
            "frozen_runtime_match_pass",
            "cuda_backend_replay_pass",
            "parent_cuda_source_hash_pass",
            "parent_cubin_hash_pass",
            "parent_compile_options_hash_pass",
            "canonical_id_uniqueness_pass",
            "canonical_id_group_order_invariance_pass",
            "canonical_full_id_field_proof_pass",
            "path_zero_parent_replay_pass",
            "serial_batch_parity_pass",
            "phase_order_pass",
            "phase_by_phase_equivalence_pass",
            "group_order_invariance_pass",
            "fresh_b4_parity_pass",
            "path_permutation_invariance_pass",
            "no_cross_path_write_pass",
            "resume_replay_pass",
            "state_updates_device_resident_pass",
            "evolving_state_host_roundtrip_pass",
        )
    }
    metrics.update(
        {
            "parent_record_count": 219,
            "path_count": 64,
            "projection_group_sizes": [10, 10, 10, 10, 10, 10, 4],
            "validation_group_sizes": [10, 4],
            "restart_steps_per_shard": 8,
            "canonical_full_id_plan_count": 89_915_392,
            "maximum_cuda_launch_lanes": 3_920,
            "mass_error": 2.0e-6,
            "uncertified_count": 0,
            "fallback_count": 0,
            "resource_cap_count": 0,
            "invalid_density_count": 0,
            "approximation_count": 0,
            "correction_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "renormalization_count": 0,
            "nonfinite_count": 0,
            "transition_id_collision_count": 0,
            "path_hash_mismatch_count": 0,
            "state_hash_mismatch_count": 0,
        }
    )
    return metrics


def _performance_metrics(*, pilot: bool) -> dict[str, object]:
    t = JacobiRBMultipathThresholds()
    steps = t.pilot_outer_steps if pilot else t.full_outer_steps
    repeats = t.pilot_repeats_per_group if pilot else t.full_repeats_per_group
    metrics: dict[str, object] = {
        name: 1
        for name in (
            "all_groups_completed_pass",
            "all_certificates_pass",
            "output_hash_replay_pass",
            "final_state_hash_replay_pass",
            "certificate_hash_replay_pass",
            "restart_shard_chain_pass",
            "state_updates_device_resident_pass",
            "evolving_state_host_roundtrip_pass",
            "path_isolation_pass",
            "group_path_id_disjoint_pass",
            "group_schedule_pass",
            "commit_reuses_packed_host_snapshot_pass",
        )
    }
    metrics.update(
        {
            "group_sizes": [10, 4],
            "outer_steps": steps,
            "repeats_per_group": repeats,
            "restart_steps_per_shard": 8,
            "maximum_cuda_launch_lanes": 3_920,
            "mass_error": 2.0e-6,
            "cuda_kernel_max_error": 2.0e-6,
            "fallback_fraction": 1.0e-4,
            "fallback_cost_fraction": 0.10,
            "peak_memory_fraction": 0.80,
            "b10_slowest_transitions_per_second": 1_300.0,
            "b4_slowest_transitions_per_second": 1_300.0,
            "projected_cache_hours": 20.0,
            "projected_effective_transitions_per_second": 1_300.0,
            "projected_transition_count": 89_915_392,
            "certificate_fraction": 1.0,
            "completed_shard_count": 2 * repeats * (steps // 8),
            "uncertified_count": 0,
            "resource_cap_count": 0,
            "invalid_density_count": 0,
            "approximation_count": 0,
            "correction_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "renormalization_count": 0,
            "nonfinite_count": 0,
            "replay_bit_mismatch_count": 0,
        }
    )
    if not pilot:
        metrics.update(
            {
                "production_support_pass": 1,
                "cdf_endpoint_certificate_pass": 1,
                "cdf_monotonicity_pass": 1,
                "normalization_pass": 1,
                "semigroup_pass": 1,
                "detailed_balance_pass": 1,
                "law_control_pass": 1,
                "precision_doubling_hash_pass": 1,
                "cuda_pair_mass_error": 2.0e-6,
                "cuda_simplex_error": 2.0e-6,
                "b10_transitions_per_repeat": 14_049_280,
                "b4_transitions_per_repeat": 5_619_712,
                "total_full_benchmark_transitions": 59_006_976,
            }
        )
    return metrics


def _target_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {
        name: 1
        for name in (
            "rao_blackwell_identity_pass",
            "population_tower_identity_pass",
            "latent_mixture_equivalence_pass",
            "pair_mass_conservation_pass",
            "h_minus_two_scaling_pass",
            "invariant_beta_pass",
            "flux_sign_negative_fixtures_pass",
            "all_four_colors_pass",
            "half_full_duration_pass",
            "density_positive_certificate_pass",
            "target_unique_rounding_pass",
            "target_rounding_certificate_pass",
            "conormal_orientation_pass",
            "synthetic_teacher_pass",
            "stationary_null_pass",
            "later_state_only_input_pass",
            "cuda_target_evaluated_pass",
            "serial_multipath_target_parity_pass",
            "target_path_isolation_pass",
        )
    }
    metrics.update(
        {
            "target_certificate_fraction": 1.0,
            "cuda_target_relative_error": 2.0e-5,
            "legacy_mixture_max_absolute_error": 1.0e-8,
            "target_uncertified_count": 0,
            "target_replay_bit_mismatch_count": 0,
            "target_nonfinite_count": 0,
            "earlier_state_input_count": 0,
            "latent_variable_input_count": 0,
            "classifier_target_count": 0,
            "value_target_count": 0,
            "h1_target_count": 0,
            "raw_euler_residual_target_count": 0,
            "gaussian_target_count": 0,
            "target_clip_count": 0,
            "target_floor_count": 0,
            "target_limiter_count": 0,
            "target_projection_count": 0,
        }
    )
    return metrics


def _decision(
    *,
    provenance: bool = True,
    preflight: dict | None = None,
    pilot: dict | None = None,
    kernel: dict | None = None,
    target: dict | None = None,
) -> str:
    return decide_multipath_workflow(
        provenance=provenance,
        preflight_gate=preflight or evaluate_multipath_preflight(_preflight_metrics()),
        pilot_gate=pilot or evaluate_multipath_pilot(_performance_metrics(pilot=True)),
        kernel_gate=kernel or evaluate_multipath_kernel(_performance_metrics(pilot=False)),
        target_gate=target or evaluate_multipath_target(_target_metrics()),
    )["decision"]


def test_frozen_thresholds_and_exact_projection_contract() -> None:
    thresholds = JacobiRBMultipathThresholds()
    assert thresholds.projection_group_sizes == (10, 10, 10, 10, 10, 10, 4)
    assert thresholds.total_full_benchmark_transitions == 59_006_976
    assert evaluate_multipath_preflight(_preflight_metrics())["passed"] == 1
    assert evaluate_multipath_pilot(_performance_metrics(pilot=True))["passed"] == 1
    assert evaluate_multipath_kernel(_performance_metrics(pilot=False))["passed"] == 1
    assert evaluate_multipath_target(_target_metrics())["passed"] == 1
    with pytest.raises(ValueError):
        JacobiRBMultipathThresholds(minimum_rate=1_299.0)


def test_scheduler_parity_failure_is_distinct_from_runtime_failure() -> None:
    metrics = _preflight_metrics()
    metrics["serial_batch_parity_pass"] = 0
    assert _decision(preflight=evaluate_multipath_preflight(metrics)) == (
        "multipath_scheduler_equivalence_invalid"
    )
    metrics = _preflight_metrics()
    metrics["frozen_runtime_match_pass"] = 0
    assert _decision(preflight=evaluate_multipath_preflight(metrics)) == (
        "multipath_cuda_runtime_invalid"
    )


def test_performance_resource_and_numerical_failures_are_distinct() -> None:
    metrics = _performance_metrics(pilot=True)
    metrics["projected_cache_hours"] = 20.0001
    gate = evaluate_multipath_pilot(metrics)
    assert gate["numerically_valid"] == 1
    assert gate["resource_valid"] == 0
    assert _decision(pilot=gate) == "multipath_performance_pilot_unresolved"

    metrics = _performance_metrics(pilot=False)
    metrics["uncertified_count"] = 1
    gate = evaluate_multipath_kernel(metrics)
    assert gate["numerically_valid"] == 0
    assert _decision(kernel=gate) == "multipath_kernel_numerically_unresolved"


@pytest.mark.parametrize(
    ("metric", "bad"),
    [
        ("certificate_fraction", 0.999999999),
        ("completed_shard_count", 383),
        ("total_full_benchmark_transitions", 59_006_975),
        ("b10_transitions_per_repeat", 14_049_279),
        ("b4_transitions_per_repeat", 5_619_711),
    ],
)
def test_full_kernel_fails_closed_on_incomplete_exact_work(metric: str, bad: object) -> None:
    metrics = _performance_metrics(pilot=False)
    metrics[metric] = bad
    gate = evaluate_multipath_kernel(metrics)
    assert gate["passed"] == 0
    assert gate["numerically_valid"] == 0


def test_closed_decision_ladder_and_target_authorization() -> None:
    assert _decision(provenance=False) == "control_provenance_invalid"
    assert _decision(
        pilot=not_evaluated_gate("pilot", "not run")
    ) == "multipath_performance_pilot_unresolved"
    assert _decision(
        kernel=not_evaluated_gate("kernel", "not run")
    ) == "multipath_kernel_computationally_infeasible"
    target = _target_metrics()
    target["serial_multipath_target_parity_pass"] = 0
    assert _decision(target=evaluate_multipath_target(target)) == "jacobi_rb_target_invalid"
    assert _decision() == "exact_jacobi_rb_multipath_kernel_and_target_feasible"
