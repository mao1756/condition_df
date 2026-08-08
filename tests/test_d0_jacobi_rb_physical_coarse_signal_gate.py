from __future__ import annotations

from mnist.d0_jacobi_rb_physical_coarse_signal_gate import (
    evaluate_panel,
    evaluate_preflight,
    evaluate_witness,
    witness_decision,
)


def _preflight_metrics() -> dict[str, float | int]:
    metrics = {
        name: 1
        for name in (
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
    }
    metrics.update(
        {
            "projected_two_panel_hours": 20.0,
            "transitions_per_second": 2500.0,
            "peak_memory_fraction": 0.2,
            "fallback_fraction": 0.0,
            "fallback_time_fraction": 0.0,
            "maximum_mass_error": 1.0e-15,
        }
    )
    return metrics


def _panel_metrics() -> dict[str, float | int]:
    metrics = {
        name: 1
        for name in (
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
    }
    metrics.update(
        {
            "path_count": 64,
            "transition_count": 89_915_392,
            "maximum_mass_error": 1.0e-15,
            "fallback_fraction": 0.0,
            "fallback_time_fraction": 0.0,
            "peak_memory_fraction": 0.2,
            "transitions_per_second": 2500.0,
        }
    )
    return metrics


def _witness_metrics() -> dict[str, int]:
    return {
        "joint_analysis_seal_valid": 1,
        "panels_opened_once": 1,
        "panel_hashes_unchanged": 1,
        "panel_path_sets_disjoint": 1,
        "cell_count": 10_976,
        "panel_a_path_count": 64,
        "panel_b_path_count": 64,
        "estimator_algebra_pass": 1,
        "bootstrap_whole_path_only": 1,
        "bootstrap_replicates": 50_000,
        "bootstrap_finite": 1,
        "influence_components_finite": 1,
        "welch_bound_finite": 1,
        "one_sided_99_percent_bounds": 1,
        "negative_values_not_truncated": 1,
        "decision_partition_pass": 1,
        "old_physical_data_excluded": 1,
        "no_training_performed": 1,
        "no_sampling_performed": 1,
    }


def test_complete_evidence_gate_accepts_every_closed_scientific_outcome() -> None:
    preflight = evaluate_preflight(_preflight_metrics())
    panel_a = evaluate_panel(
        _panel_metrics(), panel="panel-a", prerequisite_gate=preflight
    )
    panel_b = evaluate_panel(
        _panel_metrics(), panel="panel-b", prerequisite_gate=panel_a
    )
    witness = evaluate_witness(
        _witness_metrics(), panel_a_gate=panel_a, panel_b_gate=panel_b
    )
    assert preflight["passed"] == panel_a["passed"] == panel_b["passed"] == 1
    assert witness["passed"] == 1
    for outcome in (
        "exact_physical_coarse_signal_detected",
        "coarse_signal_below_preregistered_resolution",
        "physical_coarse_signal_inconclusive",
    ):
        decision = witness_decision(
            preflight_gate=preflight,
            panel_a_gate=panel_a,
            panel_b_gate=panel_b,
            witness_gate=witness,
            scientific_outcome=outcome,
        )
        assert decision["decision"] == outcome


def test_preflight_resource_boundary_fails_closed() -> None:
    metrics = _preflight_metrics()
    metrics["projected_two_panel_hours"] = 24.000001
    gate = evaluate_preflight(metrics)
    assert gate["passed"] == 0
    decision = witness_decision(
        preflight_gate=gate,
        panel_a_gate=None,
        panel_b_gate=None,
        witness_gate=None,
        scientific_outcome=None,
    )
    assert decision["decision"] == "physical_coarse_signal_computationally_infeasible"


def test_preflight_numerical_failure_is_distinct_from_resource_failure() -> None:
    metrics = _preflight_metrics()
    metrics["benchmark_certificate_fraction_one"] = 0
    gate = evaluate_preflight(metrics)
    decision = witness_decision(
        preflight_gate=gate,
        panel_a_gate=None,
        panel_b_gate=None,
        witness_gate=None,
        scientific_outcome=None,
    )
    assert decision["decision"] == "physical_coarse_signal_numerically_unresolved"


def test_panel_integrity_numerical_and_resource_failures_are_distinct() -> None:
    preflight = evaluate_preflight(_preflight_metrics())
    cases = (
        ("cell_shape_pass", 0, "physical_coarse_signal_panel_integrity_invalid"),
        ("certificate_fraction_one", 0, "physical_coarse_signal_numerically_unresolved"),
        (
            "transitions_per_second",
            1299.0,
            "physical_coarse_signal_computationally_infeasible",
        ),
    )
    for field, value, expected in cases:
        metrics = _panel_metrics()
        metrics[field] = value
        panel = evaluate_panel(
            metrics, panel="panel-a", prerequisite_gate=preflight
        )
        decision = witness_decision(
            preflight_gate=preflight,
            panel_a_gate=panel,
            panel_b_gate=None,
            witness_gate=None,
            scientific_outcome=None,
        )
        assert decision["decision"] == expected


def test_witness_inference_integrity_fails_without_reclassifying_science() -> None:
    preflight = evaluate_preflight(_preflight_metrics())
    panel_a = evaluate_panel(
        _panel_metrics(), panel="panel-a", prerequisite_gate=preflight
    )
    panel_b = evaluate_panel(
        _panel_metrics(), panel="panel-b", prerequisite_gate=panel_a
    )
    metrics = _witness_metrics()
    metrics["bootstrap_finite"] = 0
    witness = evaluate_witness(
        metrics, panel_a_gate=panel_a, panel_b_gate=panel_b
    )
    decision = witness_decision(
        preflight_gate=preflight,
        panel_a_gate=panel_a,
        panel_b_gate=panel_b,
        witness_gate=witness,
        scientific_outcome="exact_physical_coarse_signal_detected",
    )
    assert witness["passed"] == 0
    assert decision["decision"] == "physical_coarse_signal_estimator_invalid"
