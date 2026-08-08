from __future__ import annotations

from typing import Any

import pytest

from mnist.d0_jacobi_rb_dynkin_phase_observer_gate import (
    PhaseObserverThresholds,
    decide_phase_observer_workflow,
    evaluate_phase_observer_power,
    evaluate_phase_observer_preflight,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_dynkin_power_gate import DynkinPowerThresholds


_FORBIDDEN_COUNTS = (
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
)


def _preflight_metrics() -> dict[str, Any]:
    t = PhaseObserverThresholds()
    flags = (
        "production_authorizing_pass",
        "control_provenance_pass",
        "parent_failure_adjudication_pass",
        "twenty_one_parent_sources_immutable_pass",
        "parent_path_id_plan_pass",
        "parent_legacy_k512_replay_pass",
        "parent_phase_moment_oracle_pass",
        "parent_no_tower_pilot_work_pass",
        "path_id_plan_pass",
        "fresh_namespace_disjoint_pass",
        "legacy_k512_replay_pass",
        "transition_target_certificate_hash_invariance_pass",
        "observer_state_hash_invariance_pass",
        "global_subtraction_roundoff_reproduced_pass",
        "phase_local_fourier_formula_pass",
        "phase_local_quadratic_formula_pass",
        "phase_local_cubic_formula_pass",
        "phase_local_all_matchings_pass",
        "phase_local_half_full_duration_pass",
        "phase_local_facet_interior_pass",
        "phase_local_zero_mass_duration_pass",
        "structural_invariant_mask_pass",
        "structural_zero_center_pass",
        "structural_zero_radius_pass",
        "float64_arb_agreement_pass",
        "cuda_enclosure_pass",
        "quantile_enclosure_pass",
        "noninvariant_local_global_agreement_pass",
        "phase_moment_oracle_pass",
        "tower_panel_a_pass",
        "tower_panel_b_pass",
        "tower_joint_max_t_pass",
        "tower_panels_frozen_pass",
        "tower_panels_disjoint_pass",
        "tower_case_atomic_resume_pass",
        "tower_case_observer_input_hash_invariance_pass",
        "tower_case_structural_zero_center_pass",
        "tower_case_structural_zero_radius_pass",
        "tower_case_noninvariant_global_agreement_pass",
        "tower_authorizing_interval_radius_pass",
        "negative_orientation_fixture_pass",
        "negative_quadratic_factor_fixture_pass",
        "negative_cubic_factor_fixture_pass",
        "negative_pair_mass_fixture_pass",
        "negative_duration_fixture_pass",
        "negative_eigenvalue_fixture_pass",
        "negative_corrupt_enclosure_fixture_pass",
        "negative_post_state_fixture_pass",
    )
    metrics: dict[str, Any] = {name: 1 for name in flags}
    metrics.update(
        {
            "parent_record_count": t.parent_record_count,
            "parent_source_count": t.parent_source_count,
            "root_seed": t.root_seed,
            "grid_size": t.grid_size,
            "alpha": t.alpha,
            "tau_eff": t.tau_eff,
            "tower_panel_count": t.tower_panel_count,
            "tower_clusters_per_panel": t.tower_clusters_per_panel,
            "tower_family_member_count": t.tower_family_member_count,
            "structural_zero_member_count": t.structural_zero_member_count,
            "tower_bootstrap_replicates": t.tower_bootstrap_replicates,
            "tower_confidence": t.tower_confidence,
            "maximum_float64_observer_error": (
                t.maximum_float64_observer_error
            ),
            "maximum_cuda_observer_error": t.maximum_cuda_observer_error,
            "maximum_cumulative_standardized_error": (
                t.maximum_cumulative_standardized_error
            ),
            "certificate_fraction": 1.0,
            "fallback_fraction": t.maximum_fallback_fraction,
            "fallback_cost_fraction": t.maximum_fallback_cost_fraction,
            "peak_memory_fraction": t.maximum_peak_memory_fraction,
            "mass_error": t.maximum_cuda_mass_error,
            **{name: 0 for name in _FORBIDDEN_COUNTS},
        }
    )
    return metrics


def _power_metrics() -> dict[str, Any]:
    t = DynkinPowerThresholds()
    flags = (
        "production_authorizing_pass",
        "panel_a_frozen_pass",
        "panel_b_frozen_pass",
        "panel_plan_hash_pass",
        "panel_disjoint_pass",
        "panel_nonregeneration_pass",
        "pilot_production_disjoint_pass",
        "right_endpoint_coupling_unchanged_pass",
        "raw_observables_advisory_only_pass",
        "dynkin_authorizing_estimator_pass",
        "forecast_label_pass",
        "panel_a_complete_pass",
        "panel_b_complete_pass",
        "combined_complete_pass",
        "panel_a_nomination_pass",
        "panel_b_confirmation_pass",
        "combined_confirmation_pass",
        "selected_design_frozen_pass",
        "selected_design_hash_pass",
        "complete_candidate_grid_pass",
        "shard_chain_pass",
        "mass_conservation_pass",
        "state_updates_device_resident_pass",
        "pilot_certification_pass",
        "executed_panels_numerically_valid_pass",
        "candidate_resource_feasibility_pass",
    )
    metrics: dict[str, Any] = {name: 1 for name in flags}
    metrics.update(
        {
            "panel_count": t.pilot_panel_count,
            "paths_per_panel": t.pilot_paths_per_panel,
            "levels": list(t.levels),
            "candidate_main_paths": list(t.candidate_main_paths),
            "candidate_reference_paths": list(t.candidate_reference_paths),
            "selected_main_paths": 32,
            "selected_reference_paths": 16,
            "panel_a_main_half_width": t.maximum_main_half_width,
            "panel_a_reference_half_width": t.maximum_reference_half_width,
            "panel_b_main_half_width": t.maximum_main_half_width,
            "panel_b_reference_half_width": t.maximum_reference_half_width,
            "combined_main_half_width": t.maximum_main_half_width,
            "combined_reference_half_width": t.maximum_reference_half_width,
            "projected_production_hours": t.maximum_projected_hours,
            "resource_feasible_candidate_count": 3,
            "minimum_rate": t.minimum_rate,
            "certificate_fraction": 1.0,
            "fallback_fraction": t.maximum_fallback_fraction,
            "fallback_cost_fraction": t.maximum_fallback_cost_fraction,
            "peak_memory_fraction": t.maximum_peak_memory_fraction,
            "mass_error": t.maximum_cuda_mass_error,
            "maximum_cumulative_standardized_error": (
                t.maximum_cumulative_standardized_error
            ),
            **{name: 0 for name in _FORBIDDEN_COUNTS},
        }
    )
    return metrics


def _decision(
    *,
    provenance: bool | int | dict[str, Any] = True,
    preflight: dict[str, Any] | None = None,
    pilot: dict[str, Any] | None = None,
) -> str:
    return decide_phase_observer_workflow(
        provenance=provenance,
        preflight_gate=(
            evaluate_phase_observer_preflight(_preflight_metrics())
            if preflight is None
            else preflight
        ),
        pilot_gate=(
            not_evaluated_gate("pilot", "not run")
            if pilot is None
            else pilot
        ),
    )["decision"]


def test_frozen_thresholds_and_exact_boundaries_pass() -> None:
    with pytest.raises(ValueError, match="root_seed"):
        PhaseObserverThresholds(root_seed=1)
    gate = evaluate_phase_observer_preflight(_preflight_metrics())
    assert gate["passed"] == 1
    assert gate["provenance_valid"] == 1
    assert gate["parent_adjudication_valid"] == 1
    assert gate["scheduler_valid"] == 1
    assert gate["observer_algebra_valid"] == 1
    assert gate["tower_identity_valid"] == 1
    assert gate["numerically_valid"] == 1
    assert gate["resource_valid"] == 1


@pytest.mark.parametrize(
    ("metric", "failure_value", "aggregate"),
    (
        ("control_provenance_pass", 0, "provenance_valid"),
        ("parent_failure_adjudication_pass", 0, "parent_adjudication_valid"),
        ("path_id_plan_pass", 0, "scheduler_valid"),
        ("phase_local_fourier_formula_pass", 0, "observer_algebra_valid"),
        ("tower_panel_b_pass", 0, "tower_identity_valid"),
        ("structural_zero_center_pass", 0, "numerically_valid"),
        ("peak_memory_fraction", 0.81, "resource_valid"),
    ),
)
def test_preflight_subchecks_fail_their_named_aggregate(
    metric: str,
    failure_value: object,
    aggregate: str,
) -> None:
    metrics = _preflight_metrics()
    metrics[metric] = failure_value
    gate = evaluate_phase_observer_preflight(metrics)
    assert gate["passed"] == 0
    assert gate[aggregate] == 0


def test_numerical_threshold_just_above_boundary_fails() -> None:
    metrics = _preflight_metrics()
    metrics["maximum_cuda_observer_error"] = (
        PhaseObserverThresholds().maximum_cuda_observer_error + 1.0e-12
    )
    gate = evaluate_phase_observer_preflight(metrics)
    assert gate["numerically_valid"] == 0
    assert gate["passed"] == 0


def test_pilot_wrapper_preserves_unchanged_power_gate() -> None:
    gate = evaluate_phase_observer_power(_power_metrics())
    assert gate["schema"].endswith("dynkin-phase-observer-gate")
    assert gate["passed"] == 1
    assert gate["numerically_valid"] == 1
    assert gate["resource_valid"] == 1
    assert gate["panel_a_nominated"] == 1
    assert gate["panels_agree"] == 1


def test_every_closed_decision_and_authorization_boundary() -> None:
    assert _decision(provenance=False) == "control_provenance_invalid"

    metrics = _preflight_metrics()
    metrics["parent_failure_adjudication_pass"] = 0
    assert (
        _decision(preflight=evaluate_phase_observer_preflight(metrics))
        == "parent_failure_adjudication_invalid"
    )
    metrics = _preflight_metrics()
    metrics["path_id_plan_pass"] = 0
    assert (
        _decision(preflight=evaluate_phase_observer_preflight(metrics))
        == "refinement_scheduler_invalid"
    )
    metrics = _preflight_metrics()
    metrics["phase_local_quadratic_formula_pass"] = 0
    assert (
        _decision(preflight=evaluate_phase_observer_preflight(metrics))
        == "phase_observer_algebra_invalid"
    )
    metrics = _preflight_metrics()
    metrics["tower_panel_a_pass"] = 0
    assert (
        _decision(preflight=evaluate_phase_observer_preflight(metrics))
        == "dynkin_tower_identity_invalid"
    )
    metrics = _preflight_metrics()
    metrics["structural_zero_radius_pass"] = 0
    assert (
        _decision(preflight=evaluate_phase_observer_preflight(metrics))
        == "phase_observer_numerically_unresolved"
    )
    metrics = _preflight_metrics()
    metrics["peak_memory_fraction"] = 0.81
    assert (
        _decision(preflight=evaluate_phase_observer_preflight(metrics))
        == "dynkin_computationally_infeasible"
    )

    repaired = decide_phase_observer_workflow(
        provenance=True,
        preflight_gate=evaluate_phase_observer_preflight(_preflight_metrics()),
        pilot_gate=not_evaluated_gate("pilot", "not run"),
    )
    assert repaired["decision"] == "phase_local_dynkin_observer_repaired"
    assert repaired["sealed_power_pilot_authorized"] == 1
    assert repaired["production_refinement_patch_authorized"] == 0

    power = _power_metrics()
    power["panel_a_nomination_pass"] = 0
    power["selected_main_paths"] = -1
    power["selected_reference_paths"] = -1
    assert (
        _decision(pilot=evaluate_phase_observer_power(power))
        == "dynkin_power_infeasible"
    )
    power = _power_metrics()
    power["panel_b_confirmation_pass"] = 0
    assert (
        _decision(pilot=evaluate_phase_observer_power(power))
        == "dynkin_panels_disagree"
    )

    final = decide_phase_observer_workflow(
        provenance=True,
        preflight_gate=evaluate_phase_observer_preflight(_preflight_metrics()),
        pilot_gate=evaluate_phase_observer_power(_power_metrics()),
    )
    assert final["decision"] == "exact_dynkin_refinement_estimator_feasible"
    assert final["production_refinement_patch_authorized"] == 1
    assert final["sealed_power_pilot_authorized"] == 0
    assert final["physical_training_authorized"] == 0
    assert final["sampling_authorized"] == 0


@pytest.mark.parametrize(
    ("failure_domain", "expected"),
    (
        ("scheduler", "refinement_scheduler_invalid"),
        ("configuration", "refinement_scheduler_invalid"),
        ("resource", "dynkin_computationally_infeasible"),
        ("observer_algebra", "phase_observer_algebra_invalid"),
        ("numerical", "phase_observer_numerically_unresolved"),
        ("generic", "phase_observer_numerically_unresolved"),
    ),
)
def test_execution_failures_are_classified_without_false_provenance(
    failure_domain: str,
    expected: str,
) -> None:
    failure = {
        "evaluation_status": "execution_failed",
        "passed": 0,
        "failure_domain": failure_domain,
        "failure_code": "stable-fixture",
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "provenance_valid": 0,
    }
    assert _decision(preflight=failure) == expected


def test_synthetic_missing_fields_cannot_become_provenance_failure() -> None:
    failure = {"evaluation_status": "execution_failed", "passed": 0}
    assert (
        _decision(provenance=failure)
        == "phase_observer_numerically_unresolved"
    )
