from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mnist.d0_jacobi_rb_haar_gate import (
    ANTITHETIC_HAAR_PROFILE,
    NESTED_HAAR_PROFILE,
    PROFILE_ORDER,
    HaarCouplingThresholds,
    antithetic_fine_mean,
    decide_haar_workflow,
    decide_sealed_profile_selection,
    evaluate_haar_coupling,
    evaluate_haar_pilot,
    evaluate_haar_preflight,
    evaluate_haar_workflow,
    independent_pool_richardson_contrasts,
    nominate_haar_power_design,
    not_evaluated_gate,
    raw_successive_differences,
)


_FORBIDDEN = (
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
    t = HaarCouplingThresholds()
    flags = (
        "production_authorizing_pass",
        "control_provenance_pass",
        "parent_readjudication_pass",
        "parent_sources_immutable_pass",
        "parent_preflight_pass",
        "parent_pilot_numerically_valid_pass",
        "parent_pilot_resource_valid_pass",
        "parent_power_only_failure_pass",
        "normal_inverse_arb_enclosure_pass",
        "normal_cdf_arb_enclosure_pass",
        "normal_extreme_prefix_pass",
        "fused_cuda_normal_authorizer_pass",
        "haar_covariance_pass",
        "haar_within_level_independence_pass",
        "haar_parent_child_aggregation_pass",
        "antithetic_marginal_equality_pass",
        "state_independent_uniform_pass",
        "path_id_slot_plan_pass",
        "future_production_reserved_pass",
        "profile_panel_disjoint_pass",
        "path_id_uniqueness_pass",
        "intentional_haar_ancestry_only_pass",
        "order_chunk_resume_invariance_pass",
        "saved_prefix_jacobi_replay_pass",
        "arbitrary_uniform_cuda_authorizer_pass",
        "jacobi_marginal_cdf_pass",
        "jacobi_eigenmoment_pass",
        "jacobi_detailed_balance_pass",
        "rb_target_certificate_pass",
        "later_state_only_contract_pass",
        "all_colors_pass",
        "half_full_duration_pass",
        "facet_pass",
        "zero_mass_duration_pass",
        "phase_tower_identity_pass",
        "interruption_replay_pass",
        "deterministic_batching_pass",
        "candidate_under_48h_forecast_pass",
    )
    return {
        **{name: 1 for name in flags},
        "parent_record_count": t.parent_record_count,
        "parent_source_count": t.parent_source_count,
        "root_seed": t.root_seed,
        "grid_size": t.grid_size,
        "alpha": t.alpha,
        "tau_eff": t.tau_eff,
        "levels": list(t.levels),
        "maximum_prefix_bits": t.maximum_prefix_bits,
        "certificate_fraction": 1.0,
        "fallback_fraction": t.maximum_fallback_fraction,
        "fallback_cost_fraction": t.maximum_fallback_cost_fraction,
        "peak_memory_fraction": t.maximum_peak_memory_fraction,
        "mass_error": t.maximum_cuda_mass_error,
        **{name: 0 for name in _FORBIDDEN},
    }


def _coupling_metrics() -> dict[str, Any]:
    t = HaarCouplingThresholds()
    flags = (
        "production_authorizing_pass",
        "nested_profile_complete_pass",
        "antithetic_profile_complete_pass",
        "normal_cells_certified_pass",
        "uniform_cells_certified_pass",
        "fused_cuda_normal_authorizer_pass",
        "jacobi_outputs_certified_pass",
        "arbitrary_uniform_cuda_authorizer_pass",
        "haar_covariance_pass",
        "within_level_independence_pass",
        "parent_child_aggregation_pass",
        "antithetic_marginal_pass",
        "state_independent_rng_pass",
        "id_uniqueness_pass",
        "intentional_ancestry_only_pass",
        "order_invariance_pass",
        "chunk_invariance_pass",
        "resume_invariance_pass",
        "marginal_cdf_pass",
        "marginal_eigenmoment_pass",
        "marginal_detailed_balance_pass",
        "conservation_pass",
        "target_contract_pass",
        "pipeline_runtime_projection_pass",
    )
    return {
        **{name: 1 for name in flags},
        "profile_order": list(PROFILE_ORDER),
        "certificate_fraction": 1.0,
        "fallback_fraction": t.maximum_fallback_fraction,
        "fallback_cost_fraction": t.maximum_fallback_cost_fraction,
        "minimum_rate": t.minimum_rate,
        "minimum_projected_hours": t.maximum_projected_hours,
        "peak_memory_fraction": t.maximum_peak_memory_fraction,
        "mass_error": t.maximum_cuda_mass_error,
        **{name: 0 for name in _FORBIDDEN},
    }


def _pilot_metrics() -> dict[str, Any]:
    t = HaarCouplingThresholds()
    flags = (
        "production_authorizing_pass",
        "plans_frozen_pass",
        "panels_disjoint_pass",
        "panel_nonregeneration_pass",
        "profile_order_pass",
        "no_fallback_after_panel_b_pass",
        "raw_endpoint_authorizing_pass",
        "dynkin_advisory_only_pass",
        "independent_pool_variance_pass",
        "richardson_formula_pass",
        "executed_panels_complete_pass",
        "executed_panels_numerically_valid_pass",
        "shard_chain_pass",
        "mass_conservation_pass",
        "pilot_production_isolation_pass",
        "panel_a_nominated",
        "panel_b_opened",
        "panels_agree",
    )
    return {
        **{name: 1 for name in flags},
        "selected_profile": NESTED_HAAR_PROFILE,
        "panel_a_clusters": t.panel_clusters,
        "panel_b_clusters": t.panel_clusters,
        "combined_clusters": 2 * t.panel_clusters,
        "combined_main_half_width": t.maximum_main_half_width,
        "combined_generator_reference_half_width": (
            t.maximum_reference_half_width
        ),
        "combined_reference_stability_half_width": (
            t.maximum_reference_half_width
        ),
        "projected_hours": t.maximum_projected_hours,
        "minimum_rate": t.minimum_rate,
        "certificate_fraction": 1.0,
        "fallback_fraction": t.maximum_fallback_fraction,
        "fallback_cost_fraction": t.maximum_fallback_cost_fraction,
        "peak_memory_fraction": t.maximum_peak_memory_fraction,
        "mass_error": t.maximum_cuda_mass_error,
        **{name: 0 for name in _FORBIDDEN},
    }


def _candidate(
    main: int,
    reference: int,
    *,
    hours: float,
    eligible_widths: bool = True,
) -> dict[str, Any]:
    t = HaarCouplingThresholds()
    width_scale = 1.0 if eligible_widths else 2.0
    return {
        "main_paths": main,
        "reference_paths": reference,
        "predicted_main_half_width": t.maximum_main_half_width * width_scale,
        "predicted_generator_reference_half_width": (
            t.maximum_reference_half_width * width_scale
        ),
        "predicted_reference_stability_half_width": (
            t.maximum_reference_half_width * width_scale
        ),
        "projected_hours": hours,
        "conservative_rate": t.minimum_rate,
        "panel_complete_pass": 1,
        "panel_finite_pass": 1,
        "panel_certification_pass": 1,
        "panel_numerical_health_pass": 1,
        "mass_conservation_pass": 1,
        "shard_chain_pass": 1,
        "pilot_production_isolation_pass": 1,
        "pilot_means_excluded_pass": 1,
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
    }


def _nested_nomination(*, passes: bool = True) -> dict[str, Any]:
    return nominate_haar_power_design(
        profile=NESTED_HAAR_PROFILE,
        panel_role="a",
        candidates=[
            _candidate(32, 16, hours=20.0, eligible_widths=passes),
            _candidate(32, 32, hours=25.0, eligible_widths=passes),
            _candidate(64, 16, hours=30.0, eligible_widths=passes),
            _candidate(64, 32, hours=40.0, eligible_widths=passes),
        ],
    )


def _antithetic_nomination(*, passes: bool = True) -> dict[str, Any]:
    return nominate_haar_power_design(
        profile=ANTITHETIC_HAAR_PROFILE,
        panel_role="a",
        candidates=[
            _candidate(16, 16, hours=41.0, eligible_widths=passes),
        ],
    )


def _confirmation(selected: dict[str, Any], *, passes: bool = True) -> dict[str, Any]:
    t = HaarCouplingThresholds()
    scale = 1.0 if passes else 2.0
    return {
        "evaluation_status": "evaluated",
        "profile": selected["profile"],
        "main_paths": selected["main_paths"],
        "reference_paths": selected["reference_paths"],
        "complete_pass": 1,
        "finite_pass": 1,
        "certification_pass": 1,
        "numerical_health_pass": 1,
        "mass_conservation_pass": 1,
        "shard_chain_pass": 1,
        "main_half_width": t.maximum_main_half_width * scale,
        "generator_reference_half_width": t.maximum_reference_half_width * scale,
        "reference_stability_half_width": t.maximum_reference_half_width * scale,
        "projected_hours": 20.0,
        "minimum_rate": t.minimum_rate,
    }


def test_thresholds_are_frozen_and_exact_boundaries_pass() -> None:
    with pytest.raises(ValueError, match="root_seed"):
        HaarCouplingThresholds(root_seed=1)
    assert evaluate_haar_preflight(_preflight_metrics())["passed"] == 1
    assert evaluate_haar_coupling(_coupling_metrics())["passed"] == 1
    assert evaluate_haar_pilot(_pilot_metrics())["passed"] == 1


@pytest.mark.parametrize(
    ("metric", "aggregate"),
    (
        ("control_provenance_pass", "provenance_valid"),
        ("haar_covariance_pass", "rng_algebra_valid"),
        ("normal_inverse_arb_enclosure_pass", "normal_transform_valid"),
        ("fused_cuda_normal_authorizer_pass", "normal_transform_valid"),
        ("saved_prefix_jacobi_replay_pass", "jacobi_certificate_valid"),
        (
            "arbitrary_uniform_cuda_authorizer_pass",
            "jacobi_certificate_valid",
        ),
        ("jacobi_eigenmoment_pass", "marginal_law_valid"),
        ("interruption_replay_pass", "scheduler_valid"),
        ("candidate_under_48h_forecast_pass", "resource_valid"),
    ),
)
def test_preflight_failures_have_named_aggregates(
    metric: str, aggregate: str
) -> None:
    metrics = _preflight_metrics()
    metrics[metric] = 0
    gate = evaluate_haar_preflight(metrics)
    assert gate["passed"] == 0
    assert gate[aggregate] == 0


def test_antithetic_mean_and_successive_differences() -> None:
    positive = np.array([[1.0, 2.0], [3.0, 4.0]])
    negative = np.array([[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(
        antithetic_fine_mean(positive, negative),
        np.array([[2.0, 3.0], [4.0, 5.0]]),
    )
    values = {
        128: np.full((2, 1), 5.0),
        256: np.full((2, 1), 4.0),
        512: np.full((2, 1), 2.0),
        1024: np.full((2, 1), -1.0),
        2048: np.full((2, 1), -5.0),
    }
    differences = raw_successive_differences(values)
    for index, expected in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        np.testing.assert_array_equal(
            differences[f"D{index}"], np.full((2, 1), expected)
        )


def test_independent_pool_richardson_formulas_and_variances() -> None:
    d3 = np.array([[1.0], [3.0], [5.0]])
    d4 = np.array([[2.0], [4.0]])
    result = independent_pool_richardson_contrasts(d3, d4)
    np.testing.assert_allclose(
        result["generator_reference_contrast"], [3.0 + 4.0]
    )
    np.testing.assert_allclose(
        result["reference_stability_contrast"], [(3.0 - 12.0) / 3.0]
    )
    expected_var3 = np.var(d3, axis=0, ddof=1) / 3.0
    expected_var4 = (16.0 / 9.0) * np.var(d4, axis=0, ddof=1) / 2.0
    np.testing.assert_allclose(
        result["generator_reference_variance"], expected_var3 + expected_var4
    )
    np.testing.assert_allclose(
        result["reference_stability_variance"],
        expected_var3 / 9.0 + expected_var4,
    )
    assert result["independent_pool_covariance"] == 0.0


def test_nested_nomination_uses_frozen_ranking() -> None:
    nomination = _nested_nomination()
    assert nomination["passed"] == 1
    assert nomination["selected"]["main_paths"] == 32
    assert nomination["selected"]["reference_paths"] == 16
    assert nomination["selected"]["projected_hours"] == 20.0


def test_incomplete_or_duplicate_candidate_grids_fail_closed() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        nominate_haar_power_design(
            profile=NESTED_HAAR_PROFILE,
            panel_role="a",
            candidates=[_candidate(32, 16, hours=20.0)],
        )
    duplicate = [_candidate(32, 16, hours=20.0) for _ in range(4)]
    with pytest.raises(ValueError, match="duplicate"):
        nominate_haar_power_design(
            profile=NESTED_HAAR_PROFILE,
            panel_role="a",
            candidates=duplicate,
        )


def test_profile_two_opens_only_after_profile_one_a_has_no_nominee() -> None:
    nested = _nested_nomination(passes=False)
    antithetic = _antithetic_nomination()
    pending = decide_sealed_profile_selection(
        nested_panel_a=nested,
        antithetic_panel_a=antithetic,
    )
    assert pending["selected_profile"] == ANTITHETIC_HAAR_PROFILE
    assert pending["panel_b_opened"] == 0
    assert pending["selected"] is not None


def test_no_profile_fallback_after_panel_b_is_inspected() -> None:
    nested = _nested_nomination()
    selected = nested["selected"]
    with pytest.raises(ValueError, match="must remain unopened"):
        decide_sealed_profile_selection(
            nested_panel_a=nested,
            nested_panel_b=_confirmation(selected, passes=False),
            nested_combined=_confirmation(selected),
            antithetic_panel_a=_antithetic_nomination(),
        )
    result = decide_sealed_profile_selection(
        nested_panel_a=nested,
        nested_panel_b=_confirmation(selected, passes=False),
        nested_combined=_confirmation(selected),
    )
    assert result["passed"] == 0
    assert result["panel_b_opened"] == 1
    assert result["fallback_after_panel_b_permitted"] == 0


def test_sealed_panel_and_combined_must_both_confirm_exact_nominee() -> None:
    nested = _nested_nomination()
    selected = nested["selected"]
    result = decide_sealed_profile_selection(
        nested_panel_a=nested,
        nested_panel_b=_confirmation(selected),
        nested_combined=_confirmation(selected),
    )
    assert result["passed"] == 1
    assert result["panels_agree"] == 1
    wrong = _confirmation(selected)
    wrong["main_paths"] = 64
    result = decide_sealed_profile_selection(
        nested_panel_a=nested,
        nested_panel_b=wrong,
        nested_combined=_confirmation(selected),
    )
    assert result["passed"] == 0


def _decision(
    *,
    provenance: bool | int | dict[str, Any] = True,
    preflight: dict[str, Any] | None = None,
    coupling: dict[str, Any] | None = None,
    pilot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return decide_haar_workflow(
        provenance=provenance,
        preflight_gate=(
            evaluate_haar_preflight(_preflight_metrics())
            if preflight is None
            else preflight
        ),
        coupling_gate=(
            evaluate_haar_coupling(_coupling_metrics())
            if coupling is None
            else coupling
        ),
        pilot_gate=(
            evaluate_haar_pilot(_pilot_metrics()) if pilot is None else pilot
        ),
    )


def test_closed_decision_ladder_and_authorization_boundary() -> None:
    assert _decision(provenance=False)["decision"] == "control_provenance_invalid"

    failures = (
        ("haar_covariance_pass", "hierarchical_rng_algebra_invalid"),
        ("normal_cdf_arb_enclosure_pass", "certified_normal_transform_invalid"),
        (
            "saved_prefix_jacobi_replay_pass",
            "arbitrary_uniform_jacobi_certificate_invalid",
        ),
        ("jacobi_eigenmoment_pass", "hierarchical_marginal_law_invalid"),
        ("interruption_replay_pass", "hierarchical_scheduler_invalid"),
        (
            "candidate_under_48h_forecast_pass",
            "hierarchical_coupling_computationally_infeasible",
        ),
    )
    for metric, expected in failures:
        metrics = _preflight_metrics()
        metrics[metric] = 0
        result = _decision(
            preflight=evaluate_haar_preflight(metrics),
            coupling=not_evaluated_gate("coupling", "blocked"),
            pilot=not_evaluated_gate("pilot", "blocked"),
        )
        assert result["decision"] == expected

    power = _pilot_metrics()
    power["panel_a_nominated"] = 0
    power["panel_b_opened"] = 0
    power["panels_agree"] = 0
    power["combined_main_half_width"] = float("inf")
    assert (
        _decision(pilot=evaluate_haar_pilot(power))["decision"]
        == "hierarchical_power_infeasible"
    )
    power = _pilot_metrics()
    power["panels_agree"] = 0
    assert (
        _decision(pilot=evaluate_haar_pilot(power))["decision"]
        == "hierarchical_panels_disagree"
    )
    power = _pilot_metrics()
    power["profile_order_pass"] = 0
    assert (
        _decision(pilot=evaluate_haar_pilot(power))["decision"]
        == "hierarchical_scheduler_invalid"
    )
    success = _decision()
    assert (
        success["decision"]
        == "exact_haar_hierarchical_refinement_coupling_feasible"
    )
    assert success["production_refinement_patch_authorized"] == 1
    assert success["physical_training_authorized"] == 0
    assert success["sampling_authorized"] == 0


def test_workflow_required_gate_prefix_and_unknown_gate() -> None:
    workflow = evaluate_haar_workflow(
        provenance=True,
        preflight_gate=evaluate_haar_preflight(_preflight_metrics()),
        coupling_gate=evaluate_haar_coupling(_coupling_metrics()),
        pilot_gate=not_evaluated_gate("pilot", "sealed"),
        require_gate="coupling",
    )
    assert workflow["required_components"] == ["preflight", "coupling"]
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == "hierarchical_power_infeasible"
    with pytest.raises(ValueError, match="unknown required gate"):
        evaluate_haar_workflow(
            provenance=True,
            preflight_gate=None,
            coupling_gate=None,
            pilot_gate=None,
            require_gate="target",
        )


@pytest.mark.parametrize(
    ("domain", "expected"),
    (
        ("haar_rng", "hierarchical_rng_algebra_invalid"),
        ("normal_transform", "certified_normal_transform_invalid"),
        ("jacobi_certificate", "arbitrary_uniform_jacobi_certificate_invalid"),
        ("marginal_law", "hierarchical_marginal_law_invalid"),
        ("scheduler", "hierarchical_scheduler_invalid"),
        (
            "resource",
            "hierarchical_coupling_computationally_infeasible",
        ),
    ),
)
def test_execution_failure_classification(domain: str, expected: str) -> None:
    failed = {
        "evaluation_status": "execution_failed",
        "passed": 0,
        "failure_domain": domain,
    }
    result = _decision(
        preflight=failed,
        coupling=not_evaluated_gate("coupling", "blocked"),
        pilot=not_evaluated_gate("pilot", "blocked"),
    )
    assert result["decision"] == expected
