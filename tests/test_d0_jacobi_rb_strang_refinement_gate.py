from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_strang_refinement_gate import (
    StrangRefinementThresholds,
    center_scale_observable,
    decide_strang_refinement_workflow,
    dirichlet_linear_moments,
    dirichlet_observable_moments,
    evaluate_refinement_power,
    evaluate_strang_preflight,
    evaluate_strang_refinement,
    fit_weak_order,
    not_evaluated_gate,
    richardson_reference,
    select_refinement_design,
    whole_path_max_t_intervals,
    whole_path_refinement_bootstrap,
)
from mnist.d0_jacobi_rb_strang_refinement_provenance import (
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_RUN_BASENAME,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_FINGERPRINT,
    verify_exact_jacobi_rb_multipath_parent,
)


def _preflight_metrics() -> dict[str, object]:
    t = StrangRefinementThresholds()
    metrics: dict[str, object] = {
        name: 1
        for name in (
            "control_provenance_pass",
            "parent_kernel_gate_pass",
            "parent_target_gate_pass",
            "parent_strang_authorized_pass",
            "eleven_parent_sources_immutable_pass",
            "parent_scientific_config_pass",
            "parent_no_work_pass",
            "variable_k_exposure_pass",
            "palindromic_phase_order_pass",
            "nested_id_uniqueness_pass",
            "nested_id_aliasing_exact_pass",
            "nested_id_marginal_law_pass",
            "nested_id_order_invariance_pass",
            "nested_id_resume_invariance_pass",
            "legacy_k512_replay_pass",
            "local_generator_mixed_digit_pass",
            "local_generator_interior_fixture_pass",
            "k1024_support_certificate_pass",
            "k2048_support_certificate_pass",
            "stationarity_panel_a_pass",
            "stationarity_panel_b_pass",
            "stationarity_joint_max_t_pass",
            "stationarity_panels_immutable_pass",
            "stationarity_panel_disjoint_pass",
        )
    }
    metrics.update(
        {
            "parent_record_count": t.parent_record_count,
            "grid_size": t.grid_size,
            "alpha": t.alpha,
            "tau_eff": t.tau_eff,
            "levels": [*t.levels, t.reference_level],
            "stationarity_panel_count": t.preflight_panel_count,
            "stationarity_paths_per_panel": t.preflight_paths_per_panel,
            "stationarity_transitions_per_path": t.preflight_transitions_per_path,
            "local_generator_max_error": t.maximum_local_generator_error,
            "minimum_support_rate": t.minimum_rate,
            "certificate_fraction": 1.0,
            "fallback_fraction": t.maximum_fallback_fraction,
            "fallback_cost_fraction": t.maximum_fallback_cost_fraction,
            "peak_memory_fraction": t.maximum_peak_memory_fraction,
            "uncertified_count": 0,
            "resource_cap_count": 0,
            "invalid_density_count": 0,
            "approximation_count": 0,
            "correction_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "projection_count": 0,
            "renormalization_count": 0,
            "nonfinite_count": 0,
        }
    )
    return metrics


def _power_metrics() -> dict[str, object]:
    t = StrangRefinementThresholds()
    metrics: dict[str, object] = {
        name: 1
        for name in (
            "pilot_complete_pass",
            "pilot_finite_pass",
            "pilot_paths_disjoint_from_preflight_pass",
            "pilot_paths_disjoint_from_production_pass",
            "pilot_means_excluded_pass",
            "variance_only_selection_pass",
            "complete_candidate_grid_pass",
            "selected_design_frozen_pass",
            "selected_design_hash_pass",
            "pilot_certification_pass",
        )
    }
    metrics.update(
        {
            "pilot_main_paths": t.pilot_main_paths,
            "pilot_reference_paths": t.pilot_reference_paths,
            "candidate_main_paths": list(t.candidate_main_paths),
            "candidate_reference_paths": list(t.candidate_reference_paths),
            "selected_main_paths": 32,
            "selected_reference_paths": 16,
            "predicted_main_half_width": t.maximum_main_half_width,
            "predicted_reference_half_width": t.maximum_reference_half_width,
            "projected_production_hours": t.maximum_projected_hours,
            "certificate_fraction": 1.0,
            "uncertified_count": 0,
            "resource_cap_count": 0,
            "invalid_density_count": 0,
            "approximation_count": 0,
            "correction_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "projection_count": 0,
            "renormalization_count": 0,
            "nonfinite_count": 0,
        }
    )
    return metrics


def _refinement_metrics() -> dict[str, object]:
    t = StrangRefinementThresholds()
    metrics: dict[str, object] = {
        name: 1
        for name in (
            "selected_design_binding_pass",
            "production_pilot_isolation_pass",
            "production_complete_pass",
            "production_finite_pass",
            "all_levels_complete_pass",
            "reference_level_subset_pass",
            "observation_plan_pass",
            "observable_family_plan_pass",
            "dirichlet_normalization_pass",
            "paired_whole_path_bootstrap_pass",
            "linear_family_pass",
            "quadratic_family_pass",
            "cubic_family_pass",
            "pooled_family_pass",
            "stationarity_panel_a_pass",
            "stationarity_panel_b_pass",
            "stationarity_all_levels_pass",
            "stationarity_eight_sweep_k512_pass",
            "detailed_balance_max_t_pass",
            "mass_conservation_pass",
            "shard_chain_pass",
            "state_updates_device_resident_pass",
        )
    }
    metrics.update(
        {
            "image_sha256": t.required_image_sha256,
            "levels": [*t.levels, t.reference_level],
            "observation_time_fractions": list(t.observation_time_fractions),
            "bootstrap_replicates": t.bootstrap_replicates,
            "bootstrap_confidence": t.bootstrap_confidence,
            "minimum_observed_weak_order": t.minimum_observed_weak_order,
            "minimum_weak_order_interval_lower": (
                t.minimum_weak_order_interval_lower
            ),
            "weak_order_two_coverage_fraction": 1.0,
            "maximum_512_1024_upper_bound": t.maximum_512_1024_discrepancy,
            "maximum_512_reference_upper_bound": t.maximum_512_reference_error,
            "maximum_reference_stability_upper_bound": (
                t.maximum_reference_instability
            ),
            "mass_error": t.maximum_cuda_mass_error,
            "certificate_fraction": 1.0,
            "projected_or_actual_hours": t.maximum_projected_hours,
            "uncertified_count": 0,
            "resource_cap_count": 0,
            "invalid_density_count": 0,
            "approximation_count": 0,
            "correction_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "projection_count": 0,
            "renormalization_count": 0,
            "nonfinite_count": 0,
        }
    )
    return metrics


def _decision(
    *,
    provenance: bool = True,
    preflight: dict | None = None,
    power: dict | None = None,
    refinement: dict | None = None,
) -> str:
    return decide_strang_refinement_workflow(
        provenance=provenance,
        preflight_gate=preflight or evaluate_strang_preflight(_preflight_metrics()),
        power_gate=power or evaluate_refinement_power(_power_metrics()),
        refinement_gate=refinement
        or evaluate_strang_refinement(_refinement_metrics()),
    )["decision"]


def test_frozen_thresholds_and_boundary_acceptance() -> None:
    thresholds = StrangRefinementThresholds()
    assert thresholds.levels == (128, 256, 512, 1024)
    assert thresholds.reference_level == 2048
    assert thresholds.bootstrap_replicates == 20_000
    assert evaluate_strang_preflight(_preflight_metrics())["passed"] == 1
    assert evaluate_refinement_power(_power_metrics())["passed"] == 1
    assert evaluate_strang_refinement(_refinement_metrics())["passed"] == 1
    with pytest.raises(ValueError):
        StrangRefinementThresholds(maximum_projected_hours=48.01)


def test_exact_dirichlet_observable_normalization() -> None:
    moments = dirichlet_observable_moments(cell_count=2, alpha=1.0)
    quadratic = moments["quadratic_power_sum"]
    assert quadratic["mean"] == pytest.approx(2.0 / 3.0)
    assert quadratic["second_moment"] == pytest.approx(7.0 / 15.0)
    assert quadratic["variance"] == pytest.approx(1.0 / 45.0)
    cubic = moments["cubic_power_sum"]
    assert cubic["mean"] == pytest.approx(0.5)
    linear = dirichlet_linear_moments([-1.0, 1.0], alpha=1.0)
    assert linear["mean"] == 0.0
    assert linear["variance"] == pytest.approx(1.0 / 3.0)
    standardized = center_scale_observable(
        [quadratic["mean"]], mean=quadratic["mean"], rms=quadratic["rms"]
    )
    np.testing.assert_array_equal(standardized, np.zeros(1))


def test_order_and_richardson_are_exact_for_second_order_fixture() -> None:
    truth = np.asarray([0.3, -0.2])
    coefficient = np.asarray([4.0, -2.0])
    means = {
        level: truth + coefficient / float(level * level)
        for level in (128, 256, 512, 1024, 2048)
    }
    np.testing.assert_allclose(
        fit_weak_order(means, levels=(128, 256, 512)),
        np.full(2, 2.0),
        rtol=0.0,
        atol=1.0e-12,
    )
    record = richardson_reference(means[512], means[1024], means[2048])
    np.testing.assert_allclose(record["reference"], truth, atol=1.0e-15)
    np.testing.assert_allclose(record["lower_reference"], truth, atol=1.0e-15)


def test_two_sided_max_t_is_deterministic_and_fail_closed() -> None:
    values = {
        "degree-1": np.asarray([-1.0, 0.0, 1.0, 0.0]),
        "degree-2": np.asarray([0.5, -0.5, 0.5, -0.5]),
    }
    first = whole_path_max_t_intervals(values, seed=7, reps=500, chunk_size=37)
    second = whole_path_max_t_intervals(
        dict(reversed(list(values.items()))), seed=7, reps=500, chunk_size=91
    )
    assert first["passed"] == 1
    assert first["critical_value"] == second["critical_value"]
    assert first["members"] == second["members"]
    with pytest.raises(ValueError):
        whole_path_max_t_intervals(
            {"bad": [1.0, 1.0, 1.0]}, seed=1, reps=10
        )


def test_joint_refinement_bootstrap_tracks_aligned_whole_paths() -> None:
    rng = np.random.default_rng(19)
    path_count = 24
    reference_paths = 12
    shared = rng.normal(scale=0.02, size=(path_count, 2))
    coefficients = np.asarray([2.0, -1.0])
    level_values = {
        level: shared + coefficients[None, :] / float(level * level)
        for level in (128, 256, 512, 1024)
    }
    level_values[2048] = (
        shared[:reference_paths]
        + coefficients[None, :] / float(2048 * 2048)
    )
    result = whole_path_refinement_bootstrap(
        level_values, seed=31, reps=300, chunk_size=37
    )
    differently_chunked = whole_path_refinement_bootstrap(
        level_values, seed=31, reps=300, chunk_size=128
    )
    assert result["valid"] == 1
    assert result == differently_chunked
    assert result["main_path_count"] == path_count
    assert result["reference_path_count"] == reference_paths
    assert all(
        row["observed_weak_order"] == pytest.approx(2.0, abs=1.0e-8)
        for row in result["feature_metrics"]
    )


def test_design_selection_is_complete_variance_only_and_cheapest() -> None:
    rows = []
    for main in (32, 64):
        for reference in (16, 32):
            rows.append(
                {
                    "main_paths": main,
                    "reference_paths": reference,
                    "predicted_main_half_width": 0.002 if main == 64 else 0.003,
                    "predicted_reference_half_width": (
                        0.004 if reference == 32 else 0.006
                    ),
                    "projected_hours": main / 4 + reference / 8,
                    "variance_only_pass": 1,
                    "pilot_production_isolation_pass": 1,
                    "pilot_means_excluded_pass": 1,
                }
            )
    record = select_refinement_design(rows)
    assert record["passed"] == 1
    assert record["eligible_candidate_count"] == 1
    assert record["selected"]["main_paths"] == 64
    assert record["selected"]["reference_paths"] == 32
    with pytest.raises(ValueError):
        select_refinement_design(rows[:-1])


def test_closed_decision_ladder_and_authorization() -> None:
    assert _decision(provenance=False) == "control_provenance_invalid"
    metrics = _preflight_metrics()
    metrics["nested_id_uniqueness_pass"] = 0
    assert _decision(
        preflight=evaluate_strang_preflight(metrics)
    ) == "refinement_scheduler_invalid"
    metrics = _preflight_metrics()
    metrics["stationarity_panel_b_pass"] = 0
    assert _decision(
        preflight=evaluate_strang_preflight(metrics)
    ) == "jacobi_stationarity_control_invalid"
    metrics = _preflight_metrics()
    metrics["local_generator_max_error"] = 1.0001e-8
    assert _decision(
        preflight=evaluate_strang_preflight(metrics)
    ) == "strang_split_reference_invalid"
    metrics = _preflight_metrics()
    metrics["k2048_support_certificate_pass"] = 0
    assert _decision(
        preflight=evaluate_strang_preflight(metrics)
    ) == "refinement_kernel_numerically_unresolved"
    metrics = _preflight_metrics()
    metrics["minimum_support_rate"] = 1_299.99
    assert _decision(
        preflight=evaluate_strang_preflight(metrics)
    ) == "refinement_computationally_infeasible"
    assert _decision(
        power=not_evaluated_gate("power", "not run")
    ) == "refinement_power_infeasible"
    metrics = _power_metrics()
    metrics["predicted_main_half_width"] = 0.0025001
    assert _decision(
        power=evaluate_refinement_power(metrics)
    ) == "refinement_power_infeasible"
    metrics = _power_metrics()
    metrics["projected_production_hours"] = 48.001
    assert _decision(
        power=evaluate_refinement_power(metrics)
    ) == "refinement_computationally_infeasible"
    metrics = _refinement_metrics()
    metrics["maximum_reference_stability_upper_bound"] = 0.005001
    assert _decision(
        refinement=evaluate_strang_refinement(metrics)
    ) == "refinement_reference_inconclusive"
    metrics = _refinement_metrics()
    metrics["stationarity_panel_a_pass"] = 0
    assert _decision(
        refinement=evaluate_strang_refinement(metrics)
    ) == "strang_split_reference_invalid"
    metrics = _refinement_metrics()
    metrics["linear_family_pass"] = 0
    assert _decision(
        refinement=evaluate_strang_refinement(metrics)
    ) == "strang_refinement_inconclusive"
    final = decide_strang_refinement_workflow(
        provenance=True,
        preflight_gate=evaluate_strang_preflight(_preflight_metrics()),
        power_gate=evaluate_refinement_power(_power_metrics()),
        refinement_gate=evaluate_strang_refinement(_refinement_metrics()),
    )
    assert final["decision"] == "exact_state_dependent_strang_refinement_feasible"
    assert final["one_image_phase_conditioned_training_patch_authorized"] == 1
    assert final["physical_training_authorized"] == 0
    assert final["sampling_authorized"] == 0


PARENT = (
    Path("runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation")
    / PARENT_RUN_BASENAME
)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable multipath parent unavailable")
def test_exact_successful_parent_authorizes_only_strang_refinement() -> None:
    record = verify_exact_jacobi_rb_multipath_parent(PARENT)
    assert record["passed"] == 1
    assert record["parent_artifact_record_count"] == PARENT_REGISTRY_RECORD_COUNT
    assert record["parent_artifact_registry_sha256"] == PARENT_REGISTRY_SHA256
    assert record["parent_source_fingerprint"] == PARENT_SOURCE_FINGERPRINT
    assert (
        record["parent_scientific_config_sha256"]
        == PARENT_SCIENTIFIC_CONFIG_SHA256
    )
    assert record["parent_source_count"] == 11
    assert record["parent_kernel_pass"] == 1
    assert record["parent_target_pass"] == 1
    assert record["state_dependent_strang_refinement_authorized"] == 1
    assert record["physical_training_authorized"] == 0
    assert record["sampling_authorized"] == 0


def test_wrong_parent_fails_closed() -> None:
    with pytest.raises(ArtifactCompatibilityError):
        verify_exact_jacobi_rb_multipath_parent(Path("tests"))
