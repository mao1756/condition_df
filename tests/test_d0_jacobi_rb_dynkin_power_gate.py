from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_dynkin_power_gate import (
    DynkinPowerThresholds,
    build_dynkin_candidate_records,
    confirm_dynkin_design,
    decide_dynkin_power_workflow,
    evaluate_dynkin_power,
    evaluate_dynkin_preflight,
    normal_chi_square_bonferroni_projection,
    not_evaluated_gate,
    select_dynkin_panel_a_design,
)
from mnist.d0_jacobi_rb_dynkin_power_provenance import (
    PARENT_READJUDICATION,
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_RESOURCE_FEASIBLE_CANDIDATE_COUNT,
    PARENT_RUN_BASENAME,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_COUNT,
    PARENT_SOURCE_FINGERPRINT,
    verify_raw_endpoint_power_infeasible_parent,
)


def _preflight_metrics() -> dict[str, object]:
    t = DynkinPowerThresholds()
    metrics: dict[str, object] = {
        name: 1
        for name in (
            "production_authorizing_pass",
            "control_provenance_pass",
            "parent_power_adjudication_pass",
            "fifteen_parent_sources_immutable_pass",
            "parent_preflight_pass",
            "parent_power_numerically_valid_pass",
            "parent_no_work_pass",
            "path_id_plan_pass",
            "legacy_k512_id_plan_pass",
            "legacy_k512_replay_pass",
            "observer_state_hash_invariance_pass",
            "phase_moment_formula_pass",
            "phase_moment_all_colors_pass",
            "phase_moment_half_full_duration_pass",
            "phase_moment_facet_interior_pass",
            "phase_moment_zero_mass_duration_pass",
            "spectral_arb_agreement_pass",
            "cuda_enclosure_pass",
            "adversarial_p2_root_enclosure_pass",
            "cumulative_error_pass",
            "tower_panel_a_pass",
            "tower_panel_b_pass",
            "tower_joint_max_t_pass",
            "tower_panels_frozen_pass",
            "tower_panels_disjoint_pass",
            "negative_orientation_fixture_pass",
            "negative_eigenvalue_fixture_pass",
            "negative_pair_mass_fixture_pass",
            "negative_duration_fixture_pass",
            "negative_post_state_fixture_pass",
            "distribution_free_power_record_pass",
            "right_endpoint_coupling_unchanged_pass",
        )
    }
    metrics.update(
        {
            "parent_record_count": t.parent_record_count,
            "parent_source_count": t.parent_source_count,
            "grid_size": t.grid_size,
            "alpha": t.alpha,
            "tau_eff": t.tau_eff,
            "levels": list(t.levels),
            "tower_panel_count": t.tower_panel_count,
            "tower_clusters_per_panel": t.tower_clusters_per_panel,
            "tower_confidence": t.tower_confidence,
            "maximum_float64_phase_moment_error": (
                t.maximum_float64_phase_moment_error
            ),
            "maximum_cuda_phase_moment_error": (
                t.maximum_cuda_phase_moment_error
            ),
            "maximum_cumulative_standardized_error": (
                t.maximum_cumulative_standardized_error
            ),
            "certificate_fraction": 1.0,
            "fallback_fraction": t.maximum_fallback_fraction,
            "fallback_cost_fraction": t.maximum_fallback_cost_fraction,
            "peak_memory_fraction": t.maximum_peak_memory_fraction,
            "mass_error": t.maximum_cuda_mass_error,
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
    t = DynkinPowerThresholds()
    metrics: dict[str, object] = {
        name: 1
        for name in (
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
    }
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


def _candidate_records(
    role: str,
    *,
    main_width: float = 0.0025,
    reference_width: float = 0.005,
) -> list[dict[str, object]]:
    rows = []
    hours = {(32, 16): 10.0, (32, 32): 20.0, (64, 16): 30.0, (64, 32): 40.0}
    for main, reference in sorted(hours):
        rows.append(
            {
                "panel_role": role,
                "main_paths": main,
                "reference_paths": reference,
                "predicted_main_half_width": main_width,
                "predicted_reference_half_width": reference_width,
                "projected_hours": hours[(main, reference)],
                "conservative_rate": 1300.0,
                "main_variance_family_size": 120,
                "reference_variance_family_size": 40,
                "panel_complete_pass": 1,
                "panel_finite_pass": 1,
                "panel_certification_pass": 1,
                "panel_numerical_health_pass": 1,
                "mass_conservation_pass": 1,
                "shard_chain_pass": 1,
                "pilot_production_isolation_pass": 1,
                "pilot_means_excluded_pass": 1,
                "forecast_only": 1,
                "scientific_confidence_interval": 0,
            }
        )
    return rows


def _decision(
    *,
    provenance: bool = True,
    preflight: dict[str, object] | None = None,
    pilot: dict[str, object] | None = None,
) -> str:
    return decide_dynkin_power_workflow(
        provenance=provenance,
        preflight_gate=preflight or evaluate_dynkin_preflight(_preflight_metrics()),
        pilot_gate=pilot or evaluate_dynkin_power(_power_metrics()),
    )["decision"]


def test_frozen_thresholds_and_boundary_acceptance() -> None:
    thresholds = DynkinPowerThresholds()
    assert thresholds.main_feature_count == 120
    assert thresholds.reference_feature_count == 40
    assert thresholds.pilot_panel_count == 2
    assert thresholds.pilot_paths_per_panel == 8
    preflight = evaluate_dynkin_preflight(_preflight_metrics())
    assert preflight["passed"] == 1
    assert preflight["scheduler_valid"] == 1
    assert evaluate_dynkin_power(_power_metrics())["passed"] == 1


def test_test_only_evidence_cannot_authorize() -> None:
    preflight_metrics = _preflight_metrics()
    preflight_metrics["production_authorizing_pass"] = 0
    preflight = evaluate_dynkin_preflight(preflight_metrics)
    assert preflight["passed"] == 0

    power_metrics = _power_metrics()
    power_metrics["production_authorizing_pass"] = 0
    power = evaluate_dynkin_power(power_metrics)
    assert power["passed"] == 0
    decision = decide_dynkin_power_workflow(
        provenance=True,
        preflight_gate=evaluate_dynkin_preflight(_preflight_metrics()),
        pilot_gate=power,
    )
    assert decision["decision"] != "exact_dynkin_refinement_estimator_feasible"
    with pytest.raises(ValueError):
        DynkinPowerThresholds(maximum_projected_hours=48.001)


def test_normal_chi_square_projection_matches_frozen_formula() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    values = np.arange(24, dtype=np.float64).reshape(8, 3) / 10.0
    result = normal_chi_square_bonferroni_projection(
        values, candidate_paths=32
    )
    variance = np.var(values, axis=0, ddof=1)
    lower = scipy_stats.chi2.ppf(0.01 / 3.0, 7)
    sd_upper = np.sqrt(7.0 * variance / lower)
    critical = np.sqrt(2.0 * np.log(2.0 * 3.0 / 0.01))
    expected = critical * np.max(sd_upper) / np.sqrt(32.0)
    np.testing.assert_allclose(result["standard_deviation_upper"], sd_upper)
    assert result["predicted_half_width"] == pytest.approx(expected)
    assert result["forecast_only"] == 1
    assert result["scientific_confidence_interval"] == 0
    with pytest.raises(ValueError):
        normal_chi_square_bonferroni_projection(
            np.full((8, 3), np.nan), candidate_paths=32
        )


def test_candidate_builder_requires_successive_difference_family() -> None:
    rng = np.random.default_rng(991)
    main = rng.normal(scale=1.0e-5, size=(8, 120))
    reference = rng.normal(scale=1.0e-5, size=(8, 40))
    hours = {(32, 16): 10, (32, 32): 20, (64, 16): 30, (64, 32): 40}
    rows = build_dynkin_candidate_records(
        main_differences=main,
        reference_differences=reference,
        conservative_rate=2000.0,
        projected_hours_by_design=hours,
        panel_role="a",
    )
    assert len(rows) == 4
    assert {row["main_variance_family_size"] for row in rows} == {120}
    assert {row["reference_variance_family_size"] for row in rows} == {40}
    assert all(row["forecast_only"] == 1 for row in rows)
    with pytest.raises(ValueError):
        build_dynkin_candidate_records(
            main_differences=main[:, :-1],
            reference_differences=reference,
            conservative_rate=2000.0,
            projected_hours_by_design=hours,
            panel_role="a",
        )


def test_panel_a_alone_nominates_and_sealed_b_only_confirms() -> None:
    panel_a = select_dynkin_panel_a_design(_candidate_records("a"))
    assert panel_a["passed"] == 1
    assert panel_a["selected"]["main_paths"] == 32
    assert panel_a["selected"]["reference_paths"] == 16

    panel_b = _candidate_records("b")
    # Make A's nominee fail while leaving a more expensive B candidate eligible.
    panel_b[0]["predicted_main_half_width"] = 0.0025001
    combined = _candidate_records("combined")
    result = confirm_dynkin_design(panel_a, panel_b, combined)
    assert result["passed"] == 0
    assert result["selection_status"] == "panel_b_rejected"
    assert result["panel_b_nomination_performed"] == 0
    assert result["selected"] is None

    passed = confirm_dynkin_design(
        panel_a, _candidate_records("b"), _candidate_records("combined")
    )
    assert passed["passed"] == 1
    assert passed["selection_status"] == "selected"


def test_panel_a_failure_never_opens_b() -> None:
    selection = select_dynkin_panel_a_design(
        _candidate_records("a", main_width=0.0025001)
    )
    assert selection["passed"] == 0
    result = confirm_dynkin_design(selection, None, None)
    assert result["selection_status"] == "panel_a_no_eligible_design"
    assert result["panel_b_confirmation_pass"] == 0


def test_closed_decision_ladder_and_authorization() -> None:
    assert _decision(provenance=False) == "control_provenance_invalid"

    metrics = _preflight_metrics()
    metrics["control_provenance_pass"] = 0
    assert (
        _decision(preflight=evaluate_dynkin_preflight(metrics))
        == "control_provenance_invalid"
    )
    metrics = _preflight_metrics()
    metrics["path_id_plan_pass"] = 0
    preflight = evaluate_dynkin_preflight(metrics)
    assert preflight["scheduler_valid"] == 0
    assert _decision(preflight=preflight) == "refinement_scheduler_invalid"
    metrics = _preflight_metrics()
    metrics["legacy_k512_replay_pass"] = 0
    preflight = evaluate_dynkin_preflight(metrics)
    assert preflight["scheduler_valid"] == 1
    assert preflight["numerically_valid"] == 0
    assert (
        _decision(preflight=preflight)
        == "dynkin_estimator_numerically_unresolved"
    )
    metrics = _preflight_metrics()
    metrics["parent_power_adjudication_pass"] = 0
    assert (
        _decision(preflight=evaluate_dynkin_preflight(metrics))
        == "parent_power_adjudication_invalid"
    )
    metrics = _preflight_metrics()
    metrics["phase_moment_formula_pass"] = 0
    assert (
        _decision(preflight=evaluate_dynkin_preflight(metrics))
        == "jacobi_phase_moment_algebra_invalid"
    )
    metrics = _preflight_metrics()
    metrics["tower_panel_b_pass"] = 0
    assert (
        _decision(preflight=evaluate_dynkin_preflight(metrics))
        == "dynkin_tower_identity_invalid"
    )
    metrics = _preflight_metrics()
    metrics["maximum_cuda_phase_moment_error"] = 2.0001e-6
    assert (
        _decision(preflight=evaluate_dynkin_preflight(metrics))
        == "dynkin_estimator_numerically_unresolved"
    )
    metrics = _preflight_metrics()
    metrics["peak_memory_fraction"] = 0.8001
    assert (
        _decision(preflight=evaluate_dynkin_preflight(metrics))
        == "dynkin_computationally_infeasible"
    )
    assert (
        _decision(pilot=not_evaluated_gate("pilot", "not run"))
        == "dynkin_power_infeasible"
    )
    metrics = _power_metrics()
    metrics["panel_a_nomination_pass"] = 0
    metrics["selected_main_paths"] = -1
    metrics["selected_reference_paths"] = -1
    metrics["projected_production_hours"] = float("inf")
    assert (
        _decision(pilot=evaluate_dynkin_power(metrics))
        == "dynkin_power_infeasible"
    )
    metrics = _power_metrics()
    metrics["panel_b_confirmation_pass"] = 0
    assert (
        _decision(pilot=evaluate_dynkin_power(metrics))
        == "dynkin_panels_disagree"
    )
    metrics = _power_metrics()
    metrics["minimum_rate"] = 1299.99
    assert (
        _decision(pilot=evaluate_dynkin_power(metrics))
        == "dynkin_computationally_infeasible"
    )
    final = decide_dynkin_power_workflow(
        provenance=True,
        preflight_gate=evaluate_dynkin_preflight(_preflight_metrics()),
        pilot_gate=evaluate_dynkin_power(_power_metrics()),
    )
    assert final["decision"] == "exact_dynkin_refinement_estimator_feasible"
    assert final["production_refinement_patch_authorized"] == 1
    assert final["physical_training_authorized"] == 0
    assert final["sampling_authorized"] == 0


@pytest.mark.parametrize(
    ("failure_domain", "expected"),
    (
        ("scheduler", "refinement_scheduler_invalid"),
        ("configuration", "refinement_scheduler_invalid"),
        ("resource", "dynkin_computationally_infeasible"),
        ("generic", "dynkin_estimator_numerically_unresolved"),
        ("numerical", "dynkin_estimator_numerically_unresolved"),
        ("", "dynkin_estimator_numerically_unresolved"),
    ),
)
def test_execution_failures_precede_scientific_aggregates(
    failure_domain: str,
    expected: str,
) -> None:
    failed_gate = {
        "evaluation_status": "execution_failed",
        "passed": 0,
        "failure_code": "stable-test-failure",
        "failure_domain": failure_domain,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        # These zeros must not be mistaken for evaluated scientific evidence.
        "provenance_valid": 0,
        "phase_moment_algebra_valid": 0,
        "tower_identity_valid": 0,
        "numerically_valid": 0,
        "resource_valid": 0,
    }
    assert (
        _decision(
            preflight=failed_gate,
            pilot=not_evaluated_gate("pilot", "preflight failed"),
        )
        == expected
    )
    assert (
        _decision(
            preflight=evaluate_dynkin_preflight(_preflight_metrics()),
            pilot=failed_gate,
        )
        == expected
    )


def test_legacy_synthetic_stage_failures_are_numerically_unresolved() -> None:
    legacy_failure = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "subchecks": {
            "synthetic": {
                "value": 0,
                "operator": "==",
                "threshold": 1,
                "passed": 0,
            }
        },
    }
    assert (
        _decision(
            preflight=legacy_failure,
            pilot=not_evaluated_gate("pilot", "preflight failed"),
        )
        == "dynkin_estimator_numerically_unresolved"
    )
    assert (
        _decision(
            preflight=evaluate_dynkin_preflight(_preflight_metrics()),
            pilot=legacy_failure,
        )
        == "dynkin_estimator_numerically_unresolved"
    )
    synthetic_provenance_failure = {
        "evaluation_status": "execution_failed",
        "passed": 0,
    }
    assert (
        _decision(provenance=synthetic_provenance_failure)
        == "dynkin_estimator_numerically_unresolved"
    )


PARENT = (
    Path("runs/experiment12_d0_jacobi_rb_strang_refinement")
    / PARENT_RUN_BASENAME
)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Strang parent unavailable")
def test_exact_failed_parent_is_readjudicated_without_mutation() -> None:
    record = verify_raw_endpoint_power_infeasible_parent(PARENT)
    assert record["passed"] == 1
    assert record["parent_artifact_record_count"] == PARENT_REGISTRY_RECORD_COUNT
    assert record["parent_artifact_registry_sha256"] == PARENT_REGISTRY_SHA256
    assert record["parent_source_fingerprint"] == PARENT_SOURCE_FINGERPRINT
    assert record["parent_source_count"] == PARENT_SOURCE_COUNT
    assert (
        record["parent_scientific_config_sha256"]
        == PARENT_SCIENTIFIC_CONFIG_SHA256
    )
    assert record["parent_re_adjudication"] == PARENT_READJUDICATION
    assert record["parent_power_numerically_valid"] == 1
    assert record["parent_resource_sentinel_invalid"] == 1
    assert (
        record["parent_resource_feasible_candidate_count"]
        == PARENT_RESOURCE_FEASIBLE_CANDIDATE_COUNT
    )
    assert record["physical_training_authorized"] == 0
    assert record["sampling_authorized"] == 0


def test_wrong_parent_fails_closed() -> None:
    with pytest.raises(ArtifactCompatibilityError):
        verify_raw_endpoint_power_infeasible_parent(Path("tests"))
