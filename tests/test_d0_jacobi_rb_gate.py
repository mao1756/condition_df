from __future__ import annotations

import pytest

from mnist.d0_jacobi_rb_gate import (
    JacobiRBThresholds,
    decide_jacobi_rb_workflow,
    evaluate_jacobi_rb_kernel,
    evaluate_jacobi_rb_preflight,
    evaluate_jacobi_rb_target,
    evaluate_jacobi_rb_workflow,
    not_evaluated_gate,
)


def _preflight_metrics() -> dict[str, int]:
    return {
        "parent_provenance_pass": 1,
        "parent_record_count": 16,
        "parent_reclassification_pass": 1,
        "arb_backend_available": 1,
        "python_flint_exact_version_pass": 1,
        "arb_outward_rounding_pass": 1,
        "gpu_interval_enclosure_pass": 1,
        "alpha1_legendre_formula_pass": 1,
        "jacobi_wf_clock_factor_pass": 1,
        "head_fraction_orientation_pass": 1,
        "stable_conormal_formula_pass": 1,
        "lazy_dyadic_uniform_pass": 1,
        "rounding_cell_contract_pass": 1,
        "cantelli_bracket_pass": 1,
        "forbidden_approximation_count": 0,
        "nonfinite_count": 0,
    }


def _kernel_metrics() -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        name: 1
        for name in (
            "adversarial_support_pass",
            "support_case_count_pass",
            "cdf_endpoint_certificate_pass",
            "cdf_monotonicity_pass",
            "spectral_tail_enclosure_pass",
            "roundoff_enclosure_pass",
            "normalization_pass",
            "semigroup_pass",
            "detailed_balance_pass",
            "law_control_pass",
            "moment_control_pass",
            "eigenmoment_control_pass",
            "stationarity_control_pass",
            "reversibility_control_pass",
            "precision_doubling_hash_pass",
            "benchmark_output_hash_pass",
            "full_api_completed_pass",
            "cuda_evaluated_pass",
        )
    }
    metrics.update(
        {
            "quantile_certificate_fraction": 1.0,
            "float64_pair_mass_error": 1.0e-13,
            "float64_simplex_error": 1.0e-13,
            "cuda_pair_mass_error": 1.0e-7,
            "cuda_simplex_error": 1.0e-7,
            "float64_kernel_max_error": 1.0e-10,
            "cuda_kernel_max_error": 1.0e-7,
            "full_path_transition_count": 1_404_928,
            "full_path_benchmark_repeats": 3,
            "slowest_transitions_per_second": 1_300.0,
            "projected_transition_count": 89_915_392,
            "projected_cache_hours": 20.0,
            "peak_memory_fraction": 0.8,
            "arb_fallback_fraction": 1.0e-4,
            "arb_cost_fraction": 0.1,
        }
    )
    metrics.update(
        {
            name: 0
            for name in (
                "uncertified_draw_count",
                "resource_cap_count",
                "approximation_count",
                "gaussian_fallback_count",
                "euler_fallback_count",
                "finite_ancestral_proxy_count",
                "exposure_binning_count",
                "replay_y_bit_mismatch_count",
                "correction_count",
                "floor_count",
                "limiter_count",
                "renormalization_count",
                "negative_state_count",
                "nonfinite_count",
            )
        }
    )
    return metrics


def _target_metrics() -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        name: 1
        for name in (
            "rao_blackwell_identity_pass",
            "population_tower_identity_pass",
            "latent_mixture_equivalence_pass",
            "density_positive_certificate_pass",
            "target_unique_rounding_pass",
            "conormal_orientation_pass",
            "synthetic_teacher_pass",
            "stationary_null_pass",
            "all_phase_colors_pass",
            "half_full_duration_pass",
            "negative_fixtures_pass",
            "later_state_only_input_pass",
            "cuda_target_evaluated_pass",
        )
    }
    metrics.update(
        {
            "target_certificate_fraction": 1.0,
            "rb_identity_relative_error": 1.0e-8,
            "cuda_rb_relative_error": 2.0e-5,
            "legacy_mixture_max_absolute_error": 1.0e-8,
        }
    )
    metrics.update(
        {
            name: 0
            for name in (
                "target_uncertified_count",
                "target_resource_cap_count",
                "target_replay_bit_mismatch_count",
                "target_nonfinite_count",
                "earlier_state_input_count",
                "latent_variable_input_count",
                "classifier_target_count",
                "value_target_count",
                "h1_target_count",
                "raw_euler_residual_target_count",
                "gaussian_target_count",
                "target_clip_count",
            )
        }
    )
    return metrics


def _decision(
    *,
    preflight: dict | None = None,
    kernel: dict | None = None,
    target: dict | None = None,
) -> dict:
    return decide_jacobi_rb_workflow(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight_gate=preflight or evaluate_jacobi_rb_preflight(_preflight_metrics()),
        kernel_gate=kernel or evaluate_jacobi_rb_kernel(_kernel_metrics()),
        target_gate=target or evaluate_jacobi_rb_target(_target_metrics()),
    )


def test_frozen_target_error_thresholds_match_the_approved_plan() -> None:
    thresholds = JacobiRBThresholds()
    assert thresholds.maximum_rb_identity_relative_error == 1.0e-8
    assert thresholds.maximum_cuda_rb_relative_error == 2.0e-5
    assert evaluate_jacobi_rb_target(_target_metrics())["passed"] == 1
    with pytest.raises(ValueError):
        JacobiRBThresholds(maximum_rb_identity_relative_error=1.0e-7)


def test_not_evaluated_is_never_an_empty_success() -> None:
    gate = not_evaluated_gate("jacobi_rb_kernel", "preflight failed")
    assert gate["evaluation_status"] == "not_evaluated"
    assert gate["subchecks"] == {}
    assert gate["passed"] == 0
    decision = _decision(kernel=gate)
    assert decision["decision"] == "spectral_inversion_numerically_unresolved"
    assert decision["physical_training_authorized"] == 0


def test_preflight_distinguishes_missing_backend_interval_and_algebra() -> None:
    metrics = _preflight_metrics()
    metrics["arb_backend_available"] = 0
    assert _decision(preflight=evaluate_jacobi_rb_preflight(metrics))["decision"] == (
        "certified_backend_unavailable"
    )

    metrics = _preflight_metrics()
    metrics["gpu_interval_enclosure_pass"] = 0
    assert _decision(preflight=evaluate_jacobi_rb_preflight(metrics))["decision"] == (
        "spectral_interval_backend_invalid"
    )

    metrics = _preflight_metrics()
    metrics["jacobi_wf_clock_factor_pass"] = 0
    assert _decision(preflight=evaluate_jacobi_rb_preflight(metrics))["decision"] == (
        "spectral_cdf_algebra_invalid"
    )


def test_certified_but_slow_kernel_is_only_a_computational_failure() -> None:
    metrics = _kernel_metrics()
    metrics["slowest_transitions_per_second"] = 1_299.999
    kernel = evaluate_jacobi_rb_kernel(metrics)
    assert kernel["numerically_valid"] == 1
    assert kernel["resource_valid"] == 0
    assert _decision(kernel=kernel)["decision"] == (
        "spectral_inversion_computationally_infeasible"
    )


def test_uncertified_or_approximate_draw_is_a_numerical_failure() -> None:
    for name in ("uncertified_draw_count", "gaussian_fallback_count"):
        metrics = _kernel_metrics()
        metrics[name] = 1
        kernel = evaluate_jacobi_rb_kernel(metrics)
        assert kernel["numerically_valid"] == 0
        assert _decision(kernel=kernel)["decision"] == (
            "spectral_inversion_numerically_unresolved"
        )


def test_identity_only_and_input_leakage_failures_have_distinct_decisions() -> None:
    metrics = _target_metrics()
    metrics["rao_blackwell_identity_pass"] = 0
    target = evaluate_jacobi_rb_target(metrics)
    assert _decision(target=target)["decision"] == "rao_blackwell_identity_invalid"

    metrics = _target_metrics()
    metrics["earlier_state_input_count"] = 1
    target = evaluate_jacobi_rb_target(metrics)
    assert _decision(target=target)["decision"] == "jacobi_rb_target_invalid"


def test_only_complete_exact_kernel_and_target_authorize_strang_refinement() -> None:
    decision = _decision()
    assert decision["decision"] == "exact_jacobi_rb_kernel_feasible"
    assert decision["strang_refinement_authorized"] == 1
    assert decision["one_image_training_authorized"] == 0
    assert decision["physical_training_authorized"] == 0
    assert decision["sampling_authorized"] == 0

    workflow = evaluate_jacobi_rb_workflow(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight_gate=evaluate_jacobi_rb_preflight(_preflight_metrics()),
        kernel_gate=evaluate_jacobi_rb_kernel(_kernel_metrics()),
        target_gate=evaluate_jacobi_rb_target(_target_metrics()),
        require_gate="target",
    )
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == "exact_jacobi_rb_kernel_feasible"


def test_missing_required_metric_fails_closed() -> None:
    metrics = _kernel_metrics()
    del metrics["arb_fallback_fraction"]
    gate = evaluate_jacobi_rb_kernel(metrics)
    assert gate["passed"] == 0
    assert gate["resource_valid"] == 0

    metrics = _kernel_metrics()
    metrics["quantile_certificate_fraction"] = 1.01
    gate = evaluate_jacobi_rb_kernel(metrics)
    assert gate["passed"] == 0
    assert gate["numerically_valid"] == 0
