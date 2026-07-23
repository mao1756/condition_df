from __future__ import annotations

from mnist.d0_jacobi_feasibility_gate import (
    JacobiFeasibilityThresholds,
    decide_jacobi_feasibility,
    evaluate_jacobi_controls,
    evaluate_jacobi_kernel,
)


def _kernel_metrics() -> dict[str, float | int]:
    return {
        "normalization_relative_error": 1e-12,
        "cdf_endpoint_error": 0.0,
        "detailed_balance_relative_error": 1e-12,
        "semigroup_relative_error": 1e-12,
        "eigenmoment_relative_error": 1e-12,
        "arrival_score_relative_error": 1e-10,
        "cuda_relative_error": 1e-8,
        "cuda_score_relative_error": 1e-8,
        "production_spectral_support_pass": 1,
        "uncertified_draw_count": 0,
        "resource_cap_count": 0,
        "numerical_certification_failure_count": 0,
        "negative_density_count": 0,
        "nonfinite_count": 0,
        "correction_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "renormalization_count": 0,
        "distribution_control_pass": 1,
        "projected_cache_hours": 4.0,
        "peak_memory_fraction": 0.1,
    }


def _control_metrics() -> dict[str, float | int]:
    return {
        "float64_pair_mass_error": 0.0,
        "float64_simplex_error": 1e-15,
        "cuda_pair_mass_error": 1e-7,
        "cuda_simplex_error": 1e-7,
        "dirichlet_stationarity_pass": 1,
        "full_sweep_detailed_balance_pass": 1,
        "split_reference_evaluated": 1,
        "split_fixture": "exact-state-dependent-jacobi-grid28",
        "observed_weak_order": 2.0,
        "actual_eulerian_refinement_pass": 1,
        "edge_generator_observable_error": 1e-10,
        "k512_k1024_relative_error": 1e-4,
        "k512_generator_relative_error": 1e-3,
        "deterministic_identity_error": 1e-10,
        "monte_carlo_identity_pass": 1,
        "stationary_null_pass": 1,
        "orientation_fixtures_pass": 1,
        "intervention_count": 0,
        "correction_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "renormalization_count": 0,
        "nonfinite_count": 0,
    }


def test_kernel_work_cap_is_computational_not_algebraic_failure() -> None:
    metrics = _kernel_metrics()
    metrics["uncertified_draw_count"] = 1
    metrics["resource_cap_count"] = 1
    metrics["projected_cache_hours"] = float("inf")
    gate = evaluate_jacobi_kernel(metrics)
    assert not gate["passed"]
    assert gate["numerically_valid"] == 1
    assert gate["computationally_feasible"] == 0
    decision = decide_jacobi_feasibility(
        provenance_valid=True,
        adjudication_valid=True,
        preflight_gate={"passed": 1},
        kernel_gate=gate,
        controls_gate=None,
    )
    assert decision["decision"] == "jacobi_kernel_computationally_infeasible"


def test_invalid_numerical_certificate_is_not_relabelled_as_a_resource_failure() -> None:
    metrics = _kernel_metrics()
    metrics["uncertified_draw_count"] = 1
    metrics["numerical_certification_failure_count"] = 1
    gate = evaluate_jacobi_kernel(metrics)
    decision = decide_jacobi_feasibility(
        provenance_valid=True,
        adjudication_valid=True,
        preflight_gate={"passed": 1},
        kernel_gate=gate,
        controls_gate=None,
    )
    assert gate["numerically_valid"] == 0
    assert decision["decision"] == "jacobi_kernel_numerically_unresolved"


def test_only_complete_exact_controls_authorize_following_training_patch() -> None:
    kernel = evaluate_jacobi_kernel(_kernel_metrics(), JacobiFeasibilityThresholds())
    controls = evaluate_jacobi_controls(_control_metrics())
    decision = decide_jacobi_feasibility(
        provenance_valid=True,
        adjudication_valid=True,
        preflight_gate={"passed": 1},
        kernel_gate=kernel,
        controls_gate=controls,
    )
    assert kernel["passed"] == 1
    assert controls["passed"] == 1
    assert decision["decision"] == "exact_jacobi_denoising_feasible"
    assert decision["one_image_training_authorized"] == 1
    assert decision["sampling_authorized"] == 0


def test_identity_only_failure_has_its_own_closed_decision() -> None:
    controls_metrics = _control_metrics()
    controls_metrics["orientation_fixtures_pass"] = 0
    decision = decide_jacobi_feasibility(
        provenance_valid=True,
        adjudication_valid=True,
        preflight_gate={"passed": 1},
        kernel_gate=evaluate_jacobi_kernel(_kernel_metrics()),
        controls_gate=evaluate_jacobi_controls(controls_metrics),
    )
    assert decision["decision"] == "jacobi_denoising_identity_invalid"


def test_advisory_refinement_can_never_authorize_the_eulerian_split() -> None:
    controls_metrics = _control_metrics()
    controls_metrics.update({
        "split_reference_evaluated": 0,
        "split_fixture": "not_evaluated",
        "actual_eulerian_refinement_pass": 0,
        "observed_weak_order": None,
        "k512_k1024_relative_error": None,
        "k512_generator_relative_error": None,
    })
    controls = evaluate_jacobi_controls(controls_metrics)
    decision = decide_jacobi_feasibility(
        provenance_valid=True,
        adjudication_valid=True,
        preflight_gate={"passed": 1},
        kernel_gate=evaluate_jacobi_kernel(_kernel_metrics()),
        controls_gate=controls,
    )
    assert controls["passed"] == 0
    assert decision["decision"] == "jacobi_split_reference_invalid"
    assert decision["one_image_training_authorized"] == 0
