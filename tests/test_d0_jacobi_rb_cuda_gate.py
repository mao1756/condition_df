from __future__ import annotations

import json

import pytest

from mnist.d0_jacobi_rb_cuda_gate import (
    JacobiRBCudaThresholds,
    decide_jacobi_rb_cuda_workflow,
    evaluate_jacobi_rb_cuda_certificate,
    evaluate_jacobi_rb_cuda_kernel,
    evaluate_jacobi_rb_cuda_preflight,
    evaluate_jacobi_rb_cuda_target,
    evaluate_jacobi_rb_cuda_workflow,
    not_evaluated_gate,
)


def _preflight_metrics() -> dict[str, int]:
    return {
        "control_provenance_pass": 1,
        "parent_record_count": 315,
        "parent_numerically_valid_pass": 1,
        "parent_resource_infeasible_pass": 1,
        "fused_cuda_backend_available": 1,
        "cuda_floating_contract_pass": 1,
        "frozen_runtime_match_pass": 1,
        "compile_contract_pass": 1,
        "cuda_source_fingerprint_pass": 1,
        "cubin_fingerprint_pass": 1,
        "device_identity_pass": 1,
        "directed_rounding_contract_pass": 1,
        "double_double_interval_algebra_pass": 1,
        "certified_exponential_pass": 1,
        "deterministic_replay_pass": 1,
        "forbidden_approximation_count": 0,
        "nonfinite_count": 0,
    }


def _certificate_metrics() -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        name: 1
        for name in (
            "spectral_rounding_certificate_pass",
            "cdf_interval_enclosure_pass",
            "density_interval_enclosure_pass",
            "quantile_rounding_cell_pass",
            "target_rounding_cell_pass",
            "precision_doubling_hash_pass",
            "strengthening_hash_pass",
            "fresh_arb_enclosure_pass",
            "cubin_replay_pass",
            "cuda_source_replay_pass",
        )
    }
    metrics.update(
        {
            "certificate_fraction": 1.0,
            "parent_replay_count": 294,
            "fresh_certificate_count": 512,
            "cuda_certificate_fallback_fraction": 1.0e-4,
            "cuda_certificate_fallback_cost_fraction": 0.10,
            "fresh_fallback_count": 0,
            "uncertified_count": 0,
            "parent_replay_y_bit_mismatch_count": 0,
            "parent_replay_z_bit_mismatch_count": 0,
            "resource_cap_count": 0,
            "invalid_density_count": 0,
            "approximation_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "renormalization_count": 0,
            "ambiguous_rounding_count": 0,
            "correction_count": 0,
            "nonfinite_count": 0,
        }
    )
    return metrics


def _kernel_metrics() -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        name: 1
        for name in (
            "production_support_pass",
            "cdf_endpoint_certificate_pass",
            "cdf_monotonicity_pass",
            "normalization_pass",
            "semigroup_pass",
            "detailed_balance_pass",
            "law_control_pass",
            "precision_doubling_hash_pass",
            "full_api_completed_pass",
            "state_updates_device_resident_pass",
            "in_shard_host_roundtrip_pass",
            "benchmark_output_hash_pass",
            "benchmark_final_state_hash_pass",
            "restart_shard_chain_pass",
            "warmup_pass",
            "throughput_probe_pass",
        )
    }
    metrics.update(
        {
            "cuda_pair_mass_error": 2.0e-6,
            "cuda_simplex_error": 2.0e-6,
            "cuda_kernel_max_error": 2.0e-6,
            "uncertified_draw_count": 0,
            "resource_cap_count": 0,
            "invalid_density_count": 0,
            "approximation_count": 0,
            "correction_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "renormalization_count": 0,
            "replay_bit_mismatch_count": 0,
            "nonfinite_count": 0,
            "full_path_transition_count": 1_404_928,
            "full_path_benchmark_repeats": 3,
            "warmup_transition_count": 4_096,
            "throughput_transition_count": 65_536,
            "throughput_repeats": 3,
            "maximum_backend_call_size": 4_096,
            "maximum_cuda_launch_lanes": 4_096,
            "eight_step_shards_pass": 1,
            "cuda_certificate_fallback_fraction": 1.0e-4,
            "cuda_certificate_fallback_cost_fraction": 0.10,
            "slowest_transitions_per_second": 1_300.0,
            "projected_transition_count": 89_915_392,
            "projected_cache_hours": 20.0,
            "peak_memory_fraction": 0.80,
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
    provenance: bool | int | dict = True,
    preflight: dict | None = None,
    certificate: dict | None = None,
    kernel: dict | None = None,
    target: dict | None = None,
) -> dict:
    return decide_jacobi_rb_cuda_workflow(
        provenance=provenance,
        preflight_gate=(
            preflight
            if preflight is not None
            else evaluate_jacobi_rb_cuda_preflight(_preflight_metrics())
        ),
        certificate_gate=(
            certificate
            if certificate is not None
            else evaluate_jacobi_rb_cuda_certificate(_certificate_metrics())
        ),
        kernel_gate=(
            kernel
            if kernel is not None
            else evaluate_jacobi_rb_cuda_kernel(_kernel_metrics())
        ),
        target_gate=(
            target
            if target is not None
            else evaluate_jacobi_rb_cuda_target(_target_metrics())
        ),
    )


def test_threshold_boundaries_pass_and_are_frozen() -> None:
    t = JacobiRBCudaThresholds()
    assert evaluate_jacobi_rb_cuda_certificate(_certificate_metrics())["passed"] == 1
    kernel = evaluate_jacobi_rb_cuda_kernel(_kernel_metrics())
    assert kernel["passed"] == 1
    assert kernel["numerically_valid"] == 1
    assert kernel["resource_valid"] == 1
    assert evaluate_jacobi_rb_cuda_target(_target_metrics())["passed"] == 1
    with pytest.raises(ValueError):
        JacobiRBCudaThresholds(
            maximum_cuda_certificate_fallback_fraction=(
                t.maximum_cuda_certificate_fallback_fraction * 2.0
            )
        )


@pytest.mark.parametrize(
    ("metric", "decision"),
    [
        ("fused_cuda_backend_available", "fused_cuda_backend_unavailable"),
        ("cuda_floating_contract_pass", "cuda_floating_contract_invalid"),
        ("frozen_runtime_match_pass", "cuda_floating_contract_invalid"),
        ("compile_contract_pass", "cuda_floating_contract_invalid"),
        ("cubin_fingerprint_pass", "cuda_floating_contract_invalid"),
        (
            "double_double_interval_algebra_pass",
            "double_double_interval_algebra_invalid",
        ),
        ("certified_exponential_pass", "certified_exponential_invalid"),
    ],
)
def test_preflight_rejection_ladder(metric: str, decision: str) -> None:
    metrics = _preflight_metrics()
    metrics[metric] = 0
    gate = evaluate_jacobi_rb_cuda_preflight(metrics)
    assert _decision(preflight=gate)["decision"] == decision


def test_invalid_provenance_has_first_precedence() -> None:
    assert _decision(provenance=False)["decision"] == "control_provenance_invalid"


def test_rounding_and_fallback_failures_are_distinct() -> None:
    metrics = _certificate_metrics()
    metrics["spectral_rounding_certificate_pass"] = 0
    gate = evaluate_jacobi_rb_cuda_certificate(metrics)
    assert gate["numerically_valid"] == 0
    assert _decision(certificate=gate)["decision"] == (
        "spectral_rounding_certificate_invalid"
    )

    metrics = _certificate_metrics()
    metrics["cuda_certificate_fallback_fraction"] = 1.0e-4 + 1.0e-12
    gate = evaluate_jacobi_rb_cuda_certificate(metrics)
    assert gate["numerically_valid"] == 1
    assert gate["fallback_valid"] == 0
    assert _decision(certificate=gate)["decision"] == (
        "cuda_certificate_fallback_excessive"
    )


@pytest.mark.parametrize(
    ("metric", "bad_value"),
    [
        ("parent_replay_count", 293),
        ("fresh_certificate_count", 511),
        ("parent_replay_y_bit_mismatch_count", 1),
        ("parent_replay_z_bit_mismatch_count", 1),
        ("resource_cap_count", 1),
        ("invalid_density_count", 1),
        ("approximation_count", 1),
        ("floor_count", 1),
        ("limiter_count", 1),
        ("renormalization_count", 1),
        ("strengthening_hash_pass", 0),
        ("fresh_arb_enclosure_pass", 0),
        ("cubin_replay_pass", 0),
        ("cuda_source_replay_pass", 0),
    ],
)
def test_certificate_replay_and_fresh_controls_reject(
    metric: str, bad_value: int
) -> None:
    metrics = _certificate_metrics()
    metrics[metric] = bad_value
    gate = evaluate_jacobi_rb_cuda_certificate(metrics)
    assert gate["passed"] == 0
    assert gate["numerically_valid"] == 0
    assert _decision(certificate=gate)["decision"] == (
        "spectral_rounding_certificate_invalid"
    )


def test_fresh_panel_requires_zero_arb_fallbacks_explicitly() -> None:
    metrics = _certificate_metrics()
    metrics["fresh_fallback_count"] = 1
    # Guard against a malformed producer that reports a rounded-down
    # fraction: the count itself remains authorizing.
    metrics["cuda_certificate_fallback_fraction"] = 0.0
    gate = evaluate_jacobi_rb_cuda_certificate(metrics)
    assert gate["passed"] == 0
    assert gate["numerically_valid"] == 1
    assert gate["fallback_valid"] == 0


def test_kernel_distinguishes_numerical_and_resource_failure() -> None:
    metrics = _kernel_metrics()
    metrics["cuda_kernel_max_error"] = 2.0e-6 + 1.0e-12
    kernel = evaluate_jacobi_rb_cuda_kernel(metrics)
    assert kernel["numerically_valid"] == 0
    assert _decision(kernel=kernel)["decision"] == (
        "spectral_inversion_numerically_unresolved"
    )

    metrics = _kernel_metrics()
    metrics["slowest_transitions_per_second"] = 1_300.0 - 1.0e-9
    kernel = evaluate_jacobi_rb_cuda_kernel(metrics)
    assert kernel["numerically_valid"] == 1
    assert kernel["resource_valid"] == 0
    assert _decision(kernel=kernel)["decision"] == (
        "spectral_inversion_computationally_infeasible"
    )

    metrics = _kernel_metrics()
    metrics["maximum_cuda_launch_lanes"] = 4_097
    kernel = evaluate_jacobi_rb_cuda_kernel(metrics)
    assert kernel["numerically_valid"] == 1
    assert kernel["resource_valid"] == 0
    assert _decision(kernel=kernel)["decision"] == (
        "spectral_inversion_computationally_infeasible"
    )


@pytest.mark.parametrize(
    "metric",
    [
        "state_updates_device_resident_pass",
        "in_shard_host_roundtrip_pass",
    ],
)
def test_kernel_device_residency_evidence_is_authorizing(metric: str) -> None:
    metrics = _kernel_metrics()
    metrics[metric] = 0
    kernel = evaluate_jacobi_rb_cuda_kernel(metrics)
    assert kernel["passed"] == 0
    assert kernel["numerically_valid"] == 1
    assert kernel["resource_valid"] == 0
    assert _decision(kernel=kernel)["decision"] == (
        "spectral_inversion_computationally_infeasible"
    )


def test_target_rejection_and_exact_success_never_authorize_physical_work() -> None:
    metrics = _target_metrics()
    metrics["earlier_state_input_count"] = 1
    assert _decision(target=evaluate_jacobi_rb_cuda_target(metrics))["decision"] == (
        "jacobi_rb_target_invalid"
    )

    decision = _decision()
    assert decision["decision"] == (
        "exact_jacobi_rb_cuda_kernel_and_target_feasible"
    )
    assert decision["kernel_and_target_followup_authorized"] == 1
    assert decision["state_dependent_strang_refinement_authorized"] == 1
    assert decision["physical_training_authorized"] == 0
    assert decision["sampling_authorized"] == 0
    assert decision["reverse_sampling_performed"] == 0
    json.dumps(decision, allow_nan=False)


@pytest.mark.parametrize(
    "metric",
    [
        "pair_mass_conservation_pass",
        "h_minus_two_scaling_pass",
        "invariant_beta_pass",
        "flux_sign_negative_fixtures_pass",
        "all_four_colors_pass",
        "half_full_duration_pass",
        "density_positive_certificate_pass",
        "target_unique_rounding_pass",
    ],
)
def test_required_target_fixtures_reject(metric: str) -> None:
    metrics = _target_metrics()
    metrics[metric] = 0
    target = evaluate_jacobi_rb_cuda_target(metrics)
    assert target["passed"] == 0
    assert _decision(target=target)["decision"] == "jacobi_rb_target_invalid"

    metrics = _target_metrics()
    metrics["legacy_mixture_max_absolute_error"] = 1.0e-8 + 1.0e-14
    assert evaluate_jacobi_rb_cuda_target(metrics)["passed"] == 0


def test_not_evaluated_and_missing_metrics_fail_closed() -> None:
    gate = not_evaluated_gate("jacobi_rb_cuda_certificate", "preflight failed")
    assert gate["passed"] == 0
    assert gate["subchecks"] == {}
    assert _decision(certificate=gate)["decision"] == (
        "spectral_rounding_certificate_invalid"
    )

    metrics = _kernel_metrics()
    del metrics["projected_cache_hours"]
    kernel = evaluate_jacobi_rb_cuda_kernel(metrics)
    assert kernel["passed"] == 0
    assert kernel["resource_valid"] == 0


def test_workflow_prefix_is_fail_closed_and_json_serializable() -> None:
    workflow = evaluate_jacobi_rb_cuda_workflow(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight_gate=evaluate_jacobi_rb_cuda_preflight(_preflight_metrics()),
        certificate_gate=evaluate_jacobi_rb_cuda_certificate(
            _certificate_metrics()
        ),
        kernel_gate=evaluate_jacobi_rb_cuda_kernel(_kernel_metrics()),
        target_gate=evaluate_jacobi_rb_cuda_target(_target_metrics()),
        require_gate="target",
    )
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == (
        "exact_jacobi_rb_cuda_kernel_and_target_feasible"
    )
    json.dumps(workflow, allow_nan=False)
    with pytest.raises(ValueError):
        evaluate_jacobi_rb_cuda_workflow(
            provenance=True,
            preflight_gate={},
            certificate_gate={},
            kernel_gate={},
            target_gate={},
            require_gate="training",
        )
