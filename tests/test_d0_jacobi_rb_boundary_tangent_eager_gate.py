from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import nextafter

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_eager_gate import (
    CACHE_DESIGN_FLAGS,
    CACHE_EXECUTION_FLAGS,
    CONFIRM_EXECUTION_FLAGS,
    FINAL_DECISION,
    PREFLIGHT_ADJUDICATION_FLAGS,
    PREFLIGHT_DESIGN_FLAGS,
    PREFLIGHT_PROVENANCE_FLAGS,
    PREFLIGHT_REPRESENTATION_FLAGS,
    PREFLIGHT_SCHEDULE_FLAGS,
    TRAIN_BASELINE_FLAGS,
    TRAIN_OPTIMIZATION_FLAGS,
    BoundaryTangentEagerDecision,
    BoundaryTangentEagerGateError,
    BoundaryTangentEagerThresholds,
    decide_workflow,
    evaluate_cache_gate,
    evaluate_confirm_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    evaluate_train_gate,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    COMBINED_VS_BASELINE_NAMES,
    COMBINED_VS_ZERO_NAMES,
    CONFIRMATION_FAMILY_NAMES,
)


def _preflight_metrics() -> dict[str, object]:
    t = BoundaryTangentEagerThresholds()
    metrics: dict[str, object] = {
        name: 1
        for name in (
            PREFLIGHT_PROVENANCE_FLAGS
            + PREFLIGHT_ADJUDICATION_FLAGS
            + PREFLIGHT_REPRESENTATION_FLAGS
            + PREFLIGHT_SCHEDULE_FLAGS
            + PREFLIGHT_DESIGN_FLAGS
        )
    }
    metrics.update(
        {
            "eager_parent_record_count": t.eager_parent_record_count,
            "controller_v1_parent_record_count": (
                t.controller_v1_parent_record_count
            ),
            "root_seed": t.root_seed,
            "model_seeds": list(t.model_seeds),
            "reserved_control_seed": t.reserved_control_seed,
            "bootstrap_seed": t.bootstrap_seed,
            "synthetic_teacher_seed": t.synthetic_teacher_seed,
            "baseline_null_seed": t.baseline_null_seed,
            "training_path_ids": list(t.training_path_ids),
            "validation_path_ids": list(t.validation_path_ids),
            "confirmation_path_ids": list(t.confirmation_path_ids),
            "preflight_seam_path_ids": list(t.preflight_seam_path_ids),
            "forbidden_historical_v1_path_ids": list(
                t.forbidden_historical_v1_path_ids
            ),
            "train_validation_cohort_sizes": list(
                t.train_validation_cohort_sizes
            ),
            "confirmation_cohort_sizes": list(t.confirmation_cohort_sizes),
            "training_paths": t.training_paths,
            "validation_paths": t.validation_paths,
            "confirmation_paths": t.confirmation_paths,
            "projected_total_transitions": t.total_transitions,
            "projected_base_transitions": t.projected_base_transitions,
            "projected_midpoint_transitions": t.projected_midpoint_transitions,
            "candidate_modes": t.candidate_modes,
            "certificate_fraction": 1.0,
            "forbidden_event_count": 0,
            "projected_elapsed_seconds": 90_000.0,
            "projected_effective_rate": t.total_transitions / 90_000.0,
            "minimum_profile_rate": 2_000.0,
            "fallback_fraction": 0.0,
            "fallback_time_fraction": 0.0,
            "maximum_mass_error": 1.0e-13,
            "peak_memory_fraction": 0.5,
            "projected_persisted_bytes": 1_024,
            "maximum_launch_lanes": 2_048,
            "production_cache_generation_performed": 0,
            "physical_training_performed": 0,
            "confirmation_performed": 0,
        }
    )
    return metrics


def _cache_metrics() -> dict[str, object]:
    t = BoundaryTangentEagerThresholds()
    metrics: dict[str, object] = {
        name: 1 for name in CACHE_EXECUTION_FLAGS + CACHE_DESIGN_FLAGS
    }
    metrics.update(
        {
            "train_row_count": t.train_rows,
            "validation_row_count": t.validation_rows,
            "train_transition_count": t.train_transitions,
            "validation_transition_count": t.validation_transitions,
            "cache_transition_count": t.train_transitions
            + t.validation_transitions,
            "certificate_fraction": 1.0,
            "maximum_mass_error": 1.0e-13,
            "forbidden_event_count": 0,
            "minimum_role_rate": 2_000.0,
            "cache_elapsed_seconds": 60_000.0,
            "frozen_conservative_confirmation_projection_seconds": 30_000.0,
            "fallback_fraction": 0.0,
            "fallback_time_fraction": 0.0,
            "peak_memory_fraction": 0.5,
            "total_persisted_cache_bytes": 1_024,
            "maximum_launch_lanes": 2_048,
            "production_cache_generation_performed": 1,
            "physical_training_performed": 0,
            "confirmation_performed": 0,
        }
    )
    return metrics


def _train_metrics() -> dict[str, object]:
    t = BoundaryTangentEagerThresholds()
    metrics: dict[str, object] = {
        name: 1 for name in TRAIN_OPTIMIZATION_FLAGS + TRAIN_BASELINE_FLAGS
    }
    metrics.update(
        {
            "synthetic_relative_validation_mse": 0.005,
            "null_selected_update": 0,
            "model_seed_count": len(t.model_seeds),
            "maximum_updates": t.maximum_updates,
            "quotient_target_formed": 0,
            "selected_nonzero": 1,
            "production_cache_generation_performed": 1,
            "physical_training_performed": 1,
            "confirmation_performed": 0,
        }
    )
    return metrics


def _max_t_record() -> dict[str, object]:
    t = BoundaryTangentEagerThresholds()
    return {
        "method": "centered_whole_path_studentized_max_t",
        "bootstrap_unit": "whole_path_jointly_across_family",
        "quantile_method": "higher",
        "family_size": len(CONFIRMATION_FAMILY_NAMES),
        "family_names": list(CONFIRMATION_FAMILY_NAMES),
        "path_count": t.confirmation_paths,
        "path_ids": list(t.confirmation_path_ids),
        "confidence": t.simultaneous_confidence,
        "replicates": t.bootstrap_replicates,
        "seed": t.bootstrap_seed,
        "namespace": 0,
        "negative_values_truncated": 0,
        "critical_value": 2.5,
        "point_estimates": {
            name: 1.0 for name in CONFIRMATION_FAMILY_NAMES
        },
        "standard_errors": {
            name: 0.1 for name in CONFIRMATION_FAMILY_NAMES
        },
        "lower_bounds": {
            name: 0.5 for name in CONFIRMATION_FAMILY_NAMES
        },
        "passed": 1,
    }


def _confirm_metrics() -> dict[str, object]:
    t = BoundaryTangentEagerThresholds()
    metrics: dict[str, object] = {
        name: 1 for name in CONFIRM_EXECUTION_FLAGS
    }
    metrics.update(
        {
            "confirmation_path_count": t.confirmation_paths,
            "confirmation_row_count": t.confirmation_rows,
            "confirmation_transition_count": t.confirmation_transitions,
            "certificate_fraction": 1.0,
            "maximum_mass_error": 1.0e-13,
            "forbidden_event_count": 0,
            "transitions_per_second": 2_000.0,
            "cache_elapsed_seconds": 60_000.0,
            "confirmation_elapsed_seconds": 30_000.0,
            "fallback_fraction": 0.0,
            "fallback_time_fraction": 0.0,
            "peak_memory_fraction": 0.5,
            "total_persisted_bytes": 1_024,
            "maximum_launch_lanes": 2_048,
            "production_cache_generation_performed": 1,
            "physical_training_performed": 1,
            "confirmation_performed": 1,
        }
    )
    return metrics


def _passing_gates() -> tuple[dict[str, object], ...]:
    return (
        evaluate_preflight_gate(_preflight_metrics()),
        evaluate_cache_gate(_cache_metrics()),
        evaluate_train_gate(_train_metrics()),
        evaluate_confirm_gate(
            _max_t_record(),
            _confirm_metrics(),
            integrity_checks={"sealed_confirmation_valid": True},
        ),
    )


def _decision(
    preflight: dict[str, object],
    cache: dict[str, object],
    train: dict[str, object],
    confirm: dict[str, object],
) -> dict[str, object]:
    return decide_workflow(
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train,
        confirm_gate=confirm,
    )


def test_thresholds_are_frozen_at_the_exact_v2_design() -> None:
    t = BoundaryTangentEagerThresholds()
    assert (t.training_paths, t.validation_paths, t.confirmation_paths) == (
        64,
        32,
        64,
    )
    assert t.total_transitions == 337_182_720
    assert t.root_seed == 261_311
    assert t.reserved_control_seed == 261_316
    assert t.synthetic_teacher_seed == 261_317
    assert t.baseline_null_seed == 261_318
    assert t.train_validation_cohort_sizes == (10,) * 9 + (6,)
    assert t.confirmation_cohort_sizes == (10,) * 6 + (4,)
    assert t.preflight_seam_path_ids == tuple(range(0xEF000, 0xEF008))
    assert t.forbidden_historical_v1_path_ids == tuple(
        range(0xEC000, 0xEC008)
    )
    with pytest.raises(BoundaryTangentEagerGateError, match="root_seed"):
        BoundaryTangentEagerThresholds(root_seed=7)
    with pytest.raises(FrozenInstanceError):
        t.root_seed = 7  # type: ignore[misc]
    assert tuple(item.value for item in BoundaryTangentEagerDecision) == (
        "control_provenance_invalid",
        "legacy_boundary_tangent_adjudication_invalid",
        "eager_schedule_integration_invalid",
        "boundary_tangent_representation_invalid",
        "boundary_tangent_design_infeasible",
        "fresh_exact_cache_invalid",
        "boundary_tangent_cache_resource_infeasible",
        "boundary_tangent_baseline_invalid",
        "boundary_tangent_optimization_pipeline_invalid",
        "boundary_tangent_baseline_only_signal",
        "selection_false_discovery",
        "boundary_tangent_time_local_signal_not_detected",
        "boundary_tangent_audit_inconclusive",
        "paired_risk_inference_invalid",
        "exact_rb_boundary_tangent_time_local_signal_confirmed",
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("provenance_valid", "control_provenance_invalid"),
        (
            "legacy_boundary_tangent_adjudication_valid",
            "legacy_boundary_tangent_adjudication_invalid",
        ),
        (
            "eager_schedule_integration_valid",
            "eager_schedule_integration_invalid",
        ),
        (
            "boundary_tangent_representation_valid",
            "boundary_tangent_representation_invalid",
        ),
        ("path_plan_valid", "boundary_tangent_design_infeasible"),
    ],
)
def test_preflight_failures_have_distinct_closed_decisions(
    field: str, expected: str
) -> None:
    metrics = _preflight_metrics()
    metrics[field] = 0
    preflight = evaluate_preflight_gate(metrics)
    decision = _decision(
        preflight,
        not_evaluated_gate("cache", "blocked"),
        not_evaluated_gate("train", "blocked"),
        not_evaluated_gate("confirm", "blocked"),
    )
    assert preflight["passed"] == 0
    assert decision["decision"] == expected


def test_preflight_resource_boundary_is_scientifically_complete() -> None:
    t = BoundaryTangentEagerThresholds()
    metrics = _preflight_metrics()
    metrics["projected_elapsed_seconds"] = nextafter(
        t.maximum_projected_seconds, float("inf")
    )
    gate = evaluate_preflight_gate(metrics)
    assert gate["failure_domain"] == "resource_gate"
    assert gate["resource_only_failure"] == 1
    assert gate["scientific_evidence_complete"] == 1
    assert gate["resource_valid"] == 0


def test_cache_gate_checks_exact_counts_numerics_and_resources() -> None:
    passing = evaluate_cache_gate(_cache_metrics())
    assert passing["passed"] == 1
    assert passing["production_cache_generation_performed"] == 1

    numerical_metrics = _cache_metrics()
    numerical_metrics["certificate_fraction"] = nextafter(1.0, 0.0)
    numerical = evaluate_cache_gate(numerical_metrics)
    assert numerical["failure_domain"] == "numerical"
    assert numerical["numerically_valid"] == 0

    resource_metrics = _cache_metrics()
    resource_metrics["minimum_role_rate"] = nextafter(1_300.0, 0.0)
    resource = evaluate_cache_gate(resource_metrics)
    assert resource["failure_domain"] == "resource_gate"
    assert resource["resource_only_failure"] == 1
    assert resource["scientific_evidence_complete"] == 1
    preflight = evaluate_preflight_gate(_preflight_metrics())
    decision = _decision(
        preflight,
        resource,
        not_evaluated_gate("train", "blocked"),
        not_evaluated_gate("confirm", "blocked"),
    )
    assert decision["decision"] == (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_CACHE_RESOURCE_INFEASIBLE.value
    )


def test_cache_cumulative_projection_exact_boundary_and_nextafter() -> None:
    exact_metrics = _cache_metrics()
    exact_metrics["cache_elapsed_seconds"] = 60_000.0
    exact_metrics[
        "frozen_conservative_confirmation_projection_seconds"
    ] = 48_000.0
    exact = evaluate_cache_gate(exact_metrics)
    assert exact["passed"] == 1
    assert exact["projected_cache_plus_confirmation_seconds"] == 108_000.0
    assert exact["checks"]["projected_cache_plus_confirmation_seconds"][
        "passed"
    ] == 1

    over_metrics = dict(exact_metrics)
    over_metrics["cache_elapsed_seconds"] = nextafter(
        60_000.0, float("inf")
    )
    over_metrics[
        "frozen_conservative_confirmation_projection_seconds"
    ] = nextafter(48_000.0, float("inf"))
    over = evaluate_cache_gate(over_metrics)
    assert over["projected_cache_plus_confirmation_seconds"] > 108_000.0
    assert over["failure_domain"] == "resource_gate"
    assert over["resource_only_failure"] == 1
    assert over["scientific_evidence_complete"] == 1


def test_train_gate_distinguishes_baseline_pipeline_and_baseline_only() -> None:
    preflight, cache, _, confirm = _passing_gates()

    baseline_metrics = _train_metrics()
    baseline_metrics["baseline_valid"] = 0
    baseline = evaluate_train_gate(baseline_metrics)
    assert _decision(preflight, cache, baseline, confirm)["decision"] == (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_BASELINE_INVALID.value
    )

    baseline_and_optimizer_metrics = _train_metrics()
    baseline_and_optimizer_metrics["baseline_valid"] = 0
    baseline_and_optimizer_metrics["synthetic_teacher_passed"] = 0
    baseline_and_optimizer = evaluate_train_gate(
        baseline_and_optimizer_metrics
    )
    assert baseline_and_optimizer["optimization_pipeline_valid"] == 0
    assert _decision(
        preflight, cache, baseline_and_optimizer, confirm
    )["decision"] == (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_BASELINE_INVALID.value
    )

    optimizer_metrics = _train_metrics()
    optimizer_metrics["synthetic_teacher_passed"] = 0
    optimizer = evaluate_train_gate(optimizer_metrics)
    assert _decision(preflight, cache, optimizer, confirm)["decision"] == (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID.value
    )

    baseline_only_metrics = _train_metrics()
    for name in (
        "selected_nonzero",
        "selected_checkpoint_eligible",
        "selected_beats_baseline_overall",
        "selected_beats_baseline_high_reverse_time",
    ):
        baseline_only_metrics[name] = 0
    baseline_only = evaluate_train_gate(baseline_only_metrics)
    assert baseline_only["optimization_pipeline_valid"] == 1
    assert baseline_only["boundary_tangent_baseline_only"] == 1
    assert _decision(preflight, cache, baseline_only, confirm)["decision"] == (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_BASELINE_ONLY_SIGNAL.value
    )


def test_confirmation_success_authorizes_planning_only() -> None:
    gates = _passing_gates()
    confirm = gates[-1]
    assert confirm["passed"] == 1
    assert confirm["paired_risk_inference_valid"] == 1
    assert confirm["controller_control_planning_authorized"] == 0
    decision = _decision(*gates)
    assert decision["decision"] == FINAL_DECISION
    assert decision["controller_control_planning_authorized"] == 1
    assert decision["controller_control_trajectory_authorized"] == 0
    assert decision["reconstruction_authorized"] == 0
    assert decision["sampling_authorized"] == 0


@pytest.mark.parametrize(
    ("family_name", "field", "expected"),
    [
        (
            COMBINED_VS_BASELINE_NAMES[0],
            "point_estimates",
            "selection_false_discovery",
        ),
        (
            COMBINED_VS_ZERO_NAMES[0],
            "point_estimates",
            "boundary_tangent_time_local_signal_not_detected",
        ),
        (
            COMBINED_VS_ZERO_NAMES[0],
            "lower_bounds",
            "boundary_tangent_audit_inconclusive",
        ),
    ],
)
def test_confirmation_scientific_failures_are_distinct(
    family_name: str, field: str, expected: str
) -> None:
    preflight, cache, train, _ = _passing_gates()
    record = _max_t_record()
    values = dict(record[field])  # type: ignore[arg-type]
    values[family_name] = -0.1 if field == "point_estimates" else 0.0
    record[field] = values
    if field == "point_estimates":
        lower = dict(record["lower_bounds"])  # type: ignore[arg-type]
        lower[family_name] = -0.2
        record["lower_bounds"] = lower
        record["passed"] = 0
    elif field == "lower_bounds":
        record["passed"] = 0
    confirm = evaluate_confirm_gate(record, _confirm_metrics())
    assert confirm["paired_risk_inference_valid"] == 1
    assert _decision(preflight, cache, train, confirm)["decision"] == expected


def test_malformed_confirmation_and_resource_only_failure_fail_closed() -> None:
    preflight, cache, train, _ = _passing_gates()
    malformed = _max_t_record()
    malformed["family_names"] = list(reversed(CONFIRMATION_FAMILY_NAMES))
    inference = evaluate_confirm_gate(malformed, _confirm_metrics())
    assert inference["failure_domain"] == "inference"
    assert inference["paired_risk_inference_valid"] == 0
    assert _decision(preflight, cache, train, inference)["decision"] == (
        BoundaryTangentEagerDecision.PAIRED_RISK_INFERENCE_INVALID.value
    )

    wrong_paths = _max_t_record()
    wrong_paths["path_ids"] = list(range(64))
    path_gate = evaluate_confirm_gate(wrong_paths, _confirm_metrics())
    assert path_gate["paired_risk_inference_valid"] == 0
    assert path_gate["checks"]["confirmation_path_ids"]["passed"] == 0

    metrics = _confirm_metrics()
    metrics["transitions_per_second"] = nextafter(1_300.0, 0.0)
    resource = evaluate_confirm_gate(_max_t_record(), metrics)
    assert resource["failure_domain"] == "resource_gate"
    assert resource["resource_only_failure"] == 1
    assert resource["scientific_evidence_complete"] == 1
    assert resource["resource_valid"] == 0


@pytest.mark.parametrize("runtime_value", [None, float("nan")])
def test_cache_resource_only_requires_present_finite_runtime(
    runtime_value: float | None,
) -> None:
    metrics = _cache_metrics()
    if runtime_value is None:
        metrics.pop("cache_elapsed_seconds")
    else:
        metrics["cache_elapsed_seconds"] = runtime_value

    gate = evaluate_cache_gate(metrics)

    assert gate["passed"] == 0
    assert gate["resource_only_failure"] == 0
    assert gate["scientific_evidence_complete"] == 0
    assert gate["failure_domain"] != "resource_gate"


@pytest.mark.parametrize("runtime_value", [None, float("nan")])
def test_confirmation_resource_only_requires_present_finite_runtime(
    runtime_value: float | None,
) -> None:
    metrics = _confirm_metrics()
    if runtime_value is None:
        metrics.pop("confirmation_elapsed_seconds")
    else:
        metrics["confirmation_elapsed_seconds"] = runtime_value

    gate = evaluate_confirm_gate(_max_t_record(), metrics)

    assert gate["passed"] == 0
    assert gate["resource_only_failure"] == 0
    assert gate["scientific_evidence_complete"] == 0
    assert gate["failure_domain"] != "resource_gate"


def test_confirmation_resource_failure_preserves_valid_inference_decision() -> None:
    preflight, cache, train, _ = _passing_gates()
    metrics = _confirm_metrics()
    metrics["transitions_per_second"] = nextafter(1_300.0, 0.0)
    confirm = evaluate_confirm_gate(_max_t_record(), metrics)

    assert confirm["failure_domain"] == "resource_gate"
    assert confirm["resource_only_failure"] == 1
    assert confirm["paired_risk_inference_valid"] == 1
    decision = _decision(preflight, cache, train, confirm)
    assert decision["decision"] == (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE.value
    )
    assert decision["controller_control_planning_authorized"] == 0


def test_confirmation_cumulative_actual_exact_boundary_and_nextafter() -> None:
    exact_metrics = _confirm_metrics()
    exact_metrics["cache_elapsed_seconds"] = 60_000.0
    exact_metrics["confirmation_elapsed_seconds"] = 48_000.0
    exact = evaluate_confirm_gate(_max_t_record(), exact_metrics)
    assert exact["passed"] == 1
    assert exact["actual_cache_plus_confirmation_seconds"] == 108_000.0
    assert exact["checks"]["actual_cache_plus_confirmation_seconds"][
        "passed"
    ] == 1

    over_metrics = dict(exact_metrics)
    over_metrics["cache_elapsed_seconds"] = nextafter(
        60_000.0, float("inf")
    )
    over_metrics["confirmation_elapsed_seconds"] = nextafter(
        48_000.0, float("inf")
    )
    over = evaluate_confirm_gate(_max_t_record(), over_metrics)
    assert over["actual_cache_plus_confirmation_seconds"] > 108_000.0
    assert over["failure_domain"] == "resource_gate"
    assert over["resource_only_failure"] == 1
    assert over["scientific_evidence_complete"] == 1


def test_required_gate_and_execution_failure_are_fail_closed() -> None:
    preflight, cache, train, confirm = _passing_gates()
    workflow = evaluate_required_gate(
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train,
        confirm_gate=confirm,
        require_gate="confirm",
    )
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == FINAL_DECISION
    assert workflow["production_cache_generation_performed"] == 1
    assert workflow["physical_training_performed"] == 1
    assert workflow["confirmation_performed"] == 1
    assert workflow["controller_control_planning_authorized"] == 1

    failed = evaluate_cache_gate(
        {
            "evaluation_status": "execution_failed",
            "failure_code": "worker_exit",
        }
    )
    assert failed["evaluation_status"] == "execution_failed"
    assert failed["failure_code"] == "worker_exit"
    assert failed["passed"] == 0

    preflight_failed = evaluate_preflight_gate(
        {
            "evaluation_status": "execution_failed",
            "failure_code": "preflight_read_error",
        }
    )
    execution_decision = _decision(
        preflight_failed,
        not_evaluated_gate("cache", "blocked"),
        not_evaluated_gate("train", "blocked"),
        not_evaluated_gate("confirm", "blocked"),
    )
    assert execution_decision["decision"] == (
        BoundaryTangentEagerDecision.EAGER_SCHEDULE_INTEGRATION_INVALID.value
    )

    with pytest.raises(BoundaryTangentEagerGateError, match="unknown required gate"):
        evaluate_required_gate(
            preflight_gate=preflight,
            cache_gate=cache,
            train_gate=train,
            confirm_gate=confirm,
            require_gate="controller",
        )


def test_train_execution_failure_is_not_a_baseline_failure() -> None:
    preflight, cache, _, _ = _passing_gates()
    train_failure = evaluate_train_gate(
        {
            "evaluation_status": "execution_failed",
            "failure_domain": "train_execution",
            "failure_code": "worker_exit",
            "production_cache_generation_performed": 1,
            "physical_training_performed": 1,
        }
    )
    train_decision = _decision(
        preflight,
        cache,
        train_failure,
        not_evaluated_gate("confirm", "training execution failed"),
    )
    assert train_decision["decision"] == (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID.value
    )
    assert train_decision["decision"] != (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_BASELINE_INVALID.value
    )


def test_confirmation_execution_failure_is_not_an_inference_failure() -> None:
    preflight, cache, train, _ = _passing_gates()
    confirm_failure = evaluate_confirm_gate(
        {},
        {
            "evaluation_status": "execution_failed",
            "failure_domain": "confirmation_execution",
            "failure_code": "worker_exit",
            "production_cache_generation_performed": 1,
            "physical_training_performed": 1,
            "confirmation_performed": 1,
        },
    )
    confirm_decision = _decision(preflight, cache, train, confirm_failure)
    assert confirm_decision["decision"] == (
        BoundaryTangentEagerDecision.BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE.value
    )
    assert confirm_decision["decision"] != (
        BoundaryTangentEagerDecision.PAIRED_RISK_INFERENCE_INVALID.value
    )
