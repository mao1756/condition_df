from __future__ import annotations

import copy

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint
from mnist.d0_jacobi_rb_coarse_residual_gate import (
    BASELINE_AVERAGED_TABLE_NOISE,
    BASELINE_ENERGY,
    BASELINE_PANEL_MEAN_NOISE,
    BASELINE_RAW_VALUES_SERIALIZATION_SHA256,
    BASELINE_RAW_VALUES_SHA256,
    BASELINE_SCHEMA,
    BASELINE_SHAPE,
    BASELINE_SHRINKAGE,
    BASELINE_SIGNAL_ENERGY,
    BASELINE_VALUES_SERIALIZATION_SHA256,
    BASELINE_VALUES_SHA256,
    WITNESS_REGISTRY_SHA256,
    CoarseResidualDecision,
    CoarseResidualThresholds,
    decide_coarse_residual_workflow,
    evaluate_coarse_residual_cache,
    evaluate_coarse_residual_cache_set,
    evaluate_coarse_residual_confirmation,
    evaluate_coarse_residual_preflight,
    evaluate_coarse_residual_train,
    evaluate_coarse_residual_workflow,
    execution_failed_gate,
    not_evaluated_gate,
)


def _baseline() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": BASELINE_SCHEMA,
        "schema_version": 1,
        "shape": list(BASELINE_SHAPE),
        "dtype": "<f8",
        "shrinkage": BASELINE_SHRINKAGE,
        "signal_energy": BASELINE_SIGNAL_ENERGY,
        "panel_mean_noise": BASELINE_PANEL_MEAN_NOISE,
        "averaged_table_noise": BASELINE_AVERAGED_TABLE_NOISE,
        "baseline_energy": BASELINE_ENERGY,
        "raw_values_sha256": BASELINE_RAW_VALUES_SHA256,
        "values_sha256": BASELINE_VALUES_SHA256,
        "raw_values_serialization_sha256": (
            BASELINE_RAW_VALUES_SERIALIZATION_SHA256
        ),
        "values_serialization_sha256": BASELINE_VALUES_SERIALIZATION_SHA256,
        "left_path_ids": list(range(0xE5000, 0xE5040)),
        "right_path_ids": list(range(0xE5100, 0xE5140)),
        "left_cell_means_file_sha256": (
            "70d374526df5c02e5c6ab7f9b17205de373b22c694480bb27bf5684b4a579852"
        ),
        "right_cell_means_file_sha256": (
            "d64688f026cc510d586fb6b20e2303fdbe407a99b1a161b4654dc5dd04face81"
        ),
        "left_cell_means_array_sha256": (
            "1fe04953fd50ea3cb0ac163efed216ec5ebbafc58f48ce0de3f77d090c29fe08"
        ),
        "right_cell_means_array_sha256": (
            "2d949662c098783aa663672528f107a9f73f503529440aca4313cf770cad737e"
        ),
        "witness_registry_sha256": WITNESS_REGISTRY_SHA256,
        "fit_role": "historical_witness_panels_training_only",
        "target_modified": 0,
    }
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _path_plan() -> dict[str, object]:
    t = CoarseResidualThresholds()
    return {
        "path_ids": {
            "train": list(range(0xD0000, 0xD0040)),
            "validation": list(range(0xD1000, 0xD1020)),
            "confirmation": list(range(0xD2000, 0xD2040)),
        },
        "path_plan_frozen_pass": 1,
        "parent_path_collision_count": 0,
        "model_input_firewall_pass": 1,
        "earlier_state_forbidden_pass": 1,
        "certificate_input_forbidden_pass": 1,
        "confirmation_sealed_pass": 1,
        "selected_outer_steps": list(t.selected_outer_steps),
        "projected_transition_count": t.total_transitions,
        "projected_total_hours": 22.5,
        "projected_cache_bytes": 1_000_000_000,
        "test_only_reduced_workload": 0,
    }


def _cache(split: str) -> dict[str, object]:
    t = CoarseResidualThresholds()
    exact = (
        "all_shards_complete_pass",
        "cache_complete_pass",
        "cache_replay_hash_pass",
        "states_finite_pass",
        "targets_finite_pass",
        "capture_state_alignment_pass",
        "sample_key_join_pass",
        "sample_key_unique_pass",
        "selected_step_phase_coverage_pass",
        "split_role_isolation_pass",
        "path_plan_binding_pass",
        "baseline_hash_binding_pass",
        "model_input_firewall_pass",
        "exact_jacobi_transition_pass",
        "exact_rb_target_pass",
        "unmodified_binary64_target_pass",
        "state_updates_device_resident_pass",
    )
    record: dict[str, object] = {
        **{name: 1 for name in exact},
        "split": split,
        "path_count": {
            "train": t.train_paths,
            "validation": t.validation_paths,
            "confirmation": t.confirmation_paths,
        }[split],
        "sample_count": {
            "train": t.train_samples,
            "validation": t.validation_samples,
            "confirmation": t.confirmation_samples,
        }[split],
        "transition_count": {
            "train": t.train_transitions,
            "validation": t.validation_transitions,
            "confirmation": t.confirmation_transitions,
        }[split],
        "selected_outer_steps": list(t.selected_outer_steps),
        "certificate_fraction": 1.0,
        "maximum_mass_error": 1.0e-15,
        "transitions_per_second": 2_500.0,
        "peak_memory_fraction": 0.1,
        "persisted_cache_bytes": 400_000_000,
        "residual_target_persisted": 0,
        **{
            name: 0
            for name in (
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
                "target_modification_count",
            )
        },
    }
    if split == "confirmation":
        record.update(
            confirmation_seal_pass=1,
            confirmation_opened_once_pass=1,
            confirmation_plan_unchanged_pass=1,
        )
    else:
        record["confirmation_absent_pass"] = 1
    return record


def _tasks() -> list[dict[str, object]]:
    t = CoarseResidualThresholds()
    return [
        {
            "role": "synthetic_teacher",
            "seed": t.model_seeds[0],
            "complete": 1,
            "finite": 1,
            "relative_validation_mse": 0.005,
            "path_baseline_minus_model_mse": [0.1] * t.validation_paths,
        },
        {
            "role": "exact_baseline_null",
            "seed": t.model_seeds[0],
            "complete": 1,
            "finite": 1,
            "selected_update": 0,
        },
        *[
            {
                "role": f"physical_seed_{seed}",
                "seed": seed,
                "complete": 1,
                "finite": 1,
            }
            for seed in t.model_seeds
        ],
    ]


def _selection(update: int = 100) -> dict[str, object]:
    t = CoarseResidualThresholds()
    exact = (
        "selection_validation_only_pass",
        "analytic_zero_candidate_pass",
        "coarse_baseline_candidate_pass",
        "selected_checkpoint_frozen_pass",
        "baseline_hash_binding_pass",
        "unweighted_mse_against_exact_label_pass",
        "target_unmodified_pass",
        "target_scale_training_only_pass",
        "combined_prediction_loss_pass",
        "model_input_firewall_pass",
        "confirmation_gate_definition_frozen_pass",
        "confirmation_absent_pass",
    )
    return {
        **{name: 1 for name in exact},
        "selected_update": update,
        "maximum_updates": t.maximum_updates,
        "validation_interval": t.validation_interval,
        "batch_size": t.batch_size,
        "combined_validation_mse": 0.99 if update else 1.0,
        "baseline_validation_mse": 1.0,
        "combined_validation_mse_data_end": 0.98 if update else 1.0,
        "baseline_validation_mse_data_end": 1.0,
    }


def _confirmation() -> dict[str, object]:
    t = CoarseResidualThresholds()
    exact = (
        "confirmation_cache_gate_pass",
        "confirmation_sealed_pass",
        "confirmation_opened_once_pass",
        "confirmation_paths_fresh_pass",
        "confirmation_paths_disjoint_pass",
        "selected_checkpoint_hash_pass",
        "baseline_hash_binding_pass",
        "path_plan_hash_pass",
        "predictions_finite_pass",
        "risks_finite_pass",
        "unweighted_exact_label_risk_pass",
        "model_input_firewall_pass",
        "no_post_selection_refit_pass",
    )
    family = (
        "overall.baseline_vs_zero",
        "overall.combined_vs_baseline",
    )
    return {
        **{name: 1 for name in exact},
        "confirmation_path_count": t.confirmation_paths,
        "direct_derived_delta_t_max_abs_error": 1.0e-15,
        "max_t": {
            "method": "centered_whole_path_studentized_max_t",
            "bootstrap_unit": "whole_path_jointly_across_family",
            "quantile_method": "higher",
            "family_names": list(family),
            "point_estimates": {name: 0.01 for name in family},
            "standard_errors": {name: 0.001 for name in family},
            "lower_bounds": {name: 0.005 for name in family},
            "critical_value": 2.5,
            "path_count": t.confirmation_paths,
            "confidence": t.confidence,
            "replicates": t.bootstrap_replicates,
            "seed": t.bootstrap_seed,
            "namespace": 0,
            "negative_values_truncated": 0,
            "passed": 1,
        },
    }


def _passing_components() -> tuple[dict, dict, dict, dict]:
    preflight = evaluate_coarse_residual_preflight(
        provenance_valid=True,
        baseline_record=_baseline(),
        path_plan=_path_plan(),
    )
    records = {
        "train": _cache("train"),
        "validation": _cache("validation"),
        "aggregate": {
            "persisted_cache_bytes": 800_000_000,
            "confirmation_absent_pass": 1,
            "split_path_sets_disjoint_pass": 1,
        },
    }
    cache = evaluate_coarse_residual_cache_set(split_records=records)
    train = evaluate_coarse_residual_train(
        task_records=_tasks(), selection=_selection()
    )
    confirm = evaluate_coarse_residual_confirmation(
        confirmation=_confirmation()
    )
    return preflight, cache, train, confirm


def test_frozen_cardinalities_and_literal_baseline() -> None:
    t = CoarseResidualThresholds()
    assert t.train_samples == 64 * 32 * 7
    assert t.validation_samples == 32 * 32 * 7
    assert t.confirmation_samples == 64 * 32 * 7
    assert t.total_transitions == 160 * 512 * 7 * 392
    gate = evaluate_coarse_residual_preflight(
        provenance_valid=True,
        baseline_record=_baseline(),
        path_plan=_path_plan(),
    )
    assert gate["passed"] == 1
    assert gate["baseline_derivation_valid"] == 1
    assert gate["design_feasible"] == 1


def test_preflight_distinguishes_provenance_baseline_and_design() -> None:
    good = evaluate_coarse_residual_preflight(
        provenance_valid=False,
        baseline_record=_baseline(),
        path_plan=_path_plan(),
    )
    decision = decide_coarse_residual_workflow(good, None, None, None)
    assert decision["decision"] == CoarseResidualDecision.CONTROL_PROVENANCE_INVALID

    baseline = _baseline()
    baseline["shrinkage"] = BASELINE_SHRINKAGE + 1.0e-12
    bad = evaluate_coarse_residual_preflight(
        provenance_valid=True, baseline_record=baseline, path_plan=_path_plan()
    )
    assert bad["baseline_derivation_valid"] == 0
    assert (
        decide_coarse_residual_workflow(bad, None, None, None)["decision"]
        == CoarseResidualDecision.COARSE_BASELINE_DERIVATION_INVALID
    )

    plan = _path_plan()
    plan["projected_total_hours"] = 30.0001
    bad = evaluate_coarse_residual_preflight(
        provenance_valid=True, baseline_record=_baseline(), path_plan=plan
    )
    assert (
        decide_coarse_residual_workflow(bad, None, None, None)["decision"]
        == CoarseResidualDecision.COARSE_RESIDUAL_DESIGN_INFEASIBLE
    )

    provenance_failure = execution_failed_gate(
        "preflight",
        RuntimeError("fixture"),
        failure_domain="provenance",
        failure_code="fixture",
    )
    assert (
        decide_coarse_residual_workflow(
            provenance_failure, None, None, None
        )["decision"]
        == CoarseResidualDecision.CONTROL_PROVENANCE_INVALID
    )
    execution_failure = execution_failed_gate(
        "preflight",
        RuntimeError("fixture"),
        failure_domain="preflight_benchmark",
        failure_code="fixture",
    )
    assert (
        decide_coarse_residual_workflow(
            execution_failure, None, None, None
        )["decision"]
        == CoarseResidualDecision.COARSE_RESIDUAL_DESIGN_INFEASIBLE
    )


def test_cache_gate_requires_exact_counts_and_zero_forbidden_events() -> None:
    assert evaluate_coarse_residual_cache(_cache("train"), split="train")[
        "passed"
    ] == 1
    bad = _cache("train")
    bad["limiter_count"] = 1
    assert evaluate_coarse_residual_cache(bad, split="train")["passed"] == 0
    with pytest.raises(ValueError, match="unknown cache split"):
        evaluate_coarse_residual_cache({}, split="audit")

    records = {
        "train": _cache("train"),
        "validation": _cache("validation"),
        "aggregate": {
            "persisted_cache_bytes": 800_000_000,
            "confirmation_absent_pass": 1,
            "split_path_sets_disjoint_pass": 1,
        },
    }
    assert evaluate_coarse_residual_cache_set(split_records=records)["passed"] == 1
    records["aggregate"]["persisted_cache_bytes"] = 1_342_177_281
    assert evaluate_coarse_residual_cache_set(split_records=records)["passed"] == 0


def test_train_controls_and_nonzero_eligibility() -> None:
    gate = evaluate_coarse_residual_train(
        task_records=_tasks(), selection=_selection()
    )
    assert gate["passed"] == 1
    assert gate["residual_candidate_eligible"] == 1
    assert gate["confirmation_generation_authorized"] == 1

    bad_tasks = _tasks()
    bad_tasks[0]["relative_validation_mse"] = 0.0100001
    assert (
        evaluate_coarse_residual_train(
            task_records=bad_tasks, selection=_selection()
        )["passed"]
        == 0
    )
    bad_tasks = _tasks()
    bad_tasks[1]["selected_update"] = 100
    assert (
        evaluate_coarse_residual_train(
            task_records=bad_tasks, selection=_selection()
        )["passed"]
        == 0
    )

    selection = _selection()
    selection["combined_validation_mse_data_end"] = 1.0
    assert (
        evaluate_coarse_residual_train(
            task_records=_tasks(), selection=selection
        )["passed"]
        == 0
    )


def test_update_zero_is_closed_coarse_baseline_only_outcome() -> None:
    preflight, cache, _, _ = _passing_components()
    train = evaluate_coarse_residual_train(
        task_records=_tasks(), selection=_selection(update=0)
    )
    assert train["passed"] == 0
    assert train["optimization_pipeline_valid"] == 1
    assert train["confirmation_generation_authorized"] == 0
    decision = decide_coarse_residual_workflow(preflight, cache, train, None)
    assert decision["decision"] == CoarseResidualDecision.COARSE_BASELINE_ONLY_SIGNAL


def test_confirmation_requires_two_overall_simultaneous_lower_bounds() -> None:
    gate = evaluate_coarse_residual_confirmation(confirmation=_confirmation())
    assert gate["passed"] == 1
    assert gate["coarse_baseline_replicated"] == 1
    assert gate["residual_replicated"] == 1
    assert gate["reverse_controller_planning_authorized"] == 1

    extra = _confirmation()
    extra["max_t"]["family_names"].append("data_end.combined_vs_baseline")
    extra["max_t"]["point_estimates"]["data_end.combined_vs_baseline"] = 0.1
    extra["max_t"]["standard_errors"]["data_end.combined_vs_baseline"] = 0.01
    extra["max_t"]["lower_bounds"]["data_end.combined_vs_baseline"] = 0.05
    assert (
        evaluate_coarse_residual_confirmation(confirmation=extra)[
            "paired_risk_inference_valid"
        ]
        == 0
    )


def test_confirmation_decisions_are_closed_and_distinct() -> None:
    preflight, cache, train, _ = _passing_components()

    confirmation_cache_failure = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "confirmation_cache_valid": 0,
    }
    assert (
        decide_coarse_residual_workflow(
            preflight, cache, train, confirmation_cache_failure
        )["decision"]
        == CoarseResidualDecision.FRESH_EXACT_CACHE_INVALID
    )

    bad = _confirmation()
    bad["direct_derived_delta_t_max_abs_error"] = 2.0e-12
    gate = evaluate_coarse_residual_confirmation(confirmation=bad)
    assert (
        decide_coarse_residual_workflow(preflight, cache, train, gate)["decision"]
        == CoarseResidualDecision.PAIRED_RISK_INFERENCE_INVALID
    )

    bad = _confirmation()
    bad["max_t"]["lower_bounds"]["overall.baseline_vs_zero"] = -1.0e-4
    bad["max_t"]["passed"] = 0
    gate = evaluate_coarse_residual_confirmation(confirmation=bad)
    assert (
        decide_coarse_residual_workflow(preflight, cache, train, gate)["decision"]
        == CoarseResidualDecision.COARSE_BASELINE_NONREPLICATING
    )

    bad = _confirmation()
    bad["max_t"]["point_estimates"]["overall.combined_vs_baseline"] = -1.0e-4
    bad["max_t"]["lower_bounds"]["overall.combined_vs_baseline"] = -2.0e-4
    bad["max_t"]["passed"] = 0
    gate = evaluate_coarse_residual_confirmation(confirmation=bad)
    assert (
        decide_coarse_residual_workflow(preflight, cache, train, gate)["decision"]
        == CoarseResidualDecision.SELECTION_FALSE_DISCOVERY
    )

    bad = _confirmation()
    bad["max_t"]["lower_bounds"]["overall.combined_vs_baseline"] = -1.0e-5
    bad["max_t"]["passed"] = 0
    gate = evaluate_coarse_residual_confirmation(confirmation=bad)
    assert (
        decide_coarse_residual_workflow(preflight, cache, train, gate)["decision"]
        == CoarseResidualDecision.COARSE_RESIDUAL_AUDIT_INCONCLUSIVE
    )


def test_success_authorizes_planning_but_never_sampling() -> None:
    preflight, cache, train, confirm = _passing_components()
    workflow = evaluate_coarse_residual_workflow(
        preflight_gate=preflight,
        cache_gate=cache,
        train_gate=train,
        confirm_gate=confirm,
        require_gate="confirm",
    )
    assert workflow["passed"] == 1
    assert (
        workflow["decision"]["decision"]
        == CoarseResidualDecision.EXACT_RB_COARSE_RESIDUAL_LEARNABLE
    )
    assert workflow["reverse_controller_planning_authorized"] == 1
    assert workflow["reverse_sampling_authorized"] == 0
    assert workflow["sampling_authorized"] == 0
    assert workflow["reconstruction_claim_authorized"] == 0


def test_execution_failure_and_not_evaluated_records_fail_closed() -> None:
    failure = execution_failed_gate(
        "cache",
        RuntimeError("fixture"),
        failure_domain="exact_cache",
        failure_code="fixture",
    )
    assert failure["evaluation_status"] == "execution_failed"
    assert failure["scientific_evidence_complete"] == 0
    assert failure["passed"] == 0
    pending = not_evaluated_gate("confirm", "sealed")
    assert pending["evaluation_status"] == "not_evaluated"
    assert pending["reverse_sampling_authorized"] == 0
