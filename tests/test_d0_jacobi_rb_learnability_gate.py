from __future__ import annotations

import copy
import math

import pytest

from mnist.d0_jacobi_rb_learnability_gate import (
    JacobiRBLearnabilityDecision,
    JacobiRBLearnabilityThresholds,
    NO_CLAIM_AUTHORIZATION,
    decide_learnability_workflow,
    evaluate_learnability_cache,
    evaluate_learnability_confirmation,
    evaluate_learnability_physical,
    evaluate_learnability_preflight,
    evaluate_learnability_teacher,
    evaluate_learnability_workflow,
    not_evaluated_gate,
)


def _preflight_metrics() -> dict[str, object]:
    t = JacobiRBLearnabilityThresholds()
    one = {
        name: 1
        for name in (
            "parent_provenance_pass",
            "multipath_kernel_gate_pass",
            "multipath_target_gate_pass",
            "multipath_decision_pass",
            "strang_power_failure_preserved_pass",
            "haar_power_only_failure_pass",
            "haar_numerical_health_pass",
            "haar_resource_health_pass",
            "source_image_hash_pass",
            "source_image_npz_hash_pass",
            "mixed_target_hash_pass",
            "future_model_input_contract_pass",
            "parents_no_training_pass",
            "parents_no_reverse_sampling_pass",
            "parent_registries_pass",
            "source_binding_pass",
            "path_plan_frozen_pass",
            "path_plan_bounds_pass",
            "path_plan_disjoint_pass",
            "path_plan_collision_scan_pass",
            "capture_parity_pass",
            "capture_rng_neutral_pass",
            "capture_call_order_pass",
            "capture_hash_parity_pass",
            "model_input_schema_firewall_pass",
            "confirmation_absent_pass",
        )
    }
    return {
        **one,
        "outer_steps": t.outer_steps,
        "steps_per_shard": t.steps_per_shard,
        "paths_per_split": t.paths_per_split,
        "selected_outer_steps": list(t.selected_outer_steps),
        "effective_transitions_per_second": t.minimum_effective_transitions_per_second,
        "projected_total_hours": t.maximum_projected_total_hours,
        "peak_memory_fraction": t.maximum_peak_memory_fraction,
        "projected_persisted_cache_bytes": t.maximum_persisted_cache_bytes,
        "projected_transition_count": t.total_transition_count,
        "test_only_reduced_workload": 0,
    }


def _cache_metrics(split: str) -> dict[str, object]:
    t = JacobiRBLearnabilityThresholds()
    result: dict[str, object] = {
        **{
            name: 1
            for name in (
                "all_shards_complete_pass",
                "shard_chain_pass",
                "replay_hashes_pass",
                "capture_state_alignment_pass",
                "states_finite_pass",
                "targets_finite_pass",
                "sample_key_join_pass",
                "model_input_schema_firewall_pass",
                "input_label_schema_separation_pass",
                "selected_step_phase_coverage_pass",
                "state_updates_device_resident_pass",
            )
        },
        "split": split,
        "path_count": t.paths_per_split,
        "outer_steps": t.outer_steps,
        "steps_per_shard": t.steps_per_shard,
        "transition_count": t.transitions_per_split,
        "sample_count": t.samples_per_split,
        "selected_outer_steps": list(t.selected_outer_steps),
        "phases_per_selected_step": t.phases_per_step,
        "certificate_fraction": 1.0,
        "maximum_mass_error": t.maximum_mass_error,
        "persisted_cache_bytes": 1,
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
        result.update(
            selected_model_seal_pass=1,
            confirmation_opened_once_pass=1,
            confirmation_path_plan_unchanged_pass=1,
        )
    else:
        result["confirmation_absent_pass"] = 1
    return result


def _teacher_metrics() -> dict[str, object]:
    return {
        **{
            name: 1
            for name in (
                "training_complete_pass",
                "all_losses_finite_pass",
                "same_pipeline_pass",
                "selected_checkpoint_replay_hash_pass",
                "model_input_schema_firewall_pass",
                "training_only_scale_pass",
                "no_target_modification_pass",
            )
        },
        "validation_path_count": 8,
        "paths_beating_metadata_baseline": 8,
        "validation_teacher_mse": 0.009,
        "validation_metadata_baseline_mse": 1.0,
    }


def _physical_metrics() -> dict[str, object]:
    return {
        **{
            name: 1
            for name in (
                "training_complete_pass",
                "all_seeds_complete_pass",
                "all_losses_finite_pass",
                "validation_only_selection_pass",
                "selected_checkpoint_exists_pass",
                "selected_checkpoint_hash_pass",
                "selected_model_record_frozen_pass",
                "metadata_baseline_frozen_pass",
                "confirmation_gate_definition_frozen_pass",
                "confirmation_absent_pass",
                "model_input_schema_firewall_pass",
                "training_only_scale_pass",
                "unweighted_mse_objective_pass",
                "no_target_modification_pass",
            )
        },
        "model_seed_count": 3,
        "validation_path_count": 8,
        "target_scale": 0.25,
        "selected_update": 100,
    }


def _confirmation_metrics(improvements: list[float] | None = None) -> dict[str, object]:
    return {
        **{
            name: 1
            for name in (
                "predictions_finite_pass",
                "losses_finite_pass",
                "selected_model_hash_pass",
                "model_config_hash_pass",
                "metadata_baseline_hash_pass",
                "path_plan_hash_pass",
                "confirmation_opened_once_pass",
                "confirmation_paths_not_replaced_pass",
                "confirmation_paths_not_added_pass",
                "model_input_schema_firewall_pass",
            )
        },
        "confirmation_path_count": 8,
        "path_metadata_minus_model_mse": (
            improvements if improvements is not None else [0.1] * 8
        ),
        "aggregate_model_mse": 0.5,
        "aggregate_zero_mse": 1.0,
    }


def _passing_gates() -> dict[str, dict[str, object]]:
    confirmation_cache = evaluate_learnability_cache(
        _cache_metrics("confirmation"), split="confirmation"
    )
    return {
        "preflight": evaluate_learnability_preflight(_preflight_metrics()),
        "train_cache": evaluate_learnability_cache(
            _cache_metrics("train"), split="train"
        ),
        "validation_cache": evaluate_learnability_cache(
            _cache_metrics("validation"), split="validation"
        ),
        "teacher": evaluate_learnability_teacher(_teacher_metrics()),
        "physical": evaluate_learnability_physical(_physical_metrics()),
        "confirmation_cache": confirmation_cache,
        "confirmation": evaluate_learnability_confirmation(
            _confirmation_metrics(), confirmation_cache_gate=confirmation_cache
        ),
    }


def test_frozen_cardinalities_and_selected_steps() -> None:
    t = JacobiRBLearnabilityThresholds()
    assert t.selected_outer_steps == tuple(range(15, 512, 16))
    assert len(t.selected_outer_steps) == 32
    assert t.samples_per_split == 8 * 32 * 7
    assert t.transitions_per_split == 8 * 512 * 7 * 392
    assert t.total_transition_count == 3 * t.transitions_per_split


def test_preflight_passes_at_resource_boundaries_and_fails_test_workload() -> None:
    gate = evaluate_learnability_preflight(_preflight_metrics())
    assert gate["passed"] == 1
    assert gate["cache_generation_authorized"] == 1

    bad = _preflight_metrics()
    bad["test_only_reduced_workload"] = 1
    assert evaluate_learnability_preflight(bad)["passed"] == 0


@pytest.mark.parametrize("split", ["train", "validation", "confirmation"])
def test_cache_gate_passes_exact_split(split: str) -> None:
    gate = evaluate_learnability_cache(_cache_metrics(split), split=split)
    assert gate["passed"] == 1
    assert gate["numerically_valid"] == 1


def test_cache_gate_rejects_target_modification_and_wrong_selected_step() -> None:
    metrics = _cache_metrics("train")
    metrics["target_modification_count"] = 1
    assert evaluate_learnability_cache(metrics, split="train")["passed"] == 0
    metrics = _cache_metrics("train")
    metrics["selected_outer_steps"] = list(range(32))
    assert evaluate_learnability_cache(metrics, split="train")["passed"] == 0


def test_cache_mass_tolerance_matches_exact_scheduler_contract() -> None:
    thresholds = JacobiRBLearnabilityThresholds()
    assert thresholds.maximum_mass_error == 2.0e-12
    metrics = _cache_metrics("train")
    metrics["maximum_mass_error"] = math.nextafter(
        thresholds.maximum_mass_error, math.inf
    )
    assert evaluate_learnability_cache(
        metrics, split="train", thresholds=thresholds
    )["passed"] == 0


def test_teacher_boundary_and_all_path_requirement() -> None:
    gate = evaluate_learnability_teacher(_teacher_metrics())
    assert gate["passed"] == 1
    assert gate["physical_training_authorized"] == 1

    metrics = _teacher_metrics()
    metrics["paths_beating_metadata_baseline"] = 7
    assert evaluate_learnability_teacher(metrics)["passed"] == 0


def test_physical_gate_does_not_require_validation_signal() -> None:
    gate = evaluate_learnability_physical(_physical_metrics())
    assert gate["passed"] == 1
    assert gate["confirmation_generation_authorized"] == 1
    assert gate["physical_training_performed"] == 1

    failed_metrics = _physical_metrics()
    failed_metrics["all_losses_finite_pass"] = 0
    failed = evaluate_learnability_physical(failed_metrics)
    assert failed["passed"] == 0
    assert failed["physical_training_performed"] == 1


def test_eight_positive_path_signs_pass_and_tie_fails() -> None:
    cache = evaluate_learnability_cache(
        _cache_metrics("confirmation"), split="confirmation"
    )
    passed = evaluate_learnability_confirmation(
        _confirmation_metrics(), confirmation_cache_gate=cache
    )
    assert passed["passed"] == 1
    assert passed["path_sign_count"] == 8
    assert passed["one_sided_sign_test_p_value"] == 1 / 256

    tied = evaluate_learnability_confirmation(
        _confirmation_metrics([0.1] * 7 + [0.0]),
        confirmation_cache_gate=cache,
    )
    assert tied["passed"] == 0
    assert tied["one_sided_sign_test_p_value"] is None


def test_confirmation_cache_failure_dominates_signal() -> None:
    cache = evaluate_learnability_cache(
        _cache_metrics("confirmation"), split="confirmation"
    )
    cache["passed"] = 0
    gate = evaluate_learnability_confirmation(
        _confirmation_metrics(), confirmation_cache_gate=cache
    )
    assert gate["passed"] == 0
    gates = _passing_gates()
    decision = decide_learnability_workflow(
        provenance=True,
        preflight_gate=gates["preflight"],
        train_cache_gate=gates["train_cache"],
        validation_cache_gate=gates["validation_cache"],
        teacher_gate=gates["teacher"],
        physical_gate=gates["physical"],
        confirmation_cache_gate=cache,
        confirmation_gate=gate,
    )
    assert decision["decision"] == JacobiRBLearnabilityDecision.EXACT_CACHE_INVALID


def test_teacher_failure_prevents_success() -> None:
    gates = _passing_gates()
    teacher = copy.deepcopy(gates["teacher"])
    teacher["passed"] = 0
    decision = decide_learnability_workflow(
        provenance=True,
        preflight_gate=gates["preflight"],
        train_cache_gate=gates["train_cache"],
        validation_cache_gate=gates["validation_cache"],
        teacher_gate=teacher,
        physical_gate=gates["physical"],
        confirmation_cache_gate=gates["confirmation_cache"],
        confirmation_gate=gates["confirmation"],
    )
    assert (
        decision["decision"]
        == JacobiRBLearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID
    )


def test_success_authorizes_planning_only() -> None:
    gates = _passing_gates()
    workflow = evaluate_learnability_workflow(
        provenance=True,
        preflight_gate=gates["preflight"],
        train_cache_gate=gates["train_cache"],
        validation_cache_gate=gates["validation_cache"],
        teacher_gate=gates["teacher"],
        physical_gate=gates["physical"],
        confirmation_cache_gate=gates["confirmation_cache"],
        confirmation_gate=gates["confirmation"],
        require_gate="confirm",
    )
    decision = workflow["decision"]
    assert workflow["required_gate_pass"] == 1
    assert (
        decision["decision"]
        == JacobiRBLearnabilityDecision.EXACT_K512_SPLIT_CHAIN_RB_LABEL_LEARNABLE
    )
    assert decision["larger_exact_discrete_chain_training_planning_authorized"] == 1
    for name, expected in NO_CLAIM_AUTHORIZATION.items():
        assert decision[name] == expected == 0


def test_no_signal_is_closed_failure() -> None:
    gates = _passing_gates()
    confirmation = copy.deepcopy(gates["confirmation"])
    confirmation["passed"] = 0
    confirmation["subchecks"]["all_path_improvements_strictly_positive"][
        "passed"
    ] = 0
    decision = decide_learnability_workflow(
        provenance=True,
        preflight_gate=gates["preflight"],
        train_cache_gate=gates["train_cache"],
        validation_cache_gate=gates["validation_cache"],
        teacher_gate=gates["teacher"],
        physical_gate=gates["physical"],
        confirmation_cache_gate=gates["confirmation_cache"],
        confirmation_gate=confirmation,
    )
    assert (
        decision["decision"]
        == JacobiRBLearnabilityDecision.NO_DETECTABLE_ONE_IMAGE_CONDITIONAL_SIGNAL
    )
    assert decision["larger_exact_discrete_chain_training_planning_authorized"] == 0


def test_not_evaluated_confirmation_cannot_authorize() -> None:
    gates = _passing_gates()
    decision = decide_learnability_workflow(
        provenance=True,
        preflight_gate=gates["preflight"],
        train_cache_gate=gates["train_cache"],
        validation_cache_gate=gates["validation_cache"],
        teacher_gate=gates["teacher"],
        physical_gate=gates["physical"],
        confirmation_cache_gate=gates["confirmation_cache"],
        confirmation_gate=not_evaluated_gate("confirmation", "sealed"),
    )
    assert decision["evaluation_status"] == "pending"
    assert decision["decision"] is None
    assert decision["larger_exact_discrete_chain_training_planning_authorized"] == 0


def test_unknown_required_gate_rejected() -> None:
    with pytest.raises(ValueError, match="unknown required gate"):
        evaluate_learnability_workflow(provenance=True, require_gate="sampling")
