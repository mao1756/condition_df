from __future__ import annotations

import copy

import pytest

from mnist.d0_jacobi_rb_haar_power_recovery_gate import (
    HaarPowerRecoveryDecision,
    decide_recovery_workflow,
    evaluate_antithetic_pilot,
    evaluate_nested_replay,
    evaluate_recovery_preflight,
    evaluate_recovery_workflow,
    execution_failed_gate,
    not_evaluated_gate,
)


def _preflight_metrics() -> dict[str, object]:
    values = {
        name: 1
        for name in (
            "control_provenance_pass",
            "parent_registry_verified_pass",
            "parent_sources_immutable_pass",
            "parent_scientific_config_pass",
            "parent_preflight_pass",
            "parent_coupling_pass",
            "parent_pilot_execution_failure_pass",
            "parent_failure_code_pass",
            "parent_shard_layout_pass",
            "parent_shard_hashes_pass",
            "parent_shard_chains_pass",
            "parent_schedule_location_pass",
            "parent_antithetic_absent_pass",
            "parent_panel_b_absent_pass",
            "parent_selection_absent_pass",
            "parent_no_work_pass",
            "transitive_provenance_pass",
            "path_plan_frozen_pass",
        )
    }
    values.update(
        root_seed=261181,
        parent_registry_record_count=197,
        parent_source_count=35,
        parent_main_shards=16,
        parent_reference_shards=64,
    )
    return values


def _replay_metrics() -> dict[str, object]:
    values = {
        name: 1
        for name in (
            "canonical_schedule_binding_pass",
            "parent_read_only_pass",
            "no_nested_gpu_recomputation_pass",
            "observable_replay_pass",
            "candidate_reconstruction_pass",
            "candidate_numerical_health_pass",
            "candidate_resource_health_pass",
            "frozen_no_nominee_pass",
            "raw_endpoint_authorizing_pass",
            "dynkin_advisory_only_pass",
            "shard_chain_pass",
            "state_updates_device_resident_pass",
            "antithetic_path_ids_untouched_pass",
        )
    }
    values.update(
        shard_count=80,
        transition_count=120_823_808,
        certificate_fraction=1.0,
        fallback_count=38,
        fallback_fraction=3.145075513594142e-7,
        fallback_cost_fraction=5.989668245920508e-4,
        mass_error=1.3322676295501878e-15,
        peak_memory_fraction=0.006818991116410677,
        conservative_rate=4202.429019551445,
        candidate_count=4,
        eligible_candidate_count=0,
        selection_status="panel_a_no_eligible_design",
        uncertified_count=0,
        forbidden_event_count=0,
    )
    return values


def _pilot_metrics() -> dict[str, object]:
    values = {
        name: 1
        for name in (
            "plans_frozen_pass",
            "profile_order_pass",
            "panel_nonregeneration_pass",
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
            "production_authorizing_pass",
            "antithetic_panel_a_nominated",
            "antithetic_panel_b_opened",
            "antithetic_panels_agree",
        )
    }
    values.update(
        selected_profile="pairwise_haar_antithetic",
        main_paths=16,
        reference_paths=16,
        combined_main_half_width=0.0025,
        combined_generator_reference_half_width=0.005,
        combined_reference_stability_half_width=0.005,
        projected_hours=48.0,
        minimum_rate=1300.0,
        certificate_fraction=1.0,
        fallback_fraction=1.0e-4,
        fallback_cost_fraction=0.10,
        peak_memory_fraction=0.80,
        forbidden_event_count=0,
    )
    return values


def test_all_three_recovery_gates_pass_at_frozen_boundaries() -> None:
    preflight = evaluate_recovery_preflight(_preflight_metrics())
    replay = evaluate_nested_replay(_replay_metrics())
    pilot = evaluate_antithetic_pilot(_pilot_metrics())
    assert preflight["passed"] == 1
    assert replay["passed"] == 1
    assert replay["antithetic_panel_a_authorized"] == 1
    assert pilot["passed"] == 1
    assert pilot["numerically_valid"] == 1
    assert pilot["resource_valid"] == 1
    assert pilot["power_valid"] == 1


@pytest.mark.parametrize(
    ("name", "bad"),
    (
        ("parent_main_shards", 15),
        ("parent_schedule_location_pass", 0),
        ("parent_sources_immutable_pass", 0),
    ),
)
def test_preflight_fails_closed(name: str, bad: object) -> None:
    metrics = _preflight_metrics()
    metrics[name] = bad
    assert evaluate_recovery_preflight(metrics)["passed"] == 0


@pytest.mark.parametrize(
    ("name", "bad"),
    (
        ("shard_count", 79),
        ("transition_count", 120_823_807),
        ("eligible_candidate_count", 1),
        ("selection_status", "panel_a_design_nominated"),
        ("no_nested_gpu_recomputation_pass", 0),
        ("fallback_count", 39),
    ),
)
def test_replay_fails_closed(name: str, bad: object) -> None:
    metrics = _replay_metrics()
    metrics[name] = bad
    assert evaluate_nested_replay(metrics)["passed"] == 0


@pytest.mark.parametrize(
    ("name", "bad"),
    (
        ("combined_main_half_width", 0.0025000001),
        ("combined_generator_reference_half_width", 0.005000001),
        ("projected_hours", 48.0001),
        ("minimum_rate", 1299.999),
        ("antithetic_panels_agree", 0),
    ),
)
def test_pilot_fails_closed(name: str, bad: object) -> None:
    metrics = _pilot_metrics()
    metrics[name] = bad
    assert evaluate_antithetic_pilot(metrics)["passed"] == 0


def test_pending_prefixes_do_not_claim_a_failure_decision() -> None:
    preflight = evaluate_recovery_preflight(_preflight_metrics())
    replay = evaluate_nested_replay(_replay_metrics())
    before_replay = decide_recovery_workflow(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight_gate=preflight,
        replay_gate=not_evaluated_gate("replay", "not run"),
        pilot_gate=not_evaluated_gate("pilot", "not run"),
    )
    before_pilot = decide_recovery_workflow(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight_gate=preflight,
        replay_gate=replay,
        pilot_gate=not_evaluated_gate("pilot", "not run"),
    )
    assert before_replay["evaluation_status"] == "pending"
    assert before_replay["decision"] is None
    assert before_replay["replay_authorized"] == 1
    assert before_pilot["evaluation_status"] == "pending"
    assert before_pilot["decision"] is None
    assert before_pilot["antithetic_panel_a_authorized"] == 1


def test_closed_decision_ladder() -> None:
    provenance = {"evaluation_status": "evaluated", "passed": 1}
    preflight = evaluate_recovery_preflight(_preflight_metrics())
    replay = evaluate_nested_replay(_replay_metrics())
    pilot = evaluate_antithetic_pilot(_pilot_metrics())

    success = decide_recovery_workflow(
        provenance=provenance,
        preflight_gate=preflight,
        replay_gate=replay,
        pilot_gate=pilot,
    )
    assert success["decision"] == (
        HaarPowerRecoveryDecision.EXACT_HAAR_HIERARCHICAL_REFINEMENT_COUPLING_FEASIBLE
    )
    assert success["production_refinement_patch_authorized"] == 1

    no_nominee = copy.deepcopy(pilot)
    no_nominee["passed"] = 0
    no_nominee["panel_a_nominated"] = 0
    assert (
        decide_recovery_workflow(
            provenance=provenance,
            preflight_gate=preflight,
            replay_gate=replay,
            pilot_gate=no_nominee,
        )["decision"]
        == HaarPowerRecoveryDecision.HIERARCHICAL_POWER_INFEASIBLE
    )

    disagreement = copy.deepcopy(pilot)
    disagreement["passed"] = 0
    disagreement["panels_agree"] = 0
    assert (
        decide_recovery_workflow(
            provenance=provenance,
            preflight_gate=preflight,
            replay_gate=replay,
            pilot_gate=disagreement,
        )["decision"]
        == HaarPowerRecoveryDecision.HIERARCHICAL_PANELS_DISAGREE
    )


def test_execution_failure_domains_map_to_closed_outcomes() -> None:
    provenance = {"evaluation_status": "evaluated", "passed": 1}
    failure = execution_failed_gate(
        "replay",
        failure_domain="nested_replay",
        failure_code="bad",
        error_type="RuntimeError",
        error="bad",
    )
    result = decide_recovery_workflow(
        provenance=provenance,
        preflight_gate=evaluate_recovery_preflight(_preflight_metrics()),
        replay_gate=failure,
        pilot_gate=not_evaluated_gate("pilot", "not run"),
    )
    assert result["decision"] == (
        HaarPowerRecoveryDecision.NESTED_PANEL_REPLAY_INVALID
    )


def test_workflow_applies_required_prefix_only() -> None:
    provenance = {"evaluation_status": "evaluated", "passed": 1}
    workflow = evaluate_recovery_workflow(
        provenance=provenance,
        preflight_gate=evaluate_recovery_preflight(_preflight_metrics()),
        replay_gate=evaluate_nested_replay(_replay_metrics()),
        pilot_gate=not_evaluated_gate("pilot", "not run"),
        require_gate="replay",
    )
    assert workflow["required_components"] == ["preflight", "replay"]
    assert workflow["required_gate_pass"] == 1
