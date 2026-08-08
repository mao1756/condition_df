from __future__ import annotations

from math import nextafter

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_false_discovery_gate import (
    BASELINE_REPLAY_FLAGS,
    CANDIDATE_REPLAY_FLAGS,
    PREFLIGHT_FLAGS,
    CandidateAuditClassification,
    FalseDiscoveryDecision,
    FalseDiscoveryGateError,
    FalseDiscoveryThresholds,
    SealedBaselineClassification,
    decide_workflow,
    evaluate_adjudication_gate,
    evaluate_baseline_gate,
    evaluate_candidate_gate,
    evaluate_decision_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    not_evaluated_gate,
)


def _preflight_metrics() -> dict[str, object]:
    t = FalseDiscoveryThresholds()
    result: dict[str, object] = {name: 1 for name in PREFLIGHT_FLAGS}
    result.update(
        {
            "parent_terminal_decision": t.parent_terminal_decision,
            "parent_source_fingerprint": t.parent_source_fingerprint,
            "parent_scientific_config_sha256": (
                t.parent_scientific_config_sha256
            ),
            "parent_registry_semantic_sha256": (
                t.parent_registry_semantic_sha256
            ),
            "nonzero_candidate_count": t.parent_nonzero_candidate_count,
            "model_seed_count": t.parent_model_seed_count,
            "checkpoints_per_seed": t.checkpoints_per_seed,
            "first_nonzero_update": t.first_nonzero_update,
            "update_stride": t.update_stride,
            "last_nonzero_update": t.last_nonzero_update,
            "validation_path_count": t.validation_path_count,
            "confirmation_path_count": t.confirmation_path_count,
        }
    )
    return result


def _baseline_metrics(
    *, advantage: bool = False, harm: bool = False
) -> dict[str, object]:
    t = FalseDiscoveryThresholds()
    result: dict[str, object] = {name: 1 for name in BASELINE_REPLAY_FLAGS}
    result.update(
        {
            "confirmation_path_count": t.confirmation_path_count,
            "parent_family_size": t.parent_confirmation_family_size,
            "baseline_family_size": t.baseline_family_size,
            "bootstrap_replicates": t.bootstrap_replicates,
            "simultaneous_confidence": t.simultaneous_confidence,
            "maximum_three_contrast_identity_error": 0.0,
            "controller_planning_authorized": 0,
            "all_simultaneous_lower_bounds_positive": int(advantage),
            "overall_and_four_quartile_upper_bounds_negative": int(harm),
        }
    )
    return result


def _candidate_metrics(
    *,
    selected_resolved: bool = False,
    resolved: int = 0,
    directional: int = 0,
    qualifying: int = 0,
    audit_inconclusive: bool = False,
) -> dict[str, object]:
    t = FalseDiscoveryThresholds()
    result: dict[str, object] = {name: 1 for name in CANDIDATE_REPLAY_FLAGS}
    result.update(
        {
            "candidate_count": t.parent_nonzero_candidate_count,
            "validation_path_count": t.validation_path_count,
            "residual_search_family_size": t.residual_search_family_size,
            "candidate_direction_family_size": t.candidate_direction_family_size,
            "bootstrap_replicates": t.bootstrap_replicates,
            "simultaneous_confidence": t.simultaneous_confidence,
            "historical_selected_seed": t.historical_selected_seed,
            "historical_selected_update": t.historical_selected_update,
            "maximum_candidate_record_replay_error": 0.0,
            "selected_update_all_four_lower_bounds_positive": int(
                selected_resolved
            ),
            "residual_resolved_candidate_count": resolved,
            "direction_compatible_candidate_count": directional,
            "qualifying_candidate_count": qualifying,
            "selection_audit_inconclusive": int(audit_inconclusive),
        }
    )
    return result


def _passing_preflight() -> dict[str, object]:
    gate = evaluate_preflight_gate(_preflight_metrics())
    assert gate["passed"] == 1
    return gate


def _adjudication(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    gate = evaluate_adjudication_gate(
        baseline_gate=baseline, candidate_gate=candidate
    )
    assert gate["passed"] == 1
    return gate


def test_thresholds_are_frozen() -> None:
    with pytest.raises(FalseDiscoveryGateError, match="frozen"):
        FalseDiscoveryThresholds(bootstrap_replicates=100)


def test_preflight_is_strict_and_execution_failure_is_readable() -> None:
    passing = evaluate_preflight_gate(_preflight_metrics())
    assert passing["passed"] == 1
    assert passing["parent_mutations"] == 0
    assert passing["controller_control_planning_authorized"] == 0

    bad_metrics = _preflight_metrics()
    bad_metrics["candidate_checkpoint_hashes_valid"] = 0
    failed = evaluate_preflight_gate(bad_metrics)
    assert failed["passed"] == 0
    assert failed["scientific_evidence_complete"] == 0
    assert failed["failure_domain"] == "forensic_evidence"

    execution = evaluate_preflight_gate(
        {"evaluation_status": "execution_failed", "failure_code": "io_error"}
    )
    assert execution["evaluation_status"] == "execution_failed"
    assert execution["failure_code"] == "io_error"
    assert execution["passed"] == 0


@pytest.mark.parametrize(
    ("advantage", "harm", "expected"),
    [
        (True, False, SealedBaselineClassification.ADVANTAGE_CONFIRMED.value),
        (False, True, SealedBaselineClassification.HARM_CONFIRMED.value),
        (False, False, SealedBaselineClassification.NOT_ESTABLISHED.value),
    ],
)
def test_closed_baseline_classifications(
    advantage: bool, harm: bool, expected: str
) -> None:
    gate = evaluate_baseline_gate(
        _baseline_metrics(advantage=advantage, harm=harm)
    )
    assert gate["passed"] == 1
    assert gate["baseline_classification"] == expected
    assert gate["posthoc_non_authorizing"] == 1
    assert gate["controller_control_planning_authorized"] == 0


def test_baseline_degenerate_or_contradictory_evidence_fails_closed() -> None:
    degenerate = _baseline_metrics()
    degenerate["finite_positive_standard_errors"] = 0
    gate = evaluate_baseline_gate(degenerate)
    assert gate["passed"] == 0
    assert gate["baseline_classification"] == (
        SealedBaselineClassification.EVIDENCE_INVALID.value
    )

    contradictory = evaluate_baseline_gate(
        _baseline_metrics(advantage=True, harm=True)
    )
    assert contradictory["passed"] == 0
    assert contradictory["checks"]["classification_mutually_exclusive"][
        "passed"
    ] == 0


def test_three_contrast_identity_uses_inclusive_frozen_boundary() -> None:
    t = FalseDiscoveryThresholds()
    exact_metrics = _baseline_metrics()
    exact_metrics["maximum_three_contrast_identity_error"] = (
        t.maximum_three_contrast_identity_error
    )
    assert evaluate_baseline_gate(exact_metrics)["passed"] == 1

    over_metrics = dict(exact_metrics)
    over_metrics["maximum_three_contrast_identity_error"] = nextafter(
        t.maximum_three_contrast_identity_error, float("inf")
    )
    assert evaluate_baseline_gate(over_metrics)["passed"] == 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"selected_resolved": True, "resolved": 1, "directional": 1, "qualifying": 1},
            CandidateAuditClassification.RESIDUAL_SIGNAL_RESOLVED.value,
        ),
        (
            {"selected_resolved": False, "resolved": 0, "directional": 0, "qualifying": 0},
            CandidateAuditClassification.SELECTED_UPDATE_BELOW_RESOLUTION.value,
        ),
        (
            {"selected_resolved": True, "resolved": 1, "directional": 0, "qualifying": 0},
            CandidateAuditClassification.DIRECTIONALLY_INCOMPATIBLE_WITH_ZERO.value,
        ),
        (
            {"audit_inconclusive": True},
            CandidateAuditClassification.INCONCLUSIVE.value,
        ),
    ],
)
def test_closed_candidate_classifications(
    kwargs: dict[str, object], expected: str
) -> None:
    gate = evaluate_candidate_gate(_candidate_metrics(**kwargs))  # type: ignore[arg-type]
    assert gate["passed"] == 1
    assert gate["candidate_classification"] == expected
    assert gate["controller_control_planning_authorized"] == 0


def test_candidate_replay_or_count_defect_fails_closed() -> None:
    replay = _candidate_metrics()
    replay["historical_selection_hash_reproduced"] = 0
    gate = evaluate_candidate_gate(replay)
    assert gate["passed"] == 0
    assert gate["candidate_classification"] == (
        CandidateAuditClassification.REPLAY_DEFECT.value
    )

    inconsistent = _candidate_metrics(
        selected_resolved=True, resolved=0, directional=1, qualifying=0
    )
    gate = evaluate_candidate_gate(inconsistent)
    assert gate["passed"] == 0
    assert gate["checks"]["candidate_classification_counts_consistent"][
        "passed"
    ] == 0


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected", "planning_bit"),
    [
        (
            SealedBaselineClassification.ADVANTAGE_CONFIRMED,
            CandidateAuditClassification.RESIDUAL_SIGNAL_RESOLVED,
            FalseDiscoveryDecision.RETAINED_BASELINE_V3_SELECTION_DESIGN_READY,
            "retained_baseline_v3_design_planning_authorized",
        ),
        (
            SealedBaselineClassification.ADVANTAGE_CONFIRMED,
            CandidateAuditClassification.SELECTED_UPDATE_BELOW_RESOLUTION,
            FalseDiscoveryDecision.BASELINE_ONLY_REQUIRES_FRESH_CONFIRMATION_DESIGN,
            "baseline_only_fresh_confirmation_design_planning_authorized",
        ),
        (
            SealedBaselineClassification.HARM_CONFIRMED,
            CandidateAuditClassification.SELECTED_UPDATE_BELOW_RESOLUTION,
            FalseDiscoveryDecision.ZERO_BASELINE_V3_LEARNABILITY_READY,
            "zero_baseline_v3_design_planning_authorized",
        ),
        (
            SealedBaselineClassification.NOT_ESTABLISHED,
            CandidateAuditClassification.SELECTED_UPDATE_BELOW_RESOLUTION,
            FalseDiscoveryDecision.SELECTION_RESOLUTION_FAILURE_CONFIRMED,
            "selection_design_repair_planning_authorized",
        ),
        (
            SealedBaselineClassification.NOT_ESTABLISHED,
            CandidateAuditClassification.INCONCLUSIVE,
            FalseDiscoveryDecision.BASELINE_AND_RESIDUAL_UNRESOLVED,
            "training_only_variance_audit_planning_authorized",
        ),
    ],
)
def test_closed_terminal_decision_matrix_and_planning_only(
    baseline: SealedBaselineClassification,
    candidate: CandidateAuditClassification,
    expected: FalseDiscoveryDecision,
    planning_bit: str,
) -> None:
    baseline_gate = evaluate_baseline_gate(
        _baseline_metrics(
            advantage=baseline is SealedBaselineClassification.ADVANTAGE_CONFIRMED,
            harm=baseline is SealedBaselineClassification.HARM_CONFIRMED,
        )
    )
    candidate_kwargs = {
        CandidateAuditClassification.RESIDUAL_SIGNAL_RESOLVED: dict(
            selected_resolved=True, resolved=1, directional=1, qualifying=1
        ),
        CandidateAuditClassification.SELECTED_UPDATE_BELOW_RESOLUTION: dict(),
        CandidateAuditClassification.INCONCLUSIVE: dict(
            audit_inconclusive=True
        ),
    }[candidate]
    candidate_gate = evaluate_candidate_gate(_candidate_metrics(**candidate_kwargs))
    decision = decide_workflow(
        preflight_gate=_passing_preflight(),
        baseline_gate=baseline_gate,
        candidate_gate=candidate_gate,
        adjudication_gate=_adjudication(baseline_gate, candidate_gate),
    )
    assert decision["decision"] == expected.value
    assert decision[planning_bit] == 1
    assert decision["controller_control_planning_authorized"] == 0
    assert decision["physical_training_authorized"] == 0
    assert decision["confirmation_authorized"] == 0
    assert decision["sampling_authorized"] == 0
    assert decision["historical_v2_decision_remains_terminal"] == 1


def test_invalid_evidence_and_replay_have_distinct_terminal_decisions() -> None:
    invalid_baseline_metrics = _baseline_metrics()
    invalid_baseline_metrics["rows_finite"] = 0
    invalid_baseline = evaluate_baseline_gate(invalid_baseline_metrics)
    candidate = evaluate_candidate_gate(_candidate_metrics())
    forensic = decide_workflow(
        preflight_gate=_passing_preflight(),
        baseline_gate=invalid_baseline,
        candidate_gate=candidate,
    )
    assert forensic["decision"] == (
        FalseDiscoveryDecision.FORENSIC_EVIDENCE_INVALID.value
    )
    assert forensic["forensic_evidence_repair_planning_authorized"] == 1

    baseline = evaluate_baseline_gate(_baseline_metrics())
    defect_metrics = _candidate_metrics()
    defect_metrics["stored_candidate_records_reproduced"] = 0
    defect = evaluate_candidate_gate(defect_metrics)
    implementation = decide_workflow(
        preflight_gate=_passing_preflight(),
        baseline_gate=baseline,
        candidate_gate=defect,
    )
    assert implementation["decision"] == (
        FalseDiscoveryDecision.IMPLEMENTATION_OR_REPLAY_DEFECT.value
    )
    assert implementation["implementation_replay_repair_planning_authorized"] == 1


def test_required_gate_is_completion_not_favorable_scientific_outcome() -> None:
    preflight = _passing_preflight()
    baseline = evaluate_baseline_gate(_baseline_metrics(harm=True))
    candidate = evaluate_candidate_gate(_candidate_metrics())
    adjudicate = _adjudication(baseline, candidate)
    workflow = evaluate_required_gate(
        preflight_gate=preflight,
        baseline_gate=baseline,
        candidate_gate=candidate,
        adjudication_gate=adjudicate,
        require_gate="decision",
    )
    assert workflow["required_gate_pass"] == 1
    assert workflow["required_gate_exit_code"] == 0
    assert workflow["decision"]["decision"] == (
        FalseDiscoveryDecision.ZERO_BASELINE_V3_LEARNABILITY_READY.value
    )
    assert workflow["fresh_v3_learnability_design_planning_authorized"] == 1
    assert workflow["zero_baseline_v3_design_planning_authorized"] == 1
    assert workflow["artifacts_must_be_committed_before_required_gate_exit"] == 1
    assert workflow["controller_control_planning_authorized"] == 0


def test_required_gate_fails_for_incomplete_or_defective_decision() -> None:
    preflight = _passing_preflight()
    baseline = evaluate_baseline_gate(_baseline_metrics())
    defect_metrics = _candidate_metrics()
    defect_metrics["candidate_checkpoint_hashes_valid"] = 0
    candidate = evaluate_candidate_gate(defect_metrics)
    workflow = evaluate_required_gate(
        preflight_gate=preflight,
        baseline_gate=baseline,
        candidate_gate=candidate,
        require_gate="decision",
    )
    assert workflow["required_gate_pass"] == 0
    assert workflow["required_gate_exit_code"] == 1
    assert workflow["decision"]["decision"] == (
        FalseDiscoveryDecision.IMPLEMENTATION_OR_REPLAY_DEFECT.value
    )
    assert workflow["components"]["decision"]["passed"] == 0

    with pytest.raises(FalseDiscoveryGateError, match="unknown required gate"):
        evaluate_required_gate(
            preflight_gate=preflight,
            baseline_gate=baseline,
            candidate_gate=candidate,
            require_gate="controller",
        )


def test_required_decision_needs_committed_adjudication_and_matching_gate() -> None:
    preflight = _passing_preflight()
    baseline = evaluate_baseline_gate(_baseline_metrics(harm=True))
    candidate = evaluate_candidate_gate(_candidate_metrics())

    missing = evaluate_required_gate(
        preflight_gate=preflight,
        baseline_gate=baseline,
        candidate_gate=candidate,
        adjudication_gate=None,
        require_gate="decision",
    )
    assert missing["required_gate_pass"] == 0
    assert missing["components"]["adjudicate"]["evaluation_status"] == (
        "not_evaluated"
    )

    adjudication = _adjudication(baseline, candidate)
    decision = decide_workflow(
        preflight_gate=preflight,
        baseline_gate=baseline,
        candidate_gate=candidate,
        adjudication_gate=adjudication,
    )
    stale_decision = dict(decision)
    stale_decision["decision"] = (
        FalseDiscoveryDecision.BASELINE_AND_RESIDUAL_UNRESOLVED.value
    )
    stale_gate = evaluate_decision_gate(stale_decision)
    assert stale_gate["passed"] == 1

    mismatched = evaluate_required_gate(
        preflight_gate=preflight,
        baseline_gate=baseline,
        candidate_gate=candidate,
        adjudication_gate=adjudication,
        decision_gate=stale_gate,
        require_gate="decision",
    )
    assert mismatched["required_gate_pass"] == 0
    terminal = mismatched["components"]["decision"]
    assert terminal["failure_domain"] == "decision_binding"
    assert terminal["checks"][
        "terminal_decision_matches_recomputed_decision"
    ]["passed"] == 0


def test_waiting_records_are_fail_closed() -> None:
    decision = decide_workflow(
        preflight_gate=not_evaluated_gate("preflight", "not run"),
        baseline_gate=None,
        candidate_gate=None,
    )
    assert decision["decision"] == "ready_for_preflight"
    assert decision["evaluation_status"] == "not_evaluated"
    assert decision["controller_control_planning_authorized"] == 0


def test_decision_gate_rejects_smuggled_controller_authority() -> None:
    baseline = evaluate_baseline_gate(_baseline_metrics(harm=True))
    candidate = evaluate_candidate_gate(_candidate_metrics())
    decision = decide_workflow(
        preflight_gate=_passing_preflight(),
        baseline_gate=baseline,
        candidate_gate=candidate,
        adjudication_gate=_adjudication(baseline, candidate),
    )
    tampered = dict(decision)
    tampered["controller_control_planning_authorized"] = 1
    gate = evaluate_decision_gate(tampered)
    assert gate["passed"] == 0
    assert gate["checks"]["controller_control_planning_authorized"]["passed"] == 0
