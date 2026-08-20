"""Fail-closed decision gate for absolute-coordinate adjudication.

The adjudication is read-only.  It may support a finite absolute-coordinate
representation hypothesis, but it cannot authorize fresh evidence, model
training, controller execution, reconstruction, or sampling.  The strongest
scientific outcome recommends only drafting a separately reviewed learner
plan.
"""

from __future__ import annotations

from typing import Any, Mapping


ABSOLUTE_COORDINATE_GATE_VERSION = "d0-jacobi-rb-absolute-coordinate-gate-v1"
SCHEMA = ABSOLUTE_COORDINATE_GATE_VERSION
LATER_QUARTILES = ("q1", "q2", "q3")

INTEGRITY_DECISIONS = (
    "control_provenance_invalid",
    "portable_directional_parent_invalid",
    "coordinate_hypothesis_plan_invalid",
    "coarse_witness_replay_invalid",
    "translation_symmetry_audit_invalid",
    "coordinate_projection_algebra_invalid",
    "coordinate_inference_invalid",
)
SCIENTIFIC_DECISIONS = (
    "coarse_signal_nonreplicating_stop",
    "absolute_coordinate_signal_not_detected_stop",
    "absolute_coordinate_signal_partial_stop",
    "absolute_coordinate_representation_hypothesis_supported",
)
DECISION_ORDER = (*INTEGRITY_DECISIONS, *SCIENTIFIC_DECISIONS)
INVALID_DECISIONS = frozenset(INTEGRITY_DECISIONS)

ZERO_AUTHORIZATION_FIELDS = (
    "cache_generation_authorized",
    "new_path_generation_authorized",
    "physical_training_authorized",
    "new_learner_training_authorized",
    "fresh_coordinate_learner_plan_authorized",
    "fresh_fit_authorized",
    "fresh_calibration_authorized",
    "fresh_selection_authorized",
    "confirmation_authorized",
    "production_refinement_authorized",
    "controller_planning_authorized",
    "controller_execution_authorized",
    "reconstruction_authorized",
    "reverse_sampling_authorized",
    "sampling_authorized",
)
ZERO_WORK_FIELDS = (
    "new_transitions_generated",
    "fresh_physical_labels_opened",
    "physical_training_performed",
    "new_learner_training_performed",
    "optimizer_updates_performed",
    "new_checkpoints_created",
    "controller_trajectories_executed",
    "reconstructions_created",
    "reverse_sampling_performed",
    "sampling_performed",
    "samples_created",
    "parent_files_modified",
    "historical_design_evidence_authorizing",
)
ZERO_CLAIM_FIELDS = (
    "unique_representation_identified",
    "architecture_identified",
    "full_conditional_score_identified",
    "full_conditional_mean_zero_proven",
)


def _bit(value: Any) -> bool | None:
    """Read a strict JSON bit without accepting strings or arbitrary numbers."""

    if isinstance(value, (bool, int)) and int(value) in (0, 1):
        return bool(int(value))
    return None


def _named_bit(evidence: Mapping[str, Any], *names: str) -> bool | None:
    for name in names:
        if name in evidence:
            return _bit(evidence[name])
    return None


def safety_record() -> dict[str, int]:
    """Return the immutable no-work, no-authority, and restricted-claim bits."""

    return {
        **{name: 0 for name in ZERO_AUTHORIZATION_FIELDS},
        **{name: 0 for name in ZERO_WORK_FIELDS},
        **{name: 0 for name in ZERO_CLAIM_FIELDS},
    }


def _next_action(decision: str) -> str:
    actions = {
        "control_provenance_invalid": (
            "repair immutable control and coarse-witness provenance before interpretation"
        ),
        "portable_directional_parent_invalid": (
            "repair or reacquire the immutable portable directional parent archive"
        ),
        "coordinate_hypothesis_plan_invalid": (
            "repair and reseal the absolute-coordinate hypothesis plan"
        ),
        "coarse_witness_replay_invalid": (
            "repair the read-only coarse-witness replay without changing its parents"
        ),
        "translation_symmetry_audit_invalid": (
            "repair the static and dynamic translation-symmetry audit"
        ),
        "coordinate_projection_algebra_invalid": (
            "repair the frozen coordinate projection and decomposition algebra"
        ),
        "coordinate_inference_invalid": (
            "repair report-only coordinate inference without changing sealed evidence"
        ),
        "coarse_signal_nonreplicating_stop": (
            "stop; the q0 coarse positive control did not replicate"
        ),
        "absolute_coordinate_signal_not_detected_stop": (
            "stop; no later-quartile absolute-coordinate signal was detected"
        ),
        "absolute_coordinate_signal_partial_stop": (
            "stop; the coordinate signal did not replicate across every later quartile"
        ),
        "absolute_coordinate_representation_hypothesis_supported": (
            "draft a separately reviewed fresh coordinate learner plan; do not train or sample"
        ),
    }
    return actions[decision]


def _decision_record(
    decision: str,
    *,
    q0_positive_control: bool | None = None,
    later_quartile_positive: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    if decision not in DECISION_ORDER:
        raise ValueError(f"unknown absolute-coordinate decision: {decision}")
    scientific = decision in SCIENTIFIC_DECISIONS
    supported = decision == "absolute_coordinate_representation_hypothesis_supported"
    later = (
        {quartile: int(bool(later_quartile_positive[quartile])) for quartile in LATER_QUARTILES}
        if later_quartile_positive is not None
        else None
    )
    result: dict[str, Any] = {
        "schema": "d0-jacobi-rb-absolute-coordinate-decision-v1",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "decision": decision,
        "scientific_outcome": decision if scientific else None,
        "terminal": 1,
        "scientific_evidence_complete": int(scientific),
        "invalid_evidence": int(not scientific),
        "valid_scientific_stop": int(scientific and not supported),
        "absolute_coordinate_representation_hypothesis_supported": int(supported),
        "fresh_coordinate_learner_plan_drafting_recommended": int(supported),
        "recommended_next_action": _next_action(decision),
        "next_action": _next_action(decision),
        "claim_scope": (
            "finite absolute-coordinate feature family for q1-q3 under the frozen "
            "one-image exact-K=512 Jacobi/Rao-Blackwell design"
        ),
        **safety_record(),
    }
    if q0_positive_control is not None:
        result["q0_positive_control"] = int(q0_positive_control)
    if later is not None:
        result["later_quartile_positive"] = later
        result["later_quartile_positive_count"] = sum(later.values())
    return result


def _provenance_failure(evidence: Mapping[str, Any]) -> str | None:
    """Apply general provenance before the directional-parent-specific failure.

    ``provenance_valid`` is the original composite field.  New callers may
    instead provide both parent-specific fields.  When a composite field is
    present it remains authoritative, while an explicitly supplied specific
    field can still fail closed.
    """

    general = _named_bit(evidence, "control_provenance_valid", "provenance_valid")
    portable_present = "portable_directional_parent_valid" in evidence
    coarse_present = "coarse_witness_parent_valid" in evidence
    portable = _named_bit(evidence, "portable_directional_parent_valid")
    coarse = _named_bit(evidence, "coarse_witness_parent_valid")

    if general is False or (coarse_present and coarse is not True):
        return "control_provenance_invalid"
    if portable_present and portable is not True:
        return "portable_directional_parent_invalid"
    if general is None and not (portable is True and coarse is True):
        return "control_provenance_invalid"
    return None


def decide_absolute_coordinate(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply integrity precedence and the closed q1-q3 scientific partition."""

    row: Mapping[str, Any] = evidence if isinstance(evidence, Mapping) else {}

    provenance_failure = _provenance_failure(row)
    if provenance_failure is not None:
        return _decision_record(provenance_failure)

    integrity_flags = (
        (
            ("coordinate_hypothesis_plan_valid", "hypothesis_plan_valid"),
            "coordinate_hypothesis_plan_invalid",
        ),
        (("coarse_witness_replay_valid",), "coarse_witness_replay_invalid"),
        (("translation_symmetry_audit_valid",), "translation_symmetry_audit_invalid"),
        (("coordinate_projection_algebra_valid",), "coordinate_projection_algebra_invalid"),
        (("coordinate_inference_valid",), "coordinate_inference_invalid"),
    )
    for names, invalid_decision in integrity_flags:
        if _named_bit(row, *names) is not True:
            return _decision_record(invalid_decision)

    q0 = _named_bit(row, "q0_positive_control")
    later_raw = row.get("later_quartile_positive")
    if q0 is None or not isinstance(later_raw, Mapping):
        return _decision_record("coordinate_inference_invalid")
    if set(later_raw) != set(LATER_QUARTILES):
        return _decision_record("coordinate_inference_invalid")
    later = {quartile: _bit(later_raw[quartile]) for quartile in LATER_QUARTILES}
    if any(value is None for value in later.values()):
        return _decision_record("coordinate_inference_invalid")
    valid_later = {quartile: bool(later[quartile]) for quartile in LATER_QUARTILES}

    if not q0:
        decision = "coarse_signal_nonreplicating_stop"
    else:
        positive_count = sum(valid_later.values())
        if positive_count == 0:
            decision = "absolute_coordinate_signal_not_detected_stop"
        elif positive_count < len(LATER_QUARTILES):
            decision = "absolute_coordinate_signal_partial_stop"
        else:
            decision = "absolute_coordinate_representation_hypothesis_supported"
    return _decision_record(
        decision,
        q0_positive_control=q0,
        later_quartile_positive=valid_later,
    )


def decision_exit_code(decision: Mapping[str, Any]) -> int:
    """Scientific stops/support exit zero; integrity failures exit one."""

    return 0 if str(decision.get("decision", "")) in SCIENTIFIC_DECISIONS else 1


__all__ = [
    "ABSOLUTE_COORDINATE_GATE_VERSION",
    "DECISION_ORDER",
    "INTEGRITY_DECISIONS",
    "INVALID_DECISIONS",
    "LATER_QUARTILES",
    "SCHEMA",
    "SCIENTIFIC_DECISIONS",
    "ZERO_AUTHORIZATION_FIELDS",
    "ZERO_CLAIM_FIELDS",
    "ZERO_WORK_FIELDS",
    "decide_absolute_coordinate",
    "decision_exit_code",
    "safety_record",
]
