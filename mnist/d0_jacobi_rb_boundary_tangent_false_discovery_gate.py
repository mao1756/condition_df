"""Fail-closed gates for the sealed boundary-tangent false-discovery audit.

This module is deliberately pure and additive.  It classifies already-sealed
evidence; it does not read files, evaluate models, create paths, train, or
sample.  Callers can therefore write every evidence artifact atomically and
only then enforce ``required_gate_pass``.

No outcome of this historical adjudication authorizes a controller.  At most,
some outcomes authorize *planning* a fresh and independently confirmed v3
learnability design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping


SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-"
    "false-discovery-adjudication-gate"
)
SCHEMA_VERSION = 1


class FalseDiscoveryGateError(ValueError):
    """A caller requested a gate outside the closed workflow schema."""


class SealedBaselineClassification(str, Enum):
    ADVANTAGE_CONFIRMED = "sealed_baseline_advantage_confirmed"
    HARM_CONFIRMED = "sealed_baseline_harm_confirmed"
    NOT_ESTABLISHED = "sealed_baseline_not_established"
    EVIDENCE_INVALID = "sealed_baseline_evidence_invalid"


class CandidateAuditClassification(str, Enum):
    RESIDUAL_SIGNAL_RESOLVED = (
        "current_candidate_family_residual_signal_resolved"
    )
    SELECTED_UPDATE_BELOW_RESOLUTION = "selected_update_below_resolution"
    DIRECTIONALLY_INCOMPATIBLE_WITH_ZERO = (
        "residual_signal_directionally_incompatible_with_zero"
    )
    INCONCLUSIVE = "selection_audit_inconclusive"
    REPLAY_DEFECT = "implementation_or_replay_defect"


class FalseDiscoveryDecision(str, Enum):
    FORENSIC_EVIDENCE_INVALID = "forensic_evidence_invalid"
    IMPLEMENTATION_OR_REPLAY_DEFECT = "implementation_or_replay_defect"
    RETAINED_BASELINE_V3_SELECTION_DESIGN_READY = (
        "retained_baseline_v3_selection_design_ready"
    )
    BASELINE_ONLY_REQUIRES_FRESH_CONFIRMATION_DESIGN = (
        "baseline_only_requires_fresh_confirmation_design"
    )
    ZERO_BASELINE_V3_LEARNABILITY_READY = (
        "zero_baseline_v3_learnability_ready"
    )
    BASELINE_AND_RESIDUAL_UNRESOLVED = "baseline_and_residual_unresolved"
    SELECTION_RESOLUTION_FAILURE_CONFIRMED = (
        "selection_resolution_failure_confirmed"
    )


@dataclass(frozen=True)
class FalseDiscoveryThresholds:
    """Frozen audit sizes and numerical/statistical conventions."""

    parent_terminal_decision: str = "selection_false_discovery"
    parent_source_fingerprint: str = (
        "dfe9c3357c1d1ba614cccfdcaca84b3c3bf2d0967d6a3a3b15e5a0421d04243e"
    )
    parent_scientific_config_sha256: str = (
        "fadc1eb31ad0fb1ccb900f41f1eb8523c67c6ae39e09c783698aa5a20634cdec"
    )
    parent_registry_semantic_sha256: str = (
        "36bf43c0a108549954617a78625d4fd65820141c950ba84330133de1f8648580"
    )
    parent_nonzero_candidate_count: int = 120
    parent_model_seed_count: int = 3
    checkpoints_per_seed: int = 40
    first_nonzero_update: int = 100
    update_stride: int = 100
    last_nonzero_update: int = 4_000
    historical_selected_seed: int = 261_314
    historical_selected_update: int = 800
    validation_path_count: int = 32
    confirmation_path_count: int = 64
    parent_confirmation_family_size: int = 228
    baseline_family_size: int = 229
    residual_search_family_size: int = 480
    candidate_direction_family_size: int = 228
    bootstrap_replicates: int = 50_000
    simultaneous_confidence: float = 0.995
    maximum_three_contrast_identity_error: float = 5.0e-15
    maximum_candidate_record_replay_error: float = 1.0e-12

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if type(value) is not type(field.default) or value != field.default:
                raise FalseDiscoveryGateError(
                    f"{name} is frozen at {field.default}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# These bits describe work or scientific authority.  They are appended last to
# every record, so an input mapping can never smuggle an authorization through.
_NO_WORK = {
    "new_exact_transitions": 0,
    "new_path_ids": 0,
    "optimizer_updates": 0,
    "confirmation_label_writes": 0,
    "parent_mutations": 0,
    "production_cache_generation_performed": 0,
    "physical_training_performed": 0,
    "new_confirmation_performed": 0,
    "controller_control_trajectory_performed": 0,
    "full_reverse_path_performed": 0,
    "reconstruction_performed": 0,
    "reverse_sampling_performed": 0,
    "sampling_performed": 0,
}

_NO_EXECUTION_AUTHORIZATION = {
    "cache_generation_authorized": 0,
    "physical_training_authorized": 0,
    "confirmation_authorized": 0,
    "controller_control_planning_authorized": 0,
    "controller_control_trajectory_authorized": 0,
    "full_reverse_path_authorized": 0,
    "reconstruction_authorized": 0,
    "reverse_sampling_authorized": 0,
    "sampling_authorized": 0,
    "full_dataset_training_authorized": 0,
    "known_prior_claim_authorized": 0,
    "unsplit_generator_claim_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
}

_PLANNING_BITS = (
    "fresh_v3_learnability_design_planning_authorized",
    "retained_baseline_v3_design_planning_authorized",
    "baseline_only_fresh_confirmation_design_planning_authorized",
    "zero_baseline_v3_design_planning_authorized",
    "training_only_variance_audit_planning_authorized",
    "selection_design_repair_planning_authorized",
    "forensic_evidence_repair_planning_authorized",
    "implementation_replay_repair_planning_authorized",
)


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nonnegative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _check(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _failed(checks: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {
        str(name)
        for name, record in checks.items()
        if not _one(record.get("passed"))
    }


def _safety_record() -> dict[str, int]:
    return {
        **_NO_WORK,
        **_NO_EXECUTION_AUTHORIZATION,
        **{name: 0 for name in _PLANNING_BITS},
        "old_confirmation_paths_burned": 1,
        "old_confirmation_reuse_authorized": 0,
        "parent_terminal_decision_preserved": 1,
    }


def _gate(
    stage: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    failure_domain: str | None,
    scientific_evidence_complete: bool,
    stage_execution_valid: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    normalized = {str(name): dict(record) for name, record in checks.items()}
    passed = bool(normalized) and not _failed(normalized)
    return {
        "schema": f"{SCHEMA}-{stage}",
        "schema_version": SCHEMA_VERSION,
        "gate": stage,
        "evaluation_status": "evaluated",
        "checks": normalized,
        "passed": int(passed),
        "failure_domain": None if passed else str(failure_domain or "evidence"),
        "stage_execution_valid": int(stage_execution_valid),
        "scientific_evidence_complete": int(scientific_evidence_complete),
        **extra,
        **_safety_record(),
    }


def _execution_failed_gate(
    stage: str,
    metrics: Mapping[str, Any],
    *,
    failure_domain: str,
) -> dict[str, Any]:
    result = _gate(
        stage,
        {"stage_execution": _check(0, "==", 1, False)},
        failure_domain=str(metrics.get("failure_domain") or failure_domain),
        scientific_evidence_complete=False,
        stage_execution_valid=False,
    )
    result["evaluation_status"] = "execution_failed"
    result["failure_code"] = str(
        metrics.get("failure_code") or f"{stage}_execution_failed"
    )
    return result


def not_evaluated_gate(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}-{stage}",
        "schema_version": SCHEMA_VERSION,
        "gate": str(stage),
        "evaluation_status": "not_evaluated",
        "reason": str(reason),
        "passed": 0,
        "failure_domain": None,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        **_safety_record(),
    }


PREFLIGHT_FLAGS = (
    "parent_terminal_selection_false_discovery",
    "parent_scientific_evidence_complete",
    "parent_source_fingerprint_valid",
    "parent_scientific_config_hash_valid",
    "parent_registry_valid",
    "selection_seal_valid",
    "train_seal_valid",
    "confirmation_seal_valid",
    "confirmation_index_valid",
    "candidate_task_records_valid",
    "candidate_checkpoint_hashes_valid",
    "validation_cache_hashes_valid",
    "confirmation_risk_shard_hashes_valid",
    "candidate_grid_exact",
    "path_namespaces_disjoint",
    "confirmation_paths_denied_to_replay",
    "raw_confirmation_targets_absent",
    "parent_files_immutable",
)


def evaluate_preflight_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: FalseDiscoveryThresholds | None = None,
) -> dict[str, Any]:
    """Verify the sealed parent before any numerical evidence is opened."""

    t = thresholds or FalseDiscoveryThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate(
            "preflight", metrics, failure_domain="forensic_evidence"
        )
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in PREFLIGHT_FLAGS
    }
    exact_counts = {
        "nonzero_candidate_count": t.parent_nonzero_candidate_count,
        "model_seed_count": t.parent_model_seed_count,
        "checkpoints_per_seed": t.checkpoints_per_seed,
        "first_nonzero_update": t.first_nonzero_update,
        "update_stride": t.update_stride,
        "last_nonzero_update": t.last_nonzero_update,
        "validation_path_count": t.validation_path_count,
        "confirmation_path_count": t.confirmation_path_count,
    }
    checks.update(
        {
            name: _check(
                metrics.get(name), "==", expected,
                metrics.get(name) == expected,
            )
            for name, expected in exact_counts.items()
        }
    )
    exact_bindings = {
        "parent_terminal_decision": t.parent_terminal_decision,
        "parent_source_fingerprint": t.parent_source_fingerprint,
        "parent_scientific_config_sha256": t.parent_scientific_config_sha256,
        "parent_registry_semantic_sha256": t.parent_registry_semantic_sha256,
    }
    checks.update(
        {
            name: _check(
                metrics.get(name), "==", expected, metrics.get(name) == expected
            )
            for name, expected in exact_bindings.items()
        }
    )
    # A forensic child must remain read-only even if its parent is invalid.
    for name, expected in _NO_WORK.items():
        checks[name] = _check(
            metrics.get(name, expected), "==", expected,
            metrics.get(name, expected) == expected,
        )
    failed = _failed(checks)
    return _gate(
        "preflight",
        checks,
        failure_domain="forensic_evidence",
        scientific_evidence_complete=not bool(failed),
        thresholds=t.to_dict(),
    )


BASELINE_REPLAY_FLAGS = (
    "confirmation_risk_shards_hash_bound",
    "row_dtype_float64",
    "rows_finite",
    "complete_cartesian_identity",
    "unique_sample_keys",
    "three_contrast_identity_valid",
    "parent_228_replay_exact",
    "parent_point_estimates_replay_exact",
    "direct_derived_total_exact",
    "whole_path_resampling",
    "two_sided_max_abs_t",
    "higher_quantile_interpolation",
    "deterministic_philox_bootstrap",
    "simultaneous_inference_valid",
    "finite_positive_standard_errors",
    "baseline_posthoc_non_authorizing",
    "old_confirmation_paths_burned",
)


def evaluate_baseline_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: FalseDiscoveryThresholds | None = None,
) -> dict[str, Any]:
    """Classify the sealed post-hoc baseline-versus-zero evidence."""

    t = thresholds or FalseDiscoveryThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        failed = _execution_failed_gate(
            "baseline", metrics, failure_domain="sealed_baseline_evidence"
        )
        failed["baseline_classification"] = (
            SealedBaselineClassification.EVIDENCE_INVALID.value
        )
        return failed
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in BASELINE_REPLAY_FLAGS
    }
    exact_values = {
        "confirmation_path_count": t.confirmation_path_count,
        "parent_family_size": t.parent_confirmation_family_size,
        "baseline_family_size": t.baseline_family_size,
        "bootstrap_replicates": t.bootstrap_replicates,
        "simultaneous_confidence": t.simultaneous_confidence,
    }
    checks.update(
        {
            name: _check(
                metrics.get(name), "==", expected,
                metrics.get(name) == expected,
            )
            for name, expected in exact_values.items()
        }
    )
    identity_error = metrics.get("maximum_three_contrast_identity_error")
    checks["maximum_three_contrast_identity_error"] = _check(
        identity_error,
        "<=",
        t.maximum_three_contrast_identity_error,
        _finite(identity_error)
        and 0.0 <= float(identity_error)
        <= t.maximum_three_contrast_identity_error,
    )
    checks["controller_planning_authorized"] = _check(
        metrics.get("controller_planning_authorized", 0),
        "==",
        0,
        _zero(metrics.get("controller_planning_authorized", 0)),
    )
    advantage = _one(metrics.get("all_simultaneous_lower_bounds_positive"))
    harm = _one(
        metrics.get("overall_and_four_quartile_upper_bounds_negative")
    )
    mutually_exclusive = not (advantage and harm)
    checks["classification_mutually_exclusive"] = _check(
        int(mutually_exclusive), "==", 1, mutually_exclusive
    )
    failed = _failed(checks)
    if failed:
        classification = SealedBaselineClassification.EVIDENCE_INVALID
    elif advantage:
        classification = SealedBaselineClassification.ADVANTAGE_CONFIRMED
    elif harm:
        classification = SealedBaselineClassification.HARM_CONFIRMED
    else:
        classification = SealedBaselineClassification.NOT_ESTABLISHED
    return _gate(
        "baseline",
        checks,
        failure_domain="sealed_baseline_evidence",
        scientific_evidence_complete=not bool(failed),
        baseline_classification=classification.value,
        posthoc_non_authorizing=1,
        thresholds=t.to_dict(),
    )


CANDIDATE_REPLAY_FLAGS = (
    "candidate_checkpoint_hashes_valid",
    "validation_cache_hashes_valid",
    "candidate_grid_exact",
    "validation_paths_only",
    "confirmation_paths_denied",
    "raw_target_unchanged",
    "permitted_later_state_inputs_only",
    "direct_float64_mse",
    "update_zero_reproduces_baseline",
    "stored_candidate_records_reproduced",
    "historical_selection_reproduced",
    "historical_selection_hash_reproduced",
    "joint_whole_path_resampling",
    "one_sided_max_t",
    "higher_quantile_interpolation",
    "deterministic_philox_bootstrap",
    "search_aware_inference_valid",
    "finite_positive_standard_errors",
    "candidate_order_invariant",
    "path_order_invariant",
)


def evaluate_candidate_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: FalseDiscoveryThresholds | None = None,
) -> dict[str, Any]:
    """Classify the search-aware replay of the fixed candidate family."""

    t = thresholds or FalseDiscoveryThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        failed = _execution_failed_gate(
            "candidate", metrics, failure_domain="implementation_or_replay"
        )
        failed["candidate_classification"] = (
            CandidateAuditClassification.REPLAY_DEFECT.value
        )
        return failed
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in CANDIDATE_REPLAY_FLAGS
    }
    exact_values = {
        "candidate_count": t.parent_nonzero_candidate_count,
        "validation_path_count": t.validation_path_count,
        "residual_search_family_size": t.residual_search_family_size,
        "candidate_direction_family_size": t.candidate_direction_family_size,
        "bootstrap_replicates": t.bootstrap_replicates,
        "simultaneous_confidence": t.simultaneous_confidence,
        "historical_selected_seed": t.historical_selected_seed,
        "historical_selected_update": t.historical_selected_update,
    }
    checks.update(
        {
            name: _check(
                metrics.get(name), "==", expected,
                metrics.get(name) == expected,
            )
            for name, expected in exact_values.items()
        }
    )
    replay_error = metrics.get("maximum_candidate_record_replay_error")
    checks["maximum_candidate_record_replay_error"] = _check(
        replay_error,
        "<=",
        t.maximum_candidate_record_replay_error,
        _finite(replay_error)
        and 0.0 <= float(replay_error)
        <= t.maximum_candidate_record_replay_error,
    )
    selected_resolved = _one(
        metrics.get("selected_update_all_four_lower_bounds_positive")
    )
    audit_inconclusive_raw = metrics.get("selection_audit_inconclusive", 0)
    audit_inconclusive = _one(audit_inconclusive_raw)
    checks["selection_audit_inconclusive"] = _check(
        audit_inconclusive_raw,
        "in",
        [0, 1],
        _zero(audit_inconclusive_raw) or audit_inconclusive,
    )
    count_names = (
        "residual_resolved_candidate_count",
        "direction_compatible_candidate_count",
        "qualifying_candidate_count",
    )
    for name in count_names:
        value = metrics.get(name)
        checks[name] = _check(
            value,
            "in",
            f"[0,{t.parent_nonzero_candidate_count}]",
            _nonnegative_integer(value)
            and int(value) <= t.parent_nonzero_candidate_count,
        )
    resolved = (
        int(metrics.get("residual_resolved_candidate_count", -1))
        if _nonnegative_integer(metrics.get("residual_resolved_candidate_count"))
        else -1
    )
    directional = (
        int(metrics.get("direction_compatible_candidate_count", -1))
        if _nonnegative_integer(metrics.get("direction_compatible_candidate_count"))
        else -1
    )
    qualifying = (
        int(metrics.get("qualifying_candidate_count", -1))
        if _nonnegative_integer(metrics.get("qualifying_candidate_count"))
        else -1
    )
    count_consistent = (
        resolved >= 0
        and directional >= 0
        and qualifying >= 0
        and qualifying <= resolved
        and qualifying <= directional
        and (not selected_resolved or resolved >= 1)
        and (not audit_inconclusive or (resolved == 0 and qualifying == 0))
    )
    checks["candidate_classification_counts_consistent"] = _check(
        int(count_consistent), "==", 1, count_consistent
    )
    failed = _failed(checks)
    if failed:
        classification = CandidateAuditClassification.REPLAY_DEFECT
    elif qualifying > 0:
        classification = CandidateAuditClassification.RESIDUAL_SIGNAL_RESOLVED
    elif resolved > 0:
        classification = (
            CandidateAuditClassification.DIRECTIONALLY_INCOMPATIBLE_WITH_ZERO
        )
    elif audit_inconclusive:
        classification = CandidateAuditClassification.INCONCLUSIVE
    elif not selected_resolved and resolved == 0:
        classification = (
            CandidateAuditClassification.SELECTED_UPDATE_BELOW_RESOLUTION
        )
    else:
        classification = CandidateAuditClassification.INCONCLUSIVE
    return _gate(
        "candidate",
        checks,
        failure_domain="implementation_or_replay",
        scientific_evidence_complete=not bool(failed),
        candidate_classification=classification.value,
        selected_update_residual_signal_resolved=int(selected_resolved),
        current_candidate_family_residual_signal_resolved=int(qualifying > 0),
        exact_validation_direction_compatible=int(directional > 0),
        thresholds=t.to_dict(),
    )


def _status(gate: Mapping[str, Any] | None) -> str:
    return str((gate or {}).get("evaluation_status", "not_evaluated"))


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return _status(gate) == "evaluated" and _one((gate or {}).get("passed"))


def evaluate_adjudication_gate(
    *,
    baseline_gate: Mapping[str, Any] | None,
    candidate_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Verify that both closed analyses produced complete classifications."""

    baseline = baseline_gate or not_evaluated_gate("baseline", "not run")
    candidate = candidate_gate or not_evaluated_gate("candidate", "not run")
    baseline_class = baseline.get("baseline_classification")
    candidate_class = candidate.get("candidate_classification")
    valid_baseline_classes = {item.value for item in SealedBaselineClassification}
    valid_candidate_classes = {item.value for item in CandidateAuditClassification}
    checks = {
        "baseline_gate_passed": _check(
            int(_passed(baseline)), "==", 1, _passed(baseline)
        ),
        "candidate_gate_passed": _check(
            int(_passed(candidate)), "==", 1, _passed(candidate)
        ),
        "baseline_classification_closed": _check(
            baseline_class,
            "in",
            sorted(valid_baseline_classes),
            baseline_class in valid_baseline_classes,
        ),
        "candidate_classification_closed": _check(
            candidate_class,
            "in",
            sorted(valid_candidate_classes),
            candidate_class in valid_candidate_classes,
        ),
        "baseline_evidence_complete": _check(
            baseline.get("scientific_evidence_complete"),
            "==",
            1,
            _one(baseline.get("scientific_evidence_complete")),
        ),
        "candidate_evidence_complete": _check(
            candidate.get("scientific_evidence_complete"),
            "==",
            1,
            _one(candidate.get("scientific_evidence_complete")),
        ),
    }
    failed = _failed(checks)
    return _gate(
        "adjudicate",
        checks,
        failure_domain=(
            "sealed_baseline_evidence"
            if not _passed(baseline)
            else "implementation_or_replay"
        ),
        scientific_evidence_complete=not bool(failed),
        baseline_classification=baseline_class,
        candidate_classification=candidate_class,
    )


def _decision_authorizations(decision: str) -> dict[str, int]:
    result = {name: 0 for name in _PLANNING_BITS}
    if decision == FalseDiscoveryDecision.FORENSIC_EVIDENCE_INVALID.value:
        result["forensic_evidence_repair_planning_authorized"] = 1
    elif decision == FalseDiscoveryDecision.IMPLEMENTATION_OR_REPLAY_DEFECT.value:
        result["implementation_replay_repair_planning_authorized"] = 1
    elif decision == (
        FalseDiscoveryDecision.RETAINED_BASELINE_V3_SELECTION_DESIGN_READY.value
    ):
        result["fresh_v3_learnability_design_planning_authorized"] = 1
        result["retained_baseline_v3_design_planning_authorized"] = 1
    elif decision == (
        FalseDiscoveryDecision.BASELINE_ONLY_REQUIRES_FRESH_CONFIRMATION_DESIGN.value
    ):
        result["baseline_only_fresh_confirmation_design_planning_authorized"] = 1
    elif decision == (
        FalseDiscoveryDecision.ZERO_BASELINE_V3_LEARNABILITY_READY.value
    ):
        result["fresh_v3_learnability_design_planning_authorized"] = 1
        result["zero_baseline_v3_design_planning_authorized"] = 1
    elif decision == (
        FalseDiscoveryDecision.BASELINE_AND_RESIDUAL_UNRESOLVED.value
    ):
        result["training_only_variance_audit_planning_authorized"] = 1
    elif decision == (
        FalseDiscoveryDecision.SELECTION_RESOLUTION_FAILURE_CONFIRMED.value
    ):
        result["selection_design_repair_planning_authorized"] = 1
    return result


def _decision_record(
    decision: str,
    *,
    evaluation_status: str = "evaluated",
    scientific_evidence_complete: bool,
    baseline_classification: str | None = None,
    candidate_classification: str | None = None,
) -> dict[str, Any]:
    safety = _safety_record()
    safety.update(_decision_authorizations(decision))
    return {
        "schema": f"{SCHEMA}-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": evaluation_status,
        "decision": decision,
        "scientific_evidence_complete": int(scientific_evidence_complete),
        "baseline_classification": baseline_classification,
        "candidate_classification": candidate_classification,
        "historical_v2_decision": "selection_false_discovery",
        "historical_v2_decision_remains_terminal": 1,
        "historical_confirmation_can_authorize_controller": 0,
        **safety,
    }


def decide_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    baseline_gate: Mapping[str, Any] | None,
    candidate_gate: Mapping[str, Any] | None,
    adjudication_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the closed decision table without any free-form recommendation."""

    if _status(preflight_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_preflight",
            evaluation_status="not_evaluated",
            scientific_evidence_complete=False,
        )
    if not _passed(preflight_gate):
        return _decision_record(
            FalseDiscoveryDecision.FORENSIC_EVIDENCE_INVALID.value,
            scientific_evidence_complete=False,
        )
    if _status(baseline_gate) == "not_evaluated" or _status(candidate_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_adjudication",
            evaluation_status="not_evaluated",
            scientific_evidence_complete=False,
        )
    baseline_class = str((baseline_gate or {}).get("baseline_classification", ""))
    candidate_class = str((candidate_gate or {}).get("candidate_classification", ""))
    if (
        not _passed(baseline_gate)
        or baseline_class == SealedBaselineClassification.EVIDENCE_INVALID.value
    ):
        decision = FalseDiscoveryDecision.FORENSIC_EVIDENCE_INVALID
        complete = False
    elif (
        not _passed(candidate_gate)
        or candidate_class == CandidateAuditClassification.REPLAY_DEFECT.value
    ):
        decision = FalseDiscoveryDecision.IMPLEMENTATION_OR_REPLAY_DEFECT
        complete = False
    elif adjudication_gate is not None and not _passed(adjudication_gate):
        decision = FalseDiscoveryDecision.IMPLEMENTATION_OR_REPLAY_DEFECT
        complete = False
    elif baseline_class == SealedBaselineClassification.HARM_CONFIRMED.value:
        decision = FalseDiscoveryDecision.ZERO_BASELINE_V3_LEARNABILITY_READY
        complete = True
    elif baseline_class == SealedBaselineClassification.ADVANTAGE_CONFIRMED.value:
        if candidate_class == CandidateAuditClassification.RESIDUAL_SIGNAL_RESOLVED.value:
            decision = (
                FalseDiscoveryDecision.RETAINED_BASELINE_V3_SELECTION_DESIGN_READY
            )
        else:
            decision = (
                FalseDiscoveryDecision.BASELINE_ONLY_REQUIRES_FRESH_CONFIRMATION_DESIGN
            )
        complete = True
    elif candidate_class == (
        CandidateAuditClassification.SELECTED_UPDATE_BELOW_RESOLUTION.value
    ):
        decision = FalseDiscoveryDecision.SELECTION_RESOLUTION_FAILURE_CONFIRMED
        complete = True
    else:
        decision = FalseDiscoveryDecision.BASELINE_AND_RESIDUAL_UNRESOLVED
        complete = True
    return _decision_record(
        decision.value,
        scientific_evidence_complete=complete,
        baseline_classification=baseline_class or None,
        candidate_classification=candidate_class or None,
    )


def evaluate_decision_gate(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a committed terminal classification and its authority bits."""

    terminal = {item.value for item in FalseDiscoveryDecision}
    invalid = {
        FalseDiscoveryDecision.FORENSIC_EVIDENCE_INVALID.value,
        FalseDiscoveryDecision.IMPLEMENTATION_OR_REPLAY_DEFECT.value,
    }
    decision_name = decision.get("decision")
    checks = {
        "closed_terminal_decision": _check(
            decision_name, "in", sorted(terminal), decision_name in terminal
        ),
        "scientific_evidence_complete": _check(
            decision.get("scientific_evidence_complete"),
            "==",
            1,
            _one(decision.get("scientific_evidence_complete")),
        ),
        "historical_v2_decision_remains_terminal": _check(
            decision.get("historical_v2_decision_remains_terminal"),
            "==",
            1,
            _one(decision.get("historical_v2_decision_remains_terminal")),
        ),
        "old_confirmation_paths_burned": _check(
            decision.get("old_confirmation_paths_burned"),
            "==",
            1,
            _one(decision.get("old_confirmation_paths_burned")),
        ),
    }
    for name, expected in _NO_EXECUTION_AUTHORIZATION.items():
        checks[name] = _check(
            decision.get(name), "==", expected, decision.get(name) == expected
        )
    failed = _failed(checks)
    # Evidence/implementation failures are closed outcomes, but they do not
    # satisfy a required decision gate because the scientific audit is incomplete.
    valid_terminal = decision_name in terminal and decision_name not in invalid
    checks["nondefective_terminal_classification"] = _check(
        int(valid_terminal), "==", 1, valid_terminal
    )
    failed = _failed(checks)
    return _gate(
        "decision",
        checks,
        failure_domain="decision",
        scientific_evidence_complete=not bool(failed),
        terminal_decision=decision_name,
    )


REQUIRED_GATES = ("none", "preflight", "adjudicate", "decision")


def evaluate_required_gate(
    *,
    preflight_gate: Mapping[str, Any] | None,
    baseline_gate: Mapping[str, Any] | None,
    candidate_gate: Mapping[str, Any] | None,
    adjudication_gate: Mapping[str, Any] | None = None,
    decision_gate: Mapping[str, Any] | None = None,
    require_gate: str,
) -> dict[str, Any]:
    """Return a pure workflow record; callers enforce exit only after writes."""

    if require_gate not in REQUIRED_GATES:
        raise FalseDiscoveryGateError(f"unknown required gate: {require_gate}")
    preflight = dict(
        preflight_gate or not_evaluated_gate("preflight", "not run")
    )
    baseline = dict(baseline_gate or not_evaluated_gate("baseline", "not run"))
    candidate = dict(candidate_gate or not_evaluated_gate("candidate", "not run"))
    # A decision gate may only build on a separately evaluated and committed
    # adjudication stage.  Reconstructing it implicitly here would allow a
    # caller to skip the stage seal while still satisfying ``require=decision``.
    adjudicate = dict(
        adjudication_gate
        or not_evaluated_gate("adjudicate", "not run")
    )
    decision = decide_workflow(
        preflight_gate=preflight,
        baseline_gate=baseline,
        candidate_gate=candidate,
        adjudication_gate=adjudicate,
    )
    terminal_gate = dict(decision_gate or evaluate_decision_gate(decision))
    terminal_checks = {
        str(name): dict(record)
        for name, record in dict(terminal_gate.get("checks", {})).items()
    }
    terminal_matches = terminal_gate.get("terminal_decision") == decision.get(
        "decision"
    )
    terminal_checks["terminal_decision_matches_recomputed_decision"] = _check(
        terminal_gate.get("terminal_decision"),
        "==",
        decision.get("decision"),
        terminal_matches,
    )
    terminal_gate["checks"] = terminal_checks
    if not terminal_matches:
        terminal_gate["passed"] = 0
        terminal_gate["failure_domain"] = "decision_binding"
        terminal_gate["scientific_evidence_complete"] = 0
    required_components = {
        "none": (),
        "preflight": (preflight,),
        "adjudicate": (preflight, adjudicate),
        "decision": (preflight, adjudicate, terminal_gate),
    }[require_gate]
    required_pass = all(_passed(component) for component in required_components)
    workflow_safety = _safety_record()
    for name in _PLANNING_BITS:
        workflow_safety[name] = int(_one(decision.get(name)))
    return {
        "schema": f"{SCHEMA}-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "required_gate_exit_code": 0 if required_pass else 1,
        "artifacts_must_be_committed_before_required_gate_exit": 1,
        "components": {
            "preflight": preflight,
            "baseline": baseline,
            "candidate": candidate,
            "adjudicate": adjudicate,
            "decision": terminal_gate,
        },
        "decision": decision,
        "thresholds": FalseDiscoveryThresholds().to_dict(),
        **workflow_safety,
    }


# Descriptive aliases used by the planned CLI and by report-only callers.
evaluate_false_discovery_preflight = evaluate_preflight_gate
evaluate_sealed_baseline_gate = evaluate_baseline_gate
evaluate_selection_resolution_gate = evaluate_candidate_gate
evaluate_baseline_adjudication_gate = evaluate_baseline_gate
evaluate_candidate_audit_gate = evaluate_candidate_gate
evaluate_adjudicate_gate = evaluate_adjudication_gate
decide_false_discovery_workflow = decide_workflow
evaluate_false_discovery_workflow = evaluate_required_gate


BASELINE_CLASSIFICATIONS = tuple(item.value for item in SealedBaselineClassification)
CANDIDATE_CLASSIFICATIONS = tuple(item.value for item in CandidateAuditClassification)
DECISION_VALUES = tuple(item.value for item in FalseDiscoveryDecision)


__all__ = [
    "BASELINE_CLASSIFICATIONS",
    "BASELINE_REPLAY_FLAGS",
    "CANDIDATE_CLASSIFICATIONS",
    "CANDIDATE_REPLAY_FLAGS",
    "CandidateAuditClassification",
    "DECISION_VALUES",
    "FalseDiscoveryDecision",
    "FalseDiscoveryGateError",
    "FalseDiscoveryThresholds",
    "PREFLIGHT_FLAGS",
    "REQUIRED_GATES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SealedBaselineClassification",
    "decide_false_discovery_workflow",
    "decide_workflow",
    "evaluate_adjudication_gate",
    "evaluate_adjudicate_gate",
    "evaluate_baseline_gate",
    "evaluate_baseline_adjudication_gate",
    "evaluate_candidate_gate",
    "evaluate_candidate_audit_gate",
    "evaluate_decision_gate",
    "evaluate_false_discovery_preflight",
    "evaluate_false_discovery_workflow",
    "evaluate_preflight_gate",
    "evaluate_required_gate",
    "evaluate_sealed_baseline_gate",
    "evaluate_selection_resolution_gate",
    "not_evaluated_gate",
]
