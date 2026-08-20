"""Fail-closed gates for the frequency-one coordinate learnability workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping


SCHEMA = "d0-jacobi-rb-boundary-tangent-frequency1-coordinate-gate-v1"
SCHEMA_VERSION = 1
REQUIRED_GATES = (
    "none",
    "preflight",
    "cache",
    "controls",
    "train",
    "select",
    "confirm",
    "terminal",
)


class FrequencyOneCoordinateGateError(ValueError):
    """A gate record or requested gate violates the frozen contract."""


class FrequencyOneCoordinateDecision(str, Enum):
    PARENT_PROVENANCE_INVALID = "frequency1_coordinate_parent_provenance_invalid"
    CONTRACT_INVALID = "frequency1_coordinate_contract_invalid"
    PATH_OR_RESOURCE_PLAN_INVALID = "frequency1_coordinate_path_or_resource_plan_invalid"
    EXACT_CACHE_INVALID = "frequency1_coordinate_exact_cache_invalid"
    PRELABEL_CONTROLS_FAILED = "frequency1_coordinate_prelabel_controls_failed"
    PHYSICAL_TRAINING_INVALID = "frequency1_coordinate_physical_training_invalid"
    VALIDATION_INFERENCE_INVALID = "frequency1_coordinate_validation_inference_invalid"
    NO_VALIDATION_CANDIDATE = "no_frequency1_coordinate_validation_candidate"
    FRESH_CONFIRMATION_INVALID = "frequency1_coordinate_fresh_confirmation_invalid"
    SIGNAL_NOT_CONFIRMED = "frequency1_coordinate_signal_not_confirmed"
    CONFIRMED = "exact_rb_frequency1_coordinate_boundary_tangent_time_local_signal_confirmed"


DECISION_ORDER = tuple(item.value for item in FrequencyOneCoordinateDecision)
INTEGRITY_DECISIONS = frozenset(
    {
        FrequencyOneCoordinateDecision.PARENT_PROVENANCE_INVALID.value,
        FrequencyOneCoordinateDecision.CONTRACT_INVALID.value,
        FrequencyOneCoordinateDecision.PATH_OR_RESOURCE_PLAN_INVALID.value,
        FrequencyOneCoordinateDecision.EXACT_CACHE_INVALID.value,
        FrequencyOneCoordinateDecision.PRELABEL_CONTROLS_FAILED.value,
        FrequencyOneCoordinateDecision.PHYSICAL_TRAINING_INVALID.value,
        FrequencyOneCoordinateDecision.VALIDATION_INFERENCE_INVALID.value,
        FrequencyOneCoordinateDecision.FRESH_CONFIRMATION_INVALID.value,
    }
)
VALID_SCIENTIFIC_NEGATIVES = frozenset(
    {
        FrequencyOneCoordinateDecision.NO_VALIDATION_CANDIDATE.value,
        FrequencyOneCoordinateDecision.SIGNAL_NOT_CONFIRMED.value,
    }
)
FINAL_DECISION = FrequencyOneCoordinateDecision.CONFIRMED.value
SCIENTIFIC_TERMINAL_DECISIONS = frozenset((*VALID_SCIENTIFIC_NEGATIVES, FINAL_DECISION))
PENDING_DECISIONS = (
    "ready_for_preflight",
    "ready_for_cache",
    "ready_for_controls",
    "ready_for_train",
    "ready_for_select",
    "frequency1_coordinate_validation_nominee_sealed",
    "ready_for_confirm",
)


@dataclass(frozen=True)
class FrequencyOneCoordinateThresholds:
    minimum_transition_throughput: float = 1_300.0
    maximum_peak_cuda_memory_fraction: float = 0.80
    maximum_persisted_artifact_bytes: int = 3 * 1024**3
    maximum_exact_capture_seconds: float = 160.0 * 3600.0
    maximum_forward_batch_size: int = 32
    maximum_target_batch_size: int = 32
    maximum_max_t_working_bytes: int = 64 * 1024**2
    maximum_mass_error: float = 2.0e-12
    certificate_fraction: float = 1.0
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_time_fraction: float = 0.10
    train_path_count: int = 64
    validation_path_count: int = 32
    confirmation_path_count: int = 64
    train_row_count: int = 114_688
    validation_row_count: int = 57_344
    confirmation_row_count: int = 114_688
    component_count: int = 228
    candidate_count: int = 120
    search_family_size: int = 27_360
    bootstrap_replicates: int = 50_000
    confidence: float = 0.995
    synthetic_maximum_relative_validation_mse: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PREFLIGHT_PROVENANCE_FLAGS = (
    "parent_provenance_valid",
    "absolute_coordinate_parent_valid",
    "memory_v3_protocol_parent_valid",
    "coarse_witness_parent_valid",
    "portable_directional_parent_valid",
    "parent_immutability_valid",
    "source_closure_valid",
    "source_image_binding_valid",
)
PREFLIGHT_CONTRACT_FLAGS = (
    "scientific_contract_valid",
    "coordinate_feature_contract_valid",
    "coordinate_lattice_span_valid",
    "predictor_architecture_valid",
    "initialization_mapping_valid",
    "update_zero_equivalence_valid",
    "optimizer_parameter_inclusion_valid",
    "model_input_firewall_valid",
    "exact_backend_seam_valid",
)
PREFLIGHT_PLAN_FLAGS = (
    "path_plan_valid",
    "seed_plan_valid",
    "cohort_plan_valid",
    "role_opening_plan_valid",
    "resource_plan_valid",
    "bootstrap_plan_valid",
)
PREFLIGHT_FLAGS = (*PREFLIGHT_PROVENANCE_FLAGS, *PREFLIGHT_CONTRACT_FLAGS, *PREFLIGHT_PLAN_FLAGS)

CACHE_FLAGS = (
    "train_cache_valid",
    "validation_cache_valid",
    "exact_row_and_transition_counts_valid",
    "certificate_and_conservation_valid",
    "input_label_separation_valid",
    "train_validation_role_separation_valid",
    "validation_labels_unopened",
    "coordinate_absent_from_cache",
    "confirmation_namespace_unopened",
    "cache_resource_valid",
)
CONTROLS_FLAGS = (
    "coordinate_geometry_control_valid",
    "initialization_identity_control_valid",
    "symmetry_break_control_valid",
    "synthetic_coordinate_teacher_valid",
    "exact_model_null_valid",
    "firewall_batch_memory_restart_valid",
    "physical_train_labels_unopened",
    "validation_labels_unopened",
    "confirmation_namespace_unopened",
)
TRAIN_FLAGS = (
    "physical_train_label_open_order_valid",
    "three_physical_tasks_complete",
    "all_checkpoints_complete",
    "candidate_inventory_valid",
    "training_target_scale_training_only",
    "finite_training_outputs",
    "batch_and_memory_contract_valid",
    "validation_labels_unopened",
    "confirmation_namespace_unopened",
)
SELECT_INTEGRITY_FLAGS = (
    "selection_plan_sealed_before_labels",
    "validation_opened_once",
    "candidate_grid_complete",
    "component_order_valid",
    "bootstrap_shards_valid",
    "validation_inference_valid",
    "negative_values_untruncated",
    "standard_error_floor_unused",
    "ranking_rule_valid",
    "confirmation_namespace_unopened",
)
SELECT_FLAGS = (*SELECT_INTEGRITY_FLAGS, "all_228_simultaneous_lower_bounds_positive")
CONFIRM_INTEGRITY_FLAGS = (
    "sealed_nominee_unchanged",
    "confirmation_opened_once",
    "confirmation_paths_valid",
    "streaming_reductions_only",
    "confirmation_inference_valid",
    "negative_values_untruncated",
    "standard_error_floor_unused",
    "no_reselection_or_path_extension",
)
CONFIRM_FLAGS = (*CONFIRM_INTEGRITY_FLAGS, "all_228_simultaneous_lower_bounds_positive")

STAGE_FLAGS: dict[str, tuple[str, ...]] = {
    "preflight": PREFLIGHT_FLAGS,
    "cache": CACHE_FLAGS,
    "controls": CONTROLS_FLAGS,
    "train": TRAIN_FLAGS,
    "select": SELECT_FLAGS,
    "confirm": CONFIRM_FLAGS,
}


def _strict_one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and _strict_one(gate.get("passed"))
    )


def _not_evaluated(gate: Mapping[str, Any] | None) -> bool:
    return not isinstance(gate, Mapping) or gate.get("evaluation_status", "not_evaluated") == "not_evaluated"


def _valid_negative(gate: Mapping[str, Any] | None, stage: str) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and not _passed(gate)
        and _strict_one(gate.get("valid_scientific_negative"))
        and _strict_one(gate.get("stage_execution_valid", 1))
        and _strict_one(gate.get("inference_valid", 1))
        and ((stage == "select" and _strict_one(gate.get("no_validation_candidate")))
             or (stage == "confirm" and _strict_one(gate.get("signal_not_confirmed"))))
    )


def _safety_scope(**performed: bool) -> dict[str, int]:
    values = {
        "new_transitions_generated": 0,
        "physical_training_performed": 0,
        "validation_selection_performed": 0,
        "confirmation_performed": 0,
        "controller_trajectories_executed": 0,
        "reconstruction_performed": 0,
        "reverse_sampling_performed": 0,
        "sampling_performed": 0,
        "controller_execution_authorized": 0,
        "reconstruction_authorized": 0,
        "reverse_sampling_authorized": 0,
        "sampling_authorized": 0,
        "confirmation_reuse_authorized": 0,
    }
    for name, enabled in performed.items():
        if name in values:
            values[name] = int(bool(enabled))
    return values


def not_evaluated_gate(stage: str, reason: str = "not evaluated") -> dict[str, Any]:
    if stage not in STAGE_FLAGS:
        raise FrequencyOneCoordinateGateError(f"unknown gate stage: {stage}")
    return {
        "schema": f"{SCHEMA}-{stage}",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "not_evaluated",
        "stage": stage,
        "passed": 0,
        "valid_scientific_negative": 0,
        "reason": str(reason),
        **_safety_scope(),
    }


def _default_failure_domain(stage: str, checks: Mapping[str, int]) -> str:
    if stage == "preflight":
        if any(not checks[name] for name in PREFLIGHT_PROVENANCE_FLAGS):
            return "parent_provenance"
        if any(not checks[name] for name in PREFLIGHT_CONTRACT_FLAGS):
            return "frequency1_coordinate_contract"
        return "path_or_resource_plan"
    return {
        "cache": "exact_cache",
        "controls": "prelabel_controls",
        "train": "physical_training",
        "select": "validation_inference",
        "confirm": "fresh_confirmation",
    }[stage]


def evaluate_stage_gate(stage: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if stage not in STAGE_FLAGS:
        raise FrequencyOneCoordinateGateError(f"unknown stage gate: {stage}")
    status = str(metrics.get("evaluation_status", "not_evaluated"))
    flags = STAGE_FLAGS[stage]
    checks = {name: int(_strict_one(metrics.get(name))) for name in flags}
    execution_valid = int(_strict_one(metrics.get("stage_execution_valid", 1)))
    inference_valid = int(_strict_one(metrics.get("inference_valid", 1)))
    integrity_flags = (
        SELECT_INTEGRITY_FLAGS if stage == "select" else
        CONFIRM_INTEGRITY_FLAGS if stage == "confirm" else flags
    )
    integrity_valid = all(checks[name] for name in integrity_flags)
    passed = int(
        status == "evaluated"
        and bool(execution_valid)
        and bool(inference_valid)
        and all(checks.values())
    )
    no_candidate = int(
        stage == "select"
        and status == "evaluated"
        and bool(execution_valid)
        and bool(inference_valid)
        and integrity_valid
        and _strict_one(metrics.get("no_validation_candidate"))
        and not checks.get("all_228_simultaneous_lower_bounds_positive", 0)
    )
    not_confirmed = int(
        stage == "confirm"
        and status == "evaluated"
        and bool(execution_valid)
        and bool(inference_valid)
        and integrity_valid
        and _strict_one(metrics.get("signal_not_confirmed"))
        and not checks.get("all_228_simultaneous_lower_bounds_positive", 0)
    )
    valid_negative = int(bool(no_candidate or not_confirmed))
    failure_domain = metrics.get("failure_domain")
    if not passed and not valid_negative and status != "not_evaluated" and not failure_domain:
        failure_domain = _default_failure_domain(stage, checks)
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}-{stage}",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": status,
        "stage": stage,
        "passed": passed,
        "valid_scientific_negative": valid_negative,
        "stage_execution_valid": execution_valid,
        "inference_valid": inference_valid,
        "scientific_evidence_complete": int(bool(passed or valid_negative)),
        "checks": checks,
        "no_validation_candidate": no_candidate,
        "signal_not_confirmed": not_confirmed,
        "confirmation_authorized": int(stage == "select" and bool(passed)),
        "controller_control_patch_planning_authorized": int(stage == "confirm" and bool(passed)),
        "thresholds": FrequencyOneCoordinateThresholds().to_dict(),
        **_safety_scope(
            new_transitions_generated=_strict_one(metrics.get("new_transitions_generated")),
            physical_training_performed=_strict_one(metrics.get("physical_training_performed")),
            validation_selection_performed=_strict_one(metrics.get("validation_selection_performed")),
            confirmation_performed=_strict_one(metrics.get("confirmation_performed")),
        ),
    }
    if failure_domain:
        result["failure_domain"] = str(failure_domain)
    for name in ("failure_code", "message", "error", "nominee"):
        if name in metrics:
            result[name] = metrics[name]
    return result


def evaluate_preflight_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("preflight", metrics)


def evaluate_cache_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("cache", metrics)


def evaluate_controls_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("controls", metrics)


def evaluate_train_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("train", metrics)


def evaluate_select_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("select", metrics)


def evaluate_confirm_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("confirm", metrics)


def validate_resource_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the numeric preflight resource contract without tolerances."""

    t = FrequencyOneCoordinateThresholds()
    numeric = {
        "transition_throughput": float(metrics.get("transition_throughput", math.nan)),
        "peak_cuda_memory_fraction": float(metrics.get("peak_cuda_memory_fraction", math.nan)),
        "projected_persisted_bytes": float(metrics.get("projected_persisted_bytes", math.nan)),
        "projected_exact_capture_seconds": float(metrics.get("projected_exact_capture_seconds", math.nan)),
        "forward_batch_size": float(metrics.get("forward_batch_size", math.nan)),
        "target_batch_size": float(metrics.get("target_batch_size", math.nan)),
        "max_t_working_bytes": float(metrics.get("max_t_working_bytes", math.nan)),
    }
    checks = {
        "throughput": math.isfinite(numeric["transition_throughput"]) and numeric["transition_throughput"] >= t.minimum_transition_throughput,
        "memory": math.isfinite(numeric["peak_cuda_memory_fraction"]) and 0.0 <= numeric["peak_cuda_memory_fraction"] <= t.maximum_peak_cuda_memory_fraction,
        "persistence": math.isfinite(numeric["projected_persisted_bytes"]) and 0.0 <= numeric["projected_persisted_bytes"] <= t.maximum_persisted_artifact_bytes,
        "capture_time": math.isfinite(numeric["projected_exact_capture_seconds"]) and 0.0 <= numeric["projected_exact_capture_seconds"] <= t.maximum_exact_capture_seconds,
        "forward_batch": math.isfinite(numeric["forward_batch_size"]) and 1.0 <= numeric["forward_batch_size"] <= t.maximum_forward_batch_size,
        "target_batch": math.isfinite(numeric["target_batch_size"]) and 1.0 <= numeric["target_batch_size"] <= t.maximum_target_batch_size,
        "max_t_memory": math.isfinite(numeric["max_t_working_bytes"]) and 0.0 <= numeric["max_t_working_bytes"] < t.maximum_max_t_working_bytes,
    }
    return {
        "schema": f"{SCHEMA}-resource-validation",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": int(all(checks.values())),
        "checks": {name: int(value) for name, value in checks.items()},
        "observed": numeric,
        "thresholds": t.to_dict(),
    }


def _decision_record(
    decision: str,
    *,
    complete: bool,
    valid_scientific_negative: bool = False,
    training: bool = False,
    selection: bool = False,
    confirmation: bool = False,
) -> dict[str, Any]:
    pending = decision in PENDING_DECISIONS
    positive = decision == FINAL_DECISION
    actions = {
        "ready_for_preflight": "run the eight-path seam and contract checks",
        "ready_for_cache": "generate only fresh train and validation caches",
        "ready_for_controls": "run every prelabel coordinate control",
        "ready_for_train": "open only physical training labels and train three seeds",
        "ready_for_select": "seal the 27,360-member family and open validation once",
        "frequency1_coordinate_validation_nominee_sealed": "open the one sealed fresh confirmation",
        "ready_for_confirm": "open the one sealed fresh confirmation",
        FrequencyOneCoordinateDecision.NO_VALIDATION_CANDIDATE.value: "close the run without opening confirmation",
        FrequencyOneCoordinateDecision.SIGNAL_NOT_CONFIRMED.value: "retain the sealed negative; no second confirmation",
        FINAL_DECISION: "draft a separate controls-only controller-control patch",
    }
    return {
        "schema": f"{SCHEMA}-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "not_evaluated" if pending else "evaluated",
        "decision": decision,
        "terminal": int(not pending),
        "scientific_evidence_complete": int(bool(complete)),
        "valid_scientific_negative": int(bool(valid_scientific_negative)),
        "invalid_evidence": int(decision in INTEGRITY_DECISIONS),
        "cache_generation_authorized": int(decision == "ready_for_cache"),
        "controls_authorized": int(decision == "ready_for_controls"),
        "physical_training_authorized": int(decision == "ready_for_train"),
        "validation_selection_authorized": int(decision == "ready_for_select"),
        "confirmation_authorized": int(decision in {"ready_for_confirm", "frequency1_coordinate_validation_nominee_sealed"}),
        "controller_control_patch_planning_authorized": int(positive),
        "recommended_next_action": actions.get(decision, "repair only the named closed gate"),
        **_safety_scope(
            physical_training_performed=training,
            validation_selection_performed=selection,
            confirmation_performed=confirmation,
        ),
    }


def _preflight_failure(gate: Mapping[str, Any]) -> str:
    domain = str(gate.get("failure_domain", ""))
    checks = gate.get("checks") if isinstance(gate.get("checks"), Mapping) else {}
    if domain in {"parent_provenance", "control_provenance"} or any(
        int(checks.get(name, 0)) == 0 for name in PREFLIGHT_PROVENANCE_FLAGS
    ):
        return FrequencyOneCoordinateDecision.PARENT_PROVENANCE_INVALID.value
    if domain == "frequency1_coordinate_contract" or any(
        int(checks.get(name, 0)) == 0 for name in PREFLIGHT_CONTRACT_FLAGS
    ):
        return FrequencyOneCoordinateDecision.CONTRACT_INVALID.value
    return FrequencyOneCoordinateDecision.PATH_OR_RESOURCE_PLAN_INVALID.value


def decide_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    controls_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    select_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gates = (preflight_gate, cache_gate, controls_gate, train_gate, select_gate, confirm_gate)
    if any(
        isinstance(gate, Mapping)
        and not _passed(gate)
        and str(gate.get("failure_domain", "")) in {"parent_provenance", "control_provenance"}
        for gate in gates
    ):
        return _decision_record(
            FrequencyOneCoordinateDecision.PARENT_PROVENANCE_INVALID.value,
            complete=False,
        )
    if _not_evaluated(preflight_gate):
        return _decision_record("ready_for_preflight", complete=False)
    if not _passed(preflight_gate):
        return _decision_record(_preflight_failure(dict(preflight_gate or {})), complete=False)

    stages = (
        ("cache", cache_gate, FrequencyOneCoordinateDecision.EXACT_CACHE_INVALID.value),
        ("controls", controls_gate, FrequencyOneCoordinateDecision.PRELABEL_CONTROLS_FAILED.value),
        ("train", train_gate, FrequencyOneCoordinateDecision.PHYSICAL_TRAINING_INVALID.value),
        ("select", select_gate, FrequencyOneCoordinateDecision.VALIDATION_INFERENCE_INVALID.value),
        ("confirm", confirm_gate, FrequencyOneCoordinateDecision.FRESH_CONFIRMATION_INVALID.value),
    )
    ready = {
        "cache": "ready_for_cache",
        "controls": "ready_for_controls",
        "train": "ready_for_train",
        "select": "ready_for_select",
        "confirm": "ready_for_confirm",
    }
    for stage, gate, invalid in stages:
        if _not_evaluated(gate):
            return _decision_record(
                ready[stage], complete=False,
                training=stage in {"select", "confirm"},
                selection=stage == "confirm",
            )
        if _passed(gate):
            continue
        if stage == "select" and _valid_negative(gate, stage):
            return _decision_record(
                FrequencyOneCoordinateDecision.NO_VALIDATION_CANDIDATE.value,
                complete=True, valid_scientific_negative=True,
                training=True, selection=True,
            )
        if stage == "confirm" and _valid_negative(gate, stage):
            return _decision_record(
                FrequencyOneCoordinateDecision.SIGNAL_NOT_CONFIRMED.value,
                complete=True, valid_scientific_negative=True,
                training=True, selection=True, confirmation=True,
            )
        return _decision_record(
            invalid, complete=False,
            training=stage in {"train", "select", "confirm"},
            selection=stage in {"select", "confirm"},
            confirmation=stage == "confirm",
        )
    return _decision_record(
        FINAL_DECISION,
        complete=True,
        training=True,
        selection=True,
        confirmation=True,
    )


def evaluate_required_gate(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    controls_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    select_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    require_gate: str,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if require_gate not in REQUIRED_GATES:
        raise FrequencyOneCoordinateGateError(f"unknown required gate: {require_gate}")
    resolved_decision = dict(decision or decide_workflow(
        preflight_gate=preflight_gate,
        cache_gate=cache_gate,
        controls_gate=controls_gate,
        train_gate=train_gate,
        select_gate=select_gate,
        confirm_gate=confirm_gate,
    ))
    gates = {
        "preflight": preflight_gate,
        "cache": cache_gate,
        "controls": controls_gate,
        "train": train_gate,
        "select": select_gate,
        "confirm": confirm_gate,
    }
    if require_gate == "none":
        passed = True
    elif require_gate == "terminal":
        passed = (
            str(resolved_decision.get("decision")) in SCIENTIFIC_TERMINAL_DECISIONS
            and _strict_one(resolved_decision.get("scientific_evidence_complete"))
        )
    else:
        order = REQUIRED_GATES[1:-1]
        index = order.index(require_gate)
        passed = all(_passed(gates[name]) for name in order[: index + 1])
    return {
        "schema": f"{SCHEMA}-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_gate_pass": int(bool(passed)),
        "required_gate_exit_code": 0 if passed else 1,
        "artifacts_must_be_committed_before_required_gate_exit": 1,
        "decision": resolved_decision,
        "components": {name: dict(gate or not_evaluated_gate(name)) for name, gate in gates.items()},
    }


def decision_exit_code(decision: Mapping[str, Any]) -> int:
    name = str(decision.get("decision", ""))
    return 0 if name in {*PENDING_DECISIONS, *SCIENTIFIC_TERMINAL_DECISIONS} else 1


# Descriptive workflow aliases.
evaluate_frequency1_preflight_gate = evaluate_preflight_gate
evaluate_frequency1_cache_gate = evaluate_cache_gate
evaluate_frequency1_controls_gate = evaluate_controls_gate
evaluate_frequency1_train_gate = evaluate_train_gate
evaluate_frequency1_select_gate = evaluate_select_gate
evaluate_frequency1_confirm_gate = evaluate_confirm_gate
evaluate_frequency1_workflow = evaluate_required_gate


__all__ = [
    "CACHE_FLAGS",
    "CONFIRM_INTEGRITY_FLAGS",
    "CONFIRM_FLAGS",
    "CONTROLS_FLAGS",
    "DECISION_ORDER",
    "FINAL_DECISION",
    "FrequencyOneCoordinateDecision",
    "FrequencyOneCoordinateGateError",
    "FrequencyOneCoordinateThresholds",
    "INTEGRITY_DECISIONS",
    "PENDING_DECISIONS",
    "PREFLIGHT_CONTRACT_FLAGS",
    "PREFLIGHT_FLAGS",
    "PREFLIGHT_PLAN_FLAGS",
    "PREFLIGHT_PROVENANCE_FLAGS",
    "REQUIRED_GATES",
    "SCHEMA",
    "SCIENTIFIC_TERMINAL_DECISIONS",
    "SELECT_INTEGRITY_FLAGS",
    "SELECT_FLAGS",
    "STAGE_FLAGS",
    "TRAIN_FLAGS",
    "VALID_SCIENTIFIC_NEGATIVES",
    "decide_workflow",
    "decision_exit_code",
    "evaluate_cache_gate",
    "evaluate_confirm_gate",
    "evaluate_controls_gate",
    "evaluate_frequency1_cache_gate",
    "evaluate_frequency1_confirm_gate",
    "evaluate_frequency1_controls_gate",
    "evaluate_frequency1_preflight_gate",
    "evaluate_frequency1_select_gate",
    "evaluate_frequency1_train_gate",
    "evaluate_frequency1_workflow",
    "evaluate_preflight_gate",
    "evaluate_required_gate",
    "evaluate_select_gate",
    "evaluate_stage_gate",
    "evaluate_train_gate",
    "not_evaluated_gate",
    "validate_resource_metrics",
]
