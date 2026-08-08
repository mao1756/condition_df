"""Fail-closed gates for immutable-cache v3 streaming-memory recovery.

Scientific train/selection/confirmation checks are delegated to the frozen v3
gate.  This additive layer gates the new host-backed, batch-32 execution
contract and gives memory-scheduling failures their own closed decisions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from mnist.d0_jacobi_rb_boundary_tangent_v3_gate import (
    BoundaryTangentV3Thresholds,
    evaluate_confirm_gate as _evaluate_v3_confirm_gate,
    evaluate_select_gate as _evaluate_v3_select_gate,
    evaluate_train_gate as _evaluate_v3_train_gate,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-v3-memory-gate"
SCHEMA_VERSION = 1
REQUIRED_GATES = ("none", "preflight", "train", "select", "confirm")


class BoundaryTangentV3MemoryGateError(ValueError):
    """Evidence violates the frozen streaming-memory recovery contract."""


class BoundaryTangentV3MemoryDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    IMMUTABLE_CACHE_BINDING_INVALID = "immutable_cache_binding_invalid"
    TRAINING_MEMORY_SCHEDULE_INVALID = "training_memory_schedule_invalid"
    TRAINING_MEMORY_RESOURCE_INFEASIBLE = "training_memory_resource_infeasible"
    TRAINING_CONTROLS_FAILED = "training_controls_failed"
    PHYSICAL_TRAINING_INVALID = "physical_training_invalid"
    NO_VALIDATION_CANDIDATE = "no_validation_candidate"
    VALIDATION_INFERENCE_INVALID = "validation_inference_invalid"
    FRESH_CONFIRMATION_INVALID = "fresh_confirmation_invalid"
    ZERO_BASELINE_V3_SIGNAL_NOT_CONFIRMED = (
        "zero_baseline_v3_signal_not_confirmed"
    )
    EXACT_RB_ZERO_BASELINE_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED = (
        "exact_rb_zero_baseline_boundary_tangent_time_local_signal_confirmed"
    )


@dataclass(frozen=True)
class BoundaryTangentV3MemoryThresholds:
    maximum_model_forward_batch_size: int = 32
    maximum_peak_memory_fraction: float = 0.80
    synthetic_scale_maximum_relative_error: float = 2.0e-15

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if type(value) is not type(field.default) or value != field.default:
                raise BoundaryTangentV3MemoryGateError(
                    f"{name} is frozen at {field.default}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MemoryRecoveryThresholds = BoundaryTangentV3MemoryThresholds

_FORBIDDEN = {
    "controller_control_trajectory_authorized": 0,
    "controller_control_trajectory_performed": 0,
    "complete_reverse_path_authorized": 0,
    "full_reverse_path_performed": 0,
    "reconstruction_authorized": 0,
    "reconstruction_performed": 0,
    "image_sampling_authorized": 0,
    "sampling_authorized": 0,
    "sampling_performed": 0,
    "reverse_sampling_authorized": 0,
    "reverse_sampling_performed": 0,
    "full_dataset_training_authorized": 0,
    "unsplit_generator_claim_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
}


def _scope(
    *,
    training: bool = False,
    selection: bool = False,
    confirmation: bool = False,
    planning: bool = False,
) -> dict[str, int]:
    return {
        "production_cache_generation_performed": 0,
        "immutable_parent_cache_reused": 1,
        "physical_training_performed": int(training),
        "validation_selection_performed": int(selection),
        "confirmation_performed": int(confirmation),
        "controller_control_planning_authorized": int(planning),
        **_FORBIDDEN,
    }


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def _gate(
    stage: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    failure_domain: str | None,
    scientific_evidence_complete: bool,
    stage_execution_valid: bool = True,
    numerically_valid: bool = True,
    resource_valid: bool = True,
    training: bool = False,
    selection: bool = False,
    confirmation: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    normalized = {str(name): dict(record) for name, record in checks.items()}
    passed = bool(normalized) and not _failed(normalized)
    return {
        "schema": f"{SCHEMA}-{stage}-gate",
        "schema_version": SCHEMA_VERSION,
        "gate": stage,
        "evaluation_status": "evaluated",
        "checks": normalized,
        "passed": int(passed),
        "failure_domain": None if passed else failure_domain,
        "stage_execution_valid": int(stage_execution_valid),
        "scientific_evidence_complete": int(scientific_evidence_complete),
        "numerically_valid": int(numerically_valid),
        "resource_valid": int(resource_valid),
        **_scope(
            training=training,
            selection=selection,
            confirmation=confirmation,
        ),
        **extra,
    }


def not_evaluated_gate(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}-{stage}-gate",
        "schema_version": SCHEMA_VERSION,
        "gate": stage,
        "evaluation_status": "not_evaluated",
        "reason": str(reason),
        "passed": 0,
        "failure_domain": None,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "numerically_valid": 0,
        "resource_valid": 0,
        **_scope(),
    }


PREFLIGHT_PROVENANCE_FLAGS = (
    "failed_parent_valid",
    "corrected_parent_adjudication_valid",
    "complete_parent_registry_valid",
    "parent_preflight_and_cache_passed",
    "parent_immutability_valid",
    "downstream_evidence_absent",
    "confirmation_namespace_unopened",
)
PREFLIGHT_BINDING_FLAGS = (
    "immutable_cache_binding_valid",
    "cache_seal_valid",
    "cache_indexes_valid",
    "cache_read_only",
    "cache_not_copied_or_linked",
    "physical_labels_deserialized_during_binding_zero",
)
PREFLIGHT_MEMORY_FLAGS = (
    "memory_contract_valid",
    "host_backed_input_store_valid",
    "host_backed_label_store_valid",
    "label_firewall_valid",
    "maximum_forward_batch_enforced",
    "full_cache_cuda_tensor_absent",
    "host_device_batch_equivalence_valid",
    "cuda_forward_backward_seam_valid",
    "streaming_reducer_valid",
    "automatic_batch_sizing_disabled",
    "allocator_workaround_disabled",
)


def evaluate_preflight_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3MemoryThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3MemoryThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        raw_domain = str(metrics.get("failure_domain") or "memory_preflight")
        resource_failure = raw_domain in {
            "memory_resource",
            "training_memory_resource",
            "resource_gate",
        } or "out_of_memory" in str(metrics.get("failure_code") or "")
        if raw_domain in {"control_provenance", "immutable_cache_binding"}:
            failure_domain = raw_domain
        else:
            failure_domain = (
                "training_memory_resource"
                if resource_failure
                else "training_memory_schedule"
            )
        result = _gate(
            "preflight",
            {"stage_execution": _check(0, "==", 1, False)},
            failure_domain=failure_domain,
            scientific_evidence_complete=False,
            stage_execution_valid=False,
            numerically_valid=False,
            resource_valid=not resource_failure,
        )
        result["evaluation_status"] = "execution_failed"
        result["failure_code"] = str(
            metrics.get("failure_code") or "memory_preflight_execution_failed"
        )
        return result

    flags = PREFLIGHT_PROVENANCE_FLAGS + PREFLIGHT_BINDING_FLAGS + PREFLIGHT_MEMORY_FLAGS
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in flags
    }
    maximum_batch = metrics.get("maximum_observed_model_forward_batch_size")
    peak = metrics.get("peak_memory_fraction")
    checks.update(
        {
            "maximum_observed_model_forward_batch_size": _check(
                maximum_batch,
                "<=",
                t.maximum_model_forward_batch_size,
                isinstance(maximum_batch, int)
                and not isinstance(maximum_batch, bool)
                and 0 < maximum_batch <= t.maximum_model_forward_batch_size,
            ),
            "full_cache_cuda_tensor_count": _check(
                metrics.get("full_cache_cuda_tensor_count"),
                "==",
                0,
                _zero(metrics.get("full_cache_cuda_tensor_count")),
            ),
            "peak_memory_fraction": _check(
                peak,
                "<=",
                t.maximum_peak_memory_fraction,
                _finite(peak) and 0.0 <= float(peak) <= t.maximum_peak_memory_fraction,
            ),
            "synthetic_scale_relative_error": _check(
                metrics.get("synthetic_scale_relative_error"),
                "<=",
                t.synthetic_scale_maximum_relative_error,
                _finite(metrics.get("synthetic_scale_relative_error"))
                and 0.0
                <= float(metrics["synthetic_scale_relative_error"])
                <= t.synthetic_scale_maximum_relative_error,
            ),
        }
    )
    failed = _failed(checks)
    provenance = set(PREFLIGHT_PROVENANCE_FLAGS)
    binding = set(PREFLIGHT_BINDING_FLAGS)
    resource = {"peak_memory_fraction"}
    if failed & provenance:
        domain, complete = "control_provenance", False
    elif failed & binding:
        domain, complete = "immutable_cache_binding", False
    elif failed & resource:
        domain, complete = "training_memory_resource", True
    elif failed:
        domain, complete = "training_memory_schedule", False
    else:
        domain, complete = None, True
    return _gate(
        "preflight",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=complete,
        resource_valid=not bool(failed & resource),
        provenance_valid=int(not bool(failed & provenance)),
        immutable_cache_binding_valid=int(not bool(failed & binding)),
        training_memory_schedule_valid=int(not bool(failed - provenance - binding - resource)),
        thresholds=t.to_dict(),
    )


STAGE_MEMORY_FLAGS = (
    "host_backed_batches_valid",
    "maximum_forward_batch_enforced",
    "full_cache_cuda_tensor_absent",
    "streaming_reducer_valid",
    "label_firewall_valid",
    "memory_diagnostics_complete",
)


def _augment_legacy_gate(
    legacy: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    stage: str,
    thresholds: BoundaryTangentV3MemoryThresholds,
) -> dict[str, Any]:
    result = dict(legacy)
    checks = {
        str(name): dict(record)
        for name, record in dict(legacy.get("checks", {})).items()
    }
    for name in STAGE_MEMORY_FLAGS:
        checks[name] = _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
    maximum_batch = metrics.get("maximum_observed_model_forward_batch_size")
    peak = metrics.get("peak_memory_fraction")
    checks["maximum_observed_model_forward_batch_size"] = _check(
        maximum_batch,
        "<=",
        thresholds.maximum_model_forward_batch_size,
        isinstance(maximum_batch, int)
        and not isinstance(maximum_batch, bool)
        and 0 < maximum_batch <= thresholds.maximum_model_forward_batch_size,
    )
    checks["full_cache_cuda_tensor_count"] = _check(
        metrics.get("full_cache_cuda_tensor_count"),
        "==",
        0,
        _zero(metrics.get("full_cache_cuda_tensor_count")),
    )
    checks["peak_memory_fraction"] = _check(
        peak,
        "<=",
        thresholds.maximum_peak_memory_fraction,
        _finite(peak)
        and 0.0 <= float(peak) <= thresholds.maximum_peak_memory_fraction,
    )
    memory_failed = _failed(checks) - _failed(dict(legacy.get("checks", {})))
    schedule_failed = memory_failed - {"peak_memory_fraction"}
    resource_failed = "peak_memory_fraction" in memory_failed
    result.update(
        {
            "schema": f"{SCHEMA}-{stage}-gate",
            "checks": checks,
            "passed": int(_one(legacy.get("passed")) and not memory_failed),
            "failure_domain": (
                legacy.get("failure_domain")
                if not _one(legacy.get("passed"))
                else "training_memory_resource"
                if resource_failed
                else "training_memory_schedule"
                if schedule_failed
                else None
            ),
            "resource_valid": int(_one(legacy.get("resource_valid", 1)) and not resource_failed),
            "scientific_evidence_complete": int(
                _one(legacy.get("scientific_evidence_complete"))
                and not schedule_failed
            ),
            "training_memory_schedule_valid": int(not schedule_failed),
            "training_memory_resource_valid": int(not resource_failed),
            "memory_thresholds": thresholds.to_dict(),
            "immutable_parent_cache_reused": 1,
            "production_cache_generation_performed": 0,
        }
    )
    return result


def _memory_execution_failure(
    metrics: Mapping[str, Any], *, stage: str
) -> dict[str, Any] | None:
    if metrics.get("evaluation_status") != "execution_failed":
        return None
    code = str(metrics.get("failure_code") or "")
    domain = str(metrics.get("failure_domain") or "")
    if "memory" not in code and domain not in {
        "training_memory_schedule",
        "training_memory_resource",
    }:
        return None
    resource = domain in {
        "memory_resource",
        "training_memory_resource",
        "resource_gate",
    } or "out_of_memory" in code
    result = _gate(
        stage,
        {"stage_execution": _check(0, "==", 1, False)},
        failure_domain=(
            "training_memory_resource" if resource else "training_memory_schedule"
        ),
        scientific_evidence_complete=False,
        stage_execution_valid=False,
        numerically_valid=True,
        resource_valid=not resource,
        training=_one(metrics.get("physical_training_performed")),
        selection=_one(metrics.get("validation_selection_performed")),
        confirmation=_one(metrics.get("confirmation_performed")),
        training_memory_schedule_valid=0,
        training_memory_resource_valid=int(not resource),
    )
    result["evaluation_status"] = "execution_failed"
    result["failure_code"] = code or f"{stage}_memory_execution_failed"
    return result


def evaluate_train_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3MemoryThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3MemoryThresholds()
    failure = _memory_execution_failure(metrics, stage="train")
    if failure is not None:
        return failure
    legacy = _evaluate_v3_train_gate(metrics, thresholds=BoundaryTangentV3Thresholds())
    return _augment_legacy_gate(legacy, metrics, stage="train", thresholds=t)


def evaluate_select_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3MemoryThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3MemoryThresholds()
    failure = _memory_execution_failure(metrics, stage="select")
    if failure is not None:
        return failure
    legacy = _evaluate_v3_select_gate(metrics, thresholds=BoundaryTangentV3Thresholds())
    return _augment_legacy_gate(legacy, metrics, stage="select", thresholds=t)


def evaluate_confirm_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3MemoryThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3MemoryThresholds()
    failure = _memory_execution_failure(metrics, stage="confirm")
    if failure is not None:
        return failure
    legacy = _evaluate_v3_confirm_gate(metrics, thresholds=BoundaryTangentV3Thresholds())
    return _augment_legacy_gate(legacy, metrics, stage="confirm", thresholds=t)


def _status(gate: Mapping[str, Any] | None) -> str:
    return str((gate or {}).get("evaluation_status", "not_evaluated"))


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return _status(gate) == "evaluated" and _one((gate or {}).get("passed"))


def _flag(gate: Mapping[str, Any] | None, name: str) -> bool:
    return isinstance(gate, Mapping) and _one(gate.get(name))


def _decision(
    value: str,
    action: str,
    *,
    evaluation_status: str = "evaluated",
    train_authorized: bool = False,
    select_authorized: bool = False,
    confirm_authorized: bool = False,
    training: bool = False,
    selection: bool = False,
    confirmation: bool = False,
    planning: bool = False,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": evaluation_status,
        "decision": value,
        "recommended_next_action": action,
        "cache_generation_authorized": 0,
        "physical_training_authorized": int(train_authorized),
        "validation_selection_authorized": int(select_authorized),
        "confirmation_authorized": int(confirm_authorized),
        **_scope(
            training=training,
            selection=selection,
            confirmation=confirmation,
            planning=planning,
        ),
    }


def decide_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    select_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if _status(preflight_gate) == "not_evaluated":
        return _decision(
            "ready_for_preflight",
            "verify the immutable cache and batch-32 memory contract",
            evaluation_status="not_evaluated",
        )
    if not _passed(preflight_gate):
        domain = str((preflight_gate or {}).get("failure_domain"))
        if domain == "control_provenance":
            value = BoundaryTangentV3MemoryDecision.CONTROL_PROVENANCE_INVALID
        elif domain == "immutable_cache_binding":
            value = BoundaryTangentV3MemoryDecision.IMMUTABLE_CACHE_BINDING_INVALID
        elif domain == "training_memory_resource":
            value = BoundaryTangentV3MemoryDecision.TRAINING_MEMORY_RESOURCE_INFEASIBLE
        else:
            value = BoundaryTangentV3MemoryDecision.TRAINING_MEMORY_SCHEDULE_INVALID
        return _decision(value.value, "repair only the failed preflight contract")

    if _status(train_gate) == "not_evaluated":
        return _decision(
            "ready_for_train",
            "run streamed controls and physical checkpoint generation",
            evaluation_status="not_evaluated",
            train_authorized=True,
        )
    if not _passed(train_gate):
        domain = str((train_gate or {}).get("failure_domain"))
        if domain == "control_provenance":
            value = BoundaryTangentV3MemoryDecision.CONTROL_PROVENANCE_INVALID
        elif domain == "immutable_cache_binding":
            value = BoundaryTangentV3MemoryDecision.IMMUTABLE_CACHE_BINDING_INVALID
        elif domain == "training_memory_resource":
            value = BoundaryTangentV3MemoryDecision.TRAINING_MEMORY_RESOURCE_INFEASIBLE
        elif domain == "training_memory_schedule":
            value = BoundaryTangentV3MemoryDecision.TRAINING_MEMORY_SCHEDULE_INVALID
        elif domain == "training_controls":
            value = BoundaryTangentV3MemoryDecision.TRAINING_CONTROLS_FAILED
        else:
            value = BoundaryTangentV3MemoryDecision.PHYSICAL_TRAINING_INVALID
        return _decision(
            value.value,
            "retain evidence and repair only the failed training domain",
            training=_flag(train_gate, "physical_training_performed"),
        )

    if _status(select_gate) == "not_evaluated":
        return _decision(
            "ready_for_select",
            "open validation once and run the frozen search-aware family",
            evaluation_status="not_evaluated",
            select_authorized=True,
            training=True,
        )
    if not _passed(select_gate):
        domain = str((select_gate or {}).get("failure_domain"))
        if domain == "control_provenance":
            value = BoundaryTangentV3MemoryDecision.CONTROL_PROVENANCE_INVALID
        elif domain == "immutable_cache_binding":
            value = BoundaryTangentV3MemoryDecision.IMMUTABLE_CACHE_BINDING_INVALID
        else:
            no_candidate = _flag(select_gate, "no_validation_candidate")
            value = (
                BoundaryTangentV3MemoryDecision.NO_VALIDATION_CANDIDATE
                if no_candidate
                else BoundaryTangentV3MemoryDecision.VALIDATION_INFERENCE_INVALID
            )
        return _decision(
            value.value,
            "do not open confirmation",
            training=True,
            selection=_flag(select_gate, "validation_selection_performed"),
        )

    if _status(confirm_gate) == "not_evaluated":
        return _decision(
            "zero_baseline_v3_validation_nominee_sealed",
            "open the single fresh confirmation namespace",
            evaluation_status="not_evaluated",
            confirm_authorized=True,
            training=True,
            selection=True,
        )
    if not _passed(confirm_gate):
        domain = str((confirm_gate or {}).get("failure_domain"))
        if domain == "control_provenance":
            value = BoundaryTangentV3MemoryDecision.CONTROL_PROVENANCE_INVALID
        elif domain == "immutable_cache_binding":
            value = BoundaryTangentV3MemoryDecision.IMMUTABLE_CACHE_BINDING_INVALID
        else:
            valid = _flag(confirm_gate, "fresh_confirmation_valid")
            value = (
                BoundaryTangentV3MemoryDecision.ZERO_BASELINE_V3_SIGNAL_NOT_CONFIRMED
                if valid
                else BoundaryTangentV3MemoryDecision.FRESH_CONFIRMATION_INVALID
            )
        return _decision(
            value.value,
            "retain the sealed audit; no second confirmation is authorized",
            training=True,
            selection=True,
            confirmation=_flag(confirm_gate, "confirmation_performed"),
        )

    return _decision(
        BoundaryTangentV3MemoryDecision.EXACT_RB_ZERO_BASELINE_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED.value,
        "plan a separate controls-only controller-control patch",
        training=True,
        selection=True,
        confirmation=True,
        planning=True,
    )


def evaluate_required_gate(
    *,
    preflight_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    select_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    if require_gate not in REQUIRED_GATES:
        raise BoundaryTangentV3MemoryGateError(
            f"unknown required gate: {require_gate}"
        )
    components = {
        "preflight": dict(preflight_gate or not_evaluated_gate("preflight", "not run")),
        "train": dict(train_gate or not_evaluated_gate("train", "not run")),
        "select": dict(select_gate or not_evaluated_gate("select", "not run")),
        "confirm": dict(confirm_gate or not_evaluated_gate("confirm", "not run")),
    }
    order = ("preflight", "train", "select", "confirm")
    required = () if require_gate == "none" else order[: order.index(require_gate) + 1]
    decision = decide_workflow(
        preflight_gate=components["preflight"],
        train_gate=components["train"],
        select_gate=components["select"],
        confirm_gate=components["confirm"],
    )
    passed = all(_passed(components[name]) for name in required)
    return {
        "schema": f"{SCHEMA}-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_gate_pass": int(passed),
        "required_gate_exit_code": 0 if passed else 1,
        "artifacts_must_be_committed_before_required_gate_exit": 1,
        "components": components,
        "decision": decision,
        "memory_thresholds": BoundaryTangentV3MemoryThresholds().to_dict(),
        "scientific_thresholds": BoundaryTangentV3Thresholds().to_dict(),
        **_scope(
            training=_flag(components["train"], "physical_training_performed"),
            selection=_flag(components["select"], "validation_selection_performed"),
            confirmation=_flag(components["confirm"], "confirmation_performed"),
            planning=_flag(decision, "controller_control_planning_authorized"),
        ),
    }


DECISION_VALUES = tuple(item.value for item in BoundaryTangentV3MemoryDecision)
FINAL_DECISION = (
    BoundaryTangentV3MemoryDecision.EXACT_RB_ZERO_BASELINE_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED.value
)

evaluate_memory_preflight_gate = evaluate_preflight_gate
evaluate_memory_train_gate = evaluate_train_gate
evaluate_memory_select_gate = evaluate_select_gate
evaluate_memory_confirm_gate = evaluate_confirm_gate
evaluate_memory_workflow = evaluate_required_gate


__all__ = [
    "BoundaryTangentV3MemoryDecision",
    "BoundaryTangentV3MemoryGateError",
    "BoundaryTangentV3MemoryThresholds",
    "DECISION_VALUES",
    "FINAL_DECISION",
    "MemoryRecoveryThresholds",
    "REQUIRED_GATES",
    "SCHEMA",
    "decide_workflow",
    "evaluate_confirm_gate",
    "evaluate_memory_confirm_gate",
    "evaluate_memory_preflight_gate",
    "evaluate_memory_select_gate",
    "evaluate_memory_train_gate",
    "evaluate_memory_workflow",
    "evaluate_preflight_gate",
    "evaluate_required_gate",
    "evaluate_select_gate",
    "evaluate_train_gate",
    "not_evaluated_gate",
]
