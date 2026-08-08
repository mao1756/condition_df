"""Fail-closed gates for zero-baseline boundary-tangent v3 learnability.

The workflow searches a fixed 120-checkpoint family on fresh validation
paths and may open one fresh confirmation namespace.  A successful audit
authorizes planning a separate controller-control study only; this module
never authorizes a controller trajectory, reconstruction, or sampling.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-zero-baseline-v3-gate"
SCHEMA_VERSION = 1

TRAIN_ROWS = 114_688
VALIDATION_ROWS = 57_344
CONFIRMATION_ROWS = 114_688
TRAIN_TRANSITIONS = 134_873_088
VALIDATION_TRANSITIONS = 67_436_544
CONFIRMATION_TRANSITIONS = 134_873_088


class BoundaryTangentV3GateError(ValueError):
    """Evidence violates the frozen v3 gate contract."""


class BoundaryTangentV3Decision(str, Enum):
    PROVENANCE_OR_PATH_PLAN_INVALID = "provenance_or_path_plan_invalid"
    ZERO_BASELINE_CONTRACT_INVALID = "zero_baseline_contract_invalid"
    CERTIFICATE_SEMANTICS_COMPARATOR_INVALID = (
        "certificate_semantics_comparator_invalid"
    )
    EXACT_CACHE_INVALID = "exact_cache_invalid"
    TRAINING_CONTROLS_FAILED = "training_controls_failed"
    PHYSICAL_TRAINING_INVALID = "physical_training_invalid"
    VALIDATION_INFERENCE_INVALID = "validation_inference_invalid"
    NO_VALIDATION_CANDIDATE = "no_validation_candidate"
    FRESH_CONFIRMATION_INVALID = "fresh_confirmation_invalid"
    ZERO_BASELINE_V3_SIGNAL_NOT_CONFIRMED = (
        "zero_baseline_v3_signal_not_confirmed"
    )
    EXACT_RB_ZERO_BASELINE_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED = (
        "exact_rb_zero_baseline_boundary_tangent_time_local_signal_confirmed"
    )


@dataclass(frozen=True)
class BoundaryTangentV3Thresholds:
    """Frozen v3 scientific, statistical, and resource configuration."""

    root_seed: int = 261_311
    model_seeds: tuple[int, ...] = (261_312, 261_313, 261_314)
    selection_bootstrap_seed: int = 261_320
    confirmation_bootstrap_seed: int = 261_322
    synthetic_teacher_seed: int = 261_323
    exact_model_null_seed: int = 261_324
    reserved_future_control_seed: int = 261_325
    forbidden_scheduler_benchmark_seed: int = 261_321
    selection_bootstrap_namespace: int = 0x42545633
    confirmation_bootstrap_namespace: int = 0x42544333
    preflight_path_ids: tuple[int, ...] = tuple(range(0xF0000, 0xF0008))
    training_path_ids: tuple[int, ...] = tuple(range(0xF1000, 0xF1040))
    validation_path_ids: tuple[int, ...] = tuple(range(0xF1100, 0xF1120))
    confirmation_path_ids: tuple[int, ...] = tuple(range(0xF2000, 0xF2040))
    burned_v2_confirmation_path_ids: tuple[int, ...] = tuple(
        range(0xED000, 0xED040)
    )
    train_validation_cohort_sizes: tuple[int, ...] = (10,) * 9 + (6,)
    confirmation_cohort_sizes: tuple[int, ...] = (10,) * 6 + (4,)
    training_paths: int = 64
    validation_paths: int = 32
    confirmation_paths: int = 64
    bootstrap_replicates: int = 50_000
    bootstrap_shard_size: int = 1_000
    candidate_block_size: int = 20
    component_block_size: int = 57
    simultaneous_confidence: float = 0.995
    candidate_count: int = 120
    component_count: int = 228
    search_family_size: int = 27_360
    updates_per_seed: int = 40
    checkpoint_count_per_seed: int = 41
    maximum_updates: int = 4_000
    update_interval: int = 100
    synthetic_maximum_relative_validation_mse: float = 0.01
    maximum_mass_error: float = 2.0e-12
    minimum_transitions_per_second: float = 1_300.0
    maximum_peak_memory_fraction: float = 0.80
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_time_fraction: float = 0.10
    maximum_persisted_bytes: int = 5 * 1024**3 // 4
    maximum_projected_seconds: float = 108_000.0
    maximum_launch_lanes: int = 4_096
    train_rows: int = TRAIN_ROWS
    validation_rows: int = VALIDATION_ROWS
    confirmation_rows: int = CONFIRMATION_ROWS
    train_transitions: int = TRAIN_TRANSITIONS
    validation_transitions: int = VALIDATION_TRANSITIONS
    confirmation_transitions: int = CONFIRMATION_TRANSITIONS

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if type(value) is not type(field.default) or value != field.default:
                raise BoundaryTangentV3GateError(
                    f"{name} is frozen at {field.default}"
                )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "model_seeds",
            "preflight_path_ids",
            "training_path_ids",
            "validation_path_ids",
            "confirmation_path_ids",
            "burned_v2_confirmation_path_ids",
            "train_validation_cohort_sizes",
            "confirmation_cohort_sizes",
        ):
            result[name] = list(result[name])
        return result


V3Thresholds = BoundaryTangentV3Thresholds


_ALWAYS_FORBIDDEN = {
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
    cache_performed: bool = False,
    training_performed: bool = False,
    selection_performed: bool = False,
    confirmation_performed: bool = False,
    controller_planning_authorized: bool = False,
) -> dict[str, int]:
    return {
        "production_cache_generation_performed": int(cache_performed),
        "physical_training_performed": int(training_performed),
        "validation_selection_performed": int(selection_performed),
        "confirmation_performed": int(confirmation_performed),
        "controller_control_planning_authorized": int(
            controller_planning_authorized
        ),
        **_ALWAYS_FORBIDDEN,
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


def _at_most(value: Any, threshold: float) -> bool:
    return _finite(value) and 0.0 <= float(value) <= float(threshold)


def _at_least(value: Any, threshold: float) -> bool:
    return _finite(value) and float(value) >= float(threshold)


def _positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


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
        for name, value in checks.items()
        if not _one(value.get("passed"))
    }


def _gate(
    name: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    failure_domain: str | None,
    scientific_evidence_complete: bool,
    stage_execution_valid: bool = True,
    numerically_valid: bool = True,
    resource_valid: bool = True,
    cache_performed: bool = False,
    training_performed: bool = False,
    selection_performed: bool = False,
    confirmation_performed: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in checks.items()}
    passed = bool(normalized) and all(
        _one(record.get("passed")) for record in normalized.values()
    )
    return {
        "schema": f"{SCHEMA}-{name}-gate",
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "evaluation_status": "evaluated",
        "checks": normalized,
        "passed": int(passed),
        "failure_domain": None if passed else failure_domain,
        "stage_execution_valid": int(stage_execution_valid),
        "scientific_evidence_complete": int(scientific_evidence_complete),
        "numerically_valid": int(numerically_valid),
        "resource_valid": int(resource_valid),
        **_scope(
            cache_performed=cache_performed,
            training_performed=training_performed,
            selection_performed=selection_performed,
            confirmation_performed=confirmation_performed,
        ),
        **extra,
    }


def _execution_failed_gate(
    name: str,
    metrics: Mapping[str, Any],
    **scope: bool,
) -> dict[str, Any]:
    result = _gate(
        name,
        {"stage_execution": _check(0, "==", 1, False)},
        failure_domain=str(metrics.get("failure_domain") or "execution"),
        scientific_evidence_complete=False,
        stage_execution_valid=False,
        numerically_valid=False,
        resource_valid=False,
        **scope,
    )
    result["evaluation_status"] = "execution_failed"
    result["failure_code"] = str(
        metrics.get("failure_code") or f"{name}_execution_failed"
    )
    return result


def not_evaluated_gate(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}-{stage}-gate",
        "schema_version": SCHEMA_VERSION,
        "gate": str(stage),
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
    "parent_v2_valid",
    "adjudication_valid",
    "adjudication_authority_valid",
    "complete_parent_registry_valid",
    "complete_adjudication_registry_valid",
    "parent_immutability_valid",
    "source_closure_valid",
    "path_plan_valid",
    "cohort_plan_valid",
    "path_collision_scan_valid",
)
PREFLIGHT_BASELINE_FLAGS = (
    "zero_baseline_contract_valid",
    "baseline_artifacts_absent",
    "state_dict_baseline_free",
    "update_zero_exact",
)
PREFLIGHT_COMPARATOR_FLAGS = (
    "certificate_semantics_comparator_valid",
)
PREFLIGHT_EXECUTION_FLAGS = (
    "preflight_complete",
    "scheduler_seam_valid",
    "exact_kernel_contract_valid",
    "cuda_determinism_valid",
    "source_image_binding_valid",
    "selected_step_contract_valid",
    "model_input_firewall_valid",
    "raw_target_contract_valid",
    "train_validation_confirmation_unopened",
    "inherited_resource_projection_valid",
)


def evaluate_preflight_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3Thresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3Thresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate("preflight", metrics)
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in (
            PREFLIGHT_PROVENANCE_FLAGS
            + PREFLIGHT_BASELINE_FLAGS
            + PREFLIGHT_COMPARATOR_FLAGS
            + PREFLIGHT_EXECUTION_FLAGS
        )
    }
    checks.update(
        {
            "preflight_path_ids": _check(
                metrics.get("preflight_path_ids"),
                "==",
                list(t.preflight_path_ids),
                tuple(metrics.get("preflight_path_ids", ()))
                == t.preflight_path_ids,
            ),
            "preflight_path_count": _check(
                metrics.get("preflight_path_count"),
                "==",
                len(t.preflight_path_ids),
                metrics.get("preflight_path_count") == len(t.preflight_path_ids),
            ),
            "certificate_fraction": _check(
                metrics.get("certificate_fraction"),
                "==",
                1.0,
                _finite(metrics.get("certificate_fraction"))
                and float(metrics["certificate_fraction"]) == 1.0,
            ),
            "maximum_mass_error": _check(
                metrics.get("maximum_mass_error"),
                "<=",
                t.maximum_mass_error,
                _at_most(metrics.get("maximum_mass_error"), t.maximum_mass_error),
            ),
            "forbidden_event_count": _check(
                metrics.get("forbidden_event_count"),
                "==",
                0,
                _zero(metrics.get("forbidden_event_count")),
            ),
            "transitions_per_second": _check(
                metrics.get("transitions_per_second"),
                ">=",
                t.minimum_transitions_per_second,
                _at_least(
                    metrics.get("transitions_per_second"),
                    t.minimum_transitions_per_second,
                ),
            ),
            "peak_memory_fraction": _check(
                metrics.get("peak_memory_fraction"),
                "<=",
                t.maximum_peak_memory_fraction,
                _at_most(
                    metrics.get("peak_memory_fraction"),
                    t.maximum_peak_memory_fraction,
                ),
            ),
        }
    )
    failed = _failed(checks)
    provenance = set(PREFLIGHT_PROVENANCE_FLAGS)
    baseline = set(PREFLIGHT_BASELINE_FLAGS)
    comparator = set(PREFLIGHT_COMPARATOR_FLAGS)
    numerical = {
        "scheduler_seam_valid",
        "certificate_fraction",
        "maximum_mass_error",
        "forbidden_event_count",
    }
    resource = {"transitions_per_second", "peak_memory_fraction"}
    execution = failed - provenance - baseline - comparator - numerical - resource
    if failed & provenance:
        domain, complete = "provenance_or_path_plan", False
    elif failed & baseline:
        domain, complete = "zero_baseline_contract", False
    elif failed & numerical:
        domain, complete = "numerical", True
    elif failed & resource:
        domain, complete = "resource_gate", True
    elif failed & comparator:
        domain, complete = "implementation_contract", True
    elif failed:
        domain, complete = "execution", False
    else:
        domain, complete = None, True
    return _gate(
        "preflight",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=complete,
        stage_execution_valid=not bool(execution),
        numerically_valid=not bool(failed & numerical),
        resource_valid=not bool(failed & resource),
        provenance_or_path_plan_valid=int(not bool(failed & provenance)),
        zero_baseline_contract_valid=int(not bool(failed & baseline)),
        certificate_semantics_comparator_valid=int(
            not bool(failed & comparator)
        ),
        certificate_semantics_comparator_failure=int(
            domain == "implementation_contract"
        ),
        thresholds=t.to_dict(),
    )


CACHE_FLAGS = (
    "cache_complete",
    "train_cache_complete",
    "validation_cache_complete",
    "atomic_shard_chains_valid",
    "resume_replay_valid",
    "selected_sample_cartesian_valid",
    "train_validation_indexes_disjoint",
    "artifact_role_isolation_valid",
    "mixed_cohort_split_before_commit",
    "raw_target_contract_valid",
    "input_field_contract_valid",
    "confirmation_absent",
    "confirmation_namespace_unopened",
    "baseline_artifacts_absent",
)


def evaluate_cache_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3Thresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3Thresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate(
            "cache",
            metrics,
            cache_performed=_one(metrics.get("production_cache_generation_performed")),
        )
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in CACHE_FLAGS
    }
    exact = {
        "train_path_count": t.training_paths,
        "validation_path_count": t.validation_paths,
        "train_row_count": t.train_rows,
        "validation_row_count": t.validation_rows,
        "train_transition_count": t.train_transitions,
        "validation_transition_count": t.validation_transitions,
    }
    checks.update(
        {
            name: _check(metrics.get(name), "==", value, metrics.get(name) == value)
            for name, value in exact.items()
        }
    )
    checks.update(
        {
            "certificate_fraction": _check(
                metrics.get("certificate_fraction"),
                "==",
                1.0,
                _finite(metrics.get("certificate_fraction"))
                and float(metrics["certificate_fraction"]) == 1.0,
            ),
            "maximum_mass_error": _check(
                metrics.get("maximum_mass_error"),
                "<=",
                t.maximum_mass_error,
                _at_most(metrics.get("maximum_mass_error"), t.maximum_mass_error),
            ),
            "forbidden_event_count": _check(
                metrics.get("forbidden_event_count"),
                "==",
                0,
                _zero(metrics.get("forbidden_event_count")),
            ),
            "minimum_role_rate": _check(
                metrics.get("minimum_role_rate"),
                ">=",
                t.minimum_transitions_per_second,
                _at_least(metrics.get("minimum_role_rate"), t.minimum_transitions_per_second),
            ),
            "fallback_fraction": _check(
                metrics.get("fallback_fraction"),
                "<=",
                t.maximum_fallback_fraction,
                _at_most(metrics.get("fallback_fraction"), t.maximum_fallback_fraction),
            ),
            "fallback_time_fraction": _check(
                metrics.get("fallback_time_fraction"),
                "<=",
                t.maximum_fallback_time_fraction,
                _at_most(metrics.get("fallback_time_fraction"), t.maximum_fallback_time_fraction),
            ),
            "peak_memory_fraction": _check(
                metrics.get("peak_memory_fraction"),
                "<=",
                t.maximum_peak_memory_fraction,
                _at_most(metrics.get("peak_memory_fraction"), t.maximum_peak_memory_fraction),
            ),
            "total_persisted_cache_bytes": _check(
                metrics.get("total_persisted_cache_bytes"),
                "<=",
                t.maximum_persisted_bytes,
                isinstance(metrics.get("total_persisted_cache_bytes"), int)
                and not isinstance(metrics.get("total_persisted_cache_bytes"), bool)
                and 0 <= int(metrics["total_persisted_cache_bytes"]) <= t.maximum_persisted_bytes,
            ),
            "projected_cache_plus_confirmation_seconds": _check(
                metrics.get("projected_cache_plus_confirmation_seconds"),
                "<=",
                t.maximum_projected_seconds,
                _at_most(
                    metrics.get("projected_cache_plus_confirmation_seconds"),
                    t.maximum_projected_seconds,
                ),
            ),
            "production_cache_generation_performed": _check(
                metrics.get("production_cache_generation_performed"),
                "==",
                1,
                _one(metrics.get("production_cache_generation_performed")),
            ),
            "physical_training_performed": _check(
                metrics.get("physical_training_performed"),
                "==",
                0,
                _zero(metrics.get("physical_training_performed")),
            ),
            "confirmation_performed": _check(
                metrics.get("confirmation_performed"),
                "==",
                0,
                _zero(metrics.get("confirmation_performed")),
            ),
        }
    )
    failed = _failed(checks)
    numerical = {"certificate_fraction", "maximum_mass_error", "forbidden_event_count"}
    resource = {
        "minimum_role_rate",
        "fallback_fraction",
        "fallback_time_fraction",
        "peak_memory_fraction",
        "total_persisted_cache_bytes",
        "projected_cache_plus_confirmation_seconds",
    }
    return _gate(
        "cache",
        checks,
        failure_domain=None if not failed else "exact_cache",
        scientific_evidence_complete=not bool(failed - numerical - resource),
        numerically_valid=not bool(failed & numerical),
        resource_valid=not bool(failed & resource),
        cache_performed=True,
        thresholds=t.to_dict(),
    )


TRAIN_CONTROL_FLAGS = (
    "zero_initialization_control_passed",
    "synthetic_teacher_passed",
    "synthetic_every_validation_path_beats_zero",
    "exact_model_null_passed",
    "null_selected_update_zero",
    "null_parameters_bitwise_unchanged",
    "controls_before_training_label_open",
)
TRAIN_PHYSICAL_FLAGS = (
    "physical_training_complete",
    "all_physical_tasks_complete_finite",
    "training_labels_opened_after_controls",
    "validation_labels_opened_zero",
    "validation_inputs_unavailable_to_physical_trainer",
    "fixed_checkpoint_grid_complete",
    "candidate_grid_valid",
    "pointwise_checkpoint_selection_performed_zero",
    "physical_task_records_selection_free",
    "training_only_target_scale_valid",
    "baseline_artifacts_absent",
    "confirmation_absent",
)


def evaluate_train_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3Thresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3Thresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate(
            "train",
            metrics,
            cache_performed=True,
            training_performed=_one(metrics.get("physical_training_performed")),
        )
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in TRAIN_CONTROL_FLAGS + TRAIN_PHYSICAL_FLAGS
    }
    checks.update(
        {
            "synthetic_relative_validation_mse": _check(
                metrics.get("synthetic_relative_validation_mse"),
                "<=",
                t.synthetic_maximum_relative_validation_mse,
                _at_most(
                    metrics.get("synthetic_relative_validation_mse"),
                    t.synthetic_maximum_relative_validation_mse,
                ),
            ),
            "model_seed_count": _check(
                metrics.get("model_seed_count"),
                "==",
                len(t.model_seeds),
                metrics.get("model_seed_count") == len(t.model_seeds),
            ),
            "checkpoint_count": _check(
                metrics.get("checkpoint_count"),
                "==",
                len(t.model_seeds) * t.checkpoint_count_per_seed,
                metrics.get("checkpoint_count")
                == len(t.model_seeds) * t.checkpoint_count_per_seed,
            ),
            "nonzero_candidate_count": _check(
                metrics.get("nonzero_candidate_count"),
                "==",
                t.candidate_count,
                metrics.get("nonzero_candidate_count") == t.candidate_count,
            ),
            "maximum_updates": _check(
                metrics.get("maximum_updates"),
                "==",
                t.maximum_updates,
                metrics.get("maximum_updates") == t.maximum_updates,
            ),
            "physical_training_performed": _check(
                metrics.get("physical_training_performed"),
                "==",
                1,
                _one(metrics.get("physical_training_performed")),
            ),
            "validation_labels_opened": _check(
                metrics.get("validation_labels_opened"),
                "==",
                0,
                _zero(metrics.get("validation_labels_opened")),
            ),
            "pointwise_checkpoint_selection_performed": _check(
                metrics.get("pointwise_checkpoint_selection_performed"),
                "==",
                0,
                _zero(metrics.get("pointwise_checkpoint_selection_performed")),
            ),
        }
    )
    failed = _failed(checks)
    control_checks = set(TRAIN_CONTROL_FLAGS) | {
        "synthetic_relative_validation_mse"
    }
    controls_failed = bool(failed & control_checks)
    return _gate(
        "train",
        checks,
        failure_domain=(
            None
            if not failed
            else "training_controls" if controls_failed else "physical_training"
        ),
        scientific_evidence_complete=not bool(failed),
        cache_performed=True,
        training_performed=True,
        training_controls_valid=int(not controls_failed),
        physical_training_valid=int(not bool(failed - control_checks)),
        thresholds=t.to_dict(),
    )


SELECT_FLAGS = (
    "selection_complete",
    "train_stage_seal_valid",
    "search_plan_committed_before_validation_labels",
    "bootstrap_counts_committed_before_validation_labels",
    "validation_labels_opened_once",
    "update_zero_control_valid",
    "all_candidate_commits_valid",
    "candidate_table_valid",
    "family_names_and_order_valid",
    "whole_path_shared_counts_valid",
    "studentization_valid",
    "bootstrap_restart_evidence_valid",
    "quantile_rule_valid",
    "confirmation_absent",
)


def evaluate_select_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3Thresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3Thresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate(
            "select",
            metrics,
            cache_performed=True,
            training_performed=True,
            selection_performed=_one(metrics.get("validation_selection_performed")),
        )
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in SELECT_FLAGS
    }
    exact = {
        "path_count": t.validation_paths,
        "candidate_count": t.candidate_count,
        "component_count": t.component_count,
        "search_family_size": t.search_family_size,
        "bootstrap_replicates": t.bootstrap_replicates,
        "bootstrap_shard_count": t.bootstrap_replicates // t.bootstrap_shard_size,
    }
    checks.update(
        {
            name: _check(metrics.get(name), "==", value, metrics.get(name) == value)
            for name, value in exact.items()
        }
    )
    eligible = metrics.get("eligible_candidate_count")
    selected_nonzero = _one(metrics.get("selected_nonzero"))
    no_candidate = (
        isinstance(eligible, int)
        and not isinstance(eligible, bool)
        and eligible == 0
        and not selected_nonzero
        and _one(metrics.get("logical_update_zero_selected"))
    )
    checks["eligible_candidate_count"] = _check(
        eligible,
        ">=",
        1,
        isinstance(eligible, int)
        and not isinstance(eligible, bool)
        and eligible >= 1,
    )
    checks["selected_nonzero"] = _check(
        metrics.get("selected_nonzero"), "==", 1, selected_nonzero
    )
    checks["selected_minimum_lower_bound"] = _check(
        metrics.get("selected_minimum_lower_bound"),
        ">",
        0.0,
        selected_nonzero and _positive(metrics.get("selected_minimum_lower_bound")),
    )
    failed = _failed(checks)
    inference_names = set(SELECT_FLAGS) | set(exact)
    inference_invalid = bool(failed & inference_names)
    if no_candidate and not inference_invalid:
        domain, complete = "no_validation_candidate", True
    elif failed:
        domain, complete = "validation_inference", False
    else:
        domain, complete = None, True
    return _gate(
        "select",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=complete,
        cache_performed=True,
        training_performed=True,
        selection_performed=True,
        validation_inference_valid=int(not inference_invalid),
        no_validation_candidate=int(no_candidate),
        validation_nominee_sealed=int(not failed),
        thresholds=t.to_dict(),
    )


CONFIRM_FLAGS = (
    "confirmation_complete",
    "confirmation_namespace_opened_once",
    "selection_sealed_before_namespace_open",
    "bootstrap_counts_committed_before_paths",
    "same_228_family_valid",
    "whole_path_shared_counts_valid",
    "studentization_valid",
    "atomic_shard_chains_valid",
    "resume_replay_valid",
    "raw_confirmation_inputs_not_persisted",
    "raw_confirmation_labels_not_persisted",
)


def evaluate_confirm_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentV3Thresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentV3Thresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate(
            "confirm",
            metrics,
            cache_performed=True,
            training_performed=True,
            selection_performed=True,
            confirmation_performed=_one(metrics.get("confirmation_performed")),
        )
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in CONFIRM_FLAGS
    }
    exact = {
        "confirmation_path_count": t.confirmation_paths,
        "confirmation_row_count": t.confirmation_rows,
        "confirmation_transition_count": t.confirmation_transitions,
        "component_count": t.component_count,
        "bootstrap_replicates": t.bootstrap_replicates,
    }
    checks.update(
        {
            name: _check(metrics.get(name), "==", value, metrics.get(name) == value)
            for name, value in exact.items()
        }
    )
    checks.update(
        {
            "minimum_lower_bound": _check(
                metrics.get("minimum_lower_bound"),
                ">",
                0.0,
                _positive(metrics.get("minimum_lower_bound")),
            ),
            "all_lower_bounds_strictly_positive": _check(
                metrics.get("all_lower_bounds_strictly_positive"),
                "==",
                1,
                _one(metrics.get("all_lower_bounds_strictly_positive")),
            ),
            "certificate_fraction": _check(
                metrics.get("certificate_fraction"),
                "==",
                1.0,
                _finite(metrics.get("certificate_fraction"))
                and float(metrics["certificate_fraction"]) == 1.0,
            ),
            "maximum_mass_error": _check(
                metrics.get("maximum_mass_error"),
                "<=",
                t.maximum_mass_error,
                _at_most(metrics.get("maximum_mass_error"), t.maximum_mass_error),
            ),
            "forbidden_event_count": _check(
                metrics.get("forbidden_event_count"),
                "==",
                0,
                _zero(metrics.get("forbidden_event_count")),
            ),
            "confirmation_transitions_per_second": _check(
                metrics.get("confirmation_transitions_per_second"),
                ">=",
                t.minimum_transitions_per_second,
                _at_least(
                    metrics.get("confirmation_transitions_per_second"),
                    t.minimum_transitions_per_second,
                ),
            ),
            "fallback_fraction": _check(
                metrics.get("fallback_fraction"),
                "<=",
                t.maximum_fallback_fraction,
                _at_most(
                    metrics.get("fallback_fraction"), t.maximum_fallback_fraction
                ),
            ),
            "fallback_time_fraction": _check(
                metrics.get("fallback_time_fraction"),
                "<=",
                t.maximum_fallback_time_fraction,
                _at_most(
                    metrics.get("fallback_time_fraction"),
                    t.maximum_fallback_time_fraction,
                ),
            ),
            "peak_memory_fraction": _check(
                metrics.get("peak_memory_fraction"),
                "<=",
                t.maximum_peak_memory_fraction,
                _at_most(
                    metrics.get("peak_memory_fraction"),
                    t.maximum_peak_memory_fraction,
                ),
            ),
            "actual_cache_plus_confirmation_seconds": _check(
                metrics.get("actual_cache_plus_confirmation_seconds"),
                "<=",
                t.maximum_projected_seconds,
                _at_most(
                    metrics.get("actual_cache_plus_confirmation_seconds"),
                    t.maximum_projected_seconds,
                ),
            ),
            "confirmation_performed": _check(
                metrics.get("confirmation_performed"),
                "==",
                1,
                _one(metrics.get("confirmation_performed")),
            ),
        }
    )
    failed = _failed(checks)
    signal = {"minimum_lower_bound", "all_lower_bounds_strictly_positive"}
    resource = {
        "confirmation_transitions_per_second",
        "fallback_fraction",
        "fallback_time_fraction",
        "peak_memory_fraction",
        "actual_cache_plus_confirmation_seconds",
    }
    inference_invalid = bool(failed - signal - resource)
    signal_not_confirmed = bool(failed & signal) and not inference_invalid
    return _gate(
        "confirm",
        checks,
        failure_domain=(
            None
            if not failed
            else "signal_not_confirmed" if signal_not_confirmed else "fresh_confirmation"
        ),
        scientific_evidence_complete=not inference_invalid,
        numerically_valid=not bool(
            failed & {"certificate_fraction", "maximum_mass_error", "forbidden_event_count"}
        ),
        resource_valid=not bool(failed & resource),
        cache_performed=True,
        training_performed=True,
        selection_performed=True,
        confirmation_performed=True,
        fresh_confirmation_valid=int(not inference_invalid),
        signal_confirmed=int(not failed),
        thresholds=t.to_dict(),
    )


def _status(gate: Mapping[str, Any] | None) -> str:
    return str((gate or {}).get("evaluation_status", "not_evaluated"))


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return _status(gate) == "evaluated" and _one((gate or {}).get("passed"))


def _flag(gate: Mapping[str, Any] | None, name: str) -> bool:
    return isinstance(gate, Mapping) and _one(gate.get(name))


def _decision_record(
    decision: str,
    action: str,
    *,
    evaluation_status: str = "evaluated",
    cache_authorized: bool = False,
    training_authorized: bool = False,
    selection_authorized: bool = False,
    confirmation_authorized: bool = False,
    controller_planning_authorized: bool = False,
    cache_performed: bool = False,
    training_performed: bool = False,
    selection_performed: bool = False,
    confirmation_performed: bool = False,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": evaluation_status,
        "decision": decision,
        "recommended_next_action": action,
        "cache_generation_authorized": int(cache_authorized),
        "physical_training_authorized": int(training_authorized),
        "validation_selection_authorized": int(selection_authorized),
        "confirmation_authorized": int(confirmation_authorized),
        **_scope(
            cache_performed=cache_performed,
            training_performed=training_performed,
            selection_performed=selection_performed,
            confirmation_performed=confirmation_performed,
            controller_planning_authorized=controller_planning_authorized,
        ),
    }


def decide_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    select_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the frozen terminal precedence and intermediate authority rules."""

    cache_done = _flag(cache_gate, "production_cache_generation_performed")
    train_done = _flag(train_gate, "physical_training_performed")
    select_done = _flag(select_gate, "validation_selection_performed")
    confirm_done = _flag(confirm_gate, "confirmation_performed")
    if _status(preflight_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_preflight",
            "verify immutable evidence and the fresh zero-baseline v3 design",
            evaluation_status="not_evaluated",
        )
    if not _passed(preflight_gate):
        if not _flag(preflight_gate, "provenance_or_path_plan_valid"):
            decision = BoundaryTangentV3Decision.PROVENANCE_OR_PATH_PLAN_INVALID
            action = "repair immutable evidence or the frozen path/cohort allocation"
        elif not _flag(preflight_gate, "zero_baseline_contract_valid"):
            decision = BoundaryTangentV3Decision.ZERO_BASELINE_CONTRACT_INVALID
            action = "repair the exact zero-baseline representation"
        elif _flag(
            preflight_gate, "certificate_semantics_comparator_failure"
        ):
            decision = (
                BoundaryTangentV3Decision.CERTIFICATE_SEMANTICS_COMPARATOR_INVALID
            )
            action = (
                "repair the adaptive/eager certificate-semantics comparator"
            )
        else:
            decision = BoundaryTangentV3Decision.EXACT_CACHE_INVALID
            action = "repair the exact preflight seam execution"
        return _decision_record(decision.value, action)
    if _status(cache_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_cache",
            "generate fresh exact training and validation evidence",
            evaluation_status="not_evaluated",
            cache_authorized=True,
        )
    if not _passed(cache_gate):
        return _decision_record(
            BoundaryTangentV3Decision.EXACT_CACHE_INVALID.value,
            "repair only the exact cache execution or evidence contract",
            cache_performed=cache_done,
        )
    if _status(train_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_train",
            "run controls and generate the fixed physical checkpoint grid",
            evaluation_status="not_evaluated",
            training_authorized=True,
            cache_performed=True,
        )
    if not _passed(train_gate):
        controls_valid = _flag(train_gate, "training_controls_valid")
        decision = (
            BoundaryTangentV3Decision.PHYSICAL_TRAINING_INVALID
            if controls_valid
            else BoundaryTangentV3Decision.TRAINING_CONTROLS_FAILED
        )
        return _decision_record(
            decision.value,
            "repair controls" if not controls_valid else "repair fixed-grid training",
            cache_performed=True,
            training_performed=train_done,
        )
    if _status(select_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_select",
            "open validation once and run the complete search-aware max-T family",
            evaluation_status="not_evaluated",
            selection_authorized=True,
            cache_performed=True,
            training_performed=True,
        )
    if not _passed(select_gate):
        no_candidate = _flag(select_gate, "no_validation_candidate")
        decision = (
            BoundaryTangentV3Decision.NO_VALIDATION_CANDIDATE
            if no_candidate and _flag(select_gate, "validation_inference_valid")
            else BoundaryTangentV3Decision.VALIDATION_INFERENCE_INVALID
        )
        return _decision_record(
            decision.value,
            (
                "do not open confirmation; no searched nonzero checkpoint resolved"
                if no_candidate
                else "repair search-aware validation inference"
            ),
            cache_performed=True,
            training_performed=True,
            selection_performed=select_done,
        )
    if _status(confirm_gate) == "not_evaluated":
        return _decision_record(
            "zero_baseline_v3_validation_nominee_sealed",
            "open the one fresh confirmation namespace",
            evaluation_status="not_evaluated",
            confirmation_authorized=True,
            cache_performed=True,
            training_performed=True,
            selection_performed=True,
        )
    if not _passed(confirm_gate):
        valid = _flag(confirm_gate, "fresh_confirmation_valid")
        decision = (
            BoundaryTangentV3Decision.ZERO_BASELINE_V3_SIGNAL_NOT_CONFIRMED
            if valid
            else BoundaryTangentV3Decision.FRESH_CONFIRMATION_INVALID
        )
        return _decision_record(
            decision.value,
            "retain the sealed audit; no second confirmation is authorized",
            cache_performed=True,
            training_performed=True,
            selection_performed=True,
            confirmation_performed=confirm_done,
        )
    decision = (
        BoundaryTangentV3Decision.EXACT_RB_ZERO_BASELINE_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED
    )
    return _decision_record(
        decision.value,
        "plan a separate controls-only at-most-eight-phase controller-control patch",
        controller_planning_authorized=True,
        cache_performed=True,
        training_performed=True,
        selection_performed=True,
        confirmation_performed=True,
    )


REQUIRED_GATES = ("none", "preflight", "cache", "train", "select", "confirm")


def evaluate_required_gate(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    select_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    if require_gate not in REQUIRED_GATES:
        raise BoundaryTangentV3GateError(f"unknown required gate: {require_gate}")
    components = {
        "preflight": dict(preflight_gate or not_evaluated_gate("preflight", "not run")),
        "cache": dict(cache_gate or not_evaluated_gate("cache", "not run")),
        "train": dict(train_gate or not_evaluated_gate("train", "not run")),
        "select": dict(select_gate or not_evaluated_gate("select", "not run")),
        "confirm": dict(confirm_gate or not_evaluated_gate("confirm", "not run")),
    }
    order = ("preflight", "cache", "train", "select", "confirm")
    required = () if require_gate == "none" else order[: order.index(require_gate) + 1]
    decision = decide_workflow(
        preflight_gate=components["preflight"],
        cache_gate=components["cache"],
        train_gate=components["train"],
        select_gate=components["select"],
        confirm_gate=components["confirm"],
    )
    required_pass = all(_passed(components[name]) for name in required)
    return {
        "schema": f"{SCHEMA}-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "required_gate_exit_code": 0 if required_pass else 1,
        "artifacts_must_be_committed_before_required_gate_exit": 1,
        "components": components,
        "decision": decision,
        "thresholds": BoundaryTangentV3Thresholds().to_dict(),
        **_scope(
            cache_performed=_flag(components["cache"], "production_cache_generation_performed"),
            training_performed=_flag(components["train"], "physical_training_performed"),
            selection_performed=_flag(components["select"], "validation_selection_performed"),
            confirmation_performed=_flag(components["confirm"], "confirmation_performed"),
            controller_planning_authorized=_flag(
                decision, "controller_control_planning_authorized"
            ),
        ),
    }


evaluate_boundary_tangent_v3_preflight = evaluate_preflight_gate
evaluate_boundary_tangent_v3_cache = evaluate_cache_gate
evaluate_boundary_tangent_v3_train = evaluate_train_gate
evaluate_boundary_tangent_v3_select = evaluate_select_gate
evaluate_boundary_tangent_v3_confirm = evaluate_confirm_gate
evaluate_boundary_tangent_v3_workflow = evaluate_required_gate
decide_boundary_tangent_v3_workflow = decide_workflow

DECISION_VALUES = tuple(item.value for item in BoundaryTangentV3Decision)
FINAL_DECISION = (
    BoundaryTangentV3Decision.EXACT_RB_ZERO_BASELINE_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED.value
)


__all__ = [
    "BoundaryTangentV3Decision",
    "BoundaryTangentV3GateError",
    "BoundaryTangentV3Thresholds",
    "DECISION_VALUES",
    "FINAL_DECISION",
    "REQUIRED_GATES",
    "SCHEMA",
    "V3Thresholds",
    "decide_boundary_tangent_v3_workflow",
    "decide_workflow",
    "evaluate_boundary_tangent_v3_cache",
    "evaluate_boundary_tangent_v3_confirm",
    "evaluate_boundary_tangent_v3_preflight",
    "evaluate_boundary_tangent_v3_select",
    "evaluate_boundary_tangent_v3_train",
    "evaluate_boundary_tangent_v3_workflow",
    "evaluate_cache_gate",
    "evaluate_confirm_gate",
    "evaluate_preflight_gate",
    "evaluate_required_gate",
    "evaluate_select_gate",
    "evaluate_train_gate",
    "not_evaluated_gate",
]
