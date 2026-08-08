"""Fail-closed gates for the eager-prefix boundary-tangent v2 workflow.

The workflow ends after fresh cache generation, bounded one-image training,
and sealed time-local confirmation.  A successful confirmation authorizes
only planning a separate controller-control experiment; this module never
authorizes a controller trajectory, reconstruction, or sampling.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    BoundaryTangentThresholds,
    evaluate_confirmation_gate,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-eager-v2-gate"
SCHEMA_VERSION = 1

TRAIN_ROWS = 114_688
VALIDATION_ROWS = 57_344
CONFIRMATION_ROWS = 114_688
TRAIN_TRANSITIONS = 134_873_088
VALIDATION_TRANSITIONS = 67_436_544
CONFIRMATION_TRANSITIONS = 134_873_088
TOTAL_TRANSITIONS = 337_182_720
PROJECTED_BASE_TRANSITIONS = 224_788_480
PROJECTED_MIDPOINT_TRANSITIONS = 112_394_240


class BoundaryTangentEagerGateError(ValueError):
    """Evidence violates the frozen eager v2 gate contract."""


class BoundaryTangentEagerDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    LEGACY_BOUNDARY_TANGENT_ADJUDICATION_INVALID = (
        "legacy_boundary_tangent_adjudication_invalid"
    )
    EAGER_SCHEDULE_INTEGRATION_INVALID = (
        "eager_schedule_integration_invalid"
    )
    BOUNDARY_TANGENT_REPRESENTATION_INVALID = (
        "boundary_tangent_representation_invalid"
    )
    BOUNDARY_TANGENT_DESIGN_INFEASIBLE = (
        "boundary_tangent_design_infeasible"
    )
    FRESH_EXACT_CACHE_INVALID = "fresh_exact_cache_invalid"
    BOUNDARY_TANGENT_CACHE_RESOURCE_INFEASIBLE = (
        "boundary_tangent_cache_resource_infeasible"
    )
    BOUNDARY_TANGENT_BASELINE_INVALID = "boundary_tangent_baseline_invalid"
    BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID = (
        "boundary_tangent_optimization_pipeline_invalid"
    )
    BOUNDARY_TANGENT_BASELINE_ONLY_SIGNAL = (
        "boundary_tangent_baseline_only_signal"
    )
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_NOT_DETECTED = (
        "boundary_tangent_time_local_signal_not_detected"
    )
    BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE = (
        "boundary_tangent_audit_inconclusive"
    )
    PAIRED_RISK_INFERENCE_INVALID = "paired_risk_inference_invalid"
    EXACT_RB_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED = (
        "exact_rb_boundary_tangent_time_local_signal_confirmed"
    )


@dataclass(frozen=True)
class BoundaryTangentEagerThresholds:
    """Frozen v2 scientific, statistical, and resource thresholds."""

    eager_parent_record_count: int = 615
    controller_v1_parent_record_count: int = 14
    root_seed: int = 261_311
    model_seeds: tuple[int, ...] = (261_312, 261_313, 261_314)
    bootstrap_seed: int = 261_315
    reserved_control_seed: int = 261_316
    synthetic_teacher_seed: int = 261_317
    baseline_null_seed: int = 261_318
    training_path_ids: tuple[int, ...] = tuple(range(0xEC100, 0xEC140))
    validation_path_ids: tuple[int, ...] = tuple(range(0xEC200, 0xEC220))
    confirmation_path_ids: tuple[int, ...] = tuple(range(0xED000, 0xED040))
    preflight_seam_path_ids: tuple[int, ...] = tuple(
        range(0xEF000, 0xEF008)
    )
    forbidden_historical_v1_path_ids: tuple[int, ...] = tuple(
        range(0xEC000, 0xEC008)
    )
    train_validation_cohort_sizes: tuple[int, ...] = (10,) * 9 + (6,)
    confirmation_cohort_sizes: tuple[int, ...] = (10,) * 6 + (4,)
    training_paths: int = 64
    validation_paths: int = 32
    confirmation_paths: int = 64
    bootstrap_replicates: int = 50_000
    simultaneous_confidence: float = 0.995
    synthetic_maximum_relative_validation_mse: float = 0.01
    maximum_weak_law_bias: float = 0.10
    maximum_microstep_refinement_error: float = 0.05
    maximum_mass_error: float = 2.0e-12
    minimum_transitions_per_second: float = 1_300.0
    maximum_peak_memory_fraction: float = 0.80
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_time_fraction: float = 0.10
    maximum_persisted_bytes: int = 5 * 1024**3 // 4
    maximum_projected_seconds: float = 108_000.0
    minimum_projected_effective_rate: float = TOTAL_TRANSITIONS / 108_000.0
    maximum_launch_lanes: int = 4_096
    candidate_modes: int = 128
    maximum_updates: int = 4_000
    train_rows: int = TRAIN_ROWS
    validation_rows: int = VALIDATION_ROWS
    confirmation_rows: int = CONFIRMATION_ROWS
    train_transitions: int = TRAIN_TRANSITIONS
    validation_transitions: int = VALIDATION_TRANSITIONS
    confirmation_transitions: int = CONFIRMATION_TRANSITIONS
    total_transitions: int = TOTAL_TRANSITIONS
    projected_base_transitions: int = PROJECTED_BASE_TRANSITIONS
    projected_midpoint_transitions: int = PROJECTED_MIDPOINT_TRANSITIONS

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            if type(value) is not type(field.default) or value != field.default:
                raise BoundaryTangentEagerGateError(
                    f"{name} is frozen at {field.default}"
                )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model_seeds"] = list(self.model_seeds)
        return result

    def confirmation_thresholds(self) -> BoundaryTangentThresholds:
        return BoundaryTangentThresholds(
            confirmation_paths=self.confirmation_paths,
            controller_paths=self.confirmation_paths,
            bootstrap_replicates=self.bootstrap_replicates,
            simultaneous_confidence=self.simultaneous_confidence,
            bootstrap_seed=self.bootstrap_seed,
            controller_bootstrap_seed=self.reserved_control_seed,
            maximum_weak_law_bias=self.maximum_weak_law_bias,
            maximum_microstep_refinement_error=(
                self.maximum_microstep_refinement_error
            ),
            maximum_mass_error=self.maximum_mass_error,
            minimum_transitions_per_second=self.minimum_transitions_per_second,
            maximum_peak_memory_fraction=self.maximum_peak_memory_fraction,
            maximum_fallback_fraction=self.maximum_fallback_fraction,
            maximum_fallback_time_fraction=self.maximum_fallback_time_fraction,
            maximum_persisted_bytes=self.maximum_persisted_bytes,
        )


_ALWAYS_FORBIDDEN = {
    "controller_control_trajectory_authorized": 0,
    "controller_control_trajectory_performed": 0,
    "full_reverse_path_performed": 0,
    "reverse_sampling_authorized": 0,
    "reverse_sampling_performed": 0,
    "sampling_authorized": 0,
    "sampling_performed": 0,
    "image_sampling_performed": 0,
    "reconstruction_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "reconstruction_performed": 0,
    "known_prior_claim_authorized": 0,
    "unsplit_generator_claim_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
    "one_image_reconstruction_control_planning_authorized": 0,
    "full_dataset_training_authorized": 0,
}


def _claims(
    *,
    cache_performed: bool = False,
    training_performed: bool = False,
    confirmation_performed: bool = False,
    controller_planning_authorized: bool = False,
) -> dict[str, Any]:
    return {
        "production_cache_generation_performed": int(cache_performed),
        "physical_training_performed": int(training_performed),
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


def _nonnegative_sum(left: Any, right: Any) -> float | None:
    if not _finite(left) or not _finite(right):
        return None
    left_value = float(left)
    right_value = float(right)
    if left_value < 0.0 or right_value < 0.0:
        return None
    result = left_value + right_value
    return result if math.isfinite(result) else None


def _resource_measurements_complete(values: Mapping[str, Any]) -> bool:
    """Distinguish measured resource overruns from absent execution evidence."""

    return bool(values) and all(
        _finite(value) and float(value) >= 0.0 for value in values.values()
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
    resource_only_failure: bool = False,
    cache_performed: bool = False,
    training_performed: bool = False,
    confirmation_performed: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in checks.items()}
    passed = bool(normalized) and all(
        _one(value.get("passed")) for value in normalized.values()
    )
    return {
        "schema": SCHEMA + f"-{name}-gate",
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
        "resource_only_failure": int(resource_only_failure),
        **_claims(
            cache_performed=cache_performed,
            training_performed=training_performed,
            confirmation_performed=confirmation_performed,
        ),
        **extra,
    }


def _execution_failed_gate(
    name: str,
    metrics: Mapping[str, Any],
    *,
    cache_performed: bool = False,
    training_performed: bool = False,
    confirmation_performed: bool = False,
) -> dict[str, Any]:
    result = _gate(
        name,
        {"stage_execution": _check(0, "==", 1, False)},
        failure_domain=str(metrics.get("failure_domain") or "execution"),
        scientific_evidence_complete=False,
        stage_execution_valid=False,
        numerically_valid=False,
        resource_valid=False,
        cache_performed=cache_performed,
        training_performed=training_performed,
        confirmation_performed=confirmation_performed,
    )
    result["evaluation_status"] = "execution_failed"
    result["failure_code"] = str(
        metrics.get("failure_code") or f"{name}_execution_failed"
    )
    return result


def not_evaluated_gate(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA + f"-{stage}-gate",
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
        "resource_only_failure": 0,
        **_claims(),
    }


PREFLIGHT_PROVENANCE_FLAGS = (
    "provenance_valid",
    "eager_pipeline_parent_valid",
    "controller_v1_parent_valid",
    "transitive_parent_binding_valid",
    "parent_sources_immutable",
    "historical_decisions_preserved",
)
PREFLIGHT_ADJUDICATION_FLAGS = (
    "legacy_boundary_tangent_adjudication_valid",
    "legacy_resource_adjudication_valid",
    "v1_only_resource_projection_failed",
)
PREFLIGHT_REPRESENTATION_FLAGS = (
    "scientific_config_frozen",
    "exact_target_unchanged",
    "boundary_tangent_representation_valid",
    "logistic_flow_valid",
    "model_input_firewall_valid",
)
PREFLIGHT_SCHEDULE_FLAGS = (
    "eager_schedule_integration_valid",
    "eager_prefix_contract_valid",
    "eager_profile_frozen",
    "base_and_midpoint_eager_schedule_valid",
    "production_cohort_plan_valid",
)
PREFLIGHT_DESIGN_FLAGS = (
    "cross_role_execution_isolation_valid",
    "path_plan_valid",
    "train_validation_confirmation_unopened",
    "preflight_paths_not_reused",
    "preflight_seam_namespace_valid",
    "historical_v1_paths_forbidden",
    "restart_plan_valid",
    "atomic_commit_plan_valid",
    "runtime_contract_valid",
)


def evaluate_eager_preflight(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentEagerThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate immutable binding, v2 design, and eager resource evidence."""

    t = thresholds or BoundaryTangentEagerThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate("preflight", metrics)
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in (
            PREFLIGHT_PROVENANCE_FLAGS
            + PREFLIGHT_ADJUDICATION_FLAGS
            + PREFLIGHT_REPRESENTATION_FLAGS
            + PREFLIGHT_SCHEDULE_FLAGS
            + PREFLIGHT_DESIGN_FLAGS
        )
    }
    checks.update(
        {
            "eager_parent_record_count": _check(
                metrics.get("eager_parent_record_count"),
                "==",
                t.eager_parent_record_count,
                metrics.get("eager_parent_record_count")
                == t.eager_parent_record_count,
            ),
            "controller_v1_parent_record_count": _check(
                metrics.get("controller_v1_parent_record_count"),
                "==",
                t.controller_v1_parent_record_count,
                metrics.get("controller_v1_parent_record_count")
                == t.controller_v1_parent_record_count,
            ),
            "root_seed": _check(
                metrics.get("root_seed"),
                "==",
                t.root_seed,
                metrics.get("root_seed") == t.root_seed,
            ),
            "model_seeds": _check(
                metrics.get("model_seeds"),
                "==",
                list(t.model_seeds),
                isinstance(metrics.get("model_seeds"), (list, tuple))
                and tuple(metrics["model_seeds"]) == t.model_seeds,
            ),
            "reserved_control_seed": _check(
                metrics.get("reserved_control_seed"),
                "==",
                t.reserved_control_seed,
                metrics.get("reserved_control_seed")
                == t.reserved_control_seed,
            ),
            "bootstrap_seed": _check(
                metrics.get("bootstrap_seed"),
                "==",
                t.bootstrap_seed,
                metrics.get("bootstrap_seed") == t.bootstrap_seed,
            ),
            "synthetic_teacher_seed": _check(
                metrics.get("synthetic_teacher_seed"),
                "==",
                t.synthetic_teacher_seed,
                metrics.get("synthetic_teacher_seed")
                == t.synthetic_teacher_seed,
            ),
            "baseline_null_seed": _check(
                metrics.get("baseline_null_seed"),
                "==",
                t.baseline_null_seed,
                metrics.get("baseline_null_seed") == t.baseline_null_seed,
            ),
            "training_path_ids": _check(
                metrics.get("training_path_ids"),
                "==",
                list(t.training_path_ids),
                isinstance(metrics.get("training_path_ids"), (list, tuple))
                and tuple(metrics["training_path_ids"]) == t.training_path_ids,
            ),
            "validation_path_ids": _check(
                metrics.get("validation_path_ids"),
                "==",
                list(t.validation_path_ids),
                isinstance(metrics.get("validation_path_ids"), (list, tuple))
                and tuple(metrics["validation_path_ids"])
                == t.validation_path_ids,
            ),
            "confirmation_path_ids": _check(
                metrics.get("confirmation_path_ids"),
                "==",
                list(t.confirmation_path_ids),
                isinstance(metrics.get("confirmation_path_ids"), (list, tuple))
                and tuple(metrics["confirmation_path_ids"])
                == t.confirmation_path_ids,
            ),
            "preflight_seam_path_ids": _check(
                metrics.get("preflight_seam_path_ids"),
                "==",
                list(t.preflight_seam_path_ids),
                isinstance(metrics.get("preflight_seam_path_ids"), (list, tuple))
                and tuple(metrics["preflight_seam_path_ids"])
                == t.preflight_seam_path_ids,
            ),
            "forbidden_historical_v1_path_ids": _check(
                metrics.get("forbidden_historical_v1_path_ids"),
                "==",
                list(t.forbidden_historical_v1_path_ids),
                isinstance(
                    metrics.get("forbidden_historical_v1_path_ids"),
                    (list, tuple),
                )
                and tuple(metrics["forbidden_historical_v1_path_ids"])
                == t.forbidden_historical_v1_path_ids,
            ),
            "train_validation_cohort_sizes": _check(
                metrics.get("train_validation_cohort_sizes"),
                "==",
                list(t.train_validation_cohort_sizes),
                isinstance(
                    metrics.get("train_validation_cohort_sizes"),
                    (list, tuple),
                )
                and tuple(metrics["train_validation_cohort_sizes"])
                == t.train_validation_cohort_sizes,
            ),
            "confirmation_cohort_sizes": _check(
                metrics.get("confirmation_cohort_sizes"),
                "==",
                list(t.confirmation_cohort_sizes),
                isinstance(
                    metrics.get("confirmation_cohort_sizes"),
                    (list, tuple),
                )
                and tuple(metrics["confirmation_cohort_sizes"])
                == t.confirmation_cohort_sizes,
            ),
            "training_paths": _check(
                metrics.get("training_paths"),
                "==",
                t.training_paths,
                metrics.get("training_paths") == t.training_paths,
            ),
            "validation_paths": _check(
                metrics.get("validation_paths"),
                "==",
                t.validation_paths,
                metrics.get("validation_paths") == t.validation_paths,
            ),
            "confirmation_paths": _check(
                metrics.get("confirmation_paths"),
                "==",
                t.confirmation_paths,
                metrics.get("confirmation_paths") == t.confirmation_paths,
            ),
            "projected_total_transitions": _check(
                metrics.get("projected_total_transitions"),
                "==",
                t.total_transitions,
                metrics.get("projected_total_transitions") == t.total_transitions,
            ),
            "projected_base_transitions": _check(
                metrics.get("projected_base_transitions"),
                "==",
                t.projected_base_transitions,
                metrics.get("projected_base_transitions")
                == t.projected_base_transitions,
            ),
            "projected_midpoint_transitions": _check(
                metrics.get("projected_midpoint_transitions"),
                "==",
                t.projected_midpoint_transitions,
                metrics.get("projected_midpoint_transitions")
                == t.projected_midpoint_transitions,
            ),
            "candidate_modes": _check(
                metrics.get("candidate_modes"),
                "==",
                t.candidate_modes,
                metrics.get("candidate_modes") == t.candidate_modes,
            ),
            "certificate_fraction": _check(
                metrics.get("certificate_fraction"),
                "==",
                1.0,
                _finite(metrics.get("certificate_fraction"))
                and float(metrics["certificate_fraction"]) == 1.0,
            ),
            "forbidden_event_count": _check(
                metrics.get("forbidden_event_count"),
                "==",
                0,
                _zero(metrics.get("forbidden_event_count")),
            ),
            "projected_elapsed_seconds": _check(
                metrics.get("projected_elapsed_seconds"),
                "<=",
                t.maximum_projected_seconds,
                _at_most(
                    metrics.get("projected_elapsed_seconds"),
                    t.maximum_projected_seconds,
                ),
            ),
            "projected_effective_rate": _check(
                metrics.get("projected_effective_rate"),
                ">=",
                t.minimum_projected_effective_rate,
                _at_least(
                    metrics.get("projected_effective_rate"),
                    t.minimum_projected_effective_rate,
                ),
            ),
            "minimum_profile_rate": _check(
                metrics.get("minimum_profile_rate"),
                ">=",
                t.minimum_transitions_per_second,
                _at_least(
                    metrics.get("minimum_profile_rate"),
                    t.minimum_transitions_per_second,
                ),
            ),
            "fallback_fraction": _check(
                metrics.get("fallback_fraction"),
                "<=",
                t.maximum_fallback_fraction,
                _at_most(
                    metrics.get("fallback_fraction"),
                    t.maximum_fallback_fraction,
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
            "maximum_mass_error": _check(
                metrics.get("maximum_mass_error"),
                "<=",
                t.maximum_mass_error,
                _at_most(metrics.get("maximum_mass_error"), t.maximum_mass_error),
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
            "projected_persisted_bytes": _check(
                metrics.get("projected_persisted_bytes"),
                "<=",
                t.maximum_persisted_bytes,
                isinstance(metrics.get("projected_persisted_bytes"), int)
                and not isinstance(metrics.get("projected_persisted_bytes"), bool)
                and 0
                <= int(metrics["projected_persisted_bytes"])
                <= t.maximum_persisted_bytes,
            ),
            "maximum_launch_lanes": _check(
                metrics.get("maximum_launch_lanes"),
                "<=",
                t.maximum_launch_lanes,
                isinstance(metrics.get("maximum_launch_lanes"), int)
                and not isinstance(metrics.get("maximum_launch_lanes"), bool)
                and 0 < int(metrics["maximum_launch_lanes"]) <= t.maximum_launch_lanes,
            ),
            "production_cache_generation_performed": _check(
                metrics.get("production_cache_generation_performed"),
                "==",
                0,
                _zero(metrics.get("production_cache_generation_performed")),
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
    provenance = set(PREFLIGHT_PROVENANCE_FLAGS) | {
        "eager_parent_record_count",
        "controller_v1_parent_record_count",
    }
    adjudication = set(PREFLIGHT_ADJUDICATION_FLAGS)
    representation = set(PREFLIGHT_REPRESENTATION_FLAGS)
    schedule = set(PREFLIGHT_SCHEDULE_FLAGS) | {
        "training_path_ids",
        "validation_path_ids",
        "confirmation_path_ids",
        "preflight_seam_path_ids",
        "forbidden_historical_v1_path_ids",
        "train_validation_cohort_sizes",
        "confirmation_cohort_sizes",
        "projected_base_transitions",
        "projected_midpoint_transitions",
    }
    resource = {
        "projected_elapsed_seconds",
        "projected_effective_rate",
        "minimum_profile_rate",
        "fallback_fraction",
        "fallback_time_fraction",
        "peak_memory_fraction",
        "projected_persisted_bytes",
        "maximum_launch_lanes",
    }
    numerical = {"certificate_fraction", "forbidden_event_count", "maximum_mass_error"}
    resource_only = (
        bool(failed)
        and failed <= resource
        and _resource_measurements_complete(
            {name: metrics.get(name) for name in resource}
        )
    )
    if failed & provenance:
        domain = "provenance"
        complete = False
    elif failed & adjudication:
        domain = "adjudication"
        complete = False
    elif failed & schedule:
        domain = "schedule_integration"
        complete = False
    elif failed & representation:
        domain = "representation"
        complete = False
    elif failed & numerical:
        domain = "numerical"
        complete = True
    elif resource_only:
        domain = "resource_gate"
        complete = True
    elif failed:
        domain = "design"
        complete = False
    else:
        domain = None
        complete = True
    return _gate(
        "preflight",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=complete,
        numerically_valid=not bool(failed & numerical),
        resource_valid=not bool(failed & resource),
        resource_only_failure=resource_only,
        provenance_valid=int(not bool(failed & provenance)),
        legacy_boundary_tangent_adjudication_valid=int(
            not bool(failed & adjudication)
        ),
        boundary_tangent_representation_valid=int(
            not bool(failed & representation)
        ),
        eager_schedule_integration_valid=int(not bool(failed & schedule)),
        thresholds=t.to_dict(),
    )


CACHE_EXECUTION_FLAGS = (
    "cache_complete",
    "train_cache_complete",
    "validation_cache_complete",
    "atomic_shard_chains_valid",
    "resume_replay_valid",
    "completed_shard_skipping_valid",
)
CACHE_DESIGN_FLAGS = (
    "confirmation_absent",
    "production_cohort_plan_observed",
    "eager_base_prefix_schedule_valid",
    "eager_branch_prefix_schedule_valid",
    "cross_role_execution_isolation_valid",
    "artifact_role_isolation_valid",
    "input_label_separation_valid",
    "raw_target_contract_valid",
)


def evaluate_eager_cache(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentEagerThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate fresh train/validation cache generation."""

    t = thresholds or BoundaryTangentEagerThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate(
            "cache",
            metrics,
            cache_performed=_one(
                metrics.get("production_cache_generation_performed")
            ),
        )
    projected_cache_plus_confirmation_seconds = _nonnegative_sum(
        metrics.get("cache_elapsed_seconds"),
        metrics.get("frozen_conservative_confirmation_projection_seconds"),
    )
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in CACHE_EXECUTION_FLAGS + CACHE_DESIGN_FLAGS
    }
    exact_counts = {
        "train_row_count": t.train_rows,
        "validation_row_count": t.validation_rows,
        "train_transition_count": t.train_transitions,
        "validation_transition_count": t.validation_transitions,
        "cache_transition_count": t.train_transitions + t.validation_transitions,
    }
    checks.update(
        {
            name: _check(
                metrics.get(name), "==", expected, metrics.get(name) == expected
            )
            for name, expected in exact_counts.items()
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
                _at_least(
                    metrics.get("minimum_role_rate"),
                    t.minimum_transitions_per_second,
                ),
            ),
            "projected_cache_plus_confirmation_seconds": _check(
                projected_cache_plus_confirmation_seconds,
                "<=",
                t.maximum_projected_seconds,
                _at_most(
                    projected_cache_plus_confirmation_seconds,
                    t.maximum_projected_seconds,
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
            "total_persisted_cache_bytes": _check(
                metrics.get("total_persisted_cache_bytes"),
                "<=",
                t.maximum_persisted_bytes,
                isinstance(metrics.get("total_persisted_cache_bytes"), int)
                and not isinstance(metrics.get("total_persisted_cache_bytes"), bool)
                and 0
                <= int(metrics["total_persisted_cache_bytes"])
                <= t.maximum_persisted_bytes,
            ),
            "maximum_launch_lanes": _check(
                metrics.get("maximum_launch_lanes"),
                "<=",
                t.maximum_launch_lanes,
                isinstance(metrics.get("maximum_launch_lanes"), int)
                and not isinstance(metrics.get("maximum_launch_lanes"), bool)
                and 0 < int(metrics["maximum_launch_lanes"]) <= t.maximum_launch_lanes,
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
        "projected_cache_plus_confirmation_seconds",
        "fallback_fraction",
        "fallback_time_fraction",
        "peak_memory_fraction",
        "total_persisted_cache_bytes",
        "maximum_launch_lanes",
    }
    design = set(CACHE_DESIGN_FLAGS) | set(exact_counts)
    resource_only = (
        bool(failed)
        and failed <= resource
        and _resource_measurements_complete(
            {
                **{
                    name: metrics.get(name)
                    for name in resource
                    if name != "projected_cache_plus_confirmation_seconds"
                },
                "projected_cache_plus_confirmation_seconds": (
                    projected_cache_plus_confirmation_seconds
                ),
            }
        )
    )
    if failed & set(CACHE_EXECUTION_FLAGS):
        domain = "execution"
        complete = False
    elif failed & design:
        domain = "design"
        complete = False
    elif failed & numerical:
        domain = "numerical"
        complete = True
    elif resource_only:
        domain = "resource_gate"
        complete = True
    elif failed:
        domain = "execution"
        complete = False
    else:
        domain = None
        complete = True
    return _gate(
        "cache",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=complete,
        numerically_valid=not bool(failed & numerical),
        resource_valid=not bool(failed & resource),
        resource_only_failure=resource_only,
        cache_performed=True,
        cache_elapsed_seconds=metrics.get("cache_elapsed_seconds"),
        frozen_conservative_confirmation_projection_seconds=metrics.get(
            "frozen_conservative_confirmation_projection_seconds"
        ),
        projected_cache_plus_confirmation_seconds=(
            projected_cache_plus_confirmation_seconds
        ),
        thresholds=t.to_dict(),
    )


TRAIN_OPTIMIZATION_FLAGS = (
    "training_complete",
    "synthetic_teacher_passed",
    "synthetic_every_validation_path_beats_zero",
    "exact_baseline_null_passed",
    "physical_labels_opened_after_controls",
    "all_physical_tasks_complete_finite",
    "validation_only_selection",
    "selection_rule_valid",
    "selected_checkpoint_eligible",
    "selected_beats_baseline_overall",
    "selected_beats_baseline_high_reverse_time",
    "confirmation_absent",
)
TRAIN_BASELINE_FLAGS = (
    "baseline_valid",
    "baseline_training_only",
    "raw_target_contract_valid",
)


def evaluate_eager_train(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentEagerThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate optimizer controls and validation-only checkpoint selection."""

    t = thresholds or BoundaryTangentEagerThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate(
            "train",
            metrics,
            cache_performed=_one(
                metrics.get("production_cache_generation_performed")
            ),
            training_performed=_one(metrics.get("physical_training_performed")),
        )
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in TRAIN_OPTIMIZATION_FLAGS + TRAIN_BASELINE_FLAGS
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
            "null_selected_update": _check(
                metrics.get("null_selected_update"),
                "==",
                0,
                metrics.get("null_selected_update") == 0,
            ),
            "model_seed_count": _check(
                metrics.get("model_seed_count"),
                "==",
                len(t.model_seeds),
                metrics.get("model_seed_count") == len(t.model_seeds),
            ),
            "maximum_updates": _check(
                metrics.get("maximum_updates"),
                "==",
                t.maximum_updates,
                metrics.get("maximum_updates") == t.maximum_updates,
            ),
            "quotient_target_formed": _check(
                metrics.get("quotient_target_formed"),
                "==",
                0,
                _zero(metrics.get("quotient_target_formed")),
            ),
            "selected_nonzero": _check(
                metrics.get("selected_nonzero"),
                "==",
                1,
                _one(metrics.get("selected_nonzero")),
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
                1,
                _one(metrics.get("physical_training_performed")),
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
    baseline_checks = set(TRAIN_BASELINE_FLAGS) | {"quotient_target_formed"}
    baseline_failed = bool(failed & baseline_checks)
    selection_signal_checks = {
        "selected_nonzero",
        "selected_checkpoint_eligible",
        "selected_beats_baseline_overall",
        "selected_beats_baseline_high_reverse_time",
    }
    baseline_only = (
        "selected_nonzero" in failed
        and not baseline_failed
        and not bool(failed - selection_signal_checks)
    )
    optimization_failed = bool(failed - baseline_checks) and not baseline_only
    if baseline_failed:
        domain = "baseline"
    elif baseline_only:
        domain = "baseline_only"
    elif optimization_failed:
        domain = "optimization"
    else:
        domain = None
    return _gate(
        "train",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=not optimization_failed,
        numerically_valid=not baseline_failed and not optimization_failed,
        resource_valid=True,
        cache_performed=True,
        training_performed=True,
        boundary_tangent_baseline_valid=int(not baseline_failed),
        optimization_pipeline_valid=int(not optimization_failed),
        boundary_tangent_baseline_only=int(baseline_only),
        thresholds=t.to_dict(),
    )


CONFIRM_EXECUTION_FLAGS = (
    "confirmation_complete",
    "confirmation_opened_once",
    "selection_sealed_before_paths",
    "atomic_shard_chains_valid",
    "resume_replay_valid",
    "eager_base_prefix_schedule_valid",
    "eager_branch_prefix_schedule_valid",
    "raw_confirmation_labels_not_persisted",
    "complete_cartesian_rows",
)


def evaluate_eager_confirm(
    max_t_record: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    integrity_checks: Mapping[str, bool] | None = None,
    thresholds: BoundaryTangentEagerThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate sealed confirmation inference and its execution resources."""

    t = thresholds or BoundaryTangentEagerThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate(
            "confirm",
            metrics,
            cache_performed=_one(
                metrics.get("production_cache_generation_performed")
            ),
            training_performed=_one(metrics.get("physical_training_performed")),
            confirmation_performed=_one(metrics.get("confirmation_performed")),
        )
    actual_cache_plus_confirmation_seconds = _nonnegative_sum(
        metrics.get("cache_elapsed_seconds"),
        metrics.get("confirmation_elapsed_seconds"),
    )
    base = evaluate_confirmation_gate(
        max_t_record,
        thresholds=t.confirmation_thresholds(),
    )
    checks = dict(base.get("checks", {}))
    checks.update(
        {
            "confirmation_path_ids": _check(
                max_t_record.get("path_ids"),
                "==",
                list(t.confirmation_path_ids),
                isinstance(max_t_record.get("path_ids"), (list, tuple))
                and tuple(max_t_record["path_ids"])
                == t.confirmation_path_ids,
            ),
            "bootstrap_namespace": _check(
                max_t_record.get("namespace"),
                "==",
                0,
                max_t_record.get("namespace") == 0,
            ),
        }
    )
    checks.update(
        {
            name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
            for name in CONFIRM_EXECUTION_FLAGS
        }
    )
    checks.update(
        {
            str(name): _check(int(bool(value)), "==", 1, bool(value))
            for name, value in sorted(dict(integrity_checks or {}).items())
        }
    )
    checks.update(
        {
            "confirmation_path_count": _check(
                metrics.get("confirmation_path_count"),
                "==",
                t.confirmation_paths,
                metrics.get("confirmation_path_count") == t.confirmation_paths,
            ),
            "confirmation_row_count": _check(
                metrics.get("confirmation_row_count"),
                "==",
                t.confirmation_rows,
                metrics.get("confirmation_row_count") == t.confirmation_rows,
            ),
            "confirmation_transition_count": _check(
                metrics.get("confirmation_transition_count"),
                "==",
                t.confirmation_transitions,
                metrics.get("confirmation_transition_count")
                == t.confirmation_transitions,
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
            "actual_cache_plus_confirmation_seconds": _check(
                actual_cache_plus_confirmation_seconds,
                "<=",
                t.maximum_projected_seconds,
                _at_most(
                    actual_cache_plus_confirmation_seconds,
                    t.maximum_projected_seconds,
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
            "total_persisted_bytes": _check(
                metrics.get("total_persisted_bytes"),
                "<=",
                t.maximum_persisted_bytes,
                isinstance(metrics.get("total_persisted_bytes"), int)
                and not isinstance(metrics.get("total_persisted_bytes"), bool)
                and 0 <= int(metrics["total_persisted_bytes"]) <= t.maximum_persisted_bytes,
            ),
            "maximum_launch_lanes": _check(
                metrics.get("maximum_launch_lanes"),
                "<=",
                t.maximum_launch_lanes,
                isinstance(metrics.get("maximum_launch_lanes"), int)
                and not isinstance(metrics.get("maximum_launch_lanes"), bool)
                and 0 < int(metrics["maximum_launch_lanes"]) <= t.maximum_launch_lanes,
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
                1,
                _one(metrics.get("physical_training_performed")),
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
    inference_configuration = {
        "paired_risk_inference_valid",
        "confirmation_family_size",
        "confirmation_path_ids",
        "bootstrap_namespace",
    }
    statistical = {
        "combined_vs_zero_all_simultaneous_lower_positive",
        "combined_vs_baseline_all_simultaneous_lower_positive",
    }
    numerical = {"certificate_fraction", "maximum_mass_error", "forbidden_event_count"}
    resource = {
        "transitions_per_second",
        "actual_cache_plus_confirmation_seconds",
        "fallback_fraction",
        "fallback_time_fraction",
        "peak_memory_fraction",
        "total_persisted_bytes",
        "maximum_launch_lanes",
    }
    execution = set(CONFIRM_EXECUTION_FLAGS) | {
        "confirmation_path_count",
        "confirmation_row_count",
        "confirmation_transition_count",
        *dict(integrity_checks or {}),
    }
    statistical_failure = bool(failed & statistical) and _one(
        base.get("paired_risk_inference_valid")
    )
    inference_invalid = (
        not _one(base.get("paired_risk_inference_valid"))
        or bool(failed & inference_configuration)
    )
    resource_only = (
        bool(failed)
        and failed <= resource
        and _resource_measurements_complete(
            {
                **{
                    name: metrics.get(name)
                    for name in resource
                    if name != "actual_cache_plus_confirmation_seconds"
                },
                "actual_cache_plus_confirmation_seconds": (
                    actual_cache_plus_confirmation_seconds
                ),
            }
        )
    )
    if inference_invalid:
        domain = "inference"
        complete = False
    elif failed & execution:
        domain = "execution"
        complete = False
    elif failed & numerical:
        domain = "numerical"
        complete = True
    elif resource_only:
        domain = "resource_gate"
        complete = True
    elif statistical_failure:
        domain = "scientific_gate"
        complete = True
    elif failed:
        domain = "inference"
        complete = False
    else:
        domain = None
        complete = True
    return _gate(
        "confirm",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=complete,
        numerically_valid=(
            not inference_invalid and not bool(failed & numerical)
        ),
        resource_valid=not bool(failed & resource),
        resource_only_failure=resource_only,
        cache_performed=True,
        training_performed=True,
        confirmation_performed=True,
        cache_elapsed_seconds=metrics.get("cache_elapsed_seconds"),
        confirmation_elapsed_seconds=metrics.get("confirmation_elapsed_seconds"),
        actual_cache_plus_confirmation_seconds=(
            actual_cache_plus_confirmation_seconds
        ),
        paired_risk_inference_valid=int(not inference_invalid),
        combined_vs_zero_point_positive=int(
            _one(base.get("combined_vs_zero_point_positive"))
        ),
        combined_vs_zero_replicated=int(
            _one(base.get("combined_vs_zero_replicated"))
        ),
        combined_vs_baseline_point_positive=int(
            _one(base.get("combined_vs_baseline_point_positive"))
        ),
        combined_vs_baseline_replicated=int(
            _one(base.get("combined_vs_baseline_replicated"))
        ),
        legacy_confirmation_gate=base,
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
    confirmation_authorized: bool = False,
    controller_planning_authorized: bool = False,
    cache_performed: bool = False,
    training_performed: bool = False,
    confirmation_performed: bool = False,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": evaluation_status,
        "decision": decision,
        "recommended_next_action": action,
        "cache_generation_authorized": int(cache_authorized),
        "physical_training_authorized": int(training_authorized),
        "confirmation_authorized": int(confirmation_authorized),
        **_claims(
            cache_performed=cache_performed,
            training_performed=training_performed,
            confirmation_performed=confirmation_performed,
            controller_planning_authorized=controller_planning_authorized,
        ),
    }


def decide_eager_boundary_tangent_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the closed cache/train/confirm decision partition."""

    cache_done = _flag(cache_gate, "production_cache_generation_performed")
    train_done = _flag(train_gate, "physical_training_performed")
    confirm_done = _flag(confirm_gate, "confirmation_performed")
    if _status(preflight_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_preflight",
            "verify immutable parents and the eager v2 execution design",
            evaluation_status="not_evaluated",
        )
    if not _passed(preflight_gate):
        domain = str((preflight_gate or {}).get("failure_domain", ""))
        execution_failed = _status(preflight_gate) == "execution_failed"
        if domain == "provenance" or (
            not domain and not _flag(preflight_gate, "provenance_valid")
        ):
            decision = BoundaryTangentEagerDecision.CONTROL_PROVENANCE_INVALID
            action = "repair immutable eager, v1, affine, and coarse parent binding"
        elif domain == "adjudication" or (
            not domain
            and not _flag(
                preflight_gate, "legacy_boundary_tangent_adjudication_valid"
            )
        ):
            decision = (
                BoundaryTangentEagerDecision.LEGACY_BOUNDARY_TANGENT_ADJUDICATION_INVALID
            )
            action = "repair the read-only affine and legacy-resource adjudications"
        elif domain == "schedule_integration" or (
            not domain
            and not _flag(preflight_gate, "eager_schedule_integration_valid")
        ):
            decision = (
                BoundaryTangentEagerDecision.EAGER_SCHEDULE_INTEGRATION_INVALID
            )
            action = "repair the frozen eager-prefix schedule integration"
        elif domain == "representation" or (
            not domain
            and not _flag(
                preflight_gate, "boundary_tangent_representation_valid"
            )
        ):
            decision = (
                BoundaryTangentEagerDecision.BOUNDARY_TANGENT_REPRESENTATION_INVALID
            )
            action = "repair boundary-tangent representation controls"
        elif execution_failed:
            decision = BoundaryTangentEagerDecision.EAGER_SCHEDULE_INTEGRATION_INVALID
            action = "repair the failed eager v2 preflight execution"
        else:
            decision = BoundaryTangentEagerDecision.BOUNDARY_TANGENT_DESIGN_INFEASIBLE
            action = "repair the additive eager v2 execution design"
        return _decision_record(decision.value, action)
    if _status(cache_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_cache",
            "generate the fresh exact train and validation cache",
            evaluation_status="not_evaluated",
            cache_authorized=True,
        )
    if not _passed(cache_gate):
        resource_failure = (
            str((cache_gate or {}).get("failure_domain", ""))
            == "resource_gate"
            and _flag(cache_gate, "resource_only_failure")
        )
        decision = (
            BoundaryTangentEagerDecision.BOUNDARY_TANGENT_CACHE_RESOURCE_INFEASIBLE
            if resource_failure
            else BoundaryTangentEagerDecision.FRESH_EXACT_CACHE_INVALID
        )
        return _decision_record(
            decision.value,
            (
                "retain exact cache evidence and revise only the execution resource plan"
                if resource_failure
                else "retain cache evidence and repair only its failed execution domain"
            ),
            cache_performed=cache_done,
        )
    if _status(train_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_train",
            "run bounded baseline, optimizer controls, and physical selection",
            evaluation_status="not_evaluated",
            training_authorized=True,
            cache_performed=True,
        )
    if _status(train_gate) == "execution_failed":
        domain = str((train_gate or {}).get("failure_domain", ""))
        decision = (
            BoundaryTangentEagerDecision.BOUNDARY_TANGENT_BASELINE_INVALID
            if domain == "baseline"
            else BoundaryTangentEagerDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID
        )
        return _decision_record(
            decision.value,
            "repair the failed bounded training/control execution",
            cache_performed=True,
            training_performed=train_done,
        )
    if not _flag(train_gate, "boundary_tangent_baseline_valid"):
        decision = BoundaryTangentEagerDecision.BOUNDARY_TANGENT_BASELINE_INVALID
        action = "repair the training-only frozen tangent baseline"
    elif not _flag(train_gate, "optimization_pipeline_valid"):
        decision = (
            BoundaryTangentEagerDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID
        )
        action = "repair synthetic, null, or finite optimization controls"
    elif _flag(train_gate, "boundary_tangent_baseline_only"):
        decision = BoundaryTangentEagerDecision.BOUNDARY_TANGENT_BASELINE_ONLY_SIGNAL
        action = "retain the baseline and do not open confirmation paths"
    elif not _passed(train_gate):
        decision = (
            BoundaryTangentEagerDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID
        )
        action = "repair validation-only selection and checkpoint sealing"
    else:
        decision = None
        action = ""
    if decision is not None:
        return _decision_record(
            decision.value,
            action,
            cache_performed=True,
            training_performed=train_done,
        )
    if _status(confirm_gate) == "not_evaluated":
        return _decision_record(
            "ready_for_confirm",
            "open the fixed sealed confirmation paths exactly once",
            evaluation_status="not_evaluated",
            confirmation_authorized=True,
            cache_performed=True,
            training_performed=True,
        )
    if _status(confirm_gate) == "execution_failed":
        decision = BoundaryTangentEagerDecision.BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE
        action = "repair the failed sealed-confirmation execution without reopening paths"
    elif (
        str((confirm_gate or {}).get("failure_domain", "")) == "resource_gate"
        and _flag(confirm_gate, "resource_only_failure")
    ):
        decision = BoundaryTangentEagerDecision.BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE
        action = "retain the sealed audit and revise only its execution resource plan"
    elif not _flag(confirm_gate, "paired_risk_inference_valid"):
        decision = BoundaryTangentEagerDecision.PAIRED_RISK_INFERENCE_INVALID
        action = "repair report-only paired max-T validation"
    elif not _flag(confirm_gate, "combined_vs_baseline_point_positive"):
        decision = BoundaryTangentEagerDecision.SELECTION_FALSE_DISCOVERY
        action = "record that the selected residual did not beat its baseline"
    elif not _flag(confirm_gate, "combined_vs_zero_point_positive"):
        decision = (
            BoundaryTangentEagerDecision.BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_NOT_DETECTED
        )
        action = "do not construct a controller from undetected time-local signal"
    elif not (
        _flag(confirm_gate, "combined_vs_zero_replicated")
        and _flag(confirm_gate, "combined_vs_baseline_replicated")
    ):
        decision = BoundaryTangentEagerDecision.BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE
        action = "retain the sealed audit without resizing or rerunning it"
    elif not _passed(confirm_gate):
        decision = BoundaryTangentEagerDecision.PAIRED_RISK_INFERENCE_INVALID
        action = "retain scientific evidence and repair confirmation integrity/resources"
    else:
        decision = (
            BoundaryTangentEagerDecision.EXACT_RB_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED
        )
        action = "plan a separate at-most-eight-phase controller-control workflow"
    final = (
        decision
        is BoundaryTangentEagerDecision.EXACT_RB_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED
    )
    return _decision_record(
        decision.value,
        action,
        controller_planning_authorized=final,
        cache_performed=True,
        training_performed=True,
        confirmation_performed=confirm_done,
    )


def evaluate_eager_boundary_tangent_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    if require_gate not in {"none", "preflight", "cache", "train", "confirm"}:
        raise BoundaryTangentEagerGateError(f"unknown required gate: {require_gate}")
    components = {
        "preflight": dict(
            preflight_gate or not_evaluated_gate("preflight", "not run")
        ),
        "cache": dict(cache_gate or not_evaluated_gate("cache", "not run")),
        "train": dict(train_gate or not_evaluated_gate("train", "not run")),
        "confirm": dict(confirm_gate or not_evaluated_gate("confirm", "not run")),
    }
    required = {
        "none": (),
        "preflight": ("preflight",),
        "cache": ("preflight", "cache"),
        "train": ("preflight", "cache", "train"),
        "confirm": ("preflight", "cache", "train", "confirm"),
    }[require_gate]
    decision = decide_eager_boundary_tangent_workflow(
        preflight_gate=components["preflight"],
        cache_gate=components["cache"],
        train_gate=components["train"],
        confirm_gate=components["confirm"],
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_gate_pass": int(all(_passed(components[name]) for name in required)),
        "components": components,
        "decision": decision,
        "thresholds": BoundaryTangentEagerThresholds().to_dict(),
        **_claims(
            cache_performed=_flag(
                components["cache"], "production_cache_generation_performed"
            ),
            training_performed=_flag(
                components["train"], "physical_training_performed"
            ),
            confirmation_performed=_flag(
                components["confirm"], "confirmation_performed"
            ),
            controller_planning_authorized=_flag(
                decision, "controller_control_planning_authorized"
            ),
        ),
    }


# Concise public names used by the stage CLI.  The longer names above remain
# available so artifacts identify this specific eager boundary-tangent workflow.
def evaluate_preflight_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentEagerThresholds | None = None,
) -> dict[str, Any]:
    return evaluate_eager_preflight(metrics, thresholds=thresholds)


def evaluate_cache_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentEagerThresholds | None = None,
) -> dict[str, Any]:
    return evaluate_eager_cache(metrics, thresholds=thresholds)


def evaluate_train_gate(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentEagerThresholds | None = None,
) -> dict[str, Any]:
    return evaluate_eager_train(metrics, thresholds=thresholds)


def evaluate_confirm_gate(
    max_t_record: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    integrity_checks: Mapping[str, bool] | None = None,
    thresholds: BoundaryTangentEagerThresholds | None = None,
) -> dict[str, Any]:
    return evaluate_eager_confirm(
        max_t_record,
        metrics,
        integrity_checks=integrity_checks,
        thresholds=thresholds,
    )


def decide_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return decide_eager_boundary_tangent_workflow(
        preflight_gate=preflight_gate,
        cache_gate=cache_gate,
        train_gate=train_gate,
        confirm_gate=confirm_gate,
    )


def evaluate_required_gate(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    return evaluate_eager_boundary_tangent_workflow(
        preflight_gate=preflight_gate,
        cache_gate=cache_gate,
        train_gate=train_gate,
        confirm_gate=confirm_gate,
        require_gate=require_gate,
    )


REQUIRED_GATES = ("none", "preflight", "cache", "train", "confirm")
DECISION_VALUES = tuple(item.value for item in BoundaryTangentEagerDecision)
FINAL_DECISION = (
    BoundaryTangentEagerDecision.EXACT_RB_BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_CONFIRMED.value
)


__all__ = [
    "BoundaryTangentEagerDecision",
    "BoundaryTangentEagerGateError",
    "BoundaryTangentEagerThresholds",
    "CACHE_DESIGN_FLAGS",
    "CACHE_EXECUTION_FLAGS",
    "CONFIRM_EXECUTION_FLAGS",
    "CONFIRMATION_ROWS",
    "CONFIRMATION_TRANSITIONS",
    "DECISION_VALUES",
    "FINAL_DECISION",
    "PREFLIGHT_ADJUDICATION_FLAGS",
    "PREFLIGHT_DESIGN_FLAGS",
    "PREFLIGHT_PROVENANCE_FLAGS",
    "PREFLIGHT_REPRESENTATION_FLAGS",
    "PREFLIGHT_SCHEDULE_FLAGS",
    "PROJECTED_BASE_TRANSITIONS",
    "PROJECTED_MIDPOINT_TRANSITIONS",
    "REQUIRED_GATES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TOTAL_TRANSITIONS",
    "TRAIN_ROWS",
    "TRAIN_TRANSITIONS",
    "TRAIN_BASELINE_FLAGS",
    "TRAIN_OPTIMIZATION_FLAGS",
    "VALIDATION_ROWS",
    "VALIDATION_TRANSITIONS",
    "decide_eager_boundary_tangent_workflow",
    "decide_workflow",
    "evaluate_eager_boundary_tangent_workflow",
    "evaluate_eager_cache",
    "evaluate_eager_confirm",
    "evaluate_eager_preflight",
    "evaluate_eager_train",
    "evaluate_cache_gate",
    "evaluate_confirm_gate",
    "evaluate_preflight_gate",
    "evaluate_required_gate",
    "evaluate_train_gate",
    "not_evaluated_gate",
]
