"""Fail-closed gates for exact K=512 Jacobi/RB one-image learnability.

The gates in this module deliberately make no statement about convergence of
the split chain to the unsplit Eulerian generator.  They adjudicate only
whether the exact binary64 Rao--Blackwell label of the already validated
``K=512`` split chain contains held-out, later-state conditional signal.

No filesystem, CUDA, model, or sampler dependency is imported here.  This
keeps the decision logic independently testable and usable by the orchestration
CLI without creating a path from a scientific gate back into training code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


SCHEMA = "experiment12-d0-jacobi-rb-one-image-learnability-gate"
SCHEMA_VERSION = 1
STAGES = ("preflight", "cache", "train", "confirm", "report", "all")
REQUIRED_GATES = ("none", "preflight", "cache", "train", "confirm")
CLAIM_SCOPE = (
    "held-out conditional learnability of the exact Rao--Blackwell label "
    "for the frozen one-image exact K=512 split chain"
)

# These fields are copied into every gate and decision so a partial artifact
# cannot accidentally acquire authority merely by being read out of context.
NO_CLAIM_AUTHORIZATION = {
    "physical_training_authorized": 0,
    "production_refinement_authorized": 0,
    "state_dependent_strang_refinement_established": 0,
    "unsplit_generator_approximation_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
    "reverse_sampling_authorized": 0,
    "sampling_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "known_prior_claim_authorized": 0,
    "full_dataset_training_authorized": 0,
}
NO_WORK = {
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


class JacobiRBLearnabilityDecision(str, Enum):
    """The complete, closed terminal outcome set."""

    PARENT_SCOPE_INVALID = "parent_scope_invalid"
    EXACT_CACHE_INVALID = "exact_cache_invalid"
    MODEL_INPUT_CONTRACT_INVALID = "model_input_contract_invalid"
    OPTIMIZATION_PIPELINE_INVALID = "optimization_pipeline_invalid"
    NO_DETECTABLE_ONE_IMAGE_CONDITIONAL_SIGNAL = (
        "no_detectable_one_image_conditional_signal"
    )
    EXACT_K512_SPLIT_CHAIN_RB_LABEL_LEARNABLE = (
        "exact_k512_split_chain_rb_label_learnable"
    )


# Short alias useful to callers that already include ``JacobiRB`` in a module
# name.  Keeping both names also makes serialized code references unambiguous.
LearnabilityDecision = JacobiRBLearnabilityDecision


@dataclass(frozen=True)
class JacobiRBLearnabilityThresholds:
    """Frozen production thresholds and exact workload cardinalities."""

    grid_cells: int = 784
    edges_per_phase: int = 392
    outer_steps: int = 512
    phases_per_step: int = 7
    steps_per_shard: int = 8
    selected_outer_step_count: int = 32
    paths_per_split: int = 8
    samples_per_split: int = 1_792
    transitions_per_split: int = 11_239_424
    total_transition_count: int = 33_718_272
    minimum_effective_transitions_per_second: float = 1_300.0
    maximum_projected_total_hours: float = 10.0
    maximum_peak_memory_fraction: float = 0.80
    maximum_persisted_cache_bytes: int = 134_217_728
    maximum_mass_error: float = 2.0e-12
    teacher_relative_mse: float = 0.01
    validation_path_count: int = 8
    confirmation_path_count: int = 8
    maximum_updates: int = 4_000

    def __post_init__(self) -> None:
        integer_fields = (
            "grid_cells",
            "edges_per_phase",
            "outer_steps",
            "phases_per_step",
            "steps_per_shard",
            "selected_outer_step_count",
            "paths_per_split",
            "samples_per_split",
            "transitions_per_split",
            "total_transition_count",
            "maximum_persisted_cache_bytes",
            "validation_path_count",
            "confirmation_path_count",
            "maximum_updates",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "minimum_effective_transitions_per_second",
            "maximum_projected_total_hours",
            "maximum_peak_memory_fraction",
            "maximum_mass_error",
            "teacher_relative_mse",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_peak_memory_fraction > 1.0:
            raise ValueError("maximum_peak_memory_fraction must be at most one")
        if self.teacher_relative_mse > 1.0:
            raise ValueError("teacher_relative_mse must be at most one")

    @property
    def selected_outer_steps(self) -> tuple[int, ...]:
        return tuple(15 + 16 * index for index in range(self.selected_outer_step_count))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["selected_outer_steps"] = list(self.selected_outer_steps)
        return result


LearnabilityThresholds = JacobiRBLearnabilityThresholds


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _status(gate: Mapping[str, Any] | None) -> str:
    if not isinstance(gate, Mapping):
        return "not_evaluated"
    return str(gate.get("evaluation_status", "not_evaluated"))


def _passed(gate: Mapping[str, Any] | bool | int | None) -> bool:
    if isinstance(gate, Mapping):
        return _status(gate) == "evaluated" and _one(gate.get("passed"))
    return gate is True or (_integer(gate) and int(gate) == 1)


def _lookup(metrics: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return default


def _check(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _eq(metrics: Mapping[str, Any], name: str, threshold: Any) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", threshold, value == threshold)


def _eq_one(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", 1, _one(value))


def _eq_zero(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", 0, _zero(value))


def _le(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value,
        "<=",
        threshold,
        _finite(value) and 0.0 <= float(value) <= float(threshold),
    )


def _ge(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value,
        ">=",
        threshold,
        _finite(value) and float(value) >= float(threshold),
    )


def _strict_lt(
    metrics: Mapping[str, Any], left: str, right: str
) -> dict[str, Any]:
    left_value = metrics.get(left)
    right_value = metrics.get(right)
    return _check(
        {left: left_value, right: right_value},
        f"{left} < {right}",
        None,
        _finite(left_value)
        and _finite(right_value)
        and 0.0 <= float(left_value) < float(right_value),
    )


def _gate(
    name: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    claim_scope: str,
    evaluation_status: str = "evaluated",
    physical_training_performed: int = 0,
    **fields: Any,
) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in checks.items()}
    passed = (
        evaluation_status == "evaluated"
        and bool(normalized)
        and all(_one(value.get("passed")) for value in normalized.values())
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": str(name),
        "evaluation_status": str(evaluation_status),
        "claim_scope": str(claim_scope),
        "subchecks": normalized,
        "passed": int(passed),
        **fields,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
        "physical_training_performed": int(physical_training_performed),
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    result = _gate(
        name,
        {},
        claim_scope=CLAIM_SCOPE,
        evaluation_status="not_evaluated",
    )
    result["reason"] = str(reason)
    return result


def execution_failed_gate(
    name: str,
    *,
    failure_domain: str,
    failure_code: str,
    error_type: str,
    error: str,
) -> dict[str, Any]:
    result = _gate(
        name,
        {},
        claim_scope=CLAIM_SCOPE,
        evaluation_status="execution_failed",
    )
    result.update(
        stage_execution_valid=0,
        scientific_evidence_complete=0,
        failure_domain=str(failure_domain),
        failure_code=str(failure_code),
        error_type=str(error_type),
        error=str(error),
    )
    return result


def evaluate_learnability_preflight(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBLearnabilityThresholds | None = None,
) -> dict[str, Any]:
    """Gate immutable parents, capture parity, path plan, and resources."""

    t = thresholds or JacobiRBLearnabilityThresholds()
    expected_one = (
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
    checks = {name: _eq_one(metrics, name) for name in expected_one}
    checks.update(
        {
            "outer_steps": _eq(metrics, "outer_steps", t.outer_steps),
            "steps_per_shard": _eq(metrics, "steps_per_shard", t.steps_per_shard),
            "paths_per_split": _eq(metrics, "paths_per_split", t.paths_per_split),
            "selected_outer_steps": _check(
                metrics.get("selected_outer_steps"),
                "==",
                list(t.selected_outer_steps),
                isinstance(metrics.get("selected_outer_steps"), (list, tuple))
                and tuple(metrics["selected_outer_steps"]) == t.selected_outer_steps,
            ),
            "effective_transitions_per_second": _ge(
                metrics,
                "effective_transitions_per_second",
                t.minimum_effective_transitions_per_second,
            ),
            "projected_total_hours": _le(
                metrics,
                "projected_total_hours",
                t.maximum_projected_total_hours,
            ),
            "peak_memory_fraction": _le(
                metrics,
                "peak_memory_fraction",
                t.maximum_peak_memory_fraction,
            ),
            "projected_persisted_cache_bytes": _le(
                metrics,
                "projected_persisted_cache_bytes",
                float(t.maximum_persisted_cache_bytes),
            ),
            "projected_transition_count": _eq(
                metrics, "projected_transition_count", t.total_transition_count
            ),
            "test_only_reduced_workload": _eq_zero(
                metrics, "test_only_reduced_workload"
            ),
        }
    )
    result = _gate(
        "preflight",
        checks,
        claim_scope=(
            "immutable-parent, exact-capture, and resource readiness for the "
            "one-image exact K=512 split chain"
        ),
    )
    result["parent_scope_valid"] = int(
        all(
            checks[name]["passed"]
            for name in (
                "parent_provenance_pass",
                "multipath_kernel_gate_pass",
                "multipath_target_gate_pass",
                "multipath_decision_pass",
                "strang_power_failure_preserved_pass",
                "haar_power_only_failure_pass",
                "source_image_hash_pass",
                "source_image_npz_hash_pass",
                "mixed_target_hash_pass",
                "parents_no_training_pass",
                "parents_no_reverse_sampling_pass",
                "parent_registries_pass",
                "source_binding_pass",
            )
        )
    )
    result["model_input_contract_valid"] = int(
        checks["future_model_input_contract_pass"]["passed"]
        and checks["model_input_schema_firewall_pass"]["passed"]
    )
    result["cache_generation_authorized"] = int(result["passed"])
    result["thresholds"] = t.to_dict()
    return result


_FORBIDDEN_CACHE_COUNTS = (
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


def evaluate_learnability_cache(
    metrics: Mapping[str, Any],
    *,
    split: str | None = None,
    thresholds: JacobiRBLearnabilityThresholds | None = None,
) -> dict[str, Any]:
    """Gate one exact whole-path cache split.

    ``split`` is explicit whenever possible.  Reading it from the metrics is
    supported for artifact replay, but an absent or unknown role fails closed.
    """

    t = thresholds or JacobiRBLearnabilityThresholds()
    role = str(split if split is not None else metrics.get("split", "")).lower()
    expected_one = (
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
    checks = {name: _eq_one(metrics, name) for name in expected_one}
    checks.update(
        {
            "split": _check(
                role,
                "in",
                ["train", "validation", "confirmation"],
                role in {"train", "validation", "confirmation"},
            ),
            "path_count": _eq(metrics, "path_count", t.paths_per_split),
            "outer_steps": _eq(metrics, "outer_steps", t.outer_steps),
            "steps_per_shard": _eq(metrics, "steps_per_shard", t.steps_per_shard),
            "transition_count": _eq(
                metrics, "transition_count", t.transitions_per_split
            ),
            "sample_count": _eq(metrics, "sample_count", t.samples_per_split),
            "selected_outer_steps": _check(
                metrics.get("selected_outer_steps"),
                "==",
                list(t.selected_outer_steps),
                isinstance(metrics.get("selected_outer_steps"), (list, tuple))
                and tuple(metrics["selected_outer_steps"]) == t.selected_outer_steps,
            ),
            "phases_per_selected_step": _eq(
                metrics, "phases_per_selected_step", t.phases_per_step
            ),
            "certificate_fraction": _eq(metrics, "certificate_fraction", 1.0),
            "maximum_mass_error": _le(
                metrics, "maximum_mass_error", t.maximum_mass_error
            ),
            "persisted_cache_bytes": _le(
                metrics,
                "persisted_cache_bytes",
                float(t.maximum_persisted_cache_bytes),
            ),
            **{
                name: _eq_zero(metrics, name)
                for name in _FORBIDDEN_CACHE_COUNTS
            },
        }
    )
    if role in {"train", "validation"}:
        checks["confirmation_absent_pass"] = _eq_one(
            metrics, "confirmation_absent_pass"
        )
    elif role == "confirmation":
        checks["selected_model_seal_pass"] = _eq_one(
            metrics, "selected_model_seal_pass"
        )
        checks["confirmation_opened_once_pass"] = _eq_one(
            metrics, "confirmation_opened_once_pass"
        )
        checks["confirmation_path_plan_unchanged_pass"] = _eq_one(
            metrics, "confirmation_path_plan_unchanged_pass"
        )
    if "total_persisted_cache_bytes" in metrics:
        checks["total_persisted_cache_bytes"] = _le(
            metrics,
            "total_persisted_cache_bytes",
            float(t.maximum_persisted_cache_bytes),
        )

    result = _gate(
        f"{role or 'unknown'}_cache",
        checks,
        claim_scope=f"exact binary64 {role or 'unknown'} cache for {CLAIM_SCOPE}",
    )
    result["split"] = role
    result["numerically_valid"] = int(
        checks["certificate_fraction"]["passed"]
        and checks["maximum_mass_error"]["passed"]
        and all(checks[name]["passed"] for name in _FORBIDDEN_CACHE_COUNTS)
    )
    result["model_input_contract_valid"] = int(
        checks["model_input_schema_firewall_pass"]["passed"]
        and checks["input_label_schema_separation_pass"]["passed"]
    )
    result["teacher_training_authorized"] = int(
        role in {"train", "validation"} and result["passed"]
    )
    result["thresholds"] = t.to_dict()
    return result


def evaluate_learnability_teacher(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBLearnabilityThresholds | None = None,
) -> dict[str, Any]:
    """Gate the mandatory exactly representable synthetic teacher."""

    t = thresholds or JacobiRBLearnabilityThresholds()
    teacher_mse = _lookup(
        metrics, "validation_teacher_mse", "teacher_validation_mse"
    )
    baseline_mse = _lookup(
        metrics,
        "validation_metadata_baseline_mse",
        "metadata_baseline_validation_mse",
    )
    relative = (
        float(teacher_mse) / float(baseline_mse)
        if _finite(teacher_mse)
        and _finite(baseline_mse)
        and float(baseline_mse) > 0.0
        else float("nan")
    )
    checks = {
        name: _eq_one(metrics, name)
        for name in (
            "training_complete_pass",
            "all_losses_finite_pass",
            "same_pipeline_pass",
            "selected_checkpoint_replay_hash_pass",
            "model_input_schema_firewall_pass",
            "training_only_scale_pass",
            "no_target_modification_pass",
        )
    }
    checks.update(
        {
            "validation_path_count": _check(
                _lookup(metrics, "validation_path_count", default=None),
                "==",
                t.validation_path_count,
                _lookup(metrics, "validation_path_count", default=None)
                == t.validation_path_count,
            ),
            "paths_beating_metadata_baseline": _check(
                _lookup(
                    metrics,
                    "paths_beating_metadata_baseline",
                    "teacher_paths_beating_metadata_baseline",
                    default=None,
                ),
                "==",
                t.validation_path_count,
                _lookup(
                    metrics,
                    "paths_beating_metadata_baseline",
                    "teacher_paths_beating_metadata_baseline",
                    default=None,
                )
                == t.validation_path_count,
            ),
            "validation_teacher_relative_mse": _check(
                relative,
                "<=",
                t.teacher_relative_mse,
                _finite(relative) and 0.0 <= relative <= t.teacher_relative_mse,
            ),
        }
    )
    result = _gate(
        "teacher",
        checks,
        claim_scope="synthetic optimization-pipeline control using permitted inputs only",
    )
    result["physical_training_authorized"] = int(result["passed"])
    result["model_input_contract_valid"] = int(
        checks["model_input_schema_firewall_pass"]["passed"]
    )
    result["thresholds"] = t.to_dict()
    return result


def evaluate_learnability_physical(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBLearnabilityThresholds | None = None,
) -> dict[str, Any]:
    """Gate physical training completion and validation-only checkpoint sealing."""

    t = thresholds or JacobiRBLearnabilityThresholds()
    target_scale = _lookup(metrics, "target_scale", "global_target_scale")
    selected_update = _lookup(metrics, "selected_update", default=None)
    checks = {
        name: _eq_one(metrics, name)
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
    }
    checks.update(
        {
            "model_seed_count": _eq(metrics, "model_seed_count", 3),
            "validation_path_count": _eq(
                metrics, "validation_path_count", t.validation_path_count
            ),
            "target_scale": _check(
                target_scale,
                ">",
                0.0,
                _finite(target_scale) and float(target_scale) > 0.0,
            ),
            "selected_update": _check(
                selected_update,
                "in",
                [0, t.maximum_updates],
                _integer(selected_update)
                and 0 <= int(selected_update) <= t.maximum_updates,
            ),
        }
    )
    result = _gate(
        "physical",
        checks,
        claim_scope=(
            "finite validation-only selection and immutable sealing of an exact "
            "K=512 split-chain Rao--Blackwell regressor"
        ),
    )
    result["physical_training_performed"] = int(
        _one(metrics.get("physical_training_performed"))
        or _one(metrics.get("training_complete_pass"))
    )
    result["confirmation_generation_authorized"] = int(result["passed"])
    result["model_input_contract_valid"] = int(
        checks["model_input_schema_firewall_pass"]["passed"]
    )
    result["thresholds"] = t.to_dict()
    return result


def evaluate_learnability_confirmation(
    metrics: Mapping[str, Any],
    *,
    confirmation_cache_gate: Mapping[str, Any] | bool | int | None = None,
    thresholds: JacobiRBLearnabilityThresholds | None = None,
) -> dict[str, Any]:
    """Gate the single sealed eight-path confirmation evaluation."""

    t = thresholds or JacobiRBLearnabilityThresholds()
    raw_improvements = _lookup(
        metrics,
        "path_metadata_minus_model_mse",
        "path_improvements",
        "D_i",
        default=(),
    )
    improvements: tuple[float, ...]
    if isinstance(raw_improvements, Sequence) and not isinstance(
        raw_improvements, (str, bytes, bytearray)
    ):
        try:
            improvements = tuple(float(value) for value in raw_improvements)
        except (TypeError, ValueError):
            improvements = ()
    else:
        improvements = ()
    all_positive = (
        len(improvements) == t.confirmation_path_count
        and all(math.isfinite(value) and value > 0.0 for value in improvements)
    )
    model_mse = _lookup(metrics, "aggregate_model_mse")
    zero_mse = _lookup(metrics, "aggregate_zero_mse")
    checks = {
        name: _eq_one(metrics, name)
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
    }
    checks.update(
        {
            "confirmation_cache_gate": _check(
                int(_passed(confirmation_cache_gate)),
                "==",
                1,
                _passed(confirmation_cache_gate),
            ),
            "confirmation_path_count": _eq(
                metrics, "confirmation_path_count", t.confirmation_path_count
            ),
            "all_path_improvements_strictly_positive": _check(
                list(improvements),
                ">",
                0.0,
                all_positive,
            ),
            "aggregate_model_beats_zero": _strict_lt(
                {
                    "aggregate_model_mse": model_mse,
                    "aggregate_zero_mse": zero_mse,
                },
                "aggregate_model_mse",
                "aggregate_zero_mse",
            ),
        }
    )
    result = _gate(
        "confirmation",
        checks,
        claim_scope=CLAIM_SCOPE,
        physical_training_performed=1,
    )
    result.update(
        path_sign_count=sum(
            math.isfinite(value) and value > 0.0 for value in improvements
        ),
        one_sided_sign_test_p_value=(
            2.0 ** (-t.confirmation_path_count) if all_positive else None
        ),
        exact_k512_split_chain_rb_label_learnable=int(result["passed"]),
        larger_exact_discrete_chain_training_planning_authorized=int(
            result["passed"]
        ),
        model_input_contract_valid=int(
            checks["model_input_schema_firewall_pass"]["passed"]
        ),
        cache_valid=int(checks["confirmation_cache_gate"]["passed"]),
        thresholds=t.to_dict(),
    )
    return result


def _failed_names(gate: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(gate, Mapping):
        return set()
    raw = gate.get("subchecks")
    if not isinstance(raw, Mapping):
        return set()
    return {
        str(name)
        for name, check in raw.items()
        if not isinstance(check, Mapping) or not _one(check.get("passed"))
    }


_PARENT_PREFLIGHT_CHECKS = frozenset(
    {
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
        "parents_no_training_pass",
        "parents_no_reverse_sampling_pass",
        "parent_registries_pass",
        "source_binding_pass",
    }
)
_CONTRACT_CHECKS = frozenset(
    {
        "future_model_input_contract_pass",
        "model_input_schema_firewall_pass",
        "input_label_schema_separation_pass",
    }
)


def decide_learnability_workflow(
    *,
    provenance: Mapping[str, Any] | bool | int,
    preflight_gate: Mapping[str, Any] | None,
    train_cache_gate: Mapping[str, Any] | None,
    validation_cache_gate: Mapping[str, Any] | None,
    teacher_gate: Mapping[str, Any] | None,
    physical_gate: Mapping[str, Any] | None,
    confirmation_cache_gate: Mapping[str, Any] | None,
    confirmation_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Produce exactly one closed outcome with fail-closed precedence."""

    if not _passed(provenance):
        decision: JacobiRBLearnabilityDecision | None = (
            JacobiRBLearnabilityDecision.PARENT_SCOPE_INVALID
        )
        action = "repair the immutable parent/source binding"
    elif _status(preflight_gate) != "evaluated":
        decision = None
        action = "run immutable-parent and capture preflight"
    elif not _passed(preflight_gate):
        failed = _failed_names(preflight_gate)
        if failed & _CONTRACT_CHECKS:
            decision = JacobiRBLearnabilityDecision.MODEL_INPUT_CONTRACT_INVALID
            action = "repair and re-audit the later-state-only input firewall"
        elif not failed or failed & _PARENT_PREFLIGHT_CHECKS:
            decision = JacobiRBLearnabilityDecision.PARENT_SCOPE_INVALID
            action = "complete or repair immutable-parent preflight"
        else:
            decision = JacobiRBLearnabilityDecision.EXACT_CACHE_INVALID
            action = "repair exact capture, path planning, or resource feasibility"
    elif any(
        _status(gate) != "evaluated"
        for gate in (train_cache_gate, validation_cache_gate)
    ):
        decision = None
        action = "generate the exact train and validation caches"
    elif any(not _passed(gate) for gate in (train_cache_gate, validation_cache_gate)):
        failed = _failed_names(train_cache_gate) | _failed_names(validation_cache_gate)
        if failed & _CONTRACT_CHECKS:
            decision = JacobiRBLearnabilityDecision.MODEL_INPUT_CONTRACT_INVALID
            action = "repair the physically separated model-input cache"
        else:
            decision = JacobiRBLearnabilityDecision.EXACT_CACHE_INVALID
            action = "repair or complete the exact train/validation caches"
    elif _status(teacher_gate) != "evaluated":
        decision = None
        action = "run the mandatory synthetic optimization control"
    elif not _passed(teacher_gate):
        if _failed_names(teacher_gate) & _CONTRACT_CHECKS:
            decision = JacobiRBLearnabilityDecision.MODEL_INPUT_CONTRACT_INVALID
            action = "repair the permitted-input training path"
        else:
            decision = JacobiRBLearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID
            action = "repair the synthetic-teacher optimization pipeline"
    elif _status(physical_gate) != "evaluated":
        decision = None
        action = "train and seal the physical Rao--Blackwell regressor"
    elif not _passed(physical_gate):
        if _failed_names(physical_gate) & _CONTRACT_CHECKS:
            decision = JacobiRBLearnabilityDecision.MODEL_INPUT_CONTRACT_INVALID
            action = "repair the permitted-input physical training path"
        else:
            decision = JacobiRBLearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID
            action = "repair physical optimization, validation selection, or sealing"
    elif _status(confirmation_cache_gate) != "evaluated":
        decision = None
        action = "open the sealed confirmation cache exactly once"
    elif not _passed(confirmation_cache_gate):
        failed = _failed_names(confirmation_cache_gate)
        if failed & _CONTRACT_CHECKS:
            decision = JacobiRBLearnabilityDecision.MODEL_INPUT_CONTRACT_INVALID
            action = "repair the confirmation input firewall"
        else:
            decision = JacobiRBLearnabilityDecision.EXACT_CACHE_INVALID
            action = "repair or complete the sealed exact confirmation cache"
    elif _status(confirmation_gate) != "evaluated":
        decision = None
        action = "complete the single sealed confirmation evaluation"
    elif not _passed(confirmation_gate):
        failed = _failed_names(confirmation_gate)
        if "confirmation_cache_gate" in failed:
            decision = JacobiRBLearnabilityDecision.EXACT_CACHE_INVALID
            action = "repair the sealed exact confirmation cache"
        elif failed & _CONTRACT_CHECKS:
            decision = JacobiRBLearnabilityDecision.MODEL_INPUT_CONTRACT_INVALID
            action = "repair the confirmation model-input firewall"
        elif failed & {
            "selected_model_hash_pass",
            "model_config_hash_pass",
            "metadata_baseline_hash_pass",
            "path_plan_hash_pass",
            "confirmation_opened_once_pass",
            "confirmation_paths_not_replaced_pass",
            "confirmation_paths_not_added_pass",
        }:
            decision = JacobiRBLearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID
            action = "repair checkpoint sealing or confirmation isolation"
        else:
            decision = (
                JacobiRBLearnabilityDecision.NO_DETECTABLE_ONE_IMAGE_CONDITIONAL_SIGNAL
            )
            action = "retain the sealed no-signal result without adaptive escalation"
    else:
        decision = (
            JacobiRBLearnabilityDecision.EXACT_K512_SPLIT_CHAIN_RB_LABEL_LEARNABLE
        )
        action = "plan a separately named larger exact-discrete-chain training study"

    success = (
        decision
        is JacobiRBLearnabilityDecision.EXACT_K512_SPLIT_CHAIN_RB_LABEL_LEARNABLE
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated" if decision is not None else "pending",
        "decision": None if decision is None else decision.value,
        "claim_scope": CLAIM_SCOPE,
        "recommended_next_action": action,
        "larger_exact_discrete_chain_training_planning_authorized": int(success),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
        "physical_training_performed": int(
            isinstance(physical_gate, Mapping)
            and int(physical_gate.get("physical_training_performed", 0)) == 1
        ),
    }


def evaluate_learnability_workflow(
    *,
    provenance: Mapping[str, Any] | bool | int,
    preflight_gate: Mapping[str, Any] | None = None,
    train_cache_gate: Mapping[str, Any] | None = None,
    validation_cache_gate: Mapping[str, Any] | None = None,
    teacher_gate: Mapping[str, Any] | None = None,
    physical_gate: Mapping[str, Any] | None = None,
    confirmation_cache_gate: Mapping[str, Any] | None = None,
    confirmation_gate: Mapping[str, Any] | None = None,
    require_gate: str = "none",
    thresholds: JacobiRBLearnabilityThresholds | None = None,
) -> dict[str, Any]:
    """Build the cumulative workflow gate and its closed decision."""

    if require_gate not in REQUIRED_GATES:
        raise ValueError(f"unknown required gate: {require_gate}")
    components = {
        "preflight": dict(
            preflight_gate or not_evaluated_gate("preflight", "not run")
        ),
        "train_cache": dict(
            train_cache_gate or not_evaluated_gate("train_cache", "not run")
        ),
        "validation_cache": dict(
            validation_cache_gate
            or not_evaluated_gate("validation_cache", "not run")
        ),
        "teacher": dict(teacher_gate or not_evaluated_gate("teacher", "not run")),
        "physical": dict(
            physical_gate or not_evaluated_gate("physical", "not run")
        ),
        "confirmation_cache": dict(
            confirmation_cache_gate
            or not_evaluated_gate("confirmation_cache", "not run")
        ),
        "confirmation": dict(
            confirmation_gate
            or not_evaluated_gate("confirmation", "not run")
        ),
    }
    required_components = {
        "none": (),
        "preflight": ("preflight",),
        "cache": ("preflight", "train_cache", "validation_cache"),
        "train": (
            "preflight",
            "train_cache",
            "validation_cache",
            "teacher",
            "physical",
        ),
        "confirm": tuple(components),
    }[require_gate]
    provenance_pass = _passed(provenance)
    required_pass = provenance_pass and all(
        _passed(components[name]) for name in required_components
    )
    decision = decide_learnability_workflow(
        provenance=provenance,
        preflight_gate=components["preflight"],
        train_cache_gate=components["train_cache"],
        validation_cache_gate=components["validation_cache"],
        teacher_gate=components["teacher"],
        physical_gate=components["physical"],
        confirmation_cache_gate=components["confirmation_cache"],
        confirmation_gate=components["confirmation"],
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "claim_scope": CLAIM_SCOPE,
        "required_gate": require_gate,
        "required_components": list(required_components),
        "required_gate_pass": int(required_pass),
        "passed": int(required_pass),
        "components": components,
        "decision": decision,
        "thresholds": (thresholds or JacobiRBLearnabilityThresholds()).to_dict(),
        "larger_exact_discrete_chain_training_planning_authorized": int(
            decision["larger_exact_discrete_chain_training_planning_authorized"]
        ),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
        "physical_training_performed": int(
            decision["physical_training_performed"]
        ),
    }


# Concise aliases for callers and tests.
evaluate_preflight_gate = evaluate_learnability_preflight
evaluate_cache_gate = evaluate_learnability_cache
evaluate_teacher_gate = evaluate_learnability_teacher
evaluate_physical_gate = evaluate_learnability_physical
evaluate_confirmation_gate = evaluate_learnability_confirmation
evaluate_workflow_gate = evaluate_learnability_workflow


__all__ = [
    "CLAIM_SCOPE",
    "JacobiRBLearnabilityDecision",
    "JacobiRBLearnabilityThresholds",
    "LearnabilityDecision",
    "LearnabilityThresholds",
    "NO_CLAIM_AUTHORIZATION",
    "NO_WORK",
    "REQUIRED_GATES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGES",
    "decide_learnability_workflow",
    "evaluate_cache_gate",
    "evaluate_confirmation_gate",
    "evaluate_learnability_cache",
    "evaluate_learnability_confirmation",
    "evaluate_learnability_physical",
    "evaluate_learnability_preflight",
    "evaluate_learnability_teacher",
    "evaluate_learnability_workflow",
    "evaluate_physical_gate",
    "evaluate_preflight_gate",
    "evaluate_teacher_gate",
    "evaluate_workflow_gate",
    "execution_failed_gate",
    "not_evaluated_gate",
]
