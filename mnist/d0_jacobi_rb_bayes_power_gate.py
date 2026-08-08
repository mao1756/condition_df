"""Fail-closed gates for the exact noisy-Jacobi Bayes calibration.

This module is deliberately independent of filesystems, CUDA, trainers, and
samplers.  It adjudicates a synthetic, analytically soluble control around the
same exact Jacobi transition and Rao--Blackwell label used by the immutable
one-image experiment.

Passing these gates calibrates only the ability to detect a known conditional
mean.  It does not reinterpret the sealed physical no-signal result and does
not authorize physical training, refinement, reconstruction, or sampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


SCHEMA = "experiment12-d0-jacobi-rb-bayes-power-calibration-gate"
SCHEMA_VERSION = 1
STAGES = ("preflight", "cache", "train", "confirm", "report", "all")
REQUIRED_GATES = ("none", "preflight", "cache", "train", "controls")
CLAIM_SCOPE = (
    "synthetic calibration of noisy exact-Jacobi conditional-mean detection"
)

NO_CLAIM_AUTHORIZATION = {
    "physical_training_authorized": 0,
    "full_dataset_training_authorized": 0,
    "production_refinement_authorized": 0,
    "state_dependent_strang_refinement_established": 0,
    "unsplit_generator_approximation_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
    "reverse_sampling_authorized": 0,
    "sampling_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "known_prior_claim_authorized": 0,
}
NO_WORK = {
    "physical_training_performed": 0,
    "production_refinement_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


class BayesPowerDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    ANALYTIC_BAYES_IDENTITY_INVALID = "analytic_bayes_identity_invalid"
    EXACT_CONTROL_CACHE_INVALID = "exact_control_cache_invalid"
    ORACLE_PANEL_UNDERPOWERED = "oracle_panel_underpowered"
    OPTIMIZATION_PIPELINE_INVALID = "optimization_pipeline_invalid"
    NULL_FALSE_DISCOVERY = "null_false_discovery"
    NOISY_BAYES_DETECTION_PIPELINE_CALIBRATED = (
        "noisy_bayes_detection_pipeline_calibrated"
    )


@dataclass(frozen=True)
class BayesPowerThresholds:
    """Frozen production workload and scientific thresholds."""

    root_seed: int = 261_211
    model_seeds: tuple[int, ...] = (261_201, 261_202, 261_203)
    grid_cells: int = 784
    edges_per_phase: int = 392
    selected_outer_step_count: int = 32
    phases_per_step: int = 7
    paths_per_role: int = 8
    samples_per_role: int = 1_792
    transitions_per_role: int = 702_464
    total_transition_count: int = 4_214_784
    maximum_updates: int = 4_000
    maximum_float64_identity_error: float = 1.0e-10
    maximum_cuda_identity_error: float = 2.0e-6
    minimum_oracle_relative_gain: float = 0.01
    minimum_oracle_gain_recovery: float = 0.50

    def __post_init__(self) -> None:
        integer_names = (
            "root_seed",
            "grid_cells",
            "edges_per_phase",
            "selected_outer_step_count",
            "phases_per_step",
            "paths_per_role",
            "samples_per_role",
            "transitions_per_role",
            "total_transition_count",
            "maximum_updates",
        )
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not self.model_seeds
            or len(set(self.model_seeds)) != len(self.model_seeds)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.model_seeds
            )
        ):
            raise ValueError("model_seeds must be distinct positive integers")
        for name in (
            "maximum_float64_identity_error",
            "maximum_cuda_identity_error",
            "minimum_oracle_relative_gain",
            "minimum_oracle_gain_recovery",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def selected_outer_steps(self) -> tuple[int, ...]:
        return tuple(15 + 16 * index for index in range(32))

    @property
    def preconfirmation_transition_count(self) -> int:
        return 4 * self.transitions_per_role

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_seeds"] = list(self.model_seeds)
        value["selected_outer_steps"] = list(self.selected_outer_steps)
        return value


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _status(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return "not_evaluated"
    return str(value.get("evaluation_status", "not_evaluated"))


def _passed(value: Mapping[str, Any] | bool | int | None) -> bool:
    if isinstance(value, Mapping):
        return _status(value) == "evaluated" and _one(value.get("passed"))
    return value is True or (
        isinstance(value, int) and not isinstance(value, bool) and value == 1
    )


def _lookup(metrics: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return default


def _sequence(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return ()
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()
    return result if all(math.isfinite(item) for item in result) else ()


def _check(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _eq_one(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", 1, _one(value))


def _eq_zero(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", 0, _zero(value))


def _eq(metrics: Mapping[str, Any], name: str, threshold: Any) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", threshold, value == threshold)


def _le(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value,
        "<=",
        threshold,
        _finite(value) and 0.0 <= float(value) <= float(threshold),
    )


def _gate(name: str, checks: Mapping[str, Mapping[str, Any]], **fields: Any) -> dict[str, Any]:
    passed = bool(checks) and all(_one(check.get("passed")) for check in checks.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "evaluation_status": "evaluated",
        "claim_scope": CLAIM_SCOPE,
        "subchecks": {str(key): dict(value) for key, value in checks.items()},
        "passed": int(passed),
        **fields,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


def not_evaluated_gate(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": str(stage),
        "evaluation_status": "not_evaluated",
        "reason": str(reason),
        "subchecks": {},
        "passed": 0,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


def execution_failed_gate(
    stage: str,
    *,
    failure_domain: str,
    failure_code: str,
    error_type: str,
    error: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": str(stage),
        "evaluation_status": "execution_failed",
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "failure_domain": str(failure_domain),
        "failure_code": str(failure_code),
        "error_type": str(error_type),
        "error": str(error),
        "subchecks": {},
        "passed": 0,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


def evaluate_bayes_preflight(
    metrics: Mapping[str, Any],
    thresholds: BayesPowerThresholds | None = None,
) -> dict[str, Any]:
    """Gate immutable-parent binding and analytic Bayes identities."""

    t = thresholds or BayesPowerThresholds()
    exact_checks = (
        "parent_provenance_pass",
        "parent_registry_pass",
        "parent_terminal_no_signal_pass",
        "parent_only_aggregate_zero_failure_pass",
        "parent_exact_cache_pass",
        "parent_teacher_pass",
        "parent_optimizer_pass",
        "parent_seal_pass",
        "parent_no_sampling_pass",
        "parent_label_firewall_pass",
        "parent_template_allowlist_pass",
        "source_binding_pass",
        "analytic_normalization_pass",
        "analytic_positive_time_density_pass",
        "analytic_score_pass",
        "analytic_bayes_mean_pass",
        "stationary_null_identity_pass",
        "path_plan_frozen_pass",
        "path_plan_disjoint_pass",
        "path_id_uniqueness_pass",
        "confirmation_absent_pass",
    )
    checks = {name: _eq_one(metrics, name) for name in exact_checks}
    checks.update(
        {
            "root_seed": _eq(metrics, "root_seed", t.root_seed),
            "selected_outer_steps": _eq(
                metrics, "selected_outer_steps", list(t.selected_outer_steps)
            ),
            "model_seeds": _eq(metrics, "model_seeds", list(t.model_seeds)),
            "float64_identity_error": _le(
                metrics,
                "maximum_float64_identity_error",
                t.maximum_float64_identity_error,
            ),
            "cuda_identity_error": _le(
                metrics,
                "maximum_cuda_identity_error",
                t.maximum_cuda_identity_error,
            ),
            "projected_transition_count": _eq(
                metrics,
                "projected_transition_count",
                t.total_transition_count,
            ),
            "test_only_reduced_workload": _eq_zero(
                metrics, "test_only_reduced_workload"
            ),
        }
    )
    result = _gate("preflight", checks)
    result["cache_generation_authorized"] = int(result["passed"])
    result["analytic_identity_valid"] = int(
        checks["analytic_normalization_pass"]["passed"]
        and checks["analytic_positive_time_density_pass"]["passed"]
        and checks["analytic_score_pass"]["passed"]
        and checks["analytic_bayes_mean_pass"]["passed"]
        and checks["stationary_null_identity_pass"]["passed"]
        and checks["float64_identity_error"]["passed"]
        and checks["cuda_identity_error"]["passed"]
    )
    result["thresholds"] = t.to_dict()
    return result


_FORBIDDEN_COUNTS = (
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


def evaluate_bayes_cache(
    metrics: Mapping[str, Any],
    *,
    law: str,
    split: str,
    thresholds: BayesPowerThresholds | None = None,
) -> dict[str, Any]:
    """Gate one exact synthetic law/split cache."""

    if law not in {"teacher", "null"}:
        raise ValueError(f"unknown law: {law}")
    if split not in {"train", "validation", "confirmation"}:
        raise ValueError(f"unknown split: {split}")
    t = thresholds or BayesPowerThresholds()
    exact_checks = (
        "cache_complete_pass",
        "cache_replay_hash_pass",
        "states_finite_pass",
        "targets_finite_pass",
        "oracle_audit_finite_pass",
        "sample_key_join_pass",
        "role_isolation_pass",
        "model_input_schema_firewall_pass",
        "oracle_input_isolation_pass",
        "exact_jacobi_transition_pass",
        "exact_rb_target_pass",
        "whole_cluster_tower_identity_pass",
    )
    checks = {name: _eq_one(metrics, name) for name in exact_checks}
    checks.update(
        {
            "law": _eq(metrics, "law", law),
            "split": _eq(metrics, "split", split),
            "path_count": _eq(metrics, "path_count", t.paths_per_role),
            "sample_count": _eq(metrics, "sample_count", t.samples_per_role),
            "transition_count": _eq(
                metrics, "transition_count", t.transitions_per_role
            ),
            "selected_outer_steps": _eq(
                metrics, "selected_outer_steps", list(t.selected_outer_steps)
            ),
            "certificate_fraction": _check(
                metrics.get("certificate_fraction"),
                "==",
                1.0,
                _finite(metrics.get("certificate_fraction"))
                and float(metrics["certificate_fraction"]) == 1.0,
            ),
        }
    )
    checks.update({name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS})
    if split == "confirmation":
        checks.update(
            {
                "confirmation_seal_pass": _eq_one(
                    metrics, "confirmation_seal_pass"
                ),
                "confirmation_opened_once_pass": _eq_one(
                    metrics, "confirmation_opened_once_pass"
                ),
                "confirmation_plan_unchanged_pass": _eq_one(
                    metrics, "confirmation_plan_unchanged_pass"
                ),
            }
        )
    else:
        checks["confirmation_absent_pass"] = _eq_one(
            metrics, "confirmation_absent_pass"
        )
    result = _gate(f"{law}_{split}_cache", checks, law=law, split=split)
    result["numerically_valid"] = int(result["passed"])
    result["thresholds"] = t.to_dict()
    return result


def evaluate_bayes_cache_set(
    metrics: Mapping[str, Any],
    *,
    cache_gates: Mapping[str, Mapping[str, Any]],
    thresholds: BayesPowerThresholds | None = None,
) -> dict[str, Any]:
    """Gate the four train/validation caches before confirmation is opened."""

    t = thresholds or BayesPowerThresholds()
    expected = (
        "teacher_train",
        "teacher_validation",
        "null_train",
        "null_validation",
    )
    checks = {
        name + "_cache_gate": _check(
            int(_passed(cache_gates.get(name))),
            "==",
            1,
            _passed(cache_gates.get(name)),
        )
        for name in expected
    }
    checks.update(
        {
            "exact_cache_roles": _check(
                sorted(cache_gates),
                "==",
                sorted(expected),
                set(cache_gates) == set(expected),
            ),
            "transition_count": _eq(
                metrics,
                "transition_count",
                t.preconfirmation_transition_count,
            ),
            "confirmation_absent_pass": _eq_one(
                metrics, "confirmation_absent_pass"
            ),
            "role_isolation_pass": _eq_one(metrics, "role_isolation_pass"),
            "training_only_scale_source_pass": _eq_one(
                metrics, "training_only_scale_source_pass"
            ),
        }
    )
    result = _gate(
        "cache",
        checks,
        component_gates={name: dict(cache_gates[name]) for name in expected if name in cache_gates},
    )
    result["synthetic_control_training_authorized"] = int(result["passed"])
    result["thresholds"] = t.to_dict()
    return result


def evaluate_bayes_train(
    metrics: Mapping[str, Any],
    thresholds: BayesPowerThresholds | None = None,
) -> dict[str, Any]:
    """Gate synthetic teacher/null optimization and immutable sealing."""

    t = thresholds or BayesPowerThresholds()
    exact_checks = (
        "teacher_training_complete_pass",
        "null_training_complete_pass",
        "all_six_tasks_complete_pass",
        "all_losses_finite_pass",
        "same_pipeline_pass",
        "training_only_scale_pass",
        "unweighted_mse_objective_pass",
        "no_target_modification_pass",
        "validation_only_selection_pass",
        "analytic_zero_candidate_pass",
        "teacher_nonzero_checkpoint_pass",
        "teacher_checkpoint_hash_pass",
        "null_checkpoint_hash_pass",
        "selected_candidates_frozen_pass",
        "confirmation_gate_definition_frozen_pass",
        "confirmation_absent_pass",
        "model_input_schema_firewall_pass",
        "oracle_input_isolation_pass",
    )
    checks = {name: _eq_one(metrics, name) for name in exact_checks}
    checks.update(
        {
            "model_seed_count": _eq(
                metrics, "model_seed_count", len(t.model_seeds)
            ),
            "model_seeds": _eq(metrics, "model_seeds", list(t.model_seeds)),
            "validation_path_count_per_law": _eq(
                metrics,
                "validation_path_count_per_law",
                t.paths_per_role,
            ),
            "maximum_updates": _eq(
                metrics, "maximum_updates", t.maximum_updates
            ),
            "teacher_target_scale": _check(
                metrics.get("teacher_target_scale"),
                ">",
                0.0,
                _finite(metrics.get("teacher_target_scale"))
                and float(metrics["teacher_target_scale"]) > 0.0,
            ),
            "null_target_scale": _check(
                metrics.get("null_target_scale"),
                ">",
                0.0,
                _finite(metrics.get("null_target_scale"))
                and float(metrics["null_target_scale"]) > 0.0,
            ),
        }
    )
    result = _gate("train", checks)
    result["confirmation_generation_authorized"] = int(result["passed"])
    result["thresholds"] = t.to_dict()
    return result


def _strictly_positive(values: tuple[float, ...], count: int) -> bool:
    return len(values) == count and all(value > 0.0 for value in values)


def evaluate_bayes_confirmation(
    metrics: Mapping[str, Any],
    *,
    teacher_cache_gate: Mapping[str, Any] | bool | int | None = None,
    null_cache_gate: Mapping[str, Any] | bool | int | None = None,
    thresholds: BayesPowerThresholds | None = None,
) -> dict[str, Any]:
    """Gate the single sealed teacher/null confirmation.

    Oracle gain, learned recovery, and the null discovery conjunction are
    recomputed from primitive MSEs and per-path differences.
    """

    t = thresholds or BayesPowerThresholds()
    teacher_oracle_path_gain = _sequence(
        _lookup(
            metrics,
            "teacher_path_zero_minus_oracle_mse",
            "oracle_path_improvements",
            default=(),
        )
    )
    teacher_metadata_gain = _sequence(
        _lookup(
            metrics,
            "teacher_path_metadata_minus_model_mse",
            "teacher_path_improvements",
            default=(),
        )
    )
    null_metadata_gain = _sequence(
        _lookup(
            metrics,
            "null_path_metadata_minus_model_mse",
            "null_path_improvements",
            default=(),
        )
    )
    zero_mse = _lookup(metrics, "teacher_aggregate_zero_mse")
    oracle_mse = _lookup(metrics, "teacher_aggregate_oracle_mse")
    model_mse = _lookup(metrics, "teacher_aggregate_model_mse")
    null_zero_mse = _lookup(metrics, "null_aggregate_zero_mse")
    null_model_mse = _lookup(metrics, "null_aggregate_model_mse")

    denominator = (
        float(zero_mse) - float(oracle_mse)
        if _finite(zero_mse) and _finite(oracle_mse)
        else math.nan
    )
    oracle_relative_gain = (
        denominator / float(zero_mse)
        if _finite(zero_mse) and float(zero_mse) > 0.0 and denominator > 0.0
        else math.nan
    )
    recovery = (
        (float(zero_mse) - float(model_mse)) / denominator
        if _finite(model_mse) and denominator > 0.0
        else math.nan
    )
    teacher_model_beats_zero = (
        _finite(model_mse)
        and _finite(zero_mse)
        and 0.0 <= float(model_mse) < float(zero_mse)
    )
    null_model_beats_zero = (
        _finite(null_model_mse)
        and _finite(null_zero_mse)
        and 0.0 <= float(null_model_mse) < float(null_zero_mse)
    )
    null_all_path_signs = _strictly_positive(
        null_metadata_gain, t.paths_per_role
    )
    null_discovery = null_model_beats_zero and null_all_path_signs

    exact_checks = (
        "predictions_finite_pass",
        "losses_finite_pass",
        "teacher_selected_model_hash_pass",
        "null_selected_model_hash_pass",
        "model_config_hash_pass",
        "metadata_baseline_hash_pass",
        "path_plan_hash_pass",
        "confirmation_opened_once_pass",
        "confirmation_paths_not_replaced_pass",
        "confirmation_paths_not_added_pass",
        "model_input_schema_firewall_pass",
        "oracle_input_isolation_pass",
    )
    checks = {name: _eq_one(metrics, name) for name in exact_checks}
    checks.update(
        {
            "teacher_confirmation_cache_gate": _check(
                int(_passed(teacher_cache_gate)),
                "==",
                1,
                _passed(teacher_cache_gate),
            ),
            "null_confirmation_cache_gate": _check(
                int(_passed(null_cache_gate)),
                "==",
                1,
                _passed(null_cache_gate),
            ),
            "teacher_confirmation_path_count": _eq(
                metrics,
                "teacher_confirmation_path_count",
                t.paths_per_role,
            ),
            "null_confirmation_path_count": _eq(
                metrics, "null_confirmation_path_count", t.paths_per_role
            ),
            "teacher_oracle_beats_zero_all_paths": _check(
                list(teacher_oracle_path_gain),
                ">",
                0.0,
                _strictly_positive(
                    teacher_oracle_path_gain, t.paths_per_role
                ),
            ),
            "teacher_oracle_relative_gain": _check(
                oracle_relative_gain,
                ">=",
                t.minimum_oracle_relative_gain,
                math.isfinite(oracle_relative_gain)
                and oracle_relative_gain >= t.minimum_oracle_relative_gain,
            ),
            "teacher_model_beats_zero": _check(
                {"model_mse": model_mse, "zero_mse": zero_mse},
                "model_mse < zero_mse",
                None,
                teacher_model_beats_zero,
            ),
            "teacher_model_beats_metadata_all_paths": _check(
                list(teacher_metadata_gain),
                ">",
                0.0,
                _strictly_positive(teacher_metadata_gain, t.paths_per_role),
            ),
            "teacher_oracle_gain_recovery": _check(
                recovery,
                ">=",
                t.minimum_oracle_gain_recovery,
                math.isfinite(recovery)
                and recovery >= t.minimum_oracle_gain_recovery,
            ),
            "null_no_false_discovery": _check(
                {
                    "aggregate_model_beats_zero": int(null_model_beats_zero),
                    "all_path_metadata_improvements_positive": int(
                        null_all_path_signs
                    ),
                },
                "conjunction ==",
                0,
                not null_discovery,
            ),
        }
    )
    result = _gate("controls", checks)
    result.update(
        oracle_relative_gain=(
            oracle_relative_gain if math.isfinite(oracle_relative_gain) else None
        ),
        oracle_gain_recovery=recovery if math.isfinite(recovery) else None,
        null_discovery_conjunction=int(null_discovery),
        teacher_oracle_path_sign_count=sum(
            value > 0.0 for value in teacher_oracle_path_gain
        ),
        teacher_metadata_path_sign_count=sum(
            value > 0.0 for value in teacher_metadata_gain
        ),
        null_metadata_path_sign_count=sum(
            value > 0.0 for value in null_metadata_gain
        ),
        noisy_bayes_detection_pipeline_calibrated=int(result["passed"]),
        fresh_physical_witness_planning_authorized=int(result["passed"]),
        thresholds=t.to_dict(),
    )
    return result


_PROVENANCE_PREFLIGHT_CHECKS = frozenset(
    {
        "parent_provenance_pass",
        "parent_registry_pass",
        "parent_terminal_no_signal_pass",
        "parent_only_aggregate_zero_failure_pass",
        "parent_exact_cache_pass",
        "parent_teacher_pass",
        "parent_optimizer_pass",
        "parent_seal_pass",
        "parent_no_sampling_pass",
        "parent_label_firewall_pass",
        "parent_template_allowlist_pass",
        "source_binding_pass",
    }
)
_ANALYTIC_PREFLIGHT_CHECKS = frozenset(
    {
        "analytic_normalization_pass",
        "analytic_positive_time_density_pass",
        "analytic_score_pass",
        "analytic_bayes_mean_pass",
        "stationary_null_identity_pass",
        "float64_identity_error",
        "cuda_identity_error",
    }
)


def _failed_names(gate: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(gate, Mapping):
        return set()
    checks = gate.get("subchecks")
    if not isinstance(checks, Mapping):
        return set()
    return {
        str(name)
        for name, value in checks.items()
        if not isinstance(value, Mapping) or not _one(value.get("passed"))
    }


def decide_bayes_power_workflow(
    *,
    provenance: Mapping[str, Any] | bool | int,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    confirmation_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return one closed outcome, or a pending action for an unrun stage."""

    if not _passed(provenance):
        decision: BayesPowerDecision | None = (
            BayesPowerDecision.CONTROL_PROVENANCE_INVALID
        )
        action = "repair the immutable parent and label-firewall binding"
    elif _status(preflight_gate) == "execution_failed":
        if str(preflight_gate.get("failure_domain")) == "control_provenance":
            decision = BayesPowerDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable parent and label-firewall binding"
        else:
            decision = BayesPowerDecision.ANALYTIC_BAYES_IDENTITY_INVALID
            action = "repair the exact analytic Bayes control"
    elif _status(preflight_gate) != "evaluated":
        decision = None
        action = "run immutable-parent and analytic-identity preflight"
    elif not _passed(preflight_gate):
        failed = _failed_names(preflight_gate)
        if failed & _PROVENANCE_PREFLIGHT_CHECKS:
            decision = BayesPowerDecision.CONTROL_PROVENANCE_INVALID
            action = "repair immutable-parent, source, or label-access provenance"
        else:
            decision = BayesPowerDecision.ANALYTIC_BAYES_IDENTITY_INVALID
            action = "repair the exact analytic Bayes control"
    elif _status(cache_gate) == "execution_failed":
        decision = BayesPowerDecision.EXACT_CONTROL_CACHE_INVALID
        action = "repair exact synthetic cache generation or isolation"
    elif _status(cache_gate) != "evaluated":
        decision = None
        action = "generate the exact synthetic train/validation caches"
    elif not _passed(cache_gate):
        decision = BayesPowerDecision.EXACT_CONTROL_CACHE_INVALID
        action = "repair exact synthetic cache generation or isolation"
    elif _status(train_gate) == "execution_failed":
        decision = BayesPowerDecision.OPTIMIZATION_PIPELINE_INVALID
        action = "repair synthetic optimization or validation-only selection"
    elif _status(train_gate) != "evaluated":
        decision = None
        action = "train and seal the teacher and stationary-null candidates"
    elif not _passed(train_gate):
        decision = BayesPowerDecision.OPTIMIZATION_PIPELINE_INVALID
        action = "repair synthetic optimization or validation-only selection"
    elif _status(confirmation_gate) == "execution_failed":
        decision = BayesPowerDecision.OPTIMIZATION_PIPELINE_INVALID
        action = "repair noisy conditional-mean recovery without physical data"
    elif _status(confirmation_gate) != "evaluated":
        decision = None
        action = "open the sealed teacher/null confirmation once"
    elif not _passed(confirmation_gate):
        failed = _failed_names(confirmation_gate)
        if failed & {
            "teacher_oracle_beats_zero_all_paths",
            "teacher_oracle_relative_gain",
        }:
            decision = BayesPowerDecision.ORACLE_PANEL_UNDERPOWERED
            action = "retain the sealed underpowered oracle panel"
        elif "null_no_false_discovery" in failed:
            decision = BayesPowerDecision.NULL_FALSE_DISCOVERY
            action = "repair the detection or checkpoint-selection pipeline"
        else:
            decision = BayesPowerDecision.OPTIMIZATION_PIPELINE_INVALID
            action = "repair noisy conditional-mean recovery without physical data"
    else:
        decision = BayesPowerDecision.NOISY_BAYES_DETECTION_PIPELINE_CALIBRATED
        action = "plan a separately named fresh physical-signal witness"

    success = (
        decision is BayesPowerDecision.NOISY_BAYES_DETECTION_PIPELINE_CALIBRATED
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated" if decision is not None else "pending",
        "decision": None if decision is None else decision.value,
        "claim_scope": CLAIM_SCOPE,
        "recommended_next_action": action,
        "fresh_physical_witness_planning_authorized": int(success),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


def evaluate_bayes_power_workflow(
    *,
    provenance: Mapping[str, Any] | bool | int,
    preflight_gate: Mapping[str, Any] | None = None,
    cache_gate: Mapping[str, Any] | None = None,
    train_gate: Mapping[str, Any] | None = None,
    confirmation_gate: Mapping[str, Any] | None = None,
    require_gate: str = "none",
    thresholds: BayesPowerThresholds | None = None,
) -> dict[str, Any]:
    if require_gate not in REQUIRED_GATES:
        raise ValueError(f"unknown required gate: {require_gate}")
    components = {
        "preflight": dict(
            preflight_gate or not_evaluated_gate("preflight", "not run")
        ),
        "cache": dict(cache_gate or not_evaluated_gate("cache", "not run")),
        "train": dict(train_gate or not_evaluated_gate("train", "not run")),
        "controls": dict(
            confirmation_gate or not_evaluated_gate("controls", "not run")
        ),
    }
    required = {
        "none": (),
        "preflight": ("preflight",),
        "cache": ("preflight", "cache"),
        "train": ("preflight", "cache", "train"),
        "controls": ("preflight", "cache", "train", "controls"),
    }[require_gate]
    required_pass = _passed(provenance) and all(
        _passed(components[name]) for name in required
    )
    decision = decide_bayes_power_workflow(
        provenance=provenance,
        preflight_gate=components["preflight"],
        cache_gate=components["cache"],
        train_gate=components["train"],
        confirmation_gate=components["controls"],
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_components": list(required),
        "required_gate_pass": int(required_pass),
        "passed": int(required_pass),
        "components": components,
        "decision": decision,
        "thresholds": (thresholds or BayesPowerThresholds()).to_dict(),
        "fresh_physical_witness_planning_authorized": int(
            decision["fresh_physical_witness_planning_authorized"]
        ),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


# Concise aliases for the orchestration CLI and focused tests.
evaluate_preflight_gate = evaluate_bayes_preflight
evaluate_cache_gate = evaluate_bayes_cache
evaluate_cache_set_gate = evaluate_bayes_cache_set
evaluate_train_gate = evaluate_bayes_train
evaluate_confirmation_gate = evaluate_bayes_confirmation
evaluate_workflow_gate = evaluate_bayes_power_workflow
decide_workflow = decide_bayes_power_workflow


__all__ = [
    "BayesPowerDecision",
    "BayesPowerThresholds",
    "CLAIM_SCOPE",
    "NO_CLAIM_AUTHORIZATION",
    "NO_WORK",
    "REQUIRED_GATES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGES",
    "decide_bayes_power_workflow",
    "decide_workflow",
    "evaluate_bayes_cache",
    "evaluate_bayes_cache_set",
    "evaluate_bayes_confirmation",
    "evaluate_bayes_power_workflow",
    "evaluate_bayes_preflight",
    "evaluate_bayes_train",
    "evaluate_cache_gate",
    "evaluate_cache_set_gate",
    "evaluate_confirmation_gate",
    "evaluate_preflight_gate",
    "evaluate_train_gate",
    "evaluate_workflow_gate",
    "execution_failed_gate",
    "not_evaluated_gate",
]
