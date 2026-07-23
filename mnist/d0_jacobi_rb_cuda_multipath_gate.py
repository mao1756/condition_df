"""Fail-closed gates for exact multi-path Jacobi RB execution scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_jacobi_rb_cuda_gate import evaluate_jacobi_rb_cuda_target


SCHEMA = "experiment12-d0-jacobi-rb-cuda-multipath-gate"
SCHEMA_VERSION = 1
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


@dataclass(frozen=True)
class JacobiRBMultipathThresholds:
    parent_record_count: int = 219
    path_count: int = 64
    projection_group_sizes: tuple[int, ...] = (10, 10, 10, 10, 10, 10, 4)
    validation_group_sizes: tuple[int, ...] = (10, 4)
    edge_count_per_path_phase: int = 392
    maximum_launch_lanes: int = 4096
    restart_steps_per_shard: int = 8
    pilot_outer_steps: int = 64
    pilot_repeats_per_group: int = 3
    full_outer_steps: int = 512
    full_repeats_per_group: int = 3
    b10_transitions_per_repeat: int = 14_049_280
    b4_transitions_per_repeat: int = 5_619_712
    total_full_benchmark_transitions: int = 59_006_976
    projected_transition_count: int = 89_915_392
    minimum_rate: float = 1_300.0
    maximum_projected_hours: float = 20.0
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_cost_fraction: float = 0.10
    maximum_peak_memory_fraction: float = 0.80
    maximum_cuda_mass_error: float = 2.0e-6
    maximum_cuda_kernel_error: float = 2.0e-6

    def __post_init__(self) -> None:
        expected: dict[str, Any] = {
            "parent_record_count": 219,
            "path_count": 64,
            "projection_group_sizes": (10, 10, 10, 10, 10, 10, 4),
            "validation_group_sizes": (10, 4),
            "edge_count_per_path_phase": 392,
            "maximum_launch_lanes": 4096,
            "restart_steps_per_shard": 8,
            "pilot_outer_steps": 64,
            "pilot_repeats_per_group": 3,
            "full_outer_steps": 512,
            "full_repeats_per_group": 3,
            "b10_transitions_per_repeat": 14_049_280,
            "b4_transitions_per_repeat": 5_619_712,
            "total_full_benchmark_transitions": 59_006_976,
            "projected_transition_count": 89_915_392,
            "minimum_rate": 1_300.0,
            "maximum_projected_hours": 20.0,
            "maximum_fallback_fraction": 1.0e-4,
            "maximum_fallback_cost_fraction": 0.10,
            "maximum_peak_memory_fraction": 0.80,
            "maximum_cuda_mass_error": 2.0e-6,
            "maximum_cuda_kernel_error": 2.0e-6,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is frozen at {value}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["projection_group_sizes"] = list(self.projection_group_sizes)
        value["validation_group_sizes"] = list(self.validation_group_sizes)
        return value


class JacobiRBMultipathDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    MULTIPATH_CUDA_RUNTIME_INVALID = "multipath_cuda_runtime_invalid"
    MULTIPATH_SCHEDULER_EQUIVALENCE_INVALID = (
        "multipath_scheduler_equivalence_invalid"
    )
    MULTIPATH_PERFORMANCE_PILOT_UNRESOLVED = (
        "multipath_performance_pilot_unresolved"
    )
    MULTIPATH_KERNEL_NUMERICALLY_UNRESOLVED = (
        "multipath_kernel_numerically_unresolved"
    )
    MULTIPATH_KERNEL_COMPUTATIONALLY_INFEASIBLE = (
        "multipath_kernel_computationally_infeasible"
    )
    JACOBI_RB_TARGET_INVALID = "jacobi_rb_target_invalid"
    EXACT_JACOBI_RB_MULTIPATH_KERNEL_AND_TARGET_FEASIBLE = (
        "exact_jacobi_rb_multipath_kernel_and_target_feasible"
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _status(gate: Mapping[str, Any] | None) -> str:
    if not isinstance(gate, Mapping):
        return "not_evaluated"
    return str(gate.get("evaluation_status", "not_evaluated"))


def _passed(gate: bool | int | Mapping[str, Any] | None) -> bool:
    if isinstance(gate, Mapping):
        return _status(gate) == "evaluated" and _one(gate.get("passed"))
    return gate is True or (isinstance(gate, int) and gate == 1)


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


def _equal(metrics: Mapping[str, Any], name: str, threshold: Any) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", threshold, value == threshold)


def _sequence_equal(
    metrics: Mapping[str, Any], name: str, threshold: Sequence[int]
) -> dict[str, Any]:
    value = metrics.get(name)
    valid = isinstance(value, (list, tuple)) and tuple(value) == tuple(threshold)
    return _check(value, "==", list(threshold), valid)


def _le(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value, "<=", threshold,
        _finite(value) and 0.0 <= float(value) <= threshold,
    )


def _ge(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, ">=", threshold, _finite(value) and float(value) >= threshold)


def _gate(
    name: str,
    claim_scope: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_status: str = "evaluated",
) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in checks.items()}
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "claim_scope": claim_scope,
        "evaluation_status": evaluation_status,
        "subchecks": normalized,
        "passed": int(
            evaluation_status == "evaluated"
            and bool(normalized)
            and all(_one(value.get("passed")) for value in normalized.values())
        ),
        **NO_WORK,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    result = _gate(name, "not evaluated", {}, evaluation_status="not_evaluated")
    result["reason"] = str(reason)
    return result


def evaluate_multipath_preflight(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBMultipathThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or JacobiRBMultipathThresholds()
    checks = {
        name: _eq_one(metrics, name)
        for name in (
            "control_provenance_pass",
            "parent_certificate_pass",
            "parent_kernel_numerically_valid_pass",
            "parent_single_path_resource_failure_pass",
            "parent_target_not_evaluated_pass",
            "seven_parent_sources_immutable_pass",
            "frozen_runtime_match_pass",
            "cuda_backend_replay_pass",
            "parent_cuda_source_hash_pass",
            "parent_cubin_hash_pass",
            "parent_compile_options_hash_pass",
            "canonical_id_uniqueness_pass",
            "canonical_id_group_order_invariance_pass",
            "canonical_full_id_field_proof_pass",
            "path_zero_parent_replay_pass",
            "serial_batch_parity_pass",
            "phase_order_pass",
            "phase_by_phase_equivalence_pass",
            "group_order_invariance_pass",
            "fresh_b4_parity_pass",
            "path_permutation_invariance_pass",
            "no_cross_path_write_pass",
            "resume_replay_pass",
            "state_updates_device_resident_pass",
            "evolving_state_host_roundtrip_pass",
        )
    }
    checks.update({
        "parent_record_count": _equal(metrics, "parent_record_count", t.parent_record_count),
        "path_count": _equal(metrics, "path_count", t.path_count),
        "projection_group_sizes": _sequence_equal(
            metrics, "projection_group_sizes", t.projection_group_sizes
        ),
        "validation_group_sizes": _sequence_equal(
            metrics, "validation_group_sizes", t.validation_group_sizes
        ),
        "restart_steps_per_shard": _equal(
            metrics, "restart_steps_per_shard", t.restart_steps_per_shard
        ),
        "canonical_full_id_plan_count": _equal(
            metrics, "canonical_full_id_plan_count", t.projected_transition_count
        ),
        "maximum_cuda_launch_lanes": _le(
            metrics, "maximum_cuda_launch_lanes", float(t.maximum_launch_lanes)
        ),
        "mass_error": _le(metrics, "mass_error", t.maximum_cuda_mass_error),
        **{
            name: _eq_zero(metrics, name)
            for name in (
                "uncertified_count", "fallback_count", "resource_cap_count",
                "invalid_density_count", "approximation_count", "correction_count",
                "floor_count", "limiter_count", "renormalization_count",
                "nonfinite_count", "transition_id_collision_count",
                "path_hash_mismatch_count", "state_hash_mismatch_count",
            )
        },
    })
    result = _gate(
        "jacobi_rb_cuda_multipath_preflight",
        "immutable parent and exact serial/multi-path scheduling equivalence",
        checks,
    )
    result["thresholds"] = t.to_dict()
    return result


_NUMERICAL_ZERO_COUNTS = (
    "uncertified_count", "resource_cap_count", "invalid_density_count",
    "approximation_count", "correction_count", "floor_count", "limiter_count",
    "renormalization_count", "nonfinite_count", "replay_bit_mismatch_count",
)


def _performance_checks(
    metrics: Mapping[str, Any], t: JacobiRBMultipathThresholds, *, pilot: bool
) -> dict[str, dict[str, Any]]:
    expected_steps = t.pilot_outer_steps if pilot else t.full_outer_steps
    expected_repeats = t.pilot_repeats_per_group if pilot else t.full_repeats_per_group
    checks: dict[str, dict[str, Any]] = {
        name: _eq_one(metrics, name)
        for name in (
            "all_groups_completed_pass", "all_certificates_pass",
            "output_hash_replay_pass", "final_state_hash_replay_pass",
            "certificate_hash_replay_pass", "restart_shard_chain_pass",
            "state_updates_device_resident_pass", "evolving_state_host_roundtrip_pass",
            "path_isolation_pass", "group_schedule_pass",
            "group_path_id_disjoint_pass",
            "commit_reuses_packed_host_snapshot_pass",
        )
    }
    checks.update({
        "group_sizes": _sequence_equal(metrics, "group_sizes", t.validation_group_sizes),
        "outer_steps": _equal(metrics, "outer_steps", expected_steps),
        "repeats_per_group": _equal(metrics, "repeats_per_group", expected_repeats),
        "restart_steps_per_shard": _equal(
            metrics, "restart_steps_per_shard", t.restart_steps_per_shard
        ),
        "maximum_cuda_launch_lanes": _le(
            metrics, "maximum_cuda_launch_lanes", float(t.maximum_launch_lanes)
        ),
        "mass_error": _le(metrics, "mass_error", t.maximum_cuda_mass_error),
        "cuda_kernel_max_error": _le(
            metrics, "cuda_kernel_max_error", t.maximum_cuda_kernel_error
        ),
        "fallback_fraction": _le(
            metrics, "fallback_fraction", t.maximum_fallback_fraction
        ),
        "fallback_cost_fraction": _le(
            metrics, "fallback_cost_fraction", t.maximum_fallback_cost_fraction
        ),
        "peak_memory_fraction": _le(
            metrics, "peak_memory_fraction", t.maximum_peak_memory_fraction
        ),
        "b10_slowest_transitions_per_second": _ge(
            metrics, "b10_slowest_transitions_per_second", t.minimum_rate
        ),
        "b4_slowest_transitions_per_second": _ge(
            metrics, "b4_slowest_transitions_per_second", t.minimum_rate
        ),
        "projected_cache_hours": _le(
            metrics, "projected_cache_hours", t.maximum_projected_hours
        ),
        "projected_effective_transitions_per_second": _ge(
            metrics, "projected_effective_transitions_per_second", t.minimum_rate
        ),
        "projected_transition_count": _equal(
            metrics, "projected_transition_count", t.projected_transition_count
        ),
        "certificate_fraction": _equal(metrics, "certificate_fraction", 1.0),
        "completed_shard_count": _equal(
            metrics,
            "completed_shard_count",
            len(t.validation_group_sizes)
            * expected_repeats
            * (expected_steps // t.restart_steps_per_shard),
        ),
        **{name: _eq_zero(metrics, name) for name in _NUMERICAL_ZERO_COUNTS},
    })
    if not pilot:
        checks.update({
            **{
                name: _eq_one(metrics, name)
                for name in (
                    "production_support_pass",
                    "cdf_endpoint_certificate_pass",
                    "cdf_monotonicity_pass",
                    "normalization_pass",
                    "semigroup_pass",
                    "detailed_balance_pass",
                    "law_control_pass",
                    "precision_doubling_hash_pass",
                )
            },
            "cuda_pair_mass_error": _le(
                metrics, "cuda_pair_mass_error", t.maximum_cuda_mass_error
            ),
            "cuda_simplex_error": _le(
                metrics, "cuda_simplex_error", t.maximum_cuda_mass_error
            ),
            "b10_transitions_per_repeat": _equal(
                metrics, "b10_transitions_per_repeat", t.b10_transitions_per_repeat
            ),
            "b4_transitions_per_repeat": _equal(
                metrics, "b4_transitions_per_repeat", t.b4_transitions_per_repeat
            ),
            "total_full_benchmark_transitions": _equal(
                metrics, "total_full_benchmark_transitions",
                t.total_full_benchmark_transitions,
            ),
        })
    return checks


_PERFORMANCE_RESOURCE_CHECKS = frozenset({
    "fallback_fraction",
    "fallback_cost_fraction",
    "b10_slowest_transitions_per_second",
    "b4_slowest_transitions_per_second",
    "projected_cache_hours",
    "projected_effective_transitions_per_second",
    "peak_memory_fraction",
})


def _evaluate_performance(
    metrics: Mapping[str, Any], *, pilot: bool,
    thresholds: JacobiRBMultipathThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or JacobiRBMultipathThresholds()
    checks = _performance_checks(metrics, t, pilot=pilot)
    result = _gate(
        "jacobi_rb_cuda_multipath_pilot" if pilot else "jacobi_rb_cuda_multipath_kernel",
        "exact multi-path scheduling pilot" if pilot else "exact full K=512 multi-path kernel",
        checks,
    )
    result["numerically_valid"] = int(
        all(
            check["passed"]
            for name, check in checks.items()
            if name not in _PERFORMANCE_RESOURCE_CHECKS
        )
    )
    result["resource_valid"] = int(
        all(checks[name]["passed"] for name in _PERFORMANCE_RESOURCE_CHECKS)
    )
    result["thresholds"] = t.to_dict()
    return result


def evaluate_multipath_pilot(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBMultipathThresholds | None = None,
) -> dict[str, Any]:
    return _evaluate_performance(metrics, pilot=True, thresholds=thresholds)


def evaluate_multipath_kernel(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBMultipathThresholds | None = None,
) -> dict[str, Any]:
    return _evaluate_performance(metrics, pilot=False, thresholds=thresholds)


def evaluate_multipath_target(metrics: Mapping[str, Any]) -> dict[str, Any]:
    base = evaluate_jacobi_rb_cuda_target(metrics)
    checks = {str(name): dict(value) for name, value in base["subchecks"].items()}
    checks["serial_multipath_target_parity_pass"] = _eq_one(
        metrics, "serial_multipath_target_parity_pass"
    )
    checks["target_path_isolation_pass"] = _eq_one(
        metrics, "target_path_isolation_pass"
    )
    result = _gate(
        "jacobi_rb_cuda_multipath_target",
        "unchanged certified Rao--Blackwell target under multi-path scheduling",
        checks,
    )
    result["thresholds"] = dict(base.get("thresholds", {}))
    return result


def _failed_names(gate: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(gate, Mapping) or not isinstance(gate.get("subchecks"), Mapping):
        return set()
    return {
        str(name)
        for name, value in gate["subchecks"].items()
        if not isinstance(value, Mapping) or not _one(value.get("passed"))
    }


def decide_multipath_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
    kernel_gate: Mapping[str, Any] | None,
    target_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not _passed(provenance):
        decision = JacobiRBMultipathDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 219-artifact parent binding"
    elif _status(preflight_gate) != "evaluated":
        decision = JacobiRBMultipathDecision.MULTIPATH_CUDA_RUNTIME_INVALID
        action = "complete the exact scheduling preflight"
    elif not _passed(preflight_gate):
        failed = _failed_names(preflight_gate)
        if failed & {
            "control_provenance_pass", "parent_record_count",
            "parent_certificate_pass", "parent_kernel_numerically_valid_pass",
            "parent_single_path_resource_failure_pass",
            "parent_target_not_evaluated_pass", "seven_parent_sources_immutable_pass",
        }:
            decision = JacobiRBMultipathDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable parent/source binding"
        elif failed & {
            "frozen_runtime_match_pass", "cuda_backend_replay_pass",
            "parent_cuda_source_hash_pass", "parent_cubin_hash_pass",
            "parent_compile_options_hash_pass",
        }:
            decision = JacobiRBMultipathDecision.MULTIPATH_CUDA_RUNTIME_INVALID
            action = "restore the frozen CUDA runtime and certified backend"
        else:
            decision = JacobiRBMultipathDecision.MULTIPATH_SCHEDULER_EQUIVALENCE_INVALID
            action = "repair path isolation, IDs, or serial/batched equivalence"
    elif _status(pilot_gate) != "evaluated":
        decision = JacobiRBMultipathDecision.MULTIPATH_PERFORMANCE_PILOT_UNRESOLVED
        action = "complete the frozen 64-step B10/B4 pilot"
    elif not _passed(pilot_gate):
        if _one(pilot_gate.get("numerically_valid")):
            decision = JacobiRBMultipathDecision.MULTIPATH_PERFORMANCE_PILOT_UNRESOLVED
            action = "investigate exact launch scheduling before a full benchmark"
        else:
            decision = JacobiRBMultipathDecision.MULTIPATH_KERNEL_NUMERICALLY_UNRESOLVED
            action = "repair exact certification or scheduler replay"
    elif _status(kernel_gate) != "evaluated":
        decision = JacobiRBMultipathDecision.MULTIPATH_KERNEL_COMPUTATIONALLY_INFEASIBLE
        action = "complete the full K=512 B10/B4 confirmation"
    elif not _passed(kernel_gate):
        if _one(kernel_gate.get("numerically_valid")):
            decision = JacobiRBMultipathDecision.MULTIPATH_KERNEL_COMPUTATIONALLY_INFEASIBLE
            action = "optimize exact execution scheduling within the frozen budget"
        else:
            decision = JacobiRBMultipathDecision.MULTIPATH_KERNEL_NUMERICALLY_UNRESOLVED
            action = "repair exact multi-path certification or replay"
    elif _status(target_gate) != "evaluated" or not _passed(target_gate):
        decision = JacobiRBMultipathDecision.JACOBI_RB_TARGET_INVALID
        action = "complete or repair the unchanged certified target controls"
    else:
        decision = (
            JacobiRBMultipathDecision.EXACT_JACOBI_RB_MULTIPATH_KERNEL_AND_TARGET_FEASIBLE
        )
        action = "plan the separate state-dependent Strang-refinement patch"
    feasible = decision is JacobiRBMultipathDecision.EXACT_JACOBI_RB_MULTIPATH_KERNEL_AND_TARGET_FEASIBLE
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "decision": decision.value,
        "recommended_next_action": action,
        "closed_terminal_scientific_outcome": 1,
        "state_dependent_strang_refinement_authorized": int(feasible),
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        **NO_WORK,
    }


def evaluate_multipath_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
    kernel_gate: Mapping[str, Any] | None,
    target_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    order = ("preflight", "pilot", "kernel", "target")
    if require_gate not in {"none", *order}:
        raise ValueError(f"unknown required gate: {require_gate}")
    components = {
        "preflight": dict(preflight_gate or not_evaluated_gate("preflight", "not run")),
        "pilot": dict(pilot_gate or not_evaluated_gate("pilot", "not run")),
        "kernel": dict(kernel_gate or not_evaluated_gate("kernel", "not run")),
        "target": dict(target_gate or not_evaluated_gate("target", "not run")),
    }
    required = () if require_gate == "none" else order[: order.index(require_gate) + 1]
    passed = bool(_passed(provenance) and all(_passed(components[name]) for name in required))
    decision = decide_multipath_workflow(
        provenance=provenance,
        preflight_gate=components["preflight"], pilot_gate=components["pilot"],
        kernel_gate=components["kernel"], target_gate=components["target"],
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_gate_pass": int(passed),
        "components": components,
        "decision": decision,
        "thresholds": JacobiRBMultipathThresholds().to_dict(),
        **NO_WORK,
    }


__all__ = [
    "JacobiRBMultipathDecision", "JacobiRBMultipathThresholds",
    "decide_multipath_workflow", "evaluate_multipath_kernel",
    "evaluate_multipath_pilot", "evaluate_multipath_preflight",
    "evaluate_multipath_target", "evaluate_multipath_workflow",
    "not_evaluated_gate",
]
