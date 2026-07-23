"""Pure fail-closed gates for the fused-CUDA Jacobi RB implementation.

These evaluators consume already-produced metrics only.  They do not inspect
CUDA, execute a kernel, train a model, or sample a reverse path.  Skipped or
incomplete stages are failures, and numerical certification is kept separate
from the frozen throughput and memory budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping


SCHEMA = "experiment12-d0-jacobi-rb-cuda-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class JacobiRBCudaThresholds:
    """Frozen CUDA numerical and production-resource contract."""

    parent_record_count: int = 315
    parent_replay_count: int = 294
    fresh_certificate_count: int = 512
    projected_transition_count: int = 89_915_392
    full_path_transition_count: int = 1_404_928
    minimum_full_path_benchmark_repeats: int = 3
    warmup_transition_count: int = 4_096
    throughput_transition_count: int = 65_536
    throughput_repeats: int = 3
    maximum_launch_transition_count: int = 4_096
    restart_steps_per_shard: int = 8
    minimum_certificate_fraction: float = 1.0
    maximum_cuda_certificate_fallback_fraction: float = 1.0e-4
    maximum_cuda_certificate_fallback_cost_fraction: float = 0.10
    maximum_cuda_mass_error: float = 2.0e-6
    maximum_cuda_kernel_error: float = 2.0e-6
    maximum_cuda_target_relative_error: float = 2.0e-5
    maximum_legacy_mixture_error: float = 1.0e-8
    minimum_slowest_transitions_per_second: float = 1_300.0
    maximum_projected_cache_hours: float = 20.0
    maximum_peak_memory_fraction: float = 0.80

    def __post_init__(self) -> None:
        expected: dict[str, int | float] = {
            "parent_record_count": 315,
            "parent_replay_count": 294,
            "fresh_certificate_count": 512,
            "projected_transition_count": 89_915_392,
            "full_path_transition_count": 1_404_928,
            "minimum_full_path_benchmark_repeats": 3,
            "warmup_transition_count": 4_096,
            "throughput_transition_count": 65_536,
            "throughput_repeats": 3,
            "maximum_launch_transition_count": 4_096,
            "restart_steps_per_shard": 8,
            "minimum_certificate_fraction": 1.0,
            "maximum_cuda_certificate_fallback_fraction": 1.0e-4,
            "maximum_cuda_certificate_fallback_cost_fraction": 0.10,
            "maximum_cuda_mass_error": 2.0e-6,
            "maximum_cuda_kernel_error": 2.0e-6,
            "maximum_cuda_target_relative_error": 2.0e-5,
            "maximum_legacy_mixture_error": 1.0e-8,
            "minimum_slowest_transitions_per_second": 1_300.0,
            "maximum_projected_cache_hours": 20.0,
            "maximum_peak_memory_fraction": 0.80,
        }
        for name, frozen in expected.items():
            value = getattr(self, name)
            if isinstance(frozen, int):
                valid = isinstance(value, int) and not isinstance(value, bool) and value == frozen
            else:
                valid = _finite(value) and float(value) == frozen
            if not valid:
                raise ValueError(f"{name} is frozen at {frozen}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JacobiRBCudaDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    FUSED_CUDA_BACKEND_UNAVAILABLE = "fused_cuda_backend_unavailable"
    CUDA_FLOATING_CONTRACT_INVALID = "cuda_floating_contract_invalid"
    DOUBLE_DOUBLE_INTERVAL_ALGEBRA_INVALID = (
        "double_double_interval_algebra_invalid"
    )
    CERTIFIED_EXPONENTIAL_INVALID = "certified_exponential_invalid"
    SPECTRAL_ROUNDING_CERTIFICATE_INVALID = (
        "spectral_rounding_certificate_invalid"
    )
    CUDA_CERTIFICATE_FALLBACK_EXCESSIVE = (
        "cuda_certificate_fallback_excessive"
    )
    SPECTRAL_INVERSION_NUMERICALLY_UNRESOLVED = (
        "spectral_inversion_numerically_unresolved"
    )
    SPECTRAL_INVERSION_COMPUTATIONALLY_INFEASIBLE = (
        "spectral_inversion_computationally_infeasible"
    )
    JACOBI_RB_TARGET_INVALID = "jacobi_rb_target_invalid"
    EXACT_JACOBI_RB_CUDA_KERNEL_AND_TARGET_FEASIBLE = (
        "exact_jacobi_rb_cuda_kernel_and_target_feasible"
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


def _status(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return "not_evaluated"
    raw = str(value.get("evaluation_status", "not_evaluated")).lower()
    return {
        "complete": "evaluated",
        "completed": "evaluated",
        "pending": "not_evaluated",
        "skipped": "not_evaluated",
        "incomplete": "not_evaluated",
    }.get(raw, raw)


def _passed(value: bool | int | Mapping[str, Any] | None) -> bool:
    if isinstance(value, Mapping):
        return _status(value) == "evaluated" and _one(
            value.get("passed", value.get("gate_pass", 0))
        )
    return value is True or (isinstance(value, int) and value == 1)


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


def _le(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value,
        "<=",
        threshold,
        _finite(value) and 0.0 <= float(value) <= threshold,
    )


def _ge(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(
        value,
        ">=",
        threshold,
        _finite(value) and float(value) >= threshold,
    )


def _gate(
    name: str,
    claim_scope: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_status: str = "evaluated",
) -> dict[str, Any]:
    normalized = {str(name): dict(check) for name, check in checks.items()}
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
            and all(_one(check.get("passed")) for check in normalized.values())
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    result = _gate(name, "not evaluated", {}, evaluation_status="not_evaluated")
    result["reason"] = str(reason)
    return result


def evaluate_jacobi_rb_cuda_preflight(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBCudaThresholds | None = None,
) -> dict[str, Any]:
    """Gate control provenance and the deterministic CUDA arithmetic stack."""

    t = thresholds or JacobiRBCudaThresholds()
    checks = {
        "control_provenance_pass": _eq_one(metrics, "control_provenance_pass"),
        "parent_record_count": _check(
            metrics.get("parent_record_count"),
            "==",
            t.parent_record_count,
            _finite(metrics.get("parent_record_count"))
            and int(metrics["parent_record_count"]) == t.parent_record_count,
        ),
        "parent_numerically_valid_pass": _eq_one(
            metrics, "parent_numerically_valid_pass"
        ),
        "parent_resource_infeasible_pass": _eq_one(
            metrics, "parent_resource_infeasible_pass"
        ),
        "fused_cuda_backend_available": _eq_one(
            metrics, "fused_cuda_backend_available"
        ),
        "cuda_floating_contract_pass": _eq_one(
            metrics, "cuda_floating_contract_pass"
        ),
        "frozen_runtime_match_pass": _eq_one(
            metrics, "frozen_runtime_match_pass"
        ),
        "compile_contract_pass": _eq_one(metrics, "compile_contract_pass"),
        "cuda_source_fingerprint_pass": _eq_one(
            metrics, "cuda_source_fingerprint_pass"
        ),
        "cubin_fingerprint_pass": _eq_one(
            metrics, "cubin_fingerprint_pass"
        ),
        "device_identity_pass": _eq_one(metrics, "device_identity_pass"),
        "directed_rounding_contract_pass": _eq_one(
            metrics, "directed_rounding_contract_pass"
        ),
        "double_double_interval_algebra_pass": _eq_one(
            metrics, "double_double_interval_algebra_pass"
        ),
        "certified_exponential_pass": _eq_one(
            metrics, "certified_exponential_pass"
        ),
        "deterministic_replay_pass": _eq_one(
            metrics, "deterministic_replay_pass"
        ),
        "forbidden_approximation_count": _eq_zero(
            metrics, "forbidden_approximation_count"
        ),
        "nonfinite_count": _eq_zero(metrics, "nonfinite_count"),
    }
    result = _gate(
        "jacobi_rb_cuda_preflight",
        "immutable control and deterministic fused-CUDA interval arithmetic",
        checks,
    )
    result["thresholds"] = t.to_dict()
    return result


_CERTIFICATE_NUMERICAL_CHECKS = frozenset(
    {
        "spectral_rounding_certificate_pass",
        "cdf_interval_enclosure_pass",
        "density_interval_enclosure_pass",
        "quantile_rounding_cell_pass",
        "target_rounding_cell_pass",
        "precision_doubling_hash_pass",
        "strengthening_hash_pass",
        "fresh_arb_enclosure_pass",
        "cubin_replay_pass",
        "cuda_source_replay_pass",
        "certificate_fraction",
        "parent_replay_count",
        "fresh_certificate_count",
        "parent_replay_y_bit_mismatch_count",
        "parent_replay_z_bit_mismatch_count",
        "uncertified_count",
        "resource_cap_count",
        "invalid_density_count",
        "approximation_count",
        "floor_count",
        "limiter_count",
        "renormalization_count",
        "ambiguous_rounding_count",
        "correction_count",
        "nonfinite_count",
    }
)
_CERTIFICATE_FALLBACK_CHECKS = frozenset(
    {
        "fresh_fallback_count",
        "cuda_certificate_fallback_fraction",
        "cuda_certificate_fallback_cost_fraction",
    }
)


def evaluate_jacobi_rb_cuda_certificate(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBCudaThresholds | None = None,
) -> dict[str, Any]:
    """Gate every-case spectral rounding certificates and fallback frequency."""

    t = thresholds or JacobiRBCudaThresholds()
    checks: dict[str, dict[str, Any]] = {
        name: _eq_one(metrics, name)
        for name in (
            "spectral_rounding_certificate_pass",
            "cdf_interval_enclosure_pass",
            "density_interval_enclosure_pass",
            "quantile_rounding_cell_pass",
            "target_rounding_cell_pass",
            "precision_doubling_hash_pass",
            "strengthening_hash_pass",
            "fresh_arb_enclosure_pass",
            "cubin_replay_pass",
            "cuda_source_replay_pass",
        )
    }
    checks.update(
        {
            "certificate_fraction": _check(
                metrics.get("certificate_fraction"),
                "==",
                t.minimum_certificate_fraction,
                _finite(metrics.get("certificate_fraction"))
                and float(metrics["certificate_fraction"])
                == t.minimum_certificate_fraction,
            ),
            "parent_replay_count": _check(
                metrics.get("parent_replay_count"),
                "==",
                t.parent_replay_count,
                _finite(metrics.get("parent_replay_count"))
                and int(metrics["parent_replay_count"]) == t.parent_replay_count,
            ),
            "fresh_certificate_count": _check(
                metrics.get("fresh_certificate_count"),
                "==",
                t.fresh_certificate_count,
                _finite(metrics.get("fresh_certificate_count"))
                and int(metrics["fresh_certificate_count"])
                == t.fresh_certificate_count,
            ),
            **{
                name: _eq_zero(metrics, name)
                for name in (
                    "parent_replay_y_bit_mismatch_count",
                    "parent_replay_z_bit_mismatch_count",
                    "uncertified_count",
                    "resource_cap_count",
                    "invalid_density_count",
                    "approximation_count",
                    "floor_count",
                    "limiter_count",
                    "renormalization_count",
                    "ambiguous_rounding_count",
                    "correction_count",
                    "nonfinite_count",
                )
            },
            "cuda_certificate_fallback_fraction": _le(
                metrics,
                "cuda_certificate_fallback_fraction",
                t.maximum_cuda_certificate_fallback_fraction,
            ),
            # The frozen fresh panel contains 512 transitions, so the
            # fractional 1e-4 limit already implies zero fallbacks.  Keep the
            # explicit count check as authorizing evidence rather than
            # relying on that arithmetic or on a rounded fraction field.
            "fresh_fallback_count": _eq_zero(
                metrics, "fresh_fallback_count"
            ),
            "cuda_certificate_fallback_cost_fraction": _le(
                metrics,
                "cuda_certificate_fallback_cost_fraction",
                t.maximum_cuda_certificate_fallback_cost_fraction,
            ),
        }
    )
    result = _gate(
        "jacobi_rb_cuda_certificate",
        "correctly rounded fused-CUDA spectral CDF, inverse, and target",
        checks,
    )
    result["thresholds"] = t.to_dict()
    result["numerically_valid"] = int(
        all(checks[name]["passed"] for name in _CERTIFICATE_NUMERICAL_CHECKS)
    )
    result["fallback_valid"] = int(
        all(checks[name]["passed"] for name in _CERTIFICATE_FALLBACK_CHECKS)
    )
    return result


_KERNEL_NUMERICAL_CHECKS = frozenset(
    {
        "production_support_pass",
        "cdf_endpoint_certificate_pass",
        "cdf_monotonicity_pass",
        "normalization_pass",
        "semigroup_pass",
        "detailed_balance_pass",
        "law_control_pass",
        "precision_doubling_hash_pass",
        "cuda_pair_mass_error",
        "cuda_simplex_error",
        "cuda_kernel_max_error",
        "uncertified_draw_count",
        "resource_cap_count",
        "invalid_density_count",
        "approximation_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "renormalization_count",
        "replay_bit_mismatch_count",
        "nonfinite_count",
    }
)
_KERNEL_RESOURCE_CHECKS = frozenset(
    {
        "warmup_pass",
        "throughput_probe_pass",
        "full_api_completed_pass",
        "state_updates_device_resident_pass",
        "in_shard_host_roundtrip_pass",
        "benchmark_output_hash_pass",
        "benchmark_final_state_hash_pass",
        "restart_shard_chain_pass",
        "warmup_transition_count",
        "throughput_transition_count",
        "throughput_repeats",
        "maximum_backend_call_size",
        "maximum_cuda_launch_lanes",
        "eight_step_shards_pass",
        "cuda_certificate_fallback_fraction",
        "cuda_certificate_fallback_cost_fraction",
        "full_path_transition_count",
        "full_path_benchmark_repeats",
        "slowest_transitions_per_second",
        "projected_transition_count",
        "projected_cache_hours",
        "peak_memory_fraction",
    }
)


def evaluate_jacobi_rb_cuda_kernel(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBCudaThresholds | None = None,
) -> dict[str, Any]:
    """Gate the exact fused kernel independently of its resource budget."""

    t = thresholds or JacobiRBCudaThresholds()
    checks: dict[str, dict[str, Any]] = {
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
            "warmup_pass",
            "throughput_probe_pass",
            "full_api_completed_pass",
            "state_updates_device_resident_pass",
            "in_shard_host_roundtrip_pass",
            "benchmark_output_hash_pass",
            "benchmark_final_state_hash_pass",
            "restart_shard_chain_pass",
        )
    }
    checks.update(
        {
            "cuda_pair_mass_error": _le(
                metrics, "cuda_pair_mass_error", t.maximum_cuda_mass_error
            ),
            "cuda_simplex_error": _le(
                metrics, "cuda_simplex_error", t.maximum_cuda_mass_error
            ),
            "cuda_kernel_max_error": _le(
                metrics, "cuda_kernel_max_error", t.maximum_cuda_kernel_error
            ),
            **{
                name: _eq_zero(metrics, name)
                for name in (
                    "uncertified_draw_count",
                    "resource_cap_count",
                    "invalid_density_count",
                    "approximation_count",
                    "correction_count",
                    "floor_count",
                    "limiter_count",
                    "renormalization_count",
                    "replay_bit_mismatch_count",
                    "nonfinite_count",
                )
            },
            "full_path_transition_count": _ge(
                metrics,
                "full_path_transition_count",
                float(t.full_path_transition_count),
            ),
            "full_path_benchmark_repeats": _ge(
                metrics,
                "full_path_benchmark_repeats",
                float(t.minimum_full_path_benchmark_repeats),
            ),
            "warmup_transition_count": _check(
                metrics.get("warmup_transition_count"), "==",
                t.warmup_transition_count,
                _finite(metrics.get("warmup_transition_count"))
                and int(metrics["warmup_transition_count"]) == t.warmup_transition_count,
            ),
            "throughput_transition_count": _check(
                metrics.get("throughput_transition_count"), "==",
                t.throughput_transition_count,
                _finite(metrics.get("throughput_transition_count"))
                and int(metrics["throughput_transition_count"]) == t.throughput_transition_count,
            ),
            "throughput_repeats": _check(
                metrics.get("throughput_repeats"), "==", t.throughput_repeats,
                _finite(metrics.get("throughput_repeats"))
                and int(metrics["throughput_repeats"]) == t.throughput_repeats,
            ),
            "maximum_backend_call_size": _le(
                metrics, "maximum_backend_call_size",
                float(t.maximum_launch_transition_count),
            ),
            "maximum_cuda_launch_lanes": _le(
                metrics, "maximum_cuda_launch_lanes",
                float(t.maximum_launch_transition_count),
            ),
            "eight_step_shards_pass": _eq_one(metrics, "eight_step_shards_pass"),
            "cuda_certificate_fallback_fraction": _le(
                metrics, "cuda_certificate_fallback_fraction",
                t.maximum_cuda_certificate_fallback_fraction,
            ),
            "cuda_certificate_fallback_cost_fraction": _le(
                metrics, "cuda_certificate_fallback_cost_fraction",
                t.maximum_cuda_certificate_fallback_cost_fraction,
            ),
            "slowest_transitions_per_second": _ge(
                metrics,
                "slowest_transitions_per_second",
                t.minimum_slowest_transitions_per_second,
            ),
            "projected_transition_count": _check(
                metrics.get("projected_transition_count"),
                "==",
                t.projected_transition_count,
                _finite(metrics.get("projected_transition_count"))
                and int(metrics["projected_transition_count"])
                == t.projected_transition_count,
            ),
            "projected_cache_hours": _le(
                metrics, "projected_cache_hours", t.maximum_projected_cache_hours
            ),
            "peak_memory_fraction": _le(
                metrics, "peak_memory_fraction", t.maximum_peak_memory_fraction
            ),
        }
    )
    result = _gate(
        "jacobi_rb_cuda_kernel",
        "exact fused-CUDA spectral inversion on frozen production support",
        checks,
    )
    result["thresholds"] = t.to_dict()
    result["numerically_valid"] = int(
        all(checks[name]["passed"] for name in _KERNEL_NUMERICAL_CHECKS)
    )
    result["resource_valid"] = int(
        all(checks[name]["passed"] for name in _KERNEL_RESOURCE_CHECKS)
    )
    return result


def evaluate_jacobi_rb_cuda_target(
    metrics: Mapping[str, Any],
    thresholds: JacobiRBCudaThresholds | None = None,
) -> dict[str, Any]:
    """Gate the exact later-state-only Rao--Blackwellized target."""

    t = thresholds or JacobiRBCudaThresholds()
    checks: dict[str, dict[str, Any]] = {
        name: _eq_one(metrics, name)
        for name in (
            "rao_blackwell_identity_pass",
            "population_tower_identity_pass",
            "latent_mixture_equivalence_pass",
            "pair_mass_conservation_pass",
            "h_minus_two_scaling_pass",
            "invariant_beta_pass",
            "flux_sign_negative_fixtures_pass",
            "all_four_colors_pass",
            "half_full_duration_pass",
            "density_positive_certificate_pass",
            "target_unique_rounding_pass",
            "target_rounding_certificate_pass",
            "conormal_orientation_pass",
            "synthetic_teacher_pass",
            "stationary_null_pass",
            "later_state_only_input_pass",
            "cuda_target_evaluated_pass",
        )
    }
    checks.update(
        {
            "target_certificate_fraction": _check(
                metrics.get("target_certificate_fraction"),
                "==",
                t.minimum_certificate_fraction,
                _finite(metrics.get("target_certificate_fraction"))
                and float(metrics["target_certificate_fraction"])
                == t.minimum_certificate_fraction,
            ),
            "cuda_target_relative_error": _le(
                metrics,
                "cuda_target_relative_error",
                t.maximum_cuda_target_relative_error,
            ),
            "legacy_mixture_max_absolute_error": _le(
                metrics,
                "legacy_mixture_max_absolute_error",
                t.maximum_legacy_mixture_error,
            ),
            **{
                name: _eq_zero(metrics, name)
                for name in (
                    "target_uncertified_count",
                    "target_replay_bit_mismatch_count",
                    "target_nonfinite_count",
                    "earlier_state_input_count",
                    "latent_variable_input_count",
                    "classifier_target_count",
                    "value_target_count",
                    "h1_target_count",
                    "raw_euler_residual_target_count",
                    "gaussian_target_count",
                    "target_clip_count",
                    "target_floor_count",
                    "target_limiter_count",
                    "target_projection_count",
                )
            },
        }
    )
    result = _gate(
        "jacobi_rb_cuda_target",
        "certified DDPM-like Rao--Blackwell target with no input leakage",
        checks,
    )
    result["thresholds"] = t.to_dict()
    return result


def _failed_names(gate: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(gate, Mapping):
        return set()
    checks = gate.get("subchecks")
    if not isinstance(checks, Mapping):
        return set()
    return {
        str(name)
        for name, check in checks.items()
        if not isinstance(check, Mapping) or not _one(check.get("passed"))
    }


def decide_jacobi_rb_cuda_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    certificate_gate: Mapping[str, Any] | None,
    kernel_gate: Mapping[str, Any] | None,
    target_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return exactly one closed outcome in scientific dependency order."""

    if not _passed(provenance):
        decision = JacobiRBCudaDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 315-record control binding"
    elif _status(preflight_gate) != "evaluated":
        decision = JacobiRBCudaDecision.FUSED_CUDA_BACKEND_UNAVAILABLE
        action = "complete the fused-CUDA arithmetic preflight"
    elif not _passed(preflight_gate):
        failed = _failed_names(preflight_gate)
        if failed & {
            "control_provenance_pass",
            "parent_record_count",
            "parent_numerically_valid_pass",
            "parent_resource_infeasible_pass",
        }:
            decision = JacobiRBCudaDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable 315-record control binding"
        elif "fused_cuda_backend_available" in failed:
            decision = JacobiRBCudaDecision.FUSED_CUDA_BACKEND_UNAVAILABLE
            action = "provide the pinned fused-CUDA backend"
        elif "cuda_floating_contract_pass" in failed:
            decision = JacobiRBCudaDecision.CUDA_FLOATING_CONTRACT_INVALID
            action = "repair deterministic CUDA floating-point semantics"
        elif failed & {
            "frozen_runtime_match_pass",
            "compile_contract_pass",
            "cuda_source_fingerprint_pass",
            "cubin_fingerprint_pass",
            "device_identity_pass",
            "directed_rounding_contract_pass",
        }:
            decision = JacobiRBCudaDecision.CUDA_FLOATING_CONTRACT_INVALID
            action = "restore the pinned runtime, CUBIN, and directed-rounding contract"
        elif "double_double_interval_algebra_pass" in failed:
            decision = JacobiRBCudaDecision.DOUBLE_DOUBLE_INTERVAL_ALGEBRA_INVALID
            action = "repair outward-rounded double-double interval algebra"
        elif "certified_exponential_pass" in failed:
            decision = JacobiRBCudaDecision.CERTIFIED_EXPONENTIAL_INVALID
            action = "repair the certified device exponential enclosure"
        else:
            decision = JacobiRBCudaDecision.CUDA_FLOATING_CONTRACT_INVALID
            action = "repair the fused-CUDA preflight contract"
    elif _status(certificate_gate) != "evaluated":
        decision = JacobiRBCudaDecision.SPECTRAL_ROUNDING_CERTIFICATE_INVALID
        action = "complete the every-case spectral rounding certificate"
    elif not _passed(certificate_gate):
        numerical = _one(dict(certificate_gate or {}).get("numerically_valid"))
        fallback = _one(dict(certificate_gate or {}).get("fallback_valid"))
        if numerical and not fallback:
            decision = JacobiRBCudaDecision.CUDA_CERTIFICATE_FALLBACK_EXCESSIVE
            action = "reduce certified host fallback without changing the law"
        else:
            decision = JacobiRBCudaDecision.SPECTRAL_ROUNDING_CERTIFICATE_INVALID
            action = "repair spectral enclosures and unique-rounding proofs"
    elif _status(kernel_gate) != "evaluated":
        decision = JacobiRBCudaDecision.SPECTRAL_INVERSION_NUMERICALLY_UNRESOLVED
        action = "complete the certified fused-kernel evaluation"
    elif not _passed(kernel_gate):
        numerical = _one(dict(kernel_gate or {}).get("numerically_valid"))
        resource = _one(dict(kernel_gate or {}).get("resource_valid"))
        if numerical and not resource:
            decision = (
                JacobiRBCudaDecision.SPECTRAL_INVERSION_COMPUTATIONALLY_INFEASIBLE
            )
            action = "optimize the exact fused kernel within the frozen budget"
        else:
            decision = JacobiRBCudaDecision.SPECTRAL_INVERSION_NUMERICALLY_UNRESOLVED
            action = "repair exact inversion; do not emit an approximate transition"
    elif _status(target_gate) != "evaluated" or not _passed(target_gate):
        decision = JacobiRBCudaDecision.JACOBI_RB_TARGET_INVALID
        action = "complete and repair the certified later-state-only target"
    else:
        decision = (
            JacobiRBCudaDecision.EXACT_JACOBI_RB_CUDA_KERNEL_AND_TARGET_FEASIBLE
        )
        action = "run the separate state-dependent Strang-refinement gate"

    ready = (
        decision
        is JacobiRBCudaDecision.EXACT_JACOBI_RB_CUDA_KERNEL_AND_TARGET_FEASIBLE
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "decision": decision.value,
        "closed_terminal_scientific_outcome": 1,
        "recommended_next_action": action,
        "kernel_and_target_followup_authorized": int(ready),
        "state_dependent_strang_refinement_authorized": int(ready),
        "one_image_training_authorized": 0,
        "physical_training_authorized": 0,
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


def evaluate_jacobi_rb_cuda_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any],
    certificate_gate: Mapping[str, Any],
    kernel_gate: Mapping[str, Any],
    target_gate: Mapping[str, Any],
    require_gate: str = "none",
    thresholds: JacobiRBCudaThresholds | None = None,
) -> dict[str, Any]:
    """Compose the four pure gates and evaluate a requested prefix."""

    t = thresholds or JacobiRBCudaThresholds()
    if require_gate not in {"none", "preflight", "certificate", "kernel", "target"}:
        raise ValueError(
            "require_gate must be none, preflight, certificate, kernel, or target"
        )
    prefix = {
        "none": (),
        "preflight": (preflight_gate,),
        "certificate": (preflight_gate, certificate_gate),
        "kernel": (preflight_gate, certificate_gate, kernel_gate),
        "target": (preflight_gate, certificate_gate, kernel_gate, target_gate),
    }[require_gate]
    required_pass = _passed(provenance) and all(_passed(gate) for gate in prefix)
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "components": {
            "provenance": (
                dict(provenance)
                if isinstance(provenance, Mapping)
                else int(_passed(provenance))
            ),
            "preflight": dict(preflight_gate),
            "certificate": dict(certificate_gate),
            "kernel": dict(kernel_gate),
            "target": dict(target_gate),
        },
        "decision": decide_jacobi_rb_cuda_workflow(
            provenance=provenance,
            preflight_gate=preflight_gate,
            certificate_gate=certificate_gate,
            kernel_gate=kernel_gate,
            target_gate=target_gate,
        ),
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "thresholds": t.to_dict(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


# Short aliases make the module convenient without weakening the explicit API.
evaluate_preflight = evaluate_jacobi_rb_cuda_preflight
evaluate_certificate = evaluate_jacobi_rb_cuda_certificate
evaluate_kernel = evaluate_jacobi_rb_cuda_kernel
evaluate_target = evaluate_jacobi_rb_cuda_target
decide_workflow = decide_jacobi_rb_cuda_workflow
evaluate_workflow = evaluate_jacobi_rb_cuda_workflow


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "JacobiRBCudaThresholds",
    "JacobiRBCudaDecision",
    "not_evaluated_gate",
    "evaluate_jacobi_rb_cuda_preflight",
    "evaluate_jacobi_rb_cuda_certificate",
    "evaluate_jacobi_rb_cuda_kernel",
    "evaluate_jacobi_rb_cuda_target",
    "decide_jacobi_rb_cuda_workflow",
    "evaluate_jacobi_rb_cuda_workflow",
    "evaluate_preflight",
    "evaluate_certificate",
    "evaluate_kernel",
    "evaluate_target",
    "decide_workflow",
    "evaluate_workflow",
]
