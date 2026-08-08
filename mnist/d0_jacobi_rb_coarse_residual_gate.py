"""Fail-closed gates for the exact-K512 coarse-baseline residual learner.

The learner retains the exact binary64 Jacobi/Rao--Blackwell denoising label.
Its prediction is a frozen coarse table plus a permitted-input residual, and
the *combined* prediction is trained and evaluated with ordinary unweighted
MSE against the unchanged label.

Passing the terminal gate establishes one-image predictive learnability only
for the certified fixed-K512 seven-phase split chain.  This module never
authorizes a reverse sample, reconstruction, a known prior, an unsplit
Eulerian-generator claim, or spatial Dirichlet--Ferguson convergence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_jacobi_artifacts import config_fingerprint


SCHEMA = "experiment12-d0-jacobi-rb-coarse-residual-gate"
SCHEMA_VERSION = 1
STAGES = ("preflight", "cache", "train", "confirm", "report", "all")
REQUIRED_GATES = ("none", "preflight", "cache", "train", "confirm")
CLAIM_SCOPE = (
    "coarse-baseline plus permitted-input exact-RB residual learnability for "
    "one image under the certified fixed-K512 split chain"
)

WITNESS_REGISTRY_SHA256 = (
    "ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3"
)
BASELINE_SCHEMA = "d0-jacobi-rb-coarse-residual-v1-frozen-baseline-v1"
BASELINE_VALUES_SHA256 = (
    "5d4e73153c36a59e26403439befd4e13b7f4fe096f7cbf9af6b77ac26565a9df"
)
BASELINE_RAW_VALUES_SHA256 = (
    "cb66524aad30ef3a6c442e007ac0afff2e6ae745fcff07c8c489ce3fc8a941d6"
)
BASELINE_RAW_VALUES_SERIALIZATION_SHA256 = (
    "52e1e938d47ddd2d6f2210bfa6b726b69467fb17a61405ef2b772d8f9677c24a"
)
BASELINE_VALUES_SERIALIZATION_SHA256 = (
    "ff63431e776ea429667eff8de042b8308ac294aafa55872f6e7c3e4532606b23"
)
BASELINE_SHAPE = (4, 7, 392)
BASELINE_SIGNAL_ENERGY = 0.0006484248701021389
BASELINE_PANEL_MEAN_NOISE = 0.00315904482822984
BASELINE_AVERAGED_TABLE_NOISE = 0.00157952241411492
BASELINE_SHRINKAGE = 0.2910413880506186
BASELINE_ENERGY = 0.00018871847424106853

NO_CLAIM_AUTHORIZATION = {
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
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "reconstruction_performed": 0,
}


class CoarseResidualDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    COARSE_BASELINE_DERIVATION_INVALID = "coarse_baseline_derivation_invalid"
    COARSE_RESIDUAL_DESIGN_INFEASIBLE = "coarse_residual_design_infeasible"
    FRESH_EXACT_CACHE_INVALID = "fresh_exact_cache_invalid"
    RESIDUAL_OPTIMIZATION_PIPELINE_INVALID = (
        "residual_optimization_pipeline_invalid"
    )
    COARSE_BASELINE_ONLY_SIGNAL = "coarse_baseline_only_signal"
    COARSE_BASELINE_NONREPLICATING = "coarse_baseline_nonreplicating"
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    COARSE_RESIDUAL_AUDIT_INCONCLUSIVE = "coarse_residual_audit_inconclusive"
    PAIRED_RISK_INFERENCE_INVALID = "paired_risk_inference_invalid"
    EXACT_RB_COARSE_RESIDUAL_LEARNABLE = "exact_rb_coarse_residual_learnable"


@dataclass(frozen=True)
class CoarseResidualThresholds:
    """Frozen production design and scientific thresholds."""

    train_paths: int = 64
    validation_paths: int = 32
    confirmation_paths: int = 64
    selected_outer_step_count: int = 32
    phases_per_step: int = 7
    edges_per_phase: int = 392
    train_samples: int = 14_336
    validation_samples: int = 7_168
    confirmation_samples: int = 14_336
    train_transitions: int = 89_915_392
    validation_transitions: int = 44_957_696
    confirmation_transitions: int = 89_915_392
    total_transitions: int = 224_788_480
    model_seeds: tuple[int, ...] = (261_252, 261_253, 261_254)
    maximum_updates: int = 4_000
    validation_interval: int = 100
    batch_size: int = 32
    synthetic_max_relative_validation_mse: float = 0.01
    maximum_mass_error: float = 2.0e-12
    minimum_transitions_per_second: float = 1_300.0
    maximum_peak_memory_fraction: float = 0.80
    maximum_persisted_cache_bytes: int = 1_342_177_280
    maximum_projected_total_hours: float = 30.0
    bootstrap_seed: int = 261_255
    bootstrap_replicates: int = 50_000
    confidence: float = 0.99
    direct_derived_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        for name in (
            "train_paths",
            "validation_paths",
            "confirmation_paths",
            "selected_outer_step_count",
            "phases_per_step",
            "edges_per_phase",
            "train_samples",
            "validation_samples",
            "confirmation_samples",
            "train_transitions",
            "validation_transitions",
            "confirmation_transitions",
            "total_transitions",
            "maximum_updates",
            "validation_interval",
            "batch_size",
            "maximum_persisted_cache_bytes",
            "bootstrap_seed",
            "bootstrap_replicates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            len(self.model_seeds) != 3
            or len(set(self.model_seeds)) != 3
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0
                for seed in self.model_seeds
            )
        ):
            raise ValueError("model_seeds must contain three distinct integers")
        for name in (
            "synthetic_max_relative_validation_mse",
            "maximum_mass_error",
            "minimum_transitions_per_second",
            "maximum_peak_memory_fraction",
            "maximum_projected_total_hours",
            "confidence",
            "direct_derived_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_peak_memory_fraction > 1.0 or self.confidence >= 1.0:
            raise ValueError("memory fraction and confidence must be below one")
        if self.total_transitions != (
            self.train_transitions
            + self.validation_transitions
            + self.confirmation_transitions
        ):
            raise ValueError("total transition count is inconsistent")

    @property
    def selected_outer_steps(self) -> tuple[int, ...]:
        return tuple(15 + 16 * index for index in range(32))

    @property
    def preconfirmation_transitions(self) -> int:
        return self.train_transitions + self.validation_transitions

    @property
    def preconfirmation_samples(self) -> int:
        return self.train_samples + self.validation_samples

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_seeds"] = list(self.model_seeds)
        value["selected_outer_steps"] = list(self.selected_outer_steps)
        return value


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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _passed(value: Mapping[str, Any] | bool | int | None) -> bool:
    if isinstance(value, Mapping):
        return (
            value.get("evaluation_status") == "evaluated"
            and _one(value.get("passed"))
        )
    return value is True or (
        isinstance(value, int) and not isinstance(value, bool) and value == 1
    )


def _status(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return "not_evaluated"
    return str(value.get("evaluation_status", "not_evaluated"))


def _check(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _eq_one(record: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = record.get(name)
    return _check(value, "==", 1, _one(value))


def _eq_zero(record: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = record.get(name)
    return _check(value, "==", 0, _zero(value))


def _eq(record: Mapping[str, Any], name: str, expected: Any) -> dict[str, Any]:
    value = record.get(name)
    return _check(value, "==", expected, value == expected)


def _le(record: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = record.get(name)
    return _check(
        value,
        "<=",
        threshold,
        _finite(value) and 0.0 <= float(value) <= float(threshold),
    )


def _ge(record: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = record.get(name)
    return _check(
        value,
        ">=",
        threshold,
        _finite(value) and float(value) >= float(threshold),
    )


def _strict_lt(left: Any, right: Any) -> bool:
    return _finite(left) and _finite(right) and float(left) < float(right)


def _gate(
    name: str, checks: Mapping[str, Mapping[str, Any]], **fields: Any
) -> dict[str, Any]:
    passed = bool(checks) and all(_one(item.get("passed")) for item in checks.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": str(name),
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
    exc: BaseException | None = None,
    *,
    failure_domain: str = "execution",
    failure_code: str = "coarse_residual_execution_failed",
    error_type: str | None = None,
    error: str | None = None,
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
        "error_type": str(error_type or (type(exc).__name__ if exc else "RuntimeError")),
        "error": str(error if error is not None else (exc or "")),
        "subchecks": {},
        "passed": 0,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


def _path_ids(plan: Mapping[str, Any], split: str) -> tuple[int, ...]:
    raw: Any = plan.get(f"{split}_path_ids")
    nested = plan.get("path_ids")
    if raw is None and isinstance(nested, Mapping):
        raw = nested.get(split)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        return ()
    return tuple(int(value) for value in raw)


def _exact_float(record: Mapping[str, Any], name: str, expected: float) -> dict[str, Any]:
    value = record.get(name)
    return _check(
        value,
        "binary64 ==",
        expected,
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and float(value) == expected,
    )


def evaluate_coarse_residual_preflight(
    *,
    provenance_valid: bool | int | Mapping[str, Any],
    baseline_record: Mapping[str, Any],
    path_plan: Mapping[str, Any],
    thresholds: CoarseResidualThresholds | None = None,
) -> dict[str, Any]:
    """Gate immutable provenance, the literal baseline, and fresh design."""

    t = thresholds or CoarseResidualThresholds()
    provenance_pass = _passed(provenance_valid)
    baseline_checks = {
        "baseline_schema": _eq(baseline_record, "schema", BASELINE_SCHEMA),
        "baseline_schema_version": _eq(baseline_record, "schema_version", 1),
        "baseline_shape": _eq(baseline_record, "shape", list(BASELINE_SHAPE)),
        "baseline_dtype": _eq(baseline_record, "dtype", "<f8"),
        "baseline_values_sha256": _eq(
            baseline_record, "values_sha256", BASELINE_VALUES_SHA256
        ),
        "baseline_raw_values_sha256": _eq(
            baseline_record,
            "raw_values_sha256",
            BASELINE_RAW_VALUES_SHA256,
        ),
        "baseline_raw_values_serialization_sha256": _eq(
            baseline_record,
            "raw_values_serialization_sha256",
            BASELINE_RAW_VALUES_SERIALIZATION_SHA256,
        ),
        "baseline_values_serialization_sha256": _eq(
            baseline_record,
            "values_serialization_sha256",
            BASELINE_VALUES_SERIALIZATION_SHA256,
        ),
        "witness_registry_sha256": _eq(
            baseline_record,
            "witness_registry_sha256",
            WITNESS_REGISTRY_SHA256,
        ),
        "signal_energy": _exact_float(
            baseline_record, "signal_energy", BASELINE_SIGNAL_ENERGY
        ),
        "panel_mean_noise": _exact_float(
            baseline_record, "panel_mean_noise", BASELINE_PANEL_MEAN_NOISE
        ),
        "averaged_table_noise": _exact_float(
            baseline_record,
            "averaged_table_noise",
            BASELINE_AVERAGED_TABLE_NOISE,
        ),
        "shrinkage": _exact_float(
            baseline_record, "shrinkage", BASELINE_SHRINKAGE
        ),
        "baseline_energy": _exact_float(
            baseline_record, "baseline_energy", BASELINE_ENERGY
        ),
        "fit_role": _eq(
            baseline_record,
            "fit_role",
            "historical_witness_panels_training_only",
        ),
        "target_modified": _eq_zero(baseline_record, "target_modified"),
        "left_path_ids": _eq(
            baseline_record,
            "left_path_ids",
            list(range(0xE5000, 0xE5040)),
        ),
        "right_path_ids": _eq(
            baseline_record,
            "right_path_ids",
            list(range(0xE5100, 0xE5140)),
        ),
        "left_cell_means_file_sha256": _eq(
            baseline_record,
            "left_cell_means_file_sha256",
            "70d374526df5c02e5c6ab7f9b17205de373b22c694480bb27bf5684b4a579852",
        ),
        "right_cell_means_file_sha256": _eq(
            baseline_record,
            "right_cell_means_file_sha256",
            "d64688f026cc510d586fb6b20e2303fdbe407a99b1a161b4654dc5dd04face81",
        ),
        "left_cell_means_array_sha256": _eq(
            baseline_record,
            "left_cell_means_array_sha256",
            "1fe04953fd50ea3cb0ac163efed216ec5ebbafc58f48ce0de3f77d090c29fe08",
        ),
        "right_cell_means_array_sha256": _eq(
            baseline_record,
            "right_cell_means_array_sha256",
            "2d949662c098783aa663672528f107a9f73f503529440aca4313cf770cad737e",
        ),
    }
    semantic = baseline_record.get("semantic_sha256")
    if semantic is not None:
        expected_semantic = config_fingerprint(
            {
                key: value
                for key, value in baseline_record.items()
                if key != "semantic_sha256"
            }
        )
        baseline_checks["semantic_sha256"] = _check(
            semantic, "==", expected_semantic, semantic == expected_semantic
        )

    train_ids = _path_ids(path_plan, "train")
    validation_ids = _path_ids(path_plan, "validation")
    confirmation_ids = _path_ids(path_plan, "confirmation")
    all_ids = train_ids + validation_ids + confirmation_ids
    path_checks = {
        "path_plan_frozen_pass": _eq_one(path_plan, "path_plan_frozen_pass"),
        "train_path_count": _check(
            len(train_ids), "==", t.train_paths, len(train_ids) == t.train_paths
        ),
        "validation_path_count": _check(
            len(validation_ids),
            "==",
            t.validation_paths,
            len(validation_ids) == t.validation_paths,
        ),
        "confirmation_path_count": _check(
            len(confirmation_ids),
            "==",
            t.confirmation_paths,
            len(confirmation_ids) == t.confirmation_paths,
        ),
        "path_ids_unique": _check(
            len(set(all_ids)), "==", len(all_ids), len(set(all_ids)) == len(all_ids)
        ),
        "path_ids_20_bit": _check(
            int(bool(all_ids) and all(0 <= value < 2**20 for value in all_ids)),
            "==",
            1,
            bool(all_ids) and all(0 <= value < 2**20 for value in all_ids),
        ),
        "parent_path_collision_count": _eq(
            path_plan, "parent_path_collision_count", 0
        ),
        "model_input_firewall_pass": _eq_one(
            path_plan, "model_input_firewall_pass"
        ),
        "earlier_state_forbidden_pass": _eq_one(
            path_plan, "earlier_state_forbidden_pass"
        ),
        "certificate_input_forbidden_pass": _eq_one(
            path_plan, "certificate_input_forbidden_pass"
        ),
        "confirmation_sealed_pass": _eq_one(
            path_plan, "confirmation_sealed_pass"
        ),
        "selected_outer_steps": _eq(
            path_plan, "selected_outer_steps", list(t.selected_outer_steps)
        ),
        "projected_transition_count": _eq(
            path_plan, "projected_transition_count", t.total_transitions
        ),
        "projected_total_hours": _le(
            path_plan,
            "projected_total_hours",
            t.maximum_projected_total_hours,
        ),
        "projected_cache_bytes": _le(
            path_plan,
            "projected_cache_bytes",
            float(t.maximum_persisted_cache_bytes),
        ),
        "test_only_reduced_workload": _eq_zero(
            path_plan, "test_only_reduced_workload"
        ),
    }
    checks = {
        "provenance_valid": _check(
            int(provenance_pass), "==", 1, provenance_pass
        ),
        **baseline_checks,
        **path_checks,
    }
    baseline_valid = all(
        _one(item["passed"]) for item in baseline_checks.values()
    )
    design_valid = all(_one(item["passed"]) for item in path_checks.values())
    result = _gate(
        "preflight",
        checks,
        provenance_valid=int(provenance_pass),
        baseline_derivation_valid=int(baseline_valid),
        design_feasible=int(design_valid),
        baseline_values_sha256=BASELINE_VALUES_SHA256,
        baseline_shrinkage=BASELINE_SHRINKAGE,
    )
    result["cache_generation_authorized"] = int(result["passed"])
    result["thresholds"] = t.to_dict()
    return result


def evaluate_coarse_residual_cache(
    record: Mapping[str, Any],
    *,
    split: str,
    thresholds: CoarseResidualThresholds | None = None,
) -> dict[str, Any]:
    """Gate one fresh exact physical cache split."""

    if split not in {"train", "validation", "confirmation"}:
        raise ValueError(f"unknown cache split: {split}")
    t = thresholds or CoarseResidualThresholds()
    path_count = {
        "train": t.train_paths,
        "validation": t.validation_paths,
        "confirmation": t.confirmation_paths,
    }[split]
    sample_count = {
        "train": t.train_samples,
        "validation": t.validation_samples,
        "confirmation": t.confirmation_samples,
    }[split]
    transition_count = {
        "train": t.train_transitions,
        "validation": t.validation_transitions,
        "confirmation": t.confirmation_transitions,
    }[split]
    exact_names = (
        "all_shards_complete_pass",
        "cache_complete_pass",
        "cache_replay_hash_pass",
        "states_finite_pass",
        "targets_finite_pass",
        "capture_state_alignment_pass",
        "sample_key_join_pass",
        "sample_key_unique_pass",
        "selected_step_phase_coverage_pass",
        "split_role_isolation_pass",
        "path_plan_binding_pass",
        "baseline_hash_binding_pass",
        "model_input_firewall_pass",
        "exact_jacobi_transition_pass",
        "exact_rb_target_pass",
        "unmodified_binary64_target_pass",
        "state_updates_device_resident_pass",
    )
    checks = {name: _eq_one(record, name) for name in exact_names}
    checks.update(
        {
            "split": _eq(record, "split", split),
            "path_count": _eq(record, "path_count", path_count),
            "sample_count": _eq(record, "sample_count", sample_count),
            "transition_count": _eq(
                record, "transition_count", transition_count
            ),
            "selected_outer_steps": _eq(
                record, "selected_outer_steps", list(t.selected_outer_steps)
            ),
            "certificate_fraction": _check(
                record.get("certificate_fraction"),
                "==",
                1.0,
                _finite(record.get("certificate_fraction"))
                and float(record["certificate_fraction"]) == 1.0,
            ),
            "maximum_mass_error": _le(
                record, "maximum_mass_error", t.maximum_mass_error
            ),
            "transitions_per_second": _ge(
                record,
                "transitions_per_second",
                t.minimum_transitions_per_second,
            ),
            "peak_memory_fraction": _le(
                record,
                "peak_memory_fraction",
                t.maximum_peak_memory_fraction,
            ),
        }
    )
    checks.update({name: _eq_zero(record, name) for name in _FORBIDDEN_COUNTS})
    checks["residual_target_persisted"] = _eq_zero(
        record, "residual_target_persisted"
    )
    if split == "confirmation":
        checks.update(
            {
                "confirmation_seal_pass": _eq_one(
                    record, "confirmation_seal_pass"
                ),
                "confirmation_opened_once_pass": _eq_one(
                    record, "confirmation_opened_once_pass"
                ),
                "confirmation_plan_unchanged_pass": _eq_one(
                    record, "confirmation_plan_unchanged_pass"
                ),
            }
        )
    else:
        checks["confirmation_absent_pass"] = _eq_one(
            record, "confirmation_absent_pass"
        )
    result = _gate(f"{split}_cache", checks, split=split)
    result["numerically_valid"] = int(result["passed"])
    result["thresholds"] = t.to_dict()
    return result


def evaluate_coarse_residual_cache_set(
    *,
    split_records: Mapping[str, Mapping[str, Any]],
    thresholds: CoarseResidualThresholds | None = None,
) -> dict[str, Any]:
    """Gate exactly the fresh train and validation caches."""

    t = thresholds or CoarseResidualThresholds()
    expected = {"train", "validation"}
    component = {
        split: evaluate_coarse_residual_cache(
            split_records.get(split, {}), split=split, thresholds=t
        )
        for split in sorted(expected)
    }
    aggregate = split_records.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    persisted = aggregate.get(
        "persisted_cache_bytes",
        sum(
            int(split_records.get(split, {}).get("persisted_cache_bytes", 0))
            for split in expected
        ),
    )
    checks = {
        "exact_split_set": _check(
            sorted(key for key in split_records if key != "aggregate"),
            "==",
            sorted(expected),
            set(key for key in split_records if key != "aggregate") == expected,
        ),
        **{
            f"{split}_cache_gate": _check(
                component[split]["passed"],
                "==",
                1,
                _passed(component[split]),
            )
            for split in sorted(expected)
        },
        "transition_count": _check(
            sum(
                int(split_records.get(split, {}).get("transition_count", -1))
                for split in expected
            ),
            "==",
            t.preconfirmation_transitions,
            sum(
                int(split_records.get(split, {}).get("transition_count", -1))
                for split in expected
            )
            == t.preconfirmation_transitions,
        ),
        "sample_count": _check(
            sum(
                int(split_records.get(split, {}).get("sample_count", -1))
                for split in expected
            ),
            "==",
            t.preconfirmation_samples,
            sum(
                int(split_records.get(split, {}).get("sample_count", -1))
                for split in expected
            )
            == t.preconfirmation_samples,
        ),
        "persisted_cache_bytes": _check(
            persisted,
            "<=",
            t.maximum_persisted_cache_bytes,
            isinstance(persisted, int)
            and not isinstance(persisted, bool)
            and 0 <= persisted <= t.maximum_persisted_cache_bytes,
        ),
        "confirmation_absent_pass": _eq_one(
            aggregate, "confirmation_absent_pass"
        ),
        "split_path_sets_disjoint_pass": _eq_one(
            aggregate, "split_path_sets_disjoint_pass"
        ),
    }
    result = _gate("cache", checks, component_gates=component)
    result["physical_training_authorized"] = int(result["passed"])
    result["thresholds"] = t.to_dict()
    return result


def _role(record: Mapping[str, Any]) -> str:
    for name in ("role", "task", "law", "task_role"):
        if name in record:
            return str(record[name])
    return ""


def _float_sequence(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return ()
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return ()
    return result if all(math.isfinite(item) for item in result) else ()


def _pick_task(
    tasks: Sequence[Mapping[str, Any]], names: set[str]
) -> Mapping[str, Any] | None:
    matches = [record for record in tasks if _role(record) in names]
    return matches[0] if len(matches) == 1 else None


def evaluate_coarse_residual_train(
    *,
    task_records: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    thresholds: CoarseResidualThresholds | None = None,
) -> dict[str, Any]:
    """Gate controls, physical optimization, and validation-only selection."""

    t = thresholds or CoarseResidualThresholds()
    tasks = [dict(record) for record in task_records]
    synthetic = _pick_task(
        tasks, {"synthetic_teacher", "teacher", "synthetic"}
    )
    null = _pick_task(tasks, {"exact_baseline_null", "baseline_null", "null"})
    physical = [record for record in tasks if _role(record).startswith("physical")]
    synthetic_paths = _float_sequence(
        (synthetic or {}).get(
            "path_baseline_minus_model_mse",
            (synthetic or {}).get("validation_path_baseline_minus_model_mse", ()),
        )
    )
    synthetic_relative = (synthetic or {}).get(
        "relative_validation_mse",
        (synthetic or {}).get("validation_relative_mse"),
    )
    null_update = (null or {}).get(
        "selected_update", (null or {}).get("selected_step")
    )
    selected_update = selection.get(
        "selected_update", selection.get("selected_step")
    )
    combined_overall = selection.get(
        "combined_validation_mse",
        selection.get("validation_combined_mse"),
    )
    baseline_overall = selection.get(
        "baseline_validation_mse",
        selection.get("validation_baseline_mse"),
    )
    combined_high = selection.get(
        "combined_validation_mse_high_reverse_time",
        selection.get(
            "validation_combined_mse_high_reverse_time",
            selection.get(
                "combined_validation_mse_data_end",
                selection.get("validation_combined_mse_data_end"),
            ),
        ),
    )
    baseline_high = selection.get(
        "baseline_validation_mse_high_reverse_time",
        selection.get(
            "validation_baseline_mse_high_reverse_time",
            selection.get(
                "baseline_validation_mse_data_end",
                selection.get("validation_baseline_mse_data_end"),
            ),
        ),
    )
    nonzero = (
        isinstance(selected_update, int)
        and not isinstance(selected_update, bool)
        and selected_update > 0
    )
    update_zero = selected_update == 0
    nonzero_eligible = bool(
        nonzero
        and _strict_lt(combined_overall, baseline_overall)
        and _strict_lt(combined_high, baseline_high)
    )
    exact_names = (
        "selection_validation_only_pass",
        "analytic_zero_candidate_pass",
        "coarse_baseline_candidate_pass",
        "selected_checkpoint_frozen_pass",
        "baseline_hash_binding_pass",
        "unweighted_mse_against_exact_label_pass",
        "target_unmodified_pass",
        "target_scale_training_only_pass",
        "combined_prediction_loss_pass",
        "model_input_firewall_pass",
        "confirmation_gate_definition_frozen_pass",
        "confirmation_absent_pass",
    )
    checks = {name: _eq_one(selection, name) for name in exact_names}
    checks.update(
        {
            "task_record_count": _check(
                len(tasks),
                "==",
                5,
                len(tasks) == 5,
            ),
            "synthetic_task_unique": _check(
                int(synthetic is not None), "==", 1, synthetic is not None
            ),
            "null_task_unique": _check(
                int(null is not None), "==", 1, null is not None
            ),
            "physical_task_count": _check(
                len(physical), "==", 3, len(physical) == 3
            ),
            "all_tasks_complete": _check(
                int(
                    len(tasks) == 5
                    and all(_one(record.get("complete")) for record in tasks)
                ),
                "==",
                1,
                len(tasks) == 5
                and all(_one(record.get("complete")) for record in tasks),
            ),
            "all_tasks_finite": _check(
                int(
                    len(tasks) == 5
                    and all(_one(record.get("finite")) for record in tasks)
                ),
                "==",
                1,
                len(tasks) == 5
                and all(_one(record.get("finite")) for record in tasks),
            ),
            "physical_model_seeds": _check(
                sorted(record.get("seed") for record in physical),
                "==",
                sorted(t.model_seeds),
                len(physical) == 3
                and sorted(record.get("seed") for record in physical)
                == sorted(t.model_seeds),
            ),
            "maximum_updates": _eq(
                selection, "maximum_updates", t.maximum_updates
            ),
            "validation_interval": _eq(
                selection, "validation_interval", t.validation_interval
            ),
            "batch_size": _eq(selection, "batch_size", t.batch_size),
            "synthetic_relative_validation_mse": _check(
                synthetic_relative,
                "<=",
                t.synthetic_max_relative_validation_mse,
                _finite(synthetic_relative)
                and 0.0
                <= float(synthetic_relative)
                <= t.synthetic_max_relative_validation_mse,
            ),
            "synthetic_every_validation_path_beats_baseline": _check(
                len(synthetic_paths),
                f"== {t.validation_paths} and all >",
                0.0,
                len(synthetic_paths) == t.validation_paths
                and all(value > 0.0 for value in synthetic_paths),
            ),
            "null_selects_update_zero": _check(
                null_update, "==", 0, null_update == 0
            ),
            "physical_selected_update_legal": _check(
                selected_update,
                "in",
                [0, f"1..{t.maximum_updates}"],
                isinstance(selected_update, int)
                and not isinstance(selected_update, bool)
                and 0 <= selected_update <= t.maximum_updates,
            ),
            "nonzero_candidate_eligibility": _check(
                int(nonzero_eligible),
                "==",
                1,
                nonzero_eligible,
            ),
        }
    )
    pipeline_valid = all(
        _one(value["passed"])
        for name, value in checks.items()
        if name != "nonzero_candidate_eligibility"
    )
    result = _gate(
        "train",
        checks,
        selected_update=selected_update,
        optimization_pipeline_valid=int(pipeline_valid),
        coarse_baseline_only=int(update_zero),
        residual_candidate_nonzero=int(nonzero),
        residual_candidate_eligible=int(nonzero_eligible),
    )
    result["confirmation_generation_authorized"] = int(
        result["passed"] and nonzero_eligible
    )
    result["thresholds"] = t.to_dict()
    return result


def _contrast(
    record: Mapping[str, Any], name: str, scope: str
) -> tuple[float | None, tuple[float, ...]]:
    contrasts = record.get("contrasts")
    item: Mapping[str, Any] = {}
    if isinstance(contrasts, Mapping):
        raw = contrasts.get(name)
        if isinstance(raw, Mapping):
            item = raw
    point = item.get(f"{scope}_mean", item.get(f"{scope}_point_estimate"))
    paths = _float_sequence(
        item.get(f"{scope}_path_values", item.get(f"{scope}_values", ()))
    )
    return (float(point) if _finite(point) else None), paths


def _max_t_fields(record: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    points_raw = record.get("point_estimates", {})
    lower_raw = record.get("lower_bounds", {})
    if not isinstance(points_raw, Mapping) or not isinstance(lower_raw, Mapping):
        return {}, {}
    try:
        points = {str(key): float(value) for key, value in points_raw.items()}
        lower = {str(key): float(value) for key, value in lower_raw.items()}
    except (TypeError, ValueError):
        return {}, {}
    if not all(math.isfinite(value) for value in (*points.values(), *lower.values())):
        return {}, {}
    return points, lower


_MAX_T_FAMILY = (
    "overall.baseline_vs_zero",
    "overall.combined_vs_baseline",
)


def evaluate_coarse_residual_confirmation(
    *,
    confirmation: Mapping[str, Any],
    thresholds: CoarseResidualThresholds | None = None,
) -> dict[str, Any]:
    """Gate one fresh 64-path audit using simultaneous paired-risk bounds."""

    t = thresholds or CoarseResidualThresholds()
    max_t_raw = confirmation.get("max_t")
    max_t = max_t_raw if isinstance(max_t_raw, Mapping) else {}
    points, lower = _max_t_fields(max_t)
    family_names = tuple(max_t.get("family_names", ()))
    delta_b_positive = bool(
        points and points.get("overall.baseline_vs_zero", -math.inf) > 0.0
    )
    delta_r_positive = bool(
        points and points.get("overall.combined_vs_baseline", -math.inf) > 0.0
    )
    delta_b_replicated = bool(
        lower and lower.get("overall.baseline_vs_zero", -math.inf) > 0.0
    )
    delta_r_replicated = bool(
        lower and lower.get("overall.combined_vs_baseline", -math.inf) > 0.0
    )
    direct_error = confirmation.get("direct_derived_delta_t_max_abs_error")
    direct_valid = bool(
        _finite(direct_error)
        and 0.0 <= float(direct_error) <= t.direct_derived_tolerance
    )
    inference_valid = bool(
        max_t.get("method")
        in {
            "centered_whole_path_studentized_max_t",
            "studentized_whole_path_max_t",
            "whole_path_studentized_max_t",
            "paired_whole_path_max_t",
        }
        and float(max_t.get("confidence", -1.0)) == t.confidence
        and int(max_t.get("replicates", -1)) == t.bootstrap_replicates
        and int(max_t.get("seed", -1)) == t.bootstrap_seed
        and int(max_t.get("path_count", -1)) == t.confirmation_paths
        and set(family_names) == set(_MAX_T_FAMILY)
        and len(family_names) == len(_MAX_T_FAMILY)
        and set(points) == set(_MAX_T_FAMILY)
        and set(lower) == set(_MAX_T_FAMILY)
        and _finite(max_t.get("critical_value"))
        and float(max_t.get("critical_value", 0.0)) > 0.0
        and max_t.get("bootstrap_unit")
        == "whole_path_jointly_across_family"
        and max_t.get("quantile_method") == "higher"
        and _zero(max_t.get("negative_values_truncated"))
        and direct_valid
    )
    exact_names = (
        "confirmation_cache_gate_pass",
        "confirmation_sealed_pass",
        "confirmation_opened_once_pass",
        "confirmation_paths_fresh_pass",
        "confirmation_paths_disjoint_pass",
        "selected_checkpoint_hash_pass",
        "baseline_hash_binding_pass",
        "path_plan_hash_pass",
        "predictions_finite_pass",
        "risks_finite_pass",
        "unweighted_exact_label_risk_pass",
        "model_input_firewall_pass",
        "no_post_selection_refit_pass",
    )
    checks = {name: _eq_one(confirmation, name) for name in exact_names}
    checks.update(
        {
            "confirmation_path_count": _eq(
                confirmation, "confirmation_path_count", t.confirmation_paths
            ),
            "max_t_inference_valid": _check(
                int(inference_valid), "==", 1, inference_valid
            ),
            "delta_b_simultaneous_lower_bounds": _check(
                lower.get("overall.baseline_vs_zero"),
                ">",
                0.0,
                delta_b_replicated,
            ),
            "delta_r_simultaneous_lower_bounds": _check(
                lower.get("overall.combined_vs_baseline"),
                ">",
                0.0,
                delta_r_replicated,
            ),
            "direct_derived_delta_t_agreement": _check(
                direct_error,
                "<=",
                t.direct_derived_tolerance,
                direct_valid,
            ),
        }
    )
    result = _gate(
        "confirm",
        checks,
        paired_risk_inference_valid=int(inference_valid),
        coarse_baseline_point_positive=int(delta_b_positive),
        coarse_baseline_replicated=int(delta_b_replicated),
        residual_point_positive=int(delta_r_positive),
        residual_replicated=int(delta_r_replicated),
        max_t_family=list(_MAX_T_FAMILY),
        max_t_record=dict(max_t),
    )
    result["reverse_controller_planning_authorized"] = int(result["passed"])
    result["thresholds"] = t.to_dict()
    return result


def decide_coarse_residual_workflow(
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the closed scientific decision partition."""

    if _status(preflight_gate) == "not_evaluated":
        decision = "ready_for_preflight"
        action = "verify immutable parents, literal baseline, and fresh design"
    elif not _passed(preflight_gate):
        failure_domain = str((preflight_gate or {}).get("failure_domain", ""))
        if (
            _status(preflight_gate) == "execution_failed"
            and failure_domain == "provenance"
        ) or (
            _status(preflight_gate) != "execution_failed"
            and not _one((preflight_gate or {}).get("provenance_valid"))
        ):
            decision = CoarseResidualDecision.CONTROL_PROVENANCE_INVALID.value
            action = "repair immutable parent binding"
        elif (
            _status(preflight_gate) == "execution_failed"
            and failure_domain == "baseline_derivation"
        ) or (
            _status(preflight_gate) != "execution_failed"
            and not _one((preflight_gate or {}).get("baseline_derivation_valid"))
        ):
            decision = (
                CoarseResidualDecision.COARSE_BASELINE_DERIVATION_INVALID.value
            )
            action = "repair the literal frozen baseline derivation"
        else:
            decision = CoarseResidualDecision.COARSE_RESIDUAL_DESIGN_INFEASIBLE.value
            action = "repair fresh path/resource design without weakening science"
    elif _status(cache_gate) == "not_evaluated":
        decision = "ready_for_cache"
        action = "generate fresh exact train and validation caches"
    elif not _passed(cache_gate):
        decision = CoarseResidualDecision.FRESH_EXACT_CACHE_INVALID.value
        action = "repair fresh exact cache generation"
    elif _status(train_gate) == "not_evaluated":
        decision = "ready_for_train"
        action = "run synthetic/null controls and physical validation selection"
    elif _one((train_gate or {}).get("coarse_baseline_only")) and _one(
        (train_gate or {}).get("optimization_pipeline_valid")
    ):
        decision = CoarseResidualDecision.COARSE_BASELINE_ONLY_SIGNAL.value
        action = "retain the coarse baseline; do not claim a learned residual"
    elif not _passed(train_gate):
        decision = (
            CoarseResidualDecision.RESIDUAL_OPTIMIZATION_PIPELINE_INVALID.value
        )
        action = "repair optimization or selection controls"
    elif _status(confirm_gate) == "not_evaluated":
        decision = "ready_for_confirm"
        action = "open the fresh sealed 64-path audit exactly once"
    elif _zero((confirm_gate or {}).get("confirmation_cache_valid")):
        decision = CoarseResidualDecision.FRESH_EXACT_CACHE_INVALID.value
        action = "repair the fresh sealed confirmation cache"
    elif not _one((confirm_gate or {}).get("paired_risk_inference_valid")):
        decision = CoarseResidualDecision.PAIRED_RISK_INFERENCE_INVALID.value
        action = "repair report-only paired max-T inference"
    elif not _one((confirm_gate or {}).get("coarse_baseline_replicated")):
        decision = CoarseResidualDecision.COARSE_BASELINE_NONREPLICATING.value
        action = "retain the witness but do not claim operational replication"
    elif not _one((confirm_gate or {}).get("residual_replicated")):
        if not _one((confirm_gate or {}).get("residual_point_positive")):
            decision = CoarseResidualDecision.SELECTION_FALSE_DISCOVERY.value
            action = "record that validation selection did not replicate"
        else:
            decision = (
                CoarseResidualDecision.COARSE_RESIDUAL_AUDIT_INCONCLUSIVE.value
            )
            action = "retain sealed audit; do not resize or regenerate it"
    elif not _passed(confirm_gate):
        decision = CoarseResidualDecision.PAIRED_RISK_INFERENCE_INVALID.value
        action = "repair confirmation integrity without changing evidence"
    else:
        decision = CoarseResidualDecision.EXACT_RB_COARSE_RESIDUAL_LEARNABLE.value
        action = "plan a separate exact-target reverse-controller control gate"
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "decision": decision,
        "recommended_next_action": action,
        "claim_scope": CLAIM_SCOPE,
        "reverse_controller_planning_authorized": int(
            decision
            == CoarseResidualDecision.EXACT_RB_COARSE_RESIDUAL_LEARNABLE.value
        ),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


def evaluate_coarse_residual_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    require_gate: str = "none",
) -> dict[str, Any]:
    if require_gate not in REQUIRED_GATES:
        raise ValueError(f"require_gate must be one of {REQUIRED_GATES}")
    components = {
        "preflight": dict(
            preflight_gate
            or not_evaluated_gate("preflight", "preflight has not run")
        ),
        "cache": dict(cache_gate or not_evaluated_gate("cache", "cache has not run")),
        "train": dict(train_gate or not_evaluated_gate("train", "train has not run")),
        "confirm": dict(
            confirm_gate or not_evaluated_gate("confirm", "confirm has not run")
        ),
    }
    cumulative = {
        "none": True,
        "preflight": _passed(components["preflight"]),
        "cache": _passed(components["preflight"]) and _passed(components["cache"]),
        "train": (
            _passed(components["preflight"])
            and _passed(components["cache"])
            and _passed(components["train"])
        ),
        "confirm": (
            _passed(components["preflight"])
            and _passed(components["cache"])
            and _passed(components["train"])
            and _passed(components["confirm"])
        ),
    }
    decision = decide_coarse_residual_workflow(
        components["preflight"],
        components["cache"],
        components["train"],
        components["confirm"],
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "passed": int(cumulative[require_gate]),
        "required_gate_pass": int(cumulative[require_gate]),
        "components": components,
        "decision": decision,
        "reverse_controller_planning_authorized": int(
            decision["reverse_controller_planning_authorized"]
        ),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


__all__ = [
    "BASELINE_AVERAGED_TABLE_NOISE",
    "BASELINE_ENERGY",
    "BASELINE_PANEL_MEAN_NOISE",
    "BASELINE_RAW_VALUES_SERIALIZATION_SHA256",
    "BASELINE_RAW_VALUES_SHA256",
    "BASELINE_SCHEMA",
    "BASELINE_SHAPE",
    "BASELINE_SHRINKAGE",
    "BASELINE_SIGNAL_ENERGY",
    "BASELINE_VALUES_SHA256",
    "BASELINE_VALUES_SERIALIZATION_SHA256",
    "CLAIM_SCOPE",
    "CoarseResidualDecision",
    "CoarseResidualThresholds",
    "NO_CLAIM_AUTHORIZATION",
    "REQUIRED_GATES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "STAGES",
    "WITNESS_REGISTRY_SHA256",
    "decide_coarse_residual_workflow",
    "evaluate_coarse_residual_cache",
    "evaluate_coarse_residual_cache_set",
    "evaluate_coarse_residual_confirmation",
    "evaluate_coarse_residual_preflight",
    "evaluate_coarse_residual_train",
    "evaluate_coarse_residual_workflow",
    "execution_failed_gate",
    "not_evaluated_gate",
]
