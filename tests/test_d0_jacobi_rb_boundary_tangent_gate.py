from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    BoundaryTangentDecision,
    BoundaryTangentGateError,
    BoundaryTangentThresholds,
    CLAIM_SCOPE,
    COMBINED_VS_BASELINE_NAMES,
    COMBINED_VS_ZERO_NAMES,
    CONFIRMATION_FAMILY_NAMES,
    CONFIRMATION_FAMILY_SIZE,
    CONTROLLER_FAMILY_SIZE,
    claim_scope_flags,
    decide_boundary_tangent_workflow,
    evaluate_confirmation_gate,
    evaluate_controller_gate,
    evaluate_numerical_resource_gate,
    one_sided_whole_path_max_t,
    validate_confirmation_family,
    validate_controller_family,
)


def _thresholds() -> BoundaryTangentThresholds:
    return replace(
        BoundaryTangentThresholds(),
        bootstrap_replicates=96,
        bootstrap_seed=17,
        controller_bootstrap_seed=18,
    )


def _confirmation_record(
    *, thresholds: BoundaryTangentThresholds | None = None
) -> dict[str, object]:
    t = thresholds or _thresholds()
    rng = np.random.default_rng(123)
    values = np.ascontiguousarray(
        2.0 + 0.04 * rng.normal(size=(t.confirmation_paths, CONFIRMATION_FAMILY_SIZE)),
        dtype=np.float64,
    )
    return one_sided_whole_path_max_t(
        values,
        path_ids=np.arange(100, 100 + t.confirmation_paths, dtype=np.int64),
        confidence=t.simultaneous_confidence,
        replicates=t.bootstrap_replicates,
        seed=t.bootstrap_seed,
        chunk_size=13,
    )


def _controller_record(
    *,
    thresholds: BoundaryTangentThresholds | None = None,
    bias: float = 0.08,
    refinement: float = 0.04,
) -> dict[str, object]:
    t = thresholds or _thresholds()
    names = tuple(f"one_phase.f{i}.bias" for i in range(392)) + tuple(
        f"one_phase.f{i}.M8_vs_M4" for i in range(392)
    )
    upper = {
        name: float(bias if name.endswith(".bias") else refinement)
        for name in names
    }
    return {
        "method": "whole_path_rms_normalized_two_sided_studentized_max_t",
        "bootstrap_unit": "whole_path",
        "denominator_recomputed_per_resample": 1,
        "quantile_method": "higher",
        "family_size": len(names),
        "family_names": list(names),
        "path_count": t.controller_paths,
        "path_ids": list(range(0xED000, 0xED000 + t.controller_paths)),
        "confidence": t.simultaneous_confidence,
        "replicates": t.bootstrap_replicates,
        "seed": t.controller_bootstrap_seed,
        "negative_values_truncated": 0,
        "critical_value": 3.0,
        "point_estimates": {name: 0.0 for name in names},
        "standard_errors": {name: 0.01 for name in names},
        "simultaneous_upper_absolute": upper,
    }


def _healthy_metrics() -> dict[str, object]:
    return {
        "certificate_fraction": 1.0,
        "states_finite": 1,
        "states_nonnegative": 1,
        "maximum_pair_mass_error": 1.0e-13,
        "maximum_simplex_mass_error": 2.0e-13,
        "boundary_rejection_count": 0,
        "forbidden_counts": {"nonfinite_count": 0, "floor_count": 0},
        "controller_forbidden_counts": {"clip_count": 0, "projection_count": 0},
        "fallback_fraction": 0.0,
        "fallback_time_fraction": 0.0,
        "transitions_per_second": 2000.0,
        "peak_device_memory_fraction": 0.25,
        "total_persisted_bytes": 1024,
    }


def _gate(*, passed: bool, **fields: object) -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "passed": int(passed),
        **fields,
    }


def _passing_gates() -> dict[str, dict[str, object]]:
    return {
        "preflight_gate": _gate(
            passed=True,
            provenance_valid=1,
            failed_controller_adjudication_valid=1,
            boundary_tangent_representation_valid=1,
        ),
        "cache_gate": _gate(passed=True),
        "train_gate": _gate(
            passed=True,
            baseline_valid=1,
            optimization_pipeline_valid=1,
            baseline_only=0,
            physical_training_performed=1,
        ),
        "confirm_gate": _gate(
            passed=True,
            paired_risk_inference_valid=1,
            combined_vs_baseline_point_positive=1,
            combined_vs_zero_point_positive=1,
            combined_vs_baseline_replicated=1,
            combined_vs_zero_replicated=1,
        ),
        "control_gate": _gate(
            passed=True,
            controller_family_valid=1,
            numerically_valid=1,
            resource_valid=1,
            weak_law_controlled=1,
            microstep_refinement_controlled=1,
            controller_control_trajectory_performed=1,
        ),
    }


def test_confirmation_family_is_exact_224_plus_4_order() -> None:
    assert len(CONFIRMATION_FAMILY_NAMES) == 228
    assert len(COMBINED_VS_ZERO_NAMES) == 224
    assert len(COMBINED_VS_BASELINE_NAMES) == 4
    assert CONFIRMATION_FAMILY_NAMES[0] == "combined_vs_zero.q0.phase0.midpoint0"
    assert CONFIRMATION_FAMILY_NAMES[223] == "combined_vs_zero.q3.phase6.midpoint7"
    assert CONFIRMATION_FAMILY_NAMES[224:] == (
        "combined_vs_baseline.q0",
        "combined_vs_baseline.q1",
        "combined_vs_baseline.q2",
        "combined_vs_baseline.q3",
    )


def test_one_sided_max_t_is_deterministic_order_and_chunk_invariant() -> None:
    rng = np.random.default_rng(456)
    values = np.ascontiguousarray(
        0.7 + rng.normal(scale=0.05, size=(64, 228)), dtype=np.float64
    )
    paths = np.arange(900, 964, dtype=np.int64)
    first = one_sided_whole_path_max_t(
        values,
        path_ids=paths,
        confidence=0.9,
        replicates=80,
        seed=7,
        namespace=2,
        chunk_size=7,
    )
    permutation = rng.permutation(64)
    second = one_sided_whole_path_max_t(
        values[permutation],
        path_ids=paths[permutation],
        confidence=0.9,
        replicates=80,
        seed=7,
        namespace=2,
        chunk_size=31,
    )
    assert first == second
    assert first["passed"] == 1
    assert first["bootstrap_unit"] == "whole_path_jointly_across_family"
    assert first["negative_values_truncated"] == 0


def test_one_sided_max_t_rejects_wrong_dtype_family_and_degeneracy() -> None:
    values = np.ones((8, 228), dtype=np.float64)
    with pytest.raises(BoundaryTangentGateError, match="float64"):
        one_sided_whole_path_max_t(values.astype(np.float32), replicates=4)
    with pytest.raises(BoundaryTangentGateError, match="family names"):
        one_sided_whole_path_max_t(
            values,
            names=tuple(reversed(CONFIRMATION_FAMILY_NAMES)),
            replicates=4,
        )
    with pytest.raises(BoundaryTangentGateError, match="degenerate"):
        one_sided_whole_path_max_t(values, replicates=4)


def test_confirmation_validation_and_gate_are_fail_closed() -> None:
    thresholds = _thresholds()
    record = _confirmation_record(thresholds=thresholds)
    summary = validate_confirmation_family(record, thresholds=thresholds)
    assert summary == {
        "paired_risk_inference_valid": 1,
        "family_size": 228,
        "combined_vs_zero_point_positive": 1,
        "combined_vs_zero_replicated": 1,
        "combined_vs_baseline_point_positive": 1,
        "combined_vs_baseline_replicated": 1,
        "all_simultaneous_lower_bounds_positive": 1,
    }
    gate = evaluate_confirmation_gate(
        record,
        integrity_checks={"sealed_confirmation": True},
        thresholds=thresholds,
    )
    assert gate["passed"] == 1
    broken = dict(record)
    broken["family_names"] = list(reversed(record["family_names"]))
    failed = evaluate_confirmation_gate(broken, thresholds=thresholds)
    assert failed["passed"] == 0
    assert failed["paired_risk_inference_valid"] == 0
    assert failed["inference_error"]


def test_confirmation_zero_and_baseline_boundaries_are_distinct() -> None:
    thresholds = _thresholds()
    record = _confirmation_record(thresholds=thresholds)
    lower = dict(record["lower_bounds"])
    lower[COMBINED_VS_ZERO_NAMES[0]] = 0.0
    record["lower_bounds"] = lower
    record["passed"] = 0
    summary = validate_confirmation_family(record, thresholds=thresholds)
    assert summary["combined_vs_zero_replicated"] == 0
    assert summary["combined_vs_baseline_replicated"] == 1


def test_controller_784_family_thresholds_are_validated_separately() -> None:
    thresholds = _thresholds()
    valid = validate_controller_family(
        _controller_record(thresholds=thresholds), thresholds=thresholds
    )
    assert valid["trajectory_family_size"] == CONTROLLER_FAMILY_SIZE
    assert valid["weak_law_controlled"] == 1
    assert valid["microstep_refinement_controlled"] == 1
    weak = validate_controller_family(
        _controller_record(thresholds=thresholds, bias=0.1000001),
        thresholds=thresholds,
    )
    assert weak["weak_law_controlled"] == 0
    refine = validate_controller_family(
        _controller_record(thresholds=thresholds, refinement=0.0500001),
        thresholds=thresholds,
    )
    assert refine["microstep_refinement_controlled"] == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("path_count", None),
        ("path_ids", None),
        ("path_ids", [0] * 8),
        ("negative_values_truncated", None),
        ("negative_values_truncated", 1),
    ),
)
def test_controller_family_fails_closed_on_path_or_truncation_tamper(
    field: str, replacement: object
) -> None:
    thresholds = _thresholds()
    record = _controller_record(thresholds=thresholds)
    if replacement is None:
        record.pop(field)
    else:
        record[field] = replacement
    with pytest.raises(BoundaryTangentGateError, match="configuration"):
        validate_controller_family(record, thresholds=thresholds)


def test_numerical_and_resource_failures_remain_distinct() -> None:
    thresholds = _thresholds()
    valid = evaluate_numerical_resource_gate(
        _healthy_metrics(), thresholds=thresholds
    )
    assert valid["passed"] == 1
    assert valid["numerically_valid"] == 1
    assert valid["resource_valid"] == 1
    numerical_metrics = _healthy_metrics()
    numerical_metrics["maximum_pair_mass_error"] = 3.0e-12
    numerical = evaluate_numerical_resource_gate(
        numerical_metrics, thresholds=thresholds
    )
    assert numerical["numerically_valid"] == 0
    assert numerical["resource_valid"] == 1
    resource_metrics = _healthy_metrics()
    resource_metrics["transitions_per_second"] = 1299.0
    resource = evaluate_numerical_resource_gate(
        resource_metrics, thresholds=thresholds
    )
    assert resource["numerically_valid"] == 1
    assert resource["resource_valid"] == 0


def test_controller_gate_combines_family_health_and_resources() -> None:
    thresholds = _thresholds()
    gate = evaluate_controller_gate(
        _controller_record(thresholds=thresholds),
        _healthy_metrics(),
        integrity_checks={"at_most_eight_phases": True},
        thresholds=thresholds,
    )
    assert gate["passed"] == 1
    assert gate["controller_family_valid"] == 1
    assert gate["weak_law_controlled"] == 1
    assert gate["microstep_refinement_controlled"] == 1
    assert gate["physical_training_performed"] == 1
    assert gate["controller_control_trajectory_performed"] == 1
    metrics = _healthy_metrics()
    metrics["controller_forbidden_counts"] = {"clip_count": 1}
    failed = evaluate_controller_gate(
        _controller_record(thresholds=thresholds), metrics, thresholds=thresholds
    )
    assert failed["passed"] == 0
    assert failed["numerically_valid"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            ("preflight_gate", {"passed": 0, "provenance_valid": 0}),
            BoundaryTangentDecision.CONTROL_PROVENANCE_INVALID.value,
        ),
        (
            (
                "preflight_gate",
                {
                    "passed": 0,
                    "provenance_valid": 1,
                    "failed_controller_adjudication_valid": 0,
                },
            ),
            BoundaryTangentDecision.FAILED_CONTROLLER_ADJUDICATION_INVALID.value,
        ),
        (
            (
                "preflight_gate",
                {
                    "passed": 0,
                    "provenance_valid": 1,
                    "failed_controller_adjudication_valid": 1,
                    "boundary_tangent_representation_valid": 0,
                },
            ),
            BoundaryTangentDecision.BOUNDARY_TANGENT_REPRESENTATION_INVALID.value,
        ),
        (
            (
                "preflight_gate",
                {
                    "passed": 0,
                    "provenance_valid": 1,
                    "failed_controller_adjudication_valid": 1,
                    "boundary_tangent_representation_valid": 1,
                },
            ),
            BoundaryTangentDecision.BOUNDARY_TANGENT_DESIGN_INFEASIBLE.value,
        ),
        (
            ("cache_gate", {"passed": 0}),
            BoundaryTangentDecision.FRESH_EXACT_CACHE_INVALID.value,
        ),
        (
            ("train_gate", {"passed": 0, "baseline_valid": 0}),
            BoundaryTangentDecision.BOUNDARY_TANGENT_BASELINE_INVALID.value,
        ),
        (
            (
                "train_gate",
                {"passed": 0, "baseline_valid": 1, "optimization_pipeline_valid": 0},
            ),
            BoundaryTangentDecision.BOUNDARY_TANGENT_OPTIMIZATION_PIPELINE_INVALID.value,
        ),
        (
            (
                "train_gate",
                {
                    "passed": 1,
                    "baseline_valid": 1,
                    "optimization_pipeline_valid": 1,
                    "baseline_only": 1,
                },
            ),
            BoundaryTangentDecision.BOUNDARY_TANGENT_BASELINE_ONLY_SIGNAL.value,
        ),
        (
            (
                "confirm_gate",
                {
                    "passed": 0,
                    "paired_risk_inference_valid": 1,
                    "combined_vs_baseline_point_positive": 0,
                },
            ),
            BoundaryTangentDecision.SELECTION_FALSE_DISCOVERY.value,
        ),
        (
            (
                "confirm_gate",
                {
                    "passed": 0,
                    "paired_risk_inference_valid": 1,
                    "combined_vs_baseline_point_positive": 1,
                    "combined_vs_zero_point_positive": 0,
                },
            ),
            BoundaryTangentDecision.BOUNDARY_TANGENT_TIME_LOCAL_SIGNAL_NOT_DETECTED.value,
        ),
        (
            (
                "confirm_gate",
                {
                    "passed": 0,
                    "paired_risk_inference_valid": 1,
                    "combined_vs_baseline_point_positive": 1,
                    "combined_vs_zero_point_positive": 1,
                    "combined_vs_baseline_replicated": 1,
                    "combined_vs_zero_replicated": 0,
                },
            ),
            BoundaryTangentDecision.BOUNDARY_TANGENT_AUDIT_INCONCLUSIVE.value,
        ),
        (
            ("confirm_gate", {"passed": 0, "paired_risk_inference_valid": 0}),
            BoundaryTangentDecision.PAIRED_RISK_INFERENCE_INVALID.value,
        ),
        (
            (
                "control_gate",
                {
                    "passed": 0,
                    "controller_family_valid": 1,
                    "numerically_valid": 0,
                    "resource_valid": 1,
                },
            ),
            BoundaryTangentDecision.BOUNDARY_TANGENT_CONTROLLER_NUMERICALLY_INVALID.value,
        ),
        (
            (
                "control_gate",
                {
                    "passed": 0,
                    "controller_family_valid": 1,
                    "numerically_valid": 1,
                    "resource_valid": 1,
                    "weak_law_controlled": 0,
                },
            ),
            BoundaryTangentDecision.REVERSE_CONTROLLER_WEAK_LAW_FAILED.value,
        ),
        (
            (
                "control_gate",
                {
                    "passed": 0,
                    "controller_family_valid": 1,
                    "numerically_valid": 1,
                    "resource_valid": 1,
                    "weak_law_controlled": 1,
                    "microstep_refinement_controlled": 0,
                },
            ),
            BoundaryTangentDecision.REVERSE_CONTROLLER_MICROSTEP_REFINEMENT_FAILED.value,
        ),
    ),
)
def test_closed_decision_partition(
    mutation: tuple[str, dict[str, object]], expected: str
) -> None:
    gates = _passing_gates()
    stage, changes = mutation
    gates[stage].update(changes)
    result = decide_boundary_tangent_workflow(**gates)
    assert result["decision"] == expected
    assert result["one_image_reconstruction_control_planning_authorized"] == 0


def test_success_decision_has_exact_claim_scope_without_sampling() -> None:
    result = decide_boundary_tangent_workflow(**_passing_gates())
    assert result["decision"] == (
        BoundaryTangentDecision.EXACT_RB_BOUNDARY_TANGENT_CONTROLLER_CONTROLLED.value
    )
    assert result["claim_scope"] == CLAIM_SCOPE
    assert result["one_image_reconstruction_control_planning_authorized"] == 1
    assert result["physical_training_performed"] == 1
    assert result["controller_control_trajectory_performed"] == 1
    for name in (
        "reverse_sampling_authorized",
        "sampling_authorized",
        "reconstruction_authorized",
        "reverse_sampling_performed",
        "sampling_performed",
        "image_sampling_performed",
        "reconstruction_performed",
        "full_reverse_path_performed",
    ):
        assert result[name] == 0


def test_claim_flags_never_authorize_sampling_or_reconstruction() -> None:
    flags = claim_scope_flags(
        controlled=True,
        physical_training_performed=True,
        controller_control_trajectory_performed=True,
    )
    assert flags["claim_scope"] == CLAIM_SCOPE
    assert flags["physical_training_performed"] == 1
    assert flags["boundary_tangent_controller_controlled"] == 1
    assert flags["sampling_authorized"] == 0
    assert flags["reconstruction_authorized"] == 0
    assert flags["sampling_performed"] == 0
    assert flags["reconstruction_performed"] == 0


def test_module_does_not_import_training_kernel_or_sampler() -> None:
    import mnist.d0_jacobi_rb_boundary_tangent_gate as gate

    text = Path(gate.__file__).read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "reverse_sampler" not in text
    assert "sample_alpha1" not in text
