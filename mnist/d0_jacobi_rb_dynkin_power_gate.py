"""Fail-closed gates for the exact Dynkin Strang power feasibility workflow.

This module contains no transition kernel, trainer, or reverse sampler.  Its
only statistical construction is the frozen normal/chi-square/Bonferroni
engineering forecast inherited from the failed raw-endpoint pilot.  Those
forecasts select a later experiment; they are never labelled scientific
confidence intervals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "experiment12-d0-jacobi-rb-dynkin-power-gate"
SCHEMA_VERSION = 1
PLANNING_VERSION = "normal-chi-square-bonferroni-successive-differences-v2"
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


@dataclass(frozen=True)
class DynkinPowerThresholds:
    """Frozen thresholds for the controls-only feasibility claim."""

    parent_record_count: int = 1308
    parent_source_count: int = 15
    grid_size: int = 28
    cell_count: int = 784
    alpha: float = 1.0
    tau_eff: float = 5.0e-5
    levels: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    observation_time_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    observable_count: int = 10
    main_successive_difference_count: int = 3
    main_feature_count: int = 120
    reference_feature_count: int = 40
    pilot_panel_count: int = 2
    pilot_paths_per_panel: int = 8
    tower_panel_count: int = 2
    tower_clusters_per_panel: int = 128
    tower_confidence: float = 0.99
    planning_confidence: float = 0.99
    candidate_main_paths: tuple[int, ...] = (32, 64)
    candidate_reference_paths: tuple[int, ...] = (16, 32)
    maximum_main_half_width: float = 0.0025
    maximum_reference_half_width: float = 0.005
    maximum_projected_hours: float = 48.0
    minimum_rate: float = 1300.0
    maximum_float64_phase_moment_error: float = 1.0e-10
    maximum_cuda_phase_moment_error: float = 2.0e-6
    maximum_cumulative_standardized_error: float = 1.0e-8
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_cost_fraction: float = 0.10
    maximum_peak_memory_fraction: float = 0.80
    maximum_cuda_mass_error: float = 2.0e-6
    root_seed: int = 261_161

    def __post_init__(self) -> None:
        frozen: dict[str, Any] = {
            "parent_record_count": 1308,
            "parent_source_count": 15,
            "grid_size": 28,
            "cell_count": 784,
            "alpha": 1.0,
            "tau_eff": 5.0e-5,
            "levels": (128, 256, 512, 1024, 2048),
            "observation_time_fractions": (0.25, 0.5, 0.75, 1.0),
            "observable_count": 10,
            "main_successive_difference_count": 3,
            "main_feature_count": 120,
            "reference_feature_count": 40,
            "pilot_panel_count": 2,
            "pilot_paths_per_panel": 8,
            "tower_panel_count": 2,
            "tower_clusters_per_panel": 128,
            "tower_confidence": 0.99,
            "planning_confidence": 0.99,
            "candidate_main_paths": (32, 64),
            "candidate_reference_paths": (16, 32),
            "maximum_main_half_width": 0.0025,
            "maximum_reference_half_width": 0.005,
            "maximum_projected_hours": 48.0,
            "minimum_rate": 1300.0,
            "maximum_float64_phase_moment_error": 1.0e-10,
            "maximum_cuda_phase_moment_error": 2.0e-6,
            "maximum_cumulative_standardized_error": 1.0e-8,
            "maximum_fallback_fraction": 1.0e-4,
            "maximum_fallback_cost_fraction": 0.10,
            "maximum_peak_memory_fraction": 0.80,
            "maximum_cuda_mass_error": 2.0e-6,
            "root_seed": 261_161,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen at {expected}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "levels",
            "observation_time_fractions",
            "candidate_main_paths",
            "candidate_reference_paths",
        ):
            result[name] = list(result[name])
        return result


class DynkinPowerDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    REFINEMENT_SCHEDULER_INVALID = "refinement_scheduler_invalid"
    PARENT_POWER_ADJUDICATION_INVALID = "parent_power_adjudication_invalid"
    JACOBI_PHASE_MOMENT_ALGEBRA_INVALID = "jacobi_phase_moment_algebra_invalid"
    DYNKIN_ESTIMATOR_NUMERICALLY_UNRESOLVED = (
        "dynkin_estimator_numerically_unresolved"
    )
    DYNKIN_TOWER_IDENTITY_INVALID = "dynkin_tower_identity_invalid"
    DYNKIN_COMPUTATIONALLY_INFEASIBLE = "dynkin_computationally_infeasible"
    DYNKIN_POWER_INFEASIBLE = "dynkin_power_infeasible"
    DYNKIN_PANELS_DISAGREE = "dynkin_panels_disagree"
    EXACT_DYNKIN_REFINEMENT_ESTIMATOR_FEASIBLE = (
        "exact_dynkin_refinement_estimator_feasible"
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int, np.bool_, np.integer)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int, np.bool_, np.integer)) and int(value) == 0


def _status(gate: Mapping[str, Any] | None) -> str:
    if not isinstance(gate, Mapping):
        return "not_evaluated"
    return str(gate.get("evaluation_status", "not_evaluated"))


def _passed(value: bool | int | Mapping[str, Any] | None) -> bool:
    if isinstance(value, Mapping):
        return _status(value) == "evaluated" and _one(value.get("passed"))
    return value is True or _one(value)


def _explicit_provenance_invalid(
    value: bool | int | Mapping[str, Any],
) -> bool:
    """Return whether verified evidence explicitly rejects provenance.

    An execution-failure record or a mapping with missing aggregate fields is
    not provenance evidence.  Provenance verifiers normally raise on
    incompatibility; the scalar false form remains the explicit decision-API
    representation used by callers and tests.
    """

    if not isinstance(value, Mapping):
        return value is False or _zero(value)
    if _status(value) != "evaluated":
        return False
    if _zero(value.get("provenance_valid")):
        return True
    subchecks = value.get("subchecks")
    if not isinstance(subchecks, Mapping):
        return False
    return any(
        (
            "provenance" in str(name)
            or "parent_source" in str(name)
            or "parent_record" in str(name)
        )
        and isinstance(check, Mapping)
        and _zero(check.get("passed"))
        and check.get("value") is not None
        for name, check in subchecks.items()
    )


def _execution_failure_domain(gate: Mapping[str, Any] | None) -> str:
    if not isinstance(gate, Mapping):
        return ""
    return (
        str(gate.get("failure_domain", ""))
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _execution_failure_decision(
    gate: Mapping[str, Any],
) -> tuple[DynkinPowerDecision, str]:
    domain = _execution_failure_domain(gate)
    if "scheduler" in domain or "configuration" in domain or domain == "config":
        return (
            DynkinPowerDecision.REFINEMENT_SCHEDULER_INVALID,
            "repair the frozen path-ID plan or refinement scheduler",
        )
    if "resource" in domain or "comput" in domain:
        return (
            DynkinPowerDecision.DYNKIN_COMPUTATIONALLY_INFEASIBLE,
            "repair exact observer execution within the frozen resource limits",
        )
    return (
        DynkinPowerDecision.DYNKIN_ESTIMATOR_NUMERICALLY_UNRESOLVED,
        "repair the failed exact Dynkin stage execution",
    )


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


def _equal(
    metrics: Mapping[str, Any], name: str, expected: Any
) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", expected, value == expected)


def _sequence_equal(
    metrics: Mapping[str, Any], name: str, expected: Sequence[Any]
) -> dict[str, Any]:
    value = metrics.get(name)
    try:
        actual = list(value)
    except TypeError:
        actual = value
    target = list(expected)
    return _check(actual, "==", target, actual == target)


def _le(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    passed = _finite(value) and float(value) <= float(threshold)
    return _check(value, "<=", threshold, passed)


def _ge(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    passed = _finite(value) and float(value) >= float(threshold)
    return _check(value, ">=", threshold, passed)


def _gate(
    name: str,
    claim_scope: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_status: str = "evaluated",
) -> dict[str, Any]:
    passed = (
        evaluation_status == "evaluated"
        and bool(checks)
        and all(_one(record.get("passed")) for record in checks.values())
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "claim_scope": claim_scope,
        "evaluation_status": evaluation_status,
        "passed": int(passed),
        "subchecks": {str(key): dict(value) for key, value in checks.items()},
        **NO_WORK,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    result = _gate(name, "not evaluated", {}, evaluation_status="not_evaluated")
    result["reason"] = str(reason)
    return result


def normal_chi_square_bonferroni_projection(
    samples: Any,
    *,
    candidate_paths: int,
    family_confidence: float = 0.99,
) -> dict[str, Any]:
    """Return the frozen engineering half-width projection.

    This deliberately reproduces the parent's planning construction.  It is a
    model-based engineering forecast and must not be presented as a
    distribution-free coverage statement.
    """

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("samples must have shape [paths >= 2, features >= 1]")
    if not np.isfinite(values).all():
        raise ValueError("planning samples must be finite")
    if int(candidate_paths) < 2:
        raise ValueError("candidate_paths must be at least two")
    if not (0.0 < float(family_confidence) < 1.0):
        raise ValueError("family_confidence must lie strictly between zero and one")

    from scipy import stats

    path_count, feature_count = values.shape
    family_error = 1.0 - float(family_confidence)
    lower_chi = float(
        stats.chi2.ppf(family_error / float(feature_count), path_count - 1)
    )
    if not lower_chi > 0.0 or not math.isfinite(lower_chi):
        raise ValueError("chi-square variance envelope is unresolved")
    variance = np.var(values, axis=0, ddof=1)
    sd_upper = np.sqrt((path_count - 1) * variance / lower_chi)
    critical = math.sqrt(
        2.0 * math.log(2.0 * float(feature_count) / family_error)
    )
    half_widths = critical * sd_upper / math.sqrt(float(candidate_paths))
    if not np.isfinite(sd_upper).all() or not np.isfinite(half_widths).all():
        raise ValueError("planning projection is nonfinite")
    return {
        "schema": SCHEMA + "-planning-projection",
        "schema_version": SCHEMA_VERSION,
        "planning_version": PLANNING_VERSION,
        "forecast_only": 1,
        "scientific_confidence_interval": 0,
        "pilot_path_count": int(path_count),
        "candidate_path_count": int(candidate_paths),
        "feature_count": int(feature_count),
        "family_confidence": float(family_confidence),
        "chi_square_lower_quantile": lower_chi,
        "sub_gaussian_critical": critical,
        "sd_upper": sd_upper.tolist(),
        "standard_deviation_upper": sd_upper.tolist(),
        "maximum_sd_upper": float(np.max(sd_upper)),
        "predicted_half_widths": half_widths.tolist(),
        "predicted_maximum_half_width": float(np.max(half_widths)),
        "predicted_half_width": float(np.max(half_widths)),
        **NO_WORK,
    }


def _candidate_grid(
    thresholds: DynkinPowerThresholds,
) -> set[tuple[int, int]]:
    return {
        (main, reference)
        for main in thresholds.candidate_main_paths
        for reference in thresholds.candidate_reference_paths
        if reference <= main
    }


def _hours_lookup(
    projected_hours_by_design: Mapping[Any, Any],
    key: tuple[int, int],
) -> float:
    candidates = (key, f"{key[0]}/{key[1]}", f"{key[0]}-{key[1]}")
    for candidate in candidates:
        if candidate in projected_hours_by_design:
            value = projected_hours_by_design[candidate]
            if _finite(value) and float(value) >= 0.0:
                return float(value)
            raise ValueError(f"invalid projected hours for design {key}")
    raise ValueError(f"missing projected hours for design {key}")


def build_dynkin_candidate_records(
    *,
    main_differences: Any,
    reference_differences: Any,
    conservative_rate: float,
    projected_hours_by_design: Mapping[Any, Any],
    panel_role: str,
    thresholds: DynkinPowerThresholds | None = None,
) -> list[dict[str, Any]]:
    """Build the complete frozen candidate grid for one sealed panel."""

    t = thresholds or DynkinPowerThresholds()
    if panel_role not in {"a", "b", "combined"}:
        raise ValueError("panel_role must be a, b, or combined")
    if not _finite(conservative_rate) or float(conservative_rate) <= 0.0:
        raise ValueError("conservative_rate must be finite and positive")
    main = np.asarray(main_differences, dtype=np.float64)
    reference = np.asarray(reference_differences, dtype=np.float64)
    if main.ndim != 2 or main.shape[1] != t.main_feature_count:
        raise ValueError(
            f"main differences must have {t.main_feature_count} features"
        )
    if reference.ndim != 2 or reference.shape[1] != t.reference_feature_count:
        raise ValueError(
            f"reference differences must have {t.reference_feature_count} features"
        )

    rows: list[dict[str, Any]] = []
    for main_paths, reference_paths in sorted(_candidate_grid(t)):
        main_projection = normal_chi_square_bonferroni_projection(
            main,
            candidate_paths=main_paths,
            family_confidence=t.planning_confidence,
        )
        reference_projection = normal_chi_square_bonferroni_projection(
            reference,
            candidate_paths=reference_paths,
            family_confidence=t.planning_confidence,
        )
        rows.append(
            {
                "panel_role": panel_role,
                "main_paths": main_paths,
                "reference_paths": reference_paths,
                "predicted_main_half_width": main_projection[
                    "predicted_maximum_half_width"
                ],
                "predicted_reference_half_width": reference_projection[
                    "predicted_maximum_half_width"
                ],
                "projected_hours": _hours_lookup(
                    projected_hours_by_design, (main_paths, reference_paths)
                ),
                "variance_upper_confidence": t.planning_confidence,
                "variance_bound": PLANNING_VERSION,
                "main_variance_family_size": t.main_feature_count,
                "reference_variance_family_size": t.reference_feature_count,
                "conservative_rate": float(conservative_rate),
                "timing_bound": "minimum-complete-panel-rate-including-io",
                "panel_complete_pass": 1,
                "panel_finite_pass": 1,
                "panel_certification_pass": 1,
                "panel_numerical_health_pass": 1,
                "mass_conservation_pass": 1,
                "shard_chain_pass": 1,
                "pilot_production_isolation_pass": 1,
                "pilot_means_excluded_pass": 1,
                "forecast_only": 1,
                "scientific_confidence_interval": 0,
                **NO_WORK,
            }
        )
    return rows


def _candidate_eligible(
    row: Mapping[str, Any], thresholds: DynkinPowerThresholds
) -> bool:
    finite = all(
        _finite(row.get(name)) and float(row[name]) >= 0.0
        for name in (
            "predicted_main_half_width",
            "predicted_reference_half_width",
            "projected_hours",
            "conservative_rate",
        )
    )
    flags = all(
        _one(row.get(name))
        for name in (
            "panel_complete_pass",
            "panel_finite_pass",
            "panel_certification_pass",
            "panel_numerical_health_pass",
            "mass_conservation_pass",
            "shard_chain_pass",
            "pilot_production_isolation_pass",
            "pilot_means_excluded_pass",
            "forecast_only",
        )
    ) and _zero(row.get("scientific_confidence_interval"))
    return bool(
        finite
        and flags
        and float(row["predicted_main_half_width"])
        <= thresholds.maximum_main_half_width
        and float(row["predicted_reference_half_width"])
        <= thresholds.maximum_reference_half_width
        and float(row["projected_hours"]) <= thresholds.maximum_projected_hours
        and float(row["conservative_rate"]) >= thresholds.minimum_rate
    )


def _validated_grid(
    records: Sequence[Mapping[str, Any]],
    *,
    panel_role: str,
    thresholds: DynkinPowerThresholds,
) -> list[dict[str, Any]]:
    expected = _candidate_grid(thresholds)
    observed: set[tuple[int, int]] = set()
    normalized: list[dict[str, Any]] = []
    for raw in records:
        row = dict(raw)
        key = (int(row.get("main_paths", -1)), int(row.get("reference_paths", -1)))
        if key not in expected or key in observed:
            raise ValueError("candidate rows must enumerate each frozen design once")
        if row.get("panel_role") != panel_role:
            raise ValueError(f"candidate row is not from panel {panel_role}")
        if int(row.get("main_variance_family_size", -1)) != (
            thresholds.main_feature_count
        ) or int(row.get("reference_variance_family_size", -1)) != (
            thresholds.reference_feature_count
        ):
            raise ValueError("candidate variance family size changed")
        observed.add(key)
        row["eligible"] = int(_candidate_eligible(row, thresholds))
        normalized.append(row)
    if observed != expected:
        raise ValueError("candidate grid is incomplete")
    return sorted(
        normalized,
        key=lambda row: (int(row["main_paths"]), int(row["reference_paths"])),
    )


def select_dynkin_panel_a_design(
    records: Sequence[Mapping[str, Any]],
    thresholds: DynkinPowerThresholds | None = None,
) -> dict[str, Any]:
    """Nominate the cheapest eligible design using discovery panel A only."""

    t = thresholds or DynkinPowerThresholds()
    rows = _validated_grid(records, panel_role="a", thresholds=t)
    eligible = [row for row in rows if _one(row["eligible"])]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["projected_hours"]),
            int(row["main_paths"]) + int(row["reference_paths"]),
            int(row["main_paths"]),
            int(row["reference_paths"]),
        ),
        default=None,
    )
    return {
        "schema": SCHEMA + "-panel-a-nomination",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "selection_status": (
            "panel_a_nominated"
            if selected is not None
            else "panel_a_no_eligible_design"
        ),
        "passed": int(selected is not None),
        "panel_a_only_nomination": 1,
        "candidate_count": len(rows),
        "eligible_candidate_count": len(eligible),
        "candidates": rows,
        "selected": None if selected is None else dict(selected),
        "ranking": [
            "projected_hours",
            "total_paths",
            "main_paths",
            "reference_paths",
        ],
        "thresholds": {
            "maximum_main_half_width": t.maximum_main_half_width,
            "maximum_reference_half_width": t.maximum_reference_half_width,
            "maximum_projected_hours": t.maximum_projected_hours,
            "minimum_rate": t.minimum_rate,
        },
        **NO_WORK,
    }


def confirm_dynkin_design(
    panel_a_selection: Mapping[str, Any],
    panel_b_records: Sequence[Mapping[str, Any]] | None,
    combined_records: Sequence[Mapping[str, Any]] | None,
    thresholds: DynkinPowerThresholds | None = None,
) -> dict[str, Any]:
    """Confirm panel A's nominee without allowing B or combined to nominate."""

    t = thresholds or DynkinPowerThresholds()
    selected_raw = panel_a_selection.get("selected")
    if not _passed(panel_a_selection) or not isinstance(selected_raw, Mapping):
        return {
            "schema": SCHEMA + "-sealed-design-confirmation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "selection_status": "panel_a_no_eligible_design",
            "passed": 0,
            "panel_a_nomination_pass": 0,
            "panel_b_confirmation_pass": 0,
            "combined_confirmation_pass": 0,
            "selected": None,
            **NO_WORK,
        }
    nominee = (
        int(selected_raw.get("main_paths", -1)),
        int(selected_raw.get("reference_paths", -1)),
    )
    if nominee not in _candidate_grid(t):
        raise ValueError("panel A nominee is outside the frozen design grid")
    if panel_b_records is None or combined_records is None:
        return {
            "schema": SCHEMA + "-sealed-design-confirmation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "not_evaluated",
            "selection_status": "panel_a_selected_pending_panel_b",
            "passed": 0,
            "panel_a_nomination_pass": 1,
            "panel_b_confirmation_pass": 0,
            "combined_confirmation_pass": 0,
            "selected": dict(selected_raw),
            **NO_WORK,
        }

    panel_b = _validated_grid(panel_b_records, panel_role="b", thresholds=t)
    combined = _validated_grid(
        combined_records, panel_role="combined", thresholds=t
    )
    b_row = next(
        row
        for row in panel_b
        if (int(row["main_paths"]), int(row["reference_paths"])) == nominee
    )
    combined_row = next(
        row
        for row in combined
        if (int(row["main_paths"]), int(row["reference_paths"])) == nominee
    )
    b_pass = _one(b_row["eligible"])
    combined_pass = _one(combined_row["eligible"])
    if not b_pass:
        selection_status = "panel_b_rejected"
    elif not combined_pass:
        selection_status = "combined_rejected"
    else:
        selection_status = "selected"
    return {
        "schema": SCHEMA + "-sealed-design-confirmation",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "selection_status": selection_status,
        "passed": int(b_pass and combined_pass),
        "panel_a_nomination_pass": 1,
        "panel_b_confirmation_pass": int(b_pass),
        "combined_confirmation_pass": int(combined_pass),
        "panel_a_selected": dict(selected_raw),
        "panel_b_evaluated_candidate": b_row,
        "combined_evaluated_candidate": combined_row,
        "selected": dict(selected_raw) if b_pass and combined_pass else None,
        "panel_b_nomination_performed": 0,
        "combined_nomination_performed": 0,
        **NO_WORK,
    }


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
)


def evaluate_dynkin_preflight(
    metrics: Mapping[str, Any],
    thresholds: DynkinPowerThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or DynkinPowerThresholds()
    checks = {
        name: _eq_one(metrics, name)
        for name in (
            "production_authorizing_pass",
            "control_provenance_pass",
            "parent_power_adjudication_pass",
            "fifteen_parent_sources_immutable_pass",
            "parent_preflight_pass",
            "parent_power_numerically_valid_pass",
            "parent_no_work_pass",
            "path_id_plan_pass",
            "legacy_k512_id_plan_pass",
            "legacy_k512_replay_pass",
            "observer_state_hash_invariance_pass",
            "phase_moment_formula_pass",
            "phase_moment_all_colors_pass",
            "phase_moment_half_full_duration_pass",
            "phase_moment_facet_interior_pass",
            "phase_moment_zero_mass_duration_pass",
            "spectral_arb_agreement_pass",
            "cuda_enclosure_pass",
            "adversarial_p2_root_enclosure_pass",
            "cumulative_error_pass",
            "tower_panel_a_pass",
            "tower_panel_b_pass",
            "tower_joint_max_t_pass",
            "tower_panels_frozen_pass",
            "tower_panels_disjoint_pass",
            "negative_orientation_fixture_pass",
            "negative_eigenvalue_fixture_pass",
            "negative_pair_mass_fixture_pass",
            "negative_duration_fixture_pass",
            "negative_post_state_fixture_pass",
            "distribution_free_power_record_pass",
            "right_endpoint_coupling_unchanged_pass",
        )
    }
    checks.update(
        {
            "parent_record_count": _equal(
                metrics, "parent_record_count", t.parent_record_count
            ),
            "parent_source_count": _equal(
                metrics, "parent_source_count", t.parent_source_count
            ),
            "grid_size": _equal(metrics, "grid_size", t.grid_size),
            "alpha": _equal(metrics, "alpha", t.alpha),
            "tau_eff": _equal(metrics, "tau_eff", t.tau_eff),
            "levels": _sequence_equal(metrics, "levels", t.levels),
            "tower_panel_count": _equal(
                metrics, "tower_panel_count", t.tower_panel_count
            ),
            "tower_clusters_per_panel": _equal(
                metrics,
                "tower_clusters_per_panel",
                t.tower_clusters_per_panel,
            ),
            "tower_confidence": _equal(
                metrics, "tower_confidence", t.tower_confidence
            ),
            "maximum_float64_phase_moment_error": _le(
                metrics,
                "maximum_float64_phase_moment_error",
                t.maximum_float64_phase_moment_error,
            ),
            "maximum_cuda_phase_moment_error": _le(
                metrics,
                "maximum_cuda_phase_moment_error",
                t.maximum_cuda_phase_moment_error,
            ),
            "maximum_cumulative_standardized_error": _le(
                metrics,
                "maximum_cumulative_standardized_error",
                t.maximum_cumulative_standardized_error,
            ),
            "certificate_fraction": _equal(
                metrics, "certificate_fraction", 1.0
            ),
            "fallback_fraction": _le(
                metrics, "fallback_fraction", t.maximum_fallback_fraction
            ),
            "fallback_cost_fraction": _le(
                metrics,
                "fallback_cost_fraction",
                t.maximum_fallback_cost_fraction,
            ),
            "peak_memory_fraction": _le(
                metrics, "peak_memory_fraction", t.maximum_peak_memory_fraction
            ),
            "mass_error": _le(
                metrics, "mass_error", t.maximum_cuda_mass_error
            ),
            **{name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS},
        }
    )
    result = _gate(
        "jacobi_rb_dynkin_preflight",
        "immutable parent, exact phase moments, and tower controls",
        checks,
    )
    provenance_names = {
        "control_provenance_pass",
        "fifteen_parent_sources_immutable_pass",
        "parent_preflight_pass",
        "parent_power_numerically_valid_pass",
        "parent_no_work_pass",
        "parent_record_count",
        "parent_source_count",
    }
    adjudication_names = {"parent_power_adjudication_pass"}
    scheduler_names = {
        "path_id_plan_pass",
        "legacy_k512_id_plan_pass",
        "right_endpoint_coupling_unchanged_pass",
    }
    algebra_names = {
        "phase_moment_formula_pass",
        "phase_moment_all_colors_pass",
        "phase_moment_half_full_duration_pass",
        "phase_moment_facet_interior_pass",
        "phase_moment_zero_mass_duration_pass",
        "adversarial_p2_root_enclosure_pass",
        "negative_orientation_fixture_pass",
        "negative_eigenvalue_fixture_pass",
        "negative_pair_mass_fixture_pass",
        "negative_duration_fixture_pass",
        "negative_post_state_fixture_pass",
    }
    tower_names = {
        "tower_panel_a_pass",
        "tower_panel_b_pass",
        "tower_joint_max_t_pass",
        "tower_panels_frozen_pass",
        "tower_panels_disjoint_pass",
        "tower_panel_count",
        "tower_clusters_per_panel",
        "tower_confidence",
    }
    numerical_names = {
        "legacy_k512_replay_pass",
        "observer_state_hash_invariance_pass",
        "spectral_arb_agreement_pass",
        "cuda_enclosure_pass",
        "cumulative_error_pass",
        "maximum_float64_phase_moment_error",
        "maximum_cuda_phase_moment_error",
        "maximum_cumulative_standardized_error",
        "certificate_fraction",
        "mass_error",
        *_FORBIDDEN_COUNTS,
    }
    resource_names = {
        "fallback_fraction",
        "fallback_cost_fraction",
        "peak_memory_fraction",
    }
    result["provenance_valid"] = int(
        all(checks[name]["passed"] for name in provenance_names)
    )
    result["parent_adjudication_valid"] = int(
        all(checks[name]["passed"] for name in adjudication_names)
    )
    result["scheduler_valid"] = int(
        all(checks[name]["passed"] for name in scheduler_names)
    )
    result["phase_moment_algebra_valid"] = int(
        all(checks[name]["passed"] for name in algebra_names)
    )
    result["tower_identity_valid"] = int(
        all(checks[name]["passed"] for name in tower_names)
    )
    result["numerically_valid"] = int(
        all(checks[name]["passed"] for name in numerical_names)
    )
    result["resource_valid"] = int(
        all(checks[name]["passed"] for name in resource_names)
    )
    result["thresholds"] = t.to_dict()
    return result


def evaluate_dynkin_power(
    metrics: Mapping[str, Any],
    thresholds: DynkinPowerThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or DynkinPowerThresholds()
    checks = {
        name: _eq_one(metrics, name)
        for name in (
            "production_authorizing_pass",
            "panel_a_frozen_pass",
            "panel_b_frozen_pass",
            "panel_plan_hash_pass",
            "panel_disjoint_pass",
            "panel_nonregeneration_pass",
            "pilot_production_disjoint_pass",
            "right_endpoint_coupling_unchanged_pass",
            "raw_observables_advisory_only_pass",
            "dynkin_authorizing_estimator_pass",
            "forecast_label_pass",
            "panel_a_complete_pass",
            "panel_b_complete_pass",
            "combined_complete_pass",
            "panel_a_nomination_pass",
            "panel_b_confirmation_pass",
            "combined_confirmation_pass",
            "selected_design_frozen_pass",
            "selected_design_hash_pass",
            "complete_candidate_grid_pass",
            "shard_chain_pass",
            "mass_conservation_pass",
            "state_updates_device_resident_pass",
            "pilot_certification_pass",
            "executed_panels_numerically_valid_pass",
            "candidate_resource_feasibility_pass",
        )
    }
    checks.update(
        {
            "panel_count": _equal(
                metrics, "panel_count", t.pilot_panel_count
            ),
            "paths_per_panel": _equal(
                metrics, "paths_per_panel", t.pilot_paths_per_panel
            ),
            "levels": _sequence_equal(metrics, "levels", t.levels),
            "candidate_main_paths": _sequence_equal(
                metrics, "candidate_main_paths", t.candidate_main_paths
            ),
            "candidate_reference_paths": _sequence_equal(
                metrics,
                "candidate_reference_paths",
                t.candidate_reference_paths,
            ),
            "selected_main_paths": _check(
                metrics.get("selected_main_paths"),
                "in",
                list(t.candidate_main_paths),
                metrics.get("selected_main_paths") in t.candidate_main_paths,
            ),
            "selected_reference_paths": _check(
                metrics.get("selected_reference_paths"),
                "in",
                list(t.candidate_reference_paths),
                metrics.get("selected_reference_paths")
                in t.candidate_reference_paths,
            ),
            **{
                f"{role}_{family}_half_width": _le(
                    metrics,
                    f"{role}_{family}_half_width",
                    (
                        t.maximum_main_half_width
                        if family == "main"
                        else t.maximum_reference_half_width
                    ),
                )
                for role in ("panel_a", "panel_b", "combined")
                for family in ("main", "reference")
            },
            "projected_production_hours": _le(
                metrics,
                "projected_production_hours",
                t.maximum_projected_hours,
            ),
            "resource_feasible_candidate_count": _ge(
                metrics, "resource_feasible_candidate_count", 1.0
            ),
            "minimum_rate": _ge(metrics, "minimum_rate", t.minimum_rate),
            "certificate_fraction": _equal(
                metrics, "certificate_fraction", 1.0
            ),
            "fallback_fraction": _le(
                metrics, "fallback_fraction", t.maximum_fallback_fraction
            ),
            "fallback_cost_fraction": _le(
                metrics,
                "fallback_cost_fraction",
                t.maximum_fallback_cost_fraction,
            ),
            "peak_memory_fraction": _le(
                metrics, "peak_memory_fraction", t.maximum_peak_memory_fraction
            ),
            "mass_error": _le(
                metrics, "mass_error", t.maximum_cuda_mass_error
            ),
            "maximum_cumulative_standardized_error": _le(
                metrics,
                "maximum_cumulative_standardized_error",
                t.maximum_cumulative_standardized_error,
            ),
            **{name: _eq_zero(metrics, name) for name in _FORBIDDEN_COUNTS},
        }
    )
    result = _gate(
        "jacobi_rb_dynkin_power",
        "sealed A/B engineering power forecast for a later refinement run",
        checks,
    )
    numerical_names = {
        "executed_panels_numerically_valid_pass",
        "pilot_certification_pass",
        "shard_chain_pass",
        "mass_conservation_pass",
        "state_updates_device_resident_pass",
        "certificate_fraction",
        "mass_error",
        "maximum_cumulative_standardized_error",
        *_FORBIDDEN_COUNTS,
    }
    resource_names = {
        "candidate_resource_feasibility_pass",
        "resource_feasible_candidate_count",
        "minimum_rate",
        "fallback_fraction",
        "fallback_cost_fraction",
        "peak_memory_fraction",
    }
    power_names = {
        "panel_a_nomination_pass",
        "panel_b_confirmation_pass",
        "combined_confirmation_pass",
        "selected_design_frozen_pass",
        "selected_design_hash_pass",
        "complete_candidate_grid_pass",
        "selected_main_paths",
        "selected_reference_paths",
        *{
            f"{role}_{family}_half_width"
            for role in ("panel_a", "panel_b", "combined")
            for family in ("main", "reference")
        },
    }
    result["numerically_valid"] = int(
        all(checks[name]["passed"] for name in numerical_names)
    )
    result["resource_valid"] = int(
        all(checks[name]["passed"] for name in resource_names)
    )
    result["panel_a_nominated"] = int(
        checks["panel_a_nomination_pass"]["passed"]
    )
    result["panels_agree"] = int(
        checks["panel_b_confirmation_pass"]["passed"]
        and checks["combined_confirmation_pass"]["passed"]
    )
    result["power_valid"] = int(
        all(checks[name]["passed"] for name in power_names)
    )
    result["thresholds"] = t.to_dict()
    return result


def decide_dynkin_power_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if _explicit_provenance_invalid(provenance):
        decision = DynkinPowerDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 1,308-artifact parent binding"
    elif _status(preflight_gate) == "execution_failed":
        decision, action = _execution_failure_decision(preflight_gate)
    elif not _passed(provenance):
        decision = DynkinPowerDecision.DYNKIN_ESTIMATOR_NUMERICALLY_UNRESOLVED
        action = "obtain verified provenance evidence before adjudication"
    elif _status(preflight_gate) != "evaluated":
        decision = DynkinPowerDecision.JACOBI_PHASE_MOMENT_ALGEBRA_INVALID
        action = "complete the exact Dynkin preflight"
    elif not _passed(preflight_gate):
        if _zero(preflight_gate.get("provenance_valid")):
            decision = DynkinPowerDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable parent/source/config binding"
        elif _zero(preflight_gate.get("parent_adjudication_valid")):
            decision = DynkinPowerDecision.PARENT_POWER_ADJUDICATION_INVALID
            action = "repair the raw-endpoint parent re-adjudication"
        elif _zero(preflight_gate.get("scheduler_valid")):
            decision = DynkinPowerDecision.REFINEMENT_SCHEDULER_INVALID
            action = "repair the frozen path-ID plan or refinement scheduler"
        elif _zero(preflight_gate.get("phase_moment_algebra_valid")):
            decision = DynkinPowerDecision.JACOBI_PHASE_MOMENT_ALGEBRA_INVALID
            action = "repair the exact phasewise conditional moments"
        elif _zero(preflight_gate.get("tower_identity_valid")):
            decision = DynkinPowerDecision.DYNKIN_TOWER_IDENTITY_INVALID
            action = "repair the phasewise tower-identity controls"
        elif _zero(preflight_gate.get("numerically_valid")):
            decision = DynkinPowerDecision.DYNKIN_ESTIMATOR_NUMERICALLY_UNRESOLVED
            action = "repair certified Dynkin observation or accumulation"
        elif _zero(preflight_gate.get("resource_valid")):
            decision = DynkinPowerDecision.DYNKIN_COMPUTATIONALLY_INFEASIBLE
            action = "repair exact observer execution without changing the law"
        else:
            decision = DynkinPowerDecision.DYNKIN_ESTIMATOR_NUMERICALLY_UNRESOLVED
            action = "repair incomplete or legacy stage-failure evidence"
    elif _status(pilot_gate) == "execution_failed":
        decision, action = _execution_failure_decision(pilot_gate)
    elif _status(pilot_gate) != "evaluated":
        decision = DynkinPowerDecision.DYNKIN_POWER_INFEASIBLE
        action = "run the sealed Dynkin A/B power pilot"
    elif not _passed(pilot_gate):
        if _zero(pilot_gate.get("numerically_valid")):
            decision = DynkinPowerDecision.DYNKIN_ESTIMATOR_NUMERICALLY_UNRESOLVED
            action = "repair exact pilot execution or shard evidence"
        elif _zero(pilot_gate.get("resource_valid")):
            decision = DynkinPowerDecision.DYNKIN_COMPUTATIONALLY_INFEASIBLE
            action = "repair observer scheduling within the frozen 48-hour budget"
        elif _zero(pilot_gate.get("panel_a_nominated")):
            decision = DynkinPowerDecision.DYNKIN_POWER_INFEASIBLE
            action = "retain evidence and add exact-marginal hierarchical coupling"
        elif _zero(pilot_gate.get("panels_agree")):
            decision = DynkinPowerDecision.DYNKIN_PANELS_DISAGREE
            action = "retain both sealed panels and do not select a design"
        else:
            decision = DynkinPowerDecision.DYNKIN_ESTIMATOR_NUMERICALLY_UNRESOLVED
            action = "repair incomplete or legacy stage-failure evidence"
    else:
        decision = DynkinPowerDecision.EXACT_DYNKIN_REFINEMENT_ESTIMATOR_FEASIBLE
        action = "plan a fresh production refinement using the frozen Dynkin design"

    success = (
        decision
        is DynkinPowerDecision.EXACT_DYNKIN_REFINEMENT_ESTIMATOR_FEASIBLE
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "decision": decision.value,
        "recommended_next_action": action,
        "production_refinement_patch_authorized": int(success),
        "one_image_phase_conditioned_training_patch_authorized": 0,
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "closed_terminal_scientific_outcome": 0,
        **NO_WORK,
    }


__all__ = [
    "DynkinPowerDecision",
    "DynkinPowerThresholds",
    "PLANNING_VERSION",
    "build_dynkin_candidate_records",
    "confirm_dynkin_design",
    "decide_dynkin_power_workflow",
    "evaluate_dynkin_power",
    "evaluate_dynkin_preflight",
    "normal_chi_square_bonferroni_projection",
    "not_evaluated_gate",
    "select_dynkin_panel_a_design",
]
