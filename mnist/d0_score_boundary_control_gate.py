"""Pure gates for the boundary-admissible D0 implicit-score controls.

This module intentionally contains no training or filesystem code.  It keeps
the scientific thresholds, checkpoint eligibility, and terminal decisions
recomputable from the JSON evidence written by
``mnist.diag_d0_score_boundary_controls``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


__all__ = [
    "BoundaryControlDecision",
    "BoundaryControlThresholds",
    "checkpoint_is_dual_bank_eligible",
    "select_dual_bank_checkpoint",
    "evaluate_boundary_preflight",
    "evaluate_supervised_teacher",
    "evaluate_implicit_teacher_seed",
    "evaluate_implicit_teacher_study",
    "evaluate_null_seed",
    "evaluate_null_study",
    "evaluate_boundary_control_gate",
    "decide_control_repair",
    "evaluate_boundary_control_gates",
]


@dataclass(frozen=True)
class BoundaryControlThresholds:
    """Frozen production thresholds for the controls-only repair gate."""

    teacher_min_score_gain: float = 0.90
    teacher_min_overall_flux_cosine: float = 0.98
    teacher_min_bin_flux_cosine: float = 0.95
    teacher_max_overall_relative_flux_l2: float = 0.15
    teacher_max_bin_relative_flux_l2: float = 0.20
    expected_time_bins: int = 5
    expected_implicit_teacher_seeds: int = 3
    minimum_passing_implicit_teacher_seeds: int = 2
    expected_null_seeds: int = 3
    maximum_post_warmup_clip_fraction: float = 0.10
    boundary_min_flux_slope: float = 0.90
    boundary_max_flux_ratio: float = 1e-3
    bootstrap_confidence: float = 0.90

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        for name in (
            "teacher_min_score_gain",
            "teacher_min_overall_flux_cosine",
            "teacher_min_bin_flux_cosine",
            "teacher_max_overall_relative_flux_l2",
            "teacher_max_bin_relative_flux_l2",
            "maximum_post_warmup_clip_fraction",
            "boundary_max_flux_ratio",
            "bootstrap_confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 < float(self.bootstrap_confidence) < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1)")
        if int(self.expected_time_bins) <= 0:
            raise ValueError("expected_time_bins must be positive")
        if not 1 <= int(self.minimum_passing_implicit_teacher_seeds) <= int(
            self.expected_implicit_teacher_seeds
        ):
            raise ValueError("invalid passing implicit-teacher seed count")


class BoundaryControlDecision(str, Enum):
    CONTROL_PIPELINE_REPAIRED = "control_pipeline_repaired"
    BOUNDARY_DOMAIN_INVALID = "boundary_domain_invalid"
    REPRESENTATION_INVALID = "representation_invalid"
    TRACE_ESTIMATOR_INCONCLUSIVE = "trace_estimator_inconclusive"
    IMPLICIT_OBJECTIVE_UNSTABLE = "implicit_objective_unstable"
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"


def _passed(value: bool | int | Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        return bool(int(value.get("passed", 0)))
    return bool(value)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check(name: str, value: Any, operator: str, threshold: Any, passed: bool) -> tuple[str, dict[str, Any]]:
    return name, {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _finish(name: str, checks: Sequence[tuple[str, Mapping[str, Any]]], claim: str) -> dict[str, Any]:
    subchecks = {key: dict(value) for key, value in checks}
    return {
        "gate": name,
        "passed": int(all(bool(int(value["passed"])) for value in subchecks.values())),
        "subchecks": subchecks,
        "claim_scope": claim,
        "sampling_performed": 0,
    }


def _scope_lcb(bank: Mapping[str, Any], scope: str) -> float | None:
    value = bank.get(scope, {})
    if not isinstance(value, Mapping):
        return None
    for key in ("lower_bound", "improvement_lower_bound", "objective_improvement_lower_bound"):
        if key in value and _finite(value[key]):
            return float(value[key])
    return None


def checkpoint_is_dual_bank_eligible(record: Mapping[str, Any]) -> bool:
    """Return eligibility under the frozen dual-bank selection rule.

    Step zero is always a legal checkpoint.  Every nonzero EMA checkpoint must
    have a strictly positive whole-path 90% lower confidence bound in both
    selection banks, both overall and in the data-end stratum.
    """

    step = int(record.get("step", -1))
    if step == 0:
        return True
    if step < 0 or not bool(int(record.get("finite", 1))):
        return False
    banks = record.get("banks", record.get("selection_banks", {}))
    if not isinstance(banks, Mapping) or set(banks) != {"a", "b"}:
        return False
    return all(
        (value := _scope_lcb(dict(banks[name]), scope)) is not None and value > 0.0
        for name in ("a", "b")
        for scope in ("overall", "data_end")
    )


def _mean_selection_risk(record: Mapping[str, Any]) -> float:
    risks: list[float] = []
    banks = record.get("banks", record.get("selection_banks", {}))
    if isinstance(banks, Mapping):
        for value in banks.values():
            if not isinstance(value, Mapping):
                continue
            overall = value.get("overall", {})
            if isinstance(overall, Mapping):
                for key in ("model_score_risk", "model_risk", "risk"):
                    if key in overall and _finite(overall[key]):
                        risks.append(float(overall[key]))
                        break
    if not risks and _finite(record.get("mean_selection_risk")):
        risks.append(float(record["mean_selection_risk"]))
    if not risks:
        return math.inf
    return sum(risks) / len(risks)


def select_dual_bank_checkpoint(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the lowest-risk eligible EMA checkpoint, earliest on ties."""

    normalized: list[dict[str, Any]] = []
    for value in records:
        row = dict(value)
        row["selection_eligible"] = int(checkpoint_is_dual_bank_eligible(row))
        row["mean_selection_risk"] = _mean_selection_risk(row)
        normalized.append(row)
    if not any(int(row.get("step", -1)) == 0 for row in normalized):
        raise ValueError("checkpoint records must include step zero")
    eligible = [row for row in normalized if int(row["selection_eligible"]) == 1]
    selected = min(
        eligible,
        key=lambda row: (float(row["mean_selection_risk"]), int(row["step"])),
    )
    return {
        "selected_step": int(selected["step"]),
        "selected_mean_selection_risk": (
            None if not math.isfinite(float(selected["mean_selection_risk"]))
            else float(selected["mean_selection_risk"])
        ),
        "selected_record": selected,
        "records": normalized,
        "comparator": "analytic_zero_step0",
    }


def evaluate_boundary_preflight(
    metrics: Mapping[str, Any],
    thresholds: BoundaryControlThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or BoundaryControlThresholds()
    finite_names = (
        "potential_finite",
        "gradient_finite",
        "hvp_finite",
        "generator_finite",
        "energy_finite",
    )
    checks = [
        _check(name, int(bool(metrics.get(name, 0))), "==", 1, bool(metrics.get(name, 0)))
        for name in finite_names
    ]
    slope = metrics.get("incident_flux_loglog_slope")
    ratio = metrics.get("incident_flux_endpoint_ratio")
    legacy_coefficient_error = metrics.get("legacy_coefficient_relative_error")
    checks.extend(
        [
            _check(
                "incident_flux_slope", slope, ">=", thresholds.boundary_min_flux_slope,
                _finite(slope) and float(slope) >= thresholds.boundary_min_flux_slope,
            ),
            _check(
                "incident_flux_endpoint_ratio", ratio, "<=", thresholds.boundary_max_flux_ratio,
                _finite(ratio) and 0.0 <= float(ratio) <= thresholds.boundary_max_flux_ratio,
            ),
            _check(
                "legacy_barrier_rejected", int(bool(metrics.get("legacy_barrier_rejected", 0))),
                "==", 1, bool(metrics.get("legacy_barrier_rejected", 0)),
            ),
            _check(
                "legacy_coefficient", legacy_coefficient_error, "<=", 0.02,
                _finite(legacy_coefficient_error) and float(legacy_coefficient_error) <= 0.02,
            ),
            _check(
                "operator_pass", int(bool(metrics.get("operator_pass", 0))), "==", 1,
                bool(metrics.get("operator_pass", 0)),
            ),
            _check(
                "orthogonal_probe_trace", int(bool(metrics.get("orthogonal_probe_pass", metrics.get("operator_pass", 0)))),
                "==", 1, bool(metrics.get("orthogonal_probe_pass", metrics.get("operator_pass", 0))),
            ),
            _check(
                "aggregate_preflight", int(bool(metrics.get("aggregate_preflight_pass", metrics.get("operator_pass", 0)))),
                "==", 1, bool(metrics.get("aggregate_preflight_pass", metrics.get("operator_pass", 0))),
            ),
            _check(
                "production_workload_smoke",
                int(bool(metrics.get("production_workload_smoke_pass", metrics.get("operator_pass", 0)))),
                "==",
                1,
                bool(metrics.get("production_workload_smoke_pass", metrics.get("operator_pass", 0))),
            ),
        ]
    )
    return _finish("boundary_preflight", checks, "closed-simplex model/operator domain")


def _teacher_checks(
    metrics: Mapping[str, Any], thresholds: BoundaryControlThresholds, *, include_objective_banks: bool
) -> list[tuple[str, Mapping[str, Any]]]:
    gains = (
        metrics.get("audit_overall_score_gain"),
        metrics.get("audit_data_end_score_gain"),
    )
    cosines = list(metrics.get("time_bin_flux_cosines", []))
    relatives = list(metrics.get("time_bin_relative_flux_l2", []))
    checks = [
        _check("complete", int(bool(metrics.get("complete", 0))), "==", 1, bool(metrics.get("complete", 0))),
        _check("finite", int(bool(metrics.get("finite", 0))), "==", 1, bool(metrics.get("finite", 0))),
        _check("selected_nonzero", metrics.get("selected_step"), ">", 0, int(metrics.get("selected_step", 0)) > 0),
        _check("overall_score_gain", gains[0], ">=", thresholds.teacher_min_score_gain, _finite(gains[0]) and float(gains[0]) >= thresholds.teacher_min_score_gain),
        _check("data_end_score_gain", gains[1], ">=", thresholds.teacher_min_score_gain, _finite(gains[1]) and float(gains[1]) >= thresholds.teacher_min_score_gain),
        _check("overall_flux_cosine", metrics.get("overall_flux_cosine"), ">=", thresholds.teacher_min_overall_flux_cosine, _finite(metrics.get("overall_flux_cosine")) and float(metrics["overall_flux_cosine"]) >= thresholds.teacher_min_overall_flux_cosine),
        _check("all_bin_flux_cosines", cosines, ">= each", thresholds.teacher_min_bin_flux_cosine, len(cosines) == thresholds.expected_time_bins and all(_finite(v) and float(v) >= thresholds.teacher_min_bin_flux_cosine for v in cosines)),
        _check("overall_relative_flux_l2", metrics.get("overall_relative_flux_l2"), "<=", thresholds.teacher_max_overall_relative_flux_l2, _finite(metrics.get("overall_relative_flux_l2")) and 0.0 <= float(metrics["overall_relative_flux_l2"]) <= thresholds.teacher_max_overall_relative_flux_l2),
        _check("all_bin_relative_flux_l2", relatives, "<= each", thresholds.teacher_max_bin_relative_flux_l2, len(relatives) == thresholds.expected_time_bins and all(_finite(v) and 0.0 <= float(v) <= thresholds.teacher_max_bin_relative_flux_l2 for v in relatives)),
        _check("boundary_admissible", int(bool(metrics.get("boundary_admissible", 0))), "==", 1, bool(metrics.get("boundary_admissible", 0))),
        _check("post_warmup_clip_fraction", metrics.get("post_warmup_clip_fraction"), "<=", thresholds.maximum_post_warmup_clip_fraction, _finite(metrics.get("post_warmup_clip_fraction")) and 0.0 <= float(metrics["post_warmup_clip_fraction"]) <= thresholds.maximum_post_warmup_clip_fraction),
    ]
    if include_objective_banks:
        banks = metrics.get("audit_objective_banks", {})
        for name in ("a", "b"):
            bank = dict(banks.get(name, {})) if isinstance(banks, Mapping) else {}
            values = [_scope_lcb(bank, scope) for scope in ("overall", "data_end")]
            checks.append(
                _check(
                    f"audit_objective_bank_{name}", values, "> 0 each", 0.0,
                    all(value is not None and value > 0.0 for value in values),
                )
            )
    return checks


def evaluate_supervised_teacher(
    metrics: Mapping[str, Any], thresholds: BoundaryControlThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or BoundaryControlThresholds()
    return _finish(
        "supervised_teacher",
        _teacher_checks(metrics, thresholds, include_objective_banks=False),
        "representability of the exact bounded analytic score and flux",
    )


def evaluate_implicit_teacher_seed(
    metrics: Mapping[str, Any], thresholds: BoundaryControlThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or BoundaryControlThresholds()
    return _finish(
        "implicit_teacher_seed",
        _teacher_checks(metrics, thresholds, include_objective_banks=True),
        "implicit recovery of the exact bounded teacher",
    )


def evaluate_implicit_teacher_study(
    seed_results: Sequence[Mapping[str, Any]], thresholds: BoundaryControlThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or BoundaryControlThresholds()
    gates = [
        dict(value) if "subchecks" in value else evaluate_implicit_teacher_seed(value, thresholds)
        for value in seed_results
    ]
    pass_count = sum(_passed(value) for value in gates)
    seeds = [int(value.get("model_seed", value.get("seed", -1))) for value in seed_results]
    required_validity_checks = (
        "complete",
        "finite",
        "boundary_admissible",
        "post_warmup_clip_fraction",
    )
    all_tasks_valid = len(gates) == thresholds.expected_implicit_teacher_seeds and all(
        all(
            bool(int(dict(gate.get("subchecks", {})).get(name, {}).get("passed", 0)))
            for name in required_validity_checks
        )
        for gate in gates
    )
    checks = [
        _check("task_count", len(gates), "==", thresholds.expected_implicit_teacher_seeds, len(gates) == thresholds.expected_implicit_teacher_seeds),
        _check("distinct_seeds", len(set(seeds)), "==", thresholds.expected_implicit_teacher_seeds, len(set(seeds)) == thresholds.expected_implicit_teacher_seeds),
        _check(
            "all_tasks_valid",
            int(all_tasks_valid),
            "==",
            1,
            all_tasks_valid,
        ),
        _check("passing_seeds", pass_count, ">=", thresholds.minimum_passing_implicit_teacher_seeds, pass_count >= thresholds.minimum_passing_implicit_teacher_seeds),
    ]
    result = _finish("implicit_teacher_study", checks, "multi-seed bounded-teacher implicit recovery")
    result["seed_gates"] = gates
    result["passing_seed_count"] = int(pass_count)
    return result


def evaluate_null_seed(
    metrics: Mapping[str, Any], thresholds: BoundaryControlThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or BoundaryControlThresholds()
    banks = metrics.get("audit_objective_banks", {})
    checks = [
        _check("complete", int(bool(metrics.get("complete", 0))), "==", 1, bool(metrics.get("complete", 0))),
        _check("finite", int(bool(metrics.get("finite", 0))), "==", 1, bool(metrics.get("finite", 0))),
        _check("selected_step_zero", metrics.get("selected_step"), "==", 0, int(metrics.get("selected_step", -1)) == 0),
        _check("analytic_zero_comparator", metrics.get("comparator"), "==", "analytic_zero", metrics.get("comparator") in {"analytic_zero", "analytic_zero_step0"}),
        _check("boundary_admissible", int(bool(metrics.get("boundary_admissible", 0))), "==", 1, bool(metrics.get("boundary_admissible", 0))),
        _check(
            "post_warmup_clip_fraction", metrics.get("post_warmup_clip_fraction"),
            "<=", thresholds.maximum_post_warmup_clip_fraction,
            _finite(metrics.get("post_warmup_clip_fraction"))
            and 0.0 <= float(metrics["post_warmup_clip_fraction"])
            <= thresholds.maximum_post_warmup_clip_fraction,
        ),
    ]
    for name in ("a", "b"):
        bank = dict(banks.get(name, {})) if isinstance(banks, Mapping) else {}
        values = [_scope_lcb(bank, scope) for scope in ("overall", "data_end")]
        checks.append(
            _check(
                f"no_positive_audit_bank_{name}", values, "<= 0 each", 0.0,
                all(value is not None and value <= 0.0 for value in values),
            )
        )
    return _finish("null_seed", checks, "stationary Dirichlet null versus analytic zero")


def evaluate_null_study(
    seed_results: Sequence[Mapping[str, Any]], thresholds: BoundaryControlThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or BoundaryControlThresholds()
    gates = [
        dict(value) if "subchecks" in value else evaluate_null_seed(value, thresholds)
        for value in seed_results
    ]
    seeds = [int(value.get("model_seed", value.get("seed", -1))) for value in seed_results]
    checks = [
        _check("task_count", len(gates), "==", thresholds.expected_null_seeds, len(gates) == thresholds.expected_null_seeds),
        _check("distinct_seeds", len(set(seeds)), "==", thresholds.expected_null_seeds, len(set(seeds)) == thresholds.expected_null_seeds),
        _check("all_null_seeds_pass", sum(_passed(value) for value in gates), "==", thresholds.expected_null_seeds, len(gates) == thresholds.expected_null_seeds and all(_passed(value) for value in gates)),
    ]
    result = _finish("null_study", checks, "three-seed stationary-null control")
    result["seed_gates"] = gates
    return result


def evaluate_boundary_control_gate(
    *,
    provenance_pass: bool | int | Mapping[str, Any],
    boundary_preflight: bool | int | Mapping[str, Any],
    supervised_teacher: bool | int | Mapping[str, Any],
    implicit_teacher_study: bool | int | Mapping[str, Any],
    null_study: bool | int | Mapping[str, Any],
    probe_banks_agree: bool = True,
) -> dict[str, Any]:
    values = {
        "provenance": provenance_pass,
        "boundary_preflight": boundary_preflight,
        "supervised_teacher": supervised_teacher,
        "implicit_teacher_study": implicit_teacher_study,
        "null_study": null_study,
        "probe_banks_agree": bool(probe_banks_agree),
    }
    checks = [
        _check(name, int(_passed(value)), "==", 1, _passed(value))
        for name, value in values.items()
    ]
    result = _finish("boundary_controls", checks, "boundary-admissible synthetic implicit-score controls")
    result["components"] = {
        name: dict(value) if isinstance(value, Mapping) else int(bool(value))
        for name, value in values.items()
    }
    return result


def decide_control_repair(
    *,
    provenance_pass: bool | int | Mapping[str, Any],
    boundary_preflight: bool | int | Mapping[str, Any],
    supervised_teacher: bool | int | Mapping[str, Any],
    implicit_teacher_study: bool | int | Mapping[str, Any],
    null_study: bool | int | Mapping[str, Any],
    probe_banks_agree: bool = True,
) -> dict[str, Any]:
    if not _passed(provenance_pass):
        decision = BoundaryControlDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the failed-run/cache binding before running controls"
    elif not _passed(boundary_preflight):
        decision = BoundaryControlDecision.BOUNDARY_DOMAIN_INVALID
        action = "repair the model/operator boundary domain before optimization"
    elif not _passed(supervised_teacher):
        decision = BoundaryControlDecision.REPRESENTATION_INVALID
        action = "repair the boundary-smooth representation or supervised optimizer"
    elif not probe_banks_agree and _passed(implicit_teacher_study):
        decision = BoundaryControlDecision.TRACE_ESTIMATOR_INCONCLUSIVE
        action = "increase or redesign trace probes without changing the target law"
    elif not _passed(implicit_teacher_study) or not _passed(null_study):
        decision = BoundaryControlDecision.IMPLICIT_OBJECTIVE_UNSTABLE
        action = "use the predeclared density-ratio-classification fallback or repair the implicit objective"
    else:
        decision = BoundaryControlDecision.CONTROL_PIPELINE_REPAIRED
        action = "plan a separate fresh physical-score run with untouched audit paths"
    return {
        "decision": decision.value,
        "recommended_next_action": action,
        "physical_training_authorized": int(decision is BoundaryControlDecision.CONTROL_PIPELINE_REPAIRED),
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


def evaluate_boundary_control_gates(
    *,
    provenance_pass: bool | int | Mapping[str, Any],
    boundary_preflight: bool | int | Mapping[str, Any],
    supervised_teacher: bool | int | Mapping[str, Any],
    implicit_teacher_study: bool | int | Mapping[str, Any],
    null_study: bool | int | Mapping[str, Any],
    require_gate: str = "none",
    probe_banks_agree: bool = True,
) -> dict[str, Any]:
    required = str(require_gate)
    if required not in {"none", "preflight", "controls"}:
        raise ValueError("require_gate must be none, preflight, or controls")
    preflight_pass = _passed(provenance_pass) and _passed(boundary_preflight)
    control_gate = evaluate_boundary_control_gate(
        provenance_pass=provenance_pass,
        boundary_preflight=boundary_preflight,
        supervised_teacher=supervised_teacher,
        implicit_teacher_study=implicit_teacher_study,
        null_study=null_study,
        probe_banks_agree=probe_banks_agree,
    )
    requirement = {
        "none": True,
        "preflight": preflight_pass,
        "controls": _passed(control_gate),
    }[required]
    return {
        "schema": "experiment12-d0-boundary-control-gate",
        "schema_version": 1,
        "required_gate": required,
        "required_gate_pass": int(requirement),
        "preflight_pass": int(preflight_pass),
        "controls": control_gate,
        "decision": decide_control_repair(
            provenance_pass=provenance_pass,
            boundary_preflight=boundary_preflight,
            supervised_teacher=supervised_teacher,
            implicit_teacher_study=implicit_teacher_study,
            null_study=null_study,
            probe_banks_agree=probe_banks_agree,
        ),
        "sampling_performed": 0,
    }
