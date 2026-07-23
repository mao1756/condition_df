"""Pure gates for streamed D0 implicit-control stability confirmation.

The module deliberately has no Torch or filesystem dependency.  It turns the
JSON evidence written by the stability workflow into reproducible preflight,
pilot, confirmation, and terminal decisions.  The scientific teacher/null
thresholds continue to come from the boundary-control gate; the only new
thresholds concern optimizer stability and the train/selection-only pilot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_score_boundary_control_gate import (
    BoundaryControlThresholds,
    evaluate_implicit_teacher_study,
    evaluate_null_study,
)
from mnist.d0_score_control_scale_repair_gate import ProbeBankStatus


__all__ = [
    "ProbeBankStatus",
    "StabilityDecision",
    "StabilityThresholds",
    "classify_probe_bank_status",
    "not_evaluated_gate",
    "evaluate_stein_identity_preflight",
    "evaluate_pilot_candidate",
    "evaluate_stability_pilot",
    "select_stability_profile",
    "evaluate_stability_confirmation",
    "decide_stability_confirmation",
    "evaluate_stability_workflow",
]


SCHEMA = "experiment12-d0-score-control-stability-confirmation-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StabilityThresholds:
    """Frozen optimizer/pilot thresholds for the stability confirmation."""

    stein_bootstrap_confidence: float = 0.99
    stein_paths: int = 128
    stein_teacher_amplitudes: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    pilot_learning_rates: tuple[float, ...] = (1e-4, 3e-5, 1e-5, 3e-6)
    pilot_steps: int = 1000
    pilot_stability_start_step: int = 101
    pilot_final_window_steps: int = 200
    maximum_clip_fraction: float = 0.10
    confirmation_teacher_seeds: int = 3
    minimum_passing_teacher_seeds: int = 2
    confirmation_null_seeds: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < float(self.stein_bootstrap_confidence) < 1.0:
            raise ValueError("stein_bootstrap_confidence must be in (0, 1)")
        if int(self.stein_paths) <= 0:
            raise ValueError("stein_paths must be positive")
        if not self.stein_teacher_amplitudes or any(
            not _finite(value) for value in self.stein_teacher_amplitudes
        ):
            raise ValueError("stein_teacher_amplitudes must be finite and nonempty")
        if not self.pilot_learning_rates or any(
            not _finite(value) or float(value) <= 0.0
            for value in self.pilot_learning_rates
        ):
            raise ValueError("pilot_learning_rates must be finite and positive")
        if len(set(float(value) for value in self.pilot_learning_rates)) != len(
            self.pilot_learning_rates
        ):
            raise ValueError("pilot_learning_rates must be distinct")
        if not 0.0 <= float(self.maximum_clip_fraction) <= 1.0:
            raise ValueError("maximum_clip_fraction must be in [0, 1]")
        if int(self.pilot_steps) <= 0 or not (
            1 <= int(self.pilot_stability_start_step) <= int(self.pilot_steps)
        ):
            raise ValueError("pilot stability window is invalid")
        if not 1 <= int(self.pilot_final_window_steps) <= int(self.pilot_steps):
            raise ValueError("pilot final window is invalid")
        if not 1 <= int(self.minimum_passing_teacher_seeds) <= int(
            self.confirmation_teacher_seeds
        ):
            raise ValueError("confirmation teacher seed counts are invalid")
        if int(self.confirmation_null_seeds) <= 0:
            raise ValueError("confirmation_null_seeds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StabilityDecision(str, Enum):
    CONTROL_PIPELINE_REPAIRED = "control_pipeline_repaired"
    OPTIMIZER_STABILITY_UNRESOLVED = "optimizer_stability_unresolved"
    OPTIMIZER_STABILITY_INVALID = "optimizer_stability_invalid"
    TRACE_ESTIMATOR_INCONCLUSIVE = "trace_estimator_inconclusive"
    IMPLICIT_OBJECTIVE_UNSTABLE = "implicit_objective_unstable"
    OPERATOR_IDENTITY_INVALID = "operator_identity_invalid"
    STABILITY_PREFLIGHT_INVALID = "stability_preflight_invalid"
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"


def classify_probe_bank_status(
    *, studies_evaluated: bool, banks_agree: bool | None
) -> ProbeBankStatus:
    """Return an explicit tri-state; absent evidence is never agreement."""

    if not studies_evaluated or banks_agree is None:
        return ProbeBankStatus.NOT_EVALUATED
    return ProbeBankStatus.AGREE if banks_agree else ProbeBankStatus.DISAGREE


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    """Build a serializable fail-closed placeholder for a skipped stage."""

    if not str(reason).strip():
        raise ValueError("a not-evaluated gate requires a reason")
    return {
        "gate": str(name),
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "skip_reason": str(reason),
        "subchecks": {},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _passed(value: bool | int | Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        try:
            return bool(int(value.get("passed", value.get("gate_pass", 0))))
        except (TypeError, ValueError):
            return False
    return value is True or (isinstance(value, int) and value == 1)


def _status(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "evaluated"
    raw = str(value.get("evaluation_status", value.get("study_status", "evaluated")))
    aliases = {
        "complete": "evaluated",
        "completed": "evaluated",
        "pending": "not_evaluated",
        "skipped": "not_evaluated",
        "incomplete": "not_evaluated",
    }
    normalized = aliases.get(raw.strip().lower(), raw.strip().lower())
    return normalized if normalized in {"evaluated", "not_evaluated"} else "not_evaluated"


def _probe_status(value: ProbeBankStatus | str) -> ProbeBankStatus:
    if isinstance(value, ProbeBankStatus):
        return value
    try:
        return ProbeBankStatus(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ProbeBankStatus)
        raise ValueError(f"probe_bank_status must be one of {allowed}") from exc


def _check(
    name: str, value: Any, operator: str, threshold: Any, passed: bool
) -> tuple[str, dict[str, Any]]:
    return name, {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _finish(
    name: str,
    checks: Sequence[tuple[str, Mapping[str, Any]]],
    claim_scope: str,
    *,
    evaluation_status: str = "evaluated",
) -> dict[str, Any]:
    subchecks = {key: dict(value) for key, value in checks}
    passed = evaluation_status == "evaluated" and bool(subchecks) and all(
        bool(int(value.get("passed", 0))) for value in subchecks.values()
    )
    return {
        "gate": name,
        "evaluation_status": evaluation_status,
        "passed": int(passed),
        "subchecks": subchecks,
        "claim_scope": claim_scope,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _interval_contains_zero(record: Mapping[str, Any]) -> bool:
    interval = _first(
        record,
        "bootstrap_interval",
        "difference_interval",
        "measured_minus_predicted_interval",
        "measured_minus_predicted",
        "confidence_interval",
        default=None,
    )
    if isinstance(interval, Mapping):
        lower = _first(interval, "lower", "lower_bound", "lo")
        upper = _first(interval, "upper", "upper_bound", "hi")
    elif isinstance(interval, Sequence) and not isinstance(interval, (str, bytes)) and len(interval) == 2:
        lower, upper = interval
    else:
        lower = _first(record, "lower_bound", "bootstrap_lower", "difference_lower")
        upper = _first(record, "upper_bound", "bootstrap_upper", "difference_upper")
    return _finite(lower) and _finite(upper) and float(lower) <= 0.0 <= float(upper)


def evaluate_stein_identity_preflight(
    metrics: Mapping[str, Any], thresholds: StabilityThresholds | None = None
) -> dict[str, Any]:
    """Evaluate exact analytic null and bounded-teacher Stein identities."""

    thresholds = thresholds or StabilityThresholds()
    status = _status(metrics)
    null = metrics.get("null_identity", metrics.get("null", {}))
    teacher = metrics.get("teacher_identities", metrics.get("teacher", []))
    null = dict(null) if isinstance(null, Mapping) else {}
    teacher_rows = [dict(value) for value in teacher if isinstance(value, Mapping)] if isinstance(teacher, Sequence) and not isinstance(teacher, (str, bytes)) else []
    expected_a = tuple(float(value) for value in thresholds.stein_teacher_amplitudes)
    actual_a = tuple(
        float(_first(value, "a", "scale"))
        for value in teacher_rows
        if _finite(_first(value, "a", "scale"))
    )
    recorded_scales = metrics.get("teacher_scales", list(actual_a))
    if isinstance(recorded_scales, Sequence) and not isinstance(recorded_scales, (str, bytes)):
        normalized_recorded_scales = tuple(
            float(value) for value in recorded_scales if _finite(value)
        )
    else:
        normalized_recorded_scales = ()
    all_rows_finite = bool(int(metrics.get("finite", 0))) and bool(
        int(null.get("finite", 1))
    ) and all(bool(int(value.get("finite", 1))) for value in teacher_rows)
    confidence = _first(metrics, "bootstrap_confidence", "confidence")
    paths = _first(metrics, "path_count", "path_count_per_law", "paths")
    complete = _first(metrics, "complete", "passed", default=0)
    checks = [
        _check("complete", int(bool(complete)), "==", 1, bool(complete)),
        _check("finite", int(all_rows_finite), "==", 1, all_rows_finite),
        _check("path_count", paths, "==", thresholds.stein_paths, paths is not None and int(paths) == thresholds.stein_paths),
        _check("bootstrap_confidence", confidence, "==", thresholds.stein_bootstrap_confidence, _finite(confidence) and math.isclose(float(confidence), thresholds.stein_bootstrap_confidence, rel_tol=0.0, abs_tol=1e-15)),
        _check("teacher_amplitudes", list(actual_a), "==", list(expected_a), actual_a == expected_a),
        _check("recorded_teacher_scales", list(normalized_recorded_scales), "==", list(expected_a), normalized_recorded_scales == expected_a),
        _check("null_identity_interval", int(_interval_contains_zero(null)), "contains", 0.0, _interval_contains_zero(null)),
        _check("teacher_identity_intervals", [int(_interval_contains_zero(value)) for value in teacher_rows], "all contain", 0.0, len(teacher_rows) == len(expected_a) and all(_interval_contains_zero(value) for value in teacher_rows)),
        _check("physical_training_performed", int(metrics.get("physical_training_performed", 0)), "==", 0, int(metrics.get("physical_training_performed", 0)) == 0),
        _check("sampling_performed", int(metrics.get("sampling_performed", 0)), "==", 0, int(metrics.get("sampling_performed", 0)) == 0),
    ]
    result = _finish(
        "stein_identity_preflight",
        checks,
        "exact analytic Dirichlet-null and bounded-teacher Stein identities",
        evaluation_status=status,
    )
    result["operator_identity_pass"] = result["passed"]
    result["thresholds"] = thresholds.to_dict()
    return result


def _metrics(task: Any) -> dict[str, Any]:
    if not isinstance(task, Mapping):
        return {}
    value = task.get("metrics", task)
    return dict(value) if isinstance(value, Mapping) else {}


def _scope_lcb(bank: Mapping[str, Any], scope: str) -> float | None:
    value = bank.get(scope, {})
    if not isinstance(value, Mapping):
        return None
    raw = _first(
        value,
        "lower_bound",
        "improvement_lower_bound",
        "objective_improvement_lower_bound",
    )
    return float(raw) if _finite(raw) else None


def _dual_bank_lcbs(metrics: Mapping[str, Any]) -> list[float | None]:
    banks = _first(
        metrics,
        "selection_objective_banks",
        "selection_banks",
        "objective_banks",
        default={},
    )
    banks = dict(banks) if isinstance(banks, Mapping) else {}
    return [
        _scope_lcb(dict(banks.get(name, {})), scope)
        if isinstance(banks.get(name, {}), Mapping)
        else None
        for name in ("a", "b")
        for scope in ("overall", "data_end")
    ]


def _clip_windows(metrics: Mapping[str, Any]) -> tuple[Any, Any]:
    stable = _first(
        metrics,
        "clip_fraction_steps_101_1000",
        "stability_window_clip_fraction",
        "post_pilot_warmup_clip_fraction",
    )
    final = _first(
        metrics,
        "final_200_clip_fraction",
        "final_window_clip_fraction",
    )
    return stable, final


def _risk(metrics: Mapping[str, Any]) -> float | None:
    direct = _first(metrics, "mean_dual_bank_selection_risk", "mean_selection_risk")
    if _finite(direct):
        return float(direct)
    banks = _first(metrics, "selection_objective_banks", "selection_banks", default={})
    risks: list[float] = []
    if isinstance(banks, Mapping):
        for bank in banks.values():
            if not isinstance(bank, Mapping):
                continue
            overall = bank.get("overall", {})
            if isinstance(overall, Mapping):
                raw = _first(overall, "model_score_risk", "model_risk", "risk")
                if _finite(raw):
                    risks.append(float(raw))
    return sum(risks) / len(risks) if risks else None


def evaluate_pilot_candidate(
    candidate: Mapping[str, Any], thresholds: StabilityThresholds | None = None
) -> dict[str, Any]:
    """Evaluate one coupled teacher/null learning-rate pilot candidate."""

    thresholds = thresholds or StabilityThresholds()
    teacher = _metrics(candidate.get("teacher", candidate.get("teacher_metrics", {})))
    null = _metrics(candidate.get("null", candidate.get("null_metrics", {})))
    teacher_stable, teacher_final = _clip_windows(teacher)
    null_stable, null_final = _clip_windows(null)
    teacher_lcbs = _dual_bank_lcbs(teacher)
    null_lcbs = _dual_bank_lcbs(null)
    score_overall = _first(teacher, "selection_overall_score_gain", "overall_score_gain", "audit_overall_score_gain")
    score_data_end = _first(teacher, "selection_data_end_score_gain", "data_end_score_gain", "audit_data_end_score_gain")
    cosine_overall = _first(teacher, "selection_overall_flux_cosine", "overall_flux_cosine")
    cosine_data_end = _first(teacher, "selection_data_end_flux_cosine", "data_end_flux_cosine")
    relative_overall = _first(teacher, "selection_overall_relative_flux_l2", "overall_relative_flux_l2")
    relative_data_end = _first(teacher, "selection_data_end_relative_flux_l2", "data_end_relative_flux_l2")
    lr = candidate.get("learning_rate")
    task_complete = all(bool(int(value.get("complete", 0))) for value in (teacher, null))
    task_finite = all(bool(int(value.get("finite", 0))) for value in (teacher, null))
    task_boundary = all(bool(int(value.get("boundary_admissible", 0))) for value in (teacher, null))
    clipping = [teacher_stable, teacher_final, null_stable, null_final]
    selection_risk = _risk(teacher)
    observed_clips = [float(value) for value in clipping if _finite(value)]
    checks = [
        _check("learning_rate", lr, "in", list(thresholds.pilot_learning_rates), _finite(lr) and float(lr) in {float(value) for value in thresholds.pilot_learning_rates}),
        _check("tasks_complete", int(task_complete), "==", 1, task_complete),
        _check("tasks_finite", int(task_finite), "==", 1, task_finite),
        _check("tasks_boundary_admissible", int(task_boundary), "==", 1, task_boundary),
        _check("stability_clip_fractions", clipping[::2], "<= each", thresholds.maximum_clip_fraction, all(_finite(value) and 0.0 <= float(value) <= thresholds.maximum_clip_fraction for value in clipping[::2])),
        _check("final_clip_fractions", clipping[1::2], "<= each", thresholds.maximum_clip_fraction, all(_finite(value) and 0.0 <= float(value) <= thresholds.maximum_clip_fraction for value in clipping[1::2])),
        _check("teacher_selected_nonzero", teacher.get("selected_step"), ">", 0, int(teacher.get("selected_step", 0)) > 0),
        _check("teacher_dual_bank_lcbs", teacher_lcbs, "> 0 each", 0.0, all(value is not None and value > 0.0 for value in teacher_lcbs)),
        _check("teacher_selection_risk", selection_risk, "finite", True, _finite(selection_risk)),
        _check("teacher_score_gains", [score_overall, score_data_end], "> 0 each", 0.0, all(_finite(value) and float(value) > 0.0 for value in (score_overall, score_data_end))),
        _check("teacher_flux_cosines", [cosine_overall, cosine_data_end], "> 0 each", 0.0, all(_finite(value) and float(value) > 0.0 for value in (cosine_overall, cosine_data_end))),
        _check("teacher_relative_flux_l2", [relative_overall, relative_data_end], "< 1 each", 1.0, all(_finite(value) and 0.0 <= float(value) < 1.0 for value in (relative_overall, relative_data_end))),
        _check("null_selected_zero", null.get("selected_step"), "==", 0, int(null.get("selected_step", -1)) == 0),
        _check("null_no_positive_bank", null_lcbs, "<= 0 each", 0.0, all(value is not None and value <= 0.0 for value in null_lcbs)),
    ]
    result = _finish(
        "stability_pilot_candidate",
        checks,
        "train/selection-only coupled implicit-teacher and Dirichlet-null stability profile",
        evaluation_status=_status(candidate),
    )
    result.update(
        {
            "learning_rate": float(lr) if _finite(lr) else None,
            "mean_teacher_selection_risk": selection_risk,
            "maximum_clip_fraction_observed": max(observed_clips) if observed_clips else None,
        }
    )
    return result


def _pilot_gates(
    candidates: Sequence[Mapping[str, Any]], thresholds: StabilityThresholds
) -> list[dict[str, Any]]:
    return [
        dict(value)
        if value.get("gate") == "stability_pilot_candidate" and "subchecks" in value
        else evaluate_pilot_candidate(value, thresholds)
        for value in candidates
    ]


def select_stability_profile(
    candidates: Sequence[Mapping[str, Any]], thresholds: StabilityThresholds | None = None
) -> dict[str, Any]:
    """Select the eligible profile by risk, clipping, then smaller LR."""

    thresholds = thresholds or StabilityThresholds()
    gates = _pilot_gates(candidates, thresholds)
    eligible = [(index, gate) for index, gate in enumerate(gates) if _passed(gate)]
    if not eligible:
        return {
            "schema": SCHEMA + "-selected-profile",
            "schema_version": SCHEMA_VERSION,
            "selected": 0,
            "passed": 0,
            "profile": None,
            "selected_candidate_index": None,
            "ranking": [],
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    ranking = sorted(
        eligible,
        key=lambda value: (
            float(value[1]["mean_teacher_selection_risk"]),
            float(value[1]["maximum_clip_fraction_observed"]),
            float(value[1]["learning_rate"]),
        ),
    )
    index, gate = ranking[0]
    return {
        "schema": SCHEMA + "-selected-profile",
        "schema_version": SCHEMA_VERSION,
        "selected": 1,
        "passed": 1,
        "profile": {
            "learning_rate": gate["learning_rate"],
            "mean_teacher_selection_risk": gate["mean_teacher_selection_risk"],
            "maximum_clip_fraction_observed": gate["maximum_clip_fraction_observed"],
        },
        "selected_candidate_index": int(index),
        "candidate_gate": gate,
        "ranking": [
            {
                "candidate_index": int(item_index),
                "learning_rate": item["learning_rate"],
                "mean_teacher_selection_risk": item["mean_teacher_selection_risk"],
                "maximum_clip_fraction_observed": item["maximum_clip_fraction_observed"],
            }
            for item_index, item in ranking
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_stability_pilot(
    candidates: Sequence[Mapping[str, Any]], thresholds: StabilityThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or StabilityThresholds()
    gates = _pilot_gates(candidates, thresholds)
    selection = select_stability_profile(gates, thresholds)
    actual_lrs = [gate.get("learning_rate") for gate in gates]
    expected_lrs = [float(value) for value in thresholds.pilot_learning_rates]
    checks = [
        _check("candidate_count", len(gates), "==", len(expected_lrs), len(gates) == len(expected_lrs)),
        _check("learning_rate_set", sorted(value for value in actual_lrs if value is not None), "==", sorted(expected_lrs), len(actual_lrs) == len(expected_lrs) and sorted(actual_lrs) == sorted(expected_lrs)),
        _check("eligible_profile", selection["selected"], "==", 1, bool(selection["selected"])),
    ]
    status = "evaluated" if gates else "not_evaluated"
    result = _finish(
        "stability_pilot",
        checks,
        "train/selection-only learning-rate stability qualification",
        evaluation_status=status,
    )
    result["candidate_gates"] = gates
    result["selected_profile"] = selection
    return result


def _boundary_thresholds(thresholds: StabilityThresholds) -> BoundaryControlThresholds:
    return BoundaryControlThresholds(
        expected_implicit_teacher_seeds=thresholds.confirmation_teacher_seeds,
        minimum_passing_implicit_teacher_seeds=thresholds.minimum_passing_teacher_seeds,
        expected_null_seeds=thresholds.confirmation_null_seeds,
        maximum_post_warmup_clip_fraction=thresholds.maximum_clip_fraction,
    )


def _optimizer_healthy(metrics: Mapping[str, Any], maximum: float) -> bool:
    clip = metrics.get("post_warmup_clip_fraction")
    return (
        bool(int(metrics.get("complete", 0)))
        and bool(int(metrics.get("finite", 0)))
        and bool(int(metrics.get("boundary_admissible", 0)))
        and _finite(clip)
        and 0.0 <= float(clip) <= maximum
    )


def evaluate_stability_confirmation(
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    thresholds: StabilityThresholds | None = None,
    probe_bank_status: ProbeBankStatus | str = ProbeBankStatus.NOT_EVALUATED,
) -> dict[str, Any]:
    """Evaluate fresh three-seed teacher/null confirmation evidence."""

    thresholds = thresholds or StabilityThresholds()
    probe_status = _probe_status(probe_bank_status)
    teacher_metrics = [_metrics(value) for value in teacher_results]
    null_metrics = [_metrics(value) for value in null_results]
    boundary = _boundary_thresholds(thresholds)
    teacher_study = evaluate_implicit_teacher_study(teacher_metrics, boundary)
    null_study = evaluate_null_study(null_metrics, boundary)
    all_metrics = teacher_metrics + null_metrics
    optimizer_health = bool(all_metrics) and len(teacher_metrics) == thresholds.confirmation_teacher_seeds and len(null_metrics) == thresholds.confirmation_null_seeds and all(
        _optimizer_healthy(value, thresholds.maximum_clip_fraction)
        for value in all_metrics
    )
    checks = [
        _check("optimizer_health", int(optimizer_health), "==", 1, optimizer_health),
        _check("implicit_teacher_study", teacher_study["passed"], "==", 1, _passed(teacher_study)),
        _check("null_study", null_study["passed"], "==", 1, _passed(null_study)),
        _check("probe_bank_status", probe_status.value, "==", ProbeBankStatus.AGREE.value, probe_status is ProbeBankStatus.AGREE),
    ]
    evaluated = bool(teacher_results or null_results)
    result = _finish(
        "stability_confirmation",
        checks,
        "optimizer-healthy fresh streamed bounded-teacher and Dirichlet-null controls",
        evaluation_status="evaluated" if evaluated else "not_evaluated",
    )
    result.update(
        {
            "optimizer_health_pass": int(optimizer_health),
            "implicit_teacher_study": teacher_study,
            "null_study": null_study,
            "probe_bank_status": probe_status.value,
            "studies_complete": int(
                len(teacher_metrics) == thresholds.confirmation_teacher_seeds
                and len(null_metrics) == thresholds.confirmation_null_seeds
            ),
        }
    )
    return result


def decide_stability_confirmation(
    *,
    provenance: bool | int | Mapping[str, Any],
    stein: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    confirmation: bool | int | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the closed decision with fail-closed precedence."""

    confirmation_status = _status(confirmation)
    probe = _probe_status(
        confirmation.get("probe_bank_status", ProbeBankStatus.NOT_EVALUATED.value)
        if isinstance(confirmation, Mapping)
        else ProbeBankStatus.NOT_EVALUATED
    )
    optimizer_health = bool(
        isinstance(confirmation, Mapping)
        and int(confirmation.get("optimizer_health_pass", 0)) == 1
    )
    teacher_pass = bool(
        isinstance(confirmation, Mapping)
        and _passed(confirmation.get("implicit_teacher_study", 0))
    )
    null_pass = bool(
        isinstance(confirmation, Mapping)
        and _passed(confirmation.get("null_study", 0))
    )

    if not _passed(provenance):
        decision = StabilityDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the parent scale-repair and transitive provenance binding"
    elif _status(stein) != "evaluated":
        decision = StabilityDecision.STABILITY_PREFLIGHT_INVALID
        action = "complete the exact Stein-identity stability preflight"
    elif not _passed(stein):
        decision = StabilityDecision.OPERATOR_IDENTITY_INVALID
        action = "repair the analytic operator identity before more optimization"
    elif _status(pilot) != "evaluated":
        decision = StabilityDecision.OPTIMIZER_STABILITY_UNRESOLVED
        action = "run the train/selection-only stability pilot"
    elif not _passed(pilot):
        decision = StabilityDecision.OPTIMIZER_STABILITY_UNRESOLVED
        action = "investigate an explicit function-space trust or coercivity constraint"
    elif confirmation_status != "evaluated":
        decision = StabilityDecision.OPTIMIZER_STABILITY_UNRESOLVED
        action = "run the fresh three-seed stability confirmation"
    elif not optimizer_health:
        decision = StabilityDecision.OPTIMIZER_STABILITY_INVALID
        action = "repair the pilot-qualified profile's confirmation optimizer health"
    elif probe is ProbeBankStatus.DISAGREE:
        decision = StabilityDecision.TRACE_ESTIMATOR_INCONCLUSIVE
        action = "resolve the independent audit-probe disagreement"
    elif probe is ProbeBankStatus.NOT_EVALUATED:
        decision = StabilityDecision.TRACE_ESTIMATOR_INCONCLUSIVE
        action = "complete both independent audit probe banks"
    elif not teacher_pass or not null_pass:
        decision = StabilityDecision.IMPLICIT_OBJECTIVE_UNSTABLE
        action = "implement the predeclared density-ratio-classification controls"
    else:
        decision = StabilityDecision.CONTROL_PIPELINE_REPAIRED
        action = "plan fresh physical implicit-score training with untouched audit paths"

    return {
        "decision": decision.value,
        "recommended_next_action": action,
        "probe_bank_status": probe.value,
        "physical_training_authorized": int(
            decision is StabilityDecision.CONTROL_PIPELINE_REPAIRED
        ),
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


def evaluate_stability_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    stein: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    confirmation: bool | int | Mapping[str, Any],
    require_gate: str = "none",
) -> dict[str, Any]:
    """Evaluate cumulative required gates and the stability decision."""

    required = str(require_gate)
    if required not in {"none", "preflight", "pilot", "controls"}:
        raise ValueError("require_gate must be none, preflight, pilot, or controls")
    preflight_pass = _passed(provenance) and _passed(stein)
    pilot_pass = preflight_pass and _passed(pilot)
    controls_pass = pilot_pass and _passed(confirmation)
    requirement = {
        "none": True,
        "preflight": preflight_pass,
        "pilot": pilot_pass,
        "controls": controls_pass,
    }[required]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "required_gate": required,
        "required_gate_pass": int(requirement),
        "preflight_pass": int(preflight_pass),
        "pilot_pass": int(pilot_pass),
        "controls_pass": int(controls_pass),
        "components": {
            "provenance": dict(provenance) if isinstance(provenance, Mapping) else int(_passed(provenance)),
            "stein_identity": dict(stein) if isinstance(stein, Mapping) else int(_passed(stein)),
            "pilot": dict(pilot) if isinstance(pilot, Mapping) else int(_passed(pilot)),
            "confirmation": dict(confirmation) if isinstance(confirmation, Mapping) else int(_passed(confirmation)),
        },
        "decision": decide_stability_confirmation(
            provenance=provenance,
            stein=stein,
            pilot=pilot,
            confirmation=confirmation,
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
