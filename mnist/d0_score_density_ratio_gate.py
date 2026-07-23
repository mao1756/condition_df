"""Pure gates for boundary-admissible density-ratio classification controls.

The classifier is trained with equal class priors, so its population-optimal
logit is ``log(p_tau / nu)``.  This module never trains a model and never reads
files.  It codifies the asymmetric checkpoint protocol used to avoid the null
false discovery seen in the preceding implicit-score run:

* state panel A nominates exactly one nonzero checkpoint;
* independent state panel B tests only that nominee;
* untouched state panels C and D provide final audit evidence.

The analytic score and physical-flux thresholds are inherited unchanged from
the boundary-control experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_score_boundary_control_gate import BoundaryControlThresholds


__all__ = [
    "DensityRatioDecision",
    "DensityRatioThresholds",
    "not_evaluated_gate",
    "nominate_checkpoint_on_a",
    "confirm_nominee_on_b",
    "select_density_ratio_checkpoint",
    "evaluate_ratio_preflight",
    "evaluate_density_ratio_pilot_candidate",
    "evaluate_density_ratio_pilot",
    "select_density_ratio_profile",
    "evaluate_teacher_seed",
    "evaluate_teacher_study",
    "evaluate_null_seed",
    "evaluate_null_study",
    "evaluate_density_ratio_controls",
    "decide_density_ratio_controls",
    "evaluate_density_ratio_workflow",
]


SCHEMA = "experiment12-d0-score-density-ratio-control-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DensityRatioThresholds:
    # The classifier oracle is a 99% preflight check.  Checkpoint confirmation
    # and final audit retain the experiment's frozen one-sided 90% whole-path
    # bootstrap rule.
    oracle_confidence: float = 0.99
    confirm_confidence: float = 0.90
    audit_confidence: float = 0.90
    maximum_clip_fraction: float = 0.10
    pilot_learning_rates: tuple[float, ...] = (3e-4, 1e-4, 3e-5, 1e-5)
    expected_teacher_seeds: int = 3
    minimum_passing_teacher_seeds: int = 2
    expected_null_seeds: int = 3
    expected_audit_panels: tuple[str, ...] = ("c", "d")
    audit_paths_per_panel: int = 32
    anchors_per_path: int = 32
    expected_time_bins: int = 5
    analytic_derivative_tolerance: float = 1e-8
    analytic_null_tolerance: float = 1e-12
    teacher: BoundaryControlThresholds = field(default_factory=BoundaryControlThresholds)

    def __post_init__(self) -> None:
        for name in ("oracle_confidence", "confirm_confidence", "audit_confidence"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0,1)")
        if not 0.0 <= float(self.maximum_clip_fraction) <= 1.0:
            raise ValueError("maximum_clip_fraction must lie in [0,1]")
        if not self.pilot_learning_rates or any(
            not _finite(value) or float(value) <= 0.0
            for value in self.pilot_learning_rates
        ):
            raise ValueError("pilot_learning_rates must be finite and positive")
        if len(set(float(value) for value in self.pilot_learning_rates)) != len(
            self.pilot_learning_rates
        ):
            raise ValueError("pilot_learning_rates must be distinct")
        if not 1 <= int(self.minimum_passing_teacher_seeds) <= int(
            self.expected_teacher_seeds
        ):
            raise ValueError("teacher seed counts are invalid")
        if int(self.expected_null_seeds) <= 0:
            raise ValueError("expected_null_seeds must be positive")
        if tuple(self.expected_audit_panels) != ("c", "d"):
            raise ValueError("density-ratio audit panels are frozen at c,d")
        if int(self.audit_paths_per_panel) <= 0 or int(self.anchors_per_path) <= 0:
            raise ValueError("panel sizes must be positive")
        if int(self.expected_time_bins) != int(self.teacher.expected_time_bins):
            raise ValueError("time-bin count must match frozen teacher thresholds")
        if float(self.analytic_derivative_tolerance) <= 0.0:
            raise ValueError("analytic_derivative_tolerance must be positive")
        if float(self.analytic_null_tolerance) <= 0.0:
            raise ValueError("analytic_null_tolerance must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DensityRatioDecision(str, Enum):
    DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED = "density_ratio_control_pipeline_repaired"
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    RATIO_OPERATOR_INVALID = "ratio_operator_invalid"
    CLASSIFICATION_OPTIMIZER_UNRESOLVED = "classification_optimizer_unresolved"
    CLASSIFICATION_OPTIMIZER_INVALID = "classification_optimizer_invalid"
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    CLASSIFICATION_AUDIT_INCONCLUSIVE = "classification_audit_inconclusive"
    DENSITY_RATIO_VALUE_ONLY = "density_ratio_value_only"
    NO_DETECTABLE_DENSITY_RATIO_SIGNAL = "no_detectable_density_ratio_signal"


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
        "skipped": "not_evaluated",
        "pending": "not_evaluated",
        "incomplete": "not_evaluated",
    }
    return aliases.get(raw.strip().lower(), raw.strip().lower())


def _first(value: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in value:
            return value[name]
    return default


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
    gate: str,
    checks: Sequence[tuple[str, Mapping[str, Any]]],
    claim_scope: str,
    *,
    evaluation_status: str = "evaluated",
) -> dict[str, Any]:
    subchecks = {name: dict(value) for name, value in checks}
    passed = evaluation_status == "evaluated" and bool(subchecks) and all(
        _passed(value) for value in subchecks.values()
    )
    return {
        "gate": gate,
        "evaluation_status": evaluation_status,
        "passed": int(passed),
        "subchecks": subchecks,
        "claim_scope": claim_scope,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
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


def _scope(panel: Mapping[str, Any], scope: str) -> dict[str, Any]:
    value = panel.get(scope, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _panel(checkpoint: Mapping[str, Any], name: str) -> dict[str, Any]:
    panels = checkpoint.get("panels", {})
    if isinstance(panels, Mapping) and isinstance(panels.get(name), Mapping):
        return dict(panels[name])
    value = checkpoint.get(f"panel_{name}", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _bce(record: Mapping[str, Any]) -> float | None:
    raw = _first(record, "bce", "model_bce", "classification_risk", "risk")
    return float(raw) if _finite(raw) else None


def _improvement(record: Mapping[str, Any]) -> float | None:
    raw = _first(
        record,
        "improvement",
        "bce_improvement",
        "classification_improvement",
        "mean_improvement",
    )
    return float(raw) if _finite(raw) else None


def _lower_bound(record: Mapping[str, Any]) -> float | None:
    raw = _first(
        record,
        "lower_bound",
        "improvement_lower_bound",
        "classification_improvement_lower_bound",
        "objective_improvement_lower_bound",
        "bootstrap_lower_bound",
    )
    return float(raw) if _finite(raw) else None


def nominate_checkpoint_on_a(
    checkpoints: Sequence[Mapping[str, Any]],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    """Nominate one EMA checkpoint using panel A only.

    The lowest finite panel-A overall BCE wins, with ties toward the earliest
    step.  Panel B/C/D fields are deliberately ignored.  Step zero remains the
    fallback selection outcome but is not a scientific nominee.
    """

    del thresholds
    rows: list[dict[str, Any]] = []
    step_zero_present = False
    for raw in checkpoints:
        row = dict(raw)
        try:
            step = int(row.get("step", -1))
        except (TypeError, ValueError):
            step = -1
        if step == 0:
            step_zero_present = bool(int(row.get("finite", 1)))
            continue
        panel_a = _panel(row, "a")
        overall = _scope(panel_a, "overall")
        data_end = _scope(panel_a, "data_end")
        risk = _bce(overall)
        finite = bool(int(row.get("finite", 0)))
        ema = bool(int(row.get("ema", row.get("is_ema", 1))))
        if step > 0 and finite and ema and risk is not None:
            rows.append(
                {
                    "step": step,
                    "panel_a_overall_bce": risk,
                    "panel_a_lower_bounds": [
                        _lower_bound(overall),
                        _lower_bound(data_end),
                    ],
                    "panel_a_confidence": _first(
                        panel_a,
                        "confidence",
                        "bootstrap_confidence",
                        default=_first(overall, "confidence"),
                    ),
                }
            )
    rows.sort(key=lambda row: (float(row["panel_a_overall_bce"]), int(row["step"])))
    nominee = rows[0] if rows else None
    checks = [
        _check("step_zero_present", int(step_zero_present), "==", 1, step_zero_present),
        _check("nonzero_nominee_available", int(nominee is not None), "==", 1, nominee is not None),
    ]
    result = _finish(
        "density_ratio_panel_a_nomination",
        checks,
        "single nonzero EMA checkpoint nomination using discovery state panel A only",
    )
    result.update(
        {
            "nominee_step": None if nominee is None else int(nominee["step"]),
            "nominee_panel_a_overall_bce": None
            if nominee is None
            else float(nominee["panel_a_overall_bce"]),
            "nominee_panel_a_lower_bounds": []
            if nominee is None
            else list(nominee["panel_a_lower_bounds"]),
            "nominee_panel_a_confidence": None
            if nominee is None
            else nominee["panel_a_confidence"],
            "candidate_count": len(rows),
        }
    )
    return result


def confirm_nominee_on_b(
    nomination: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    """Test the single A nominee on B without scanning B checkpoints."""

    thresholds = thresholds or DensityRatioThresholds()
    nominee = nomination.get("nominee_step")
    matches = []
    if nominee is not None:
        matches = [row for row in checkpoints if int(row.get("step", -1)) == int(nominee)]
    checkpoint = dict(matches[0]) if len(matches) == 1 else {}
    panel_b = _panel(checkpoint, "b")
    overall = _scope(panel_b, "overall")
    data_end = _scope(panel_b, "data_end")
    bounds = [_lower_bound(overall), _lower_bound(data_end)]
    risk = _bce(overall)
    confidence = _first(
        panel_b,
        "confidence",
        default=_first(overall, "confidence", default=_first(data_end, "confidence")),
    )
    checks = [
        _check("a_nomination_pass", int(_passed(nomination)), "==", 1, _passed(nomination)),
        _check("unique_nominee_record", len(matches), "==", 1, len(matches) == 1),
        _check("confirm_confidence", confidence, ">=", thresholds.confirm_confidence, _finite(confidence) and float(confidence) >= thresholds.confirm_confidence),
        _check("panel_b_lower_bounds", bounds, "> 0 each", 0.0, all(value is not None and value > 0.0 for value in bounds)),
    ]
    result = _finish(
        "density_ratio_panel_b_confirmation",
        checks,
        "independent confirmation state panel B tests only the panel-A nominee",
    )
    accepted = _passed(result)
    result.update(
        {
            "nominee_step": None if nominee is None else int(nominee),
            "accepted": int(accepted),
            "selected_step": int(nominee) if nominee is not None and accepted else 0,
            "panel_b_lower_bounds": bounds,
            "panel_b_overall_bce": risk,
            "panel_b_confidence": confidence,
        }
    )
    return result


def select_density_ratio_checkpoint(
    checkpoints: Sequence[Mapping[str, Any]],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    nomination = nominate_checkpoint_on_a(checkpoints, thresholds)
    confirmation = confirm_nominee_on_b(nomination, checkpoints, thresholds)
    return {
        "gate": "density_ratio_checkpoint_selection",
        "evaluation_status": "evaluated",
        "passed": int(_passed(nomination)),
        "selected_step": int(confirmation.get("selected_step", 0)),
        "nominee_step": nomination.get("nominee_step"),
        "nomination": nomination,
        "confirmation": confirmation,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    candidate = value.get("metrics", value)
    return dict(candidate) if isinstance(candidate, Mapping) else {}


def _selection(metrics: Mapping[str, Any], thresholds: DensityRatioThresholds) -> dict[str, Any]:
    checkpoints = metrics.get("checkpoints", metrics.get("checkpoint_records", []))
    if isinstance(checkpoints, Sequence) and not isinstance(checkpoints, (str, bytes)) and checkpoints:
        return select_density_ratio_checkpoint(
            [dict(value) for value in checkpoints if isinstance(value, Mapping)], thresholds
        )
    value = metrics.get("selection", metrics.get("checkpoint_selection", {}))
    return dict(value) if isinstance(value, Mapping) else {}


def _optimizer_checks(
    metrics: Mapping[str, Any], thresholds: DensityRatioThresholds
) -> list[tuple[str, Mapping[str, Any]]]:
    clip = metrics.get("post_warmup_clip_fraction")
    return [
        _check("complete", int(bool(metrics.get("complete", 0))), "==", 1, bool(metrics.get("complete", 0))),
        _check("finite", int(bool(metrics.get("finite", 0))), "==", 1, bool(metrics.get("finite", 0))),
        _check("boundary_admissible", int(bool(metrics.get("boundary_admissible", 0))), "==", 1, bool(metrics.get("boundary_admissible", 0))),
        _check("post_warmup_clip_fraction", clip, "<=", thresholds.maximum_clip_fraction, _finite(clip) and 0.0 <= float(clip) <= thresholds.maximum_clip_fraction),
    ]


def evaluate_ratio_preflight(
    metrics: Mapping[str, Any],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    normalization = metrics.get("teacher_normalization_interval", {})
    normalization = dict(normalization) if isinstance(normalization, Mapping) else {}
    norm_lower = _first(normalization, "lower", "lower_bound")
    norm_upper = _first(normalization, "upper", "upper_bound")
    teacher_bounds = metrics.get("teacher_bce_improvement_lower_bounds", {})
    teacher_bounds = dict(teacher_bounds) if isinstance(teacher_bounds, Mapping) else {}
    bce_bounds = [_lower_bound(_scope(teacher_bounds, scope)) for scope in ("overall", "data_end")]
    derivative_errors = [
        metrics.get("analytic_logit_max_error"),
        metrics.get("analytic_score_max_error"),
        metrics.get("analytic_flux_max_error"),
    ]
    null_errors = [
        metrics.get("null_bce_error"),
        metrics.get("null_score_max_abs"),
        metrics.get("null_flux_max_abs"),
    ]
    oracle_confidence = _first(
        metrics,
        "oracle_bootstrap_confidence",
        "teacher_oracle_bootstrap_confidence",
        "bootstrap_confidence",
    )
    operator_pass = _first(
        metrics,
        "operator_preflight_pass",
        "boundary_operator_preflight_pass",
        "operator_pass",
        default=0,
    )
    device_smoke_pass = _first(
        metrics,
        "device_smoke_pass",
        "production_device_smoke_pass",
        "forward_backward_device_smoke_pass",
        default=0,
    )
    checks = [
        _check("complete", int(bool(metrics.get("complete", 0))), "==", 1, bool(metrics.get("complete", 0))),
        _check("finite", int(bool(metrics.get("finite", 0))), "==", 1, bool(metrics.get("finite", 0))),
        _check("analytic_teacher_derivatives", derivative_errors, "<= each", thresholds.analytic_derivative_tolerance, all(_finite(value) and 0.0 <= float(value) <= thresholds.analytic_derivative_tolerance for value in derivative_errors)),
        _check("teacher_normalization", [norm_lower, norm_upper], "contains", 1.0, _finite(norm_lower) and _finite(norm_upper) and float(norm_lower) <= 1.0 <= float(norm_upper)),
        _check("teacher_bce_signal", bce_bounds, "> 0 each", 0.0, all(value is not None and value > 0.0 for value in bce_bounds)),
        _check("oracle_bootstrap_confidence", oracle_confidence, "==", thresholds.oracle_confidence, _finite(oracle_confidence) and math.isclose(float(oracle_confidence), thresholds.oracle_confidence, rel_tol=0.0, abs_tol=1e-15)),
        _check("null_exact_zero", null_errors, "<= each", thresholds.analytic_null_tolerance, all(_finite(value) and 0.0 <= float(value) <= thresholds.analytic_null_tolerance for value in null_errors)),
        _check("class_balance", int(bool(metrics.get("class_balance_pass", 0))), "==", 1, bool(metrics.get("class_balance_pass", 0))),
        _check("time_strata", int(bool(metrics.get("time_strata_pass", 0))), "==", 1, bool(metrics.get("time_strata_pass", 0))),
        _check("null_exchangeability", int(bool(metrics.get("null_exchangeability_pass", 0))), "==", 1, bool(metrics.get("null_exchangeability_pass", 0))),
        _check("independent_class_namespaces", int(bool(metrics.get("independent_class_namespaces", 0))), "==", 1, bool(metrics.get("independent_class_namespaces", 0))),
        _check("stream_replay", int(bool(metrics.get("stream_replay_pass", 0))), "==", 1, bool(metrics.get("stream_replay_pass", 0))),
        _check("panel_isolation", int(bool(metrics.get("panel_isolation_pass", 0))), "==", 1, bool(metrics.get("panel_isolation_pass", 0))),
        _check("boundary_certificate", int(bool(metrics.get("boundary_admissible", 0))), "==", 1, bool(metrics.get("boundary_admissible", 0))),
        _check("operator_preflight", int(bool(operator_pass)), "==", 1, bool(operator_pass)),
        _check("production_device_smoke", int(bool(device_smoke_pass)), "==", 1, bool(device_smoke_pass)),
        _check("physical_training_performed", int(metrics.get("physical_training_performed", 0)), "==", 0, int(metrics.get("physical_training_performed", 0)) == 0),
        _check("sampling_performed", int(metrics.get("sampling_performed", 0)), "==", 0, int(metrics.get("sampling_performed", 0)) == 0),
    ]
    result = _finish(
        "density_ratio_preflight",
        checks,
        "equal-prior Bayes-logit identity, null exchangeability, and isolated state panels",
        evaluation_status=_status(metrics),
    )
    result["thresholds"] = thresholds.to_dict()
    return result


def _selection_b_risk(selection: Mapping[str, Any]) -> float | None:
    confirmation = selection.get("confirmation", {})
    if not isinstance(confirmation, Mapping):
        return None
    value = confirmation.get("panel_b_overall_bce")
    if _finite(value):
        return float(value)
    return None


def _selection_a_risk(selection: Mapping[str, Any]) -> float | None:
    nomination = selection.get("nomination", {})
    if not isinstance(nomination, Mapping):
        return None
    value = nomination.get("nominee_panel_a_overall_bce")
    return float(value) if _finite(value) else None


def _selection_bounds(
    selection: Mapping[str, Any], panel: str
) -> list[float | None]:
    if panel == "a":
        nomination = selection.get("nomination", {})
        source = dict(nomination) if isinstance(nomination, Mapping) else {}
        raw = source.get("nominee_panel_a_lower_bounds", [])
    elif panel == "b":
        confirmation = selection.get("confirmation", {})
        source = dict(confirmation) if isinstance(confirmation, Mapping) else {}
        raw = source.get("panel_b_lower_bounds", [])
    else:
        raise ValueError("selection panel must be a or b")
    values = list(raw) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []
    return [float(value) if _finite(value) else None for value in values]


def _pilot_analytic_metrics(teacher: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the selected-checkpoint analytic metrics used by the pilot."""

    for name in (
        "selected_analytic_metrics",
        "selection_analytic_metrics",
        "panel_b_analytic_metrics",
        "analytic_metrics",
    ):
        value = teacher.get(name)
        if isinstance(value, Mapping):
            source = dict(value)
            break
    else:
        source = dict(teacher)
    overall = _scope(source, "overall")
    data_end = _scope(source, "data_end")
    bin_cosines = source.get("time_bin_flux_cosines", [])
    bin_relatives = source.get("time_bin_relative_flux_l2", [])
    bin_cosines = list(bin_cosines) if isinstance(bin_cosines, Sequence) and not isinstance(bin_cosines, (str, bytes)) else []
    bin_relatives = list(bin_relatives) if isinstance(bin_relatives, Sequence) and not isinstance(bin_relatives, (str, bytes)) else []
    return {
        "score_gain_overall": _first(
            source,
            "selection_overall_score_gain",
            "audit_overall_score_gain",
            "overall_score_gain",
            default=_first(overall, "score_gain"),
        ),
        "score_gain_data_end": _first(
            source,
            "selection_data_end_score_gain",
            "audit_data_end_score_gain",
            "data_end_score_gain",
            default=_first(data_end, "score_gain"),
        ),
        "flux_cosine_overall": _first(
            source,
            "selection_overall_flux_cosine",
            "overall_flux_cosine",
            default=_first(overall, "flux_cosine"),
        ),
        "flux_cosine_data_end": _first(
            source,
            "selection_data_end_flux_cosine",
            "data_end_flux_cosine",
            default=_first(
                data_end,
                "flux_cosine",
                default=bin_cosines[-1] if bin_cosines else None,
            ),
        ),
        "relative_flux_l2_overall": _first(
            source,
            "selection_overall_relative_flux_l2",
            "overall_relative_flux_l2",
            default=_first(overall, "flux_relative_l2", "relative_flux_l2"),
        ),
        "relative_flux_l2_data_end": _first(
            source,
            "selection_data_end_relative_flux_l2",
            "data_end_relative_flux_l2",
            default=_first(
                data_end,
                "flux_relative_l2",
                "relative_flux_l2",
                default=bin_relatives[-1] if bin_relatives else None,
            ),
        ),
    }


def evaluate_density_ratio_pilot_candidate(
    candidate: Mapping[str, Any],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    teacher = _metrics(candidate.get("teacher", candidate.get("teacher_metrics", {})))
    null = _metrics(candidate.get("null", candidate.get("null_metrics", {})))
    teacher_selection = _selection(teacher, thresholds)
    null_selection = _selection(null, thresholds)
    teacher_selected = int(teacher_selection.get("selected_step", teacher.get("selected_step", 0)))
    null_selected = int(null_selection.get("selected_step", null.get("selected_step", -1)))
    teacher_confirmed = bool(int(dict(teacher_selection.get("confirmation", {})).get("accepted", teacher_selected > 0)))
    null_confirmed = bool(int(dict(null_selection.get("confirmation", {})).get("accepted", null_selected > 0)))
    analytic = _pilot_analytic_metrics(teacher)
    score_gains = [analytic["score_gain_overall"], analytic["score_gain_data_end"]]
    cosines = [analytic["flux_cosine_overall"], analytic["flux_cosine_data_end"]]
    relatives = [
        analytic["relative_flux_l2_overall"],
        analytic["relative_flux_l2_data_end"],
    ]
    null_a_bounds = _selection_bounds(null_selection, "a")
    null_b_bounds = _selection_bounds(null_selection, "b")
    optimizer_checks = _optimizer_checks(teacher, thresholds) + [
        (f"null_{name}", value) for name, value in _optimizer_checks(null, thresholds)
    ]
    panel_a_risk = _selection_a_risk(teacher_selection)
    panel_b_risk = _selection_b_risk(teacher_selection)
    mean_ab_risk = (
        0.5 * (panel_a_risk + panel_b_risk)
        if panel_a_risk is not None and panel_b_risk is not None
        else None
    )
    checks = [
        _check("learning_rate", candidate.get("learning_rate"), "in", list(thresholds.pilot_learning_rates), _finite(candidate.get("learning_rate")) and float(candidate["learning_rate"]) in {float(value) for value in thresholds.pilot_learning_rates}),
        *optimizer_checks,
        _check("teacher_b_confirmed", int(teacher_confirmed), "==", 1, teacher_confirmed),
        _check("teacher_selected_nonzero", teacher_selected, ">", 0, teacher_selected > 0),
        _check("teacher_score_gains", score_gains, "> 0 each", 0.0, all(_finite(value) and float(value) > 0.0 for value in score_gains)),
        _check("teacher_flux_cosines", cosines, "> 0 each", 0.0, all(_finite(value) and float(value) > 0.0 for value in cosines)),
        _check("teacher_relative_flux_l2", relatives, "< 1 each", 1.0, all(_finite(value) and 0.0 <= float(value) < 1.0 for value in relatives)),
        _check("null_b_rejected", int(not null_confirmed), "==", 1, not null_confirmed),
        _check("null_selected_zero", null_selected, "==", 0, null_selected == 0),
        _check("null_panel_a_lower_bounds", null_a_bounds, "<= 0 each", 0.0, len(null_a_bounds) == 2 and all(value is not None and value <= 0.0 for value in null_a_bounds)),
        _check("null_panel_b_lower_bounds", null_b_bounds, "<= 0 each", 0.0, len(null_b_bounds) == 2 and all(value is not None and value <= 0.0 for value in null_b_bounds)),
        _check("teacher_mean_ab_bce", mean_ab_risk, "finite", True, _finite(mean_ab_risk)),
    ]
    clips = [teacher.get("post_warmup_clip_fraction"), null.get("post_warmup_clip_fraction")]
    result = _finish(
        "density_ratio_pilot_candidate",
        checks,
        "train/selection-only classifier learning-rate profile",
        evaluation_status=_status(candidate),
    )
    result.update(
        {
            "learning_rate": float(candidate["learning_rate"]) if _finite(candidate.get("learning_rate")) else None,
            "teacher_panel_a_bce": panel_a_risk,
            "teacher_panel_b_bce": panel_b_risk,
            "teacher_mean_ab_bce": mean_ab_risk,
            "maximum_clip_fraction_observed": max((float(value) for value in clips if _finite(value)), default=None),
            "optimizer_health_pass": int(all(_passed(value) for name, value in optimizer_checks)),
            "teacher_selection": teacher_selection,
            "null_selection": null_selection,
            "teacher_analytic_metrics": analytic,
        }
    )
    return result


def _pilot_gates(
    candidates: Sequence[Mapping[str, Any]], thresholds: DensityRatioThresholds
) -> list[dict[str, Any]]:
    return [
        dict(value)
        if value.get("gate") == "density_ratio_pilot_candidate" and "subchecks" in value
        else evaluate_density_ratio_pilot_candidate(value, thresholds)
        for value in candidates
    ]


def select_density_ratio_profile(
    candidates: Sequence[Mapping[str, Any]],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    gates = _pilot_gates(candidates, thresholds)
    eligible = [
        (index, gate)
        for index, gate in enumerate(gates)
        if _passed(gate)
        and _finite(gate.get("teacher_mean_ab_bce"))
        and _finite(gate.get("maximum_clip_fraction_observed"))
    ]
    eligible.sort(
        key=lambda value: (
            float(value[1]["teacher_mean_ab_bce"]),
            float(value[1]["maximum_clip_fraction_observed"]),
            float(value[1]["learning_rate"]),
        )
    )
    if not eligible:
        return {
            "schema": SCHEMA + "-selected-profile",
            "schema_version": 1,
            "selected": 0,
            "passed": 0,
            "selected_candidate_index": None,
            "profile": None,
            "ranking": [],
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    index, gate = eligible[0]
    return {
        "schema": SCHEMA + "-selected-profile",
        "schema_version": 1,
        "selected": 1,
        "passed": 1,
        "selected_candidate_index": int(index),
        "profile": {
            "learning_rate": gate["learning_rate"],
            "teacher_mean_ab_bce": gate["teacher_mean_ab_bce"],
            "teacher_panel_b_bce": gate["teacher_panel_b_bce"],
            "maximum_clip_fraction_observed": gate["maximum_clip_fraction_observed"],
        },
        "ranking": [
            {
                "candidate_index": int(item_index),
                "learning_rate": item["learning_rate"],
                "teacher_mean_ab_bce": item["teacher_mean_ab_bce"],
                "teacher_panel_b_bce": item["teacher_panel_b_bce"],
                "maximum_clip_fraction_observed": item["maximum_clip_fraction_observed"],
            }
            for item_index, item in eligible
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_density_ratio_pilot(
    candidates: Sequence[Mapping[str, Any]],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    gates = _pilot_gates(candidates, thresholds)
    profile = select_density_ratio_profile(gates, thresholds)
    actual_lrs = [gate.get("learning_rate") for gate in gates]
    expected_lrs = [float(value) for value in thresholds.pilot_learning_rates]
    selected_index = profile.get("selected_candidate_index")
    optimizer_health = bool(profile.get("selected", 0)) and selected_index is not None and bool(
        int(gates[int(selected_index)].get("optimizer_health_pass", 0))
    )
    checks = [
        _check("candidate_count", len(gates), "==", len(expected_lrs), len(gates) == len(expected_lrs)),
        _check("learning_rate_set", sorted(value for value in actual_lrs if value is not None), "==", sorted(expected_lrs), len(actual_lrs) == len(expected_lrs) and sorted(actual_lrs) == sorted(expected_lrs)),
        _check("eligible_profile", profile["selected"], "==", 1, bool(profile["selected"])),
    ]
    result = _finish(
        "density_ratio_pilot",
        checks,
        "train/selection-only classifier optimizer qualification",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result["candidate_gates"] = gates
    result["selected_profile"] = profile
    result["optimizer_health_pass"] = int(optimizer_health)
    return result


def _audit_panels(metrics: Mapping[str, Any]) -> dict[str, Any]:
    value = metrics.get("audit_panels", metrics.get("audit_state_panels", {}))
    return dict(value) if isinstance(value, Mapping) else {}


def _classification_bounds(panel: Mapping[str, Any]) -> list[float | None]:
    source = panel.get("classification_improvement", panel)
    source = dict(source) if isinstance(source, Mapping) else {}
    return [_lower_bound(_scope(source, scope)) for scope in ("overall", "data_end")]


def _teacher_panel_gate(
    name: str, panel: Mapping[str, Any], thresholds: DensityRatioThresholds
) -> dict[str, Any]:
    teacher = thresholds.teacher
    bounds = _classification_bounds(panel)
    confidence = _first(panel, "confidence", "bootstrap_confidence")
    score_gains = [
        _first(panel, "overall_score_gain", "audit_overall_score_gain"),
        _first(panel, "data_end_score_gain", "audit_data_end_score_gain"),
    ]
    cosines = list(panel.get("time_bin_flux_cosines", []))
    relatives = list(panel.get("time_bin_relative_flux_l2", []))
    path_count = panel.get("path_count")
    anchors = panel.get("anchors_per_path")
    checks = [
        _check("finite", int(bool(panel.get("finite", 0))), "==", 1, bool(panel.get("finite", 0))),
        _check("path_count", path_count, "==", thresholds.audit_paths_per_panel, path_count is not None and int(path_count) == thresholds.audit_paths_per_panel),
        _check("anchors_per_path", anchors, "==", thresholds.anchors_per_path, anchors is not None and int(anchors) == thresholds.anchors_per_path),
        _check("audit_confidence", confidence, ">=", thresholds.audit_confidence, _finite(confidence) and float(confidence) >= thresholds.audit_confidence),
        _check("classification_lower_bounds", bounds, "> 0 each", 0.0, all(value is not None and value > 0.0 for value in bounds)),
        _check("score_gains", score_gains, ">= each", teacher.teacher_min_score_gain, all(_finite(value) and float(value) >= teacher.teacher_min_score_gain for value in score_gains)),
        _check("overall_flux_cosine", panel.get("overall_flux_cosine"), ">=", teacher.teacher_min_overall_flux_cosine, _finite(panel.get("overall_flux_cosine")) and float(panel["overall_flux_cosine"]) >= teacher.teacher_min_overall_flux_cosine),
        _check("time_bin_flux_cosines", cosines, ">= each", teacher.teacher_min_bin_flux_cosine, len(cosines) == thresholds.expected_time_bins and all(_finite(value) and float(value) >= teacher.teacher_min_bin_flux_cosine for value in cosines)),
        _check("overall_relative_flux_l2", panel.get("overall_relative_flux_l2"), "<=", teacher.teacher_max_overall_relative_flux_l2, _finite(panel.get("overall_relative_flux_l2")) and 0.0 <= float(panel["overall_relative_flux_l2"]) <= teacher.teacher_max_overall_relative_flux_l2),
        _check("time_bin_relative_flux_l2", relatives, "<= each", teacher.teacher_max_bin_relative_flux_l2, len(relatives) == thresholds.expected_time_bins and all(_finite(value) and 0.0 <= float(value) <= teacher.teacher_max_bin_relative_flux_l2 for value in relatives)),
    ]
    result = _finish(
        f"density_ratio_teacher_audit_panel_{name}",
        checks,
        f"untouched teacher audit state panel {name.upper()}",
    )
    result["classification_pass"] = int(
        all(value is not None and value > 0.0 for value in bounds)
    )
    derivative_names = {
        "score_gains",
        "overall_flux_cosine",
        "time_bin_flux_cosines",
        "overall_relative_flux_l2",
        "time_bin_relative_flux_l2",
    }
    result["derivative_pass"] = int(
        all(_passed(result["subchecks"][key]) for key in derivative_names)
    )
    return result


def evaluate_teacher_seed(
    metrics: Mapping[str, Any],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    metrics = _metrics(metrics)
    selection = _selection(metrics, thresholds)
    selected = int(selection.get("selected_step", metrics.get("selected_step", 0)))
    confirmation = selection.get("confirmation", {})
    confirmed = bool(int(dict(confirmation).get("accepted", selected > 0))) if isinstance(confirmation, Mapping) else selected > 0
    panels = _audit_panels(metrics)
    panel_gates = {
        name: _teacher_panel_gate(
            name,
            dict(panels.get(name, {})) if isinstance(panels.get(name), Mapping) else {},
            thresholds,
        )
        for name in thresholds.expected_audit_panels
    }
    checks = [
        *_optimizer_checks(metrics, thresholds),
        _check("panel_b_confirmed", int(confirmed), "==", 1, confirmed),
        _check("selected_nonzero", selected, ">", 0, selected > 0),
        *[
            _check(f"audit_panel_{name}", gate["passed"], "==", 1, _passed(gate))
            for name, gate in panel_gates.items()
        ],
    ]
    result = _finish(
        "density_ratio_teacher_seed",
        checks,
        "held-out density-ratio classification and frozen analytic score/flux recovery",
        evaluation_status=_status(metrics),
    )
    classifications = [bool(int(gate["classification_pass"])) for gate in panel_gates.values()]
    derivatives = [bool(int(gate["derivative_pass"])) for gate in panel_gates.values()]
    result.update(
        {
            "model_seed": metrics.get("model_seed", metrics.get("seed")),
            "selected_step": selected,
            "selection": selection,
            "audit_panel_gates": panel_gates,
            "classification_pass": int(all(classifications)),
            "derivative_pass": int(all(derivatives)),
            "panel_disagreement": int(len(set(gate["passed"] for gate in panel_gates.values())) > 1 or len(set(classifications)) > 1),
            "optimizer_health_pass": int(all(_passed(value) for _, value in _optimizer_checks(metrics, thresholds))),
        }
    )
    return result


def evaluate_teacher_study(
    seed_results: Sequence[Mapping[str, Any]],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "density_ratio_teacher_seed" and "subchecks" in value
        else evaluate_teacher_seed(value, thresholds)
        for value in seed_results
    ]
    seeds = [gate.get("model_seed") for gate in gates]
    optimizer_valid = len(gates) == thresholds.expected_teacher_seeds and all(
        bool(int(gate.get("optimizer_health_pass", 0))) for gate in gates
    )
    pass_count = sum(_passed(gate) for gate in gates)
    classification_count = sum(bool(int(gate.get("classification_pass", 0))) for gate in gates)
    derivative_count = sum(bool(int(gate.get("derivative_pass", 0))) for gate in gates)
    disagreement = any(bool(int(gate.get("panel_disagreement", 0))) for gate in gates)
    checks = [
        _check("task_count", len(gates), "==", thresholds.expected_teacher_seeds, len(gates) == thresholds.expected_teacher_seeds),
        _check("distinct_seeds", len(set(seeds)), "==", thresholds.expected_teacher_seeds, None not in seeds and len(set(seeds)) == thresholds.expected_teacher_seeds),
        _check("all_optimizers_valid", int(optimizer_valid), "==", 1, optimizer_valid),
        _check("passing_seeds", pass_count, ">=", thresholds.minimum_passing_teacher_seeds, pass_count >= thresholds.minimum_passing_teacher_seeds),
        _check("audit_panels_agree", int(not disagreement), "==", 1, not disagreement),
    ]
    result = _finish(
        "density_ratio_teacher_study",
        checks,
        "three-seed density-ratio bounded-teacher control",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result.update(
        {
            "seed_gates": gates,
            "passing_seed_count": pass_count,
            "classification_passing_seed_count": classification_count,
            "derivative_passing_seed_count": derivative_count,
            "optimizer_health_pass": int(optimizer_valid),
            "panel_disagreement": int(disagreement),
        }
    )
    return result


def _null_panel_gate(
    name: str, panel: Mapping[str, Any], thresholds: DensityRatioThresholds
) -> dict[str, Any]:
    bounds = _classification_bounds(panel)
    confidence = _first(panel, "confidence", "bootstrap_confidence")
    checks = [
        _check("finite", int(bool(panel.get("finite", 0))), "==", 1, bool(panel.get("finite", 0))),
        _check("path_count", panel.get("path_count"), "==", thresholds.audit_paths_per_panel, panel.get("path_count") is not None and int(panel["path_count"]) == thresholds.audit_paths_per_panel),
        _check("anchors_per_path", panel.get("anchors_per_path"), "==", thresholds.anchors_per_path, panel.get("anchors_per_path") is not None and int(panel["anchors_per_path"]) == thresholds.anchors_per_path),
        _check("audit_confidence", confidence, ">=", thresholds.audit_confidence, _finite(confidence) and float(confidence) >= thresholds.audit_confidence),
        _check("no_positive_lower_bound", bounds, "<= 0 each", 0.0, all(value is not None and value <= 0.0 for value in bounds)),
    ]
    result = _finish(
        f"density_ratio_null_nominee_audit_panel_{name}",
        checks,
        f"untouched null audit of the panel-A nominee on state panel {name.upper()}",
    )
    result["positive_false_discovery"] = int(
        any(value is not None and value > 0.0 for value in bounds)
    )
    return result


def evaluate_null_seed(
    metrics: Mapping[str, Any],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    metrics = _metrics(metrics)
    selection = _selection(metrics, thresholds)
    selected = int(selection.get("selected_step", metrics.get("selected_step", -1)))
    nominee = selection.get("nominee_step", metrics.get("nominee_step"))
    confirmation = selection.get("confirmation", {})
    accepted = bool(int(dict(confirmation).get("accepted", selected > 0))) if isinstance(confirmation, Mapping) else selected > 0
    panels = _audit_panels(metrics)
    panel_gates = {
        name: _null_panel_gate(
            name,
            dict(panels.get(name, {})) if isinstance(panels.get(name), Mapping) else {},
            thresholds,
        )
        for name in thresholds.expected_audit_panels
    }
    false_discovery = accepted or selected != 0 or any(
        bool(int(gate.get("positive_false_discovery", 0))) for gate in panel_gates.values()
    )
    checks = [
        *_optimizer_checks(metrics, thresholds),
        _check("analytic_zero_comparator", metrics.get("comparator"), "==", "analytic_zero", metrics.get("comparator") in {"analytic_zero", "analytic_zero_step0"}),
        _check("nonzero_a_nominee_audited", nominee, ">", 0, nominee is not None and int(nominee) > 0),
        _check("panel_b_rejected", int(not accepted), "==", 1, not accepted),
        _check("selected_step_zero", selected, "==", 0, selected == 0),
        *[
            _check(f"audit_panel_{name}", gate["passed"], "==", 1, _passed(gate))
            for name, gate in panel_gates.items()
        ],
    ]
    result = _finish(
        "density_ratio_null_seed",
        checks,
        "stationary Dirichlet classification null with explicit rejected-nominee audit",
        evaluation_status=_status(metrics),
    )
    result.update(
        {
            "model_seed": metrics.get("model_seed", metrics.get("seed")),
            "selected_step": selected,
            "nominee_step": nominee,
            "selection": selection,
            "audit_panel_gates": panel_gates,
            "false_discovery": int(false_discovery),
            "optimizer_health_pass": int(all(_passed(value) for _, value in _optimizer_checks(metrics, thresholds))),
        }
    )
    return result


def evaluate_null_study(
    seed_results: Sequence[Mapping[str, Any]],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "density_ratio_null_seed" and "subchecks" in value
        else evaluate_null_seed(value, thresholds)
        for value in seed_results
    ]
    seeds = [gate.get("model_seed") for gate in gates]
    optimizer_valid = len(gates) == thresholds.expected_null_seeds and all(
        bool(int(gate.get("optimizer_health_pass", 0))) for gate in gates
    )
    false_discoveries = sum(bool(int(gate.get("false_discovery", 0))) for gate in gates)
    checks = [
        _check("task_count", len(gates), "==", thresholds.expected_null_seeds, len(gates) == thresholds.expected_null_seeds),
        _check("distinct_seeds", len(set(seeds)), "==", thresholds.expected_null_seeds, None not in seeds and len(set(seeds)) == thresholds.expected_null_seeds),
        _check("all_optimizers_valid", int(optimizer_valid), "==", 1, optimizer_valid),
        _check("all_null_seeds_pass", sum(_passed(gate) for gate in gates), "==", thresholds.expected_null_seeds, len(gates) == thresholds.expected_null_seeds and all(_passed(gate) for gate in gates)),
        _check("false_discovery_count", false_discoveries, "==", 0, false_discoveries == 0),
    ]
    result = _finish(
        "density_ratio_null_study",
        checks,
        "three-seed stationary-Dirichlet classification null",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result.update(
        {
            "seed_gates": gates,
            "optimizer_health_pass": int(optimizer_valid),
            "false_discovery_count": int(false_discoveries),
        }
    )
    return result


def evaluate_density_ratio_controls(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    teacher = evaluate_teacher_study(teacher_results, thresholds)
    null = evaluate_null_study(null_results, thresholds)
    checks = [
        _check("provenance", int(_passed(provenance)), "==", 1, _passed(provenance)),
        _check("preflight", int(_passed(preflight)), "==", 1, _passed(preflight)),
        _check("pilot", int(_passed(pilot)), "==", 1, _passed(pilot)),
        _check("teacher_study", teacher["passed"], "==", 1, _passed(teacher)),
        _check("null_study", null["passed"], "==", 1, _passed(null)),
    ]
    result = _finish(
        "density_ratio_controls",
        checks,
        "optimizer-healthy density-ratio classification teacher and null controls",
        evaluation_status="evaluated" if teacher_results or null_results else "not_evaluated",
    )
    result["teacher_study"] = teacher
    result["null_study"] = null
    result["optimizer_health_pass"] = int(
        bool(int(teacher.get("optimizer_health_pass", 0)))
        and bool(int(null.get("optimizer_health_pass", 0)))
    )
    return result


def decide_density_ratio_controls(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    teacher = controls.get("teacher_study", {})
    null = controls.get("null_study", {})
    teacher = dict(teacher) if isinstance(teacher, Mapping) else {}
    null = dict(null) if isinstance(null, Mapping) else {}
    if not _passed(provenance):
        decision = DensityRatioDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the stability-run and transitive provenance binding"
    elif _status(preflight) != "evaluated" or not _passed(preflight):
        decision = DensityRatioDecision.RATIO_OPERATOR_INVALID
        action = "repair the Bayes-logit, exchangeability, or panel-isolation preflight"
    elif _status(pilot) != "evaluated":
        decision = DensityRatioDecision.CLASSIFICATION_OPTIMIZER_UNRESOLVED
        action = "run the train/selection-only density-ratio pilot"
    elif not _passed(pilot):
        decision = DensityRatioDecision.CLASSIFICATION_OPTIMIZER_UNRESOLVED
        action = "no pilot profile qualified; inspect optimizer and selection-only evidence"
    elif not bool(int(controls.get("optimizer_health_pass", 0))):
        decision = DensityRatioDecision.CLASSIFICATION_OPTIMIZER_INVALID
        action = "repair incomplete, nonfinite, clipped, or incompatible classifier tasks"
    elif int(null.get("false_discovery_count", 0)) > 0:
        decision = DensityRatioDecision.SELECTION_FALSE_DISCOVERY
        action = "repair discovery/confirmation calibration before more score learning"
    elif bool(int(teacher.get("panel_disagreement", 0))):
        decision = DensityRatioDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "rerun unchanged with 64 whole paths per audit panel"
    elif _passed(controls):
        decision = DensityRatioDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        action = "plan a fresh physical density-ratio-score experiment with new audit paths"
    elif int(teacher.get("classification_passing_seed_count", 0)) >= 2:
        decision = DensityRatioDecision.DENSITY_RATIO_VALUE_ONLY
        action = "add derivative or physical-flux regularization before physical training"
    else:
        decision = DensityRatioDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL
        action = "revisit classifier capacity or optimization on the exact synthetic law"
    return {
        "decision": decision.value,
        "recommended_next_action": action,
        "physical_training_authorized": int(
            decision is DensityRatioDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        ),
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


def evaluate_density_ratio_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    require_gate: str = "none",
    thresholds: DensityRatioThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DensityRatioThresholds()
    required = str(require_gate)
    if required not in {"none", "preflight", "pilot", "controls"}:
        raise ValueError("require_gate must be none, preflight, pilot, or controls")
    controls = evaluate_density_ratio_controls(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        teacher_results=teacher_results,
        null_results=null_results,
        thresholds=thresholds,
    )
    preflight_pass = _passed(provenance) and _passed(preflight)
    pilot_pass = preflight_pass and _passed(pilot)
    controls_pass = pilot_pass and _passed(controls)
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
            "preflight": dict(preflight) if isinstance(preflight, Mapping) else int(_passed(preflight)),
            "pilot": dict(pilot) if isinstance(pilot, Mapping) else int(_passed(pilot)),
            "controls": controls,
        },
        "decision": decide_density_ratio_controls(
            provenance=provenance,
            preflight=preflight,
            pilot=pilot,
            controls=controls,
        ),
        "thresholds": thresholds.to_dict(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
