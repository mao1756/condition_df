"""Pure gates for the gradient-ratio-controlled D0 H1 experiment.

This module is intentionally additive and side-effect free.  It contains no
model, optimiser, filesystem, or sampler code.  The balanced-BCE gradient and
the stopped-EMA H1 gradient are composed by the orchestration layer; the gates
below verify that the requested component ratio was realised and adjudicate
the fixed-step synthetic teacher/null evidence.

The important distinction from the preceding H1 experiment is that step 4000
is the only scientific endpoint.  Checkpoints before that step are restart and
learning-curve artefacts only.  A non-zero ratio must pass the inherited
derivative thresholds and demonstrate a matched reduction relative to the
``rho=0`` arm at the same endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_score_density_ratio_h1_trust_gate import H1TrustThresholds


SCHEMA = "experiment12-d0-score-density-ratio-h1-gradient-control-gate"
SCHEMA_VERSION = 1


__all__ = [
    "H1GradientControlDecision",
    "H1GradientControlThresholds",
    "GradientControlDecision",
    "GradientControlThresholds",
    "not_evaluated_gate",
    "normalize_gradient_control_candidate",
    "evaluate_gradient_controller_preflight",
    "evaluate_h1_gradient_controller_preflight",
    "evaluate_gradient_control_preflight",
    "evaluate_gradient_control_pilot_candidate",
    "evaluate_h1_gradient_control_pilot_candidate",
    "rank_gradient_control_candidates",
    "evaluate_gradient_control_pilot",
    "evaluate_h1_gradient_control_pilot",
    "evaluate_gradient_control_confirmation_seed",
    "evaluate_gradient_control_confirmation",
    "evaluate_h1_gradient_control_confirmation",
    "decide_gradient_control_workflow",
    "decide_h1_gradient_control_workflow",
    "evaluate_gradient_control_workflow",
    "evaluate_h1_gradient_control_workflow",
]


@dataclass(frozen=True)
class H1GradientControlThresholds:
    """Frozen design and inherited scientific thresholds."""

    h1_trust: H1TrustThresholds = field(default_factory=H1TrustThresholds)
    target_ratios: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0)
    fixed_endpoint_step: int = 4_000
    ramp_steps: int = 100
    gradient_norm_floor: float = 1e-12
    maximum_ratio_relative_error: float = 1e-4
    minimum_controller_active_fraction: float = 0.99
    minimum_relative_l2_reduction: float = 0.10
    body_learning_rate: float = 3e-5
    accumulation_steps: int = 8
    base_channels: int = 32
    expected_confirmation_seeds: int = 3
    minimum_passing_confirmation_seeds: int = 2
    confirmation_panel_roles: tuple[str, ...] = ("b", "c", "d")
    pilot_null_family_size: int = 8
    confirmation_family_size: int = 18

    def __post_init__(self) -> None:
        if self.h1_trust != H1TrustThresholds():
            raise ValueError("inherited H1/density-ratio thresholds must remain frozen")
        if tuple(float(x) for x in self.target_ratios) != (0.0, 0.1, 0.3, 1.0):
            raise ValueError("target ratios are frozen at 0, 0.1, 0.3, and 1")
        frozen = {
            "fixed_endpoint_step": (int(self.fixed_endpoint_step), 4_000),
            "ramp_steps": (int(self.ramp_steps), 100),
            "accumulation_steps": (int(self.accumulation_steps), 8),
            "base_channels": (int(self.base_channels), 32),
            "expected_confirmation_seeds": (
                int(self.expected_confirmation_seeds),
                3,
            ),
            "minimum_passing_confirmation_seeds": (
                int(self.minimum_passing_confirmation_seeds),
                2,
            ),
            "pilot_null_family_size": (int(self.pilot_null_family_size), 8),
            "confirmation_family_size": (int(self.confirmation_family_size), 18),
        }
        for name, (actual, expected) in frozen.items():
            if actual != expected:
                raise ValueError(f"{name} is frozen at {expected}")
        floats = {
            "gradient_norm_floor": (float(self.gradient_norm_floor), 1e-12),
            "maximum_ratio_relative_error": (
                float(self.maximum_ratio_relative_error),
                1e-4,
            ),
            "minimum_controller_active_fraction": (
                float(self.minimum_controller_active_fraction),
                0.99,
            ),
            "minimum_relative_l2_reduction": (
                float(self.minimum_relative_l2_reduction),
                0.10,
            ),
            "body_learning_rate": (float(self.body_learning_rate), 3e-5),
        }
        for name, (actual, expected) in floats.items():
            if not math.isfinite(actual) or actual != expected:
                raise ValueError(f"{name} is frozen at {expected}")
        if tuple(str(x).lower() for x in self.confirmation_panel_roles) != (
            "b",
            "c",
            "d",
        ):
            raise ValueError("confirmation panel roles are frozen at B,C,D")

    @property
    def teacher(self) -> Any:
        return self.h1_trust.teacher

    @property
    def maximum_clip_fraction(self) -> float:
        return self.h1_trust.maximum_clip_fraction

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class H1GradientControlDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    H1_GRADIENT_CONTROLLER_INVALID = "h1_gradient_controller_invalid"
    EVIDENCE_PANEL_UNDERPOWERED = "evidence_panel_underpowered"
    H1_CONTROLLER_OPTIMIZER_INVALID = "h1_controller_optimizer_invalid"
    H1_CONTROLLER_OVERREGULARIZED = "h1_controller_overregularized"
    H1_STRENGTH_GRID_UNRESOLVED = "h1_strength_grid_unresolved"
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    H1_CAUSAL_EFFECT_UNCONFIRMED = "h1_causal_effect_unconfirmed"
    CLASSIFICATION_AUDIT_INCONCLUSIVE = "classification_audit_inconclusive"
    H1_EFFECT_AUDIT_INCONCLUSIVE = "h1_effect_audit_inconclusive"
    H1_DENSITY_RATIO_VALUE_ONLY = "h1_density_ratio_value_only"
    DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED = (
        "density_ratio_control_pipeline_repaired"
    )


# Short aliases are useful to callers which already carry the H1 context in
# their module name.
GradientControlThresholds = H1GradientControlThresholds
GradientControlDecision = H1GradientControlDecision


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _status(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "evaluated"
    raw = str(value.get("evaluation_status", "evaluated")).strip().lower()
    return {
        "complete": "evaluated",
        "completed": "evaluated",
        "pending": "not_evaluated",
        "skipped": "not_evaluated",
        "incomplete": "not_evaluated",
    }.get(raw, raw)


def _passed(value: bool | int | Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        return _one(value.get("passed", value.get("gate_pass", 0)))
    return value is True or (isinstance(value, int) and value == 1)


def _deep_first(value: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in value:
            return value[name]
    for child in value.values():
        if isinstance(child, Mapping):
            found = _deep_first(child, names)
            if found is not None:
                return found
    return None


def _array(value: Any) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if all(_finite(item) for item in values):
            return [float(item) for item in values]
    return []


def _check(
    name: str,
    value: Any,
    operator: str,
    threshold: Any,
    passed: bool,
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
    return {
        "gate": gate,
        "evaluation_status": evaluation_status,
        "passed": int(
            evaluation_status == "evaluated"
            and bool(subchecks)
            and all(_passed(value) for value in subchecks.values())
        ),
        "claim_scope": claim_scope,
        "subchecks": subchecks,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def not_evaluated_gate(gate: str, reason: str) -> dict[str, Any]:
    result = _finish(gate, [], "not evaluated", evaluation_status="not_evaluated")
    result["reason"] = str(reason)
    return result


def _first(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    value = _deep_first(record, names)
    return default if value is None else value


def _panel_record(record: Mapping[str, Any], role: str) -> dict[str, Any]:
    role = str(role).strip().lower()
    panels = _mapping(record.get("panels", {}))
    audit_panels = _mapping(record.get("audit_panels", {}))
    direct = record.get(f"panel_{role}", panels.get(role, audit_panels.get(role)))
    if isinstance(direct, Mapping):
        return dict(direct)
    # Flat orchestration summaries are also accepted.  Copy only role-prefixed
    # fields first; fall back to the full record for legacy compact fixtures.
    prefix = f"panel_{role}_"
    compact = {
        str(key)[len(prefix) :]: value
        for key, value in record.items()
        if str(key).startswith(prefix)
    }
    return compact or dict(record)


def _normalize_panel(record: Mapping[str, Any], role: str) -> dict[str, Any]:
    panel = _panel_record(record, role)
    lower = _first(
        panel,
        "bce_improvement_lower_bounds",
        "lower_bounds",
        "teacher_lower_bounds",
        f"teacher_panel_{role}_lower_bounds",
        default=[],
    )
    if isinstance(lower, Mapping):
        lower = [lower.get("overall"), lower.get("data_end")]
    cosines = _array(
        _first(
            panel,
            "time_bin_flux_cosines",
            "flux_cosine_by_time_bin",
            "teacher_time_bin_flux_cosines",
            default=[],
        )
    )
    relatives = _array(
        _first(
            panel,
            "time_bin_relative_flux_l2",
            "relative_flux_l2_by_time_bin",
            "teacher_time_bin_relative_flux_l2",
            default=[],
        )
    )
    result = {
        "evaluation_status": str(_first(panel, "evaluation_status", default="evaluated")),
        "opened": _first(panel, "opened", "evaluated", default=1),
        "evaluation_count": _first(panel, "evaluation_count", "open_count", default=1),
        "confirmed": _first(panel, "confirmed", "accepted", default=1),
        "bce_improvement_lower_bounds": _array(lower),
        "bce": _first(panel, "bce", "overall_bce", "teacher_panel_b_bce"),
        "score_gain_overall": _first(
            panel, "score_gain_overall", "overall_score_gain", "teacher_score_gain_overall"
        ),
        "score_gain_data_end": _first(
            panel, "score_gain_data_end", "data_end_score_gain", "teacher_score_gain_data_end"
        ),
        "flux_cosine_overall": _first(
            panel, "flux_cosine_overall", "overall_flux_cosine", "teacher_flux_cosine_overall"
        ),
        "time_bin_flux_cosines": cosines,
        "relative_flux_l2_overall": _first(
            panel,
            "relative_flux_l2_overall",
            "overall_relative_flux_l2",
            "teacher_relative_flux_l2_overall",
        ),
        "relative_flux_l2_data_end": _first(
            panel,
            "relative_flux_l2_data_end",
            "data_end_relative_flux_l2",
            "teacher_relative_flux_l2_data_end",
        ),
        "time_bin_relative_flux_l2": relatives,
    }
    return result


def _task(record: Mapping[str, Any], name: str) -> dict[str, Any]:
    aliases = {
        "teacher": ("teacher", "bounded_teacher", "selected_teacher"),
        "baseline": ("baseline", "rho_zero", "baseline_teacher"),
        "null": ("null", "dirichlet_null"),
    }[name]
    for alias in aliases:
        value = record.get(alias)
        if isinstance(value, Mapping):
            metrics = value.get("metrics")
            return dict(metrics) if isinstance(metrics, Mapping) else dict(value)
    return dict(record)


def _clip_max(record: Mapping[str, Any]) -> Any:
    direct = _first(record, "maximum_clip_fraction_observed", "max_clip_fraction")
    if _finite(direct):
        return float(direct)
    raw = _first(record, "clipping_windows", "clip_fractions", default={})
    values: list[Any] = []
    if isinstance(raw, Mapping):
        values = list(raw.values())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    finite = [float(value) for value in values if _finite(value)]
    return max(finite) if finite else None


def _normalize_task_health(record: Mapping[str, Any], ratio: float | None) -> dict[str, Any]:
    controller = _mapping(record.get("controller", record.get("controller_diagnostics", {})))

    def first(*names: str, default: Any = None) -> Any:
        value = _deep_first(controller, names)
        if value is None:
            value = _deep_first(record, names)
        return default if value is None else value

    return {
        "complete": first("complete"),
        "finite": first("finite"),
        "boundary_admissible": first("boundary_admissible"),
        "optimizer_health_pass": first("optimizer_health_pass"),
        "controller_health_pass": first("controller_health_pass"),
        "fixed_endpoint_step": first(
            "fixed_endpoint_step", "endpoint_step", "completed_step", "step"
        ),
        "target_ratio": ratio if ratio is not None else first("target_ratio", "rho", "ratio"),
        "controller_active_fraction": first(
            "controller_active_fraction",
            "post_ramp_active_fraction",
            "active_fraction_post_ramp",
        ),
        "maximum_ratio_relative_error": first(
            "maximum_ratio_relative_error",
            "max_ratio_relative_error",
            "ratio_tracking_relative_error",
        ),
        "post_ramp_h1_floor_hit_count": first(
            "post_ramp_h1_floor_hit_count", "h1_floor_hit_count_post_ramp", default=0
        ),
        "nonfinite_coefficient_count": first(
            "nonfinite_coefficient_count", "nonfinite_lambda_count", default=0
        ),
        "maximum_clip_fraction_observed": _clip_max(record),
    }


def _health_pass(
    record: Mapping[str, Any],
    ratio: float,
    thresholds: H1GradientControlThresholds,
) -> tuple[bool, dict[str, Any]]:
    value = _normalize_task_health(record, ratio)
    common = (
        _one(value.get("complete"))
        and _one(value.get("finite"))
        and _one(value.get("boundary_admissible"))
        and _one(value.get("optimizer_health_pass"))
        and _one(value.get("controller_health_pass"))
        and value.get("fixed_endpoint_step") is not None
        and int(value["fixed_endpoint_step"]) == thresholds.fixed_endpoint_step
        and _finite(value.get("maximum_clip_fraction_observed"))
        and 0.0 <= float(value["maximum_clip_fraction_observed"])
        <= thresholds.maximum_clip_fraction
        and _zero(value.get("post_ramp_h1_floor_hit_count", 0))
        and _zero(value.get("nonfinite_coefficient_count", 0))
    )
    if float(ratio) == 0.0:
        tracking = (
            value.get("maximum_ratio_relative_error") is None
            or (
                _finite(value.get("maximum_ratio_relative_error"))
                and float(value["maximum_ratio_relative_error"])
                <= thresholds.maximum_ratio_relative_error
            )
        )
    else:
        tracking = (
            _finite(value.get("controller_active_fraction"))
            and float(value["controller_active_fraction"])
            >= thresholds.minimum_controller_active_fraction
            and _finite(value.get("maximum_ratio_relative_error"))
            and float(value["maximum_ratio_relative_error"])
            <= thresholds.maximum_ratio_relative_error
        )
    return bool(common and tracking), value


def normalize_gradient_control_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize compact or nested pilot records without inspecting files."""

    raw = dict(candidate)
    ratio = _first(raw, "target_ratio", "rho", "ratio", "h1_ratio")
    teacher = _task(raw, "teacher")
    null = _task(raw, "null")
    result = {
        "evaluation_status": str(_first(raw, "evaluation_status", default="evaluated")),
        "target_ratio": ratio,
        "learning_rate": _first(raw, "learning_rate", "body_learning_rate"),
        "accumulation_steps": _first(raw, "accumulation_steps", "accumulation_level"),
        "base_channels": _first(raw, "base_channels", default=32),
        "teacher": teacher,
        "null": null,
        "teacher_panel_a": _normalize_panel(teacher, "a"),
        "teacher_panel_b": _normalize_panel(teacher, "b"),
        "teacher_health": _normalize_task_health(teacher, float(ratio) if _finite(ratio) else None),
        "null_health": _normalize_task_health(null, float(ratio) if _finite(ratio) else None),
        "matched_effects": _mapping(
            raw.get("matched_effects", raw.get("matched_reductions", {}))
        ),
        "panel_b_evaluation_count": _first(
            raw,
            "panel_b_evaluation_count",
            default=_normalize_panel(teacher, "b").get("evaluation_count"),
        ),
    }
    # Explicit compact summaries override recursively discovered values.
    for key in tuple(result):
        if key in raw:
            result[key] = raw[key]
    return result


def evaluate_gradient_controller_preflight(record: Mapping[str, Any]) -> dict[str, Any]:
    """Gate the deterministic controller algebra and replay contract."""

    aliases = {
        "exact_target_ratio_algebra_pass": ("exact_target_ratio_algebra_pass", "ratio_algebra_pass"),
        "stopped_coefficient_pass": ("stopped_coefficient_pass", "stop_gradient_coefficient_pass"),
        "positive_rescaling_invariance_pass": ("positive_rescaling_invariance_pass", "h1_rescaling_invariance_pass"),
        "ramp_endpoints_pass": ("ramp_endpoints_pass", "ramp_schedule_pass"),
        "floor_branches_pass": ("floor_branches_pass", "norm_floor_branches_pass"),
        "fixed_point_pass": ("fixed_point_pass", "identical_anchor_fixed_point_pass"),
        "rho_zero_regression_pass": ("rho_zero_regression_pass", "ratio_zero_regression_pass"),
        "cuda_second_order_backward_pass": ("cuda_second_order_backward_pass", "finite_cuda_backward_pass"),
        "boundary_admissibility_pass": ("boundary_admissibility_pass", "boundary_finite_pass"),
        "candidate_order_invariance_pass": ("candidate_order_invariance_pass",),
        "stateless_stream_replay_pass": ("stateless_stream_replay_pass", "stream_replay_pass"),
        "interruption_replay_pass": ("interruption_replay_pass", "exact_resume_pass"),
        "no_sampler_import_pass": ("no_sampler_import_pass", "sampler_absence_pass"),
        "no_physical_state_training_pass": ("no_physical_state_training_pass", "synthetic_controls_only_pass"),
    }
    checks = [
        _check("complete", record.get("complete"), "==", 1, _one(record.get("complete"))),
        _check("finite", record.get("finite"), "==", 1, _one(record.get("finite"))),
    ]
    for canonical, names in aliases.items():
        value = _first(record, *names)
        checks.append(_check(canonical, value, "==", 1, _one(value)))
    return _finish(
        "h1_gradient_controller_preflight",
        checks,
        "gradient-ratio algebra, fixed-point, boundary, and deterministic replay validity",
        evaluation_status=_status(record),
    )


evaluate_h1_gradient_controller_preflight = evaluate_gradient_controller_preflight


def evaluate_gradient_control_preflight(
    *,
    inherited_preflight: bool | int | Mapping[str, Any],
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    controller_gate = (
        dict(controller)
        if controller.get("gate") == "h1_gradient_controller_preflight"
        else evaluate_gradient_controller_preflight(controller)
    )
    checks = [
        _check("inherited_preflight", int(_passed(inherited_preflight)), "==", 1, _passed(inherited_preflight)),
        _check("gradient_controller", controller_gate.get("passed"), "==", 1, _passed(controller_gate)),
    ]
    result = _finish(
        "h1_gradient_control_preflight",
        checks,
        "inherited H1/operator validity plus the online gradient-ratio controller",
        evaluation_status=(
            "evaluated" if _status(controller_gate) == "evaluated" else "not_evaluated"
        ),
    )
    result["inherited_preflight"] = (
        dict(inherited_preflight)
        if isinstance(inherited_preflight, Mapping)
        else int(_passed(inherited_preflight))
    )
    result["controller_gate"] = controller_gate
    return result


def _classification_pass(panel: Mapping[str, Any], thresholds: H1GradientControlThresholds) -> bool:
    bounds = _array(panel.get("bce_improvement_lower_bounds"))
    return (
        _status(panel) == "evaluated"
        and _one(panel.get("opened", 1))
        and _one(panel.get("confirmed", 1))
        and len(bounds) == 2
        and all(value > 0.0 for value in bounds)
        and _finite(panel.get("score_gain_overall"))
        and float(panel["score_gain_overall"]) >= thresholds.teacher.teacher_min_score_gain
        and _finite(panel.get("score_gain_data_end"))
        and float(panel["score_gain_data_end"]) >= thresholds.teacher.teacher_min_score_gain
    )


def _derivative_pass(panel: Mapping[str, Any], thresholds: H1GradientControlThresholds) -> bool:
    cosines = _array(panel.get("time_bin_flux_cosines"))
    relatives = _array(panel.get("time_bin_relative_flux_l2"))
    data_end = panel.get("relative_flux_l2_data_end")
    return (
        _finite(panel.get("flux_cosine_overall"))
        and float(panel["flux_cosine_overall"])
        >= thresholds.teacher.teacher_min_overall_flux_cosine
        and len(cosines) == int(thresholds.teacher.expected_time_bins)
        and all(value >= thresholds.teacher.teacher_min_bin_flux_cosine for value in cosines)
        and _finite(panel.get("relative_flux_l2_overall"))
        and 0.0 <= float(panel["relative_flux_l2_overall"])
        <= thresholds.teacher.teacher_max_overall_relative_flux_l2
        and _finite(data_end)
        and 0.0 <= float(data_end) <= thresholds.teacher.teacher_max_bin_relative_flux_l2
        and len(relatives) == int(thresholds.teacher.expected_time_bins)
        and all(
            0.0 <= value <= thresholds.teacher.teacher_max_bin_relative_flux_l2
            for value in relatives
        )
    )


def _matched_record(record: Mapping[str, Any], role: str) -> dict[str, Any]:
    matches = _mapping(record.get("matched_effects", record.get("matched_reductions", {})))
    value = matches.get(role)
    if isinstance(value, Mapping):
        return dict(value)
    direct = record.get(f"panel_{role}_matched_effect")
    return dict(direct) if isinstance(direct, Mapping) else {}


def _matched_reductions(
    selected: Mapping[str, Any],
    baseline: Mapping[str, Any],
    supplied: Mapping[str, Any],
) -> tuple[list[float | None], list[float]]:
    point = supplied.get("point_reductions", supplied.get("relative_l2_reductions"))
    if isinstance(point, Mapping):
        point = [point.get("overall"), point.get("data_end")]
    points: list[float | None]
    array = _array(point)
    if len(array) == 2:
        points = array
    else:
        points = []
        for name in ("relative_flux_l2_overall", "relative_flux_l2_data_end"):
            numerator = selected.get(name)
            denominator = baseline.get(name)
            if _finite(numerator) and _finite(denominator) and float(denominator) > 0.0:
                points.append(1.0 - float(numerator) / float(denominator))
            else:
                points.append(None)
    bounds = supplied.get(
        "simultaneous_lower_bounds",
        supplied.get("lower_bounds", supplied.get("matched_lower_bounds", [])),
    )
    if isinstance(bounds, Mapping):
        bounds = [bounds.get("overall"), bounds.get("data_end")]
    return points, _array(bounds)


def evaluate_gradient_control_pilot_candidate(
    candidate: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None = None,
    panel_role: str = "a",
    thresholds: H1GradientControlThresholds | None = None,
) -> dict[str, Any]:
    """Gate one ratio on A or B at the common step-4000 endpoint."""

    thresholds = thresholds or H1GradientControlThresholds()
    value = normalize_gradient_control_candidate(candidate)
    ratio = value.get("target_ratio")
    known = _finite(ratio) and float(ratio) in thresholds.target_ratios
    ratio_f = float(ratio) if _finite(ratio) else math.nan
    baseline_value = normalize_gradient_control_candidate(baseline or {})
    selected_panel = value.get(f"teacher_panel_{panel_role}")
    if not isinstance(selected_panel, Mapping):
        selected_panel = _normalize_panel(_task(value, "teacher"), panel_role)
    baseline_panel = baseline_value.get(f"teacher_panel_{panel_role}")
    if not isinstance(baseline_panel, Mapping):
        baseline_panel = {}
    teacher_health, teacher_health_record = _health_pass(
        _task(value, "teacher"), ratio_f, thresholds
    ) if known else (False, {})
    null_health, null_health_record = _health_pass(
        _task(value, "null"), ratio_f, thresholds
    ) if known else (False, {})
    health = teacher_health and null_health
    classification = _classification_pass(selected_panel, thresholds)
    derivative = _derivative_pass(selected_panel, thresholds)
    supplied = _matched_record(candidate, panel_role)
    points, bounds = _matched_reductions(selected_panel, baseline_panel, supplied)
    reduction = (
        ratio_f != 0.0
        and len(points) == 2
        and all(
            _finite(item) and float(item) >= thresholds.minimum_relative_l2_reduction
            for item in points
        )
        and len(bounds) == 2
        and all(bound > 0.0 for bound in bounds)
    )
    is_baseline = known and ratio_f == 0.0
    checks = [
        _check("known_target_ratio", ratio, "in", list(thresholds.target_ratios), known),
        _check(
            "learning_rate",
            value.get("learning_rate"),
            "==",
            thresholds.body_learning_rate,
            _finite(value.get("learning_rate"))
            and float(value["learning_rate"]) == thresholds.body_learning_rate,
        ),
        _check(
            "accumulation_steps",
            value.get("accumulation_steps"),
            "==",
            thresholds.accumulation_steps,
            value.get("accumulation_steps") is not None
            and int(value["accumulation_steps"]) == thresholds.accumulation_steps,
        ),
        _check(
            "base_channels",
            value.get("base_channels"),
            "==",
            thresholds.base_channels,
            value.get("base_channels") is not None
            and int(value["base_channels"]) == thresholds.base_channels,
        ),
        _check("teacher_and_null_health", int(health), "==", 1, health),
        _check("teacher_classification", int(classification), "==", 1, classification),
    ]
    if not is_baseline:
        checks.extend(
            [
                _check("strict_derivative_thresholds", int(derivative), "==", 1, derivative),
                _check(
                    "matched_relative_l2_point_reductions",
                    points,
                    ">= each",
                    thresholds.minimum_relative_l2_reduction,
                    len(points) == 2
                    and all(
                        _finite(item)
                        and float(item) >= thresholds.minimum_relative_l2_reduction
                        for item in points
                    ),
                ),
                _check(
                    "matched_relative_l2_simultaneous_bounds",
                    bounds,
                    "> each",
                    0.0,
                    len(bounds) == 2 and all(bound > 0.0 for bound in bounds),
                ),
            ]
        )
    result = _finish(
        "h1_gradient_control_pilot_candidate",
        checks,
        f"fixed-step panel-{panel_role.upper()} classification, derivative, and matched-effect evidence",
        evaluation_status=_status(value),
    )
    relatives = _array(selected_panel.get("time_bin_relative_flux_l2"))
    cosines = _array(selected_panel.get("time_bin_flux_cosines"))
    overall_l2 = selected_panel.get("relative_flux_l2_overall")
    overall_cos = selected_panel.get("flux_cosine_overall")
    result.update(
        {
            "target_ratio": ratio,
            "is_baseline": int(is_baseline),
            "panel_role": str(panel_role).lower(),
            "optimizer_and_controller_health_pass": int(health),
            "teacher_health": teacher_health_record,
            "null_health": null_health_record,
            "classification_pass": int(classification),
            "derivative_pass": int(derivative),
            "matched_relative_l2_reductions": points,
            "matched_relative_l2_lower_bounds": bounds,
            "matched_relative_l2_reduction_pass": int(reduction),
            "ranking_max_relative_flux_l2": (
                max(float(overall_l2), *relatives)
                if _finite(overall_l2) and relatives
                else None
            ),
            "ranking_min_flux_cosine": (
                min(float(overall_cos), *cosines)
                if _finite(overall_cos) and cosines
                else None
            ),
            "ranking_bce": selected_panel.get("bce"),
            "panel_metrics": dict(selected_panel),
        }
    )
    return result


evaluate_h1_gradient_control_pilot_candidate = evaluate_gradient_control_pilot_candidate


def rank_gradient_control_candidates(
    candidate_gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [
        (index, dict(value))
        for index, value in enumerate(candidate_gates)
        if _passed(value) and not _one(value.get("is_baseline", 0))
    ]

    def key(item: tuple[int, Mapping[str, Any]]) -> tuple[float, float, float, float, int]:
        index, value = item
        return (
            float(value["ranking_max_relative_flux_l2"])
            if _finite(value.get("ranking_max_relative_flux_l2"))
            else math.inf,
            -float(value["ranking_min_flux_cosine"])
            if _finite(value.get("ranking_min_flux_cosine"))
            else math.inf,
            float(value["ranking_bce"])
            if _finite(value.get("ranking_bce"))
            else math.inf,
            float(value["target_ratio"])
            if _finite(value.get("target_ratio"))
            else math.inf,
            index,
        )

    ranked = sorted(eligible, key=key)
    if not ranked:
        return {
            "selected": 0,
            "selected_candidate_index": None,
            "selected_ratio": None,
            "eligible_candidate_indices": [],
            "ranking_rule": [
                "minimum worst overall/per-bin relative flux L2",
                "maximum minimum-bin flux cosine",
                "minimum panel-A BCE",
                "smaller target ratio",
            ],
        }
    index, value = ranked[0]
    return {
        "selected": 1,
        "selected_candidate_index": index,
        "selected_ratio": float(value["target_ratio"]),
        "eligible_candidate_indices": [item[0] for item in ranked],
        "ranking_max_relative_flux_l2": float(value["ranking_max_relative_flux_l2"]),
        "ranking_min_flux_cosine": float(value["ranking_min_flux_cosine"]),
        "ranking_bce": float(value["ranking_bce"]),
        "ranking_rule": [
            "minimum worst overall/per-bin relative flux L2",
            "maximum minimum-bin flux cosine",
            "minimum panel-A BCE",
            "smaller target ratio",
        ],
    }


def _family_members(family: Mapping[str, Any]) -> list[dict[str, Any]]:
    nested = family.get("max_t_record", family.get("family_record", {}))
    raw = family.get("members", [])
    if not raw and isinstance(nested, Mapping):
        raw = nested.get("members", [])
    return [dict(value) for value in raw if isinstance(value, Mapping)]


def _family_size(family: Mapping[str, Any]) -> int | None:
    members = _family_members(family)
    if members:
        return len(members)
    nested = family.get("max_t_record", family.get("family_record", {}))
    value = family.get("family_size", family.get("member_count"))
    if value is None and isinstance(nested, Mapping):
        value = nested.get("family_size", nested.get("member_count"))
    return int(value) if value is not None else None


def _member_lower(member: Mapping[str, Any]) -> Any:
    return member.get("simultaneous_lower_bound", member.get("lower_bound"))


def _null_family_valid(family: Mapping[str, Any], expected_size: int) -> bool:
    members = _family_members(family)
    finite_nonpositive = (
        all(_finite(_member_lower(value)) and float(_member_lower(value)) <= 0.0 for value in members)
        if members
        else not _one(family.get("familywise_false_discovery", family.get("false_discovery", 0)))
    )
    return (
        _status(family) == "evaluated"
        and _passed(family)
        and _family_size(family) == int(expected_size)
        and finite_nonpositive
    )


def _positive_family_valid(family: Mapping[str, Any], expected_size: int) -> bool:
    members = _family_members(family)
    finite_positive = (
        all(_finite(_member_lower(value)) and float(_member_lower(value)) > 0.0 for value in members)
        if members
        else _one(family.get("all_simultaneous_lower_bounds_positive", family.get("all_positive", 0)))
    )
    return (
        _status(family) == "evaluated"
        and _passed(family)
        and _family_size(family) == int(expected_size)
        and finite_positive
    )


def _matched_family_summary(
    family: Mapping[str, Any],
    *,
    expected_size: int,
    minimum_passing_seeds: int,
) -> tuple[bool, int, set[str]]:
    """Adjudicate the 18-member effect family at the whole-seed level.

    All members remain in one simultaneous max-T family, but the frozen gate
    requires all six B/C/D x overall/data-end bounds for *two of three* seeds,
    not all eighteen bounds.  This helper deliberately keeps those two ideas
    separate.
    """

    members = _family_members(family)
    failed_roles: set[str] = set()
    passing_seed_count = 0
    if members:
        by_seed: dict[str, list[Mapping[str, Any]]] = {}
        for member in members:
            seed = str(member.get("seed", member.get("model_seed", "")))
            by_seed.setdefault(seed, []).append(member)
            lower = _member_lower(member)
            if not _finite(lower) or float(lower) <= 0.0:
                failed_roles.add(
                    str(member.get("panel_role", member.get("role", ""))).lower()
                )
        passing_seed_count = sum(
            len(rows) == 6
            and all(_finite(_member_lower(row)) and float(_member_lower(row)) > 0.0 for row in rows)
            for rows in by_seed.values()
        )
    else:
        passing_raw = family.get(
            "passing_seed_count", family.get("matched_effect_passing_seed_count")
        )
        if passing_raw is not None:
            passing_seed_count = int(passing_raw)
        elif _one(
            family.get(
                "all_simultaneous_lower_bounds_positive", family.get("all_positive", 0)
            )
        ):
            passing_seed_count = 3
        for value in family.get("failed_roles", []):
            failed_roles.add(str(value).lower())
    valid = (
        _status(family) == "evaluated"
        and _passed(family)
        and _family_size(family) == int(expected_size)
        and passing_seed_count >= int(minimum_passing_seeds)
    )
    return valid, passing_seed_count, failed_roles


def _positive_roles(family: Mapping[str, Any]) -> set[str]:
    roles: set[str] = set()
    for member in _family_members(family):
        if _finite(_member_lower(member)) and float(_member_lower(member)) > 0.0:
            roles.add(str(member.get("panel_role", member.get("role", ""))).lower())
    for key in ("positive_member_names_by_role", "failed_roles", "positive_roles"):
        value = family.get(key)
        if isinstance(value, Mapping):
            roles.update(str(role).lower() for role, names in value.items() if names)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            roles.update(str(role).lower() for role in value)
    return roles


def evaluate_gradient_control_pilot(
    candidates: Sequence[Mapping[str, Any]],
    *,
    panel_power: bool | int | Mapping[str, Any],
    null_family: Mapping[str, Any],
    thresholds: H1GradientControlThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or H1GradientControlThresholds()
    normalized = [normalize_gradient_control_candidate(value) for value in candidates]
    baselines = [
        value
        for value in normalized
        if _finite(value.get("target_ratio")) and float(value["target_ratio"]) == 0.0
    ]
    baseline = baselines[0] if len(baselines) == 1 else None
    a_gates = [
        evaluate_gradient_control_pilot_candidate(
            value,
            baseline=baseline if _finite(value.get("target_ratio")) and float(value["target_ratio"]) != 0.0 else None,
            panel_role="a",
            thresholds=thresholds,
        )
        for value in normalized
    ]
    nomination = rank_gradient_control_candidates(a_gates)
    b_gate: dict[str, Any] = not_evaluated_gate(
        "h1_gradient_control_pilot_candidate", "panel A nominated no ratio"
    )
    baseline_b: dict[str, Any] = not_evaluated_gate(
        "h1_gradient_control_pilot_candidate", "panel A nominated no ratio"
    )
    sealed = False
    if _one(nomination.get("selected")) and baseline is not None:
        index = int(nomination["selected_candidate_index"])
        selected = normalized[index]
        b_gate = evaluate_gradient_control_pilot_candidate(
            selected, baseline=baseline, panel_role="b", thresholds=thresholds
        )
        baseline_b = evaluate_gradient_control_pilot_candidate(
            baseline, panel_role="b", thresholds=thresholds
        )
        selected_count = selected.get("panel_b_evaluation_count")
        baseline_count = baseline.get("panel_b_evaluation_count")
        nonselected_counts = [
            value.get("panel_b_evaluation_count")
            for row_index, value in enumerate(normalized)
            if row_index not in {index, normalized.index(baseline)}
        ]
        sealed = (
            selected_count is not None
            and int(selected_count) == 1
            and baseline_count is not None
            and int(baseline_count) == 1
            and all(count is not None and int(count) == 0 for count in nonselected_counts)
        )
    ratios = sorted(
        float(value["target_ratio"])
        for value in normalized
        if _finite(value.get("target_ratio"))
    )
    all_health = len(a_gates) == len(thresholds.target_ratios) and all(
        _one(value.get("optimizer_and_controller_health_pass")) for value in a_gates
    )
    nonzero = [value for value in a_gates if not _one(value.get("is_baseline"))]
    overregularized = bool(nonzero) and all(
        not _one(value.get("classification_pass")) for value in nonzero
    )
    null_behavior = len(normalized) == len(thresholds.target_ratios) and all(
        _first(value.get("null", {}), "selected_step") is not None
        and int(_first(value.get("null", {}), "selected_step")) == 0
        and _deep_first(
            _mapping(value.get("null", {})), ("accepted", "panel_b_accepted")
        )
        is not None
        and not _one(
            _deep_first(
                _mapping(value.get("null", {})),
                ("accepted", "panel_b_accepted"),
            )
        )
        for value in normalized
    )
    null_pass = _null_family_valid(null_family, thresholds.pilot_null_family_size)
    checks = [
        _check("oracle_qualified_panels", int(_passed(panel_power)), "==", 1, _passed(panel_power)),
        _check("candidate_count", len(normalized), "==", len(thresholds.target_ratios), len(normalized) == len(thresholds.target_ratios)),
        _check("exact_ratio_grid", ratios, "==", sorted(thresholds.target_ratios), ratios == sorted(thresholds.target_ratios)),
        _check("unique_baseline", len(baselines), "==", 1, len(baselines) == 1),
        _check("all_optimizer_controllers_healthy", int(all_health), "==", 1, all_health),
        _check("panel_a_nominated_ratio", nomination.get("selected"), "==", 1, _one(nomination.get("selected"))),
        _check("sealed_single_use_panel_b", int(sealed), "==", 1, sealed),
        _check("baseline_panel_b", baseline_b.get("passed"), "==", 1, _passed(baseline_b)),
        _check("nominee_panel_b", b_gate.get("passed"), "==", 1, _passed(b_gate)),
        _check("all_nulls_select_analytic_zero", int(null_behavior), "==", 1, null_behavior),
        _check("simultaneous_null_b_family", int(null_pass), "==", 1, null_pass),
    ]
    result = _finish(
        "h1_gradient_control_pilot",
        checks,
        "fixed-step four-ratio pilot with sealed matched B and global null-B evidence",
        evaluation_status="evaluated" if normalized else "not_evaluated",
    )
    result.update(
        {
            "candidate_a_gates": a_gates,
            "nominated_profile": nomination,
            "selected_profile": nomination if _passed(b_gate) and sealed and null_pass else {**nomination, "selected": 0},
            "nominee_panel_b_gate": b_gate,
            "baseline_panel_b_gate": baseline_b,
            "optimizer_and_controller_health_pass": int(all_health),
            "overregularized": int(overregularized),
            "matched_effect_unconfirmed": int(
                _status(b_gate) == "evaluated"
                and _one(b_gate.get("classification_pass"))
                and _one(b_gate.get("derivative_pass"))
                and not _one(b_gate.get("matched_relative_l2_reduction_pass"))
            ),
            "selection_false_discovery": int(
                (
                    _status(b_gate) == "evaluated"
                    and not _one(b_gate.get("classification_pass"))
                )
                or not null_behavior
            ),
            "null_family": dict(null_family),
            "all_nulls_select_analytic_zero": int(null_behavior),
            "null_family_pass": int(null_pass),
            "null_positive_roles": sorted(_positive_roles(null_family)),
            "panel_power": dict(panel_power) if isinstance(panel_power, Mapping) else int(_passed(panel_power)),
        }
    )
    return result


evaluate_h1_gradient_control_pilot = evaluate_gradient_control_pilot


def _confirmation_tasks(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected = _task(record, "teacher")
    baseline = _task(record, "baseline")
    null = _task(record, "null")
    return selected, baseline, null


def evaluate_gradient_control_confirmation_seed(
    record: Mapping[str, Any],
    *,
    selected_ratio: float,
    thresholds: H1GradientControlThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or H1GradientControlThresholds()
    selected, baseline, null = _confirmation_tasks(record)
    selected_health, selected_health_record = _health_pass(selected, float(selected_ratio), thresholds)
    baseline_health, baseline_health_record = _health_pass(baseline, 0.0, thresholds)
    null_health, null_health_record = _health_pass(null, float(selected_ratio), thresholds)
    panel_gates: dict[str, Any] = {}
    for role in thresholds.confirmation_panel_roles:
        selected_panel = _normalize_panel(selected, role)
        baseline_panel = _normalize_panel(baseline, role)
        points, _ = _matched_reductions(
            selected_panel,
            baseline_panel,
            _matched_record(record, role),
        )
        classification = _classification_pass(selected_panel, thresholds)
        derivative = _derivative_pass(selected_panel, thresholds)
        point_pass = len(points) == 2 and all(
            _finite(value) and float(value) >= thresholds.minimum_relative_l2_reduction
            for value in points
        )
        panel_gates[role] = {
            "evaluation_status": _status(selected_panel),
            "passed": int(classification and derivative and point_pass),
            "classification_pass": int(classification),
            "derivative_pass": int(derivative),
            "matched_point_reductions": points,
            "matched_point_reduction_pass": int(point_pass),
            "selected_metrics": selected_panel,
            "baseline_metrics": baseline_panel,
        }
    healthy = selected_health and baseline_health and null_health
    science = all(_passed(value) for value in panel_gates.values())
    result = _finish(
        "h1_gradient_control_confirmation_seed",
        [
            _check("three_tasks_healthy", int(healthy), "==", 1, healthy),
            _check("all_bcd_science_panels", int(science), "==", 1, science),
        ],
        "fixed-step selected-ratio teacher versus matched rho-zero teacher",
        evaluation_status=_status(record),
    )
    result.update(
        {
            "seed": record.get("seed", record.get("model_seed")),
            "selected_ratio": float(selected_ratio),
            "optimizer_and_controller_health_pass": int(healthy),
            "selected_teacher_health": selected_health_record,
            "baseline_teacher_health": baseline_health_record,
            "null_health": null_health_record,
            "panel_gates": panel_gates,
            "classification_pass": int(
                all(_one(value.get("classification_pass")) for value in panel_gates.values())
            ),
            "derivative_pass": int(
                all(_one(value.get("derivative_pass")) for value in panel_gates.values())
            ),
            "matched_point_reduction_pass": int(
                all(_one(value.get("matched_point_reduction_pass")) for value in panel_gates.values())
            ),
        }
    )
    return result


def evaluate_gradient_control_confirmation(
    seed_records: Sequence[Mapping[str, Any]],
    *,
    selected_ratio: float,
    panel_power: bool | int | Mapping[str, Any],
    matched_effect_family: Mapping[str, Any],
    null_family: Mapping[str, Any],
    thresholds: H1GradientControlThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or H1GradientControlThresholds()
    seeds = [
        evaluate_gradient_control_confirmation_seed(
            value, selected_ratio=selected_ratio, thresholds=thresholds
        )
        for value in seed_records
    ]
    identifiers = [value.get("seed") for value in seeds]
    all_health = len(seeds) == thresholds.expected_confirmation_seeds and all(
        _one(value.get("optimizer_and_controller_health_pass")) for value in seeds
    )
    passing = sum(_passed(value) for value in seeds)
    matched_pass, matched_passing_seeds, matched_failed_roles = _matched_family_summary(
        matched_effect_family,
        expected_size=thresholds.confirmation_family_size,
        minimum_passing_seeds=thresholds.minimum_passing_confirmation_seeds,
    )
    null_pass = _null_family_valid(null_family, thresholds.confirmation_family_size)
    checks = [
        _check("oracle_qualified_panels", int(_passed(panel_power)), "==", 1, _passed(panel_power)),
        _check("seed_count", len(seeds), "==", thresholds.expected_confirmation_seeds, len(seeds) == thresholds.expected_confirmation_seeds),
        _check("unique_seed_count", len(set(map(str, identifiers))), "==", thresholds.expected_confirmation_seeds, len(set(map(str, identifiers))) == thresholds.expected_confirmation_seeds),
        _check("all_nine_tasks_healthy", int(all_health), "==", 1, all_health),
        _check("passing_teacher_seed_count", passing, ">=", thresholds.minimum_passing_confirmation_seeds, passing >= thresholds.minimum_passing_confirmation_seeds),
        _check("simultaneous_matched_effect_family", int(matched_pass), "==", 1, matched_pass),
        _check("simultaneous_stationary_null_family", int(null_pass), "==", 1, null_pass),
    ]
    result = _finish(
        "h1_gradient_control_confirmation",
        checks,
        "three-seed fixed-step derivative accuracy, matched H1 effect, and stationary-null evidence",
        evaluation_status="evaluated" if seeds else "not_evaluated",
    )
    result.update(
        {
            "seed_gates": seeds,
            "selected_ratio": float(selected_ratio),
            "optimizer_and_controller_health_pass": int(all_health),
            "passing_teacher_seed_count": int(passing),
            "classification_passing_seed_count": sum(
                _one(value.get("classification_pass")) for value in seeds
            ),
            "derivative_passing_seed_count": sum(
                _one(value.get("derivative_pass")) for value in seeds
            ),
            "matched_point_passing_seed_count": sum(
                _one(value.get("matched_point_reduction_pass")) for value in seeds
            ),
            "matched_effect_family": dict(matched_effect_family),
            "matched_effect_family_pass": int(matched_pass),
            "matched_effect_passing_seed_count": int(matched_passing_seeds),
            "matched_effect_failed_roles": sorted(matched_failed_roles),
            "null_family": dict(null_family),
            "null_family_pass": int(null_pass),
            "null_positive_roles": sorted(_positive_roles(null_family)),
            "panel_power": dict(panel_power) if isinstance(panel_power, Mapping) else int(_passed(panel_power)),
            "classification_failed_roles": sorted(
                {
                    role
                    for value in seeds
                    for role, gate in value.get("panel_gates", {}).items()
                    if not _one(gate.get("classification_pass"))
                }
            ),
            "derivative_failed_roles": sorted(
                {
                    role
                    for value in seeds
                    for role, gate in value.get("panel_gates", {}).items()
                    if not _one(gate.get("derivative_pass"))
                }
            ),
        }
    )
    return result


evaluate_h1_gradient_control_confirmation = evaluate_gradient_control_confirmation


def decide_gradient_control_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    controller_preflight: bool | int | Mapping[str, Any],
    pilot_panel_power: bool | int | Mapping[str, Any],
    pilot: Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    interim = False
    pilot = dict(pilot)
    confirmation = dict(confirmation)
    if not _passed(provenance):
        decision: H1GradientControlDecision | str = H1GradientControlDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the exact immutable H1-parent registry and transitive provenance"
    elif _status(controller_preflight) != "evaluated":
        decision = "gradient_controller_preflight_not_evaluated"
        action = "run the deterministic gradient-ratio controller preflight"
        interim = True
    elif not _passed(controller_preflight):
        decision = H1GradientControlDecision.H1_GRADIENT_CONTROLLER_INVALID
        action = "repair gradient composition, ratio tracking, fixed-point, or replay semantics"
    elif _status(pilot_panel_power) == "evaluated" and not _passed(pilot_panel_power):
        decision = H1GradientControlDecision.EVIDENCE_PANEL_UNDERPOWERED
        action = "stop before optimization and preserve the fixed underpowered panels"
    elif _status(pilot) != "evaluated":
        decision = "h1_gradient_controller_preflight_passed"
        action = "run the fixed-step four-ratio pilot"
        interim = True
    elif not _one(pilot.get("optimizer_and_controller_health_pass")):
        decision = H1GradientControlDecision.H1_CONTROLLER_OPTIMIZER_INVALID
        action = "repair controller tracking, task health, or clipping before scientific interpretation"
    elif _one(pilot.get("selection_false_discovery")) or "b" in set(pilot.get("null_positive_roles", [])) or (
        "null_family_pass" in pilot and not _one(pilot.get("null_family_pass"))
    ):
        decision = H1GradientControlDecision.SELECTION_FALSE_DISCOVERY
        action = "repair the global stationary-null panel-B family"
    elif not _passed(pilot):
        if _one(pilot.get("overregularized")):
            decision = H1GradientControlDecision.H1_CONTROLLER_OVERREGULARIZED
            action = "the controlled H1 grid suppresses teacher classification"
        elif _one(pilot.get("matched_effect_unconfirmed")):
            decision = H1GradientControlDecision.H1_CAUSAL_EFFECT_UNCONFIRMED
            action = "the sealed matched comparator did not confirm a ten-percent H1 effect"
        else:
            decision = H1GradientControlDecision.H1_STRENGTH_GRID_UNRESOLVED
            action = "the fixed ratio grid did not meet the unchanged derivative gate"
    elif _status(confirmation_panel_power) == "evaluated" and not _passed(confirmation_panel_power):
        decision = H1GradientControlDecision.EVIDENCE_PANEL_UNDERPOWERED
        action = "stop before confirmation and preserve the fixed underpowered panels"
    elif _status(confirmation) != "evaluated":
        decision = "h1_gradient_control_pilot_passed"
        action = "run fresh fixed-step three-seed confirmation"
        interim = True
    elif not _one(confirmation.get("optimizer_and_controller_health_pass")):
        decision = H1GradientControlDecision.H1_CONTROLLER_OPTIMIZER_INVALID
        action = "repair confirmation task or controller health"
    elif "b" in set(confirmation.get("null_positive_roles", [])):
        decision = H1GradientControlDecision.SELECTION_FALSE_DISCOVERY
        action = "the sealed stationary-null B family produced a false discovery"
    elif set(confirmation.get("null_positive_roles", [])) & {"c", "d"}:
        decision = H1GradientControlDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "stationary-null evidence failed on untouched audit panels"
    elif not _one(confirmation.get("null_family_pass", 1)):
        decision = H1GradientControlDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "the global stationary-null confirmation family was invalid or inconclusive"
    elif "b" in set(confirmation.get("classification_failed_roles", [])):
        decision = H1GradientControlDecision.SELECTION_FALSE_DISCOVERY
        action = "the selected teacher effect did not replicate on sealed panel B"
    elif set(confirmation.get("classification_failed_roles", [])) & {"c", "d"}:
        decision = H1GradientControlDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "the selected teacher classifier did not replicate on untouched audits"
    elif not _one(confirmation.get("matched_effect_family_pass")):
        failed_roles = set(confirmation.get("matched_effect_failed_roles", []))
        if failed_roles & {"c", "d"}:
            decision = H1GradientControlDecision.H1_EFFECT_AUDIT_INCONCLUSIVE
            action = "the matched H1 effect failed on untouched audit panels"
        else:
            decision = H1GradientControlDecision.H1_CAUSAL_EFFECT_UNCONFIRMED
            action = "the matched step-4000 effect was not simultaneously positive"
    elif _passed(confirmation):
        decision = H1GradientControlDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        action = "plan fresh physical one-image density-ratio score training"
    elif int(confirmation.get("classification_passing_seed_count", 0)) >= 2:
        decision = H1GradientControlDecision.H1_DENSITY_RATIO_VALUE_ONLY
        action = "stop tuning EMA-proximal H1; the classifier remains derivative-inaccurate"
    else:
        decision = H1GradientControlDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "the synthetic teacher did not replicate across sealed confirmation panels"
    value = decision.value if isinstance(decision, H1GradientControlDecision) else decision
    repaired = value == H1GradientControlDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value
    return {
        "decision": value,
        "recommended_next_action": action,
        "interim_stage_success": int(interim and value.endswith("_passed")),
        "closed_terminal_scientific_outcome": int(not interim),
        "physical_training_authorized": int(repaired),
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


decide_h1_gradient_control_workflow = decide_gradient_control_workflow


def evaluate_gradient_control_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    controller_preflight: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot_panel_power: bool | int | Mapping[str, Any],
    pilot: Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    confirmation: Mapping[str, Any],
    require_gate: str = "none",
    thresholds: H1GradientControlThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or H1GradientControlThresholds()
    if require_gate not in {"none", "preflight", "pilot", "controls"}:
        raise ValueError("require_gate must be none, preflight, pilot, or controls")
    required_pass = {
        "none": True,
        "preflight": _passed(provenance) and _passed(preflight),
        "pilot": (
            _passed(provenance)
            and _passed(preflight)
            and _passed(pilot_panel_power)
            and _passed(pilot)
        ),
        "controls": (
            _passed(provenance)
            and _passed(preflight)
            and _passed(pilot_panel_power)
            and _passed(pilot)
            and _passed(confirmation_panel_power)
            and _passed(confirmation)
        ),
    }[require_gate]
    decision = decide_gradient_control_workflow(
        provenance=provenance,
        controller_preflight=controller_preflight,
        pilot_panel_power=pilot_panel_power,
        pilot=pilot,
        confirmation_panel_power=confirmation_panel_power,
        confirmation=confirmation,
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "components": {
            "provenance": dict(provenance) if isinstance(provenance, Mapping) else int(_passed(provenance)),
            "controller_preflight": dict(controller_preflight) if isinstance(controller_preflight, Mapping) else int(_passed(controller_preflight)),
            "preflight": dict(preflight) if isinstance(preflight, Mapping) else int(_passed(preflight)),
            "pilot_panel_power": dict(pilot_panel_power) if isinstance(pilot_panel_power, Mapping) else int(_passed(pilot_panel_power)),
            "pilot": dict(pilot),
            "confirmation_panel_power": dict(confirmation_panel_power) if isinstance(confirmation_panel_power, Mapping) else int(_passed(confirmation_panel_power)),
            "confirmation": dict(confirmation),
        },
        "decision": decision,
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "thresholds": thresholds.to_dict(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


evaluate_h1_gradient_control_workflow = evaluate_gradient_control_workflow
