"""Pure gates for the D0 density-ratio H1 function-step experiment.

This module is deliberately additive.  It does not import a sampler or any
training orchestration.  Its job is to turn already-computed operator,
calibration, pilot, and confirmation records into fail-closed decisions.

The pilot compares four fixed H1 multipliers.  The zero-multiplier arm is the
frozen density-ratio baseline; a nonzero arm is eligible only when it satisfies
the full sealed-panel-B derivative thresholds *and* reduces relative flux L2 by
at least ten percent, both overall and at the data end.  This is intentionally
stricter than merely improving the classification objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_score_density_ratio_selection_power_gate import (
    SelectionPowerThresholds,
)


__all__ = [
    "H1TrustDecision",
    "H1TrustThresholds",
    "not_evaluated_gate",
    "normalize_h1_candidate",
    "evaluate_h1_operator_preflight",
    "evaluate_h1_calibration",
    "evaluate_h1_preflight",
    "evaluate_h1_pilot_candidate",
    "rank_h1_pilot_candidates",
    "evaluate_h1_pilot",
    "evaluate_h1_controls",
    "decide_h1_workflow",
    "evaluate_h1_workflow",
]


SCHEMA = "experiment12-d0-score-density-ratio-h1-trust-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class H1TrustThresholds:
    """Frozen candidate grid and inherited scientific thresholds."""

    selection_power: SelectionPowerThresholds = field(
        default_factory=SelectionPowerThresholds
    )
    multipliers: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0)
    minimum_relative_l2_reduction: float = 0.10
    body_learning_rate: float = 3e-5
    accumulation_steps: int = 8
    base_channels: int = 32

    def __post_init__(self) -> None:
        if self.selection_power != SelectionPowerThresholds():
            raise ValueError("inherited density-ratio thresholds must remain frozen")
        if tuple(float(value) for value in self.multipliers) != (0.0, 0.1, 0.3, 1.0):
            raise ValueError("H1 multipliers are frozen at 0, 0.1, 0.3, and 1.0")
        if float(self.minimum_relative_l2_reduction) != 0.10:
            raise ValueError("minimum relative-L2 reduction is frozen at 0.10")
        if float(self.body_learning_rate) != 3e-5:
            raise ValueError("body learning rate is frozen at 3e-5")
        if int(self.accumulation_steps) != 8:
            raise ValueError("gradient accumulation is frozen at eight")
        if int(self.base_channels) != 32:
            raise ValueError("model width is frozen at 32")

    @property
    def teacher(self) -> Any:
        return self.selection_power.head.stability.density_ratio.teacher

    @property
    def maximum_clip_fraction(self) -> float:
        return float(
            self.selection_power.head.stability.density_ratio.maximum_clip_fraction
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class H1TrustDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    H1_OPERATOR_INVALID = "h1_operator_invalid"
    H1_CALIBRATION_INVALID = "h1_calibration_invalid"
    EVIDENCE_PANEL_UNDERPOWERED = "evidence_panel_underpowered"
    H1_OPTIMIZER_INVALID = "h1_optimizer_invalid"
    H1_OVERREGULARIZED = "h1_overregularized"
    H1_FUNCTION_STEP_UNRESOLVED = "h1_function_step_unresolved"
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    CLASSIFICATION_AUDIT_INCONCLUSIVE = "classification_audit_inconclusive"
    H1_DENSITY_RATIO_VALUE_ONLY = "h1_density_ratio_value_only"
    DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED = (
        "density_ratio_control_pipeline_repaired"
    )


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
    value = _finish(gate, [], "not evaluated", evaluation_status="not_evaluated")
    value["reason"] = str(reason)
    return value


def _deep_first(value: Mapping[str, Any], names: Sequence[str]) -> Any:
    """Find a named scalar/array while preferring shallower summaries.

    Orchestrators may supply a compact flat candidate or the existing nested
    ``teacher/null -> metrics -> selection`` task result.  Supporting both here
    keeps the scientific checks in one place without coupling this pure module
    to checkpoint or filesystem schemas.
    """

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


def _bounds(value: Mapping[str, Any], *names: str) -> list[float]:
    raw = _deep_first(value, names)
    if isinstance(raw, Mapping):
        raw = [raw.get("overall"), raw.get("data_end")]
    return _array(raw)


def normalize_h1_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize either a compact candidate or nested teacher/null task pair."""

    raw = dict(candidate)
    teacher = _mapping(raw.get("teacher", raw.get("bounded_teacher", {})))
    null = _mapping(raw.get("null", raw.get("dirichlet_null", {})))
    teacher_metrics = _mapping(teacher.get("metrics", teacher))
    null_metrics = _mapping(null.get("metrics", null))

    def first(*names: str, default: Any = None) -> Any:
        value = _deep_first(raw, names)
        return default if value is None else value

    def teacher_first(*names: str, default: Any = None) -> Any:
        value = _deep_first(teacher_metrics, names)
        if value is None:
            value = _deep_first(raw, names)
        return default if value is None else value

    def null_first(*names: str, default: Any = None) -> Any:
        value = _deep_first(null_metrics, names)
        if value is None:
            value = _deep_first(raw, names)
        return default if value is None else value

    teacher_selection = _mapping(teacher_metrics.get("selection", {}))
    teacher_confirmation = _mapping(teacher_selection.get("confirmation", {}))
    null_selection = _mapping(null_metrics.get("selection", {}))
    null_confirmation = _mapping(null_selection.get("confirmation", {}))

    teacher_bounds = _array(
        teacher_confirmation.get(
            "panel_b_lower_bounds",
            teacher_first("teacher_panel_b_lower_bounds", default=[]),
        )
    )
    null_bounds = _array(
        null_confirmation.get(
            "panel_b_lower_bounds",
            null_first("null_panel_b_lower_bounds", default=[]),
        )
    )
    bin_cosines = _array(
        teacher_first(
            "teacher_time_bin_flux_cosines",
            "time_bin_flux_cosines",
            "flux_cosine_by_time_bin",
            default=[],
        )
    )
    bin_l2 = _array(
        teacher_first(
            "teacher_time_bin_relative_flux_l2",
            "time_bin_relative_flux_l2",
            "relative_flux_l2_by_time_bin",
            default=[],
        )
    )

    normalized = {
        "evaluation_status": str(first("evaluation_status", default="evaluated")),
        "multiplier": first("multiplier", "h1_multiplier", "lambda_multiplier"),
        "learning_rate": first("learning_rate", "body_learning_rate"),
        "accumulation_steps": first("accumulation_steps", "accumulation_level"),
        "base_channels": first("base_channels", default=32),
        "complete": first("complete"),
        "finite": first("finite"),
        "boundary_admissible": first("boundary_admissible"),
        "optimizer_health_pass": first("optimizer_health_pass"),
        "h1_health_pass": first("h1_health_pass", "penalty_health_pass"),
        "maximum_clip_fraction_observed": first(
            "maximum_clip_fraction_observed", "max_clip_fraction"
        ),
        "teacher_complete": teacher_first("complete"),
        "teacher_finite": teacher_first("finite"),
        "teacher_boundary_admissible": teacher_first("boundary_admissible"),
        "teacher_selected_step": teacher_selection.get(
            "selected_step",
            teacher_first("teacher_selected_step", "selected_step", default=0),
        ),
        "teacher_panel_b_confirmed": teacher_confirmation.get(
            "accepted",
            teacher_first("teacher_panel_b_confirmed", default=0),
        ),
        "teacher_panel_b_lower_bounds": teacher_bounds,
        "teacher_panel_b_bce": teacher_confirmation.get(
            "panel_b_overall_bce",
            teacher_first("teacher_panel_b_bce", "panel_b_overall_bce"),
        ),
        "teacher_score_gain_overall": teacher_first(
            "teacher_score_gain_overall", "score_gain_overall", "overall_score_gain"
        ),
        "teacher_score_gain_data_end": teacher_first(
            "teacher_score_gain_data_end", "score_gain_data_end", "data_end_score_gain"
        ),
        "teacher_flux_cosine_overall": teacher_first(
            "teacher_flux_cosine_overall", "overall_flux_cosine", "flux_cosine_overall"
        ),
        "teacher_time_bin_flux_cosines": bin_cosines,
        "teacher_relative_flux_l2_overall": teacher_first(
            "teacher_relative_flux_l2_overall",
            "overall_relative_flux_l2",
            "relative_flux_l2_overall",
        ),
        "teacher_relative_flux_l2_data_end": teacher_first(
            "teacher_relative_flux_l2_data_end",
            "data_end_relative_flux_l2",
            "relative_flux_l2_data_end",
        ),
        "teacher_time_bin_relative_flux_l2": bin_l2,
        "null_complete": null_first("complete"),
        "null_finite": null_first("finite"),
        "null_boundary_admissible": null_first("boundary_admissible"),
        "null_optimizer_health_pass": null_first("optimizer_health_pass"),
        "null_selected_step": null_selection.get(
            "selected_step", null_first("null_selected_step", default=0)
        ),
        "null_panel_b_rejected": int(
            not _one(null_confirmation.get("accepted", 0))
            if null_confirmation
            else _one(null_first("null_panel_b_rejected", default=0))
        ),
        "null_panel_b_lower_bounds": null_bounds,
    }
    # Explicit compact summaries take precedence over recursively discovered
    # values.  This is useful when task records contain metrics from multiple
    # checkpoints but the orchestration has already frozen the selected one.
    for key in tuple(normalized):
        if key in raw:
            normalized[key] = raw[key]
    return normalized


def evaluate_h1_operator_preflight(record: Mapping[str, Any]) -> dict[str, Any]:
    """Gate exact discrete Gamma/H1 implementation and stopped-anchor rules."""

    required = (
        "gamma_symmetry_pass",
        "gamma_positivity_pass",
        "orientation_invariance_pass",
        "analytic_agreement_pass",
        "identical_anchor_zero_pass",
        "identical_anchor_gradient_zero_pass",
        "constant_shift_l2_detection_pass",
        "stopped_anchor_pass",
        "boundary_finite_pass",
        "cuda_second_order_backward_pass",
        "lambda_zero_regression_pass",
        "stateless_stream_replay_pass",
        "candidate_order_invariance_pass",
        "teacher_null_namespace_isolation_pass",
    )
    checks = [
        _check("complete", record.get("complete"), "==", 1, _one(record.get("complete"))),
        _check("finite", record.get("finite"), "==", 1, _one(record.get("finite"))),
        *[
            _check(name, record.get(name), "==", 1, _one(record.get(name)))
            for name in required
        ],
    ]
    return _finish(
        "h1_operator_preflight",
        checks,
        "implementation validity of the stopped-anchor discrete carré-du-champ penalty",
        evaluation_status=_status(record),
    )


def evaluate_h1_calibration(record: Mapping[str, Any]) -> dict[str, Any]:
    """Gate train-only, deterministic, shared teacher/null H1 calibration."""

    positive = ("value_scale", "energy_scale", "lambda_base")
    checks = [
        _check("complete", record.get("complete"), "==", 1, _one(record.get("complete"))),
        _check("finite", record.get("finite"), "==", 1, _one(record.get("finite"))),
        *[
            _check(
                name,
                record.get(name),
                ">",
                0.0,
                _finite(record.get(name)) and float(record[name]) > 0.0,
            )
            for name in positive
        ],
        _check(
            "training_only",
            record.get("training_only"),
            "==",
            1,
            _one(record.get("training_only")),
        ),
        _check(
            "evidence_overlap_path_count",
            record.get("evidence_overlap_path_count"),
            "==",
            0,
            _zero(record.get("evidence_overlap_path_count")),
        ),
        _check(
            "shared_teacher_null",
            record.get("shared_teacher_null"),
            "==",
            1,
            _one(record.get("shared_teacher_null")),
        ),
        _check(
            "deterministic_replay_pass",
            record.get("deterministic_replay_pass"),
            "==",
            1,
            _one(record.get("deterministic_replay_pass")),
        ),
    ]
    return _finish(
        "h1_calibration",
        checks,
        "frozen train-only H1 units shared by teacher and stationary-null arms",
        evaluation_status=_status(record),
    )


def evaluate_h1_preflight(
    *,
    inherited_preflight: bool | int | Mapping[str, Any],
    operator: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    operator_gate = (
        dict(operator)
        if operator.get("gate") == "h1_operator_preflight"
        else evaluate_h1_operator_preflight(operator)
    )
    calibration_gate = (
        dict(calibration)
        if calibration.get("gate") == "h1_calibration"
        else evaluate_h1_calibration(calibration)
    )
    checks = [
        _check(
            "inherited_preflight",
            int(_passed(inherited_preflight)),
            "==",
            1,
            _passed(inherited_preflight),
        ),
        _check("h1_operator", operator_gate["passed"], "==", 1, _passed(operator_gate)),
        _check(
            "h1_calibration",
            calibration_gate["passed"],
            "==",
            1,
            _passed(calibration_gate),
        ),
    ]
    result = _finish(
        "h1_trust_preflight",
        checks,
        "inherited density-ratio validity plus H1 operator and scale calibration",
        evaluation_status=(
            "evaluated"
            if _status(operator_gate) == _status(calibration_gate) == "evaluated"
            else "not_evaluated"
        ),
    )
    result.update(
        {
            "inherited_preflight": dict(inherited_preflight)
            if isinstance(inherited_preflight, Mapping)
            else int(_passed(inherited_preflight)),
            "operator_gate": operator_gate,
            "calibration_gate": calibration_gate,
        }
    )
    return result


def _healthy(normalized: Mapping[str, Any], maximum_clip: float) -> bool:
    flags = (
        "complete",
        "finite",
        "boundary_admissible",
        "optimizer_health_pass",
        "h1_health_pass",
        "teacher_complete",
        "teacher_finite",
        "teacher_boundary_admissible",
        "null_complete",
        "null_finite",
        "null_boundary_admissible",
        "null_optimizer_health_pass",
    )
    return (
        all(_one(normalized.get(name)) for name in flags)
        and _finite(normalized.get("maximum_clip_fraction_observed"))
        and float(normalized["maximum_clip_fraction_observed"]) <= maximum_clip
    )


def evaluate_h1_pilot_candidate(
    candidate: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None = None,
    thresholds: H1TrustThresholds | None = None,
) -> dict[str, Any]:
    """Gate one H1 multiplier; ``baseline`` is required for nonzero arms."""

    thresholds = thresholds or H1TrustThresholds()
    value = normalize_h1_candidate(candidate)
    multiplier = value.get("multiplier")
    is_zero = _finite(multiplier) and float(multiplier) == 0.0
    teacher = thresholds.teacher
    bins_cos = _array(value.get("teacher_time_bin_flux_cosines"))
    bins_l2 = _array(value.get("teacher_time_bin_relative_flux_l2"))
    overall_l2 = value.get("teacher_relative_flux_l2_overall")
    end_l2 = value.get("teacher_relative_flux_l2_data_end")
    baseline_value = normalize_h1_candidate(baseline or {})
    base_overall = baseline_value.get("teacher_relative_flux_l2_overall")
    base_end = baseline_value.get("teacher_relative_flux_l2_data_end")
    reductions = [None, None]
    if (
        not is_zero
        and _finite(overall_l2)
        and _finite(end_l2)
        and _finite(base_overall)
        and _finite(base_end)
        and float(base_overall) > 0.0
        and float(base_end) > 0.0
    ):
        reductions = [
            1.0 - float(overall_l2) / float(base_overall),
            1.0 - float(end_l2) / float(base_end),
        ]

    teacher_bounds = _array(value.get("teacher_panel_b_lower_bounds"))
    null_bounds = _array(value.get("null_panel_b_lower_bounds"))
    health = _healthy(value, thresholds.maximum_clip_fraction)
    classification = (
        int(value.get("teacher_selected_step", 0)) > 0
        and _one(value.get("teacher_panel_b_confirmed"))
        and len(teacher_bounds) == 2
        and all(bound > 0.0 for bound in teacher_bounds)
        and _finite(value.get("teacher_score_gain_overall"))
        and float(value["teacher_score_gain_overall"])
        >= float(teacher.teacher_min_score_gain)
        and _finite(value.get("teacher_score_gain_data_end"))
        and float(value["teacher_score_gain_data_end"])
        >= float(teacher.teacher_min_score_gain)
    )
    null_valid = (
        int(value.get("null_selected_step", -1)) == 0
        and _one(value.get("null_panel_b_rejected"))
        and len(null_bounds) == 2
        and all(_finite(bound) for bound in null_bounds)
    )
    derivative = (
        _finite(value.get("teacher_flux_cosine_overall"))
        and float(value["teacher_flux_cosine_overall"])
        >= float(teacher.teacher_min_overall_flux_cosine)
        and len(bins_cos) == int(teacher.expected_time_bins)
        and all(value >= float(teacher.teacher_min_bin_flux_cosine) for value in bins_cos)
        and _finite(overall_l2)
        and float(overall_l2) <= float(teacher.teacher_max_overall_relative_flux_l2)
        and len(bins_l2) == int(teacher.expected_time_bins)
        and all(value <= float(teacher.teacher_max_bin_relative_flux_l2) for value in bins_l2)
    )
    reduction = (
        len(reductions) == 2
        and all(
            _finite(item)
            and float(item) >= thresholds.minimum_relative_l2_reduction
            for item in reductions
        )
    )

    checks = [
        _check("known_multiplier", multiplier, "in", list(thresholds.multipliers), _finite(multiplier) and float(multiplier) in thresholds.multipliers),
        _check("learning_rate", value.get("learning_rate"), "==", thresholds.body_learning_rate, _finite(value.get("learning_rate")) and float(value["learning_rate"]) == thresholds.body_learning_rate),
        _check("accumulation_steps", value.get("accumulation_steps"), "==", thresholds.accumulation_steps, value.get("accumulation_steps") is not None and int(value["accumulation_steps"]) == thresholds.accumulation_steps),
        _check("base_channels", value.get("base_channels"), "==", thresholds.base_channels, value.get("base_channels") is not None and int(value["base_channels"]) == thresholds.base_channels),
        _check("optimizer_and_task_health", int(health), "==", 1, health),
        _check("teacher_classification", int(classification), "==", 1, classification),
        _check("stationary_null", int(null_valid), "==", 1, null_valid),
    ]
    if not is_zero:
        checks.extend(
            [
                _check("strict_derivative_thresholds", int(derivative), "==", 1, derivative),
                _check(
                    "relative_l2_reduction_overall_and_data_end",
                    reductions,
                    ">= each",
                    thresholds.minimum_relative_l2_reduction,
                    reduction,
                ),
            ]
        )
    result = _finish(
        "h1_pilot_candidate",
        checks,
        "sealed-B classification, stationary-null, and function-step derivative evidence",
        evaluation_status=_status(value),
    )
    max_l2 = None
    if _finite(overall_l2) and bins_l2:
        max_l2 = max(float(overall_l2), *bins_l2)
    min_cos = None
    if _finite(value.get("teacher_flux_cosine_overall")) and bins_cos:
        min_cos = min(float(value["teacher_flux_cosine_overall"]), *bins_cos)
    result.update(
        {
            **value,
            "is_baseline": int(is_zero),
            "optimizer_health_pass": int(health),
            "classification_pass": int(classification),
            "null_pass": int(null_valid),
            "derivative_pass": int(derivative),
            "relative_l2_reductions": reductions,
            "relative_l2_reduction_pass": int(reduction),
            "ranking_max_relative_flux_l2": max_l2,
            "ranking_min_flux_cosine": min_cos,
        }
    )
    return result


def rank_h1_pilot_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select an eligible nonzero arm with the predeclared lexicographic rule."""

    eligible = [
        (index, dict(value))
        for index, value in enumerate(candidates)
        if _passed(value) and not _one(value.get("is_baseline", 0))
    ]

    def key(item: tuple[int, Mapping[str, Any]]) -> tuple[float, float, float, float, int]:
        index, value = item
        max_l2 = value.get("ranking_max_relative_flux_l2")
        min_cos = value.get("ranking_min_flux_cosine")
        bce = value.get("teacher_panel_b_bce")
        multiplier = value.get("multiplier")
        return (
            float(max_l2) if _finite(max_l2) else math.inf,
            -float(min_cos) if _finite(min_cos) else math.inf,
            float(bce) if _finite(bce) else math.inf,
            float(multiplier) if _finite(multiplier) else math.inf,
            index,
        )

    ranked = sorted(eligible, key=key)
    if not ranked:
        return {
            "selected": 0,
            "selected_candidate_index": None,
            "selected_multiplier": None,
            "ranking_rule": [
                "minimum maximum overall/per-bin relative flux L2",
                "maximum minimum-bin flux cosine",
                "minimum sealed-panel-B BCE",
                "smaller H1 multiplier",
            ],
            "eligible_candidate_indices": [],
        }
    index, value = ranked[0]
    return {
        "selected": 1,
        "selected_candidate_index": index,
        "selected_multiplier": float(value["multiplier"]),
        "learning_rate": float(value["learning_rate"]),
        "accumulation_steps": int(value["accumulation_steps"]),
        "ranking_max_relative_flux_l2": float(value["ranking_max_relative_flux_l2"]),
        "ranking_min_flux_cosine": float(value["ranking_min_flux_cosine"]),
        "teacher_panel_b_bce": float(value["teacher_panel_b_bce"]),
        "ranking_rule": [
            "minimum maximum overall/per-bin relative flux L2",
            "maximum minimum-bin flux cosine",
            "minimum sealed-panel-B BCE",
            "smaller H1 multiplier",
        ],
        "eligible_candidate_indices": [index for index, _ in ranked],
    }


def evaluate_h1_pilot(
    candidates: Sequence[Mapping[str, Any]],
    *,
    panel_power: bool | int | Mapping[str, Any],
    null_family: bool | int | Mapping[str, Any],
    thresholds: H1TrustThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or H1TrustThresholds()
    normalized = [normalize_h1_candidate(value) for value in candidates]
    baseline_rows = [
        value
        for value in normalized
        if _finite(value.get("multiplier")) and float(value["multiplier"]) == 0.0
    ]
    baseline = baseline_rows[0] if len(baseline_rows) == 1 else None
    gates = [
        evaluate_h1_pilot_candidate(
            value,
            baseline=baseline if float(value.get("multiplier", math.nan)) != 0.0 else None,
            thresholds=thresholds,
        )
        for value in normalized
    ]
    multipliers = sorted(
        float(value.get("multiplier"))
        for value in normalized
        if _finite(value.get("multiplier"))
    )
    profile = rank_h1_pilot_candidates(gates)
    all_healthy = len(gates) == len(thresholds.multipliers) and all(
        _one(value.get("optimizer_health_pass")) for value in gates
    )
    nonzero = [value for value in gates if not _one(value.get("is_baseline"))]
    overregularized = bool(nonzero) and all(
        not _one(value.get("classification_pass")) for value in nonzero
    )
    checks = [
        _check("oracle_qualified_panels", int(_passed(panel_power)), "==", 1, _passed(panel_power)),
        _check("candidate_count", len(gates), "==", len(thresholds.multipliers), len(gates) == len(thresholds.multipliers)),
        _check("exact_multiplier_set", multipliers, "==", sorted(thresholds.multipliers), multipliers == sorted(thresholds.multipliers)),
        _check("unique_baseline", len(baseline_rows), "==", 1, len(baseline_rows) == 1),
        _check("all_optimizers_healthy", int(all_healthy), "==", 1, all_healthy),
        _check("simultaneous_null_family", int(_passed(null_family)), "==", 1, _passed(null_family)),
        _check("eligible_nonzero_profile", profile["selected"], "==", 1, _one(profile["selected"])),
    ]
    result = _finish(
        "h1_function_step_pilot",
        checks,
        "four-arm H1 multiplier pilot with sealed B and simultaneous null evidence",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result.update(
        {
            "candidate_gates": gates,
            "selected_profile": profile,
            "optimizer_health_pass": int(all_healthy),
            "overregularized": int(overregularized),
            "null_family": dict(null_family)
            if isinstance(null_family, Mapping)
            else int(_passed(null_family)),
            "null_family_pass": int(_passed(null_family)),
            "oracle_panel_power": dict(panel_power)
            if isinstance(panel_power, Mapping)
            else int(_passed(panel_power)),
        }
    )
    return result


def evaluate_h1_controls(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    teacher_study: Mapping[str, Any],
    null_family: Mapping[str, Any],
) -> dict[str, Any]:
    teacher = dict(teacher_study)
    null = dict(null_family)
    optimizer = _one(teacher.get("optimizer_health_pass")) and _one(
        null.get("optimizer_health_pass")
    )
    checks = [
        _check("provenance", int(_passed(provenance)), "==", 1, _passed(provenance)),
        _check("preflight", int(_passed(preflight)), "==", 1, _passed(preflight)),
        _check("pilot", int(_passed(pilot)), "==", 1, _passed(pilot)),
        _check("oracle_qualified_confirmation_panels", int(_passed(confirmation_panel_power)), "==", 1, _passed(confirmation_panel_power)),
        _check("all_confirmation_optimizers_healthy", int(optimizer), "==", 1, optimizer),
        _check("teacher_study", teacher.get("passed"), "==", 1, _passed(teacher)),
        _check("simultaneous_null_family", null.get("passed"), "==", 1, _passed(null)),
    ]
    result = _finish(
        "h1_density_ratio_controls",
        checks,
        "strict derivative-accurate H1 density-ratio controls with simultaneous null evidence",
        evaluation_status=(
            "evaluated"
            if _status(teacher) == _status(null) == "evaluated"
            else "not_evaluated"
        ),
    )
    result.update(
        {
            "teacher_study": teacher,
            "null_family": null,
            "optimizer_health_pass": int(optimizer),
            "confirmation_panel_power": dict(confirmation_panel_power)
            if isinstance(confirmation_panel_power, Mapping)
            else int(_passed(confirmation_panel_power)),
        }
    )
    return result


def _false_discovery(null: Mapping[str, Any]) -> bool:
    return any(
        _one(null.get(name))
        for name in (
            "familywise_false_discovery",
            "selection_false_discovery",
            "false_discovery",
        )
    ) or int(null.get("false_discovery_count", 0)) > 0


def decide_h1_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    operator: bool | int | Mapping[str, Any],
    calibration: bool | int | Mapping[str, Any],
    pilot_panel_power: bool | int | Mapping[str, Any],
    pilot: Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    teacher = _mapping(controls.get("teacher_study", {}))
    null = _mapping(controls.get("null_family", {}))
    interim = False
    if not _passed(provenance):
        decision: H1TrustDecision | str = H1TrustDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the exact 263-record parent and 123-to-125-to-332-to-222-to-381 binding"
    elif _status(operator) != "evaluated":
        decision = "h1_preflight_not_evaluated"
        action = "run the exact discrete H1 operator preflight"
        interim = True
    elif not _passed(operator):
        decision = H1TrustDecision.H1_OPERATOR_INVALID
        action = "repair the discrete Gamma operator, stopped anchors, or lambda-zero regression"
    elif _status(calibration) != "evaluated":
        decision = "h1_operator_preflight_passed"
        action = "calibrate frozen train-only value and energy units"
        interim = True
    elif not _passed(calibration):
        decision = H1TrustDecision.H1_CALIBRATION_INVALID
        action = "repair train-only H1 scale calibration before optimization"
    elif _status(pilot_panel_power) != "evaluated":
        decision = "h1_preflight_passed"
        action = "freeze and oracle-qualify the pilot evidence panels"
        interim = True
    elif not _passed(pilot_panel_power):
        decision = H1TrustDecision.EVIDENCE_PANEL_UNDERPOWERED
        action = "stop before optimization and preserve the fixed underpowered panels"
    elif _status(pilot) != "evaluated":
        decision = "h1_preflight_passed"
        action = "run the four-arm H1 function-step pilot"
        interim = True
    elif _false_discovery(_mapping(pilot.get("null_family", {}))) or (
        "null_family_pass" in pilot and not _one(pilot.get("null_family_pass"))
    ):
        decision = H1TrustDecision.SELECTION_FALSE_DISCOVERY
        action = "repair stationary-null simultaneous calibration before derivative learning"
    elif not _one(pilot.get("optimizer_health_pass")):
        decision = H1TrustDecision.H1_OPTIMIZER_INVALID
        action = "repair incomplete, nonfinite, clipped, or penalty-unstable pilot tasks"
    elif not _passed(pilot):
        if _one(pilot.get("overregularized")):
            decision = H1TrustDecision.H1_OVERREGULARIZED
            action = "the fixed H1 grid suppresses classification; revisit the penalty coordinate"
        else:
            decision = H1TrustDecision.H1_FUNCTION_STEP_UNRESOLVED
            action = "revisit the H1 function-step constraint without weakening derivative thresholds"
    elif _status(confirmation_panel_power) == "evaluated" and not _passed(
        confirmation_panel_power
    ):
        decision = H1TrustDecision.EVIDENCE_PANEL_UNDERPOWERED
        action = "stop before confirmation and preserve the fixed underpowered panels"
    elif _status(controls) != "evaluated":
        decision = "h1_pilot_passed"
        action = "run fresh three-seed H1 density-ratio confirmation"
        interim = True
    elif not _one(controls.get("optimizer_health_pass")):
        decision = H1TrustDecision.H1_OPTIMIZER_INVALID
        action = "repair confirmation optimizer or H1-health failures"
    elif _false_discovery(null):
        decision = H1TrustDecision.SELECTION_FALSE_DISCOVERY
        action = "repair stationary-null selection or familywise evidence"
    elif _one(teacher.get("panel_disagreement")) or _one(
        null.get("panel_disagreement")
    ) or _one(null.get("audit_inconclusive")):
        decision = H1TrustDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "rerun the frozen confirmation because sealed audit panels disagree"
    elif _passed(controls):
        decision = H1TrustDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        action = "plan fresh physical one-image density-ratio score training"
    elif int(teacher.get("classification_passing_seed_count", 0)) >= 2:
        decision = H1TrustDecision.H1_DENSITY_RATIO_VALUE_ONLY
        action = "the H1-constrained classifier remains value-only; revisit score-target theory"
    else:
        decision = H1TrustDecision.H1_FUNCTION_STEP_UNRESOLVED
        action = "revisit the H1 function-step constraint or density-ratio control objective"
    value = decision.value if isinstance(decision, H1TrustDecision) else decision
    repaired = value == H1TrustDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value
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


def evaluate_h1_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    operator: bool | int | Mapping[str, Any],
    calibration: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot_panel_power: bool | int | Mapping[str, Any],
    pilot: Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    teacher_study: Mapping[str, Any],
    null_family: Mapping[str, Any],
    require_gate: str = "none",
    thresholds: H1TrustThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or H1TrustThresholds()
    if require_gate not in {"none", "preflight", "pilot", "controls"}:
        raise ValueError("require_gate must be none, preflight, pilot, or controls")
    controls = evaluate_h1_controls(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        confirmation_panel_power=confirmation_panel_power,
        teacher_study=teacher_study,
        null_family=null_family,
    )
    required_pass = {
        "none": True,
        "preflight": _passed(provenance) and _passed(preflight),
        "pilot": (
            _passed(provenance)
            and _passed(preflight)
            and _passed(pilot_panel_power)
            and _passed(pilot)
        ),
        "controls": _passed(controls),
    }[require_gate]
    decision = decide_h1_workflow(
        provenance=provenance,
        operator=operator,
        calibration=calibration,
        pilot_panel_power=pilot_panel_power,
        pilot=pilot,
        confirmation_panel_power=confirmation_panel_power,
        controls=controls,
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "components": {
            "provenance": dict(provenance)
            if isinstance(provenance, Mapping)
            else int(_passed(provenance)),
            "operator": dict(operator)
            if isinstance(operator, Mapping)
            else int(_passed(operator)),
            "calibration": dict(calibration)
            if isinstance(calibration, Mapping)
            else int(_passed(calibration)),
            "preflight": dict(preflight)
            if isinstance(preflight, Mapping)
            else int(_passed(preflight)),
            "pilot_panel_power": dict(pilot_panel_power)
            if isinstance(pilot_panel_power, Mapping)
            else int(_passed(pilot_panel_power)),
            "pilot": dict(pilot),
            "confirmation_panel_power": dict(confirmation_panel_power)
            if isinstance(confirmation_panel_power, Mapping)
            else int(_passed(confirmation_panel_power)),
            "controls": controls,
        },
        "decision": decision,
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "thresholds": thresholds.to_dict(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
