"""Pure gates for paired-mixture density-ratio stability controls.

This module is intentionally free of training and filesystem code.  It wraps
the frozen density-ratio scientific gates with the additional optimizer-health
contract needed by the variance-reduced workflow:

* pilot accumulation levels are tried hierarchically in the order 2, 4, 8;
* each level contains exactly the two frozen learning rates 3e-5 and 1e-5;
* clipping must be at most ten percent after warmup, over the final 500
  updates, and over the final 200 updates;
* confirmation retains the original strict classification, score, flux, and
  null thresholds without modification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_score_density_ratio_gate import (
    DensityRatioThresholds,
    evaluate_density_ratio_pilot_candidate,
    evaluate_null_seed as evaluate_base_null_seed,
    evaluate_teacher_seed as evaluate_base_teacher_seed,
)


__all__ = [
    "RatioStabilityDecision",
    "RatioStabilityThresholds",
    "not_evaluated_gate",
    "evaluate_paired_ratio_preflight",
    "evaluate_stability_pilot_candidate",
    "select_stability_profile",
    "evaluate_stability_pilot_level",
    "evaluate_stability_pilot",
    "evaluate_teacher_seed",
    "evaluate_teacher_study",
    "evaluate_null_seed",
    "evaluate_null_study",
    "evaluate_ratio_stability_controls",
    "decide_ratio_stability",
    "evaluate_ratio_stability_workflow",
]


SCHEMA = "experiment12-d0-score-density-ratio-stability-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RatioStabilityThresholds:
    """Frozen thresholds for the additive paired-estimator workflow."""

    density_ratio: DensityRatioThresholds = field(default_factory=DensityRatioThresholds)
    accumulation_levels: tuple[int, ...] = (2, 4, 8)
    pilot_learning_rates: tuple[float, ...] = (3e-5, 1e-5)
    maximum_clip_fraction: float = 0.10
    final_clip_windows: tuple[int, ...] = (500, 200)
    preflight_confidence: float = 0.99
    preflight_paths: int = 128
    loss_algebra_tolerance: float = 1e-12
    expanded_loss_tolerance: float = 1e-7
    expanded_gradient_tolerance: float = 1e-6
    accumulation_gradient_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.accumulation_levels != (2, 4, 8):
            raise ValueError("accumulation levels are frozen at 2,4,8")
        if self.pilot_learning_rates != (3e-5, 1e-5):
            raise ValueError("pilot learning rates are frozen at 3e-5,1e-5")
        if self.final_clip_windows != (500, 200):
            raise ValueError("final clipping windows are frozen at 500,200")
        if not 0.0 <= float(self.maximum_clip_fraction) <= 1.0:
            raise ValueError("maximum_clip_fraction must lie in [0,1]")
        if not 0.0 < float(self.preflight_confidence) < 1.0:
            raise ValueError("preflight_confidence must lie in (0,1)")
        if int(self.preflight_paths) != 128:
            raise ValueError("preflight_paths are frozen at 128")
        for name in (
            "loss_algebra_tolerance",
            "expanded_loss_tolerance",
            "expanded_gradient_tolerance",
            "accumulation_gradient_tolerance",
        ):
            if not _finite(getattr(self, name)) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        # The inherited scientific thresholds are part of this workflow's
        # fingerprint.  In particular, do not weaken the derivative gates.
        if self.density_ratio != DensityRatioThresholds():
            raise ValueError("density-ratio scientific thresholds must remain frozen")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RatioStabilityDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    PAIRED_RATIO_ESTIMATOR_INVALID = "paired_ratio_estimator_invalid"
    CLASSIFICATION_VARIANCE_REDUCTION_UNRESOLVED = (
        "classification_variance_reduction_unresolved"
    )
    CLASSIFICATION_OPTIMIZER_INVALID = "classification_optimizer_invalid"
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    CLASSIFICATION_AUDIT_INCONCLUSIVE = "classification_audit_inconclusive"
    DENSITY_RATIO_VALUE_ONLY = "density_ratio_value_only"
    NO_DETECTABLE_DENSITY_RATIO_SIGNAL = "no_detectable_density_ratio_signal"
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


def _passed(value: bool | int | Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        return _one(value.get("passed", 0))
    return bool(value)


def _status(value: Mapping[str, Any]) -> str:
    return str(value.get("evaluation_status", "evaluated"))


def _metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = value.get("metrics", value)
    return dict(raw) if isinstance(raw, Mapping) else {}


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
    evaluated = evaluation_status == "evaluated"
    return {
        "gate": gate,
        "evaluation_status": evaluation_status,
        "passed": int(evaluated and all(_passed(value) for value in subchecks.values())),
        "claim_scope": claim_scope,
        "subchecks": subchecks,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def not_evaluated_gate(gate: str, reason: str) -> dict[str, Any]:
    result = _finish(gate, [], "not evaluated", evaluation_status="not_evaluated")
    result["reason"] = str(reason)
    return result


def _clip_value(metrics: Mapping[str, Any], window: str) -> Any:
    aliases = {
        "post_warmup": (
            "post_warmup_clip_fraction",
            "postwarm_clip_fraction",
        ),
        "final_500": (
            "final_500_clip_fraction",
            "final500_clip_fraction",
        ),
        "final_200": (
            "final_200_clip_fraction",
            "final200_clip_fraction",
        ),
    }
    diagnostics = metrics.get("optimization_diagnostics", {})
    diagnostics = dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
    for name in aliases[window]:
        if name in metrics:
            return metrics[name]
        if name in diagnostics:
            return diagnostics[name]
    return None


def _clip_checks(
    metrics: Mapping[str, Any], thresholds: RatioStabilityThresholds, prefix: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    checks: list[tuple[str, dict[str, Any]]] = []
    for window in ("post_warmup", "final_500", "final_200"):
        value = _clip_value(metrics, window)
        checks.append(
            _check(
                f"{prefix}{window}_clip_fraction",
                value,
                "<=",
                thresholds.maximum_clip_fraction,
                _finite(value)
                and 0.0 <= float(value) <= thresholds.maximum_clip_fraction,
            )
        )
    return checks


def evaluate_paired_ratio_preflight(
    metrics: Mapping[str, Any],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    """Gate exactness of the new estimator, not its advisory variance ratio."""

    thresholds = thresholds or RatioStabilityThresholds()
    flags = (
        "parent_provenance_pass",
        "mixture_coefficients_pass",
        "dirichlet_marginals_pass",
        "common_gamma_coupling_pass",
        "time_strata_pass",
        "class_balance_pass",
        "stream_replay_pass",
        "candidate_order_invariance_pass",
        "fresh_panel_isolation_pass",
        "simultaneous_loss_interval_contains_zero",
        "simultaneous_directional_gradient_intervals_contain_zero",
        "boundary_operator_pass",
        "device_smoke_pass",
    )
    checks: list[tuple[str, Mapping[str, Any]]] = [
        _check("complete", metrics.get("complete"), "==", 1, _one(metrics.get("complete"))),
        _check("finite", metrics.get("finite"), "==", 1, _one(metrics.get("finite"))),
        _check(
            "preflight_paths",
            metrics.get("preflight_paths"),
            "==",
            thresholds.preflight_paths,
            metrics.get("preflight_paths") is not None
            and int(metrics["preflight_paths"]) == thresholds.preflight_paths,
        ),
        _check(
            "preflight_confidence",
            metrics.get("preflight_confidence"),
            ">=",
            thresholds.preflight_confidence,
            _finite(metrics.get("preflight_confidence"))
            and float(metrics["preflight_confidence"])
            >= thresholds.preflight_confidence,
        ),
        _check(
            "loss_algebra_max_error",
            metrics.get("loss_algebra_max_error"),
            "<=",
            thresholds.loss_algebra_tolerance,
            _finite(metrics.get("loss_algebra_max_error"))
            and 0.0
            <= float(metrics["loss_algebra_max_error"])
            <= thresholds.loss_algebra_tolerance,
        ),
        _check(
            "expanded_loss_max_error",
            metrics.get("expanded_loss_max_error"),
            "<=",
            thresholds.expanded_loss_tolerance,
            _finite(metrics.get("expanded_loss_max_error"))
            and 0.0
            <= float(metrics["expanded_loss_max_error"])
            <= thresholds.expanded_loss_tolerance,
        ),
        _check(
            "expanded_gradient_max_error",
            metrics.get("expanded_gradient_max_error"),
            "<=",
            thresholds.expanded_gradient_tolerance,
            _finite(metrics.get("expanded_gradient_max_error"))
            and 0.0
            <= float(metrics["expanded_gradient_max_error"])
            <= thresholds.expanded_gradient_tolerance,
        ),
        _check(
            "accumulation_gradient_max_error",
            metrics.get("accumulation_gradient_max_error"),
            "<=",
            thresholds.accumulation_gradient_tolerance,
            _finite(metrics.get("accumulation_gradient_max_error"))
            and 0.0
            <= float(metrics["accumulation_gradient_max_error"])
            <= thresholds.accumulation_gradient_tolerance,
        ),
    ]
    checks.extend(
        _check(name, metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in flags
    )
    checks.extend(
        [
            _check(
                "parent_loss_scale_reused",
                metrics.get("parent_loss_scale_reused"),
                "==",
                1,
                _one(metrics.get("parent_loss_scale_reused")),
            ),
            _check(
                "adaptive_loss_scaling",
                metrics.get("adaptive_loss_scaling"),
                "==",
                0,
                int(metrics.get("adaptive_loss_scaling", -1)) == 0,
            ),
            _check(
                "physical_training_performed",
                metrics.get("physical_training_performed", 0),
                "==",
                0,
                int(metrics.get("physical_training_performed", 0)) == 0,
            ),
            _check(
                "sampling_performed",
                metrics.get("sampling_performed", 0),
                "==",
                0,
                int(metrics.get("sampling_performed", 0)) == 0,
            ),
        ]
    )
    result = _finish(
        "paired_ratio_preflight",
        checks,
        "exact unbiased paired-mixture BCE and deterministic accumulation",
        evaluation_status=_status(metrics),
    )
    result["variance_forensics_gate_eligible"] = 0
    return result


def evaluate_stability_pilot_candidate(
    candidate: Mapping[str, Any],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    teacher = _metrics(dict(candidate.get("teacher", {})))
    null = _metrics(dict(candidate.get("null", {})))
    base = evaluate_density_ratio_pilot_candidate(
        candidate, thresholds.density_ratio
    )
    accumulation = candidate.get("accumulation_steps")
    learning_rate = candidate.get("learning_rate")
    checks: list[tuple[str, Mapping[str, Any]]] = [
        _check("base_science_and_postwarm", base.get("passed"), "==", 1, _passed(base)),
        _check(
            "accumulation_steps",
            accumulation,
            "in",
            list(thresholds.accumulation_levels),
            accumulation is not None
            and int(accumulation) in thresholds.accumulation_levels,
        ),
        _check(
            "learning_rate",
            learning_rate,
            "in",
            list(thresholds.pilot_learning_rates),
            _finite(learning_rate)
            and float(learning_rate) in thresholds.pilot_learning_rates,
        ),
        *_clip_checks(teacher, thresholds, "teacher_"),
        *_clip_checks(null, thresholds, "null_"),
    ]
    result = _finish(
        "paired_ratio_stability_pilot_candidate",
        checks,
        "train/selection-only paired-estimator optimizer qualification",
        evaluation_status=_status(candidate),
    )
    result.update(
        {
            "learning_rate": float(learning_rate) if _finite(learning_rate) else None,
            "accumulation_steps": int(accumulation)
            if accumulation is not None
            else None,
            "base_gate": base,
            "teacher_mean_ab_bce": base.get("teacher_mean_ab_bce"),
            "teacher_panel_b_bce": base.get("teacher_panel_b_bce"),
            "maximum_clip_fraction_observed": max(
                [
                    float(value[1]["value"])
                    for value in checks
                    if value[0].endswith("clip_fraction")
                    and _finite(value[1].get("value"))
                ],
                default=math.inf,
            ),
        }
    )
    result["optimizer_health_pass"] = int(
        all(
            _passed(value)
            for name, value in checks
            if name.endswith("clip_fraction")
        )
    )
    return result


def select_stability_profile(
    candidate_gates: Sequence[Mapping[str, Any]],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    gates = [dict(value) for value in candidate_gates]
    eligible = [
        (index, value)
        for index, value in enumerate(gates)
        if _passed(value)
    ]
    eligible.sort(
        key=lambda item: (
            float(item[1].get("teacher_mean_ab_bce", math.inf)),
            float(item[1].get("maximum_clip_fraction_observed", math.inf)),
            float(item[1].get("learning_rate", math.inf)),
        )
    )
    selected = eligible[0] if eligible else None
    profile = None
    if selected is not None:
        index, value = selected
        profile = {
            "candidate_index": int(index),
            "learning_rate": float(value["learning_rate"]),
            "accumulation_steps": int(value["accumulation_steps"]),
            "teacher_mean_ab_bce": float(value["teacher_mean_ab_bce"]),
            "teacher_panel_b_bce": float(value["teacher_panel_b_bce"]),
            "maximum_clip_fraction_observed": float(
                value["maximum_clip_fraction_observed"]
            ),
        }
    return {
        "schema": SCHEMA + "-selected-profile",
        "schema_version": SCHEMA_VERSION,
        "selected": int(selected is not None),
        "passed": int(selected is not None),
        "selected_candidate_index": selected[0] if selected is not None else None,
        "profile": profile,
        "ranking": [
            {
                "candidate_index": int(index),
                "learning_rate": float(value["learning_rate"]),
                "accumulation_steps": int(value["accumulation_steps"]),
                "teacher_mean_ab_bce": value.get("teacher_mean_ab_bce"),
                "maximum_clip_fraction_observed": value.get(
                    "maximum_clip_fraction_observed"
                ),
            }
            for index, value in eligible
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_stability_pilot_level(
    candidates: Sequence[Mapping[str, Any]],
    accumulation_steps: int,
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "paired_ratio_stability_pilot_candidate"
        and "subchecks" in value
        else evaluate_stability_pilot_candidate(value, thresholds)
        for value in candidates
    ]
    rates = [value.get("learning_rate") for value in gates]
    accumulations = [value.get("accumulation_steps") for value in gates]
    profile = select_stability_profile(gates, thresholds)
    checks = [
        _check("candidate_count", len(gates), "==", 2, len(gates) == 2),
        _check(
            "learning_rate_set",
            sorted(value for value in rates if _finite(value)),
            "==",
            sorted(thresholds.pilot_learning_rates),
            len(rates) == 2
            and sorted(float(value) for value in rates if _finite(value))
            == sorted(thresholds.pilot_learning_rates),
        ),
        _check(
            "accumulation_level",
            accumulations,
            "== each",
            int(accumulation_steps),
            len(accumulations) == 2
            and all(
                value is not None and int(value) == int(accumulation_steps)
                for value in accumulations
            ),
        ),
        _check("eligible_profile", profile["selected"], "==", 1, _one(profile["selected"])),
    ]
    result = _finish(
        "paired_ratio_stability_pilot_level",
        checks,
        f"paired-estimator accumulation level {int(accumulation_steps)}",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result.update(
        {
            "accumulation_steps": int(accumulation_steps),
            "candidate_gates": gates,
            "selected_profile": profile,
        }
    )
    return result


def evaluate_stability_pilot(
    candidates: Sequence[Mapping[str, Any]],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the first passing complete level, rejecting later peeking."""

    thresholds = thresholds or RatioStabilityThresholds()
    raw = [dict(value) for value in candidates]
    by_level: dict[int, list[dict[str, Any]]] = {
        level: [] for level in thresholds.accumulation_levels
    }
    unexpected = 0
    for candidate in raw:
        try:
            level = int(candidate.get("accumulation_steps", -1))
        except (TypeError, ValueError):
            unexpected += 1
            continue
        if level not in by_level:
            unexpected += 1
            continue
        by_level[level].append(candidate)

    level_gates: list[dict[str, Any]] = []
    selected_level: int | None = None
    selected_profile: dict[str, Any] | None = None
    hierarchy_valid = unexpected == 0
    stopped_for_incomplete = False
    for level in thresholds.accumulation_levels:
        rows = by_level[level]
        if not rows:
            stopped_for_incomplete = True
            # Once an earlier level passes, later levels must be absent.
            if selected_level is not None:
                continue
            if any(
                by_level[later]
                for later in thresholds.accumulation_levels
                if later > level
            ):
                hierarchy_valid = False
            break
        if len(rows) != 2:
            hierarchy_valid = False
        gate = evaluate_stability_pilot_level(rows, level, thresholds)
        level_gates.append(gate)
        if selected_level is None and _passed(gate):
            selected_level = level
            selected_profile = dict(gate["selected_profile"])
            # Any later evidence is forbidden by the predeclared stopping rule.
            if any(by_level[later] for later in thresholds.accumulation_levels if later > level):
                hierarchy_valid = False
            break

    all_levels_complete = all(len(by_level[level]) == 2 for level in thresholds.accumulation_levels)
    terminal_evaluation = selected_level is not None or all_levels_complete
    if selected_profile is None:
        selected_profile = {
            "schema": SCHEMA + "-selected-profile",
            "schema_version": SCHEMA_VERSION,
            "selected": 0,
            "passed": 0,
            "selected_candidate_index": None,
            "profile": None,
            "ranking": [],
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    checks = [
        _check("hierarchical_order", int(hierarchy_valid), "==", 1, hierarchy_valid),
        _check(
            "terminal_evaluation",
            int(terminal_evaluation),
            "==",
            1,
            terminal_evaluation,
        ),
        _check(
            "eligible_profile",
            selected_profile.get("selected", 0),
            "==",
            1,
            _one(selected_profile.get("selected", 0)),
        ),
    ]
    result = _finish(
        "paired_ratio_stability_pilot",
        checks,
        "hierarchical train/selection-only optimizer qualification",
        evaluation_status="evaluated" if terminal_evaluation else "not_evaluated",
    )
    result.update(
        {
            "level_gates": level_gates,
            "selected_accumulation_level": selected_level,
            "selected_profile": selected_profile,
            "all_levels_complete": int(all_levels_complete),
            "stopped_for_incomplete": int(stopped_for_incomplete),
        }
    )
    return result


def _wrap_seed_gate(
    base: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    name: str,
    claim_scope: str,
    thresholds: RatioStabilityThresholds,
) -> dict[str, Any]:
    checks = [
        _check("frozen_scientific_gate", base.get("passed"), "==", 1, _passed(base)),
        *_clip_checks(metrics, thresholds),
    ]
    result = _finish(name, checks, claim_scope, evaluation_status=_status(metrics))
    result.update({key: value for key, value in base.items() if key not in {"gate", "passed", "subchecks", "claim_scope", "evaluation_status"}})
    result["frozen_scientific_gate"] = dict(base)
    result["optimizer_health_pass"] = int(
        all(_passed(value) for key, value in checks if key.endswith("clip_fraction"))
        and bool(int(base.get("optimizer_health_pass", 0)))
    )
    return result


def evaluate_teacher_seed(
    value: Mapping[str, Any],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    metrics = _metrics(value)
    base = evaluate_base_teacher_seed(metrics, thresholds.density_ratio)
    return _wrap_seed_gate(
        base,
        metrics,
        name="paired_ratio_teacher_seed",
        claim_scope="optimizer-healthy frozen teacher classification and derivatives",
        thresholds=thresholds,
    )


def evaluate_null_seed(
    value: Mapping[str, Any],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    metrics = _metrics(value)
    base = evaluate_base_null_seed(metrics, thresholds.density_ratio)
    return _wrap_seed_gate(
        base,
        metrics,
        name="paired_ratio_null_seed",
        claim_scope="optimizer-healthy stationary null with rejected-nominee audits",
        thresholds=thresholds,
    )


def evaluate_teacher_study(
    values: Sequence[Mapping[str, Any]],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "paired_ratio_teacher_seed" and "subchecks" in value
        else evaluate_teacher_seed(value, thresholds)
        for value in values
    ]
    expected = thresholds.density_ratio.expected_teacher_seeds
    seeds = [value.get("model_seed") for value in gates]
    passing = sum(_passed(value) for value in gates)
    classification = sum(bool(int(value.get("classification_pass", 0))) for value in gates)
    derivatives = sum(bool(int(value.get("derivative_pass", 0))) for value in gates)
    disagreement = any(bool(int(value.get("panel_disagreement", 0))) for value in gates)
    optimizer = len(gates) == expected and all(
        bool(int(value.get("optimizer_health_pass", 0))) for value in gates
    )
    checks = [
        _check("task_count", len(gates), "==", expected, len(gates) == expected),
        _check("distinct_seeds", len(set(seeds)), "==", expected, None not in seeds and len(set(seeds)) == expected),
        _check("all_optimizers_valid", int(optimizer), "==", 1, optimizer),
        _check(
            "passing_seeds",
            passing,
            ">=",
            thresholds.density_ratio.minimum_passing_teacher_seeds,
            passing >= thresholds.density_ratio.minimum_passing_teacher_seeds,
        ),
        _check("audit_panels_agree", int(not disagreement), "==", 1, not disagreement),
    ]
    result = _finish(
        "paired_ratio_teacher_study",
        checks,
        "three-seed paired-estimator bounded-teacher confirmation",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result.update(
        {
            "seed_gates": gates,
            "passing_seed_count": passing,
            "classification_passing_seed_count": classification,
            "derivative_passing_seed_count": derivatives,
            "optimizer_health_pass": int(optimizer),
            "panel_disagreement": int(disagreement),
        }
    )
    return result


def evaluate_null_study(
    values: Sequence[Mapping[str, Any]],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "paired_ratio_null_seed" and "subchecks" in value
        else evaluate_null_seed(value, thresholds)
        for value in values
    ]
    expected = thresholds.density_ratio.expected_null_seeds
    seeds = [value.get("model_seed") for value in gates]
    optimizer = len(gates) == expected and all(
        bool(int(value.get("optimizer_health_pass", 0))) for value in gates
    )
    false_discoveries = sum(bool(int(value.get("false_discovery", 0))) for value in gates)
    checks = [
        _check("task_count", len(gates), "==", expected, len(gates) == expected),
        _check("distinct_seeds", len(set(seeds)), "==", expected, None not in seeds and len(set(seeds)) == expected),
        _check("all_optimizers_valid", int(optimizer), "==", 1, optimizer),
        _check("all_null_seeds_pass", sum(_passed(value) for value in gates), "==", expected, len(gates) == expected and all(_passed(value) for value in gates)),
        _check("false_discovery_count", false_discoveries, "==", 0, false_discoveries == 0),
    ]
    result = _finish(
        "paired_ratio_null_study",
        checks,
        "three-seed stationary-Dirichlet independent-state null",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result.update(
        {
            "seed_gates": gates,
            "optimizer_health_pass": int(optimizer),
            "false_discovery_count": false_discoveries,
        }
    )
    return result


def evaluate_ratio_stability_controls(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    teacher = evaluate_teacher_study(teacher_results, thresholds)
    null = evaluate_null_study(null_results, thresholds)
    teacher_seeds = {
        value.get("model_seed") for value in teacher.get("seed_gates", [])
    }
    null_seeds = {
        value.get("model_seed") for value in null.get("seed_gates", [])
    }
    paired_seed_set = (
        None not in teacher_seeds
        and None not in null_seeds
        and len(teacher_seeds) == thresholds.density_ratio.expected_teacher_seeds
        and len(null_seeds) == thresholds.density_ratio.expected_null_seeds
        and teacher_seeds == null_seeds
    )
    checks = [
        _check("provenance", int(_passed(provenance)), "==", 1, _passed(provenance)),
        _check("preflight", int(_passed(preflight)), "==", 1, _passed(preflight)),
        _check("pilot", int(_passed(pilot)), "==", 1, _passed(pilot)),
        _check(
            "paired_teacher_null_seed_set",
            {
                "teacher": sorted(teacher_seeds, key=lambda value: str(value)),
                "null": sorted(null_seeds, key=lambda value: str(value)),
            },
            "==",
            "same complete seed set",
            paired_seed_set,
        ),
        _check("teacher_study", teacher["passed"], "==", 1, _passed(teacher)),
        _check("null_study", null["passed"], "==", 1, _passed(null)),
    ]
    result = _finish(
        "paired_ratio_stability_controls",
        checks,
        "optimizer-healthy strict bounded-teacher and null confirmation",
        evaluation_status="evaluated" if teacher_results or null_results else "not_evaluated",
    )
    result.update(
        {
            "teacher_study": teacher,
            "null_study": null,
            "paired_teacher_null_seed_set_pass": int(paired_seed_set),
            "optimizer_health_pass": int(
                bool(int(teacher.get("optimizer_health_pass", 0)))
                and bool(int(null.get("optimizer_health_pass", 0)))
            ),
        }
    )
    return result


def decide_ratio_stability(
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
    interim = False
    if not _passed(provenance):
        decision: RatioStabilityDecision | str = (
            RatioStabilityDecision.CONTROL_PROVENANCE_INVALID
        )
        action = "repair the exact 222-record density-ratio parent binding"
    elif _status(dict(preflight) if isinstance(preflight, Mapping) else {}) != "evaluated":
        decision = "paired_ratio_preflight_not_evaluated"
        action = "run the paired-mixture estimator preflight"
        interim = True
    elif not _passed(preflight):
        decision = RatioStabilityDecision.PAIRED_RATIO_ESTIMATOR_INVALID
        action = "repair paired-mixture exactness, laws, or accumulation"
    elif _status(dict(pilot) if isinstance(pilot, Mapping) else {}) != "evaluated":
        decision = "paired_ratio_preflight_passed"
        action = "run the hierarchical paired-mixture optimizer pilot"
        interim = True
    elif not _passed(pilot):
        decision = RatioStabilityDecision.CLASSIFICATION_VARIANCE_REDUCTION_UNRESOLVED
        action = "no hierarchical pilot profile qualified; investigate a separately gated function-space trust constraint"
    elif _status(controls) != "evaluated":
        decision = "paired_ratio_pilot_passed"
        action = "run the fresh three-seed paired-mixture confirmation"
        interim = True
    elif not bool(int(controls.get("optimizer_health_pass", 0))):
        decision = RatioStabilityDecision.CLASSIFICATION_OPTIMIZER_INVALID
        action = "repair incomplete, nonfinite, clipped, or incompatible confirmation tasks"
    elif int(null.get("false_discovery_count", 0)) > 0:
        decision = RatioStabilityDecision.SELECTION_FALSE_DISCOVERY
        action = "repair discovery/confirmation calibration before score learning"
    elif bool(int(teacher.get("panel_disagreement", 0))):
        decision = RatioStabilityDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "rerun unchanged with 64 whole paths per audit panel"
    elif _passed(controls):
        decision = RatioStabilityDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        action = "plan fresh physical one-image density-ratio score training"
    elif int(teacher.get("classification_passing_seed_count", 0)) >= 2:
        decision = RatioStabilityDecision.DENSITY_RATIO_VALUE_ONLY
        action = "plan a separate derivative or physical-flux regularization control"
    else:
        decision = RatioStabilityDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL
        action = "revisit classifier capacity on the exact bounded synthetic law"
    decision_value = decision.value if isinstance(decision, RatioStabilityDecision) else decision
    return {
        "decision": decision_value,
        "recommended_next_action": action,
        "interim_stage_success": int(interim and decision_value.endswith("_passed")),
        "closed_terminal_scientific_outcome": int(not interim),
        "physical_training_authorized": int(
            decision
            is RatioStabilityDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        ),
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


def evaluate_ratio_stability_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    require_gate: str = "none",
    thresholds: RatioStabilityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or RatioStabilityThresholds()
    if require_gate not in {"none", "preflight", "pilot", "controls"}:
        raise ValueError("require_gate must be none, preflight, pilot, or controls")
    controls = evaluate_ratio_stability_controls(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        teacher_results=teacher_results,
        null_results=null_results,
        thresholds=thresholds,
    )
    required_pass = {
        "none": True,
        "preflight": _passed(provenance) and _passed(preflight),
        "pilot": _passed(provenance) and _passed(preflight) and _passed(pilot),
        "controls": _passed(controls),
    }[require_gate]
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "components": {
            "provenance": dict(provenance)
            if isinstance(provenance, Mapping)
            else int(_passed(provenance)),
            "preflight": dict(preflight)
            if isinstance(preflight, Mapping)
            else int(_passed(preflight)),
            "pilot": dict(pilot)
            if isinstance(pilot, Mapping)
            else int(_passed(pilot)),
            "controls": controls,
        },
        "decision": decide_ratio_stability(
            provenance=provenance,
            preflight=preflight,
            pilot=pilot,
            controls=controls,
        ),
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "thresholds": thresholds.to_dict(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
