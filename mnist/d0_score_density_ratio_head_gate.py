"""Pure gates for the normalized-head density-ratio coordinate repair.

The workflow changes only the parameter coordinate used by the scalar output
head.  Consequently this module keeps the paired-mixture density-ratio
scientific gates frozen and adds two contracts:

* preflight must establish functional and optimizer-coordinate equivalence;
* the pilot uses the already selected accumulation level eight and exactly the
  two body learning rates 3e-5 and 1e-5.

There is deliberately no filesystem, model, training, or sampling code here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_score_density_ratio_stability_gate import (
    RatioStabilityThresholds,
    evaluate_null_seed as evaluate_frozen_null_seed,
    evaluate_stability_pilot_candidate as evaluate_frozen_pilot_candidate,
    evaluate_teacher_seed as evaluate_frozen_teacher_seed,
)


__all__ = [
    "HeadCoordinateThresholds",
    "HeadCoordinateDecision",
    "not_evaluated_gate",
    "evaluate_normalized_head_preflight",
    "evaluate_head_pilot_candidate",
    "select_head_profile",
    "evaluate_head_pilot",
    "evaluate_teacher_seed",
    "evaluate_teacher_study",
    "evaluate_null_seed",
    "evaluate_null_study",
    "evaluate_head_controls",
    "decide_head_coordinate",
    "evaluate_head_workflow",
]


SCHEMA = "experiment12-d0-score-density-ratio-head-coordinate-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HeadCoordinateThresholds:
    """Frozen numerical and scientific thresholds for the repair."""

    stability: RatioStabilityThresholds = field(default_factory=RatioStabilityThresholds)
    grid_cells: int = 784
    base_channels: int = 32
    accumulation_steps: int = 8
    pilot_learning_rates: tuple[float, ...] = (3e-5, 1e-5)
    cuda_equivalence_tolerance: float = 2e-6
    float64_equivalence_tolerance: float = 1e-9
    derivative_equivalence_tolerance: float = 2e-6
    gradient_scaling_tolerance: float = 2e-6
    optimizer_conjugacy_tolerance: float = 2e-6
    minimum_legacy_head_squared_gradient_share: float = 0.95

    def __post_init__(self) -> None:
        if self.stability != RatioStabilityThresholds():
            raise ValueError("paired density-ratio thresholds must remain frozen")
        if int(self.grid_cells) != 784:
            raise ValueError("grid_cells are frozen at 784")
        if int(self.base_channels) != 32:
            raise ValueError("base_channels are frozen at 32")
        if int(self.accumulation_steps) != 8:
            raise ValueError("gradient accumulation is frozen at 8")
        if self.pilot_learning_rates != (3e-5, 1e-5):
            raise ValueError("pilot learning rates are frozen at 3e-5,1e-5")
        for name in (
            "cuda_equivalence_tolerance",
            "float64_equivalence_tolerance",
            "derivative_equivalence_tolerance",
            "gradient_scaling_tolerance",
            "optimizer_conjugacy_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        share = float(self.minimum_legacy_head_squared_gradient_share)
        if not math.isfinite(share) or not 0.0 <= share <= 1.0:
            raise ValueError(
                "minimum_legacy_head_squared_gradient_share must lie in [0,1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeadCoordinateDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    NORMALIZED_HEAD_COORDINATE_INVALID = "normalized_head_coordinate_invalid"
    CLASSIFICATION_COORDINATE_REPAIR_UNRESOLVED = (
        "classification_coordinate_repair_unresolved"
    )
    CLASSIFICATION_OPTIMIZER_INVALID = "classification_optimizer_invalid"
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    CLASSIFICATION_AUDIT_INCONCLUSIVE = "classification_audit_inconclusive"
    NO_DETECTABLE_DENSITY_RATIO_SIGNAL = "no_detectable_density_ratio_signal"
    DENSITY_RATIO_VALUE_ONLY = "density_ratio_value_only"
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


def _bounded_error_check(
    metrics: Mapping[str, Any],
    name: str,
    tolerance: float,
) -> tuple[str, dict[str, Any]]:
    value = metrics.get(name)
    return _check(
        name,
        value,
        "<=",
        tolerance,
        _finite(value) and 0.0 <= float(value) <= tolerance,
    )


def evaluate_normalized_head_preflight(
    metrics: Mapping[str, Any],
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    """Gate exact function/gradient/AdamW coordinate equivalence."""

    thresholds = thresholds or HeadCoordinateThresholds()
    checks: list[tuple[str, Mapping[str, Any]]] = [
        _check("complete", metrics.get("complete"), "==", 1, _one(metrics.get("complete"))),
        _check("finite", metrics.get("finite"), "==", 1, _one(metrics.get("finite"))),
        _check(
            "parent_provenance_pass",
            metrics.get("parent_provenance_pass"),
            "==",
            1,
            _one(metrics.get("parent_provenance_pass")),
        ),
        _check(
            "grid_cells",
            metrics.get("grid_cells"),
            "==",
            thresholds.grid_cells,
            metrics.get("grid_cells") is not None
            and int(metrics["grid_cells"]) == thresholds.grid_cells,
        ),
        _check(
            "base_channels",
            metrics.get("base_channels"),
            "==",
            thresholds.base_channels,
            metrics.get("base_channels") is not None
            and int(metrics["base_channels"]) == thresholds.base_channels,
        ),
        _check(
            "preflight_paths",
            metrics.get("preflight_paths"),
            "==",
            thresholds.stability.preflight_paths,
            metrics.get("preflight_paths") is not None
            and int(metrics["preflight_paths"])
            == thresholds.stability.preflight_paths,
        ),
        _check(
            "preflight_confidence",
            metrics.get("preflight_confidence"),
            ">=",
            thresholds.stability.preflight_confidence,
            _finite(metrics.get("preflight_confidence"))
            and float(metrics["preflight_confidence"])
            >= thresholds.stability.preflight_confidence,
        ),
        _bounded_error_check(
            metrics,
            "loss_algebra_max_error",
            thresholds.stability.loss_algebra_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "expanded_loss_max_error",
            thresholds.stability.expanded_loss_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "expanded_gradient_max_error",
            thresholds.stability.expanded_gradient_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "accumulation_gradient_max_error",
            thresholds.stability.accumulation_gradient_tolerance,
        ),
        _bounded_error_check(
            metrics, "cuda_logit_max_abs_error", thresholds.cuda_equivalence_tolerance
        ),
        _bounded_error_check(
            metrics, "cuda_bce_max_abs_error", thresholds.cuda_equivalence_tolerance
        ),
        _bounded_error_check(
            metrics,
            "float64_logit_max_abs_error",
            thresholds.float64_equivalence_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "float64_bce_max_abs_error",
            thresholds.float64_equivalence_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "state_gradient_relative_error",
            thresholds.derivative_equivalence_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "edge_score_relative_error",
            thresholds.derivative_equivalence_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "flux_relative_error",
            thresholds.derivative_equivalence_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "head_gradient_scale_relative_error",
            thresholds.gradient_scaling_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "backbone_gradient_relative_error",
            thresholds.gradient_scaling_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "adamw_coordinate_max_relative_error",
            thresholds.optimizer_conjugacy_tolerance,
        ),
        _bounded_error_check(
            metrics,
            "ema_coordinate_max_relative_error",
            thresholds.optimizer_conjugacy_tolerance,
        ),
    ]
    for name in (
        "head_gradient_one_over_n_pass",
        "backbone_gradient_unchanged_pass",
        "adamw_group_learning_rate_pass",
        "adamw_group_epsilon_pass",
        "adamw_group_weight_decay_pass",
        "legacy_checkpoint_report_only_pass",
        "boundary_operator_pass",
        "finite_device_backward_pass",
        "stream_replay_pass",
        "paired_estimator_pass",
        "mixture_coefficients_pass",
        "dirichlet_marginals_pass",
        "common_gamma_coupling_pass",
        "exact_seed_namespaces_pass",
        "null_pool_swap_structure_pass",
        "time_strata_pass",
        "class_balance_pass",
        "candidate_order_invariance_pass",
        "nested_accumulation_prefix_pass",
        "parent_forensic_finite",
    ):
        checks.append(_check(name, metrics.get(name), "==", 1, _one(metrics.get(name))))
    share = metrics.get("median_legacy_head_squared_gradient_share")
    checks.append(
        _check(
            "median_legacy_head_squared_gradient_share",
            share,
            ">=",
            thresholds.minimum_legacy_head_squared_gradient_share,
            _finite(share)
            and thresholds.minimum_legacy_head_squared_gradient_share
            <= float(share)
            <= 1.0,
        )
    )
    checks.extend(
        (
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
                metrics.get("adaptive_loss_scaling") is not None
                and int(metrics["adaptive_loss_scaling"]) == 0,
            ),
        )
    )
    for name in ("physical_training_performed", "sampling_performed"):
        value = metrics.get(name, 0)
        checks.append(_check(name, value, "==", 0, int(value) == 0))
    result = _finish(
        "normalized_head_coordinate_preflight",
        checks,
        "exact spatial-sum/spatial-mean function and optimizer-coordinate equivalence",
        evaluation_status=_status(metrics),
    )
    result["width_ablation_gate_eligible"] = 0
    return result


def evaluate_head_pilot_candidate(
    candidate: Mapping[str, Any],
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    base = evaluate_frozen_pilot_candidate(candidate, thresholds.stability)
    accumulation = candidate.get("accumulation_steps")
    learning_rate = candidate.get("learning_rate")
    checks = [
        _check("frozen_paired_gate", base.get("passed"), "==", 1, _passed(base)),
        _check(
            "accumulation_steps",
            accumulation,
            "==",
            thresholds.accumulation_steps,
            accumulation is not None and int(accumulation) == thresholds.accumulation_steps,
        ),
        _check(
            "body_learning_rate",
            learning_rate,
            "in",
            list(thresholds.pilot_learning_rates),
            _finite(learning_rate)
            and float(learning_rate) in thresholds.pilot_learning_rates,
        ),
    ]
    result = _finish(
        "normalized_head_pilot_candidate",
        checks,
        "train/selection-only coordinate-repaired classifier qualification",
        evaluation_status=_status(candidate),
    )
    result.update(
        {
            key: value
            for key, value in base.items()
            if key
            not in {"gate", "passed", "subchecks", "claim_scope", "evaluation_status"}
        }
    )
    result["frozen_paired_gate"] = base
    result["accumulation_steps"] = (
        int(accumulation) if accumulation is not None else None
    )
    result["learning_rate"] = float(learning_rate) if _finite(learning_rate) else None
    result["body_learning_rate"] = result["learning_rate"]
    result["head_learning_rate"] = (
        thresholds.grid_cells * float(learning_rate) if _finite(learning_rate) else None
    )
    return result


def select_head_profile(
    candidate_gates: Sequence[Mapping[str, Any]],
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    gates = [dict(value) for value in candidate_gates]
    eligible = [(index, value) for index, value in enumerate(gates) if _passed(value)]
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
        body_lr = float(value["learning_rate"])
        profile = {
            "candidate_index": int(index),
            "accumulation_steps": thresholds.accumulation_steps,
            # ``learning_rate`` is the body rate retained for checkpoint and
            # confirmation-profile compatibility; the explicit aliases make
            # the two optimizer coordinate groups unambiguous.
            "learning_rate": body_lr,
            "body_learning_rate": body_lr,
            "head_learning_rate": thresholds.grid_cells * body_lr,
            "teacher_mean_ab_bce": float(value["teacher_mean_ab_bce"]),
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
                "body_learning_rate": float(value["learning_rate"]),
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


def evaluate_head_pilot(
    candidates: Sequence[Mapping[str, Any]],
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "normalized_head_pilot_candidate"
        and "subchecks" in value
        else evaluate_head_pilot_candidate(value, thresholds)
        for value in candidates
    ]
    rates = [value.get("learning_rate") for value in gates]
    accumulations = [value.get("accumulation_steps") for value in gates]
    profile = select_head_profile(gates, thresholds)
    checks = [
        _check("candidate_count", len(gates), "==", 2, len(gates) == 2),
        _check(
            "body_learning_rate_set",
            sorted(value for value in rates if _finite(value)),
            "==",
            sorted(thresholds.pilot_learning_rates),
            len(rates) == 2
            and sorted(float(value) for value in rates if _finite(value))
            == sorted(thresholds.pilot_learning_rates),
        ),
        _check(
            "all_accumulation_eight",
            accumulations,
            "== each",
            thresholds.accumulation_steps,
            len(accumulations) == 2
            and all(
                value is not None and int(value) == thresholds.accumulation_steps
                for value in accumulations
            ),
        ),
        _check("eligible_profile", profile["selected"], "==", 1, _one(profile["selected"])),
    ]
    result = _finish(
        "normalized_head_pilot",
        checks,
        "fixed accumulation-eight normalized-head optimizer pilot",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result.update({"candidate_gates": gates, "selected_profile": profile})
    return result


def _rename_gate(base: Mapping[str, Any], name: str, claim_scope: str) -> dict[str, Any]:
    result = dict(base)
    result["gate"] = name
    result["claim_scope"] = claim_scope
    return result


def evaluate_teacher_seed(
    value: Mapping[str, Any],
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    return _rename_gate(
        evaluate_frozen_teacher_seed(value, thresholds.stability),
        "normalized_head_teacher_seed",
        "optimizer-healthy frozen teacher classification and derivatives",
    )


def evaluate_null_seed(
    value: Mapping[str, Any],
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    return _rename_gate(
        evaluate_frozen_null_seed(value, thresholds.stability),
        "normalized_head_null_seed",
        "optimizer-healthy stationary null with rejected-nominee audits",
    )


def evaluate_teacher_study(
    values: Sequence[Mapping[str, Any]],
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "normalized_head_teacher_seed" and "subchecks" in value
        else evaluate_teacher_seed(value, thresholds)
        for value in values
    ]
    expected = thresholds.stability.density_ratio.expected_teacher_seeds
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
        _check(
            "distinct_seeds",
            len(set(seeds)),
            "==",
            expected,
            None not in seeds and len(set(seeds)) == expected,
        ),
        _check("all_optimizers_valid", int(optimizer), "==", 1, optimizer),
        _check(
            "passing_seeds",
            passing,
            ">=",
            thresholds.stability.density_ratio.minimum_passing_teacher_seeds,
            passing
            >= thresholds.stability.density_ratio.minimum_passing_teacher_seeds,
        ),
        _check("audit_panels_agree", int(not disagreement), "==", 1, not disagreement),
    ]
    result = _finish(
        "normalized_head_teacher_study",
        checks,
        "three-seed normalized-head bounded-teacher confirmation",
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
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "normalized_head_null_seed" and "subchecks" in value
        else evaluate_null_seed(value, thresholds)
        for value in values
    ]
    expected = thresholds.stability.density_ratio.expected_null_seeds
    seeds = [value.get("model_seed") for value in gates]
    optimizer = len(gates) == expected and all(
        bool(int(value.get("optimizer_health_pass", 0))) for value in gates
    )
    false_discoveries = sum(bool(int(value.get("false_discovery", 0))) for value in gates)
    checks = [
        _check("task_count", len(gates), "==", expected, len(gates) == expected),
        _check(
            "distinct_seeds",
            len(set(seeds)),
            "==",
            expected,
            None not in seeds and len(set(seeds)) == expected,
        ),
        _check("all_optimizers_valid", int(optimizer), "==", 1, optimizer),
        _check(
            "all_null_seeds_pass",
            sum(_passed(value) for value in gates),
            "==",
            expected,
            len(gates) == expected and all(_passed(value) for value in gates),
        ),
        _check("false_discovery_count", false_discoveries, "==", 0, false_discoveries == 0),
    ]
    result = _finish(
        "normalized_head_null_study",
        checks,
        "three-seed stationary-Dirichlet normalized-head null",
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


def evaluate_head_controls(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    teacher = evaluate_teacher_study(teacher_results, thresholds)
    null = evaluate_null_study(null_results, thresholds)
    teacher_seeds = {value.get("model_seed") for value in teacher.get("seed_gates", [])}
    null_seeds = {value.get("model_seed") for value in null.get("seed_gates", [])}
    expected = thresholds.stability.density_ratio.expected_teacher_seeds
    paired = (
        None not in teacher_seeds
        and None not in null_seeds
        and len(teacher_seeds) == expected
        and len(null_seeds) == expected
        and teacher_seeds == null_seeds
    )
    checks = [
        _check("provenance", int(_passed(provenance)), "==", 1, _passed(provenance)),
        _check("preflight", int(_passed(preflight)), "==", 1, _passed(preflight)),
        _check("pilot", int(_passed(pilot)), "==", 1, _passed(pilot)),
        _check("paired_teacher_null_seed_set", int(paired), "==", 1, paired),
        _check("teacher_study", teacher["passed"], "==", 1, _passed(teacher)),
        _check("null_study", null["passed"], "==", 1, _passed(null)),
    ]
    result = _finish(
        "normalized_head_controls",
        checks,
        "strict derivative-accurate normalized-head teacher/null confirmation",
        evaluation_status="evaluated" if teacher_results or null_results else "not_evaluated",
    )
    result.update(
        {
            "teacher_study": teacher,
            "null_study": null,
            "paired_teacher_null_seed_set_pass": int(paired),
            "optimizer_health_pass": int(
                bool(int(teacher.get("optimizer_health_pass", 0)))
                and bool(int(null.get("optimizer_health_pass", 0)))
            ),
        }
    )
    return result


def decide_head_coordinate(
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
        decision: HeadCoordinateDecision | str = HeadCoordinateDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the exact 332-record paired-density-ratio parent binding"
    elif _status(dict(preflight) if isinstance(preflight, Mapping) else {}) != "evaluated":
        decision = "normalized_head_preflight_not_evaluated"
        action = "run the normalized-head coordinate preflight"
        interim = True
    elif not _passed(preflight):
        decision = HeadCoordinateDecision.NORMALIZED_HEAD_COORDINATE_INVALID
        action = "repair functional, gradient, AdamW, or EMA coordinate equivalence"
    elif _status(dict(pilot) if isinstance(pilot, Mapping) else {}) != "evaluated":
        decision = "normalized_head_preflight_passed"
        action = "run the fixed accumulation-eight normalized-head pilot"
        interim = True
    elif not _passed(pilot):
        decision = HeadCoordinateDecision.CLASSIFICATION_COORDINATE_REPAIR_UNRESOLVED
        action = "investigate a separately gated H1-like function-step trust region"
    elif _status(controls) != "evaluated":
        decision = "normalized_head_pilot_passed"
        action = "run the fresh three-seed normalized-head confirmation"
        interim = True
    elif not bool(int(controls.get("optimizer_health_pass", 0))):
        decision = HeadCoordinateDecision.CLASSIFICATION_OPTIMIZER_INVALID
        action = "repair incomplete, nonfinite, clipped, or incompatible confirmation tasks"
    elif int(null.get("false_discovery_count", 0)) > 0:
        decision = HeadCoordinateDecision.SELECTION_FALSE_DISCOVERY
        action = "repair discovery/confirmation calibration before score learning"
    elif bool(int(teacher.get("panel_disagreement", 0))):
        decision = HeadCoordinateDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "rerun unchanged with 64 whole paths per audit panel"
    elif _passed(controls):
        decision = HeadCoordinateDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        action = "plan fresh physical one-image density-ratio score training"
    elif int(teacher.get("classification_passing_seed_count", 0)) >= 2:
        decision = HeadCoordinateDecision.DENSITY_RATIO_VALUE_ONLY
        action = "do not shrink the model; investigate derivative learning separately"
    else:
        decision = HeadCoordinateDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL
        action = "revisit classifier capacity on the exact bounded synthetic law"
    value = decision.value if isinstance(decision, HeadCoordinateDecision) else decision
    return {
        "decision": value,
        "recommended_next_action": action,
        "interim_stage_success": int(interim and value.endswith("_passed")),
        "closed_terminal_scientific_outcome": int(not interim),
        "physical_training_authorized": int(
            decision is HeadCoordinateDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        ),
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


def evaluate_head_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    require_gate: str = "none",
    thresholds: HeadCoordinateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or HeadCoordinateThresholds()
    if require_gate not in {"none", "preflight", "pilot", "controls"}:
        raise ValueError("require_gate must be none, preflight, pilot, or controls")
    controls = evaluate_head_controls(
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
            "pilot": dict(pilot) if isinstance(pilot, Mapping) else int(_passed(pilot)),
            "controls": controls,
        },
        "decision": decide_head_coordinate(
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
