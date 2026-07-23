"""Pure gates for the D0 boundary-control optimizer-scale repair.

This module is deliberately independent of Torch, training, and filesystem
code.  Version one of the boundary-control gate combined analytic teacher
quality with optimizer clipping in one ``supervised_teacher`` result.  That
made a clipping-only failure look like ``representation_invalid`` and allowed
empty probe studies to be described as agreeing.  The version-two gate keeps
those claims separate and represents probe-bank evidence with an explicit
three-state value.

Scientific score/flux thresholds are imported from the original gate module;
the scale-repair patch does not change them.
"""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_score_boundary_control_gate import BoundaryControlThresholds


__all__ = [
    "ProbeBankStatus",
    "ScaleRepairDecision",
    "classify_probe_bank_status",
    "not_evaluated_study",
    "evaluate_loss_scale_calibration",
    "split_supervised_teacher_gate",
    "evaluate_optimizer_task_health",
    "evaluate_scale_repair_gate",
    "decide_scale_repair",
    "evaluate_scale_repair_gates",
]


SCHEMA = "experiment12-d0-score-control-scale-repair-gate"
SCHEMA_VERSION = 2
CALIBRATION_STATE_COUNT = 256


class ProbeBankStatus(str, Enum):
    """Whether both required independent probe banks were actually compared."""

    NOT_EVALUATED = "not_evaluated"
    AGREE = "agree"
    DISAGREE = "disagree"


class ScaleRepairDecision(str, Enum):
    CONTROL_PIPELINE_REPAIRED = "control_pipeline_repaired"
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    BOUNDARY_DOMAIN_INVALID = "boundary_domain_invalid"
    OPTIMIZER_SCALE_INVALID = "optimizer_scale_invalid"
    REPRESENTATION_INVALID = "representation_invalid"
    TRACE_ESTIMATOR_INCONCLUSIVE = "trace_estimator_inconclusive"
    IMPLICIT_OBJECTIVE_UNSTABLE = "implicit_objective_unstable"


def _passed(value: bool | int | Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        return bool(int(value.get("passed", value.get("gate_pass", 0))))
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError("gate values must be a mapping, boolean, or 0/1")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
    name: str,
    checks: Sequence[tuple[str, Mapping[str, Any]]],
    claim_scope: str,
    *,
    evaluation_status: str = "evaluated",
) -> dict[str, Any]:
    subchecks = {key: dict(value) for key, value in checks}
    passed = bool(subchecks) and evaluation_status == "evaluated" and all(
        bool(int(value.get("passed", 0))) for value in subchecks.values()
    )
    return {
        "gate": name,
        "evaluation_status": evaluation_status,
        "passed": int(passed),
        "subchecks": subchecks,
        "claim_scope": claim_scope,
        "sampling_performed": 0,
    }


def _evaluation_status(value: bool | int | Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        return "evaluated"
    raw = str(
        value.get("evaluation_status", value.get("study_status", "evaluated"))
    ).strip().lower()
    aliases = {
        "complete": "evaluated",
        "completed": "evaluated",
        "skipped": "not_evaluated",
        "incomplete": "not_evaluated",
        "pending": "not_evaluated",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"evaluated", "not_evaluated"}:
        raise ValueError(f"unknown evaluation status {raw!r}")
    return normalized


def not_evaluated_study(name: str, reason: str) -> dict[str, Any]:
    """Return an explicit fail-closed record for a skipped study."""

    if not str(reason).strip():
        raise ValueError("a skipped study requires a reason")
    return {
        "gate": str(name),
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "skip_reason": str(reason),
        "subchecks": {},
        "sampling_performed": 0,
    }


def classify_probe_bank_status(
    *,
    studies_evaluated: bool,
    banks_agree: bool | None,
) -> ProbeBankStatus:
    """Classify probe evidence without treating absent evidence as agreement."""

    if not studies_evaluated or banks_agree is None:
        return ProbeBankStatus.NOT_EVALUATED
    return ProbeBankStatus.AGREE if banks_agree else ProbeBankStatus.DISAGREE


def _probe_status(value: ProbeBankStatus | str) -> ProbeBankStatus:
    if isinstance(value, ProbeBankStatus):
        return value
    try:
        return ProbeBankStatus(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ProbeBankStatus)
        raise ValueError(f"probe_bank_status must be one of {allowed}") from exc


def evaluate_loss_scale_calibration(
    record: Mapping[str, Any],
    *,
    expected_initial_grad_target: float,
    expected_state_count: int = CALIBRATION_STATE_COUNT,
    expected_objective_kind: str | None = None,
) -> dict[str, Any]:
    """Verify the frozen training-only initial-gradient calibration formula."""

    target = float(expected_initial_grad_target)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("expected_initial_grad_target must be finite and positive")
    if int(expected_state_count) <= 0:
        raise ValueError("expected_state_count must be positive")

    raw_norm = record.get(
        "unscaled_initial_gradient_norm",
        record.get("unscaled_initial_grad_norm", record.get("initial_unscaled_grad_norm")),
    )
    multiplier = record.get("loss_scale", record.get("multiplier"))
    scaled_norm = record.get(
        "scaled_initial_gradient_norm", record.get("scaled_initial_grad_norm")
    )
    recorded_target = record.get(
        "target_initial_gradient_norm", record.get("initial_grad_target")
    )
    count = record.get("state_count", record.get("calibration_state_count"))
    raw_valid = _finite(raw_norm) and float(raw_norm) > 0.0
    expected_multiplier = min(1.0, target / float(raw_norm)) if raw_valid else None
    multiplier_valid = (
        _finite(multiplier) and 0.0 < float(multiplier) <= 1.0
    )
    formula_pass = (
        expected_multiplier is not None
        and multiplier_valid
        and math.isclose(
            float(multiplier),
            float(expected_multiplier),
            rel_tol=1e-10,
            abs_tol=1e-12,
        )
    )
    expected_scaled_norm = (
        float(raw_norm) * float(multiplier)
        if raw_valid and multiplier_valid
        else None
    )
    scaled_norm_pass = (
        expected_scaled_norm is not None
        and _finite(scaled_norm)
        and float(scaled_norm) > 0.0
        and float(scaled_norm) <= target * (1.0 + 1e-12)
        and math.isclose(
            float(scaled_norm), expected_scaled_norm, rel_tol=1e-10, abs_tol=1e-12
        )
    )
    objective_kind = str(record.get("objective_kind", ""))
    objective_kind_pass = bool(objective_kind) and (
        expected_objective_kind is None
        or objective_kind == str(expected_objective_kind)
    )
    calibration_split = str(record.get("calibration_split", ""))
    state_hash = str(record.get("calibration_state_sha256", ""))
    binding = record.get("binding")
    checks = [
        _check("complete", int(bool(record.get("complete", 0))), "==", 1, bool(record.get("complete", 0))),
        _check("finite", int(bool(record.get("finite", 0))), "==", 1, bool(record.get("finite", 0))),
        _check("training_only", int(bool(record.get("training_only", 0))), "==", 1, bool(record.get("training_only", 0))),
        _check("state_count", count, "==", int(expected_state_count), count is not None and int(count) == int(expected_state_count)),
        _check("unscaled_initial_grad_norm", raw_norm, ">", 0.0, raw_valid),
        _check("initial_grad_target", recorded_target, "==", target, _finite(recorded_target) and math.isclose(float(recorded_target), target, rel_tol=0.0, abs_tol=1e-15)),
        _check("loss_scale_range", multiplier, "in", "(0, 1]", multiplier_valid),
        _check("loss_scale_formula", multiplier, "==", expected_multiplier, formula_pass),
        _check("scaled_initial_gradient_norm", scaled_norm, "== raw*scale <=", target, scaled_norm_pass),
        _check("objective_kind", objective_kind, "==", expected_objective_kind or "nonempty", objective_kind_pass),
        _check("calibration_split", calibration_split, "==", "train", calibration_split == "train"),
        _check("calibration_state_sha256", state_hash, "nonempty", True, bool(state_hash)),
        _check("binding", int(isinstance(binding, Mapping) and bool(binding)), "==", 1, isinstance(binding, Mapping) and bool(binding)),
    ]
    result = _finish(
        "loss_scale_calibration",
        checks,
        "training-only optimizer-unit calibration",
    )
    result["expected_loss_scale"] = expected_multiplier
    result["expected_initial_grad_target"] = target
    result["expected_state_count"] = int(expected_state_count)
    result["expected_objective_kind"] = expected_objective_kind
    return result


_SUPERVISED_OPTIMIZER_CHECKS = frozenset(
    {"complete", "finite", "post_warmup_clip_fraction"}
)
_SUPERVISED_REPRESENTATION_CHECKS = frozenset(
    {
        "selected_nonzero",
        "overall_score_gain",
        "data_end_score_gain",
        "overall_flux_cosine",
        "all_bin_flux_cosines",
        "overall_relative_flux_l2",
        "all_bin_relative_flux_l2",
        "boundary_admissible",
    }
)


def split_supervised_teacher_gate(
    supervised_gate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Split a legacy teacher gate into optimizer and scientific claims.

    The original gate's thresholds and individual check results are preserved
    byte-for-value.  Only their aggregation is changed.
    """

    subchecks = supervised_gate.get("subchecks", {})
    if not isinstance(subchecks, Mapping) or not subchecks:
        optimizer = _finish(
            "supervised_optimizer_health",
            [],
            "finite supervised optimization within the frozen clipping limit",
            evaluation_status="not_evaluated",
        )
        representation = _finish(
            "supervised_representation",
            [],
            "bounded analytic score and flux representability",
            evaluation_status="not_evaluated",
        )
        return {"optimizer": optimizer, "representation": representation}

    def required_checks(names: frozenset[str]) -> list[tuple[str, Mapping[str, Any]]]:
        checks: list[tuple[str, Mapping[str, Any]]] = []
        for name in sorted(names):
            value = subchecks.get(name)
            if isinstance(value, Mapping):
                checks.append((name, dict(value)))
            else:
                checks.append(
                    (
                        name,
                        {
                            "value": None,
                            "operator": "present",
                            "threshold": 1,
                            "passed": 0,
                        },
                    )
                )
        return checks

    optimizer_checks = required_checks(_SUPERVISED_OPTIMIZER_CHECKS)
    representation_checks = required_checks(_SUPERVISED_REPRESENTATION_CHECKS)
    optimizer = _finish(
        "supervised_optimizer_health",
        optimizer_checks,
        "finite supervised optimization within the frozen clipping limit",
    )
    representation = _finish(
        "supervised_representation",
        representation_checks,
        "bounded analytic score and flux representability",
    )
    return {"optimizer": optimizer, "representation": representation}


def evaluate_optimizer_task_health(
    metrics: Mapping[str, Any],
    thresholds: BoundaryControlThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate optimizer health without making a scientific-fit claim."""

    thresholds = thresholds or BoundaryControlThresholds()
    status = _evaluation_status(metrics)
    clip = metrics.get("post_warmup_clip_fraction")
    checks = [
        _check("complete", int(bool(metrics.get("complete", 0))), "==", 1, bool(metrics.get("complete", 0))),
        _check("finite", int(bool(metrics.get("finite", 0))), "==", 1, bool(metrics.get("finite", 0))),
        _check(
            "post_warmup_clip_fraction",
            clip,
            "<=",
            thresholds.maximum_post_warmup_clip_fraction,
            _finite(clip)
            and 0.0 <= float(clip)
            <= thresholds.maximum_post_warmup_clip_fraction,
        ),
    ]
    return _finish(
        "optimizer_task_health",
        checks,
        "finite optimization within the frozen clipping limit",
        evaluation_status=status,
    )


def evaluate_scale_repair_gate(
    *,
    provenance_pass: bool | int | Mapping[str, Any],
    boundary_preflight: bool | int | Mapping[str, Any],
    supervised_calibration: bool | int | Mapping[str, Any],
    implicit_calibration: bool | int | Mapping[str, Any],
    supervised_optimizer: bool | int | Mapping[str, Any],
    supervised_representation: bool | int | Mapping[str, Any],
    downstream_optimizer: bool | int | Mapping[str, Any],
    implicit_teacher_study: bool | int | Mapping[str, Any],
    null_study: bool | int | Mapping[str, Any],
    probe_bank_status: ProbeBankStatus | str = ProbeBankStatus.NOT_EVALUATED,
) -> dict[str, Any]:
    status = _probe_status(probe_bank_status)
    components: dict[str, bool | int | Mapping[str, Any]] = {
        "provenance": provenance_pass,
        "boundary_preflight": boundary_preflight,
        "supervised_calibration": supervised_calibration,
        "implicit_calibration": implicit_calibration,
        "supervised_optimizer": supervised_optimizer,
        "supervised_representation": supervised_representation,
        "downstream_optimizer": downstream_optimizer,
        "implicit_teacher_study": implicit_teacher_study,
        "null_study": null_study,
    }
    checks = [
        _check(name, int(_passed(value)), "==", 1, _passed(value))
        for name, value in components.items()
    ]
    checks.append(
        _check(
            "probe_bank_status",
            status.value,
            "==",
            ProbeBankStatus.AGREE.value,
            status is ProbeBankStatus.AGREE,
        )
    )
    result = _finish(
        "scale_repair_controls",
        checks,
        "boundary-admissible, optimizer-healthy synthetic implicit-score controls",
    )
    result["probe_bank_status"] = status.value
    result["components"] = {
        name: dict(value) if isinstance(value, Mapping) else int(bool(value))
        for name, value in components.items()
    }
    return result


def decide_scale_repair(
    *,
    provenance_pass: bool | int | Mapping[str, Any],
    boundary_preflight: bool | int | Mapping[str, Any],
    supervised_calibration: bool | int | Mapping[str, Any],
    implicit_calibration: bool | int | Mapping[str, Any],
    supervised_optimizer: bool | int | Mapping[str, Any],
    supervised_representation: bool | int | Mapping[str, Any],
    downstream_optimizer: bool | int | Mapping[str, Any],
    implicit_teacher_study: bool | int | Mapping[str, Any],
    null_study: bool | int | Mapping[str, Any],
    probe_bank_status: ProbeBankStatus | str = ProbeBankStatus.NOT_EVALUATED,
) -> dict[str, Any]:
    status = _probe_status(probe_bank_status)
    studies_evaluated = (
        _evaluation_status(implicit_teacher_study) == "evaluated"
        and _evaluation_status(null_study) == "evaluated"
    )

    if not _passed(provenance_pass):
        decision = ScaleRepairDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the parent/transitive provenance binding"
    elif not _passed(boundary_preflight):
        decision = ScaleRepairDecision.BOUNDARY_DOMAIN_INVALID
        action = "repair the model/operator boundary domain"
    elif not (
        _passed(supervised_calibration)
        and _passed(supervised_optimizer)
    ):
        decision = ScaleRepairDecision.OPTIMIZER_SCALE_INVALID
        action = "repair supervised loss-scale calibration or optimizer health"
    elif not _passed(supervised_representation):
        decision = ScaleRepairDecision.REPRESENTATION_INVALID
        action = "repair the boundary-smooth representation without changing scientific thresholds"
    elif not _passed(implicit_calibration):
        decision = ScaleRepairDecision.OPTIMIZER_SCALE_INVALID
        action = "repair the shared implicit/null loss-scale calibration"
    elif (
        _evaluation_status(downstream_optimizer) == "evaluated"
        and not _passed(downstream_optimizer)
    ):
        decision = ScaleRepairDecision.OPTIMIZER_SCALE_INVALID
        action = "repair implicit/null optimizer health without changing scientific thresholds"
    elif not studies_evaluated or status is ProbeBankStatus.NOT_EVALUATED:
        decision = ScaleRepairDecision.IMPLICIT_OBJECTIVE_UNSTABLE
        action = "complete both implicit and null studies before making a control claim"
    elif status is ProbeBankStatus.DISAGREE and _passed(implicit_teacher_study):
        decision = ScaleRepairDecision.TRACE_ESTIMATOR_INCONCLUSIVE
        action = "increase or redesign trace probes without changing the target law"
    elif not _passed(implicit_teacher_study) or not _passed(null_study):
        decision = ScaleRepairDecision.IMPLICIT_OBJECTIVE_UNSTABLE
        action = "use the predeclared density-ratio fallback or repair the implicit objective"
    else:
        decision = ScaleRepairDecision.CONTROL_PIPELINE_REPAIRED
        action = "plan a fresh physical-score run with untouched audit paths"

    return {
        "decision": decision.value,
        "recommended_next_action": action,
        "probe_bank_status": status.value,
        "studies_evaluated": int(studies_evaluated),
        "physical_training_authorized": int(
            decision is ScaleRepairDecision.CONTROL_PIPELINE_REPAIRED
        ),
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


def evaluate_scale_repair_gates(
    *,
    provenance_pass: bool | int | Mapping[str, Any],
    boundary_preflight: bool | int | Mapping[str, Any],
    supervised_calibration: bool | int | Mapping[str, Any],
    implicit_calibration: bool | int | Mapping[str, Any],
    supervised_optimizer: bool | int | Mapping[str, Any],
    supervised_representation: bool | int | Mapping[str, Any],
    downstream_optimizer: bool | int | Mapping[str, Any],
    implicit_teacher_study: bool | int | Mapping[str, Any],
    null_study: bool | int | Mapping[str, Any],
    probe_bank_status: ProbeBankStatus | str = ProbeBankStatus.NOT_EVALUATED,
    require_gate: str = "none",
) -> dict[str, Any]:
    """Evaluate the cumulative v2 repair gates and closed terminal decision."""

    required = str(require_gate)
    if required not in {"none", "preflight", "controls"}:
        raise ValueError("require_gate must be none, preflight, or controls")
    kwargs = {
        "provenance_pass": provenance_pass,
        "boundary_preflight": boundary_preflight,
        "supervised_calibration": supervised_calibration,
        "implicit_calibration": implicit_calibration,
        "supervised_optimizer": supervised_optimizer,
        "supervised_representation": supervised_representation,
        "downstream_optimizer": downstream_optimizer,
        "implicit_teacher_study": implicit_teacher_study,
        "null_study": null_study,
        "probe_bank_status": probe_bank_status,
    }
    controls = evaluate_scale_repair_gate(**kwargs)
    preflight_pass = _passed(provenance_pass) and _passed(boundary_preflight)
    requirement = {
        "none": True,
        "preflight": preflight_pass,
        "controls": _passed(controls),
    }[required]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "required_gate": required,
        "required_gate_pass": int(requirement),
        "preflight_pass": int(preflight_pass),
        "controls": controls,
        "decision": decide_scale_repair(**kwargs),
        "scientific_thresholds": asdict(BoundaryControlThresholds()),
        "sampling_performed": 0,
    }
