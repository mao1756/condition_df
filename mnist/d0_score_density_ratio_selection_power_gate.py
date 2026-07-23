"""Pure gates for oracle-qualified density-ratio evidence panels.

This additive module leaves the normalized-head model and its scientific gates
unchanged.  It adds an evidence-power contract around them:

* the saved sixteen-path oracle forensic must be reproducible;
* an independent 256-path calibration panel and its predetermined halves must
  detect the exact Bayes classifier;
* every fixed A/B (and, for confirmation, C/D) teacher panel must detect that
  same exact classifier *before* an optimizer step is taken.

The conservative legacy null gate remains authorization-critical.  A positive
discovery-panel-A bound followed by rejection on sealed panel B is additionally
reported as a multiplicity result; it never turns a failed null gate into a
passing gate.

There is deliberately no filesystem, model, training, or sampling code here.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from mnist.d0_score_density_ratio_head_gate import (
    HeadCoordinateThresholds,
    evaluate_head_pilot_candidate as evaluate_frozen_pilot_candidate,
    evaluate_null_seed as evaluate_frozen_null_seed,
    evaluate_teacher_seed as evaluate_frozen_teacher_seed,
    select_head_profile,
)


__all__ = [
    "SelectionPowerDecision",
    "SelectionPowerThresholds",
    "not_evaluated_gate",
    "evaluate_saved_oracle_forensic",
    "evaluate_oracle_panel_power",
    "evaluate_oracle_calibration",
    "evaluate_oracle_panel_set",
    "evaluate_selection_power_preflight",
    "evaluate_power_pilot_candidate",
    "evaluate_power_pilot",
    "analyze_null_multiplicity",
    "evaluate_power_teacher_seed",
    "evaluate_power_teacher_study",
    "evaluate_power_null_seed",
    "evaluate_power_null_study",
    "evaluate_power_controls",
    "decide_selection_power",
    "evaluate_selection_power_workflow",
]


SCHEMA = "experiment12-d0-score-density-ratio-selection-power-gate"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SelectionPowerThresholds:
    """Frozen evidence sizes plus the unchanged normalized-head thresholds."""

    head: HeadCoordinateThresholds = field(default_factory=HeadCoordinateThresholds)
    calibration_paths: int = 256
    calibration_half_paths: int = 128
    evidence_panel_paths: int = 128
    anchors_per_path: int = 32
    calibration_confidence: float = 0.99
    evidence_confidence: float = 0.90
    bootstrap_replicates: int = 10_000
    saved_forensic_paths: int = 16
    saved_forensic_tolerance: float = 5e-5
    saved_a_lower_bounds: tuple[float, float] = (
        0.00872031148,
        0.00517841895,
    )
    saved_b_lower_bounds: tuple[float, float] = (
        -0.00913743689,
        -0.00565862046,
    )

    def __post_init__(self) -> None:
        if self.head != HeadCoordinateThresholds():
            raise ValueError("normalized-head scientific thresholds must remain frozen")
        if int(self.calibration_paths) != 256:
            raise ValueError("calibration_paths are frozen at 256")
        if int(self.calibration_half_paths) != 128:
            raise ValueError("calibration_half_paths are frozen at 128")
        if int(self.evidence_panel_paths) != 128:
            raise ValueError("evidence_panel_paths are frozen at 128")
        if int(self.anchors_per_path) != 32:
            raise ValueError("anchors_per_path are frozen at 32")
        if float(self.calibration_confidence) != 0.99:
            raise ValueError("calibration_confidence is frozen at 0.99")
        if float(self.evidence_confidence) != 0.90:
            raise ValueError("evidence_confidence is frozen at 0.90")
        if int(self.bootstrap_replicates) != 10_000:
            raise ValueError("bootstrap_replicates are frozen at 10000")
        if int(self.saved_forensic_paths) != 16:
            raise ValueError("saved_forensic_paths are frozen at 16")
        if not _finite(self.saved_forensic_tolerance) or float(
            self.saved_forensic_tolerance
        ) <= 0.0:
            raise ValueError("saved_forensic_tolerance must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelectionPowerDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    ORACLE_POWER_INVALID = "oracle_power_invalid"
    EVIDENCE_PANEL_UNDERPOWERED = "evidence_panel_underpowered"
    NULL_GATE_MULTIPLICITY_INCONCLUSIVE = "null_gate_multiplicity_inconclusive"
    CLASSIFICATION_POWER_CONFIRMATION_UNRESOLVED = (
        "classification_power_confirmation_unresolved"
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
        return _one(value.get("passed", value.get("gate_pass", 0)))
    return value is True or (isinstance(value, int) and value == 1)


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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first(value: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in value:
            return value[name]
    return default


def _scope(record: Mapping[str, Any], name: str) -> dict[str, Any]:
    direct = record.get(name)
    if isinstance(direct, Mapping):
        return dict(direct)
    for parent_name in (
        "classification_improvement",
        "objective_improvement",
        "bootstrap",
        "bounds",
    ):
        parent = record.get(parent_name)
        if isinstance(parent, Mapping) and isinstance(parent.get(name), Mapping):
            return dict(parent[name])
    return {}


def _lower_bound(record: Mapping[str, Any]) -> float | None:
    value = _first(
        record,
        "lower_bound",
        "improvement_lower_bound",
        "classification_improvement_lower_bound",
        "objective_improvement_lower_bound",
        "bootstrap_lower_bound",
    )
    return float(value) if _finite(value) else None


def _two_lower_bounds(record: Mapping[str, Any]) -> list[float | None]:
    raw = record.get("lower_bounds")
    if isinstance(raw, Mapping):
        values = [raw.get("overall"), raw.get("data_end")]
        return [float(value) if _finite(value) else None for value in values]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
        if len(values) >= 2:
            return [
                float(values[0]) if _finite(values[0]) else None,
                float(values[1]) if _finite(values[1]) else None,
            ]
    return [_lower_bound(_scope(record, name)) for name in ("overall", "data_end")]


def evaluate_saved_oracle_forensic(
    metrics: Mapping[str, Any],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    """Require reproduction of the immutable sixteen-path oracle forensic."""

    thresholds = thresholds or SelectionPowerThresholds()
    a = _mapping(metrics.get("panel_a", metrics.get("a", {})))
    b = _mapping(metrics.get("panel_b", metrics.get("b", {})))
    a_bounds = _two_lower_bounds(a)
    b_bounds = _two_lower_bounds(b)
    tolerance = thresholds.saved_forensic_tolerance

    def close(values: Sequence[float | None], expected: Sequence[float]) -> bool:
        return len(values) == 2 and all(
            value is not None and abs(float(value) - float(target)) <= tolerance
            for value, target in zip(values, expected)
        )

    checks = [
        _check("complete", metrics.get("complete", 1), "==", 1, _one(metrics.get("complete", 1))),
        _check("finite", metrics.get("finite"), "==", 1, _one(metrics.get("finite"))),
        _check(
            "path_count",
            metrics.get("path_count"),
            "==",
            thresholds.saved_forensic_paths,
            metrics.get("path_count") is not None
            and int(metrics["path_count"]) == thresholds.saved_forensic_paths,
        ),
        _check(
            "anchors_per_path",
            metrics.get("anchors_per_path"),
            "==",
            thresholds.anchors_per_path,
            metrics.get("anchors_per_path") is not None
            and int(metrics["anchors_per_path"]) == thresholds.anchors_per_path,
        ),
        _check(
            "saved_panel_hashes_verified",
            metrics.get("saved_panel_hashes_verified"),
            "==",
            1,
            _one(metrics.get("saved_panel_hashes_verified")),
        ),
        _check(
            "panel_a_lower_bounds",
            a_bounds,
            "within absolute tolerance",
            {"expected": list(thresholds.saved_a_lower_bounds), "atol": tolerance},
            close(a_bounds, thresholds.saved_a_lower_bounds),
        ),
        _check(
            "panel_b_lower_bounds",
            b_bounds,
            "within absolute tolerance",
            {"expected": list(thresholds.saved_b_lower_bounds), "atol": tolerance},
            close(b_bounds, thresholds.saved_b_lower_bounds),
        ),
    ]
    result = _finish(
        "saved_sixteen_path_oracle_forensic",
        checks,
        "reproduction of the inspected immutable sixteen-path exact-oracle evidence",
        evaluation_status=_status(metrics),
    )
    result.update({"panel_a_lower_bounds": a_bounds, "panel_b_lower_bounds": b_bounds})
    return result


def evaluate_oracle_panel_power(
    metrics: Mapping[str, Any],
    *,
    expected_path_count: int,
    required_confidence: float,
    gate_name: str = "exact_teacher_oracle_panel_power",
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    """Gate one fixed panel using exact-Bayes whole-path lower bounds."""

    thresholds = thresholds or SelectionPowerThresholds()
    bounds = _two_lower_bounds(metrics)
    confidence = _first(metrics, "confidence", "bootstrap_confidence")
    checks = [
        _check("complete", metrics.get("complete", 1), "==", 1, _one(metrics.get("complete", 1))),
        _check("finite", metrics.get("finite"), "==", 1, _one(metrics.get("finite"))),
        _check(
            "path_count",
            metrics.get("path_count"),
            "==",
            int(expected_path_count),
            metrics.get("path_count") is not None
            and int(metrics["path_count"]) == int(expected_path_count),
        ),
        _check(
            "anchors_per_path",
            metrics.get("anchors_per_path"),
            "==",
            thresholds.anchors_per_path,
            metrics.get("anchors_per_path") is not None
            and int(metrics["anchors_per_path"]) == thresholds.anchors_per_path,
        ),
        _check(
            "bootstrap_replicates",
            metrics.get("bootstrap_replicates"),
            "==",
            thresholds.bootstrap_replicates,
            metrics.get("bootstrap_replicates") is not None
            and int(metrics["bootstrap_replicates"])
            == thresholds.bootstrap_replicates,
        ),
        _check(
            "confidence",
            confidence,
            ">=",
            float(required_confidence),
            _finite(confidence) and float(confidence) >= float(required_confidence),
        ),
        _check(
            "lower_bounds",
            bounds,
            "> 0 each",
            0.0,
            len(bounds) == 2
            and all(value is not None and float(value) > 0.0 for value in bounds),
        ),
    ]
    result = _finish(
        gate_name,
        checks,
        "exact bounded-teacher log-density-ratio detection on a frozen whole-path panel",
        evaluation_status=_status(metrics),
    )
    result["lower_bounds"] = bounds
    return result


def evaluate_oracle_calibration(
    record: Mapping[str, Any],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    """Gate the independent 256-path calibration panel and fixed halves."""

    thresholds = thresholds or SelectionPowerThresholds()
    full_metrics = _mapping(record.get("full", record.get("panel", {})))
    halves_raw = record.get("halves", [])
    halves = [dict(value) for value in halves_raw if isinstance(value, Mapping)]
    full = evaluate_oracle_panel_power(
        full_metrics,
        expected_path_count=thresholds.calibration_paths,
        required_confidence=thresholds.calibration_confidence,
        gate_name="oracle_calibration_full_256",
        thresholds=thresholds,
    )
    half_gates = [
        evaluate_oracle_panel_power(
            value,
            expected_path_count=thresholds.calibration_half_paths,
            required_confidence=thresholds.evidence_confidence,
            gate_name=f"oracle_calibration_half_{index}",
            thresholds=thresholds,
        )
        for index, value in enumerate(halves)
    ]
    checks = [
        _check("full_panel", full["passed"], "==", 1, _passed(full)),
        _check("half_count", len(half_gates), "==", 2, len(half_gates) == 2),
        _check(
            "both_predetermined_halves",
            sum(_passed(value) for value in half_gates),
            "==",
            2,
            len(half_gates) == 2 and all(_passed(value) for value in half_gates),
        ),
        _check(
            "predetermined_split",
            record.get("predetermined_split"),
            "==",
            1,
            _one(record.get("predetermined_split")),
        ),
        _check(
            "halves_disjoint",
            record.get("halves_disjoint"),
            "==",
            1,
            _one(record.get("halves_disjoint")),
        ),
        _check(
            "evaluation_overlap_path_count",
            record.get("evaluation_overlap_path_count"),
            "==",
            0,
            record.get("evaluation_overlap_path_count") is not None
            and int(record["evaluation_overlap_path_count"]) == 0,
        ),
        _check(
            "panel_frozen_before_inspection",
            record.get("panel_frozen_before_inspection"),
            "==",
            1,
            _one(record.get("panel_frozen_before_inspection")),
        ),
        _check(
            "regenerated_after_inspection",
            record.get("regenerated_after_inspection"),
            "==",
            0,
            record.get("regenerated_after_inspection") is not None
            and int(record["regenerated_after_inspection"]) == 0,
        ),
    ]
    result = _finish(
        "oracle_power_calibration",
        checks,
        "independent exact-oracle calibration; never model selection or audit data",
        evaluation_status=_status(record),
    )
    result.update({"full_panel_gate": full, "half_panel_gates": half_gates})
    return result


def evaluate_oracle_panel_set(
    record: Mapping[str, Any],
    *,
    expected_roles: Sequence[str],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    """Oracle-qualify immutable A/B or A/B/C/D panels before training."""

    thresholds = thresholds or SelectionPowerThresholds()
    roles = tuple(str(value).lower() for value in expected_roles)
    raw_panels = _mapping(record.get("panels", {}))
    panels = {str(key).lower(): _mapping(value) for key, value in raw_panels.items()}
    panel_gates = {
        role: evaluate_oracle_panel_power(
            panels.get(role, {}),
            expected_path_count=thresholds.evidence_panel_paths,
            required_confidence=thresholds.evidence_confidence,
            gate_name=f"exact_teacher_oracle_panel_{role}",
            thresholds=thresholds,
        )
        for role in roles
    }
    checks = [
        _check(
            "exact_role_set",
            sorted(panels),
            "==",
            sorted(roles),
            set(panels) == set(roles),
        ),
        _check(
            "all_panels_powered",
            sum(_passed(value) for value in panel_gates.values()),
            "==",
            len(roles),
            len(panel_gates) == len(roles)
            and all(_passed(value) for value in panel_gates.values()),
        ),
        _check(
            "pairwise_disjoint",
            record.get("pairwise_disjoint"),
            "==",
            1,
            _one(record.get("pairwise_disjoint")),
        ),
        _check(
            "calibration_overlap_path_count",
            record.get("calibration_overlap_path_count"),
            "==",
            0,
            record.get("calibration_overlap_path_count") is not None
            and int(record["calibration_overlap_path_count"]) == 0,
        ),
        _check(
            "frozen_before_training",
            record.get("frozen_before_training"),
            "==",
            1,
            _one(record.get("frozen_before_training")),
        ),
        _check(
            "optimizer_steps_before_oracle_gate",
            record.get("optimizer_steps_before_oracle_gate"),
            "==",
            0,
            record.get("optimizer_steps_before_oracle_gate") is not None
            and int(record["optimizer_steps_before_oracle_gate"]) == 0,
        ),
        _check(
            "regenerated_after_inspection",
            record.get("regenerated_after_inspection"),
            "==",
            0,
            record.get("regenerated_after_inspection") is not None
            and int(record["regenerated_after_inspection"]) == 0,
        ),
    ]
    result = _finish(
        "oracle_qualified_" + "".join(roles) + "_panel_set",
        checks,
        "pre-optimizer feasibility of every immutable teacher evidence panel",
        evaluation_status=_status(record),
    )
    result.update({"expected_roles": list(roles), "panel_gates": panel_gates})
    return result


def evaluate_selection_power_preflight(
    *,
    normalized_head_preflight: bool | int | Mapping[str, Any],
    saved_forensic: Mapping[str, Any],
    calibration: Mapping[str, Any],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    forensic_gate = (
        dict(saved_forensic)
        if saved_forensic.get("gate") == "saved_sixteen_path_oracle_forensic"
        else evaluate_saved_oracle_forensic(saved_forensic, thresholds)
    )
    calibration_gate = (
        dict(calibration)
        if calibration.get("gate") == "oracle_power_calibration"
        else evaluate_oracle_calibration(calibration, thresholds)
    )
    checks = [
        _check(
            "normalized_head_preflight",
            int(_passed(normalized_head_preflight)),
            "==",
            1,
            _passed(normalized_head_preflight),
        ),
        _check("saved_oracle_forensic", forensic_gate["passed"], "==", 1, _passed(forensic_gate)),
        _check("oracle_calibration", calibration_gate["passed"], "==", 1, _passed(calibration_gate)),
    ]
    result = _finish(
        "selection_power_preflight",
        checks,
        "inherited coordinate validity plus exact-oracle evidence calibration",
        evaluation_status=(
            "evaluated"
            if _status(forensic_gate) == _status(calibration_gate) == "evaluated"
            else "not_evaluated"
        ),
    )
    result.update(
        {
            "normalized_head_preflight": dict(normalized_head_preflight)
            if isinstance(normalized_head_preflight, Mapping)
            else int(_passed(normalized_head_preflight)),
            "saved_oracle_forensic": forensic_gate,
            "oracle_calibration": calibration_gate,
        }
    )
    return result


def evaluate_power_pilot_candidate(
    candidate: Mapping[str, Any],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    base = evaluate_frozen_pilot_candidate(candidate, thresholds.head)
    result = dict(base)
    result["gate"] = "selection_power_pilot_candidate"
    result["claim_scope"] = (
        "unchanged normalized-head candidate on oracle-qualified 128-path panels"
    )
    result["frozen_normalized_head_gate"] = dict(base)
    return result


def _find_gate(value: Any, gate_name: str) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        if value.get("gate") == gate_name:
            return dict(value)
        for child in value.values():
            found = _find_gate(child, gate_name)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found = _find_gate(child, gate_name)
            if found is not None:
                return found
    return None


def analyze_null_multiplicity(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Advisory analysis of A-only null excursions.

    The result is intentionally non-authorizing.  ``a_only_explains_failure``
    is true only when at least one candidate failed and every failed candidate
    differs from the conservative legacy gate solely at
    ``null_panel_a_lower_bounds``, with its sealed B nominee rejected.
    """

    rows: list[dict[str, Any]] = []
    failed_rows = 0
    a_only_rows = 0
    for index, candidate in enumerate(candidates):
        base = _find_gate(candidate, "density_ratio_pilot_candidate")
        if base is None:
            rows.append(
                {
                    "candidate_index": index,
                    "evaluated": 0,
                    "a_only_null_violation": 0,
                    "failed_subchecks": [],
                }
            )
            continue
        subchecks = _mapping(base.get("subchecks", {}))
        failures = sorted(
            name for name, check in subchecks.items() if not _passed(_mapping(check))
        )
        selection = _mapping(base.get("null_selection", {}))
        confirmation = _mapping(selection.get("confirmation", {}))
        b_bounds_raw = confirmation.get("panel_b_lower_bounds", [])
        b_bounds = (
            [float(value) if _finite(value) else None for value in b_bounds_raw]
            if isinstance(b_bounds_raw, Sequence)
            and not isinstance(b_bounds_raw, (str, bytes))
            else []
        )
        b_rejected = (
            not _one(confirmation.get("accepted", 0))
            and int(selection.get("selected_step", -1)) == 0
            and len(b_bounds) == 2
            and all(value is not None and value <= 0.0 for value in b_bounds)
        )
        a_only = failures == ["null_panel_a_lower_bounds"] and b_rejected
        if failures:
            failed_rows += 1
        if a_only:
            a_only_rows += 1
        rows.append(
            {
                "candidate_index": index,
                "evaluated": 1,
                "a_only_null_violation": int(a_only),
                "sealed_panel_b_rejected": int(b_rejected),
                "failed_subchecks": failures,
            }
        )
    explains = failed_rows > 0 and a_only_rows == failed_rows
    return {
        "schema": SCHEMA + "-null-multiplicity-analysis",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated" if candidates else "not_evaluated",
        "authorizing": 0,
        "candidate_count": len(candidates),
        "failed_candidate_count": failed_rows,
        "a_only_candidate_count": a_only_rows,
        "a_only_explains_failure": int(explains),
        "rows": rows,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_power_pilot(
    candidates: Sequence[Mapping[str, Any]],
    *,
    panel_power: bool | int | Mapping[str, Any],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "selection_power_pilot_candidate"
        else evaluate_power_pilot_candidate(value, thresholds)
        for value in candidates
    ]
    rates = [value.get("learning_rate") for value in gates]
    accumulations = [value.get("accumulation_steps") for value in gates]
    profile = select_head_profile(gates, thresholds.head)
    multiplicity = analyze_null_multiplicity(gates)
    checks = [
        _check("oracle_qualified_panels", int(_passed(panel_power)), "==", 1, _passed(panel_power)),
        _check("candidate_count", len(gates), "==", 2, len(gates) == 2),
        _check(
            "body_learning_rate_set",
            sorted(float(value) for value in rates if _finite(value)),
            "==",
            sorted(thresholds.head.pilot_learning_rates),
            len(rates) == 2
            and sorted(float(value) for value in rates if _finite(value))
            == sorted(thresholds.head.pilot_learning_rates),
        ),
        _check(
            "all_accumulation_eight",
            accumulations,
            "== each",
            thresholds.head.accumulation_steps,
            len(accumulations) == 2
            and all(
                value is not None
                and int(value) == thresholds.head.accumulation_steps
                for value in accumulations
            ),
        ),
        _check("eligible_profile", profile["selected"], "==", 1, _one(profile["selected"])),
    ]
    status = "evaluated" if gates else "not_evaluated"
    result = _finish(
        "selection_power_pilot",
        checks,
        "fixed 128-path A/B train-selection pilot after exact-oracle qualification",
        evaluation_status=status,
    )
    result.update(
        {
            "candidate_gates": gates,
            "selected_profile": profile,
            "oracle_panel_power": dict(panel_power)
            if isinstance(panel_power, Mapping)
            else int(_passed(panel_power)),
            "null_multiplicity_analysis": multiplicity,
            "optimizer_health_pass": int(
                bool(gates)
                and all(bool(int(value.get("optimizer_health_pass", 0))) for value in gates)
            ),
        }
    )
    return result


def _metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = value.get("metrics", value)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _with_legacy_audit_path_count(
    value: Mapping[str, Any],
    thresholds: SelectionPowerThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt only the old size check; all scientific values remain untouched."""

    raw = copy.deepcopy(_metrics(value))
    panels = _mapping(raw.get("audit_panels", {}))
    size_checks: dict[str, Any] = {}
    legacy_count = thresholds.head.stability.density_ratio.audit_paths_per_panel
    for role in thresholds.head.stability.density_ratio.expected_audit_panels:
        panel = _mapping(panels.get(role, {}))
        actual = panel.get("path_count")
        anchors = panel.get("anchors_per_path")
        size_checks[role] = {
            "path_count": actual,
            "anchors_per_path": anchors,
            "passed": int(
                actual is not None
                and int(actual) == thresholds.evidence_panel_paths
                and anchors is not None
                and int(anchors) == thresholds.anchors_per_path
            ),
        }
        # The inherited gate hard-codes the old 32-path experiment.  Replacing
        # this metadata value is safe because the new explicit size gate above
        # proves 128 paths and no metric or bound is changed.
        if panel:
            panel["path_count"] = legacy_count
            panels[role] = panel
    raw["audit_panels"] = panels
    return raw, size_checks


def _wrap_confirmation_seed(
    value: Mapping[str, Any],
    *,
    teacher: bool,
    thresholds: SelectionPowerThresholds,
) -> dict[str, Any]:
    adapted, sizes = _with_legacy_audit_path_count(value, thresholds)
    base = (
        evaluate_frozen_teacher_seed(adapted, thresholds.head)
        if teacher
        else evaluate_frozen_null_seed(adapted, thresholds.head)
    )
    checks = [
        _check("frozen_scientific_gate", base.get("passed"), "==", 1, _passed(base)),
        *[
            _check(
                f"audit_panel_{role}_size",
                {"path_count": row.get("path_count"), "anchors_per_path": row.get("anchors_per_path")},
                "==",
                {"path_count": thresholds.evidence_panel_paths, "anchors_per_path": thresholds.anchors_per_path},
                _one(row.get("passed")),
            )
            for role, row in sizes.items()
        ],
    ]
    name = "selection_power_teacher_seed" if teacher else "selection_power_null_seed"
    result = _finish(
        name,
        checks,
        "strict frozen confirmation science on powered 128-path audit panels",
        evaluation_status=_status(value),
    )
    result.update(
        {
            key: val
            for key, val in base.items()
            if key
            not in {"gate", "passed", "subchecks", "claim_scope", "evaluation_status"}
        }
    )
    result["frozen_normalized_head_gate"] = base
    result["audit_panel_size_checks"] = sizes
    return result


def evaluate_power_teacher_seed(
    value: Mapping[str, Any],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    return _wrap_confirmation_seed(value, teacher=True, thresholds=thresholds)


def evaluate_power_null_seed(
    value: Mapping[str, Any],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    return _wrap_confirmation_seed(value, teacher=False, thresholds=thresholds)


def evaluate_power_teacher_study(
    values: Sequence[Mapping[str, Any]],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "selection_power_teacher_seed"
        else evaluate_power_teacher_seed(value, thresholds)
        for value in values
    ]
    expected = thresholds.head.stability.density_ratio.expected_teacher_seeds
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
            thresholds.head.stability.density_ratio.minimum_passing_teacher_seeds,
            passing
            >= thresholds.head.stability.density_ratio.minimum_passing_teacher_seeds,
        ),
        _check("audit_panels_agree", int(not disagreement), "==", 1, not disagreement),
    ]
    result = _finish(
        "selection_power_teacher_study",
        checks,
        "three-seed exact-oracle-qualified bounded-teacher confirmation",
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


def evaluate_power_null_study(
    values: Sequence[Mapping[str, Any]],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    gates = [
        dict(value)
        if value.get("gate") == "selection_power_null_seed"
        else evaluate_power_null_seed(value, thresholds)
        for value in values
    ]
    expected = thresholds.head.stability.density_ratio.expected_null_seeds
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
        "selection_power_null_study",
        checks,
        "three-seed conservative stationary-null confirmation",
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


def evaluate_power_controls(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot: bool | int | Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    teacher = evaluate_power_teacher_study(teacher_results, thresholds)
    null = evaluate_power_null_study(null_results, thresholds)
    teacher_seeds = {value.get("model_seed") for value in teacher.get("seed_gates", [])}
    null_seeds = {value.get("model_seed") for value in null.get("seed_gates", [])}
    expected = thresholds.head.stability.density_ratio.expected_teacher_seeds
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
        _check(
            "oracle_qualified_confirmation_panels",
            int(_passed(confirmation_panel_power)),
            "==",
            1,
            _passed(confirmation_panel_power),
        ),
        _check("paired_teacher_null_seed_set", int(paired), "==", 1, paired),
        _check("teacher_study", teacher["passed"], "==", 1, _passed(teacher)),
        _check("null_study", null["passed"], "==", 1, _passed(null)),
    ]
    result = _finish(
        "selection_power_controls",
        checks,
        "strict derivative-accurate controls on oracle-qualified 128-path panels",
        evaluation_status="evaluated" if teacher_results or null_results else "not_evaluated",
    )
    result.update(
        {
            "teacher_study": teacher,
            "null_study": null,
            "confirmation_panel_power": dict(confirmation_panel_power)
            if isinstance(confirmation_panel_power, Mapping)
            else int(_passed(confirmation_panel_power)),
            "paired_teacher_null_seed_set_pass": int(paired),
            "optimizer_health_pass": int(
                bool(int(teacher.get("optimizer_health_pass", 0)))
                and bool(int(null.get("optimizer_health_pass", 0)))
            ),
        }
    )
    return result


def decide_selection_power(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot_panel_power: bool | int | Mapping[str, Any],
    pilot: Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    teacher = _mapping(controls.get("teacher_study", {}))
    null = _mapping(controls.get("null_study", {}))
    multiplicity = _mapping(pilot.get("null_multiplicity_analysis", {}))
    interim = False
    h1_authorized = False
    if not _passed(provenance):
        decision: SelectionPowerDecision | str = SelectionPowerDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the exact 125-record parent and 332-to-222-to-381 provenance binding"
    elif _status(preflight) != "evaluated":
        decision = "selection_power_preflight_not_evaluated"
        action = "run the oracle-power preflight"
        interim = True
    elif not _passed(preflight):
        decision = SelectionPowerDecision.ORACLE_POWER_INVALID
        action = "repair exact-oracle reproduction, calibration, or inherited coordinate validity"
    elif _status(pilot_panel_power) != "evaluated":
        decision = "oracle_power_preflight_passed"
        action = "freeze and oracle-qualify the 128-path pilot A/B panels"
        interim = True
    elif not _passed(pilot_panel_power):
        decision = SelectionPowerDecision.EVIDENCE_PANEL_UNDERPOWERED
        action = "stop before training; preserve the fixed underpowered panel evidence"
    elif _status(pilot) != "evaluated":
        decision = "selection_power_preflight_passed"
        action = "run the unchanged normalized-head pilot"
        interim = True
    elif not _passed(pilot):
        if _one(multiplicity.get("a_only_explains_failure")):
            decision = SelectionPowerDecision.NULL_GATE_MULTIPLICITY_INCONCLUSIVE
            action = "retain the conservative null failure and plan a separately multiplicity-aware confirmation"
        elif not _one(pilot.get("optimizer_health_pass", 0)):
            decision = SelectionPowerDecision.CLASSIFICATION_OPTIMIZER_INVALID
            action = "repair incomplete, nonfinite, clipped, or incompatible pilot tasks"
        else:
            decision = SelectionPowerDecision.CLASSIFICATION_POWER_CONFIRMATION_UNRESOLVED
            action = "plan the predeclared H1-like function-step trust patch"
            h1_authorized = True
    elif _status(confirmation_panel_power) == "evaluated" and not _passed(
        confirmation_panel_power
    ):
        decision = SelectionPowerDecision.EVIDENCE_PANEL_UNDERPOWERED
        action = "stop before confirmation training; preserve the fixed underpowered panel evidence"
    elif _status(controls) != "evaluated":
        decision = "selection_power_pilot_passed"
        action = "oracle-qualify A/B/C/D and run fresh three-seed confirmation"
        interim = True
    elif not bool(int(controls.get("optimizer_health_pass", 0))):
        decision = SelectionPowerDecision.CLASSIFICATION_OPTIMIZER_INVALID
        action = "repair incomplete, nonfinite, clipped, or incompatible confirmation tasks"
    elif int(null.get("false_discovery_count", 0)) > 0:
        decision = SelectionPowerDecision.SELECTION_FALSE_DISCOVERY
        action = "repair discovery/confirmation calibration before score learning"
    elif bool(int(teacher.get("panel_disagreement", 0))):
        decision = SelectionPowerDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "rerun the frozen powered confirmation without changing thresholds"
    elif _passed(controls):
        decision = SelectionPowerDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        action = "plan fresh physical one-image density-ratio score training"
    elif int(teacher.get("classification_passing_seed_count", 0)) >= 2:
        decision = SelectionPowerDecision.DENSITY_RATIO_VALUE_ONLY
        action = "investigate derivative learning; do not shrink the model"
    else:
        decision = SelectionPowerDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL
        action = "revisit classifier learning on the exact bounded synthetic law"
    value = decision.value if isinstance(decision, SelectionPowerDecision) else decision
    return {
        "decision": value,
        "recommended_next_action": action,
        "interim_stage_success": int(interim and value.endswith("_passed")),
        "closed_terminal_scientific_outcome": int(not interim),
        "h1_function_step_patch_authorized": int(h1_authorized),
        "physical_training_authorized": int(
            decision is SelectionPowerDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        ),
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


def evaluate_selection_power_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight: bool | int | Mapping[str, Any],
    pilot_panel_power: bool | int | Mapping[str, Any],
    pilot: Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    require_gate: str = "none",
    thresholds: SelectionPowerThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SelectionPowerThresholds()
    if require_gate not in {"none", "preflight", "pilot", "controls"}:
        raise ValueError("require_gate must be none, preflight, pilot, or controls")
    controls = evaluate_power_controls(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        confirmation_panel_power=confirmation_panel_power,
        teacher_results=teacher_results,
        null_results=null_results,
        thresholds=thresholds,
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
    decision = decide_selection_power(
        provenance=provenance,
        preflight=preflight,
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
