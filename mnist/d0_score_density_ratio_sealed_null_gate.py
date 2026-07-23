"""Multiplicity-aware sealed-null statistics and pure workflow gates.

The density-ratio workflow uses panel A to scan checkpoints and nominate one
fixed function.  Treating the nominee's unadjusted panel-A interval as a
second confirmatory test is invalid because A has already been used for
selection.  This module repairs that bookkeeping additively:

* a deterministic, studentized whole-path max-T bootstrap supplies
  simultaneous lower bounds for explicitly enumerated null families;
* immutable parent candidates are replayed only when their legacy failure is
  exactly the panel-A bound and their independently sealed panel B rejects;
* confirmation null evidence is evaluated as a predeclared family while all
  optimizer, sealing, and analytic-zero semantics remain strict.

No filesystem, model, optimizer, or sampler code is imported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mnist.d0_score_density_ratio_selection_power_gate import (
    SelectionPowerThresholds,
)


__all__ = [
    "SealedNullDecision",
    "SealedNullThresholds",
    "studentized_whole_path_max_t",
    "evaluate_max_t_null_family",
    "evaluate_simultaneous_bootstrap_preflight",
    "evaluate_parent_replay_candidate",
    "select_recovered_profile",
    "evaluate_parent_pilot_replay",
    "evaluate_confirmation_null_family",
    "evaluate_sealed_null_controls",
    "decide_sealed_null_workflow",
    "evaluate_sealed_null_workflow",
    "not_evaluated_gate",
]


SCHEMA = "experiment12-d0-score-density-ratio-sealed-null-gate"
SCHEMA_VERSION = 1
MAX_T_VERSION = "studentized-whole-path-centered-bootstrap-max-t-v1"


@dataclass(frozen=True)
class SealedNullThresholds:
    """Frozen multiplicity and inherited density-ratio thresholds."""

    selection_power: SelectionPowerThresholds = field(
        default_factory=SelectionPowerThresholds
    )
    confidence: float = 0.95
    bootstrap_replicates: int = 50_000
    minimum_paths: int = 2
    standard_error_floor: float = 1e-15
    expected_pilot_candidates: int = 2
    expected_pilot_learning_rates: tuple[float, ...] = (3e-5, 1e-5)
    expected_accumulation_steps: int = 8
    expected_confirmation_null_seeds: int = 3

    def __post_init__(self) -> None:
        if self.selection_power != SelectionPowerThresholds():
            raise ValueError("selection-power scientific thresholds must remain frozen")
        if float(self.confidence) != 0.95:
            raise ValueError("max-T confidence is frozen at 0.95")
        if int(self.bootstrap_replicates) != 50_000:
            raise ValueError("bootstrap_replicates are frozen at 50000")
        if int(self.minimum_paths) < 2:
            raise ValueError("minimum_paths must be at least two")
        if not _finite(self.standard_error_floor) or float(
            self.standard_error_floor
        ) <= 0.0:
            raise ValueError("standard_error_floor must be finite and positive")
        if int(self.expected_pilot_candidates) != 2:
            raise ValueError("expected_pilot_candidates are frozen at two")
        if self.expected_pilot_learning_rates != (3e-5, 1e-5):
            raise ValueError("pilot learning rates are frozen at 3e-5,1e-5")
        if int(self.expected_accumulation_steps) != 8:
            raise ValueError("gradient accumulation is frozen at eight")
        if int(self.expected_confirmation_null_seeds) != 3:
            raise ValueError("confirmation null seeds are frozen at three")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SealedNullDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    SIMULTANEOUS_BOOTSTRAP_INVALID = "simultaneous_bootstrap_invalid"
    PROFILE_RECOVERY_INVALID = "profile_recovery_invalid"
    EVIDENCE_PANEL_UNDERPOWERED = "evidence_panel_underpowered"
    CLASSIFICATION_OPTIMIZER_INVALID = "classification_optimizer_invalid"
    SELECTION_FALSE_DISCOVERY = "selection_false_discovery"
    CLASSIFICATION_AUDIT_INCONCLUSIVE = "classification_audit_inconclusive"
    NO_DETECTABLE_DENSITY_RATIO_SIGNAL = "no_detectable_density_ratio_signal"
    CLASSIFICATION_POWER_CONFIRMATION_UNRESOLVED = (
        "classification_power_confirmation_unresolved"
    )
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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _canonical_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _derived_seed(root_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}\0{namespace}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _quantile_higher(values: np.ndarray, quantile: float) -> float:
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, quantile, interpolation="higher"))


def _canonicalize_family(
    members: Mapping[str, Mapping[str, Any]],
    *,
    minimum_paths: int,
) -> tuple[dict[str, tuple[list[Any], np.ndarray]], list[dict[str, Any]], str]:
    if not isinstance(members, Mapping) or not members:
        raise ValueError("max-T family must contain at least one member")
    blocks_raw: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for raw_name, raw_record in members.items():
        name = str(raw_name)
        record = _mapping(raw_record)
        block = str(record.get("resampling_block", record.get("block", name)))
        path_ids = record.get("path_ids")
        values = record.get("path_values", record.get("values"))
        if not isinstance(path_ids, Sequence) or isinstance(path_ids, (str, bytes)):
            raise ValueError(f"member {name!r} is missing path_ids")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"member {name!r} is missing path_values")
        if len(path_ids) != len(values) or len(path_ids) < int(minimum_paths):
            raise ValueError(f"member {name!r} has an invalid whole-path sample")
        keys = [_canonical_key(value) for value in path_ids]
        if len(set(keys)) != len(keys):
            raise ValueError(f"member {name!r} has duplicate path IDs")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"member {name!r} has nonfinite path values")
        record["_path_keys"] = keys
        record["_path_ids"] = list(path_ids)
        record["_values"] = array
        blocks_raw.setdefault(block, []).append((name, record))

    blocks: dict[str, tuple[list[Any], np.ndarray]] = {}
    member_meta: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    for block in sorted(blocks_raw):
        rows = sorted(blocks_raw[block], key=lambda item: item[0])
        reference_keys = sorted(rows[0][1]["_path_keys"])
        reference_set = set(reference_keys)
        canonical_ids_by_key = {
            key: path_id
            for key, path_id in zip(rows[0][1]["_path_keys"], rows[0][1]["_path_ids"])
        }
        columns: list[np.ndarray] = []
        names: list[str] = []
        metadata_by_name: dict[str, tuple[str, str]] = {}
        for name, record in rows:
            keys = list(record["_path_keys"])
            if set(keys) != reference_set:
                raise ValueError(
                    f"members in resampling block {block!r} must share path IDs"
                )
            lookup = {
                key: float(value)
                for key, value in zip(keys, np.asarray(record["_values"]))
            }
            ordered = np.asarray([lookup[key] for key in reference_keys], dtype=np.float64)
            columns.append(ordered)
            names.append(name)
            panel_role = str(
                record.get("panel_role", record.get("role", block))
            ).strip().lower()
            scope = str(record.get("scope", "")).strip().lower()
            metadata_by_name[name] = (panel_role, scope)
            fingerprint_rows.append(
                {
                    "name": name,
                    "block": block,
                    "panel_role": panel_role,
                    "scope": scope,
                    "path_ids": reference_keys,
                    "path_values": ordered.tolist(),
                }
            )
        matrix = np.column_stack(columns)
        canonical_ids = [canonical_ids_by_key[key] for key in reference_keys]
        blocks[block] = (canonical_ids, matrix)
        for column, name in enumerate(names):
            panel_role, scope = metadata_by_name[name]
            member_meta.append(
                {
                    "name": name,
                    "resampling_block": block,
                    "column": column,
                    "path_count": int(matrix.shape[0]),
                    "panel_role": panel_role,
                    "scope": scope,
                }
            )
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return blocks, member_meta, fingerprint


def studentized_whole_path_max_t(
    members: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    confidence: float = 0.95,
    reps: int = 50_000,
    minimum_paths: int = 2,
    standard_error_floor: float = 1e-15,
    bootstrap_chunk_size: int = 256,
) -> dict[str, Any]:
    """Return deterministic one-sided simultaneous whole-path lower bounds.

    Members sharing ``resampling_block`` must have the same path IDs and use
    common bootstrap indices, retaining their observed dependence.  Disjoint
    blocks receive independently derived stateless bootstrap streams.  Both
    member names and path IDs are canonicalized, so input ordering is
    irrelevant.
    """

    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if int(reps) <= 0:
        raise ValueError("reps must be positive")
    if int(minimum_paths) < 2:
        raise ValueError("minimum_paths must be at least two")
    if not _finite(standard_error_floor) or float(standard_error_floor) <= 0.0:
        raise ValueError("standard_error_floor must be finite and positive")
    if int(bootstrap_chunk_size) <= 0:
        raise ValueError("bootstrap_chunk_size must be positive")

    blocks, member_meta, fingerprint = _canonicalize_family(
        members, minimum_paths=int(minimum_paths)
    )
    observed_by_name: dict[str, dict[str, float | int | str]] = {}
    max_statistics = np.full(int(reps), -np.inf, dtype=np.float64)
    floor = float(standard_error_floor)

    for block in sorted(blocks):
        path_ids, matrix = blocks[block]
        n_paths, n_members = matrix.shape
        means = matrix.mean(axis=0)
        standard_deviations = matrix.std(axis=0, ddof=1)
        standard_errors = standard_deviations / math.sqrt(n_paths)
        centered = matrix - means[None, :]
        block_meta = sorted(
            (value for value in member_meta if value["resampling_block"] == block),
            key=lambda value: int(value["column"]),
        )
        for column, meta in enumerate(block_meta):
            mean = float(means[column])
            sd = float(standard_deviations[column])
            se = float(standard_errors[column])
            exact_zero = bool(np.count_nonzero(matrix[:, column]) == 0)
            if se > floor:
                statistic = mean / se
            elif exact_zero:
                statistic = 0.0
            else:
                raise ValueError(
                    f"member {meta['name']!r} has a nonzero degenerate "
                    "whole-path statistic"
                )
            observed_by_name[str(meta["name"])] = {
                "name": str(meta["name"]),
                "resampling_block": block,
                "panel_role": str(meta["panel_role"]),
                "scope": str(meta["scope"]),
                "path_count": int(n_paths),
                "mean": mean,
                "standard_deviation": sd,
                "standard_error": se,
                "t_statistic": float(statistic),
                "exact_zero_path_vector": int(exact_zero),
            }

        rng = np.random.default_rng(_derived_seed(int(seed), block))
        for start in range(0, int(reps), int(bootstrap_chunk_size)):
            stop = min(int(reps), start + int(bootstrap_chunk_size))
            indices = rng.integers(
                0, n_paths, size=(stop - start, n_paths), endpoint=False
            )
            samples = centered[indices, :]
            boot_means = samples.mean(axis=1)
            boot_sd = samples.std(axis=1, ddof=1)
            denominator = boot_sd / math.sqrt(n_paths)
            statistics = np.divide(
                boot_means,
                denominator,
                out=np.zeros_like(boot_means),
                where=denominator > floor,
            )
            max_statistics[start:stop] = np.maximum(
                max_statistics[start:stop], statistics.max(axis=1)
            )

    if not np.isfinite(max_statistics).all():
        raise ValueError("max-T bootstrap produced nonfinite statistics")
    critical = _quantile_higher(max_statistics, float(confidence))
    ordered_members: list[dict[str, Any]] = []
    for name in sorted(observed_by_name):
        row = dict(observed_by_name[name])
        mean = float(row["mean"])
        se = float(row["standard_error"])
        statistic = float(row["t_statistic"])
        lower = mean - critical * se
        exceedances = int(np.count_nonzero(max_statistics >= statistic))
        adjusted_p = float((1 + exceedances) / (int(reps) + 1))
        row.update(
            {
                "simultaneous_lower_bound": float(lower),
                "adjusted_p_value": adjusted_p,
                "positive_simultaneous_lower_bound": int(lower > 0.0),
            }
        )
        ordered_members.append(row)

    positive = [
        value["name"]
        for value in ordered_members
        if _one(value["positive_simultaneous_lower_bound"])
    ]
    return {
        "schema": SCHEMA + "-studentized-max-t",
        "schema_version": SCHEMA_VERSION,
        "version": MAX_T_VERSION,
        "evaluation_status": "evaluated",
        "valid": 1,
        "finite": 1,
        "seed": int(seed),
        "confidence": float(confidence),
        "bootstrap_replicates": int(reps),
        "bootstrap_chunk_size": int(bootstrap_chunk_size),
        "minimum_paths": int(minimum_paths),
        "standard_error_floor": float(standard_error_floor),
        "family_fingerprint": fingerprint,
        "family_size": len(ordered_members),
        "resampling_block_count": len(blocks),
        "critical_value": float(critical),
        "members": ordered_members,
        "positive_member_names": positive,
        "positive_member_count": len(positive),
        "familywise_false_discovery": int(bool(positive)),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_max_t_null_family(
    value: Mapping[str, Any],
    *,
    expected_members: Sequence[str] | None = None,
    expected_member_count: int | None = None,
    required_confidence: float = 0.95,
    required_replicates: int = 50_000,
    gate_name: str = "sealed_null_max_t_family",
) -> dict[str, Any]:
    """Gate a precomputed max-T family without changing its member set."""

    members = [
        dict(item)
        for item in value.get("members", [])
        if isinstance(item, Mapping)
    ]
    names = [str(item.get("name")) for item in members]
    expected_names = None if expected_members is None else sorted(map(str, expected_members))
    count = int(expected_member_count) if expected_member_count is not None else (
        len(expected_names) if expected_names is not None else len(members)
    )
    role_counts: dict[str, int] = {}
    positive_names_by_role: dict[str, list[str]] = {}
    for item in members:
        role = str(item.get("panel_role", "")).strip().lower()
        role_counts[role] = role_counts.get(role, 0) + 1
        if _one(item.get("positive_simultaneous_lower_bound", 0)):
            positive_names_by_role.setdefault(role, []).append(str(item.get("name")))
    checks = [
        _check("valid", value.get("valid"), "==", 1, _one(value.get("valid"))),
        _check("finite", value.get("finite"), "==", 1, _one(value.get("finite"))),
        _check("version", value.get("version"), "==", MAX_T_VERSION, value.get("version") == MAX_T_VERSION),
        _check("family_size", len(members), "==", count, len(members) == count),
        _check(
            "exact_member_set",
            sorted(names),
            "==",
            expected_names if expected_names is not None else "predeclared family",
            expected_names is None or sorted(names) == expected_names,
        ),
        _check(
            "confidence",
            value.get("confidence"),
            "==",
            required_confidence,
            _finite(value.get("confidence"))
            and float(value["confidence"]) == float(required_confidence),
        ),
        _check(
            "bootstrap_replicates",
            value.get("bootstrap_replicates"),
            "==",
            int(required_replicates),
            value.get("bootstrap_replicates") is not None
            and int(value["bootstrap_replicates"]) == int(required_replicates),
        ),
        _check(
            "no_familywise_false_discovery",
            value.get("familywise_false_discovery"),
            "==",
            0,
            value.get("familywise_false_discovery") is not None
            and int(value["familywise_false_discovery"]) == 0,
        ),
    ]
    result = _finish(
        gate_name,
        checks,
        "one-sided studentized simultaneous whole-path null family",
        evaluation_status=_status(value),
    )
    result.update(
        {
            "max_t_record": dict(value),
            "member_names": names,
            "familywise_false_discovery": int(
                bool(int(value.get("familywise_false_discovery", 0)))
            ),
            "role_counts": role_counts,
            "positive_member_names_by_role": positive_names_by_role,
        }
    )
    return result


def evaluate_simultaneous_bootstrap_preflight(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    checks = [
        _check("complete", metrics.get("complete"), "==", 1, _one(metrics.get("complete"))),
        _check("finite", metrics.get("finite"), "==", 1, _one(metrics.get("finite"))),
        _check("version", metrics.get("version"), "==", MAX_T_VERSION, metrics.get("version") == MAX_T_VERSION),
    ]
    for name in (
        "deterministic_replay_pass",
        "member_order_invariance_pass",
        "path_order_invariance_pass",
        "shared_block_coupling_pass",
        "disjoint_block_stream_pass",
        "studentization_reference_pass",
        "simultaneous_coverage_fixture_pass",
        "whole_path_only_pass",
        "parent_family_coverage_pass",
    ):
        checks.append(_check(name, metrics.get(name), "==", 1, _one(metrics.get(name))))
    for name in ("physical_training_performed", "sampling_performed"):
        value = metrics.get(name, 0)
        checks.append(_check(name, value, "==", 0, int(value) == 0))
    return _finish(
        "simultaneous_bootstrap_preflight",
        checks,
        "implementation validity of deterministic studentized whole-path max-T",
        evaluation_status=_status(metrics),
    )


def _find_gate(value: Any, gate_name: str) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        if value.get("gate") == gate_name:
            return dict(value)
        for child in value.values():
            result = _find_gate(child, gate_name)
            if result is not None:
                return result
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            result = _find_gate(child, gate_name)
            if result is not None:
                return result
    return None


def _failed_subchecks(gate: Mapping[str, Any]) -> list[str]:
    return sorted(
        name
        for name, check in _mapping(gate.get("subchecks", {})).items()
        if not _passed(_mapping(check))
    )


def _bounds(selection: Mapping[str, Any], panel: str) -> list[float | None]:
    if panel == "a":
        raw = _mapping(selection.get("nomination", {})).get(
            "nominee_panel_a_lower_bounds", []
        )
    else:
        raw = _mapping(selection.get("confirmation", {})).get(
            "panel_b_lower_bounds", []
        )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [float(value) if _finite(value) else None for value in raw]


def evaluate_parent_replay_candidate(
    candidate: Mapping[str, Any],
    *,
    sealed_b_binding: bool | int | Mapping[str, Any],
    thresholds: SealedNullThresholds | None = None,
) -> dict[str, Any]:
    """Reconstruct one parent candidate, allowing only its invalid A test."""

    thresholds = thresholds or SealedNullThresholds()
    base = _find_gate(candidate, "density_ratio_pilot_candidate")
    base = {} if base is None else base
    failures = _failed_subchecks(base)
    selection = _mapping(base.get("null_selection", {}))
    confirmation = _mapping(selection.get("confirmation", {}))
    b_bounds = _bounds(selection, "b")
    learning_rate = candidate.get("learning_rate", base.get("learning_rate"))
    accumulation = candidate.get("accumulation_steps")
    risk = candidate.get("teacher_mean_ab_bce", base.get("teacher_mean_ab_bce"))
    clip = candidate.get(
        "maximum_clip_fraction_observed", base.get("maximum_clip_fraction_observed")
    )
    checks = [
        _check("legacy_gate_found", int(bool(base)), "==", 1, bool(base)),
        _check(
            "legacy_failure_is_a_only",
            failures,
            "==",
            ["null_panel_a_lower_bounds"],
            failures == ["null_panel_a_lower_bounds"],
        ),
        _check("optimizer_health", candidate.get("optimizer_health_pass"), "==", 1, _one(candidate.get("optimizer_health_pass"))),
        _check(
            "learning_rate",
            learning_rate,
            "in",
            list(thresholds.expected_pilot_learning_rates),
            _finite(learning_rate)
            and float(learning_rate) in thresholds.expected_pilot_learning_rates,
        ),
        _check(
            "accumulation_steps",
            accumulation,
            "==",
            thresholds.expected_accumulation_steps,
            accumulation is not None
            and int(accumulation) == thresholds.expected_accumulation_steps,
        ),
        _check("teacher_mean_ab_bce", risk, "finite", True, _finite(risk)),
        _check(
            "maximum_clip_fraction_observed",
            clip,
            "<=",
            thresholds.selection_power.head.stability.maximum_clip_fraction,
            _finite(clip)
            and 0.0
            <= float(clip)
            <= thresholds.selection_power.head.stability.maximum_clip_fraction,
        ),
        _check("sealed_b_binding", int(_passed(sealed_b_binding)), "==", 1, _passed(sealed_b_binding)),
        _check("sealed_b_rejected", confirmation.get("accepted"), "==", 0, not _one(confirmation.get("accepted", 0))),
        _check("null_selected_step", selection.get("selected_step"), "==", 0, selection.get("selected_step") is not None and int(selection["selected_step"]) == 0),
        _check(
            "sealed_b_lower_bounds",
            b_bounds,
            "<= 0 each",
            0.0,
            len(b_bounds) == 2
            and all(value is not None and value <= 0.0 for value in b_bounds),
        ),
    ]
    for name in ("physical_training_performed", "sampling_performed"):
        value = candidate.get(name, 0)
        checks.append(_check(name, value, "==", 0, int(value) == 0))
    result = _finish(
        "sealed_null_parent_replay_candidate",
        checks,
        "immutable normalized-head candidate with A advisory and sealed B authoritative",
        evaluation_status=_status(candidate),
    )
    result.update(
        {
            "learning_rate": float(learning_rate) if _finite(learning_rate) else None,
            "body_learning_rate": float(learning_rate) if _finite(learning_rate) else None,
            "accumulation_steps": int(accumulation) if accumulation is not None else None,
            "teacher_mean_ab_bce": float(risk) if _finite(risk) else None,
            "maximum_clip_fraction_observed": float(clip) if _finite(clip) else None,
            "legacy_failed_subchecks": failures,
            "discovery_a_lower_bounds_advisory": _bounds(selection, "a"),
            "sealed_b_lower_bounds": b_bounds,
            "legacy_gate": base,
            "sealed_b_binding": dict(sealed_b_binding)
            if isinstance(sealed_b_binding, Mapping)
            else int(_passed(sealed_b_binding)),
            "optimizer_health_pass": int(_one(candidate.get("optimizer_health_pass"))),
        }
    )
    return result


def select_recovered_profile(
    candidate_gates: Sequence[Mapping[str, Any]],
    thresholds: SealedNullThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SealedNullThresholds()
    eligible = [
        (index, dict(value))
        for index, value in enumerate(candidate_gates)
        if _passed(value)
        and _finite(value.get("teacher_mean_ab_bce"))
        and _finite(value.get("maximum_clip_fraction_observed"))
        and _finite(value.get("learning_rate"))
    ]
    eligible.sort(
        key=lambda item: (
            float(item[1]["teacher_mean_ab_bce"]),
            float(item[1]["maximum_clip_fraction_observed"]),
            float(item[1]["learning_rate"]),
        )
    )
    selected = eligible[0] if eligible else None
    profile = None
    if selected is not None:
        index, value = selected
        body_lr = float(value["learning_rate"])
        profile = {
            "candidate_index": int(index),
            "learning_rate": body_lr,
            "body_learning_rate": body_lr,
            "head_learning_rate": thresholds.selection_power.head.grid_cells * body_lr,
            "accumulation_steps": thresholds.expected_accumulation_steps,
            "teacher_mean_ab_bce": float(value["teacher_mean_ab_bce"]),
            "maximum_clip_fraction_observed": float(
                value["maximum_clip_fraction_observed"]
            ),
        }
    return {
        "schema": SCHEMA + "-recovered-profile",
        "schema_version": SCHEMA_VERSION,
        "selected": int(selected is not None),
        "passed": int(selected is not None),
        "selected_candidate_index": None if selected is None else int(selected[0]),
        "profile": profile,
        "ranking": [
            {
                "candidate_index": int(index),
                "learning_rate": float(value["learning_rate"]),
                "teacher_mean_ab_bce": float(value["teacher_mean_ab_bce"]),
                "maximum_clip_fraction_observed": float(
                    value["maximum_clip_fraction_observed"]
                ),
            }
            for index, value in eligible
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_parent_pilot_replay(
    candidates: Sequence[Mapping[str, Any]],
    *,
    sealed_b_bindings: Sequence[bool | int | Mapping[str, Any]],
    discovery_family: bool | int | Mapping[str, Any] | None = None,
    sealed_b_family: bool | int | Mapping[str, Any],
    thresholds: SealedNullThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SealedNullThresholds()
    bindings = list(sealed_b_bindings)
    gates = [
        evaluate_parent_replay_candidate(
            candidate,
            sealed_b_binding=bindings[index] if index < len(bindings) else False,
            thresholds=thresholds,
        )
        for index, candidate in enumerate(candidates)
    ]
    profile = select_recovered_profile(gates, thresholds)
    rates = [value.get("learning_rate") for value in gates]
    all_nulls = len(gates) == thresholds.expected_pilot_candidates and all(
        _passed(value) for value in gates
    )
    sealed_roles = _mapping(sealed_b_family).get("role_counts", {})
    sealed_roles = _mapping(sealed_roles)
    sealed_family_shape = (
        int(sealed_roles.get("b", -1)) == 4
        and sum(int(value) for value in sealed_roles.values()) == 4
    )
    checks = [
        _check(
            "candidate_count",
            len(gates),
            "==",
            thresholds.expected_pilot_candidates,
            len(gates) == thresholds.expected_pilot_candidates,
        ),
        _check(
            "sealed_binding_count",
            len(bindings),
            "==",
            thresholds.expected_pilot_candidates,
            len(bindings) == thresholds.expected_pilot_candidates,
        ),
        _check(
            "learning_rate_set",
            sorted(float(value) for value in rates if _finite(value)),
            "==",
            sorted(thresholds.expected_pilot_learning_rates),
            len(rates) == thresholds.expected_pilot_candidates
            and sorted(float(value) for value in rates if _finite(value))
            == sorted(thresholds.expected_pilot_learning_rates),
        ),
        _check("sealed_b_max_t_family", int(_passed(sealed_b_family)), "==", 1, _passed(sealed_b_family)),
        _check(
            "sealed_b_family_shape",
            sealed_roles,
            "==",
            {"b": 4},
            sealed_family_shape,
        ),
        _check("every_candidate_sealed_null_pass", int(all_nulls), "==", 1, all_nulls),
        _check("recovered_profile", profile["selected"], "==", 1, _one(profile["selected"])),
    ]
    result = _finish(
        "sealed_null_parent_pilot_replay",
        checks,
        "read-only replay with discovery A advisory and simultaneous sealed-B evidence",
        evaluation_status="evaluated" if gates else "not_evaluated",
    )
    result.update(
        {
            "candidate_gates": gates,
            "selected_profile": profile,
            "discovery_family_advisory": dict(discovery_family)
            if isinstance(discovery_family, Mapping)
            else None,
            "discovery_family_authorizing": 0,
            "sealed_b_family_gate": dict(sealed_b_family)
            if isinstance(sealed_b_family, Mapping)
            else int(_passed(sealed_b_family)),
            "familywise_false_discovery": int(
                bool(
                    int(
                        _mapping(sealed_b_family).get(
                            "familywise_false_discovery", 0
                        )
                    )
                )
            ),
            "optimizer_health_pass": int(
                bool(gates)
                and all(_one(value.get("optimizer_health_pass")) for value in gates)
            ),
            "replay_only": 1,
            "optimizer_steps_performed": 0,
        }
    )
    return result


def _selection_from_null(value: Mapping[str, Any]) -> dict[str, Any]:
    direct = value.get("selection")
    if isinstance(direct, Mapping):
        return dict(direct)
    found = _find_gate(value, "density_ratio_checkpoint_selection")
    return {} if found is None else found


def evaluate_confirmation_null_family(
    null_results: Sequence[Mapping[str, Any]],
    *,
    max_t_family: bool | int | Mapping[str, Any],
    sealed_b_bindings: Sequence[bool | int | Mapping[str, Any]],
    thresholds: SealedNullThresholds | None = None,
) -> dict[str, Any]:
    """Gate three null tasks structurally plus their simultaneous B/C/D family."""

    thresholds = thresholds or SealedNullThresholds()
    bindings = list(sealed_b_bindings)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(null_results):
        value = _mapping(raw.get("metrics", raw))
        selection = _selection_from_null(value)
        confirmation = _mapping(selection.get("confirmation", {}))
        seed = value.get("model_seed", value.get("seed"))
        optimizer = _one(value.get("optimizer_health_pass"))
        if not optimizer:
            clips = [
                value.get("post_warmup_clip_fraction"),
                value.get("final_500_clip_fraction"),
                value.get("final_200_clip_fraction"),
            ]
            optimizer = all(
                _finite(item)
                and float(item)
                <= thresholds.selection_power.head.stability.maximum_clip_fraction
                for item in clips
            )
        structural = (
            _one(value.get("complete"))
            and _one(value.get("finite"))
            and _one(value.get("boundary_admissible"))
            and selection.get("nominee_step") is not None
            and int(selection["nominee_step"]) > 0
            and selection.get("selected_step") is not None
            and int(selection["selected_step"]) == 0
            and not _one(confirmation.get("accepted", 0))
            and index < len(bindings)
            and _passed(bindings[index])
            and int(value.get("physical_training_performed", 0)) == 0
            and int(value.get("sampling_performed", 0)) == 0
        )
        rows.append(
            {
                "model_seed": seed,
                "complete_finite_boundary_and_sealed": int(structural),
                "optimizer_health_pass": int(optimizer),
                "legacy_false_discovery_advisory": int(
                    bool(int(value.get("false_discovery", 0)))
                ),
                "selected_step": selection.get("selected_step"),
                "nominee_step": selection.get("nominee_step"),
            }
        )
    expected = thresholds.expected_confirmation_null_seeds
    seeds = [value.get("model_seed") for value in rows]
    structural_pass = len(rows) == expected and all(
        _one(value.get("complete_finite_boundary_and_sealed")) for value in rows
    )
    optimizer_pass = len(rows) == expected and all(
        _one(value.get("optimizer_health_pass")) for value in rows
    )
    role_counts = _mapping(max_t_family).get("role_counts", {})
    role_counts = _mapping(role_counts)
    expected_role_counts = {"b": 6, "c": 6, "d": 6}
    exact_role_family = role_counts == expected_role_counts
    positive_by_role = _mapping(max_t_family).get(
        "positive_member_names_by_role", {}
    )
    positive_by_role = _mapping(positive_by_role)
    positive_b = len(positive_by_role.get("b", []))
    positive_audit = len(positive_by_role.get("c", [])) + len(
        positive_by_role.get("d", [])
    )
    positive_unknown = sum(
        len(values)
        for role, values in positive_by_role.items()
        if role not in {"b", "c", "d"} and isinstance(values, Sequence)
    )
    recorded_familywise_positive = bool(
        int(_mapping(max_t_family).get("familywise_false_discovery", 0))
    )
    unclassified_positive = (
        recorded_familywise_positive
        and positive_b + positive_audit + positive_unknown == 0
    )
    checks = [
        _check("task_count", len(rows), "==", expected, len(rows) == expected),
        _check("sealed_binding_count", len(bindings), "==", expected, len(bindings) == expected),
        _check("distinct_seed_count", len(set(seeds)), "==", expected, None not in seeds and len(set(seeds)) == expected),
        _check("all_tasks_structurally_valid", int(structural_pass), "==", 1, structural_pass),
        _check("all_optimizers_valid", int(optimizer_pass), "==", 1, optimizer_pass),
        _check(
            "exact_b_c_d_family",
            role_counts,
            "==",
            expected_role_counts,
            exact_role_family,
        ),
        _check("simultaneous_null_family", int(_passed(max_t_family)), "==", 1, _passed(max_t_family)),
    ]
    result = _finish(
        "sealed_null_confirmation_family",
        checks,
        "three-seed analytic-zero null with familywise B/C/D evidence",
        evaluation_status="evaluated" if rows else "not_evaluated",
    )
    result.update(
        {
            "task_rows": rows,
            "max_t_family_gate": dict(max_t_family)
            if isinstance(max_t_family, Mapping)
            else int(_passed(max_t_family)),
            "optimizer_health_pass": int(optimizer_pass),
            "legacy_false_discovery_count_advisory": sum(
                _one(value.get("legacy_false_discovery_advisory")) for value in rows
            ),
            "familywise_false_discovery": int(
                bool(int(_mapping(max_t_family).get("familywise_false_discovery", 0)))
            ),
            "selection_false_discovery": int(positive_b > 0),
            "audit_false_discovery": int(positive_audit > 0),
            "role_categorization_invalid": int(
                not exact_role_family or positive_unknown > 0 or unclassified_positive
            ),
            "positive_b_member_count": int(positive_b),
            "positive_audit_member_count": int(positive_audit),
            "positive_unknown_member_count": int(positive_unknown),
            "role_counts": role_counts,
        }
    )
    return result


def evaluate_sealed_null_controls(
    *,
    provenance: bool | int | Mapping[str, Any],
    simultaneous_bootstrap: bool | int | Mapping[str, Any],
    replay: bool | int | Mapping[str, Any],
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
        _check("simultaneous_bootstrap", int(_passed(simultaneous_bootstrap)), "==", 1, _passed(simultaneous_bootstrap)),
        _check("profile_replay", int(_passed(replay)), "==", 1, _passed(replay)),
        _check("confirmation_panel_power", int(_passed(confirmation_panel_power)), "==", 1, _passed(confirmation_panel_power)),
        _check("teacher_study", int(_passed(teacher)), "==", 1, _passed(teacher)),
        _check("null_family", int(_passed(null)), "==", 1, _passed(null)),
        _check("all_optimizers_valid", int(optimizer), "==", 1, optimizer),
    ]
    result = _finish(
        "sealed_null_density_ratio_controls",
        checks,
        "oracle-powered derivative-accurate teacher and simultaneous null confirmation",
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
        }
    )
    return result


def decide_sealed_null_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    simultaneous_bootstrap: bool | int | Mapping[str, Any],
    replay: Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    teacher = _mapping(controls.get("teacher_study", {}))
    null = _mapping(controls.get("null_family", {}))
    interim = False
    if not _passed(provenance):
        decision: SealedNullDecision | str = SealedNullDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable selection-power parent and transitive registry binding"
    elif _status(simultaneous_bootstrap) != "evaluated":
        decision = "sealed_null_preflight_not_evaluated"
        action = "run the simultaneous whole-path bootstrap preflight"
        interim = True
    elif not _passed(simultaneous_bootstrap):
        decision = SealedNullDecision.SIMULTANEOUS_BOOTSTRAP_INVALID
        action = "repair max-T studentization, clustering, determinism, or family coverage"
    elif _status(replay) != "evaluated":
        decision = "sealed_null_preflight_passed"
        action = "replay the immutable pilot under the sealed-null family gate"
        interim = True
    elif bool(int(replay.get("familywise_false_discovery", 0))):
        decision = SealedNullDecision.SELECTION_FALSE_DISCOVERY
        action = "stop: the simultaneous parent null family contains positive evidence"
    elif not _passed(replay):
        decision = SealedNullDecision.PROFILE_RECOVERY_INVALID
        action = "repair replay coverage or retain the original unresolved profile outcome"
    elif _status(confirmation_panel_power) == "evaluated" and not _passed(
        confirmation_panel_power
    ):
        decision = SealedNullDecision.EVIDENCE_PANEL_UNDERPOWERED
        action = "stop before training and preserve the fixed underpowered panels"
    elif _status(controls) != "evaluated":
        decision = "sealed_null_profile_recovered"
        action = "run the fresh oracle-qualified three-seed confirmation"
        interim = True
    elif not _one(controls.get("optimizer_health_pass")):
        decision = SealedNullDecision.CLASSIFICATION_OPTIMIZER_INVALID
        action = "repair incomplete, nonfinite, clipped, or incompatible confirmation tasks"
    elif bool(int(null.get("role_categorization_invalid", 0))):
        decision = SealedNullDecision.SIMULTANEOUS_BOOTSTRAP_INVALID
        action = "repair the complete predeclared B/C/D simultaneous family"
    elif bool(int(null.get("selection_false_discovery", 0))):
        decision = SealedNullDecision.SELECTION_FALSE_DISCOVERY
        action = "stop: a sealed-B simultaneous null member is positive"
    elif bool(int(null.get("audit_false_discovery", 0))) or bool(
        int(teacher.get("panel_disagreement", 0))
    ):
        decision = SealedNullDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE
        action = "repeat the frozen confirmation after a C/D family or teacher-panel disagreement"
    elif _passed(controls):
        decision = SealedNullDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        action = "plan fresh physical one-image density-ratio score training"
    elif int(teacher.get("classification_passing_seed_count", 0)) >= 2:
        decision = SealedNullDecision.DENSITY_RATIO_VALUE_ONLY
        action = "plan the predeclared H1-like function-step trust patch; do not shrink the model"
    elif int(
        teacher.get(
            "positive_point_estimate_seed_count",
            teacher.get("classification_passing_seed_count", 0),
        )
    ) > 0:
        decision = SealedNullDecision.CLASSIFICATION_POWER_CONFIRMATION_UNRESOLVED
        action = "plan the predeclared H1-like function-step trust patch"
    else:
        decision = SealedNullDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL
        action = "revisit learning on the exact bounded synthetic density-ratio law"
    value = decision.value if isinstance(decision, SealedNullDecision) else decision
    return {
        "decision": value,
        "recommended_next_action": action,
        "interim_stage_success": int(interim and value.endswith("_passed") or value == "sealed_null_profile_recovered"),
        "closed_terminal_scientific_outcome": int(not interim),
        "h1_function_step_patch_authorized": int(
            decision
            in {
                SealedNullDecision.CLASSIFICATION_POWER_CONFIRMATION_UNRESOLVED,
                SealedNullDecision.DENSITY_RATIO_VALUE_ONLY,
            }
        ),
        "physical_training_authorized": int(
            decision is SealedNullDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED
        ),
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }


def evaluate_sealed_null_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    simultaneous_bootstrap: bool | int | Mapping[str, Any],
    replay: Mapping[str, Any],
    confirmation_panel_power: bool | int | Mapping[str, Any],
    teacher_study: Mapping[str, Any],
    null_family: Mapping[str, Any],
    require_gate: str = "none",
    thresholds: SealedNullThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or SealedNullThresholds()
    if require_gate not in {"none", "preflight", "replay", "controls"}:
        raise ValueError("require_gate must be none, preflight, replay, or controls")
    controls = evaluate_sealed_null_controls(
        provenance=provenance,
        simultaneous_bootstrap=simultaneous_bootstrap,
        replay=replay,
        confirmation_panel_power=confirmation_panel_power,
        teacher_study=teacher_study,
        null_family=null_family,
    )
    required_pass = {
        "none": True,
        "preflight": _passed(provenance) and _passed(simultaneous_bootstrap),
        "replay": _passed(provenance)
        and _passed(simultaneous_bootstrap)
        and _passed(replay),
        "controls": _passed(controls),
    }[require_gate]
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "components": {
            "provenance": dict(provenance)
            if isinstance(provenance, Mapping)
            else int(_passed(provenance)),
            "simultaneous_bootstrap": dict(simultaneous_bootstrap)
            if isinstance(simultaneous_bootstrap, Mapping)
            else int(_passed(simultaneous_bootstrap)),
            "replay": dict(replay),
            "confirmation_panel_power": dict(confirmation_panel_power)
            if isinstance(confirmation_panel_power, Mapping)
            else int(_passed(confirmation_panel_power)),
            "controls": controls,
        },
        "decision": decide_sealed_null_workflow(
            provenance=provenance,
            simultaneous_bootstrap=simultaneous_bootstrap,
            replay=replay,
            confirmation_panel_power=confirmation_panel_power,
            controls=controls,
        ),
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "thresholds": thresholds.to_dict(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
