from __future__ import annotations

"""Fail-closed statistics for the D0 implicit Dirichlet-score experiment.

The helpers in this module intentionally know nothing about Torch, caches, or
reverse samplers.  They consume paired, already-aggregated path metrics and
enforce the statistical contract of the implicit-score learnability gate:

* model seeds and Hutchinson/witness banks are fixed experimental factors;
* uncertainty resamples whole forward paths, never states or model seeds;
* both independent trace banks and both reporting scopes must agree; and
* a score-risk improvement is not promoted unless probe-free Stein witnesses
  and cross-seed nonlinear-flux agreement support it.

Canonical task-result and row schemas are documented on
``evaluate_score_seed``.  A few conservative aliases are accepted so report
writers can use either sums or already-normalized paired deltas.
"""

import hashlib
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


AUDIT_BANKS: tuple[str, str] = ("audit_a", "audit_b")
AUDIT_SCOPES: tuple[str, str] = ("overall", "data_end")
STEIN_BANKS: tuple[str, str] = ("stein_a", "stein_b")
CLAIM_SCOPE = (
    "positive-time relative-score learnability beyond a frozen state-linear "
    "baseline for one lambda-mixed image"
)


@dataclass(frozen=True)
class DirichletScoreGateThresholds:
    """Frozen production thresholds from the implicit-score patch plan."""

    expected_model_seeds: int = 3
    min_passing_model_seeds: int = 2
    expected_audit_paths: int = 24
    expected_data_end_states_per_path: int = 16
    expected_data_end_states: int = 384
    min_score_risk_delta: float = 0.0
    min_stein_improvement: float = 0.0
    bootstrap_confidence: float = 0.90
    bootstrap_reps: int = 10_000
    bootstrap_seed: int = 260760
    median_of_means_groups: int = 4
    nonlinear_flux_cosine: float = 0.50
    nonlinear_flux_cosine_lcb: float = 0.25

    # Positive and null synthetic controls.
    teacher_min_score_gain: float = 0.90
    teacher_min_overall_flux_cosine: float = 0.98
    teacher_min_bin_flux_cosine: float = 0.95
    teacher_max_overall_relative_flux_l2: float = 0.15
    teacher_max_bin_relative_flux_l2: float = 0.20
    teacher_expected_time_bins: int = 5

    def __post_init__(self) -> None:
        positive_ints = {
            "expected_model_seeds": self.expected_model_seeds,
            "min_passing_model_seeds": self.min_passing_model_seeds,
            "expected_audit_paths": self.expected_audit_paths,
            "expected_data_end_states_per_path": self.expected_data_end_states_per_path,
            "expected_data_end_states": self.expected_data_end_states,
            "bootstrap_reps": self.bootstrap_reps,
            "median_of_means_groups": self.median_of_means_groups,
            "teacher_expected_time_bins": self.teacher_expected_time_bins,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.min_passing_model_seeds) > int(self.expected_model_seeds):
            raise ValueError(
                "min_passing_model_seeds cannot exceed expected_model_seeds"
            )
        if int(self.median_of_means_groups) > int(self.expected_audit_paths):
            raise ValueError(
                "median_of_means_groups cannot exceed expected_audit_paths"
            )
        if (
            int(self.expected_audit_paths)
            * int(self.expected_data_end_states_per_path)
            != int(self.expected_data_end_states)
        ):
            raise ValueError(
                "expected_data_end_states must equal paths times states per path"
            )
        if int(self.bootstrap_seed) < 0:
            raise ValueError("bootstrap_seed must be non-negative")
        if not 0.5 < float(self.bootstrap_confidence) < 1.0:
            raise ValueError("bootstrap_confidence must be in (0.5, 1)")
        finite_values = {
            "min_score_risk_delta": self.min_score_risk_delta,
            "min_stein_improvement": self.min_stein_improvement,
            "nonlinear_flux_cosine": self.nonlinear_flux_cosine,
            "nonlinear_flux_cosine_lcb": self.nonlinear_flux_cosine_lcb,
            "teacher_min_score_gain": self.teacher_min_score_gain,
            "teacher_min_overall_flux_cosine": self.teacher_min_overall_flux_cosine,
            "teacher_min_bin_flux_cosine": self.teacher_min_bin_flux_cosine,
            "teacher_max_overall_relative_flux_l2": self.teacher_max_overall_relative_flux_l2,
            "teacher_max_bin_relative_flux_l2": self.teacher_max_bin_relative_flux_l2,
        }
        if not all(math.isfinite(float(value)) for value in finite_values.values()):
            raise ValueError("all floating gate thresholds must be finite")
        for name in (
            "nonlinear_flux_cosine",
            "nonlinear_flux_cosine_lcb",
            "teacher_min_overall_flux_cosine",
            "teacher_min_bin_flux_cosine",
        ):
            value = float(getattr(self, name))
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        if not 0.0 <= float(self.teacher_min_score_gain) <= 1.0:
            raise ValueError("teacher_min_score_gain must be in [0, 1]")
        for name in (
            "teacher_max_overall_relative_flux_l2",
            "teacher_max_bin_relative_flux_l2",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


class ImplicitScoreDecision(str, Enum):
    IMPLICIT_SCORE_SIGNAL = "implicit_score_signal"
    LINEAR_SPATIOTEMPORAL_ONLY = "linear_spatiotemporal_only"
    OBJECTIVE_ONLY_SIGNAL = "objective_only_signal"
    TRACE_ESTIMATOR_INCONCLUSIVE = "trace_estimator_inconclusive"
    BOUNDARY_OR_OUTLIER_ARTIFACT = "boundary_or_outlier_artifact"
    PATH_MEMORIZATION_ONLY = "path_memorization_only"
    NO_DETECTABLE_IMPLICIT_SCORE = "no_detectable_implicit_score"
    OPERATOR_INVALID = "operator_invalid"
    CACHE_INVALID = "cache_invalid"
    OPTIMIZATION_PIPELINE_INVALID = "optimization_pipeline_invalid"


def _jsonable(value: Any) -> Any:
    """Return strict-JSON-compatible data (no NumPy scalars or NaN)."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _check(
    name: str,
    value: Any,
    operator: str,
    threshold: Any,
    passed: bool,
    *,
    reason: str | None = None,
) -> tuple[str, dict[str, Any]]:
    record: dict[str, Any] = {
        "passed": int(bool(passed)),
        "value": _jsonable(value),
        "operator": str(operator),
        "threshold": _jsonable(threshold),
    }
    if reason:
        record["reason"] = str(reason)
    return str(name), record


def _finish_gate(
    name: str,
    checks: Sequence[tuple[str, Mapping[str, Any]]],
    claim_scope: str,
) -> dict[str, Any]:
    subchecks = {key: _jsonable(dict(value)) for key, value in checks}
    passed = bool(subchecks) and all(
        bool(int(record.get("passed", 0))) for record in subchecks.values()
    )
    return {
        "gate": str(name),
        "passed": int(passed),
        f"{name}_pass": int(passed),
        "subchecks": subchecks,
        "claim_scope": str(claim_scope),
    }


def _passed(value: bool | int | Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        return bool(int(value.get("passed", value.get("gate_pass", 0))))
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    raise ValueError("gate values must be a mapping, boolean, or 0/1")


def _finite_float(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _normalize_bank(value: Any, *, stein: bool = False) -> str:
    banks = STEIN_BANKS if stein else AUDIT_BANKS
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "0": banks[0],
        "1": banks[1],
        "a": banks[0],
        "b": banks[1],
        "bank_a": banks[0],
        "bank_b": banks[1],
        "audit_0": AUDIT_BANKS[0],
        "audit_1": AUDIT_BANKS[1],
        "stein_0": STEIN_BANKS[0],
        "stein_1": STEIN_BANKS[1],
    }
    text = aliases.get(text, text)
    if text not in banks:
        kind = "Stein" if stein else "audit"
        raise ValueError(f"unknown {kind} bank {value!r}")
    return text


def _normalize_scope(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {"all": "overall", "dataend": "data_end", "endpoint": "data_end"}
    text = aliases.get(text, text)
    if text not in AUDIT_SCOPES:
        raise ValueError(f"unknown audit scope {value!r}")
    return text


def _model_seed(result: Mapping[str, Any]) -> int:
    value = result.get("model_seed", result.get("training_seed", result.get("seed")))
    seed = int(value)
    if seed < 0:
        raise ValueError("model_seed must be non-negative")
    return seed


def _row_seed(row: Mapping[str, Any], default: int | None = None) -> int:
    value = row.get("model_seed", row.get("training_seed", row.get("seed", default)))
    if value is None:
        raise ValueError("path metric is missing model_seed")
    seed = int(value)
    if seed < 0:
        raise ValueError("model_seed must be non-negative")
    return seed


def _path_id(row: Mapping[str, Any]) -> int:
    path = int(row["path_id"])
    if path < 0:
        raise ValueError("path_id must be non-negative")
    return path


def _state_count(row: Mapping[str, Any]) -> int:
    count = int(row.get("state_count", row.get("slice_count", row.get("count", -1))))
    if count <= 0:
        raise ValueError("state_count must be positive")
    return count


def _finite_row(row: Mapping[str, Any]) -> bool:
    value = row.get("finite_fraction", row.get("metric_finite_fraction", 1.0))
    try:
        return float(value) == 1.0 and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _paired_delta(row: Mapping[str, Any], baseline: str) -> float:
    aliases = (
        f"score_risk_delta_vs_{baseline}",
        f"risk_delta_vs_{baseline}",
        f"delta_vs_{baseline}",
    )
    for key in aliases:
        if key in row:
            return _finite_float(row[key], key)
    full_keys = ("full_risk", "model_risk", "score_risk")
    base_keys = (f"{baseline}_risk", f"{baseline}_score_risk")
    full_key = next((key for key in full_keys if key in row), None)
    base_key = next((key for key in base_keys if key in row), None)
    if full_key is not None and base_key is not None:
        return _finite_float(row[base_key], base_key) - _finite_float(
            row[full_key], full_key
        )
    full_sum_key = next(
        (key for key in ("full_risk_sum", "model_risk_sum") if key in row), None
    )
    base_sum_key = next(
        (
            key
            for key in (f"{baseline}_risk_sum", f"{baseline}_score_risk_sum")
            if key in row
        ),
        None,
    )
    if full_sum_key is not None and base_sum_key is not None:
        return (
            _finite_float(row[base_sum_key], base_sum_key)
            - _finite_float(row[full_sum_key], full_sum_key)
        ) / float(_state_count(row))
    raise ValueError(f"row is missing a paired score-risk delta versus {baseline}")


def _stein_improvement(row: Mapping[str, Any]) -> float:
    for key in (
        "stein_discrepancy_improvement",
        "discrepancy_improvement",
        "stein_improvement",
    ):
        if key in row:
            return _finite_float(row[key], key)
    if "linear_discrepancy" in row and "full_discrepancy" in row:
        return _finite_float(row["linear_discrepancy"], "linear_discrepancy") - _finite_float(
            row["full_discrepancy"], "full_discrepancy"
        )
    if "linear_discrepancy_sum" in row and "full_discrepancy_sum" in row:
        return (
            _finite_float(row["linear_discrepancy_sum"], "linear_discrepancy_sum")
            - _finite_float(row["full_discrepancy_sum"], "full_discrepancy_sum")
        ) / float(_state_count(row))
    raise ValueError("row is missing Stein-discrepancy improvement")


def derive_resampling_seed(base_seed: int, *parts: Any) -> int:
    """Derive an order-independent seed for one frozen reported statistic."""

    payload = ":".join(["d0-dirichlet-score", str(int(base_seed))] + [str(p) for p in parts])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def _path_seed_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    expected_model_seeds: Sequence[int] | None = None,
    expected_path_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: dict[tuple[int, int], float] = {}
    seen_seeds: set[int] = set()
    seen_paths: set[int] = set()
    for row in rows:
        seed = _row_seed(row)
        path = _path_id(row)
        key = (path, seed)
        if key in values:
            raise ValueError("path bootstrap rows contain a duplicate path/model seed")
        values[key] = _finite_float(row[value_key], value_key)
        seen_seeds.add(seed)
        seen_paths.add(path)
    seeds = (
        tuple(sorted(int(value) for value in expected_model_seeds))
        if expected_model_seeds is not None
        else tuple(sorted(seen_seeds))
    )
    paths = (
        tuple(sorted(int(value) for value in expected_path_ids))
        if expected_path_ids is not None
        else tuple(sorted(seen_paths))
    )
    if not seeds or not paths or set(seeds) != seen_seeds or set(paths) != seen_paths:
        raise ValueError("bootstrap rows do not match the expected seeds and paths")
    expected = {(path, seed) for path in paths for seed in seeds}
    if set(values) != expected:
        raise ValueError("each whole path must contain every fixed model seed exactly once")
    matrix = np.asarray(
        [[values[(path, seed)] for seed in seeds] for path in paths], dtype=np.float64
    )
    return np.asarray(paths, dtype=np.int64), np.asarray(seeds, dtype=np.int64), matrix


def bootstrap_whole_path_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "value",
    reps: int = 10_000,
    confidence: float = 0.90,
    seed: int = 260760,
    expected_model_seeds: Sequence[int] | None = None,
    expected_path_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Bootstrap a paired mean delta over whole paths.

    Every resample selects path clusters.  All model seeds belonging to a
    selected path remain in the draw and model seeds are never resampled.
    """

    if int(reps) <= 0:
        raise ValueError("bootstrap reps must be positive")
    if not math.isfinite(float(confidence)) or not 0.5 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must be in (0.5, 1)")
    paths, seeds, matrix = _path_seed_matrix(
        rows,
        value_key=value_key,
        expected_model_seeds=expected_model_seeds,
        expected_path_ids=expected_path_ids,
    )
    # Average the fixed model-seed factor within a path, then resample paths.
    path_values = matrix.mean(axis=1, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, paths.size, size=(int(reps), paths.size))
    draws = path_values[selected].mean(axis=1, dtype=np.float64)
    alpha = 1.0 - float(confidence)
    return {
        "point_delta": float(path_values.mean(dtype=np.float64)),
        "lower_bound": float(np.quantile(draws, alpha)),
        "upper_bound": float(np.quantile(draws, float(confidence))),
        "bootstrap_mean": float(draws.mean(dtype=np.float64)),
        "confidence": float(confidence),
        "reps": int(reps),
        "seed": int(seed),
        "path_count": int(paths.size),
        "model_seed_count": int(seeds.size),
        "model_seeds": seeds.tolist(),
        "cluster_unit": "whole_path_id",
        "resampled_factors": ["whole_path_id"],
        "fixed_factors": ["model_seed"],
        "value_key": str(value_key),
    }


def median_of_means_whole_path_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "value",
    groups: int = 4,
    seed: int = 260760,
    expected_model_seeds: Sequence[int] | None = None,
    expected_path_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute a deterministic whole-path median of balanced group means."""

    paths, seeds, matrix = _path_seed_matrix(
        rows,
        value_key=value_key,
        expected_model_seeds=expected_model_seeds,
        expected_path_ids=expected_path_ids,
    )
    if int(groups) <= 0 or int(groups) > int(paths.size):
        raise ValueError("median-of-means groups must be in [1, path_count]")
    keyed = sorted(
        range(paths.size),
        key=lambda index: hashlib.sha256(
            f"d0-score-mom:{int(seed)}:{int(paths[index])}".encode("utf-8")
        ).digest(),
    )
    chunks = np.array_split(np.asarray(keyed, dtype=np.int64), int(groups))
    path_values = matrix.mean(axis=1, dtype=np.float64)
    group_means = [float(path_values[chunk].mean(dtype=np.float64)) for chunk in chunks]
    return {
        "median_of_means": float(np.median(np.asarray(group_means, dtype=np.float64))),
        "group_means": group_means,
        "group_path_ids": [[int(paths[index]) for index in chunk] for chunk in chunks],
        "groups": int(groups),
        "seed": int(seed),
        "path_count": int(paths.size),
        "model_seed_count": int(seeds.size),
        "fixed_factors": ["model_seed"],
    }


def bootstrap_cross_seed_cosine(
    rows: Sequence[Mapping[str, Any]],
    *,
    reps: int = 10_000,
    confidence: float = 0.90,
    seed: int = 260760,
    expected_model_seeds: Sequence[int] | None = None,
    expected_path_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Bootstrap the median nonlinear-flux cosine over whole paths.

    Each path must contain every unordered pair of the fixed model seeds.
    """

    if int(reps) <= 0:
        raise ValueError("bootstrap reps must be positive")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must be in (0.5, 1)")
    values: dict[tuple[int, int, int], float] = {}
    seen_seeds: set[int] = set()
    seen_paths: set[int] = set()
    for row in rows:
        path = _path_id(row)
        first = int(row.get("seed_a", row.get("model_seed_a", -1)))
        second = int(row.get("seed_b", row.get("model_seed_b", -1)))
        if first < 0 or second < 0 or first == second:
            raise ValueError("cosine rows require two distinct non-negative seeds")
        pair = tuple(sorted((first, second)))
        key = (path, pair[0], pair[1])
        if key in values:
            raise ValueError("duplicate path/model-pair cosine")
        cosine = _finite_float(row.get("cosine", row.get("nonlinear_flux_cosine")), "cosine")
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("nonlinear-flux cosine must lie in [-1, 1]")
        values[key] = cosine
        seen_paths.add(path)
        seen_seeds.update(pair)
    seeds = tuple(
        sorted(int(value) for value in (expected_model_seeds or sorted(seen_seeds)))
    )
    paths = tuple(sorted(int(value) for value in (expected_path_ids or sorted(seen_paths))))
    if set(seeds) != seen_seeds or set(paths) != seen_paths or len(seeds) < 2:
        raise ValueError("cosine rows do not match expected seeds and paths")
    pairs = tuple((left, right) for index, left in enumerate(seeds) for right in seeds[index + 1 :])
    expected = {(path, left, right) for path in paths for left, right in pairs}
    if set(values) != expected:
        raise ValueError("every path must contain every fixed cross-seed pair")
    matrix = np.asarray(
        [[values[(path, left, right)] for left, right in pairs] for path in paths],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(paths), size=(int(reps), len(paths)))
    draws = np.median(matrix[selected].reshape(int(reps), -1), axis=1)
    alpha = 1.0 - float(confidence)
    return {
        "point_median": float(np.median(matrix)),
        "lower_bound": float(np.quantile(draws, alpha)),
        "upper_bound": float(np.quantile(draws, float(confidence))),
        "bootstrap_median": float(np.median(draws)),
        "confidence": float(confidence),
        "reps": int(reps),
        "seed": int(seed),
        "path_count": len(paths),
        "model_seeds": list(seeds),
        "pair_count": len(pairs),
        "cluster_unit": "whole_path_id",
        "resampled_factors": ["whole_path_id"],
        "fixed_factors": ["model_seed_pair"],
    }


def evaluate_control_bundle(
    *,
    operator_gate: bool | int | Mapping[str, Any],
    positive_teacher_gate: bool | int | Mapping[str, Any],
    null_control_gate: bool | int | Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate the three mandatory pre-physical optimization controls."""

    operator = _passed(operator_gate)
    teacher = _passed(positive_teacher_gate)
    null = _passed(null_control_gate)
    result = _finish_gate(
        "controls",
        [
            _check("operator", int(operator), "==", 1, operator),
            _check("positive_teacher", int(teacher), "==", 1, teacher),
            _check("null_control", int(null), "==", 1, null),
        ],
        "operator correctness and synthetic implicit-score optimization controls",
    )
    result["operator_pass"] = int(operator)
    result["positive_teacher_pass"] = int(teacher)
    result["null_control_pass"] = int(null)
    return result


def evaluate_positive_teacher_control(
    metrics: Mapping[str, Any],
    thresholds: DirichletScoreGateThresholds = DirichletScoreGateThresholds(),
) -> dict[str, Any]:
    """Evaluate the exact nonuniform-Dirichlet positive teacher control."""

    overall_gain = metrics.get("audit_overall_score_gain")
    data_end_gain = metrics.get("audit_data_end_score_gain")
    overall_cosine = metrics.get("overall_flux_cosine")
    bin_cosines = list(metrics.get("time_bin_flux_cosines", []))
    overall_l2 = metrics.get("overall_relative_flux_l2")
    bin_l2 = list(metrics.get("time_bin_relative_flux_l2", []))
    complete = int(metrics.get("complete", 0)) == 1
    selected_step = int(metrics.get("selected_step", 0))
    nonlinear_gain = metrics.get("nonlinear_gain_vs_linear")

    def finite_all(values: Sequence[Any]) -> bool:
        return bool(values) and all(
            isinstance(value, (int, float, np.number)) and math.isfinite(float(value))
            for value in values
        )

    checks = [
        _check("complete", int(complete), "==", 1, complete),
        _check("selected_step", selected_step, ">", 0, selected_step > 0),
        _check(
            "overall_score_gain",
            overall_gain,
            ">=",
            thresholds.teacher_min_score_gain,
            _is_at_least(overall_gain, thresholds.teacher_min_score_gain),
        ),
        _check(
            "data_end_score_gain",
            data_end_gain,
            ">=",
            thresholds.teacher_min_score_gain,
            _is_at_least(data_end_gain, thresholds.teacher_min_score_gain),
        ),
        _check(
            "overall_flux_cosine",
            overall_cosine,
            ">=",
            thresholds.teacher_min_overall_flux_cosine,
            _is_at_least(overall_cosine, thresholds.teacher_min_overall_flux_cosine),
        ),
        _check(
            "all_bin_flux_cosines",
            bin_cosines,
            ">= each",
            thresholds.teacher_min_bin_flux_cosine,
            len(bin_cosines) == int(thresholds.teacher_expected_time_bins)
            and finite_all(bin_cosines)
            and min(float(value) for value in bin_cosines)
            >= thresholds.teacher_min_bin_flux_cosine,
        ),
        _check(
            "overall_relative_flux_l2",
            overall_l2,
            "<=",
            thresholds.teacher_max_overall_relative_flux_l2,
            _is_at_most_nonnegative(
                overall_l2, thresholds.teacher_max_overall_relative_flux_l2
            ),
        ),
        _check(
            "all_bin_relative_flux_l2",
            bin_l2,
            "<= each",
            thresholds.teacher_max_bin_relative_flux_l2,
            len(bin_l2) == int(thresholds.teacher_expected_time_bins)
            and finite_all(bin_l2)
            and min(float(value) for value in bin_l2) >= 0.0
            and max(float(value) for value in bin_l2)
            <= thresholds.teacher_max_bin_relative_flux_l2,
        ),
        _check(
            "nonlinear_component_recovered",
            nonlinear_gain,
            ">",
            0.0,
            _is_greater(nonlinear_gain, 0.0),
        ),
    ]
    return _finish_gate("positive_teacher", checks, "exact nonuniform-Dirichlet teacher")


def evaluate_null_control(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Require the stationary Dirichlet null to have no robust positive signal."""

    complete = int(metrics.get("complete", 0)) == 1
    comparator = str(metrics.get("comparator", ""))
    comparator_valid = comparator == "frozen_training_only_linear_spline_step0"
    lower_bound = metrics.get("audit_improvement_lower_bound")
    finite = _is_finite(lower_bound)
    checks = [
        _check("complete", int(complete), "==", 1, complete),
        _check(
            "frozen_linear_comparator",
            comparator,
            "==",
            "frozen_training_only_linear_spline_step0",
            comparator_valid,
        ),
        _check(
            "no_positive_audit_lower_bound",
            lower_bound,
            "<=",
            0.0,
            comparator_valid and finite and float(lower_bound) <= 0.0,
        ),
    ]
    return _finish_gate("null_control", checks, "stationary Dirichlet zero-score null")


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_greater(value: Any, threshold: float) -> bool:
    return _is_finite(value) and float(value) > float(threshold)


def _is_at_least(value: Any, threshold: float) -> bool:
    return _is_finite(value) and float(value) >= float(threshold)


def _is_at_most_nonnegative(value: Any, threshold: float) -> bool:
    return _is_finite(value) and 0.0 <= float(value) <= float(threshold)


def _selection_scope(result: Mapping[str, Any], scope: str) -> Mapping[str, Any]:
    metrics = result.get("selection_metrics", {})
    if isinstance(metrics, Mapping) and scope in metrics and isinstance(metrics[scope], Mapping):
        return metrics[scope]
    if isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
        selected = [row for row in metrics if _normalize_scope(row.get("scope")) == scope]
        if len(selected) == 1:
            return selected[0]
    raise ValueError(f"selection_metrics must contain exactly one {scope} row")


def _train_overall_delta(result: Mapping[str, Any]) -> float | None:
    metrics = result.get("train_metrics")
    if not isinstance(metrics, Mapping):
        return None
    row = metrics.get("overall", metrics)
    if not isinstance(row, Mapping):
        return None
    try:
        return _paired_delta(row, "linear")
    except ValueError:
        return None


def _declared_paths(result: Mapping[str, Any]) -> tuple[int, ...]:
    raw = result.get("audit_path_ids")
    if raw is None and isinstance(result.get("split_path_ids"), Mapping):
        raw = result["split_path_ids"].get("audit")
    if raw is None:
        raw = sorted(
            {
                int(row["path_id"])
                for row in result.get("audit_path_metrics", [])
                if "path_id" in row
            }
        )
    paths = tuple(sorted(int(value) for value in raw))
    if not paths or len(set(paths)) != len(paths) or any(value < 0 for value in paths):
        raise ValueError("audit_path_ids must contain unique non-negative paths")
    return paths


def _canonical_audit_rows(
    result: Mapping[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], tuple[int, ...], bool]:
    paths = _declared_paths(result)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    finite = True
    for raw in result.get("audit_path_metrics", []):
        if not isinstance(raw, Mapping):
            raise ValueError("audit_path_metrics rows must be mappings")
        row_seed = _row_seed(raw, seed)
        if row_seed != seed:
            raise ValueError("audit path metric model_seed does not match its task")
        bank = _normalize_bank(raw.get("audit_bank", raw.get("bank")))
        scope = _normalize_scope(raw.get("scope"))
        path = _path_id(raw)
        key = (bank, scope, path)
        if key in seen:
            raise ValueError("duplicate audit bank/scope/path row")
        seen.add(key)
        count = _state_count(raw)
        linear = _paired_delta(raw, "linear")
        zero = _paired_delta(raw, "zero")
        finite = finite and _finite_row(raw)
        rows.append(
            {
                "model_seed": seed,
                "path_id": path,
                "audit_bank": bank,
                "scope": scope,
                "state_count": count,
                "score_risk_delta_vs_linear": linear,
                "score_risk_delta_vs_zero": zero,
                "finite_fraction": 1.0 if _finite_row(raw) else 0.0,
            }
        )
    expected = {
        (bank, scope, path)
        for bank in AUDIT_BANKS
        for scope in AUDIT_SCOPES
        for path in paths
    }
    coverage = seen == expected
    return rows, paths, bool(coverage and finite)


def _canonical_stein_rows(
    result: Mapping[str, Any],
    seed: int,
    paths: Sequence[int],
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    finite = True
    for raw in result.get("stein_path_metrics", []):
        if not isinstance(raw, Mapping):
            raise ValueError("stein_path_metrics rows must be mappings")
        row_seed = _row_seed(raw, seed)
        if row_seed != seed:
            raise ValueError("Stein path metric model_seed does not match its task")
        bank = _normalize_bank(raw.get("stein_bank", raw.get("witness_bank", raw.get("bank"))), stein=True)
        path = _path_id(raw)
        key = (bank, path)
        if key in seen:
            raise ValueError("duplicate Stein bank/path row")
        seen.add(key)
        count = _state_count(raw)
        witness_contract = (
            int(raw.get("time_bin_count", -1)) == 5
            and int(raw.get("linear_witness_count", -1)) == 32
            and int(raw.get("quadratic_witness_count", -1)) == 32
            and int(raw.get("witness_count", -1)) == 64
            and str(raw.get("aggregation", ""))
            == "square-within-path-and-time-bin-then-equal-average"
        )
        raw_bin_counts = raw.get("time_bin_state_counts", [])
        witness_contract = witness_contract and isinstance(raw_bin_counts, Sequence) and len(raw_bin_counts) == 5 and all(
            int(value) > 0 for value in raw_bin_counts
        )
        improvement = _stein_improvement(raw)
        finite = finite and _finite_row(raw) and witness_contract
        rows.append(
            {
                "model_seed": seed,
                "path_id": path,
                "stein_bank": bank,
                "state_count": count,
                "stein_discrepancy_improvement": improvement,
                "finite_fraction": 1.0 if _finite_row(raw) else 0.0,
                "witness_contract": int(witness_contract),
            }
        )
    expected = {(bank, int(path)) for bank in STEIN_BANKS for path in paths}
    return rows, bool(seen == expected and finite)


def evaluate_score_seed(
    result: Mapping[str, Any],
    thresholds: DirichletScoreGateThresholds = DirichletScoreGateThresholds(),
) -> dict[str, Any]:
    """Evaluate one selected physical model seed.

    Canonical input::

        {
          "model_seed": int, "complete": 1, "selected_step": int,
          "selection_metrics": {
            "overall": paired-risk-row, "data_end": paired-risk-row},
          "train_metrics": {"overall": paired-risk-row},
          "audit_path_ids": [24 ids],
          "audit_path_metrics": [
            {"audit_bank": "audit_a|audit_b",
             "scope": "overall|data_end", "path_id": int,
             "state_count": int, paired risk fields...}],
          "stein_path_metrics": [
            {"stein_bank": "stein_a|stein_b", "path_id": int,
             "state_count": int, discrepancy fields...}]
        }

    Paired risk rows may provide ``score_risk_delta_vs_linear`` and
    ``score_risk_delta_vs_zero`` directly, risks, or risk sums plus a count.
    Positive delta always means that the full model is better.
    """

    parsing_error: str | None = None
    try:
        seed = _model_seed(result)
        complete = int(result.get("complete", result.get("task_complete", 0))) == 1
        selected_step = int(result.get("selected_step", 0))
        selection: dict[str, dict[str, float]] = {}
        for scope in AUDIT_SCOPES:
            row = _selection_scope(result, scope)
            selection[scope] = {
                "linear": _paired_delta(row, "linear"),
                "zero": _paired_delta(row, "zero"),
                "finite": float(_finite_row(row)),
            }
        audit_rows, paths, audit_rows_valid = _canonical_audit_rows(result, seed)
        stein_rows, stein_rows_valid = _canonical_stein_rows(result, seed, paths)
        train_delta = _train_overall_delta(result)
    except (KeyError, TypeError, ValueError) as exc:
        parsing_error = f"{type(exc).__name__}: {exc}"
        seed = int(result.get("model_seed", result.get("training_seed", -1)))
        complete = False
        selected_step = 0
        selection = {}
        audit_rows = []
        stein_rows = []
        paths = ()
        audit_rows_valid = False
        stein_rows_valid = False
        train_delta = None

    expected_paths = int(thresholds.expected_audit_paths)
    path_count_ok = len(paths) == expected_paths
    data_end_counts = [
        int(row["state_count"])
        for row in audit_rows
        if row["scope"] == "data_end"
    ]
    data_end_per_path_ok = bool(data_end_counts) and all(
        count == int(thresholds.expected_data_end_states_per_path)
        for count in data_end_counts
    )
    data_end_bank_totals = {
        bank: sum(
            int(row["state_count"])
            for row in audit_rows
            if row["audit_bank"] == bank and row["scope"] == "data_end"
        )
        for bank in AUDIT_BANKS
    }
    data_end_totals_ok = all(
        value == int(thresholds.expected_data_end_states)
        for value in data_end_bank_totals.values()
    )
    selection_positive = bool(selection) and all(
        float(selection[scope][baseline]) > float(thresholds.min_score_risk_delta)
        and float(selection[scope]["finite"]) == 1.0
        for scope in AUDIT_SCOPES
        for baseline in ("linear", "zero")
    )

    audit_means: dict[str, dict[str, dict[str, float | None]]] = {}
    for bank in AUDIT_BANKS:
        audit_means[bank] = {}
        for scope in AUDIT_SCOPES:
            selected = [
                row
                for row in audit_rows
                if row["audit_bank"] == bank and row["scope"] == scope
            ]
            audit_means[bank][scope] = {
                baseline: (
                    float(np.mean([row[f"score_risk_delta_vs_{baseline}"] for row in selected]))
                    if selected
                    else None
                )
                for baseline in ("linear", "zero")
            }
    audit_positive = audit_rows_valid and all(
        value is not None and float(value) > float(thresholds.min_score_risk_delta)
        for bank in AUDIT_BANKS
        for scope in AUDIT_SCOPES
        for value in audit_means[bank][scope].values()
    )
    stein_means: dict[str, float | None] = {}
    for bank in STEIN_BANKS:
        selected = [
            float(row["stein_discrepancy_improvement"])
            for row in stein_rows
            if row["stein_bank"] == bank
        ]
        stein_means[bank] = float(np.mean(selected)) if selected else None
    stein_positive = stein_rows_valid and all(
        value is not None and value > float(thresholds.min_stein_improvement)
        for value in stein_means.values()
    )
    finite = (
        parsing_error is None
        and all(float(row["finite_fraction"]) == 1.0 for row in audit_rows)
        and all(float(row["finite_fraction"]) == 1.0 for row in stein_rows)
        and all(
            float(selection[scope]["finite"]) == 1.0 for scope in AUDIT_SCOPES
        )
    )
    coverage = (
        parsing_error is None
        and path_count_ok
        and audit_rows_valid
        and stein_rows_valid
        and data_end_per_path_ok
        and data_end_totals_ok
    )
    seed_pass = (
        complete
        and selected_step > 0
        and finite
        and coverage
        and selection_positive
        and audit_positive
        and stein_positive
    )
    checks = [
        _check(
            "parse",
            parsing_error,
            "is",
            None,
            parsing_error is None,
            reason=parsing_error,
        ),
        _check("task_complete", int(complete), "==", 1, complete),
        _check("selected_step", selected_step, ">", 0, selected_step > 0),
        _check("all_metrics_finite", int(finite), "==", 1, finite),
        _check("audit_path_count", len(paths), "==", expected_paths, path_count_ok),
        _check(
            "audit_bank_scope_coverage",
            int(audit_rows_valid),
            "==",
            1,
            audit_rows_valid,
        ),
        _check(
            "stein_bank_path_coverage",
            int(stein_rows_valid),
            "==",
            1,
            stein_rows_valid,
        ),
        _check(
            "data_end_states_per_path",
            data_end_counts,
            "== each",
            thresholds.expected_data_end_states_per_path,
            data_end_per_path_ok,
        ),
        _check(
            "data_end_states_per_bank",
            data_end_bank_totals,
            "== each",
            thresholds.expected_data_end_states,
            data_end_totals_ok,
        ),
        _check(
            "selection_beats_linear_and_zero",
            selection,
            ">",
            thresholds.min_score_risk_delta,
            selection_positive,
        ),
        _check(
            "audit_beats_linear_and_zero",
            audit_means,
            "> each",
            thresholds.min_score_risk_delta,
            audit_positive,
        ),
        _check(
            "stein_banks_improve",
            stein_means,
            "> each",
            thresholds.min_stein_improvement,
            stein_positive,
        ),
    ]
    gate = _finish_gate("score_seed", checks, CLAIM_SCOPE)
    gate.update(
        {
            "model_seed": seed,
            "seed_signal_pass": int(seed_pass),
            "passed": int(seed_pass),
            "score_seed_pass": int(seed_pass),
            "selection_deltas": _jsonable(selection),
            "audit_mean_deltas": _jsonable(audit_means),
            "stein_mean_improvements": _jsonable(stein_means),
            "train_overall_delta_vs_linear": _jsonable(train_delta),
            "audit_path_ids": list(paths),
            "canonical_audit_path_metrics": _jsonable(audit_rows),
            "canonical_stein_path_metrics": _jsonable(stein_rows),
            "diagnostics": {
                "coverage_pass": int(coverage),
                "selection_point_signal": int(selection_positive),
                "audit_point_signal": int(audit_positive),
                "stein_point_signal": int(stein_positive),
                "train_point_signal": int(
                    train_delta is not None
                    and train_delta > float(thresholds.min_score_risk_delta)
                ),
            },
        }
    )
    return _jsonable(gate)


def _mean_or_none(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()) if array.size and np.isfinite(array).all() else None


def _median_or_none(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)) if array.size and np.isfinite(array).all() else None


def evaluate_score_study(
    seed_results: Sequence[Mapping[str, Any]],
    nonlinear_flux_cosines: Sequence[Mapping[str, Any]],
    thresholds: DirichletScoreGateThresholds = DirichletScoreGateThresholds(),
) -> dict[str, Any]:
    """Evaluate the complete three-seed physical implicit-score study."""

    seed_gates = [evaluate_score_seed(result, thresholds) for result in seed_results]
    seeds = [int(gate["model_seed"]) for gate in seed_gates]
    seed_set_valid = (
        len(seed_gates) == int(thresholds.expected_model_seeds)
        and len(set(seeds)) == len(seeds)
        and all(seed >= 0 for seed in seeds)
    )
    common_paths = (
        tuple(int(value) for value in seed_gates[0].get("audit_path_ids", []))
        if seed_gates
        else ()
    )
    common_path_set = bool(common_paths) and all(
        tuple(int(value) for value in gate.get("audit_path_ids", [])) == common_paths
        for gate in seed_gates
    )
    coverage = (
        seed_set_valid
        and common_path_set
        and len(common_paths) == int(thresholds.expected_audit_paths)
        and all(bool(int(gate["diagnostics"]["coverage_pass"])) for gate in seed_gates)
    )
    task_set_complete = seed_set_valid and all(
        bool(int(gate["subchecks"]["task_complete"]["passed"]))
        and bool(int(gate["subchecks"]["selected_step"]["passed"]))
        and bool(int(gate["subchecks"]["all_metrics_finite"]["passed"]))
        for gate in seed_gates
    )
    passing_seed_count = sum(int(gate.get("seed_signal_pass", 0)) for gate in seed_gates)

    audit_rows: list[dict[str, Any]] = []
    stein_rows: list[dict[str, Any]] = []
    for gate in seed_gates:
        audit_rows.extend(dict(row) for row in gate["canonical_audit_path_metrics"])
        stein_rows.extend(dict(row) for row in gate["canonical_stein_path_metrics"])

    audit_statistics: dict[str, dict[str, dict[str, Any]]] = {}
    bootstrap_ok = bool(coverage)
    for bank in AUDIT_BANKS:
        audit_statistics[bank] = {}
        for scope in AUDIT_SCOPES:
            selected = [
                row
                for row in audit_rows
                if row["audit_bank"] == bank and row["scope"] == scope
            ]
            scope_stats: dict[str, Any] = {}
            for baseline in ("linear", "zero"):
                key = f"score_risk_delta_vs_{baseline}"
                values = [float(row[key]) for row in selected]
                per_seed_values = [
                    _mean_or_none(
                        [
                            float(row[key])
                            for row in selected
                            if int(row["model_seed"]) == seed
                        ]
                    )
                    for seed in seeds
                ]
                try:
                    interval = bootstrap_whole_path_delta(
                        selected,
                        value_key=key,
                        reps=thresholds.bootstrap_reps,
                        confidence=thresholds.bootstrap_confidence,
                        # A single frozen path-resampling plan is shared by
                        # every probe bank, scope, baseline, witness bank, and
                        # cross-seed statistic.  Thus one sampled path carries
                        # all correlated measurements in a replicate.
                        seed=thresholds.bootstrap_seed,
                        expected_model_seeds=seeds,
                        expected_path_ids=common_paths,
                    )
                    mom = median_of_means_whole_path_delta(
                        selected,
                        value_key=key,
                        groups=thresholds.median_of_means_groups,
                        seed=thresholds.bootstrap_seed,
                        expected_model_seeds=seeds,
                        expected_path_ids=common_paths,
                    )
                except (KeyError, TypeError, ValueError):
                    interval = {}
                    mom = {}
                    bootstrap_ok = False
                scope_stats[baseline] = {
                    "mean_delta": _mean_or_none(values),
                    "median_seed_delta": _median_or_none(
                        [value for value in per_seed_values if value is not None]
                    ),
                    "per_seed_deltas": _jsonable(per_seed_values),
                    "bootstrap": interval,
                    "median_of_means": mom,
                }
            audit_statistics[bank][scope] = scope_stats

    stein_statistics: dict[str, Any] = {}
    for bank in STEIN_BANKS:
        selected = [row for row in stein_rows if row["stein_bank"] == bank]
        values = [float(row["stein_discrepancy_improvement"]) for row in selected]
        per_seed_values = [
            _mean_or_none(
                [
                    float(row["stein_discrepancy_improvement"])
                    for row in selected
                    if int(row["model_seed"]) == seed
                ]
            )
            for seed in seeds
        ]
        try:
            interval = bootstrap_whole_path_delta(
                selected,
                value_key="stein_discrepancy_improvement",
                reps=thresholds.bootstrap_reps,
                confidence=thresholds.bootstrap_confidence,
                seed=thresholds.bootstrap_seed,
                expected_model_seeds=seeds,
                expected_path_ids=common_paths,
            )
        except (KeyError, TypeError, ValueError):
            interval = {}
            bootstrap_ok = False
        stein_statistics[bank] = {
            "mean_improvement": _mean_or_none(values),
            "median_seed_improvement": _median_or_none(
                [value for value in per_seed_values if value is not None]
            ),
            "per_seed_improvements": _jsonable(per_seed_values),
            "bootstrap": interval,
        }

    try:
        cosine = bootstrap_cross_seed_cosine(
            nonlinear_flux_cosines,
            reps=thresholds.bootstrap_reps,
            confidence=thresholds.bootstrap_confidence,
            seed=thresholds.bootstrap_seed,
            expected_model_seeds=seeds,
            expected_path_ids=common_paths,
        )
        cosine_ok = True
    except (KeyError, TypeError, ValueError):
        cosine = {}
        cosine_ok = False

    def all_audit(predicate: Any, *, baselines: Sequence[str]) -> bool:
        return all(
            predicate(audit_statistics[bank][scope][baseline])
            for bank in AUDIT_BANKS
            for scope in AUDIT_SCOPES
            for baseline in baselines
        )

    median_positive = all_audit(
        lambda item: _is_greater(
            item.get("median_seed_delta"), thresholds.min_score_risk_delta
        ),
        baselines=("linear", "zero"),
    )
    bootstrap_positive = bootstrap_ok and all_audit(
        lambda item: _is_greater(
            item.get("bootstrap", {}).get("lower_bound"),
            thresholds.min_score_risk_delta,
        ),
        baselines=("linear", "zero"),
    )
    # The four-group robustness check is primary versus the frozen linear
    # baseline.  Beating linear already entails the scientifically relevant
    # state-dependent comparison; the zero comparator remains fully gated by
    # seed, median, and bootstrap checks.
    mom_positive = all_audit(
        lambda item: _is_greater(
            item.get("median_of_means", {}).get("median_of_means"),
            thresholds.min_score_risk_delta,
        ),
        baselines=("linear",),
    )
    stein_mean_positive = all(
        _is_greater(item.get("median_seed_improvement"), thresholds.min_stein_improvement)
        for item in stein_statistics.values()
    )
    stein_bootstrap_positive = all(
        _is_greater(
            item.get("bootstrap", {}).get("lower_bound"),
            thresholds.min_stein_improvement,
        )
        for item in stein_statistics.values()
    )
    cosine_median = cosine.get("point_median")
    cosine_lcb = cosine.get("lower_bound")
    cosine_pass = (
        cosine_ok
        and _is_at_least(cosine_median, thresholds.nonlinear_flux_cosine)
        and _is_at_least(cosine_lcb, thresholds.nonlinear_flux_cosine_lcb)
    )
    checks = [
        _check(
            "model_seed_count",
            len(seed_gates),
            "== unique",
            thresholds.expected_model_seeds,
            seed_set_valid,
        ),
        _check("all_tasks_complete", int(task_set_complete), "==", 1, task_set_complete),
        _check("complete_audit_coverage", int(coverage), "==", 1, coverage),
        _check(
            "passing_model_seed_count",
            passing_seed_count,
            ">=",
            thresholds.min_passing_model_seeds,
            passing_seed_count >= thresholds.min_passing_model_seeds,
        ),
        _check(
            "median_audit_deltas",
            audit_statistics,
            "> each",
            thresholds.min_score_risk_delta,
            median_positive,
        ),
        _check(
            "audit_bootstrap_lower_bounds",
            {
                bank: {
                    scope: {
                        baseline: audit_statistics[bank][scope][baseline]["bootstrap"].get(
                            "lower_bound"
                        )
                        for baseline in ("linear", "zero")
                    }
                    for scope in AUDIT_SCOPES
                }
                for bank in AUDIT_BANKS
            },
            "> each",
            thresholds.min_score_risk_delta,
            bootstrap_positive,
        ),
        _check(
            "four_group_median_of_means",
            {
                bank: {
                    scope: audit_statistics[bank][scope]["linear"][
                        "median_of_means"
                    ].get("median_of_means")
                    for scope in AUDIT_SCOPES
                }
                for bank in AUDIT_BANKS
            },
            "> each",
            thresholds.min_score_risk_delta,
            mom_positive,
        ),
        _check(
            "stein_mean_improvements",
            {
                bank: item.get("median_seed_improvement")
                for bank, item in stein_statistics.items()
            },
            "> each",
            thresholds.min_stein_improvement,
            stein_mean_positive,
        ),
        _check(
            "stein_bootstrap_lower_bounds",
            {
                bank: item.get("bootstrap", {}).get("lower_bound")
                for bank, item in stein_statistics.items()
            },
            "> each",
            thresholds.min_stein_improvement,
            stein_bootstrap_positive,
        ),
        _check(
            "nonlinear_flux_cosine_median",
            cosine_median,
            ">=",
            thresholds.nonlinear_flux_cosine,
            cosine_ok
            and _is_at_least(cosine_median, thresholds.nonlinear_flux_cosine),
        ),
        _check(
            "nonlinear_flux_cosine_bootstrap_lcb",
            cosine_lcb,
            ">=",
            thresholds.nonlinear_flux_cosine_lcb,
            cosine_ok
            and _is_at_least(cosine_lcb, thresholds.nonlinear_flux_cosine_lcb),
        ),
    ]
    gate = _finish_gate("score", checks, CLAIM_SCOPE)

    bank_point_signals = {
        bank: all(
            _is_greater(
                audit_statistics[bank][scope]["linear"].get("median_seed_delta"),
                thresholds.min_score_risk_delta,
            )
            for scope in AUDIT_SCOPES
        )
        for bank in AUDIT_BANKS
    }
    zero_point_signal = all_audit(
        lambda item: _is_greater(
            item.get("median_seed_delta"), thresholds.min_score_risk_delta
        ),
        baselines=("zero",),
    )
    linear_point_signal = all(bank_point_signals.values())
    selection_point_seeds = sum(
        int(bool(gate_item["diagnostics"]["selection_point_signal"]))
        for gate_item in seed_gates
    )
    train_point_seeds = sum(
        int(bool(gate_item["diagnostics"]["train_point_signal"]))
        for gate_item in seed_gates
    )
    gate.update(
        {
            "seed_gates": seed_gates,
            "audit_statistics": _jsonable(audit_statistics),
            "stein_statistics": _jsonable(stein_statistics),
            "nonlinear_flux_cosine": _jsonable(cosine),
            "diagnostics": {
                "task_set_complete": int(task_set_complete),
                "coverage_pass": int(coverage),
                "passing_model_seeds": int(passing_seed_count),
                "bank_point_signals": {
                    bank: int(value) for bank, value in bank_point_signals.items()
                },
                "trace_bank_sign_disagreement": int(
                    len(set(bank_point_signals.values())) > 1
                ),
                "zero_point_signal": int(zero_point_signal),
                "linear_point_signal": int(linear_point_signal),
                "objective_robust_signal": int(
                    median_positive and bootstrap_positive and mom_positive
                ),
                "stein_point_signal": int(stein_mean_positive),
                "stein_robust_signal": int(
                    stein_mean_positive and stein_bootstrap_positive
                ),
                "cosine_point_signal": int(
                    _is_at_least(cosine_median, thresholds.nonlinear_flux_cosine)
                ),
                "cosine_robust_signal": int(cosine_pass),
                "selection_point_seed_count": int(selection_point_seeds),
                "train_point_seed_count": int(train_point_seeds),
                "bootstrap_and_mom_pass": int(bootstrap_positive and mom_positive),
            },
        }
    )
    return _jsonable(gate)


_DECISION_ACTIONS: dict[ImplicitScoreDecision, str] = {
    ImplicitScoreDecision.IMPLICIT_SCORE_SIGNAL: (
        "plan a separately named positive-time score-to-flux one-image reconstruction patch"
    ),
    ImplicitScoreDecision.LINEAR_SPATIOTEMPORAL_ONLY: (
        "retain the frozen linear baseline result; do not sample or enlarge the same neural run"
    ),
    ImplicitScoreDecision.OBJECTIVE_ONLY_SIGNAL: (
        "audit the score objective and Stein witnesses before further physical training"
    ),
    ImplicitScoreDecision.TRACE_ESTIMATOR_INCONCLUSIVE: (
        "repair or increase independent trace-estimator precision without sampling"
    ),
    ImplicitScoreDecision.BOUNDARY_OR_OUTLIER_ARTIFACT: (
        "inspect path-level boundary and outlier evidence before changing the model"
    ),
    ImplicitScoreDecision.PATH_MEMORIZATION_ONLY: (
        "increase genuinely independent paths or add a theory-justified variance reduction"
    ),
    ImplicitScoreDecision.NO_DETECTABLE_IMPLICIT_SCORE: (
        "revisit the positive-time score target or model class; do not repeat unchanged"
    ),
    ImplicitScoreDecision.OPERATOR_INVALID: (
        "repair the Dirichlet generator/operator implementation before cache or training"
    ),
    ImplicitScoreDecision.CACHE_INVALID: (
        "repair the positive-time state cache before optimization"
    ),
    ImplicitScoreDecision.OPTIMIZATION_PIPELINE_INVALID: (
        "repair the synthetic controls or incomplete optimization pipeline before physical claims"
    ),
}


def decide_score_learnability(
    *,
    preflight_gate: bool | int | Mapping[str, Any],
    cache_gate: bool | int | Mapping[str, Any],
    controls_gate: bool | int | Mapping[str, Any],
    score_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Map closed gate evidence to exactly one approved terminal outcome."""

    preflight = _passed(preflight_gate)
    cache = _passed(cache_gate)
    controls = _passed(controls_gate)
    score = _passed(score_gate)
    diagnostics = dict(score_gate.get("diagnostics", {}))
    if not preflight:
        decision = ImplicitScoreDecision.OPERATOR_INVALID
    elif not cache:
        decision = ImplicitScoreDecision.CACHE_INVALID
    elif not controls or not bool(int(diagnostics.get("task_set_complete", 0))):
        decision = ImplicitScoreDecision.OPTIMIZATION_PIPELINE_INVALID
    elif score:
        decision = ImplicitScoreDecision.IMPLICIT_SCORE_SIGNAL
    elif bool(int(diagnostics.get("trace_bank_sign_disagreement", 0))):
        decision = ImplicitScoreDecision.TRACE_ESTIMATOR_INCONCLUSIVE
    elif bool(int(diagnostics.get("objective_robust_signal", 0))) and not bool(
        int(diagnostics.get("stein_robust_signal", 0))
    ):
        decision = ImplicitScoreDecision.OBJECTIVE_ONLY_SIGNAL
    elif (
        bool(int(diagnostics.get("linear_point_signal", 0)))
        and bool(int(diagnostics.get("stein_point_signal", 0)))
        and bool(int(diagnostics.get("cosine_point_signal", 0)))
        and not (
            bool(int(diagnostics.get("bootstrap_and_mom_pass", 0)))
            and bool(int(diagnostics.get("stein_robust_signal", 0)))
            and bool(int(diagnostics.get("cosine_robust_signal", 0)))
        )
    ):
        decision = ImplicitScoreDecision.BOUNDARY_OR_OUTLIER_ARTIFACT
    elif bool(int(diagnostics.get("zero_point_signal", 0))) and not bool(
        int(diagnostics.get("linear_point_signal", 0))
    ):
        decision = ImplicitScoreDecision.LINEAR_SPATIOTEMPORAL_ONLY
    elif (
        int(diagnostics.get("selection_point_seed_count", 0)) >= 2
        and int(diagnostics.get("train_point_seed_count", 0)) >= 2
        and not bool(int(diagnostics.get("linear_point_signal", 0)))
    ):
        decision = ImplicitScoreDecision.PATH_MEMORIZATION_ONLY
    else:
        decision = ImplicitScoreDecision.NO_DETECTABLE_IMPLICIT_SCORE
    return {
        "decision": decision.value,
        "recommended_next_action": _DECISION_ACTIONS[decision],
        "sampling_performed": 0,
        "sampling_authorized": 0,
        "claim_scope": CLAIM_SCOPE,
    }


def evaluate_dirichlet_score_gates(
    *,
    preflight_gate: bool | int | Mapping[str, Any],
    cache_gate: bool | int | Mapping[str, Any],
    controls_gate: bool | int | Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    nonlinear_flux_cosines: Sequence[Mapping[str, Any]],
    require_gate: str = "none",
    thresholds: DirichletScoreGateThresholds = DirichletScoreGateThresholds(),
) -> dict[str, Any]:
    """Aggregate preflight, cache, controls, and physical score evidence."""

    required = str(require_gate).strip().lower()
    if required not in {"none", "preflight", "cache", "controls", "score"}:
        raise ValueError(
            "require_gate must be none, preflight, cache, controls, or score"
        )
    score_gate = evaluate_score_study(seed_results, nonlinear_flux_cosines, thresholds)
    preflight = _passed(preflight_gate)
    cache = _passed(cache_gate)
    controls = _passed(controls_gate)
    score = bool(int(score_gate["passed"]))
    cumulative = {
        "preflight": preflight,
        "cache": preflight and cache,
        "controls": preflight and cache and controls,
        "score": preflight and cache and controls and score,
    }
    decision = decide_score_learnability(
        preflight_gate=preflight_gate,
        cache_gate=cache_gate,
        controls_gate=controls_gate,
        score_gate=score_gate,
    )
    required_pass = True if required == "none" else cumulative[required]
    return _jsonable(
        {
            "schema": "experiment12-d0-dirichlet-score-gate",
            "schema_version": 1,
            "required_gate": required,
            "required_gate_pass": int(required_pass),
            "preflight": _jsonable(preflight_gate),
            "cache": _jsonable(cache_gate),
            "controls": _jsonable(controls_gate),
            "score": score_gate,
            "cumulative_pass": {key: int(value) for key, value in cumulative.items()},
            "decision": decision,
            "thresholds": asdict(thresholds),
            "claim_scope": CLAIM_SCOPE,
            "excluded_claims": [
                "finite-step Bayes reversal of the limiter",
                "time-zero density or reconstruction below t_min",
                "conditional covariance calibration",
                "multi-image sample quality",
                "spatial Dirichlet-Ferguson convergence",
            ],
            "sampling_performed": 0,
            "sampling_authorized": 0,
        }
    )


__all__ = [
    "AUDIT_BANKS",
    "AUDIT_SCOPES",
    "CLAIM_SCOPE",
    "STEIN_BANKS",
    "DirichletScoreGateThresholds",
    "ImplicitScoreDecision",
    "bootstrap_cross_seed_cosine",
    "bootstrap_whole_path_delta",
    "decide_score_learnability",
    "derive_resampling_seed",
    "evaluate_control_bundle",
    "evaluate_dirichlet_score_gates",
    "evaluate_null_control",
    "evaluate_positive_teacher_control",
    "evaluate_score_seed",
    "evaluate_score_study",
    "median_of_means_whole_path_delta",
]
