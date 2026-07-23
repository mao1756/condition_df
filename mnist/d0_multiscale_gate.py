from __future__ import annotations

"""Metrics and fail-closed gates for the D0 multiscale learnability pilot.

This module deliberately contains no cache builder, training loop, or reverse
sampler.  It evaluates already-computed temporal-block predictions while
preserving three important contracts:

* all splits are isolated by whole forward-path id;
* the time-only baseline is fitted from training paths only; and
* uncertainty is bootstrapped over whole paths, never over correlated slices.

The orchestration CLI owns checkpoint selection and artifact provenance.  The
small primitives here are reusable by that CLI and by report-only analysis.
"""

import hashlib
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mnist.d0_one_image_gate import atomic_write_csv, atomic_write_json


TIME_BIN_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SPLIT_NAMES: tuple[str, ...] = ("train", "selection", "audit")
CLAIM_SCOPE = (
    "held-out temporal-block direct-residual learnability for one fixed image"
)


@dataclass(frozen=True)
class MultiscaleGateThresholds:
    """Frozen acceptance thresholds for the optimization-only pilot."""

    expected_training_seeds: int = 3
    min_passing_seeds: int = 2
    min_data_end_slices: int = 192
    expected_audit_paths: int = 12
    expected_data_end_slices_per_path: int = 16
    min_prediction_gain: float = 0.0
    min_tau_baseline_gain: float = 0.0
    bootstrap_confidence: float = 0.90
    bootstrap_reps: int = 10_000
    bootstrap_seed: int = 260726
    teacher_min_gain: float = 0.90
    memorization_train_gain: float = 0.50
    data_end_bin_index: int = 4
    elementary_stride: int = 1

    def __post_init__(self) -> None:
        if int(self.expected_training_seeds) <= 0:
            raise ValueError("expected_training_seeds must be positive")
        if not 1 <= int(self.min_passing_seeds) <= int(self.expected_training_seeds):
            raise ValueError(
                "min_passing_seeds must be in [1, expected_training_seeds]"
            )
        if int(self.min_data_end_slices) <= 0:
            raise ValueError("min_data_end_slices must be positive")
        if int(self.expected_audit_paths) <= 0:
            raise ValueError("expected_audit_paths must be positive")
        if int(self.expected_data_end_slices_per_path) <= 0:
            raise ValueError("expected_data_end_slices_per_path must be positive")
        for name, value in {
            "min_prediction_gain": self.min_prediction_gain,
            "min_tau_baseline_gain": self.min_tau_baseline_gain,
        }.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if (
            not math.isfinite(float(self.bootstrap_confidence))
            or not 0.5 < float(self.bootstrap_confidence) < 1.0
        ):
            raise ValueError("bootstrap_confidence must be finite and in (0.5, 1)")
        if int(self.bootstrap_reps) <= 0:
            raise ValueError("bootstrap_reps must be positive")
        if int(self.bootstrap_seed) < 0:
            raise ValueError("bootstrap_seed must be non-negative")
        if (
            not math.isfinite(float(self.teacher_min_gain))
            or not 0.0 < float(self.teacher_min_gain) <= 1.0
        ):
            raise ValueError("teacher_min_gain must be finite and in (0, 1]")
        if (
            not math.isfinite(float(self.memorization_train_gain))
            or not 0.0 <= float(self.memorization_train_gain) <= 1.0
        ):
            raise ValueError(
                "memorization_train_gain must be finite and in [0, 1]"
            )
        if not 0 <= int(self.data_end_bin_index) < len(TIME_BIN_EDGES) - 1:
            raise ValueError("data_end_bin_index is outside the fixed time bins")
        if int(self.elementary_stride) <= 0:
            raise ValueError("elementary_stride must be positive")


@dataclass(frozen=True)
class TauBinBaseline:
    """A train-only, piecewise-constant conditional-mean baseline."""

    bin_edges: tuple[float, ...]
    means: np.ndarray
    counts: tuple[int, ...]
    empty_bin_policy: str = "global-train-mean"

    def __post_init__(self) -> None:
        edges = _validate_bin_edges(self.bin_edges)
        means = np.asarray(self.means, dtype=np.float64)
        counts = tuple(int(value) for value in self.counts)
        if means.ndim != 2 or means.shape[0] != len(edges) - 1:
            raise ValueError("tau baseline means must have shape (bins, features)")
        if len(counts) != means.shape[0] or any(value < 0 for value in counts):
            raise ValueError("tau baseline counts must be non-negative and match bins")
        if means.shape[1] <= 0 or not np.isfinite(means).all():
            raise ValueError("tau baseline means must be finite and non-empty")
        if self.empty_bin_policy != "global-train-mean":
            raise ValueError("unsupported empty-bin policy")
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "means", np.ascontiguousarray(means))
        object.__setattr__(self, "counts", counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bin_edges": list(self.bin_edges),
            "means": self.means.tolist(),
            "counts": list(self.counts),
            "empty_bin_policy": self.empty_bin_policy,
            "fit_scope": "training paths only",
        }


class LearnabilityDecision(str, Enum):
    CACHE_INVALID = "cache_invalid"
    OPTIMIZATION_PIPELINE_INVALID = "optimization_pipeline_invalid"
    ELEMENTARY_SIGNAL = "elementary_signal"
    COARSE_ONLY_SIGNAL = "coarse_only_signal"
    PATH_MEMORIZATION_ONLY = "path_memorization_only"
    NO_DETECTABLE_CONDITIONAL_SIGNAL = "no_detectable_conditional_signal"
    NO_CONFIRMED_CONDITIONAL_SIGNAL = "no_confirmed_conditional_signal"
    INCONCLUSIVE = "inconclusive"


def _normalize_study_profile(value: str) -> str:
    profile = str(value).strip().lower()
    if profile not in {"pilot", "confirmation"}:
        raise ValueError("study_profile must be pilot or confirmation")
    return profile


def _binary_flag(value: bool | int, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    raise ValueError(f"{name} must be boolean or 0/1")


def _validate_bin_edges(edges: Sequence[float]) -> tuple[float, ...]:
    out = tuple(float(value) for value in edges)
    if len(out) < 2 or not all(math.isfinite(value) for value in out):
        raise ValueError("time-bin edges must contain at least two finite values")
    if out[0] != 0.0 or out[-1] != 1.0:
        raise ValueError("time-bin edges must start at 0 and end at 1")
    if any(right <= left for left, right in zip(out, out[1:])):
        raise ValueError("time-bin edges must be strictly increasing")
    return out


def _as_matrix(value: Any, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] <= 0 or out.shape[1] <= 0:
        raise ValueError(f"{name} must have shape (slices, features)")
    return np.ascontiguousarray(out)


def _as_tau(value: Any, count: int) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size != int(count) or not np.isfinite(out).all():
        raise ValueError("tau_fractions must be finite and align with slices")
    tolerance = 1e-12
    if np.any(out < -tolerance) or np.any(out > 1.0 + tolerance):
        raise ValueError("tau_fractions must lie in [0, 1]")
    return np.clip(out, 0.0, 1.0)


def _as_path_ids(value: Any, count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != int(count):
        raise ValueError("path_ids must be one-dimensional and align with slices")
    if not np.issubdtype(raw.dtype, np.integer):
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("path_ids must be integers")
    out = np.asarray(raw, dtype=np.int64)
    if np.any(out < 0):
        raise ValueError("path_ids must be non-negative")
    return out


def _bin_indices(tau: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    validated = _validate_bin_edges(edges)
    # searchsorted assigns tau==1 to the last bin after clipping.
    return np.minimum(
        np.searchsorted(np.asarray(validated), tau, side="right") - 1,
        len(validated) - 2,
    ).astype(np.int64)


def fit_tau_bin_baseline(
    targets: Any,
    tau_fractions: Any,
    *,
    bin_edges: Sequence[float] = TIME_BIN_EDGES,
) -> TauBinBaseline:
    """Fit a time-only target mean without consulting non-training rows."""

    matrix = _as_matrix(targets, "targets")
    if not np.isfinite(matrix).all():
        raise ValueError("training targets for the tau baseline must be finite")
    tau = _as_tau(tau_fractions, matrix.shape[0])
    edges = _validate_bin_edges(bin_edges)
    assignments = _bin_indices(tau, edges)
    global_mean = matrix.mean(axis=0, dtype=np.float64)
    means: list[np.ndarray] = []
    counts: list[int] = []
    for index in range(len(edges) - 1):
        selected = assignments == index
        count = int(np.count_nonzero(selected))
        counts.append(count)
        means.append(
            matrix[selected].mean(axis=0, dtype=np.float64)
            if count > 0
            else global_mean.copy()
        )
    return TauBinBaseline(
        bin_edges=edges,
        means=np.stack(means, axis=0),
        counts=tuple(counts),
    )


def predict_tau_bin_baseline(
    baseline: TauBinBaseline,
    tau_fractions: Any,
) -> np.ndarray:
    tau = np.asarray(tau_fractions, dtype=np.float64).reshape(-1)
    tau = _as_tau(tau, tau.size)
    assignments = _bin_indices(tau, baseline.bin_edges)
    return np.ascontiguousarray(baseline.means[assignments])


def _metric_summary(
    target: np.ndarray,
    prediction: np.ndarray,
    tau_prediction: np.ndarray,
) -> dict[str, Any]:
    if (
        target.ndim != 2
        or target.shape != prediction.shape
        or target.shape != tau_prediction.shape
    ):
        raise ValueError("target, prediction, and tau baseline must have matching matrices")
    finite = (
        np.isfinite(target)
        & np.isfinite(prediction)
        & np.isfinite(tau_prediction)
    )
    result: dict[str, Any] = {
        "slice_count": int(target.shape[0]),
        "feature_count": int(target.shape[1]),
        "finite_fraction": float(finite.mean()) if finite.size else float("nan"),
        "target_finite_fraction": (
            float(np.isfinite(target).mean()) if target.size else float("nan")
        ),
    }
    empty_or_bad = target.size == 0 or not bool(finite.all())
    if empty_or_bad:
        result.update(
            {
                "primary_mse": float("nan"),
                "zero_baseline_mse": float("nan"),
                "tau_baseline_mse": float("nan"),
                "prediction_gain": float("nan"),
                "prediction_gain_vs_tau_baseline": float("nan"),
                "target_rms": float("nan"),
                "prediction_rms": float("nan"),
                "residual_rms": float("nan"),
                "target_prediction_covariance": float("nan"),
                "residual_covariance_trace": float("nan"),
                "residual_covariance_trace_per_feature": float("nan"),
                "squared_error_sum": float("nan"),
                "zero_squared_error_sum": float("nan"),
                "tau_baseline_squared_error_sum": float("nan"),
                "loss_delta_vs_zero": float("nan"),
                "loss_delta_vs_tau_baseline": float("nan"),
                "element_count": int(target.size),
            }
        )
        return result

    residual = prediction - target
    tau_residual = tau_prediction - target
    squared_error_sum = float(np.square(residual).sum(dtype=np.float64))
    zero_sum = float(np.square(target).sum(dtype=np.float64))
    tau_sum = float(np.square(tau_residual).sum(dtype=np.float64))
    element_count = int(target.size)
    primary_mse = squared_error_sum / float(element_count)
    zero_mse = zero_sum / float(element_count)
    tau_mse = tau_sum / float(element_count)
    target_centered = target - float(target.mean())
    prediction_centered = prediction - float(prediction.mean())
    covariance = float(np.mean(target_centered * prediction_centered))
    trace = float(np.var(residual, axis=0, ddof=0).sum())
    result.update(
        {
            "primary_mse": primary_mse,
            "zero_baseline_mse": zero_mse,
            "tau_baseline_mse": tau_mse,
            "prediction_gain": (
                1.0 - squared_error_sum / zero_sum if zero_sum > 0.0 else float("nan")
            ),
            "prediction_gain_vs_tau_baseline": (
                1.0 - squared_error_sum / tau_sum if tau_sum > 0.0 else float("nan")
            ),
            "target_rms": float(math.sqrt(zero_mse)),
            "prediction_rms": float(math.sqrt(np.square(prediction).mean())),
            "residual_rms": float(math.sqrt(primary_mse)),
            "target_prediction_covariance": covariance,
            "residual_covariance_trace": trace,
            "residual_covariance_trace_per_feature": trace / float(target.shape[1]),
            "squared_error_sum": squared_error_sum,
            "zero_squared_error_sum": zero_sum,
            "tau_baseline_squared_error_sum": tau_sum,
            "loss_delta_vs_zero": zero_sum - squared_error_sum,
            "loss_delta_vs_tau_baseline": tau_sum - squared_error_sum,
            "element_count": element_count,
        }
    )
    return result


def _validate_whole_path_splits(
    path_ids: np.ndarray,
    split_path_ids: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    missing_names = sorted(set(SPLIT_NAMES).difference(split_path_ids))
    extra_names = sorted(set(split_path_ids).difference(SPLIT_NAMES))
    if missing_names or extra_names:
        raise ValueError(
            "split_path_ids must contain exactly train, selection, and audit"
        )
    normalized: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for name in SPLIT_NAMES:
        values = np.asarray(split_path_ids[name], dtype=np.int64).reshape(-1)
        if values.size == 0 or np.any(values < 0):
            raise ValueError(f"{name} path split must be non-empty and non-negative")
        if np.unique(values).size != values.size:
            raise ValueError(f"{name} path split contains duplicates")
        overlap = seen.intersection(int(value) for value in values.tolist())
        if overlap:
            raise ValueError("whole-path splits overlap")
        seen.update(int(value) for value in values.tolist())
        normalized[name] = np.sort(values)
    available = set(int(value) for value in np.unique(path_ids).tolist())
    if seen != available:
        raise ValueError("whole-path splits must cover every and only available path id")
    return normalized


def compute_multiscale_split_metrics(
    targets: Any,
    predictions: Any,
    tau_fractions: Any,
    path_ids: Any,
    split_path_ids: Mapping[str, Sequence[int]],
    *,
    stride: int,
    training_seed: int,
    selected_step: int,
    complete: bool = True,
    bin_edges: Sequence[float] = TIME_BIN_EDGES,
) -> dict[str, Any]:
    """Evaluate one selected stride/seed checkpoint on all whole-path splits.

    The tau-only baseline is fitted exactly once from the rows assigned to the
    training split.  Returned per-path sufficient statistics are suitable for
    path-cluster bootstrap without retaining feature tensors.
    """

    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    if int(selected_step) < 0:
        raise ValueError("selected_step must be non-negative")
    target = _as_matrix(targets, "targets")
    prediction = _as_matrix(predictions, "predictions")
    if target.shape != prediction.shape:
        raise ValueError("targets and predictions must have the same shape")
    tau = _as_tau(tau_fractions, target.shape[0])
    paths = _as_path_ids(path_ids, target.shape[0])
    edges = _validate_bin_edges(bin_edges)
    splits = _validate_whole_path_splits(paths, split_path_ids)
    train_mask = np.isin(paths, splits["train"])
    baseline = fit_tau_bin_baseline(target[train_mask], tau[train_mask], bin_edges=edges)
    tau_prediction = predict_tau_bin_baseline(baseline, tau)
    assignments = _bin_indices(tau, edges)
    common = {
        "stride": int(stride),
        "training_seed": int(training_seed),
        "selected_step": int(selected_step),
        "task_complete": int(bool(complete)),
    }
    split_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []

    for split_name in SPLIT_NAMES:
        split_mask = np.isin(paths, splits[split_name])
        split_rows.append(
            {
                **common,
                "split": split_name,
                "bin": "overall",
                "bin_index": -1,
                "tau_fraction_lo": 0.0,
                "tau_fraction_hi": 1.0,
                **_metric_summary(
                    target[split_mask],
                    prediction[split_mask],
                    tau_prediction[split_mask],
                ),
            }
        )
        for bin_index, (lo, hi) in enumerate(zip(edges, edges[1:])):
            mask = split_mask & (assignments == int(bin_index))
            time_rows.append(
                {
                    **common,
                    "split": split_name,
                    "bin": f"tau_bin{bin_index}",
                    "bin_index": int(bin_index),
                    "tau_fraction_lo": float(lo),
                    "tau_fraction_hi": float(hi),
                    **_metric_summary(
                        target[mask], prediction[mask], tau_prediction[mask]
                    ),
                }
            )

        for path_id in splits[split_name].tolist():
            path_mask = paths == int(path_id)
            scopes = [(-1, "overall", path_mask)] + [
                (
                    int(bin_index),
                    f"tau_bin{bin_index}",
                    path_mask & (assignments == int(bin_index)),
                )
                for bin_index in range(len(edges) - 1)
            ]
            for bin_index, bin_name, mask in scopes:
                path_rows.append(
                    {
                        **common,
                        "split": split_name,
                        "path_id": int(path_id),
                        "bin": bin_name,
                        "bin_index": int(bin_index),
                        **_metric_summary(
                            target[mask], prediction[mask], tau_prediction[mask]
                        ),
                    }
                )

    return {
        "schema": "experiment12-d0-multiscale-seed-metrics",
        "schema_version": 1,
        **common,
        "time_bin_edges": list(edges),
        "tau_baseline": baseline.to_dict(),
        "split_path_ids": {
            name: values.tolist() for name, values in splits.items()
        },
        "split_metrics": split_rows,
        "time_bin_metrics": time_rows,
        "per_path_metrics": path_rows,
    }


def derive_bootstrap_seed(
    base_seed: int,
    *,
    stride: int,
    scope: str,
    baseline: str,
) -> int:
    """Derive an order-independent NumPy seed for one reported interval."""

    payload = f"d0-multiscale:{int(base_seed)}:{int(stride)}:{scope}:{baseline}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def bootstrap_whole_path_gain(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_error_key: str,
    reps: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a gain ratio by resampling complete path clusters.

    Multiple rows for one path (for example, one per training seed) remain in
    the same cluster.  Input order therefore cannot change the interval.
    """

    if int(reps) <= 0:
        raise ValueError("bootstrap reps must be positive")
    if not math.isfinite(float(confidence)) or not 0.5 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must be in (0.5, 1)")
    grouped: dict[int, tuple[float, float]] = {}
    for row in rows:
        path_id = int(row["path_id"])
        model_error = float(row["squared_error_sum"])
        baseline_error = float(row[baseline_error_key])
        if (
            path_id < 0
            or not math.isfinite(model_error)
            or not math.isfinite(baseline_error)
            or model_error < 0.0
            or baseline_error < 0.0
        ):
            raise ValueError("bootstrap rows must contain finite non-negative errors")
        old_model, old_baseline = grouped.get(path_id, (0.0, 0.0))
        grouped[path_id] = (
            old_model + model_error,
            old_baseline + baseline_error,
        )
    if not grouped:
        raise ValueError("bootstrap requires at least one whole path")
    ordered = sorted(grouped)
    model = np.asarray([grouped[path][0] for path in ordered], dtype=np.float64)
    baseline = np.asarray([grouped[path][1] for path in ordered], dtype=np.float64)
    baseline_total = float(baseline.sum())
    if baseline_total <= 0.0:
        raise ValueError("bootstrap baseline error must be positive")
    point = 1.0 - float(model.sum()) / baseline_total
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(reps), dtype=np.float64)
    path_count = len(ordered)
    for index in range(int(reps)):
        selected = rng.integers(0, path_count, size=path_count)
        denominator = float(baseline[selected].sum())
        draws[index] = (
            1.0 - float(model[selected].sum()) / denominator
            if denominator > 0.0
            else float("nan")
        )
    if not np.isfinite(draws).all():
        raise ValueError("bootstrap encountered a zero-error resample")
    alpha = 1.0 - float(confidence)
    return {
        "point_gain": float(point),
        "lower_bound": float(np.quantile(draws, alpha)),
        "upper_bound": float(np.quantile(draws, float(confidence))),
        "bootstrap_mean": float(draws.mean()),
        "confidence": float(confidence),
        "reps": int(reps),
        "seed": int(seed),
        "path_count": int(path_count),
        "cluster_unit": "whole_path_id",
        "baseline_error_key": str(baseline_error_key),
    }


def _check(
    name: str,
    value: Any,
    operator: str,
    threshold: Any,
    passed: bool,
) -> tuple[str, dict[str, Any]]:
    return name, {
        "passed": int(bool(passed)),
        "value": value,
        "operator": operator,
        "threshold": threshold,
    }


def _finish_gate(
    name: str,
    checks: Sequence[tuple[str, dict[str, Any]]],
    claim_scope: str,
) -> dict[str, Any]:
    subchecks = dict(checks)
    passed = bool(subchecks) and all(
        bool(int(check.get("passed", 0))) for check in subchecks.values()
    )
    return {
        "gate": name,
        "passed": int(passed),
        f"{name}_pass": int(passed),
        "subchecks": subchecks,
        "claim_scope": claim_scope,
    }


def _one_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    bin_index: int,
) -> Mapping[str, Any]:
    selected = [
        row
        for row in rows
        if str(row.get("split")) == split
        and int(row.get("bin_index", -999)) == int(bin_index)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one {split} metric row for bin_index={bin_index}, got {len(selected)}"
        )
    return selected[0]


def _finite_unit_fraction(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric == 1.0


def evaluate_stride_pass(
    stride: int,
    seed_results: Sequence[Mapping[str, Any]],
    thresholds: MultiscaleGateThresholds = MultiscaleGateThresholds(),
) -> dict[str, Any]:
    """Evaluate robust held-out signal for one temporal block stride."""

    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    results = [dict(result) for result in seed_results]
    strict_audit_coverage = (
        int(thresholds.expected_audit_paths)
        * int(thresholds.expected_data_end_slices_per_path)
        == int(thresholds.min_data_end_slices)
    )
    seed_summaries: list[dict[str, Any]] = []
    audit_path_rows: list[dict[str, Any]] = []
    parsing_ok = True
    for result in results:
        try:
            if int(result.get("stride", -1)) != int(stride):
                raise ValueError("seed result stride mismatch")
            seed = int(result["training_seed"])
            selected_step = int(result.get("selected_step", -1))
            complete = int(result.get("task_complete", 0)) == 1
            split_rows = list(result.get("split_metrics", []))
            time_rows = list(result.get("time_bin_metrics", []))
            audit = _one_row(split_rows, split="audit", bin_index=-1)
            train = _one_row(split_rows, split="train", bin_index=-1)
            selection = _one_row(split_rows, split="selection", bin_index=-1)
            data_end = _one_row(
                time_rows,
                split="audit",
                bin_index=int(thresholds.data_end_bin_index),
            )
            selection_data_end = _one_row(
                time_rows,
                split="selection",
                bin_index=int(thresholds.data_end_bin_index),
            )
            overall_gain = float(audit.get("prediction_gain", float("nan")))
            data_end_gain = float(data_end.get("prediction_gain", float("nan")))
            selection_overall_gain = float(
                selection.get("prediction_gain", float("nan"))
            )
            selection_data_end_gain = float(
                selection_data_end.get("prediction_gain", float("nan"))
            )
            tau_gain = float(
                audit.get("prediction_gain_vs_tau_baseline", float("nan"))
            )
            audit_covariance = float(
                audit.get("target_prediction_covariance", float("nan"))
            )
            train_gain = float(train.get("prediction_gain", float("nan")))
            count = int(data_end.get("slice_count", 0))
            declared_audit_paths = {
                int(value)
                for value in dict(result.get("split_path_ids", {})).get("audit", [])
            }
            audit_data_end_rows = [
                row
                for row in result.get("per_path_metrics", [])
                if str(row.get("split")) == "audit"
                and int(row.get("bin_index", -999))
                == int(thresholds.data_end_bin_index)
            ]
            observed_audit_paths = {
                int(row.get("path_id", -1)) for row in audit_data_end_rows
            }
            audit_coverage = (
                (
                    len(declared_audit_paths) == int(thresholds.expected_audit_paths)
                    and observed_audit_paths == declared_audit_paths
                    and len(audit_data_end_rows) == int(thresholds.expected_audit_paths)
                    and all(
                        int(row.get("slice_count", -1))
                        == int(thresholds.expected_data_end_slices_per_path)
                        for row in audit_data_end_rows
                    )
                    and count == int(thresholds.min_data_end_slices)
                )
                if strict_audit_coverage
                else count >= int(thresholds.min_data_end_slices)
            )
            finite = (
                _finite_unit_fraction(audit.get("finite_fraction"))
                and _finite_unit_fraction(data_end.get("finite_fraction"))
                and _finite_unit_fraction(selection.get("finite_fraction"))
                and _finite_unit_fraction(selection_data_end.get("finite_fraction"))
            )
            seed_signal = (
                complete
                and selected_step > 0
                and finite
                and audit_coverage
                and math.isfinite(selection_overall_gain)
                and selection_overall_gain > float(thresholds.min_prediction_gain)
                and math.isfinite(selection_data_end_gain)
                and selection_data_end_gain > float(thresholds.min_prediction_gain)
                and math.isfinite(overall_gain)
                and overall_gain > float(thresholds.min_prediction_gain)
                and math.isfinite(data_end_gain)
                and data_end_gain > float(thresholds.min_prediction_gain)
            )
            seed_summaries.append(
                {
                    "training_seed": seed,
                    "complete": int(complete),
                    "selected_step": selected_step,
                    "selection_overall_gain": selection_overall_gain,
                    "selection_data_end_gain": selection_data_end_gain,
                    "audit_overall_gain": overall_gain,
                    "audit_data_end_gain": data_end_gain,
                    "audit_gain_vs_tau_baseline": tau_gain,
                    "audit_target_prediction_covariance": audit_covariance,
                    "train_overall_gain": train_gain,
                    "data_end_slice_count": count,
                    "audit_path_count": len(declared_audit_paths),
                    "audit_coverage_pass": int(audit_coverage),
                    "finite": int(finite),
                    "seed_signal_pass": int(seed_signal),
                }
            )
            audit_path_rows.extend(
                dict(row)
                for row in result.get("per_path_metrics", [])
                if str(row.get("split")) == "audit"
                and int(row.get("bin_index", -999))
                in {-1, int(thresholds.data_end_bin_index)}
            )
        except (KeyError, TypeError, ValueError):
            parsing_ok = False

    seeds = [int(row["training_seed"]) for row in seed_summaries]
    unique_seeds = len(set(seeds)) == len(seeds)
    complete_tasks = bool(seed_summaries) and all(
        bool(row["complete"]) for row in seed_summaries
    )
    positive_steps = bool(seed_summaries) and all(
        int(row["selected_step"]) > 0 for row in seed_summaries
    )
    finite_tasks = bool(seed_summaries) and all(
        bool(row["finite"])
        and all(
            math.isfinite(float(row[key]))
            for key in (
                "audit_overall_gain",
                "audit_data_end_gain",
                "audit_gain_vs_tau_baseline",
                "selection_overall_gain",
                "selection_data_end_gain",
                "audit_target_prediction_covariance",
                "train_overall_gain",
            )
        )
        for row in seed_summaries
    )
    counts_ok = bool(seed_summaries) and all(
        (
            int(row["data_end_slice_count"]) == int(thresholds.min_data_end_slices)
            and int(row["audit_path_count"]) == int(thresholds.expected_audit_paths)
            and bool(int(row["audit_coverage_pass"]))
            if strict_audit_coverage
            else int(row["data_end_slice_count"])
            >= int(thresholds.min_data_end_slices)
        )
        for row in seed_summaries
    )
    passing_seeds = sum(int(row["seed_signal_pass"]) for row in seed_summaries)

    def median(key: str) -> float:
        values = [float(row[key]) for row in seed_summaries]
        return float(np.median(values)) if values and np.isfinite(values).all() else float("nan")

    median_overall = median("audit_overall_gain")
    median_data_end = median("audit_data_end_gain")
    median_tau = median("audit_gain_vs_tau_baseline")
    median_selection_overall = median("selection_overall_gain")
    median_selection_data_end = median("selection_data_end_gain")
    median_covariance = median("audit_target_prediction_covariance")
    median_train = median("train_overall_gain")
    finite_train_gains = [
        float(row["train_overall_gain"])
        for row in seed_summaries
        if math.isfinite(float(row["train_overall_gain"]))
    ]
    max_train = max(finite_train_gains, default=float("nan"))

    def bootstrap(scope_bin: int, baseline_key: str, label: str) -> dict[str, Any]:
        selected = [
            row
            for row in audit_path_rows
            if int(row.get("bin_index", -999)) == int(scope_bin)
        ]
        return bootstrap_whole_path_gain(
            selected,
            baseline_error_key=baseline_key,
            reps=int(thresholds.bootstrap_reps),
            confidence=float(thresholds.bootstrap_confidence),
            seed=derive_bootstrap_seed(
                int(thresholds.bootstrap_seed),
                stride=int(stride),
                scope="overall" if scope_bin == -1 else "data_end",
                baseline=label,
            ),
        )

    try:
        bootstrap_overall = bootstrap(-1, "zero_squared_error_sum", "zero")
        bootstrap_data_end = bootstrap(
            int(thresholds.data_end_bin_index),
            "zero_squared_error_sum",
            "zero",
        )
        bootstrap_ok = True
    except (KeyError, TypeError, ValueError):
        bootstrap_overall = {}
        bootstrap_data_end = {}
        bootstrap_ok = False

    # The train-only tau mean is a required point-estimate comparator, but its
    # bootstrap interval is advisory.  In particular, a perfect/zero-error tau
    # baseline has no defined gain-ratio bootstrap and must not erase otherwise
    # valid zero-baseline intervals.
    try:
        bootstrap_tau = {
            **bootstrap(
                -1,
                "tau_baseline_squared_error_sum",
                "tau_mean",
            ),
            "status": "available",
            "gated": 0,
        }
    except (KeyError, TypeError, ValueError) as exc:
        bootstrap_tau = {
            "status": "unavailable",
            "gated": 0,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    overall_lcb = float(bootstrap_overall.get("lower_bound", float("nan")))
    data_end_lcb = float(bootstrap_data_end.get("lower_bound", float("nan")))
    expected_count = int(thresholds.expected_training_seeds)
    task_set_complete = (
        parsing_ok
        and len(seed_summaries) == expected_count
        and unique_seeds
        and complete_tasks
        and positive_steps
        and finite_tasks
        and counts_ok
    )
    checks = [
        _check(
            "seed_result_count",
            len(seed_summaries),
            "==",
            expected_count,
            parsing_ok and len(seed_summaries) == expected_count and unique_seeds,
        ),
        _check("all_tasks_complete", int(complete_tasks), "==", 1, complete_tasks),
        _check("selected_steps_positive", int(positive_steps), "==", 1, positive_steps),
        _check("all_metrics_finite", int(finite_tasks), "==", 1, finite_tasks),
        _check(
            "data_end_count",
            [int(row["data_end_slice_count"]) for row in seed_summaries],
            "== each" if strict_audit_coverage else ">= each",
            int(thresholds.min_data_end_slices),
            counts_ok,
        ),
        _check(
            "audit_path_coverage",
            [int(row["audit_path_count"]) for row in seed_summaries],
            "== each" if strict_audit_coverage else "diagnostic",
            int(thresholds.expected_audit_paths) if strict_audit_coverage else "not frozen",
            counts_ok,
        ),
        _check(
            "passing_seed_count",
            passing_seeds,
            ">=",
            int(thresholds.min_passing_seeds),
            passing_seeds >= int(thresholds.min_passing_seeds),
        ),
        _check(
            "median_selection_overall_gain",
            median_selection_overall,
            ">",
            float(thresholds.min_prediction_gain),
            math.isfinite(median_selection_overall)
            and median_selection_overall > float(thresholds.min_prediction_gain),
        ),
        _check(
            "median_selection_data_end_gain",
            median_selection_data_end,
            ">",
            float(thresholds.min_prediction_gain),
            math.isfinite(median_selection_data_end)
            and median_selection_data_end > float(thresholds.min_prediction_gain),
        ),
        _check(
            "median_overall_gain",
            median_overall,
            ">",
            float(thresholds.min_prediction_gain),
            math.isfinite(median_overall)
            and median_overall > float(thresholds.min_prediction_gain),
        ),
        _check(
            "median_data_end_gain",
            median_data_end,
            ">",
            float(thresholds.min_prediction_gain),
            math.isfinite(median_data_end)
            and median_data_end > float(thresholds.min_prediction_gain),
        ),
        _check(
            "median_tau_baseline_gain",
            median_tau,
            ">",
            float(thresholds.min_tau_baseline_gain),
            math.isfinite(median_tau)
                and median_tau > float(thresholds.min_tau_baseline_gain),
        ),
        _check(
            "median_audit_target_prediction_covariance",
            median_covariance,
            ">",
            0.0,
            math.isfinite(median_covariance) and median_covariance > 0.0,
        ),
        _check(
            "overall_path_bootstrap_lcb",
            overall_lcb,
            ">",
            float(thresholds.min_prediction_gain),
            bootstrap_ok
            and math.isfinite(overall_lcb)
            and overall_lcb > float(thresholds.min_prediction_gain),
        ),
        _check(
            "data_end_path_bootstrap_lcb",
            data_end_lcb,
            ">",
            float(thresholds.min_prediction_gain),
            bootstrap_ok
            and math.isfinite(data_end_lcb)
            and data_end_lcb > float(thresholds.min_prediction_gain),
        ),
    ]
    gate = _finish_gate(
        "stride_learnability",
        checks,
        "held-out conditional-mean signal at one temporal block stride",
    )
    gate.update(
        {
            "stride": int(stride),
            "seed_summaries": seed_summaries,
            "bootstrap": {
                "overall_vs_zero": bootstrap_overall,
                "data_end_vs_zero": bootstrap_data_end,
                "overall_vs_tau_mean": bootstrap_tau,
            },
            "diagnostics": {
                "median_audit_overall_gain": median_overall,
                "median_audit_data_end_gain": median_data_end,
                "median_audit_gain_vs_tau_baseline": median_tau,
                "median_selection_overall_gain": median_selection_overall,
                "median_selection_data_end_gain": median_selection_data_end,
                "median_audit_target_prediction_covariance": median_covariance,
                "median_train_overall_gain": median_train,
                "max_train_overall_gain": max_train,
                "audit_point_signal": int(
                    task_set_complete
                    and any(
                        (
                            math.isfinite(float(row["audit_overall_gain"]))
                            and float(row["audit_overall_gain"])
                            > float(thresholds.min_prediction_gain)
                        )
                        or (
                            math.isfinite(float(row["audit_data_end_gain"]))
                            and float(row["audit_data_end_gain"])
                            > float(thresholds.min_prediction_gain)
                        )
                        for row in seed_summaries
                    )
                ),
                "train_point_signal": int(
                    task_set_complete
                    and (
                        math.isfinite(median_train)
                        and median_train
                        >= float(thresholds.memorization_train_gain)
                    )
                    or (
                        math.isfinite(max_train)
                        and max_train
                        >= float(thresholds.memorization_train_gain)
                    )
                ),
                "all_tasks_complete": int(task_set_complete),
            },
        }
    )
    return gate


def evaluate_teacher_control(
    metrics: Mapping[str, Any],
    thresholds: MultiscaleGateThresholds = MultiscaleGateThresholds(),
) -> dict[str, Any]:
    """Fail-closed control proving that the probe/optimizer can learn a teacher."""

    complete = int(metrics.get("complete", 0))
    selected_step = int(metrics.get("selected_step", -1))
    finite_fraction = float(metrics.get("finite_fraction", float("nan")))
    overall_gain = float(metrics.get("audit_overall_gain", float("nan")))
    data_end_gain = float(metrics.get("audit_data_end_gain", float("nan")))
    data_end_count = int(metrics.get("audit_data_end_slice_count", 0))
    minimum = float(thresholds.teacher_min_gain)
    checks = [
        _check("complete", complete, "==", 1, complete == 1),
        _check("selected_step", selected_step, ">", 0, selected_step > 0),
        _check(
            "finite_fraction",
            finite_fraction,
            "==",
            1.0,
            math.isfinite(finite_fraction) and finite_fraction == 1.0,
        ),
        _check(
            "overall_gain",
            overall_gain,
            ">=",
            minimum,
            math.isfinite(overall_gain) and minimum <= overall_gain <= 1.0,
        ),
        _check(
            "data_end_count",
            data_end_count,
            ">=",
            int(thresholds.min_data_end_slices),
            data_end_count >= int(thresholds.min_data_end_slices),
        ),
        _check(
            "data_end_gain",
            data_end_gain,
            ">=",
            minimum,
            data_end_count >= int(thresholds.min_data_end_slices)
            and math.isfinite(data_end_gain)
            and minimum <= data_end_gain <= 1.0,
        ),
    ]
    return _finish_gate(
        "teacher",
        checks,
        "synthetic deterministic teacher learnability control",
    )


def _passed(gate: Mapping[str, Any] | bool) -> bool:
    if isinstance(gate, bool):
        return gate
    return bool(int(gate.get("passed", gate.get("required_gate_pass", 0))))


def decide_learnability(
    *,
    cache_gate: Mapping[str, Any] | bool,
    teacher_gate: Mapping[str, Any] | bool,
    stride_gates: Mapping[int, Mapping[str, Any]],
    elementary_stride: int = 1,
    memorization_train_gain: float = 0.50,
    study_profile: str = "pilot",
    profile_conformant: bool | int = True,
    authoritative_decision: bool | int = False,
) -> dict[str, Any]:
    """Classify an optimization-only study and prescribe its valid next step.

    The default ``pilot`` profile preserves the original state machine.  A
    conformant ``confirmation`` profile is deliberately terminal when no
    stride passes: weak positive point estimates do not trigger an unbounded
    sequence of larger reruns.  Confirmation still reports path memorization
    explicitly when that stronger diagnostic is present.
    """

    profile = _normalize_study_profile(study_profile)
    conformant = _binary_flag(profile_conformant, "profile_conformant")
    authoritative = _binary_flag(authoritative_decision, "authoritative_decision")
    if authoritative and not conformant:
        raise ValueError("an authoritative decision must be profile conformant")
    gates = {int(stride): dict(gate) for stride, gate in stride_gates.items()}
    cache_pass = _passed(cache_gate)
    teacher_pass = _passed(teacher_gate)
    elementary_pass = _passed(gates.get(int(elementary_stride), {}))
    coarse_passes = sorted(
        stride
        for stride, gate in gates.items()
        if stride != int(elementary_stride) and _passed(gate)
    )
    diagnostics = [
        dict(gate.get("diagnostics", {})) for gate in gates.values()
    ]
    all_tasks_complete = bool(diagnostics) and all(
        bool(int(item.get("all_tasks_complete", 0))) for item in diagnostics
    )
    any_audit_point_signal = any(
        bool(int(item.get("audit_point_signal", 0))) for item in diagnostics
    )
    if (
        not math.isfinite(float(memorization_train_gain))
        or not 0.0 <= float(memorization_train_gain) <= 1.0
    ):
        raise ValueError("memorization_train_gain must be finite and in [0, 1]")

    def reaches_memorization_threshold(item: Mapping[str, Any]) -> bool:
        values = (
            item.get("median_train_overall_gain", float("nan")),
            item.get("max_train_overall_gain", float("nan")),
        )
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric) and numeric >= float(memorization_train_gain):
                return True
        return False

    any_train_memorization_signal = any(
        reaches_memorization_threshold(item) for item in diagnostics
    )

    if not cache_pass:
        decision = LearnabilityDecision.CACHE_INVALID
        action = "fix the multiscale cache accumulator or numerical-health failure"
    elif not teacher_pass:
        decision = LearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID
        action = "fix the probe model, objective, or optimizer before interpreting D0 targets"
    elif not all_tasks_complete:
        decision = LearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID
        action = "repair or exactly resume every incomplete stride/seed task before interpretation"
    elif elementary_pass:
        decision = LearnabilityDecision.ELEMENTARY_SIGNAL
        action = "launch a fresh strict r=1 reconstruction run"
    elif coarse_passes:
        decision = LearnabilityDecision.COARSE_ONLY_SIGNAL
        action = (
            "plan a separately named coarse sampler plus conditional-noise calibration; "
            "this workflow performs no sampling"
        )
    elif profile == "confirmation" and conformant:
        if any_train_memorization_signal:
            decision = LearnabilityDecision.PATH_MEMORIZATION_ONLY
            action = (
                "investigate a separately validated variance-reduction/data-generation "
                "redesign; do not repeat this profile or sample"
            )
        else:
            decision = LearnabilityDecision.NO_CONFIRMED_CONDITIONAL_SIGNAL
            action = (
                "revisit cumulative/score targets or introduce a separately validated "
                "variance-reduction method; do not repeat this profile or sample"
            )
    elif any_audit_point_signal:
        decision = LearnabilityDecision.INCONCLUSIVE
        action = "rerun unchanged with 128 independent paths and five training seeds"
    elif any_train_memorization_signal:
        decision = LearnabilityDecision.PATH_MEMORIZATION_ONLY
        action = "increase independent forward paths or variance reduction; do not sample"
    else:
        decision = LearnabilityDecision.NO_DETECTABLE_CONDITIONAL_SIGNAL
        action = "revisit the cumulative/score target theory before model scaling or sampling"

    terminal_confirmation = (
        profile == "confirmation"
        and conformant
        and cache_pass
        and teacher_pass
        and all_tasks_complete
        and not elementary_pass
        and not coarse_passes
    )
    return {
        "decision": decision.value,
        "recommended_next_action": action,
        "study_profile": profile,
        "profile_conformant": int(conformant),
        "authoritative_decision": int(authoritative),
        "confirmation_profile": int(profile == "confirmation"),
        "confirmation_exhausted": int(terminal_confirmation),
        "repeat_same_profile_authorized": 0,
        "cache_pass": int(cache_pass),
        "teacher_pass": int(teacher_pass),
        "elementary_stride": int(elementary_stride),
        "elementary_signal": int(elementary_pass),
        "passing_coarse_strides": coarse_passes,
        "any_scale_signal": int(elementary_pass or bool(coarse_passes)),
        "all_tasks_complete": int(all_tasks_complete),
        "memorization_train_gain_threshold": float(memorization_train_gain),
        "sampling_performed": 0,
        "sampling_authorized": 0,
        "claim_scope": CLAIM_SCOPE,
        "excluded_claims": [
            "reverse reconstruction",
            "sampler validity",
            "direct h-transform semantics for strides greater than one",
            "spatial Dirichlet-Ferguson convergence",
            "held-out digit generalization",
        ],
    }


def evaluate_multiscale_gates(
    *,
    cache_gate: Mapping[str, Any] | bool,
    teacher_gate: Mapping[str, Any] | bool,
    stride_gates: Mapping[int, Mapping[str, Any]],
    require_gate: str = "none",
    thresholds: MultiscaleGateThresholds = MultiscaleGateThresholds(),
    study_profile: str = "pilot",
    profile_conformant: bool | int = True,
) -> dict[str, Any]:
    """Combine cache, control, and stride evidence without authorizing sampling."""

    required = str(require_gate).strip().lower()
    if required not in {"none", "cache", "teacher", "any-scale", "elementary"}:
        raise ValueError(
            "require_gate must be none, cache, teacher, any-scale, or elementary"
        )
    profile = _normalize_study_profile(study_profile)
    conformant = _binary_flag(profile_conformant, "profile_conformant")
    authoritative = required != "none" and conformant
    cache_pass = _passed(cache_gate)
    teacher_pass = cache_pass and _passed(teacher_gate)
    stride_passes = {
        int(stride): teacher_pass and _passed(gate)
        for stride, gate in stride_gates.items()
    }
    any_scale = any(stride_passes.values())
    elementary = stride_passes.get(int(thresholds.elementary_stride), False)
    cumulative = {
        "cache": cache_pass,
        "teacher": teacher_pass,
        "any-scale": any_scale,
        "elementary": elementary,
    }
    decision = decide_learnability(
        cache_gate=cache_gate,
        teacher_gate=teacher_gate,
        stride_gates=stride_gates,
        elementary_stride=int(thresholds.elementary_stride),
        memorization_train_gain=float(thresholds.memorization_train_gain),
        study_profile=profile,
        profile_conformant=conformant,
        authoritative_decision=authoritative,
    )
    return {
        "schema": "experiment12-d0-multiscale-learnability-gate",
        "schema_version": 2,
        "study_profile": profile,
        "profile_conformant": int(conformant),
        "authoritative_decision": int(authoritative),
        "confirmation_profile": int(profile == "confirmation"),
        "confirmation_exhausted": int(decision["confirmation_exhausted"]),
        "repeat_same_profile_authorized": int(
            decision["repeat_same_profile_authorized"]
        ),
        "required_gate": required,
        "required_gate_pass": int(
            True if required == "none" else conformant and cumulative[required]
        ),
        "cumulative_pass": {key: int(value) for key, value in cumulative.items()},
        "cache": dict(cache_gate) if isinstance(cache_gate, Mapping) else {"passed": int(cache_gate)},
        "teacher": dict(teacher_gate) if isinstance(teacher_gate, Mapping) else {"passed": int(teacher_gate)},
        "strides": {str(stride): dict(gate) for stride, gate in sorted(stride_gates.items())},
        "decision": decision,
        "thresholds": asdict(thresholds),
        "sampling_performed": 0,
        "sampling_authorized": 0,
        "claim_scope": CLAIM_SCOPE,
    }


def write_multiscale_gate_artifacts(
    output_dir: str | Path,
    seed_results: Sequence[Mapping[str, Any]],
    gate_report: Mapping[str, Any],
) -> dict[str, Path]:
    """Atomically write flattened metric tables and the final decision report."""

    destination = Path(output_dir)
    split_rows = [
        dict(row)
        for result in seed_results
        for row in result.get("split_metrics", [])
    ]
    time_rows = [
        dict(row)
        for result in seed_results
        for row in result.get("time_bin_metrics", [])
    ]
    path_rows = [
        dict(row)
        for result in seed_results
        for row in result.get("per_path_metrics", [])
    ]
    paths = {
        "split_metrics": destination / "learnability_split_metrics.csv",
        "time_bins": destination / "learnability_time_bins.csv",
        "per_path": destination / "learnability_per_path.csv",
        "decision": destination / "learnability_decision.json",
    }
    atomic_write_csv(paths["split_metrics"], split_rows)
    atomic_write_csv(paths["time_bins"], time_rows)
    atomic_write_csv(paths["per_path"], path_rows)
    atomic_write_json(paths["decision"], gate_report)
    return paths


__all__ = [
    "CLAIM_SCOPE",
    "LearnabilityDecision",
    "MultiscaleGateThresholds",
    "SPLIT_NAMES",
    "TIME_BIN_EDGES",
    "TauBinBaseline",
    "bootstrap_whole_path_gain",
    "compute_multiscale_split_metrics",
    "decide_learnability",
    "derive_bootstrap_seed",
    "evaluate_multiscale_gates",
    "evaluate_stride_pass",
    "evaluate_teacher_control",
    "fit_tau_bin_baseline",
    "predict_tau_bin_baseline",
    "write_multiscale_gate_artifacts",
]
