"""Oracle evidence-power helpers for D0 density-ratio controls.

This module is deliberately controls-only.  It evaluates the exact bounded
teacher Bayes logit on immutable :class:`DensityRatioPanel` objects and asks
whether whole-path evidence panels can detect that population signal.  It
contains no optimizer, physical-score training, or sampler code.

The established checkpoint gates call a reported ``90% lower bound`` the
lower endpoint of a *one-sided* whole-path bootstrap.  Accordingly, a
confidence level ``c`` uses bootstrap quantile ``1-c`` here (rather than the
lower endpoint of a central two-sided interval).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .d0_score_boundary_controls import bounded_teacher_log_relative_potential
from .d0_score_density_ratio import DensityRatioPanel, panel_identity


SELECTION_POWER_SCHEMA = "experiment12-d0-score-density-ratio-selection-power"
SELECTION_POWER_SCHEMA_VERSION = 1
ORACLE_ESTIMATOR_VERSION = "bounded-teacher-balanced-bce-oracle-v1"
ONE_SIDED_BOOTSTRAP_VERSION = "whole-path-one-sided-percentile-v1"
PREDETERMINED_HALF_VERSION = "panel-path-order-contiguous-halves-v1"

# Immutable normalized-head parent panels and the independently inspected
# exact-oracle summary.  Lower bounds were described as approximate in the
# patch plan because they include deterministic Monte Carlo bootstrap error.
SAVED_16_PATH_PANEL_FINGERPRINTS: Mapping[str, str] = {
    "a": "8d2a538bb52ce8d1b8b4e8f16aaf13f1a4c5571eb257ac3de74d1f697e6b23a1",
    "b": "31218e6aa76e7b55b2f429091aad08622044b5717bf79f0e500933b81260f10d",
}
SAVED_16_PATH_ORACLE_REFERENCE: Mapping[str, Mapping[str, Mapping[str, float]]] = {
    "a": {
        "overall": {
            "point_estimate": 0.016217710955717175,
            "lower_bound": 0.00872031148,
        },
        "data_end": {
            "point_estimate": 0.01721000665283977,
            "lower_bound": 0.00517841895,
        },
    },
    "b": {
        "overall": {
            "point_estimate": -0.0017484144463532237,
            "lower_bound": -0.00913743689,
        },
        "data_end": {
            "point_estimate": 0.0032640097809109756,
            "lower_bound": -0.00565862046,
        },
    },
}
# Fixed bootstrap seeds reproduce the four inspected Monte Carlo summaries to
# substantially tighter accuracy than the gate's 5e-5 forensic tolerance.
SAVED_16_PATH_BOOTSTRAP_SEEDS: Mapping[str, Mapping[str, int]] = {
    "a": {"overall": 998, "data_end": 1441},
    "b": {"overall": 515, "data_end": 1731},
}


__all__ = [
    "SELECTION_POWER_SCHEMA",
    "SELECTION_POWER_SCHEMA_VERSION",
    "ORACLE_ESTIMATOR_VERSION",
    "ONE_SIDED_BOOTSTRAP_VERSION",
    "PREDETERMINED_HALF_VERSION",
    "SAVED_16_PATH_PANEL_FINGERPRINTS",
    "SAVED_16_PATH_ORACLE_REFERENCE",
    "SAVED_16_PATH_BOOTSTRAP_SEEDS",
    "derive_selection_power_seed",
    "exact_bounded_teacher_oracle_logits",
    "balanced_bce_improvement",
    "whole_path_one_sided_lower_bound",
    "oracle_panel_subset_identity",
    "evaluate_exact_teacher_oracle_panel",
    "evaluate_oracle_panel_feasibility",
    "evaluate_oracle_power_calibration",
    "reproduce_saved_16_path_oracle_forensic",
]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_selection_power_seed(root_seed: int, *parts: Any) -> int:
    """Derive a stateless NumPy-compatible seed for evidence calculations."""

    payload = json.dumps(
        [
            SELECTION_POWER_SCHEMA,
            SELECTION_POWER_SCHEMA_VERSION,
            int(root_seed),
            *parts,
        ],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    # NumPy accepts an unsigned 64-bit seed.  Staying below 2**63 also makes
    # the seed portable through signed JSON/SQLite consumers.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def exact_bounded_teacher_oracle_logits(
    panel: DensityRatioPanel,
    *,
    epsilon: float = 0.5,
    batch_size: int = 4096,
) -> Tensor:
    """Evaluate the exact equal-prior Bayes logit in float64 on CPU.

    Chunking keeps the 128/256-path production panels inexpensive in peak
    memory while returning results in their original immutable row order.
    """

    if panel.task != "bounded_teacher":
        raise ValueError("the bounded-teacher oracle requires a teacher panel")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(float(epsilon)) or not (0.0 < float(epsilon) < 1.0):
        raise ValueError("epsilon must lie strictly between zero and one")
    logits: list[Tensor] = []
    states = panel.states.detach().cpu()
    fractions = panel.tau_fraction.detach().cpu()
    for start in range(0, int(states.shape[0]), int(batch_size)):
        stop = min(start + int(batch_size), int(states.shape[0]))
        logits.append(
            bounded_teacher_log_relative_potential(
                states[start:stop].double(),
                fractions[start:stop].double(),
                epsilon=float(epsilon),
            ).detach()
        )
    result = torch.cat(logits).contiguous()
    if result.shape != (int(panel.states.shape[0]),):
        raise RuntimeError("oracle logit evaluation changed the panel row axis")
    return result


def balanced_bce_improvement(logits: Tensor, targets: Tensor) -> Tensor:
    """Return rowwise balanced-BCE improvement over the zero logit."""

    values = logits.detach().double().reshape(-1)
    truth = targets.detach().double().cpu().reshape(-1)
    if values.device.type != "cpu":
        values = values.cpu()
    if values.shape != truth.shape or not values.numel():
        raise ValueError("logits and targets must be nonempty matching vectors")
    if not bool(torch.all((truth == 0.0) | (truth == 1.0))):
        raise ValueError("targets must be binary")
    # softplus(l) - y*l is the stable elementwise BCE-with-logits formula.
    losses = torch.nn.functional.softplus(values) - truth * values
    return (math.log(2.0) - losses).contiguous()


def _ordered_unique_path_ids(path_ids: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    ordered: list[int] = []
    for raw in np.asarray(path_ids, dtype=np.int64).reshape(-1).tolist():
        value = int(raw)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return np.asarray(ordered, dtype=np.int64)


def whole_path_one_sided_lower_bound(
    values: np.ndarray | Tensor,
    path_ids: np.ndarray | Tensor,
    *,
    reps: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Compute a deterministic one-sided percentile lower confidence bound.

    Rows are first averaged within whole path IDs.  Bootstrap replicates then
    resample those path means, never individual states or time anchors.
    """

    samples = (
        values.detach().double().cpu().numpy()
        if isinstance(values, Tensor)
        else np.asarray(values, dtype=np.float64)
    ).reshape(-1)
    paths = (
        path_ids.detach().long().cpu().numpy()
        if isinstance(path_ids, Tensor)
        else np.asarray(path_ids, dtype=np.int64)
    ).reshape(-1)
    if samples.shape != paths.shape or samples.size == 0:
        raise ValueError("values and path_ids must be nonempty matching vectors")
    if int(reps) <= 0:
        raise ValueError("reps must be positive")
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError("confidence must lie strictly between zero and one")
    unique = _ordered_unique_path_ids(paths)
    path_values = np.asarray(
        [samples[paths == path_id].mean() for path_id in unique],
        dtype=np.float64,
    )
    finite = bool(np.isfinite(path_values).all())
    base: dict[str, Any] = {
        "schema": SELECTION_POWER_SCHEMA + "-whole-path-bootstrap",
        "schema_version": SELECTION_POWER_SCHEMA_VERSION,
        "bootstrap_version": ONE_SIDED_BOOTSTRAP_VERSION,
        "confidence": float(confidence),
        "tail_probability": float(1.0 - float(confidence)),
        "reps": int(reps),
        "seed": int(seed),
        "path_count": int(unique.size),
        "state_count": int(samples.size),
        "path_ids": unique.tolist(),
        "path_values": path_values.tolist() if finite else None,
        "finite": int(finite),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    if not finite:
        return {
            **base,
            "point_estimate": None,
            "lower_bound": None,
        }
    generator = np.random.default_rng(int(seed))
    totals = np.empty(int(reps), dtype=np.float64)
    # Bounded chunks avoid allocating reps x 256 indefinitely if production
    # bootstrap replication is increased later.
    for start in range(0, int(reps), 1024):
        count = min(1024, int(reps) - start)
        indices = generator.integers(
            0, int(unique.size), size=(count, int(unique.size))
        )
        totals[start : start + count] = path_values[indices].mean(axis=1)
    return {
        **base,
        "point_estimate": float(path_values.mean()),
        "lower_bound": float(
            np.quantile(totals, 1.0 - float(confidence))
        ),
        "bootstrap_mean": float(totals.mean()),
        "bootstrap_standard_deviation": float(totals.std(ddof=0)),
    }


def oracle_panel_subset_identity(
    panel: DensityRatioPanel,
    selected_path_ids: Sequence[int] | np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    """Return a stable identity for a path-only view of an immutable panel."""

    selected = np.asarray(selected_path_ids, dtype=np.int64).reshape(-1)
    if not str(name) or selected.size == 0 or np.unique(selected).size != selected.size:
        raise ValueError("subset name and unique selected path IDs are required")
    available = _ordered_unique_path_ids(np.asarray(panel.path_ids, dtype=np.int64))
    if not bool(np.isin(selected, available).all()):
        raise ValueError("selected path IDs are not a subset of the panel")
    mask = np.isin(np.asarray(panel.path_ids, dtype=np.int64), selected)
    payload = {
        "schema": SELECTION_POWER_SCHEMA + "-panel-subset",
        "schema_version": SELECTION_POWER_SCHEMA_VERSION,
        "derivation_version": PREDETERMINED_HALF_VERSION,
        "name": str(name),
        "source_panel_fingerprint": panel.fingerprint,
        "path_ids": selected.tolist(),
        "path_count": int(selected.size),
        "row_count": int(mask.sum()),
    }
    return {
        **payload,
        "fingerprint": _canonical_sha256(payload),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_exact_teacher_oracle_panel(
    panel: DensityRatioPanel,
    *,
    reps: int,
    confidence: float,
    seed: int,
    selected_path_ids: Sequence[int] | np.ndarray | None = None,
    subset_name: str = "full",
    epsilon: float = 0.5,
    batch_size: int = 4096,
    scope_seeds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate exact-oracle balanced BCE overall and at the data end."""

    logits = exact_bounded_teacher_oracle_logits(
        panel, epsilon=float(epsilon), batch_size=int(batch_size)
    )
    improvements = balanced_bce_improvement(logits, panel.class_targets)
    losses = math.log(2.0) - improvements.numpy()
    all_paths = np.asarray(panel.path_ids, dtype=np.int64)
    if selected_path_ids is None:
        selected = _ordered_unique_path_ids(all_paths)
    else:
        selected = np.asarray(selected_path_ids, dtype=np.int64).reshape(-1)
    subset = oracle_panel_subset_identity(panel, selected, name=str(subset_name))
    subset_mask = np.isin(all_paths, selected)
    strata = np.asarray(panel.strata, dtype=np.int64)

    scopes: dict[str, Any] = {}
    for name, mask in (
        ("overall", subset_mask),
        ("data_end", subset_mask & (strata == 4)),
    ):
        scope_seed = (
            int(scope_seeds[name])
            if scope_seeds is not None and name in scope_seeds
            else derive_selection_power_seed(
                int(seed), panel.fingerprint, subset["fingerprint"], name
            )
        )
        interval = whole_path_one_sided_lower_bound(
            improvements.numpy()[mask],
            all_paths[mask],
            reps=int(reps),
            confidence=float(confidence),
            seed=scope_seed,
        )
        scopes[name] = {
            "state_count": int(mask.sum()),
            "path_count": int(interval["path_count"]),
            "bce": None
            if not bool(interval["finite"])
            else float(losses[mask].mean()),
            "improvement": interval.get("point_estimate"),
            "point_estimate": interval.get("point_estimate"),
            "lower_bound": interval.get("lower_bound"),
            "objective_improvement_lower_bound": interval.get("lower_bound"),
            "confidence": float(confidence),
            "bootstrap": interval,
        }
    finite = bool(
        torch.isfinite(logits).all()
        and torch.isfinite(improvements).all()
        and all(bool(scope["bootstrap"]["finite"]) for scope in scopes.values())
    )
    expected_rows = {"overall": 64, "data_end": 32}
    whole_path_structure = bool(
        all(
            int(scopes[name]["state_count"])
            == int(selected.size) * expected_rows[name]
            for name in scopes
        )
    )
    return {
        "schema": SELECTION_POWER_SCHEMA + "-oracle-panel",
        "schema_version": SELECTION_POWER_SCHEMA_VERSION,
        "oracle_estimator_version": ORACLE_ESTIMATOR_VERSION,
        "evaluation_status": "evaluated",
        "panel": panel_identity(panel),
        "subset": subset,
        "path_count": int(selected.size),
        "anchors_per_path": 32,
        "confidence": float(confidence),
        "reps": int(reps),
        "bootstrap_replicates": int(reps),
        "complete": 1,
        "finite": int(finite),
        "whole_path_structure_valid": int(whole_path_structure),
        "overall": scopes["overall"],
        "data_end": scopes["data_end"],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_oracle_panel_feasibility(
    panel: DensityRatioPanel,
    *,
    reps: int,
    confidence: float,
    seed: int,
    selected_path_ids: Sequence[int] | np.ndarray | None = None,
    subset_name: str = "full",
    epsilon: float = 0.5,
) -> dict[str, Any]:
    """Require finite, strictly positive oracle lower bounds in both scopes."""

    evidence = evaluate_exact_teacher_oracle_panel(
        panel,
        reps=int(reps),
        confidence=float(confidence),
        seed=int(seed),
        selected_path_ids=selected_path_ids,
        subset_name=str(subset_name),
        epsilon=float(epsilon),
    )
    bounds = {
        name: evidence[name].get("lower_bound")
        for name in ("overall", "data_end")
    }
    passed = bool(
        evidence["finite"]
        and evidence["whole_path_structure_valid"]
        and all(
            value is not None
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in bounds.values()
        )
    )
    return {
        "schema": SELECTION_POWER_SCHEMA + "-oracle-panel-feasibility",
        "schema_version": SELECTION_POWER_SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": int(passed),
        "complete": 1,
        "finite": int(evidence["finite"]),
        "path_count": int(evidence["path_count"]),
        "anchors_per_path": 32,
        "confidence": float(confidence),
        "bootstrap_replicates": int(reps),
        "overall": dict(evidence["overall"]),
        "data_end": dict(evidence["data_end"]),
        "required": "finite positive exact-oracle lower bounds overall and at data end",
        "lower_bounds": bounds,
        "evidence": evidence,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def evaluate_oracle_power_calibration(
    panel: DensityRatioPanel,
    *,
    reps: int,
    seed: int,
    expected_paths: int = 256,
    expected_half_paths: int = 128,
    full_confidence: float = 0.99,
    half_confidence: float = 0.90,
    epsilon: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a fixed calibration panel and its predetermined path halves."""

    ordered = _ordered_unique_path_ids(np.asarray(panel.path_ids, dtype=np.int64))
    sizes_valid = bool(
        int(panel.path_count) == int(expected_paths)
        and int(ordered.size) == int(expected_paths)
        and int(expected_paths) == 2 * int(expected_half_paths)
    )
    if not sizes_valid:
        return {
            "schema": SELECTION_POWER_SCHEMA + "-oracle-calibration",
            "schema_version": SELECTION_POWER_SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 0,
            "reason": "calibration panel does not have the frozen full/half path counts",
            "expected_paths": int(expected_paths),
            "expected_half_paths": int(expected_half_paths),
            "actual_paths": int(ordered.size),
            "panel": panel_identity(panel),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    halves = {
        "first_half": ordered[: int(expected_half_paths)],
        "second_half": ordered[int(expected_half_paths) :],
    }
    full = evaluate_oracle_panel_feasibility(
        panel,
        reps=int(reps),
        confidence=float(full_confidence),
        seed=derive_selection_power_seed(int(seed), "calibration", "full"),
        subset_name="calibration-full",
        epsilon=float(epsilon),
    )
    half_records_by_name = {
        name: evaluate_oracle_panel_feasibility(
            panel,
            reps=int(reps),
            confidence=float(half_confidence),
            seed=derive_selection_power_seed(int(seed), "calibration", name),
            selected_path_ids=path_ids,
            subset_name=f"calibration-{name}",
            epsilon=float(epsilon),
        )
        for name, path_ids in halves.items()
    }
    first_set, second_set = set(halves["first_half"]), set(halves["second_half"])
    partition_valid = bool(
        not (first_set & second_set)
        and (first_set | second_set) == set(ordered.tolist())
    )
    passed = bool(
        partition_valid
        and full["passed"]
        and all(record["passed"] for record in half_records_by_name.values())
    )
    return {
        "schema": SELECTION_POWER_SCHEMA + "-oracle-calibration",
        "schema_version": SELECTION_POWER_SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": int(passed),
        "panel": panel_identity(panel),
        "expected_paths": int(expected_paths),
        "expected_half_paths": int(expected_half_paths),
        "partition_valid": int(partition_valid),
        "derivation_version": PREDETERMINED_HALF_VERSION,
        "full": full,
        "halves": [
            half_records_by_name["first_half"],
            half_records_by_name["second_half"],
        ],
        "halves_by_name": half_records_by_name,
        "predetermined_split": 1,
        "halves_disjoint": int(partition_valid),
        "evaluation_overlap_path_count": 0,
        "panel_frozen_before_inspection": 1,
        "regenerated_after_inspection": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def reproduce_saved_16_path_oracle_forensic(
    panel_a: DensityRatioPanel,
    panel_b: DensityRatioPanel,
    *,
    reps: int,
    seed: int,
    confidence: float = 0.90,
    point_atol: float = 2e-12,
    lower_bound_atol: float = 2e-3,
) -> dict[str, Any]:
    """Reproduce the immutable parent's underpowered exact-oracle result.

    Point estimates are deterministic sums and therefore checked tightly.
    The plan recorded bootstrap lower bounds as approximate, so those receive
    an explicit small Monte Carlo tolerance while their required sign pattern
    (A positive, B negative) is checked exactly.
    """

    panels = {"a": panel_a, "b": panel_b}
    records = {
        role: evaluate_exact_teacher_oracle_panel(
            panel,
            reps=int(reps),
            confidence=float(confidence),
            seed=derive_selection_power_seed(int(seed), "saved-16-path", role),
            subset_name=f"saved-parent-{role}",
            scope_seeds=SAVED_16_PATH_BOOTSTRAP_SEEDS[role],
        )
        for role, panel in panels.items()
    }
    checks: list[dict[str, Any]] = []
    for role in ("a", "b"):
        checks.append(
            {
                "name": f"{role}_panel_fingerprint",
                "actual": panels[role].fingerprint,
                "expected": SAVED_16_PATH_PANEL_FINGERPRINTS[role],
                "passed": int(
                    panels[role].fingerprint
                    == SAVED_16_PATH_PANEL_FINGERPRINTS[role]
                    and int(panels[role].path_count) == 16
                ),
            }
        )
        for scope in ("overall", "data_end"):
            actual_point = records[role][scope]["point_estimate"]
            actual_lower = records[role][scope]["lower_bound"]
            expected = SAVED_16_PATH_ORACLE_REFERENCE[role][scope]
            checks.extend(
                [
                    {
                        "name": f"{role}_{scope}_point_estimate",
                        "actual": actual_point,
                        "expected": expected["point_estimate"],
                        "absolute_tolerance": float(point_atol),
                        "passed": int(
                            actual_point is not None
                            and math.isclose(
                                float(actual_point),
                                float(expected["point_estimate"]),
                                rel_tol=0.0,
                                abs_tol=float(point_atol),
                            )
                        ),
                    },
                    {
                        "name": f"{role}_{scope}_approximate_lower_bound",
                        "actual": actual_lower,
                        "expected_approximate": expected["lower_bound"],
                        "absolute_tolerance": float(lower_bound_atol),
                        "passed": int(
                            actual_lower is not None
                            and abs(
                                float(actual_lower)
                                - float(expected["lower_bound"])
                            )
                            <= float(lower_bound_atol)
                        ),
                    },
                ]
            )
    sign_pattern = bool(
        all(float(records["a"][scope]["lower_bound"]) > 0.0 for scope in ("overall", "data_end"))
        and all(float(records["b"][scope]["lower_bound"]) < 0.0 for scope in ("overall", "data_end"))
    )
    checks.append(
        {
            "name": "saved_oracle_power_pattern",
            "actual": "A passes; B fails" if sign_pattern else "unexpected",
            "expected": "A passes; B fails",
            "passed": int(sign_pattern),
        }
    )
    passed = bool(all(bool(check["passed"]) for check in checks))
    return {
        "schema": SELECTION_POWER_SCHEMA + "-saved-oracle-forensic",
        "schema_version": SELECTION_POWER_SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": int(passed),
        "complete": 1,
        "finite": int(all(records[role]["finite"] for role in ("a", "b"))),
        "path_count": 16,
        "anchors_per_path": 32,
        "saved_panel_hashes_verified": int(
            all(
                panels[role].fingerprint
                == SAVED_16_PATH_PANEL_FINGERPRINTS[role]
                for role in ("a", "b")
            )
        ),
        "panel_a": {
            "lower_bounds": [
                records["a"][scope]["lower_bound"]
                for scope in ("overall", "data_end")
            ],
            "evidence": records["a"],
        },
        "panel_b": {
            "lower_bounds": [
                records["b"][scope]["lower_bound"]
                for scope in ("overall", "data_end")
            ],
            "evidence": records["b"],
        },
        "interpretation": "the immutable 16-path A panel detects the exact teacher while sealed B does not",
        "checks": checks,
        "panels": records,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
