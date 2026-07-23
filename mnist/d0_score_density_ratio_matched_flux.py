"""Matched whole-path physical-flux comparisons for D0 controls.

This module evaluates two density-ratio potentials on *the same* fixed
``DensityRatioPanel`` and reports the change in relative physical-flux L2
error.  The elementary bootstrap unit is a complete path cluster.  Overall
and data-end statistics are recomputed from the jointly resampled path
energies, so neither slices nor edges are treated as independent evidence.

The helper is deliberately additive and controls-only.  It contains no
training loop, physical-score data, or sampler functionality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .d0_dirichlet_score import (
    edge_difference_channels,
    physical_flux_from_edge_score,
)
from .d0_score_boundary_controls import bounded_teacher_edge_score
from .d0_score_density_ratio import DensityRatioPanel
from .eulerian_flux_mnist import DirectFluxMNISTConfig


MATCHED_FLUX_SCHEMA = "experiment12-d0-density-ratio-matched-flux"
MATCHED_FLUX_SCHEMA_VERSION = 1
MATCHED_FLUX_EVALUATION_VERSION = "d0-matched-teacher-flux-path-energy-v1"
MATCHED_FLUX_BOOTSTRAP_VERSION = (
    "d0-joint-whole-path-centered-max-shortfall-v1"
)
MATCHED_FLUX_SCOPES = ("overall", "data_end")


__all__ = [
    "MATCHED_FLUX_SCHEMA",
    "MATCHED_FLUX_SCHEMA_VERSION",
    "MATCHED_FLUX_EVALUATION_VERSION",
    "MATCHED_FLUX_BOOTSTRAP_VERSION",
    "MATCHED_FLUX_SCOPES",
    "model_state_fingerprint",
    "evaluate_matched_teacher_flux_path_energies",
    "joint_whole_path_relative_flux_reduction_bootstrap",
    "evaluate_matched_teacher_flux_reduction",
    "joint_matched_flux_family_bootstrap",
]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_state_fingerprint(model: nn.Module) -> str:
    """Hash a model class and its complete ordered state dictionary."""

    digest = hashlib.sha256()
    cls = type(model)
    digest.update(f"{cls.__module__}.{cls.__qualname__}".encode("utf-8"))
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _model_physical_flux(
    model: nn.Module,
    *,
    states: Tensor,
    tau: Tensor,
    labels: Tensor,
    config: DirectFluxMNISTConfig,
    batch_size: int,
) -> Tensor:
    """Evaluate one potential's physical flux without retaining model graphs."""

    values: list[Tensor] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.enable_grad():
            for start in range(0, int(states.shape[0]), int(batch_size)):
                end = min(start + int(batch_size), int(states.shape[0]))
                state_batch = states[start:end].detach().clone().requires_grad_(True)
                logits = model(
                    tau[start:end], state_batch, labels[start:end]
                ).reshape(-1)
                if logits.shape != (end - start,):
                    raise ValueError(
                        "density-ratio model must return one raw logit per state"
                    )
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("density-ratio model logits are nonfinite")
                gradient = torch.autograd.grad(logits.sum(), state_batch)[0]
                edge_score = edge_difference_channels(
                    gradient, int(config.grid_size)
                )
                flux = physical_flux_from_edge_score(
                    edge_score, state_batch.detach(), config
                )
                if not bool(torch.isfinite(flux).all()):
                    raise FloatingPointError("predicted physical flux is nonfinite")
                values.append(flux.detach().to(device="cpu", dtype=torch.float64))
    finally:
        model.train(was_training)
    return torch.cat(values, dim=0)


def _scope_record(
    *,
    scope: str,
    path_ids: np.ndarray,
    mask: np.ndarray,
    selected_flux: Tensor,
    baseline_flux: Tensor,
    target_flux: Tensor,
) -> dict[str, Any]:
    ids = np.flatnonzero(mask)
    if ids.size <= 0:
        raise ValueError(f"matched flux scope {scope!r} is empty")
    path_scope = np.asarray(path_ids, dtype=np.int64)[ids]
    unique_paths = np.unique(path_scope)
    if unique_paths.size <= 0:
        raise ValueError(f"matched flux scope {scope!r} has no whole paths")

    index = torch.as_tensor(ids, dtype=torch.long)
    selected = selected_flux.index_select(0, index)
    baseline = baseline_flux.index_select(0, index)
    target = target_flux.index_select(0, index)
    selected_per_state = (selected - target).square().flatten(1).sum(dim=1)
    baseline_per_state = (baseline - target).square().flatten(1).sum(dim=1)
    target_per_state = target.square().flatten(1).sum(dim=1)

    records: list[dict[str, Any]] = []
    for path_id in unique_paths.tolist():
        path_mask = torch.as_tensor(path_scope == int(path_id), dtype=torch.bool)
        state_count = int(path_mask.sum())
        records.append(
            {
                "path_id": int(path_id),
                "state_count": state_count,
                "edge_value_count": int(state_count * target[0].numel()),
                "selected_error_energy": float(
                    selected_per_state[path_mask].sum().item()
                ),
                "baseline_error_energy": float(
                    baseline_per_state[path_mask].sum().item()
                ),
                "target_flux_energy": float(target_per_state[path_mask].sum().item()),
            }
        )
    for record in records:
        for name in (
            "selected_error_energy",
            "baseline_error_energy",
            "target_flux_energy",
        ):
            value = float(record[name])
            if not math.isfinite(value) or value < 0.0:
                raise FloatingPointError(f"{scope} path energy {name} is invalid")

    selected_energy = math.fsum(
        float(record["selected_error_energy"]) for record in records
    )
    baseline_energy = math.fsum(
        float(record["baseline_error_energy"]) for record in records
    )
    target_energy = math.fsum(
        float(record["target_flux_energy"]) for record in records
    )
    if target_energy <= 0.0:
        raise ValueError(f"{scope} target physical-flux energy must be positive")
    if baseline_energy <= 0.0:
        raise ValueError(f"{scope} baseline physical-flux error must be positive")
    selected_relative = math.sqrt(selected_energy / target_energy)
    baseline_relative = math.sqrt(baseline_energy / target_energy)
    reduction = 1.0 - selected_relative / baseline_relative
    if not all(
        math.isfinite(value)
        for value in (selected_relative, baseline_relative, reduction)
    ):
        raise FloatingPointError(f"{scope} matched relative-flux result is nonfinite")

    path_sha = _fingerprint(records)
    return {
        "scope": scope,
        "path_count": len(records),
        "state_count": int(sum(int(record["state_count"]) for record in records)),
        "edge_value_count": int(
            sum(int(record["edge_value_count"]) for record in records)
        ),
        "selected_error_energy": selected_energy,
        "baseline_error_energy": baseline_energy,
        "target_flux_energy": target_energy,
        "selected_relative_flux_l2": selected_relative,
        "baseline_relative_flux_l2": baseline_relative,
        "point_relative_flux_l2_reduction": reduction,
        "path_energies": records,
        "path_energy_sha256": path_sha,
    }


def evaluate_matched_teacher_flux_path_energies(
    selected_model: nn.Module,
    baseline_model: nn.Module,
    panel: DensityRatioPanel,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    evaluate_class_target: int = 1,
    epsilon: float = 0.5,
    selected_role: str = "selected",
    baseline_role: str = "rho_zero",
    evaluation_role: str | None = None,
) -> dict[str, Any]:
    """Accumulate matched per-path flux energies on bounded-teacher states."""

    if panel.task != "bounded_teacher":
        raise ValueError("matched teacher flux requires a bounded-teacher panel")
    if int(config.grid_size) ** 2 != int(panel.states.shape[1]):
        raise ValueError("panel and dynamics grid sizes disagree")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    class_value = int(evaluate_class_target)
    if class_value not in (0, 1):
        raise ValueError("evaluate_class_target must be zero or one")
    if not str(selected_role).strip() or not str(baseline_role).strip():
        raise ValueError("selected and baseline roles must be nonempty")

    targets = panel.class_targets.detach().cpu().numpy().astype(np.int64)
    chosen = np.flatnonzero(targets == class_value)
    if chosen.size <= 0:
        raise ValueError("matched teacher flux class selection is empty")
    chosen_tensor = torch.as_tensor(chosen, dtype=torch.long, device=panel.states.device)
    target_device = torch.device(device)
    states = panel.states.index_select(0, chosen_tensor).to(target_device)
    tau = panel.tau.index_select(0, chosen_tensor).to(target_device)
    tau_fraction = panel.tau_fraction.index_select(0, chosen_tensor).to(target_device)
    labels = panel.labels.index_select(0, chosen_tensor).to(target_device)
    path_ids = np.asarray(panel.path_ids, dtype=np.int64)[chosen]
    strata = np.asarray(panel.strata, dtype=np.int64)[chosen]

    selected_flux = _model_physical_flux(
        selected_model,
        states=states,
        tau=tau,
        labels=labels,
        config=config,
        batch_size=int(batch_size),
    )
    baseline_flux = _model_physical_flux(
        baseline_model,
        states=states,
        tau=tau,
        labels=labels,
        config=config,
        batch_size=int(batch_size),
    )
    target_edge = bounded_teacher_edge_score(
        states, tau_fraction, epsilon=float(epsilon)
    )
    target_flux = physical_flux_from_edge_score(target_edge, states, config)
    if not bool(torch.isfinite(target_flux).all()):
        raise FloatingPointError("analytic target physical flux is nonfinite")
    target_flux_cpu = target_flux.detach().to(device="cpu", dtype=torch.float64)

    scopes = {
        "overall": _scope_record(
            scope="overall",
            path_ids=path_ids,
            mask=np.ones(chosen.size, dtype=bool),
            selected_flux=selected_flux,
            baseline_flux=baseline_flux,
            target_flux=target_flux_cpu,
        ),
        "data_end": _scope_record(
            scope="data_end",
            path_ids=path_ids,
            mask=strata == 4,
            selected_flux=selected_flux,
            baseline_flux=baseline_flux,
            target_flux=target_flux_cpu,
        ),
    }
    common_paths = [
        int(record["path_id"]) for record in scopes["overall"]["path_energies"]
    ]
    if common_paths != [
        int(record["path_id"]) for record in scopes["data_end"]["path_energies"]
    ]:
        raise ValueError("overall and data-end scopes do not share aligned whole paths")

    result: dict[str, Any] = {
        "schema": MATCHED_FLUX_SCHEMA + "-path-energies",
        "schema_version": MATCHED_FLUX_SCHEMA_VERSION,
        "evaluation_version": MATCHED_FLUX_EVALUATION_VERSION,
        "evaluation_status": "evaluated",
        "finite": 1,
        "phase": panel.phase,
        "panel_role": panel.role,
        "evaluation_role": str(evaluation_role or panel.role),
        "task": panel.task,
        "evaluate_class_target": class_value,
        "epsilon": float(epsilon),
        "panel_fingerprint": panel.fingerprint,
        "selected_role": str(selected_role),
        "baseline_role": str(baseline_role),
        "selected_model_sha256": model_state_fingerprint(selected_model),
        "baseline_model_sha256": model_state_fingerprint(baseline_model),
        "path_ids": common_paths,
        "path_ids_sha256": _array_fingerprint(
            np.asarray(common_paths, dtype=np.int64)
        ),
        "scopes": scopes,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result["record_sha256"] = _fingerprint(result)
    return result


def _ordered_energy_arrays(
    evaluation: Mapping[str, Any],
) -> tuple[list[int], dict[str, dict[str, np.ndarray]]]:
    scopes = evaluation.get("scopes")
    if not isinstance(scopes, Mapping):
        raise ValueError("matched flux evaluation lacks scope records")
    ordered_paths = sorted(int(value) for value in evaluation.get("path_ids", []))
    if not ordered_paths or len(set(ordered_paths)) != len(ordered_paths):
        raise ValueError("matched flux evaluation has invalid path IDs")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for scope in MATCHED_FLUX_SCOPES:
        raw_scope = scopes.get(scope)
        if not isinstance(raw_scope, Mapping):
            raise ValueError(f"matched flux evaluation lacks {scope} scope")
        records = raw_scope.get("path_energies")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ValueError(f"{scope} path energies are invalid")
        by_path: dict[int, Mapping[str, Any]] = {}
        for value in records:
            if not isinstance(value, Mapping):
                raise ValueError(f"{scope} path energy record is invalid")
            path_id = int(value.get("path_id", -1))
            if path_id in by_path:
                raise ValueError(f"{scope} contains a duplicate path ID")
            by_path[path_id] = value
        if sorted(by_path) != ordered_paths:
            raise ValueError(f"{scope} does not contain the aligned path family")
        scope_arrays: dict[str, np.ndarray] = {}
        for key in (
            "selected_error_energy",
            "baseline_error_energy",
            "target_flux_energy",
        ):
            array = np.asarray(
                [float(by_path[path_id].get(key, math.nan)) for path_id in ordered_paths],
                dtype=np.float64,
            )
            if not np.isfinite(array).all() or np.any(array < 0.0):
                raise ValueError(f"{scope} {key} values are invalid")
            scope_arrays[key] = array
        if float(scope_arrays["target_flux_energy"].sum()) <= 0.0:
            raise ValueError(f"{scope} target energy must be positive")
        if float(scope_arrays["baseline_error_energy"].sum()) <= 0.0:
            raise ValueError(f"{scope} baseline error energy must be positive")
        arrays[scope] = scope_arrays
    return ordered_paths, arrays


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility.
        return float(np.quantile(values, probability, interpolation="higher"))


def joint_whole_path_relative_flux_reduction_bootstrap(
    evaluation: Mapping[str, Any],
    *,
    reps: int = 50_000,
    confidence: float = 0.95,
    seed: int,
    chunk_size: int = 2_048,
) -> dict[str, Any]:
    """Return joint one-sided lower bounds for two matched flux reductions.

    A single path-index matrix is used for both scopes.  Each bootstrap
    replicate recomputes the ratio of aggregate L2 norms.  Simultaneous lower
    bounds use the ``confidence`` quantile of the largest centered shortfall
    across the two scopes; conservative ``higher`` interpolation is frozen.
    """

    count = int(reps)
    if count <= 0:
        raise ValueError("bootstrap reps must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must lie strictly between zero and one")
    if int(chunk_size) <= 0:
        raise ValueError("bootstrap chunk_size must be positive")
    path_ids, arrays = _ordered_energy_arrays(evaluation)
    n_paths = len(path_ids)

    point: dict[str, float] = {}
    for scope in MATCHED_FLUX_SCOPES:
        selected_sum = float(arrays[scope]["selected_error_energy"].sum())
        baseline_sum = float(arrays[scope]["baseline_error_energy"].sum())
        if baseline_sum <= 0.0:
            raise ValueError(f"{scope} baseline relative-flux error is zero")
        point[scope] = 1.0 - math.sqrt(selected_sum / baseline_sum)

    generator = np.random.default_rng(int(seed))
    reductions = {
        scope: np.empty(count, dtype=np.float64) for scope in MATCHED_FLUX_SCOPES
    }
    cursor = 0
    while cursor < count:
        width = min(int(chunk_size), count - cursor)
        indices = generator.integers(0, n_paths, size=(width, n_paths))
        for scope in MATCHED_FLUX_SCOPES:
            selected = arrays[scope]["selected_error_energy"][indices].sum(axis=1)
            baseline = arrays[scope]["baseline_error_energy"][indices].sum(axis=1)
            if np.any(baseline <= 0.0):
                raise ValueError(
                    f"{scope} bootstrap encountered zero baseline error energy"
                )
            reductions[scope][cursor : cursor + width] = 1.0 - np.sqrt(
                selected / baseline
            )
        cursor += width

    matrix = np.stack([reductions[scope] for scope in MATCHED_FLUX_SCOPES], axis=1)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("matched flux bootstrap is nonfinite")
    point_vector = np.asarray([point[scope] for scope in MATCHED_FLUX_SCOPES])
    maximum_shortfall = np.max(point_vector[None, :] - matrix, axis=1)
    critical = max(0.0, _higher_quantile(maximum_shortfall, float(confidence)))

    scope_records: dict[str, dict[str, Any]] = {}
    for index, scope in enumerate(MATCHED_FLUX_SCOPES):
        samples = reductions[scope]
        lower = point[scope] - critical
        scope_records[scope] = {
            "point_relative_flux_l2_reduction": point[scope],
            "simultaneous_lower_bound": lower,
            "positive_simultaneous_lower_bound": int(lower > 0.0),
            "marginal_lower_quantile": _higher_quantile(
                samples, 1.0 - float(confidence)
            ),
            "replicate_mean": float(samples.mean()),
            "replicate_standard_deviation": float(samples.std(ddof=1))
            if count > 1
            else 0.0,
            "replicate_reduction_sha256": _array_fingerprint(samples),
            "scope_index": index,
        }

    plan = {
        "bootstrap_version": MATCHED_FLUX_BOOTSTRAP_VERSION,
        "seed": int(seed),
        "replicates": count,
        "confidence": float(confidence),
        "tail_probability": 1.0 - float(confidence),
        "quantile_interpolation": "higher",
        "cluster_unit": "whole_path_id",
        "scope_coupling": "same_resampled_path_indices",
        "scopes": list(MATCHED_FLUX_SCOPES),
        "path_count": n_paths,
        "path_ids_sha256": _array_fingerprint(
            np.asarray(path_ids, dtype=np.int64)
        ),
        "source_evaluation_sha256": str(evaluation.get("record_sha256", "")),
        "panel_fingerprint": str(evaluation.get("panel_fingerprint", "")),
        "panel_role": str(evaluation.get("panel_role", "")),
        "evaluation_role": str(evaluation.get("evaluation_role", "")),
        "selected_role": str(evaluation.get("selected_role", "")),
        "baseline_role": str(evaluation.get("baseline_role", "")),
    }
    result = {
        "schema": MATCHED_FLUX_SCHEMA + "-bootstrap",
        "schema_version": MATCHED_FLUX_SCHEMA_VERSION,
        **plan,
        "plan_sha256": _fingerprint(plan),
        "finite": 1,
        "simultaneous_critical_shortfall": critical,
        "scopes": scope_records,
        "point_reductions": [
            point[scope] for scope in MATCHED_FLUX_SCOPES
        ],
        "simultaneous_lower_bounds": [
            scope_records[scope]["simultaneous_lower_bound"]
            for scope in MATCHED_FLUX_SCOPES
        ],
        "joint_shortfall_sha256": _array_fingerprint(maximum_shortfall),
        "all_simultaneous_lower_bounds_positive": int(
            all(
                int(scope_records[scope]["positive_simultaneous_lower_bound"]) == 1
                for scope in MATCHED_FLUX_SCOPES
            )
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result["record_sha256"] = _fingerprint(result)
    return result


def joint_matched_flux_family_bootstrap(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    reps: int = 50_000,
    confidence: float = 0.95,
    chunk_size: int = 1_024,
) -> dict[str, Any]:
    """Simultaneously bound a seed x panel-role x scope effect family.

    Whole-path indices are shared across model seeds within a panel role and
    are drawn independently for different panel roles.  This is the coupling
    used by the gradient-controlled H1 confirmation gate.  The statistic for
    every member remains the relative physical-flux L2 reduction
    ``1 - ||err_selected|| / ||err_baseline||``.
    """

    count = int(reps)
    if count <= 0:
        raise ValueError("bootstrap reps must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must lie strictly between zero and one")
    if int(chunk_size) <= 0:
        raise ValueError("bootstrap chunk_size must be positive")
    if not evaluations:
        raise ValueError("matched flux family is empty")

    prepared: list[dict[str, Any]] = []
    role_paths: dict[str, list[int]] = {}
    for raw in evaluations:
        value = dict(raw)
        role = str(value.get("panel_role", value.get("evaluation_role", ""))).lower()
        model_seed = value.get("model_seed", value.get("seed"))
        if not role or model_seed is None:
            raise ValueError("matched flux family member lacks seed or panel role")
        path_ids, arrays = _ordered_energy_arrays(value)
        if role in role_paths and role_paths[role] != path_ids:
            raise ValueError(f"{role} path IDs are not aligned across model seeds")
        role_paths.setdefault(role, path_ids)
        prepared.append(
            {
                "record": value,
                "model_seed": int(model_seed),
                "panel_role": role,
                "path_ids": path_ids,
                "arrays": arrays,
            }
        )
    identity_keys = [
        (item["model_seed"], item["panel_role"]) for item in prepared
    ]
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("matched flux family contains a duplicate seed/panel role")
    prepared.sort(key=lambda item: (item["model_seed"], item["panel_role"]))

    member_specs: list[tuple[dict[str, Any], str]] = [
        (item, scope) for item in prepared for scope in MATCHED_FLUX_SCOPES
    ]
    points = np.asarray(
        [
            1.0
            - math.sqrt(
                float(item["arrays"][scope]["selected_error_energy"].sum())
                / float(item["arrays"][scope]["baseline_error_energy"].sum())
            )
            for item, scope in member_specs
        ],
        dtype=np.float64,
    )
    if not np.isfinite(points).all():
        raise FloatingPointError("matched flux family point estimates are nonfinite")

    generator = np.random.default_rng(int(seed))
    maximum_shortfall = np.empty(count, dtype=np.float64)
    cursor = 0
    roles = sorted(role_paths)
    while cursor < count:
        width = min(int(chunk_size), count - cursor)
        role_indices = {
            role: generator.integers(
                0, len(role_paths[role]), size=(width, len(role_paths[role]))
            )
            for role in roles
        }
        replicate_columns: list[np.ndarray] = []
        for item, scope in member_specs:
            indices = role_indices[item["panel_role"]]
            arrays = item["arrays"][scope]
            selected = arrays["selected_error_energy"][indices].sum(axis=1)
            baseline = arrays["baseline_error_energy"][indices].sum(axis=1)
            if np.any(baseline <= 0.0):
                raise ValueError("matched flux family encountered zero baseline energy")
            replicate_columns.append(1.0 - np.sqrt(selected / baseline))
        matrix = np.stack(replicate_columns, axis=1)
        if not np.isfinite(matrix).all():
            raise FloatingPointError("matched flux family bootstrap is nonfinite")
        maximum_shortfall[cursor : cursor + width] = np.max(
            points[None, :] - matrix, axis=1
        )
        cursor += width

    critical = max(0.0, _higher_quantile(maximum_shortfall, float(confidence)))
    lowers = points - critical
    members: list[dict[str, Any]] = []
    for index, ((item, scope), point, lower) in enumerate(
        zip(member_specs, points, lowers, strict=True)
    ):
        members.append(
            {
                "member_index": index,
                "name": f"seed-{item['model_seed']}/{item['panel_role']}/{scope}",
                "seed": int(item["model_seed"]),
                "model_seed": int(item["model_seed"]),
                "panel_role": item["panel_role"],
                "scope": scope,
                "point_relative_flux_l2_reduction": float(point),
                "simultaneous_lower_bound": float(lower),
                "positive_simultaneous_lower_bound": int(float(lower) > 0.0),
                "panel_fingerprint": item["record"].get("panel_fingerprint"),
                "source_evaluation_sha256": item["record"].get("record_sha256"),
            }
        )
    result = {
        "schema": MATCHED_FLUX_SCHEMA + "-family-bootstrap",
        "schema_version": MATCHED_FLUX_SCHEMA_VERSION,
        "bootstrap_version": MATCHED_FLUX_BOOTSTRAP_VERSION + "-family-v1",
        "evaluation_status": "evaluated",
        "passed": 1,
        "finite": 1,
        "seed": int(seed),
        "replicates": count,
        "confidence": float(confidence),
        "quantile_interpolation": "higher",
        "path_coupling": "joint-within-role-independent-across-roles",
        "family_size": len(members),
        "simultaneous_critical_shortfall": critical,
        "members": members,
        "all_simultaneous_lower_bounds_positive": int(
            all(float(value["simultaneous_lower_bound"]) > 0.0 for value in members)
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result["record_sha256"] = _fingerprint(result)
    return result


def evaluate_matched_teacher_flux_reduction(
    selected_model: nn.Module,
    baseline_model: nn.Module,
    panel: DensityRatioPanel,
    config: DirectFluxMNISTConfig,
    *,
    seed: int,
    reps: int = 50_000,
    confidence: float = 0.95,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
    evaluate_class_target: int = 1,
    epsilon: float = 0.5,
    selected_role: str = "selected",
    baseline_role: str = "rho_zero",
    evaluation_role: str | None = None,
) -> dict[str, Any]:
    """Evaluate both models and attach deterministic simultaneous bounds."""

    evaluation = evaluate_matched_teacher_flux_path_energies(
        selected_model,
        baseline_model,
        panel,
        config,
        device=device,
        batch_size=int(batch_size),
        evaluate_class_target=int(evaluate_class_target),
        epsilon=float(epsilon),
        selected_role=selected_role,
        baseline_role=baseline_role,
        evaluation_role=evaluation_role,
    )
    bootstrap = joint_whole_path_relative_flux_reduction_bootstrap(
        evaluation,
        reps=int(reps),
        confidence=float(confidence),
        seed=int(seed),
    )
    result = {
        "schema": MATCHED_FLUX_SCHEMA + "-report",
        "schema_version": MATCHED_FLUX_SCHEMA_VERSION,
        "evaluation": evaluation,
        "simultaneous_bootstrap": bootstrap,
        # Compact aliases are consumed directly by the pure gate layer.  Their
        # order is frozen as [overall, data_end].
        "scope_order": list(MATCHED_FLUX_SCOPES),
        "point_reductions": list(bootstrap["point_reductions"]),
        "simultaneous_lower_bounds": list(
            bootstrap["simultaneous_lower_bounds"]
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result["record_sha256"] = _fingerprint(result)
    return result
