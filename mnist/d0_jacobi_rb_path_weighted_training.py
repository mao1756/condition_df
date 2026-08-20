"""Training utilities for the path-weighted Jacobi capacity experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping

import numpy as np
import torch
from torch import Tensor, nn

from mnist import eulerian_jacobi_ddpm as core
from mnist.d0_jacobi_rb_boundary_tangent import edge_pair_geometry
from mnist.d0_jacobi_rb_candidate_training_cache import (
    CandidatePrefixCache,
    cache_mobility_numpy,
)
from mnist.d0_jacobi_rb_global_large import LargeEulerianJacobiDDPMModel
from mnist.d0_jacobi_rb_path_weighted_loss import (
    PathWeightedLossConfig,
    mobility_weight_statistics,
    path_weighted_raw_target_mse,
)

PATH_WEIGHTED_TRAINING_VERSION = "d0-jacobi-rb-path-weighted-training-v1"
ArchitectureName = Literal["small", "large"]
LossName = Literal["old", "path-weighted"]


class PathWeightedTrainingError(core.EulerianJacobiDDPMError):
    """The training or validation contract was violated."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_capacity_model(architecture: ArchitectureName) -> nn.Module:
    if architecture == "small":
        return core.make_model()
    if architecture == "large":
        return LargeEulerianJacobiDDPMModel()
    raise PathWeightedTrainingError(f"unsupported architecture: {architecture!r}")


def _prevalidated_q_and_m(
    model: nn.Module, inputs: core.ModelInputs
) -> tuple[Tensor, Tensor, Tensor]:
    mobility = edge_pair_geometry(inputs).mobility
    q = model.predictor.score_prediction_prevalidated(inputs).to(dtype=torch.float64)
    prediction = torch.where(mobility == 0.0, torch.zeros_like(q), mobility * q)
    return q, prediction, mobility


@dataclass(frozen=True)
class CacheLossScales:
    path_weighted_target_scale_squared: float
    old_target_energy: float
    mobility_floor: float
    mobility_decile_edges: tuple[float, ...]
    mobility_statistics: Mapping[str, Any]
    active_lane_count: int
    total_lane_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "mobility_decile_edges": list(self.mobility_decile_edges),
            "mobility_statistics": dict(self.mobility_statistics),
        }


def compute_cache_loss_scales(
    cache: CandidatePrefixCache,
    *,
    loss_config: PathWeightedLossConfig | None = None,
    rows_per_chunk: int = 256,
    maximum_quantile_sample: int = 1_000_000,
) -> CacheLossScales:
    """Compute training-only normalizers without loading the cache into RAM."""

    config = loss_config or PathWeightedLossConfig()
    weighted_sum = 0.0
    target_sum = 0.0
    active_count = 0
    total_count = 0
    mobility_sample: list[np.ndarray] = []
    sample_count = 0
    stride = max(1, (len(cache) * core.EDGES_PER_PHASE) // int(maximum_quantile_sample))
    for indices in cache.iter_indices(rows_per_chunk):
        later = np.asarray(cache.array("later_states")[indices], dtype=np.float64)
        colors = np.asarray(cache.array("color")[indices], dtype=np.int64)
        targets = np.asarray(cache.array("targets")[indices], dtype=np.float64)
        mobility = cache_mobility_numpy(later, colors)
        active = mobility > 0.0
        if np.any(targets[~active] != 0.0):
            raise PathWeightedTrainingError(
                "cache target is nonzero on a zero-mobility lane"
            )
        denominator = np.maximum(mobility, float(config.mobility_floor))
        weighted_sum += float(np.sum(targets[active] ** 2 / denominator[active]))
        target_sum += float(np.sum(targets**2))
        active_count += int(np.count_nonzero(active))
        total_count += int(targets.size)
        flat = mobility[active].reshape(-1)
        if flat.size and sample_count < maximum_quantile_sample:
            sampled = flat[::stride]
            remaining = int(maximum_quantile_sample) - sample_count
            sampled = sampled[:remaining]
            mobility_sample.append(np.asarray(sampled, dtype=np.float64))
            sample_count += int(sampled.size)
    if active_count <= 0 or total_count <= 0:
        raise PathWeightedTrainingError("cache has no trainable target lanes")
    weighted_scale = weighted_sum / active_count
    old_energy = target_sum / total_count
    if (
        not math.isfinite(weighted_scale)
        or weighted_scale <= 0.0
        or not math.isfinite(old_energy)
        or old_energy <= 0.0
    ):
        raise PathWeightedTrainingError("training target normalizer is invalid")
    sampled_mobility = np.concatenate(mobility_sample)
    quantiles = np.quantile(sampled_mobility, np.linspace(0.0, 1.0, 11))
    # Repeated values are valid; bucket assignment uses searchsorted.
    statistics = mobility_weight_statistics(sampled_mobility, config=config).to_record()
    return CacheLossScales(
        path_weighted_target_scale_squared=float(weighted_scale),
        old_target_energy=float(old_energy),
        mobility_floor=float(config.mobility_floor),
        mobility_decile_edges=tuple(float(value) for value in quantiles),
        mobility_statistics=statistics,
        active_lane_count=active_count,
        total_lane_count=total_count,
    )


@dataclass(frozen=True)
class CapacityTrainingConfig:
    architecture: ArchitectureName
    loss_name: LossName
    updates: int
    batch_size: int
    learning_rate: float
    validation_interval: int = 250
    validation_batch_size: int = 128
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    seed: int = 0xD051A7

    def __post_init__(self) -> None:
        if self.architecture not in {"small", "large"}:
            raise PathWeightedTrainingError("architecture must be small or large")
        if self.loss_name not in {"old", "path-weighted"}:
            raise PathWeightedTrainingError("loss_name is unsupported")
        if (
            int(self.updates) <= 0
            or int(self.batch_size) <= 0
            or float(self.learning_rate) <= 0.0
            or int(self.validation_interval) <= 0
            or int(self.validation_batch_size) <= 0
            or not 0.0 <= float(self.ema_decay) < 1.0
            or float(self.gradient_clip_norm) <= 0.0
        ):
            raise PathWeightedTrainingError("training hyperparameters are invalid")

    def to_record(self) -> dict[str, Any]:
        return {"schema": PATH_WEIGHTED_TRAINING_VERSION + "-config", **asdict(self)}


class _ScalarAccumulator:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def add(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64).reshape(-1)
        if finite.size:
            self.sum += float(np.sum(finite, dtype=np.float64))
            self.count += int(finite.size)

    def mean(self) -> float:
        return self.sum / self.count if self.count else math.nan


def _empty_strata() -> dict[str, MutableMapping[str, _ScalarAccumulator]]:
    return {
        "time_quartile": {str(index): _ScalarAccumulator() for index in range(4)},
        "phase": {str(index): _ScalarAccumulator() for index in range(7)},
        "midpoint": {str(index): _ScalarAccumulator() for index in range(8)},
        "mobility_decile": {str(index): _ScalarAccumulator() for index in range(10)},
    }


def evaluate_capacity_model(
    model: nn.Module,
    cache: CandidatePrefixCache,
    *,
    device: str | torch.device,
    scales: CacheLossScales,
    batch_size: int = 128,
    loss_config: PathWeightedLossConfig | None = None,
) -> dict[str, Any]:
    """Evaluate both losses and their declared temporal/geometric strata."""

    config = loss_config or PathWeightedLossConfig(
        mobility_floor=scales.mobility_floor
    )
    active_device = torch.device(device)
    model = model.to(active_device).eval()
    weighted = _ScalarAccumulator()
    unweighted = _ScalarAccumulator()
    q_squares = _ScalarAccumulator()
    m_squares = _ScalarAccumulator()
    maximum_q = 0.0
    maximum_m = 0.0
    strata = _empty_strata()
    edges = np.asarray(scales.mobility_decile_edges, dtype=np.float64)
    with torch.no_grad():
        for indices in cache.iter_indices(batch_size):
            inputs, target = cache.batch(indices, device=active_device)
            q, prediction, mobility = _prevalidated_q_and_m(model, inputs)
            active = mobility > 0.0
            denominator = mobility.clamp_min(float(config.mobility_floor))
            residual_squared = (prediction - target).square()
            weighted_lane = torch.zeros_like(residual_squared)
            weighted_lane[active] = residual_squared[active] / denominator[active]
            weighted_values = weighted_lane[active].detach().cpu().numpy()
            unweighted_values = residual_squared.detach().cpu().numpy()
            weighted.add(weighted_values)
            unweighted.add(unweighted_values)
            q_values = q.detach().cpu().numpy()
            m_values = prediction.detach().cpu().numpy()
            q_squares.add(q_values**2)
            m_squares.add(m_values**2)
            maximum_q = max(maximum_q, float(np.max(np.abs(q_values))))
            maximum_m = max(maximum_m, float(np.max(np.abs(m_values))))

            row_weighted = weighted_lane.detach().cpu().numpy()
            active_np = active.detach().cpu().numpy()
            phases = np.asarray(cache.array("phase")[indices], dtype=np.int64)
            steps = np.asarray(cache.array("outer_steps")[indices], dtype=np.int64)
            midpoint = np.asarray(
                cache.array("midpoint_indices")[indices], dtype=np.int64
            )
            for row in range(len(indices)):
                row_values = row_weighted[row][active_np[row]]
                strata["time_quartile"][str(min(3, int(steps[row]) // 128))].add(
                    row_values
                )
                strata["phase"][str(int(phases[row]))].add(row_values)
                strata["midpoint"][str(int(midpoint[row]))].add(row_values)
            mobility_np = mobility.detach().cpu().numpy()
            deciles = np.searchsorted(edges[1:-1], mobility_np, side="right")
            for decile in range(10):
                mask = active_np & (deciles == decile)
                strata["mobility_decile"][str(decile)].add(row_weighted[mask])
    weighted_raw = weighted.mean()
    unweighted_raw = unweighted.mean()
    if not math.isfinite(weighted_raw) or not math.isfinite(unweighted_raw):
        raise PathWeightedTrainingError("validation produced a nonfinite loss")
    return {
        "schema": PATH_WEIGHTED_TRAINING_VERSION + "-evaluation",
        "record_count": len(cache),
        "path_weighted_raw_mse": weighted_raw,
        "path_weighted_normalized_mse": (
            weighted_raw / scales.path_weighted_target_scale_squared
        ),
        "unweighted_raw_mse": unweighted_raw,
        "old_normalized_mse": unweighted_raw / scales.old_target_energy,
        "q_rms": math.sqrt(q_squares.mean()),
        "m_rms": math.sqrt(m_squares.mean()),
        "maximum_absolute_q": maximum_q,
        "maximum_absolute_m": maximum_m,
        "strata_path_weighted_raw_mse": {
            name: {key: accumulator.mean() for key, accumulator in values.items()}
            for name, values in strata.items()
        },
    }


def _primary_metric(evaluation: Mapping[str, Any], loss_name: LossName) -> float:
    key = (
        "path_weighted_normalized_mse"
        if loss_name == "path-weighted"
        else "old_normalized_mse"
    )
    return float(evaluation[key])


def _loss_for_batch(
    prediction: Tensor,
    target: Tensor,
    mobility: Tensor,
    *,
    loss_name: LossName,
    scales: CacheLossScales,
    loss_config: PathWeightedLossConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    if loss_name == "path-weighted":
        return path_weighted_raw_target_mse(
            prediction,
            target,
            mobility,
            target_scale_squared=scales.path_weighted_target_scale_squared,
            config=loss_config,
        )
    residual = prediction.to(torch.float64) - target.to(torch.float64)
    raw = torch.mean(residual.square())
    normalized = raw / float(scales.old_target_energy)
    return normalized, raw, raw


def train_capacity_model(
    train_cache: CandidatePrefixCache,
    validation_cache: CandidatePrefixCache,
    *,
    config: CapacityTrainingConfig,
    device: str | torch.device,
    scales: CacheLossScales,
    output_dir: str | Path,
    loss_config: PathWeightedLossConfig | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train, EMA-average, select, and persist one experiment row."""

    objective = loss_config or PathWeightedLossConfig(
        mobility_floor=scales.mobility_floor
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_record = {
        **config.to_record(),
        "loss_config": objective.to_record(),
        "scales": scales.to_record(),
    }
    config_hash = _json_sha256(config_record)
    _atomic_json(destination / "config.json", {**config_record, "sha256": config_hash})
    checkpoint_path = destination / "training_checkpoint.pt"
    selected_path = destination / "selected_model.pt"

    active_device = torch.device(device)
    torch.manual_seed(int(config.seed))
    if active_device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))
    model = make_capacity_model(config.architecture).to(active_device)
    ema = make_capacity_model(config.architecture).to(active_device)
    ema.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.learning_rate), weight_decay=0.0
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed) ^ 0x51A7)
    history: list[dict[str, Any]] = []
    best_metric = math.inf
    best_update = -1
    best_state: dict[str, Tensor] | None = None
    completed_update = 0

    if resume and checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("config_sha256") != config_hash:
            raise PathWeightedTrainingError("checkpoint config does not match this run")
        model.load_state_dict(payload["model_state_dict"])
        ema.load_state_dict(payload["ema_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload["history"])
        best_metric = float(payload["best_metric"])
        best_update = int(payload["best_update"])
        best_state = payload.get("best_state_dict")
        completed_update = int(payload["completed_update"])
        generator.set_state(payload["batch_generator_state"])

    if completed_update == 0:
        zero_evaluation = evaluate_capacity_model(
            ema,
            validation_cache,
            device=active_device,
            scales=scales,
            batch_size=config.validation_batch_size,
            loss_config=objective,
        )
        history.append(
            {
                "update": 0,
                "eligible": 0,
                "primary_validation_metric": _primary_metric(
                    zero_evaluation, config.loss_name
                ),
                "validation": zero_evaluation,
            }
        )

    for update in range(completed_update + 1, int(config.updates) + 1):
        indices = torch.randint(
            0,
            len(train_cache),
            (int(config.batch_size),),
            generator=generator,
        ).numpy()
        inputs, target = train_cache.batch(indices, device=active_device)
        _q, prediction, mobility = _prevalidated_q_and_m(model, inputs)
        normalized, weighted_or_raw, raw_unweighted = _loss_for_batch(
            prediction,
            target,
            mobility,
            loss_name=config.loss_name,
            scales=scales,
            loss_config=objective,
        )
        optimizer.zero_grad(set_to_none=True)
        normalized.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config.gradient_clip_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                ema.parameters(), model.parameters(), strict=True
            ):
                ema_parameter.mul_(float(config.ema_decay)).add_(
                    parameter, alpha=1.0 - float(config.ema_decay)
                )

        if update % int(config.validation_interval) == 0 or update == int(config.updates):
            evaluation = evaluate_capacity_model(
                ema,
                validation_cache,
                device=active_device,
                scales=scales,
                batch_size=config.validation_batch_size,
                loss_config=objective,
            )
            metric = _primary_metric(evaluation, config.loss_name)
            row = {
                "update": update,
                "eligible": int(math.isfinite(metric)),
                "primary_validation_metric": metric,
                "training_batch_normalized_loss": float(normalized.detach().cpu()),
                "training_batch_weighted_or_raw_loss": float(
                    weighted_or_raw.detach().cpu()
                ),
                "training_batch_unweighted_mse": float(raw_unweighted.detach().cpu()),
                "gradient_norm_before_clipping": float(
                    torch.as_tensor(gradient_norm).detach().cpu()
                ),
                "validation": evaluation,
            }
            history.append(row)
            if math.isfinite(metric) and (
                metric < best_metric or (metric == best_metric and update < best_update)
            ):
                best_metric = metric
                best_update = update
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in ema.state_dict().items()
                }
            checkpoint = {
                "schema": PATH_WEIGHTED_TRAINING_VERSION + "-checkpoint",
                "config_sha256": config_hash,
                "completed_update": update,
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "ema_state_dict": {
                    name: value.detach().cpu() for name, value in ema.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "batch_generator_state": generator.get_state(),
                "history": history,
                "best_metric": best_metric,
                "best_update": best_update,
                "best_state_dict": best_state,
            }
            _atomic_torch_save(checkpoint_path, checkpoint)
            _atomic_json(destination / "history.json", {"rows": history})

    if best_state is None or best_update <= 0:
        raise PathWeightedTrainingError("no finite nonzero checkpoint was selected")
    selected_model = make_capacity_model(config.architecture)
    selected_model.load_state_dict(best_state)
    selected_evaluation = evaluate_capacity_model(
        selected_model,
        validation_cache,
        device=active_device,
        scales=scales,
        batch_size=config.validation_batch_size,
        loss_config=objective,
    )
    selected_payload = {
        "schema": PATH_WEIGHTED_TRAINING_VERSION + "-selected-model",
        "config_sha256": config_hash,
        "architecture": config.architecture,
        "loss_name": config.loss_name,
        "selected_update": best_update,
        "selected_primary_metric": best_metric,
        "state_dict": best_state,
        "validation": selected_evaluation,
    }
    _atomic_torch_save(selected_path, selected_payload)
    report = {
        "schema": PATH_WEIGHTED_TRAINING_VERSION + "-report",
        "architecture": config.architecture,
        "loss_name": config.loss_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "completed_updates": int(config.updates),
        "selected_update": best_update,
        "selected_primary_metric": best_metric,
        "selected_validation": selected_evaluation,
        "history_length": len(history),
        "selected_model": str(selected_path.name),
    }
    _atomic_json(destination / "report.json", report)
    return report


def load_selected_model(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, Mapping[str, Any]]:
    payload = torch.load(
        Path(directory) / "selected_model.pt", map_location="cpu", weights_only=False
    )
    architecture = str(payload["architecture"])
    if architecture not in {"small", "large"}:
        raise PathWeightedTrainingError("selected model has unknown architecture")
    model = make_capacity_model(architecture)  # type: ignore[arg-type]
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def run_large_memorization_gate(
    cache: CandidatePrefixCache,
    *,
    device: str | torch.device,
    output_dir: str | Path,
    loss_config: PathWeightedLossConfig | None = None,
    rows: int = 256,
    updates: int = 1_000,
    batch_size: int = 32,
    learning_rate: float = 3.0e-4,
    required_reduction: float = 0.90,
    seed: int = 0x1A26E,
) -> dict[str, Any]:
    """Prove that the large architecture can fit a fixed finite target subset."""

    objective = loss_config or PathWeightedLossConfig()
    count = min(int(rows), len(cache))
    if count <= 0:
        raise PathWeightedTrainingError("memorization subset is empty")
    subset = np.linspace(0, len(cache) - 1, count, dtype=np.int64)
    inputs_all, target_all = cache.batch(subset, device=device)
    mobility_all = edge_pair_geometry(inputs_all).mobility
    active = mobility_all > 0.0
    denominator = mobility_all.clamp_min(float(objective.mobility_floor))
    scale = torch.mean(target_all[active].square() / denominator[active]).detach()
    model = make_capacity_model("large").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        _q, initial_prediction, _mobility = _prevalidated_q_and_m(model, inputs_all)
        initial, _, _ = path_weighted_raw_target_mse(
            initial_prediction,
            target_all,
            mobility_all,
            target_scale_squared=scale,
            config=objective,
        )
    history = [{"update": 0, "normalized_loss": float(initial.cpu())}]
    for update in range(1, int(updates) + 1):
        local = torch.randint(0, count, (int(batch_size),), generator=generator)
        inputs = inputs_all.index_select(local.to(inputs_all.later_full_state.device))
        target = target_all.index_select(0, local.to(target_all.device))
        _q, prediction, mobility = _prevalidated_q_and_m(model, inputs)
        normalized, _raw, _unweighted = path_weighted_raw_target_mse(
            prediction,
            target,
            mobility,
            target_scale_squared=scale,
            config=objective,
        )
        optimizer.zero_grad(set_to_none=True)
        normalized.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 100 == 0 or update == int(updates):
            with torch.no_grad():
                _q, full_prediction, _mobility = _prevalidated_q_and_m(
                    model, inputs_all
                )
                full, _, _ = path_weighted_raw_target_mse(
                    full_prediction,
                    target_all,
                    mobility_all,
                    target_scale_squared=scale,
                    config=objective,
                )
            history.append({"update": update, "normalized_loss": float(full.cpu())})
    final = float(history[-1]["normalized_loss"])
    baseline = float(history[0]["normalized_loss"])
    reduction = 1.0 - final / baseline
    report = {
        "schema": PATH_WEIGHTED_TRAINING_VERSION + "-memorization-gate",
        "rows": count,
        "updates": int(updates),
        "initial_normalized_loss": baseline,
        "final_normalized_loss": final,
        "relative_reduction": reduction,
        "required_reduction": float(required_reduction),
        "passed": int(math.isfinite(reduction) and reduction >= required_reduction),
        "history": history,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "report.json", report)
    _atomic_torch_save(
        destination / "model.pt",
        {
            "schema": PATH_WEIGHTED_TRAINING_VERSION + "-memorization-model",
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "report": report,
        },
    )
    return report


__all__ = [
    "PATH_WEIGHTED_TRAINING_VERSION",
    "CacheLossScales",
    "CapacityTrainingConfig",
    "PathWeightedTrainingError",
    "compute_cache_loss_scales",
    "evaluate_capacity_model",
    "load_selected_model",
    "make_capacity_model",
    "run_large_memorization_gate",
    "train_capacity_model",
]
