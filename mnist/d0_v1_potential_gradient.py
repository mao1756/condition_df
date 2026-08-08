"""Minimal D0-v1 potential-gradient reconstruction experiment."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .d0_dirichlet_score import physical_flux_from_potential
from .d0_score_density_ratio_head import (
    D0BoundarySmoothMeanHeadPotentialUNet,
    build_coordinate_conjugate_adamw,
)
from .d0_v0_density_ratio import (
    D0V0Cache,
    cache_indices_for_paths,
    dynamics_config,
    run_paired_sampling,
    split_paths,
    write_metrics_csv,
)
from .eulerian_flux_mnist import free_drift_flux_torch, flux_divergence_torch


@dataclass(frozen=True)
class D0V1Config:
    grid_size: int = 28
    alpha_eff: float = 1.0
    lambda_mix: float = 0.35
    label: int = 3
    sample_steps: int = 512
    reference_substeps: int = 256
    tau_eff: float = 5e-5
    mass_floor: float = 1e-7
    limiter_fraction: float = 1.0
    cache_paths: int = 64
    train_paths: int = 48
    time_slices_per_path: int = 16
    base_channels: int = 32
    train_steps: int = 4_000
    batch_size: int = 8
    accumulation_steps: int = 4
    body_lr: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.99
    validate_every: int = 250
    sample_count: int = 8
    seed: int = 260727

    @classmethod
    def smoke(cls) -> "D0V1Config":
        return replace(
            cls(),
            grid_size=4,
            sample_steps=2,
            reference_substeps=2,
            cache_paths=4,
            train_paths=3,
            time_slices_per_path=4,
            base_channels=4,
            train_steps=2,
            batch_size=4,
            accumulation_steps=1,
            validate_every=1,
            sample_count=1,
        )


def mixed_target(cache: D0V0Cache, config: D0V1Config) -> np.ndarray:
    uniform = np.full_like(cache.target, 1.0 / cache.target.size)
    return (
        (1.0 - config.lambda_mix) * cache.target + config.lambda_mix * uniform
    ).astype(np.float32)


def reverse_rates(
    tau: Tensor, cache: D0V0Cache, config: D0V1Config
) -> Tensor:
    outer_dt = cache.horizon / config.sample_steps
    forward_outer = torch.floor((cache.horizon - tau) / outer_dt).long()
    forward_outer.clamp_(0, len(cache.rate_schedule) - 1)
    schedule = torch.as_tensor(
        cache.rate_schedule, device=tau.device, dtype=tau.dtype
    )
    return schedule.index_select(0, forward_outer)


def time_quartiles(tau: np.ndarray, horizon: float) -> np.ndarray:
    return np.minimum((4.0 * np.asarray(tau) / horizon).astype(np.int64), 3)


def sample_stratified_batch(
    cache: D0V0Cache,
    indices: np.ndarray,
    batch_size: int,
    *,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if batch_size % 4:
        raise ValueError("batch_size must be divisible by four")
    quartiles = time_quartiles(cache.tau[indices], cache.horizon)
    per_quartile = batch_size // 4
    chosen = np.concatenate(
        [
            rng.choice(indices[quartiles == quartile], size=per_quartile, replace=True)
            for quartile in range(4)
        ]
    )
    rng.shuffle(chosen)
    return (
        torch.as_tensor(cache.states[chosen], device=device),
        torch.as_tensor(cache.tau[chosen], device=device),
        torch.as_tensor(cache.labels[chosen], dtype=torch.long, device=device),
    )


def conditioning_target_increment(
    states: Tensor,
    tau: Tensor,
    cache: D0V0Cache,
    config: D0V1Config,
) -> tuple[Tensor, Tensor]:
    outer_dt = cache.horizon / config.sample_steps
    endpoint = torch.as_tensor(
        mixed_target(cache, config), device=states.device, dtype=states.dtype
    )
    remaining = (cache.horizon - tau).clamp_min(outer_dt)
    contraction = (outer_dt / remaining).clamp_max(1.0)
    total_increment = contraction[:, None] * (endpoint[None, :] - states)
    rates = reverse_rates(tau, cache, config)
    free_edge_increment = (
        rates[:, None, None, None]
        * free_drift_flux_torch(states, dynamics_config(config))
        * outer_dt
    )
    free_increment = flux_divergence_torch(free_edge_increment).flatten(1)
    return total_increment - free_increment, rates


def predicted_conditioning_increment(
    model: torch.nn.Module,
    states: Tensor,
    tau: Tensor,
    labels: Tensor,
    rates: Tensor,
    cache: D0V0Cache,
    config: D0V1Config,
    *,
    create_graph: bool,
) -> Tensor:
    flux, _, _ = physical_flux_from_potential(
        model,
        tau,
        states,
        labels,
        dynamics_config(config),
        time_change=rates,
        create_graph=create_graph,
    )
    outer_dt = cache.horizon / config.sample_steps
    return flux_divergence_torch(flux * outer_dt).flatten(1)


def normalized_increment_loss(predicted: Tensor, target: Tensor) -> Tensor:
    scale = target.square().mean(dim=1).sqrt().detach().clamp_min(1e-8)
    return ((predicted - target) / scale[:, None]).square().mean()


def potential_gradient_loss(
    model: torch.nn.Module,
    states: Tensor,
    tau: Tensor,
    labels: Tensor,
    cache: D0V0Cache,
    config: D0V1Config,
    *,
    create_graph: bool,
) -> Tensor:
    target, rates = conditioning_target_increment(states, tau, cache, config)
    predicted = predicted_conditioning_increment(
        model,
        states,
        tau,
        labels,
        rates,
        cache,
        config,
        create_graph=create_graph,
    )
    return normalized_increment_loss(predicted, target)


def _ema_update(ema: dict[str, Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evaluate(
    model: torch.nn.Module,
    cache: D0V0Cache,
    indices: np.ndarray,
    config: D0V1Config,
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(indices), config.batch_size):
        selected = indices[start : start + config.batch_size]
        states = torch.as_tensor(cache.states[selected], device=device)
        tau = torch.as_tensor(cache.tau[selected], device=device)
        labels = torch.as_tensor(
            cache.labels[selected], dtype=torch.long, device=device
        )
        loss = potential_gradient_loss(
            model,
            states,
            tau,
            labels,
            cache,
            config,
            create_graph=False,
        )
        total += float(loss.detach()) * len(selected)
    return total / len(indices)


def train(
    cache: D0V0Cache,
    config: D0V1Config,
    run_dir: Path,
    *,
    device: torch.device,
    show_progress: bool,
) -> dict[str, Any]:
    train_paths, validation_paths = split_paths(config)
    train_indices = cache_indices_for_paths(cache, train_paths)
    validation_indices = cache_indices_for_paths(cache, validation_paths)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("whole-path split produced an empty cache partition")
    if set(time_quartiles(cache.tau[train_indices], cache.horizon)) != set(range(4)):
        raise ValueError("training cache does not cover all four reverse-time quartiles")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics_config(config), base_channels=config.base_channels
    ).to(device)
    optimizer = build_coordinate_conjugate_adamw(
        model,
        body_lr=config.body_lr,
        weight_decay=config.weight_decay,
    )
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    start_step = 0
    best_step = -1
    best_loss = math.inf
    latest_path = run_dir / "latest.pt"
    if latest_path.is_file():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        ema = {name: value.to(device) for name, value in payload["ema"].items()}
        history = list(payload["history"])
        validations = list(payload["validations"])
        start_step = int(payload["step"])
        best_step = int(payload["best_step"])
        best_loss = float(payload["best_loss"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), device)

    for step in range(start_step + 1, config.train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(config.accumulation_steps):
            states, tau, labels = sample_stratified_batch(
                cache,
                train_indices,
                config.batch_size,
                rng=rng,
                device=device,
            )
            loss = potential_gradient_loss(
                model,
                states,
                tau,
                labels,
                cache,
                config,
                create_graph=True,
            )
            (loss / config.accumulation_steps).backward()
            loss_sum += float(loss.detach())
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        )
        optimizer.step()
        _ema_update(ema, model, config.ema_decay)
        history.append(
            {
                "step": step,
                "train_gradient_loss": loss_sum / config.accumulation_steps,
                "grad_norm": grad_norm,
            }
        )

        if step % config.validate_every == 0 or step == config.train_steps:
            current = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema)
            validation_loss = evaluate(
                model,
                cache,
                validation_indices,
                config,
                device=device,
            )
            model.load_state_dict(current)
            validations.append(
                {"step": step, "validation_gradient_loss": validation_loss}
            )
            if math.isfinite(validation_loss) and validation_loss < best_loss:
                best_loss = validation_loss
                best_step = step
                _atomic_torch_save(
                    run_dir / "best_ema.pt",
                    {"step": step, "model": {k: v.cpu() for k, v in ema.items()}},
                )
            payload = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": {name: value.cpu() for name, value in ema.items()},
                "history": history,
                "validations": validations,
                "best_step": best_step,
                "best_loss": best_loss,
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
            }
            _atomic_torch_save(latest_path, payload)
            write_metrics_csv(run_dir / "training.csv", history)
            (run_dir / "validation.json").write_text(
                json.dumps(validations, indent=2), encoding="utf-8"
            )
            if show_progress:
                print(
                    f"step={step} "
                    f"train_gradient_loss={history[-1]['train_gradient_loss']:.6f} "
                    f"validation_gradient_loss={validation_loss:.6f}",
                    flush=True,
                )
    return {
        "best_step": best_step,
        "best_validation_gradient_loss": best_loss,
    }


def load_best_model(
    run_dir: Path, config: D0V1Config, device: torch.device
) -> D0BoundarySmoothMeanHeadPotentialUNet:
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics_config(config), base_channels=config.base_channels
    ).to(device)
    payload = torch.load(run_dir / "best_ema.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _correlation(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered_samples = samples - samples.mean(axis=1, keepdims=True)
    centered_target = target - target.mean()
    numerator = (centered_samples * centered_target[None, :]).sum(axis=1)
    denominator = np.sqrt(
        np.square(centered_samples).sum(axis=1)
        * float(np.square(centered_target).sum())
    )
    return numerator / np.maximum(denominator, 1e-30)


def run_v1_paired_sampling(
    model: torch.nn.Module,
    cache: D0V0Cache,
    config: D0V1Config,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    samples, _, safety = run_paired_sampling(model, cache, config, device=device)
    target_mixed = mixed_target(cache, config)
    terminal = samples["terminal_states"]
    reference = samples["samples_strength0"]
    learned = samples["samples_strength1"]

    corr_terminal = _correlation(terminal, target_mixed)
    corr_reference = _correlation(reference, target_mixed)
    corr_learned = _correlation(learned, target_mixed)
    l1_terminal = np.abs(terminal - target_mixed).sum(axis=1)
    l1_reference = np.abs(reference - target_mixed).sum(axis=1)
    l1_learned = np.abs(learned - target_mixed).sum(axis=1)
    raw_corr = _correlation(learned, cache.target)
    raw_l1 = np.abs(learned - cache.target).sum(axis=1)

    rows = [
        {
            "path_index": int(path),
            "corr_terminal_mixed": float(corr_terminal[index]),
            "corr_reference_mixed": float(corr_reference[index]),
            "corr_learned_mixed": float(corr_learned[index]),
            "corr_improvement_over_terminal": float(
                corr_learned[index] - corr_terminal[index]
            ),
            "l1_terminal_mixed": float(l1_terminal[index]),
            "l1_reference_mixed": float(l1_reference[index]),
            "l1_learned_mixed": float(l1_learned[index]),
            "relative_l1_reduction_from_terminal": float(
                (l1_terminal[index] - l1_learned[index])
                / max(l1_terminal[index], 1e-30)
            ),
            "corr_learned_raw": float(raw_corr[index]),
            "l1_learned_raw": float(raw_l1[index]),
        }
        for index, path in enumerate(samples["path_indices"])
    ]
    summary = {
        **{
            key: safety[key]
            for key in (
                "num_samples",
                "nonfinite_edges",
                "max_simplex_mass_error",
                "limiter_fraction",
                "mobility_weighted_limiter_fraction",
            )
        },
        "median_corr_improvement_over_terminal": float(
            np.median([row["corr_improvement_over_terminal"] for row in rows])
        ),
        "median_relative_l1_reduction_from_terminal": float(
            np.median(
                [row["relative_l1_reduction_from_terminal"] for row in rows]
            )
        ),
    }
    samples["mixed_target"] = target_mixed
    return samples, rows, summary


def save_contact_sheet(
    path: Path, samples: dict[str, np.ndarray], grid_size: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(samples["samples_strength0"])
    target = samples["target"].reshape(grid_size, grid_size)
    target_mixed = samples["mixed_target"].reshape(grid_size, grid_size)
    vmax = float(target_mixed.max())
    rows = (
        ("raw target", None),
        ("mixed target", None),
        ("terminal", "terminal_states"),
        ("reference", "samples_strength0"),
        ("learned", "samples_strength1"),
    )
    figure, axes = plt.subplots(
        len(rows), count, figsize=(1.7 * count, 1.7 * len(rows)), squeeze=False
    )
    for column in range(count):
        for row, (label, key) in enumerate(rows):
            image = (
                target
                if label == "raw target"
                else target_mixed
                if label == "mixed target"
                else samples[key][column].reshape(grid_size, grid_size)
            )
            axes[row, column].imshow(
                image, cmap="gray", vmin=0.0, vmax=vmax
            )
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].text(
                    -0.08,
                    0.5,
                    label,
                    transform=axes[row, column].transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    clip_on=False,
                )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def acceptance_summary(
    training: dict[str, Any], sampling: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "eight_samples": sampling["num_samples"] == 8,
        "finite": sampling["nonfinite_edges"] == 0,
        "simplex": sampling["max_simplex_mass_error"] <= 2e-6,
        "raw_limiter": sampling["limiter_fraction"] <= 0.10,
        "mobility_limiter": sampling["mobility_weighted_limiter_fraction"] <= 0.05,
        "correlation_over_terminal": (
            sampling["median_corr_improvement_over_terminal"] >= 0.10
        ),
        "relative_l1_over_terminal": (
            sampling["median_relative_l1_reduction_from_terminal"] >= 0.10
        ),
    }
    passed = all(checks.values())
    return {
        "status": "candidate_for_visual_review" if passed else "stop_after_first_run",
        "checks": checks,
        "training": training,
        "sampling": sampling,
        "next_action": "review_contact_sheet" if passed else "inspect_without_tuning",
    }


__all__ = [
    "D0V1Config",
    "acceptance_summary",
    "conditioning_target_increment",
    "evaluate",
    "load_best_model",
    "mixed_target",
    "normalized_increment_loss",
    "potential_gradient_loss",
    "predicted_conditioning_increment",
    "reverse_rates",
    "run_v1_paired_sampling",
    "sample_stratified_batch",
    "save_contact_sheet",
    "time_quartiles",
    "train",
]
