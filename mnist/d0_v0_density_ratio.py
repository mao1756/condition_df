"""Minimal D0-v0 density-ratio trainer and paired one-image sampler.

This module is intentionally a narrow bridge to a first generated image.  It
reuses the established D0 reference dynamics, but does not participate in the
later certification, replay, or estimator-selection frameworks.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .d0_dirichlet_score import physical_flux_from_potential
from .d0_score_density_ratio_head import (
    D0BoundarySmoothMeanHeadPotentialUNet,
    build_coordinate_conjugate_adamw,
)
from .eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon
from .experiment12_d0 import (
    D0TrainingCache,
    Experiment12D0Config,
    _direct_doob_reverse_substep,
    build_d0_training_cache,
    make_rate_schedule,
)


@dataclass(frozen=True)
class D0V0Config:
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
    positive_batch_size: int = 32
    accumulation_steps: int = 8
    body_lr: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    loss_scale: float = 0.05173607018770852
    ema_decay: float = 0.99
    validate_every: int = 250
    sample_count: int = 8
    seed: int = 260727

    @classmethod
    def smoke(cls) -> "D0V0Config":
        return replace(
            cls(),
            grid_size=4,
            sample_steps=2,
            reference_substeps=2,
            cache_paths=4,
            train_paths=3,
            time_slices_per_path=2,
            base_channels=4,
            train_steps=2,
            positive_batch_size=2,
            accumulation_steps=1,
            validate_every=1,
            sample_count=1,
        )


@dataclass(frozen=True)
class D0V0Cache:
    states: np.ndarray
    tau: np.ndarray
    labels: np.ndarray
    path_indices: np.ndarray
    terminal_states: np.ndarray
    terminal_labels: np.ndarray
    target: np.ndarray
    rate_schedule: np.ndarray
    horizon: float


def dynamics_config(config: D0V0Config) -> DirectFluxMNISTConfig:
    low_resolution = min(7, config.grid_size)
    return DirectFluxMNISTConfig(
        grid_size=config.grid_size,
        alpha_eff=config.alpha_eff,
        edge_alpha_mode="alpha_eff",
        limiter_fraction=config.limiter_fraction,
        mass_floor=config.mass_floor,
        source_lowfreq_size=low_resolution,
        ot_lowres_size=low_resolution,
    )


def experiment_config(config: D0V0Config) -> Experiment12D0Config:
    return Experiment12D0Config(
        cache_paths=config.cache_paths,
        cache_batch_size=min(config.cache_paths, 64),
        cache_build_mode="outer",
        cache_time_sampling="uniform",
        time_slices_per_path=config.time_slices_per_path,
        teacher_stride_substeps=config.reference_substeps,
        lambda_mix=config.lambda_mix,
        sample_steps=config.sample_steps,
        reference_substeps=config.reference_substeps,
        tau_eff=config.tau_eff,
        single_image_overfit=True,
        single_image_index=0,
        single_image_label=config.label,
        seed=config.seed,
        use_amp=False,
    )


def split_paths(config: D0V0Config) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(config.cache_paths)
    return (
        np.sort(order[: config.train_paths]),
        np.sort(order[config.train_paths :]),
    )


def cache_indices_for_paths(cache: D0V0Cache, paths: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(cache.path_indices, np.asarray(paths, dtype=np.int64)))


def _from_training_cache(cache: D0TrainingCache, target: np.ndarray) -> D0V0Cache:
    return D0V0Cache(
        states=cache.states.cpu().numpy().astype(np.float32),
        tau=cache.tau.cpu().numpy().astype(np.float32),
        labels=cache.labels.cpu().numpy().astype(np.int64),
        path_indices=cache.path_indices.cpu().numpy().astype(np.int64),
        terminal_states=np.asarray(cache.terminal_states, dtype=np.float32).reshape(
            len(cache.terminal_states), -1
        ),
        terminal_labels=np.asarray(cache.requested_labels, dtype=np.int64),
        target=np.asarray(target, dtype=np.float32).reshape(-1),
        rate_schedule=np.asarray(cache.rate_schedule, dtype=np.float64),
        horizon=float(cache.horizon),
    )


def build_cache(
    image: np.ndarray,
    config: D0V0Config,
    *,
    device: torch.device,
    show_progress: bool,
) -> D0V0Cache:
    target = np.asarray(image, dtype=np.float64).reshape(-1)
    target /= target.sum()
    cache = build_d0_training_cache(
        dataset_images=target[None, :],
        dataset_labels=np.asarray([config.label], dtype=np.int64),
        dynamics_config=dynamics_config(config),
        d0_config=experiment_config(config),
        device=device,
        rng=np.random.default_rng(config.seed),
        show_progress=show_progress,
    )
    return _from_training_cache(cache, target)


def build_smoke_cache(config: D0V0Config) -> D0V0Cache:
    rng = np.random.default_rng(config.seed)
    cells = config.grid_size**2
    target = np.arange(1, cells + 1, dtype=np.float64)
    target /= target.sum()
    states: list[np.ndarray] = []
    tau: list[float] = []
    labels: list[int] = []
    paths: list[int] = []
    horizon = natural_horizon(dynamics_config(config))
    for path in range(config.cache_paths):
        terminal = rng.dirichlet(np.ones(cells))
        for index in range(config.time_slices_per_path):
            fraction = (index + 1) / (config.time_slices_per_path + 1)
            state = (1.0 - fraction) * target + fraction * terminal
            states.append(state.astype(np.float32))
            tau.append(float(fraction * horizon))
            labels.append(config.label)
            paths.append(path)
    terminal_states = rng.dirichlet(
        np.ones(cells), size=config.cache_paths
    ).astype(np.float32)
    rates = make_rate_schedule(
        config.sample_steps,
        tau_eff=config.tau_eff,
        horizon=horizon,
    )
    return D0V0Cache(
        states=np.asarray(states, dtype=np.float32),
        tau=np.asarray(tau, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        path_indices=np.asarray(paths, dtype=np.int64),
        terminal_states=terminal_states,
        terminal_labels=np.full(config.cache_paths, config.label, dtype=np.int64),
        target=target.astype(np.float32),
        rate_schedule=rates,
        horizon=horizon,
    )


def save_cache(path: Path, cache: D0V0Cache) -> None:
    np.savez_compressed(
        path,
        states=cache.states,
        tau=cache.tau,
        labels=cache.labels,
        path_indices=cache.path_indices,
        terminal_states=cache.terminal_states,
        terminal_labels=cache.terminal_labels,
        target=cache.target,
        rate_schedule=cache.rate_schedule,
        horizon=np.asarray(cache.horizon),
    )


def load_cache(path: Path) -> D0V0Cache:
    with np.load(path) as values:
        return D0V0Cache(
            states=values["states"],
            tau=values["tau"],
            labels=values["labels"],
            path_indices=values["path_indices"],
            terminal_states=values["terminal_states"],
            terminal_labels=values["terminal_labels"],
            target=values["target"],
            rate_schedule=values["rate_schedule"],
            horizon=float(values["horizon"]),
        )


def sample_balanced_batch(
    cache: D0V0Cache,
    indices: np.ndarray,
    batch_size: int,
    *,
    alpha_eff: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    chosen = rng.choice(indices, size=batch_size, replace=True)
    positive = torch.as_tensor(cache.states[chosen], device=device)
    tau = torch.as_tensor(cache.tau[chosen], device=device)
    labels = torch.as_tensor(cache.labels[chosen], dtype=torch.long, device=device)
    concentration = torch.full_like(positive, float(alpha_eff))
    negative = torch.distributions.Dirichlet(concentration).sample()
    states = torch.cat([positive, negative], dim=0)
    matched_tau = torch.cat([tau, tau], dim=0)
    matched_labels = torch.cat([labels, labels], dim=0)
    targets = torch.cat(
        [torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)]
    )
    return states, matched_tau, matched_labels, targets


def _ema_update(ema: dict[str, Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _evaluate(
    model: torch.nn.Module,
    cache: D0V0Cache,
    indices: np.ndarray,
    config: D0V0Config,
    *,
    device: torch.device,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        states, tau, labels, targets = sample_balanced_batch(
            cache,
            indices,
            max(32, config.positive_batch_size),
            alpha_eff=config.alpha_eff,
            rng=rng,
            device=device,
        )
        return float(F.binary_cross_entropy_with_logits(model(tau, states, labels), targets))


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_metrics_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def train(
    cache: D0V0Cache,
    config: D0V0Config,
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
    best_bce = math.inf
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
        best_bce = float(payload["best_bce"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), device)

    for step in range(start_step + 1, config.train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(config.accumulation_steps):
            states, tau, labels, targets = sample_balanced_batch(
                cache,
                train_indices,
                config.positive_batch_size,
                alpha_eff=config.alpha_eff,
                rng=rng,
                device=device,
            )
            raw_loss = F.binary_cross_entropy_with_logits(
                model(tau, states, labels), targets
            )
            (
                raw_loss
                * config.loss_scale
                / float(config.accumulation_steps)
            ).backward()
            loss_sum += float(raw_loss.detach())
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        )
        optimizer.step()
        _ema_update(ema, model, config.ema_decay)
        history.append(
            {
                "step": step,
                "train_bce": loss_sum / config.accumulation_steps,
                "grad_norm": grad_norm,
            }
        )

        if step % config.validate_every == 0 or step == config.train_steps:
            current = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema)
            validation_bce = _evaluate(
                model,
                cache,
                validation_indices,
                config,
                device=device,
                seed=config.seed + step,
            )
            model.load_state_dict(current)
            validations.append(
                {
                    "step": step,
                    "validation_bce": validation_bce,
                    "zero_baseline_bce": math.log(2.0),
                }
            )
            if math.isfinite(validation_bce) and validation_bce < best_bce:
                best_bce = validation_bce
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
                "best_bce": best_bce,
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
                    f"step={step} train_bce={history[-1]['train_bce']:.6f} "
                    f"validation_bce={validation_bce:.6f}",
                    flush=True,
                )
    return {
        "best_step": best_step,
        "best_validation_bce": best_bce,
        "zero_baseline_bce": math.log(2.0),
    }


def load_best_model(
    run_dir: Path, config: D0V0Config, device: torch.device
) -> D0BoundarySmoothMeanHeadPotentialUNet:
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics_config(config), base_channels=config.base_channels
    ).to(device)
    payload = torch.load(run_dir / "best_ema.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def density_ratio_flux_delta(
    model: torch.nn.Module,
    tau: Tensor,
    states: Tensor,
    labels: Tensor,
    config: D0V0Config,
    *,
    rate: float,
    dt: float,
) -> Tensor:
    flux, _, _ = physical_flux_from_potential(
        model,
        tau,
        states,
        labels,
        dynamics_config(config),
        time_change=rate,
    )
    return flux.detach() * float(dt)


def _correlation(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered_samples = samples - samples.mean(axis=1, keepdims=True)
    centered_target = target - target.mean()
    numerator = (centered_samples * centered_target[None, :]).sum(axis=1)
    denominator = np.sqrt(
        (centered_samples**2).sum(axis=1) * (centered_target**2).sum()
    )
    return numerator / np.maximum(denominator, 1e-30)


def run_paired_sampling(
    model: torch.nn.Module,
    cache: D0V0Cache,
    config: D0V0Config,
    *,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    _, validation_paths = split_paths(config)
    selected = validation_paths[: min(config.sample_count, len(validation_paths))]
    initial = torch.as_tensor(cache.terminal_states[selected], device=device).reshape(
        len(selected), -1
    )
    labels = torch.as_tensor(
        cache.terminal_labels[selected], dtype=torch.long, device=device
    )
    total_substeps = config.sample_steps * config.reference_substeps
    dt = cache.horizon / total_substeps
    arms: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for strength in (0.0, 1.0):
        states = initial.clone()
        shared_noise = torch.Generator(device=device)
        shared_noise.manual_seed(config.seed + 1)
        totals = {
            "limited_edges": torch.zeros((), dtype=torch.float64, device=device),
            "proposed_edges": torch.zeros((), dtype=torch.float64, device=device),
            "limited_mobility_weight_sum": torch.zeros(
                (), dtype=torch.float64, device=device
            ),
            "mobility_weight_sum": torch.zeros(
                (), dtype=torch.float64, device=device
            ),
            "nonfinite_edges": torch.zeros((), dtype=torch.float64, device=device),
            "max_simplex_mass_error": torch.zeros(
                (), dtype=torch.float64, device=device
            ),
        }
        for q in range(total_substeps - 1, 0, -1):
            outer = q // config.reference_substeps
            rate = float(cache.rate_schedule[min(outer, config.sample_steps - 1)])
            standard_normal = torch.randn(
                (len(selected), 2, config.grid_size, config.grid_size),
                device=device,
                generator=shared_noise,
            )
            if strength == 0.0:
                learned_delta = torch.zeros(
                    (len(selected), 2, config.grid_size, config.grid_size),
                    device=device,
                    dtype=states.dtype,
                )
            else:
                tau_value = max(cache.horizon - (q + 1) * dt, dt)
                tau = torch.full(
                    (len(selected),), tau_value, device=device, dtype=states.dtype
                )
                learned_delta = strength * density_ratio_flux_delta(
                    model,
                    tau,
                    states,
                    labels,
                    config,
                    rate=rate,
                    dt=dt,
                )
            result = _direct_doob_reverse_substep(
                states,
                learned_delta,
                rate=rate,
                dt=dt,
                dynamics_config=dynamics_config(config),
                standard_normal=standard_normal,
                diagnostics_device=True,
            )
            states = result.states.detach()
            for key in totals:
                value = torch.as_tensor(
                    result.diagnostics[key], dtype=torch.float64, device=device
                )
                if key == "max_simplex_mass_error":
                    totals[key] = torch.maximum(totals[key], value)
                else:
                    totals[key].add_(value)
        arms.append(states.cpu().numpy())
        diagnostics.append({key: float(value.cpu()) for key, value in totals.items()})

    samples0, samples1 = arms
    corr0, corr1 = _correlation(samples0, cache.target), _correlation(
        samples1, cache.target
    )
    l10 = np.abs(samples0 - cache.target).sum(axis=1)
    l11 = np.abs(samples1 - cache.target).sum(axis=1)
    rows = [
        {
            "path_index": int(path),
            "corr_strength0": float(corr0[index]),
            "corr_strength1": float(corr1[index]),
            "corr_improvement": float(corr1[index] - corr0[index]),
            "l1_strength0": float(l10[index]),
            "l1_strength1": float(l11[index]),
            "relative_l1_reduction": float((l10[index] - l11[index]) / max(l10[index], 1e-30)),
        }
        for index, path in enumerate(selected)
    ]
    learned = diagnostics[1]
    summary = {
        "num_samples": len(selected),
        "nonfinite_edges": int(learned["nonfinite_edges"]),
        "max_simplex_mass_error": learned["max_simplex_mass_error"],
        "limiter_fraction": learned["limited_edges"]
        / max(learned["proposed_edges"], 1.0),
        "mobility_weighted_limiter_fraction": learned[
            "limited_mobility_weight_sum"
        ]
        / max(learned["mobility_weight_sum"], 1e-30),
        "median_corr_improvement": float(
            np.median([row["corr_improvement"] for row in rows])
        ),
        "median_relative_l1_reduction": float(
            np.median([row["relative_l1_reduction"] for row in rows])
        ),
    }
    return (
        {
            "path_indices": selected,
            "terminal_states": initial.cpu().numpy(),
            "samples_strength0": samples0,
            "samples_strength1": samples1,
            "target": cache.target,
        },
        rows,
        summary,
    )


def save_contact_sheet(path: Path, samples: dict[str, np.ndarray], grid_size: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(samples["samples_strength0"])
    figure, axes = plt.subplots(3, count, figsize=(1.7 * count, 5.1), squeeze=False)
    for column in range(count):
        for row, key in enumerate(
            ("terminal_states", "samples_strength0", "samples_strength1")
        ):
            axes[row, column].imshow(
                samples[key][column].reshape(grid_size, grid_size), cmap="gray"
            )
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("terminal")
    axes[1, 0].set_ylabel("strength 0")
    axes[2, 0].set_ylabel("strength 1")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def acceptance_summary(
    training: dict[str, Any], sampling: dict[str, Any]
) -> dict[str, Any]:
    baseline = float(training["zero_baseline_bce"])
    validation_gain = (baseline - float(training["best_validation_bce"])) / baseline
    checks = {
        "eight_samples": sampling["num_samples"] == 8,
        "finite": sampling["nonfinite_edges"] == 0,
        "simplex": sampling["max_simplex_mass_error"] <= 2e-6,
        "raw_limiter": sampling["limiter_fraction"] <= 0.10,
        "mobility_limiter": sampling["mobility_weighted_limiter_fraction"] <= 0.05,
        "validation_bce": validation_gain >= 0.05,
        "correlation": sampling["median_corr_improvement"] >= 0.10,
        "relative_l1": sampling["median_relative_l1_reduction"] >= 0.10,
    }
    return {
        "status": "pass" if all(checks.values()) else "stop_after_first_run",
        "checks": checks,
        "validation_relative_improvement": validation_gain,
        "training": training,
        "sampling": sampling,
        "next_action": (
            "advance_to_a_broader_model"
            if all(checks.values())
            else "inspect_artifacts_without_tuning"
        ),
    }


__all__ = [
    "D0V0Cache",
    "D0V0Config",
    "acceptance_summary",
    "build_cache",
    "build_smoke_cache",
    "cache_indices_for_paths",
    "density_ratio_flux_delta",
    "dynamics_config",
    "experiment_config",
    "load_best_model",
    "load_cache",
    "run_paired_sampling",
    "sample_balanced_batch",
    "save_cache",
    "save_contact_sheet",
    "split_paths",
    "train",
    "write_metrics_csv",
]
