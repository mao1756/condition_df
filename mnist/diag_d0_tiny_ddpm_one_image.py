from __future__ import annotations

"""Train and sample a roughly 30k-parameter DDPM on one MNIST image."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .mnist_generation_benchmark import (
    MNIST_ARFF_SHA256,
    model_space_to_uint8,
    read_mnist_arff_slice,
    sha256_file,
    write_contact_sheet,
)
from .pixel_ddpm import (
    DDPMSchedule,
    epsilon_from_x0,
    epsilon_prediction_loss,
    make_linear_ddpm_schedule,
    q_sample,
    sample_reverse,
    update_ema_,
)
from .tiny_ddpm import TINY_DDPM_PARAMETER_COUNT, TinyClassConditionalUNet28


VERSION = "tiny-pixel-ddpm-one-image-v1"

PRODUCTION_CONFIG: dict[str, Any] = {
    "schema": VERSION,
    "research_mode": "exploratory",
    "decision": (
        "can a roughly 30k-parameter conventional pixel DDPM memorize one "
        "MNIST image and produce target-like complete samples"
    ),
    "source": {"dataset_index": 7, "label": 3, "class_index": 0},
    "model": {
        "name": "TinyClassConditionalUNet28",
        "parameter_count": TINY_DDPM_PARAMETER_COUNT,
        "channels": [8, 16, 8],
        "time_embedding": 32,
        "conditioning": 52,
    },
    "schedule": {"steps": 1000, "beta_start": 1e-4, "beta_end": 2e-2},
    "training": {
        "updates": 10_000,
        "batch_size": 128,
        "learning_rate": 2e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "gradient_norm_clip": 1.0,
        "ema_decay": 0.999,
        "validation_interval": 250,
        "validation_size": 1024,
    },
    "controls": {
        "horizons": [99, 499, 999],
        "paths_per_horizon": 4,
        "oracle_maximum_mse": 1e-6,
        "minimum_learned_beats_zero_count": 3,
        "minimum_median_relative_mse_improvement": 0.0,
    },
    "sampling": {
        "sample_count": 16,
        "anchors": [0, 250, 500, 750, 1000],
    },
    "diagnostics": {
        "minimum_median_correlation": 0.8,
        "minimum_beats_zero_count": 12,
        "minimum_median_relative_mse_improvement": 0.0,
    },
    "seeds": {
        "model": 0x30_0001,
        "training": 0x30_0002,
        "validation": 0x30_0003,
        "control_forward": 0x30_0004,
        "control_reverse": 0x30_0005,
        "prior_start": 0x30_0006,
        "prior_reverse": 0x30_0007,
    },
}


class TinyDDPMRunError(RuntimeError):
    pass


class ZeroEpsilon(nn.Module):
    def forward(self, images: Tensor, _: Tensor, __: Tensor) -> Tensor:
        return torch.zeros_like(images)


class SourceEpsilonOracle(nn.Module):
    def __init__(self, source: Tensor, schedule: DDPMSchedule) -> None:
        super().__init__()
        self.register_buffer("source", source.detach().clone())
        self.schedule = schedule

    def forward(self, images: Tensor, timesteps: Tensor, _: Tensor) -> Tensor:
        source = self.source.expand(images.shape[0], -1, -1, -1)
        return epsilon_from_x0(images, source, timesteps, self.schedule)


def _smoke_config() -> dict[str, Any]:
    config = copy.deepcopy(PRODUCTION_CONFIG)
    config["research_mode"] = "engineering-control"
    config["decision"] = (
        "does the tiny-DDPM training, control, sampling, and artifact pipeline "
        "execute correctly on CPU"
    )
    config["schedule"] = {"steps": 8, "beta_start": 1e-4, "beta_end": 2e-2}
    config["training"].update(
        {
            "updates": 3,
            "batch_size": 4,
            "validation_interval": 1,
            "validation_size": 8,
        }
    )
    config["controls"] = {
        "horizons": [3, 7],
        "paths_per_horizon": 2,
        "oracle_maximum_mse": 1e-6,
        "minimum_learned_beats_zero_count": 1,
        "minimum_median_relative_mse_improvement": 0.0,
    }
    config["sampling"] = {"sample_count": 2, "anchors": [0, 4, 8]}
    config["diagnostics"]["minimum_beats_zero_count"] = 1
    config["smoke"] = True
    return config


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_revision(repository_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_hashes(repository_root: Path) -> dict[str, str]:
    paths = (
        "mnist/diag_d0_tiny_ddpm_one_image.py",
        "mnist/tiny_ddpm.py",
        "mnist/pixel_ddpm.py",
        "mnist/mnist_generation_benchmark.py",
    )
    return {relative: _sha256_file(repository_root / relative) for relative in paths}


def _reproduction_command(args: argparse.Namespace) -> str:
    parts = [
        sys.executable,
        "-B",
        "-m",
        "mnist.diag_d0_tiny_ddpm_one_image",
        "--device",
        str(args.device),
        "--data-root",
        str(args.data_root),
        "--runs-root",
        str(args.runs_root),
        "--run-name",
        str(args.run_name),
    ]
    if args.smoke:
        parts.append("--smoke")
    if args.no_progress:
        parts.append("--no-progress")
    return subprocess.list2cmdline(parts)


def _environment(repository_root: Path, device: str) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "os": platform.platform(),
        "device": device,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_devices": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else []
        ),
        "git_revision": _git_revision(repository_root),
        "source_hashes": _source_hashes(repository_root),
    }


def _new_run_dir(runs_root: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = runs_root / f"{stamp}_{run_name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _synthetic_three() -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.uint8)
    image[4:7, 7:21] = 255
    image[12:15, 10:21] = 255
    image[21:24, 7:21] = 255
    image[6:22, 18:21] = 255
    return image


def _load_source(data_root: Path, smoke: bool) -> tuple[np.ndarray, dict[str, Any]]:
    if smoke:
        image = _synthetic_three()
        return image, {
            "role": "synthetic-smoke-only",
            "dataset_index": -1,
            "class_index": 0,
            "label": 3,
            "data_sha256": None,
        }

    arff_path = data_root if data_root.suffix.lower() == ".arff" else data_root / "mnist_784.arff"
    if not arff_path.is_file():
        raise FileNotFoundError(f"MNIST ARFF file not found: {arff_path}")
    actual_hash = sha256_file(arff_path)
    if actual_hash != MNIST_ARFF_SHA256:
        raise TinyDDPMRunError(
            f"MNIST hash mismatch: expected {MNIST_ARFF_SHA256}, got {actual_hash}"
        )
    images, labels = read_mnist_arff_slice(arff_path, 0, 8)
    choices = np.flatnonzero(labels == 3)
    if choices.tolist() != [7]:
        raise TinyDDPMRunError("the frozen first label-3 MNIST source changed")
    return images[7], {
        "role": "one-image-training-development-and-sampling-target",
        "dataset_index": 7,
        "class_index": 0,
        "label": 3,
        "data_sha256": actual_hash,
    }


def _model_space(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32)[None, None] / np.float32(127.5)) - np.float32(1.0)


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def _validation_mse(
    model: nn.Module,
    source: Tensor,
    label: int,
    schedule: DDPMSchedule,
    timesteps: Tensor,
    noise: Tensor,
    *,
    batch_size: int,
) -> float:
    model.eval()
    squared_sum = 0.0
    count = 0
    for start in range(0, timesteps.shape[0], batch_size):
        stop = min(start + batch_size, timesteps.shape[0])
        t_batch = timesteps[start:stop].to(source.device)
        noise_batch = noise[start:stop].to(source.device)
        x_0 = source.expand(stop - start, -1, -1, -1)
        labels = torch.full((stop - start,), label, device=source.device, dtype=torch.long)
        x_t = q_sample(x_0, t_batch, noise_batch, schedule)
        prediction = model(x_t, t_batch, labels)
        if not bool(torch.isfinite(prediction).all()):
            raise TinyDDPMRunError("validation prediction became nonfinite")
        squared_sum += float((prediction - noise_batch).square().sum().cpu())
        count += prediction.numel()
    model.train()
    return squared_sum / count


def _train(
    run_dir: Path,
    source: Tensor,
    label: int,
    schedule: DDPMSchedule,
    config: Mapping[str, Any],
    *,
    show_progress: bool,
) -> tuple[TinyClassConditionalUNet28, dict[str, Any]]:
    training = config["training"]
    seeds = config["seeds"]
    torch.manual_seed(int(seeds["model"]))
    if source.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seeds["model"]))
    model = TinyClassConditionalUNet28().to(source.device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )

    validation_generator = torch.Generator().manual_seed(int(seeds["validation"]))
    validation_size = int(training["validation_size"])
    validation_timesteps = torch.randint(
        0,
        schedule.num_steps,
        (validation_size,),
        generator=validation_generator,
        dtype=torch.long,
    )
    validation_noise = torch.randn(
        (validation_size, 1, 28, 28), generator=validation_generator
    )
    _write_npz(
        run_dir / "training/validation_bank.npz",
        timesteps=validation_timesteps.numpy(),
        noise=validation_noise.numpy(),
    )

    zero_validation_mse = float(validation_noise.square().mean())
    initial_validation_mse = _validation_mse(
        ema,
        source,
        label,
        schedule,
        validation_timesteps,
        validation_noise,
        batch_size=int(training["batch_size"]),
    )
    best_mse = float("inf")
    best_update = -1
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    train_generator = _generator(source.device, int(seeds["training"]))
    updates = int(training["updates"])
    batch_size = int(training["batch_size"])
    interval = int(training["validation_interval"])
    started = time.perf_counter()

    model.train()
    for update in range(1, updates + 1):
        timesteps = torch.randint(
            0,
            schedule.num_steps,
            (batch_size,),
            generator=train_generator,
            device=source.device,
        )
        noise = torch.randn(
            (batch_size, 1, 28, 28),
            generator=train_generator,
            device=source.device,
        )
        labels = torch.full(
            (batch_size,), label, device=source.device, dtype=torch.long
        )
        x_0 = source.expand(batch_size, -1, -1, -1)
        optimizer.zero_grad(set_to_none=True)
        loss = epsilon_prediction_loss(model, x_0, timesteps, labels, noise, schedule)
        if not bool(torch.isfinite(loss)):
            raise TinyDDPMRunError("training loss became nonfinite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_norm_clip"])
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise TinyDDPMRunError("training gradient became nonfinite")
        optimizer.step()
        update_ema_(ema, model, float(training["ema_decay"]))

        if update % interval != 0 and update != updates:
            continue
        validation_mse = _validation_mse(
            ema,
            source,
            label,
            schedule,
            validation_timesteps,
            validation_noise,
            batch_size=batch_size,
        )
        row = {
            "update": update,
            "train_epsilon_mse": float(loss.detach().cpu()),
            "ema_validation_epsilon_mse": validation_mse,
            "zero_validation_epsilon_mse": zero_validation_mse,
            "preclip_gradient_norm": float(torch.as_tensor(gradient_norm).cpu()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_update = update
            best_state = _cpu_state(ema)
        if show_progress:
            print(
                f"update={update}/{updates} train={row['train_epsilon_mse']:.6g} "
                f"validation={validation_mse:.6g}",
                flush=True,
            )

    if best_state is None or best_update <= 0:
        raise TinyDDPMRunError("no finite nonzero checkpoint was selected")
    _write_csv(run_dir / "training/history.csv", history)
    best_checkpoint = run_dir / "training/best.pt"
    torch.save(
        {
            "schema": VERSION + "-selected-checkpoint",
            "update": best_update,
            "validation_mse": best_mse,
            "parameter_count": TINY_DDPM_PARAMETER_COUNT,
            "model_state_dict": best_state,
        },
        best_checkpoint,
    )
    final_checkpoint = run_dir / "training/final.pt"
    torch.save(
        {
            "schema": VERSION + "-final-checkpoint",
            "update": updates,
            "model_state_dict": _cpu_state(model),
            "ema_state_dict": _cpu_state(ema),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_checkpoint,
    )
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "selected_update": best_update,
        "selected_validation_epsilon_mse": best_mse,
        "initial_validation_epsilon_mse": initial_validation_mse,
        "zero_validation_epsilon_mse": zero_validation_mse,
        "updates": updates,
        "elapsed_seconds": time.perf_counter() - started,
        "selected_checkpoint": "training/best.pt",
        "selected_checkpoint_sha256": _sha256_file(best_checkpoint),
        "final_checkpoint": "training/final.pt",
        "final_checkpoint_sha256": _sha256_file(final_checkpoint),
    }


@torch.no_grad()
def _reconstruction_panel(
    source: Tensor,
    label: int,
    schedule: DDPMSchedule,
    config: Mapping[str, Any],
    learned: nn.Module | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    controls = config["controls"]
    seeds = config["seeds"]
    path_count = int(controls["paths_per_horizon"])
    models: dict[str, nn.Module] = {
        "zero": ZeroEpsilon().to(source.device),
        "oracle": SourceEpsilonOracle(source, schedule).to(source.device),
    }
    if learned is not None:
        models["learned"] = learned

    records: list[dict[str, Any]] = []
    stored: dict[str, list[np.ndarray]] = {
        "initial": [],
        **{name: [] for name in models},
    }
    target = source.expand(path_count, -1, -1, -1)
    labels = torch.full((path_count,), label, device=source.device, dtype=torch.long)
    for horizon in controls["horizons"]:
        horizon = int(horizon)
        forward_noise = torch.randn(
            target.shape,
            generator=_generator(
                source.device, int(seeds["control_forward"]) + horizon
            ),
            device=source.device,
        )
        t_batch = torch.full(
            (path_count,), horizon, device=source.device, dtype=torch.long
        )
        initial = q_sample(target, t_batch, forward_noise, schedule)
        stored["initial"].append(initial.cpu().numpy())
        zero_mse: np.ndarray | None = None
        for name, model in models.items():
            final, _ = sample_reverse(
                model,
                labels,
                initial,
                schedule,
                generator=_generator(
                    source.device, int(seeds["control_reverse"]) + horizon
                ),
                start_t=horizon,
            )
            array = final.cpu().numpy()
            stored[name].append(array)
            mse = np.mean((array - target.cpu().numpy()) ** 2, axis=(1, 2, 3))
            if name == "zero":
                zero_mse = mse
            for path_index, value in enumerate(mse):
                records.append(
                    {
                        "horizon": horizon,
                        "path_index": path_index,
                        "model": name,
                        "endpoint_mse": float(value),
                        "relative_mse_improvement_over_zero": (
                            None
                            if name == "zero" or zero_mse is None
                            else float(1.0 - value / max(zero_mse[path_index], 1e-30))
                        ),
                    }
                )
    return {name: np.stack(values) for name, values in stored.items()}, records


def _endpoint_statistics(array: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.asarray(array, dtype=np.float64).reshape(array.shape[0], -1)
    target_flat = np.broadcast_to(target, array.shape).reshape(array.shape[0], -1).astype(np.float64)
    finite = np.isfinite(flat).all(axis=1)
    mse = np.full(array.shape[0], np.nan)
    l1 = np.full(array.shape[0], np.nan)
    correlation = np.full(array.shape[0], np.nan)
    for index in np.flatnonzero(finite):
        mse[index] = np.mean((flat[index] - target_flat[index]) ** 2)
        l1[index] = np.mean(np.abs(flat[index] - target_flat[index]))
        correlation[index] = np.corrcoef(flat[index], target_flat[index])[0, 1]
    return {"finite": finite, "mse": mse, "l1": l1, "correlation": correlation}


def _reconstruction_summary(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    controls = config["controls"]
    by_horizon: dict[str, Any] = {}
    for horizon in controls["horizons"]:
        learned = [
            row
            for row in rows
            if int(row["horizon"]) == int(horizon) and row["model"] == "learned"
        ]
        improvements = np.asarray(
            [row["relative_mse_improvement_over_zero"] for row in learned],
            dtype=np.float64,
        )
        mse = np.asarray([row["endpoint_mse"] for row in learned], dtype=np.float64)
        finite = np.isfinite(improvements) & np.isfinite(mse)
        beats = int(np.sum(finite & (improvements > 0.0)))
        median_improvement = (
            float(np.median(improvements[finite])) if np.any(finite) else None
        )
        passed = int(
            int(np.sum(finite)) == len(learned)
            and beats >= int(controls["minimum_learned_beats_zero_count"])
            and median_improvement is not None
            and median_improvement
            > float(controls["minimum_median_relative_mse_improvement"])
        )
        by_horizon[str(int(horizon))] = {
            "path_count": len(learned),
            "finite_count": int(np.sum(finite)),
            "learned_beats_zero_count": beats,
            "median_learned_endpoint_mse": (
                float(np.median(mse[finite])) if np.any(finite) else None
            ),
            "median_learned_relative_mse_improvement": median_improvement,
            "diagnostic_pass": passed,
        }
    complete_horizon = str(max(int(value) for value in controls["horizons"]))
    short_horizons = [
        str(int(value))
        for value in controls["horizons"]
        if int(value) != int(complete_horizon)
    ]
    passing_horizons = [
        int(horizon)
        for horizon, summary in by_horizon.items()
        if summary["diagnostic_pass"]
    ]
    return {
        "by_horizon": by_horizon,
        "complete_forward_terminal_horizon": int(complete_horizon),
        "complete_forward_terminal_reconstruction_pass": by_horizon[complete_horizon][
            "diagnostic_pass"
        ],
        "any_short_horizon_reconstruction_pass": int(
            any(by_horizon[horizon]["diagnostic_pass"] for horizon in short_horizons)
        ),
        "longest_passing_horizon": max(passing_horizons) if passing_horizons else None,
    }


@torch.no_grad()
def _prior_panel(
    model: nn.Module,
    source: Tensor,
    label: int,
    schedule: DDPMSchedule,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    sampling = config["sampling"]
    seeds = config["seeds"]
    count = int(sampling["sample_count"])
    labels = torch.full((count,), label, device=source.device, dtype=torch.long)
    initial = torch.randn(
        (count, 1, 28, 28),
        generator=_generator(source.device, int(seeds["prior_start"])),
        device=source.device,
    )
    models = {
        "zero": ZeroEpsilon().to(source.device),
        "oracle": SourceEpsilonOracle(source, schedule).to(source.device),
        "learned": model,
    }
    finals: dict[str, np.ndarray] = {}
    anchors: dict[str, np.ndarray] = {}
    for name, active in models.items():
        final, retained = sample_reverse(
            active,
            labels,
            initial,
            schedule,
            generator=_generator(source.device, int(seeds["prior_reverse"])),
            anchor_steps=tuple(int(value) for value in sampling["anchors"]),
        )
        finals[name] = final.cpu().numpy()
        anchors[name] = np.stack(
            [retained[int(value)].cpu().numpy() for value in sampling["anchors"]]
        )

    target = source.cpu().numpy()
    statistics = {
        name: _endpoint_statistics(array, target) for name, array in finals.items()
    }
    rows: list[dict[str, Any]] = []
    for index in range(count):
        zero_mse = statistics["zero"]["mse"][index]
        for name in models:
            values = statistics[name]
            finite = bool(values["finite"][index])
            rows.append(
                {
                    "path_index": index,
                    "model": name,
                    "finite": int(finite),
                    "endpoint_mse": float(values["mse"][index]) if finite else None,
                    "endpoint_l1": float(values["l1"][index]) if finite else None,
                    "endpoint_correlation": (
                        float(values["correlation"][index]) if finite else None
                    ),
                    "relative_mse_improvement_over_zero": (
                        None
                        if name == "zero" or not finite or not np.isfinite(zero_mse)
                        else float(
                            1.0
                            - values["mse"][index] / max(float(zero_mse), 1e-30)
                        )
                    ),
                }
            )

    learned = statistics["learned"]
    improvements = 1.0 - learned["mse"] / np.maximum(statistics["zero"]["mse"], 1e-30)
    mse_valid = (
        learned["finite"]
        & statistics["zero"]["finite"]
        & np.isfinite(improvements)
    )
    correlation_valid = learned["finite"] & np.isfinite(learned["correlation"])
    diagnostics = config["diagnostics"]
    summary = {
        "sample_count": count,
        "finite_learned_count": int(np.sum(learned["finite"])),
        "learned_beats_zero_count": int(
            np.sum(mse_valid & (learned["mse"] < statistics["zero"]["mse"]))
        ),
        "median_learned_relative_mse_improvement": (
            float(np.median(improvements[mse_valid])) if np.any(mse_valid) else None
        ),
        "median_learned_correlation": (
            float(np.median(learned["correlation"][correlation_valid]))
            if np.any(correlation_valid)
            else None
        ),
        "learned_saturated_pixel_fraction": float(
            np.mean(np.abs(np.nan_to_num(finals["learned"], nan=0.0)) >= 1.0)
        ),
        "finite_oracle_count": int(np.sum(statistics["oracle"]["finite"])),
        "oracle_saved_states_all_finite": int(
            np.isfinite(initial.cpu().numpy()).all()
            and np.isfinite(finals["oracle"]).all()
            and np.isfinite(anchors["oracle"]).all()
        ),
        "maximum_oracle_endpoint_mse": (
            float(np.max(statistics["oracle"]["mse"]))
            if bool(np.isfinite(statistics["oracle"]["mse"]).all())
            else None
        ),
    }
    summary["prior_oracle_integrity_pass"] = int(
        summary["finite_oracle_count"] == count
        and summary["oracle_saved_states_all_finite"]
        and summary["maximum_oracle_endpoint_mse"] is not None
        and summary["maximum_oracle_endpoint_mse"]
        <= float(config["controls"]["oracle_maximum_mse"])
    )
    summary["learner_diagnostic_pass"] = int(
        summary["finite_learned_count"] == count
        and summary["learned_beats_zero_count"]
        >= int(diagnostics["minimum_beats_zero_count"])
        and summary["median_learned_relative_mse_improvement"] is not None
        and summary["median_learned_relative_mse_improvement"]
        > float(diagnostics["minimum_median_relative_mse_improvement"])
        and summary["median_learned_correlation"] is not None
        and summary["median_learned_correlation"]
        >= float(diagnostics["minimum_median_correlation"])
    )
    summary["diagnostic_pass"] = int(
        summary["prior_oracle_integrity_pass"]
        and summary["learner_diagnostic_pass"]
    )
    arrays = {"initial": initial.cpu().numpy()}
    arrays.update({f"{name}_final": value for name, value in finals.items()})
    arrays.update({f"{name}_anchors": value for name, value in anchors.items()})
    arrays["anchor_steps"] = np.asarray(sampling["anchors"], dtype=np.int64)
    return arrays, rows, summary


def _safe_uint8(value: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    return model_space_to_uint8(clean)


def _render_rows(
    path: Path,
    source_uint8: np.ndarray,
    rows: Mapping[str, np.ndarray],
) -> None:
    count = next(iter(rows.values())).shape[0]
    images = [np.repeat(source_uint8[None], count, axis=0)]
    captions = [f"source {index}" for index in range(count)]
    for name, values in rows.items():
        images.append(_safe_uint8(values))
        captions.extend(f"{name} {index}" for index in range(count))
    write_contact_sheet(
        path,
        np.concatenate(images),
        columns=count,
        scale=3,
        captions=captions,
    )


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "artifact_manifest.json"
    files = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    rows = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema": VERSION + "-artifact-manifest",
        "artifact_count": len(rows),
        "artifact_bytes": sum(row["size"] for row in rows),
        "artifacts": rows,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _report(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    training: Mapping[str, Any],
    oracle: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    sampling: Mapping[str, Any],
    provenance: Mapping[str, Any],
    replay_command: str,
) -> str:
    integrity = bool(oracle["integrity_pass"])
    reconstruction_pass = bool(
        reconstruction["complete_forward_terminal_reconstruction_pass"]
    )
    sampling_pass = bool(sampling["learner_diagnostic_pass"])
    if not integrity:
        outcome = "ddpm_reverse_composition_integrity_failed"
        next_action = "Repair the sampler or schedule before changing the learner."
    elif config.get("smoke"):
        outcome = "engineering_smoke_complete"
        next_action = (
            "Run the frozen production exploratory command; the three-update "
            "synthetic smoke carries no model-capacity conclusion."
        )
    elif reconstruction_pass and sampling_pass:
        outcome = "tiny_one_image_ddpm_sampling_promising"
        next_action = "Freeze this run as a small conventional system control."
    elif reconstruction_pass:
        outcome = "forward_terminal_reconstruction_only"
        next_action = (
            "Inspect the saved independent-prior anchors, then change terminal-time "
            "coverage or parameterization before width."
        )
    elif sampling_pass:
        outcome = "prior_pass_reconstruction_anomaly"
        next_action = (
            "Audit the reconstruction panel and pairing before treating the prior "
            "result as evidence."
        )
    elif reconstruction["any_short_horizon_reconstruction_pass"]:
        outcome = "short_horizon_reconstruction_only"
        next_action = (
            "Localize accumulation and late-time error from the saved horizon panels, "
            "then change time coverage or parameterization before model width."
        )
    else:
        outcome = "tiny_one_image_ddpm_sampling_not_yet_promising"
        next_action = (
            "Compare one prespecified larger model or x0 parameterization; do not "
            "claim DDPM impossibility."
        )
    claim_boundary = (
        "This synthetic three-update smoke establishes only that the interfaces, "
        "controls, sampling path, and artifact writers execute. It says nothing "
        "about MNIST learning or model capacity."
        if config.get("smoke")
        else (
            "This result concerns one image, one model seed, one fixed schedule, "
            "and one exploratory diagnostic. It does not establish diversity, "
            "held-out-image generalization, a dataset-level generator, DDPM "
            "superiority, or an Eulerian result."
        )
    )
    return f"""# Tiny pixel-DDPM one-image run

Primary mode: {config['research_mode']}.

Decision: {config['decision']}.

Outcome: `{outcome}`. Sampling ran regardless of the diagnostic result; all zero,
oracle, learned, intermediate, and failed outputs are retained.

## Exact design

- Source: label `{source['label']}`, class index `{source['class_index']}`, dataset
  index `{source['dataset_index']}`, image SHA-256 `{source['image_sha256']}`.
- Model: `{config['model']}`.
- Schedule: `{config['schedule']}`.
- Training: `{config['training']}`.
- Loss: unweighted pixel-mean epsilon-prediction MSE.
- Selected update: `{training['selected_update']}`; selected fixed-bank validation
  MSE: `{training['selected_validation_epsilon_mse']}`; zero MSE:
  `{training['zero_validation_epsilon_mse']}`.
- Selected checkpoint: `{training['selected_checkpoint']}`, SHA-256
  `{training['selected_checkpoint_sha256']}`.
- Source revision: `{provenance['git_revision']}`. Exact source-file hashes and the
  software/accelerator environment are in `environment.json`.
- Replay from the repository root: `{replay_command}`. The same command is in
  `command.txt`; the complete evidence inventory and hashes are in
  `artifact_manifest.json`.

## Controls and objective result

The analytic epsilon oracle used the same DDPM posterior and reverse composition.
Maximum preflight oracle reconstruction MSE was
`{oracle['maximum_preflight_oracle_mse']}`; complete independent-prior oracle MSE
was `{oracle['maximum_prior_oracle_mse']}`; integrity gate pass:
`{oracle['integrity_pass']}`.

Learned forward-terminal reconstruction summary: `{dict(reconstruction)}`.

Complete independent-prior sampling summary: `{dict(sampling)}`.

Primary artifacts are `sampling/paired-final.png`, `sampling/prior_panel.npz`, the
trajectory contact sheets, `controls/reconstruction-panel-*.png`, and the raw control
arrays. Pixel-space clipping in PNGs does not replace the stored float arrays.

## Claim boundary and next action

{claim_boundary}

Required next action: {next_action}
"""


def run(args: argparse.Namespace) -> Path:
    config = _smoke_config() if args.smoke else copy.deepcopy(PRODUCTION_CONFIG)
    run_dir = _new_run_dir(Path(args.runs_root), str(args.run_name))
    repository_root = Path(__file__).resolve().parents[1]
    command = _reproduction_command(args)
    provenance = _environment(repository_root, str(args.device))
    _write_json(run_dir / "config.json", config)
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    _write_json(run_dir / "environment.json", provenance)
    _write_json(
        run_dir / "status.json",
        {"schema": VERSION + "-status", "state": "running", "error": None},
    )

    try:
        random.seed(int(config["seeds"]["model"]))
        np.random.seed(int(config["seeds"]["model"]) & 0xFFFFFFFF)
        torch.manual_seed(int(config["seeds"]["model"]))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise TinyDDPMRunError("CUDA was requested but is unavailable")

        source_uint8, source_record = _load_source(Path(args.data_root), bool(args.smoke))
        source_array = _model_space(source_uint8)
        source_record.update(
            {
                "image_sha256": _array_sha256(source_uint8),
                "model_space_sha256": _array_sha256(source_array),
            }
        )
        _write_json(run_dir / "source/source.json", source_record)
        _write_npz(
            run_dir / "source/source.npz",
            image_uint8=source_uint8,
            model_space=source_array,
            label=np.asarray([source_record["label"]], dtype=np.int64),
        )
        write_contact_sheet(
            run_dir / "source/source.png", source_uint8[None], columns=1, scale=6
        )

        source = torch.from_numpy(source_array).to(device)
        schedule_config = config["schedule"]
        schedule = make_linear_ddpm_schedule(
            int(schedule_config["steps"]),
            float(schedule_config["beta_start"]),
            float(schedule_config["beta_end"]),
            device=device,
        )
        label = int(source_record["label"])

        preflight_arrays, preflight_rows = _reconstruction_panel(
            source, label, schedule, config, None
        )
        _write_npz(run_dir / "controls/oracle_preflight.npz", **preflight_arrays)
        _write_csv(run_dir / "controls/oracle_preflight_metrics.csv", preflight_rows)
        oracle_values = np.asarray(
            [
                row["endpoint_mse"]
                for row in preflight_rows
                if row["model"] == "oracle"
            ],
            dtype=np.float64,
        )
        preflight_saved_states_all_finite = bool(
            oracle_values.size > 0
            and np.isfinite(oracle_values).all()
            and np.isfinite(preflight_arrays["initial"]).all()
            and np.isfinite(preflight_arrays["oracle"]).all()
        )
        maximum_preflight_oracle_mse = (
            float(np.max(oracle_values))
            if preflight_saved_states_all_finite
            else None
        )
        preflight_integrity_pass = int(
            maximum_preflight_oracle_mse is not None
            and maximum_preflight_oracle_mse
            <= float(config["controls"]["oracle_maximum_mse"])
        )
        oracle_summary = {
            "threshold": float(config["controls"]["oracle_maximum_mse"]),
            "maximum_preflight_oracle_mse": maximum_preflight_oracle_mse,
            "preflight_saved_states_all_finite": int(
                preflight_saved_states_all_finite
            ),
            "preflight_integrity_pass": preflight_integrity_pass,
            "maximum_prior_oracle_mse": None,
            "prior_saved_states_all_finite": None,
            "prior_integrity_pass": None,
            "integrity_pass": preflight_integrity_pass,
        }
        _write_json(run_dir / "controls/oracle_gate.json", oracle_summary)
        if not oracle_summary["preflight_integrity_pass"]:
            raise TinyDDPMRunError("analytic epsilon oracle integrity gate failed")

        model, training_summary = _train(
            run_dir,
            source,
            label,
            schedule,
            config,
            show_progress=not bool(args.no_progress),
        )
        _write_json(run_dir / "training/summary.json", training_summary)

        reconstruction_arrays, reconstruction_rows = _reconstruction_panel(
            source, label, schedule, config, model
        )
        _write_npz(
            run_dir / "controls/reconstruction_panel.npz", **reconstruction_arrays
        )
        _write_csv(
            run_dir / "controls/reconstruction_metrics.csv", reconstruction_rows
        )
        reconstruction_summary = _reconstruction_summary(
            reconstruction_rows, config
        )
        _write_json(
            run_dir / "controls/reconstruction_summary.json",
            reconstruction_summary,
        )
        for horizon_index, horizon in enumerate(config["controls"]["horizons"]):
            _render_rows(
                run_dir / f"controls/reconstruction-panel-{int(horizon):04d}.png",
                source_uint8,
                {
                    name: reconstruction_arrays[name][horizon_index]
                    for name in ("zero", "oracle", "learned")
                },
            )

        prior_arrays, prior_rows, sampling_summary = _prior_panel(
            model, source, label, schedule, config
        )
        _write_npz(run_dir / "sampling/prior_panel.npz", **prior_arrays)
        _write_csv(run_dir / "sampling/path_metrics.csv", prior_rows)
        _write_json(run_dir / "sampling/summary.json", sampling_summary)
        oracle_summary.update(
            {
                "maximum_prior_oracle_mse": sampling_summary[
                    "maximum_oracle_endpoint_mse"
                ],
                "prior_saved_states_all_finite": sampling_summary[
                    "oracle_saved_states_all_finite"
                ],
                "prior_integrity_pass": sampling_summary[
                    "prior_oracle_integrity_pass"
                ],
            }
        )
        oracle_summary["integrity_pass"] = int(
            oracle_summary["preflight_integrity_pass"]
            and oracle_summary["prior_integrity_pass"]
        )
        _write_json(run_dir / "controls/oracle_gate.json", oracle_summary)
        _render_rows(
            run_dir / "sampling/paired-final.png",
            source_uint8,
            {
                name: prior_arrays[f"{name}_final"]
                for name in ("zero", "oracle", "learned")
            },
        )
        for anchor_index, anchor in enumerate(config["sampling"]["anchors"]):
            _render_rows(
                run_dir / f"sampling/trajectory-step-{int(anchor):04d}.png",
                source_uint8,
                {
                    name: prior_arrays[f"{name}_anchors"][anchor_index]
                    for name in ("zero", "oracle", "learned")
                },
            )

        (run_dir / "REPORT.md").write_text(
            _report(
                config,
                source_record,
                training_summary,
                oracle_summary,
                reconstruction_summary,
                sampling_summary,
                provenance,
                command,
            ),
            encoding="utf-8",
        )
        objective_diagnostic_pass = int(
            oracle_summary["integrity_pass"]
            and reconstruction_summary[
                "complete_forward_terminal_reconstruction_pass"
            ]
            and sampling_summary["learner_diagnostic_pass"]
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema": VERSION + "-status",
                "state": "complete",
                "error": None,
                "integrity_pass": oracle_summary["integrity_pass"],
                "complete_forward_terminal_reconstruction_pass": (
                    reconstruction_summary[
                        "complete_forward_terminal_reconstruction_pass"
                    ]
                ),
                "learner_sampling_pass": sampling_summary[
                    "learner_diagnostic_pass"
                ],
                "diagnostic_pass": objective_diagnostic_pass,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        _artifact_manifest(run_dir)
        return run_dir
    except (Exception, KeyboardInterrupt) as error:
        _write_json(
            run_dir / "status.json",
            {
                "schema": VERSION + "-status",
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if not (run_dir / "REPORT.md").exists():
            (run_dir / "REPORT.md").write_text(
                "# Tiny pixel-DDPM one-image run\n\n"
                f"Run failed before completion: `{type(error).__name__}: {error}`.\n\n"
                "Partial controls, training evidence, and images remain in this directory.\n",
                encoding="utf-8",
            )
        _artifact_manifest(run_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment13-tiny-ddpm-one-image"),
    )
    parser.add_argument("--run-name", default="production-label3-image0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = run(args)
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
