from __future__ import annotations

"""Frozen DDPM denoised-target adapter for the Eulerian MNIST sampler."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    _edge_classes_torch,
    edge_alpha_value,
    eulerian_flux_step_torch,
    flux_divergence_torch,
    free_drift_flux_torch,
    natural_horizon,
    poisson_flux_from_velocity_torch,
)
from mnist.pixel_ddpm import (
    ClassConditionalUNet28,
    DDPMSchedule,
    make_linear_ddpm_schedule,
    predict_x0_from_epsilon,
    q_sample,
)


DDPM_CHECKPOINT_RELATIVE_PATH = Path("training/selected_checkpoint.pt")
DDPM_SELECTION_RELATIVE_PATH = Path("training/selection.json")
DDPM_CONFIG_RELATIVE_PATH = Path("config.json")
DDPM_SCHEDULE_RELATIVE_PATH = Path("controls/schedule.npz")
FROZEN_DDPM_CHECKPOINT_SHA256 = "5f4065da8753ad5611ec4efd61b6d13082ce3c9cccaa62258f8019118e95dfc8"
FROZEN_DDPM_CHECKPOINT_BYTES = 5_541_595
FROZEN_DDPM_SELECTION_SHA256 = "6206d8b11ce73196bc1cd31240d7f676ec8b8c89073f08dce04d94aa480e852c"
FROZEN_DDPM_SELECTION_BYTES = 283
FROZEN_DDPM_CONFIG_SHA256 = "8b02ec490d987312f7752fc70ce69708bed99f39efe6369a792b09bcacfee08d"
FROZEN_DDPM_CONFIG_BYTES = 2_236
FROZEN_DDPM_SCHEDULE_SHA256 = "8ff79b881ff72ca76989945b7f8a37b6230ef6e375d88d9b47b13d900f69a09a"
FROZEN_DDPM_SCHEDULE_BYTES = 11_056
FROZEN_DDPM_PARAMETER_COUNT = 1_378_593
FROZEN_TIME_MAP = "linear_remaining_fraction_round"
FROZEN_LATENT_POLICY = "persistent_path_latent"
FROZEN_FLUX_PROJECTION = "periodic_minimum_energy_minus_free"


__all__ = [
    "BoundDDPMGenerator",
    "DDPMEulerianAdapter",
    "DDPMEulerianAdapterConfig",
    "DDPMEulerianStep",
    "FROZEN_DDPM_CHECKPOINT_SHA256",
    "FROZEN_FLUX_PROJECTION",
    "FROZEN_LATENT_POLICY",
    "FROZEN_TIME_MAP",
    "ddpm_denoised_mass",
    "desired_mass_velocity",
    "eulerian_flux_step_with_standard_normal_torch",
    "load_bound_ddpm_generator",
    "mass_to_ddpm_model_space",
    "remaining_time_to_ddpm_timestep",
    "velocity_to_periodic_controller_flux",
]


@dataclass(frozen=True)
class DDPMEulerianAdapterConfig:
    num_ddpm_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    mass_scale_numerator: int = 25_471
    mass_scale_denominator: int = 255
    min_tau_fraction: float = 0.03
    mass_floor: float = 1e-8
    time_map: str = FROZEN_TIME_MAP
    latent_policy: str = FROZEN_LATENT_POLICY
    flux_projection: str = FROZEN_FLUX_PROJECTION

    def __post_init__(self) -> None:
        if self.num_ddpm_steps <= 0:
            raise ValueError("num_ddpm_steps must be positive")
        if not 0.0 < self.beta_start <= self.beta_end < 1.0:
            raise ValueError("DDPM betas must satisfy 0 < beta_start <= beta_end < 1")
        if self.mass_scale_numerator <= 0 or self.mass_scale_denominator <= 0:
            raise ValueError("mass scale numerator and denominator must be positive")
        if not 0.0 < self.min_tau_fraction <= 1.0:
            raise ValueError("min_tau_fraction must lie in (0, 1]")
        if not math.isfinite(self.mass_floor) or self.mass_floor <= 0.0:
            raise ValueError("mass_floor must be positive and finite")
        if self.time_map != FROZEN_TIME_MAP:
            raise ValueError(f"unsupported time_map: {self.time_map}")
        if self.latent_policy != FROZEN_LATENT_POLICY:
            raise ValueError(f"unsupported latent_policy: {self.latent_policy}")
        if self.flux_projection != FROZEN_FLUX_PROJECTION:
            raise ValueError(f"unsupported flux_projection: {self.flux_projection}")


@dataclass(frozen=True)
class DDPMEulerianStep:
    conditioning_flux: Tensor
    predicted_mass: Tensor
    desired_velocity: Tensor
    ddpm_timestep: Tensor
    epsilon_rms: Tensor
    score_rms: Tensor
    render_saturation_fraction: Tensor
    x0_saturation_fraction: Tensor
    divergence_residual_linf: Tensor
    current_render_mean: Tensor
    current_render_std: Tensor
    current_render_min: Tensor
    current_render_max: Tensor
    noisy_input_mean: Tensor
    noisy_input_std: Tensor
    noisy_input_min: Tensor
    noisy_input_max: Tensor
    predicted_x0_rms: Tensor


@dataclass(frozen=True)
class BoundDDPMGenerator:
    model: ClassConditionalUNet28
    schedule: DDPMSchedule
    checkpoint_path: Path
    checkpoint_bytes: int
    checkpoint_sha256: str
    selection_path: Path
    selection_sha256: str
    selection_metadata: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    schedule_path: Path
    schedule_bytes: int
    schedule_sha256: str
    parameter_count: int
    model_state_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _batch_vector(value: Tensor, *, batch_size: int, reference: Tensor, name: str) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.expand(batch_size)
    elif result.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar or have shape (B,)")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def eulerian_flux_step_with_standard_normal_torch(
    states: Tensor,
    conditioning_flux: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    deterministic: bool = False,
    free_weight: float | None = None,
    noise_weight: float | None = None,
    learned_weight: float | None = None,
    standard_normal_flat: Tensor | None = None,
) -> tuple[Tensor, int, int]:
    """Run the frozen Euler step with optional caller-supplied edge innovations.

    The additive pilot owns this seam so the provenance-bound shared Eulerian core
    remains byte-identical. ``None`` delegates directly to the historical step.
    """

    if standard_normal_flat is None:
        return eulerian_flux_step_torch(
            states,
            conditioning_flux,
            dt,
            config,
            deterministic=deterministic,
            free_weight=free_weight,
            noise_weight=noise_weight,
            learned_weight=learned_weight,
        )
    if states.ndim != 2:
        raise ValueError("states must have shape (B, N)")
    if conditioning_flux.ndim != 4 or conditioning_flux.shape[1] != 2:
        raise ValueError("conditioning_flux must have shape (B, 2, H, W)")
    if dt < 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be non-negative and finite")
    if dt == 0.0:
        return states.clone(), 0, 0
    n = int(config.grid_size)
    if states.shape[1] != n * n:
        raise ValueError("states have the wrong number of pixels")
    if conditioning_flux.shape[2:] != (n, n):
        raise ValueError("conditioning_flux has the wrong grid size")
    if deterministic:
        raise ValueError("standard_normal_flat is incompatible with deterministic=True")
    standard_normal = standard_normal_flat.to(device=states.device, dtype=states.dtype)
    if standard_normal.shape != (states.shape[0], 2 * n * n):
        raise ValueError("standard_normal_flat must have shape (B, 2 * H * W)")
    if not bool(torch.isfinite(standard_normal).all()):
        raise ValueError("standard_normal_flat must be finite")

    free_w = float(config.free_weight if free_weight is None else free_weight)
    noise_w = float(config.noise_weight if noise_weight is None else noise_weight)
    learned_w = float(config.learned_weight if learned_weight is None else learned_weight)
    out = states.clone()
    inv_h2 = float(n * n)
    alpha = edge_alpha_value(config)
    tiny = float(config.mass_floor)
    flat_flux = torch.cat(
        [
            conditioning_flux[:, 0].reshape(states.shape[0], -1),
            conditioning_flux[:, 1].reshape(states.shape[0], -1),
        ],
        dim=1,
    )
    clipped = 0
    proposed = 0
    for edge_class in _edge_classes_torch(n, states.device):
        tails = edge_class.tails
        heads = edge_class.heads
        a = out[:, tails]
        b = out[:, heads]
        denom = a + b
        harmonic = torch.where(
            denom > tiny,
            a * b / denom.clamp_min(tiny),
            torch.zeros_like(denom),
        )
        ratio = torch.where(
            denom > tiny,
            (a - b) / denom.clamp_min(tiny),
            torch.zeros_like(denom),
        )
        theta = ((2.0 * alpha + 1.0) / alpha) * harmonic
        free_flux = (2.0 * alpha + 1.0) * inv_h2 * ratio
        learned_flux = flat_flux[:, edge_class.flux_indices]
        d_flux = (free_w * free_flux + learned_w * learned_flux) * float(dt)
        if noise_w > 0.0:
            noise_std = noise_w * torch.sqrt(
                (2.0 * theta * inv_h2 * float(dt)).clamp_min(0.0)
            )
            d_flux = d_flux + noise_std * standard_normal[:, edge_class.flux_indices]
        pos_clip = d_flux > float(config.limiter_fraction) * a
        neg_clip = d_flux < -float(config.limiter_fraction) * b
        clipped += int(pos_clip.count_nonzero().detach().cpu())
        clipped += int(neg_clip.count_nonzero().detach().cpu())
        proposed += int(d_flux.numel())
        d_flux = torch.minimum(d_flux, float(config.limiter_fraction) * a)
        d_flux = torch.maximum(d_flux, -float(config.limiter_fraction) * b)
        out[:, tails] = out[:, tails] - d_flux
        out[:, heads] = out[:, heads] + d_flux
        out = out.clamp_min(tiny)
        out = out / out.sum(dim=1, keepdim=True).clamp_min(tiny)
    return out, clipped, proposed


def mass_to_ddpm_model_space(masses: Tensor, *, scale: float) -> tuple[Tensor, Tensor]:
    """Render simplex masses continuously into the DDPM ``[-1, 1]`` range."""

    if masses.ndim != 2 or masses.shape[1] != 28 * 28:
        raise ValueError("masses must have shape (B, 784)")
    if not masses.is_floating_point():
        raise ValueError("masses must be floating point")
    if not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    if not bool(torch.isfinite(masses).all()):
        raise ValueError("masses must be finite")
    if bool((masses < 0.0).any()):
        raise ValueError("masses must be nonnegative")
    if not bool(torch.allclose(
        masses.sum(dim=1),
        torch.ones(masses.shape[0], device=masses.device, dtype=masses.dtype),
        rtol=0.0,
        atol=2e-6,
    )):
        raise ValueError("masses must sum to one")
    intensity = masses * float(scale)
    clipped = intensity.clamp(0.0, 1.0)
    saturation = ((intensity < 0.0) | (intensity > 1.0)).to(masses.dtype).mean(dim=1)
    return (2.0 * clipped - 1.0).reshape(-1, 1, 28, 28), saturation


def remaining_time_to_ddpm_timestep(
    remaining_time: Tensor,
    *,
    horizon: float,
    num_steps: int,
) -> Tensor:
    """Map remaining Eulerian time to the frozen DDPM timestep index."""

    if not math.isfinite(float(horizon)) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    remaining = torch.as_tensor(remaining_time)
    if not remaining.is_floating_point():
        remaining = remaining.to(dtype=torch.float32)
    if not bool(torch.isfinite(remaining).all()):
        raise ValueError("remaining_time must be finite")
    scaled = (float(num_steps - 1) * remaining / float(horizon)).round()
    return scaled.clamp(0, num_steps - 1).to(dtype=torch.long)


def ddpm_denoised_mass(
    model: Callable[[Tensor, Tensor, Tensor], Tensor],
    model_space: Tensor,
    labels: Tensor,
    ddpm_timestep: Tensor,
    latent_z: Tensor,
    schedule: DDPMSchedule,
    *,
    mass_floor: float,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Forward-noise a rendered state and convert epsilon prediction to mass."""

    if model_space.ndim != 4 or model_space.shape[1:] != (1, 28, 28):
        raise ValueError("model_space must have shape (B, 1, 28, 28)")
    batch_size = int(model_space.shape[0])
    labels_t = torch.as_tensor(labels, device=model_space.device, dtype=torch.long)
    timesteps = torch.as_tensor(ddpm_timestep, device=model_space.device, dtype=torch.long)
    latent = torch.as_tensor(latent_z, device=model_space.device, dtype=model_space.dtype)
    if labels_t.shape != (batch_size,):
        raise ValueError("labels must have shape (B,)")
    if timesteps.shape != (batch_size,):
        raise ValueError("ddpm_timestep must have shape (B,)")
    if latent.shape != model_space.shape:
        raise ValueError("latent_z must have shape (B, 1, 28, 28)")
    if bool(((timesteps < 0) | (timesteps >= schedule.num_steps)).any()):
        raise ValueError("ddpm_timestep is outside the schedule")
    if not bool(torch.isfinite(model_space).all()) or not bool(torch.isfinite(latent).all()):
        raise ValueError("model_space and latent_z must be finite")
    if not math.isfinite(float(mass_floor)) or mass_floor <= 0.0:
        raise ValueError("mass_floor must be positive and finite")

    noisy_input = q_sample(model_space, timesteps, latent, schedule)
    epsilon_hat = model(noisy_input, timesteps, labels_t)
    if epsilon_hat.shape != model_space.shape:
        raise ValueError("epsilon predictor returned the wrong shape")
    if not bool(torch.isfinite(epsilon_hat).all()):
        raise ValueError("epsilon predictor returned nonfinite values")
    x0_hat = predict_x0_from_epsilon(noisy_input, timesteps, epsilon_hat, schedule)
    positive = (x0_hat + 1.0) * 0.5 + float(mass_floor)
    flat = positive.reshape(batch_size, -1)
    predicted_mass = flat / flat.sum(dim=1, keepdim=True)

    alpha_bar = schedule.alpha_bars.to(device=model_space.device, dtype=model_space.dtype)[timesteps]
    score = -epsilon_hat / torch.sqrt(1.0 - alpha_bar).reshape(-1, 1, 1, 1)
    diagnostics: Mapping[str, Tensor] = {
        "noisy_input": noisy_input,
        "epsilon_hat": epsilon_hat,
        "x0_hat": x0_hat,
        "noisy_input_mean": noisy_input.mean(dim=(1, 2, 3)),
        "noisy_input_std": noisy_input.flatten(1).std(dim=1, unbiased=False),
        "noisy_input_min": noisy_input.amin(dim=(1, 2, 3)),
        "noisy_input_max": noisy_input.amax(dim=(1, 2, 3)),
        "predicted_x0_rms": x0_hat.square().mean(dim=(1, 2, 3)).sqrt(),
        "epsilon_rms": epsilon_hat.square().mean(dim=(1, 2, 3)).sqrt(),
        "score_rms": score.square().mean(dim=(1, 2, 3)).sqrt(),
        "x0_saturation_fraction": (x0_hat.abs() >= 1.0).to(model_space.dtype).mean(dim=(1, 2, 3)),
    }
    return predicted_mass, diagnostics


def desired_mass_velocity(
    current: Tensor,
    target: Tensor,
    *,
    remaining_time: Tensor,
    minimum_time: float,
) -> Tensor:
    """Return the zero-sum remaining-time relaxation velocity toward target."""

    if current.ndim != 2 or target.shape != current.shape:
        raise ValueError("current and target must have the same shape (B, N)")
    if not math.isfinite(float(minimum_time)) or minimum_time <= 0.0:
        raise ValueError("minimum_time must be positive and finite")
    if not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("current and target must be finite")
    if bool((current < 0.0).any()) or bool((target < 0.0).any()):
        raise ValueError("current and target must be nonnegative")
    ones = torch.ones(current.shape[0], device=current.device, dtype=current.dtype)
    if not bool(torch.allclose(current.sum(dim=1), ones, rtol=0.0, atol=2e-6)):
        raise ValueError("current masses must sum to one")
    if not bool(torch.allclose(target.sum(dim=1), ones, rtol=0.0, atol=2e-6)):
        raise ValueError("target masses must sum to one")
    remaining = _batch_vector(
        remaining_time,
        batch_size=int(current.shape[0]),
        reference=current,
        name="remaining_time",
    )
    effective = remaining.clamp_min(float(minimum_time)).reshape(-1, 1)
    velocity = (target - current) / effective
    return velocity - velocity.mean(dim=1, keepdim=True)


def velocity_to_periodic_controller_flux(
    desired_velocity: Tensor,
    current_masses: Tensor,
    eulerian_config: DirectFluxMNISTConfig,
    *,
    free_weight: float,
) -> tuple[Tensor, Tensor]:
    """Project node velocity to periodic flux and cancel separately added free drift."""

    n = int(eulerian_config.grid_size)
    if desired_velocity.ndim != 2 or desired_velocity.shape[1] != n * n:
        raise ValueError("desired_velocity has the wrong shape")
    if current_masses.shape != desired_velocity.shape:
        raise ValueError("current_masses must match desired_velocity")
    if not math.isfinite(float(free_weight)):
        raise ValueError("free_weight must be finite")
    velocity_for_solve = desired_velocity.to(dtype=torch.float64)
    total_flux = poisson_flux_from_velocity_torch(velocity_for_solve, grid_size=n)
    reconstructed = flux_divergence_torch(total_flux).reshape_as(velocity_for_solve)
    reference_residual = (reconstructed - velocity_for_solve).abs().amax(dim=1)
    controller_flux_high_precision = total_flux - float(free_weight) * free_drift_flux_torch(
        current_masses.to(dtype=total_flux.dtype), eulerian_config
    )
    controller_flux = controller_flux_high_precision.to(dtype=current_masses.dtype)
    applied_total_flux = controller_flux + float(free_weight) * free_drift_flux_torch(
        current_masses, eulerian_config
    )
    applied_reconstructed = flux_divergence_torch(applied_total_flux).reshape_as(desired_velocity)
    applied_residual = (
        applied_reconstructed.to(dtype=torch.float64) - velocity_for_solve
    ).abs().amax(dim=1)
    residual = torch.maximum(reference_residual, applied_residual)
    return controller_flux, residual


def load_bound_ddpm_generator(
    run_dir: Path,
    *,
    device: torch.device,
    expected_sha256: str,
) -> BoundDDPMGenerator:
    """Load the hash-bound selected DDPM generator without opening outcome evidence."""

    root = Path(run_dir).resolve()
    checkpoint_path = root / DDPM_CHECKPOINT_RELATIVE_PATH
    selection_path = root / DDPM_SELECTION_RELATIVE_PATH
    config_path = root / DDPM_CONFIG_RELATIVE_PATH
    schedule_path = root / DDPM_SCHEDULE_RELATIVE_PATH
    for path in (checkpoint_path, selection_path, config_path, schedule_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = str(expected_sha256).lower()
    actual = _file_sha256(checkpoint_path)
    if actual != expected:
        raise RuntimeError("selected DDPM checkpoint hash changed")
    if expected == FROZEN_DDPM_CHECKPOINT_SHA256 and checkpoint_path.stat().st_size != FROZEN_DDPM_CHECKPOINT_BYTES:
        raise RuntimeError("selected DDPM checkpoint byte size changed")

    selection_sha256 = _file_sha256(selection_path)
    config_sha256 = _file_sha256(config_path)
    schedule_sha256 = _file_sha256(schedule_path)
    if expected == FROZEN_DDPM_CHECKPOINT_SHA256:
        frozen_receipts = (
            (selection_path, FROZEN_DDPM_SELECTION_BYTES, selection_sha256, FROZEN_DDPM_SELECTION_SHA256),
            (config_path, FROZEN_DDPM_CONFIG_BYTES, config_sha256, FROZEN_DDPM_CONFIG_SHA256),
            (schedule_path, FROZEN_DDPM_SCHEDULE_BYTES, schedule_sha256, FROZEN_DDPM_SCHEDULE_SHA256),
        )
        for path, expected_bytes, observed_sha256, expected_receipt_sha256 in frozen_receipts:
            if path.stat().st_size != expected_bytes or observed_sha256 != expected_receipt_sha256:
                raise RuntimeError(f"frozen DDPM authority changed: {path.name}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_selection_keys = {
        "checkpoint_sha256",
        "completed_epochs",
        "learned_epsilon_rms",
        "selected_epoch",
        "validation_mse",
        "zero_predictor_mse",
    }
    if type(selection) is not dict or set(selection) != expected_selection_keys:
        raise RuntimeError("DDPM selection metadata schema changed")
    if selection["checkpoint_sha256"] != actual:
        raise RuntimeError("DDPM selection checkpoint binding changed")
    if config.get("schema") != "pixel-ddpm-calibration-v1":
        raise RuntimeError("DDPM run config schema changed")
    schedule_config = config.get("schedule")
    if schedule_config != {"steps": 1000, "beta_start": 1e-4, "beta_end": 2e-2}:
        raise RuntimeError("DDPM schedule binding changed")
    model_config = config.get("model")
    if type(model_config) is not dict or model_config.get("kind") != "unet28":
        raise RuntimeError("DDPM model kind changed")
    if int(model_config.get("parameter_count", -1)) != FROZEN_DDPM_PARAMETER_COUNT:
        raise RuntimeError("DDPM model parameter count changed")

    expected_schedule = make_linear_ddpm_schedule(
        int(schedule_config["steps"]),
        float(schedule_config["beta_start"]),
        float(schedule_config["beta_end"]),
        device="cpu",
    )
    with np.load(schedule_path, allow_pickle=False) as archive:
        if set(archive.files) != {"betas", "alphas", "alpha_bars"}:
            raise RuntimeError("DDPM schedule receipt schema changed")
        schedule_arrays = {
            "betas": expected_schedule.betas.numpy(),
            "alphas": expected_schedule.alphas.numpy(),
            "alpha_bars": expected_schedule.alpha_bars.numpy(),
        }
        if any(
            archive[name].dtype != np.float32
            or archive[name].shape != (int(schedule_config["steps"]),)
            or not np.array_equal(archive[name], expected_array)
            for name, expected_array in schedule_arrays.items()
        ):
            raise RuntimeError("DDPM schedule receipt values changed")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if type(payload) is not dict or set(payload) != {"state_dict", "selected_epoch", "validation_mse"}:
        raise RuntimeError("DDPM checkpoint payload schema changed")
    if payload["selected_epoch"] != selection["selected_epoch"]:
        raise RuntimeError("DDPM selected epoch binding changed")
    if payload["validation_mse"] != selection["validation_mse"]:
        raise RuntimeError("DDPM validation loss binding changed")
    if not isinstance(payload["state_dict"], Mapping) or not all(
        isinstance(value, Tensor) for value in payload["state_dict"].values()
    ):
        raise RuntimeError("DDPM checkpoint state_dict is malformed")

    model = ClassConditionalUNet28()
    model.load_state_dict(payload["state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != FROZEN_DDPM_PARAMETER_COUNT:
        raise RuntimeError("loaded DDPM parameter count changed")
    model.requires_grad_(False)
    model.eval()
    model.to(torch.device(device))
    schedule = expected_schedule.to(torch.device(device))
    return BoundDDPMGenerator(
        model=model,
        schedule=schedule,
        checkpoint_path=checkpoint_path,
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_sha256=actual,
        selection_path=selection_path,
        selection_sha256=selection_sha256,
        selection_metadata=MappingProxyType(dict(selection)),
        config_path=config_path,
        config_sha256=config_sha256,
        schedule_path=schedule_path,
        schedule_bytes=schedule_path.stat().st_size,
        schedule_sha256=schedule_sha256,
        parameter_count=parameter_count,
        model_state_sha256=_model_state_sha256(model),
    )


class DDPMEulerianAdapter:
    def __init__(
        self,
        model: ClassConditionalUNet28,
        schedule: DDPMSchedule,
        eulerian_config: DirectFluxMNISTConfig,
        adapter_config: DDPMEulerianAdapterConfig,
    ) -> None:
        if int(eulerian_config.grid_size) != 28:
            raise ValueError("DDPM Eulerian adapter requires a 28x28 grid")
        if schedule.num_steps != adapter_config.num_ddpm_steps:
            raise ValueError("DDPM schedule length does not match adapter config")
        expected_schedule = make_linear_ddpm_schedule(
            adapter_config.num_ddpm_steps,
            adapter_config.beta_start,
            adapter_config.beta_end,
            device="cpu",
            dtype=schedule.betas.dtype,
        ).to(schedule.betas.device)
        if not all(
            torch.equal(observed, expected)
            for observed, expected in (
                (schedule.betas, expected_schedule.betas),
                (schedule.alphas, expected_schedule.alphas),
                (schedule.alpha_bars, expected_schedule.alpha_bars),
            )
        ):
            raise ValueError("DDPM schedule does not match adapter config")
        if float(eulerian_config.mass_floor) != float(adapter_config.mass_floor):
            raise ValueError("adapter and Eulerian mass floors must match")
        model.requires_grad_(False)
        model.eval()
        self.model = model
        self.schedule = schedule
        self.eulerian_config = eulerian_config
        self.adapter_config = adapter_config
        self.model_state_sha256 = _model_state_sha256(model)

    @torch.no_grad()
    def predict(
        self,
        current_masses: Tensor,
        labels: Tensor,
        remaining_time: Tensor,
        latent_z: Tensor,
    ) -> DDPMEulerianStep:
        if self.model.training:
            raise RuntimeError("DDPM model must remain in evaluation mode")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("DDPM model parameters must remain frozen")
        batch_size = int(current_masses.shape[0]) if current_masses.ndim > 0 else 0
        remaining = _batch_vector(
            remaining_time,
            batch_size=batch_size,
            reference=current_masses,
            name="remaining_time",
        )
        scale = float(self.adapter_config.mass_scale_numerator) / float(
            self.adapter_config.mass_scale_denominator
        )
        model_space, render_saturation = mass_to_ddpm_model_space(current_masses, scale=scale)
        render_flat = model_space.flatten(1)
        horizon = natural_horizon(self.eulerian_config)
        timestep = remaining_time_to_ddpm_timestep(
            remaining,
            horizon=horizon,
            num_steps=self.adapter_config.num_ddpm_steps,
        )
        predicted_mass, diagnostics = ddpm_denoised_mass(
            self.model,
            model_space,
            labels,
            timestep,
            latent_z,
            self.schedule,
            mass_floor=self.adapter_config.mass_floor,
        )
        velocity = desired_mass_velocity(
            current_masses,
            predicted_mass.to(dtype=current_masses.dtype),
            remaining_time=remaining,
            minimum_time=float(self.adapter_config.min_tau_fraction) * horizon,
        )
        conditioning_flux, residual = velocity_to_periodic_controller_flux(
            velocity,
            current_masses,
            self.eulerian_config,
            free_weight=float(self.eulerian_config.free_weight),
        )
        return DDPMEulerianStep(
            conditioning_flux=conditioning_flux,
            predicted_mass=predicted_mass,
            desired_velocity=velocity,
            ddpm_timestep=timestep,
            epsilon_rms=diagnostics["epsilon_rms"],
            score_rms=diagnostics["score_rms"],
            render_saturation_fraction=render_saturation,
            x0_saturation_fraction=diagnostics["x0_saturation_fraction"],
            divergence_residual_linf=residual,
            current_render_mean=render_flat.mean(dim=1),
            current_render_std=render_flat.std(dim=1, unbiased=False),
            current_render_min=render_flat.amin(dim=1),
            current_render_max=render_flat.amax(dim=1),
            noisy_input_mean=diagnostics["noisy_input_mean"],
            noisy_input_std=diagnostics["noisy_input_std"],
            noisy_input_min=diagnostics["noisy_input_min"],
            noisy_input_max=diagnostics["noisy_input_max"],
            predicted_x0_rms=diagnostics["predicted_x0_rms"],
        )
