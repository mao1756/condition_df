from __future__ import annotations

"""Minimal pixel-space DDPM mathematics and the frozen MNIST U-Net."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DDPMSchedule:
    """Tensor-valued variance-preserving DDPM schedule."""

    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor

    def __post_init__(self) -> None:
        if self.betas.ndim != 1 or self.betas.numel() == 0:
            raise ValueError("betas must be a nonempty vector")
        if self.alphas.shape != self.betas.shape or self.alpha_bars.shape != self.betas.shape:
            raise ValueError("schedule tensors must have the same shape")

    @property
    def num_steps(self) -> int:
        return int(self.betas.numel())

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> "DDPMSchedule":
        kwargs: dict[str, object] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return DDPMSchedule(
            betas=self.betas.to(**kwargs),
            alphas=self.alphas.to(**kwargs),
            alpha_bars=self.alpha_bars.to(**kwargs),
        )


def make_linear_ddpm_schedule(
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> DDPMSchedule:
    """Construct the frozen linear variance schedule."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not 0.0 < beta_start <= beta_end < 1.0:
        raise ValueError("betas must satisfy 0 < beta_start <= beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    return DDPMSchedule(betas=betas, alphas=alphas, alpha_bars=torch.cumprod(alphas, dim=0))


def sinusoidal_timestep_embedding(timesteps: Tensor, dim: int = 128) -> Tensor:
    """Return the standard transformer-style sinusoidal timestep embedding."""

    if dim <= 0:
        raise ValueError("embedding dimension must be positive")
    timesteps = timesteps.reshape(-1).to(dtype=torch.float32)
    half = dim // 2
    if half == 0:
        return torch.zeros((timesteps.shape[0], 1), device=timesteps.device)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    angles = timesteps[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionedResidualBlock(nn.Module):
    """Two-convolution residual block with additive time/class conditioning."""

    def __init__(self, in_channels: int, out_channels: int, conditioning_dim: int = 256) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conditioning = nn.Linear(conditioning_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.conditioning(conditioning)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class ClassConditionalUNet28(nn.Module):
    """Frozen 1,378,593-parameter class-conditional U-Net for 28x28 MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.label_embedding = nn.Embedding(10, 256)

        self.input_conv = nn.Conv2d(1, 32, 3, padding=1)
        self.down1 = ConditionedResidualBlock(32, 32)
        self.downsample1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.down2 = ConditionedResidualBlock(64, 64)
        self.downsample2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)

        self.middle1 = ConditionedResidualBlock(128, 128)
        self.middle2 = ConditionedResidualBlock(128, 128)

        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up2 = ConditionedResidualBlock(128, 64)
        self.upsample1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.up1 = ConditionedResidualBlock(64, 32)

        self.output_norm = nn.GroupNorm(8, 32)
        self.output_conv = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, images: Tensor, timesteps: Tensor, labels: Tensor) -> Tensor:
        time_embedding = sinusoidal_timestep_embedding(timesteps, 128).to(dtype=images.dtype)
        conditioning = self.time_mlp(time_embedding) + self.label_embedding(labels)

        hidden = self.input_conv(images)
        skip1 = self.down1(hidden, conditioning)
        hidden = self.downsample1(skip1)
        skip2 = self.down2(hidden, conditioning)
        hidden = self.downsample2(skip2)
        hidden = self.middle1(hidden, conditioning)
        hidden = self.middle2(hidden, conditioning)
        hidden = self.upsample2(hidden)
        hidden = self.up2(torch.cat((hidden, skip2), dim=1), conditioning)
        hidden = self.upsample1(hidden)
        hidden = self.up1(torch.cat((hidden, skip1), dim=1), conditioning)
        return self.output_conv(F.silu(self.output_norm(hidden)))


def _extract(values: Tensor, timesteps: Tensor | int, reference: Tensor) -> Tensor:
    indices = torch.as_tensor(timesteps, device=values.device, dtype=torch.long)
    selected = values[indices]
    if selected.ndim == 0:
        shape = (1,) * reference.ndim
    else:
        shape = (*selected.shape, *((1,) * (reference.ndim - selected.ndim)))
    return selected.reshape(shape).to(device=reference.device, dtype=reference.dtype)


def q_sample(
    x_0: Tensor,
    timesteps: Tensor | int,
    noise: Tensor,
    schedule: DDPMSchedule,
) -> Tensor:
    """Sample the closed-form forward marginal q(x_t | x_0)."""

    alpha_bar = _extract(schedule.alpha_bars, timesteps, x_0)
    return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * noise


def epsilon_from_x0(
    x_t: Tensor,
    x_0: Tensor,
    timesteps: Tensor | int,
    schedule: DDPMSchedule,
) -> Tensor:
    """Recover the forward epsilon associated with a known clean image."""

    alpha_bar = _extract(schedule.alpha_bars, timesteps, x_t)
    return (x_t - torch.sqrt(alpha_bar) * x_0) / torch.sqrt(1.0 - alpha_bar)


def predict_x0_from_epsilon(
    x_t: Tensor,
    timesteps: Tensor | int,
    epsilon_hat: Tensor,
    schedule: DDPMSchedule,
) -> Tensor:
    """Convert an epsilon prediction to the clipped clean-image prediction."""

    alpha_bar = _extract(schedule.alpha_bars, timesteps, x_t)
    prediction = (x_t - torch.sqrt(1.0 - alpha_bar) * epsilon_hat) / torch.sqrt(alpha_bar)
    return prediction.clamp(-1.0, 1.0)


def ddpm_step_from_epsilon(
    x_t: Tensor,
    timesteps: Tensor | int,
    epsilon_hat: Tensor,
    schedule: DDPMSchedule,
    noise: Tensor | None = None,
) -> Tensor:
    """Apply the fixed-variance DDPM posterior step from an epsilon prediction."""

    if noise is None:
        indices = torch.as_tensor(timesteps, device=x_t.device)
        noise = torch.randn_like(x_t) if torch.any(indices > 0) else torch.zeros_like(x_t)
    beta = _extract(schedule.betas, timesteps, x_t)
    alpha = _extract(schedule.alphas, timesteps, x_t)
    alpha_bar = _extract(schedule.alpha_bars, timesteps, x_t)
    previous_bars = torch.cat((torch.ones_like(schedule.alpha_bars[:1]), schedule.alpha_bars[:-1]))
    alpha_bar_previous = _extract(previous_bars, timesteps, x_t)

    x_0_hat = predict_x0_from_epsilon(x_t, timesteps, epsilon_hat, schedule)
    denominator = 1.0 - alpha_bar
    posterior_mean = (
        beta * torch.sqrt(alpha_bar_previous) / denominator * x_0_hat
        + (1.0 - alpha_bar_previous) * torch.sqrt(alpha) / denominator * x_t
    )
    posterior_variance = beta * (1.0 - alpha_bar_previous) / denominator
    nonzero = _extract(
        (torch.arange(schedule.num_steps, device=schedule.betas.device) > 0).to(schedule.betas),
        timesteps,
        x_t,
    )
    return posterior_mean + nonzero * torch.sqrt(posterior_variance) * noise


def epsilon_prediction_loss(
    model: Callable[[Tensor, Tensor, Tensor], Tensor],
    x_0: Tensor,
    timesteps: Tensor,
    labels: Tensor,
    noise: Tensor,
    schedule: DDPMSchedule,
) -> Tensor:
    """Unweighted pixel-mean epsilon-prediction objective."""

    x_t = q_sample(x_0, timesteps, noise, schedule)
    return F.mse_loss(model(x_t, timesteps, labels), noise)


@torch.no_grad()
def update_ema_(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update an exponential-moving-average model in place."""

    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must lie in [0, 1]")
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters(), strict=True):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def sample_reverse(
    model: Callable[[Tensor, Tensor, Tensor], Tensor],
    labels: Tensor,
    initial_state: Tensor,
    schedule: DDPMSchedule,
    *,
    generator: torch.Generator,
    start_t: int | None = None,
    anchor_steps: Sequence[int] = (),
) -> tuple[Tensor, dict[int, Tensor]]:
    """Reverse-sample from a supplied state and retain requested completed-step anchors."""

    if start_t is None:
        start_t = schedule.num_steps - 1
    if not 0 <= start_t < schedule.num_steps:
        raise ValueError("start_t is outside the schedule")
    requested = {int(step) for step in anchor_steps}
    total_steps = start_t + 1
    if any(step < 0 or step > total_steps for step in requested):
        raise ValueError("anchor steps must be completed-step counts within the reverse horizon")

    state = initial_state.clone()
    labels = labels.to(device=state.device, dtype=torch.long)
    anchors: dict[int, Tensor] = {}
    if 0 in requested:
        anchors[0] = state.clone()

    for completed, timestep in enumerate(range(start_t, -1, -1), start=1):
        t_batch = torch.full(
            (state.shape[0],), timestep, device=state.device, dtype=torch.long
        )
        epsilon_hat = model(state, t_batch, labels)
        if timestep > 0:
            noise = torch.randn(
                state.shape,
                generator=generator,
                device=state.device,
                dtype=state.dtype,
            )
        else:
            noise = torch.zeros_like(state)
        state = ddpm_step_from_epsilon(state, t_batch, epsilon_hat, schedule, noise)
        if completed in requested:
            anchors[completed] = state.clone()
    return state, anchors
