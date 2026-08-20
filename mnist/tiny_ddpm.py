from __future__ import annotations

"""A roughly 30k-parameter class-conditional denoiser for 28x28 DDPM tests."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .pixel_ddpm import ConditionedResidualBlock, sinusoidal_timestep_embedding


TINY_DDPM_PARAMETER_COUNT = 29_913


class TinyClassConditionalUNet28(nn.Module):
    """One-level MNIST U-Net with exactly 29,913 trainable parameters."""

    time_embedding_dim = 32
    conditioning_dim = 52

    def __init__(self) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.conditioning_dim),
            nn.SiLU(),
            nn.Linear(self.conditioning_dim, self.conditioning_dim),
        )
        self.label_embedding = nn.Embedding(10, self.conditioning_dim)

        self.input_conv = nn.Conv2d(1, 8, 3, padding=1)
        self.down = ConditionedResidualBlock(8, 8, self.conditioning_dim)
        self.downsample = nn.Conv2d(8, 16, 4, stride=2, padding=1)
        self.middle = nn.ModuleList(
            ConditionedResidualBlock(16, 16, self.conditioning_dim)
            for _ in range(3)
        )
        self.upsample = nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
        self.up = ConditionedResidualBlock(16, 8, self.conditioning_dim)
        self.output_norm = nn.GroupNorm(8, 8)
        self.output_conv = nn.Conv2d(8, 1, 3, padding=1)

        count = sum(parameter.numel() for parameter in self.parameters())
        if count != TINY_DDPM_PARAMETER_COUNT:
            raise RuntimeError(
                f"tiny DDPM parameter contract changed: {count} != "
                f"{TINY_DDPM_PARAMETER_COUNT}"
            )

    def forward(self, images: Tensor, timesteps: Tensor, labels: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("images must have shape (B,1,28,28)")
        batch = images.shape[0]
        timesteps = timesteps.to(device=images.device, dtype=torch.long).reshape(-1)
        labels = labels.to(device=images.device, dtype=torch.long).reshape(-1)
        if timesteps.shape != (batch,) or labels.shape != (batch,):
            raise ValueError("timesteps and labels must have shape (B,)")
        if torch.any((labels < 0) | (labels >= 10)):
            raise ValueError("labels must lie in 0,...,9")

        time = sinusoidal_timestep_embedding(
            timesteps, self.time_embedding_dim
        ).to(dtype=images.dtype)
        conditioning = self.time_mlp(time) + self.label_embedding(labels)

        hidden = self.input_conv(images)
        skip = self.down(hidden, conditioning)
        hidden = self.downsample(skip)
        for block in self.middle:
            hidden = block(hidden, conditioning)
        hidden = self.upsample(hidden)
        hidden = self.up(torch.cat((hidden, skip), dim=1), conditioning)
        return self.output_conv(F.silu(self.output_norm(hidden)))


__all__ = ["TINY_DDPM_PARAMETER_COUNT", "TinyClassConditionalUNet28"]
