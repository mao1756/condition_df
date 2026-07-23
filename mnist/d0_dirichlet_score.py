"""Dirichlet-form implicit-score operators for the D0 experiment.

This module is deliberately independent of the D0 pathwise-residual caches and
samplers.  It implements the fixed-grid symmetric Dirichlet-form objective

    E_p[Gamma(f, f) + 2 L f],

in the numerically scaled coordinates used by the D0 implicit-score gate.  If
``p = v nu`` then its population minimizer is ``log(v)`` up to a time-dependent
constant.  Edge orientation is right/down and positive edge flux moves mass
from a pixel to its right/down neighbour, matching ``eulerian_flux_mnist``.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    edge_alpha_value,
    natural_horizon,
)


__all__ = [
    "D0DirichletScorePotentialUNet",
    "DirichletScoreObjective",
    "D0LinearSplinePotential",
    "D0LinearSplineFitResult",
    "edge_endpoint_channels",
    "edge_difference_channels",
    "edge_incidence",
    "edge_ratio_channels",
    "harmonic_mobility_exact",
    "carre_du_champ_from_gradients",
    "cell_gradient_and_hessian",
    "exact_generator_from_derivatives",
    "exact_dirichlet_generator",
    "dirichlet_score_objective",
    "physical_flux_from_potential",
    "physical_flux_from_edge_score",
    "rademacher_edge_probes",
    "teacher_fourier_pattern",
    "teacher_dirichlet_parameters",
    "teacher_log_relative_potential",
    "teacher_cell_score",
    "teacher_edge_score",
    "teacher_cell_hessian",
    "sample_teacher_dirichlet",
    "cubic_bspline_basis",
    "fit_linear_spline_baseline",
    "stein_residual_from_derivatives",
    "stein_residual",
    "run_small_grid_operator_preflight",
    "run_operator_preflight",
]


def _validate_cells(values: Tensor, grid_size: int, *, name: str) -> None:
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape (B, N)")
    n = int(grid_size)
    if n <= 1 or values.shape[1] != n * n:
        raise ValueError(f"{name} has the wrong number of grid cells")


def _alpha(config: DirectFluxMNISTConfig) -> float:
    return float(edge_alpha_value(config))


def edge_endpoint_channels(states: Tensor, grid_size: int) -> tuple[Tensor, Tensor]:
    """Return tail and head values for right/down periodic oriented edges.

    Both outputs have shape ``(B, 2, H, W)``.  Channel zero is horizontal
    (right-facing) and channel one is vertical (down-facing).
    """

    _validate_cells(states, grid_size, name="states")
    n = int(grid_size)
    tail = states.reshape(-1, n, n)
    heads = torch.stack(
        [
            torch.roll(tail, shifts=-1, dims=-1),
            torch.roll(tail, shifts=-1, dims=-2),
        ],
        dim=1,
    )
    return tail.unsqueeze(1).expand_as(heads), heads


def edge_difference_channels(cell_values: Tensor, grid_size: int) -> Tensor:
    """Return ``value(head) - value(tail)`` on right/down edges."""

    tail, head = edge_endpoint_channels(cell_values, grid_size)
    return head - tail


def edge_incidence(edge_values: Tensor) -> Tensor:
    """Apply conservative incidence (incoming minus outgoing).

    ``edge_values`` must have shape ``(B, 2, H, W)`` and the result has shape
    ``(B, H*W)``.  This is also ``D.T @ edge_values`` when
    ``D value = value(head) - value(tail)``.
    """

    if edge_values.ndim != 4 or edge_values.shape[1] != 2:
        raise ValueError("edge_values must have shape (B, 2, H, W)")
    if edge_values.shape[-1] != edge_values.shape[-2]:
        raise ValueError("edge_values must live on a square grid")
    fx, fy = edge_values[:, 0], edge_values[:, 1]
    result = (
        torch.roll(fx, shifts=1, dims=-1)
        - fx
        + torch.roll(fy, shifts=1, dims=-2)
        - fy
    )
    return result.reshape(result.shape[0], -1)


def edge_ratio_channels(states: Tensor, grid_size: int) -> Tensor:
    """Return the manuscript ratio ``R(tail, head)`` on every edge."""

    tail, head = edge_endpoint_channels(states, grid_size)
    denom = tail + head
    # The mathematical extension at the all-zero endpoint is zero.  Avoid a
    # configured numerical mass floor here: score-cache states are strictly
    # positive and the operator gate must test the actual closed-form formula.
    safe = denom.clamp_min(torch.finfo(states.dtype).tiny)
    return torch.where(denom > 0.0, (tail - head) / safe, torch.zeros_like(denom))


def harmonic_mobility_exact(states: Tensor, config: DirectFluxMNISTConfig) -> Tensor:
    """Return exact renormalized harmonic mobility ``theta_e(states)``.

    Unlike an Euler limiter, this form coefficient does not threshold small
    positive states.  It is extended by zero only when both endpoints vanish.
    """

    n = int(config.grid_size)
    tail, head = edge_endpoint_channels(states, n)
    denom = tail + head
    safe = denom.clamp_min(torch.finfo(states.dtype).tiny)
    harmonic = torch.where(denom > 0.0, tail * head / safe, torch.zeros_like(denom))
    alpha = _alpha(config)
    return ((2.0 * alpha + 1.0) / alpha) * harmonic


def _edge_indices(grid_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    n = int(grid_size)
    cells = torch.arange(n * n, device=device).reshape(n, n)
    tails = torch.cat([cells.reshape(-1), cells.reshape(-1)])
    heads = torch.cat(
        [
            torch.roll(cells, shifts=-1, dims=-1).reshape(-1),
            torch.roll(cells, shifts=-1, dims=-2).reshape(-1),
        ]
    )
    return tails, heads


def _edge_flat(channels: Tensor) -> Tensor:
    return torch.cat([channels[:, 0].reshape(channels.shape[0], -1), channels[:, 1].reshape(channels.shape[0], -1)], dim=1)


def carre_du_champ_from_gradients(
    left_gradient: Tensor,
    right_gradient: Tensor,
    states: Tensor,
    config: DirectFluxMNISTConfig,
) -> Tensor:
    """Return the unscaled carré du champ ``Gamma(left, right)`` per state."""

    n = int(config.grid_size)
    _validate_cells(left_gradient, n, name="left_gradient")
    _validate_cells(right_gradient, n, name="right_gradient")
    if left_gradient.shape != right_gradient.shape or states.shape != left_gradient.shape:
        raise ValueError("states and both gradients must have identical shapes")
    left_edge = edge_difference_channels(left_gradient, n)
    right_edge = edge_difference_channels(right_gradient, n)
    theta = harmonic_mobility_exact(states, config)
    return float(n * n) * (theta * left_edge * right_edge).flatten(1).sum(dim=1)


def cell_gradient_and_hessian(
    potential: Tensor,
    states: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor]:
    """Differentiate a per-state scalar potential into cell gradient/Hessian.

    The potential must be sample-separable.  The helper is intended for exact
    small-grid controls and witnesses; production training uses Hutchinson HVPs.
    """

    if potential.ndim != 1 or potential.shape[0] != states.shape[0]:
        raise ValueError("potential must have shape (B,)")
    if not states.requires_grad:
        raise ValueError("states must require gradients")
    gradient = torch.autograd.grad(
        potential.sum(), states, create_graph=True, retain_graph=True
    )[0]
    rows: list[Tensor] = []
    if gradient.requires_grad:
        for index in range(states.shape[1]):
            row = torch.autograd.grad(
                gradient[:, index].sum(),
                states,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )[0]
            rows.append(torch.zeros_like(states) if row is None else row)
    else:  # pragma: no cover - autograd normally preserves a zero graph.
        rows = [torch.zeros_like(states) for _ in range(states.shape[1])]
    hessian = torch.stack(rows, dim=1)
    return gradient, hessian


def exact_generator_from_derivatives(
    states: Tensor,
    cell_gradient: Tensor,
    cell_hessian: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    time_change: Tensor | float = 1.0,
) -> Tensor:
    """Evaluate the exact fixed-grid reference generator from derivatives."""

    n = int(config.grid_size)
    _validate_cells(states, n, name="states")
    if cell_gradient.shape != states.shape:
        raise ValueError("cell_gradient must have the same shape as states")
    expected_hessian = (states.shape[0], states.shape[1], states.shape[1])
    if cell_hessian.shape != expected_hessian:
        raise ValueError(f"cell_hessian must have shape {expected_hessian}")
    tails, heads = _edge_indices(n, states.device)
    edge_gradient = cell_gradient[:, heads] - cell_gradient[:, tails]
    diagonal = torch.diagonal(cell_hessian, dim1=1, dim2=2)
    edge_hessian = (
        diagonal[:, heads]
        + diagonal[:, tails]
        - cell_hessian[:, heads, tails]
        - cell_hessian[:, tails, heads]
    )
    theta = _edge_flat(harmonic_mobility_exact(states, config))
    ratio = _edge_flat(edge_ratio_channels(states, n))
    alpha = _alpha(config)
    generator = float(n * n) * (
        theta * edge_hessian + (2.0 * alpha + 1.0) * ratio * edge_gradient
    ).sum(dim=1)
    scale = torch.as_tensor(time_change, dtype=states.dtype, device=states.device)
    if scale.ndim == 0:
        scale = scale.expand(states.shape[0])
    if scale.shape != (states.shape[0],):
        raise ValueError("time_change must be scalar or have shape (B,)")
    return generator * scale


def exact_dirichlet_generator(
    potential_fn: Callable[[Tensor], Tensor],
    states: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    time_change: Tensor | float = 1.0,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate ``L potential_fn`` exactly; return generator, gradient, Hessian."""

    states_req = states if states.requires_grad else states.detach().clone().requires_grad_(True)
    values = potential_fn(states_req)
    if values.ndim == 0:
        values = values.expand(states_req.shape[0])
    gradient, hessian = cell_gradient_and_hessian(values, states_req, create_graph=create_graph)
    generator = exact_generator_from_derivatives(
        states_req, gradient, hessian, config, time_change=time_change
    )
    return generator, gradient, hessian


@dataclass(frozen=True)
class DirichletScoreObjective:
    """Decomposed, ``h^2``-scaled generalized score-matching objective."""

    loss: Tensor
    per_sample: Tensor
    energy: Tensor
    trace: Tensor
    drift: Tensor
    potential: Tensor
    edge_score: Tensor
    state_gradient: Tensor


def rademacher_edge_probes(
    num_probes: int,
    batch_size: int,
    grid_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Create edge Rademacher probes with shape ``(M,B,2,H,W)``."""

    if int(num_probes) <= 0 or int(batch_size) <= 0:
        raise ValueError("num_probes and batch_size must be positive")
    n = int(grid_size)
    values = torch.randint(
        0,
        2,
        (int(num_probes), int(batch_size), 2, n, n),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    return values.to(dtype=dtype).mul_(2.0).sub_(1.0)


def _normalize_probes(probes: Tensor, states: Tensor, grid_size: int) -> Tensor:
    n = int(grid_size)
    if probes.ndim == 4:
        probes = probes.unsqueeze(0)
    expected = (states.shape[0], 2, n, n)
    if probes.ndim != 5 or tuple(probes.shape[1:]) != expected:
        raise ValueError(f"probes must have shape (M,{expected[0]},2,{n},{n}) or {expected}")
    if probes.shape[0] <= 0:
        raise ValueError("at least one probe is required")
    return probes.to(device=states.device, dtype=states.dtype)


def dirichlet_score_objective(
    model: nn.Module,
    tau: Tensor | float,
    states: Tensor,
    labels: Tensor,
    config: DirectFluxMNISTConfig,
    probes: Tensor,
    *,
    create_graph: bool = True,
) -> DirichletScoreObjective:
    """Evaluate the normalized ``h^2[Gamma(f,f)+2Lf]`` objective.

    ``probes`` may have shape ``(B,2,H,W)`` or ``(M,B,2,H,W)``.  Mobility,
    drift ratios, and tangent probe vectors are computed from detached states.
    This is essential: differentiating them inside the HVP would count mobility
    derivatives a second time even though they already generate the canonical
    reference drift.
    """

    n = int(config.grid_size)
    _validate_cells(states, n, name="states")
    states_req = states.detach().clone().requires_grad_(True)
    labels_req = labels.to(device=states.device, dtype=torch.long).reshape(-1)
    if labels_req.shape != (states.shape[0],):
        raise ValueError("labels must have shape (B,)")
    potential = model(tau, states_req, labels_req)
    if potential.shape != (states.shape[0],):
        raise ValueError("model must return one scalar potential per state")
    state_gradient = torch.autograd.grad(
        potential.sum(), states_req, create_graph=True, retain_graph=True
    )[0]
    edge_score = edge_difference_channels(state_gradient, n)

    coefficient_states = states_req.detach()
    theta = harmonic_mobility_exact(coefficient_states, config).detach()
    ratio = edge_ratio_channels(coefficient_states, n).detach()
    energy = (theta * edge_score.square()).flatten(1).mean(dim=1)
    drift = (
        2.0 * (2.0 * _alpha(config) + 1.0) * ratio * edge_score
    ).flatten(1).mean(dim=1)

    probe_bank = _normalize_probes(probes, states_req, n)
    trace_estimates: list[Tensor] = []
    for probe in probe_bank.unbind(0):
        tangent = edge_incidence(torch.sqrt(theta) * probe).detach()
        if state_gradient.requires_grad:
            directional = (state_gradient * tangent).sum()
            hvp = torch.autograd.grad(
                directional,
                states_req,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            # The exact objective has 2*trace divided by 2N, hence trace/N.
            trace_estimates.append((hvp * tangent).sum(dim=1) / float(n * n))
        else:
            trace_estimates.append(states_req.new_zeros(states_req.shape[0]))
    trace = torch.stack(trace_estimates, dim=0).mean(dim=0)
    per_sample = energy + trace + drift
    return DirichletScoreObjective(
        loss=per_sample.mean(),
        per_sample=per_sample,
        energy=energy,
        trace=trace,
        drift=drift,
        potential=potential,
        edge_score=edge_score,
        state_gradient=state_gradient,
    )


def physical_flux_from_edge_score(
    edge_score: Tensor,
    states: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    time_change: Tensor | float = 1.0,
) -> Tensor:
    """Convert ``grad_head f - grad_tail f`` to physical Doob flux."""

    n = int(config.grid_size)
    if edge_score.shape != (states.shape[0], 2, n, n):
        raise ValueError("edge_score must have shape (B,2,H,W)")
    theta = harmonic_mobility_exact(states, config)
    scale = torch.as_tensor(time_change, device=states.device, dtype=states.dtype)
    if scale.ndim == 0:
        scale = scale.expand(states.shape[0])
    if scale.shape != (states.shape[0],):
        raise ValueError("time_change must be scalar or have shape (B,)")
    return 2.0 * float(n * n) * scale[:, None, None, None] * theta * edge_score


def physical_flux_from_potential(
    model: nn.Module,
    tau: Tensor | float,
    states: Tensor,
    labels: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    time_change: Tensor | float = 1.0,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return physical flux, potential, and edge score for a scalar model."""

    n = int(config.grid_size)
    states_req = states if states.requires_grad else states.detach().clone().requires_grad_(True)
    potential = model(tau, states_req, labels)
    gradient = torch.autograd.grad(
        potential.sum(), states_req, create_graph=create_graph, retain_graph=create_graph
    )[0]
    edge_score = edge_difference_channels(gradient, n)
    flux = physical_flux_from_edge_score(
        edge_score, states_req.detach(), config, time_change=time_change
    )
    return flux, potential, edge_score


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _SmoothConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _SmoothUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, padding_mode="circular"
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class D0DirichletScorePotentialUNet(nn.Module):
    """Twice-differentiable, position-sensitive scalar potential U-Net.

    Inputs are density relative to uniform, its logarithm, normalized reverse
    time, a one-hot label, and periodic sine/cosine coordinates.  The scalar is
    a sum of a full-resolution spatial energy map.  The final layer is exactly
    zero-initialized, making step zero the zero-residual baseline.
    """

    def __init__(
        self,
        config: DirectFluxMNISTConfig,
        *,
        base_channels: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        n = int(config.grid_size)
        if n % 4 != 0:
            raise ValueError("grid_size must be divisible by four")
        if int(base_channels) <= 0 or int(num_classes) <= 0:
            raise ValueError("base_channels and num_classes must be positive")
        self.config = config
        self.base_channels = int(base_channels)
        self.num_classes = int(num_classes)
        coords = torch.arange(n, dtype=torch.float32) / float(n)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        coord_channels = torch.stack(
            [
                torch.sin(2.0 * math.pi * xx),
                torch.cos(2.0 * math.pi * xx),
                torch.sin(2.0 * math.pi * yy),
                torch.cos(2.0 * math.pi * yy),
            ],
            dim=0,
        ).unsqueeze(0)
        self.register_buffer("periodic_coordinates", coord_channels, persistent=True)

        c = int(base_channels)
        in_channels = 2 + 1 + self.num_classes + 4
        self.enc1 = _SmoothConvBlock(in_channels, c)
        self.down1 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1, padding_mode="circular")
        self.enc2 = _SmoothConvBlock(2 * c, 2 * c)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1, padding_mode="circular")
        self.mid = _SmoothConvBlock(4 * c, 4 * c)
        self.up2 = _SmoothUpsample(4 * c, 2 * c)
        self.dec2 = _SmoothConvBlock(4 * c, 2 * c)
        self.up1 = _SmoothUpsample(2 * c, c)
        self.dec1 = _SmoothConvBlock(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _inputs(self, tau: Tensor | float, states: Tensor, labels: Tensor) -> Tensor:
        n = int(self.config.grid_size)
        _validate_cells(states, n, name="states")
        batch = int(states.shape[0])
        labels = labels.to(device=states.device, dtype=torch.long).reshape(-1)
        if labels.shape != (batch,):
            raise ValueError("labels must have shape (B,)")
        if torch.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError("labels are outside the configured class range")
        tau_tensor = torch.as_tensor(tau, device=states.device, dtype=states.dtype)
        if tau_tensor.ndim == 0:
            tau_tensor = tau_tensor.expand(batch)
        if tau_tensor.shape != (batch,):
            raise ValueError("tau must be scalar or have shape (B,)")

        density = states.reshape(batch, 1, n, n) * float(n * n)
        # Production score states are required to be strictly positive.  There
        # is intentionally no clamp here: a clamp is a nonsmooth, implicit
        # boundary convention and would corrupt the second-order objective.
        log_relative_density = torch.log(density)
        tau_plane = (
            tau_tensor / max(float(natural_horizon(self.config)), 1e-30)
        ).reshape(batch, 1, 1, 1).expand(batch, 1, n, n)
        labels_plane = F.one_hot(labels, num_classes=self.num_classes).to(states.dtype)
        labels_plane = labels_plane.reshape(batch, self.num_classes, 1, 1).expand(
            batch, self.num_classes, n, n
        )
        coords = self.periodic_coordinates.to(device=states.device, dtype=states.dtype).expand(
            batch, 4, n, n
        )
        return torch.cat(
            [density, log_relative_density, tau_plane, labels_plane, coords], dim=1
        )

    def potential_map(self, tau: Tensor | float, states: Tensor, labels: Tensor) -> Tensor:
        """Return the position-sensitive spatial energy map ``(B,1,H,W)``."""

        x1 = self.enc1(self._inputs(tau, states, labels))
        x2 = self.enc2(F.silu(self.down1(x1)))
        x3 = self.mid(F.silu(self.down2(x2)))
        y2 = self.dec2(torch.cat([self.up2(x3), x2], dim=1))
        y1 = self.dec1(torch.cat([self.up1(y2), x1], dim=1))
        return self.out(y1)

    def forward(self, tau: Tensor | float, states: Tensor, labels: Tensor) -> Tensor:
        return self.potential_map(tau, states, labels).flatten(1).sum(dim=1)


def teacher_fourier_pattern(
    grid_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return the fixed zero-mean, unit-amplitude Fourier teacher pattern."""

    n = int(grid_size)
    coords = torch.arange(n, device=device, dtype=dtype) / float(n)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    pattern = (
        torch.sin(2.0 * math.pi * xx)
        + 0.5 * torch.cos(2.0 * math.pi * yy)
        + 0.25 * torch.sin(2.0 * math.pi * (xx + yy))
    )
    pattern = pattern - pattern.mean()
    return (pattern / pattern.abs().max().clamp_min(torch.finfo(dtype).tiny)).reshape(-1)


def teacher_dirichlet_parameters(
    reverse_fraction: Tensor,
    grid_size: int,
    *,
    base_concentration: float = 6.0,
    amplitude: float = 0.5,
) -> Tensor:
    """Return ``gamma_i(x)=6+0.5*(0.5+0.5*x)*q_i``."""

    x = torch.as_tensor(reverse_fraction)
    if x.ndim == 0:
        x = x.unsqueeze(0)
    if x.ndim != 1:
        raise ValueError("reverse_fraction must be scalar or one-dimensional")
    pattern = teacher_fourier_pattern(
        grid_size, device=x.device, dtype=x.dtype
    )
    rho = 0.5 + 0.5 * x
    gamma = float(base_concentration) + float(amplitude) * rho[:, None] * pattern[None, :]
    if torch.any(gamma <= 0.0):
        raise ValueError("teacher Dirichlet parameters must be positive")
    return gamma


def teacher_cell_score(
    states: Tensor,
    reverse_fraction: Tensor,
    *,
    reference_alpha: float = 1.0,
) -> Tensor:
    """Return the analytic cell gradient of the teacher log density ratio."""

    n = int(round(math.sqrt(states.shape[1])))
    _validate_cells(states, n, name="states")
    gamma = teacher_dirichlet_parameters(
        reverse_fraction.to(device=states.device, dtype=states.dtype), n
    )
    if gamma.shape[0] == 1 and states.shape[0] != 1:
        gamma = gamma.expand(states.shape[0], -1)
    if gamma.shape != states.shape:
        raise ValueError("reverse_fraction has an incompatible batch dimension")
    return (gamma - float(reference_alpha)) / states


def teacher_log_relative_potential(
    states: Tensor,
    reverse_fraction: Tensor,
    *,
    reference_alpha: float = 1.0,
) -> Tensor:
    """Return the teacher log density ratio up to its time-only normalizer."""

    n = int(round(math.sqrt(states.shape[1])))
    gamma = teacher_dirichlet_parameters(
        reverse_fraction.to(device=states.device, dtype=states.dtype), n
    )
    if gamma.shape[0] == 1 and states.shape[0] != 1:
        gamma = gamma.expand(states.shape[0], -1)
    if gamma.shape != states.shape:
        raise ValueError("reverse_fraction has an incompatible batch dimension")
    return ((gamma - float(reference_alpha)) * torch.log(states)).sum(dim=1)


def teacher_edge_score(
    states: Tensor,
    reverse_fraction: Tensor,
    *,
    reference_alpha: float = 1.0,
) -> Tensor:
    n = int(round(math.sqrt(states.shape[1])))
    return edge_difference_channels(
        teacher_cell_score(states, reverse_fraction, reference_alpha=reference_alpha), n
    )


def teacher_cell_hessian(
    states: Tensor,
    reverse_fraction: Tensor,
    *,
    reference_alpha: float = 1.0,
) -> Tensor:
    """Return the diagonal analytic Hessian of the teacher log density ratio."""

    n = int(round(math.sqrt(states.shape[1])))
    gamma = teacher_dirichlet_parameters(
        reverse_fraction.to(device=states.device, dtype=states.dtype), n
    )
    if gamma.shape[0] == 1 and states.shape[0] != 1:
        gamma = gamma.expand(states.shape[0], -1)
    diagonal = -(gamma - float(reference_alpha)) / states.square()
    return torch.diag_embed(diagonal)


def sample_teacher_dirichlet(
    reverse_fraction: Tensor,
    grid_size: int,
    *,
    seed: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample exact nonuniform-Dirichlet teacher states deterministically."""

    target_device = torch.device("cpu" if device is None else device)
    fractions = torch.as_tensor(reverse_fraction, device=target_device, dtype=dtype)
    if fractions.ndim == 0:
        fractions = fractions.unsqueeze(0)
    gamma = teacher_dirichlet_parameters(fractions, grid_size)
    generator_device = target_device.type if target_device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    draws = torch._standard_gamma(gamma, generator=generator)
    return draws / draws.sum(dim=1, keepdim=True)


_DEFAULT_SPLINE_KNOTS = (0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0)


def cubic_bspline_basis(
    x: Tensor,
    knots: tuple[float, ...] = _DEFAULT_SPLINE_KNOTS,
) -> Tensor:
    """Evaluate the open cubic B-spline basis (eight bases by default)."""

    if x.ndim == 0:
        x = x.unsqueeze(0)
    if x.ndim != 1:
        raise ValueError("x must be scalar or one-dimensional")
    degree = 3
    knot = torch.as_tensor(knots, device=x.device, dtype=x.dtype)
    if knot.numel() < 2 * (degree + 1) or torch.any(knot[1:] < knot[:-1]):
        raise ValueError("knots must be a nondecreasing open cubic knot vector")
    # Degree-zero intervals.  Include the right endpoint in the final interval.
    basis = ((x[:, None] >= knot[:-1]) & (x[:, None] < knot[1:])).to(x.dtype)
    at_end = x == knot[-1]
    for current_degree in range(1, degree + 1):
        count = knot.numel() - current_degree - 1
        next_basis = x.new_zeros((x.shape[0], count))
        for index in range(count):
            left_denom = knot[index + current_degree] - knot[index]
            right_denom = knot[index + current_degree + 1] - knot[index + 1]
            if float(left_denom) > 0.0:
                next_basis[:, index] += (
                    (x - knot[index]) / left_denom
                ) * basis[:, index]
            if float(right_denom) > 0.0:
                next_basis[:, index] += (
                    (knot[index + current_degree + 1] - x) / right_denom
                ) * basis[:, index + 1]
        basis = next_basis
    if torch.any(at_end):
        basis[at_end] = 0.0
        basis[at_end, -1] = 1.0
    return basis


class D0LinearSplinePotential(nn.Module):
    """Frozen time-spline, state-linear potential baseline."""

    def __init__(
        self,
        config: DirectFluxMNISTConfig,
        coefficients: Tensor,
        *,
        knots: tuple[float, ...] = _DEFAULT_SPLINE_KNOTS,
    ) -> None:
        super().__init__()
        expected_bases = len(knots) - 4
        if coefficients.shape != (expected_bases, int(config.grid_size) ** 2):
            raise ValueError("coefficients have the wrong shape")
        self.config = config
        self.knots = tuple(float(value) for value in knots)
        centered = coefficients - coefficients.mean(dim=1, keepdim=True)
        self.register_buffer("coefficients", centered.detach().clone())

    def cell_coefficients(self, tau: Tensor | float, *, like: Tensor) -> Tensor:
        tau_tensor = torch.as_tensor(tau, device=like.device, dtype=like.dtype)
        if tau_tensor.ndim == 0:
            tau_tensor = tau_tensor.expand(like.shape[0])
        if tau_tensor.shape != (like.shape[0],):
            raise ValueError("tau must be scalar or have shape (B,)")
        fraction = tau_tensor / max(float(natural_horizon(self.config)), 1e-30)
        basis = cubic_bspline_basis(fraction, self.knots)
        return basis @ self.coefficients.to(device=like.device, dtype=like.dtype)

    def forward(self, tau: Tensor | float, states: Tensor, labels: Tensor | None = None) -> Tensor:
        del labels
        return (states * self.cell_coefficients(tau, like=states)).sum(dim=1)


@dataclass(frozen=True)
class D0LinearSplineFitResult:
    model: D0LinearSplinePotential
    iterations: int
    relative_residual: float
    converged: bool


def _project_zero_mean(coefficients: Tensor) -> Tensor:
    return coefficients - coefficients.mean(dim=1, keepdim=True)


def fit_linear_spline_baseline(
    states: Tensor,
    tau: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    knots: tuple[float, ...] = _DEFAULT_SPLINE_KNOTS,
    tolerance: float = 1e-10,
    max_iterations: int = 2000,
) -> D0LinearSplineFitResult:
    """Fit the exact convex state-linear score-risk baseline with CG.

    The solve is matrix-free, float64, zero-sum projected, and uses a Jacobi
    preconditioner.  No statistical ridge is added.
    """

    n = int(config.grid_size)
    _validate_cells(states, n, name="states")
    if tau.shape != (states.shape[0],):
        raise ValueError("tau must have shape (B,)")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")
    work_states = states.detach().to(dtype=torch.float64)
    work_tau = tau.detach().to(device=states.device, dtype=torch.float64)
    fractions = work_tau / max(float(natural_horizon(config)), 1e-30)
    basis = cubic_bspline_basis(fractions, knots)
    theta = harmonic_mobility_exact(work_states, config).detach()
    ratio = edge_ratio_channels(work_states, n).detach()
    alpha = _alpha(config)

    def operator(coefficients: Tensor) -> Tensor:
        c = _project_zero_mean(coefficients)
        sample_cells = basis @ c
        sample_edges = edge_difference_channels(sample_cells, n)
        cell_gradient = edge_incidence(theta * sample_edges)
        result = basis.transpose(0, 1) @ cell_gradient / float(states.shape[0])
        return _project_zero_mean(result)

    linear_cells = edge_incidence((2.0 * alpha + 1.0) * ratio)
    linear = basis.transpose(0, 1) @ linear_cells / float(states.shape[0])
    rhs = -_project_zero_mean(linear)

    # Diagonal of D.T diag(theta) D, ignoring only cross-time-basis entries.
    degree = edge_incidence(theta)  # signed and therefore not the diagonal
    del degree
    tail, head = edge_endpoint_channels(work_states, n)
    del tail, head
    theta_x, theta_y = theta[:, 0], theta[:, 1]
    node_degree = (
        theta_x
        + torch.roll(theta_x, shifts=1, dims=-1)
        + theta_y
        + torch.roll(theta_y, shifts=1, dims=-2)
    ).reshape(states.shape[0], -1)
    diagonal = torch.einsum("sb,sn,sb->bn", basis, node_degree, basis) / float(states.shape[0])
    diagonal = diagonal.clamp_min(torch.finfo(torch.float64).eps)

    x = torch.zeros_like(rhs)
    residual = rhs - operator(x)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(torch.float64).tiny)
    z = _project_zero_mean(residual / diagonal)
    direction = z.clone()
    rz = (residual * z).sum()
    relative = float((torch.linalg.vector_norm(residual) / rhs_norm).detach().cpu())
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        applied = operator(direction)
        denom = (direction * applied).sum()
        if not torch.isfinite(denom) or float(denom.abs()) <= torch.finfo(torch.float64).tiny:
            break
        step = rz / denom
        x = _project_zero_mean(x + step * direction)
        residual = _project_zero_mean(residual - step * applied)
        relative = float((torch.linalg.vector_norm(residual) / rhs_norm).detach().cpu())
        iterations = iteration
        if relative <= float(tolerance):
            break
        z = _project_zero_mean(residual / diagonal)
        next_rz = (residual * z).sum()
        direction = _project_zero_mean(z + (next_rz / rz) * direction)
        rz = next_rz
    model = D0LinearSplinePotential(config, x.to(dtype=states.dtype), knots=knots)
    return D0LinearSplineFitResult(
        model=model,
        iterations=int(iterations),
        relative_residual=float(relative),
        converged=bool(math.isfinite(relative) and relative <= float(tolerance)),
    )


def stein_residual_from_derivatives(
    score_gradient: Tensor,
    witness_gradient: Tensor,
    witness_hessian: Tensor,
    states: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    time_change: Tensor | float = 1.0,
) -> Tensor:
    """Return ``L witness + Gamma(score, witness)`` for every state."""

    generator = exact_generator_from_derivatives(
        states, witness_gradient, witness_hessian, config, time_change=time_change
    )
    gamma = carre_du_champ_from_gradients(
        score_gradient, witness_gradient, states, config
    )
    scale = torch.as_tensor(time_change, device=states.device, dtype=states.dtype)
    if scale.ndim == 0:
        scale = scale.expand(states.shape[0])
    if scale.shape != (states.shape[0],):
        raise ValueError("time_change must be scalar or have shape (B,)")
    return generator + scale * gamma


def stein_residual(
    model: nn.Module,
    tau: Tensor | float,
    states: Tensor,
    labels: Tensor,
    witness_fn: Callable[[Tensor], Tensor],
    config: DirectFluxMNISTConfig,
    *,
    time_change: Tensor | float = 1.0,
    create_graph: bool = False,
) -> Tensor:
    """Autodifferentiate a score model and witness and return Stein residuals."""

    states_req = states.detach().clone().requires_grad_(True)
    score = model(tau, states_req, labels)
    score_gradient = torch.autograd.grad(
        score.sum(), states_req, create_graph=create_graph, retain_graph=True
    )[0]
    witness = witness_fn(states_req)
    if witness.ndim == 0:
        witness = witness.expand(states.shape[0])
    witness_gradient, witness_hessian = cell_gradient_and_hessian(
        witness, states_req, create_graph=create_graph
    )
    return stein_residual_from_derivatives(
        score_gradient,
        witness_gradient,
        witness_hessian,
        states_req,
        config,
        time_change=time_change,
    )


def run_small_grid_operator_preflight(
    config: DirectFluxMNISTConfig,
    *,
    seed: int = 260750,
    num_states: int = 8,
    hutchinson_probes: int = 4096,
) -> dict[str, float | int | bool]:
    """Run deterministic small-grid algebra and trace-estimator controls."""

    n = int(config.grid_size)
    if n > 8:
        raise ValueError("operator preflight is intended for grids no larger than 8x8")
    if num_states <= 0 or hutchinson_probes <= 0:
        raise ValueError("num_states and hutchinson_probes must be positive")
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    concentration = torch.full((int(num_states), n * n), 3.0, dtype=dtype)
    raw = torch._standard_gamma(concentration, generator=generator)
    states = raw / raw.sum(dim=1, keepdim=True)
    cells = torch.randn((n * n,), dtype=dtype, generator=generator)
    cells = cells - cells.mean()
    random_matrix = torch.randn((n * n, n * n), dtype=dtype, generator=generator)
    # A positive-semidefinite Hessian gives the relative Hutchinson check a
    # well-conditioned, non-cancelling denominator.  Indefinite traces can be
    # arbitrarily close to zero and make relative error a meaningless gate.
    matrix = random_matrix.transpose(0, 1) @ random_matrix / float(n * n)
    linear_gradient = cells.expand(num_states, -1)
    zero_hessian = torch.zeros((num_states, n * n, n * n), dtype=dtype)
    quadratic_gradient = states @ matrix.transpose(0, 1)
    quadratic_hessian = matrix.expand(num_states, -1, -1)

    constant_generator = exact_generator_from_derivatives(
        states, torch.zeros_like(states), zero_hessian, config
    )
    mass_generator = exact_generator_from_derivatives(
        states, torch.ones_like(states), zero_hessian, config
    )
    linear_generator = exact_generator_from_derivatives(
        states, linear_gradient, zero_hessian, config
    )
    quadratic_generator = exact_generator_from_derivatives(
        states, quadratic_gradient, quadratic_hessian, config
    )

    f = (states * cells).sum(dim=1)
    g = 0.5 * torch.einsum("bi,ij,bj->b", states, matrix, states)
    product_gradient = g[:, None] * linear_gradient + f[:, None] * quadratic_gradient
    product_hessian = (
        g[:, None, None] * zero_hessian
        + f[:, None, None] * quadratic_hessian
        + linear_gradient[:, :, None] * quadratic_gradient[:, None, :]
        + quadratic_gradient[:, :, None] * linear_gradient[:, None, :]
    )
    product_generator = exact_generator_from_derivatives(
        states, product_gradient, product_hessian, config
    )
    product_rhs = (
        f * quadratic_generator
        + g * linear_generator
        + 2.0
        * carre_du_champ_from_gradients(
            linear_gradient, quadratic_gradient, states, config
        )
    )
    product_error = float((product_generator - product_rhs).abs().max())

    theta = harmonic_mobility_exact(states, config)
    tails, heads = _edge_indices(n, states.device)
    diagonal = torch.diagonal(quadratic_hessian, dim1=1, dim2=2)
    edge_hessian = (
        diagonal[:, heads]
        + diagonal[:, tails]
        - quadratic_hessian[:, heads, tails]
        - quadratic_hessian[:, tails, heads]
    )
    exact_trace = (_edge_flat(theta) * edge_hessian).sum(dim=1)
    estimate = torch.zeros_like(exact_trace)
    remaining = int(hutchinson_probes)
    while remaining > 0:
        count = min(remaining, 256)
        probes = rademacher_edge_probes(
            count, num_states, n, device="cpu", dtype=dtype, generator=generator
        )
        tangents = torch.stack(
            [edge_incidence(torch.sqrt(theta) * probe) for probe in probes.unbind(0)],
            dim=0,
        )
        estimate += torch.einsum("mbi,bij,mbj->b", tangents, quadratic_hessian, tangents)
        remaining -= count
    estimate /= float(hutchinson_probes)
    trace_relative = float(
        torch.linalg.vector_norm(estimate - exact_trace)
        / torch.linalg.vector_norm(exact_trace).clamp_min(torch.finfo(dtype).tiny)
    )
    finite = bool(
        torch.isfinite(linear_generator).all()
        and torch.isfinite(quadratic_generator).all()
        and math.isfinite(product_error)
        and math.isfinite(trace_relative)
    )
    passed = bool(
        finite
        and float(constant_generator.abs().max()) <= 1e-12
        and float(mass_generator.abs().max()) <= 1e-12
        and product_error <= 1e-10
        and trace_relative <= 0.01
    )
    return {
        "grid_size": n,
        "num_states": int(num_states),
        "hutchinson_probes": int(hutchinson_probes),
        "constant_generator_max_abs": float(constant_generator.abs().max()),
        "mass_gauge_generator_max_abs": float(mass_generator.abs().max()),
        "linear_generator_rms": float(linear_generator.square().mean().sqrt()),
        "quadratic_generator_rms": float(quadratic_generator.square().mean().sqrt()),
        "product_rule_max_abs_error": product_error,
        "hutchinson_trace_relative_error": trace_relative,
        "finite": finite,
        "passed": passed,
    }


def run_operator_preflight(
    config: DirectFluxMNISTConfig,
    device: torch.device | str | None = None,
    hutchinson_probes: int = 4096,
) -> dict[str, object]:
    """Run the mandatory 4x4 operator preflight and return named checks.

    The algebra is intentionally evaluated in float64 on CPU even when a CUDA
    training device is supplied; this makes the theory-lock tolerance stable
    across GPU models.  ``requested_device`` remains in the artifact so the
    orchestrator can report that policy explicitly.
    """

    small_config = copy.deepcopy(config)
    object.__setattr__(small_config, "grid_size", 4)
    metrics = run_small_grid_operator_preflight(
        small_config,
        hutchinson_probes=int(hutchinson_probes),
    )
    checks = {
        "constant_gauge": {
            "value": metrics["constant_generator_max_abs"],
            "threshold": 1e-12,
            "passed": float(metrics["constant_generator_max_abs"]) <= 1e-12,
        },
        "mass_gauge": {
            "value": metrics["mass_gauge_generator_max_abs"],
            "threshold": 1e-12,
            "passed": float(metrics["mass_gauge_generator_max_abs"]) <= 1e-12,
        },
        "product_rule": {
            "value": metrics["product_rule_max_abs_error"],
            "threshold": 1e-10,
            "passed": float(metrics["product_rule_max_abs_error"]) <= 1e-10,
        },
        "hutchinson_trace": {
            "value": metrics["hutchinson_trace_relative_error"],
            "threshold": 0.01,
            "passed": float(metrics["hutchinson_trace_relative_error"]) <= 0.01,
        },
        "finite": {"passed": bool(metrics["finite"])},
    }
    return {
        **metrics,
        "requested_device": str(torch.device("cpu" if device is None else device)),
        "evaluation_device": "cpu",
        "checks": checks,
        "passed": bool(all(bool(check["passed"]) for check in checks.values())),
    }
