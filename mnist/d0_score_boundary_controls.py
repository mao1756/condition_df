"""Boundary-admissible controls for the D0 implicit-score experiment.

The first D0 score gate used ``log(N * state)`` as a neural-network input.
Although that feature is smooth in the open simplex, it admits potentials with
non-vanishing conormal flux at a simplex face and is therefore too large a
domain for the Dirichlet-form integration-by-parts identity.  This module is a
versioned, controls-only replacement.  It deliberately contains no physical
cache, training orchestration, or sampler for the learned D0 model.

The neural potential uses only functions that are smooth on the *closed*
simplex.  The exact nonlinear teacher is a bounded density ratio with respect
to ``Dirichlet(1)`` and can be sampled without rejection or approximation.
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .d0_dirichlet_score import (
    D0DirichletScorePotentialUNet,
    cell_gradient_and_hessian,
    edge_difference_channels,
    edge_endpoint_channels,
    edge_incidence,
    edge_ratio_channels,
    exact_generator_from_derivatives,
    harmonic_mobility_exact,
    physical_flux_from_edge_score,
    run_operator_preflight,
)
from .eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    edge_alpha_value,
    natural_horizon,
)


BOUNDARY_SMOOTH_MODEL_VERSION = "d0-boundary-smooth-potential-unet-v1"
BOUNDED_TEACHER_VERSION = "d0-bounded-four-anchor-mixture-v1"
ORTHOGONAL_HADAMARD_PROBE_VERSION = "d0-randomized-orthogonal-hadamard-v1"


__all__ = [
    "BOUNDARY_SMOOTH_MODEL_VERSION",
    "BOUNDED_TEACHER_VERSION",
    "ORTHOGONAL_HADAMARD_PROBE_VERSION",
    "D0BoundarySmoothPotentialUNet",
    "bounded_teacher_anchor_indices",
    "bounded_teacher_weights",
    "bounded_teacher_density_ratio",
    "bounded_teacher_log_relative_potential",
    "bounded_teacher_cell_score",
    "bounded_teacher_cell_hessian",
    "bounded_teacher_hessian_vector_product",
    "bounded_teacher_edge_score",
    "bounded_teacher_physical_flux",
    "sample_bounded_teacher_mixture",
    "orthogonal_hadamard_edge_probes",
    "randomized_orthogonal_hadamard_edge_probes",
    "legacy_log_barrier_trace_drift_coefficient",
    "run_orthogonal_probe_preflight",
    "run_facet_ray_preflight",
    "run_boundary_model_facet_preflight",
    "run_legacy_log_barrier_preflight",
    "run_boundary_operator_preflight",
]


def _validate_states(states: Tensor, grid_size: int) -> None:
    n = int(grid_size)
    if states.ndim != 2 or states.shape[1] != n * n:
        raise ValueError(f"states must have shape (B, {n * n})")


class D0BoundarySmoothPotentialUNet(D0DirichletScorePotentialUNet):
    """Closed-simplex-smooth version of the D0 scalar-potential U-Net.

    Its module topology and state-dict keys intentionally match
    :class:`D0DirichletScorePotentialUNet`, but the second state channel is
    ``log1p(N*s)`` instead of ``log(N*s)``.  Loading a legacy state dict can be
    useful for advisory reporting, but orchestrators must use ``model_version``
    to prevent treating it as a compatible control checkpoint.
    """

    model_version = BOUNDARY_SMOOTH_MODEL_VERSION
    state_feature_names = ("relative_density", "log1p_relative_density")

    def _inputs(self, tau: Tensor | float, states: Tensor, labels: Tensor) -> Tensor:
        n = int(self.config.grid_size)
        _validate_states(states, n)
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
        # Both channels, including every derivative of log1p on [0, N], are
        # finite at a closed-simplex face.  No clamp or hidden mass floor is
        # used here.
        smooth_log_density = torch.log1p(density)
        tau_plane = (
            tau_tensor / max(float(natural_horizon(self.config)), 1e-30)
        ).reshape(batch, 1, 1, 1).expand(batch, 1, n, n)
        labels_plane = F.one_hot(labels, num_classes=self.num_classes).to(states.dtype)
        labels_plane = labels_plane.reshape(batch, self.num_classes, 1, 1).expand(
            batch, self.num_classes, n, n
        )
        coords = self.periodic_coordinates.to(
            device=states.device, dtype=states.dtype
        ).expand(batch, 4, n, n)
        return torch.cat(
            [density, smooth_log_density, tau_plane, labels_plane, coords], dim=1
        )


def bounded_teacher_anchor_indices(
    grid_size: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return TL/TR/BL/BR quarter-grid cell indices.

    The D0 production grid is divisible by four.  Requiring the same here
    keeps the anchor convention unambiguous across cache and control artifacts.
    """

    n = int(grid_size)
    if n <= 0 or n % 4 != 0:
        raise ValueError("grid_size must be a positive multiple of four")
    lo, hi = n // 4, 3 * n // 4
    return torch.tensor(
        [lo * n + lo, lo * n + hi, hi * n + lo, hi * n + hi],
        dtype=torch.long,
        device=device,
    )


def bounded_teacher_weights(reverse_fraction: Tensor | float) -> Tensor:
    """Return the four normalized nonnegative teacher weights.

    For ``u=tau/T`` the order is ``[(1-u)^2, u(1-u), u(1-u), u^2]``.
    """

    fraction = torch.as_tensor(reverse_fraction)
    if not (fraction.is_floating_point() or fraction.is_complex()):
        fraction = fraction.to(torch.float64)
    if fraction.is_complex():
        raise ValueError("reverse_fraction must be real")
    scalar = fraction.ndim == 0
    if scalar:
        fraction = fraction.unsqueeze(0)
    if fraction.ndim != 1:
        raise ValueError("reverse_fraction must be scalar or one-dimensional")
    if not bool(torch.isfinite(fraction).all()):
        raise ValueError("reverse_fraction must be finite")
    if bool(torch.any((fraction < 0.0) | (fraction > 1.0))):
        raise ValueError("reverse_fraction must lie in [0, 1]")
    one_minus = 1.0 - fraction
    cross = fraction * one_minus
    weights = torch.stack(
        [one_minus.square(), cross, cross, fraction.square()], dim=1
    )
    # The polynomial already sums to one algebraically.  This final division
    # removes only floating-point roundoff and fixes exact categorical plans.
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights[0] if scalar else weights


def _teacher_fraction_batch(
    states: Tensor, reverse_fraction: Tensor | float
) -> Tensor:
    fraction = torch.as_tensor(
        reverse_fraction, device=states.device, dtype=states.dtype
    )
    if fraction.ndim == 0:
        fraction = fraction.expand(states.shape[0])
    if fraction.shape == (1,) and states.shape[0] != 1:
        fraction = fraction.expand(states.shape[0])
    if fraction.shape != (states.shape[0],):
        raise ValueError("reverse_fraction is incompatible with the state batch")
    if not bool(torch.isfinite(fraction).all()):
        raise ValueError("reverse_fraction must be finite")
    if bool(torch.any((fraction < 0.0) | (fraction > 1.0))):
        raise ValueError("reverse_fraction must lie in [0, 1]")
    return fraction


def _teacher_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not math.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError("epsilon must be finite and strictly between zero and one")
    return value


def _teacher_coefficients(
    states: Tensor,
    reverse_fraction: Tensor | float,
    *,
    epsilon: float,
) -> Tensor:
    n_cells = int(states.shape[1])
    grid_size = int(round(math.sqrt(n_cells)))
    _validate_states(states, grid_size)
    fraction = _teacher_fraction_batch(states, reverse_fraction)
    weights = bounded_teacher_weights(fraction).to(
        device=states.device, dtype=states.dtype
    )
    anchors = bounded_teacher_anchor_indices(grid_size, device=states.device)
    coefficients = states.new_zeros(states.shape)
    coefficients[:, anchors] = float(_teacher_epsilon(epsilon) * n_cells) * weights
    return coefficients


def bounded_teacher_density_ratio(
    states: Tensor,
    reverse_fraction: Tensor | float,
    *,
    epsilon: float = 0.5,
) -> Tensor:
    """Evaluate the exact bounded ratio ``p_tau / Dirichlet(1)``."""

    eps = _teacher_epsilon(epsilon)
    coefficients = _teacher_coefficients(
        states, reverse_fraction, epsilon=eps
    )
    return (1.0 - eps) + (coefficients * states).sum(dim=1)


def bounded_teacher_log_relative_potential(
    states: Tensor,
    reverse_fraction: Tensor | float,
    *,
    epsilon: float = 0.5,
) -> Tensor:
    """Return the normalized bounded teacher log density ratio."""

    return torch.log(
        bounded_teacher_density_ratio(states, reverse_fraction, epsilon=epsilon)
    )


def bounded_teacher_cell_score(
    states: Tensor,
    reverse_fraction: Tensor | float,
    *,
    epsilon: float = 0.5,
) -> Tensor:
    """Return the analytic state gradient of the bounded potential."""

    coefficients = _teacher_coefficients(
        states, reverse_fraction, epsilon=epsilon
    )
    ratio = (1.0 - _teacher_epsilon(epsilon)) + (coefficients * states).sum(dim=1)
    return coefficients / ratio[:, None]


def bounded_teacher_cell_hessian(
    states: Tensor,
    reverse_fraction: Tensor | float,
    *,
    epsilon: float = 0.5,
) -> Tensor:
    """Return the analytic dense Hessian of the bounded potential.

    The Hessian is rank one.  Production callers that only need directional
    derivatives should use :func:`bounded_teacher_hessian_vector_product` to
    avoid materializing an ``N x N`` matrix.
    """

    score = bounded_teacher_cell_score(
        states, reverse_fraction, epsilon=epsilon
    )
    return -score[:, :, None] * score[:, None, :]


def bounded_teacher_hessian_vector_product(
    states: Tensor,
    reverse_fraction: Tensor | float,
    vectors: Tensor,
    *,
    epsilon: float = 0.5,
) -> Tensor:
    """Apply the teacher Hessian to one or more batches of vectors."""

    if vectors.ndim == 2:
        if vectors.shape != states.shape:
            raise ValueError("vectors must match states")
        score = bounded_teacher_cell_score(
            states, reverse_fraction, epsilon=epsilon
        )
        return -score * (score * vectors).sum(dim=1, keepdim=True)
    if vectors.ndim == 3:
        if tuple(vectors.shape[1:]) != tuple(states.shape):
            raise ValueError("vectors must have shape (M, B, N)")
        score = bounded_teacher_cell_score(
            states, reverse_fraction, epsilon=epsilon
        )
        return -score.unsqueeze(0) * (
            score.unsqueeze(0) * vectors
        ).sum(dim=2, keepdim=True)
    raise ValueError("vectors must have shape (B, N) or (M, B, N)")


def bounded_teacher_edge_score(
    states: Tensor,
    reverse_fraction: Tensor | float,
    *,
    epsilon: float = 0.5,
) -> Tensor:
    """Return ``gradient(head)-gradient(tail)`` for the bounded teacher."""

    grid_size = int(round(math.sqrt(states.shape[1])))
    return edge_difference_channels(
        bounded_teacher_cell_score(states, reverse_fraction, epsilon=epsilon),
        grid_size,
    )


def bounded_teacher_physical_flux(
    states: Tensor,
    reverse_fraction: Tensor | float,
    config: DirectFluxMNISTConfig,
    *,
    epsilon: float = 0.5,
    time_change: Tensor | float = 1.0,
) -> Tensor:
    """Convert the exact bounded edge score to physical Doob flux."""

    _validate_states(states, int(config.grid_size))
    edge_score = bounded_teacher_edge_score(
        states, reverse_fraction, epsilon=epsilon
    )
    return physical_flux_from_edge_score(
        edge_score, states, config, time_change=time_change
    )


def sample_bounded_teacher_mixture(
    reverse_fraction: Tensor | float,
    grid_size: int,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    epsilon: float = 0.5,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Sample the bounded teacher exactly as a finite Dirichlet mixture.

    With probability ``1-epsilon`` this draws ``Dirichlet(1)``.  Otherwise it
    selects an anchor with the time-dependent weights and draws
    ``Dirichlet(1 + e_anchor)``.  When ``return_components`` is true, the
    result is ``(states, tilted_mask, anchor_choice)``; choices are ``-1`` for
    base-component samples and ``0..3`` for tilted samples.
    """

    eps = _teacher_epsilon(epsilon)
    n = int(grid_size)
    if device is None and isinstance(reverse_fraction, Tensor):
        device = reverse_fraction.device
    anchors = bounded_teacher_anchor_indices(n, device=device)
    target_device = anchors.device
    fraction = torch.as_tensor(
        reverse_fraction, device=target_device, dtype=dtype
    )
    if fraction.ndim == 0:
        fraction = fraction.unsqueeze(0)
    if fraction.ndim != 1 or fraction.numel() <= 0:
        raise ValueError("reverse_fraction must be scalar or a nonempty vector")
    weights = bounded_teacher_weights(fraction).to(
        device=target_device, dtype=dtype
    )
    if seed is not None and generator is not None:
        raise ValueError("pass either seed or generator, not both")
    if generator is None:
        generator_device = target_device.type if target_device.type in {"cpu", "cuda"} else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(0 if seed is None else int(seed))

    batch = int(fraction.numel())
    tilted = torch.rand(
        (batch,), device=target_device, dtype=dtype, generator=generator
    ) < eps
    categorical_uniform = torch.rand(
        (batch,), device=target_device, dtype=dtype, generator=generator
    )
    choice = (
        categorical_uniform[:, None] > weights.cumsum(dim=1)
    ).sum(dim=1).clamp_max(3).to(torch.long)
    selected_cells = anchors[choice]
    parameters = torch.ones((batch, n * n), device=target_device, dtype=dtype)
    parameters = parameters + tilted[:, None].to(dtype) * F.one_hot(
        selected_cells, num_classes=n * n
    ).to(dtype)
    draws = torch._standard_gamma(parameters, generator=generator)
    states = draws / draws.sum(dim=1, keepdim=True)
    if not return_components:
        return states
    reported_choice = torch.where(tilted, choice, torch.full_like(choice, -1))
    return states, tilted, reported_choice


def _sylvester_hadamard(order: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    result = torch.ones((1, 1), device=device, dtype=dtype)
    while result.shape[0] < order:
        result = torch.cat(
            [torch.cat([result, result], dim=1), torch.cat([result, -result], dim=1)],
            dim=0,
        )
    return result


def orthogonal_hadamard_edge_probes(
    num_probes: int,
    batch_size: int,
    grid_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Create randomized Rademacher probes in orthogonal Hadamard banks.

    The result has shape ``(M,B,2,H,W)``.  For each state, rows inside a bank
    are exactly orthogonal and have squared norm ``2*H*W``.  The bank capacity
    is the largest power of two dividing the edge count (32 on the 28x28 D0
    grid).  Requests larger than one bank receive independently randomized
    banks.  Independent column signs make every row an unbiased Hutchinson
    probe; a complete bank is exact when the bank capacity equals edge count.
    """

    count = int(num_probes)
    batch = int(batch_size)
    n = int(grid_size)
    if count <= 0 or batch <= 0 or n <= 0:
        raise ValueError("num_probes, batch_size, and grid_size must be positive")
    target_device = torch.device(device)
    edge_count = 2 * n * n
    bank_capacity = edge_count & -edge_count
    hadamard = _sylvester_hadamard(
        bank_capacity, device=target_device, dtype=dtype
    ).repeat(1, edge_count // bank_capacity)
    flat = torch.empty(
        (count, batch, edge_count), device=target_device, dtype=dtype
    )
    start = 0
    while start < count:
        bank_count = min(bank_capacity, count - start)
        # A batched random ordering avoids one GPU launch per state.  Ties in
        # continuous uniform keys have probability zero for the supported
        # float32/float64 control dtypes and do not affect unbiasedness.
        row_keys = torch.rand(
            (batch, bank_capacity),
            device=target_device,
            generator=generator,
            dtype=torch.float32,
        )
        row_order = torch.argsort(row_keys, dim=1)[:, :bank_count]
        signs = torch.randint(
            0,
            2,
            (batch, edge_count),
            device=target_device,
            generator=generator,
            dtype=torch.int64,
        ).to(dtype=dtype).mul_(2.0).sub_(1.0)
        randomized = hadamard[row_order] * signs[:, None, :]
        flat[start : start + bank_count] = randomized.permute(1, 0, 2)
        start += bank_count
    return flat.reshape(count, batch, 2, n, n)


# A long descriptive alias helps call sites distinguish this from the legacy
# independent-Rademacher helper without duplicating an implementation.
randomized_orthogonal_hadamard_edge_probes = orthogonal_hadamard_edge_probes


def run_orthogonal_probe_preflight(
    config: DirectFluxMNISTConfig,
    *,
    seed: int = 260752,
    num_states: int = 4,
) -> dict[str, object]:
    """Verify a complete 4x4 Hadamard bank against an exact dense trace."""

    if int(config.grid_size) != 4:
        raise ValueError("orthogonal probe preflight requires a 4x4 config")
    if int(num_states) <= 0:
        raise ValueError("num_states must be positive")
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    n, n_cells, edge_count = 4, 16, 32
    raw = torch._standard_gamma(
        torch.full((int(num_states), n_cells), 2.0, dtype=dtype),
        generator=generator,
    )
    states = raw / raw.sum(dim=1, keepdim=True)
    raw_matrix = torch.randn(
        (int(num_states), n_cells, n_cells), dtype=dtype, generator=generator
    )
    hessian = 0.5 * (raw_matrix + raw_matrix.transpose(1, 2))
    theta = harmonic_mobility_exact(states, config)
    probes = orthogonal_hadamard_edge_probes(
        edge_count,
        int(num_states),
        n,
        device="cpu",
        dtype=dtype,
        generator=generator,
    )
    tangents = torch.stack(
        [edge_incidence(torch.sqrt(theta) * probe) for probe in probes.unbind(0)],
        dim=0,
    )
    estimated = torch.einsum(
        "mbi,bij,mbj->mb", tangents, hessian, tangents
    ).mean(dim=0)

    cells = torch.arange(n_cells).reshape(n, n)
    tails = torch.cat([cells.reshape(-1), cells.reshape(-1)])
    heads = torch.cat(
        [
            torch.roll(cells, shifts=-1, dims=-1).reshape(-1),
            torch.roll(cells, shifts=-1, dims=-2).reshape(-1),
        ]
    )
    diagonal = torch.diagonal(hessian, dim1=1, dim2=2)
    edge_hessian = (
        diagonal[:, heads]
        + diagonal[:, tails]
        - hessian[:, heads, tails]
        - hessian[:, tails, heads]
    )
    theta_flat = torch.cat(
        [theta[:, 0].reshape(num_states, -1), theta[:, 1].reshape(num_states, -1)],
        dim=1,
    )
    exact = (theta_flat * edge_hessian).sum(dim=1)
    max_abs_error = float((estimated - exact).abs().max())
    relative_error = float(
        torch.linalg.vector_norm(estimated - exact)
        / torch.linalg.vector_norm(exact).clamp_min(torch.finfo(dtype).tiny)
    )
    flattened = probes[:, 0].reshape(edge_count, edge_count)
    gram_error = float(
        (
            flattened @ flattened.transpose(0, 1)
            - float(edge_count) * torch.eye(edge_count, dtype=dtype)
        )
        .abs()
        .max()
    )
    finite = bool(
        torch.isfinite(estimated).all()
        and torch.isfinite(exact).all()
        and math.isfinite(max_abs_error)
        and math.isfinite(relative_error)
    )
    checks = {
        "finite": {"passed": finite},
        "hadamard_gram": {
            "value": gram_error,
            "threshold": 1e-12,
            "passed": bool(gram_error <= 1e-12),
        },
        "exact_trace": {
            "max_abs_error": max_abs_error,
            "relative_error": relative_error,
            "threshold": 1e-11,
            "passed": bool(relative_error <= 1e-11),
        },
    }
    return {
        "probe_version": ORTHOGONAL_HADAMARD_PROBE_VERSION,
        "grid_size": n,
        "num_states": int(num_states),
        "num_probes": edge_count,
        "max_abs_error": max_abs_error,
        "relative_error": relative_error,
        "gram_max_abs_error": gram_error,
        "checks": checks,
        "passed": bool(all(bool(check["passed"]) for check in checks.values())),
    }


def legacy_log_barrier_trace_drift_coefficient(
    states: Tensor,
    config: DirectFluxMNISTConfig,
) -> Tensor:
    """Return the linear null-risk coefficient for ``sum_i log(s_i)``.

    At edge concentration alpha=1 this is exactly

    ``-(6/N) * sum_oriented_edges 1/(s_tail+s_head)``.

    Its expectation under ``Dirichlet(1)`` is ``-12*(N-1)``.  A negative
    coefficient is not a learnable score: the barrier violates the conormal
    boundary condition that justifies integration by parts.
    """

    n = int(config.grid_size)
    _validate_states(states, n)
    if abs(float(edge_alpha_value(config)) - 1.0) > 1e-12:
        raise ValueError("the -12*(N-1) legacy fixture requires edge alpha=1")
    if not bool(torch.isfinite(states).all()) or bool(torch.any(states <= 0.0)):
        raise ValueError("legacy log-barrier states must be finite and positive")
    tail, head = edge_endpoint_channels(states, n)
    return -(6.0 / float(n * n)) * (1.0 / (tail + head)).flatten(1).sum(dim=1)


def _facet_ray_states(
    grid_size: int,
    epsilons: Tensor,
    *,
    face_index: int = 0,
) -> Tensor:
    n_cells = int(grid_size) ** 2
    if not 0 <= int(face_index) < n_cells:
        raise ValueError("face_index is outside the grid")
    if bool(torch.any((epsilons <= 0.0) | (epsilons >= 1.0))):
        raise ValueError("facet epsilons must lie strictly between zero and one")
    result = ((1.0 - epsilons) / float(n_cells - 1))[:, None].expand(
        epsilons.numel(), n_cells
    ).clone()
    result[:, int(face_index)] = epsilons
    return result


def _incident_edge_values(edge_values: Tensor, face_index: int) -> Tensor:
    if edge_values.ndim != 4 or edge_values.shape[1] != 2:
        raise ValueError("edge_values must have shape (B,2,H,W)")
    n = int(edge_values.shape[-1])
    row, col = divmod(int(face_index), n)
    return torch.stack(
        [
            edge_values[:, 0, row, col],
            edge_values[:, 1, row, col],
            edge_values[:, 0, row, (col - 1) % n],
            edge_values[:, 1, (row - 1) % n, col],
        ],
        dim=1,
    )


def _log_log_slope(x: Tensor, y: Tensor) -> float:
    log_x = torch.log(x)
    log_y = torch.log(y.clamp_min(torch.finfo(y.dtype).tiny))
    centered = log_x - log_x.mean()
    return float(
        ((centered * (log_y - log_y.mean())).sum() / centered.square().sum())
        .detach()
        .cpu()
    )


def run_facet_ray_preflight(
    config: DirectFluxMNISTConfig,
    *,
    epsilons: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
    face_index: int = 0,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Evaluate a nontrivial ``log1p(N*s)`` potential along a simplex face.

    This deterministic analytic witness isolates the boundary behavior of the
    new state features from neural optimization.  It checks the same potential,
    gradient, HVP, generator, energy, and conormal quantities required of the
    model domain.
    """

    n = int(config.grid_size)
    n_cells = n * n
    epsilon_tensor = torch.as_tensor(tuple(epsilons), dtype=torch.float64)
    if epsilon_tensor.ndim != 1 or epsilon_tensor.numel() < 2:
        raise ValueError("at least two facet epsilons are required")
    if not bool(torch.isfinite(epsilon_tensor).all()):
        raise ValueError("facet epsilons must be finite")
    order = torch.argsort(epsilon_tensor, descending=True)
    epsilon_tensor = epsilon_tensor[order]
    states = _facet_ray_states(n, epsilon_tensor, face_index=face_index)

    coefficients = torch.linspace(-1.0, 1.0, n_cells, dtype=torch.float64)
    coefficients = coefficients - coefficients.mean()
    relative_density = float(n_cells) * states
    potential = (coefficients * torch.log1p(relative_density)).sum(dim=1)
    gradient = coefficients * float(n_cells) / (1.0 + relative_density)
    hessian_diagonal = (
        -coefficients * float(n_cells * n_cells) / (1.0 + relative_density).square()
    )
    direction = torch.cos(
        torch.arange(n_cells, dtype=torch.float64) * (2.0 * math.pi / float(n_cells))
    )
    hvp = hessian_diagonal * direction
    edge_score = edge_difference_channels(gradient, n)
    theta = harmonic_mobility_exact(states, config)
    ratio = edge_ratio_channels(states, n)
    # For a diagonal cell Hessian the edge-direction second derivative is the
    # sum, not the difference, of its two endpoint diagonal entries.
    hessian_tail, hessian_head = edge_endpoint_channels(hessian_diagonal, n)
    edge_hessian = hessian_tail + hessian_head
    alpha = float(edge_alpha_value(config))
    generator_values = float(n_cells) * (
        theta * edge_hessian + (2.0 * alpha + 1.0) * ratio * edge_score
    ).flatten(1).sum(dim=1)
    energy_values = float(n_cells) * (theta * edge_score.square()).flatten(1).sum(dim=1)
    conormal = _incident_edge_values(theta * edge_score, face_index).abs().amax(dim=1)

    asymptotic_mask = epsilon_tensor <= 1e-4
    if int(asymptotic_mask.sum()) < 2:
        raise ValueError("facet epsilons must contain at least two values <=1e-4")
    slope = _log_log_slope(
        epsilon_tensor[asymptotic_mask], conormal[asymptotic_mask]
    )
    index_1e4 = int(torch.argmin((epsilon_tensor - 1e-4).abs()))
    index_1e8 = int(torch.argmin((epsilon_tensor - 1e-8).abs()))
    if not math.isclose(float(epsilon_tensor[index_1e4]), 1e-4, rel_tol=1e-8):
        raise ValueError("facet epsilons must include 1e-4")
    if not math.isclose(float(epsilon_tensor[index_1e8]), 1e-8, rel_tol=1e-8):
        raise ValueError("facet epsilons must include 1e-8")
    decay_ratio = float(
        conormal[index_1e8]
        / conormal[index_1e4].clamp_min(torch.finfo(conormal.dtype).tiny)
    )
    # The old sum-log feature is included on the identical rays as a negative
    # boundary regression.  Its state gradient diverges as 1/epsilon, exactly
    # cancelling the linear mobility decay, so its conormal flux stays O(1).
    barrier_gradient = 1.0 / states
    barrier_conormal = _incident_edge_values(
        theta * edge_difference_channels(barrier_gradient, n), face_index
    ).abs().amax(dim=1)
    barrier_slope = _log_log_slope(
        epsilon_tensor[asymptotic_mask], barrier_conormal[asymptotic_mask]
    )
    barrier_decay_ratio = float(
        barrier_conormal[index_1e8]
        / barrier_conormal[index_1e4].clamp_min(
            torch.finfo(barrier_conormal.dtype).tiny
        )
    )
    legacy_rejected = bool(
        math.isfinite(barrier_slope)
        and abs(barrier_slope) <= 0.1
        and math.isfinite(barrier_decay_ratio)
        and barrier_decay_ratio >= 0.5
    )
    finite = bool(
        torch.isfinite(potential).all()
        and torch.isfinite(gradient).all()
        and torch.isfinite(hvp).all()
        and torch.isfinite(generator_values).all()
        and torch.isfinite(energy_values).all()
        and torch.isfinite(conormal).all()
    )
    checks = {
        "smooth_quantities_finite": {"passed": finite},
        "conormal_log_log_slope": {
            "value": slope,
            "threshold": 0.9,
            "comparison": ">=",
            "passed": bool(math.isfinite(slope) and slope >= 0.9),
        },
        "conormal_four_decade_decay": {
            "value": decay_ratio,
            "threshold": 1e-3,
            "comparison": "<=",
            "passed": bool(math.isfinite(decay_ratio) and decay_ratio <= 1e-3),
        },
        "legacy_barrier_nonvanishing": {
            "log_log_slope": barrier_slope,
            "four_decade_ratio": barrier_decay_ratio,
            "passed": legacy_rejected,
        },
    }
    rows = []
    for index in range(epsilon_tensor.numel()):
        rows.append(
            {
                "epsilon": float(epsilon_tensor[index]),
                "potential": float(potential[index]),
                "gradient_max_abs": float(gradient[index].abs().max()),
                "hvp_max_abs": float(hvp[index].abs().max()),
                "generator": float(generator_values[index]),
                "energy": float(energy_values[index]),
                "incident_conormal_max_abs": float(conormal[index]),
                "legacy_incident_conormal_max_abs": float(barrier_conormal[index]),
            }
        )
    return {
        "model_version": BOUNDARY_SMOOTH_MODEL_VERSION,
        "grid_size": n,
        "face_index": int(face_index),
        "requested_device": str(torch.device("cpu" if device is None else device)),
        "evaluation_device": "cpu",
        "rows": rows,
        "incident_flux_loglog_slope": slope,
        "incident_flux_endpoint_ratio": decay_ratio,
        "legacy_barrier_loglog_slope": barrier_slope,
        "legacy_barrier_endpoint_ratio": barrier_decay_ratio,
        "legacy_barrier_rejected": legacy_rejected,
        "checks": checks,
        "passed": bool(all(bool(check["passed"]) for check in checks.values())),
    }


def run_boundary_model_facet_preflight(
    config: DirectFluxMNISTConfig,
    *,
    epsilons: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
    face_index: int = 0,
) -> dict[str, object]:
    """Exercise the versioned U-Net and its second derivatives on facet rays.

    A fixed, nonzero parameter pattern avoids the vacuous zero-output
    initialization while keeping the check deterministic and independent of
    optimizer RNG.  The production task certificate additionally requires
    this exact model schema and finite selected-EMA parameters.
    """

    n = int(config.grid_size)
    if n != 4:
        raise ValueError("boundary model facet preflight requires a 4x4 config")
    epsilon_tensor = torch.as_tensor(tuple(epsilons), dtype=torch.float64)
    order = torch.argsort(epsilon_tensor, descending=True)
    epsilon_tensor = epsilon_tensor[order]
    states = _facet_ray_states(n, epsilon_tensor, face_index=face_index).requires_grad_(True)
    labels = torch.full((states.shape[0],), 3, dtype=torch.long)
    tau = torch.full(
        (states.shape[0],), 0.5 * float(natural_horizon(config)), dtype=torch.float64
    )
    model = D0BoundarySmoothPotentialUNet(config, base_channels=4).double().eval()
    with torch.no_grad():
        for parameter_index, (name, parameter) in enumerate(model.named_parameters()):
            phase = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
            perturbation = 0.02 * torch.sin(phase + float(parameter_index + 1))
            if name.endswith("weight") and parameter.ndim == 1:
                parameter.copy_(1.0 + perturbation)
            else:
                parameter.copy_(perturbation)

    potential = model(tau, states, labels)
    gradient, hessian = cell_gradient_and_hessian(potential, states, create_graph=False)
    direction = torch.cos(
        torch.arange(n * n, dtype=torch.float64) * (2.0 * math.pi / float(n * n))
    )
    hvp = torch.einsum("bij,j->bi", hessian, direction)
    generator = exact_generator_from_derivatives(states, gradient, hessian, config)
    edge_score = edge_difference_channels(gradient, n)
    theta = harmonic_mobility_exact(states, config)
    energy = float(n * n) * (theta * edge_score.square()).flatten(1).sum(dim=1)
    conormal = _incident_edge_values(theta * edge_score, face_index).abs().amax(dim=1)

    asymptotic = epsilon_tensor <= 1e-4
    slope = _log_log_slope(epsilon_tensor[asymptotic], conormal[asymptotic])
    index_1e4 = int(torch.argmin((epsilon_tensor - 1e-4).abs()))
    index_1e8 = int(torch.argmin((epsilon_tensor - 1e-8).abs()))
    ratio = float(
        (
            conormal[index_1e8]
            / conormal[index_1e4].clamp_min(torch.finfo(conormal.dtype).tiny)
        )
        .detach()
        .cpu()
    )
    finite = bool(
        torch.isfinite(potential).all()
        and torch.isfinite(gradient).all()
        and torch.isfinite(hvp).all()
        and torch.isfinite(generator).all()
        and torch.isfinite(energy).all()
        and torch.isfinite(conormal).all()
    )
    checks = {
        "model_schema": {
            "value": model.model_version,
            "expected": BOUNDARY_SMOOTH_MODEL_VERSION,
            "passed": model.model_version == BOUNDARY_SMOOTH_MODEL_VERSION,
        },
        "quantities_finite": {"passed": finite},
        "conormal_log_log_slope": {
            "value": slope,
            "threshold": 0.9,
            "comparison": ">=",
            "passed": bool(math.isfinite(slope) and slope >= 0.9),
        },
        "conormal_four_decade_decay": {
            "value": ratio,
            "threshold": 1e-3,
            "comparison": "<=",
            "passed": bool(math.isfinite(ratio) and 0.0 <= ratio <= 1e-3),
        },
    }
    return {
        "model_version": model.model_version,
        "grid_size": n,
        "face_index": int(face_index),
        "parameter_initialization": "deterministic-nonzero-sinusoidal-v1",
        "rows": [
            {
                "epsilon": float(epsilon_tensor[index]),
                "potential": float(potential[index].detach().cpu()),
                "gradient_max_abs": float(gradient[index].abs().max().detach().cpu()),
                "hvp_max_abs": float(hvp[index].abs().max().detach().cpu()),
                "generator": float(generator[index].detach().cpu()),
                "energy": float(energy[index].detach().cpu()),
                "incident_conormal_max_abs": float(conormal[index].detach().cpu()),
            }
            for index in range(epsilon_tensor.numel())
        ],
        "incident_flux_loglog_slope": slope,
        "incident_flux_endpoint_ratio": ratio,
        "checks": checks,
        "passed": bool(all(bool(check["passed"]) for check in checks.values())),
    }


def run_legacy_log_barrier_preflight(
    config: DirectFluxMNISTConfig,
    *,
    seed: int = 260751,
    num_states: int = 4096,
) -> dict[str, object]:
    """Show that the legacy log barrier is a negative boundary fixture."""

    if int(num_states) <= 0:
        raise ValueError("num_states must be positive")
    n = int(config.grid_size)
    n_cells = n * n
    if abs(float(edge_alpha_value(config)) - 1.0) > 1e-12:
        return {
            "grid_size": n,
            "edge_alpha": float(edge_alpha_value(config)),
            "passed": False,
            "error": "legacy fixture requires edge alpha=1",
        }
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    concentration = torch.ones((int(num_states), n_cells), dtype=torch.float64)
    draws = torch._standard_gamma(concentration, generator=generator)
    states = draws / draws.sum(dim=1, keepdim=True)
    coefficient = legacy_log_barrier_trace_drift_coefficient(states, config)
    expected = -12.0 * float(n_cells - 1)
    empirical = float(coefficient.mean())
    relative_error = abs(empirical - expected) / abs(expected)

    ray_eps = torch.tensor([1e-4, 1e-5, 1e-6, 1e-7, 1e-8], dtype=torch.float64)
    ray_states = _facet_ray_states(n, ray_eps, face_index=0)
    barrier_gradient = 1.0 / ray_states
    barrier_edge_score = edge_difference_channels(barrier_gradient, n)
    barrier_conormal = _incident_edge_values(
        harmonic_mobility_exact(ray_states, config) * barrier_edge_score, 0
    ).abs().amax(dim=1)
    barrier_slope = _log_log_slope(ray_eps, barrier_conormal)
    boundary_ratio = float(barrier_conormal[-1] / barrier_conormal[0])
    checks = {
        "expected_negative_coefficient": {
            "value": expected,
            "expected": -12.0 * float(n_cells - 1),
            "passed": bool(expected < 0.0),
        },
        "empirical_coefficient": {
            "value": empirical,
            "expected": expected,
            "relative_error": relative_error,
            "threshold": 0.10,
            "passed": bool(math.isfinite(relative_error) and relative_error <= 0.10),
        },
        "conormal_does_not_vanish": {
            "log_log_slope": barrier_slope,
            "boundary_ratio": boundary_ratio,
            "slope_threshold": 0.1,
            "ratio_threshold": 0.5,
            "passed": bool(
                math.isfinite(barrier_slope)
                and abs(barrier_slope) <= 0.1
                and math.isfinite(boundary_ratio)
                and boundary_ratio >= 0.5
            ),
        },
        "fixture_rejected": {"passed": True},
    }
    return {
        "grid_size": n,
        "num_states": int(num_states),
        "seed": int(seed),
        "edge_alpha": float(edge_alpha_value(config)),
        "expected_trace_plus_drift_coefficient": expected,
        "empirical_trace_plus_drift_coefficient": empirical,
        "empirical_relative_error": relative_error,
        "boundary_rows": [
            {
                "epsilon": float(ray_eps[index]),
                "incident_conormal_max_abs": float(barrier_conormal[index]),
            }
            for index in range(ray_eps.numel())
        ],
        "checks": checks,
        # Passing means that the negative regression was successfully detected
        # and rejected, not that the barrier is admissible.
        "admissible": False,
        "passed": bool(all(bool(check["passed"]) for check in checks.values())),
    }


def run_boundary_operator_preflight(
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device | str | None = None,
    hutchinson_probes: int = 4096,
    legacy_num_states: int = 4096,
) -> dict[str, object]:
    """Run algebra, smooth-facet, and legacy-domain regression controls."""

    operator = run_operator_preflight(
        config, device=device, hutchinson_probes=int(hutchinson_probes)
    )
    # The facet and barrier controls run on 4x4 in float64.  Their purpose is
    # checking the operator domain, and this keeps the gate deterministic and
    # inexpensive while the operator provenance still records the requested
    # production configuration.
    small_config = copy.deepcopy(config)
    object.__setattr__(small_config, "grid_size", 4)
    facet = run_facet_ray_preflight(small_config)
    model_facet = run_boundary_model_facet_preflight(small_config)
    probes = run_orthogonal_probe_preflight(small_config)
    legacy = run_legacy_log_barrier_preflight(
        small_config, num_states=int(legacy_num_states)
    )
    checks = {
        "operator": {"passed": bool(operator.get("passed", False))},
        "orthogonal_probe_trace": {"passed": bool(probes.get("passed", False))},
        "boundary_smooth_domain": {"passed": bool(facet.get("passed", False))},
        "boundary_model_domain": {"passed": bool(model_facet.get("passed", False))},
        "legacy_log_barrier_rejected": {"passed": bool(legacy.get("passed", False))},
    }
    return {
        "model_version": BOUNDARY_SMOOTH_MODEL_VERSION,
        "teacher_version": BOUNDED_TEACHER_VERSION,
        "probe_version": ORTHOGONAL_HADAMARD_PROBE_VERSION,
        "requested_grid_size": int(config.grid_size),
        "requested_device": str(torch.device("cpu" if device is None else device)),
        "operator": operator,
        "orthogonal_probe_preflight": probes,
        "facet_ray": facet,
        "model_facet_ray": model_facet,
        "legacy_log_barrier": legacy,
        "checks": checks,
        "passed": bool(all(bool(check["passed"]) for check in checks.values())),
    }
