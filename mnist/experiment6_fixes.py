from __future__ import annotations

"""Practical fixes for Experiment 6 (MNIST weighted-point-cloud generation).

This file adds three things on top of the original codebase:

1. A *reparameterized* Monte Carlo drift estimator with much lower variance than
   the score-function estimator based on (Y - x) / tau.
2. A noisy training dataset / training wrapper so the terminal classifier is
   trained on the same kind of perturbed inputs it will see inside the h-transform.
3. An optional initial-position bank so generation starts from an empirical prior
   on the MNIST manifold instead of from a centered Gaussian blob.

The code is intended to be drop-in for the repository in ``condition_df_copyyy``.
"""

from dataclasses import dataclass
from typing import Any, Optional
import copy
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mnist.conditioned_diffusion import (
    GeneratedPointCloudSet,
    TerminalSetClassifier,
    WeightedPointCloudDataset,
    _resolve_device,
    _clip_drift_norm,
    _resolve_drift_clip_norm,
    draw_joint_mass_position_vectors_from_bank,
    draw_mass_vectors_from_bank,
    project_positions,
    rasterize_weighted_point_clouds,
    sample_initial_positions,
    evaluate_terminal_set_classifier,
)


class NoisyWeightedPointCloudDataset(WeightedPointCloudDataset):
    """Dataset that samples a random diffusion time and perturbs positions.

    This is useful because the terminal classifier is evaluated inside
    ``u_t(x) = E[g(Y)]`` at noisy terminal positions ``Y``. Training only on clean
    point clouds makes the classifier unstable exactly where generation needs it.
    """

    def __init__(
        self,
        masses: np.ndarray,
        positions: np.ndarray,
        labels: np.ndarray,
        *,
        position_jitter_std: float = 0.0,
        projection: str = "reflect",
        max_tau: float = 0.0,
        tau_sampling: str = "uniform",
    ) -> None:
        super().__init__(
            masses,
            positions,
            labels,
            position_jitter_std=position_jitter_std,
            projection=projection,
        )
        if max_tau < 0.0 or not np.isfinite(max_tau):
            raise ValueError("max_tau must be finite and non-negative")
        if tau_sampling not in {"uniform", "quadratic_bias_to_zero"}:
            raise ValueError("tau_sampling must be 'uniform' or 'quadratic_bias_to_zero'")
        self.max_tau = float(max_tau)
        self.tau_sampling = tau_sampling

    def _sample_tau(self) -> float:
        if self.max_tau <= 0.0:
            return 0.0
        u = float(np.random.rand())
        if self.tau_sampling == "uniform":
            return self.max_tau * u
        # More mass near small times, which are the hardest numerically.
        return self.max_tau * (u ** 2)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        masses, positions, label = super().__getitem__(idx)
        tau = self._sample_tau()
        if tau > 0.0:
            sigma = torch.sqrt((2.0 * tau) / masses).unsqueeze(-1)
            positions = positions + sigma * torch.randn_like(positions)
            positions = project_positions(positions, mode=self.projection)
        return masses, positions, label


def train_terminal_set_classifier_noisy(
    model: TerminalSetClassifier,
    train_masses: np.ndarray,
    train_positions: np.ndarray,
    train_labels: np.ndarray,
    *,
    val_masses: Optional[np.ndarray] = None,
    val_positions: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    position_jitter_std: float = 0.0,
    max_tau: float = 0.0,
    projection: str = "reflect",
    tau_sampling: str = "uniform",
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Like the original trainer, but with tau-noise augmentation."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model_device = _resolve_device(device)
    model = model.to(model_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = NoisyWeightedPointCloudDataset(
        train_masses,
        train_positions,
        train_labels,
        position_jitter_std=position_jitter_std,
        projection=projection,
        max_tau=max_tau,
        tau_sampling=tau_sampling,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_state: Optional[dict[str, Tensor]] = None
    best_metric = -np.inf

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_items = 0

        for batch_masses, batch_positions, batch_labels in train_loader:
            batch_masses = batch_masses.to(model_device)
            batch_positions = batch_positions.to(model_device)
            batch_labels = batch_labels.to(model_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_masses, batch_positions)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size_actual = int(batch_labels.shape[0])
            running_loss += float(loss.item()) * batch_size_actual
            running_correct += int((torch.argmax(logits, dim=1) == batch_labels).sum().item())
            running_items += batch_size_actual

        train_loss = running_loss / max(running_items, 1)
        train_accuracy = running_correct / max(running_items, 1)
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_accuracy))

        if val_masses is not None and val_positions is not None and val_labels is not None:
            val_metrics = evaluate_terminal_set_classifier(
                model,
                val_masses,
                val_positions,
                val_labels,
                batch_size=batch_size,
                device=model_device,
            )
            val_loss = float(val_metrics["loss"])
            val_accuracy = float(val_metrics["accuracy"])
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            selection_metric = val_accuracy
        else:
            history["val_loss"].append(float("nan"))
            history["val_accuracy"].append(float("nan"))
            selection_metric = train_accuracy

        if selection_metric > best_metric:
            best_metric = selection_metric
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            val_message = (
                f", val loss = {history['val_loss'][-1]:.4f}, val acc = {history['val_accuracy'][-1]:.4f}"
                if np.isfinite(history["val_accuracy"][-1])
                else ""
            )
            print(
                f"[terminal noisy] epoch {epoch + 1:03d}/{epochs:03d}: "
                f"train loss = {train_loss:.4f}, train acc = {train_accuracy:.4f}{val_message}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def draw_position_vectors_from_bank(
    position_bank: np.ndarray,
    target_labels: np.ndarray,
    *,
    bank_labels: Optional[np.ndarray] = None,
    class_conditional: bool = False,
    jitter_std: float = 0.0,
    projection: str = "reflect",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw initial positions from an empirical bank rather than from a blob.

    ``class_conditional=False`` is the more honest generative setting because the
    initial cloud need not already match the target class. ``True`` is useful as
    a sanity-check or warm-start ablation.
    """
    x = np.asarray(position_bank, dtype=np.float64)
    y = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError("position_bank must have shape (N, K, 2)")
    rng = np.random.default_rng() if rng is None else rng

    if class_conditional:
        if bank_labels is None:
            raise ValueError("bank_labels are required when class_conditional=True")
        bank_labels_arr = np.asarray(bank_labels, dtype=np.int64).reshape(-1)
        if bank_labels_arr.shape != (x.shape[0],):
            raise ValueError("bank_labels must have shape (N,)")
    else:
        bank_labels_arr = np.zeros(x.shape[0], dtype=np.int64)

    out = np.empty((len(y), x.shape[1], 2), dtype=np.float64)
    for label in np.unique(y):
        mask = y == label
        if class_conditional:
            candidates = np.flatnonzero(bank_labels_arr == label)
            if len(candidates) == 0:
                raise ValueError(f"no initial positions available for label {label}")
        else:
            candidates = np.arange(x.shape[0])
        draw = rng.choice(candidates, size=int(np.sum(mask)), replace=True)
        out[mask] = x[draw]

    if jitter_std > 0.0:
        out = out + jitter_std * rng.normal(size=out.shape)
        out = np.asarray(project_positions(out, mode=projection), dtype=np.float64)
    return out


def _infer_num_points(
    *,
    num_points: Optional[int],
    mass_bank: Optional[np.ndarray],
    initial_position_bank: Optional[np.ndarray],
) -> int:
    """Infer the particle count from explicit input or from empirical banks."""
    inferred_num_points = None
    if num_points is not None:
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        inferred_num_points = int(num_points)

    if mass_bank is not None:
        masses = np.asarray(mass_bank, dtype=np.float64)
        if masses.ndim != 2:
            raise ValueError("mass_bank must have shape (N, K)")
        if inferred_num_points is None:
            inferred_num_points = int(masses.shape[1])
        elif inferred_num_points != int(masses.shape[1]):
            raise ValueError("num_points and mass_bank disagree about K")

    if initial_position_bank is not None:
        positions = np.asarray(initial_position_bank, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("initial_position_bank must have shape (N, K, 2)")
        if inferred_num_points is None:
            inferred_num_points = int(positions.shape[1])
        elif inferred_num_points != int(positions.shape[1]):
            raise ValueError("num_points and initial_position_bank disagree about K")

    if inferred_num_points is None:
        raise ValueError(
            "pass num_points explicitly or provide mass_bank / initial_position_bank "
            "so the generator can infer K"
        )
    return inferred_num_points


def sample_truncated_poisson_dirichlet_masses(
    num_samples: int,
    num_points: int,
    *,
    beta: Optional[float] = None,
    max_terms: Optional[int] = None,
    block_size: int = 2048,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample fixed-length mass vectors from a truncated Poisson--Dirichlet law.

    We use the stick-breaking construction with ``V_j ~ Beta(1, beta)``, form the
    corresponding GEM weights, keep the largest ``num_points`` atoms of a finite
    truncation, and renormalize them to sum to one. The output rows are sorted in
    descending order to match the ranked Poisson--Dirichlet convention.
    """
    if num_samples <= 0 or num_points <= 0:
        raise ValueError("num_samples and num_points must be positive")
    if beta is None:
        beta = float(max(2 * num_points, 1))
    beta = float(beta)
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be positive and finite")
    if max_terms is None:
        max_terms = int(max(16 * math.ceil(beta), 16 * num_points, num_points))
    if max_terms < num_points:
        raise ValueError("max_terms must be at least num_points")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = np.random.default_rng() if rng is None else rng
    out = np.empty((num_samples, num_points), dtype=np.float64)

    for start in range(0, num_samples, block_size):
        stop = min(start + block_size, num_samples)
        size = stop - start

        sticks = rng.beta(1.0, beta, size=(size, max_terms))
        remaining_before = np.concatenate(
            [
                np.ones((size, 1), dtype=np.float64),
                np.cumprod(1.0 - sticks[:, :-1], axis=1, dtype=np.float64),
            ],
            axis=1,
        )
        weights = sticks * remaining_before
        tail = np.prod(1.0 - sticks, axis=1, dtype=np.float64)
        weights = np.concatenate([weights, tail[:, None]], axis=1)

        topk_idx = np.argpartition(weights, kth=weights.shape[1] - num_points, axis=1)[:, -num_points:]
        topk = np.take_along_axis(weights, topk_idx, axis=1)
        topk.sort(axis=1)
        topk = topk[:, ::-1]
        out[start:stop] = topk / topk.sum(axis=1, keepdims=True)

    return out


@torch.enable_grad()
def estimate_reparameterized_guided_drift(
    model: TerminalSetClassifier,
    masses: Tensor,
    positions: Tensor,
    labels: Tensor,
    tau: float,
    *,
    terminal_mc_samples: int = 128,
    guidance_scale: float = 1.0,
    terminal_projection: str = "reflect",
    antithetic: bool = True,
) -> Tensor:
    r"""Low-variance Monte Carlo drift estimator.

    The original estimator uses

        E_w[(Y - x)] / tau,

    which has variance of order ``1 / tau`` and becomes unusable for very small
    horizons. For differentiable neural terminal weights ``g_theta = exp(f_theta)``,
    we can instead use the pathwise identity

        b_i(t, x; y) = 2 / s_i * E_w[∇_{y_i} f_theta(Y)_y],

    with the same Gibbs weights ``w``. This is exact for differentiable terminal
    networks and dramatically more stable in practice.
    """
    if terminal_mc_samples <= 0:
        raise ValueError("terminal_mc_samples must be positive")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    if tau <= 1e-12:
        return torch.zeros_like(positions)

    batch_size, num_points, dimension = positions.shape
    mc = int(terminal_mc_samples)
    if antithetic:
        half = (mc + 1) // 2
        z = torch.randn(
            batch_size,
            half,
            num_points,
            dimension,
            device=positions.device,
            dtype=positions.dtype,
        )
        noise = torch.cat([z, -z], dim=1)[:, :mc]
    else:
        noise = torch.randn(
            batch_size,
            mc,
            num_points,
            dimension,
            device=positions.device,
            dtype=positions.dtype,
        )

    x = positions.detach().clone().requires_grad_(True)
    sigma = torch.sqrt((2.0 * tau) / masses).unsqueeze(1).unsqueeze(-1)
    terminal_positions = x.unsqueeze(1) + sigma * noise
    eval_positions = project_positions(terminal_positions, mode=terminal_projection)

    masses_expanded = masses[:, None, :].expand(batch_size, mc, num_points)
    logits = model(
        masses_expanded.reshape(batch_size * mc, num_points),
        eval_positions.reshape(batch_size * mc, num_points, dimension),
    ).reshape(batch_size, mc, model.num_classes)

    target_logits = logits.gather(
        2,
        labels[:, None, None].expand(batch_size, mc, 1),
    ).squeeze(-1)

    log_u_hat = torch.logsumexp(guidance_scale * target_logits, dim=1) - math.log(mc)
    grad_x = torch.autograd.grad(log_u_hat.sum(), x, create_graph=False, retain_graph=False)[0]
    drift = (2.0 / masses.unsqueeze(-1)) * grad_x
    return drift.detach()


@torch.no_grad()
def generate_guided_point_clouds_reparam(
    model: TerminalSetClassifier,
    mass_bank: Optional[np.ndarray],
    target_labels: np.ndarray,
    *,
    bank_labels: Optional[np.ndarray] = None,
    num_points: Optional[int] = None,
    mass_sampling_mode: str = "truncated_poisson_dirichlet",
    class_conditional_mass_sampling: bool = True,
    poisson_dirichlet_beta: Optional[float] = None,
    poisson_dirichlet_max_terms: Optional[int] = None,
    horizon: float = 5e-3,
    step_size: float = 5e-5,
    terminal_mc_samples: int = 128,
    guidance_scale: float = 3.0,
    initial_position_mode: str = "uniform",
    initial_position_scale: float = 0.12,
    initial_position_bank: Optional[np.ndarray] = None,
    initial_position_bank_labels: Optional[np.ndarray] = None,
    class_conditional_initial_positions: bool = False,
    joint_bank_sampling: bool = False,
    initial_position_jitter: float = 0.02,
    state_projection: str = "reflect",
    terminal_projection: str = "reflect",
    diffusion_temperature: float = 1.0,
    drift_clip_norm: Optional[float] = 20.0,
    drift_clip_total_displacement: Optional[float] = None,
    batch_size: int = 64,
    return_trajectories: bool = False,
    rasterize: bool = True,
    image_size: int = 28,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedPointCloudSet:
    """Generate weighted point clouds with a lower-variance guided drift.

    The default start is intentionally non-cheating: particle locations are drawn
    uniformly on the canvas and masses are drawn from a truncated
    Poisson--Dirichlet law rather than copied from the MNIST training bank.
    Empirical-bank starts remain available as explicit ablations.
    """
    if horizon <= 0.0 or not np.isfinite(horizon):
        raise ValueError("horizon must be positive and finite")
    if step_size <= 0.0 or not np.isfinite(step_size):
        raise ValueError("step_size must be positive and finite")
    ratio = horizon / step_size
    num_steps = int(round(ratio))
    if num_steps <= 0 or not np.isclose(ratio, num_steps, atol=1e-10, rtol=1e-10):
        raise ValueError("horizon / step_size must be an integer")
    if diffusion_temperature <= 0.0 or not np.isfinite(diffusion_temperature):
        raise ValueError("diffusion_temperature must be positive and finite")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if joint_bank_sampling and initial_position_mode != "bank":
        raise ValueError("joint_bank_sampling=True requires initial_position_mode='bank'")
    if mass_sampling_mode not in {"bank", "truncated_poisson_dirichlet"}:
        raise ValueError("mass_sampling_mode must be 'bank' or 'truncated_poisson_dirichlet'")

    labels = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng() if rng is None else rng

    inferred_num_points = _infer_num_points(
        num_points=num_points,
        mass_bank=mass_bank,
        initial_position_bank=initial_position_bank,
    )

    if initial_position_mode == "bank" and joint_bank_sampling:
        if mass_sampling_mode != "bank":
            raise ValueError("joint_bank_sampling=True requires mass_sampling_mode='bank'")
        if mass_bank is None:
            raise ValueError("mass_bank is required when mass_sampling_mode='bank'")
        if initial_position_bank is None:
            raise ValueError("initial_position_bank is required when initial_position_mode='bank'")

        joint_bank_labels = bank_labels
        if joint_bank_labels is None:
            joint_bank_labels = initial_position_bank_labels
        elif initial_position_bank_labels is not None:
            initial_position_bank_labels_arr = np.asarray(initial_position_bank_labels, dtype=np.int64).reshape(-1)
            joint_bank_labels_arr = np.asarray(joint_bank_labels, dtype=np.int64).reshape(-1)
            if initial_position_bank_labels_arr.shape != joint_bank_labels_arr.shape or not np.array_equal(
                initial_position_bank_labels_arr,
                joint_bank_labels_arr,
            ):
                raise ValueError(
                    "bank_labels and initial_position_bank_labels must agree when joint_bank_sampling=True"
                )

        masses_np, initial_positions_np = draw_joint_mass_position_vectors_from_bank(
            mass_bank,
            initial_position_bank,
            labels,
            bank_labels=joint_bank_labels,
            class_conditional=class_conditional_mass_sampling,
            rng=rng,
        )
        if initial_position_jitter > 0.0:
            initial_positions_np = initial_positions_np + initial_position_jitter * rng.normal(
                size=initial_positions_np.shape
            )
    else:
        if mass_sampling_mode == "bank":
            if mass_bank is None:
                raise ValueError("mass_bank is required when mass_sampling_mode='bank'")
            masses_np = draw_mass_vectors_from_bank(
                mass_bank,
                labels,
                bank_labels=bank_labels,
                class_conditional=class_conditional_mass_sampling,
                rng=rng,
            )
        else:
            masses_np = sample_truncated_poisson_dirichlet_masses(
                len(labels),
                inferred_num_points,
                beta=poisson_dirichlet_beta,
                max_terms=poisson_dirichlet_max_terms,
                rng=rng,
            )

        num_samples, num_points_resolved = masses_np.shape

        if initial_position_mode == "bank":
            if initial_position_bank is None:
                raise ValueError("initial_position_bank is required when initial_position_mode='bank'")
            initial_positions_np = draw_position_vectors_from_bank(
                initial_position_bank,
                labels,
                bank_labels=initial_position_bank_labels,
                class_conditional=class_conditional_initial_positions,
                jitter_std=initial_position_jitter,
                projection=state_projection,
                rng=rng,
            )
        elif initial_position_mode in {"uniform", "centered_gaussian"}:
            initial_positions_np = sample_initial_positions(
                num_samples,
                num_points_resolved,
                mode=initial_position_mode,
                scale=initial_position_scale,
                rng=rng,
            )
        else:
            raise ValueError(
                "initial_position_mode must be 'bank', 'uniform', or 'centered_gaussian'"
            )

    initial_positions_np = np.asarray(
        project_positions(initial_positions_np, mode=state_projection),
        dtype=np.float64,
    )
    num_samples, num_points_resolved = masses_np.shape
    resolved_drift_clip_norm = _resolve_drift_clip_norm(
        horizon=horizon,
        drift_clip_norm=drift_clip_norm,
        drift_clip_total_displacement=drift_clip_total_displacement,
    )

    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()

    masses = torch.from_numpy(masses_np).to(device=model_device, dtype=torch.float32)
    positions = torch.from_numpy(initial_positions_np).to(device=model_device, dtype=torch.float32)
    label_tensor = torch.from_numpy(labels).to(device=model_device, dtype=torch.long)

    if return_trajectories:
        trajectories = np.empty((num_steps + 1, num_samples, num_points_resolved, 2), dtype=np.float64)
        trajectories[0] = initial_positions_np
    else:
        trajectories = None

    times = np.linspace(0.0, horizon, num_steps + 1, dtype=np.float64)

    for step in range(num_steps):
        tau = float(horizon - times[step])
        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            batch_masses = masses[start:stop]
            batch_positions = positions[start:stop]
            batch_labels = label_tensor[start:stop]

            with torch.enable_grad():
                drift = estimate_reparameterized_guided_drift(
                    model,
                    batch_masses,
                    batch_positions,
                    batch_labels,
                    tau,
                    terminal_mc_samples=terminal_mc_samples,
                    guidance_scale=guidance_scale,
                    terminal_projection=terminal_projection,
                    antithetic=True,
                )
            drift = _clip_drift_norm(drift, resolved_drift_clip_norm)
            noise_scale = torch.sqrt((2.0 * diffusion_temperature * step_size) / batch_masses).unsqueeze(-1)
            batch_positions = batch_positions + step_size * drift + noise_scale * torch.randn_like(batch_positions)
            batch_positions = project_positions(batch_positions, mode=state_projection)
            positions[start:stop] = batch_positions

        if trajectories is not None:
            trajectories[step + 1] = positions.detach().cpu().numpy().astype(np.float64)

    final_positions = positions.detach().cpu().numpy().astype(np.float64)
    final_images = None
    if rasterize:
        final_images = rasterize_weighted_point_clouds(masses_np, final_positions, image_size=image_size)

    if was_training:
        model.train()
    return GeneratedPointCloudSet(
        masses=masses_np.astype(np.float64),
        positions=final_positions,
        labels=labels.astype(np.int64),
        images=final_images,
        trajectories=trajectories,
    )


@torch.no_grad()
def generate_balanced_synthetic_dataset_reparam(
    model: TerminalSetClassifier,
    mass_bank: Optional[np.ndarray],
    *,
    bank_labels: Optional[np.ndarray] = None,
    num_points: Optional[int] = None,
    num_per_class: int,
    mass_sampling_mode: str = "truncated_poisson_dirichlet",
    class_conditional_mass_sampling: bool = True,
    poisson_dirichlet_beta: Optional[float] = None,
    poisson_dirichlet_max_terms: Optional[int] = None,
    horizon: float = 5e-5,
    step_size: float = 5e-7,
    terminal_mc_samples: int = 128,
    guidance_scale: float = 3.0,
    initial_position_mode: str = "uniform",
    initial_position_scale: float = 0.12,
    initial_position_bank: Optional[np.ndarray] = None,
    initial_position_bank_labels: Optional[np.ndarray] = None,
    class_conditional_initial_positions: bool = False,
    joint_bank_sampling: bool = False,
    initial_position_jitter: float = 0.02,
    state_projection: str = "reflect",
    terminal_projection: str = "reflect",
    diffusion_temperature: float = 1.0,
    drift_clip_norm: Optional[float] = 20.0,
    drift_clip_total_displacement: Optional[float] = None,
    batch_size: int = 64,
    rasterize: bool = True,
    image_size: int = 28,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedPointCloudSet:
    """Balanced class-conditional wrapper around the reparameterized generator."""
    labels = np.repeat(np.arange(model.num_classes, dtype=np.int64), num_per_class)
    return generate_guided_point_clouds_reparam(
        model,
        mass_bank,
        labels,
        bank_labels=bank_labels,
        num_points=num_points,
        mass_sampling_mode=mass_sampling_mode,
        class_conditional_mass_sampling=class_conditional_mass_sampling,
        poisson_dirichlet_beta=poisson_dirichlet_beta,
        poisson_dirichlet_max_terms=poisson_dirichlet_max_terms,
        horizon=horizon,
        step_size=step_size,
        terminal_mc_samples=terminal_mc_samples,
        guidance_scale=guidance_scale,
        initial_position_mode=initial_position_mode,
        initial_position_scale=initial_position_scale,
        initial_position_bank=initial_position_bank,
        initial_position_bank_labels=initial_position_bank_labels,
        class_conditional_initial_positions=class_conditional_initial_positions,
        joint_bank_sampling=joint_bank_sampling,
        initial_position_jitter=initial_position_jitter,
        state_projection=state_projection,
        terminal_projection=terminal_projection,
        diffusion_temperature=diffusion_temperature,
        drift_clip_norm=drift_clip_norm,
        drift_clip_total_displacement=drift_clip_total_displacement,
        batch_size=batch_size,
        return_trajectories=False,
        rasterize=rasterize,
        image_size=image_size,
        device=device,
        rng=rng,
    )
