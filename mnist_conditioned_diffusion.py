from __future__ import annotations

r"""Class-conditional generation of weighted point clouds on MNIST.

This module implements a practical experiment around the manuscript's finite-
dimensional conditioned dynamics.  Each image is represented as a weighted point
cloud / atomic probability measure, a positive terminal function

    g_\theta(\mu, y) = exp(f_\theta(\mu)_y)

is learned from data, and the drifted particle system is simulated with a Monte
Carlo approximation of the Doob ``h``-transform drift.

The key identity used in the code is the Gaussian smoothing formula

    u_t(x, y) = E[g_\theta(\Pi(Y), y)],
    Y_i = x_i + sqrt(2 (T-t) / s_i) Z_i,

which yields the drift estimator

    b_i(t, x; y)
      = 1 / (T-t)
        * E[g_\theta(\Pi(Y), y) (Y_i - x_i)] / E[g_\theta(\Pi(Y), y)].

Here ``Pi`` is an optional projection back to the image canvas (reflection,
clipping, or wrapping) used only for the terminal network evaluation.  This is
an exact identity for the terminal weight ``g_\theta \circ Pi`` under the free
Euclidean particle system.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import copy

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
import math

from conditioning_utils import validate_probability_vector
from mnist_weighted_point_cloud import (
    WeightedPointCloudBatch,
    rasterize_weighted_point_clouds,
)
from wasserstein_conditioning_algorithms import sinkhorn_plan

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "GeneratedPointCloudSet",
    "WeightedPointCloudDataset",
    "ImageDataset",
    "TerminalSetClassifier",
    "SmallMnistCNN",
    "train_terminal_set_classifier",
    "evaluate_terminal_set_classifier",
    "predict_terminal_logits",
    "confusion_matrix_from_predictions",
    "terminal_g_accuracy",
    "sample_initial_positions",
    "draw_mass_vectors_from_bank",
    "draw_joint_mass_position_vectors_from_bank",
    "project_positions",
    "estimate_monte_carlo_guided_drift",
    "generate_guided_point_clouds",
    "generate_balanced_synthetic_dataset",
    "train_image_classifier",
    "evaluate_image_classifier",
    "compute_cas_score",
    "sinkhorn_transport_cost",
    "sinkhorn_divergence",
    "pairwise_sinkhorn_divergence",
    "one_nn_leave_one_out_accuracy",
    "coverage_unique_argmin",
    "evaluate_generation_metrics",
]


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedPointCloudSet:
    """Generated weighted point clouds and optional auxiliary outputs."""

    masses: FloatArray
    positions: FloatArray
    labels: IntArray
    images: Optional[FloatArray] = None
    trajectories: Optional[FloatArray] = None

    def __post_init__(self) -> None:
        if self.masses.ndim != 2:
            raise ValueError("masses must have shape (N, K)")
        if self.positions.ndim != 3 or self.positions.shape[:2] != self.masses.shape:
            raise ValueError("positions must have shape (N, K, 2) and match masses")
        if self.positions.shape[2] != 2:
            raise ValueError("positions must have shape (N, K, 2)")
        if self.labels.shape != (self.masses.shape[0],):
            raise ValueError("labels must have shape (N,)")
        if self.images is not None and self.images.shape[0] != self.masses.shape[0]:
            raise ValueError("images must have first dimension N")
        if self.trajectories is not None:
            if self.trajectories.ndim != 4:
                raise ValueError("trajectories must have shape (M+1, N, K, 2)")
            if self.trajectories.shape[1:] != self.positions.shape:
                raise ValueError("trajectories and positions have incompatible shapes")

    def __len__(self) -> int:
        return int(self.masses.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.masses.shape[1])

    def subset(self, indices: Sequence[int] | slice | IntArray) -> "GeneratedPointCloudSet":
        idx = np.arange(len(self))[indices] if isinstance(indices, slice) else np.asarray(indices)
        return GeneratedPointCloudSet(
            masses=np.asarray(self.masses[idx], dtype=np.float64),
            positions=np.asarray(self.positions[idx], dtype=np.float64),
            labels=np.asarray(self.labels[idx], dtype=np.int64),
            images=None if self.images is None else np.asarray(self.images[idx], dtype=np.float64),
            trajectories=None if self.trajectories is None else np.asarray(self.trajectories[:, idx], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Torch datasets and model utilities
# ---------------------------------------------------------------------------


class WeightedPointCloudDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Torch dataset for weighted point clouds."""

    def __init__(
        self,
        masses: np.ndarray,
        positions: np.ndarray,
        labels: np.ndarray,
        *,
        position_jitter_std: float = 0.0,
        projection: str = "clip",
    ) -> None:
        self.masses = np.asarray(masses, dtype=np.float32)
        self.positions = np.asarray(positions, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.position_jitter_std = float(position_jitter_std)
        self.projection = projection

        if self.masses.ndim != 2:
            raise ValueError("masses must have shape (N, K)")
        if self.positions.shape != (*self.masses.shape, 2):
            raise ValueError("positions must have shape (N, K, 2)")
        if self.labels.shape != (self.masses.shape[0],):
            raise ValueError("labels must have shape (N,)")

    def __len__(self) -> int:
        return int(self.masses.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        masses = torch.from_numpy(self.masses[idx]).to(dtype=torch.float32)
        positions = torch.from_numpy(self.positions[idx]).to(dtype=torch.float32)
        if self.position_jitter_std > 0.0:
            positions = positions + self.position_jitter_std * torch.randn_like(positions)
            positions = project_positions(positions, mode=self.projection)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return masses, positions, label


class ImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Torch dataset for grayscale images and integer labels."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, None, :, :]
        if arr.ndim != 4 or arr.shape[1] != 1:
            raise ValueError("images must have shape (N, H, W) or (N, 1, H, W)")
        self.images = arr
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if self.labels.shape != (arr.shape[0],):
            raise ValueError("labels must have shape (N,)")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image = torch.from_numpy(self.images[idx]).to(dtype=torch.float32)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return image, label


class TerminalSetClassifier(nn.Module):
    r"""Simple DeepSets-style classifier for weighted point clouds.

    The network outputs class logits ``f_theta(\mu)`` and defines the positive
    terminal function by

        g_theta(\mu, y) = exp(f_theta(\mu)_y).

    This is sufficient because the h-transform drift only depends on ratios of
    expectations weighted by ``g_theta``.
    """

    def __init__(
        self,
        *,
        point_feature_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if point_feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("point_feature_dim and hidden_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2")

        self.num_classes = int(num_classes)
        self.point_mlp = nn.Sequential(
            nn.Linear(4, point_feature_dim),
            nn.GELU(),
            nn.Linear(point_feature_dim, point_feature_dim),
            nn.GELU(),
            nn.Linear(point_feature_dim, point_feature_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * point_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, masses: Tensor, positions: Tensor) -> Tensor:
        if masses.ndim != 2:
            raise ValueError("masses must have shape (B, K)")
        if positions.ndim != 3 or positions.shape[:2] != masses.shape or positions.shape[2] != 2:
            raise ValueError("positions must have shape (B, K, 2) and match masses")

        log_masses = torch.log(masses.clamp_min(1e-8))
        point_features = torch.cat(
            [positions, masses.unsqueeze(-1), log_masses.unsqueeze(-1)],
            dim=-1,
        )
        h = self.point_mlp(point_features)
        weights = masses.unsqueeze(-1)
        mean = torch.sum(weights * h, dim=1)
        second = torch.sum(weights * h.square(), dim=1)
        std = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-8)
        maximum = torch.max(h, dim=1).values
        pooled = torch.cat([mean, std, maximum], dim=-1)
        return self.head(pooled)

    def log_g(self, masses: Tensor, positions: Tensor, labels: Tensor) -> Tensor:
        r"""Return ``log g_theta(\mu, y)`` for a batch of labels."""
        logits = self(masses, positions)
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("labels must have shape (B,)")
        return logits.gather(1, labels[:, None]).squeeze(1)


class SmallMnistCNN(nn.Module):
    """Small CNN used for the CAS-style score."""

    def __init__(self, *, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("images must have shape (B, 1, H, W)")
        features = self.features(images).flatten(1)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: Optional[str | torch.device]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def _accuracy_from_logits(logits: Tensor, labels: Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).to(dtype=torch.float32).mean().item())


def confusion_matrix_from_predictions(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    num_classes: int = 10,
    normalize: Optional[str] = None,
) -> FloatArray:
    """Build a confusion matrix for diagnostic plots.

    Rows are true labels and columns are predicted labels.  With
    ``normalize='true'`` each row is divided by its row sum, so diagonal entries
    are per-class accuracies.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if normalize not in {None, "true", "pred", "all"}:
        raise ValueError("normalize must be one of None, 'true', 'pred', or 'all'")

    true_arr = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    pred_arr = np.asarray(predicted_labels, dtype=np.int64).reshape(-1)
    if true_arr.shape != pred_arr.shape:
        raise ValueError("true_labels and predicted_labels must have the same shape")

    matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.float64)
    valid = (0 <= true_arr) & (true_arr < num_classes) & (0 <= pred_arr) & (pred_arr < num_classes)
    np.add.at(matrix, (true_arr[valid], pred_arr[valid]), 1.0)

    if normalize == "true":
        denom = matrix.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        denom = matrix.sum(axis=0, keepdims=True)
    elif normalize == "all":
        denom = np.asarray([[matrix.sum()]], dtype=np.float64)
    else:
        return matrix
    return matrix / np.maximum(denom, 1.0)


def predict_terminal_logits(
    model: TerminalSetClassifier,
    masses: np.ndarray,
    positions: np.ndarray,
    *,
    batch_size: int = 512,
    device: Optional[str | torch.device] = None,
) -> FloatArray:
    """Return class logits for a batch of weighted point clouds."""
    model_device = _resolve_device(device)
    dataset = WeightedPointCloudDataset(masses, positions, np.zeros(len(masses), dtype=np.int64))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    was_training = model.training
    model = model.to(model_device)
    model.eval()
    outputs: list[np.ndarray] = []
    for batch_masses, batch_positions, _ in loader:
        logits = model(batch_masses.to(model_device), batch_positions.to(model_device))
        outputs.append(logits.detach().cpu().numpy().astype(np.float64))
    if was_training:
        model.train()
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def evaluate_terminal_set_classifier(
    model: TerminalSetClassifier,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 512,
    device: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    model_device = _resolve_device(device)
    dataset = WeightedPointCloudDataset(masses, positions, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    was_training = model.training
    model = model.to(model_device)
    model.eval()

    total_loss = 0.0
    total_items = 0
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for batch_masses, batch_positions, batch_labels in loader:
        batch_masses = batch_masses.to(model_device)
        batch_positions = batch_positions.to(model_device)
        batch_labels = batch_labels.to(model_device)
        logits = model(batch_masses, batch_positions)
        loss = criterion(logits, batch_labels)
        batch_size_actual = int(batch_labels.shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_items += batch_size_actual
        logits_list.append(logits.detach().cpu().numpy().astype(np.float64))
        labels_list.append(batch_labels.detach().cpu().numpy().astype(np.int64))

    logits_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    predictions = np.argmax(logits_np, axis=1)
    accuracy = float(np.mean(predictions == labels_np))

    if was_training:
        model.train()
    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": accuracy,
        "predictions": predictions,
        "logits": logits_np,
    }


def train_terminal_set_classifier(
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
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train the terminal classifier with cross-entropy on class labels."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model_device = _resolve_device(device)
    model = model.to(model_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = WeightedPointCloudDataset(
        train_masses,
        train_positions,
        train_labels,
        position_jitter_std=position_jitter_std,
        projection="clip",
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
                f"[terminal] epoch {epoch + 1:03d}/{epochs:03d}: "
                f"train loss = {train_loss:.4f}, train acc = {train_accuracy:.4f}{val_message}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def terminal_g_accuracy(
    model: TerminalSetClassifier,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 512,
    device: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    """Accuracy of generated samples under ``argmax_y g_theta(mu, y)``."""
    metrics = evaluate_terminal_set_classifier(
        model,
        masses,
        positions,
        labels,
        batch_size=batch_size,
        device=device,
    )
    logits = metrics["logits"]
    labels_array = np.asarray(labels, dtype=np.int64)
    target_logits = logits[np.arange(len(labels_array)), labels_array]
    target_probabilities = np.exp(target_logits - np.logaddexp.reduce(logits, axis=1))
    return {
        "accuracy": float(metrics["accuracy"]),
        "mean_target_logit": float(np.mean(target_logits)),
        "mean_target_probability": float(np.mean(target_probabilities)),
        "predictions": metrics["predictions"],
    }


@torch.no_grad()
def evaluate_image_classifier(
    model: SmallMnistCNN,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 256,
    device: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    model_device = _resolve_device(device)
    dataset = ImageDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    was_training = model.training
    model = model.to(model_device)
    model.eval()

    total_loss = 0.0
    total_items = 0
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for batch_images, batch_labels in loader:
        batch_images = batch_images.to(model_device)
        batch_labels = batch_labels.to(model_device)
        logits = model(batch_images)
        loss = criterion(logits, batch_labels)
        batch_size_actual = int(batch_labels.shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_items += batch_size_actual
        logits_list.append(logits.detach().cpu().numpy().astype(np.float64))
        labels_list.append(batch_labels.detach().cpu().numpy().astype(np.int64))

    logits_np = np.concatenate(logits_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)
    predictions = np.argmax(logits_np, axis=1)

    if was_training:
        model.train()
    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": float(np.mean(predictions == labels_np)),
        "predictions": predictions,
        "logits": logits_np,
    }


def train_image_classifier(
    model: SmallMnistCNN,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    *,
    val_images: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    model_device = _resolve_device(device)
    model = model.to(model_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(ImageDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)

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

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(model_device)
            batch_labels = batch_labels.to(model_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(batch_labels.shape[0])
            running_loss += float(loss.item()) * batch_size_actual
            running_correct += int((torch.argmax(logits, dim=1) == batch_labels).sum().item())
            running_items += batch_size_actual

        train_loss = running_loss / max(running_items, 1)
        train_accuracy = running_correct / max(running_items, 1)
        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_accuracy))

        if val_images is not None and val_labels is not None:
            val_metrics = evaluate_image_classifier(
                model,
                val_images,
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
                f"[CAS] epoch {epoch + 1:03d}/{epochs:03d}: "
                f"train loss = {train_loss:.4f}, train acc = {train_accuracy:.4f}{val_message}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _rescale_images_for_cas(images: np.ndarray, image_scale: Optional[float]) -> np.ndarray:
    """Apply the same numeric rescaling to CAS train and test images."""
    arr = np.asarray(images, dtype=np.float32)
    if image_scale is None:
        return arr
    scale = float(image_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("image_scale must be positive and finite, or None")
    return arr * scale


def compute_cas_score(
    synthetic_images: np.ndarray,
    synthetic_labels: np.ndarray,
    real_test_images: np.ndarray,
    real_test_labels: np.ndarray,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    image_scale: Optional[float] = 28.0 * 28.0,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute the CAS-style score.

    A fresh image classifier is trained *only* on synthetic samples and then
    evaluated on the real MNIST test set.  The point-cloud images are normalized
    probability measures, so by default both synthetic and real images are
    multiplied by ``28 * 28`` before CAS training/evaluation.  This keeps the
    CNN inputs on a more conventional numerical scale without changing the
    relative comparison between synthetic and real images.
    """
    synthetic_images_for_cas = _rescale_images_for_cas(synthetic_images, image_scale)
    real_test_images_for_cas = _rescale_images_for_cas(real_test_images, image_scale)

    classifier = SmallMnistCNN()
    history = train_image_classifier(
        classifier,
        synthetic_images_for_cas,
        synthetic_labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        verbose=verbose,
    )
    test_metrics = evaluate_image_classifier(
        classifier,
        real_test_images_for_cas,
        real_test_labels,
        batch_size=batch_size,
        device=device,
    )
    return {
        "history": history,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_metrics": test_metrics,
        "classifier": classifier,
        "image_scale": None if image_scale is None else float(image_scale),
    }


# ---------------------------------------------------------------------------
# Guided diffusion
# ---------------------------------------------------------------------------


def project_positions(x: Tensor | np.ndarray, *, mode: str = "reflect") -> Tensor | np.ndarray:
    """Project positions back to the unit square.

    Supported modes are:
        ``'none'``    - leave coordinates unchanged,
        ``'clip'``    - clamp to ``[0, 1]``,
        ``'reflect'`` - reflect at both boundaries,
        ``'wrap'``    - wrap modulo 1 (torus chart).
    """
    if mode not in {"none", "clip", "reflect", "wrap"}:
        raise ValueError("mode must be one of {'none', 'clip', 'reflect', 'wrap'}")

    if isinstance(x, np.ndarray):
        if mode == "none":
            return x
        if mode == "clip":
            return np.clip(x, 0.0, 1.0)
        if mode == "wrap":
            return np.mod(x, 1.0)
        y = np.mod(x, 2.0)
        return np.where(y <= 1.0, y, 2.0 - y)

    if mode == "none":
        return x
    if mode == "clip":
        return torch.clamp(x, 0.0, 1.0)
    if mode == "wrap":
        return torch.remainder(x, 1.0)
    y = torch.remainder(x, 2.0)
    return torch.where(y <= 1.0, y, 2.0 - y)


def sample_initial_positions(
    num_samples: int,
    num_points: int,
    *,
    mode: str = "centered_gaussian",
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.12,
    rng: Optional[np.random.Generator] = None,
) -> FloatArray:
    """Sample initial particle positions for generation."""
    if num_samples <= 0 or num_points <= 0:
        raise ValueError("num_samples and num_points must be positive")
    rng = np.random.default_rng() if rng is None else rng
    center_array = np.asarray(center, dtype=np.float64).reshape(1, 1, 2)

    if mode == "uniform":
        return rng.uniform(0.0, 1.0, size=(num_samples, num_points, 2)).astype(np.float64)
    if mode == "centered_gaussian":
        return (center_array + scale * rng.normal(size=(num_samples, num_points, 2))).astype(np.float64)
    raise ValueError("mode must be 'uniform' or 'centered_gaussian'")


def draw_mass_vectors_from_bank(
    mass_bank: np.ndarray,
    target_labels: np.ndarray,
    *,
    bank_labels: Optional[np.ndarray] = None,
    class_conditional: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> FloatArray:
    """Sample frozen mass vectors from an empirical bank."""
    masses = np.asarray(mass_bank, dtype=np.float64)
    labels = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if masses.ndim != 2:
        raise ValueError("mass_bank must have shape (N, K)")
    rng = np.random.default_rng() if rng is None else rng

    if class_conditional:
        if bank_labels is None:
            raise ValueError("bank_labels are required when class_conditional=True")
        bank_labels_array = np.asarray(bank_labels, dtype=np.int64).reshape(-1)
        if bank_labels_array.shape != (masses.shape[0],):
            raise ValueError("bank_labels must have shape (N,)")
    else:
        bank_labels_array = np.zeros(masses.shape[0], dtype=np.int64)

    out = np.empty((len(labels), masses.shape[1]), dtype=np.float64)
    for label in np.unique(labels):
        mask = labels == label
        if class_conditional:
            candidates = np.flatnonzero(bank_labels_array == label)
            if len(candidates) == 0:
                raise ValueError(f"no mass vectors available for label {label}")
        else:
            candidates = np.arange(masses.shape[0])
        draw = rng.choice(candidates, size=int(np.sum(mask)), replace=True)
        out[mask] = masses[draw]
    return out


def draw_joint_mass_position_vectors_from_bank(
    mass_bank: np.ndarray,
    position_bank: np.ndarray,
    target_labels: np.ndarray,
    *,
    bank_labels: Optional[np.ndarray] = None,
    class_conditional: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> tuple[FloatArray, FloatArray]:
    """Sample masses and positions from the same empirical bank rows.

    This preserves the coupling between atom masses and atom locations that is
    present in the original weighted point-cloud dataset.
    """
    masses = np.asarray(mass_bank, dtype=np.float64)
    positions = np.asarray(position_bank, dtype=np.float64)
    labels = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if masses.ndim != 2:
        raise ValueError("mass_bank must have shape (N, K)")
    if positions.shape != (*masses.shape, 2):
        raise ValueError("position_bank must have shape (N, K, 2) and match mass_bank")
    rng = np.random.default_rng() if rng is None else rng

    if class_conditional:
        if bank_labels is None:
            raise ValueError("bank_labels are required when class_conditional=True")
        bank_labels_array = np.asarray(bank_labels, dtype=np.int64).reshape(-1)
        if bank_labels_array.shape != (masses.shape[0],):
            raise ValueError("bank_labels must have shape (N,)")
    else:
        bank_labels_array = np.zeros(masses.shape[0], dtype=np.int64)

    out_masses = np.empty((len(labels), masses.shape[1]), dtype=np.float64)
    out_positions = np.empty((len(labels), positions.shape[1], 2), dtype=np.float64)
    for label in np.unique(labels):
        mask = labels == label
        if class_conditional:
            candidates = np.flatnonzero(bank_labels_array == label)
            if len(candidates) == 0:
                raise ValueError(f"no joint bank samples available for label {label}")
        else:
            candidates = np.arange(masses.shape[0])
        draw = rng.choice(candidates, size=int(np.sum(mask)), replace=True)
        out_masses[mask] = masses[draw]
        out_positions[mask] = positions[draw]
    return out_masses, out_positions


def _clip_drift_norm(drift: Tensor, max_norm: Optional[float]) -> Tensor:
    if max_norm is None:
        return drift
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    norms = torch.linalg.norm(drift, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norms, max=1.0)
    return drift * scale


def _resolve_drift_clip_norm(
    *,
    horizon: float,
    drift_clip_norm: Optional[float],
    drift_clip_total_displacement: Optional[float],
) -> Optional[float]:
    """Resolve an optional drift clip from either a norm or a path budget."""
    if drift_clip_norm is not None and drift_clip_total_displacement is not None:
        raise ValueError(
            "pass at most one of drift_clip_norm and drift_clip_total_displacement"
        )
    if drift_clip_total_displacement is None:
        return drift_clip_norm
    if horizon <= 0.0 or not np.isfinite(horizon):
        raise ValueError("horizon must be positive and finite")
    if drift_clip_total_displacement <= 0.0 or not np.isfinite(drift_clip_total_displacement):
        raise ValueError("drift_clip_total_displacement must be positive and finite")
    return float(drift_clip_total_displacement / horizon)


"""
@torch.no_grad()
def estimate_monte_carlo_guided_drift(
    model: TerminalSetClassifier,
    masses: Tensor,
    positions: Tensor,
    labels: Tensor,
    tau: float,
    *,
    terminal_mc_samples: int = 64,
    guidance_scale: float = 1.0,
    terminal_projection: str = "reflect",
) -> Tensor:
    #Estimate the h-transform drift by Monte Carlo terminal sampling.
    if terminal_mc_samples <= 0:
        raise ValueError("terminal_mc_samples must be positive")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    if tau <= 1e-12:
        return torch.zeros_like(positions)

    batch_size, num_points, dimension = positions.shape
    sigma = torch.sqrt((2.0 * tau) / masses).unsqueeze(1).unsqueeze(-1)
    noise = torch.randn(
        batch_size,
        terminal_mc_samples,
        num_points,
        dimension,
        device=positions.device,
        dtype=positions.dtype,
    )
    terminal_positions = positions.unsqueeze(1) + sigma * noise
    eval_positions = project_positions(terminal_positions, mode=terminal_projection)

    masses_expanded = masses[:, None, :].expand(batch_size, terminal_mc_samples, num_points)
    logits = model(
        masses_expanded.reshape(batch_size * terminal_mc_samples, num_points),
        eval_positions.reshape(batch_size * terminal_mc_samples, num_points, dimension),
    ).reshape(batch_size, terminal_mc_samples, model.num_classes)

    log_g = logits.gather(2, labels[:, None, None].expand(batch_size, terminal_mc_samples, 1)).squeeze(-1)
    log_g = guidance_scale * log_g
    weights = torch.softmax(log_g, dim=1)
    deltas = terminal_positions - positions.unsqueeze(1)
    drift = torch.sum(weights[..., None, None] * deltas, dim=1) / tau
    return drift
"""

@torch.enable_grad()
def estimate_monte_carlo_guided_drift(
    model,
    masses,
    positions,
    labels,
    tau,
    *,
    terminal_mc_samples=128,
    guidance_scale=3.0,
    terminal_projection="reflect",
):
    if tau <= 1e-12:
        return torch.zeros_like(positions)

    B, K, d = positions.shape

    # Antithetic samples reduce variance further.
    half = (terminal_mc_samples + 1) // 2
    z = torch.randn(B, half, K, d, device=positions.device, dtype=positions.dtype)
    z = torch.cat([z, -z], dim=1)[:, :terminal_mc_samples]

    x = positions.detach().clone().requires_grad_(True)
    sigma = torch.sqrt((2.0 * tau) / masses).unsqueeze(1).unsqueeze(-1)
    y = project_positions(x.unsqueeze(1) + sigma * z, mode=terminal_projection)

    masses_rep = masses[:, None, :].expand(B, terminal_mc_samples, K)
    logits = model(
        masses_rep.reshape(B * terminal_mc_samples, K),
        y.reshape(B * terminal_mc_samples, K, d),
    ).reshape(B, terminal_mc_samples, model.num_classes)

    target_logits = logits.gather(
        2, labels[:, None, None].expand(B, terminal_mc_samples, 1)
    ).squeeze(-1)

    # log u_t approximation
    log_u = torch.logsumexp(guidance_scale * target_logits, dim=1) - math.log(terminal_mc_samples)

    grad_x = torch.autograd.grad(log_u.sum(), x)[0]
    drift = (2.0 / masses.unsqueeze(-1)) * grad_x
    return drift.detach()

@torch.no_grad()
def generate_guided_point_clouds(
    model: TerminalSetClassifier,
    mass_bank: np.ndarray,
    target_labels: np.ndarray,
    *,
    bank_labels: Optional[np.ndarray] = None,
    class_conditional_mass_sampling: bool = True,
    horizon: float = 0.05,
    step_size: float = 5e-4,
    terminal_mc_samples: int = 64,
    guidance_scale: float = 1.0,
    initial_position_mode: str = "centered_gaussian",
    initial_position_scale: float = 0.12,
    state_projection: str = "reflect",
    terminal_projection: str = "reflect",
    drift_clip_norm: Optional[float] = 2.0,
    batch_size: int = 64,
    return_trajectories: bool = False,
    rasterize: bool = True,
    image_size: int = 28,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedPointCloudSet:
    """Generate class-conditional weighted point clouds by guided diffusion."""
    if horizon <= 0.0 or not np.isfinite(horizon):
        raise ValueError("horizon must be positive and finite")
    if step_size <= 0.0 or not np.isfinite(step_size):
        raise ValueError("step_size must be positive and finite")
    ratio = horizon / step_size
    num_steps = int(round(ratio))
    if num_steps <= 0 or not np.isclose(ratio, num_steps, atol=1e-10, rtol=1e-10):
        raise ValueError("horizon / step_size must be an integer")

    labels = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng() if rng is None else rng
    masses_np = draw_mass_vectors_from_bank(
        mass_bank,
        labels,
        bank_labels=bank_labels,
        class_conditional=class_conditional_mass_sampling,
        rng=rng,
    )
    num_samples, num_points = masses_np.shape
    initial_positions_np = sample_initial_positions(
        num_samples,
        num_points,
        mode=initial_position_mode,
        scale=initial_position_scale,
        rng=rng,
    )
    initial_positions_np = np.asarray(
        project_positions(initial_positions_np, mode=state_projection),
        dtype=np.float64,
    )

    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()

    masses = torch.from_numpy(masses_np).to(device=model_device, dtype=torch.float32)
    positions = torch.from_numpy(initial_positions_np).to(device=model_device, dtype=torch.float32)
    label_tensor = torch.from_numpy(labels).to(device=model_device, dtype=torch.long)

    trajectories: Optional[np.ndarray]
    if return_trajectories:
        trajectories = np.empty((num_steps + 1, num_samples, num_points, 2), dtype=np.float64)
        trajectories[0] = initial_positions_np
    else:
        trajectories = None

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    times = np.linspace(0.0, horizon, num_steps + 1, dtype=np.float64)

    for step in range(num_steps):
        tau = float(horizon - times[step])
        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            batch_masses = masses[start:stop]
            batch_positions = positions[start:stop]
            batch_labels = label_tensor[start:stop]

            with torch.enable_grad():
                drift = estimate_monte_carlo_guided_drift(
                    model,
                    batch_masses,
                    batch_positions,
                    batch_labels,
                    tau,
                    terminal_mc_samples=terminal_mc_samples,
                    guidance_scale=guidance_scale,
                    terminal_projection=terminal_projection,
                )
            drift = _clip_drift_norm(drift, drift_clip_norm)
            noise_scale = torch.sqrt((2.0 * step_size) / batch_masses).unsqueeze(-1)
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
def generate_balanced_synthetic_dataset(
    model: TerminalSetClassifier,
    mass_bank: np.ndarray,
    *,
    bank_labels: Optional[np.ndarray] = None,
    num_per_class: int,
    class_conditional_mass_sampling: bool = True,
    horizon: float = 0.05,
    step_size: float = 5e-4,
    terminal_mc_samples: int = 64,
    guidance_scale: float = 1.0,
    initial_position_mode: str = "centered_gaussian",
    initial_position_scale: float = 0.12,
    state_projection: str = "reflect",
    terminal_projection: str = "reflect",
    drift_clip_norm: Optional[float] = 2.0,
    batch_size: int = 64,
    rasterize: bool = True,
    image_size: int = 28,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedPointCloudSet:
    labels = np.repeat(np.arange(model.num_classes, dtype=np.int64), num_per_class)
    return generate_guided_point_clouds(
        model,
        mass_bank,
        labels,
        bank_labels=bank_labels,
        class_conditional_mass_sampling=class_conditional_mass_sampling,
        horizon=horizon,
        step_size=step_size,
        terminal_mc_samples=terminal_mc_samples,
        guidance_scale=guidance_scale,
        initial_position_mode=initial_position_mode,
        initial_position_scale=initial_position_scale,
        state_projection=state_projection,
        terminal_projection=terminal_projection,
        drift_clip_norm=drift_clip_norm,
        batch_size=batch_size,
        return_trajectories=False,
        rasterize=rasterize,
        image_size=image_size,
        device=device,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Sinkhorn metrics
# ---------------------------------------------------------------------------


def _point_cloud_cost_matrix(a_positions: np.ndarray, b_positions: np.ndarray) -> FloatArray:
    a = np.asarray(a_positions, dtype=np.float64)
    b = np.asarray(b_positions, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("positions must have shape (K, 2)")
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def sinkhorn_transport_cost(
    a_masses: np.ndarray,
    a_positions: np.ndarray,
    b_masses: np.ndarray,
    b_positions: np.ndarray,
    *,
    epsilon: float = 0.02,
    iterations: int = 100,
    tol: Optional[float] = 1e-6,
) -> float:
    """Entropic transport cost between two atomic measures."""
    a = validate_probability_vector(a_masses, name="a_masses", normalize=True)
    b = validate_probability_vector(b_masses, name="b_masses", normalize=True)
    cost = _point_cloud_cost_matrix(a_positions, b_positions)
    plan = sinkhorn_plan(cost, a, b, epsilon=epsilon, iterations=iterations, tol=tol)
    return float(np.sum(plan * cost))


def sinkhorn_divergence(
    a_masses: np.ndarray,
    a_positions: np.ndarray,
    b_masses: np.ndarray,
    b_positions: np.ndarray,
    *,
    epsilon: float = 0.02,
    iterations: int = 100,
    tol: Optional[float] = 1e-6,
) -> float:
    """Sinkhorn divergence based on the entropic transport cost."""
    ab = sinkhorn_transport_cost(
        a_masses,
        a_positions,
        b_masses,
        b_positions,
        epsilon=epsilon,
        iterations=iterations,
        tol=tol,
    )
    aa = sinkhorn_transport_cost(
        a_masses,
        a_positions,
        a_masses,
        a_positions,
        epsilon=epsilon,
        iterations=iterations,
        tol=tol,
    )
    bb = sinkhorn_transport_cost(
        b_masses,
        b_positions,
        b_masses,
        b_positions,
        epsilon=epsilon,
        iterations=iterations,
        tol=tol,
    )
    return float(ab - 0.5 * aa - 0.5 * bb)


def pairwise_sinkhorn_divergence(
    masses_a: np.ndarray,
    positions_a: np.ndarray,
    masses_b: np.ndarray,
    positions_b: np.ndarray,
    *,
    epsilon: float = 0.02,
    iterations: int = 100,
    tol: Optional[float] = 1e-6,
) -> FloatArray:
    """Pairwise Sinkhorn divergence matrix between two collections."""
    a_m = np.asarray(masses_a, dtype=np.float64)
    a_x = np.asarray(positions_a, dtype=np.float64)
    b_m = np.asarray(masses_b, dtype=np.float64)
    b_x = np.asarray(positions_b, dtype=np.float64)

    if a_m.ndim != 2 or a_x.shape != (*a_m.shape, 2):
        raise ValueError("masses_a and positions_a have incompatible shapes")
    if b_m.ndim != 2 or b_x.shape != (*b_m.shape, 2):
        raise ValueError("masses_b and positions_b have incompatible shapes")

    n_a = a_m.shape[0]
    n_b = b_m.shape[0]
    self_a = np.empty(n_a, dtype=np.float64)
    self_b = np.empty(n_b, dtype=np.float64)
    for i in range(n_a):
        self_a[i] = sinkhorn_transport_cost(
            a_m[i], a_x[i], a_m[i], a_x[i], epsilon=epsilon, iterations=iterations, tol=tol
        )
    for j in range(n_b):
        self_b[j] = sinkhorn_transport_cost(
            b_m[j], b_x[j], b_m[j], b_x[j], epsilon=epsilon, iterations=iterations, tol=tol
        )

    out = np.empty((n_a, n_b), dtype=np.float64)
    for i in range(n_a):
        for j in range(n_b):
            ab = sinkhorn_transport_cost(
                a_m[i],
                a_x[i],
                b_m[j],
                b_x[j],
                epsilon=epsilon,
                iterations=iterations,
                tol=tol,
            )
            out[i, j] = ab - 0.5 * self_a[i] - 0.5 * self_b[j]
    return out


def one_nn_leave_one_out_accuracy(
    real_masses: np.ndarray,
    real_positions: np.ndarray,
    generated_masses: np.ndarray,
    generated_positions: np.ndarray,
    *,
    epsilon: float = 0.02,
    iterations: int = 100,
    tol: Optional[float] = 1e-6,
) -> dict[str, Any]:
    """1-NN leave-one-out accuracy on the real-vs-generated two-sample problem.

    If the two distributions match perfectly, the expected accuracy is close to
    50%.  Values much larger than 50% indicate that real and generated samples
    are easy to separate.
    """
    rr = pairwise_sinkhorn_divergence(
        real_masses,
        real_positions,
        real_masses,
        real_positions,
        epsilon=epsilon,
        iterations=iterations,
        tol=tol,
    )
    gg = pairwise_sinkhorn_divergence(
        generated_masses,
        generated_positions,
        generated_masses,
        generated_positions,
        epsilon=epsilon,
        iterations=iterations,
        tol=tol,
    )
    rg = pairwise_sinkhorn_divergence(
        real_masses,
        real_positions,
        generated_masses,
        generated_positions,
        epsilon=epsilon,
        iterations=iterations,
        tol=tol,
    )
    n_real = rr.shape[0]
    n_gen = gg.shape[0]

    rr = rr.copy()
    gg = gg.copy()
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(gg, np.inf)

    upper = np.concatenate([rr, rg], axis=1)
    lower = np.concatenate([rg.T, gg], axis=1)
    full = np.concatenate([upper, lower], axis=0)
    domain = np.concatenate([np.zeros(n_real, dtype=np.int64), np.ones(n_gen, dtype=np.int64)])

    nearest = np.argmin(full, axis=1)
    predictions = domain[nearest]
    accuracy = float(np.mean(predictions == domain))
    real_accuracy = float(np.mean(predictions[:n_real] == 0))
    generated_accuracy = float(np.mean(predictions[n_real:] == 1))
    return {
        "accuracy": accuracy,
        "real_accuracy": real_accuracy,
        "generated_accuracy": generated_accuracy,
        "distance_matrix": full,
    }


def coverage_unique_argmin(
    real_masses: np.ndarray,
    real_positions: np.ndarray,
    generated_masses: np.ndarray,
    generated_positions: np.ndarray,
    *,
    epsilon: float = 0.02,
    iterations: int = 100,
    tol: Optional[float] = 1e-6,
) -> dict[str, Any]:
    """Coverage metric based on unique nearest real neighbours."""
    gr = pairwise_sinkhorn_divergence(
        generated_masses,
        generated_positions,
        real_masses,
        real_positions,
        epsilon=epsilon,
        iterations=iterations,
        tol=tol,
    )
    nearest_real = np.argmin(gr, axis=1)
    unique_hits = np.unique(nearest_real)
    return {
        "coverage": float(len(unique_hits) / max(len(real_masses), 1)),
        "unique_argmins": unique_hits,
        "distance_matrix": gr,
    }


# ---------------------------------------------------------------------------
# High-level evaluation
# ---------------------------------------------------------------------------


def _subset_by_label(
    masses: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    label: int,
    *,
    max_count: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(np.asarray(labels, dtype=np.int64) == label)
    if len(indices) == 0:
        raise ValueError(f"no samples available for label {label}")
    if max_count is not None and len(indices) > max_count:
        indices = rng.choice(indices, size=max_count, replace=False)
    return masses[indices], positions[indices]


def evaluate_generation_metrics(
    terminal_model: TerminalSetClassifier,
    generated: GeneratedPointCloudSet,
    real_reference: WeightedPointCloudBatch,
    real_test_images: np.ndarray,
    real_test_labels: np.ndarray,
    *,
    cas_epochs: int = 10,
    cas_batch_size: int = 128,
    cas_lr: float = 1e-3,
    cas_image_scale: Optional[float] = 28.0 * 28.0,
    sinkhorn_epsilon: float = 0.02,
    sinkhorn_iterations: int = 50,
    sinkhorn_subsample_per_class: int = 64,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute the metrics requested for the MNIST point-cloud experiment."""
    rng = np.random.default_rng() if rng is None else rng
    g_metrics = terminal_g_accuracy(
        terminal_model,
        generated.masses,
        generated.positions,
        generated.labels,
        device=device,
    )

    synthetic_images = (
        generated.images
        if generated.images is not None
        else rasterize_weighted_point_clouds(generated.masses, generated.positions)
    )
    cas_metrics = compute_cas_score(
        synthetic_images,
        generated.labels,
        real_test_images,
        real_test_labels,
        epochs=cas_epochs,
        batch_size=cas_batch_size,
        lr=cas_lr,
        image_scale=cas_image_scale,
        device=device,
        verbose=verbose,
    )

    per_label: dict[int, dict[str, float]] = {}
    one_nn_values = []
    coverage_values = []
    real_to_generated_values = []
    generated_to_real_values = []
    diversity_values = []

    for label in np.unique(generated.labels):
        gen_m, gen_x = _subset_by_label(
            generated.masses,
            generated.positions,
            generated.labels,
            int(label),
            max_count=sinkhorn_subsample_per_class,
            rng=rng,
        )
        real_m, real_x = _subset_by_label(
            real_reference.masses,
            real_reference.positions,
            np.asarray(real_reference.labels, dtype=np.int64),
            int(label),
            max_count=sinkhorn_subsample_per_class,
            rng=rng,
        )

        one_nn = one_nn_leave_one_out_accuracy(
            real_m,
            real_x,
            gen_m,
            gen_x,
            epsilon=sinkhorn_epsilon,
            iterations=sinkhorn_iterations,
        )
        coverage = coverage_unique_argmin(
            real_m,
            real_x,
            gen_m,
            gen_x,
            epsilon=sinkhorn_epsilon,
            iterations=sinkhorn_iterations,
        )
        rg = pairwise_sinkhorn_divergence(
            real_m,
            real_x,
            gen_m,
            gen_x,
            epsilon=sinkhorn_epsilon,
            iterations=sinkhorn_iterations,
        )
        gg = pairwise_sinkhorn_divergence(
            gen_m,
            gen_x,
            gen_m,
            gen_x,
            epsilon=sinkhorn_epsilon,
            iterations=sinkhorn_iterations,
        )
        np.fill_diagonal(gg, np.inf)

        mean_real_to_generated = float(np.mean(np.min(rg, axis=1)))
        mean_generated_to_real = float(np.mean(np.min(rg, axis=0)))
        mean_generated_diversity = float(np.mean(np.min(gg, axis=1)))

        per_label[int(label)] = {
            "one_nn_accuracy": float(one_nn["accuracy"]),
            "coverage": float(coverage["coverage"]),
            "mean_real_to_generated_sinkhorn": mean_real_to_generated,
            "mean_generated_to_real_sinkhorn": mean_generated_to_real,
            "mean_generated_diversity": mean_generated_diversity,
        }
        one_nn_values.append(one_nn["accuracy"])
        coverage_values.append(coverage["coverage"])
        real_to_generated_values.append(mean_real_to_generated)
        generated_to_real_values.append(mean_generated_to_real)
        diversity_values.append(mean_generated_diversity)

    return {
        "g_accuracy": float(g_metrics["accuracy"]),
        "g_mean_target_probability": float(g_metrics["mean_target_probability"]),
        "cas_accuracy": float(cas_metrics["test_accuracy"]),
        "cas_image_scale": None if cas_image_scale is None else float(cas_image_scale),
        "one_nn_accuracy_macro": float(np.mean(one_nn_values)),
        "coverage_macro": float(np.mean(coverage_values)),
        "mean_real_to_generated_sinkhorn_macro": float(np.mean(real_to_generated_values)),
        "mean_generated_to_real_sinkhorn_macro": float(np.mean(generated_to_real_values)),
        "mean_generated_diversity_macro": float(np.mean(diversity_values)),
        "per_label": per_label,
        "cas_details": cas_metrics,
    }
