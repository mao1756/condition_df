"""Smoke checks for MNIST-CP contour utilities.

Run with
    .venv\\Scripts\\python.exe -m tests.test_smoke_mnist_cp
"""

from __future__ import annotations

import numpy as np
import torch

from mnist.conditioned_diffusion import GeneratedPointCloudSet, TerminalSetClassifier
from mnist.mnist_cp import (
    balanced_indices_by_label,
    chamfer_distance,
    coverage_unique_argmin_chamfer,
    evaluate_mnist_cp_generation_metrics,
    mnist_cp_points_to_unit_square,
    one_nn_leave_one_out_chamfer,
    pairwise_chamfer_distance_matrix,
    uniform_point_cloud_masses,
    unit_square_to_mnist_cp_points,
)
from mnist.weighted_point_cloud import WeightedPointCloudBatch


def _circle_clouds(num_per_class: int = 3, num_points: int = 24) -> WeightedPointCloudBatch:
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    clouds = []
    labels = []
    for label, radius in [(0, 0.20), (1, 0.32)]:
        for i in range(num_per_class):
            center = np.asarray([0.42 + 0.04 * label, 0.50 + 0.01 * i])
            points = center + radius * np.column_stack([np.cos(theta), np.sin(theta)])
            clouds.append(points)
            labels.append(label)
    positions = np.asarray(clouds, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    return WeightedPointCloudBatch(
        masses=uniform_point_cloud_masses(len(labels_arr), num_points, dtype=np.float64),
        positions=positions,
        labels=labels_arr,
    )


def test_coordinate_conversion_and_balanced_indices() -> None:
    original = np.asarray([[[-1.0, -0.5], [0.0, 1.0], [1.0, 0.25]]], dtype=np.float64)
    unit = mnist_cp_points_to_unit_square(original)
    restored = unit_square_to_mnist_cp_points(unit)
    assert np.allclose(restored, original)
    assert np.all((0.0 <= unit) & (unit <= 1.0))

    labels = np.repeat(np.arange(2), 5)
    first, second = balanced_indices_by_label(labels, [2, 2], seed=4, num_classes=2)
    assert len(first) == 4
    assert len(second) == 4
    assert set(first).isdisjoint(set(second))
    assert np.array_equal(np.bincount(labels[first], minlength=2), np.asarray([2, 2]))


def test_chamfer_metrics_smoke() -> None:
    batch = _circle_clouds(num_per_class=3, num_points=24)
    same = chamfer_distance(batch.positions[0], batch.positions[0])
    shifted = chamfer_distance(batch.positions[0], batch.positions[1])
    assert same < 1e-12
    assert shifted > same

    mat = pairwise_chamfer_distance_matrix(batch.positions[:2], batch.positions[2:4])
    assert mat.shape == (2, 2)
    assert np.all(np.isfinite(mat))

    one_nn = one_nn_leave_one_out_chamfer(batch.positions[:3], batch.positions[3:6])
    coverage = coverage_unique_argmin_chamfer(batch.positions[:3], batch.positions[3:6])
    assert 0.0 <= one_nn["accuracy"] <= 1.0
    assert 0.0 <= coverage["coverage"] <= 1.0


def test_generation_metrics_smoke() -> None:
    torch.manual_seed(0)
    real = _circle_clouds(num_per_class=3, num_points=16)
    generated = GeneratedPointCloudSet(
        masses=real.masses.copy(),
        positions=real.positions.copy(),
        labels=real.labels.copy(),
    )
    classifier = TerminalSetClassifier(point_feature_dim=8, hidden_dim=16, num_classes=2, dropout=0.0)
    metrics = evaluate_mnist_cp_generation_metrics(
        classifier,
        generated,
        real,
        chamfer_subsample_per_class=2,
        device="cpu",
        rng=np.random.default_rng(1),
    )
    assert "aux_point_cloud_classifier_accuracy" in metrics
    assert "one_nn_chamfer_accuracy_macro" in metrics
    assert "coverage_chamfer_macro" in metrics
    assert set(metrics["per_label"].keys()) == {0, 1}


if __name__ == "__main__":
    test_coordinate_conversion_and_balanced_indices()
    test_chamfer_metrics_smoke()
    test_generation_metrics_smoke()
    print("All mnist_cp smoke tests passed.")
