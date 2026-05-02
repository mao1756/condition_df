from __future__ import annotations

r"""MNIST-CP contour point-cloud utilities.

The script :mod:`mnist.generate_mnist_cp` saves contour point clouds in the
paper-style coordinate system ``[-1, 1]^2`` with a fixed number of points per
sample.  The score-matching code in :mod:`mnist.score_matching` expects atomic
measures in the unit square, so this module provides a small adapter:

    points in ``[-1, 1]^2``  ->  uniform-mass atoms in ``[0, 1]^2``.

The metrics below are contour-native.  Since MNIST-CP uses uniformly sampled
outline points rather than grayscale pixel masses, nearest-neighbour Chamfer
statistics are often more interpretable than raster/CAS scores.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from mnist.conditioned_diffusion import (
    GeneratedPointCloudSet,
    TerminalSetClassifier,
    terminal_g_accuracy,
)
from mnist.weighted_point_cloud import WeightedPointCloudBatch

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

try:  # pragma: no cover - exercised when scipy is available in the environment.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover - fallback keeps this module dependency-light.
    _KDTree = None

__all__ = [
    "MnistCPSplit",
    "mnist_cp_points_to_unit_square",
    "unit_square_to_mnist_cp_points",
    "uniform_point_cloud_masses",
    "balanced_indices_by_label",
    "load_mnist_cp_splits",
    "chamfer_distance",
    "pairwise_chamfer_distance_matrix",
    "one_nn_leave_one_out_chamfer",
    "coverage_unique_argmin_chamfer",
    "evaluate_mnist_cp_generation_metrics",
]


@dataclass(frozen=True)
class MnistCPSplit:
    """Balanced train/validation/test splits for MNIST-CP."""

    train: WeightedPointCloudBatch
    val: WeightedPointCloudBatch
    test: WeightedPointCloudBatch
    normalization: str
    source_path: Path


# ---------------------------------------------------------------------------
# Dataset loading and conversion
# ---------------------------------------------------------------------------


def mnist_cp_points_to_unit_square(points: np.ndarray) -> np.ndarray:
    """Map MNIST-CP points from ``[-1, 1]^2`` to the unit square."""
    arr = np.asarray(points)
    if arr.shape[-1] != 2:
        raise ValueError("points must have final coordinate dimension 2")
    return 0.5 * (arr + 1.0)


def unit_square_to_mnist_cp_points(positions: np.ndarray) -> np.ndarray:
    """Map unit-square positions back to the MNIST-CP ``[-1, 1]^2`` convention."""
    arr = np.asarray(positions)
    if arr.shape[-1] != 2:
        raise ValueError("positions must have final coordinate dimension 2")
    return 2.0 * arr - 1.0


def uniform_point_cloud_masses(
    num_clouds: int,
    num_points: int,
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return uniform mass vectors for fixed-cardinality contour clouds."""
    if num_clouds <= 0 or num_points <= 0:
        raise ValueError("num_clouds and num_points must be positive")
    return np.full((int(num_clouds), int(num_points)), 1.0 / float(num_points), dtype=dtype)


def balanced_indices_by_label(
    labels: np.ndarray,
    counts: Sequence[int],
    *,
    seed: int = 0,
    num_classes: int = 10,
) -> tuple[np.ndarray, ...]:
    """Draw disjoint balanced index sets for several splits.

    Parameters
    ----------
    labels:
        Integer label vector.
    counts:
        Number of examples per class for each returned split.
    seed:
        Random seed controlling the within-class permutations.
    num_classes:
        Number of expected digit classes.
    """
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts_arr = [int(c) for c in counts]
    if any(c < 0 for c in counts_arr):
        raise ValueError("split counts must be non-negative")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    rng = np.random.default_rng(seed)
    split_parts: list[list[np.ndarray]] = [[] for _ in counts_arr]
    required = int(sum(counts_arr))

    for label in range(num_classes):
        class_idx = np.flatnonzero(labels_arr == label)
        if len(class_idx) < required:
            raise ValueError(
                f"label {label} has only {len(class_idx)} examples, but {required} are required"
            )
        class_idx = rng.permutation(class_idx)
        start = 0
        for split_id, count in enumerate(counts_arr):
            stop = start + count
            split_parts[split_id].append(class_idx[start:stop])
            start = stop

    output = []
    for parts in split_parts:
        if parts:
            idx = np.concatenate(parts)
            rng.shuffle(idx)
        else:
            idx = np.empty(0, dtype=np.int64)
        output.append(idx.astype(np.int64, copy=False))
    return tuple(output)


def _maybe_subsample_points(
    points: np.ndarray,
    *,
    num_points: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError("points must have shape (N, K, 2)")
    original_points = int(points.shape[1])
    if num_points is None or int(num_points) == original_points:
        return points
    if int(num_points) <= 0:
        raise ValueError("num_points must be positive")
    if int(num_points) > original_points:
        raise ValueError("num_points cannot exceed the saved MNIST-CP point count")

    num_points_resolved = int(num_points)
    out = np.empty((points.shape[0], num_points_resolved, 2), dtype=points.dtype)
    for i in range(points.shape[0]):
        idx = rng.choice(original_points, size=num_points_resolved, replace=False)
        out[i] = points[i, idx]
    return out


def _as_normalized_images(images: Optional[np.ndarray], *, dtype: np.dtype | type) -> Optional[np.ndarray]:
    if images is None:
        return None
    arr = np.asarray(images)
    if arr.dtype.kind in {"u", "i"}:
        return (arr.astype(dtype) / 255.0).astype(dtype, copy=False)
    return arr.astype(dtype, copy=False)


def _make_batch(
    points_minus_one_one: np.ndarray,
    labels: np.ndarray,
    *,
    images: Optional[np.ndarray],
    num_points: Optional[int],
    point_dtype: np.dtype | type,
    mass_dtype: np.dtype | type,
    rng: np.random.Generator,
) -> WeightedPointCloudBatch:
    points = _maybe_subsample_points(points_minus_one_one, num_points=num_points, rng=rng)
    positions = mnist_cp_points_to_unit_square(points).astype(point_dtype, copy=False)
    masses = uniform_point_cloud_masses(len(positions), positions.shape[1], dtype=mass_dtype)
    image_arr = _as_normalized_images(images, dtype=point_dtype)
    return WeightedPointCloudBatch(
        masses=masses,
        positions=positions,
        labels=np.asarray(labels, dtype=np.int64),
        images=image_arr,
        pixel_indices=None,
    )


def load_mnist_cp_splits(
    npz_path: str | Path,
    *,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    num_points: Optional[int] = None,
    seed: int = 0,
    load_images: bool = True,
    point_dtype: np.dtype | type = np.float32,
    mass_dtype: np.dtype | type = np.float32,
) -> MnistCPSplit:
    """Load balanced MNIST-CP splits as uniform-mass point-cloud batches.

    The bundled ``mnist_cp.npz`` currently contains one 60k split generated from
    the OpenML ARFF stream.  This helper draws disjoint balanced train, val, and
    reference/test splits from that pool.  Set ``num_points`` below the saved
    cardinality to uniformly subsample each contour for faster score-matching
    experiments; leave it as ``None`` to use all 800 points.
    """
    if train_per_class <= 0 or val_per_class <= 0 or test_per_class <= 0:
        raise ValueError("train_per_class, val_per_class, and test_per_class must be positive")

    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        if "points" not in data or "labels" not in data:
            raise ValueError(f"{path} must contain 'points' and 'labels' arrays")
        labels_all = np.asarray(data["labels"], dtype=np.int64)
        train_idx, val_idx, test_idx = balanced_indices_by_label(
            labels_all,
            [train_per_class, val_per_class, test_per_class],
            seed=seed,
            num_classes=10,
        )
        points_all = np.asarray(data["points"])
        images_all = np.asarray(data["images"]) if load_images and "images" in data else None
        normalization_arr = data["normalization"] if "normalization" in data else np.asarray(["unknown"])
        normalization = str(np.asarray(normalization_arr).reshape(-1)[0])

        def build(indices: np.ndarray, offset: int) -> WeightedPointCloudBatch:
            split_images = None if images_all is None else images_all[indices]
            return _make_batch(
                points_all[indices],
                labels_all[indices],
                images=split_images,
                num_points=num_points,
                point_dtype=point_dtype,
                mass_dtype=mass_dtype,
                rng=np.random.default_rng(seed + offset),
            )

        return MnistCPSplit(
            train=build(train_idx, 11),
            val=build(val_idx, 17),
            test=build(test_idx, 23),
            normalization=normalization,
            source_path=path,
        )


# ---------------------------------------------------------------------------
# Contour-native Chamfer metrics
# ---------------------------------------------------------------------------


def _validate_positions(positions: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (K, 2)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _nearest_squared_distances_numpy(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    mins = []
    for start in range(0, query.shape[0], int(chunk_size)):
        chunk = query[start : start + int(chunk_size)]
        diff = chunk[:, None, :] - reference[None, :, :]
        mins.append(np.sum(diff * diff, axis=-1).min(axis=1))
    return np.concatenate(mins, axis=0)


def _nearest_squared_distances(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    reference_tree: Any = None,
    chunk_size: int = 256,
) -> np.ndarray:
    if reference_tree is not None:
        distances = reference_tree.query(query, k=1)[0]
        return np.square(distances)
    return _nearest_squared_distances_numpy(query, reference, chunk_size=chunk_size)


def chamfer_distance(
    a_positions: np.ndarray,
    b_positions: np.ndarray,
    *,
    squared: bool = True,
    chunk_size: int = 256,
) -> float:
    """Symmetric Chamfer distance between two unweighted contour point clouds."""
    a = _validate_positions(a_positions, name="a_positions")
    b = _validate_positions(b_positions, name="b_positions")
    if _KDTree is not None:
        tree_a = _KDTree(a)
        tree_b = _KDTree(b)
    else:
        tree_a = None
        tree_b = None
    a_to_b = _nearest_squared_distances(a, b, reference_tree=tree_b, chunk_size=chunk_size)
    b_to_a = _nearest_squared_distances(b, a, reference_tree=tree_a, chunk_size=chunk_size)
    if squared:
        return 0.5 * float(np.mean(a_to_b) + np.mean(b_to_a))
    return 0.5 * float(np.mean(np.sqrt(a_to_b)) + np.mean(np.sqrt(b_to_a)))


def pairwise_chamfer_distance_matrix(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    *,
    squared: bool = True,
    chunk_size: int = 256,
) -> np.ndarray:
    """Pairwise symmetric Chamfer matrix between two point-cloud collections."""
    a = np.asarray(positions_a, dtype=np.float64)
    b = np.asarray(positions_b, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 2:
        raise ValueError("positions_a must have shape (N, K, 2)")
    if b.ndim != 3 or b.shape[2] != 2:
        raise ValueError("positions_b must have shape (M, K, 2)")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("positions contain non-finite values")

    trees_a = [_KDTree(a_i) for a_i in a] if _KDTree is not None else [None] * len(a)
    trees_b = [_KDTree(b_j) for b_j in b] if _KDTree is not None else [None] * len(b)
    out = np.empty((len(a), len(b)), dtype=np.float64)
    for i, a_i in enumerate(a):
        for j, b_j in enumerate(b):
            a_to_b = _nearest_squared_distances(
                a_i,
                b_j,
                reference_tree=trees_b[j],
                chunk_size=chunk_size,
            )
            b_to_a = _nearest_squared_distances(
                b_j,
                a_i,
                reference_tree=trees_a[i],
                chunk_size=chunk_size,
            )
            if squared:
                out[i, j] = 0.5 * (float(np.mean(a_to_b)) + float(np.mean(b_to_a)))
            else:
                out[i, j] = 0.5 * (
                    float(np.mean(np.sqrt(a_to_b))) + float(np.mean(np.sqrt(b_to_a)))
                )
    return out


def one_nn_leave_one_out_chamfer(
    real_positions: np.ndarray,
    generated_positions: np.ndarray,
    *,
    squared: bool = True,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """1-NN real-vs-generated two-sample accuracy using Chamfer distance.

    Values close to 0.5 indicate that the two samples are hard to distinguish by
    nearest-neighbour Chamfer distance.  Larger values mean the samples are more
    easily separable.
    """
    rr = pairwise_chamfer_distance_matrix(
        real_positions,
        real_positions,
        squared=squared,
        chunk_size=chunk_size,
    )
    gg = pairwise_chamfer_distance_matrix(
        generated_positions,
        generated_positions,
        squared=squared,
        chunk_size=chunk_size,
    )
    rg = pairwise_chamfer_distance_matrix(
        real_positions,
        generated_positions,
        squared=squared,
        chunk_size=chunk_size,
    )
    rr = rr.copy()
    gg = gg.copy()
    np.fill_diagonal(rr, np.inf)
    np.fill_diagonal(gg, np.inf)

    full = np.concatenate(
        [np.concatenate([rr, rg], axis=1), np.concatenate([rg.T, gg], axis=1)],
        axis=0,
    )
    domain = np.concatenate(
        [np.zeros(rr.shape[0], dtype=np.int64), np.ones(gg.shape[0], dtype=np.int64)]
    )
    nearest = np.argmin(full, axis=1)
    predictions = domain[nearest]
    return {
        "accuracy": float(np.mean(predictions == domain)),
        "real_accuracy": float(np.mean(predictions[: rr.shape[0]] == 0)),
        "generated_accuracy": float(np.mean(predictions[rr.shape[0] :] == 1)),
        "distance_matrix": full,
    }


def coverage_unique_argmin_chamfer(
    real_positions: np.ndarray,
    generated_positions: np.ndarray,
    *,
    squared: bool = True,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Coverage based on unique nearest real neighbours under Chamfer distance."""
    gr = pairwise_chamfer_distance_matrix(
        generated_positions,
        real_positions,
        squared=squared,
        chunk_size=chunk_size,
    )
    nearest_real = np.argmin(gr, axis=1)
    unique_hits = np.unique(nearest_real)
    return {
        "coverage": float(len(unique_hits) / max(len(real_positions), 1)),
        "unique_argmins": unique_hits,
        "distance_matrix": gr,
    }


def _subset_positions_by_label(
    positions: np.ndarray,
    labels: np.ndarray,
    label: int,
    *,
    max_count: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    idx = np.flatnonzero(labels_arr == int(label))
    if len(idx) == 0:
        raise ValueError(f"no samples available for label {label}")
    if max_count is not None and len(idx) > int(max_count):
        idx = rng.choice(idx, size=int(max_count), replace=False)
    return np.asarray(positions[idx], dtype=np.float64)


def evaluate_mnist_cp_generation_metrics(
    eval_classifier: TerminalSetClassifier,
    generated: GeneratedPointCloudSet,
    real_reference: WeightedPointCloudBatch,
    *,
    chamfer_subsample_per_class: int = 24,
    squared_chamfer: bool = True,
    chunk_size: int = 256,
    device: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Evaluate class fidelity and contour quality for MNIST-CP generation.

    This mirrors the spirit of Example 7's two-sample evaluation, but replaces
    Sinkhorn/CAS with Chamfer statistics that are natural for uniform contour
    point clouds.
    """
    if real_reference.labels is None:
        raise ValueError("real_reference must include labels")
    if chamfer_subsample_per_class <= 1:
        raise ValueError("chamfer_subsample_per_class must be at least 2")

    rng = np.random.default_rng() if rng is None else rng
    classifier_metrics = terminal_g_accuracy(
        eval_classifier,
        generated.masses,
        generated.positions,
        generated.labels,
        device=device,
    )

    per_label: dict[int, dict[str, float]] = {}
    one_nn_values = []
    coverage_values = []
    real_to_generated_values = []
    generated_to_real_values = []
    diversity_values = []

    real_labels = np.asarray(real_reference.labels, dtype=np.int64)
    for label in np.unique(generated.labels):
        gen_x = _subset_positions_by_label(
            generated.positions,
            generated.labels,
            int(label),
            max_count=chamfer_subsample_per_class,
            rng=rng,
        )
        real_x = _subset_positions_by_label(
            real_reference.positions,
            real_labels,
            int(label),
            max_count=chamfer_subsample_per_class,
            rng=rng,
        )

        one_nn = one_nn_leave_one_out_chamfer(
            real_x,
            gen_x,
            squared=squared_chamfer,
            chunk_size=chunk_size,
        )
        coverage = coverage_unique_argmin_chamfer(
            real_x,
            gen_x,
            squared=squared_chamfer,
            chunk_size=chunk_size,
        )
        rg = pairwise_chamfer_distance_matrix(
            real_x,
            gen_x,
            squared=squared_chamfer,
            chunk_size=chunk_size,
        )
        gg = pairwise_chamfer_distance_matrix(
            gen_x,
            gen_x,
            squared=squared_chamfer,
            chunk_size=chunk_size,
        )
        np.fill_diagonal(gg, np.inf)

        mean_real_to_generated = float(np.mean(np.min(rg, axis=1)))
        mean_generated_to_real = float(np.mean(np.min(rg, axis=0)))
        mean_generated_diversity = float(np.mean(np.min(gg, axis=1)))

        per_label[int(label)] = {
            "one_nn_chamfer_accuracy": float(one_nn["accuracy"]),
            "coverage_chamfer": float(coverage["coverage"]),
            "mean_real_to_generated_chamfer": mean_real_to_generated,
            "mean_generated_to_real_chamfer": mean_generated_to_real,
            "mean_generated_diversity_chamfer": mean_generated_diversity,
        }
        one_nn_values.append(float(one_nn["accuracy"]))
        coverage_values.append(float(coverage["coverage"]))
        real_to_generated_values.append(mean_real_to_generated)
        generated_to_real_values.append(mean_generated_to_real)
        diversity_values.append(mean_generated_diversity)

    return {
        "aux_point_cloud_classifier_accuracy": float(classifier_metrics["accuracy"]),
        "aux_point_cloud_mean_target_probability": float(
            classifier_metrics["mean_target_probability"]
        ),
        "one_nn_chamfer_accuracy_macro": float(np.mean(one_nn_values)),
        "coverage_chamfer_macro": float(np.mean(coverage_values)),
        "mean_real_to_generated_chamfer_macro": float(np.mean(real_to_generated_values)),
        "mean_generated_to_real_chamfer_macro": float(np.mean(generated_to_real_values)),
        "mean_generated_diversity_chamfer_macro": float(np.mean(diversity_values)),
        "squared_chamfer": bool(squared_chamfer),
        "chamfer_subsample_per_class": float(chamfer_subsample_per_class),
        "per_label": per_label,
        "classifier_details": classifier_metrics,
    }
