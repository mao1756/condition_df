
#!/usr/bin/env python3

"""
Generate an MNIST-CP-style point-cloud dataset from MNIST images.

MNIST-CP in the ShapeGF paper is a dataset of 2D contour point clouds extracted
from MNIST, with 800 points per shape. This script extracts digit stroke contours,
resamples them to a fixed number of 2D points, and normalizes each point cloud
to a bounding-box-centered [-1, 1]^2 coordinate system.

Dependencies:
    pip install numpy scikit-image matplotlib

Sample usage:
python generate_mnist_cp.py \
  --download \
  --arff mnist_784.arff \
  --per-class 1 \
  --n-points 800 \
  --out-npz mnist_cp_small.npz \
  --out-png mnist_cp_samples.png

For larger dataset, replace --per-class 1 with something like:

python generate_mnist_cp.py \
  --download \
  --arff mnist_784.arff \
  --n-samples 1000 \
  --n-points 800 \
  --out-npz mnist_cp_1000.npz \
  --out-png mnist_cp_1000_preview.png

The saved .npz file contains:
    points: float32 array with shape (N, 800, 2) containing the contour point clouds.
    labels: int64 array with shape (N,) containing the digit labels.
    images: uint8 array with shape (N, 28, 28) containing the original MNIST images.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
from skimage import filters, measure


OPENML_MNIST_ARFF_URL = "https://www.openml.org/data/download/52667/mnist_784.arff"


def maybe_download(url: str, path: Path) -> None:
    """Download a file if it is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def read_mnist_arff(
    arff_path: str | Path,
    n_samples: int | None = None,
    per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream MNIST examples from an OpenML-style ARFF file.

    Args:
        arff_path: Path to mnist_784.arff.
        n_samples: Number of examples to read. Ignored when per_class is set.
        per_class: Read this many examples per digit class, useful for demos.

    Returns:
        images: uint8 array with shape (N, 28, 28).
        labels: int64 array with shape (N,).
    """
    arff_path = Path(arff_path)
    images, labels = [], []
    counts = {i: 0 for i in range(10)}

    in_data = False
    with arff_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not in_data:
                if line.upper() == "@DATA":
                    in_data = True
                continue
            if not line:
                continue

            values = np.fromstring(line, sep=",")
            if values.size < 785:
                continue

            label = int(values[-1])
            if per_class is not None and counts[label] >= per_class:
                continue

            images.append(values[:784].reshape(28, 28).astype(np.uint8))
            labels.append(label)
            counts[label] += 1

            if per_class is not None and all(counts[d] >= per_class for d in range(10)):
                break
            if per_class is None and n_samples is not None and len(images) >= n_samples:
                break

    if not images:
        raise RuntimeError("No MNIST examples were read. Check the ARFF path.")

    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def resample_polyline(points: np.ndarray, n: int, closed: bool = True) -> np.ndarray:
    """
    Uniformly resample a 2D polyline by arc length.

    Args:
        points: Array of shape (M, 2).
        n: Number of output points.
        closed: Whether to close the curve.

    Returns:
        Array of shape (n, 2).
    """
    points = np.asarray(points, dtype=np.float64)
    if n <= 0:
        return np.empty((0, 2), dtype=np.float64)

    if closed and np.linalg.norm(points[0] - points[-1]) > 1e-8:
        points = np.vstack([points, points[0]])

    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        return np.repeat(points[:1], n, axis=0)

    starts = points[:-1][valid]
    segments = segments[valid]
    lengths = lengths[valid]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total_length = cumulative[-1]

    targets = np.linspace(0.0, total_length, n, endpoint=False)
    seg_idx = np.searchsorted(cumulative[1:], targets, side="right")
    local = (targets - cumulative[seg_idx]) / lengths[seg_idx]
    return starts[seg_idx] + local[:, None] * segments[seg_idx]


def normalize_to_unit_square(points: np.ndarray) -> np.ndarray:
    """
    Center a point cloud by its bounding box and scale so its longest side is 2.
    The resulting points lie inside [-1, 1]^2.
    """
    points = np.asarray(points, dtype=np.float64)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = 0.5 * (mn + mx)
    scale = 0.5 * np.max(mx - mn)
    if scale < 1e-12:
        return points - center
    return (points - center) / scale


def image_to_mnist_cp_points(
    image: np.ndarray,
    n_points: int = 800,
    threshold: float = 0.25,
    smooth_sigma: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Convert one 28x28 MNIST image to an MNIST-CP-style contour point cloud.

    Steps:
      1. Smooth the grayscale image slightly.
      2. Extract iso-contours from the digit strokes.
      3. Allocate samples across contours in proportion to contour length.
      4. Uniformly resample by arc length to exactly n_points.
      5. Normalize to the paper-style [-1, 1]^2 coordinate system.

    Args:
        image: 28x28 MNIST image with pixel values in [0, 255].
        n_points: Number of contour points to output.
        threshold: Iso-contour threshold after scaling image to [0, 1].
        smooth_sigma: Gaussian smoothing sigma before contour extraction.
        seed: Random seed used only for final point-order shuffling/fallbacks.

    Returns:
        points: float32 array with shape (n_points, 2).
    """
    rng = np.random.default_rng(seed)

    img = image.astype(np.float32) / 255.0
    if smooth_sigma > 0:
        img = filters.gaussian(img, sigma=smooth_sigma, preserve_range=True)

    # skimage returns contours as (row, col). Convert to Cartesian-like (x, y)
    # and flip y so plotted point clouds look upright.
    raw_contours = measure.find_contours(img, level=threshold)

    contours = []
    lengths = []
    for contour in raw_contours:
        if contour.shape[0] < 3:
            continue
        xy = np.column_stack([contour[:, 1], -contour[:, 0]])
        if np.linalg.norm(xy[0] - xy[-1]) > 1e-8:
            xy = np.vstack([xy, xy[0]])
        length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        if length > 1.0:
            contours.append(xy)
            lengths.append(length)

    # Fallback for unusual/blank examples.
    if not contours:
        rows, cols = np.nonzero(img > threshold)
        if len(rows) == 0:
            return np.zeros((n_points, 2), dtype=np.float32)
        points = np.column_stack([cols, -rows]).astype(np.float64)
        points = points[rng.choice(len(points), size=n_points, replace=True)]
        return normalize_to_unit_square(points).astype(np.float32)

    lengths = np.asarray(lengths, dtype=np.float64)
    raw_counts = n_points * lengths / lengths.sum()
    counts = np.floor(raw_counts).astype(int)
    counts = np.maximum(counts, 1)

    # Make counts sum to exactly n_points.
    while counts.sum() > n_points:
        idx = int(np.argmax(counts))
        if counts[idx] <= 1:
            break
        counts[idx] -= 1

    remainder = n_points - counts.sum()
    if remainder > 0:
        fractional = raw_counts - np.floor(raw_counts)
        for idx in np.argsort(-fractional)[:remainder]:
            counts[idx] += 1

    pieces = [
        resample_polyline(contour, int(count), closed=True)
        for contour, count in zip(contours, counts)
        if count > 0
    ]
    points = np.concatenate(pieces, axis=0)

    # Guard against any count mismatch.
    if len(points) > n_points:
        points = points[rng.choice(len(points), size=n_points, replace=False)]
    elif len(points) < n_points:
        extra = points[rng.choice(len(points), size=n_points - len(points), replace=True)]
        points = np.vstack([points, extra])

    rng.shuffle(points)
    points = normalize_to_unit_square(points)
    return points.astype(np.float32)


def build_mnist_cp(
    images: np.ndarray,
    labels: np.ndarray,
    n_points: int = 800,
    threshold: float = 0.25,
    smooth_sigma: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a batch of MNIST images to contour point clouds."""
    clouds = []
    for i, image in enumerate(images):
        clouds.append(
            image_to_mnist_cp_points(
                image,
                n_points=n_points,
                threshold=threshold,
                smooth_sigma=smooth_sigma,
                seed=seed + i,
            )
        )
    return np.stack(clouds, axis=0), labels.astype(np.int64)


def save_sample_figure(
    images: np.ndarray,
    labels: np.ndarray,
    clouds: np.ndarray,
    out_png: str | Path,
    max_examples: int = 10,
) -> None:
    """Save a side-by-side visualization of MNIST images and contour point clouds."""
    import matplotlib.pyplot as plt

    n = min(max_examples, len(images))
    fig, axes = plt.subplots(2, n, figsize=(1.55 * n, 3.4), constrained_layout=True)

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(images[i], cmap="gray")
        axes[0, i].set_title(f"{labels[i]}")
        axes[0, i].axis("off")

        axes[1, i].scatter(clouds[i, :, 0], clouds[i, :, 1], s=1)
        axes[1, i].set_aspect("equal")
        axes[1, i].set_xlim(-1.08, 1.08)
        axes[1, i].set_ylim(-1.08, 1.08)
        axes[1, i].axis("off")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arff", type=Path, default=Path("mnist_784.arff"))
    parser.add_argument("--download", action="store_true", help="Download MNIST ARFF if missing.")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of examples to convert.")
    parser.add_argument("--per-class", type=int, default=None, help="Examples per digit class for balanced demos.")
    parser.add_argument("--n-points", type=int, default=800, help="Points per contour cloud.")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--smooth-sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-npz", type=Path, default=Path("mnist_cp_small.npz"))
    parser.add_argument("--out-png", type=Path, default=Path("mnist_cp_samples.png"))
    args = parser.parse_args()

    if args.download:
        maybe_download(OPENML_MNIST_ARFF_URL, args.arff)

    images, labels = read_mnist_arff(args.arff, n_samples=args.n_samples, per_class=args.per_class)
    clouds, labels = build_mnist_cp(
        images,
        labels,
        n_points=args.n_points,
        threshold=args.threshold,
        smooth_sigma=args.smooth_sigma,
        seed=args.seed,
    )

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        points=clouds,
        labels=labels,
        images=images,
        n_points=np.array([args.n_points]),
        normalization=np.array(["bbox_centered_longest_side_to_[-1,1]"]),
    )

    save_sample_figure(images, labels, clouds, args.out_png, max_examples=min(10, len(images)))

    print(f"Saved point clouds: {args.out_npz}")
    print(f"Saved sample figure: {args.out_png}")
    print(f"points shape: {clouds.shape}; labels: {labels.tolist()}")
    print(f"value range: [{clouds.min():.3f}, {clouds.max():.3f}]")


if __name__ == "__main__":
    main()
