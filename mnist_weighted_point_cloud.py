from __future__ import annotations

r"""MNIST-as-measure utilities for weighted point-cloud experiments.

This module turns an MNIST image into a finite atomic probability measure

    \mu = \sum_{i=1}^n s_i \delta_{x_i},

by selecting the ``top_k`` brightest pixels, using their normalized intensities
as masses and their pixel centers as atom locations.  The construction is meant
for the finite-dimensional approximation developed in the manuscript, where the
masses are frozen and only the atom locations evolve.

The code is intentionally lightweight and does not depend on ``torchvision`` so
that it can run in environments where torchvision is unavailable.  A minimal
IDX downloader/reader is provided instead.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import gzip
import struct
import urllib.error
import urllib.request

import numpy as np
from numpy.typing import NDArray

from conditioning_utils import as_float_array, validate_probability_vector

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "MNIST_URL_MIRRORS",
    "WeightedPointCloudBatch",
    "ensure_mnist_downloaded",
    "load_mnist_arrays",
    "normalize_images_to_measures",
    "pixel_centers",
    "images_to_weighted_point_clouds",
    "image_to_weighted_point_cloud",
    "rasterize_weighted_point_clouds",
    "rasterize_weighted_point_cloud",
]


MNIST_URL_MIRRORS: tuple[str, ...] = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://yann.lecun.com/exdb/mnist/",
)

_MNIST_FILES: tuple[str, ...] = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


@dataclass(frozen=True)
class WeightedPointCloudBatch:
    """Container for a batch of weighted point clouds.

    Attributes
    ----------
    masses:
        Array of shape ``(N, K)`` with positive rows summing to one.
    positions:
        Array of shape ``(N, K, 2)`` with coordinates in the unit square.
    labels:
        Optional integer labels of shape ``(N,)``.
    images:
        Optional raster images of shape ``(N, H, W)``.  When present, these are
        usually the normalized-to-one input images from which the measures were
        extracted.
    pixel_indices:
        Optional selected pixel indices of shape ``(N, K)``.
    """

    masses: FloatArray
    positions: FloatArray
    labels: Optional[IntArray] = None
    images: Optional[FloatArray] = None
    pixel_indices: Optional[IntArray] = None

    def __post_init__(self) -> None:
        if self.masses.ndim != 2:
            raise ValueError("masses must have shape (N, K)")
        if self.positions.ndim != 3 or self.positions.shape[:2] != self.masses.shape:
            raise ValueError("positions must have shape (N, K, 2) and match masses")
        if self.positions.shape[2] != 2:
            raise ValueError("positions must have two spatial coordinates")
        if self.labels is not None and self.labels.shape != (len(self),):
            raise ValueError("labels must have shape (N,)")
        if self.images is not None and self.images.shape[0] != len(self):
            raise ValueError("images must have first dimension N")
        if self.pixel_indices is not None and self.pixel_indices.shape != self.masses.shape:
            raise ValueError("pixel_indices must have shape (N, K)")

    def __len__(self) -> int:
        return int(self.masses.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.masses.shape[1])

    def subset(self, indices: Iterable[int] | slice | IntArray) -> "WeightedPointCloudBatch":
        idx = np.arange(len(self))[indices] if isinstance(indices, slice) else np.asarray(indices)
        return WeightedPointCloudBatch(
            masses=np.asarray(self.masses[idx], dtype=np.float64),
            positions=np.asarray(self.positions[idx], dtype=np.float64),
            labels=None if self.labels is None else np.asarray(self.labels[idx], dtype=np.int64),
            images=None if self.images is None else np.asarray(self.images[idx], dtype=np.float64),
            pixel_indices=None if self.pixel_indices is None else np.asarray(self.pixel_indices[idx], dtype=np.int64),
        )


def _download_url(url: str, destination: Path, *, timeout: float = 30.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def ensure_mnist_downloaded(
    root: str | Path,
    *,
    mirrors: tuple[str, ...] = MNIST_URL_MIRRORS,
    force: bool = False,
) -> Path:
    """Download the raw MNIST IDX files into ``root`` if they are missing.

    The downloader tries a small sequence of mirrors because some environments
    can reach one mirror but not another.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    for filename in _MNIST_FILES:
        path = root_path / filename
        if path.exists() and not force:
            continue

        last_error: Optional[Exception] = None
        for base in mirrors:
            url = base + filename
            try:
                _download_url(url, path)
                last_error = None
                break
            except Exception as exc:  # pragma: no cover - network failures are environment specific.
                last_error = exc
        if last_error is not None:
            raise RuntimeError(
                f"failed to download {filename}; tried mirrors {mirrors}"
            ) from last_error
    return root_path


def _read_idx_images_gz(path: str | Path) -> FloatArray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path} does not look like an IDX image file (magic={magic})")
        payload = handle.read()
    data = np.frombuffer(payload, dtype=np.uint8)
    expected = count * rows * cols
    if data.size != expected:
        raise ValueError(f"{path} contains {data.size} bytes, expected {expected}")
    return data.reshape(count, rows, cols).astype(np.float64) / 255.0


def _read_idx_labels_gz(path: str | Path) -> IntArray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path} does not look like an IDX label file (magic={magic})")
        payload = handle.read()
    data = np.frombuffer(payload, dtype=np.uint8)
    if data.size != count:
        raise ValueError(f"{path} contains {data.size} labels, expected {count}")
    return data.astype(np.int64)


@dataclass(frozen=True)
class _MNISTArrays:
    train_images: FloatArray
    train_labels: IntArray
    test_images: FloatArray
    test_labels: IntArray


def load_mnist_arrays(
    root: str | Path,
    *,
    download: bool = True,
    normalize_to_measure: bool = False,
) -> dict[str, FloatArray | IntArray]:
    """Load MNIST arrays from raw IDX files.

    Parameters
    ----------
    root:
        Directory holding the four gzipped IDX files.
    download:
        If ``True``, missing files are downloaded automatically.
    normalize_to_measure:
        If ``True``, each image is divided by its total intensity so that it can
        be interpreted as a probability measure on the pixel grid.
    """
    root_path = Path(root)
    if download:
        ensure_mnist_downloaded(root_path)

    train_images = _read_idx_images_gz(root_path / "train-images-idx3-ubyte.gz")
    train_labels = _read_idx_labels_gz(root_path / "train-labels-idx1-ubyte.gz")
    test_images = _read_idx_images_gz(root_path / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_idx_labels_gz(root_path / "t10k-labels-idx1-ubyte.gz")

    if normalize_to_measure:
        train_images = normalize_images_to_measures(train_images)
        test_images = normalize_images_to_measures(test_images)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def normalize_images_to_measures(images: np.ndarray, *, eps: float = 1e-12) -> FloatArray:
    """Normalize each image to have unit total mass.

    The manuscript's state space consists of probability measures, so for the
    MNIST experiment we normalize grayscale images into discrete probability
    measures on the 28x28 pixel grid.
    """
    arr = as_float_array(images, name="images")
    if arr.ndim == 2:
        arr = arr[None, :, :]
        squeeze = True
    elif arr.ndim == 3:
        squeeze = False
    else:
        raise ValueError("images must have shape (H, W) or (N, H, W)")

    flat = arr.reshape(arr.shape[0], -1)
    sums = flat.sum(axis=1, keepdims=True)
    zero_mask = sums[:, 0] <= eps
    if np.any(zero_mask):
        flat = flat.copy()
        flat[zero_mask] = 1.0 / flat.shape[1]
        sums = flat.sum(axis=1, keepdims=True)
    normalized = flat / sums
    out = normalized.reshape(arr.shape)
    return out[0] if squeeze else out


def pixel_centers(image_size: int = 28) -> FloatArray:
    """Return pixel-center coordinates in ``[0, 1]^2``.

    The first coordinate is horizontal ``x`` and the second is vertical ``y``.
    The vertical coordinate increases downward, matching the array convention of
    image tensors and ``matplotlib.imshow`` with the default origin.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    rows, cols = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")
    x = (cols.astype(np.float64) + 0.5) / image_size
    y = (rows.astype(np.float64) + 0.5) / image_size
    return np.stack([x.reshape(-1), y.reshape(-1)], axis=1)


def image_to_weighted_point_cloud(
    image: np.ndarray,
    *,
    top_k: int,
    mass_floor: float = 0.0,
    pixel_positions: Optional[FloatArray] = None,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Convert a single image into a weighted point cloud.

    The ``top_k`` brightest pixels are selected.  Their intensities are shifted
    by ``mass_floor`` (if requested) and renormalized to obtain a strictly
    positive mass vector.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    image_array = as_float_array(image, name="image")
    if image_array.ndim != 2:
        raise ValueError("image must have shape (H, W)")
    if image_array.shape[0] != image_array.shape[1]:
        raise ValueError("image must be square")
    image_size = image_array.shape[0]
    if top_k > image_size * image_size:
        raise ValueError("top_k cannot exceed the number of pixels")

    flat = image_array.reshape(-1)
    idx = np.argpartition(flat, -top_k)[-top_k:]
    masses = flat[idx]
    order = np.argsort(-masses, kind="mergesort")
    idx = idx[order].astype(np.int64)
    masses = masses[order]

    if mass_floor < 0.0:
        raise ValueError("mass_floor must be non-negative")
    if mass_floor > 0.0:
        masses = masses + mass_floor

    if np.sum(masses) <= 0.0:
        masses = np.full(top_k, 1.0 / top_k, dtype=np.float64)
    else:
        masses = validate_probability_vector(masses, name="masses", normalize=True)

    positions = pixel_centers(image_size) if pixel_positions is None else np.asarray(pixel_positions, dtype=np.float64)
    if positions.shape != (image_size * image_size, 2):
        raise ValueError(f"pixel_positions must have shape ({image_size * image_size}, 2)")
    selected_positions = positions[idx]
    return masses, selected_positions, idx


def images_to_weighted_point_clouds(
    images: np.ndarray,
    *,
    top_k: int,
    labels: Optional[np.ndarray] = None,
    mass_floor: float = 0.0,
    normalize_to_measure: bool = True,
) -> WeightedPointCloudBatch:
    """Vectorized conversion of a stack of images into weighted point clouds."""
    arr = as_float_array(images, name="images")
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("images must have shape (H, W) or (N, H, W)")
    if arr.shape[1] != arr.shape[2]:
        raise ValueError("images must be square")
    if top_k <= 0 or top_k > arr.shape[1] * arr.shape[2]:
        raise ValueError("top_k must lie between 1 and the number of pixels")
    if mass_floor < 0.0:
        raise ValueError("mass_floor must be non-negative")

    n_samples, image_size, _ = arr.shape
    work = normalize_images_to_measures(arr) if normalize_to_measure else np.asarray(arr, dtype=np.float64)
    flat = work.reshape(n_samples, -1)

    idx = np.argpartition(flat, -top_k, axis=1)[:, -top_k:]
    values = np.take_along_axis(flat, idx, axis=1)
    order = np.argsort(-values, axis=1, kind="mergesort")
    idx = np.take_along_axis(idx, order, axis=1).astype(np.int64)
    values = np.take_along_axis(values, order, axis=1)

    if mass_floor > 0.0:
        values = values + mass_floor

    sums = values.sum(axis=1, keepdims=True)
    zero_mask = sums[:, 0] <= 0.0
    if np.any(zero_mask):
        values = values.copy()
        values[zero_mask] = 1.0 / top_k
        sums = values.sum(axis=1, keepdims=True)

    masses = values / sums
    positions_lookup = pixel_centers(image_size)
    positions = positions_lookup[idx]

    labels_array = None
    if labels is not None:
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_array.shape != (n_samples,):
            raise ValueError("labels must have shape (N,)")

    return WeightedPointCloudBatch(
        masses=masses.astype(np.float64),
        positions=positions.astype(np.float64),
        labels=labels_array,
        images=work.astype(np.float64),
        pixel_indices=idx,
    )


def _prepare_raster_inputs(
    masses: np.ndarray,
    positions: np.ndarray,
) -> tuple[FloatArray, FloatArray, bool]:
    m = as_float_array(masses, name="masses")
    x = as_float_array(positions, name="positions")
    if m.ndim == 1:
        m = m[None, :]
        squeeze = True
    elif m.ndim == 2:
        squeeze = False
    else:
        raise ValueError("masses must have shape (K,) or (N, K)")

    if x.ndim == 2:
        x = x[None, :, :]
    elif x.ndim != 3:
        raise ValueError("positions must have shape (K, 2) or (N, K, 2)")

    if x.shape[:2] != m.shape:
        raise ValueError("masses and positions must agree on their first two dimensions")
    if x.shape[2] != 2:
        raise ValueError("positions must have two spatial coordinates")

    row_sums = m.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("each mass vector must have positive total mass")
    return m.astype(np.float64), x.astype(np.float64), squeeze


def rasterize_weighted_point_clouds(
    masses: np.ndarray,
    positions: np.ndarray,
    *,
    image_size: int = 28,
    renormalize: bool = True,
) -> FloatArray:
    """Rasterize weighted point clouds by bilinear splatting.

    This is the right inverse of the point-cloud representation used in the
    experiment only in an approximate sense: the generated particles move in
    continuous space, so we splat their masses back onto the 28x28 grid.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    m, x, squeeze = _prepare_raster_inputs(masses, positions)
    n_samples, top_k = m.shape
    images = np.zeros((n_samples, image_size, image_size), dtype=np.float64)

    u = np.clip(x[..., 0], 0.0, 1.0) * image_size - 0.5
    v = np.clip(x[..., 1], 0.0, 1.0) * image_size - 0.5

    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    dx = u - x0
    dy = v - y0

    sample_ids = np.broadcast_to(np.arange(n_samples)[:, None], (n_samples, top_k))

    for ox, wx in ((0, 1.0 - dx), (1, dx)):
        xi = np.clip(x0 + ox, 0, image_size - 1)
        for oy, wy in ((0, 1.0 - dy), (1, dy)):
            yi = np.clip(y0 + oy, 0, image_size - 1)
            weights = m * wx * wy
            np.add.at(images, (sample_ids, yi, xi), weights)

    if renormalize:
        denom = images.reshape(n_samples, -1).sum(axis=1, keepdims=True)
        denom = np.where(denom <= 0.0, 1.0, denom)
        images = images / denom.reshape(n_samples, 1, 1)

    return images[0] if squeeze else images


def rasterize_weighted_point_cloud(
    masses: np.ndarray,
    positions: np.ndarray,
    *,
    image_size: int = 28,
    renormalize: bool = True,
) -> FloatArray:
    """Single-sample wrapper around :func:`rasterize_weighted_point_clouds`."""
    return rasterize_weighted_point_clouds(
        masses,
        positions,
        image_size=image_size,
        renormalize=renormalize,
    )
