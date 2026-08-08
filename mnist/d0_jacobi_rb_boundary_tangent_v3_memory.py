"""Memory-safe primitives for the immutable-cache boundary-tangent v3 recovery.

This module is deliberately additive.  It does not generate Jacobi paths, open
labels implicitly, select checkpoints, or run confirmation.  Its only job is
to make the already-frozen v3 objectives executable on bounded-memory devices:

* cache arrays remain on the host in writable C-order NumPy storage;
* a maximum of 32 rows is materialized on the accelerator for a model call;
* physical labels are opened through a separate, role-bound authorization;
* reductions have a fixed, row-major float64 order; and
* controls and predictions are accumulated without retaining device outputs.

The functions accept ordinary ``torch.nn.Module`` objects so that the CLI can
reuse the unchanged v3 predictor and checkpoint formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_boundary_tangent import synthetic_tangent_target
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import (
    load_eager_role_inputs,
    load_eager_role_labels,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    exact_zero_baseline_prediction,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    MODEL_INPUT_FIELDS,
    ModelInputs,
    call_model,
)


STREAMING_MEMORY_VERSION = "d0-jacobi-rb-boundary-tangent-v3-streaming-memory-v1"
MAXIMUM_MODEL_FORWARD_BATCH_SIZE = 32
_INPUT_ROLES = frozenset({"train", "validation"})
_LABEL_PURPOSES = {
    "train": frozenset({"physical_training"}),
    "validation": frozenset({"validation_selection"}),
}
_SHA256_LENGTH = 64


class StreamingMemoryError(RuntimeError):
    """Fail-closed violation of the streaming-memory contract."""


def _as_writable_c_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    if not array.flags.c_contiguous or not array.flags.writeable:
        raise StreamingMemoryError(f"{name} is not writable C-order storage")
    return array


def _copy_array_mapping(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    copied = {
        str(name): _as_writable_c_array(np.asarray(value), name=str(name))
        for name, value in values.items()
    }
    return MappingProxyType(copied)


def _row_count(values: Mapping[str, np.ndarray], *, names: Iterable[str]) -> int:
    expected: int | None = None
    for name in names:
        if name not in values:
            raise StreamingMemoryError(f"cache field is missing: {name}")
        array = np.asarray(values[name])
        if array.ndim == 0:
            raise StreamingMemoryError(f"cache field has no row dimension: {name}")
        count = int(array.shape[0])
        if expected is None:
            expected = count
        elif count != expected:
            raise StreamingMemoryError("cache fields have inconsistent row counts")
    return int(expected or 0)


def _validate_sha256(value: str, *, name: str) -> str:
    text = str(value).lower()
    if len(text) != _SHA256_LENGTH or any(c not in "0123456789abcdef" for c in text):
        raise StreamingMemoryError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _normalize_indices(
    indices: slice | Sequence[int] | np.ndarray | Tensor,
    *,
    row_count: int,
) -> np.ndarray:
    if isinstance(indices, slice):
        start, stop, step = indices.indices(row_count)
        values = np.arange(start, stop, step, dtype=np.int64)
    elif isinstance(indices, Tensor):
        values = indices.detach().to(device="cpu", dtype=torch.long).numpy().copy()
    else:
        values = np.asarray(indices, dtype=np.int64)
    values = np.ascontiguousarray(values.reshape(-1), dtype=np.int64)
    if values.size and (int(values.min()) < 0 or int(values.max()) >= row_count):
        raise StreamingMemoryError("row index is outside the host store")
    return values


@dataclass(frozen=True)
class HostInputStore:
    """Permitted model inputs held wholly in writable host NumPy arrays."""

    cache_root: Path
    role: str
    arrays: Mapping[str, np.ndarray]
    index: Mapping[str, Any] = field(default_factory=dict)
    _row_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.cache_root).resolve()
        role = str(self.role)
        if role not in _INPUT_ROLES:
            raise StreamingMemoryError("input role must be train or validation")
        copied = _copy_array_mapping(self.arrays)
        count = _row_count(copied, names=MODEL_INPUT_FIELDS)
        if tuple(copied["later_full_state"].shape) != (count, 784):
            raise StreamingMemoryError("later_full_state must have shape [N,784]")
        for name in ("reverse_time", "phase", "color", "duration", "label"):
            if tuple(copied[name].shape) != (count,):
                raise StreamingMemoryError(f"{name} must have shape [N]")
        # Audit coordinates may be carried by the input archive but never enter
        # ModelInputs.  If present, they must still share the canonical row order.
        for name, value in copied.items():
            if np.asarray(value).ndim and int(np.asarray(value).shape[0]) != count:
                raise StreamingMemoryError(f"input audit field has wrong rows: {name}")
        index = dict(self.index)
        if "role" in index and str(index["role"]) != role:
            raise StreamingMemoryError("input index role changed")
        if "input_row_count" in index and int(index["input_row_count"]) != count:
            raise StreamingMemoryError("input index row count changed")
        object.__setattr__(self, "cache_root", root)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "arrays", copied)
        object.__setattr__(self, "index", MappingProxyType(index))
        object.__setattr__(self, "_row_count", count)

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        role: str,
        cache_root: str | Path = ".",
        index: Mapping[str, Any] | None = None,
    ) -> "HostInputStore":
        return cls(Path(cache_root), role, arrays, index or {})

    @property
    def row_count(self) -> int:
        return self._row_count

    def row_array(self, name: str) -> np.ndarray:
        if name not in self.arrays:
            raise StreamingMemoryError(f"unknown input field: {name}")
        return self.arrays[name]

    def batch(
        self,
        indices: slice | Sequence[int] | np.ndarray | Tensor,
        *,
        device: str | torch.device,
    ) -> ModelInputs:
        rows = _normalize_indices(indices, row_count=self.row_count)
        if rows.size > MAXIMUM_MODEL_FORWARD_BATCH_SIZE:
            raise StreamingMemoryError(
                f"input batch {rows.size} exceeds frozen maximum "
                f"{MAXIMUM_MODEL_FORWARD_BATCH_SIZE}"
            )

        def copied(name: str) -> np.ndarray:
            # np.take gives an owning, writable result for arbitrary/repeated rows.
            return np.array(np.take(self.arrays[name], rows, axis=0), copy=True, order="C")

        target = torch.device(device)
        return ModelInputs(
            later_full_state=torch.as_tensor(
                copied("later_full_state"), dtype=torch.float32, device=target
            ),
            reverse_time=torch.as_tensor(
                copied("reverse_time"), dtype=torch.float64, device=target
            ),
            phase=torch.as_tensor(copied("phase"), dtype=torch.long, device=target),
            color=torch.as_tensor(copied("color"), dtype=torch.long, device=target),
            duration=torch.as_tensor(
                copied("duration"), dtype=torch.float32, device=target
            ),
            label=torch.as_tensor(copied("label"), dtype=torch.long, device=target),
        )

    def sequential_batches(self, *, batch_size: int = 32) -> Iterable[np.ndarray]:
        size = validate_batch_size(batch_size)
        for start in range(0, self.row_count, size):
            yield np.arange(start, min(self.row_count, start + size), dtype=np.int64)


@dataclass(frozen=True)
class LabelOpenAuthorization:
    """Explicit permission binding a physical-label opening to one role/purpose."""

    cache_root: Path
    role: str
    purpose: str
    opening_seal_sha256: str

    def __post_init__(self) -> None:
        root = Path(self.cache_root).resolve()
        role = str(self.role)
        purpose = str(self.purpose)
        if role not in _LABEL_PURPOSES or purpose not in _LABEL_PURPOSES[role]:
            raise StreamingMemoryError("label role and opening purpose are incompatible")
        digest = _validate_sha256(
            self.opening_seal_sha256, name="opening_seal_sha256"
        )
        object.__setattr__(self, "cache_root", root)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "opening_seal_sha256", digest)


@dataclass(frozen=True)
class HostLabelStore:
    """Physical labels opened separately from the permitted model inputs."""

    cache_root: Path
    role: str
    purpose: str
    opening_seal_sha256: str
    arrays: Mapping[str, np.ndarray]
    index: Mapping[str, Any] = field(default_factory=dict)
    _row_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.cache_root).resolve()
        role = str(self.role)
        purpose = str(self.purpose)
        if role not in _LABEL_PURPOSES or purpose not in _LABEL_PURPOSES[role]:
            raise StreamingMemoryError("label store has incompatible role/purpose")
        digest = _validate_sha256(
            self.opening_seal_sha256, name="opening_seal_sha256"
        )
        copied = _copy_array_mapping(self.arrays)
        count = _row_count(copied, names=("denoising_target",))
        if tuple(copied["denoising_target"].shape) != (count, EDGES_PER_PHASE):
            raise StreamingMemoryError("denoising_target must have shape [N,392]")
        for name, value in copied.items():
            if np.asarray(value).ndim and int(np.asarray(value).shape[0]) != count:
                raise StreamingMemoryError(f"label audit field has wrong rows: {name}")
        index = dict(self.index)
        if "role" in index and str(index["role"]) != role:
            raise StreamingMemoryError("label index role changed")
        if "label_row_count" in index and int(index["label_row_count"]) != count:
            raise StreamingMemoryError("label index row count changed")
        object.__setattr__(self, "cache_root", root)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "opening_seal_sha256", digest)
        object.__setattr__(self, "arrays", copied)
        object.__setattr__(self, "index", MappingProxyType(index))
        object.__setattr__(self, "_row_count", count)

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        authorization: LabelOpenAuthorization,
        index: Mapping[str, Any] | None = None,
    ) -> "HostLabelStore":
        return cls(
            authorization.cache_root,
            authorization.role,
            authorization.purpose,
            authorization.opening_seal_sha256,
            arrays,
            index or {},
        )

    @property
    def row_count(self) -> int:
        return self._row_count

    def row_array(self, name: str) -> np.ndarray:
        if name not in self.arrays:
            raise StreamingMemoryError(f"unknown label field: {name}")
        return self.arrays[name]

    def target_batch(
        self,
        indices: slice | Sequence[int] | np.ndarray | Tensor,
        *,
        device: str | torch.device,
    ) -> Tensor:
        rows = _normalize_indices(indices, row_count=self.row_count)
        if rows.size > MAXIMUM_MODEL_FORWARD_BATCH_SIZE:
            raise StreamingMemoryError(
                f"label batch {rows.size} exceeds frozen maximum "
                f"{MAXIMUM_MODEL_FORWARD_BATCH_SIZE}"
            )
        values = np.array(
            np.take(self.arrays["denoising_target"], rows, axis=0),
            copy=True,
            order="C",
        )
        return torch.as_tensor(values, dtype=torch.float64, device=torch.device(device))


def open_external_input_store(
    cache_root: str | Path, role: str
) -> HostInputStore:
    """Open permitted input artifacts without touching any label archive."""

    root = Path(cache_root).resolve()
    arrays, index = load_eager_role_inputs(root, role)
    return HostInputStore.from_arrays(arrays, role=role, cache_root=root, index=index)


def open_external_label_store(
    cache_root: str | Path,
    role: str,
    *,
    authorization: LabelOpenAuthorization,
) -> HostLabelStore:
    """Open labels only when the authorization exactly binds root and role."""

    root = Path(cache_root).resolve()
    if authorization.cache_root != root or authorization.role != str(role):
        raise StreamingMemoryError("label authorization does not bind this cache role")
    arrays, index = load_eager_role_labels(root, role)
    return HostLabelStore.from_arrays(arrays, authorization=authorization, index=index)


def validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise StreamingMemoryError("batch size must be an integer")
    if not 1 <= batch_size <= MAXIMUM_MODEL_FORWARD_BATCH_SIZE:
        raise StreamingMemoryError(
            f"batch size must lie in [1,{MAXIMUM_MODEL_FORWARD_BATCH_SIZE}]"
        )
    return batch_size


@dataclass
class ModelCallBatchGuard:
    """Instrument and enforce the frozen model-forward batch ceiling."""

    maximum_batch_size: int = MAXIMUM_MODEL_FORWARD_BATCH_SIZE
    call_count: int = 0
    maximum_observed_batch_size: int = 0
    observed_batch_sizes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.maximum_batch_size = validate_batch_size(self.maximum_batch_size)

    def call(self, model: nn.Module, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise StreamingMemoryError("guard accepts only exact ModelInputs")
        batch = inputs.batch_size
        if batch > self.maximum_batch_size:
            raise StreamingMemoryError(
                f"model forward batch {batch} exceeds frozen maximum "
                f"{self.maximum_batch_size}"
            )
        self.call_count += 1
        self.maximum_observed_batch_size = max(
            self.maximum_observed_batch_size, batch
        )
        self.observed_batch_sizes.append(batch)
        return call_model(model, inputs)

    def record(self) -> dict[str, Any]:
        return {
            "schema": STREAMING_MEMORY_VERSION + "-model-call-batches",
            "schema_version": 1,
            "maximum_allowed_batch_size": self.maximum_batch_size,
            "call_count": self.call_count,
            "maximum_observed_batch_size": self.maximum_observed_batch_size,
            "all_calls_within_limit": int(
                self.maximum_observed_batch_size <= self.maximum_batch_size
            ),
            "observed_batch_sizes_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    np.asarray(self.observed_batch_sizes, dtype=np.int64)
                ).tobytes(order="C")
            ).hexdigest(),
        }


@dataclass
class CanonicalRowSquareReducer:
    """Fixed row-major float64 sum-of-squares accumulator.

    Each row is reduced independently using NumPy's deterministic float64
    C-order reduction.  Row totals are then combined with ``math.fsum``.  This
    makes the answer independent of streaming batch boundaries.
    """

    row_square_sums: list[float] = field(default_factory=list)
    element_count: int = 0

    def update(self, values: np.ndarray | Tensor) -> None:
        if isinstance(values, Tensor):
            array = values.detach().to(device="cpu", dtype=torch.float64).numpy()
        else:
            array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            raise StreamingMemoryError("square reducer requires a row dimension")
        matrix = np.ascontiguousarray(array.reshape(array.shape[0], -1), dtype=np.float64)
        if not bool(np.isfinite(matrix).all()):
            raise StreamingMemoryError("square reducer received nonfinite values")
        squared = np.square(matrix, dtype=np.float64)
        totals = np.sum(squared, axis=1, dtype=np.float64)
        self.row_square_sums.extend(float(value) for value in totals.tolist())
        self.element_count += int(matrix.size)

    @property
    def square_sum(self) -> float:
        return float(math.fsum(self.row_square_sums))

    @property
    def mean_square(self) -> float:
        if self.element_count <= 0:
            raise StreamingMemoryError("square reducer is empty")
        return self.square_sum / float(self.element_count)

    @property
    def rms(self) -> float:
        value = self.mean_square
        if not math.isfinite(value) or value < 0.0:
            raise StreamingMemoryError("square reducer produced an invalid mean")
        return math.sqrt(value)

    def record(self) -> dict[str, Any]:
        return {
            "schema": STREAMING_MEMORY_VERSION + "-canonical-square-reduction",
            "schema_version": 1,
            "row_count": len(self.row_square_sums),
            "element_count": self.element_count,
            "square_sum": self.square_sum,
            "mean_square": self.mean_square,
            "rms": self.rms,
            "combiner": "math.fsum_over_c_order_float64_row_sums",
        }


def canonical_row_square_reduction(
    values: np.ndarray | Tensor,
    *,
    batch_size: int = MAXIMUM_MODEL_FORWARD_BATCH_SIZE,
) -> dict[str, Any]:
    size = validate_batch_size(batch_size)
    total_rows = int(values.shape[0])
    reducer = CanonicalRowSquareReducer()
    for start in range(0, total_rows, size):
        reducer.update(values[start : min(total_rows, start + size)])
    return reducer.record()


TargetProvider = Callable[[ModelInputs], Tensor]


def synthetic_target_provider(inputs: ModelInputs) -> Tensor:
    return synthetic_tangent_target(inputs).to(dtype=torch.float64)


def canonical_streamed_target_scale(
    inputs: HostInputStore,
    *,
    device: str | torch.device,
    target_provider: TargetProvider = synthetic_target_provider,
    batch_size: int = MAXIMUM_MODEL_FORWARD_BATCH_SIZE,
) -> tuple[float, dict[str, Any]]:
    size = validate_batch_size(batch_size)
    reducer = CanonicalRowSquareReducer()
    with torch.no_grad():
        for rows in inputs.sequential_batches(batch_size=size):
            batch = inputs.batch(rows, device=device)
            target = target_provider(batch)
            if tuple(target.shape) != (len(rows), EDGES_PER_PHASE):
                raise StreamingMemoryError("target provider returned the wrong shape")
            reducer.update(target)
    record = reducer.record()
    scale = float(record["rms"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise StreamingMemoryError("streamed target scale is not finite and positive")
    return scale, record


def stream_zero_initialization(
    model: nn.Module,
    stores: Mapping[str, HostInputStore],
    *,
    device: str | torch.device,
    guard: ModelCallBatchGuard | None = None,
    batch_size: int = MAXIMUM_MODEL_FORWARD_BATCH_SIZE,
    baseline_provider: TargetProvider = exact_zero_baseline_prediction,
) -> dict[str, Any]:
    """Scan zero initialization without ever constructing full device outputs."""

    size = validate_batch_size(batch_size)
    active_guard = guard or ModelCallBatchGuard()
    was_training = model.training
    model.eval()
    roles: dict[str, Any] = {}
    try:
        with torch.no_grad():
            for role in sorted(stores):
                store = stores[role]
                exact_prediction = True
                exact_baseline = True
                maximum_prediction = 0.0
                maximum_baseline = 0.0
                for rows in store.sequential_batches(batch_size=size):
                    batch = store.batch(rows, device=device)
                    prediction = active_guard.call(model, batch).to(torch.float64)
                    baseline = baseline_provider(batch).to(torch.float64)
                    if tuple(baseline.shape) != tuple(prediction.shape):
                        raise StreamingMemoryError("baseline returned the wrong shape")
                    exact_prediction &= bool(torch.all(prediction == 0.0))
                    exact_baseline &= bool(torch.all(baseline == 0.0))
                    if prediction.numel():
                        maximum_prediction = max(
                            maximum_prediction,
                            float(torch.max(torch.abs(prediction)).cpu()),
                        )
                        maximum_baseline = max(
                            maximum_baseline,
                            float(torch.max(torch.abs(baseline)).cpu()),
                        )
                roles[role] = {
                    "row_count": store.row_count,
                    "prediction_exact_zero": int(exact_prediction),
                    "baseline_exact_zero": int(exact_baseline),
                    "maximum_absolute_prediction": maximum_prediction,
                    "maximum_absolute_baseline": maximum_baseline,
                }
    finally:
        model.train(was_training)
    return {
        "schema": STREAMING_MEMORY_VERSION + "-zero-initialization",
        "schema_version": 1,
        "roles": roles,
        "passed": int(
            bool(roles)
            and all(
                row["prediction_exact_zero"] == 1
                and row["baseline_exact_zero"] == 1
                for row in roles.values()
            )
        ),
        "model_call_batches": active_guard.record(),
    }


def synthetic_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: HostInputStore,
    indices: Sequence[int] | np.ndarray | Tensor,
    *,
    scale: float,
    device: str | torch.device,
    guard: ModelCallBatchGuard,
    gradient_norm_clip: float = 1.0,
    target_provider: TargetProvider = synthetic_target_provider,
) -> dict[str, float]:
    """Execute one unchanged synthetic-teacher optimizer update."""

    if not math.isfinite(scale) or scale <= 0.0:
        raise StreamingMemoryError("training scale must be finite and positive")
    rows = _normalize_indices(indices, row_count=inputs.row_count)
    batch = inputs.batch(rows, device=device)
    with torch.no_grad():
        target = target_provider(batch).detach().to(dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    prediction = guard.call(model, batch).to(dtype=torch.float64)
    raw = torch.mean((prediction - target).square())
    loss = raw / (float(scale) * float(scale))
    if not bool(torch.isfinite(loss)):
        raise StreamingMemoryError("synthetic training loss became nonfinite")
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm_clip)
    optimizer.step()
    return {
        "raw_mse": float(raw.detach().cpu()),
        "scaled_loss": float(loss.detach().cpu()),
        "preclip_gradient_norm": float(gradient),
    }


def stream_target_metrics(
    model: nn.Module,
    inputs: HostInputStore,
    *,
    device: str | torch.device,
    target_provider: TargetProvider = synthetic_target_provider,
    path_rows: np.ndarray | None = None,
    guard: ModelCallBatchGuard | None = None,
    batch_size: int = MAXIMUM_MODEL_FORWARD_BATCH_SIZE,
) -> dict[str, Any]:
    """Evaluate target risk and whole-path risk without device concatenation."""

    size = validate_batch_size(batch_size)
    active_guard = guard or ModelCallBatchGuard()
    paths = np.asarray(
        inputs.row_array("path_id") if path_rows is None else path_rows,
        dtype=np.int64,
    ).reshape(-1)
    if paths.shape != (inputs.row_count,):
        raise StreamingMemoryError("path_rows has wrong shape")
    totals: dict[int, tuple[list[float], list[float]]] = {}
    residual_rows_all: list[float] = []
    zero_rows_all: list[float] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for rows in inputs.sequential_batches(batch_size=size):
                batch = inputs.batch(rows, device=device)
                prediction = active_guard.call(model, batch).to(torch.float64)
                target = target_provider(batch).to(torch.float64)
                # Preserve the legacy control's operation order: mean over each
                # edge row on CUDA, followed by a binary64 mean over rows.
                residual_rows = (
                    torch.mean((prediction - target).square(), dim=1)
                    .cpu()
                    .numpy()
                )
                zero_rows = torch.mean(target.square(), dim=1).cpu().numpy()
                residual_rows_all.extend(float(v) for v in residual_rows.tolist())
                zero_rows_all.extend(float(v) for v in zero_rows.tolist())
                for offset, row_index in enumerate(rows.tolist()):
                    path = int(paths[row_index])
                    residual_values, zero_values = totals.setdefault(path, ([], []))
                    residual_values.append(float(residual_rows[offset]))
                    zero_values.append(float(zero_rows[offset]))
    finally:
        model.train(was_training)
    if not residual_rows_all:
        raise StreamingMemoryError("target evaluation has no elements")
    rows_out: list[dict[str, Any]] = []
    every = True
    for path, (residual_values, zero_values) in sorted(totals.items()):
        model_mse = float(np.mean(residual_values, dtype=np.float64))
        zero_mse = float(np.mean(zero_values, dtype=np.float64))
        beats = model_mse < zero_mse
        every &= beats
        rows_out.append(
            {
                "path_id": path,
                "model_mse": model_mse,
                "zero_mse": zero_mse,
                "beats_zero": int(beats),
            }
        )
    mse = float(np.mean(residual_rows_all, dtype=np.float64))
    zero_mse = float(np.mean(zero_rows_all, dtype=np.float64))
    return {
        "schema": STREAMING_MEMORY_VERSION + "-target-metrics",
        "schema_version": 1,
        "model_mse": mse,
        "zero_mse": zero_mse,
        "relative_mse": mse / zero_mse if zero_mse > 0.0 else math.inf,
        "every_path_beats_zero": int(every),
        "path_metrics": rows_out,
        "model_call_batches": active_guard.record(),
    }


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def exact_null_batchwise_one_step(
    teacher: nn.Module,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_inputs: HostInputStore,
    validation_inputs: HostInputStore,
    *,
    device: str | torch.device,
    guard: ModelCallBatchGuard | None = None,
    batch_size: int = MAXIMUM_MODEL_FORWARD_BATCH_SIZE,
) -> dict[str, Any]:
    """Apply the exact-model null's one full-dataset optimizer step in batches."""

    size = validate_batch_size(batch_size)
    active_guard = guard or ModelCallBatchGuard()
    teacher.eval()
    student.train()
    before = _clone_state_dict(student)
    target_energy = CanonicalRowSquareReducer()
    element_count = train_inputs.row_count * EDGES_PER_PHASE
    if element_count <= 0:
        raise StreamingMemoryError("null training store is empty")
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for rows in train_inputs.sequential_batches(batch_size=size):
        batch = train_inputs.batch(rows, device=device)
        with torch.no_grad():
            target = active_guard.call(teacher, batch).detach().to(torch.float64)
        target_energy.update(target)
        prediction = active_guard.call(student, batch).to(torch.float64)
        squared_sum = torch.sum((prediction - target).square())
        batch_loss = squared_sum / float(element_count)
        if not bool(torch.isfinite(batch_loss)):
            raise StreamingMemoryError("exact-null loss became nonfinite")
        batch_loss.backward()
        loss_value += float(batch_loss.detach().cpu())
    gradient_exact = all(
        parameter.grad is None or bool(torch.all(parameter.grad == 0.0))
        for parameter in student.parameters()
    )
    optimizer.step()
    after = _clone_state_dict(student)
    unchanged = all(torch.equal(before[name], after[name]) for name in before)

    validation_squared = 0.0
    validation_elements = 0
    student.eval()
    with torch.no_grad():
        for rows in validation_inputs.sequential_batches(batch_size=size):
            batch = validation_inputs.batch(rows, device=device)
            target = active_guard.call(teacher, batch).detach().to(torch.float64)
            prediction = active_guard.call(student, batch).to(torch.float64)
            row_sums = (
                torch.sum((prediction - target).square(), dim=1).cpu().numpy()
            )
            validation_squared += math.fsum(float(v) for v in row_sums.tolist())
            validation_elements += len(rows) * EDGES_PER_PHASE
    validation_loss = validation_squared / validation_elements
    energy = target_energy.mean_square
    return {
        "schema": STREAMING_MEMORY_VERSION + "-exact-model-null",
        "schema_version": 1,
        "target_energy": energy,
        "update_zero_loss": loss_value,
        "update_zero_validation_loss": validation_loss,
        "update_zero_gradients_exact": int(gradient_exact),
        "parameters_bitwise_unchanged": int(unchanged),
        "selected_update": 0,
        "optimizer_step_count": 1,
        "passed": int(
            energy > 0.0
            and loss_value == 0.0
            and validation_loss == 0.0
            and gradient_exact
            and unchanged
        ),
        "model_call_batches": active_guard.record(),
    }


def predict_to_cpu(
    model: nn.Module,
    inputs: HostInputStore,
    *,
    device: str | torch.device,
    guard: ModelCallBatchGuard | None = None,
    batch_size: int = MAXIMUM_MODEL_FORWARD_BATCH_SIZE,
    output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict into one preallocated CPU buffer; never concatenate on CUDA."""

    size = validate_batch_size(batch_size)
    active_guard = guard or ModelCallBatchGuard()
    output = np.empty((inputs.row_count, EDGES_PER_PHASE), dtype=output_dtype, order="C")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for rows in inputs.sequential_batches(batch_size=size):
                batch = inputs.batch(rows, device=device)
                prediction = active_guard.call(model, batch).to(torch.float64)
                values = prediction.detach().cpu().numpy().astype(output.dtype, copy=False)
                output[rows] = values
    finally:
        model.train(was_training)
    if not output.flags.c_contiguous or not output.flags.writeable:
        raise StreamingMemoryError("prediction output is not writable C-order storage")
    if not bool(np.isfinite(output).all()):
        raise StreamingMemoryError("prediction output is nonfinite")
    return output, active_guard.record()


__all__ = [
    "CanonicalRowSquareReducer",
    "HostInputStore",
    "HostLabelStore",
    "LabelOpenAuthorization",
    "MAXIMUM_MODEL_FORWARD_BATCH_SIZE",
    "ModelCallBatchGuard",
    "STREAMING_MEMORY_VERSION",
    "StreamingMemoryError",
    "canonical_row_square_reduction",
    "canonical_streamed_target_scale",
    "exact_null_batchwise_one_step",
    "open_external_input_store",
    "open_external_label_store",
    "predict_to_cpu",
    "stream_target_metrics",
    "stream_zero_initialization",
    "synthetic_target_provider",
    "synthetic_training_step",
    "validate_batch_size",
]
