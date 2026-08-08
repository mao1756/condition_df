from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mnist import d0_jacobi_rb_boundary_tangent_v3_memory as memory
from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE, ModelInputs


def _input_arrays(rows: int) -> dict[str, np.ndarray]:
    state = np.full((rows, 784), 1.0 / 784.0, dtype=np.float32)
    return {
        "sample_key": np.arange(rows, dtype=np.int64),
        "path_id": (100 + np.arange(rows) // 4).astype(np.int64),
        "outer_step": np.arange(rows, dtype=np.int16),
        "midpoint_index": np.zeros(rows, dtype=np.int8),
        "midpoint_fraction": np.full(rows, 1.0 / 16.0, dtype=np.float64),
        "later_full_state": state,
        "reverse_time": np.linspace(0.1, 0.9, rows, dtype=np.float64),
        "phase": np.zeros(rows, dtype=np.int8),
        "color": np.zeros(rows, dtype=np.int8),
        "duration": np.full(rows, 0.5, dtype=np.float64),
        "label": np.full(rows, 3, dtype=np.int64),
    }


def _store(rows: int, *, role: str = "train") -> memory.HostInputStore:
    return memory.HostInputStore.from_arrays(
        _input_arrays(rows), role=role, cache_root="."
    )


class _TinyModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.full((EDGES_PER_PHASE,), value, dtype=torch.float64)
        )

    def forward(self, inputs: ModelInputs) -> torch.Tensor:
        return inputs.reverse_time[:, None] * self.weight[None, :]


def _linear_target(inputs: ModelInputs) -> torch.Tensor:
    return inputs.reverse_time[:, None].repeat(1, EDGES_PER_PHASE)


def test_host_store_is_writable_c_order_and_materializes_only_selected_rows() -> None:
    source = _input_arrays(40)
    source["later_full_state"] = source["later_full_state"][:, ::-1]
    store = memory.HostInputStore.from_arrays(source, role="train")
    assert store.row_count == 40
    assert all(value.flags.c_contiguous for value in store.arrays.values())
    assert all(value.flags.writeable for value in store.arrays.values())

    batch = store.batch([5, 2, 5], device="cpu")
    assert batch.batch_size == 3
    assert batch.later_full_state.dtype == torch.float32
    assert batch.reverse_time.dtype == torch.float64
    assert batch.duration.dtype == torch.float32
    assert torch.equal(batch.reverse_time, torch.tensor([source["reverse_time"][5], source["reverse_time"][2], source["reverse_time"][5]], dtype=torch.float64))

    with pytest.raises(memory.StreamingMemoryError):
        store.batch(np.arange(33), device="cpu")
    with pytest.raises(memory.StreamingMemoryError):
        store.batch([-1], device="cpu")


def test_external_input_open_never_opens_labels_and_label_open_is_role_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"inputs": 0, "labels": 0}

    def input_loader(root: Path, role: str):
        calls["inputs"] += 1
        return _input_arrays(4), {"role": role, "input_row_count": 4}

    def label_loader(root: Path, role: str):
        calls["labels"] += 1
        return {
            "denoising_target": np.ones((4, EDGES_PER_PHASE), dtype=np.float64),
            "path_id": np.arange(4, dtype=np.int64),
        }, {"role": role, "label_row_count": 4}

    monkeypatch.setattr(memory, "load_eager_role_inputs", input_loader)
    monkeypatch.setattr(memory, "load_eager_role_labels", label_loader)
    store = memory.open_external_input_store(tmp_path, "train")
    assert store.row_count == 4
    assert calls == {"inputs": 1, "labels": 0}

    authorization = memory.LabelOpenAuthorization(
        tmp_path.resolve(), "train", "physical_training", "a" * 64
    )
    labels = memory.open_external_label_store(
        tmp_path, "train", authorization=authorization
    )
    assert labels.row_count == 4
    assert calls == {"inputs": 1, "labels": 1}
    assert labels.target_batch([0, 3], device="cpu").shape == (2, EDGES_PER_PHASE)

    wrong = memory.LabelOpenAuthorization(
        tmp_path.resolve(), "validation", "validation_selection", "b" * 64
    )
    with pytest.raises(memory.StreamingMemoryError):
        memory.open_external_label_store(tmp_path, "train", authorization=wrong)
    assert calls["labels"] == 1
    with pytest.raises(memory.StreamingMemoryError):
        memory.LabelOpenAuthorization(
            tmp_path.resolve(), "validation", "physical_training", "c" * 64
        )


def test_batch_guard_enforces_and_records_frozen_limit() -> None:
    guard = memory.ModelCallBatchGuard()
    model = _TinyModel()
    store = _store(33)
    prediction = guard.call(model, store.batch(np.arange(32), device="cpu"))
    assert prediction.shape == (32, EDGES_PER_PHASE)
    assert guard.record()["maximum_observed_batch_size"] == 32

    arrays = _input_arrays(33)
    oversized = ModelInputs(
        later_full_state=torch.from_numpy(arrays["later_full_state"]),
        reverse_time=torch.from_numpy(arrays["reverse_time"]),
        phase=torch.from_numpy(arrays["phase"].astype(np.int64)),
        color=torch.from_numpy(arrays["color"].astype(np.int64)),
        duration=torch.from_numpy(arrays["duration"].astype(np.float32)),
        label=torch.from_numpy(arrays["label"]),
    )
    with pytest.raises(memory.StreamingMemoryError):
        guard.call(model, oversized)


def test_canonical_square_reduction_is_batch_partition_invariant() -> None:
    values = np.arange(77, dtype=np.float64).reshape(11, 7) / 13.0
    first = memory.canonical_row_square_reduction(values, batch_size=3)
    second = memory.canonical_row_square_reduction(values, batch_size=11)
    expected_rows = np.sum(np.square(values), axis=1, dtype=np.float64)
    expected = math.fsum(float(value) for value in expected_rows.tolist())
    assert first["square_sum"] == second["square_sum"] == expected
    assert first["element_count"] == values.size
    assert first["rms"] == math.sqrt(expected / values.size)


def test_streamed_zero_scan_and_cpu_prediction_never_exceed_32() -> None:
    train = _store(70, role="train")
    validation = _store(35, role="validation")
    model = _TinyModel(0.0)
    guard = memory.ModelCallBatchGuard()
    record = memory.stream_zero_initialization(
        model,
        {"train": train, "validation": validation},
        device="cpu",
        guard=guard,
        baseline_provider=lambda inputs: torch.zeros(
            (inputs.batch_size, EDGES_PER_PHASE), dtype=torch.float64
        ),
    )
    assert record["passed"] == 1
    assert guard.observed_batch_sizes == [32, 32, 6, 32, 3]

    model.weight.data.fill_(1.0)
    prediction, prediction_record = memory.predict_to_cpu(
        model, train, device="cpu"
    )
    assert prediction.shape == (70, EDGES_PER_PHASE)
    assert prediction.flags.c_contiguous and prediction.flags.writeable
    assert prediction_record["maximum_observed_batch_size"] == 32
    np.testing.assert_array_equal(
        prediction[:, 0], train.row_array("reverse_time")
    )


def test_synthetic_scale_training_step_and_metrics_stream() -> None:
    train = _store(40, role="train")
    validation = _store(20, role="validation")
    scale, reduction = memory.canonical_streamed_target_scale(
        train, device="cpu", target_provider=_linear_target, batch_size=7
    )
    expected = np.repeat(
        train.row_array("reverse_time")[:, None], EDGES_PER_PHASE, axis=1
    )
    expected_scale = math.sqrt(float(np.mean(np.square(expected), dtype=np.float64)))
    assert math.isclose(scale, expected_scale, rel_tol=2.0e-15)
    assert reduction["element_count"] == expected.size

    model = _TinyModel(0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    guard = memory.ModelCallBatchGuard()
    update = memory.synthetic_training_step(
        model,
        optimizer,
        train,
        np.arange(32),
        scale=scale,
        device="cpu",
        guard=guard,
        target_provider=_linear_target,
    )
    assert math.isfinite(update["scaled_loss"])
    assert guard.maximum_observed_batch_size == 32

    model.weight.data.fill_(1.0)
    metrics = memory.stream_target_metrics(
        model,
        validation,
        device="cpu",
        target_provider=_linear_target,
    )
    assert metrics["model_mse"] == 0.0
    assert metrics["relative_mse"] == 0.0
    assert metrics["every_path_beats_zero"] == 1
    assert metrics["model_call_batches"]["maximum_observed_batch_size"] <= 32


def test_exact_null_accumulates_one_full_dataset_step_without_state_change() -> None:
    train = _store(67, role="train")
    validation = _store(35, role="validation")
    teacher = _TinyModel(0.75)
    student = _TinyModel(0.75)
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-3, weight_decay=0.0)
    before = student.weight.detach().clone()
    record = memory.exact_null_batchwise_one_step(
        teacher,
        student,
        optimizer,
        train,
        validation,
        device="cpu",
    )
    assert record["passed"] == 1
    assert record["optimizer_step_count"] == 1
    assert record["target_energy"] > 0.0
    assert record["update_zero_loss"] == 0.0
    assert record["update_zero_validation_loss"] == 0.0
    assert record["model_call_batches"]["maximum_observed_batch_size"] <= 32
    assert torch.equal(student.weight, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward_seam_respects_32_row_limit() -> None:
    store = _store(32)
    model = _TinyModel(0.0).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    guard = memory.ModelCallBatchGuard()
    torch.cuda.reset_peak_memory_stats()
    scale, _ = memory.canonical_streamed_target_scale(
        store, device="cuda", target_provider=_linear_target
    )
    memory.synthetic_training_step(
        model,
        optimizer,
        store,
        np.arange(32),
        scale=scale,
        device="cuda",
        guard=guard,
        target_provider=_linear_target,
    )
    assert guard.maximum_observed_batch_size == 32
    total = torch.cuda.get_device_properties(0).total_memory
    assert torch.cuda.max_memory_allocated() / total <= 0.80

