from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mnist import d0_jacobi_rb_cuda_controls as controls
from mnist.d0_jacobi_artifacts import config_fingerprint
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_multipath import ExactMultipathCapturePayload
from mnist import d0_jacobi_rb_physical_coarse_signal_panel as panel


def _payload(path_ids: tuple[int, ...], start_step: int = 8) -> ExactMultipathCapturePayload:
    block_count = 8 * 7
    targets = np.empty((block_count, len(path_ids), 392), dtype=np.float64)
    for block in range(block_count):
        for path_index in range(len(path_ids)):
            targets[block, path_index] = (
                1_000.0 * path_index + 10.0 * block + np.arange(392)
            )
    later = np.full_like(targets, 0.5)
    codes = np.full(targets.shape, 15, dtype=np.uint8)
    states = np.full(
        (block_count, len(path_ids), 784), 1.0 / 784.0, dtype=np.float64
    )
    for value in (targets, later, codes, states):
        value.setflags(write=False)
    return ExactMultipathCapturePayload(
        path_ids=path_ids,
        start_step=start_step,
        outer_steps=tuple(
            step for step in range(start_step, start_step + 8) for _ in range(7)
        ),
        phases=tuple(phase for _step in range(8) for phase in range(7)),
        later_head_fractions=later,
        denoising_targets=targets,
        certificate_codes=codes,
        post_phase_states=states,
    )


def test_selected_target_contribution_preserves_binary64_rows() -> None:
    payload = _payload((7, 9))
    result = panel.selected_target_contribution(
        payload, selected_outer_step=15, expected_path_ids=(7, 9)
    )
    assert result.shape == (2, 7, 392)
    assert result.dtype == np.float64
    assert result.flags.c_contiguous and result.flags.writeable
    expected = np.transpose(payload.denoising_targets[-7:], (1, 0, 2))
    np.testing.assert_array_equal(result, expected)


def test_reduce_selected_contributions_matches_explicit_slow_sum() -> None:
    contributions = []
    for observation, step in enumerate(panel.SELECTED_OUTER_STEPS):
        value = np.empty((3, 7, 392), dtype=np.float64)
        for path_index in range(3):
            for phase in range(7):
                value[path_index, phase] = (
                    path_index
                    + 0.01 * phase
                    + 0.001 * observation
                    + np.arange(392, dtype=np.float64) * 1.0e-7
                )
        contributions.append((step, value))
    actual = panel.reduce_selected_contributions(contributions, path_count=3)
    expected = np.empty_like(actual)
    for path_index in range(3):
        for quartile in range(4):
            for phase in range(7):
                for edge in range(392):
                    values = [
                        contributions[index][1][path_index, phase, edge]
                        for index in range(8 * quartile, 8 * (quartile + 1))
                    ]
                    total = np.float64(0.0)
                    for value in values:
                        total = np.float64(total + value)
                    expected[path_index, quartile, phase, edge] = total / 8.0
    np.testing.assert_array_equal(actual, expected)


def _fake_result(
    initial: np.ndarray, final: np.ndarray, *, path_ids: tuple[int, ...]
) -> SimpleNamespace:
    diagnostics = {
        "start_step": 8,
        "step_count": 8,
        "path_ids": list(path_ids),
        "group_sizes": [len(path_ids)],
        "transition_count": len(path_ids) * 8 * 7 * 392,
        "phase_state_trace_enabled": 1,
    }
    record = {
        "schema": "jacobi-rb-cuda-exact-multipath-v1-shard",
        "schema_version": 1,
        "path_records": [
            {
                "path_id": path_id,
                "input_state_sha256": controls._digest_arrays(initial[index]),
                "final_state_sha256": controls._digest_arrays(final[index]),
            }
            for index, path_id in enumerate(path_ids)
        ],
        "phase_state_records": [],
        "batch_output_sha256": "a" * 64,
        "batch_final_state_sha256": controls._digest_arrays(final),
        "batch_certificate_sha256": "b" * 64,
        "diagnostics": diagnostics,
    }
    return SimpleNamespace(
        committed_final_states=final,
        to_record=lambda: record,
    )


def test_committed_shard_resume_and_accumulator_corruption(tmp_path: Path) -> None:
    path_ids = (0xE5000, 0xE5001)
    initial = np.full((2, 784), 1.0 / 784.0, dtype=np.float64)
    final = initial.copy()
    sums = np.zeros((2, 4, 7, 392), dtype=np.float64)
    compensations = np.zeros_like(sums)
    counts = np.zeros(4, dtype=np.int16)
    contribution = np.arange(2 * 7 * 392, dtype=np.float64).reshape(2, 7, 392)
    panel.update_cell_accumulator(
        sums, compensations, counts, contribution, quartile=0
    )
    input_sums = np.zeros_like(sums)
    input_compensations = np.zeros_like(sums)
    input_counts = np.zeros_like(counts)
    result = _fake_result(initial, final, path_ids=path_ids)
    profile_hash = "d" * 64
    panel._persist_shard(
        tmp_path,
        panel="a",
        group_index=0,
        start_step=8,
        path_ids=path_ids,
        root_seed=261241,
        input_states=initial,
        scientific_config_sha256="b" * 64,
        path_plan_sha256="c" * 64,
        profile_sha256=profile_hash,
        result=result,
        input_accumulator_sha256=panel._accumulator_sha256(
            input_sums, input_compensations, input_counts
        ),
        cell_sums=sums,
        cell_compensations=compensations,
        cell_counts=counts,
        accumulator_expected=True,
        complete_pipeline_started_at=0.0,
    )
    (
        valid,
        restored,
        restored_sums,
        restored_compensations,
        restored_counts,
        metadata,
    ) = panel._valid_committed_shard(
        tmp_path,
        panel="a",
        group_index=0,
        start_step=8,
        path_ids=path_ids,
        root_seed=261241,
        expected_input_states=initial,
        expected_cell_sums=input_sums,
        expected_cell_compensations=input_compensations,
        expected_cell_counts=input_counts,
        accumulator_expected=True,
        scientific_config_sha256="b" * 64,
        path_plan_sha256="c" * 64,
        profile_sha256=profile_hash,
    )
    assert valid and metadata is not None
    assert metadata["physical_training_performed"] == 0
    assert metadata["sampling_performed"] == 0
    assert metadata["reverse_sampling_performed"] == 0
    assert restored is not None and restored.flags.writeable
    np.testing.assert_array_equal(restored, final)
    np.testing.assert_array_equal(restored_sums, sums)
    np.testing.assert_array_equal(restored_compensations, compensations)
    np.testing.assert_array_equal(restored_counts, counts)
    assert metadata["raw_target_observations_persisted"] == 0

    _state, accumulator_path, _metadata = panel._shard_paths(
        tmp_path, group_index=0, start_step=8
    )
    accumulator_path.write_bytes(b"corrupt")
    assert not panel._valid_committed_shard(
        tmp_path,
        panel="a",
        group_index=0,
        start_step=8,
        path_ids=path_ids,
        root_seed=261241,
        expected_input_states=initial,
        expected_cell_sums=input_sums,
        expected_cell_compensations=input_compensations,
        expected_cell_counts=input_counts,
        accumulator_expected=True,
        scientific_config_sha256="b" * 64,
        path_plan_sha256="c" * 64,
        profile_sha256=profile_hash,
    )[0]


def test_selected_schedule_is_exactly_eight_observations_per_quartile() -> None:
    selected = panel.validate_selected_outer_steps()
    assert selected == tuple(range(15, 512, 16))
    assert [
        sum(step // 128 == quartile for step in selected) for quartile in range(4)
    ] == [8, 8, 8, 8]
    assert config_fingerprint({"selected": list(selected)})


def test_reduced_panel_runner_integrates_capture_accumulation_and_resume(
    tmp_path: Path, monkeypatch
) -> None:
    selected = (7, 15, 23, 31)
    calls: list[int] = []

    def fake_scheduler(
        states,
        *,
        path_ids,
        start_step,
        capture_training_payload,
        **_kwargs,
    ):
        calls.append(int(start_step))
        initial = states.detach().cpu().numpy().copy()
        final = initial.copy()
        path_tuple = tuple(int(value) for value in path_ids)
        diagnostics = {
            "start_step": int(start_step),
            "step_count": 8,
            "path_ids": list(path_tuple),
            "group_sizes": [len(path_tuple)],
            "transition_count": len(path_tuple) * 8 * 7 * 392,
            "certified_count": len(path_tuple) * 8 * 7 * 392,
            "fallback_count": 0,
            "fallback_elapsed_seconds": 0.0,
            "maximum_mass_error": 0.0,
            "maximum_cuda_launch_lanes": len(path_tuple) * 392,
            "state_updates_device_resident": 1,
            "phase_state_trace_enabled": int(capture_training_payload),
            **{name: 0 for name in panel.FORBIDDEN_COUNTS},
        }
        record = {
            "schema": "jacobi-rb-cuda-exact-multipath-v1-shard",
            "schema_version": 1,
            "path_records": [
                {
                    "path_id": path_id,
                    "input_state_sha256": controls._digest_arrays(initial[index]),
                    "final_state_sha256": controls._digest_arrays(final[index]),
                }
                for index, path_id in enumerate(path_tuple)
            ],
            "phase_state_records": [],
            "batch_output_sha256": f"{start_step:064x}",
            "batch_final_state_sha256": controls._digest_arrays(final),
            "batch_certificate_sha256": "b" * 64,
            "diagnostics": diagnostics,
        }
        return SimpleNamespace(
            committed_final_states=final,
            capture_payload=(
                _payload(path_tuple, start_step=int(start_step))
                if capture_training_payload
                else None
            ),
            to_record=lambda: record,
        )

    monkeypatch.setattr(panel, "PANEL_PATH_COUNT", 8)
    monkeypatch.setattr(panel, "OUTER_STEPS", 32)
    monkeypatch.setattr(panel, "validate_selected_outer_steps", lambda: selected)
    monkeypatch.setattr(panel, "run_exact_multipath_shard", fake_scheduler)
    path_ids = tuple(range(0xE5200, 0xE5208))
    target = np.full(784, 1.0 / 784.0, dtype=np.float64)
    kwargs = {
        "panel": "a",
        "path_ids": path_ids,
        "mixed_target": target,
        "root_seed": 261241,
        "profile": JacobiRBCudaProfile(),
        "device": torch.device("cpu"),
        "scientific_config_sha256": "a" * 64,
        "path_plan_sha256": "c" * 64,
    }
    first = panel.run_physical_panel(tmp_path, **kwargs)
    assert calls == [0, 8, 16, 24]
    assert first.cell_means.shape == (8, 4, 7, 392)
    assert first.metrics["raw_target_observations_persisted"] == 0
    assert not list(tmp_path.rglob("*contribution*.npz"))

    calls.clear()
    replay = panel.run_physical_panel(tmp_path, **kwargs)
    assert calls == []
    np.testing.assert_array_equal(replay.cell_means, first.cell_means)
