from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist import d0_jacobi_rb_cuda_controls as cuda_controls
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_controls import (
    FULL_PATH_REPEATS,
    FULL_PATH_TRANSITIONS,
    MAX_CUDA_CHUNK_SIZE,
    PARENT_REPLAY_COUNT,
    STEPS_PER_SHARD,
    benchmark_shard_ranges,
    benchmark_input_block,
    canonical_transition_ids,
    certificate_panel_plan,
    deterministic_fresh_certificate_inputs,
    kernel_benchmark_plan,
    run_certificate_panel,
    run_stateful_path_shard,
)


def _fake_cuda_result(x: torch.Tensor, exposure: torch.Tensor, **_kwargs):
    later = torch.clamp(x + 0.01 * exposure, 0.0, 1.0)
    target = torch.zeros_like(x)
    count = x.numel()
    return SimpleNamespace(
        later_head_fraction=later,
        denoising_target=target,
        certificate_codes=torch.full(
            (count,), 15, dtype=torch.uint8, device=x.device
        ),
        quantile_lower=later,
        quantile_upper=later,
        target_lower=target,
        target_upper=target,
        fallback_mask=torch.zeros(count, dtype=torch.bool, device=x.device),
        candidate_match_mask=torch.ones(count, dtype=torch.bool, device=x.device),
        prefix_bits=torch.full(
            (count,), 64, dtype=torch.int32, device=x.device
        ),
        diagnostics={
            "arb_fallback_elapsed_seconds": torch.zeros(
                (), dtype=torch.float64, device=x.device
            ),
            "fused_authorizer_elapsed_seconds": torch.zeros(
                (), dtype=torch.float64, device=x.device
            ),
            "candidate_elapsed_seconds": torch.zeros(
                (), dtype=torch.float64, device=x.device
            ),
            "maximum_cuda_launch_lanes": torch.as_tensor(
                count, dtype=torch.int64, device=x.device
            ),
            "fused_authorizer_launch_count": torch.ones(
                (), dtype=torch.int64, device=x.device
            ),
        },
    )


def test_production_plans_are_frozen_and_chunk_capped() -> None:
    certificate = certificate_panel_plan()
    benchmark = kernel_benchmark_plan()
    assert certificate.parent_replay_count == PARENT_REPLAY_COUNT == 294
    assert certificate.fresh_count == 512
    assert benchmark.warmup_transitions == 4_096
    assert benchmark.throughput_transitions == 65_536
    assert benchmark.throughput_repeats == 3
    assert benchmark.full_path_transitions == FULL_PATH_TRANSITIONS == 1_404_928
    assert benchmark.full_path_repeats == FULL_PATH_REPEATS == 3
    assert benchmark.chunk_size == MAX_CUDA_CHUNK_SIZE == 4_096
    assert benchmark.steps_per_shard == STEPS_PER_SHARD == 8
    with pytest.raises(ValueError):
        certificate_panel_plan(parent_replay_count=1)
    with pytest.raises(ValueError):
        kernel_benchmark_plan(chunk_size=4_097, test_only=True)


def test_benchmark_rate_uses_complete_repeat_and_keeps_shard_min_advisory() -> None:
    rows = [
        {
            "repeat": 0, "transition_count": 50,
            "wall_elapsed_seconds": 0.50, "transitions_per_second": 100.0,
        },
        {
            "repeat": 0, "transition_count": 50,
            "wall_elapsed_seconds": 0.01, "transitions_per_second": 5_000.0,
        },
        {
            "repeat": 1, "transition_count": 40,
            "wall_elapsed_seconds": 0.10, "transitions_per_second": 400.0,
        },
        {
            "repeat": 1, "transition_count": 60,
            "wall_elapsed_seconds": 0.30, "transitions_per_second": 200.0,
        },
    ]
    metrics = cuda_controls.summarize_benchmark(
        rows, expected_transitions=100, expected_repeats=2,
    )
    assert metrics["full_api_completed_pass"] == 1
    assert metrics["slowest_transitions_per_second"] == pytest.approx(100.0 / 0.51)
    assert metrics["slowest_shard_transitions_per_second"] == 100.0
    assert metrics["repeat_transitions_per_second"]["0"] == pytest.approx(100.0 / 0.51)
    assert metrics["repeat_transitions_per_second"]["1"] == pytest.approx(250.0)
    assert metrics["repeat_wall_elapsed_seconds"] == pytest.approx(
        {"0": 0.51, "1": 0.40}
    )

    incomplete = cuda_controls.summarize_benchmark(
        rows[:-1], expected_transitions=100, expected_repeats=2,
    )
    assert incomplete["full_api_completed_pass"] == 0
    assert incomplete["slowest_transitions_per_second"] == 0.0


def test_fresh_panel_is_64_dirichlet_grids_by_four_colors_by_two_durations() -> None:
    x, exposure = deterministic_fresh_certificate_inputs(512, 261_132)
    replay_x, replay_exposure = deterministic_fresh_certificate_inputs(512, 261_132)
    assert x.shape == exposure.shape == (512,)
    assert np.array_equal(x, replay_x)
    assert np.array_equal(exposure, replay_exposure)
    assert np.all((x > 0.0) & (x < 1.0))
    reshaped = exposure.reshape(64, 4, 2)
    assert np.all(reshaped[:, :, 0] * 2.0 == reshaped[:, :, 1])


def test_certificate_panel_test_double_is_measured_not_fabricated() -> None:
    plan = certificate_panel_plan(
        root_seed=17, test_only=True, parent_replay_count=2, fresh_count=3,
    )
    parent = [
        {"head_fraction": 0.2, "exposure": 0.1},
        {"head_fraction": 0.7, "exposure": 0.2},
    ]
    rows, metrics = run_certificate_panel(
        parent,
        device=torch.device("cpu"),
        profile=JacobiRBCudaProfile(),
        plan=plan,
        sampler=_fake_cuda_result,
    )
    assert len(rows) == 5
    assert metrics["certificate_fraction"] == 1.0
    assert metrics["uncertified_count"] == 0
    assert metrics["cuda_certificate_fallback_fraction"] == 0.0
    for name in (
        "parent_replay_z_bit_mismatch_count", "strengthening_hash_pass",
        "fresh_arb_enclosure_pass", "resource_cap_count", "invalid_density_count",
        "approximation_count", "floor_count", "limiter_count",
        "renormalization_count",
    ):
        assert name in metrics
    assert {row["panel"] for row in rows} == {"parent_replay", "fresh"}


def test_benchmark_ranges_and_stateful_shard_obey_hard_caps(monkeypatch) -> None:
    ranges = benchmark_shard_ranges(65_537, chunk_size=4_096)
    assert sum(count for _, count in ranges) == 65_537
    assert all(count <= 4_096 * 8 for _, count in ranges)
    def forbidden_host_helper(*_args, **_kwargs):
        raise AssertionError("a per-phase host conversion helper was called")

    # The old benchmark called each of these after every phase.  The fused
    # benchmark may cross to the host only after the complete shard.
    monkeypatch.setattr(cuda_controls, "_outputs", forbidden_host_helper)
    monkeypatch.setattr(cuda_controls, "_certified_mask", forbidden_host_helper)
    monkeypatch.setattr(cuda_controls, "_diagnostic_scalar", forbidden_host_helper)
    monkeypatch.setattr(cuda_controls, "_numpy", forbidden_host_helper)

    state, row = run_stateful_path_shard(
        np.full(28 * 28, 1.0 / (28 * 28)),
        start_step=0,
        step_count=8,
        repeat=0,
        root_seed=9,
        device=torch.device("cpu"),
        profile=JacobiRBCudaProfile(),
        sampler=_fake_cuda_result,
    )
    assert state.shape == (28 * 28,)
    assert row["transition_count"] == 8 * 7 * 392
    assert row["maximum_backend_call_size"] == 392
    assert row["maximum_cuda_launch_lanes"] == 392
    assert row["fused_authorizer_launch_count"] == 8 * 7
    assert row["uncertified_count"] == 0
    assert row["state_updates_device_resident"] == 1
    assert row["device_residency_metric_scope"].startswith(
        "evolving_state_and_matching_updates"
    )
    assert row["in_shard_host_roundtrip_count"] == 0
    assert row["shard_summary_synchronization_count"] == 1
    with pytest.raises(ValueError):
        run_stateful_path_shard(
            np.ones(28 * 28), start_step=0, step_count=9, repeat=0, root_seed=1,
            device=torch.device("cpu"), profile=JacobiRBCudaProfile(),
            sampler=_fake_cuda_result,
        )


def test_production_probe_inputs_and_ids_are_chunk_invariant() -> None:
    full = benchmark_input_block(32, 19, 0, device=torch.device("cpu"))
    left = benchmark_input_block(13, 19, 0, device=torch.device("cpu"))
    right = benchmark_input_block(19, 19, 13, device=torch.device("cpu"))
    for whole, first, second in zip(full, left, right, strict=True):
        assert torch.equal(whole, torch.cat([first, second]))
    x, exposure, transition_ids = full
    assert torch.all((x > 0.0) & (x < 1.0))
    assert torch.all(exposure > 0.0)
    assert transition_ids.dtype == torch.uint64
    assert torch.unique(transition_ids).numel() == transition_ids.numel()
    expected = canonical_transition_ids(
        path=0, outer_step=0, phase=0, edge_start=0, count=32,
        device=torch.device("cpu"),
    )
    assert torch.equal(transition_ids, expected)


def test_parent_fallback_is_advisory_but_fresh_fallback_authorizes() -> None:
    def sampler(x: torch.Tensor, exposure: torch.Tensor, **kwargs):
        result = _fake_cuda_result(x, exposure, **kwargs)
        ids = kwargs["transition_ids"].to(torch.int64)
        result.fallback_mask = ids < 2
        return result

    plan = certificate_panel_plan(
        root_seed=17, test_only=True, parent_replay_count=2, fresh_count=3,
    )
    parent = [
        {"head_fraction": 0.2, "exposure": 0.1},
        {"head_fraction": 0.7, "exposure": 0.2},
    ]
    _rows, metrics = run_certificate_panel(
        parent, device=torch.device("cpu"), profile=JacobiRBCudaProfile(),
        plan=plan, sampler=sampler,
    )
    assert metrics["parent_adversarial_fallback_count"] == 2
    assert metrics["fresh_fallback_count"] == 0
    assert metrics["cuda_certificate_fallback_fraction"] == 0.0
