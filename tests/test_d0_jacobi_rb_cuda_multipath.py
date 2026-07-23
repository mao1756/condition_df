from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_controls import canonical_transition_ids
from mnist.d0_jacobi_rb_cuda_multipath import (
    EDGES_PER_PHASE,
    FROZEN_PROJECTION_GROUP_SIZES,
    FROZEN_PROJECTION_PATH_COUNT,
    FROZEN_VALIDATION_GROUP_SIZES,
    MAX_PATHS_PER_GROUP,
    SHARD_STEPS,
    canonical_same_phase_transition_ids,
    run_exact_multipath_shard,
    run_frozen_projection_shard,
)


class _RecordingSampler:
    def __init__(self, *, record_calls: bool = True) -> None:
        self.record_calls = bool(record_calls)
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(
        self,
        x: torch.Tensor,
        exposure: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        ids = kwargs["transition_ids"]
        assert isinstance(ids, torch.Tensor)
        if self.record_calls:
            self.calls.append(
                {
                    "x": x.detach().clone(),
                    "exposure": exposure.detach().clone(),
                    "ids": ids.detach().clone(),
                }
            )
        count = int(x.numel())
        # A purely lane-local deterministic stand-in: call shape and order do
        # not affect any output bit.
        jitter = torch.remainder(ids.to(torch.int64), 17).to(torch.float64) * 2.0**-42
        later = torch.clamp(x + jitter, 0.0, 1.0)
        target = later - x
        zeros_i64 = torch.zeros((), dtype=torch.int64, device=x.device)
        zeros_f64 = torch.zeros((), dtype=torch.float64, device=x.device)
        return SimpleNamespace(
            later_head_fraction=later,
            denoising_target=target,
            certificate_codes=torch.full(
                (count,), 15, dtype=torch.uint8, device=x.device
            ),
            fallback_mask=torch.zeros(count, dtype=torch.bool, device=x.device),
            strengthened_mask=torch.zeros(
                count, dtype=torch.bool, device=x.device
            ),
            arb_fallback_reason_codes=torch.zeros(
                count, dtype=torch.uint8, device=x.device
            ),
            mode_counts=torch.full(
                (count,), 128, dtype=torch.int32, device=x.device
            ),
            prefix_bits=torch.full(
                (count,), 64, dtype=torch.int32, device=x.device
            ),
            diagnostics={
                "arb_fallback_elapsed_seconds": zeros_f64,
                "fused_authorizer_elapsed_seconds": zeros_f64,
                "candidate_elapsed_seconds": zeros_f64,
                "maximum_cuda_launch_lanes": torch.as_tensor(
                    count, dtype=torch.int64, device=x.device
                ),
                "fused_authorizer_launch_count": torch.ones(
                    (), dtype=torch.int64, device=x.device
                ),
                "resource_cap_count": zeros_i64,
                "invalid_density_count": zeros_i64,
                "approximation_count": zeros_i64,
                "correction_count": zeros_i64,
                "floor_count": zeros_i64,
                "limiter_count": zeros_i64,
                "renormalization_count": zeros_i64,
                "nonfinite_count": zeros_i64,
            },
        )


def _states(path_count: int) -> torch.Tensor:
    generator = np.random.Generator(np.random.Philox(261131))
    values = np.stack(
        [generator.dirichlet(np.ones(28 * 28, dtype=np.float64)) for _ in range(path_count)]
    )
    return torch.as_tensor(values, dtype=torch.float64).contiguous()


def test_frozen_group_contract_exactly_covers_64_paths() -> None:
    assert MAX_PATHS_PER_GROUP == 10
    assert SHARD_STEPS == 8
    assert FROZEN_VALIDATION_GROUP_SIZES == (10, 4)
    assert FROZEN_PROJECTION_GROUP_SIZES == (10, 10, 10, 10, 10, 10, 4)
    assert FROZEN_PROJECTION_PATH_COUNT == 64
    assert sum(FROZEN_PROJECTION_GROUP_SIZES) == 64
    assert max(FROZEN_PROJECTION_GROUP_SIZES) * EDGES_PER_PHASE == 3920


def test_same_phase_ids_are_path_major_canonical_and_batch_invariant() -> None:
    device = torch.device("cpu")
    ids = canonical_same_phase_transition_ids(
        [3, 61], outer_step=127, phase=5, device=device
    )
    expected = torch.cat(
        [
            canonical_transition_ids(
                path=path,
                outer_step=127,
                phase=5,
                edge_start=0,
                count=392,
                device=device,
            )
            for path in (3, 61)
        ]
    )
    assert ids.dtype == torch.uint64
    assert ids.is_contiguous()
    assert torch.equal(ids, expected)
    assert torch.unique(ids).numel() == ids.numel()
    assert torch.equal(
        ids[:392],
        canonical_same_phase_transition_ids(
            [3], outer_step=127, phase=5, device=device
        ),
    )


def test_batched_and_serial_group_schedules_are_bit_and_hash_identical() -> None:
    states = _states(3)
    paths = (2, 7, 11)
    batched_sampler = _RecordingSampler()
    serial_sampler = _RecordingSampler()
    batched = run_exact_multipath_shard(
        states,
        path_ids=paths,
        start_step=0,
        root_seed=261131,
        profile=JacobiRBCudaProfile(),
        group_sizes=(3,),
        sampler=batched_sampler,
        capture_phase_state_trace=True,
    )
    serial = run_exact_multipath_shard(
        states,
        path_ids=paths,
        start_step=0,
        root_seed=261131,
        profile=JacobiRBCudaProfile(),
        group_sizes=(1, 1, 1),
        sampler=serial_sampler,
        capture_phase_state_trace=True,
    )

    assert torch.equal(batched.final_states, serial.final_states)
    assert len(batched_sampler.calls) == 8 * 7
    assert len(serial_sampler.calls) == 3 * 8 * 7
    assert batched.diagnostics["maximum_backend_call_size"] == 3 * 392
    assert batched.diagnostics["backend_call_count"] == 8 * 7
    assert batched.diagnostics["evolving_state_host_roundtrip_count"] == 0
    assert batched.diagnostics["shard_summary_synchronization_count"] == 1
    assert batched.diagnostics["uncertified_count"] == 0
    assert batched.diagnostics["fallback_count"] == 0
    assert batched.diagnostics["certificate_code_counts"] == {"15": 3 * 8 * 7 * 392}
    assert batched.diagnostics["mode_count_counts"] == {"128": 3 * 8 * 7 * 392}
    assert batched.diagnostics["prefix_bit_counts"] == {"64": 3 * 8 * 7 * 392}
    assert batched.diagnostics["arb_fallback_reason_code_counts"] == {
        "0": 3 * 8 * 7 * 392
    }
    assert batched.diagnostics["shard_summary_device_to_host_transfer_count"] == 1
    assert batched.diagnostics["maximum_mass_error"] <= 2.0e-12
    assert batched.batch_output_sha256 == serial.batch_output_sha256
    assert batched.batch_final_state_sha256 == serial.batch_final_state_sha256
    assert batched.phase_state_records == serial.phase_state_records
    assert len(batched.phase_state_records) == 8 * 7
    assert batched.diagnostics["phase_state_trace_record_count"] == 8 * 7
    assert batched.diagnostics["evolving_state_host_roundtrip_count"] == 0
    assert all(
        len(record.path_state_sha256_by_id) == 3
        for record in batched.phase_state_records
    )
    for left, right in zip(
        batched.path_records, serial.path_records, strict=True
    ):
        assert left.path_id == right.path_id
        assert left.input_state_sha256 == right.input_state_sha256
        assert left.output_sha256 == right.output_sha256
        assert left.final_state_sha256 == right.final_state_sha256
        assert left.certificate_sha256 == right.certificate_sha256
        assert left.certified_count == 8 * 7 * 392
        assert left.maximum_mode_count == 128
        assert left.maximum_prefix_bits == 64
        assert left.certificate_code_counts == {"15": 8 * 7 * 392}
        assert left.mode_count_counts == {"128": 8 * 7 * 392}
        assert left.prefix_bit_counts == {"64": 8 * 7 * 392}
        assert left.arb_fallback_reason_code_counts == {"0": 8 * 7 * 392}


def test_batch_hashes_are_path_id_canonical_under_permutation_and_regrouping() -> None:
    states = _states(3)
    original_paths = (8, 2, 5)
    original = run_exact_multipath_shard(
        states,
        path_ids=original_paths,
        start_step=24,
        root_seed=261131,
        profile=JacobiRBCudaProfile(),
        group_sizes=(3,),
        sampler=_RecordingSampler(record_calls=False),
        capture_phase_state_trace=True,
    )
    permutation = torch.as_tensor((2, 0, 1), dtype=torch.int64)
    permuted_paths = tuple(original_paths[index] for index in permutation.tolist())
    permuted = run_exact_multipath_shard(
        states.index_select(0, permutation).contiguous(),
        path_ids=permuted_paths,
        start_step=24,
        root_seed=261131,
        profile=JacobiRBCudaProfile(),
        group_sizes=(1, 2),
        sampler=_RecordingSampler(record_calls=False),
        capture_phase_state_trace=True,
    )

    assert original.batch_output_sha256 == permuted.batch_output_sha256
    assert original.batch_final_state_sha256 == permuted.batch_final_state_sha256
    assert original.phase_state_records == permuted.phase_state_records
    original_by_path = {record.path_id: record for record in original.path_records}
    permuted_by_path = {record.path_id: record for record in permuted.path_records}
    for path_id in original_paths:
        assert original_by_path[path_id] == permuted_by_path[path_id]


def test_phase_trace_directly_matches_independent_p1_execution() -> None:
    states = _states(3)
    path_ids = (13, 4, 29)
    batched = run_exact_multipath_shard(
        states,
        path_ids=path_ids,
        start_step=32,
        root_seed=261131,
        profile=JacobiRBCudaProfile(),
        group_sizes=(3,),
        sampler=_RecordingSampler(record_calls=False),
        capture_phase_state_trace=True,
    )
    batched_trace = [dict(record.path_state_sha256_by_id) for record in batched.phase_state_records]
    for index, path_id in enumerate(path_ids):
        independent = run_exact_multipath_shard(
            states[index : index + 1].contiguous(),
            path_ids=(path_id,),
            start_step=32,
            root_seed=261131,
            profile=JacobiRBCudaProfile(),
            group_sizes=(1,),
            sampler=_RecordingSampler(record_calls=False),
            capture_phase_state_trace=True,
        )
        assert len(independent.phase_state_records) == len(batched.phase_state_records)
        assert [
            record.path_state_sha256_by_id[0][1]
            for record in independent.phase_state_records
        ] == [record[path_id] for record in batched_trace]
        assert torch.equal(
            independent.final_states[0], batched.final_states[index]
        )


def test_committed_final_states_are_read_only_and_isolated_from_device_state() -> None:
    result = run_exact_multipath_shard(
        _states(1),
        path_ids=(3,),
        start_step=0,
        root_seed=261131,
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(record_calls=False),
    )
    device_snapshot = result.final_states.detach().cpu().numpy().copy()
    np.testing.assert_array_equal(result.committed_final_states, device_snapshot)
    assert result.committed_final_states.flags.c_contiguous
    assert not result.committed_final_states.flags.writeable
    assert not np.shares_memory(
        result.committed_final_states,
        result.final_states.detach().cpu().numpy(),
    )
    result.final_states[0, 0] = result.final_states[0, 0] + 1.0
    np.testing.assert_array_equal(result.committed_final_states, device_snapshot)
    with pytest.raises(ValueError, match="read-only"):
        result.committed_final_states[0, 0] = 0.0
    assert "committed_final_states" not in result.to_record()


def test_groups_are_phase_serial_and_each_call_is_path_major() -> None:
    sampler = _RecordingSampler()
    paths = (4, 6, 9)
    result = run_exact_multipath_shard(
        _states(3),
        path_ids=paths,
        start_step=16,
        root_seed=5,
        profile=JacobiRBCudaProfile(),
        group_sizes=(2, 1),
        sampler=sampler,
    )
    assert result.diagnostics["group_sizes"] == [2, 1]
    assert len(sampler.calls) == 8 * 7 * 2
    expected = ((16, 0, (4, 6)), (16, 0, (9,)), (16, 1, (4, 6)))
    for call, (outer_step, phase, call_paths) in zip(
        sampler.calls[:3], expected, strict=True
    ):
        expected_ids = canonical_same_phase_transition_ids(
            call_paths,
            outer_step=outer_step,
            phase=phase,
            device=torch.device("cpu"),
        )
        assert torch.equal(call["ids"], expected_ids)
        assert call["x"].shape == (len(call_paths) * 392,)


def test_frozen_projection_runs_six_tens_and_one_four_under_lane_cap() -> None:
    sampler = _RecordingSampler(record_calls=False)
    result = run_frozen_projection_shard(
        _states(64),
        start_step=0,
        root_seed=261131,
        profile=JacobiRBCudaProfile(),
        sampler=sampler,
    )
    diagnostics = result.diagnostics
    assert result.final_states.shape == (64, 784)
    assert result.final_states.device.type == "cpu"
    assert diagnostics["path_ids"] == list(range(64))
    assert diagnostics["group_sizes"] == [10, 10, 10, 10, 10, 10, 4]
    assert diagnostics["backend_call_count"] == 8 * 7 * 7
    assert diagnostics["maximum_backend_call_size"] == 3920
    assert diagnostics["maximum_cuda_launch_lanes"] == 3920
    assert diagnostics["transition_count"] == 64 * 8 * 7 * 392
    assert diagnostics["certified_count"] == diagnostics["transition_count"]
    assert diagnostics["fused_authorizer_launch_count"] == 8 * 7 * 7
    assert len(result.path_records) == 64
    assert len({record.certificate_sha256 for record in result.path_records}) == 64
    assert len(result.batch_output_sha256) == 64
    assert len(result.batch_final_state_sha256) == 64
    assert len(result.batch_certificate_sha256) == 64
    assert result.to_record()["diagnostics"]["path_count"] == 64


@pytest.mark.parametrize(
    ("paths", "groups", "message"),
    [
        (tuple(range(11)), None, "explicit group schedule"),
        (tuple(range(11)), (11,), "at most ten"),
        (tuple(range(11)), (10,), "partition"),
        ((1, 1), (2,), "unique"),
    ],
)
def test_invalid_path_group_contracts_fail_closed(
    paths: tuple[int, ...], groups: tuple[int, ...] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_exact_multipath_shard(
            _states(len(paths)),
            path_ids=paths,
            start_step=0,
            root_seed=1,
            profile=JacobiRBCudaProfile(),
            group_sizes=groups,
            sampler=_RecordingSampler(record_calls=False),
        )


def test_state_and_restart_boundary_contracts_fail_closed() -> None:
    sampler = _RecordingSampler(record_calls=False)
    with pytest.raises(ValueError, match="exactly eight"):
        run_exact_multipath_shard(
            _states(1),
            path_ids=(0,),
            start_step=0,
            root_seed=1,
            profile=JacobiRBCudaProfile(),
            sampler=sampler,
            step_count=1,
        )
    with pytest.raises(ValueError, match="restart boundary"):
        run_exact_multipath_shard(
            _states(1),
            path_ids=(0,),
            start_step=1,
            root_seed=1,
            profile=JacobiRBCudaProfile(),
            sampler=sampler,
        )
    with pytest.raises(ValueError, match="20-bit"):
        run_exact_multipath_shard(
            _states(1),
            path_ids=(1 << 20,),
            start_step=0,
            root_seed=1,
            profile=JacobiRBCudaProfile(),
            sampler=sampler,
        )
    with pytest.raises(ValueError, match="contiguous"):
        run_exact_multipath_shard(
            _states(2).t(),
            path_ids=(0, 1),
            start_step=0,
            root_seed=1,
            profile=JacobiRBCudaProfile(),
            sampler=sampler,
        )


def test_persisted_resume_chain_replays_and_recovers_only_corrupt_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mnist import diag_d0_jacobi_rb_cuda_multipath_confirmation as workflow

    run_dir = tmp_path
    args = SimpleNamespace(device="cpu", root_seed=261131, no_progress=True)
    scheduler_calls: list[int] = []

    def run_with_fake_sampler(
        states: torch.Tensor, **kwargs: object
    ) -> object:
        scheduler_calls.append(int(kwargs["start_step"]))
        return run_exact_multipath_shard(
            states,
            **kwargs,
            sampler=_RecordingSampler(record_calls=False),
        )

    monkeypatch.setattr(
        workflow, "run_exact_multipath_shard", run_with_fake_sampler
    )
    uninterrupted = workflow._run_performance_family(
        run_dir,
        args,
        family="pilot",
        outer_steps=16,
        repeats=1,
    )
    workflow._validate_completed_shard_family(
        run_dir, args, family="pilot", outer_steps=16, repeats=1
    )
    assert scheduler_calls == [0, 8, 0, 8]
    uninterrupted_tail = next(
        row
        for row in uninterrupted
        if int(row["group_size"]) == 10 and int(row["start_step"]) == 8
    )
    tail_path = (
        run_dir
        / "multipath_shards"
        / "pilot"
        / "b10-repeat-00-steps-008-015.json"
    )
    assert tail_path.is_file()
    assert tail_path.with_suffix(".npz").is_file()

    # A fully committed family is skipped without entering the scheduler.
    scheduler_calls.clear()
    skipped = workflow._run_performance_family(
        run_dir,
        args,
        family="pilot",
        outer_steps=16,
        repeats=1,
    )
    assert scheduler_calls == []
    assert [row["chain_sha256"] for row in skipped] == [
        row["chain_sha256"] for row in uninterrupted
    ]

    # Keep the tail internally hashed but bind it to the wrong predecessor.
    # The loader must reject that chain, reload the valid step-8 NPZ, and
    # recompute only steps 8..15.  Matching hashes then prove persisted resume
    # equality against the first run's uninterrupted device carry.
    payload = workflow._load(tail_path)
    payload["row"]["previous_shard_sha256"] = "0" * 64
    payload["row_sha256"] = workflow.config_fingerprint(payload["row"])
    workflow.atomic_write_json(tail_path, payload)
    with pytest.raises(workflow.ArtifactCompatibilityError):
        workflow._validate_completed_shard_family(
            run_dir, args, family="pilot", outer_steps=16, repeats=1
        )
    scheduler_calls.clear()
    recovered = workflow._run_performance_family(
        run_dir,
        args,
        family="pilot",
        outer_steps=16,
        repeats=1,
    )
    assert scheduler_calls == [8]
    recovered_tail = next(
        row
        for row in recovered
        if int(row["group_size"]) == 10 and int(row["start_step"]) == 8
    )
    for name in (
        "batch_output_sha256",
        "batch_final_state_sha256",
        "batch_certificate_sha256",
        "persisted_final_states_sha256",
        "chain_sha256",
    ):
        assert recovered_tail[name] == uninterrupted_tail[name]
    workflow._validate_completed_shard_family(
        run_dir, args, family="pilot", outer_steps=16, repeats=1
    )
