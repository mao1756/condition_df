from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import pytest
import torch

from mnist import d0_jacobi_rb_cuda as rb_cuda
from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    FORBIDDEN_DIAGNOSTICS,
    sample_midpoint_branches,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule import (
    BASE_TRANSITIONS_PER_PILOT_PATH,
    BoundaryTangentScheduleError,
    CONFIRMATION_COHORT_SIZES,
    MAXIMUM_LAUNCH_LANES,
    MAXIMUM_PROJECTED_SECONDS,
    MIDPOINT_TRANSITIONS_PER_PILOT_PATH,
    PILOT_PROFILE_NAMES,
    PROFILE_CACHE_P10,
    PROFILE_CACHE_P6,
    PROFILE_PATH_COUNTS,
    PROFILE_STREAM_P10,
    PROFILE_STREAM_P4,
    PROJECTED_BASE_TRANSITIONS,
    PROJECTED_MIDPOINT_TRANSITIONS,
    PROJECTED_TOTAL_TRANSITIONS,
    PilotRepeatRecord,
    TRAIN_VALIDATION_COHORT_SIZES,
    build_fused_launch_plan,
    expected_profile_transition_counts,
    frozen_path_plan,
    frozen_production_cohort_plan,
    frozen_repeat_order,
    project_frozen_schedule,
    sample_fused_midpoint_branches,
    split_co_scheduled_payload_by_role,
    validate_repeat_records,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_reverse_controller import controller_transition_ids


class _RecordingSampler:
    def __init__(self) -> None:
        self.ids: list[torch.Tensor] = []
        self.lane_counts: list[int] = []

    def __call__(self, head, exposure, *, rng_key, transition_ids, profile):
        del rng_key, profile
        self.ids.append(transition_ids.detach().clone())
        self.lane_counts.append(int(head.numel()))
        later = torch.clamp(head + exposure * 1.0e-3, 0.0, 1.0)
        target = later * (1.0 - later) - 0.125
        return SimpleNamespace(
            later_head_fraction=later,
            denoising_target=target,
            certificate_codes=torch.full_like(later, 0b1111, dtype=torch.uint8),
            mode_counts=torch.full_like(later, 32, dtype=torch.int32),
            prefix_bits=torch.full_like(later, 64, dtype=torch.int32),
            fallback_mask=torch.zeros_like(later, dtype=torch.bool),
            strengthened_mask=torch.zeros_like(later, dtype=torch.bool),
            diagnostics={
                "maximum_cuda_launch_lanes": int(head.numel()),
                "fused_authorizer_launch_count": 1,
            },
        )


def _states(paths: int) -> torch.Tensor:
    values = torch.arange(1, paths * 784 + 1, dtype=torch.float64).reshape(paths, 784)
    return (values / values.sum(dim=1, keepdim=True)).contiguous()


def _frozen_cuda_available() -> bool:
    if not torch.cuda.is_available() or rb_cuda._reference._arb is None:
        return False
    properties = torch.cuda.get_device_properties(0)
    return (
        str(torch.__version__) == "2.11.0+cu128"
        and str(torch.version.cuda) == "12.8"
        and (properties.major, properties.minor) == (12, 0)
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _forbidden(**updates: int) -> dict[str, int]:
    output = {name: 0 for name in FORBIDDEN_DIAGNOSTICS}
    output.update(updates)
    return output


def _repeat(
    profile: str,
    repeat: int,
    *,
    elapsed: float = 100.0,
    committed_bytes: int = 1_000,
    output_suffix: str = "stable",
    forbidden=None,
) -> PilotRepeatRecord:
    base, midpoint, total = expected_profile_transition_counts(profile)
    return PilotRepeatRecord(
        profile_name=profile,
        repeat_index=repeat,
        execution_order_index=frozen_repeat_order(repeat).index(profile),
        elapsed_seconds=elapsed,
        base_transition_count=base,
        midpoint_transition_count=midpoint,
        certified_count=total,
        fallback_count=0,
        fallback_elapsed_seconds=0.0,
        maximum_mass_error=1.0e-15,
        peak_memory_fraction=0.1,
        committed_bytes=committed_bytes,
        maximum_launch_lanes=3920,
        output_sha256=_digest(profile + "-output-" + output_suffix),
        final_state_sha256=_digest(profile + "-state-" + output_suffix),
        forbidden_counts=_forbidden() if forbidden is None else forbidden,
    )


def _panel(*, elapsed: float = 100.0) -> list[PilotRepeatRecord]:
    return [
        _repeat(profile, repeat, elapsed=elapsed)
        for repeat in range(3)
        for profile in frozen_repeat_order(repeat)
    ]


def test_frozen_cohort_and_path_plans_are_exact_and_role_sealed() -> None:
    assert TRAIN_VALIDATION_COHORT_SIZES == (10,) * 9 + (6,)
    assert CONFIRMATION_COHORT_SIZES == (10,) * 6 + (4,)
    path_plan = frozen_path_plan()
    all_ids = [value for values in path_plan["roles"].values() for value in values]
    assert len(all_ids) == len(set(all_ids))

    production = frozen_production_cohort_plan()
    assert [len(group) for group in production["train_validation"]] == list(
        TRAIN_VALIDATION_COHORT_SIZES
    )
    assert [len(group) for group in production["confirmation"]] == list(
        CONFIRMATION_COHORT_SIZES
    )
    roles = production["path_roles"]
    assert sum(value == "train" for value in roles.values()) == 64
    assert sum(value == "validation" for value in roles.values()) == 32
    assert sum(value == "confirmation" for value in roles.values()) == 64
    assert production["cross_role_artifact_commit"] == 0


def test_cross_role_payload_is_physically_split_before_commit() -> None:
    # The seventh 10-path cohort crosses the 64/32 train-validation boundary.
    paths = tuple(range(0xEC13C, 0xEC140)) + tuple(range(0xEC200, 0xEC206))
    tensor = torch.arange(30, dtype=torch.float64).reshape(10, 3)
    array = torch.arange(20, dtype=torch.int64).reshape(10, 2).numpy()
    split = split_co_scheduled_payload_by_role(
        paths, {"tensor": tensor, "array": array}
    )
    assert set(split) == {"train", "validation"}
    assert split["train"]["tensor"].shape == (4, 3)
    assert split["validation"]["tensor"].shape == (6, 3)
    assert split["train"]["array"].flags.c_contiguous
    assert split["validation"]["tensor"].is_contiguous()
    split["train"]["tensor"][0, 0] = -1.0
    split["train"]["array"][0, 0] = -1
    assert tensor[0, 0] == 0.0
    assert array[0, 0] == 0

    with pytest.raises(BoundaryTangentScheduleError, match="unknown path"):
        split_co_scheduled_payload_by_role((0,), {"tensor": tensor[:1]})


def test_repeat_orders_rotate_cyclically() -> None:
    assert frozen_repeat_order(0) == PILOT_PROFILE_NAMES
    assert frozen_repeat_order(1) == PILOT_PROFILE_NAMES[1:] + PILOT_PROFILE_NAMES[:1]
    assert frozen_repeat_order(2) == PILOT_PROFILE_NAMES[2:] + PILOT_PROFILE_NAMES[:2]
    with pytest.raises(BoundaryTangentScheduleError):
        frozen_repeat_order(3)


@pytest.mark.parametrize("paths", [1, 4, 6, 10])
def test_fused_launch_plan_is_contiguous_and_capped(paths: int) -> None:
    plan = build_fused_launch_plan(paths)
    assert plan.total_lanes == 8 * paths * 392
    assert plan.maximum_chunk_lanes <= MAXIMUM_LAUNCH_LANES
    assert plan.chunk_ranges[0][0] == 0
    assert plan.chunk_ranges[-1][1] == plan.total_lanes
    assert all(
        left[1] == right[0]
        for left, right in zip(plan.chunk_ranges, plan.chunk_ranges[1:])
    )
    with pytest.raises(BoundaryTangentScheduleError):
        build_fused_launch_plan(11)


def test_fused_midpoint_order_matches_exact_controller_ids() -> None:
    sampler = _RecordingSampler()
    paths = (0xEE000, 0xEE001, 0xEE002, 0xEE003, 0xEE004, 0xEE005)
    result = sample_fused_midpoint_branches(
        _states(len(paths)),
        path_ids=paths,
        outer_step=15,
        phase=3,
        profile=JacobiRBCudaProfile(),
        sampler=sampler,
    )
    observed = torch.cat(sampler.ids)
    expected = torch.stack(
        [
            controller_transition_ids(
                paths,
                outer_step=15,
                phase=3,
                reverse_microstep=midpoint,
                role="partial_phase_target_prefix",
                device="cpu",
            )
            for midpoint in range(8)
        ]
    ).reshape(-1)
    assert torch.equal(observed, expected)
    assert max(sampler.lane_counts) <= MAXIMUM_LAUNCH_LANES
    assert sampler.lane_counts == [4096, 4096, 4096, 4096, 2432]
    assert result.batch.later_full_state.shape == (8, 6, 784)
    assert result.batch.denoising_target.shape == (8, 6, 392)
    assert result.batch.certified_count == result.batch.transition_count


def test_fused_midpoint_is_bit_identical_to_legacy_calls() -> None:
    paths = (0xEE000, 0xEE001, 0xEE002, 0xEE003)
    states = _states(len(paths))
    legacy = sample_midpoint_branches(
        states,
        path_ids=paths,
        outer_step=31,
        phase=5,
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
    )
    fused = sample_fused_midpoint_branches(
        states,
        path_ids=paths,
        outer_step=31,
        phase=5,
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
    )
    assert torch.equal(fused.batch.later_full_state, legacy.later_full_state)
    assert torch.equal(fused.batch.later_head_fraction, legacy.later_head_fraction)
    assert torch.equal(fused.batch.denoising_target, legacy.denoising_target)
    assert torch.equal(fused.batch.certificate_codes, legacy.certificate_codes)
    assert fused.output_sha256() == legacy.output_sha256()


@pytest.mark.skipif(
    not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable"
)
def test_real_cuda_fused_p10_branch_is_certified_and_conservative() -> None:
    """Exercise the production authorizer through a full 10-path fused call."""

    paths = tuple(range(0xEE000, 0xEE00A))
    states = torch.full(
        (len(paths), 784),
        1.0 / 784.0,
        dtype=torch.float64,
        device="cuda",
    ).contiguous()
    result = sample_fused_midpoint_branches(
        states,
        path_ids=paths,
        outer_step=15,
        phase=3,
        profile=JacobiRBCudaProfile(),
    )
    torch.cuda.synchronize()

    assert result.batch.transition_count == 8 * 10 * 392
    assert result.batch.certified_count == result.batch.transition_count
    assert result.launch_plan.maximum_chunk_lanes == 4096
    assert result.launch_count == 8
    assert sum(result.batch.forbidden_counts.values()) == 0
    assert torch.isfinite(result.batch.denoising_target).all()

    later_mass = result.batch.later_full_state.sum(dim=2)
    reference_mass = states.sum(dim=1).reshape(1, -1)
    assert torch.max(torch.abs(later_mass - reference_mass)).item() <= 2.0e-12


def test_fused_results_are_path_permutation_invariant() -> None:
    paths = (0xEE000, 0xEE001, 0xEE002, 0xEE003, 0xEE004, 0xEE005)
    states = _states(len(paths))
    direct = sample_fused_midpoint_branches(
        states,
        path_ids=paths,
        outer_step=47,
        phase=2,
        sampler=_RecordingSampler(),
    )
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    permuted_paths = tuple(paths[index] for index in permutation.tolist())
    permuted = sample_fused_midpoint_branches(
        states.index_select(0, permutation).contiguous(),
        path_ids=permuted_paths,
        outer_step=47,
        phase=2,
        sampler=_RecordingSampler(),
    )
    inverse = torch.argsort(permutation)
    assert torch.equal(
        direct.batch.later_full_state,
        permuted.batch.later_full_state.index_select(1, inverse),
    )
    assert torch.equal(
        direct.batch.denoising_target,
        permuted.batch.denoising_target.index_select(1, inverse),
    )


def test_fused_cohort_matches_regrouped_singletons() -> None:
    paths = (0xEE000, 0xEE001, 0xEE002, 0xEE003, 0xEE004, 0xEE005)
    states = _states(len(paths))
    together = sample_fused_midpoint_branches(
        states,
        path_ids=paths,
        outer_step=63,
        phase=6,
        sampler=_RecordingSampler(),
    )
    singleton_states = []
    singleton_targets = []
    singleton_codes = []
    for index, path in enumerate(paths):
        result = sample_fused_midpoint_branches(
            states[index : index + 1].contiguous(),
            path_ids=(path,),
            outer_step=63,
            phase=6,
            sampler=_RecordingSampler(),
        )
        singleton_states.append(result.batch.later_full_state)
        singleton_targets.append(result.batch.denoising_target)
        singleton_codes.append(result.batch.certificate_codes)
    assert torch.equal(together.batch.later_full_state, torch.cat(singleton_states, dim=1))
    assert torch.equal(together.batch.denoising_target, torch.cat(singleton_targets, dim=1))
    assert torch.equal(together.batch.certificate_codes, torch.cat(singleton_codes, dim=1))


def test_fused_branch_aggregates_every_chunk_diagnostic() -> None:
    class DiagnosticSampler(_RecordingSampler):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            result.fallback_mask[0] = True
            result.strengthened_mask[-1] = True
            result.diagnostics.update(
                {
                    "arb_fallback_elapsed_seconds": 0.25,
                    "fused_authorizer_elapsed_seconds": 0.5,
                    "correction_count": 1,
                }
            )
            return result

    result = sample_fused_midpoint_branches(
        _states(10),
        path_ids=tuple(range(0xEE000, 0xEE00A)),
        outer_step=79,
        phase=0,
        sampler=DiagnosticSampler(),
    )
    chunks = len(result.launch_plan.chunk_ranges)
    assert int(result.batch.fallback_mask.sum().item()) == chunks
    assert int(result.batch.strengthened_mask.sum().item()) == chunks
    assert result.batch.forbidden_counts["correction_count"] == chunks
    assert result.batch.fallback_elapsed_seconds == 0.25 * chunks
    assert result.batch.backend_elapsed_seconds == 0.5 * chunks
    assert set(result.batch.forbidden_counts) == set(FORBIDDEN_DIAGNOSTICS)


def test_profile_transition_and_projection_arithmetic_is_exact() -> None:
    assert BASE_TRANSITIONS_PER_PILOT_PATH == 175_616
    assert MIDPOINT_TRANSITIONS_PER_PILOT_PATH == 87_808
    assert expected_profile_transition_counts(PROFILE_CACHE_P10) == (
        1_756_160,
        878_080,
        2_634_240,
    )
    assert expected_profile_transition_counts(PROFILE_CACHE_P6)[2] == 1_580_544
    assert expected_profile_transition_counts(PROFILE_STREAM_P4)[2] == 1_053_696
    assert PROJECTED_BASE_TRANSITIONS == 224_788_480
    assert PROJECTED_MIDPOINT_TRANSITIONS == 112_394_240
    assert PROJECTED_TOTAL_TRANSITIONS == 337_182_720

    projection = project_frozen_schedule(_panel())
    assert projection.projected_seconds == 8 * 17 * 100.0
    assert projection.projected_effective_rate == pytest.approx(
        PROJECTED_TOTAL_TRANSITIONS / (8 * 17 * 100.0)
    )
    assert projection.passed


def test_projection_uses_slowest_repeat_not_average() -> None:
    records = _panel(elapsed=100.0)
    target = next(
        index
        for index, value in enumerate(records)
        if value.profile_name == PROFILE_CACHE_P10 and value.repeat_index == 2
    )
    records[target] = _repeat(PROFILE_CACHE_P10, 2, elapsed=250.0)
    projection = project_frozen_schedule(records)
    assert projection.slowest_profile_seconds[PROFILE_CACHE_P10] == 250.0
    assert projection.projected_seconds == 8 * (9 * 250.0 + 8 * 100.0)


def test_exact_30_hour_boundary_passes_and_nextafter_fails() -> None:
    per_profile_seconds = MAXIMUM_PROJECTED_SECONDS / (8 * 17)
    exact = project_frozen_schedule(_panel(elapsed=per_profile_seconds))
    assert exact.projected_seconds == pytest.approx(MAXIMUM_PROJECTED_SECONDS)
    assert exact.passed

    slower = math.nextafter(per_profile_seconds, math.inf)
    failed = project_frozen_schedule(_panel(elapsed=slower))
    assert failed.projected_seconds > MAXIMUM_PROJECTED_SECONDS
    assert not failed.passed
    assert "projected_seconds" in failed.failed_checks
    assert "projected_effective_rate" in failed.failed_checks


def test_repeat_panel_rejects_hash_and_forbidden_event_changes() -> None:
    records = _panel()
    records[0] = _repeat(
        records[0].profile_name,
        records[0].repeat_index,
        output_suffix="changed",
    )
    with pytest.raises(BoundaryTangentScheduleError, match="hashes changed"):
        validate_repeat_records(records)

    records = _panel()
    profile = records[0].profile_name
    repeat = records[0].repeat_index
    records[0] = _repeat(
        profile,
        repeat,
        forbidden=_forbidden(nonfinite_count=1),
    )
    projection = project_frozen_schedule(records)
    assert not projection.passed
    assert projection.forbidden_total == 1
    assert "forbidden_events" in projection.failed_checks


def test_record_rejects_changed_order_or_lane_cap() -> None:
    base, midpoint, total = expected_profile_transition_counts(PROFILE_CACHE_P6)
    kwargs = dict(
        profile_name=PROFILE_CACHE_P6,
        repeat_index=0,
        execution_order_index=1,
        elapsed_seconds=100.0,
        base_transition_count=base,
        midpoint_transition_count=midpoint,
        certified_count=total,
        fallback_count=0,
        fallback_elapsed_seconds=0.0,
        maximum_mass_error=0.0,
        peak_memory_fraction=0.1,
        committed_bytes=0,
        maximum_launch_lanes=4097,
        output_sha256=_digest("output"),
        final_state_sha256=_digest("state"),
        forbidden_counts=_forbidden(),
    )
    with pytest.raises(BoundaryTangentScheduleError, match="lane cap"):
        PilotRepeatRecord(**kwargs)
    kwargs["maximum_launch_lanes"] = 3920
    kwargs["execution_order_index"] = 0
    with pytest.raises(BoundaryTangentScheduleError, match="execution order"):
        PilotRepeatRecord(**kwargs)


def test_repeat_record_round_trip_validates_derived_fields() -> None:
    value = _repeat(PROFILE_STREAM_P10, 1)
    record = value.to_record()
    restored = PilotRepeatRecord.from_record(record)
    assert restored == value
    record["transition_count"] += 1
    with pytest.raises(BoundaryTangentScheduleError, match="derived fields"):
        PilotRepeatRecord.from_record(record)


def test_profile_path_counts_match_frozen_names() -> None:
    assert PROFILE_PATH_COUNTS == {
        PROFILE_CACHE_P10: 10,
        PROFILE_CACHE_P6: 6,
        PROFILE_STREAM_P10: 10,
        PROFILE_STREAM_P4: 4,
    }
