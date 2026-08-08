from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mnist import d0_jacobi_rb_dynkin_phase_observer_path_ids as path_ids
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    SUPPORTED_SAMPLE_STEPS,
    canonical_refinement_transition_ids,
    finest_tick_for_step,
)


def test_frozen_phase_observer_allocations_and_hash_round_trip() -> None:
    plan = path_ids.PhaseObserverPathIDPlan()
    assert plan.legacy_replay_path_ids == tuple(range(20_000, 20_008))
    assert plan.tower_case_path_ids("a", 0) == tuple(
        range(0x60000, 0x60080)
    )
    assert plan.tower_case_path_ids("a", 7) == tuple(
        range(0x60380, 0x60400)
    )
    assert plan.tower_case_path_ids("b", 0) == tuple(
        range(0x70000, 0x70080)
    )
    assert plan.pilot_path_ids("a") == tuple(range(0x80000, 0x80008))
    assert plan.pilot_path_ids("b") == tuple(range(0x90000, 0x90008))
    assert plan.designated_production_path_ids == tuple(
        range(0xF0000, 0xF0040)
    )

    frozen = plan.to_frozen_record()
    assert (
        frozen["version"]
        == path_ids.PHASE_OBSERVER_PATH_ID_PLAN_VERSION
    )
    assert frozen["path_id_plan_sha256"] == plan.sha256
    assert path_ids.PhaseObserverPathIDPlan.from_frozen_record(frozen) == plan


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_phase_observer_path_id_rejects_nonintegers(value: object) -> None:
    with pytest.raises(TypeError):
        path_ids.validate_path_id(value)


def test_phase_observer_path_id_exact_20_bit_boundary() -> None:
    assert path_ids.validate_path_id(np.int64((1 << 20) - 1)) == (1 << 20) - 1
    with pytest.raises(ValueError, match="20-bit"):
        path_ids.validate_path_id(1 << 20)
    with pytest.raises(ValueError, match="20-bit"):
        path_ids.validate_path_id(-1)


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    (
        ("tower_path_count", True, TypeError),
        ("tower_path_count", 0, ValueError),
        ("tower_path_count", 129, ValueError),
        ("pilot_path_count", 1.5, TypeError),
        ("pilot_path_count", 0, ValueError),
        ("pilot_path_count", 9, ValueError),
    ),
)
def test_test_counts_can_reduce_but_not_exceed_slots(
    keyword: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        path_ids.PhaseObserverPathIDPlan(**{keyword: value})
    reduced = path_ids.PhaseObserverPathIDPlan(
        tower_path_count=3,
        pilot_path_count=2,
    )
    assert reduced.tower_case_path_ids("b", 2) == (
        0x70100,
        0x70101,
        0x70102,
    )
    assert reduced.pilot_path_ids("a") == (0x80000, 0x80001)


def test_fresh_roles_are_disjoint_from_each_other_prior_run_and_reserve() -> None:
    plan = path_ids.PhaseObserverPathIDPlan()
    fresh_roles = [
        set(plan.tower_panel_path_ids("a")),
        set(plan.tower_panel_path_ids("b")),
        set(plan.pilot_path_ids("a")),
        set(plan.pilot_path_ids("b")),
    ]
    prior = [
        set(range(start, stop))
        for start, stop in path_ids.PRIOR_STOCHASTIC_SLOTS
    ]
    production = set(
        range(
            path_ids.RESERVED_PRODUCTION_START,
            path_ids.RESERVED_PRODUCTION_STOP,
        )
    )
    for index, left in enumerate(fresh_roles):
        assert all(left.isdisjoint(right) for right in fresh_roles[index + 1 :])
        assert all(left.isdisjoint(right) for right in prior)
        assert left.isdisjoint(production)
        assert all(0 <= value < path_ids.PATH_ID_LIMIT for value in left)


def test_role_validation_and_plan_tampering_fail_closed() -> None:
    plan = path_ids.PhaseObserverPathIDPlan()
    expected = plan.tower_case_path_ids("a", 3)
    assert (
        plan.validate_role_path_ids("tower_a", expected, case_index=3)
        == expected
    )
    with pytest.raises(ValueError, match="frozen"):
        plan.validate_role_path_ids("tower_a", expected, case_index=2)
    with pytest.raises(ValueError, match="unique"):
        plan.validate_role_path_ids(
            "tower_a",
            expected[:-1] + (expected[0],),
            case_index=3,
        )

    tampered = copy.deepcopy(plan.to_frozen_record())
    tampered["roles"]["tower_b"]["cases"][0][0] += 1
    body = {
        key: value
        for key, value in tampered.items()
        if key != "path_id_plan_sha256"
    }
    tampered["path_id_plan_sha256"] = path_ids.path_id_plan_sha256(body)
    with pytest.raises(ValueError, match="record changed"):
        path_ids.PhaseObserverPathIDPlan.from_frozen_record(tampered)

    bad_hash = plan.to_frozen_record()
    bad_hash["path_id_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        path_ids.PhaseObserverPathIDPlan.from_frozen_record(bad_hash)


def test_128_path_tower_uses_sixteen_unchanged_canonical_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = path_ids.PhaseObserverPathIDPlan()
    paths = plan.tower_case_path_ids("a", 5)
    chunk_sizes: list[int] = []

    def recording_helper(
        chunk: tuple[int, ...],
        **kwargs: object,
    ) -> torch.Tensor:
        chunk_sizes.append(len(chunk))
        return canonical_refinement_transition_ids(chunk, **kwargs)

    monkeypatch.setattr(
        path_ids,
        "canonical_refinement_transition_ids",
        recording_helper,
    )
    actual = path_ids.canonical_tower_transition_ids(
        paths,
        sample_steps=2048,
        outer_step=2047,
        phase=6,
        device=torch.device("cpu"),
    )
    assert chunk_sizes == [8] * 16
    assert actual.shape == (128 * EDGES_PER_PHASE,)
    assert actual.dtype == torch.uint64
    assert actual.is_contiguous()
    assert torch.unique(actual.to(torch.int64)).numel() == actual.numel()
    assert int(actual.to(torch.int64).max()) < (
        path_ids.PACKED_TRANSITION_ID_LIMIT
    )


def test_tower_assembly_is_path_major_and_permutation_invariant() -> None:
    plan = path_ids.PhaseObserverPathIDPlan()
    paths = plan.tower_case_path_ids("b", 7)
    kwargs = {
        "sample_steps": 512,
        "outer_step": 19,
        "phase": 4,
        "device": torch.device("cpu"),
    }
    actual = path_ids.canonical_tower_transition_ids(paths, **kwargs).reshape(
        len(paths), EDGES_PER_PHASE
    )
    expected = torch.cat(
        [
            canonical_refinement_transition_ids(paths[start : start + 8], **kwargs)
            for start in range(0, len(paths), 8)
        ]
    ).reshape(len(paths), EDGES_PER_PHASE)
    assert torch.equal(actual, expected)
    reversed_ids = path_ids.canonical_tower_transition_ids(
        tuple(reversed(paths)),
        **kwargs,
    ).reshape(len(paths), EDGES_PER_PHASE)
    assert torch.equal(reversed_ids.flip(0), actual)


def test_cross_level_aliases_only_share_aligned_finest_ticks() -> None:
    plan = path_ids.PhaseObserverPathIDPlan(tower_path_count=1)
    path = plan.tower_case_path_ids("a", 0)
    by_id: dict[int, set[tuple[int, int]]] = {}
    for level in SUPPORTED_SAMPLE_STEPS:
        for outer_step in range(level):
            packed = path_ids.canonical_tower_transition_ids(
                path,
                sample_steps=level,
                outer_step=outer_step,
                phase=3,
                device=torch.device("cpu"),
            )
            by_id.setdefault(int(packed[0]), set()).add(
                (level, finest_tick_for_step(level, outer_step))
            )
    assert any(len(rows) > 1 for rows in by_id.values())
    assert all(len({tick for _, tick in rows}) == 1 for rows in by_id.values())

