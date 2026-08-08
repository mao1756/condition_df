from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mnist import d0_jacobi_rb_dynkin_path_ids as path_ids
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    SUPPORTED_SAMPLE_STEPS,
    canonical_refinement_transition_ids,
    finest_tick_for_step,
)


def test_frozen_plan_has_exact_versioned_allocations_and_stable_hash() -> None:
    plan = path_ids.DynkinPathIDPlan()

    assert plan.legacy_replay_path_ids == tuple(range(20_000, 20_008))
    assert plan.tower_case_path_ids("a", 0) == tuple(
        range(0x20000, 0x20080)
    )
    assert plan.tower_case_path_ids("a", 7) == tuple(
        range(0x20380, 0x20400)
    )
    assert plan.tower_case_path_ids("b", 0) == tuple(
        range(0x30000, 0x30080)
    )
    assert plan.pilot_path_ids("a") == tuple(range(0x40000, 0x40008))
    assert plan.pilot_path_ids("b") == tuple(range(0x50000, 0x50008))
    assert plan.designated_production_path_ids == tuple(
        range(0xF0000, 0xF0040)
    )

    frozen = plan.to_frozen_record()
    assert frozen["version"] == path_ids.DYNKIN_PATH_ID_PLAN_VERSION
    assert frozen["roles"]["reserved_production"]["slot"] == [
        0xF0000,
        0x100000,
    ]
    assert frozen["path_id_plan_sha256"] == plan.sha256
    assert path_ids.DynkinPathIDPlan.from_frozen_record(frozen) == plan
    assert path_ids.DynkinPathIDPlan().sha256 == plan.sha256


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_path_id_rejects_noninteger_types(value: object) -> None:
    with pytest.raises(TypeError):
        path_ids.validate_path_id(value)


def test_path_id_accepts_integer_protocol_and_checks_exact_20_bit_boundary() -> None:
    assert path_ids.validate_path_id(np.int64((1 << 20) - 1)) == (1 << 20) - 1
    with pytest.raises(ValueError, match="20-bit"):
        path_ids.validate_path_id(1 << 20)
    with pytest.raises(ValueError, match="20-bit"):
        path_ids.validate_path_id(-1)


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    [
        ("tower_path_count", True, TypeError),
        ("tower_path_count", 0, ValueError),
        ("tower_path_count", 129, ValueError),
        ("pilot_path_count", 1.5, TypeError),
        ("pilot_path_count", 0, ValueError),
        ("pilot_path_count", 9, ValueError),
    ],
)
def test_test_only_counts_may_reduce_but_not_exceed_slots(
    keyword: str, value: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        path_ids.DynkinPathIDPlan(**{keyword: value})

    reduced = path_ids.DynkinPathIDPlan(tower_path_count=3, pilot_path_count=2)
    assert reduced.tower_case_path_ids("a", 1) == (
        0x20080,
        0x20081,
        0x20082,
    )
    assert reduced.pilot_path_ids("b") == (0x50000, 0x50001)


def test_all_roles_and_tower_cases_are_exhaustively_unique_and_disjoint() -> None:
    plan = path_ids.DynkinPathIDPlan()
    tower_a_cases = [
        set(plan.tower_case_path_ids("a", index))
        for index in range(path_ids.TOWER_CASE_COUNT)
    ]
    tower_b_cases = [
        set(plan.tower_case_path_ids("b", index))
        for index in range(path_ids.TOWER_CASE_COUNT)
    ]
    for cases in (tower_a_cases, tower_b_cases):
        assert len(set.union(*cases)) == (
            path_ids.TOWER_CASE_COUNT * path_ids.MAX_TOWER_PATHS
        )
        for index, left in enumerate(cases):
            assert all(
                left.isdisjoint(right)
                for right in cases[index + 1 :]
            )

    roles = [
        set(plan.legacy_replay_path_ids),
        set.union(*tower_a_cases),
        set.union(*tower_b_cases),
        set(plan.pilot_path_ids("a")),
        set(plan.pilot_path_ids("b")),
        set(range(
            path_ids.RESERVED_PRODUCTION_START,
            path_ids.RESERVED_PRODUCTION_STOP,
        )),
    ]
    for index, left in enumerate(roles):
        assert all(left.isdisjoint(right) for right in roles[index + 1 :])
    assert all(
        0 <= value < path_ids.PATH_ID_LIMIT
        for role in roles
        for value in role
    )


def test_role_validation_rejects_wrong_slot_order_duplicates_and_case() -> None:
    plan = path_ids.DynkinPathIDPlan()
    expected = plan.tower_case_path_ids("a", 3)
    assert plan.validate_role_path_ids(
        "tower_a", expected, case_index=3
    ) == expected
    with pytest.raises(ValueError, match="frozen"):
        plan.validate_role_path_ids("tower_a", expected, case_index=2)
    with pytest.raises(ValueError, match="unique"):
        plan.validate_role_path_ids(
            "tower_a", expected[:-1] + (expected[0],), case_index=3
        )
    with pytest.raises(ValueError, match="frozen"):
        plan.validate_role_path_ids(
            "tower_a", tuple(reversed(expected)), case_index=3
        )
    with pytest.raises(ValueError, match="require case_index"):
        plan.validate_role_path_ids("tower_a", expected)


def test_plan_and_hash_tampering_are_rejected() -> None:
    plan = path_ids.DynkinPathIDPlan()
    tampered_panel = copy.deepcopy(plan.to_frozen_record())
    tampered_panel["roles"]["tower_a"]["cases"][0][0] += 1
    tampered_panel["path_id_plan_sha256"] = path_ids.path_id_plan_sha256(
        {
            key: value
            for key, value in tampered_panel.items()
            if key != "path_id_plan_sha256"
        }
    )
    with pytest.raises(ValueError, match="does not match"):
        path_ids.DynkinPathIDPlan.from_frozen_record(tampered_panel)

    tampered_hash = plan.to_frozen_record()
    tampered_hash["path_id_plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        path_ids.DynkinPathIDPlan.from_frozen_record(tampered_hash)


def test_128_path_tower_assembly_uses_sixteen_canonical_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = path_ids.DynkinPathIDPlan()
    paths = plan.tower_case_path_ids("a", 5)
    chunk_sizes: list[int] = []

    def recording_helper(
        chunk: tuple[int, ...],
        **kwargs: object,
    ) -> torch.Tensor:
        chunk_sizes.append(len(chunk))
        return canonical_refinement_transition_ids(chunk, **kwargs)

    monkeypatch.setattr(
        path_ids, "canonical_refinement_transition_ids", recording_helper
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


def test_chunking_is_path_major_and_invariant_under_path_permutation() -> None:
    plan = path_ids.DynkinPathIDPlan()
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

    permutation = tuple(reversed(paths))
    permuted = path_ids.canonical_tower_transition_ids(
        permutation, **kwargs
    ).reshape(len(paths), EDGES_PER_PHASE)
    assert torch.equal(permuted.flip(0), actual)


def test_cross_level_aliases_occur_exactly_at_aligned_finest_ticks() -> None:
    plan = path_ids.DynkinPathIDPlan(tower_path_count=1)
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
            identifier = int(packed[0])
            by_id.setdefault(identifier, set()).add(
                (level, finest_tick_for_step(level, outer_step))
            )

    assert any(len(rows) > 1 for rows in by_id.values())
    for rows in by_id.values():
        assert len({tick for _, tick in rows}) == 1

    aligned_128 = path_ids.canonical_tower_transition_ids(
        path,
        sample_steps=128,
        outer_step=0,
        phase=3,
        device=torch.device("cpu"),
    )
    aligned_256 = path_ids.canonical_tower_transition_ids(
        path,
        sample_steps=256,
        outer_step=1,
        phase=3,
        device=torch.device("cpu"),
    )
    unaligned_256 = path_ids.canonical_tower_transition_ids(
        path,
        sample_steps=256,
        outer_step=0,
        phase=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(aligned_128, aligned_256)
    assert not torch.equal(aligned_128, unaligned_256)
