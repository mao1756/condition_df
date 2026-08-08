"""Canonical path-ID namespace for the phase-local Dynkin observer repair.

The immutable refinement scheduler accepts at most eight paths per canonical
ID call.  Tower tensors are therefore assembled from ordered eight-path
chunks without relaxing that scheduler contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    MAX_REFINEMENT_PATHS_PER_GROUP,
    canonical_refinement_transition_ids,
)


PHASE_OBSERVER_PATH_ID_PLAN_SCHEMA = (
    "d0-jacobi-rb-dynkin-phase-observer-path-id-plan"
)
PHASE_OBSERVER_PATH_ID_PLAN_VERSION = (
    "d0-jacobi-rb-dynkin-phase-observer-path-id-v1"
)
PATH_ID_BITS = 20
PATH_ID_LIMIT = 1 << PATH_ID_BITS
PACKED_TRANSITION_ID_BITS = 43
PACKED_TRANSITION_ID_LIMIT = 1 << PACKED_TRANSITION_ID_BITS
LOW_COORDINATE_BITS = PACKED_TRANSITION_ID_BITS - PATH_ID_BITS

LEGACY_REPLAY_START = 20_000
LEGACY_REPLAY_COUNT = 8
TOWER_A_START = 0x60000
TOWER_B_START = 0x70000
TOWER_CASE_COUNT = 8
TOWER_CASE_STRIDE = 128
MAX_TOWER_PATHS = 128
PILOT_A_START = 0x80000
PILOT_B_START = 0x90000
MAX_PILOT_PATHS = 8
RESERVED_PRODUCTION_START = 0xF0000
RESERVED_PRODUCTION_STOP = 0x100000
DESIGNATED_PRODUCTION_PATHS = 64

# Inspected namespaces from the immutable failed ID-fix run.  The legacy
# replay and future production reservation are intentionally retained; all
# fresh stochastic tower/pilot roles must avoid these prior stochastic slots.
PRIOR_TOWER_A_SLOT = (0x20000, 0x20400)
PRIOR_TOWER_B_SLOT = (0x30000, 0x30400)
PRIOR_PILOT_A_SLOT = (0x40000, 0x40008)
PRIOR_PILOT_B_SLOT = (0x50000, 0x50008)
PRIOR_STOCHASTIC_SLOTS = (
    PRIOR_TOWER_A_SLOT,
    PRIOR_TOWER_B_SLOT,
    PRIOR_PILOT_A_SLOT,
    PRIOR_PILOT_B_SLOT,
)


def validate_path_id(value: Any) -> int:
    """Return one valid frozen 20-bit path ID."""

    if isinstance(value, bool):
        raise TypeError("path IDs must be integers, not bool")
    try:
        path_id = operator.index(value)
    except TypeError as exc:
        raise TypeError("path IDs must be integers") from exc
    if not 0 <= path_id < PATH_ID_LIMIT:
        raise ValueError("path IDs must fit the frozen 20-bit field")
    return path_id


def _bounded_count(value: Any, *, name: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not 1 <= count <= maximum:
        raise ValueError(f"{name} must lie in [1,{maximum}]")
    return count


def _panel(value: str) -> str:
    if value not in {"a", "b"}:
        raise ValueError("panel must be 'a' or 'b'")
    return value


def _case_index(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("case_index must be an integer, not bool")
    try:
        index = operator.index(value)
    except TypeError as exc:
        raise TypeError("case_index must be an integer") from exc
    if not 0 <= index < TOWER_CASE_COUNT:
        raise ValueError("case_index must lie in [0,8)")
    return index


def _unique_path_ids(
    path_ids: Sequence[int],
    *,
    maximum: int,
) -> tuple[int, ...]:
    result = tuple(validate_path_id(value) for value in path_ids)
    if not result:
        raise ValueError("at least one path ID is required")
    if len(result) > maximum:
        raise ValueError(f"at most {maximum} path IDs are allowed")
    if len(set(result)) != len(result):
        raise ValueError("path IDs must be unique")
    return result


def _intervals_disjoint(intervals: Sequence[tuple[int, int]]) -> bool:
    ordered = sorted(intervals)
    return all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


@dataclass(frozen=True)
class PhaseObserverPathIDPlan:
    """Complete namespace, including reduced nonauthorizing test counts."""

    tower_path_count: int = MAX_TOWER_PATHS
    pilot_path_count: int = MAX_PILOT_PATHS
    version: str = PHASE_OBSERVER_PATH_ID_PLAN_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tower_path_count",
            _bounded_count(
                self.tower_path_count,
                name="tower_path_count",
                maximum=MAX_TOWER_PATHS,
            ),
        )
        object.__setattr__(
            self,
            "pilot_path_count",
            _bounded_count(
                self.pilot_path_count,
                name="pilot_path_count",
                maximum=MAX_PILOT_PATHS,
            ),
        )
        if self.version != PHASE_OBSERVER_PATH_ID_PLAN_VERSION:
            raise ValueError("unsupported phase-observer path-ID plan version")
        self.validate()

    @property
    def legacy_replay_path_ids(self) -> tuple[int, ...]:
        return tuple(
            range(LEGACY_REPLAY_START, LEGACY_REPLAY_START + LEGACY_REPLAY_COUNT)
        )

    def tower_case_path_ids(
        self,
        panel: str,
        case_index: int,
    ) -> tuple[int, ...]:
        base = TOWER_A_START if _panel(panel) == "a" else TOWER_B_START
        start = base + _case_index(case_index) * TOWER_CASE_STRIDE
        return tuple(range(start, start + self.tower_path_count))

    def tower_panel_path_ids(self, panel: str) -> tuple[int, ...]:
        return tuple(
            path_id
            for case_index in range(TOWER_CASE_COUNT)
            for path_id in self.tower_case_path_ids(panel, case_index)
        )

    def pilot_path_ids(self, panel: str) -> tuple[int, ...]:
        base = PILOT_A_START if _panel(panel) == "a" else PILOT_B_START
        return tuple(range(base, base + self.pilot_path_count))

    @property
    def designated_production_path_ids(self) -> tuple[int, ...]:
        return tuple(
            range(
                RESERVED_PRODUCTION_START,
                RESERVED_PRODUCTION_START + DESIGNATED_PRODUCTION_PATHS,
            )
        )

    def validate_role_path_ids(
        self,
        role: str,
        path_ids: Sequence[int],
        *,
        case_index: int | None = None,
    ) -> tuple[int, ...]:
        maximums = {
            "legacy_replay": LEGACY_REPLAY_COUNT,
            "tower_a": MAX_TOWER_PATHS,
            "tower_b": MAX_TOWER_PATHS,
            "pilot_a": MAX_PILOT_PATHS,
            "pilot_b": MAX_PILOT_PATHS,
            "reserved_production": DESIGNATED_PRODUCTION_PATHS,
        }
        if role not in maximums:
            raise ValueError("unknown phase-observer path-ID role")
        paths = _unique_path_ids(path_ids, maximum=maximums[role])
        if role == "legacy_replay":
            expected = self.legacy_replay_path_ids
        elif role.startswith("tower_"):
            if case_index is None:
                raise ValueError("tower roles require case_index")
            expected = self.tower_case_path_ids(role[-1], case_index)
        elif role.startswith("pilot_"):
            if case_index is not None:
                raise ValueError("pilot roles do not have case_index")
            expected = self.pilot_path_ids(role[-1])
        else:
            expected = self.designated_production_path_ids
        if paths != expected:
            raise ValueError(f"path IDs do not match the frozen {role} allocation")
        return paths

    def validate(self) -> None:
        role_slots = (
            (LEGACY_REPLAY_START, LEGACY_REPLAY_START + LEGACY_REPLAY_COUNT),
            (
                TOWER_A_START,
                TOWER_A_START + TOWER_CASE_COUNT * TOWER_CASE_STRIDE,
            ),
            (
                TOWER_B_START,
                TOWER_B_START + TOWER_CASE_COUNT * TOWER_CASE_STRIDE,
            ),
            (PILOT_A_START, PILOT_A_START + MAX_PILOT_PATHS),
            (PILOT_B_START, PILOT_B_START + MAX_PILOT_PATHS),
            (RESERVED_PRODUCTION_START, RESERVED_PRODUCTION_STOP),
        )
        for start, stop in role_slots:
            validate_path_id(start)
            validate_path_id(stop - 1)
        if not _intervals_disjoint(role_slots):
            raise ValueError("phase-observer path-ID role slots overlap")

        fresh_slots = role_slots[1:5]
        if any(
            _overlap(fresh, prior)
            for fresh in fresh_slots
            for prior in PRIOR_STOCHASTIC_SLOTS
        ):
            raise ValueError("fresh phase-observer IDs overlap the failed run")

        roles = (
            self.legacy_replay_path_ids,
            self.tower_panel_path_ids("a"),
            self.tower_panel_path_ids("b"),
            self.pilot_path_ids("a"),
            self.pilot_path_ids("b"),
            self.designated_production_path_ids,
        )
        seen: set[int] = set()
        for paths in roles:
            if seen.intersection(paths):
                raise ValueError("phase-observer path-ID roles overlap")
            seen.update(paths)

        maximum_packed_id = (
            ((PATH_ID_LIMIT - 1) << LOW_COORDINATE_BITS)
            | ((1 << LOW_COORDINATE_BITS) - 1)
        )
        if maximum_packed_id >= PACKED_TRANSITION_ID_LIMIT:
            raise ValueError("path IDs exceed the frozen 43-bit packed-ID field")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PHASE_OBSERVER_PATH_ID_PLAN_SCHEMA,
            "version": self.version,
            "path_id_bits": PATH_ID_BITS,
            "packed_transition_id_bits": PACKED_TRANSITION_ID_BITS,
            "tower_case_count": TOWER_CASE_COUNT,
            "tower_case_stride": TOWER_CASE_STRIDE,
            "tower_path_count": self.tower_path_count,
            "pilot_path_count": self.pilot_path_count,
            "prior_stochastic_slots": [list(slot) for slot in PRIOR_STOCHASTIC_SLOTS],
            "roles": {
                "legacy_replay": {
                    "slot": [
                        LEGACY_REPLAY_START,
                        LEGACY_REPLAY_START + LEGACY_REPLAY_COUNT,
                    ],
                    "path_ids": list(self.legacy_replay_path_ids),
                },
                "tower_a": {
                    "slot": [
                        TOWER_A_START,
                        TOWER_A_START + TOWER_CASE_COUNT * TOWER_CASE_STRIDE,
                    ],
                    "cases": [
                        list(self.tower_case_path_ids("a", index))
                        for index in range(TOWER_CASE_COUNT)
                    ],
                },
                "tower_b": {
                    "slot": [
                        TOWER_B_START,
                        TOWER_B_START + TOWER_CASE_COUNT * TOWER_CASE_STRIDE,
                    ],
                    "cases": [
                        list(self.tower_case_path_ids("b", index))
                        for index in range(TOWER_CASE_COUNT)
                    ],
                },
                "pilot_a": {
                    "slot": [PILOT_A_START, PILOT_A_START + MAX_PILOT_PATHS],
                    "path_ids": list(self.pilot_path_ids("a")),
                },
                "pilot_b": {
                    "slot": [PILOT_B_START, PILOT_B_START + MAX_PILOT_PATHS],
                    "path_ids": list(self.pilot_path_ids("b")),
                },
                "reserved_production": {
                    "slot": [
                        RESERVED_PRODUCTION_START,
                        RESERVED_PRODUCTION_STOP,
                    ],
                    "designated_path_ids": list(
                        self.designated_production_path_ids
                    ),
                },
            },
        }

    @property
    def sha256(self) -> str:
        return path_id_plan_sha256(self.to_record())

    def to_frozen_record(self) -> dict[str, Any]:
        return {**self.to_record(), "path_id_plan_sha256": self.sha256}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PhaseObserverPathIDPlan":
        if not isinstance(record, Mapping):
            raise TypeError("phase-observer path-ID plan must be a mapping")
        try:
            plan = cls(
                tower_path_count=record["tower_path_count"],
                pilot_path_count=record["pilot_path_count"],
                version=record["version"],
            )
        except KeyError as exc:
            raise ValueError("phase-observer path-ID plan is incomplete") from exc
        if dict(record) != plan.to_record():
            raise ValueError("phase-observer path-ID plan record changed")
        return plan

    @classmethod
    def from_frozen_record(
        cls,
        record: Mapping[str, Any],
    ) -> "PhaseObserverPathIDPlan":
        body = dict(record)
        claimed = body.pop("path_id_plan_sha256", None)
        if not isinstance(claimed, str):
            raise ValueError("frozen phase-observer path-ID plan has no hash")
        plan = cls.from_record(body)
        if claimed != plan.sha256:
            raise ValueError("frozen phase-observer path-ID plan hash mismatch")
        return plan


def path_id_plan_sha256(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_canonical_transition_ids(
    transition_ids: Tensor,
    *,
    expected_path_count: int,
    expected_edges: int = EDGES_PER_PHASE,
) -> None:
    if not isinstance(transition_ids, Tensor):
        raise TypeError("transition_ids must be a torch tensor")
    if transition_ids.dtype != torch.uint64:
        raise TypeError("transition_ids must have dtype uint64")
    path_count = _bounded_count(
        expected_path_count,
        name="expected_path_count",
        maximum=MAX_TOWER_PATHS,
    )
    edge_count = _bounded_count(
        expected_edges,
        name="expected_edges",
        maximum=EDGES_PER_PHASE,
    )
    if transition_ids.ndim != 1 or transition_ids.numel() != path_count * edge_count:
        raise ValueError("canonical transition-ID tensor has the wrong shape")
    host = transition_ids.detach().to(device="cpu", dtype=torch.int64)
    if bool((host < 0).any()) or bool((host >= PACKED_TRANSITION_ID_LIMIT).any()):
        raise ValueError("canonical transition IDs exceed the 43-bit field")
    if torch.unique(host).numel() != host.numel():
        raise ValueError("canonical transition IDs must be unique")
    if not transition_ids.is_contiguous():
        raise ValueError("canonical transition IDs must be contiguous")


def canonical_tower_transition_ids(
    path_ids: Sequence[int],
    *,
    sample_steps: int,
    outer_step: int,
    phase: int,
    device: torch.device,
) -> Tensor:
    """Build a path-major tower tensor through canonical eight-path calls."""

    paths = _unique_path_ids(path_ids, maximum=MAX_TOWER_PATHS)
    chunks = [
        canonical_refinement_transition_ids(
            paths[start : start + MAX_REFINEMENT_PATHS_PER_GROUP],
            sample_steps=sample_steps,
            outer_step=outer_step,
            phase=phase,
            device=device,
        )
        for start in range(0, len(paths), MAX_REFINEMENT_PATHS_PER_GROUP)
    ]
    result = torch.cat(chunks).to(torch.uint64).contiguous()
    validate_canonical_transition_ids(
        result,
        expected_path_count=len(paths),
        expected_edges=EDGES_PER_PHASE,
    )
    return result


__all__ = [
    "DESIGNATED_PRODUCTION_PATHS",
    "LEGACY_REPLAY_COUNT",
    "LEGACY_REPLAY_START",
    "MAX_PILOT_PATHS",
    "MAX_TOWER_PATHS",
    "PACKED_TRANSITION_ID_LIMIT",
    "PATH_ID_LIMIT",
    "PHASE_OBSERVER_PATH_ID_PLAN_SCHEMA",
    "PHASE_OBSERVER_PATH_ID_PLAN_VERSION",
    "PILOT_A_START",
    "PILOT_B_START",
    "PRIOR_STOCHASTIC_SLOTS",
    "PhaseObserverPathIDPlan",
    "RESERVED_PRODUCTION_START",
    "RESERVED_PRODUCTION_STOP",
    "TOWER_A_START",
    "TOWER_B_START",
    "TOWER_CASE_COUNT",
    "TOWER_CASE_STRIDE",
    "canonical_tower_transition_ids",
    "path_id_plan_sha256",
    "validate_canonical_transition_ids",
    "validate_path_id",
]
