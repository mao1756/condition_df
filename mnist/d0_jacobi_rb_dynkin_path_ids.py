"""Versioned path-ID namespace for the Dynkin power confirmation.

The parent Strang-refinement scheduler deliberately accepts at most eight
paths.  This module leaves that scheduler unchanged and assembles larger
Dynkin tower tensors by concatenating canonical eight-path cohorts.
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


DYNKIN_PATH_ID_PLAN_SCHEMA = "d0-jacobi-rb-dynkin-path-id-plan"
DYNKIN_PATH_ID_PLAN_VERSION = "d0-jacobi-rb-dynkin-path-id-v1"
PATH_ID_BITS = 20
PATH_ID_LIMIT = 1 << PATH_ID_BITS
PACKED_TRANSITION_ID_BITS = 43
PACKED_TRANSITION_ID_LIMIT = 1 << PACKED_TRANSITION_ID_BITS
LOW_COORDINATE_BITS = PACKED_TRANSITION_ID_BITS - PATH_ID_BITS

LEGACY_REPLAY_START = 20_000
LEGACY_REPLAY_COUNT = 8
TOWER_A_START = 0x20000
TOWER_B_START = 0x30000
TOWER_CASE_COUNT = 8
TOWER_CASE_STRIDE = 128
MAX_TOWER_PATHS = 128
PILOT_A_START = 0x40000
PILOT_B_START = 0x50000
MAX_PILOT_PATHS = 8
RESERVED_PRODUCTION_START = 0xF0000
RESERVED_PRODUCTION_STOP = 0x100000
DESIGNATED_PRODUCTION_PATHS = 64


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


def _unique_path_ids(path_ids: Sequence[int], *, maximum: int) -> tuple[int, ...]:
    paths = tuple(validate_path_id(value) for value in path_ids)
    if not paths:
        raise ValueError("at least one path ID is required")
    if len(paths) > maximum:
        raise ValueError(f"at most {maximum} path IDs are allowed")
    if len(set(paths)) != len(paths):
        raise ValueError("path IDs must be unique")
    return paths


def _intervals_disjoint(intervals: Sequence[tuple[int, int]]) -> bool:
    ordered = sorted(intervals)
    return all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))


@dataclass(frozen=True)
class DynkinPathIDPlan:
    """Complete deterministic Dynkin namespace, including reduced test counts."""

    tower_path_count: int = MAX_TOWER_PATHS
    pilot_path_count: int = MAX_PILOT_PATHS
    version: str = DYNKIN_PATH_ID_PLAN_VERSION

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
        if self.version != DYNKIN_PATH_ID_PLAN_VERSION:
            raise ValueError("unsupported Dynkin path-ID plan version")
        self.validate()

    @property
    def legacy_replay_path_ids(self) -> tuple[int, ...]:
        return tuple(
            range(LEGACY_REPLAY_START, LEGACY_REPLAY_START + LEGACY_REPLAY_COUNT)
        )

    def tower_case_path_ids(
        self, panel: str, case_index: int
    ) -> tuple[int, ...]:
        panel_name = _panel(panel)
        index = _case_index(case_index)
        base = TOWER_A_START if panel_name == "a" else TOWER_B_START
        start = base + index * TOWER_CASE_STRIDE
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
        """Validate one derived panel against its exact frozen allocation."""

        role_maximums = {
            "legacy_replay": LEGACY_REPLAY_COUNT,
            "tower_a": MAX_TOWER_PATHS,
            "tower_b": MAX_TOWER_PATHS,
            "pilot_a": MAX_PILOT_PATHS,
            "pilot_b": MAX_PILOT_PATHS,
            "reserved_production": DESIGNATED_PRODUCTION_PATHS,
        }
        if role not in role_maximums:
            raise ValueError("unknown Dynkin path-ID role")
        maximum = role_maximums[role]
        paths = _unique_path_ids(path_ids, maximum=maximum)
        if role == "legacy_replay":
            expected = self.legacy_replay_path_ids
        elif role in {"tower_a", "tower_b"}:
            if case_index is None:
                raise ValueError("tower roles require case_index")
            expected = self.tower_case_path_ids(role[-1], case_index)
        elif role in {"pilot_a", "pilot_b"}:
            if case_index is not None:
                raise ValueError("pilot roles do not have case_index")
            expected = self.pilot_path_ids(role[-1])
        elif role == "reserved_production":
            expected = self.designated_production_path_ids
        if paths != expected:
            raise ValueError(f"path IDs do not match the frozen {role} allocation")
        return paths

    def validate(self) -> None:
        """Validate bounds, slot ownership, uniqueness, and role disjointness."""

        role_intervals = (
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
        for start, stop in role_intervals:
            validate_path_id(start)
            validate_path_id(stop - 1)
            if start >= stop:
                raise ValueError("path-ID slots must be nonempty")
        if not _intervals_disjoint(role_intervals):
            raise ValueError("Dynkin path-ID role slots overlap")

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
            checked = _unique_path_ids(paths, maximum=max(len(paths), 1))
            if seen.intersection(checked):
                raise ValueError("Dynkin path-ID roles overlap")
            seen.update(checked)
        maximum_packed_id = (
            ((PATH_ID_LIMIT - 1) << LOW_COORDINATE_BITS)
            | ((1 << LOW_COORDINATE_BITS) - 1)
        )
        if maximum_packed_id >= PACKED_TRANSITION_ID_LIMIT:
            raise ValueError("path IDs exceed the frozen 43-bit packed-ID field")

    def to_record(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible plan body."""

        return {
            "schema": DYNKIN_PATH_ID_PLAN_SCHEMA,
            "version": self.version,
            "path_id_bits": PATH_ID_BITS,
            "packed_transition_id_bits": PACKED_TRANSITION_ID_BITS,
            "tower_case_count": TOWER_CASE_COUNT,
            "tower_case_stride": TOWER_CASE_STRIDE,
            "tower_path_count": self.tower_path_count,
            "pilot_path_count": self.pilot_path_count,
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
        record = self.to_record()
        return {**record, "path_id_plan_sha256": self.sha256}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DynkinPathIDPlan":
        if not isinstance(record, Mapping):
            raise TypeError("Dynkin path-ID plan record must be a mapping")
        try:
            plan = cls(
                tower_path_count=record["tower_path_count"],
                pilot_path_count=record["pilot_path_count"],
                version=record["version"],
            )
        except KeyError as exc:
            raise ValueError("Dynkin path-ID plan record is incomplete") from exc
        if dict(record) != plan.to_record():
            raise ValueError("Dynkin path-ID plan record does not match its version")
        return plan

    @classmethod
    def from_frozen_record(
        cls, record: Mapping[str, Any]
    ) -> "DynkinPathIDPlan":
        if not isinstance(record, Mapping):
            raise TypeError("frozen Dynkin path-ID plan must be a mapping")
        body = dict(record)
        claimed_hash = body.pop("path_id_plan_sha256", None)
        if not isinstance(claimed_hash, str):
            raise ValueError("frozen Dynkin path-ID plan has no valid hash")
        plan = cls.from_record(body)
        if claimed_hash != plan.sha256:
            raise ValueError("frozen Dynkin path-ID plan hash mismatch")
        return plan


def path_id_plan_sha256(record: Mapping[str, Any]) -> str:
    """Hash a plan body using its canonical artifact JSON encoding."""

    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_tower_transition_ids(
    path_ids: Sequence[int],
    *,
    sample_steps: int,
    outer_step: int,
    phase: int,
    device: torch.device,
) -> Tensor:
    """Assemble up to 128 path-major IDs through canonical eight-path calls."""

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
        result, expected_path_count=len(paths), expected_edges=EDGES_PER_PHASE
    )
    return result


def validate_canonical_transition_ids(
    transition_ids: Tensor,
    *,
    expected_path_count: int,
    expected_edges: int = EDGES_PER_PHASE,
) -> None:
    """Check one canonical tensor for shape, uniqueness, and 43-bit safety."""

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
        expected_edges, name="expected_edges", maximum=EDGES_PER_PHASE
    )
    expected_size = path_count * edge_count
    if transition_ids.ndim != 1 or transition_ids.numel() != expected_size:
        raise ValueError("canonical transition-ID tensor has the wrong shape")
    host = transition_ids.detach().to(device="cpu", dtype=torch.int64)
    if bool((host < 0).any()) or bool((host >= PACKED_TRANSITION_ID_LIMIT).any()):
        raise ValueError("canonical transition IDs must fit the frozen 43-bit field")
    if torch.unique(host).numel() != host.numel():
        raise ValueError("canonical transition IDs must be unique")
    if not transition_ids.is_contiguous():
        raise ValueError("canonical transition IDs must be contiguous")


__all__ = [
    "DESIGNATED_PRODUCTION_PATHS",
    "DYNKIN_PATH_ID_PLAN_SCHEMA",
    "DYNKIN_PATH_ID_PLAN_VERSION",
    "DynkinPathIDPlan",
    "LEGACY_REPLAY_COUNT",
    "LEGACY_REPLAY_START",
    "MAX_PILOT_PATHS",
    "MAX_TOWER_PATHS",
    "PACKED_TRANSITION_ID_LIMIT",
    "PATH_ID_LIMIT",
    "PILOT_A_START",
    "PILOT_B_START",
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
