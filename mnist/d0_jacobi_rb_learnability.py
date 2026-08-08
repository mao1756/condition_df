"""Pure helpers for exact-K=512 Jacobi/RB one-image learnability.

This module deliberately contains no CLI, provenance gate, exact-transition
scheduler, or reverse sampler.  It defines the small supervised experiment
that consumes caches produced by the certified Jacobi scheduler:

* a frozen one-image/path/training configuration;
* physically separated model-input and label/audit cache schemas;
* a strict later-state-only model-input firewall;
* the fixed phase-conditioned local-affine-plus-CNN predictor;
* training-only metadata and synthetic-teacher controls; and
* deterministic MSE, checkpoint-selection, and whole-path sign helpers.

The supervised target remains the unmodified binary64 Rao--Blackwell label.
Only one positive global loss scale is supported; all scientific metrics are
reported in the original target units.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


LEARNABILITY_VERSION = "d0-jacobi-rb-one-image-learnability-v1"
SCIENTIFIC_CONFIG_SCHEMA = LEARNABILITY_VERSION + "-scientific-config"
PATH_ID_PLAN_SCHEMA = LEARNABILITY_VERSION + "-path-id-plan"
INPUT_CACHE_SCHEMA = LEARNABILITY_VERSION + "-model-input-cache"
LABEL_AUDIT_CACHE_SCHEMA = LEARNABILITY_VERSION + "-label-audit-cache"
METADATA_BASELINE_SCHEMA = LEARNABILITY_VERSION + "-metadata-baseline"
MODEL_VERSION = LEARNABILITY_VERSION + "-local-affine-cnn"
TRAINING_VERSION = LEARNABILITY_VERSION + "-deterministic-training"

GRID_SIZE = 28
STATE_SIZE = GRID_SIZE * GRID_SIZE
EDGES_PER_PHASE = STATE_SIZE // 2
OUTER_STEPS = 512
STEPS_PER_SHARD = 8
PHASE_MATCHINGS = (0, 1, 2, 3, 2, 1, 0)
PHASE_DURATIONS = (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
PHASE_COUNT = len(PHASE_MATCHINGS)
SELECTED_OUTER_STEPS = tuple(range(15, OUTER_STEPS, 16))
PATH_ID_LIMIT = 1 << 20

# The plan proposed 0x60000/0x61000/0x62000 as preferred fresh slots.  Those
# slots are already claimed by the versioned phase-observer path plan in this
# repository.  The final free block below Haar's A--D panels and the frozen
# 0xF0000 production reservation is therefore used instead.
TRAIN_PATH_IDS = tuple(range(0xE0000, 0xE0008))
VALIDATION_PATH_IDS = tuple(range(0xE1000, 0xE1008))
CONFIRMATION_PATH_IDS = tuple(range(0xE2000, 0xE2008))

MODEL_INPUT_FIELDS = (
    "later_full_state",
    "reverse_time",
    "phase",
    "color",
    "duration",
    "label",
)
INPUT_CACHE_FIELDS = ("sample_key",) + MODEL_INPUT_FIELDS
LABEL_AUDIT_CACHE_FIELDS = (
    "sample_key",
    "path_id",
    "outer_step",
    "phase",
    "denoising_target",
    "certificate_codes",
)
FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    {
        "earlier_state",
        "path_id",
        "outer_step",
        "sample_key",
        "uniform_bits",
        "normal_variables",
        "later_head_fraction",
        "later_head_fractions",
        "certificate_codes",
        "denoising_target",
        "oracle_target",
        "target",
    }
)

_INPUT_DTYPES: Mapping[str, np.dtype[Any]] = {
    "sample_key": np.dtype(np.int64),
    "later_full_state": np.dtype(np.float64),
    "reverse_time": np.dtype(np.float64),
    "phase": np.dtype(np.int8),
    "color": np.dtype(np.int8),
    "duration": np.dtype(np.float64),
    "label": np.dtype(np.int64),
}
_AUDIT_DTYPES: Mapping[str, np.dtype[Any]] = {
    "sample_key": np.dtype(np.int64),
    "path_id": np.dtype(np.int64),
    "outer_step": np.dtype(np.int16),
    "phase": np.dtype(np.int8),
    "denoising_target": np.dtype(np.float64),
    "certificate_codes": np.dtype(np.uint8),
}


def _build_matching_index_arrays() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Construct the four frozen oriented torus perfect matchings."""

    result: list[tuple[np.ndarray, np.ndarray]] = []
    for matching_index in range(4):
        tails: list[int] = []
        heads: list[int] = []
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                if matching_index < 2 and column % 2 == matching_index:
                    tails.append(row * GRID_SIZE + column)
                    heads.append(row * GRID_SIZE + ((column + 1) % GRID_SIZE))
                elif matching_index >= 2 and row % 2 == matching_index - 2:
                    tails.append(row * GRID_SIZE + column)
                    heads.append(((row + 1) % GRID_SIZE) * GRID_SIZE + column)
        result.append(
            (
                np.asarray(tails, dtype=np.int64),
                np.asarray(heads, dtype=np.int64),
            )
        )
    return tuple(result)


_MATCHING_INDEX_ARRAYS = _build_matching_index_arrays()


class LearnabilityContractError(ValueError):
    """A cache, namespace, or model-input contract is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the workflow's stable semantic JSON encoding."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _readonly_c_array(
    value: Any, dtype: np.dtype[Any], *, strict_dtype: bool = False
) -> np.ndarray:
    source = np.asarray(value)
    if strict_dtype and source.dtype != dtype:
        raise LearnabilityContractError(
            f"array dtype {source.dtype} does not match required {dtype}"
        )
    array = np.array(source, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _index(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def validate_path_id(value: Any) -> int:
    path_id = _index(value, "path_id")
    if not 0 <= path_id < PATH_ID_LIMIT:
        raise LearnabilityContractError("path IDs must fit the canonical 20-bit field")
    return path_id


def selected_reverse_time(outer_step: Any, phase: Any) -> float:
    """Normalized remaining exact split-chain phase coordinate."""

    step = _index(outer_step, "outer_step")
    phase_index = _index(phase, "phase")
    if not 0 <= step < OUTER_STEPS or not 0 <= phase_index < PHASE_COUNT:
        raise LearnabilityContractError("outer_step or phase is outside the frozen chain")
    return 1.0 - (PHASE_COUNT * step + phase_index + 1) / (
        PHASE_COUNT * OUTER_STEPS
    )


def sample_key(path_id: Any, outer_step: Any, phase: Any) -> int:
    """Pack one cache-row key without an edge coordinate."""

    path = validate_path_id(path_id)
    step = _index(outer_step, "outer_step")
    phase_index = _index(phase, "phase")
    if not 0 <= step < (1 << 10) or not 0 <= phase_index < (1 << 3):
        raise LearnabilityContractError("sample-key coordinate exceeds its frozen field")
    return (path << 13) | (step << 3) | phase_index


@dataclass(frozen=True)
class LearnabilityScientificConfig:
    """Frozen scientific and resource configuration for the one-image gate."""

    schema: str = SCIENTIFIC_CONFIG_SCHEMA
    schema_version: int = 1
    label: int = 3
    class_index: int = 0
    lambda_mix: float = 0.35
    image_sha256: str = (
        "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
    )
    mixed_target_sha256: str = (
        "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
    )
    source_image_npz_sha256: str = (
        "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
    )
    grid_size: int = GRID_SIZE
    outer_steps: int = OUTER_STEPS
    steps_per_shard: int = STEPS_PER_SHARD
    selected_outer_steps: tuple[int, ...] = SELECTED_OUTER_STEPS
    phase_matchings: tuple[int, ...] = PHASE_MATCHINGS
    phase_durations: tuple[float, ...] = PHASE_DURATIONS
    edges_per_phase: int = EDGES_PER_PHASE
    root_seed: int = 261_191
    minimum_effective_transitions_per_second: float = 1_300.0
    maximum_projected_total_hours: float = 10.0
    maximum_peak_memory_fraction: float = 0.80
    maximum_persisted_cache_bytes: int = 134_217_728

    def __post_init__(self) -> None:
        if self.schema != SCIENTIFIC_CONFIG_SCHEMA or self.schema_version != 1:
            raise LearnabilityContractError("unsupported scientific configuration schema")
        if (
            self.grid_size != GRID_SIZE
            or self.outer_steps != OUTER_STEPS
            or self.steps_per_shard != STEPS_PER_SHARD
            or tuple(self.selected_outer_steps) != SELECTED_OUTER_STEPS
            or tuple(self.phase_matchings) != PHASE_MATCHINGS
            or tuple(self.phase_durations) != PHASE_DURATIONS
            or self.edges_per_phase != EDGES_PER_PHASE
        ):
            raise LearnabilityContractError("scientific split-chain configuration changed")
        if not 0.0 < self.lambda_mix < 1.0:
            raise LearnabilityContractError("lambda_mix must lie strictly inside (0,1)")
        if (
            self.minimum_effective_transitions_per_second <= 0
            or self.maximum_projected_total_hours <= 0
            or not 0 < self.maximum_peak_memory_fraction <= 1
            or self.maximum_persisted_cache_bytes <= 0
        ):
            raise LearnabilityContractError("resource limits must be positive")

    @property
    def selected_outer_step_count(self) -> int:
        return len(self.selected_outer_steps)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["selected_outer_step_count"] = self.selected_outer_step_count
        record["reverse_time_semantics"] = (
            "normalized remaining exact K=512 split-chain phase index"
        )
        return record

    @property
    def sha256(self) -> str:
        return semantic_sha256(self.to_record())

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any]
    ) -> LearnabilityScientificConfig:
        body = dict(record)
        body.pop("selected_outer_step_count", None)
        body.pop("reverse_time_semantics", None)
        for name in ("selected_outer_steps", "phase_matchings", "phase_durations"):
            if name in body:
                body[name] = tuple(body[name])
        result = cls(**body)
        if semantic_sha256(result.to_record()) != semantic_sha256(dict(record)):
            raise LearnabilityContractError("scientific configuration record changed")
        return result


def expected_transition_count(path_count: Any) -> int:
    count = _index(path_count, "path_count")
    if count < 0:
        raise LearnabilityContractError("path_count must be nonnegative")
    return count * OUTER_STEPS * PHASE_COUNT * EDGES_PER_PHASE


def expected_selected_sample_count(path_count: Any) -> int:
    count = _index(path_count, "path_count")
    if count < 0:
        raise LearnabilityContractError("path_count must be nonnegative")
    return count * len(SELECTED_OUTER_STEPS) * PHASE_COUNT


@dataclass(frozen=True)
class PathIDClaim:
    """One existing exact ID or half-open slot found by the collision scan."""

    source: str
    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.stop <= PATH_ID_LIMIT:
            raise LearnabilityContractError("path-ID claim is outside the 20-bit namespace")

    def overlaps(self, path_ids: Iterable[int]) -> tuple[int, ...]:
        return tuple(sorted(value for value in path_ids if self.start <= value < self.stop))


@dataclass(frozen=True)
class PathIDCollision:
    source: str
    name: str
    path_ids: tuple[int, ...]


def _claim_from_named_value(
    claims: list[PathIDClaim], *, source: str, name: str, value: Any
) -> None:
    upper = name.upper()
    relevant = any(
        token in upper
        for token in (
            "PATH_ID",
            "PATH_IDS",
            "PATHS",
            "SLOT",
            "RESERVED",
            "BASES",
            "TOWER",
            "PILOT",
            "MARGINAL",
            "PROFILE",
        )
    )
    if not relevant:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if 0 <= value < PATH_ID_LIMIT:
            width = 0x1000 if ("BASE" in upper or "SLOT" in upper) else 1
            claims.append(
                PathIDClaim(source, name, value, min(PATH_ID_LIMIT, value + width))
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _claim_from_named_value(
                claims, source=source, name=f"{name}.{key}", value=child
            )
        return
    if isinstance(value, (list, tuple)):
        integers = [
            item for item in value if isinstance(item, int) and not isinstance(item, bool)
        ]
        if (
            len(value) == 2
            and len(integers) == 2
            and any(token in upper for token in ("SLOT", "RANGE", "RESERVED"))
            and 0 <= integers[0] < integers[1] <= PATH_ID_LIMIT
        ):
            claims.append(PathIDClaim(source, name, integers[0], integers[1]))
            return
        for index, child in enumerate(value):
            _claim_from_named_value(
                claims, source=source, name=f"{name}[{index}]", value=child
            )


def _json_path_claims(path: Path) -> list[PathIDClaim]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    claims: list[PathIDClaim] = []

    def visit(value: Any, name: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_name = f"{name}.{key}" if name else str(key)
                _claim_from_named_value(
                    claims, source=str(path), name=child_name, value=child
                )
                if isinstance(child, Mapping):
                    visit(child, child_name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, Mapping):
                    visit(child, f"{name}[{index}]")

    visit(payload, "")
    return claims


def _python_path_claims(path: Path) -> list[PathIDClaim]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    claims: list[PathIDClaim] = []
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if len(node.targets) == 1 else None
        )
        value_node = node.value
        if not isinstance(target, ast.Name) or value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            continue
        constants[target.id] = value
        _claim_from_named_value(
            claims, source=str(path), name=target.id, value=value
        )

    # Common START/STOP pairs are most accurately represented as intervals.
    for name, start in constants.items():
        if not name.endswith("_START") or not isinstance(start, int):
            continue
        prefix = name[:-6]
        stop_name = prefix + "_STOP"
        stop = constants.get(stop_name)
        if stop is None:
            count = constants.get(prefix + "_COUNT")
            if not isinstance(count, int):
                if "TOWER" in prefix:
                    case_count = constants.get("TOWER_CASE_COUNT")
                    stride = constants.get("TOWER_CASE_STRIDE")
                    count = (
                        case_count * stride
                        if isinstance(case_count, int) and isinstance(stride, int)
                        else None
                    )
                elif "PILOT" in prefix:
                    count = constants.get("MAX_PILOT_PATHS")
                elif "LEGACY" in prefix:
                    count = constants.get("LEGACY_REPLAY_COUNT")
            if isinstance(count, int):
                stop = start + count
        if isinstance(stop, int) and 0 <= start < stop <= PATH_ID_LIMIT:
            claims.append(PathIDClaim(str(path), prefix, start, stop))
    return claims


def discover_repository_path_id_claims(repository_root: str | Path) -> tuple[PathIDClaim, ...]:
    """Inspect versioned source/JSON path plans without importing them.

    The scan is intentionally semantic and read-only.  Python files are parsed
    with :mod:`ast`; JSON plan files are decoded as data.  Runtime cache shards
    and artifact registries are excluded.
    """

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates: list[Path] = []
    for base in (root / "mnist", root / "runs"):
        if not base.is_dir():
            continue
        candidates.extend(base.rglob("*path*id*.py"))
        candidates.extend(base.rglob("*path*id*.json"))
        candidates.extend(base.rglob("*path_id_plan.json"))
        candidates.extend(base.rglob("*haar_path_id_plan.json"))
    claims: list[PathIDClaim] = []
    for path in sorted(set(candidates)):
        if any(part in {"shards", "checkpoints", "__pycache__"} for part in path.parts):
            continue
        claims.extend(
            _python_path_claims(path) if path.suffix == ".py" else _json_path_claims(path)
        )
    unique = {
        (claim.source, claim.name, claim.start, claim.stop): claim for claim in claims
    }
    return tuple(unique[key] for key in sorted(unique))


def scan_path_id_collisions(
    path_ids: Sequence[int],
    claims_or_repository: Sequence[PathIDClaim] | str | Path,
) -> tuple[PathIDCollision, ...]:
    """Return every existing namespace claim that intersects ``path_ids``."""

    ids = tuple(validate_path_id(value) for value in path_ids)
    if len(ids) != len(set(ids)):
        raise LearnabilityContractError("candidate path IDs must be unique")
    claims = (
        discover_repository_path_id_claims(claims_or_repository)
        if isinstance(claims_or_repository, (str, Path))
        else tuple(claims_or_repository)
    )
    collisions = []
    for claim in claims:
        overlap = claim.overlaps(ids)
        if overlap:
            collisions.append(PathIDCollision(claim.source, claim.name, overlap))
    return tuple(collisions)


@dataclass(frozen=True)
class LearnabilityPathPlan:
    """Three sealed whole-path roles for train, validation, and confirmation."""

    train: tuple[int, ...] = TRAIN_PATH_IDS
    validation: tuple[int, ...] = VALIDATION_PATH_IDS
    confirmation: tuple[int, ...] = CONFIRMATION_PATH_IDS
    version: str = LEARNABILITY_VERSION + "-path-id-v1"

    def __post_init__(self) -> None:
        roles = {
            "train": tuple(validate_path_id(value) for value in self.train),
            "validation": tuple(validate_path_id(value) for value in self.validation),
            "confirmation": tuple(validate_path_id(value) for value in self.confirmation),
        }
        for name, values in roles.items():
            if not values:
                raise LearnabilityContractError(f"{name} path role is empty")
            if len(values) != len(set(values)):
                raise LearnabilityContractError(f"{name} path IDs are not unique")
            object.__setattr__(self, name, values)
        flattened = [value for values in roles.values() for value in values]
        if len(flattened) != len(set(flattened)):
            raise LearnabilityContractError("train/validation/confirmation paths overlap")

    @property
    def all_path_ids(self) -> tuple[int, ...]:
        return self.train + self.validation + self.confirmation

    def to_record(self) -> dict[str, Any]:
        body = {
            "schema": PATH_ID_PLAN_SCHEMA,
            "schema_version": 1,
            "version": self.version,
            "roles": {
                "train": list(self.train),
                "validation": list(self.validation),
                "confirmation": list(self.confirmation),
            },
            "checks": {
                "integer_20_bit_pass": 1,
                "role_disjoint_pass": 1,
                "confirmation_sealed": 1,
            },
        }
        return {**body, "path_id_plan_sha256": semantic_sha256(body)}

    @property
    def sha256(self) -> str:
        return str(self.to_record()["path_id_plan_sha256"])

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> LearnabilityPathPlan:
        body = dict(record)
        claimed_hash = body.pop("path_id_plan_sha256", None)
        if claimed_hash != semantic_sha256(body):
            raise LearnabilityContractError("path-ID plan semantic hash mismatch")
        if (
            body.get("schema") != PATH_ID_PLAN_SCHEMA
            or body.get("schema_version") != 1
            or not isinstance(body.get("roles"), Mapping)
        ):
            raise LearnabilityContractError("path-ID plan record has wrong schema")
        roles = body["roles"]
        result = cls(
            train=tuple(roles.get("train", ())),
            validation=tuple(roles.get("validation", ())),
            confirmation=tuple(roles.get("confirmation", ())),
            version=str(body.get("version")),
        )
        if result.to_record() != dict(record):
            raise LearnabilityContractError("path-ID plan record changed")
        return result

    def assert_collision_free(
        self, claims_or_repository: Sequence[PathIDClaim] | str | Path
    ) -> None:
        collisions = scan_path_id_collisions(self.all_path_ids, claims_or_repository)
        if collisions:
            detail = ", ".join(
                f"{collision.source}:{collision.name}={list(collision.path_ids)}"
                for collision in collisions
            )
            raise LearnabilityContractError(f"path-ID namespace collision: {detail}")


def frozen_scientific_config() -> LearnabilityScientificConfig:
    return LearnabilityScientificConfig()


def frozen_path_plan() -> LearnabilityPathPlan:
    return LearnabilityPathPlan()


@dataclass(frozen=True)
class LearnabilityInputCache:
    sample_key: np.ndarray = field(repr=False, compare=False)
    later_full_state: np.ndarray = field(repr=False, compare=False)
    reverse_time: np.ndarray = field(repr=False, compare=False)
    phase: np.ndarray = field(repr=False, compare=False)
    color: np.ndarray = field(repr=False, compare=False)
    duration: np.ndarray = field(repr=False, compare=False)
    label: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, dtype in _INPUT_DTYPES.items():
            object.__setattr__(
                self,
                name,
                _readonly_c_array(getattr(self, name), dtype, strict_dtype=True),
            )
        _validate_input_cache(self)

    @property
    def sample_count(self) -> int:
        return int(self.sample_key.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in INPUT_CACHE_FIELDS}


@dataclass(frozen=True)
class LearnabilityLabelAuditCache:
    sample_key: np.ndarray = field(repr=False, compare=False)
    path_id: np.ndarray = field(repr=False, compare=False)
    outer_step: np.ndarray = field(repr=False, compare=False)
    phase: np.ndarray = field(repr=False, compare=False)
    denoising_target: np.ndarray = field(repr=False, compare=False)
    certificate_codes: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, dtype in _AUDIT_DTYPES.items():
            object.__setattr__(
                self,
                name,
                _readonly_c_array(getattr(self, name), dtype, strict_dtype=True),
            )
        _validate_label_audit_cache(self)

    @property
    def sample_count(self) -> int:
        return int(self.sample_key.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in LABEL_AUDIT_CACHE_FIELDS}


@dataclass(frozen=True)
class LearnabilityCacheBundle:
    inputs: LearnabilityInputCache
    labels_audit: LearnabilityLabelAuditCache

    def __post_init__(self) -> None:
        validate_cache_bundle(self)

    @property
    def sample_count(self) -> int:
        return self.inputs.sample_count


def _validate_input_cache(cache: LearnabilityInputCache) -> None:
    n = int(cache.sample_key.shape[0])
    expected = {
        "sample_key": (n,),
        "later_full_state": (n, STATE_SIZE),
        "reverse_time": (n,),
        "phase": (n,),
        "color": (n,),
        "duration": (n,),
        "label": (n,),
    }
    for name, shape in expected.items():
        value = getattr(cache, name)
        if value.dtype != _INPUT_DTYPES[name] or value.shape != shape:
            raise LearnabilityContractError(f"invalid {name} dtype or shape")
        if not value.flags.c_contiguous or value.flags.writeable:
            raise LearnabilityContractError(f"{name} must be read-only C-contiguous")
    if len(np.unique(cache.sample_key)) != n:
        raise LearnabilityContractError("model-input sample keys must be unique")
    if not (
        np.isfinite(cache.later_full_state).all()
        and np.isfinite(cache.reverse_time).all()
        and np.isfinite(cache.duration).all()
    ):
        raise LearnabilityContractError("model-input cache contains nonfinite values")
    if (cache.later_full_state < 0).any():
        raise LearnabilityContractError("later states must be nonnegative")
    if ((cache.phase < 0) | (cache.phase >= PHASE_COUNT)).any():
        raise LearnabilityContractError("phase is outside [0,7)")
    expected_color = np.asarray(PHASE_MATCHINGS, dtype=np.int8)[cache.phase]
    expected_duration = np.asarray(PHASE_DURATIONS, dtype=np.float64)[cache.phase]
    if not np.array_equal(cache.color, expected_color):
        raise LearnabilityContractError("color does not match the frozen phase plan")
    if not np.array_equal(cache.duration, expected_duration):
        raise LearnabilityContractError("duration does not match the frozen phase plan")
    if ((cache.label < 0) | (cache.label >= 10)).any():
        raise LearnabilityContractError("labels are outside [0,10)")


def _validate_label_audit_cache(cache: LearnabilityLabelAuditCache) -> None:
    n = int(cache.sample_key.shape[0])
    expected = {
        "sample_key": (n,),
        "path_id": (n,),
        "outer_step": (n,),
        "phase": (n,),
        "denoising_target": (n, EDGES_PER_PHASE),
        "certificate_codes": (n, EDGES_PER_PHASE),
    }
    for name, shape in expected.items():
        value = getattr(cache, name)
        if value.dtype != _AUDIT_DTYPES[name] or value.shape != shape:
            raise LearnabilityContractError(f"invalid {name} dtype or shape")
        if not value.flags.c_contiguous or value.flags.writeable:
            raise LearnabilityContractError(f"{name} must be read-only C-contiguous")
    if len(np.unique(cache.sample_key)) != n:
        raise LearnabilityContractError("label/audit sample keys must be unique")
    if any(not 0 <= int(value) < PATH_ID_LIMIT for value in cache.path_id):
        raise LearnabilityContractError("audit path ID is outside the 20-bit field")
    if ((cache.outer_step < 0) | (cache.outer_step >= OUTER_STEPS)).any():
        raise LearnabilityContractError("outer step is outside [0,512)")
    if ((cache.phase < 0) | (cache.phase >= PHASE_COUNT)).any():
        raise LearnabilityContractError("phase is outside [0,7)")
    if not np.isfinite(cache.denoising_target).all():
        raise LearnabilityContractError("denoising target contains nonfinite values")
    expected_keys = np.fromiter(
        (
            sample_key(path, step, phase)
            for path, step, phase in zip(
                cache.path_id, cache.outer_step, cache.phase, strict=True
            )
        ),
        dtype=np.int64,
        count=n,
    )
    if not np.array_equal(cache.sample_key, expected_keys):
        raise LearnabilityContractError("sample keys do not encode audit coordinates")


def validate_cache_bundle(
    bundle: LearnabilityCacheBundle,
    *,
    expected_path_ids: Sequence[int] | None = None,
    expected_outer_steps: Sequence[int] | None = None,
) -> None:
    inputs, audit = bundle.inputs, bundle.labels_audit
    if inputs.sample_count != audit.sample_count:
        raise LearnabilityContractError("input and label/audit row counts differ")
    if not np.array_equal(inputs.sample_key, audit.sample_key):
        raise LearnabilityContractError("input and label/audit joins are not exact")
    if not np.array_equal(inputs.phase, audit.phase):
        raise LearnabilityContractError("input and audit phase columns differ")
    expected_time = np.fromiter(
        (
            selected_reverse_time(step, phase)
            for step, phase in zip(audit.outer_step, audit.phase, strict=True)
        ),
        dtype=np.float64,
        count=audit.sample_count,
    )
    if not np.array_equal(inputs.reverse_time, expected_time):
        raise LearnabilityContractError("reverse_time does not match the split-chain index")
    if expected_path_ids is not None:
        expected_paths = tuple(validate_path_id(value) for value in expected_path_ids)
        if tuple(sorted(np.unique(audit.path_id).tolist())) != tuple(
            sorted(expected_paths)
        ):
            raise LearnabilityContractError("cache path set differs from the frozen role")
    if expected_outer_steps is not None:
        expected_steps = tuple(_index(value, "outer_step") for value in expected_outer_steps)
        if tuple(sorted(np.unique(audit.outer_step).tolist())) != tuple(
            sorted(expected_steps)
        ):
            raise LearnabilityContractError("cache outer-step set differs from the plan")
        expected_pairs = {
            (path, step, phase)
            for path in np.unique(audit.path_id).tolist()
            for step in expected_steps
            for phase in range(PHASE_COUNT)
        }
        observed_pairs = set(
            zip(
                audit.path_id.tolist(),
                audit.outer_step.tolist(),
                audit.phase.tolist(),
                strict=True,
            )
        )
        if observed_pairs != expected_pairs:
            raise LearnabilityContractError(
                "cache does not contain exactly one row per path/step/phase"
            )


def _atomic_save_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_input_cache(path: str | Path, cache: LearnabilityInputCache) -> None:
    _validate_input_cache(cache)
    _atomic_save_npz(path, cache.arrays())


def save_label_audit_cache(
    path: str | Path, cache: LearnabilityLabelAuditCache
) -> None:
    _validate_label_audit_cache(cache)
    _atomic_save_npz(path, cache.arrays())


def _strict_load_npz(
    path: str | Path,
    *,
    fields: Sequence[str],
    dtypes: Mapping[str, np.dtype[Any]],
) -> dict[str, np.ndarray]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            raise LearnabilityContractError(
                f"{source.name} has unexpected or missing cache fields"
            )
        result = {}
        for name in fields:
            value = np.asarray(archive[name])
            if value.dtype != dtypes[name]:
                raise LearnabilityContractError(f"{source.name}:{name} dtype changed")
            result[name] = value
    return result


def load_input_cache(path: str | Path) -> LearnabilityInputCache:
    return LearnabilityInputCache(
        **_strict_load_npz(path, fields=INPUT_CACHE_FIELDS, dtypes=_INPUT_DTYPES)
    )


def load_label_audit_cache(path: str | Path) -> LearnabilityLabelAuditCache:
    return LearnabilityLabelAuditCache(
        **_strict_load_npz(
            path, fields=LABEL_AUDIT_CACHE_FIELDS, dtypes=_AUDIT_DTYPES
        )
    )


def save_cache_bundle(
    input_path: str | Path,
    label_audit_path: str | Path,
    bundle: LearnabilityCacheBundle,
) -> None:
    validate_cache_bundle(bundle)
    save_input_cache(input_path, bundle.inputs)
    save_label_audit_cache(label_audit_path, bundle.labels_audit)


def load_cache_bundle(
    input_path: str | Path,
    label_audit_path: str | Path,
    *,
    expected_path_ids: Sequence[int] | None = None,
    expected_outer_steps: Sequence[int] | None = None,
) -> LearnabilityCacheBundle:
    bundle = LearnabilityCacheBundle(
        load_input_cache(input_path), load_label_audit_cache(label_audit_path)
    )
    validate_cache_bundle(
        bundle,
        expected_path_ids=expected_path_ids,
        expected_outer_steps=expected_outer_steps,
    )
    return bundle


@dataclass(frozen=True)
class ModelInputs:
    """The complete and exclusive set of tensors allowed into ``model.forward``."""

    later_full_state: Tensor
    reverse_time: Tensor
    phase: Tensor
    color: Tensor
    duration: Tensor
    label: Tensor

    def __post_init__(self) -> None:
        values = {name: getattr(self, name) for name in MODEL_INPUT_FIELDS}
        if any(not isinstance(value, Tensor) for value in values.values()):
            raise TypeError("every ModelInputs field must be a torch.Tensor")
        batch = int(self.later_full_state.shape[0]) if self.later_full_state.ndim else -1
        expected = {
            "later_full_state": (batch, STATE_SIZE),
            "reverse_time": (batch,),
            "phase": (batch,),
            "color": (batch,),
            "duration": (batch,),
            "label": (batch,),
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise LearnabilityContractError(f"ModelInputs.{name} has wrong shape")
        if self.later_full_state.dtype not in {torch.float32, torch.float64}:
            raise LearnabilityContractError("later_full_state must be floating point")
        if self.reverse_time.dtype not in {torch.float32, torch.float64}:
            raise LearnabilityContractError("reverse_time must be floating point")
        if self.duration.dtype not in {torch.float32, torch.float64}:
            raise LearnabilityContractError("duration must be floating point")
        if self.phase.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise LearnabilityContractError("phase must be integral")
        if self.color.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise LearnabilityContractError("color must be integral")
        if self.label.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise LearnabilityContractError("label must be integral")
        devices = {value.device for value in values.values()}
        if len(devices) != 1:
            raise LearnabilityContractError("ModelInputs fields must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.later_full_state.shape[0])

    def index_select(self, indices: Tensor | Sequence[int]) -> ModelInputs:
        device = self.later_full_state.device
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        return ModelInputs(
            **{
                name: getattr(self, name).index_select(0, index_tensor)
                for name in MODEL_INPUT_FIELDS
            }
        )

    def to(self, device: str | torch.device) -> ModelInputs:
        return ModelInputs(
            **{name: getattr(self, name).to(device) for name in MODEL_INPUT_FIELDS}
        )


@dataclass(frozen=True)
class AuditTargets:
    """Targets and clustering coordinates kept outside the model firewall."""

    denoising_target: Tensor
    sample_key: Tensor
    path_id: Tensor
    outer_step: Tensor
    phase: Tensor
    certificate_codes: Tensor

    def __post_init__(self) -> None:
        if not all(
            isinstance(getattr(self, name), Tensor)
            for name in (
                "denoising_target",
                "sample_key",
                "path_id",
                "outer_step",
                "phase",
                "certificate_codes",
            )
        ):
            raise TypeError("every AuditTargets field must be a torch.Tensor")
        n = int(self.denoising_target.shape[0]) if self.denoising_target.ndim else -1
        if self.denoising_target.shape != (n, EDGES_PER_PHASE):
            raise LearnabilityContractError("AuditTargets target has wrong shape")
        if self.certificate_codes.shape != (n, EDGES_PER_PHASE):
            raise LearnabilityContractError("AuditTargets certificates have wrong shape")
        for name in ("sample_key", "path_id", "outer_step", "phase"):
            if getattr(self, name).shape != (n,):
                raise LearnabilityContractError(f"AuditTargets.{name} has wrong shape")
        devices = {
            getattr(self, name).device
            for name in (
                "denoising_target",
                "sample_key",
                "path_id",
                "outer_step",
                "phase",
                "certificate_codes",
            )
        }
        if len(devices) != 1:
            raise LearnabilityContractError("AuditTargets fields must share one device")

    @property
    def sample_count(self) -> int:
        return int(self.denoising_target.shape[0])

    def index_select(self, indices: Tensor | Sequence[int]) -> AuditTargets:
        index_tensor = torch.as_tensor(
            indices, dtype=torch.long, device=self.denoising_target.device
        )
        return AuditTargets(
            **{
                name: getattr(self, name).index_select(0, index_tensor)
                for name in (
                    "denoising_target",
                    "sample_key",
                    "path_id",
                    "outer_step",
                    "phase",
                    "certificate_codes",
                )
            }
        )

    def to(self, device: str | torch.device) -> AuditTargets:
        return AuditTargets(
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "denoising_target",
                    "sample_key",
                    "path_id",
                    "outer_step",
                    "phase",
                    "certificate_codes",
                )
            }
        )


def audit_targets_from_cache(
    cache: LearnabilityLabelAuditCache,
    *,
    device: str | torch.device = "cpu",
) -> AuditTargets:
    return AuditTargets(
        denoising_target=torch.as_tensor(
            np.asarray(cache.denoising_target).copy(),
            dtype=torch.float64,
            device=device,
        ),
        sample_key=torch.as_tensor(
            np.asarray(cache.sample_key).copy(), dtype=torch.long, device=device
        ),
        path_id=torch.as_tensor(
            np.asarray(cache.path_id).copy(), dtype=torch.long, device=device
        ),
        outer_step=torch.as_tensor(
            np.asarray(cache.outer_step).copy(), dtype=torch.long, device=device
        ),
        phase=torch.as_tensor(
            np.asarray(cache.phase).copy(), dtype=torch.long, device=device
        ),
        certificate_codes=torch.as_tensor(
            np.asarray(cache.certificate_codes).copy(),
            dtype=torch.uint8,
            device=device,
        ),
    )


def model_inputs_from_mapping(
    values: Mapping[str, Tensor], *, reject_forbidden: bool = True
) -> ModelInputs:
    keys = set(values)
    expected = set(MODEL_INPUT_FIELDS)
    if reject_forbidden and keys.intersection(FORBIDDEN_MODEL_INPUT_FIELDS):
        names = sorted(keys.intersection(FORBIDDEN_MODEL_INPUT_FIELDS))
        raise LearnabilityContractError(f"forbidden model input fields: {names}")
    if keys != expected:
        raise LearnabilityContractError(
            f"model input fields must be exactly {sorted(expected)}; got {sorted(keys)}"
        )
    return ModelInputs(**{name: values[name] for name in MODEL_INPUT_FIELDS})


def model_inputs_from_cache(
    cache: LearnabilityInputCache,
    *,
    device: str | torch.device = "cpu",
    floating_dtype: torch.dtype = torch.float32,
) -> ModelInputs:
    if floating_dtype not in {torch.float32, torch.float64}:
        raise ValueError("floating_dtype must be float32 or float64")
    return ModelInputs(
        later_full_state=torch.as_tensor(
            np.asarray(cache.later_full_state).copy(),
            dtype=floating_dtype,
            device=device,
        ),
        reverse_time=torch.as_tensor(
            np.asarray(cache.reverse_time).copy(),
            dtype=floating_dtype,
            device=device,
        ),
        phase=torch.as_tensor(
            np.asarray(cache.phase).copy(), dtype=torch.long, device=device
        ),
        color=torch.as_tensor(
            np.asarray(cache.color).copy(), dtype=torch.long, device=device
        ),
        duration=torch.as_tensor(
            np.asarray(cache.duration).copy(),
            dtype=floating_dtype,
            device=device,
        ),
        label=torch.as_tensor(
            np.asarray(cache.label).copy(), dtype=torch.long, device=device
        ),
    )


def call_model(model: nn.Module, inputs: ModelInputs) -> Tensor:
    """Single firewall entry point used by training and evaluation."""

    if type(inputs) is not ModelInputs:
        raise LearnabilityContractError("model.forward accepts only exact ModelInputs")
    prediction = model(inputs)
    if not isinstance(prediction, Tensor):
        raise TypeError("model.forward must return a torch.Tensor")
    if prediction.shape != (inputs.batch_size, EDGES_PER_PHASE):
        raise LearnabilityContractError("model prediction must have shape [B,392]")
    return prediction


def matching_indices(
    *, device: str | torch.device = "cpu"
) -> tuple[Tensor, Tensor]:
    tails = torch.as_tensor(
        np.stack([value[0] for value in _MATCHING_INDEX_ARRAYS]),
        dtype=torch.long,
        device=device,
    )
    heads = torch.as_tensor(
        np.stack([value[1] for value in _MATCHING_INDEX_ARRAYS]),
        dtype=torch.long,
        device=device,
    )
    return tails.contiguous(), heads.contiguous()


class PhaseConditionedLocalAffineCNN(nn.Module):
    """Fixed phase-conditioned local-affine-plus-CNN RB-label predictor."""

    def __init__(self, *, width: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        if width <= 0 or num_classes <= 0:
            raise ValueError("width and num_classes must be positive")
        self.width = int(width)
        self.num_classes = int(num_classes)
        metadata_channels = 1 + PHASE_COUNT + 4 + 1 + self.num_classes
        input_channels = 1 + metadata_channels
        self.conv1 = nn.Conv2d(
            input_channels,
            self.width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.conv3 = nn.Conv2d(
            self.width,
            self.width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.spatial_output = nn.Conv2d(self.width, 4, kernel_size=1)
        # [scaled tail, scaled head] plus the same permitted metadata.
        self.local_affine = nn.Linear(2 + metadata_channels, 1)
        tails, heads = matching_indices()
        self.register_buffer("tail_indices", tails, persistent=True)
        self.register_buffer("head_indices", heads, persistent=True)
        # Starting with no spatial residual leaves a transparent, exactly
        # representable local-affine teacher path.
        nn.init.zeros_(self.spatial_output.weight)
        nn.init.zeros_(self.spatial_output.bias)

    def _validated_metadata(self, inputs: ModelInputs, dtype: torch.dtype) -> Tensor:
        phase = inputs.phase.to(dtype=torch.long)
        color = inputs.color.to(dtype=torch.long)
        label = inputs.label.to(dtype=torch.long)
        if (
            bool(torch.any((phase < 0) | (phase >= PHASE_COUNT)))
            or bool(torch.any((color < 0) | (color >= 4)))
            or bool(torch.any((label < 0) | (label >= self.num_classes)))
        ):
            raise LearnabilityContractError("phase/color/label is outside its range")
        expected_color = torch.as_tensor(
            PHASE_MATCHINGS, dtype=torch.long, device=phase.device
        )[phase]
        expected_duration = torch.as_tensor(
            PHASE_DURATIONS, dtype=inputs.duration.dtype, device=phase.device
        )[phase]
        if not torch.equal(color, expected_color):
            raise LearnabilityContractError("color does not match phase")
        if not torch.equal(inputs.duration, expected_duration):
            raise LearnabilityContractError("duration does not match phase")
        pieces = [
            inputs.reverse_time.to(dtype=dtype).reshape(-1, 1),
            F.one_hot(phase, num_classes=PHASE_COUNT).to(dtype=dtype),
            F.one_hot(color, num_classes=4).to(dtype=dtype),
            inputs.duration.to(dtype=dtype).reshape(-1, 1),
            F.one_hot(label, num_classes=self.num_classes).to(dtype=dtype),
        ]
        return torch.cat(pieces, dim=1)

    def forward(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise LearnabilityContractError("forward accepts only exact ModelInputs")
        state = inputs.later_full_state
        dtype = self.conv1.weight.dtype
        state = state.to(dtype=dtype)
        metadata = self._validated_metadata(inputs, dtype)
        batch = inputs.batch_size
        density = state.reshape(batch, 1, GRID_SIZE, GRID_SIZE) * float(STATE_SIZE)
        metadata_planes = metadata[:, :, None, None].expand(
            batch, metadata.shape[1], GRID_SIZE, GRID_SIZE
        )
        hidden = F.silu(self.conv1(torch.cat([density, metadata_planes], dim=1)))
        hidden = F.silu(self.conv2(hidden))
        hidden = F.silu(self.conv3(hidden))
        spatial = self.spatial_output(hidden).reshape(batch, 4, STATE_SIZE)

        colors = inputs.color.to(dtype=torch.long)
        rows = torch.arange(batch, device=state.device)
        heads = self.head_indices[colors]
        tails = self.tail_indices[colors]
        active_spatial = spatial[rows, colors].gather(1, heads)
        head_mass = state.gather(1, heads) * float(STATE_SIZE)
        tail_mass = state.gather(1, tails) * float(STATE_SIZE)
        local_metadata = metadata[:, None, :].expand(
            batch, EDGES_PER_PHASE, metadata.shape[1]
        )
        local_features = torch.cat(
            [tail_mass[:, :, None], head_mass[:, :, None], local_metadata], dim=2
        )
        local = self.local_affine(local_features).squeeze(-1)
        return active_spatial + local


# Concise public alias used by the orchestration module.
JacobiRBPhasePredictor = PhaseConditionedLocalAffineCNN


def configure_exact_synthetic_teacher(model: PhaseConditionedLocalAffineCNN) -> None:
    """Set the local skip to the exact synthetic teacher and zero the CNN.

    This helper is for algebra/regression controls, not physical initialization.
    It proves representability without using an oracle in physical training.
    """

    with torch.no_grad():
        for layer in (model.conv1, model.conv2, model.conv3, model.spatial_output):
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
        model.local_affine.weight.zero_()
        model.local_affine.bias.zero_()
        # features: scaled tail, scaled head, tau, phase[7], color[4],
        # duration, label[10].
        weight = model.local_affine.weight[0]
        weight[0] = -1.0
        weight[1] = 1.0
        weight[2] = 0.5  # 0.25 * (2*tau - 1)
        phase_start = 3
        for phase in range(PHASE_COUNT):
            weight[phase_start + phase] = 0.05 * phase
        duration_index = 3 + PHASE_COUNT + 4
        weight[duration_index] = 0.10
        model.local_affine.bias.fill_(-0.25 - 0.15 - 0.075)


def synthetic_teacher_target(inputs: ModelInputs) -> Tensor:
    """Exact local teacher using only permitted later-state information."""

    tails, heads = matching_indices(device=inputs.later_full_state.device)
    colors = inputs.color.to(dtype=torch.long)
    rows = torch.arange(inputs.batch_size, device=inputs.later_full_state.device)
    active_tails = tails[colors]
    active_heads = heads[colors]
    state = inputs.later_full_state.to(dtype=torch.float64)
    tail = state.gather(1, active_tails)
    head = state.gather(1, active_heads)
    tau = inputs.reverse_time.to(dtype=torch.float64).reshape(-1, 1)
    phase = inputs.phase.to(dtype=torch.float64).reshape(-1, 1)
    duration = inputs.duration.to(dtype=torch.float64).reshape(-1, 1)
    return (
        float(STATE_SIZE) * (head - tail)
        + 0.25 * (2.0 * tau - 1.0)
        + 0.05 * (phase - 3.0)
        + 0.10 * (duration - 0.75)
    )


def stable_sum(values: np.ndarray | Tensor | Iterable[float]) -> float:
    if isinstance(values, Tensor):
        array = values.detach().to(device="cpu", dtype=torch.float64).numpy()
        return math.fsum(float(value) for value in array.reshape(-1))
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=np.float64)
        return math.fsum(float(value) for value in array.reshape(-1))
    return math.fsum(float(value) for value in values)


def stable_mse(prediction: np.ndarray | Tensor, target: np.ndarray | Tensor) -> float:
    prediction_array = (
        prediction.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(prediction, Tensor)
        else np.asarray(prediction, dtype=np.float64)
    )
    target_array = (
        target.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(target, Tensor)
        else np.asarray(target, dtype=np.float64)
    )
    if prediction_array.shape != target_array.shape or prediction_array.size == 0:
        raise LearnabilityContractError("MSE arrays must have equal nonempty shapes")
    difference = prediction_array - target_array
    if not np.isfinite(difference).all():
        return math.inf
    return stable_sum(difference * difference) / difference.size


def exact_global_target_scale(target: np.ndarray | Tensor) -> float:
    """Training-only ``sqrt(mean(Z**2))`` in binary64 units."""

    target_array = (
        target.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(target, Tensor)
        else np.asarray(target, dtype=np.float64)
    )
    if target_array.size == 0 or not np.isfinite(target_array).all():
        raise LearnabilityContractError("target scale requires finite nonempty targets")
    scale = math.sqrt(stable_sum(target_array * target_array) / target_array.size)
    if not math.isfinite(scale) or scale <= 0.0:
        raise LearnabilityContractError("global target scale must be finite and positive")
    return scale


def globally_scaled_mse(
    prediction: Tensor, target: Tensor, target_scale: float
) -> tuple[Tensor, Tensor]:
    """Return ``(optimizer_loss, raw_mse)`` without component weighting."""

    scale = float(target_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise LearnabilityContractError("target_scale must be finite and positive")
    if prediction.shape != target.shape:
        raise LearnabilityContractError("prediction and target shapes differ")
    raw = torch.mean(
        (prediction.to(dtype=torch.float64) - target.to(dtype=torch.float64)).square()
    )
    return raw / (scale * scale), raw


@dataclass(frozen=True)
class MetadataBaseline:
    values: np.ndarray = field(repr=False, compare=False)
    counts: np.ndarray = field(repr=False, compare=False)
    schema: str = METADATA_BASELINE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "values", _readonly_c_array(self.values, np.dtype(np.float64))
        )
        object.__setattr__(
            self, "counts", _readonly_c_array(self.counts, np.dtype(np.int64))
        )
        if self.values.shape != (4, PHASE_COUNT, EDGES_PER_PHASE):
            raise LearnabilityContractError("metadata baseline has wrong value shape")
        if self.counts.shape != (4, PHASE_COUNT):
            raise LearnabilityContractError("metadata baseline has wrong count shape")
        if (
            not np.isfinite(self.values).all()
            or (self.counts <= 0).any()
            or self.schema != METADATA_BASELINE_SCHEMA
        ):
            raise LearnabilityContractError("metadata baseline is incomplete or invalid")

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        for array in (self.values, self.counts):
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def predict(self, outer_step: np.ndarray, phase: np.ndarray) -> np.ndarray:
        steps = np.asarray(outer_step, dtype=np.int64)
        phases = np.asarray(phase, dtype=np.int64)
        if steps.shape != phases.shape or steps.ndim != 1:
            raise LearnabilityContractError("baseline coordinates must be equal 1-D arrays")
        if (
            ((steps < 0) | (steps >= OUTER_STEPS)).any()
            or ((phases < 0) | (phases >= PHASE_COUNT)).any()
        ):
            raise LearnabilityContractError("baseline coordinate is outside the chain")
        return np.ascontiguousarray(self.values[steps // 128, phases])


def fit_metadata_baseline(
    target: np.ndarray | Tensor,
    outer_step: np.ndarray | Tensor,
    phase: np.ndarray | Tensor,
) -> MetadataBaseline:
    """Fit the frozen time-quartile/phase/edge mean from training rows only."""

    targets = (
        target.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(target, Tensor)
        else np.asarray(target, dtype=np.float64)
    )
    steps = (
        outer_step.detach().to(device="cpu", dtype=torch.long).numpy()
        if isinstance(outer_step, Tensor)
        else np.asarray(outer_step, dtype=np.int64)
    )
    phases = (
        phase.detach().to(device="cpu", dtype=torch.long).numpy()
        if isinstance(phase, Tensor)
        else np.asarray(phase, dtype=np.int64)
    )
    if (
        targets.ndim != 2
        or targets.shape[1] != EDGES_PER_PHASE
        or steps.shape != (targets.shape[0],)
        or phases.shape != (targets.shape[0],)
        or not np.isfinite(targets).all()
    ):
        raise LearnabilityContractError("metadata baseline inputs have invalid shapes")
    values = np.empty((4, PHASE_COUNT, EDGES_PER_PHASE), dtype=np.float64)
    counts = np.zeros((4, PHASE_COUNT), dtype=np.int64)
    quartiles = steps // 128
    for quartile in range(4):
        for phase_index in range(PHASE_COUNT):
            mask = (quartiles == quartile) & (phases == phase_index)
            counts[quartile, phase_index] = int(mask.sum())
            if not bool(mask.any()):
                raise LearnabilityContractError(
                    "training cache does not populate every metadata baseline cell"
                )
            values[quartile, phase_index] = np.mean(
                targets[mask], axis=0, dtype=np.float64
            )
    return MetadataBaseline(values, counts)


def save_metadata_baseline(
    npz_path: str | Path, baseline: MetadataBaseline
) -> None:
    _atomic_save_npz(
        npz_path, {"values": baseline.values, "counts": baseline.counts}
    )


def load_metadata_baseline(path: str | Path) -> MetadataBaseline:
    arrays = _strict_load_npz(
        path,
        fields=("values", "counts"),
        dtypes={"values": np.dtype(np.float64), "counts": np.dtype(np.int64)},
    )
    return MetadataBaseline(**arrays)


@dataclass(frozen=True)
class TrainingPlan:
    optimizer: str = "Adam"
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    maximum_updates: int = 4_000
    validation_interval: int = 100
    gradient_norm_clip: float = 1.0
    model_seeds: tuple[int, ...] = (261_201, 261_202, 261_203)
    deterministic: bool = True
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        if self.optimizer != "Adam" or self.mixed_precision:
            raise LearnabilityContractError("the frozen optimizer plan changed")
        if (
            self.learning_rate != 1.0e-3
            or self.weight_decay != 0.0
            or self.batch_size != 32
            or self.maximum_updates != 4_000
            or self.validation_interval != 100
            or self.gradient_norm_clip != 1.0
            or tuple(self.model_seeds) != (261_201, 261_202, 261_203)
        ):
            raise LearnabilityContractError("the frozen training plan changed")

    def to_record(self) -> dict[str, Any]:
        return {"version": TRAINING_VERSION, **asdict(self)}


def frozen_training_plan() -> TrainingPlan:
    return TrainingPlan()


def deterministic_batch_indices(
    sample_count: Any, batch_size: Any, update: Any, seed: Any
) -> np.ndarray:
    """Return a stateless deterministic batch from epoch-specific permutations."""

    n = _index(sample_count, "sample_count")
    batch = _index(batch_size, "batch_size")
    cursor = _index(update, "update") * batch
    root_seed = _index(seed, "seed")
    if n <= 0 or batch <= 0 or update < 0:
        raise LearnabilityContractError("batch schedule coordinates are invalid")
    result: list[np.ndarray] = []
    remaining = batch
    while remaining:
        epoch, offset = divmod(cursor, n)
        generator = np.random.Generator(
            np.random.Philox([root_seed, int(epoch), 0x4A52424C])
        )
        permutation = generator.permutation(n)
        take = min(remaining, n - offset)
        result.append(permutation[offset : offset + take].astype(np.int64, copy=False))
        cursor += take
        remaining -= take
    return np.ascontiguousarray(np.concatenate(result))


def enable_deterministic_torch() -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def clone_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in state_dict.items()
    }


@dataclass(frozen=True)
class CheckpointCandidate:
    seed: int
    update: int
    validation_mse: float
    state_sha256: str
    state_dict: Mapping[str, Tensor] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.update < 0 or not math.isfinite(self.validation_mse):
            raise LearnabilityContractError("checkpoint candidate is nonfinite or invalid")
        if len(self.state_sha256) != 64:
            raise LearnabilityContractError("checkpoint state hash is invalid")


def select_checkpoint_candidate(
    candidates: Sequence[CheckpointCandidate],
) -> CheckpointCandidate:
    if not candidates:
        raise LearnabilityContractError("no checkpoint candidates were supplied")
    return min(
        candidates,
        key=lambda candidate: (
            float(candidate.validation_mse),
            int(candidate.seed),
            int(candidate.update),
        ),
    )


def restore_checkpoint_candidate(
    model: nn.Module, candidate: CheckpointCandidate
) -> None:
    """Load one selected state only after verifying its deterministic hash."""

    if state_dict_sha256(candidate.state_dict) != candidate.state_sha256:
        raise LearnabilityContractError("checkpoint candidate hash does not match its state")
    model.load_state_dict(candidate.state_dict, strict=True)
    if state_dict_sha256(model.state_dict()) != candidate.state_sha256:
        raise LearnabilityContractError("checkpoint replay hash mismatch")


@torch.no_grad()
def predict_in_batches(
    model: nn.Module,
    inputs: ModelInputs,
    *,
    batch_size: int = 32,
) -> Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    outputs = []
    for start in range(0, inputs.batch_size, int(batch_size)):
        stop = min(inputs.batch_size, start + int(batch_size))
        index = torch.arange(
            start, stop, device=inputs.later_full_state.device, dtype=torch.long
        )
        outputs.append(call_model(model, inputs.index_select(index)).to(torch.float64))
    if was_training:
        model.train()
    return torch.cat(outputs, dim=0)


def evaluate_model_mse(
    model: nn.Module,
    inputs: ModelInputs,
    target: Tensor,
    *,
    batch_size: int = 32,
) -> tuple[float, Tensor]:
    prediction = predict_in_batches(model, inputs, batch_size=batch_size)
    return stable_mse(prediction, target), prediction


@dataclass(frozen=True)
class TrainingRunResult:
    seed: int
    selected: CheckpointCandidate
    history: tuple[Mapping[str, float | int], ...]
    finite: bool


@dataclass(frozen=True)
class TrainingResumeSnapshot:
    """Exact validation-boundary state for deterministic task resume."""

    seed: int
    completed_update: int
    model_state_dict: Mapping[str, Tensor] = field(repr=False, compare=False)
    optimizer_state_dict: Mapping[str, Any] = field(repr=False, compare=False)
    best_candidate: CheckpointCandidate = field(repr=False, compare=False)
    history: tuple[Mapping[str, float | int], ...]
    finite: bool
    torch_rng_state: Tensor = field(repr=False, compare=False)
    cuda_rng_states: tuple[Tensor, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.completed_update < 0 or self.seed != self.best_candidate.seed:
            raise LearnabilityContractError("training resume cursor is invalid")
        if state_dict_sha256(self.model_state_dict) == "":
            raise LearnabilityContractError("training resume model state is invalid")


def train_deterministic_regressor(
    model_factory: Callable[[], nn.Module],
    train_inputs: ModelInputs,
    train_target: Tensor,
    validation_inputs: ModelInputs,
    validation_target: Tensor,
    *,
    target_scale: float,
    seed: int,
    plan: TrainingPlan | None = None,
    maximum_updates: int | None = None,
    resume_snapshot: TrainingResumeSnapshot | None = None,
    checkpoint_callback: Callable[[TrainingResumeSnapshot], None] | None = None,
) -> TrainingRunResult:
    """Run the frozen deterministic Adam path and select on validation only.

    ``maximum_updates`` exists solely for reduced CPU integration tests.  The
    production caller must omit it and persist the frozen :class:`TrainingPlan`.
    """

    active = plan or TrainingPlan()
    updates = active.maximum_updates if maximum_updates is None else int(maximum_updates)
    if updates < 0 or updates > active.maximum_updates:
        raise LearnabilityContractError("test update count is outside the frozen maximum")
    if train_target.shape != (train_inputs.batch_size, EDGES_PER_PHASE):
        raise LearnabilityContractError("training target shape is invalid")
    if validation_target.shape != (validation_inputs.batch_size, EDGES_PER_PHASE):
        raise LearnabilityContractError("validation target shape is invalid")
    enable_deterministic_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = model_factory().to(train_inputs.later_full_state.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=active.learning_rate,
        weight_decay=active.weight_decay,
    )
    candidates: list[CheckpointCandidate] = []
    history: list[Mapping[str, float | int]] = []
    completed_update = 0

    def validate(update: int) -> None:
        mse, _ = evaluate_model_mse(
            model,
            validation_inputs,
            validation_target,
            batch_size=active.batch_size,
        )
        state = clone_state_dict(model.state_dict())
        candidate = CheckpointCandidate(
            seed=int(seed),
            update=int(update),
            validation_mse=float(mse),
            state_sha256=state_dict_sha256(state),
            state_dict=state,
        )
        candidates.append(candidate)
        history.append({"update": int(update), "validation_mse": float(mse)})

    def snapshot(update: int, finite_value: bool) -> None:
        if checkpoint_callback is None:
            return
        checkpoint_callback(
            TrainingResumeSnapshot(
                seed=int(seed),
                completed_update=int(update),
                model_state_dict=clone_state_dict(model.state_dict()),
                optimizer_state_dict=optimizer.state_dict(),
                best_candidate=select_checkpoint_candidate(candidates),
                history=tuple(history),
                finite=bool(finite_value),
                torch_rng_state=torch.get_rng_state().clone(),
                cuda_rng_states=(
                    tuple(state.clone() for state in torch.cuda.get_rng_state_all())
                    if torch.cuda.is_available()
                    else ()
                ),
            )
        )

    finite = True
    if resume_snapshot is None:
        validate(0)
        snapshot(0, True)
    else:
        if (
            int(resume_snapshot.seed) != int(seed)
            or resume_snapshot.completed_update > updates
            or state_dict_sha256(resume_snapshot.model_state_dict) == ""
        ):
            raise LearnabilityContractError("training resume snapshot is incompatible")
        model.load_state_dict(dict(resume_snapshot.model_state_dict), strict=True)
        optimizer.load_state_dict(dict(resume_snapshot.optimizer_state_dict))
        candidates.append(resume_snapshot.best_candidate)
        history.extend(dict(row) for row in resume_snapshot.history)
        completed_update = int(resume_snapshot.completed_update)
        finite = bool(resume_snapshot.finite)
        torch.set_rng_state(resume_snapshot.torch_rng_state)
        if torch.cuda.is_available() and resume_snapshot.cuda_rng_states:
            torch.cuda.set_rng_state_all(list(resume_snapshot.cuda_rng_states))
    if not finite:
        return TrainingRunResult(
            int(seed),
            select_checkpoint_candidate(candidates),
            tuple(history),
            False,
        )
    model.train()
    for update in range(completed_update + 1, updates + 1):
        indices_np = deterministic_batch_indices(
            train_inputs.batch_size, active.batch_size, update - 1, int(seed)
        )
        indices = torch.as_tensor(
            indices_np,
            dtype=torch.long,
            device=train_inputs.later_full_state.device,
        )
        batch_inputs = train_inputs.index_select(indices)
        batch_target = train_target.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        prediction = call_model(model, batch_inputs)
        loss, raw = globally_scaled_mse(prediction, batch_target, target_scale)
        if not bool(torch.isfinite(loss)):
            finite = False
            snapshot(update - 1, False)
            break
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), active.gradient_norm_clip
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            finite = False
            snapshot(update - 1, False)
            break
        optimizer.step()
        if update % active.validation_interval == 0 or update == updates:
            history.append(
                {
                    "update": int(update),
                    "train_raw_mse": float(raw.detach().cpu()),
                    "scaled_loss": float(loss.detach().cpu()),
                    "preclip_gradient_norm": float(
                        torch.as_tensor(gradient_norm).detach().cpu()
                    ),
                }
            )
            validate(update)
            snapshot(update, True)
            model.train()
    if not candidates:
        raise LearnabilityContractError("training produced no validation checkpoint")
    selected = select_checkpoint_candidate(candidates)
    return TrainingRunResult(int(seed), selected, tuple(history), finite)


@dataclass(frozen=True)
class PathMSE:
    path_id: int
    model_mse: float
    metadata_mse: float
    zero_mse: float

    @property
    def metadata_improvement(self) -> float:
        return self.metadata_mse - self.model_mse

    @property
    def relative_metadata_improvement(self) -> float:
        if self.metadata_mse <= 0:
            return -math.inf if self.model_mse > 0 else 0.0
        return 1.0 - self.model_mse / self.metadata_mse


@dataclass(frozen=True)
class PathMetricSummary:
    paths: tuple[PathMSE, ...]
    aggregate_model_mse: float
    aggregate_metadata_mse: float
    aggregate_zero_mse: float
    aggregate_relative_metadata_improvement: float
    median_relative_metadata_improvement: float

    @property
    def improvements(self) -> tuple[float, ...]:
        return tuple(path.metadata_improvement for path in self.paths)


def path_mse_summary(
    prediction: np.ndarray | Tensor,
    target: np.ndarray | Tensor,
    metadata_prediction: np.ndarray | Tensor,
    path_id: np.ndarray | Tensor,
) -> PathMetricSummary:
    prediction_np = (
        prediction.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(prediction, Tensor)
        else np.asarray(prediction, dtype=np.float64)
    )
    target_np = (
        target.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(target, Tensor)
        else np.asarray(target, dtype=np.float64)
    )
    metadata_np = (
        metadata_prediction.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(metadata_prediction, Tensor)
        else np.asarray(metadata_prediction, dtype=np.float64)
    )
    paths_np = (
        path_id.detach().to(device="cpu", dtype=torch.long).numpy()
        if isinstance(path_id, Tensor)
        else np.asarray(path_id, dtype=np.int64)
    )
    if (
        prediction_np.shape != target_np.shape
        or metadata_np.shape != target_np.shape
        or target_np.ndim != 2
        or target_np.shape[1] != EDGES_PER_PHASE
        or paths_np.shape != (target_np.shape[0],)
        or target_np.size == 0
    ):
        raise LearnabilityContractError("path metric arrays have invalid shapes")
    if not (
        np.isfinite(prediction_np).all()
        and np.isfinite(target_np).all()
        and np.isfinite(metadata_np).all()
    ):
        raise LearnabilityContractError("path metric arrays must be finite")
    rows: list[PathMSE] = []
    for path in sorted(np.unique(paths_np).tolist()):
        mask = paths_np == path
        rows.append(
            PathMSE(
                int(path),
                stable_mse(prediction_np[mask], target_np[mask]),
                stable_mse(metadata_np[mask], target_np[mask]),
                stable_mse(np.zeros_like(target_np[mask]), target_np[mask]),
            )
        )
    aggregate_model = stable_mse(prediction_np, target_np)
    aggregate_metadata = stable_mse(metadata_np, target_np)
    aggregate_zero = stable_mse(np.zeros_like(target_np), target_np)
    relative = (
        1.0 - aggregate_model / aggregate_metadata
        if aggregate_metadata > 0
        else (-math.inf if aggregate_model > 0 else 0.0)
    )
    median = float(np.median([row.relative_metadata_improvement for row in rows]))
    return PathMetricSummary(
        tuple(rows),
        aggregate_model,
        aggregate_metadata,
        aggregate_zero,
        relative,
        median,
    )


@dataclass(frozen=True)
class PathSignResult:
    path_count: int
    positive_count: int
    zero_count: int
    negative_count: int
    all_strictly_positive: bool
    one_sided_all_positive_p_value: float


def all_positive_path_sign_test(improvements: Sequence[float]) -> PathSignResult:
    values = tuple(float(value) for value in improvements)
    if not values or not all(math.isfinite(value) for value in values):
        raise LearnabilityContractError("path improvements must be finite and nonempty")
    positive = sum(value > 0.0 for value in values)
    zero = sum(value == 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    passed = positive == len(values)
    return PathSignResult(
        len(values),
        positive,
        zero,
        negative,
        passed,
        2.0 ** (-len(values)) if passed else 1.0,
    )


def confirmation_signal_pass(summary: PathMetricSummary) -> bool:
    """Closed performance portion of the confirmation gate.

    Cache, seal, and hash checks live in the gate/orchestration modules.
    """

    signs = all_positive_path_sign_test(summary.improvements)
    return bool(
        signs.all_strictly_positive
        and summary.aggregate_model_mse < summary.aggregate_zero_mse
    )


__all__ = [
    "AuditTargets",
    "CheckpointCandidate",
    "CONFIRMATION_PATH_IDS",
    "EDGES_PER_PHASE",
    "FORBIDDEN_MODEL_INPUT_FIELDS",
    "GRID_SIZE",
    "INPUT_CACHE_FIELDS",
    "INPUT_CACHE_SCHEMA",
    "JacobiRBPhasePredictor",
    "LABEL_AUDIT_CACHE_FIELDS",
    "LABEL_AUDIT_CACHE_SCHEMA",
    "LEARNABILITY_VERSION",
    "LearnabilityCacheBundle",
    "LearnabilityContractError",
    "LearnabilityInputCache",
    "LearnabilityLabelAuditCache",
    "LearnabilityPathPlan",
    "LearnabilityScientificConfig",
    "METADATA_BASELINE_SCHEMA",
    "MODEL_INPUT_FIELDS",
    "MODEL_VERSION",
    "MetadataBaseline",
    "ModelInputs",
    "OUTER_STEPS",
    "PHASE_COUNT",
    "PHASE_DURATIONS",
    "PHASE_MATCHINGS",
    "PathIDClaim",
    "PathIDCollision",
    "PathMSE",
    "PathMetricSummary",
    "PathSignResult",
    "PhaseConditionedLocalAffineCNN",
    "SELECTED_OUTER_STEPS",
    "STATE_SIZE",
    "TRAIN_PATH_IDS",
    "TRAINING_VERSION",
    "TrainingPlan",
    "TrainingResumeSnapshot",
    "TrainingRunResult",
    "VALIDATION_PATH_IDS",
    "all_positive_path_sign_test",
    "audit_targets_from_cache",
    "call_model",
    "canonical_json_bytes",
    "clone_state_dict",
    "configure_exact_synthetic_teacher",
    "confirmation_signal_pass",
    "deterministic_batch_indices",
    "discover_repository_path_id_claims",
    "enable_deterministic_torch",
    "evaluate_model_mse",
    "exact_global_target_scale",
    "expected_selected_sample_count",
    "expected_transition_count",
    "fit_metadata_baseline",
    "frozen_path_plan",
    "frozen_scientific_config",
    "frozen_training_plan",
    "globally_scaled_mse",
    "load_cache_bundle",
    "load_input_cache",
    "load_label_audit_cache",
    "load_metadata_baseline",
    "matching_indices",
    "model_inputs_from_cache",
    "model_inputs_from_mapping",
    "path_mse_summary",
    "predict_in_batches",
    "sample_key",
    "save_cache_bundle",
    "save_input_cache",
    "save_label_audit_cache",
    "save_metadata_baseline",
    "scan_path_id_collisions",
    "select_checkpoint_candidate",
    "restore_checkpoint_candidate",
    "selected_reverse_time",
    "semantic_sha256",
    "stable_mse",
    "stable_sum",
    "state_dict_sha256",
    "synthetic_teacher_target",
    "train_deterministic_regressor",
    "validate_cache_bundle",
    "validate_path_id",
]
