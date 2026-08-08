"""Pure helpers for exact noisy-Jacobi Bayes-power calibration.

This additive module calibrates the existing exact Jacobi/Rao--Blackwell
learnability pipeline on a bounded law with a known conditional mean.  The
model is trained on the unchanged noisy Rao--Blackwell transition label.  The
analytic Bayes mean is stored in a physically separate audit cache and is
never exposed through :class:`~mnist.d0_jacobi_rb_learnability.ModelInputs`.

The module intentionally contains no transition sampler, CLI, provenance
reader, or confirmation opener.  Orchestration supplies certified transition
outputs and uses the schemas and deterministic helpers below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    GRID_SIZE,
    INPUT_CACHE_FIELDS,
    MODEL_INPUT_FIELDS,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    SELECTED_OUTER_STEPS,
    STATE_SIZE,
    AuditTargets,
    JacobiRBPhasePredictor,
    LearnabilityCacheBundle,
    LearnabilityContractError,
    LearnabilityInputCache,
    LearnabilityLabelAuditCache,
    MetadataBaseline,
    ModelInputs,
    TrainingPlan,
    audit_targets_from_cache,
    call_model,
    exact_global_target_scale,
    fit_metadata_baseline,
    frozen_training_plan,
    matching_indices,
    load_input_cache,
    model_inputs_from_cache,
    sample_key,
    save_input_cache,
    selected_reverse_time,
    stable_mse,
    train_deterministic_regressor,
    validate_path_id,
)


BAYES_POWER_VERSION = "d0-jacobi-rb-noisy-bayes-power-v1"
SCIENTIFIC_CONFIG_SCHEMA = BAYES_POWER_VERSION + "-scientific-config"
PATH_ID_PLAN_SCHEMA = BAYES_POWER_VERSION + "-path-id-plan"
LABEL_CACHE_SCHEMA = BAYES_POWER_VERSION + "-noisy-label-cache"
ORACLE_AUDIT_CACHE_SCHEMA = BAYES_POWER_VERSION + "-oracle-audit-cache"

ROOT_SEED = 261_211
MODEL_SEEDS = (261_201, 261_202, 261_203)
TEACHER_LAW = 1
NULL_LAW = 0
LAW_NAMES = {NULL_LAW: "stationary_null", TEACHER_LAW: "bounded_teacher"}

TEACHER_TRAIN_PATH_IDS = tuple(range(0xE3000, 0xE3008))
TEACHER_VALIDATION_PATH_IDS = tuple(range(0xE3010, 0xE3018))
TEACHER_CONFIRMATION_PATH_IDS = tuple(range(0xE3020, 0xE3028))
NULL_TRAIN_PATH_IDS = tuple(range(0xE4000, 0xE4008))
NULL_VALIDATION_PATH_IDS = tuple(range(0xE4010, 0xE4018))
NULL_CONFIRMATION_PATH_IDS = tuple(range(0xE4020, 0xE4028))

BAYES_LABEL_CACHE_FIELDS = (
    "sample_key",
    "path_id",
    "outer_step",
    "phase",
    "law",
    "denoising_target",
    "certificate_codes",
)
BAYES_ORACLE_AUDIT_FIELDS = (
    "sample_key",
    "path_id",
    "outer_step",
    "phase",
    "law",
    "earlier_head_fraction",
    "arrival_head_fraction",
    "exposure",
    "oracle_conditional_mean",
)
BAYES_FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    set(FORBIDDEN_MODEL_INPUT_FIELDS)
    | {
        "law",
        "exposure",
        "earlier_head_fraction",
        "arrival_head_fraction",
        "oracle_conditional_mean",
        "oracle_audit",
    }
)

_LABEL_DTYPES: Mapping[str, np.dtype[Any]] = {
    "sample_key": np.dtype(np.int64),
    "path_id": np.dtype(np.int64),
    "outer_step": np.dtype(np.int16),
    "phase": np.dtype(np.int8),
    "law": np.dtype(np.int8),
    "denoising_target": np.dtype(np.float64),
    "certificate_codes": np.dtype(np.uint8),
}
_ORACLE_DTYPES: Mapping[str, np.dtype[Any]] = {
    "sample_key": np.dtype(np.int64),
    "path_id": np.dtype(np.int64),
    "outer_step": np.dtype(np.int16),
    "phase": np.dtype(np.int8),
    "law": np.dtype(np.int8),
    "earlier_head_fraction": np.dtype(np.float64),
    "arrival_head_fraction": np.dtype(np.float64),
    "exposure": np.dtype(np.float64),
    "oracle_conditional_mean": np.dtype(np.float64),
}


class BayesPowerContractError(LearnabilityContractError):
    """A calibration law, cache, namespace, or audit contract is invalid."""


def _index(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _readonly_array(
    value: Any, dtype: np.dtype[Any], *, strict_dtype: bool = True
) -> np.ndarray:
    source = np.asarray(value)
    if strict_dtype and source.dtype != dtype:
        raise BayesPowerContractError(
            f"array dtype {source.dtype} does not match required {dtype}"
        )
    result = np.array(source, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def normalize_law(value: str | int) -> int:
    if isinstance(value, str):
        names = {name: code for code, name in LAW_NAMES.items()}
        try:
            return names[value]
        except KeyError as exc:
            raise BayesPowerContractError(f"unknown control law {value!r}") from exc
    code = _index(value, "law")
    if code not in LAW_NAMES:
        raise BayesPowerContractError("control law must be teacher or null")
    return code


@dataclass(frozen=True)
class BayesPowerScientificConfig:
    """Frozen scientific design for the analytic noisy-label calibration."""

    schema: str = SCIENTIFIC_CONFIG_SCHEMA
    schema_version: int = 1
    grid_size: int = GRID_SIZE
    outer_steps: int = OUTER_STEPS
    selected_outer_steps: tuple[int, ...] = SELECTED_OUTER_STEPS
    phase_matchings: tuple[int, ...] = PHASE_MATCHINGS
    phase_durations: tuple[float, ...] = PHASE_DURATIONS
    edges_per_phase: int = EDGES_PER_PHASE
    paths_per_role_per_law: int = 8
    root_seed: int = ROOT_SEED
    model_width: int = 32
    model_seeds: tuple[int, ...] = MODEL_SEEDS
    teacher_amplitude: float = 0.5
    oracle_minimum_relative_gain: float = 0.01
    minimum_oracle_gain_recovery: float = 0.50

    def __post_init__(self) -> None:
        frozen = (
            self.grid_size == GRID_SIZE
            and self.outer_steps == OUTER_STEPS
            and tuple(self.selected_outer_steps) == SELECTED_OUTER_STEPS
            and tuple(self.phase_matchings) == PHASE_MATCHINGS
            and tuple(self.phase_durations) == PHASE_DURATIONS
            and self.edges_per_phase == EDGES_PER_PHASE
            and self.paths_per_role_per_law == 8
            and self.root_seed == ROOT_SEED
            and self.model_width == 32
            and tuple(self.model_seeds) == MODEL_SEEDS
            and self.teacher_amplitude == 0.5
            and self.oracle_minimum_relative_gain == 0.01
            and self.minimum_oracle_gain_recovery == 0.50
        )
        if self.schema != SCIENTIFIC_CONFIG_SCHEMA or self.schema_version != 1:
            raise BayesPowerContractError("unsupported Bayes-power config schema")
        if not frozen:
            raise BayesPowerContractError("frozen Bayes-power design changed")

    @property
    def total_transition_count(self) -> int:
        return (
            2
            * 3
            * self.paths_per_role_per_law
            * len(self.selected_outer_steps)
            * PHASE_COUNT
            * EDGES_PER_PHASE
        )

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "total_transition_count": self.total_transition_count,
            "training_target": "exact_binary64_rao_blackwell_label",
            "oracle_role": "audit_only_never_model_input_or_training_target",
        }

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.to_record())

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any]
    ) -> "BayesPowerScientificConfig":
        body = dict(record)
        body.pop("total_transition_count", None)
        body.pop("training_target", None)
        body.pop("oracle_role", None)
        for name in ("selected_outer_steps", "phase_matchings", "phase_durations", "model_seeds"):
            if name in body:
                body[name] = tuple(body[name])
        result = cls(**body)
        if _semantic_sha256(result.to_record()) != _semantic_sha256(dict(record)):
            raise BayesPowerContractError("Bayes-power config record changed")
        return result


@dataclass(frozen=True)
class BayesPowerPathPlan:
    """Disjoint train/validation/confirmation roles for teacher and null."""

    teacher_train: tuple[int, ...] = TEACHER_TRAIN_PATH_IDS
    teacher_validation: tuple[int, ...] = TEACHER_VALIDATION_PATH_IDS
    teacher_confirmation: tuple[int, ...] = TEACHER_CONFIRMATION_PATH_IDS
    null_train: tuple[int, ...] = NULL_TRAIN_PATH_IDS
    null_validation: tuple[int, ...] = NULL_VALIDATION_PATH_IDS
    null_confirmation: tuple[int, ...] = NULL_CONFIRMATION_PATH_IDS
    version: str = BAYES_POWER_VERSION + "-path-id-v1"

    def __post_init__(self) -> None:
        roles = self.roles
        flattened: list[int] = []
        for name, raw_values in roles.items():
            values = tuple(validate_path_id(value) for value in raw_values)
            if len(values) != 8 or len(set(values)) != 8:
                raise BayesPowerContractError(
                    f"{name} must contain eight unique path IDs"
                )
            object.__setattr__(self, name, values)
            flattened.extend(values)
        if len(flattened) != len(set(flattened)):
            raise BayesPowerContractError("Bayes-power path roles overlap")

    @property
    def roles(self) -> dict[str, tuple[int, ...]]:
        return {
            "teacher_train": self.teacher_train,
            "teacher_validation": self.teacher_validation,
            "teacher_confirmation": self.teacher_confirmation,
            "null_train": self.null_train,
            "null_validation": self.null_validation,
            "null_confirmation": self.null_confirmation,
        }

    @property
    def all_path_ids(self) -> tuple[int, ...]:
        return tuple(value for role in self.roles.values() for value in role)

    def to_record(self) -> dict[str, Any]:
        body = {
            "schema": PATH_ID_PLAN_SCHEMA,
            "schema_version": 1,
            "version": self.version,
            "roles": {name: list(values) for name, values in self.roles.items()},
            "checks": {
                "integer_20_bit_pass": 1,
                "role_disjoint_pass": 1,
                "confirmation_sealed_until_selection": 1,
            },
        }
        return {**body, "path_id_plan_sha256": _semantic_sha256(body)}

    @property
    def sha256(self) -> str:
        return str(self.to_record()["path_id_plan_sha256"])

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "BayesPowerPathPlan":
        body = dict(record)
        claimed = body.pop("path_id_plan_sha256", None)
        if claimed != _semantic_sha256(body):
            raise BayesPowerContractError("Bayes-power path-plan hash mismatch")
        if (
            body.get("schema") != PATH_ID_PLAN_SCHEMA
            or body.get("schema_version") != 1
            or not isinstance(body.get("roles"), Mapping)
        ):
            raise BayesPowerContractError("Bayes-power path plan has wrong schema")
        roles = body["roles"]
        result = cls(
            **{
                name: tuple(roles.get(name, ()))
                for name in (
                    "teacher_train",
                    "teacher_validation",
                    "teacher_confirmation",
                    "null_train",
                    "null_validation",
                    "null_confirmation",
                )
            },
            version=str(body.get("version")),
        )
        if _semantic_sha256(result.to_record()) != _semantic_sha256(dict(record)):
            raise BayesPowerContractError("Bayes-power path plan changed")
        return result


def frozen_scientific_config() -> BayesPowerScientificConfig:
    return BayesPowerScientificConfig()


def frozen_path_plan() -> BayesPowerPathPlan:
    return BayesPowerPathPlan()


def expected_control_sample_count(path_count: Any) -> int:
    count = _index(path_count, "path_count")
    if count < 0:
        raise BayesPowerContractError("path_count must be nonnegative")
    return count * len(SELECTED_OUTER_STEPS) * PHASE_COUNT


def expected_control_transition_count(path_count: Any) -> int:
    return expected_control_sample_count(path_count) * EDGES_PER_PHASE


def _finite_unit_interval(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or ((array < 0.0) | (array > 1.0)).any():
        raise BayesPowerContractError(f"{name} must lie in [0,1]")
    return array


def _finite_exposure(value: Any) -> np.ndarray:
    exposure = np.asarray(value, dtype=np.float64)
    if not np.isfinite(exposure).all() or (exposure < 0.0).any():
        raise BayesPowerContractError("exposure must be finite and nonnegative")
    return exposure


def bounded_teacher_initial_density_ratio(head_fraction: Any) -> np.ndarray:
    """Return ``q_0(x)=1+0.5(2x-1)=x+0.5``."""

    x = _finite_unit_interval(head_fraction, "head_fraction")
    return 1.0 + 0.5 * (2.0 * x - 1.0)


def bounded_teacher_arrival_density_ratio(
    arrival_head_fraction: Any, exposure: Any
) -> np.ndarray:
    """Return ``q_u(y)=1+0.5 exp(-2u)(2y-1)``."""

    y = _finite_unit_interval(arrival_head_fraction, "arrival_head_fraction")
    u = _finite_exposure(exposure)
    y, u = np.broadcast_arrays(y, u)
    return 1.0 + 0.5 * np.exp(-2.0 * u) * (2.0 * y - 1.0)


def bounded_teacher_arrival_score(
    arrival_head_fraction: Any, exposure: Any
) -> np.ndarray:
    """Derivative in ``y`` of ``log(q_u(y))``."""

    y = _finite_unit_interval(arrival_head_fraction, "arrival_head_fraction")
    u = _finite_exposure(exposure)
    y, u = np.broadcast_arrays(y, u)
    decay = np.exp(-2.0 * u)
    return decay / bounded_teacher_arrival_density_ratio(y, u)


def bounded_teacher_oracle_mean(
    arrival_head_fraction: Any, exposure: Any
) -> np.ndarray:
    """Known Bayes mean of the exact noisy RB label."""

    y = _finite_unit_interval(arrival_head_fraction, "arrival_head_fraction")
    u = _finite_exposure(exposure)
    y, u = np.broadcast_arrays(y, u)
    return y * (1.0 - y) * bounded_teacher_arrival_score(y, u)


def null_oracle_mean(arrival_head_fraction: Any, exposure: Any) -> np.ndarray:
    y = _finite_unit_interval(arrival_head_fraction, "arrival_head_fraction")
    u = _finite_exposure(exposure)
    shape = np.broadcast_shapes(y.shape, u.shape)
    return np.zeros(shape, dtype=np.float64)


def oracle_conditional_mean(
    law: str | int, arrival_head_fraction: Any, exposure: Any
) -> np.ndarray:
    return (
        bounded_teacher_oracle_mean(arrival_head_fraction, exposure)
        if normalize_law(law) == TEACHER_LAW
        else null_oracle_mean(arrival_head_fraction, exposure)
    )


def sample_bounded_teacher_initial(
    rng: np.random.Generator, shape: int | Sequence[int]
) -> np.ndarray:
    """Sample the exact ``0.5*Uniform + 0.5*Beta(2,1)`` mixture.

    Three fixed-size random arrays are consumed regardless of mixture
    outcomes, which keeps replay independent of branch counts.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    selector = rng.random(shape)
    uniform = rng.random(shape)
    beta_two_one = np.sqrt(rng.random(shape))
    return np.ascontiguousarray(
        np.where(selector < 0.5, uniform, beta_two_one), dtype=np.float64
    )


def sample_stationary_null_initial(
    rng: np.random.Generator, shape: int | Sequence[int]
) -> np.ndarray:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    return np.ascontiguousarray(rng.random(shape), dtype=np.float64)


def sample_control_initial(
    law: str | int, rng: np.random.Generator, shape: int | Sequence[int]
) -> np.ndarray:
    return (
        sample_bounded_teacher_initial(rng, shape)
        if normalize_law(law) == TEACHER_LAW
        else sample_stationary_null_initial(rng, shape)
    )


def _matching_arrays() -> tuple[np.ndarray, np.ndarray]:
    tails, heads = matching_indices()
    return tails.cpu().numpy(), heads.cpu().numpy()


def extract_pair_mass_templates(
    later_full_state: Any, color: Any
) -> np.ndarray:
    """Extract pair totals for the selected perfect matching in each row."""

    state = np.asarray(later_full_state, dtype=np.float64)
    colors = np.asarray(color, dtype=np.int64)
    if state.ndim != 2 or state.shape[1] != STATE_SIZE:
        raise BayesPowerContractError("later_full_state must have shape [R,784]")
    if colors.shape != (state.shape[0],) or ((colors < 0) | (colors >= 4)).any():
        raise BayesPowerContractError("color must have shape [R] with values in [0,4)")
    if (
        not np.isfinite(state).all()
        or (state < 0.0).any()
        or not np.all(np.abs(state.sum(axis=1) - 1.0) <= 2.0e-12)
    ):
        raise BayesPowerContractError("template states must lie on the simplex")
    tails, heads = _matching_arrays()
    rows = np.arange(state.shape[0])[:, None]
    selected_tails = tails[colors]
    selected_heads = heads[colors]
    return np.ascontiguousarray(
        state[rows, selected_tails] + state[rows, selected_heads],
        dtype=np.float64,
    )


def construct_later_full_states(
    pair_mass_templates: Any,
    arrival_head_fraction: Any,
    color: Any,
) -> np.ndarray:
    """Construct full later states without changing any pair total.

    Each matching is perfect, so assigning ``r*(1-y)`` to its tail and
    ``r*y`` to its head fills every cell exactly once.
    """

    pair_mass = np.asarray(pair_mass_templates, dtype=np.float64)
    arrival = _finite_unit_interval(
        arrival_head_fraction, "arrival_head_fraction"
    )
    colors = np.asarray(color, dtype=np.int64)
    if (
        pair_mass.ndim != 2
        or pair_mass.shape[1] != EDGES_PER_PHASE
        or arrival.shape != pair_mass.shape
        or colors.shape != (pair_mass.shape[0],)
    ):
        raise BayesPowerContractError("pair masses, arrivals, or colors have wrong shape")
    if (
        not np.isfinite(pair_mass).all()
        or (pair_mass < 0.0).any()
        or not np.all(np.abs(pair_mass.sum(axis=1) - 1.0) <= 2.0e-12)
        or ((colors < 0) | (colors >= 4)).any()
    ):
        raise BayesPowerContractError("pair-mass templates must partition simplex mass")
    tails, heads = _matching_arrays()
    states = np.empty((pair_mass.shape[0], STATE_SIZE), dtype=np.float64)
    for row, active_color in enumerate(colors):
        states[row, tails[active_color]] = pair_mass[row] * (1.0 - arrival[row])
        states[row, heads[active_color]] = pair_mass[row] * arrival[row]
    if (
        not np.isfinite(states).all()
        or (states < 0.0).any()
        or not np.all(np.abs(states.sum(axis=1) - 1.0) <= 2.0e-12)
    ):
        raise BayesPowerContractError("constructed later states left the simplex")
    return np.ascontiguousarray(states)


@dataclass(frozen=True)
class BayesPowerLabelCache:
    """Noisy training labels and certificate data; contains no oracle."""

    sample_key: np.ndarray = field(repr=False, compare=False)
    path_id: np.ndarray = field(repr=False, compare=False)
    outer_step: np.ndarray = field(repr=False, compare=False)
    phase: np.ndarray = field(repr=False, compare=False)
    law: np.ndarray = field(repr=False, compare=False)
    denoising_target: np.ndarray = field(repr=False, compare=False)
    certificate_codes: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, dtype in _LABEL_DTYPES.items():
            object.__setattr__(
                self, name, _readonly_array(getattr(self, name), dtype)
            )
        _validate_label_cache(self)

    @property
    def sample_count(self) -> int:
        return int(self.sample_key.shape[0])

    @property
    def law_code(self) -> int:
        return int(self.law[0])

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in BAYES_LABEL_CACHE_FIELDS}

    def to_learnability_cache(self) -> LearnabilityLabelAuditCache:
        return LearnabilityLabelAuditCache(
            sample_key=self.sample_key,
            path_id=self.path_id,
            outer_step=self.outer_step,
            phase=self.phase,
            denoising_target=self.denoising_target,
            certificate_codes=self.certificate_codes,
        )


@dataclass(frozen=True)
class BayesPowerOracleAuditCache:
    """Analytic quantities physically excluded from labels and model inputs."""

    sample_key: np.ndarray = field(repr=False, compare=False)
    path_id: np.ndarray = field(repr=False, compare=False)
    outer_step: np.ndarray = field(repr=False, compare=False)
    phase: np.ndarray = field(repr=False, compare=False)
    law: np.ndarray = field(repr=False, compare=False)
    earlier_head_fraction: np.ndarray = field(repr=False, compare=False)
    arrival_head_fraction: np.ndarray = field(repr=False, compare=False)
    exposure: np.ndarray = field(repr=False, compare=False)
    oracle_conditional_mean: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name, dtype in _ORACLE_DTYPES.items():
            object.__setattr__(
                self, name, _readonly_array(getattr(self, name), dtype)
            )
        _validate_oracle_cache(self)

    @property
    def sample_count(self) -> int:
        return int(self.sample_key.shape[0])

    @property
    def law_code(self) -> int:
        return int(self.law[0])

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in BAYES_ORACLE_AUDIT_FIELDS}


@dataclass(frozen=True)
class BayesPowerCacheBundle:
    inputs: LearnabilityInputCache
    labels: BayesPowerLabelCache
    oracle_audit: BayesPowerOracleAuditCache

    def __post_init__(self) -> None:
        validate_control_cache_bundle(self)

    @property
    def sample_count(self) -> int:
        return self.inputs.sample_count

    @property
    def law_code(self) -> int:
        return self.labels.law_code

    def training_bundle(self) -> LearnabilityCacheBundle:
        """Return the only two cache objects visible to training."""

        return LearnabilityCacheBundle(
            self.inputs, self.labels.to_learnability_cache()
        )


def _validate_common_coordinates(
    sample_keys: np.ndarray,
    path_ids: np.ndarray,
    outer_steps: np.ndarray,
    phases: np.ndarray,
) -> None:
    n = int(sample_keys.shape[0])
    if any(value.shape != (n,) for value in (path_ids, outer_steps, phases)):
        raise BayesPowerContractError("cache coordinate columns have wrong shape")
    if len(np.unique(sample_keys)) != n:
        raise BayesPowerContractError("cache sample keys must be unique")
    if any(not 0 <= int(value) < (1 << 20) for value in path_ids):
        raise BayesPowerContractError("cache path ID exceeds 20 bits")
    if ((outer_steps < 0) | (outer_steps >= OUTER_STEPS)).any():
        raise BayesPowerContractError("cache outer step is outside [0,512)")
    if ((phases < 0) | (phases >= PHASE_COUNT)).any():
        raise BayesPowerContractError("cache phase is outside [0,7)")
    expected = np.fromiter(
        (
            sample_key(path_id, outer_step, phase)
            for path_id, outer_step, phase in zip(
                path_ids, outer_steps, phases, strict=True
            )
        ),
        dtype=np.int64,
        count=n,
    )
    if not np.array_equal(sample_keys, expected):
        raise BayesPowerContractError("sample keys do not encode cache coordinates")


def _validate_law_column(law: np.ndarray, n: int) -> None:
    if law.shape != (n,) or n == 0:
        raise BayesPowerContractError("law column must be nonempty and one-dimensional")
    unique = np.unique(law)
    if unique.size != 1 or int(unique[0]) not in LAW_NAMES:
        raise BayesPowerContractError("each cache must contain exactly one control law")


def _validate_label_cache(cache: BayesPowerLabelCache) -> None:
    n = int(cache.sample_key.shape[0])
    expected = {
        "sample_key": (n,),
        "path_id": (n,),
        "outer_step": (n,),
        "phase": (n,),
        "law": (n,),
        "denoising_target": (n, EDGES_PER_PHASE),
        "certificate_codes": (n, EDGES_PER_PHASE),
    }
    for name, shape in expected.items():
        value = getattr(cache, name)
        if value.shape != shape or value.dtype != _LABEL_DTYPES[name]:
            raise BayesPowerContractError(f"invalid label-cache {name}")
    _validate_common_coordinates(
        cache.sample_key, cache.path_id, cache.outer_step, cache.phase
    )
    _validate_law_column(cache.law, n)
    if not np.isfinite(cache.denoising_target).all():
        raise BayesPowerContractError("noisy denoising target contains nonfinite values")


def _validate_oracle_cache(cache: BayesPowerOracleAuditCache) -> None:
    n = int(cache.sample_key.shape[0])
    expected = {
        "sample_key": (n,),
        "path_id": (n,),
        "outer_step": (n,),
        "phase": (n,),
        "law": (n,),
        "earlier_head_fraction": (n, EDGES_PER_PHASE),
        "arrival_head_fraction": (n, EDGES_PER_PHASE),
        "exposure": (n, EDGES_PER_PHASE),
        "oracle_conditional_mean": (n, EDGES_PER_PHASE),
    }
    for name, shape in expected.items():
        value = getattr(cache, name)
        if value.shape != shape or value.dtype != _ORACLE_DTYPES[name]:
            raise BayesPowerContractError(f"invalid oracle-cache {name}")
    _validate_common_coordinates(
        cache.sample_key, cache.path_id, cache.outer_step, cache.phase
    )
    _validate_law_column(cache.law, n)
    _finite_unit_interval(cache.earlier_head_fraction, "earlier_head_fraction")
    _finite_unit_interval(cache.arrival_head_fraction, "arrival_head_fraction")
    _finite_exposure(cache.exposure)
    if not np.isfinite(cache.oracle_conditional_mean).all():
        raise BayesPowerContractError("oracle mean contains nonfinite values")
    expected_oracle = oracle_conditional_mean(
        int(cache.law[0]), cache.arrival_head_fraction, cache.exposure
    )
    if not np.array_equal(cache.oracle_conditional_mean, expected_oracle):
        raise BayesPowerContractError("oracle mean does not match the analytic law")


def validate_control_cache_bundle(
    bundle: BayesPowerCacheBundle,
    *,
    expected_path_ids: Sequence[int] | None = None,
    expected_outer_steps: Sequence[int] | None = None,
) -> None:
    inputs, labels, oracle = bundle.inputs, bundle.labels, bundle.oracle_audit
    if not isinstance(inputs, LearnabilityInputCache):
        raise TypeError("inputs must be an exact LearnabilityInputCache")
    if inputs.sample_count != labels.sample_count or labels.sample_count != oracle.sample_count:
        raise BayesPowerContractError("input, label, and oracle row counts differ")
    for name in ("sample_key", "path_id", "outer_step", "phase", "law"):
        if name == "sample_key":
            left = inputs.sample_key
        else:
            left = getattr(labels, name)
        right = getattr(oracle, name)
        if not np.array_equal(left, right):
            raise BayesPowerContractError(f"cache join differs in {name}")
    if not np.array_equal(inputs.phase, labels.phase):
        raise BayesPowerContractError("input and label phases differ")
    expected_time = np.fromiter(
        (
            selected_reverse_time(step, phase)
            for step, phase in zip(
                labels.outer_step, labels.phase, strict=True
            )
        ),
        dtype=np.float64,
        count=labels.sample_count,
    )
    if not np.array_equal(inputs.reverse_time, expected_time):
        raise BayesPowerContractError("reverse time differs from exact phase index")
    if expected_path_ids is not None:
        expected = tuple(sorted(validate_path_id(value) for value in expected_path_ids))
        observed = tuple(sorted(np.unique(labels.path_id).tolist()))
        if observed != expected:
            raise BayesPowerContractError("cache path role differs from its plan")
    if expected_outer_steps is not None:
        expected_steps = tuple(sorted(_index(value, "outer_step") for value in expected_outer_steps))
        if tuple(sorted(np.unique(labels.outer_step).tolist())) != expected_steps:
            raise BayesPowerContractError("cache outer steps differ from their plan")


def build_control_cache_bundle(
    *,
    path_id: Any,
    outer_step: Any,
    phase: Any,
    pair_mass_templates: Any,
    earlier_head_fraction: Any | None = None,
    arrival_head_fraction: Any,
    exposure: Any,
    denoising_target: Any,
    certificate_codes: Any,
    law: str | int,
    label: int = 3,
) -> BayesPowerCacheBundle:
    """Build joined caches while keeping oracle data out of training."""

    paths = np.asarray(path_id, dtype=np.int64)
    steps = np.asarray(outer_step, dtype=np.int16)
    phases = np.asarray(phase, dtype=np.int8)
    if paths.ndim != 1 or steps.shape != paths.shape or phases.shape != paths.shape:
        raise BayesPowerContractError("row coordinates must be equal one-dimensional arrays")
    n = paths.size
    colors = np.asarray(PHASE_MATCHINGS, dtype=np.int8)[phases]
    durations = np.asarray(PHASE_DURATIONS, dtype=np.float64)[phases]
    pair_mass = np.asarray(pair_mass_templates, dtype=np.float64)
    earlier = (
        np.asarray(arrival_head_fraction, dtype=np.float64)
        if earlier_head_fraction is None
        else np.asarray(earlier_head_fraction, dtype=np.float64)
    )
    arrival = np.asarray(arrival_head_fraction, dtype=np.float64)
    exposures = np.asarray(exposure, dtype=np.float64)
    targets = np.asarray(denoising_target, dtype=np.float64)
    certificates = np.asarray(certificate_codes)
    expected_edge_shape = (n, EDGES_PER_PHASE)
    if any(
        value.shape != expected_edge_shape
        for value in (pair_mass, earlier, arrival, exposures, targets, certificates)
    ):
        raise BayesPowerContractError("transition arrays have wrong shape")
    states = construct_later_full_states(pair_mass, arrival, colors)
    keys = np.fromiter(
        (
            sample_key(path_value, step_value, phase_value)
            for path_value, step_value, phase_value in zip(
                paths, steps, phases, strict=True
            )
        ),
        dtype=np.int64,
        count=n,
    )
    reverse_time = np.fromiter(
        (
            selected_reverse_time(step_value, phase_value)
            for step_value, phase_value in zip(steps, phases, strict=True)
        ),
        dtype=np.float64,
        count=n,
    )
    law_code = normalize_law(law)
    law_column = np.full(n, law_code, dtype=np.int8)
    inputs = LearnabilityInputCache(
        sample_key=keys,
        later_full_state=states,
        reverse_time=reverse_time,
        phase=phases,
        color=colors,
        duration=durations,
        label=np.full(n, _index(label, "label"), dtype=np.int64),
    )
    labels = BayesPowerLabelCache(
        sample_key=keys,
        path_id=paths,
        outer_step=steps,
        phase=phases,
        law=law_column,
        denoising_target=targets,
        certificate_codes=np.asarray(certificates, dtype=np.uint8),
    )
    oracle = BayesPowerOracleAuditCache(
        sample_key=keys,
        path_id=paths,
        outer_step=steps,
        phase=phases,
        law=law_column,
        earlier_head_fraction=earlier,
        arrival_head_fraction=arrival,
        exposure=exposures,
        oracle_conditional_mean=oracle_conditional_mean(
            law_code, arrival, exposures
        ),
    )
    return BayesPowerCacheBundle(inputs, labels, oracle)


@dataclass(frozen=True)
class ControlTransitionBatch:
    """Host representation returned by an injected exact transition sampler."""

    later_head_fraction: np.ndarray = field(repr=False, compare=False)
    denoising_target: np.ndarray = field(repr=False, compare=False)
    certificate_codes: np.ndarray = field(repr=False, compare=False)
    diagnostics: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        later = np.ascontiguousarray(
            np.asarray(self.later_head_fraction, dtype=np.float64).reshape(-1)
        )
        target = np.ascontiguousarray(
            np.asarray(self.denoising_target, dtype=np.float64).reshape(-1)
        )
        raw_codes = np.asarray(self.certificate_codes).reshape(-1)
        if (
            later.shape != target.shape
            or raw_codes.shape != later.shape
            or not np.isfinite(later).all()
            or not np.isfinite(target).all()
            or ((later < 0.0) | (later > 1.0)).any()
            or (raw_codes < 0).any()
            or (raw_codes > np.iinfo(np.uint8).max).any()
        ):
            raise BayesPowerContractError("exact transition result is invalid")
        codes = np.ascontiguousarray(raw_codes, dtype=np.uint8)
        later.setflags(write=False)
        target.setflags(write=False)
        codes.setflags(write=False)
        object.__setattr__(self, "later_head_fraction", later)
        object.__setattr__(self, "denoising_target", target)
        object.__setattr__(self, "certificate_codes", codes)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@dataclass(frozen=True)
class ControlRoleGeneration:
    bundle: BayesPowerCacheBundle
    diagnostics: Mapping[str, int | float]


def canonical_control_transition_ids(
    path_id: int, outer_step: int, phase: int
) -> np.ndarray:
    """Return the unchanged 20/10/3/10-bit path-major edge IDs."""

    path = validate_path_id(path_id)
    step = _index(outer_step, "outer_step")
    phase_index = _index(phase, "phase")
    if not 0 <= step < (1 << 10) or not 0 <= phase_index < (1 << 3):
        raise BayesPowerContractError("transition coordinate exceeds its field")
    base = (path << 23) | (step << 13) | (phase_index << 10)
    return np.ascontiguousarray(
        np.arange(EDGES_PER_PHASE, dtype=np.uint64) + np.uint64(base)
    )


def decode_input_sample_keys(
    input_cache: LearnabilityInputCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.asarray(input_cache.sample_key, dtype=np.int64)
    paths = np.right_shift(keys, 13).astype(np.int64, copy=False)
    steps = np.bitwise_and(np.right_shift(keys, 3), (1 << 10) - 1).astype(
        np.int16, copy=False
    )
    phases = np.bitwise_and(keys, (1 << 3) - 1).astype(np.int8, copy=False)
    if not np.array_equal(phases, input_cache.phase):
        raise BayesPowerContractError("template sample keys disagree with phases")
    return (
        np.ascontiguousarray(paths),
        np.ascontiguousarray(steps),
        np.ascontiguousarray(phases),
    )


def _row_initial_rng(
    *, root_seed: int, law: int, path_id: int, outer_step: int, phase: int
) -> np.random.Generator:
    return np.random.Generator(
        np.random.Philox(
            [
                int(root_seed),
                int(law),
                int(path_id),
                int(outer_step),
                int(phase),
                0x42505952,
            ]
        )
    )


def coerce_control_transition_batch(value: Any) -> ControlTransitionBatch:
    if isinstance(value, ControlTransitionBatch):
        return value

    def array(name: str) -> np.ndarray:
        item = getattr(value, name)
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        return np.asarray(item)

    diagnostics = getattr(value, "diagnostics", {})
    scalars: dict[str, int | float] = {}
    if isinstance(diagnostics, Mapping):
        for name, item in diagnostics.items():
            if hasattr(item, "detach") and item.numel() == 1:
                item = item.detach().cpu().item()
            if isinstance(item, (int, float, np.integer, np.floating)):
                scalars[str(name)] = (
                    int(item)
                    if isinstance(item, (int, np.integer))
                    else float(item)
                )
    return ControlTransitionBatch(
        later_head_fraction=array("later_head_fraction"),
        denoising_target=array("denoising_target"),
        certificate_codes=array("certificate_codes"),
        diagnostics=scalars,
    )


def sample_control_transitions_cuda(
    earlier_head_fraction: np.ndarray,
    exposure: np.ndarray,
    *,
    rng_key: Any,
    transition_ids: np.ndarray,
    device: str = "cuda",
    profile: Any | None = None,
) -> ControlTransitionBatch:
    """Adapter from host role generation to the unchanged certified CUDA API."""

    import torch
    from mnist.d0_jacobi_rb_cuda import (
        JacobiRBCudaProfile,
        sample_alpha1_rb_transition_batch_cuda,
    )

    active_profile = JacobiRBCudaProfile() if profile is None else profile
    x = torch.as_tensor(
        np.array(earlier_head_fraction, dtype=np.float64, copy=True, order="C"),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    u = torch.as_tensor(
        np.array(exposure, dtype=np.float64, copy=True, order="C"),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    ids = torch.as_tensor(
        np.array(transition_ids, dtype=np.uint64, copy=True, order="C"),
        dtype=torch.uint64,
        device=device,
    ).contiguous()
    return coerce_control_transition_batch(
        sample_alpha1_rb_transition_batch_cuda(
            x,
            u,
            rng_key=rng_key,
            transition_ids=ids,
            profile=active_profile,
        )
    )


def generate_control_role_cache(
    template_inputs: LearnabilityInputCache,
    *,
    target_path_ids: Sequence[int],
    law: str | int,
    sampler: Callable[..., Any],
    root_seed: int = ROOT_SEED,
    tau_eff: float = 5.0e-5,
    maximum_rows_per_call: int = 10,
) -> ControlRoleGeneration:
    """Generate one complete role from immutable pair-mass/time templates.

    ``sampler`` receives flat NumPy ``earlier_head_fraction``, ``exposure``,
    ``transition_ids``, and ``rng_key`` arguments.  Production passes
    :func:`sample_control_transitions_cuda`; tests may inject a deterministic
    certified sampler with the same contract.
    """

    if not callable(sampler):
        raise TypeError("sampler must be callable")
    if maximum_rows_per_call <= 0 or maximum_rows_per_call * EDGES_PER_PHASE > 4096:
        raise BayesPowerContractError("sampler calls must contain 1..4096 lanes")
    law_code = normalize_law(law)
    targets = tuple(validate_path_id(value) for value in target_path_ids)
    if not targets or len(targets) != len(set(targets)):
        raise BayesPowerContractError("target path role must be nonempty and unique")
    source_paths, source_steps, source_phases = decode_input_sample_keys(template_inputs)
    source_unique = tuple(sorted(np.unique(source_paths).tolist()))
    if len(source_unique) != len(targets):
        raise BayesPowerContractError("template and target path counts differ")
    source_to_target = dict(zip(source_unique, targets, strict=True))
    order = np.lexsort((source_phases, source_steps, source_paths))
    source_paths = source_paths[order]
    steps = source_steps[order]
    phases = source_phases[order]
    colors = np.asarray(template_inputs.color, dtype=np.int8)[order]
    durations = np.asarray(template_inputs.duration, dtype=np.float64)[order]
    template_states = np.asarray(template_inputs.later_full_state, dtype=np.float64)[
        order
    ]
    mapped_paths = np.asarray(
        [source_to_target[int(value)] for value in source_paths], dtype=np.int64
    )
    if tuple(sorted(np.unique(mapped_paths).tolist())) != tuple(sorted(targets)):
        raise BayesPowerContractError("template remapping lost a target path")
    pair_mass = extract_pair_mass_templates(template_states, colors)
    n = mapped_paths.size
    earlier = np.empty((n, EDGES_PER_PHASE), dtype=np.float64)
    exposure = np.zeros_like(earlier)
    transition_ids = np.empty((n, EDGES_PER_PHASE), dtype=np.uint64)
    coefficient = 3.0 * (float(tau_eff) / OUTER_STEPS) / ((1.0 / GRID_SIZE) ** 2)
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise BayesPowerContractError("tau_eff must be finite and positive")
    for row in range(n):
        generator = _row_initial_rng(
            root_seed=int(root_seed),
            law=law_code,
            path_id=int(mapped_paths[row]),
            outer_step=int(steps[row]),
            phase=int(phases[row]),
        )
        earlier[row] = sample_control_initial(
            law_code, generator, EDGES_PER_PHASE
        )
        positive = pair_mass[row] > 0.0
        exposure[row, positive] = (
            coefficient * durations[row] / pair_mass[row, positive]
        )
        transition_ids[row] = canonical_control_transition_ids(
            int(mapped_paths[row]), int(steps[row]), int(phases[row])
        )
    later = np.empty_like(earlier)
    labels = np.empty_like(earlier)
    codes = np.empty_like(earlier, dtype=np.uint8)
    aggregate: dict[str, int | float] = {}
    for start in range(0, n, maximum_rows_per_call):
        stop = min(n, start + maximum_rows_per_call)
        result = coerce_control_transition_batch(
            sampler(
                earlier[start:stop].reshape(-1),
                exposure[start:stop].reshape(-1),
                rng_key=(
                    BAYES_POWER_VERSION,
                    int(root_seed),
                    LAW_NAMES[law_code],
                ),
                transition_ids=transition_ids[start:stop].reshape(-1),
            )
        )
        lane_count = (stop - start) * EDGES_PER_PHASE
        if result.later_head_fraction.size != lane_count:
            raise BayesPowerContractError("sampler returned the wrong lane count")
        later[start:stop] = result.later_head_fraction.reshape(
            stop - start, EDGES_PER_PHASE
        )
        labels[start:stop] = result.denoising_target.reshape(
            stop - start, EDGES_PER_PHASE
        )
        codes[start:stop] = result.certificate_codes.reshape(
            stop - start, EDGES_PER_PHASE
        )
        for name, value in result.diagnostics.items():
            aggregate[name] = aggregate.get(name, 0) + value
    active = exposure > 0.0
    certified = np.bitwise_and(codes, np.uint8(0x0F)) == np.uint8(0x0F)
    if not np.all(certified[active]):
        raise BayesPowerContractError("control cache contains uncertified transitions")
    bundle = build_control_cache_bundle(
        path_id=mapped_paths,
        outer_step=steps,
        phase=phases,
        pair_mass_templates=pair_mass,
        earlier_head_fraction=earlier,
        arrival_head_fraction=later,
        exposure=exposure,
        denoising_target=labels,
        certificate_codes=codes,
        law=law_code,
    )
    active_count = int(active.sum())
    diagnostics: dict[str, int | float] = {
        **aggregate,
        "sample_count": int(n),
        "transition_count": int(n * EDGES_PER_PHASE),
        "active_transition_count": active_count,
        "certified_transition_count": int(certified[active].sum()),
        "certificate_fraction": (
            float(certified[active].mean()) if active_count else 1.0
        ),
        "nonfinite_count": 0,
    }
    return ControlRoleGeneration(bundle=bundle, diagnostics=diagnostics)


def _atomic_save_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(
                handle,
                **{
                    name: np.ascontiguousarray(value)
                    for name, value in arrays.items()
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_bayes_label_cache(path: str | Path, cache: BayesPowerLabelCache) -> None:
    _validate_label_cache(cache)
    _atomic_save_npz(path, cache.arrays())


def save_bayes_oracle_audit_cache(
    path: str | Path, cache: BayesPowerOracleAuditCache
) -> None:
    _validate_oracle_cache(cache)
    _atomic_save_npz(path, cache.arrays())


def _strict_load_npz(
    path: str | Path,
    *,
    fields: Sequence[str],
    dtypes: Mapping[str, np.dtype[Any]],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            raise BayesPowerContractError("cache has unexpected or missing fields")
        result = {}
        for name in fields:
            value = np.asarray(archive[name])
            if value.dtype != dtypes[name]:
                raise BayesPowerContractError(f"cache field {name} changed dtype")
            result[name] = value
    return result


def load_bayes_label_cache(path: str | Path) -> BayesPowerLabelCache:
    return BayesPowerLabelCache(
        **_strict_load_npz(
            path, fields=BAYES_LABEL_CACHE_FIELDS, dtypes=_LABEL_DTYPES
        )
    )


def load_bayes_oracle_audit_cache(
    path: str | Path,
) -> BayesPowerOracleAuditCache:
    return BayesPowerOracleAuditCache(
        **_strict_load_npz(
            path, fields=BAYES_ORACLE_AUDIT_FIELDS, dtypes=_ORACLE_DTYPES
        )
    )


def save_control_cache_bundle(
    input_path: str | Path,
    label_path: str | Path,
    oracle_path: str | Path,
    bundle: BayesPowerCacheBundle,
) -> None:
    validate_control_cache_bundle(bundle)
    save_input_cache(input_path, bundle.inputs)
    save_bayes_label_cache(label_path, bundle.labels)
    save_bayes_oracle_audit_cache(oracle_path, bundle.oracle_audit)


def load_control_cache_bundle(
    input_path: str | Path,
    label_path: str | Path,
    oracle_path: str | Path,
    *,
    expected_path_ids: Sequence[int] | None = None,
    expected_outer_steps: Sequence[int] | None = None,
) -> BayesPowerCacheBundle:
    bundle = BayesPowerCacheBundle(
        inputs=load_input_cache(input_path),
        labels=load_bayes_label_cache(label_path),
        oracle_audit=load_bayes_oracle_audit_cache(oracle_path),
    )
    validate_control_cache_bundle(
        bundle,
        expected_path_ids=expected_path_ids,
        expected_outer_steps=expected_outer_steps,
    )
    return bundle


def tower_witness_products(
    noisy_label: Any,
    oracle_mean: Any,
    arrival_head_fraction: Any,
) -> np.ndarray:
    """Return fixed audit products ``(Z-m)*[1,z,P2(z),m]``."""

    label = np.asarray(noisy_label, dtype=np.float64)
    oracle = np.asarray(oracle_mean, dtype=np.float64)
    arrival = _finite_unit_interval(
        arrival_head_fraction, "arrival_head_fraction"
    )
    if label.shape != oracle.shape or label.shape != arrival.shape:
        raise BayesPowerContractError("tower arrays must have equal shapes")
    if label.ndim != 2 or label.shape[1] != EDGES_PER_PHASE:
        raise BayesPowerContractError("tower arrays must have shape [R,392]")
    if not np.isfinite(label).all() or not np.isfinite(oracle).all():
        raise BayesPowerContractError("tower arrays contain nonfinite values")
    z = 2.0 * arrival - 1.0
    p2 = 0.5 * (3.0 * z * z - 1.0)
    witnesses = np.stack(
        [np.ones_like(z), z, p2, oracle], axis=-1
    )
    return np.ascontiguousarray((label - oracle)[..., None] * witnesses)


@dataclass(frozen=True)
class PathOracleRisk:
    path_id: int
    model_mse: float
    metadata_mse: float
    zero_mse: float
    oracle_mse: float

    @property
    def model_beats_metadata(self) -> bool:
        return self.model_mse < self.metadata_mse

    @property
    def oracle_beats_zero(self) -> bool:
        return self.oracle_mse < self.zero_mse


@dataclass(frozen=True)
class OracleMetricSummary:
    model_mse: float
    metadata_mse: float
    zero_mse: float
    oracle_mse: float
    model_relative_gain_over_zero: float
    oracle_relative_gain_over_zero: float
    oracle_gain_recovery: float
    path_risks: tuple[PathOracleRisk, ...]

    @property
    def model_beats_zero(self) -> bool:
        return self.model_mse < self.zero_mse

    @property
    def model_beats_metadata_all_paths(self) -> bool:
        return bool(self.path_risks) and all(
            path.model_beats_metadata for path in self.path_risks
        )

    @property
    def oracle_beats_zero_all_paths(self) -> bool:
        return bool(self.path_risks) and all(
            path.oracle_beats_zero for path in self.path_risks
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "model_mse": self.model_mse,
            "metadata_mse": self.metadata_mse,
            "zero_mse": self.zero_mse,
            "oracle_mse": self.oracle_mse,
            "model_relative_gain_over_zero": self.model_relative_gain_over_zero,
            "oracle_relative_gain_over_zero": self.oracle_relative_gain_over_zero,
            "oracle_gain_recovery": self.oracle_gain_recovery,
            "model_beats_zero": int(self.model_beats_zero),
            "model_beats_metadata_all_paths": int(
                self.model_beats_metadata_all_paths
            ),
            "oracle_beats_zero_all_paths": int(
                self.oracle_beats_zero_all_paths
            ),
            "path_risks": [asdict(path) for path in self.path_risks],
        }


def _relative_gain(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        return 0.0 if candidate == reference else -math.inf
    return 1.0 - candidate / reference


def oracle_metric_summary(
    model_prediction: Any,
    noisy_target: Any,
    oracle_mean: Any,
    metadata_prediction: Any,
    path_id: Any,
) -> OracleMetricSummary:
    model = np.asarray(model_prediction, dtype=np.float64)
    target = np.asarray(noisy_target, dtype=np.float64)
    oracle = np.asarray(oracle_mean, dtype=np.float64)
    metadata = np.asarray(metadata_prediction, dtype=np.float64)
    paths = np.asarray(path_id, dtype=np.int64)
    if (
        model.shape != target.shape
        or oracle.shape != target.shape
        or metadata.shape != target.shape
        or target.ndim != 2
        or target.shape[1] != EDGES_PER_PHASE
        or paths.shape != (target.shape[0],)
        or target.size == 0
    ):
        raise BayesPowerContractError("oracle metric arrays have incompatible shapes")
    if not all(np.isfinite(value).all() for value in (model, target, oracle, metadata)):
        raise BayesPowerContractError("oracle metric arrays contain nonfinite values")
    zero = np.zeros_like(target)
    model_mse = stable_mse(model, target)
    metadata_mse = stable_mse(metadata, target)
    zero_mse = stable_mse(zero, target)
    oracle_mse = stable_mse(oracle, target)
    denominator = zero_mse - oracle_mse
    recovery = (
        (zero_mse - model_mse) / denominator
        if denominator > 0.0
        else -math.inf
    )
    path_risks = []
    for path in sorted(np.unique(paths).tolist()):
        mask = paths == path
        path_risks.append(
            PathOracleRisk(
                path_id=int(path),
                model_mse=stable_mse(model[mask], target[mask]),
                metadata_mse=stable_mse(metadata[mask], target[mask]),
                zero_mse=stable_mse(zero[mask], target[mask]),
                oracle_mse=stable_mse(oracle[mask], target[mask]),
            )
        )
    return OracleMetricSummary(
        model_mse=model_mse,
        metadata_mse=metadata_mse,
        zero_mse=zero_mse,
        oracle_mse=oracle_mse,
        model_relative_gain_over_zero=_relative_gain(zero_mse, model_mse),
        oracle_relative_gain_over_zero=_relative_gain(zero_mse, oracle_mse),
        oracle_gain_recovery=float(recovery),
        path_risks=tuple(path_risks),
    )


def teacher_confirmation_pass(
    summary: OracleMetricSummary,
    *,
    expected_path_count: int = 8,
    minimum_oracle_relative_gain: float = 0.01,
    minimum_oracle_gain_recovery: float = 0.50,
) -> bool:
    return bool(
        len(summary.path_risks) == expected_path_count
        and summary.oracle_beats_zero_all_paths
        and summary.oracle_relative_gain_over_zero >= minimum_oracle_relative_gain
        and summary.model_beats_zero
        and summary.model_beats_metadata_all_paths
        and summary.oracle_gain_recovery >= minimum_oracle_gain_recovery
    )


def null_discovery_signal(summary: OracleMetricSummary, *, expected_path_count: int = 8) -> bool:
    """The same aggregate-plus-eight-sign conjunction used for discovery."""

    return bool(
        len(summary.path_risks) == expected_path_count
        and summary.model_beats_zero
        and summary.model_beats_metadata_all_paths
    )


# Public aliases make reuse by the orchestration layer explicit while leaving
# every parent implementation byte-identical.
BayesPowerPredictor = JacobiRBPhasePredictor
BayesPowerTrainingPlan = TrainingPlan
frozen_bayes_training_plan = frozen_training_plan
bayes_model_inputs_from_cache = model_inputs_from_cache
bayes_audit_targets_from_cache = audit_targets_from_cache
train_bayes_regressor = train_deterministic_regressor
fit_bayes_metadata_baseline = fit_metadata_baseline
