"""Pure core for the exact-K=512 coarse-baseline residual learner.

The module deliberately contains no cache runner, optimizer orchestration, or
sampler.  It provides four small, auditable pieces:

* derivation and integrity-checked loading of the frozen witness baseline;
* a predictor which adds that baseline to the unchanged Jacobi/RB model;
* exact-target MSE and whole-path risk contrasts; and
* deterministic one-sided studentized max-T inference.

The denoising target is never changed.  Writing the predictor as ``b + r`` is
only a parameterization of the same unweighted squared-error problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    MODEL_INPUT_FIELDS,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    SELECTED_OUTER_STEPS,
    JacobiRBPhasePredictor,
    ModelInputs,
    call_model,
    semantic_sha256,
    stable_sum,
)


COARSE_RESIDUAL_VERSION = "d0-jacobi-rb-coarse-residual-v1"
BASELINE_SCHEMA = COARSE_RESIDUAL_VERSION + "-frozen-baseline-v1"
BASELINE_FILE_SCHEMA = COARSE_RESIDUAL_VERSION + "-baseline-file-v1"
PATH_CONTRAST_SCHEMA = COARSE_RESIDUAL_VERSION + "-path-contrasts-v1"
MAX_T_SCHEMA = COARSE_RESIDUAL_VERSION + "-studentized-max-t-v1"

TIME_QUARTILES = 4
WITNESS_PATHS_PER_PANEL = 64
WITNESS_CELL_SHAPE = (TIME_QUARTILES, PHASE_COUNT, EDGES_PER_PHASE)
WITNESS_PANEL_SHAPE = (WITNESS_PATHS_PER_PANEL, *WITNESS_CELL_SHAPE)
WITNESS_REGISTRY_SHA256 = (
    "ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3"
)
WITNESS_REGISTRY_RECORD_COUNT = 2_616
WITNESS_PATH_PLAN_SHA256 = (
    "76f44f7c83f5f294ebda5d55f610c0942e7f16f1ab2ea2940ae70a5f2b059a65"
)
WITNESS_STATISTIC_PLAN_SHA256 = (
    "91cee2ce9eb5a1688dcfc72ba2a27c02c53a73b88e29807cc556aff75c403ec0"
)
WITNESS_PANEL_A_FILE_SHA256 = (
    "70d374526df5c02e5c6ab7f9b17205de373b22c694480bb27bf5684b4a579852"
)
WITNESS_PANEL_B_FILE_SHA256 = (
    "d64688f026cc510d586fb6b20e2303fdbe407a99b1a161b4654dc5dd04face81"
)
WITNESS_PANEL_A_ARRAY_SHA256 = (
    "1fe04953fd50ea3cb0ac163efed216ec5ebbafc58f48ce0de3f77d090c29fe08"
)
WITNESS_PANEL_B_ARRAY_SHA256 = (
    "2d949662c098783aa663672528f107a9f73f503529440aca4313cf770cad737e"
)
WITNESS_PANEL_A_PATH_IDS = tuple(range(0xE5000, 0xE5040))
WITNESS_PANEL_B_PATH_IDS = tuple(range(0xE5100, 0xE5140))

# These values are consequences of the two immutable witness panels, not
# tunable hyperparameters.
WITNESS_SIGNAL_ENERGY = 0.0006484248701021389
WITNESS_PANEL_MEAN_NOISE = 0.00315904482822984
WITNESS_AVERAGED_TABLE_NOISE = 0.00157952241411492
GLOBAL_SHRINKAGE = 0.2910413880506186
WITNESS_BASELINE_ENERGY = 0.00018871847424106853
WITNESS_RAW_VALUES_SHA256 = (
    "cb66524aad30ef3a6c442e007ac0afff2e6ae745fcff07c8c489ce3fc8a941d6"
)
WITNESS_VALUES_SHA256 = (
    "5d4e73153c36a59e26403439befd4e13b7f4fe096f7cbf9af6b77ac26565a9df"
)
# Secondary, shape/dtype-aware serialization hashes.  The public scientific
# binding above is deliberately the raw float64 C-order values hash requested
# by the frozen plan.
WITNESS_RAW_VALUES_SERIALIZATION_SHA256 = (
    "52e1e938d47ddd2d6f2210bfa6b726b69467fb17a61405ef2b772d8f9677c24a"
)
WITNESS_VALUES_SERIALIZATION_SHA256 = (
    "ff63431e776ea429667eff8de042b8308ac294aafa55872f6e7c3e4532606b23"
)

DEFAULT_MAX_T_SEED = 261_255
DEFAULT_MAX_T_REPLICATES = 50_000
DEFAULT_MAX_T_CONFIDENCE = 0.99
PRIMARY_CONTRAST_NAMES = (
    "overall.baseline_vs_zero",
    "overall.combined_vs_baseline",
)
ALL_CONTRAST_NAMES = (
    "overall.baseline_vs_zero",
    "overall.combined_vs_baseline",
    "overall.combined_vs_zero",
    "data_end.baseline_vs_zero",
    "data_end.combined_vs_baseline",
    "data_end.combined_vs_zero",
)
_MAX_T_NAMESPACE = 0x43524D54


class CoarseResidualContractError(ValueError):
    """Raised when a coarse-residual scientific contract is violated."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    """Secondary shape/dtype-aware serialization hash."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _witness_seal_array_sha256(value: np.ndarray) -> str:
    """Hash convention used by the immutable witness panel seals."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _float64_c_order_sha256(value: np.ndarray) -> str:
    """SHA-256 of exactly the binary64 C-order values, with no metadata."""

    array = np.asarray(value)
    if array.dtype != np.float64:
        raise CoarseResidualContractError(
            "C-order scientific hash requires binary64 values"
        )
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any],
    shape: tuple[int, ...] | None = None,
    name: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != dtype:
        raise CoarseResidualContractError(
            f"{name} must have dtype {dtype.str}; got {raw.dtype.str}"
        )
    if shape is not None and raw.shape != shape:
        raise CoarseResidualContractError(
            f"{name} must have shape {shape}; got {raw.shape}"
        )
    if raw.size == 0:
        raise CoarseResidualContractError(f"{name} must be nonempty")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(raw).all():
        raise CoarseResidualContractError(f"{name} contains nonfinite values")
    result = np.ascontiguousarray(raw)
    result.setflags(write=False)
    return result


def _canonical_path_ids(
    value: Any, *, name: str, expected_count: int | None = None
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise CoarseResidualContractError(f"{name} must be a one-dimensional integer array")
    result = np.asarray(raw, dtype=np.int64)
    if expected_count is not None and result.size != int(expected_count):
        raise CoarseResidualContractError(
            f"{name} must contain exactly {expected_count} paths"
        )
    if (
        result.size == 0
        or ((result < 0) | (result >= (1 << 20))).any()
        or np.unique(result).size != result.size
        or not np.array_equal(result, np.sort(result))
    ):
        raise CoarseResidualContractError(
            f"{name} must be unique, sorted, and inside the 20-bit field"
        )
    return _readonly_array(
        np.ascontiguousarray(result),
        dtype=np.dtype(np.int64),
        shape=(result.size,),
        name=name,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoarseResidualContractError(f"cannot read JSON artifact {path}") from exc
    if not isinstance(value, dict):
        raise CoarseResidualContractError(f"JSON artifact {path} is not an object")
    return value


@dataclass(frozen=True)
class FrozenCoarseBaseline:
    """Immutable globally-shrunk coarse predictor from the witness panels."""

    raw_values: np.ndarray = field(repr=False, compare=False)
    values: np.ndarray = field(repr=False, compare=False)
    left_path_ids: np.ndarray = field(repr=False, compare=False)
    right_path_ids: np.ndarray = field(repr=False, compare=False)
    shrinkage: float
    signal_energy: float
    panel_mean_noise: float
    averaged_table_noise: float
    left_cell_means_file_sha256: str
    right_cell_means_file_sha256: str
    left_cell_means_array_sha256: str
    right_cell_means_array_sha256: str
    witness_registry_sha256: str
    schema: str = BASELINE_SCHEMA

    def __post_init__(self) -> None:
        raw_values = _readonly_array(
            self.raw_values,
            dtype=np.dtype(np.float64),
            shape=WITNESS_CELL_SHAPE,
            name="raw_values",
        )
        values = _readonly_array(
            self.values,
            dtype=np.dtype(np.float64),
            shape=WITNESS_CELL_SHAPE,
            name="values",
        )
        left_paths = _canonical_path_ids(
            self.left_path_ids,
            name="left_path_ids",
            expected_count=WITNESS_PATHS_PER_PANEL,
        )
        right_paths = _canonical_path_ids(
            self.right_path_ids,
            name="right_path_ids",
            expected_count=WITNESS_PATHS_PER_PANEL,
        )
        if set(left_paths.tolist()).intersection(right_paths.tolist()):
            raise CoarseResidualContractError("witness panel path IDs overlap")
        scalars = (
            float(self.shrinkage),
            float(self.signal_energy),
            float(self.panel_mean_noise),
            float(self.averaged_table_noise),
        )
        if not all(math.isfinite(item) for item in scalars):
            raise CoarseResidualContractError("baseline scalar is nonfinite")
        if (
            not 0.0 < scalars[0] <= 1.0
            or scalars[1] <= 0.0
            or scalars[2] < 0.0
            or scalars[3] != 0.5 * scalars[2]
        ):
            raise CoarseResidualContractError("baseline shrinkage/noise is invalid")
        expected_shrinkage = scalars[1] / (scalars[1] + scalars[3])
        if scalars[0] != expected_shrinkage:
            raise CoarseResidualContractError("baseline shrinkage formula changed")
        expected_values = np.ascontiguousarray(raw_values * scalars[0])
        if not np.array_equal(values, expected_values):
            raise CoarseResidualContractError("baseline values do not match shrinkage")
        hashes = (
            self.left_cell_means_file_sha256,
            self.right_cell_means_file_sha256,
            self.left_cell_means_array_sha256,
            self.right_cell_means_array_sha256,
            self.witness_registry_sha256,
        )
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            raise CoarseResidualContractError("baseline provenance hash is invalid")
        if self.schema != BASELINE_SCHEMA:
            raise CoarseResidualContractError("baseline schema changed")
        object.__setattr__(self, "raw_values", raw_values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "left_path_ids", left_paths)
        object.__setattr__(self, "right_path_ids", right_paths)

    @property
    def raw_values_sha256(self) -> str:
        return _float64_c_order_sha256(self.raw_values)

    @property
    def values_sha256(self) -> str:
        return _float64_c_order_sha256(self.values)

    @property
    def raw_values_serialization_sha256(self) -> str:
        return _array_sha256(self.raw_values)

    @property
    def values_serialization_sha256(self) -> str:
        return _array_sha256(self.values)

    @property
    def baseline_energy(self) -> float:
        # This is frozen to NumPy's binary64 pairwise mean convention because
        # the production gate binds the exact historical scalar.
        return float(np.mean(self.values * self.values, dtype=np.float64))

    def to_record(self) -> dict[str, Any]:
        body = {
            "schema": self.schema,
            "schema_version": 1,
            "shape": list(WITNESS_CELL_SHAPE),
            "dtype": self.values.dtype.str,
            "shrinkage": float(self.shrinkage),
            "signal_energy": float(self.signal_energy),
            "panel_mean_noise": float(self.panel_mean_noise),
            "averaged_table_noise": float(self.averaged_table_noise),
            "baseline_energy": self.baseline_energy,
            "raw_values_sha256": self.raw_values_sha256,
            "values_sha256": self.values_sha256,
            "raw_values_serialization_sha256": (
                self.raw_values_serialization_sha256
            ),
            "values_serialization_sha256": self.values_serialization_sha256,
            "left_path_ids": self.left_path_ids.tolist(),
            "right_path_ids": self.right_path_ids.tolist(),
            "left_cell_means_file_sha256": self.left_cell_means_file_sha256,
            "right_cell_means_file_sha256": self.right_cell_means_file_sha256,
            "left_cell_means_array_sha256": self.left_cell_means_array_sha256,
            "right_cell_means_array_sha256": self.right_cell_means_array_sha256,
            "witness_registry_sha256": self.witness_registry_sha256,
            "fit_role": "historical_witness_panels_training_only",
            "target_modified": 0,
        }
        return {**body, "semantic_sha256": semantic_sha256(body)}

    @property
    def fingerprint(self) -> str:
        return str(self.to_record()["semantic_sha256"])

    def predict_numpy(self, reverse_time: Any, phase: Any) -> np.ndarray:
        quartile = reverse_time_quartile_numpy(reverse_time, phase)
        phases = np.asarray(phase, dtype=np.int64)
        return np.ascontiguousarray(self.values[quartile, phases])


def derive_frozen_witness_baseline(
    left_cell_means: Any,
    right_cell_means: Any,
    left_path_ids: Any,
    right_path_ids: Any,
    *,
    left_cell_means_file_sha256: str,
    right_cell_means_file_sha256: str,
    witness_registry_sha256: str,
) -> FrozenCoarseBaseline:
    """Derive the exact global-shrinkage table from two 64-path panels."""

    left = _readonly_array(
        left_cell_means,
        dtype=np.dtype(np.float64),
        shape=WITNESS_PANEL_SHAPE,
        name="left_cell_means",
    )
    right = _readonly_array(
        right_cell_means,
        dtype=np.dtype(np.float64),
        shape=WITNESS_PANEL_SHAPE,
        name="right_cell_means",
    )
    left_paths = _canonical_path_ids(
        left_path_ids,
        name="left_path_ids",
        expected_count=WITNESS_PATHS_PER_PANEL,
    )
    right_paths = _canonical_path_ids(
        right_path_ids,
        name="right_path_ids",
        expected_count=WITNESS_PATHS_PER_PANEL,
    )
    if set(left_paths.tolist()).intersection(right_paths.tolist()):
        raise CoarseResidualContractError("witness panels are not independent")
    left_mean = np.mean(left, axis=0, dtype=np.float64)
    right_mean = np.mean(right, axis=0, dtype=np.float64)
    signal = stable_sum(left_mean * right_mean) / left_mean.size
    difference = left_mean - right_mean
    panel_noise = 0.5 * stable_sum(difference * difference) / difference.size
    averaged_noise = 0.5 * panel_noise
    denominator = signal + averaged_noise
    if (
        not all(math.isfinite(value) for value in (signal, panel_noise, denominator))
        or signal <= 0.0
        or panel_noise < 0.0
        or denominator <= 0.0
    ):
        raise CoarseResidualContractError(
            "witness does not define a positive finite shrinkage"
        )
    shrinkage = signal / denominator
    raw_values = np.ascontiguousarray(0.5 * (left_mean + right_mean))
    values = np.ascontiguousarray(raw_values * shrinkage)
    return FrozenCoarseBaseline(
        raw_values=raw_values,
        values=values,
        left_path_ids=left_paths,
        right_path_ids=right_paths,
        shrinkage=float(shrinkage),
        signal_energy=float(signal),
        panel_mean_noise=float(panel_noise),
        averaged_table_noise=float(averaged_noise),
        left_cell_means_file_sha256=str(left_cell_means_file_sha256),
        right_cell_means_file_sha256=str(right_cell_means_file_sha256),
        left_cell_means_array_sha256=_witness_seal_array_sha256(left),
        right_cell_means_array_sha256=_witness_seal_array_sha256(right),
        witness_registry_sha256=str(witness_registry_sha256),
    )


def _registry_index(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = registry.get("records")
    if not isinstance(records, list) or int(registry.get("record_count", -1)) != len(
        records
    ):
        raise CoarseResidualContractError("witness artifact registry is malformed")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or record["path"] in result
        ):
            raise CoarseResidualContractError(
                "witness artifact registry contains an invalid record"
            )
        result[str(record["path"]).replace("\\", "/")] = record
    return result


def _verify_registered_file(
    run_dir: Path,
    index: Mapping[str, Mapping[str, Any]],
    relative: str,
) -> tuple[Path, str]:
    normalized = relative.replace("\\", "/")
    if normalized.startswith("/") or ".." in Path(normalized).parts:
        raise CoarseResidualContractError("registered artifact path escapes run directory")
    record = index.get(normalized)
    if record is None:
        raise CoarseResidualContractError(f"artifact is not registered: {normalized}")
    path = run_dir.joinpath(*normalized.split("/"))
    if not path.is_file():
        raise CoarseResidualContractError(f"registered artifact is missing: {normalized}")
    size = path.stat().st_size
    digest = _file_sha256(path)
    if size != int(record.get("size", -1)) or digest != record.get("sha256"):
        raise CoarseResidualContractError(
            f"registered artifact hash/size mismatch: {normalized}"
        )
    return path, digest


def _load_panel_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"cell_means", "path_ids"}:
                raise CoarseResidualContractError(
                    "witness cell-means archive fields changed"
                )
            means = np.asarray(archive["cell_means"])
            paths = np.asarray(archive["path_ids"])
    except (OSError, ValueError, KeyError) as exc:
        raise CoarseResidualContractError(
            f"cannot load witness cell-means archive {path}"
        ) from exc
    return means, paths


def load_frozen_witness_baseline(
    witness_run_dir: str | Path,
    *,
    expected_registry_sha256: str | None = WITNESS_REGISTRY_SHA256,
) -> FrozenCoarseBaseline:
    """Load and bind the two sealed witness panels.

    Passing ``expected_registry_sha256=None`` is intended only for isolated
    integrity fixtures.  Production callers use the frozen default.
    """

    run_dir = Path(witness_run_dir).resolve()
    if not run_dir.is_dir():
        raise CoarseResidualContractError("witness run directory does not exist")
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    registry = _load_json(registry_path)
    status = _load_json(status_path)
    registry_file_sha = _file_sha256(registry_path)
    if (
        status.get("artifact_registry_file_sha256") != registry_file_sha
        or int(status.get("artifact_registry_record_count", -1))
        != int(registry.get("record_count", -2))
        or status.get("decision") != "exact_physical_coarse_signal_detected"
        or status.get("state") != "completed"
    ):
        raise CoarseResidualContractError("witness status/registry binding is invalid")
    semantic_registry_sha = str(status.get("artifact_registry_sha256", ""))
    if (
        expected_registry_sha256 is not None
        and semantic_registry_sha != expected_registry_sha256
    ):
        raise CoarseResidualContractError("unexpected witness registry fingerprint")
    if (
        expected_registry_sha256 == WITNESS_REGISTRY_SHA256
        and int(registry.get("record_count", -1)) != WITNESS_REGISTRY_RECORD_COUNT
    ):
        raise CoarseResidualContractError("unexpected witness registry size")
    index = _registry_index(registry)
    gate_path, _ = _verify_registered_file(
        run_dir, index, "coarse_signal_witness_gate.json"
    )
    gate = _load_json(gate_path)
    if gate.get("passed") != 1 or gate.get("evaluation_status") != "evaluated":
        raise CoarseResidualContractError("witness gate did not pass")

    panel_values: list[np.ndarray] = []
    panel_paths: list[np.ndarray] = []
    panel_file_hashes: list[str] = []
    panel_array_hashes: list[str] = []
    for role in ("a", "b"):
        seal_path, _ = _verify_registered_file(
            run_dir, index, f"panel_{role}_seal.json"
        )
        seal = _load_json(seal_path)
        relative = str(seal.get("cell_means_file", ""))
        panel_path, panel_file_sha = _verify_registered_file(run_dir, index, relative)
        if seal.get("cell_means_file_sha256") != panel_file_sha:
            raise CoarseResidualContractError("panel seal file hash mismatch")
        means, paths = _load_panel_npz(panel_path)
        array_sha = _witness_seal_array_sha256(means)
        if (
            seal.get("cell_means_array_sha256") != array_sha
            or seal.get("path_ids") != paths.tolist()
            or seal.get("path_plan_sha256") != WITNESS_PATH_PLAN_SHA256
            or seal.get("statistic_plan_sha256") != WITNESS_STATISTIC_PLAN_SHA256
        ):
            raise CoarseResidualContractError("panel seal content mismatch")
        panel_values.append(means)
        panel_paths.append(paths)
        panel_file_hashes.append(panel_file_sha)
        panel_array_hashes.append(array_sha)

    baseline = derive_frozen_witness_baseline(
        panel_values[0],
        panel_values[1],
        panel_paths[0],
        panel_paths[1],
        left_cell_means_file_sha256=panel_file_hashes[0],
        right_cell_means_file_sha256=panel_file_hashes[1],
        witness_registry_sha256=semantic_registry_sha,
    )
    if baseline.left_cell_means_array_sha256 != panel_array_hashes[0] or (
        baseline.right_cell_means_array_sha256 != panel_array_hashes[1]
    ):
        raise CoarseResidualContractError("derived panel array hash mismatch")
    if expected_registry_sha256 == WITNESS_REGISTRY_SHA256:
        exact_checks = (
            panel_file_hashes[0] == WITNESS_PANEL_A_FILE_SHA256,
            panel_file_hashes[1] == WITNESS_PANEL_B_FILE_SHA256,
            panel_array_hashes[0] == WITNESS_PANEL_A_ARRAY_SHA256,
            panel_array_hashes[1] == WITNESS_PANEL_B_ARRAY_SHA256,
            tuple(panel_paths[0].tolist()) == WITNESS_PANEL_A_PATH_IDS,
            tuple(panel_paths[1].tolist()) == WITNESS_PANEL_B_PATH_IDS,
            baseline.signal_energy == WITNESS_SIGNAL_ENERGY,
            baseline.panel_mean_noise == WITNESS_PANEL_MEAN_NOISE,
            baseline.averaged_table_noise == WITNESS_AVERAGED_TABLE_NOISE,
            baseline.shrinkage == GLOBAL_SHRINKAGE,
            baseline.baseline_energy == WITNESS_BASELINE_ENERGY,
            baseline.raw_values_sha256 == WITNESS_RAW_VALUES_SHA256,
            baseline.values_sha256 == WITNESS_VALUES_SHA256,
            baseline.raw_values_serialization_sha256
            == WITNESS_RAW_VALUES_SERIALIZATION_SHA256,
            baseline.values_serialization_sha256
            == WITNESS_VALUES_SERIALIZATION_SHA256,
        )
        if not all(exact_checks):
            raise CoarseResidualContractError(
                "loaded witness baseline differs from the frozen production result"
            )
    return baseline


def save_frozen_coarse_baseline(
    path: str | Path, baseline: FrozenCoarseBaseline
) -> dict[str, Any]:
    """Atomically persist one baseline and its self-verifying metadata."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = baseline.to_record()
    metadata = {
        "schema": BASELINE_FILE_SCHEMA,
        "schema_version": 1,
        "baseline_record": record,
    }
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                raw_values=baseline.raw_values,
                values=baseline.values,
                left_path_ids=baseline.left_path_ids,
                right_path_ids=baseline.right_path_ids,
                metadata_json=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(target),
        "size": target.stat().st_size,
        "sha256": _file_sha256(target),
        "baseline_semantic_sha256": baseline.fingerprint,
    }


def load_frozen_coarse_baseline(
    path: str | Path, *, expected_sha256: str | None = None
) -> FrozenCoarseBaseline:
    """Load a baseline file and recompute all array and semantic bindings."""

    source = Path(path)
    if expected_sha256 is not None and _file_sha256(source) != expected_sha256:
        raise CoarseResidualContractError("baseline file fingerprint mismatch")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {
                "raw_values",
                "values",
                "left_path_ids",
                "right_path_ids",
                "metadata_json",
            }:
                raise CoarseResidualContractError("baseline archive fields changed")
            raw_values = np.asarray(archive["raw_values"])
            values = np.asarray(archive["values"])
            left_paths = np.asarray(archive["left_path_ids"])
            right_paths = np.asarray(archive["right_path_ids"])
            metadata_bytes = np.asarray(archive["metadata_json"])
    except (OSError, ValueError, KeyError) as exc:
        raise CoarseResidualContractError("cannot load baseline archive") from exc
    if metadata_bytes.dtype != np.uint8 or metadata_bytes.ndim != 1:
        raise CoarseResidualContractError("baseline metadata encoding changed")
    try:
        metadata = json.loads(metadata_bytes.tobytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CoarseResidualContractError("baseline metadata is invalid") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != BASELINE_FILE_SCHEMA
        or not isinstance(metadata.get("baseline_record"), dict)
    ):
        raise CoarseResidualContractError("baseline file metadata schema changed")
    record = metadata["baseline_record"]
    result = FrozenCoarseBaseline(
        raw_values=raw_values,
        values=values,
        left_path_ids=left_paths,
        right_path_ids=right_paths,
        shrinkage=float(record.get("shrinkage", math.nan)),
        signal_energy=float(record.get("signal_energy", math.nan)),
        panel_mean_noise=float(record.get("panel_mean_noise", math.nan)),
        averaged_table_noise=float(record.get("averaged_table_noise", math.nan)),
        left_cell_means_file_sha256=str(
            record.get("left_cell_means_file_sha256", "")
        ),
        right_cell_means_file_sha256=str(
            record.get("right_cell_means_file_sha256", "")
        ),
        left_cell_means_array_sha256=str(
            record.get("left_cell_means_array_sha256", "")
        ),
        right_cell_means_array_sha256=str(
            record.get("right_cell_means_array_sha256", "")
        ),
        witness_registry_sha256=str(record.get("witness_registry_sha256", "")),
    )
    if (
        record.get("raw_values_sha256") != result.raw_values_sha256
        or record.get("values_sha256") != result.values_sha256
        or record.get("raw_values_serialization_sha256")
        != result.raw_values_serialization_sha256
        or record.get("values_serialization_sha256")
        != result.values_serialization_sha256
        or record.get("left_path_ids") != result.left_path_ids.tolist()
        or record.get("right_path_ids") != result.right_path_ids.tolist()
        or record.get("semantic_sha256") != result.fingerprint
        or result.to_record() != record
    ):
        raise CoarseResidualContractError("baseline archive metadata was tampered")
    return result


def _reverse_time_step_numpy(reverse_time: Any, phase: Any) -> np.ndarray:
    times_raw = np.asarray(reverse_time)
    phases_raw = np.asarray(phase)
    if (
        times_raw.ndim != 1
        or phases_raw.shape != times_raw.shape
        or times_raw.dtype.kind not in "fc"
        or phases_raw.dtype.kind not in "iu"
        or not np.isfinite(times_raw).all()
    ):
        raise CoarseResidualContractError("reverse-time coordinates are malformed")
    times = np.asarray(times_raw, dtype=np.float64)
    phases = np.asarray(phases_raw, dtype=np.int64)
    if ((phases < 0) | (phases >= PHASE_COUNT)).any():
        raise CoarseResidualContractError("phase lies outside the exact split chain")
    phase_tick_real = (1.0 - times) * float(PHASE_COUNT * OUTER_STEPS)
    phase_tick = np.rint(phase_tick_real).astype(np.int64)
    tolerance = 2.0e-4 if times_raw.dtype == np.float32 else 2.0e-10
    if np.max(np.abs(phase_tick_real - phase_tick), initial=0.0) > tolerance:
        raise CoarseResidualContractError(
            "reverse time is not an exact split-chain coordinate"
        )
    numerator = phase_tick - phases - 1
    if (numerator % PHASE_COUNT != 0).any():
        raise CoarseResidualContractError("reverse time and phase are inconsistent")
    steps = numerator // PHASE_COUNT
    if (
        ((steps < 0) | (steps >= OUTER_STEPS)).any()
        or not np.isin(steps, np.asarray(SELECTED_OUTER_STEPS)).all()
    ):
        raise CoarseResidualContractError(
            "reverse time is outside the selected exact-K=512 rows"
        )
    return np.ascontiguousarray(steps)


def reverse_time_quartile_numpy(reverse_time: Any, phase: Any) -> np.ndarray:
    """Recover the time quartile using permitted reverse time and phase only."""

    return np.ascontiguousarray(_reverse_time_step_numpy(reverse_time, phase) // 128)


def reverse_time_quartile_tensor(reverse_time: Tensor, phase: Tensor) -> Tensor:
    """Torch counterpart of :func:`reverse_time_quartile_numpy`."""

    if (
        not isinstance(reverse_time, Tensor)
        or not isinstance(phase, Tensor)
        or reverse_time.ndim != 1
        or phase.shape != reverse_time.shape
        or not reverse_time.dtype.is_floating_point
        or phase.dtype
        not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        or reverse_time.device != phase.device
    ):
        raise CoarseResidualContractError("tensor reverse-time coordinates are malformed")
    if not bool(torch.isfinite(reverse_time).all()):
        raise CoarseResidualContractError("reverse time contains nonfinite values")
    phases = phase.to(dtype=torch.int64)
    if bool(torch.any((phases < 0) | (phases >= PHASE_COUNT))):
        raise CoarseResidualContractError("phase lies outside the exact split chain")
    real_tick = (1.0 - reverse_time.to(dtype=torch.float64)) * float(
        PHASE_COUNT * OUTER_STEPS
    )
    tick = torch.round(real_tick).to(dtype=torch.int64)
    tolerance = 2.0e-4 if reverse_time.dtype == torch.float32 else 2.0e-10
    if bool(torch.any(torch.abs(real_tick - tick.to(torch.float64)) > tolerance)):
        raise CoarseResidualContractError(
            "reverse time is not an exact split-chain coordinate"
        )
    numerator = tick - phases - 1
    if bool(torch.any(torch.remainder(numerator, PHASE_COUNT) != 0)):
        raise CoarseResidualContractError("reverse time and phase are inconsistent")
    steps = torch.div(numerator, PHASE_COUNT, rounding_mode="floor")
    selected = torch.as_tensor(
        SELECTED_OUTER_STEPS, dtype=torch.int64, device=steps.device
    )
    if bool(torch.any((steps < 0) | (steps >= OUTER_STEPS))) or not bool(
        torch.isin(steps, selected).all()
    ):
        raise CoarseResidualContractError(
            "reverse time is outside the selected exact-K=512 rows"
        )
    return torch.div(steps, 128, rounding_mode="floor")


def zero_initialize_residual(model: nn.Module) -> None:
    """Make the unchanged Jacobi/RB model's output identically zero.

    Hidden convolutional features retain their ordinary initialization.  Only
    the two output paths are zeroed, so gradients can immediately update the
    local-affine head and the spatial head can begin learning on the next step.
    """

    if not isinstance(model, JacobiRBPhasePredictor):
        raise CoarseResidualContractError(
            "residual must be the unchanged JacobiRBPhasePredictor"
        )
    with torch.no_grad():
        for layer_name in ("spatial_output", "local_affine"):
            layer = getattr(model, layer_name, None)
            if not isinstance(layer, (nn.Conv2d, nn.Linear)):
                raise CoarseResidualContractError(
                    "Jacobi/RB residual output-head contract changed"
                )
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()


class CoarseResidualPredictor(nn.Module):
    """Frozen coarse table plus the unchanged permitted-input residual model."""

    def __init__(
        self,
        baseline: FrozenCoarseBaseline,
        residual: JacobiRBPhasePredictor | None = None,
        *,
        zero_residual: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(baseline, FrozenCoarseBaseline):
            raise CoarseResidualContractError("baseline has the wrong type")
        active = residual if residual is not None else JacobiRBPhasePredictor(width=32)
        if not isinstance(active, JacobiRBPhasePredictor):
            raise CoarseResidualContractError(
                "residual must be the unchanged JacobiRBPhasePredictor"
            )
        self.residual = active
        self.baseline_fingerprint = baseline.fingerprint
        self.register_buffer(
            "_coarse_values",
            torch.from_numpy(np.array(baseline.values, copy=True)),
            persistent=True,
        )
        if zero_residual:
            zero_initialize_residual(self.residual)

    def baseline_prediction(self, inputs: ModelInputs, *, dtype: torch.dtype) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise CoarseResidualContractError(
                "predictor accepts only the exact permitted ModelInputs"
            )
        quartile = reverse_time_quartile_tensor(inputs.reverse_time, inputs.phase)
        phase = inputs.phase.to(dtype=torch.int64)
        values = self._coarse_values.to(device=phase.device, dtype=dtype)
        return values[quartile, phase]

    def forward(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise CoarseResidualContractError(
                "predictor accepts only the exact permitted ModelInputs"
            )
        residual = call_model(self.residual, inputs)
        # The scientific baseline is SHA-bound as binary64 C-order values.
        # Keep it binary64 in the combined prediction; casting B through the
        # float32 residual coordinate would silently change every table cell.
        baseline = self.baseline_prediction(inputs, dtype=torch.float64)
        return baseline + residual.to(dtype=torch.float64)


def exact_combined_target_scale(target: Tensor | np.ndarray) -> float:
    """Training-only binary64 RMS of the unchanged exact target."""

    if isinstance(target, Tensor):
        if target.dtype != torch.float64:
            raise CoarseResidualContractError(
                "exact target scale requires binary64 targets"
            )
        values = target.detach().to(device="cpu").numpy()
    else:
        values = np.asarray(target)
    if values.dtype != np.float64 or values.size == 0 or not np.isfinite(values).all():
        raise CoarseResidualContractError(
            "exact target scale requires finite binary64 targets"
        )
    scale = math.sqrt(stable_sum(values * values) / values.size)
    if not math.isfinite(scale) or scale <= 0.0:
        raise CoarseResidualContractError("exact target scale is not positive")
    return float(scale)


def combined_exact_mse(
    combined_prediction: Tensor,
    exact_target: Tensor,
    target_scale: float,
) -> tuple[Tensor, Tensor]:
    """Return ``(optimizer_loss, raw_mse)`` for direct exact-target MSE."""

    if (
        not isinstance(combined_prediction, Tensor)
        or not isinstance(exact_target, Tensor)
        or combined_prediction.shape != exact_target.shape
        or combined_prediction.ndim != 2
        or combined_prediction.shape[1] != EDGES_PER_PHASE
    ):
        raise CoarseResidualContractError("combined prediction/target shape is invalid")
    scale = float(target_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise CoarseResidualContractError("target scale must be finite and positive")
    prediction64 = combined_prediction.to(dtype=torch.float64)
    target64 = exact_target.to(dtype=torch.float64)
    raw = torch.mean((prediction64 - target64).square())
    return raw / (scale * scale), raw


def residualized_mse_algebra_error(
    baseline_prediction: Tensor,
    residual_prediction: Tensor,
    exact_target: Tensor,
) -> float:
    """Numerical audit that residualized and direct MSE are the same objective."""

    if not (
        baseline_prediction.shape
        == residual_prediction.shape
        == exact_target.shape
    ):
        raise CoarseResidualContractError("MSE algebra arrays have different shapes")
    direct = torch.mean(
        (
            baseline_prediction.to(torch.float64)
            + residual_prediction.to(torch.float64)
            - exact_target.to(torch.float64)
        ).square()
    )
    residualized = torch.mean(
        (
            residual_prediction.to(torch.float64)
            - (
                exact_target.to(torch.float64)
                - baseline_prediction.to(torch.float64)
            )
        ).square()
    )
    return float(torch.abs(direct - residualized).detach().cpu())


@dataclass(frozen=True)
class PathContrastTable:
    """Canonical path-level paired loss differences."""

    path_ids: np.ndarray = field(repr=False, compare=False)
    values: np.ndarray = field(repr=False, compare=False)
    names: tuple[str, ...] = ALL_CONTRAST_NAMES

    def __post_init__(self) -> None:
        paths = _canonical_path_ids(self.path_ids, name="path_ids")
        values = _readonly_array(
            self.values,
            dtype=np.dtype(np.float64),
            shape=(paths.size, len(self.names)),
            name="path_contrast_values",
        )
        if tuple(self.names) != ALL_CONTRAST_NAMES:
            raise CoarseResidualContractError("path contrast family changed")
        object.__setattr__(self, "path_ids", paths)
        object.__setattr__(self, "values", values)

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    def column(self, name: str) -> np.ndarray:
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise CoarseResidualContractError(f"unknown contrast {name}") from exc
        return self.values[:, index]

    def to_record(self) -> dict[str, Any]:
        body = {
            "schema": PATH_CONTRAST_SCHEMA,
            "schema_version": 1,
            "path_ids": self.path_ids.tolist(),
            "path_count": self.path_count,
            "names": list(self.names),
            "values_shape": list(self.values.shape),
            "values_sha256": _array_sha256(self.values),
            "bootstrap_unit": "whole_path",
        }
        return {**body, "semantic_sha256": semantic_sha256(body)}


def _stable_mse_numpy(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = prediction - target
    return stable_sum(difference * difference) / difference.size


def path_loss_contrasts(
    exact_target: Any,
    baseline_prediction: Any,
    combined_prediction: Any,
    path_id: Any,
    reverse_time: Any,
    phase: Any,
) -> PathContrastTable:
    """Compute paired overall/data-end risk contrasts by whole path."""

    target = np.asarray(exact_target)
    baseline = np.asarray(baseline_prediction)
    combined = np.asarray(combined_prediction)
    raw_paths = np.asarray(path_id)
    if (
        target.dtype != np.float64
        or baseline.dtype != np.float64
        or combined.dtype != np.float64
        or target.ndim != 2
        or target.shape[1] != EDGES_PER_PHASE
        or baseline.shape != target.shape
        or combined.shape != target.shape
        or raw_paths.shape != (target.shape[0],)
        or raw_paths.dtype.kind not in "iu"
        or not (
            np.isfinite(target).all()
            and np.isfinite(baseline).all()
            and np.isfinite(combined).all()
        )
    ):
        raise CoarseResidualContractError("path contrast inputs are invalid")
    quartile = reverse_time_quartile_numpy(reverse_time, phase)
    paths = _canonical_path_ids(np.unique(raw_paths), name="unique_path_ids")
    path_values = np.empty((paths.size, len(ALL_CONTRAST_NAMES)), dtype=np.float64)
    expected_overall_count: int | None = None
    expected_data_end_count: int | None = None
    for row, path in enumerate(paths.tolist()):
        path_mask = raw_paths == path
        data_end_mask = path_mask & (quartile == 3)
        overall_count = int(path_mask.sum())
        data_end_count = int(data_end_mask.sum())
        if overall_count <= 0 or data_end_count <= 0:
            raise CoarseResidualContractError("path does not populate both scopes")
        if expected_overall_count is None:
            expected_overall_count = overall_count
            expected_data_end_count = data_end_count
        elif (
            overall_count != expected_overall_count
            or data_end_count != expected_data_end_count
        ):
            raise CoarseResidualContractError("paths have unequal scope row counts")
        output: list[float] = []
        for mask in (path_mask, data_end_mask):
            zero = np.zeros_like(target[mask])
            zero_mse = _stable_mse_numpy(zero, target[mask])
            baseline_mse = _stable_mse_numpy(baseline[mask], target[mask])
            combined_mse = _stable_mse_numpy(combined[mask], target[mask])
            output.extend(
                (
                    zero_mse - baseline_mse,
                    baseline_mse - combined_mse,
                    zero_mse - combined_mse,
                )
            )
        path_values[row] = output
    if not np.array_equal(
        path_values[:, 0] + path_values[:, 1], path_values[:, 2]
    ) or not np.array_equal(
        path_values[:, 3] + path_values[:, 4], path_values[:, 5]
    ):
        # Binary64 subtraction can differ by a few ulps under the two algebraic
        # routes.  Permit only that rounding effect.
        first = np.max(
            np.abs(path_values[:, 0] + path_values[:, 1] - path_values[:, 2])
        )
        second = np.max(
            np.abs(path_values[:, 3] + path_values[:, 4] - path_values[:, 5])
        )
        tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(path_values)))
        )
        if max(float(first), float(second)) > tolerance:
            raise CoarseResidualContractError("path contrast algebra is inconsistent")
    return PathContrastTable(paths, np.ascontiguousarray(path_values))


@dataclass(frozen=True)
class StudentizedMaxTResult:
    """One-sided simultaneous lower bounds for a path-contrast family."""

    family_names: tuple[str, ...]
    point_estimates: np.ndarray = field(repr=False, compare=False)
    standard_errors: np.ndarray = field(repr=False, compare=False)
    lower_bounds: np.ndarray = field(repr=False, compare=False)
    critical_value: float
    path_count: int
    confidence: float
    replicates: int
    seed: int
    namespace: int

    def __post_init__(self) -> None:
        names = tuple(self.family_names)
        if not names or len(set(names)) != len(names):
            raise CoarseResidualContractError("max-T family names are invalid")
        expected = (len(names),)
        points = _readonly_array(
            self.point_estimates,
            dtype=np.dtype(np.float64),
            shape=expected,
            name="point_estimates",
        )
        errors = _readonly_array(
            self.standard_errors,
            dtype=np.dtype(np.float64),
            shape=expected,
            name="standard_errors",
        )
        lower = _readonly_array(
            self.lower_bounds,
            dtype=np.dtype(np.float64),
            shape=expected,
            name="lower_bounds",
        )
        if (
            (errors <= 0.0).any()
            or int(self.path_count) < 8
            or not 0.0 < float(self.confidence) < 1.0
            or int(self.replicates) <= 0
            or int(self.seed) < 0
            or int(self.namespace) < 0
            or not math.isfinite(float(self.critical_value))
        ):
            raise CoarseResidualContractError("max-T result is invalid")
        object.__setattr__(self, "family_names", names)
        object.__setattr__(self, "point_estimates", points)
        object.__setattr__(self, "standard_errors", errors)
        object.__setattr__(self, "lower_bounds", lower)

    @property
    def passed(self) -> bool:
        return bool(np.all(self.lower_bounds > 0.0))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": MAX_T_SCHEMA,
            "schema_version": 1,
            "method": "centered_whole_path_studentized_max_t",
            "bootstrap_unit": "whole_path_jointly_across_family",
            "quantile_method": "higher",
            "family_names": list(self.family_names),
            "point_estimates": {
                name: float(value)
                for name, value in zip(self.family_names, self.point_estimates)
            },
            "standard_errors": {
                name: float(value)
                for name, value in zip(self.family_names, self.standard_errors)
            },
            "lower_bounds": {
                name: float(value)
                for name, value in zip(self.family_names, self.lower_bounds)
            },
            "critical_value": float(self.critical_value),
            "path_count": int(self.path_count),
            "confidence": float(self.confidence),
            "replicates": int(self.replicates),
            "seed": int(self.seed),
            "namespace": int(self.namespace),
            "negative_values_truncated": 0,
            "passed": int(self.passed),
        }


def one_sided_studentized_max_t(
    contrasts: PathContrastTable,
    *,
    family_names: Sequence[str] = PRIMARY_CONTRAST_NAMES,
    seed: int = DEFAULT_MAX_T_SEED,
    namespace: int = 0,
    replicates: int = DEFAULT_MAX_T_REPLICATES,
    confidence: float = DEFAULT_MAX_T_CONFIDENCE,
    chunk_size: int = 512,
) -> StudentizedMaxTResult:
    """Deterministic one-sided studentized max-T whole-path bootstrap."""

    if not isinstance(contrasts, PathContrastTable):
        raise CoarseResidualContractError("max-T requires a PathContrastTable")
    names = tuple(str(name) for name in family_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in contrasts.names for name in names)
        or int(replicates) <= 0
        or int(chunk_size) <= 0
        or not 0.0 < float(confidence) < 1.0
        or int(seed) < 0
        or int(namespace) < 0
    ):
        raise CoarseResidualContractError("max-T configuration is invalid")
    if contrasts.path_count < 8:
        raise CoarseResidualContractError("max-T requires at least eight paths")
    columns = [contrasts.names.index(name) for name in names]
    values = np.ascontiguousarray(contrasts.values[:, columns])
    points = np.mean(values, axis=0, dtype=np.float64)
    standard_errors = np.std(values, axis=0, ddof=1) / math.sqrt(
        contrasts.path_count
    )
    if (
        not np.isfinite(points).all()
        or not np.isfinite(standard_errors).all()
        or (standard_errors <= 0.0).any()
    ):
        raise CoarseResidualContractError(
            "max-T family has a degenerate/nonfinite standard error"
        )
    generator = np.random.Generator(
        np.random.Philox([int(seed), int(namespace), _MAX_T_NAMESPACE])
    )
    maxima = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), int(chunk_size)):
        stop = min(int(replicates), start + int(chunk_size))
        count = stop - start
        indices = generator.integers(
            0,
            contrasts.path_count,
            size=(count, contrasts.path_count),
            dtype=np.int64,
        )
        sampled = values[indices]
        means = np.mean(sampled, axis=1, dtype=np.float64)
        errors = np.std(sampled, axis=1, ddof=1) / math.sqrt(
            contrasts.path_count
        )
        if not np.isfinite(errors).all() or (errors <= 0.0).any():
            raise CoarseResidualContractError(
                "bootstrap produced degenerate studentization"
            )
        studentized = (means - points[None, :]) / errors
        maxima[start:stop] = np.max(studentized, axis=1)
    if not np.isfinite(maxima).all():
        raise CoarseResidualContractError("max-T bootstrap is nonfinite")
    critical = float(np.quantile(maxima, float(confidence), method="higher"))
    lower = np.ascontiguousarray(points - critical * standard_errors)
    return StudentizedMaxTResult(
        family_names=names,
        point_estimates=np.ascontiguousarray(points),
        standard_errors=np.ascontiguousarray(standard_errors),
        lower_bounds=lower,
        critical_value=critical,
        path_count=contrasts.path_count,
        confidence=float(confidence),
        replicates=int(replicates),
        seed=int(seed),
        namespace=int(namespace),
    )


__all__ = [
    "ALL_CONTRAST_NAMES",
    "BASELINE_FILE_SCHEMA",
    "BASELINE_SCHEMA",
    "COARSE_RESIDUAL_VERSION",
    "CoarseResidualContractError",
    "CoarseResidualPredictor",
    "DEFAULT_MAX_T_CONFIDENCE",
    "DEFAULT_MAX_T_REPLICATES",
    "DEFAULT_MAX_T_SEED",
    "FrozenCoarseBaseline",
    "GLOBAL_SHRINKAGE",
    "MAX_T_SCHEMA",
    "PATH_CONTRAST_SCHEMA",
    "PRIMARY_CONTRAST_NAMES",
    "PathContrastTable",
    "StudentizedMaxTResult",
    "WITNESS_AVERAGED_TABLE_NOISE",
    "WITNESS_BASELINE_ENERGY",
    "WITNESS_PANEL_MEAN_NOISE",
    "WITNESS_RAW_VALUES_SERIALIZATION_SHA256",
    "WITNESS_RAW_VALUES_SHA256",
    "WITNESS_REGISTRY_SHA256",
    "WITNESS_SIGNAL_ENERGY",
    "WITNESS_VALUES_SERIALIZATION_SHA256",
    "WITNESS_VALUES_SHA256",
    "combined_exact_mse",
    "derive_frozen_witness_baseline",
    "exact_combined_target_scale",
    "load_frozen_coarse_baseline",
    "load_frozen_witness_baseline",
    "one_sided_studentized_max_t",
    "path_loss_contrasts",
    "residualized_mse_algebra_error",
    "reverse_time_quartile_numpy",
    "reverse_time_quartile_tensor",
    "save_frozen_coarse_baseline",
    "zero_initialize_residual",
]
