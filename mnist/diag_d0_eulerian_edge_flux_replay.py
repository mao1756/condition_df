from __future__ import annotations

"""Fresh factor-one replay of the historical global Eulerian edge-flux model.

This additive runner never trains, ranks, rejects, or replaces candidate images.
Adaptive integrator retry attempts remain part of the frozen current sampler law.
"""

import argparse
import csv
import dataclasses
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from mnist.conditioned_diffusion import SmallMnistCNN
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    DirectFluxUNet,
    _sample_source_batch_torch,
    eulerian_flux_step_torch,
    flux_divergence_torch,
    free_drift_flux_torch,
    natural_horizon,
    poisson_flux_from_velocity_torch,
    step_component_rms_torch,
)
from mnist.mnist_generation_benchmark import (
    compute_generation_metrics,
    exact_duplicate_metrics,
    read_mnist_arff_slice,
    score_human_review,
    sha256_file,
    within_class_nn_diversity,
    write_blinded_review_bundle,
    write_contact_sheet,
)


VERSION = "eulerian-edge-flux-factor-one-replay-v1"
LEGACY_CHECKPOINT_BYTES = 13_947_413
LEGACY_CHECKPOINT_SHA256 = "8be77d1701887522f86099673431a928ad7dd2d350a06f7a94ade5c30a658cc3"
LEGACY_CONFIG_SHA256 = "7ad01d9b4cadc017b9221728b0c0c9d286059ee6add98b38fa6103686dc32878"
MNIST_ARFF_SHA256 = "418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b"
MNIST_ARFF_BYTES = 127_888_265

EXPECTED_NUMPY_VERSION = "2.4.4"
EXPECTED_TORCH_VERSION = "2.11.0+cu128"
EXPECTED_CHECKPOINT_KEYS = frozenset(
    {
        "model_state_dict", "config", "args", "run_metadata", "history", "labels",
        "clipping_fraction", "final_metrics", "classifier_metrics", "class_shape_stats",
        "goodbad_metrics", "source_metrics", "component_metrics", "sample_quality_metrics",
    }
)
EXPECTED_STATE_TENSORS = 50
EXPECTED_PARAMETER_COUNT = 1_747_874

PROTECTED_SOURCE_HASHES = {
    "mnist/eulerian_flux_mnist.py": "4dca1c40f25eb04b3d615bd0094891c7cedb8cea8a673607eb02e1ab977e4f19",
    "mnist/mnist_generation_benchmark.py": "2ebf13e37e03646222b8decde91034f10dff564fdc0d3f7a967af789ef3cbfd6",
    "mnist/conditioned_diffusion.py": "96906c6c1cf7fed4de191e56d6861621446e65b6952171cbf6fa556303450892",
}

K128_MANIFEST_SHA256 = "e6e25b297cf5e407fa0dcfdfd06755db56fc57bf724720723c8ca88631115b7a"
K128_TREE_DIGEST = "33f191ef2753b12f5dbe8365003cc5b312e4bd35764479809939ce5abe39e039"
K128_STATUS_SHA256 = "20e46de2d56a5bc2cbcd9f7c90088ba97cd7fa25599d24ebb0d1968c43560183"
K128_OUTCOME_SHA256 = "4d4bb2675668d90fddfc0c99c0ebc77cb9a156405bd1134e1c3d5167d3705c90"
K128_REPORT_SHA256 = "f0767a1d18b9b705a9388dd1b4d7cc8d1f1e2b91bf55a677c4636ade9942370f"
K128_MANIFEST_BYTES = 65_004
K128_STATUS_BYTES = 204
K128_OUTCOME_BYTES = 2_393
K128_REPORT_BYTES = 6_202

DDPM_TREE_DIGEST = "56e6be5c7e8066c64492a18f6309456a11868a0b4a9c446f6ca4544f9af58c28"
DDPM_MANIFEST_SHA256 = "79aa5d9ae1ca6615a46c9d699f947bea4b6a380cc32e86547cc7e49cee612953"
EVALUATOR_BYTES = 99_755
EVALUATOR_SHA256 = "3d31d42a14fee0ecc72adc1644718a037cc48e649948407da6c272b981840c92"
EVALUATOR_SELECTION_SHA256 = "e6cd9e49ca61237d3a10e9ad2fe0ad09f7a33ea22911fdd73fd99f3a07e4c668"
DDPM_MANIFEST_BYTES = 17_891
EVALUATOR_SELECTION_BYTES = 343

PATH_COUNT = 160
PATHS_PER_CLASS = 16
PATH_PREFIX = "efr-v1-"
ANCHORS = (0, 64, 128, 192, 256)
OUTER_STEPS = 256
SOURCE_SEED_BASE = 0xE14F1000
INVENTORY_SEED = 0xE14F0001
ROW_ROOT_SEEDS = {"null": 0xE14F2001, "teacher": 0xE14F3001, "learned": 0xE14F4001}
REVIEW_SEED = 0xE14F5001
SMOKE_SEED = 0xE14FF001
REVIEW_WITHIN_CLASS = (0, 5, 10, 15)

MASS_SCALE_NUMERATOR = 25_471
MASS_SCALE_DENOMINATOR = 255
MASS_SCALE_HEX = "0x1.8f8b8b8b8b8b9p+6"
TRAIN_START, TRAIN_STOP = 0, 55_000
VALIDATION_START, VALIDATION_STOP = 55_000, 60_000

MAX_ACTIVE_SECONDS = 240.0
MAX_STORAGE_MIB = 100.0
MAX_CUDA_FRACTION = 0.75
TERMINAL_RESERVE_SECONDS = 30.0
MAX_QUANTUM_SECONDS = 60.0

RESEARCH_MODE = "exploratory"
K128_REQUIRED_ROUTE = "v0_negative_pivot_experiment10"
REVIEW_POSITIVE_RECOGNIZABILITY = 0.90
REVIEW_POSITIVE_AGREEMENT = 0.75
CLASSIFIER_POSITIVE_ACCURACY = 0.80
DIVERSITY_POSITIVE_RATIO = 0.25


class EdgeFluxReplayError(RuntimeError):
    pass


class IntegrityFailure(EdgeFluxReplayError):
    pass


class ResourceStop(EdgeFluxReplayError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityFailure(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, destination: Path, *, attempts: int = 20, delay_seconds: float = 0.05) -> None:
    """Atomically replace with a bounded retry for transient Windows file locks."""
    last_error: PermissionError | None = None
    for attempt in range(int(attempts)):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            last_error = error
            if attempt + 1 < int(attempts):
                time.sleep(float(delay_seconds))
    assert last_error is not None
    try:
        source.unlink(missing_ok=True)
    except OSError:
        pass
    raise IntegrityFailure(f"atomic replace remained unavailable for {destination}") from last_error


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_json_bytes(_jsonable(value)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityFailure(f"{path} must contain a JSON object")
    return value


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    _replace_with_retry(temporary, path)


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    _replace_with_retry(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})
    _replace_with_retry(temporary, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _storage_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _manifest_rows(run_dir: Path) -> list[dict[str, Any]]:
    ignored = {"artifact_manifest.json", "SHA256SUMS.txt"}
    rows = []
    for path in sorted((item for item in run_dir.rglob("*") if item.is_file()), key=lambda item: item.relative_to(run_dir).as_posix()):
        relative = path.relative_to(run_dir).as_posix()
        if relative in ignored:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def _tree_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes([{"path": row["path"], "bytes": int(row["bytes"]), "sha256": row["sha256"]} for row in rows]))


def _git_revision(repository_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable-use-source-hashes"


def _verify_external_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_tree_digest: str,
) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    _require(manifest_path.is_file(), f"external manifest is absent: {root}")
    _require(sha256_file(manifest_path) == expected_manifest_sha256, f"external manifest hash mismatch: {root}")
    manifest = _read_json(manifest_path)
    observed_tree_digest = manifest.get("tree_digest")
    if observed_tree_digest is None:
        tree_rows = [
            (path.relative_to(root).as_posix(), path.stat().st_size, sha256_file(path))
            for path in sorted(item for item in root.rglob("*") if item.is_file())
        ]
        observed_tree_digest = hashlib.sha256(
            json.dumps(tree_rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    _require(observed_tree_digest == expected_tree_digest, f"external tree digest mismatch: {root}")
    entries = manifest.get("artifacts", manifest.get("files"))
    _require(isinstance(entries, list) and entries, f"external manifest inventory is invalid: {root}")
    seen: set[str] = set()
    total_bytes = 0
    for entry in entries:
        _require(type(entry) is dict, "external manifest entry must be an object")
        relative = str(entry.get("path", ""))
        candidate = Path(relative)
        _require(relative and not candidate.is_absolute() and ".." not in candidate.parts, "external manifest path is unsafe")
        _require(relative not in seen, "external manifest path is duplicated")
        seen.add(relative)
        path = root / candidate
        _require(path.is_file(), f"external artifact is absent: {relative}")
        expected_bytes = int(entry.get("bytes", entry.get("size", -1)))
        _require(path.stat().st_size == expected_bytes, f"external artifact byte mismatch: {relative}")
        _require(sha256_file(path) == str(entry.get("sha256")), f"external artifact hash mismatch: {relative}")
        total_bytes += expected_bytes
    _require(int(manifest.get("artifact_count", -1)) == len(entries), "external manifest count mismatch")
    _require(int(manifest.get("artifact_bytes", -1)) == total_bytes, "external manifest byte total mismatch")
    return {
        "root": str(root.resolve()),
        "manifest_sha256": expected_manifest_sha256,
        "tree_digest": expected_tree_digest,
        "artifact_count": len(entries),
        "artifact_bytes": total_bytes,
    }


def _seal_manifest(run_dir: Path) -> dict[str, Any]:
    rows = _manifest_rows(run_dir)
    manifest = {
        "schema": VERSION + "-artifact-manifest",
        "artifact_count": len(rows),
        "artifact_bytes": sum(int(row["bytes"]) for row in rows),
        "tree_digest": _tree_digest(rows),
        "files": rows,
    }
    _write_json(run_dir / "artifact_manifest.json", manifest)
    sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    sums += f"{sha256_file(run_dir / 'artifact_manifest.json')}  artifact_manifest.json\n"
    _atomic_bytes(run_dir / "SHA256SUMS.txt", sums.encode("utf-8"))
    return manifest


@dataclass(frozen=True)
class ResourceBudget:
    max_active_seconds: float = MAX_ACTIVE_SECONDS
    max_storage_bytes: int = int(MAX_STORAGE_MIB * 1024 * 1024)
    max_cuda_fraction: float = MAX_CUDA_FRACTION
    reserve_seconds: float = 30.0
    maximum_quantum_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.max_active_seconds) <= MAX_ACTIVE_SECONDS):
            raise ValueError(f"max_active_seconds must be in (0,{MAX_ACTIVE_SECONDS}]")
        if not (0 < int(self.max_storage_bytes) <= int(MAX_STORAGE_MIB * 1024 * 1024)):
            raise ValueError("max_storage_bytes exceeds the frozen 100 MiB envelope")
        if not (0.0 < float(self.max_cuda_fraction) <= MAX_CUDA_FRACTION):
            raise ValueError(f"max_cuda_fraction must be in (0,{MAX_CUDA_FRACTION}]")
        if not (0.0 <= float(self.reserve_seconds) <= float(self.max_active_seconds)):
            raise ValueError("reserve_seconds is invalid")
        if not (0.0 < float(self.maximum_quantum_seconds) <= MAX_QUANTUM_SECONDS):
            raise ValueError("maximum_quantum_seconds exceeds the frozen envelope")


@dataclass
class RowResult:
    row: str
    anchors: np.ndarray
    labels: np.ndarray
    path_ids: np.ndarray
    telemetry: list[dict[str, Any]]
    root_seed: int
    scientific_digest: str = ""
    anchor_steps: np.ndarray | None = None

    @property
    def endpoints(self) -> np.ndarray:
        return self.anchors[-1]


class ResourceGovernor:
    def __init__(self, run_dir: str | Path, budget: ResourceBudget, *, device: str | torch.device) -> None:
        self.run_dir = Path(run_dir)
        self.budget = budget
        self.device = torch.device(device)
        self.active_seconds = 0.0
        self.events: list[dict[str, Any]] = []
        self.failed_admission: dict[str, Any] | None = None
        self._open: dict[str, float] = {}

    @classmethod
    def rehydrate(
        cls,
        run_dir: str | Path,
        *,
        device: str | torch.device,
        recover_interrupted: bool = False,
    ) -> "ResourceGovernor":
        root = Path(run_dir)
        ledger = _read_json(root / "resource_ledger.json")
        budget_payload = ledger.get("budget")
        _require(type(budget_payload) is dict, "resource ledger budget is absent")
        budget = ResourceBudget(**budget_payload)
        governor = cls(root, budget, device=device)
        governor.active_seconds = float(ledger.get("active_seconds", 0.0))
        governor.events = list(ledger.get("events", []))
        governor.failed_admission = ledger.get("failed_admission")
        _require(governor.failed_admission is None, "resource ledger contains a failed admission")
        open_events = list(ledger.get("open_events", []))
        if open_events:
            _require(recover_interrupted, "resource ledger has an unresolved open event")
            for kind in open_events:
                admissions = [
                    event
                    for event in governor.events
                    if event.get("event") == "admit" and event.get("kind") == kind
                ]
                completions = [
                    event
                    for event in governor.events
                    if event.get("event") in {"complete", "failed-complete", "interrupted-close"}
                    and event.get("kind") == kind
                ]
                _require(len(admissions) == len(completions) + 1, f"unmatched interrupted resource event is malformed: {kind}")
                charged = float(admissions[-1]["predicted_seconds"])
                governor.active_seconds += charged
                governor.events.append(
                    {
                        "event": "interrupted-close",
                        "kind": kind,
                        "charged_predicted_seconds": charged,
                        "active_seconds_after": governor.active_seconds,
                        "recorded_at": _utc_now(),
                    }
                )
        _require(math.isfinite(governor.active_seconds) and governor.active_seconds >= 0.0, "resource ledger active time is invalid")
        if open_events:
            governor.write()
        return governor

    def _cuda_receipt(self) -> tuple[int, int, float]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0, 0, 0.0
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        allocated = int(torch.cuda.max_memory_allocated(index))
        total = int(torch.cuda.get_device_properties(index).total_memory)
        return allocated, total, (0.0 if total <= 0 else allocated / total)

    def write(self) -> None:
        _write_json(
            self.run_dir / "resource_ledger.json",
            {
                "schema": VERSION + "-resource-ledger",
                "budget": dataclasses.asdict(self.budget),
                "active_seconds": self.active_seconds,
                "events": self.events,
                "failed_admission": self.failed_admission,
                "open_events": sorted(self._open),
            },
        )

    def admit(self, kind: str, *, predicted_seconds: float, predicted_next_bytes: int,
              reserve_remaining_seconds: float | None = None) -> dict[str, Any]:
        if kind in self._open:
            raise IntegrityFailure(f"resource event is already open: {kind}")
        predicted_seconds = float(predicted_seconds)
        predicted_next_bytes = int(predicted_next_bytes)
        reserve = self.budget.reserve_seconds if reserve_remaining_seconds is None else float(reserve_remaining_seconds)
        allocated, total, fraction = self._cuda_receipt()
        storage = _storage_bytes(self.run_dir)
        checks = {
            "active": self.active_seconds + predicted_seconds + reserve <= self.budget.max_active_seconds,
            "storage": storage + predicted_next_bytes <= self.budget.max_storage_bytes,
            "cuda": fraction <= self.budget.max_cuda_fraction,
            "quantum": predicted_seconds <= self.budget.maximum_quantum_seconds or reserve == 0.0,
        }
        receipt = {
            "kind": kind,
            "predicted_seconds": predicted_seconds,
            "predicted_next_bytes": predicted_next_bytes,
            "active_seconds_before": self.active_seconds,
            "reserve_remaining_seconds": reserve,
            "storage_bytes_before": storage,
            "cuda_allocated_bytes": allocated,
            "cuda_total_bytes": total,
            "cuda_fraction": fraction,
            "checks": checks,
            "passed": int(all(checks.values())),
        }
        if not all(checks.values()):
            self.failed_admission = receipt
            self.write()
            raise ResourceStop(f"resource admission failed for {kind}: {checks}")
        self._open[kind] = time.perf_counter()
        self.events.append({**receipt, "event": "admit", "recorded_at": _utc_now()})
        self.write()
        return receipt

    def complete(self, kind: str, *, candidate_transitions: int = 0,
                 model_evaluations: int = 0) -> dict[str, Any]:
        if kind not in self._open:
            raise IntegrityFailure(f"resource event is not open: {kind}")
        elapsed = time.perf_counter() - self._open.pop(kind)
        self.active_seconds += elapsed
        allocated, total, fraction = self._cuda_receipt()
        receipt = {
            "event": "complete",
            "kind": kind,
            "elapsed_seconds": elapsed,
            "active_seconds_after": self.active_seconds,
            "storage_bytes_after": _storage_bytes(self.run_dir),
            "cuda_allocated_bytes": allocated,
            "cuda_total_bytes": total,
            "cuda_fraction": fraction,
            "candidate_transitions": int(candidate_transitions),
            "model_evaluations": int(model_evaluations),
            "recorded_at": _utc_now(),
        }
        self.events.append(receipt)
        self.write()
        post_checks = {
            "quantum": elapsed <= self.budget.maximum_quantum_seconds,
            "active": self.active_seconds <= self.budget.max_active_seconds,
            "storage": int(receipt["storage_bytes_after"]) <= self.budget.max_storage_bytes,
            "cuda": float(receipt["cuda_fraction"]) <= self.budget.max_cuda_fraction,
        }
        if not all(post_checks.values()):
            self.failed_admission = {
                "kind": kind,
                "phase": "post-completion",
                "checks": post_checks,
                "receipt": receipt,
                "passed": 0,
            }
            self.write()
            raise ResourceStop(f"resource post-completion check failed for {kind}: {post_checks}")
        return receipt

    def close_open_as_failed(self) -> None:
        """Conservatively charge every open event before terminalizing a failure."""
        for kind, began in list(self._open.items()):
            elapsed = max(0.0, time.perf_counter() - began)
            self.active_seconds += elapsed
            self.events.append(
                {
                    "event": "failed-complete",
                    "kind": kind,
                    "elapsed_seconds": elapsed,
                    "active_seconds_after": self.active_seconds,
                    "storage_bytes_after": _storage_bytes(self.run_dir),
                    "recorded_at": _utc_now(),
                }
            )
            self._open.pop(kind, None)
        self.write()


STAGE_ORDER = (
    "initialize_and_bind",
    "checkpoint_extract",
    "data_and_inventory",
    "preflight",
    "teacher_row",
    "null_row",
    "learned_row",
    "population_seal",
    "scoring",
    "review_prepare",
    "machine_terminalization",
    "human_review_terminalization",
)


def _stage_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "stage_ledger.json"
    if not path.is_file():
        return []
    ledger = _read_json(path)
    events = ledger.get("events")
    _require(isinstance(events, list), "stage ledger events are invalid")
    return list(events)


def _record_stage(run_dir: Path, stage: str) -> None:
    _require(stage in STAGE_ORDER, f"unknown stage: {stage}")
    events = _stage_events(run_dir)
    completed = [str(event.get("stage")) for event in events if event.get("state") == "completed"]
    _require(stage not in completed, f"stage already completed: {stage}")
    if completed:
        _require(STAGE_ORDER.index(stage) > STAGE_ORDER.index(completed[-1]), "stage order regressed")
    events.append({"stage": stage, "state": "completed", "recorded_at": _utc_now()})
    _write_json(
        run_dir / "stage_ledger.json",
        {"schema": VERSION + "-stage-ledger", "events": events},
    )


def safe_extract_legacy_checkpoint(
    checkpoint_path: str | Path,
    clean_state_path: str | Path | None = None,
    *,
    expected_bytes: int = LEGACY_CHECKPOINT_BYTES,
    expected_sha256: str = LEGACY_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    clean_path = None if clean_state_path is None else Path(clean_state_path)
    stat = checkpoint.stat()
    _require(stat.st_size == int(expected_bytes), "legacy checkpoint byte size mismatch")
    digest = sha256_file(checkpoint)
    _require(digest == str(expected_sha256), "legacy checkpoint SHA-256 mismatch")
    _require(np.__version__ == EXPECTED_NUMPY_VERSION, f"NumPy version mismatch: {np.__version__}")
    _require(torch.__version__ == EXPECTED_TORCH_VERSION, f"PyTorch version mismatch: {torch.__version__}")
    safe_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType,
        np.dtypes.Float64DType,
    ]
    ambient_safe_globals = list(torch.serialization.get_safe_globals())
    expected_load_authorities = _safe_global_authority_counter(safe_globals)
    expected_ambient_authorities = _safe_global_authority_counter(ambient_safe_globals)
    torch.serialization.clear_safe_globals()
    try:
        with torch.serialization.safe_globals(safe_globals):
            active = list(torch.serialization.get_safe_globals())
            _require(
                _safe_global_authority_counter(active) == expected_load_authorities,
                "legacy load safe-global scope is not exactly the five NumPy authorities",
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    finally:
        torch.serialization.clear_safe_globals()
        if ambient_safe_globals:
            torch.serialization.add_safe_globals(ambient_safe_globals)
        restored = list(torch.serialization.get_safe_globals())
        _require(
            _safe_global_authority_counter(restored) == expected_ambient_authorities,
            "torch serialization ambient safe globals were not restored exactly",
        )
    config, state = _validate_checkpoint_payload(payload)
    model = DirectFluxUNet(config, base_channels=48, num_classes=10)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(parameter_count == EXPECTED_PARAMETER_COUNT, "legacy model parameter count mismatch")
    clean_hash = None
    clean_bytes = None
    if clean_path is not None:
        clean = OrderedDict((name, tensor.detach().cpu().contiguous().clone()) for name, tensor in state.items())
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = clean_path.with_name(clean_path.name + ".tmp")
        torch.save(clean, temporary)
        _replace_with_retry(temporary, clean_path)
        clean_hash = sha256_file(clean_path)
        clean_bytes = clean_path.stat().st_size
        reloaded = torch.load(clean_path, map_location="cpu", weights_only=True)
        _require(isinstance(reloaded, (dict, OrderedDict)), "clean state is not a mapping")
        model.load_state_dict(reloaded, strict=True)
        _require(all(torch.equal(clean[name], reloaded[name]) for name in clean), "clean state changed on reload")
    config_payload = payload["config"]
    return {
        "schema": VERSION + "-legacy-checkpoint-receipt",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_bytes": stat.st_size,
        "checkpoint_sha256": digest,
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "weights_only": True,
        "safe_globals": [f"{item.__module__}.{item.__qualname__}" for item in safe_globals],
        "payload_keys": sorted(payload),
        "config": _jsonable(config_payload),
        "config_semantic_sha256": _sha256_bytes(_canonical_json_bytes(config_payload)),
        "historical_selection_fields": {
            "sample_rejection_factor": int(config_payload["sample_rejection_factor"]),
            "sample_selection_metric": str(config_payload["sample_selection_metric"]),
        },
        "replay_policy": {
            "generated_candidates_per_path": 1,
            "selector": None,
            "all_candidates_retained": 1,
        },
        "tensor_count": len(state),
        "parameter_count": parameter_count,
        "clean_state_path": None if clean_path is None else str(clean_path.resolve()),
        "clean_state_bytes": clean_bytes,
        "clean_state_sha256": clean_hash,
    }


def _safe_global_authority_counter(
    values: Sequence[Callable[..., Any] | tuple[Callable[..., Any], str]],
) -> Counter[tuple[int, str]]:
    """Canonicalize PyTorch's unordered user-safe-global registry semantically.

    PyTorch 2.11 stores this registry as a set, so ``get_safe_globals()`` does
    not promise list order.  The actual authority is the callable identity plus
    its serialized fully-qualified name (explicit for tuple entries, otherwise
    derived exactly as PyTorch does).  A Counter also preserves the unlikely but
    meaningful case where two distinct registry entries encode the same authority.
    """

    authorities: Counter[tuple[int, str]] = Counter()
    for entry in values:
        if isinstance(entry, tuple):
            _require(
                len(entry) == 2
                and callable(entry[0])
                and isinstance(entry[1], str)
                and bool(entry[1]),
                "torch serialization safe-global tuple is invalid",
            )
            target, serialized_name = entry
        else:
            _require(callable(entry), "torch serialization safe global is not callable")
            target = entry
            module = getattr(target, "__module__", None)
            qualname = getattr(target, "__qualname__", None)
            _require(
                isinstance(module, str)
                and bool(module)
                and isinstance(qualname, str)
                and bool(qualname),
                "torch serialization safe global has no qualified name",
            )
            serialized_name = f"{module}.{qualname}"
        authorities[(id(target), serialized_name)] += 1
    return authorities


def _load_legacy_checkpoint(
    checkpoint_path: str | Path, clean_state_path: str | Path | None = None
) -> dict[str, Any]:
    return safe_extract_legacy_checkpoint(checkpoint_path, clean_state_path)


def _validate_checkpoint_payload(payload: Any) -> tuple[DirectFluxMNISTConfig, Mapping[str, torch.Tensor]]:
    _require(type(payload) is dict, "legacy checkpoint top level must be an exact dict")
    _require(frozenset(payload) == EXPECTED_CHECKPOINT_KEYS, "legacy checkpoint key set mismatch")
    config_payload = payload.get("config")
    _require(type(config_payload) is dict, "legacy checkpoint config must be an exact dict")
    config_hash = _sha256_bytes(_canonical_json_bytes(config_payload))
    _require(config_hash == LEGACY_CONFIG_SHA256, "legacy config semantic SHA-256 mismatch")
    _validate_deserialized_tree(payload)
    config = DirectFluxMNISTConfig(**config_payload)
    state = payload.get("model_state_dict")
    _require(isinstance(state, (dict, OrderedDict)), "model_state_dict must be a mapping")
    _require(len(state) == EXPECTED_STATE_TENSORS, "legacy state tensor count mismatch")
    expected = DirectFluxUNet(config, base_channels=48, num_classes=10).state_dict()
    _require(list(state) == list(expected), "legacy state tensor name/order mismatch")
    for name, tensor in state.items():
        _require(type(tensor) is torch.Tensor, f"state value {name} is not an exact tensor")
        _require(tensor.dtype == torch.float32, f"state tensor {name} is not float32")
        _require(tuple(tensor.shape) == tuple(expected[name].shape), f"state tensor {name} shape mismatch")
        _require(bool(torch.isfinite(tensor).all()), f"state tensor {name} is nonfinite")
    return config, state


def _validate_deserialized_tree(value: Any, location: str = "payload") -> None:
    if value is None or isinstance(value, (bool, int, float, str, np.integer, np.floating, np.dtype)):
        return
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, np.ndarray):
        _require(value.dtype.kind in "biuf", f"{location} ndarray dtype is forbidden")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(isinstance(key, (str, int)), f"{location} mapping key type is forbidden")
            _validate_deserialized_tree(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_deserialized_tree(item, f"{location}[{index}]")
        return
    raise IntegrityFailure(f"{location} deserialized forbidden type {type(value).__module__}.{type(value).__qualname__}")


def read_mnist_development_prefix(
    arff_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read only rows 0..59999; whole-file hashing is authority-only."""
    path = Path(arff_path)
    digest = sha256_file(path)
    _require(digest == MNIST_ARFF_SHA256, "MNIST ARFF SHA-256 mismatch")
    with path.open("r", encoding="utf-8") as handle:
        images, labels, audit = _parse_mnist_arff_prefix(handle, stop=VALIDATION_STOP)
    return images, labels, {
        **audit,
        "arff_path": str(path.resolve()),
        "arff_sha256": digest,
        "full_file_read_purpose": "sha256-only",
        "train_slice": [TRAIN_START, TRAIN_STOP],
        "validation_slice": [VALIDATION_START, VALIDATION_STOP],
        "test_slice": [60_000, 70_000],
    }


def _parse_mnist_arff_prefix(
    lines: Iterable[str], *, stop: int = VALIDATION_STOP
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Parse ``stop`` content rows and never request content row ``stop``."""
    iterator = iter(lines)
    images: list[np.ndarray] = []
    labels: list[int] = []
    in_data = False
    line_number = 0
    content_row = 0
    while True:
        try:
            line = next(iterator)
        except StopIteration:
            break
        line_number += 1
        text = line.strip()
        if not in_data:
            if text.upper() == "@DATA":
                in_data = True
            continue
        if not text or text.startswith("%"):
            continue
        fields = text.split(",")
        _require(len(fields) == 785, f"ARFF content row {content_row} must have 785 fields")
        try:
            values = np.asarray(fields, dtype=np.float64)
        except ValueError as error:
            raise IntegrityFailure(f"ARFF content row {content_row} is nonnumeric") from error
        _require(bool(np.all(np.isfinite(values))), f"ARFF content row {content_row} is nonfinite")
        _require(bool(np.all(values == np.rint(values))), f"ARFF content row {content_row} is nonintegral")
        pixels = values[:784]
        label = int(values[-1])
        _require(bool(np.all((pixels >= 0) & (pixels <= 255))), f"ARFF content row {content_row} has invalid pixels")
        _require(0 <= label <= 9, f"ARFF content row {content_row} has invalid label")
        images.append(pixels.astype(np.uint8).reshape(28, 28))
        labels.append(label)
        content_row += 1
        # This check is deliberately inside the just-consumed row.  The next
        # iterator element (terminal row 60000) is never fetched.
        if content_row == int(stop):
            break
    _require(in_data, "ARFF has no @DATA marker")
    _require(content_row == int(stop), f"ARFF has only {content_row} content rows")
    return (
        np.stack(images),
        np.asarray(labels, dtype=np.int64),
        {
            "content_rows_parsed": content_row,
            "last_content_row_index": content_row - 1,
            "terminal_content_rows_parsed": max(0, content_row - 60_000),
            "last_file_line_number": line_number,
        },
    )


def derive_mass_to_uint8_authority(train_images: np.ndarray) -> dict[str, Any]:
    images = np.asarray(train_images)
    _require(images.dtype == np.uint8 and images.shape == (TRAIN_STOP, 28, 28), "transform requires uint8 training rows [0,55000)")
    sums = np.sort(images.reshape(len(images), -1).sum(axis=1, dtype=np.int64))
    lower = int(sums[(len(sums) - 1) // 2])
    upper = int(sums[len(sums) // 2])
    _require((lower, upper) == (25_470, 25_472), "training ink central sums mismatch")
    scale = np.float64.fromhex(MASS_SCALE_HEX)
    _require(scale == np.float64(MASS_SCALE_NUMERATOR) / np.float64(MASS_SCALE_DENOMINATOR), "mass scale hex/rational mismatch")
    return {
        "schema": VERSION + "-mass-to-uint8",
        "derivation_slice": [TRAIN_START, TRAIN_STOP],
        "development_slice": [TRAIN_START, TRAIN_STOP],
        "central_sums": [lower, upper],
        "numerator": MASS_SCALE_NUMERATOR,
        "denominator": MASS_SCALE_DENOMINATOR,
        "decimal": float(scale),
        "float_hex": scale.hex(),
        "formula": "rint(255 * clip((25471/255) * mass, 0, 1))",
    }


def _derive_mass_transform(train_images: np.ndarray) -> dict[str, Any]:
    return derive_mass_to_uint8_authority(train_images)


def mass_to_uint8(masses: np.ndarray, authority: Mapping[str, Any] | None = None) -> np.ndarray:
    if authority is not None:
        _require(authority.get("numerator") == MASS_SCALE_NUMERATOR, "mass transform numerator mismatch")
        _require(authority.get("denominator") == MASS_SCALE_DENOMINATOR, "mass transform denominator mismatch")
        _require(authority.get("float_hex") == MASS_SCALE_HEX, "mass transform float hex mismatch")
    array = np.asarray(masses)
    _require(array.dtype.kind == "f" and array.shape[-1] == 784, "masses must be floating (...,784)")
    _require(bool(np.all(np.isfinite(array))), "masses must be finite")
    scale = np.float64.fromhex(MASS_SCALE_HEX)
    rendered = np.rint(np.float64(255.0) * np.clip(scale * array.astype(np.float64), 0.0, 1.0)).astype(np.uint8)
    return rendered.reshape(*array.shape[:-1], 28, 28)


def _mass_to_uint8(masses: np.ndarray) -> np.ndarray:
    return mass_to_uint8(masses)


def build_path_inventory() -> dict[str, np.ndarray]:
    indices = np.arange(PATH_COUNT, dtype=np.int64)
    labels = indices // PATHS_PER_CLASS
    within = indices % PATHS_PER_CLASS
    return {
        "path_ids": np.asarray([f"{PATH_PREFIX}{index:03d}" for index in indices], dtype=np.str_),
        "path_indices": indices,
        "requested_labels": labels,
        "labels": labels.copy(),
        "within_class_indices": within,
        "within_class": within.copy(),
        "source_seeds": np.asarray([SOURCE_SEED_BASE + int(index) for index in indices], dtype=np.uint64),
        "generated_candidates_per_path": np.ones(PATH_COUNT, dtype=np.int64),
        "retained": np.ones(PATH_COUNT, dtype=np.int64),
    }


def _path_inventory() -> dict[str, np.ndarray]:
    return build_path_inventory()


def build_start_bank(
    config: DirectFluxMNISTConfig,
    inventory: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    roles = build_path_inventory() if inventory is None else inventory
    seeds = np.asarray(roles["source_seeds"], dtype=np.uint64)
    _require(seeds.shape == (PATH_COUNT,), "source seed inventory must contain exactly 160 paths")
    rows = []
    for seed in seeds.tolist():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            source = _sample_source_batch_torch(1, config, device=torch.device("cpu"), dtype=torch.float32)
        rows.append(source.masses.detach().cpu().numpy().astype(np.float32, copy=False)[0])
    result = np.stack(rows).astype(np.float32, copy=False)
    _require(result.shape == (PATH_COUNT, int(config.grid_size) ** 2), "start bank shape mismatch")
    _require(bool(np.all(np.isfinite(result))) and float(result.min()) >= 0.0, "start bank health failed")
    _require(float(np.max(np.abs(result.sum(axis=1, dtype=np.float64) - 1.0))) <= 2e-6, "start bank mass error")
    _require(len({_hash_array(row) for row in result}) == PATH_COUNT, "start bank sources are not unique")
    return result


def _build_start_bank(config: DirectFluxMNISTConfig) -> np.ndarray:
    return build_start_bank(config)


def build_teacher_target_bank(
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    inventory: Mapping[str, np.ndarray] | None = None,
    *,
    mass_floor: float = 1e-8,
) -> dict[str, np.ndarray]:
    roles = build_path_inventory() if inventory is None else inventory
    images = np.asarray(validation_images)
    labels = np.asarray(validation_labels, dtype=np.int64)
    _require(images.dtype == np.uint8 and images.shape == (5_000, 28, 28), "validation images must be rows [55000,60000)")
    _require(labels.shape == (5_000,), "validation labels shape mismatch")
    selected: list[int] = []
    for digit in range(10):
        matches = np.flatnonzero(labels == digit)
        _require(len(matches) >= PATHS_PER_CLASS, f"validation class {digit} has fewer than 16 examples")
        selected.extend(matches[:PATHS_PER_CLASS].tolist())
    local_ids = np.asarray(selected, dtype=np.int64)
    target_images = images[local_ids]
    target_labels = labels[local_ids]
    requested = np.asarray(roles["requested_labels"], dtype=np.int64)
    _require(np.array_equal(target_labels, requested), "teacher target labels do not match requested labels")
    flat = target_images.reshape(PATH_COUNT, -1).astype(np.float32)
    flat = np.maximum(flat, np.float32(mass_floor))
    masses = (flat / flat.sum(axis=1, keepdims=True)).astype(np.float32)
    return {
        "masses": masses,
        "images_uint8": target_images.copy(),
        "requested_labels": target_labels.copy(),
        "validation_local_ids": local_ids,
        "arff_global_row_ids": local_ids + VALIDATION_START,
        "path_ids": np.asarray(roles["path_ids"], dtype=np.str_).copy(),
    }


def _select_teacher_targets(validation_images: np.ndarray, validation_labels: np.ndarray) -> dict[str, np.ndarray]:
    return build_teacher_target_bank(validation_images, validation_labels)


def _scientific_row_digest(anchors: np.ndarray, telemetry: Sequence[Mapping[str, Any]]) -> str:
    excluded = {"elapsed_seconds", "cuda_allocated_bytes", "cuda_peak_allocated_bytes"}
    deterministic = [{key: value for key, value in row.items() if key not in excluded} for row in telemetry]
    digest = hashlib.sha256()
    digest.update(_hash_array(anchors).encode("ascii"))
    digest.update(_canonical_json_bytes(deterministic))
    return digest.hexdigest()


def _validate_row_inputs(
    starts: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    row: str,
    model: DirectFluxUNet | None,
    targets: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    starts_np = np.asarray(starts)
    labels_np = np.asarray(labels, dtype=np.int64)
    _require(starts_np.dtype == np.float32 and starts_np.ndim == 2, "starts must be float32 (B,N)")
    _require(starts_np.shape[1] == int(config.grid_size) ** 2, "starts have wrong grid size")
    _require(labels_np.shape == (len(starts_np),), "labels shape mismatch")
    _require(bool(np.all(np.isfinite(starts_np))) and float(starts_np.min()) >= 0.0, "starts are invalid")
    _require(bool(np.all((labels_np >= 0) & (labels_np <= 9))), "labels are invalid")
    if row == "null":
        _require(model is None and targets is None, "null row cannot receive model or targets")
    elif row == "teacher":
        _require(model is None and targets is not None, "teacher row requires targets and forbids model")
        target_np = np.asarray(targets)
        _require(target_np.dtype == np.float32 and target_np.shape == starts_np.shape, "teacher targets are invalid")
    elif row == "learned":
        _require(model is not None and targets is None, "learned row requires model and forbids targets")
        _require(model.config == config, "learned model config mismatch")
    else:
        raise IntegrityFailure(f"unknown row: {row}")
    return starts_np, labels_np


def run_row(
    starts: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    row: str,
    root_seed: int,
    device: str | torch.device,
    model: DirectFluxUNet | None = None,
    targets: np.ndarray | None = None,
    path_ids: Sequence[str] | np.ndarray | None = None,
    num_steps: int = OUTER_STEPS,
    schedule_steps: int | None = None,
    anchors: Sequence[int] = ANCHORS,
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> RowResult:
    starts_np, labels_np = _validate_row_inputs(starts, labels, config, row=row, model=model, targets=targets)
    count = len(starts_np)
    path_ids_np = (
        np.asarray([f"path-{index:03d}" for index in range(count)], dtype=np.str_)
        if path_ids is None
        else np.asarray(path_ids, dtype=np.str_)
    )
    _require(path_ids_np.shape == (count,), "path IDs shape mismatch")
    _require(len(set(path_ids_np.tolist())) == count, "path IDs are not unique")
    steps = int(num_steps)
    schedule_count = steps if schedule_steps is None else int(schedule_steps)
    anchor_steps = tuple(int(value) for value in anchors)
    _require(steps > 0, "num_steps must be positive")
    _require(schedule_count >= steps, "schedule_steps must be at least num_steps")
    _require(anchor_steps and anchor_steps[0] == 0 and anchor_steps[-1] == steps, "anchors must include 0 and num_steps")
    _require(tuple(sorted(set(anchor_steps))) == anchor_steps, "anchors must be unique and sorted")
    device_t = torch.device(device)
    states = torch.as_tensor(starts_np, dtype=torch.float32, device=device_t).clone()
    source_condition = states.clone()
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device_t)
    targets_t = None if targets is None else torch.as_tensor(targets, dtype=torch.float32, device=device_t)
    if model is not None:
        model.to(device_t)
        model.eval()
    horizon = natural_horizon(config)
    dt = horizon / float(schedule_count)
    saved: list[np.ndarray] = [states.detach().cpu().numpy().astype(np.float32, copy=True)]
    saved_steps: list[int] = [0]
    telemetry: list[dict[str, Any]] = []
    devices: list[int] = []
    if device_t.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA device requested but unavailable")
        devices = [device_t.index if device_t.index is not None else torch.cuda.current_device()]
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(root_seed))
        if device_t.type == "cuda":
            torch.cuda.manual_seed_all(int(root_seed))
        for outer in range(steps):
            if device_t.type == "cuda":
                torch.cuda.synchronize(device_t)
            began = time.perf_counter()
            tau_value = max(horizon - float(outer) * dt, 0.0)
            state_before = states
            attempts: list[dict[str, Any]] = []
            accepted_components: list[dict[str, float]] = []
            accepted_substeps = 0
            accepted_clipped = 0
            accepted_proposed = 0
            schedule = (1, 2, 4) if bool(config.adaptive_sampling) else (1,)
            for substeps in schedule:
                if substeps > int(config.max_substeps):
                    break
                local = state_before.clone()
                clipped_total = 0
                proposed_total = 0
                components: list[dict[str, float]] = []
                sub_dt = dt / float(substeps)
                for sub_index in range(substeps):
                    sub_tau = max(tau_value - float(sub_index) * sub_dt, 0.0)
                    if row == "null":
                        flux = torch.zeros(
                            (count, 2, int(config.grid_size), int(config.grid_size)),
                            dtype=torch.float32,
                            device=device_t,
                        )
                    elif row == "teacher":
                        assert targets_t is not None
                        remaining = max(sub_tau, float(config.min_tau_fraction) * horizon)
                        velocity = (targets_t - local) / float(remaining)
                        velocity = velocity - velocity.mean(dim=1, keepdim=True)
                        total_flux = poisson_flux_from_velocity_torch(velocity, grid_size=int(config.grid_size))
                        flux = total_flux - float(config.free_weight) * free_drift_flux_torch(local, config)
                    else:
                        assert model is not None
                        tau = torch.full((count,), sub_tau, dtype=local.dtype, device=device_t)
                        context = (
                            torch.amp.autocast(device_type="cuda", enabled=True)
                            if device_t.type == "cuda"
                            else nullcontext()
                        )
                        with context:
                            flux = model.predict_flux(tau, local, labels_t, source_masses=source_condition)
                        flux = flux.float()
                    components.append(
                        step_component_rms_torch(
                            local,
                            flux,
                            sub_dt,
                            config,
                            free_weight=float(config.free_weight),
                            noise_weight=float(config.noise_weight),
                            learned_weight=1.0,
                        )
                    )
                    local, clipped, proposed = eulerian_flux_step_torch(
                        local,
                        flux,
                        sub_dt,
                        config,
                        deterministic=False,
                        free_weight=float(config.free_weight),
                        noise_weight=float(config.noise_weight),
                        learned_weight=1.0,
                    )
                    clipped_total += int(clipped)
                    proposed_total += int(proposed)
                clip_fraction = 0.0 if proposed_total == 0 else clipped_total / proposed_total
                attempts.append(
                    {
                        "substeps": substeps,
                        "clipped": clipped_total,
                        "proposed": proposed_total,
                        "clipping_fraction": clip_fraction,
                    }
                )
                if (
                    not bool(config.adaptive_sampling)
                    or clip_fraction <= float(config.clip_target)
                    or substeps >= int(config.max_substeps)
                ):
                    states = local
                    accepted_substeps = substeps
                    accepted_clipped = clipped_total
                    accepted_proposed = proposed_total
                    accepted_components = components
                    break
            _require(accepted_substeps in (1, 2, 4), "adaptive sampler accepted no attempt")
            finite = bool(torch.isfinite(states).all())
            minimum = float(states.min().detach().cpu())
            maximum = float(states.max().detach().cpu())
            mass_error = float(torch.max(torch.abs(states.sum(dim=1) - 1.0)).detach().cpu())
            if not (finite and minimum >= 0.0 and mass_error <= 2e-6):
                error = IntegrityFailure(f"{row} numerical health failed at step {outer + 1}")
                valid_saved = list(saved)
                valid_steps = list(saved_steps)
                last_valid_step = outer
                if valid_steps[-1] != last_valid_step:
                    valid_saved.append(state_before.detach().cpu().numpy().astype(np.float32, copy=True))
                    valid_steps.append(last_valid_step)
                valid_anchors = np.stack(valid_saved).astype(np.float32, copy=False)
                setattr(
                    error,
                    "partial_row_result",
                    RowResult(
                        row=row,
                        anchors=valid_anchors,
                        labels=labels_np.copy(),
                        path_ids=path_ids_np.copy(),
                        telemetry=list(telemetry),
                        root_seed=int(root_seed),
                        scientific_digest=_scientific_row_digest(valid_anchors, telemetry),
                        anchor_steps=np.asarray(valid_steps, dtype=np.int64),
                    ),
                )
                raise error
            means = {
                key: float(np.mean([entry[key] for entry in accepted_components])) if accepted_components else 0.0
                for key in ("learned_step_rms", "free_step_rms", "noise_step_rms")
            }
            increment_rms = float((states - state_before).float().square().mean().sqrt().detach().cpu())
            allocated = int(torch.cuda.memory_allocated(device_t)) if device_t.type == "cuda" else 0
            peak = int(torch.cuda.max_memory_allocated(device_t)) if device_t.type == "cuda" else 0
            if device_t.type == "cuda":
                torch.cuda.synchronize(device_t)
            record: dict[str, Any] = {
                "row": row,
                "completed_step": outer + 1,
                "accepted_substeps": accepted_substeps,
                "rejected_attempt_count": len(attempts) - 1,
                "attempts": attempts,
                "accepted_clipped": accepted_clipped,
                "accepted_proposed": accepted_proposed,
                "accepted_clipping_fraction": 0.0 if accepted_proposed == 0 else accepted_clipped / accepted_proposed,
                **means,
                "state_increment_rms": increment_rms,
                "minimum_mass": minimum,
                "maximum_mass": maximum,
                "maximum_mass_error": mass_error,
                "nonfinite_count": 0,
                "elapsed_seconds": time.perf_counter() - began,
                "cuda_allocated_bytes": allocated,
                "cuda_peak_allocated_bytes": peak,
            }
            telemetry.append(record)
            if outer + 1 in anchor_steps:
                saved.append(states.detach().cpu().numpy().astype(np.float32, copy=True))
                saved_steps.append(outer + 1)
            if outer_step_callback is not None:
                callback_value = {
                    "row": row,
                    "completed_step": outer + 1,
                    "state": states,
                    "saved_anchors": saved,
                    "saved_steps": saved_steps,
                    "telemetry": telemetry,
                }
                try:
                    outer_step_callback(callback_value)
                except Exception as error:
                    partial_saved = list(saved)
                    partial_steps = list(saved_steps)
                    if partial_steps[-1] != outer + 1:
                        partial_saved.append(states.detach().cpu().numpy().astype(np.float32, copy=True))
                        partial_steps.append(outer + 1)
                    partial_anchors = np.stack(partial_saved).astype(np.float32, copy=False)
                    setattr(
                        error,
                        "partial_row_result",
                        RowResult(
                            row=row,
                            anchors=partial_anchors,
                            labels=labels_np.copy(),
                            path_ids=path_ids_np.copy(),
                            telemetry=list(telemetry),
                            root_seed=int(root_seed),
                            scientific_digest=_scientific_row_digest(partial_anchors, telemetry),
                            anchor_steps=np.asarray(partial_steps, dtype=np.int64),
                        ),
                    )
                    raise
    anchor_array = np.stack(saved).astype(np.float32, copy=False)
    _require(saved_steps == list(anchor_steps), "row anchors are incomplete")
    return RowResult(
        row=row,
        anchors=anchor_array,
        labels=labels_np.copy(),
        path_ids=path_ids_np.copy(),
        telemetry=telemetry,
        root_seed=int(root_seed),
        scientific_digest=_scientific_row_digest(anchor_array, telemetry),
        anchor_steps=np.asarray(anchor_steps, dtype=np.int64),
    )


def run_null_row(starts: np.ndarray, labels: np.ndarray, config: DirectFluxMNISTConfig, *,
                 root_seed: int = ROW_ROOT_SEEDS["null"], device: str | torch.device = "cpu",
                  path_ids: Sequence[str] | np.ndarray | None = None, num_steps: int = OUTER_STEPS,
                  schedule_steps: int | None = None,
                 anchors: Sequence[int] = ANCHORS,
                 outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None) -> RowResult:
    return run_row(starts, labels, config, row="null", root_seed=root_seed, device=device,
                   path_ids=path_ids, num_steps=num_steps, schedule_steps=schedule_steps, anchors=anchors,
                   outer_step_callback=outer_step_callback)


def run_teacher_row(starts: np.ndarray, labels: np.ndarray, targets: np.ndarray,
                    config: DirectFluxMNISTConfig, *, root_seed: int = ROW_ROOT_SEEDS["teacher"],
                    device: str | torch.device = "cpu",
                    path_ids: Sequence[str] | np.ndarray | None = None, num_steps: int = OUTER_STEPS,
                    schedule_steps: int | None = None,
                    anchors: Sequence[int] = ANCHORS,
                    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None) -> RowResult:
    return run_row(starts, labels, config, row="teacher", root_seed=root_seed, device=device,
                   targets=targets, path_ids=path_ids, num_steps=num_steps, schedule_steps=schedule_steps, anchors=anchors,
                   outer_step_callback=outer_step_callback)


def run_learned_row(starts: np.ndarray, labels: np.ndarray, model: DirectFluxUNet,
                    config: DirectFluxMNISTConfig, *, root_seed: int = ROW_ROOT_SEEDS["learned"],
                    device: str | torch.device = "cpu",
                    path_ids: Sequence[str] | np.ndarray | None = None, num_steps: int = OUTER_STEPS,
                    schedule_steps: int | None = None,
                    anchors: Sequence[int] = ANCHORS,
                    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None) -> RowResult:
    return run_row(starts, labels, config, row="learned", root_seed=root_seed, device=device,
                   model=model, path_ids=path_ids, num_steps=num_steps, schedule_steps=schedule_steps, anchors=anchors,
                   outer_step_callback=outer_step_callback)


def run_cpu_smoke() -> dict[str, Any]:
    """Run the bounded, test-only assembled CPU smoke without external evidence.

    This is deliberately not a miniature production run: it opens no checkpoint,
    ARFF, predecessor, evaluator, run directory, or CUDA context.  It exercises the
    actual null and target-teacher row composition on two synthetic measures and
    repeats the teacher row to bind deterministic scientific bytes.  Timing and
    allocator observations are intentionally absent from the receipt.
    """

    grid_size = 8
    path_count = 2
    outer_steps = 4
    schedule_steps = outer_steps
    labels = np.asarray([0, 1], dtype=np.int64)
    path_ids = np.asarray(["smoke-000", "smoke-001"], dtype=np.str_)
    starts = np.full(
        (path_count, grid_size * grid_size),
        np.float32(1.0 / (grid_size * grid_size)),
        dtype=np.float32,
    )
    targets = np.float32(0.70) * starts
    targets[0, 4 * grid_size + 4] += np.float32(0.30)
    targets[1, 2 * grid_size + 5] += np.float32(0.30)
    targets /= targets.sum(axis=1, keepdims=True, dtype=np.float32)
    config = dataclasses.replace(
        DirectFluxMNISTConfig(),
        grid_size=grid_size,
        num_steps=outer_steps,
        free_weight=0.0,
        noise_weight=0.0,
        adaptive_sampling=False,
        max_substeps=1,
    )
    null = run_null_row(
        starts,
        labels,
        config,
        root_seed=SMOKE_SEED,
        device="cpu",
        path_ids=path_ids,
        num_steps=outer_steps,
        schedule_steps=schedule_steps,
        anchors=(0, 1, 2, 3, 4),
    )
    teacher = run_teacher_row(
        starts,
        labels,
        targets,
        config,
        root_seed=SMOKE_SEED,
        device="cpu",
        path_ids=path_ids,
        num_steps=outer_steps,
        schedule_steps=schedule_steps,
        anchors=(0, 1, 2, 3, 4),
    )
    teacher_replay = run_teacher_row(
        starts,
        labels,
        targets,
        config,
        root_seed=SMOKE_SEED,
        device="cpu",
        path_ids=path_ids,
        num_steps=outer_steps,
        schedule_steps=schedule_steps,
        anchors=(0, 1, 2, 3, 4),
    )
    expected_anchor_steps = np.arange(outer_steps + 1, dtype=np.int64)
    _require(
        all(
            np.array_equal(result.anchor_steps, expected_anchor_steps)
            and len(result.telemetry) == outer_steps
            and [int(entry["completed_step"]) for entry in result.telemetry]
            == list(range(1, outer_steps + 1))
            for result in (null, teacher, teacher_replay)
        ),
        "CPU smoke anchors or telemetry are incomplete",
    )
    _require(
        np.array_equal(null.anchors, np.broadcast_to(starts, null.anchors.shape)),
        "CPU smoke null row was not an exact structural no-op",
    )
    _require(
        np.array_equal(teacher.anchors, teacher_replay.anchors)
        and teacher.scientific_digest == teacher_replay.scientific_digest,
        "CPU smoke teacher replay was not deterministic",
    )
    _require(not np.array_equal(teacher.anchors[-1], starts), "CPU smoke teacher did not move the state")
    centered_velocity = torch.as_tensor(targets - starts, dtype=torch.float32)
    centered_velocity -= centered_velocity.mean(dim=1, keepdim=True)
    orientation_flux = poisson_flux_from_velocity_torch(
        centered_velocity,
        grid_size=grid_size,
    )
    orientation_divergence = flux_divergence_torch(orientation_flux).reshape_as(centered_velocity)
    orientation_max_abs_error = float(
        torch.max(torch.abs(orientation_divergence - centered_velocity)).detach().cpu()
    )
    _require(
        orientation_max_abs_error <= 2.0e-6,
        "CPU smoke teacher flux orientation is inconsistent",
    )
    target_firewall_exact = (
        "targets" not in inspect.signature(run_null_row).parameters
        and "targets" not in inspect.signature(run_learned_row).parameters
        and "targets" in inspect.signature(run_teacher_row).parameters
    )
    _require(target_firewall_exact, "CPU smoke target firewall changed")
    maximum_mass_error = float(
        max(
            np.max(np.abs(null.anchors.sum(axis=2, dtype=np.float64) - 1.0)),
            np.max(np.abs(teacher.anchors.sum(axis=2, dtype=np.float64) - 1.0)),
        )
    )
    _require(
        bool(np.all(np.isfinite(teacher.anchors)))
        and float(teacher.anchors.min()) >= 0.0
        and maximum_mass_error <= 2.0e-6,
        "CPU smoke teacher numerical health failed",
    )
    deterministic_authority = {
        "seed": SMOKE_SEED,
        "grid_size": grid_size,
        "path_ids": path_ids.tolist(),
        "labels": labels.tolist(),
        "starts_sha256": _hash_array(starts),
        "targets_sha256": _hash_array(targets),
        "null_scientific_digest": null.scientific_digest,
        "teacher_scientific_digest": teacher.scientific_digest,
        "teacher_endpoint_sha256": _hash_array(teacher.anchors[-1]),
        "orientation_max_abs_error": orientation_max_abs_error,
        "target_firewall_exact": int(target_firewall_exact),
    }
    return {
        "schema": VERSION + "-cpu-smoke",
        "passed": 1,
        "test_only": 1,
        "production_launched": 0,
        "persisted_artifact_count": 0,
        "device": "cpu",
        "seed": SMOKE_SEED,
        "grid_size": grid_size,
        "path_count": path_count,
        "outer_steps": outer_steps,
        "schedule_steps": schedule_steps,
        "null_noop_exact": 1,
        "teacher_replay_exact": 1,
        "teacher_endpoint_changed": 1,
        "maximum_mass_error": maximum_mass_error,
        "orientation_max_abs_error": orientation_max_abs_error,
        "target_firewall_exact": int(target_firewall_exact),
        "anchors_complete": 1,
        "telemetry_complete": 1,
        "scientific_digest": hashlib.sha256(
            _canonical_json_bytes(deterministic_authority)
        ).hexdigest(),
    }


def resource_projection(*, charged_active_seconds: float, teacher8_seconds: float,
                        null8_seconds: float, learned8_seconds: float,
                        projected_persisted_bytes: int, peak_cuda_fraction: float,
                        budget: ResourceBudget | None = None) -> dict[str, Any]:
    active_budget = ResourceBudget() if budget is None else budget
    values = [charged_active_seconds, teacher8_seconds, null8_seconds, learned8_seconds, peak_cuda_fraction]
    _require(all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values), "resource projection inputs must be finite/nonnegative")
    projected_rows = 1.25 * 32.0 * (float(teacher8_seconds) + float(null8_seconds) + float(learned8_seconds))
    projected_total = float(charged_active_seconds) + projected_rows + float(active_budget.reserve_seconds)
    checks = {
        "active": projected_total <= float(active_budget.max_active_seconds),
        "storage": int(projected_persisted_bytes) <= int(active_budget.max_storage_bytes),
        "cuda": float(peak_cuda_fraction) <= float(active_budget.max_cuda_fraction),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": VERSION + "-resource-projection",
        "charged_active_seconds": float(charged_active_seconds),
        "teacher8_seconds": float(teacher8_seconds),
        "null8_seconds": float(null8_seconds),
        "learned8_seconds": float(learned8_seconds),
        "projected_rows_seconds": projected_rows,
        "terminal_reserve_seconds": float(active_budget.reserve_seconds),
        "projected_total_seconds": projected_total,
        "projected_persisted_bytes": int(projected_persisted_bytes),
        "peak_cuda_fraction": float(peak_cuda_fraction),
        "checks": checks,
        "passed": int(all(checks.values())),
        "stop_reason": None if not failed else "resource projection failed: " + ",".join(failed),
    }


_resource_projection = resource_projection


def _save_row_result(run_dir: Path, result: RowResult, *, partial: bool = False) -> Path:
    name = f"partial_{result.row}.npz" if partial else f"{result.row}.npz"
    path = run_dir / "populations" / name
    _write_npz(
        path,
        anchors=result.anchors.astype(np.float32, copy=False),
        anchor_steps=(
            np.asarray(ANCHORS, dtype=np.int64)
            if result.anchor_steps is None and len(result.anchors) == len(ANCHORS)
            else np.asarray(result.anchor_steps, dtype=np.int64)
        ),
        labels=result.labels.astype(np.int64, copy=False),
        path_ids=result.path_ids.astype(np.str_, copy=False),
        root_seed=np.asarray([result.root_seed], dtype=np.uint64),
        config_sha256=np.asarray([sha256_file(run_dir / "config.json")], dtype=np.str_),
        checkpoint_sha256=np.asarray([LEGACY_CHECKPOINT_SHA256], dtype=np.str_),
        scientific_digest=np.asarray([result.scientific_digest], dtype=np.str_),
        telemetry_json=np.asarray(
            [json.dumps(_jsonable(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False) for record in result.telemetry],
            dtype=np.str_,
        ),
    )
    telemetry_rows = []
    for record in result.telemetry:
        row = dict(record)
        row["attempts"] = json.dumps(row["attempts"], sort_keys=True, separators=(",", ":"))
        telemetry_rows.append(row)
    _write_csv(run_dir / "telemetry" / (f"partial_{result.row}_steps.csv" if partial else f"{result.row}_steps.csv"), telemetry_rows)
    return path


def _load_row_population(path: Path, *, expected_row: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "anchors",
            "anchor_steps",
            "labels",
            "path_ids",
            "root_seed",
            "config_sha256",
            "checkpoint_sha256",
            "scientific_digest",
            "telemetry_json",
        }
        _require(set(archive.files) == required, f"{path} keys mismatch")
        result = {key: archive[key].copy() for key in archive.files}
    anchors = result["anchors"]
    labels = result["labels"]
    path_ids = result["path_ids"]
    _require(anchors.dtype == np.float32 and anchors.shape == (len(ANCHORS), PATH_COUNT, 784), f"{expected_row} anchors mismatch")
    _require(np.array_equal(result["anchor_steps"], np.asarray(ANCHORS, dtype=np.int64)), f"{expected_row} anchor steps mismatch")
    _require(labels.dtype == np.int64 and labels.shape == (PATH_COUNT,), f"{expected_row} labels mismatch")
    _require(path_ids.shape == (PATH_COUNT,) and len(set(path_ids.astype(str).tolist())) == PATH_COUNT, f"{expected_row} path IDs mismatch")
    _require(
        result["config_sha256"].shape == (1,)
        and str(result["config_sha256"][0]) == sha256_file(path.parents[1] / "config.json"),
        f"{expected_row} config binding mismatch",
    )
    _require(
        result["checkpoint_sha256"].shape == (1,)
        and str(result["checkpoint_sha256"][0]) == LEGACY_CHECKPOINT_SHA256,
        f"{expected_row} checkpoint binding mismatch",
    )
    _require(bool(np.all(np.isfinite(anchors))) and float(anchors.min()) >= 0.0, f"{expected_row} population invalid")
    _require(float(np.max(np.abs(anchors.sum(axis=2, dtype=np.float64) - 1.0))) <= 2e-6, f"{expected_row} population mass error")
    telemetry = [json.loads(str(value)) for value in result["telemetry_json"].tolist()]
    _require(len(telemetry) == OUTER_STEPS, f"{expected_row} telemetry must contain 256 outer steps")
    _require([int(record["completed_step"]) for record in telemetry] == list(range(1, OUTER_STEPS + 1)), f"{expected_row} telemetry step inventory mismatch")
    digest = _scientific_row_digest(anchors, telemetry)
    _require(result["scientific_digest"].shape == (1,) and str(result["scientific_digest"][0]) == digest, f"{expected_row} scientific digest mismatch")
    result["telemetry"] = np.asarray(telemetry, dtype=object)
    return result


def seal_populations(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    authority = _read_json(root / "input_bindings" / "mass_to_uint8.json")
    start_seal = _read_json(root / "inventory" / "START_BANK_SEALED.json")
    _require(start_seal["start_bank_sha256"] == sha256_file(root / "inventory" / "start_bank.npz"), "sealed start-bank file changed")
    with np.load(root / "inventory" / "start_bank.npz", allow_pickle=False) as archive:
        starts = archive["starts"].copy()
        sealed_labels = archive["labels"].copy()
        sealed_path_ids = archive["path_ids"].copy()
    _require(starts.dtype == np.float32 and starts.shape == (PATH_COUNT, 784), "sealed start bank shape mismatch")
    rows = {name: _load_row_population(root / "populations" / f"{name}.npz", expected_row=name) for name in ("teacher", "null", "learned")}
    labels = rows["teacher"]["labels"]
    path_ids = rows["teacher"]["path_ids"]
    for name, row in rows.items():
        _require(np.array_equal(row["labels"], labels), f"{name} labels do not match teacher")
        _require(np.array_equal(row["path_ids"], path_ids), f"{name} path IDs do not match teacher")
        _require(np.array_equal(row["anchors"][0], starts), f"{name} anchor zero does not match sealed start bank")
        _require(np.array_equal(row["labels"], sealed_labels), f"{name} labels do not match sealed inventory")
        _require(np.array_equal(row["path_ids"], sealed_path_ids), f"{name} path IDs do not match sealed inventory")
    rendered: dict[str, np.ndarray] = {}
    bindings: dict[str, Any] = {}
    (root / "images").mkdir(parents=True, exist_ok=True)
    for name, row in rows.items():
        uint8 = mass_to_uint8(row["anchors"], authority)
        rendered[name] = uint8
        uint_path = root / "populations" / f"{name}_uint8.npz"
        _write_npz(
            uint_path,
            anchors=uint8,
            anchor_steps=np.asarray(ANCHORS, dtype=np.int64),
            labels=labels,
            path_ids=path_ids,
        )
        captions = [f"{path_ids[index]} y={int(labels[index])}" for index in range(PATH_COUNT)]
        write_contact_sheet(root / "images" / f"{name}_final.png", uint8[-1], columns=16, scale=2, captions=captions)
        trajectory_indices = np.concatenate(
            [np.arange(digit * PATHS_PER_CLASS, digit * PATHS_PER_CLASS + 4, dtype=np.int64) for digit in range(10)]
        )
        trajectory = uint8[:, trajectory_indices].reshape(-1, 28, 28)
        trajectory_captions = [
            f"a{ANCHORS[anchor]} {path_ids[index]}"
            for anchor in range(len(ANCHORS))
            for index in trajectory_indices
        ]
        write_contact_sheet(root / "images" / f"{name}_trajectory.png", trajectory, columns=20, scale=2, captions=trajectory_captions)
        bindings[name] = {
            "raw_file": f"populations/{name}.npz",
            "raw_file_sha256": sha256_file(root / "populations" / f"{name}.npz"),
            "raw_anchor_array_sha256": _hash_array(row["anchors"]),
            "uint8_file": f"populations/{name}_uint8.npz",
            "uint8_file_sha256": sha256_file(uint_path),
            "uint8_anchor_array_sha256": _hash_array(uint8),
            "endpoint_count": PATH_COUNT,
            "telemetry_file": f"telemetry/{name}_steps.csv",
            "telemetry_file_sha256": sha256_file(root / "telemetry" / f"{name}_steps.csv"),
            "telemetry_row_count": OUTER_STEPS,
            "scientific_digest": str(row["scientific_digest"][0]),
        }
    comparison = np.concatenate([rendered["teacher"][-1], rendered["null"][-1], rendered["learned"][-1]], axis=0)
    write_contact_sheet(root / "images" / "comparison_final.png", comparison, columns=16, scale=2)
    seal = {
        "schema": VERSION + "-populations-sealed",
        "created_at": _utc_now(),
        "rows": bindings,
        "path_ids_sha256": _hash_array(path_ids),
        "labels_sha256": _hash_array(labels),
        "mass_transform_sha256": sha256_file(root / "input_bindings" / "mass_to_uint8.json"),
        "all_rows_same_initial_bank": 1,
        "start_bank_seal_sha256": sha256_file(root / "inventory" / "START_BANK_SEALED.json"),
        "start_bank_sha256": sha256_file(root / "inventory" / "start_bank.npz"),
        "generated_candidates_per_path": 1,
        "selector": None,
        "all_endpoints_retained": 1,
    }
    _write_json(root / "populations" / "POPULATIONS_SEALED.json", seal)
    return seal


_ROW_TELEMETRY_KEYS = frozenset(
    {
        "row",
        "completed_step",
        "accepted_substeps",
        "rejected_attempt_count",
        "attempts",
        "accepted_clipped",
        "accepted_proposed",
        "accepted_clipping_fraction",
        "learned_step_rms",
        "free_step_rms",
        "noise_step_rms",
        "state_increment_rms",
        "minimum_mass",
        "maximum_mass",
        "maximum_mass_error",
        "nonfinite_count",
        "elapsed_seconds",
        "cuda_allocated_bytes",
        "cuda_peak_allocated_bytes",
    }
)


def _verifier_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    _require(path.is_file(), f"required NPZ is absent: {path}")
    with np.load(path, allow_pickle=False) as archive:
        _require(set(archive.files) == required, f"{path} key inventory changed")
        return {key: archive[key].copy() for key in archive.files}


def _verify_row_telemetry(
    run_dir: Path,
    name: str,
    raw: Mapping[str, np.ndarray],
    *,
    partial: bool = False,
) -> list[dict[str, Any]]:
    telemetry = list(np.asarray(raw["telemetry"], dtype=object).tolist())
    if partial:
        _require(0 <= len(telemetry) < OUTER_STEPS, f"partial {name} telemetry length changed")
    else:
        _require(len(telemetry) == OUTER_STEPS, f"{name} telemetry length changed")
    anchor_steps = np.asarray(raw["anchor_steps"], dtype=np.int64)
    for index, record in enumerate(telemetry, 1):
        _require(type(record) is dict and frozenset(record) == _ROW_TELEMETRY_KEYS, f"{name} telemetry schema changed at step {index}")
        _require(record["row"] == name and int(record["completed_step"]) == index, f"{name} telemetry identity changed at step {index}")
        attempts = record["attempts"]
        _require(type(attempts) is list and 1 <= len(attempts) <= 3, f"{name} attempt inventory changed at step {index}")
        _require(
            [int(item["substeps"]) for item in attempts] == [1, 2, 4][: len(attempts)],
            f"{name} retry schedule changed at step {index}",
        )
        _require(int(record["rejected_attempt_count"]) == len(attempts) - 1, f"{name} retry count changed at step {index}")
        for attempt_index, attempt in enumerate(attempts):
            _require(
                type(attempt) is dict
                and set(attempt) == {"substeps", "clipped", "proposed", "clipping_fraction"},
                f"{name} attempt schema changed at step {index}",
            )
            clipped = int(attempt["clipped"])
            proposed = int(attempt["proposed"])
            fraction = float(attempt["clipping_fraction"])
            expected_proposed = int(attempt["substeps"]) * PATH_COUNT * 2 * 28 * 28
            _require(
                clipped >= 0 and proposed == expected_proposed and clipped <= proposed,
                f"{name} clipping counts are invalid at step {index}",
            )
            expected_fraction = 0.0 if proposed == 0 else clipped / proposed
            _require(math.isfinite(fraction) and math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-15), f"{name} clipping fraction changed at step {index}")
            if attempt_index + 1 < len(attempts):
                _require(fraction > 0.03, f"{name} accepted a retry-ineligible attempt at step {index}")
        accepted = attempts[-1]
        _require(int(record["accepted_substeps"]) == int(accepted["substeps"]), f"{name} accepted substeps changed at step {index}")
        _require(int(record["accepted_clipped"]) == int(accepted["clipped"]), f"{name} accepted clipping numerator changed at step {index}")
        _require(int(record["accepted_proposed"]) == int(accepted["proposed"]), f"{name} accepted clipping denominator changed at step {index}")
        _require(
            math.isclose(float(record["accepted_clipping_fraction"]), float(accepted["clipping_fraction"]), rel_tol=0.0, abs_tol=1e-15),
            f"{name} accepted clipping fraction changed at step {index}",
        )
        _require(
            float(record["accepted_clipping_fraction"]) <= 0.03
            or int(record["accepted_substeps"]) == 4,
            f"{name} accepted a retry-eligible clipping fraction at step {index}",
        )
        for field in (
            "learned_step_rms",
            "free_step_rms",
            "noise_step_rms",
            "state_increment_rms",
            "minimum_mass",
            "maximum_mass",
            "maximum_mass_error",
            "elapsed_seconds",
        ):
            value = float(record[field])
            _require(math.isfinite(value) and value >= 0.0, f"{name} telemetry {field} is invalid at step {index}")
        _require(float(record["maximum_mass"]) >= float(record["minimum_mass"]), f"{name} telemetry mass range changed at step {index}")
        _require(float(record["maximum_mass_error"]) <= 2e-6, f"{name} telemetry mass error is invalid at step {index}")
        _require(int(record["nonfinite_count"]) == 0, f"{name} telemetry reports nonfinite state at step {index}")
        _require(int(record["cuda_allocated_bytes"]) >= 0 and int(record["cuda_peak_allocated_bytes"]) >= 0, f"{name} CUDA telemetry is invalid at step {index}")
        _require(int(record["cuda_peak_allocated_bytes"]) >= int(record["cuda_allocated_bytes"]), f"{name} CUDA peak telemetry is invalid at step {index}")
        if index in anchor_steps[1:]:
            anchor = np.asarray(raw["anchors"])[int(np.flatnonzero(anchor_steps == index)[0])]
            _require(math.isclose(float(record["minimum_mass"]), float(anchor.min()), rel_tol=0.0, abs_tol=2e-7), f"{name} anchor minimum disagrees with telemetry at step {index}")
            _require(math.isclose(float(record["maximum_mass"]), float(anchor.max()), rel_tol=0.0, abs_tol=2e-7), f"{name} anchor maximum disagrees with telemetry at step {index}")

    csv_path = run_dir / "telemetry" / (f"partial_{name}_steps.csv" if partial else f"{name}_steps.csv")
    _require(csv_path.is_file(), f"{name} telemetry CSV is absent")
    if partial and not telemetry:
        _require(csv_path.read_bytes() in {b"", b"\n", b"\r\n"}, f"partial {name} empty telemetry CSV changed")
        return telemetry
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames is not None
            and len(reader.fieldnames) == len(_ROW_TELEMETRY_KEYS)
            and set(reader.fieldnames) == set(_ROW_TELEMETRY_KEYS),
            f"{name} telemetry CSV columns changed",
        )
        csv_rows = list(reader)
    _require(len(csv_rows) == len(telemetry), f"{name} telemetry CSV length changed")
    integer_fields = {
        "completed_step",
        "accepted_substeps",
        "rejected_attempt_count",
        "accepted_clipped",
        "accepted_proposed",
        "nonfinite_count",
        "cuda_allocated_bytes",
        "cuda_peak_allocated_bytes",
    }
    for index, (saved, expected) in enumerate(zip(csv_rows, telemetry, strict=True), 1):
        _require(saved["row"] == name, f"{name} telemetry CSV row identity changed at step {index}")
        _require(json.loads(saved["attempts"]) == expected["attempts"], f"{name} telemetry CSV attempts changed at step {index}")
        for field in integer_fields:
            _require(int(saved[field]) == int(expected[field]), f"{name} telemetry CSV {field} changed at step {index}")
        for field in _ROW_TELEMETRY_KEYS - integer_fields - {"row", "attempts"}:
            _require(
                math.isclose(float(saved[field]), float(expected[field]), rel_tol=0.0, abs_tol=0.0),
                f"{name} telemetry CSV {field} changed at step {index}",
            )
    return telemetry


def _verify_sheet_pixels(
    path: Path,
    images: np.ndarray,
    *,
    columns: int,
    scale: int,
    captions: Sequence[str] | None,
) -> None:
    from PIL import ImageDraw

    _require(path.is_file(), f"contact sheet is absent: {path}")
    source = np.asarray(images)
    _require(source.dtype == np.uint8 and source.ndim == 3 and source.shape[1:] == (28, 28), f"contact-sheet source is invalid: {path}")
    _require(captions is None or len(captions) == len(source), f"contact-sheet caption inventory changed: {path}")
    cell_width = 28 * int(scale)
    cell_height = cell_width + (12 if captions is not None else 0)
    rows = (len(source) + int(columns) - 1) // int(columns)
    rendered = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    _require(rendered.shape == (rows * cell_height, int(columns) * cell_width), f"contact-sheet dimensions changed: {path}")
    expected_sheet = Image.new("L", (int(columns) * cell_width, rows * cell_height), 255)
    draw = ImageDraw.Draw(expected_sheet)
    for index, image in enumerate(source):
        row, column = divmod(index, int(columns))
        x, y = column * cell_width, row * cell_height
        expected_sheet.paste(
            Image.fromarray(image).resize((cell_width, cell_width), Image.Resampling.NEAREST),
            (x, y),
        )
        if captions is not None:
            draw.text((x + 1, y + cell_width + 1), str(captions[index]), fill=0)
    _require(np.array_equal(rendered, np.asarray(expected_sheet, dtype=np.uint8)), f"contact-sheet pixels changed: {path}")


def _verify_population_seal(run_dir: Path) -> dict[str, Any]:
    seal_path = run_dir / "populations" / "POPULATIONS_SEALED.json"
    seal = _read_json(seal_path)
    _require(
        set(seal)
        == {
            "schema",
            "created_at",
            "rows",
            "path_ids_sha256",
            "labels_sha256",
            "mass_transform_sha256",
            "all_rows_same_initial_bank",
            "start_bank_seal_sha256",
            "start_bank_sha256",
            "generated_candidates_per_path",
            "selector",
            "all_endpoints_retained",
        },
        "population seal schema changed",
    )
    _require(seal["schema"] == VERSION + "-populations-sealed", "population seal version changed")
    _require(
        seal["generated_candidates_per_path"] == 1
        and seal["selector"] is None
        and seal["all_endpoints_retained"] == 1
        and seal["all_rows_same_initial_bank"] == 1,
        "factor-one population policy changed",
    )
    _require(type(seal["rows"]) is dict and set(seal["rows"]) == {"teacher", "null", "learned"}, "population row inventory changed")

    transform_path = run_dir / "input_bindings" / "mass_to_uint8.json"
    authority = _read_json(transform_path)
    scale = np.float64.fromhex(MASS_SCALE_HEX)
    _require(authority.get("derivation_slice") == [TRAIN_START, TRAIN_STOP], "mass transform derivation slice changed")
    _require(authority.get("central_sums") == [25_470, 25_472], "mass transform central sums changed")
    _require(authority.get("numerator") == MASS_SCALE_NUMERATOR and authority.get("denominator") == MASS_SCALE_DENOMINATOR, "mass transform rational changed")
    _require(authority.get("float_hex") == MASS_SCALE_HEX and float(authority.get("decimal")) == float(scale), "mass transform float authority changed")
    _require(seal["mass_transform_sha256"] == sha256_file(transform_path), "population mass-transform binding changed")

    start_path = run_dir / "inventory" / "start_bank.npz"
    starts = _verifier_npz(start_path, {"starts", "labels", "path_ids", "source_seeds"})
    expected_inventory = build_path_inventory()
    _require(starts["starts"].dtype == np.float32 and starts["starts"].shape == (PATH_COUNT, 784), "sealed start states changed")
    _require(np.array_equal(starts["labels"], expected_inventory["requested_labels"]), "sealed start labels changed")
    _require(np.array_equal(starts["path_ids"].astype(str), expected_inventory["path_ids"].astype(str)), "sealed start path IDs changed")
    _require(np.array_equal(starts["source_seeds"].astype(np.uint64), expected_inventory["source_seeds"]), "sealed source seeds changed")
    _require(bool(np.all(np.isfinite(starts["starts"]))) and float(starts["starts"].min()) >= 0.0, "sealed start states are invalid")
    _require(float(np.max(np.abs(starts["starts"].sum(axis=1, dtype=np.float64) - 1.0))) <= 2e-6, "sealed start mass changed")
    _require(len({_hash_array(row) for row in starts["starts"]}) == PATH_COUNT, "sealed starts are not unique")
    start_seal_path = run_dir / "inventory" / "START_BANK_SEALED.json"
    start_seal = _read_json(start_seal_path)
    _require(start_seal.get("start_bank_sha256") == sha256_file(start_path), "start-bank file binding changed")
    _require(start_seal.get("starts_sha256") == _hash_array(starts["starts"]), "start-bank state binding changed")
    _require(start_seal.get("labels_sha256") == _hash_array(starts["labels"]), "start-bank label binding changed")
    _require(start_seal.get("path_ids_sha256") == _hash_array(starts["path_ids"]), "start-bank path-ID binding changed")
    _require(seal["start_bank_sha256"] == sha256_file(start_path), "population start-bank binding changed")
    _require(seal["start_bank_seal_sha256"] == sha256_file(start_seal_path), "population start-seal binding changed")

    row_arrays: dict[str, dict[str, np.ndarray]] = {}
    rendered_arrays: dict[str, np.ndarray] = {}
    expected_row_binding_keys = {
        "raw_file",
        "raw_file_sha256",
        "raw_anchor_array_sha256",
        "uint8_file",
        "uint8_file_sha256",
        "uint8_anchor_array_sha256",
        "endpoint_count",
        "telemetry_file",
        "telemetry_file_sha256",
        "telemetry_row_count",
        "scientific_digest",
    }
    for name in ("teacher", "null", "learned"):
        raw_path = run_dir / "populations" / f"{name}.npz"
        raw = _load_row_population(raw_path, expected_row=name)
        _require(np.asarray(raw["root_seed"]).dtype == np.uint64 and np.asarray(raw["root_seed"]).shape == (1,), f"{name} root-seed schema changed")
        _require(int(raw["root_seed"][0]) == ROW_ROOT_SEEDS[name], f"{name} root seed changed")
        _require(np.array_equal(raw["labels"], expected_inventory["requested_labels"]), f"{name} labels changed")
        _require(np.array_equal(raw["path_ids"].astype(str), expected_inventory["path_ids"].astype(str)), f"{name} path IDs changed")
        _require(np.array_equal(raw["anchors"][0], starts["starts"]), f"{name} start anchor changed")

        uint_path = run_dir / "populations" / f"{name}_uint8.npz"
        uint_archive = _verifier_npz(uint_path, {"anchors", "anchor_steps", "labels", "path_ids"})
        uint8 = uint_archive["anchors"]
        _require(uint8.dtype == np.uint8 and uint8.shape == (len(ANCHORS), PATH_COUNT, 28, 28), f"{name} uint8 population schema changed")
        _require(np.array_equal(uint_archive["anchor_steps"], np.asarray(ANCHORS, dtype=np.int64)), f"{name} uint8 anchor steps changed")
        expected = mass_to_uint8(raw["anchors"], authority)
        _require(np.array_equal(uint8, expected), f"{name} uint8 rendering does not match raw masses")
        _require(np.array_equal(uint_archive["labels"], raw["labels"]) and np.array_equal(uint_archive["path_ids"].astype(str), raw["path_ids"].astype(str)), f"{name} uint8 identity mismatch")
        telemetry = _verify_row_telemetry(run_dir, name, raw)

        binding = seal["rows"][name]
        _require(type(binding) is dict and set(binding) == expected_row_binding_keys, f"{name} population binding schema changed")
        _require(binding["raw_file"] == f"populations/{name}.npz", f"{name} raw path binding changed")
        _require(binding["uint8_file"] == f"populations/{name}_uint8.npz", f"{name} uint8 path binding changed")
        _require(binding["telemetry_file"] == f"telemetry/{name}_steps.csv", f"{name} telemetry path binding changed")
        _require(binding["raw_file_sha256"] == sha256_file(raw_path), f"{name} raw seal mismatch")
        _require(binding["uint8_file_sha256"] == sha256_file(uint_path), f"{name} uint8 seal mismatch")
        _require(binding["telemetry_file_sha256"] == sha256_file(run_dir / "telemetry" / f"{name}_steps.csv"), f"{name} telemetry seal mismatch")
        _require(binding["raw_anchor_array_sha256"] == _hash_array(raw["anchors"]), f"{name} raw array seal mismatch")
        _require(binding["uint8_anchor_array_sha256"] == _hash_array(uint8), f"{name} uint8 array seal mismatch")
        _require(int(binding["endpoint_count"]) == PATH_COUNT and int(binding["telemetry_row_count"]) == OUTER_STEPS, f"{name} population counts changed")
        _require(binding["scientific_digest"] == _scientific_row_digest(raw["anchors"], telemetry), f"{name} scientific binding changed")
        row_arrays[name] = raw
        rendered_arrays[name] = uint8

    _require(seal["path_ids_sha256"] == _hash_array(starts["path_ids"]), "population path-ID authority changed")
    _require(seal["labels_sha256"] == _hash_array(starts["labels"]), "population label authority changed")
    _require(not list((run_dir / "populations").glob("partial_*.npz")), "partial row artifacts remain after population seal")
    _require(not list((run_dir / "telemetry").glob("partial_*_steps.csv")), "partial telemetry remains after population seal")
    _require(not list((run_dir / "images").glob("partial_*_latest.png")), "partial image artifacts remain after population seal")

    trajectory_indices = np.concatenate(
        [np.arange(digit * PATHS_PER_CLASS, digit * PATHS_PER_CLASS + 4, dtype=np.int64) for digit in range(10)]
    )
    for name in ("teacher", "null", "learned"):
        final_captions = [
            f"{row_arrays[name]['path_ids'][index]} y={int(row_arrays[name]['labels'][index])}"
            for index in range(PATH_COUNT)
        ]
        _verify_sheet_pixels(
            run_dir / "images" / f"{name}_final.png",
            rendered_arrays[name][-1],
            columns=16,
            scale=2,
            captions=final_captions,
        )
        trajectory_captions = [
            f"a{ANCHORS[anchor]} {row_arrays[name]['path_ids'][index]}"
            for anchor in range(len(ANCHORS))
            for index in trajectory_indices
        ]
        _verify_sheet_pixels(
            run_dir / "images" / f"{name}_trajectory.png",
            rendered_arrays[name][:, trajectory_indices].reshape(-1, 28, 28),
            columns=20,
            scale=2,
            captions=trajectory_captions,
        )
    comparison = np.concatenate(
        [rendered_arrays["teacher"][-1], rendered_arrays["null"][-1], rendered_arrays["learned"][-1]],
        axis=0,
    )
    _verify_sheet_pixels(run_dir / "images" / "comparison_final.png", comparison, columns=16, scale=2, captions=None)
    return seal


def prepare_blind_review(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    seal = _verify_population_seal(root)
    indices = np.asarray(
        [digit * PATHS_PER_CLASS + offset for digit in range(10) for offset in REVIEW_WITHIN_CLASS],
        dtype=np.int64,
    )
    _require(indices.shape == (40,), "review index inventory mismatch")
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    opaque_members: list[str] = []
    membership: list[dict[str, Any]] = []
    for row_name in ("learned", "null"):
        with np.load(root / "populations" / f"{row_name}_uint8.npz", allow_pickle=False) as archive:
            endpoints = archive["anchors"][-1]
            row_labels = archive["labels"]
            path_ids = archive["path_ids"].astype(str)
        rows.append(endpoints[indices])
        labels.append(row_labels[indices])
        for index in indices.tolist():
            member_id = f"review-member-{len(opaque_members):03d}"
            opaque_members.append(member_id)
            membership.append(
                {
                    "member_id": member_id,
                    "row": row_name,
                    "path_id": str(path_ids[index]),
                    "requested_label": int(row_labels[index]),
                    "path_index": int(index),
                }
            )
    images = np.concatenate(rows, axis=0)
    requested = np.concatenate(labels, axis=0).astype(np.int64)
    review_root = root / "review"
    review_root.mkdir(parents=True, exist_ok=True)
    _write_npy(review_root / "review_indices.npy", indices)
    _write_json(review_root / "private_membership.json", {"schema": VERSION + "-review-membership", "entries": membership})
    bundle = write_blinded_review_bundle(
        review_root,
        images,
        requested,
        np.asarray(opaque_members, dtype=np.str_),
        seed=REVIEW_SEED,
        columns=10,
        scale=4,
    )
    ready = {
        "schema": VERSION + "-review-ready",
        "created_at": _utc_now(),
        "population_seal_sha256": sha256_file(root / "populations" / "POPULATIONS_SEALED.json"),
        "population_tree_binding": _sha256_bytes(_canonical_json_bytes(seal)),
        "sample_count": 80,
        "learned_count": 40,
        "null_count": 40,
        "review_seed": REVIEW_SEED,
        "template_sha256": sha256_file(Path(bundle["template"])),
        "contact_sheet_sha256": sha256_file(Path(bundle["contact_sheet"])),
        "review_key_sha256": sha256_file(Path(bundle["key"])),
        "membership_sha256": sha256_file(review_root / "private_membership.json"),
    }
    _write_json(review_root / "READY.json", ready)
    return ready


def _load_evaluator_after_population_seal(run_dir: Path, *, device: str | torch.device) -> SmallMnistCNN:
    seal_path = run_dir / "populations" / "POPULATIONS_SEALED.json"
    _require(seal_path.is_file(), "evaluator firewall: population seal is absent")
    _verify_population_seal(run_dir)
    binding = _read_json(run_dir / "input_bindings" / "ddpm_evaluator_binding.json")
    open_event = _read_json(run_dir / "evaluation" / "EVALUATOR_OPEN_EVENT.json")
    _require(open_event["population_seal_sha256"] == sha256_file(seal_path), "evaluator open event has stale population seal")
    _require(open_event["evaluator_binding_sha256"] == sha256_file(run_dir / "input_bindings" / "ddpm_evaluator_binding.json"), "evaluator open event binding mismatch")
    checkpoint = run_dir / str(binding["copied_checkpoint"])
    _require(checkpoint.stat().st_size == EVALUATOR_BYTES, "evaluator checkpoint byte mismatch")
    _require(sha256_file(checkpoint) == EVALUATOR_SHA256, "evaluator checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(type(payload) is dict and set(payload) == {"state_dict", "selected_epoch"}, "evaluator payload schema mismatch")
    _require(isinstance(payload["selected_epoch"], int), "evaluator selected epoch is invalid")
    model = SmallMnistCNN()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _reference_subset(images: np.ndarray, labels: np.ndarray, *, per_class: int = PATHS_PER_CLASS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = []
    for digit in range(10):
        matches = np.flatnonzero(labels == digit)
        _require(len(matches) >= per_class, f"reference class {digit} is too small")
        indices.extend(matches[:per_class].tolist())
    selected = np.asarray(indices, dtype=np.int64)
    return images[selected], labels[selected], selected


def _exact_match_count(images: np.ndarray, reference: np.ndarray) -> int:
    keys = {row.tobytes() for row in np.ascontiguousarray(reference).reshape(len(reference), -1)}
    return sum(row.tobytes() in keys for row in np.ascontiguousarray(images).reshape(len(images), -1))


def _fixed_render_statistics(images: np.ndarray) -> dict[str, Any]:
    """Return compact statistics of the one frozen global uint8 rendering."""
    array = np.asarray(images)
    _require(array.dtype == np.uint8 and array.ndim == 3 and array.shape[1:] == (28, 28), "render statistics require uint8 MNIST images")
    values = array.astype(np.float64)
    per_image_mean = values.reshape(len(values), -1).mean(axis=1)
    per_image_nonzero = np.mean(values.reshape(len(values), -1) > 0, axis=1)
    return {
        "sample_count": int(len(array)),
        "pixel_mean": float(values.mean()),
        "pixel_standard_deviation": float(values.std()),
        "zero_fraction": float(np.mean(values == 0)),
        "saturated_fraction": float(np.mean(values == 255)),
        "per_image_mean_median": float(np.median(per_image_mean)),
        "per_image_nonzero_fraction_median": float(np.median(per_image_nonzero)),
        "render_authority": "global-rint-25471-over-255-no-per-image-normalization",
    }


def _split_metric_arrays(metrics: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Keep classifier arrays in NPZ and only compact scalars/tables in JSON."""
    classifier = dict(metrics["classifier"])
    arrays = {
        "predictions": np.asarray(classifier.pop("predictions"), dtype=np.int64),
        "logits": np.asarray(classifier.pop("logits"), dtype=np.float64),
        "sample_ids": np.asarray(classifier.pop("sample_ids", []), dtype=np.str_),
        "requested_labels": np.asarray(classifier.pop("requested_labels", []), dtype=np.int64),
    }
    compact = dict(metrics)
    compact["classifier"] = classifier
    return compact, arrays


def _teacher_positive_control(
    root: Path,
    *,
    teacher_anchors: np.ndarray,
    teacher_accuracy: float,
) -> dict[str, Any]:
    target_path = root / "inventory" / "teacher_target_bank.npz"
    _require(target_path.is_file(), "teacher target bank is absent at Gate D")
    with np.load(target_path, allow_pickle=False) as archive:
        _require("masses" in archive.files, "teacher target bank has no masses")
        targets = archive["masses"].astype(np.float32, copy=True)
        target_labels = archive["requested_labels"].astype(np.int64, copy=True)
        target_path_ids = archive["path_ids"].astype(str, copy=True)
    _require(targets.shape == (PATH_COUNT, 784), "teacher target mass shape mismatch")
    _require(bool(np.all(np.isfinite(targets))) and float(targets.min()) >= 0.0, "teacher targets are numerically invalid")
    _require(float(np.max(np.abs(targets.sum(axis=1, dtype=np.float64) - 1.0))) <= 2e-6, "teacher target mass error")
    inventory = build_path_inventory()
    _require(np.array_equal(target_labels, inventory["requested_labels"]), "teacher target labels changed")
    _require(np.array_equal(target_path_ids, inventory["path_ids"].astype(str)), "teacher target path IDs changed")
    errors = np.sum(
        (teacher_anchors.astype(np.float64) - targets.astype(np.float64)[None, :, :]) ** 2,
        axis=2,
    )
    ratios = errors / np.maximum(errors[0:1], np.finfo(np.float64).tiny)
    anchor64_index = ANCHORS.index(64)
    endpoint_index = ANCHORS.index(256)
    median_ratio64 = float(np.median(ratios[anchor64_index]))
    median_ratio256 = float(np.median(ratios[endpoint_index]))
    improved = errors[endpoint_index] < errors[0]
    target_uint8 = mass_to_uint8(targets, _read_json(root / "input_bindings" / "mass_to_uint8.json"))
    conditions = {
        "median_ratio_anchor64_at_most_0_80": median_ratio64 <= 0.80,
        "median_ratio_endpoint_at_most_0_20": median_ratio256 <= 0.20,
        "improved_path_count_at_least_144": int(improved.sum()) >= 144,
        "teacher_classifier_accuracy_at_least_0_80": float(teacher_accuracy) >= CLASSIFIER_POSITIVE_ACCURACY,
        "target_render_health": target_uint8.shape == (PATH_COUNT, 28, 28),
        "teacher_render_health": bool(np.all(np.isfinite(teacher_anchors))) and float(teacher_anchors.min()) >= 0.0,
    }
    arrays_path = root / "controls" / "teacher_gate_arrays.npz"
    _write_npz(
        arrays_path,
        squared_l2=errors.astype(np.float64),
        relative_squared_l2=ratios.astype(np.float64),
        endpoint_improved=improved.astype(np.uint8),
        anchor_steps=np.asarray(ANCHORS, dtype=np.int64),
    )
    report = {
        "schema": VERSION + "-teacher-positive-control",
        "gate_type": "execution/integrity",
        "downstream_action_controlled": "learner attribution and interpretation of learned/null task results",
        "median_relative_squared_l2_anchor64": median_ratio64,
        "median_relative_squared_l2_endpoint": median_ratio256,
        "endpoint_improved_path_count": int(improved.sum()),
        "path_count": PATH_COUNT,
        "teacher_requested_label_accuracy": float(teacher_accuracy),
        "conditions": {key: int(value) for key, value in conditions.items()},
        "passed": int(all(conditions.values())),
        "arrays_file": "controls/teacher_gate_arrays.npz",
        "arrays_sha256": sha256_file(arrays_path),
        "target_bank_sha256": sha256_file(target_path),
        "failure_means": "the common teacher/controller/integrator/limiter/render interface is not a valid positive control",
        "failure_does_not_mean": "the learned checkpoint or Eulerian generation generally failed",
    }
    _write_json(root / "controls" / "teacher_gate.json", report)
    return report


def evaluate_sealed_populations(
    run_dir: str | Path,
    *,
    arff_path: str | Path,
    device: str | torch.device,
    development_images: np.ndarray | None = None,
    development_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    seal = _verify_population_seal(root)
    if development_images is None or development_labels is None:
        development_images, development_labels, _ = read_mnist_development_prefix(arff_path)
    development_images = np.asarray(development_images)
    development_labels = np.asarray(development_labels, dtype=np.int64)
    _require(development_images.shape == (60_000, 28, 28), "development image inventory mismatch")
    arff = Path(arff_path)
    _require(arff.is_file(), "MNIST ARFF is absent at terminal-test opening")
    _require(sha256_file(arff) == MNIST_ARFF_SHA256, "MNIST ARFF changed before terminal-test opening")
    event = {
        "schema": VERSION + "-test-open-event",
        "opened_at": _utc_now(),
        "arff_sha256": MNIST_ARFF_SHA256,
        "population_seal_sha256": sha256_file(root / "populations" / "POPULATIONS_SEALED.json"),
        "population_rows": {name: seal["rows"][name]["raw_file_sha256"] for name in ("teacher", "null", "learned")},
        "generation_after_event_forbidden": 1,
    }
    _write_json(root / "data" / "test_open_event.json", event)
    terminal_images, terminal_labels = read_mnist_arff_slice(arff, 60_000, 70_000)
    test_reference, test_reference_labels, test_reference_indices = _reference_subset(terminal_images, terminal_labels)
    _write_npz(
        root / "evaluation" / "terminal_reference_uint8.npz",
        images=test_reference,
        labels=test_reference_labels,
        terminal_local_indices=test_reference_indices,
    )
    _write_json(
        root / "evaluation" / "EVALUATOR_OPEN_EVENT.json",
        {
            "schema": VERSION + "-evaluator-open-event",
            "opened_at": _utc_now(),
            "population_seal_sha256": sha256_file(root / "populations" / "POPULATIONS_SEALED.json"),
            "evaluator_binding_sha256": sha256_file(root / "input_bindings" / "ddpm_evaluator_binding.json"),
        },
    )
    model = _load_evaluator_after_population_seal(root, device=device)
    row_metrics: dict[str, Any] = {}
    raw_arrays: dict[str, np.ndarray] = {}
    for name in ("teacher", "null", "learned"):
        with np.load(root / "populations" / f"{name}_uint8.npz", allow_pickle=False) as archive:
            endpoints = archive["anchors"][-1].copy()
            labels = archive["labels"].astype(np.int64, copy=True)
            path_ids = archive["path_ids"].astype(str, copy=True)
        full_metrics = compute_generation_metrics(
            model,
            endpoints,
            labels,
            path_ids,
            real_reference_images=test_reference,
            real_reference_labels=test_reference_labels,
            train_images=development_images[:TRAIN_STOP],
            test_images=terminal_images,
            device=device,
        )
        metrics, classifier_arrays = _split_metric_arrays(full_metrics)
        metrics["exact_validation_match_count"] = _exact_match_count(endpoints, development_images[VALIDATION_START:VALIDATION_STOP])
        metrics["fixed_render_statistics"] = _fixed_render_statistics(endpoints)
        _write_json(root / "evaluation" / f"{name}_metrics.json", metrics)
        row_metrics[name] = metrics
        raw_arrays[f"{name}_sample_ids"] = np.asarray(path_ids, dtype=np.str_)
        raw_arrays[f"{name}_requested_labels"] = labels
        raw_arrays[f"{name}_predictions"] = classifier_arrays["predictions"]
        raw_arrays[f"{name}_logits"] = classifier_arrays["logits"]
    _write_npz(root / "evaluation" / "predictions.npz", **raw_arrays)
    learned_accuracy = float(row_metrics["learned"]["classifier"]["requested_label_accuracy"])
    null_accuracy = float(row_metrics["null"]["classifier"]["requested_label_accuracy"])
    effects = {
        "schema": VERSION + "-learned-minus-null",
        "classifier_accuracy_difference": learned_accuracy - null_accuracy,
        "learned_classifier_accuracy": learned_accuracy,
        "null_classifier_accuracy": null_accuracy,
        "learned_duplicate_pair_count": int(row_metrics["learned"]["duplicates"]["duplicate_pair_count"]),
        "learned_diversity_ratio": float(row_metrics["learned"]["diversity"]["aggregate_median_ratio"]),
        "row_effects_are_stochastic_unpaired": 1,
    }
    _write_json(root / "evaluation" / "learned_minus_null.json", effects)
    contextual = {
        "schema": VERSION + "-contextual-ddpm",
        "role": "contextual exploratory calibration, not a paired or hypothesis-test baseline",
        "classifier_accuracy": 0.925,
        "human_requested_label_agreement": 0.925,
        "human_recognizability": 1.0,
        "duplicate_pair_count": 0,
        "diversity_ratio": 1.0925057312146145,
        "ddpm_tree_digest": DDPM_TREE_DIGEST,
    }
    _write_json(root / "evaluation" / "contextual_ddpm_comparison.json", contextual)
    teacher_raw = _load_row_population(root / "populations" / "teacher.npz", expected_row="teacher")
    teacher_control = _teacher_positive_control(
        root,
        teacher_anchors=teacher_raw["anchors"],
        teacher_accuracy=float(row_metrics["teacher"]["classifier"]["requested_label_accuracy"]),
    )
    ready = {
        "schema": VERSION + "-scoring-ready",
        "created_at": _utc_now(),
        "population_seal_sha256": sha256_file(root / "populations" / "POPULATIONS_SEALED.json"),
        "test_open_event_sha256": sha256_file(root / "data" / "test_open_event.json"),
        "predictions_sha256": sha256_file(root / "evaluation" / "predictions.npz"),
        "metrics_sha256": {name: sha256_file(root / "evaluation" / f"{name}_metrics.json") for name in row_metrics},
        "teacher_control_sha256": sha256_file(root / "controls" / "teacher_gate.json"),
        "teacher_control_passed": int(teacher_control["passed"]),
    }
    _write_json(root / "evaluation" / "SCORING_READY.json", ready)
    return {"rows": row_metrics, "effects": effects, "teacher_control": teacher_control, "ready": ready}


prepare_review = prepare_blind_review


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    _replace_with_retry(temporary, destination)


def _source_binding_payload(repository_root: Path) -> dict[str, Any]:
    paths = {
        **PROTECTED_SOURCE_HASHES,
        Path(__file__).resolve().relative_to(repository_root).as_posix(): sha256_file(Path(__file__).resolve()),
    }
    doc = repository_root / "docs" / "eulerian_edge_flux_fresh_replay.md"
    if doc.is_file():
        paths[doc.relative_to(repository_root).as_posix()] = sha256_file(doc)
    for relative, expected in PROTECTED_SOURCE_HASHES.items():
        path = repository_root / relative
        _require(path.is_file() and sha256_file(path) == expected, f"protected source changed: {relative}")
    return {
        "schema": VERSION + "-source-bindings",
        "git_revision": _git_revision(repository_root),
        "files": [
            {"path": relative, "bytes": (repository_root / relative).stat().st_size, "sha256": digest}
            for relative, digest in sorted(paths.items())
        ],
    }


def _file_authority_observation(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    candidate = path.resolve()
    exists = candidate.exists()
    kind = "file" if candidate.is_file() else ("directory" if candidate.is_dir() else ("other" if exists else "missing"))
    observed_bytes = candidate.stat().st_size if kind == "file" else None
    observed_sha256 = sha256_file(candidate) if kind == "file" else None
    matched = kind == "file" and observed_bytes == int(expected_bytes) and observed_sha256 == expected_sha256
    return {
        "path": str(candidate),
        "expected_kind": "file",
        "expected_bytes": int(expected_bytes),
        "expected_sha256": expected_sha256,
        "observed_exists": int(exists),
        "observed_kind": kind,
        "observed_bytes": observed_bytes,
        "observed_sha256": observed_sha256,
        "matched": int(matched),
    }


def _manifested_directory_observation(
    path: Path,
    *,
    expected_manifest_bytes: int,
    expected_manifest_sha256: str,
    expected_tree_digest: str,
    auxiliary: Mapping[str, tuple[str, int, str]],
    k128_semantics: bool = False,
) -> dict[str, Any]:
    candidate = path.resolve()
    exists = candidate.exists()
    kind = "directory" if candidate.is_dir() else ("file" if candidate.is_file() else ("other" if exists else "missing"))
    manifest = _file_authority_observation(
        candidate / "artifact_manifest.json",
        expected_bytes=expected_manifest_bytes,
        expected_sha256=expected_manifest_sha256,
    )
    try:
        _verify_external_manifest(
            candidate,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_tree_digest=expected_tree_digest,
        )
        manifest_check = {"passed": 1, "error": None}
    except Exception as error:
        manifest_check = {"passed": 0, "error": f"{type(error).__name__}: {error}"}
    auxiliary_observations = {
        name: _file_authority_observation(
            candidate / relative,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        for name, (relative, expected_bytes, expected_sha256) in auxiliary.items()
    }
    semantic: dict[str, Any] | None = None
    if k128_semantics:
        try:
            status = _read_json(candidate / "status.json")
            outcome = _read_json(candidate / "outcome.json")
            semantic = {
                "readable": 1,
                "status_route": status.get("route"),
                "outcome_route": outcome.get("route"),
                "full_scale_auto_launched": outcome.get("full_scale_auto_launched"),
                "matched": int(
                    status.get("route") == "complete"
                    and outcome.get("route") == K128_REQUIRED_ROUTE
                    and outcome.get("full_scale_auto_launched") == 0
                ),
                "error": None,
            }
        except Exception as error:
            semantic = {
                "readable": 0,
                "status_route": None,
                "outcome_route": None,
                "full_scale_auto_launched": None,
                "matched": 0,
                "error": f"{type(error).__name__}: {error}",
            }
    matched = (
        kind == "directory"
        and int(manifest_check["passed"]) == 1
        and all(int(value["matched"]) == 1 for value in auxiliary_observations.values())
        and (semantic is None or int(semantic["matched"]) == 1)
    )
    return {
        "path": str(candidate),
        "expected_kind": "directory",
        "expected_tree_digest": expected_tree_digest,
        "observed_exists": int(exists),
        "observed_kind": kind,
        "manifest": manifest,
        "full_manifest_check": manifest_check,
        "auxiliary": auxiliary_observations,
        "semantic": semantic,
        "matched": int(matched),
    }


def _input_authority_observations(
    *,
    legacy_checkpoint: Path,
    arff: Path,
    k128_run_dir: Path,
    ddpm_run_dir: Path,
    recorded_at: str,
) -> dict[str, Any]:
    inputs = {
        "legacy_checkpoint": _file_authority_observation(
            legacy_checkpoint,
            expected_bytes=LEGACY_CHECKPOINT_BYTES,
            expected_sha256=LEGACY_CHECKPOINT_SHA256,
        ),
        "arff": _file_authority_observation(
            arff,
            expected_bytes=MNIST_ARFF_BYTES,
            expected_sha256=MNIST_ARFF_SHA256,
        ),
        "k128_run_dir": _manifested_directory_observation(
            k128_run_dir,
            expected_manifest_bytes=K128_MANIFEST_BYTES,
            expected_manifest_sha256=K128_MANIFEST_SHA256,
            expected_tree_digest=K128_TREE_DIGEST,
            auxiliary={
                "status": ("status.json", K128_STATUS_BYTES, K128_STATUS_SHA256),
                "outcome": ("outcome.json", K128_OUTCOME_BYTES, K128_OUTCOME_SHA256),
                "report": ("REPORT.md", K128_REPORT_BYTES, K128_REPORT_SHA256),
            },
            k128_semantics=True,
        ),
        "ddpm_run_dir": _manifested_directory_observation(
            ddpm_run_dir,
            expected_manifest_bytes=DDPM_MANIFEST_BYTES,
            expected_manifest_sha256=DDPM_MANIFEST_SHA256,
            expected_tree_digest=DDPM_TREE_DIGEST,
            auxiliary={
                "evaluator_selection": (
                    "evaluator/selection.json",
                    EVALUATOR_SELECTION_BYTES,
                    EVALUATOR_SELECTION_SHA256,
                ),
                "evaluator_checkpoint": (
                    "evaluator/selected_checkpoint.pt",
                    EVALUATOR_BYTES,
                    EVALUATOR_SHA256,
                ),
            },
        ),
    }
    return {
        "schema": VERSION + "-input-authority-observations",
        "recorded_at": recorded_at,
        "inputs": inputs,
        "all_expected_authorities_matched": int(all(int(value["matched"]) == 1 for value in inputs.values())),
    }


def _bind_external_authorities(
    run_dir: Path,
    *,
    repository_root: Path,
    legacy_checkpoint: Path,
    arff: Path,
    k128_run_dir: Path,
    ddpm_run_dir: Path,
) -> dict[str, Any]:
    # Bind the live source closure before any fallible predecessor check so an
    # initialization failure still has a source-authenticated terminal tree.
    _write_json(run_dir / "source_bindings.json", _source_binding_payload(repository_root))
    observations = _input_authority_observations(
        legacy_checkpoint=legacy_checkpoint,
        arff=arff,
        k128_run_dir=k128_run_dir,
        ddpm_run_dir=ddpm_run_dir,
        recorded_at=_utc_now(),
    )
    _write_json(run_dir / "input_bindings" / "input_authority_observations.json", observations)
    k128 = _verify_external_manifest(
        k128_run_dir,
        expected_manifest_sha256=K128_MANIFEST_SHA256,
        expected_tree_digest=K128_TREE_DIGEST,
    )
    for relative, expected in (
        ("status.json", K128_STATUS_SHA256),
        ("outcome.json", K128_OUTCOME_SHA256),
        ("REPORT.md", K128_REPORT_SHA256),
    ):
        _require(sha256_file(k128_run_dir / relative) == expected, f"K128 {relative} authority changed")
    k128_status = _read_json(k128_run_dir / "status.json")
    k128_outcome = _read_json(k128_run_dir / "outcome.json")
    _require(k128_status.get("route") == "complete", "K128 predecessor is not complete")
    _require(k128_outcome.get("route") == K128_REQUIRED_ROUTE, "K128 predecessor route mismatch")
    _require(int(k128_outcome.get("full_scale_auto_launched", -1)) == 0, "K128 predecessor launched forbidden full scale")
    ddpm = _verify_external_manifest(
        ddpm_run_dir,
        expected_manifest_sha256=DDPM_MANIFEST_SHA256,
        expected_tree_digest=DDPM_TREE_DIGEST,
    )
    selection = ddpm_run_dir / "evaluator" / "selection.json"
    checkpoint = ddpm_run_dir / "evaluator" / "selected_checkpoint.pt"
    _require(selection.is_file() and sha256_file(selection) == EVALUATOR_SELECTION_SHA256, "DDPM evaluator selection changed")
    _require(checkpoint.is_file() and checkpoint.stat().st_size == EVALUATOR_BYTES, "DDPM evaluator checkpoint byte mismatch")
    _require(sha256_file(checkpoint) == EVALUATOR_SHA256, "DDPM evaluator checkpoint hash mismatch")
    copied = run_dir / "input_bindings" / "selected_checkpoint.pt"
    _atomic_copy(checkpoint, copied)
    evaluator_binding = {
        "schema": VERSION + "-ddpm-evaluator-binding",
        "source_run": str(ddpm_run_dir.resolve()),
        "source_manifest_sha256": DDPM_MANIFEST_SHA256,
        "source_tree_digest": DDPM_TREE_DIGEST,
        "selection_file_sha256": EVALUATOR_SELECTION_SHA256,
        "checkpoint_bytes": EVALUATOR_BYTES,
        "checkpoint_sha256": EVALUATOR_SHA256,
        "copied_checkpoint": "input_bindings/selected_checkpoint.pt",
        "copied_checkpoint_sha256": sha256_file(copied),
        "weights_loaded_before_population_seal": 0,
    }
    _write_json(run_dir / "input_bindings" / "ddpm_evaluator_binding.json", evaluator_binding)
    predecessor = {
        "schema": VERSION + "-predecessor-bindings",
        "k128": {
            **k128,
            "status_sha256": K128_STATUS_SHA256,
            "outcome_sha256": K128_OUTCOME_SHA256,
            "report_sha256": K128_REPORT_SHA256,
            "required_route": K128_REQUIRED_ROUTE,
        },
        "ddpm": ddpm,
    }
    _write_json(run_dir / "input_bindings" / "predecessors.json", predecessor)
    return predecessor


def _write_inventory_authorities(
    run_dir: Path,
    *,
    config: DirectFluxMNISTConfig,
    development_images: np.ndarray,
    development_labels: np.ndarray,
    data_audit: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    _write_json(run_dir / "data" / "development_roles.json", data_audit)
    authority = derive_mass_to_uint8_authority(development_images[TRAIN_START:TRAIN_STOP])
    _write_json(run_dir / "input_bindings" / "mass_to_uint8.json", authority)
    inventory = build_path_inventory()
    starts = build_start_bank(config, inventory)
    targets = build_teacher_target_bank(
        development_images[VALIDATION_START:VALIDATION_STOP],
        development_labels[VALIDATION_START:VALIDATION_STOP],
        inventory,
    )
    inventory_rows = [
        {
            "path_id": str(inventory["path_ids"][index]),
            "path_index": int(inventory["path_indices"][index]),
            "requested_label": int(inventory["requested_labels"][index]),
            "within_class_index": int(inventory["within_class_indices"][index]),
            "source_seed": int(inventory["source_seeds"][index]),
            "generated_candidates": 1,
            "retained": 1,
        }
        for index in range(PATH_COUNT)
    ]
    _write_csv(run_dir / "inventory" / "path_inventory.csv", inventory_rows)
    _write_npz(
        run_dir / "inventory" / "start_bank.npz",
        starts=starts,
        labels=inventory["requested_labels"].astype(np.int64),
        path_ids=inventory["path_ids"].astype(np.str_),
        source_seeds=inventory["source_seeds"].astype(np.uint64),
    )
    _write_npz(
        run_dir / "inventory" / "teacher_target_bank.npz",
        masses=targets["masses"],
        source_images_uint8=targets["images_uint8"],
        rendered_images_uint8=mass_to_uint8(targets["masses"], authority),
        requested_labels=targets["requested_labels"],
        validation_local_ids=targets["validation_local_ids"],
        arff_global_row_ids=targets["arff_global_row_ids"],
        path_ids=targets["path_ids"],
    )
    _write_csv(
        run_dir / "inventory" / "teacher_target_ids.csv",
        [
            {
                "path_id": str(targets["path_ids"][index]),
                "requested_label": int(targets["requested_labels"][index]),
                "validation_local_id": int(targets["validation_local_ids"][index]),
                "arff_global_row_id": int(targets["arff_global_row_ids"][index]),
            }
            for index in range(PATH_COUNT)
        ],
    )
    _write_json(
        run_dir / "inventory" / "row_seeds.json",
        {
            "schema": VERSION + "-row-seeds",
            "row_root_seeds": ROW_ROOT_SEEDS,
            "inventory_seed": INVENTORY_SEED,
            "source_seed_base": SOURCE_SEED_BASE,
            "source_seed_count": PATH_COUNT,
            "review_seed": REVIEW_SEED,
            "test_only_smoke_seed": SMOKE_SEED,
            "rows_are_separately_randomized_not_crn": 1,
        },
    )
    start_seal = {
        "schema": VERSION + "-start-bank-sealed",
        "created_at": _utc_now(),
        "path_count": PATH_COUNT,
        "start_bank_sha256": sha256_file(run_dir / "inventory" / "start_bank.npz"),
        "starts_sha256": _hash_array(starts),
        "labels_sha256": _hash_array(inventory["requested_labels"]),
        "path_ids_sha256": _hash_array(inventory["path_ids"]),
        "path_inventory_sha256": sha256_file(run_dir / "inventory" / "path_inventory.csv"),
        "teacher_target_bank_sha256": sha256_file(run_dir / "inventory" / "teacher_target_bank.npz"),
        "teacher_target_mass_sha256": _hash_array(targets["masses"]),
        "teacher_target_ids_sha256": sha256_file(run_dir / "inventory" / "teacher_target_ids.csv"),
        "row_seeds_sha256": sha256_file(run_dir / "inventory" / "row_seeds.json"),
        "mass_transform_sha256": sha256_file(run_dir / "input_bindings" / "mass_to_uint8.json"),
        "evaluator_binding_sha256": sha256_file(run_dir / "input_bindings" / "ddpm_evaluator_binding.json"),
        "config_sha256": sha256_file(run_dir / "config.json"),
        "source_bindings_sha256": sha256_file(run_dir / "source_bindings.json"),
        "development_roles_sha256": sha256_file(run_dir / "data" / "development_roles.json"),
        "predecessor_bindings_sha256": sha256_file(run_dir / "input_bindings" / "predecessors.json"),
        "legacy_checkpoint_receipt_sha256": sha256_file(run_dir / "input_bindings" / "legacy_checkpoint_receipt.json"),
        "clean_state_sha256": sha256_file(run_dir / "input_bindings" / "clean_model_state.pt"),
        "clean_state_receipt_sha256": sha256_file(run_dir / "input_bindings" / "clean_model_state_receipt.json"),
        "terminal_test_content_rows_parsed": 0,
        "evaluator_weights_loaded": 0,
    }
    _write_json(run_dir / "inventory" / "START_BANK_SEALED.json", start_seal)
    return inventory, starts, targets, authority


def _model_state_semantic_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_clean_model(
    clean_state_path: Path,
    *,
    config: DirectFluxMNISTConfig,
    device: str | torch.device,
) -> DirectFluxUNet:
    state = torch.load(clean_state_path, map_location="cpu", weights_only=True)
    _require(isinstance(state, (dict, OrderedDict)) and len(state) == EXPECTED_STATE_TENSORS, "clean model state is invalid")
    model = DirectFluxUNet(config, base_channels=48, num_classes=10)
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _synthetic_teacher_preflight(
    run_dir: Path,
    *,
    starts: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    config: DirectFluxMNISTConfig,
) -> dict[str, Any]:
    deterministic = dataclasses.replace(
        config,
        adaptive_sampling=False,
        max_substeps=1,
        noise_weight=0.0,
    )
    result = run_teacher_row(
        starts[:4],
        labels[:4],
        targets[:4],
        deterministic,
        root_seed=ROW_ROOT_SEEDS["teacher"],
        device="cpu",
        path_ids=np.asarray([f"synthetic-teacher-{index}" for index in range(4)], dtype=np.str_),
        num_steps=1,
        schedule_steps=OUTER_STEPS,
        anchors=(0, 1),
    )
    before = np.sum((result.anchors[0].astype(np.float64) - targets[:4].astype(np.float64)) ** 2, axis=1)
    after = np.sum((result.anchors[-1].astype(np.float64) - targets[:4].astype(np.float64)) ** 2, axis=1)
    signature_checks = {
        "null_has_no_targets_parameter": "targets" not in inspect.signature(run_null_row).parameters,
        "learned_has_no_targets_parameter": "targets" not in inspect.signature(run_learned_row).parameters,
        "teacher_has_no_model_parameter": "model" not in inspect.signature(run_teacher_row).parameters,
        "teacher_moves_all_four_toward_target": bool(np.all(after < before)),
        "teacher_states_finite": bool(np.all(np.isfinite(result.anchors))),
        "teacher_states_nonnegative": float(result.anchors.min()) >= 0.0,
    }
    report = {
        "schema": VERSION + "-synthetic-teacher-preflight",
        "gate_type": "execution/integrity",
        "path_count": 4,
        "squared_l2_before": before.tolist(),
        "squared_l2_after": after.tolist(),
        "checks": {key: int(value) for key, value in signature_checks.items()},
        "passed": int(all(signature_checks.values())),
    }
    _write_json(run_dir / "preflight" / "synthetic_teacher.json", report)
    _require(bool(report["passed"]), "synthetic teacher or target firewall preflight failed")
    return report


def _probe_row(
    governor: ResourceGovernor,
    *,
    kind: str,
    row: str,
    starts: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    config: DirectFluxMNISTConfig,
    model: DirectFluxUNet,
    device: str | torch.device,
    steps: int,
) -> tuple[RowResult, dict[str, Any]]:
    governor.admit(
        kind,
        predicted_seconds=MAX_QUANTUM_SECONDS,
        predicted_next_bytes=2 * 1024 * 1024,
    )
    if row == "teacher":
        result = run_teacher_row(
            starts,
            labels,
            targets,
            config,
            root_seed=ROW_ROOT_SEEDS[row],
            device=device,
            path_ids=build_path_inventory()["path_ids"],
            num_steps=steps,
            schedule_steps=OUTER_STEPS,
            anchors=(0, steps),
        )
    elif row == "null":
        result = run_null_row(
            starts,
            labels,
            config,
            root_seed=ROW_ROOT_SEEDS[row],
            device=device,
            path_ids=build_path_inventory()["path_ids"],
            num_steps=steps,
            schedule_steps=OUTER_STEPS,
            anchors=(0, steps),
        )
    else:
        result = run_learned_row(
            starts,
            labels,
            model,
            config,
            root_seed=ROW_ROOT_SEEDS[row],
            device=device,
            path_ids=build_path_inventory()["path_ids"],
            num_steps=steps,
            schedule_steps=OUTER_STEPS,
            anchors=(0, steps),
        )
    receipt = governor.complete(
        kind,
        candidate_transitions=PATH_COUNT * steps,
        model_evaluations=(PATH_COUNT * steps if row == "learned" else 0),
    )
    return result, receipt


def _run_device_preflight(
    run_dir: Path,
    *,
    governor: ResourceGovernor,
    starts: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    config: DirectFluxMNISTConfig,
    model: DirectFluxUNet,
    device: str | torch.device,
) -> dict[str, Any]:
    model_digest_before = _model_state_semantic_digest(model)
    _, warmup_receipt = _probe_row(
        governor,
        kind="device_warmup",
        row="learned",
        starts=starts,
        labels=labels,
        targets=targets,
        config=config,
        model=model,
        device=device,
        steps=1,
    )
    first, first_receipt = _probe_row(
        governor,
        kind="learned_determinism_probe_1",
        row="learned",
        starts=starts,
        labels=labels,
        targets=targets,
        config=config,
        model=model,
        device=device,
        steps=8,
    )
    second, second_receipt = _probe_row(
        governor,
        kind="learned_determinism_probe_2",
        row="learned",
        starts=starts,
        labels=labels,
        targets=targets,
        config=config,
        model=model,
        device=device,
        steps=8,
    )
    null_result, null_receipt = _probe_row(
        governor,
        kind="null_timing_probe",
        row="null",
        starts=starts,
        labels=labels,
        targets=targets,
        config=config,
        model=model,
        device=device,
        steps=8,
    )
    teacher_result, teacher_receipt = _probe_row(
        governor,
        kind="teacher_timing_probe",
        row="teacher",
        starts=starts,
        labels=labels,
        targets=targets,
        config=config,
        model=model,
        device=device,
        steps=8,
    )
    model_digest_after = _model_state_semantic_digest(model)
    deterministic_checks = {
        "learned_anchor_bytes_identical": bool(np.array_equal(first.anchors, second.anchors)),
        "learned_scientific_digest_identical": first.scientific_digest == second.scientific_digest,
        "learned_retry_counts_identical": [row["accepted_substeps"] for row in first.telemetry]
        == [row["accepted_substeps"] for row in second.telemetry],
        "learned_clipping_counts_identical": [row["accepted_clipped"] for row in first.telemetry]
        == [row["accepted_clipped"] for row in second.telemetry],
        "model_state_unchanged": model_digest_before == model_digest_after,
        "null_health": bool(np.all(np.isfinite(null_result.anchors))) and float(null_result.anchors.min()) >= 0.0,
        "teacher_health": bool(np.all(np.isfinite(teacher_result.anchors))) and float(teacher_result.anchors.min()) >= 0.0,
    }
    deterministic_report = {
        "schema": VERSION + "-deterministic-replay",
        "schedule_steps": OUTER_STEPS,
        "executed_probe_steps": 8,
        "checks": {key: int(value) for key, value in deterministic_checks.items()},
        "passed": int(all(deterministic_checks.values())),
        "first_scientific_digest": first.scientific_digest,
        "second_scientific_digest": second.scientific_digest,
        "first_anchor_sha256": _hash_array(first.anchors),
        "second_anchor_sha256": _hash_array(second.anchors),
        "model_state_semantic_sha256_before": model_digest_before,
        "model_state_semantic_sha256_after": model_digest_after,
        "timing_seconds": {
            "charged_warmup": float(warmup_receipt["elapsed_seconds"]),
            "learned8_first": float(first_receipt["elapsed_seconds"]),
            "learned8_second": float(second_receipt["elapsed_seconds"]),
            "null8": float(null_receipt["elapsed_seconds"]),
            "teacher8": float(teacher_receipt["elapsed_seconds"]),
        },
        "timing_and_allocator_excluded_from_scientific_digest": 1,
    }
    _write_json(run_dir / "preflight" / "deterministic_replay.json", deterministic_report)
    _require(bool(deterministic_report["passed"]), "production-device deterministic replay failed")
    peak_fraction = max(
        [float(event.get("cuda_fraction", 0.0)) for event in governor.events] + [0.0]
    )
    projection = resource_projection(
        charged_active_seconds=governor.active_seconds,
        teacher8_seconds=float(teacher_receipt["elapsed_seconds"]),
        null8_seconds=float(null_receipt["elapsed_seconds"]),
        learned8_seconds=max(float(first_receipt["elapsed_seconds"]), float(second_receipt["elapsed_seconds"])),
        projected_persisted_bytes=70 * 1024 * 1024,
        peak_cuda_fraction=peak_fraction,
        budget=governor.budget,
    )
    _write_json(run_dir / "preflight" / "resource_projection.json", projection)
    if not bool(projection["passed"]):
        raise ResourceStop(str(projection["stop_reason"]))
    return {
        "determinism": deterministic_report,
        "projection": projection,
        "row_quantum_seconds": {
            "teacher": max(0.001, 1.25 * float(teacher_receipt["elapsed_seconds"])),
            "null": max(0.001, 1.25 * float(null_receipt["elapsed_seconds"])),
            "learned": max(0.001, 1.25 * max(float(first_receipt["elapsed_seconds"]), float(second_receipt["elapsed_seconds"]))),
        },
    }


def _partial_result_from_callback(value: Mapping[str, Any], *, labels: np.ndarray, path_ids: np.ndarray, root_seed: int) -> RowResult:
    saved = [np.asarray(array, dtype=np.float32) for array in value["saved_anchors"]]
    steps = [int(item) for item in value["saved_steps"]]
    completed = int(value["completed_step"])
    if steps[-1] != completed:
        saved.append(value["state"].detach().cpu().numpy().astype(np.float32, copy=True))
        steps.append(completed)
    anchors = np.stack(saved).astype(np.float32, copy=False)
    telemetry = list(value["telemetry"])
    return RowResult(
        row=str(value["row"]),
        anchors=anchors,
        labels=np.asarray(labels, dtype=np.int64).copy(),
        path_ids=np.asarray(path_ids, dtype=np.str_).copy(),
        telemetry=telemetry,
        root_seed=int(root_seed),
        scientific_digest=_scientific_row_digest(anchors, telemetry),
        anchor_steps=np.asarray(steps, dtype=np.int64),
    )


def _execute_full_row(
    run_dir: Path,
    *,
    governor: ResourceGovernor,
    row: str,
    starts: np.ndarray,
    labels: np.ndarray,
    path_ids: np.ndarray,
    targets: np.ndarray,
    config: DirectFluxMNISTConfig,
    model: DirectFluxUNet,
    device: str | torch.device,
    predicted_eight_step_seconds: float,
) -> RowResult:
    block = 0
    open_kind = f"{row}_row_q{block:02d}"
    governor.admit(
        open_kind,
        predicted_seconds=min(MAX_QUANTUM_SECONDS, float(predicted_eight_step_seconds)),
        predicted_next_bytes=4 * 1024 * 1024,
    )

    def callback(value: Mapping[str, Any]) -> None:
        nonlocal block, open_kind
        completed = int(value["completed_step"])
        if completed % 8 != 0:
            return
        partial = _partial_result_from_callback(
            value,
            labels=labels,
            path_ids=path_ids,
            root_seed=ROW_ROOT_SEEDS[row],
        )
        _save_row_result(run_dir, partial, partial=(completed < OUTER_STEPS))
        if completed == OUTER_STEPS:
            return
        governor.complete(
            open_kind,
            candidate_transitions=PATH_COUNT * 8,
            model_evaluations=(PATH_COUNT * 8 if row == "learned" else 0),
        )
        block += 1
        open_kind = f"{row}_row_q{block:02d}"
        governor.admit(
            open_kind,
            predicted_seconds=min(MAX_QUANTUM_SECONDS, float(predicted_eight_step_seconds)),
            predicted_next_bytes=4 * 1024 * 1024,
        )

    if row == "teacher":
        result = run_teacher_row(
            starts,
            labels,
            targets,
            config,
            root_seed=ROW_ROOT_SEEDS[row],
            device=device,
            path_ids=path_ids,
            outer_step_callback=callback,
        )
    elif row == "null":
        result = run_null_row(
            starts,
            labels,
            config,
            root_seed=ROW_ROOT_SEEDS[row],
            device=device,
            path_ids=path_ids,
            outer_step_callback=callback,
        )
    else:
        before = _model_state_semantic_digest(model)
        result = run_learned_row(
            starts,
            labels,
            model,
            config,
            root_seed=ROW_ROOT_SEEDS[row],
            device=device,
            path_ids=path_ids,
            outer_step_callback=callback,
        )
        after = _model_state_semantic_digest(model)
        _require(before == after, "learned sampling mutated the clean model state")
        _write_json(
            run_dir / "telemetry" / "model_state_identity.json",
            {"before_sha256": before, "after_sha256": after, "identical": 1},
        )
    _require((run_dir / "populations" / f"{row}.npz").is_file(), f"{row} final population was not durably persisted")
    try:
        governor.complete(
            open_kind,
            candidate_transitions=PATH_COUNT * 8,
            model_evaluations=(PATH_COUNT * 8 if row == "learned" else 0),
        )
    except BaseException as error:
        # The final population is already durable.  Attach it only so the outer
        # failure transaction can render the required last-valid task image.
        setattr(error, "durable_full_row_result", result)
        raise
    (run_dir / "populations" / f"partial_{row}.npz").unlink(missing_ok=True)
    (run_dir / "telemetry" / f"partial_{row}_steps.csv").unlink(missing_ok=True)
    return result


def _persist_partial_failure(
    run_dir: Path,
    partial: RowResult,
) -> None:
    _save_row_result(run_dir, partial, partial=True)
    authority = _read_json(run_dir / "input_bindings" / "mass_to_uint8.json")
    last = mass_to_uint8(partial.anchors[-1], authority)
    write_contact_sheet(
        run_dir / "images" / f"partial_{partial.row}_latest.png",
        last,
        columns=16,
        scale=2,
    )


def _persist_durable_full_failure_image(run_dir: Path, result: RowResult) -> None:
    authority = _read_json(run_dir / "input_bindings" / "mass_to_uint8.json")
    last = mass_to_uint8(result.anchors[-1], authority)
    write_contact_sheet(
        run_dir / "images" / f"partial_{result.row}_latest.png",
        last,
        columns=16,
        scale=2,
    )


def _resource_kind_closed(events: Sequence[Mapping[str, Any]], kind: str) -> bool:
    admitted = 0
    closed = 0
    for event in events:
        if str(event.get("kind", "")) != kind:
            continue
        if event.get("event") == "admit":
            admitted += 1
        elif event.get("event") in {"complete", "failed-complete", "interrupted-close"}:
            closed += 1
    _require(admitted in {0, 1} and closed in {0, 1} and closed <= admitted, f"resource history for {kind} is ambiguous")
    return admitted == 1 and closed == 1


def _resume_existing_production(
    run_dir: Path,
    *,
    mode: str,
    governor: ResourceGovernor,
    scientific_config: DirectFluxMNISTConfig,
    development_images: np.ndarray,
    development_labels: np.ndarray,
    inventory: Mapping[str, np.ndarray],
    starts: np.ndarray,
    targets: Mapping[str, np.ndarray],
    clean_path: Path,
    arff_path: Path,
    device: torch.device,
) -> int:
    """Resume only at the two whole-run boundaries allowed by the plan."""

    _require(mode in {"rerun_all_rows", "continue_sealed"}, "invalid production re-entry mode")
    current_stage = "preflight" if mode == "rerun_all_rows" else "scoring"
    current_row: str | None = None
    try:
        if mode == "rerun_all_rows":
            governor.admit(
                "cpu_preflight",
                predicted_seconds=30.0,
                predicted_next_bytes=1 * 1024 * 1024,
            )
            _synthetic_teacher_preflight(
                run_dir,
                starts=starts,
                labels=inventory["requested_labels"],
                targets=targets["masses"],
                config=scientific_config,
            )
            governor.complete("cpu_preflight")
            governor.admit(
                "model_load_to_device",
                predicted_seconds=20.0,
                predicted_next_bytes=1 * 1024 * 1024,
            )
            model = _load_clean_model(clean_path, config=scientific_config, device=device)
            governor.complete("model_load_to_device")
            preflight = _run_device_preflight(
                run_dir,
                governor=governor,
                starts=starts,
                labels=inventory["requested_labels"],
                targets=targets["masses"],
                config=scientific_config,
                model=model,
                device=device,
            )
            _record_stage(run_dir, "preflight")
            row_results: dict[str, RowResult] = {}
            for row in ("teacher", "null", "learned"):
                current_row = row
                current_stage = f"{row}_row"
                result = _execute_full_row(
                    run_dir,
                    governor=governor,
                    row=row,
                    starts=starts,
                    labels=inventory["requested_labels"],
                    path_ids=inventory["path_ids"],
                    targets=targets["masses"],
                    config=scientific_config,
                    model=model,
                    device=device,
                    predicted_eight_step_seconds=float(preflight["row_quantum_seconds"][row]),
                )
                row_results[row] = result
                _record_stage(run_dir, current_stage)
                current_row = None
            _write_json(
                run_dir / "telemetry" / "summary.json",
                {
                    "schema": VERSION + "-telemetry-summary",
                    "rows": {
                        name: {
                            "step_count": len(result.telemetry),
                            "scientific_digest": result.scientific_digest,
                            "maximum_mass_error": max(float(entry["maximum_mass_error"]) for entry in result.telemetry),
                            "maximum_clipping_fraction": max(float(entry["accepted_clipping_fraction"]) for entry in result.telemetry),
                            "maximum_accepted_substeps": max(int(entry["accepted_substeps"]) for entry in result.telemetry),
                        }
                        for name, result in row_results.items()
                    },
                },
            )
            current_stage = "population_seal"
            governor.admit(
                "population_seal_and_scoring",
                predicted_seconds=30.0,
                predicted_next_bytes=25 * 1024 * 1024,
            )
            seal_populations(run_dir)
            _record_stage(run_dir, "population_seal")
        else:
            _verify_population_seal(run_dir)
            for row in ("teacher", "null", "learned"):
                _verify_one_row_result(run_dir, row, partial=False)
            _verify_telemetry_summary(run_dir)
            stages = [str(event.get("stage")) for event in _stage_events(run_dir)]
            if "scoring" not in stages:
                _clear_postseal_outputs(run_dir)
                governor.admit(
                    "population_seal_and_scoring",
                    predicted_seconds=30.0,
                    predicted_next_bytes=25 * 1024 * 1024,
                )

        stages = [str(event.get("stage")) for event in _stage_events(run_dir)]
        if "scoring" not in stages:
            current_stage = "scoring"
            evaluation = evaluate_sealed_populations(
                run_dir,
                arff_path=arff_path,
                device=device,
                development_images=development_images,
                development_labels=development_labels,
            )
            _record_stage(run_dir, "scoring")
            governor.complete("population_seal_and_scoring")
        else:
            evaluation = _verify_evaluation(run_dir, _verify_population_seal(run_dir))

        gates = _machine_gates(
            run_dir,
            governor=governor,
            teacher_control=evaluation["teacher_control"],
        )
        _require(int(gates["gate_d"]["passed"]) == 1, "full-interface target-informed positive control failed Gate D")

        stages = [str(event.get("stage")) for event in _stage_events(run_dir)]
        if "review_prepare" not in stages:
            current_stage = "review_prepare"
            governor.admit(
                "review_prepare",
                predicted_seconds=10.0,
                predicted_next_bytes=15 * 1024 * 1024,
            )
            prepare_blind_review(run_dir)
            governor.complete("review_prepare")
            _record_stage(run_dir, "review_prepare")
        else:
            _verify_review_bundle(run_dir, _verify_population_seal(run_dir))

        current_stage = "machine_terminalization"
        terminal_already_closed = _resource_kind_closed(
            governor.events,
            "machine_terminalization",
        )
        if not terminal_already_closed:
            governor.admit(
                "machine_terminalization",
                predicted_seconds=5.0,
                predicted_next_bytes=2 * 1024 * 1024,
                reserve_remaining_seconds=0.0,
            )
            _machine_gates(
                run_dir,
                governor=governor,
                teacher_control=evaluation["teacher_control"],
            )
        _write_json(
            run_dir / "status.json",
            {
                "schema": VERSION + "-status",
                "state": "awaiting_human_review",
                "route": "awaiting_human_review",
                "error": None,
                "updated_at": _utc_now(),
                "whole_run_restart_required": 0,
            },
        )
        if "machine_terminalization" not in [str(event.get("stage")) for event in _stage_events(run_dir)]:
            _record_stage(run_dir, "machine_terminalization")
        _write_reports(run_dir, None)
        _seal_manifest(run_dir)
        if not terminal_already_closed:
            governor.complete("machine_terminalization")
        _write_reports(run_dir, None)
        manifest = _seal_manifest(run_dir)
        receipt = verify_run(run_dir)
        print(
            json.dumps(
                {
                    "passed": int(receipt["passed"]),
                    "state": "awaiting_human_review",
                    "artifact_count": manifest["artifact_count"],
                    "tree_digest": manifest["tree_digest"],
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except BaseException as raw_error:
        error: BaseException
        if isinstance(raw_error, (IntegrityFailure, ResourceStop)):
            error = raw_error
        else:
            error = IntegrityFailure(f"operational failure {type(raw_error).__name__}: {raw_error}")
        partial = getattr(raw_error, "partial_row_result", None)
        if isinstance(partial, RowResult):
            _persist_partial_failure(run_dir, partial)
        durable_full = getattr(raw_error, "durable_full_row_result", None)
        if isinstance(durable_full, RowResult):
            _persist_durable_full_failure_image(run_dir, durable_full)
        result = _finalize_failure(
            run_dir,
            error,
            governor=governor,
            failed_stage=current_stage,
            partial_row=current_row,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 3 if isinstance(error, ResourceStop) else 4


def _record_review_transaction(
    run_dir: str | Path,
    answers: str | Path,
    *,
    reviewer: str,
    confirm_manual_review: bool,
    recovered_governor: ResourceGovernor | None = None,
    transaction_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    _require(confirm_manual_review is True, "manual-review confirmation is required")
    _require(_read_json(root / "status.json").get("state") == "awaiting_human_review", "run is not awaiting review")
    run_config = _read_json(root / "config.json")
    review_device = str(run_config.get("execution_authority", {}).get("device", "cpu"))
    if recovered_governor is None:
        governor: ResourceGovernor | None = None
    else:
        governor = recovered_governor
    _verify_population_seal(root)
    ready = _read_json(root / "review" / "READY.json")
    scoring_ready = _read_json(root / "evaluation" / "SCORING_READY.json")
    _require(scoring_ready["population_seal_sha256"] == sha256_file(root / "populations" / "POPULATIONS_SEALED.json"), "scoring-ready population seal mismatch")
    _require(ready["population_seal_sha256"] == sha256_file(root / "populations" / "POPULATIONS_SEALED.json"), "review-ready population seal mismatch")
    _require(ready["template_sha256"] == sha256_file(root / "review" / "human_review_template.csv"), "review template changed")
    _require(ready["review_key_sha256"] == sha256_file(root / "review" / "review_key.json"), "review key changed")
    if governor is None:
        # Readiness is checked before the full tree receipt so even a compact test
        # fixture fails for the named authority; production then requires the full
        # read-only verifier before opening the review resource event.
        verify_run(root)
        governor = ResourceGovernor.rehydrate(root, device=review_device)
    if transaction_state is not None:
        transaction_state["governor"] = governor
    governor.admit(
        "human_review_terminalization",
        predicted_seconds=10.0,
        predicted_next_bytes=2 * 1024 * 1024,
        reserve_remaining_seconds=0.0,
    )
    answer_path = Path(answers)
    submitted = root / "review" / "human_review_answers.csv"
    _atomic_bytes(submitted, answer_path.read_bytes())
    overall = score_human_review(
        submitted,
        root / "review" / "review_key.json",
        reviewer=reviewer,
        confirm_manual_review=True,
    )
    key_data = _read_json(root / "review" / "review_key.json")
    membership_data = _read_json(root / "review" / "private_membership.json")
    key = {str(entry["sample_id"]): entry for entry in key_data["entries"]}
    membership = {str(entry["member_id"]): entry for entry in membership_data["entries"]}
    scored: list[dict[str, Any]] = []
    with submitted.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            blind = str(row["sample_id"])
            key_entry = key[blind]
            member = membership[str(key_entry["source_sample_id"])]
            assignment = str(row["assigned_label"]).strip().lower()
            recognizable = int(assignment.isdigit())
            agreement = int(assignment == str(member["requested_label"]))
            scored.append({**member, "blind_id": blind, "assigned_label": assignment, "recognizable": recognizable, "agreement": agreement})
    by_row: dict[str, Any] = {}
    for row_name in ("learned", "null"):
        entries = [entry for entry in scored if entry["row"] == row_name]
        _require(len(entries) == 40, f"review row {row_name} does not have 40 entries")
        by_row[row_name] = {
            "sample_count": len(entries),
            "recognizable_count": sum(int(entry["recognizable"]) for entry in entries),
            "human_recognizability": float(np.mean([entry["recognizable"] for entry in entries])),
            "requested_label_agreement_count": sum(int(entry["agreement"]) for entry in entries),
            "human_requested_label_agreement": float(np.mean([entry["agreement"] for entry in entries])),
            "by_class": {
                str(digit): {
                    "count": len([entry for entry in entries if entry["requested_label"] == digit]),
                    "recognizable_count": sum(entry["recognizable"] for entry in entries if entry["requested_label"] == digit),
                    "agreement_count": sum(entry["agreement"] for entry in entries if entry["requested_label"] == digit),
                }
                for digit in range(10)
            },
        }
    _write_json(root / "review" / "human_review.json", overall)
    _write_json(root / "review" / "human_review_by_row.json", {"schema": VERSION + "-human-by-row", "rows": by_row, "scored_entries": scored})
    gates = _read_json(root / "gates.json")
    learned_metrics = _read_json(root / "evaluation" / "learned_metrics.json")
    null_metrics = _read_json(root / "evaluation" / "null_metrics.json")
    learned_human_positive = (
        by_row["learned"]["human_recognizability"] >= REVIEW_POSITIVE_RECOGNIZABILITY
        and by_row["learned"]["human_requested_label_agreement"] >= REVIEW_POSITIVE_AGREEMENT
    )
    human_exceeds_null = (
        by_row["learned"]["human_requested_label_agreement"]
        > by_row["null"]["human_requested_label_agreement"]
    )
    learned_classifier = float(learned_metrics["classifier"]["requested_label_accuracy"])
    null_classifier = float(null_metrics["classifier"]["requested_label_accuracy"])
    classifier_components = {
        "absolute_accuracy": learned_classifier >= CLASSIFIER_POSITIVE_ACCURACY,
        "exceeds_null": learned_classifier > null_classifier,
    }
    noncollapse_components = {
        "zero_duplicate_pairs": int(learned_metrics["duplicates"]["duplicate_pair_count"]) == 0,
        "diversity": float(learned_metrics["diversity"]["aggregate_median_ratio"]) >= DIVERSITY_POSITIVE_RATIO,
    }
    classifier_positive = all(classifier_components.values())
    noncollapse_positive = all(noncollapse_components.values())
    gates_a_to_d = all(int(gates[name]["passed"]) == 1 for name in ("gate_a", "gate_b", "gate_c", "gate_d"))
    route = route_outcome(
        gates_a_to_d_passed=gates_a_to_d,
        learned_human_positive=learned_human_positive,
        learned_classifier_positive=classifier_positive,
        learned_noncollapse_positive=noncollapse_positive,
        learned_exceeds_null=human_exceeds_null,
    )
    gate_e_conditions = {
        "learned_human_recognizability_at_least_0_90": by_row["learned"]["human_recognizability"] >= REVIEW_POSITIVE_RECOGNIZABILITY,
        "learned_human_requested_label_agreement_at_least_0_75": by_row["learned"]["human_requested_label_agreement"] >= REVIEW_POSITIVE_AGREEMENT,
        "learned_classifier_accuracy_at_least_0_80": learned_classifier >= CLASSIFIER_POSITIVE_ACCURACY,
        "learned_zero_duplicate_pairs": int(learned_metrics["duplicates"]["duplicate_pair_count"]) == 0,
        "learned_diversity_ratio_at_least_0_25": float(learned_metrics["diversity"]["aggregate_median_ratio"]) >= DIVERSITY_POSITIVE_RATIO,
        "learned_human_agreement_exceeds_null": human_exceeds_null,
        "learned_classifier_accuracy_exceeds_null": learned_classifier > null_classifier,
        "gates_a_to_d_passed": gates_a_to_d,
    }
    gates["gate_e"] = {
        "gate_type": "diagnostic threshold",
        "state": "complete",
        "passed": int(all(gate_e_conditions.values())),
        "conditions": {key: int(value) for key, value in gate_e_conditions.items()},
        "values": {
            "learned_human_recognizability": by_row["learned"]["human_recognizability"],
            "learned_human_requested_label_agreement": by_row["learned"]["human_requested_label_agreement"],
            "null_human_requested_label_agreement": by_row["null"]["human_requested_label_agreement"],
            "learned_classifier_accuracy": learned_classifier,
            "null_classifier_accuracy": null_classifier,
            "learned_duplicate_pair_count": int(learned_metrics["duplicates"]["duplicate_pair_count"]),
            "learned_diversity_ratio": float(learned_metrics["diversity"]["aggregate_median_ratio"]),
        },
    }
    _write_json(root / "gates.json", gates)
    outcome = {
        "schema": VERSION + "-outcome",
        "research_mode": RESEARCH_MODE,
        "state": "complete",
        "route": route,
        "gates_a_to_d_passed": int(gates_a_to_d),
        "human_marker": {
            "passed": int(learned_human_positive and human_exceeds_null),
            "learned": by_row["learned"],
            "null": by_row["null"],
            "learned_agreement_exceeds_null": int(human_exceeds_null),
        },
        "classifier_marker": {
            "passed": int(classifier_positive),
            "components": {key: int(value) for key, value in classifier_components.items()},
            "learned_accuracy": learned_classifier,
            "null_accuracy": null_classifier,
            "learned_minus_null": learned_classifier - null_classifier,
        },
        "noncollapse_marker": {
            "passed": int(noncollapse_positive),
            "components": {key: int(value) for key, value in noncollapse_components.items()},
        },
        "full_scale_auto_launched": 0,
        "next_action": _next_action(route),
    }
    _write_json(root / "outcome.json", outcome)
    _write_json(
        root / "status.json",
        {
            "schema": VERSION + "-status",
            "state": "complete",
            "route": route,
            "error": None,
            "updated_at": _utc_now(),
            "whole_run_restart_required": 0,
        },
    )
    _record_stage(root, "human_review_terminalization")
    _write_reports(root, outcome)
    _seal_manifest(root)
    governor.complete("human_review_terminalization")
    _write_reports(root, outcome)
    _seal_manifest(root)
    receipt = verify_run(root)
    return {**outcome, "verification": receipt}


def _governor_from_failure_ledger(run_dir: Path, *, device: str | torch.device) -> ResourceGovernor:
    """Hydrate a ledger for terminal failure sealing, including a post-check stop."""

    try:
        return ResourceGovernor.rehydrate(
            run_dir,
            device=device,
            recover_interrupted=True,
        )
    except IntegrityFailure as error:
        ledger = _read_json(run_dir / "resource_ledger.json")
        _require(
            type(ledger.get("failed_admission")) is dict,
            f"cannot recover review resource ledger: {error}",
        )
        governor = ResourceGovernor(
            run_dir,
            ResourceBudget(**ledger["budget"]),
            device=device,
        )
        governor.active_seconds = float(ledger["active_seconds"])
        governor.events = list(ledger["events"])
        governor.failed_admission = dict(ledger["failed_admission"])
        _require(not ledger.get("open_events"), "failed review ledger also contains an open event")
        return governor


def record_review(run_dir: str | Path, answers: str | Path, *, reviewer: str,
                  confirm_manual_review: bool) -> dict[str, Any]:
    """Record review transactionally; every admitted failure is terminalized."""

    root = Path(run_dir)
    ledger_path = root / "resource_ledger.json"
    initial_event_count = (
        len(_read_json(ledger_path).get("events", [])) if ledger_path.is_file() else 0
    )
    recovered_governor: ResourceGovernor | None = None
    transaction_state: dict[str, Any] = {}
    if ledger_path.is_file() and (root / "status.json").is_file():
        ledger_before = _read_json(ledger_path)
        if (
            _read_json(root / "status.json").get("state") == "awaiting_human_review"
            and ledger_before.get("open_events") == ["human_review_terminalization"]
            and ledger_before.get("failed_admission") is None
        ):
            recovered_governor = ResourceGovernor.rehydrate(
                root,
                device=str(
                    _read_json(root / "config.json")
                    .get("execution_authority", {})
                    .get("device", "cpu")
                ),
                recover_interrupted=True,
            )
            for relative in (
                "review/human_review_answers.csv",
                "review/human_review.json",
                "review/human_review_by_row.json",
                "outcome.json",
            ):
                (root / relative).unlink(missing_ok=True)
            if (root / "gates.json").is_file():
                gates = _read_json(root / "gates.json")
                gates["gate_e"] = {
                    "gate_type": "diagnostic threshold",
                    "state": "pending",
                    "passed": None,
                    "conditions": {},
                }
                _write_json(root / "gates.json", gates)
            events = [
                event
                for event in _stage_events(root)
                if event.get("stage") != "human_review_terminalization"
            ]
            _write_json(
                root / "stage_ledger.json",
                {"schema": VERSION + "-stage-ledger", "events": events},
            )
            # The interrupted receipt is now durable.  Restore and authenticate
            # the exact pre-review tree before opening a second review attempt.
            _write_reports(root, None)
            _seal_manifest(root)
            verify_run(root)
    try:
        return _record_review_transaction(
            root,
            answers,
            reviewer=reviewer,
            confirm_manual_review=confirm_manual_review,
            recovered_governor=recovered_governor,
            transaction_state=transaction_state,
        )
    except BaseException as raw_error:
        if not ledger_path.is_file():
            raise
        ledger = _read_json(ledger_path)
        if len(ledger.get("events", [])) == initial_event_count and not ledger.get("open_events"):
            # Preconditions failed before any externally visible review mutation.
            raise
        error: BaseException
        if isinstance(raw_error, (IntegrityFailure, ResourceStop)):
            error = raw_error
        else:
            error = IntegrityFailure(
                f"operational review failure {type(raw_error).__name__}: {raw_error}"
            )
        active_governor = transaction_state.get("governor")
        governor = (
            active_governor
            if isinstance(active_governor, ResourceGovernor)
            else _governor_from_failure_ledger(
                root,
                device=str(
                    _read_json(root / "config.json")
                    .get("execution_authority", {})
                    .get("device", "cpu")
                ),
            )
        )
        # A failed terminal review is evidence, not a completed Gate-E route.
        if (root / "gates.json").is_file():
            gates = _read_json(root / "gates.json")
            gates["gate_e"] = {
                "gate_type": "diagnostic threshold",
                "state": "pending",
                "passed": None,
                "conditions": {},
            }
            _write_json(root / "gates.json", gates)
        (root / "outcome.json").unlink(missing_ok=True)
        result = _finalize_failure(
            root,
            error,
            governor=governor,
            failed_stage="human_review_terminalization",
        )
        return result


def route_outcome(*, gates_a_to_d_passed: bool, learned_human_positive: bool,
                  learned_classifier_positive: bool,
                  learned_noncollapse_positive: bool = True,
                  learned_exceeds_null: bool) -> str:
    if not gates_a_to_d_passed:
        return "invalid_repair_same_experiment"
    if not learned_exceeds_null or not learned_human_positive or not learned_noncollapse_positive:
        return "factor_one_negative_stop_checkpoint_line"
    if learned_classifier_positive:
        return "factor_one_feasible"
    return "human_positive_evaluator_disagreement"


_outcome_route = route_outcome


def _next_action(route: str) -> str:
    actions = {
        "invalid_repair_same_experiment": "Repair only the localized integrity/interface defect, then rerun this unchanged decision once.",
        "factor_one_feasible": "Freeze the exploratory global edge-flux result; any replication requires separate approval and no local checkpoint tuning.",
        "human_positive_evaluator_disagreement": "Preserve the human-positive result and audit symmetric evaluator/render disagreement; do not tune this evaluator or select samples.",
        "factor_one_negative_stop_checkpoint_line": "Stop this historical checkpoint line; separately plan and approve a materially different fixed-grid/on-policy architecture or stop the hypothesis.",
    }
    _require(route in actions, f"unknown outcome route: {route}")
    return actions[route]


def _route_scoped_claim(route: str | None) -> str:
    if route == "factor_one_feasible":
        return (
            "On 160 prespecified fresh low-frequency source measures, the hash-pinned historical "
            "global edge-flux checkpoint under the bound current sampler produced a factor-one, "
            "unselected exploratory population that met the frozen recognizability, requested-label, "
            "uniqueness, and diversity markers and exceeded the separately randomized zero-conditioning "
            "population in aggregate human and classifier label agreement."
        )
    if route in {
        "factor_one_negative_stop_checkpoint_line",
        "human_positive_evaluator_disagreement",
    }:
        return (
            "Under the bound current sampler, fixed global rasterization, 160 prespecified fresh "
            "low-frequency starts, and no candidate rejection or post-hoc selection, the pinned "
            "checkpoint did not establish all frozen exploratory image-feasibility markers, although "
            "the target-informed full-interface control passed. This scopes the result to this "
            "checkpoint, source law, sampler, and transform."
        )
    if route == "invalid_repair_same_experiment":
        return (
            "This run establishes only a scoped execution or integrity failure; it does not establish "
            "a task-level result for the learned checkpoint."
        )
    return (
        "The machine controls and objective artifacts are not yet paired with the prespecified blinded "
        "human result, so no final image-feasibility route is available."
    )


def _command_argv(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "mnist.diag_d0_eulerian_edge_flux_replay",
        "run",
        "--run-dir",
        str(args.run_dir),
        "--legacy-checkpoint",
        str(args.legacy_checkpoint),
        "--ddpm-run-dir",
        str(args.ddpm_run_dir),
        "--k128-run-dir",
        str(args.k128_run_dir),
        "--arff",
        str(args.arff),
        "--device",
        str(args.device),
        "--approval-id",
        str(args.approval_id),
        "--max-active-seconds",
        str(args.max_active_seconds),
        "--max-storage-mib",
        str(args.max_storage_mib),
        "--max-cuda-fraction",
        str(args.max_cuda_fraction),
    ]


def _initial_config(
    args: argparse.Namespace,
    *,
    repository_root: Path,
    budget: ResourceBudget,
) -> dict[str, Any]:
    argv = _command_argv(args)
    return {
        "schema": VERSION + "-config",
        "version": VERSION,
        "research_mode": RESEARCH_MODE,
        "created_at": _utc_now(),
        "command": subprocess.list2cmdline(argv),
        "argv": argv,
        "repository_root": str(repository_root.resolve()),
        "git_revision": _git_revision(repository_root),
        "restart_count": 0,
        "scientific_configuration": {
            "path_count": PATH_COUNT,
            "paths_per_class": PATHS_PER_CLASS,
            "outer_steps": OUTER_STEPS,
            "anchors": list(ANCHORS),
            "row_root_seeds": ROW_ROOT_SEEDS,
            "inventory_seed": INVENTORY_SEED,
            "source_seed_base": SOURCE_SEED_BASE,
            "review_seed": REVIEW_SEED,
            "test_only_smoke_seed": SMOKE_SEED,
            "review_offsets": list(REVIEW_WITHIN_CLASS),
            "replay_policy": {
                "generated_candidates_per_path": 1,
                "selector": None,
                "all_candidates_retained": 1,
                "adaptive_numerical_retry_substeps": [1, 2, 4],
            },
            "thresholds": {
                "teacher_median_ratio64": 0.80,
                "teacher_median_ratio256": 0.20,
                "teacher_improved_paths": 144,
                "teacher_classifier_accuracy": 0.80,
                "human_recognizability": REVIEW_POSITIVE_RECOGNIZABILITY,
                "human_requested_label_agreement": REVIEW_POSITIVE_AGREEMENT,
                "classifier_requested_label_accuracy": CLASSIFIER_POSITIVE_ACCURACY,
                "duplicate_pair_count": 0,
                "diversity_ratio": DIVERSITY_POSITIVE_RATIO,
            },
            "legacy_config_semantic_sha256": LEGACY_CONFIG_SHA256,
            "mass_transform": {
                "derivation_slice": [TRAIN_START, TRAIN_STOP],
                "numerator": MASS_SCALE_NUMERATOR,
                "denominator": MASS_SCALE_DENOMINATOR,
                "float_hex": MASS_SCALE_HEX,
            },
        },
        "execution_authority": {
            "approval_id": str(args.approval_id),
            "device": str(args.device),
            "max_active_seconds": budget.max_active_seconds,
            "max_storage_bytes": budget.max_storage_bytes,
            "max_cuda_fraction": budget.max_cuda_fraction,
            "reserve_seconds": budget.reserve_seconds,
            "maximum_quantum_seconds": budget.maximum_quantum_seconds,
        },
        "input_paths": {
            "legacy_checkpoint": str(Path(args.legacy_checkpoint).resolve()),
            "ddpm_run_dir": str(Path(args.ddpm_run_dir).resolve()),
            "k128_run_dir": str(Path(args.k128_run_dir).resolve()),
            "arff": str(Path(args.arff).resolve()),
        },
    }


def _classify_reentry(run_dir: str | Path, args: argparse.Namespace | None = None) -> str:
    """Classify a production re-entry without mutating the run tree.

    A sealed terminal tree is verification-only.  An unsealed tree with the frozen
    start authority is a whole-run restart; it is never a partial-row resume.  A
    population-sealed tree may continue only with scoring/review terminalization.
    """

    root = Path(run_dir)
    _require(root.is_dir(), "re-entry run directory is absent")
    status_path = root / "status.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    state = str(status.get("state", ""))
    if state in {"resource_stopped", "integrity_failed"}:
        return "verify_only"
    if state in {"awaiting_human_review", "complete"}:
        try:
            verify_run(root)
            return "verify_only"
        except Exception as terminal_error:
            ledger_path = root / "resource_ledger.json"
            _require(ledger_path.is_file(), f"terminal tree is invalid: {terminal_error}")
            ledger = _read_json(ledger_path)
            _require(ledger.get("failed_admission") is None, f"terminal tree is invalid: {terminal_error}")
            open_kinds: set[str] = set()
            admitted_counts: Counter[str] = Counter()
            closure_counts: Counter[str] = Counter()
            for event in ledger.get("events", []):
                kind = str(event.get("kind", ""))
                if event.get("event") == "admit":
                    _require(kind not in open_kinds, "terminal resource event was admitted twice")
                    open_kinds.add(kind)
                    admitted_counts[kind] += 1
                elif event.get("event") in {"complete", "failed-complete", "interrupted-close"}:
                    open_kinds.discard(kind)
                    closure_counts[kind] += 1
            terminal_kind = (
                "machine_terminalization"
                if state == "awaiting_human_review"
                else "human_review_terminalization"
            )
            terminal_history = [
                event
                for event in ledger.get("events", [])
                if event.get("kind") == terminal_kind
                and event.get("event")
                in {"admit", "complete", "failed-complete", "interrupted-close"}
            ]
            if terminal_kind == "machine_terminalization":
                history_valid = (
                    len(terminal_history) in {1, 2}
                    and terminal_history[0].get("event") == "admit"
                    and (
                        len(terminal_history) == 1
                        or terminal_history[1].get("event")
                        in {"complete", "interrupted-close"}
                    )
                    and open_kinds
                    == ({terminal_kind} if len(terminal_history) == 1 else set())
                )
            else:
                # Review may have several crash-retry attempts.  Every earlier
                # attempt must close conservatively as interrupted; the final
                # attempt is either still open, interrupted before the terminal
                # reseal, or completed before that reseal.
                pair_count = len(terminal_history) // 2
                history_valid = (
                    bool(terminal_history)
                    and all(
                        terminal_history[2 * index].get("event") == "admit"
                        and terminal_history[2 * index + 1].get("event")
                        == "interrupted-close"
                        for index in range(max(0, pair_count - (0 if len(terminal_history) % 2 else 1)))
                    )
                )
                if history_valid and len(terminal_history) % 2 == 1:
                    history_valid = (
                        terminal_history[-1].get("event") == "admit"
                        and open_kinds == {terminal_kind}
                    )
                elif history_valid:
                    history_valid = (
                        len(terminal_history) >= 2
                        and terminal_history[-2].get("event") == "admit"
                        and terminal_history[-1].get("event")
                        in {"interrupted-close", "complete"}
                        and not open_kinds
                    )
            _require(
                history_valid
                and (root / "populations" / "POPULATIONS_SEALED.json").is_file(),
                f"terminal tree is invalid and is not a recoverable terminalization crash: {terminal_error}",
            )
            _verify_terminal_recovery_evidence(root, state=state, resource_ledger=ledger)
            return "continue_sealed"
    if (root / "populations" / "POPULATIONS_SEALED.json").is_file():
        return "continue_sealed"
    _require(
        (root / "inventory" / "START_BANK_SEALED.json").is_file(),
        "existing run has no sealed start authority; preserve it and use a fresh directory",
    )
    return "rerun_all_rows"


def _verify_terminal_recovery_evidence(
    run_dir: Path,
    *,
    state: str,
    resource_ledger: Mapping[str, Any],
) -> None:
    """Authenticate all scientific bytes before a stale terminal tree is mutated."""

    _require(state in {"awaiting_human_review", "complete"}, "invalid terminal recovery state")
    population_seal = _verify_population_seal(run_dir)
    for row in ("teacher", "null", "learned"):
        _verify_one_row_result(run_dir, row, partial=False)
    _verify_telemetry_summary(run_dir)
    evaluation = _verify_evaluation(run_dir, population_seal)
    review = _verify_review_bundle(run_dir, population_seal)
    gates = _verify_machine_gates(
        run_dir,
        resource_ledger,
        population_seal,
        complete=state == "complete",
    )
    if state == "complete":
        status = _read_json(run_dir / "status.json")
        human = _replay_human_review(run_dir, review)
        _verify_complete_outcome(run_dir, status, gates, human, evaluation)


def _assert_reentry_args(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    budget: ResourceBudget,
) -> dict[str, Any]:
    """Require the re-entered command to match the frozen execution authority."""

    config = _read_json(run_dir / "config.json")
    inputs = config.get("input_paths", {})
    expected_paths = {
        "legacy_checkpoint": Path(args.legacy_checkpoint).resolve(),
        "ddpm_run_dir": Path(args.ddpm_run_dir).resolve(),
        "k128_run_dir": Path(args.k128_run_dir).resolve(),
        "arff": Path(args.arff).resolve(),
    }
    for name, supplied in expected_paths.items():
        _require(
            name in inputs and Path(inputs[name]).resolve() == supplied,
            f"re-entry input changed: {name}",
        )
    execution = config.get("execution_authority", {})
    _require(str(execution.get("device")) == str(args.device) == "cuda:0", "re-entry device changed")
    _require(str(execution.get("approval_id")) == str(args.approval_id), "re-entry approval changed")
    _require(
        float(execution.get("max_active_seconds")) == budget.max_active_seconds
        and int(execution.get("max_storage_bytes")) == budget.max_storage_bytes
        and float(execution.get("max_cuda_fraction")) == budget.max_cuda_fraction
        and float(execution.get("reserve_seconds")) == budget.reserve_seconds
        and float(execution.get("maximum_quantum_seconds")) == budget.maximum_quantum_seconds,
        "re-entry resource authority changed",
    )
    return config


def _record_restart_authority(run_dir: Path, *, mode: str) -> dict[str, Any]:
    """Record a restart while preserving and rebinding the frozen start authority."""

    _require(mode in {"rerun_all_rows", "continue_sealed"}, "invalid restart mode")
    config_path = run_dir / "config.json"
    start_seal_path = run_dir / "inventory" / "START_BANK_SEALED.json"
    config = _read_json(config_path)
    start_seal = _read_json(start_seal_path)
    old_config_sha256 = sha256_file(config_path)
    old_start_seal_sha256 = sha256_file(start_seal_path)
    history_path = run_dir / "restart_history.json"
    if history_path.is_file():
        history = _read_json(history_path)
        _require(
            set(history) == {"schema", "events"}
            and history["schema"] == VERSION + "-restart-history"
            and isinstance(history["events"], list),
            "restart history changed",
        )
        events = list(history["events"])
    else:
        events = []
    count = len(events) + 1
    event = {
        "restart_index": count,
        "mode": mode,
        "recorded_at": _utc_now(),
        "old_config_sha256": old_config_sha256,
        "old_start_bank_seal_sha256": old_start_seal_sha256,
        "population_sealed_before_restart": int(
            (run_dir / "populations" / "POPULATIONS_SEALED.json").is_file()
        ),
        "partial_resume_used": 0,
    }
    # The start seal is immutable.  Restart count is an append-only operational
    # authority outside the scientific start closure.
    event["new_config_sha256"] = old_config_sha256
    event["new_start_bank_seal_sha256"] = old_start_seal_sha256
    events.append(event)
    _write_json(
        history_path,
        {"schema": VERSION + "-restart-history", "events": events},
    )
    return event


def _clear_directory_files(directory: Path) -> None:
    if not directory.is_dir():
        return
    root = directory.resolve()
    paths = sorted(directory.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        resolved = path.resolve()
        _require(resolved == root or root in resolved.parents, "restart cleanup escaped its run directory")
        if path.is_symlink():
            raise IntegrityFailure(f"restart cleanup refuses linked artifact: {path}")
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def _clear_unsealed_outputs(run_dir: str | Path) -> None:
    """Discard an incomplete attempt, never any sealed population or input."""

    root = Path(run_dir)
    _require(not (root / "populations" / "POPULATIONS_SEALED.json").exists(), "sealed populations cannot be cleared")
    for relative in ("preflight", "populations", "telemetry", "images", "controls", "evaluation", "review"):
        _clear_directory_files(root / relative)
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "data" / "test_open_event.json").unlink(missing_ok=True)
    for name in (
        "gates.json",
        "outcome.json",
        "failure.json",
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "REPORT.md",
        "HANDOFF.md",
    ):
        (root / name).unlink(missing_ok=True)
    events = _stage_events(root)
    preserved = [
        event
        for event in events
        if event.get("state") == "completed"
        and event.get("stage") in {"initialize_and_bind", "checkpoint_extract", "data_and_inventory"}
    ]
    stages = [event["stage"] for event in preserved]
    _require(
        stages in (
            ["initialize_and_bind", "checkpoint_extract"],
            ["initialize_and_bind", "checkpoint_extract", "data_and_inventory"],
        ),
        "restart requires the exact completed authority-stage prefix",
    )
    if stages == ["initialize_and_bind", "checkpoint_extract"]:
        preserved.append(
            {"stage": "data_and_inventory", "state": "completed", "recorded_at": _utc_now()}
        )
    _write_json(root / "stage_ledger.json", {"schema": VERSION + "-stage-ledger", "events": preserved})


def _clear_postseal_outputs(run_dir: Path) -> None:
    """Clear an interrupted scoring attempt without touching sealed populations."""

    _require((run_dir / "populations" / "POPULATIONS_SEALED.json").is_file(), "population seal is absent")
    for relative in ("controls", "evaluation", "review"):
        _clear_directory_files(run_dir / relative)
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / "test_open_event.json").unlink(missing_ok=True)
    for name in (
        "gates.json",
        "outcome.json",
        "failure.json",
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "REPORT.md",
        "HANDOFF.md",
    ):
        (run_dir / name).unlink(missing_ok=True)
    events = _stage_events(run_dir)
    expected = list(STAGE_ORDER[: STAGE_ORDER.index("population_seal") + 1])
    preserved = [event for event in events if event.get("stage") in expected]
    stages = [str(event.get("stage")) for event in preserved]
    if stages == expected[:-1]:
        # A crash after the seal write but before the stage receipt is recoverable
        # because the seal was semantically verified before this helper is called.
        preserved.append(
            {"stage": "population_seal", "state": "completed", "recorded_at": _utc_now()}
        )
        stages.append("population_seal")
    _require(stages == expected, "sealed continuation stage prefix changed")
    _write_json(
        run_dir / "stage_ledger.json",
        {"schema": VERSION + "-stage-ledger", "events": preserved},
    )


def _load_frozen_inventory(run_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    inventory = build_path_inventory()
    with np.load(run_dir / "inventory" / "start_bank.npz", allow_pickle=False) as archive:
        starts = archive["starts"].astype(np.float32, copy=True)
    with np.load(run_dir / "inventory" / "teacher_target_bank.npz", allow_pickle=False) as archive:
        targets = {key: archive[key].copy() for key in archive.files}
    return inventory, starts, targets


def _validate_reentry_base(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    budget: ResourceBudget,
    device: torch.device,
) -> tuple[
    ResourceGovernor,
    dict[str, Any],
    DirectFluxMNISTConfig,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Replay every immutable pre-generation authority before any restart write."""

    config = _assert_reentry_args(run_dir, args, budget=budget)
    ledger = _read_json(run_dir / "resource_ledger.json")
    config = _verify_config_and_sources(run_dir, ledger)
    _verify_predecessor_bindings(run_dir, config)
    scientific_config = _verify_checkpoint_extract(run_dir, config)
    development_images, development_labels = _verify_data_and_inventory(
        run_dir,
        config,
        scientific_config,
    )
    inventory, starts, targets = _load_frozen_inventory(run_dir)
    governor = ResourceGovernor.rehydrate(
        run_dir,
        device=device,
        recover_interrupted=True,
    )
    return (
        governor,
        config,
        scientific_config,
        development_images,
        development_labels,
        inventory,
        starts,
        targets,
    )


def _reset_running_status(run_dir: Path, *, whole_run_restart_required: int = 0) -> None:
    _write_json(
        run_dir / "status.json",
        {
            "schema": VERSION + "-status",
            "state": "running",
            "route": "running",
            "error": None,
            "updated_at": _utc_now(),
            "whole_run_restart_required": int(whole_run_restart_required),
        },
    )


def _machine_gates(
    run_dir: Path,
    *,
    governor: ResourceGovernor,
    teacher_control: Mapping[str, Any],
) -> dict[str, Any]:
    deterministic = _read_json(run_dir / "preflight" / "deterministic_replay.json")
    projection = _read_json(run_dir / "preflight" / "resource_projection.json")
    population_seal = _verify_population_seal(run_dir)
    gate_a_conditions = {
        "checkpoint_and_clean_state_bound": (run_dir / "input_bindings" / "clean_model_state_receipt.json").is_file(),
        "source_bindings_present": (run_dir / "source_bindings.json").is_file(),
        "data_roles_strict_prefix": int(_read_json(run_dir / "data" / "development_roles.json")["terminal_content_rows_parsed"]) == 0,
        "start_bank_sealed": (run_dir / "inventory" / "START_BANK_SEALED.json").is_file(),
        "predecessor_route_bound": _read_json(run_dir / "input_bindings" / "predecessors.json")["k128"]["required_route"] == K128_REQUIRED_ROUTE,
    }
    gate_b_conditions = {
        "sealed_start_count_160": int(_read_json(run_dir / "inventory" / "START_BANK_SEALED.json")["path_count"]) == PATH_COUNT,
        "three_rows_160_retained": all(int(population_seal["rows"][name]["endpoint_count"]) == PATH_COUNT for name in ("teacher", "null", "learned")),
        "factor_one_no_selector": population_seal["generated_candidates_per_path"] == 1 and population_seal["selector"] is None,
        "target_firewall": "targets" not in inspect.signature(run_null_row).parameters and "targets" not in inspect.signature(run_learned_row).parameters,
        "terminal_and_evaluator_opened_only_after_population_seal": (run_dir / "data" / "test_open_event.json").is_file() and (run_dir / "evaluation" / "EVALUATOR_OPEN_EVENT.json").is_file(),
    }
    peak_fraction = max([float(event.get("cuda_fraction", 0.0)) for event in governor.events] + [0.0])
    gate_c_conditions = {
        "deterministic_replay": int(deterministic["passed"]) == 1,
        "resource_projection": int(projection["passed"]) == 1,
        "population_semantic_seal": True,
        "no_failed_resource_admission": governor.failed_admission is None,
        "active_cap": governor.active_seconds <= governor.budget.max_active_seconds,
        "storage_cap": _storage_bytes(run_dir) <= governor.budget.max_storage_bytes,
        "cuda_cap": peak_fraction <= governor.budget.max_cuda_fraction,
    }
    gates = {
        "schema": VERSION + "-gates",
        "gate_a": {
            "gate_type": "execution/integrity",
            "passed": int(all(gate_a_conditions.values())),
            "conditions": {key: int(value) for key, value in gate_a_conditions.items()},
        },
        "gate_b": {
            "gate_type": "execution/integrity",
            "passed": int(all(gate_b_conditions.values())),
            "conditions": {key: int(value) for key, value in gate_b_conditions.items()},
        },
        "gate_c": {
            "gate_type": "execution/integrity",
            "passed": int(all(gate_c_conditions.values())),
            "conditions": {key: int(value) for key, value in gate_c_conditions.items()},
        },
        "gate_d": {
            "gate_type": "execution/integrity",
            "passed": int(teacher_control["passed"]),
            "conditions": dict(teacher_control["conditions"]),
            "teacher_control_sha256": sha256_file(run_dir / "controls" / "teacher_gate.json"),
        },
        "gate_e": {
            "gate_type": "diagnostic threshold",
            "state": "pending",
            "passed": None,
            "conditions": {},
        },
    }
    _write_json(run_dir / "gates.json", gates)
    return gates


def _finalize_failure(
    run_dir: Path,
    error: BaseException,
    *,
    governor: ResourceGovernor,
    failed_stage: str,
    partial_row: str | None = None,
) -> dict[str, Any]:
    governor.close_open_as_failed()
    original_failed_admission = governor.failed_admission
    terminal_admitted = False
    try:
        governor.admit(
            "failure_terminalization",
            predicted_seconds=5.0,
            predicted_next_bytes=1 * 1024 * 1024,
            reserve_remaining_seconds=0.0,
        )
        terminal_admitted = True
    except ResourceStop:
        pass
    state = "resource_stopped" if isinstance(error, ResourceStop) else "integrity_failed"
    failure = {
        "schema": VERSION + "-failure",
        "state": state,
        "route": state,
        "failed_stage": failed_stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "recorded_at": _utc_now(),
        "partial_row": partial_row,
        "original_failed_admission": original_failed_admission,
        "scientific_result_available": 0,
        "whole_run_restart_required": int(not (run_dir / "populations" / "POPULATIONS_SEALED.json").is_file()),
    }
    _write_json(run_dir / "failure.json", failure)
    _write_json(
        run_dir / "status.json",
        {
            "schema": VERSION + "-status",
            "state": state,
            "route": state,
            "error": str(error),
            "updated_at": _utc_now(),
            "whole_run_restart_required": failure["whole_run_restart_required"],
        },
    )
    _write_reports(run_dir, None)
    terminal_error: ResourceStop | None = None
    if terminal_admitted:
        try:
            governor.complete("failure_terminalization")
        except ResourceStop as completion_error:
            terminal_error = completion_error
    if terminal_error is not None and state != "resource_stopped":
        failure["original_error_type"] = failure["error_type"]
        failure["original_message"] = failure["message"]
        failure["state"] = "resource_stopped"
        failure["route"] = "resource_stopped"
        failure["error_type"] = "ResourceStop"
        failure["message"] = str(terminal_error)
        state = "resource_stopped"
        _write_json(run_dir / "failure.json", failure)
        _write_json(
            run_dir / "status.json",
            {
                "schema": VERSION + "-status",
                "state": state,
                "route": state,
                "error": str(terminal_error),
                "updated_at": _utc_now(),
                "whole_run_restart_required": failure["whole_run_restart_required"],
            },
        )
    _write_reports(run_dir, None)
    manifest = _seal_manifest(run_dir)
    receipt = verify_run(run_dir)
    _require(int(receipt.get("passed", 0)) == 1, "terminal failure tree did not pass semantic verification")
    return {**failure, "artifact_count": manifest["artifact_count"], "tree_digest": manifest["tree_digest"], "verification": receipt}


def _write_reports(run_dir: Path, outcome: Mapping[str, Any] | None) -> None:
    status = _read_json(run_dir / "status.json")
    config = _read_json(run_dir / "config.json")
    gates = _read_json(run_dir / "gates.json") if (run_dir / "gates.json").is_file() else None
    resource = _read_json(run_dir / "resource_ledger.json") if (run_dir / "resource_ledger.json").is_file() else None
    teacher_control = _read_json(run_dir / "controls" / "teacher_gate.json") if (run_dir / "controls" / "teacher_gate.json").is_file() else None
    machine_metrics = {
        name: _read_json(run_dir / "evaluation" / f"{name}_metrics.json")
        for name in ("teacher", "null", "learned")
        if (run_dir / "evaluation" / f"{name}_metrics.json").is_file()
    }
    execution = config.get("execution_authority", {})
    run_command = str(config.get("command", ""))
    review_command = (
        f'{sys.executable} -B -m mnist.diag_d0_eulerian_edge_flux_replay record-review '
        f'--run-dir "{run_dir}" --answers <completed-review-csv> --reviewer <reviewer-id> --confirm-manual-review'
    )
    verify_command = (
        f'{sys.executable} -B -m mnist.diag_d0_eulerian_edge_flux_replay verify --run-dir "{run_dir}"'
    )
    lines = [
        "# Eulerian edge-flux factor-one replay",
        "",
        f"- Research mode: `{RESEARCH_MODE}`",
        "- Decision: fresh global edge-flux checkpoint compatibility without candidate selection or replacement.",
        f"- State: `{status.get('state')}`",
        f"- Route: `{status.get('route')}`",
        "- Proxy-only patches since the last objective-bearing experiment: 0",
        "- Factor one is a retention policy; adaptive numerical attempts may retry 1/2/4 substeps.",
        "- Historical checkpoint factor=4/composite metadata was preserved but never used for this replay.",
        "",
        "## Authorities and controls",
        "",
        f"- Source revision: `{config.get('git_revision', '')}` with exact source hashes in `source_bindings.json`.",
        f"- Production command: `{run_command}`",
        f"- Review command: `{review_command}`",
        f"- Verification command: `{verify_command}`",
        f"- Approval: `{execution.get('approval_id', '')}`; device `{execution.get('device', '')}`.",
        f"- K128 predecessor tree: `{K128_TREE_DIGEST}`",
        "- Rows: target-informed teacher, null, learned; 160 retained paths each.",
        "- Anchors: 0, 64, 128, 192, 256.",
        "- Raster: training rows [0,55000), exact 25471/255 scale.",
    ]
    if gates is not None:
        lines.extend(["", "## Typed gates", ""])
        for name in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e"):
            if name in gates:
                label = "Gate " + name.removeprefix("gate_").upper()
                lines.append(
                    f"- {label}: state `{gates[name].get('state', 'complete')}`, "
                    f"passed `{gates[name].get('passed')}` ({gates[name].get('gate_type')})."
                )
                for key, value in gates[name].get("values", {}).items():
                    lines.append(f"  - {key}: `{value}`")
    if teacher_control is not None and {
        "median_relative_squared_l2_anchor64",
        "median_relative_squared_l2_endpoint",
        "endpoint_improved_path_count",
        "teacher_requested_label_accuracy",
    }.issubset(teacher_control):
        lines.extend(
            [
                "",
                "## Known-positive teacher",
                "",
                f"- Median relative squared L2 at anchor 64: `{teacher_control['median_relative_squared_l2_anchor64']}`.",
                f"- Median relative squared L2 at endpoint: `{teacher_control['median_relative_squared_l2_endpoint']}`.",
                f"- Improved paths: `{teacher_control['endpoint_improved_path_count']}/{PATH_COUNT}`.",
                f"- Teacher requested-label accuracy: `{teacher_control['teacher_requested_label_accuracy']}`.",
                f"- median_relative_squared_l2_anchor64: `{teacher_control['median_relative_squared_l2_anchor64']}`",
                f"- median_relative_squared_l2_endpoint: `{teacher_control['median_relative_squared_l2_endpoint']}`",
                f"- endpoint_improved_path_count: `{teacher_control['endpoint_improved_path_count']}`",
                f"- teacher_requested_label_accuracy: `{teacher_control['teacher_requested_label_accuracy']}`",
                "- Exact path arrays: `controls/teacher_gate_arrays.npz`; gate receipt: `controls/teacher_gate.json`.",
            ]
        )
    if machine_metrics:
        lines.extend(["", "## Machine objective metrics", ""])
        for name in ("teacher", "null", "learned"):
            if name not in machine_metrics:
                continue
            metric = machine_metrics[name]
            lines.append(
                f"- {name}: requested-label accuracy `{metric['classifier']['requested_label_accuracy']}`, "
                f"duplicate pairs `{metric['duplicates']['duplicate_pair_count']}`, "
                f"diversity ratio `{metric['diversity']['aggregate_median_ratio']}`."
            )
        lines.append("- Per-class values are in `evaluation/{teacher,null,learned}_metrics.json`; raw logits and predictions are in `evaluation/predictions.npz`.")
    if outcome is not None:
        lines.extend(
            [
                "",
                "## Objective result",
                "",
                f"- Human learned recognizability: `{outcome['human_marker']['learned']['human_recognizability']}`",
                f"- Human null recognizability: `{outcome['human_marker']['null']['human_recognizability']}`",
                f"- Human learned requested-label agreement: `{outcome['human_marker']['learned']['human_requested_label_agreement']}`",
                f"- Human null requested-label agreement: `{outcome['human_marker']['null']['human_requested_label_agreement']}`",
                f"- Human agreement learned-minus-null: `{outcome['human_marker']['learned']['human_requested_label_agreement'] - outcome['human_marker']['null']['human_requested_label_agreement']}`",
                f"- Classifier learned/null accuracy: `{outcome['classifier_marker']['learned_accuracy']}` / `{outcome['classifier_marker']['null_accuracy']}`",
                f"- Learned duplicate pairs / diversity ratio: `{outcome['noncollapse_marker']['components']['zero_duplicate_pairs']}` pass / `{machine_metrics['learned']['diversity']['aggregate_median_ratio']}`",
                "- Exact per-class human values: `review/human_review_by_row.json`; submitted answers: `review/human_review_answers.csv`.",
                f"- This result establishes: {_route_scoped_claim(str(outcome['route']))}",
                "- This result does not establish exact h-transform correctness, reference-prior matching, confirmatory population quality, or general Eulerian feasibility/failure.",
                f"- Next action: {outcome['next_action']}",
            ]
        )
    elif status.get("state") == "awaiting_human_review":
        lines.extend(
            [
                "",
                "## Pending human decision",
                "",
                "- Machine scoring and the sealed 80-image blind bundle are complete.",
                "- No outcome exists until the prespecified manual review is recorded.",
                "- Review only `review/blinded-contact-sheet.png` or the individual `review/samples/` before opening the private key or machine metrics.",
            ]
        )
    if resource is not None:
        budget = resource.get("budget", {})
        events = resource.get("events", [])
        peak_cuda = max([float(event.get("cuda_fraction", 0.0)) for event in events] + [0.0])
        lines.extend(
            [
                "",
                "## Resources and health",
                "",
                f"- Charged active seconds: `{resource.get('active_seconds')}` / `{budget.get('max_active_seconds')}`.",
                f"- Persisted bytes at report time: `{_storage_bytes(run_dir)}` / `{budget.get('max_storage_bytes')}`.",
                f"- Peak recorded CUDA fraction: `{peak_cuda}` / `{budget.get('max_cuda_fraction')}`.",
                f"- Failed admission: `{resource.get('failed_admission')}`.",
            ]
        )
    if (run_dir / "failure.json").is_file():
        failure = _read_json(run_dir / "failure.json")
        lines.extend(
            [
                "",
                "## Saved failure",
                "",
                f"- Failed stage: `{failure.get('failed_stage')}`; type `{failure.get('error_type')}`.",
                f"- Message: {failure.get('message')}",
                f"- Partial row: `{failure.get('partial_row')}`; retained evidence is listed in `artifact_manifest.json`.",
                "- Observed input existence, size, hash, and predecessor checks: `input_bindings/input_authority_observations.json`.",
                "- The terminal tree is returned only after the read-only semantic verifier passes.",
            ]
        )
    lines.extend(
        [
            "",
            "## Evidence map",
            "",
            "- `input_bindings/input_authority_observations.json`: observed file/predecessor authority, including failed early propositions.",
            "- `controls/teacher_gate.json` and `controls/teacher_gate_arrays.npz`: full-interface target-informed control.",
            "- `evaluation/learned_metrics.json` and `evaluation/predictions.npz`: machine objective values and raw predictions.",
            "- `review/human_review_by_row.json` and `review/human_review_answers.csv`: blinded human result after review.",
            "- `populations/POPULATIONS_SEALED.json`: immutable three-row population authority.",
            "- `gates.json` and `outcome.json`: typed thresholds and final action route.",
            "",
            "## Claim boundary",
            "",
            "This exploratory replay concerns one hash-pinned historical checkpoint under the current fixed-grid sampler and historical low-frequency source law. It is not an exact h-transform, reference-prior, confirmatory, or DDPM-superiority claim.",
            "",
            "No larger population, automatic DSM, training, tuning, candidate ranking, or next experiment was launched.",
            "",
            "## Deliberate omissions",
            "",
            "- No checkpoint training or reselection, candidate ranking, replacement, fourth production row, exact h-transform/reference-prior audit, confidence interval, or confirmatory test.",
            "- Timing and allocator observations are excluded from scientific replay hashes but retained in resource and step telemetry.",
        ]
    )
    _atomic_bytes(run_dir / "REPORT.md", ("\n".join(lines) + "\n").encode("utf-8"))
    route = str(status.get("route"))
    result_claim = _route_scoped_claim(route if outcome is not None else None)
    handoff_sections = [
        "# Eulerian edge-flux factor-one replay: research handoff",
        "",
        "## 1. Program objective",
        "Establish or decisively falsify a DDPM-like MNIST generator based on an Eulerian approximation. The concrete artifact here is a retained 160-image learned population with null and target-informed full-interface controls.",
        "",
        "## 2. Current milestone and distance to goal",
        f"This objective-bearing replay is `{status.get('state')}` on route `{route}`. Proxy-only patches since the last objective-bearing experiment: 0.",
        "",
        "## 3. Strategy review",
        f"Strategy status follows the prespecified route. Next action: {outcome['next_action'] if outcome is not None else 'finish the blinded review; do not launch another experiment automatically.'}",
        "",
        "## 4. Research mode and evidence roles",
        "Primary mode: exploratory. Development rows are 0:55000; exploratory validation rows are 55000:60000; terminal rows 60000:70000 open only after population sealing for evaluator context.",
        "",
        "## 5. Exact result of the latest run",
        f"{result_claim}",
        "",
        "## 6. Confirmed facts, current inferences, and open hypotheses",
        "Confirmed facts are the sealed controls, populations, machine metrics, and (after completion) blind-review values below. Interpretations remain scoped to one checkpoint/source/sampler/transform tuple; architecture, controller, source-law mismatch, evaluator mismatch, and strategy failure remain revisable hypotheses.",
        "",
        "## 7. Decision the next patch must resolve",
        "If complete, execute the exact outcome action rather than another edge-flux calibration; if awaiting review, resolve only the fixed human marker.",
        "",
        "## 8. Candidate actions and value of information",
        "A positive result supports freezing this exploratory mechanism; a healthy negative stops this checkpoint line; an invalid control repairs only the localized defect. No branch auto-launches compute.",
        "",
        "## 9. Recommended next patch",
        f"{outcome['next_action'] if outcome is not None else 'No code patch yet: complete the already-prepared blinded review.'}",
        "",
        "## 10. Gates and claim boundaries",
        "Gate A-D are execution/integrity gates. Gate E is a diagnostic threshold. Their exact states and raw values appear in the evidence detail below.",
        "",
        "## 11. Outcome-to-action table",
        "| Outcome | Interpretation | Required next action |\n|---|---|---|\n| factor_one_feasible | scoped exploratory feasibility | freeze; separately approve any replication |\n| human_positive_evaluator_disagreement | evaluator/render disagreement | preserve human result; audit symmetrically |\n| factor_one_negative_stop_checkpoint_line | this checkpoint line lacks feasibility | stop it; separately plan a material alternative or stop |\n| invalid_repair_same_experiment | invalid system evidence | repair only the localized defect |",
        "",
        "## 12. Constraints",
        "Preserve sealed starts, all failed-looking outputs, source/checkpoint/evaluator bindings, data firewalls, and factor-one retention. Architecture, training, controller, source law, and strategy remain revisable after this result.",
        "",
        "## 13. Resource budget and stop rule",
        f"Bound authority: `{execution}`. Exact receipts are in `resource_ledger.json`; no silent cap increase or automatic continuation is allowed.",
        "",
        "## 14. Alternative and pivot plan",
        "A healthy negative stops this historical checkpoint line. Any fixed-grid/on-policy or prior-matched alternative requires a separate plan and approval.",
        "",
        "## 15. Evidence map",
        "See the exact paths in the evidence detail below, especially `populations/POPULATIONS_SEALED.json`, `controls/teacher_gate.json`, `evaluation/predictions.npz`, `review/human_review_by_row.json`, `gates.json`, and `outcome.json`.",
        "",
        "## 16. Deliberate omissions",
        "No retraining, checkpoint reselection, candidate ranking, reference-prior proof, confidence interval, exact h-transform audit, or confirmatory population claim was attempted.",
        "",
        "## 17. Reproduction commands",
        f"Production: `{run_command}`\n\nReview: `{review_command}`\n\nVerify: `{verify_command}`",
        "",
        "## 18. Bundle-integrity audit",
        "Use `artifact_manifest.json` and `SHA256SUMS.txt`; `verify` replays semantic authorities read-only and reports the tree digest.",
        "",
        "## 19. Exact deliverable for the receiving agent",
        f"Honor route `{route}` and its one next action; do not tune, select samples, or auto-launch a larger or DSM experiment.",
        "",
        "## Evidence detail",
        "",
        *lines[2:],
    ]
    handoff = "\n".join(handoff_sections) + "\n"
    _atomic_bytes(run_dir / "HANDOFF.md", handoff.encode("utf-8"))


def _verify_manifest_read_only(run_dir: Path) -> dict[str, Any]:
    _require(run_dir.is_dir(), "run directory is absent")
    _require(all(not path.is_symlink() for path in run_dir.rglob("*")), "linked run artifacts are forbidden")
    manifest_path = run_dir / "artifact_manifest.json"
    sums_path = run_dir / "SHA256SUMS.txt"
    manifest = _read_json(manifest_path)
    _require(
        set(manifest) == {"schema", "artifact_count", "artifact_bytes", "tree_digest", "files"},
        "artifact manifest schema changed",
    )
    _require(manifest["schema"] == VERSION + "-artifact-manifest", "artifact manifest version changed")
    files = manifest["files"]
    _require(type(files) is list, "artifact manifest file inventory is invalid")
    actual = _manifest_rows(run_dir)
    _require(files == actual, "artifact inventory, byte size, or SHA-256 changed")
    _require(int(manifest["artifact_count"]) == len(actual), "artifact manifest count changed")
    _require(int(manifest["artifact_bytes"]) == sum(int(row["bytes"]) for row in actual), "artifact manifest byte total changed")
    _require(manifest["tree_digest"] == _tree_digest(actual), "artifact tree digest changed")
    expected_sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in actual)
    expected_sums += f"{sha256_file(manifest_path)}  artifact_manifest.json\n"
    _require(sums_path.read_text(encoding="utf-8") == expected_sums, "SHA256SUMS changed")
    return manifest


def _verify_stage_and_status(run_dir: Path) -> tuple[dict[str, Any], list[str], str | None]:
    status = _read_json(run_dir / "status.json")
    state = status.get("state")
    _require(
        state in {"awaiting_human_review", "complete", "resource_stopped", "integrity_failed"},
        "terminal status state is invalid",
    )
    ledger = _read_json(run_dir / "stage_ledger.json")
    _require(set(ledger) == {"schema", "events"}, "stage ledger schema changed")
    _require(ledger["schema"] == VERSION + "-stage-ledger", "stage ledger version changed")
    events = ledger["events"]
    _require(type(events) is list, "stage ledger events are invalid")
    completed: list[str] = []
    for index, event in enumerate(events):
        _require(
            type(event) is dict
            and set(event) == {"stage", "state", "recorded_at"}
            and event["state"] == "completed"
            and event["stage"] in STAGE_ORDER
            and isinstance(event["recorded_at"], str)
            and bool(event["recorded_at"].strip()),
            f"stage ledger event {index} is invalid",
        )
        completed.append(str(event["stage"]))
    operational = list(STAGE_ORDER[: STAGE_ORDER.index("machine_terminalization")])
    if "machine_terminalization" in completed:
        terminal_index = completed.index("machine_terminalization")
        _require(
            completed[:terminal_index] == operational[:terminal_index],
            "completed work stages are not an exact ordered prefix",
        )
        _require(
            completed[terminal_index:]
            in (["machine_terminalization"], ["machine_terminalization", "human_review_terminalization"]),
            "terminal stage order changed",
        )
    else:
        _require(completed == operational[: len(completed)], "completed work stages are not an exact ordered prefix")

    failure_stage: str | None = None
    if state == "awaiting_human_review":
        _require(completed == list(STAGE_ORDER[:-1]), "awaiting-human-review stage terminal is incoherent")
        _require(set(status) == {"schema", "state", "route", "error", "updated_at", "whole_run_restart_required"}, "awaiting-human-review status schema changed")
        _require(status["schema"] == VERSION + "-status", "awaiting-human-review status version changed")
        _require(status["error"] is None and status["whole_run_restart_required"] == 0, "awaiting-human-review status health changed")
        _require(status.get("route") == "awaiting_human_review", "awaiting-human-review status route changed")
        _require(not (run_dir / "failure.json").exists(), "awaiting-human-review run has failure authority")
    elif state == "complete":
        _require(completed == list(STAGE_ORDER), "complete stage terminal is incoherent")
        _require(set(status) == {"schema", "state", "route", "error", "updated_at", "whole_run_restart_required"}, "complete status schema changed")
        _require(status["schema"] == VERSION + "-status", "complete status version changed")
        _require(status["error"] is None and status["whole_run_restart_required"] == 0, "complete status health changed")
        _require(not (run_dir / "failure.json").exists(), "complete run has failure authority")
    else:
        failure = _read_json(run_dir / "failure.json")
        base_failure_fields = {
                "schema",
                "state",
                "route",
                "failed_stage",
                "error_type",
                "message",
                "recorded_at",
                "partial_row",
                "original_failed_admission",
                "scientific_result_available",
                "whole_run_restart_required",
            }
        terminal_conversion_fields = base_failure_fields | {
            "original_error_type",
            "original_message",
        }
        _require(
            frozenset(failure) in {frozenset(base_failure_fields), frozenset(terminal_conversion_fields)},
            "failure authority schema changed",
        )
        _require(failure["schema"] == VERSION + "-failure", "failure authority version changed")
        _require(
            set(status) == {"schema", "state", "route", "error", "updated_at", "whole_run_restart_required"},
            "failure status schema changed",
        )
        _require(status["schema"] == VERSION + "-status", "failure status version changed")
        failure_stage = str(failure.get("failed_stage", ""))
        _require(failure_stage in STAGE_ORDER, "failure stage is invalid")
        completed_work = [stage for stage in completed if stage in operational]
        if "machine_terminalization" in completed:
            last_index = STAGE_ORDER.index("machine_terminalization")
        else:
            last_index = -1 if not completed_work else operational.index(completed_work[-1])
        _require(
            failure_stage == "machine_terminalization"
            or STAGE_ORDER.index(failure_stage) in {last_index, last_index + 1},
            "failure stage is inconsistent with completed stage prefix",
        )
        _require(failure.get("state") == state and failure.get("route") == state, "failure route changed")
        if state == "resource_stopped":
            _require(failure.get("error_type") == "ResourceStop", "resource failure type changed")
        else:
            _require(failure.get("error_type") != "ResourceStop", "integrity failure type changed")
        if set(failure) == terminal_conversion_fields:
            _require(state == "resource_stopped", "terminal resource conversion has the wrong state")
            _require(
                failure["original_error_type"] != "ResourceStop"
                and isinstance(failure["original_error_type"], str)
                and bool(failure["original_error_type"].strip())
                and isinstance(failure["original_message"], str)
                and bool(failure["original_message"].strip()),
                "terminal resource conversion omits the original integrity failure",
            )
            _require(
                failure["original_failed_admission"] is None,
                "terminal resource conversion contains an earlier resource stop",
            )
        _require(isinstance(failure.get("message"), str) and bool(failure["message"].strip()), "failure message is absent")
        _require(isinstance(failure.get("recorded_at"), str) and bool(failure["recorded_at"].strip()), "failure time is absent")
        _require(status.get("error") == failure["message"], "failure status message changed")
        _require(status.get("whole_run_restart_required") == failure["whole_run_restart_required"], "failure restart policy changed")
        _require(failure["scientific_result_available"] == 0, "failure scientific-result boundary changed")
        _require(failure["partial_row"] in {None, "teacher", "null", "learned"}, "failure partial-row identity changed")
        _require(
            int(failure["whole_run_restart_required"])
            == int(not (run_dir / "populations" / "POPULATIONS_SEALED.json").is_file()),
            "failure restart policy does not match the population seal",
        )
        if failure["partial_row"] is not None:
            _require(failure_stage == f"{failure['partial_row']}_row", "failure partial-row and failed-stage identities disagree")
        _require(status.get("route") == state, "failure status route changed")
    return status, completed, failure_stage


def _verify_resource_ledger(run_dir: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    ledger = _read_json(run_dir / "resource_ledger.json")
    _require(
        set(ledger) == {"schema", "budget", "active_seconds", "events", "failed_admission", "open_events"},
        "resource ledger schema changed",
    )
    _require(ledger["schema"] == VERSION + "-resource-ledger", "resource ledger version changed")
    budget = ledger["budget"]
    _require(
        type(budget) is dict
        and set(budget)
        == {
            "max_active_seconds",
            "max_storage_bytes",
            "max_cuda_fraction",
            "reserve_seconds",
            "maximum_quantum_seconds",
        },
        "resource budget schema changed",
    )
    ResourceBudget(
        max_active_seconds=float(budget["max_active_seconds"]),
        max_storage_bytes=int(budget["max_storage_bytes"]),
        max_cuda_fraction=float(budget["max_cuda_fraction"]),
        reserve_seconds=float(budget["reserve_seconds"]),
        maximum_quantum_seconds=float(budget["maximum_quantum_seconds"]),
    )
    events = ledger["events"]
    _require(type(events) is list, "resource events are invalid")
    open_events: dict[str, dict[str, Any]] = {}
    charged = 0.0

    def admission_checks(receipt: Mapping[str, Any]) -> dict[str, bool]:
        return {
            "active": float(receipt["active_seconds_before"])
            + float(receipt["predicted_seconds"])
            + float(receipt["reserve_remaining_seconds"])
            <= float(budget["max_active_seconds"]),
            "storage": int(receipt["storage_bytes_before"])
            + int(receipt["predicted_next_bytes"])
            <= int(budget["max_storage_bytes"]),
            "cuda": float(receipt["cuda_fraction"]) <= float(budget["max_cuda_fraction"]),
            "quantum": float(receipt["predicted_seconds"])
            <= float(budget["maximum_quantum_seconds"])
            or float(receipt["reserve_remaining_seconds"]) == 0.0,
        }

    def verify_cuda_receipt(receipt: Mapping[str, Any], context: str) -> None:
        allocated = int(receipt["cuda_allocated_bytes"])
        total = int(receipt["cuda_total_bytes"])
        fraction = float(receipt["cuda_fraction"])
        _require(
            allocated >= 0 and total >= 0 and math.isfinite(fraction) and fraction >= 0.0,
            f"{context} CUDA receipt is invalid",
        )
        expected = 0.0 if total == 0 else allocated / total
        _require(
            (total > 0 or allocated == 0)
            and (total == 0 or allocated <= total)
            and math.isclose(fraction, expected, rel_tol=0.0, abs_tol=0.0),
            f"{context} CUDA receipt changed",
        )

    for index, event in enumerate(events):
        _require(type(event) is dict, f"resource event {index} is invalid")
        kind = str(event.get("kind", ""))
        _require(bool(kind), f"resource event {index} has no kind")
        event_type = event.get("event")
        if event_type == "admit":
            _require(
                set(event)
                == {
                    "kind",
                    "predicted_seconds",
                    "predicted_next_bytes",
                    "active_seconds_before",
                    "reserve_remaining_seconds",
                    "storage_bytes_before",
                    "cuda_allocated_bytes",
                    "cuda_total_bytes",
                    "cuda_fraction",
                    "checks",
                    "passed",
                    "event",
                    "recorded_at",
                },
                f"resource admission {kind} schema changed",
            )
            _require(kind not in open_events, f"resource event {kind} was admitted twice")
            predicted = float(event.get("predicted_seconds", -1.0))
            predicted_bytes = int(event.get("predicted_next_bytes", -1))
            reserve = float(event.get("reserve_remaining_seconds", -1.0))
            active_before = float(event.get("active_seconds_before", -1.0))
            storage_before = int(event.get("storage_bytes_before", -1))
            cuda_fraction = float(event.get("cuda_fraction", -1.0))
            _require(
                all(math.isfinite(value) for value in (predicted, reserve, active_before, cuda_fraction))
                and predicted > 0.0
                and predicted_bytes > 0
                and reserve >= 0.0
                and storage_before >= 0
                and math.isclose(active_before, charged, rel_tol=0.0, abs_tol=1e-9),
                f"resource admission {kind} inputs changed",
            )
            _require(isinstance(event["recorded_at"], str) and bool(event["recorded_at"].strip()), f"resource admission {kind} time is absent")
            verify_cuda_receipt(event, f"resource admission {kind}")
            expected_reserve = (
                0.0 if "terminalization" in kind else float(budget["reserve_seconds"])
            )
            _require(math.isclose(reserve, expected_reserve, rel_tol=0.0, abs_tol=0.0), f"resource admission {kind} reserve changed")
            expected_checks = admission_checks(event)
            _require(
                event.get("checks") == expected_checks
                and int(event.get("passed", 0)) == int(all(expected_checks.values())),
                f"resource admission {kind} inequality replay changed",
            )
            _require(all(expected_checks.values()), f"failed resource admission {kind} was recorded as an event")
            open_events[kind] = event
        elif event_type in {"complete", "failed-complete"}:
            expected_fields = (
                {
                    "event",
                    "kind",
                    "elapsed_seconds",
                    "active_seconds_after",
                    "storage_bytes_after",
                    "cuda_allocated_bytes",
                    "cuda_total_bytes",
                    "cuda_fraction",
                    "candidate_transitions",
                    "model_evaluations",
                    "recorded_at",
                }
                if event_type == "complete"
                else {"event", "kind", "elapsed_seconds", "active_seconds_after", "storage_bytes_after", "recorded_at"}
            )
            _require(set(event) == expected_fields, f"resource completion {kind} schema changed")
            _require(kind in open_events, f"resource completion {kind} has no admission")
            _require(isinstance(event["recorded_at"], str) and bool(event["recorded_at"].strip()), f"resource completion {kind} time is absent")
            elapsed = float(event.get("elapsed_seconds", -1.0))
            active_after = float(event.get("active_seconds_after", -1.0))
            _require(math.isfinite(elapsed) and elapsed >= 0.0, f"resource completion {kind} elapsed time is invalid")
            charged += elapsed
            _require(math.isclose(active_after, charged, rel_tol=0.0, abs_tol=1e-9), f"resource completion {kind} cumulative time changed")
            if event_type == "complete":
                expected_transitions = 0
                expected_model_evaluations = 0
                if kind == "device_warmup":
                    expected_transitions = PATH_COUNT
                    expected_model_evaluations = PATH_COUNT
                elif kind in {
                    "learned_determinism_probe_1",
                    "learned_determinism_probe_2",
                    "null_timing_probe",
                    "teacher_timing_probe",
                } or "_row_q" in kind:
                    expected_transitions = PATH_COUNT * 8
                    if kind.startswith("learned"):
                        expected_model_evaluations = PATH_COUNT * 8
                _require(
                    int(event.get("candidate_transitions", -1)) == expected_transitions
                    and int(event.get("model_evaluations", -1)) == expected_model_evaluations,
                    f"resource completion {kind} work counts changed",
                )
                for field in ("storage_bytes_after", "cuda_allocated_bytes", "cuda_total_bytes"):
                    _require(int(event.get(field, -1)) >= 0, f"resource completion {kind} {field} is invalid")
                verify_cuda_receipt(event, f"resource completion {kind}")
            open_events.pop(kind)
        elif event_type == "interrupted-close":
            _require(
                set(event) == {"event", "kind", "charged_predicted_seconds", "active_seconds_after", "recorded_at"},
                f"interrupted resource closure {kind} schema changed",
            )
            _require(kind in open_events, f"interrupted resource closure {kind} has no admission")
            _require(isinstance(event["recorded_at"], str) and bool(event["recorded_at"].strip()), f"interrupted resource closure {kind} time is absent")
            charged_predicted = float(event.get("charged_predicted_seconds", -1.0))
            _require(
                math.isfinite(charged_predicted)
                and math.isclose(
                    charged_predicted,
                    float(open_events[kind]["predicted_seconds"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                f"interrupted resource closure {kind} charge changed",
            )
            charged += charged_predicted
            _require(
                math.isclose(float(event.get("active_seconds_after", -1.0)), charged, rel_tol=0.0, abs_tol=1e-9),
                f"interrupted resource closure {kind} cumulative time changed",
            )
            open_events.pop(kind)
        else:
            raise IntegrityFailure(f"resource event {index} has an unknown type")
    _require(math.isclose(float(ledger["active_seconds"]), charged, rel_tol=0.0, abs_tol=1e-9), "resource active-time total changed")
    _require(sorted(open_events) == sorted(str(item) for item in ledger["open_events"]), "resource open-event inventory changed")
    _require(not open_events, "terminal resource ledger has an unresolved event")
    failed = ledger["failed_admission"]
    if failed is not None:
        _require(type(failed) is dict and int(failed.get("passed", 1)) == 0, "failed resource admission is invalid")
        if failed.get("phase") == "post-completion":
            _require(set(failed) == {"kind", "phase", "checks", "receipt", "passed"}, "post-completion resource-stop schema changed")
            receipt = failed.get("receipt")
            _require(type(receipt) is dict and receipt.get("event") == "complete", "post-completion resource receipt is invalid")
            expected_checks = {
                "quantum": float(receipt["elapsed_seconds"]) <= float(budget["maximum_quantum_seconds"]),
                "active": float(receipt["active_seconds_after"]) <= float(budget["max_active_seconds"]),
                "storage": int(receipt["storage_bytes_after"]) <= int(budget["max_storage_bytes"]),
                "cuda": float(receipt["cuda_fraction"]) <= float(budget["max_cuda_fraction"]),
            }
            _require(failed.get("checks") == expected_checks and not all(expected_checks.values()), "post-completion resource stop changed")
            _require(any(event == receipt for event in events), "post-completion resource receipt is not in the ledger")
        else:
            _require(
                set(failed)
                == {
                    "kind",
                    "predicted_seconds",
                    "predicted_next_bytes",
                    "active_seconds_before",
                    "reserve_remaining_seconds",
                    "storage_bytes_before",
                    "cuda_allocated_bytes",
                    "cuda_total_bytes",
                    "cuda_fraction",
                    "checks",
                    "passed",
                },
                "failed resource admission schema changed",
            )
            for field in (
                "predicted_seconds",
                "predicted_next_bytes",
                "active_seconds_before",
                "reserve_remaining_seconds",
                "storage_bytes_before",
                "cuda_fraction",
            ):
                _require(field in failed, f"failed resource admission omits {field}")
            predicted = float(failed["predicted_seconds"])
            predicted_bytes = int(failed["predicted_next_bytes"])
            reserve = float(failed["reserve_remaining_seconds"])
            active_before = float(failed["active_seconds_before"])
            storage_before = int(failed["storage_bytes_before"])
            _require(
                all(math.isfinite(value) for value in (predicted, reserve, active_before))
                and predicted > 0.0
                and predicted_bytes > 0
                and reserve >= 0.0
                and storage_before >= 0
                and math.isclose(active_before, charged, rel_tol=0.0, abs_tol=1e-9),
                "failed resource admission inputs changed",
            )
            expected_reserve = (
                0.0
                if "terminalization" in str(failed["kind"])
                else float(budget["reserve_seconds"])
            )
            _require(math.isclose(reserve, expected_reserve, rel_tol=0.0, abs_tol=0.0), "failed resource admission reserve changed")
            verify_cuda_receipt(failed, "failed resource admission")
            expected_checks = admission_checks(failed)
            _require(failed.get("checks") == expected_checks and not all(expected_checks.values()), "failed resource admission inequality changed")
    if status["state"] in {"awaiting_human_review", "complete"}:
        _require(not open_events and failed is None, "successful route has unfinished or failed resource authority")
        _require(charged <= float(budget["max_active_seconds"]), "successful route exceeded active-time cap")
        _require(_storage_bytes(run_dir) <= int(budget["max_storage_bytes"]), "successful route exceeded storage cap")
    elif status["state"] == "resource_stopped":
        projection_path = run_dir / "preflight" / "resource_projection.json"
        projection_failed = projection_path.is_file() and int(_read_json(projection_path).get("passed", 1)) == 0
        _require(
            failed is not None or charged > float(budget["max_active_seconds"]) or projection_failed,
            "resource-stopped route has no resource stop authority",
        )
    else:
        _require(
            failed is None or str(failed.get("kind")) == "failure_terminalization",
            "integrity-failed route contains an unrelated resource-stop authority",
        )
    if "route" in status:
        terminal_kind = {
            "awaiting_human_review": "machine_terminalization",
            "complete": "human_review_terminalization",
        }.get(str(status["state"]), "failure_terminalization")
        terminal_admits = [event for event in events if event.get("event") == "admit" and event.get("kind") == terminal_kind]
        terminal_closures = [
            event
            for event in events
            if event.get("event") in {"complete", "failed-complete", "interrupted-close"}
            and event.get("kind") == terminal_kind
        ]
        if not terminal_admits:
            _require(
                status["state"] in {"resource_stopped", "integrity_failed"}
                and failed is not None
                and failed.get("kind") == "failure_terminalization",
                "terminal resource admission is absent",
            )
        elif terminal_kind == "human_review_terminalization":
            _require(
                len(terminal_admits) == len(terminal_closures) >= 1,
                "human-review terminal resource pairing changed",
            )
            filtered = [
                event
                for event in events
                if event.get("kind") == terminal_kind
                and event.get("event") in {"admit", "complete", "failed-complete", "interrupted-close"}
            ]
            _require(
                len(filtered) == 2 * len(terminal_admits)
                and all(filtered[2 * index].get("event") == "admit" for index in range(len(terminal_admits)))
                and all(
                    filtered[2 * index + 1].get("event") == "interrupted-close"
                    for index in range(len(terminal_admits) - 1)
                )
                and filtered[-1].get("event") in {"complete", "failed-complete"},
                "human-review terminal recovery order changed",
            )
        else:
            _require(len(terminal_admits) == len(terminal_closures) == 1, "terminal resource event pairing changed")
    return ledger


def _bound_evaluator_replay_device(run_config: Mapping[str, Any]) -> str:
    execution = run_config.get("execution_authority")
    _require(type(execution) is dict, "evaluator replay execution authority changed")
    device = execution.get("device")
    _require(isinstance(device, str) and bool(device.strip()), "evaluator replay device changed")
    return device


def _verify_evaluator_replay_arrays(
    name: str,
    replay_arrays: Mapping[str, np.ndarray],
    *,
    sample_ids: np.ndarray,
    requested_labels: np.ndarray,
    predictions: np.ndarray,
    logits: np.ndarray,
) -> None:
    _require(
        np.array_equal(replay_arrays["sample_ids"].astype(str), sample_ids),
        f"{name} evaluator replay sample IDs changed",
    )
    _require(
        np.array_equal(replay_arrays["requested_labels"], requested_labels),
        f"{name} evaluator replay labels changed",
    )
    _require(
        np.array_equal(replay_arrays["predictions"], predictions),
        f"{name} evaluator replay predictions changed",
    )
    _require(
        np.array_equal(replay_arrays["logits"], logits),
        f"{name} evaluator replay logits changed",
    )


def _verify_evaluation(run_dir: Path, population_seal: Mapping[str, Any]) -> dict[str, Any]:
    seal_path = run_dir / "populations" / "POPULATIONS_SEALED.json"
    test_event = _read_json(run_dir / "data" / "test_open_event.json")
    _require(test_event.get("schema") == VERSION + "-test-open-event", "terminal-test open-event version changed")
    _require(test_event.get("population_seal_sha256") == sha256_file(seal_path), "terminal-test open-event seal changed")
    _require(test_event.get("arff_sha256") == MNIST_ARFF_SHA256, "terminal-test open-event ARFF authority changed")
    _require(test_event.get("generation_after_event_forbidden") == 1, "post-test generation firewall changed")
    _require(
        test_event.get("population_rows")
        == {name: population_seal["rows"][name]["raw_file_sha256"] for name in ("teacher", "null", "learned")},
        "terminal-test open-event population authority changed",
    )
    _require(
        set(test_event)
        == {
            "schema",
            "opened_at",
            "arff_sha256",
            "population_seal_sha256",
            "population_rows",
            "generation_after_event_forbidden",
        },
        "terminal-test open-event schema changed",
    )
    evaluator_event = _read_json(run_dir / "evaluation" / "EVALUATOR_OPEN_EVENT.json")
    _require(evaluator_event.get("schema") == VERSION + "-evaluator-open-event", "evaluator open-event version changed")
    _require(evaluator_event.get("population_seal_sha256") == sha256_file(seal_path), "evaluator open-event seal changed")
    _require(
        evaluator_event.get("evaluator_binding_sha256")
        == sha256_file(run_dir / "input_bindings" / "ddpm_evaluator_binding.json"),
        "evaluator open-event binding changed",
    )
    _require(
        set(evaluator_event) == {"schema", "opened_at", "population_seal_sha256", "evaluator_binding_sha256"},
        "evaluator open-event schema changed",
    )

    run_config = _read_json(run_dir / "config.json")
    arff_path = Path(run_config["input_paths"]["arff"])
    development_images, development_labels, _ = read_mnist_development_prefix(arff_path)
    terminal_images, terminal_labels = read_mnist_arff_slice(arff_path, 60_000, 70_000)
    expected_reference, expected_reference_labels, expected_reference_indices = _reference_subset(
        terminal_images,
        terminal_labels,
    )

    reference = _verifier_npz(
        run_dir / "evaluation" / "terminal_reference_uint8.npz",
        {"images", "labels", "terminal_local_indices"},
    )
    _require(reference["images"].dtype == np.uint8 and reference["images"].shape == (PATH_COUNT, 28, 28), "terminal reference images changed")
    _require(np.array_equal(reference["images"], expected_reference), "terminal reference pixels changed")
    _require(np.array_equal(reference["labels"], expected_reference_labels), "terminal reference labels changed")
    terminal_indices = reference["terminal_local_indices"]
    _require(terminal_indices.dtype == np.int64 and terminal_indices.shape == (PATH_COUNT,), "terminal reference indices changed")
    _require(np.array_equal(terminal_indices, expected_reference_indices), "terminal reference index authority changed")

    replay_device = _bound_evaluator_replay_device(run_config)
    evaluator = _load_evaluator_after_population_seal(run_dir, device=replay_device)

    prediction_keys = {
        f"{name}_{suffix}"
        for name in ("teacher", "null", "learned")
        for suffix in ("sample_ids", "requested_labels", "predictions", "logits")
    }
    predictions = _verifier_npz(run_dir / "evaluation" / "predictions.npz", prediction_keys)
    row_metrics: dict[str, dict[str, Any]] = {}
    for name in ("teacher", "null", "learned"):
        population = _verifier_npz(
            run_dir / "populations" / f"{name}_uint8.npz",
            {"anchors", "anchor_steps", "labels", "path_ids"},
        )
        endpoints = population["anchors"][-1]
        labels = population["labels"].astype(np.int64, copy=False)
        path_ids = population["path_ids"].astype(str)
        sample_ids = predictions[f"{name}_sample_ids"].astype(str)
        requested = predictions[f"{name}_requested_labels"]
        predicted = predictions[f"{name}_predictions"]
        logits = predictions[f"{name}_logits"]
        _require(np.array_equal(sample_ids, path_ids), f"{name} evaluator sample IDs changed")
        _require(requested.dtype == np.int64 and np.array_equal(requested, labels), f"{name} evaluator requested labels changed")
        _require(predicted.dtype == np.int64 and predicted.shape == (PATH_COUNT,), f"{name} evaluator predictions changed")
        _require(bool(np.all((predicted >= 0) & (predicted <= 9))), f"{name} evaluator predictions are invalid")
        _require(logits.dtype == np.float64 and logits.shape == (PATH_COUNT, 10) and bool(np.all(np.isfinite(logits))), f"{name} evaluator logits changed")

        metrics = _read_json(run_dir / "evaluation" / f"{name}_metrics.json")
        _require(set(metrics) == {"classifier", "duplicates", "diversity", "exact_reference_match_count", "exact_validation_match_count", "fixed_render_statistics"}, f"{name} metric schema changed")
        classifier = metrics["classifier"]
        _require(set(classifier) == {"loss", "accuracy", "requested_label_accuracy", "per_class"}, f"{name} compact classifier schema changed")
        accuracy = float(np.mean(predicted == labels))
        _require(math.isclose(float(classifier["accuracy"]), accuracy, rel_tol=0.0, abs_tol=1e-15), f"{name} classifier accuracy changed")
        _require(math.isclose(float(classifier["requested_label_accuracy"]), accuracy, rel_tol=0.0, abs_tol=1e-15), f"{name} requested-label accuracy changed")
        maximum = logits.max(axis=1, keepdims=True)
        log_partition = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
        loss = float(np.mean(log_partition - logits[np.arange(PATH_COUNT), labels]))
        _require(math.isclose(float(classifier["loss"]), loss, rel_tol=1e-6, abs_tol=1e-6), f"{name} classifier loss changed")
        expected_per_class = {
            str(digit): {
                "count": PATHS_PER_CLASS,
                "accuracy": float(np.mean(predicted[labels == digit] == digit)),
            }
            for digit in range(10)
        }
        _require(classifier["per_class"] == expected_per_class, f"{name} per-class classifier metrics changed")
        _require(metrics["duplicates"] == _jsonable(exact_duplicate_metrics(endpoints, labels, path_ids)), f"{name} duplicate metrics changed")
        expected_diversity = within_class_nn_diversity(
            endpoints,
            labels,
            reference["images"],
            reference["labels"],
        )
        _require(metrics["diversity"] == _jsonable(expected_diversity), f"{name} diversity metrics changed")
        _require(metrics["fixed_render_statistics"] == _fixed_render_statistics(endpoints), f"{name} fixed-render statistics changed")
        matches = metrics["exact_reference_match_count"]
        _require(type(matches) is dict and set(matches) == {"train", "test"}, f"{name} reference-match schema changed")
        _require(
            matches
            == {
                "train": _exact_match_count(endpoints, development_images[TRAIN_START:TRAIN_STOP]),
                "test": _exact_match_count(endpoints, terminal_images),
            },
            f"{name} exact reference-match counts changed",
        )
        _require(
            metrics["exact_validation_match_count"]
            == _exact_match_count(endpoints, development_images[VALIDATION_START:VALIDATION_STOP]),
            f"{name} exact validation-match count changed",
        )
        replay_full = compute_generation_metrics(
            evaluator,
            endpoints,
            labels,
            path_ids,
            real_reference_images=expected_reference,
            real_reference_labels=expected_reference_labels,
            train_images=development_images[TRAIN_START:TRAIN_STOP],
            test_images=terminal_images,
            device=replay_device,
        )
        replay_metrics, replay_arrays = _split_metric_arrays(replay_full)
        _verify_evaluator_replay_arrays(
            name,
            replay_arrays,
            sample_ids=sample_ids,
            requested_labels=requested,
            predictions=predicted,
            logits=logits,
        )
        _require(replay_metrics["duplicates"] == metrics["duplicates"] and replay_metrics["diversity"] == metrics["diversity"], f"{name} evaluator metric replay changed")
        row_metrics[name] = metrics

    effects = _read_json(run_dir / "evaluation" / "learned_minus_null.json")
    learned_accuracy = float(row_metrics["learned"]["classifier"]["requested_label_accuracy"])
    null_accuracy = float(row_metrics["null"]["classifier"]["requested_label_accuracy"])
    expected_effects = {
        "schema": VERSION + "-learned-minus-null",
        "classifier_accuracy_difference": learned_accuracy - null_accuracy,
        "learned_classifier_accuracy": learned_accuracy,
        "null_classifier_accuracy": null_accuracy,
        "learned_duplicate_pair_count": int(row_metrics["learned"]["duplicates"]["duplicate_pair_count"]),
        "learned_diversity_ratio": float(row_metrics["learned"]["diversity"]["aggregate_median_ratio"]),
        "row_effects_are_stochastic_unpaired": 1,
    }
    _require(effects == expected_effects, "learned-minus-null evaluator effect changed")
    contextual = _read_json(run_dir / "evaluation" / "contextual_ddpm_comparison.json")
    _require(
        contextual
        == {
            "schema": VERSION + "-contextual-ddpm",
            "role": "contextual exploratory calibration, not a paired or hypothesis-test baseline",
            "classifier_accuracy": 0.925,
            "human_requested_label_agreement": 0.925,
            "human_recognizability": 1.0,
            "duplicate_pair_count": 0,
            "diversity_ratio": 1.0925057312146145,
            "ddpm_tree_digest": DDPM_TREE_DIGEST,
        },
        "contextual DDPM authority changed",
    )

    target = _verifier_npz(
        run_dir / "inventory" / "teacher_target_bank.npz",
        {
            "masses",
            "source_images_uint8",
            "rendered_images_uint8",
            "requested_labels",
            "validation_local_ids",
            "arff_global_row_ids",
            "path_ids",
        },
    )
    teacher = _load_row_population(run_dir / "populations" / "teacher.npz", expected_row="teacher")
    targets = target["masses"].astype(np.float32, copy=False)
    _require(targets.shape == (PATH_COUNT, 784) and bool(np.all(np.isfinite(targets))) and float(targets.min()) >= 0.0, "teacher target masses changed")
    _require(float(np.max(np.abs(targets.sum(axis=1, dtype=np.float64) - 1.0))) <= 2e-6, "teacher target mass changed")
    expected_inventory = build_path_inventory()
    _require(np.array_equal(target["requested_labels"], expected_inventory["requested_labels"]), "teacher target labels changed")
    _require(np.array_equal(target["path_ids"].astype(str), expected_inventory["path_ids"].astype(str)), "teacher target path IDs changed")
    _require(np.array_equal(target["arff_global_row_ids"], target["validation_local_ids"] + VALIDATION_START), "teacher target ARFF IDs changed")
    source_images = target["source_images_uint8"]
    _require(source_images.dtype == np.uint8 and source_images.shape == (PATH_COUNT, 28, 28), "teacher source-image authority changed")
    expected_target_masses = np.maximum(source_images.reshape(PATH_COUNT, -1).astype(np.float32), np.float32(1e-8))
    expected_target_masses = (expected_target_masses / expected_target_masses.sum(axis=1, keepdims=True)).astype(np.float32)
    _require(np.array_equal(targets, expected_target_masses), "teacher target masses do not match source validation pixels")
    _require(
        np.array_equal(
            target["rendered_images_uint8"],
            mass_to_uint8(targets, _read_json(run_dir / "input_bindings" / "mass_to_uint8.json")),
        ),
        "teacher target rendering changed",
    )
    errors = np.sum((teacher["anchors"].astype(np.float64) - targets.astype(np.float64)[None]) ** 2, axis=2)
    ratios = errors / np.maximum(errors[0:1], np.finfo(np.float64).tiny)
    improved = errors[-1] < errors[0]
    control_arrays = _verifier_npz(
        run_dir / "controls" / "teacher_gate_arrays.npz",
        {"squared_l2", "relative_squared_l2", "endpoint_improved", "anchor_steps"},
    )
    _require(np.array_equal(control_arrays["squared_l2"], errors), "teacher squared-error evidence changed")
    _require(np.array_equal(control_arrays["relative_squared_l2"], ratios), "teacher relative-error evidence changed")
    _require(np.array_equal(control_arrays["endpoint_improved"], improved.astype(np.uint8)), "teacher improvement evidence changed")
    _require(np.array_equal(control_arrays["anchor_steps"], np.asarray(ANCHORS, dtype=np.int64)), "teacher control anchors changed")
    control = _read_json(run_dir / "controls" / "teacher_gate.json")
    conditions = {
        "median_ratio_anchor64_at_most_0_80": float(np.median(ratios[ANCHORS.index(64)])) <= 0.80,
        "median_ratio_endpoint_at_most_0_20": float(np.median(ratios[-1])) <= 0.20,
        "improved_path_count_at_least_144": int(improved.sum()) >= 144,
        "teacher_classifier_accuracy_at_least_0_80": float(row_metrics["teacher"]["classifier"]["requested_label_accuracy"]) >= CLASSIFIER_POSITIVE_ACCURACY,
        "target_render_health": True,
        "teacher_render_health": True,
    }
    _require(control.get("schema") == VERSION + "-teacher-positive-control", "teacher control version changed")
    _require(control.get("gate_type") == "execution/integrity", "teacher control gate type changed")
    _require(math.isclose(float(control.get("median_relative_squared_l2_anchor64")), float(np.median(ratios[ANCHORS.index(64)])), rel_tol=0.0, abs_tol=1e-15), "teacher anchor-64 metric changed")
    _require(math.isclose(float(control.get("median_relative_squared_l2_endpoint")), float(np.median(ratios[-1])), rel_tol=0.0, abs_tol=1e-15), "teacher endpoint metric changed")
    _require(int(control.get("endpoint_improved_path_count")) == int(improved.sum()), "teacher improvement count changed")
    _require(control.get("conditions") == {key: int(value) for key, value in conditions.items()}, "teacher control conditions changed")
    _require(int(control.get("passed")) == int(all(conditions.values())), "teacher control result changed")
    _require(control.get("arrays_sha256") == sha256_file(run_dir / "controls" / "teacher_gate_arrays.npz"), "teacher control array binding changed")
    _require(control.get("target_bank_sha256") == sha256_file(run_dir / "inventory" / "teacher_target_bank.npz"), "teacher target-bank binding changed")

    ready = _read_json(run_dir / "evaluation" / "SCORING_READY.json")
    _require(ready.get("schema") == VERSION + "-scoring-ready", "scoring-ready version changed")
    _require(ready.get("population_seal_sha256") == sha256_file(seal_path), "scoring-ready population binding changed")
    _require(ready.get("test_open_event_sha256") == sha256_file(run_dir / "data" / "test_open_event.json"), "scoring-ready test-event binding changed")
    _require(ready.get("predictions_sha256") == sha256_file(run_dir / "evaluation" / "predictions.npz"), "scoring-ready prediction binding changed")
    _require(ready.get("metrics_sha256") == {name: sha256_file(run_dir / "evaluation" / f"{name}_metrics.json") for name in row_metrics}, "scoring-ready metric bindings changed")
    _require(ready.get("teacher_control_sha256") == sha256_file(run_dir / "controls" / "teacher_gate.json"), "scoring-ready teacher binding changed")
    _require(int(ready.get("teacher_control_passed")) == int(control["passed"]), "scoring-ready teacher result changed")
    return {"rows": row_metrics, "effects": effects, "teacher_control": control, "ready": ready}


def _verify_review_bundle(run_dir: Path, population_seal: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        (run_dir / "review" / "README.md").read_text(encoding="utf-8")
        == "Complete the blinded CSV before opening review_key.json or machine metrics.\n",
        "blind-review instructions changed",
    )
    indices_path = run_dir / "review" / "review_indices.npy"
    indices = np.load(indices_path, allow_pickle=False)
    expected_indices = np.asarray(
        [digit * PATHS_PER_CLASS + offset for digit in range(10) for offset in REVIEW_WITHIN_CLASS],
        dtype=np.int64,
    )
    _require(indices.dtype == np.int64 and np.array_equal(indices, expected_indices), "blind-review index inventory changed")
    membership_data = _read_json(run_dir / "review" / "private_membership.json")
    _require(membership_data.get("schema") == VERSION + "-review-membership", "review membership version changed")
    entries = membership_data.get("entries")
    _require(type(entries) is list and len(entries) == 80, "review membership count changed")
    expected_membership: list[dict[str, Any]] = []
    review_images: list[np.ndarray] = []
    review_labels: list[np.ndarray] = []
    for name in ("learned", "null"):
        population = _verifier_npz(
            run_dir / "populations" / f"{name}_uint8.npz",
            {"anchors", "anchor_steps", "labels", "path_ids"},
        )
        review_images.append(population["anchors"][-1, expected_indices])
        review_labels.append(population["labels"][expected_indices])
        for index in expected_indices.tolist():
            member_index = len(expected_membership)
            expected_membership.append(
                {
                    "member_id": f"review-member-{member_index:03d}",
                    "row": name,
                    "path_id": str(population["path_ids"][index]),
                    "requested_label": int(population["labels"][index]),
                    "path_index": index,
                }
            )
    _require(entries == expected_membership, "private review membership changed")
    images = np.concatenate(review_images, axis=0)
    labels = np.concatenate(review_labels, axis=0).astype(np.int64)
    member_ids = np.asarray([entry["member_id"] for entry in expected_membership], dtype=np.str_)
    order = np.random.default_rng(REVIEW_SEED).permutation(80)

    key = _read_json(run_dir / "review" / "review_key.json")
    expected_key_entries = [
        {
            "review_order": index,
            "sample_id": f"blind-{index:03d}",
            "source_sample_id": str(member_ids[source_index]),
            "requested_label": int(labels[source_index]),
        }
        for index, source_index in enumerate(order.tolist())
    ]
    _require(key == {"schema": "mnist-blinded-review-v1", "seed": REVIEW_SEED, "entries": expected_key_entries}, "blind-review key changed")
    sample_files = sorted((run_dir / "review" / "samples").glob("sample-*.png"))
    _require([path.name for path in sample_files] == [f"sample-{index:03d}.png" for index in range(80)], "blind-review sample inventory changed")
    for index, path in enumerate(sample_files):
        _require(np.array_equal(np.asarray(Image.open(path).convert("L"), dtype=np.uint8), images[order[index]]), f"blind-review sample pixels changed at {index}")
    _verify_sheet_pixels(
        run_dir / "review" / "blinded-contact-sheet.png",
        images[order],
        columns=10,
        scale=4,
        captions=[f"blind-{index:03d}" for index in range(80)],
    )
    with (run_dir / "review" / "human_review_template.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == ["review_order", "sample_id", "assigned_label", "notes"], "blind-review template columns changed")
        template_rows = list(reader)
    _require(
        template_rows
        == [
            {"review_order": str(index), "sample_id": f"blind-{index:03d}", "assigned_label": "", "notes": ""}
            for index in range(80)
        ],
        "blind-review template rows changed",
    )
    ready = _read_json(run_dir / "review" / "READY.json")
    _require(ready.get("schema") == VERSION + "-review-ready", "review-ready version changed")
    _require(ready.get("population_seal_sha256") == sha256_file(run_dir / "populations" / "POPULATIONS_SEALED.json"), "review-ready population seal changed")
    _require(ready.get("population_tree_binding") == _sha256_bytes(_canonical_json_bytes(population_seal)), "review-ready population tree binding changed")
    _require((ready.get("sample_count"), ready.get("learned_count"), ready.get("null_count"), ready.get("review_seed")) == (80, 40, 40, REVIEW_SEED), "review-ready inventory changed")
    _require(ready.get("template_sha256") == sha256_file(run_dir / "review" / "human_review_template.csv"), "review-ready template binding changed")
    _require(ready.get("contact_sheet_sha256") == sha256_file(run_dir / "review" / "blinded-contact-sheet.png"), "review-ready contact-sheet binding changed")
    _require(ready.get("review_key_sha256") == sha256_file(run_dir / "review" / "review_key.json"), "review-ready key binding changed")
    _require(ready.get("membership_sha256") == sha256_file(run_dir / "review" / "private_membership.json"), "review-ready membership binding changed")
    return {"ready": ready, "key": key, "membership": membership_data}


def _replay_human_review(run_dir: Path, review: Mapping[str, Any]) -> dict[str, Any]:
    answers_path = run_dir / "review" / "human_review_answers.csv"
    overall = _read_json(run_dir / "review" / "human_review.json")
    expected_overall = score_human_review(
        answers_path,
        run_dir / "review" / "review_key.json",
        reviewer=str(overall.get("reviewer", "")),
        confirm_manual_review=True,
        timestamp=str(overall.get("recorded_at", "")),
    )
    _require(overall == expected_overall, "human-review aggregate changed")
    key_by_id = {str(entry["sample_id"]): entry for entry in review["key"]["entries"]}
    member_by_id = {str(entry["member_id"]): entry for entry in review["membership"]["entries"]}
    with answers_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == ["review_order", "sample_id", "assigned_label", "notes"], "submitted review columns changed")
        answer_rows = list(reader)
    _require(len(answer_rows) == 80, "submitted review row count changed")
    scored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for answer in answer_rows:
        blind = str(answer["sample_id"])
        _require(blind in key_by_id and blind not in seen, "submitted review membership changed")
        seen.add(blind)
        key_entry = key_by_id[blind]
        _require(int(answer["review_order"]) == int(key_entry["review_order"]), "submitted review order changed")
        member = member_by_id[str(key_entry["source_sample_id"])]
        assignment = str(answer["assigned_label"]).strip().lower()
        _require(assignment in {*(str(index) for index in range(10)), "noise", "ambiguous"}, "submitted review assignment is invalid")
        scored.append(
            {
                **member,
                "blind_id": blind,
                "assigned_label": assignment,
                "recognizable": int(assignment.isdigit()),
                "agreement": int(assignment == str(member["requested_label"])),
            }
        )
    expected_rows: dict[str, Any] = {}
    for name in ("learned", "null"):
        entries = [entry for entry in scored if entry["row"] == name]
        expected_rows[name] = {
            "sample_count": 40,
            "recognizable_count": sum(int(entry["recognizable"]) for entry in entries),
            "human_recognizability": float(np.mean([entry["recognizable"] for entry in entries])),
            "requested_label_agreement_count": sum(int(entry["agreement"]) for entry in entries),
            "human_requested_label_agreement": float(np.mean([entry["agreement"] for entry in entries])),
            "by_class": {
                str(digit): {
                    "count": len([entry for entry in entries if entry["requested_label"] == digit]),
                    "recognizable_count": sum(entry["recognizable"] for entry in entries if entry["requested_label"] == digit),
                    "agreement_count": sum(entry["agreement"] for entry in entries if entry["requested_label"] == digit),
                }
                for digit in range(10)
            },
        }
    by_row = _read_json(run_dir / "review" / "human_review_by_row.json")
    _require(by_row == {"schema": VERSION + "-human-by-row", "rows": expected_rows, "scored_entries": scored}, "row-stratified human review changed")
    return {"overall": overall, "rows": expected_rows, "scored": scored}


def _verify_input_authority_observations(
    run_dir: Path,
    config: Mapping[str, Any],
    *,
    require_all_matched: bool,
) -> dict[str, Any]:
    path = run_dir / "input_bindings" / "input_authority_observations.json"
    saved = _read_json(path)
    _require(
        set(saved) == {"schema", "recorded_at", "inputs", "all_expected_authorities_matched"}
        and saved["schema"] == VERSION + "-input-authority-observations"
        and isinstance(saved["recorded_at"], str)
        and bool(saved["recorded_at"].strip()),
        "input authority observation schema changed",
    )
    inputs = config["input_paths"]
    expected = _input_authority_observations(
        legacy_checkpoint=Path(inputs["legacy_checkpoint"]),
        arff=Path(inputs["arff"]),
        k128_run_dir=Path(inputs["k128_run_dir"]),
        ddpm_run_dir=Path(inputs["ddpm_run_dir"]),
        recorded_at=saved["recorded_at"],
    )
    _require(saved == expected, "observed input authority changed")
    if require_all_matched:
        _require(saved["all_expected_authorities_matched"] == 1, "bound input authority does not match the frozen input")
    return saved


def _verify_config_and_sources(
    run_dir: Path,
    resource_ledger: Mapping[str, Any],
    *,
    allow_input_mismatch: bool = False,
) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    _require(
        set(config)
        == {
            "schema",
            "version",
            "research_mode",
            "created_at",
            "command",
            "argv",
            "repository_root",
            "git_revision",
            "restart_count",
            "scientific_configuration",
            "execution_authority",
            "input_paths",
        },
        "config schema changed",
    )
    repository_root = _repository_root().resolve()
    _require(config["schema"] == VERSION + "-config" and config["version"] == VERSION, "config version changed")
    _require(config["research_mode"] == RESEARCH_MODE, "research mode changed")
    _require(isinstance(config["created_at"], str) and bool(config["created_at"].strip()), "config creation time is absent")
    _require(Path(config["repository_root"]).resolve() == repository_root, "config repository root is not the live source root")
    _require(config["git_revision"] == _git_revision(repository_root), "config git revision changed")
    _require(type(config["restart_count"]) is int and config["restart_count"] >= 0, "config restart count is invalid")

    expected_scientific = {
        "path_count": PATH_COUNT,
        "paths_per_class": PATHS_PER_CLASS,
        "outer_steps": OUTER_STEPS,
        "anchors": list(ANCHORS),
        "row_root_seeds": ROW_ROOT_SEEDS,
        "source_seed_base": SOURCE_SEED_BASE,
        "inventory_seed": INVENTORY_SEED,
        "test_only_smoke_seed": SMOKE_SEED,
        "review_seed": REVIEW_SEED,
        "review_offsets": list(REVIEW_WITHIN_CLASS),
        "replay_policy": {
            "generated_candidates_per_path": 1,
            "selector": None,
            "all_candidates_retained": 1,
            "adaptive_numerical_retry_substeps": [1, 2, 4],
        },
        "thresholds": {
            "teacher_median_ratio64": 0.80,
            "teacher_median_ratio256": 0.20,
            "teacher_improved_paths": 144,
            "teacher_classifier_accuracy": 0.80,
            "human_recognizability": REVIEW_POSITIVE_RECOGNIZABILITY,
            "human_requested_label_agreement": REVIEW_POSITIVE_AGREEMENT,
            "classifier_requested_label_accuracy": CLASSIFIER_POSITIVE_ACCURACY,
            "duplicate_pair_count": 0,
            "diversity_ratio": DIVERSITY_POSITIVE_RATIO,
        },
        "legacy_config_semantic_sha256": LEGACY_CONFIG_SHA256,
        "mass_transform": {
            "derivation_slice": [TRAIN_START, TRAIN_STOP],
            "numerator": MASS_SCALE_NUMERATOR,
            "denominator": MASS_SCALE_DENOMINATOR,
            "float_hex": MASS_SCALE_HEX,
        },
    }
    _require(config["scientific_configuration"] == expected_scientific, "scientific configuration changed")

    execution = config["execution_authority"]
    _require(
        type(execution) is dict
        and set(execution)
        == {
            "approval_id",
            "device",
            "max_active_seconds",
            "max_storage_bytes",
            "max_cuda_fraction",
            "reserve_seconds",
            "maximum_quantum_seconds",
        },
        "execution authority schema changed",
    )
    approval_id = execution["approval_id"]
    _require(
        isinstance(approval_id, str)
        and len(approval_id.strip()) >= 12
        and "placeholder" not in approval_id.lower()
        and "<" not in approval_id
        and ">" not in approval_id,
        "approval ID authority changed",
    )
    _require(execution["device"] == "cuda:0", "production device authority changed")
    bound_budget = ResourceBudget(
        max_active_seconds=float(execution["max_active_seconds"]),
        max_storage_bytes=int(execution["max_storage_bytes"]),
        max_cuda_fraction=float(execution["max_cuda_fraction"]),
        reserve_seconds=float(execution["reserve_seconds"]),
        maximum_quantum_seconds=float(execution["maximum_quantum_seconds"]),
    )
    _require(
        dataclasses.asdict(bound_budget) == resource_ledger["budget"],
        "execution authority and resource ledger budget disagree",
    )
    _require(
        bound_budget.reserve_seconds == TERMINAL_RESERVE_SECONDS
        and bound_budget.maximum_quantum_seconds == MAX_QUANTUM_SECONDS,
        "frozen reserve or quantum changed",
    )

    inputs = config["input_paths"]
    _require(
        type(inputs) is dict and set(inputs) == {"legacy_checkpoint", "ddpm_run_dir", "k128_run_dir", "arff"},
        "input-path authority changed",
    )
    for name, value in inputs.items():
        path = Path(value)
        _require(path.is_absolute() and path.resolve() == path, f"input path {name} is not canonical")
    _verify_input_authority_observations(run_dir, config, require_all_matched=not allow_input_mismatch)
    if not allow_input_mismatch:
        _require(Path(inputs["legacy_checkpoint"]).is_file(), "legacy checkpoint path is not a file")
        _require(Path(inputs["arff"]).is_file(), "ARFF path is not a file")
        _require(Path(inputs["ddpm_run_dir"]).is_dir() and Path(inputs["k128_run_dir"]).is_dir(), "bound predecessor path is not a directory")
        _require(Path(inputs["legacy_checkpoint"]).stat().st_size == LEGACY_CHECKPOINT_BYTES, "bound legacy checkpoint byte size changed")
        _require(sha256_file(Path(inputs["legacy_checkpoint"])) == LEGACY_CHECKPOINT_SHA256, "bound legacy checkpoint hash changed")
        _require(Path(inputs["arff"]).stat().st_size == MNIST_ARFF_BYTES, "bound ARFF byte size changed")
        _require(sha256_file(Path(inputs["arff"])) == MNIST_ARFF_SHA256, "bound ARFF hash changed")

    argv = config["argv"]
    _require(type(argv) is list and all(isinstance(value, str) for value in argv), "canonical argv is invalid")
    _require(
        len(argv) == 25
        and argv[1:5] == ["-B", "-m", "mnist.diag_d0_eulerian_edge_flux_replay", "run"],
        "canonical run argv prefix changed",
    )
    _require(config["command"] == subprocess.list2cmdline(argv), "canonical command and argv disagree")
    option_values = dict(zip(argv[5::2], argv[6::2], strict=True))
    _require(
        set(option_values)
        == {
            "--run-dir",
            "--legacy-checkpoint",
            "--ddpm-run-dir",
            "--k128-run-dir",
            "--arff",
            "--device",
            "--approval-id",
            "--max-active-seconds",
            "--max-storage-mib",
            "--max-cuda-fraction",
        },
        "canonical run option inventory changed",
    )
    _require(Path(option_values["--run-dir"]).resolve() == run_dir.resolve(), "canonical run directory changed")
    for option, input_name in (
        ("--legacy-checkpoint", "legacy_checkpoint"),
        ("--ddpm-run-dir", "ddpm_run_dir"),
        ("--k128-run-dir", "k128_run_dir"),
        ("--arff", "arff"),
    ):
        _require(Path(option_values[option]).resolve() == Path(inputs[input_name]), f"canonical option {option} changed")
    _require(option_values["--device"] == execution["device"], "canonical device changed")
    _require(option_values["--approval-id"] == execution["approval_id"], "canonical approval changed")
    _require(float(option_values["--max-active-seconds"]) == bound_budget.max_active_seconds, "canonical active cap changed")
    _require(int(float(option_values["--max-storage-mib"]) * 1024 * 1024) == bound_budget.max_storage_bytes, "canonical storage cap changed")
    _require(float(option_values["--max-cuda-fraction"]) == bound_budget.max_cuda_fraction, "canonical CUDA cap changed")

    source_bindings = _read_json(run_dir / "source_bindings.json")
    _require(source_bindings == _source_binding_payload(repository_root), "source bindings changed")
    _require((run_dir / "command.txt").read_text(encoding="utf-8") == config["command"] + "\n", "command receipt changed")
    _require(
        _read_json(run_dir / "claim_boundary.json")
        == {
            "schema": VERSION + "-claim-boundary",
            "research_mode": RESEARCH_MODE,
            "decision": "Does the pinned global edge-flux checkpoint produce factor-one task-visible MNIST under the current sampler and historical low-frequency source law?",
            "positive_scope": "exploratory compatibility of one checkpoint/source/sampler/transform tuple",
            "not_claimed": [
                "exact Doob h-transform or reference-prior correctness",
                "confirmatory generator quality",
                "DDPM superiority or stochastic pairing across rows",
                "general Eulerian-model feasibility or failure",
            ],
            "proxy_only_patches_since_last_objective_bearing_experiment": 0,
        },
        "claim boundary changed",
    )
    _require(
        _read_json(run_dir / "deterministic_execution.json")
        == {
            "schema": VERSION + "-deterministic-execution",
            "torch_deterministic_algorithms": 1,
            "cudnn_benchmark": 0,
            "cudnn_deterministic": 1,
            "tf32_policy_changed_by_runner": 0,
        },
        "deterministic execution authority changed",
    )
    return config


def _verify_predecessor_bindings(run_dir: Path, config: Mapping[str, Any]) -> None:
    inputs = config["input_paths"]
    k128_root = Path(inputs["k128_run_dir"])
    ddpm_root = Path(inputs["ddpm_run_dir"])
    k128 = _verify_external_manifest(
        k128_root,
        expected_manifest_sha256=K128_MANIFEST_SHA256,
        expected_tree_digest=K128_TREE_DIGEST,
    )
    _require(sha256_file(k128_root / "status.json") == K128_STATUS_SHA256, "K128 status authority changed")
    _require(sha256_file(k128_root / "outcome.json") == K128_OUTCOME_SHA256, "K128 outcome authority changed")
    _require(sha256_file(k128_root / "REPORT.md") == K128_REPORT_SHA256, "K128 report authority changed")
    _require(_read_json(k128_root / "status.json").get("route") == "complete", "K128 terminal route changed")
    k128_outcome = _read_json(k128_root / "outcome.json")
    _require(k128_outcome.get("route") == K128_REQUIRED_ROUTE, "K128 decision route changed")
    _require(k128_outcome.get("full_scale_auto_launched") == 0, "K128 forbidden automatic launch changed")
    ddpm = _verify_external_manifest(
        ddpm_root,
        expected_manifest_sha256=DDPM_MANIFEST_SHA256,
        expected_tree_digest=DDPM_TREE_DIGEST,
    )
    _require(sha256_file(ddpm_root / "evaluator" / "selection.json") == EVALUATOR_SELECTION_SHA256, "DDPM evaluator selection changed")
    evaluator_source = ddpm_root / "evaluator" / "selected_checkpoint.pt"
    _require(
        evaluator_source.stat().st_size == EVALUATOR_BYTES
        and sha256_file(evaluator_source) == EVALUATOR_SHA256,
        "DDPM evaluator checkpoint changed",
    )
    expected_predecessors = {
        "schema": VERSION + "-predecessor-bindings",
        "k128": {
            **k128,
            "status_sha256": K128_STATUS_SHA256,
            "outcome_sha256": K128_OUTCOME_SHA256,
            "report_sha256": K128_REPORT_SHA256,
            "required_route": K128_REQUIRED_ROUTE,
        },
        "ddpm": ddpm,
    }
    _require(_read_json(run_dir / "input_bindings" / "predecessors.json") == expected_predecessors, "predecessor bindings changed")
    copied = run_dir / "input_bindings" / "selected_checkpoint.pt"
    _require(copied.stat().st_size == EVALUATOR_BYTES and sha256_file(copied) == EVALUATOR_SHA256, "copied evaluator checkpoint changed")
    expected_evaluator = {
        "schema": VERSION + "-ddpm-evaluator-binding",
        "source_run": str(ddpm_root.resolve()),
        "source_manifest_sha256": DDPM_MANIFEST_SHA256,
        "source_tree_digest": DDPM_TREE_DIGEST,
        "selection_file_sha256": EVALUATOR_SELECTION_SHA256,
        "checkpoint_bytes": EVALUATOR_BYTES,
        "checkpoint_sha256": EVALUATOR_SHA256,
        "copied_checkpoint": "input_bindings/selected_checkpoint.pt",
        "copied_checkpoint_sha256": EVALUATOR_SHA256,
        "weights_loaded_before_population_seal": 0,
    }
    _require(_read_json(run_dir / "input_bindings" / "ddpm_evaluator_binding.json") == expected_evaluator, "evaluator binding changed")


def _verify_restart_history(run_dir: Path, config: Mapping[str, Any]) -> None:
    history_path = run_dir / "restart_history.json"
    if not history_path.is_file():
        _require(config["restart_count"] == 0, "config claims a restart without restart history")
        return
    history = _read_json(history_path)
    _require(set(history) == {"schema", "events"} and history["schema"] == VERSION + "-restart-history", "restart history schema changed")
    events = history["events"]
    _require(type(events) is list and bool(events), "restart history is empty or invalid")
    config_hash = sha256_file(run_dir / "config.json")
    start_seal_hash = sha256_file(run_dir / "inventory" / "START_BANK_SEALED.json")
    for index, event in enumerate(events, 1):
        _require(
            type(event) is dict
            and set(event)
            == {
                "restart_index",
                "mode",
                "recorded_at",
                "old_config_sha256",
                "old_start_bank_seal_sha256",
                "population_sealed_before_restart",
                "partial_resume_used",
                "new_config_sha256",
                "new_start_bank_seal_sha256",
            },
            f"restart event {index} schema changed",
        )
        _require(event["restart_index"] == index, f"restart event {index} index changed")
        _require(event["mode"] in {"rerun_all_rows", "continue_sealed"}, f"restart event {index} mode changed")
        _require(isinstance(event["recorded_at"], str) and bool(event["recorded_at"].strip()), f"restart event {index} time is absent")
        _require(event["partial_resume_used"] == 0, f"restart event {index} used a forbidden partial resume")
        _require(event["population_sealed_before_restart"] == int(event["mode"] == "continue_sealed"), f"restart event {index} seal/mode changed")
        _require(
            event["old_config_sha256"] == event["new_config_sha256"] == config_hash,
            f"restart event {index} config authority changed",
        )
        _require(
            event["old_start_bank_seal_sha256"] == event["new_start_bank_seal_sha256"] == start_seal_hash,
            f"restart event {index} start authority changed",
        )
    _require(config["restart_count"] == 0, "sealed scientific config was mutated by a restart")


def _verify_checkpoint_extract(run_dir: Path, config: Mapping[str, Any]) -> DirectFluxMNISTConfig:
    receipt_path = run_dir / "input_bindings" / "legacy_checkpoint_receipt.json"
    receipt = _read_json(receipt_path)
    _require(
        set(receipt)
        == {
            "schema",
            "checkpoint_path",
            "checkpoint_bytes",
            "checkpoint_sha256",
            "numpy_version",
            "pytorch_version",
            "weights_only",
            "safe_globals",
            "payload_keys",
            "config",
            "config_semantic_sha256",
            "historical_selection_fields",
            "replay_policy",
            "tensor_count",
            "parameter_count",
            "clean_state_path",
            "clean_state_bytes",
            "clean_state_sha256",
        },
        "legacy checkpoint receipt schema changed",
    )
    checkpoint = Path(config["input_paths"]["legacy_checkpoint"])
    _require(receipt["schema"] == VERSION + "-legacy-checkpoint-receipt", "legacy checkpoint receipt version changed")
    _require(Path(receipt["checkpoint_path"]).resolve() == checkpoint, "legacy checkpoint receipt path changed")
    _require(receipt["checkpoint_bytes"] == LEGACY_CHECKPOINT_BYTES and receipt["checkpoint_sha256"] == LEGACY_CHECKPOINT_SHA256, "legacy checkpoint receipt identity changed")
    _require(receipt["numpy_version"] == EXPECTED_NUMPY_VERSION and receipt["pytorch_version"] == EXPECTED_TORCH_VERSION, "legacy extraction runtime changed")
    _require(receipt["weights_only"] is True, "legacy checkpoint was not weights-only loaded")
    expected_safe_globals = [
        "numpy._core.multiarray._reconstruct",
        "numpy.ndarray",
        "numpy.dtype",
        "numpy.dtypes.Int64DType",
        "numpy.dtypes.Float64DType",
    ]
    _require(receipt["safe_globals"] == expected_safe_globals, "legacy extraction safe-global authority changed")
    _require(receipt["payload_keys"] == sorted(EXPECTED_CHECKPOINT_KEYS), "legacy checkpoint payload inventory changed")
    _require(_sha256_bytes(_canonical_json_bytes(receipt["config"])) == LEGACY_CONFIG_SHA256, "legacy config receipt changed")
    _require(receipt["config_semantic_sha256"] == LEGACY_CONFIG_SHA256, "legacy config semantic binding changed")
    _require(
        receipt["historical_selection_fields"]
        == {"sample_rejection_factor": 4, "sample_selection_metric": "composite"},
        "historical selection metadata changed",
    )
    _require(
        receipt["replay_policy"]
        == {"generated_candidates_per_path": 1, "selector": None, "all_candidates_retained": 1},
        "factor-one replay policy changed",
    )
    _require(receipt["tensor_count"] == EXPECTED_STATE_TENSORS and receipt["parameter_count"] == EXPECTED_PARAMETER_COUNT, "legacy model inventory changed")
    clean_path = run_dir / "input_bindings" / "clean_model_state.pt"
    _require(Path(receipt["clean_state_path"]).resolve() == clean_path.resolve(), "clean state receipt path changed")
    _require(receipt["clean_state_bytes"] == clean_path.stat().st_size, "clean state byte size changed")
    _require(receipt["clean_state_sha256"] == sha256_file(clean_path), "clean state hash changed")
    clean_receipt = _read_json(run_dir / "input_bindings" / "clean_model_state_receipt.json")
    _require(
        clean_receipt
        == {
            "schema": VERSION + "-clean-model-state-receipt",
            "path": "input_bindings/clean_model_state.pt",
            "bytes": clean_path.stat().st_size,
            "sha256": sha256_file(clean_path),
            "tensor_count": EXPECTED_STATE_TENSORS,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "strict_reload_passed": 1,
        },
        "clean model state receipt changed",
    )
    scientific_config = DirectFluxMNISTConfig(**receipt["config"])
    _require(int(scientific_config.grid_size) == 28, "legacy grid size changed")
    _load_clean_model(clean_path, config=scientific_config, device="cpu")
    return scientific_config


def _verify_data_and_inventory(
    run_dir: Path,
    config: Mapping[str, Any],
    scientific_config: DirectFluxMNISTConfig,
) -> tuple[np.ndarray, np.ndarray]:
    development_images, development_labels, audit = read_mnist_development_prefix(config["input_paths"]["arff"])
    _require(_read_json(run_dir / "data" / "development_roles.json") == audit, "development data-role audit changed")
    transform_path = run_dir / "input_bindings" / "mass_to_uint8.json"
    expected_transform = derive_mass_to_uint8_authority(development_images[TRAIN_START:TRAIN_STOP])
    _require(_read_json(transform_path) == expected_transform, "mass-to-uint8 authority changed")
    inventory = build_path_inventory()
    expected_rows = [
        {
            "path_id": str(inventory["path_ids"][index]),
            "path_index": str(int(inventory["path_indices"][index])),
            "requested_label": str(int(inventory["requested_labels"][index])),
            "within_class_index": str(int(inventory["within_class_indices"][index])),
            "source_seed": str(int(inventory["source_seeds"][index])),
            "generated_candidates": "1",
            "retained": "1",
        }
        for index in range(PATH_COUNT)
    ]
    with (run_dir / "inventory" / "path_inventory.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames
            == [
                "path_id",
                "path_index",
                "requested_label",
                "within_class_index",
                "source_seed",
                "generated_candidates",
                "retained",
            ],
            "path inventory columns changed",
        )
        _require(list(reader) == expected_rows, "path inventory rows changed")

    start_path = run_dir / "inventory" / "start_bank.npz"
    start = _verifier_npz(start_path, {"starts", "labels", "path_ids", "source_seeds"})
    expected_starts = build_start_bank(scientific_config, inventory)
    _require(np.array_equal(start["starts"], expected_starts), "start bank changed from the frozen source-seed law")
    _require(np.array_equal(start["labels"], inventory["requested_labels"]), "start-bank labels changed")
    _require(np.array_equal(start["path_ids"].astype(str), inventory["path_ids"].astype(str)), "start-bank path IDs changed")
    _require(np.array_equal(start["source_seeds"].astype(np.uint64), inventory["source_seeds"]), "start-bank source seeds changed")

    target_path = run_dir / "inventory" / "teacher_target_bank.npz"
    target = _verifier_npz(
        target_path,
        {
            "masses",
            "source_images_uint8",
            "rendered_images_uint8",
            "requested_labels",
            "validation_local_ids",
            "arff_global_row_ids",
            "path_ids",
        },
    )
    expected_target = build_teacher_target_bank(
        development_images[VALIDATION_START:VALIDATION_STOP],
        development_labels[VALIDATION_START:VALIDATION_STOP],
        inventory,
    )
    _require(np.array_equal(target["masses"], expected_target["masses"]), "teacher target masses changed")
    _require(np.array_equal(target["source_images_uint8"], expected_target["images_uint8"]), "teacher source validation pixels changed")
    _require(np.array_equal(target["rendered_images_uint8"], mass_to_uint8(expected_target["masses"], expected_transform)), "teacher rendered target pixels changed")
    for key in ("requested_labels", "validation_local_ids", "arff_global_row_ids", "path_ids"):
        _require(np.array_equal(target[key].astype(str) if key == "path_ids" else target[key], expected_target[key].astype(str) if key == "path_ids" else expected_target[key]), f"teacher target {key} changed")
    expected_target_rows = [
        {
            "path_id": str(expected_target["path_ids"][index]),
            "requested_label": str(int(expected_target["requested_labels"][index])),
            "validation_local_id": str(int(expected_target["validation_local_ids"][index])),
            "arff_global_row_id": str(int(expected_target["arff_global_row_ids"][index])),
        }
        for index in range(PATH_COUNT)
    ]
    with (run_dir / "inventory" / "teacher_target_ids.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == ["path_id", "requested_label", "validation_local_id", "arff_global_row_id"], "teacher target ID columns changed")
        _require(list(reader) == expected_target_rows, "teacher target ID rows changed")
    _require(
        _read_json(run_dir / "inventory" / "row_seeds.json")
        == {
            "schema": VERSION + "-row-seeds",
            "row_root_seeds": ROW_ROOT_SEEDS,
            "inventory_seed": INVENTORY_SEED,
            "source_seed_base": SOURCE_SEED_BASE,
            "source_seed_count": PATH_COUNT,
            "review_seed": REVIEW_SEED,
            "test_only_smoke_seed": SMOKE_SEED,
            "rows_are_separately_randomized_not_crn": 1,
        },
        "row-seed authority changed",
    )
    start_seal = _read_json(run_dir / "inventory" / "START_BANK_SEALED.json")
    _require(start_seal.get("schema") == VERSION + "-start-bank-sealed", "start-bank seal version changed")
    _require(start_seal.get("path_count") == PATH_COUNT, "start-bank path count changed")
    expected_bindings = {
        "start_bank_sha256": sha256_file(start_path),
        "starts_sha256": _hash_array(start["starts"]),
        "labels_sha256": _hash_array(start["labels"]),
        "path_ids_sha256": _hash_array(start["path_ids"]),
        "path_inventory_sha256": sha256_file(run_dir / "inventory" / "path_inventory.csv"),
        "teacher_target_bank_sha256": sha256_file(target_path),
        "teacher_target_mass_sha256": _hash_array(target["masses"]),
        "teacher_target_ids_sha256": sha256_file(run_dir / "inventory" / "teacher_target_ids.csv"),
        "row_seeds_sha256": sha256_file(run_dir / "inventory" / "row_seeds.json"),
        "mass_transform_sha256": sha256_file(transform_path),
        "evaluator_binding_sha256": sha256_file(run_dir / "input_bindings" / "ddpm_evaluator_binding.json"),
        "config_sha256": sha256_file(run_dir / "config.json"),
        "source_bindings_sha256": sha256_file(run_dir / "source_bindings.json"),
        "development_roles_sha256": sha256_file(run_dir / "data" / "development_roles.json"),
        "predecessor_bindings_sha256": sha256_file(run_dir / "input_bindings" / "predecessors.json"),
        "legacy_checkpoint_receipt_sha256": sha256_file(run_dir / "input_bindings" / "legacy_checkpoint_receipt.json"),
        "clean_state_sha256": sha256_file(run_dir / "input_bindings" / "clean_model_state.pt"),
        "clean_state_receipt_sha256": sha256_file(run_dir / "input_bindings" / "clean_model_state_receipt.json"),
        "terminal_test_content_rows_parsed": 0,
        "evaluator_weights_loaded": 0,
    }
    _require(set(start_seal) == {"schema", "created_at", "path_count", *expected_bindings}, "start-bank seal schema changed")
    for key, expected in expected_bindings.items():
        _require(start_seal[key] == expected, f"start-bank seal binding changed: {key}")
    return development_images, development_labels


def _verify_one_row_result(run_dir: Path, name: str, *, partial: bool) -> None:
    filename = f"partial_{name}.npz" if partial else f"{name}.npz"
    raw_archive = _verifier_npz(
        run_dir / "populations" / filename,
        {
            "anchors",
            "anchor_steps",
            "labels",
            "path_ids",
            "root_seed",
            "config_sha256",
            "checkpoint_sha256",
            "scientific_digest",
            "telemetry_json",
        },
    )
    anchors = raw_archive["anchors"]
    anchor_steps = raw_archive["anchor_steps"]
    if partial:
        _require(anchors.dtype == np.float32 and anchors.ndim == 3 and anchors.shape[1:] == (PATH_COUNT, 784), f"partial {name} anchor schema changed")
        _require(anchor_steps.dtype == np.int64 and anchor_steps.shape == (len(anchors),), f"partial {name} anchor-step schema changed")
        completed = len(raw_archive["telemetry_json"])
        expected_steps = [step for step in ANCHORS if step <= completed]
        if expected_steps[-1] != completed:
            expected_steps.append(completed)
        _require(anchor_steps.tolist() == expected_steps and 0 <= completed < OUTER_STEPS, f"partial {name} anchor schedule changed")
    else:
        loaded = _load_row_population(run_dir / "populations" / filename, expected_row=name)
        anchors = loaded["anchors"]
        anchor_steps = loaded["anchor_steps"]
    inventory = build_path_inventory()
    _require(np.array_equal(raw_archive["labels"], inventory["requested_labels"]), f"{name} row labels changed")
    _require(np.array_equal(raw_archive["path_ids"].astype(str), inventory["path_ids"].astype(str)), f"{name} row path IDs changed")
    _require(raw_archive["root_seed"].dtype == np.uint64 and raw_archive["root_seed"].tolist() == [ROW_ROOT_SEEDS[name]], f"{name} row root seed changed")
    _require(
        raw_archive["config_sha256"].shape == (1,)
        and str(raw_archive["config_sha256"][0]) == sha256_file(run_dir / "config.json"),
        f"{name} row config binding changed",
    )
    _require(
        raw_archive["checkpoint_sha256"].shape == (1,)
        and str(raw_archive["checkpoint_sha256"][0]) == LEGACY_CHECKPOINT_SHA256,
        f"{name} row checkpoint binding changed",
    )
    _require(bool(np.all(np.isfinite(anchors))) and float(anchors.min()) >= 0.0, f"{name} row states are invalid")
    _require(float(np.max(np.abs(anchors.sum(axis=2, dtype=np.float64) - 1.0))) <= 2e-6, f"{name} row mass changed")
    starts = _verifier_npz(run_dir / "inventory" / "start_bank.npz", {"starts", "labels", "path_ids", "source_seeds"})
    _require(np.array_equal(anchors[0], starts["starts"]), f"{name} row start anchor changed")
    telemetry = [json.loads(str(value)) for value in raw_archive["telemetry_json"].tolist()]
    raw = {**raw_archive, "telemetry": np.asarray(telemetry, dtype=object)}
    _verify_row_telemetry(run_dir, name, raw, partial=partial)
    _require(raw_archive["scientific_digest"].shape == (1,), f"{name} scientific digest schema changed")
    _require(str(raw_archive["scientific_digest"][0]) == _scientific_row_digest(anchors, telemetry), f"{name} scientific digest changed")


def _verify_synthetic_preflight_report(run_dir: Path) -> None:
    synthetic = _read_json(run_dir / "preflight" / "synthetic_teacher.json")
    _require(
        set(synthetic)
        == {"schema", "gate_type", "path_count", "squared_l2_before", "squared_l2_after", "checks", "passed"},
        "synthetic teacher preflight schema changed",
    )
    before = np.asarray(synthetic["squared_l2_before"], dtype=np.float64)
    after = np.asarray(synthetic["squared_l2_after"], dtype=np.float64)
    expected_checks = {
        "null_has_no_targets_parameter": "targets" not in inspect.signature(run_null_row).parameters,
        "learned_has_no_targets_parameter": "targets" not in inspect.signature(run_learned_row).parameters,
        "teacher_has_no_model_parameter": "model" not in inspect.signature(run_teacher_row).parameters,
        "teacher_moves_all_four_toward_target": bool(before.shape == (4,) and after.shape == (4,) and np.all(after < before)),
        "teacher_states_finite": bool(np.all(np.isfinite(before)) and np.all(np.isfinite(after))),
        "teacher_states_nonnegative": True,
    }
    _require(synthetic["schema"] == VERSION + "-synthetic-teacher-preflight", "synthetic teacher preflight version changed")
    _require(synthetic["gate_type"] == "execution/integrity" and synthetic["path_count"] == 4, "synthetic teacher preflight authority changed")
    _require(synthetic["checks"] == {key: int(value) for key, value in expected_checks.items()}, "synthetic teacher checks changed")
    _require(synthetic["passed"] == int(all(expected_checks.values())), "synthetic teacher result changed")


def _verify_preflight(run_dir: Path, resource_ledger: Mapping[str, Any]) -> None:
    _verify_synthetic_preflight_report(run_dir)
    deterministic = _read_json(run_dir / "preflight" / "deterministic_replay.json")
    _require(
        set(deterministic)
        == {
            "schema",
            "schedule_steps",
            "executed_probe_steps",
            "checks",
            "passed",
            "first_scientific_digest",
            "second_scientific_digest",
            "first_anchor_sha256",
            "second_anchor_sha256",
            "model_state_semantic_sha256_before",
            "model_state_semantic_sha256_after",
            "timing_seconds",
            "timing_and_allocator_excluded_from_scientific_digest",
        },
        "deterministic preflight schema changed",
    )
    deterministic_checks = {
        "learned_anchor_bytes_identical": deterministic["first_anchor_sha256"] == deterministic["second_anchor_sha256"],
        "learned_scientific_digest_identical": deterministic["first_scientific_digest"] == deterministic["second_scientific_digest"],
        "learned_retry_counts_identical": True,
        "learned_clipping_counts_identical": True,
        "model_state_unchanged": deterministic["model_state_semantic_sha256_before"] == deterministic["model_state_semantic_sha256_after"],
        "null_health": True,
        "teacher_health": True,
    }
    _require(deterministic["schema"] == VERSION + "-deterministic-replay", "deterministic preflight version changed")
    _require(deterministic["schedule_steps"] == OUTER_STEPS and deterministic["executed_probe_steps"] == 8, "deterministic probe schedule changed")
    _require(deterministic["checks"] == {key: int(value) for key, value in deterministic_checks.items()}, "deterministic preflight checks changed")
    _require(deterministic["passed"] == int(all(deterministic_checks.values())), "deterministic preflight result changed")
    _require(deterministic["timing_and_allocator_excluded_from_scientific_digest"] == 1, "deterministic scientific-digest boundary changed")
    timing = deterministic["timing_seconds"]
    _require(
        type(timing) is dict
        and set(timing) == {"charged_warmup", "learned8_first", "learned8_second", "null8", "teacher8"}
        and all(math.isfinite(float(value)) and float(value) >= 0.0 for value in timing.values()),
        "deterministic timing receipt changed",
    )
    projection = _read_json(run_dir / "preflight" / "resource_projection.json")
    relevant_kinds = {
        "device_warmup",
        "learned_determinism_probe_1",
        "learned_determinism_probe_2",
        "null_timing_probe",
        "teacher_timing_probe",
    }
    completions = {
        str(event["kind"]): float(event["elapsed_seconds"])
        for event in resource_ledger["events"]
        if event.get("event") == "complete" and event.get("kind") in relevant_kinds
    }
    _require(set(completions) == relevant_kinds, "preflight resource-event inventory changed")
    _require(math.isclose(completions["device_warmup"], float(timing["charged_warmup"]), rel_tol=0.0, abs_tol=0.0), "warm-up timing binding changed")
    _require(math.isclose(completions["learned_determinism_probe_1"], float(timing["learned8_first"]), rel_tol=0.0, abs_tol=0.0), "learned probe-1 timing binding changed")
    _require(math.isclose(completions["learned_determinism_probe_2"], float(timing["learned8_second"]), rel_tol=0.0, abs_tol=0.0), "learned probe-2 timing binding changed")
    _require(math.isclose(completions["null_timing_probe"], float(timing["null8"]), rel_tol=0.0, abs_tol=0.0), "null probe timing binding changed")
    _require(math.isclose(completions["teacher_timing_probe"], float(timing["teacher8"]), rel_tol=0.0, abs_tol=0.0), "teacher probe timing binding changed")
    for event in resource_ledger["events"]:
        if event.get("event") == "admit" and event.get("kind") in relevant_kinds:
            _require(
                float(event["predicted_seconds"]) == MAX_QUANTUM_SECONDS
                and int(event["predicted_next_bytes"]) == 2 * 1024 * 1024,
                f"preflight admission pricing changed: {event['kind']}",
            )
    charged_at_projection = 0.0
    peak_fraction_at_projection = 0.0
    found_projection_boundary = False
    for event in resource_ledger["events"]:
        peak_fraction_at_projection = max(peak_fraction_at_projection, float(event.get("cuda_fraction", 0.0)))
        if event.get("event") in {"complete", "failed-complete"}:
            charged_at_projection += float(event["elapsed_seconds"])
        elif event.get("event") == "interrupted-close":
            charged_at_projection += float(event["charged_predicted_seconds"])
        if event.get("event") == "complete" and event.get("kind") == "teacher_timing_probe":
            found_projection_boundary = True
            break
    _require(found_projection_boundary, "resource projection boundary is absent")
    _require(
        math.isclose(float(projection["charged_active_seconds"]), charged_at_projection, rel_tol=0.0, abs_tol=1e-9),
        "resource projection charged-time basis changed",
    )
    _require(
        math.isclose(float(projection["peak_cuda_fraction"]), peak_fraction_at_projection, rel_tol=0.0, abs_tol=0.0),
        "resource projection CUDA-fraction basis changed",
    )
    expected_projection = resource_projection(
        charged_active_seconds=charged_at_projection,
        teacher8_seconds=float(timing["teacher8"]),
        null8_seconds=float(timing["null8"]),
        learned8_seconds=max(float(timing["learned8_first"]), float(timing["learned8_second"])),
        projected_persisted_bytes=70 * 1024 * 1024,
        peak_cuda_fraction=peak_fraction_at_projection,
        budget=ResourceBudget(**resource_ledger["budget"]),
    )
    _require(projection == expected_projection, "resource projection changed")
    row_predictions = {
        "teacher": min(MAX_QUANTUM_SECONDS, max(0.001, 1.25 * float(timing["teacher8"]))),
        "null": min(MAX_QUANTUM_SECONDS, max(0.001, 1.25 * float(timing["null8"]))),
        "learned": min(
            MAX_QUANTUM_SECONDS,
            max(0.001, 1.25 * max(float(timing["learned8_first"]), float(timing["learned8_second"]))),
        ),
    }
    for row, predicted in row_predictions.items():
        admissions = [
            event
            for event in resource_ledger["events"]
            if event.get("event") == "admit" and str(event.get("kind", "")).startswith(f"{row}_row_q")
        ]
        _require(len(admissions) <= OUTER_STEPS // 8, f"{row} row has too many resource quanta")
        _require(
            [event["kind"] for event in admissions] == [f"{row}_row_q{index:02d}" for index in range(len(admissions))],
            f"{row} row resource-quantum sequence changed",
        )
        for admission in admissions:
            _require(
                math.isclose(float(admission["predicted_seconds"]), predicted, rel_tol=0.0, abs_tol=0.0)
                and int(admission["predicted_next_bytes"]) == 4 * 1024 * 1024,
                f"{row} row resource pricing changed",
            )


def _verify_machine_gates(
    run_dir: Path,
    resource_ledger: Mapping[str, Any],
    population_seal: Mapping[str, Any],
    *,
    complete: bool,
) -> dict[str, Any]:
    gates = _read_json(run_dir / "gates.json")
    _require(set(gates) == {"schema", "gate_a", "gate_b", "gate_c", "gate_d", "gate_e"}, "gate inventory changed")
    _require(gates["schema"] == VERSION + "-gates", "gate version changed")
    development_roles = _read_json(run_dir / "data" / "development_roles.json")
    gate_a_conditions = {
        "checkpoint_and_clean_state_bound": (run_dir / "input_bindings" / "clean_model_state_receipt.json").is_file(),
        "source_bindings_present": (run_dir / "source_bindings.json").is_file(),
        "data_roles_strict_prefix": int(development_roles["terminal_content_rows_parsed"]) == 0,
        "start_bank_sealed": (run_dir / "inventory" / "START_BANK_SEALED.json").is_file(),
        "predecessor_route_bound": _read_json(run_dir / "input_bindings" / "predecessors.json")["k128"]["required_route"] == K128_REQUIRED_ROUTE,
    }
    gate_b_conditions = {
        "sealed_start_count_160": int(_read_json(run_dir / "inventory" / "START_BANK_SEALED.json")["path_count"]) == PATH_COUNT,
        "three_rows_160_retained": all(int(population_seal["rows"][name]["endpoint_count"]) == PATH_COUNT for name in ("teacher", "null", "learned")),
        "factor_one_no_selector": population_seal["generated_candidates_per_path"] == 1 and population_seal["selector"] is None,
        "target_firewall": "targets" not in inspect.signature(run_null_row).parameters and "targets" not in inspect.signature(run_learned_row).parameters,
        "terminal_and_evaluator_opened_only_after_population_seal": (run_dir / "data" / "test_open_event.json").is_file() and (run_dir / "evaluation" / "EVALUATOR_OPEN_EVENT.json").is_file(),
    }
    deterministic = _read_json(run_dir / "preflight" / "deterministic_replay.json")
    projection = _read_json(run_dir / "preflight" / "resource_projection.json")
    peak_fraction = max([float(event.get("cuda_fraction", 0.0)) for event in resource_ledger["events"]] + [0.0])
    gate_c_conditions = {
        "deterministic_replay": int(deterministic["passed"]) == 1,
        "resource_projection": int(projection["passed"]) == 1,
        "population_semantic_seal": True,
        "no_failed_resource_admission": resource_ledger["failed_admission"] is None,
        "active_cap": float(resource_ledger["active_seconds"]) <= float(resource_ledger["budget"]["max_active_seconds"]),
        "storage_cap": _storage_bytes(run_dir) <= int(resource_ledger["budget"]["max_storage_bytes"]),
        "cuda_cap": peak_fraction <= float(resource_ledger["budget"]["max_cuda_fraction"]),
    }
    expected = {
        "gate_a": gate_a_conditions,
        "gate_b": gate_b_conditions,
        "gate_c": gate_c_conditions,
    }
    for name, conditions in expected.items():
        gate = gates[name]
        _require(set(gate) == {"gate_type", "passed", "conditions"}, f"{name} schema changed")
        _require(gate["gate_type"] == "execution/integrity", f"{name} type changed")
        _require(gate["conditions"] == {key: int(value) for key, value in conditions.items()}, f"{name} conditions changed")
        _require(gate["passed"] == int(all(conditions.values())), f"{name} result changed")
    teacher = _read_json(run_dir / "controls" / "teacher_gate.json")
    gate_d = gates["gate_d"]
    _require(set(gate_d) == {"gate_type", "passed", "conditions", "teacher_control_sha256"}, "gate_d schema changed")
    _require(gate_d["gate_type"] == "execution/integrity", "gate_d type changed")
    _require(gate_d["conditions"] == teacher["conditions"] and gate_d["passed"] == teacher["passed"], "gate_d control replay changed")
    _require(gate_d["teacher_control_sha256"] == sha256_file(run_dir / "controls" / "teacher_gate.json"), "gate_d teacher binding changed")
    gate_e = gates["gate_e"]
    _require(gate_e["gate_type"] == "diagnostic threshold", "gate_e type changed")
    if not complete:
        _require(gate_e == {"gate_type": "diagnostic threshold", "state": "pending", "passed": None, "conditions": {}}, "pending gate_e changed")
    return gates


def _verify_complete_outcome(
    run_dir: Path,
    status: Mapping[str, Any],
    gates: Mapping[str, Any],
    human: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    by_row = human["rows"]
    learned_metrics = evaluation["rows"]["learned"]
    null_metrics = evaluation["rows"]["null"]
    learned_human_positive = (
        float(by_row["learned"]["human_recognizability"]) >= REVIEW_POSITIVE_RECOGNIZABILITY
        and float(by_row["learned"]["human_requested_label_agreement"]) >= REVIEW_POSITIVE_AGREEMENT
    )
    human_exceeds_null = (
        float(by_row["learned"]["human_requested_label_agreement"])
        > float(by_row["null"]["human_requested_label_agreement"])
    )
    learned_classifier = float(learned_metrics["classifier"]["requested_label_accuracy"])
    null_classifier = float(null_metrics["classifier"]["requested_label_accuracy"])
    classifier_components = {
        "absolute_accuracy": learned_classifier >= CLASSIFIER_POSITIVE_ACCURACY,
        "exceeds_null": learned_classifier > null_classifier,
    }
    noncollapse_components = {
        "zero_duplicate_pairs": int(learned_metrics["duplicates"]["duplicate_pair_count"]) == 0,
        "diversity": float(learned_metrics["diversity"]["aggregate_median_ratio"]) >= DIVERSITY_POSITIVE_RATIO,
    }
    gates_a_to_d = all(int(gates[name]["passed"]) == 1 for name in ("gate_a", "gate_b", "gate_c", "gate_d"))
    gate_e_conditions = {
        "learned_human_recognizability_at_least_0_90": float(by_row["learned"]["human_recognizability"]) >= REVIEW_POSITIVE_RECOGNIZABILITY,
        "learned_human_requested_label_agreement_at_least_0_75": float(by_row["learned"]["human_requested_label_agreement"]) >= REVIEW_POSITIVE_AGREEMENT,
        "learned_classifier_accuracy_at_least_0_80": learned_classifier >= CLASSIFIER_POSITIVE_ACCURACY,
        "learned_zero_duplicate_pairs": int(learned_metrics["duplicates"]["duplicate_pair_count"]) == 0,
        "learned_diversity_ratio_at_least_0_25": float(learned_metrics["diversity"]["aggregate_median_ratio"]) >= DIVERSITY_POSITIVE_RATIO,
        "learned_human_agreement_exceeds_null": human_exceeds_null,
        "learned_classifier_accuracy_exceeds_null": learned_classifier > null_classifier,
        "gates_a_to_d_passed": gates_a_to_d,
    }
    expected_gate_e = {
        "gate_type": "diagnostic threshold",
        "state": "complete",
        "passed": int(all(gate_e_conditions.values())),
        "conditions": {key: int(value) for key, value in gate_e_conditions.items()},
        "values": {
            "learned_human_recognizability": by_row["learned"]["human_recognizability"],
            "learned_human_requested_label_agreement": by_row["learned"]["human_requested_label_agreement"],
            "null_human_requested_label_agreement": by_row["null"]["human_requested_label_agreement"],
            "learned_classifier_accuracy": learned_classifier,
            "null_classifier_accuracy": null_classifier,
            "learned_duplicate_pair_count": int(learned_metrics["duplicates"]["duplicate_pair_count"]),
            "learned_diversity_ratio": float(learned_metrics["diversity"]["aggregate_median_ratio"]),
        },
    }
    _require(gates["gate_e"] == expected_gate_e, "complete gate_e replay changed")
    classifier_positive = all(classifier_components.values())
    noncollapse_positive = all(noncollapse_components.values())
    route = route_outcome(
        gates_a_to_d_passed=gates_a_to_d,
        learned_human_positive=learned_human_positive,
        learned_classifier_positive=classifier_positive,
        learned_noncollapse_positive=noncollapse_positive,
        learned_exceeds_null=human_exceeds_null,
    )
    expected_outcome = {
        "schema": VERSION + "-outcome",
        "research_mode": RESEARCH_MODE,
        "state": "complete",
        "route": route,
        "gates_a_to_d_passed": int(gates_a_to_d),
        "human_marker": {
            "passed": int(learned_human_positive and human_exceeds_null),
            "learned": by_row["learned"],
            "null": by_row["null"],
            "learned_agreement_exceeds_null": int(human_exceeds_null),
        },
        "classifier_marker": {
            "passed": int(classifier_positive),
            "components": {key: int(value) for key, value in classifier_components.items()},
            "learned_accuracy": learned_classifier,
            "null_accuracy": null_classifier,
            "learned_minus_null": learned_classifier - null_classifier,
        },
        "noncollapse_marker": {
            "passed": int(noncollapse_positive),
            "components": {key: int(value) for key, value in noncollapse_components.items()},
        },
        "full_scale_auto_launched": 0,
        "next_action": _next_action(route),
    }
    outcome = _read_json(run_dir / "outcome.json")
    _require(outcome == expected_outcome, "prespecified outcome replay changed")
    _require(status["route"] == route, "complete status and outcome route disagree")
    return outcome


def _verify_report_contract(run_dir: Path, status: Mapping[str, Any]) -> None:
    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    handoff = (run_dir / "HANDOFF.md").read_text(encoding="utf-8")
    gates = _read_json(run_dir / "gates.json") if (run_dir / "gates.json").is_file() else None
    teacher = (
        _read_json(run_dir / "controls" / "teacher_gate.json")
        if (run_dir / "controls" / "teacher_gate.json").is_file()
        else None
    )
    outcome = _read_json(run_dir / "outcome.json") if (run_dir / "outcome.json").is_file() else None
    evidence_paths = (
        "input_bindings/input_authority_observations.json",
        "controls/teacher_gate.json",
        "controls/teacher_gate_arrays.npz",
        "evaluation/learned_metrics.json",
        "evaluation/predictions.npz",
        "review/human_review_by_row.json",
        "review/human_review_answers.csv",
        "populations/POPULATIONS_SEALED.json",
        "gates.json",
        "outcome.json",
    )
    for text, name in ((report, "REPORT.md"), (handoff, "HANDOFF.md")):
        _require("factor-one" in text.lower(), f"{name} omits factor-one scope")
        _require("candidate selection or replacement" in text.lower(), f"{name} omits selection scope")
        _require("Proxy-only patches since the last objective-bearing experiment: 0" in text, f"{name} proxy counter changed")
        _require(K128_TREE_DIGEST in text, f"{name} omits K128 evidence binding")
        _require("automatic DSM" in text and "No larger population" in text, f"{name} automatic-follow-up boundary changed")
        _require(str(status.get("state")) in text and str(status.get("route")) in text, f"{name} terminal status changed")
        _require(all(relative in text for relative in evidence_paths), f"{name} evidence map changed")
        if gates is not None:
            for gate_name in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e"):
                gate = gates[gate_name]
                label = "Gate " + gate_name.removeprefix("gate_").upper()
                summary = (
                    f"- {label}: state `{gate.get('state', 'complete')}`, "
                    f"passed `{gate.get('passed')}` ({gate.get('gate_type')})."
                )
                _require(summary in text, f"{name} {label} summary changed")
            for key, value in gates["gate_e"].get("values", {}).items():
                _require(
                    f"  - {key}: `{value}`" in text,
                    f"{name} Gate E report scalar changed: {key}",
                )
        if teacher is not None:
            for key in (
                "median_relative_squared_l2_anchor64",
                "median_relative_squared_l2_endpoint",
                "endpoint_improved_path_count",
                "teacher_requested_label_accuracy",
            ):
                _require(f"- {key}: `{teacher[key]}`" in text, f"{name} teacher report scalar changed: {key}")
        if outcome is not None:
            route = str(outcome["route"])
            _require(f"- Route: `{route}`" in text, f"{name} outcome route changed")
            _require(
                f"- This result establishes: {_route_scoped_claim(route)}" in text,
                f"{name} scoped outcome claim changed",
            )
            _require(f"- Next action: {outcome['next_action']}" in text, f"{name} outcome next action changed")


def _verify_lifecycle_artifacts(
    run_dir: Path,
    status: Mapping[str, Any],
    completed: Sequence[str],
    failure_stage: str | None,
) -> None:
    required_top = {
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "status.json",
        "config.json",
        "source_bindings.json",
        "resource_ledger.json",
        "stage_ledger.json",
        "command.txt",
        "claim_boundary.json",
        "deterministic_execution.json",
        "REPORT.md",
        "HANDOFF.md",
    }
    _require(all((run_dir / relative).is_file() for relative in required_top), "terminal run omits a required top-level authority")
    state = str(status["state"])
    human_paths = {
        "review/human_review_answers.csv",
        "review/human_review.json",
        "review/human_review_by_row.json",
        "outcome.json",
    }
    if state == "awaiting_human_review":
        _require(all(not (run_dir / relative).exists() for relative in human_paths), "awaiting-human-review route contains opened human evidence")
    elif state == "complete":
        _require(all((run_dir / relative).is_file() for relative in human_paths), "complete route omits human outcome evidence")
    else:
        _require((run_dir / "failure.json").is_file(), "failure route omits failure authority")
        if failure_stage != "human_review_terminalization":
            _require(all(not (run_dir / relative).exists() for relative in human_paths), "failure route contains later human outcome evidence")

    if failure_stage is None:
        return
    limit = STAGE_ORDER.index(failure_stage)

    def artifact_stage(relative: str) -> str | None:
        if relative in {
            "input_bindings/input_authority_observations.json",
            "input_bindings/predecessors.json",
            "input_bindings/ddpm_evaluator_binding.json",
            "input_bindings/selected_checkpoint.pt",
        }:
            return "initialize_and_bind"
        if relative in {
            "input_bindings/legacy_checkpoint_receipt.json",
            "input_bindings/clean_model_state.pt",
            "input_bindings/clean_model_state_receipt.json",
        }:
            return "checkpoint_extract"
        if relative == "data/development_roles.json" or relative == "input_bindings/mass_to_uint8.json" or relative.startswith("inventory/"):
            return "data_and_inventory"
        if relative.startswith("preflight/"):
            return "preflight"
        for row in ("teacher", "null", "learned"):
            if relative in {
                f"populations/{row}.npz",
                f"populations/partial_{row}.npz",
                f"telemetry/{row}_steps.csv",
                f"telemetry/partial_{row}_steps.csv",
                f"images/partial_{row}_latest.png",
            }:
                return f"{row}_row"
        if relative in {"telemetry/summary.json", "telemetry/model_state_identity.json"}:
            return "learned_row"
        if (
            relative == "populations/POPULATIONS_SEALED.json"
            or relative.endswith("_uint8.npz")
            or (relative.startswith("images/") and not Path(relative).name.startswith("partial_"))
        ):
            return "population_seal"
        if relative.startswith("evaluation/") or relative.startswith("controls/") or relative == "data/test_open_event.json" or relative == "gates.json":
            return "scoring"
        if relative.startswith("review/") and relative not in human_paths:
            return "review_prepare"
        if relative in human_paths:
            return "human_review_terminalization"
        return None

    for path in (item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        stage = artifact_stage(relative)
        if stage is not None:
            _require(STAGE_ORDER.index(stage) <= limit, f"artifact appears beyond failed stage: {relative}")


def _verify_telemetry_summary(run_dir: Path) -> None:
    rows: dict[str, Any] = {}
    for name in ("teacher", "null", "learned"):
        raw = _load_row_population(run_dir / "populations" / f"{name}.npz", expected_row=name)
        telemetry = list(raw["telemetry"].tolist())
        rows[name] = {
            "step_count": len(telemetry),
            "scientific_digest": str(raw["scientific_digest"][0]),
            "maximum_mass_error": max(float(entry["maximum_mass_error"]) for entry in telemetry),
            "maximum_clipping_fraction": max(float(entry["accepted_clipping_fraction"]) for entry in telemetry),
            "maximum_accepted_substeps": max(int(entry["accepted_substeps"]) for entry in telemetry),
        }
    _require(
        _read_json(run_dir / "telemetry" / "summary.json")
        == {"schema": VERSION + "-telemetry-summary", "rows": rows},
        "telemetry summary changed",
    )
    checkpoint_receipt = _read_json(run_dir / "input_bindings" / "legacy_checkpoint_receipt.json")
    model = _load_clean_model(
        run_dir / "input_bindings" / "clean_model_state.pt",
        config=DirectFluxMNISTConfig(**checkpoint_receipt["config"]),
        device="cpu",
    )
    digest = _model_state_semantic_digest(model)
    _require(
        _read_json(run_dir / "telemetry" / "model_state_identity.json")
        == {"before_sha256": digest, "after_sha256": digest, "identical": 1},
        "learned model-state identity changed",
    )


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    manifest = _verify_manifest_read_only(root)
    manifest_sha256 = sha256_file(root / "artifact_manifest.json")
    sums_sha256 = sha256_file(root / "SHA256SUMS.txt")
    status, completed, failure_stage = _verify_stage_and_status(root)
    _verify_lifecycle_artifacts(root, status, completed, failure_stage)
    resource = _verify_resource_ledger(root, status)
    if status["state"] in {"resource_stopped", "integrity_failed"} and (root / "failure.json").is_file():
        failure = _read_json(root / "failure.json")
        original_failed = failure["original_failed_admission"]
        if status["state"] == "resource_stopped":
            projection_path = root / "preflight" / "resource_projection.json"
            projection_failed = projection_path.is_file() and int(_read_json(projection_path).get("passed", 1)) == 0
            terminal_conversion = "original_error_type" in failure
            terminal_failed = resource["failed_admission"]
            if terminal_conversion:
                _require(
                    original_failed is None
                    and type(terminal_failed) is dict
                    and terminal_failed.get("kind") == "failure_terminalization"
                    and terminal_failed.get("phase") == "post-completion",
                    "terminal resource conversion has no authenticated post-completion stop",
                )
                expected_message = (
                    "resource post-completion check failed for failure_terminalization: "
                    f"{terminal_failed['checks']}"
                )
                _require(failure["message"] == expected_message, "terminal resource conversion message changed")
            _require(
                (type(original_failed) is dict and int(original_failed.get("passed", 1)) == 0)
                or (original_failed is None and projection_failed)
                or terminal_conversion,
                "resource failure omits its original stop authority",
            )
            if original_failed is None and not terminal_conversion:
                _require(_read_json(root / "failure.json")["message"] == _read_json(projection_path)["stop_reason"], "projected resource-stop message changed")
            elif resource["failed_admission"] is not None and resource["failed_admission"].get("kind") != "failure_terminalization":
                _require(original_failed == resource["failed_admission"], "resource failure original admission changed")
        else:
            _require(original_failed is None, "integrity failure contains an earlier resource stop")
    config = _verify_config_and_sources(
        root,
        resource,
        allow_input_mismatch=(
            status["state"] in {"resource_stopped", "integrity_failed"}
            and failure_stage in {"initialize_and_bind", "checkpoint_extract", "data_and_inventory"}
        ),
    )
    if config:
        _verify_restart_history(root, config)
    _verify_report_contract(root, status)

    if "initialize_and_bind" in completed or (root / "input_bindings" / "predecessors.json").is_file():
        _verify_predecessor_bindings(root, config)
    scientific_config: DirectFluxMNISTConfig | None = None
    if "checkpoint_extract" in completed or (root / "input_bindings" / "legacy_checkpoint_receipt.json").is_file():
        scientific_config = _verify_checkpoint_extract(root, config)
    if "data_and_inventory" in completed or (root / "inventory" / "START_BANK_SEALED.json").is_file():
        if config:
            _require(scientific_config is not None, "data stage has no checkpoint configuration")
            _verify_data_and_inventory(root, config, scientific_config)

    if "preflight" in completed or (root / "preflight" / "resource_projection.json").is_file():
        _verify_preflight(root, resource)
    elif (root / "preflight" / "synthetic_teacher.json").is_file():
        _verify_synthetic_preflight_report(root)

    for row in ("teacher", "null", "learned"):
        stage = f"{row}_row"
        full_path = root / "populations" / f"{row}.npz"
        partial_path = root / "populations" / f"partial_{row}.npz"
        if stage in completed:
            _verify_one_row_result(root, row, partial=False)
            _require(not partial_path.exists(), f"completed {row} row retains a partial population")
        elif full_path.is_file():
            _require(failure_stage == stage, f"uncompleted {row} row has a final population")
            _verify_one_row_result(root, row, partial=False)
            raw = _verifier_npz(
                full_path,
                {
                    "anchors",
                    "anchor_steps",
                    "labels",
                    "path_ids",
                    "root_seed",
                    "config_sha256",
                    "checkpoint_sha256",
                    "scientific_digest",
                    "telemetry_json",
                },
            )
            _verify_sheet_pixels(
                root / "images" / f"partial_{row}_latest.png",
                mass_to_uint8(raw["anchors"][-1], _read_json(root / "input_bindings" / "mass_to_uint8.json")),
                columns=16,
                scale=2,
                captions=None,
            )
        if partial_path.is_file():
            _require(failure_stage == stage, f"partial {row} row is outside its failed stage")
            _verify_one_row_result(root, row, partial=True)
            image_path = root / "images" / f"partial_{row}_latest.png"
            raw = _verifier_npz(
                partial_path,
                {
                    "anchors",
                    "anchor_steps",
                    "labels",
                    "path_ids",
                    "root_seed",
                    "config_sha256",
                    "checkpoint_sha256",
                    "scientific_digest",
                    "telemetry_json",
                },
            )
            _verify_sheet_pixels(
                image_path,
                mass_to_uint8(raw["anchors"][-1], _read_json(root / "input_bindings" / "mass_to_uint8.json")),
                columns=16,
                scale=2,
                captions=None,
            )
    if "learned_row" in completed:
        _verify_telemetry_summary(root)

    population_seal: dict[str, Any] | None = None
    if "population_seal" in completed or (root / "populations" / "POPULATIONS_SEALED.json").is_file():
        population_seal = _verify_population_seal(root)

    evaluation: dict[str, Any] | None = None
    scoring_ready = root / "evaluation" / "SCORING_READY.json"
    if "scoring" in completed or scoring_ready.is_file():
        _require(population_seal is not None, "scoring has no verified population seal")
        if scoring_ready.is_file():
            evaluation = _verify_evaluation(root, population_seal)

    review: dict[str, Any] | None = None
    if "review_prepare" in completed or (root / "review" / "READY.json").is_file():
        _require(population_seal is not None, "review has no verified population seal")
        review = _verify_review_bundle(root, population_seal)

    gates: dict[str, Any] | None = None
    if (root / "gates.json").is_file():
        _require(population_seal is not None and evaluation is not None, "gates have no verified scoring evidence")
        gates = _verify_machine_gates(root, resource, population_seal, complete=status["state"] == "complete")

    if status["state"] == "awaiting_human_review":
        _require(population_seal is not None and evaluation is not None and review is not None and gates is not None, "awaiting-human-review route is incomplete")
        _require(all(int(gates[name]["passed"]) == 1 for name in ("gate_a", "gate_b", "gate_c", "gate_d")), "awaiting-human-review route has a failed execution gate")
    elif status["state"] == "complete":
        _require(population_seal is not None and evaluation is not None and review is not None and gates is not None, "complete route is missing machine evidence")
        human = _replay_human_review(root, review)
        _verify_complete_outcome(root, status, gates, human, evaluation)
    else:
        _require(not (root / "outcome.json").exists(), "failure route has a scientific outcome")

    _require(_manifest_rows(root) == manifest["files"], "verification mutated the run tree")
    _require(sha256_file(root / "artifact_manifest.json") == manifest_sha256, "verification mutated the manifest")
    _require(sha256_file(root / "SHA256SUMS.txt") == sums_sha256, "verification mutated the checksum receipt")
    return {
        "schema": VERSION + "-verification-receipt",
        "passed": 1,
        "state": status["state"],
        "route": status["route"],
        "artifact_count": manifest["artifact_count"],
        "artifact_bytes": manifest["artifact_bytes"],
        "tree_digest": manifest["tree_digest"],
        "read_only": 1,
    }


def run_production(args: argparse.Namespace) -> int:
    repository_root = _repository_root()
    run_dir = Path(args.run_dir).resolve()
    device = torch.device(str(args.device))
    _require(str(device) == "cuda:0", "production device must be exactly cuda:0")
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA production device is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    approval_id = str(args.approval_id).strip()
    _require(
        len(approval_id) >= 12
        and "placeholder" not in approval_id.lower()
        and "<" not in approval_id
        and ">" not in approval_id,
        "a fresh non-placeholder approval ID is required",
    )
    budget = ResourceBudget(
        max_active_seconds=float(args.max_active_seconds),
        max_storage_bytes=int(float(args.max_storage_mib) * 1024 * 1024),
        max_cuda_fraction=float(args.max_cuda_fraction),
        reserve_seconds=TERMINAL_RESERVE_SECONDS,
        maximum_quantum_seconds=MAX_QUANTUM_SECONDS,
    )
    fresh_run = not run_dir.exists()
    reentry_mode: str | None = None
    if fresh_run:
        run_dir.mkdir(parents=True)
        for relative in (
            "input_bindings",
            "inventory",
            "preflight",
            "populations",
            "telemetry",
            "images",
            "controls",
            "data",
            "evaluation",
            "review",
        ):
            (run_dir / relative).mkdir(parents=True, exist_ok=True)
        governor = ResourceGovernor(run_dir, budget, device=device)
        governor.write()
        _write_json(run_dir / "stage_ledger.json", {"schema": VERSION + "-stage-ledger", "events": []})
        config = _initial_config(args, repository_root=repository_root, budget=budget)
        _write_json(run_dir / "config.json", config)
        _atomic_bytes(run_dir / "command.txt", (config["command"] + "\n").encode("utf-8"))
        _write_json(
            run_dir / "claim_boundary.json",
            {
                "schema": VERSION + "-claim-boundary",
                "research_mode": RESEARCH_MODE,
                "decision": "Does the pinned global edge-flux checkpoint produce factor-one task-visible MNIST under the current sampler and historical low-frequency source law?",
                "positive_scope": "exploratory compatibility of one checkpoint/source/sampler/transform tuple",
                "not_claimed": [
                    "exact Doob h-transform or reference-prior correctness",
                    "confirmatory generator quality",
                    "DDPM superiority or stochastic pairing across rows",
                    "general Eulerian-model feasibility or failure",
                ],
                "proxy_only_patches_since_last_objective_bearing_experiment": 0,
            },
        )
        _reset_running_status(run_dir)
        _write_json(
            run_dir / "deterministic_execution.json",
            {
                "schema": VERSION + "-deterministic-execution",
                "torch_deterministic_algorithms": int(torch.are_deterministic_algorithms_enabled()),
                "cudnn_benchmark": int(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": int(torch.backends.cudnn.deterministic),
                "tf32_policy_changed_by_runner": 0,
            },
        )
    else:
        reentry_mode = _classify_reentry(run_dir, args)
        if reentry_mode == "verify_only":
            receipt = verify_run(run_dir)
            print(json.dumps(receipt, sort_keys=True, allow_nan=False))
            return 0 if receipt["state"] in {"awaiting_human_review", "complete"} else 3
        (
            governor,
            config,
            scientific_config,
            development_images,
            development_labels,
            inventory,
            starts,
            targets,
        ) = _validate_reentry_base(run_dir, args, budget=budget, device=device)
        old_status = _read_json(run_dir / "status.json")
        _verify_population_seal(run_dir) if reentry_mode == "continue_sealed" else None

        # A crash after a terminal status write is recovered only by closing the
        # authenticated terminal quantum and resealing; no generation or scoring is
        # reachable from this branch.
        completed_stages = [str(event.get("stage")) for event in _stage_events(run_dir)]
        if reentry_mode == "continue_sealed" and old_status.get("state") in {
            "awaiting_human_review",
            "complete",
        } and (
            "machine_terminalization" in completed_stages
            or "human_review_terminalization" in completed_stages
        ):
            terminal_state = str(old_status["state"])
            population_seal = _verify_population_seal(run_dir)
            for row in ("teacher", "null", "learned"):
                _verify_one_row_result(run_dir, row, partial=False)
            _verify_telemetry_summary(run_dir)
            evaluation = _verify_evaluation(run_dir, population_seal)
            review = _verify_review_bundle(run_dir, population_seal)
            gates = _verify_machine_gates(
                run_dir,
                _read_json(run_dir / "resource_ledger.json"),
                population_seal,
                complete=terminal_state == "complete",
            )
            expected_terminal_stage = (
                "human_review_terminalization"
                if terminal_state == "complete"
                else "machine_terminalization"
            )
            recommit_human_terminal = False
            if terminal_state == "complete":
                human = _replay_human_review(run_dir, review)
                _verify_complete_outcome(
                    run_dir,
                    old_status,
                    gates,
                    human,
                    evaluation,
                )
                human_terminal_events = [
                    event
                    for event in governor.events
                    if event.get("kind") == "human_review_terminalization"
                    and event.get("event")
                    in {"admit", "complete", "failed-complete", "interrupted-close"}
                ]
                _require(human_terminal_events, "human terminal recovery has no resource authority")
                recommit_human_terminal = (
                    human_terminal_events[-1].get("event") == "interrupted-close"
                )
            try:
                _record_restart_authority(run_dir, mode=reentry_mode)
                if recommit_human_terminal:
                    governor.admit(
                        "human_review_terminalization",
                        predicted_seconds=10.0,
                        predicted_next_bytes=2 * 1024 * 1024,
                        reserve_remaining_seconds=0.0,
                    )
                if expected_terminal_stage not in completed_stages:
                    _record_stage(run_dir, expected_terminal_stage)
                _write_json(run_dir / "status.json", old_status)
                outcome = _read_json(run_dir / "outcome.json") if terminal_state == "complete" else None
                _write_reports(run_dir, outcome)
                _seal_manifest(run_dir)
                if recommit_human_terminal:
                    governor.complete("human_review_terminalization")
                _write_reports(run_dir, outcome)
                manifest = _seal_manifest(run_dir)
                receipt = verify_run(run_dir)
                print(json.dumps({**receipt, "artifact_count": manifest["artifact_count"]}, sort_keys=True, allow_nan=False))
                return 0
            except BaseException as raw_error:
                error = (
                    raw_error
                    if isinstance(raw_error, (IntegrityFailure, ResourceStop))
                    else IntegrityFailure(
                        f"operational terminal recovery failure "
                        f"{type(raw_error).__name__}: {raw_error}"
                    )
                )
                if terminal_state == "complete":
                    gates_payload = _read_json(run_dir / "gates.json")
                    gates_payload["gate_e"] = {
                        "gate_type": "diagnostic threshold",
                        "state": "pending",
                        "passed": None,
                        "conditions": {},
                    }
                    _write_json(run_dir / "gates.json", gates_payload)
                    (run_dir / "outcome.json").unlink(missing_ok=True)
                result = _finalize_failure(
                    run_dir,
                    error,
                    governor=governor,
                    failed_stage=expected_terminal_stage,
                )
                print(json.dumps(result, sort_keys=True, allow_nan=False))
                return 3 if isinstance(error, ResourceStop) else 4

        _record_restart_authority(run_dir, mode=reentry_mode)
        if reentry_mode == "rerun_all_rows":
            _clear_unsealed_outputs(run_dir)
        _reset_running_status(run_dir)
        clean_path = run_dir / "input_bindings" / "clean_model_state.pt"

    current_stage = "initialize_and_bind" if fresh_run else ("scoring" if reentry_mode == "continue_sealed" else "preflight")
    current_row: str | None = None
    try:
        if not fresh_run:
            return _resume_existing_production(
                run_dir,
                mode=str(reentry_mode),
                governor=governor,
                scientific_config=scientific_config,
                development_images=development_images,
                development_labels=development_labels,
                inventory=inventory,
                starts=starts,
                targets=targets,
                clean_path=clean_path,
                arff_path=Path(args.arff).resolve(),
                device=device,
            )
        governor.admit(
            "initialize_and_bind",
            predicted_seconds=30.0,
            predicted_next_bytes=5 * 1024 * 1024,
        )
        _bind_external_authorities(
            run_dir,
            repository_root=repository_root,
            legacy_checkpoint=Path(args.legacy_checkpoint).resolve(),
            arff=Path(args.arff).resolve(),
            k128_run_dir=Path(args.k128_run_dir).resolve(),
            ddpm_run_dir=Path(args.ddpm_run_dir).resolve(),
        )
        governor.complete("initialize_and_bind")
        _record_stage(run_dir, "initialize_and_bind")

        current_stage = "checkpoint_extract"
        governor.admit(
            "checkpoint_extract",
            predicted_seconds=30.0,
            predicted_next_bytes=16 * 1024 * 1024,
        )
        clean_path = run_dir / "input_bindings" / "clean_model_state.pt"
        checkpoint_receipt = safe_extract_legacy_checkpoint(
            Path(args.legacy_checkpoint).resolve(),
            clean_path,
        )
        _write_json(run_dir / "input_bindings" / "legacy_checkpoint_receipt.json", checkpoint_receipt)
        _write_json(
            run_dir / "input_bindings" / "clean_model_state_receipt.json",
            {
                "schema": VERSION + "-clean-model-state-receipt",
                "path": "input_bindings/clean_model_state.pt",
                "bytes": checkpoint_receipt["clean_state_bytes"],
                "sha256": checkpoint_receipt["clean_state_sha256"],
                "tensor_count": checkpoint_receipt["tensor_count"],
                "parameter_count": checkpoint_receipt["parameter_count"],
                "strict_reload_passed": 1,
            },
        )
        governor.complete("checkpoint_extract")
        _record_stage(run_dir, "checkpoint_extract")
        scientific_config = DirectFluxMNISTConfig(**checkpoint_receipt["config"])
        _require(int(scientific_config.grid_size) == 28, "legacy checkpoint grid size is not 28")

        current_stage = "data_and_inventory"
        governor.admit(
            "data_and_inventory",
            predicted_seconds=30.0,
            predicted_next_bytes=20 * 1024 * 1024,
        )
        development_images, development_labels, data_audit = read_mnist_development_prefix(
            Path(args.arff).resolve()
        )
        inventory, starts, targets, _ = _write_inventory_authorities(
            run_dir,
            config=scientific_config,
            development_images=development_images,
            development_labels=development_labels,
            data_audit=data_audit,
        )
        governor.complete("data_and_inventory")
        _record_stage(run_dir, "data_and_inventory")

        current_stage = "preflight"
        governor.admit(
            "cpu_preflight",
            predicted_seconds=30.0,
            predicted_next_bytes=1 * 1024 * 1024,
        )
        synthetic = _synthetic_teacher_preflight(
            run_dir,
            starts=starts,
            labels=inventory["requested_labels"],
            targets=targets["masses"],
            config=scientific_config,
        )
        governor.complete("cpu_preflight")
        governor.admit(
            "model_load_to_device",
            predicted_seconds=20.0,
            predicted_next_bytes=1 * 1024 * 1024,
        )
        model = _load_clean_model(clean_path, config=scientific_config, device=device)
        governor.complete("model_load_to_device")
        preflight = _run_device_preflight(
            run_dir,
            governor=governor,
            starts=starts,
            labels=inventory["requested_labels"],
            targets=targets["masses"],
            config=scientific_config,
            model=model,
            device=device,
        )
        _record_stage(run_dir, "preflight")

        row_results: dict[str, RowResult] = {}
        for row in ("teacher", "null", "learned"):
            current_row = row
            current_stage = f"{row}_row"
            result = _execute_full_row(
                run_dir,
                governor=governor,
                row=row,
                starts=starts,
                labels=inventory["requested_labels"],
                path_ids=inventory["path_ids"],
                targets=targets["masses"],
                config=scientific_config,
                model=model,
                device=device,
                predicted_eight_step_seconds=float(preflight["row_quantum_seconds"][row]),
            )
            row_results[row] = result
            _record_stage(run_dir, current_stage)
            current_row = None
        _write_json(
            run_dir / "telemetry" / "summary.json",
            {
                "schema": VERSION + "-telemetry-summary",
                "rows": {
                    name: {
                        "step_count": len(result.telemetry),
                        "scientific_digest": result.scientific_digest,
                        "maximum_mass_error": max(float(entry["maximum_mass_error"]) for entry in result.telemetry),
                        "maximum_clipping_fraction": max(float(entry["accepted_clipping_fraction"]) for entry in result.telemetry),
                        "maximum_accepted_substeps": max(int(entry["accepted_substeps"]) for entry in result.telemetry),
                    }
                    for name, result in row_results.items()
                },
            },
        )

        current_stage = "population_seal"
        governor.admit(
            "population_seal_and_scoring",
            predicted_seconds=30.0,
            predicted_next_bytes=25 * 1024 * 1024,
        )
        seal_populations(run_dir)
        _record_stage(run_dir, "population_seal")
        current_stage = "scoring"
        evaluation = evaluate_sealed_populations(
            run_dir,
            arff_path=Path(args.arff).resolve(),
            device=device,
            development_images=development_images,
            development_labels=development_labels,
        )
        _record_stage(run_dir, "scoring")
        governor.complete("population_seal_and_scoring")
        gates = _machine_gates(
            run_dir,
            governor=governor,
            teacher_control=evaluation["teacher_control"],
        )
        if int(gates["gate_d"]["passed"]) != 1:
            raise IntegrityFailure("full-interface target-informed positive control failed Gate D")

        current_stage = "review_prepare"
        governor.admit(
            "review_prepare",
            predicted_seconds=10.0,
            predicted_next_bytes=15 * 1024 * 1024,
        )
        prepare_blind_review(run_dir)
        governor.complete("review_prepare")
        _record_stage(run_dir, "review_prepare")

        current_stage = "machine_terminalization"
        governor.admit(
            "machine_terminalization",
            predicted_seconds=5.0,
            predicted_next_bytes=2 * 1024 * 1024,
            reserve_remaining_seconds=0.0,
        )
        _machine_gates(
            run_dir,
            governor=governor,
            teacher_control=evaluation["teacher_control"],
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema": VERSION + "-status",
                "state": "awaiting_human_review",
                "route": "awaiting_human_review",
                "error": None,
                "updated_at": _utc_now(),
                "whole_run_restart_required": 0,
            },
        )
        _record_stage(run_dir, "machine_terminalization")
        _write_reports(run_dir, None)
        _seal_manifest(run_dir)
        governor.complete("machine_terminalization")
        _write_reports(run_dir, None)
        manifest = _seal_manifest(run_dir)
        receipt = verify_run(run_dir)
        output = {
            "passed": int(receipt["passed"]),
            "state": "awaiting_human_review",
            "artifact_count": manifest["artifact_count"],
            "tree_digest": manifest["tree_digest"],
        }
        print(json.dumps(output, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as raw_error:
        error: BaseException
        if isinstance(raw_error, (IntegrityFailure, ResourceStop)):
            error = raw_error
        else:
            error = IntegrityFailure(f"operational failure {type(raw_error).__name__}: {raw_error}")
        partial = getattr(raw_error, "partial_row_result", None)
        if isinstance(partial, RowResult):
            _persist_partial_failure(run_dir, partial)
        durable_full = getattr(raw_error, "durable_full_row_result", None)
        if isinstance(durable_full, RowResult):
            _persist_durable_full_failure_image(run_dir, durable_full)
        result = _finalize_failure(
            run_dir,
            error,
            governor=governor,
            failed_stage=current_stage,
            partial_row=current_row,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 3 if isinstance(error, ResourceStop) else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the fixed factor-one machine phase")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--legacy-checkpoint", required=True)
    run.add_argument("--ddpm-run-dir", required=True)
    run.add_argument("--k128-run-dir", required=True)
    run.add_argument("--arff", required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--approval-id", required=True)
    run.add_argument("--max-active-seconds", type=float, default=MAX_ACTIVE_SECONDS)
    run.add_argument("--max-storage-mib", type=float, default=MAX_STORAGE_MIB)
    run.add_argument("--max-cuda-fraction", type=float, default=MAX_CUDA_FRACTION)
    review = commands.add_parser("record-review", help="record the fixed blind review")
    review.add_argument("--run-dir", required=True)
    review.add_argument("--answers", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--confirm-manual-review", action="store_true")
    verify = commands.add_parser("verify", help="verify a run tree without mutation")
    verify.add_argument("--run-dir", required=True)
    commands.add_parser(
        "smoke",
        help="run the bounded test-only synthetic CPU composition smoke",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(run_cpu_smoke(), sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run":
        return run_production(args)
    if args.command == "record-review":
        result = record_review(args.run_dir, args.answers, reviewer=args.reviewer,
                               confirm_manual_review=args.confirm_manual_review)
    else:
        result = verify_run(args.run_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if args.command == "record-review":
        state = str(result.get("state", ""))
        if state == "resource_stopped":
            return 3
        if state == "integrity_failed":
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
