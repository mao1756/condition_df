"""Lean exploratory complete-path runner for the candidate Jacobi/RB law.

This is Stage D/E infrastructure, not a confirmatory generator.  It deliberately
reuses the low-level fused scheduler and candidate CUDA proposal while avoiding the
exact-continuation state machine and exact authorizer.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_global_dilated import (
    GLOBAL_DILATED_PARAMETER_COUNT,
    GlobalDilatedZeroBaselinePredictor,
    global_dilated_architecture_contract,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    ModelInputs,
    enable_deterministic_torch,
    semantic_sha256,
    state_dict_sha256,
)
from mnist.d0_jacobi_rb_tangent_fused import (
    CANDIDATE_REFERENCE_CONTRACT,
    CandidateApproximateFusedReference,
    FUSED_SHARD_PHASES,
    FusedRowSpec,
    FusedTangentControllerBank,
    prepare_deferred_reference_rng_seed_map,
    run_fused_reverse_family,
)
from mnist.d0_jacobi_rb_tangent_rollout import (
    ScaledTangentScoreController,
    TargetFractionOracleController,
    atomic_rollout_npz,
    fixed_rendering_scale,
    load_rollout_state_npz,
    load_verified_source_target,
    paired_metric_improvement,
    raw_state_metrics,
    render_background_demixed,
    render_raw_density,
    render_source_image,
    reverse_suffix_sequence,
    rollout_array_sha256,
    rollout_file_sha256,
    rollout_semantic_record,
    save_png,
)

VERSION = "d0-jacobi-rb-candidate-complete-v1"
LEGACY_ROW_ORDER = ("zero", "global-plus-1", "source-informed")
SCHEDULE_ROW_ORDER = (
    "zero",
    "global-plus-1",
    "global-cutoff-176",
    "global-cutoff-216",
    "source-informed",
)
STAGE_E_CUTOFF_ROW_ORDER = (
    "zero", "global-plus-1", "global-cutoff-216", "source-informed",
)
STATE_SIZE = 784
OUTER_STEPS = 512
MICROSTEPS = 2
SHARD_STEPS = 8
SHARD_COUNT = 64
PER_ROW_SHARD_TRANSITIONS = FUSED_SHARD_PHASES * 2 * MICROSTEPS * EDGES_PER_PHASE
MASS_TOLERANCE = 2.0e-12
STORAGE_CAP_BYTES = 2 * 1024**3
TERMINAL_STORAGE_RESERVE_BYTES = 64 * 1024**2
CUDA_FRACTION_CAP = 0.80
TERMINAL_RESERVE_SECONDS = 300.0
BOOTSTRAP_SHARD_SECONDS = 300.0
RENDER_HORIZONS = (0, 8, 16, 128, 256, 384, 512)
SCHEDULE_RENDER_HORIZONS = (0, 8, 16, 128, 176, 192, 216, 224, 256, 384, 512)
STAGE_E_CUTOFF_RENDER_HORIZONS = (0, 8, 16, 128, 216, 224, 256, 384, 512)
CHECKPOINT_SHA256 = "5831a950a979726bf7a648d4c276bdc13f032f17ad1bc739c5d73c25d4841d38"
CHECKPOINT_STATE_SHA256 = "1df9888bef6c63db10f41f89a58891321e058e55ed7d8b36622c9cdf9827a218"
SOURCE_JSON_SHA256 = "e4f6918a6bd9b01f36ebdebdcf262242dfa714e908af199bde47cb9e025591eb"
SOURCE_NPZ_SHA256 = "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
SOURCE_ARRAY_SHA256 = "9d1f95ff3487901dabcd9b8261e0241511d2191997cd9daa260e390a3bc26c96"
TARGET_ARRAY_SHA256 = "14513cf5aa1aceda2bcff9befdd685297040ea33a3a773809e2b03d9401d5fc8"
ANCHOR_SHA256 = "64f577f93d33d0f8986271d2a55a87b025d9afce21b3fa75a2cd4edb9bc6c280"
ANCHOR_STATE_SHA256 = "16290bb00837c4def108f10615aa4555fe942583d9ee7c6902b128923ce3d0c9"
def _core_copy_specs() -> dict[str, tuple[str, str, str]]:
    """Return the current frozen input authorities without caching overrides."""
    return {
        "checkpoint": ("inputs/checkpoint.pt", "inputs/model/update-3100.pt", CHECKPOINT_SHA256),
        "source_json": ("inputs/source/source_image.json", "inputs/source/source_image.json", SOURCE_JSON_SHA256),
        "source_npz": ("inputs/source/source_image.npz", "inputs/source/source_image.npz", SOURCE_NPZ_SHA256),
        "anchor": ("inputs/anchor-step-0511.npz", "forward/anchor-step-0511.npz", ANCHOR_SHA256),
    }
EXACT_PREFIX_HASHES = {
    "shard-0000.json": "96cc9d196652e0942c552f2c8c50c67b6e5a7c4f5059884727674d41a78f80a6",
    "shard-0000.npz": "34079822f4da88bf9bc269f38ac0ae65fbe16a28c4c09e2b56b3773ec6a5bbd9",
    "shard-0001.json": "f89f899f97c82da2c6facf215dc9b2d0aaabae4fdced387cdabb1f101e60271d",
    "shard-0001.npz": "b4d5415d48118da8cbaaaf562cc827705db03548930bf644ea9d36390083bee3",
}
EXACT_PREFIX_RELATIVE = Path("reverse/fused_families/same-path-three-row/complete-512")
SCHEDULE_BASELINE_HASHES = {
    "artifact_manifest.json": "1ad190d10e04270af0b0a8587c3333ce0fef54b1c2f0e6befa6e9a0fac88c086",
    "config.json": "12e11e1721efd718d2ac4580e5b6c2d38b3cfd68cbe7f85402d563abccaf2c08",
    "outcome.json": "70b1b64d32656841c76408874c4561c595d977b52b7a6e1b477f13bccb61732b",
    "reverse/trajectory_boundaries.npz": "4067cbbc7e97b09416c7891076c82fcfa19a53c11a8f8b160883f93b47b77b8a",
    "reverse/first16_audit.json": "994e3539e2666c9ac95f1591aa776433d31211a5201026b6defdf05bc74ed551",
}
SCHEDULE_BASELINE_COPY_ROOT = Path("inputs/stage_d_schedule_baseline")
SCHEDULE_BASELINE_COPY_NAMES = {
    relative: Path(relative).name for relative in SCHEDULE_BASELINE_HASHES
}
STAGE_E_SCHEDULE_PREDECESSOR_HASHES = {
    "artifact_manifest.json": "385c6b82e9fa219ab32096437c62194144c95ae818754e052e85ba4b30bdc94f",
    "config.json": "693bd8d599261f39fea61a2bea981193f5077c6add13c5c4d1a4d57f9e422537",
    "bindings.json": "d9706eb66c09267ae4e2ac1f345cb78b2bd9078cc0fd3de19332fef40923d268",
    "outcome.json": "d7fe1cbbd52dc57a40a6be3d9202335d986f2d45a553d666aa18ded4f1d61db1",
    "reverse/health.json": "2270598922131b7580c06af8db5833e98c095e5942626c3b7c3cdd6d85d60d0a",
    "reverse/first16_audit.json": "1406c9d27f25a9d01e4c82f1488fd6e1d33c2c4c627282b4025614ee434cd478",
}
STAGE_E_SCHEDULE_PREDECESSOR_COPY_ROOT = Path("inputs/stage_d_schedule_predecessor")
STAGE_E_SCHEDULE_PREDECESSOR_COPY_NAMES = {
    relative: Path(relative).name for relative in STAGE_E_SCHEDULE_PREDECESSOR_HASHES
}
STAGE_D_APPROVAL_REFERENCE_CAVEAT = "<fresh-approval-reference>"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_COUNTS = {
    "resource_cap_count", "invalid_density_count", "clipping_count",
    "correction_count", "floor_count", "limiter_count", "projection_count",
    "renormalization_count", "nonfinite_count",
}
_DIRECT_SOURCE_FILES = (
    "mnist/diag_d0_jacobi_rb_candidate_complete.py",
    "mnist/d0_jacobi_rb_cuda.py",
    "mnist/d0_jacobi_rb_cuda_deferred.py",
    "mnist/d0_jacobi_rb_tangent_fused.py",
    "mnist/d0_jacobi_rb_boundary_tangent_fused.py",
    "mnist/d0_jacobi_rb_reverse_controller.py",
    "mnist/d0_jacobi_rb_tangent_rollout.py",
    "mnist/d0_jacobi_rb_global_dilated.py",
    "mnist/d0_jacobi_rb_boundary_tangent.py",
    "mnist/d0_jacobi_rb_boundary_tangent_frequency1_coordinate.py",
    "mnist/d0_jacobi_rb_boundary_tangent_zero_baseline.py",
    "mnist/d0_jacobi_rb_coarse_residual.py",
    "mnist/d0_jacobi_rb_absolute_coordinate.py",
    "mnist/d0_jacobi_rb_learnability.py",
    "mnist/d0_jacobi_artifacts.py",
)
_CONTROLLER_FORBIDDEN_COUNTS = (
    "clipping_count", "floor_count", "projection_count", "nonfinite_score_count",
)


class CandidateRunError(RuntimeError):
    pass


class ResourcePause(CandidateRunError):
    def __init__(self, projection: Mapping[str, Any]) -> None:
        self.projection = dict(projection)
        super().__init__("candidate resource projection exceeds the approved cap")


class CompletedStepCutoffTangentScoreController(ScaledTangentScoreController):
    """Fixed completed-step prefix schedule for the two exploratory rows."""

    def __init__(self, base_controller: Any, cutoff_completed_reverse_steps: int) -> None:
        if isinstance(cutoff_completed_reverse_steps, bool) or cutoff_completed_reverse_steps not in {176, 216}:
            raise CandidateRunError("cutoff must be one of the two frozen completed-step boundaries")
        super().__init__(base_controller, gain=1.0)
        self.cutoff_completed_reverse_steps = int(cutoff_completed_reverse_steps)

    def _active_mask(self, inputs: ModelInputs, dtype: torch.dtype) -> torch.Tensor:
        if type(inputs) is not ModelInputs:
            raise TypeError("cutoff controller requires exact ModelInputs")
        return (
            inputs.reverse_time < self.cutoff_completed_reverse_steps / OUTER_STEPS
        ).reshape(inputs.batch_size, 1).to(dtype=dtype)

    def score_prediction(self, inputs: ModelInputs) -> torch.Tensor:
        score = super().score_prediction(inputs)
        return score * self._active_mask(inputs, score.dtype)

    def score_prediction_deferred(self, inputs: ModelInputs) -> torch.Tensor:
        score = super().score_prediction_deferred(inputs)
        mask = self._active_mask(inputs, score.dtype)
        unscaled = self._last_deferred_unscaled
        self._last_deferred_unscaled = unscaled * mask.to(dtype=unscaled.dtype)
        return score * mask


@dataclass(frozen=True)
class StartStateSpec:
    kind: Literal["forward_anchor", "dirichlet_prior"]
    seed: int | None
    expected_state_sha256: str | None

    def to_record(self) -> dict[str, Any]:
        return {"kind": self.kind, "seed": self.seed, "expected_state_sha256": self.expected_state_sha256}


@dataclass(frozen=True)
class CandidateExperimentSpec:
    name: Literal[
        "stage-d-anchor-v1", "stage-e-prior-v1", "stage-d-schedule-window-v1",
        "stage-e-prior-cutoff-216-v1",
    ]
    start: StartStateSpec
    path_id: int
    root_seed: int
    stream_role: str
    render_horizons: tuple[int, ...] = RENDER_HORIZONS
    label: int = 3
    outer_steps: int = OUTER_STEPS
    phase_count: int = 7
    microsteps: int = MICROSTEPS
    candidate_modes: int = 128
    candidate_bisection_steps: int = 56

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name, "start": self.start.to_record(), "path_id": self.path_id,
            "root_seed": self.root_seed, "stream_role": self.stream_role,
            "render_horizons": list(self.render_horizons), "label": self.label,
            "outer_steps": self.outer_steps, "phase_count": self.phase_count,
            "microsteps": self.microsteps, "candidate_modes": self.candidate_modes,
            "candidate_bisection_steps": self.candidate_bisection_steps,
        }


def experiment_spec(name: str) -> CandidateExperimentSpec:
    if name == "stage-d-anchor-v1":
        return CandidateExperimentSpec(
            name="stage-d-anchor-v1",
            start=StartStateSpec("forward_anchor", None, ANCHOR_STATE_SHA256),
            path_id=1_028_864,
            root_seed=261_402,
            stream_role="global_dilated_positive_complete_exact",
        )
    if name == "stage-e-prior-v1":
        return CandidateExperimentSpec(
            name="stage-e-prior-v1",
            start=StartStateSpec("dirichlet_prior", 261_403, None),
            path_id=1_028_865,
            root_seed=261_404,
            stream_role="global_dilated_positive_prior_candidate",
        )
    if name == "stage-e-prior-cutoff-216-v1":
        return CandidateExperimentSpec(
            name="stage-e-prior-cutoff-216-v1",
            start=StartStateSpec("dirichlet_prior", 261_403, None),
            path_id=1_028_865,
            root_seed=261_404,
            stream_role="global_dilated_positive_prior_candidate",
            render_horizons=STAGE_E_CUTOFF_RENDER_HORIZONS,
        )
    if name == "stage-d-schedule-window-v1":
        return CandidateExperimentSpec(
            name="stage-d-schedule-window-v1",
            start=StartStateSpec("forward_anchor", None, ANCHOR_STATE_SHA256),
            path_id=1_028_864,
            root_seed=261_402,
            stream_role="global_dilated_positive_complete_exact",
            render_horizons=SCHEDULE_RENDER_HORIZONS,
        )
    raise CandidateRunError("unknown named experiment specification")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateRunError(f"JSON object required: {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    except Exception as exc:
        raise CandidateRunError(f"NPZ cannot be opened: {path}") from exc


def _reject_run_links(root: Path) -> None:
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path); attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
                if entry.is_symlink() or attributes & 0x400:
                    raise CandidateRunError(f"run artifact is linked or reparse-backed: {path.relative_to(root)}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif not entry.is_file(follow_symlinks=False) or path.stat().st_nlink != 1:
                    raise CandidateRunError(f"run artifact is not an independent regular file: {path.relative_to(root)}")


def _copy_file(source: Path, destination: Path, expected_hash: str | None = None) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise CandidateRunError(f"input is absent or linked: {source}")
    digest = file_fingerprint(source)
    if expected_hash is not None and digest != expected_hash:
        raise CandidateRunError(f"input hash changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor); temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary); os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if destination.is_symlink() or destination.stat().st_nlink != 1 or file_fingerprint(destination) != digest:
        raise CandidateRunError(f"independent input copy failed: {destination}")
    return {"size": destination.stat().st_size, "sha256": digest}


def _module_bindings(repository_root: Path) -> dict[str, Any]:
    rows = {name: file_fingerprint(repository_root / name) for name in _DIRECT_SOURCE_FILES}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"], cwd=repository_root, check=True,
            capture_output=True, text=False,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", b""
    return {
        "repository_revision": revision,
        "dirty_diff_sha256": hashlib.sha256(dirty).hexdigest() if dirty else None,
        "direct_source_files": rows,
    }


def _row_specs(spec: CandidateExperimentSpec) -> tuple[FusedRowSpec, ...]:
    legacy = (
        FusedRowSpec("zero", spec.path_id, "zero", "zero", "same-path-complete"),
        FusedRowSpec(
            "global-plus-1", spec.path_id, "learned", "global-dilated",
            "same-path-complete", gain=1.0,
            controller_binding={"checkpoint_state_sha256": CHECKPOINT_STATE_SHA256},
        ),
        FusedRowSpec(
            "source-informed", spec.path_id, "oracle", "mixed-target-fraction",
            "same-path-complete", controller_binding={"target_sha256": TARGET_ARRAY_SHA256},
        ),
    )
    if spec.name not in {"stage-d-schedule-window-v1", "stage-e-prior-cutoff-216-v1"}:
        return legacy
    cutoff_176 = FusedRowSpec(
        "global-cutoff-176", spec.path_id, "learned",
        "global-dilated-cutoff-176", "same-path-complete", gain=1.0,
        controller_binding={
            "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
            "schedule_kind": "completed_reverse_step_prefix",
            "cutoff_completed_reverse_steps": 176,
            "active_predicate": "reverse_time_lt_cutoff_over_512",
            "active_outer_step_min_inclusive": 336,
            "inactive_outer_step_max_inclusive": 335,
        },
    )
    cutoff_216 = FusedRowSpec(
        "global-cutoff-216", spec.path_id, "learned",
        "global-dilated-cutoff-216", "same-path-complete", gain=1.0,
        controller_binding={
            "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
            "schedule_kind": "completed_reverse_step_prefix",
            "cutoff_completed_reverse_steps": 216,
            "active_predicate": "reverse_time_lt_cutoff_over_512",
            "active_outer_step_min_inclusive": 296,
            "inactive_outer_step_max_inclusive": 295,
        },
    )
    if spec.name == "stage-e-prior-cutoff-216-v1":
        return legacy[:2] + (cutoff_216,) + legacy[2:]
    return legacy[:2] + (cutoff_176, cutoff_216) + legacy[2:]


def _row_order(spec: CandidateExperimentSpec) -> tuple[str, ...]:
    if spec.name == "stage-e-prior-cutoff-216-v1":
        return STAGE_E_CUTOFF_ROW_ORDER
    return tuple(row.row_key for row in _row_specs(spec))


def _family_name(spec: CandidateExperimentSpec) -> str:
    if spec.name == "stage-d-schedule-window-v1":
        return "same-path-five-row"
    if spec.name == "stage-e-prior-cutoff-216-v1":
        return "same-path-four-row"
    return "same-path-three-row"


def _shard_relative(spec: CandidateExperimentSpec) -> Path:
    return Path("reverse/fused_families") / _family_name(spec) / "complete-512"


def _shard_transition_count(spec: CandidateExperimentSpec) -> int:
    return PER_ROW_SHARD_TRANSITIONS * len(_row_specs(spec))


def _controller_binding(spec: CandidateExperimentSpec) -> dict[str, Any]:
    return {
        "row_table": [row.to_record() for row in _row_specs(spec)],
        "global_state_sha256": CHECKPOINT_STATE_SHA256,
        "target_sha256": TARGET_ARRAY_SHA256,
        "dispatch": "stable_one_row_canonical_order",
        "model_input_contract": "exact_ModelInputs_six_fields",
    }


def _rng_binding(spec: CandidateExperimentSpec) -> dict[str, Any]:
    return {"root_seed": spec.root_seed, "stream_role": spec.stream_role, "canonical_path_id": spec.path_id}


def _validate_core_inputs(run_dir: Path) -> tuple[Any, np.ndarray]:
    expected = {target: digest for target, _source, digest in _core_copy_specs().values()}
    for relative, digest in expected.items():
        path = run_dir / relative
        if not path.is_file() or path.is_symlink() or file_fingerprint(path) != digest:
            raise CandidateRunError(f"core input changed: {relative}")
    source = load_verified_source_target(run_dir / "inputs/source")
    if rollout_array_sha256(source.source_image) != SOURCE_ARRAY_SHA256 or rollout_array_sha256(source.mixed_target) != TARGET_ARRAY_SHA256:
        raise CandidateRunError("source arrays changed")
    anchor = _npz(run_dir / "inputs/anchor-step-0511.npz")
    state = anchor.get("state")
    if set(anchor) != {"state"} or state is None or state.dtype != np.float64 or state.shape != (STATE_SIZE,) or rollout_array_sha256(state) != ANCHOR_STATE_SHA256:
        raise CandidateRunError("forward terminal anchor changed")
    return source, state


def _validate_start_state(run_dir: Path, config: Mapping[str, Any] | None = None) -> np.ndarray:
    frozen = _read_json(run_dir / "config.json") if config is None else dict(config)
    binding = _read_json(run_dir / "bindings.json")
    path = run_dir / "inputs/start_state.npz"; archive = _npz(path)
    state = archive.get("state")
    if (
        set(archive) != {"state"} or state is None or state.dtype != np.float64
        or state.shape != (STATE_SIZE,) or rollout_array_sha256(state) != frozen.get("start_state_sha256")
        or binding.get("start_state_sha256") != frozen.get("start_state_sha256")
        or binding.get("start_state_file_sha256") != rollout_file_sha256(path)
    ):
        raise CandidateRunError("start-state authority changed")
    spec = experiment_spec(str(frozen.get("experiment_name")))
    if spec.start.kind == "forward_anchor":
        if rollout_array_sha256(state) != ANCHOR_STATE_SHA256 or frozen.get("prior") is not None:
            raise CandidateRunError("Stage D start state is not the frozen anchor")
    else:
        expected, prior = _prior_state(int(spec.start.seed))
        if not np.array_equal(state, expected) or frozen.get("prior") != prior:
            raise CandidateRunError("Stage E prior construction changed")
    return state


def _load_model(run_dir: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(run_dir / "inputs/checkpoint.pt", map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or payload.get("state_sha256") != CHECKPOINT_STATE_SHA256 or state_dict_sha256(state) != CHECKPOINT_STATE_SHA256:
        raise CandidateRunError("checkpoint state authority changed")
    architecture = global_dilated_architecture_contract()
    if architecture.get("passed") != 1 or architecture.get("trainable_parameter_count") != GLOBAL_DILATED_PARAMETER_COUNT:
        raise CandidateRunError("global architecture contract changed")
    model = GlobalDilatedZeroBaselinePredictor(zero_residual=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def _prior_state(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    draws = np.ascontiguousarray(rng.gamma(shape=1.0, scale=1.0, size=STATE_SIZE), dtype=np.float64)
    total = float(np.sum(draws, dtype=np.float64)); state = np.ascontiguousarray(draws / total, dtype=np.float64)
    return state, {
        "law": "Dirichlet(1,...,1)", "construction": "PCG64-float64-Gamma(1,1)-then-one-initialization-normalization",
        "numpy_version": np.__version__, "bit_generator": "PCG64",
        "seed": int(seed), "normalization_is_initialization_not_dynamics": 1,
        "pre_normalization_sum": total, "sum": float(np.sum(state)), "minimum": float(np.min(state)),
        "maximum": float(np.max(state)), "state_sha256": rollout_array_sha256(state),
    }


def _validate_exact_prefix(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spec = experiment_spec("stage-d-anchor-v1"); row_table = [row.to_record() for row in _row_specs(spec)]
    sequence = reverse_suffix_sequence(511)
    previous: np.ndarray | None = None
    for index in range(2):
        json_path = source_root / f"shard-{index:04d}.json"; npz_path = source_root / f"shard-{index:04d}.npz"
        for path in (json_path, npz_path):
            if not path.is_file() or path.is_symlink() or file_fingerprint(path) != EXACT_PREFIX_HASHES[path.name]:
                raise CandidateRunError(f"exact prefix file incompatible: {path.name}")
        record = _read_json(json_path); state = load_rollout_state_npz(npz_path, expected_rows=3)
        shard_sequence = sequence[index * FUSED_SHARD_PHASES:(index + 1) * FUSED_SHARD_PHASES]
        if (
            record.get("shard_index") != index or record.get("row_keys") != list(LEGACY_ROW_ORDER)
            or record.get("row_table") != row_table or record.get("committed") != 1
            or record.get("controller_binding_sha256") != semantic_sha256(_controller_binding(spec))
            or record.get("rng_binding_sha256") != semantic_sha256(_rng_binding(spec))
            or record.get("sequence_start") != list(shard_sequence[0])
            or record.get("sequence_end") != list(shard_sequence[-1])
            or record.get("sequence_sha256") != semantic_sha256([list(item) for item in shard_sequence])
            or record.get("label") != spec.label or record.get("microsteps") != spec.microsteps
            or record.get("variant_in_rng_key") != 0
        ):
            raise CandidateRunError("exact prefix record incompatible")
        if record.get("output_state_sha256") != rollout_array_sha256(state) or record.get("state_file_sha256") != rollout_file_sha256(npz_path):
            raise CandidateRunError("exact prefix state binding incompatible")
        if previous is not None and record.get("input_state_sha256") != rollout_array_sha256(previous):
            raise CandidateRunError("exact prefix chain incompatible")
        previous = state
        rows.append({"index": index, "json": json_path, "npz": npz_path, "record": record, "state": state})
    return rows


def _validate_schedule_baseline_payload(root: Path, *, copied: bool) -> None:
    def selected(relative: str) -> Path:
        return root / (SCHEDULE_BASELINE_COPY_NAMES[relative] if copied else relative)

    for relative, digest in SCHEDULE_BASELINE_HASHES.items():
        path = selected(relative)
        if (
            not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
            or file_fingerprint(path) != digest
        ):
            raise CandidateRunError(f"Stage D schedule baseline authority changed: {relative}")
    baseline = experiment_spec("stage-d-anchor-v1")
    config = _read_json(selected("config.json"))
    outcome = _read_json(selected("outcome.json"))
    final = outcome.get("primary_final_metrics")
    if (
        config.get("experiment_name") != baseline.name
        or config.get("spec") != baseline.to_record()
        or config.get("row_order") != list(LEGACY_ROW_ORDER)
        or config.get("row_table") != [row.to_record() for row in _row_specs(baseline)]
        or config.get("controller_binding") != _controller_binding(baseline)
        or config.get("rng_binding") != _rng_binding(baseline)
        or config.get("start_state_sha256") != ANCHOR_STATE_SHA256
        or config.get("prior") is not None
        or outcome.get("run_state") != "complete"
        or outcome.get("scientific_objective_completed") != 1
        or outcome.get("completed_reverse_steps") != OUTER_STEPS
        or outcome.get("health_passed") != 1
        or outcome.get("oracle_directionally_improves_zero") != 1
        or outcome.get("learned_directionally_improves_zero") != 0
        or not isinstance(final, Mapping)
        or float(final.get("source-informed", {}).get("paired_squared_l2_improvement_over_zero", 0.0)) <= 0.0
        or float(final.get("global-plus-1", {}).get("paired_squared_l2_improvement_over_zero", 0.0)) >= 0.0
    ):
        raise CandidateRunError("Stage D schedule baseline semantics changed")
    trajectory = _npz(selected("reverse/trajectory_boundaries.npz"))
    boundaries = np.arange(0, OUTER_STEPS + SHARD_STEPS, SHARD_STEPS, dtype=np.int64)
    states = trajectory.get("states")
    if (
        set(trajectory) != {"completed_reverse_steps", "states"}
        or not np.array_equal(trajectory["completed_reverse_steps"], boundaries)
        or states is None or states.dtype != np.float64
        or states.shape != (SHARD_COUNT + 1, len(LEGACY_ROW_ORDER), STATE_SIZE)
        or not states.flags.c_contiguous
    ):
        raise CandidateRunError("Stage D schedule baseline trajectory changed")


def _validate_schedule_baseline_source(source_root: Path) -> None:
    if not source_root.is_dir():
        raise CandidateRunError("Stage D schedule baseline directory is absent")
    _verify_manifest(source_root)
    if file_fingerprint(source_root / "artifact_manifest.json") != SCHEDULE_BASELINE_HASHES["artifact_manifest.json"]:
        raise CandidateRunError("Stage D schedule baseline manifest authority changed")
    _validate_schedule_baseline_payload(source_root, copied=False)
    health = _read_json(source_root / "reverse/health.json")
    binding = _read_json(source_root / "bindings.json")
    if (
        health.get("passed") != 1
        or binding.get("source_image_sha256") != SOURCE_ARRAY_SHA256
        or binding.get("mixed_target_sha256") != TARGET_ARRAY_SHA256
        or binding.get("start_state_sha256") != ANCHOR_STATE_SHA256
    ):
        raise CandidateRunError("Stage D schedule baseline evidence binding changed")


def _copy_schedule_baseline(source_root: Path, run_dir: Path) -> dict[str, Any]:
    _validate_schedule_baseline_source(source_root)
    files = []
    for relative, digest in SCHEDULE_BASELINE_HASHES.items():
        destination_relative = SCHEDULE_BASELINE_COPY_ROOT / SCHEDULE_BASELINE_COPY_NAMES[relative]
        files.append({
            "source_path": relative,
            "path": destination_relative.as_posix(),
            **_copy_file(source_root / relative, run_dir / destination_relative, digest),
        })
    return {
        "run_dir": str(source_root),
        "fixed_authority_sha256": dict(SCHEDULE_BASELINE_HASHES),
        "files": files,
    }


def _validate_schedule_baseline_copy(run_dir: Path, authority: Any) -> np.ndarray:
    if not isinstance(authority, Mapping):
        raise CandidateRunError("schedule run lost its Stage D baseline binding")
    if authority.get("fixed_authority_sha256") != SCHEDULE_BASELINE_HASHES:
        raise CandidateRunError("Stage D schedule baseline fixed authorities changed")
    claims = authority.get("files")
    expected_paths = {
        (SCHEDULE_BASELINE_COPY_ROOT / name).as_posix()
        for name in SCHEDULE_BASELINE_COPY_NAMES.values()
    }
    if (
        not isinstance(claims, list)
        or not all(isinstance(claim, Mapping) for claim in claims)
        or {claim.get("path") for claim in claims} != expected_paths
        or {claim.get("source_path") for claim in claims} != set(SCHEDULE_BASELINE_HASHES)
    ):
        raise CandidateRunError("Stage D schedule baseline copy inventory changed")
    for claim in claims:
        path = run_dir / str(claim["path"])
        source_path = str(claim["source_path"])
        if (
            claim.get("sha256") != SCHEDULE_BASELINE_HASHES[source_path]
            or not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
            or claim.get("size") != path.stat().st_size
            or file_fingerprint(path) != claim.get("sha256")
        ):
            raise CandidateRunError("Stage D schedule baseline copied authority changed")
    copy_root = run_dir / SCHEDULE_BASELINE_COPY_ROOT
    _validate_schedule_baseline_payload(copy_root, copied=True)
    return _npz(copy_root / SCHEDULE_BASELINE_COPY_NAMES["reverse/trajectory_boundaries.npz"])["states"]


def _validate_stage_e_predecessor_payload(root: Path, *, copied: bool) -> None:
    def selected(relative: str) -> Path:
        name = STAGE_E_SCHEDULE_PREDECESSOR_COPY_NAMES[relative] if copied else relative
        return root / name

    for relative, digest in STAGE_E_SCHEDULE_PREDECESSOR_HASHES.items():
        path = selected(relative)
        if (
            not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
            or file_fingerprint(path) != digest
        ):
            raise CandidateRunError(f"Stage D schedule predecessor authority changed: {relative}")
    spec = experiment_spec("stage-d-schedule-window-v1")
    config = _read_json(selected("config.json"))
    bindings = _read_json(selected("bindings.json"))
    outcome = _read_json(selected("outcome.json"))
    health = _read_json(selected("reverse/health.json"))
    audit = _read_json(selected("reverse/first16_audit.json"))
    if (
        config.get("experiment_name") != spec.name
        or config.get("spec") != spec.to_record()
        or config.get("row_order") != list(SCHEDULE_ROW_ORDER)
        or config.get("row_table") != [row.to_record() for row in _row_specs(spec)]
        or config.get("controller_binding") != _controller_binding(spec)
        or config.get("rng_binding") != _rng_binding(spec)
        or config.get("start_state_sha256") != ANCHOR_STATE_SHA256
        or bindings.get("source_image_sha256") != SOURCE_ARRAY_SHA256
        or bindings.get("mixed_target_sha256") != TARGET_ARRAY_SHA256
        or bindings.get("start_state_sha256") != ANCHOR_STATE_SHA256
        or outcome.get("run_state") != "complete"
        or outcome.get("scientific_objective_completed") != 1
        or outcome.get("completed_reverse_steps") != OUTER_STEPS
        or outcome.get("health_passed") != 1
        or outcome.get("schedule_identity_passed") != 1
        or outcome.get("oracle_directionally_improves_zero") != 1
        or outcome.get("selected_cutoff_completed_reverse_steps") != 216
        or outcome.get("selected_cutoff_ready_for_review") != 1
        or health.get("passed") != 1
        or health.get("committed_shards") != SHARD_COUNT
        or health.get("completed_reverse_steps") != OUTER_STEPS
        or health.get("transition_count") != PER_ROW_SHARD_TRANSITIONS * 5 * SHARD_COUNT
        or health.get("schedule_identity", {}).get("passed") != 1
        or audit.get("status") != "available"
        or audit.get("audit_complete") != 1
        or audit.get("first16_not_completely_off") != 1
    ):
        raise CandidateRunError("Stage D schedule predecessor semantics changed")


def _validate_stage_e_predecessor_source(source_root: Path) -> None:
    if not source_root.is_dir():
        raise CandidateRunError("Stage D schedule predecessor directory is absent")
    _verify_manifest(source_root)
    _validate_stage_e_predecessor_payload(source_root, copied=False)
    ledger = _read_json(source_root / "resource_ledger.json")
    history = ledger.get("cap_history")
    if not isinstance(history, list) or not any(
        isinstance(row, Mapping)
        and row.get("approval_reference") == STAGE_D_APPROVAL_REFERENCE_CAVEAT
        for row in history
    ):
        raise CandidateRunError("Stage D schedule predecessor approval caveat changed")


def _copy_stage_e_predecessor(source_root: Path, run_dir: Path) -> dict[str, Any]:
    _validate_stage_e_predecessor_source(source_root)
    files = []
    for relative, digest in STAGE_E_SCHEDULE_PREDECESSOR_HASHES.items():
        destination_relative = (
            STAGE_E_SCHEDULE_PREDECESSOR_COPY_ROOT
            / STAGE_E_SCHEDULE_PREDECESSOR_COPY_NAMES[relative]
        )
        files.append({
            "source_path": relative,
            "path": destination_relative.as_posix(),
            **_copy_file(source_root / relative, run_dir / destination_relative, digest),
        })
    return {
        "run_dir_at_initialization": str(source_root),
        "fixed_authority_sha256": dict(STAGE_E_SCHEDULE_PREDECESSOR_HASHES),
        "files": files,
        "external_predecessor_required_after_initialization": 0,
        "predecessor_approval_reference_caveat": STAGE_D_APPROVAL_REFERENCE_CAVEAT,
        "predecessor_approval_caveat_is_not_child_compute_approval": 1,
    }


def _validate_stage_e_predecessor_copy(run_dir: Path, authority: Any) -> None:
    if not isinstance(authority, Mapping):
        raise CandidateRunError("Stage E lost its Stage D schedule predecessor binding")
    if (
        authority.get("fixed_authority_sha256") != STAGE_E_SCHEDULE_PREDECESSOR_HASHES
        or authority.get("external_predecessor_required_after_initialization") != 0
        or authority.get("predecessor_approval_reference_caveat") != STAGE_D_APPROVAL_REFERENCE_CAVEAT
        or authority.get("predecessor_approval_caveat_is_not_child_compute_approval") != 1
    ):
        raise CandidateRunError("Stage E schedule predecessor binding changed")
    claims = authority.get("files")
    expected_paths = {
        (STAGE_E_SCHEDULE_PREDECESSOR_COPY_ROOT / name).as_posix()
        for name in STAGE_E_SCHEDULE_PREDECESSOR_COPY_NAMES.values()
    }
    if (
        not isinstance(claims, list)
        or not all(isinstance(claim, Mapping) for claim in claims)
        or {claim.get("path") for claim in claims} != expected_paths
        or {claim.get("source_path") for claim in claims}
        != set(STAGE_E_SCHEDULE_PREDECESSOR_HASHES)
    ):
        raise CandidateRunError("Stage E schedule predecessor copy inventory changed")
    for claim in claims:
        source_path = str(claim["source_path"])
        path = run_dir / str(claim["path"])
        if (
            claim.get("sha256") != STAGE_E_SCHEDULE_PREDECESSOR_HASHES[source_path]
            or not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
            or claim.get("size") != path.stat().st_size
            or file_fingerprint(path) != claim.get("sha256")
        ):
            raise CandidateRunError("Stage E schedule predecessor copied authority changed")
    _validate_stage_e_predecessor_payload(
        run_dir / STAGE_E_SCHEDULE_PREDECESSOR_COPY_ROOT, copied=True,
    )


def _new_ledger(maximum: float, approval_reference: str) -> dict[str, Any]:
    approval = approval_reference.strip()
    if (
        not math.isfinite(maximum) or maximum <= TERMINAL_RESERVE_SECONDS
        or not approval or (approval.startswith("<") and approval.endswith(">"))
    ):
        raise CandidateRunError("maximum active seconds must exceed the terminal reserve")
    return {
        "schema": VERSION + "-resource-ledger", "maximum_active_seconds": float(maximum),
        "terminal_reserve_seconds": TERMINAL_RESERVE_SECONDS, "events": [],
        "active_attempt": None, "active_seconds": 0.0,
        "cap_history": [{
            "old": None, "new": float(maximum), "at": _utc_now(),
            "reason": "fresh_explicit_cap", "approval_reference": approval,
        }],
        "latest_projection": None,
    }


def _validate_ledger(run_dir: Path) -> dict[str, Any]:
    ledger = _read_json(run_dir / "resource_ledger.json")
    events, history = ledger.get("events"), ledger.get("cap_history")
    if (
        ledger.get("schema") != VERSION + "-resource-ledger"
        or ledger.get("terminal_reserve_seconds") != TERMINAL_RESERVE_SECONDS
        or not isinstance(events, list) or not isinstance(history, list) or not history
        or ledger.get("active_attempt") is not None and not isinstance(ledger.get("active_attempt"), Mapping)
    ):
        raise CandidateRunError("resource ledger schema changed")
    identifiers: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping) or set(event) != {"id", "role", "elapsed_seconds", "failed"}:
            raise CandidateRunError("resource event schema changed")
        identifier, elapsed = event["id"], float(event["elapsed_seconds"])
        if not isinstance(identifier, str) or not identifier or identifier in identifiers or not math.isfinite(elapsed) or elapsed < 0.0 or event["failed"] not in (0, 1):
            raise CandidateRunError("resource event authority changed")
        identifiers.add(identifier)
    active = ledger.get("active_attempt")
    if active is not None and (set(active) != {"id", "role", "started_at"} or active["id"] in identifiers):
        raise CandidateRunError("active resource attempt changed")
    total = math.fsum(float(event["elapsed_seconds"]) for event in events)
    if ledger.get("active_seconds") != total:
        raise CandidateRunError("resource elapsed sum changed")
    previous: float | None = None
    for index, row in enumerate(history):
        approval = row.get("approval_reference") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("reason"), str) or not row["reason"].strip()
            or not isinstance(approval, str) or not approval.strip()
            or (approval.strip().startswith("<") and approval.strip().endswith(">"))
        ):
            raise CandidateRunError("resource cap history changed")
        old, new = row.get("old"), float(row.get("new", math.nan))
        if not math.isfinite(new) or new <= TERMINAL_RESERVE_SECONDS or (index == 0 and old is not None) or (index and (old != previous or new <= float(previous))):
            raise CandidateRunError("resource cap amendment changed")
        previous = new
    if ledger.get("maximum_active_seconds") != previous:
        raise CandidateRunError("active-time cap changed")
    return ledger


def _begin_attempt(run_dir: Path, role: str) -> tuple[str, float]:
    ledger = _validate_ledger(run_dir)
    if ledger.get("active_attempt") is not None:
        raise CandidateRunError("resource attempt is already active")
    attempt_id = f"{role}-{len(ledger['events']) + 1}"
    ledger["active_attempt"] = {"id": attempt_id, "role": role, "started_at": _utc_now()}
    atomic_write_json(run_dir / "resource_ledger.json", ledger)
    return attempt_id, time.perf_counter()


def _finish_attempt(run_dir: Path, attempt_id: str, started: float, *, failed: bool) -> None:
    ledger = _validate_ledger(run_dir); active = ledger.get("active_attempt")
    if not isinstance(active, Mapping) or active.get("id") != attempt_id:
        raise CandidateRunError("resource attempt authority changed")
    elapsed = max(0.0, time.perf_counter() - started) + 5.0
    ledger["events"].append({"id": attempt_id, "role": active["role"], "elapsed_seconds": elapsed, "failed": int(failed)})
    ledger["active_attempt"] = None
    ledger["active_seconds"] = math.fsum(float(row["elapsed_seconds"]) for row in ledger["events"])
    atomic_write_json(run_dir / "resource_ledger.json", ledger)


def _reconcile_attempt(run_dir: Path) -> None:
    ledger = _validate_ledger(run_dir); active = ledger.get("active_attempt")
    if not isinstance(active, Mapping):
        return
    try:
        started = dt.datetime.fromisoformat(str(active["started_at"])); now = dt.datetime.now(dt.timezone.utc)
        elapsed = max(0.0, (now - started).total_seconds()) + 5.0
    except Exception as exc:
        raise CandidateRunError("active resource attempt is malformed") from exc
    ledger["events"].append({"id": active["id"], "role": str(active["role"]) + "_interrupted", "elapsed_seconds": elapsed, "failed": 1})
    ledger["active_attempt"] = None
    ledger["active_seconds"] = math.fsum(float(row["elapsed_seconds"]) for row in ledger["events"])
    atomic_write_json(run_dir / "resource_ledger.json", ledger)


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def _shard_root(run_dir: Path) -> Path:
    spec = experiment_spec(_read_json(run_dir / "config.json")["experiment_name"])
    return run_dir / _shard_relative(spec)


def _scan_prefix(run_dir: Path) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    config = _read_json(run_dir / "config.json"); spec = experiment_spec(config["experiment_name"])
    row_order = _row_order(spec)
    start = _validate_start_state(run_dir, config)
    previous = np.repeat(start[None, :], len(row_order), axis=0)
    records: list[dict[str, Any]] = []; states: list[np.ndarray] = [previous]
    sequence = reverse_suffix_sequence(511); root = _shard_root(run_dir)
    for index in range(SHARD_COUNT):
        record_path = root / f"shard-{index:04d}.json"; state_path = root / f"shard-{index:04d}.npz"
        if not record_path.exists():
            if any(
                (root / f"shard-{later:04d}{suffix}").exists()
                for later in range(index + 1, SHARD_COUNT)
                for suffix in (".json", ".npz", ".failure.json")
            ):
                raise CandidateRunError("candidate committed chain has a gap")
            break
        if (
            not record_path.is_file() or record_path.is_symlink() or record_path.stat().st_nlink != 1
            or not state_path.is_file() or state_path.is_symlink() or state_path.stat().st_nlink != 1
        ):
            raise CandidateRunError("committed candidate shard lacks NPZ")
        record = _read_json(record_path); body = {key: value for key, value in record.items() if key != "semantic_sha256"}
        expected_sequence = sequence[index * FUSED_SHARD_PHASES:(index + 1) * FUSED_SHARD_PHASES]
        if (
            rollout_semantic_record(body) != record or record.get("committed") != 1 or record.get("shard_index") != index
            or record.get("reference_contract") != CANDIDATE_REFERENCE_CONTRACT
            or record.get("schema") != "d0-jacobi-rb-tangent-fused-v1-reverse-shard"
            or record.get("scheduler_version") != "d0-jacobi-rb-tangent-fused-v1"
            or record.get("family_name") != _family_name(spec) or record.get("segment_name") != "complete-512"
            or record.get("row_table") != config["row_table"] or record.get("row_keys") != list(row_order)
            or record.get("canonical_path_ids") != [spec.path_id] * len(row_order)
            or record.get("sequence_start") != list(expected_sequence[0]) or record.get("sequence_end") != list(expected_sequence[-1])
            or record.get("sequence_sha256") != semantic_sha256([list(item) for item in expected_sequence])
            or record.get("microsteps") != spec.microsteps or record.get("label") != spec.label
            or record.get("variant_in_rng_key") != 0
            or record.get("input_state_sha256") != rollout_array_sha256(previous)
            or record.get("controller_binding_sha256") != semantic_sha256(config["controller_binding"])
            or record.get("rng_binding_sha256") != semantic_sha256(config["rng_binding"])
        ):
            raise CandidateRunError(f"candidate shard {index} binding changed")
        state = load_rollout_state_npz(state_path, expected_rows=len(row_order))
        if (
            record.get("state_file_sha256") != rollout_file_sha256(state_path)
            or record.get("state_file_size") != state_path.stat().st_size
            or record.get("output_state_sha256") != rollout_array_sha256(state)
        ):
            raise CandidateRunError(f"candidate shard {index} state changed")
        _validate_shard_health(record, state, spec)
        records.append(record); states.append(state); previous = state
    return records, states


def _validate_shard_health(record: Mapping[str, Any], state: np.ndarray, spec: CandidateExperimentSpec) -> None:
    diagnostics = record.get("diagnostics"); reference = diagnostics.get("reference") if isinstance(diagnostics, Mapping) else None
    forbidden = reference.get("forbidden_counts") if isinstance(reference, Mapping) else None
    active = int(reference.get("active_count", -1)) if isinstance(reference, Mapping) else -1
    noop = int(reference.get("structural_noop_count", -1)) if isinstance(reference, Mapping) else -1
    bracket = float(reference.get("maximum_candidate_bracket_width", math.inf)) if isinstance(reference, Mapping) else math.inf
    per_row = reference.get("per_row") if isinstance(reference, Mapping) else None
    row_order = _row_order(spec)
    shard_transitions = _shard_transition_count(spec)
    if (
        not isinstance(reference, Mapping) or reference.get("reference_contract") != CANDIDATE_REFERENCE_CONTRACT
        or reference.get("candidate_modes") != 128 or reference.get("candidate_bisection_steps") != 56
        or reference.get("root_seed") != spec.root_seed or reference.get("stream_role") != spec.stream_role
        or reference.get("variant_in_rng_key") != 0
        or reference.get("certificate_fraction") != "not_applicable" or int(reference.get("approximation_count", -2)) != active
        or int(record.get("transition_count", -1)) != shard_transitions
        or int(reference.get("transition_count", -1)) != shard_transitions
        or int(reference.get("transition_count", -1)) != active + noop or int(reference.get("invalid_count", -1)) != 0
        or not isinstance(per_row, list) or len(per_row) != len(row_order)
        or not isinstance(forbidden, Mapping) or set(forbidden) != _FORBIDDEN_COUNTS or any(int(value) for value in forbidden.values())
        or int(record.get("synchronous_replay_performed", -1)) != 0
        or active < 0 or noop < 0 or not math.isfinite(bracket) or bracket < 0.0
        or not np.isfinite(state).all() or np.any(state < 0.0) or np.any(state > 1.0)
        or float(np.max(np.abs(np.sum(state, axis=1) - 1.0))) > MASS_TOLERANCE
        or float(diagnostics.get("maximum_mass_error", math.inf)) > MASS_TOLERANCE
    ):
        raise CandidateRunError("candidate shard numerical/integrity health failed")
    row_totals = {name: 0 for name in ("transition_count", "active_count", "structural_noop_count", "approximation_count", "invalid_count")}
    for item in per_row:
        row_active = int(item.get("active_count", -1)) if isinstance(item, Mapping) else -1
        row_noop = int(item.get("structural_noop_count", -1)) if isinstance(item, Mapping) else -1
        if (
            not isinstance(item, Mapping)
            or int(item.get("transition_count", -1)) != PER_ROW_SHARD_TRANSITIONS
            or int(item.get("transition_count", -1)) != row_active + row_noop
            or row_active < 0 or row_noop < 0
            or int(item.get("approximation_count", -2)) != row_active
            or int(item.get("invalid_count", -1)) != 0
            or item.get("certificate_fraction") != "not_applicable"
            or not math.isfinite(float(item.get("maximum_candidate_bracket_width", math.inf)))
            or float(item.get("maximum_candidate_bracket_width", -1.0)) < 0.0
        ):
            raise CandidateRunError("candidate per-row numerical health failed")
        for name in row_totals:
            row_totals[name] += int(item[name])
    if any(row_totals[name] != int(reference[name]) for name in row_totals):
        raise CandidateRunError("candidate per-row totals do not match the shard aggregate")
    controller_rows = record.get("controller_diagnostics")
    dynamics_rows = record.get("per_row_diagnostics")
    expected_rows = _row_specs(spec)
    if (
        not isinstance(controller_rows, list) or len(controller_rows) != len(expected_rows)
        or not isinstance(dynamics_rows, list) or len(dynamics_rows) != len(expected_rows)
    ):
        raise CandidateRunError("candidate controller telemetry inventory changed")
    for observed, dynamics, expected in zip(controller_rows, dynamics_rows, expected_rows, strict=True):
        if (
            not isinstance(observed, Mapping)
            or not isinstance(dynamics, Mapping)
            or observed.get("row_key") != expected.row_key
            or dynamics.get("row_key") != expected.row_key
            or observed.get("controller_kind") != expected.controller_kind
            or observed.get("gain") != expected.gain
            or any(int(observed.get(name, -1)) != 0 for name in _CONTROLLER_FORBIDDEN_COUNTS)
        ):
            raise CandidateRunError("candidate controller telemetry failed")


def _archive_evidence(path: Path, archive: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise CandidateRunError("recovery evidence is not an independent regular file")
    digest = file_fingerprint(path); destination = archive / f"{label}-{digest[:12]}{path.suffix}"
    if destination.exists():
        if file_fingerprint(destination) != digest:
            raise CandidateRunError("recovery archive collision")
        path.unlink()
    else:
        os.replace(path, destination)


def _recover_uncommitted(run_dir: Path) -> None:
    root = _shard_root(run_dir); root.mkdir(parents=True, exist_ok=True)
    archive = run_dir / "failures"; archive.mkdir(parents=True, exist_ok=True)
    for committed_index in range(SHARD_COUNT):
        record = root / f"shard-{committed_index:04d}.json"; state = root / f"shard-{committed_index:04d}.npz"
        if not record.is_file() or not state.is_file():
            break
        stale_failure = root / f"shard-{committed_index:04d}.failure.json"
        if stale_failure.exists():
            _archive_evidence(stale_failure, archive, f"stale-failure-shard-{committed_index:04d}")
    records, _ = _scan_prefix(run_dir); index = len(records)
    json_path = root / f"shard-{index:04d}.json"; npz_path = root / f"shard-{index:04d}.npz"; failure = root / f"shard-{index:04d}.failure.json"
    if json_path.exists() and not npz_path.exists():
        raise CandidateRunError("JSON-without-NPZ candidate commit is invalid")
    if npz_path.exists() and not json_path.exists():
        _archive_evidence(npz_path, archive, f"orphan-shard-{index:04d}")
    if failure.exists():
        value = _read_json(failure)
        if value.get("failure_type") != "ResourcePause":
            raise CandidateRunError("non-resource shard failure is not resumable")
        _archive_evidence(failure, archive, f"resource-pause-shard-{index:04d}")


def _projection(run_dir: Path, attempt_started: float, committed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ledger = _validate_ledger(run_dir)
    active = float(ledger["active_seconds"]) + max(0.0, time.perf_counter() - attempt_started)
    observed = [float(row.get("elapsed_seconds", 0.0)) for row in committed]
    if len(observed) < 2:
        estimate = max(
            BOOTSTRAP_SHARD_SECONDS,
            1.2 * max(observed, default=0.0),
        )
        priced_shards = 2 - len(observed)
        method = "two-shard-bootstrap"
    else:
        estimate = 1.2 * max(observed)
        priced_shards = SHARD_COUNT - len(observed)
        method = "1.2-times-maximum-observed-candidate-shard"
    projected = active + priced_shards * estimate + TERMINAL_RESERVE_SECONDS
    result = {
        "at": _utc_now(), "committed_shards": len(observed), "active_seconds": active,
        "estimated_next_shard_seconds": estimate, "priced_remaining_shards": priced_shards,
        "method": method, "terminal_reserve_seconds": TERMINAL_RESERVE_SECONDS,
        "projected_total_seconds": projected, "maximum_active_seconds": float(ledger["maximum_active_seconds"]),
        "passed": int(projected <= float(ledger["maximum_active_seconds"])),
    }
    ledger["latest_projection"] = result; atomic_write_json(run_dir / "resource_ledger.json", ledger)
    return result


def _metric_rows(
    spec: CandidateExperimentSpec, boundaries: Sequence[int],
    states: Sequence[np.ndarray], source: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for completed, state in zip(boundaries, states, strict=True):
        zero = raw_state_metrics(state[0], source.mixed_target)
        for index, key in enumerate(_row_order(spec)):
            primary = raw_state_metrics(state[index], source.mixed_target); secondary = raw_state_metrics(state[index], source.source_image)
            separation = state[index] - state[0]
            demixed = (state[index] - float(source.metadata["lambda_mix"]) / STATE_SIZE) / (1.0 - float(source.metadata["lambda_mix"]))
            display = demixed / float(np.max(source.source_image))
            improvement = {"paired_squared_l2_improvement_over_zero": 0.0, "relative_paired_squared_l2_improvement_over_zero": 0.0} if index == 0 else paired_metric_improvement(primary, zero)
            rows.append({
                "completed_reverse_steps": int(completed), "row_key": key,
                **{f"mixed_target_{name}": value for name, value in primary.to_dict().items()},
                **{f"source_image_{name}": value for name, value in secondary.to_dict().items()},
                **improvement, "state_vs_zero_squared_l2": float(np.dot(separation, separation)),
                "state_vs_zero_l1": float(np.sum(np.abs(separation))),
                "state_nonfinite_count": int(np.count_nonzero(~np.isfinite(state[index]))),
                "state_negative_count": int(np.count_nonzero(state[index] < 0.0)),
                "boundary_mass": float(np.sum(state[index][state[index] <= 1.0e-15])),
                "demixed_source_squared_l2_before_clipping": float(np.dot(demixed - source.source_image, demixed - source.source_image)),
                "demixed_source_l1_before_clipping": float(np.sum(np.abs(demixed - source.source_image))),
                "below_display_fraction_before_clipping": float(np.mean(display < 0.0)),
                "below_display_magnitude_before_clipping": float(np.sum(np.maximum(-display, 0.0))),
                "above_display_fraction_before_clipping": float(np.mean(display > 1.0)),
                "above_display_magnitude_before_clipping": float(np.sum(np.maximum(display - 1.0, 0.0))),
                "render_clipping_count": int(np.count_nonzero((display < 0.0) | (display > 1.0))),
            })
    return rows


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, str]:
    a = left - np.mean(left); b = right - np.mean(right); denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1.0e-30:
        return (1.0 if np.array_equal(left, right) else 0.0), "near_constant_exact_equality"
    return float(np.dot(a, b) / denominator), "standard_centered"


def _first16_audit(run_dir: Path, states: Sequence[np.ndarray], source: Any) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    spec = experiment_spec(config["experiment_name"])
    row_order = _row_order(spec)
    binding = _read_json(run_dir / "bindings.json"); authority = binding.get("exact_prefix")
    if not isinstance(authority, Mapping) or authority.get("status") != "available":
        return {"schema": VERSION + "-first16-audit", "status": "unavailable", "reason": authority.get("reason", "not_stage_d") if isinstance(authority, Mapping) else "not_bound", "first16_not_completely_off": None, "blocks_artifact_completion": 0}
    exact_rows = _validate_exact_prefix(run_dir / "inputs/exact_prefix")
    candidate_records, _ = _scan_prefix(run_dir)
    reached = min(2, len(candidate_records), len(states) - 1)
    if reached < 1:
        raise CandidateRunError("candidate prefix has not reached an exact-audit horizon")
    expected_candidate_start = rollout_array_sha256(states[0])
    expected_exact_start = rollout_array_sha256(np.repeat(states[0][0:1], len(LEGACY_ROW_ORDER), axis=0))
    results: list[dict[str, Any]] = []; learned_contrast_pass = True
    if spec.name == "stage-d-schedule-window-v1":
        row_mapping = (0, 1, 1, 1, 2)
        witness_fields = (
            "rng_binding_sha256", "sequence_start", "sequence_end", "sequence_sha256",
            "label", "microsteps", "variant_in_rng_key",
        )
    else:
        row_mapping = tuple(range(len(LEGACY_ROW_ORDER)))
        witness_fields = (
            "row_table", "row_keys", "controller_binding_sha256", "rng_binding_sha256",
            "sequence_start", "sequence_end", "sequence_sha256", "label", "microsteps",
            "variant_in_rng_key",
        )
    for index, completed in enumerate((8, 16)[:reached], start=1):
        exact_record = exact_rows[index - 1]["record"]; candidate_record = candidate_records[index - 1]
        if any(exact_record.get(name) != candidate_record.get(name) for name in witness_fields):
            raise CandidateRunError("candidate/exact shared-randomness witness changed")
        if index == 1 and (
            exact_record.get("input_state_sha256") != expected_exact_start
            or candidate_record.get("input_state_sha256") != expected_candidate_start
        ):
            raise CandidateRunError("candidate/exact audit does not share the frozen start state")
        exact = exact_rows[index - 1]["state"]
        candidate = states[index]
        row_records = []
        for row_index, (row_key, exact_index) in enumerate(zip(row_order, row_mapping, strict=True)):
            difference = candidate[row_index] - exact[exact_index]; correlation, convention = _correlation(candidate[row_index], exact[exact_index])
            row_record = {
                "row_key": row_key,
                "l1_discrepancy": float(np.sum(np.abs(difference))),
                "squared_l2_discrepancy": float(np.dot(difference, difference)),
                "maximum_absolute_discrepancy": float(np.max(np.abs(difference))),
                "total_variation_discrepancy": float(0.5 * np.sum(np.abs(difference))),
                "centered_correlation": correlation, "correlation_convention": convention,
                "row_not_completely_off": int(float(np.sum(np.abs(difference))) <= 0.02 and float(np.max(np.abs(difference))) <= 0.002 and correlation >= 0.999),
            }
            if spec.name == "stage-d-schedule-window-v1":
                row_record["exact_row_key"] = LEGACY_ROW_ORDER[exact_index]
            row_records.append(row_record)
        def contrast(row_index: int, exact_index: int) -> dict[str, float]:
            exact_contrast = exact[exact_index] - exact[0]
            candidate_contrast = candidate[row_index] - candidate[0]
            delta = candidate_contrast - exact_contrast
            exact_norm = float(np.linalg.norm(exact_contrast))
            candidate_norm = float(np.linalg.norm(candidate_contrast))
            discrepancy = float(np.linalg.norm(delta))
            return {
                "exact_l2": exact_norm, "candidate_l2": candidate_norm,
                "discrepancy_l2": discrepancy,
                "relative_error": discrepancy / max(exact_norm, candidate_norm, 1.0e-15),
            }
        learned_contrasts = {
            row_order[row_index]: contrast(row_index, row_mapping[row_index])
            for row_index in range(1, len(row_order) - 1)
        }
        oracle_contrast = contrast(len(row_order) - 1, 2)
        learned_contrast_pass &= all(value["relative_error"] <= 0.25 for value in learned_contrasts.values())
        horizon_record = {
            "completed_reverse_steps": completed, "rows": row_records,
            "same_randomness_witnesses": {name: candidate_record.get(name) for name in witness_fields},
            "exact_input_state_sha256": exact_record.get("input_state_sha256"),
            "candidate_input_state_sha256": candidate_record.get("input_state_sha256"),
            "learned_minus_zero_contrast": learned_contrasts["global-plus-1"],
            "oracle_minus_zero_contrast": oracle_contrast,
            "learned_minus_zero_contrast_relative_error": learned_contrasts["global-plus-1"]["relative_error"],
            "oracle_minus_zero_contrast_relative_error": oracle_contrast["relative_error"],
        }
        if spec.name == "stage-d-schedule-window-v1":
            horizon_record["learned_minus_zero_contrasts"] = learned_contrasts
        results.append(horizon_record)
    passed = learned_contrast_pass and all(row["row_not_completely_off"] for horizon in results for row in horizon["rows"])
    largest_discrepancy = max(row["squared_l2_discrepancy"] for horizon in results for row in horizon["rows"])
    endpoint_separation = float(np.dot(states[-1][1] - states[-1][0], states[-1][1] - states[-1][0]))
    return {
        "schema": VERSION + "-first16-audit", "status": "available", "horizons": results,
        "reached_horizons": [int(row["completed_reverse_steps"]) for row in results],
        "audit_complete": int(reached == 2),
        "thresholds": {"l1": 0.02, "maximum_absolute": 0.002, "centered_correlation": 0.999, "learned_contrast_relative_error": 0.25},
        "first16_not_completely_off": int(passed) if reached == 2 else (0 if not passed else None),
        "blocks_artifact_completion": 0,
        "largest_audited_squared_l2_discrepancy": largest_discrepancy,
        "endpoint_learned_zero_squared_l2_separation": endpoint_separation,
        "endpoint_effect_over_largest_discrepancy": endpoint_separation / largest_discrepancy if largest_discrepancy else None,
        "endpoint_effect_at_least_four_times_discrepancy": int(endpoint_separation >= 4.0 * largest_discrepancy),
        "effect_dominance_is_descriptive_not_a_gate": 1,
        "does_not_prove_full_path_equivalence": 1,
    }


def _contact_sheet(path: Path, cells: Sequence[np.ndarray]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CandidateRunError("Pillow is required for contact sheets") from exc
    canvas = Image.new("L", (28 * len(cells), 28), color=0)
    for index, cell in enumerate(cells):
        canvas.paste(Image.fromarray(np.asarray(cell, dtype=np.uint8), mode="L"), (28 * index, 0))
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path, format="PNG")


def _render(run_dir: Path, boundaries: Sequence[int], states: Sequence[np.ndarray], source: Any) -> None:
    shutil.rmtree(run_dir / "images", ignore_errors=True)
    spec = experiment_spec(_read_json(run_dir / "config.json")["experiment_name"])
    scale = fixed_rendering_scale(source.source_image, source.mixed_target, float(source.metadata["lambda_mix"]))
    selected = [value for value in spec.render_horizons if value in boundaries]
    if boundaries[-1] not in selected:
        selected.append(boundaries[-1])
    for completed in selected:
        state = states[boundaries.index(completed)]
        for kind, renderer in (("raw", render_raw_density), ("demixed", render_background_demixed)):
            cells = []
            for index, key in enumerate(_row_order(spec)):
                image = renderer(state[index], scale); cells.append(image)
                path = run_dir / f"images/individual/{kind}/{key}/step-{completed:03d}.png"; save_png(path, image)
            sheet = run_dir / f"images/contact_sheets/{kind}-step-{completed:03d}.png"; _contact_sheet(sheet, cells)
    for name, image in (("source", render_source_image(source.source_image, scale)), ("mixed-target", render_raw_density(source.mixed_target, scale))):
        save_png(run_dir / f"images/context/{name}.png", image)


def _health(
    records: Sequence[Mapping[str, Any]], states: Sequence[np.ndarray],
    schedule_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    references = [record["diagnostics"]["reference"] for record in records]
    forbidden = {name: sum(int(reference["forbidden_counts"][name]) for reference in references) for name in _FORBIDDEN_COUNTS}
    peak_cuda = max((int(reference.get("peak_cuda_memory_bytes", 0)) for reference in references), default=0)
    total_cuda = max((int(reference.get("total_cuda_memory_bytes", 0)) for reference in references), default=0)
    result = {
        "schema": VERSION + "-health",
        "passed": int(schedule_identity is None or schedule_identity.get("passed") == 1),
        "committed_shards": len(records),
        "completed_reverse_steps": len(records) * SHARD_STEPS,
        "transition_count": sum(int(record["transition_count"]) for record in records),
        "approximation_count": sum(int(reference["approximation_count"]) for reference in references),
        "invalid_count": sum(int(reference["invalid_count"]) for reference in references),
        "certificate_fraction": "not_applicable", "fallback_invocation_count": 0,
        "exact_authorizer_invocation_count": 0,
        "synchronous_replay_count": sum(int(record.get("synchronous_replay_performed", 0)) for record in records),
        "forbidden_counts": forbidden,
        "maximum_candidate_bracket_width": max((float(reference["maximum_candidate_bracket_width"]) for reference in references), default=0.0),
        "maximum_mass_error": max((float(record["diagnostics"]["maximum_mass_error"]) for record in records), default=0.0),
        "peak_cuda_memory_bytes": peak_cuda, "total_cuda_memory_bytes": total_cuda,
        "maximum_cuda_memory_fraction": peak_cuda / total_cuda if total_cuda else 0.0,
        "final_nonfinite_count": int(np.count_nonzero(~np.isfinite(states[-1]))),
        "final_negative_count": int(np.count_nonzero(states[-1] < 0.0)),
    }
    if schedule_identity is not None:
        result["schedule_identity"] = dict(schedule_identity)
    return result


def _schedule_identity(
    run_dir: Path, records: Sequence[Mapping[str, Any]], states: Sequence[np.ndarray],
) -> dict[str, Any]:
    spec = experiment_spec(_read_json(run_dir / "config.json")["experiment_name"])
    scheduled = {"stage-d-schedule-window-v1", "stage-e-prior-cutoff-216-v1"}
    if spec.name not in scheduled:
        return {"schema": VERSION + "-schedule-identity", "status": "not_applicable", "passed": 1}
    boundaries = [index * SHARD_STEPS for index in range(len(states))]
    telemetry_fields = (
        "score_squared_sum", "score_maximum_absolute",
        "unscaled_score_squared_sum", "unscaled_score_maximum_absolute",
    )
    logistic_fields = ("logistic_shift_squared_sum", "logistic_shift_maximum_absolute")

    def cutoff_identity(cutoff: int, row_index: int) -> dict[str, Any]:
        pre_mismatches = [
            boundary for boundary, state in zip(boundaries, states, strict=True)
            if boundary <= cutoff and not np.array_equal(state[row_index], state[1])
        ]
        first_post_shard = cutoff // SHARD_STEPS
        checked_shards = list(range(first_post_shard, len(records)))
        telemetry_mismatches: list[dict[str, Any]] = []
        for shard_index in checked_shards:
            controller = records[shard_index]["controller_diagnostics"][row_index]
            dynamics = records[shard_index]["per_row_diagnostics"][row_index]
            invalid: dict[str, Any] = {}
            for name, row in (
                *((name, controller) for name in telemetry_fields),
                *((name, dynamics) for name in logistic_fields),
            ):
                value = row.get(name)
                if (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or float(value) != 0.0
                ):
                    invalid[name] = value
            if invalid:
                telemetry_mismatches.append({"shard_index": shard_index, "invalid_or_nonzero": invalid})
        return {
            "cutoff_completed_reverse_steps": cutoff,
            "pre_cutoff_checked_boundaries": [value for value in boundaries if value <= cutoff],
            "pre_cutoff_status": "failed" if pre_mismatches else "passed",
            "pre_cutoff_mismatch_boundaries": pre_mismatches,
            "post_cutoff_first_shard_index": first_post_shard,
            "post_cutoff_checked_shards": checked_shards,
            "post_cutoff_status": (
                "not_reached" if not checked_shards
                else "failed" if telemetry_mismatches else "passed"
            ),
            "post_cutoff_telemetry_mismatches": telemetry_mismatches,
        }

    if spec.name == "stage-e-prior-cutoff-216-v1":
        cutoff = cutoff_identity(216, 2)
        mismatch = cutoff["pre_cutoff_status"] == "failed" or cutoff["post_cutoff_status"] == "failed"
        return {
            "schema": VERSION + "-schedule-identity",
            "status": "failed" if mismatch else "passed", "passed": int(not mismatch),
            "checked_boundaries": boundaries,
            "cutoff_rows": {"global-cutoff-216": cutoff},
        }

    binding = _read_json(run_dir / "bindings.json")
    baseline = _validate_schedule_baseline_copy(run_dir, binding.get("stage_d_schedule_baseline"))
    shared_mapping = {"zero": (0, 0), "global-plus-1": (1, 1), "source-informed": (4, 2)}
    shared_mismatches = {
        key: [
            boundary for boundary, state in zip(boundaries, states, strict=True)
            if not np.array_equal(state[new_index], baseline[boundary // SHARD_STEPS, old_index])
        ]
        for key, (new_index, old_index) in shared_mapping.items()
    }
    cutoff_records: dict[str, Any] = {}
    for cutoff, row_index in ((176, 2), (216, 3)):
        key = f"global-cutoff-{cutoff}"
        cutoff_records[key] = cutoff_identity(cutoff, row_index)
    mismatch = (
        any(shared_mismatches.values())
        or any(row["pre_cutoff_status"] == "failed" or row["post_cutoff_status"] == "failed"
               for row in cutoff_records.values())
    )
    return {
        "schema": VERSION + "-schedule-identity",
        "status": "failed" if mismatch else "passed",
        "passed": int(not mismatch),
        "checked_boundaries": boundaries,
        "baseline_identity": {
            "status": "failed" if any(shared_mismatches.values()) else "passed",
            "row_mapping": {"zero": 0, "global-plus-1": 1, "source-informed": 2},
            "mismatch_boundaries": shared_mismatches,
        },
        "cutoff_rows": cutoff_records,
    }


def _mechanism(
    spec: CandidateExperimentSpec, records: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(_row_order(spec)):
        values = [record["per_row_diagnostics"][index] for record in records]
        item: dict[str, Any] = {"row_key": key}
        for prefix in ("score", "logistic_shift", "reference_fraction_displacement", "control_fraction_displacement"):
            count = sum(int(value.get(prefix + "_count", 0)) for value in values)
            squared = math.fsum(float(value.get(prefix + "_squared_sum", 0.0)) for value in values)
            item[prefix + "_rms"] = math.sqrt(squared / count) if count else 0.0
            item[prefix + "_maximum_absolute"] = max((float(value.get(prefix + "_maximum_absolute", 0.0)) for value in values), default=0.0)
        reference = item["reference_fraction_displacement_rms"]
        item["control_reference_rms_ratio"] = item["control_fraction_displacement_rms"] / reference if reference else 0.0
        item["oracle_unreachable_count"] = sum(int(record["controller_diagnostics"][index].get("target_oracle_unreachable_boundary_count", 0)) for record in records)
        rows.append(item)
    learned_keys = tuple(key for key in _row_order(spec) if key.startswith("global-"))
    learned_summaries: dict[str, Any] = {}
    for key in learned_keys:
        learned = [row for row in metric_rows if row["row_key"] == key]
        improvements = [float(row["paired_squared_l2_improvement_over_zero"]) for row in learned]
        best_index = max(range(len(learned)), key=lambda index: improvements[index])
        learned_summaries[key] = {
            "first_strictly_negative_improvement_horizon": next(
                (int(row["completed_reverse_steps"]) for row in learned
                 if float(row["paired_squared_l2_improvement_over_zero"]) < 0.0),
                None,
            ),
            "best_improvement_horizon": int(learned[best_index]["completed_reverse_steps"]),
            "best_improvement": improvements[best_index],
            "terminal_improvement": (
                improvements[-1]
                if learned and int(learned[-1]["completed_reverse_steps"]) == OUTER_STEPS
                else None
            ),
            "state_vs_zero_horizon_series": [
                {
                    "completed_reverse_steps": int(row["completed_reverse_steps"]),
                    "state_vs_zero_squared_l2": row["state_vs_zero_squared_l2"],
                }
                for row in learned
            ],
        }
    always = learned_summaries["global-plus-1"]
    result = {
        "schema": VERSION + "-mechanism", "per_row": rows,
        "first_negative_learned_improvement_horizon": always["first_strictly_negative_improvement_horizon"],
        "on_policy_drift": "not_available_without_a_bound_training-calibration_input",
        "on_policy_proxy": [
            {
                "completed_reverse_steps": row["completed_reverse_steps"],
                "learned_state_vs_zero_squared_l2": row["state_vs_zero_squared_l2"],
            }
            for row in always["state_vs_zero_horizon_series"]
        ],
    }
    if spec.name in {"stage-d-schedule-window-v1", "stage-e-prior-cutoff-216-v1"}:
        result["learned_schedule_rows"] = learned_summaries
    return result


def _next_action(
    *, experiment_name: str, run_state: str, complete: bool,
    audit: Mapping[str, Any], oracle_positive: bool, learned_positive: bool,
    metric_rows: Sequence[Mapping[str, Any]],
) -> str:
    if run_state == "resource_paused":
        return "resume this run only after an explicit cap-extension approval; reuse every committed shard"
    if not complete:
        if run_state == "interrupted":
            return "resume this unchanged run from its longest valid candidate prefix"
        return "fix the candidate execution or integrity defect before continuing this objective"
    if experiment_name == "stage-e-prior-v1":
        if oracle_positive and learned_positive:
            return "request Stage F dataset-level implementation and launch approval"
        return "diagnose terminal-prior mismatch or controller transfer before Stage F"
    if audit.get("status") == "available" and audit.get("first16_not_completely_off") != 1:
        return "repair or audit the candidate kernel on the frozen first-16 prefix before Stage E"
    if audit.get("status") != "available":
        return "restore a compatible first-16 exact comparison before making a Stage E proxy-fidelity decision"
    if not oracle_positive:
        return "repair the oracle, schedule, or composition interface before changing the learner"
    if learned_positive:
        return "request Stage E launch approval"
    learned = [
        float(row["paired_squared_l2_improvement_over_zero"])
        for row in metric_rows if row["row_key"] == "global-plus-1"
    ]
    if any(value > 0.0 for value in learned[:-1]):
        return "target rollout or on-policy training, or revise the late application schedule"
    return "make one material learner or controller change; do not buy more exact provenance"


def _schedule_next_action(
    *, run_state: str, complete: bool, health: Mapping[str, Any],
    audit: Mapping[str, Any], oracle_positive: bool, selected: int | None,
    metric_rows: Sequence[Mapping[str, Any]],
) -> str:
    if run_state == "resource_paused":
        return "resume this run only after an explicit cap-extension approval; reuse every committed shard"
    if health.get("passed") != 1:
        return "repair the localized numerical, schedule, or pairing defect before interpreting cutoff metrics"
    if not complete:
        if run_state == "interrupted":
            return "resume this unchanged run from its longest valid candidate prefix"
        return "fix the candidate execution or integrity defect before continuing this objective"
    if audit.get("status") != "available" or audit.get("first16_not_completely_off") != 1:
        return "audit the candidate kernel on the frozen first-16 prefix before transferring a schedule"
    if not oracle_positive:
        return "repair the composition, backend, or oracle interface before changing the learner"
    if selected is not None:
        return "freeze the selected cutoff for human review and request separate Stage E implementation and compute approval"
    cutoff_rows = [
        row for row in metric_rows
        if row["row_key"] in {"global-cutoff-176", "global-cutoff-216"}
    ]
    if any(
        int(row["completed_reverse_steps"]) < OUTER_STEPS
        and float(row["paired_squared_l2_improvement_over_zero"]) > 0.0
        for row in cutoff_rows
    ):
        return "stop nearby cutoff tuning; move to rollout-aware/on-policy training or a material controller change"
    return "change learner scale or representation; do not attribute the result only to late scheduling"


def _stage_e_cutoff_next_action(
    *, run_state: str, complete: bool, health: Mapping[str, Any],
    oracle_validates: bool, cutoff_wins: bool,
) -> str:
    if run_state == "resource_paused":
        return "resume this run only after an explicit cap-extension approval; reuse every committed shard"
    if health.get("passed") != 1:
        return "repair only the localized integrity or cutoff-identity defect and rerun unchanged"
    if not complete:
        if run_state == "interrupted":
            return "resume this unchanged run from its longest valid candidate prefix"
        return "fix the candidate execution defect before interpreting the Stage E objective"
    if not oracle_validates:
        return "fix the prior, oracle, or composition path before attributing the learner result"
    if not cutoff_wins:
        return "stop static-cutoff work; pivot to rollout-aware training or a material controller"
    return "perform the required endpoint recognizability review; pivot if noise-like, otherwise plan bounded Stage F without auto-launch"


def _outcome_record(
    records: Sequence[Mapping[str, Any]], metric_rows: Sequence[Mapping[str, Any]],
    health: Mapping[str, Any], audit: Mapping[str, Any], run_state: str,
    error: str | None, experiment_name: str,
) -> dict[str, Any]:
    spec = experiment_spec(experiment_name)
    completed = len(records) * SHARD_STEPS
    final = {
        key: next(
            row for row in metric_rows
            if row["completed_reverse_steps"] == completed and row["row_key"] == key
        )
        for key in _row_order(spec)
    }
    complete = len(records) == SHARD_COUNT and run_state == "complete"
    oracle_positive = float(final["source-informed"]["paired_squared_l2_improvement_over_zero"]) > 0.0
    learned_positive = float(final["global-plus-1"]["paired_squared_l2_improvement_over_zero"]) > 0.0
    learned_relative = float(final["global-plus-1"]["relative_paired_squared_l2_improvement_over_zero"])
    if spec.name == "stage-d-schedule-window-v1":
        comparisons: dict[str, Any] = {}
        improvements: dict[int, float] = {}
        for cutoff in (176, 216):
            row = final[f"global-cutoff-{cutoff}"]
            improvement = float(row["paired_squared_l2_improvement_over_zero"])
            relative = float(row["relative_paired_squared_l2_improvement_over_zero"])
            improvements[cutoff] = improvement
            comparisons[str(cutoff)] = {
                "row_key": f"global-cutoff-{cutoff}",
                "paired_mixed_target_squared_l2_improvement_over_zero": improvement,
                "relative_paired_mixed_target_squared_l2_improvement_over_zero": relative,
                "positive": int(improvement > 0.0),
                "exceeds_one_percent_marker": int(relative >= 0.01),
            }
        selected: int | None = None
        if complete and health.get("passed") == 1 and (improvements[176] > 0.0 or improvements[216] > 0.0):
            selected = 216 if improvements[216] > improvements[176] else 176
        ready_for_review = (
            selected is not None and oracle_positive
            and audit.get("status") == "available"
            and audit.get("first16_not_completely_off") == 1
        )
        next_action = _schedule_next_action(
            run_state=run_state, complete=complete, health=health, audit=audit,
            oracle_positive=oracle_positive, selected=selected, metric_rows=metric_rows,
        )
        return {
            "schema": VERSION + "-outcome", "research_mode": "exploratory", "run_state": run_state,
            "scientific_objective_completed": int(complete and health.get("passed") == 1),
            "completed_reverse_steps": completed,
            "primary_final_metrics": final, "health_passed": health["passed"],
            "schedule_identity_passed": health.get("schedule_identity", {}).get("passed"),
            "first16_not_completely_off": audit.get("first16_not_completely_off"),
            "oracle_directionally_improves_zero": int(oracle_positive),
            "learned_directionally_improves_zero": int(learned_positive),
            "cutoff_terminal_comparisons": comparisons,
            "selected_cutoff_completed_reverse_steps": selected,
            "selected_cutoff_ready_for_review": int(ready_for_review),
            "selection_tie_rule": "select_176_on_exact_equality",
            "one_percent_marker_type": "diagnostic_threshold",
            "stage_e_machine_eligible": 0, "stage_e_automatically_launched": 0,
            "stage_f_machine_eligible": 0, "stage_f_automatically_launched": 0,
            "human_visual_review_required": 1, "confirmatory_claim": 0, "error": error,
            "next_action": next_action,
        }
    if spec.name == "stage-e-prior-cutoff-216-v1":
        zero = final["zero"]
        always = final["global-plus-1"]
        cutoff = final["global-cutoff-216"]
        oracle = final["source-informed"]
        cutoff_error = float(cutoff["mixed_target_squared_l2_error"])
        zero_error = float(zero["mixed_target_squared_l2_error"])
        always_error = float(always["mixed_target_squared_l2_error"])
        oracle_improvement = float(oracle["paired_squared_l2_improvement_over_zero"])
        cutoff_improvement = float(cutoff["paired_squared_l2_improvement_over_zero"])
        secondary_improvement = always_error - cutoff_error
        oracle_validates = float(oracle["relative_paired_squared_l2_improvement_over_zero"]) >= 0.01
        cutoff_wins = cutoff_error < zero_error and cutoff_error < always_error
        next_action = _stage_e_cutoff_next_action(
            run_state=run_state, complete=complete, health=health,
            oracle_validates=oracle_validates, cutoff_wins=cutoff_wins,
        )
        return {
            "schema": VERSION + "-outcome", "research_mode": "exploratory", "run_state": run_state,
            "scientific_objective_completed": int(complete and health.get("passed") == 1),
            "completed_reverse_steps": completed,
            "primary_final_metrics": final, "health_passed": health["passed"],
            "schedule_identity_passed": health.get("schedule_identity", {}).get("passed"),
            "first16_not_completely_off": audit.get("first16_not_completely_off"),
            "oracle_directionally_improves_zero": int(oracle_positive),
            "oracle_exceeds_one_percent_attribution_threshold": int(oracle_validates),
            "cutoff_vs_zero": {
                "paired_mixed_target_squared_l2_improvement": cutoff_improvement,
                "relative_paired_mixed_target_squared_l2_improvement": float(
                    cutoff["relative_paired_squared_l2_improvement_over_zero"]
                ),
                "mixed_target_centered_contrast_correlation": cutoff[
                    "mixed_target_centered_contrast_correlation"
                ],
                "strictly_better": int(cutoff_error < zero_error),
            },
            "cutoff_vs_always_on": {
                "paired_mixed_target_squared_l2_improvement": secondary_improvement,
                "relative_paired_mixed_target_squared_l2_improvement": (
                    secondary_improvement / always_error if always_error else 0.0
                ),
                "strictly_better": int(cutoff_error < always_error),
            },
            "cutoff_oracle_gap_closure": (
                cutoff_improvement / oracle_improvement if oracle_improvement else None
            ),
            "cutoff_strictly_better_than_zero_and_always_on": int(cutoff_wins),
            "cutoff_exceeds_one_percent_marker": int(
                float(cutoff["relative_paired_squared_l2_improvement_over_zero"]) >= 0.01
            ),
            "one_percent_marker_type": "diagnostic_threshold",
            "human_visual_review_required": 1,
            "human_endpoint_recognizability": "not_automated",
            "stage_e_machine_eligible": 0, "stage_e_automatically_launched": 0,
            "stage_f_machine_eligible": 0, "stage_f_automatically_launched": 0,
            "confirmatory_claim": 0, "error": error, "next_action": next_action,
        }
    stage_d = experiment_name == "stage-d-anchor-v1"
    stage_e_eligible = stage_d and complete and oracle_positive and learned_positive and audit.get("first16_not_completely_off") == 1
    stage_f_eligible = not stage_d and complete and oracle_positive and learned_positive
    next_action = _next_action(
        experiment_name=experiment_name, run_state=run_state, complete=complete,
        audit=audit, oracle_positive=oracle_positive, learned_positive=learned_positive,
        metric_rows=metric_rows,
    )
    return {
        "schema": VERSION + "-outcome", "research_mode": "exploratory", "run_state": run_state,
        "scientific_objective_completed": int(complete), "completed_reverse_steps": completed,
        "primary_final_metrics": final, "health_passed": health["passed"],
        "first16_not_completely_off": audit.get("first16_not_completely_off"),
        "oracle_directionally_improves_zero": int(oracle_positive),
        "learned_directionally_improves_zero": int(learned_positive),
        "learned_exceeds_one_percent_planning_marker": int(learned_relative >= 0.01),
        "one_percent_marker_type": "diagnostic_threshold",
        "stage_e_machine_eligible": int(stage_e_eligible), "stage_e_automatically_launched": 0,
        "stage_f_machine_eligible": int(stage_f_eligible), "stage_f_automatically_launched": 0,
        "human_visual_review_required": 1, "confirmatory_claim": 0, "error": error,
        "next_action": next_action,
    }


def _terminalize(run_dir: Path, run_state: str, error: str | None = None) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json"); spec = experiment_spec(config["experiment_name"])
    source, _ = _validate_core_inputs(run_dir); records, states = _scan_prefix(run_dir)
    boundaries = [index * SHARD_STEPS for index in range(len(states))]
    atomic_rollout_npz(run_dir / "reverse/trajectory_boundaries.npz", {"completed_reverse_steps": np.asarray(boundaries, dtype=np.int64), "states": np.ascontiguousarray(np.stack(states), dtype=np.float64)})
    metrics = _metric_rows(spec, boundaries, states, source); atomic_write_csv(run_dir / "reverse/metrics.csv", metrics)
    identity = _schedule_identity(run_dir, records, states) if spec.name in {
        "stage-d-schedule-window-v1", "stage-e-prior-cutoff-216-v1",
    } else None
    health = _health(records, states, identity); atomic_write_json(run_dir / "reverse/health.json", health)
    audit = _first16_audit(run_dir, states, source) if len(states) >= 2 else {"schema": VERSION + "-first16-audit", "status": "not_reached", "first16_not_completely_off": None, "blocks_artifact_completion": 0}
    atomic_write_json(run_dir / "reverse/first16_audit.json", audit)
    mechanism = _mechanism(spec, records, metrics); atomic_write_json(run_dir / "reverse/mechanism.json", mechanism)
    _render(run_dir, boundaries, states, source)
    outcome = _outcome_record(records, metrics, health, audit, run_state, error, spec.name)
    atomic_write_json(run_dir / "outcome.json", outcome)
    return {"outcome": outcome, "health": health, "audit": audit}


def _report_text(
    run_dir: Path, status: Mapping[str, Any], outcome: Mapping[str, Any],
    health: Mapping[str, Any], audit: Mapping[str, Any], config: Mapping[str, Any],
) -> str:
    final = outcome.get("primary_final_metrics", {})
    def objective(key: str) -> str:
        row = final.get(key, {})
        return (
            f"mixed-target L2^2={row.get('mixed_target_squared_l2_error', 'unavailable')}, "
            f"paired improvement={row.get('paired_squared_l2_improvement_over_zero', 'unavailable')}, "
            f"relative={row.get('relative_paired_squared_l2_improvement_over_zero', 'unavailable')}"
        )
    key_artifacts = (
        "reverse/trajectory_boundaries.npz", "reverse/metrics.csv", "reverse/health.json",
        "reverse/first16_audit.json", "reverse/mechanism.json", "outcome.json",
        "images/contact_sheets/raw-step-000.png", "images/context/source.png",
        "resource_ledger.json", "status.json",
    )
    if config["experiment_name"] == "stage-e-prior-cutoff-216-v1":
        key_artifacts += (
            "images/individual/raw/global-cutoff-216/step-512.png",
            "images/individual/demixed/global-cutoff-216/step-512.png",
            "images/contact_sheets/raw-step-512.png",
            "images/contact_sheets/demixed-step-512.png",
        )
    existing = [relative for relative in key_artifacts if (run_dir / relative).is_file()]
    if config["experiment_name"] == "stage-d-schedule-window-v1":
        identity = health.get("schedule_identity", {})
        report = f"""# Stage D schedule-window exploratory report

Research mode: exploratory. Independent unit: one already-opened image/path.

Decision: does stopping the frozen learned controller after completed reverse-step 176 or 216 turn its late full-path harm into a positive terminal improvement over paired zero control?

Named specification: stage-d-schedule-window-v1; path ID {config['spec']['path_id']}; root seed {config['spec']['root_seed']}. Frozen row order: zero, global-plus-1, global-cutoff-176, global-cutoff-216, source-informed. Every row uses the same canonical path and random transition IDs.

Cutoff semantics: learned gain is exactly 1.0 when `reverse_time < cutoff/512`; the step reaching boundary 176 (outer k=336) or 216 (outer k=296) remains active, and every subsequent controller contribution is exactly zero.

Status: {status['state']}; completed reverse steps: {status['completed_reverse_steps']}; active seconds: {status['active_seconds']} / {status['maximum_active_seconds']}.

Resource-accounting note: an interrupted attempt is billed by a conservative UTC wall-interval upper bound plus five seconds; that upper bound can include offline downtime and is not accelerator utilization.

Measured candidate shard seconds: {status['committed_shard_seconds']}; candidate setup/orchestration seconds: {status['candidate_setup_orchestration_seconds']}.

Numerical contract: fixed 28x28 grid; split-time Eulerian/Jacobi dynamics; binary64 finite Legendre inverse-CDF candidate (128-mode profile, adaptive through 1024, 56 bisections); learned Rao--Blackwell score surrogate. This is not an exact-law, continuum, population, or generator claim.

Primary endpoint metrics:
- zero: {objective('zero')}
- global-plus-1: {objective('global-plus-1')}
- global-cutoff-176: {objective('global-cutoff-176')}
- global-cutoff-216: {objective('global-cutoff-216')}
- source-informed: {objective('source-informed')}

Schedule identity: {identity.get('status', 'unavailable')}. Oracle terminal direction: {outcome.get('oracle_directionally_improves_zero')}. First-16 candidate/exact diagnostic: {audit.get('status')}; not completely off: {audit.get('first16_not_completely_off')}.

Selected cutoff: {outcome.get('selected_cutoff_completed_reverse_steps')}. Exact ties select 176. The 1% relative-improvement marker is diagnostic only and cannot veto a strictly positive endpoint.

Stage E machine eligibility: 0. Stage E was not automatically launched. Stage F machine eligibility: 0. Stage F was not automatically launched. Any schedule transfer requires a separate implementation review and compute approval.

Required next action: {outcome.get('next_action', 'none')}.

Claim boundary: one opened image/path under the candidate fixed-grid law only; no exact-law, confirmatory, population, diversity, or MNIST-generator claim. The cutoffs were selected post hoc from the preceding Stage D trajectory.

Error/stop: {status.get('error') or 'none'}.

Key existing artifact paths ({len(existing)}; exhaustive hashes are in `artifact_manifest.json`):
""" + "\n".join(f"- `{path}`" for path in existing) + "\n"
        return report
    if config["experiment_name"] == "stage-e-prior-cutoff-216-v1":
        identity = health.get("schedule_identity", {})
        primary = outcome.get("cutoff_vs_zero", {})
        secondary = outcome.get("cutoff_vs_always_on", {})
        report = f"""# Stage E prior-start cutoff-216 exploratory report

Research mode: exploratory. Independent unit: one opened target-specific model and one frozen Dirichlet-prior path.

Decision: on the frozen prior path, does cutoff 216 outperform paired zero and always-on control while the oracle validates composition, or must the learner/controller strategy pivot before Stage F?

Named specification: stage-e-prior-cutoff-216-v1; path ID {config['spec']['path_id']}; root seed {config['spec']['root_seed']}; Dirichlet seed {config['spec']['start']['seed']}. Frozen row order: zero, global-plus-1, global-cutoff-216, source-informed. Every row uses the same canonical path and random transition IDs.

Cutoff semantics: learned gain is exactly 1.0 through completed reverse-step 216 and every later applied score, unscaled score, and logistic shift is exactly zero.

The completed Stage D schedule predecessor was authenticated by its full manifest/link tree and six pinned copied authorities. Its ledger contained the literal placeholder `{STAGE_D_APPROVAL_REFERENCE_CAVEAT}`; this provenance caveat was preserved and is not approval for this child run.

Status: {status['state']}; completed reverse steps: {status['completed_reverse_steps']}; active seconds: {status['active_seconds']} / {status['maximum_active_seconds']}.

Resource-accounting note: an interrupted attempt is billed by a conservative UTC wall-interval upper bound plus five seconds; that upper bound can include offline downtime and is not accelerator utilization.

Numerical contract: fixed 28x28 grid; split-time Eulerian/Jacobi dynamics; binary64 finite Legendre inverse-CDF candidate (128-mode profile, adaptive through 1024, 56 bisections); learned Rao--Blackwell score surrogate. This is not an exact-law, continuum, population, or generator claim.

Endpoint metrics:
- zero: {objective('zero')}
- global-plus-1: {objective('global-plus-1')}
- global-cutoff-216: {objective('global-cutoff-216')}
- source-informed: {objective('source-informed')}

Primary cutoff-versus-zero result: paired improvement={primary.get('paired_mixed_target_squared_l2_improvement')}, relative={primary.get('relative_paired_mixed_target_squared_l2_improvement')}, centered correlation={primary.get('mixed_target_centered_contrast_correlation')}, strictly better={primary.get('strictly_better')}.

Secondary cutoff-versus-always-on result: paired improvement={secondary.get('paired_mixed_target_squared_l2_improvement')}, relative={secondary.get('relative_paired_mixed_target_squared_l2_improvement')}, strictly better={secondary.get('strictly_better')}. Oracle-gap closure={outcome.get('cutoff_oracle_gap_closure')}.

Schedule identity: {identity.get('status', 'unavailable')}. Oracle improves zero by at least 1% for learner attribution: {outcome.get('oracle_exceeds_one_percent_attribution_threshold')}. The cutoff 1% marker is diagnostic only.

All boundary metrics are in `reverse/metrics.csv`; controller magnitudes are in `reverse/mechanism.json`; raw, demixed, and contact-sheet endpoint images are under `images/` and are retained even for an adverse result.

Human endpoint recognizability review is required and is not automated. Stage F machine eligibility: 0. Stage F was not automatically launched.

Required next action: {outcome.get('next_action', 'none')}.

Claim boundary: one opened target-specific model, one fixed prior seed/path, and the approximate candidate law only; no exact-law, population, diversity, confirmatory, or generator claim.

Error/stop: {status.get('error') or 'none'}.

Key existing artifact paths ({len(existing)}; exhaustive hashes are in `artifact_manifest.json`):
""" + "\n".join(f"- `{path}`" for path in existing) + "\n"
        return report
    report = f"""# Candidate complete-path exploratory report

Research mode: exploratory. Independent unit: one opened image/path.

Decision: can the frozen one-image Eulerian controller complete 512 candidate-law reverse steps and improve paired zero while the source-informed oracle controls composition?

Named specification: {config['experiment_name']}; path ID {config['spec']['path_id']}; root seed {config['spec']['root_seed']}. Evidence roles: zero=null control, global-plus-1=frozen learned checkpoint, source-informed=target-aware positive system control. Checkpoint selection is inherited and frozen at SHA-256 {CHECKPOINT_SHA256}.

Status: {status['state']}; completed reverse steps: {status['completed_reverse_steps']}; active seconds: {status['active_seconds']} / {status['maximum_active_seconds']}.

Resource-accounting note: an interrupted attempt is billed by a conservative UTC wall-interval upper bound plus five seconds; that upper bound can include offline downtime and is not accelerator utilization.

Measured candidate shard seconds: {status['committed_shard_seconds']}; candidate setup/orchestration seconds: {status['candidate_setup_orchestration_seconds']}.

Numerical contract: fixed 28x28 grid; split-time Eulerian/Jacobi dynamics; binary64 finite Legendre inverse-CDF candidate (128-mode profile, adaptive through 1024, 56 bisections); learned Rao--Blackwell score surrogate. This is not an exact-law, continuum, population, or generator claim.

Health: candidate approximation count {health.get('approximation_count', 0)}, invalid count {health.get('invalid_count', 0)}, certification not applicable, maximum mass error {health.get('maximum_mass_error', 0.0)}.

Primary endpoint metrics:
- zero: {objective('zero')}
- global-plus-1: {objective('global-plus-1')}
- source-informed: {objective('source-informed')}

First-16 candidate/exact diagnostic: {audit.get('status')}; not completely off: {audit.get('first16_not_completely_off')}.

Stage E machine eligibility: {outcome.get('stage_e_machine_eligible', 0)}. Stage E was not automatically launched.

Stage F machine eligibility: {outcome.get('stage_f_machine_eligible', 0)}. Stage F was not automatically launched.

Required next action: {outcome.get('next_action', 'none')}.

Claim boundary: one opened image/path under the candidate fixed-grid law only; no exact-law, confirmatory, population, diversity, or MNIST-generator claim. On-policy calibration is deliberately unavailable because no calibration artifact is bound.

Error/stop: {status.get('error') or 'none'}.

Key existing artifact paths ({len(existing)}; exhaustive hashes are in `artifact_manifest.json`):
""" + "\n".join(f"- `{path}`" for path in existing) + "\n"
    return report


def _write_status_report(
    run_dir: Path, state: str, derived: Mapping[str, Any], error: str | None,
    *, terminalization_target_state: str | None = None,
    terminalization_target_error: str | None = None,
) -> None:
    records, _ = _scan_prefix(run_dir); ledger = _validate_ledger(run_dir); config = _read_json(run_dir / "config.json")
    shard_seconds = math.fsum(float(record.get("elapsed_seconds", 0.0)) for record in records)
    execution_seconds = math.fsum(
        float(event["elapsed_seconds"]) for event in ledger["events"]
        if str(event["role"]).startswith("candidate_execution")
    )
    status = {
        "schema": VERSION + "-status", "state": state, "updated_at": _utc_now(),
        "committed_shards": len(records), "completed_reverse_steps": len(records) * SHARD_STEPS,
        "resumable": int(state in {"resource_paused", "interrupted", "derived_failed"}),
        "active_seconds": ledger["active_seconds"], "maximum_active_seconds": ledger["maximum_active_seconds"],
        "committed_shard_seconds": shard_seconds,
        "candidate_setup_orchestration_seconds": max(0.0, execution_seconds - shard_seconds),
        "terminalization_target_state": terminalization_target_state,
        "terminalization_target_error": terminalization_target_error,
        "error": error,
    }
    atomic_write_json(run_dir / "status.json", status)
    outcome = derived.get("outcome", {}); health = derived.get("health", {}); audit = derived.get("audit", {})
    _atomic_text(run_dir / "REPORT.md", _report_text(run_dir, status, outcome, health, audit, config))


def _refresh_manifest(run_dir: Path) -> dict[str, Any]:
    _reject_run_links(run_dir)
    rows = []
    for path in sorted(run_dir.rglob("*"), key=lambda value: value.relative_to(run_dir).as_posix()):
        if not path.is_file() or path == run_dir / "artifact_manifest.json":
            continue
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise CandidateRunError("run artifact tree contains a link or hardlink")
        rows.append({"path": path.relative_to(run_dir).as_posix(), "size": path.stat().st_size, "sha256": file_fingerprint(path)})
    manifest = {"schema": VERSION + "-artifact-manifest", "artifacts": rows, "artifact_count": len(rows), "artifact_bytes": sum(row["size"] for row in rows)}
    atomic_write_json(run_dir / "artifact_manifest.json", manifest); return manifest


def _initialize_run(args: argparse.Namespace, spec: CandidateExperimentSpec) -> Path:
    repository_root = Path(__file__).resolve().parents[1]; reference = Path(args.reference_run_dir).resolve(); runs_root = Path(args.runs_root).resolve()
    if not _SAFE_NAME.fullmatch(args.run_name) or not reference.is_dir():
        raise CandidateRunError("run name/reference directory is invalid")
    if reference == runs_root or reference in runs_root.parents:
        raise CandidateRunError("reference and output roots must be disjoint")
    needs_stage_d = spec.name in {
        "stage-e-prior-v1", "stage-d-schedule-window-v1", "stage-e-prior-cutoff-216-v1",
    }
    stage_d_argument = getattr(args, "stage_d_run_dir", None)
    stage_d_path = Path(stage_d_argument).resolve() if needs_stage_d and stage_d_argument else None
    if needs_stage_d and stage_d_path is None:
        raise CandidateRunError(f"{spec.name} requires --stage-d-run-dir")
    if stage_d_path is not None and (stage_d_path == runs_root or stage_d_path in runs_root.parents):
        raise CandidateRunError("Stage D predecessor and output roots must be disjoint")
    destination = runs_root / args.run_name
    for immutable in (reference, stage_d_path):
        if immutable is not None and (
            destination == immutable or immutable in destination.parents or destination in immutable.parents
        ):
            raise CandidateRunError("fresh run destination overlaps an immutable input")
    runs_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CandidateRunError("fresh run destination already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.run_name}.", dir=runs_root))
    try:
        copies = {
            name: {"path": target, **_copy_file(reference / source_path, temporary / target, digest)}
            for name, (target, source_path, digest) in _core_copy_specs().items()
        }
        source, anchor = _validate_core_inputs(temporary)
        exact_binding: dict[str, Any] = {"status": "not_applicable", "reason": "stage_e_has_different_start_law", "files": []}
        if spec.name in {"stage-d-anchor-v1", "stage-d-schedule-window-v1"}:
            try:
                exact_rows = _validate_exact_prefix(reference / EXACT_PREFIX_RELATIVE)
                exact_binding = {"status": "available", "reason": None, "files": []}
                for row in exact_rows:
                    for kind in ("json", "npz"):
                        source_path = row[kind]; destination_path = temporary / f"inputs/exact_prefix/{source_path.name}"
                        exact_binding["files"].append({
                            "path": f"inputs/exact_prefix/{source_path.name}",
                            **_copy_file(source_path, destination_path, EXACT_PREFIX_HASHES[source_path.name]),
                        })
            except CandidateRunError as exc:
                exact_binding = {"status": "unavailable", "reason": str(exc), "files": []}
        stage_d_binding: dict[str, Any] | None = None
        schedule_baseline_binding: dict[str, Any] | None = None
        schedule_predecessor_binding: dict[str, Any] | None = None
        if spec.name == "stage-d-schedule-window-v1":
            assert stage_d_path is not None
            schedule_baseline_binding = _copy_schedule_baseline(stage_d_path, temporary)
        if spec.name == "stage-e-prior-cutoff-216-v1":
            assert stage_d_path is not None
            schedule_predecessor_binding = _copy_stage_e_predecessor(stage_d_path, temporary)
        if spec.start.kind == "forward_anchor":
            start, prior = anchor.copy(), None
        else:
            assert stage_d_path is not None
            if spec.name == "stage-e-prior-v1":
                stage_d = verify_run(stage_d_path)
                if stage_d["outcome"].get("stage_e_machine_eligible") != 1:
                    raise CandidateRunError("cited Stage D run is not machine-eligible for Stage E")
                stage_d_binding = {
                    "run_dir": str(stage_d_path), "manifest_sha256": file_fingerprint(stage_d_path / "artifact_manifest.json"),
                    "outcome_sha256": file_fingerprint(stage_d_path / "outcome.json"),
                    "first16_audit_sha256": file_fingerprint(stage_d_path / "reverse/first16_audit.json"),
                }
            start, prior = _prior_state(int(spec.start.seed))
        atomic_rollout_npz(temporary / "inputs/start_state.npz", {"state": start})
        start_file_sha256 = rollout_file_sha256(temporary / "inputs/start_state.npz")
        config = {
            "schema": VERSION + "-config", "experiment_name": spec.name, "spec": spec.to_record(),
            "row_order": list(_row_order(spec)), "row_table": [row.to_record() for row in _row_specs(spec)],
            "controller_binding": _controller_binding(spec), "rng_binding": _rng_binding(spec),
            "start_state_sha256": rollout_array_sha256(start), "prior": prior,
            "reference_contract": CANDIDATE_REFERENCE_CONTRACT,
            "research_mode": "exploratory", "confirmation_evidence_opened": 0,
        }
        atomic_write_json(temporary / "config.json", config)
        bindings_record = {
            "schema": VERSION + "-bindings", "reference_run_dir": str(reference), "copies": copies,
            "source_image_sha256": rollout_array_sha256(source.source_image), "mixed_target_sha256": rollout_array_sha256(source.mixed_target),
            "start_state_sha256": config["start_state_sha256"], "start_state_file_sha256": start_file_sha256,
            "exact_prefix": exact_binding, "stage_d_predecessor": stage_d_binding,
            "source": _module_bindings(repository_root), "spec_sha256": config_fingerprint(spec.to_record()),
        }
        if schedule_baseline_binding is not None:
            bindings_record["stage_d_schedule_baseline"] = schedule_baseline_binding
        if schedule_predecessor_binding is not None:
            bindings_record["stage_d_schedule_predecessor"] = schedule_predecessor_binding
        atomic_write_json(temporary / "bindings.json", bindings_record)
        atomic_write_json(
            temporary / "resource_ledger.json",
            _new_ledger(float(args.maximum_active_seconds), args.approval_reference),
        )
        atomic_write_json(temporary / "status.json", {"schema": VERSION + "-status", "state": "initialized", "resumable": 1, "committed_shards": 0})
        _atomic_text(temporary / "command.txt", subprocess.list2cmdline(sys.argv) + "\n")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise
    return destination


def _build_family(run_dir: Path, device: torch.device) -> tuple[Any, ...]:
    config = _read_json(run_dir / "config.json"); spec = experiment_spec(config["experiment_name"]); source, _ = _validate_core_inputs(run_dir)
    model = _load_model(run_dir, device); specs = _row_specs(spec)
    controllers: dict[str, Any] = {
        "global-plus-1": ScaledTangentScoreController(model, 1.0),
        "source-informed": TargetFractionOracleController(source.mixed_target, microsteps=MICROSTEPS).to(device),
    }
    if spec.name in {"stage-d-schedule-window-v1", "stage-e-prior-cutoff-216-v1"}:
        controllers["global-cutoff-216"] = CompletedStepCutoffTangentScoreController(model, 216)
    if spec.name == "stage-d-schedule-window-v1":
        controllers["global-cutoff-176"] = CompletedStepCutoffTangentScoreController(model, 176)
    bank = FusedTangentControllerBank(specs, controllers)
    from mnist.d0_jacobi_rb_cuda_deferred import prepare_alpha1_rb_transition_batch_cuda_candidate
    profile = JacobiRBCudaProfile(); prepared = prepare_alpha1_rb_transition_batch_cuda_candidate(device=device, profile=profile)
    backend_record = {
        "schema": VERSION + "-candidate-backend", "device": str(prepared.device),
        "candidate_binary_sha256": prepared.candidate_binary_sha256,
        "candidate_modes": int(profile.candidate_modes),
        "candidate_bisection_steps": int(profile.candidate_bisection_steps),
        "exact_authorizer_authority": 0, "synchronous_replay_authority": 0,
    }
    backend_path = run_dir / "backend.json"
    if backend_path.exists():
        if _read_json(backend_path) != backend_record:
            raise CandidateRunError("candidate backend binary changed on resume")
    else:
        atomic_write_json(backend_path, backend_record)
    seeds = prepare_deferred_reference_rng_seed_map(prepared_backend=prepared, root_seed=spec.root_seed, stream_role=spec.stream_role)
    def factory(_index: int) -> CandidateApproximateFusedReference:
        return CandidateApproximateFusedReference(profile=profile, root_seed=spec.root_seed, stream_role=spec.stream_role, prepared_backend=prepared, prepared_rng_seeds=seeds)
    return spec, source, specs, bank, factory


def _execute(run_dir: Path, device_name: str) -> tuple[str, str | None]:
    _validate_core_inputs(run_dir); _recover_uncommitted(run_dir); records, _ = _scan_prefix(run_dir)
    if len(records) == SHARD_COUNT:
        return "complete", None
    initial_projection = _projection(run_dir, time.perf_counter(), records)
    initial_storage = _directory_bytes(run_dir)
    if initial_storage >= STORAGE_CAP_BYTES - TERMINAL_STORAGE_RESERVE_BYTES:
        raise CandidateRunError("candidate artifact storage reached the terminal-reserve integrity stop")
    if not initial_projection["passed"]:
        return "resource_paused", json.dumps({
            **initial_projection, "storage_bytes": initial_storage,
            "pre_cuda_admission": 1, "pause_reason": "active_time_projection",
        }, sort_keys=True)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise CandidateRunError("candidate production requires CUDA")
    attempt, started = _begin_attempt(run_dir, "candidate_execution")
    try:
        atomic_write_json(run_dir / "status.json", {
            "schema": VERSION + "-status", "state": "running", "updated_at": _utc_now(),
            "resumable": 1, "committed_shards": len(records),
            "completed_reverse_steps": len(records) * SHARD_STEPS, "error": None,
        })
        enable_deterministic_torch()
        spec, _source, specs, bank, factory = _build_family(run_dir, device)
        start = _npz(run_dir / "inputs/start_state.npz")["state"]
        initial = np.repeat(start[None, :], len(specs), axis=0)
        def admit(_plan: Any) -> None:
            committed, committed_states = _scan_prefix(run_dir)
            if spec.name in {"stage-d-schedule-window-v1", "stage-e-prior-cutoff-216-v1"}:
                identity = _schedule_identity(run_dir, committed, committed_states)
                if identity.get("passed") != 1:
                    raise CandidateRunError("committed schedule identity failed before the next shard")
            projection = _projection(run_dir, started, committed)
            storage = _directory_bytes(run_dir); total = int(torch.cuda.get_device_properties(device).total_memory); peak = int(torch.cuda.max_memory_allocated(device))
            if storage >= STORAGE_CAP_BYTES - TERMINAL_STORAGE_RESERVE_BYTES:
                raise CandidateRunError("candidate artifact storage reached the terminal-reserve integrity stop")
            if total and peak / total >= CUDA_FRACTION_CAP:
                raise CandidateRunError("candidate CUDA allocation reached the 80% integrity stop")
            if not projection["passed"]:
                raise ResourcePause({
                    **projection, "storage_bytes": storage, "peak_cuda_bytes": peak,
                    "total_cuda_bytes": total, "pause_reason": "active_time_projection",
                })
        run_fused_reverse_family(
            initial, sequence=reverse_suffix_sequence(511), output_dir=run_dir / "reverse",
            family_name=_family_name(spec), segment_name="complete-512", row_specs=specs,
            controller_bank=bank, reference_factory=factory, controller_binding=_controller_binding(spec),
            rng_binding=_rng_binding(spec), label=spec.label, microsteps=spec.microsteps, device=device,
            before_uncommitted_shard=admit, reference_contract="candidate_approximate",
        )
    except ResourcePause as exc:
        _finish_attempt(run_dir, attempt, started, failed=False); return "resource_paused", json.dumps(exc.projection, sort_keys=True)
    except KeyboardInterrupt:
        _finish_attempt(run_dir, attempt, started, failed=True); return "interrupted", "KeyboardInterrupt"
    except Exception:
        _finish_attempt(run_dir, attempt, started, failed=True); raise
    _finish_attempt(run_dir, attempt, started, failed=False); return "complete", None


def _terminalize_accounted(run_dir: Path, state: str, error: str | None) -> dict[str, Any]:
    attempt, started = _begin_attempt(run_dir, "terminalization")
    finished = False
    try:
        derived = _terminalize(run_dir, state, error)
        _finish_attempt(run_dir, attempt, started, failed=False); finished = True
        _write_status_report(run_dir, state, derived, error); _refresh_manifest(run_dir)
        if _directory_bytes(run_dir) >= STORAGE_CAP_BYTES:
            raise CandidateRunError("terminalized run exceeds storage cap")
        verify_run(run_dir)
        return derived
    except Exception as exc:
        if not finished:
            _finish_attempt(run_dir, attempt, started, failed=True)
        derived = {"outcome": {}, "health": {}, "audit": {}}
        _write_status_report(
            run_dir, "derived_failed", derived, str(exc),
            terminalization_target_state=state, terminalization_target_error=error,
        )
        _refresh_manifest(run_dir)
        raise


def _verify_manifest(run_dir: Path) -> dict[str, Any]:
    _reject_run_links(run_dir)
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.stat().st_nlink != 1:
        raise CandidateRunError("artifact manifest is not an independent regular file")
    manifest = _read_json(manifest_path); rows = manifest.get("artifacts")
    if manifest.get("schema") != VERSION + "-artifact-manifest" or not isinstance(rows, list):
        raise CandidateRunError("artifact manifest changed")
    actual = sorted(
        path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*")
        if path.is_file() and path != run_dir / "artifact_manifest.json"
    )
    if [row.get("path") for row in rows] != actual:
        raise CandidateRunError("artifact manifest path set changed")
    for row in rows:
        path = run_dir / row["path"]
        if path.is_symlink() or path.stat().st_nlink != 1 or path.stat().st_size != row.get("size") or file_fingerprint(path) != row.get("sha256"):
            raise CandidateRunError(f"artifact changed: {row['path']}")
    if manifest.get("artifact_count") != len(rows) or manifest.get("artifact_bytes") != sum(int(row["size"]) for row in rows):
        raise CandidateRunError("artifact manifest totals changed")
    return manifest


def _validate_run_authority(run_dir: Path) -> tuple[dict[str, Any], CandidateExperimentSpec, Any, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    _reject_run_links(run_dir)
    config = _read_json(run_dir / "config.json"); spec = experiment_spec(config["experiment_name"])
    if (
        config.get("schema") != VERSION + "-config" or config.get("spec") != spec.to_record()
        or config.get("row_order") != list(_row_order(spec))
        or config.get("row_table") != [row.to_record() for row in _row_specs(spec)]
        or config.get("controller_binding") != _controller_binding(spec)
        or config.get("rng_binding") != _rng_binding(spec)
        or config.get("reference_contract") != CANDIDATE_REFERENCE_CONTRACT
        or config.get("research_mode") != "exploratory" or config.get("confirmation_evidence_opened") != 0
    ):
        raise CandidateRunError("frozen experiment specification changed")
    source, _ = _validate_core_inputs(run_dir); bindings = _read_json(run_dir / "bindings.json")
    if (
        bindings.get("schema") != VERSION + "-bindings"
        or bindings.get("spec_sha256") != config_fingerprint(spec.to_record())
        or bindings.get("start_state_sha256") != config.get("start_state_sha256")
    ):
        raise CandidateRunError("input/spec binding changed")
    expected_copies = {name: row[0] for name, row in _core_copy_specs().items()}
    copies = bindings.get("copies")
    if not isinstance(copies, Mapping) or set(copies) != set(expected_copies):
        raise CandidateRunError("core input copy inventory changed")
    for name, relative in expected_copies.items():
        claim, path = copies[name], run_dir / relative
        if not isinstance(claim, Mapping) or claim.get("path") != relative or claim.get("size") != path.stat().st_size or claim.get("sha256") != file_fingerprint(path):
            raise CandidateRunError("core input copy binding changed")
    source_files = bindings.get("source", {}).get("direct_source_files")
    if (
        not isinstance(source_files, Mapping)
        or set(source_files) != set(_DIRECT_SOURCE_FILES)
        or any(not re.fullmatch(r"[0-9a-f]{64}", str(digest)) for digest in source_files.values())
    ):
        raise CandidateRunError("bound source inventory changed")
    for relative, digest in source_files.items():
        path = Path(__file__).resolve().parents[1] / relative
        if not path.is_file() or file_fingerprint(path) != digest:
            raise CandidateRunError(f"bound source changed: {relative}")
    _validate_start_state(run_dir, config); ledger = _validate_ledger(run_dir)
    exact_binding = bindings.get("exact_prefix", {})
    exact_files = run_dir / "inputs/exact_prefix"
    expected_exact_statuses = {
        "stage-d-anchor-v1": {"available", "unavailable"},
        "stage-d-schedule-window-v1": {"available", "unavailable"},
        "stage-e-prior-v1": {"not_applicable"},
        "stage-e-prior-cutoff-216-v1": {"not_applicable"},
    }
    if exact_binding.get("status") not in expected_exact_statuses[spec.name]:
        raise CandidateRunError("exact-prefix evidence role changed")
    if exact_binding.get("status") == "available":
        _validate_exact_prefix(exact_files)
        claims = exact_binding.get("files")
        expected_paths = {f"inputs/exact_prefix/{name}" for name in EXACT_PREFIX_HASHES}
        if not isinstance(claims, list) or not all(isinstance(claim, Mapping) for claim in claims) or {claim.get("path") for claim in claims} != expected_paths:
            raise CandidateRunError("exact-prefix copy inventory changed")
        for claim in claims:
            path = run_dir / claim["path"]
            if claim.get("size") != path.stat().st_size or claim.get("sha256") != file_fingerprint(path):
                raise CandidateRunError("exact-prefix copy binding changed")
    else:
        if exact_files.exists() and any(exact_files.iterdir()):
            raise CandidateRunError("unbound exact-prefix evidence is present")
        if exact_binding.get("files") != []:
            raise CandidateRunError("unavailable exact-prefix evidence has file claims")
        if spec.name in {"stage-d-anchor-v1", "stage-d-schedule-window-v1"} and not str(exact_binding.get("reason", "")).strip():
            raise CandidateRunError("unavailable exact-prefix evidence lacks a reason")
    stage_d_binding = bindings.get("stage_d_predecessor")
    schedule_baseline_binding = bindings.get("stage_d_schedule_baseline")
    schedule_predecessor_binding = bindings.get("stage_d_schedule_predecessor")
    if spec.name == "stage-e-prior-v1":
        if not isinstance(stage_d_binding, Mapping):
            raise CandidateRunError("Stage E lost its successful Stage D citation")
        stage_d_path = Path(str(stage_d_binding.get("run_dir", ""))).resolve()
        if stage_d_path == run_dir or stage_d_path in run_dir.parents or run_dir in stage_d_path.parents:
            raise CandidateRunError("Stage E predecessor locator overlaps its child")
        stage_d_result = verify_run(stage_d_path)
        if (
            stage_d_result["outcome"].get("stage_e_machine_eligible") != 1
            or file_fingerprint(stage_d_path / "artifact_manifest.json") != stage_d_binding.get("manifest_sha256")
            or file_fingerprint(stage_d_path / "outcome.json") != stage_d_binding.get("outcome_sha256")
            or file_fingerprint(stage_d_path / "reverse/first16_audit.json") != stage_d_binding.get("first16_audit_sha256")
        ):
            raise CandidateRunError("Stage D predecessor citation changed")
    elif stage_d_binding is not None:
        raise CandidateRunError("Stage D unexpectedly cites a predecessor")
    if spec.name == "stage-d-schedule-window-v1":
        _validate_schedule_baseline_copy(run_dir, schedule_baseline_binding)
    elif schedule_baseline_binding is not None:
        raise CandidateRunError("non-schedule run unexpectedly binds a schedule baseline")
    if spec.name == "stage-e-prior-cutoff-216-v1":
        _validate_stage_e_predecessor_copy(run_dir, schedule_predecessor_binding)
    elif schedule_predecessor_binding is not None:
        raise CandidateRunError("non-Stage-E-cutoff run unexpectedly binds a schedule predecessor")
    records, states = _scan_prefix(run_dir)
    backend_path = run_dir / "backend.json"
    if backend_path.exists():
        backend = _read_json(backend_path)
        if (
            backend.get("schema") != VERSION + "-candidate-backend"
            or not re.fullmatch(r"[0-9a-f]{64}", str(backend.get("candidate_binary_sha256", "")))
            or backend.get("candidate_modes") != spec.candidate_modes
            or backend.get("candidate_bisection_steps") != spec.candidate_bisection_steps
            or backend.get("exact_authorizer_authority") != 0
            or backend.get("synchronous_replay_authority") != 0
        ):
            raise CandidateRunError("candidate backend authority changed")
    elif records:
        raise CandidateRunError("committed candidate shards lack backend authority")
    return config, spec, source, bindings, ledger, records, states


def verify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve(); before = {path.relative_to(run_dir).as_posix(): (path.stat().st_size, file_fingerprint(path)) for path in run_dir.rglob("*") if path.is_file()}
    config, spec, source, _bindings, ledger, records, states = _validate_run_authority(run_dir)
    manifest = _verify_manifest(run_dir)
    trajectory = _npz(run_dir / "reverse/trajectory_boundaries.npz")
    expected_steps = np.arange(len(states), dtype=np.int64) * SHARD_STEPS; expected_states = np.ascontiguousarray(np.stack(states), dtype=np.float64)
    if set(trajectory) != {"completed_reverse_steps", "states"} or not np.array_equal(trajectory["completed_reverse_steps"], expected_steps) or not np.array_equal(trajectory["states"], expected_states):
        raise CandidateRunError("derived trajectory changed")
    expected_metrics = _metric_rows(spec, expected_steps.tolist(), states, source)
    with (run_dir / "reverse/metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); observed = list(reader); observed_fields = reader.fieldnames
    expected_fields = list(expected_metrics[0])
    if observed_fields != expected_fields or len(observed) != len(expected_metrics) or any(
        any(("" if expected[key] is None else str(expected[key])) != row.get(key) for key in expected_fields)
        for row, expected in zip(observed, expected_metrics, strict=True)
    ):
        raise CandidateRunError("derived metrics changed")
    identity = _schedule_identity(run_dir, records, states) if spec.name in {
        "stage-d-schedule-window-v1", "stage-e-prior-cutoff-216-v1",
    } else None
    expected_health = _health(records, states, identity)
    if _read_json(run_dir / "reverse/health.json") != expected_health:
        raise CandidateRunError("derived health changed")
    expected_audit = _first16_audit(run_dir, states, source) if len(states) >= 2 else {
        "schema": VERSION + "-first16-audit", "status": "not_reached",
        "first16_not_completely_off": None, "blocks_artifact_completion": 0,
    }
    if _read_json(run_dir / "reverse/first16_audit.json") != expected_audit:
        raise CandidateRunError("derived first-16 audit changed")
    expected_mechanism = _mechanism(spec, records, expected_metrics)
    if _read_json(run_dir / "reverse/mechanism.json") != expected_mechanism:
        raise CandidateRunError("derived mechanism changed")
    status = _read_json(run_dir / "status.json")
    terminal_state = status.get("state")
    expected_resumable = int(terminal_state in {"resource_paused", "interrupted", "derived_failed"})
    target_state = status.get("terminalization_target_state")
    target_error = status.get("terminalization_target_error")
    shard_seconds = math.fsum(float(record.get("elapsed_seconds", 0.0)) for record in records)
    execution_seconds = math.fsum(
        float(event["elapsed_seconds"]) for event in ledger["events"]
        if str(event["role"]).startswith("candidate_execution")
    )
    if (
        status.get("schema") != VERSION + "-status"
        or terminal_state not in {"complete", "resource_paused", "interrupted", "failed", "derived_failed"}
        or status.get("resumable") != expected_resumable
        or (terminal_state == "derived_failed" and target_state not in {"complete", "resource_paused", "interrupted", "failed"})
        or (terminal_state != "derived_failed" and (target_state is not None or target_error is not None))
        or (terminal_state == "complete" and len(records) != SHARD_COUNT)
        or (terminal_state in {"resource_paused", "interrupted"} and len(records) == SHARD_COUNT)
        or (terminal_state == "complete" and status.get("error") is not None)
        or (terminal_state != "complete" and not str(status.get("error", "")).strip())
        or status.get("committed_shards") != len(records)
        or status.get("completed_reverse_steps") != len(records) * SHARD_STEPS
        or status.get("active_seconds") != ledger["active_seconds"]
        or status.get("maximum_active_seconds") != ledger["maximum_active_seconds"]
        or status.get("committed_shard_seconds") != shard_seconds
        or status.get("candidate_setup_orchestration_seconds") != max(0.0, execution_seconds - shard_seconds)
    ):
        raise CandidateRunError("run status changed")
    expected_outcome = _outcome_record(records, expected_metrics, expected_health, expected_audit, str(status.get("state")), status.get("error"), spec.name)
    outcome = _read_json(run_dir / "outcome.json")
    if outcome != expected_outcome:
        raise CandidateRunError("derived outcome changed")
    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    if report != _report_text(run_dir, status, outcome, expected_health, expected_audit, config):
        raise CandidateRunError("human-readable report authority changed")
    for relative in re.findall(r"^- `([^`]+)`$", report, flags=re.MULTILINE):
        if not (run_dir / relative).is_file():
            raise CandidateRunError(f"report references an absent artifact: {relative}")
    from PIL import Image
    scale = fixed_rendering_scale(source.source_image, source.mixed_target, float(source.metadata["lambda_mix"]))
    selected = [value for value in spec.render_horizons if value in expected_steps]
    if int(expected_steps[-1]) not in selected:
        selected.append(int(expected_steps[-1]))
    expected_image_paths: set[str] = set()
    for completed in selected:
        state = states[int(completed // SHARD_STEPS)]
        for kind, renderer in (("raw", render_raw_density), ("demixed", render_background_demixed)):
            cells = []
            for index, key in enumerate(_row_order(spec)):
                path = run_dir / f"images/individual/{kind}/{key}/step-{completed:03d}.png"
                expected = renderer(state[index], scale); cells.append(expected)
                with Image.open(path) as image:
                    if not np.array_equal(np.asarray(image.convert("L")), expected):
                        raise CandidateRunError(f"rendered pixels changed: {path.relative_to(run_dir)}")
                expected_image_paths.add(path.relative_to(run_dir).as_posix())
            sheet = run_dir / f"images/contact_sheets/{kind}-step-{completed:03d}.png"
            with Image.open(sheet) as image:
                if not np.array_equal(np.asarray(image.convert("L")), np.concatenate(cells, axis=1)):
                    raise CandidateRunError(f"contact-sheet pixels changed: {sheet.relative_to(run_dir)}")
            expected_image_paths.add(sheet.relative_to(run_dir).as_posix())
    for name, expected in (("source", render_source_image(source.source_image, scale)), ("mixed-target", render_raw_density(source.mixed_target, scale))):
        path = run_dir / f"images/context/{name}.png"
        with Image.open(path) as image:
            if not np.array_equal(np.asarray(image.convert("L")), expected):
                raise CandidateRunError(f"context pixels changed: {path.relative_to(run_dir)}")
        expected_image_paths.add(path.relative_to(run_dir).as_posix())
    actual_image_paths = {path.relative_to(run_dir).as_posix() for path in (run_dir / "images").rglob("*") if path.is_file()}
    if actual_image_paths != expected_image_paths:
        raise CandidateRunError("image artifact path set changed")
    after = {path.relative_to(run_dir).as_posix(): (path.stat().st_size, file_fingerprint(path)) for path in run_dir.rglob("*") if path.is_file()}
    if before != after:
        raise CandidateRunError("read-only verification mutated the run")
    return {"passed": 1, "manifest": manifest, "outcome": outcome, "committed_shards": len(records)}


def _amend_cap(
    run_dir: Path, new_cap: float, reason: str, *, verified: bool = False,
    allowed_states: frozenset[str] = frozenset({"resource_paused"}),
) -> None:
    if not verified:
        verify_run(run_dir)
    status = _read_json(run_dir / "status.json"); ledger = _validate_ledger(run_dir)
    old = float(ledger["maximum_active_seconds"])
    approval = reason.strip()
    if (
        ledger.get("active_attempt") is not None or status.get("state") not in allowed_states
        or not math.isfinite(new_cap) or new_cap <= old or not approval
        or (approval.startswith("<") and approval.endswith(">"))
    ):
        raise CandidateRunError("cap amendment requires an allowed paused run, larger cap, and reason")
    ledger["maximum_active_seconds"] = float(new_cap)
    ledger["cap_history"].append({
        "old": old, "new": float(new_cap), "at": _utc_now(),
        "reason": "explicit_cap_amendment", "approval_reference": approval,
        "command": subprocess.list2cmdline(sys.argv),
    })
    atomic_write_json(run_dir / "resource_ledger.json", ledger); (run_dir / "artifact_manifest.json").unlink(missing_ok=True)


def _run_or_resume(run_dir: Path, device: str) -> int:
    try:
        state, error = _execute(run_dir, device)
    except Exception as exc:
        try:
            _terminalize_accounted(run_dir, "failed", str(exc))
        except Exception:
            pass
        raise
    _terminalize_accounted(run_dir, state, error)
    return 0 if state == "complete" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    for command, experiment in (
        ("run-anchor", "stage-d-anchor-v1"),
        ("run-prior", "stage-e-prior-v1"),
        ("run-schedule-window", "stage-d-schedule-window-v1"),
        ("run-prior-cutoff-216", "stage-e-prior-cutoff-216-v1"),
    ):
        item = commands.add_parser(command); item.set_defaults(experiment_name=experiment)
        item.add_argument("--reference-run-dir", required=True); item.add_argument("--runs-root", required=True); item.add_argument("--run-name", required=True)
        item.add_argument("--device", default="cuda:0"); item.add_argument("--maximum-active-seconds", required=True, type=float)
        item.add_argument("--approval-reference", required=True)
        if command in {"run-prior", "run-schedule-window", "run-prior-cutoff-216"}: item.add_argument("--stage-d-run-dir", required=True)
        else: item.set_defaults(stage_d_run_dir=None)
    resume = commands.add_parser("resume"); resume.add_argument("--run-dir", required=True); resume.add_argument("--device", default="cuda:0")
    resume.add_argument("--extend-maximum-active-seconds", type=float); resume.add_argument("--cap-amendment-reason")
    verify = commands.add_parser("verify"); verify.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"run-anchor", "run-prior", "run-schedule-window", "run-prior-cutoff-216"}:
        run_dir = _initialize_run(args, experiment_spec(args.experiment_name)); return _run_or_resume(run_dir, args.device)
    run_dir = Path(args.run_dir).resolve()
    if args.command == "verify":
        verify_run(run_dir); return 0
    _validate_run_authority(run_dir)
    status = _read_json(run_dir / "status.json"); prior_state = str(status.get("state"))
    if status.get("schema") != VERSION + "-status" or status.get("resumable") != 1 or prior_state not in {"initialized", "running", "interrupted", "resource_paused", "derived_failed"}:
        raise CandidateRunError("run status is not resumable")
    if prior_state == "derived_failed":
        target_state = status.get("terminalization_target_state")
        target_error = status.get("terminalization_target_error")
        if target_state not in {"complete", "resource_paused", "interrupted", "failed"}:
            raise CandidateRunError("derived-failure recovery target changed")
        _reconcile_attempt(run_dir)
        if args.extend_maximum_active_seconds is not None:
            _amend_cap(
                run_dir, float(args.extend_maximum_active_seconds), args.cap_amendment_reason or "",
                verified=True, allowed_states=frozenset({"derived_failed"}),
            )
        ledger = _validate_ledger(run_dir)
        if float(ledger["active_seconds"]) + TERMINAL_RESERVE_SECONDS >= float(ledger["maximum_active_seconds"]):
            raise CandidateRunError("derived-artifact recovery requires an explicit larger cap")
        (run_dir / "artifact_manifest.json").unlink(missing_ok=True)
        _terminalize_accounted(run_dir, str(target_state), target_error if isinstance(target_error, str) else None)
        return 0 if target_state == "complete" else 2
    if args.extend_maximum_active_seconds is not None:
        if prior_state != "resource_paused":
            raise CandidateRunError("only a resource-paused run may amend its cap")
        verify_run(run_dir)
        _amend_cap(run_dir, float(args.extend_maximum_active_seconds), args.cap_amendment_reason or "", verified=True)
    else:
        _reconcile_attempt(run_dir)
        ledger = _validate_ledger(run_dir)
        if float(ledger["active_seconds"]) + TERMINAL_RESERVE_SECONDS >= float(ledger["maximum_active_seconds"]):
            (run_dir / "artifact_manifest.json").unlink(missing_ok=True)
            _terminalize_accounted(
                run_dir, "resource_paused",
                "reconciled active time leaves no approved terminal reserve; explicit cap extension required",
            )
            return 2
        if prior_state == "resource_paused":
            status_cap = status.get("maximum_active_seconds")
            if not isinstance(status_cap, (int, float)) or not float(status_cap) < float(ledger["maximum_active_seconds"]):
                raise CandidateRunError("resource-paused resume requires an explicit cap amendment")
    (run_dir / "artifact_manifest.json").unlink(missing_ok=True)
    return _run_or_resume(run_dir, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
