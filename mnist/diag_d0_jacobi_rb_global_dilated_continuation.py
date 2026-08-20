"""Exact same-path 512-step continuation for the global-dilated controller.

This module is intentionally independent of the completed high-level v3 runner.
It authenticates that runner's immutable evidence, copies the load-bearing prefix
into a new child, and owns the remaining forward/reverse work and terminal bundle.

The experiment is exploratory: its independent unit is one already-opened path.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid
import numpy as np
import torch
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    v3_transitive_source_paths,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_global_dilated import (
    GLOBAL_DILATED_PARAMETER_COUNT,
    GlobalDilatedZeroBaselinePredictor,
    global_dilated_architecture_contract,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_COUNT,
    semantic_sha256,
    state_dict_sha256,
)
from mnist.d0_jacobi_rb_tangent_fused import (
    DeferredCertifiedFusedReference,
    FUSED_SHARD_PHASES,
    FUSED_TANGENT_VERSION,
    FusedRowSpec,
    FusedShardExecutionPlan,
    FusedTangentControllerBank,
    prepare_deferred_reference_rng_seed_map,
    run_fused_reverse_family,
)
from mnist.d0_jacobi_rb_tangent_rollout import (
    EXPLORATORY_REFERENCE_RNG_NAMESPACE,
    FixedRenderingScale,
    ScaledTangentScoreController,
    TargetFractionOracleController,
    TANGENT_ROLLOUT_VERSION,
    aggregate_exact_forward_shards,
    atomic_rollout_npz,
    fixed_rendering_scale,
    load_verified_source_target,
    paired_metric_improvement,
    raw_state_metrics,
    render_background_demixed,
    render_raw_density,
    reverse_suffix_sequence,
    rollout_array_sha256,
    rollout_file_sha256,
    save_png,
)
VERSION = "d0-jacobi-rb-global-dilated-continuation-v2"
V2_VERSION = "d0-jacobi-rb-global-dilated-continuation-v1"
RUN_SCHEMA = "experiment12-d0-jacobi-rb-global-dilated-continuation"
STAGES = ("prepare", "controls", "forward_tail", "reverse_complete", "report_verify")
ROW_ORDER = ("zero", "global-plus-1", "source-informed")
PARENT_SELECTED_UPDATE = 3100
PATH_ID = 1_028_864
FORWARD_ROOT_SEED = 261_401
REVERSE_ROOT_SEED = 261_402
OUTER_STEPS = 512
PHASES = 7
STATE_SIZE = 784
MICROSTEPS = 2
FORWARD_TRANSITION_COUNT = 1_404_928
PARENT_PREFIX_TRANSITION_COUNT = 351_232
REVERSE_TRANSITION_COUNT = 16_859_136
REVERSE_SHARD_TRANSITION_COUNT = 263_424
ACTIVE_SECONDS_CAP = 22_500.0
V2_ACTIVE_SECONDS_CAP = 21_600.0
STORAGE_CAP_BYTES = 2 * 1024**3
CUDA_MEMORY_FRACTION_CAP = 0.80
REPORT_RESERVE_SECONDS = 600.0
POSTPROCESS_RESERVE_SECONDS = 30.0
PRACTICAL_RELATIVE_THRESHOLD = 0.01
DEFAULT_PROFILE_SHA256 = "75ed39fcdc20bb8c675bf9321ae3b31b8fa409370f9d5620f3c9f5b75821fda4"
CHECKPOINT_FILE_SHA256 = "5831a950a979726bf7a648d4c276bdc13f032f17ad1bc739c5d73c25d4841d38"
CHECKPOINT_STATE_SHA256 = "1df9888bef6c63db10f41f89a58891321e058e55ed7d8b36622c9cdf9827a218"
SOURCE_JSON_SHA256 = "e4f6918a6bd9b01f36ebdebdcf262242dfa714e908af199bde47cb9e025591eb"
SOURCE_NPZ_SHA256 = "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
SOURCE_IMAGE_ARRAY_SHA256 = "9d1f95ff3487901dabcd9b8261e0241511d2191997cd9daa260e390a3bc26c96"
MIXED_TARGET_ARRAY_SHA256 = "14513cf5aa1aceda2bcff9befdd685297040ea33a3a773809e2b03d9401d5fc8"
STEP127_ARCHIVE_SHA256 = "aeaf61244943dd21759be50f11d72b328964e44d9e8b0e69e6c684b8e12a4f5e"
STEP127_STATE_SHA256 = "7ab18075fc1bb71e169bc2d5bbfd2aa6afe74c16039162a088191c8d17225c00"
STEP511_ARCHIVE_SHA256 = "64f577f93d33d0f8986271d2a55a87b025d9afce21b3fa75a2cd4edb9bc6c280"
STEP511_STATE_SHA256 = "16290bb00837c4def108f10615aa4555fe942583d9ee7c6902b128923ce3d0c9"
CALIBRATION_FILE_SHA256 = "73d62bb15d76e32ab41b4d66a69096bb485343ab5b19289fea11de774975d850"
PARENT_RUN_MANIFEST_SEMANTIC_SHA256 = "8c37b461dbad725f454d5f3f0a219c5329172bc1d27471ba9bb3cc468fc393c4"
PARENT_MANIFEST_FILE_SHA256 = "6542ff34926dc1298df15420f361fa6f36cbcf84475d9f7e71b5e0de6934eb7c"
PARENT_MANIFEST_SEMANTIC_SHA256 = "327885100381d72c996a552a7a543db2c45451c5acb40036d995bb52712ddf47"
PARENT_CHECKSUM_FILE_SHA256 = "bfd253f6b174dd9533d4d7f6a294af08ab54c685505df750727691ed3bdc1f49"
PARENT_VERIFICATION_SEMANTIC_SHA256 = "e11c761732608b2eb1fef3ff4a67c7ab8dfefbdf8edd13ff694d6d2b52a79578"
PARENT_STORAGE_SEMANTIC_SHA256 = "9d440f7fd679341a6c63e980b1fc6a32a349696fc2188ffb4683b0112f4aeec2"
PARENT_SOURCE_CLOSURE_SHA256 = "65d85cb4345a14fb8b4e442ff978c66bc77bd6bba7051f295706b82bac3a6014"
PARENT_OUTCOME_SEMANTIC_SHA256 = "cd5b2877b5fdf2082014751267d61eb7572aea54e97172f6bf96cc91fee464b4"
PARENT_BRANCH_SEMANTIC_SHA256 = "d7eeb155a73e6564a55f03605b3df2aa8f329db096ad9507cd6217c0ca41655b"
PARENT_TREE_SHA256 = "2790043c1363cf9f75b7c64bce0b2792c3afa8622b660d2a92f69107d511c452"
PARENT_REPORT_MARKER_FILE_SHA256 = "4d8a0221c25e17ce4285a8f2701b25489a6bdcdc25b042fa15e2c9ab91b0ceb6"
PARENT_REPORT_MARKER_SEMANTIC_SHA256 = "5a065d2384acb9ba74de34213c372db2f28b9aa61a346f5b1b498c6dff2216a4"
REVERSE_STREAM_ROLE = "global_dilated_positive_complete_exact"
FORWARD_SHARDS = 64
PARENT_FORWARD_PREFIX_SHARDS = 16
IMPORTED_FORWARD_SHARDS = 64
REVERSE_SHARDS = 64
IMPORTED_REVERSE_SHARDS = 1
REVERSE_BASELINE_SECONDS = 223.4172105359996
V2_CARRIED_ACTIVE_SECONDS = 1945.6831628999998
V2_PEAK_CUDA_BYTES = 46_834_176
V2_TOTAL_CUDA_BYTES = 8_546_484_224
V2_FILE_COUNT = 158
V2_EXACT_BYTES = 2_887_822
V2_MANIFEST_ROWS = 154
V2_CHECKSUM_ENTRIES = 155
V2_OPERATIONAL_COPY_COUNT = 136
V2_OPERATIONAL_COPY_BYTES = 2_654_708
MILESTONE_STEPS = (0, 128, 256, 384, 512)
CAPTURE_COORDINATES = {
    (384, 0): "completed-128",
    (256, 0): "completed-256",
    (128, 0): "completed-384",
    (0, 0): "completed-512",
}
COMPETING_HYPOTHESES = (
    "implementation_or_orientation_defect",
    "controller_integrator_or_interface_failure",
    "inadequate_global_architecture_or_parameterization",
    "gain_or_late_time_calibration_failure",
    "useful_short_horizon_signal_but_dynamically_negligible_complete_effect",
    "on_policy_distribution_shift",
    "terminal_reference_prior_mismatch",
    "proxy_gate_misaligned_with_trajectory_objective",
    "exact_backend_cost_or_discrepancy",
    "current_jacobi_rb_strategy_failure",
    "other_evidence_supported_explanation",
)

FROZEN26_SIZE = 387_813
FROZEN26_SHA256 = "2356ddb38d39e75689ca1193094fc9114660915933235dece67d0b8490e32351"
FROZEN27_SIZE = 387_863
FROZEN27_SHA256 = "9258ad5c49474250b7f150c26fe78fa9db892a602e17d085511d4e39391fd98d"
FROZEN27_DELTA = b'                "betas": list(TRAINING["betas"]),\n'
class ContinuationError(RuntimeError):
    """Base continuation error."""
class ContinuationArgumentError(ContinuationError):
    """Invalid or ambiguous command-line invocation."""
class ContinuationIntegrityError(ContinuationError):
    """An immutable authority, chain, or terminal invariant changed."""
class ParentBindingError(ContinuationIntegrityError):
    """The sealed v3 parent or its producer binding changed."""
class ResourceBoundaryError(ContinuationError):
    """A durable resource boundary was reached."""
class CompositionControlError(ContinuationError):
    """The complete source-informed control did not authorize interpretation."""
class TerminalRunError(ContinuationError):
    """A terminal child was opened through an execution mode."""
_FORBIDDEN_EXACT_COUNTS = (
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "clipping_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "projection_count",
    "renormalization_count",
    "nonfinite_count",
)
_FUSED_INVALID_FIELDS = (
    "input_invalid",
    "reference_fraction_invalid",
    "score_invalid",
    "logistic_shift_invalid",
    "state_invalid",
    "mass_invalid",
    "metadata_invalid",
    "reference_invalid_count",
    "reference_unauthorized_count",
)
_FUSED_PHASE_PREFIXES = (
    "reference_fraction_displacement",
    "control_fraction_displacement",
    "score",
    "logistic_shift",
)
_FUSED_CONTROLLER_INTEGER_FIELDS = (
    "call_count", "lane_count", "score_count", "movable_count",
    "already_equal_count", "zero_pair_mass_count", "zero_duration_count",
    "target_oracle_unreachable_boundary_count", "clipping_count",
    "floor_count", "projection_count", "nonfinite_score_count",
)
_FUSED_CONTROLLER_FLOAT_FIELDS = (
    "score_squared_sum", "score_maximum_absolute",
    "unscaled_score_squared_sum", "unscaled_score_maximum_absolute",
    "score_rms", "unscaled_score_rms",
)
_TERMINAL_EXCLUSIONS = (
    "SHA256SUMS.txt",
    "artifact_manifest.json",
    "stages/report_verify.json",
    "terminal_storage_authority.json",
    "verification.json",
)
_PARENT_EXPECTED_HASHES = {
    "run_manifest.json": (None, PARENT_RUN_MANIFEST_SEMANTIC_SHA256),
    "artifact_manifest.json": (PARENT_MANIFEST_FILE_SHA256, PARENT_MANIFEST_SEMANTIC_SHA256),
    "SHA256SUMS.txt": (PARENT_CHECKSUM_FILE_SHA256, None),
    "verification.json": (None, PARENT_VERIFICATION_SEMANTIC_SHA256),
    "terminal_storage_authority.json": (None, PARENT_STORAGE_SEMANTIC_SHA256),
    "outcome.json": (None, PARENT_OUTCOME_SEMANTIC_SHA256),
    "positive_branch.json": (None, PARENT_BRANCH_SEMANTIC_SHA256),
    "stages/report_verify.json": (PARENT_REPORT_MARKER_FILE_SHA256, PARENT_REPORT_MARKER_SEMANTIC_SHA256),
}
_V2_EXPECTED_HASHES = {
    "run_manifest.json": ("08229be1ba57a7327e22b40c50c3a7ee9d5f9a5b063d48576a5ff1c5ec7b15ac", "e2816cbd9b72061a674500fa52d351d9898cae12d86d4a3950bca75c24795ad4"),
    "scientific_config.json": ("12b3acab5b8c40b6a59b56b7f0eb9cfa6dcf5a61f37db6c219a21981e849c775", "c00e8f6f7b561ef272a8f0c170ef6e7a574dede892d6777f13c1dc395668ac34"),
    "continuation_freeze.json": ("15ce60fc00993d4ae6c5f747a40404dcfa361d3a344dbb583f1388761f20622c", "b89c3bd1efce4c3a671c6b2f0badb5246ca431e5cc824690284757970508856b"),
    "parent_binding.json": ("89482d2a0320dfedf047403f1716711c9b53f6c473e7a6c9c925251b689c2831", "79fd37775dd32da1ba9e1fc6189384faa732e693292f59dd48387ec1d383557e"),
    "input_bindings.json": ("d317e0e972666c08d3acb3cddf221366e589ff59ba1f3c521f9c4851d0a1ad83", "5b00d2f0a64fb34bf5037dd35038ff374eb990039d751367d4c38d4d8d21345b"),
    "forward/prefix_import.json": ("04f50aad1a09bd6076239df3ccae76ca64708ce0645737a42c188991ad8ce08d", "3bfb4570093e47fecc26eb997985670954299d49eb145aaffa68d143c3ff71a7"),
    "forward/forward_summary.json": ("bdece1fe4543786521e45b11bc6fe0f1364659a4fe6834b2d8ff9a2b4f9be4e7", "6282b5634373fe85a7cfeee639e5c6ecdc9b16e6968e3d6c67e50b21034400d6"),
    "forward/anchor-step-0511.npz": (STEP511_ARCHIVE_SHA256, None),
    "resource_ledger.json": ("47f9b491ac0e2616f1e03fe977a42cd0475bad80bf7e3e25501d7ccd1fba608e", "b559e0110178e51d6062f627c4bcc3f0d949294f20a902698a988910d7fb4335"),
    "reverse/fused_families/same-path-three-row/complete-512/shard-0000.json": ("96cc9d196652e0942c552f2c8c50c67b6e5a7c4f5059884727674d41a78f80a6", "ed0a7eafc240e4a8f117ff0c85b083da7c63db11edc3caa540b53c04ae2f6cf9"),
    "reverse/fused_families/same-path-three-row/complete-512/shard-0000.npz": ("34079822f4da88bf9bc269f38ac0ae65fbe16a28c4c09e2b56b3773ec6a5bbd9", None),
    "reverse/fused_families/same-path-three-row/complete-512/shard-0001.failure.json": ("62deb145057128f9b2ee62c2a5b1fe3faf7f586751c670221d34121be8c8d73d", "5b816613c4c75462c7579bda4b1b7ed19c16fbf2284ba366bde0d29336378b46"),
    "terminal_failure.json": ("c5e1cceaa2b9021d7484b1d177bf188d28e8c4387278125239bb15bccadc136f", "43ccb2315afa7fda391005805059404a80724a0328d21da1da8a248c5c7b8927"),
    "terminal_storage_authority.json": ("e7a7beb58aa2d3fa49fe9224aff5f46bae5b526d607d93e0311f53e7363e4948", "54db340a9ad49a2e679ebf8d5f6b75f3d92bce688e55e10c071612c24560ca24"),
    "verification.json": ("208f8a07efdeeceaf99e49db2960679c5eb0bacc44c3516c1f687b8eb85ecbe7", "5a01a4647bfbd6416f06d67b98fb7e9169ae7662ed5ec5e54187713588d32024"),
    "artifact_manifest.json": ("407b41946fbdec4f191ab8eb5467b1e952399023934cefbf683b38571fa828ee", "9f91a69117d89c873586e2b899ecbfb456d3cc93fe3e27e6a085fa3b3231c3a3"),
    "SHA256SUMS.txt": ("c7258bd91c3b3e89688e9451273231700e0f65d5c478dc1caecb399296296cfe", None),
}
_V2_CAPSULE_PATHS = (
    "run_manifest.json", "scientific_config.json", "continuation_freeze.json",
    "parent_binding.json", "input_bindings.json", "forward/prefix_import.json",
    "forward/forward_summary.json", "resource_ledger.json",
    "reverse/fused_families/same-path-three-row/complete-512/shard-0001.failure.json",
    "failure_capture.json", "last_valid_evidence.json", "terminal_failure.json",
    "terminal_storage_authority.json", "verification.json", "artifact_manifest.json",
    "SHA256SUMS.txt",
)
_PREPARED_BACKENDS: dict[str, Any] = {}
_PREPARED_SEEDS: dict[tuple[int, int, str], Mapping[str, Any]] = {}
_UNDEBITED_START: float | None = None
def _default_profile() -> JacobiRBCudaProfile:
    profile = JacobiRBCudaProfile()
    if semantic_sha256(profile.to_dict()) != DEFAULT_PROFILE_SHA256:
        raise ContinuationIntegrityError("default exact CUDA profile changed")
    return profile
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def _reject_links_and_reparse_points(root: Path) -> None:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ContinuationIntegrityError(f"evidence root is absent or linked: {root}")
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ContinuationIntegrityError(f"evidence tree contains a link: {path}")
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if attributes & reparse:
            raise ContinuationIntegrityError(f"evidence tree contains a reparse point: {path}")
def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
def _semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("semantic_sha256", None)
    body["semantic_sha256"] = semantic_sha256(body)
    return body
def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
def _write_semantic(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    body = _semantic(value)
    data = _canonical_json_bytes(body)
    if path.is_file() and path.read_bytes() == data:
        return body
    _write_bytes_atomic(path, data)
    return body
def _read_json(path: Path, *, semantic: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContinuationIntegrityError(f"could not read JSON authority: {path}") from exc
    if not isinstance(value, dict):
        raise ContinuationIntegrityError(f"JSON authority is not an object: {path}")
    if semantic:
        observed = value.get("semantic_sha256")
        if not isinstance(observed, str) or _semantic(value).get("semantic_sha256") != observed:
            raise ContinuationIntegrityError(f"semantic JSON hash changed: {path}")
    return value
def _directory_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())
def _snapshot_tree(root: Path) -> tuple[tuple[str, int, str], ...]:
    if not root.exists():
        return ()
    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ContinuationIntegrityError(f"tree contains a symbolic link: {path}")
        if path.is_file():
            rows.append((path.relative_to(root).as_posix(), int(path.stat().st_size), _sha256_file(path)))
    return tuple(rows)
def _tree_hash(rows: Iterable[tuple[str, int, str]]) -> str:
    return semantic_sha256(
        [{"path": path, "size": size, "sha256": digest} for path, size, digest in rows]
    )
def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ContinuationIntegrityError(f"cannot load NPZ evidence: {path}") from exc
def _verify_orphan_replay_matches(
    archived_path: Path, replayed_path: Path, *, expected_rows: int = 1
) -> dict[str, Any]:
    """Compare a deterministic NPZ replay without modifying either byte stream."""

    archived_path = Path(archived_path)
    replayed_path = Path(replayed_path)
    archived_bytes = archived_path.read_bytes()
    replayed_bytes = replayed_path.read_bytes()
    archived = _load_npz(archived_path)
    replayed = _load_npz(replayed_path)
    if set(archived) != {"state"} or set(replayed) != {"state"}:
        raise ContinuationIntegrityError("orphan replay archive schema changed")
    left = archived["state"]
    right = replayed["state"]
    expected_shape = (int(expected_rows), STATE_SIZE)
    if (
        left.dtype != np.float64
        or right.dtype != np.float64
        or left.shape != expected_shape
        or right.shape != expected_shape
        or rollout_array_sha256(left) != rollout_array_sha256(right)
    ):
        if archived_path.read_bytes() != archived_bytes or replayed_path.read_bytes() != replayed_bytes:
            raise ContinuationIntegrityError("orphan comparison mutated evidence")
        raise ContinuationIntegrityError("orphan deterministic replay changed")
    return {
        "passed": 1,
        "archived_file_sha256": _sha256_bytes(archived_bytes),
        "replayed_file_sha256": _sha256_bytes(replayed_bytes),
        "state_sha256": rollout_array_sha256(left),
    }
def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ContinuationIntegrityError("cannot write an empty metric table")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(normalized[0]))
    writer.writeheader()
    writer.writerows(normalized)
    _write_bytes_atomic(path, output.getvalue().encode("utf-8"))
def _source_revision(repository_root: Path) -> dict[str, Any]:
    def command(*items: str) -> str:
        try:
            return subprocess.run(
                list(items), cwd=repository_root, check=True, capture_output=True,
                text=True, encoding="utf-8",
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = command("git", "status", "--porcelain")
    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": int(bool(status and status != "unavailable")),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
    }
def _current_source_closure(repository_root: Path) -> tuple[dict[str, Any], str]:
    entry = Path(__file__).resolve()
    rows: dict[str, Any] = {}
    for path in v3_transitive_source_paths((entry,)):
        relative = path.relative_to(repository_root).as_posix()
        rows[relative] = {"size": int(path.stat().st_size), "sha256": _sha256_file(path)}
    return rows, semantic_sha256(rows)
def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default=None)
    parser.add_argument("--repository-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--verify-run-dir", type=Path, default=None)
    parser.add_argument("--prefix-run-dir", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser
def _resolve_mode(args: argparse.Namespace) -> str:
    if args.verify_run_dir is not None:
        return "verify"
    if args.resume_run_dir is not None:
        return "resume"
    if args.runs_root is not None or args.run_name is not None:
        return "fresh"
    raise ContinuationArgumentError("exactly one of fresh, resume, or verify mode is required")
def _validate_cli_combination(args: argparse.Namespace, *, mode: str) -> None:
    if mode == "verify":
        forbidden = (args.stage, args.device, args.runs_root, args.run_name, args.resume_run_dir)
        if any(value is not None for value in forbidden):
            raise ContinuationArgumentError("verify mode forbids stage/device/run creation/resume options")
    elif mode == "resume":
        if args.verify_run_dir is not None or args.runs_root is not None or args.run_name is not None:
            raise ContinuationArgumentError("resume mode cannot create or verify a different run")
        if args.stage is None or args.device is None:
            raise ContinuationArgumentError("resume mode requires --stage and --device")
    elif mode == "fresh":
        if args.verify_run_dir is not None or args.resume_run_dir is not None:
            raise ContinuationArgumentError("fresh mode cannot resume or verify")
        if args.runs_root is None or not args.run_name or args.stage not in ("prepare", "all"):
            raise ContinuationArgumentError("fresh mode requires runs-root, run-name, and stage prepare/all")
        if args.device is None:
            raise ContinuationArgumentError("fresh mode requires --device")
        if (
            not isinstance(args.run_name, str)
            or args.run_name in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_name) is None
        ):
            raise ContinuationArgumentError("run-name must be a safe ASCII stem")
    else:
        raise ContinuationArgumentError(f"unknown mode: {mode}")
    if mode != "verify" and args.stage != "prepare" and args.device != "cuda":
        raise ContinuationArgumentError("sampling-capable continuation stages require CUDA")
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_combination(args, mode=_resolve_mode(args))
    except ContinuationArgumentError as exc:
        parser.error(str(exc))
    return args
def _paths_overlap(left: Path, right: Path) -> bool:
    left = Path(left).resolve()
    right = Path(right).resolve()
    return left == right or left in right.parents or right in left.parents
def _resolve_paths(args: argparse.Namespace, *, mode: str) -> SimpleNamespace:
    repository_root = (args.repository_root or Path.cwd()).resolve()
    actual_repository_root = Path(__file__).resolve().parents[1]
    if (
        repository_root != actual_repository_root
        or not repository_root.is_dir()
        or (repository_root / Path(__file__).resolve().relative_to(actual_repository_root)).resolve()
        != Path(__file__).resolve()
    ):
        raise ContinuationArgumentError("repository-root is not this continuation checkout")
    if mode == "fresh":
        runs_root = args.runs_root.resolve()
        run_dir = (runs_root / args.run_name).resolve()
        if run_dir.parent != runs_root:
            raise ContinuationArgumentError("run-name escapes the runs root")
    elif mode == "resume":
        run_dir = args.resume_run_dir.resolve()
    else:
        run_dir = args.verify_run_dir.resolve()
    prefix_run_dir = args.prefix_run_dir.resolve()
    parent_run_dir = args.parent_run_dir.resolve()
    source_run_dir = args.source_run_dir.resolve()
    external = (prefix_run_dir, parent_run_dir, source_run_dir)
    try:
        for root in external:
            _reject_links_and_reparse_points(root)
    except ContinuationIntegrityError as exc:
        raise ContinuationArgumentError(str(exc)) from exc
    roots = (run_dir, *external)
    if any(_paths_overlap(left, right) for index, left in enumerate(roots) for right in roots[index + 1 :]):
        raise ContinuationArgumentError("child, prefix, parent, and source trees must be pairwise disjoint")
    return SimpleNamespace(
        repository_root=repository_root,
        run_dir=run_dir,
        prefix_run_dir=prefix_run_dir,
        parent_run_dir=parent_run_dir,
        source_run_dir=source_run_dir,
    )
def _canonical_fresh_command(
    *, repository_root: Path, run_dir: Path, prefix_run_dir: Path,
    parent_run_dir: Path, source_run_dir: Path
) -> bytes:
    repository_root = Path(repository_root).resolve()
    run_dir = Path(run_dir).resolve()
    return (
        "python -m mnist.diag_d0_jacobi_rb_global_dilated_continuation "
        f"--stage all --repository-root {repository_root} "
        f"--runs-root {run_dir.parent} --run-name {run_dir.name} "
        f"--prefix-run-dir {Path(prefix_run_dir).resolve()} "
        f"--parent-run-dir {Path(parent_run_dir).resolve()} "
        f"--source-run-dir {Path(source_run_dir).resolve()} --device cuda\n"
    ).encode("utf-8")
def _initial_resource_ledger() -> dict[str, Any]:
    return _semantic(
        {
            "schema": VERSION + "-resource-ledger",
            "schema_version": 1,
            "caps": {
                "active_seconds": ACTIVE_SECONDS_CAP,
                "storage_bytes": STORAGE_CAP_BYTES,
                "cuda_fraction": CUDA_MEMORY_FRACTION_CAP,
                "prepaid_report_seconds": REPORT_RESERVE_SECONDS,
            },
            "active_seconds": 0.0,
            "persisted_storage_bytes": 0,
            "peak_cuda_bytes": 0,
            "total_cuda_bytes": 0,
            "events": [],
            "limits_passed": 1,
            "breached_limits": [],
        }
    )

_RESOURCE_BREACH_ORDER = ("active_seconds", "persisted_storage", "cuda_memory", "report_deadline")
_RESOURCE_LEDGER_KEYS = {
    "schema", "schema_version", "caps", "active_seconds", "persisted_storage_bytes",
    "peak_cuda_bytes", "total_cuda_bytes", "events", "limits_passed",
    "breached_limits", "semantic_sha256",
}
def _resource_caps(*, active_seconds: float = ACTIVE_SECONDS_CAP) -> dict[str, float | int]:
    return {
        "active_seconds": float(active_seconds), "storage_bytes": STORAGE_CAP_BYTES,
        "cuda_fraction": CUDA_MEMORY_FRACTION_CAP,
        "prepaid_report_seconds": REPORT_RESERVE_SECONDS,
    }
def _resource_breaches(
    active: float, persisted: int, peak: int, total: int, prior: Iterable[str] = (),
    *, caps: Mapping[str, Any] | None = None,
) -> list[str]:
    bound = dict(caps or _resource_caps())
    found = set(prior)
    if active > float(bound["active_seconds"]): found.add("active_seconds")
    if persisted >= int(bound["storage_bytes"]): found.add("persisted_storage")
    if total and peak / total >= float(bound["cuda_fraction"]): found.add("cuda_memory")
    return [name for name in _RESOURCE_BREACH_ORDER if name in found]
def _resource_run_identity(run_dir: Path) -> str:
    manifest = Path(run_dir) / "run_manifest.json"
    return str(_read_json(manifest, semantic=True)["semantic_sha256"]) if manifest.is_file() else str(Path(run_dir).resolve())
def _validate_resource_ledger(
    run_dir: Path, ledger: Mapping[str, Any] | None = None, *,
    expected_version: str = VERSION,
    expected_caps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the full resource-event ledger and its cumulative authorities."""

    run_dir = Path(run_dir)
    value = dict(ledger) if ledger is not None else _read_json(run_dir / "resource_ledger.json", semantic=True)
    caps = dict(expected_caps or _resource_caps())
    if set(value) != _RESOURCE_LEDGER_KEYS or value.get("schema") != expected_version + "-resource-ledger" or value.get("schema_version") != 1 or value.get("caps") != caps or _semantic(value) != value:
        raise ContinuationIntegrityError("resource ledger schema/caps changed")
    def integer(raw: Any, name: str) -> int:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)) or int(raw) < 0:
            raise ContinuationIntegrityError(f"resource ledger {name} is invalid")
        return int(raw)
    def number(raw: Any, name: str) -> float:
        if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)) or not math.isfinite(float(raw)) or float(raw) < 0.0:
            raise ContinuationIntegrityError(f"resource ledger {name} is invalid")
        return float(raw)

    events = value.get("events")
    if not isinstance(events, list): raise ContinuationIntegrityError("resource events are not a list")
    active_parts: list[float] = []; peak = total = 0; ids: set[str] = set(); prior: list[str] = []
    run_identity = _resource_run_identity(run_dir)
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping): raise ContinuationIntegrityError("resource event is malformed")
        event = dict(raw); role = event.get("role"); attempt = integer(event.get("attempt"), "event.attempt")
        if attempt < 1 or not isinstance(role, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,95}", role) is None or _semantic(event) != event:
            raise ContinuationIntegrityError("resource event identity/schema changed")
        abandoned = role.endswith("_abandoned_attempt"); base_role = role.removesuffix("_abandoned_attempt")
        journal_id = event.get("journal_id"); expected_journal = _sha256_bytes(f"{run_identity}\0{base_role}\0{attempt}".encode("utf-8"))
        expected_event = _sha256_bytes(f"abandoned\0{expected_journal}".encode("utf-8")) if abandoned else expected_journal
        expected_keys = {
            "schema", "schema_version", "event_id", "journal_id", "role", "attempt",
            "elapsed_seconds", "failed", "peak_cuda_bytes", "total_cuda_bytes", "detail",
            "breaches", "limits_passed", "semantic_sha256",
            "reconciled_at" if abandoned else "recorded_at",
        }
        if abandoned: expected_keys.add("durable_attempt_id")
        event_id = event.get("event_id")
        if set(event) != expected_keys or event.get("schema") != expected_version + "-resource-event" or event.get("schema_version") != 1 or journal_id != expected_journal or event_id != expected_event or event_id in ids or (abandoned and event.get("durable_attempt_id") != journal_id):
            raise ContinuationIntegrityError(f"resource event {index} binding changed")
        ids.add(str(event_id)); elapsed = number(event.get("elapsed_seconds"), "elapsed_seconds")
        event_peak = integer(event.get("peak_cuda_bytes"), "event.peak"); event_total = integer(event.get("total_cuda_bytes"), "event.total")
        breaches = event.get("breaches")
        if (event_total == 0 and event_peak != 0) or (event_total and event_peak > event_total) or event.get("failed") not in (0, 1) or not isinstance(event.get("detail"), Mapping) or not isinstance(breaches, list) or breaches != [name for name in _RESOURCE_BREACH_ORDER if name in set(breaches)] or not set(prior).issubset(breaches) or event.get("limits_passed") != int(not breaches):
            raise ContinuationIntegrityError(f"resource event {index} telemetry changed")
        stamp = event["reconciled_at" if abandoned else "recorded_at"]
        try: parsed = dt.datetime.fromisoformat(str(stamp))
        except (TypeError, ValueError) as exc: raise ContinuationIntegrityError("resource event timestamp changed") from exc
        if parsed.tzinfo is None: raise ContinuationIntegrityError("resource event timestamp lacks timezone")
        active_parts.append(elapsed); peak = max(peak, event_peak); total = max(total, event_total)
        required = _resource_breaches(math.fsum(active_parts), 0, peak, total, prior, caps=caps)
        if not set(required).issubset(breaches): raise ContinuationIntegrityError("resource event omits a cumulative breach")
        prior = list(breaches)
    active = number(value.get("active_seconds"), "active_seconds"); persisted = integer(value.get("persisted_storage_bytes"), "persisted_storage")
    ledger_peak = integer(value.get("peak_cuda_bytes"), "peak_cuda"); ledger_total = integer(value.get("total_cuda_bytes"), "total_cuda")
    measured = _directory_bytes(run_dir)
    breaches = value.get("breached_limits")
    expected_breaches = _resource_breaches(active, persisted, ledger_peak, ledger_total, prior, caps=caps)
    if active != math.fsum(active_parts) or ledger_peak != peak or ledger_total != total or (ledger_total == 0 and ledger_peak != 0) or (ledger_total and ledger_peak > ledger_total) or persisted > measured + 65_536 or not isinstance(breaches, list) or breaches != expected_breaches or value.get("limits_passed") != int(not breaches):
        raise ContinuationIntegrityError("resource ledger cumulative authority changed")
    return value
def _converge_resource_storage(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "resource_ledger.json"
    for _ in range(8):
        ledger = _read_json(path, semantic=True); measured = _directory_bytes(Path(run_dir))
        if ledger.get("persisted_storage_bytes") == measured: return _validate_resource_ledger(Path(run_dir), ledger)
        breaches = _resource_breaches(float(ledger["active_seconds"]), measured, int(ledger["peak_cuda_bytes"]), int(ledger["total_cuda_bytes"]), ledger["breached_limits"], caps=ledger["caps"])
        events = list(ledger["events"])
        if events: events[-1] = _semantic({**events[-1], "breaches": breaches, "limits_passed": int(not breaches)})
        ledger.update(events=events, persisted_storage_bytes=measured, breached_limits=breaches, limits_passed=int(not breaches)); _write_semantic(path, ledger)
    raise ContinuationIntegrityError("resource-ledger storage fixed point did not converge")
def _scientific_config() -> dict[str, Any]:
    sequence = tuple(reverse_suffix_sequence(511))
    profile = _default_profile()
    return _semantic(
        {
            "schema": VERSION + "-scientific-config",
            "schema_version": 1,
            "runner": VERSION,
            "run_schema": RUN_SCHEMA,
            "stage_order": list(STAGES),
            "research_mode": "exploratory",
            "independent_unit": "one already-opened path/image",
            "program_objective": "DDPM-like MNIST generator based on the Eulerian approximation",
            "decision": "does the global +1 advantage survive the exact same-path complete reverse composition",
            "competing_hypotheses": list(COMPETING_HYPOTHESES),
            "claim_boundaries": {
                "one_path_scope": 1,
                "confirmatory_claim": 0,
                "population_inference": 0,
                "reference_prior_tested": 0,
                "protected_confirmation_opened": 0,
            },
            "parent": {
                "selected_update": PARENT_SELECTED_UPDATE,
                "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
                "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
                "source_closure_sha256": PARENT_SOURCE_CLOSURE_SHA256,
            },
            "path_id": PATH_ID,
            "forward_root_seed": FORWARD_ROOT_SEED,
            "reverse_root_seed": REVERSE_ROOT_SEED,
            "reverse_stream_role": REVERSE_STREAM_ROLE,
            "label": 3,
            "microsteps": MICROSTEPS,
            "phase_count": PHASE_COUNT,
            "rows": list(ROW_ORDER),
            "row_definitions": [
                {"key": "zero", "controller": "zero", "gain": None},
                {"key": "global-plus-1", "controller": "global-dilated", "gain": 1.0},
                {"key": "source-informed", "controller": "mixed-target-fraction", "gain": None},
            ],
            "outer_steps": OUTER_STEPS,
            "imported_forward_shards": IMPORTED_FORWARD_SHARDS,
            "generated_forward_shards": 0,
            "imported_reverse_shards": IMPORTED_REVERSE_SHARDS,
            "generated_reverse_shards": REVERSE_SHARDS - IMPORTED_REVERSE_SHARDS,
            "reverse_shards": REVERSE_SHARDS,
            "reverse_sequence_sha256": semantic_sha256([list(item) for item in sequence]),
            "capture_coordinates": [
                {"coordinate": list(coordinate), "name": name}
                for coordinate, name in CAPTURE_COORDINATES.items()
            ],
            "milestones": list(MILESTONE_STEPS),
            "exact_backend": "certified_exact",
            "default_profile": profile.to_dict(),
            "default_profile_sha256": DEFAULT_PROFILE_SHA256,
            "practical_relative_threshold": PRACTICAL_RELATIVE_THRESHOLD,
            "source_control_minimum_relative_improvement": PRACTICAL_RELATIVE_THRESHOLD,
            "resource_caps": {
                "active_seconds": ACTIVE_SECONDS_CAP,
                "storage_bytes": STORAGE_CAP_BYTES,
                "cuda_fraction": CUDA_MEMORY_FRACTION_CAP,
                "report_reserve_seconds": REPORT_RESERVE_SECONDS,
            },
            "retraining_performed": 0,
            "reselection_performed": 0,
            "new_path_allocated": 0,
            "tuning_performed": 0,
        }
    )
def _initialize_child_atomically(args: argparse.Namespace, *, paths: SimpleNamespace) -> Path:
    run_dir = Path(paths.run_dir)
    if run_dir.exists() or (run_dir.parent / f".{run_dir.name}.resume-probe.json").exists():
        raise FileExistsError(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_dir.parent / f".{run_dir.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_semantic(temporary / "resource_ledger.json", _initial_resource_ledger())
        _write_semantic(temporary / "scientific_config.json", _scientific_config())
        closure, closure_sha = _current_source_closure(Path(paths.repository_root))
        command = _canonical_fresh_command(
            repository_root=paths.repository_root,
            run_dir=run_dir,
            prefix_run_dir=paths.prefix_run_dir,
            parent_run_dir=paths.parent_run_dir,
            source_run_dir=paths.source_run_dir,
        )
        _write_bytes_atomic(temporary / "exact_command.txt", command)
        _write_semantic(
            temporary / "run_manifest.json",
            {
                "schema": RUN_SCHEMA,
                "schema_version": 1,
                "created_at": _utc_now(),
                "run_dir": str(run_dir),
                "prefix_run_dir": str(paths.prefix_run_dir),
                "parent_run_dir": str(paths.parent_run_dir),
                "source_run_dir": str(paths.source_run_dir),
                "source_revision": _source_revision(Path(paths.repository_root)),
                "source_closure": closure,
                "source_closure_sha256": closure_sha,
                "scientific_config_file_sha256": _sha256_file(temporary / "scientific_config.json"),
                "exact_command_file_sha256": _sha256_file(temporary / "exact_command.txt"),
            },
        )
        os.replace(temporary, run_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return run_dir
def _audit_frozen26_producer_read_only(path: Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    occurrence_count = data.count(FROZEN27_DELTA)
    if occurrence_count != 1:
        raise ParentBindingError("FROZEN27 producer delta occurrence count changed")
    current_sha = _sha256_bytes(data)
    reconstructed = data.replace(FROZEN27_DELTA, b"", 1)
    reconstructed_sha = _sha256_bytes(reconstructed)
    if len(data) != FROZEN27_SIZE or current_sha != FROZEN27_SHA256:
        raise ParentBindingError("current FROZEN27 producer bytes changed")
    if len(reconstructed) != FROZEN26_SIZE or reconstructed_sha != FROZEN26_SHA256:
        raise ParentBindingError("in-memory FROZEN26 reconstruction changed")
    return {
        "passed": 1,
        "occurrence_count": occurrence_count,
        "current_size": len(data),
        "current_sha256": current_sha,
        "reconstructed_size": len(reconstructed),
        "reconstructed_sha256": reconstructed_sha,
    }
def _copy_bound_file(source: Path, destination: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    source = Path(source)
    destination = Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ContinuationIntegrityError(f"copy source is not an independent regular file: {source}")
    source_hash = _sha256_file(source)
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ContinuationIntegrityError(f"copy source hash changed: {source}")
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != source_hash:
            raise ContinuationIntegrityError(f"existing child copy differs: {destination}")
        if os.path.samestat(source.stat(), destination.stat()):
            raise ContinuationIntegrityError("child input is not an independent copy")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            shutil.copyfile(source, temporary)
            if _sha256_file(temporary) != source_hash:
                raise ContinuationIntegrityError("temporary copy hash changed")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {
        "source": str(source),
        "destination": str(destination),
        "size": int(destination.stat().st_size),
        "sha256": source_hash,
        "samefile": int(os.path.samestat(source.stat(), destination.stat())),
    }
def _archive_orphan_before_replay(
    orphan_path: Path,
    recovery_root: Path,
    *,
    role: str,
    shard_index: int,
    attempt_id: str,
) -> dict[str, Any]:
    orphan_path = Path(orphan_path)
    recovery_root = Path(recovery_root)
    if orphan_path.is_symlink() or not orphan_path.is_file():
        raise ContinuationIntegrityError(f"orphan evidence is not a regular file: {orphan_path}")
    original_hash = _sha256_file(orphan_path)
    suffixes = "".join(orphan_path.suffixes) or ".bin"
    archive = recovery_root / role / f"shard-{shard_index:04d}.{attempt_id}.original{suffixes}"
    _copy_bound_file(orphan_path, archive, expected_sha256=original_hash)
    try:
        run_root = recovery_root.parents[1]
        archive_relative = archive.relative_to(run_root).as_posix()
    except (IndexError, ValueError):
        archive_relative = archive.as_posix()
    return _semantic(
        {
            "schema": VERSION + "-orphan-recovery",
            "schema_version": 1,
            "role": role,
            "shard_index": int(shard_index),
            "attempt_id": attempt_id,
            "original_path": str(orphan_path),
            "archive_relative_path": archive_relative,
            "original_size": int(orphan_path.stat().st_size),
            "original_sha256": original_hash,
        }
    )
def _classify_complete_outcome(
    *,
    zero_error: float,
    global_error: float,
    source_error: float,
    intermediate: Mapping[int, Mapping[str, float]],
) -> dict[str, Any]:
    finite = all(math.isfinite(float(value)) for value in (zero_error, global_error, source_error))
    if not finite or float(zero_error) <= 0.0:
        source_label = "invalid_objective"
        source_relative: float | None = None
        authorized = 0
    else:
        source_relative = (float(zero_error) - float(source_error)) / float(zero_error)
        if source_relative >= PRACTICAL_RELATIVE_THRESHOLD:
            source_label, authorized = "source_informative", 1
        elif source_relative > 0.0:
            source_label, authorized = "source_positive_small_uninformative", 0
        else:
            source_label, authorized = "source_adverse", 0
    global_relative: float | None = (
        (float(zero_error) - float(global_error)) / float(zero_error)
        if finite and float(zero_error) > 0.0
        else None
    )
    material_intermediate: list[int] = []
    intermediate_effects: list[dict[str, Any]] = []
    for step, values in intermediate.items():
        z = float(values.get("zero_error", math.nan))
        g = float(values.get("global_error", math.nan))
        relative = (z - g) / z if math.isfinite(z) and math.isfinite(g) and z > 0.0 else None
        intermediate_effects.append(
            {
                "completed_reverse_steps": int(step),
                "zero_error": z if math.isfinite(z) else None,
                "global_error": g if math.isfinite(g) else None,
                "paired_delta": z - g if math.isfinite(z) and math.isfinite(g) else None,
                "relative_improvement": relative,
                "material_at_one_percent": int(
                    relative is not None and relative >= PRACTICAL_RELATIVE_THRESHOLD
                ),
            }
        )
        if relative is not None and relative >= PRACTICAL_RELATIVE_THRESHOLD:
            material_intermediate.append(int(step))
    if global_relative is None or not math.isfinite(global_relative):
        global_label = "invalid_objective"
        base_action = "seal_invalid_objective_failure"
    elif global_relative >= PRACTICAL_RELATIVE_THRESHOLD:
        global_label = "global_material_improvement"
        base_action = "run_stage_e_reference_prior"
    elif global_relative > 0.0:
        global_label = "global_positive_small"
        base_action = "run_one_new_independent_path_replication"
    elif material_intermediate:
        global_label = "global_early_help_late_adverse"
        base_action = "run_predeclared_time_window_schedule_ablation"
    else:
        global_label = "global_complete_adverse"
        base_action = "run_conventional_mnist_ddpm_reconstruction_sanity"
    action = base_action if authorized else (
        "audit_oracle_controller_composition" if source_label == "source_positive_small_uninformative" else
        "fix_complete_controller_composition" if source_label == "source_adverse" else
        "seal_invalid_objective_failure"
    )
    return _semantic(
        {
            "schema": VERSION + "-outcome",
            "schema_version": 1,
            "zero_error": float(zero_error) if math.isfinite(float(zero_error)) else None,
            "global_error": float(global_error) if math.isfinite(float(global_error)) else None,
            "source_error": float(source_error) if math.isfinite(float(source_error)) else None,
            "global_relative_improvement": global_relative,
            "source_relative_improvement": source_relative,
            "global_paired_delta": (
                float(zero_error) - float(global_error) if finite else None
            ),
            "source_paired_delta": (
                float(zero_error) - float(source_error) if finite else None
            ),
            "practical_relative_threshold": PRACTICAL_RELATIVE_THRESHOLD,
            "intermediate_effects": intermediate_effects,
            "source_effect_label": source_label,
            "global_effect_label": global_label,
            "horizon_classification": global_label,
            "material_intermediate_steps": material_intermediate,
            "learned_interpretation_authorized": authorized,
            "required_next_action": action,
        }
    )

class _Projection(SimpleNamespace):
    projected_active_seconds: float
    admitted: bool
def _resource_projection(
    *,
    active_seconds: float,
    current_attempt_wall: float,
    remaining_forward_shards: int,
    remaining_reverse_shards: int,
    forward_next_seconds: float,
    reverse_next_seconds: float,
    postprocess_seconds: float,
    report_seconds: float,
    active_seconds_cap: float = ACTIVE_SECONDS_CAP,
) -> _Projection:
    projected = math.fsum(
        (
            float(active_seconds),
            float(current_attempt_wall),
            int(remaining_forward_shards) * float(forward_next_seconds),
            int(remaining_reverse_shards) * float(reverse_next_seconds),
            float(postprocess_seconds),
            float(report_seconds),
        )
    )
    return _Projection(projected_active_seconds=projected, admitted=projected <= float(active_seconds_cap))
def _prepaid_report_admission(*, active_seconds: float, now_monotonic: float, active_seconds_cap: float = ACTIVE_SECONDS_CAP) -> dict[str, Any]:
    after = math.fsum((float(active_seconds), REPORT_RESERVE_SECONDS))
    admitted = after <= float(active_seconds_cap)
    return _semantic(
        {
            "schema": VERSION + "-report-admission",
            "schema_version": 1,
            "admitted": int(admitted),
            "charged_seconds": REPORT_RESERVE_SECONDS if admitted else 0.0,
            "active_seconds_before_charge": float(active_seconds),
            "active_seconds_after_charge": after if admitted else float(active_seconds),
            "started_monotonic": float(now_monotonic),
            "deadline_monotonic": float(now_monotonic) + REPORT_RESERVE_SECONDS,
        }
    )
def _require_report_deadline(admission: Mapping[str, Any], *, now_monotonic: float) -> None:
    if int(admission.get("admitted", 0)) != 1 or float(now_monotonic) > float(admission["deadline_monotonic"]):
        raise ResourceBoundaryError("report reserve deadline exceeded")
def _record_report_overrun(run_dir: Path) -> None:
    path = Path(run_dir) / "report_overrun.json"
    if not path.is_file(): _write_semantic(path, {"schema": VERSION + "-report-overrun", "schema_version": 1, "resource_boundary": "prepaid_report_deadline", "observed_at": _utc_now()})
    ledger_path = Path(run_dir) / "resource_ledger.json"; ledger = _validate_resource_ledger(Path(run_dir)); breaches = _resource_breaches(float(ledger["active_seconds"]), _directory_bytes(Path(run_dir)), int(ledger["peak_cuda_bytes"]), int(ledger["total_cuda_bytes"]), (*ledger["breached_limits"], "report_deadline"), caps=ledger["caps"])
    events = list(ledger["events"])
    if not events: return
    events[-1] = _semantic({**events[-1], "breaches": breaches, "limits_passed": 0})
    ledger.update(events=events, breached_limits=breaches, limits_passed=0); _write_semantic(ledger_path, ledger); _converge_resource_storage(Path(run_dir))
def _enforce_report_deadline(run_dir: Path, admission: Mapping[str, Any]) -> None:
    try: _require_report_deadline(admission, now_monotonic=time.perf_counter())
    except ResourceBoundaryError: _record_report_overrun(run_dir); raise
def _journal_path(run_dir: Path, attempt_id: str) -> Path:
    return Path(run_dir) / "journals" / f"{attempt_id}.json"
def _resume_probe_binding(run_dir: Path, identity: Mapping[str, Any], prefix_run_dir: Path, parent_run_dir: Path, source_run_dir: Path) -> dict[str, Any]:
    return {"schema": VERSION + "-resume-probe", "schema_version": 1, "run_dir": str(Path(run_dir).resolve()), "prefix_run_dir": str(Path(prefix_run_dir).resolve()), "parent_run_dir": str(Path(parent_run_dir).resolve()), "source_run_dir": str(Path(source_run_dir).resolve()), "run_manifest_semantic_sha256": identity.get("run_manifest_semantic_sha256"), "scientific_config_semantic_sha256": identity.get("scientific_config_semantic_sha256"), "source_closure_sha256": identity.get("source_closure_sha256")}
def _begin_resume_probe(run_dir: Path, identity: Mapping[str, Any], prefix_run_dir: Path, parent_run_dir: Path, source_run_dir: Path, *, started_at: str | None = None, started_monotonic: float | None = None) -> tuple[Path, dict[str, Any]]:
    path = Path(run_dir).parent / f".{Path(run_dir).name}.resume-probe.json"
    binding = _resume_probe_binding(run_dir, identity, prefix_run_dir, parent_run_dir, source_run_dir)
    owned_monotonic = float(time.perf_counter())
    if path.is_file(): probe = _read_json(path, semantic=True)
    else: probe = _write_semantic(path, {**binding, "ledger_event_ids": [event["event_id"] for event in _validate_resource_ledger(run_dir)["events"]], "started_at": started_at or _utc_now(), "started_monotonic": float(owned_monotonic if started_monotonic is None else started_monotonic), "owned_monotonic": owned_monotonic})
    event_ids = [event["event_id"] for event in _validate_resource_ledger(run_dir)["events"]]
    if {key: probe.get(key) for key in binding} != binding or set(probe) != {*binding, "ledger_event_ids", "started_at", "started_monotonic", "owned_monotonic", "semantic_sha256"} or not isinstance(probe.get("ledger_event_ids"), list) or probe["ledger_event_ids"] != event_ids[:len(probe["ledger_event_ids"])] or not all(math.isfinite(float(probe.get(key, math.nan))) for key in ("started_monotonic", "owned_monotonic")) or float(probe["owned_monotonic"]) < float(probe["started_monotonic"]) or dt.datetime.fromisoformat(str(probe.get("started_at"))).tzinfo is None: raise ContinuationIntegrityError("resume probe ownership changed")
    return path, probe
def _restart_resume_probe(path: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    now = float(time.perf_counter()); return _write_semantic(path, {**binding, "ledger_event_ids": [event["event_id"] for event in _validate_resource_ledger(Path(str(binding["run_dir"])))["events"]], "started_at": _utc_now(), "started_monotonic": now, "owned_monotonic": now})
def _resume_probe_elapsed(probe: Mapping[str, Any], *, now_utc: str | None = None, now_monotonic: float | None = None) -> float:
    utc_delta = (dt.datetime.fromisoformat(now_utc or _utc_now()) - dt.datetime.fromisoformat(str(probe["started_at"]))).total_seconds(); monotonic_delta = float(time.perf_counter() if now_monotonic is None else now_monotonic) - float(probe["started_monotonic"])
    return max(utc_delta, monotonic_delta, 0.0)
def _resume_probe_is_covered(run_dir: Path, probe: Mapping[str, Any]) -> bool:
    known = set(probe["ledger_event_ids"]); return any(event["event_id"] not in known for event in _validate_resource_ledger(run_dir)["events"])
def _validate_durable_journal(run_dir: Path, journal: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(journal)
    keys = {
        "schema", "schema_version", "attempt_id", "role", "attempt", "run_identity",
        "ledger_active_seconds_at_start", "durable_elapsed_seconds_at_start",
        "started_at", "started_monotonic", "semantic_sha256",
    }
    role = value.get("role"); attempt = value.get("attempt")
    try: started = dt.datetime.fromisoformat(str(value.get("started_at")))
    except (TypeError, ValueError) as exc: raise ContinuationIntegrityError("durable journal timestamp changed") from exc
    numbers = (value.get("ledger_active_seconds_at_start"), value.get("durable_elapsed_seconds_at_start"), value.get("started_monotonic"))
    if (set(value) != keys or value.get("schema") != VERSION + "-durable-attempt" or value.get("schema_version") != 1 or not isinstance(role, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,72}", role) is None or role.endswith("_abandoned_attempt") or isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1 or started.tzinfo is None or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) < 0.0 for item in numbers) or value.get("run_identity") != _resource_run_identity(run_dir) or value.get("attempt_id") != _sha256_bytes(f"{value['run_identity']}\0{role}\0{attempt}".encode("utf-8")) or _semantic(value) != value):
        raise ContinuationIntegrityError("durable attempt journal contract changed")
    return value
def _begin_durable_attempt(
    run_dir: Path,
    *,
    role: str,
    attempt: int,
    durable_elapsed_seconds: float,
    now_utc: str | None = None,
    now_monotonic: float | None = None,
) -> dict[str, Any]:
    global _UNDEBITED_START
    run_dir = Path(run_dir).resolve()
    ledger = _validate_resource_ledger(run_dir)
    if ledger["limits_passed"] != 1: raise ResourceBoundaryError("cannot begin work on a breached resource ledger")
    run_identity = _resource_run_identity(run_dir)
    attempt_id = _sha256_bytes(f"{run_identity}\0{role}\0{attempt}".encode("utf-8"))
    journal = _semantic(
        {
            "schema": VERSION + "-durable-attempt",
            "schema_version": 1,
            "attempt_id": attempt_id,
            "role": role,
            "attempt": int(attempt),
            "run_identity": run_identity,
            "ledger_active_seconds_at_start": float(ledger.get("active_seconds", 0.0)),
            "durable_elapsed_seconds_at_start": float(durable_elapsed_seconds),
            "started_at": now_utc or _utc_now(),
            "started_monotonic": float(now_monotonic if now_monotonic is not None else (_UNDEBITED_START if _UNDEBITED_START is not None else time.perf_counter())),
        }
    )
    _UNDEBITED_START = None; path = _journal_path(run_dir, attempt_id)
    if path.is_file():
        existing = _validate_durable_journal(run_dir, _read_json(path, semantic=True))
        if existing != journal:
            raise ContinuationIntegrityError("durable attempt journal changed")
        return existing
    return _validate_durable_journal(run_dir, _write_semantic(path, journal))
def _reconcile_durable_attempt(
    run_dir: Path,
    journal: Mapping[str, Any],
    *,
    durable_elapsed_seconds: float,
    now_utc: str,
    now_monotonic: float,
    peak_cuda_bytes: int = 0,
    total_cuda_bytes: int = 0,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    journal = _validate_durable_journal(run_dir, journal)
    attempt_id = str(journal["attempt_id"])
    event_id = _sha256_bytes(f"abandoned\0{attempt_id}".encode("utf-8"))
    ledger_path = run_dir / "resource_ledger.json"
    ledger = _validate_resource_ledger(run_dir)
    normal = [
        event
        for event in ledger.get("events", [])
        if event.get("event_id") == attempt_id
    ]
    if normal:
        if len(normal) != 1:
            raise ContinuationIntegrityError("duplicate normal resource events cover a journal")
        event = normal[0]
        if (
            event.get("schema") != VERSION + "-resource-event"
            or event.get("role") != journal.get("role")
            or event.get("attempt") != journal.get("attempt")
        ):
            raise ContinuationIntegrityError("normal resource event conflicts with journal")
        path = _journal_path(run_dir, attempt_id)
        if path.is_file():
            path.unlink()
        ledger = _converge_resource_storage(run_dir)
        if ledger.get("limits_passed") != 1:
            raise ResourceBoundaryError("recovered normal event belongs to a breached ledger")
        return event
    for event in ledger.get("events", []):
        if event.get("event_id") == event_id:
            path = _journal_path(run_dir, attempt_id)
            if path.is_file():
                path.unlink()
            ledger = _converge_resource_storage(run_dir)
            if ledger.get("limits_passed") != 1:
                raise ResourceBoundaryError("reconciled resource ledger is breached")
            return event
    try:
        started = dt.datetime.fromisoformat(str(journal["started_at"]))
        ended = dt.datetime.fromisoformat(now_utc)
        utc_elapsed = max((ended - started).total_seconds(), 0.0)
    except Exception as exc:
        raise ContinuationIntegrityError("durable attempt UTC authority is invalid") from exc
    elapsed = max(
        utc_elapsed,
        float(now_monotonic) - float(journal["started_monotonic"]),
        float(durable_elapsed_seconds) - float(journal["durable_elapsed_seconds_at_start"]),
        0.0,
    ) + 5.0
    event = _semantic(
        {
            "schema": VERSION + "-resource-event",
            "schema_version": 1,
            "event_id": event_id,
            "role": str(journal["role"]) + "_abandoned_attempt",
            "attempt": int(journal["attempt"]),
            "durable_attempt_id": attempt_id,
            "journal_id": attempt_id,
            "elapsed_seconds": elapsed,
            "failed": 1,
            "peak_cuda_bytes": int(peak_cuda_bytes),
            "total_cuda_bytes": int(total_cuda_bytes),
            "reconciled_at": now_utc,
            "detail": {"hard_crash_reconciled": 1},
        }
    )
    events = list(ledger.get("events", [])) + [event]
    active = math.fsum(float(item["elapsed_seconds"]) for item in events)
    persisted = _directory_bytes(run_dir)
    peak = max(int(ledger.get("peak_cuda_bytes", 0)), int(peak_cuda_bytes))
    total = max(int(ledger.get("total_cuda_bytes", 0)), int(total_cuda_bytes))
    breaches = _resource_breaches(active, persisted, peak, total, ledger["breached_limits"], caps=ledger["caps"])
    event = _semantic({**event, "breaches": list(breaches), "limits_passed": int(not breaches)})
    events[-1] = event
    ledger.update(
        events=events,
        active_seconds=active,
        persisted_storage_bytes=persisted,
        peak_cuda_bytes=peak,
        total_cuda_bytes=total,
        breached_limits=breaches,
        limits_passed=int(not breaches),
    )
    _write_semantic(ledger_path, ledger)
    path = _journal_path(run_dir, attempt_id)
    if path.is_file():
        path.unlink()
    ledger = _converge_resource_storage(run_dir)
    if ledger["breached_limits"]:
        raise ResourceBoundaryError("reconciled resource boundary crossed")
    return event
def _finish_durable_attempt(
    run_dir: Path,
    journal: Mapping[str, Any],
    *,
    durable_elapsed_seconds: float,
    invocation_wall_seconds: float,
    peak_cuda_bytes: int = 0,
    total_cuda_bytes: int = 0,
    failed: bool = False,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    journal = _validate_durable_journal(run_dir, journal)
    ledger_path = run_dir / "resource_ledger.json"
    ledger = _validate_resource_ledger(run_dir)
    event_id = str(journal["attempt_id"])
    existing = next(
        (event for event in ledger.get("events", []) if event.get("event_id") == event_id),
        None,
    )
    if existing is not None:
        if (
            existing.get("schema") != VERSION + "-resource-event"
            or existing.get("role") != journal.get("role")
            or existing.get("attempt") != journal.get("attempt")
            or existing.get("event_id") != journal.get("attempt_id")
        ):
            raise ContinuationIntegrityError("existing resource event conflicts with journal")
        _journal_path(run_dir, event_id).unlink(missing_ok=True)
        ledger = _converge_resource_storage(run_dir)
        if ledger.get("limits_passed") != 1:
            raise ResourceBoundaryError("existing resource event belongs to a breached ledger")
        return existing
    durable_delta = max(
        float(durable_elapsed_seconds)
        - float(journal.get("durable_elapsed_seconds_at_start", 0.0)),
        0.0,
    )
    elapsed = max(float(invocation_wall_seconds), durable_delta, 0.0)
    event = _semantic(
        {
            "schema": VERSION + "-resource-event",
            "schema_version": 1,
            "event_id": event_id,
            "journal_id": event_id,
            "role": str(journal["role"]),
            "attempt": int(journal["attempt"]),
            "elapsed_seconds": elapsed,
            "failed": int(failed),
            "recorded_at": _utc_now(),
            "peak_cuda_bytes": int(peak_cuda_bytes),
            "total_cuda_bytes": int(total_cuda_bytes),
            "detail": dict(detail or {}),
        }
    )
    events = list(ledger.get("events", [])) + [event]
    active = math.fsum(float(item["elapsed_seconds"]) for item in events)
    peak = max(int(ledger.get("peak_cuda_bytes", 0)), int(peak_cuda_bytes))
    total = max(int(ledger.get("total_cuda_bytes", 0)), int(total_cuda_bytes))
    breaches = _resource_breaches(active, _directory_bytes(run_dir), peak, total, ledger["breached_limits"], caps=ledger["caps"])
    persisted = _directory_bytes(run_dir)
    event = _semantic({**event, "breaches": list(breaches), "limits_passed": int(not breaches)})
    events[-1] = event
    ledger.update(
        events=events,
        active_seconds=active,
        peak_cuda_bytes=peak,
        total_cuda_bytes=total,
        persisted_storage_bytes=persisted,
        breached_limits=breaches,
        limits_passed=int(not breaches),
    )
    _write_semantic(ledger_path, ledger)
    _journal_path(run_dir, event_id).unlink(missing_ok=True)
    ledger = _converge_resource_storage(run_dir)
    breaches = list(ledger["breached_limits"])
    if breaches:
        raise ResourceBoundaryError(f"resource boundary crossed: {', '.join(breaches)}")
    return event
def _prefix_carry_detail() -> dict[str, Any]:
    return {
        "v2_run_manifest_semantic_sha256": _V2_EXPECTED_HASHES["run_manifest.json"][1],
        "v2_resource_ledger_file_sha256": _V2_EXPECTED_HASHES["resource_ledger.json"][0],
        "v2_resource_ledger_semantic_sha256": _V2_EXPECTED_HASHES["resource_ledger.json"][1],
        "v2_verification_file_sha256": _V2_EXPECTED_HASHES["verification.json"][0],
        "v2_verification_semantic_sha256": _V2_EXPECTED_HASHES["verification.json"][1],
        "v2_terminal_failure_semantic_sha256": _V2_EXPECTED_HASHES["terminal_failure.json"][1],
        "v2_terminal_storage_file_sha256": _V2_EXPECTED_HASHES["terminal_storage_authority.json"][0],
        "v2_terminal_storage_semantic_sha256": _V2_EXPECTED_HASHES["terminal_storage_authority.json"][1],
        "source_event_count": 4, "carried_active_seconds": V2_CARRIED_ACTIVE_SECONDS,
        "accounting_scope": "authenticated_predecessor_cost",
    }
def _append_prefix_resource_carry(
    run_dir: Path, prefix_authority: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically carry authenticated predecessor cost once, without a crash journal."""

    run_dir = Path(run_dir).resolve()
    capsule_values: dict[str, Mapping[str, Any]] = {}
    for relative in ("resource_ledger.json", "verification.json", "terminal_failure.json", "terminal_storage_authority.json"):
        imported_path = run_dir / f"imports/v2/{relative}"
        raw_hash, semantic_hash = _V2_EXPECTED_HASHES[relative]
        value = _read_json(imported_path, semantic=True)
        if _sha256_file(imported_path) != raw_hash or value.get("semantic_sha256") != semantic_hash:
            raise ContinuationIntegrityError(f"imported v2 carry authority changed: {relative}")
        capsule_values[relative] = value
    imported = capsule_values["resource_ledger.json"]
    authenticated = prefix_authority.get("resource_ledger")
    if (
        not isinstance(authenticated, Mapping)
        or _canonical_json_bytes(imported) != _canonical_json_bytes(authenticated)
        or imported.get("active_seconds") != V2_CARRIED_ACTIVE_SECONDS
        or imported.get("peak_cuda_bytes") != V2_PEAK_CUDA_BYTES
        or imported.get("total_cuda_bytes") != V2_TOTAL_CUDA_BYTES
        or len(imported.get("events", [])) != 4
    ):
        raise ContinuationIntegrityError("imported v2 resource carry authority changed")
    ledger_path = run_dir / "resource_ledger.json"
    ledger = _validate_resource_ledger(run_dir)
    run_identity = _resource_run_identity(run_dir)
    role, attempt = "prefix_resource_carry", 1
    event_id = _sha256_bytes(f"{run_identity}\0{role}\0{attempt}".encode("utf-8"))
    detail = _prefix_carry_detail()
    stable = {
        "schema": VERSION + "-resource-event", "schema_version": 1,
        "event_id": event_id, "journal_id": event_id,
        "role": role, "attempt": attempt,
        "elapsed_seconds": V2_CARRIED_ACTIVE_SECONDS, "failed": 0,
        "peak_cuda_bytes": V2_PEAK_CUDA_BYTES,
        "total_cuda_bytes": V2_TOTAL_CUDA_BYTES, "detail": detail,
    }
    matches = [item for item in ledger["events"] if item.get("event_id") == event_id or item.get("role") == "prefix_resource_carry"]
    if matches:
        if len(matches) != 1 or any(matches[0].get(key) != value for key, value in stable.items()):
            raise ContinuationIntegrityError("prefix resource carry conflicts with existing event")
        return matches[0]
    active = math.fsum((float(ledger["active_seconds"]), V2_CARRIED_ACTIVE_SECONDS))
    peak = max(int(ledger["peak_cuda_bytes"]), V2_PEAK_CUDA_BYTES)
    total = max(int(ledger["total_cuda_bytes"]), V2_TOTAL_CUDA_BYTES)
    breaches = _resource_breaches(active, _directory_bytes(run_dir), peak, total, ledger["breached_limits"], caps=ledger["caps"])
    event = _semantic({**stable, "recorded_at": _utc_now(), "breaches": breaches, "limits_passed": int(not breaches)})
    ledger.update(
        events=[*ledger["events"], event], active_seconds=active,
        peak_cuda_bytes=peak, total_cuda_bytes=total,
        persisted_storage_bytes=_directory_bytes(run_dir),
        breached_limits=breaches, limits_passed=int(not breaches),
    )
    _write_semantic(ledger_path, ledger)
    _converge_resource_storage(run_dir)
    if breaches: raise ResourceBoundaryError("prefix resource carry crosses child resource boundary")
    return event
def _resource_event_count(run_dir: Path, role: str) -> int:
    ledger = _validate_resource_ledger(Path(run_dir))
    return sum(
        1
        for event in ledger.get("events", [])
        if event.get("role") in {role, role + "_abandoned_attempt"}
    )
def _cuda_memory_snapshot(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0, 0
    selected = torch.device(device)
    return (
        int(torch.cuda.max_memory_allocated(selected)),
        int(torch.cuda.get_device_properties(selected).total_memory),
    )
def _resource_storage_reserves(run_dir: Path) -> dict[str, int]:
    """Derive the sealed conservative storage model from authenticated v3 bytes."""

    binding = _read_json(Path(run_dir) / "parent_binding.json", semantic=True)
    parent = Path(str(binding.get("supplied_parent_path", ""))).resolve()
    prefix_root = parent / "fresh_forward/forward_shards/fresh-main-path"
    reverse_root = parent / "suffix/fused_families/fresh-five-row/suffix-128"
    def maximum_pair(root: Path, count: int) -> int:
        values: list[int] = []
        for index in range(count):
            paths = (root / f"shard-{index:04d}.json", root / f"shard-{index:04d}.npz")
            if any(not path.is_file() for path in paths):
                raise ContinuationIntegrityError("storage-reserve parent shard pair is missing")
            values.append(sum(int(path.stat().st_size) for path in paths))
        return max(values)

    forward_pair = math.ceil(1.20 * maximum_pair(prefix_root, PARENT_FORWARD_PREFIX_SHARDS))
    reverse_pair = math.ceil(1.20 * maximum_pair(reverse_root, 16))
    parent_derived = math.fsum(
        path.stat().st_size
        for path in (parent / "suffix").rglob("*")
        if path.is_file() and reverse_root not in path.parents
    )
    derived = max(16 * 1024**2, math.ceil(3.0 * parent_derived))
    return {
        "forward_pair_bytes": int(forward_pair),
        "reverse_pair_bytes": int(reverse_pair),
        "derived_and_terminal_bytes": int(derived),
    }
def _admit_storage_projection(
    run_dir: Path,
    *,
    remaining_forward_shards: int,
    remaining_reverse_shards: int,
) -> dict[str, int]:
    ledger = _validate_resource_ledger(Path(run_dir))
    if ledger["limits_passed"] != 1: raise ResourceBoundaryError("storage admission found a breached ledger")
    freeze_path = Path(run_dir) / "continuation_freeze.json"
    if freeze_path.is_file():
        freeze = _read_json(freeze_path, semantic=True)
        reserve = freeze.get("storage_reserves")
    else:
        reserve = None
    if not isinstance(reserve, Mapping):
        reserve = _resource_storage_reserves(run_dir)
    projected = (
        _directory_bytes(Path(run_dir))
        + int(remaining_forward_shards) * int(reserve["forward_pair_bytes"])
        + int(remaining_reverse_shards) * int(reserve["reverse_pair_bytes"])
        + int(reserve["derived_and_terminal_bytes"])
    )
    if projected >= STORAGE_CAP_BYTES:
        raise ResourceBoundaryError("storage projection cannot preserve the terminal bundle")
    return {**{key: int(value) for key, value in reserve.items()}, "projected_bytes": projected}
def _committed_elapsed(root: Path, *, minimum_index: int = 0) -> float:
    values: list[float] = []
    if root.is_dir():
        for record_path in sorted(root.glob("shard-*.json")):
            if record_path.name.endswith(".failure.json"):
                continue
            match = re.fullmatch(r"shard-(\d{4})", record_path.stem)
            if match is None:
                continue
            index = int(match.group(1))
            record = _read_json(record_path, semantic=True)
            if index >= minimum_index and record.get("committed") == 1:
                value = record.get("elapsed_seconds")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ContinuationIntegrityError("committed shard elapsed authority changed")
                values.append(float(value))
    return math.fsum(values)
def _committed_cuda_authority(
    records: Sequence[Mapping[str, Any]], *, reverse: bool
) -> tuple[int, int]:
    peaks: list[int] = []; totals: list[int] = []
    for record in records:
        source = record.get("diagnostics", {}).get("reference", {}) if reverse else record
        peak = source.get("peak_cuda_memory_bytes", source.get("peak_cuda_memory_allocated_bytes"))
        total = source.get("total_cuda_memory_bytes")
        if peak is None and total is None: continue
        if isinstance(peak, bool) or isinstance(total, bool) or not isinstance(peak, int) or not isinstance(total, int) or peak < 0 or total <= 0 or peak > total:
            raise ContinuationIntegrityError("committed shard CUDA authority changed")
        peaks.append(peak); totals.append(total)
    return (max(peaks), max(totals)) if peaks else (0, 0)
def _reconcile_live_stage_journals(run_dir: Path) -> list[dict[str, Any]]:
    """Debit every abandoned compatible-child interval before repeated stage work."""

    run_dir = Path(run_dir)
    _validate_resource_ledger(run_dir)
    _reconcile_recovery_intents(run_dir)
    root = run_dir / "journals"
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise ContinuationIntegrityError("durable journal root changed")
    allowed = {
        "prepare",
        "controls",
        "forward_tail",
        "reverse_complete",
        "reverse_postprocess",
        "report_verify",
        "resume_validation",
    }
    recovered: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink() or re.fullmatch(r"[0-9a-f]{64}\.json", path.name) is None:
            raise ContinuationIntegrityError("unexpected durable journal path")
        journal = _validate_durable_journal(run_dir, _read_json(path, semantic=True))
        role = journal.get("role")
        if role not in allowed or path.stem != journal.get("attempt_id"):
            raise ContinuationIntegrityError("durable journal role/identity changed")
        if role == "forward_tail":
            records = _scan_forward_chain(run_dir)["records"][IMPORTED_FORWARD_SHARDS:]
            durable = math.fsum(float(item["elapsed_seconds"]) for item in records)
            peak, total = _committed_cuda_authority(records, reverse=False)
        elif role == "reverse_complete":
            records = _scan_reverse_chain(run_dir)["records"]
            durable = math.fsum(float(item["elapsed_seconds"]) for item in records)
            peak, total = _committed_cuda_authority(records, reverse=True)
        else:
            durable = 0.0; peak = total = 0
        if role in {"forward_tail", "reverse_complete"}:
            live_peak, live_total = _cuda_memory_snapshot(torch.device("cuda")); peak, total = max(peak, live_peak), max(total, live_total)
        recovered.append(
            _reconcile_durable_attempt(
                run_dir,
                journal,
                durable_elapsed_seconds=durable,
                now_utc=_utc_now(),
                now_monotonic=time.perf_counter(),
                peak_cuda_bytes=peak,
                total_cuda_bytes=total,
            )
        )
    return recovered
def _verify_failure_domain_read_only(
    run_dir: Path,
    failure: Mapping[str, Any],
    *,
    require_terminal_bundle: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    domain = failure.get("failure_domain")
    allowed = {
        "parent_binding",
        "source_binding",
        "source_closure",
        "child_integrity",
        "composition_control",
        "resource_boundary",
        "numerical_health",
        "postprocessing",
        "report_verification",
        "unexpected_engineering_failure",
    }
    if domain not in allowed:
        raise ContinuationIntegrityError("unknown terminal failure domain")
    if require_terminal_bundle and not (run_dir / "terminal_failure.json").is_file():
        raise ContinuationIntegrityError("terminal failure bundle is incomplete")
    raw_paths = failure.get("available_raw_paths", [])
    if not isinstance(raw_paths, list):
        raise ContinuationIntegrityError("failure raw-path authority changed")
    verified = 0
    for item in raw_paths:
        if isinstance(item, Mapping):
            if set(item) != {"path", "size", "sha256"}: raise ContinuationIntegrityError("failure raw evidence row changed")
            relative = str(item["path"])
        else:
            relative = str(item)
        path = (run_dir / relative).resolve()
        if run_dir.resolve() not in path.parents:
            raise ContinuationIntegrityError("failure raw evidence escapes child")
        if not path.is_file() or path.is_symlink():
            raise ContinuationIntegrityError("failure raw evidence is missing")
        if isinstance(item, Mapping) and (path.stat().st_size != item["size"] or _sha256_file(path) != item["sha256"]):
            raise ContinuationIntegrityError("failure raw evidence changed")
        verified += 1
    return {
        "passed": 1,
        "documented_failed_predicate": domain,
        "raw_paths_verified": verified,
        "learned_interpretation_authorized": int(failure.get("learned_interpretation_authorized", 0)),
    }
def _compute_terminal_records_fixed_point(
    *,
    run_dir: Path,
    terminal_kind: str,
    terminalized_at: str,
    resource_ledger_semantic_sha256: str,
    verification_body: Mapping[str, Any],
    stage_body: Mapping[str, Any],
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    targets = (
        run_dir / "terminal_storage_authority.json",
        run_dir / "verification.json",
        run_dir / "stages/report_verify.json",
    )
    base_bytes = _directory_bytes(run_dir) - sum(
        path.stat().st_size for path in targets if path.is_file()
    )
    exact_total = base_bytes
    serialized: dict[str, bytes] = {}
    iterations = 0
    for iterations in range(1, 9):
        authority = _semantic(
            {
                "schema": VERSION + "-terminal-storage-authority",
                "schema_version": 1,
                "terminal_kind": terminal_kind,
                "terminalized_at": terminalized_at,
                "resource_ledger_semantic_sha256": resource_ledger_semantic_sha256,
                "exact_recursive_file_bytes": exact_total,
                "storage_cap_passed": int(exact_total < STORAGE_CAP_BYTES),
            }
        )
        verification = _semantic({**verification_body, "terminal_storage_semantic_sha256": authority["semantic_sha256"]})
        stage = _semantic(
            {
                **stage_body,
                "detail": {
                    **dict(stage_body.get("detail", {})),
                    "terminal_storage_semantic_sha256": authority["semantic_sha256"],
                    "verification_semantic_sha256": verification["semantic_sha256"],
                },
            }
        )
        candidate = {
            "terminal_storage_authority.json": _canonical_json_bytes(authority),
            "verification.json": _canonical_json_bytes(verification),
            "stages/report_verify.json": _canonical_json_bytes(stage),
        }
        next_total = base_bytes + sum(len(value) for value in candidate.values())
        serialized = candidate
        if next_total == exact_total:
            break
        exact_total = next_total
    else:
        raise ContinuationIntegrityError("terminal storage fixed point did not converge")
    return {
        "iterations": iterations,
        "exact_recursive_file_bytes": exact_total,
        "serialized_bytes": serialized,
    }
def _verify_terminal_child_contents_read_only(
    child: Path,
    *,
    prefix_run_dir: Path,
    parent_run_dir: Path,
    source_run_dir: Path,
    authenticated_imports: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    child = Path(child)
    failure_path = child / "terminal_failure.json"
    terminal_kind = "failure" if failure_path.is_file() else "success"
    inventory = _verify_terminal_inventory_read_only(child, terminal_kind)
    verification = _read_json(child / "verification.json", semantic=True)
    storage = _read_json(child / "terminal_storage_authority.json", semantic=True)
    manifest = inventory["manifest"]
    ledger_path = child / "resource_ledger.json"
    ledger_bound = storage.get("resource_ledger_file_sha256") == _sha256_file(ledger_path) if terminal_kind == "failure" else storage.get("resource_ledger_semantic_sha256") == _read_json(ledger_path, semantic=True).get("semantic_sha256")
    if (
        verification.get("passed") != 1
        or verification.get("terminal_kind") != terminal_kind
        or verification.get("terminal_storage_semantic_sha256") != storage.get("semantic_sha256")
        or verification.get("artifact_manifest_semantic_sha256") != manifest.get("semantic_sha256")
        or verification.get("checksums_file_sha256") != _sha256_file(child / "SHA256SUMS.txt")
        or storage.get("terminal_kind") != terminal_kind
        or storage.get("exact_recursive_file_bytes") != _directory_bytes(child)
        or storage.get("storage_cap_passed") != int(_directory_bytes(child) < STORAGE_CAP_BYTES)
        or terminal_kind == "success" and storage.get("storage_cap_passed") != 1
        or not ledger_bound
    ):
        raise ContinuationIntegrityError("terminal verification cross-binding changed")
    if terminal_kind == "success":
        marker = _read_stage_marker_exact(child, "report_verify")
        if (
            marker["detail"].get("terminal_storage_semantic_sha256") != storage["semantic_sha256"]
            or marker["detail"].get("verification_semantic_sha256") != verification["semantic_sha256"]
            or verification.get("scientific_objective_completed") != 1
        ):
            raise ContinuationIntegrityError("terminal success marker binding changed")
        evidence = _deep_verify_scientific_evidence_read_only(child)
    else:
        failure = _read_json(failure_path, semantic=True)
        capture = _read_json(child / "failure_capture.json", semantic=True)
        last_valid = _read_json(child / "last_valid_evidence.json", semantic=True)
        if (
            failure.get("failure_capture_semantic_sha256") != capture.get("semantic_sha256")
            or failure.get("last_valid_evidence_semantic_sha256") != last_valid.get("semantic_sha256")
            or last_valid.get("failure_capture_semantic_sha256") != capture.get("semantic_sha256")
            or verification.get("terminal_failure_semantic_sha256") != failure.get("semantic_sha256")
            or (child / "stages/report_verify.json").exists()
        ):
            raise ContinuationIntegrityError("terminal failure evidence closure changed")
        if (child / "stages/prepare.json").is_file():
            _completed_stage_artifacts_read_only(child, "prepare")
            if authenticated_imports is None: _verify_imported_inputs(child)
        evidence = _verify_failure_domain_read_only(child, failure)
    binding_path = child / "parent_binding.json"; predecessor_path = child / "predecessor_binding.json"
    if binding_path.is_file() and predecessor_path.is_file():
        binding = _read_json(binding_path, semantic=True); predecessor = _read_json(predecessor_path, semantic=True)
        if (
            _tree_hash(_snapshot_tree(Path(prefix_run_dir).resolve())) != predecessor.get("prefix_tree_sha256")
            or predecessor.get("supplied_prefix_path") != str(Path(prefix_run_dir).resolve())
            or _tree_hash(_snapshot_tree(Path(parent_run_dir).resolve())) != binding.get("parent_tree_sha256")
            or _tree_hash(_snapshot_tree(Path(source_run_dir).resolve())) != binding.get("source_tree_sha256")
        ):
            raise ContinuationIntegrityError("terminal external evidence tree changed")
    return {"passed": 1, "terminal_kind": terminal_kind, "inventory": inventory, "evidence": evidence}
def _verify_terminal_child_read_only(
    child: Path,
    *,
    prefix_run_dir: Path,
    parent_run_dir: Path,
    source_run_dir: Path,
) -> dict[str, Any]:
    roots = (Path(child), Path(prefix_run_dir), Path(parent_run_dir), Path(source_run_dir))
    before = tuple(_snapshot_tree(root) for root in roots)
    result = _verify_terminal_child_contents_read_only(
        Path(child), prefix_run_dir=Path(prefix_run_dir), parent_run_dir=Path(parent_run_dir), source_run_dir=Path(source_run_dir)
    )
    after = tuple(_snapshot_tree(root) for root in roots)
    if after != before:
        raise ContinuationIntegrityError("read-only verification mutated an evidence tree")
    return result
def _verify_resume_compatibility_read_only(
    child: Path,
    *,
    identity: Mapping[str, Any],
    prefix_run_dir: Path,
    parent_run_dir: Path,
    source_run_dir: Path,
) -> dict[str, Any]:
    child = Path(child)
    roots = (child, Path(prefix_run_dir).resolve(), Path(parent_run_dir).resolve(), Path(source_run_dir).resolve())
    before = tuple(_snapshot_tree(root) for root in roots)
    if (child / "stages/report_verify.json").is_file():
        raise TerminalRunError("terminal success child is verification-only")
    if (child / "verification.json").is_file() and (child / "terminal_failure.json").is_file():
        raise TerminalRunError("terminal failure child is verification-only")
    manifest = _read_json(child / "run_manifest.json", semantic=True)
    if identity.get("source_closure") != manifest.get("source_closure"):
        raise ContinuationIntegrityError("resume identity differs from child manifest")
    if str(Path(prefix_run_dir).resolve()) != manifest.get("prefix_run_dir"):
        raise ContinuationIntegrityError("resume prefix locator changed")
    if str(Path(parent_run_dir).resolve()) != manifest.get("parent_run_dir"):
        raise ContinuationIntegrityError("resume parent locator changed")
    if str(Path(source_run_dir).resolve()) != manifest.get("source_run_dir"):
        raise ContinuationIntegrityError("resume source locator changed")
    repository_root = Path(__file__).resolve().parents[1]
    closure = identity.get("source_closure")
    if not isinstance(closure, dict) or not closure:
        raise ContinuationIntegrityError("resume source closure is absent")
    for relative, authority in closure.items():
        path = repository_root / str(relative)
        if not path.is_file() or path.stat().st_size != int(authority.get("size", -1)) or _sha256_file(path) != authority.get("sha256"):
            raise ContinuationIntegrityError("resume source closure changed")
    parent_binding_path = child / "parent_binding.json"; predecessor_path = child / "predecessor_binding.json"
    if parent_binding_path.is_file() and predecessor_path.is_file():
        binding = _read_json(parent_binding_path, semantic=True); predecessor = _read_json(predecessor_path, semantic=True)
        if (
            _tree_hash(_snapshot_tree(Path(prefix_run_dir))) != predecessor.get("prefix_tree_sha256")
            or predecessor.get("supplied_prefix_path") != str(Path(prefix_run_dir).resolve())
            or _tree_hash(_snapshot_tree(Path(parent_run_dir))) != binding.get("parent_tree_sha256")
            or _tree_hash(_snapshot_tree(Path(source_run_dir))) != binding.get("source_tree_sha256")
        ):
            raise ContinuationIntegrityError("resume external evidence tree changed")
    after = tuple(_snapshot_tree(root) for root in roots)
    if after != before:
        raise ContinuationIntegrityError("resume compatibility verification mutated evidence")
    return {"passed": 1}
def _load_child_identity_read_only(run_dir: Path) -> dict[str, Any]:
    """Load the sealed, pre-stage child identity without trusting CLI locators."""
    run_dir = Path(run_dir).resolve()
    before = _snapshot_tree(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json", semantic=True)
    config = _read_json(run_dir / "scientific_config.json", semantic=True)
    closure = manifest.get("source_closure")
    repository_root = Path(__file__).resolve().parents[1]
    current_closure, current_closure_sha = _current_source_closure(repository_root)
    current_revision = _source_revision(repository_root)
    prefix_text = manifest.get("prefix_run_dir")
    parent_text = manifest.get("parent_run_dir")
    source_text = manifest.get("source_run_dir")
    try:
        prefix = Path(str(prefix_text)).resolve()
        parent = Path(str(parent_text)).resolve()
        source = Path(str(source_text)).resolve()
        created_at = dt.datetime.fromisoformat(str(manifest.get("created_at")))
    except (OSError, ValueError) as exc:
        raise ContinuationIntegrityError("sealed child locator/timestamp is invalid") from exc
    revision = manifest.get("source_revision")
    manifest_keys = {
        "schema", "schema_version", "created_at", "run_dir", "prefix_run_dir", "parent_run_dir",
        "source_run_dir", "source_revision", "source_closure",
        "source_closure_sha256", "scientific_config_file_sha256",
        "exact_command_file_sha256", "semantic_sha256",
    }
    expected_command = _canonical_fresh_command(
        repository_root=repository_root,
        run_dir=run_dir,
        prefix_run_dir=prefix,
        parent_run_dir=parent,
        source_run_dir=source,
    )
    if (
        set(manifest) != manifest_keys
        or manifest.get("schema") != RUN_SCHEMA
        or manifest.get("schema_version") != 1
        or isinstance(manifest.get("schema_version"), bool)
        or manifest.get("run_dir") != str(run_dir.resolve())
        or prefix_text != str(prefix)
        or parent_text != str(parent)
        or source_text != str(source)
        or any(_paths_overlap(left, right) for index, left in enumerate((run_dir, prefix, parent, source)) for right in (run_dir, prefix, parent, source)[index + 1 :])
        or created_at.tzinfo is None
        or not isinstance(revision, Mapping)
        or set(revision) != {"commit", "dirty", "status_sha256"}
        or not isinstance(revision.get("commit"), str)
        or not revision.get("commit")
        or revision.get("commit") != current_revision.get("commit")
        or revision.get("dirty") not in {0, 1}
        or isinstance(revision.get("dirty"), bool)
        or not isinstance(revision.get("dirty"), int)
        or re.fullmatch(r"[0-9a-f]{64}", str(revision.get("status_sha256"))) is None
        or not isinstance(closure, dict)
        or not closure
        or _canonical_json_bytes(closure) != _canonical_json_bytes(current_closure)
        or manifest.get("source_closure_sha256") != current_closure_sha
        or manifest.get("scientific_config_file_sha256")
        != _sha256_file(run_dir / "scientific_config.json")
        or _canonical_json_bytes(config) != _canonical_json_bytes(_scientific_config())
        or manifest.get("exact_command_file_sha256")
        != _sha256_file(run_dir / "exact_command.txt")
        or (run_dir / "exact_command.txt").read_bytes() != expected_command
    ):
        raise ContinuationIntegrityError("sealed child identity changed")
    completed = _stage_prefix_read_only(run_dir)
    if _snapshot_tree(run_dir) != before:
        raise ContinuationIntegrityError("child identity audit mutated the run")
    return {
        "run_manifest_semantic_sha256": manifest["semantic_sha256"],
        "scientific_config_semantic_sha256": config["semantic_sha256"],
        "source_closure": closure,
        "source_closure_sha256": manifest["source_closure_sha256"],
        "prefix_run_dir": str(prefix),
        "parent_run_dir": str(parent),
        "source_run_dir": str(source),
        "completed_stage_prefix": list(completed),
    }
def _read_stage_marker_exact(run_dir: Path, stage: str) -> dict[str, Any]:
    marker = _read_json(Path(run_dir) / f"stages/{stage}.json", semantic=True)
    expected_keys = {
        "schema", "schema_version", "stage", "passed", "completed_at",
        "detail", "semantic_sha256",
    }
    try:
        completed_at = dt.datetime.fromisoformat(str(marker.get("completed_at")))
    except ValueError as exc:
        raise ContinuationIntegrityError(
            f"completed stage timestamp is invalid: {stage}"
        ) from exc
    if (
        set(marker) != expected_keys
        or marker.get("schema") != VERSION + "-stage"
        or marker.get("schema_version") != 1
        or isinstance(marker.get("schema_version"), bool)
        or marker.get("stage") != stage
        or marker.get("passed") != 1
        or isinstance(marker.get("passed"), bool)
        or completed_at.tzinfo is None
        or not isinstance(marker.get("detail"), Mapping)
    ):
        raise ContinuationIntegrityError(f"completed stage marker is invalid: {stage}")
    return marker
def _stage_prefix_read_only(run_dir: Path) -> tuple[str, ...]:
    run_dir = Path(run_dir)
    stage_root = run_dir / "stages"
    if stage_root.exists() and (not stage_root.is_dir() or stage_root.is_symlink()):
        raise ContinuationIntegrityError("stage marker root is not a regular directory")
    entries = tuple(stage_root.iterdir()) if stage_root.is_dir() else ()
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ContinuationIntegrityError("stage marker tree contains a nonregular entry")
    actual = {path.name for path in entries}
    expected_names = {f"{name}.json" for name in STAGES}
    if actual - expected_names:
        raise ContinuationIntegrityError("unexpected stage marker path exists")
    completed: list[str] = []
    seen_missing = False
    for name in STAGES:
        marker_path = stage_root / f"{name}.json"
        exists = marker_path.is_file()
        if not exists:
            seen_missing = True
        elif seen_missing:
            raise ContinuationIntegrityError("stage markers are not a contiguous prefix")
        if exists:
            if marker_path.is_symlink():
                raise ContinuationIntegrityError("stage marker is a link")
            _read_stage_marker_exact(run_dir, name)
            completed.append(name)
    return tuple(completed)
def _require_predecessors(run_dir: Path, stage: str) -> None:
    index = STAGES.index(stage)
    completed = _stage_prefix_read_only(run_dir)
    if stage != STAGES[0]:
        required = STAGES[index - 1]
        if required not in completed:
            raise ContinuationIntegrityError(f"stage predecessor is missing: {required}")
        _completed_stage_artifacts_read_only(Path(run_dir), required)
def _stage_marker(run_dir: Path, stage: str, detail: Mapping[str, Any]) -> dict[str, Any]:
    return _write_semantic(
        Path(run_dir) / f"stages/{stage}.json",
        {
            "schema": VERSION + "-stage",
            "schema_version": 1,
            "stage": stage,
            "passed": 1,
            "completed_at": _utc_now(),
            "detail": dict(detail),
        },
    )
def _args_paths(args: argparse.Namespace, run_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        repository_root=Path(args.repository_root or Path.cwd()).resolve(),
        run_dir=Path(run_dir).resolve(),
        prefix_run_dir=Path(args.prefix_run_dir).resolve(),
        parent_run_dir=Path(args.parent_run_dir).resolve(),
        source_run_dir=Path(args.source_run_dir).resolve(),
    )
def _verify_external_trees(run_dir: Path, args: argparse.Namespace) -> None:
    binding = _read_json(Path(run_dir) / "parent_binding.json", semantic=True)
    predecessor = _read_json(Path(run_dir) / "predecessor_binding.json", semantic=True)
    if (
        _tree_hash(_snapshot_tree(Path(args.prefix_run_dir).resolve())) != predecessor["prefix_tree_sha256"]
        or predecessor.get("supplied_prefix_path") != str(Path(args.prefix_run_dir).resolve())
        or _tree_hash(_snapshot_tree(Path(args.parent_run_dir).resolve())) != binding["parent_tree_sha256"]
        or _tree_hash(_snapshot_tree(Path(args.source_run_dir).resolve())) != binding["source_tree_sha256"]
    ):
        raise ContinuationIntegrityError("external prefix/parent/source tree changed")
def _load_child_source(run_dir: Path) -> Any:
    source = load_verified_source_target(Path(run_dir) / "inputs/source")
    if (
        source.source_json_sha256 != SOURCE_JSON_SHA256
        or source.source_npz_sha256 != SOURCE_NPZ_SHA256
        or rollout_array_sha256(source.source_image) != SOURCE_IMAGE_ARRAY_SHA256
        or rollout_array_sha256(source.mixed_target) != MIXED_TARGET_ARRAY_SHA256
    ):
        raise ContinuationIntegrityError("child source copy changed")
    return source
def _strict_load_global_checkpoint(run_dir: Path, *, device: torch.device | str = "cpu") -> tuple[Any, dict[str, Any]]:
    path = Path(run_dir) / "inputs/model/update-3100.pt"
    if _sha256_file(path) != CHECKPOINT_FILE_SHA256:
        raise ContinuationIntegrityError("child global checkpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if (
        not isinstance(state, Mapping)
        or payload.get("seed") != 261_372
        or payload.get("update") != PARENT_SELECTED_UPDATE
        or payload.get("state_sha256") != CHECKPOINT_STATE_SHA256
        or state_dict_sha256(state) != CHECKPOINT_STATE_SHA256
    ):
        raise ContinuationIntegrityError("child global checkpoint payload changed")
    architecture = global_dilated_architecture_contract()
    if (
        architecture.get("passed") != 1
        or architecture.get("trainable_parameter_count") != GLOBAL_DILATED_PARAMETER_COUNT
    ):
        raise ContinuationIntegrityError("global architecture contract changed")
    model = GlobalDilatedZeroBaselinePredictor(zero_residual=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    if state_dict_sha256(model.state_dict()) != CHECKPOINT_STATE_SHA256:
        raise ContinuationIntegrityError("strict-loaded global model changed")
    return model, payload
def _verify_imported_inputs(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve(); identity = _load_child_identity_read_only(run_dir)
    prefix = Path(str(identity["prefix_run_dir"])).resolve(); parent = Path(str(identity["parent_run_dir"])).resolve(); source_root = Path(str(identity["source_run_dir"])).resolve()
    predecessor = _read_json(run_dir / "predecessor_binding.json", semantic=True); parent_binding = _read_json(run_dir / "parent_binding.json", semantic=True)
    inputs = _read_json(run_dir / "input_bindings.json", semantic=True); forward_import = _read_json(run_dir / "forward/prefix_import.json", semantic=True); reverse_import = _read_json(run_dir / "reverse/prefix_import.json", semantic=True)
    if (
        predecessor.get("schema") != VERSION + "-predecessor-binding" or predecessor.get("binding_passed") != 1
        or predecessor.get("supplied_prefix_path") != str(prefix) or predecessor.get("prefix_file_count") != V2_FILE_COUNT
        or predecessor.get("prefix_bytes") != V2_EXACT_BYTES or predecessor.get("operational_copy_count") != V2_OPERATIONAL_COPY_COUNT
        or predecessor.get("operational_copy_bytes") != V2_OPERATIONAL_COPY_BYTES or predecessor.get("source_ledger_active_seconds") != V2_CARRIED_ACTIVE_SECONDS
        or predecessor.get("source_peak_cuda_bytes") != V2_PEAK_CUDA_BYTES or predecessor.get("source_total_cuda_bytes") != V2_TOTAL_CUDA_BYTES
        or parent_binding.get("binding_passed") != 1 or parent_binding.get("supplied_parent_path") != str(parent) or parent_binding.get("supplied_source_path") != str(source_root)
        or inputs.get("predecessor_binding_semantic_sha256") != predecessor["semantic_sha256"]
        or forward_import.get("predecessor_binding_semantic_sha256") != predecessor["semantic_sha256"]
        or reverse_import.get("predecessor_binding_semantic_sha256") != predecessor["semantic_sha256"]
        or _tree_hash(_snapshot_tree(prefix)) != predecessor.get("prefix_tree_sha256")
        or _tree_hash(_snapshot_tree(parent)) != parent_binding.get("parent_tree_sha256")
        or _tree_hash(_snapshot_tree(source_root)) != parent_binding.get("source_tree_sha256")
    ): raise ContinuationIntegrityError("successor import binding changed")
    capsule_manifest_path = run_dir / "imports/v2/artifact_manifest.json"
    if _sha256_file(capsule_manifest_path) != _V2_EXPECTED_HASHES["artifact_manifest.json"][0]: raise ContinuationIntegrityError("imported v2 manifest changed")
    capsule_manifest = _read_json(capsule_manifest_path, semantic=True); manifest_rows = {str(row["path"]): row for row in capsule_manifest["artifacts"]}
    copy_rows = list(inputs.get("copies", [])) + list(forward_import.get("anchor_copies", []))
    for entry in forward_import.get("entries", []): copy_rows.extend((entry.get("json"), entry.get("npz")))
    copy_rows.extend(reverse_import.get("shard_0_copies", [])); expected_paths = _v2_operational_paths()
    if len(copy_rows) != V2_OPERATIONAL_COPY_COUNT or [row.get("role") for row in copy_rows if isinstance(row, Mapping)] != list(expected_paths) or sum(int(row.get("size", -1)) for row in copy_rows if isinstance(row, Mapping)) != V2_OPERATIONAL_COPY_BYTES:
        raise ContinuationIntegrityError("successor operational-copy inventory changed")
    for relative, row in zip(expected_paths, copy_rows, strict=True):
        if not isinstance(row, Mapping) or relative not in manifest_rows: raise ContinuationIntegrityError("successor operational-copy row changed")
        source_path = prefix / relative; destination = run_dir / relative; authority = manifest_rows[relative]
        if (
            row.get("source_locator") != str(source_path) or row.get("source_relative_path") != relative or row.get("child_relative_path") != relative or row.get("role") != relative
            or row.get("copy_mode") != "atomic_byte_copy" or row.get("samefile") != 0 or row.get("sha256") != authority.get("sha256") or row.get("size") != authority.get("size")
            or not source_path.is_file() or not destination.is_file() or source_path.is_symlink() or destination.is_symlink() or destination.stat().st_nlink != 1
            or os.path.samestat(source_path.stat(), destination.stat()) or destination.stat().st_size != source_path.stat().st_size or _sha256_file(source_path) != row.get("sha256") or _sha256_file(destination) != row.get("sha256")
        ): raise ContinuationIntegrityError(f"successor operational copy changed: {relative}")
    capsule = predecessor.get("provenance_capsule_copies")
    if not isinstance(capsule, list) or len(capsule) != len(_V2_CAPSULE_PATHS): raise ContinuationIntegrityError("v2 capsule inventory changed")
    for relative, row in zip(_V2_CAPSULE_PATHS, capsule, strict=True):
        source_path = prefix / relative; destination = run_dir / f"imports/v2/{relative}"
        if row.get("source_relative_path") != relative or row.get("child_relative_path") != f"imports/v2/{relative}" or not destination.is_file() or destination.stat().st_nlink != 1 or os.path.samestat(source_path.stat(), destination.stat()) or _sha256_file(destination) != _sha256_file(source_path):
            raise ContinuationIntegrityError(f"v2 capsule copy changed: {relative}")
    carries = [event for event in _validate_resource_ledger(run_dir)["events"] if event.get("role") == "prefix_resource_carry"]
    carry_id = _sha256_bytes(f"{_resource_run_identity(run_dir)}\0prefix_resource_carry\0{1}".encode("utf-8"))
    if (len(carries) != 1 or carries[0].get("event_id") != carry_id or carries[0].get("journal_id") != carry_id
            or carries[0].get("attempt") != 1 or carries[0].get("failed") != 0 or carries[0].get("detail") != _prefix_carry_detail()
            or carries[0].get("elapsed_seconds") != V2_CARRIED_ACTIVE_SECONDS or carries[0].get("peak_cuda_bytes") != V2_PEAK_CUDA_BYTES or carries[0].get("total_cuda_bytes") != V2_TOTAL_CUDA_BYTES):
        raise ContinuationIntegrityError("successor prefix resource carry changed")
    _strict_load_global_checkpoint(run_dir); source = _load_child_source(run_dir)
    for relative, file_hash, state_hash in (("forward/anchor-step-0127.npz", STEP127_ARCHIVE_SHA256, STEP127_STATE_SHA256), ("forward/anchor-step-0511.npz", STEP511_ARCHIVE_SHA256, STEP511_STATE_SHA256)):
        arrays = _load_npz(run_dir / relative); state = arrays.get("state")
        if _sha256_file(run_dir / relative) != file_hash or set(arrays) != {"state"} or not isinstance(state, np.ndarray) or state.dtype != np.float64 or state.shape != (STATE_SIZE,) or rollout_array_sha256(state) != state_hash:
            raise ContinuationIntegrityError("successor anchor copy changed")
    forward = _scan_forward_chain(run_dir)
    if forward["first_missing"] != FORWARD_SHARDS or forward["orphan"] is not None or forward["failure"] is not None or forward["records"][0].get("input_state_sha256") != rollout_array_sha256(np.ascontiguousarray(source.mixed_target[None, :], dtype=np.float64)):
        raise ContinuationIntegrityError("successor forward prefix changed")
    forward_health = aggregate_exact_forward_shards(forward["records"], expected_shard_count=FORWARD_SHARDS, expected_transition_count=FORWARD_TRANSITION_COUNT, expected_path_ids=(PATH_ID,))
    final_forward = _load_npz(run_dir / "forward/forward_shards/fresh-main-path/shard-0063.npz")["state"][0]
    if forward_health.get("passed") != 1 or forward_health.get("certificate_fraction") != 1.0 or not np.array_equal(final_forward, _load_anchor_511(run_dir)) or forward_import.get("imported_shard_indices") != list(range(FORWARD_SHARDS)) or forward_import.get("generated_child_shard_count") != 0:
        raise ContinuationIntegrityError("successor forward health/import authority changed")
    reverse = _scan_reverse_chain(run_dir); reverse_root = run_dir / "reverse/fused_families/same-path-three-row/complete-512"
    if reverse["first_missing"] < IMPORTED_REVERSE_SHARDS or reverse_import.get("imported_shard_indices") != [0] or reverse_import.get("generated_child_shard_indices") != list(range(1, REVERSE_SHARDS)):
        raise ContinuationIntegrityError("successor reverse prefix changed")
    state = _load_npz(reverse_root / "shard-0000.npz")["state"]
    reverse_health = _strict_fused_exact_health(final_state=state, shard_records=reverse["records"][:IMPORTED_REVERSE_SHARDS], expected_shard_count=IMPORTED_REVERSE_SHARDS)
    return {"passed": 1, "predecessor": predecessor, "inputs": inputs, "forward_import": forward_import, "reverse_import": reverse_import, "forward_health": forward_health, "reverse_health": reverse_health}
def _build_complete_rows(
    *, model: Any, mixed_target: np.ndarray, device: torch.device
) -> tuple[tuple[FusedRowSpec, ...], FusedTangentControllerBank, dict[str, Any]]:
    specs = (
        FusedRowSpec("zero", PATH_ID, "zero", "zero", "same-path-complete"),
        FusedRowSpec(
            "global-plus-1", PATH_ID, "learned", "global-dilated",
            "same-path-complete", gain=1.0,
            controller_binding={"checkpoint_state_sha256": CHECKPOINT_STATE_SHA256},
        ),
        FusedRowSpec(
            "source-informed", PATH_ID, "oracle", "mixed-target-fraction",
            "same-path-complete",
            controller_binding={"target_sha256": MIXED_TARGET_ARRAY_SHA256},
        ),
    )
    controllers = {
        "global-plus-1": ScaledTangentScoreController(model, 1.0),
        "source-informed": TargetFractionOracleController(
            mixed_target, microsteps=MICROSTEPS
        ).to(device),
    }
    bank = FusedTangentControllerBank(specs, controllers)
    return specs, bank, {
        "row_table": [item.to_record() for item in specs],
        "global_state_sha256": CHECKPOINT_STATE_SHA256,
        "target_sha256": MIXED_TARGET_ARRAY_SHA256,
        "dispatch": "stable_one_row_canonical_order",
        "model_input_contract": "exact_ModelInputs_six_fields",
    }
def _prepared_exact_backend(device: torch.device, profile: JacobiRBCudaProfile) -> Any:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ContinuationIntegrityError("production exact continuation requires CUDA")
    from mnist.d0_jacobi_rb_cuda_deferred import (
        prepare_alpha1_rb_transition_batch_cuda_deferred,
    )

    key = f"{device}:{DEFAULT_PROFILE_SHA256}"
    if key not in _PREPARED_BACKENDS:
        _PREPARED_BACKENDS[key] = prepare_alpha1_rb_transition_batch_cuda_deferred(
            device=device, profile=profile
        )
    return _PREPARED_BACKENDS[key]
def _exact_reference_factory(
    *, prepared: Any, profile: JacobiRBCudaProfile
) -> Callable[[int], DeferredCertifiedFusedReference]:
    key = (id(prepared), REVERSE_ROOT_SEED, REVERSE_STREAM_ROLE)
    if key not in _PREPARED_SEEDS:
        _PREPARED_SEEDS[key] = prepare_deferred_reference_rng_seed_map(
            prepared_backend=prepared,
            root_seed=REVERSE_ROOT_SEED,
            stream_role=REVERSE_STREAM_ROLE,
        )
    seeds = _PREPARED_SEEDS[key]
    def factory(_index: int) -> DeferredCertifiedFusedReference:
        return DeferredCertifiedFusedReference(
            profile=profile,
            root_seed=REVERSE_ROOT_SEED,
            stream_role=REVERSE_STREAM_ROLE,
            prepared_backend=prepared,
            prepared_rng_seeds=seeds,
        )

    return factory
def _complete_row_table_authority() -> list[dict[str, Any]]:
    return [
        FusedRowSpec("zero", PATH_ID, "zero", "zero", "same-path-complete").to_record(),
        FusedRowSpec(
            "global-plus-1", PATH_ID, "learned", "global-dilated",
            "same-path-complete", gain=1.0,
            controller_binding={"checkpoint_state_sha256": CHECKPOINT_STATE_SHA256},
        ).to_record(),
        FusedRowSpec(
            "source-informed", PATH_ID, "oracle", "mixed-target-fraction",
            "same-path-complete",
            controller_binding={"target_sha256": MIXED_TARGET_ARRAY_SHA256},
        ).to_record(),
    ]
def _complete_controller_binding_authority() -> dict[str, Any]:
    return {
        "row_table": _complete_row_table_authority(),
        "global_state_sha256": CHECKPOINT_STATE_SHA256,
        "target_sha256": MIXED_TARGET_ARRAY_SHA256,
        "dispatch": "stable_one_row_canonical_order",
        "model_input_contract": "exact_ModelInputs_six_fields",
    }
def _complete_rng_binding_authority() -> dict[str, Any]:
    return {
        "root_seed": REVERSE_ROOT_SEED,
        "stream_role": REVERSE_STREAM_ROLE,
        "canonical_path_id": PATH_ID,
    }
def _continuation_freeze_body(run_dir: Path) -> dict[str, Any]:
    source = _load_child_source(run_dir); profile = _default_profile(); sequence = tuple(reverse_suffix_sequence(511))
    scale = fixed_rendering_scale(source.source_image, source.mixed_target, float(source.metadata["lambda_mix"]))
    ledger = _validate_resource_ledger(run_dir); carries = [event for event in ledger["events"] if event.get("role") == "prefix_resource_carry"]
    if len(carries) != 1: raise ContinuationIntegrityError("freeze requires exactly one prefix resource carry")
    carry = carries[0]; reverse_import = _read_json(run_dir / "reverse/prefix_import.json", semantic=True)
    next_reverse = max(REVERSE_BASELINE_SECONDS, 1.20 * float(reverse_import["shard_0_elapsed_seconds"]))
    return {
        "schema": VERSION + "-freeze", "schema_version": 1, "sealed": 1,
        "child_sampling_performed_before_seal": 0,
        "imported_forward_shards_before_seal": FORWARD_SHARDS,
        "imported_reverse_shards_before_seal": IMPORTED_REVERSE_SHARDS,
        "predecessor_sampling_present": 1,
        "scientific_config_semantic_sha256": _read_json(run_dir / "scientific_config.json", semantic=True)["semantic_sha256"],
        "run_manifest_semantic_sha256": _read_json(run_dir / "run_manifest.json", semantic=True)["semantic_sha256"],
        "predecessor_binding_semantic_sha256": _read_json(run_dir / "predecessor_binding.json", semantic=True)["semantic_sha256"],
        "parent_binding_semantic_sha256": _read_json(run_dir / "parent_binding.json", semantic=True)["semantic_sha256"],
        "input_bindings_semantic_sha256": _read_json(run_dir / "input_bindings.json", semantic=True)["semantic_sha256"],
        "forward_prefix_import_semantic_sha256": _read_json(run_dir / "forward/prefix_import.json", semantic=True)["semantic_sha256"],
        "reverse_prefix_import_semantic_sha256": reverse_import["semantic_sha256"],
        "prefix_resource_event_id": carry["event_id"],
        "prefix_resource_carry_seconds": carry["elapsed_seconds"],
        "row_table": _complete_row_table_authority(), "controller_binding": _complete_controller_binding_authority(),
        "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256, "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
        "source_target_sha256": MIXED_TARGET_ARRAY_SHA256, "profile": profile.to_dict(), "profile_sha256": DEFAULT_PROFILE_SHA256,
        "reverse_sequence": [list(item) for item in sequence], "reverse_sequence_sha256": semantic_sha256([list(item) for item in sequence]),
        "capture_coordinates": [{"coordinate": list(key), "name": value} for key, value in CAPTURE_COORDINATES.items()],
        "rng_binding": _complete_rng_binding_authority(), "variant_in_rng_key": 0, "reference_contract": "certified_exact",
        "rendering_scale": scale.to_dict(),
        "imported_reverse_shard_0_json_sha256": _sha256_file(run_dir / "reverse/fused_families/same-path-three-row/complete-512/shard-0000.json"),
        "imported_reverse_shard_0_npz_sha256": _sha256_file(run_dir / "reverse/fused_families/same-path-three-row/complete-512/shard-0000.npz"),
        "step_511_state_sha256": STEP511_STATE_SHA256,
        "remaining_reverse_shards": REVERSE_SHARDS - IMPORTED_REVERSE_SHARDS,
        "initial_adaptive_reverse_seconds": next_reverse,
        "nominal_carried_projection_seconds": math.fsum((V2_CARRIED_ACTIVE_SECONDS, (REVERSE_SHARDS - IMPORTED_REVERSE_SHARDS) * next_reverse, POSTPROCESS_RESERVE_SECONDS, REPORT_RESERVE_SECONDS)),
        "resource_baselines": {"parent_five_row_shard_seconds": 310.3016812999995, "reverse_row_scaling": 3.0 / 5.0, "safety_multiplier": 1.20, "reverse_shard_seconds": REVERSE_BASELINE_SECONDS, "postprocess_seconds": POSTPROCESS_RESERVE_SECONDS},
        "storage_reserves": _resource_storage_reserves(run_dir), "resource_caps": _scientific_config()["resource_caps"],
    }
def _verify_continuation_freeze_exact(run_dir: Path) -> dict[str, Any]:
    actual = _read_json(Path(run_dir) / "continuation_freeze.json", semantic=True)
    if _canonical_json_bytes(actual) != _canonical_json_bytes(_semantic(_continuation_freeze_body(Path(run_dir)))): raise ContinuationIntegrityError("sealed continuation freeze changed")
    return actual
def _strict_fused_exact_health(
    *,
    final_state: np.ndarray,
    shard_records: Sequence[Mapping[str, Any]],
    row_count: int = len(ROW_ORDER),
    expected_shard_count: int = REVERSE_SHARDS,
) -> dict[str, Any]:
    """Fail closed on every exact/restart/mechanism field used downstream."""

    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, (int, np.integer))
        or int(row_count) != len(ROW_ORDER)
    ):
        raise ContinuationIntegrityError("fused exact row-count authority changed")
    state = np.asarray(final_state)
    if (
        state.dtype != np.float64
        or state.shape != (int(row_count), STATE_SIZE)
        or not np.isfinite(state).all()
        or np.any(state < 0.0)
        or float(np.max(np.abs(np.sum(state, axis=1) - 1.0))) > 2.0e-12
        or isinstance(expected_shard_count, bool)
        or not isinstance(expected_shard_count, (int, np.integer))
        or int(expected_shard_count) < IMPORTED_REVERSE_SHARDS
        or int(expected_shard_count) > REVERSE_SHARDS
        or len(shard_records) != int(expected_shard_count)
    ):
        raise ContinuationIntegrityError("fused exact final-state/family health failed")
    def integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ContinuationIntegrityError(f"fused exact {name} is not an integer")
        return int(value)
    def finite(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise ContinuationIntegrityError(f"fused exact {name} is not finite")
        return float(value)
    def reject_nonfinite_numerics(value: Any, name: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                reject_nonfinite_numerics(child, f"{name}.{key}")
        elif isinstance(value, (list, tuple)):
            for child_index, child in enumerate(value):
                reject_nonfinite_numerics(child, f"{name}[{child_index}]")
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise ContinuationIntegrityError(
                f"fused exact telemetry {name} is nonfinite"
            )

    total_active = total_certified = total_fallback = total_transitions = 0
    maximum_mass_error = 0.0
    expected_row_table = _complete_row_table_authority()
    expected_controller_binding_sha256 = semantic_sha256(
        _complete_controller_binding_authority()
    )
    expected_rng_binding_sha256 = semantic_sha256(_complete_rng_binding_authority())
    complete_sequence = tuple(reverse_suffix_sequence(511))
    if len(complete_sequence) != REVERSE_SHARDS * FUSED_SHARD_PHASES:
        raise ContinuationIntegrityError("fused exact complete sequence changed")
    expected_per_row = FUSED_SHARD_PHASES * 2 * MICROSTEPS * EDGES_PER_PHASE
    expected_per_shard = expected_per_row * len(ROW_ORDER)
    for index, shard in enumerate(shard_records):
        reject_nonfinite_numerics(shard, f"shard[{index}]")
        diagnostics = shard.get("diagnostics")
        reference = diagnostics.get("reference") if isinstance(diagnostics, Mapping) else None
        phase_rows = shard.get("per_row_diagnostics")
        controller_rows = shard.get("controller_diagnostics")
        row_table = shard.get("row_table")
        plan = shard.get("execution_plan")
        if (
            shard.get("schema") != FUSED_TANGENT_VERSION + "-reverse-shard"
            or integer(shard.get("schema_version"), "schema_version") != 1
            or shard.get("scheduler_version") != FUSED_TANGENT_VERSION
            or shard.get("family_name") != "same-path-three-row"
            or shard.get("segment_name") != "complete-512"
            or integer(shard.get("committed"), "committed") != 1
            or integer(shard.get("shard_index"), "shard_index") != index
            or shard.get("row_keys") != list(ROW_ORDER)
            or shard.get("canonical_path_ids") != [PATH_ID] * len(ROW_ORDER)
            or integer(shard.get("variant_in_rng_key"), "variant_in_rng_key") != 0
            or integer(shard.get("microsteps"), "microsteps") != MICROSTEPS
            or integer(shard.get("label"), "label") != 3
            or shard.get("controller_binding_sha256")
            != expected_controller_binding_sha256
            or shard.get("rng_binding_sha256") != expected_rng_binding_sha256
            or "reference_contract" in shard
            or integer(shard.get("transition_count"), "transition_count") != expected_per_shard
            or integer(shard.get("synchronous_replay_performed"), "replay") not in {0, 1}
            or not isinstance(plan, Mapping)
            or integer(plan.get("row_count"), "plan.row_count") != len(ROW_ORDER)
            or integer(plan.get("transition_count"), "plan.transition_count") != expected_per_shard
            or not isinstance(plan.get("sequence"), list)
            or plan["sequence"] != [
                list(item)
                for item in complete_sequence[
                    index * FUSED_SHARD_PHASES : (index + 1) * FUSED_SHARD_PHASES
                ]
            ]
            or shard.get("sequence_start") != plan["sequence"][0]
            or shard.get("sequence_end") != plan["sequence"][-1]
            or shard.get("sequence_sha256") != semantic_sha256(plan["sequence"])
            or not isinstance(row_table, list)
            or len(row_table) != len(ROW_ORDER)
            or not isinstance(phase_rows, list)
            or len(phase_rows) != len(ROW_ORDER)
            or not isinstance(controller_rows, list)
            or len(controller_rows) != len(ROW_ORDER)
            or not isinstance(diagnostics, Mapping)
            or not isinstance(reference, Mapping)
        ):
            raise ContinuationIntegrityError(f"fused exact shard schema changed: {index}")
        if row_table != expected_row_table:
            raise ContinuationIntegrityError("fused exact row-table authority changed")

        forbidden = diagnostics.get("forbidden_counts")
        reference_forbidden = reference.get("forbidden_counts")
        if (
            not isinstance(forbidden, Mapping)
            or set(forbidden) != set(_FORBIDDEN_EXACT_COUNTS)
            or any(integer(forbidden[name], name) != 0 for name in _FORBIDDEN_EXACT_COUNTS)
            or not isinstance(reference_forbidden, Mapping)
            or set(reference_forbidden) != set(_FORBIDDEN_EXACT_COUNTS)
            or any(integer(reference_forbidden[name], name) != 0 for name in _FORBIDDEN_EXACT_COUNTS)
            or integer(diagnostics.get("transition_count"), "diagnostics.transition_count")
            != expected_per_shard
            or finite(diagnostics.get("certificate_fraction"), "certificate_fraction") != 1.0
            or finite(reference.get("certificate_fraction"), "reference.certificate_fraction") != 1.0
        ):
            raise ContinuationIntegrityError("fused exact certificate/forbidden authority changed")
        transition = integer(reference.get("transition_count"), "reference.transition_count")
        active = integer(reference.get("active_count"), "reference.active_count")
        noops = integer(reference.get("structural_noop_count"), "reference.noops")
        certified = integer(reference.get("certified_count"), "reference.certified")
        fallback = integer(reference.get("fallback_count"), "reference.fallback")
        reference_rows = reference.get("per_row")
        replayed = integer(shard.get("synchronous_replay_performed"), "replay")
        expected_reference_schema = (
            TANGENT_ROLLOUT_VERSION + "-certified-reference"
            if replayed == 1
            else FUSED_TANGENT_VERSION + "-deferred-reference-shard"
        )
        if (
            reference.get("schema") != expected_reference_schema
            or integer(reference.get("root_seed"), "reference.root_seed")
            != REVERSE_ROOT_SEED
            or reference.get("stream_role") != REVERSE_STREAM_ROLE
            or reference.get("rng_namespace") != EXPLORATORY_REFERENCE_RNG_NAMESPACE
            or integer(reference.get("variant_in_rng_key"), "reference.variant") != 0
            or integer(
                reference.get("needs_synchronous_replay"),
                "reference.needs_synchronous_replay",
            )
            != 0
            or (
                replayed == 1
                and integer(
                    reference.get("speculative_attempt_discarded"),
                    "reference.speculative_attempt_discarded",
                )
                != 1
            )
            or (replayed == 0 and "speculative_attempt_discarded" in reference)
            or transition != expected_per_shard
            or active <= 0
            or active + noops != transition
            or certified != active
            or fallback < 0
            or fallback > transition
            or integer(reference.get("unauthorized_count"), "reference.unauthorized") != 0
            or integer(reference.get("invalid_count"), "reference.invalid") != 0
            or integer(diagnostics.get("fallback_count"), "diagnostics.fallback") != fallback
            or not isinstance(reference_rows, list)
            or len(reference_rows) != len(ROW_ORDER)
        ):
            raise ContinuationIntegrityError("fused exact transition authority changed")
        maximum_mass_error = max(
            maximum_mass_error,
            finite(diagnostics.get("maximum_mass_error"), "maximum_mass_error"),
        )
        if maximum_mass_error > 2.0e-12:
            raise ContinuationIntegrityError("fused exact mass health failed")

        row_transition_sum = row_active_sum = row_certified_sum = row_fallback_sum = 0
        shard_pair_mass_error = 0.0
        shard_simplex_mass_error = 0.0
        for row_index, (phase, controller, reference_row) in enumerate(
            zip(phase_rows, controller_rows, reference_rows, strict=True)
        ):
            expected_row = expected_row_table[row_index]
            expected_gain = expected_row["gain"]
            if (
                not isinstance(phase, Mapping)
                or not isinstance(controller, Mapping)
                or not isinstance(reference_row, Mapping)
                or phase.get("row_key") != ROW_ORDER[row_index]
                or controller.get("row_key") != ROW_ORDER[row_index]
                or controller.get("controller_kind") != expected_row["controller_kind"]
                or type(controller.get("gain")) is not type(expected_gain)
                or controller.get("gain") != expected_gain
                or reference_row.get("row_key", ROW_ORDER[row_index]) != ROW_ORDER[row_index]
                or any(integer(phase.get(name, 0), name) != 0 for name in _FUSED_INVALID_FIELDS)
                or any(integer(controller.get(name, 0), name) != 0 for name in (
                    "clipping_count", "floor_count", "projection_count", "nonfinite_score_count"
                ))
                or not set(_FUSED_CONTROLLER_INTEGER_FIELDS).issubset(controller)
                or not set(_FUSED_CONTROLLER_FLOAT_FIELDS).issubset(controller)
            ):
                raise ContinuationIntegrityError("fused exact per-row telemetry changed")
            for name in _FUSED_CONTROLLER_INTEGER_FIELDS:
                if integer(controller[name], f"controller.{name}") < 0:
                    raise ContinuationIntegrityError("negative fused controller count")
            for name in _FUSED_CONTROLLER_FLOAT_FIELDS:
                if finite(controller[name], f"controller.{name}") < 0.0:
                    raise ContinuationIntegrityError("negative fused controller magnitude")
            boundary_count = integer(
                phase.get("boundary_fraction_count"), "boundary_fraction_count"
            )
            phase_score_count = integer(phase.get("score_count"), "score.count")
            pair_mass_error = finite(
                phase.get("maximum_pair_mass_error"), "maximum_pair_mass_error"
            )
            simplex_mass_error = finite(
                phase.get("maximum_simplex_mass_error"),
                "maximum_simplex_mass_error",
            )
            if (
                boundary_count < 0
                or boundary_count > phase_score_count
                or pair_mass_error < 0.0
                or simplex_mass_error < 0.0
                or pair_mass_error > 2.0e-12
                or simplex_mass_error > 2.0e-12
            ):
                raise ContinuationIntegrityError(
                    "fused phase boundary or mass telemetry changed"
                )
            shard_pair_mass_error = max(shard_pair_mass_error, pair_mass_error)
            shard_simplex_mass_error = max(
                shard_simplex_mass_error, simplex_mass_error
            )
            for prefix in _FUSED_PHASE_PREFIXES:
                count = integer(phase.get(prefix + "_count"), prefix + ".count")
                squared = finite(phase.get(prefix + "_squared_sum"), prefix + ".squared")
                maximum = finite(
                    phase.get(prefix + "_maximum_absolute"), prefix + ".maximum"
                )
                rms = finite(phase.get(prefix + "_rms"), prefix + ".rms")
                if count < 0 or squared < 0.0 or maximum < 0.0 or rms < 0.0:
                    raise ContinuationIntegrityError("negative fused phase telemetry")
                expected_rms = math.sqrt(squared / count) if count else 0.0
                if not math.isclose(
                    rms, expected_rms, rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    raise ContinuationIntegrityError("fused phase RMS telemetry changed")
            controller_score_count = integer(
                controller.get("score_count"), "controller.score_count"
            )
            for prefix in ("score", "unscaled_score"):
                squared = finite(
                    controller.get(prefix + "_squared_sum"),
                    f"controller.{prefix}.squared",
                )
                rms = finite(
                    controller.get(prefix + "_rms"), f"controller.{prefix}.rms"
                )
                expected_rms = (
                    math.sqrt(squared / controller_score_count)
                    if controller_score_count
                    else 0.0
                )
                if not math.isclose(
                    rms, expected_rms, rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    raise ContinuationIntegrityError(
                        f"fused controller {prefix} RMS telemetry changed"
                    )
            if controller_score_count != phase_score_count or any(
                not math.isclose(
                    finite(
                        controller.get("score_" + suffix),
                        f"controller.score.{suffix}",
                    ),
                    finite(phase.get("score_" + suffix), f"phase.score.{suffix}"),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
                for suffix in ("squared_sum", "maximum_absolute", "rms")
            ):
                raise ContinuationIntegrityError(
                    "fused phase/controller score telemetry changed"
                )
            row_transition = integer(phase.get("reference_transition_count"), "row.transition")
            row_active = integer(phase.get("reference_active_count"), "row.active")
            row_noops = integer(phase.get("reference_structural_noop_count"), "row.noops")
            row_certified = integer(phase.get("reference_certified_count"), "row.certified")
            row_fallback = integer(phase.get("reference_fallback_count"), "row.fallback")
            reference_row_transition = integer(
                reference_row.get("transition_count"), "reference.row.transition"
            )
            reference_row_active = integer(
                reference_row.get("active_count"), "reference.row.active"
            )
            reference_row_noops = integer(
                reference_row.get("structural_noop_count"), "reference.row.noops"
            )
            reference_row_certified = integer(
                reference_row.get("certified_count"), "reference.row.certified"
            )
            reference_row_fallback = integer(
                reference_row.get("fallback_count"), "reference.row.fallback"
            )
            if (
                integer(phase.get("transition_count"), "row.phase_transition") != expected_per_row
                or row_transition != expected_per_row
                or row_active <= 0
                or row_active + row_noops != row_transition
                or row_certified != row_active
                or row_fallback < 0
                or row_fallback > row_transition
                or integer(phase.get("reference_unauthorized_count"), "row.unauthorized") != 0
                or integer(phase.get("reference_invalid_count"), "row.invalid") != 0
                or finite(phase.get("reference_certificate_fraction"), "row.certificate") != 1.0
                or reference_row_transition != row_transition
                or reference_row_active != row_active
                or reference_row_noops != row_noops
                or reference_row_certified != row_certified
                or reference_row_fallback != row_fallback
                or integer(
                    reference_row.get("unauthorized_count"),
                    "reference.row.unauthorized",
                ) != 0
                or integer(reference_row.get("invalid_count"), "reference.row.invalid") != 0
                or finite(
                    reference_row.get("certificate_fraction"),
                    "reference.row.certificate",
                ) != 1.0
            ):
                raise ContinuationIntegrityError("fused per-row transition authority changed")
            row_transition_sum += row_transition
            row_active_sum += row_active
            row_certified_sum += row_certified
            row_fallback_sum += row_fallback
        if (row_transition_sum, row_active_sum, row_certified_sum, row_fallback_sum) != (
            transition, active, certified, fallback
        ):
            raise ContinuationIntegrityError("fused per-row aggregates changed")
        expected_mass_error = max(shard_pair_mass_error, shard_simplex_mass_error)
        if finite(
            diagnostics.get("maximum_mass_error"), "diagnostics.maximum_mass_error"
        ) != expected_mass_error:
            raise ContinuationIntegrityError("fused shard mass aggregate changed")
        total_transitions += transition
        total_active += active
        total_certified += certified
        total_fallback += fallback
    expected_total = int(expected_shard_count) * REVERSE_SHARD_TRANSITION_COUNT
    if total_transitions != expected_total or total_active != total_certified:
        raise ContinuationIntegrityError("fused complete-family transition authority changed")
    return {
        "passed": 1,
        "row_count": len(ROW_ORDER),
        "shard_count": int(expected_shard_count),
        "transition_count": total_transitions,
        "active_count": total_active,
        "certified_count": total_certified,
        "certificate_fraction": 1.0,
        "fallback_count": total_fallback,
        "forbidden_event_count": 0,
        "maximum_mass_error": maximum_mass_error,
        "final_state_nonfinite_count": 0,
        "final_state_negative_count": 0,
    }
def _save_contact_sheet(path: Path, cells: Sequence[np.ndarray], *, columns: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ContinuationIntegrityError("Pillow is required for contact sheets") from exc
    if not cells or columns <= 0:
        raise ContinuationIntegrityError("contact sheet cells/columns are invalid")
    rows = math.ceil(len(cells) / columns)
    canvas = Image.new("L", (columns * 28, rows * 28), color=0)
    for index, cell in enumerate(cells):
        value = np.asarray(cell)
        if value.dtype != np.uint8 or value.shape != (28, 28):
            raise ContinuationIntegrityError("contact sheet cell changed")
        canvas.paste(
            Image.fromarray(value, mode="L"),
            ((index % columns) * 28, (index // columns) * 28),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        canvas.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
def _aggregate_reverse_boundaries(
    run_dir: Path, *, shard_records: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    root = Path(run_dir) / "reverse/fused_families/same-path-three-row/complete-512"
    anchor = _load_anchor_511(run_dir)
    boundary_states = [np.repeat(anchor[None, :], len(ROW_ORDER), axis=0)]
    previous = rollout_array_sha256(boundary_states[0])
    if len(shard_records) != REVERSE_SHARDS:
        raise ContinuationIntegrityError("reverse raw family is incomplete")
    for index, record in enumerate(shard_records):
        path = root / f"shard-{index:04d}.npz"
        arrays = _load_npz(path)
        state = arrays.get("state")
        if (
            set(arrays) != {"state"}
            or not isinstance(state, np.ndarray)
            or state.dtype != np.float64
            or state.shape != (len(ROW_ORDER), STATE_SIZE)
            or record.get("input_state_sha256") != previous
            or record.get("state_file_sha256") != _sha256_file(path)
            or record.get("output_state_sha256") != rollout_array_sha256(state)
            or not np.isfinite(state).all()
            or np.any(state < 0.0)
            or float(np.max(np.abs(np.sum(state, axis=1) - 1.0))) > 2.0e-12
        ):
            raise ContinuationIntegrityError(f"reverse raw boundary changed: {index}")
        previous = rollout_array_sha256(state)
        boundary_states.append(np.ascontiguousarray(state))
    states = np.ascontiguousarray(np.stack(boundary_states, axis=1), dtype=np.float64)
    completed = np.arange(0, OUTER_STEPS + 1, 8, dtype=np.int64)
    milestones = np.ascontiguousarray(states[:, [0, 16, 32, 48, 64], :])
    trajectory_path = Path(run_dir) / "reverse/trajectory_shard_boundaries.npz"
    milestone_path = Path(run_dir) / "reverse/milestones.npz"
    atomic_rollout_npz(trajectory_path, {"states": states, "completed_reverse_steps": completed})
    atomic_rollout_npz(
        milestone_path,
        {"states": milestones, "completed_reverse_steps": np.asarray(MILESTONE_STEPS, dtype=np.int64)},
    )
    record = _write_semantic(
        Path(run_dir) / "reverse/trajectory_aggregation.json",
        {
            "schema": VERSION + "-trajectory-aggregation",
            "schema_version": 1,
            "row_order": list(ROW_ORDER),
            "shard_count": REVERSE_SHARDS,
            "states_shape": list(states.shape),
            "states_sha256": rollout_array_sha256(states),
            "trajectory_file_sha256": _sha256_file(trajectory_path),
            "milestone_steps": list(MILESTONE_STEPS),
            "milestone_states_sha256": rollout_array_sha256(milestones),
            "milestone_file_sha256": _sha256_file(milestone_path),
            "chain_valid": 1,
        },
    )
    return states, milestones, record
def _quarter_mechanism(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row_index, row_key in enumerate(ROW_ORDER):
        quarters: dict[str, Any] = {}
        for quarter in range(4):
            selected = [
                shard["per_row_diagnostics"][row_index]
                for shard in shards[quarter * 16 : (quarter + 1) * 16]
            ]
            sums: dict[str, float] = {}
            maxima: dict[str, float] = {}
            for entry in selected:
                for key, value in entry.items():
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                        continue
                    if key.endswith("_squared_sum") or key.endswith("_count"):
                        sums[key] = sums.get(key, 0.0) + float(value)
                    elif "maximum" in key:
                        maxima[key] = max(maxima.get(key, 0.0), float(value))
            def rms(prefix: str) -> float | None:
                count = sums.get(prefix + "_count", 0.0)
                squared = sums.get(prefix + "_squared_sum", 0.0)
                return math.sqrt(squared / count) if count > 0.0 else None

            score = rms("score")
            control = rms("control_fraction_displacement")
            reference = rms("reference_fraction_displacement")
            quarters[str(quarter)] = {
                "reverse_progress_steps": [quarter * 128, (quarter + 1) * 128],
                "shard_count": len(selected),
                "score_rms": score,
                "control_fraction_displacement_rms": control,
                "reference_fraction_displacement_rms": reference,
                "control_reference_ratio": (
                    control / reference if control is not None and reference not in (None, 0.0) else None
                ),
                "maxima": maxima,
            }
        result[row_key] = quarters
    return result
def _build_reverse_derived(
    run_dir: Path,
    *,
    states: np.ndarray,
    milestones: np.ndarray,
    shard_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_dir = Path(run_dir)
    source = _load_child_source(run_dir)
    freeze = _read_json(run_dir / "continuation_freeze.json", semantic=True)
    scale = FixedRenderingScale(**freeze["rendering_scale"])
    completed = np.arange(0, OUTER_STEPS + 1, 8, dtype=np.int64)
    if states.shape != (len(ROW_ORDER), 65, STATE_SIZE) or milestones.shape != (len(ROW_ORDER), 5, STATE_SIZE):
        raise ContinuationIntegrityError("reverse derived input shape changed")
    rows: list[dict[str, Any]] = []
    intermediate: dict[int, dict[str, float]] = {}
    for boundary_index, reverse_steps in enumerate(completed.tolist()):
        zero_state = states[0, boundary_index]
        zero_metric = raw_state_metrics(zero_state, source.mixed_target)
        for row_index, row_key in enumerate(ROW_ORDER):
            state = states[row_index, boundary_index]
            mixed = raw_state_metrics(state, source.mixed_target).to_dict()
            unmixed = raw_state_metrics(state, source.source_image).to_dict()
            separation = state - zero_state
            rows.append(
                {
                    "row_index": row_index,
                    "row_key": row_key,
                    "boundary_index": boundary_index,
                    "completed_reverse_steps": int(reverse_steps),
                    **{f"mixed_target_{key}": value for key, value in mixed.items()},
                    **{f"unmixed_source_{key}": value for key, value in unmixed.items()},
                    "row_vs_zero_squared_separation": float(np.dot(separation, separation)),
                    "nonfinite_count": int(np.count_nonzero(~np.isfinite(state))),
                    "negative_count": int(np.count_nonzero(state < 0.0)),
                    "is_milestone": int(int(reverse_steps) in MILESTONE_STEPS),
                }
            )
        if int(reverse_steps) in MILESTONE_STEPS:
            global_metric = raw_state_metrics(states[1, boundary_index], source.mixed_target)
            intermediate[int(reverse_steps)] = {
                "zero_error": zero_metric.squared_l2_error,
                "global_error": global_metric.squared_l2_error,
            }
    if len(rows) != 195:
        raise ContinuationIntegrityError("reverse metric row count changed")
    metrics_path = run_dir / "reverse/metrics.csv"
    _atomic_csv(metrics_path, rows)

    calibration = _load_npz(run_dir / "inputs/calibration/on_policy_validation_calibration.npz")
    means = calibration["training_means"]
    p95 = calibration["training_p95"]
    validation = calibration["validation_sorted_ratios"]
    counts = calibration["validation_counts"]
    drift: list[dict[str, Any]] = []
    for boundary_index, reverse_steps in enumerate(completed.tolist()):
        current_outer = max(511 - int(reverse_steps), 0)
        quartile = min(current_outer // 128, 3)
        radius = float(np.linalg.norm(states[1, boundary_index] - means[quartile]))
        ratio = radius / float(p95[quartile])
        valid = np.asarray(validation[quartile, : int(counts[quartile])])
        drift.append(
            {
                "boundary_index": boundary_index,
                "completed_reverse_steps": int(reverse_steps),
                "matching_training_quartile": int(quartile),
                "global_radius_from_training_quartile_mean": radius,
                "global_radius_normalized_by_training_p95": ratio,
                "validation_calibrated_percentile": float(np.searchsorted(valid, ratio, side="right") / valid.size),
                "threshold_or_execution_gate": 0,
            }
        )
    mechanism = _write_semantic(
        run_dir / "reverse/mechanism.json",
        {
            "schema": VERSION + "-mechanism",
            "schema_version": 1,
            "per_reverse_quarter": _quarter_mechanism(shard_records),
            "global_vs_zero_horizon_trace": [
                {
                    "completed_reverse_steps": int(step),
                    "zero_error": float(raw_state_metrics(states[0, index], source.mixed_target).squared_l2_error),
                    "global_error": float(raw_state_metrics(states[1, index], source.mixed_target).squared_l2_error),
                    "paired_delta": float(
                        raw_state_metrics(states[0, index], source.mixed_target).squared_l2_error
                        - raw_state_metrics(states[1, index], source.mixed_target).squared_l2_error
                    ),
                    "relative_improvement": float(
                        (
                            raw_state_metrics(states[0, index], source.mixed_target).squared_l2_error
                            - raw_state_metrics(states[1, index], source.mixed_target).squared_l2_error
                        )
                        / raw_state_metrics(states[0, index], source.mixed_target).squared_l2_error
                    )
                    if raw_state_metrics(states[0, index], source.mixed_target).squared_l2_error > 0.0
                    else None,
                }
                for index, step in enumerate(completed.tolist())
            ],
            "on_policy_drift": drift,
            "trajectory_states_sha256": rollout_array_sha256(states),
            "metrics_file_sha256": _sha256_file(metrics_path),
            "drift_is_descriptive_only": 1,
            "off_policy_failure_inferred": 0,
        },
    )

    image_records: list[dict[str, Any]] = []
    contact_sheet_records: list[dict[str, Any]] = []
    all_raw: list[np.ndarray] = []
    all_demixed: list[np.ndarray] = []
    for milestone_index, milestone in enumerate(MILESTONE_STEPS):
        milestone_raw: list[np.ndarray] = []
        milestone_demixed: list[np.ndarray] = []
        for row_index, row_key in enumerate(ROW_ORDER):
            state = milestones[row_index, milestone_index]
            raw = render_raw_density(state, scale)
            demixed = render_background_demixed(state, scale)
            raw_path = run_dir / f"images/raw/{row_key}/step-{milestone:03d}.png"
            demixed_path = run_dir / f"images/demixed/{row_key}/step-{milestone:03d}.png"
            save_png(raw_path, raw)
            save_png(demixed_path, demixed)
            all_raw.append(raw)
            all_demixed.append(demixed)
            milestone_raw.append(raw)
            milestone_demixed.append(demixed)
            for rendering, rendered, rendered_path in (
                ("raw", raw, raw_path), ("demixed", demixed, demixed_path)
            ):
                image_records.append(
                    {
                        "row_key": row_key,
                        "completed_reverse_steps": milestone,
                        "rendering": rendering,
                        "state_sha256": rollout_array_sha256(state),
                        "pixel_sha256": rollout_array_sha256(rendered),
                        "path": rendered_path.relative_to(run_dir).as_posix(),
                        "file_sha256": _sha256_file(rendered_path),
                    }
                )

        for rendering, cells in (("raw", milestone_raw), ("demixed", milestone_demixed)):
            sheet = run_dir / f"images/contact-sheets/milestone-{milestone:03d}-{rendering}.png"
            _save_contact_sheet(sheet, cells, columns=3)
            contact_sheet_records.append(
                {
                    "kind": "milestone",
                    "completed_reverse_steps": milestone,
                    "rendering": rendering,
                    "columns": 3,
                    "cell_count": 3,
                    "path": sheet.relative_to(run_dir).as_posix(),
                    "file_sha256": _sha256_file(sheet),
                }
            )
    for kind, rendering, cells in (
        ("all-milestones", "raw", all_raw),
        ("all-milestones", "demixed", all_demixed),
        ("final", "raw", all_raw[-3:]),
        ("final", "demixed", all_demixed[-3:]),
    ):
        sheet = run_dir / f"images/contact-sheets/{kind}-{rendering}.png"
        _save_contact_sheet(sheet, cells, columns=3)
        contact_sheet_records.append(
            {
                "kind": kind,
                "rendering": rendering,
                "columns": 3,
                "cell_count": len(cells),
                "path": sheet.relative_to(run_dir).as_posix(),
                "file_sha256": _sha256_file(sheet),
            }
        )

    final_metrics: dict[str, Any] = {}
    zero = raw_state_metrics(states[0, -1], source.mixed_target)
    for index, row_key in enumerate(ROW_ORDER):
        metric = raw_state_metrics(states[index, -1], source.mixed_target)
        final_metrics[row_key] = {
            **metric.to_dict(),
            **(
                {"paired_squared_l2_improvement_over_zero": 0.0, "relative_paired_squared_l2_improvement_over_zero": 0.0}
                if index == 0
                else paired_metric_improvement(metric, zero)
            ),
        }
    summary = _write_semantic(
        run_dir / "reverse/summary.json",
        {
            "schema": VERSION + "-reverse-summary",
            "schema_version": 1,
            "research_mode": "exploratory",
            "independent_unit": "one already-opened path/image",
            "row_order": list(ROW_ORDER),
            "primary_final_objectives": final_metrics,
            "metric_row_count": len(rows),
            "metric_file_sha256": _sha256_file(metrics_path),
            "mechanism_file_sha256": _sha256_file(run_dir / "reverse/mechanism.json"),
            "rendering_scale": scale.to_dict(),
            "images": image_records,
            "contact_sheets": contact_sheet_records,
            "individual_image_count": len(image_records),
            "contact_sheet_count": len(contact_sheet_records),
            "failed_or_adverse_outputs_suppressed": 0,
            "one_path_scope": 1,
            "confirmatory_claim": 0,
        },
    )
    outcome = _classify_complete_outcome(
        zero_error=final_metrics["zero"]["squared_l2_error"],
        global_error=final_metrics["global-plus-1"]["squared_l2_error"],
        source_error=final_metrics["source-informed"]["squared_l2_error"],
        intermediate=intermediate,
    )
    outcome.update(
        source_control_passed=int(outcome["source_effect_label"] == "source_informative"),
        exact_health_passed=1,
        one_path_scope=1,
        confirmatory_claim=0,
    )
    outcome = _write_semantic(run_dir / "outcome.json", outcome)
    return summary, mechanism, outcome
def _scan_forward_chain(run_dir: Path) -> dict[str, Any]:
    root = Path(run_dir) / "forward/forward_shards/fresh-main-path"
    previous: str | None = None
    first_missing = FORWARD_SHARDS
    records: list[dict[str, Any]] = []
    orphan: Path | None = None
    failure: Path | None = None
    for index in range(FORWARD_SHARDS):
        record_path = root / f"shard-{index:04d}.json"
        archive_path = root / f"shard-{index:04d}.npz"
        failure_path = root / f"shard-{index:04d}.failure.json"
        if record_path.exists() and archive_path.exists() and failure_path.exists():
            raise ContinuationIntegrityError(
                "committed forward shard retains a live failure record"
            )
        if not record_path.exists():
            first_missing = index
            if archive_path.is_file():
                orphan = archive_path
            if failure_path.is_file():
                failure = failure_path
            break
        if not archive_path.is_file():
            raise ContinuationIntegrityError("forward JSON-first orphan is forbidden")
        record = _read_json(record_path, semantic=True)
        arrays = _load_npz(archive_path)
        state = arrays.get("state")
        if (
            record.get("committed") != 1
            or record.get("shard_index") != index
            or record.get("start_step") != index * 8
            or record.get("trajectory_name") != "fresh-main-path"
            or record.get("path_ids") != [PATH_ID]
            or record.get("root_seed") != FORWARD_ROOT_SEED
            or record.get("profile_sha256") != DEFAULT_PROFILE_SHA256
            or record.get("state_file_sha256") != _sha256_file(archive_path)
            or not isinstance(state, np.ndarray)
            or state.dtype != np.float64
            or state.shape != (1, STATE_SIZE)
            or record.get("output_state_sha256") != rollout_array_sha256(state)
            or (previous is not None and record.get("input_state_sha256") != previous)
        ):
            raise ContinuationIntegrityError(f"forward shard chain changed: {index}")
        previous = str(record["output_state_sha256"])
        records.append(record)
    for path in root.glob("shard-*.*"):
        match = re.match(r"shard-(\d{4})", path.name)
        if match:
            evidence_index = int(match.group(1))
            if evidence_index >= FORWARD_SHARDS or evidence_index > first_missing:
                raise ContinuationIntegrityError("forward evidence exists after the first gap")
            if evidence_index == first_missing and not (
                path == orphan or path == failure
            ):
                raise ContinuationIntegrityError("unexpected forward evidence exists at the gap")
    return {
        "first_missing": first_missing,
        "records": records,
        "orphan": orphan,
        "failure": failure,
    }
def _scan_reverse_chain(run_dir: Path) -> dict[str, Any]:
    root = Path(run_dir) / "reverse/fused_families/same-path-three-row/complete-512"
    first_missing = REVERSE_SHARDS
    records: list[dict[str, Any]] = []
    orphan: Path | None = None
    failure: Path | None = None
    previous = rollout_array_sha256(
        np.repeat(_load_anchor_511(run_dir)[None, :], len(ROW_ORDER), axis=0)
    )
    for index in range(REVERSE_SHARDS):
        record_path = root / f"shard-{index:04d}.json"
        archive_path = root / f"shard-{index:04d}.npz"
        failure_path = root / f"shard-{index:04d}.failure.json"
        if record_path.exists() and archive_path.exists() and failure_path.exists():
            raise ContinuationIntegrityError(
                "committed reverse shard retains a live failure record"
            )
        if not record_path.exists():
            first_missing = index
            if archive_path.is_file():
                orphan = archive_path
            if failure_path.is_file():
                failure = failure_path
            break
        if not archive_path.is_file():
            raise ContinuationIntegrityError("reverse JSON-first orphan is forbidden")
        record = _read_json(record_path, semantic=True)
        arrays = _load_npz(archive_path)
        state = arrays.get("state")
        if (
            record.get("committed") != 1
            or record.get("shard_index") != index
            or record.get("family_name") != "same-path-three-row"
            or record.get("segment_name") != "complete-512"
            or record.get("row_keys") != list(ROW_ORDER)
            or record.get("canonical_path_ids") != [PATH_ID] * len(ROW_ORDER)
            or record.get("variant_in_rng_key") != 0
            or record.get("input_state_sha256") != previous
            or record.get("state_file_sha256") != _sha256_file(archive_path)
            or record.get("state_file_size") != archive_path.stat().st_size
            or not isinstance(state, np.ndarray)
            or state.dtype != np.float64
            or state.shape != (len(ROW_ORDER), STATE_SIZE)
            or record.get("output_state_sha256") != rollout_array_sha256(state)
        ):
            raise ContinuationIntegrityError(f"reverse shard chain changed: {index}")
        previous = str(record["output_state_sha256"])
        records.append(record)
    for path in root.glob("shard-*.*"):
        match = re.match(r"shard-(\d{4})", path.name)
        if match:
            evidence_index = int(match.group(1))
            if evidence_index >= REVERSE_SHARDS or evidence_index > first_missing:
                raise ContinuationIntegrityError("reverse evidence exists after the first gap")
            if evidence_index == first_missing and not (
                path == orphan or path == failure
            ):
                raise ContinuationIntegrityError("unexpected reverse evidence exists at the gap")
    return {
        "first_missing": first_missing,
        "records": records,
        "orphan": orphan,
        "failure": failure,
    }
def _load_anchor_511(run_dir: Path) -> np.ndarray:
    arrays = _load_npz(Path(run_dir) / "forward/anchor-step-0511.npz")
    state = arrays.get("state")
    if (
        not isinstance(state, np.ndarray)
        or state.dtype != np.float64
        or state.shape != (STATE_SIZE,)
        or not np.isfinite(state).all()
        or np.any(state < 0.0)
        or abs(float(np.sum(state)) - 1.0) > 2.0e-12
    ):
        raise ContinuationIntegrityError("step-511 anchor health changed")
    return np.ascontiguousarray(state)
def _archive_replay_evidence(
    run_dir: Path, *, root: Path, role: str, shard_index: int, attempt_id: str
) -> tuple[dict[str, Any], str]:
    record = _recover_replay_evidence(
        run_dir, root=root, role=role, shard_index=shard_index,
        attempt_id=attempt_id, require_orphan=True,
    )
    return record, str(record["orphan"]["original_sha256"])
def _archive_live_shard_failure_before_replay(
    run_dir: Path,
    *,
    failure: Path,
    role: str,
    shard_index: int,
    attempt_id: str,
) -> dict[str, Any]:
    base_role = role.removesuffix("-failure")
    record = _recover_replay_evidence(
        run_dir, root=Path(failure).parent, role=base_role,
        shard_index=shard_index, attempt_id=attempt_id, require_orphan=False,
    )
    return dict(record["prior_failure"])
def _planned_recovery_file(
    source: Path, recovery_root: Path, *, role: str, shard_index: int, attempt_id: str
) -> dict[str, Any]:
    source = Path(source); recovery_root = Path(recovery_root)
    if source.is_symlink() or not source.is_file():
        raise ContinuationIntegrityError(f"recovery evidence is not a regular file: {source}")
    archive = recovery_root / role / f"shard-{shard_index:04d}.{attempt_id}.original{''.join(source.suffixes) or '.bin'}"
    try: relative = archive.relative_to(recovery_root.parents[1]).as_posix()
    except (IndexError, ValueError): relative = archive.as_posix()
    return _semantic({
        "schema": VERSION + "-orphan-recovery", "schema_version": 1,
        "role": role, "shard_index": int(shard_index), "attempt_id": attempt_id,
        "original_path": str(source.resolve()), "archive_relative_path": relative,
        "original_size": int(source.stat().st_size), "original_sha256": _sha256_file(source),
    })
def _recover_replay_evidence(
    run_dir: Path, *, root: Path, role: str, shard_index: int,
    attempt_id: str, require_orphan: bool,
) -> dict[str, Any]:
    """Create a durable intent, then archive, index, and retire in that order."""

    run_dir = Path(run_dir).resolve(); root = Path(root).resolve()
    if run_dir != root and run_dir not in root.parents:
        raise ContinuationIntegrityError("recovery root escapes child")
    recovery_root = run_dir / "recovery/orphans"
    orphan_path = root / f"shard-{shard_index:04d}.npz"
    failure_path = root / f"shard-{shard_index:04d}.failure.json"
    orphan = _planned_recovery_file(
        orphan_path, recovery_root, role=role, shard_index=shard_index,
        attempt_id=attempt_id,
    ) if require_orphan else None
    failure = _planned_recovery_file(
        failure_path, recovery_root, role=role + "-failure", shard_index=shard_index,
        attempt_id=attempt_id,
    ) if failure_path.is_file() else None
    if orphan is None and failure is None:
        raise ContinuationIntegrityError("recovery intent has no live evidence")
    primary = orphan or failure
    recovery = _semantic({
        "schema": VERSION + "-replay-evidence", "schema_version": 1,
        "orphan": orphan, "prior_failure": failure,
        "archive_relative_path": primary["archive_relative_path"],
        "original_sha256": primary["original_sha256"],
    })
    operation_id = _sha256_bytes(f"{role}\0{shard_index}\0{attempt_id}".encode("utf-8"))
    intent_path = run_dir / "recovery/intents" / f"{operation_id}.json"
    body = {
        "schema": VERSION + "-recovery-intent", "schema_version": 1,
        "operation_id": operation_id, "role": role, "shard_index": shard_index,
        "attempt_id": attempt_id, "created_at": _utc_now(), "recovery": recovery,
    }
    if intent_path.is_file():
        existing = _read_json(intent_path, semantic=True)
        existing_core = {key: value for key, value in existing.items() if key not in {"created_at", "semantic_sha256"}}
        expected_core = {key: value for key, value in body.items() if key != "created_at"}
        if existing_core != expected_core:
            raise ContinuationIntegrityError("recovery intent conflicts with live evidence")
    else:
        _write_semantic(intent_path, body)
    completed = _reconcile_recovery_intents(run_dir)
    match = next((item for item in completed if item.get("semantic_sha256") == recovery["semantic_sha256"]), None)
    if match is None: raise ContinuationIntegrityError("recovery intent did not reconcile")
    return match
def _reconcile_recovery_intents(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir).resolve(); root = run_dir / "recovery/intents"
    if not root.exists(): return []
    if not root.is_dir() or root.is_symlink(): raise ContinuationIntegrityError("recovery intent root changed")
    completed: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink() or re.fullmatch(r"[0-9a-f]{64}\.json", path.name) is None:
            raise ContinuationIntegrityError("unexpected recovery intent path")
        intent = _read_json(path, semantic=True); recovery = intent.get("recovery")
        keys = {"schema", "schema_version", "operation_id", "role", "shard_index", "attempt_id", "created_at", "recovery", "semantic_sha256"}
        operation = _sha256_bytes(f"{intent.get('role')}\0{intent.get('shard_index')}\0{intent.get('attempt_id')}".encode("utf-8"))
        if set(intent) != keys or intent.get("schema") != VERSION + "-recovery-intent" or intent.get("schema_version") != 1 or path.stem != operation or intent.get("operation_id") != operation or not isinstance(recovery, Mapping) or _semantic(recovery) != recovery:
            raise ContinuationIntegrityError("recovery intent authority changed")
        records = [item for item in (recovery.get("orphan"), recovery.get("prior_failure")) if item is not None]
        if not records: raise ContinuationIntegrityError("recovery intent is empty")
        for record in records:
            source = Path(str(record.get("original_path"))).resolve()
            archive = (run_dir / str(record.get("archive_relative_path"))).resolve()
            if (run_dir not in source.parents or run_dir not in archive.parents or record.get("schema") != VERSION + "-orphan-recovery" or record.get("shard_index") != intent.get("shard_index") or record.get("attempt_id") != intent.get("attempt_id") or _semantic(record) != record):
                raise ContinuationIntegrityError("recovery evidence binding changed")
            digest = str(record["original_sha256"]); size = int(record["original_size"])
            if archive.is_file():
                if archive.is_symlink() or archive.stat().st_size != size or _sha256_file(archive) != digest:
                    raise ContinuationIntegrityError("recovery archive changed")
            else:
                if not source.is_file() or source.is_symlink() or source.stat().st_size != size or _sha256_file(source) != digest:
                    raise ContinuationIntegrityError("live recovery evidence changed")
                _copy_bound_file(source, archive, expected_sha256=digest)
            if source.is_file() and _sha256_file(source) != digest:
                raise ContinuationIntegrityError("live evidence changed after archive")
        _append_recovery_record(run_dir, recovery)  # durable index precedes retirement/replay
        failure = recovery.get("prior_failure")
        if failure is not None:
            live = Path(str(failure["original_path"]))
            if live.exists(): live.unlink()
            if live.exists(): raise ContinuationIntegrityError("live failure retirement failed")
        path.unlink(); completed.append(dict(recovery))
    try: root.rmdir()
    except OSError: pass
    return completed
def _append_recovery_record(run_dir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(run_dir) / "recovery/orphan-replays.json"
    entries: list[dict[str, Any]] = []
    if path.is_file():
        current = _read_json(path, semantic=True)
        raw = current.get("entries")
        if set(current) != {"schema", "schema_version", "entries", "semantic_sha256"} or current.get("schema") != VERSION + "-orphan-replays" or current.get("schema_version") != 1 or not isinstance(raw, list) or any(not isinstance(item, Mapping) or _semantic(item) != item for item in raw):
            raise ContinuationIntegrityError("orphan recovery index changed")
        entries = [dict(item) for item in raw]
    semantic = str(record.get("semantic_sha256", ""))
    if _semantic(record) != record: raise ContinuationIntegrityError("recovery record semantic authority changed")
    if not any(item.get("semantic_sha256") == semantic for item in entries):
        entries.append(dict(record))
    result = _write_semantic(
        path,
        {
            "schema": VERSION + "-orphan-replays",
            "schema_version": 1,
            "entries": entries,
        },
    )
    if not any(item.get("semantic_sha256") == semantic for item in result["entries"]):
        raise ContinuationIntegrityError("recovery index commit failed")
    return result
def _run_prepare(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir = Path(run_dir)
    paths = _args_paths(args, run_dir)
    role = "prepare"
    attempt = _resource_event_count(run_dir, role) + 1
    journal = _begin_durable_attempt(
        run_dir, role=role, attempt=attempt, durable_elapsed_seconds=0.0,
    )
    started = float(journal["started_monotonic"])
    external_before = tuple(_snapshot_tree(root) for root in (paths.prefix_run_dir, paths.parent_run_dir, paths.source_run_dir))
    parent_authority = _verify_parent_bundle_read_only(paths.parent_run_dir, source_run_dir=paths.source_run_dir)
    prefix_authority = _verify_v2_prefix_bundle_read_only(paths.prefix_run_dir, parent_run_dir=paths.parent_run_dir, source_run_dir=paths.source_run_dir)
    def copy(relative: str, destination: str | None = None) -> dict[str, Any]:
        source_path = paths.prefix_run_dir / relative; child_relative = destination or relative
        bound = _copy_bound_file(source_path, run_dir / child_relative, expected_sha256=_sha256_file(source_path))
        result = {"role": child_relative, "source_locator": str(source_path), "source_relative_path": relative, "child_relative_path": child_relative, "size": bound["size"], "sha256": bound["sha256"], "samefile": bound["samefile"], "copy_mode": "atomic_byte_copy"}
        destination_path = run_dir / child_relative
        if destination_path.stat().st_nlink != 1 or os.path.samestat(source_path.stat(), destination_path.stat()) or _sha256_file(destination_path) != result["sha256"]:
            raise ContinuationIntegrityError("successor copy is not byte-independent")
        return result
    capsule = [copy(relative, f"imports/v2/{relative}") for relative in _V2_CAPSULE_PATHS]
    if len(capsule) != len(_V2_CAPSULE_PATHS) or any(_sha256_file(run_dir / row["child_relative_path"]) != row["sha256"] for row in capsule):
        raise ContinuationIntegrityError("v2 provenance capsule copy/reopen failed")
    carry = _append_prefix_resource_carry(run_dir, prefix_authority)
    operational = [copy(relative) for relative in _v2_operational_paths()]
    if len(operational) != V2_OPERATIONAL_COPY_COUNT or sum(int(row["size"]) for row in operational) != V2_OPERATIONAL_COPY_BYTES:
        raise ContinuationIntegrityError("v2 operational copy count/bytes changed")
    by_role = {row["role"]: row for row in operational}
    predecessor = _write_semantic(run_dir / "predecessor_binding.json", {
        "schema": VERSION + "-predecessor-binding", "schema_version": 1,
        "historical_prefix_path": prefix_authority["values"]["run_manifest.json"]["run_dir"], "supplied_prefix_path": str(paths.prefix_run_dir),
        "prefix_file_count": prefix_authority["prefix_file_count"], "prefix_bytes": prefix_authority["prefix_bytes"], "prefix_tree_sha256": prefix_authority["prefix_tree_sha256"],
        "pinned_terminal_hashes": {relative: {"file_sha256": hashes[0], "semantic_sha256": hashes[1]} for relative, hashes in _V2_EXPECTED_HASHES.items()},
        "manifest_artifact_count": prefix_authority["manifest_artifact_count"], "checksum_entry_count": prefix_authority["checksum_entry_count"],
        "operational_copy_count": len(operational), "operational_copy_bytes": sum(int(row["size"]) for row in operational), "provenance_capsule_copies": capsule,
        "source_ledger_active_seconds": prefix_authority["resource_ledger"]["active_seconds"], "source_peak_cuda_bytes": prefix_authority["resource_ledger"]["peak_cuda_bytes"], "source_total_cuda_bytes": prefix_authority["resource_ledger"]["total_cuda_bytes"],
        "forward_health": prefix_authority["forward_health"], "reverse_shard_0_health": prefix_authority["reverse_shard_0_health"],
        "v2_parent_run_dir": prefix_authority["v2_parent_run_dir"], "v2_source_run_dir": prefix_authority["v2_source_run_dir"],
        "confirmation_opened": 0, "binding_passed": 1,
    })
    model, payload = _strict_load_global_checkpoint(run_dir); del model
    source = _load_child_source(run_dir)
    parent_binding = _write_semantic(run_dir / "parent_binding.json", {"schema": VERSION + "-parent-binding", "schema_version": 1, **parent_authority, "historical_parent_path": _read_json(paths.parent_run_dir / "run_manifest.json", semantic=True)["run_dir"], "supplied_parent_path": str(paths.parent_run_dir), "supplied_source_path": str(paths.source_run_dir), "selected_training_fingerprint": payload.get("training_fingerprint")})
    inputs = _write_semantic(run_dir / "input_bindings.json", {
        "schema": VERSION + "-input-bindings", "schema_version": 1, "predecessor_binding_semantic_sha256": predecessor["semantic_sha256"],
        "copies": [by_role[path] for path in _v2_operational_paths()[:4]], "source_image_array_sha256": rollout_array_sha256(source.source_image), "mixed_target_array_sha256": rollout_array_sha256(source.mixed_target),
        "parent_tree_sha256": parent_authority["parent_tree_sha256"], "source_tree_sha256": parent_authority["source_tree_sha256"],
    })
    forward_entries = [{"shard_index": index, "ownership": "authenticated_v2_import", **{suffix: by_role[f"forward/forward_shards/fresh-main-path/shard-{index:04d}.{suffix}"] for suffix in ("json", "npz")}} for index in range(FORWARD_SHARDS)]
    forward_import = _write_semantic(run_dir / "forward/prefix_import.json", {
        "schema": VERSION + "-forward-prefix-import", "schema_version": 1, "predecessor_binding_semantic_sha256": predecessor["semantic_sha256"], "trajectory_name": "fresh-main-path",
        "imported_shard_indices": list(range(FORWARD_SHARDS)), "imported_step_range": [0, 511], "transition_count": FORWARD_TRANSITION_COUNT,
        "anchor_copies": [by_role[path] for path in _v2_operational_paths()[4:6]], "entries": forward_entries, "imported_shard_count": FORWARD_SHARDS, "generated_child_shard_count": 0, "prefix_chain_valid": 1,
        "predecessor_forward_summary_file_sha256": _V2_EXPECTED_HASHES["forward/forward_summary.json"][0], "predecessor_forward_summary_semantic_sha256": _V2_EXPECTED_HASHES["forward/forward_summary.json"][1],
    })
    reverse_root = "reverse/fused_families/same-path-three-row/complete-512"; shard_state = _load_npz(run_dir / reverse_root / "shard-0000.npz")["state"]
    zero_metric = raw_state_metrics(shard_state[0], source.mixed_target)
    reverse_import = _write_semantic(run_dir / "reverse/prefix_import.json", {
        "schema": VERSION + "-reverse-prefix-import", "schema_version": 1, "predecessor_binding_semantic_sha256": predecessor["semantic_sha256"],
        "imported_shard_indices": [0], "generated_child_shard_indices": list(range(1, REVERSE_SHARDS)), "imported_reverse_steps": 8, "remaining_reverse_steps": 504,
        "shard_0_copies": [by_role[f"{reverse_root}/shard-0000.{suffix}"] for suffix in ("json", "npz")], "shard_0_elapsed_seconds": float(_read_json(run_dir / reverse_root / "shard-0000.json", semantic=True)["elapsed_seconds"]),
        "shard_0_health": prefix_authority["reverse_shard_0_health"], "shard_1_failure_capsule_sha256": _V2_EXPECTED_HASHES[f"{reverse_root}/shard-0001.failure.json"][0],
        "partial_mixed_target_metrics": {"completed_reverse_steps": 8, "zero_squared_l2_error": zero_metric.squared_l2_error, "global": paired_metric_improvement(raw_state_metrics(shard_state[1], source.mixed_target), zero_metric), "source": paired_metric_improvement(raw_state_metrics(shard_state[2], source.mixed_target), zero_metric), "interpretation_scope": "integrity_cross_check_only"},
        "operational_failure_record_present": 0, "prefix_chain_valid": 1,
    })
    _verify_imported_inputs(run_dir)
    if tuple(_snapshot_tree(root) for root in (paths.prefix_run_dir, paths.parent_run_dir, paths.source_run_dir)) != external_before:
        raise ContinuationIntegrityError("prepare mutated an external evidence tree")
    event = _finish_durable_attempt(
        run_dir, journal, durable_elapsed_seconds=0.0,
        invocation_wall_seconds=time.perf_counter() - started + 5.0,
        detail={"operational_copy_count": len(operational), "capsule_copy_count": len(capsule), "binding_passed": 1},
    )
    _stage_marker(
        run_dir, "prepare",
        {
            "predecessor_binding_semantic_sha256": predecessor["semantic_sha256"], "parent_binding_semantic_sha256": parent_binding["semantic_sha256"],
            "input_bindings_semantic_sha256": inputs["semantic_sha256"], "forward_prefix_import_semantic_sha256": forward_import["semantic_sha256"], "reverse_prefix_import_semantic_sha256": reverse_import["semantic_sha256"],
            "prefix_resource_event_id": carry["event_id"], "prepare_resource_event_id": event["event_id"],
        },
    )
def _run_controls(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir = Path(run_dir)
    journal = _begin_durable_attempt(
        run_dir, role="controls", attempt=_resource_event_count(run_dir, "controls") + 1,
        durable_elapsed_seconds=0.0,
    )
    started = float(journal["started_monotonic"])
    _verify_external_trees(run_dir, args)
    _verify_imported_inputs(run_dir)
    source = _load_child_source(run_dir)
    forward_root = run_dir / "forward/forward_shards/fresh-main-path"
    reverse_root = run_dir / "reverse/fused_families/same-path-three-row/complete-512"
    before = (_snapshot_tree(forward_root), _snapshot_tree(reverse_root))
    forward = _scan_forward_chain(run_dir); reverse = _scan_reverse_chain(run_dir)
    if forward["first_missing"] != FORWARD_SHARDS or len(forward["records"]) != FORWARD_SHARDS or reverse["first_missing"] != IMPORTED_REVERSE_SHARDS or len(reverse["records"]) != IMPORTED_REVERSE_SHARDS or reverse["orphan"] is not None or reverse["failure"] is not None:
        raise ContinuationIntegrityError("copied operational prefix is not exactly 64 forward plus reverse shard 0")
    forward_health = aggregate_exact_forward_shards(forward["records"], expected_shard_count=FORWARD_SHARDS, expected_transition_count=FORWARD_TRANSITION_COUNT, expected_path_ids=(PATH_ID,))
    shard_state = _load_npz(reverse_root / "shard-0000.npz")["state"]
    reverse_health = _strict_fused_exact_health(final_state=shard_state, shard_records=reverse["records"], expected_shard_count=IMPORTED_REVERSE_SHARDS)
    if forward_health.get("passed") != 1 or forward_health.get("certificate_fraction") != 1.0:
        raise ContinuationIntegrityError("copied full forward health changed")
    zero_metric = raw_state_metrics(shard_state[0], source.mixed_target)
    partial = {
        "completed_reverse_steps": 8,
        "zero_mixed_target_squared_l2_error": zero_metric.squared_l2_error,
        "global": paired_metric_improvement(raw_state_metrics(shard_state[1], source.mixed_target), zero_metric),
        "source": paired_metric_improvement(raw_state_metrics(shard_state[2], source.mixed_target), zero_metric),
        "interpretation_scope": "integrity_cross_check_only",
    }
    ledger = _validate_resource_ledger(run_dir)
    next_seconds = max(REVERSE_BASELINE_SECONDS, 1.20 * float(reverse["records"][0]["elapsed_seconds"]))
    projection = _resource_projection(
        active_seconds=float(ledger["active_seconds"]),
        current_attempt_wall=time.perf_counter() - started,
        remaining_forward_shards=0,
        remaining_reverse_shards=REVERSE_SHARDS - IMPORTED_REVERSE_SHARDS,
        forward_next_seconds=0.0,
        reverse_next_seconds=next_seconds,
        postprocess_seconds=POSTPROCESS_RESERVE_SECONDS,
        report_seconds=REPORT_RESERVE_SECONDS,
        active_seconds_cap=float(ledger["caps"]["active_seconds"]),
    )
    if not projection.admitted:
        raise ResourceBoundaryError("successor continuation does not fit the 22500-second cap after setup")
    _admit_storage_projection(run_dir, remaining_forward_shards=0, remaining_reverse_shards=REVERSE_SHARDS - IMPORTED_REVERSE_SHARDS)
    if (run_dir / "forward/forward_summary.json").exists() or (run_dir / "reverse/family_summary.json").exists() or (run_dir / "recovery").exists():
        raise ContinuationIntegrityError("child-generated evidence exists before continuation freeze")
    freeze_body = _continuation_freeze_body(run_dir)
    freeze_path = run_dir / "continuation_freeze.json"
    expected_freeze = _semantic(freeze_body)
    if freeze_path.is_file():
        freeze = _read_json(freeze_path, semantic=True)
        if _canonical_json_bytes(freeze) != _canonical_json_bytes(expected_freeze):
            raise ContinuationIntegrityError("sealed continuation freeze changed")
    else:
        freeze = _write_semantic(freeze_path, freeze_body)
    if (_snapshot_tree(forward_root), _snapshot_tree(reverse_root)) != before:
        raise ContinuationIntegrityError("controls modified copied operational evidence")
    _verify_external_trees(run_dir, args)
    event = _finish_durable_attempt(
        run_dir, journal, durable_elapsed_seconds=0.0,
        invocation_wall_seconds=time.perf_counter() - started + 5.0,
        detail={"freeze_semantic_sha256": freeze["semantic_sha256"], "partial_metrics": partial, "forward_health_semantic_sha256": forward_health["semantic_sha256"], "reverse_health": reverse_health},
    )
    _stage_marker(
        run_dir, "controls",
        {"freeze_semantic_sha256": freeze["semantic_sha256"], "resource_event_id": event["event_id"]},
    )
def _run_forward_tail(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir = Path(run_dir)
    root = run_dir / "forward/forward_shards/fresh-main-path"
    journal = _begin_durable_attempt(
        run_dir, role="forward_tail",
        attempt=_resource_event_count(run_dir, "forward_tail") + 1,
        durable_elapsed_seconds=0.0,
    )
    started = float(journal["started_monotonic"])
    _verify_external_trees(run_dir, args)
    _verify_imported_inputs(run_dir)
    _verify_continuation_freeze_exact(run_dir)
    operational_before = _snapshot_tree(root)
    scan = _scan_forward_chain(run_dir)
    if scan["first_missing"] != FORWARD_SHARDS or len(scan["records"]) != FORWARD_SHARDS or scan["orphan"] is not None or scan["failure"] is not None:
        raise ContinuationIntegrityError("imported complete forward chain is incomplete")
    diagnostics = aggregate_exact_forward_shards(
        scan["records"], expected_shard_count=FORWARD_SHARDS,
        expected_transition_count=FORWARD_TRANSITION_COUNT, expected_path_ids=(PATH_ID,),
    )
    if (
        diagnostics.get("passed") != 1
        or diagnostics.get("restart_chain_valid") != 1
        or diagnostics.get("path_ids") != [PATH_ID]
        or diagnostics.get("authorization_fraction") != 1.0
        or diagnostics.get("certificate_fraction") != 1.0
        or diagnostics.get("forbidden_event_count") != 0
        or diagnostics.get("output_state_nonfinite_count") != 0
        or diagnostics.get("output_state_negative_count") != 0
        or diagnostics.get("maximum_output_state_mass_error", math.inf) > 2.0e-12
        or max(diagnostics.get(name, math.inf) for name in ("maximum_global_mass_error", "maximum_pair_mass_error", "maximum_simplex_mass_error")) > 2.0e-12
    ):
        raise ContinuationIntegrityError("imported complete forward health failed")
    final_state = np.ascontiguousarray(_load_npz(root / "shard-0063.npz")["state"][0], dtype=np.float64)
    anchor = _load_anchor_511(run_dir)
    if not np.array_equal(final_state, anchor) or rollout_array_sha256(anchor) != STEP511_STATE_SHA256 or _sha256_file(run_dir / "forward/anchor-step-0511.npz") != STEP511_ARCHIVE_SHA256:
        raise ContinuationIntegrityError("imported step-511 anchor differs from shard 63")
    reverse = _scan_reverse_chain(run_dir); observed = [float(row["elapsed_seconds"]) for row in reverse["records"]]
    next_seconds = max(REVERSE_BASELINE_SECONDS, 1.20 * max(observed))
    ledger = _validate_resource_ledger(run_dir)
    projection = _resource_projection(
        active_seconds=float(ledger["active_seconds"]), current_attempt_wall=time.perf_counter() - started,
        remaining_forward_shards=0, remaining_reverse_shards=REVERSE_SHARDS - IMPORTED_REVERSE_SHARDS,
        forward_next_seconds=0.0, reverse_next_seconds=next_seconds,
        postprocess_seconds=POSTPROCESS_RESERVE_SECONDS, report_seconds=REPORT_RESERVE_SECONDS,
        active_seconds_cap=float(ledger["caps"]["active_seconds"]),
    )
    if not projection.admitted: raise ResourceBoundaryError("imported prefix setup cannot preserve reverse/report reserve")
    _admit_storage_projection(run_dir, remaining_forward_shards=0, remaining_reverse_shards=REVERSE_SHARDS - IMPORTED_REVERSE_SHARDS)
    summary = _write_semantic(
        run_dir / "forward/forward_summary.json",
        {
            "schema": VERSION + "-forward-summary", "schema_version": 1,
            "trajectory_name": "fresh-main-path",
            "path_id": PATH_ID, "root_seed": FORWARD_ROOT_SEED,
            "initial_mixed_target_sha256": MIXED_TARGET_ARRAY_SHA256,
            "forward_prefix_import_semantic_sha256": _read_json(run_dir / "forward/prefix_import.json", semantic=True)["semantic_sha256"],
            "predecessor_forward_summary_semantic_sha256": _V2_EXPECTED_HASHES["forward/forward_summary.json"][1],
            "profile_sha256": DEFAULT_PROFILE_SHA256,
            "shard_count": FORWARD_SHARDS,
            "imported_shard_count": IMPORTED_FORWARD_SHARDS,
            "generated_child_shard_count": 0, "sampler_called": 0,
            "transition_count": FORWARD_TRANSITION_COUNT,
            "predecessor_forward_elapsed_seconds": math.fsum(float(row["elapsed_seconds"]) for row in scan["records"]),
            "child_generated_forward_elapsed_seconds": 0.0,
            "child_validation_wall_seconds_through_summary": time.perf_counter() - started,
            "step_127_anchor_sha256": STEP127_STATE_SHA256,
            "step_127_archive_sha256": STEP127_ARCHIVE_SHA256,
            "step_511_anchor_sha256": rollout_array_sha256(final_state),
            "step_511_archive_sha256": _sha256_file(run_dir / "forward/anchor-step-0511.npz"),
            "strict_forward_health": diagnostics,
            "strict_forward_health_semantic_sha256": diagnostics["semantic_sha256"],
            "resource_projection_seconds": projection.projected_active_seconds,
        },
    )
    if _snapshot_tree(root) != operational_before: raise ContinuationIntegrityError("forward no-op stage modified an imported shard")
    _verify_external_trees(run_dir, args)
    event = _finish_durable_attempt(
        run_dir, journal, durable_elapsed_seconds=0.0,
        invocation_wall_seconds=time.perf_counter() - started + 5.0,
        detail={"sampler_called": 0, "imported_shards": FORWARD_SHARDS, "generated_shards": 0},
    )
    _stage_marker(
        run_dir, "forward_tail",
        {"forward_summary_semantic_sha256": summary["semantic_sha256"], "resource_event_id": event["event_id"]},
    )
def _run_reverse_complete(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir = Path(run_dir)
    role = "reverse_complete"
    root = run_dir / "reverse/fused_families/same-path-three-row/complete-512"
    durable_before = _committed_elapsed(root)
    journal = _begin_durable_attempt(
        run_dir,
        role=role,
        attempt=_resource_event_count(run_dir, role) + 1,
        durable_elapsed_seconds=durable_before,
    )
    started = float(journal["started_monotonic"])
    _verify_external_trees(run_dir, args)
    _verify_imported_inputs(run_dir)
    freeze = _verify_continuation_freeze_exact(run_dir)
    sequence = tuple(reverse_suffix_sequence(511))
    sequence_record = [list(item) for item in sequence]
    sequence_sha256 = semantic_sha256(sequence_record)
    capture_contract = [
        {"coordinate": list(coordinate), "name": name}
        for coordinate, name in CAPTURE_COORDINATES.items()
    ]
    if (
        freeze.get("sealed") != 1
        or freeze.get("reverse_sequence") != sequence_record
        or freeze.get("row_table") != _complete_row_table_authority()
        or freeze.get("checkpoint_state_sha256") != CHECKPOINT_STATE_SHA256
        or freeze.get("source_target_sha256") != MIXED_TARGET_ARRAY_SHA256
        or freeze.get("controller_binding") != _complete_controller_binding_authority()
        or freeze.get("rng_binding") != _complete_rng_binding_authority()
        or freeze.get("variant_in_rng_key") != 0
    ):
        raise ContinuationIntegrityError("continuation freeze cannot authorize reverse sampling")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ContinuationIntegrityError("production reverse continuation requires available CUDA")
    peak_before, total_before = _cuda_memory_snapshot(device)
    if total_before <= 0 or peak_before / total_before >= CUDA_MEMORY_FRACTION_CAP:
        raise ResourceBoundaryError("CUDA allocation cannot admit reverse preparation")
    source = _load_child_source(run_dir)
    model, _ = _strict_load_global_checkpoint(run_dir, device=device)
    specs, bank, controller_binding = _build_complete_rows(
        model=model, mixed_target=source.mixed_target, device=device
    )
    if controller_binding != freeze["controller_binding"]:
        raise ContinuationIntegrityError("runtime controller binding differs from freeze")
    anchor = _load_anchor_511(run_dir)
    imported_paths = (root / "shard-0000.json", root / "shard-0000.npz")
    imported_hashes = tuple(_sha256_file(path) for path in imported_paths)
    if imported_hashes != (freeze.get("imported_reverse_shard_0_json_sha256"), freeze.get("imported_reverse_shard_0_npz_sha256")):
        raise ContinuationIntegrityError("imported reverse shard 0 differs from the freeze")
    scan = _scan_reverse_chain(run_dir)
    first_missing = int(scan["first_missing"])
    if first_missing < IMPORTED_REVERSE_SHARDS:
        raise ContinuationIntegrityError("reverse successor would replay imported shard 0")
    if scan["records"]:
        prefix_state = _load_npz(root / f"shard-{first_missing - 1:04d}.npz")["state"]
        _strict_fused_exact_health(final_state=prefix_state, shard_records=scan["records"], expected_shard_count=first_missing)
    recovery: dict[str, Any] | None = None
    if scan["orphan"] is not None:
        recovery, _ = _archive_replay_evidence(
            run_dir,
            root=root,
            role="reverse",
            shard_index=first_missing,
            attempt_id=str(journal["attempt_id"]),
        )
    elif scan.get("failure") is not None:
        recovery = _recover_replay_evidence(
            run_dir, root=root, role="reverse",
            shard_index=first_missing,
            attempt_id=str(journal["attempt_id"]),
            require_orphan=False,
        )

    profile = _default_profile()
    prepared = _prepared_exact_backend(device, profile)
    reference_factory = _exact_reference_factory(prepared=prepared, profile=profile)
    rng_binding = _complete_rng_binding_authority()
    expected_per_shard = FUSED_SHARD_PHASES * 2 * MICROSTEPS * len(ROW_ORDER) * EDGES_PER_PHASE
    def admit(plan: FusedShardExecutionPlan) -> None:
        if (
            plan.shard_index < IMPORTED_REVERSE_SHARDS
            or plan.shard_index < first_missing
            or plan.row_count != len(ROW_ORDER)
            or len(plan.sequence) != FUSED_SHARD_PHASES
            or plan.transition_count != expected_per_shard
            or tuple(plan.sequence)
            != sequence[
                plan.shard_index * FUSED_SHARD_PHASES :
                (plan.shard_index + 1) * FUSED_SHARD_PHASES
            ]
        ):
            raise ContinuationIntegrityError("reverse sampler execution plan changed")
        committed = _scan_reverse_chain(run_dir)
        if committed["first_missing"] != plan.shard_index or tuple(_sha256_file(path) for path in imported_paths) != imported_hashes:
            raise ContinuationIntegrityError("reverse committed prefix or imported shard 0 changed")
        prior_state = _load_npz(root / f"shard-{plan.shard_index - 1:04d}.npz")["state"]
        _strict_fused_exact_health(final_state=prior_state, shard_records=committed["records"], expected_shard_count=plan.shard_index)
        observed = [
            float(record["elapsed_seconds"])
            for record in _scan_reverse_chain(run_dir)["records"]
        ]
        next_seconds = max(
            REVERSE_BASELINE_SECONDS,
            1.20 * max(observed) if observed else 0.0,
        )
        ledger = _validate_resource_ledger(run_dir)
        projection = _resource_projection(
            active_seconds=float(ledger["active_seconds"]),
            current_attempt_wall=time.perf_counter() - started,
            remaining_forward_shards=0,
            remaining_reverse_shards=REVERSE_SHARDS - int(plan.shard_index),
            forward_next_seconds=0.0,
            reverse_next_seconds=next_seconds,
            postprocess_seconds=POSTPROCESS_RESERVE_SECONDS,
            report_seconds=REPORT_RESERVE_SECONDS,
            active_seconds_cap=float(ledger["caps"]["active_seconds"]),
        )
        if not projection.admitted:
            raise ResourceBoundaryError(
                f"reverse shard {plan.shard_index} cannot preserve postprocess/report reserve"
            )
        _admit_storage_projection(
            run_dir,
            remaining_forward_shards=0,
            remaining_reverse_shards=REVERSE_SHARDS - int(plan.shard_index),
        )
        peak, total = _cuda_memory_snapshot(device)
        if total <= 0 or peak / total >= CUDA_MEMORY_FRACTION_CAP:
            raise ResourceBoundaryError("CUDA allocation cannot admit another reverse shard")

    result = run_fused_reverse_family(
        np.repeat(anchor[None, :], len(ROW_ORDER), axis=0),
        sequence=sequence,
        output_dir=run_dir / "reverse",
        family_name="same-path-three-row",
        segment_name="complete-512",
        row_specs=specs,
        controller_bank=bank,
        reference_factory=reference_factory,
        controller_binding=controller_binding,
        rng_binding=rng_binding,
        label=3,
        microsteps=MICROSTEPS,
        device=device,
        capture_coordinates=CAPTURE_COORDINATES,
        before_uncommitted_shard=admit,
        reference_contract="certified_exact",
    )
    scan = _scan_reverse_chain(run_dir)
    if scan["first_missing"] != REVERSE_SHARDS or len(scan["records"]) != REVERSE_SHARDS:
        raise ContinuationIntegrityError("complete reverse family is incomplete")
    if tuple(_sha256_file(path) for path in imported_paths) != imported_hashes:
        raise ContinuationIntegrityError("reverse execution modified imported shard 0")
    if recovery is not None and recovery.get("orphan") is not None:
        _verify_orphan_replay_matches(
            run_dir / str(recovery["archive_relative_path"]),
            root / f"shard-{first_missing:04d}.npz",
            expected_rows=len(ROW_ORDER),
        )
    returned_state = getattr(result, "final_state", None)
    final_arrays = _load_npz(root / "shard-0063.npz")
    committed_final_state = final_arrays.get("state")
    final_record = scan["records"][-1]
    if (
        set(final_arrays) != {"state"}
        or not isinstance(returned_state, np.ndarray)
        or returned_state.dtype != np.float64
        or returned_state.shape != (len(ROW_ORDER), STATE_SIZE)
        or not returned_state.flags.c_contiguous
        or not isinstance(committed_final_state, np.ndarray)
        or committed_final_state.dtype != np.float64
        or committed_final_state.shape != returned_state.shape
        or not np.array_equal(returned_state, committed_final_state)
        or final_record.get("output_state_sha256")
        != rollout_array_sha256(returned_state)
        or final_record.get("state_file_sha256")
        != _sha256_file(root / "shard-0063.npz")
    ):
        raise ContinuationIntegrityError(
            "returned reverse final state differs from committed shard-0063"
        )
    health = _strict_fused_exact_health(
        final_state=returned_state,
        shard_records=scan["records"],
        row_count=len(ROW_ORDER),
    )
    to_record = getattr(result, "to_record", None)
    if not callable(to_record): raise ContinuationIntegrityError("reverse family result record is absent")
    raw_result_record = to_record()
    if not isinstance(raw_result_record, Mapping): raise ContinuationIntegrityError("reverse family result record is malformed")
    result_record = dict(raw_result_record)
    if (
            result_record.get("schema")
            != FUSED_TANGENT_VERSION + "-reverse-family-result"
            or result_record.get("row_table") != _complete_row_table_authority()
            or result_record.get("final_state_sha256")
            != rollout_array_sha256(returned_state)
            or result_record.get("transition_count") != REVERSE_TRANSITION_COUNT
            or result_record.get("shard_count") != REVERSE_SHARDS
    ):
        raise ContinuationIntegrityError("reverse family result authority changed")
    shard_record_paths = [
        "reverse/fused_families/same-path-three-row/complete-512/"
        f"shard-{index:04d}.json"
        for index in range(REVERSE_SHARDS)
    ]
    family_body: dict[str, Any] = {
        "schema": VERSION + "-family-summary",
        "schema_version": 1,
        "completed": 1,
        "row_order": list(ROW_ORDER),
        "row_table": _complete_row_table_authority(),
        "family_name": "same-path-three-row",
        "segment_name": "complete-512",
        "shard_count": REVERSE_SHARDS,
        "imported_reverse_shard_count": IMPORTED_REVERSE_SHARDS,
        "child_generated_reverse_shard_count": REVERSE_SHARDS - IMPORTED_REVERSE_SHARDS,
        "imported_reverse_steps": 8, "child_generated_reverse_steps": 504,
        "transition_count": REVERSE_TRANSITION_COUNT,
        "freeze_semantic_sha256": freeze["semantic_sha256"],
        "sequence": sequence_record,
        "sequence_sha256": sequence_sha256,
        "capture_coordinates": capture_contract,
        "reference_contract": "certified_exact",
        "controller_binding_sha256": semantic_sha256(controller_binding),
        "rng_binding": rng_binding,
        "rng_binding_sha256": semantic_sha256(rng_binding),
        "initial_state_sha256": rollout_array_sha256(
            np.repeat(anchor[None, :], len(ROW_ORDER), axis=0)
        ),
        "final_state_sha256": rollout_array_sha256(returned_state),
        "shard_record_paths": shard_record_paths,
        "strict_exact_health": health,
        "failed_rows_suppressed": 0,
        "orphan_recovery": recovery,
    }
    family_body["result"] = result_record
    family_summary = _write_semantic(
        run_dir / "reverse/family_summary.json",
        family_body,
    )
    durable_after = _committed_elapsed(root)
    peak, total = _cuda_memory_snapshot(device)
    event = _finish_durable_attempt(
        run_dir,
        journal,
        durable_elapsed_seconds=durable_after,
        invocation_wall_seconds=time.perf_counter() - started + 5.0,
        peak_cuda_bytes=peak,
        total_cuda_bytes=total,
        detail={
            "first_missing": first_missing,
            "shard_count": REVERSE_SHARDS,
            "family_summary_semantic_sha256": family_summary["semantic_sha256"],
        },
    )

    ledger = _validate_resource_ledger(run_dir)
    if (
        math.fsum((float(ledger["active_seconds"]), POSTPROCESS_RESERVE_SECONDS, REPORT_RESERVE_SECONDS))
        > float(ledger["caps"]["active_seconds"])
    ):
        raise ResourceBoundaryError("postprocessing cannot preserve report reserve")
    post_journal = _begin_durable_attempt(
        run_dir,
        role="reverse_postprocess",
        attempt=_resource_event_count(run_dir, "reverse_postprocess") + 1,
        durable_elapsed_seconds=0.0,
    )
    post_started = float(post_journal["started_monotonic"])
    states, milestones, aggregation = _aggregate_reverse_boundaries(
        run_dir, shard_records=scan["records"]
    )
    summary, mechanism, outcome = _build_reverse_derived(
        run_dir,
        states=states,
        milestones=milestones,
        shard_records=scan["records"],
    )
    _verify_external_trees(run_dir, args)
    post_event = _finish_durable_attempt(
        run_dir,
        post_journal,
        durable_elapsed_seconds=0.0,
        invocation_wall_seconds=time.perf_counter() - post_started + 5.0,
        detail={
            "aggregation_semantic_sha256": aggregation["semantic_sha256"],
            "summary_semantic_sha256": summary["semantic_sha256"],
            "mechanism_semantic_sha256": mechanism["semantic_sha256"],
            "outcome_semantic_sha256": outcome["semantic_sha256"],
        },
    )
    if int(outcome.get("learned_interpretation_authorized", 0)) != 1:
        raise CompositionControlError(
            "complete source-informed control was not practically informative"
        )
    _stage_marker(
        run_dir,
        "reverse_complete",
        {
            "family_summary_semantic_sha256": family_summary["semantic_sha256"],
            "summary_semantic_sha256": summary["semantic_sha256"],
            "outcome_semantic_sha256": outcome["semantic_sha256"],
            "sampler_resource_event_id": event["event_id"],
            "postprocess_resource_event_id": post_event["event_id"],
        },
    )
def _completed_stage_artifacts_read_only(run_dir: Path, stage: str) -> dict[str, Any]:
    marker = _read_stage_marker_exact(run_dir, stage)
    detail = marker["detail"]
    checks: dict[str, Path] = {
        "predecessor_binding_semantic_sha256": run_dir / "predecessor_binding.json",
        "parent_binding_semantic_sha256": run_dir / "parent_binding.json",
        "input_bindings_semantic_sha256": run_dir / "input_bindings.json",
        "forward_prefix_import_semantic_sha256": run_dir / "forward/prefix_import.json",
        "reverse_prefix_import_semantic_sha256": run_dir / "reverse/prefix_import.json",
        "freeze_semantic_sha256": run_dir / "continuation_freeze.json",
        "forward_summary_semantic_sha256": run_dir / "forward/forward_summary.json",
        "family_summary_semantic_sha256": run_dir / "reverse/family_summary.json",
        "summary_semantic_sha256": run_dir / "reverse/summary.json",
        "outcome_semantic_sha256": run_dir / "outcome.json",
    }
    required = {
        "prepare": ("predecessor_binding_semantic_sha256", "parent_binding_semantic_sha256", "input_bindings_semantic_sha256", "forward_prefix_import_semantic_sha256", "reverse_prefix_import_semantic_sha256"),
        "controls": ("freeze_semantic_sha256",),
        "forward_tail": ("forward_summary_semantic_sha256",),
        "reverse_complete": ("family_summary_semantic_sha256", "summary_semantic_sha256", "outcome_semantic_sha256"),
        "report_verify": (),
    }[stage]
    for key in required:
        if not checks[key].is_file() or _read_json(checks[key], semantic=True)["semantic_sha256"] != detail.get(key):
            raise ContinuationIntegrityError(f"completed {stage} artifact binding changed: {key}")
    if stage == "forward_tail":
        scan = _scan_forward_chain(run_dir)
        if scan["first_missing"] != FORWARD_SHARDS: raise ContinuationIntegrityError("completed forward chain changed")
        health = aggregate_exact_forward_shards(scan["records"], expected_shard_count=FORWARD_SHARDS, expected_transition_count=FORWARD_TRANSITION_COUNT, expected_path_ids=(PATH_ID,))
        if health.get("passed") != 1 or health.get("authorization_fraction") != 1.0 or max(health.get(key, math.inf) for key in ("maximum_global_mass_error", "maximum_pair_mass_error", "maximum_simplex_mass_error")) > 2e-12: raise ContinuationIntegrityError("completed forward health changed")
    elif stage == "reverse_complete":
        scan = _scan_reverse_chain(run_dir)
        if scan["first_missing"] != REVERSE_SHARDS: raise ContinuationIntegrityError("completed reverse chain changed")
        state = _load_npz(run_dir / "reverse/fused_families/same-path-three-row/complete-512/shard-0063.npz")["state"]
        _strict_fused_exact_health(final_state=state, shard_records=scan["records"])
    elif stage == "controls": _verify_continuation_freeze_exact(run_dir)
    return marker
def _manifest_rows(run_dir: Path, *, exclusions: Sequence[str] = _TERMINAL_EXCLUSIONS) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    _reject_links_and_reparse_points(run_dir)
    excluded = set(exclusions)
    rows: list[dict[str, Any]] = []
    for path in sorted((item for item in run_dir.rglob("*") if item.is_file()), key=lambda item: item.relative_to(run_dir).as_posix()):
        relative = path.relative_to(run_dir).as_posix()
        if relative in excluded:
            continue
        if ".tmp" in path.name or path.stat().st_nlink != 1:
            raise ContinuationIntegrityError("terminal tree contains temporary/linked evidence")
        rows.append({"path": relative, "size": int(path.stat().st_size), "sha256": _sha256_file(path)})
    return rows
def _expected_artifact_path_set(run_dir: Path, terminal_kind: str) -> set[str]:
    exclusions = _TERMINAL_EXCLUSIONS if terminal_kind == "success" else (
        "SHA256SUMS.txt", "artifact_manifest.json", "terminal_storage_authority.json", "verification.json",
    )
    return {row["path"] for row in _manifest_rows(run_dir, exclusions=exclusions)} | set(exclusions)
def _write_manifest_and_checksums(run_dir: Path, *, terminal_kind: str) -> tuple[dict[str, Any], str]:
    exclusions = _TERMINAL_EXCLUSIONS if terminal_kind == "success" else (
        "SHA256SUMS.txt", "artifact_manifest.json", "terminal_storage_authority.json", "verification.json",
    )
    rows = _manifest_rows(run_dir, exclusions=exclusions)
    manifest = _write_semantic(run_dir / "artifact_manifest.json", {
        "schema": VERSION + "-artifact-manifest", "schema_version": 1,
        "terminal_kind": terminal_kind, "artifact_count": len(rows),
        "excluded_self_referential_paths": list(exclusions), "artifacts": rows,
    })
    checks = sorted((*((row["path"], row["sha256"]) for row in rows), ("artifact_manifest.json", _sha256_file(run_dir / "artifact_manifest.json"))))
    _write_bytes_atomic(run_dir / "SHA256SUMS.txt", "".join(f"{digest}  {path}\n" for path, digest in checks).encode("utf-8"))
    return manifest, _sha256_file(run_dir / "SHA256SUMS.txt")
def _verify_terminal_inventory_read_only(run_dir: Path, terminal_kind: str, *, pending_success_marker: bool = False) -> dict[str, Any]:
    _reject_links_and_reparse_points(run_dir)
    manifest = _read_json(run_dir / "artifact_manifest.json", semantic=True)
    rows = manifest.get("artifacts")
    exclusions = manifest.get("excluded_self_referential_paths")
    expected_exclusions = list(_TERMINAL_EXCLUSIONS) if terminal_kind == "success" else [
        "SHA256SUMS.txt", "artifact_manifest.json", "terminal_storage_authority.json", "verification.json",
    ]
    if (manifest.get("terminal_kind") != terminal_kind or not isinstance(rows, list)
            or manifest.get("artifact_count") != len(rows) or exclusions != expected_exclusions):
        raise ContinuationIntegrityError("terminal manifest contract changed")
    paths: set[str] = set()
    for row in rows:
        relative = str(row.get("path", "")); path = (run_dir / relative).resolve()
        if path.parent != run_dir.resolve() and run_dir.resolve() not in path.parents:
            raise ContinuationIntegrityError("manifest path escapes child")
        if relative in paths or not path.is_file() or path.stat().st_nlink != 1 or path.stat().st_size != row.get("size") or _sha256_file(path) != row.get("sha256"):
            raise ContinuationIntegrityError("terminal manifest artifact changed")
        paths.add(relative)
    checksum: dict[str, str] = {}
    for line in (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] in checksum: raise ContinuationIntegrityError("checksum syntax/path changed")
        checksum[parts[1]] = parts[0]
    expected = paths | {"artifact_manifest.json"}
    if list(checksum) != sorted(expected) or any(_sha256_file(run_dir / path) != checksum[path] for path in expected):
        raise ContinuationIntegrityError("terminal checksum inventory changed")
    actual = {path for path, _, _ in _snapshot_tree(run_dir)}
    expected_actual = paths | set(expected_exclusions)
    if pending_success_marker and terminal_kind == "success": expected_actual.remove("stages/report_verify.json")
    if actual != expected_actual: raise ContinuationIntegrityError("terminal physical path set changed")
    return {"passed": 1, "artifact_count": len(rows), "checksum_count": len(checksum), "manifest": manifest}
def _deep_verify_scientific_evidence_read_only(run_dir: Path, args: argparse.Namespace | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    imports = _verify_imported_inputs(run_dir)
    for stage in STAGES[:4]: _completed_stage_artifacts_read_only(run_dir, stage)
    forward = _scan_forward_chain(run_dir)
    if forward["first_missing"] != FORWARD_SHARDS: raise ContinuationIntegrityError("terminal forward chain incomplete")
    forward_health = aggregate_exact_forward_shards(forward["records"], expected_shard_count=FORWARD_SHARDS, expected_transition_count=FORWARD_TRANSITION_COUNT, expected_path_ids=(PATH_ID,))
    reverse = _scan_reverse_chain(run_dir)
    if reverse["first_missing"] != REVERSE_SHARDS: raise ContinuationIntegrityError("terminal reverse chain incomplete")
    final = _load_npz(run_dir / "reverse/fused_families/same-path-three-row/complete-512/shard-0063.npz")["state"]
    health = _strict_fused_exact_health(final_state=final, shard_records=reverse["records"])
    trajectory = _load_npz(run_dir / "reverse/trajectory_shard_boundaries.npz")
    milestones = _load_npz(run_dir / "reverse/milestones.npz")
    boundaries = [np.repeat(_load_anchor_511(run_dir)[None, :], 3, axis=0)] + [
        _load_npz(run_dir / f"reverse/fused_families/same-path-three-row/complete-512/shard-{index:04d}.npz")["state"] for index in range(REVERSE_SHARDS)
    ]
    expected_states = np.ascontiguousarray(np.stack(boundaries, axis=1))
    if (set(trajectory) != {"states", "completed_reverse_steps"} or rollout_array_sha256(trajectory["states"]) != rollout_array_sha256(expected_states)
            or set(milestones) != {"states", "completed_reverse_steps"} or rollout_array_sha256(milestones["states"]) != rollout_array_sha256(expected_states[:, (0,16,32,48,64), :])):
        raise ContinuationIntegrityError("terminal trajectory aggregation changed")
    summary = _read_json(run_dir / "reverse/summary.json", semantic=True); outcome = _read_json(run_dir / "outcome.json", semantic=True)
    source = _load_child_source(run_dir)
    final_metrics = {
        key: raw_state_metrics(expected_states[index, -1], source.mixed_target).squared_l2_error
        for index, key in enumerate(ROW_ORDER)
    }
    intermediate = {
        step: {
            "zero_error": raw_state_metrics(expected_states[0, index], source.mixed_target).squared_l2_error,
            "global_error": raw_state_metrics(expected_states[1, index], source.mixed_target).squared_l2_error,
        }
        for index, step in ((16, 128), (32, 256), (48, 384))
    }
    recomputed = _classify_complete_outcome(
        zero_error=final_metrics["zero"], global_error=final_metrics["global-plus-1"],
        source_error=final_metrics["source-informed"], intermediate=intermediate,
    )
    for key in ("zero_error", "global_error", "source_error", "global_effect_label", "source_effect_label", "horizon_classification", "required_next_action"):
        if outcome.get(key) != recomputed.get(key): raise ContinuationIntegrityError("terminal outcome changed")
    metric_rows = list(csv.DictReader((run_dir / "reverse/metrics.csv").open(encoding="utf-8", newline="")))
    if len(metric_rows) != 195 or [(int(row["boundary_index"]), int(row["row_index"])) for row in metric_rows] != [(boundary, row) for boundary in range(65) for row in range(3)]:
        raise ContinuationIntegrityError("terminal metric table changed")
    for index, row in enumerate(metric_rows):
        boundary, row_index = divmod(index, 3); state = expected_states[row_index, boundary]
        expected_metric = raw_state_metrics(state, source.mixed_target).squared_l2_error
        if row.get("row_key") != ROW_ORDER[row_index] or int(row["completed_reverse_steps"]) != boundary * 8 or float(row["mixed_target_squared_l2_error"]) != expected_metric:
            raise ContinuationIntegrityError("terminal metric values changed")
    images = summary.get("images"); sheets = summary.get("contact_sheets")
    if not isinstance(images, list) or len(images) != 30 or not isinstance(sheets, list) or len(sheets) != 14 or summary.get("individual_image_count") != 30 or summary.get("contact_sheet_count") != 14:
        raise ContinuationIntegrityError("terminal image inventory changed")
    for row in (*images, *sheets):
        path = run_dir / str(row.get("path", ""))
        if not path.is_file() or _sha256_file(path) != row.get("file_sha256"): raise ContinuationIntegrityError("terminal image changed")
    scale = FixedRenderingScale(**summary["rendering_scale"])
    expected_pixels = {(ROW_ORDER[row], step, rendering): render(states, scale) for row in range(3) for step, state_index in zip(MILESTONE_STEPS, (0,16,32,48,64)) for rendering, render in (("raw", render_raw_density), ("demixed", render_background_demixed)) for states in (expected_states[row, state_index],)}
    if {(row["row_key"], row["completed_reverse_steps"], row["rendering"]): row["pixel_sha256"] for row in images} != {key: rollout_array_sha256(value) for key, value in expected_pixels.items()}:
        raise ContinuationIntegrityError("terminal rendered pixels changed")
    from PIL import Image
    image_rows = {(row["row_key"], row["completed_reverse_steps"], row["rendering"]): row for row in images}
    for key, pixels in expected_pixels.items():
        row = image_rows[key]; expected_path = f"images/{key[2]}/{key[0]}/step-{key[1]:03d}.png"
        with Image.open(run_dir / expected_path) as opened: actual_pixels = np.asarray(opened.convert("L"))
        if row["path"] != expected_path or not np.array_equal(actual_pixels, pixels): raise ContinuationIntegrityError("terminal image pixels/layout changed")
    for row in sheets:
        rendering = row["rendering"]
        if row["kind"] == "milestone": keys = [(key, row["completed_reverse_steps"], rendering) for key in ROW_ORDER]
        elif row["kind"] == "all-milestones": keys = [(key, step, rendering) for step in MILESTONE_STEPS for key in ROW_ORDER]
        elif row["kind"] == "final": keys = [(key, 512, rendering) for key in ROW_ORDER]
        else: raise ContinuationIntegrityError("terminal contact-sheet kind changed")
        canvas = np.zeros((math.ceil(len(keys) / 3) * 28, 84), dtype=np.uint8)
        for index, key in enumerate(keys): canvas[(index // 3)*28:(index // 3+1)*28, (index % 3)*28:(index % 3+1)*28] = expected_pixels[key]
        with Image.open(run_dir / row["path"]) as opened: actual_canvas = np.asarray(opened.convert("L"))
        if row.get("columns") != 3 or row.get("cell_count") != len(keys) or not np.array_equal(actual_canvas, canvas): raise ContinuationIntegrityError("terminal contact-sheet pixels/layout changed")
    ledger = _validate_resource_ledger(run_dir)
    if (ledger.get("limits_passed") != 1 or ledger.get("breached_limits") != [] or _directory_bytes(run_dir) >= STORAGE_CAP_BYTES
            or (run_dir / "journals").exists() and any((run_dir / "journals").iterdir())):
        raise ContinuationIntegrityError("terminal resource ledger changed")
    if args is not None: _verify_external_trees(run_dir, args)
    return {
        "passed": 1, "forward_health": forward_health, "reverse_health": health,
        "outcome": outcome, "summary": summary, "ledger": ledger, "imports": imports,
        "manifest": _read_json(run_dir / "run_manifest.json", semantic=True),
        "exact_command": (run_dir / "exact_command.txt").read_text(encoding="utf-8").strip(),
    }
def _report_bytes(evidence: Mapping[str, Any], *, failure: Mapping[str, Any] | None = None, prefix_locator: Path | str | None = None) -> tuple[bytes, bytes]:
    imports = evidence.get("imports", {}); predecessor = imports.get("predecessor", {}) if isinstance(imports, Mapping) else {}
    ledger = evidence.get("ledger", {}); events = ledger.get("events", []) if isinstance(ledger, Mapping) else []
    locator = predecessor.get("supplied_prefix_path") if isinstance(predecessor, Mapping) else None
    locator = str(locator or (Path(prefix_locator).resolve() if prefix_locator is not None else "unavailable before authenticated prepare"))
    carried = [event for event in events if isinstance(event, Mapping) and event.get("role") == "prefix_resource_carry"]
    active = float(ledger.get("active_seconds", 0.0)) if isinstance(ledger, Mapping) else 0.0
    carry_complete = isinstance(imports, Mapping) and imports.get("passed") == 1 and len(carried) == 1 and carried[0].get("elapsed_seconds") == V2_CARRIED_ACTIVE_SECONDS and carried[0].get("detail") == _prefix_carry_detail()
    cap = float(ledger.get("caps", {}).get("active_seconds", ACTIVE_SECONDS_CAP)) if isinstance(ledger, Mapping) else ACTIVE_SECONDS_CAP
    child_cost = math.fsum((active, -V2_CARRIED_ACTIVE_SECONDS)) if carry_complete else active
    manifest = evidence.get("manifest", {}); command = evidence.get("exact_command", "unavailable before manifest authentication")
    forward_health = evidence.get("forward_health", imports.get("forward_health", {}) if isinstance(imports, Mapping) else {}); reverse_health = evidence.get("reverse_health", {})
    pinned = predecessor.get("pinned_terminal_hashes", {}).get("run_manifest.json", {}) if isinstance(predecessor, Mapping) else {}
    provenance = (f"Authenticated v2 predecessor locator: `{locator}`. Predecessor binding semantic SHA256: `{predecessor.get('semantic_sha256')}`; prefix tree SHA256: `{predecessor.get('prefix_tree_sha256')}`; pinned v2 manifest file/semantic SHA256: `{pinned.get('file_sha256')}` / `{pinned.get('semantic_sha256')}`.\n\nResource accounting: authenticated v2 carry = {V2_CARRIED_ACTIVE_SECONDS} seconds; current ledger active = {active} seconds; child-ledger cost excluding the carry = {child_cost} seconds; hard active-time cap = {cap} seconds; active-time headroom = {cap - active} seconds." if carry_complete else f"V2 predecessor authentication/carry is incomplete; locator `{locator}` and the configured {V2_CARRIED_ACTIVE_SECONDS}-second predecessor cost are not authenticated or charged by this report. Current ledger active = {active} seconds; hard active-time cap = {cap} seconds; active-time headroom = {cap - active} seconds.")
    shards = "Forward shards: 64 authenticated imports, 0 child-generated. Reverse shards: 1 authenticated import (8 steps), 63 child-generated (504 steps)."
    planned_shards = "Planned full-completion ownership: " + shards
    artifact_paths = "`reverse/trajectory_shard_boundaries.npz`, `reverse/milestones.npz`, `reverse/metrics.csv`, `reverse/mechanism.json`, `reverse/summary.json`, `reverse/family_summary.json`, `outcome.json`, `images/`, `predecessor_binding.json`, `resource_ledger.json`" if failure is None else "`last_valid_evidence.json`, `terminal_failure.json`, `predecessor_binding.json`, `resource_ledger.json`, and the exact existing paths enumerated by `last_valid_evidence.json`"
    reproducibility = f"Exact command: `{command}`. Source revision/closure SHA256: `{manifest.get('source_revision')}` / `{manifest.get('source_closure_sha256')}`. Checkpoint file/state SHA256: `{CHECKPOINT_FILE_SHA256}` / `{CHECKPOINT_STATE_SHA256}`.\n\nExact health: forward passed={forward_health.get('passed')}, certificate_fraction={forward_health.get('certificate_fraction')}, maximum_global_mass_error={forward_health.get('maximum_global_mass_error')}; reverse passed={reverse_health.get('passed')}, certificate_fraction={reverse_health.get('certificate_fraction')}, maximum_mass_error={reverse_health.get('maximum_mass_error')}, fallback_count={reverse_health.get('fallback_count')}.\n\nArtifact paths: {artifact_paths}."
    if failure is None:
        outcome = evidence["outcome"]
        metrics = f"Primary mixed-target squared-L2 errors: zero={outcome.get('zero_error')}, global={outcome.get('global_error')}, source={outcome.get('source_error')}. Relative improvements: global={outcome.get('global_relative_improvement')}, source={outcome.get('source_relative_improvement')}."
        text = f"# Complete same-path continuation\n\nMode: exploratory.\n\n{provenance}\n\n{shards}\n\n{metrics}\n\nOutcome: `{outcome.get('global_effect_label')}`. Source gate: `{outcome.get('source_effect_label')}`; learned interpretation authorized = {outcome.get('learned_interpretation_authorized')}. Required next action: `{outcome.get('required_next_action')}`.\n\n{reproducibility}\n\nAll failed or adverse outputs are retained. This is one opened path under the certified exact backend; it is not a reference-prior, population, diversity, or confirmatory claim.\n"
    else:
        last_valid = evidence.get("last_valid", {}); completed = evidence.get("completed_stages", []); objective = int(evidence.get("objective_completed", 0)); outcome = evidence.get("outcome", {})
        decision_ready = objective == 1 and isinstance(outcome, Mapping) and all(key in outcome for key in ("zero_error", "global_error", "source_error", "global_relative_improvement", "source_relative_improvement", "global_effect_label", "required_next_action"))
        decision = (f"Primary mixed-target squared-L2 errors: zero={outcome['zero_error']}, global={outcome['global_error']}, source={outcome['source_error']}. Relative improvements: global={outcome['global_relative_improvement']}, source={outcome['source_relative_improvement']}. Complete-path effect: `{outcome['global_effect_label']}`. Required next action: `{outcome['required_next_action']}`." if decision_ready else "No complete-path metric, effect classification, or required-action decision is authorized.")
        committed = evidence.get("committed_reverse_shards"); health = evidence.get("reverse_health", {}); family = evidence.get("family", {}); summary = evidence.get("summary", {})
        scope = f"Objective-bearing complete path: {objective}. Committed reverse scope: {committed} of {REVERSE_SHARDS} shards; child-generated committed shards: {max(0, int(committed) - IMPORTED_REVERSE_SHARDS) if isinstance(committed, int) else None}. Exact committed reverse health: passed={health.get('passed')}, certificate_fraction={health.get('certificate_fraction')}, maximum_mass_error={health.get('maximum_mass_error')}, fallback_count={health.get('fallback_count')}. Family/summary semantic SHA256: `{family.get('semantic_sha256')}` / `{summary.get('semantic_sha256')}`."
        text = f"# Continuation terminal failure\n\nMode: exploratory. Domain: `{failure['failure_domain']}`. Message: {failure['message']}\n\n{provenance}\n\n{planned_shards}\n\nCompleted stage scope: {completed}. {scope}\n\n{decision}\n\nLast-valid evidence: semantic SHA256 `{last_valid.get('semantic_sha256')}`, artifact count {len(last_valid.get('artifacts', []))}, raw-path count {len(last_valid.get('available_raw_paths', []))}.\n\n{reproducibility}\n\nAvailable failed, adverse, and partial artifacts are preserved. No learned interpretation is authorized. This remains an exploratory one opened path result with no reference-prior, population, diversity, or confirmatory claim.\n"
    handoff = text + "\nThe immutable v2 predecessor, v3 parent, source tree, and exact child command are bound in the run manifest.\n"
    return text.encode("utf-8"), handoff.encode("utf-8")
def _failure_domain(exc: BaseException) -> str:
    if isinstance(exc, ParentBindingError): return "parent_binding"
    if isinstance(exc, CompositionControlError): return "composition_control"
    if isinstance(exc, ResourceBoundaryError): return "resource_boundary"
    if isinstance(exc, ContinuationIntegrityError): return "child_integrity"
    return "unexpected_engineering_failure"
def _capture_failure(run_dir: Path, stage: str, exc: BaseException) -> dict[str, Any]:
    path = Path(run_dir) / "failure_capture.json"
    captured_at = _utc_now()
    ledger_path = Path(run_dir) / "resource_ledger.json"
    try:
        ledger = _read_json(ledger_path, semantic=True) if ledger_path.is_file() else {}
        ledger_error = None
    except Exception as ledger_exc:
        ledger = {}
        ledger_error = {"type": type(ledger_exc).__name__, "message": str(ledger_exc), "file_sha256": _sha256_file(ledger_path) if ledger_path.is_file() else None}
    body = {
        "schema": VERSION + "-failure-capture", "schema_version": 1,
        "failure_domain": _failure_domain(exc), "stage": stage,
        "exception_type": type(exc).__name__, "message": str(exc),
        "captured_at": captured_at, "learned_interpretation_authorized": 0,
        "resource_ledger_semantic_sha256": ledger.get("semantic_sha256"),
        "resource_ledger_read_error": ledger_error,
        "active_journals": sorted(path.name for path in (Path(run_dir) / "journals").glob("*.json")) if (Path(run_dir) / "journals").is_dir() else [],
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    body["failure_id"] = semantic_sha256({"stage": stage, "type": type(exc).__name__, "message": str(exc), "captured_at": captured_at, "ledger": ledger.get("semantic_sha256")})
    expected = _semantic(body)
    if path.is_file():
        existing = _read_json(path, semantic=True)
        if existing["failure_domain"] != expected["failure_domain"] or existing["stage"] != stage or existing["message"] != str(exc):
            raise ContinuationIntegrityError("failure capture conflicts with prior failure")
        return existing
    return _write_semantic(path, body)
def _last_valid_evidence(run_dir: Path, capture: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir); existing_path = run_dir / "last_valid_evidence.json"
    if existing_path.is_file():
        existing = _read_json(existing_path, semantic=True)
        if existing.get("failure_capture_semantic_sha256") != capture.get("semantic_sha256"): raise ContinuationIntegrityError("last-valid capture binding changed")
        return existing
    excluded = set(_TERMINAL_EXCLUSIONS) | {"artifact_manifest.json", "SHA256SUMS.txt", "terminal_failure.json", "last_valid_evidence.json", "REPORT.md", "HANDOFF.md"}
    rows: list[dict[str, Any]] = []
    for path in sorted((item for item in run_dir.rglob("*") if item.is_file() and not item.is_symlink()), key=lambda item: item.as_posix()):
        relative = path.relative_to(run_dir).as_posix()
        if relative in excluded or relative == "failure_capture.json" or ".tmp" in path.name: continue
        rows.append({"path": relative, "size": int(path.stat().st_size), "sha256": _sha256_file(path)})
    return _write_semantic(run_dir / "last_valid_evidence.json", {
        "schema": VERSION + "-last-valid-evidence", "schema_version": 1,
        "failure_capture_semantic_sha256": capture["semantic_sha256"], "artifacts": rows,
        "available_raw_paths": [row for row in rows if row["path"].endswith((".npz", ".json", ".csv", ".png"))],
    })
def _finalize_failure_package(run_dir: Path, args: argparse.Namespace, capture: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(run_dir); authenticated_imports: Mapping[str, Any] | None = None
    if (run_dir / "stages/prepare.json").is_file():
        _completed_stage_artifacts_read_only(run_dir, "prepare")
        authenticated_imports = _verify_imported_inputs(run_dir)
    evidence = _last_valid_evidence(run_dir, capture)
    terminal = _write_semantic(run_dir / "terminal_failure.json", {
        "schema": VERSION + "-terminal-failure", "schema_version": 1,
        "failure_domain": capture["failure_domain"], "stage": capture["stage"], "message": capture["message"],
        "failure_capture_semantic_sha256": capture["semantic_sha256"], "last_valid_evidence_semantic_sha256": evidence["semantic_sha256"],
        "available_raw_paths": evidence["available_raw_paths"], "learned_interpretation_authorized": 0,
        "resume_same_child_authorized": 0, "scientific_objective_completed": int((run_dir / "reverse/family_summary.json").is_file()),
    })
    context: dict[str, Any] = {"imports": authenticated_imports or {}, "last_valid": evidence, "completed_stages": [stage for stage in STAGES[:4] if (run_dir / f"stages/{stage}.json").is_file()]}
    try: context["ledger"] = _validate_resource_ledger(run_dir)
    except ContinuationError: context["ledger"] = {}
    for key, relative in (("outcome", "outcome.json"), ("summary", "reverse/summary.json"), ("family", "reverse/family_summary.json")):
        try: context[key] = _read_json(run_dir / relative, semantic=True) if (run_dir / relative).is_file() else {}
        except Exception as exc: context[key + "_read_error"] = f"{type(exc).__name__}: {exc}"
    context["objective_completed"] = terminal["scientific_objective_completed"]
    try:
        scan = _scan_reverse_chain(run_dir); context["committed_reverse_shards"] = len(scan["records"])
        if scan["records"]: context["reverse_health"] = _strict_fused_exact_health(final_state=_load_npz(run_dir / f"reverse/fused_families/same-path-three-row/complete-512/shard-{len(scan['records'])-1:04d}.npz")["state"], shard_records=scan["records"], expected_shard_count=len(scan["records"]))
    except Exception as exc: context["reverse_scope_error"] = f"{type(exc).__name__}: {exc}"
    if authenticated_imports is not None:
        context["manifest"] = _read_json(run_dir / "run_manifest.json", semantic=True)
        context["exact_command"] = (run_dir / "exact_command.txt").read_text(encoding="utf-8").strip()
    report, handoff = _report_bytes(context, failure=capture, prefix_locator=args.prefix_run_dir if authenticated_imports is not None else None); _write_bytes_atomic(run_dir / "REPORT.md", report); _write_bytes_atomic(run_dir / "HANDOFF.md", handoff)
    manifest, checks = _write_manifest_and_checksums(run_dir, terminal_kind="failure")
    ledger_path = run_dir / "resource_ledger.json"; now = _utc_now()
    exclusions = (run_dir / "terminal_storage_authority.json", run_dir / "verification.json")
    base = _directory_bytes(run_dir) - sum(path.stat().st_size for path in exclusions if path.is_file()); total = base
    for _ in range(8):
        authority = _semantic({"schema": VERSION + "-terminal-storage-authority", "schema_version": 1, "terminal_kind": "failure", "terminalized_at": now, "resource_ledger_file_sha256": _sha256_file(ledger_path), "exact_recursive_file_bytes": total, "storage_cap_passed": int(total < STORAGE_CAP_BYTES)})
        verification = _semantic({"schema": VERSION + "-verification", "schema_version": 1, "passed": 1, "terminal_kind": "failure", "scientific_objective_completed": terminal["scientific_objective_completed"], "learned_interpretation_authorized": 0, "terminal_failure_semantic_sha256": terminal["semantic_sha256"], "artifact_manifest_semantic_sha256": manifest["semantic_sha256"], "checksums_file_sha256": checks, "terminal_storage_semantic_sha256": authority["semantic_sha256"]})
        blobs = (_canonical_json_bytes(authority), _canonical_json_bytes(verification)); next_total = base + sum(map(len, blobs))
        if next_total == total: break
        total = next_total
    else: raise ContinuationIntegrityError("failure terminal storage fixed point did not converge")
    _write_bytes_atomic(exclusions[0], blobs[0]); _write_bytes_atomic(exclusions[1], blobs[1])
    return _verify_terminal_child_contents_read_only(run_dir, prefix_run_dir=args.prefix_run_dir, parent_run_dir=args.parent_run_dir, source_run_dir=args.source_run_dir, authenticated_imports=authenticated_imports)
def _finalize_success(run_dir: Path, args: argparse.Namespace, admission: Mapping[str, Any]) -> dict[str, Any]:
    evidence = _deep_verify_scientific_evidence_read_only(run_dir, args)
    report, handoff = _report_bytes(evidence, prefix_locator=args.prefix_run_dir)
    _write_bytes_atomic(run_dir / "REPORT.md", report); _write_bytes_atomic(run_dir / "HANDOFF.md", handoff)
    manifest, checksum_hash = _write_manifest_and_checksums(run_dir, terminal_kind="success")
    now = _utc_now(); ledger = _read_json(run_dir / "resource_ledger.json", semantic=True)
    verification_body = {
        "schema": VERSION + "-verification", "schema_version": 1, "passed": 1,
        "terminal_kind": "success", "scientific_objective_completed": 1,
        "learned_interpretation_authorized": 1, "resource_limits_passed": 1,
        "resource_ledger_semantic_sha256": ledger["semantic_sha256"],
        "artifact_manifest_semantic_sha256": manifest["semantic_sha256"],
        "checksums_file_sha256": checksum_hash, "report_admission": dict(admission),
    }
    stage_body = {"schema": VERSION + "-stage", "schema_version": 1, "stage": "report_verify", "passed": 1, "completed_at": now, "detail": {}}
    fixed = _compute_terminal_records_fixed_point(run_dir=run_dir, terminal_kind="success", terminalized_at=now, resource_ledger_semantic_sha256=ledger["semantic_sha256"], verification_body=verification_body, stage_body=stage_body)
    if fixed["exact_recursive_file_bytes"] >= STORAGE_CAP_BYTES: raise ResourceBoundaryError("success terminal package reaches the storage cap")
    _enforce_report_deadline(run_dir, admission)
    for relative in ("terminal_storage_authority.json", "verification.json"):
        _write_bytes_atomic(run_dir / relative, fixed["serialized_bytes"][relative])
    expected_before_marker = fixed["exact_recursive_file_bytes"] - len(fixed["serialized_bytes"]["stages/report_verify.json"])
    if _directory_bytes(run_dir) != expected_before_marker: raise ContinuationIntegrityError("terminal storage changed before final marker")
    # Complete the expensive read-only work before the final marker.  Inventory
    # reopens every pre-marker byte; science was already deeply recomputed above.
    _verify_terminal_inventory_read_only(run_dir, "success", pending_success_marker=True)
    _enforce_report_deadline(run_dir, admission)
    _write_bytes_atomic(run_dir / "stages/report_verify.json", fixed["serialized_bytes"]["stages/report_verify.json"])
    if _directory_bytes(run_dir) != fixed["exact_recursive_file_bytes"]: raise ContinuationIntegrityError("terminal storage fixed point changed after commit")
    try:
        result = _verify_terminal_child_contents_read_only(run_dir, prefix_run_dir=args.prefix_run_dir, parent_run_dir=args.parent_run_dir, source_run_dir=args.source_run_dir)
        _enforce_report_deadline(run_dir, admission)
        return result
    except BaseException:
        (run_dir / "stages/report_verify.json").unlink(missing_ok=True)
        if time.perf_counter() > float(admission["deadline_monotonic"]): _record_report_overrun(run_dir)
        raise
def _run_report_verify(run_dir: Path, args: argparse.Namespace) -> None:
    ledger = _read_json(run_dir / "resource_ledger.json", semantic=True)
    prior = [event for event in ledger.get("events", []) if event.get("role") == "report_verify"]
    if prior:
        if len(prior) != 1 or prior[0].get("elapsed_seconds") != REPORT_RESERVE_SECONDS:
            raise ContinuationIntegrityError("prepaid report event changed")
        admission = prior[0].get("detail", {}).get("prepaid_report_admission")
        if not isinstance(admission, Mapping): raise ContinuationIntegrityError("prepaid report admission is missing")
        _enforce_report_deadline(run_dir, admission)
        _finalize_success(run_dir, args, admission)
        return
    admission = _prepaid_report_admission(active_seconds=float(ledger["active_seconds"]), now_monotonic=time.perf_counter(), active_seconds_cap=float(ledger["caps"]["active_seconds"]))
    if admission["admitted"] != 1: raise ResourceBoundaryError("report reserve cannot be prepaid")
    journal = _begin_durable_attempt(run_dir, role="report_verify", attempt=_resource_event_count(run_dir, "report_verify") + 1, durable_elapsed_seconds=0.0)
    _finish_durable_attempt(run_dir, journal, durable_elapsed_seconds=0.0, invocation_wall_seconds=REPORT_RESERVE_SECONDS, detail={"prepaid_report_admission": admission})
    _finalize_success(run_dir, args, admission)
def _run_requested_stages(run_dir: Path, args: argparse.Namespace) -> None:
    _stage_prefix_read_only(Path(run_dir))
    requested = STAGES if args.stage == "all" else (args.stage,)
    for name in requested:
        marker = Path(run_dir) / f"stages/{name}.json"
        if marker.is_file():
            _completed_stage_artifacts_read_only(Path(run_dir), name)
            continue
        _require_predecessors(Path(run_dir), name)
        try:
            globals()[f"_run_{name}"](Path(run_dir), args)
        except BaseException as exc:
            try: setattr(exc, "_continuation_stage", name)
            except Exception: pass
            raise
        _read_stage_marker_exact(Path(run_dir), name)
        _stage_prefix_read_only(Path(run_dir))
def _v2_operational_paths() -> tuple[str, ...]:
    paths = [
        "inputs/model/update-3100.pt",
        "inputs/calibration/on_policy_validation_calibration.npz",
        "inputs/source/source_image.json", "inputs/source/source_image.npz",
        "forward/anchor-step-0127.npz", "forward/anchor-step-0511.npz",
    ]
    paths.extend(
        f"forward/forward_shards/fresh-main-path/shard-{index:04d}.{suffix}"
        for index in range(FORWARD_SHARDS) for suffix in ("json", "npz")
    )
    paths.extend(
        f"reverse/fused_families/same-path-three-row/complete-512/shard-0000.{suffix}"
        for suffix in ("json", "npz")
    )
    return tuple(paths)
def _verify_v2_prefix_bundle_read_only(
    prefix_run_dir: Path, *, parent_run_dir: Path, source_run_dir: Path
) -> dict[str, Any]:
    """Authenticate the one frozen v2 resource-stop bundle without current-policy drift."""

    prefix = Path(prefix_run_dir).resolve(); parent = Path(parent_run_dir).resolve(); source_root = Path(source_run_dir).resolve()
    roots = (prefix, parent, source_root)
    for root in roots: _reject_links_and_reparse_points(root)
    if any(_paths_overlap(left, right) for index, left in enumerate(roots) for right in roots[index + 1 :]):
        raise ParentBindingError("v2 prefix, v3 parent, and source roots overlap")
    before = tuple(_snapshot_tree(root) for root in roots); prefix_rows = before[0]
    if len(prefix_rows) != V2_FILE_COUNT or _directory_bytes(prefix) != V2_EXACT_BYTES:
        raise ParentBindingError("v2 physical file/byte authority changed")
    identities: set[tuple[int, int]] = set()
    for relative, _, _ in prefix_rows:
        path = prefix / relative; metadata = path.stat(); identity = (int(metadata.st_dev), int(metadata.st_ino))
        if ".tmp" in path.name or metadata.st_nlink != 1 or identity in identities:
            raise ParentBindingError("v2 tree contains temporary or same-inode evidence")
        identities.add(identity)
    values: dict[str, dict[str, Any]] = {}
    for relative, (file_hash, semantic_hash) in _V2_EXPECTED_HASHES.items():
        path = prefix / relative
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != file_hash:
            raise ParentBindingError(f"v2 pinned file changed: {relative}")
        if relative.endswith(".json"):
            value = _read_json(path, semantic=True)
            if value.get("semantic_sha256") != semantic_hash:
                raise ParentBindingError(f"v2 pinned semantic bytes changed: {relative}")
            values[relative] = value
    inventory = _verify_terminal_inventory_read_only(prefix, "failure")
    manifest = inventory["manifest"]
    if inventory["artifact_count"] != V2_MANIFEST_ROWS or inventory["checksum_count"] != V2_CHECKSUM_ENTRIES:
        raise ParentBindingError("v2 manifest/checksum count changed")
    manifest_rows = manifest.get("artifacts")
    if not isinstance(manifest_rows, list): raise ParentBindingError("v2 manifest rows are absent")
    by_path = {str(row.get("path")): dict(row) for row in manifest_rows if isinstance(row, Mapping)}
    if len(by_path) != V2_MANIFEST_ROWS: raise ParentBindingError("v2 manifest paths are not unique")
    run_manifest = values["run_manifest.json"]; config = values["scientific_config.json"]
    parent_binding = values["parent_binding.json"]; terminal = values["terminal_failure.json"]
    verification = values["verification.json"]; storage = values["terminal_storage_authority.json"]
    if (
        run_manifest.get("schema") != RUN_SCHEMA
        or run_manifest.get("run_dir") != str(prefix)
        or run_manifest.get("parent_run_dir") != str(parent)
        or run_manifest.get("source_run_dir") != str(source_root)
        or config.get("runner") != V2_VERSION
        or config.get("imported_forward_shards") != PARENT_FORWARD_PREFIX_SHARDS
        or config.get("generated_forward_shards") != FORWARD_SHARDS - PARENT_FORWARD_PREFIX_SHARDS
        or config.get("resource_caps", {}).get("active_seconds") != V2_ACTIVE_SECONDS_CAP
        or parent_binding.get("supplied_parent_path") != str(parent)
        or parent_binding.get("supplied_source_path") != str(source_root)
        or parent_binding.get("parent_tree_sha256") != _tree_hash(before[1])
        or parent_binding.get("source_tree_sha256") != _tree_hash(before[2])
        or terminal.get("failure_domain") != "resource_boundary"
        or terminal.get("message") != "reverse shard 1 cannot preserve postprocess/report reserve"
        or terminal.get("resume_same_child_authorized") != 0
        or terminal.get("scientific_objective_completed") != 0
        or verification.get("passed") != 1
        or verification.get("terminal_kind") != "failure"
        or verification.get("terminal_failure_semantic_sha256") != terminal.get("semantic_sha256")
        or verification.get("terminal_storage_semantic_sha256") != storage.get("semantic_sha256")
        or storage.get("exact_recursive_file_bytes") != V2_EXACT_BYTES
        or storage.get("terminal_kind") != "failure"
    ):
        raise ParentBindingError("v2 terminal/scientific binding changed")
    v2_caps = _resource_caps(active_seconds=V2_ACTIVE_SECONDS_CAP)
    ledger = _validate_resource_ledger(prefix, expected_version=V2_VERSION, expected_caps=v2_caps)
    if (
        ledger.get("active_seconds") != V2_CARRIED_ACTIVE_SECONDS
        or ledger.get("peak_cuda_bytes") != V2_PEAK_CUDA_BYTES
        or ledger.get("total_cuda_bytes") != V2_TOTAL_CUDA_BYTES
        or ledger.get("limits_passed") != 1 or ledger.get("breached_limits") != []
        or len(ledger.get("events", [])) != 4
    ):
        raise ParentBindingError("v2 resource ledger authority changed")
    source = load_verified_source_target(prefix / "inputs/source")
    if (source.source_json_sha256, source.source_npz_sha256, rollout_array_sha256(source.source_image), rollout_array_sha256(source.mixed_target)) != (SOURCE_JSON_SHA256, SOURCE_NPZ_SHA256, SOURCE_IMAGE_ARRAY_SHA256, MIXED_TARGET_ARRAY_SHA256):
        raise ParentBindingError("v2 source pair changed")
    forward = _scan_forward_chain(prefix)
    if forward["first_missing"] != FORWARD_SHARDS or forward["orphan"] is not None or forward["failure"] is not None or forward["records"][0].get("input_state_sha256") != rollout_array_sha256(np.ascontiguousarray(source.mixed_target[None, :], dtype=np.float64)):
        raise ParentBindingError("v2 forward chain boundary changed")
    forward_health = aggregate_exact_forward_shards(forward["records"], expected_shard_count=FORWARD_SHARDS, expected_transition_count=FORWARD_TRANSITION_COUNT, expected_path_ids=(PATH_ID,))
    anchor = _load_anchor_511(prefix); final_forward = _load_npz(prefix / "forward/forward_shards/fresh-main-path/shard-0063.npz")["state"][0]
    if (
        forward_health.get("passed") != 1 or forward_health.get("authorization_fraction") != 1.0
        or forward_health.get("certificate_fraction") != 1.0 or forward_health.get("forbidden_event_count") != 0
        or max(float(forward_health.get(key, math.inf)) for key in ("maximum_global_mass_error", "maximum_pair_mass_error", "maximum_simplex_mass_error", "maximum_output_state_mass_error")) > 2e-12
        or _sha256_file(prefix / "forward/anchor-step-0511.npz") != STEP511_ARCHIVE_SHA256
        or rollout_array_sha256(anchor) != STEP511_STATE_SHA256 or not np.array_equal(anchor, final_forward)
    ):
        raise ParentBindingError("v2 complete forward health changed")
    reverse = _scan_reverse_chain(prefix); failure_path = prefix / "reverse/fused_families/same-path-three-row/complete-512/shard-0001.failure.json"
    if reverse["first_missing"] != IMPORTED_REVERSE_SHARDS or len(reverse["records"]) != IMPORTED_REVERSE_SHARDS or reverse["orphan"] is not None or reverse["failure"] != failure_path:
        raise ParentBindingError("v2 reverse prefix/gap changed")
    shard_state = _load_npz(prefix / "reverse/fused_families/same-path-three-row/complete-512/shard-0000.npz")["state"]
    reverse_health = _strict_fused_exact_health(final_state=shard_state, shard_records=reverse["records"], expected_shard_count=IMPORTED_REVERSE_SHARDS)
    after = tuple(_snapshot_tree(root) for root in roots)
    if after != before: raise ParentBindingError("v2 prefix audit mutated an external tree")
    return {
        "binding_passed": 1, "prefix_file_count": len(prefix_rows), "prefix_bytes": V2_EXACT_BYTES,
        "prefix_tree_sha256": _tree_hash(prefix_rows), "prefix_tree_rows": [{"path": path, "size": size, "sha256": digest} for path, size, digest in prefix_rows],
        "manifest_artifact_count": V2_MANIFEST_ROWS, "checksum_entry_count": V2_CHECKSUM_ENTRIES,
        "manifest_rows_by_path": by_path, "values": values, "resource_ledger": ledger,
        "forward_health": forward_health, "reverse_shard_0_health": reverse_health,
        "reverse_shard_0_state_sha256": rollout_array_sha256(shard_state),
        "parent_tree_sha256": _tree_hash(before[1]), "source_tree_sha256": _tree_hash(before[2]),
        "v2_parent_run_dir": str(parent), "v2_source_run_dir": str(source_root),
    }
def _verify_parent_manifest_and_checksums(parent_run_dir: Path) -> dict[str, Any]:
    parent_run_dir = Path(parent_run_dir)
    manifest_path = parent_run_dir / "artifact_manifest.json"
    checksum_path = parent_run_dir / "SHA256SUMS.txt"
    if (
        _sha256_file(manifest_path) != PARENT_MANIFEST_FILE_SHA256
        or _sha256_file(checksum_path) != PARENT_CHECKSUM_FILE_SHA256
    ):
        raise ParentBindingError("parent manifest/checksum file authority changed")
    manifest = _read_json(manifest_path, semantic=True)
    artifacts = manifest.get("artifacts")
    exclusions = manifest.get("excluded_self_referential_paths")
    if (
        manifest.get("semantic_sha256") != PARENT_MANIFEST_SEMANTIC_SHA256
        or not isinstance(artifacts, list)
        or len(artifacts) != 207
        or manifest.get("artifact_count") != 207
        or sorted(exclusions or []) != sorted(_TERMINAL_EXCLUSIONS)
    ):
        raise ParentBindingError("parent artifact manifest schema/count changed")
    seen: set[str] = set()
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise ParentBindingError("parent manifest row is malformed")
        relative = str(row.get("path", ""))
        path = parent_run_dir / relative
        if relative in seen or relative in _TERMINAL_EXCLUSIONS or not path.is_file():
            raise ParentBindingError("parent manifest path set changed")
        seen.add(relative)
        if path.stat().st_size != int(row.get("size", -1)) or _sha256_file(path) != row.get("sha256"):
            raise ParentBindingError(f"parent manifest artifact changed: {relative}")
    checksum_rows: list[tuple[str, str]] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ParentBindingError("parent checksum format changed")
        checksum_rows.append((parts[1], parts[0]))
    expected_paths = sorted((*seen, "artifact_manifest.json"))
    if (
        len(checksum_rows) != 208
        or sorted(path for path, _ in checksum_rows) != expected_paths
        or len({path for path, _ in checksum_rows}) != 208
    ):
        raise ParentBindingError("parent checksum path/count authority changed")
    for relative, expected in checksum_rows:
        if _sha256_file(parent_run_dir / relative) != expected:
            raise ParentBindingError(f"parent checksum artifact changed: {relative}")
    return {
        "manifest": manifest,
        "artifacts": artifacts,
        "artifact_paths": seen,
        "checksums": checksum_rows,
    }
def _verify_parent_bundle_read_only(parent_run_dir: Path, *, source_run_dir: Path) -> dict[str, Any]:
    parent_run_dir = Path(parent_run_dir).resolve()
    source_run_dir = Path(source_run_dir).resolve()
    _reject_links_and_reparse_points(parent_run_dir)
    _reject_links_and_reparse_points(source_run_dir)
    parent_before = _snapshot_tree(parent_run_dir)
    source_before = _snapshot_tree(source_run_dir)
    if len(parent_before) != 212 or _directory_bytes(parent_run_dir) != 12_520_738 or _tree_hash(parent_before) != PARENT_TREE_SHA256:
        raise ParentBindingError("parent physical file/byte authority changed")

    values: dict[str, dict[str, Any]] = {}
    for relative, (file_hash, semantic_hash) in _PARENT_EXPECTED_HASHES.items():
        path = parent_run_dir / relative
        if not path.is_file() or path.is_symlink():
            raise ParentBindingError(f"parent terminal authority is absent: {relative}")
        if file_hash is not None and _sha256_file(path) != file_hash:
            raise ParentBindingError(f"parent terminal file hash changed: {relative}")
        if relative.endswith(".json"):
            value = _read_json(path, semantic=True)
            if semantic_hash is not None and value.get("semantic_sha256") != semantic_hash:
                raise ParentBindingError(f"parent terminal semantic hash changed: {relative}")
            values[relative] = value

    inventory = _verify_parent_manifest_and_checksums(parent_run_dir)
    artifacts = inventory["artifacts"]
    checksum_rows = inventory["checksums"]

    run_manifest = values["run_manifest.json"]
    closure = run_manifest.get("source_closure")
    if (
        run_manifest.get("source_closure_sha256") != PARENT_SOURCE_CLOSURE_SHA256
        or not isinstance(closure, Mapping)
        or len(closure) != 41
    ):
        raise ParentBindingError("parent source closure authority changed")
    repository_root = Path(__file__).resolve().parents[1]
    producer_relative = "mnist/diag_d0_jacobi_rb_global_dilated_rollout.py"
    producer_audit = _audit_frozen26_producer_read_only(repository_root / producer_relative)
    for relative, authority in closure.items():
        if relative == producer_relative:
            if authority.get("size") != FROZEN26_SIZE or authority.get("sha256") != FROZEN26_SHA256:
                raise ParentBindingError("parent producer closure row changed")
            continue
        path = repository_root / str(relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(authority.get("size", -1))
            or _sha256_file(path) != authority.get("sha256")
        ):
            raise ParentBindingError(f"parent source dependency changed: {relative}")

    for stage in ("prepare", "controls", "train_select_freeze", "evaluate_exact", "report_verify"):
        marker = _read_json(parent_run_dir / f"stages/{stage}.json", semantic=True)
        if marker.get("stage") != stage or marker.get("passed") != 1:
            raise ParentBindingError(f"parent stage marker changed: {stage}")
    verification = values["verification.json"]
    storage = values["terminal_storage_authority.json"]
    if (
        verification.get("passed") != 1
        or verification.get("resource_limits_passed") != 1
        or verification.get("terminal_storage_authority_semantic_sha256") != PARENT_STORAGE_SEMANTIC_SHA256
        or storage.get("exact_recursive_file_bytes") != 12_520_738
        or storage.get("resource_limits_passed") != 1
    ):
        raise ParentBindingError("parent terminal verification/storage changed")

    selection = _read_json(parent_run_dir / "selection.json", semantic=True)
    freeze = _read_json(parent_run_dir / "evaluation_freeze.json", semantic=True)
    path_usage = _read_json(parent_run_dir / "path_usage.json", semantic=True)
    bindings = _read_json(parent_run_dir / "input_bindings.json", semantic=True)
    outcome = values["outcome.json"]
    branch = values["positive_branch.json"]
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ParentBindingError("parent selected checkpoint is absent")
    checkpoint_path = parent_run_dir / str(selected.get("checkpoint_path"))
    if (
        selected.get("update") != PARENT_SELECTED_UPDATE
        or selected.get("checkpoint_file_sha256") != CHECKPOINT_FILE_SHA256
        or selected.get("state_sha256") != CHECKPOINT_STATE_SHA256
        or freeze.get("global_checkpoint") != selected
        or not checkpoint_path.is_file()
        or _sha256_file(checkpoint_path) != CHECKPOINT_FILE_SHA256
    ):
        raise ParentBindingError("parent selected checkpoint authority changed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if (
        not isinstance(state, Mapping)
        or payload.get("seed") != 261_372
        or payload.get("update") != PARENT_SELECTED_UPDATE
        or payload.get("state_sha256") != CHECKPOINT_STATE_SHA256
        or state_dict_sha256(state) != CHECKPOINT_STATE_SHA256
        or payload.get("training_fingerprint") != freeze.get("global_training_fingerprint")
    ):
        raise ParentBindingError("parent checkpoint payload changed")

    source = load_verified_source_target(source_run_dir)
    source_record = bindings.get("source_target")
    if (
        not isinstance(source_record, Mapping)
        or source.source_json_sha256 != SOURCE_JSON_SHA256
        or source.source_npz_sha256 != SOURCE_NPZ_SHA256
        or rollout_array_sha256(source.source_image) != SOURCE_IMAGE_ARRAY_SHA256
        or rollout_array_sha256(source.mixed_target) != MIXED_TARGET_ARRAY_SHA256
        or source_record.get("source_json_sha256") != SOURCE_JSON_SHA256
        or source_record.get("source_npz_sha256") != SOURCE_NPZ_SHA256
        or source.metadata.get("physical_training_performed") != 0
        or source.metadata.get("sampling_performed") != 0
        or source.metadata.get("reverse_sampling_performed") != 0
    ):
        raise ParentBindingError("source pair or parent source binding changed")
    anchor = parent_run_dir / "fresh_forward/anchor-step-0127.npz"
    anchor_arrays = _load_npz(anchor)
    if (
        _sha256_file(anchor) != STEP127_ARCHIVE_SHA256
        or set(anchor_arrays) != {"state"}
        or rollout_array_sha256(anchor_arrays["state"]) != STEP127_STATE_SHA256
    ):
        raise ParentBindingError("parent step-127 anchor changed")
    profile = _default_profile()
    prefix_rows: list[dict[str, Any]] = []
    prefix_root = parent_run_dir / "fresh_forward/forward_shards/fresh-main-path"
    for index in range(PARENT_FORWARD_PREFIX_SHARDS):
        record_path = prefix_root / f"shard-{index:04d}.json"
        archive_path = prefix_root / f"shard-{index:04d}.npz"
        record = _read_json(record_path, semantic=True)
        if (
            record.get("committed") != 1
            or record.get("trajectory_name") != "fresh-main-path"
            or record.get("path_ids") != [PATH_ID]
            or record.get("root_seed") != FORWARD_ROOT_SEED
            or record.get("profile_sha256") != DEFAULT_PROFILE_SHA256
            or record.get("state_file_sha256") != _sha256_file(archive_path)
        ):
            raise ParentBindingError(f"parent prefix shard changed: {index}")
        prefix_rows.append(record)
    aggregate = aggregate_exact_forward_shards(
        prefix_rows,
        expected_shard_count=PARENT_FORWARD_PREFIX_SHARDS,
        expected_transition_count=PARENT_PREFIX_TRANSITION_COUNT,
        expected_path_ids=(PATH_ID,),
    )
    if aggregate.get("passed") != 1 or aggregate.get("certificate_fraction") != 1.0:
        raise ParentBindingError("parent forward prefix health changed")
    if (
        path_usage.get("fresh_path_id") != PATH_ID
        or outcome.get("outcome") != "global_material_improvement"
        or outcome.get("primary_objectives", {}).get("global-plus-1", {}).get(
            "relative_paired_squared_l2_improvement_over_zero"
        ) != 0.0745378989614584
        or branch.get("triggered") != 1
        or branch.get("attempted") != 0
        or branch.get("completed", 0) != 0
        or branch.get("projected_forward_tail_seconds") != 2700.0
        or branch.get("projected_reverse_seconds") != 14298.701474303974
        or bindings.get("confirmation_namespace_opened") != 0
        or bindings.get("protected_confirmation_evidence_used") != 0
    ):
        raise ParentBindingError("parent scientific continuation trigger changed")

    parent_after = _snapshot_tree(parent_run_dir)
    source_after = _snapshot_tree(source_run_dir)
    if parent_after != parent_before or source_after != source_before:
        raise ParentBindingError("parent/source audit mutated evidence")
    return {
        "binding_passed": 1,
        "parent_file_count": len(parent_before),
        "parent_bytes": _directory_bytes(parent_run_dir),
        "parent_tree_sha256": _tree_hash(parent_before),
        "source_tree_sha256": _tree_hash(source_before),
        "manifest_artifact_count": len(artifacts),
        "checksum_entry_count": len(checksum_rows),
        "run_manifest_semantic_sha256": run_manifest["semantic_sha256"],
        "source_closure_sha256": run_manifest["source_closure_sha256"],
        "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
        "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
        "source_json_sha256": SOURCE_JSON_SHA256,
        "source_npz_sha256": SOURCE_NPZ_SHA256,
        "frozen26_audit": producer_audit,
        "selected_update": PARENT_SELECTED_UPDATE,
        "path_id": PATH_ID,
        "confirmation_opened": 0,
        "profile": profile.to_dict(),
        "profile_sha256": DEFAULT_PROFILE_SHA256,
        "prefix_health": aggregate,
        "parent_tree_rows": [
            {"path": path, "size": size, "sha256": digest}
            for path, size, digest in parent_before
        ],
        "source_tree_rows": [
            {"path": path, "size": size, "sha256": digest}
            for path, size, digest in source_before
        ],
    }
def main(argv: Sequence[str] | None = None) -> int:
    global _UNDEBITED_START
    args = parse_args(argv)
    mode = _resolve_mode(args)
    paths = _resolve_paths(args, mode=mode)
    resume_error: BaseException | None = None
    if mode == "verify":
        _verify_parent_bundle_read_only(paths.parent_run_dir, source_run_dir=paths.source_run_dir)
        _verify_v2_prefix_bundle_read_only(paths.prefix_run_dir, parent_run_dir=paths.parent_run_dir, source_run_dir=paths.source_run_dir)
        _verify_terminal_child_read_only(
            paths.run_dir,
            prefix_run_dir=paths.prefix_run_dir,
            parent_run_dir=paths.parent_run_dir,
            source_run_dir=paths.source_run_dir,
        )
        return 0
    if mode == "fresh":
        run_dir = _initialize_child_atomically(args, paths=paths)
        _UNDEBITED_START = time.perf_counter()
    else:
        run_dir = paths.run_dir
        _UNDEBITED_START = time.perf_counter()
        resume_started_at = _utc_now()
        identity = _load_child_identity_read_only(run_dir)
        _verify_resume_compatibility_read_only(run_dir, identity=identity, prefix_run_dir=paths.prefix_run_dir, parent_run_dir=paths.parent_run_dir, source_run_dir=paths.source_run_dir)
        probe_path, probe = _begin_resume_probe(run_dir, identity, paths.prefix_run_dir, paths.parent_run_dir, paths.source_run_dir, started_at=resume_started_at, started_monotonic=_UNDEBITED_START)
        binding = _resume_probe_binding(run_dir, identity, paths.prefix_run_dir, paths.parent_run_dir, paths.source_run_dir)
        try:
            if _resume_probe_is_covered(run_dir, probe): probe = _restart_resume_probe(probe_path, binding)
            repeated_identity = _load_child_identity_read_only(run_dir)
            if _canonical_json_bytes(repeated_identity) != _canonical_json_bytes(identity): raise ContinuationIntegrityError("resume identity changed after ownership")
            _verify_resume_compatibility_read_only(run_dir, identity=repeated_identity, prefix_run_dir=paths.prefix_run_dir, parent_run_dir=paths.parent_run_dir, source_run_dir=paths.source_run_dir)
        except (KeyboardInterrupt, SystemExit): raise
        except BaseException as exc: resume_error = exc
        try:
            prior_event_ids = {event["event_id"] for event in _validate_resource_ledger(run_dir)["events"]}
            recovered = _reconcile_live_stage_journals(run_dir)
            if any(event.get("event_id") not in prior_event_ids and str(event.get("role", "")).endswith("_abandoned_attempt") for event in recovered): probe = _restart_resume_probe(probe_path, binding)
        except (KeyboardInterrupt, SystemExit): raise
        except BaseException as exc:
            if resume_error is None: resume_error = exc
        try:
            journal = _begin_durable_attempt(run_dir, role="resume_validation", attempt=_resource_event_count(run_dir, "resume_validation") + 1, durable_elapsed_seconds=0.0, now_utc=str(probe["started_at"]), now_monotonic=float(probe["started_monotonic"]))
            _finish_durable_attempt(run_dir, journal, durable_elapsed_seconds=0.0, invocation_wall_seconds=_resume_probe_elapsed(probe) + 5.0, detail={"compatible_child_verified": int(resume_error is None)})
            probe_path.unlink(missing_ok=True)
        except (KeyboardInterrupt, SystemExit): raise
        except BaseException as exc:
            if resume_error is None: resume_error = exc
    capture_path = run_dir / "failure_capture.json"
    if capture_path.is_file():
        capture = _read_json(capture_path, semantic=True)
        try:
            _reconcile_live_stage_journals(run_dir)
        except ContinuationError:
            pass
        _finalize_failure_package(run_dir, args, capture)
        if mode == "resume": probe_path.unlink(missing_ok=True)
        return 1
    try:
        if resume_error is not None: raise resume_error
        if mode == "resume": _reconcile_live_stage_journals(run_dir)
        _run_requested_stages(run_dir, args)
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        stage = str(getattr(exc, "_continuation_stage", "unknown"))
        capture = _capture_failure(run_dir, stage, exc)
        try:
            _reconcile_live_stage_journals(run_dir)
        except ContinuationError:
            pass
        _finalize_failure_package(run_dir, args, capture)
        if mode == "resume": probe_path.unlink(missing_ok=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
