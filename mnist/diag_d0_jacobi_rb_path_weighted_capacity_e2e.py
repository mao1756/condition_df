"""One-command path-weighted Jacobi/RB capacity experiment.

This runner performs the complete requested comparison on one MNIST image:

* existing 34,974-parameter global-dilated controller, new path-weighted loss;
* 2,390,174-parameter global residual controller, the same new loss;
* the existing controller with its historical unweighted loss as an internal
  attribution control;
* paired null and source-informed oracle rollouts.

Both learned models are evaluated from (a) an exact forward terminal endpoint
and (b) an independent Dirichlet(1) start.  All controller rows within a start
mode reuse the same candidate transition IDs and RNG root, so their reference
noise is paired.  The runner is resumable at cache/training stages and writes a
read-only verification manifest at completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist import eulerian_jacobi_ddpm as core
from mnist.d0_jacobi_rb_candidate_training_cache import (
    CandidatePrefixCacheSpec,
    build_candidate_prefix_cache,
)
from mnist.d0_jacobi_rb_cuda import sample_alpha1_rb_transition_batch_cuda
from mnist.d0_jacobi_rb_cuda_deferred import (
    CandidateRBCudaBatch,
    enqueue_alpha1_rb_transition_batch_cuda_candidate,
)
from mnist.d0_jacobi_rb_global_large import large_global_architecture_contract
from mnist.d0_jacobi_rb_path_weighted_loss import PathWeightedLossConfig
from mnist.d0_jacobi_rb_path_weighted_training import (
    CapacityTrainingConfig,
    compute_cache_loss_scales,
    load_selected_model,
    run_large_memorization_gate,
    train_capacity_model,
)
from mnist.eulerian_jacobi_ddpm_candidate import (
    CANDIDATE_BACKEND_NAME,
    CANDIDATE_TARGET_SEMANTICS,
    CandidateRuntime,
    forward_terminal_states_candidate,
    prepare_candidate_runtime,
    reverse_sample_candidate,
)


VERSION = "d0-jacobi-rb-path-weighted-capacity-e2e-v1"
DEFAULT_ANCHORS = (0, 8, 16, 128, 256, 384, 512)
CONTROLLER_ROWS = (
    "zero",
    "small-old",
    "small-weighted",
    "large-weighted",
    "oracle",
)


class ExperimentError(RuntimeError):
    """The unattended experiment failed its declared contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(event: str, **payload: Any) -> None:
    print(
        json.dumps({"at": _utc_now(), "event": event, **payload}, sort_keys=True),
        flush=True,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stage_path(run_dir: Path, name: str) -> Path:
    return run_dir / "stages" / f"{name}.json"


def _stage_complete(run_dir: Path, name: str) -> bool:
    path = _stage_path(run_dir, name)
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "complete"


def _complete_stage(run_dir: Path, name: str, **payload: Any) -> None:
    _atomic_json(
        _stage_path(run_dir, name),
        {
            "schema": VERSION + "-stage",
            "name": name,
            "status": "complete",
            "completed_at": _utc_now(),
            **payload,
        },
    )
    _log("stage_complete", stage=name, **payload)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _environment_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": VERSION + "-environment",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": int(torch.cuda.is_available()),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update(
            cuda_device_name=properties.name,
            cuda_total_memory=int(properties.total_memory),
            cuda_capability=[properties.major, properties.minor],
            cuda_runtime=torch.version.cuda,
        )
    return record


def _load_mnist_source(
    data_dir: Path, *, index: int, download: bool
) -> tuple[np.ndarray, int, dict[str, Any]]:
    try:
        from torchvision.datasets import MNIST
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ExperimentError(
            "torchvision is required; run tools/runpod_weighted_e2e/install_environment.sh"
        ) from exc
    dataset = MNIST(root=str(data_dir), train=True, download=bool(download))
    source_index = int(index)
    if not 0 <= source_index < len(dataset):
        raise ExperimentError("MNIST index is outside the training split")
    image, label = dataset[source_index]
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.shape != (28, 28):
        raise ExperimentError("MNIST source image has the wrong shape")
    values = pixels.reshape(-1).astype(np.float64)
    total = float(values.sum(dtype=np.float64))
    if total <= 0.0:
        raise ExperimentError("selected MNIST image has zero intensity")
    clean = np.ascontiguousarray(values / total, dtype=np.float64)
    return clean, int(label), {
        "dataset": "torchvision.datasets.MNIST",
        "split": "train",
        "index": source_index,
        "label": int(label),
        "pixel_sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
        "unit_mass_sha256": hashlib.sha256(
            clean.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
    }


def _save_source_images(
    directory: Path, clean: np.ndarray, mixed: np.ndarray
) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ExperimentError("Pillow is required for image artifacts") from exc
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(core.rasterize_unit_masses(clean)).save(directory / "clean.png")
    Image.fromarray(core.rasterize_unit_masses(mixed)).save(directory / "mixed.png")
    _save_npz(directory / "states.npz", clean=clean, mixed=mixed)


@dataclass(frozen=True)
class ExperimentConfig:
    run_dir: str
    device: str
    data_dir: str
    mnist_index: int
    download_mnist: bool
    train_paths: int
    validation_paths: int
    small_updates: int
    large_updates: int
    batch_size: int
    validation_interval: int
    mobility_floor: float
    include_old_control: bool
    hard_wall_seconds: int
    seed: int

    def __post_init__(self) -> None:
        if (
            int(self.train_paths) <= 0
            or int(self.validation_paths) <= 0
            or int(self.small_updates) <= 0
            or int(self.large_updates) <= 0
            or int(self.batch_size) <= 0
            or int(self.validation_interval) <= 0
            or int(self.hard_wall_seconds) <= 0
        ):
            raise ExperimentError("experiment counts and budgets must be positive")
        PathWeightedLossConfig(mobility_floor=float(self.mobility_floor))

    def to_record(self) -> dict[str, Any]:
        return {"schema": VERSION + "-config", **asdict(self)}


SEED_OFFSETS = {
    "audit": 0x001,
    "train_forward": 0x101,
    "validation_forward": 0x102,
    "evaluation_forward": 0x103,
    "same_path_reverse": 0x201,
    "prior_reverse": 0x202,
    "prior_start": 0x203,
    "small_training": 0x301,
    "large_training": 0x302,
    "memorization": 0x303,
}


def _seed(config: ExperimentConfig, name: str) -> int:
    return int(config.seed) + int(SEED_OFFSETS[name])


def _candidate_rng_keys(config: ExperimentConfig) -> tuple[tuple[Any, ...], ...]:
    keys: list[tuple[Any, ...]] = [(_seed(config, "audit"), "candidate-audit")]
    for name in ("train_forward", "validation_forward", "evaluation_forward"):
        keys.append((_seed(config, name), "forward"))
    for name in ("same_path_reverse", "prior_reverse"):
        root = _seed(config, name)
        keys.extend(
            (root, "reverse", micro, side)
            for micro in range(core.CONTROLLER_MICROSTEPS)
            for side in ("pre", "post")
        )
    return tuple(keys)


def _run_candidate_audit(
    run_dir: Path,
    runtime: CandidateRuntime,
    *,
    seed: int,
    lanes: int = 64,
) -> dict[str, Any]:
    """Compare the production candidate batch with its certified reference."""

    count = int(lanes)
    if count <= 0:
        raise ExperimentError("candidate audit lane count must be positive")
    x = np.linspace(0.02, 0.98, count, dtype=np.float64)
    exposure = np.geomspace(0.04, 1.0, count, dtype=np.float64)
    transition_ids_np = (
        np.arange(count, dtype=np.uint64) + np.uint64((1 << 19) << 23)
    )
    device = runtime.device
    x_tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    exposure_tensor = torch.as_tensor(exposure, dtype=torch.float64, device=device)
    ids_tensor = torch.as_tensor(transition_ids_np, dtype=torch.uint64, device=device)
    key = (int(seed), "candidate-audit")
    candidate = enqueue_alpha1_rb_transition_batch_cuda_candidate(
        x_tensor,
        exposure_tensor,
        rng_key=key,
        transition_ids=ids_tensor,
        prepared=runtime.prepared,
        prepared_rng_seed=runtime.prepared_seeds[key],
    )
    if not isinstance(candidate, CandidateRBCudaBatch):
        raise ExperimentError("candidate audit dispatched the wrong backend")
    certified = sample_alpha1_rb_transition_batch_cuda(
        x_tensor,
        exposure_tensor,
        rng_key=key,
        transition_ids=ids_tensor,
        profile=runtime.profile,
    )
    torch.cuda.synchronize(device)
    candidate_later = candidate.later_head_fraction.detach().cpu().numpy()
    candidate_target = candidate.denoising_target.detach().cpu().numpy()
    certified_candidate_later = (
        certified.candidate_later_head_fraction.detach().cpu().numpy()
    )
    certified_candidate_target = (
        certified.candidate_denoising_target.detach().cpu().numpy()
    )
    certified_later = certified.later_head_fraction.detach().cpu().numpy()
    certified_target = certified.denoising_target.detach().cpu().numpy()
    candidate_identity = bool(
        np.array_equal(candidate_later, certified_candidate_later)
        and np.array_equal(candidate_target, certified_candidate_target)
    )
    later_error = float(np.max(np.abs(candidate_later - certified_later)))
    target_error = float(np.max(np.abs(candidate_target - certified_target)))
    report = {
        "schema": VERSION + "-candidate-audit",
        "lanes": count,
        "candidate_backend": CANDIDATE_BACKEND_NAME,
        "target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "candidate_binary_sha256": runtime.candidate_binary_sha256,
        "candidate_matches_certified_candidate_bytes": int(candidate_identity),
        "maximum_later_fraction_error": later_error,
        "maximum_target_error": target_error,
        "later_error_threshold": 2.0e-10,
        "target_error_threshold": 2.0e-8,
        "certified_lane_count": int(
            certified.certified_mask.detach().cpu().sum().item()
        ),
        "fallback_lane_count": int(
            certified.fallback_mask.detach().cpu().sum().item()
        ),
        "passed": int(
            candidate_identity
            and later_error <= 2.0e-10
            and target_error <= 2.0e-8
            and np.isfinite(candidate_later).all()
            and np.isfinite(candidate_target).all()
        ),
    }
    _save_npz(
        run_dir / "candidate_audit" / "outputs.npz",
        earlier_fraction=x,
        exposure=exposure,
        transition_ids=transition_ids_np,
        candidate_later=candidate_later,
        candidate_target=candidate_target,
        certified_later=certified_later,
        certified_target=certified_target,
    )
    _atomic_json(run_dir / "candidate_audit" / "report.json", report)
    if not report["passed"]:
        raise ExperimentError("candidate-versus-certified audit failed")
    return report


def _controller_metric(state: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    generated = np.asarray(state, dtype=np.float64).reshape(-1)
    reference = np.asarray(target, dtype=np.float64).reshape(-1)
    if generated.shape != (core.STATE_SIZE,) or reference.shape != generated.shape:
        raise ExperimentError("metric states must be aligned [784]")
    difference = generated - reference
    centered_generated = generated - float(generated.mean())
    centered_reference = reference - float(reference.mean())
    centered_denominator = float(
        np.linalg.norm(centered_generated) * np.linalg.norm(centered_reference)
    )
    cosine_denominator = float(np.linalg.norm(generated) * np.linalg.norm(reference))
    image = generated.reshape(28, 28)
    horizontal = np.roll(image, -1, axis=1) - image
    vertical = np.roll(image, -1, axis=0) - image
    return {
        "squared_l2": float(np.sum(difference**2, dtype=np.float64)),
        "l1": float(np.sum(np.abs(difference), dtype=np.float64)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "centered_correlation": (
            float(np.dot(centered_generated, centered_reference) / centered_denominator)
            if centered_denominator > 0.0
            else math.nan
        ),
        "cosine_similarity": (
            float(np.dot(generated, reference) / cosine_denominator)
            if cosine_denominator > 0.0
            else math.nan
        ),
        "periodic_total_variation": float(
            np.sum(np.abs(horizontal), dtype=np.float64)
            + np.sum(np.abs(vertical), dtype=np.float64)
        ),
        "mass_error": abs(float(generated.sum(dtype=np.float64)) - 1.0),
        "minimum_entry": float(np.min(generated)),
        "maximum_entry": float(np.max(generated)),
        "finite": int(np.isfinite(generated).all()),
        "nonnegative": int(np.all(generated >= 0.0)),
    }


def _render_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = core.rasterize_unit_masses(state)
    demixed = core.rasterize_unit_masses(core.demix_unit_masses(state))
    return np.asarray(raw, dtype=np.uint8), np.asarray(demixed, dtype=np.uint8)


def _save_sampling_images(
    directory: Path,
    result: core.SamplingResult,
    *,
    target: np.ndarray,
) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ExperimentError("Pillow is required for sample rendering") from exc
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "starts": np.asarray(result.starts),
        "final_states": np.asarray(result.final_states),
    }
    for anchor, values in sorted(result.anchors.items()):
        arrays[f"anchor_{int(anchor):04d}"] = np.asarray(values)
        raw, demixed = _render_state(np.asarray(values)[0])
        Image.fromarray(raw).save(directory / f"anchor_{int(anchor):04d}_raw.png")
        Image.fromarray(demixed).save(
            directory / f"anchor_{int(anchor):04d}_demixed.png"
        )
    difference = np.abs(np.asarray(result.final_states)[0] - target)
    scaled = np.rint(255.0 * difference / max(float(np.max(difference)), 1.0e-30)).astype(
        np.uint8
    )
    Image.fromarray(scaled.reshape(28, 28)).save(directory / "final_difference.png")
    _save_npz(directory / "states.npz", **arrays)
    _atomic_json(directory / "telemetry.json", dict(result.telemetry))


def _write_contact_sheet(
    run_dir: Path,
    mode: str,
    row_names: Sequence[str],
    anchors: Sequence[int],
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        raise ExperimentError("Pillow is required for contact sheets") from exc
    scale = 4
    cell = 28 * scale
    left = 160
    top = 36
    sheet = Image.new(
        "L", (left + cell * len(anchors), top + cell * len(row_names)), color=255
    )
    draw = ImageDraw.Draw(sheet)
    for column, anchor in enumerate(anchors):
        draw.text((left + column * cell + 2, 8), str(anchor), fill=0)
    for row, name in enumerate(row_names):
        draw.text((4, top + row * cell + cell // 2 - 5), name, fill=0)
        for column, anchor in enumerate(anchors):
            path = (
                run_dir
                / "sampling"
                / mode
                / name
                / f"anchor_{int(anchor):04d}_demixed.png"
            )
            image = Image.open(path).convert("L").resize((cell, cell), resample=Image.NEAREST)
            sheet.paste(image, (left + column * cell, top + row * cell))
    target = run_dir / "sampling" / f"{mode}_contact_sheet.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)


def _run_sampling_mode(
    run_dir: Path,
    *,
    mode: str,
    starts: np.ndarray,
    label: int,
    path_id: int,
    target: np.ndarray,
    runtime: CandidateRuntime,
    reverse_seed: int,
    models: Mapping[str, torch.nn.Module],
    include_old_control: bool,
) -> dict[str, Any]:
    row_names = ["zero"]
    if include_old_control:
        row_names.append("small-old")
    row_names.extend(["small-weighted", "large-weighted", "oracle"])
    metrics: dict[str, Any] = {}
    for row in row_names:
        _log("sampling_row_start", mode=mode, controller=row)
        if row == "zero":
            controller = "null"
            model = None
            oracle_targets = None
        elif row == "oracle":
            controller = "oracle"
            model = None
            oracle_targets = target[None, :]
        else:
            controller = "learned"
            model = models[row]
            oracle_targets = None
        result = reverse_sample_candidate(
            starts,
            np.asarray([label], dtype=np.int64),
            [path_id],
            controller=controller,  # type: ignore[arg-type]
            root_seed=int(reverse_seed),
            runtime=runtime,
            model=model,  # type: ignore[arg-type]
            oracle_targets=oracle_targets,
            anchors=DEFAULT_ANCHORS,
            sample_steps=512,
        )
        directory = run_dir / "sampling" / mode / row
        _save_sampling_images(directory, result, target=target)
        clean_target = core.demix_unit_masses(target)
        metrics[row] = {
            "to_mixed_source": _controller_metric(result.final_states[0], target),
            "to_demixed_source": _controller_metric(
                core.demix_unit_masses(result.final_states[0]), clean_target
            ),
            "telemetry": dict(result.telemetry),
        }
        _atomic_json(directory / "metrics.json", metrics[row])
        _log(
            "sampling_row_complete",
            mode=mode,
            controller=row,
            squared_l2=metrics[row]["to_mixed_source"]["squared_l2"],
        )
    _write_contact_sheet(run_dir, mode, row_names, DEFAULT_ANCHORS)
    report = {
        "schema": VERSION + "-sampling-mode",
        "mode": mode,
        "rows": row_names,
        "anchors": list(DEFAULT_ANCHORS),
        "metrics": metrics,
    }
    _atomic_json(run_dir / "sampling" / f"{mode}_metrics.json", report)
    return report


def _scientific_interpretation(
    same_path: Mapping[str, Any], prior: Mapping[str, Any], include_old: bool
) -> dict[str, Any]:
    same = same_path["metrics"]
    prior_metrics = prior["metrics"]
    zero_same = float(same["zero"]["to_mixed_source"]["squared_l2"])
    small_same = float(same["small-weighted"]["to_mixed_source"]["squared_l2"])
    large_same = float(same["large-weighted"]["to_mixed_source"]["squared_l2"])
    oracle_same = float(same["oracle"]["to_mixed_source"]["squared_l2"])
    result: dict[str, Any] = {
        "same_path_improvement_small_weighted_over_zero": zero_same - small_same,
        "same_path_improvement_large_weighted_over_zero": zero_same - large_same,
        "same_path_improvement_large_over_small": small_same - large_same,
        "oracle_improves_over_zero": int(oracle_same < zero_same),
        "small_weighted_improves_over_zero": int(small_same < zero_same),
        "large_weighted_improves_over_zero": int(large_same < zero_same),
        "prior_small_weighted_improves_over_zero": int(
            float(prior_metrics["small-weighted"]["to_mixed_source"]["squared_l2"])
            < float(prior_metrics["zero"]["to_mixed_source"]["squared_l2"])
        ),
        "prior_large_weighted_improves_over_zero": int(
            float(prior_metrics["large-weighted"]["to_mixed_source"]["squared_l2"])
            < float(prior_metrics["zero"]["to_mixed_source"]["squared_l2"])
        ),
    }
    if include_old:
        old_same = float(same["small-old"]["to_mixed_source"]["squared_l2"])
        result.update(
            same_path_improvement_weighted_over_old=old_same - small_same,
            weighted_small_beats_old=int(small_same < old_same),
        )
    if not result["oracle_improves_over_zero"]:
        conclusion = "reverse-composition-or-oracle-control-failure"
    elif result["large_weighted_improves_over_zero"]:
        conclusion = (
            "capacity-likely-material"
            if large_same < small_same
            else "weighted-small-already-sufficient"
        )
    elif result["small_weighted_improves_over_zero"]:
        conclusion = "weighted-loss-helps-but-large-capacity-does-not"
    else:
        conclusion = "learned-time-local-score-does-not-yet-drive-reconstruction"
    result["automatic_interpretation"] = conclusion
    result["visual_recognizability_requires_contact_sheet_review"] = 1
    return result


def _write_report(
    run_dir: Path,
    *,
    config: ExperimentConfig,
    source: Mapping[str, Any],
    training_reports: Mapping[str, Mapping[str, Any]],
    same_path: Mapping[str, Any],
    prior: Mapping[str, Any],
    interpretation: Mapping[str, Any],
) -> None:
    rows = ["zero"]
    if config.include_old_control:
        rows.append("small-old")
    rows.extend(["small-weighted", "large-weighted", "oracle"])
    lines = [
        "# Path-weighted Jacobi/Rao–Blackwell capacity experiment",
        "",
        f"Run completed: `{_utc_now()}`",
        f"Source MNIST index/label: `{source['index']}` / `{source['label']}`",
        f"Candidate backend: `{CANDIDATE_BACKEND_NAME}`",
        "",
        "## Same-forward-endpoint reconstruction",
        "",
        "| Controller | Squared L2 to mixed source | L1 | Correlation |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        metric = same_path["metrics"][row]["to_mixed_source"]
        lines.append(
            f"| {row} | {metric['squared_l2']:.8g} | {metric['l1']:.8g} | "
            f"{metric['centered_correlation']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Independent Dirichlet-prior reconstruction",
            "",
            "| Controller | Squared L2 to mixed source | L1 | Correlation |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        metric = prior["metrics"][row]["to_mixed_source"]
        lines.append(
            f"| {row} | {metric['squared_l2']:.8g} | {metric['l1']:.8g} | "
            f"{metric['centered_correlation']:.6g} |"
        )
    lines.extend(["", "## Selected training checkpoints", ""])
    for name, report in training_reports.items():
        lines.append(
            f"- **{name}:** update {report['selected_update']}, validation metric "
            f"{report['selected_primary_metric']:.8g}, parameters {report['parameter_count']:,}."
        )
    lines.extend(
        [
            "",
            "## Automatic interpretation",
            "",
            f"`{interpretation['automatic_interpretation']}`",
            "",
            "The numerical comparison is paired, but visual recognizability must be checked in "
            "`sampling/same_path_contact_sheet.png` and `sampling/prior_contact_sheet.png`.",
            "",
        ]
    )
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    excluded_names = {
        "artifact_manifest.json",
        "worker.log",
        "watchdog.log",
        "runpod_finalization.json",
        "verification.json",
    }
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in excluded_names or path.suffix in {".zst", ".gz"}:
            continue
        relative = path.relative_to(run_dir).as_posix()
        files[relative] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    return {
        "schema": VERSION + "-artifact-manifest",
        "created_at": _utc_now(),
        "file_count": len(files),
        "files": files,
    }


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    required = [
        "run_config.json",
        "source/source.json",
        "candidate_audit/report.json",
        "training/small-weighted/report.json",
        "training/large-weighted/report.json",
        "sampling/same_path_metrics.json",
        "sampling/prior_metrics.json",
        "REPORT.md",
        "outcome.json",
        "artifact_manifest.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    manifest = (
        json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
        if not missing and (root / "artifact_manifest.json").is_file()
        else {"files": {}}
    )
    mismatches: dict[str, Any] = {}
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            mismatches[relative] = "missing"
            continue
        observed = _sha256_file(path)
        if observed != expected.get("sha256") or path.stat().st_size != expected.get("bytes"):
            mismatches[relative] = {
                "expected": expected,
                "observed_sha256": observed,
                "observed_bytes": path.stat().st_size,
            }
    report = {
        "schema": VERSION + "-verification",
        "run_dir": str(root.resolve()),
        "missing": missing,
        "mismatches": mismatches,
        "passed": int(not missing and not mismatches),
    }
    _atomic_json(root / "verification.json", report)
    return report


def _initialize_run(config: ExperimentConfig) -> Path:
    run_dir = Path(config.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    record = config.to_record()
    record["config_sha256"] = _json_sha256(record)
    path = run_dir / "run_config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != record:
            raise ExperimentError("run directory belongs to a different configuration")
    else:
        _atomic_json(path, record)
    return run_dir


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    started = time.monotonic()
    run_dir = _initialize_run(config)
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ExperimentError("production experiment requires a CUDA device")
    environment = _environment_record(device)
    _atomic_json(run_dir / "environment.json", environment)
    _complete_stage(run_dir, "00_environment", device=str(device))

    clean, label, source_record = _load_mnist_source(
        Path(config.data_dir), index=config.mnist_index, download=config.download_mnist
    )
    mixed = core.mix_unit_masses(clean)
    _save_source_images(run_dir / "source", clean, mixed)
    _atomic_json(run_dir / "source" / "source.json", source_record)
    _complete_stage(run_dir, "01_source", label=label, index=config.mnist_index)

    runtime = prepare_candidate_runtime(device=device, rng_keys=_candidate_rng_keys(config))
    _atomic_json(
        run_dir / "candidate_backend.json",
        {
            "schema": VERSION + "-candidate-backend",
            "backend": CANDIDATE_BACKEND_NAME,
            "target_semantics": CANDIDATE_TARGET_SEMANTICS,
            "candidate_binary_sha256": runtime.candidate_binary_sha256,
            "candidate_modes": int(runtime.profile.candidate_modes),
            "candidate_bisection_steps": int(runtime.profile.candidate_bisection_steps),
            "supported_outer_steps": [128, 512],
        },
    )
    if not _stage_complete(run_dir, "02_candidate_audit"):
        audit = _run_candidate_audit(
            run_dir, runtime, seed=_seed(config, "audit")
        )
        _complete_stage(run_dir, "02_candidate_audit", passed=audit["passed"])

    cache_root = run_dir / "cache"
    train_paths = tuple(range(0, int(config.train_paths)))
    validation_paths = tuple(
        range(10_000, 10_000 + int(config.validation_paths))
    )
    train_cache = build_candidate_prefix_cache(
        cache_root / "train",
        mixed,
        label=label,
        path_ids=train_paths,
        root_seed=_seed(config, "train_forward"),
        runtime=runtime,
        spec=CandidatePrefixCacheSpec(),
    )
    _complete_stage(
        run_dir,
        "03_train_cache",
        records=len(train_cache),
        paths=len(train_paths),
    )
    validation_cache = build_candidate_prefix_cache(
        cache_root / "validation",
        mixed,
        label=label,
        path_ids=validation_paths,
        root_seed=_seed(config, "validation_forward"),
        runtime=runtime,
        spec=CandidatePrefixCacheSpec(),
    )
    _complete_stage(
        run_dir,
        "04_validation_cache",
        records=len(validation_cache),
        paths=len(validation_paths),
    )

    loss_config = PathWeightedLossConfig(mobility_floor=config.mobility_floor)
    scales_path = run_dir / "training" / "loss_scales.json"
    # Recompute to retain one implementation path and catch cache corruption.
    scales = compute_cache_loss_scales(train_cache, loss_config=loss_config)
    _atomic_json(scales_path, scales.to_record())

    memorization_dir = run_dir / "training" / "large_memorization_gate"
    memorization_report_path = memorization_dir / "report.json"
    if memorization_report_path.is_file():
        memorization = json.loads(memorization_report_path.read_text(encoding="utf-8"))
    else:
        memorization = run_large_memorization_gate(
            train_cache,
            device=device,
            output_dir=memorization_dir,
            loss_config=loss_config,
            seed=_seed(config, "memorization"),
        )
    if not int(memorization.get("passed", 0)):
        raise ExperimentError("large architecture failed the finite-subset memorization gate")
    _complete_stage(
        run_dir,
        "05_memorization_gate",
        reduction=memorization["relative_reduction"],
    )

    training_reports: dict[str, Mapping[str, Any]] = {}
    if config.include_old_control:
        training_reports["small-old"] = train_capacity_model(
            train_cache,
            validation_cache,
            config=CapacityTrainingConfig(
                architecture="small",
                loss_name="old",
                updates=config.small_updates,
                batch_size=config.batch_size,
                learning_rate=1.0e-3,
                validation_interval=config.validation_interval,
                seed=_seed(config, "small_training"),
            ),
            device=device,
            scales=scales,
            output_dir=run_dir / "training" / "small-old",
            loss_config=loss_config,
        )
        _complete_stage(run_dir, "06_train_small_old")
    training_reports["small-weighted"] = train_capacity_model(
        train_cache,
        validation_cache,
        config=CapacityTrainingConfig(
            architecture="small",
            loss_name="path-weighted",
            updates=config.small_updates,
            batch_size=config.batch_size,
            learning_rate=1.0e-3,
            validation_interval=config.validation_interval,
            seed=_seed(config, "small_training"),
        ),
        device=device,
        scales=scales,
        output_dir=run_dir / "training" / "small-weighted",
        loss_config=loss_config,
    )
    _complete_stage(run_dir, "07_train_small_weighted")
    training_reports["large-weighted"] = train_capacity_model(
        train_cache,
        validation_cache,
        config=CapacityTrainingConfig(
            architecture="large",
            loss_name="path-weighted",
            updates=config.large_updates,
            batch_size=config.batch_size,
            learning_rate=3.0e-4,
            validation_interval=config.validation_interval,
            seed=_seed(config, "large_training"),
        ),
        device=device,
        scales=scales,
        output_dir=run_dir / "training" / "large-weighted",
        loss_config=loss_config,
    )
    _complete_stage(run_dir, "08_train_large_weighted")

    models: dict[str, torch.nn.Module] = {}
    if config.include_old_control:
        models["small-old"], _ = load_selected_model(
            run_dir / "training" / "small-old", device=device
        )
    models["small-weighted"], _ = load_selected_model(
        run_dir / "training" / "small-weighted", device=device
    )
    models["large-weighted"], _ = load_selected_model(
        run_dir / "training" / "large-weighted", device=device
    )

    evaluation_path_id = 20_000
    terminal_path = run_dir / "sampling" / "forward_terminal.npz"
    if terminal_path.is_file():
        terminal = np.load(terminal_path, allow_pickle=False)["terminal"]
    else:
        terminal, telemetry = forward_terminal_states_candidate(
            mixed[None, :],
            [evaluation_path_id],
            root_seed=_seed(config, "evaluation_forward"),
            runtime=runtime,
            sample_steps=512,
        )
        _save_npz(terminal_path, terminal=terminal)
        _atomic_json(run_dir / "sampling" / "forward_terminal_telemetry.json", telemetry)
    _complete_stage(run_dir, "09_forward_terminal")

    same_path_metrics_path = run_dir / "sampling" / "same_path_metrics.json"
    if _stage_complete(run_dir, "10_same_path_sampling") and same_path_metrics_path.is_file():
        same_path = json.loads(same_path_metrics_path.read_text(encoding="utf-8"))
    else:
        same_path = _run_sampling_mode(
            run_dir,
            mode="same_path",
            starts=np.asarray(terminal, dtype=np.float64),
            label=label,
            path_id=evaluation_path_id,
            target=mixed,
            runtime=runtime,
            reverse_seed=_seed(config, "same_path_reverse"),
            models=models,
            include_old_control=config.include_old_control,
        )
        _complete_stage(run_dir, "10_same_path_sampling")

    prior_metrics_path = run_dir / "sampling" / "prior_metrics.json"
    if _stage_complete(run_dir, "11_prior_sampling") and prior_metrics_path.is_file():
        prior = json.loads(prior_metrics_path.read_text(encoding="utf-8"))
    else:
        prior_start = core.sample_dirichlet_starts(
            [evaluation_path_id], root_seed=_seed(config, "prior_start")
        )
        prior = _run_sampling_mode(
            run_dir,
            mode="prior",
            starts=prior_start,
            label=label,
            path_id=evaluation_path_id,
            target=mixed,
            runtime=runtime,
            reverse_seed=_seed(config, "prior_reverse"),
            models=models,
            include_old_control=config.include_old_control,
        )
        _complete_stage(run_dir, "11_prior_sampling")

    interpretation = _scientific_interpretation(
        same_path, prior, config.include_old_control
    )
    _atomic_json(run_dir / "interpretation.json", interpretation)
    _atomic_json(
        run_dir / "architecture_contracts.json",
        {
            "small_parameter_count": core.GLOBAL_DILATED_PARAMETER_COUNT,
            "large": large_global_architecture_contract(),
        },
    )
    _write_report(
        run_dir,
        config=config,
        source=source_record,
        training_reports=training_reports,
        same_path=same_path,
        prior=prior,
        interpretation=interpretation,
    )
    outcome = {
        "schema": VERSION + "-outcome",
        "execution_status": "complete",
        "automatic_interpretation": interpretation["automatic_interpretation"],
        "visual_review_pending": 1,
        "elapsed_seconds": float(time.monotonic() - started),
        "hard_wall_seconds": int(config.hard_wall_seconds),
        "finished_at": _utc_now(),
    }
    _atomic_json(run_dir / "outcome.json", outcome)
    _complete_stage(run_dir, "12_complete", elapsed_seconds=outcome["elapsed_seconds"])
    manifest = _artifact_manifest(run_dir)
    _atomic_json(run_dir / "artifact_manifest.json", manifest)
    verification = verify_run(run_dir)
    if not verification["passed"]:
        raise ExperimentError("final artifact verification failed")
    return {"outcome": outcome, "verification": verification}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run or resume the full experiment")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--data-dir", default="/workspace/mnist_data")
    run.add_argument("--mnist-index", type=int, default=0)
    run.add_argument(
        "--download-mnist", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--train-paths", type=int, default=64)
    run.add_argument("--validation-paths", type=int, default=32)
    run.add_argument("--small-updates", type=int, default=12_000)
    run.add_argument("--large-updates", type=int, default=12_000)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--validation-interval", type=int, default=250)
    run.add_argument("--mobility-floor", type=float, default=1.0e-4)
    run.add_argument(
        "--include-old-control", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument("--hard-wall-seconds", type=int, default=21_600)
    run.add_argument("--seed", type=int, default=2_026_082_020)

    verify = subparsers.add_parser("verify", help="verify a completed run directory")
    verify.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        report = verify_run(args.run_dir)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0 if report["passed"] else 1
    config = ExperimentConfig(
        run_dir=args.run_dir,
        device=args.device,
        data_dir=args.data_dir,
        mnist_index=args.mnist_index,
        download_mnist=args.download_mnist,
        train_paths=args.train_paths,
        validation_paths=args.validation_paths,
        small_updates=args.small_updates,
        large_updates=args.large_updates,
        batch_size=args.batch_size,
        validation_interval=args.validation_interval,
        mobility_floor=args.mobility_floor,
        include_old_control=args.include_old_control,
        hard_wall_seconds=args.hard_wall_seconds,
        seed=args.seed,
    )
    try:
        result = run_experiment(config)
    except Exception as exc:
        run_dir = Path(config.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            run_dir / "failure.json",
            {
                "schema": VERSION + "-failure",
                "at": _utc_now(),
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        _log("experiment_failed", error_type=type(exc).__name__, message=str(exc))
        raise
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
