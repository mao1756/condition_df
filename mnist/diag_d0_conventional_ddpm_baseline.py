from __future__ import annotations

"""Bounded exploratory conventional pixel-DDPM calibration experiment."""

import argparse, copy, csv, hashlib, json, math, os, platform, random, re, subprocess, sys, tempfile, time
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from mnist.conditioned_diffusion import SmallMnistCNN, evaluate_image_classifier
from mnist.mnist_generation_benchmark import compute_generation_metrics, load_test_mnist_terminal, load_train_validation_mnist, model_to_uint8, score_human_review, train_frozen_image_evaluator, uint8_to_eval, uint8_to_model, validate_generated_batch, write_blinded_review_bundle, write_contact_sheet
from mnist.pixel_ddpm import ClassConditionalUNet28, ddpm_step_from_epsilon, epsilon_from_x0, epsilon_prediction_loss, make_linear_ddpm_schedule, q_sample, sample_reverse, update_ema_


VERSION = "pixel-ddpm-calibration-v1"
EXPECTED_ARFF_SHA256 = "418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b"
DIRECT_SOURCE_FILES = ("mnist/diag_d0_conventional_ddpm_baseline.py", "mnist/pixel_ddpm.py", "mnist/mnist_generation_benchmark.py", "mnist/conditioned_diffusion.py", "mnist/weighted_point_cloud.py", "core/conditioning_utils.py", "core/wasserstein_conditioning_algorithms.py", "mnist/__init__.py", "core/__init__.py")
PLACEHOLDER = re.compile(r"(?:<[^>]+>|fresh[ _-]*approval|placeholder|todo|tbd)", re.I)

FROZEN_CONFIG: dict[str, Any] = {
    "schema": VERSION,
    "research_mode": "exploratory",
    "decision": "can one fixed conventional Gaussian pixel DDPM establish the common MNIST benchmark",
    "benchmark_role": "calibration control for a materially different future Eulerian formulation",
    "data": {"sha256": EXPECTED_ARFF_SHA256, "train": [0, 55_000], "validation": [55_000, 60_000], "test": [60_000, 70_000]},
    "model": {"kind": "unet28", "parameter_count": 1_378_593, "time_embedding": 128, "conditioning": 256},
    "schedule": {"steps": 1000, "beta_start": 1e-4, "beta_end": 2e-2},
    "training": {"epochs": 40, "batch_size": 128, "lr": 2e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "clip_norm": 1.0, "ema_decay": 0.999, "validation_per_class": 100},
    "evaluator": {"epochs": 8, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4, "minimum_accuracy": 0.97},
    "reconstruction": {"per_class": 2, "start_timesteps": [99, 499, 999], "oracle_max_mse": 1e-6, "oracle_min_reduction": 0.99},
    "sampling": {"batches": 4, "per_class_per_batch": 4, "anchors": [0, 250, 500, 750, 1000], "review_within_class": [0, 5, 10, 15]},
    "diagnostic": {"classifier_accuracy": 0.80, "human_agreement": 0.75, "duplicate_pairs": 0, "diversity_ratio": 0.25},
    "seeds": {"model": 0xDD1001, "permutation": 0xDD1002, "train_noise": 0xDD1003, "validation": 0xDD1004, "evaluator": 0xDD1005, "reconstruction_forward": 0xDD1006, "reconstruction_reverse": 0xDD1007, "prior_start": 0xDD2000, "prior_reverse": 0xDD3000, "review": 0xDD4000},
    "resource_defaults": {"active_seconds": 7200.0, "cuda_fraction": 0.75, "storage_mib": 500.0, "terminal_reserve_seconds": 900.0},
    "automatic_launches": 0,
}


class DDPMRunError(RuntimeError): pass


class ResourcePause(DDPMRunError): pass


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise DDPMRunError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(_canonical_bytes({"dtype": array.dtype.str, "shape": array.shape}))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        for attempt in range(6):
            try: os.replace(temporary, path); break
            except PermissionError:
                if attempt == 5: raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _replace(path, lambda tmp: tmp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    names = list(fieldnames or (list(rows[0]) if rows else []))
    def writer(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=names)
            output.writeheader()
            output.writerows(rows)
    _replace(path, writer)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    def writer(tmp: Path) -> None:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
    _replace(path, writer)


def _write_npy(path: Path, value: np.ndarray) -> None:
    def writer(tmp: Path) -> None:
        with tmp.open("wb") as handle: np.save(handle, value, allow_pickle=False)
    _replace(path, writer)


def _write_torch(path: Path, value: Any) -> None:
    _replace(path, lambda tmp: torch.save(value, tmp))


def _utc_now() -> str:
    import datetime; return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _tree_digest(path: Path) -> str:
    return _semantic_sha256([(p.relative_to(path).as_posix(), p.stat().st_size, _file_sha256(p)) for p in sorted(path.rglob("*")) if p.is_file()])


def _contact_array(images: np.ndarray, columns: int, captions: Sequence[str] | None = None) -> np.ndarray:
    images = np.asarray(images); width, height, caption_height = 112, 112, 12 if captions is not None else 0; rows = math.ceil(len(images) / columns)
    sheet = Image.new("L", (columns * width, rows * (height + caption_height)), 255); draw = ImageDraw.Draw(sheet)
    for index, array in enumerate(images):
        row, column = divmod(index, columns); x, y = column * width, row * (height + caption_height); sheet.paste(Image.fromarray(array).resize((width, height), Image.Resampling.NEAREST), (x, y))
        if captions is not None: draw.text((x + 1, y + height + 1), str(captions[index]), fill=0)
    return np.asarray(sheet)


def _verify_npz(path: Path, expected: Mapping[str, np.ndarray], message: str) -> None:
    with np.load(path, allow_pickle=False) as archive: _require(set(archive.files) == set(expected) and all(np.array_equal(archive[key], value) for key, value in expected.items()), message)


def _noise_id(array: np.ndarray) -> str:
    alphabet = "abcdefghijklmnop"; return "id-" + "".join(alphabet[n] for byte in hashlib.sha256(array.tobytes()).digest()[:10] for n in (byte >> 4, byte & 15))


def _refresh_manifest(run_dir: Path) -> dict[str, Any]:
    _require(all(not p.is_symlink() for p in run_dir.rglob("*")), "manifest refuses linked paths")
    files = [p for p in sorted(run_dir.rglob("*")) if p.is_file() and p != run_dir / "artifact_manifest.json"]
    _require(all(not p.is_symlink() and p.stat().st_nlink == 1 for p in files), "manifest refuses linked artifacts")
    rows = [{"path": p.relative_to(run_dir).as_posix(), "size": p.stat().st_size, "sha256": _file_sha256(p)} for p in files]
    manifest = {"schema": VERSION + "-artifact-manifest", "artifact_count": len(rows), "artifact_bytes": sum(r["size"] for r in rows), "artifacts": rows}
    _write_json(run_dir / "artifact_manifest.json", manifest)
    return manifest


def _verify_manifest(run_dir: Path, allowed_extra: set[str] | None = None) -> dict[str, Any]:
    _require(all(not p.is_symlink() for p in run_dir.rglob("*")), "manifest contains a linked path")
    manifest = _read_json(run_dir / "artifact_manifest.json")
    rows = manifest.get("artifacts")
    _require(isinstance(rows, list), "artifact manifest rows changed")
    expected = [r.get("path") for r in rows]
    actual = [p.relative_to(run_dir).as_posix() for p in sorted(run_dir.rglob("*")) if p.is_file() and p != run_dir / "artifact_manifest.json" and p.relative_to(run_dir).as_posix() not in (allowed_extra or set())]
    _require(expected == actual, "artifact manifest inventory changed")
    for row in rows:
        path = run_dir / str(row["path"])
        _require(not path.is_symlink() and path.stat().st_nlink == 1 and path.stat().st_size == row.get("size") and _file_sha256(path) == row.get("sha256"), f"artifact changed: {row['path']}")
    _require(manifest.get("schema") == VERSION + "-artifact-manifest" and manifest.get("artifact_count") == len(rows) and manifest.get("artifact_bytes") == sum(r["size"] for r in rows), "artifact manifest totals changed")
    return manifest


def _approval(value: str) -> str:
    raw, cleaned = str(value), str(value).strip()
    _require(bool(cleaned) and not PLACEHOLDER.search(cleaned), "a real explicit approval reference is required")
    return raw


def _source_hashes(repository_root: Path) -> dict[str, str]:
    result = {}
    for relative in DIRECT_SOURCE_FILES:
        path = repository_root / relative
        _require(path.is_file(), f"load-bearing source is missing: {relative}")
        result[relative] = _file_sha256(path)
    return result


def _git_revision(repository_root: Path) -> str:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository_root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise DDPMRunError("source revision is unavailable") from error
    _require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "source revision is unavailable"); return revision


def _environment(device: str) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "cuda_available": cuda, "device": device, "gpu": torch.cuda.get_device_name(torch.device(device)) if cuda and torch.device(device).type == "cuda" else None, "os": platform.platform()}


def _status(run_dir: Path, state: str, *, resumable: bool, error: str | None = None) -> None:
    _write_json(run_dir / "status.json", {"schema": VERSION + "-status", "state": state, "resumable": int(resumable), "error": error, "updated_at": _utc_now()})


def _ledger(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "resource_ledger.json")


def _charge(run_dir: Path, role: str, seconds: float, failed: bool = False) -> None:
    ledger = _ledger(run_dir)
    seconds = max(0.0, float(seconds))
    ledger["active_seconds"] = math.fsum((float(ledger["active_seconds"]), seconds))
    ledger["events"].append({"role": role, "seconds": seconds, "failed": int(failed), "at": _utc_now()})
    _write_json(run_dir / "resource_ledger.json", ledger)


def _charged(role: str) -> Any:
    def decorate(function: Any) -> Any:
        @wraps(function)
        def call(run_dir: Path, *args: Any, **kwargs: Any) -> Any:
            device = next((value for value in (*args, *kwargs.values()) if isinstance(value, torch.device)), torch.device("cpu")); reserve = 900.0 if role == "evaluator_training" else 0.0
            _resource_check(Path(run_dir), device, reserve=reserve); started = time.perf_counter()
            try: result = function(run_dir, *args, **kwargs)
            except BaseException: _charge(Path(run_dir), role, time.perf_counter() - started, True); raise
            _charge(Path(run_dir), role, time.perf_counter() - started); _resource_check(Path(run_dir), device, reserve=reserve); return result
        return call
    return decorate


def _resource_check(run_dir: Path, device: torch.device, *, inflight: float = 0.0, reserve: float = 0.0) -> None:
    ledger = _ledger(run_dir)
    storage = _directory_bytes(run_dir); ledger["peak_storage_bytes"] = max(int(ledger.get("peak_storage_bytes", 0)), storage)
    fraction = 0.0
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.max_memory_allocated(device); fraction = allocated / total if total else 0.0; ledger["peak_cuda_allocated_bytes"] = max(int(ledger.get("peak_cuda_allocated_bytes", 0)), allocated); ledger["peak_cuda_fraction"] = max(float(ledger.get("peak_cuda_fraction", 0.0)), fraction)
    _write_json(run_dir / "resource_ledger.json", ledger)
    if float(ledger["active_seconds"]) + inflight + reserve >= float(ledger["maximum_active_seconds"]): raise ResourcePause("approved active-time cap reached")
    if storage >= int(ledger["maximum_storage_bytes"]): raise ResourcePause("persisted-storage cap reached")
    if fraction >= float(ledger["maximum_cuda_fraction"]): raise ResourcePause("CUDA allocation cap reached")


def _schedule(config: Mapping[str, Any], device: torch.device | str = "cpu") -> Any:
    row = config["schedule"]
    return make_linear_ddpm_schedule(int(row["steps"]), float(row["beta_start"]), float(row["beta_end"]), device=device)


def _schedule_arrays(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    schedule = _schedule(config)
    return {"betas": schedule.betas.numpy(), "alphas": schedule.alphas.numpy(), "alpha_bars": schedule.alpha_bars.numpy()}


def initialize_run(
    repository_root: Path,
    arff: Path,
    run_dir: Path,
    *,
    device: str,
    maximum_active_seconds: float,
    maximum_cuda_fraction: float,
    maximum_storage_mib: float,
    approval_reference: str,
    resume: bool = False,
    config: Mapping[str, Any] = FROZEN_CONFIG,
) -> Path:
    repository_root, arff, run_dir = Path(repository_root).resolve(), Path(arff).resolve(), Path(run_dir).resolve()
    approval = _approval(approval_reference)
    _require(repository_root.is_dir() and arff.is_file(), "repository or authenticated ARFF is missing")
    _require(run_dir != repository_root and run_dir not in arff.parents and arff not in run_dir.parents, "output overlaps an input")
    _require(maximum_active_seconds > float(config["resource_defaults"]["terminal_reserve_seconds"]) and 0 < maximum_cuda_fraction <= 0.75 and 0 < maximum_storage_mib <= 500, "resource cap is invalid")
    source_hashes, revision, data_hash, config_hash = _source_hashes(repository_root), _git_revision(repository_root), _file_sha256(arff), _semantic_sha256(config)
    _require(data_hash == EXPECTED_ARFF_SHA256, "authenticated MNIST ARFF hash mismatch")
    if resume:
        _require(run_dir.is_dir(), "resume run directory is missing")
        bindings, ledger = _read_json(run_dir / "source_bindings.json"), _ledger(run_dir)
        _require(bindings.get("source_files") == source_hashes and bindings.get("git_revision") == revision and bindings.get("config_sha256") == config_hash and bindings.get("data_sha256") == data_hash, "source/config/data mismatch blocks resume")
        saved_status = _read_json(run_dir / "status.json"); _require(_read_json(run_dir / "environment.json") == _environment(device), "resume environment changed")
        _require(maximum_active_seconds >= float(ledger["maximum_active_seconds"]), "active cap cannot decrease on resume")
        _require(float(ledger["maximum_cuda_fraction"]) == maximum_cuda_fraction and int(ledger["maximum_storage_bytes"]) == int(maximum_storage_mib * 1024**2), "CUDA/storage caps changed on resume")
        if maximum_active_seconds == float(ledger["maximum_active_seconds"]): _require(approval == ledger["approval_reference"], "resume approval differs from the bound approval")
        else: _require(approval != ledger["approval_reference"], "a cap extension requires a new approval reference")
        if saved_status.get("state") in {"awaiting_human_review", "complete"}: return run_dir
        _require(saved_status.get("resumable") == 1, "run is not resumable")
        if (run_dir / "artifact_manifest.json").is_file(): _verify_manifest(run_dir); (run_dir / "artifact_manifest.json").unlink()
        if maximum_active_seconds > float(ledger["maximum_active_seconds"]):
            ledger["amendments"].append({"old_cap": ledger["maximum_active_seconds"], "new_cap": maximum_active_seconds, "approval_reference": approval, "at": _utc_now()})
            ledger["maximum_active_seconds"] = maximum_active_seconds
            ledger["approval_reference"] = approval
            _write_json(run_dir / "resource_ledger.json", ledger)
        ledger = _ledger(run_dir); ledger["restart_count"] += 1; _write_json(run_dir / "resource_ledger.json", ledger)
        return run_dir
    _require(not run_dir.exists(), "run directory already exists; use --resume")
    train_u8, train_y, validation_u8, validation_y = load_train_validation_mnist(arff)
    _require((train_u8.shape[0], validation_u8.shape[0]) == (55_000, 5_000), "fixed train/validation slices changed")
    run_dir.mkdir(parents=True)
    for name in ("data", "controls/images", "evaluator", "training", "evaluation/images", "review"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    indices = {"train": np.arange(55_000, dtype=np.int64), "validation": np.arange(55_000, 60_000, dtype=np.int64), "test": np.arange(60_000, 70_000, dtype=np.int64)}
    for role, values in indices.items():
        _write_npy(run_dir / f"data/{role}_indices.npy", values)
    split = {role: {"start": int(values[0]), "stop": int(values[-1] + 1), "count": len(values), "sha256": _array_sha256(values)} for role, values in indices.items()}
    split["pairwise_disjoint"] = 1
    _write_json(run_dir / "data/split.json", split)
    _write_json(run_dir / "config.json", dict(config))
    _write_json(run_dir / "source_bindings.json", {"repository_root": str(repository_root), "git_revision": revision, "arff": str(arff), "data_sha256": data_hash, "config_sha256": config_hash, "source_files": source_hashes})
    _write_json(run_dir / "environment.json", _environment(device))
    _write_json(run_dir / "resource_ledger.json", {"schema": VERSION + "-resource-ledger", "maximum_active_seconds": float(maximum_active_seconds), "maximum_cuda_fraction": float(maximum_cuda_fraction), "maximum_storage_bytes": int(maximum_storage_mib * 1024**2), "approval_reference": approval, "active_seconds": 0.0, "peak_cuda_allocated_bytes": 0, "peak_cuda_fraction": 0.0, "peak_storage_bytes": 0, "events": [], "amendments": [], "restart_count": 0, "latest_projection": None})
    _replace(run_dir / "command.txt", lambda p: p.write_text(subprocess.list2cmdline([sys.executable, "-B", "-m", "mnist.diag_d0_conventional_ddpm_baseline", *sys.argv[1:]]) + "\n", encoding="utf-8"))
    _write_npz(run_dir / "controls/schedule.npz", **_schedule_arrays(config))
    _status(run_dir, "initialized", resumable=True)
    return run_dir


def _panel(validation_u8: np.ndarray, validation_y: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    per_class = int(config["reconstruction"]["per_class"])
    local = np.concatenate([np.flatnonzero(validation_y == digit)[:per_class] for digit in range(10)]).astype(np.int64)
    _require(local.shape == (10 * per_class,), "validation reconstruction panel is incomplete")
    horizons = np.asarray(config["reconstruction"]["start_timesteps"], dtype=np.int64)
    x0 = torch.from_numpy(uint8_to_model(validation_u8[local])).float()
    generator = torch.Generator().manual_seed(int(config["seeds"]["reconstruction_forward"]))
    noise = torch.randn((len(local), len(horizons), 1, 28, 28), generator=generator)
    schedule = _schedule(config)
    starts = torch.stack([q_sample(x0, int(t), noise[:, j], schedule) for j, t in enumerate(horizons)], dim=1)
    seeds = np.arange(len(local) * len(horizons), dtype=np.int64).reshape(len(local), len(horizons)) + int(config["seeds"]["reconstruction_reverse"])
    return {"validation_local_indices": local, "global_indices": local + 55_000, "labels": validation_y[local], "start_timesteps": horizons, "x0": x0.numpy(), "forward_noise": noise.numpy(), "starts": starts.numpy(), "reverse_seeds": seeds}


def _paired_reverse(start: torch.Tensor, x0: torch.Tensor, label: int, start_t: int, seed: int, schedule: Any, model: nn.Module | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    rows = 3 if model is not None else 2
    state = start.repeat(rows, 1, 1, 1)
    target = x0.repeat(rows, 1, 1, 1)
    generator = torch.Generator(device=state.device).manual_seed(int(seed))
    for timestep in range(start_t, -1, -1):
        eps = torch.zeros_like(state)
        eps[1:2] = epsilon_from_x0(state[1:2], target[1:2], timestep, schedule)
        if model is not None:
            t = torch.full((1,), timestep, dtype=torch.long, device=state.device)
            eps[2:3] = model(state[2:3], t, torch.tensor([label], device=state.device))
        noise = torch.randn(start.shape, generator=generator, device=state.device) if timestep else torch.zeros_like(start)
        state = ddpm_step_from_epsilon(state, timestep, eps, schedule, noise.repeat(rows, 1, 1, 1))
    return state, target


def run_oracle_preflight(run_dir: Path, validation_u8: np.ndarray, validation_y: np.ndarray, device: torch.device, config: Mapping[str, Any] = FROZEN_CONFIG) -> dict[str, Any]:
    _resource_check(run_dir, device, reserve=float(config["resource_defaults"]["terminal_reserve_seconds"]))
    panel = _panel(validation_u8, validation_y, config)
    _write_npz(run_dir / "controls/reconstruction_panel.npz", **panel)
    schedule = _schedule(config, device)
    endpoints, rows = [], []
    started = time.perf_counter()
    try:
        for i, label in enumerate(panel["labels"]):
            for j, horizon in enumerate(panel["start_timesteps"]):
                state, target = _paired_reverse(torch.from_numpy(panel["starts"][i, j:j + 1]).to(device), torch.from_numpy(panel["x0"][i:i + 1]).to(device), int(label), int(horizon), int(panel["reverse_seeds"][i, j]), schedule)
                mse = torch.mean((state - target) ** 2, dim=(1, 2, 3)).cpu().numpy()
                reduction = 1.0 - float(mse[1]) / max(float(mse[0]), np.finfo(float).tiny)
                rows.append({"panel_index": i, "horizon": int(horizon), "zero_mse": float(mse[0]), "oracle_mse": float(mse[1]), "oracle_reduction": reduction})
                endpoints.append(state.cpu().numpy())
        maximum = max(row["oracle_mse"] for row in rows)
        medians = {str(int(t)): float(np.median([r["oracle_reduction"] for r in rows if r["horizon"] == t])) for t in panel["start_timesteps"]}
        passed = int(np.isfinite(np.asarray(endpoints)).all() and maximum <= float(config["reconstruction"]["oracle_max_mse"]) and min(medians.values()) >= float(config["reconstruction"]["oracle_min_reduction"]))
        result = {"schema": VERSION + "-oracle-preflight", "gate": "B", "passed": passed, "maximum_oracle_mse": maximum, "median_reduction_by_horizon": medians, "rows": rows}
        _write_npz(run_dir / "controls/oracle_preflight_endpoints.npz", endpoints=np.asarray(endpoints, dtype=np.float32))
        _write_json(run_dir / "controls/oracle_preflight.json", result)
        _require(passed, "Gate B analytic reverse-composition oracle failed")
        _resource_check(run_dir, device, inflight=time.perf_counter() - started, reserve=float(config["resource_defaults"]["terminal_reserve_seconds"]))
        return result
    finally:
        _charge(run_dir, "oracle_preflight", time.perf_counter() - started, failed=not (run_dir / "controls/oracle_preflight.json").is_file())


@_charged("evaluator_training")
def train_or_load_evaluator(run_dir: Path, train_u8: np.ndarray, train_y: np.ndarray, validation_u8: np.ndarray, validation_y: np.ndarray, device: torch.device, config: Mapping[str, Any] = FROZEN_CONFIG) -> tuple[SmallMnistCNN, dict[str, Any]]:
    checkpoint = run_dir / "evaluator/selected_checkpoint.pt"
    if all((run_dir / path).is_file() for path in ("evaluator/selected_checkpoint.pt", "evaluator/history.csv", "evaluator/selection.json", "evaluator/real_validation_metrics.json")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        selection = _read_json(run_dir / "evaluator/selection.json"); _require(selection["checkpoint_sha256"] == _file_sha256(checkpoint) and selection["selected_epoch"] == payload["selected_epoch"], "evaluator checkpoint binding changed")
        model = SmallMnistCNN(); model.load_state_dict(payload["state_dict"])
        return model.to(device).eval(), selection
    row = config["evaluator"]
    model, result = train_frozen_image_evaluator(uint8_to_eval(train_u8), train_y, uint8_to_eval(validation_u8), validation_y, epochs=int(row["epochs"]), batch_size=int(row["batch_size"]), lr=float(row["lr"]), weight_decay=float(row["weight_decay"]), seed=int(config["seeds"]["evaluator"]), device=device, verbose=True)
    restored = evaluate_image_classifier(model, uint8_to_eval(validation_u8), validation_y, batch_size=int(row["batch_size"]), device=device)
    result["restored_validation_accuracy"], result["restored_validation_loss"] = float(restored["accuracy"]), float(restored["loss"])
    history = [{"epoch": i + 1, **{key: values[i] for key, values in result["history"].items()}} for i in range(int(row["epochs"]))]
    _write_csv(run_dir / "evaluator/history.csv", history)
    selection = {key: value for key, value in result.items() if key != "history"}
    selection["gate_c_validation_passed"] = int(float(selection["restored_validation_accuracy"]) >= float(row["minimum_accuracy"]))
    _write_torch(checkpoint, {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "selected_epoch": selection["selected_epoch"]})
    selection["checkpoint_sha256"] = _file_sha256(checkpoint)
    _write_json(run_dir / "evaluator/selection.json", selection)
    _write_json(run_dir / "evaluator/real_validation_metrics.json", {"accuracy": selection["restored_validation_accuracy"], "loss": selection["restored_validation_loss"], "gate_c_validation_passed": selection["gate_c_validation_passed"]})
    return model.eval(), selection


@torch.no_grad()
def _validation_mse(model: nn.Module, bank: Mapping[str, np.ndarray], schedule: Any, device: torch.device, batch_size: int) -> float:
    model.eval(); total = 0.0
    for start in range(0, len(bank["labels"]), batch_size):
        sl = slice(start, start + batch_size)
        x0 = torch.from_numpy(bank["x0"][sl]).to(device)
        labels = torch.from_numpy(bank["labels"][sl]).to(device)
        timesteps = torch.from_numpy(bank["timesteps"][sl]).to(device)
        noise = torch.from_numpy(bank["noise"][sl]).to(device)
        prediction = model(q_sample(x0, timesteps, noise, schedule), timesteps, labels)
        total += float(torch.sum((prediction - noise) ** 2).item())
    return total / (len(bank["labels"]) * 784)


@torch.no_grad()
def _validation_prediction_rms(model: nn.Module, bank: Mapping[str, np.ndarray], schedule: Any, device: torch.device, batch_size: int) -> float:
    model.eval(); total = 0.0
    for start in range(0, len(bank["labels"]), batch_size):
        sl = slice(start, start + batch_size); x0 = torch.from_numpy(bank["x0"][sl]).to(device); labels = torch.from_numpy(bank["labels"][sl]).to(device); timesteps = torch.from_numpy(bank["timesteps"][sl]).to(device); noise = torch.from_numpy(bank["noise"][sl]).to(device)
        total += float(torch.sum(model(q_sample(x0, timesteps, noise, schedule), timesteps, labels) ** 2).item())
    return math.sqrt(total / (len(bank["labels"]) * 784))


def _make_validation_bank(validation_u8: np.ndarray, validation_y: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    per_class = int(config["training"]["validation_per_class"]); local = np.concatenate([np.flatnonzero(validation_y == digit)[:per_class] for digit in range(10)]).astype(np.int64)
    _require(len(local) == 10 * per_class, "fixed validation bank is incomplete"); generator = torch.Generator().manual_seed(int(config["seeds"]["validation"]))
    return {"validation_local_indices": local, "global_indices": local + 55_000, "labels": validation_y[local].astype(np.int64), "timesteps": torch.randint(int(config["schedule"]["steps"]), (len(local),), generator=generator).numpy(), "noise": torch.randn((len(local), 1, 28, 28), generator=generator).numpy(), "x0": uint8_to_model(validation_u8[local])}


def _validation_bank(run_dir: Path, validation_u8: np.ndarray, validation_y: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = run_dir / "controls/validation_noise_bank.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    bank = _make_validation_bank(validation_u8, validation_y, config)
    _write_npz(path, **bank)
    return bank


def train_or_resume_generator(run_dir: Path, train_u8: np.ndarray, train_y: np.ndarray, validation_u8: np.ndarray, validation_y: np.ndarray, device: torch.device, config: Mapping[str, Any] = FROZEN_CONFIG) -> dict[str, Any]:
    row, seeds, setup_started = config["training"], config["seeds"], time.perf_counter()
    random.seed(int(seeds["model"])); np.random.seed(int(seeds["model"])); torch.manual_seed(int(seeds["model"])); torch.cuda.manual_seed_all(int(seeds["model"]))
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = ClassConditionalUNet28().to(device)
    _require(sum(p.numel() for p in model.parameters()) == int(config["model"]["parameter_count"]), "frozen generator parameter count changed")
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(row["lr"]), betas=tuple(row["betas"]), eps=float(row["eps"]), weight_decay=float(row["weight_decay"]))
    bank, schedule = _validation_bank(run_dir, validation_u8, validation_y, config), _schedule(config, device)
    latest = run_dir / "training/latest.pt"
    history: list[dict[str, Any]] = []
    start_epoch, best_mse, best_epoch = 1, math.inf, 0
    if latest.is_file():
        payload = torch.load(latest, map_location=device, weights_only=True)
        _require(payload["config_sha256"] == _semantic_sha256(config) and payload["data_sha256"] == _read_json(run_dir / "source_bindings.json")["data_sha256"], "generator resume binding changed")
        model.load_state_dict(payload["model_state"]); ema.load_state_dict(payload["ema_state"]); optimizer.load_state_dict(payload["optimizer_state"])
        history, start_epoch, best_mse, best_epoch = payload["history"], int(payload["completed_epoch"]) + 1, float(payload["best_mse"]), int(payload["best_epoch"])
    else:
        raw0 = _validation_mse(model, bank, schedule, device, int(row["batch_size"])); ema0 = _validation_mse(ema, bank, schedule, device, int(row["batch_size"]))
        history.append({"epoch": 0, "raw_validation_mse": raw0, "ema_validation_mse": ema0, "eligible": 0, "train_mse": None, "maximum_gradient_norm": None})
    _charge(run_dir, "generator_setup", time.perf_counter() - setup_started); reserve = float(config["resource_defaults"]["terminal_reserve_seconds"]); _resource_check(run_dir, device, reserve=reserve) if reserve else None
    for epoch in range(start_epoch, int(row["epochs"]) + 1):
        started = time.perf_counter(); model.train(); total_loss = 0.0; maximum_gradient = 0.0
        permutation = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(int(seeds["permutation"]) + epoch))
        generator = torch.Generator(device=device).manual_seed(int(seeds["train_noise"]) + epoch)
        try:
            for offset in range(0, len(train_y), int(row["batch_size"])):
                indices = permutation[offset:offset + int(row["batch_size"])].numpy()
                x0 = torch.from_numpy(uint8_to_model(train_u8[indices])).to(device); labels = torch.from_numpy(train_y[indices]).to(device)
                timesteps = torch.randint(int(config["schedule"]["steps"]), (len(indices),), generator=generator, device=device)
                noise = torch.randn(x0.shape, generator=generator, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = epsilon_prediction_loss(model, x0, timesteps, labels, noise, schedule)
                _require(torch.isfinite(loss).item(), "nonfinite generator loss")
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(row["clip_norm"])).item())
                _require(math.isfinite(gradient), "nonfinite generator gradient")
                optimizer.step(); _require(all(torch.isfinite(p).all() for p in model.parameters()), "nonfinite generator parameter"); update_ema_(ema, model, float(row["ema_decay"])); total_loss += float(loss.item()) * len(indices); maximum_gradient = max(maximum_gradient, gradient)
                _resource_check(run_dir, device, inflight=time.perf_counter() - started, reserve=float(config["resource_defaults"]["terminal_reserve_seconds"]))
        except BaseException:
            _charge(run_dir, f"generator_epoch_{epoch}", time.perf_counter() - started, failed=True)
            raise
        raw_mse = _validation_mse(model, bank, schedule, device, int(row["batch_size"])); ema_mse = _validation_mse(ema, bank, schedule, device, int(row["batch_size"]))
        history.append({"epoch": epoch, "train_mse": total_loss / len(train_y), "raw_validation_mse": raw_mse, "ema_validation_mse": ema_mse, "maximum_gradient_norm": maximum_gradient, "eligible": 1})
        if math.isfinite(ema_mse) and ema_mse < best_mse:
            best_mse, best_epoch = ema_mse, epoch
            _write_torch(run_dir / "training/selected_checkpoint.pt", {"state_dict": {k: v.detach().cpu() for k, v in ema.state_dict().items()}, "selected_epoch": epoch, "validation_mse": ema_mse})
        payload = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "ema_state": {k: v.detach().cpu() for k, v in ema.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "completed_epoch": epoch, "updates": epoch * math.ceil(len(train_y) / int(row["batch_size"])), "best_mse": best_mse, "best_epoch": best_epoch, "history": history, "config_sha256": _semantic_sha256(config), "data_sha256": _read_json(run_dir / "source_bindings.json")["data_sha256"]}
        _write_torch(latest, payload); _write_csv(run_dir / "training/history.csv", history)
        elapsed = time.perf_counter() - started; _charge(run_dir, f"generator_epoch_{epoch}", elapsed)
        if epoch == 1:
            ledger = _ledger(run_dir); projected = float(ledger["active_seconds"]) + elapsed * (int(row["epochs"]) - 1) + float(config["resource_defaults"]["terminal_reserve_seconds"])
            ledger["latest_projection"] = {"after_epoch": 1, "projected_active_seconds": projected, "passed": int(projected <= float(ledger["maximum_active_seconds"]))}; _write_json(run_dir / "resource_ledger.json", ledger)
            if projected > float(ledger["maximum_active_seconds"]): raise ResourcePause("epoch-1 projection exceeds the approved cap")
    selection_started = time.perf_counter(); selected_payload = torch.load(run_dir / "training/selected_checkpoint.pt", map_location="cpu", weights_only=True); selected_model = ClassConditionalUNet28().to(device); selected_model.load_state_dict(selected_payload["state_dict"])
    selection = {"selected_epoch": best_epoch, "validation_mse": best_mse, "checkpoint_sha256": _file_sha256(run_dir / "training/selected_checkpoint.pt"), "completed_epochs": int(row["epochs"]), "zero_predictor_mse": float(np.mean(bank["noise"].astype(np.float64) ** 2)), "learned_epsilon_rms": _validation_prediction_rms(selected_model, bank, schedule, device, int(row["batch_size"]))}
    _write_json(run_dir / "training/selection.json", selection)
    _charge(run_dir, "generator_selection", time.perf_counter() - selection_started); _resource_check(run_dir, device, reserve=float(config["resource_defaults"]["terminal_reserve_seconds"]))
    return selection


def _load_generator(run_dir: Path, device: torch.device, config: Mapping[str, Any]) -> nn.Module:
    payload = torch.load(run_dir / "training/selected_checkpoint.pt", map_location="cpu", weights_only=True)
    selection = _read_json(run_dir / "training/selection.json"); _require(selection["checkpoint_sha256"] == _file_sha256(run_dir / "training/selected_checkpoint.pt") and selection["selected_epoch"] == payload["selected_epoch"], "selected generator binding changed")
    model = ClassConditionalUNet28(); model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


@_charged("reconstruction")
@torch.no_grad()
def run_reconstruction_panel(run_dir: Path, evaluator: SmallMnistCNN, device: torch.device, config: Mapping[str, Any] = FROZEN_CONFIG) -> dict[str, Any]:
    with np.load(run_dir / "controls/reconstruction_panel.npz", allow_pickle=False) as archive: panel = {key: archive[key] for key in archive.files}
    model, schedule, endpoints, records, started = _load_generator(run_dir, device, config), _schedule(config, device), [], [], time.perf_counter()
    for i, label in enumerate(panel["labels"]):
        for j, horizon in enumerate(panel["start_timesteps"]):
            state, target = _paired_reverse(torch.from_numpy(panel["starts"][i, j:j + 1]).to(device), torch.from_numpy(panel["x0"][i:i + 1]).to(device), int(label), int(horizon), int(panel["reverse_seeds"][i, j]), schedule, model)
            errors = torch.mean((state - target) ** 2, dim=(1, 2, 3)).cpu().numpy(); endpoints.append(state.cpu().numpy())
            records.append({"panel_index": i, "requested_label": int(label), "horizon": int(horizon), "zero_mse": float(errors[0]), "oracle_mse": float(errors[1]), "learned_mse": float(errors[2])})
            _resource_check(run_dir, device, inflight=time.perf_counter() - started)
    values = np.asarray(endpoints, dtype=np.float32)
    with np.load(run_dir / "controls/oracle_preflight_endpoints.npz", allow_pickle=False) as archive: preflight = archive["endpoints"]
    _require(np.array_equal(values[:, :2], preflight), "zero/oracle controls changed after training")
    converted = model_to_uint8(values.reshape(-1, 1, 28, 28)); labels = np.repeat(np.repeat(panel["labels"], len(panel["start_timesteps"])), 3)
    predictions = evaluate_image_classifier(evaluator, uint8_to_eval(converted), labels, device=device)["predictions"]
    for record, prediction in zip(records, np.asarray(predictions).reshape(-1, 3), strict=True): record.update({"zero_prediction": int(prediction[0]), "oracle_prediction": int(prediction[1]), "learned_prediction": int(prediction[2])})
    _write_csv(run_dir / "controls/reconstruction_metrics.csv", records); _write_npz(run_dir / "controls/reconstruction_trajectories.npz", starts=panel["starts"], endpoints=values)
    for row, name in enumerate(("zero", "oracle", "learned")): write_contact_sheet(run_dir / f"controls/images/{name}-endpoints.png", model_to_uint8(values[:, row]), columns=len(panel["start_timesteps"]))
    result = {"zero_oracle_hash_match": 1, "learned_mse_by_horizon": {str(int(t)): float(np.median([r["learned_mse"] for r in records if r["horizon"] == t])) for t in panel["start_timesteps"]}}
    _write_json(run_dir / "controls/reconstruction_summary.json", result)
    return result


@_charged("prior_generation")
@torch.no_grad()
def generate_prior_gallery(run_dir: Path, device: torch.device, config: Mapping[str, Any] = FROZEN_CONFIG) -> dict[str, np.ndarray]:
    model, schedule, sample = _load_generator(run_dir, device, config), _schedule(config, device), config["sampling"]
    starts_path, manifest_path, started = run_dir / "evaluation/prior_starts.npz", run_dir / "evaluation/sampling_manifest.csv", time.perf_counter(); batch_size = 10 * int(sample["per_class_per_batch"]); total = int(sample["batches"]) * batch_size
    calls = [0]
    def bounded(*args: Any) -> torch.Tensor: calls[0] += 1; _resource_check(run_dir, device, inflight=time.perf_counter() - started) if calls[0] % 50 == 0 else None; return model(*args)
    if not starts_path.is_file() or not manifest_path.is_file():
        starts, records = [], []
        for batch_id in range(int(sample["batches"])):
            starts.append(torch.randn((batch_size, 1, 28, 28), generator=torch.Generator(device=device).manual_seed(int(config["seeds"]["prior_start"]) + batch_id), device=device).cpu().numpy())
            for digit in range(10):
                for within in range(int(sample["per_class_per_batch"])):
                    index = batch_id * int(sample["per_class_per_batch"]) + within; records.append({"generation_order": len(records), "batch_id": batch_id, "requested_label": digit, "within_class_index": index, "start_seed": int(config["seeds"]["prior_start"]) + batch_id, "reverse_seed": int(config["seeds"]["prior_reverse"]) + batch_id})
        raw = np.concatenate(starts); order = np.asarray(sorted(range(total), key=lambda i: (records[i]["requested_label"], records[i]["within_class_index"])), dtype=np.int64); records = [records[int(i)] | {"output_index": j, "sample_id": _noise_id(raw[int(i)])} for j, i in enumerate(order)]
        labels = np.asarray([r["requested_label"] for r in records], dtype=np.int64); ids = np.asarray([r["sample_id"] for r in records]); _write_npz(starts_path, starts=raw[order], requested_labels=labels, sample_ids=ids); _write_csv(manifest_path, records)
    with np.load(starts_path, allow_pickle=False) as archive: starts_array, labels, ids = archive["starts"], archive["requested_labels"], archive["sample_ids"]
    finals_array = np.empty_like(starts_array); trajectory_array = np.empty((total, len(sample["anchors"]), 1, 28, 28), dtype=np.float32)
    with manifest_path.open(newline="", encoding="utf-8") as handle: records = list(csv.DictReader(handle))
    for batch_id in range(int(sample["batches"])):
        mask = np.asarray([int(row["batch_id"]) == batch_id for row in records]); initial = torch.from_numpy(starts_array[mask]).to(device); batch_labels = torch.from_numpy(labels[mask]).to(device)
        final, anchors = sample_reverse(bounded, batch_labels, initial, schedule, generator=torch.Generator(device=device).manual_seed(int(config["seeds"]["prior_reverse"]) + batch_id), anchor_steps=sample["anchors"])
        finals_array[mask] = final.cpu().numpy(); trajectory_array[mask] = torch.stack([anchors[int(a)] for a in sample["anchors"]], dim=1).cpu().numpy(); elapsed = time.perf_counter() - started; _resource_check(run_dir, device, inflight=elapsed * int(sample["batches"]) / (batch_id + 1))
    images = model_to_uint8(finals_array); validate_generated_batch(images, labels, ids)
    _write_npz(run_dir / "evaluation/prior_trajectories.npz", completed_steps=np.asarray(sample["anchors"], dtype=np.int64), states=trajectory_array, requested_labels=labels, sample_ids=ids)
    _write_npz(run_dir / "evaluation/samples_uint8.npz", images=images, requested_labels=labels, sample_ids=ids)
    _write_json(run_dir / "evaluation/trajectory_health.json", {str(anchor): {"mean": float(trajectory_array[:, j].mean()), "std": float(trajectory_array[:, j].std()), "minimum": float(trajectory_array[:, j].min()), "maximum": float(trajectory_array[:, j].max()), "finite": int(np.isfinite(trajectory_array[:, j]).all())} for j, anchor in enumerate(sample["anchors"])})
    _write_json(run_dir / "evaluation/GALLERY_READY.json", {"starts_sha256": _file_sha256(starts_path), "samples_sha256": _file_sha256(run_dir / "evaluation/samples_uint8.npz"), "trajectories_sha256": _file_sha256(run_dir / "evaluation/prior_trajectories.npz"), "count": len(images)})
    return {"images": images, "requested_labels": labels, "sample_ids": ids, "trajectories": trajectory_array}


def _load_gallery(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run_dir / "evaluation/samples_uint8.npz", allow_pickle=False) as samples, np.load(run_dir / "evaluation/prior_trajectories.npz", allow_pickle=False) as paths:
        return {"images": samples["images"], "requested_labels": samples["requested_labels"], "sample_ids": samples["sample_ids"], "trajectories": paths["states"]}


@_charged("terminal_scoring")
def open_test_and_score(run_dir: Path, arff: Path, train_u8: np.ndarray, evaluator: SmallMnistCNN, device: torch.device, config: Mapping[str, Any] = FROZEN_CONFIG) -> dict[str, Any]:
    gallery = _load_gallery(run_dir)
    frozen = ["training/selected_checkpoint.pt", "evaluator/selected_checkpoint.pt", "evaluation/prior_starts.npz", "evaluation/samples_uint8.npz", "evaluation/sampling_manifest.csv"]
    hashes = {relative: _file_sha256(run_dir / relative) for relative in frozen}
    _write_json(run_dir / "data/test_open_event.json", {"opened_at": _utc_now(), "frozen_hashes": hashes, "test_loader_called_after_freeze": 1})
    test_u8, test_y = load_test_mnist_terminal(arff)
    real = evaluate_image_classifier(evaluator, uint8_to_eval(test_u8), test_y, batch_size=int(config["evaluator"]["batch_size"]), device=device)
    real_record = {"accuracy": float(real["accuracy"]), "loss": float(real["loss"]), "gate_c_test_passed": int(float(real["accuracy"]) >= float(config["evaluator"]["minimum_accuracy"]))}; _write_json(run_dir / "evaluator/real_test_metrics.json", real_record)
    metrics = compute_generation_metrics(evaluator, gallery["images"], gallery["requested_labels"], gallery["sample_ids"], real_reference_images=test_u8, real_reference_labels=test_y, train_images=train_u8, test_images=test_u8, batch_size=int(config["evaluator"]["batch_size"]), device=device)
    metrics["terminal_evaluator"] = real_record; metrics["saturated_pixel_fraction"] = float(np.mean((gallery["images"] == 0) | (gallery["images"] == 255)))
    _write_json(run_dir / "evaluation/metrics.json", metrics)
    rows = [{"class": digit, "classifier_accuracy": metrics["classifier"]["per_class"][str(digit)]["accuracy"], "duplicate_pairs": metrics["duplicates"]["by_class"][str(digit)]["duplicate_pair_count"], "diversity_ratio": metrics["diversity"]["by_class"][str(digit)]["ratio"]} for digit in range(10)]
    _write_csv(run_dir / "evaluation/per_class_metrics.csv", rows)
    _write_json(run_dir / "evaluation/SCORING_READY.json", {"metrics_sha256": _file_sha256(run_dir / "evaluation/metrics.json"), "per_class_sha256": _file_sha256(run_dir / "evaluation/per_class_metrics.csv")})
    return metrics


@_charged("rendering")
def prepare_human_review(run_dir: Path, config: Mapping[str, Any] = FROZEN_CONFIG) -> dict[str, Any]:
    gallery = _load_gallery(run_dir); images, labels, ids, paths = gallery["images"], gallery["requested_labels"], gallery["sample_ids"], gallery["trajectories"]
    write_contact_sheet(run_dir / "evaluation/images/balanced-final.png", images, columns=16); write_contact_sheet(run_dir / "evaluation/images/class-3-final.png", images[labels == 3], columns=8)
    wanted = set(config["sampling"]["review_within_class"]); per_class = int(config["sampling"]["batches"]) * int(config["sampling"]["per_class_per_batch"]); review = np.asarray([i for i in range(len(ids)) if i % per_class in wanted], dtype=np.int64)
    for j, anchor in enumerate(config["sampling"]["anchors"]): write_contact_sheet(run_dir / f"evaluation/images/trajectory-step-{int(anchor):04d}.png", model_to_uint8(paths[review, j]), columns=8)
    bundle = write_blinded_review_bundle(run_dir / "review", images[review], labels[review], ids[review], seed=int(config["seeds"]["review"]), columns=8)
    _write_npy(run_dir / "review/review_indices.npy", review); sample_files = sorted((run_dir / "review/samples").glob("*.png")); _require(len(sample_files) == 10 * len(wanted), "review bundle is incomplete"); _write_json(run_dir / "review/READY.json", {"sample_count": len(review), "sample_ids_sha256": _array_sha256(ids[review]), "files": [p.name for p in sample_files]})
    return {"review_count": len(review), "template": str(Path(bundle["template"]).relative_to(run_dir))}


def _outcome(run_dir: Path, human: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    metrics, validation, test, ledger = _read_json(run_dir / "evaluation/metrics.json"), _read_json(run_dir / "evaluator/real_validation_metrics.json"), _read_json(run_dir / "evaluator/real_test_metrics.json"), _ledger(run_dir)
    gate_c = int(validation["gate_c_validation_passed"] and test["gate_c_test_passed"])
    values = {"classifier_accuracy": float(metrics["classifier"]["requested_label_accuracy"]), "human_agreement": float(human["human_requested_label_agreement"]), "human_recognizability": float(human["human_recognizability"]), "duplicate_pairs": int(metrics["duplicates"]["duplicate_pair_count"]), "diversity_ratio": float(metrics["diversity"]["aggregate_median_ratio"])}
    d = config["diagnostic"]; passed = int(gate_c and values["classifier_accuracy"] >= d["classifier_accuracy"] and values["human_agreement"] >= d["human_agreement"] and values["duplicate_pairs"] == d["duplicate_pairs"] and values["diversity_ratio"] >= d["diversity_ratio"]); gate_d = int(float(ledger["active_seconds"]) <= float(ledger["maximum_active_seconds"]) and float(ledger["peak_cuda_fraction"]) < float(ledger["maximum_cuda_fraction"]) and int(ledger["peak_storage_bytes"]) < int(ledger["maximum_storage_bytes"]))
    if not gate_c: route, action = "repair_evaluator", "preserve samples and repair only the evaluator/rendering path"
    elif values["human_recognizability"] >= 0.75 and values["human_agreement"] >= d["human_agreement"] and values["classifier_accuracy"] < d["classifier_accuracy"]: route, action = "audit_evaluator_numeric_transform", "humans recognize the requested digits; audit evaluator and numeric transforms"
    elif values["classifier_accuracy"] >= d["classifier_accuracy"] and values["human_recognizability"] < 0.5: route, action = "metric_misalignment", "treat the automated metric as gamed or misaligned and audit evaluator/rendering"
    elif passed: route, action = "freeze_conventional_benchmark", "freeze this benchmark and plan one materially different Eulerian formulation through the same image pipeline"
    elif values["human_recognizability"] < 0.5: route, action = "reference_ddpm_audit", "compare the unchanged recipe once against a recognized DDPM implementation or localize the shared training/normalization defect"
    elif values["duplicate_pairs"] or values["diversity_ratio"] < d["diversity_ratio"]: route, action = "bounded_diversity_correction", "permit at most one prespecified diversity/training correction"
    else: route, action = "one_specific_correction", "use saved artifacts to justify at most one specific correction; do not sweep"
    return {"schema": VERSION + "-outcome", "diagnostic_e_passed": int(passed and gate_d), "gate_a_passed": 1, "gate_b_passed": 1, "gate_c_passed": gate_c, "gate_d_passed": gate_d, "values": values, "route": route, "next_action": action, "automatic_launches": 0, "claim_scope": "one fixed exploratory class-conditional pixel-DDPM training seed and 160 prespecified Gaussian starts"}


def _report(run_dir: Path, outcome: Mapping[str, Any] | None) -> str:
    config, bindings, ledger = _read_json(run_dir / "config.json"), _read_json(run_dir / "source_bindings.json"), _ledger(run_dir)
    generator, evaluator, validation, test, oracle, reconstruction, metrics = (_read_json(run_dir / path) for path in ("training/selection.json", "evaluator/selection.json", "evaluator/real_validation_metrics.json", "evaluator/real_test_metrics.json", "controls/oracle_preflight.json", "controls/reconstruction_summary.json", "evaluation/metrics.json"))
    state = "awaiting_human_review" if outcome is None else "complete"
    lines = ["# Conventional pixel-DDPM calibration benchmark", "", "Primary mode: exploratory. Decision: can the fixed Gaussian pixel DDPM establish the common MNIST image benchmark? This is a calibration control, not an Eulerian replacement.", "", f"Status: `{state}`. Source revision/data/config hashes: `{bindings['git_revision']}`, `{bindings['data_sha256']}`, `{bindings['config_sha256']}`; direct source hashes: `{bindings['source_files']}`. Fixed roles: train 0:55000, validation 55000:60000, terminal test 60000:70000.", f"Recipe: `{config['model']}`, `{config['schedule']}`, `{config['training']}`. Exact command: `command.txt`.", f"Gates A/B: pass (oracle max MSE `{oracle['maximum_oracle_mse']}`). Gate C validation/test values: `{validation}` / `{test}`. Gate D active/cap seconds: `{ledger['active_seconds']}` / `{ledger['maximum_active_seconds']}`.", f"Selected generator/evaluator epochs and hashes: `{generator}` / `{evaluator}`. Mechanism diagnostics: reconstruction `{reconstruction}`, saturation `{metrics['saturated_pixel_fraction']}`. Controls: `controls/reconstruction_metrics.csv`; outputs including failures: `evaluation/samples_uint8.npz`, `evaluation/images/balanced-final.png`."]
    if recovery := bindings.get("verifier_recovery"): lines.append(f"Verifier-only terminal recovery: execution source hashes `{recovery['execution_source_files']}`; verification source hashes `{recovery['verification_source_files']}`; receipt `{recovery['receipt_path']}`. No CUDA was re-executed.")
    if outcome is None: lines.append("Human review is pending at `review/human_review_template.csv`; no scientific route or outcome is assigned yet.")
    else: lines.extend([f"Primary results: `{outcome['values']}`. Diagnostic E: `{outcome['diagnostic_e_passed']}`.", f"Route: `{outcome['route']}`. Required next action: {outcome['next_action']}."])
    lines.extend([f"Resources/restarts: `{ledger}`.", "Claim boundary: one authenticated split, fixed recipe, one model seed, and 160 samples; no confirmatory, population, superiority, or Eulerian-generator claim."])
    return "\n\n".join(lines) + "\n"


def finalize_run(run_dir: Path, config: Mapping[str, Any] = FROZEN_CONFIG, human: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    started = time.perf_counter(); _resource_check(run_dir, torch.device("cpu"))
    outcome = _outcome(run_dir, human, config) if human is not None else None
    _charge(run_dir, "terminalization", time.perf_counter() - started); _resource_check(run_dir, torch.device("cpu"))
    if human is not None: outcome = _outcome(run_dir, human, config)
    if outcome is not None: _write_json(run_dir / "outcome.json", outcome)
    _replace(run_dir / "REPORT.md", lambda p: p.write_text(_report(run_dir, outcome), encoding="utf-8"))
    _status(run_dir, "complete" if outcome else "awaiting_human_review", resumable=False)
    _refresh_manifest(run_dir)
    return outcome


def record_human_review(run_dir: Path, answers: Path, reviewer: str, confirm_manual_review: bool, config: Mapping[str, Any] = FROZEN_CONFIG) -> dict[str, Any]:
    run_dir, answers = Path(run_dir).resolve(), Path(answers).resolve(); relative = answers.relative_to(run_dir).as_posix() if answers.is_relative_to(run_dir) else None; had_manifest = (run_dir / "artifact_manifest.json").is_file()
    state = _read_json(run_dir / "status.json")["state"]
    if state == "complete" and (run_dir / "review/human_review.json").is_file(): _require(had_manifest, "completed run manifest is missing"); verify_run(run_dir); return _read_json(run_dir / "outcome.json")
    if not had_manifest: _refresh_manifest(run_dir)
    _verify_manifest(run_dir, {relative} if had_manifest and relative and relative != "review/human_review_template.csv" else None)
    _require(state == "awaiting_human_review", "run is not awaiting review")
    human = score_human_review(answers, run_dir / "review/review_key.json", reviewer=reviewer, confirm_manual_review=confirm_manual_review)
    (run_dir / "artifact_manifest.json").unlink()
    destination = run_dir / "review/human_review_answers.csv"
    if answers != destination: _replace(destination, lambda p: p.write_bytes(answers.read_bytes()))
    _write_json(run_dir / "review/human_review.json", human)
    return finalize_run(run_dir, config, human) or {}


def verify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve(); before = _tree_digest(run_dir); manifest = _verify_manifest(run_dir); config = _read_json(run_dir / "config.json")
    if config.get("schema") == VERSION + "-smoke":
        payload = torch.load(run_dir / "smoke.pt", map_location="cpu", weights_only=True); model = ClassConditionalUNet28(); model.load_state_dict(payload["state_dict"])
        with np.load(run_dir / "samples.npz", allow_pickle=False) as archive: _require(archive["images"].shape == (10, 28, 28), "smoke samples changed")
    else:
        _require(config == FROZEN_CONFIG, "frozen config changed")
        bindings = _read_json(run_dir / "source_bindings.json"); repository, arff, sources, recovery = Path(bindings["repository_root"]), Path(bindings["arff"]), _source_hashes(Path(bindings["repository_root"])), bindings.get("verifier_recovery"); receipt = _read_json(run_dir / recovery["receipt_path"]) if recovery else None; original_bindings = _read_json(run_dir / "recovery/original_source_bindings.json") if recovery else None; inventory = _read_json(run_dir / "recovery/immutable_inventory.json") if recovery else None
        _require(bindings["source_files"] == sources and bindings["git_revision"] == _git_revision(repository) and bindings["data_sha256"] == _file_sha256(arff) and bindings["config_sha256"] == _semantic_sha256(config) and (not recovery or (recovery.get("schema") == "pixel-ddpm-verifier-recovery-v1" and recovery.get("receipt_path") == "recovery/recovery_receipt.json" and recovery.get("verification_source_files") == sources and recovery.get("execution_source_files") == original_bindings.get("source_files") and set(recovery["execution_source_files"]) == set(sources) and receipt.get("schema") == "pixel-ddpm-terminal-recovery-v1" and receipt.get("cuda_reexecuted") == 0 and receipt.get("immutable_artifact_count") == inventory.get("artifact_count") == len(inventory.get("artifacts", [])) == 91 and receipt.get("execution_runner_sha256") == recovery["execution_source_files"][DIRECT_SOURCE_FILES[0]] and receipt.get("verification_runner_sha256") == sources[DIRECT_SOURCE_FILES[0]] and receipt.get("execution_runner_archive_path") == "recovery/execution_runner.py" and receipt.get("execution_runner_archive_sha256") == receipt.get("execution_runner_sha256") == _file_sha256(run_dir / "recovery/execution_runner.py") and receipt.get("immutable_inventory_path") == "recovery/immutable_inventory.json" and receipt.get("immutable_inventory_sha256") == _file_sha256(run_dir / "recovery/immutable_inventory.json") and inventory.get("schema") == "pixel-ddpm-immutable-inventory-v1" and inventory.get("original_manifest_sha256") == receipt.get("original_manifest_sha256") == _file_sha256(run_dir / "recovery/original_artifact_manifest.json") and all(receipt.get("original_authority_sha256", {}).get(path) == _file_sha256(run_dir / path) for path in ("recovery/original_artifact_manifest.json", "recovery/original_source_bindings.json", "recovery/original_status.json", "recovery/original_failure.json", "recovery/original_REPORT.md", "recovery/original_resource_ledger.json")) and all((run_dir / ("recovery/original_failure.json" if row["path"] == "failure.json" else row["path"])).stat().st_size == row["size"] and _file_sha256(run_dir / ("recovery/original_failure.json" if row["path"] == "failure.json" else row["path"])) == row["sha256"] for row in inventory["artifacts"]))), "source/config/data or verifier-recovery binding changed")
        split = _read_json(run_dir / "data/split.json")
        for role, (start, stop) in {"train": (0, 55_000), "validation": (55_000, 60_000), "test": (60_000, 70_000)}.items():
            values = np.load(run_dir / f"data/{role}_indices.npy", allow_pickle=False); _require(np.array_equal(values, np.arange(start, stop)) and split[role]["sha256"] == _array_sha256(values), "split binding changed")
        _verify_npz(run_dir / "controls/schedule.npz", _schedule_arrays(config), "schedule changed")
        oracle_record = _read_json(run_dir / "controls/oracle_preflight.json"); _require(oracle_record.get("passed") == 1, "saved Gate B result is not valid")
        train_u8, _, validation_u8, validation_y = load_train_validation_mnist(arff); _verify_npz(run_dir / "controls/validation_noise_bank.npz", _make_validation_bank(validation_u8, validation_y, config), "validation bank changed"); expected_panel = _panel(validation_u8, validation_y, config); _verify_npz(run_dir / "controls/reconstruction_panel.npz", expected_panel, "reconstruction panel changed")
        with np.load(run_dir / "controls/oracle_preflight_endpoints.npz", allow_pickle=False) as oracle_paths, np.load(run_dir / "controls/reconstruction_trajectories.npz", allow_pickle=False) as recon: oracle_endpoints, reconstruction_endpoints, reconstruction_starts = oracle_paths["endpoints"], recon["endpoints"], recon["starts"]
        cases = len(expected_panel["labels"]) * len(expected_panel["start_timesteps"]); targets = np.repeat(expected_panel["x0"], len(expected_panel["start_timesteps"]), axis=0); oracle_errors = torch.mean((torch.from_numpy(oracle_endpoints) - torch.from_numpy(targets[:, None])) ** 2, dim=(2, 3, 4)).numpy(); reductions = np.asarray([1.0 - float(row[1]) / max(float(row[0]), np.finfo(float).tiny) for row in oracle_errors]); horizon_medians = {str(int(t)): float(np.median(reductions[j::len(expected_panel["start_timesteps"])])) for j, t in enumerate(expected_panel["start_timesteps"])}; _require(oracle_endpoints.shape == (cases, 2, 1, 28, 28) and reconstruction_endpoints.shape == (cases, 3, 1, 28, 28) and np.array_equal(reconstruction_starts, expected_panel["starts"]) and np.array_equal(reconstruction_endpoints[:, :2], oracle_endpoints) and np.isfinite(reconstruction_endpoints).all() and float(oracle_errors[:, 1].max()) <= float(config["reconstruction"]["oracle_max_mse"]) and min(horizon_medians.values()) >= float(config["reconstruction"]["oracle_min_reduction"]) and math.isclose(float(oracle_record["maximum_oracle_mse"]), float(oracle_errors[:, 1].max()), rel_tol=1e-5, abs_tol=1e-8) and all(math.isclose(float(oracle_record["median_reduction_by_horizon"][key]), value, rel_tol=1e-5, abs_tol=1e-8) for key, value in horizon_medians.items()), "reconstruction/oracle arrays changed")
        evaluator_payload = torch.load(run_dir / "evaluator/selected_checkpoint.pt", map_location="cpu", weights_only=True); evaluator_selection = _read_json(run_dir / "evaluator/selection.json"); _require(evaluator_selection["checkpoint_sha256"] == _file_sha256(run_dir / "evaluator/selected_checkpoint.pt") and evaluator_selection["selected_epoch"] == evaluator_payload["selected_epoch"], "evaluator checkpoint binding changed"); evaluator = SmallMnistCNN(); evaluator.load_state_dict(evaluator_payload["state_dict"]); _load_generator(run_dir, torch.device("cpu"), config); latest = torch.load(run_dir / "training/latest.pt", map_location="cpu", weights_only=True)
        validation_eval = evaluate_image_classifier(evaluator, uint8_to_eval(validation_u8), validation_y, batch_size=int(config["evaluator"]["batch_size"]), device="cpu"); expected_validation = {"accuracy": float(validation_eval["accuracy"]), "loss": float(validation_eval["loss"]), "gate_c_validation_passed": int(float(validation_eval["accuracy"]) >= float(config["evaluator"]["minimum_accuracy"]))}; saved_validation = _read_json(run_dir / "evaluator/real_validation_metrics.json"); _require(saved_validation["accuracy"] == expected_validation["accuracy"] and saved_validation["gate_c_validation_passed"] == expected_validation["gate_c_validation_passed"] and evaluator_selection["restored_validation_accuracy"] == expected_validation["accuracy"] and evaluator_selection["gate_c_validation_passed"] == expected_validation["gate_c_validation_passed"] and math.isclose(saved_validation["loss"], expected_validation["loss"], rel_tol=2e-4, abs_tol=2e-5) and math.isclose(evaluator_selection["restored_validation_loss"], expected_validation["loss"], rel_tol=2e-4, abs_tol=2e-5), "saved validation metrics changed")
        selection = _read_json(run_dir / "training/selection.json"); eligible = [row for row in latest["history"] if row["eligible"]]; best = min(eligible, key=lambda row: (row["ema_validation_mse"], row["epoch"])); _require(latest["completed_epoch"] == config["training"]["epochs"] and latest["best_epoch"] == selection["selected_epoch"] == best["epoch"] and latest["best_mse"] == selection["validation_mse"] == best["ema_validation_mse"], "generator selection/latest binding changed")
        gallery = _load_gallery(run_dir); validate_generated_batch(gallery["images"], gallery["requested_labels"], gallery["sample_ids"])
        total = int(config["sampling"]["batches"]) * int(config["sampling"]["per_class_per_batch"]) * 10; per_class, per_batch = total // 10, int(config["sampling"]["per_class_per_batch"]); expected_labels = np.repeat(np.arange(10), per_class)
        with np.load(run_dir / "evaluation/prior_starts.npz", allow_pickle=False) as starts, np.load(run_dir / "evaluation/prior_trajectories.npz", allow_pickle=False) as paths:
            starts_array, trajectories = starts["starts"], paths["states"]; _require(starts_array.shape == (total, 1, 28, 28) and trajectories.shape == (total, len(config["sampling"]["anchors"]), 1, 28, 28) and gallery["images"].shape == (total, 28, 28) and np.array_equal(paths["completed_steps"], config["sampling"]["anchors"]), "prior artifact shape/anchors changed")
            _require(np.array_equal(gallery["requested_labels"], expected_labels) and np.array_equal(starts["requested_labels"], expected_labels) and np.array_equal(paths["requested_labels"], expected_labels) and np.array_equal(starts["sample_ids"], gallery["sample_ids"]) and np.array_equal(paths["sample_ids"], gallery["sample_ids"]) and len(set(gallery["sample_ids"].tolist())) == total and all(str(value) == _noise_id(starts_array[i]) for i, value in enumerate(gallery["sample_ids"])) and np.array_equal(trajectories[:, 0], starts_array) and np.array_equal(model_to_uint8(trajectories[:, -1]), gallery["images"]), "prior labels/sample IDs or trajectory endpoints changed")
        with (run_dir / "evaluation/sampling_manifest.csv").open(newline="", encoding="utf-8") as handle: sampling_rows = list(csv.DictReader(handle))
        expected_rows = [{"output_index": i, "sample_id": _noise_id(starts_array[i]), "requested_label": i // per_class, "within_class_index": i % per_class, "batch_id": (i % per_class) // per_batch, "generation_order": ((i % per_class) // per_batch) * 10 * per_batch + (i // per_class) * per_batch + (i % per_batch), "start_seed": int(config["seeds"]["prior_start"]) + (i % per_class) // per_batch, "reverse_seed": int(config["seeds"]["prior_reverse"]) + (i % per_class) // per_batch} for i in range(total)]
        _require(len(sampling_rows) == total and all(all(str(row[key]) == str(value) for key, value in expected_rows[i].items()) for i, row in enumerate(sampling_rows)), "sampling manifest alignment changed")
        gallery_ready, scoring_ready = _read_json(run_dir / "evaluation/GALLERY_READY.json"), _read_json(run_dir / "evaluation/SCORING_READY.json"); _require(gallery_ready == {"starts_sha256": _file_sha256(run_dir / "evaluation/prior_starts.npz"), "samples_sha256": _file_sha256(run_dir / "evaluation/samples_uint8.npz"), "trajectories_sha256": _file_sha256(run_dir / "evaluation/prior_trajectories.npz"), "count": total} and scoring_ready == {"metrics_sha256": _file_sha256(run_dir / "evaluation/metrics.json"), "per_class_sha256": _file_sha256(run_dir / "evaluation/per_class_metrics.csv")}, "stage closure binding changed")
        event = _read_json(run_dir / "data/test_open_event.json"); frozen = {"training/selected_checkpoint.pt", "evaluator/selected_checkpoint.pt", "evaluation/prior_starts.npz", "evaluation/samples_uint8.npz", "evaluation/sampling_manifest.csv"}; _require(set(event["frozen_hashes"]) == frozen and event.get("test_loader_called_after_freeze") == 1 and all(_file_sha256(run_dir / path) == digest for path, digest in event["frozen_hashes"].items()), "test-open frozen hash changed")
        test_u8, test_y = load_test_mnist_terminal(arff); observed = _read_json(run_dir / "evaluation/metrics.json")
        recomputed = compute_generation_metrics(evaluator, gallery["images"], gallery["requested_labels"], gallery["sample_ids"], real_reference_images=test_u8, real_reference_labels=test_y, train_images=train_u8, test_images=test_u8, device="cpu")
        real = evaluate_image_classifier(evaluator, uint8_to_eval(test_u8), test_y, batch_size=int(config["evaluator"]["batch_size"]), device="cpu"); expected_test = {"accuracy": float(real["accuracy"]), "loss": float(real["loss"]), "gate_c_test_passed": int(float(real["accuracy"]) >= float(config["evaluator"]["minimum_accuracy"]))}; saved_test = _read_json(run_dir / "evaluator/real_test_metrics.json"); _require(observed["duplicates"] == _jsonable(recomputed["duplicates"]) and observed["diversity"] == _jsonable(recomputed["diversity"]) and observed["exact_reference_match_count"] == recomputed["exact_reference_match_count"] and observed["classifier"]["requested_label_accuracy"] == recomputed["classifier"]["requested_label_accuracy"] and observed["classifier"]["per_class"] == _jsonable(recomputed["classifier"]["per_class"]) and saved_test["accuracy"] == expected_test["accuracy"] and saved_test["gate_c_test_passed"] == expected_test["gate_c_test_passed"] and math.isclose(saved_test["loss"], expected_test["loss"], rel_tol=2e-4, abs_tol=2e-5), "saved sample metrics changed")
        review_indices = np.load(run_dir / "review/review_indices.npy", allow_pickle=False); ready = _read_json(run_dir / "review/READY.json"); wanted = set(config["sampling"]["review_within_class"]); expected_review = np.asarray([i for i in range(total) if i % per_class in wanted]); key_entries = _read_json(run_dir / "review/review_key.json")["entries"]; review_order = np.random.default_rng(int(config["seeds"]["review"])).permutation(len(review_indices)); ordered_review = review_indices[review_order]
        _require(np.array_equal(review_indices, expected_review) and len(review_indices) == ready["sample_count"] and ready["sample_ids_sha256"] == _array_sha256(gallery["sample_ids"][review_indices]) and ready["files"] == [f"sample-{i:03d}.png" for i in range(len(review_indices))] and [(entry["review_order"], entry["sample_id"], entry["source_sample_id"], entry["requested_label"]) for entry in key_entries] == [(i, f"blind-{i:03d}", str(gallery["sample_ids"][index]), int(gallery["requested_labels"][index])) for i, index in enumerate(ordered_review)], "review image/key membership changed")
        expected_pngs = {f"controls/images/{name}-endpoints.png": _contact_array(model_to_uint8(reconstruction_endpoints[:, row]), len(config["reconstruction"]["start_timesteps"])) for row, name in enumerate(("zero", "oracle", "learned"))} | {"evaluation/images/balanced-final.png": _contact_array(gallery["images"], 16), "evaluation/images/class-3-final.png": _contact_array(gallery["images"][gallery["requested_labels"] == 3], 8), "review/blinded-contact-sheet.png": _contact_array(gallery["images"][ordered_review], 8, [f"blind-{i:03d}" for i in range(len(ordered_review))])} | {f"evaluation/images/trajectory-step-{int(anchor):04d}.png": _contact_array(model_to_uint8(trajectories[review_indices, j]), 8) for j, anchor in enumerate(config["sampling"]["anchors"])} | {f"review/samples/sample-{i:03d}.png": gallery["images"][index] for i, index in enumerate(ordered_review)}
        _require({path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*.png")} == set(expected_pngs), "PNG inventory changed")
        for relative, pixels in expected_pngs.items():
            with Image.open(run_dir / relative) as opened: _require(opened.mode == "L" and np.array_equal(np.asarray(opened), pixels), f"PNG pixels changed: {relative}")
        status = _read_json(run_dir / "status.json"); outcome_path = run_dir / "outcome.json"
        if status["state"] == "complete":
            human = _read_json(run_dir / "review/human_review.json"); recomputed = score_human_review(run_dir / "review/human_review_answers.csv", run_dir / "review/review_key.json", reviewer=human["reviewer"], confirm_manual_review=True, timestamp=human["recorded_at"])
            expected = _outcome(run_dir, recomputed, config); _require(human == recomputed and _read_json(outcome_path) == expected and (run_dir / "REPORT.md").read_text() == _report(run_dir, expected), "final review/outcome/report changed")
        else: _require(status["state"] == "awaiting_human_review" and not outcome_path.exists() and (run_dir / "REPORT.md").read_text() == _report(run_dir, None), "awaiting-review state changed")
    after = _tree_digest(run_dir); _require(before == after, "read-only verification mutated the run")
    return {"passed": 1, "tree_digest": before, "artifact_count": manifest["artifact_count"]}


def _smoke(output_dir: Path, device: str) -> int:
    output_dir = Path(output_dir).resolve(); _require(not output_dir.exists(), "smoke output already exists"); output_dir.mkdir(parents=True)
    target = torch.device(device); torch.manual_seed(7); model = ClassConditionalUNet28().to(target); schedule = make_linear_ddpm_schedule(4, 1e-4, 2e-2, device=target); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(2):
        x0 = torch.zeros((2, 1, 28, 28), device=target); labels = torch.tensor([0, 1], device=target); t = torch.tensor([0, 3], device=target); noise = torch.randn_like(x0); optimizer.zero_grad(); loss = epsilon_prediction_loss(model, x0, t, labels, noise, schedule); loss.backward(); optimizer.step()
    labels = torch.arange(10, device=target); initial = torch.randn((10, 1, 28, 28), generator=torch.Generator(device=target).manual_seed(8), device=target); final, _ = sample_reverse(model.eval(), labels, initial, schedule, generator=torch.Generator(device=target).manual_seed(9), anchor_steps=(0, 4))
    images = model_to_uint8(final.cpu().numpy()); _write_json(output_dir / "config.json", {"schema": VERSION + "-smoke", "steps": 4, "updates": 2}); _write_torch(output_dir / "smoke.pt", {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}); _write_npz(output_dir / "samples.npz", images=images); write_contact_sheet(output_dir / "samples.png", images, columns=5); _replace(output_dir / "REPORT.md", lambda p: p.write_text("Synthetic four-step/two-update CPU smoke; no scientific claim.\n")); _write_json(output_dir / "status.json", {"state": "complete"}); _refresh_manifest(output_dir); verify_run(output_dir)
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.resume:
        run_dir = Path(args.run_dir).resolve(); bindings = _read_json(run_dir / "source_bindings.json")
        repository, arff = Path(bindings["repository_root"]), Path(bindings["arff"])
    else:
        _require(args.repository_root and args.arff and args.runs_root, "new run requires repository root, ARFF, and runs root")
        repository, arff = Path(args.repository_root), Path(args.arff); run_dir = Path(args.runs_root).resolve() / args.run_name
    run_dir = initialize_run(repository, arff, run_dir, device=args.device, maximum_active_seconds=args.maximum_active_seconds, maximum_cuda_fraction=args.maximum_cuda_fraction, maximum_storage_mib=args.maximum_storage_mib, approval_reference=args.approval_reference, resume=args.resume)
    terminal_state = _read_json(run_dir / "status.json")["state"]; machine_closed = all((run_dir / path).is_file() for path in ("evaluation/GALLERY_READY.json", "evaluation/SCORING_READY.json", "review/READY.json"))
    if terminal_state in {"awaiting_human_review", "complete"} or machine_closed:
        _require(machine_closed, "terminal stage closure is incomplete")
        if not (run_dir / "artifact_manifest.json").is_file(): finalize_run(run_dir, human=_read_json(run_dir / "review/human_review.json") if (run_dir / "review/human_review.json").is_file() else None)
        verify_run(run_dir); return 0
    device = torch.device(args.device); _require(device.type == "cuda" and torch.cuda.is_available(), "production run requires an available CUDA device")
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if (run_dir / "status.json").is_file() and _read_json(run_dir / "status.json")["state"] in {"awaiting_human_review", "complete"}: verify_run(run_dir); return 0
    _status(run_dir, "running", resumable=True)
    try:
        train_u8, train_y, validation_u8, validation_y = load_train_validation_mnist(arff)
        if not (run_dir / "controls/oracle_preflight.json").is_file(): run_oracle_preflight(run_dir, validation_u8, validation_y, device)
        _require(_read_json(run_dir / "controls/oracle_preflight.json").get("passed") == 1, "Gate B failed; learner training is forbidden"); _resource_check(run_dir, device, reserve=900.0)
        evaluator, _ = train_or_load_evaluator(run_dir, train_u8, train_y, validation_u8, validation_y, device)
        train_or_resume_generator(run_dir, train_u8, train_y, validation_u8, validation_y, device)
        if not (run_dir / "controls/reconstruction_summary.json").is_file(): run_reconstruction_panel(run_dir, evaluator, device)
        if not (run_dir / "evaluation/GALLERY_READY.json").is_file(): generate_prior_gallery(run_dir, device)
        else:
            ready = _read_json(run_dir / "evaluation/GALLERY_READY.json"); _require(ready["starts_sha256"] == _file_sha256(run_dir / "evaluation/prior_starts.npz") and ready["samples_sha256"] == _file_sha256(run_dir / "evaluation/samples_uint8.npz") and ready["trajectories_sha256"] == _file_sha256(run_dir / "evaluation/prior_trajectories.npz"), "gallery closure changed")
        if not (run_dir / "evaluation/SCORING_READY.json").is_file(): open_test_and_score(run_dir, arff, train_u8, evaluator, device)
        else:
            ready = _read_json(run_dir / "evaluation/SCORING_READY.json"); _require(ready["metrics_sha256"] == _file_sha256(run_dir / "evaluation/metrics.json") and ready["per_class_sha256"] == _file_sha256(run_dir / "evaluation/per_class_metrics.csv"), "scoring closure changed")
        if not (run_dir / "review/READY.json").is_file(): prepare_human_review(run_dir)
        finalize_run(run_dir); verify_run(run_dir); return 0
    except ResourcePause as error:
        _status(run_dir, "resource_paused", resumable=True, error=str(error)); _write_json(run_dir / "failure.json", {"kind": "resource_pause", "message": str(error), "at": _utc_now()}); _refresh_manifest(run_dir); return 2
    except BaseException as error:
        resumable = not ((run_dir / "controls/oracle_preflight.json").is_file() and _read_json(run_dir / "controls/oracle_preflight.json").get("passed") == 0); _status(run_dir, "failed", resumable=resumable, error=str(error)); _write_json(run_dir / "failure.json", {"kind": type(error).__name__, "message": str(error), "at": _utc_now()}); _refresh_manifest(run_dir); raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed exploratory conventional pixel-DDPM calibration benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke"); smoke.add_argument("--output-dir", required=True); smoke.add_argument("--device", default="cpu")
    run = commands.add_parser("run")
    run.add_argument("--repository-root"); run.add_argument("--arff"); run.add_argument("--runs-root"); run.add_argument("--run-name", default=VERSION); run.add_argument("--run-dir"); run.add_argument("--resume", action="store_true"); run.add_argument("--device", default="cuda:0")
    run.add_argument("--maximum-active-seconds", type=float, default=7200.0); run.add_argument("--maximum-cuda-fraction", type=float, default=0.75); run.add_argument("--maximum-storage-mib", type=float, default=500.0); run.add_argument("--approval-reference", required=True)
    review = commands.add_parser("record-review"); review.add_argument("--run-dir", required=True); review.add_argument("--answers", required=True); review.add_argument("--reviewer", required=True); review.add_argument("--confirm-manual-review", action="store_true")
    verify = commands.add_parser("verify"); verify.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke": return _smoke(Path(args.output_dir), args.device)
    if args.command == "verify": print(json.dumps(verify_run(Path(args.run_dir)), sort_keys=True)); return 0
    if args.command == "record-review": record_human_review(Path(args.run_dir), Path(args.answers), args.reviewer, args.confirm_manual_review); return 0
    _require(not args.resume or args.run_dir, "--resume requires --run-dir")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
