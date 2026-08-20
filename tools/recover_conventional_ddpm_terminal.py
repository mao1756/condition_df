from __future__ import annotations

"""One-shot CPU-only recovery of the audited conventional-DDPM terminal run."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import mnist.diag_d0_conventional_ddpm_baseline as runner


SOURCE_NAME = "pixel-ddpm-calibration-v1-retry1"
OUTPUT_NAME = "pixel-ddpm-calibration-v1-cpu-recovered"
RUNNER_PATH = "mnist/diag_d0_conventional_ddpm_baseline.py"
EXPECTED_TREE_SHA256 = "632db7179660246b8b2bd4de5f20f8c368ab112343499e8de24a34ae38086f20"
EXPECTED_MANIFEST_SHA256 = "d78ec0b111084073eec6cdc5e8cb70b7d6906bf9fdeae130468f118703c9aa77"
EXPECTED_MANIFEST_COUNT = 95
EXPECTED_MANIFEST_BYTES = 35_909_346
EXPECTED_EXECUTION_RUNNER_SHA256 = "278a9f6f18ad55460b9ea9f968f1bfa3463ffbeed307537deb6e26b23bd99e67"
EXPECTED_VERIFICATION_RUNNER_SHA256 = "343f0b405b4d6739ad39719dfceef1c6ed8393d1541c14c8f9a90205b442ca30"
EXPECTED_EXECUTION_SOURCES = {
    "core/__init__.py": "41f5c6cccd60bb014bb410b3042b4c59a2dca6282d48534f4b47af55d3d916bf",
    "core/conditioning_utils.py": "5538702bb561ebabc437c0e699f81efba8e5dc1de79def636d01a3971540a692",
    "core/wasserstein_conditioning_algorithms.py": "15abea52014dfb2b8b0196f2a2bcad8c007f30e31da88afa7b8ede1f04436514",
    "mnist/__init__.py": "1afbf919b879fc8c499db24009ce92e92ee03b198cfb427830a18df37df86ce4",
    "mnist/conditioned_diffusion.py": "96906c6c1cf7fed4de191e56d6861621446e65b6952171cbf6fa556303450892",
    RUNNER_PATH: EXPECTED_EXECUTION_RUNNER_SHA256,
    "mnist/mnist_generation_benchmark.py": "2ebf13e37e03646222b8decde91034f10dff564fdc0d3f7a967af789ef3cbfd6",
    "mnist/pixel_ddpm.py": "30d9c7497ac008e6d60f3d91860d59b29e77e4952df80c9a41699ae2727e5381",
    "mnist/weighted_point_cloud.py": "b70db19c8adbaf7cd89818a61a7dc8b167ec83e013911682702161c7e28fca7d",
}
EXPECTED_AUTHORITY_SHA256 = {
    "recovery/original_artifact_manifest.json": EXPECTED_MANIFEST_SHA256,
    "recovery/original_source_bindings.json": "5cd8479193fb378c3acf51c3ef019e1f3f9a7c6c8e3eaa7412bfc29298d9c50a",
    "recovery/original_status.json": "a5d33c4e290159b07c2be6d761773c49fe060bc121474c1d9b9b396db82ccb59",
    "recovery/original_failure.json": "fc76dd97dbbce64351411e0767dc473bdca59217586ac4b80ab7bb22be433eb1",
    "recovery/original_REPORT.md": "5db1a65d7246b1e13b8c32ce28c83680b345f5c868f109170cdef648d30d4d24",
    "recovery/original_resource_ledger.json": "d3844b4529016ebabeee901d23ea62a5a65a17c7c83eefa6deeb8803a08cf720",
}
MUTABLE_ORIGINALS = {"source_bindings.json", "status.json", "REPORT.md", "resource_ledger.json"}


class RecoveryError(RuntimeError):
    pass


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _validate_source(source: Path, archive: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(source.name == SOURCE_NAME and source.is_dir(), "source is not the audited retry1 run")
    _require(archive.is_file() and not archive.is_symlink(), "execution-source archive is missing or linked")
    _require(runner._file_sha256(archive) == EXPECTED_EXECUTION_RUNNER_SHA256, "execution-source archive hash changed")
    _require(runner._file_sha256(source / "artifact_manifest.json") == EXPECTED_MANIFEST_SHA256, "original manifest hash changed")
    _require(runner._tree_digest(source) == EXPECTED_TREE_SHA256, "original run tree changed")
    manifest = runner._verify_manifest(source)
    _require(manifest["artifact_count"] == EXPECTED_MANIFEST_COUNT and manifest["artifact_bytes"] == EXPECTED_MANIFEST_BYTES, "original manifest totals changed")
    bindings = _read(source / "source_bindings.json")
    _require(bindings.get("source_files") == EXPECTED_EXECUTION_SOURCES and "verifier_recovery" not in bindings, "execution source binding changed")
    status, failure, ledger = (_read(source / name) for name in ("status.json", "failure.json", "resource_ledger.json"))
    _require(status.get("state") == "failed" and status.get("resumable") == 1 and status.get("error") == "saved validation metrics changed", "source terminal status is not the audited false positive")
    _require(failure.get("kind") == "DDPMRunError" and failure.get("message") == "saved validation metrics changed", "source failure authority changed")
    _require(runner._file_sha256(source / "resource_ledger.json") == EXPECTED_AUTHORITY_SHA256["recovery/original_resource_ledger.json"], "source resource ledger changed")
    roles = {row["role"] for row in ledger["events"]}
    _require(not any(row["failed"] for row in ledger["events"]) and {"generator_epoch_40", "reconstruction", "prior_generation", "terminal_scoring", "rendering", "terminalization"} <= roles, "machine stages did not close cleanly")
    _require(float(ledger["active_seconds"]) < float(ledger["maximum_active_seconds"]), "source active-time cap failed")
    gallery, scoring, review = (_read(source / name) for name in ("evaluation/GALLERY_READY.json", "evaluation/SCORING_READY.json", "review/READY.json"))
    _require(gallery == {"starts_sha256": runner._file_sha256(source / "evaluation/prior_starts.npz"), "samples_sha256": runner._file_sha256(source / "evaluation/samples_uint8.npz"), "trajectories_sha256": runner._file_sha256(source / "evaluation/prior_trajectories.npz"), "count": 160}, "gallery closure changed")
    _require(scoring == {"metrics_sha256": runner._file_sha256(source / "evaluation/metrics.json"), "per_class_sha256": runner._file_sha256(source / "evaluation/per_class_metrics.csv")}, "scoring closure changed")
    _require(review.get("sample_count") == 40 and len(review.get("files", [])) == 40 and all((source / "review/samples" / name).is_file() for name in review["files"]), "review closure changed")
    event = _read(source / "data/test_open_event.json")
    _require(event.get("test_loader_called_after_freeze") == 1 and len(event.get("frozen_hashes", {})) == 5 and all(runner._file_sha256(source / name) == digest for name, digest in event["frozen_hashes"].items()), "test-open firewall changed")
    latest = runner.torch.load(source / "training/latest.pt", map_location="cpu", weights_only=True)
    selection = _read(source / "training/selection.json")
    _require(latest.get("completed_epoch") == selection.get("completed_epochs") == 40 and latest.get("best_epoch") == selection.get("selected_epoch") == 40, "generator completion changed")
    _require(not (source / "outcome.json").exists() and not (source / "review/human_review.json").exists(), "source unexpectedly contains human outcome evidence")
    return manifest, bindings


def _write_capsule(clone: Path, archive: Path, source: Path, output: Path, manifest: dict[str, Any], execution_bindings: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recovery = clone / "recovery"
    recovery.mkdir()
    authorities = {
        "artifact_manifest.json": "original_artifact_manifest.json",
        "source_bindings.json": "original_source_bindings.json",
        "status.json": "original_status.json",
        "failure.json": "original_failure.json",
        "REPORT.md": "original_REPORT.md",
        "resource_ledger.json": "original_resource_ledger.json",
    }
    for original, copied in authorities.items():
        shutil.copy2(clone / original, recovery / copied)
    shutil.copy2(archive, recovery / "execution_runner.py")
    authority_hashes = {name: runner._file_sha256(clone / name) for name in EXPECTED_AUTHORITY_SHA256}
    _require(authority_hashes == EXPECTED_AUTHORITY_SHA256, "copied original authority changed")
    rows = [dict(row) for row in manifest["artifacts"] if row["path"] not in MUTABLE_ORIGINALS]
    _require(len(rows) == 91 and any(row["path"] == "failure.json" for row in rows), "immutable inventory membership changed")
    inventory = {"schema": "pixel-ddpm-immutable-inventory-v1", "original_manifest_sha256": EXPECTED_MANIFEST_SHA256, "artifact_count": len(rows), "artifact_bytes": sum(int(row["size"]) for row in rows), "artifacts": rows}
    inventory_path = recovery / "immutable_inventory.json"
    runner._write_json(inventory_path, inventory)
    verification_sources = runner._source_hashes(Path(execution_bindings["repository_root"]))
    _require(verification_sources[RUNNER_PATH] == EXPECTED_VERIFICATION_RUNNER_SHA256, "verification runner hash changed")
    _require(all(verification_sources[name] == digest for name, digest in EXPECTED_EXECUTION_SOURCES.items() if name != RUNNER_PATH), "non-runner verification source changed")
    receipt = {
        "schema": "pixel-ddpm-terminal-recovery-v1",
        "original_run": str(source),
        "recovered_run": str(output),
        "original_tree_sha256": EXPECTED_TREE_SHA256,
        "original_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "execution_runner_sha256": EXPECTED_EXECUTION_RUNNER_SHA256,
        "verification_runner_sha256": EXPECTED_VERIFICATION_RUNNER_SHA256,
        "execution_runner_archive_path": "recovery/execution_runner.py",
        "execution_runner_archive_sha256": EXPECTED_EXECUTION_RUNNER_SHA256,
        "immutable_inventory_path": "recovery/immutable_inventory.json",
        "immutable_inventory_sha256": runner._file_sha256(inventory_path),
        "immutable_artifact_count": len(rows),
        "cuda_reexecuted": 0,
        "original_authority_sha256": authority_hashes,
    }
    runner._write_json(recovery / "recovery_receipt.json", receipt)
    bindings = dict(execution_bindings)
    bindings["source_files"] = verification_sources
    bindings["verifier_recovery"] = {"schema": "pixel-ddpm-verifier-recovery-v1", "receipt_path": "recovery/recovery_receipt.json", "execution_source_files": execution_bindings["source_files"], "verification_source_files": verification_sources}
    runner._write_json(clone / "source_bindings.json", bindings)
    return receipt, rows


def _verify_immutable(clone: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        path = clone / ("recovery/original_failure.json" if row["path"] == "failure.json" else row["path"])
        _require(path.is_file() and path.stat().st_size == row["size"] and runner._file_sha256(path) == row["sha256"], f"immutable artifact changed: {row['path']}")
    _require(runner._file_sha256(clone / "resource_ledger.json") == EXPECTED_AUTHORITY_SHA256["recovery/original_resource_ledger.json"], "recovery charged or changed the resource ledger")


def recover(source: Path, output: Path, archive: Path) -> dict[str, Any]:
    source, output, archive = source.resolve(), output.resolve(), archive.resolve()
    _require(output.name == OUTPUT_NAME and output.parent.is_dir() and not output.exists(), "output must be a new cpu-recovered run")
    _require(output != source and output not in source.parents and source not in output.parents, "source and output overlap")
    manifest, execution_bindings = _validate_source(source, archive)
    created = False
    try:
        output.mkdir()
        created = True
        shutil.copytree(source, output, dirs_exist_ok=True)
        receipt, rows = _write_capsule(output, archive, source, output, manifest, execution_bindings)
        (output / "failure.json").unlink()
        runner._replace(output / "REPORT.md", lambda path: path.write_text(runner._report(output, None), encoding="utf-8"))
        runner._status(output, "awaiting_human_review", resumable=False)
        runner._refresh_manifest(output)
        verified = runner.verify_run(output)
        _verify_immutable(output, rows)
        _require(runner._file_sha256(source / "artifact_manifest.json") == EXPECTED_MANIFEST_SHA256 and runner._tree_digest(source) == EXPECTED_TREE_SHA256, "recovery mutated the source run")
        return {"schema": receipt["schema"], "source_preserved": 1, "cuda_reexecuted": 0, "immutable_artifact_count": len(rows), "output_run": str(output), "verification": verified}
    except BaseException:
        if created:
            shutil.rmtree(output)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--execution-runner-archive", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(recover(Path(args.source_run), Path(args.output_run), Path(args.execution_runner_archive)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
