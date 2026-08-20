"""Relocation-safe continuation of the sealed directional adjudication.

The Windows ``ready_for_fittrace`` run is immutable and is never resumed in
place.  This workflow verifies its complete registered byte inventory, maps
the two explicit parent trees by content, and creates a new continuation child
whose operational paths are native to the current host.  Scientific stages
delegate to the unchanged directional adjudication implementation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication
    as _base,
)
from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_gate import (
    decision_exit_code,
    safety_record,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_provenance import (
    compare_parent_snapshots,
    snapshot_parent_run,
)
from mnist.d0_jacobi_rb_quartile_directional_portable import (
    PortableContinuationError,
    REPORT_NAMES,
    portable_role_loading,
    verify_legacy_source_closure,
    verify_portable_continuation,
    verify_ready_predecessor,
    verify_relocated_parent_snapshots,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-quartile-directional-portable-continuation"
RUN_SCHEMA_VERSION = 1
STAGES = ("relocate", "fittrace", "nominate", "adjudicate", "report", "all")
REQUIRED_GATES = ("none", "relocate", "fittrace", "nominate", "adjudicate")
EXPECTED_RUNTIME = {
    "python": "3.14.4",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "numpy": "2.4.4",
    "python-flint": "0.9.0",
    "cuda_build": "12.8",
    "compute_capability": [12, 0],
    # Match PyTorch's reported memory for the marketed 8 GB predecessor GPU.
    "minimum_cuda_memory_bytes": 8_000_000_000,
    "minimum_free_workspace_bytes": 4 * 1024**3,
}

_IMPORTED_EVIDENCE = (
    "candidate_component_plan.json",
    "inference_plan.json",
    "bootstrap_index_seal.json",
    "role_firewall.json",
    "preflight_metrics.json",
    "preflight_gate.json",
    "historical_gain_table_replay.json",
    "historical_rank_table_replay.json",
    "historical_replay_gate.json",
    "replay_artifact_seal.json",
    "quadratic_moment_algebra_control.json",
    "component_recomposition_control.json",
    "synthetic_mechanism_controls.json",
    "exact_zero_control.json",
    "controls_gate.json",
    "controls_artifact_seal.json",
)

_RELOCATION_ARTIFACTS = (
    "run_manifest.json",
    "scientific_config.json",
    "legacy_scientific_config.json",
    "source_closure.json",
    "relocation_mount_map.json",
    "portable_relocation_identity.json",
    "portable_predecessor_binding.json",
    "portable_legacy_source_closure.json",
    "portable_parent_tree_binding.json",
    "parent_provenance.json",
    "parent_immutability_before.json",
    "predecessor_resource_projection.json",
    "resource_projection.json",
    "portable_runtime_contract.json",
    *_IMPORTED_EVIDENCE,
    "relocation_gate.json",
)


class PortableWorkflowError(RuntimeError):
    def __init__(self, message: str, *, failure_domain: str, failure_code: str) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _semantic(record: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in record.items() if key != "semantic_sha256"}
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ArtifactCompatibilityError(f"{Path(path).name} is not a JSON object")
    return value


def _verify_semantic(record: Mapping[str, Any], description: str) -> None:
    expected = config_fingerprint(
        {key: value for key, value in record.items() if key != "semantic_sha256"}
    )
    if record.get("semantic_sha256") != expected:
        raise ArtifactCompatibilityError(f"{description} semantic hash changed")


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _portable_source_closure(
    legacy_sources: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in legacy_sources.get("sources", []):
        relative = str(item["path"])
        path = repo_root / relative
        rows.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
                "legacy_scientific_source": 1,
            }
        )
    for path in (
        Path(__file__).resolve(),
        repo_root / "mnist/d0_jacobi_rb_quartile_directional_portable.py",
    ):
        relative = path.relative_to(repo_root).as_posix()
        if any(row["path"] == relative for row in rows):
            continue
        rows.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
                "legacy_scientific_source": 0,
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    content_sha = config_fingerprint(rows)
    return _semantic(
        {
            "schema": f"{RUN_SCHEMA}-source-closure",
            "schema_version": RUN_SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "source_count": len(rows),
            "portable_content_sha256": content_sha,
            "legacy_source_fingerprint": legacy_sources[
                "legacy_source_fingerprint"
            ],
            "sources": rows,
            **safety_record(),
        }
    )


def _science_core(record: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "schema",
        "schema_version",
        "semantic_sha256",
        "parent_quartile_specialist_run_dir",
        "parent_time_local_run_dir",
        "source_fingerprint",
    }
    return {key: value for key, value in record.items() if key not in ignored}


def _mount_map(args: argparse.Namespace) -> dict[str, Any]:
    return _semantic(
        {
            "schema": f"{RUN_SCHEMA}-mount-map",
            "schema_version": RUN_SCHEMA_VERSION,
            "source_adjudication_run_dir": str(args.source_adjudication_run_dir),
            "parent_quartile_specialist_run_dir": str(
                args.parent_quartile_specialist_run_dir
            ),
            "parent_time_local_run_dir": str(args.parent_time_local_run_dir),
            "repo_root": str(Path(__file__).resolve().parents[1]),
            "mounts_must_remain_stable_for_resume": 1,
            **safety_record(),
        }
    )


def _scientific_config(
    args: argparse.Namespace,
    *,
    legacy: Mapping[str, Any],
    identity_sha256: str,
    source_closure: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        key: value
        for key, value in legacy.items()
        if key not in {"semantic_sha256", "schema", "schema_version"}
    }
    body.update(
        {
            "schema": f"{RUN_SCHEMA}-scientific-config",
            "schema_version": RUN_SCHEMA_VERSION,
            "parent_quartile_specialist_run_dir": str(
                args.parent_quartile_specialist_run_dir
            ),
            "parent_time_local_run_dir": str(args.parent_time_local_run_dir),
            "source_adjudication_run_dir": str(args.source_adjudication_run_dir),
            "legacy_scientific_config_sha256": legacy["semantic_sha256"],
            "legacy_source_fingerprint": legacy["source_fingerprint"],
            "scientific_contract_sha256": config_fingerprint(_science_core(legacy)),
            "portable_relocation_identity_sha256": identity_sha256,
            "portable_source_content_sha256": source_closure[
                "portable_content_sha256"
            ],
            "relocation_changes_scientific_contract": 0,
            **safety_record(),
        }
    )
    return _semantic(body)


def _runtime_contract(device: torch.device, *, test_only: bool) -> dict[str, Any]:
    versions = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": str(torch.__version__),
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": str(np.__version__),
        "python-flint": importlib.metadata.version("python-flint"),
        "cuda_build": str(torch.version.cuda),
    }
    cuda_available = bool(torch.cuda.is_available())
    capability: list[int] | None = None
    total_memory = 0
    device_name = None
    device_uuid = None
    if device.type == "cuda" and cuda_available:
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        capability = [int(properties.major), int(properties.minor)]
        total_memory = int(properties.total_memory)
        device_name = str(properties.name)
        device_uuid = str(getattr(properties, "uuid", ""))
    backend = {
        "deterministic_algorithms": int(torch.are_deterministic_algorithms_enabled()),
        "cuda_matmul_allow_tf32": int(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": int(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": int(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": int(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    workspace_free = int(shutil.disk_usage(Path.cwd()).free)
    exact_versions = all(versions[key] == value for key, value in EXPECTED_RUNTIME.items() if key in versions)
    exact_backend = backend == {
        "deterministic_algorithms": 0,
        "cuda_matmul_allow_tf32": 0,
        "cudnn_allow_tf32": 1,
        "cudnn_deterministic": 0,
        "cudnn_benchmark": 0,
        "cublas_workspace_config": None,
    }
    resource = (
        cuda_available
        and capability == EXPECTED_RUNTIME["compute_capability"]
        and total_memory >= int(EXPECTED_RUNTIME["minimum_cuda_memory_bytes"])
        and workspace_free >= int(EXPECTED_RUNTIME["minimum_free_workspace_bytes"])
    )
    passed = bool(test_only or (device.type == "cuda" and exact_versions and exact_backend and resource))
    return _semantic(
        {
            "schema": f"{RUN_SCHEMA}-runtime-contract",
            "schema_version": RUN_SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": int(passed),
            "versions": versions,
            "backend": backend,
            "cuda_available": int(cuda_available),
            "cuda_device_name": device_name,
            "cuda_device_uuid": device_uuid,
            "cuda_compute_capability": capability,
            "cuda_total_memory": total_memory,
            "workspace_free_bytes": workspace_free,
            "expected": dict(EXPECTED_RUNTIME),
            "test_only": int(test_only),
            **safety_record(),
        }
    )


def _seal(run_dir: Path, names: Sequence[str], *, stage: str) -> dict[str, Any]:
    rows = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"relocation artifact is missing: {name}")
        rows.append(
            {"path": name, "sha256": file_fingerprint(path), "size": path.stat().st_size}
        )
    record = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-{stage}-artifact-seal",
            "schema_version": RUN_SCHEMA_VERSION,
            "stage": stage,
            "artifacts": rows,
            **safety_record(),
        }
    )
    atomic_write_json(run_dir / f"{stage}_artifact_seal.json", record)
    return record


def _verify_seal(run_dir: Path, *, stage: str) -> dict[str, Any]:
    record = _load_json(run_dir / f"{stage}_artifact_seal.json")
    _verify_semantic(record, f"{stage} artifact seal")
    if record.get("stage") != stage:
        raise ArtifactCompatibilityError("portable stage seal identity changed")
    for row in record.get("artifacts", []):
        path = run_dir / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size"])
            or file_fingerprint(path) != row["sha256"]
        ):
            raise ArtifactCompatibilityError(f"portable sealed artifact changed: {row['path']}")
    return record


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}_{args.run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir, False


def _write_native_parent_snapshot(run_dir: Path, args: argparse.Namespace) -> None:
    specialist = snapshot_parent_run(args.parent_quartile_specialist_run_dir)
    time_local = snapshot_parent_run(args.parent_time_local_run_dir)
    atomic_write_json(
        run_dir / "parent_immutability_before.json",
        _base._semantic(  # noqa: SLF001
            {
                "schema": f"{_base.RUN_SCHEMA}-parent-immutability-before",
                "schema_version": 1,
                "quartile_specialist": specialist,
                "time_local": time_local,
                **safety_record(),
            }
        ),
    )


def _import_predecessor_evidence(run_dir: Path, source: Path) -> None:
    for name in _IMPORTED_EVIDENCE:
        _atomic_copy(source / name, run_dir / name)
    _atomic_copy(
        source / "resource_projection.json",
        run_dir / "predecessor_resource_projection.json",
    )
    _atomic_copy(
        source / "scientific_config.json", run_dir / "legacy_scientific_config.json"
    )


def _relocate_stage(run_dir: Path, args: argparse.Namespace) -> None:
    gate_path = run_dir / "relocation_gate.json"
    if gate_path.is_file() and (run_dir / "relocation_artifact_seal.json").is_file():
        gate = _load_json(gate_path)
        if int(gate.get("passed", 0)) == 1:
            _verify_seal(run_dir, stage="relocation")
            return
    repo_root = Path(__file__).resolve().parents[1]
    predecessor = verify_ready_predecessor(args.source_adjudication_run_dir)
    legacy_sources = verify_legacy_source_closure(
        args.source_adjudication_run_dir,
        repo_root=repo_root,
    )
    parents = verify_relocated_parent_snapshots(
        args.source_adjudication_run_dir,
        specialist_run_dir=args.parent_quartile_specialist_run_dir,
        time_local_run_dir=args.parent_time_local_run_dir,
    )
    evidence = verify_portable_continuation(
        args.source_adjudication_run_dir,
        specialist_run_dir=args.parent_quartile_specialist_run_dir,
        time_local_run_dir=args.parent_time_local_run_dir,
        repo_root=repo_root,
    )
    identity_sha = str(evidence["semantic_sha256"])
    portable_sources = _portable_source_closure(legacy_sources, repo_root=repo_root)
    legacy_config = _load_json(
        Path(args.source_adjudication_run_dir) / "scientific_config.json"
    )
    config = _scientific_config(
        args,
        legacy=legacy_config,
        identity_sha256=identity_sha,
        source_closure=portable_sources,
    )
    mount_map = _mount_map(args)
    manifest = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-manifest",
            "schema_version": RUN_SCHEMA_VERSION,
            "created_at": _now(),
            "run_dir": str(run_dir),
            "scientific_config_sha256": config["semantic_sha256"],
            "portable_source_content_sha256": portable_sources[
                "portable_content_sha256"
            ],
            "portable_relocation_identity_sha256": identity_sha,
            "mount_map_semantic_sha256": mount_map["semantic_sha256"],
            "predecessor_registry_file_sha256": predecessor[
                "registry_file_sha256"
            ],
            "test_only": int(args.test_only),
            **safety_record(),
        }
    )
    atomic_write_json(run_dir / "scientific_config.json", config)
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "source_closure.json", portable_sources)
    atomic_write_json(run_dir / "relocation_mount_map.json", mount_map)
    atomic_write_json(run_dir / REPORT_NAMES["predecessor"], predecessor)
    atomic_write_json(run_dir / REPORT_NAMES["legacy_sources"], legacy_sources)
    atomic_write_json(run_dir / REPORT_NAMES["parents"], parents)
    atomic_write_json(run_dir / REPORT_NAMES["identity"], evidence)
    atomic_write_json(
        run_dir / "parent_provenance.json",
        _semantic(
            {
                "schema": f"{RUN_SCHEMA}-parent-provenance",
                "schema_version": RUN_SCHEMA_VERSION,
                "portable_relocation_identity_sha256": identity_sha,
                "predecessor": predecessor,
                "parents": parents,
                **safety_record(),
            }
        ),
    )
    _import_predecessor_evidence(run_dir, Path(args.source_adjudication_run_dir))
    _write_native_parent_snapshot(run_dir, args)
    runtime = _runtime_contract(torch.device(args.device), test_only=args.test_only)
    atomic_write_json(run_dir / "portable_runtime_contract.json", runtime)
    if int(runtime["passed"]) != 1:
        raise PortableWorkflowError(
            "RunPod runtime does not match the frozen continuation contract",
            failure_domain="portable_runtime",
            failure_code="portable_runtime_invalid",
        )
    with portable_role_loading(args.parent_quartile_specialist_run_dir):
        resource = _base._resource_pilot(args)  # noqa: SLF001
    atomic_write_json(run_dir / "resource_projection.json", resource)
    imported_valid = all(
        (run_dir / name).is_file()
        and file_fingerprint(run_dir / name)
        == file_fingerprint(Path(args.source_adjudication_run_dir) / name)
        for name in _IMPORTED_EVIDENCE
    )
    science_unchanged = config_fingerprint(_science_core(legacy_config)) == config[
        "scientific_contract_sha256"
    ]
    passed = bool(
        int(predecessor["passed"])
        and int(legacy_sources["passed"])
        and int(parents["passed"])
        and imported_valid
        and science_unchanged
        and int(runtime["passed"])
        and int(resource["within_limits"])
    )
    gate = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-relocation-gate",
            "schema_version": RUN_SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "stage_execution_valid": 1,
            "passed": int(passed),
            "predecessor_valid": int(predecessor["passed"]),
            "legacy_source_closure_valid": int(legacy_sources["passed"]),
            "relocated_parent_trees_valid": int(parents["passed"]),
            "imported_readiness_evidence_valid": int(imported_valid),
            "scientific_contract_unchanged": int(science_unchanged),
            "runtime_contract_valid": int(runtime["passed"]),
            "resource_projection_valid": int(resource["within_limits"]),
            "portable_relocation_identity_sha256": identity_sha,
            **safety_record(),
        }
    )
    atomic_write_json(gate_path, gate)
    _seal(run_dir, _RELOCATION_ARTIFACTS, stage="relocation")
    if not passed:
        raise PortableWorkflowError(
            "portable relocation gate failed",
            failure_domain="portable_relocation",
            failure_code="portable_relocation_invalid",
        )
    _base._status(  # noqa: SLF001
        run_dir, stage="relocate", state="running", decision="ready_for_fittrace"
    )


def _verify_resume(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate = _load_json(run_dir / "relocation_gate.json")
    _verify_semantic(gate, "relocation gate")
    if int(gate.get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("portable relocation gate did not pass")
    _verify_seal(run_dir, stage="relocation")
    mount_map = _mount_map(args)
    saved_mount = _load_json(run_dir / "relocation_mount_map.json")
    if saved_mount != mount_map:
        raise ArtifactCompatibilityError("portable continuation mount map changed")
    evidence = verify_portable_continuation(
        args.source_adjudication_run_dir,
        specialist_run_dir=args.parent_quartile_specialist_run_dir,
        time_local_run_dir=args.parent_time_local_run_dir,
        repo_root=Path(__file__).resolve().parents[1],
    )
    if evidence != _load_json(run_dir / REPORT_NAMES["identity"]):
        raise ArtifactCompatibilityError("portable relocation identity changed")
    config = _load_json(run_dir / "scientific_config.json")
    manifest = _load_json(run_dir / "run_manifest.json")
    _verify_semantic(config, "portable scientific config")
    _verify_semantic(manifest, "portable manifest")
    if (
        manifest.get("run_dir") != str(run_dir)
        or manifest.get("scientific_config_sha256") != config["semantic_sha256"]
        or config.get("portable_relocation_identity_sha256")
        != evidence["semantic_sha256"]
    ):
        raise ArtifactCompatibilityError("portable continuation resume binding changed")
    before = _load_json(run_dir / "parent_immutability_before.json")
    specialist = snapshot_parent_run(args.parent_quartile_specialist_run_dir)
    time_local = snapshot_parent_run(args.parent_time_local_run_dir)
    compare_parent_snapshots(before["quartile_specialist"], specialist)
    compare_parent_snapshots(before["time_local"], time_local)
    return evidence


def _relocation_resume_is_uncommitted(run_dir: Path) -> bool:
    """Allow only an interrupted, never-sealed relocation to restart in place."""

    if (run_dir / "relocation_artifact_seal.json").exists():
        return False
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return True
    status = _load_json(status_path)
    return (
        status.get("state") in {"running", "interrupted"}
        and status.get("decision") in {"running_relocate", "interrupted_relocate"}
    )


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("relocate", "fittrace", "nominate", "adjudicate", "report")
    return (stage,)


def _delegate_stage(run_dir: Path, args: argparse.Namespace, stage: str) -> None:
    if stage == "fittrace":
        if not _base._passed(run_dir / "controls_gate.json"):  # noqa: SLF001
            raise ArtifactCompatibilityError("fittrace requires imported controls")
        with portable_role_loading(args.parent_quartile_specialist_run_dir):
            _base._fittrace_stage(run_dir, args)  # noqa: SLF001
    elif stage == "nominate":
        if not _base._passed(run_dir / "fittrace_gate.json"):  # noqa: SLF001
            raise ArtifactCompatibilityError("nomination requires passing fittrace")
        with portable_role_loading(args.parent_quartile_specialist_run_dir):
            _base._nominate_stage(run_dir, args)  # noqa: SLF001
    elif stage == "adjudicate":
        if not _base._passed(run_dir / "nominate_gate.json"):  # noqa: SLF001
            raise ArtifactCompatibilityError("adjudication requires passing nomination")
        with portable_role_loading(args.parent_quartile_specialist_run_dir):
            _base._adjudicate_stage(run_dir, args)  # noqa: SLF001
    elif stage == "report":
        if not _base._passed(run_dir / "adjudicate_gate.json"):  # noqa: SLF001
            raise ArtifactCompatibilityError("report requires passing adjudication")
        _base._report_stage(run_dir, args)  # noqa: SLF001
    else:
        raise ValueError(f"unknown delegated stage: {stage}")


def _required_gate_passed(run_dir: Path, require_gate: str) -> bool:
    if require_gate == "none":
        return True
    if require_gate == "relocate":
        return int(_load_json(run_dir / "relocation_gate.json").get("passed", 0)) == 1
    return _base._passed(run_dir / f"{require_gate}_gate.json")  # noqa: SLF001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--source-adjudication-run-dir", type=Path, required=True)
    parser.add_argument("--parent-quartile-specialist-run-dir", type=Path, required=True)
    parser.add_argument("--parent-time-local-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_quartile_directional_portable_continuation"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-runpod-quartile-directional-continuation"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in (
        "source_adjudication_run_dir",
        "parent_quartile_specialist_run_dir",
        "parent_time_local_run_dir",
        "resume_run_dir",
        "runs_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    if args.resume_run_dir is None and args.stage not in {"relocate", "all"}:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    expected = {
        "relocate": "relocate",
        "fittrace": "fittrace",
        "nominate": "nominate",
        "adjudicate": "adjudicate",
        "report": "none",
        "all": "adjudicate",
    }[args.stage]
    if args.require_gate not in {"none", expected}:
        parser.error(f"--stage {args.stage} cannot require {args.require_gate}")
    if args.device != "cuda" and not args.test_only and args.stage != "report":
        parser.error("production portable continuation requires --device cuda")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    initialized = False
    active_stage = "initialize"
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"portable directional run directory: {run_dir}", flush=True)
        if resumed:
            if not (
                args.stage == "relocate"
                and _relocation_resume_is_uncommitted(run_dir)
            ):
                _verify_resume(run_dir, args)
        initialized = True
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            if stage == "relocate":
                _relocate_stage(run_dir, args)
            else:
                _verify_resume(run_dir, args)
                _base._status(  # noqa: SLF001
                    run_dir, stage=stage, state="running", decision=f"running_{stage}"
                )
                _delegate_stage(run_dir, args, stage)
        workflow = _base._workflow_record(run_dir, "none")  # noqa: SLF001
        decision = workflow["decision"]
        name = str(decision["decision"])
        exit_code = decision_exit_code(decision)
        if not _required_gate_passed(run_dir, args.require_gate):
            exit_code = 1
        terminal = bool(decision.get("terminal", 0))
        state = "running"
        if terminal:
            state = (
                "complete"
                if int(decision.get("unique_representation_identified", 0))
                else "valid_scientific_stop"
            )
        if exit_code != 0:
            state = "gate_failed"
        _base._status(  # noqa: SLF001
            run_dir,
            stage=args.stage,
            state=state,
            decision=name,
            failure_domain="scientific_gate" if state == "valid_scientific_stop" else None,
            failure_code=name if state == "valid_scientific_stop" else None,
        )
        _base._artifact_registry(run_dir)  # noqa: SLF001
        print(f"portable directional decision: {name}", flush=True)
        return exit_code
    except KeyboardInterrupt:
        if run_dir is not None and initialized:
            _base._status(  # noqa: SLF001
                run_dir,
                stage=active_stage,
                state="interrupted",
                decision=f"interrupted_{active_stage}",
                message="interrupted; resume this portable child run directory",
                failure_domain="interruption",
                failure_code="keyboard_interrupt",
            )
            _base._artifact_registry(run_dir)  # noqa: SLF001
        raise
    except Exception as exc:
        if run_dir is not None:
            failure = _semantic(
                {
                    "schema": f"{RUN_SCHEMA}-execution-failure",
                    "schema_version": RUN_SCHEMA_VERSION,
                    "evaluation_status": "execution_failed",
                    "stage": active_stage,
                    "failure_domain": str(
                        getattr(exc, "failure_domain", "portable_execution")
                    ),
                    "failure_code": str(
                        getattr(exc, "failure_code", "portable_continuation_failed")
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **safety_record(),
                }
            )
            atomic_write_json(run_dir / f"{active_stage}_execution_failure.json", failure)
            if active_stage == "relocate" and not (run_dir / "relocation_gate.json").is_file():
                atomic_write_json(
                    run_dir / "relocation_gate.json",
                    _semantic(
                        {
                            "schema": f"{RUN_SCHEMA}-relocation-gate",
                            "schema_version": RUN_SCHEMA_VERSION,
                            "evaluation_status": "execution_failed",
                            "stage_execution_valid": 0,
                            "passed": 0,
                            "failure_domain": failure["failure_domain"],
                            "failure_code": failure["failure_code"],
                            **safety_record(),
                        }
                    ),
                )
            elif active_stage in {"fittrace", "nominate", "adjudicate"}:
                _base._commit_failed_stage_gate(  # noqa: SLF001
                    run_dir, stage=active_stage, failure=failure
                )
            _base._status(  # noqa: SLF001
                run_dir,
                stage=active_stage,
                state="execution_failed",
                decision="portable_continuation_invalid",
                message=str(exc),
                failure_domain=failure["failure_domain"],
                failure_code=failure["failure_code"],
            )
            _base._artifact_registry(run_dir)  # noqa: SLF001
        print(f"portable directional error: {exc}", flush=True)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
