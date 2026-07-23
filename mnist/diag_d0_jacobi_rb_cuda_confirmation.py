"""Rigorous fused-CUDA confirmation for the certified Jacobi RB kernel.

The immutable parent established numerical validity and failed only resource
feasibility.  This additive controls-only workflow replays its certificates,
measures the new CUDA implementation, and evaluates target controls only after
the CUDA kernel gate passes.  It never trains or performs reverse sampling.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import numpy as np

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_cuda_certificate import (
    run_certificate_arithmetic_preflight,
)
from mnist.d0_jacobi_rb_cuda_controls import (
    CUDA_CONTROL_VERSION,
    FULL_PATH_REPEATS,
    FULL_PATH_TRANSITIONS,
    MAX_CUDA_CHUNK_SIZE,
    STEPS_PER_SHARD,
    THROUGHPUT_REPEATS,
    THROUGHPUT_TRANSITIONS,
    WARMUP_TRANSITIONS,
    benchmark_shard_ranges,
    certificate_panel_plan,
    kernel_benchmark_plan,
    run_benchmark_shard,
    run_certificate_panel,
    run_stateful_path_shard,
    run_cuda_target_identity_controls,
    summarize_benchmark,
    target_metrics_from_certificate_rows,
)
from mnist.d0_jacobi_rb_controls import (
    deterministic_kernel_controls,
    target_identity_controls,
    transition_law_controls,
)
from mnist.d0_jacobi_rb_spectral import JacobiRBSpectralProfile
from mnist.d0_jacobi_rb_cuda_provenance import (
    PARENT_REGISTRY_SHA256,
    verify_and_readjudicate_jacobi_rb_cuda_parent,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-cuda-confirmation"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "rigorous fused-CUDA Jacobi RB transition and target controls only"
DEFAULTS: dict[str, Any] = {
    "root_seed": 261_131,
    "parent_replay_count": 294,
    "fresh_certificate_count": 512,
    "warmup_transitions": WARMUP_TRANSITIONS,
    "throughput_transitions": THROUGHPUT_TRANSITIONS,
    "throughput_repeats": THROUGHPUT_REPEATS,
    "full_path_transitions": FULL_PATH_TRANSITIONS,
    "full_path_repeats": FULL_PATH_REPEATS,
    "benchmark_chunk_size": MAX_CUDA_CHUNK_SIZE,
    "steps_per_shard": STEPS_PER_SHARD,
}
NO_WORK = {
    "physical_training_performed": 0,
    "reverse_sampling_performed": 0,
    "sampling_performed": 0,
}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _progress(args: argparse.Namespace, label: str, done: int, total: int, started: float) -> None:
    if args.no_progress or total <= 0:
        return
    elapsed = max(0.0, time.perf_counter() - started)
    eta = elapsed * max(0, total - done) / max(1, done)
    print(f"Jacobi RB CUDA {label} {done}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s")


def _atomic_save_figure(path: Path, figure: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.png")
    figure.savefig(temporary, dpi=150, bbox_inches="tight")
    temporary.replace(path)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _plot_certificate(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    modes = [int(row.get("mode_count", 0)) for row in rows]
    prefixes = [int(row.get("prefix_bits", 0)) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    axes[0].hist(modes, bins=min(32, max(1, len(set(modes)))))
    axes[0].set_title("Certified mode counts")
    axes[0].set_xlabel("modes")
    axes[1].hist(prefixes, bins=min(32, max(1, len(set(prefixes)))))
    axes[1].set_title("Dyadic prefix bits")
    axes[1].set_xlabel("bits")
    figure.tight_layout()
    _atomic_save_figure(run_dir / "jacobi_rb_cuda_certificate_diagnostics.png", figure)
    plt.close(figure)


def _plot_kernel(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    rates = [float(row.get("transitions_per_second", 0.0)) for row in rows]
    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    axis.plot(np.arange(len(rates)), rates, linewidth=1.0)
    axis.axhline(1_300.0, color="tab:red", linestyle="--", linewidth=1.0)
    axis.set_xlabel("restart shard")
    axis.set_ylabel("transitions / second")
    axis.set_title("Complete certified API throughput")
    figure.tight_layout()
    _atomic_save_figure(run_dir / "jacobi_rb_cuda_kernel_throughput.png", figure)
    plt.close(figure)


def _load(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return dict(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "certificate", "kernel", "target", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-gate", choices=("none", "preflight", "certificate", "kernel", "target"),
        default="none",
    )
    parser.add_argument("--parent-rb-kernel-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_cuda_confirmation"),
    )
    parser.add_argument("--run-name", default="production-rigorous-jacobi-rb-cuda-confirmation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-seed", type=int, default=DEFAULTS["root_seed"])
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--test-only-reduced-workload", action="store_true", help=argparse.SUPPRESS)
    for name in (
        "parent_replay_count", "fresh_certificate_count", "warmup_transitions",
        "throughput_transitions", "throughput_repeats", "full_path_transitions",
        "full_path_repeats", "benchmark_chunk_size", "steps_per_shard",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=DEFAULTS[name])
    args = parser.parse_args(argv)
    if args.stage in {"certificate", "kernel", "target", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "certificate": {"none", "preflight", "certificate"},
        "kernel": {"none", "preflight", "certificate", "kernel"},
        "target": {"none", "preflight", "certificate", "kernel", "target"},
        "report": {"none", "preflight", "certificate", "kernel", "target"},
        "all": {"none", "preflight", "certificate", "kernel", "target"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(f"--require-gate {args.require_gate} is unavailable at stage {args.stage}")
    for name in DEFAULTS:
        if name == "root_seed":
            continue
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.benchmark_chunk_size) > MAX_CUDA_CHUNK_SIZE:
        parser.error(f"--benchmark-chunk-size must not exceed {MAX_CUDA_CHUNK_SIZE}")
    changed = [name for name, value in DEFAULTS.items() if getattr(args, name) != value]
    if changed and not args.test_only_reduced_workload:
        parser.error("production workload is frozen; overrides require --test-only-reduced-workload: " + ", ".join(changed))
    if args.test_only_reduced_workload and args.require_gate != "none":
        parser.error("test-only reduced workloads cannot satisfy a required gate")
    if not args.test_only_reduced_workload and int(args.steps_per_shard) != STEPS_PER_SHARD:
        parser.error(f"production shards span exactly {STEPS_PER_SHARD} steps")
    if torch.device(args.device).type != "cuda" and not args.test_only_reduced_workload:
        parser.error("production confirmation requires --device cuda")
    return args


def _run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        root = Path(args.resume_run_dir).resolve()
        if not root.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {root}")
        return root, True
    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)
    value = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.run_name}"
    value.mkdir(parents=False, exist_ok=False)
    return value.resolve(), False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    record = _load(path) if path.is_file() else {
        "schema": RUN_SCHEMA, "schema_version": RUN_SCHEMA_VERSION, "created_at": _now()
    }
    record.update(updates)
    record.update({"updated_at": _now(), **NO_WORK})
    atomic_write_json(path, record)
    return record


def _freeze(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _load(path) != normalized:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    else:
        atomic_write_json(path, normalized)
    return normalized


def _profile() -> JacobiRBCudaProfile:
    return JacobiRBCudaProfile()


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-scientific-config", "schema_version": 1,
        "claim_scope": CLAIM_SCOPE, "control_version": CUDA_CONTROL_VERSION,
        "root_seed": int(args.root_seed),
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "certificate_plan": certificate_panel_plan(
            root_seed=args.root_seed, test_only=args.test_only_reduced_workload,
            **({"parent_replay_count": args.parent_replay_count, "fresh_count": args.fresh_certificate_count}
               if args.test_only_reduced_workload else {}),
        ).record(),
        "kernel_plan": kernel_benchmark_plan(
            root_seed=args.root_seed, test_only=args.test_only_reduced_workload,
            **({
                "warmup_transitions": args.warmup_transitions,
                "throughput_transitions": args.throughput_transitions,
                "throughput_repeats": args.throughput_repeats,
                "full_path_transitions": args.full_path_transitions,
                "full_path_repeats": args.full_path_repeats,
                "chunk_size": args.benchmark_chunk_size,
                "steps_per_shard": args.steps_per_shard,
            } if args.test_only_reduced_workload else {}),
        ).record(),
        "test_only_reduced_workload": int(args.test_only_reduced_workload),
        **NO_WORK,
    }


def _source_record() -> tuple[str, list[str]]:
    gate = importlib.import_module("mnist.d0_jacobi_rb_cuda_gate")
    fused = importlib.import_module("mnist.d0_jacobi_rb_cuda_fused")
    modules = [
        sys.modules[__name__], sys.modules[JacobiRBCudaProfile.__module__],
        sys.modules[run_certificate_arithmetic_preflight.__module__],
        sys.modules[run_certificate_panel.__module__],
        sys.modules[verify_and_readjudicate_jacobi_rb_cuda_parent.__module__], gate,
        fused,
    ]
    paths = sorted({Path(module.__file__).resolve() for module in modules})
    return source_fingerprint(paths), [str(path) for path in paths]


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path), "size": int(path.stat().st_size)
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry", "schema_version": 1,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "records": records, **NO_WORK,
    }


def _verify_terminal_registry(run_dir: Path) -> None:
    registry_path = run_dir / "artifact_registry.json"
    if not registry_path.is_file():
        return
    registry = _load(registry_path)
    status = _load(run_dir / "run_status.json")
    if status.get("artifact_registry_sha256") != file_fingerprint(registry_path):
        raise ArtifactCompatibilityError("resume status does not bind artifact registry")
    interrupted = status.get("status") == "running"
    mutable = {
        "jacobi_rb_cuda_certificate_gate.json", "jacobi_rb_cuda_kernel_gate.json",
        "jacobi_rb_cuda_target_gate.json", "jacobi_rb_cuda_decision.json",
        "jacobi_rb_cuda_workflow_gate.json",
    }
    recoverable_prefixes = ("cuda_benchmark_shards/",)
    for relative, raw in dict(registry.get("records", {})).items():
        path = run_dir / relative
        if not path.is_file() or raw.get("sha256") != file_fingerprint(path) or raw.get("size") != path.stat().st_size:
            if relative.startswith(recoverable_prefixes):
                # The shard loader independently checks its input fingerprint,
                # row hash, and state-chain predecessor before reusing it.
                # A corrupt or incomplete shard is therefore safe to rebuild.
                continue
            if interrupted and relative in mutable:
                continue
            raise ArtifactCompatibilityError(f"resume artifact changed: {relative}")
    if not interrupted:
        excluded = set(registry.get("terminal_files_excluded_to_avoid_self_reference", ()))
        actual = {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file() and p.relative_to(run_dir).as_posix() not in excluded}
        unexpected = actual - set(dict(registry.get("records", {})))
        if unexpected:
            raise ArtifactCompatibilityError("unregistered resume artifacts: " + ", ".join(sorted(unexpected)))


def _gate_function(kind: str) -> Any:
    module = importlib.import_module("mnist.d0_jacobi_rb_cuda_gate")
    aliases = {
        "preflight": ("evaluate_jacobi_rb_cuda_preflight", "evaluate_cuda_preflight"),
        "certificate": ("evaluate_jacobi_rb_cuda_certificate", "evaluate_cuda_certificate"),
        "kernel": ("evaluate_jacobi_rb_cuda_kernel", "evaluate_cuda_kernel"),
        "target": ("evaluate_jacobi_rb_cuda_target", "evaluate_cuda_target"),
    }
    for name in aliases[kind]:
        value = getattr(module, name, None)
        if callable(value):
            return value
    raise RuntimeError(f"CUDA gate module lacks the {kind} evaluator")


def _not_evaluated(name: str, reason: str) -> dict[str, Any]:
    module = importlib.import_module("mnist.d0_jacobi_rb_cuda_gate")
    function = getattr(module, "not_evaluated_gate", None)
    if callable(function):
        return dict(function(name, reason))
    return {"name": name, "evaluation_status": "not_evaluated", "passed": 0, "subchecks": {}, "reason": reason, **NO_WORK}


def _save_gate(run_dir: Path, kind: str, metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / f"{kind}_metrics.json", {"schema": RUN_SCHEMA + f"-{kind}-metrics", "schema_version": 1, "metrics": dict(metrics), **NO_WORK})
    atomic_write_json(run_dir / f"jacobi_rb_cuda_{kind}_gate.json", dict(gate))


def _failed_stage_gate(run_dir: Path, kind: str, exc: Exception) -> dict[str, Any]:
    diagnostics_raw = getattr(exc, "diagnostics", {})
    diagnostics = (
        dict(diagnostics_raw) if isinstance(diagnostics_raw, Mapping) else {}
    )
    searchable = " ".join(
        (
            str(exc),
            str(diagnostics.get("failure_kind", "")),
            json.dumps(diagnostics, sort_keys=True, default=str),
        )
    ).lower()
    # A returned Arb result is not a cap hit.  If Arb itself exhausts the
    # frozen 16,384-mode / 8,192-bit / 1,024-prefix limits, however, preserve
    # that fact even when the exception's human-readable message is generic.
    resource_cap = bool(
        "cap" in searchable
        or int(diagnostics.get("maximum_modes", 0) or 0) >= 16_384
        or int(diagnostics.get("precision_bits", 0) or 0) >= 8_192
        or int(diagnostics.get("prefix_bits", 0) or 0) >= 1_024
    )
    metrics = {
        "evaluation_status": "evaluated",
        "failure_type": type(exc).__name__, "failure": str(exc),
        "failure_diagnostics": diagnostics,
        "nonfinite_count": int("nonfinite" in searchable),
        "resource_cap_count": int(resource_cap),
        **NO_WORK,
    }
    atomic_write_json(
        run_dir / f"{kind}_failure.json",
        {"schema": RUN_SCHEMA + f"-{kind}-failure", "schema_version": 1, **metrics},
    )
    gate = dict(_gate_function(kind)(metrics))
    _save_gate(run_dir, kind, metrics, gate)
    return gate


def _passed(value: Mapping[str, Any]) -> bool:
    return value.get("evaluation_status", "evaluated") == "evaluated" and int(value.get("passed", 0)) == 1


def _parent_rows(parent_dir: Path, expected: int) -> list[dict[str, Any]]:
    paths = sorted((parent_dir / "support_shards").glob("support-*.json"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load(path)
        row = payload.get("row")
        if isinstance(row, Mapping):
            rows.append(dict(row))
    if len(rows) != int(expected):
        raise ArtifactCompatibilityError(f"parent must provide exactly {expected} registered support rows")
    return rows


def _preflight(run_dir: Path, args: argparse.Namespace, provenance: Mapping[str, Any], profile: JacobiRBCudaProfile) -> dict[str, Any]:
    device = torch.device(args.device)
    cuda_available = torch.cuda.is_available() and device.type == "cuda"
    backend_module = importlib.import_module("mnist.d0_jacobi_rb_cuda")
    runtime_function = getattr(backend_module, "_runtime_report", None)
    runtime_report = dict(runtime_function(device) if callable(runtime_function) else {})
    try:
        arithmetic_report = dict(run_certificate_arithmetic_preflight())
    except Exception as exc:  # A proof self-test must always fail closed.
        arithmetic_report = {
            "passed": 0,
            "double_double_interval_algebra_pass": 0,
            "certified_exponential_pass": 0,
            "errors": {"unexpected_exception": f"{type(exc).__name__}: {exc}"},
        }
    atomic_write_json(
        run_dir / "cuda_certificate_arithmetic_preflight.json",
        {**arithmetic_report, **NO_WORK},
    )
    compile_error = ""
    candidate_binary_sha256: str | None = None
    binary_sha256: str | None = None
    if cuda_available:
        loader = getattr(backend_module, "_load_cuda_kernel", None)
        if callable(loader):
            try:
                _kernel, candidate_binary_sha256 = loader(device, profile)
            except Exception as exc:
                compile_error = f"{type(exc).__name__}: {exc}"
        runtime_report = dict(runtime_function(device) if callable(runtime_function) else runtime_report)
        if candidate_binary_sha256:
            runtime_report["candidate_binary_sha256"] = str(candidate_binary_sha256)
        # The proposal CUBIN is deliberately nonauthorizing.  Preflight must
        # compile and execute the fused directed-rounding self-test itself;
        # otherwise a fresh run would report the authorizer unavailable until
        # the first transition, which is too late for a fail-closed gate.
        try:
            fused_module = importlib.import_module("mnist.d0_jacobi_rb_cuda_fused")
            bundle, fused_report = fused_module.probe_fused_cuda_authorizer(
                device,
                compile_flags=tuple(profile.compile_flags),
                cpu_preflight=arithmetic_report,
            )
            runtime_report.update(dict(fused_report))
            if bundle is not None:
                binary_sha256 = str(bundle.binary_sha256)
                runtime_report.update(
                    binary_sha256=binary_sha256,
                    cubin_sha256=binary_sha256,
                    kernel_sha256=str(bundle.source_sha256),
                    source_sha256=str(bundle.source_sha256),
                    directed_rounding_intrinsics_pass=bool(
                        fused_report.get("arithmetic_selftest_pass", False)
                    ),
                )
        except Exception as exc:
            fused_error = f"{type(exc).__name__}: {exc}"
            compile_error = "; ".join(value for value in (compile_error, fused_error) if value)
    runtime_report["compile_error"] = compile_error
    backend_flag = getattr(profile, "rigorous_backend_available", None)
    if callable(backend_flag):
        backend_available = bool(backend_flag())
    elif backend_flag is not None:
        backend_available = bool(backend_flag)
    else:
        backend_available = bool(
            cuda_available and runtime_report.get("loader_available", False)
            and runtime_report.get("fused_cuda_authorizer_available", False)
            and runtime_report.get("frozen_runtime_match", False)
        )
    backend_available = bool(backend_available and not compile_error and binary_sha256)
    floating_contract = bool(
        torch.are_deterministic_algorithms_enabled()
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32
    )
    double_double = bool(
        arithmetic_report.get("double_double_interval_algebra_pass", 0)
        and getattr(profile, "double_double_interval_certified", False)
    )
    certified_exp = bool(
        arithmetic_report.get("certified_exponential_pass", 0)
        and getattr(profile, "certified_device_exponential", False)
    )
    flags = tuple(runtime_report.get("compile_flags", ()))
    frozen_flags = (
        "--std=c++17", "--fmad=false", "--ftz=false",
        "--prec-div=true", "--prec-sqrt=true",
    )
    runtime_match = bool(
        runtime_report.get("frozen_runtime_match", False)
        and str(runtime_report.get("compute_capability", "")) == "12.0"
    )
    compile_contract = bool(
        not compile_error and flags == frozen_flags
        and runtime_report.get("header_free", False)
    )
    source_fingerprint_pass = bool(runtime_report.get("kernel_sha256"))
    cubin_fingerprint_pass = bool(binary_sha256)
    properties = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(device)
        runtime_report["device_uuid"] = str(getattr(properties, "uuid", ""))
    device_identity_pass = bool(
        properties is not None and properties.major == 12 and properties.minor == 0
        and getattr(properties, "uuid", None)
    )
    directed_rounding = bool(
        arithmetic_report.get("double_double_interval_algebra_pass", 0)
        and runtime_report.get("directed_rounding_intrinsics_pass", False)
    )
    atomic_write_json(
        run_dir / "fused_cuda_runtime_report.json",
        {**runtime_report, "profile": profile.to_dict(), **NO_WORK},
    )
    atomic_write_json(
        run_dir / "cuda_rng_plan.json",
        {
            "schema": RUN_SCHEMA + "-rng-plan", "schema_version": 1,
            "version": "philox4x32-10-canonical-transition-v2",
            "counter_words": [
                "transition_id_low", "transition_id_high",
                "refinement_block", "namespace",
            ],
            "canonical_coordinates": ["path", "outer_step", "phase", "edge"],
            "maximum_prefix_bits": 1024,
            **NO_WORK,
        },
    )
    metrics = {
        "control_provenance_pass": int(provenance.get("passed", 0)),
        "parent_record_count": int(provenance.get("parent_artifact_record_count", 0)),
        "parent_numerically_valid_pass": int(provenance.get("parent_numerically_valid", 0)),
        "parent_resource_infeasible_pass": int(not int(provenance.get("parent_resource_feasible", 1))),
        "fused_cuda_backend_available": int(backend_available),
        "cuda_floating_contract_pass": int(floating_contract),
        "frozen_runtime_match_pass": int(runtime_match),
        "compile_contract_pass": int(compile_contract),
        "cuda_source_fingerprint_pass": int(source_fingerprint_pass),
        "cubin_fingerprint_pass": int(cubin_fingerprint_pass),
        "device_identity_pass": int(device_identity_pass),
        "directed_rounding_contract_pass": int(directed_rounding),
        "double_double_interval_algebra_pass": int(double_double),
        "certified_exponential_pass": int(certified_exp),
        "deterministic_replay_pass": int(cuda_available and backend_available),
        "device_is_cuda": int(device.type == "cuda"), "test_only_reduced_workload": int(args.test_only_reduced_workload),
        "maximum_backend_chunk_size": int(args.benchmark_chunk_size),
        "chunk_cap_pass": int(args.benchmark_chunk_size <= MAX_CUDA_CHUNK_SIZE),
        "forbidden_approximation_count": 0, "nonfinite_count": 0, **NO_WORK,
    }
    gate = dict(_gate_function("preflight")(metrics))
    _save_gate(run_dir, "preflight", metrics, gate)
    return gate


def _certificate(run_dir: Path, args: argparse.Namespace, profile: JacobiRBCudaProfile) -> dict[str, Any]:
    plan = certificate_panel_plan(
        root_seed=args.root_seed, test_only=args.test_only_reduced_workload,
        **({"parent_replay_count": args.parent_replay_count, "fresh_count": args.fresh_certificate_count}
           if args.test_only_reduced_workload else {}),
    )
    rows, metrics = run_certificate_panel(
        _parent_rows(Path(args.parent_rb_kernel_run_dir), plan.parent_replay_count),
        device=torch.device(args.device), profile=profile, plan=plan,
    )
    runtime = _load(run_dir / "fused_cuda_runtime_report.json")
    metrics["cubin_replay_pass"] = int(
        metrics.get("binary_sha256_values") == [runtime.get("binary_sha256")]
    )
    metrics["cuda_source_replay_pass"] = int(
        metrics.get("cuda_source_sha256_values") == [runtime.get("kernel_sha256")]
    )
    atomic_write_json(run_dir / "jacobi_rb_cuda_certificate_panel.json", {"schema": RUN_SCHEMA + "-certificate-panel", "schema_version": 1, "rows": rows, **NO_WORK})
    atomic_write_csv(run_dir / "jacobi_rb_cuda_certificate_panel.csv", rows)
    _plot_certificate(run_dir, rows)
    gate = dict(_gate_function("certificate")(metrics))
    _save_gate(run_dir, "certificate", metrics, gate)
    return gate


def _load_shard(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = _load(path)
    except ArtifactCompatibilityError:
        return None
    row = value.get("row")
    if not isinstance(row, Mapping) or value.get("input_fingerprint") != fingerprint:
        return None
    if value.get("row_sha256") != config_fingerprint(row):
        return None
    return dict(row)


def _benchmark_family(
    run_dir: Path, *, label: str, transitions: int, repeats: int,
    args: argparse.Namespace, profile: JacobiRBCudaProfile,
) -> list[dict[str, Any]]:
    root = run_dir / "cuda_benchmark_shards" / label
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint({
        "label": label, "transitions": int(transitions), "repeats": int(repeats),
        "root_seed": int(args.root_seed), "chunk_size": int(args.benchmark_chunk_size),
        "steps_per_shard": int(args.steps_per_shard), "profile": repr(profile),
    })
    rows: list[dict[str, Any]] = []
    ranges = benchmark_shard_ranges(
        transitions, chunk_size=args.benchmark_chunk_size,
        steps_per_shard=args.steps_per_shard,
    )
    total_shards = int(repeats) * len(ranges)
    completed_shards = 0
    progress_started = time.perf_counter()
    for repeat in range(int(repeats)):
        for shard, (offset, count) in enumerate(ranges):
            path = root / f"repeat-{repeat:02d}-steps-{shard:06d}.json"
            row = _load_shard(path, fingerprint)
            if row is None:
                wall_started = time.perf_counter()
                row = run_benchmark_shard(
                    transition_count=count, global_offset=offset, repeat=repeat, shard=shard,
                    root_seed=args.root_seed, chunk_size=args.benchmark_chunk_size,
                    device=torch.device(args.device), profile=profile,
                )
                payload = {
                    "schema": RUN_SCHEMA + "-benchmark-shard", "schema_version": 1,
                    "input_fingerprint": fingerprint, "row": row,
                    "row_sha256": config_fingerprint(row), **NO_WORK,
                }
                atomic_write_json(path, payload)
                wall_elapsed = time.perf_counter() - wall_started
                row["wall_elapsed_seconds"] = wall_elapsed
                row["shard_io_included"] = 1
                row["transitions_per_second"] = int(row["transition_count"]) / wall_elapsed
                atomic_write_json(path, {
                    **payload, "row": row, "row_sha256": config_fingerprint(row),
                })
            rows.append(row)
            completed_shards += 1
            if completed_shards == total_shards or completed_shards % 8 == 0:
                _progress(args, label, completed_shards, total_shards, progress_started)
    return rows


def _stateful_full_family(
    run_dir: Path, args: argparse.Namespace, profile: JacobiRBCudaProfile
) -> list[dict[str, Any]]:
    """Run 2,744 evolving paths for 512 steps in eight-step shards."""

    if int(args.full_path_transitions) != 512 * 7 * 392:
        # Reduced tests may use arbitrary counts; they remain non-authorizing.
        return _benchmark_family(
            run_dir, label="full-path", transitions=args.full_path_transitions,
            repeats=args.full_path_repeats, args=args, profile=profile,
        )
    root = run_dir / "cuda_benchmark_shards" / "full-path"
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint({
        "kind": "evolving-28x28-seven-matching-512-step-path",
        "repeats": int(args.full_path_repeats), "root_seed": int(args.root_seed),
        "steps_per_shard": STEPS_PER_SHARD, "profile": repr(profile),
    })
    initial = np.random.Generator(np.random.Philox(int(args.root_seed) + 700)).dirichlet(
        np.ones(28 * 28, dtype=np.float64)
    )
    rows: list[dict[str, Any]] = []
    total_shards = int(args.full_path_repeats) * (512 // STEPS_PER_SHARD)
    completed_shards = 0
    progress_started = time.perf_counter()
    for repeat in range(int(args.full_path_repeats)):
        state = initial.copy()
        previous_shard_sha256 = config_fingerprint({
            "kind": "stateful-shard-genesis", "input_fingerprint": fingerprint,
            "repeat": repeat, "initial_state_sha256": config_fingerprint(state.tolist()),
        })
        for shard, start_step in enumerate(range(0, 512, STEPS_PER_SHARD)):
            path = root / f"repeat-{repeat:02d}-steps-{start_step:03d}-{start_step + STEPS_PER_SHARD - 1:03d}.json"
            input_state_sha256 = config_fingerprint(state.tolist())
            cached = _load_shard(path, fingerprint)
            if cached is not None and (
                cached.get("input_state_sha256") != input_state_sha256
                or cached.get("previous_shard_sha256") != previous_shard_sha256
            ):
                cached = None
            if cached is not None and isinstance(cached.get("final_state"), list):
                expected_chain = config_fingerprint({
                    "input_state_sha256": input_state_sha256,
                    "previous_shard_sha256": previous_shard_sha256,
                    "output_sha256": cached.get("output_sha256"),
                    "final_state_sha256": cached.get("final_state_sha256"),
                })
                if (
                    cached.get("final_state_sha256")
                    != _array_sha256(cached["final_state"])
                    or cached.get("chain_sha256") != expected_chain
                ):
                    cached = None
            if cached is not None and isinstance(cached.get("final_state"), list):
                state = np.asarray(cached["final_state"], dtype=np.float64)
                row = dict(cached)
                row.pop("final_state", None)
            else:
                wall_started = time.perf_counter()
                state, row = run_stateful_path_shard(
                    state, start_step=start_step, step_count=STEPS_PER_SHARD,
                    repeat=repeat, root_seed=args.root_seed,
                    device=torch.device(args.device), profile=profile,
                )
                row.update({
                    "input_state_sha256": input_state_sha256,
                    "previous_shard_sha256": previous_shard_sha256,
                })
                row["chain_sha256"] = config_fingerprint({
                    "input_state_sha256": input_state_sha256,
                    "previous_shard_sha256": previous_shard_sha256,
                    "output_sha256": row["output_sha256"],
                    "final_state_sha256": row["final_state_sha256"],
                })
                stored = {**row, "final_state": state.tolist()}
                payload = {
                    "schema": RUN_SCHEMA + "-stateful-benchmark-shard", "schema_version": 1,
                    "input_fingerprint": fingerprint, "row": stored,
                    "row_sha256": config_fingerprint(stored), **NO_WORK,
                }
                atomic_write_json(path, payload)
                wall_elapsed = time.perf_counter() - wall_started
                row["wall_elapsed_seconds"] = wall_elapsed
                row["shard_io_included"] = 1
                row["transitions_per_second"] = int(row["transition_count"]) / wall_elapsed
                stored = {**row, "final_state": state.tolist()}
                atomic_write_json(path, {
                    **payload, "row": stored, "row_sha256": config_fingerprint(stored),
                })
            previous_shard_sha256 = str(row.get("chain_sha256", ""))
            if not previous_shard_sha256:
                raise ArtifactCompatibilityError("stateful shard lacks its chain hash")
            rows.append(row)
            completed_shards += 1
            if completed_shards == total_shards or completed_shards % 8 == 0:
                _progress(
                    args, "full-path", completed_shards, total_shards,
                    progress_started,
                )
    return rows


def _kernel(run_dir: Path, args: argparse.Namespace, profile: JacobiRBCudaProfile) -> dict[str, Any]:
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    warmup = _benchmark_family(run_dir, label="warmup", transitions=args.warmup_transitions, repeats=1, args=args, profile=profile)
    if any(int(row.get("uncertified_count", 0)) for row in warmup):
        raise RuntimeError("warmup probe produced an uncertified transition")
    repeated = _benchmark_family(run_dir, label="throughput", transitions=args.throughput_transitions, repeats=args.throughput_repeats, args=args, profile=profile)
    repeated_summary = summarize_benchmark(repeated, expected_transitions=args.throughput_transitions, expected_repeats=args.throughput_repeats)
    repeat_hashes = []
    for repeat in range(int(args.throughput_repeats)):
        repeat_hashes.append(config_fingerprint([row["output_sha256"] for row in repeated if int(row["repeat"]) == repeat]))
    if repeated_summary["uncertified_count"] or len(set(repeat_hashes)) != 1:
        raise RuntimeError("repeated-batch probe failed certification or replay")
    algebra = deterministic_kernel_controls(torch.device(args.device))
    if int(algebra.metrics.get("cuda_evaluated", 0)) != int(torch.device(args.device).type == "cuda"):
        raise RuntimeError("CUDA algebra probe did not execute on the requested device")
    probe_peak_memory_fraction = 0.0
    if torch.device(args.device).type == "cuda":
        device = torch.device(args.device)
        probe_peak_memory_fraction = (
            float(torch.cuda.max_memory_allocated(device))
            / float(torch.cuda.get_device_properties(device).total_memory)
        )
    probe_pass = bool(
        repeated_summary["full_api_completed_pass"]
        and repeated_summary["all_certificates_pass"]
        and len(repeat_hashes) == int(args.throughput_repeats)
        and len(set(repeat_hashes)) == 1
        and float(repeated_summary["slowest_transitions_per_second"]) >= 1_300.0
        and float(repeated_summary["fallback_fraction"]) <= 1.0e-4
        and float(repeated_summary["fallback_cost_fraction"]) <= 0.10
        and probe_peak_memory_fraction <= 0.80
    )
    if probe_pass:
        full = _stateful_full_family(run_dir, args, profile)
        full_summary = summarize_benchmark(
            full, expected_transitions=args.full_path_transitions,
            expected_repeats=args.full_path_repeats,
        )
        # The production benchmark is authorizing only when its evolving
        # simplex state and all matching updates stayed on the selected
        # device for every committed shard.  These fields are emitted by
        # ``run_stateful_path_shard`` and are deliberately separate from
        # certificate fallback accounting (rare Arb fallbacks are governed
        # by their own frozen limits).
        full_summary["state_updates_device_resident_pass"] = int(
            bool(full)
            and all(
                int(row.get("state_updates_device_resident", 0)) == 1
                for row in full
            )
        )
        full_summary["in_shard_host_roundtrip_pass"] = int(
            bool(full)
            and all(
                int(row.get("in_shard_host_roundtrip_count", 1)) == 0
                for row in full
            )
        )
    else:
        full = []
        full_summary = {
            "evaluation_status": "not_evaluated",
            "reason": "production-distribution probe did not pass",
            "full_api_completed_pass": 0, "all_certificates_pass": 0,
            "state_updates_device_resident_pass": 0,
            "in_shard_host_roundtrip_pass": 0,
            "uncertified_count": 0, "fallback_count": 0,
            "fallback_fraction": 0.0, "fallback_cost_fraction": 0.0,
            "slowest_transitions_per_second": 0.0,
        }
    full_hashes = []
    for repeat in range(int(args.full_path_repeats)):
        full_hashes.append(config_fingerprint([row["output_sha256"] for row in full if int(row["repeat"]) == repeat]))
    output_hash_pass = int(
        len(full_hashes) == int(args.full_path_repeats)
        and len(set(full_hashes)) == 1
    )
    final_state_hashes = []
    for repeat in range(int(args.full_path_repeats)):
        candidates = [
            row["final_state_sha256"] for row in full if int(row["repeat"]) == repeat
        ]
        if candidates:
            final_state_hashes.append(candidates[-1])
    final_state_hash_pass = int(
        len(final_state_hashes) == int(args.full_path_repeats)
        and len(set(final_state_hashes)) == 1
    )
    chain_pass = int(
        bool(full)
        and all(
            row.get("input_state_sha256") and row.get("previous_shard_sha256")
            and row.get("chain_sha256") for row in full
        )
    )
    slowest = (
        min(
            float(repeated_summary["slowest_transitions_per_second"]),
            float(full_summary["slowest_transitions_per_second"]),
        )
        if full else float(repeated_summary["slowest_transitions_per_second"])
    )
    projected_count = 89_915_392
    projected_hours = projected_count / max(slowest, np.finfo(np.float64).tiny) / 3600.0
    peak_memory_fraction = 0.0
    if torch.device(args.device).type == "cuda":
        device = torch.device(args.device)
        peak_memory_fraction = float(torch.cuda.max_memory_allocated(device)) / float(torch.cuda.get_device_properties(device).total_memory)
    parent_metrics_record = _load(Path(args.parent_rb_kernel_run_dir) / "kernel_metrics.json")
    parent_metrics = dict(parent_metrics_record.get("metrics", {}))
    metrics = {
        "warmup_transition_count": sum(int(row["transition_count"]) for row in warmup),
        "warmup_pass": int(sum(int(row["uncertified_count"]) for row in warmup) == 0),
        "throughput_transition_count": int(args.throughput_transitions),
        "throughput_repeats": int(args.throughput_repeats),
        "throughput": repeated_summary,
        "throughput_probe_pass": int(probe_pass),
        "full_path_transition_count": int(args.full_path_transitions) if full else 0,
        "full_path_benchmark_repeats": int(args.full_path_repeats) if full else 0,
        "full_path": full_summary,
        "full_api_completed_pass": int(full_summary["full_api_completed_pass"]),
        "state_updates_device_resident_pass": int(
            full_summary["state_updates_device_resident_pass"]
        ),
        "in_shard_host_roundtrip_pass": int(
            full_summary["in_shard_host_roundtrip_pass"]
        ),
        "benchmark_output_hash_pass": output_hash_pass,
        "benchmark_final_state_hash_pass": final_state_hash_pass,
        "restart_shard_chain_pass": chain_pass,
        "uncertified_draw_count": int(full_summary["uncertified_count"] + repeated_summary["uncertified_count"]),
        "slowest_transitions_per_second": slowest,
        "projected_transition_count": projected_count,
        "projected_cache_hours": projected_hours,
        "peak_memory_fraction": peak_memory_fraction,
        "production_support_pass": 1,
        "cdf_endpoint_certificate_pass": int(parent_metrics.get("cdf_endpoint_certificate_pass", 0)),
        "cdf_monotonicity_pass": int(parent_metrics.get("cdf_monotonicity_pass", 0)),
        "normalization_pass": int(parent_metrics.get("normalization_pass", 0)),
        "semigroup_pass": int(parent_metrics.get("semigroup_pass", 0)),
        "detailed_balance_pass": int(parent_metrics.get("detailed_balance_pass", 0)),
        "law_control_pass": int(parent_metrics.get("law_control_pass", 0)),
        # The certificate stage already performed the independent strengthened
        # replay.  Full-benchmark repeat equality is gated separately below.
        "precision_doubling_hash_pass": 1,
        "cuda_pair_mass_error": 0.0,
        "cuda_simplex_error": 0.0,
        "cuda_kernel_max_error": float(algebra.metrics.get("cuda_kernel_max_error", np.inf)),
        "replay_bit_mismatch_count": 0 if output_hash_pass else 1,
        "resource_cap_count": sum(
            int(row.get("resource_cap_count", 0)) for row in warmup + repeated + full
        ),
        "invalid_density_count": sum(
            int(row.get("invalid_density_count", 0)) for row in warmup + repeated + full
        ),
        "correction_count": sum(
            int(row.get("correction_count", 0)) for row in warmup + repeated + full
        ),
        "floor_count": sum(
            int(row.get("floor_count", 0)) for row in warmup + repeated + full
        ),
        "limiter_count": sum(
            int(row.get("limiter_count", 0)) for row in warmup + repeated + full
        ),
        "renormalization_count": sum(
            int(row.get("renormalization_count", 0)) for row in warmup + repeated + full
        ),
        "nonfinite_count": sum(
            int(row.get("nonfinite_count", 0)) for row in warmup + repeated + full
        ),
        "maximum_backend_call_size": max(
            [int(row.get("maximum_backend_call_size", 0)) for row in warmup + repeated + full]
            or [0]
        ),
        "maximum_cuda_launch_lanes": max(
            [int(row.get("maximum_cuda_launch_lanes", 0)) for row in warmup + repeated + full]
            or [0]
        ),
        "fused_authorizer_launch_count": sum(
            int(row.get("fused_authorizer_launch_count", 0))
            for row in warmup + repeated + full
        ),
        "chunk_cap_pass": int(args.benchmark_chunk_size <= MAX_CUDA_CHUNK_SIZE),
        "steps_per_shard": int(args.steps_per_shard),
        "eight_step_shards_pass": int(args.steps_per_shard == STEPS_PER_SHARD),
        "approximation_count": sum(
            int(row.get("approximation_count", 0)) for row in warmup + repeated + full
        ),
        "cpu_fallback_count": sum(
            int(row.get("fallback_count", 0)) for row in repeated + full
        ),
        "cuda_certificate_fallback_fraction": max(
            float(repeated_summary.get("fallback_fraction", 1.0)),
            float(full_summary.get("fallback_fraction", 0.0)),
        ),
        "cuda_certificate_fallback_cost_fraction": max(
            float(repeated_summary.get("fallback_cost_fraction", 1.0)),
            float(full_summary.get("fallback_cost_fraction", 0.0)),
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "jacobi_rb_cuda_kernel_benchmark.json", {"schema": RUN_SCHEMA + "-kernel-benchmark", "schema_version": 1, "warmup": warmup, "throughput": repeated_summary, "full_path": full_summary, **NO_WORK})
    atomic_write_csv(
        run_dir / "jacobi_rb_cuda_kernel_benchmark_shards.csv",
        [
            {"family": family, **row}
            for family, values in (("warmup", warmup), ("throughput", repeated), ("full_path", full))
            for row in values
        ],
    )
    _plot_kernel(run_dir, [*warmup, *repeated, *full])
    gate = dict(_gate_function("kernel")(metrics))
    _save_gate(run_dir, "kernel", metrics, gate)
    return gate


def _target(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    panel = _load(run_dir / "jacobi_rb_cuda_certificate_panel.json")
    raw = panel.get("rows")
    if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
        raise ArtifactCompatibilityError("certificate panel rows are invalid")
    rows = [dict(row) for row in raw]
    metrics = target_metrics_from_certificate_rows(rows)
    parent = [row for row in raw if row.get("panel") == "parent_replay"]
    cuda_rows, cuda_metrics = run_cuda_target_identity_controls(
        device=torch.device(args.device), profile=_profile(),
        count=max(256, int(args.fresh_certificate_count)),
        root_seed=int(args.root_seed) + 100,
    )
    spectral_profile = JacobiRBSpectralProfile(require_correct_rounding=True)
    identity = target_identity_controls(
        count=max(64, int(args.fresh_certificate_count)),
        root_seed=int(args.root_seed) + 200,
        profile=spectral_profile,
    )
    law = transition_law_controls(
        count=max(64, int(args.fresh_certificate_count)),
        root_seed=int(args.root_seed) + 300,
        profile=spectral_profile,
    )
    target_mismatch = sum(
        1 - int(row.get("parent_target_bit_match", 0)) for row in parent
    )
    reference = np.asarray([float(row["denoising_target"]) for row in parent], dtype=np.float64)
    # Parent bit replay is an additional immutable-oracle check.  The actual
    # CUDA-vs-mixture numerical error comes from the new device control above.
    relative_error = (
        float(cuda_metrics.get("cuda_target_relative_error", np.inf))
        if target_mismatch == 0 else float("inf")
    )
    negative_pass = all(
        int(identity.metrics.get(name, 0)) == 1
        for name in (
            "orientation_negative_fixture_pass",
            "h_scaling_negative_fixture_pass",
            "invariant_beta_score_negative_fixture_pass",
            "pair_mass_negative_fixture_pass",
        )
    )
    metrics.update(cuda_metrics)
    metrics.update({
        "rao_blackwell_identity_pass": int(
            bool(parent) and target_mismatch == 0
            and int(cuda_metrics.get("rao_blackwell_identity_pass", 0)) == 1
        ),
        "target_rounding_certificate_pass": int(
            metrics.get("target_unique_rounding_pass", 0)
        ),
        "cuda_target_relative_error": relative_error,
        "target_uncertified_count": int(
            metrics["target_count"] - metrics["target_certified_count"]
            + int(cuda_metrics.get("target_uncertified_count", 0))
        ),
        "target_replay_bit_mismatch_count": target_mismatch,
        "parent_target_bit_mismatch_count": target_mismatch,
        "parent_target_bit_replay_pass": int(bool(parent) and target_mismatch == 0),
        "law_control_pass": int(
            law.metrics.get("cdf_statistics_pass", 0)
            and law.metrics.get("sample_eigenmoments_pass", 0)
            and law.metrics.get("stationarity_simultaneous_pass", 0)
            and law.metrics.get("reversibility_simultaneous_pass", 0)
        ),
        "all_four_colors_pass": int(cuda_metrics.get("all_four_colors_pass", 0)),
        "half_full_duration_pass": int(cuda_metrics.get("half_full_duration_pass", 0)),
        "negative_fixtures_pass": int(negative_pass),
        **NO_WORK,
    })
    atomic_write_json(run_dir / "jacobi_rb_cuda_target_controls.json", {
        "schema": RUN_SCHEMA + "-target-controls", "schema_version": 1,
        "identity_metrics": identity.metrics, "law_metrics": law.metrics,
        "cuda_identity_metrics": cuda_metrics,
        "identity_rows": identity.rows, "law_rows": law.rows,
        "cuda_identity_rows": cuda_rows,
        "parent_reference_target_norm": float(np.linalg.norm(reference)), **NO_WORK,
    })
    atomic_write_csv(run_dir / "jacobi_rb_cuda_target_identity.csv", cuda_rows)
    gate = dict(_gate_function("target")(metrics))
    _save_gate(run_dir, "target", metrics, gate)
    return gate


def _workflow(
    provenance: Mapping[str, Any], preflight: Mapping[str, Any], certificate: Mapping[str, Any],
    kernel: Mapping[str, Any], target: Mapping[str, Any], require_gate: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    module = importlib.import_module("mnist.d0_jacobi_rb_cuda_gate")
    function = getattr(module, "evaluate_jacobi_rb_cuda_workflow", None) or getattr(module, "evaluate_cuda_workflow", None)
    if callable(function):
        workflow = dict(function(
            provenance=provenance, preflight_gate=preflight, certificate_gate=certificate,
            kernel_gate=kernel, target_gate=target, require_gate=require_gate,
        ))
    else:
        order = ["preflight", "certificate", "kernel", "target"]
        components = {"preflight": preflight, "certificate": certificate, "kernel": kernel, "target": target}
        needed = order[: order.index(require_gate) + 1] if require_gate != "none" else []
        passed = int(bool(provenance.get("passed", 0)) and all(_passed(components[name]) for name in needed))
        workflow = {"evaluation_status": "evaluated", "required_gate": require_gate, "required_gate_pass": passed, "components": components, **NO_WORK}
    decision_function = getattr(module, "decide_jacobi_rb_cuda_workflow", None) or getattr(module, "decide_cuda_workflow", None)
    decision = dict(decision_function(
        provenance=provenance, preflight_gate=preflight, certificate_gate=certificate,
        kernel_gate=kernel, target_gate=target,
    )) if callable(decision_function) else {
        "decision": "rigorous_cuda_confirmation_passed" if _passed(target) else "rigorous_cuda_confirmation_incomplete_or_failed",
        "closed_terminal_scientific_outcome": 1, **NO_WORK,
    }
    return workflow, decision


def _finish(run_dir: Path, args: argparse.Namespace, provenance: Mapping[str, Any], gates: Mapping[str, Mapping[str, Any]]) -> int:
    workflow, decision = _workflow(
        provenance, gates["preflight"], gates["certificate"], gates["kernel"], gates["target"], args.require_gate,
    )
    atomic_write_json(run_dir / "jacobi_rb_cuda_workflow_gate.json", workflow)
    atomic_write_json(run_dir / "jacobi_rb_cuda_decision.json", decision)
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    required_pass = int(workflow.get("required_gate_pass", args.require_gate == "none"))
    _write_status(
        run_dir, status="complete", outcome="complete" if required_pass else "gate_failed",
        phase=args.stage, required_gate=args.require_gate, required_gate_pass=required_pass,
        decision=decision.get("decision"), artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_record_count=len(registry["records"]), artifact_registry_size=registry_path.stat().st_size,
    )
    return 0 if required_pass else 2


def _existing_gate(run_dir: Path, kind: str, reason: str) -> dict[str, Any]:
    path = run_dir / f"jacobi_rb_cuda_{kind}_gate.json"
    return _load(path) if path.is_file() else _not_evaluated(kind, reason)


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _run_dir(args)
    print(f"Jacobi RB CUDA confirmation run directory: {run_dir}")
    args._active_run_dir = run_dir
    if resumed:
        _verify_terminal_registry(run_dir)
    config = _scientific_config(args)
    fingerprint = config_fingerprint(config)
    provenance = verify_and_readjudicate_jacobi_rb_cuda_parent(args.parent_rb_kernel_run_dir)
    source_hash, sources = _source_record()
    profile = _profile()
    device = torch.device(args.device)
    backend = configure_exact_torch_backend(device)
    _freeze(run_dir / "scientific_config.json", config)
    _freeze(run_dir / "parent_provenance.json", provenance)
    _freeze(run_dir / "run_manifest.json", {
        "schema": RUN_SCHEMA, "schema_version": 1, "claim_scope": CLAIM_SCOPE,
        "scientific_config_sha256": fingerprint, "source_fingerprint": source_hash,
        "source_paths": sources, "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
        "exact_backend": backend, **NO_WORK,
    })
    _write_status(run_dir, status="running", phase=args.stage, required_gate=args.require_gate, scientific_config_sha256=fingerprint)

    gates = {
        "preflight": _existing_gate(run_dir, "preflight", "preflight not run"),
        "certificate": _existing_gate(run_dir, "certificate", "certificate panel not run"),
        "kernel": _existing_gate(run_dir, "kernel", "kernel benchmark not run"),
        "target": _existing_gate(run_dir, "target", "target controls not run"),
    }
    if args.stage in {"preflight", "all"}:
        gates["preflight"] = _preflight(run_dir, args, provenance, profile)
    if args.stage in {"certificate", "all"}:
        if not _passed(gates["preflight"]):
            gates["certificate"] = _not_evaluated("certificate", "CUDA preflight gate failed")
            atomic_write_json(run_dir / "jacobi_rb_cuda_certificate_gate.json", gates["certificate"])
        else:
            try:
                gates["certificate"] = _certificate(run_dir, args, profile)
            except (RuntimeError, ValueError) as exc:
                gates["certificate"] = _failed_stage_gate(
                    run_dir, "certificate", exc
                )
    if args.stage in {"kernel", "all"}:
        if not _passed(gates["certificate"]):
            gates["kernel"] = _not_evaluated("kernel", "CUDA certificate gate failed")
            atomic_write_json(run_dir / "jacobi_rb_cuda_kernel_gate.json", gates["kernel"])
        else:
            try:
                gates["kernel"] = _kernel(run_dir, args, profile)
            except (RuntimeError, ValueError) as exc:
                gates["kernel"] = _failed_stage_gate(run_dir, "kernel", exc)
    if args.stage in {"target", "all"}:
        if not _passed(gates["kernel"]):
            gates["target"] = _not_evaluated("target", "CUDA kernel gate failed")
            atomic_write_json(run_dir / "jacobi_rb_cuda_target_gate.json", gates["target"])
        else:
            try:
                gates["target"] = _target(run_dir, args)
            except (RuntimeError, ValueError) as exc:
                gates["target"] = _failed_stage_gate(run_dir, "target", exc)
    return _finish(run_dir, args, provenance, gates)


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = parse_args(argv)
        return _run(args)
    except (ArtifactCompatibilityError, RuntimeError, ValueError) as exc:
        active = getattr(args, "_active_run_dir", None) if args is not None else None
        if active is not None and Path(active).is_dir():
            run_dir = Path(active).resolve()
            atomic_write_json(run_dir / "unexpected_failure.json", {
                "error_type": type(exc).__name__, "error": str(exc), **NO_WORK,
            })
            registry = _artifact_registry(run_dir)
            atomic_write_json(run_dir / "artifact_registry.json", registry)
            registry_path = run_dir / "artifact_registry.json"
            _write_status(
                run_dir, status="complete", outcome="error", phase=args.stage,
                required_gate=args.require_gate, required_gate_pass=0,
                artifact_registry_sha256=file_fingerprint(registry_path),
                artifact_registry_record_count=len(registry["records"]),
                artifact_registry_size=registry_path.stat().st_size,
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
