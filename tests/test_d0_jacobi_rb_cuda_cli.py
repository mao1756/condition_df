from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mnist.diag_d0_jacobi_rb_cuda_confirmation as cli
from mnist.d0_jacobi_artifacts import file_fingerprint


def test_cli_exact_parent_argument_defaults_and_stage_prefixes() -> None:
    args = cli.parse_args(["--parent-rb-kernel-run-dir", "parent", "--stage", "preflight"])
    assert args.root_seed == 261_131
    assert args.runs_root == Path("runs/experiment12_d0_jacobi_rb_cuda_confirmation")
    assert args.parent_replay_count == 294
    assert args.fresh_certificate_count == 512
    assert args.full_path_transitions == 1_404_928
    assert args.full_path_repeats == 3
    assert args.benchmark_chunk_size == 4_096
    assert args.steps_per_shard == 8
    with pytest.raises(SystemExit):
        cli.parse_args(["--parent-jacobi-rb-run-dir", "parent"])
    with pytest.raises(SystemExit):
        cli.parse_args([
            "--parent-rb-kernel-run-dir", "parent", "--stage", "certificate",
        ])
    with pytest.raises(SystemExit):
        cli.parse_args([
            "--parent-rb-kernel-run-dir", "parent", "--stage", "preflight",
            "--require-gate", "certificate",
        ])


def test_production_overrides_fail_and_test_only_cannot_require_gate() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([
            "--parent-rb-kernel-run-dir", "parent", "--fresh-certificate-count", "8",
        ])
    reduced = cli.parse_args([
        "--parent-rb-kernel-run-dir", "parent", "--fresh-certificate-count", "8",
        "--parent-replay-count", "2", "--warmup-transitions", "4",
        "--throughput-transitions", "8", "--throughput-repeats", "1",
        "--full-path-transitions", "16", "--full-path-repeats", "1",
        "--test-only-reduced-workload", "--device", "cpu",
    ])
    assert reduced.test_only_reduced_workload
    with pytest.raises(SystemExit):
        cli.parse_args([
            "--parent-rb-kernel-run-dir", "parent", "--test-only-reduced-workload",
            "--device", "cpu", "--require-gate", "preflight",
        ])


def test_controls_only_imports_no_trainer_strang_or_reverse_sampler() -> None:
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {
        "mnist.experiment12_d0",
        "mnist.eulerian_flux_mnist",
        "mnist.d0_one_image_sampler",
    }
    assert imported.isdisjoint(forbidden)
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert source.index('if args.stage in {"kernel", "all"}') < source.index(
        'if args.stage in {"target", "all"}'
    )
    assert 'if not _passed(gates["kernel"])' in source


def test_artifact_registry_excludes_terminal_self_reference(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifact_registry.json").write_text("{}", encoding="utf-8")
    registry = cli._artifact_registry(tmp_path)
    assert set(registry["records"]) == {"evidence.json"}
    assert registry["physical_training_performed"] == 0
    assert registry["reverse_sampling_performed"] == 0


def test_resume_source_fingerprint_binds_fused_authorizer() -> None:
    digest, paths = cli._source_record()
    assert len(digest) == 64
    names = {Path(path).name for path in paths}
    assert "d0_jacobi_rb_cuda.py" in names
    assert "d0_jacobi_rb_cuda_fused.py" in names
    assert "d0_jacobi_rb_cuda_certificate.py" in names


def test_failed_stage_preserves_diagnostic_resource_cap(tmp_path: Path) -> None:
    from mnist.d0_jacobi_rb_spectral import JacobiRBCertificationError

    error = JacobiRBCertificationError(
        "candidate-local Arb target remained unresolved",
        {
            "failure_kind": "target_interval",
            "maximum_modes": 16_384,
            "precision_bits": 8_192,
        },
    )
    gate = cli._failed_stage_gate(tmp_path, "certificate", error)
    failure = cli._load(tmp_path / "certificate_failure.json")
    assert failure["resource_cap_count"] == 1
    assert failure["failure_diagnostics"]["maximum_modes"] == 16_384
    assert gate["passed"] == 0


def test_registered_benchmark_shard_corruption_is_recoverable(tmp_path: Path) -> None:
    shard = tmp_path / "cuda_benchmark_shards" / "full-path" / "shard.json"
    shard.parent.mkdir(parents=True)
    shard.write_text("corrupt", encoding="utf-8")
    registry = {
        "terminal_files_excluded_to_avoid_self_reference": [
            "artifact_registry.json", "run_status.json",
        ],
        "records": {
            shard.relative_to(tmp_path).as_posix(): {"sha256": "0" * 64, "size": 1}
        },
    }
    (tmp_path / "artifact_registry.json").write_text(
        __import__("json").dumps(registry), encoding="utf-8"
    )
    (tmp_path / "run_status.json").write_text(
        __import__("json").dumps({
            "status": "complete",
            "artifact_registry_sha256": file_fingerprint(tmp_path / "artifact_registry.json"),
        }),
        encoding="utf-8",
    )
    cli._verify_terminal_registry(tmp_path)
