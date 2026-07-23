from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import mnist.diag_d0_jacobi_rb_cuda_multipath_confirmation as cli


def test_cli_frozen_defaults_and_stage_dependencies() -> None:
    args = cli.parse_args(
        ["--parent-cuda-run-dir", "parent", "--stage", "preflight"]
    )
    assert args.root_seed == 261_141
    assert args.pilot_outer_steps == 64
    assert args.full_outer_steps == 512
    assert args.steps_per_shard == 8
    with pytest.raises(SystemExit):
        cli.parse_args(["--parent-cuda-run-dir", "parent", "--stage", "pilot"])
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-cuda-run-dir",
                "parent",
                "--stage",
                "preflight",
                "--require-gate",
                "pilot",
            ]
        )


def test_production_workload_is_frozen_and_reduced_runs_cannot_gate() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            ["--parent-cuda-run-dir", "parent", "--pilot-outer-steps", "8"]
        )
    reduced = cli.parse_args(
        [
            "--parent-cuda-run-dir",
            "parent",
            "--device",
            "cpu",
            "--pilot-outer-steps",
            "8",
            "--full-outer-steps",
            "8",
            "--pilot-repeats-per-group",
            "1",
            "--full-repeats-per-group",
            "1",
            "--test-only-reduced-workload",
        ]
    )
    assert reduced.test_only_reduced_workload
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-cuda-run-dir",
                "parent",
                "--device",
                "cpu",
                "--test-only-reduced-workload",
                "--require-gate",
                "preflight",
            ]
        )


def test_projection_formula_uses_six_b10_groups_and_one_b4_group(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "kernel_metrics.json").write_text(
        '{"metrics":{"cuda_kernel_max_error":0.0}}', encoding="utf-8"
    )
    rows = []
    transitions = {10: 10 * 512 * cli.TRANSITIONS_PER_PATH_STEP, 4: 4 * 512 * cli.TRANSITIONS_PER_PATH_STEP}
    walls = {10: 100.0, 4: 50.0}
    for group_size in (10, 4):
        for repeat in range(3):
            rows.append(
                {
                    "group_size": group_size,
                    "repeat": repeat,
                    "start_step": 0,
                    "chain_sha256": "a" * 64,
                    "batch_output_sha256": "o",
                    "batch_final_state_sha256": "s",
                    "batch_certificate_sha256": "c",
                    "path_records": [
                        {"path_id": index} for index in range(group_size)
                    ],
                    "diagnostics": {
                        "transition_count": transitions[group_size],
                        "certified_count": transitions[group_size],
                        "group_sizes": [group_size],
                        "state_updates_device_resident": 1,
                        "evolving_state_host_roundtrip_count": 0,
                        "maximum_cuda_launch_lanes": group_size * 392,
                        "maximum_mass_error": 0.0,
                        "fallback_elapsed_seconds": 0.0,
                        "elapsed_seconds": walls[group_size],
                    },
                    "wall_elapsed_seconds": walls[group_size],
                    "commit_reuses_packed_host_snapshot": 1,
                    "state_npz_and_metadata_commit_included": 1,
                }
            )
    metrics = cli._performance_metrics(
        rows, outer_steps=512, repeats=3, pilot=False, parent_dir=parent
    )
    assert metrics["projected_cache_seconds"] == pytest.approx(650.0)
    assert metrics["projected_cache_hours"] == pytest.approx(650.0 / 3600.0)
    assert metrics["total_full_benchmark_transitions"] == 59_006_976
    assert metrics["b10_transitions_per_repeat"] == 14_049_280
    assert metrics["b4_transitions_per_repeat"] == 5_619_712
    # This synthetic fixture deliberately has one row per repeat rather than
    # the required 64 shards, so the fail-closed shard-count check remains 0.
    assert metrics["completed_shard_count"] == 6


def test_controls_only_cli_imports_no_trainer_strang_or_reverse_sampler() -> None:
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported.isdisjoint(
        {
            "mnist.experiment12_d0",
            "mnist.eulerian_flux_mnist",
            "mnist.d0_one_image_sampler",
        }
    )
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert source.index('if args.stage in {"kernel", "all"}') < source.index(
        'if args.stage in {"target", "all"}'
    )


def test_source_fingerprint_binds_parent_and_new_scheduler_modules() -> None:
    parent = Path(
        "runs/experiment12_d0_jacobi_rb_cuda_confirmation"
    ) / "20260722-233246_codex-final-validation-certified-cuda-dd-jacobi-rb"
    if not parent.is_dir():
        pytest.skip("immutable CUDA parent unavailable")
    digest, paths = cli._source_record(parent)
    names = {Path(path).name for path in paths}
    assert len(digest) == 64
    assert {
        "d0_jacobi_rb_cuda_multipath.py",
        "d0_jacobi_rb_cuda_multipath_gate.py",
        "d0_jacobi_rb_cuda_multipath_provenance.py",
        "diag_d0_jacobi_rb_cuda_multipath_confirmation.py",
    } <= names


def test_finalized_npz_shard_round_trip_and_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.json"
    state_path = path.with_suffix(".npz")
    final = np.arange(2 * 784, dtype=np.float64).reshape(2, 784)
    cli._atomic_write_state_npz(state_path, final)
    fingerprint = "f" * 64
    input_hash = "i" * 64
    previous = "p" * 64
    row = {
        "group_size": 2,
        "shard_input_fingerprint": fingerprint,
        "input_states_sha256": input_hash,
        "previous_shard_sha256": previous,
        "batch_output_sha256": "o" * 64,
        "batch_final_state_sha256": "s" * 64,
        "batch_certificate_sha256": "c" * 64,
        "persisted_final_states_sha256": cli._array_sha256(final),
        "state_npz_name": state_path.name,
        "state_npz_sha256": cli.file_fingerprint(state_path),
        "state_npz_size": state_path.stat().st_size,
        "commit_reuses_packed_host_snapshot": 1,
        "state_npz_and_metadata_commit_included": 1,
        "conservative_timing_bound_pass": 1,
        "wall_elapsed_seconds": 1.0,
        "transitions_per_second": 2.0,
        "timing_finalization_attempt": 1,
        "timing_allowance_seconds": 0.1,
        "diagnostics": {"transition_count": 2},
    }
    row["chain_sha256"] = cli.config_fingerprint(
        {
            "input_states_sha256": input_hash,
            "previous_shard_sha256": previous,
            "batch_output_sha256": row["batch_output_sha256"],
            "batch_final_state_sha256": row["batch_final_state_sha256"],
            "batch_certificate_sha256": row["batch_certificate_sha256"],
            "state_npz_sha256": row["state_npz_sha256"],
            "state_npz_size": row["state_npz_size"],
        }
    )

    def write() -> None:
        cli.atomic_write_json(
            path,
            {
                "input_fingerprint": fingerprint,
                "row": row,
                "row_sha256": cli.config_fingerprint(row),
            },
        )

    write()
    loaded = cli._load_shard(path, fingerprint, input_hash, previous)
    assert loaded is not None
    assert np.array_equal(loaded[1], final)

    row["conservative_timing_bound_pass"] = 0
    write()
    assert cli._load_shard(path, fingerprint, input_hash, previous) is None

    row["conservative_timing_bound_pass"] = 1
    write()
    state_path.write_bytes(b"corrupt")
    assert cli._load_shard(path, fingerprint, input_hash, previous) is None


def test_pilot_and_kernel_throughput_plots_are_written(tmp_path: Path) -> None:
    rows = [
        {
            "group_size": group,
            "repeat": repeat,
            "transitions_per_second": 1_400.0 - repeat,
            "wall_elapsed_seconds": 2.0 + repeat,
        }
        for group in (10, 4)
        for repeat in range(3)
    ]
    cli._plot_performance(tmp_path, "pilot", rows)
    cli._plot_performance(tmp_path, "kernel", rows)
    assert (tmp_path / "multipath_pilot_throughput.png").stat().st_size > 0
    assert (tmp_path / "multipath_kernel_throughput.png").stat().st_size > 0


def test_rejected_corrupt_resume_is_byte_for_byte_immutable(tmp_path: Path) -> None:
    registry = tmp_path / "artifact_registry.json"
    status = tmp_path / "run_status.json"
    sentinel = tmp_path / "sentinel.bin"
    registry.write_text('{"records":{}}', encoding="utf-8")
    status.write_text(
        '{"status":"complete","artifact_registry_sha256":"wrong"}',
        encoding="utf-8",
    )
    sentinel.write_bytes(b"immutable")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    code = cli.main(
        [
            "--parent-cuda-run-dir",
            "not-consulted",
            "--resume-run-dir",
            str(tmp_path),
            "--stage",
            "report",
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        ]
    )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert code == 2
    assert after == before
    assert not (tmp_path / "unexpected_failure.json").exists()


@pytest.mark.parametrize("stage", ["kernel", "target", "report"])
def test_later_stage_cannot_reseal_corrupt_completed_pilot_shard(
    tmp_path: Path, stage: str
) -> None:
    shard_dir = tmp_path / "multipath_shards" / "pilot"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "b10-repeat-00-steps-000-007.json"
    shard.write_bytes(b"original committed pilot evidence")
    registry = cli._artifact_registry(tmp_path)
    cli.atomic_write_json(tmp_path / "artifact_registry.json", registry)
    registry_path = tmp_path / "artifact_registry.json"
    cli.atomic_write_json(
        tmp_path / "run_status.json",
        {
            "status": "complete",
            "artifact_registry_sha256": cli.file_fingerprint(registry_path),
        },
    )
    # Corrupt a prerequisite only after the terminal registry was sealed.
    shard.write_bytes(b"corrupt completed pilot evidence")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    code = cli.main(
        [
            "--parent-cuda-run-dir",
            "not-consulted",
            "--resume-run-dir",
            str(tmp_path),
            "--stage",
            stage,
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        ]
    )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert code == 2
    assert after == before
    assert not (tmp_path / "unexpected_failure.json").exists()


def test_stage_failure_writes_evidence_before_gate_failure(tmp_path: Path) -> None:
    gate = cli._failed_stage_gate(tmp_path, "pilot", RuntimeError("boom"))
    assert gate["passed"] == 0
    assert (tmp_path / "pilot_failure.json").is_file()
    assert (tmp_path / "multipath_pilot_gate.json").is_file()


def test_performance_metrics_recompute_and_reject_tampered_shard_chain(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent-chain"
    parent.mkdir()
    (parent / "kernel_metrics.json").write_text(
        '{"metrics":{"cuda_kernel_max_error":0.0}}', encoding="utf-8"
    )
    rows: list[dict] = []
    for group_size, offset in ((10, 0), (4, 100)):
        fingerprint = f"fingerprint-{group_size}"
        input_hash = f"initial-{group_size}"
        previous = cli.config_fingerprint(
            {
                "kind": "multipath-genesis",
                "fingerprint": fingerprint,
                "initial_states_sha256": input_hash,
            }
        )
        for start_step in (0, 8):
            persisted = f"persisted-{group_size}-{start_step}"
            row = {
                "group_size": group_size,
                "repeat": 0,
                "start_step": start_step,
                "shard_input_fingerprint": fingerprint,
                "input_states_sha256": input_hash,
                "previous_shard_sha256": previous,
                "persisted_final_states_sha256": persisted,
                "batch_output_sha256": f"output-{group_size}-{start_step}",
                "batch_final_state_sha256": f"state-{group_size}-{start_step}",
                "batch_certificate_sha256": f"cert-{group_size}-{start_step}",
                "state_npz_sha256": f"npz-{group_size}-{start_step}",
                "state_npz_size": 10,
                "path_records": [
                    {"path_id": 10_000 + offset + index}
                    for index in range(group_size)
                ],
                "diagnostics": {
                    "transition_count": group_size
                    * 8
                    * cli.TRANSITIONS_PER_PATH_STEP,
                    "certified_count": group_size
                    * 8
                    * cli.TRANSITIONS_PER_PATH_STEP,
                    "group_sizes": [group_size],
                    "state_updates_device_resident": 1,
                    "evolving_state_host_roundtrip_count": 0,
                    "maximum_cuda_launch_lanes": group_size * 392,
                    "maximum_mass_error": 0.0,
                },
                "wall_elapsed_seconds": 1.0,
                "commit_reuses_packed_host_snapshot": 1,
                "state_npz_and_metadata_commit_included": 1,
            }
            row["chain_sha256"] = cli.config_fingerprint(
                {
                    "input_states_sha256": input_hash,
                    "previous_shard_sha256": previous,
                    "batch_output_sha256": row["batch_output_sha256"],
                    "batch_final_state_sha256": row["batch_final_state_sha256"],
                    "batch_certificate_sha256": row["batch_certificate_sha256"],
                    "state_npz_sha256": row["state_npz_sha256"],
                    "state_npz_size": row["state_npz_size"],
                }
            )
            rows.append(row)
            previous = row["chain_sha256"]
            input_hash = persisted
    metrics = cli._performance_metrics(
        rows, outer_steps=16, repeats=1, pilot=True, parent_dir=parent
    )
    assert metrics["restart_shard_chain_pass"] == 1
    assert metrics["group_path_id_disjoint_pass"] == 1

    tail = next(
        row for row in rows
        if row["group_size"] == 10 and row["start_step"] == 8
    )
    tail["previous_shard_sha256"] = "tampered"
    tail["chain_sha256"] = cli.config_fingerprint(
        {
            "input_states_sha256": tail["input_states_sha256"],
            "previous_shard_sha256": tail["previous_shard_sha256"],
            "batch_output_sha256": tail["batch_output_sha256"],
            "batch_final_state_sha256": tail["batch_final_state_sha256"],
            "batch_certificate_sha256": tail["batch_certificate_sha256"],
            "state_npz_sha256": tail["state_npz_sha256"],
            "state_npz_size": tail["state_npz_size"],
        }
    )
    tampered = cli._performance_metrics(
        rows, outer_steps=16, repeats=1, pilot=True, parent_dir=parent
    )
    assert tampered["restart_shard_chain_pass"] == 0
