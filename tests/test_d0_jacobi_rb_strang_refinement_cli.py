from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mnist.diag_d0_jacobi_rb_strang_refinement as cli


def _base(*extra: str) -> list[str]:
    return ["--parent-multipath-run-dir", "parent", *extra]


def test_cli_frozen_defaults_and_stage_dependencies() -> None:
    args = cli.parse_args(_base("--stage", "preflight"))
    assert args.root_seed == 261_151
    assert args.sample_steps == (128, 256, 512, 1024, 2048)
    assert args.stationarity_panel_paths == 128
    assert args.bootstrap_reps == 20_000
    assert args.steps_per_shard == 8
    assert args.lambda_mix == pytest.approx(0.35)
    assert args.label == 3
    assert args.class_index == 0

    for stage in ("power", "refinement", "report"):
        with pytest.raises(SystemExit):
            cli.parse_args(_base("--stage", stage))
    with pytest.raises(SystemExit):
        cli.parse_args(
            _base("--stage", "preflight", "--require-gate", "power")
        )
    cli.parse_args(
        _base(
            "--stage",
            "refinement",
            "--resume-run-dir",
            "run",
            "--require-gate",
            "refinement",
        )
    )


def test_production_workload_is_frozen_and_reduced_runs_cannot_gate() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(_base("--bootstrap-reps", "4"))
    reduced = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--stationarity-panel-paths",
            "2",
            "--stationarity-transitions-per-path",
            "2",
            "--pilot-main-paths",
            "2",
            "--pilot-reference-paths",
            "1",
            "--bootstrap-reps",
            "8",
            "--test-only-reduced-workload",
        )
    )
    assert reduced.test_only_reduced_workload
    with pytest.raises(SystemExit):
        cli.parse_args(
            _base(
                "--device",
                "cpu",
                "--test-only-reduced-workload",
                "--require-gate",
                "preflight",
            )
        )


def test_source_image_selection_is_exact_and_training_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image0 = np.arange(1, 785, dtype=np.float64)
    image0 /= image0.sum()
    image1 = image0[::-1].copy()
    dataset = SimpleNamespace(
        train_images=np.stack((image0, image1)),
        train_labels=np.asarray([3, 7], dtype=np.int64),
    )
    monkeypatch.setattr(cli, "load_mnist_measure_dataset", lambda *a, **k: dataset)
    args = cli.parse_args(
        _base("--stage", "preflight", "--test-only-reduced-workload", "--device", "cpu")
    )
    record = cli._source_image(args)
    expected = 0.65 * image0 + 0.35 / 784.0
    expected /= expected.sum()
    assert record["dataset_index"] == 0
    assert record["label"] == 3
    assert record["image"].dtype == np.float64
    assert np.allclose(record["mixed_target"], expected)
    assert record["image_sha256"] == cli._measure_digest(image0)
    assert record["mixed_target_sha256"] == cli._measure_digest(expected)


def test_source_image_hash_gate_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full(784, 1.0 / 784.0, dtype=np.float64)
    dataset = SimpleNamespace(
        train_images=image[None, :],
        train_labels=np.asarray([3], dtype=np.int64),
    )
    monkeypatch.setattr(cli, "load_mnist_measure_dataset", lambda *a, **k: dataset)
    args = cli.parse_args(_base("--stage", "preflight"))
    with pytest.raises(cli.ArtifactCompatibilityError, match="source image SHA-256"):
        cli._source_image(args)


def test_powered_stationarity_panel_is_frozen_and_tamper_evident(
    tmp_path: Path,
) -> None:
    seed = 991
    path_id_start = 700_000
    panel, _, payload = cli._powered_stationarity_panel(
        path_count=4,
        transitions_per_path=2,
        root_seed=seed,
        reps=100,
        path_id_start=path_id_start,
    )
    npz_path = tmp_path / "powered_stationarity_panel_a.npz"
    metadata_path = tmp_path / "stationarity_panel_a.json"
    cli._atomic_write_npz(npz_path, **payload)
    panel = {
        **panel,
        "npz_name": npz_path.name,
        "npz_sha256": cli.file_fingerprint(npz_path),
        "npz_size": npz_path.stat().st_size,
        "panel_plan_sha256": cli.config_fingerprint(
            {
                "root_seed": seed,
                "rng_namespace": "powered-stationarity",
                "path_id_start": path_id_start,
                "path_count": 4,
                "transitions_per_path": 2,
            }
        ),
    }
    cli.atomic_write_json(metadata_path, panel)
    loaded, rows = cli._load_frozen_powered_stationarity_panel(
        metadata_path,
        npz_path,
        root_seed=seed,
        path_id_start=path_id_start,
        path_count=4,
        transitions_per_path=2,
    )
    assert loaded == panel
    assert len(rows) == 9
    data = bytearray(npz_path.read_bytes())
    data[-1] ^= 1
    npz_path.write_bytes(data)
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._load_frozen_powered_stationarity_panel(
            metadata_path,
            npz_path,
            root_seed=seed,
            path_id_start=path_id_start,
            path_count=4,
            transitions_per_path=2,
        )


def test_nested_id_plan_preserves_aligned_words_and_uniform_marginals() -> None:
    metrics = cli._id_preflight(torch.device("cpu"))
    assert metrics["nested_id_uniqueness_pass"] == 1
    assert metrics["nested_id_aliasing_exact_pass"] == 1
    assert metrics["nested_id_aligned_philox_word_pass"] == 1
    assert metrics["nested_id_marginal_law_pass"] == 1
    assert metrics["marginal_law_sample_count"] == 4096
    assert (
        metrics["coupled_uniform_ks"]
        <= metrics["one_sample_ks_99_limit"]
    )
    assert (
        metrics["independent_uniform_ks"]
        <= metrics["one_sample_ks_99_limit"]
    )
    assert (
        metrics["coupled_independent_ks"]
        <= metrics["two_sample_ks_99_limit"]
    )


def test_local_generator_uses_the_unscaled_production_observables() -> None:
    state = np.full(784, 1.0 / 784.0, dtype=np.float64)
    error, rows = cli._local_generator_fixture(state)
    assert error <= 1.0e-8
    assert len(rows) == 4 * 10
    assert {row["observable"] for row in rows} == set(
        cli.refinement_observable_spec().names
    )
    quadratic = [
        row["analytic_generator"]
        for row in rows
        if row["observable"] == "quadratic_mass"
    ]
    assert min(quadratic) > 1_000.0  # no hidden division by 784


def test_stationarity_family_contains_moments_drift_and_balance_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_level(
        run_dir: Path,
        args: object,
        *,
        initial_states: np.ndarray,
        sample_steps: int,
        **kwargs: object,
    ) -> dict[str, object]:
        values = np.asarray(
            cli.evaluate_refinement_observables(initial_states),
            dtype=np.float64,
        )
        return {
            "sample_steps": sample_steps,
            "checkpoint_values": {1: values, 8: values},
        }

    monkeypatch.setattr(cli, "_run_level_panel", fake_level)
    args = SimpleNamespace(root_seed=261_151, bootstrap_reps=100)
    record, _, _ = cli._stationarity_sweep_panel(
        tmp_path,
        args,
        panel="a",
        path_count=8,
        seed=77,
    )
    names = [member["name"] for member in record["inference"]["members"]]
    assert len(names) == len(set(names)) == 145
    assert sum("initial_stationarity_exact_moment" in name for name in names) == 10
    assert sum("stationarity_exact_moment" in name for name in names) == 60
    assert sum("_stationarity_f" in name for name in names) == 50
    assert sum("_balance_" in name for name in names) == 25
    assert sum("eight_sweep" in name for name in names) == 10


def test_authorizing_timing_uses_persisted_complete_wall_upper() -> None:
    row = {
        "transition_count": 100,
        "certified_count": 100,
        "fallback_count": 0,
        "elapsed_seconds": 0.01,
        "wall_elapsed_seconds": 0.02,
        "complete_wall_upper_seconds": 12.5,
        "mass_error": 0.0,
        "state_updates_device_resident_pass": 1,
        **{name: 0 for name in cli._FORBIDDEN_COUNTS},
    }
    aggregate = cli._aggregate_execution([row])
    assert aggregate["wall_elapsed_seconds"] == pytest.approx(0.02)
    assert aggregate["complete_wall_upper_seconds"] == pytest.approx(12.5)


def test_controls_only_cli_imports_no_trainer_or_reverse_sampler() -> None:
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
            "mnist.d0_one_image_sampler",
            "mnist.diag_d0_one_image_overfit",
        }
    )
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert source.index('if args.stage in {"power", "all"}') < source.index(
        'if args.stage in {"refinement", "all"}'
    )


def test_source_record_preserves_parent_sources_and_adds_only_new_workflow() -> None:
    parent = Path(
        "runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation"
    ) / "20260723-092105_production-multipath-jacobi-rb"
    if not parent.is_dir():
        pytest.skip("immutable multipath parent unavailable")
    digest, paths = cli._source_record(parent)
    names = {Path(path).name for path in paths}
    manifest = cli._load(parent / "run_manifest.json")
    parent_names = {Path(path).name for path in manifest["source_paths"]}
    assert len(digest) == 64
    assert parent_names <= names
    assert {
        "d0_jacobi_rb_strang_refinement.py",
        "d0_jacobi_rb_strang_refinement_gate.py",
        "d0_jacobi_rb_strang_refinement_provenance.py",
        "diag_d0_jacobi_rb_strang_refinement.py",
    } <= names


def test_failed_stage_commits_evidence_before_required_gate_failure(
    tmp_path: Path,
) -> None:
    gate = cli._failed_stage_gate(tmp_path, "power", RuntimeError("boom"))
    assert gate["evaluation_status"] == "evaluated"
    assert gate["passed"] == 0
    assert (tmp_path / "power_failure.json").is_file()
    assert (tmp_path / "strang_power_gate.json").is_file()


def test_resume_registry_corruption_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
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
        _base(
            "--resume-run-dir",
            str(tmp_path),
            "--stage",
            "report",
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        )
    )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert code == 2
    assert after == before
    assert not (tmp_path / "unexpected_failure.json").exists()


def test_all_stage_respects_prerequisite_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def preflight(*args, **kwargs):
        calls.append("preflight")
        return cli._synthetic_gate("preflight", passed=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("downstream stage ran after failed preflight")

    monkeypatch.setattr(cli, "_run_preflight_stage", preflight)
    monkeypatch.setattr(cli, "_run_power_stage", forbidden)
    monkeypatch.setattr(cli, "_run_refinement_stage", forbidden)
    monkeypatch.setattr(
        cli,
        "verify_exact_jacobi_rb_multipath_parent",
        lambda path: {"evaluation_status": "evaluated", "passed": 1},
    )
    monkeypatch.setattr(
        cli,
        "_source_image",
        lambda args: {
            "label": 3,
            "class_index": 0,
            "dataset_index": 0,
            "image_sha256": cli.EXPECTED_IMAGE_SHA256,
            "mixed_target_sha256": "mixed",
            "image": np.full(784, 1.0 / 784.0, dtype=np.float32),
            "mixed_target": np.full(784, 1.0 / 784.0, dtype=np.float32),
        },
    )
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda device: {})
    monkeypatch.setattr(cli, "_source_record", lambda parent: ("hash", []))
    code = cli.main(
        _base(
            "--runs-root",
            str(tmp_path),
            "--run-name",
            "tiny",
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        )
    )
    assert code == 0
    assert calls == ["preflight"]
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert cli._load(run_dir / "strang_power_gate.json")["evaluation_status"] == "not_evaluated"
    assert (
        cli._load(run_dir / "strang_refinement_gate.json")["evaluation_status"]
        == "not_evaluated"
    )
    status = cli._load(run_dir / "run_status.json")
    assert status["physical_training_performed"] == 0
    assert status["reverse_sampling_performed"] == 0


def test_report_stage_is_read_only_scientifically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("preflight", "power", "refinement"):
        cli.atomic_write_json(
            tmp_path / f"strang_{name}_gate.json",
            cli._synthetic_gate(name, passed=True),
        )
    cli.atomic_write_json(tmp_path / "parent_provenance.json", {"passed": 1})
    cli.atomic_write_json(tmp_path / "scientific_config.json", {"frozen": 1})
    cli.atomic_write_json(tmp_path / "source_image.json", {"image_sha256": "x"})
    cli.atomic_write_json(tmp_path / "run_manifest.json", {"source_paths": []})
    cli.atomic_write_json(
        tmp_path / "run_status.json",
        {"schema": cli.RUN_SCHEMA, "schema_version": 1, "status": "running"},
    )
    monkeypatch.setattr(cli, "_verify_terminal_registry", lambda *a, **k: None)
    monkeypatch.setattr(
        cli,
        "verify_exact_jacobi_rb_multipath_parent",
        lambda path: {"passed": 1},
    )
    monkeypatch.setattr(cli, "_source_record", lambda parent: ("hash", []))
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda device: {})
    monkeypatch.setattr(
        cli,
        "_freeze",
        lambda path, value, require_existing=False: dict(value),
    )
    code = cli.main(
        _base(
            "--resume-run-dir",
            str(tmp_path),
            "--stage",
            "report",
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        )
    )
    assert code == 0
    assert cli._load(tmp_path / "run_status.json")["status"] == "complete"
