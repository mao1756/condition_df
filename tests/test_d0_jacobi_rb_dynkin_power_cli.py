from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

import mnist.diag_d0_jacobi_rb_dynkin_power_confirmation as cli


def _base(*extra: str) -> list[str]:
    return ["--parent-strang-run-dir", "parent", *extra]


def test_cli_frozen_defaults_and_stage_dependencies() -> None:
    args = cli.parse_args(_base("--stage", "preflight"))
    assert args.root_seed == 261_161
    assert args.sample_steps == (128, 256, 512, 1024, 2048)
    assert args.tower_panel_clusters == 128
    assert args.pilot_panel_paths == 8
    for stage in ("pilot", "report"):
        with pytest.raises(SystemExit):
            cli.parse_args(_base("--stage", stage))
    with pytest.raises(SystemExit):
        cli.parse_args(
            _base("--stage", "preflight", "--require-gate", "pilot")
        )
    cli.parse_args(
        _base(
            "--stage",
            "pilot",
            "--resume-run-dir",
            "run",
            "--require-gate",
            "pilot",
        )
    )


def test_production_workload_is_frozen_and_test_runs_cannot_gate() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(_base("--pilot-panel-paths", "2"))
    reduced = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--tower-panel-clusters",
            "2",
            "--pilot-panel-paths",
            "2",
            "--pilot-stop-steps",
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
    with pytest.raises(SystemExit):
        cli.parse_args(
            _base(
                "--device",
                "cpu",
                "--tower-panel-clusters",
                "129",
                "--test-only-reduced-workload",
            )
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            _base(
                "--device",
                "cpu",
                "--pilot-panel-paths",
                "9",
                "--test-only-reduced-workload",
            )
        )


def test_distribution_free_report_is_explicitly_nonauthorizing() -> None:
    record = cli._distribution_free_power_impossibility()
    assert record["authorizing"] == 0
    assert record["documented_required_paths_upper_order"] == pytest.approx(2.4e18)
    assert "engineering forecast" in record["conclusion"]


@pytest.mark.skipif(
    cli._spectral._arb is None, reason="python-flint/Arb is unavailable"
)
def test_phase_oracle_uses_full_arb_and_catches_p2_root_fixture() -> None:
    metrics, rows = cli._phase_moment_oracle_controls(torch.device("cpu"))
    assert len(rows) == 8
    assert metrics["spectral_arb_agreement_pass"] == 1
    assert metrics["cuda_enclosure_pass"] == 1
    assert metrics["adversarial_p2_root_enclosure_pass"] == 1


def test_feature_matrices_cover_successive_and_richardson_families() -> None:
    results: dict[int, dict[str, object]] = {}
    for level in cli.PILOT_LEVELS:
        checkpoints = {}
        for fraction in cli.OBSERVATION_TIME_FRACTIONS:
            step = int(round(level * fraction))
            values = np.arange(80, dtype=np.float64).reshape(8, 10)
            checkpoints[step] = values + level + fraction
        results[level] = {"dynkin_checkpoint_values": checkpoints}
    main, reference = cli._projection_feature_matrices(
        results, key="dynkin_checkpoint_values"
    )
    assert main.shape == (8, 120)
    assert reference.shape == (8, 40)
    assert np.isfinite(main).all()
    assert np.isfinite(reference).all()


def test_sealed_panel_plans_are_disjoint_and_tamper_evident(tmp_path: Path) -> None:
    args = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--tower-panel-clusters",
            "2",
            "--pilot-panel-paths",
            "2",
            "--test-only-reduced-workload",
        )
    )
    path_id_plan = cli._build_path_id_plan(args)
    cli._freeze_path_id_plan(
        tmp_path, path_id_plan, require_existing=False
    )
    registry = cli._freeze_panel_plans(tmp_path, args, path_id_plan)
    assert registry["panels_disjoint"] == 1
    assert registry["future_production_namespace_disjoint"] == 1
    assert registry["path_id_plan_sha256"] == path_id_plan.sha256
    a = cli._load(tmp_path / "panel_a_plan.json")
    b = cli._load(tmp_path / "panel_b_plan.json")
    assert set(a["path_ids"]).isdisjoint(b["path_ids"])
    assert a["path_ids"] == [0x40000, 0x40001]
    assert b["path_ids"] == [0x50000, 0x50001]
    assert max(a["path_ids"] + b["path_ids"]) < (1 << 20)
    a["path_ids"][0] += 1
    cli.atomic_write_json(tmp_path / "panel_a_plan.json", a)
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._freeze_panel_plans(tmp_path, args, path_id_plan)


def test_path_id_preflight_uses_real_canonical_ids_and_is_tamper_evident(
    tmp_path: Path,
) -> None:
    args = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        )
    )
    plan = cli._build_path_id_plan(args)
    cli._freeze_path_id_plan(tmp_path, plan, require_existing=False)
    evidence = cli._path_id_plan_preflight(tmp_path, args, plan)
    assert evidence["passed"] == 1
    assert evidence["checks"]["canonical_id_smoke_pass"] == 1
    assert evidence["checks"]["tower_chunking_pass"] == 1
    assert len(evidence["tower_transition_id_hashes"]) == 16
    assert evidence["maximum_packed_transition_id"] < (1 << 43)

    evidence["checks"]["path_major_order_pass"] = 0
    cli.atomic_write_json(tmp_path / "path_id_plan_preflight.json", evidence)
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._path_id_plan_preflight(tmp_path, args, plan)


def test_tower_id_allocation_oom_remains_a_resource_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--tower-panel-clusters",
            "1",
            "--test-only-reduced-workload",
        )
    )
    cli._freeze_path_id_plan(
        tmp_path,
        cli.DynkinPathIDPlan(tower_path_count=1, pilot_path_count=8),
        require_existing=False,
    )

    def allocation_oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic allocation failure")

    monkeypatch.setattr(
        cli, "canonical_tower_transition_ids", allocation_oom
    )
    with pytest.raises(torch.cuda.OutOfMemoryError):
        cli._tower_panel(tmp_path, args, panel="a", seed=1)


def test_failed_stage_commits_artifacts_before_gate_failure(tmp_path: Path) -> None:
    gate = cli._failed_stage_gate(
        tmp_path,
        "power",
        RuntimeError("boom"),
        failure_domain="scheduler_execution",
        failure_code="test_scheduler_failure",
    )
    assert gate["passed"] == 0
    assert gate["evaluation_status"] == "execution_failed"
    assert gate["stage_execution_valid"] == 0
    assert gate["scientific_evidence_complete"] == 0
    assert gate["failure_domain"] == "scheduler_execution"
    assert gate["failure_code"] == "test_scheduler_failure"
    assert (tmp_path / "power_failure.json").is_file()
    assert (tmp_path / "dynkin_power_gate.json").is_file()


def test_legacy_resume_without_corrected_path_plan_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    cli.atomic_write_json(
        tmp_path / "run_manifest.json",
        {"source_fingerprint": "legacy"},
    )
    cli.atomic_write_json(tmp_path / "scientific_config.json", {})
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"immutable")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    args = cli.parse_args(
        _base("--device", "cpu", "--test-only-reduced-workload")
    )
    source = np.full(784, 1.0 / 784.0, dtype=np.float64)
    with pytest.raises(cli.ArtifactCompatibilityError, match="corrected"):
        cli._verify_resume_contract(
            tmp_path,
            expected_plan=cli.DynkinPathIDPlan(),
            expected_config={},
            expected_manifest={},
            expected_provenance={},
            expected_source={
                "image": source,
                "mixed_target": source,
                "image_sha256": cli.EXPECTED_IMAGE_SHA256,
                "mixed_target_sha256": cli.EXPECTED_MIXED_TARGET_SHA256,
            },
            args=args,
        )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before


def test_main_rejects_legacy_resume_without_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "legacy-failed-run"
    run_dir.mkdir()
    cli.atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": "legacy",
            "status": "gate_failed",
            "decision": "control_provenance_invalid",
        },
    )
    cli.atomic_write_json(
        run_dir / "legacy_evidence.json",
        {"immutable": 1, "failure": "path_id_namespace"},
    )
    before = {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        cli, "_verify_terminal_registry", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli,
        "verify_raw_endpoint_power_infeasible_parent",
        lambda path: {
            "evaluation_status": "evaluated",
            "passed": 1,
            "parent_artifact_record_count": 1308,
            "parent_source_count": 15,
        },
    )
    monkeypatch.setattr(cli, "_source_record", lambda parent: ("source", []))
    image = np.full(784, 1.0 / 784.0, dtype=np.float64)
    monkeypatch.setattr(
        cli,
        "_load_source_image",
        lambda parent: {
            "image": image,
            "mixed_target": image,
            "image_sha256": cli.EXPECTED_IMAGE_SHA256,
            "mixed_target_sha256": cli.EXPECTED_MIXED_TARGET_SHA256,
        },
    )

    code = cli.main(
        _base(
            "--resume-run-dir",
            str(run_dir),
            "--stage",
            "preflight",
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        )
    )

    after = {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert code == 2
    assert after == before


def test_corrected_resume_binds_semantic_and_file_path_plan_hashes(
    tmp_path: Path,
) -> None:
    args = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--tower-panel-clusters",
            "2",
            "--pilot-panel-paths",
            "2",
            "--test-only-reduced-workload",
        )
    )
    plan = cli.DynkinPathIDPlan(tower_path_count=2, pilot_path_count=2)
    cli._freeze_path_id_plan(tmp_path, plan, require_existing=False)
    cli._freeze_panel_plans(tmp_path, args, plan)
    file_hash = cli.file_fingerprint(tmp_path / "path_id_plan.json")
    config = cli._scientific_config(args, plan)
    config_sha = cli.config_fingerprint(config)
    manifest = {
        "source_fingerprint": "current-source",
        "scientific_config_sha256": config_sha,
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "path_id_plan_file_sha256": file_hash,
    }
    provenance = {"passed": 1, "record": "immutable-parent"}
    image = np.full(784, 1.0 / 784.0, dtype=np.float64)
    cli._atomic_write_npz(
        tmp_path / "source_image.npz",
        image=image,
        mixed_target=image,
    )
    cli.atomic_write_json(
        tmp_path / "source_image.json",
        {
            "image_sha256": cli.EXPECTED_IMAGE_SHA256,
            "mixed_target_sha256": cli.EXPECTED_MIXED_TARGET_SHA256,
            "source_npz_sha256": cli.file_fingerprint(
                tmp_path / "source_image.npz"
            ),
            "source_npz_size": (tmp_path / "source_image.npz").stat().st_size,
        },
    )
    source = {
        "image": image,
        "mixed_target": image,
        "image_sha256": cli.EXPECTED_IMAGE_SHA256,
        "mixed_target_sha256": cli.EXPECTED_MIXED_TARGET_SHA256,
    }
    cli.atomic_write_json(tmp_path / "run_manifest.json", manifest)
    cli.atomic_write_json(tmp_path / "scientific_config.json", config)
    cli.atomic_write_json(tmp_path / "parent_provenance.json", provenance)
    cli.atomic_write_json(
        tmp_path / "run_status.json",
        {
            "scientific_config_sha256": config_sha,
            "source_fingerprint": "current-source",
        },
    )
    cli._verify_resume_contract(
        tmp_path,
        expected_plan=plan,
        expected_config=config,
        expected_manifest=manifest,
        expected_provenance=provenance,
        expected_source=source,
        args=args,
    )

    tampered_config = dict(config)
    tampered_config["root_seed"] = int(config["root_seed"]) + 1
    cli.atomic_write_json(tmp_path / "scientific_config.json", tampered_config)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(cli.ArtifactCompatibilityError, match="corrected"):
        cli._verify_resume_contract(
            tmp_path,
            expected_plan=plan,
            expected_config=config,
            expected_manifest=manifest,
            expected_provenance=provenance,
            expected_source=source,
            args=args,
        )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before
    cli.atomic_write_json(tmp_path / "scientific_config.json", config)

    with (tmp_path / "path_id_plan.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(cli.ArtifactCompatibilityError, match="corrected"):
        cli._verify_resume_contract(
            tmp_path,
            expected_plan=plan,
            expected_config=config,
            expected_manifest=manifest,
            expected_provenance=provenance,
            expected_source=source,
            args=args,
        )


def test_resume_registry_tamper_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    cli.atomic_write_json(tmp_path / "artifact_registry.json", {"records": {}})
    cli.atomic_write_json(
        tmp_path / "run_status.json",
        {
            "schema": cli.RUN_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "artifact_registry_sha256": "wrong",
        },
    )
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"immutable")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    code = cli.main(
        _base(
            "--stage",
            "report",
            "--resume-run-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        )
    )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert code == 2
    assert after == before


def test_completed_pilot_shard_corruption_is_not_recoverable(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "dynkin_shards" / "pilot" / "a" / "shard.bin"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"original")
    registry = cli._artifact_registry(tmp_path)
    cli.atomic_write_json(tmp_path / "artifact_registry.json", registry)
    cli.atomic_write_json(
        tmp_path / "run_status.json",
        {
            "status": "complete",
            "artifact_registry_sha256": cli.file_fingerprint(
                tmp_path / "artifact_registry.json"
            ),
        },
    )
    shard.write_bytes(b"corrupt")
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_terminal_registry(tmp_path, stage="pilot")


def test_completed_resume_rejects_unregistered_artifact(tmp_path: Path) -> None:
    registry = cli._artifact_registry(tmp_path)
    cli.atomic_write_json(tmp_path / "artifact_registry.json", registry)
    cli.atomic_write_json(
        tmp_path / "run_status.json",
        {
            "status": "complete",
            "artifact_registry_sha256": cli.file_fingerprint(
                tmp_path / "artifact_registry.json"
            ),
        },
    )
    (tmp_path / "unexpected.bin").write_bytes(b"not registered")
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_terminal_registry(tmp_path, stage="report")


def test_resumed_parent_mismatch_does_not_mutate_run(tmp_path: Path) -> None:
    registry = cli._artifact_registry(tmp_path)
    cli.atomic_write_json(tmp_path / "artifact_registry.json", registry)
    cli.atomic_write_json(
        tmp_path / "run_status.json",
        {
            "schema": cli.RUN_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "artifact_registry_sha256": cli.file_fingerprint(
                tmp_path / "artifact_registry.json"
            ),
        },
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    code = cli.main(
        [
            "--parent-strang-run-dir",
            str(tmp_path / "missing-parent"),
            "--stage",
            "report",
            "--resume-run-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        ]
    )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert code == 2
    assert after == before


def _panel_record(*, healthy: bool = True) -> dict[str, object]:
    forbidden = 0 if healthy else 1
    execution = {
        "transition_count": 100,
        "certified_count": 100 - forbidden,
        "certificate_fraction": 1.0 if healthy else 0.99,
        "fallback_count": 0,
        "fallback_fraction": 0.0,
        "elapsed_seconds": 1.0,
        "fallback_elapsed_seconds": 0.0,
        "complete_wall_upper_seconds": 1.1,
        "conservative_rate": 1300.0,
        "mass_error": 1.0e-12,
        "peak_memory_fraction": 0.1,
        "state_updates_device_resident_pass": 1,
        "shard_chain_pass": 1,
        **{name: 0 for name in cli._FORBIDDEN_COUNTS},
    }
    execution["uncertified_count"] = forbidden
    return {
        "main_differences": np.ones((8, 120), dtype=np.float64).tolist(),
        "reference_differences": np.ones((8, 40), dtype=np.float64).tolist(),
        "raw_main_differences": (
            2.0 * np.ones((8, 120), dtype=np.float64)
        ).tolist(),
        "raw_reference_differences": (
            2.0 * np.ones((8, 40), dtype=np.float64)
        ).tolist(),
        "execution": execution,
        "complete": 1,
        "finite": 1,
        "maximum_cumulative_standardized_error": 1.0e-12,
    }


def test_candidate_nomination_uses_actual_execution_health() -> None:
    parent = (
        Path("runs/experiment12_d0_jacobi_rb_strang_refinement")
        / "20260723-230629_production-state-dependent-strang-refinement"
    )
    if not parent.is_dir():
        pytest.skip("immutable Strang parent unavailable")
    rows = cli._candidate_rows_for_panel(
        _panel_record(healthy=False),
        role="a",
        parent_dir=parent,
        conservative_rate=1300.0,
    )
    assert not any(int(row["panel_certification_pass"]) for row in rows)
    assert cli.select_dynkin_panel_a_design(rows)["passed"] == 0


def test_combined_panel_and_feature_diagnostics_preserve_health(
    tmp_path: Path,
) -> None:
    panel_a = _panel_record()
    panel_b = _panel_record()
    combined = cli._combined_panel_record(panel_a, panel_b)
    assert combined["complete"] == 1
    assert combined["finite"] == 1
    assert combined["execution"]["shard_chain_pass"] == 1
    cli._write_feature_power_diagnostics(
        tmp_path, role="combined", record=combined
    )
    assert (tmp_path / "feature_power_combined.csv").is_file()
    summary = cli._load(tmp_path / "raw_dynkin_power_combined.json")
    assert summary["authorizing"] == 0
    assert summary["families"]["main"]["feature_count"] == 120
    assert summary["families"]["reference"]["feature_count"] == 40


def test_all_stage_stops_after_failed_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def preflight(*args, **kwargs):
        calls.append("preflight")
        return cli._synthetic_gate("preflight", passed=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("pilot ran after failed preflight")

    image = np.full(784, 1.0 / 784.0, dtype=np.float64)
    monkeypatch.setattr(cli, "_run_preflight_stage", preflight)
    monkeypatch.setattr(cli, "_run_pilot_stage", forbidden)
    monkeypatch.setattr(
        cli,
        "verify_raw_endpoint_power_infeasible_parent",
        lambda path: {
            "passed": 1,
            "parent_artifact_record_count": 1308,
            "parent_source_count": 15,
        },
    )
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda device: {})
    monkeypatch.setattr(cli, "_source_record", lambda parent: ("hash", []))
    monkeypatch.setattr(
        cli,
        "_load_source_image",
        lambda parent: {
            "image": image,
            "mixed_target": image,
            "image_sha256": cli.EXPECTED_IMAGE_SHA256,
            "mixed_target_sha256": cli.EXPECTED_MIXED_TARGET_SHA256,
            "source_npz_sha256": "parent",
        },
    )
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
    assert cli._load(run_dir / "dynkin_power_gate.json")[
        "evaluation_status"
    ] == "not_evaluated"
    status = cli._load(run_dir / "run_status.json")
    assert status["physical_training_performed"] == 0
    assert status["reverse_sampling_performed"] == 0


def test_stage_local_artifact_error_is_not_misclassified_as_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full(784, 1.0 / 784.0, dtype=np.float64)

    def failed_preflight(*args, **kwargs):
        raise cli.ArtifactCompatibilityError("local tower artifact is malformed")

    monkeypatch.setattr(cli, "_run_preflight_stage", failed_preflight)
    monkeypatch.setattr(
        cli,
        "verify_raw_endpoint_power_infeasible_parent",
        lambda path: {
            "evaluation_status": "evaluated",
            "passed": 1,
            "parent_artifact_record_count": 1308,
            "parent_source_count": 15,
        },
    )
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda device: {})
    monkeypatch.setattr(cli, "_source_record", lambda parent: ("hash", []))
    monkeypatch.setattr(
        cli,
        "_load_source_image",
        lambda parent: {
            "image": image,
            "mixed_target": image,
            "image_sha256": cli.EXPECTED_IMAGE_SHA256,
            "mixed_target_sha256": cli.EXPECTED_MIXED_TARGET_SHA256,
            "source_npz_sha256": "parent",
        },
    )
    code = cli.main(
        _base(
            "--runs-root",
            str(tmp_path),
            "--run-name",
            "artifact-failure",
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--test-only-reduced-workload",
        )
    )
    assert code == 0
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    gate = cli._load(run_dir / "dynkin_preflight_gate.json")
    status = cli._load(run_dir / "run_status.json")
    assert gate["evaluation_status"] == "execution_failed"
    assert gate["failure_code"] == "preflight_artifact_incompatibility"
    assert status["decision"] == "dynkin_estimator_numerically_unresolved"
    assert status["decision"] != "control_provenance_invalid"


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


def test_source_record_preserves_parent_sources_and_binds_all_new_modules() -> None:
    parent = (
        Path("runs/experiment12_d0_jacobi_rb_strang_refinement")
        / "20260723-230629_production-state-dependent-strang-refinement"
    )
    if not parent.is_dir():
        pytest.skip("immutable Strang parent unavailable")
    digest, paths = cli._source_record(parent)
    names = {Path(path).name for path in paths}
    assert len(digest) == 64
    assert {
        "d0_jacobi_rb_dynkin.py",
        "d0_jacobi_rb_dynkin_cuda.py",
        "d0_jacobi_rb_dynkin_path_ids.py",
        "d0_jacobi_rb_dynkin_power_gate.py",
        "d0_jacobi_rb_dynkin_power_provenance.py",
        "diag_d0_jacobi_rb_dynkin_power_confirmation.py",
    } <= names
    parent_manifest = cli._load(parent / "run_manifest.json")
    assert {Path(path).name for path in parent_manifest["source_paths"]} <= names
