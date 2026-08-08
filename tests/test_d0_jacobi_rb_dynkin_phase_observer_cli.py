from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mnist.diag_d0_jacobi_rb_dynkin_phase_observer_confirmation as cli


def _base(*extra: str) -> list[str]:
    return ["--parent-dynkin-idfix-run-dir", "parent", *extra]


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_cli_frozen_defaults_and_stage_requirements() -> None:
    args = cli.parse_args(_base("--stage", "preflight"))
    assert args.root_seed == 261_171
    assert args.sample_steps == (128, 256, 512, 1024, 2048)
    assert args.tower_panel_clusters == 128
    assert args.pilot_panel_paths == 8
    assert args.require_gate == "none"

    for stage in ("pilot", "report"):
        with pytest.raises(SystemExit):
            cli.parse_args(_base("--stage", stage))

    with pytest.raises(SystemExit):
        cli.parse_args(
            _base("--stage", "preflight", "--require-gate", "pilot")
        )

    resumed = cli.parse_args(
        _base(
            "--stage",
            "pilot",
            "--resume-run-dir",
            "run",
            "--require-gate",
            "pilot",
        )
    )
    assert resumed.resume_run_dir == Path("run")


def test_production_overrides_are_rejected_and_reduced_runs_cannot_gate() -> None:
    for option, value in (
        ("--root-seed", "7"),
        ("--sample-steps", "128,512"),
        ("--tower-panel-clusters", "2"),
        ("--pilot-panel-paths", "2"),
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(_base(option, value))

    reduced = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--root-seed",
            "7",
            "--sample-steps",
            "128,512",
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
    assert reduced.root_seed == 7
    assert reduced.sample_steps == (128, 512)

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
                "--pilot-stop-steps",
                "7",
                "--test-only-reduced-workload",
            )
        )


def test_test_sampler_identity_is_bound_even_at_production_numeric_defaults() -> None:
    production = cli.parse_args(_base("--device", "cuda"))
    reduced = cli.parse_args(
        _base(
            "--device",
            "cuda",
            "--test-only-reduced-workload",
        )
    )
    plan = cli._build_path_id_plan(production)
    assert cli._build_path_id_plan(reduced) == plan

    production_config = cli._scientific_config(production, plan)
    reduced_config = cli._scientific_config(reduced, plan)
    assert production_config["test_only_reduced_workload"] == 0
    assert reduced_config["test_only_reduced_workload"] == 1
    assert production_config["sampler_identity"] != reduced_config[
        "sampler_identity"
    ]
    assert cli.config_fingerprint(production_config) != cli.config_fingerprint(
        reduced_config
    )


def test_path_and_panel_plans_are_frozen_before_execution_and_disjoint(
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
    plan = cli._build_path_id_plan(args)
    plan_record = cli._freeze_path_id_plan(
        tmp_path, plan, require_existing=False
    )
    registry = cli._freeze_panel_plans(
        tmp_path, args, plan, require_existing=False
    )

    assert plan_record["path_id_plan_sha256"] == plan.sha256
    assert registry["panels_frozen_before_device_execution"] == 1
    assert registry["panels_disjoint"] == 1
    assert registry["future_production_namespace_disjoint"] == 1

    a = cli._load(tmp_path / "panel_a_plan.json")
    b = cli._load(tmp_path / "panel_b_plan.json")
    assert a["tower_case_path_ids"][0] == [0x60000, 0x60001]
    assert b["tower_case_path_ids"][0] == [0x70000, 0x70001]
    assert a["pilot_path_ids"] == [0x80000, 0x80001]
    assert b["pilot_path_ids"] == [0x90000, 0x90001]

    all_a = {
        path_id
        for case in a["tower_case_path_ids"]
        for path_id in case
    } | set(a["pilot_path_ids"])
    all_b = {
        path_id
        for case in b["tower_case_path_ids"]
        for path_id in case
    } | set(b["pilot_path_ids"])
    assert all_a.isdisjoint(all_b)
    assert all_a.isdisjoint(plan.designated_production_path_ids)
    assert all_b.isdisjoint(plan.designated_production_path_ids)

    # Freezing is idempotent before work starts.
    before = _snapshot(tmp_path)
    cli._freeze_path_id_plan(tmp_path, plan, require_existing=True)
    cli._freeze_panel_plans(
        tmp_path, args, plan, require_existing=True
    )
    assert _snapshot(tmp_path) == before


def test_frozen_panel_tamper_is_rejected_without_mutation(
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
    plan = cli._build_path_id_plan(args)
    cli._freeze_path_id_plan(tmp_path, plan, require_existing=False)
    cli._freeze_panel_plans(
        tmp_path, args, plan, require_existing=False
    )

    path = tmp_path / "panel_a_plan.json"
    record = cli._load(path)
    record["pilot_path_ids"][0] += 1
    cli.atomic_write_json(path, record)
    before = _snapshot(tmp_path)
    with pytest.raises(cli.ArtifactCompatibilityError, match="frozen"):
        cli._freeze_panel_plans(
            tmp_path, args, plan, require_existing=True
        )
    assert _snapshot(tmp_path) == before


def test_path_id_preflight_uses_fresh_real_ids_and_is_tamper_evident(
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
    plan = cli._build_path_id_plan(args)
    cli._freeze_path_id_plan(tmp_path, plan, require_existing=False)
    evidence = cli._path_id_plan_preflight(tmp_path, args, plan)

    assert evidence["passed"] == 1
    assert len(evidence["canonical_transition_id_hashes"]) == 16
    assert evidence["checks"]["canonical_id_uniqueness_pass"] == 1
    assert evidence["checks"]["fresh_namespace_disjoint_pass"] == 1
    assert evidence["maximum_packed_transition_id"] < (1 << 43)

    evidence["checks"]["fresh_namespace_disjoint_pass"] = 0
    cli.atomic_write_json(
        tmp_path / "phase_observer_path_id_preflight.json", evidence
    )
    before = _snapshot(tmp_path)
    with pytest.raises(cli.ArtifactCompatibilityError, match="frozen"):
        cli._path_id_plan_preflight(tmp_path, args, plan)
    assert _snapshot(tmp_path) == before


def test_tower_case_is_atomic_resumable_and_rejects_corrupt_payload(
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
    plan = cli._build_path_id_plan(args)
    states = cli._tower_initial_states(
        2, root_seed=args.root_seed, panel="a"
    )

    first = cli._tower_case(
        tmp_path,
        args,
        panel="a",
        case_index=0,
        states_np=states,
        plan=plan,
    )
    before = _snapshot(tmp_path)
    second = cli._tower_case(
        tmp_path,
        args,
        panel="a",
        case_index=0,
        states_np=states,
        plan=plan,
    )
    assert second == first
    assert _snapshot(tmp_path) == before

    payload = tmp_path / "tower_cases" / "a" / "case-00.npz"
    payload.write_bytes(b"corrupt")
    corrupt = _snapshot(tmp_path)
    with pytest.raises(cli.ArtifactCompatibilityError, match="changed"):
        cli._tower_case(
            tmp_path,
            args,
            panel="a",
            case_index=0,
            states_np=states,
            plan=plan,
        )
    assert _snapshot(tmp_path) == corrupt


def test_terminal_registry_detects_resume_tamper_without_mutation(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"immutable evidence")
    registry_binding = cli._finalize_registry(tmp_path)
    cli._write_status(tmp_path, status="complete", **registry_binding)
    cli._verify_terminal_registry(tmp_path)

    evidence.write_bytes(b"tampered evidence")
    before = _snapshot(tmp_path)
    with pytest.raises(cli.ArtifactCompatibilityError, match="changed"):
        cli._verify_terminal_registry(tmp_path)
    assert _snapshot(tmp_path) == before


def test_failed_stage_commits_evidence_before_returning_gate(
    tmp_path: Path,
) -> None:
    gate = cli._failed_stage_gate(
        tmp_path,
        "preflight",
        RuntimeError("synthetic observer failure"),
        failure_domain="observer_execution",
        failure_code="synthetic_failure",
    )

    assert gate["evaluation_status"] == "execution_failed"
    assert gate["passed"] == 0
    assert gate["stage_execution_valid"] == 0
    assert gate["scientific_evidence_complete"] == 0
    assert gate["failure_domain"] == "observer_execution"
    assert gate["failure_code"] == "synthetic_failure"
    assert (tmp_path / "preflight_failure.json").is_file()
    assert (tmp_path / "phase_observer_preflight_gate.json").is_file()
    failure = cli._load(tmp_path / "preflight_failure.json")
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "synthetic observer failure"


def test_required_gate_failure_finalizes_readable_artifacts_first(
    tmp_path: Path,
) -> None:
    cli.atomic_write_json(tmp_path / "evidence.json", {"complete": 1})
    provenance = {
        "evaluation_status": "evaluated",
        "passed": 1,
        "provenance_valid": 1,
    }
    preflight = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "numerically_valid": 0,
    }
    pilot = cli.not_evaluated_gate("pilot", "preflight failed")
    code = cli._finish(
        tmp_path,
        SimpleNamespace(require_gate="preflight", stage="preflight"),
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
    )

    assert code == 1
    assert (tmp_path / "phase_observer_workflow_gate.json").is_file()
    assert (tmp_path / "phase_observer_decision.json").is_file()
    assert (tmp_path / "artifact_registry.json").is_file()
    status = cli._load(tmp_path / "run_status.json")
    registry = cli._load(tmp_path / "artifact_registry.json")
    assert status["outcome"] == "gate_failed"
    assert status["required_gate_pass"] == 0
    assert status["artifact_registry_record_count"] >= 1
    assert status["physical_training_performed"] == 0
    assert set(
        registry["terminal_files_excluded_to_avoid_self_reference"]
    ) == {"artifact_registry.json", "run_status.json"}
    assert {
        "phase_observer_workflow_gate.json",
        "phase_observer_decision.json",
    } <= set(registry["records"])


def test_resume_contract_rejects_changed_scientific_config_without_mutation(
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
    plan = cli._build_path_id_plan(args)
    cli._freeze_path_id_plan(tmp_path, plan, require_existing=False)
    cli._freeze_panel_plans(
        tmp_path, args, plan, require_existing=False
    )

    uniform = np.full(cli.PATH_STATE_SIZE, 1.0 / cli.PATH_STATE_SIZE)
    source = {
        "image": uniform,
        "mixed_target": uniform,
        "image_sha256": cli.EXPECTED_IMAGE_SHA256,
        "mixed_target_sha256": cli.EXPECTED_MIXED_TARGET_SHA256,
        "parent_npz_sha256": "parent-source",
    }
    cli._freeze_source_image(
        tmp_path, source, require_existing=False
    )
    config = cli._scientific_config(args, plan)
    manifest = {"schema": "test-manifest", "source_fingerprint": "source"}
    provenance = {"schema": "test-provenance", "passed": 1}
    cli.atomic_write_json(tmp_path / "scientific_config.json", config)
    cli.atomic_write_json(tmp_path / "run_manifest.json", manifest)
    cli.atomic_write_json(tmp_path / "parent_provenance.json", provenance)

    cli._verify_resume_contract(
        tmp_path,
        expected_plan=plan,
        expected_config=config,
        expected_manifest=manifest,
        expected_provenance=provenance,
        expected_source=source,
        args=args,
    )

    tampered = dict(config)
    tampered["root_seed"] = int(tampered["root_seed"]) + 1
    cli.atomic_write_json(tmp_path / "scientific_config.json", tampered)
    before = _snapshot(tmp_path)
    with pytest.raises(cli.ArtifactCompatibilityError, match="configuration"):
        cli._verify_resume_contract(
            tmp_path,
            expected_plan=plan,
            expected_config=config,
            expected_manifest=manifest,
            expected_provenance=provenance,
            expected_source=source,
            args=args,
        )
    assert _snapshot(tmp_path) == before


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
            "mnist.diag_d0_jacobi_reverse_sampling",
        }
    )


def test_source_record_preserves_parent_sources_and_binds_new_modules() -> None:
    parent = (
        Path("runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation")
        / "20260724-184842_production-dynkin-strang-power-idfix"
    )
    if not parent.is_dir():
        pytest.skip("immutable Dynkin ID-fix parent unavailable")
    digest, paths = cli._source_record(parent)
    names = {Path(path).name for path in paths}
    parent_manifest = cli._load(parent / "run_manifest.json")

    assert len(digest) == 64
    assert {Path(path).name for path in parent_manifest["source_paths"]} <= names
    assert {
        "d0_jacobi_rb_dynkin_phase_observer.py",
        "d0_jacobi_rb_dynkin_phase_observer_gate.py",
        "d0_jacobi_rb_dynkin_phase_observer_path_ids.py",
        "d0_jacobi_rb_dynkin_phase_observer_provenance.py",
        "diag_d0_jacobi_rb_dynkin_phase_observer_confirmation.py",
    } <= names


def test_reduced_preflight_orchestration_commits_and_resumes_exactly(
    tmp_path: Path,
) -> None:
    parent = (
        Path("runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation")
        / "20260724-184842_production-dynkin-strang-power-idfix"
    )
    if not parent.is_dir():
        pytest.skip("immutable Dynkin ID-fix parent unavailable")
    args = cli.parse_args(
        [
            "--parent-dynkin-idfix-run-dir",
            str(parent),
            "--stage",
            "preflight",
            "--device",
            "cpu",
            "--tower-panel-clusters",
            "2",
            "--pilot-panel-paths",
            "2",
            "--test-only-reduced-workload",
        ]
    )
    plan = cli._build_path_id_plan(args)
    cli._freeze_path_id_plan(tmp_path, plan, require_existing=False)
    cli._freeze_panel_plans(
        tmp_path, args, plan, require_existing=False
    )
    provenance = {
        "passed": 1,
        "parent_re_adjudication": "tower_observer_roundoff_invalid",
        "parent_failure_message": "nonzero degenerate whole-path statistic",
        "parent_source_count": cli.PARENT_SOURCE_COUNT,
        "parent_source_fingerprint": cli.PARENT_SOURCE_FINGERPRINT,
        "parent_artifact_record_count": cli.PARENT_REGISTRY_RECORD_COUNT,
        "parent_path_id_plan_pass": 1,
        "parent_legacy_k512_replay_pass": 1,
        "parent_phase_moment_oracle_pass": 1,
        "parent_tower_inference_performed": 0,
        "parent_pilot_performed": 0,
    }

    gate = cli._run_preflight_stage(tmp_path, args, provenance)
    assert gate["evaluation_status"] == "evaluated"
    assert gate["passed"] == 0  # reduced workloads are never authorizing
    assert (tmp_path / "phase_observer_preflight_metrics.json").is_file()
    assert (tmp_path / "tower_panel_a.json").is_file()
    assert (tmp_path / "tower_panel_b.json").is_file()
    assert (tmp_path / "global_subtraction_roundoff_forensic.json").is_file()

    persisted = cli._load(tmp_path / "phase_observer_preflight_gate.json")
    before = _snapshot(tmp_path)
    assert cli._run_preflight_stage(tmp_path, args, provenance) == persisted
    assert _snapshot(tmp_path) == before


def test_reduced_main_freezes_all_plans_before_device_and_runs_both_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    uniform = np.full(cli.PATH_STATE_SIZE, 1.0 / cli.PATH_STATE_SIZE)
    provenance = {
        "schema": "test-provenance",
        "evaluation_status": "evaluated",
        "passed": 1,
        "provenance_valid": 1,
        "parent_re_adjudication": "tower_observer_roundoff_invalid",
    }

    monkeypatch.setattr(
        cli, "verify_tower_observer_roundoff_parent", lambda path: provenance
    )
    monkeypatch.setattr(cli, "_source_record", lambda path: ("source", []))
    monkeypatch.setattr(
        cli,
        "_load_parent_source_image",
        lambda path: {
            "image": uniform,
            "mixed_target": uniform,
            "image_sha256": cli.EXPECTED_IMAGE_SHA256,
            "mixed_target_sha256": cli.EXPECTED_MIXED_TARGET_SHA256,
            "parent_npz_sha256": "parent",
        },
    )

    def configure(device):
        run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
        required = {
            "phase_observer_path_id_plan.json",
            "scientific_config.json",
            "parent_provenance.json",
            "source_image.json",
            "source_image.npz",
            "run_manifest.json",
            "panel_a_plan.json",
            "panel_b_plan.json",
            "sealed_panel_registry.json",
            "distribution_free_power_impossibility.json",
        }
        assert required <= {path.name for path in run_dir.iterdir()}
        assert not (run_dir / "run_status.json").exists()
        assert cli._load(run_dir / "run_manifest.json")[
            "frozen_before_device_execution"
        ] == 1
        assert cli._load(run_dir / "sealed_panel_registry.json")[
            "panels_frozen_before_device_execution"
        ] == 1
        calls.append("device")
        return {"device_type": "cpu-test-double"}

    def preflight(run_dir, args, observed_provenance):
        assert calls == ["device"]
        assert observed_provenance == provenance
        gate = {
            "evaluation_status": "evaluated",
            "passed": 0,
            "subchecks": {
                "production_authorizing_pass": {"passed": 0},
                "tower_clusters_per_panel": {"passed": 0},
                "tower_bootstrap_replicates": {"passed": 0},
                "all_scientific_controls": {"passed": 1},
            },
        }
        cli.atomic_write_json(
            run_dir / "phase_observer_preflight_gate.json", gate
        )
        calls.append("preflight")
        return gate

    def pilot(run_dir, args, source):
        assert calls == ["device", "preflight"]
        assert np.array_equal(source["mixed_target"], uniform)
        gate = {
            "evaluation_status": "evaluated",
            "passed": 0,
            "numerically_valid": 1,
            "resource_valid": 1,
            "panel_a_nominated": 0,
        }
        cli.atomic_write_json(
            run_dir / "phase_observer_pilot_gate.json", gate
        )
        calls.append("pilot")
        return gate

    monkeypatch.setattr(cli, "configure_exact_torch_backend", configure)
    monkeypatch.setattr(cli, "_run_preflight_stage", preflight)
    monkeypatch.setattr(cli, "_run_pilot_stage", pilot)

    code = cli.main(
        [
            "--parent-dynkin-idfix-run-dir",
            "immutable-parent",
            "--runs-root",
            str(tmp_path),
            "--run-name",
            "tiny",
            "--stage",
            "all",
            "--device",
            "cpu",
            "--tower-panel-clusters",
            "2",
            "--pilot-panel-paths",
            "2",
            "--test-only-reduced-workload",
        ]
    )

    assert code == 0
    assert calls == ["device", "preflight", "pilot"]
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    status = cli._load(run_dir / "run_status.json")
    assert status["status"] == "complete"
    assert status["required_gate_pass"] == 1
    assert status["physical_training_performed"] == 0
    assert status["sampling_performed"] == 0
    assert status["reverse_sampling_performed"] == 0
    assert (run_dir / "artifact_registry.json").is_file()
