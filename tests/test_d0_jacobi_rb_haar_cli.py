from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mnist.d0_jacobi_rb_haar_cuda as haar_cuda
import mnist.diag_d0_jacobi_rb_hierarchical_coupling_confirmation as cli


def _base(*extra: str) -> list[str]:
    return ["--parent-phase-observer-run-dir", "parent", *extra]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _pass_gate(name: str) -> dict[str, object]:
    return {
        "schema": f"test-{name}",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


def test_cli_frozen_defaults_and_stage_requirements() -> None:
    args = cli.parse_args(_base("--stage", "preflight"))
    assert args.root_seed == 261_181
    assert args.panel_clusters == 8
    assert args.device == "cuda"

    for stage in ("coupling", "pilot", "report"):
        with pytest.raises(SystemExit):
            cli.parse_args(_base("--stage", stage))

    with pytest.raises(SystemExit):
        cli.parse_args(
            _base("--stage", "preflight", "--require-gate", "coupling")
        )

    resumed = cli.parse_args(
        _base(
            "--stage",
            "coupling",
            "--resume-run-dir",
            "run",
            "--require-gate",
            "coupling",
        )
    )
    assert resumed.resume_run_dir == Path("run")


def test_production_overrides_require_test_flag_and_test_cannot_gate() -> None:
    for option, value in (("--root-seed", "7"), ("--panel-clusters", "2")):
        with pytest.raises(SystemExit):
            cli.parse_args(_base(option, value))

    reduced = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--root-seed",
            "7",
            "--panel-clusters",
            "2",
            "--test-only-reduced-workload",
        )
    )
    assert reduced.root_seed == 7
    assert reduced.panel_clusters == 2

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


def test_path_plan_freezes_slots_and_only_intentional_level_aliases() -> None:
    args = cli.parse_args(_base("--stage", "preflight"))
    plan = cli._build_path_id_plan(args)
    profiles = plan["profiles"]

    nested_a = profiles[cli.NESTED_HAAR_PROFILE]["a"]["roles"]
    assert nested_a["main"]["levels"]["128"] == nested_a["main"]["levels"]["1024"]
    assert (
        nested_a["reference"]["levels"]["512"]
        == nested_a["reference"]["levels"]["2048"]
    )
    antithetic_a = profiles[cli.ANTITHETIC_HAAR_PROFILE]["a"]["roles"]
    assert antithetic_a["128-256"]["fine_detail_signs"] == [1, -1]

    role_sets = [
        set(profiles[profile][panel]["path_ids"])
        for profile in cli.PROFILE_ORDER
        for panel in ("a", "b")
    ]
    role_sets += [
        set(plan["marginal_panels"][panel]["path_ids"])
        for panel in ("a", "b")
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(role_sets)
        for right in role_sets[index + 1 :]
    )
    assert all(value < 0xF0000 for values in role_sets for value in values)
    assert plan["reserved_production_slot"] == [0xF0000, 0x100000]
    assert plan["maximum_path_id"] < 1 << 20


def test_plan_freeze_is_idempotent_and_tamper_evident(tmp_path: Path) -> None:
    args = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--panel-clusters",
            "2",
            "--test-only-reduced-workload",
        )
    )
    cli._freeze_plans(tmp_path, args, require_existing=False)
    before = _snapshot(tmp_path)
    cli._freeze_plans(tmp_path, args, require_existing=True)
    assert _snapshot(tmp_path) == before

    plan = cli._load(tmp_path / "haar_path_id_plan.json")
    plan["maximum_path_id"] += 1
    cli.atomic_write_json(tmp_path / "haar_path_id_plan.json", plan)
    corrupt = _snapshot(tmp_path)
    with pytest.raises(cli.ArtifactCompatibilityError, match="frozen"):
        cli._freeze_plans(tmp_path, args, require_existing=True)
    assert _snapshot(tmp_path) == corrupt


def test_real_reduced_backend_smoke_is_exact_and_replayable() -> None:
    args = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--panel-clusters",
            "1",
            "--test-only-reduced-workload",
        )
    )
    first = cli._backend_smoke(
        args,
        role="nested_a",
        path_ids=[0xA0000],
        sample_steps=128,
    )
    second = cli._backend_smoke(
        args,
        role="nested_a",
        path_ids=[0xA0000],
        sample_steps=128,
    )
    assert first["certificate_count"] == first["certificate_denominator"]
    assert first["jacobi_output_pass"] == 1
    assert first["fallback_count"] == first["fallback_denominator"]
    assert first["normal_fallback_count"] == first["sample_count"]
    assert first["jacobi_fallback_count"] == first["sample_count"]
    assert first["uncertified_count"] == 0
    assert first["later_sha256"] == second["later_sha256"]
    assert first["target_sha256"] == second["target_sha256"]


def test_uncertified_count_is_derived_from_required_certificate_masks() -> None:
    normal = np.array([True, False, True, False], dtype=bool)
    jacobi = np.array([True, True, False, True], dtype=bool)

    assert cli._measured_uncertified_count(normal, jacobi, {}) == 3
    assert (
        cli._measured_uncertified_count(
            normal,
            jacobi,
            {"uncertified_count": np.array(1, dtype=np.int64)},
        )
        == 3
    )


def test_uncertified_diagnostic_must_match_certificate_mask() -> None:
    normal = np.ones(4, dtype=bool)
    jacobi = np.array([True, False, True, True], dtype=bool)

    with pytest.raises(cli.HaarExecutionError, match="disagrees"):
        cli._measured_uncertified_count(
            normal,
            jacobi,
            {"uncertified_count": np.array(0, dtype=np.int64)},
        )
    with pytest.raises(cli.HaarExecutionError, match="invalid"):
        cli._measured_uncertified_count(
            normal,
            jacobi,
            {"uncertified_count": np.array(1.0, dtype=np.float64)},
        )


def test_backend_smoke_derives_omitted_uncertified_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--panel-clusters",
            "1",
            "--test-only-reduced-workload",
        )
    )
    original = (
        haar_cuda.sample_alpha1_rb_transition_batch_from_uniform_cells_cpu
    )

    def omit_redundant_counter(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        return replace(
            result,
            diagnostics={
                name: value
                for name, value in result.diagnostics.items()
                if name != "uncertified_count"
            },
        )

    monkeypatch.setattr(
        haar_cuda,
        "sample_alpha1_rb_transition_batch_from_uniform_cells_cpu",
        omit_redundant_counter,
    )
    record = cli._backend_smoke(
        args,
        role="nested_a",
        path_ids=[0xA0000],
        sample_steps=128,
    )

    assert record["certificate_count"] == record["certificate_denominator"]
    assert record["uncertified_count"] == 0


def test_source_record_binds_fused_backend_and_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_source = tmp_path / "parent_source.py"
    parent_source.write_text("PARENT = 1\n", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        '{"source_paths": [' + f'"{parent_source.as_posix()}"' + "]}",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "PARENT_SOURCE_COUNT", 1)

    _fingerprint, paths = cli._source_record(tmp_path)
    names = {Path(path).name for path in paths}

    assert "d0_jacobi_rb_haar_fused.py" in names
    assert "d0_jacobi_rb_haar_scheduler.py" in names


def test_backend_smoke_rejects_missing_transition_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = cli.parse_args(
        _base(
            "--device",
            "cpu",
            "--panel-clusters",
            "1",
            "--test-only-reduced-workload",
        )
    )

    def missing_certificate(
        head: np.ndarray,
        exposure: np.ndarray,
        cells: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        count = int(np.asarray(head).size)
        return SimpleNamespace(
            later_head_fraction=np.asarray(head, dtype=np.float64),
            denoising_target=np.zeros(count, dtype=np.float64),
            fallback_mask=np.zeros(count, dtype=bool),
        )

    monkeypatch.setattr(
        haar_cuda,
        "sample_alpha1_rb_transition_batch_from_uniform_cells_cpu",
        missing_certificate,
    )
    with pytest.raises(cli.HaarExecutionError, match="certified_mask"):
        cli._backend_smoke(
            args,
            role="nested_a",
            path_ids=[0xA0000],
            sample_steps=128,
        )


def test_missing_measured_controls_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        panel_clusters=8,
        test_only_reduced_workload=False,
        root_seed=cli.ROOT_SEED,
        device="cpu",
    )
    cli._freeze_plans(tmp_path, args, require_existing=False)

    def smoke(
        args: object,
        *,
        phase: int = 0,
        **kwargs: object,
    ) -> dict[str, object]:
        return {
            "sample_count": 4,
            "certificate_count": 8,
            "certificate_denominator": 8,
            "fallback_count": 0,
            "fallback_denominator": 8,
            "maximum_prefix_bits": 128,
            "normal_interval_pass": 1,
            "uniform_interval_pass": 1,
            "jacobi_output_pass": 1,
            "normal_cells_certified_pass": 1,
            "uniform_cells_certified_pass": 1,
            "jacobi_outputs_certified_pass": 1,
            "transition_id_unique_pass": 1,
            "phase": phase,
            "facet_pass": 1,
            "zero_duration_pass": 1,
            "later_sha256": "a" * 64,
            "target_sha256": "b" * 64,
            "uniform_lower_sha256": "c" * 64,
            "uniform_upper_sha256": "d" * 64,
            "elapsed_seconds": 1.0,
            "fallback_elapsed_seconds": 0.0,
            "fused_cuda_normal_authorizer_pass": 1,
            "arbitrary_uniform_cuda_authorizer_pass": 1,
            **{name: 0 for name in cli._base_forbidden()},
        }

    monkeypatch.setattr(cli, "_backend_smoke", smoke)
    monkeypatch.setattr(
        cli,
        "_load_or_run_preflight_controls",
        lambda *args, **kwargs: ({}, {}, {}),
    )
    provenance = {
        "passed": 1,
        "parent_re_adjudication": "right_endpoint_coupling_power_infeasible",
        "parent_source_fingerprint": cli.PARENT_SOURCE_FINGERPRINT,
        "parent_preflight_pass": 1,
        "parent_pilot_numerically_valid": 1,
        "parent_pilot_resource_valid": 1,
        "parent_pilot_power_valid": 0,
        "parent_artifact_record_count": cli.PARENT_REGISTRY_RECORD_COUNT,
        "parent_source_count": cli.PARENT_SOURCE_COUNT,
    }

    preflight = cli._collect_preflight_metrics(tmp_path, args, provenance)
    for name in (
        "haar_marginal_controls.json",
        "haar_phase_tower_controls.json",
        "haar_coupling_scheduler_benchmark.json",
    ):
        cli.atomic_write_json(tmp_path / name, {})
    coupling = cli._collect_coupling_metrics(tmp_path, args)

    assert preflight["jacobi_marginal_cdf_pass"] == 0
    assert preflight["phase_tower_identity_pass"] == 0
    assert preflight["candidate_under_48h_forecast_pass"] == 0
    assert preflight["mass_error"] is None
    assert cli.evaluate_haar_preflight(preflight)["passed"] == 0
    assert coupling["nested_profile_complete_pass"] == 0
    assert coupling["marginal_eigenmoment_pass"] == 0
    assert coupling["pipeline_runtime_projection_pass"] == 0
    assert coupling["mass_error"] is None
    assert cli.evaluate_haar_coupling(coupling)["passed"] == 0

    candidate = cli._candidate_row(
        profile=cli.ANTITHETIC_HAAR_PROFILE,
        main_paths=16,
        reference_paths=16,
        main_width=1.0e-4,
        reference_width=2.0e-4,
        projected_hours=1.0,
        rate=2000.0,
    )
    nomination = cli.nominate_haar_power_design(
        profile=cli.ANTITHETIC_HAAR_PROFILE,
        panel_role="a",
        candidates=[candidate],
    )
    assert nomination["selected"] is None


def test_sealed_profile_order_opens_antithetic_only_after_nested_a_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        test_only_reduced_workload=True,
        panel_clusters=8,
    )
    calls: list[tuple[str, str]] = []
    cli._freeze_plans(tmp_path, args, require_existing=False)
    measured = {
        name: 1
        for name in (
            "panel_complete_pass",
            "panel_finite_pass",
            "panel_certification_pass",
            "panel_numerical_health_pass",
            "mass_conservation_pass",
            "shard_chain_pass",
            "pilot_production_isolation_pass",
            "pilot_means_excluded_pass",
            "raw_endpoint_authorizing_pass",
            "dynkin_advisory_only_pass",
        )
    }

    nested_candidates = [
        cli._candidate_row(
            profile=cli.NESTED_HAAR_PROFILE,
            main_paths=main,
            reference_paths=reference,
            main_width=1.0,
            reference_width=1.0,
            projected_hours=1.0,
            rate=2000.0,
            measured=measured,
        )
        for main in (32, 64)
        for reference in (16, 32)
    ]
    nested_a = cli.nominate_haar_power_design(
        profile=cli.NESTED_HAAR_PROFILE,
        panel_role="a",
        candidates=nested_candidates,
    )
    anti_row = cli._candidate_row(
        profile=cli.ANTITHETIC_HAAR_PROFILE,
        main_paths=16,
        reference_paths=16,
        main_width=1.0e-4,
        reference_width=2.0e-4,
        projected_hours=2.0,
        rate=2000.0,
        measured=measured,
    )
    anti_a = cli.nominate_haar_power_design(
        profile=cli.ANTITHETIC_HAAR_PROFILE,
        panel_role="a",
        candidates=[anti_row],
    )

    def panel(
        run_dir: Path,
        args: object,
        *,
        profile: str,
        panel: str,
        selected: object = None,
    ) -> dict[str, object]:
        calls.append((profile, panel))
        if profile == cli.NESTED_HAAR_PROFILE:
            assert panel == "a" and selected is None
            return nested_a
        if panel == "a":
            assert selected is None
            return anti_a
        assert selected is not None
        return {
            "evaluation_status": "evaluated",
            "profile": cli.ANTITHETIC_HAAR_PROFILE,
            "main_paths": 16,
            "reference_paths": 16,
            "complete_pass": 1,
            "finite_pass": 1,
            "certification_pass": 1,
            "numerical_health_pass": 1,
            "mass_conservation_pass": 1,
            "shard_chain_pass": 1,
            "main_half_width": 1.0e-4,
            "generator_reference_half_width": 2.0e-4,
            "reference_stability_half_width": 2.0e-4,
            "projected_hours": 2.0,
            "minimum_rate": 2000.0,
        }

    monkeypatch.setattr(cli, "_run_profile_panel", panel)
    gate = cli._run_pilot_stage(tmp_path, args)
    selection = cli._load(tmp_path / "sealed_profile_selection.json")

    assert calls == [
        (cli.NESTED_HAAR_PROFILE, "a"),
        (cli.ANTITHETIC_HAAR_PROFILE, "a"),
        (cli.ANTITHETIC_HAAR_PROFILE, "b"),
    ]
    assert selection["selected_profile"] == cli.ANTITHETIC_HAAR_PROFILE
    assert selection["panels_agree"] == 1
    # Reduced evidence can exercise sealing but is never authorizing.
    assert gate["passed"] == 0


def test_reused_panel_a_is_rederived_from_verified_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    cli.atomic_write_json(
        parent / "source_image.json",
        {"source_npz_sha256": "source"},
    )
    args = SimpleNamespace(
        test_only_reduced_workload=False,
        panel_clusters=8,
        root_seed=cli.ROOT_SEED,
        parent_phase_observer_run_dir=parent,
        device="cuda",
    )
    cli._freeze_plans(tmp_path, args, require_existing=False)
    path_plan = cli._load(tmp_path / "haar_path_id_plan.json")
    roles = path_plan["profiles"][cli.NESTED_HAAR_PROFILE]["a"]["roles"]
    measured = {
        name: 1
        for name in (
            "panel_complete_pass",
            "panel_finite_pass",
            "panel_certification_pass",
            "panel_numerical_health_pass",
            "mass_conservation_pass",
            "shard_chain_pass",
            "pilot_production_isolation_pass",
            "pilot_means_excluded_pass",
            "raw_endpoint_authorizing_pass",
            "dynkin_advisory_only_pass",
        )
    }
    candidates = [
        cli._candidate_row(
            profile=cli.NESTED_HAAR_PROFILE,
            main_paths=main,
            reference_paths=reference,
            main_width=1.0e-4,
            reference_width=2.0e-4,
            projected_hours=float(main + reference),
            rate=2000.0,
            measured=measured,
        )
        for main in (32, 64)
        for reference in (16, 32)
    ]
    evidence = {
        "profile": cli.NESTED_HAAR_PROFILE,
        "panel": "a",
        "root_seed": cli.ROOT_SEED,
        "path_id_plan_sha256": path_plan["path_id_plan_sha256"],
        "path_id_pools": {
            "main": roles["main"]["root_path_ids"],
            "reference": roles["reference"]["root_path_ids"],
        },
        "main_path_ids": roles["main"]["root_path_ids"],
        "reference_path_ids": roles["reference"]["root_path_ids"],
        "source_npz_sha256": "source",
        "candidates": candidates,
        "observable_payload": {"path": "payload.npz", "sha256": "payload"},
    }
    evidence_path = (
        tmp_path / f"{cli.NESTED_HAAR_PROFILE}_panel_a_evidence.json"
    )
    cli.atomic_write_json(evidence_path, {"sealed": 1})
    monkeypatch.setattr(
        cli,
        "verify_certified_haar_power_panel_evidence",
        lambda **kwargs: evidence,
    )
    binding = {
        "path": evidence_path.name,
        "sha256": cli.file_fingerprint(evidence_path),
        "observable_payload": evidence["observable_payload"],
    }
    expected = cli._rederive_profile_panel_record(
        evidence,
        profile=cli.NESTED_HAAR_PROFILE,
        panel="a",
        selected=None,
        evidence_binding=binding,
    )
    panel_path = tmp_path / f"{cli.NESTED_HAAR_PROFILE}_panel_a.json"
    cli.atomic_write_json(panel_path, expected)

    assert (
        cli._run_profile_panel(
            tmp_path,
            args,
            profile=cli.NESTED_HAAR_PROFILE,
            panel="a",
        )
        == expected
    )

    changed = {**expected, "selection_status": "tampered"}
    cli.atomic_write_json(panel_path, changed)
    with pytest.raises(
        cli.ArtifactCompatibilityError,
        match="panel a record changed",
    ):
        cli._run_profile_panel(
            tmp_path,
            args,
            profile=cli.NESTED_HAAR_PROFILE,
            panel="a",
        )

    numeric_type_drift = {
        **expected,
        "eligible_candidate_count": float(
            expected["eligible_candidate_count"]
        ),
    }
    assert numeric_type_drift == expected
    cli.atomic_write_json(panel_path, numeric_type_drift)
    with pytest.raises(
        cli.ArtifactCompatibilityError,
        match="panel a record changed",
    ):
        cli._run_profile_panel(
            tmp_path,
            args,
            profile=cli.NESTED_HAAR_PROFILE,
            panel="a",
        )


def test_required_failure_writes_decision_registry_and_status_first(
    tmp_path: Path,
) -> None:
    cli.atomic_write_json(tmp_path / "evidence.json", {"complete": 1})
    provenance = _pass_gate("provenance")
    preflight = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "rng_algebra_valid": 0,
    }
    coupling = cli.not_evaluated_gate("coupling", "preflight failed")
    pilot = cli.not_evaluated_gate("pilot", "preflight failed")
    code = cli._finish(
        tmp_path,
        SimpleNamespace(require_gate="preflight", stage="preflight"),
        provenance=provenance,
        preflight=preflight,
        coupling=coupling,
        pilot=pilot,
    )

    assert code == 1
    assert (tmp_path / "haar_workflow_gate.json").is_file()
    assert (tmp_path / "haar_coupling_decision.json").is_file()
    assert (tmp_path / "artifact_registry.json").is_file()
    status = cli._load(tmp_path / "run_status.json")
    assert status["outcome"] == "gate_failed"
    assert status["required_gate_pass"] == 0
    assert status["physical_training_performed"] == 0
    assert status["production_refinement_performed"] == 0


def test_typed_scheduler_adapter_failure_is_fail_closed_and_restart_incompatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = {
        **_pass_gate("provenance"),
        "parent_re_adjudication": "right_endpoint_coupling_power_infeasible",
    }
    monkeypatch.setattr(
        cli, "verify_right_endpoint_coupling_parent", lambda path: provenance
    )
    source_hash = {"value": "a" * 64}
    monkeypatch.setattr(
        cli,
        "_source_record",
        lambda parent: (source_hash["value"], [str(Path(__file__))]),
    )
    monkeypatch.setattr(
        cli,
        "configure_exact_torch_backend",
        lambda device: {"device": str(device), "exact": 1},
    )

    def fail_adapter(
        run_dir: Path, args: object, parent: object
    ) -> dict[str, object]:
        raise cli.HaarSchedulerError(
            "certificate tensor shape does not match flattened transition input",
            failure_domain="scheduler_execution",
            failure_code="hierarchical_interval_adapter_shape_invalid",
            transition_shape=[3136],
            certificate_shape=[8, 392],
        )

    monkeypatch.setattr(cli, "_run_preflight_stage", fail_adapter)
    code = cli.main(
        [
            "--parent-phase-observer-run-dir",
            "parent",
            "--runs-root",
            str(tmp_path),
            "--run-name",
            "adapter-failure",
            "--stage",
            "preflight",
            "--device",
            "cuda",
            "--require-gate",
            "preflight",
        ]
    )
    assert code == 1
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())

    failure = cli._load(run_dir / "preflight_failure.json")
    gate = cli._load(run_dir / "haar_preflight_gate.json")
    decision = cli._load(run_dir / "haar_coupling_decision.json")
    status = cli._load(run_dir / "run_status.json")
    assert failure == gate
    assert failure["evaluation_status"] == "execution_failed"
    assert failure["passed"] == 0
    assert failure["stage_execution_valid"] == 0
    assert failure["scientific_evidence_complete"] == 0
    assert failure["failure_domain"] == "scheduler_execution"
    assert (
        failure["failure_code"]
        == "hierarchical_interval_adapter_shape_invalid"
    )
    assert failure["failure_diagnostics"] == {
        "certificate_shape": [8, 392],
        "transition_shape": [3136],
    }
    assert decision["decision"] == "hierarchical_scheduler_invalid"
    assert status["status"] == "complete"
    assert status["outcome"] == "gate_failed"
    assert status["required_gate_pass"] == 0
    assert (run_dir / "haar_workflow_gate.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()
    registered = set(
        cli._load(run_dir / "artifact_registry.json")["records"]
    )
    assert {
        "preflight_failure.json",
        "haar_preflight_gate.json",
        "haar_workflow_gate.json",
        "haar_coupling_decision.json",
    }.issubset(registered)

    # A source change makes this completed failed run incompatible.  The
    # rejection happens before any artifact in the old directory is mutated.
    before = _snapshot(run_dir)
    source_hash["value"] = "b" * 64
    resume_code = cli.main(
        [
            "--parent-phase-observer-run-dir",
            "parent",
            "--resume-run-dir",
            str(run_dir),
            "--stage",
            "preflight",
            "--device",
            "cuda",
            "--require-gate",
            "preflight",
        ]
    )
    assert resume_code == 2
    assert _snapshot(run_dir) == before


def test_tiny_all_stage_and_report_only_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = {
        **_pass_gate("provenance"),
        "parent_re_adjudication": "right_endpoint_coupling_power_infeasible",
    }
    monkeypatch.setattr(
        cli, "verify_right_endpoint_coupling_parent", lambda path: provenance
    )
    monkeypatch.setattr(
        cli, "_source_record", lambda parent: ("a" * 64, [str(Path(__file__))])
    )
    monkeypatch.setattr(
        cli,
        "configure_exact_torch_backend",
        lambda device: {"device": str(device), "exact": 1},
    )

    calls: list[str] = []

    def preflight(run_dir: Path, args: object, parent: object) -> dict[str, object]:
        calls.append("preflight")
        gate = _pass_gate("preflight")
        cli.atomic_write_json(run_dir / "haar_preflight_gate.json", gate)
        return gate

    def coupling(run_dir: Path, args: object) -> dict[str, object]:
        calls.append("coupling")
        gate = _pass_gate("coupling")
        cli.atomic_write_json(run_dir / "haar_coupling_gate.json", gate)
        return gate

    def pilot(run_dir: Path, args: object) -> dict[str, object]:
        calls.append("pilot")
        gate = _pass_gate("pilot")
        cli.atomic_write_json(run_dir / "haar_pilot_gate.json", gate)
        return gate

    monkeypatch.setattr(cli, "_run_preflight_stage", preflight)
    monkeypatch.setattr(cli, "_run_coupling_stage", coupling)
    monkeypatch.setattr(cli, "_run_pilot_stage", pilot)

    code = cli.main(
        [
            "--parent-phase-observer-run-dir",
            "parent",
            "--runs-root",
            str(tmp_path),
            "--run-name",
            "tiny",
            "--stage",
            "all",
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        ]
    )
    assert code == 0
    assert calls == ["preflight", "coupling", "pilot"]
    run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    first_registry = cli.file_fingerprint(run_dir / "artifact_registry.json")

    calls.clear()
    code = cli.main(
        [
            "--parent-phase-observer-run-dir",
            "parent",
            "--stage",
            "report",
            "--resume-run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--test-only-reduced-workload",
        ]
    )
    assert code == 0
    assert calls == []
    assert cli.file_fingerprint(run_dir / "artifact_registry.json") == first_registry
    status = cli._load(run_dir / "run_status.json")
    assert status["phase"] == "report"
    assert status["status"] == "complete"


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
