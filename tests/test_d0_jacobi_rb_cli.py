from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import mnist.diag_d0_jacobi_rb_denoising_feasibility as cli
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_spectral import JacobiRBSpectralProfile


def _parent(path: Path) -> dict[str, object]:
    return {
        "schema": "d0-jacobi-rao-blackwell-parent-provenance",
        "schema_version": 1,
        "passed": 1,
        "parent_run_dir": str(path),
        "parent_artifact_registry_sha256": cli.PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": 123,
        "parent_artifact_record_count": cli.PARENT_REGISTRY_RECORD_COUNT,
        "parent_scientific_fingerprint": "scientific",
        "parent_source_fingerprint": "source",
        "readjudicated_decision": "ancestral_representation_infeasible",
        "ddpm_population_target_preserved": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _passing_preflight() -> dict[str, object]:
    return {"gate": "jacobi_rb_preflight", "evaluation_status": "evaluated", "passed": 1}


def test_cli_contract_and_production_overrides() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([
            "--parent-jacobi-feasibility-run-dir", "parent",
            "--stage", "kernel",
        ])
    with pytest.raises(SystemExit):
        cli.parse_args([
            "--parent-jacobi-feasibility-run-dir", "parent",
            "--stage", "preflight",
            "--require-gate", "preflight",
            "--support-draws", "1",
        ])
    args = cli.parse_args([
        "--parent-jacobi-feasibility-run-dir", "parent",
        "--stage", "preflight",
        "--require-gate", "none",
        "--support-draws", "1",
        "--benchmark-path-transitions", "2",
    ])
    assert args.support_draws == 1
    assert args.benchmark_path_transitions == 2


def test_preflight_writes_terminal_evidence_and_no_work_flags(tmp_path, monkeypatch) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.setattr(cli, "verify_and_readjudicate_jacobi_parent", lambda _: _parent(parent))
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda _: {"deterministic": 1})

    def fake_preflight(args, run_dir, parent_record, device, profile):
        gate = _passing_preflight()
        cli.atomic_write_json(run_dir / "target_convention.json", {
            "stored_target": "Zbar=E[Z|X,Y,u]",
            "ddpm_population_target_preserved": 1,
            **cli._no_work(),
        })
        cli.atomic_write_json(run_dir / "jacobi_rb_preflight_gate.json", gate)
        return gate

    monkeypatch.setattr(cli, "_run_preflight", fake_preflight)
    result = cli.main([
        "--runs-root", str(tmp_path / "runs"),
        "--run-name", "tiny",
        "--device", "cpu",
        "--stage", "preflight",
        "--parent-jacobi-feasibility-run-dir", str(parent),
        "--require-gate", "none",
    ])
    assert result == 0
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    convention = json.loads((run_dir / "target_convention.json").read_text(encoding="utf-8"))
    assert status["physical_training_performed"] == 0
    assert status["reverse_sampling_performed"] == 0
    assert status["artifact_registry_sha256"]
    assert manifest["scientific_config"]["target"]["latent_formula"] == "Z=L-MY"
    assert manifest["scientific_config"]["target"]["classifier_or_value_target"] == 0
    assert convention["ddpm_population_target_preserved"] == 1
    assert (run_dir / "jacobi_rb_kernel_gate.json").is_file()
    assert (run_dir / "jacobi_rb_target_gate.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()


def test_parent_failure_commits_artifacts_before_required_failure(tmp_path, monkeypatch) -> None:
    parent = tmp_path / "invalid-parent"
    parent.mkdir()
    monkeypatch.setattr(
        cli,
        "verify_and_readjudicate_jacobi_parent",
        lambda _: (_ for _ in ()).throw(ArtifactCompatibilityError("registry mismatch")),
    )
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda _: {"deterministic": 1})
    result = cli.main([
        "--runs-root", str(tmp_path / "runs"),
        "--run-name", "bad-parent",
        "--device", "cpu",
        "--stage", "preflight",
        "--parent-jacobi-feasibility-run-dir", str(parent),
        "--require-gate", "preflight",
    ])
    assert result == 2
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "parent_provenance.json").read_text(encoding="utf-8"))
    decision = json.loads((run_dir / "jacobi_rb_decision.json").read_text(encoding="utf-8"))
    assert provenance["passed"] == 0
    assert decision["decision"] == "control_provenance_invalid"
    assert status["outcome"] == "gate_failed"
    assert status["artifact_registry_sha256"]


def test_unexpected_post_mutation_failure_is_atomically_finalized(
    tmp_path, monkeypatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.setattr(cli, "verify_and_readjudicate_jacobi_parent", lambda _: _parent(parent))
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda _: {"deterministic": 1})
    monkeypatch.setattr(
        cli, "_run_preflight", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("plot backend failed"))
    )
    result = cli.main([
        "--runs-root", str(tmp_path / "runs"),
        "--run-name", "unexpected",
        "--device", "cpu",
        "--stage", "preflight",
        "--parent-jacobi-feasibility-run-dir", str(parent),
        "--require-gate", "none",
    ])
    assert result == 2
    run_dir = next((tmp_path / "runs").iterdir())
    failure = json.loads((run_dir / "unexpected_failure.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "OSError"
    assert status["outcome"] == "error"
    assert status["artifact_registry_sha256"]
    assert (run_dir / "artifact_registry.json").is_file()


def test_required_gate_helper_selects_only_requested_gate() -> None:
    passed = {"passed": 1}
    failed = {"passed": 0}
    assert cli._requested_gate_pass("none", preflight=failed, kernel=failed, target=failed)
    assert cli._requested_gate_pass("preflight", preflight=passed, kernel=failed, target=failed)
    assert not cli._requested_gate_pass("kernel", preflight=failed, kernel=passed, target=failed)
    assert cli._requested_gate_pass("kernel", preflight=passed, kernel=passed, target=failed)
    assert not cli._requested_gate_pass("target", preflight=passed, kernel=passed, target=failed)


def test_resume_rejects_missing_directory(tmp_path) -> None:
    result = cli.main([
        "--stage", "kernel",
        "--resume-run-dir", str(tmp_path / "missing"),
        "--parent-jacobi-feasibility-run-dir", str(tmp_path / "parent"),
    ])
    assert result == 2


def test_production_support_panel_is_the_frozen_cartesian_product() -> None:
    args = SimpleNamespace(
        tau_eff=5.0e-5,
        sample_steps=512,
        grid_size=28,
        support_draws=294,
        root_seed=261121,
    )
    rows = cli._support_rows(args)
    assert len(rows) == 7 * 2 * 7 * 3 == 294
    assert [row["support_index"] for row in rows] == list(range(294))
    assert len({
        (
            row["pair_total"], row["duration_fraction"],
            row["head_fraction"], row["uniform_prefix_class"],
        )
        for row in rows
    }) == 294
    assert {row["pair_total"] for row in rows} == {
        1.0, 0.25, 0.1, 0.025, 2.0 / 784.0, 1.0e-3, 1.0e-5,
    }
    assert {row["duration_fraction"] for row in rows} == {0.5, 1.0}
    assert {row["uniform_prefix_class"] for row in rows} == {
        "extreme_low", "midrange", "extreme_high",
    }


def test_terminal_registry_is_strict_but_allows_interrupted_stage_recovery(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cli.atomic_write_json(run_dir / "jacobi_rb_kernel_gate.json", {"passed": 0})
    registry = cli._artifact_registry(run_dir)
    cli.atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_hash = cli.file_fingerprint(run_dir / "artifact_registry.json")
    cli.atomic_write_json(
        run_dir / "run_status.json",
        {"status": "complete", "artifact_registry_sha256": registry_hash},
    )
    cli._verify_terminal_registry(run_dir)

    cli.atomic_write_json(run_dir / "jacobi_rb_kernel_gate.json", {"passed": 1})
    with pytest.raises(ArtifactCompatibilityError):
        cli._verify_terminal_registry(run_dir)
    cli.atomic_write_json(
        run_dir / "run_status.json",
        {"status": "running", "artifact_registry_sha256": registry_hash},
    )
    (run_dir / "new-stage-evidence.json").write_text("{}", encoding="utf-8")
    cli._verify_terminal_registry(run_dir)

    cli.atomic_write_json(
        run_dir / "run_status.json",
        {"status": "complete", "artifact_registry_sha256": registry_hash},
    )
    with pytest.raises(ArtifactCompatibilityError):
        cli._verify_terminal_registry(run_dir)


def test_support_shards_resume_and_recover_corrupt_row(tmp_path, monkeypatch) -> None:
    args = SimpleNamespace(
        tau_eff=5.0e-5,
        sample_steps=512,
        grid_size=28,
        support_draws=1,
        root_seed=261121,
        no_progress=True,
    )
    calls = 0

    def fake_sample(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            later_head_fraction=np.asarray(0.375, dtype=np.float64),
            denoising_target=np.asarray(0.125, dtype=np.float64),
            certificate_codes=np.asarray(15, dtype=np.uint8),
            quantile_lower=np.asarray(0.375, dtype=np.float64),
            quantile_upper=np.asarray(0.375, dtype=np.float64),
            target_lower=np.asarray(0.125, dtype=np.float64),
            target_upper=np.asarray(0.125, dtype=np.float64),
            diagnostics=SimpleNamespace(
                certified=True,
                maximum_modes_used=32,
                interval_escalation_count=1,
            ),
        )

    monkeypatch.setattr(cli, "sample_alpha1_rb_transition_batch", fake_sample)
    rows, summary = cli._sample_support(args, JacobiRBSpectralProfile(), tmp_path)
    assert calls == 4
    assert rows[0]["certified"] == 1
    assert summary["certified_count"] == 1
    shard = next((tmp_path / "support_shards").glob("support-*.json"))
    initial_payload = json.loads(shard.read_text(encoding="utf-8"))
    assert initial_payload["schema"] == "d0-jacobi-rb-support-shard"
    assert initial_payload["schema_version"] == 1
    assert initial_payload["row_sha256"] == cli.config_fingerprint(
        initial_payload["row"]
    )
    assert initial_payload["row"]["certificate_code"] & 8
    assert initial_payload["physical_training_performed"] == 0
    assert initial_payload["reverse_sampling_performed"] == 0
    assert initial_payload["sampling_performed"] == 0

    def fail_if_sampled(*_args, **_kwargs):
        raise AssertionError("a valid completed support shard must be reused")

    monkeypatch.setattr(cli, "sample_alpha1_rb_transition_batch", fail_if_sampled)
    resumed_rows, resumed_summary = cli._sample_support(
        args, JacobiRBSpectralProfile(), tmp_path
    )
    assert resumed_rows == rows
    assert resumed_summary["certified_count"] == 1

    shard.write_text("{corrupt", encoding="utf-8")
    calls = 0
    monkeypatch.setattr(cli, "sample_alpha1_rb_transition_batch", fake_sample)
    recovered_rows, recovered_summary = cli._sample_support(
        args, JacobiRBSpectralProfile(), tmp_path
    )
    assert calls == 4
    assert recovered_rows[0]["certified"] == 1
    assert recovered_summary["certified_count"] == 1
    assert json.loads(shard.read_text(encoding="utf-8"))["row"]["certified"] == 1

    # Rehash each mutation so the test exercises semantic validation rather
    # than succeeding merely because the byte-integrity hash changed.
    def certified_code_without_rounding(payload):
        payload["row"]["certificate_code"] = 1

    def incomplete_certified_row(payload):
        del payload["row"]["target_upper"]

    def mismatched_support_index(payload):
        payload["row"]["support_index"] += 1

    def nonfinite_output(payload):
        payload["row"]["later_head_fraction"] = "NaN"

    def changed_no_work_flag(payload):
        payload["physical_training_performed"] = 1

    def changed_schema(payload):
        payload["schema_version"] = 2

    for mutate in (
        certified_code_without_rounding,
        incomplete_certified_row,
        mismatched_support_index,
        nonfinite_output,
        changed_no_work_flag,
        changed_schema,
    ):
        payload = json.loads(shard.read_text(encoding="utf-8"))
        mutate(payload)
        payload["row_sha256"] = cli.config_fingerprint(payload["row"])
        shard.write_text(json.dumps(payload), encoding="utf-8")
        calls = 0
        monkeypatch.setattr(cli, "sample_alpha1_rb_transition_batch", fake_sample)
        semantic_rows, semantic_summary = cli._sample_support(
            args, JacobiRBSpectralProfile(), tmp_path
        )
        assert calls == 4
        assert semantic_rows[0]["certified"] == 1
        assert semantic_rows[0]["certificate_code"] & 8
        assert semantic_summary["certified_count"] == 1
        repaired = json.loads(shard.read_text(encoding="utf-8"))
        assert repaired["schema_version"] == 1
        assert repaired["physical_training_performed"] == 0
        assert repaired["row_sha256"] == cli.config_fingerprint(repaired["row"])


def test_fresh_support_result_without_rounding_bit_fails_closed(
    tmp_path, monkeypatch
) -> None:
    args = SimpleNamespace(
        tau_eff=5.0e-5,
        sample_steps=512,
        grid_size=28,
        support_draws=1,
        root_seed=261121,
        no_progress=True,
    )

    def uncertified_rounding(*_args, **_kwargs):
        return SimpleNamespace(
            later_head_fraction=np.asarray(0.375, dtype=np.float64),
            denoising_target=np.asarray(0.125, dtype=np.float64),
            certificate_codes=np.asarray(1, dtype=np.uint8),
            quantile_lower=np.asarray(0.375, dtype=np.float64),
            quantile_upper=np.asarray(0.375, dtype=np.float64),
            target_lower=np.asarray(0.125, dtype=np.float64),
            target_upper=np.asarray(0.125, dtype=np.float64),
            diagnostics=SimpleNamespace(
                certified=True,
                maximum_modes_used=32,
                interval_escalation_count=0,
            ),
        )

    monkeypatch.setattr(
        cli, "sample_alpha1_rb_transition_batch", uncertified_rounding
    )
    rows, summary = cli._sample_support(
        args, JacobiRBSpectralProfile(), tmp_path
    )
    assert rows[0]["certified"] == 0
    assert rows[0]["failure_kind"] == "support_semantic_validation"
    assert summary["certified_count"] == 0
    assert summary["uncertified_draw_count"] == 1
    shard = next((tmp_path / "support_shards").glob("support-*.json"))
    payload = json.loads(shard.read_text(encoding="utf-8"))
    assert payload["row_sha256"] == cli.config_fingerprint(payload["row"])


def test_registered_post_support_failure_can_import_exact_shards(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    args = SimpleNamespace(
        tau_eff=5.0e-5,
        sample_steps=512,
        grid_size=28,
        support_draws=1,
        root_seed=261121,
        no_progress=True,
        support_shard_source_run_dir=None,
    )

    def fake_sample(*_args, **_kwargs):
        return SimpleNamespace(
            later_head_fraction=np.asarray(0.375, dtype=np.float64),
            denoising_target=np.asarray(0.125, dtype=np.float64),
            certificate_codes=np.asarray(15, dtype=np.uint8),
            quantile_lower=np.asarray(0.375, dtype=np.float64),
            quantile_upper=np.asarray(0.375, dtype=np.float64),
            target_lower=np.asarray(0.125, dtype=np.float64),
            target_upper=np.asarray(0.125, dtype=np.float64),
            diagnostics=SimpleNamespace(
                certified=True,
                maximum_modes_used=32,
                interval_escalation_count=1,
            ),
        )

    monkeypatch.setattr(cli, "sample_alpha1_rb_transition_batch", fake_sample)
    cli._sample_support(args, JacobiRBSpectralProfile(), source)
    cli.atomic_write_json(source / "jacobi_rb_preflight_gate.json", {"passed": 1})
    cli.atomic_write_json(source / "unexpected_failure.json", {
        "error_type": "JacobiRBCertificationError",
        "error": "Arb returned nonfinite endpoints",
        **cli._no_work(),
    })
    cli.atomic_write_json(source / "run_manifest.json", {
        "parent_artifact_registry_sha256": cli.PARENT_REGISTRY_SHA256,
    })
    registry = cli._artifact_registry(source)
    cli.atomic_write_json(source / "artifact_registry.json", registry)
    registry_hash = cli.file_fingerprint(source / "artifact_registry.json")
    cli.atomic_write_json(source / "run_status.json", {
        "status": "complete",
        "outcome": "error",
        "stage": "kernel",
        "artifact_registry_sha256": registry_hash,
        "artifact_registry_record_count": len(registry["records"]),
        **cli._no_work(),
    })

    args.support_shard_source_run_dir = source
    provenance = cli._import_support_shards(
        args, JacobiRBSpectralProfile(), destination
    )
    assert provenance is not None
    assert provenance["all_source_rows_certified"] == 1
    assert provenance["source_retry_count"] == 0
    assert provenance["validated_destination_count"] == 1
    imported = destination / "support_shards" / "support-0000.json"
    assert imported.read_bytes() == (
        source / "support_shards" / "support-0000.json"
    ).read_bytes()
    # Repeating the import is idempotent and leaves the frozen provenance
    # unchanged.
    second = cli._import_support_shards(
        args, JacobiRBSpectralProfile(), destination
    )
    assert second == provenance


def test_support_import_retries_only_obsolete_arb_nonfinite_rows(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    args = SimpleNamespace(
        tau_eff=5.0e-5,
        sample_steps=512,
        grid_size=28,
        support_draws=2,
        root_seed=261121,
        no_progress=True,
        support_shard_source_run_dir=None,
    )

    def fake_sample(*_args, **_kwargs):
        return SimpleNamespace(
            later_head_fraction=np.asarray(0.375, dtype=np.float64),
            denoising_target=np.asarray(0.125, dtype=np.float64),
            certificate_codes=np.asarray(15, dtype=np.uint8),
            quantile_lower=np.asarray(0.375, dtype=np.float64),
            quantile_upper=np.asarray(0.375, dtype=np.float64),
            target_lower=np.asarray(0.125, dtype=np.float64),
            target_upper=np.asarray(0.125, dtype=np.float64),
            diagnostics=SimpleNamespace(
                certified=True,
                maximum_modes_used=32,
                interval_escalation_count=1,
            ),
        )

    monkeypatch.setattr(cli, "sample_alpha1_rb_transition_batch", fake_sample)
    cli._sample_support(args, JacobiRBSpectralProfile(), source)
    retry_index = int(cli._support_rows(args)[1]["support_index"])
    retry_path = source / "support_shards" / f"support-{retry_index:04d}.json"
    retry_shard = json.loads(retry_path.read_text(encoding="utf-8"))
    retry_shard["row"].update({
        "certified": 0,
        "failure": "Arb returned nonfinite endpoints",
        "failure_kind": "arb_nonfinite",
    })
    retry_shard["row_sha256"] = cli.config_fingerprint(retry_shard["row"])
    cli.atomic_write_json(retry_path, retry_shard)
    cli.atomic_write_json(source / "jacobi_rb_preflight_gate.json", {"passed": 1})
    cli.atomic_write_json(source / "unexpected_failure.json", {
        "error_type": "JacobiRBCertificationError",
        "error": "Arb returned nonfinite endpoints",
        **cli._no_work(),
    })
    cli.atomic_write_json(source / "run_manifest.json", {
        "parent_artifact_registry_sha256": cli.PARENT_REGISTRY_SHA256,
    })
    registry = cli._artifact_registry(source)
    cli.atomic_write_json(source / "artifact_registry.json", registry)
    cli.atomic_write_json(source / "run_status.json", {
        "status": "complete",
        "outcome": "error",
        "stage": "kernel",
        "artifact_registry_sha256": cli.file_fingerprint(
            source / "artifact_registry.json"
        ),
        "artifact_registry_record_count": len(registry["records"]),
        **cli._no_work(),
    })

    args.support_shard_source_run_dir = source
    provenance = cli._import_support_shards(
        args, JacobiRBSpectralProfile(), destination
    )
    assert provenance is not None
    assert provenance["source_certified_count"] == 1
    assert provenance["source_retry_count"] == 1
    assert provenance["all_source_rows_semantically_valid"] == 1
    assert provenance["all_source_rows_certified"] == 0
    assert (destination / "support_shards" / "support-0000.json").is_file()
    assert not (
        destination / "support_shards" / f"support-{retry_index:04d}.json"
    ).exists()


def test_full_path_benchmark_writes_complete_restartable_shards(
    tmp_path, monkeypatch
) -> None:
    args = SimpleNamespace(
        benchmark_path_transitions=4,
        benchmark_repeats=3,
        benchmark_chunk_size=4,
        projected_transition_count=12,
        maximum_projected_cache_hours=1.0e6,
        grid_size=2,
        sample_steps=1,
        tau_eff=1.0e-2,
        root_seed=17,
    )
    calls = 0

    def fake_hybrid(x, exposure, **_kwargs):
        nonlocal calls
        calls += 1
        values = np.asarray(x, dtype=np.float64)
        active = np.asarray(exposure, dtype=np.float64) > 0.0
        return SimpleNamespace(
            later_head_fraction=np.clip(values + 0.01, 0.0, 1.0),
            denoising_target=np.zeros_like(values),
            certificate_codes=np.where(active, 15, 0).astype(np.uint8),
            active_mask=active,
        )

    monkeypatch.setattr(cli, "_benchmark_certified_call", fake_hybrid)
    rows, summary = cli._benchmark_sampler(
        args, JacobiRBSpectralProfile(), cli.torch.device("cpu"), tmp_path
    )
    assert summary["full_api_completed"] == 1
    assert summary["output_hashes_identical"] == 1
    assert summary["completed_repeats"] == 3
    assert len(list((tmp_path / "benchmark_shards").glob("repeat-*.npz"))) == 3
    assert all(row["completed_transitions"] == 4 for row in rows if row["repeat"] >= 0)
    first = cli._load_benchmark_shard(
        tmp_path / "benchmark_shards" / "repeat-00.npz",
        input_fingerprint=cli._benchmark_input_fingerprint(
            args, JacobiRBSpectralProfile(), cli.torch.device("cpu")
        ),
        requested=4,
    )
    assert first is not None
    assert first["later"].shape == (4,)

    calls = 0
    _, resumed = cli._benchmark_sampler(
        args, JacobiRBSpectralProfile(), cli.torch.device("cpu"), tmp_path
    )
    assert resumed["full_api_completed"] == 1
    assert calls == 1  # the resource probe runs; all complete repeats are reused

    (tmp_path / "benchmark_shards" / "repeat-01.npz").write_bytes(b"corrupt")
    calls = 0
    _, recovered = cli._benchmark_sampler(
        args, JacobiRBSpectralProfile(), cli.torch.device("cpu"), tmp_path
    )
    assert recovered["full_api_completed"] == 1
    assert recovered["output_hashes_identical"] == 1
    assert calls == 3  # one probe plus two matching calls for the corrupt repeat


def test_controls_only_import_does_not_load_trainers_or_reverse_sampler() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import mnist.diag_d0_jacobi_rb_denoising_feasibility; "
            "assert 'mnist.experiment12_d0' not in sys.modules; "
            "assert 'mnist.eulerian_flux_mnist' not in sys.modules; "
            "assert 'mnist.d0_one_image_sampler' not in sys.modules"
        ),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
