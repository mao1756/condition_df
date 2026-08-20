from __future__ import annotations

import csv
from dataclasses import fields
import hashlib
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mnist.d0_jacobi_rb_boundary_tangent as d0_tangent
import mnist.d0_jacobi_rb_reverse_controller as d0_reverse
import mnist.eulerian_jacobi_ddpm as ddpm
from mnist.d0_jacobi_rb_global_dilated import (
    GLOBAL_DILATED_PARAMETER_COUNT,
    GlobalDilatedZeroBaselinePredictor,
)
from mnist.d0_jacobi_rb_spectral import (
    JacobiRBSpectralProfile,
    philox_uniform_prefix,
    propose_alpha1_rb_transition_batch_torch,
)


def _model_inputs(
    state: torch.Tensor | None = None,
    *,
    phase: int = 0,
    label: int = 3,
) -> ddpm.ModelInputs:
    if state is None:
        state = torch.full(
            (2, ddpm.STATE_SIZE),
            1.0 / ddpm.STATE_SIZE,
            dtype=torch.float64,
        )
    batch = state.shape[0]
    return ddpm.ModelInputs(
        later_full_state=state,
        reverse_time=torch.full((batch,), 0.5, dtype=state.dtype),
        phase=torch.full((batch,), phase, dtype=torch.long),
        color=torch.full((batch,), ddpm.PHASE_MATCHINGS[phase], dtype=torch.long),
        duration=torch.full(
            (batch,), ddpm.PHASE_DURATIONS[phase], dtype=state.dtype
        ),
        label=torch.full((batch,), label, dtype=torch.long),
    )


def _as_record(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    method = getattr(value, "to_record", None)
    assert callable(method)
    record = method()
    assert isinstance(record, dict)
    return record


def test_frozen_config_is_balanced_global_k128_and_fixed_scale() -> None:
    config = _as_record(ddpm.frozen_config())

    assert ddpm.OUTER_STEPS == 128
    assert ddpm.REFERENCE_OUTER_STEPS == 512
    assert ddpm.TRAIN_PATH_COUNT == 4_000
    assert ddpm.VALIDATION_PATH_COUNT == 1_000
    assert ddpm.RASTER_SCALE == 25_471 / 255
    assert config["research_mode"] == "exploratory"
    assert config["training_paths"] == 4_000
    assert config["validation_paths"] == 1_000
    assert config["paths_per_train_class"] == 400
    assert config["paths_per_validation_class"] == 100
    assert config["outer_steps"] == 128
    assert config["reference_outer_steps"] == 512
    assert config["full_forward_transition_count"] == 1_756_160_000
    assert config["raster_scale"] == 25_471 / 255
    assert config["model"] == "GlobalDilatedZeroBaselinePredictor"
    assert config["model_parameter_count"] == GLOBAL_DILATED_PARAMETER_COUNT


def test_runner_budgets_the_full_cuda_workload_without_silent_downscaling() -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    config = runner.FROZEN_CONFIG
    assert config["records"]["projected_cache_pair_transitions"] == 1_756_160_000
    assert config["resource_projection"]["full_cache_pair_transitions"] == 1_756_160_000
    assert config["resource_projection"]["expected_accelerator_seconds"] > 0
    assert config["records"]["train_paths"] == 4_000
    assert config["records"]["validation_paths"] == 1_000
    assert config["chain"]["outer_steps"] == 128
    assert config["chain"]["controller_microsteps"] == 2
    assert config["chain"]["model_evaluations_per_path"] == 1_792
    assert config["model"]["architecture_fallback"] is None
    pilot = config["objective_pilot"]
    assert pilot["train_paths"] == 250
    assert pilot["validation_paths"] == 100
    assert 500 <= pilot["training_updates"] <= 1_000
    assert pilot["prior_paths"] == 20
    assert pilot["forward_terminal_paths"] == 20
    assert pilot["oracle_paths"] == 10
    assert pilot["all_ten_classes_required"] == 1
    assert pilot["full_cache_launch_requires_pass"] == 1
    assert pilot["projected_cache_pair_transitions"] == 122_931_200
    assert pilot["projected_reverse_paths"] == 100
    assert pilot["projected_reverse_reference_transitions"] == 140_492_800
    assert pilot["projected_forward_sampling_paths"] == 30
    assert pilot["projected_forward_sampling_transitions"] == 10_536_960
    assert pilot["projected_sampling_transition_work"] == 151_029_760
    assert pilot["projected_base_transition_work"] == 273_960_960
    assert pilot["shared_k128_k512_audit_transitions"] == 8_780_800
    assert pilot["projected_transition_work_including_shared_audit"] == 282_741_760
    projection = config["resource_projection"]
    assert projection["full_reverse_rows"] == 420
    assert projection["full_reverse_reference_transitions"] == 590_069_760
    assert projection["full_forward_sampling_rows"] == 50
    assert projection["full_forward_sampling_transitions"] == 17_561_600
    assert projection["full_sampling_transition_work"] == 607_631_360
    assert projection["full_base_transition_work"] == 2_363_791_360
    assert projection["k128_k512_audit_transitions"] == 8_780_800
    assert projection["full_transition_work_including_audit"] == 2_372_572_160
    assert config["path_ids"]["preflight_k128_k512"]["start"] == 0xB2100
    assert config["path_ids"]["preflight_k128_k512"]["stop_exclusive"] == 0xB2101
    evaluator = config["evaluator"]
    assert evaluator["accepted_checkpoint_sha256"] == (
        "3d31d42a14fee0ecc72adc1644718a037cc48e649948407da6c272b981840c92"
    )
    assert evaluator["accepted_selection_sha256"] == (
        "e6cd9e49ca61237d3a10e9ad2fe0ad09f7a33ea22911fdd73fd99f3a07e4c668"
    )
    assert evaluator["accepted_contextual_metrics_sha256"] == (
        "2e2fc75b6398f25a84bdaef0558c2f99c51117c71a009a3a94ed0afe8d27be33"
    )
    assert evaluator["accepted_run_manifest_sha256"] == (
        "79aa5d9ae1ca6615a46c9d699f947bea4b6a380cc32e86547cc7e49cee612953"
    )
    assert evaluator["accepted_run_status"] == "complete"


def test_production_resource_checks_preserve_the_frozen_terminal_reserve(
    tmp_path: Path,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    assert runner.FROZEN_CONFIG["resource_defaults"][
        "terminal_reserve_seconds"
    ] == 900.0
    runner._write_json(tmp_path / "config.json", runner.FROZEN_CONFIG)  # noqa: SLF001
    runner._write_json(tmp_path / "environment.json", {"device": "cpu"})  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        tmp_path / "resource_ledger.json",
        {
            "active_seconds": 50.0,
            "events": [],
            "maximum_active_seconds": 1_000.0,
            "maximum_storage_bytes": 1_000_000_000,
            "maximum_cuda_fraction": 1.0,
            "peak_storage_bytes": 0,
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_fraction": 0.0,
        },
    )
    with pytest.raises(runner.ResourceStop, match="active-time"):
        runner._resource_check(tmp_path, projected_seconds=50.0)  # noqa: SLF001
    runner._resource_check(  # noqa: SLF001
        tmp_path,
        projected_seconds=50.0,
        preserve_terminal_reserve=False,
    )
    assert runner._run_stage(  # noqa: SLF001
        tmp_path,
        "terminal_open_and_load",
        lambda: "charged",
        preserve_terminal_reserve=False,
    ) == "charged"
    ledger = runner._read_json(tmp_path / "resource_ledger.json")  # noqa: SLF001
    assert ledger["events"][-1]["role"] == "terminal_open_and_load"
    assert ledger["events"][-1]["failed"] == 0
    assert ledger["active_seconds"] == pytest.approx(
        math.fsum(float(row["seconds"]) for row in ledger["events"])
        + 50.0
    )


def test_terminal_finalize_and_record_review_charge_explicit_reserve_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    run_dir = tmp_path / "terminal-accounting"
    (run_dir / "review").mkdir(parents=True)
    runner._write_json(run_dir / "config.json", runner.FROZEN_CONFIG)  # noqa: SLF001
    runner._write_json(run_dir / "environment.json", {"device": "cpu"})  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        run_dir / "resource_ledger.json",
        {
            "active_seconds": 0.0,
            "events": [],
            "maximum_active_seconds": 100.0,
            "maximum_storage_bytes": 1_000_000_000,
            "maximum_cuda_fraction": 1.0,
            "peak_storage_bytes": 0,
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_fraction": 0.0,
        },
    )
    runner._write_json(  # noqa: SLF001
        run_dir / "TERMINAL_EVIDENCE_OPENED.json", {"opened_after_population_seal": 1}
    )
    runner._write_json(run_dir / "review/READY.json", {"ready": 1})  # noqa: SLF001
    monkeypatch.setattr(runner, "_population_seal", lambda *_: {"sealed": 1})
    monkeypatch.setattr(runner, "_production_report", lambda *_args, **_kwargs: "# report\n")
    monkeypatch.setattr(runner, "verify_run", lambda *_: {"passed": 1})
    assert runner.finalize_and_verify(run_dir)["passed"] == 1
    ledger = runner._read_json(run_dir / "resource_ledger.json")  # noqa: SLF001
    assert ledger["events"][-1]["role"] == "terminal_finalize_awaiting_review"

    answers = run_dir / "answers.csv"
    runner._write_text(answers, "sample_id,answer\nfixture,noise\n")  # noqa: SLF001
    monkeypatch.setattr(runner, "_verify_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_validate_population_semantics", lambda *_: {"passed": 1})
    monkeypatch.setattr(runner, "_validate_review_bundle", lambda *_: {"passed": 1})
    monkeypatch.setattr(runner, "_validate_full_decision_semantics", lambda *_: {"passed": 1})
    monkeypatch.setattr(runner, "score_human_review", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "_review_row_metrics",
        lambda *_: {"learned-prior": {}, "null-prior": {}},
    )
    monkeypatch.setattr(runner, "_outcome", lambda *_: {"decision": "fixture"})
    monkeypatch.setattr(runner, "_production_handoff", lambda *_args, **_kwargs: "# handoff\n")
    outcome = runner.record_human_review(
        run_dir,
        answers,
        reviewer="unit-test-reviewer",
        confirm_manual_review=True,
    )
    assert outcome == {"decision": "fixture"}
    ledger = runner._read_json(run_dir / "resource_ledger.json")  # noqa: SLF001
    assert [row["role"] for row in ledger["events"]][-2:] == [
        "terminal_finalize_awaiting_review",
        "terminal_record_human_review",
    ]
    source = inspect.getsource(runner.execute_full_experiment)
    for role in (
        "terminal_open_and_load",
        "terminal_evaluator_real_health",
        "terminal_prior_generation_metrics",
        "terminal_forward_classifier_metrics",
        "terminal_review_bundle",
    ):
        assert f'"{role}"' in source
    assert source.count("preserve_terminal_reserve=False") >= 5


def test_runner_exposes_only_the_prescribed_workflow_subcommands() -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    parser = runner.build_parser()
    commands = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    assert set(commands.choices) == {"smoke", "run", "record-review", "verify"}
    run_parser = commands.choices["run"]
    run_options = {
        option
        for action in run_parser._actions
        for option in action.option_strings
    }
    assert "--pilot-dir" not in run_options
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--run-dir",
                "run",
                "--arff",
                "mnist.arff",
                "--ddpm-run-dir",
                "ddpm",
                "--device",
                "cuda",
                "--approval-id",
                "approved",
                "--max-active-seconds",
                "1",
                "--max-storage-mib",
                "1",
                "--max-cuda-fraction",
                "1",
                "--pilot-dir",
                "external-pilot",
            ]
        )


def test_csv_writer_preserves_the_union_of_heterogeneous_training_keys(
    tmp_path: Path,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    rows = [
        {"update": 0, "validation_normalized_mse": 1.0, "eligible": 0},
        {
            "update": 250,
            "validation_normalized_mse": 0.75,
            "training_batch_raw_mse": 0.5,
            "training_batch_normalized_mse": 0.8,
            "eligible": 1,
        },
    ]
    path = tmp_path / "training_history.csv"
    runner._write_csv(path, rows)  # noqa: SLF001
    with path.open("r", encoding="utf-8", newline="") as handle:
        replay = list(csv.DictReader(handle))
    assert replay[0]["training_batch_raw_mse"] == ""
    assert replay[1]["training_batch_raw_mse"] == "0.5"
    assert replay[1]["training_batch_normalized_mse"] == "0.8"
    assert set(replay[0]) == {
        "update",
        "validation_normalized_mse",
        "training_batch_raw_mse",
        "training_batch_normalized_mse",
        "eligible",
    }


def test_initialize_rejects_self_consistent_unaccepted_evaluator_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    arff = tmp_path / "authenticated.arff"
    runner._write_text(arff, "synthetic authenticated fixture\n")  # noqa: SLF001
    ddpm_run = tmp_path / "coherent-unaccepted-ddpm"
    (ddpm_run / "evaluator").mkdir(parents=True)
    (ddpm_run / "evaluation").mkdir()
    checkpoint = ddpm_run / "evaluator/selected_checkpoint.pt"
    selection = ddpm_run / "evaluator/selection.json"
    metrics = ddpm_run / "evaluation/metrics.json"
    status = ddpm_run / "status.json"
    manifest = ddpm_run / "artifact_manifest.json"
    runner._write_torch(checkpoint, {"coherent_fixture": 1})  # noqa: SLF001
    checkpoint_hash = runner._file_sha256(checkpoint)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        selection,
        {"selected_checkpoint_sha256": checkpoint_hash, "state": "selected"},
    )
    selection_hash = runner._file_sha256(selection)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        metrics,
        {
            "evaluator_checkpoint_sha256": checkpoint_hash,
            "evaluator_selection_sha256": selection_hash,
        },
    )
    metrics_hash = runner._file_sha256(metrics)  # noqa: SLF001
    runner._write_json(status, {"state": "complete"})  # noqa: SLF001
    rows = [
        {
            "path": path.relative_to(ddpm_run).as_posix(),
            "sha256": runner._file_sha256(path),  # noqa: SLF001
            "size": path.stat().st_size,
        }
        for path in (checkpoint, selection, metrics, status)
    ]
    runner._write_json(  # noqa: SLF001
        manifest,
        {
            "artifact_count": len(rows),
            "artifact_bytes": sum(int(row["size"]) for row in rows),
            "artifacts": rows,
        },
    )
    manifest_hash = runner._file_sha256(manifest)  # noqa: SLF001
    assert checkpoint_hash != runner.ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256

    monkeypatch.setattr(runner, "MNIST_ARFF_SHA256", runner._file_sha256(arff))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runner, "ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256", selection_hash
    )
    monkeypatch.setattr(runner, "ACCEPTED_DDPM_METRICS_SHA256", metrics_hash)
    monkeypatch.setattr(runner, "ACCEPTED_DDPM_MANIFEST_SHA256", manifest_hash)
    run_dir = tmp_path / "must-not-initialize"
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="frozen user-accepted evaluator authority",
    ):
        runner.initialize_run(
            Path(runner.__file__).resolve().parents[1],
            arff,
            ddpm_run,
            run_dir,
            device="cuda",
            maximum_active_seconds=1.0,
            maximum_storage_mib=1.0,
            maximum_cuda_fraction=1.0,
            approval_id="unit-test-approval",
        )
    assert not run_dir.exists()


def test_balanced_path_selection_uses_exact_whole_image_counts() -> None:
    labels = np.tile(np.arange(10, dtype=np.int64), 7_000)

    train = ddpm.balanced_class_indices(labels, per_class=400, start=0, stop=55_000)
    validation = ddpm.balanced_class_indices(
        labels, per_class=100, start=55_000, stop=60_000
    )

    assert train.dtype == np.int64 and validation.dtype == np.int64
    assert train.shape == (4_000,) and validation.shape == (1_000,)
    np.testing.assert_array_equal(
        np.bincount(labels[train], minlength=10), np.full(10, 400)
    )
    np.testing.assert_array_equal(
        np.bincount(labels[validation], minlength=10), np.full(10, 100)
    )
    assert np.all((0 <= train) & (train < 55_000))
    assert np.all((55_000 <= validation) & (validation < 60_000))
    assert not set(train.tolist()).intersection(validation.tolist())


def test_fresh_path_id_roles_are_disjoint_and_collision_free() -> None:
    assert ddpm.PREFLIGHT_PATH_ID_START == 0xB2000
    assert ddpm.TRAIN_PATH_IDS == tuple(range(0xB3000, 0xB3000 + 4_000))
    assert ddpm.VALIDATION_PATH_IDS == tuple(range(0xB4000, 0xB4000 + 1_000))
    assert ddpm.EVALUATION_PATH_ID_START == 0xB5000

    roles = {
        "train": set(ddpm.TRAIN_PATH_IDS),
        "validation": set(ddpm.VALIDATION_PATH_IDS),
    }
    assert roles["train"].isdisjoint(roles["validation"])
    assert not any(0xB0000 <= value < 0xB2000 for values in roles.values() for value in values)
    inventory = ddpm.path_id_inventory_contract()
    assert inventory["collision_count"] == 0
    assert inventory["repository_scan_passed"] == 1
    assert inventory["fresh_roles_disjoint"] == 1


def test_repository_path_id_scan_honors_half_open_legacy_ranges(
    tmp_path: Path,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    repository_root = Path(__file__).resolve().parents[1]
    audit = runner._scan_path_id_collisions(repository_root)  # noqa: SLF001
    assert audit["passed"] == 1
    assert audit["collision_count"] == 0
    assert {
        "path": "mnist/d0_jacobi_rb_haar.py",
        "start": 0xB1000,
        "stop_exclusive": 0xB2000,
    } in audit["semantic_half_open_ranges"]

    synthetic_root = tmp_path / "synthetic-repository"
    synthetic_mnist = synthetic_root / "mnist"
    synthetic_mnist.mkdir(parents=True)
    (synthetic_mnist / "d0_jacobi_rb_haar.py").write_text(
        "HAAR_ROLE_SLOTS = {'legacy': (0xB1000, 0xB2000)}\n"
        "HAAR_PRODUCTION_RESERVED = (0xF0000, 0x100000)\n",
        encoding="utf-8",
    )
    (synthetic_mnist / "legacy_collision.py").write_text(
        "CLAIMED_PATH_ID = 0xB2001\n",
        encoding="utf-8",
    )
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="fresh path-ID range collides with legacy source",
    ):
        runner._scan_path_id_collisions(synthetic_root)  # noqa: SLF001


def test_k128_schedule_preserves_nominal_exposure_but_is_not_k512() -> None:
    schedule = ddpm.k128_schedule_contract()

    assert schedule["outer_steps"] == 128
    assert schedule["reference_outer_steps"] == 512
    assert schedule["phase_matchings"] == list(ddpm.PHASE_MATCHINGS)
    assert schedule["phase_durations"] == list(ddpm.PHASE_DURATIONS)
    assert schedule["cumulative_schedule_integral"] == pytest.approx(
        schedule["reference_cumulative_schedule_integral"], rel=0, abs=1e-20
    )
    assert schedule["macrostep_schedule_integral"] == pytest.approx(
        4.0 * schedule["reference_macrostep_schedule_integral"],
        rel=0,
        abs=1e-20,
    )
    assert schedule["scientifically_identical_to_k512"] == 0

    pair_totals = np.asarray([0.001, 0.01, 0.1, 0.9], dtype=np.float64)
    audit = ddpm.paired_schedule_exposure_audit(pair_totals)
    assert audit["passed"] == 1
    assert audit["same_nominal_cumulative_exposure"] == 1
    assert audit["same_finite_split_chain_law"] == 0
    assert audit["k128_per_phase_exposure_ratio"] == pytest.approx(4.0)


def test_model_is_the_fixed_global_predictor_and_applies_mobility_once() -> None:
    model = ddpm.make_model().to(dtype=torch.float64)
    assert type(model) is ddpm.EulerianJacobiDDPMModel
    assert type(model.predictor) is GlobalDilatedZeroBaselinePredictor
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_974

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.predictor.residual_score.local_affine.bias.fill_(2.0)

    inputs = _model_inputs()
    q = model.score_prediction(inputs)
    m = model(inputs)
    torch.testing.assert_close(q, torch.full_like(q, 2.0), rtol=0, atol=0)
    torch.testing.assert_close(m, torch.full_like(m, 0.5), rtol=0, atol=0)

    target = torch.zeros_like(m)
    normalized, raw = ddpm.direct_m_loss(
        m, target, training_target_energy=torch.tensor(4.0, dtype=torch.float64)
    )
    assert float(raw.detach()) == pytest.approx(0.25)
    assert float(normalized.detach()) == pytest.approx(0.25 / 4.0)

    tails, _ = ddpm.matching_indices()
    facet_state = torch.zeros((1, ddpm.STATE_SIZE), dtype=torch.float64)
    facet_state[:, tails[0]] = 1.0 / ddpm.EDGES_PER_PHASE
    facet_inputs = _model_inputs(facet_state)
    facet_m = model(facet_inputs)
    torch.testing.assert_close(facet_m, torch.zeros_like(facet_m), rtol=0, atol=0)
    facet_loss, facet_raw = ddpm.direct_m_loss(
        facet_m,
        torch.ones_like(facet_m),
        training_target_energy=torch.tensor(1.0, dtype=torch.float64),
    )
    assert torch.isfinite(facet_loss) and torch.isfinite(facet_raw)
    assert float(facet_loss.detach()) == pytest.approx(1.0)
    assert float(facet_raw.detach()) == pytest.approx(1.0)


def test_model_input_firewall_is_exact_and_rejects_every_forbidden_payload() -> None:
    expected = {
        "later_full_state",
        "reverse_time",
        "phase",
        "color",
        "duration",
        "label",
    }
    assert {field.name for field in fields(ddpm.ModelInputs)} == expected
    assert not expected.intersection(ddpm.FORBIDDEN_MODEL_INPUT_FIELDS)
    for forbidden in (
        "earlier_state",
        "target_image",
        "source_image",
        "forward_uniforms",
        "uniform_bits",
        "path_id",
        "denoising_target",
    ):
        assert forbidden in ddpm.FORBIDDEN_MODEL_INPUT_FIELDS

    model = ddpm.make_model()
    inputs = _model_inputs(state=torch.full((1, ddpm.STATE_SIZE), 1 / 784))
    payload = {name: getattr(inputs, name) for name in expected}
    payload["source_image"] = torch.zeros_like(inputs.later_full_state)
    with pytest.raises((TypeError, ValueError), match="forbidden|ModelInputs"):
        model(payload)  # type: ignore[arg-type]


def test_rb_orientation_and_direct_target_match_the_d0_certified_fixture() -> None:
    x = np.asarray([0.2, 0.5, 0.8], dtype=np.float64)
    y = np.asarray([0.25, 0.55, 0.75], dtype=np.float64)
    exposure = np.asarray([0.75, 1.0, 0.75], dtype=np.float64)

    measured = ddpm.fast_rb_target(x, y, exposure)
    reference = ddpm.certified_rb_target_fixture(x, y, exposure)

    assert measured["orientation"] == "head-fraction"
    np.testing.assert_allclose(
        measured["denoising_target"],
        reference["denoising_target"],
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        measured["denoising_target"],
        y * (1.0 - y) * reference["arrival_score"],
        rtol=0,
        atol=2e-14,
    )


def test_controller_reuses_d0_semantics_and_preserves_facets_pairs_and_null() -> None:
    assert ddpm.frozen_score_logistic_fraction is d0_tangent.frozen_score_logistic_fraction
    assert ddpm.frozen_score_logistic_flow is d0_tangent.frozen_score_logistic_flow

    y = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=torch.float64)
    positive_q = torch.full_like(y, 1.7)
    moved = ddpm.frozen_score_logistic_fraction(y, positive_q, 0.1)
    assert moved[0] == 0.0 and moved[-1] == 1.0
    assert bool(torch.all(moved[1:-1] > y[1:-1]))
    torch.testing.assert_close(
        torch.logit(moved[1:-1]) - torch.logit(y[1:-1]),
        torch.full_like(y[1:-1], 2.0 * 1.7 * 0.1),
        rtol=2e-15,
        atol=2e-16,
    )
    torch.testing.assert_close(
        ddpm.frozen_score_logistic_fraction(y, torch.zeros_like(y), 0.1),
        y,
        rtol=0,
        atol=0,
    )

    generator = torch.Generator(device="cpu").manual_seed(73)
    state = torch.rand((3, ddpm.STATE_SIZE), generator=generator, dtype=torch.float64)
    state /= state.sum(dim=1, keepdim=True)
    q = torch.linspace(-3, 3, ddpm.EDGES_PER_PHASE, dtype=torch.float64).repeat(3, 1)
    tails, heads = ddpm.matching_indices()
    changed = ddpm.frozen_score_logistic_flow(state, (tails[0], heads[0]), q, 0.07)
    null = ddpm.frozen_score_logistic_flow(
        state, (tails[0], heads[0]), torch.zeros_like(q), 0.07
    )
    torch.testing.assert_close(null, state, rtol=0, atol=1e-18)
    torch.testing.assert_close(changed.sum(1), state.sum(1), rtol=0, atol=3e-16)
    torch.testing.assert_close(
        changed[:, tails[0]] + changed[:, heads[0]],
        state[:, tails[0]] + state[:, heads[0]],
        rtol=0,
        atol=5e-19,
    )
    assert bool(torch.all((changed >= 0.0) & (changed <= 1.0)))


def test_reverse_midpoint_times_match_d0_and_reach_the_model_in_execution_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_step = 511
    phase = 6
    expected = [
        d0_reverse.internal_reverse_time(outer_step, phase, q)
        for q in (0.75, 0.25)
    ]
    measured = [
        ddpm.reverse_midpoint_time(
            outer_step,
            phase,
            reverse_index,
            sample_steps=512,
        )
        for reverse_index in range(2)
    ]
    assert measured == expected
    assert measured[0] < measured[1]

    monkeypatch.setattr(
        ddpm,
        "propose_alpha1_rb_transition_batch_torch",
        lambda fraction, *_args, **_kwargs: SimpleNamespace(
            proposed_later_head_fraction=fraction.clone()
        ),
    )
    observed: list[float] = []

    class StopAfterTwoModelCalls(RuntimeError):
        pass

    class RecordingModel(torch.nn.Module):
        def score_prediction(self, inputs: ddpm.ModelInputs) -> torch.Tensor:
            observed.append(float(inputs.reverse_time[0]))
            if len(observed) == 2:
                raise StopAfterTwoModelCalls
            return torch.zeros(
                (inputs.later_full_state.shape[0], ddpm.EDGES_PER_PHASE),
                dtype=inputs.later_full_state.dtype,
                device=inputs.later_full_state.device,
            )

    with pytest.raises(StopAfterTwoModelCalls):
        ddpm.reverse_sample(
            np.full((1, ddpm.STATE_SIZE), 1.0 / ddpm.STATE_SIZE),
            np.asarray([3], dtype=np.int64),
            (0xB2208,),
            controller="learned",
            root_seed=26_140_009,
            model=RecordingModel(),  # type: ignore[arg-type]
            device="cpu",
            anchors=(0, 512),
            sample_steps=512,
        )
    assert observed == [float(np.float32(value)) for value in measured]


def test_dirichlet_start_bank_is_path_seeded_rebatchable_and_label_independent() -> None:
    path_ids = np.asarray([91, 17, 203, 44], dtype=np.int64)
    first = ddpm.sample_dirichlet_starts(path_ids, root_seed=26_140_001)
    replay = ddpm.sample_dirichlet_starts(path_ids, root_seed=26_140_001)
    reordered = ddpm.sample_dirichlet_starts(path_ids[::-1], root_seed=26_140_001)

    assert first.dtype == np.float64
    assert first.shape == (4, ddpm.STATE_SIZE)
    np.testing.assert_array_equal(first, replay)
    np.testing.assert_array_equal(first, reordered[::-1])
    np.testing.assert_allclose(first.sum(axis=1), 1.0, rtol=0, atol=3e-16)
    assert np.all(first > 0.0)
    assert "label" not in inspect.signature(ddpm.sample_dirichlet_starts).parameters


def test_fixed_demix_and_mass_rasterization_round_trip_exact_uint8() -> None:
    pixels = np.zeros(ddpm.STATE_SIZE, dtype=np.uint8)
    pixels[:99] = 255
    pixels[99] = 226
    assert int(pixels.sum()) == 25_471
    unit_mass = pixels.astype(np.float64) / 25_471.0

    mixed = ddpm.mix_unit_masses(unit_mass)
    demixed = ddpm.demix_unit_masses(mixed)
    rendered = ddpm.rasterize_unit_masses(demixed)

    np.testing.assert_allclose(demixed, unit_mass, rtol=0, atol=2e-18)
    np.testing.assert_array_equal(rendered.reshape(-1), pixels)
    assert rendered.dtype == np.uint8 and rendered.shape == (28, 28)


def test_small_fast_kernel_audit_is_oriented_finite_and_conservative() -> None:
    result = ddpm.fast_vs_certified_audit(transition_count=8, seed=26_140_002)
    assert result["transition_count"] == 8
    assert result["orientation_identical"] == 1
    assert result["transition_ids_identical"] == 1
    assert result["nonfinite_count"] == 0
    assert result["maximum_state_error"] <= 2e-10
    assert result["maximum_target_error"] <= 2e-8
    assert result["maximum_pair_total_error"] <= 2e-12
    assert result["passed"] == 1
    assert result["device"] == "cpu"


def test_transition_id_rng_is_order_and_rebatching_invariant() -> None:
    rng = np.random.Generator(np.random.PCG64(26_140_006))
    states = rng.dirichlet(np.ones(ddpm.STATE_SIZE), size=2)
    path_ids = (0xB3000, 0xB3001)
    profile = JacobiRBSpectralProfile(
        device_proposal_modes=4,
        device_bisection_steps=4,
    )

    def one_phase(
        rows: np.ndarray, ids: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        later, target = ddpm._fast_forward_phase(  # noqa: SLF001
            torch.as_tensor(rows.copy(), dtype=torch.float64),
            ids,
            outer_step=7,
            phase=3,
            root_seed=26_140_006,
            sample_steps=ddpm.OUTER_STEPS,
            profile=profile,
        )
        return later.numpy(), target.numpy()

    batched_state, batched_target = one_phase(states, path_ids)
    reversed_state, reversed_target = one_phase(states[::-1], path_ids[::-1])
    single_rows = [one_phase(states[index : index + 1], (path_ids[index],)) for index in range(2)]
    single_state = np.concatenate([row[0] for row in single_rows], axis=0)
    single_target = np.concatenate([row[1] for row in single_rows], axis=0)

    np.testing.assert_array_equal(batched_state, reversed_state[::-1])
    np.testing.assert_array_equal(batched_target, reversed_target[::-1])
    np.testing.assert_array_equal(batched_state, single_state)
    np.testing.assert_array_equal(batched_target, single_target)


def test_spectral_proposer_without_explicit_ids_preserves_legacy_flat_index_rng() -> None:
    key = (26_140_007, "legacy-flat-index-replay")
    x = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    exposure = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)
    profile = JacobiRBSpectralProfile(
        device_proposal_modes=4,
        device_bisection_steps=2,
    )

    proposal = propose_alpha1_rb_transition_batch_torch(
        x,
        exposure,
        rng_key=key,
        profile=profile,
    )
    expected = np.asarray(
        [philox_uniform_prefix(key, sample_index=index, bits=64)[2] for index in range(3)]
    )
    np.testing.assert_array_equal(proposal.uniform_midpoint.numpy(), expected)


def test_paired_k128_k512_audit_reports_discrepancy_instead_of_equality() -> None:
    path_ids = (0xB2201, 0xB2207)
    result = ddpm.paired_k128_k512_oracle_audit(
        path_count=2,
        path_ids=path_ids,
        grid_size=4,
        seed=26_140_003,
    )
    assert result["k128_outer_steps"] == 128
    assert result["k512_outer_steps"] == 512
    assert result["shared_nominal_cumulative_exposure"] == 1
    assert result["finite_chain_identity_claimed"] == 0
    assert math.isfinite(result["paired_law_discrepancy"])
    assert math.isfinite(result["paired_oracle_discrepancy"])
    assert result["pair_total_health_passed"] == 1
    assert result["simplex_health_passed"] == 1
    assert result["admission_capable"] == 0
    assert result["paired_initial_states"] == 0
    assert result["aligned_transition_randomness_coupled"] == 0
    assert result["full_path_common_random_numbers_claimed"] == 0
    assert result["backend"] == "schedule-structure-only"
    assert result["path_ids"] == list(path_ids)
    assert result["path_ids_sha256"] == hashlib.sha256(
        np.asarray(path_ids, dtype="<i8").tobytes()
    ).hexdigest()
    assert result["passed"] == 1

    with pytest.raises(ddpm.EulerianJacobiDDPMError, match="dimensions"):
        ddpm.paired_k128_k512_oracle_audit(
            path_count=2,
            path_ids=(path_ids[0],),
            grid_size=4,
        )
    with pytest.raises(ddpm.EulerianJacobiDDPMError, match="dimensions"):
        ddpm.paired_k128_k512_oracle_audit(
            path_count=2,
            path_ids=(path_ids[0], path_ids[0]),
            grid_size=4,
        )


def test_structural_k128_k512_smoke_cannot_authorize_production_gate_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    runner._write_json(tmp_path / "config.json", runner.FROZEN_CONFIG)  # noqa: SLF001
    runner._write_json(tmp_path / "environment.json", {"device": "cpu"})  # noqa: SLF001
    runner._write_json(tmp_path / "path_id_audit.json", {"passed": 1})  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        tmp_path / "resource_ledger.json",
        {
            "active_seconds": 0.0,
            "events": [],
            "maximum_active_seconds": 1e9,
            "maximum_storage_bytes": 1_000_000_000,
            "maximum_cuda_fraction": 1.0,
            "peak_storage_bytes": 0,
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_fraction": 0.0,
        },
    )
    monkeypatch.setattr(
        runner.core,
        "fast_vs_certified_audit",
        lambda **_: {"passed": 1, "device": "cpu"},
    )
    structural = ddpm.paired_k128_k512_oracle_audit(
        path_count=2,
        grid_size=4,
        seed=26_140_008,
    )
    monkeypatch.setattr(
        runner.core,
        "paired_k128_k512_oracle_audit",
        lambda **_: structural,
    )

    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="paired K=128/K=512"):
        runner.run_numerical_preflight(tmp_path)


def test_tiny_synthetic_training_and_sampling_smoke_is_cpu_and_complete() -> None:
    result = ddpm.tiny_synthetic_smoke(seed=26_140_004)

    assert result["device"] == "cpu"
    assert result["grid_size"] == 4
    assert result["class_count"] == 2
    assert result["outer_steps"] == 8
    assert result["optimizer_updates"] >= 1
    assert math.isfinite(result["initial_loss"])
    assert math.isfinite(result["final_loss"])
    assert set(result["sample_rows"]) == {"null", "learned", "oracle"}
    assert result["finite"] == 1
    assert result["nonnegative"] == 1
    assert result["simplex_preserved"] == 1
    assert result["passed"] == 1


def test_training_update_zero_is_the_analytical_q_zero_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_count = 2
    dataset = ddpm.ForwardRecordDataset(
        later_states=np.full(
            (record_count, ddpm.STATE_SIZE),
            1.0 / ddpm.STATE_SIZE,
            dtype=np.float32,
        ),
        reverse_time=np.asarray([0.25, 0.75], dtype=np.float32),
        phase=np.asarray([0, 1], dtype=np.int64),
        color=np.asarray(ddpm.PHASE_MATCHINGS[:2], dtype=np.int64),
        duration=np.asarray(ddpm.PHASE_DURATIONS[:2], dtype=np.float32),
        labels=np.asarray([0, 1], dtype=np.int64),
        targets=np.full(
            (record_count, ddpm.EDGES_PER_PHASE), 0.25, dtype=np.float32
        ),
        path_ids=np.asarray([0xB3000, 0xB3001], dtype=np.int64),
        outer_steps=np.asarray([0, 96], dtype=np.int64),
    )

    class DeliberatelyNonzeroModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, inputs: ddpm.ModelInputs) -> torch.Tensor:
            return self.bias.expand(
                inputs.later_full_state.shape[0], ddpm.EDGES_PER_PHASE
            )

    monkeypatch.setattr(ddpm, "make_model", DeliberatelyNonzeroModel)
    result = ddpm.train_jacobi_ddpm(
        dataset,
        dataset,
        device="cpu",
        updates=1,
        batch_size=2,
        learning_rate=0.0,
        validation_interval=1,
        seed=26_140_010,
    )
    assert result.history[0]["update"] == 0
    assert result.history[0]["eligible"] == 0
    assert result.history[0]["validation_normalized_mse"] == pytest.approx(1.0)
    assert result.history[1]["validation_normalized_mse"] > 10.0


def _write_objective_pilot_fixture(
    pilot_dir: Path,
    *,
    learned_improves: bool,
    prior_task_passed: bool = True,
    production_schema: bool = False,
    write_start_authority: bool = False,
    seal: bool = True,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    pilot_dir.mkdir()
    uniform = np.full((1, ddpm.STATE_SIZE), 1.0 / ddpm.STATE_SIZE, dtype=np.float64)

    def repeated(count: int) -> np.ndarray:
        return np.repeat(uniform, count, axis=0)

    def shifted(count: int, amount: float) -> np.ndarray:
        values = repeated(count)
        values[:, 0] -= amount
        values[:, 1] += amount
        return values

    null_forward = shifted(20, 1e-4)
    learned_forward = shifted(20, 5e-5 if learned_improves else 1.5e-4)
    forward_targets = repeated(20)
    null_oracle = shifted(10, 1e-4)
    oracle = repeated(10)
    oracle_targets = repeated(10)
    prior = repeated(20)
    prior_labels = np.tile(np.arange(10, dtype=np.int64), 2)
    forward_labels = np.tile(np.arange(10, dtype=np.int64), 2)
    oracle_labels = np.arange(10, dtype=np.int64)
    path_config = runner.FROZEN_CONFIG["path_ids"]

    def role_ids(name: str) -> np.ndarray:
        role = path_config[name]
        return np.arange(role["start"], role["stop_exclusive"], dtype=np.int64)

    prior_path_ids = role_ids("pilot_prior")
    forward_path_ids = role_ids("pilot_forward_terminal")
    oracle_path_ids = role_ids("pilot_oracle")

    null_l1 = np.sum(np.abs(null_forward - forward_targets), axis=1, dtype=np.float64)
    learned_l1 = np.sum(
        np.abs(learned_forward - forward_targets), axis=1, dtype=np.float64
    )
    relative = (
        float(np.sum(null_l1, dtype=np.float64))
        - float(np.sum(learned_l1, dtype=np.float64))
    ) / float(np.sum(null_l1, dtype=np.float64))
    telemetry = [
        {
            "population": population,
            "time_quarter": quarter,
            "controller_rms": 0.1 if population.startswith("learned-") else 0.0,
            "score_count": (
                10 if population == "oracle" else 20
            )
            * 175_616,
            "finite": 1,
            "nonnegative": 1,
            "microsteps": 2,
            "maximum_mass_error": 0.0,
            "maximum_pair_total_error": 0.0,
            "exact_facet_count": 0,
        }
        for population in (
            "null-prior",
            "learned-prior",
            "null-forward-terminal",
            "learned-forward-terminal",
            "oracle",
        )
        for quarter in range(4)
    ]
    learned_rms = [
        float(row["controller_rms"])
        for row in telemetry
        if str(row["population"]).startswith("learned-")
    ]
    metrics = {
        "gates": {"gate_c_passed": 1, "health_passed": 1},
        "forward_terminal": {
            "learned_l1_win_count": int(np.sum(learned_l1 < null_l1)),
            "aggregate_l1_relative_improvement": relative,
        },
        "oracle": {"l1_win_count": 10},
        "controller": {
            "learned_rms": math.sqrt(float(np.mean(np.square(learned_rms))))
        },
    }
    sample_ids = {
        "prior": np.asarray(
            [f"pilot-prior-{value:x}" for value in prior_path_ids], dtype=np.str_
        ),
        "forward": np.asarray(
            [f"pilot-forward-{value:x}" for value in forward_path_ids], dtype=np.str_
        ),
        "oracle": np.asarray(
            [f"pilot-oracle-{value:x}" for value in oracle_path_ids], dtype=np.str_
        ),
    }
    null_logits = np.zeros((20, 10), dtype=np.float64)
    null_logits[np.arange(20), (prior_labels + 1) % 10] = 1.0
    learned_logits = null_logits.copy()
    improved_rows = 12 if prior_task_passed else 4
    learned_logits[np.arange(improved_rows), prior_labels[:improved_rows]] = 2.0

    def classifier_rows(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted_logits)
        probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        log_probabilities = shifted_logits - np.log(
            np.sum(exponentials, axis=1, keepdims=True)
        )
        return (
            np.argmax(logits, axis=1).astype(np.int64),
            probabilities,
            log_probabilities[np.arange(20), prior_labels],
        )

    null_predictions, null_probabilities, null_requested = classifier_rows(null_logits)
    learned_predictions, learned_probabilities, learned_requested = classifier_rows(
        learned_logits
    )
    paired_delta = learned_requested - null_requested
    prior_summary = {
        "gate_type": "diagnostic threshold",
        "terminal_test_rows_used": 0,
        "path_count": 20,
        "null_requested_label_accuracy": float(
            np.mean(null_predictions == prior_labels)
        ),
        "learned_requested_label_accuracy": float(
            np.mean(learned_predictions == prior_labels)
        ),
        "paired_requested_log_probability_win_count": int(
            np.sum(paired_delta > 0.0)
        ),
        "mean_paired_requested_log_probability_improvement": float(
            np.mean(paired_delta, dtype=np.float64)
        ),
        "passed": int(prior_task_passed),
    }
    metrics["prior_task_signal"] = prior_summary
    prior_classifier_outputs = {
        "requested_labels": prior_labels,
        "sample_ids": sample_ids["prior"],
        "null_predictions": null_predictions,
        "null_logits": null_logits,
        "null_probabilities": null_probabilities,
        "null_requested_log_probabilities": null_requested,
        "learned_predictions": learned_predictions,
        "learned_logits": learned_logits,
        "learned_probabilities": learned_probabilities,
        "learned_requested_log_probabilities": learned_requested,
    }
    if write_start_authority:
        runner._write_prior_start_authority(  # noqa: SLF001
            pilot_dir,
            prior,
            prior_labels,
            prior_path_ids,
            sample_ids["prior"],
        )
    population_rows = {
        "null_prior": prior,
        "learned_prior": prior,
        "null_forward_terminal": null_forward,
        "learned_forward_terminal": learned_forward,
        "forward_targets": forward_targets,
        "null_oracle": null_oracle,
        "oracle": oracle,
        "oracle_targets": oracle_targets,
    }
    demixed_rows = {
        name + "_demixed": np.stack(
            [ddpm.demix_unit_masses(row) for row in values]
        )
        for name, values in population_rows.items()
    }
    identity_rows = {
        "prior_requested_labels": prior_labels,
        "prior_path_ids": prior_path_ids,
        "prior_sample_ids": sample_ids["prior"],
        "forward_requested_labels": forward_labels,
        "forward_path_ids": forward_path_ids,
        "forward_sample_ids": sample_ids["forward"],
        "oracle_requested_labels": oracle_labels,
        "oracle_path_ids": oracle_path_ids,
        "oracle_sample_ids": sample_ids["oracle"],
    }
    prior_trajectories = np.repeat(prior[:, None, :], 5, axis=1)
    runner._write_json(  # noqa: SLF001 - exercise the runner's sealed format
        pilot_dir / "config.json",
        (
            dict(runner.FROZEN_CONFIG) | {"run_scope": "objective_pilot"}
            if production_schema
            else {
                "run_scope": "objective_pilot",
                "objective_pilot": runner.FROZEN_CONFIG["objective_pilot"],
            }
        ),
    )
    runner._write_json(pilot_dir / "metrics.json", metrics)  # noqa: SLF001
    runner._write_npz(  # noqa: SLF001
        pilot_dir / "prior_classifier_outputs.npz", **prior_classifier_outputs
    )
    runner._write_json(  # noqa: SLF001
        pilot_dir / "prior_classifier_metrics.json", prior_summary
    )
    runner._write_npz(  # noqa: SLF001
        pilot_dir / "populations.npz",
        **population_rows,
        **demixed_rows,
        **identity_rows,
        prior_completed_steps=np.asarray([0, 32, 64, 96, 128], dtype=np.int64),
        null_prior_trajectories=prior_trajectories,
        learned_prior_trajectories=prior_trajectories,
    )
    start_bank_rows = {
        "prior_starts": prior,
        "prior_requested_labels": prior_labels,
        "prior_path_ids": prior_path_ids,
        "prior_sample_ids": sample_ids["prior"],
    }
    if production_schema:
        start_bank_rows.update(
            forward_terminal_starts=forward_targets,
            oracle_starts=oracle_targets,
        )
    runner._write_npz(  # noqa: SLF001
        pilot_dir / "start_banks.npz", **start_bank_rows
    )
    rendered_rows = {
        name: np.stack(
            [ddpm.rasterize_unit_masses(ddpm.demix_unit_masses(row)) for row in values]
        )
        for name, values in population_rows.items()
    }
    runner._write_npz(  # noqa: SLF001
        pilot_dir / "uint8_populations.npz",
        **rendered_rows,
        **identity_rows,
    )
    runner._write_csv(pilot_dir / "telemetry.csv", telemetry)  # noqa: SLF001
    runner._write_torch(  # noqa: SLF001
        pilot_dir / "selected_checkpoint.pt", {"fixture": "objective-pilot"}
    )
    if production_schema:
        def write_stage(
            name: str,
            starts: np.ndarray,
            finals: np.ndarray,
            labels: np.ndarray,
            path_ids: np.ndarray,
            ids: np.ndarray,
        ) -> None:
            anchors = {
                step: (starts if step < 128 else finals)
                for step in (0, 32, 64, 96, 128)
            }
            runner._write_population_stage(  # noqa: SLF001
                pilot_dir,
                name,
                ddpm.SamplingResult(
                    starts=np.asarray(starts, dtype=np.float64),
                    final_states=np.asarray(finals, dtype=np.float64),
                    anchors=anchors,
                    telemetry={
                        "finite": 1,
                        "nonnegative": 1,
                        "microsteps": 2,
                        "by_time_quarter": [{}, {}, {}, {}],
                    },
                ),
                labels,
                path_ids,
                ids,
            )

        write_stage(
            "null_prior", prior, prior, prior_labels, prior_path_ids, sample_ids["prior"]
        )
        write_stage(
            "learned_prior", prior, prior, prior_labels, prior_path_ids, sample_ids["prior"]
        )
        write_stage(
            "null_forward_terminal",
            forward_targets,
            null_forward,
            forward_labels,
            forward_path_ids,
            sample_ids["forward"],
        )
        write_stage(
            "learned_forward_terminal",
            forward_targets,
            learned_forward,
            forward_labels,
            forward_path_ids,
            sample_ids["forward"],
        )
        write_stage(
            "null_oracle",
            oracle_targets,
            null_oracle,
            oracle_labels,
            oracle_path_ids,
            sample_ids["oracle"],
        )
        write_stage(
            "oracle",
            oracle_targets,
            oracle,
            oracle_labels,
            oracle_path_ids,
            sample_ids["oracle"],
        )
    if seal:
        runner.seal_populations(pilot_dir)
        runner._seal_manifest(pilot_dir)  # noqa: SLF001


def test_pilot_commits_prior_start_authority_before_first_reverse_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    parent = tmp_path / "parent"
    parent.mkdir()
    for name in (
        "config.json",
        "source_bindings.json",
        "environment.json",
        "theory_code_identity.json",
        "kernel_audit.json",
        "k128_k512_audit.json",
        "preflight_projection.json",
        "path_id_audit.json",
    ):
        runner._write_json(parent / name, {"fixture": name})  # noqa: SLF001

    fake_records = ddpm.ForwardRecordDataset(
        later_states=np.zeros((1, ddpm.STATE_SIZE), dtype=np.float32),
        reverse_time=np.zeros(1, dtype=np.float32),
        phase=np.zeros(1, dtype=np.int64),
        color=np.zeros(1, dtype=np.int64),
        duration=np.zeros(1, dtype=np.float32),
        labels=np.zeros(1, dtype=np.int64),
        targets=np.zeros((1, ddpm.EDGES_PER_PHASE), dtype=np.float32),
        path_ids=np.asarray([0xB3000], dtype=np.int64),
        outer_steps=np.zeros(1, dtype=np.int64),
    )
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in ddpm.make_model().state_dict().items()
    }
    training = ddpm.TrainingResult(
        model_state_dict=model_state,
        ema_state_dict=model_state,
        selected_state_dict=model_state,
        selected_update=250,
        selected_validation_mse=1.0,
        training_target_energy=1.0,
        history=(
            {
                "update": 250,
                "validation_normalized_mse": 1.0,
                "eligible": 1,
            },
        ),
        completed_updates=750,
    )

    def fake_stage(
        _run_dir: Path,
        role: str,
        function: object,
        *,
        commit: object | None = None,
    ) -> object:
        if role.startswith("pilot_forward_cache_"):
            return fake_records
        if role == "pilot_training_750_updates":
            return training
        assert callable(function)
        result = function()
        if commit is not None:
            assert callable(commit)
            commit(result)
        return result

    monkeypatch.setattr(runner, "_run_stage", fake_stage)
    pilot = parent / "objective_pilot"
    reverse_calls: list[str] = []

    class StopAtFirstReverse(RuntimeError):
        pass

    def inspect_first_reverse(
        starts: np.ndarray,
        labels: np.ndarray,
        path_ids: np.ndarray,
        **kwargs: object,
    ) -> object:
        reverse_calls.append(str(kwargs["controller"]))
        if len(reverse_calls) == 1:
            authority_path = pilot / "prior_start_authority.npz"
            authority_record = runner._read_json(  # noqa: SLF001
                pilot / "prior_start_authority.json"
            )
            assert reverse_calls == ["null"]
            assert authority_record["committed_before_sampling"] == 1
            assert authority_record["npz_sha256"] == runner._file_sha256(  # noqa: SLF001
                authority_path
            )
            assert not (pilot / "start_banks.npz").exists()
            with np.load(authority_path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["prior_starts"], starts)
                np.testing.assert_array_equal(
                    archive["prior_requested_labels"], labels
                )
                np.testing.assert_array_equal(archive["prior_path_ids"], path_ids)
                assert archive["prior_sample_ids"].shape == (20,)
            anchors = {
                step: np.asarray(starts, dtype=np.float64).copy()
                for step in (0, 32, 64, 96, 128)
            }
            return ddpm.SamplingResult(
                starts=np.asarray(starts, dtype=np.float64).copy(),
                final_states=np.asarray(starts, dtype=np.float64).copy(),
                anchors=anchors,
                telemetry={"controller": "null", "finite": 1},
            )
        assert reverse_calls == ["null", "learned"]
        completed = pilot / "population_stages/null_prior.npz"
        assert completed.is_file()
        with np.load(completed, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["final_states"], starts)
            np.testing.assert_array_equal(archive["path_ids"], path_ids)
            assert archive["anchor_steps"].tolist() == [0, 32, 64, 96, 128]
        raise StopAtFirstReverse

    monkeypatch.setattr(runner, "_reverse_cohorts", inspect_first_reverse)
    train_y = np.repeat(np.arange(10, dtype=np.int64), 25)
    validation_y = np.repeat(np.arange(10, dtype=np.int64), 10)
    train_u8 = np.ones((250, 28, 28), dtype=np.uint8)
    validation_u8 = np.ones((100, 28, 28), dtype=np.uint8)
    with pytest.raises(StopAtFirstReverse):
        runner.execute_objective_pilot(
            parent,
            pilot,
            train_u8,
            train_y,
            validation_u8,
            validation_y,
            device="cpu",
        )
    assert reverse_calls == ["null", "learned"]
    completed = pilot / "population_stages/null_prior.npz"
    completed_hash = runner._file_sha256(completed)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        parent / "failure.json",
        {"error_type": "StopAtFirstReverse", "message": "injected failure"},
    )
    runner._status(parent, "failed", error="injected failure")  # noqa: SLF001
    manifest = runner._seal_manifest(parent)  # noqa: SLF001
    assert any(
        row["path"] == "objective_pilot/population_stages/null_prior.npz"
        and row["sha256"] == completed_hash
        for row in manifest["artifacts"]
    )
    assert runner._verify_manifest(parent)["artifact_count"] == manifest[  # noqa: SLF001
        "artifact_count"
    ]


def test_full_scale_requires_a_raw_evidence_bound_positive_objective_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="pilot"):
        runner.require_pilot_for_full_run(tmp_path / "missing")

    negative = tmp_path / "negative-pilot"
    _write_objective_pilot_fixture(negative, learned_improves=False)
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="authority|device|frozen scientific",
    ):
        runner.validate_objective_pilot(negative)
    monkeypatch.setattr(
        runner,
        "_validate_real_pilot_authority",
        lambda *_: {"passed": 1},
    )
    monkeypatch.setattr(
        runner,
        "_validate_pilot_prior_classifier",
        lambda pilot: runner._read_json(  # noqa: SLF001
            Path(pilot) / "prior_classifier_metrics.json"
        ),
    )
    before = runner._directory_digest(negative)  # noqa: SLF001
    decision = runner.validate_objective_pilot(negative)
    assert decision["full_scale_admitted"] == 0
    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="negative"):
        runner.require_pilot_for_full_run(negative)
    assert runner._directory_digest(negative) == before  # noqa: SLF001

    forged_metrics = {
        "gates": {"gate_c_passed": 1, "health_passed": 1},
        "forward_terminal": {
            "learned_l1_win_count": 20,
            "aggregate_l1_relative_improvement": 0.5,
        },
        "oracle": {"l1_win_count": 10},
        "controller": {"learned_rms": 0.1},
    }
    runner._write_json(negative / "metrics.json", forged_metrics)  # noqa: SLF001
    runner._seal_manifest(negative)  # noqa: SLF001
    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="raw evidence"):
        runner.validate_objective_pilot(negative)

    prior_negative = tmp_path / "prior-negative-pilot"
    _write_objective_pilot_fixture(
        prior_negative,
        learned_improves=True,
        prior_task_passed=False,
    )
    prior_decision = runner.validate_objective_pilot(prior_negative)
    assert prior_decision["full_scale_admitted"] == 0
    assert prior_decision["route"] == "pilot_prior_negative_stop_before_scale"

    positive = tmp_path / "positive-pilot"
    _write_objective_pilot_fixture(positive, learned_improves=True)
    admitted = runner.require_pilot_for_full_run(positive)
    assert admitted["integrity_passed"] == 1
    assert admitted["gate_c_passed"] == 1
    assert admitted["full_scale_admitted"] == 1


def test_pilot_prior_task_signal_recomputes_raw_logits_and_rejects_resealed_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    labels = np.tile(np.arange(10, dtype=np.int64), 2)
    sample_ids = np.asarray([f"prior-{index}" for index in range(20)], dtype=np.str_)
    null_images = np.zeros((20, 28, 28), dtype=np.uint8)
    learned_images = np.ones((20, 28, 28), dtype=np.uint8)
    null_logits = np.zeros((20, 10), dtype=np.float64)
    null_logits[np.arange(20), (labels + 1) % 10] = 2.0
    learned_logits = null_logits.copy()
    learned_logits[np.arange(4), labels[:4]] = 3.0
    learned_logits[np.arange(4, 12), labels[4:12]] = 1.0

    def fake_evaluation(
        _evaluator: object,
        images: np.ndarray,
        _labels: np.ndarray,
        _sample_ids: np.ndarray,
        **_kwargs: object,
    ) -> dict[str, np.ndarray]:
        logits = null_logits if int(images[0, 0, 0]) == 0 else learned_logits
        return {
            "logits": logits.copy(),
            "predictions": np.argmax(logits, axis=1).astype(np.int64),
        }

    monkeypatch.setattr(runner, "evaluate_generated_labels", fake_evaluation)
    terminal_loads: list[Path] = []

    def forbidden_terminal_load(path: Path) -> object:
        terminal_loads.append(Path(path))
        raise AssertionError("pilot opened terminal-test data")

    monkeypatch.setattr(runner, "load_test_mnist_terminal", forbidden_terminal_load)
    arrays, summary = runner._pilot_prior_classifier_evidence(  # noqa: SLF001
        object(), null_images, learned_images, labels, sample_ids
    )
    assert set(arrays) == {
        "requested_labels",
        "sample_ids",
        "null_predictions",
        "null_logits",
        "null_probabilities",
        "null_requested_log_probabilities",
        "learned_predictions",
        "learned_logits",
        "learned_probabilities",
        "learned_requested_log_probabilities",
    }
    assert summary["terminal_test_rows_used"] == 0
    assert summary["learned_requested_label_accuracy"] == pytest.approx(0.20)
    assert summary["learned_requested_label_accuracy"] > summary[
        "null_requested_label_accuracy"
    ]
    assert summary["paired_requested_log_probability_win_count"] == 12
    assert summary["mean_paired_requested_log_probability_improvement"] > 0.0
    assert summary["passed"] == 1
    np.testing.assert_allclose(
        arrays["learned_probabilities"].sum(axis=1), 1.0, rtol=0, atol=5e-16
    )
    assert terminal_loads == []
    assert "load_test_mnist_terminal" not in inspect.getsource(
        runner.execute_objective_pilot
    )

    pilot = tmp_path / "prior-classifier-pilot"
    (pilot / "authority").mkdir(parents=True)
    runner._write_npz(  # noqa: SLF001
        pilot / "uint8_populations.npz",
        null_prior=null_images,
        learned_prior=learned_images,
        prior_requested_labels=labels,
        prior_sample_ids=sample_ids,
    )
    runner._write_json(  # noqa: SLF001
        pilot / "authority/source_bindings.json", {"fixture": 1}
    )
    runner._write_npz(pilot / "prior_classifier_outputs.npz", **arrays)  # noqa: SLF001
    runner._write_json(pilot / "prior_classifier_metrics.json", summary)  # noqa: SLF001
    monkeypatch.setattr(runner, "_load_accepted_evaluator", lambda *_args, **_kwargs: object())
    assert runner._validate_pilot_prior_classifier(pilot) == summary  # noqa: SLF001

    forged = {name: np.asarray(value).copy() for name, value in arrays.items()}
    forged_logits = forged["learned_logits"]
    forged_logits[0, labels[0]] = 4.0
    shifted = forged_logits - np.max(forged_logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    forged["learned_probabilities"] = exponentials / np.sum(
        exponentials, axis=1, keepdims=True
    )
    forged_log_probabilities = shifted - np.log(
        np.sum(exponentials, axis=1, keepdims=True)
    )
    forged["learned_requested_log_probabilities"] = forged_log_probabilities[
        np.arange(20), labels
    ]
    forged["learned_predictions"] = np.argmax(forged_logits, axis=1).astype(np.int64)
    forged_delta = (
        forged["learned_requested_log_probabilities"]
        - forged["null_requested_log_probabilities"]
    )
    forged_summary = dict(summary)
    forged_summary["mean_paired_requested_log_probability_improvement"] = float(
        np.mean(forged_delta, dtype=np.float64)
    )
    runner._write_npz(  # noqa: SLF001
        pilot / "prior_classifier_outputs.npz", **forged
    )
    runner._write_json(  # noqa: SLF001
        pilot / "prior_classifier_metrics.json", forged_summary
    )
    runner._write_json(  # noqa: SLF001
        pilot / "metrics.json", {"prior_task_signal": forged_summary}
    )
    runner._seal_manifest(pilot)  # noqa: SLF001
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="prior-classifier raw output changed",
    ):
        runner._validate_pilot_prior_classifier(pilot)  # noqa: SLF001
    assert terminal_loads == []


def test_full_verifier_rejects_missing_or_resealed_tampered_admission_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    missing = tmp_path / "missing-full-admission"
    missing.mkdir()
    with pytest.raises((OSError, runner.EulerianJacobiDDPMRunError)):
        runner._validate_full_pilot_admission(missing)  # noqa: SLF001

    full = tmp_path / "full-admission"
    full.mkdir()
    pilot = full / runner.EMBEDDED_PILOT_DIRECTORY
    (pilot / "authority").mkdir(parents=True)
    for path in (
        pilot / "artifact_manifest.json",
        pilot / "config.json",
        pilot / "metrics.json",
        pilot / "status.json",
        pilot / "authority/source_bindings.json",
        pilot / "authority/environment.json",
    ):
        runner._write_json(path, {"fixture": path.name})  # noqa: SLF001
    result = {
        "integrity_passed": 1,
        "gate_c_passed": 1,
        "health_passed": 1,
        "full_scale_admitted": 1,
        "route": "full_scale_admitted",
        "learned_l1_win_count": 20,
        "tree_digest": "pilot-tree-digest",
    }
    admission = dict(result) | {
        "pilot_directory": runner.EMBEDDED_PILOT_DIRECTORY
    }
    runner._write_json(  # noqa: SLF001
        full / "objective_pilot_admission.json", admission
    )
    runner._write_pilot_admission_authority(full, pilot, admission)  # noqa: SLF001
    replayed_pilot_paths: list[Path] = []

    def replay_embedded_pilot(
        pilot_dir: Path,
        _parent_run_dir: Path,
    ) -> dict[str, object]:
        replayed_pilot_paths.append(Path(pilot_dir).resolve())
        return dict(result)

    monkeypatch.setattr(runner, "require_pilot_for_full_run", replay_embedded_pilot)
    assert runner._validate_full_pilot_admission(full)[  # noqa: SLF001
        "full_scale_admitted"
    ] == 1
    assert replayed_pilot_paths[-1] == pilot.resolve()

    moved = tmp_path / "moved-full-admission"
    full.rename(moved)
    assert runner._validate_full_pilot_admission(moved)[  # noqa: SLF001
        "full_scale_admitted"
    ] == 1
    assert replayed_pilot_paths[-1] == (
        moved / runner.EMBEDDED_PILOT_DIRECTORY
    ).resolve()
    for invalid in (
        str((moved / runner.EMBEDDED_PILOT_DIRECTORY).resolve()),
        "../objective_pilot",
        "another_pilot",
    ):
        invalid_admission = dict(admission)
        invalid_admission["pilot_directory"] = invalid
        runner._write_json(  # noqa: SLF001
            moved / "objective_pilot_admission.json", invalid_admission
        )
        with pytest.raises(
            runner.EulerianJacobiDDPMRunError,
            match="embedded relative directory",
        ):
            runner._validate_full_pilot_admission(moved)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        moved / "objective_pilot_admission.json", admission
    )
    forged_admission = dict(admission)
    forged_admission["learned_l1_win_count"] = 21
    runner._write_json(  # noqa: SLF001
        moved / "objective_pilot_admission.json", forged_admission
    )
    record = runner._read_json(  # noqa: SLF001
        moved / "pilot_admission_authority/record.json"
    )
    record["admission"] = forged_admission
    runner._write_json(  # noqa: SLF001
        moved / "pilot_admission_authority/record.json", record
    )
    runner._seal_manifest(moved)  # noqa: SLF001
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="objective-pilot admission changed",
    ):
        runner._validate_full_pilot_admission(moved)  # noqa: SLF001

    resources = tmp_path / "full-resource-admission"
    resources.mkdir()
    runner._write_json(  # noqa: SLF001
        resources / "config.json",
        {
            "execution_authority": {
                "device": "cuda",
                "approval_id": "frozen-approval",
                "maximum_active_seconds": 1_000.0,
                "maximum_storage_mib": 1_000_000_000 / 1024**2,
                "maximum_cuda_fraction": 1.0,
            }
        },
    )
    runner._write_json(  # noqa: SLF001
        resources / "source_bindings.json",
        {
            "evaluator_checkpoint_sha256": runner.ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256,
            "evaluator_selection_sha256": runner.ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256,
            "ddpm_metrics_sha256": runner.ACCEPTED_DDPM_METRICS_SHA256,
            "ddpm_manifest_sha256": runner.ACCEPTED_DDPM_MANIFEST_SHA256,
            "ddpm_status": "complete",
        },
    )
    receipt = {
        "checked_after_positive_pilot": 1,
        "approval_id": "frozen-approval",
        "ledger_event_count_at_check": 1,
        "active_seconds_at_check": 10.0,
        "storage_bytes_at_check": 0,
        "peak_cuda_fraction_at_check": 0.0,
        "maximum_active_seconds": 1_000.0,
        "maximum_storage_bytes": 1_000_000_000,
        "maximum_cuda_fraction": 1.0,
        "full_projected_seconds": 200.0,
        "full_projected_storage_bytes": 0,
        "terminal_reserve_seconds": 900.0,
        "time_cap_passed": 1,
        "storage_cap_passed": 1,
        "cuda_cap_passed": 1,
        "passed": 1,
    }
    projection = {
        "full": {"projected_seconds": 200.0},
        "full_projected_storage_bytes": 0,
        "terminal_reserve_seconds": 900.0,
        "post_pilot_full_admission": receipt,
    }
    runner._write_json(  # noqa: SLF001
        resources / "preflight_projection.json", projection
    )
    runner._write_json(  # noqa: SLF001
        resources / "resource_ledger.json",
        {
            "projected_cache_pair_transitions": 1_756_160_000,
            "projected_sampling_transition_work": 607_631_360,
            "projected_base_transition_work": 2_363_791_360,
            "active_seconds": 10.0,
            "events": [{"seconds": 10.0}],
            "maximum_active_seconds": 1_000.0,
            "maximum_storage_bytes": 1_000_000_000,
            "maximum_cuda_fraction": 1.0,
            "peak_cuda_fraction": 0.0,
            "approval_id": "frozen-approval",
            "latest_projection": projection,
        },
    )
    runner._seal_manifest(resources)  # noqa: SLF001
    monkeypatch.setattr(runner, "_validate_full_pilot_admission", lambda *_: {"passed": 1})
    monkeypatch.setattr(runner, "_validate_training_selection", lambda *_args, **_kwargs: {"passed": 1})
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="resource workload authority changed",
    ):
        runner._validate_full_decision_semantics(resources)  # noqa: SLF001

    forged_receipt = dict(receipt)
    forged_receipt["approval_id"] = "coordinated-forged-approval"
    forged_receipt["maximum_active_seconds"] = 2_000.0
    forged_projection = dict(projection)
    forged_projection["post_pilot_full_admission"] = forged_receipt
    runner._write_json(  # noqa: SLF001
        resources / "preflight_projection.json", forged_projection
    )
    forged_ledger = runner._read_json(  # noqa: SLF001
        resources / "resource_ledger.json"
    )
    forged_ledger["maximum_active_seconds"] = 2_000.0
    forged_ledger["approval_id"] = "coordinated-forged-approval"
    forged_ledger["latest_projection"] = forged_projection
    runner._write_json(  # noqa: SLF001
        resources / "resource_ledger.json", forged_ledger
    )
    runner._seal_manifest(resources)  # noqa: SLF001
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="resource.*authority|cap|approval",
    ):
        runner._validate_full_decision_semantics(resources)  # noqa: SLF001


def test_population_seal_recomputes_start_bank_and_trajectory_semantics(
    tmp_path: Path,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    def remove_integrity_records(run_dir: Path) -> None:
        for name in (
            "POPULATIONS_SEALED.json",
            "artifact_manifest.json",
            "SHA256SUMS.txt",
        ):
            (run_dir / name).unlink()

    start_tamper = tmp_path / "start-tamper"
    _write_objective_pilot_fixture(start_tamper, learned_improves=True)
    remove_integrity_records(start_tamper)
    with np.load(start_tamper / "start_banks.npz", allow_pickle=False) as archive:
        starts = {name: np.asarray(archive[name]).copy() for name in archive.files}
    starts["prior_sample_ids"][0] = "misbound-sample"
    runner._write_npz(start_tamper / "start_banks.npz", **starts)  # noqa: SLF001
    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="paired start"):
        runner.seal_populations(start_tamper)

    trajectory_tamper = tmp_path / "trajectory-tamper"
    _write_objective_pilot_fixture(trajectory_tamper, learned_improves=True)
    remove_integrity_records(trajectory_tamper)
    with np.load(trajectory_tamper / "populations.npz", allow_pickle=False) as archive:
        populations = {
            name: np.asarray(archive[name]).copy() for name in archive.files
        }
    learned_trajectory = populations["learned_prior_trajectories"]
    learned_trajectory[:, -1, 0] -= 1e-5
    learned_trajectory[:, -1, 1] += 1e-5
    runner._write_npz(  # noqa: SLF001
        trajectory_tamper / "populations.npz", **populations
    )
    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="endpoints"):
        runner.seal_populations(trajectory_tamper)


def test_production_population_seal_requires_and_binds_prior_start_authority(
    tmp_path: Path,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    missing = tmp_path / "missing-prior-authority"
    _write_objective_pilot_fixture(
        missing,
        learned_improves=True,
        production_schema=True,
        write_start_authority=False,
        seal=False,
    )
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="prior_start_authority",
    ):
        runner.seal_populations(missing)

    coordinated = tmp_path / "coordinated-downstream-start-tamper"
    _write_objective_pilot_fixture(
        coordinated,
        learned_improves=True,
        production_schema=True,
        write_start_authority=True,
        seal=False,
    )
    with np.load(coordinated / "start_banks.npz", allow_pickle=False) as archive:
        starts = {name: np.asarray(archive[name]).copy() for name in archive.files}
    shifted_start = starts["prior_starts"][0].copy()
    shifted_start[0] -= 1e-6
    shifted_start[1] += 1e-6
    starts["prior_starts"][0] = shifted_start
    runner._write_npz(coordinated / "start_banks.npz", **starts)  # noqa: SLF001
    with np.load(coordinated / "populations.npz", allow_pickle=False) as archive:
        populations = {
            name: np.asarray(archive[name]).copy() for name in archive.files
        }
    populations["null_prior_trajectories"][0, 0] = shifted_start
    populations["learned_prior_trajectories"][0, 0] = shifted_start
    runner._write_npz(  # noqa: SLF001
        coordinated / "populations.npz", **populations
    )
    with pytest.raises(
        runner.EulerianJacobiDDPMRunError,
        match="prior-start authority changed: prior_starts",
    ):
        runner.seal_populations(coordinated)


def test_production_review_uses_the_sealed_prior_identity_schema(tmp_path: Path) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    labels = np.arange(10, dtype=np.int64)
    images = np.zeros((10, 28, 28), dtype=np.uint8)
    runner._write_json(  # noqa: SLF001
        tmp_path / "config.json",
        {
            "schema": runner.VERSION,
            "populations": {"review_per_row_per_class": 1},
        },
    )
    runner._write_npz(  # noqa: SLF001
        tmp_path / "uint8_populations.npz",
        learned_prior=images,
        null_prior=images,
        prior_requested_labels=labels,
    )

    review_images, review_labels, sample_ids = runner._review_population(tmp_path)  # noqa: SLF001
    assert review_images.shape == (20, 28, 28)
    np.testing.assert_array_equal(review_labels, np.concatenate((labels, labels)))
    assert len(set(sample_ids.tolist())) == 20


def test_smoke_artifacts_seal_before_terminal_open_and_verify_clean(
    tmp_path: Path,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    run_dir = tmp_path / "smoke"
    runner.initialize_smoke_run(run_dir)
    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="seal"):
        runner.open_terminal_evidence(run_dir)
    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="seal"):
        runner.create_review_bundle(run_dir)

    runner.execute_smoke_run(run_dir)
    seal = runner.seal_populations(run_dir)
    assert seal["sealed"] == 1
    runner.open_terminal_evidence(run_dir)
    runner.create_review_bundle(run_dir)
    verified = runner.finalize_and_verify(run_dir)
    assert verified["passed"] == 1
    assert (run_dir / "REPORT.md").is_file()
    assert (run_dir / "artifact_manifest.json").is_file()
    assert (run_dir / "SHA256SUMS.txt").is_file()

    populations = run_dir / "populations.npz"
    populations.write_bytes(populations.read_bytes() + b"tamper")
    with pytest.raises(runner.EulerianJacobiDDPMRunError, match="hash|manifest|changed"):
        runner.verify_run(run_dir)


def test_production_operational_failure_is_retained_sealed_and_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_eulerian_jacobi_ddpm_mnist as runner

    run_dir = tmp_path / "failed-production"
    run_dir.mkdir()
    repository_root = Path(runner.__file__).resolve().parents[1]
    runtime_config = dict(runner.FROZEN_CONFIG)
    runtime_config["execution_authority"] = {
        "approval_id": "test-approval",
        "device": "cuda",
        "maximum_active_seconds": 1.0,
        "maximum_storage_mib": 1.0,
        "maximum_cuda_fraction": 1.0,
        "whole_run_restart_only": 1,
        "full_scale_launch_supported": 1,
    }
    runner._write_json(run_dir / "config.json", runtime_config)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        run_dir / "source_bindings.json",
        {
            "repository_root": str(repository_root),
            "source_files": runner._source_hashes(repository_root),  # noqa: SLF001
            "config_sha256": runner._semantic_sha256(runtime_config),  # noqa: SLF001
        },
    )

    monkeypatch.setattr(runner, "initialize_run", lambda *_, **__: run_dir)

    def fail_preflight(_: Path) -> None:
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(runner, "run_numerical_preflight", fail_preflight)
    args = SimpleNamespace(
        arff="unused.arff",
        ddpm_run_dir="unused-ddpm",
        run_dir=str(run_dir),
        device="cuda",
        max_active_seconds=1.0,
        max_storage_mib=1.0,
        max_cuda_fraction=1.0,
        approval_id="test-approval",
        pilot_dir=None,
    )

    assert runner.run_production(args) == 4
    failure = runner._read_json(run_dir / "failure.json")  # noqa: SLF001
    assert failure["error_type"] == "RuntimeError"
    assert failure["message"] == "synthetic preflight failure"
    assert failure["completed_pilot_route"] is None
    assert failure["population_sealed"] == 0
    assert failure["terminal_test_opened"] == 0
    assert runner._read_json(run_dir / "status.json")["state"] == "failed"  # noqa: SLF001
    before = runner._directory_digest(run_dir)  # noqa: SLF001
    verified = runner.verify_run(run_dir)
    assert verified["passed"] == 1
    assert verified["route"] == "failed"
    assert runner._directory_digest(run_dir) == before  # noqa: SLF001
