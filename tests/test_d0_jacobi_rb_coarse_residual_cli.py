from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist import diag_d0_jacobi_rb_coarse_residual_learnability as cli
from mnist.d0_jacobi_rb_coarse_residual import (
    PRIMARY_CONTRAST_NAMES,
    PathContrastTable,
    CoarseResidualPredictor,
    FrozenCoarseBaseline,
    exact_combined_target_scale,
    load_frozen_witness_baseline,
    one_sided_studentized_max_t,
    save_frozen_coarse_baseline,
)
from mnist.d0_jacobi_rb_coarse_residual_gate import (
    decide_coarse_residual_workflow,
    evaluate_coarse_residual_workflow,
)
from mnist.d0_jacobi_artifacts import atomic_write_json, file_fingerprint
from mnist.d0_jacobi_rb_learnability import ModelInputs, selected_reverse_time


WITNESS = Path(
    "runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness/"
    "20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix"
)
LEARNER = Path(
    "runs/experiment12_d0_jacobi_rb_one_image_learnability/"
    "20260729-015817_production-exact-k512-rb-one-image-learnability"
)


def _args(tmp_path: Path, *, test_only: bool = False) -> argparse.Namespace:
    witness = tmp_path / "witness"
    witness.mkdir(exist_ok=True)
    (witness / "physical_capture_benchmark.json").write_text(
        json.dumps({"transitions_per_second": 2_700.0}),
        encoding="utf-8",
    )
    return argparse.Namespace(
        runs_root=tmp_path,
        run_name="fixture",
        device="cpu",
        stage="preflight",
        require_gate="none",
        parent_coarse_witness_run_dir=witness,
        parent_one_image_run_dir=tmp_path / "learner",
        resume_run_dir=None,
        test_only=test_only,
        test_paths_per_role=1,
        test_outer_steps=16,
        test_maximum_updates=100,
    )


def test_parser_exposes_staged_fail_closed_workflow(tmp_path: Path) -> None:
    args = cli.parse_args(
        [
            "--runs-root",
            str(tmp_path),
            "--stage",
            "confirm",
            "--require-gate",
            "confirm",
            "--parent-coarse-witness-run-dir",
            str(tmp_path / "witness"),
            "--parent-one-image-run-dir",
            str(tmp_path / "learner"),
        ]
    )
    assert args.stage == "confirm"
    assert args.require_gate == "confirm"
    assert args.parent_coarse_witness_run_dir.is_absolute()
    assert args.parent_one_image_run_dir.is_absolute()


def test_frozen_path_plan_has_disjoint_production_roles(tmp_path: Path) -> None:
    record = cli._path_plan(_args(tmp_path))
    roles = record["roles"]
    assert len(roles["train"]) == 64
    assert len(roles["validation"]) == 32
    assert len(roles["confirmation"]) == 64
    assert len(roles["benchmark"]) == 8
    flattened = [value for values in roles.values() for value in values]
    assert len(flattened) == len(set(flattened))
    assert all(0 <= value < 2**20 for value in flattened)
    assert record["projected_transition_count"] == 224_788_480
    assert record["test_only_reduced_workload"] == 0


def test_reduced_path_plan_is_explicitly_nonauthorizing(tmp_path: Path) -> None:
    args = _args(tmp_path, test_only=True)
    record = cli._path_plan(args)
    assert record["test_only_reduced_workload"] == 1
    assert all(len(values) == 1 for values in record["roles"].values())
    config = cli._scientific_config(args)
    assert config["authorizing"] == 0
    assert config["outer_steps"] == 16


@pytest.mark.skipif(not WITNESS.is_dir(), reason="production witness unavailable")
def test_literal_baseline_record_is_idempotent(tmp_path: Path) -> None:
    first = cli._baseline_record(tmp_path, WITNESS.resolve())
    second = cli._baseline_record(tmp_path, WITNESS.resolve())
    assert first == second
    assert first["values_sha256"] == cli.EXPECTED_BASELINE_VALUES_SHA256
    assert first["baseline_energy"] == cli.EXPECTED_BASELINE_ENERGY
    assert first["signed_values_retained"] == 1
    assert first["no_clipping"] == 1
    assert first["no_thresholding"] == 1
    assert first["no_adaptive_refit"] == 1


def test_confirmation_seal_is_not_confirmation_evidence(tmp_path: Path) -> None:
    assert cli._no_confirmation_artifacts(tmp_path)
    assert cli._no_confirmation_evidence(tmp_path)
    (tmp_path / "confirmation_seal.json").write_text("{}", encoding="utf-8")
    assert not cli._no_confirmation_artifacts(tmp_path)
    assert cli._no_confirmation_evidence(tmp_path)
    (tmp_path / "confirmation_open.json").write_text("{}", encoding="utf-8")
    assert not cli._no_confirmation_evidence(tmp_path)


@pytest.mark.skipif(not WITNESS.is_dir(), reason="production witness unavailable")
def test_candidate_metrics_gate_high_reverse_time_quartile() -> None:
    baseline = load_frozen_witness_baseline(WITNESS.resolve())
    model = CoarseResidualPredictor(baseline)
    steps = (15, 143, 271, 399)
    phase = torch.zeros(4, dtype=torch.long)
    inputs = ModelInputs(
        later_full_state=torch.full((4, 784), 1.0 / 784.0),
        reverse_time=torch.tensor(
            [selected_reverse_time(step, 0) for step in steps],
            dtype=torch.float32,
        ),
        phase=phase,
        color=torch.zeros(4, dtype=torch.long),
        duration=torch.full((4,), 0.5),
        label=torch.full((4,), 3, dtype=torch.long),
    )
    prediction = cli._predict_model(model, inputs)
    target = prediction + torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64
    ).expand(-1, 392)
    metrics, replay = cli._candidate_metrics(
        model,
        inputs,
        target,
        baseline_mse_overall=7.5,
        baseline_mse_high_reverse_time=1.0,
    )
    assert torch.equal(replay, prediction)
    assert metrics["validation_mse"] == pytest.approx(7.5)
    assert metrics["validation_high_reverse_time_mse"] == pytest.approx(1.0)
    assert "validation_data_end_mse" not in metrics


def test_artifact_registry_is_atomic_and_self_consistent(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text('{"passed":1}', encoding="utf-8")
    record = cli._artifact_registry(tmp_path)
    assert record["artifact_count"] == 1
    assert record["artifacts"][0]["path"] == "evidence.json"
    verified = cli._verify_registry(tmp_path)
    assert verified == record


def test_artifact_registry_excludes_uncommitted_npz_temporary(
    tmp_path: Path,
) -> None:
    (tmp_path / "committed.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shard.tmp.npz").write_bytes(b"uncommitted")
    record = cli._artifact_registry(tmp_path)
    assert [item["path"] for item in record["artifacts"]] == ["committed.json"]


def test_physical_label_and_training_start_seals_are_resume_idempotent(
    tmp_path: Path,
) -> None:
    (tmp_path / "cache").mkdir()
    atomic_write_json(
        tmp_path / "optimization_control_gate.json",
        {"evaluation_status": "evaluated", "passed": 1},
    )
    (tmp_path / "cache" / "train_labels_audit.npz").write_bytes(b"train")
    (tmp_path / "cache" / "validation_labels_audit.npz").write_bytes(b"validation")
    train = SimpleNamespace(sample_count=11)
    validation = SimpleNamespace(sample_count=7)
    first = cli._open_physical_labels(
        tmp_path,
        train_audit=train,
        validation_audit=validation,
        target_scale=1.25,
    )
    second = cli._open_physical_labels(
        tmp_path,
        train_audit=train,
        validation_audit=validation,
        target_scale=1.25,
    )
    assert second == first
    started_first = cli._freeze_physical_training_started(
        tmp_path, target_scale=1.25
    )
    started_second = cli._freeze_physical_training_started(
        tmp_path, target_scale=1.25
    )
    assert started_second == started_first


def test_confirmation_seal_and_open_are_resume_idempotent(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    plan = cli._path_plan(args)
    atomic_write_json(tmp_path / "path_id_plan.json", plan)
    (tmp_path / "selected_model.pt").write_bytes(b"selected")
    selected = {
        "semantic_sha256": "a" * 64,
        "selected_model_sha256": file_fingerprint(tmp_path / "selected_model.pt"),
        "nonzero_residual_selected": 1,
    }
    seal_first = cli._freeze_confirmation_seal(
        tmp_path, args, selected_model=selected
    )
    seal_second = cli._freeze_confirmation_seal(
        tmp_path, args, selected_model=selected
    )
    assert seal_second == seal_first
    atomic_write_json(tmp_path / "selected_model.json", selected)
    open_first = cli._open_confirmation(tmp_path, args)
    open_second = cli._open_confirmation(tmp_path, args)
    assert open_second == open_first
    assert open_first["open_count"] == 1


def test_split_cache_gate_adapter_requires_resource_and_projection_fields(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    thresholds = cli._gate_thresholds(args)
    record = {
        "split": "train",
        "path_count": thresholds.train_paths,
        "sample_count": thresholds.train_samples,
        "transition_count": thresholds.train_transitions,
        "selected_outer_steps": list(thresholds.selected_outer_steps),
        "certificate_fraction": 1.0,
        "maximum_mass_error": 1.0e-15,
        "transitions_per_second": 2_000.0,
        "peak_memory_fraction": 0.1,
        "cache_complete_pass": 1,
        "cache_replay_hash_pass": 1,
        "all_shards_complete_pass": 1,
        "capture_state_alignment_pass": 1,
        "sample_key_unique_pass": 1,
        "selected_step_phase_coverage_pass": 1,
        "states_finite_pass": 1,
        "targets_finite_pass": 1,
        "sample_key_join_pass": 1,
        "split_role_isolation_pass": 1,
        "path_plan_binding_pass": 1,
        "baseline_hash_binding_pass": 1,
        "model_input_firewall_pass": 1,
        "exact_jacobi_transition_pass": 1,
        "exact_rb_target_pass": 1,
        "unmodified_binary64_target_pass": 1,
        "state_updates_device_resident_pass": 1,
        "confirmation_absent_pass": 1,
        "uncertified_count": 0,
        "resource_cap_count": 0,
        "invalid_density_count": 0,
        "approximation_count": 0,
        "correction_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "projection_count": 0,
        "renormalization_count": 0,
        "nonfinite_count": 0,
        "target_modification_count": 0,
        "residual_target_persisted": 0,
    }
    assert cli._evaluate_cache_gate(record, args)["passed"] == 1
    record["projection_count"] = 1
    assert cli._evaluate_cache_gate(record, args)["passed"] == 0
    record["projection_count"] = 0
    record["transitions_per_second"] = 1_299.0
    assert cli._evaluate_cache_gate(record, args)["passed"] == 0


def test_confirmation_max_t_keeps_exact_two_name_family() -> None:
    paths = np.arange(64, dtype=np.int64) + 0xE8000
    values = np.zeros((64, 6), dtype=np.float64)
    values[:, 0] = np.linspace(0.1, 0.2, 64)
    values[:, 1] = np.linspace(0.05, 0.1, 64)
    values[:, 2] = values[:, 0] + values[:, 1]
    values[:, 3:] = values[:, :3]
    table = PathContrastTable(paths, values)
    result = one_sided_studentized_max_t(
        table,
        family_names=PRIMARY_CONTRAST_NAMES,
        seed=cli.BOOTSTRAP_SEED,
        replicates=128,
    ).to_record()
    assert result["family_names"] == list(PRIMARY_CONTRAST_NAMES)
    assert set(result["point_estimates"]) == set(PRIMARY_CONTRAST_NAMES)
    assert not any(name.startswith("delta_") for name in result["family_names"])


def test_baseline_only_is_closed_gate_failure_not_pipeline_failure() -> None:
    passed = {"evaluation_status": "evaluated", "passed": 1}
    train = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "coarse_baseline_only": 1,
        "optimization_pipeline_valid": 1,
    }
    decision = decide_coarse_residual_workflow(passed, passed, train, None)
    assert decision["decision"] == "coarse_baseline_only_signal"
    workflow = evaluate_coarse_residual_workflow(
        preflight_gate=passed,
        cache_gate=passed,
        train_gate=train,
        confirm_gate=None,
        require_gate="train",
    )
    assert workflow["required_gate_pass"] == 0


def test_tiny_cpu_training_resume_is_exact(tmp_path: Path) -> None:
    raw = np.linspace(-0.02, 0.02, 4 * 7 * 392, dtype=np.float64).reshape(
        4, 7, 392
    )
    baseline = FrozenCoarseBaseline(
        raw_values=raw,
        values=0.5 * raw,
        left_path_ids=np.arange(64, dtype=np.int64),
        right_path_ids=np.arange(64, 128, dtype=np.int64),
        shrinkage=0.5,
        signal_energy=1.0,
        panel_mean_noise=2.0,
        averaged_table_noise=1.0,
        left_cell_means_file_sha256="1" * 64,
        right_cell_means_file_sha256="2" * 64,
        left_cell_means_array_sha256="3" * 64,
        right_cell_means_array_sha256="4" * 64,
        witness_registry_sha256="5" * 64,
    )
    save_frozen_coarse_baseline(
        tmp_path / "frozen_coarse_baseline.npz", baseline
    )
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "train_inputs.npz").write_bytes(b"train")
    (tmp_path / "cache" / "validation_inputs.npz").write_bytes(b"validation")
    (tmp_path / "training_plan.json").write_text(
        json.dumps({"maximum_updates": 1}), encoding="utf-8"
    )
    (tmp_path / "scientific_config.json").write_text(
        json.dumps(
            {
                "semantic_sha256": "6" * 64,
                "training": {"maximum_updates": 1},
            }
        ),
        encoding="utf-8",
    )
    batch = 8
    inputs = ModelInputs(
        later_full_state=torch.full((batch, 784), 1.0 / 784.0),
        reverse_time=torch.full(
            (batch,), selected_reverse_time(15, 0), dtype=torch.float32
        ),
        phase=torch.zeros(batch, dtype=torch.long),
        color=torch.zeros(batch, dtype=torch.long),
        duration=torch.full((batch,), 0.5),
        label=torch.full((batch,), 3, dtype=torch.long),
    )
    initial = cli._model_factory(tmp_path)()
    target = initial.baseline_prediction(inputs, dtype=torch.float64).detach()
    scale = exact_combined_target_scale(target)
    kwargs = {
        "task": "cpu-null",
        "seed": 261252,
        "train_inputs": inputs,
        "train_target": target,
        "validation_inputs": inputs,
        "validation_target": target,
        "target_scale": scale,
        "target_kind": "tiny-cpu-exact-baseline-null",
        "physical": False,
    }
    first = cli._train_task(tmp_path, argparse.Namespace(), **kwargs)
    second = cli._train_task(tmp_path, argparse.Namespace(), **kwargs)
    assert first == second
    assert first["complete"] == 1
    assert first["selected"]["update"] == 0
    assert first["update_zero_baseline_exact_pass"] == 1


def test_stage_sequence_never_opens_confirmation_during_train() -> None:
    assert cli._stage_sequence("train") == ("train",)
    assert cli._stage_sequence("all") == (
        "preflight",
        "cache",
        "train",
        "confirm",
    )


def test_source_has_no_reverse_sampler_import() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "reverse_sampler" not in source
    assert "reconstruction_sampler" not in source
