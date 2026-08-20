from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist import diag_d0_jacobi_rb_rollout_reweighted as workflow
from mnist.d0_jacobi_artifacts import atomic_write_json, file_fingerprint
from mnist.d0_jacobi_rb_global_dilated import GlobalDilatedZeroBaselinePredictor
from mnist.d0_jacobi_rb_learnability import CheckpointCandidate, TrainingResumeSnapshot, semantic_sha256, state_dict_sha256
from mnist.d0_jacobi_rb_tangent_rollout import atomic_rollout_npz


def _simplex(delta: float = 0.0) -> np.ndarray:
    state = np.full(784, 1.0 / 784.0, dtype=np.float64)
    state[0] += delta
    state[1] -= delta
    return state


def _model_state() -> dict[str, torch.Tensor]:
    model = GlobalDilatedZeroBaselinePredictor(zero_residual=False)
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _snapshot(
    update: int, mse: float, state: dict[str, torch.Tensor], history: tuple[dict, ...] | None = None,
) -> TrainingResumeSnapshot:
    candidate = CheckpointCandidate(
        workflow.TRAINING_SEED, update, mse, state_dict_sha256(state), state,
    )
    return TrainingResumeSnapshot(
        seed=workflow.TRAINING_SEED,
        completed_update=update,
        model_state_dict=state,
        optimizer_state_dict={},
        best_candidate=candidate,
        history=history or ({"update": update, "validation_mse": mse},),
        finite=True,
        torch_rng_state=torch.get_rng_state().clone(),
        cuda_rng_states=(),
    )


def _stage_e_capsule(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, np.ndarray]:
    capsule = root / "stage-e"
    row_order = ["source-informed", "zero", "global-cutoff-216", "global-plus-1"]
    family = np.stack([_simplex((row + 1) * 1.0e-5) for row in range(4)])
    states = np.repeat(family[None, :, :], 65, axis=0)
    atomic_write_json(
        capsule / "config.json",
        {"experiment_name": "stage-e-prior-cutoff-216-v1", "row_order": row_order,
         "rng_binding": {"canonical_path_id": 1_028_865}},
    )
    atomic_write_json(
        capsule / "outcome.json",
        {"run_state": "complete", "health_passed": 1,
         "schedule_identity_passed": 1, "scientific_objective_completed": 1},
    )
    atomic_rollout_npz(
        capsule / "reverse/trajectory_boundaries.npz",
        {"completed_reverse_steps": np.arange(0, 513, 8, dtype=np.int64), "states": states},
    )
    monkeypatch.setattr(workflow, "_verify_stage_e_manifest", lambda _root: None)
    hashes = {relative: file_fingerprint(capsule / relative) for relative in workflow.STAGE_E_HASHES}
    monkeypatch.setattr(workflow, "STAGE_E_HASHES", hashes)
    return capsule, states


def _args(root: Path, *, run_name: str = workflow.RUN_NAME) -> argparse.Namespace:
    return argparse.Namespace(
        repository_root=str(root / "repository"), training_parent=str(root / "training-parent"),
        baseline_run_dir=str(root / "baseline"), stage_e_run_dir=str(root / "stage-e"),
        runs_root=str(root / "runs"), run_name=run_name, device="cpu",
        maximum_active_seconds=workflow.ACTIVE_SECONDS_CAP,
        approval_reference=workflow.APPROVAL_REFERENCE,
    )


def _rows(frozen: float, candidate: float, oracle: float = 0.1, correlation: float = 0.6) -> dict:
    return {
        "zero": {"squared_l2_error": 1.0, "centered_correlation": 0.0},
        "frozen-cutoff-216": {"squared_l2_error": frozen, "centered_correlation": 0.2},
        "reweighted-cutoff-216": {"squared_l2_error": candidate, "centered_correlation": correlation},
        "source-informed": {"squared_l2_error": oracle, "centered_correlation": 0.9},
    }


def _summaries(
    *, endpoint: tuple[float, float, float, float] = (0.8, 0.4, 0.1, 0.6),
    boundary: tuple[float, float, float, float] = (0.8, 0.5, 0.1, 0.6),
) -> list[dict]:
    endpoint_rows = _rows(*endpoint)
    boundary_rows = _rows(*boundary)
    horizons = {"208": boundary_rows, "216": boundary_rows, "512": endpoint_rows}
    return [{"path_id": path_id, "horizons": horizons} for path_id in workflow.EVAL_PATH_IDS]


def _review(label: str = "3") -> dict:
    return {
        "schema": workflow.VERSION + "-human-review",
        "status": "reviewed",
        "reviewer": "test-reviewer",
        "labels": [{"path_id": path_id, "label": label} for path_id in workflow.EVAL_PATH_IDS],
        "allowed_labels": list(workflow.HUMAN_LABELS),
        "automated_recognizability": 0,
    }


def _torch_inputs(count: int, *, simplex: bool = False) -> workflow.ModelInputs:
    states = np.repeat(_simplex()[None, :], count, axis=0) if simplex else np.zeros((count, 784))
    return workflow.ModelInputs(
        later_full_state=torch.as_tensor(states, dtype=torch.float32), reverse_time=torch.zeros(count),
        phase=torch.zeros(count, dtype=torch.long), color=torch.zeros(count, dtype=torch.long),
        duration=torch.ones(count), label=torch.full((count,), 3, dtype=torch.long),
    )


def test_literal_experiment_contract_has_fresh_disjoint_three_path_families() -> None:
    assert workflow.VERSION == "d0-jacobi-rb-rollout-reweighted-v1"
    assert workflow.RUN_NAME == "stage-e-rollout-reweighted-v1"
    assert workflow.EVAL_PATH_IDS == (0xE9008, 0xE9009, 0xE900A)
    assert workflow.ROW_ORDER == ("zero", "frozen-cutoff-216", "reweighted-cutoff-216", "source-informed")
    assert workflow.RENDER_HORIZONS == (0, 16, 128, 208, 216, 224, 256, 384, 512)
    assert set(workflow.EVAL_PATH_IDS).isdisjoint(workflow.PROTECTED_CONFIRMATION_PATH_IDS)
    config = workflow._frozen_config()
    assert config["evaluation"]["path_ids"] == list(workflow.EVAL_PATH_IDS)
    assert config["evaluation"]["row_order"] == list(workflow.ROW_ORDER)
    assert config["confirmation_evidence_opened"] == 0
    assert config["automatic_stage_f_launch"] == 0
    for path_id in workflow.EVAL_PATH_IDS:
        family = workflow._row_specs(path_id, "a" * 64)
        assert tuple(row.row_key for row in family) == workflow.ROW_ORDER
        assert {row.canonical_path_id for row in family} == {path_id}
        assert len(family) == 4
        assert workflow._rng_binding(path_id) == {
            "root_seed": workflow.TRANSITION_ROOT_SEED, "stream_role": workflow.STREAM_ROLE,
            "canonical_path_id": path_id, "same_random_bits_across_rows": 1,
        }


def test_stage_e_capsule_is_hash_pinned_name_selected_and_tamper_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, states = _stage_e_capsule(tmp_path, monkeypatch)
    loaded = workflow._validate_stage_e_capsule(capsule)
    np.testing.assert_array_equal(loaded["selected_states"], states[:, 2, :])

    outcome = json.loads((capsule / "outcome.json").read_text(encoding="utf-8"))
    outcome["health_passed"] = 0
    atomic_write_json(capsule / "outcome.json", outcome)
    with pytest.raises(workflow.RolloutReweightedRunError, match="authority changed"):
        workflow._validate_stage_e_capsule(capsule)


def test_checkpoint_requires_both_file_and_state_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _model_state()
    state_hash = state_dict_sha256(state)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": state, "state_sha256": state_hash}, checkpoint)
    monkeypatch.setattr(workflow, "BASELINE_FILE_SHA256", file_fingerprint(checkpoint))
    monkeypatch.setattr(workflow, "BASELINE_STATE_SHA256", state_hash)
    loaded = workflow._load_checkpoint(checkpoint, "cpu")
    assert state_dict_sha256(loaded.state_dict()) == state_hash

    with checkpoint.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(workflow.RolloutReweightedRunError, match="file changed"):
        workflow._load_checkpoint(checkpoint, "cpu")


def test_checkpoint_selection_excludes_update_zero_nonfinite_and_uses_earlier_tie() -> None:
    selected = workflow.select_nonzero_checkpoint(
        [
            {"update": 0, "validation_mse": 0.0, "name": "parent"},
            {"update": 100, "validation_mse": float("nan")},
            {"update": 300, "validation_mse": 1.0, "name": "late"},
            {"update": 200, "validation_mse": 1.0, "name": "winner"},
        ]
    )
    assert selected == {"update": 200, "validation_mse": 1.0, "name": "winner"}
    with pytest.raises(workflow.RolloutReweightedRunError, match="no finite nonzero"):
        workflow.select_nonzero_checkpoint(
            [{"update": 0, "validation_mse": 0.0}, {"update": 100, "validation_mse": float("inf")}]
        )


def test_resume_reconciles_best_and_selected_checkpoint_is_a_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _model_state()
    history = (
        {"update": 100, "validation_mse": 0.3},
        {"update": 200, "validation_mse": 0.2},
    )
    snapshot = _snapshot(200, 0.2, state, history)
    best_path = tmp_path / "training/best_nonzero.pt"
    workflow._atomic_torch(best_path, workflow._candidate_from_resume(_snapshot(100, 0.3, state)))
    workflow._reconcile_best_from_resume(snapshot, best_path)
    assert torch.load(best_path, weights_only=True)["update"] == 200

    selected = tmp_path / "training/selected_checkpoint.pt"
    selected.touch()
    sentinel = {"fine_tune_update": 200}
    monkeypatch.setattr(workflow, "_selected_payload", lambda path: sentinel if path == selected else None)
    monkeypatch.setattr(workflow, "_prepare_training_data", lambda *_args: pytest.fail("fast path retrained"))
    assert workflow._train(tmp_path, tmp_path, torch.device("cpu")) is sentinel


def test_selected_payload_strictly_enforces_gate_b_authorities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    state = {"weight": torch.zeros(1)}
    state_hash = state_dict_sha256(state)
    summary = {split: {"original_row_count": 1, "augmented_row_count": 2}
               for split in ("train", "validation")}
    summary["semantic_sha256"] = semantic_sha256(summary)
    payload = {
        "schema": workflow.VERSION + "-selected-checkpoint", "fine_tune_update": 4000,
        "validation_mse": 0.5, "state_sha256": state_hash, "state_dict": state,
        "parent_file_sha256": workflow.BASELINE_FILE_SHA256, "parent_state_sha256": workflow.BASELINE_STATE_SHA256,
        "reweighting_semantic_sha256": summary["semantic_sha256"],
    }
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    mse_fields = ("parent_weighted_mse", "parent_original_unweighted_mse",
                  "candidate_weighted_mse", "candidate_original_unweighted_mse")
    diagnostics = {
        "schema": workflow.VERSION + "-training-diagnostics", "selected_update": 4000,
        "selected_state_sha256": state_hash,
        "splits": {split: {"original_row_count": 1, "weighted_row_count": 2,
                            **{name: 0.5 for name in mse_fields}}
                   for split in ("train", "validation")},
    }
    records = {"summary.json": summary, "selection.json": metadata, "diagnostics.json": diagnostics}
    history = ({"update": 4000, "validation_mse": 0.5},)
    monkeypatch.setattr(workflow, "_read_json", lambda path: records[path.name])
    monkeypatch.setattr(workflow, "_verify_reweighting", lambda *_args: None)
    monkeypatch.setattr(workflow, "_validated_model_state", lambda value, _digest: dict(value))
    monkeypatch.setattr(workflow, "_resume_snapshot", lambda _path: SimpleNamespace(completed_update=4000, finite=True, history=history))
    monkeypatch.setattr(workflow, "_history_csv", lambda _path: history)
    monkeypatch.setattr(workflow, "_validated_best_nonzero", lambda *_args: {
        "update": 4000, "validation_mse": 0.5, "state_sha256": state_hash,
    })
    selected = run / "training/selected_checkpoint.pt"
    workflow._atomic_torch(run / "training/best_nonzero.pt", {})
    workflow._atomic_torch(selected, payload)
    assert workflow._selected_payload(selected)["fine_tune_update"] == 4000

    workflow._atomic_torch(selected, {**payload, "fine_tune_update": 0})
    with pytest.raises(workflow.RolloutReweightedRunError, match="selected candidate"):
        workflow._selected_payload(selected)
    workflow._atomic_torch(selected, payload)
    history = ({"update": 100, "validation_mse": 0.4}, {"update": 4000, "validation_mse": 0.5})
    with pytest.raises(workflow.RolloutReweightedRunError, match="frozen best"):
        workflow._selected_payload(selected)


def test_training_commits_selected_checkpoint_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    ledger = {"active_seconds": 0.0, "events": [], "maximum_active_seconds": workflow.ACTIVE_SECONDS_CAP}
    atomic_write_json(run / "resource_ledger.json", ledger)
    state = _model_state()
    snapshot = _snapshot(4000, 0.5, state)
    reweighting = SimpleNamespace(record={"semantic_sha256": "a" * 64})
    monkeypatch.setattr(workflow, "_prepare_training_data", lambda *_args: (None, None, None, None, reweighting))
    monkeypatch.setattr(workflow, "_resource_check", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "_training_diagnostics", lambda *_args: {"schema": "test"})

    def fake_train(*_args, **kwargs):
        kwargs["checkpoint_callback"](snapshot)
        return SimpleNamespace(finite=True, history=snapshot.history)

    monkeypatch.setattr(workflow, "train_deterministic_regressor", fake_train)
    monkeypatch.setattr(workflow, "_selected_payload", lambda path: torch.load(path, weights_only=True))
    order = []
    def traced(writer):
        def wrapped(path: Path, value) -> None:
            order.append(path.name)
            writer(path, value)
        return wrapped

    monkeypatch.setattr(workflow, "_atomic_torch", traced(workflow._atomic_torch))
    monkeypatch.setattr(workflow, "atomic_write_json", traced(workflow.atomic_write_json))
    monkeypatch.setattr(workflow, "atomic_write_csv", traced(workflow.atomic_write_csv))
    workflow._train(run, tmp_path, torch.device("cpu"))
    assert order[-1] == "selected_checkpoint.pt"
    assert order.index("history.csv") < order.index("selection.json") < order.index("selected_checkpoint.pt")


@pytest.mark.parametrize("failure", [workflow.ResourcePause("pause"), RuntimeError("boom")])
def test_failed_training_attempt_is_charged_and_pending_time_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    run = tmp_path / type(failure).__name__
    ledger = {"active_seconds": 0.0, "events": [], "maximum_active_seconds": workflow.ACTIVE_SECONDS_CAP}
    atomic_write_json(run / "resource_ledger.json", ledger)
    prepared = (None, None, None, None, SimpleNamespace(record={}))
    monkeypatch.setattr(workflow, "_prepare_training_data", lambda *_args: prepared)
    def fail(*_args, **_kwargs):
        raise failure
    monkeypatch.setattr(workflow, "train_deterministic_regressor", fail)
    ticks = iter((0.0, 1.0, 1.0, 4.0))
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(type(failure)):
        workflow._train(run, tmp_path, torch.device("cpu"))
    ledger = json.loads((run / "resource_ledger.json").read_text())
    assert ledger["active_seconds"] == 4.0
    assert [(row["role"], row["failed"]) for row in ledger["events"]] == [
        ("training-data-preparation", 0),
        ("training-attempt", 1),
    ]

    ledger["active_seconds"] = workflow.ACTIVE_SECONDS_CAP - 1.0
    atomic_write_json(run / "resource_ledger.json", ledger)
    with pytest.raises(workflow.ResourcePause, match="active-time cap"):
        workflow._resource_check(run, pending_seconds=1.0)
    monkeypatch.setattr(workflow, "_terminal_inputs", lambda _run: (_ for _ in ()).throw(RuntimeError("terminal")))
    ticks = iter((10.0, 12.0))
    with pytest.raises(RuntimeError, match="terminal"):
        workflow._finalize(run)
    assert json.loads((run / "resource_ledger.json").read_text())["events"][-1]["role"] == "terminalization"


def test_four_row_health_and_cutoff_telemetry_are_enforced() -> None:
    controller_keys = (
        "clipping_count", "floor_count", "projection_count", "nonfinite_score_count",
        "score_squared_sum", "score_maximum_absolute", "unscaled_score_squared_sum",
        "unscaled_score_maximum_absolute",
    )
    dynamics_keys = (
        "input_invalid", "state_invalid", "mass_invalid", "metadata_invalid", "score_invalid",
        "logistic_shift_invalid", "reference_fraction_invalid", "reference_invalid_count",
        "logistic_shift_squared_sum", "logistic_shift_maximum_absolute",
    )
    controller = dict.fromkeys(controller_keys, 0)
    dynamics = dict.fromkeys(dynamics_keys, 0)
    dynamics.update(transition_count=87_808, reference_transition_count=87_808)
    records = [
        {
            "transition_count": workflow.PER_SHARD_TRANSITIONS,
            "controller_diagnostics": [dict(controller) for _ in workflow.ROW_ORDER],
            "per_row_diagnostics": [dict(dynamics) for _ in workflow.ROW_ORDER],
            "diagnostics": {"maximum_mass_error": 0.0, "invalid_count": 0,
                "fallback_count": 0, "forbidden_counts": {}, "reference": {
                    "root_seed": workflow.TRANSITION_ROOT_SEED, "stream_role": workflow.STREAM_ROLE,
                    "variant_in_rng_key": 0}},
        }
        for _ in range(workflow.SHARD_COUNT)
    ]
    state = np.full((4, 784), 1.0 / 784.0, dtype=np.float64)
    health = workflow._path_health(records, [state] * (workflow.SHARD_COUNT + 1))
    assert health["passed"] == 1
    assert health["cutoff_identity"]["checked_shards"] == list(range(27, 64))

    records[27]["controller_diagnostics"][2]["score_squared_sum"] = 1.0e-12
    assert workflow._path_health(records, [state] * 65)["passed"] == 0


def test_real_fused_bank_dispatches_distinct_parent_and_candidate_models() -> None:
    class ConstantScore(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def score_prediction(self, inputs: workflow.ModelInputs) -> torch.Tensor:
            return self.score_prediction_prevalidated(inputs)

        def score_prediction_prevalidated(self, inputs: workflow.ModelInputs) -> torch.Tensor:
            return torch.full((inputs.batch_size, 392), self.value, dtype=torch.float64)

    specs = workflow._row_specs(workflow.EVAL_PATH_IDS[0], "a" * 64)
    bank = workflow.FusedTangentControllerBank(
        specs,
        {
            workflow.ROW_ORDER[1]: workflow.CompletedStepCutoffTangentScoreController(ConstantScore(1.0), 216),
            workflow.ROW_ORDER[2]: workflow.CompletedStepCutoffTangentScoreController(ConstantScore(2.0), 216),
            workflow.ROW_ORDER[3]: workflow.TargetFractionOracleController(_simplex(), microsteps=2),
        },
    )
    bank.prepare_device("cpu")
    scores = bank.score_prediction(_torch_inputs(4, simplex=True))
    assert torch.count_nonzero(scores[0]) == 0
    torch.testing.assert_close(scores[1], torch.ones(392, dtype=torch.float64))
    torch.testing.assert_close(scores[2], torch.full((392,), 2.0, dtype=torch.float64))


def test_npz_only_orphan_is_archived_byte_exactly_before_replay(tmp_path: Path) -> None:
    path_id = workflow.EVAL_PATH_IDS[0]
    shard = (
        workflow._eval_root(tmp_path, path_id)
        / "reverse/fused_families/same-path-four-row/complete-512/shard-0000.npz"
    )
    atomic_rollout_npz(shard, {"state": np.repeat(_simplex()[None, :], 4, axis=0)})
    digest = file_fingerprint(shard)
    recovered = workflow._recover_eval_orphan(tmp_path, path_id)
    assert recovered is not None and recovered.is_file() and not shard.exists()
    assert file_fingerprint(recovered) == digest

    atomic_rollout_npz(shard, {"state": np.repeat(_simplex(1.0e-5)[None, :], 4, axis=0)})
    assert shard.is_file() and recovered.is_file()


def test_objective_and_mechanism_diagnostics_use_the_prespecified_formulas() -> None:
    rows = _rows(frozen=0.9, candidate=0.5, oracle=0.2)
    assert workflow._gap_closure(rows) == pytest.approx((1.0 - 0.5) / (1.0 - 0.2))

    state = np.stack((_simplex(), _simplex(1.0e-5), _simplex(2.0e-5), _simplex(3.0e-5)))
    source = SimpleNamespace(source_image=_simplex(), mixed_target=_simplex(3.0e-5))
    metrics = workflow._metric_rows(workflow.EVAL_PATH_IDS[0], [state], source)
    candidate = next(row for row in metrics if row["row_key"] == workflow.ROW_ORDER[2])
    assert candidate["relative_improvement_over_frozen"] == pytest.approx(
        candidate["paired_improvement_over_frozen"]
        / next(row for row in metrics if row["row_key"] == workflow.ROW_ORDER[1])["target_squared_l2_error"]
    )
    delta = state[2] - state[1]
    assert candidate["state_vs_frozen_squared_l2"] == pytest.approx(float(np.dot(delta, delta)))

    dynamics = [{
        "reference_fraction_displacement_count": 4, "reference_fraction_displacement_squared_sum": 4.0,
        "control_fraction_displacement_count": 4, "control_fraction_displacement_squared_sum": 16.0,
    } for _ in workflow.ROW_ORDER]
    record = {"controller_diagnostics": [{} for _ in workflow.ROW_ORDER], "per_row_diagnostics": dynamics}
    mechanism = workflow._mechanism([record])
    assert mechanism["per_row"][2]["control_to_reference_rms_ratio"] == 2.0


def test_training_diagnostics_report_weighted_and_original_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyModel:
        def __init__(self, value: float) -> None:
            self.value = value

        def to(self, _device): return self
        def eval(self): return self
        def requires_grad_(self, _value): return self
        def load_state_dict(self, _state, strict=True): return None

    monkeypatch.setattr(workflow, "_load_checkpoint", lambda *_args, **_kwargs: DummyModel(1.0))
    monkeypatch.setattr(workflow, "GlobalDilatedZeroBaselinePredictor", lambda **_kwargs: DummyModel(2.0))
    evaluate = lambda model, values, _target, **_kwargs: (100.0 * model.value + values.batch_size, torch.empty(0))
    monkeypatch.setattr(workflow, "evaluate_model_mse", evaluate)
    reweighting = SimpleNamespace(record={"train": {"original_row_count": 2},
                                          "validation": {"original_row_count": 1}})
    result = workflow._training_diagnostics(
        tmp_path, {"update": 100, "state_sha256": "a" * 64, "state_dict": {}},
        _torch_inputs(4), torch.zeros((4, 392)), _torch_inputs(3), torch.zeros((3, 392)),
        reweighting, torch.device("cpu"),
    )
    assert result["splits"]["train"] == {
        "original_row_count": 2, "weighted_row_count": 4,
        "parent_weighted_mse": 104.0, "parent_original_unweighted_mse": 102.0,
        "candidate_weighted_mse": 204.0, "candidate_original_unweighted_mse": 202.0,
    }
    assert result["splits"]["validation"]["weighted_row_count"] == 3
    assert result["splits"]["validation"]["candidate_original_unweighted_mse"] == 201.0


def test_outcome_routes_gate_e_and_never_launches_compute() -> None:
    not_reviewed = workflow._human_template()
    cases = [
        (_summaries(), _review(), {"passed": 1}, "stage_f_plan", 1),
        (_summaries(), _review(), {"passed": 0}, "repair", 0),
        (_summaries(endpoint=(0.8, 0.4, 1.0, 0.6)), _review(), {"passed": 1}, "common_path_repair", 0),
        (_summaries(), not_reviewed, {"passed": 1}, "human_review_required", 0),
        (_summaries(endpoint=(0.95, 0.97, 0.1, 0.1)), _review("noise"),
         {"passed": 1}, "material_controller_comparison", 0),
        (_summaries(endpoint=(0.8, 0.9, 0.1, 0.1), boundary=(0.8, 0.9, 0.1, 0.1)),
         _review("noise"), {"passed": 1}, "conventional_ddpm", 0),
    ]
    for summaries, review, health, route, gate_e in cases:
        outcome = workflow._outcome(summaries, review, health)
        assert outcome["route"] == route
        assert outcome["gate_e_passed"] == gate_e
        assert outcome["confirmatory_claim"] == 0
        assert outcome["stage_f_machine_eligible"] == 0
        assert outcome["stage_f_automatically_launched"] == 0
        assert outcome["automatic_compute_launched"] == 0


def test_human_review_records_all_paths_and_refinalizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    (run / "evaluation").mkdir(parents=True)
    monkeypatch.setattr(workflow, "verify_run", lambda _run: {"outcome": {"human_review_completed": 0}})
    finalized = []
    def fake_finalize(root: Path) -> dict:
        finalized.append(json.loads((root / "evaluation/human_review.json").read_text()))
        return {"route": "conventional_ddpm"}
    monkeypatch.setattr(workflow, "_finalize", fake_finalize)
    labels = {path_id: "noise" for path_id in workflow.EVAL_PATH_IDS}
    assert workflow.record_human_review(run, labels, reviewer="human")["route"] == "conventional_ddpm"
    assert [row["path_id"] for row in finalized[0]["labels"]] == list(workflow.EVAL_PATH_IDS)
    assert finalized[0]["automated_recognizability"] == 0


def test_full_verifier_is_read_only_and_rejects_manifested_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    run = tmp_path / "run"
    summaries = _summaries()
    by_id = {row["path_id"]: row for row in summaries}
    state = np.full((4, 784), 1.0 / 784.0, dtype=np.float64)
    states = [state] * 65
    source = SimpleNamespace(source_image=_simplex(), mixed_target=_simplex(), metadata={"lambda_mix": 0.1})
    health = {"passed": 1}
    aggregate_health = {"passed": 1, "paths": [health] * 3}
    patches = {
        "_load_checkpoint": lambda *_args, **_kwargs: None,
        "_selected_payload": lambda *_args: {},
        "load_verified_source_target": lambda *_args: source,
        "_scan_eval": lambda *_args: ([], states),
        "_path_health": lambda *_args: health,
        "_metric_rows": lambda path_id, *_args: [{"path_id": path_id, "value": 1.0}],
        "_path_summary": lambda path_id, _rows: by_id[path_id],
        "_verify_source_files": lambda *_args: None,
        "_verify_copied_inputs": lambda *_args: None,
        "_validate_ledger": lambda *_args, **_kwargs: {"passed": 1, "active_seconds": 0.0},
        "_report_text": lambda *_args: "report\n",
    }
    for name, value in patches.items():
        monkeypatch.setattr(workflow, name, value)
    starts = {"path_ids": np.asarray(workflow.EVAL_PATH_IDS),
              "prior_seeds": np.asarray(workflow.EVAL_PRIOR_SEEDS),
              "states": np.stack([workflow._prior_state(seed)[0] for seed in workflow.EVAL_PRIOR_SEEDS])}
    trajectory = {"completed_reverse_steps": np.arange(0, 513, 8), "states": np.stack(states)}
    monkeypatch.setattr(workflow, "_npz", lambda path: starts if path.name == "evaluation_start_states.npz" else trajectory)
    atomic_write_json(run / "config.json", workflow._frozen_config())
    atomic_write_json(run / "bindings.json", {"copies": {}})
    atomic_write_json(run / "inputs/role_inventory.json", workflow._role_inventory())
    all_rows = []
    for path_id in workflow.EVAL_PATH_IDS:
        root = workflow._eval_root(run, path_id)
        rows = workflow._metric_rows(path_id, states, source)
        atomic_write_json(root / "health.json", health)
        atomic_write_json(root / "mechanism.json", workflow._mechanism([]))
        workflow.atomic_write_csv(root / "metrics.csv", rows)
        workflow._render_path(root, states, source)
        all_rows.extend(rows)
    assert len(list((run / "evaluation").rglob("*.png"))) == 3 * 92
    sample_root = workflow._eval_root(run, workflow.EVAL_PATH_IDS[0]) / "images"
    with Image.open(sample_root / "contact_sheets/raw-step-512.png") as image:
        assert image.size == (112, 28)
    with Image.open(sample_root / "individual/raw/zero/step-512.png") as image:
        assert image.size == (28, 28)
    review = _review()
    outcome = workflow._outcome(summaries, review, aggregate_health)
    atomic_write_json(run / "evaluation/aggregate_mechanism.json",
                      {"schema": workflow.VERSION + "-aggregate-mechanism", "path_summaries": summaries})
    workflow.atomic_write_csv(run / "evaluation/aggregate_metrics.csv", all_rows)
    atomic_write_json(run / "evaluation/human_review.json", review)
    atomic_write_json(run / "outcome.json", outcome)
    workflow._atomic_text(run / "REPORT.md", "report\n")
    atomic_write_json(run / "status.json",
                      {"state": "complete", "resumable": 0, "completed_evaluation_paths": 3, "active_seconds": 0.0})
    workflow._refresh_manifest(run)
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    assert workflow.verify_run(run)["passed"] == 1
    assert before == {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    workflow.save_png(workflow._eval_root(run, workflow.EVAL_PATH_IDS[0]) / "images/context/source.png",
                      np.zeros((28, 28), dtype=np.uint8))
    with pytest.raises(workflow.RolloutReweightedRunError, match="pixels changed"):
        workflow.verify_run(run)


def test_approval_and_initialization_reject_placeholders_and_external_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(workflow.RolloutReweightedRunError, match="real approval"):
        workflow._approval("<fresh-explicit-approval>")
    args = _args(tmp_path)
    for name in ("repository", "training-parent", "baseline", "stage-e", "runs"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(workflow, "discover_repository_path_id_claims", lambda _root: ())
    monkeypatch.setattr(workflow, "scan_path_id_collisions", lambda _ids, _claims: ("collision",))
    with pytest.raises(workflow.RolloutReweightedRunError, match="collide"):
        workflow._initialize_run(args)

    monkeypatch.setattr(workflow, "scan_path_id_collisions", lambda _ids, _claims: ())
    monkeypatch.setattr(workflow, "_committed_numerical_path_ids", lambda _root: set())
    capsule = {"steps": np.arange(0, 513, 8, dtype=np.int64),
               "selected_states": np.repeat(_simplex()[None, :], 65, axis=0)}
    monkeypatch.setattr(workflow, "_validate_stage_e_capsule", lambda _root: capsule)

    def fake_copy(_source: Path, destination: Path, _digest: str) -> dict:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x")
        return {"size": 1, "sha256": "0" * 64}

    monkeypatch.setattr(workflow, "_copy", fake_copy)
    run = workflow._initialize_run(args)
    bindings = json.loads((run / "bindings.json").read_text(encoding="utf-8"))
    assert all((run / item["path"]).is_file() for item in bindings["copies"].values())
