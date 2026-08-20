from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist import d0_jacobi_rb_cuda_deferred as deferred
from mnist import diag_d0_jacobi_rb_candidate_complete as workflow
from mnist.d0_jacobi_artifacts import atomic_write_json, file_fingerprint
from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE, ModelInputs, semantic_sha256
from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time
from mnist.d0_jacobi_rb_tangent_fused import fused_transition_ids
from mnist.d0_jacobi_rb_tangent_rollout import (
    atomic_rollout_npz,
    reverse_suffix_sequence,
    rollout_array_sha256,
    rollout_file_sha256,
    rollout_semantic_record,
    source_measure_sha256,
)


class _NonzeroScoreModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[ModelInputs] = []

    def _predict(self, inputs: ModelInputs) -> torch.Tensor:
        assert type(inputs) is ModelInputs
        self.seen.append(inputs)
        return torch.ones(
            (inputs.batch_size, EDGES_PER_PHASE),
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )

    def score_prediction(self, inputs: ModelInputs) -> torch.Tensor:
        return self._predict(inputs)

    def score_prediction_prevalidated(self, inputs: ModelInputs) -> torch.Tensor:
        return self._predict(inputs)


def _model_inputs(reverse_time: np.ndarray) -> ModelInputs:
    times = torch.as_tensor(reverse_time, dtype=torch.float64)
    batch = int(times.numel())
    return ModelInputs(
        later_full_state=torch.full((batch, workflow.STATE_SIZE), 1.0 / workflow.STATE_SIZE),
        reverse_time=times,
        phase=torch.zeros(batch, dtype=torch.int64),
        color=torch.zeros(batch, dtype=torch.int64),
        duration=torch.full((batch,), 0.5, dtype=torch.float64),
        label=torch.full((batch,), 3, dtype=torch.int64),
    )

def _args(reference: Path, runs_root: Path, name: str, *, stage_d: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        reference_run_dir=str(reference),
        runs_root=str(runs_root),
        run_name=name,
        device="cuda:0",
        maximum_active_seconds=1_000.0,
        stage_d_run_dir=None if stage_d is None else str(stage_d),
        approval_reference="fixture-explicit-cap-approval",
    )

def _candidate_state(
    start: np.ndarray, target: np.ndarray, committed_shards: int, *, helpful: bool
) -> np.ndarray:
    progress = committed_shards / workflow.SHARD_COUNT
    learned = (1.0 - 0.25 * progress) * start + 0.25 * progress * target
    oracle = (1.0 - 0.85 * progress) * start + 0.85 * progress * target
    if not helpful:
        learned = start
        oracle = start
    return np.ascontiguousarray(np.stack((start, learned, oracle)), dtype=np.float64)

def _make_reference(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, exact: str = "match"
) -> tuple[Path, np.ndarray, np.ndarray]:
    repository_root = Path(workflow.__file__).resolve().parents[1]
    source_files = {name: file_fingerprint(repository_root / name) for name in workflow._DIRECT_SOURCE_FILES}
    monkeypatch.setattr(workflow, "_module_bindings", lambda _root: {
        "repository_revision": "fixture", "dirty_status_sha256": None, "direct_source_files": source_files})
    reference = root / "reference"
    source = np.linspace(1.0, 3.0, workflow.STATE_SIZE, dtype=np.float64)
    source /= np.sum(source)
    mix = 0.2
    target = np.ascontiguousarray((1.0 - mix) * source + mix / workflow.STATE_SIZE)
    start = np.ascontiguousarray(source[::-1])

    checkpoint = reference / "inputs/model/update-3100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake-checkpoint-not-loaded-by-cpu-tests")
    source_npz = reference / "inputs/source/source_image.npz"
    atomic_rollout_npz(source_npz, {"image": source, "mixed_target": target})
    source_json = reference / "inputs/source/source_image.json"
    atomic_write_json(
        source_json,
        {
            "npz_sha256": file_fingerprint(source_npz),
            "npz_size": source_npz.stat().st_size,
            "image_sha256": source_measure_sha256(source),
            "mixed_target_sha256": source_measure_sha256(target),
            "lambda_mix": mix,
            "label": 3,
            "dataset_index": 7,
        },
    )
    anchor = reference / "forward/anchor-step-0511.npz"
    atomic_rollout_npz(anchor, {"state": start})

    monkeypatch.setattr(workflow, "CHECKPOINT_SHA256", file_fingerprint(checkpoint))
    monkeypatch.setattr(workflow, "SOURCE_JSON_SHA256", file_fingerprint(source_json))
    monkeypatch.setattr(workflow, "SOURCE_NPZ_SHA256", file_fingerprint(source_npz))
    monkeypatch.setattr(workflow, "SOURCE_ARRAY_SHA256", rollout_array_sha256(source))
    monkeypatch.setattr(workflow, "TARGET_ARRAY_SHA256", rollout_array_sha256(target))
    monkeypatch.setattr(workflow, "ANCHOR_SHA256", file_fingerprint(anchor))
    monkeypatch.setattr(workflow, "ANCHOR_STATE_SHA256", rollout_array_sha256(start))

    hashes: dict[str, str] = {}
    if exact != "missing":
        spec = workflow.experiment_spec("stage-d-anchor-v1")
        rows = workflow._row_specs(spec)
        sequence = reverse_suffix_sequence(511)
        previous = np.repeat(start[None, :], len(workflow.LEGACY_ROW_ORDER), axis=0)
        exact_root = reference / workflow.EXACT_PREFIX_RELATIVE
        for index in range(2):
            state = _candidate_state(start, target, index + 1, helpful=True)
            if exact == "mismatch":
                state = np.ascontiguousarray(np.roll(state, 31, axis=1))
            npz_path = exact_root / f"shard-{index:04d}.npz"
            atomic_rollout_npz(npz_path, {"state": state})
            lo = index * workflow.FUSED_SHARD_PHASES
            shard_sequence = sequence[lo : lo + workflow.FUSED_SHARD_PHASES]
            record = {
                "committed": 1,
                "shard_index": index,
                "row_keys": list(workflow.LEGACY_ROW_ORDER),
                "row_table": [row.to_record() for row in rows],
                "canonical_path_ids": [spec.path_id] * len(rows),
                "sequence_start": list(shard_sequence[0]),
                "sequence_end": list(shard_sequence[-1]),
                "sequence_sha256": semantic_sha256([list(item) for item in shard_sequence]),
                "label": spec.label,
                "microsteps": spec.microsteps,
                "input_state_sha256": rollout_array_sha256(previous),
                "output_state_sha256": rollout_array_sha256(state),
                "state_file_sha256": rollout_file_sha256(npz_path),
                "controller_binding_sha256": semantic_sha256(
                    workflow._controller_binding(spec)
                ),
                "rng_binding_sha256": semantic_sha256(workflow._rng_binding(spec)),
                "variant_in_rng_key": 0,
            }
            if exact == "wrong-witness":
                record["rng_binding_sha256"] = "f" * 64
            json_path = exact_root / f"shard-{index:04d}.json"
            atomic_write_json(json_path, record)
            hashes[json_path.name] = file_fingerprint(json_path)
            hashes[npz_path.name] = file_fingerprint(npz_path)
            previous = state
    monkeypatch.setattr(workflow, "EXACT_PREFIX_HASHES", hashes)
    return reference, start, target

def _initialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exact: str = "match", name: str = "candidate"
) -> tuple[Path, np.ndarray, np.ndarray]:
    reference, start, target = _make_reference(tmp_path, monkeypatch, exact=exact)
    run = workflow._initialize_run(
        _args(reference, tmp_path / "runs", name),
        workflow.experiment_spec("stage-d-anchor-v1"),
    )
    return run, start, target


def _make_schedule_baseline(
    root: Path, monkeypatch: pytest.MonkeyPatch,
    start: np.ndarray, target: np.ndarray,
) -> tuple[Path, np.ndarray]:
    baseline = root / "stage-d-baseline"
    spec = workflow.experiment_spec("stage-d-anchor-v1")
    states = np.ascontiguousarray(np.stack([
        _candidate_state(start, target, index, helpful=True)
        for index in range(workflow.SHARD_COUNT + 1)
    ]), dtype=np.float64)
    atomic_write_json(baseline / "config.json", {
        "experiment_name": spec.name, "spec": spec.to_record(),
        "row_order": list(workflow.LEGACY_ROW_ORDER),
        "row_table": [row.to_record() for row in workflow._row_specs(spec)],
        "controller_binding": workflow._controller_binding(spec),
        "rng_binding": workflow._rng_binding(spec),
        "start_state_sha256": workflow.ANCHOR_STATE_SHA256, "prior": None,
    })
    atomic_write_json(baseline / "outcome.json", {
        "run_state": "complete", "scientific_objective_completed": 1,
        "completed_reverse_steps": workflow.OUTER_STEPS, "health_passed": 1,
        "oracle_directionally_improves_zero": 1,
        "learned_directionally_improves_zero": 0,
        "primary_final_metrics": {
            "global-plus-1": {"paired_squared_l2_improvement_over_zero": -1.0},
            "source-informed": {"paired_squared_l2_improvement_over_zero": 1.0},
        },
    })
    atomic_rollout_npz(
        baseline / "reverse/trajectory_boundaries.npz",
        {
            "completed_reverse_steps": np.arange(
                0, workflow.OUTER_STEPS + workflow.SHARD_STEPS,
                workflow.SHARD_STEPS, dtype=np.int64,
            ),
            "states": states,
        },
    )
    atomic_write_json(baseline / "reverse/first16_audit.json", {"fixture": 1})
    atomic_write_json(baseline / "reverse/health.json", {"passed": 1})
    atomic_write_json(baseline / "bindings.json", {
        "source_image_sha256": workflow.SOURCE_ARRAY_SHA256,
        "mixed_target_sha256": workflow.TARGET_ARRAY_SHA256,
        "start_state_sha256": workflow.ANCHOR_STATE_SHA256,
    })
    workflow._refresh_manifest(baseline)
    hashes = {
        relative: file_fingerprint(baseline / relative)
        for relative in workflow.SCHEDULE_BASELINE_HASHES
    }
    monkeypatch.setattr(workflow, "SCHEDULE_BASELINE_HASHES", hashes)
    return baseline, states


def _make_stage_e_schedule_predecessor(
    root: Path, monkeypatch: pytest.MonkeyPatch,
    start: np.ndarray, target: np.ndarray,
) -> Path:
    predecessor = root / "stage-d-schedule-predecessor"
    spec = workflow.experiment_spec("stage-d-schedule-window-v1")
    atomic_write_json(predecessor / "config.json", {
        "experiment_name": spec.name, "spec": spec.to_record(),
        "row_order": list(workflow._row_order(spec)),
        "row_table": [row.to_record() for row in workflow._row_specs(spec)],
        "controller_binding": workflow._controller_binding(spec),
        "rng_binding": workflow._rng_binding(spec),
        "start_state_sha256": workflow.ANCHOR_STATE_SHA256, "prior": None,
        "research_mode": "exploratory", "confirmation_evidence_opened": 0,
    })
    atomic_write_json(predecessor / "bindings.json", {
        "source_image_sha256": workflow.SOURCE_ARRAY_SHA256,
        "mixed_target_sha256": workflow.TARGET_ARRAY_SHA256,
        "start_state_sha256": workflow.ANCHOR_STATE_SHA256,
    })
    atomic_write_json(predecessor / "outcome.json", _schedule_outcome(0.2, 0.3))
    atomic_write_json(predecessor / "reverse/health.json", {
        "passed": 1, "committed_shards": workflow.SHARD_COUNT,
        "completed_reverse_steps": workflow.OUTER_STEPS,
        "transition_count": (
            workflow.PER_ROW_SHARD_TRANSITIONS * 5 * workflow.SHARD_COUNT
        ),
        "forbidden_counts": {name: 0 for name in workflow._FORBIDDEN_COUNTS},
        "schedule_identity": {"passed": 1},
    })
    atomic_write_json(predecessor / "reverse/first16_audit.json", {
        "status": "available", "audit_complete": 1,
        "first16_not_completely_off": 1,
    })
    atomic_write_json(predecessor / "resource_ledger.json", {
        "cap_history": [{
            "approval_reference": workflow.STAGE_D_APPROVAL_REFERENCE_CAVEAT,
        }],
    })
    workflow._refresh_manifest(predecessor)
    hashes = {
        relative: file_fingerprint(predecessor / relative)
        for relative in workflow.STAGE_E_SCHEDULE_PREDECESSOR_HASHES
    }
    monkeypatch.setattr(workflow, "STAGE_E_SCHEDULE_PREDECESSOR_HASHES", hashes)
    return predecessor


def _write_chain(
    run: Path, start: np.ndarray, target: np.ndarray, count: int, *, helpful: bool = True
) -> list[Path]:
    config = workflow._read_json(run / "config.json")
    spec = workflow.experiment_spec(config["experiment_name"])
    row_order = workflow._row_order(spec)
    atomic_write_json(run / "backend.json", {
        "schema": workflow.VERSION + "-candidate-backend", "device": "cuda:0",
        "candidate_binary_sha256": "a" * 64, "candidate_modes": config["spec"]["candidate_modes"],
        "candidate_bisection_steps": config["spec"]["candidate_bisection_steps"],
        "exact_authorizer_authority": 0, "synchronous_replay_authority": 0,
    })
    sequence = reverse_suffix_sequence(511)
    previous = np.repeat(start[None, :], len(row_order), axis=0)
    paths: list[Path] = []
    root = workflow._shard_root(run)
    for index in range(count):
        legacy_state = _candidate_state(start, target, index + 1, helpful=helpful)
        if spec.name == "stage-d-schedule-window-v1":
            state = np.ascontiguousarray(legacy_state[[0, 1, 1, 1, 2]])
        elif spec.name == "stage-e-prior-cutoff-216-v1":
            state = np.ascontiguousarray(legacy_state[[0, 1, 1, 2]])
        else:
            state = legacy_state
        npz_path = root / f"shard-{index:04d}.npz"
        atomic_rollout_npz(npz_path, {"state": state})
        lo = index * workflow.FUSED_SHARD_PHASES
        shard_sequence = sequence[lo : lo + workflow.FUSED_SHARD_PHASES]
        active = workflow._shard_transition_count(spec)
        reference = {
            "reference_contract": workflow.CANDIDATE_REFERENCE_CONTRACT, "candidate_modes": 128, "candidate_bisection_steps": 56,
            "root_seed": config["spec"]["root_seed"], "stream_role": config["spec"]["stream_role"], "variant_in_rng_key": 0,
            "certificate_fraction": "not_applicable",
            "transition_count": active,
            "active_count": active,
            "structural_noop_count": 0,
            "approximation_count": active,
            "invalid_count": 0,
            "forbidden_counts": {name: 0 for name in workflow._FORBIDDEN_COUNTS},
            "maximum_candidate_bracket_width": 1.0e-16,
            "per_row": [{
                "transition_count": active // len(row_order), "active_count": active // len(row_order), "structural_noop_count": 0,
                "approximation_count": active // len(row_order), "invalid_count": 0,
                "maximum_candidate_bracket_width": 1.0e-16, "certificate_fraction": "not_applicable",
            } for _ in row_order],
        }
        per_row = [
            {
                "row_key": key,
                "score_count": 1,
                "score_squared_sum": 0.0,
                "score_maximum_absolute": 0.0,
                "logistic_shift_count": 1,
                "logistic_shift_squared_sum": 0.0,
                "logistic_shift_maximum_absolute": 0.0,
                "reference_fraction_displacement_count": 1,
                "reference_fraction_displacement_squared_sum": 1.0e-8,
                "reference_fraction_displacement_maximum_absolute": 1.0e-4,
                "control_fraction_displacement_count": 1,
                "control_fraction_displacement_squared_sum": 0.0,
                "control_fraction_displacement_maximum_absolute": 0.0,
            }
            for key in row_order
        ]
        controllers = [{
            "row_key": row.row_key, "controller_kind": row.controller_kind, "gain": row.gain,
            "score_squared_sum": 0.0, "score_maximum_absolute": 0.0,
            "unscaled_score_squared_sum": 0.0, "unscaled_score_maximum_absolute": 0.0,
            "clipping_count": 0, "floor_count": 0, "projection_count": 0, "nonfinite_score_count": 0,
            "target_oracle_unreachable_boundary_count": 0,
        } for row in workflow._row_specs(spec)]
        body = {
            "committed": 1, "schema": "d0-jacobi-rb-tangent-fused-v1-reverse-shard", "schema_version": 1,
            "scheduler_version": "d0-jacobi-rb-tangent-fused-v1",
            "family_name": workflow._family_name(spec), "segment_name": "complete-512",
            "shard_index": index,
            "reference_contract": workflow.CANDIDATE_REFERENCE_CONTRACT,
            "row_table": config["row_table"],
            "row_keys": list(row_order), "canonical_path_ids": [config["spec"]["path_id"]] * len(row_order),
            "sequence_start": list(shard_sequence[0]),
            "sequence_end": list(shard_sequence[-1]),
            "sequence_sha256": semantic_sha256([list(item) for item in shard_sequence]),
            "label": config["spec"]["label"],
            "microsteps": config["spec"]["microsteps"],
            "variant_in_rng_key": 0,
            "input_state_sha256": rollout_array_sha256(previous),
            "output_state_sha256": rollout_array_sha256(state),
            "state_file_sha256": rollout_file_sha256(npz_path), "state_file_size": npz_path.stat().st_size,
            "controller_binding_sha256": semantic_sha256(config["controller_binding"]),
            "rng_binding_sha256": semantic_sha256(config["rng_binding"]),
            "transition_count": active,
            "elapsed_seconds": 1.0,
            "synchronous_replay_performed": 0,
            "diagnostics": {"reference": reference, "maximum_mass_error": 0.0},
            "per_row_diagnostics": per_row,
            "controller_diagnostics": controllers,
        }
        json_path = root / f"shard-{index:04d}.json"
        atomic_write_json(json_path, rollout_semantic_record(body))
        paths.extend((json_path, npz_path))
        previous = state
    return paths

def _terminalize(run: Path, state: str = "resource_paused") -> None:
    workflow._terminalize_accounted(run, state, "fixture stop" if state != "complete" else None)

def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path): file_fingerprint(path) for path in paths}

def test_named_specs_and_cli_expose_only_frozen_science() -> None:
    anchor = workflow.experiment_spec("stage-d-anchor-v1")
    prior = workflow.experiment_spec("stage-e-prior-v1")
    assert anchor.start.kind == "forward_anchor"
    assert prior.start.kind == "dirichlet_prior" and prior.start.seed == 261_403
    assert anchor.render_horizons == (0, 8, 16, 128, 256, 384, 512)
    assert tuple(row.row_key for row in workflow._row_specs(anchor)) == workflow.LEGACY_ROW_ORDER
    assert workflow._row_specs(anchor)[1].gain == 1.0
    parser = workflow._parser()
    base = ["run-anchor", "--reference-run-dir", "ref", "--runs-root", "runs",
            "--run-name", "run", "--maximum-active-seconds", "1000",
            "--approval-reference", "user-approved-candidate-cap"]
    parsed = parser.parse_args(base)
    assert parsed.experiment_name == "stage-d-anchor-v1"
    assert parsed.approval_reference == "user-approved-candidate-cap"
    with pytest.raises(SystemExit):
        parser.parse_args([item for item in base if item != "--approval-reference"][:-1])
    for forbidden in ("--gain", "--root-seed", "--candidate-modes"):
        with pytest.raises(SystemExit):
            parser.parse_args(base + [forbidden, "2"])
    assert workflow._correlation(np.ones(8), np.ones(8)) == (
        1.0,
        "near_constant_exact_equality",
    )


def test_schedule_spec_and_cli_are_literal_and_legacy_is_unchanged() -> None:
    schedule = workflow.experiment_spec("stage-d-schedule-window-v1")
    legacy = workflow.experiment_spec("stage-d-anchor-v1")
    legacy_prior = workflow.experiment_spec("stage-e-prior-v1")
    prior_cutoff = workflow.experiment_spec("stage-e-prior-cutoff-216-v1")
    assert schedule.start == legacy.start
    assert (schedule.path_id, schedule.root_seed, schedule.stream_role) == (
        legacy.path_id, legacy.root_seed, legacy.stream_role,
    )
    assert workflow._row_order(schedule) == workflow.SCHEDULE_ROW_ORDER == (
        "zero", "global-plus-1", "global-cutoff-176",
        "global-cutoff-216", "source-informed",
    )
    assert workflow._row_order(legacy) == workflow.LEGACY_ROW_ORDER
    assert workflow._family_name(schedule) == "same-path-five-row"
    assert workflow._family_name(legacy) == "same-path-three-row"
    assert workflow._shard_transition_count(schedule) == 439_040
    assert workflow._shard_transition_count(legacy) == 263_424
    assert schedule.render_horizons == (0, 8, 16, 128, 176, 192, 216, 224, 256, 384, 512)
    rows = workflow._row_specs(schedule)
    assert [row.canonical_path_id for row in rows] == [1_028_864] * 5
    assert [row.gain for row in rows[1:4]] == [1.0, 1.0, 1.0]

    assert prior_cutoff.start == legacy_prior.start
    assert (
        prior_cutoff.path_id, prior_cutoff.root_seed, prior_cutoff.stream_role,
    ) == (legacy_prior.path_id, legacy_prior.root_seed, legacy_prior.stream_role)
    assert workflow._row_order(prior_cutoff) == workflow.STAGE_E_CUTOFF_ROW_ORDER == (
        "zero", "global-plus-1", "global-cutoff-216", "source-informed",
    )
    assert workflow._row_order(legacy_prior) == workflow.LEGACY_ROW_ORDER
    assert workflow._family_name(prior_cutoff) == "same-path-four-row"
    assert workflow._shard_transition_count(prior_cutoff) == 351_232
    assert prior_cutoff.render_horizons == workflow.STAGE_E_CUTOFF_RENDER_HORIZONS == (
        0, 8, 16, 128, 216, 224, 256, 384, 512,
    )
    prior_rows = workflow._row_specs(prior_cutoff)
    assert [row.canonical_path_id for row in prior_rows] == [1_028_865] * 4
    assert [prior_rows[index].gain for index in (1, 2)] == [1.0, 1.0]
    assert prior_rows[2].controller_binding["cutoff_completed_reverse_steps"] == 216
    assert workflow._rng_binding(prior_cutoff) == workflow._rng_binding(legacy_prior)
    transition_ids = fused_transition_ids(
        prior_rows, outer_step=511, phase=6, reverse_microstep=0,
        role="reverse_reference_pre_control_M2", device="cpu",
    )
    assert all(torch.equal(transition_ids[0], transition_ids[index]) for index in range(1, 4))

    parser = workflow._parser()
    command = [
        "run-schedule-window", "--reference-run-dir", "ref",
        "--stage-d-run-dir", "stage-d", "--runs-root", "runs",
        "--run-name", "schedule", "--maximum-active-seconds", "1800",
        "--approval-reference", "fresh-schedule-approval",
    ]
    parsed = parser.parse_args(command)
    assert parsed.experiment_name == "stage-d-schedule-window-v1"
    for forbidden in ("--gain", "--cutoff", "--rows", "--sweep", "--stage-e"):
        with pytest.raises(SystemExit):
            parser.parse_args(command + [forbidden, "1"])

    prior_command = [
        "run-prior-cutoff-216", "--reference-run-dir", "ref",
        "--stage-d-run-dir", "stage-d-schedule", "--runs-root", "runs",
        "--run-name", "prior-cutoff", "--maximum-active-seconds", "1200",
        "--approval-reference", "fresh-stage-e-cutoff-approval",
    ]
    parsed = parser.parse_args(prior_command)
    assert parsed.experiment_name == "stage-e-prior-cutoff-216-v1"
    for forbidden in ("--gain", "--cutoff", "--rows", "--sweep", "--root-seed"):
        with pytest.raises(SystemExit):
            parser.parse_args(prior_command + [forbidden, "1"])


@pytest.mark.parametrize(
    ("cutoff", "first_active_k", "last_inactive_k"),
    ((176, 336, 335), (216, 296, 295)),
)
def test_completed_step_cutoff_masks_every_midpoint_without_changing_model_inputs(
    cutoff: int, first_active_k: int, last_inactive_k: int
) -> None:
    coordinates = [
        (k, phase, q)
        for k in range(511, -1, -1)
        for phase in range(6, -1, -1)
        for q in (0.25, 0.75)
    ]
    times = np.asarray(
        [internal_reverse_time(k, phase, q) for k, phase, q in coordinates],
        dtype=np.float64,
    )
    expected = np.asarray([k >= first_active_k for k, _phase, _q in coordinates])
    assert not np.any(times == cutoff / workflow.OUTER_STEPS)
    assert np.array_equal(times < cutoff / workflow.OUTER_STEPS, expected)
    assert all(expected[index] for index, row in enumerate(coordinates) if row[0] == first_active_k)
    assert not any(expected[index] for index, row in enumerate(coordinates) if row[0] == last_inactive_k)

    model = _NonzeroScoreModel()
    controller = workflow.CompletedStepCutoffTangentScoreController(model, cutoff)
    inputs = _model_inputs(times)
    deferred = controller.score_prediction_deferred(inputs)
    mask = torch.as_tensor(expected).reshape(-1, 1)
    assert controller.gain == 1.0
    assert torch.equal(deferred != 0.0, mask.expand_as(deferred))
    assert torch.equal(controller._last_deferred_unscaled != 0.0, mask.expand_as(deferred))
    assert len(model.seen) == 1 and model.seen[0] is inputs

    endpoint_inputs = _model_inputs(np.asarray([times[0], times[-1]]))
    ordinary = controller.score_prediction(endpoint_inputs)
    endpoint_mask = endpoint_inputs.reverse_time < cutoff / workflow.OUTER_STEPS
    assert torch.equal(ordinary != 0.0, endpoint_mask.reshape(-1, 1).expand_as(ordinary))
    assert model.seen[-1] is endpoint_inputs


def test_stage_e_cutoff_build_family_dispatches_only_cutoff216_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "stage-e-build-family"
    spec = workflow.experiment_spec("stage-e-prior-cutoff-216-v1")
    atomic_write_json(run / "config.json", {"experiment_name": spec.name})
    model = _NonzeroScoreModel()
    source = SimpleNamespace(
        mixed_target=np.full(workflow.STATE_SIZE, 1.0 / workflow.STATE_SIZE),
    )
    prepared = SimpleNamespace(
        device=torch.device("cpu"), candidate_binary_sha256="b" * 64,
    )
    monkeypatch.setattr(workflow, "_validate_core_inputs", lambda _run: (source, np.empty(0)))
    monkeypatch.setattr(workflow, "_load_model", lambda _run, _device: model)
    monkeypatch.setattr(
        deferred, "prepare_alpha1_rb_transition_batch_cuda_candidate",
        lambda **_kwargs: prepared,
    )
    monkeypatch.setattr(
        workflow, "prepare_deferred_reference_rng_seed_map", lambda **_kwargs: {},
    )

    built_spec, _source, rows, bank, _factory = workflow._build_family(
        run, torch.device("cpu"),
    )
    assert built_spec == spec and tuple(row.row_key for row in rows) == (
        "zero", "global-plus-1", "global-cutoff-216", "source-informed",
    )
    assert set(bank.controllers) == {
        "global-plus-1", "global-cutoff-216", "source-informed",
    }
    cutoff = bank.controllers["global-cutoff-216"]
    assert isinstance(cutoff, workflow.CompletedStepCutoffTangentScoreController)
    assert cutoff.cutoff_completed_reverse_steps == 216
    assert cutoff.base_controller is model


def test_schedule_identity_covers_partial_and_observed_defects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reference, start, target = _make_reference(tmp_path, monkeypatch, exact="missing")
    baseline = np.stack([
        _candidate_state(start, target, index, helpful=True)
        for index in range(workflow.SHARD_COUNT + 1)
    ])
    run = tmp_path / "identity"
    atomic_write_json(run / "config.json", {"experiment_name": "stage-d-schedule-window-v1"})
    atomic_write_json(run / "bindings.json", {"stage_d_schedule_baseline": {}})
    monkeypatch.setattr(workflow, "_validate_schedule_baseline_copy", lambda *_args: baseline)
    states = [np.ascontiguousarray(row[[0, 1, 1, 1, 2]]) for row in baseline[:29]]
    records = [{
        "controller_diagnostics": [{
            name: (key if name == "row_key" else 0.0)
            for name in ("row_key", "score_squared_sum", "score_maximum_absolute",
                         "unscaled_score_squared_sum", "unscaled_score_maximum_absolute")
        } for key in workflow.SCHEDULE_ROW_ORDER],
        "per_row_diagnostics": [{
            "row_key": key, "logistic_shift_squared_sum": 0.0,
            "logistic_shift_maximum_absolute": 0.0,
        } for key in workflow.SCHEDULE_ROW_ORDER],
    } for _ in range(28)]
    assert workflow._schedule_identity(run, records, states)["passed"] == 1
    partial = workflow._schedule_identity(run, records[:10], states[:11])
    assert all(row["post_cutoff_status"] == "not_reached" for row in partial["cutoff_rows"].values())
    defects = (("state", 1, 2), ("score", 22, 2), ("shift", 27, 3))
    for kind, index, row in defects:
        changed_records, changed_states = copy.deepcopy(records), [value.copy() for value in states]
        if kind == "state": changed_states[index][row] = np.roll(changed_states[index][row], 1)
        elif kind == "score": changed_records[index]["controller_diagnostics"][row]["score_squared_sum"] = 1.0
        else: changed_records[index]["per_row_diagnostics"][row]["logistic_shift_maximum_absolute"] = 1.0
        assert workflow._schedule_identity(run, changed_records, changed_states)["passed"] == 0


def test_stage_e_cutoff_identity_covers_partial_complete_and_observed_defects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reference, start, target = _make_reference(tmp_path, monkeypatch, exact="missing")
    run = tmp_path / "stage-e-cutoff-identity"
    atomic_write_json(run / "config.json", {
        "experiment_name": "stage-e-prior-cutoff-216-v1",
    })
    rows = workflow.STAGE_E_CUTOFF_ROW_ORDER
    states = []
    for index in range(workflow.SHARD_COUNT + 1):
        legacy = _candidate_state(start, target, index, helpful=True)
        state = np.ascontiguousarray(legacy[[0, 1, 1, 2]])
        if index * workflow.SHARD_STEPS > 216:
            state[2] = legacy[0]
        states.append(state)
    records = [{
        "controller_diagnostics": [{
            name: (key if name == "row_key" else 0.0)
            for name in (
                "row_key", "score_squared_sum", "score_maximum_absolute",
                "unscaled_score_squared_sum", "unscaled_score_maximum_absolute",
            )
        } for key in rows],
        "per_row_diagnostics": [{
            "row_key": key, "logistic_shift_squared_sum": 0.0,
            "logistic_shift_maximum_absolute": 0.0,
        } for key in rows],
    } for _ in range(workflow.SHARD_COUNT)]

    partial = workflow._schedule_identity(run, records[:10], states[:11])
    cutoff = partial["cutoff_rows"]["global-cutoff-216"]
    assert partial["passed"] == 1 and cutoff["post_cutoff_status"] == "not_reached"
    complete = workflow._schedule_identity(run, records, states)
    cutoff = complete["cutoff_rows"]["global-cutoff-216"]
    assert complete["passed"] == 1
    assert cutoff["pre_cutoff_checked_boundaries"][-1] == 216
    assert cutoff["post_cutoff_first_shard_index"] == 27
    assert cutoff["post_cutoff_checked_shards"] == list(range(27, 64))

    changed_states = [state.copy() for state in states]
    changed_states[27][2] = np.roll(changed_states[27][2], 1)
    assert workflow._schedule_identity(run, records, changed_states)["passed"] == 0
    changed_records = copy.deepcopy(records)
    changed_records[27]["controller_diagnostics"][2]["score_squared_sum"] = 1.0
    assert workflow._schedule_identity(run, changed_records, states)["passed"] == 0


def test_schedule_complete_artifacts_render_and_verify_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, start, target = _make_reference(tmp_path, monkeypatch, exact="match")
    baseline, _baseline_states = _make_schedule_baseline(
        tmp_path, monkeypatch, start, target,
    )
    original_verify, original_manifest = workflow.verify_run, workflow._verify_manifest
    manifest_checks: list[Path] = []
    monkeypatch.setattr(workflow, "verify_run", lambda _path: (_ for _ in ()).throw(
        AssertionError("schedule initialization called full legacy verify_run")))
    monkeypatch.setattr(workflow, "_verify_manifest", lambda path: (
        manifest_checks.append(Path(path)), original_manifest(path))[1])
    run = workflow._initialize_run(
        _args(reference, tmp_path / "runs", "schedule-complete", stage_d=baseline),
        workflow.experiment_spec("stage-d-schedule-window-v1"),
    )
    assert manifest_checks == [baseline.resolve()]
    monkeypatch.setattr(workflow, "verify_run", original_verify)
    monkeypatch.setattr(workflow, "_verify_manifest", original_manifest)
    bindings = workflow._read_json(run / "bindings.json")
    assert bindings["stage_d_predecessor"] is None
    shutil.rmtree(baseline)
    workflow._validate_schedule_baseline_copy(run, bindings["stage_d_schedule_baseline"])
    _write_chain(run, start, target, workflow.SHARD_COUNT)
    _terminalize(run, "complete")
    before = _snapshot([path for path in run.rglob("*") if path.is_file()])
    result = workflow.verify_run(run)
    assert result["passed"] == 1
    assert _snapshot([path for path in run.rglob("*") if path.is_file()]) == before

    trajectory = workflow._npz(run / "reverse/trajectory_boundaries.npz")
    assert trajectory["states"].shape == (65, 5, workflow.STATE_SIZE)
    with (run / "reverse/metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 325
    from PIL import Image
    with Image.open(run / "images/contact_sheets/raw-step-512.png") as image:
        assert image.size == (140, 28)
    actual_horizons = {
        int(path.stem.rsplit("-", 1)[1])
        for path in (run / "images/contact_sheets").glob("raw-step-*.png")
    }
    assert actual_horizons == set(workflow.SCHEDULE_RENDER_HORIZONS)
    outcome = workflow._read_json(run / "outcome.json")
    assert outcome["selected_cutoff_completed_reverse_steps"] == 176
    assert outcome["stage_e_machine_eligible"] == 0
    audit = workflow._read_json(run / "reverse/first16_audit.json")
    assert [row["exact_row_key"] for row in audit["horizons"][0]["rows"]] == [
        "zero", "global-plus-1", "global-plus-1", "global-plus-1", "source-informed",
    ]
    assert set(audit["horizons"][0]["learned_minus_zero_contrasts"]) == {
        "global-plus-1", "global-cutoff-176", "global-cutoff-216",
    }

    copied_config = run / workflow.SCHEDULE_BASELINE_COPY_ROOT / "config.json"
    copied_config.write_bytes(copied_config.read_bytes() + b"tamper")
    workflow._refresh_manifest(run)
    with pytest.raises(workflow.CandidateRunError, match="copied authority changed"):
        workflow.verify_run(run)


def test_candidate_prepare_loads_only_proposal_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[str] = []
    def load(device: object, _profile: object) -> tuple[object, str]:
        calls.append(str(device)); return sentinel, "b" * 64
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("candidate preparation touched exact authority")
    monkeypatch.setattr(deferred, "_load_cuda_kernel", load)
    monkeypatch.setattr(deferred, "probe_fused_cuda_authorizer", forbidden)
    monkeypatch.setattr(deferred, "launch_fused_cuda_authorizer", forbidden)
    monkeypatch.setattr(deferred, "sample_alpha1_rb_transition_batch_cuda", forbidden)
    prepared = deferred.prepare_alpha1_rb_transition_batch_cuda_candidate(
        device="cuda:7", profile=workflow.JacobiRBCudaProfile()
    )
    assert isinstance(prepared, deferred.PreparedCandidateRBCudaBackend)
    assert calls == ["cuda:7"] and prepared.candidate_kernel is sentinel
    assert prepared.candidate_binary_sha256 == "b" * 64
    assert not hasattr(prepared, "fused_bundle") and not hasattr(prepared, "fused_report")


def test_stage_e_requires_verified_machine_eligible_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, _start, _target = _make_reference(tmp_path, monkeypatch, exact="missing")
    spec = workflow.experiment_spec("stage-e-prior-v1")
    with pytest.raises(workflow.CandidateRunError, match="requires --stage-d-run-dir"):
        workflow._initialize_run(_args(reference, tmp_path, "no-parent"), spec)
    parent = tmp_path / "stage-d"; parent.mkdir()
    atomic_write_json(parent / "outcome.json", {"scientific_objective_completed": 1, "stage_e_machine_eligible": 1})
    atomic_write_json(parent / "reverse/first16_audit.json", {"first16_not_completely_off": 1})
    atomic_write_json(parent / "artifact_manifest.json", {"artifacts": [{"path": relative, "sha256": file_fingerprint(parent / relative)} for relative in ("outcome.json", "reverse/first16_audit.json")]})
    monkeypatch.setattr(workflow, "verify_run", lambda _path: {"outcome": {"stage_e_machine_eligible": 0}})
    with pytest.raises(workflow.CandidateRunError, match="not machine-eligible"):
        workflow._initialize_run(
            _args(reference, tmp_path, "bad-parent", stage_d=parent), spec
        )
    monkeypatch.setattr(workflow, "verify_run", lambda _path: {"outcome": {"stage_e_machine_eligible": 1}})
    run = workflow._initialize_run(
        _args(reference, tmp_path, "good-parent", stage_d=parent), spec
    )
    state = workflow._npz(run / "inputs/start_state.npz")["state"]
    assert np.array_equal(state, workflow._prior_state(261_403)[0])
    assert float(np.sum(state)) == pytest.approx(1.0, abs=2.0e-15)
    bindings = workflow._read_json(run / "bindings.json"); assert bindings["exact_prefix"]["status"] == "not_applicable"
    assert bindings["stage_d_predecessor"] == {"run_dir": str(parent.resolve()), "manifest_sha256": file_fingerprint(parent / "artifact_manifest.json"), "outcome_sha256": file_fingerprint(parent / "outcome.json"), "first16_audit_sha256": file_fingerprint(parent / "reverse/first16_audit.json")}


def test_stage_e_cutoff_predecessor_is_copied_external_independent_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, start, target = _make_reference(tmp_path, monkeypatch, exact="missing")
    predecessor = _make_stage_e_schedule_predecessor(
        tmp_path, monkeypatch, start, target,
    )
    spec = workflow.experiment_spec("stage-e-prior-cutoff-216-v1")
    with pytest.raises(workflow.CandidateRunError, match="disjoint|overlap"):
        workflow._initialize_run(
            _args(reference, predecessor, "overlap", stage_d=predecessor), spec,
        )

    original_manifest = workflow._verify_manifest
    manifest_checks: list[Path] = []
    monkeypatch.setattr(
        workflow, "verify_run",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("Stage E initialization called predecessor verify_run")
        ),
    )
    monkeypatch.setattr(workflow, "_verify_manifest", lambda path: (
        manifest_checks.append(Path(path)), original_manifest(path),
    )[1])
    run = workflow._initialize_run(
        _args(reference, tmp_path / "runs", "stage-e-cutoff", stage_d=predecessor),
        spec,
    )
    assert manifest_checks == [predecessor.resolve()]
    bindings = workflow._read_json(run / "bindings.json")
    authority = bindings["stage_d_schedule_predecessor"]
    assert bindings["stage_d_predecessor"] is None
    assert authority["external_predecessor_required_after_initialization"] == 0
    assert authority["predecessor_approval_reference_caveat"] == "<fresh-approval-reference>"
    assert len(authority["files"]) == 6
    assert {claim["source_path"] for claim in authority["files"]} == set(
        workflow.STAGE_E_SCHEDULE_PREDECESSOR_HASHES
    )
    for claim in authority["files"]:
        copied = run / claim["path"]
        assert copied.is_file() and not copied.is_symlink()
        assert copied.stat().st_nlink == 1
        assert file_fingerprint(copied) == claim["sha256"]
    prior = workflow._prior_state(261_403)[0]
    assert np.array_equal(workflow._npz(run / "inputs/start_state.npz")["state"], prior)
    assert bindings["exact_prefix"]["status"] == "not_applicable"

    shutil.rmtree(predecessor)
    workflow._validate_run_authority(run)
    calls: list[Path] = []
    monkeypatch.setattr(
        workflow, "_run_or_resume",
        lambda path, _device: calls.append(Path(path)) or 0,
    )
    assert workflow.main(["resume", "--run-dir", str(run)]) == 0
    assert calls == [run.resolve()]

    copied_config = (
        run / workflow.STAGE_E_SCHEDULE_PREDECESSOR_COPY_ROOT / "config.json"
    )
    copied_config.write_bytes(copied_config.read_bytes() + b"tamper")
    with pytest.raises(workflow.CandidateRunError, match="copied authority changed"):
        workflow._validate_run_authority(run)


@pytest.mark.parametrize(
    ("relative", "field", "replacement"),
    (
        ("outcome.json", "selected_cutoff_completed_reverse_steps", 176),
        ("reverse/health.json", "passed", 0),
        ("reverse/first16_audit.json", "first16_not_completely_off", 0),
    ),
)
def test_stage_e_cutoff_predecessor_rejects_semantic_drift_under_rebound_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    relative: str, field: str, replacement: object,
) -> None:
    _reference, start, target = _make_reference(tmp_path, monkeypatch, exact="missing")
    predecessor = _make_stage_e_schedule_predecessor(
        tmp_path, monkeypatch, start, target,
    )
    path = predecessor / relative
    value = workflow._read_json(path); value[field] = replacement
    atomic_write_json(path, value)
    workflow._refresh_manifest(predecessor)
    hashes = {
        name: file_fingerprint(predecessor / name)
        for name in workflow.STAGE_E_SCHEDULE_PREDECESSOR_HASHES
    }
    monkeypatch.setattr(workflow, "STAGE_E_SCHEDULE_PREDECESSOR_HASHES", hashes)
    with pytest.raises(workflow.CandidateRunError, match="semantics changed"):
        workflow._validate_stage_e_predecessor_source(predecessor)


def test_stage_e_cutoff_complete_artifacts_render_manifest_and_verify_cpu_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, start, target = _make_reference(tmp_path, monkeypatch, exact="missing")
    predecessor = _make_stage_e_schedule_predecessor(
        tmp_path, monkeypatch, start, target,
    )
    run = workflow._initialize_run(
        _args(reference, tmp_path / "runs", "stage-e-cutoff-complete", stage_d=predecessor),
        workflow.experiment_spec("stage-e-prior-cutoff-216-v1"),
    )
    shutil.rmtree(predecessor)
    _write_chain(run, workflow._npz(run / "inputs/start_state.npz")["state"], target, workflow.SHARD_COUNT)

    def no_cuda(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Stage E CPU artifact/verification path touched CUDA")

    monkeypatch.setattr(workflow.torch.cuda, "is_available", no_cuda)
    _terminalize(run, "complete")
    before = _snapshot([path for path in run.rglob("*") if path.is_file()])
    verified = workflow.verify_run(run)
    assert verified["passed"] == 1 and verified["committed_shards"] == 64
    assert _snapshot([path for path in run.rglob("*") if path.is_file()]) == before

    trajectory = workflow._npz(run / "reverse/trajectory_boundaries.npz")
    assert trajectory["states"].shape == (65, 4, workflow.STATE_SIZE)
    assert np.array_equal(
        trajectory["completed_reverse_steps"],
        np.arange(0, 513, 8, dtype=np.int64),
    )
    with (run / "reverse/metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 260
    from PIL import Image
    with Image.open(run / "images/contact_sheets/raw-step-512.png") as image:
        assert image.size == (112, 28)
    assert len(list((run / "images").rglob("*.png"))) == 92
    assert {
        int(path.stem.rsplit("-", 1)[1])
        for path in (run / "images/contact_sheets").glob("raw-step-*.png")
    } == set(workflow.STAGE_E_CUTOFF_RENDER_HORIZONS)

    health = workflow._read_json(run / "reverse/health.json")
    outcome = workflow._read_json(run / "outcome.json")
    audit = workflow._read_json(run / "reverse/first16_audit.json")
    manifest = workflow._read_json(run / "artifact_manifest.json")
    assert health["passed"] == 1 and health["transition_count"] == 22_478_848
    assert health["schedule_identity"]["passed"] == 1
    assert outcome["scientific_objective_completed"] == 1
    assert outcome["stage_f_machine_eligible"] == outcome["stage_f_automatically_launched"] == 0
    assert audit["status"] == "unavailable" and audit["blocks_artifact_completion"] == 0
    assert manifest["artifact_count"] == len(manifest["artifacts"]) == 244
    assert len(list(workflow._shard_root(run).glob("shard-*.json"))) == 64
    assert len(list(workflow._shard_root(run).glob("shard-*.npz"))) == 64
    report = (run / "REPORT.md").read_text(encoding="utf-8")
    assert "<fresh-approval-reference>" in report
    assert "Stage F machine eligibility: 0" in report


def test_stage_e_outcome_routes_to_stage_f_without_exact_audit() -> None:
    metrics = [{"completed_reverse_steps": 512, "row_key": key, "paired_squared_l2_improvement_over_zero": int(key != "zero"), "relative_paired_squared_l2_improvement_over_zero": 0.02} for key in workflow.LEGACY_ROW_ORDER]
    outcome = workflow._outcome_record([{}] * workflow.SHARD_COUNT, metrics, {"passed": 1}, {"first16_not_completely_off": None}, "complete", None, "stage-e-prior-v1")
    assert (outcome["stage_e_machine_eligible"], outcome["stage_f_machine_eligible"], outcome["stage_f_automatically_launched"]) == (0, 1, 0) and "Stage F" in outcome["next_action"]
    failed = workflow._outcome_record([{}] * workflow.SHARD_COUNT, metrics, {"passed": 1}, {"first16_not_completely_off": None}, "failed", "post-commit failure", "stage-e-prior-v1")
    assert (failed["scientific_objective_completed"], failed["stage_f_machine_eligible"]) == (0, 0)


def _schedule_outcome(
    cutoff_176: float, cutoff_216: float, *, health: int = 1,
    audit: int = 1, oracle: float = 1.0, earlier: bool = False,
    complete: bool = True,
) -> dict[str, object]:
    completed = 512 if complete else 8
    values = {"zero": 0.0, "global-plus-1": -1.0,
              "global-cutoff-176": cutoff_176, "global-cutoff-216": cutoff_216,
              "source-informed": oracle}
    metrics = ([{"completed_reverse_steps": 128, "row_key": "global-cutoff-176",
                 "paired_squared_l2_improvement_over_zero": 0.1,
                 "relative_paired_squared_l2_improvement_over_zero": 0.01}]
               if earlier else [])
    metrics.extend({"completed_reverse_steps": completed, "row_key": key,
                    "paired_squared_l2_improvement_over_zero": values[key],
                    "relative_paired_squared_l2_improvement_over_zero": values[key] / 10.0}
                   for key in workflow.SCHEDULE_ROW_ORDER)
    return workflow._outcome_record(
        [{}] * (workflow.SHARD_COUNT if complete else 1), metrics,
        {"passed": health, "schedule_identity": {"passed": health}},
        {"status": "available" if complete else "not_reached",
         "first16_not_completely_off": audit if complete else None},
        "complete" if complete else "resource_paused",
         None if complete else "fixture pause", "stage-d-schedule-window-v1")


def _stage_e_cutoff_outcome(
    *, cutoff_error: float = 0.7, always_error: float = 0.8,
    oracle_error: float = 0.2, health: int = 1,
    state: str = "complete",
) -> dict[str, object]:
    completed = workflow.OUTER_STEPS if state == "complete" else workflow.SHARD_STEPS
    errors = {
        "zero": 1.0, "global-plus-1": always_error,
        "global-cutoff-216": cutoff_error, "source-informed": oracle_error,
    }
    metrics = [{
        "completed_reverse_steps": completed, "row_key": key,
        "mixed_target_squared_l2_error": error,
        "paired_squared_l2_improvement_over_zero": 1.0 - error,
        "relative_paired_squared_l2_improvement_over_zero": 1.0 - error,
        "mixed_target_centered_contrast_correlation": 0.9,
    } for key, error in errors.items()]
    return workflow._outcome_record(
        [{}] * (workflow.SHARD_COUNT if state == "complete" else 1), metrics,
        {"passed": health, "schedule_identity": {"passed": health}},
        {"status": "unavailable", "first16_not_completely_off": None},
        state, None if state == "complete" else "fixture stop",
        "stage-e-prior-cutoff-216-v1",
    )


@pytest.mark.parametrize(("left", "right", "selected"), (
    (0.2, -0.1, 176), (-0.1, 0.2, 216), (0.3, 0.2, 176),
    (0.2, 0.3, 216), (0.2, 0.2, 176), (0.001, -0.1, 176), (0.0, -0.1, None),
))
def test_schedule_outcome_uses_literal_selection_and_never_enables_follow_on(
    left: float, right: float, selected: int | None
) -> None:
    outcome = _schedule_outcome(left, right)
    assert outcome["selected_cutoff_completed_reverse_steps"] == selected
    assert outcome["selected_cutoff_ready_for_review"] == int(selected is not None)
    assert all(outcome[key] == 0 for key in (
        "stage_e_machine_eligible", "stage_e_automatically_launched",
        "stage_f_machine_eligible", "stage_f_automatically_launched",
        "confirmatory_claim",
    ))
    if left == 0.001:
        assert outcome["cutoff_terminal_comparisons"]["176"]["exceeds_one_percent_marker"] == 0


@pytest.mark.parametrize(
    ("health", "audit", "oracle", "earlier", "expected"),
    (
        (0, 1, 1.0, False, "localized numerical, schedule, or pairing defect"),
        (1, 0, 1.0, False, "audit the candidate kernel"),
        (1, 1, 0.0, False, "repair the composition, backend, or oracle interface"),
        (1, 1, 1.0, True, "stop nearby cutoff tuning"),
        (1, 1, 1.0, False, "change learner scale or representation"),
    ),
)
def test_schedule_negative_routes_are_decision_distinct(
    health: int, audit: int, oracle: float, earlier: bool, expected: str
) -> None:
    outcome = _schedule_outcome(-0.1, -0.1, health=health, audit=audit,
                                oracle=oracle, earlier=earlier)
    assert expected in outcome["next_action"]
    assert outcome["selected_cutoff_completed_reverse_steps"] is None


def test_schedule_partial_run_never_selects_a_cutoff() -> None:
    outcome = _schedule_outcome(1.0, 1.0, complete=False)
    assert outcome["selected_cutoff_completed_reverse_steps"] is None
    assert outcome["scientific_objective_completed"] == 0
    assert "resume this run" in outcome["next_action"]


@pytest.mark.parametrize(
    ("kwargs", "expected_action", "objective"),
    (
        ({"health": 0}, "localized integrity", 0),
        ({"oracle_error": 0.995}, "prior, oracle, or composition", 1),
        ({"cutoff_error": 0.9, "always_error": 0.8}, "stop static-cutoff", 1),
        ({}, "endpoint recognizability review", 1),
        ({"state": "resource_paused"}, "resume this run", 0),
        ({"state": "interrupted"}, "resume this unchanged run", 0),
    ),
)
def test_stage_e_cutoff_outcome_branches_and_permanently_disables_auto_stage_f(
    kwargs: dict[str, object], expected_action: str, objective: int,
) -> None:
    outcome = _stage_e_cutoff_outcome(**kwargs)
    assert expected_action in str(outcome["next_action"])
    assert outcome["scientific_objective_completed"] == objective
    assert all(outcome[key] == 0 for key in (
        "stage_e_machine_eligible", "stage_e_automatically_launched",
        "stage_f_machine_eligible", "stage_f_automatically_launched",
        "confirmatory_claim",
    ))
    assert outcome["one_percent_marker_type"] == "diagnostic_threshold"
    assert outcome["human_endpoint_recognizability"] == "not_automated"


def test_terminalization_invokes_read_only_verifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _start, _target = _initialize(tmp_path, monkeypatch, exact="missing"); calls: list[Path] = []
    monkeypatch.setattr(workflow, "verify_run", lambda path: calls.append(Path(path)) or {"passed": 1})
    workflow._terminalize_accounted(run, "resource_paused", "fixture stop")
    assert calls == [run]


def test_derived_failure_resumes_terminalization_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="missing"); _write_chain(run, start, target, 2)
    original_verify = workflow.verify_run
    monkeypatch.setattr(workflow, "verify_run", lambda _path: (_ for _ in ()).throw(workflow.CandidateRunError("injected verifier failure")))
    with pytest.raises(workflow.CandidateRunError, match="injected verifier failure"):
        workflow._terminalize_accounted(run, "resource_paused", "fixture stop")
    status = workflow._read_json(run / "status.json")
    assert (status["state"], status["terminalization_target_state"]) == ("derived_failed", "resource_paused")
    ledger = workflow._read_json(run / "resource_ledger.json")
    ledger["events"].append({"id": "fixture-terminal-pressure", "role": "fixture", "elapsed_seconds": 900.0, "failed": 0})
    ledger["active_seconds"] = workflow.math.fsum(float(row["elapsed_seconds"]) for row in ledger["events"]); atomic_write_json(run / "resource_ledger.json", ledger)
    monkeypatch.setattr(workflow, "verify_run", original_verify)
    monkeypatch.setattr(workflow, "_execute", lambda *_args: (_ for _ in ()).throw(AssertionError("derived recovery touched CUDA execution")))
    with pytest.raises(workflow.CandidateRunError, match="explicit larger cap"):
        workflow.main(["resume", "--run-dir", str(run)])
    assert workflow.main(["resume", "--run-dir", str(run), "--extend-maximum-active-seconds", "2000", "--cap-amendment-reason", "approved derived recovery"]) == 2
    assert workflow._read_json(run / "status.json")["state"] == "resource_paused"
    assert original_verify(run)["passed"] == 1

@pytest.mark.parametrize(("state", "complete", "audit_ok", "oracle", "learned", "expected"), (
    ("resource_paused", False, 1, True, (0.0,), "resume this run only after an explicit cap-extension approval; reuse every committed shard"),
    ("failed", False, 1, True, (0.0,), "fix the candidate execution or integrity defect before continuing this objective"),
    ("complete", True, 0, True, (1.0,), "repair or audit the candidate kernel on the frozen first-16 prefix before Stage E"),
    ("complete", True, 1, False, (1.0,), "repair the oracle, schedule, or composition interface before changing the learner"),
    ("complete", True, 1, True, (0.1, -0.1), "target rollout or on-policy training, or revise the late application schedule"),
    ("complete", True, 1, True, (0.0, -0.1), "make one material learner or controller change; do not buy more exact provenance"),
))
def test_stage_d_actions_are_decision_distinct(state: str, complete: bool, audit_ok: int, oracle: bool, learned: tuple[float, ...], expected: str) -> None:
    metrics = [{"row_key": "global-plus-1", "paired_squared_l2_improvement_over_zero": value} for value in learned]
    action = workflow._next_action(experiment_name="stage-d-anchor-v1", run_state=state, complete=complete, audit={"status": "available", "first16_not_completely_off": audit_ok}, oracle_positive=oracle, learned_positive=learned[-1] > 0.0, metric_rows=metrics)
    assert action == expected

@pytest.mark.parametrize("exact", ("missing", "wrong-witness"))
def test_stage_d_optional_exact_unavailable_or_incompatible_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exact: str
) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact=exact)
    binding = workflow._read_json(run / "bindings.json")["exact_prefix"]
    assert binding["status"] == "unavailable"
    assert isinstance(binding["reason"], str) and binding["reason"]
    assert binding["files"] == []
    _write_chain(run, start, target, 2)
    _terminalize(run)
    audit = workflow._read_json(run / "reverse/first16_audit.json")
    assert audit["status"] == "unavailable"
    assert audit["reason"] == binding["reason"]
    assert audit["first16_not_completely_off"] is None
    assert (run / "images/contact_sheets/raw-step-016.png").is_file()


def test_step8_exact_audit_is_saved_on_a_one_shard_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="match"); _write_chain(run, start, target, 1); _terminalize(run)
    audit = workflow._read_json(run / "reverse/first16_audit.json")
    assert audit["status"] == "available" and [row["completed_reverse_steps"] for row in audit["horizons"]] == [8]
    assert audit["audit_complete"] == 0 and audit["first16_not_completely_off"] is None


def test_candidate_chain_rejects_an_exact_state_splice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="mismatch")
    _write_chain(run, start, target, 2)
    records, _states = workflow._scan_prefix(run)
    assert len(records) == 2
    exact = workflow.load_rollout_state_npz(
        run / "inputs/exact_prefix/shard-0000.npz", expected_rows=3
    )
    path = workflow._shard_root(run) / "shard-0001.json"
    record = workflow._read_json(path)
    body = {key: value for key, value in record.items() if key != "semantic_sha256"}
    body["input_state_sha256"] = rollout_array_sha256(exact)
    atomic_write_json(path, rollout_semantic_record(body))
    with pytest.raises(workflow.CandidateRunError, match="binding changed"):
        workflow._scan_prefix(run)


@pytest.mark.parametrize("field", ("order", "clipping_count", "floor_count", "projection_count", "nonfinite_score_count"))
def test_controller_diagnostics_are_ordered_and_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="missing"); _write_chain(run, start, target, 1)
    path = workflow._shard_root(run) / "shard-0000.json"; record = workflow._read_json(path)
    body = {key: value for key, value in record.items() if key != "semantic_sha256"}
    if field == "order": body["controller_diagnostics"].reverse()
    else: body["controller_diagnostics"][1][field] = 1
    atomic_write_json(path, rollout_semantic_record(body))
    with pytest.raises(workflow.CandidateRunError): workflow._scan_prefix(run)


def test_first16_uses_vector_learned_minus_zero_contrast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = np.linspace(1.0, 2.0, workflow.STATE_SIZE, dtype=np.float64)
    target /= np.sum(target)
    exact = np.repeat(target[None, :], 3, axis=0)
    candidate = exact.copy()
    epsilon = 1.0e-5
    exact[1, 0] += epsilon
    exact[1, 1] -= epsilon
    candidate[1, 2] += epsilon
    candidate[1, 3] -= epsilon
    run = tmp_path / "run"
    atomic_write_json(run / "config.json", {"experiment_name": "stage-d-anchor-v1"})
    atomic_write_json(run / "bindings.json", {"exact_prefix": {"status": "available"}})
    start = np.repeat(target[None, :], 3, axis=0)
    common = {
        "row_table": [], "row_keys": list(workflow.LEGACY_ROW_ORDER),
        "controller_binding_sha256": "a", "rng_binding_sha256": "b",
        "sequence_start": [511, 6], "sequence_end": [504, 0],
        "sequence_sha256": "c", "label": 3, "microsteps": 2,
        "variant_in_rng_key": 0, "input_state_sha256": rollout_array_sha256(start),
    }
    monkeypatch.setattr(
        workflow, "_validate_exact_prefix",
        lambda _root: [{"record": dict(common), "state": exact}, {"record": dict(common), "state": exact}],
    )
    monkeypatch.setattr(workflow, "_scan_prefix", lambda _run: ([dict(common), dict(common)], []))
    source = SimpleNamespace(mixed_target=target)
    audit = workflow._first16_audit(
        run,
        [start, candidate, candidate],
        source,
    )
    relative = audit["horizons"][0]["learned_minus_zero_contrast_relative_error"]
    assert relative > 0.25
    assert audit["first16_not_completely_off"] == 0


def test_diagnostics_never_veto_complete_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="match")
    _write_chain(run, start, target, workflow.SHARD_COUNT, helpful=False)
    _terminalize(run, "complete")
    outcome = workflow._read_json(run / "outcome.json")
    audit = workflow._read_json(run / "reverse/first16_audit.json")
    assert outcome["scientific_objective_completed"] == 1
    assert outcome["stage_e_machine_eligible"] == 0
    assert audit["first16_not_completely_off"] == 0
    assert (run / "images/contact_sheets/raw-step-512.png").is_file()
    assert workflow.verify_run(run)["passed"] == 1

def test_resume_rebuilds_images_without_stale_prior_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="missing"); _write_chain(run, start, target, 3); _terminalize(run)
    stale = run / "images/contact_sheets/raw-step-024.png"; assert stale.is_file()
    _write_chain(run, start, target, 4); _terminalize(run)
    assert not stale.exists() and (run / "images/contact_sheets/raw-step-032.png").is_file()


def test_source_binding_inventory_deletion_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _start, _target = _initialize(tmp_path, monkeypatch, exact="missing", name="source-bound")
    workflow._validate_run_authority(run); bindings = workflow._read_json(run / "bindings.json")
    bindings["source"]["direct_source_files"].pop(next(iter(workflow._DIRECT_SOURCE_FILES)))
    atomic_write_json(run / "bindings.json", bindings)
    with pytest.raises(workflow.CandidateRunError): workflow._validate_run_authority(run)

def test_projection_bootstraps_two_then_prices_remaining_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, *_ = _initialize(tmp_path, monkeypatch, exact="missing"); initial = workflow._projection(run, workflow.time.perf_counter(), []); measured = workflow._projection(run, workflow.time.perf_counter(), [{"elapsed_seconds": 10.0}, {"elapsed_seconds": 20.0}])
    assert (initial["priced_remaining_shards"], measured["priced_remaining_shards"]) == (2, workflow.SHARD_COUNT - 2)


@pytest.mark.parametrize("approval", ("<fresh-approval-reference>", "  <placeholder>  "))
def test_placeholder_shaped_fresh_approval_is_rejected(
    tmp_path: Path, approval: str,
) -> None:
    with pytest.raises(workflow.CandidateRunError):
        workflow._new_ledger(1_200.0, approval)
    ledger = workflow._new_ledger(1_200.0, "real explicit approval")
    ledger["cap_history"][0]["approval_reference"] = approval
    atomic_write_json(tmp_path / "resource_ledger.json", ledger)
    with pytest.raises(workflow.CandidateRunError, match="cap history"):
        workflow._validate_ledger(tmp_path)


def test_storage_stop_is_not_mislabeled_as_time_cap_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, *_ = _initialize(tmp_path, monkeypatch, exact="missing")
    monkeypatch.setattr(workflow, "_directory_bytes", lambda _run: workflow.STORAGE_CAP_BYTES)
    monkeypatch.setattr(workflow.torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("storage stop touched CUDA")))
    with pytest.raises(workflow.CandidateRunError, match="storage reached"):
        workflow._execute(run, "cuda:0")


def test_resource_resume_reconciles_attempt_and_amends_cap_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="missing")
    shard_paths = _write_chain(run, start, target, 2)
    root = workflow._shard_root(run)
    atomic_rollout_npz(root / "shard-0002.npz", {"state": _candidate_state(start, target, 3, helpful=True)})
    atomic_write_json(root / "shard-0002.failure.json", {"failure_type": "ResourcePause"})
    _terminalize(run)
    committed_before = _snapshot(shard_paths)
    ledger = workflow._read_json(run / "resource_ledger.json")
    old_cap = float(ledger["maximum_active_seconds"])
    before = (run / "resource_ledger.json").read_bytes()
    with pytest.raises(workflow.CandidateRunError, match="larger cap"):
        workflow._amend_cap(run, old_cap, "not larger")
    assert (run / "resource_ledger.json").read_bytes() == before
    with pytest.raises(workflow.CandidateRunError):
        workflow._amend_cap(run, old_cap + 500.0, "<cap-approval-placeholder>")
    assert (run / "resource_ledger.json").read_bytes() == before

    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(workflow, "_run_or_resume", lambda path, device: calls.append((Path(path), device)) or 0)
    assert workflow.main(
        [
            "resume", "--run-dir", str(run), "--device", "cuda:0",
            "--extend-maximum-active-seconds", str(old_cap + 500.0),
            "--cap-amendment-reason", "explicit fixture extension",
        ]
    ) == 0
    assert calls == [(run.resolve(), "cuda:0")]
    assert _snapshot(shard_paths) == committed_before
    amended = workflow._read_json(run / "resource_ledger.json")
    assert amended["maximum_active_seconds"] == old_cap + 500.0
    assert amended["cap_history"][-1]["reason"] == "explicit_cap_amendment"
    assert amended["cap_history"][-1]["approval_reference"] == "explicit fixture extension"
    workflow._recover_uncommitted(run)
    assert not (root / "shard-0002.npz").exists()
    assert not (root / "shard-0002.failure.json").exists()
    assert len(list((run / "failures").glob("*shard-0002*"))) == 2


def test_running_hard_crash_reconciles_without_cap_amendment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _start, _target = _initialize(tmp_path, monkeypatch, exact="missing")
    workflow._begin_attempt(run, "candidate_execution")
    status = workflow._read_json(run / "status.json"); status["state"] = "running"
    atomic_write_json(run / "status.json", status)
    assert not (run / "artifact_manifest.json").exists()
    calls: list[Path] = []
    monkeypatch.setattr(workflow, "_run_or_resume", lambda path, _device: calls.append(Path(path)) or 0)
    assert workflow.main(["resume", "--run-dir", str(run)]) == 0
    assert calls == [run.resolve()]
    ledger = workflow._read_json(run / "resource_ledger.json")
    assert ledger["active_attempt"] is None
    assert ledger["events"][-1]["role"] == "candidate_execution_interrupted"


def test_reconciled_cap_exhaustion_pauses_before_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _start, _target = _initialize(tmp_path, monkeypatch, exact="missing"); workflow._begin_attempt(run, "candidate_execution")
    ledger = workflow._read_json(run / "resource_ledger.json"); ledger["active_attempt"]["started_at"] = "2000-01-01T00:00:00+00:00"; atomic_write_json(run / "resource_ledger.json", ledger)
    status = workflow._read_json(run / "status.json"); status["state"] = "running"; atomic_write_json(run / "status.json", status)
    monkeypatch.setattr(workflow, "_run_or_resume", lambda *_args: (_ for _ in ()).throw(AssertionError("cap-exhausted resume touched CUDA execution")))
    assert workflow.main(["resume", "--run-dir", str(run)]) == 2
    assert workflow._read_json(run / "status.json")["state"] == "resource_paused"


@pytest.mark.parametrize(
    ("relative", "field", "replacement"),
    (
        ("reverse/health.json", "passed", 0),
        ("reverse/first16_audit.json", "first16_not_completely_off", 1),
        ("reverse/mechanism.json", "first_negative_learned_improvement_horizon", 999),
        ("outcome.json", "stage_e_machine_eligible", 1),
    ),
)
def test_verify_recomputes_derived_authorities_after_manifest_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    field: str,
    replacement: object,
) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="missing")
    _write_chain(run, start, target, 2)
    _terminalize(run)

    def no_cuda(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("read-only verification touched CUDA")

    monkeypatch.setattr(workflow.torch.cuda, "is_available", no_cuda)
    before = _snapshot([path for path in run.rglob("*") if path.is_file()])
    assert workflow.verify_run(run)["passed"] == 1
    assert _snapshot([path for path in run.rglob("*") if path.is_file()]) == before
    path = run / relative
    value = workflow._read_json(path)
    value[field] = replacement
    atomic_write_json(path, value)
    workflow._refresh_manifest(run)
    with pytest.raises(workflow.CandidateRunError):
        workflow.verify_run(run)


def test_verify_rejects_a_rebound_truncated_metrics_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, start, target = _initialize(tmp_path, monkeypatch, exact="missing"); _write_chain(run, start, target, 2); _terminalize(run)
    path = run / "reverse/metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    removed = list(rows[0])[-1]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[name for name in rows[0] if name != removed], lineterminator="\n"); writer.writeheader(); writer.writerows([{key: value for key, value in row.items() if key != removed} for row in rows])
    workflow._refresh_manifest(run)
    with pytest.raises(workflow.CandidateRunError, match="derived metrics changed"):
        workflow.verify_run(run)


def test_verify_rejects_a_rebound_report_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _start, _target = _initialize(tmp_path, monkeypatch, exact="missing"); _terminalize(run)
    report = run / "REPORT.md"; report.write_text(report.read_text(encoding="utf-8").replace("global-plus-1: mixed-target", "global-plus-1: FABRICATED mixed-target"), encoding="utf-8")
    workflow._refresh_manifest(run)
    with pytest.raises(workflow.CandidateRunError, match="report authority changed"):
        workflow.verify_run(run)


def test_nested_manifest_name_is_not_excluded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _start, _target = _initialize(tmp_path, monkeypatch, exact="missing"); _terminalize(run)
    nested = run / "reverse/artifact_manifest.json"; atomic_write_json(nested, {"nested": 1}); manifest = workflow._refresh_manifest(run)
    assert "reverse/artifact_manifest.json" in {row["path"] for row in manifest["artifacts"]}
    assert workflow.verify_run(run)["passed"] == 1
    atomic_write_json(nested, {"nested": 2})
    with pytest.raises(workflow.CandidateRunError, match="artifact changed"):
        workflow.verify_run(run)


def test_manifest_rejects_a_generated_hardlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _start, _target = _initialize(tmp_path, monkeypatch, exact="missing"); source = tmp_path / "outside.bin"; source.write_bytes(b"linked")
    (run / "reverse").mkdir()
    os.link(source, run / "reverse/linked.bin")
    with pytest.raises(workflow.CandidateRunError, match="independent regular file"):
        workflow._refresh_manifest(run)
