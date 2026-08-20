from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mnist import d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability as workflow
from mnist import d0_jacobi_rb_boundary_tangent_frequency1_coordinate_gate as gate
from mnist.d0_jacobi_artifacts import atomic_write_json


def test_frozen_paths_seeds_family_and_test_cohorts() -> None:
    production = workflow.build_path_plan()
    assert production["roles"]["preflight_seam"] == list(range(0xF8000, 0xF8008))
    assert production["roles"]["training"] == list(range(0xF8100, 0xF8140))
    assert production["roles"]["validation"] == list(range(0xF8200, 0xF8220))
    assert production["roles"]["confirmation"] == list(range(0xF9000, 0xF9040))
    seeds = workflow.seed_plan()
    assert seeds["root_physical_path_seed"] == 261371
    assert seeds["physical_model_seeds"] == [261372, 261373, 261374]
    assert len(workflow.FAMILY_NAMES) == 228
    assert len(workflow.SEARCH_FAMILY_NAMES) == 27_360

    fixture = workflow.build_path_plan(test_only=True, test_path_count=8)
    cohorts = workflow.eager_cohorts(workflow.build_cohort_plan(fixture), "train_validation")
    assert max(len(row.path_ids) for row in cohorts) <= 10
    assert {role for row in cohorts for role in row.path_roles} == {"train", "validation"}


def _table(center: float, candidate_count: int = 2) -> workflow.Frequency1CandidateTable:
    paths = np.arange(0x1A00, 0x1A08, dtype=np.int64)
    path = np.arange(8, dtype=np.float64)[:, None, None]
    component = np.arange(228, dtype=np.float64)[None, None, :]
    candidate = np.arange(candidate_count, dtype=np.float64)[None, :, None]
    values = center + 0.01 * np.sin(path + component * 0.03 + candidate)
    return workflow.build_candidate_table(
        seeds=np.asarray(workflow.MODEL_SEEDS[:candidate_count], dtype=np.int64),
        updates=np.ones(candidate_count, dtype=np.int64),
        path_ids=paths,
        path_values=np.ascontiguousarray(values, dtype=np.float64),
        forbidden_path_ids=np.arange(0x1B00, 0x1B08, dtype=np.int64),
    )


def test_restartable_selection_is_seed_aware_and_immutable(tmp_path: Path) -> None:
    table = _table(2.0)
    workflow.prepare_bootstrap_count_shards(
        tmp_path / "counts",
        seed=workflow.SELECTION_BOOTSTRAP_SEED,
        namespace=workflow.SELECTION_NAMESPACE,
        path_count=8,
        replicates=8,
        shard_size=8,
    )
    result, ranking = workflow.restartable_selection_max_t(
        table,
        count_directory=tmp_path / "counts",
        maxima_directory=tmp_path / "maxima",
        replicates=8,
        shard_size=8,
    )
    assert result.lower_bounds.shape == (2, 228)
    assert ranking["decision"] == "frequency1_coordinate_validation_nominee_sealed"
    assert ranking["selected_seed"] in workflow.MODEL_SEEDS
    result2, ranking2 = workflow.restartable_selection_max_t(
        table,
        count_directory=tmp_path / "counts",
        maxima_directory=tmp_path / "maxima",
        replicates=8,
        shard_size=8,
    )
    np.testing.assert_array_equal(result.lower_bounds, result2.lower_bounds)
    assert ranking["maxima_sha256"] == ranking2["maxima_sha256"]

    data = next((tmp_path / "counts").glob("*.npz"))
    with np.load(data, allow_pickle=False) as archive:
        counts = np.array(archive["counts"], copy=True)
    counts[0, 0] += 1
    with data.open("wb") as handle:
        np.savez(handle, counts=counts)
    with pytest.raises(Exception):
        workflow.restartable_selection_max_t(
            table,
            count_directory=tmp_path / "counts",
            maxima_directory=tmp_path / "maxima",
            replicates=8,
            shard_size=8,
        )


def test_candidate_firewall_ranking_and_negative_gate(tmp_path: Path) -> None:
    with pytest.raises(workflow.Frequency1CoordinateWorkflowError):
        workflow.build_candidate_table(
            seeds=np.asarray([workflow.MODEL_SEEDS[0]], dtype=np.int64),
            updates=np.asarray([100], dtype=np.int64),
            path_ids=np.arange(8, dtype=np.int64),
            path_values=np.zeros((8, 1, 228), dtype=np.float64),
            forbidden_path_ids=np.asarray([3], dtype=np.int64),
        )
    table = _table(-2.0, candidate_count=1)
    workflow.prepare_bootstrap_count_shards(
        tmp_path / "negative-counts",
        seed=workflow.SELECTION_BOOTSTRAP_SEED,
        namespace=workflow.SELECTION_NAMESPACE,
        path_count=8,
        replicates=8,
        shard_size=8,
    )
    _result, ranking = workflow.restartable_selection_max_t(
        table,
        count_directory=tmp_path / "negative-counts",
        maxima_directory=tmp_path / "negative-maxima",
        replicates=8,
        shard_size=8,
    )
    assert ranking["decision"] == "no_frequency1_coordinate_validation_candidate"
    metrics = {
        "evaluation_status": "evaluated",
        **{name: 1 for name in gate.SELECT_INTEGRITY_FLAGS},
        "all_228_simultaneous_lower_bounds_positive": 0,
        "no_validation_candidate": 1,
        "stage_execution_valid": 1,
        "inference_valid": 1,
    }
    record = gate.evaluate_select_gate(metrics)
    assert record["passed"] == 0
    assert record["valid_scientific_negative"] == 1
    assert record["confirmation_authorized"] == 0


def test_stage_seals_detect_mutation_and_role_order(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "preflight_gate.json", {"evaluation_status": "evaluated", "passed": 1})
    atomic_write_json(tmp_path / "evidence.json", {"value": 1})
    workflow.seal_artifacts(
        tmp_path,
        ["preflight_gate.json", "evidence.json"],
        "preflight_artifact_seal.json",
    )
    workflow.validate_stage_entry(tmp_path, "cache")
    atomic_write_json(tmp_path / "evidence.json", {"value": 2})
    with pytest.raises(workflow.Frequency1CoordinateWorkflowError):
        workflow.verify_artifact_seal(tmp_path, "preflight_artifact_seal.json")


def test_safety_scope_never_authorizes_sampling() -> None:
    positive = gate.decide_workflow(
        preflight_gate=gate.evaluate_preflight_gate(
            {"evaluation_status": "evaluated", **{name: 1 for name in gate.PREFLIGHT_FLAGS}}
        ),
        cache_gate=gate.evaluate_cache_gate(
            {"evaluation_status": "evaluated", **{name: 1 for name in gate.CACHE_FLAGS}}
        ),
        controls_gate=gate.evaluate_controls_gate(
            {"evaluation_status": "evaluated", **{name: 1 for name in gate.CONTROLS_FLAGS}}
        ),
        train_gate=gate.evaluate_train_gate(
            {"evaluation_status": "evaluated", **{name: 1 for name in gate.TRAIN_FLAGS}}
        ),
        select_gate=gate.evaluate_select_gate(
            {"evaluation_status": "evaluated", **{name: 1 for name in gate.SELECT_FLAGS}}
        ),
        confirm_gate=gate.evaluate_confirm_gate(
            {"evaluation_status": "evaluated", **{name: 1 for name in gate.CONFIRM_FLAGS}}
        ),
    )
    assert positive["decision"] == gate.FINAL_DECISION
    assert positive["controller_control_patch_planning_authorized"] == 1
    for name in (
        "controller_execution_authorized",
        "reconstruction_authorized",
        "reverse_sampling_authorized",
        "sampling_authorized",
    ):
        assert positive[name] == 0
