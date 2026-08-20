from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_artifacts import atomic_write_json
from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE, STATE_SIZE, ModelInputs
from mnist.d0_jacobi_rb_tangent_rollout import (
    atomic_rollout_npz,
    fixed_rendering_scale,
    rollout_array_sha256,
    rollout_file_sha256,
)
from mnist import diag_d0_jacobi_rb_global_dilated_rollout as workflow


def _semantic(path: Path, body: dict[str, object]) -> dict[str, object]:
    record = workflow._semantic(body)
    atomic_write_json(path, record)
    return record


def _source_target() -> SimpleNamespace:
    source = np.zeros(STATE_SIZE, dtype=np.float64)
    source[0] = 1.0
    mixed = 0.65 * source + 0.35 / STATE_SIZE
    return SimpleNamespace(
        source_image=source,
        mixed_target=np.ascontiguousarray(mixed),
        metadata={"lambda_mix": 0.35, "label": 3, "dataset_index": 7},
    )


def _write_recovery_anchor_fixture(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    anchor_root = root / "input_bindings"
    step_0127 = np.ascontiguousarray(
        np.linspace(1.0, 2.0, STATE_SIZE, dtype=np.float64)
    )
    step_0127 /= np.sum(step_0127)
    step_0511 = np.ascontiguousarray(step_0127[::-1])
    archive = anchor_root / "recovery_anchors.npz"
    atomic_rollout_npz(
        archive,
        {"step_0127": step_0127, "step_0511": step_0511},
    )
    binding = _semantic(
        anchor_root / "recovery_anchor_binding.json",
        {
            "schema": "fixture-recovery-anchor-binding",
            "schema_version": 1,
            "file_sha256": rollout_file_sha256(archive),
            "file_size": archive.stat().st_size,
            "array_sha256": {
                "step_0127": rollout_array_sha256(step_0127.reshape(1, STATE_SIZE)),
                "step_0511": rollout_array_sha256(step_0511.reshape(1, STATE_SIZE)),
            },
            "passed": 1,
        },
    )
    return step_0127, step_0511, binding


def _model_inputs(batch: int = 2) -> ModelInputs:
    return ModelInputs(
        later_full_state=torch.full((batch, STATE_SIZE), 1.0 / STATE_SIZE),
        reverse_time=torch.full((batch,), 0.5, dtype=torch.float64),
        phase=torch.full((batch,), 0, dtype=torch.long),
        color=torch.full((batch,), 0, dtype=torch.long),
        duration=torch.full((batch,), 0.5),
        label=torch.full((batch,), 3, dtype=torch.long),
    )


def test_preflight_anchor_uses_sealed_legacy_one_row_hash_convention(
    tmp_path: Path,
) -> None:
    step_0127, step_0511, binding = _write_recovery_anchor_fixture(tmp_path)
    hashes = binding["array_sha256"]
    assert isinstance(hashes, dict)
    # The immutable producer stored each member as `[784]`, but deliberately
    # committed both through `_core_array_sha256(value.reshape(1, 784))`.
    assert rollout_array_sha256(step_0127) != hashes["step_0127"]
    assert rollout_array_sha256(step_0511) != hashes["step_0511"]
    assert (
        rollout_array_sha256(step_0127.reshape(1, STATE_SIZE))
        == hashes["step_0127"]
    )
    assert (
        rollout_array_sha256(step_0511.reshape(1, STATE_SIZE))
        == hashes["step_0511"]
    )
    loaded = workflow._load_preflight_anchor(
        SimpleNamespace(v4_run_dir=tmp_path)
    )
    assert loaded.shape == (STATE_SIZE,)
    assert loaded.dtype == np.float64
    assert np.array_equal(loaded, step_0127)


def test_preflight_anchor_rejects_content_tamper_even_if_file_is_rebound(
    tmp_path: Path,
) -> None:
    step_0127, step_0511, binding = _write_recovery_anchor_fixture(tmp_path)
    tampered = step_0127.copy()
    tampered[0], tampered[1] = tampered[1], tampered[0]
    archive = tmp_path / "input_bindings/recovery_anchors.npz"
    atomic_rollout_npz(
        archive,
        {"step_0127": tampered, "step_0511": step_0511},
    )
    # Rebind only the container bytes.  The immutable legacy array commitment
    # must still reject changed numerical content.
    body = {key: value for key, value in binding.items() if key != "semantic_sha256"}
    body["file_sha256"] = rollout_file_sha256(archive)
    body["file_size"] = archive.stat().st_size
    _semantic(tmp_path / "input_bindings/recovery_anchor_binding.json", body)
    with pytest.raises(
        workflow.GlobalDilatedRolloutError,
        match="step-127 development anchor hash changed",
    ):
        workflow._load_preflight_anchor(SimpleNamespace(v4_run_dir=tmp_path))


def test_preflight_anchor_rejects_archive_file_tamper(tmp_path: Path) -> None:
    _write_recovery_anchor_fixture(tmp_path)
    archive = tmp_path / "input_bindings/recovery_anchors.npz"
    with archive.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(
        workflow.GlobalDilatedRolloutError,
        match="development anchor archive changed",
    ):
        workflow._load_preflight_anchor(SimpleNamespace(v4_run_dir=tmp_path))


def test_wrapped_training_loss_uses_forward_m_not_score_prediction() -> None:
    class WrappedSpy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, inputs: ModelInputs) -> Tensor:
            self.forward_calls += 1
            return torch.full(
                (inputs.batch_size, EDGES_PER_PHASE), 0.25, dtype=torch.float64
            )

        def score_prediction(self, _inputs: ModelInputs) -> Tensor:  # pragma: no cover
            raise AssertionError("training must not call raw q")

    model = WrappedSpy()
    target = torch.full((2, EDGES_PER_PHASE), 0.125, dtype=torch.float64)
    loss, raw, prediction = workflow._wrapped_training_loss(
        model, _model_inputs(), target, 0.5
    )
    assert model.forward_calls == 1
    assert torch.equal(prediction, torch.full_like(target, 0.25))
    assert float(raw) == pytest.approx(0.125**2, abs=0.0)
    assert float(loss) == pytest.approx((0.125**2) / (0.5**2), abs=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA restore regression")
def test_cuda_mapped_progress_restores_cpu_and_cuda_rng_states_exactly(
    tmp_path: Path,
) -> None:
    original_cpu = torch.get_rng_state().clone()
    original_cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    try:
        torch.manual_seed(101)
        torch.cuda.manual_seed_all(202)
        expected_cpu = torch.get_rng_state().clone()
        expected_cuda = tuple(
            state.clone() for state in torch.cuda.get_rng_state_all()
        )
        progress_path = tmp_path / "progress.pt"
        torch.save(
            {
                "torch_rng_state": expected_cpu,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_rng_states": expected_cuda,
            },
            progress_path,
        )
        saved = torch.load(
            progress_path, map_location=torch.device("cuda"), weights_only=False
        )
        assert saved["torch_rng_state"].device.type == "cuda"
        assert all(state.device.type == "cuda" for state in saved["cuda_rng_states"])

        torch.manual_seed(303)
        torch.cuda.manual_seed_all(404)
        workflow._restore_training_rng_state(saved, device=torch.device("cuda"))

        assert torch.equal(torch.get_rng_state(), expected_cpu)
        restored_cuda = torch.cuda.get_rng_state_all()
        assert len(restored_cuda) == len(expected_cuda)
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(restored_cuda, expected_cuda, strict=True)
        )
    finally:
        torch.set_rng_state(original_cpu)
        torch.cuda.set_rng_state_all(list(original_cuda))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA restore regression")
def test_malformed_rng_payload_is_rejected_before_any_default_rng_mutation() -> None:
    original_cpu = torch.get_rng_state().clone()
    original_cuda = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    device_count = torch.cuda.device_count()

    def assert_defaults_unchanged() -> None:
        assert torch.equal(torch.get_rng_state(), original_cpu)
        actual_cuda = torch.cuda.get_rng_state_all()
        assert len(actual_cuda) == len(original_cuda)
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(actual_cuda, original_cuda, strict=True)
        )

    valid_cuda = tuple(state.clone() for state in original_cuda)
    malformed = (
        # Nonempty/rank/dtype-valid CPU states still need generator validation.
        {
            "torch_rng_state": torch.zeros(1, dtype=torch.uint8),
            "cuda_device_count": device_count,
            "cuda_rng_states": valid_cuda,
        },
        {
            "torch_rng_state": torch.zeros_like(original_cpu),
            "cuda_device_count": device_count,
            "cuda_rng_states": valid_cuda,
        },
        # Correct list topology does not make an individual state valid.
        {
            "torch_rng_state": original_cpu,
            "cuda_device_count": device_count,
            "cuda_rng_states": tuple(
                torch.empty(0, dtype=torch.uint8) if index == 0 else state
                for index, state in enumerate(valid_cuda)
            ),
        },
        {
            "torch_rng_state": original_cpu,
            "cuda_device_count": device_count,
            "cuda_rng_states": tuple(
                torch.zeros(1, dtype=torch.uint8) if index == 0 else state
                for index, state in enumerate(valid_cuda)
            ),
        },
    )
    try:
        for saved in malformed:
            with pytest.raises(
                workflow.GlobalDilatedRolloutError,
                match="training optimizer/RNG authority changed",
            ):
                workflow._restore_training_rng_state(
                    saved, device=torch.device("cuda")
                )
            assert_defaults_unchanged()
    finally:
        torch.set_rng_state(original_cpu)
        torch.cuda.set_rng_state_all(list(original_cuda))


def test_rng_restore_rejects_dtype_shape_and_container_tamper_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setter_calls: list[str] = []
    monkeypatch.setattr(
        torch, "set_rng_state", lambda _state: setter_calls.append("cpu")
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda _states: setter_calls.append("cuda"),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    valid_cpu = torch.zeros(8, dtype=torch.uint8)
    valid_cuda = torch.zeros(8, dtype=torch.uint8)
    invalid_payloads = (
        {
            "torch_rng_state": torch.zeros(8, dtype=torch.float32),
            "cuda_device_count": 2,
            "cuda_rng_states": (valid_cuda, valid_cuda),
        },
        {
            "torch_rng_state": torch.zeros((1, 8), dtype=torch.uint8),
            "cuda_device_count": 2,
            "cuda_rng_states": (valid_cuda, valid_cuda),
        },
        {
            "torch_rng_state": valid_cpu,
            "cuda_device_count": 2,
            "cuda_rng_states": (
                torch.zeros(8, dtype=torch.int64),
                valid_cuda,
            ),
        },
        {
            "torch_rng_state": valid_cpu,
            "cuda_device_count": 2,
            "cuda_rng_states": (
                torch.zeros((1, 8), dtype=torch.uint8),
                valid_cuda,
            ),
        },
        {
            "torch_rng_state": valid_cpu,
            "cuda_device_count": 2,
            "cuda_rng_states": "tampered",
        },
        {
            "torch_rng_state": valid_cpu,
            "cuda_device_count": 2,
            "cuda_rng_states": (),
        },
        {
            "torch_rng_state": valid_cpu,
            "cuda_device_count": 2,
            "cuda_rng_states": (valid_cuda,),
        },
        {
            "torch_rng_state": valid_cpu,
            "cuda_device_count": 2,
            "cuda_rng_states": (valid_cuda, valid_cuda, valid_cuda),
        },
        {
            "torch_rng_state": valid_cpu,
            "cuda_device_count": 1,
            "cuda_rng_states": (valid_cuda, valid_cuda),
        },
    )
    for saved in invalid_payloads:
        with pytest.raises(
            workflow.GlobalDilatedRolloutError,
            match="training optimizer/RNG authority changed",
        ):
            workflow._restore_training_rng_state(saved, device=torch.device("cuda"))
    assert setter_calls == []


def test_cpu_rng_restore_requires_zero_cuda_topology_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setter_calls: list[str] = []
    monkeypatch.setattr(
        torch, "set_rng_state", lambda _state: setter_calls.append("cpu")
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda _states: setter_calls.append("cuda"),
    )
    cpu_state = torch.get_rng_state().clone()
    workflow._restore_training_rng_state(
        {
            "torch_rng_state": cpu_state,
            "cuda_device_count": 0,
            "cuda_rng_states": (),
        },
        device=torch.device("cpu"),
    )
    assert setter_calls == ["cpu"]

    setter_calls.clear()
    with pytest.raises(
        workflow.GlobalDilatedRolloutError,
        match="training optimizer/RNG authority changed",
    ):
        workflow._restore_training_rng_state(
            {
                "torch_rng_state": cpu_state,
                "cuda_device_count": 1,
                "cuda_rng_states": (cpu_state,),
            },
            device=torch.device("cpu"),
        )
    assert setter_calls == []


def test_theory_to_code_control_commits_bar_z_m_q_and_sign_contract(
    tmp_path: Path,
) -> None:
    record = workflow._run_theory_to_code_control(tmp_path, SimpleNamespace())
    assert record["passed"] == 1
    assert record["checks"]["bar_Z_divided_by_mobility"] == 0
    assert record["checks"]["training_calls_wrapped_forward"] == 1
    assert record["checks"]["rollout_calls_score_prediction"] == 1
    assert record["checks"]["positive_q_increases_declared_head_logit"] == 1
    reopened = json.loads((tmp_path / "controls/theory_to_code.json").read_text())
    assert reopened == record


def test_external_label_authorization_uses_semantic_opening_seal(
    tmp_path: Path,
) -> None:
    opening = _semantic(
        tmp_path / "physical_train_label_open.json",
        {"schema": "fixture", "role": "train"},
    )
    authorization = workflow._label_authorization(tmp_path, "train")
    assert authorization.opening_seal_sha256 == opening["semantic_sha256"]
    assert authorization.opening_seal_sha256 != workflow.file_fingerprint(
        tmp_path / "physical_train_label_open.json"
    )


def test_resource_events_are_idempotent_by_role_and_detail(tmp_path: Path) -> None:
    first = workflow._record_resource_event(
        tmp_path,
        role="fixed-operation",
        elapsed_seconds=3.0,
        detail={"path": 7},
    )
    second = workflow._record_resource_event(
        tmp_path,
        role="fixed-operation",
        elapsed_seconds=99.0,
        detail={"path": 7},
    )
    third = workflow._record_resource_event(
        tmp_path,
        role="fixed-operation",
        elapsed_seconds=2.0,
        detail={"path": 8},
    )
    assert first["active_seconds"] == 3.0
    assert second["active_seconds"] == 3.0
    assert len(second["events"]) == 1
    assert third["active_seconds"] == 5.0
    assert len(third["events"]) == 2


def test_allocation_requires_freeze_and_selects_smallest_collision_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(repository_root=tmp_path)
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="requires sealed"):
        workflow._allocate_fresh_path(tmp_path / "run", args)

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    _semantic(
        run_dir / "evaluation_freeze.json",
        {"schema": "fixture-freeze", "sealed": 1},
    )
    monkeypatch.setattr(
        workflow,
        "_committed_numerical_path_ids",
        lambda _root: {workflow.FRESH_PATH_POOL[0], workflow.FRESH_PATH_POOL[2]},
    )
    record = workflow._allocate_fresh_path(run_dir, args)
    assert record["fresh_path_id"] == workflow.FRESH_PATH_POOL[1]
    assert record["evaluation_freeze_file_sha256"] == workflow.file_fingerprint(
        run_dir / "evaluation_freeze.json"
    )


def test_seal_refuses_existing_fresh_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_target()
    monkeypatch.setattr(workflow, "load_verified_source_target", lambda _path: source)
    _semantic(tmp_path / "controls/theory_to_code.json", {"schema": "theory", "passed": 1})
    _semantic(tmp_path / "controls/preflight_controls.json", {"schema": "preflight", "passed": 1})
    _semantic(
        tmp_path / "selection.json",
        {
            "schema": "selection",
            "selected": {
                "update": 100,
                "state_sha256": "a" * 64,
                "checkpoint_path": "training/checkpoints/update-0100.pt",
                "checkpoint_file_sha256": "b" * 64,
            },
        },
    )
    _semantic(
        tmp_path / "input_bindings.json",
        {
            "schema": "bindings",
            "frozen_v4_checkpoint": {"state_sha256": "c" * 64},
        },
    )
    (tmp_path / "fresh_forward").mkdir()
    args = SimpleNamespace(source_run_dir=tmp_path)
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="before evaluation freeze"):
        workflow._seal_evaluation(tmp_path, args)


def test_forward_to_127_starts_from_mixed_target_not_unmixed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_target()
    monkeypatch.setattr(workflow, "load_verified_source_target", lambda _path: source)
    freeze = _semantic(
        tmp_path / "evaluation_freeze.json",
        {
            "schema": "fixture-freeze",
            "sealed": 1,
            "source_target_sha256": rollout_array_sha256(source.mixed_target),
        },
    )
    path_usage = {
        "evaluation_freeze_file_sha256": workflow.file_fingerprint(
            tmp_path / "evaluation_freeze.json"
        ),
        "fresh_path_id": workflow.FRESH_PATH_POOL[0],
    }
    _semantic(tmp_path / "path_usage.json", {"schema": "paths", **path_usage})
    captured: dict[str, np.ndarray] = {}

    class Result:
        anchors = {127: source.mixed_target.copy()}
        diagnostics: dict[str, object] = {
            "passed": 1,
            "restart_chain_valid": 1,
            "authorization_fraction": 1.0,
            "certificate_fraction": 1.0,
            "forbidden_event_count": 0,
            "output_state_nonfinite_count": 0,
            "output_state_negative_count": 0,
            "maximum_output_state_mass_error": 0.0,
        }
        shard_records: tuple[dict[str, object], ...] = ()
        elapsed_seconds = 1.0

        def to_record(self) -> dict[str, object]:
            return {"anchor": 127}

    def fake_forward(initial_state: np.ndarray, **kwargs: object) -> Result:
        captured["initial"] = np.array(initial_state, copy=True)
        assert kwargs["step_limit"] == 128
        assert kwargs["anchor_steps"] == (127,)
        return Result()

    monkeypatch.setattr(workflow, "run_forward_trajectory", fake_forward)
    args = SimpleNamespace(source_run_dir=tmp_path, device="cpu")
    workflow._run_forward_to_127(tmp_path, args, path_usage)
    assert np.array_equal(captured["initial"], source.mixed_target)
    assert not np.array_equal(captured["initial"], source.source_image)
    assert (tmp_path / "fresh_forward/anchor-step-0127.npz").is_file()
    assert freeze["sealed"] == 1


def _write_fake_exact_shards(run_dir: Path) -> np.ndarray:
    anchor = np.full(STATE_SIZE, 1.0 / STATE_SIZE, dtype=np.float64)
    atomic_rollout_npz(run_dir / "fresh_forward/anchor-step-0127.npz", {"state": anchor})
    state = np.repeat(anchor[None, :], len(workflow.ROW_ORDER), axis=0)
    expected = [state.copy()]
    root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
    for index in range(16):
        input_hash = rollout_array_sha256(state)
        next_state = state.copy()
        for row in range(1, len(workflow.ROW_ORDER)):
            amount = (row + 1) * 1.0e-8
            next_state[row, 0] += amount
            next_state[row, 1] -= amount
        state = next_state
        archive_path = root / f"shard-{index:04d}.npz"
        atomic_rollout_npz(archive_path, {"state": state})
        _semantic(
            root / f"shard-{index:04d}.json",
            {
                "schema": "fixture-shard",
                "committed": 1,
                "input_state_sha256": input_hash,
                "output_state_sha256": rollout_array_sha256(state),
                "state_file_sha256": rollout_file_sha256(archive_path),
                "sequence_start": [127 - 8 * index, 6],
                "per_row_diagnostics": [
                    {
                        "row_key": row_key,
                        "score_squared_sum": float(row),
                        "score_count": 1,
                        "control_fraction_displacement_squared_sum": float(row) / 10,
                        "control_fraction_displacement_count": 1,
                        "reference_fraction_displacement_squared_sum": 1.0,
                        "reference_fraction_displacement_count": 1,
                    }
                    for row, row_key in enumerate(workflow.ROW_ORDER)
                ],
            },
        )
        expected.append(state.copy())
    return np.stack(expected, axis=1)


def test_aggregate_reopens_all_16_shards_and_slices_milestones_bitwise(
    tmp_path: Path,
) -> None:
    expected = _write_fake_exact_shards(tmp_path)
    record = workflow._aggregate_existing_shards(tmp_path)
    with np.load(tmp_path / "suffix/trajectory_shard_boundaries.npz") as archive:
        assert np.array_equal(archive["states"], expected)
        assert np.array_equal(
            archive["completed_reverse_steps"], np.arange(0, 129, 8)
        )
    with np.load(tmp_path / "suffix/milestones.npz") as archive:
        assert np.array_equal(archive["states"], expected[:, [0, 4, 8, 12, 16], :])
        assert np.array_equal(
            archive["completed_reverse_steps"], workflow.MILESTONE_STEPS
        )
    assert record["chain_valid"] == 1


def test_mechanism_quarters_are_reverse_progress_not_outer_step_quartiles() -> None:
    shards = [
        {
            "sequence_start": [127 - index * 8, 6],
            "per_row_diagnostics": [
                {
                    "score_squared_sum": 1.0,
                    "score_count": 1,
                    "control_fraction_displacement_squared_sum": 1.0,
                    "control_fraction_displacement_count": 1,
                    "reference_fraction_displacement_squared_sum": 1.0,
                    "reference_fraction_displacement_count": 1,
                }
                for _ in workflow.ROW_ORDER
            ],
        }
        for index in range(16)
    ]
    result = workflow._row_quarter_mechanism(shards)
    for row in workflow.ROW_ORDER:
        assert [result[row][str(q)]["shard_count"] for q in range(4)] == [4, 4, 4, 4]
        assert result[row]["0"]["reverse_progress_steps"] == [0, 32]
        assert result[row]["3"]["reverse_progress_steps"] == [96, 128]


def test_strict_fused_health_accepts_certified_fallback_but_rejects_unauthorized_row() -> None:
    state = np.full((1, STATE_SIZE), 1.0 / STATE_SIZE, dtype=np.float64)
    transition_count = 2 * workflow.MICROSTEPS * EDGES_PER_PHASE
    active_count = transition_count - 8
    phase_prefixes = (
        "reference_fraction_displacement",
        "control_fraction_displacement",
        "score",
        "logistic_shift",
    )
    row_table = {
        "row_key": "row",
        "canonical_path_id": 1,
        "controller_kind": "learned",
        "variant": "fixture",
        "horizon": "fixture",
        "gain": 1.0,
        "controller_binding": {},
    }
    phase_row: dict[str, object] = {
        "row_key": "row",
        "transition_count": transition_count,
        "boundary_fraction_count": 0,
        "maximum_pair_mass_error": 0.0,
        "maximum_simplex_mass_error": 0.0,
        "reference_transition_count": transition_count,
        "reference_active_count": active_count,
        "reference_structural_noop_count": 8,
        "reference_certified_count": active_count,
        "reference_fallback_count": 2,
        "reference_unauthorized_count": 1,
        "reference_invalid_count": 0,
        "reference_certificate_fraction": 1.0,
        **{name: 0 for name in workflow._FUSED_INVALID_FIELDS[:7]},
    }
    for prefix in phase_prefixes:
        phase_row[f"{prefix}_count"] = active_count
        phase_row[f"{prefix}_squared_sum"] = 0.0
        phase_row[f"{prefix}_maximum_absolute"] = 0.0
        phase_row[f"{prefix}_rms"] = 0.0
    controller_row = {
        "row_key": "row",
        "controller_kind": "learned",
        "gain": 1.0,
        "call_count": 1,
        "lane_count": EDGES_PER_PHASE,
        "score_count": active_count,
        "movable_count": 0,
        "already_equal_count": 0,
        "zero_pair_mass_count": 0,
        "zero_duration_count": 0,
        "target_oracle_unreachable_boundary_count": 7,
        "clipping_count": 0,
        "floor_count": 0,
        "projection_count": 0,
        "nonfinite_score_count": 0,
        "score_squared_sum": 0.0,
        "score_maximum_absolute": 0.0,
        "unscaled_score_squared_sum": 0.0,
        "unscaled_score_maximum_absolute": 0.0,
        "score_rms": 0.0,
        "unscaled_score_rms": 0.0,
    }
    shard = {
        "committed": 1,
        "transition_count": transition_count,
        "microsteps": workflow.MICROSTEPS,
        "execution_plan": {
            "row_count": 1,
            "transition_count": transition_count,
            "sequence": [[127, 6]],
        },
        "row_table": [row_table],
        "controller_diagnostics": [controller_row],
        "diagnostics": {
            "transition_count": transition_count,
            "certificate_fraction": 1.0,
            "maximum_mass_error": 0.0,
            "fallback_count": 2,
            "forbidden_counts": {name: 0 for name in workflow._FORBIDDEN_EXACT_COUNTS},
            "reference": {
                "transition_count": transition_count,
                "active_count": active_count,
                "structural_noop_count": 8,
                "certified_count": active_count,
                "forbidden_counts": {name: 0 for name in workflow._FORBIDDEN_EXACT_COUNTS},
                "fallback_count": 2,
                "unauthorized_count": 1,
                "invalid_count": 0,
                "per_row": [
                    {
                        "active_count": active_count,
                        "structural_noop_count": 8,
                        "certified_count": active_count,
                        "transition_count": transition_count,
                        "certificate_fraction": 1.0,
                        "invalid_count": 0,
                        "unauthorized_count": 1,
                        "fallback_count": 2,
                    }
                ],
            },
        },
        "per_row_diagnostics": [phase_row],
    }
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="health|authority|telemetry"):
        workflow._strict_fused_exact_health(
            final_state=state, shard_records=[shard], row_count=1
        )
    shard["diagnostics"]["reference"]["per_row"][0]["unauthorized_count"] = 0
    shard["diagnostics"]["reference"]["unauthorized_count"] = 0
    phase_row["reference_unauthorized_count"] = 0
    passed = workflow._strict_fused_exact_health(
        final_state=state, shard_records=[shard], row_count=1
    )
    assert passed["passed"] == 1
    assert passed["fallback_count"] == 2


def test_existing_preflight_path_is_reused_after_crash(tmp_path: Path) -> None:
    root = tmp_path / "controls/exact_smoke/fused_families/five-row/one-shard"
    selected = workflow.PREFLIGHT_PATH_POOL[3]
    _semantic(
        root / "shard-0000.json",
        {
            "schema": "fixture",
            "committed": 1,
            "canonical_path_ids": [selected] * len(workflow.ROW_ORDER),
        },
    )
    assert workflow._existing_preflight_path_id(tmp_path) == selected


def test_quartile_reference_persists_only_finite_arrays() -> None:
    class Inputs:
        def row_array(self, name: str) -> np.ndarray:
            if name == "later_full_state":
                return np.vstack(
                    [
                        np.full((2, STATE_SIZE), (1.0 + quarter * 1e-3) / STATE_SIZE)
                        for quarter in range(4)
                    ]
                )
            if name == "outer_step":
                return np.array([0, 1, 128, 129, 256, 257, 384, 385])
            raise KeyError(name)

    # Add a tiny within-quartile perturbation so every p95 scale is positive.
    inputs = Inputs()
    states = inputs.row_array("later_full_state")
    for index in (0, 2, 4, 6):
        states[index, 0] += 1e-5
        states[index, 1] -= 1e-5
    inputs.row_array = lambda name: states if name == "later_full_state" else np.array(
        [0, 1, 128, 129, 256, 257, 384, 385]
    )
    result = workflow._quartile_reference(inputs)  # type: ignore[arg-type]
    assert all(np.isfinite(value).all() for value in result.values())


def test_metrics_images_and_final_verification_keep_every_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _write_fake_exact_shards(tmp_path)
    workflow._aggregate_existing_shards(tmp_path)
    source = _source_target()
    monkeypatch.setattr(workflow, "load_verified_source_target", lambda _path: source)
    scale = fixed_rendering_scale(source.source_image, source.mixed_target, 0.35)
    _semantic(
        tmp_path / "evaluation_freeze.json",
        {"schema": "fixture-freeze", "sealed": 1, "rendering_scale": scale.to_dict()},
    )
    validation_ratios = np.tile(np.linspace(0.0, 2.0, 32), (4, 1))
    atomic_rollout_npz(
        tmp_path / "training/on_policy_validation_calibration.npz",
        {
            "training_means": np.full((4, STATE_SIZE), 1.0 / STATE_SIZE),
            "training_p95": np.ones(4),
            "validation_sorted_ratios": validation_ratios,
            "validation_counts": np.full(4, 32, dtype=np.int64),
        },
    )
    args = SimpleNamespace(source_run_dir=tmp_path)
    summary = workflow._compute_metrics_and_images(tmp_path, args)
    assert set(summary["primary_final_objectives"]) == set(workflow.ROW_ORDER)
    assert summary["metric_row_count"] == 5 * 17
    assert len(summary["images"]) == 5 * 5
    for row_key in workflow.ROW_ORDER:
        assert (tmp_path / f"images/raw/{row_key}/step-128.png").is_file()
        assert (tmp_path / f"images/demixed/{row_key}/step-128.png").is_file()
    monkeypatch.setattr(
        workflow, "_verify_scientific_evidence_read_only", lambda *_args: {"passed": 1}
    )
    verified = workflow._verify_raw_and_derived(tmp_path, args)
    assert verified["milestones_bitwise_recomputed"] == 1
    assert verified["primary_metrics_recomputed"] == 1
    assert verified["images_decoded_and_reproduced"] == 1
    assert verified["all_five_rows_present"] == 1
    assert np.array_equal(expected, np.load(tmp_path / "suffix/trajectory_shard_boundaries.npz")["states"])

    # Verification covers every generated milestone, not only the final PNG.
    from PIL import Image

    tampered_path = tmp_path / "images/raw/global-plus-1/step-032.png"
    tampered = np.asarray(Image.open(tampered_path).convert("L")).copy()
    tampered[0, 0] ^= np.uint8(1)
    Image.fromarray(tampered, mode="L").save(tampered_path)
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="milestone image"):
        workflow._verify_raw_and_derived(tmp_path, args)


@pytest.mark.parametrize(
    ("global_relative", "sign_relative", "selected_risk", "expected"),
    [
        (0.02, -0.1, 0.9, "global_material_improvement"),
        (-0.1, 0.02, 1.1, "sign_order_leading"),
        (0.001, -0.1, 1.1, "global_positive_small"),
        (-0.1, -0.1, 0.9, "validation_better_suffix_adverse"),
        (-0.1, -0.1, 1.1, "all_learned_adverse_controls_pass"),
    ],
)
def test_outcome_classification_has_distinct_predeclared_actions(
    tmp_path: Path,
    global_relative: float,
    sign_relative: float,
    selected_risk: float,
    expected: str,
) -> None:
    objectives = {
        row: {
            "relative_paired_squared_l2_improvement_over_zero": -0.1,
            "paired_squared_l2_improvement_over_zero": -0.01,
        }
        for row in workflow.ROW_ORDER
    }
    objectives["zero"]["relative_paired_squared_l2_improvement_over_zero"] = 0.0
    objectives["global-plus-1"]["relative_paired_squared_l2_improvement_over_zero"] = global_relative
    objectives["v4-minus-0p5"]["relative_paired_squared_l2_improvement_over_zero"] = sign_relative
    objectives["source-informed"]["relative_paired_squared_l2_improvement_over_zero"] = 0.2
    _semantic(
        tmp_path / "suffix/summary.json",
        {"schema": "summary", "primary_final_objectives": objectives},
    )
    _semantic(
        tmp_path / "selection.json",
        {
            "schema": "selection",
            "selected": {"normalized_validation_mse": selected_risk},
            "zero_checkpoint": {"normalized_validation_mse": 1.0},
        },
    )
    record = workflow._classify_outcome(tmp_path)
    assert record["outcome"] == expected
    assert record["required_next_action"]


def test_semantic_reader_rejects_missing_seal(tmp_path: Path) -> None:
    path = tmp_path / "unsealed.json"
    atomic_write_json(path, {"schema": "fixture", "passed": 1})
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="semantic hash"):
        workflow._read_json(path, semantic=True)


def test_resource_cap_crossing_is_durably_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 1.0)
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="durably recorded"):
        workflow._record_resource_event(
            tmp_path, role="over-cap-attempt", elapsed_seconds=2.0, detail={"attempt": 0}
        )
    ledger = workflow._read_json(tmp_path / "resource_ledger.json", semantic=True)
    assert ledger["active_seconds"] == 2.0
    assert ledger["limits_passed"] == 0
    assert ledger["breaches"] == ["active_seconds_cap"]
    assert ledger["events"][-1]["limits_passed"] == 0
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="remains crossed"):
        workflow._record_resource_event(
            tmp_path, role="over-cap-attempt", elapsed_seconds=2.0, detail={"attempt": 0}
        )


def test_timed_resource_charges_each_failed_retry(tmp_path: Path) -> None:
    for _ in range(2):
        with pytest.raises(RuntimeError, match="fixture failure"):
            with workflow._timed_resource(tmp_path, "retrying-operation"):
                raise RuntimeError("fixture failure")
    ledger = workflow._resource_ledger(tmp_path)
    events = [
        item for item in ledger["events"] if item["role"] == "retrying-operation"
    ]
    assert len(events) == 2
    assert [item["detail"]["attempt"] for item in events] == [0, 1]
    assert all(item["detail"]["failed"] == 1 for item in events)


def test_exact_attempt_resume_recovers_durable_prefix_without_double_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter((1.0, 11.0))
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: next(clock))
    first = workflow._record_attempt_wall(
        tmp_path,
        role="fixture-exact-family-attempt",
        started=0.0,
        durable_before_seconds=0.0,
        durable_elapsed_seconds=5.0,
        failed=True,
    )
    assert first["active_seconds"] == 5.0
    second = workflow._record_attempt_wall(
        tmp_path,
        role="fixture-exact-family-attempt",
        started=10.0,
        durable_before_seconds=5.0,
        durable_elapsed_seconds=7.0,
        failed=False,
    )
    assert second["active_seconds"] == 7.0
    events = second["events"]
    assert [item["detail"]["attempt"] for item in events] == [0, 1]
    assert events[1]["detail"]["recovered_unaccounted_prefix_seconds"] == 0.0


def test_exact_attempt_recovers_hard_crash_prefix_conservatively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 0.25)
    ledger = workflow._record_attempt_wall(
        tmp_path,
        role="fixture-hard-crash-recovery",
        started=0.0,
        durable_before_seconds=5.0,
        durable_elapsed_seconds=5.0,
        failed=False,
    )
    event = ledger["events"][0]
    assert event["elapsed_seconds"] == 5.25
    assert event["detail"]["recovered_unaccounted_prefix_seconds"] == 5.0


def test_resource_ledger_rejects_self_sealed_favorable_tamper(tmp_path: Path) -> None:
    workflow._record_resource_event(
        tmp_path, role="real-work", elapsed_seconds=4.0, detail={"attempt": 0}
    )
    ledger = workflow._read_json(tmp_path / "resource_ledger.json", semantic=True)
    ledger["events"][0]["elapsed_seconds"] = 0.0
    body = dict(ledger)
    body.pop("semantic_sha256")
    atomic_write_json(tmp_path / "resource_ledger.json", workflow._semantic(body))
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="aggregate authority"):
        workflow._resource_ledger(tmp_path)


def test_device_is_part_of_immutable_scientific_fingerprint() -> None:
    cpu = workflow._scientific_config(SimpleNamespace(device="cpu"))
    cuda = workflow._scientific_config(SimpleNamespace(device="cuda"))
    assert cpu["execution_device"] == "cpu"
    assert cpu["training"]["device"] == "cpu"
    assert cpu["semantic_sha256"] != cuda["semantic_sha256"]


def test_scientific_config_is_identical_after_json_roundtrip(tmp_path: Path) -> None:
    live = workflow._scientific_config(SimpleNamespace(device="cuda"))
    path = tmp_path / "scientific_config.json"
    atomic_write_json(path, live)

    assert isinstance(live["training"]["betas"], list)
    assert workflow._read_json(path, semantic=True) == live


def test_bound_input_remeasurement_must_equal_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = _semantic(
        tmp_path / "input_bindings.json",
        {"schema": "fixture-bindings", "execution_device": "cuda", "value": 1},
    )
    monkeypatch.setattr(
        workflow,
        "_measure_input_bindings",
        lambda _args: {**sealed, "value": 2},
    )
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="immutable input"):
        workflow._verify_bound_paths(tmp_path, SimpleNamespace())


def test_source_interface_control_failure_routes_before_global_success(
    tmp_path: Path,
) -> None:
    objectives = {
        row: {
            "relative_paired_squared_l2_improvement_over_zero": -0.1,
            "paired_squared_l2_improvement_over_zero": -0.01,
        }
        for row in workflow.ROW_ORDER
    }
    objectives["zero"]["relative_paired_squared_l2_improvement_over_zero"] = 0.0
    objectives["global-plus-1"]["relative_paired_squared_l2_improvement_over_zero"] = 0.1
    _semantic(
        tmp_path / "suffix/summary.json",
        {"schema": "summary", "primary_final_objectives": objectives},
    )
    _semantic(
        tmp_path / "selection.json",
        {
            "schema": "selection",
            "selected": {"normalized_validation_mse": 0.5},
            "zero_checkpoint": {"normalized_validation_mse": 1.0},
        },
    )
    record = workflow._classify_outcome(tmp_path)
    assert record["outcome"] == "source_interface_control_failed"
    assert "repair" in record["required_next_action"]


def test_optional_failure_preserves_completed_mandatory_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_positive_complete_path_impl",
        lambda *_args: (_ for _ in ()).throw(
            workflow.GlobalDilatedRolloutError("optional numerical health failed")
        ),
    )
    record = workflow._maybe_run_positive_complete_path(tmp_path, SimpleNamespace())
    assert record["completed"] == 0
    assert record["mandatory_suffix_objective_preserved"] == 1
    assert record["failure_domain"] == "optional_execution_integrity"
    assert record["complete_path_claim_authorized"] == 0


def test_optional_failure_charges_only_unreconciled_outer_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter((0.0, 10.0))
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: next(clock))

    def fail_after_inner_accounting(run_dir: Path, _args: object) -> None:
        workflow._record_resource_event(
            run_dir,
            role="fixture-optional-inner",
            elapsed_seconds=8.0,
            detail={"attempt": 0},
        )
        raise workflow.GlobalDilatedRolloutError("late optional postprocess failure")

    monkeypatch.setattr(
        workflow, "_run_positive_complete_path_impl", fail_after_inner_accounting
    )
    record = workflow._maybe_run_positive_complete_path(tmp_path, SimpleNamespace())
    ledger = workflow._resource_ledger(tmp_path)
    failed = [
        event for event in ledger["events"] if event["role"] == "positive_branch_failed_attempt"
    ]
    assert record["mandatory_suffix_objective_preserved"] == 1
    assert ledger["active_seconds"] == 10.0
    assert len(failed) == 1
    assert failed[0]["elapsed_seconds"] == 2.0
    assert failed[0]["detail"]["inner_accounted_seconds"] == 8.0


def test_optional_forward_exception_is_accounted_before_outer_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _semantic(
        tmp_path / "outcome.json",
        {"schema": "fixture-outcome", "outcome": "global_material_improvement"},
    )
    _semantic(
        tmp_path / "controls/preflight_controls.json",
        {
            "schema": "fixture-preflight",
            "five_row_exact_shard_elapsed_seconds": 1.0,
            "durable_five_row_exact_shard_seconds": 1.0,
        },
    )
    _semantic(
        tmp_path / "path_usage.json",
        {"schema": "fixture-path", "fresh_path_id": workflow.FRESH_PATH_POOL[0]},
    )
    monkeypatch.setattr(
        workflow, "load_verified_source_target", lambda _path: _source_target()
    )
    monkeypatch.setattr(
        workflow,
        "_optional_postprocess_reserve",
        lambda _run_dir: {
            "semantic_sha256": "1" * 64,
            "reserve_seconds": 30.0,
            "reserve_storage_bytes": 1024,
        },
    )
    monkeypatch.setattr(
        workflow,
        "run_forward_trajectory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forward failed")),
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        workflow._run_positive_complete_path_impl(
            tmp_path, SimpleNamespace(source_run_dir=tmp_path, device="cpu")
        )
    events = workflow._resource_ledger(tmp_path)["events"]
    forward_events = [
        event
        for event in events
        if event["role"] == "positive_branch_forward_128_to_511_attempt"
    ]
    assert len(forward_events) == 1
    assert forward_events[0]["detail"]["failed"] == 1


def test_integrity_message_containing_resource_never_authorizes_same_run_resume(
    tmp_path: Path,
) -> None:
    for name in ("scientific_config.json", "exact_command.txt", "run_manifest.json"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    error = workflow.GlobalDilatedRolloutError(
        "resource ledger authority changed"
    )
    workflow._capture_failure(tmp_path, "initialization", error)
    verification = workflow._finalize_failure(tmp_path, "initialization", error)
    terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    assert verification["failure_domain"] == "execution_integrity"
    assert terminal["failure_code"] == "global_rollout_integrity_or_numerical_failure"
    assert terminal["resume_same_frozen_run_authorized"] == 0
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="does not authorize"):
        workflow._verify_resume_compatibility(
            tmp_path, SimpleNamespace(resume_run_dir=tmp_path)
        )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("journal_relative", "role", "durable_root_relative", "minimum_start"),
    [
        (
            "fresh_forward/active-forward-to-127-attempt.json",
            "fresh_exact_forward_to_127_attempt",
            "fresh_forward/forward_shards/fresh-main-path",
            None,
        ),
        (
            "suffix/active-mandatory-exact-attempt.json",
            "mandatory_exact_five_row_suffix_attempt",
            "suffix/fused_families/fresh-five-row/suffix-128",
            None,
        ),
        (
            "positive/active-forward-tail-attempt.json",
            "positive_branch_forward_128_to_511_attempt",
            "fresh_forward/forward_shards/fresh-main-path",
            128,
        ),
        (
            "positive/active-complete-reverse-attempt.json",
            "positive_branch_complete_three_row_exact_attempt",
            "positive/fused_families/same-path-three-row/complete-512",
            None,
        ),
    ],
)
def test_uncommitted_exact_shard_hard_crash_debits_unknown_interval_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_relative: str,
    role: str,
    durable_root_relative: str,
    minimum_start: int | None,
) -> None:
    clock = iter((100.0, 109.0))
    monkeypatch.setattr(workflow.time, "time", lambda: next(clock))
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 2.0)
    detail: dict[str, object] = {
        "durable_root_relative": durable_root_relative,
        "fixture": "sampler-entered-no-commit",
    }
    if minimum_start is not None:
        detail["durable_minimum_start_step"] = minimum_start
    workflow._begin_durable_attempt(
        tmp_path,
        journal_relative=journal_relative,
        role=role,
        detail=detail,
    )
    # Hard process death: sampler entered, but no exact shard or finish event
    # reached durable storage.  The next invocation must reconcile first.
    recovered = workflow._reconcile_durable_attempt_journal(
        tmp_path, journal_relative=journal_relative, role=role
    )
    assert recovered is not None
    event = workflow._resource_ledger(tmp_path)["events"][-1]
    assert event["role"] == role + "_abandoned_attempt"
    assert event["elapsed_seconds"] == 14.0
    assert event["detail"]["unknown_active_interval_seconds"] == 9.0
    assert event["detail"]["durable_committed_shard_seconds"] == 0.0
    assert event["detail"]["idle_or_powered_off_time_may_be_included"] == 1


def test_preflight_exact_smoke_hard_death_keeps_journal_and_reconciles_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter((100.0, 108.0))
    monkeypatch.setattr(workflow.time, "time", lambda: next(clock))
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 3.0)

    def hard_death(*_args: object, **_kwargs: object) -> None:
        assert (tmp_path / "controls/active-exact-preflight-attempt.json").is_file()
        raise KeyboardInterrupt("injected uncommitted exact preflight death")

    monkeypatch.setattr(workflow, "_run_five_row_exact_smoke_impl", hard_death)
    with pytest.raises(KeyboardInterrupt, match="injected uncommitted"):
        workflow._run_five_row_exact_smoke(tmp_path, SimpleNamespace())
    journal = tmp_path / "controls/active-exact-preflight-attempt.json"
    assert journal.is_file()
    recovered = workflow._reconcile_durable_attempt_journal(
        tmp_path,
        journal_relative="controls/active-exact-preflight-attempt.json",
        role="five_row_exact_preflight_attempt",
    )
    assert recovered is not None
    event = workflow._resource_ledger(tmp_path)["events"][-1]
    assert event["role"] == "five_row_exact_preflight_attempt_abandoned_attempt"
    assert event["elapsed_seconds"] == 13.0
    assert event["detail"]["unknown_active_interval_seconds"] == 8.0
    assert event["detail"]["durable_committed_shard_seconds"] == 0.0


def test_preflight_event_commit_then_journal_cleanup_crash_is_not_double_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_path = tmp_path / "controls/preflight_controls.json"
    original_impl = workflow._run_five_row_exact_smoke_impl

    def fake_impl(
        run_dir: Path,
        _args: object,
        *,
        durable_attempt_id: str | None = None,
        recovered_durable_covered_seconds: float = 0.0,
    ) -> dict[str, object]:
        del recovered_durable_covered_seconds
        if control_path.is_file():
            return workflow._read_json(control_path, semantic=True)
        assert durable_attempt_id is not None
        record = _semantic(
            control_path,
            {"schema": "fixture-preflight", "passed": 1},
        )
        workflow._record_resource_event(
            run_dir,
            role="five_row_exact_preflight",
            elapsed_seconds=7.0,
            detail={
                "durable_attempt_id": durable_attempt_id,
                "fixture": "event-committed-before-journal-cleanup",
            },
        )
        return record

    monkeypatch.setattr(workflow, "_run_five_row_exact_smoke_impl", fake_impl)
    original_unlink = Path.unlink
    crashed = {"done": False}

    def crash_journal_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "active-exact-preflight-attempt.json" and not crashed["done"]:
            crashed["done"] = True
            raise KeyboardInterrupt("injected post-event journal cleanup crash")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", crash_journal_unlink)
    with pytest.raises(KeyboardInterrupt, match="post-event"):
        workflow._run_five_row_exact_smoke(tmp_path, SimpleNamespace())
    journal = tmp_path / "controls/active-exact-preflight-attempt.json"
    assert journal.is_file()
    before = workflow._resource_ledger(tmp_path)
    assert before["active_seconds"] == 7.0

    monkeypatch.setattr(Path, "unlink", original_unlink)
    # Exercise the real existing-control branch on resume.  It must be pure;
    # validation-only wall accounting belongs to the wrapper.
    monkeypatch.setattr(workflow, "_run_five_row_exact_smoke_impl", original_impl)
    resumed = workflow._run_five_row_exact_smoke(tmp_path, SimpleNamespace())
    assert resumed["passed"] == 1
    assert not journal.exists()
    after = workflow._resource_ledger(tmp_path)
    exact_events = [
        event for event in after["events"] if event["role"] == "five_row_exact_preflight"
    ]
    assert len(exact_events) == 1
    assert exact_events[0]["elapsed_seconds"] == 7.0
    assert not any(
        event["role"] == "five_row_exact_preflight_attempt_abandoned_attempt"
        for event in after["events"]
    )
    validation_events = [
        event
        for event in after["events"]
        if event["role"] == "five_row_exact_preflight_resume_validation"
    ]
    assert len(validation_events) == 1
    assert validation_events[0]["detail"]["durable_before_seconds"] == 0.0
    assert validation_events[0]["detail"]["durable_committed_shard_seconds"] == 0.0
    assert validation_events[0]["detail"][
        "validation_only_no_durable_preflight_recharge"
    ] == 1


def test_optional_forward_runtime_degradation_stops_before_next_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 700.0)
    monkeypatch.setattr(workflow, "REPORT_RESERVE_SECONDS", 600.0)
    monkeypatch.setattr(workflow, "FORWARD_RESERVE_SECONDS", 160.0)
    forward_root = tmp_path / "fresh_forward/forward_shards/fresh-main-path"
    _semantic(
        forward_root / "shard-0016.json",
        {
            "schema": "fixture-forward-shard",
            "committed": 1,
            "start_step": 128,
            "elapsed_seconds": 90.0,
        },
    )
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 0.0)
    sampler_calls = {"count": 0}

    def sampler_must_not_run(*_args: object, **_kwargs: object) -> None:
        sampler_calls["count"] += 1

    def execute_next_boundary() -> None:
        workflow._admit_optional_forward_shard(
            tmp_path,
            forward_root=forward_root,
            attempt_started=0.0,
            shard_index=17,
            projected_reverse_seconds=0.0,
            postprocess_reserve_seconds=50.0,
            postprocess_reserve_storage_bytes=1,
        )
        sampler_must_not_run()

    with pytest.raises(workflow.ResourceBoundaryError, match="cannot preserve"):
        execute_next_boundary()
    assert sampler_calls["count"] == 0


def test_optional_postprocess_reserve_binds_measured_mandatory_authority(
    tmp_path: Path,
) -> None:
    workflow._record_resource_event(
        tmp_path,
        role="mandatory_objective_postprocessing",
        elapsed_seconds=40.0,
        detail={"attempt": 0, "failed": 0},
    )
    for relative in (
        "suffix/trajectory_shard_boundaries.npz",
        "suffix/milestones.npz",
        "suffix/metrics.csv",
        "suffix/mechanism.json",
        "suffix/summary.json",
        "images/raw/zero/step-000.png",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((relative + "\n").encode("utf-8"))
    first = workflow._optional_postprocess_reserve(tmp_path)
    assert first is not None
    assert first["reserve_seconds"] == 60.0
    assert first["reserve_storage_bytes"] >= 16 * 1024**2
    assert first["source_successful_event_ids"] == [
        workflow._resource_ledger(tmp_path)["events"][0]["event_id"]
    ]
    assert workflow._optional_postprocess_reserve(tmp_path) == first
    (tmp_path / "suffix/summary.json").write_bytes(b"changed\n")
    with pytest.raises(
        workflow.GlobalDilatedRolloutError,
        match="reserve differs from mandatory authority",
    ):
        workflow._optional_postprocess_reserve(tmp_path)


def test_optional_reverse_gate_preserves_postprocess_reserve_before_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 700.0)
    monkeypatch.setattr(workflow, "REPORT_RESERVE_SECONDS", 600.0)
    _semantic(
        tmp_path / "controls/preflight_controls.json",
        {
            "schema": "fixture-preflight",
            "five_row_exact_shard_elapsed_seconds": 1.0,
            "durable_five_row_exact_shard_seconds": 1.0,
        },
    )
    callback = workflow._suffix_admission_callback(
        tmp_path,
        tmp_path / "positive/fused_families/same-path-three-row/complete-512",
        attempt_started=0.0,
        additional_active_reserve_seconds=100.0,
        additional_storage_reserve_bytes=1,
    )
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 0.0)
    sampler_calls = {"count": 0}

    def sampler_boundary() -> None:
        callback(SimpleNamespace(shard_index=0))
        sampler_calls["count"] += 1

    with pytest.raises(workflow.ResourceBoundaryError, match="preserve report reserve"):
        sampler_boundary()
    assert sampler_calls["count"] == 0


def test_optional_postprocessing_hard_death_is_reconciled_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def build(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise KeyboardInterrupt("injected optional postprocess death")
        return {
            "schema": workflow.VERSION + "-positive-branch",
            "schema_version": 1,
            "triggered": 1,
            "attempted": 1,
            "completed": 1,
        }

    monkeypatch.setattr(
        workflow, "_build_optional_positive_postprocess_artifacts", build
    )
    reserve = {
        "semantic_sha256": "2" * 64,
        "reserve_seconds": 30.0,
        "reserve_storage_bytes": 1024,
    }
    kwargs = {
        "source": object(),
        "anchor": np.zeros(workflow.STATE_SIZE, dtype=np.float64),
        "forward": object(),
        "result": object(),
        "strict_health": {},
        "reserve": reserve,
    }
    with pytest.raises(KeyboardInterrupt, match="optional postprocess death"):
        workflow._run_optional_positive_postprocessing(tmp_path, **kwargs)
    relative = "positive/active-positive-postprocessing.json"
    abandoned = workflow._read_json(tmp_path / relative, semantic=True)
    completed = workflow._run_optional_positive_postprocessing(tmp_path, **kwargs)
    assert completed["completed"] == 1
    assert not (tmp_path / relative).exists()
    events = workflow._resource_ledger(tmp_path)["events"]
    assert [event["role"] for event in events] == [
        "optional_positive_postprocessing_abandoned_attempt",
        "optional_positive_postprocessing",
    ]
    assert sum(
        event["detail"].get("durable_attempt_id") == abandoned["attempt_id"]
        for event in events
    ) == 1
    assert events[0]["detail"]["unknown_active_interval_seconds"] >= 0.0
    assert events[1]["elapsed_seconds"] >= 5.0


def test_optional_postprocess_reconcile_cap_defers_before_any_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _semantic(
        tmp_path / "outcome.json",
        {"schema": "fixture-outcome", "outcome": "global_material_improvement"},
    )
    relative = "positive/active-positive-postprocessing.json"
    journal, _ = workflow._begin_durable_attempt(
        tmp_path,
        journal_relative=relative,
        role="optional_positive_postprocessing",
        detail={"fixture": "hard death during optional images"},
    )
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 1.0)
    sampler_calls = {"count": 0}

    def sampler_must_not_run(*_args: object, **_kwargs: object) -> None:
        sampler_calls["count"] += 1
        raise AssertionError("optional sampler launched after reconciliation cap")

    monkeypatch.setattr(workflow, "run_forward_trajectory", sampler_must_not_run)
    result = workflow._run_positive_complete_path_impl(
        tmp_path, SimpleNamespace()
    )
    assert result["completed"] == 0
    assert result["failure_domain"] == "resource_budget"
    assert result["mandatory_suffix_objective_preserved"] == 1
    assert sampler_calls["count"] == 0
    assert not (tmp_path / relative).exists()
    covering = [
        event
        for event in workflow._resource_ledger(tmp_path)["events"]
        if event["detail"].get("durable_attempt_id") == journal["attempt_id"]
    ]
    assert len(covering) == 1
    assert covering[0]["role"] == "optional_positive_postprocessing_abandoned_attempt"


def test_terminal_complete_resume_exits_without_mutating_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _semantic(tmp_path / "sentinel.json", {"schema": "fixture-sentinel", "value": 1})
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    args = SimpleNamespace(resume_run_dir=tmp_path)
    verified = {"count": 0}
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(workflow, "_initialize_run", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_stage_complete",
        lambda _run_dir, stage: stage == "report_verify",
    )
    monkeypatch.setattr(
        workflow,
        "_verify_completed_report_read_only",
        lambda *_args: verified.__setitem__("count", verified["count"] + 1),
    )
    monkeypatch.setattr(
        workflow,
        "_record_attempt_wall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal-complete resume must not mutate the ledger")
        ),
    )
    assert workflow.main([]) == 0
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert verified["count"] == 1
    assert after == before


def test_completed_report_resume_accepts_roundtripped_config_byte_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = [
        "--stage",
        "report_verify",
        "--resume-run-dir",
        str(tmp_path),
        "--repository-root",
        str(tmp_path),
        "--device",
        "cpu",
    ]
    config = workflow._scientific_config(SimpleNamespace(device="cpu"))
    atomic_write_json(tmp_path / "scientific_config.json", config)
    (tmp_path / "exact_command.txt").write_text(
        "fixture completed invocation\n", encoding="utf-8"
    )
    closure = [{"path": "fixture.py", "sha256": "a" * 64}]
    closure_hash = "b" * 64
    _semantic(
        tmp_path / "run_manifest.json",
        {
            "schema": "fixture-run-manifest",
            "source_closure": closure,
            "source_closure_sha256": closure_hash,
            "exact_command_file_sha256": workflow.file_fingerprint(
                tmp_path / "exact_command.txt"
            ),
        },
    )
    for stage in workflow.STAGES:
        _semantic(
            tmp_path / "stages" / f"{stage}.json",
            {"schema": "fixture-stage", "stage": stage, "passed": 1},
        )

    monkeypatch.setattr(
        workflow, "_current_source_closure", lambda _args: (closure, closure_hash)
    )
    monkeypatch.setattr(
        workflow,
        "_verify_completed_stage",
        lambda _run_dir, _args, stage: {"stage": stage, "passed": 1},
    )
    verified = {"count": 0}
    monkeypatch.setattr(
        workflow,
        "_verify_completed_report_read_only",
        lambda *_args: verified.__setitem__("count", verified["count"] + 1),
    )
    monkeypatch.setattr(
        workflow, "_complete_pending_failure_retirement", lambda _run_dir: None
    )
    monkeypatch.setattr(
        workflow,
        "_recover_unmarked_completed_objective_resource_stop",
        lambda _run_dir, _args: None,
    )
    monkeypatch.setattr(
        workflow,
        "_pending_completed_objective_report_reserve_stop",
        lambda _run_dir: False,
    )
    monkeypatch.setattr(
        workflow, "_pending_terminal_storage_finalization", lambda _run_dir: False
    )
    monkeypatch.setattr(
        workflow,
        "_pending_completed_objective_resource_packaging",
        lambda _run_dir: False,
    )

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert workflow.main(argv) == 0
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert verified["count"] == 1
    assert after == before


def test_integrity_terminal_failure_rejects_resume_before_mutation(
    tmp_path: Path,
) -> None:
    for name in ("scientific_config.json", "exact_command.txt", "run_manifest.json"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    error = workflow.GlobalDilatedRolloutError("integrity mismatch")
    workflow._capture_failure(tmp_path, "prepare", error)
    workflow._finalize_failure(tmp_path, "prepare", error)
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    with pytest.raises(workflow.GlobalDilatedRolloutError, match="does not authorize"):
        workflow._verify_resume_compatibility(
            tmp_path, SimpleNamespace(resume_run_dir=tmp_path)
        )
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_post_capture_preterminal_integrity_crash_packages_only_then_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in ("scientific_config.json", "exact_command.txt", "run_manifest.json"):
        (tmp_path / relative).write_text("fixture\n", encoding="utf-8")
    error = workflow.GlobalDilatedRolloutError(
        "captured integrity failure before terminal"
    )
    capture = workflow._capture_failure(tmp_path, "evaluate_exact", error)
    assert capture["failure_generation"] == 0
    assert capture["failure_domain"] == "execution_integrity"
    assert capture["resume_same_frozen_run_authorized"] == 0
    assert capture["mandatory_objective_authority"][
        "scientific_objective_completed"
    ] == 0
    assert set(capture["immutable_context_file_sha256"]) == {
        "scientific_config.json",
        "exact_command.txt",
        "run_manifest.json",
    }
    assert not (tmp_path / "terminal_failure.json").exists()

    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(
        workflow,
        "_record_attempt_wall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("captured-only packaging must precede resume accounting")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_run_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("captured-only packaging must not re-enter science")
        ),
    )
    assert workflow.main([]) == 1
    terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    assert terminal["active_failure_capture_semantic_sha256"] == capture[
        "semantic_sha256"
    ]
    assert terminal["resume_same_frozen_run_authorized"] == 0
    assert workflow._verify_completed_failure_read_only(tmp_path)[
        "failure_domain"
    ] == "execution_integrity"
    ledger_path = tmp_path / "resource_ledger.json"
    if ledger_path.is_file():
        assert not any(
            event["role"] == "resume_compatibility_verification"
            for event in workflow._resource_ledger(tmp_path)["events"]
        )

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert workflow.main([]) == 1
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("failure_kind", ["keyboard_interrupt", "disk_error"])
def test_minimal_capture_precedes_last_valid_evidence_failure_and_packages_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    for relative in ("scientific_config.json", "exact_command.txt", "run_manifest.json"):
        (tmp_path / relative).write_text("fixture\n", encoding="utf-8")
    anchor = np.full(STATE_SIZE, 1.0 / STATE_SIZE, dtype=np.float64)
    atomic_rollout_npz(tmp_path / "fresh_forward/anchor-step-0127.npz", {"state": anchor})
    original_atomic_npz = workflow.atomic_rollout_npz

    def fail_last_valid(path: Path, arrays: object) -> None:
        if Path(path).name == "last_valid_states.npz":
            assert (tmp_path / "active_failure_capture.json").is_file()
            assert (tmp_path / "failure/failure.json").is_file()
            if failure_kind == "keyboard_interrupt":
                raise KeyboardInterrupt("injected death before last-valid commit")
            raise OSError("injected disk error before last-valid commit")
        original_atomic_npz(path, arrays)  # type: ignore[arg-type]

    monkeypatch.setattr(workflow, "atomic_rollout_npz", fail_last_valid)
    original_error = workflow.GlobalDilatedRolloutError(
        "original integrity failure requiring capture"
    )
    expected_exception: type[BaseException] = (
        KeyboardInterrupt if failure_kind == "keyboard_interrupt" else OSError
    )
    with pytest.raises(expected_exception, match="before last-valid commit"):
        workflow._capture_failure(tmp_path, "evaluate_exact", original_error)
    capture = workflow._read_json(
        tmp_path / "active_failure_capture.json", semantic=True
    )
    assert capture["message"] == str(original_error)
    assert capture["last_valid_states_saved_at_atomic_capture"] == 0
    assert not (tmp_path / "terminal_failure.json").exists()
    evidence_path = tmp_path / "failure/failure_evidence.json"
    assert evidence_path.is_file() == (failure_kind == "disk_error")

    monkeypatch.setattr(workflow, "atomic_rollout_npz", original_atomic_npz)
    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(
        workflow,
        "_record_attempt_wall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evidence recovery must precede resume accounting")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_run_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("evidence recovery must not re-enter science")
        ),
    )
    assert workflow.main([]) == 1
    terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    evidence = workflow._read_json(evidence_path, semantic=True)
    assert terminal["message"] == str(original_error)
    assert terminal["resume_same_frozen_run_authorized"] == 0
    assert terminal["failure_evidence_semantic_sha256"] == evidence["semantic_sha256"]
    assert evidence["last_valid_states_saved"] == 0
    assert workflow._verify_completed_failure_read_only(tmp_path)["passed"] == 1


@pytest.mark.parametrize(
    ("journal_relative", "role"),
    [
        (
            "training/active-checkpoint-interval.json",
            "global_training_checkpoint_interval",
        ),
        (
            "training/active-checkpoint-selection-validation.json",
            "checkpoint_selection_validation",
        ),
    ],
)
def test_hard_crash_attempt_journal_debits_unknown_interval_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_relative: str,
    role: str,
) -> None:
    wall_clock = iter((100.0, 112.0))
    monkeypatch.setattr(workflow.time, "time", lambda: next(wall_clock))
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 5.0)
    journal, started = workflow._begin_durable_attempt(
        tmp_path,
        journal_relative=journal_relative,
        role=role,
        detail={"fixture": 1},
    )
    assert started == 5.0
    assert (tmp_path / journal_relative).is_file()
    # Simulate process death by deliberately omitting the finish call.
    recovered = workflow._reconcile_durable_attempt_journal(
        tmp_path, journal_relative=journal_relative, role=role
    )
    assert recovered == journal
    assert not (tmp_path / journal_relative).exists()
    ledger = workflow._resource_ledger(tmp_path)
    event = ledger["events"][-1]
    assert event["role"] == role + "_abandoned_attempt"
    assert event["elapsed_seconds"] == 17.0
    assert event["detail"]["unknown_active_interval_seconds"] == 12.0
    assert event["detail"]["idle_or_powered_off_time_may_be_included"] == 1
    assert (
        event["detail"]["accounting_classification"]
        == "conservative_upper_bound_not_measured_active_compute"
    )


def test_crash_after_attempt_ledger_commit_does_not_double_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow.time, "time", lambda: 100.0)
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 1.0)
    relative = "training/active-checkpoint-interval.json"
    role = "global_training_checkpoint_interval"
    journal, _started = workflow._begin_durable_attempt(
        tmp_path,
        journal_relative=relative,
        role=role,
        detail={"start_update": 1, "end_update": 100},
    )
    workflow._record_resource_event(
        tmp_path,
        role=role,
        elapsed_seconds=9.0,
        detail={
            "attempt": journal["attempt"],
            "durable_attempt_id": journal["attempt_id"],
            "failed": 0,
        },
    )
    before = workflow._resource_ledger(tmp_path)["active_seconds"]
    workflow._reconcile_durable_attempt_journal(
        tmp_path, journal_relative=relative, role=role
    )
    assert workflow._resource_ledger(tmp_path)["active_seconds"] == before
    assert not (tmp_path / relative).exists()


def test_progress_committed_before_interval_finish_is_not_double_debited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "training/active-checkpoint-interval.json"
    role = "global_training_checkpoint_interval"
    journal, _started = workflow._begin_durable_attempt(
        tmp_path,
        journal_relative=relative,
        role=role,
        detail={"start_update": 1, "end_update": 100},
    )
    # This is the exact crash seam: progress/history is already durable, while
    # the interval event has not committed and the live start journal remains.
    history = [
        {
            "update": 100,
            "accounted_interval_seconds": 37.0,
            "peak_cuda_memory_bytes": 0,
            "total_cuda_memory_bytes": 0,
            "resource_attempt": journal["attempt"],
            "resource_attempt_id": journal["attempt_id"],
        }
    ]
    workflow._reconcile_durable_attempt_journal(
        tmp_path, journal_relative=relative, role=role
    )
    before = workflow._resource_ledger(tmp_path)
    assert len(before["events"]) == 1
    abandoned = before["events"][0]
    assert abandoned["role"] == role + "_abandoned_attempt"
    assert abandoned["detail"]["durable_attempt_id"] == journal["attempt_id"]

    workflow._restore_training_history_resource_events(
        tmp_path, history, device=torch.device("cpu")
    )
    after = workflow._resource_ledger(tmp_path)
    matching = [
        event
        for event in after["events"]
        if event.get("detail", {}).get("durable_attempt_id")
        == journal["attempt_id"]
    ]
    assert matching == [abandoned]
    assert after["active_seconds"] == before["active_seconds"]
    assert not any(event["role"] == role for event in after["events"])


def test_stale_training_interval_is_debited_before_repeated_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "training/active-checkpoint-interval.json"
    role = "global_training_checkpoint_interval"
    journal, _started = workflow._begin_durable_attempt(
        tmp_path,
        journal_relative=relative,
        role=role,
        detail={"start_update": 1, "end_update": 100},
    )
    resumed_at = float(journal["started_unix_seconds"]) + workflow.ACTIVE_SECONDS_CAP + 1.0
    monkeypatch.setattr(workflow.time, "time", lambda: resumed_at)
    store_calls = {"count": 0}

    def forbidden_store_open(*_args: object, **_kwargs: object) -> object:
        store_calls["count"] += 1
        raise AssertionError("preparation began before abandoned interval debit")

    monkeypatch.setattr(workflow, "open_external_input_store", forbidden_store_open)
    args = SimpleNamespace(training_parent=tmp_path, device="cpu")
    with pytest.raises(workflow.ResourceBoundaryError, match="durably recorded"):
        workflow._train_and_select_global(tmp_path, args)
    assert store_calls["count"] == 0
    event = workflow._resource_ledger(tmp_path)["events"][-1]
    assert event["role"] == role + "_abandoned_attempt"
    assert event["detail"]["durable_attempt_id"] == journal["attempt_id"]
    assert event["detail"]["unknown_active_interval_seconds"] >= (
        workflow.ACTIVE_SECONDS_CAP + 1.0
    )
    # The cap-crossing event is durable even though its first append raised;
    # the next reconciliation only clears the now-covered journal.
    workflow._reconcile_durable_attempt_journal(
        tmp_path, journal_relative=relative, role=role
    )
    assert not (tmp_path / relative).exists()


def test_hard_death_during_training_preparation_leaves_reconcilable_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        workflow,
        "open_external_input_store",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("injected preparation death")
        ),
    )
    args = SimpleNamespace(training_parent=tmp_path, device="cpu")
    with pytest.raises(KeyboardInterrupt, match="preparation death"):
        workflow._train_and_select_global(tmp_path, args)
    relative = "training/active-stage-preparation.json"
    journal = workflow._read_json(tmp_path / relative, semantic=True)
    assert journal["role"] == "training_store_open_target_rms_and_quartile_reference"
    workflow._reconcile_durable_attempt_journal(
        tmp_path,
        journal_relative=relative,
        role="training_store_open_target_rms_and_quartile_reference",
    )
    event = workflow._resource_ledger(tmp_path)["events"][-1]
    assert event["role"].endswith("_abandoned_attempt")
    assert event["detail"]["durable_attempt_id"] == journal["attempt_id"]
    assert event["detail"]["idle_or_powered_off_time_may_be_included"] == 1


def test_hard_death_during_validation_timing_probe_is_debited_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "open_external_input_store", lambda *_args: object())
    monkeypatch.setattr(workflow, "open_external_label_store", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(workflow, "_label_authorization", lambda *_args: {})
    calls = {"count": 0}

    def validation(*_args: object, **_kwargs: object) -> tuple[float, float]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise KeyboardInterrupt("injected validation probe death")
        return 2.0, 0.5

    monkeypatch.setattr(workflow, "_validation_mse", validation)
    kwargs = {
        "parent": tmp_path,
        "model": torch.nn.Identity(),
        "target_rms": 2.0,
        "device": torch.device("cpu"),
    }
    with pytest.raises(KeyboardInterrupt, match="validation probe death"):
        workflow._run_validation_timing_probe(tmp_path, **kwargs)
    relative = "training/active-validation-timing-probe.json"
    abandoned = workflow._read_json(tmp_path / relative, semantic=True)
    result = workflow._run_validation_timing_probe(tmp_path, **kwargs)
    assert result[:2] == (2.0, 0.5)
    assert result[3] == 1
    assert not (tmp_path / relative).exists()
    events = workflow._resource_ledger(tmp_path)["events"]
    assert [event["role"] for event in events] == [
        "validation_timing_probe_abandoned_attempt",
        "validation_timing_probe",
    ]
    assert events[0]["detail"]["durable_attempt_id"] == abandoned["attempt_id"]
    assert events[0]["detail"]["unknown_active_interval_seconds"] >= 0.0
    assert events[1]["detail"]["durable_attempt_id"] == result[4]


def test_hard_death_during_validation_calibration_replays_with_one_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_inputs = object()
    validation_labels = object()
    monkeypatch.setattr(
        workflow, "open_external_input_store", lambda *_args: validation_inputs
    )
    monkeypatch.setattr(
        workflow,
        "open_external_label_store",
        lambda *_args, **_kwargs: validation_labels,
    )
    monkeypatch.setattr(workflow, "_label_authorization", lambda *_args: {})
    calls = {"count": 0}
    reference = {
        "means": np.zeros(4, dtype=np.float64),
        "p95": np.zeros(4, dtype=np.float64),
    }

    def quartiles(*_args: object, **_kwargs: object) -> dict[str, np.ndarray]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise KeyboardInterrupt("injected validation calibration death")
        return {
            "sorted_ratios": np.zeros((4, 1), dtype=np.float64),
            "counts": np.ones(4, dtype=np.int64),
        }

    monkeypatch.setattr(workflow, "_quartile_reference", quartiles)
    kwargs = {
        "parent": tmp_path,
        "history": [],
        "training_reference": reference,
        "device": torch.device("cpu"),
    }
    with pytest.raises(KeyboardInterrupt, match="validation calibration death"):
        workflow._prepare_validation_calibration(tmp_path, **kwargs)
    relative = "training/active-validation-calibration.json"
    abandoned = workflow._read_json(tmp_path / relative, semantic=True)
    result = workflow._prepare_validation_calibration(tmp_path, **kwargs)
    assert result[0] is validation_inputs
    assert result[1] is validation_labels
    assert not (tmp_path / relative).exists()
    events = workflow._resource_ledger(tmp_path)["events"]
    assert [event["role"] for event in events] == [
        "validation_calibration_preparation_abandoned_attempt",
        "validation_calibration_preparation",
    ]
    matching = [
        event
        for event in events
        if event["detail"].get("durable_attempt_id") == abandoned["attempt_id"]
    ]
    assert len(matching) == 1
    assert events[0]["detail"]["unknown_active_interval_seconds"] >= 0.0
    assert events[1]["elapsed_seconds"] >= 5.0
    assert events[1]["peak_cuda_memory_bytes"] == 0
    assert events[1]["total_cuda_memory_bytes"] == 0


def test_hard_death_during_mandatory_postprocessing_replays_with_one_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "_aggregate_existing_shards", lambda *_args: None)
    calls = {"count": 0}

    def metrics(*_args: object) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise KeyboardInterrupt("injected mandatory image/metrics death")

    monkeypatch.setattr(workflow, "_compute_metrics_and_images", metrics)
    monkeypatch.setattr(
        workflow,
        "_classify_outcome",
        lambda *_args: {"outcome": "fixture", "required_next_action": "fixture"},
    )
    with pytest.raises(KeyboardInterrupt, match="image/metrics death"):
        workflow._run_mandatory_objective_postprocessing(
            tmp_path, SimpleNamespace()
        )
    relative = "suffix/active-mandatory-objective-postprocessing.json"
    abandoned = workflow._read_json(tmp_path / relative, semantic=True)
    outcome = workflow._run_mandatory_objective_postprocessing(
        tmp_path, SimpleNamespace()
    )
    assert outcome["outcome"] == "fixture"
    assert not (tmp_path / relative).exists()
    events = workflow._resource_ledger(tmp_path)["events"]
    assert [event["role"] for event in events] == [
        "mandatory_objective_postprocessing_abandoned_attempt",
        "mandatory_objective_postprocessing",
    ]
    matching = [
        event
        for event in events
        if event["detail"].get("durable_attempt_id") == abandoned["attempt_id"]
    ]
    assert len(matching) == 1
    assert events[1]["elapsed_seconds"] >= 5.0
    assert events[1]["peak_cuda_memory_bytes"] == 0


def test_mandatory_postprocessing_cap_stop_skips_optional_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "_require_stage", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_allocate_fresh_path",
        lambda *_args: {"fresh_path_id": 2500},
    )
    monkeypatch.setattr(workflow, "_run_forward_to_127", lambda *_args: None)
    monkeypatch.setattr(workflow, "_run_exact_suffix", lambda *_args: None)
    outcome = _semantic(
        tmp_path / "outcome.json",
        {
            "schema": "fixture-outcome",
            "outcome": "global_material_improvement",
            "required_next_action": "attempt optional path",
        },
    )
    monkeypatch.setattr(
        workflow,
        "_run_mandatory_objective_postprocessing",
        lambda *_args: (_ for _ in ()).throw(
            workflow.ResourceBoundaryError("postprocessing cap crossed")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_maybe_run_positive_complete_path",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("optional branch ran after postprocessing cap stop")
        ),
    )
    workflow._run_stage(tmp_path, SimpleNamespace(), "evaluate_exact")
    marker = workflow._read_json(
        tmp_path / "stages/evaluate_exact.json", semantic=True
    )
    positive = workflow._read_json(tmp_path / "positive_branch.json", semantic=True)
    assert marker["detail"]["outcome"] == outcome["outcome"]
    assert marker["detail"]["positive_branch_attempted"] == 0
    assert positive["attempted"] == 0
    assert positive["failure_domain"] == "resource_budget"
    assert positive["mandatory_suffix_objective_preserved"] == 1


@pytest.mark.parametrize("derived_complete", [False, True])
def test_postprocessing_reconcile_cap_cross_has_no_loop_and_routes_truthfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    derived_complete: bool,
) -> None:
    relative = "suffix/active-mandatory-objective-postprocessing.json"
    role = "mandatory_objective_postprocessing"
    journal, _started = workflow._begin_durable_attempt(
        tmp_path,
        journal_relative=relative,
        role=role,
        detail={"fixture": 1},
    )
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 1.0)
    # Exact crash seam: the abandoned-attempt cap event committed, then the
    # process died before the still-live journal could be unlinked.
    with pytest.raises(workflow.ResourceBoundaryError, match="durably recorded"):
        workflow._record_resource_event(
            tmp_path,
            role=role + "_abandoned_attempt",
            elapsed_seconds=6.0,
            detail={
                "attempt": journal["attempt"],
                "durable_attempt_id": journal["attempt_id"],
                "journal_relative": relative,
                "abandoned_hard_crash": 1,
                "wall_to_resume_upper_bound_seconds": 1.0,
                "unknown_active_interval_seconds": 1.0,
                "idle_or_powered_off_time_may_be_included": 1,
                "accounting_classification": (
                    "conservative_upper_bound_not_measured_active_compute"
                ),
                "conservative_commit_overhead_seconds": 5.0,
                "durable_committed_shard_seconds": 0.0,
                "original_detail": dict(journal["detail"]),
            },
        )
    assert (tmp_path / relative).is_file()
    monkeypatch.setattr(workflow, "_require_stage", lambda *_args: None)
    def verify_derived(*_args: object) -> dict[str, str]:
        if not derived_complete:
            raise workflow.GlobalDilatedRolloutError("derived evidence incomplete")
        return {
            "outcome": "global_material_improvement",
            "required_next_action": "fixture",
        }

    monkeypatch.setattr(
        workflow, "_verify_mandatory_postprocessing_complete", verify_derived
    )
    if derived_complete:
        _semantic(
            tmp_path / "path_usage.json",
            {"schema": "fixture-path", "fresh_path_id": 2500},
        )
        workflow._run_stage(tmp_path, SimpleNamespace(), "evaluate_exact")
        assert workflow._stage_complete(tmp_path, "evaluate_exact")
        positive = workflow._read_json(
            tmp_path / "positive_branch.json", semantic=True
        )
        assert positive["attempted"] == 0
        assert positive["mandatory_suffix_objective_preserved"] == 1
    else:
        with pytest.raises(
            workflow.PostprocessingResourceStopError,
            match="postprocessing is incomplete",
        ):
            workflow._run_stage(tmp_path, SimpleNamespace(), "evaluate_exact")
        stop = workflow._read_json(
            tmp_path / "postprocessing_resource_stop.json", semantic=True
        )
        assert stop["mandatory_postprocessing_complete"] == 0
        assert stop["optional_branch_attempted"] == 0
        assert stop["resume_same_frozen_run_authorized"] == 0
        classification = workflow._failure_classification(
            workflow.PostprocessingResourceStopError("fixture")
        )
        assert classification["failure_domain"] == "resource_budget"
        assert classification["resume_same_frozen_run_authorized"] == 0
    assert not (tmp_path / relative).exists()
    events = workflow._resource_ledger(tmp_path)["events"]
    matching = [
        event
        for event in events
        if event["detail"].get("durable_attempt_id") == journal["attempt_id"]
    ]
    assert len(matching) == 1
    assert matching[0]["role"] == role + "_abandoned_attempt"
    assert matching[0]["limits_passed"] == 0
    assert (
        workflow._reconcile_durable_attempt_journal(
            tmp_path, journal_relative=relative, role=role
        )
        is None
    )
    assert not (tmp_path / relative).exists()


def test_postprocessing_reconcile_itself_commits_cap_event_and_retires_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "suffix/active-mandatory-objective-postprocessing.json"
    role = "mandatory_objective_postprocessing"
    monkeypatch.setattr(workflow.time, "time", lambda: 100.0)
    journal, _started = workflow._begin_durable_attempt(
        tmp_path,
        journal_relative=relative,
        role=role,
        detail={"fixture": 1},
    )
    monkeypatch.setattr(workflow.time, "time", lambda: 102.0)
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 1.0)
    monkeypatch.setattr(workflow, "_require_stage", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_verify_mandatory_postprocessing_complete",
        lambda *_args: {
            "outcome": "global_material_improvement",
            "required_next_action": "fixture",
        },
    )
    _semantic(
        tmp_path / "path_usage.json",
        {"schema": "fixture-path", "fresh_path_id": 2500},
    )
    workflow._run_stage(tmp_path, SimpleNamespace(), "evaluate_exact")
    assert not (tmp_path / relative).exists()
    event = workflow._resource_ledger(tmp_path)["events"][-1]
    assert event["role"] == role + "_abandoned_attempt"
    assert event["detail"]["durable_attempt_id"] == journal["attempt_id"]
    assert event["limits_passed"] == 0
    assert workflow._stage_complete(tmp_path, "evaluate_exact")


def test_completed_evaluate_breached_ledger_resumes_packaging_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_completed_mandatory_authority(tmp_path)
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 1.0)
    with pytest.raises(workflow.ResourceBoundaryError, match="durably recorded"):
        workflow._record_resource_event(
            tmp_path,
            role="mandatory_objective_postprocessing_abandoned_attempt",
            elapsed_seconds=2.0,
            detail={"attempt": 0, "durable_attempt_id": "a" * 64},
        )
    assert workflow._pending_completed_objective_resource_packaging(tmp_path)
    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(workflow, "_initialize_run", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_record_attempt_wall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed resource packaging must precede resume accounting")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_run_stage",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("completed resource packaging must not re-enter science")
        ),
    )

    def package(run_dir: Path, _args: object) -> dict[str, int]:
        _semantic(
            run_dir / "completed_objective_resource_stop.json",
            {"schema": "fixture-stop", "scientific_objective_completed": 1},
        )
        workflow._mark_stage(run_dir, "report_verify", {"fixture": 1})
        return {"passed": 1}

    monkeypatch.setattr(workflow, "_finalize_and_verify", package)
    assert workflow.main([]) == 1
    assert workflow._stage_complete(tmp_path, "report_verify")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        workflow, "_verify_completed_report_read_only", lambda *_args: {"passed": 1}
    )
    assert workflow.main([]) == 1
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("cap_role", "expected_optional_attempted"),
    [
        ("mandatory_objective_postprocessing", 0),
        ("optional_positive_postprocessing", 1),
    ],
)
def test_unmarked_completed_objective_resource_stop_repairs_only_terminal_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cap_role: str,
    expected_optional_attempted: int,
) -> None:
    _write_completed_mandatory_authority(tmp_path)
    (tmp_path / "stages/evaluate_exact.json").unlink()
    workflow._mark_stage(tmp_path, "train_select_freeze", {"fixture": 1})
    outcome = _semantic(
        tmp_path / "outcome.json",
        {
            "schema": "fixture-outcome",
            "outcome": "global_material_improvement",
            "required_next_action": "fixture",
        },
    )
    _semantic(
        tmp_path / "path_usage.json",
        {"schema": "fixture-path", "fresh_path_id": 2500},
    )
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 1.0)
    with pytest.raises(workflow.ResourceBoundaryError, match="durably recorded"):
        workflow._record_resource_event(
            tmp_path,
            role=cap_role,
            elapsed_seconds=2.0,
            detail={"attempt": 0, "failed": 0},
        )
    monkeypatch.setattr(
        workflow,
        "_verify_mandatory_postprocessing_complete",
        lambda *_args: outcome,
    )
    recovered = workflow._recover_unmarked_completed_objective_resource_stop(
        tmp_path, SimpleNamespace()
    )
    assert recovered is not None
    assert recovered["mandatory_objective_authority"][
        "scientific_objective_completed"
    ] == 1
    positive = workflow._read_json(tmp_path / "positive_branch.json", semantic=True)
    marker = workflow._read_json(
        tmp_path / "stages/evaluate_exact.json", semantic=True
    )
    assert positive["attempted"] == expected_optional_attempted
    assert positive["completed"] == 0
    assert positive["mandatory_suffix_objective_preserved"] == 1
    assert marker["detail"]["packaging_only_recovery_after_resource_boundary"] == 1
    assert marker["detail"]["positive_branch_attempted"] == expected_optional_attempted
    assert not any(
        event["role"] == "resume_compatibility_verification"
        for event in workflow._resource_ledger(tmp_path)["events"]
    )


def test_cap_commit_then_death_before_evaluate_marker_resumes_packaging_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_completed_mandatory_authority(tmp_path)
    (tmp_path / "stages/evaluate_exact.json").unlink()
    workflow._mark_stage(tmp_path, "train_select_freeze", {"fixture": 1})
    outcome = _semantic(
        tmp_path / "outcome.json",
        {
            "schema": "fixture-outcome",
            "outcome": "global_material_improvement",
            "required_next_action": "fixture",
        },
    )
    _semantic(
        tmp_path / "path_usage.json",
        {"schema": "fixture-path", "fresh_path_id": 2500},
    )
    monkeypatch.setattr(workflow, "_require_stage", lambda *_args: None)
    monkeypatch.setattr(
        workflow, "_allocate_fresh_path", lambda *_args: {"fresh_path_id": 2500}
    )
    monkeypatch.setattr(workflow, "_run_forward_to_127", lambda *_args: None)
    monkeypatch.setattr(workflow, "_run_exact_suffix", lambda *_args: None)
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 1.0)

    def cap_after_derived(*_args: object) -> None:
        workflow._record_resource_event(
            tmp_path,
            role="mandatory_objective_postprocessing",
            elapsed_seconds=2.0,
            detail={"attempt": 0, "failed": 0},
        )

    monkeypatch.setattr(
        workflow, "_run_mandatory_objective_postprocessing", cap_after_derived
    )
    original_mark = workflow._mark_stage

    def die_before_evaluate_marker(
        run_dir: Path, stage: str, detail: object | None = None
    ) -> None:
        if stage == "evaluate_exact":
            raise KeyboardInterrupt("injected death before evaluate marker")
        original_mark(run_dir, stage, detail)  # type: ignore[arg-type]

    monkeypatch.setattr(workflow, "_mark_stage", die_before_evaluate_marker)
    with pytest.raises(KeyboardInterrupt, match="before evaluate marker"):
        workflow._run_stage(tmp_path, SimpleNamespace(), "evaluate_exact")
    assert workflow._resource_ledger(tmp_path)["limits_passed"] == 0
    assert (tmp_path / "positive_branch.json").is_file()
    assert not (tmp_path / "stages/evaluate_exact.json").exists()

    monkeypatch.setattr(workflow, "_mark_stage", original_mark)
    monkeypatch.setattr(
        workflow,
        "_verify_mandatory_postprocessing_complete",
        lambda *_args: outcome,
    )
    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(workflow, "_initialize_run", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_record_attempt_wall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("packaging recovery appended a resume event")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_run_stage",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("packaging recovery re-entered science")
        ),
    )

    def package(run_dir: Path, _args: object) -> dict[str, int]:
        assert workflow._stage_complete(run_dir, "evaluate_exact")
        _semantic(
            run_dir / "completed_objective_resource_stop.json",
            {"schema": "fixture-stop", "scientific_objective_completed": 1},
        )
        workflow._mark_stage(run_dir, "report_verify", {"fixture": 1})
        return {"passed": 1}

    monkeypatch.setattr(workflow, "_finalize_and_verify", package)
    assert workflow.main([]) == 1
    assert workflow._stage_complete(tmp_path, "evaluate_exact")
    assert workflow._stage_complete(tmp_path, "report_verify")
    assert not any(
        event["role"] == "resume_compatibility_verification"
        for event in workflow._resource_ledger(tmp_path)["events"]
    )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        workflow, "_verify_completed_report_read_only", lambda *_args: {"passed": 1}
    )
    assert workflow.main([]) == 1
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_suffix_finish_cap_cross_converts_to_raw_only_nonresumable_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fake_exact_shards(tmp_path)
    freeze = _semantic(
        tmp_path / "evaluation_freeze.json",
        {"schema": "fixture-freeze", "source_target_sha256": "a" * 64},
    )
    path_usage = _semantic(
        tmp_path / "path_usage.json",
        {
            "schema": "fixture-path",
            "fresh_path_id": workflow.FRESH_PATH_POOL[0],
            "evaluation_freeze_file_sha256": workflow.file_fingerprint(
                tmp_path / "evaluation_freeze.json"
            ),
        },
    )
    anchor_path = tmp_path / "fresh_forward/anchor-step-0127.npz"
    anchor = np.load(anchor_path)["state"]
    _semantic(
        tmp_path / "fresh_forward/forward_summary.json",
        {
            "schema": "fixture-forward",
            "path_id": path_usage["fresh_path_id"],
            "source_target_sha256": freeze["source_target_sha256"],
            "evaluation_freeze_file_sha256": workflow.file_fingerprint(
                tmp_path / "evaluation_freeze.json"
            ),
            "path_usage_file_sha256": workflow.file_fingerprint(
                tmp_path / "path_usage.json"
            ),
            "anchor_state_sha256": rollout_array_sha256(anchor),
            "anchor_archive_sha256": rollout_file_sha256(anchor_path),
            "diagnostics": {
                "passed": 1,
                "restart_chain_valid": 1,
                "authorization_fraction": 1.0,
                "certificate_fraction": 1.0,
                "forbidden_event_count": 0,
                "output_state_nonfinite_count": 0,
                "output_state_negative_count": 0,
                "maximum_output_state_mass_error": 0.0,
            },
        },
    )
    strict = {"passed": 1, "fixture_strict_telemetry": 1}
    monkeypatch.setattr(
        workflow, "_strict_fused_exact_health", lambda **_kwargs: strict
    )
    _semantic(
        tmp_path / "suffix/family_summary.json",
        {
            "schema": "fixture-family",
            "completed": 1,
            "reference_contract": "certified_exact",
            "row_order": list(workflow.ROW_ORDER),
            "strict_exact_health": strict,
            "shard_record_paths": [
                f"suffix/fused_families/fresh-five-row/suffix-128/shard-{index:04d}.json"
                for index in range(16)
            ],
            "failed_rows_suppressed": 0,
        },
    )
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 1.0)
    with pytest.raises(workflow.ResourceBoundaryError, match="durably recorded"):
        workflow._record_resource_event(
            tmp_path,
            role="mandatory_exact_five_row_suffix_attempt",
            elapsed_seconds=2.0,
            detail={
                "attempt": 0,
                "durable_attempt_id": "b" * 64,
                "failed": 0,
                "shard_count": 16,
                "row_count": len(workflow.ROW_ORDER),
            },
        )
    assert workflow._pending_raw_suffix_postprocessing_resource_stop(tmp_path)
    original = workflow.ResourceBoundaryError(
        "suffix finish resource limit crossed and durably recorded"
    )
    workflow._capture_failure(tmp_path, "evaluate_exact", original)
    workflow._finalize_failure(tmp_path, "evaluate_exact", original)
    initial_terminal = workflow._read_json(
        tmp_path / "terminal_failure.json", semantic=True
    )
    assert initial_terminal["resume_same_frozen_run_authorized"] == 1

    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(workflow, "_initialize_run", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_record_attempt_wall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("raw-only terminal conversion appended resume accounting")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_run_stage",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("raw-only terminal conversion replayed postprocessing")
        ),
    )
    original_supersede = workflow._supersede_authorized_failure
    crashed = {"done": False}

    def crash_after_supersession(run_dir: Path) -> dict[str, object]:
        record = original_supersede(run_dir)
        assert (run_dir / workflow._RAW_SUFFIX_CONVERSION_INTENT).is_file()
        if not crashed["done"]:
            crashed["done"] = True
            raise KeyboardInterrupt("injected death after raw-suffix supersession")
        return record

    monkeypatch.setattr(
        workflow, "_supersede_authorized_failure", crash_after_supersession
    )
    with pytest.raises(
        KeyboardInterrupt, match="death after raw-suffix supersession"
    ):
        workflow._convert_raw_suffix_resource_failure_to_nonresumable(tmp_path)
    intent = workflow._read_json(
        tmp_path / workflow._RAW_SUFFIX_CONVERSION_INTENT, semantic=True
    )
    assert intent["source_terminal_semantic_sha256"] == initial_terminal[
        "semantic_sha256"
    ]
    assert intent["target_failure_generation"] == 1
    assert workflow._live_terminal_failure(tmp_path) is None
    monkeypatch.setattr(
        workflow, "_supersede_authorized_failure", original_supersede
    )
    assert workflow.main([]) == 1
    terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    stop = workflow._read_json(
        tmp_path / "postprocessing_resource_stop.json", semantic=True
    )
    assert terminal["failure_code"] == (
        "raw_exact_suffix_complete_postprocessing_incomplete"
    )
    assert terminal["resume_same_frozen_run_authorized"] == 0
    assert terminal["committed_exact_suffix_shard_count"] == 16
    assert terminal["scientific_objective_completed"] == 0
    assert stop["raw_exact_suffix_complete"] == 1
    assert stop["mandatory_postprocessing_complete"] == 0
    assert stop["raw_exact_suffix_authority"]["strict_exact_health"] == strict
    assert stop["raw_suffix_conversion_intent_semantic_sha256"] == intent[
        "semantic_sha256"
    ]
    assert len(workflow._failure_supersession_records(tmp_path)) == 1
    assert not (tmp_path / workflow._RAW_SUFFIX_CONVERSION_INTENT).exists()
    assert not any(
        event["role"] == "resume_compatibility_verification"
        for event in workflow._resource_ledger(tmp_path)["events"]
    )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert workflow.main([]) == 1
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_next_invocation_finishes_crashed_failure_packaging_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("scientific_config.json", "exact_command.txt", "run_manifest.json"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    error = workflow.GlobalDilatedRolloutError("integrity packaging crash")
    workflow._capture_failure(tmp_path, "prepare", error)
    original_text = workflow._atomic_text
    crashed = {"done": False}

    def crash_after_terminal(path: Path, text: str) -> None:
        if Path(path).name == "REPORT.md" and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("injected packaging crash")
        original_text(path, text)

    monkeypatch.setattr(workflow, "_atomic_text", crash_after_terminal)
    with pytest.raises(RuntimeError, match="injected packaging crash"):
        workflow._finalize_failure(tmp_path, "prepare", error)
    assert (tmp_path / "terminal_failure.json").is_file()
    assert not (tmp_path / "verification.json").exists()
    monkeypatch.setattr(workflow, "_atomic_text", original_text)
    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(workflow, "_initialize_run", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_run_stage",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("failure recovery must not re-enter scientific stages")
        ),
    )
    assert workflow.main([]) == 1
    verification = workflow._verify_completed_failure_read_only(tmp_path)
    assert verification["partial_failure_evidence_packaged"] == 1
    assert verification["failure_domain"] == "execution_integrity"


def test_failure_finalizer_seals_partial_exact_prefix(tmp_path: Path) -> None:
    root = tmp_path / "suffix/fused_families/fresh-five-row/suffix-128"
    state = np.full((len(workflow.ROW_ORDER), STATE_SIZE), 1.0 / STATE_SIZE)
    atomic_rollout_npz(root / "shard-0000.npz", {"state": state})
    _semantic(
        root / "shard-0000.json",
        {"schema": "fixture-shard", "committed": 1, "elapsed_seconds": 2.0},
    )
    error = workflow.ResourceBoundaryError("cannot preserve report reserve")
    workflow._capture_failure(tmp_path, "evaluate_exact", error)
    verification = workflow._finalize_failure(tmp_path, "evaluate_exact", error)
    assert verification["scientific_objective_completed"] == 0
    assert verification["failure_domain"] == "resource_budget"
    terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    assert terminal["committed_exact_suffix_shard_count"] == 1
    manifest = workflow._read_json(tmp_path / "artifact_manifest.json", semantic=True)
    assert {row["path"] for row in manifest["artifacts"]} == {
        row["path"] for row in workflow._manifest_rows(tmp_path)
    }
    checksum_lines = (tmp_path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    checksum_paths = set()
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        checksum_paths.add(relative)
        assert workflow.file_fingerprint(tmp_path / relative) == digest
    assert checksum_paths == {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS.txt", "verification.json"}
        and path.suffix != ".tmp"
    }


def _write_completed_mandatory_authority(run_dir: Path) -> None:
    root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
    for index in range(16):
        _semantic(
            root / f"shard-{index:04d}.json",
            {"schema": "fixture-shard", "committed": 1, "elapsed_seconds": 1.0},
        )
    _semantic(
        run_dir / "suffix/family_summary.json",
        {
            "schema": "fixture-family",
            "completed": 1,
            "row_order": list(workflow.ROW_ORDER),
            "strict_exact_health": {"passed": 1},
        },
    )
    workflow._mark_stage(run_dir, "evaluate_exact", {"fixture": 1})


def test_failure_finalizer_preserves_completed_mandatory_objective(
    tmp_path: Path,
) -> None:
    _write_completed_mandatory_authority(tmp_path)
    error = workflow.ResourceBoundaryError("report resource budget exhausted")
    workflow._capture_failure(tmp_path, "report_verify", error)
    verification = workflow._finalize_failure(tmp_path, "report_verify", error)
    terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    assert verification["scientific_objective_completed"] == 1
    assert terminal["scientific_objective_completed"] == 1
    assert terminal["committed_exact_suffix_shard_count"] == 16
    assert terminal["failure_domain"] == "resource_budget"
    assert "remains scientifically authoritative" in (
        tmp_path / "REPORT.md"
    ).read_text(encoding="utf-8")


def test_report_reserve_crossing_seals_completed_objective_resource_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "ACTIVE_SECONDS_CAP", 100.0)
    _write_completed_mandatory_authority(tmp_path)
    _semantic(
        tmp_path / "positive_branch.json",
        {
            "schema": "fixture-positive",
            "triggered": 1,
            "attempted": 1,
            "completed": 0,
            "reason": "optional branch stopped",
        },
    )
    workflow._record_resource_event(
        tmp_path, role="fixture-science", elapsed_seconds=50.0, detail={"attempt": 0}
    )
    monkeypatch.setattr(workflow, "_write_reports", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_verify_raw_and_derived",
        lambda *_args: {"fixture_deep_verification": 1},
    )
    monkeypatch.setattr(
        workflow, "_verify_completed_report_read_only", lambda *_args: {"passed": 1}
    )
    verification = workflow._finalize_and_verify(tmp_path, SimpleNamespace())
    stop = workflow._read_json(
        tmp_path / "completed_objective_resource_stop.json", semantic=True
    )
    ledger = workflow._resource_ledger(tmp_path)
    assert verification["scientific_objective_completed"] == 1
    assert verification["resource_limits_passed"] == 0
    assert stop["scientific_objective_completed"] == 1
    assert stop["mandatory_objective_authority"][
        "committed_exact_suffix_shard_count"
    ] == 16
    assert stop["optional_branch"] == {
        "triggered": 1,
        "attempted": 1,
        "completed": 0,
        "reason": "optional branch stopped",
    }
    assert ledger["limits_passed"] == 0
    assert "active_seconds_cap" in ledger["breaches"]


@pytest.mark.parametrize("crosses_during_assembly", [False, True])
def test_terminal_storage_fixed_point_is_exact_and_truthful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crosses_during_assembly: bool,
) -> None:
    real_completed_verifier = workflow._verify_completed_report_read_only
    _write_completed_mandatory_authority(tmp_path)
    _semantic(
        tmp_path / "positive_branch.json",
        {
            "schema": "fixture-positive",
            "triggered": 0,
            "attempted": 0,
            "completed": 0,
            "reason": "not triggered",
        },
    )
    initial_bytes = workflow._directory_bytes(tmp_path)
    monkeypatch.setattr(
        workflow,
        "STORAGE_CAP_BYTES",
        initial_bytes + (4096 if crosses_during_assembly else 1_000_000),
    )

    def write_large_final_reports(run_dir: Path, _args: object) -> None:
        workflow._atomic_text(run_dir / "REPORT.md", "R" * 4096)
        workflow._atomic_text(run_dir / "HANDOFF.md", "H" * 4096)

    monkeypatch.setattr(workflow, "_write_reports", write_large_final_reports)
    def write_fixture_inventories(run_dir: Path, _args: object) -> dict[str, int]:
        workflow._write_semantic(
            run_dir / "artifact_manifest.json",
            {
                "schema": "fixture-manifest",
                "artifact_count": 0,
                "artifacts": [],
            },
        )
        workflow._atomic_text(run_dir / "SHA256SUMS.txt", "")
        return {"fixture_deep_verification": 1}

    monkeypatch.setattr(
        workflow, "_verify_raw_and_derived", write_fixture_inventories
    )
    monkeypatch.setattr(
        workflow, "_verify_completed_report_read_only", lambda *_args: {"passed": 1}
    )
    verification = workflow._finalize_and_verify(tmp_path, SimpleNamespace())
    authority = workflow._read_json(
        tmp_path / "terminal_storage_authority.json", semantic=True
    )
    ledger = workflow._resource_ledger(tmp_path)
    exact_bytes = workflow._directory_bytes(tmp_path)
    assert authority["exact_recursive_file_bytes"] == exact_bytes
    assert ledger["persisted_storage_bytes"] == exact_bytes
    assert verification["terminal_storage_authority_semantic_sha256"] == authority[
        "semantic_sha256"
    ]
    reserve = next(
        event
        for event in ledger["events"]
        if event["role"] == "report_and_final_verification_reserved_charge"
    )
    terminal = next(
        event
        for event in ledger["events"]
        if event["role"] == workflow._TERMINAL_STORAGE_ROLE
    )
    assert reserve["limits_passed"] == 1
    assert terminal["detail"]["exact_final_storage_authority"] == 1
    stop_path = tmp_path / "completed_objective_resource_stop.json"
    assert stop_path.is_file() == crosses_during_assembly
    assert ledger["limits_passed"] == int(not crosses_during_assembly)
    assert ("persisted_storage_cap" in ledger["breaches"]) == crosses_during_assembly
    if crosses_during_assembly:
        stop = workflow._read_json(stop_path, semantic=True)
        assert stop["scientific_objective_completed"] == 1
        assert stop["exact_final_persisted_storage_bytes"] == exact_bytes

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    workflow._finalize_and_verify(tmp_path, SimpleNamespace())
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    monkeypatch.setattr(
        workflow, "_verify_scientific_evidence_read_only", lambda *_args: {}
    )
    monkeypatch.setattr(workflow, "_directory_bytes", lambda _path: exact_bytes + 1)
    with pytest.raises(
        workflow.GlobalDilatedRolloutError,
        match="completed objective/resource status changed",
    ):
        real_completed_verifier(tmp_path, SimpleNamespace())


@pytest.mark.parametrize(
    ("breach_kind", "expected_breach"),
    [
        ("active", "active_seconds_cap"),
        ("storage", "persisted_storage_cap"),
        ("cuda", "cuda_memory_fraction_cap"),
    ],
)
def test_report_reserve_cap_commit_crash_resumes_packaging_before_any_resume_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    breach_kind: str,
    expected_breach: str,
) -> None:
    monkeypatch.setattr(
        workflow, "ACTIVE_SECONDS_CAP", 100.0 if breach_kind == "active" else 10_000.0
    )
    if breach_kind == "storage":
        monkeypatch.setattr(workflow, "STORAGE_CAP_BYTES", 1)
    _write_completed_mandatory_authority(tmp_path)
    _semantic(
        tmp_path / "positive_branch.json",
        {
            "schema": "fixture-positive",
            "triggered": 1,
            "attempted": 0,
            "completed": 0,
            "reason": "not admitted",
        },
    )
    if breach_kind == "active":
        workflow._record_resource_event(
            tmp_path,
            role="fixture-science",
            elapsed_seconds=50.0,
            detail={"attempt": 0},
        )
    else:
        with pytest.raises(workflow.ResourceBoundaryError, match="durably recorded"):
            workflow._record_resource_event(
                tmp_path,
                role="fixture-subordinate-resource-stop",
                elapsed_seconds=0.0,
                peak_cuda_memory_bytes=80 if breach_kind == "cuda" else 0,
                total_cuda_memory_bytes=100 if breach_kind == "cuda" else 0,
                detail={"attempt": 0, "subordinate_optional_stop": 1},
            )
    monkeypatch.setattr(workflow, "_write_reports", lambda *_args: None)
    monkeypatch.setattr(
        workflow,
        "_verify_raw_and_derived",
        lambda *_args: {"fixture_deep_verification": 1},
    )
    monkeypatch.setattr(
        workflow, "_verify_completed_report_read_only", lambda *_args: {"passed": 1}
    )
    original_write_semantic = workflow._write_semantic
    crashed = {"done": False}

    def crash_before_stop_sentinel(path: Path, body: object) -> dict[str, object]:
        if (
            Path(path).name == "completed_objective_resource_stop.json"
            and not crashed["done"]
        ):
            crashed["done"] = True
            raise RuntimeError("injected crash before resource-stop sentinel")
        return original_write_semantic(path, body)  # type: ignore[arg-type]

    monkeypatch.setattr(workflow, "_write_semantic", crash_before_stop_sentinel)
    with pytest.raises(RuntimeError, match="before resource-stop sentinel"):
        workflow._finalize_and_verify(tmp_path, SimpleNamespace())
    ledger = workflow._resource_ledger(tmp_path)
    assert ledger["limits_passed"] == 0
    assert ledger["breaches"] == [expected_breach]
    assert any(
        event["role"] == "report_and_final_verification_reserved_charge"
        and event["limits_passed"] == 0
        for event in ledger["events"]
    )
    assert not (tmp_path / "completed_objective_resource_stop.json").exists()
    assert workflow._pending_completed_objective_report_reserve_stop(tmp_path)

    monkeypatch.setattr(workflow, "_write_semantic", original_write_semantic)
    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(workflow, "_initialize_run", lambda *_args: None)
    assert workflow.main([]) == 1
    assert (tmp_path / "completed_objective_resource_stop.json").is_file()
    assert workflow._stage_complete(tmp_path, "report_verify")
    ledger = workflow._resource_ledger(tmp_path)
    assert not any(
        event["role"] == "resume_compatibility_verification"
        for event in ledger["events"]
    )

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert workflow.main([]) == 1
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_resumable_failure_supersession_survives_second_crash_then_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = workflow.ResourceBoundaryError("identical resource boundary")
    workflow._capture_failure(tmp_path, "evaluate_exact", first)
    workflow._finalize_failure(tmp_path, "evaluate_exact", first)
    first_terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    first_supersession = workflow._supersede_authorized_failure(tmp_path)
    assert workflow._live_terminal_failure(tmp_path) is None
    assert not (tmp_path / "terminal_failure.json").exists()
    assert (tmp_path / "failure_history/retirement-0000.json").is_file()
    assert (
        first_supersession["terminal_failure_semantic_sha256"]
        == first_terminal["semantic_sha256"]
    )

    # The resumed attempt crashes again and seals a new, current package.
    second = workflow.ResourceBoundaryError("identical resource boundary")
    workflow._capture_failure(tmp_path, "evaluate_exact", second)
    workflow._finalize_failure(tmp_path, "evaluate_exact", second)
    second_terminal = workflow._live_terminal_failure(tmp_path)
    assert second_terminal is not None
    assert second_terminal["message"] == "identical resource boundary"
    assert second_terminal["failure_generation"] == 1
    assert second_terminal["semantic_sha256"] != first_terminal["semantic_sha256"]
    second_supersession = workflow._supersede_authorized_failure(tmp_path)
    assert second_supersession["sequence_index"] == 1
    assert (
        second_supersession["previous_supersession_semantic_sha256"]
        == first_supersession["semantic_sha256"]
    )
    assert workflow._live_terminal_failure(tmp_path) is None
    assert (tmp_path / "failure_history/retirement-0001.json").is_file()

    # Emulate eventual successful finalization.  Historical failures remain
    # append-only, while there is no active terminal ambiguity.  A subsequent
    # completed verifier invocation is byte-identical.
    _semantic(
        tmp_path / "verification.json",
        {"schema": "fixture-success-verification", "passed": 1},
    )
    workflow._mark_stage(tmp_path, "report_verify", {"fixture_success": 1})
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(workflow, "_initialize_run", lambda *_args: None)
    monkeypatch.setattr(
        workflow, "_verify_completed_report_read_only", lambda *_args: {"passed": 1}
    )
    assert workflow.main([]) == 0
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert workflow._live_terminal_failure(tmp_path) is None


def test_retired_failure_record_cannot_block_next_generation_capture_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = workflow.ResourceBoundaryError("generation-zero resource boundary")
    workflow._capture_failure(tmp_path, "evaluate_exact", first)
    workflow._finalize_failure(tmp_path, "evaluate_exact", first)
    supersession = workflow._supersede_authorized_failure(tmp_path)
    assert supersession["failure_generation"] == 0
    assert (tmp_path / "failure_history/retirement-0000.json").is_file()
    assert not (tmp_path / "failure/failure.json").exists()

    original_writer = workflow._write_failure_record_from_capture

    def die_after_minimal_capture(
        _run_dir: Path, capture: object
    ) -> dict[str, object]:
        assert workflow._read_json(
            tmp_path / "active_failure_capture.json", semantic=True
        )["failure_generation"] == 1
        assert not (tmp_path / "failure/failure.json").exists()
        raise KeyboardInterrupt("injected death before generation-one failure record")

    monkeypatch.setattr(
        workflow, "_write_failure_record_from_capture", die_after_minimal_capture
    )
    second = workflow.ResourceBoundaryError("generation-one resource boundary")
    with pytest.raises(KeyboardInterrupt, match="generation-one failure record"):
        workflow._capture_failure(tmp_path, "evaluate_exact", second)
    capture = workflow._live_failure_capture(tmp_path)
    assert capture is not None and capture["failure_generation"] == 1
    assert not (tmp_path / "failure/failure.json").exists()

    monkeypatch.setattr(
        workflow, "_write_failure_record_from_capture", original_writer
    )
    args = SimpleNamespace(resume_run_dir=tmp_path)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_paths", lambda _args: None)
    monkeypatch.setattr(workflow, "_make_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(
        workflow,
        "_record_attempt_wall",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("captured generation recovery must precede resume accounting")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_run_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("captured generation recovery must not re-enter science")
        ),
    )
    assert workflow.main([]) == 1
    terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    assert terminal["failure_generation"] == 1
    assert terminal["message"] == str(second)
    assert workflow._verify_completed_failure_read_only(tmp_path)["passed"] == 1


def test_completed_objective_failure_packaging_crash_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_completed_mandatory_authority(tmp_path)
    error = workflow.GlobalDilatedRolloutError("integrity failure after objective")
    workflow._capture_failure(tmp_path, "report_verify", error)
    original_text = workflow._atomic_text
    crashed = {"done": False}

    def crash_after_terminal(path: Path, text: str) -> None:
        if Path(path).name == "REPORT.md" and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("injected completed-objective packaging crash")
        original_text(path, text)

    monkeypatch.setattr(workflow, "_atomic_text", crash_after_terminal)
    with pytest.raises(RuntimeError, match="completed-objective packaging crash"):
        workflow._finalize_failure(tmp_path, "report_verify", error)
    terminal = workflow._read_json(tmp_path / "terminal_failure.json", semantic=True)
    assert terminal["scientific_objective_completed"] == 1
    monkeypatch.setattr(workflow, "_atomic_text", original_text)
    recovered = workflow._recover_incomplete_failure_package(tmp_path)
    assert recovered["scientific_objective_completed"] == 1
    assert workflow._verify_completed_failure_read_only(tmp_path)[
        "scientific_objective_completed"
    ] == 1
