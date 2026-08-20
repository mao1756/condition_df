from __future__ import annotations

import ast
import contextlib
import csv
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist import diag_d0_eulerian_edge_flux_replay as runner
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    flux_divergence_torch,
    simulate_direct_flux_generation,
)


def _mass_image(total: int) -> np.ndarray:
    flat = np.full(784, total // 784, dtype=np.uint8)
    flat[: total % 784] += 1
    assert int(flat.sum()) == total
    return flat.reshape(28, 28)


def _tiny_config(**overrides: object) -> DirectFluxMNISTConfig:
    values: dict[str, object] = {
        "grid_size": 4,
        "source_lowfreq_size": 2,
        "ot_lowres_size": 2,
        "source_blur_sigma": 0.0,
        "num_steps": 1,
        "adaptive_sampling": True,
        "clip_target": 0.03,
        "max_substeps": 4,
        "free_weight": 0.015,
        "noise_weight": 0.0,
        "learned_weight": 1.0,
    }
    values.update(overrides)
    return DirectFluxMNISTConfig(**values)


def _synthetic_population_run(root: Path) -> Path:
    runner._write_json(
        root / "config.json",
        {
            "command": "synthetic",
            "execution_authority": {"approval_id": "test-only", "device": "cpu"},
        },
    )
    authority = {
        "schema": runner.VERSION + "-mass-to-uint8",
        "derivation_slice": [0, 55_000],
        "central_sums": [25_470, 25_472],
        "numerator": runner.MASS_SCALE_NUMERATOR,
        "denominator": runner.MASS_SCALE_DENOMINATOR,
        "decimal": 25_471 / 255,
        "float_hex": runner.MASS_SCALE_HEX,
    }
    runner._write_json(root / "input_bindings" / "mass_to_uint8.json", authority)
    inventory = runner.build_path_inventory()
    labels = inventory["requested_labels"].astype(np.int64)
    path_ids = inventory["path_ids"].astype(np.str_)
    base = np.full((160, 784), np.float32(1 / 784), dtype=np.float32)
    for index in range(160):
        delta = np.float32((index + 1) * 1e-7)
        base[index, index] += delta
        base[index, (index + 1) % 784] -= delta
    anchors = np.repeat(base[None], len(runner.ANCHORS), axis=0)
    runner._write_npz(
        root / "inventory" / "start_bank.npz",
        starts=base,
        labels=labels,
        path_ids=path_ids,
        source_seeds=inventory["source_seeds"],
    )
    runner._write_json(
        root / "inventory" / "START_BANK_SEALED.json",
        {
            "start_bank_sha256": runner.sha256_file(root / "inventory" / "start_bank.npz"),
            "starts_sha256": runner._hash_array(base),
            "labels_sha256": runner._hash_array(labels),
            "path_ids_sha256": runner._hash_array(path_ids),
        },
    )
    for row_index, row in enumerate(("teacher", "null", "learned")):
        row_anchors = anchors.copy()
        row_anchors[-1] = np.roll(row_anchors[-1], row_index, axis=1)
        proposed_per_step = runner.PATH_COUNT * 2 * 28 * 28
        telemetry = [
            {
                "row": row,
                "completed_step": step,
                "accepted_substeps": 1,
                "rejected_attempt_count": 0,
                "attempts": [
                    {
                        "substeps": 1,
                        "clipped": 0,
                        "proposed": proposed_per_step,
                        "clipping_fraction": 0.0,
                    }
                ],
                "accepted_clipped": 0,
                "accepted_proposed": proposed_per_step,
                "accepted_clipping_fraction": 0.0,
                "learned_step_rms": 0.0,
                "free_step_rms": 0.0,
                "noise_step_rms": 0.0,
                "state_increment_rms": 0.0,
                "minimum_mass": float(row_anchors[-1].min()),
                "maximum_mass": float(row_anchors[-1].max()),
                "maximum_mass_error": 0.0,
                "nonfinite_count": 0,
                "elapsed_seconds": 0.0,
                "cuda_allocated_bytes": 0,
                "cuda_peak_allocated_bytes": 0,
            }
            for step in range(1, runner.OUTER_STEPS + 1)
        ]
        result = runner.RowResult(
            row=row,
            anchors=row_anchors,
            labels=labels,
            path_ids=path_ids,
            root_seed=runner.ROW_ROOT_SEEDS[row],
            telemetry=telemetry,
            scientific_digest=runner._scientific_row_digest(row_anchors, telemetry),
        )
        runner._save_row_result(root, result)
    return root


def _write_synthetic_review_authorities(root: Path) -> None:
    runner._write_json(root / "status.json", {"state": "awaiting_human_review"})
    runner._write_json(
        root / "config.json",
        {
            "command": "synthetic",
            "execution_authority": {"approval_id": "test-only", "device": "cpu"},
        },
    )
    runner._write_json(
        root / "gates.json",
        {
            "schema": runner.VERSION + "-gates",
            **{
                name: {"passed": 1, "gate_type": "execution/integrity"}
                for name in ("gate_a", "gate_b", "gate_c", "gate_d")
            },
            "gate_e": {
                "gate_type": "diagnostic threshold",
                "state": "pending",
                "passed": None,
                "conditions": {},
            },
        },
    )
    for row, accuracy in (("teacher", 0.95), ("learned", 0.9), ("null", 0.1)):
        runner._write_json(
            root / "evaluation" / f"{row}_metrics.json",
            {
                "classifier": {"requested_label_accuracy": accuracy},
                "duplicates": {"duplicate_pair_count": 0},
                "diversity": {"aggregate_median_ratio": 0.5},
            },
        )
    runner._write_json(root / "data" / "test_open_event.json", {"opened": 1})
    runner._write_npz(root / "evaluation" / "predictions.npz", values=np.zeros(1))
    runner._write_json(
        root / "controls" / "teacher_gate.json",
        {
            "passed": 1,
            "median_relative_squared_l2_anchor64": 0.5,
            "median_relative_squared_l2_endpoint": 0.1,
            "endpoint_improved_path_count": 160,
            "teacher_requested_label_accuracy": 0.95,
        },
    )
    runner._write_npz(
        root / "controls" / "teacher_gate_arrays.npz",
        initial_squared_l2=np.ones(runner.PATH_COUNT, dtype=np.float64),
        endpoint_squared_l2=np.full(runner.PATH_COUNT, 0.1, dtype=np.float64),
    )
    runner._write_json(
        root / "evaluation" / "SCORING_READY.json",
        {
            "schema": runner.VERSION + "-scoring-ready",
            "population_seal_sha256": runner.sha256_file(
                root / "populations" / "POPULATIONS_SEALED.json"
            ),
            "test_open_event_sha256": runner.sha256_file(root / "data" / "test_open_event.json"),
            "predictions_sha256": runner.sha256_file(root / "evaluation" / "predictions.npz"),
            "metrics_sha256": {
                row: runner.sha256_file(root / "evaluation" / f"{row}_metrics.json")
                for row in ("teacher", "null", "learned")
            },
            "teacher_control_sha256": runner.sha256_file(root / "controls" / "teacher_gate.json"),
            "teacher_control_passed": 1,
        },
    )
    runner._write_json(
        root / "resource_ledger.json",
        {
            "schema": runner.VERSION + "-resource-ledger",
            "budget": dataclasses.asdict(runner.ResourceBudget()),
            "active_seconds": 7.25,
            "events": [
                {
                    "kind": "machine_terminalization",
                    "predicted_seconds": 5.0,
                    "predicted_next_bytes": 1,
                    "active_seconds_before": 0.0,
                    "reserve_remaining_seconds": 0.0,
                    "storage_bytes_before": 0,
                    "cuda_allocated_bytes": 0,
                    "cuda_total_bytes": 0,
                    "cuda_fraction": 0.0,
                    "checks": {
                        "active": True,
                        "storage": True,
                        "cuda": True,
                        "quantum": True,
                    },
                    "passed": 1,
                    "event": "admit",
                    "recorded_at": "synthetic-machine-admit",
                },
                {
                    "event": "complete",
                    "kind": "machine_terminalization",
                    "elapsed_seconds": 7.25,
                    "active_seconds_after": 7.25,
                    "storage_bytes_after": 0,
                    "cuda_allocated_bytes": 0,
                    "cuda_total_bytes": 0,
                    "cuda_fraction": 0.0,
                    "candidate_transitions": 0,
                    "model_evaluations": 0,
                    "recorded_at": "synthetic-machine-complete",
                },
            ],
            "failed_admission": None,
            "open_events": [],
        },
    )


def _write_valid_review_answers(run_dir: Path, path: Path) -> None:
    key = runner._read_json(run_dir / "review" / "review_key.json")["entries"]
    membership = {
        entry["member_id"]: entry
        for entry in runner._read_json(run_dir / "review" / "private_membership.json")[
            "entries"
        ]
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["review_order", "sample_id", "assigned_label", "notes"],
        )
        writer.writeheader()
        for entry in key:
            member = membership[entry["source_sample_id"]]
            writer.writerow(
                {
                    "review_order": entry["review_order"],
                    "sample_id": entry["sample_id"],
                    "assigned_label": (
                        str(member["requested_label"])
                        if member["row"] == "learned"
                        else "noise"
                    ),
                    "notes": "manual",
                }
            )


def _patch_verifier_to_population_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {"state": "resource_stopped", "route": "resource_stopped"}
    monkeypatch.setattr(
        runner,
        "_verify_stage_and_status",
        lambda root: (
            status,
            [
                "checkpoint_extract",
                "data_and_inventory",
                "teacher_row",
                "null_row",
                "learned_row",
            ],
            "population_seal",
        ),
    )
    monkeypatch.setattr(runner, "_verify_lifecycle_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_resource_ledger", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_verify_config_and_sources", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_verify_checkpoint_extract", lambda *args, **kwargs: _tiny_config())
    monkeypatch.setattr(
        runner,
        "_verify_data_and_inventory",
        lambda *args, **kwargs: (np.empty((0, 784), dtype=np.uint8), np.empty(0, dtype=np.int64)),
    )
    monkeypatch.setattr(runner, "_verify_report_contract", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_telemetry_summary", lambda *args, **kwargs: None)


def test_frozen_scale_uses_training_slice_and_exact_float64_hex() -> None:
    scale = np.float64(runner.MASS_SCALE_NUMERATOR) / np.float64(runner.MASS_SCALE_DENOMINATOR)
    assert (runner.TRAIN_START, runner.TRAIN_STOP) == (0, 55_000)
    assert (runner.MASS_SCALE_NUMERATOR, runner.MASS_SCALE_DENOMINATOR) == (25_471, 255)
    assert scale == np.float64(25_471 / 255)
    assert scale.hex() == runner.MASS_SCALE_HEX == "0x1.8f8b8b8b8b8b9p+6"
    # The planning document's shortened spelling is a different float.
    assert np.float64.fromhex("0x1.8f8b8b8b8b9p+6") != scale


def test_mass_authority_and_raster_are_global_not_per_image() -> None:
    train = np.empty((55_000, 28, 28), dtype=np.uint8)
    train[:27_500] = _mass_image(25_470)
    train[27_500:] = _mass_image(25_472)
    authority = runner.derive_mass_to_uint8_authority(train)
    assert authority["derivation_slice"] == [0, 55_000]
    assert authority["central_sums"] == [25_470, 25_472]
    assert authority["numerator"] == 25_471
    assert authority["denominator"] == 255
    assert authority["float_hex"] == "0x1.8f8b8b8b8b8b9p+6"

    masses = np.zeros((2, 784), dtype=np.float32)
    masses[0, :2] = [0.25, 0.75]
    masses[1, :2] = [0.50, 0.50]
    rendered = runner.mass_to_uint8(masses, authority)
    expected = np.rint(np.clip(25_471 * masses.astype(np.float64), 0, 255)).astype(np.uint8)
    assert rendered.shape == (2, 28, 28)
    assert np.array_equal(rendered.reshape(2, 784), expected)
    assert int(rendered[0].max()) == 255 and int(rendered[1].max()) == 255

    raw = np.arange(1, 785, dtype=np.float64)
    normalized = raw / raw.sum()
    rescaled_then_normalized = (raw * 17.0) / (raw * 17.0).sum()
    assert np.array_equal(
        runner.mass_to_uint8(normalized[None], authority),
        runner.mass_to_uint8(rescaled_then_normalized[None], authority),
    )


@pytest.mark.parametrize(
    ("mismatch", "observation_key", "failed_stage"),
    [
        ("missing_checkpoint", "legacy_checkpoint", "checkpoint_extract"),
        ("missing_arff", "arff", "data_and_inventory"),
        ("wrong_k128", "k128_run_dir", "initialize_and_bind"),
    ],
)
def test_early_input_mismatch_is_observed_recomputed_and_verifier_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    observation_key: str,
    failed_stage: str,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    checkpoint = inputs / "legacy.pt"
    checkpoint.write_bytes(b"synthetic-legacy-checkpoint")
    arff = inputs / "mnist.arff"
    arff.write_bytes(b"synthetic-arff")
    k128 = inputs / "k128"
    ddpm = inputs / "ddpm"
    (ddpm / "evaluator").mkdir(parents=True)
    k128.mkdir()
    k128_files = {
        "artifact_manifest.json": b"synthetic-k128-manifest",
        "status.json": runner._canonical_json_bytes({"route": "complete"}),
        "outcome.json": runner._canonical_json_bytes(
            {
                "route": runner.K128_REQUIRED_ROUTE,
                "full_scale_auto_launched": 0,
            }
        ),
        "REPORT.md": b"synthetic K128 report\n",
    }
    for relative, payload in k128_files.items():
        (k128 / relative).write_bytes(payload)
    ddpm_files = {
        "artifact_manifest.json": b"synthetic-ddpm-manifest",
        "evaluator/selection.json": b"synthetic-selection",
        "evaluator/selected_checkpoint.pt": b"synthetic-evaluator",
    }
    for relative, payload in ddpm_files.items():
        (ddpm / relative).write_bytes(payload)

    def bind_file_constants(prefix: str, path: Path) -> None:
        monkeypatch.setattr(runner, prefix + "_BYTES", path.stat().st_size)
        monkeypatch.setattr(runner, prefix + "_SHA256", runner.sha256_file(path))

    bind_file_constants("LEGACY_CHECKPOINT", checkpoint)
    bind_file_constants("MNIST_ARFF", arff)
    bind_file_constants("K128_MANIFEST", k128 / "artifact_manifest.json")
    bind_file_constants("K128_STATUS", k128 / "status.json")
    bind_file_constants("K128_OUTCOME", k128 / "outcome.json")
    bind_file_constants("K128_REPORT", k128 / "REPORT.md")
    bind_file_constants("DDPM_MANIFEST", ddpm / "artifact_manifest.json")
    bind_file_constants("EVALUATOR_SELECTION", ddpm / "evaluator" / "selection.json")
    bind_file_constants("EVALUATOR", ddpm / "evaluator" / "selected_checkpoint.pt")
    monkeypatch.setattr(runner, "K128_TREE_DIGEST", "a" * 64)
    monkeypatch.setattr(runner, "DDPM_TREE_DIGEST", "b" * 64)

    def verify_synthetic_manifest(
        path: Path,
        *,
        expected_manifest_sha256: str,
        expected_tree_digest: str,
    ) -> dict[str, object]:
        manifest = Path(path) / "artifact_manifest.json"
        if not manifest.is_file():
            raise runner.IntegrityFailure("synthetic predecessor manifest is absent")
        assert runner.sha256_file(manifest) == expected_manifest_sha256
        return {
            "manifest_sha256": expected_manifest_sha256,
            "tree_digest": expected_tree_digest,
            "passed": 1,
        }

    monkeypatch.setattr(runner, "_verify_external_manifest", verify_synthetic_manifest)
    scientific_config = _tiny_config(grid_size=28)

    def fake_extract(source: Path, clean_state: Path, **kwargs: object) -> dict[str, object]:
        clean = Path(clean_state)
        clean.parent.mkdir(parents=True, exist_ok=True)
        clean.write_bytes(b"synthetic-clean-state")
        return {
            "config": dataclasses.asdict(scientific_config),
            "clean_state_bytes": clean.stat().st_size,
            "clean_state_sha256": runner.sha256_file(clean),
            "tensor_count": 1,
            "parameter_count": 1,
        }

    if mismatch != "missing_checkpoint":
        monkeypatch.setattr(runner, "safe_extract_legacy_checkpoint", fake_extract)
    monkeypatch.setattr(
        runner,
        "_verify_checkpoint_extract",
        lambda *args, **kwargs: scientific_config,
    )
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        runner.ResourceGovernor,
        "_cuda_receipt",
        lambda self: (0, 1, 0.0),
    )

    selected_checkpoint = checkpoint
    selected_arff = arff
    selected_k128 = k128
    if mismatch == "missing_checkpoint":
        selected_checkpoint = inputs / "missing-legacy.pt"
    elif mismatch == "missing_arff":
        selected_arff = inputs / "missing-mnist.arff"
    else:
        selected_k128 = inputs / "wrong-k128"

    run_dir = tmp_path / "run"
    args = _production_args(run_dir)
    args.legacy_checkpoint = str(selected_checkpoint)
    args.arff = str(selected_arff)
    args.k128_run_dir = str(selected_k128)
    args.ddpm_run_dir = str(ddpm)
    captured: dict[str, object] = {}
    finalize = runner._finalize_failure

    def capture_failure(*args: object, **kwargs: object) -> dict[str, object]:
        result = finalize(*args, **kwargs)
        captured.update(result)
        return result

    monkeypatch.setattr(runner, "_finalize_failure", capture_failure)
    assert runner.run_production(args) == 4

    saved = runner._read_json(
        run_dir / "input_bindings" / "input_authority_observations.json"
    )
    expected = runner._input_authority_observations(
        legacy_checkpoint=selected_checkpoint,
        arff=selected_arff,
        k128_run_dir=selected_k128,
        ddpm_run_dir=ddpm,
        recorded_at=saved["recorded_at"],
    )
    assert saved == expected
    assert saved["all_expected_authorities_matched"] == 0
    assert saved["inputs"][observation_key]["matched"] == 0
    assert all(
        value["matched"] == int(key != observation_key)
        for key, value in saved["inputs"].items()
    )
    assert captured["failed_stage"] == failed_stage
    assert captured["state"] == "integrity_failed"
    assert captured["verification"]["passed"] == 1


def test_every_torch_load_is_weights_only_and_selector_is_absent() -> None:
    tree = ast.parse(inspect.getsource(runner))
    loads = []
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "load":
                loads.append(node)
            if "select_generation_result" in node.func.attr:
                forbidden.append(node.func.attr)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            forbidden.extend(
                alias.name for alias in node.names if "select_generation_result" in alias.name
            )
    assert loads, "checkpoint and evaluator loads must be explicit"
    for call in loads:
        keyword = next((kw.value for kw in call.keywords if kw.arg == "weights_only"), None)
        assert isinstance(keyword, ast.Constant) and keyword.value is True
    assert forbidden == []


def test_checkpoint_stat_and_hash_fail_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"synthetic")
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("deserialization must remain unreachable")

    monkeypatch.setattr(runner.torch, "load", forbidden_load)
    with pytest.raises(runner.IntegrityFailure, match="size"):
        runner.safe_extract_legacy_checkpoint(checkpoint, expected_bytes=999, expected_sha256="0" * 64)
    assert not called
    with pytest.raises(runner.IntegrityFailure, match="SHA-256"):
        runner.safe_extract_legacy_checkpoint(
            checkpoint, expected_bytes=len(b"synthetic"), expected_sha256="0" * 64
        )
    assert not called


def test_checkpoint_package_mismatch_precedes_safe_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"synthetic")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "EXPECTED_NUMPY_VERSION", "definitely-not-current")
    monkeypatch.setattr(
        runner.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("load must be unreachable on version mismatch"),
    )
    with pytest.raises(runner.IntegrityFailure, match="NumPy version mismatch"):
        runner.safe_extract_legacy_checkpoint(
            checkpoint, expected_bytes=checkpoint.stat().st_size, expected_sha256=digest
        )


def test_safe_loader_uses_exact_scoped_globals_and_preserves_historical_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"safe-synthetic-checkpoint")
    payload = {
        "config": {"sample_rejection_factor": 4, "sample_selection_metric": "composite"},
        **{key: {} for key in runner.EXPECTED_CHECKPOINT_KEYS if key != "config"},
    }
    load_calls: list[dict[str, object]] = []
    allowed: list[str] = []
    real_torch_load = runner.torch.load

    active_user_globals: list[object] = []

    @contextlib.contextmanager
    def capture_safe_globals(values: list[object]):
        assert active_user_globals == []
        active_user_globals.extend(values)
        allowed.extend(f"{item.__module__}.{item.__qualname__}" for item in values)  # type: ignore[attr-defined]
        try:
            yield
        finally:
            active_user_globals.clear()

    def fake_load(path: Path, **kwargs: object) -> object:
        load_calls.append({"path": path, **kwargs})
        if Path(path) == checkpoint:
            return payload
        return real_torch_load(path, **kwargs)

    class TinyModel:
        def __init__(self, *_: object, **__: object) -> None:
            self.weight = torch.nn.Parameter(torch.ones(1))

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            assert state == {"weight": torch.ones(1)} and strict is True

        def parameters(self):
            yield self.weight

    monkeypatch.setattr(runner.torch.serialization, "safe_globals", capture_safe_globals)
    monkeypatch.setattr(runner.torch.serialization, "get_safe_globals", lambda: list(active_user_globals))
    monkeypatch.setattr(runner.torch, "load", fake_load)
    monkeypatch.setattr(runner, "DirectFluxUNet", TinyModel)
    monkeypatch.setattr(runner, "EXPECTED_PARAMETER_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "_validate_checkpoint_payload",
        lambda value: (_tiny_config(sample_rejection_factor=4, sample_selection_metric="composite"), {"weight": torch.ones(1)}),
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    clean_state = tmp_path / "clean-state.pt"
    receipt = runner.safe_extract_legacy_checkpoint(
        checkpoint,
        clean_state,
        expected_bytes=checkpoint.stat().st_size,
        expected_sha256=digest,
    )
    assert load_calls == [
        {"path": checkpoint, "map_location": "cpu", "weights_only": True},
        {"path": clean_state, "map_location": "cpu", "weights_only": True},
    ]
    assert allowed == [
        "numpy._core.multiarray._reconstruct",
        "numpy.ndarray",
        "numpy.dtype",
        "numpy.dtypes.Int64DType",
        "numpy.dtypes.Float64DType",
    ]
    assert receipt["historical_selection_fields"] == {
        "sample_rejection_factor": 4,
        "sample_selection_metric": "composite",
    }
    assert receipt["replay_policy"] == {
        "generated_candidates_per_path": 1,
        "selector": None,
        "all_candidates_retained": 1,
    }
    assert receipt["clean_state_sha256"] == runner.sha256_file(clean_state)
    assert receipt["clean_state_bytes"] == clean_state.stat().st_size
    assert active_user_globals == []

    monkeypatch.setattr(runner, "EXPECTED_PARAMETER_COUNT", 2)
    with pytest.raises(runner.IntegrityFailure, match="parameter count"):
        runner.safe_extract_legacy_checkpoint(
            checkpoint,
            expected_bytes=checkpoint.stat().st_size,
            expected_sha256=digest,
        )


def test_safe_loader_isolates_and_restores_preexisting_user_safe_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"synthetic")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ambient = [str]
    active: list[object] = list(ambient)
    seen_during_load: list[str] = []

    def clear() -> None:
        active.clear()

    def add(values: list[object]) -> None:
        active.extend(values)

    @contextlib.contextmanager
    def scoped(values: list[object]):
        assert active == []
        active.extend(values)
        try:
            yield
        finally:
            active.clear()

    payload = {
        "config": {"sample_rejection_factor": 4, "sample_selection_metric": "composite"},
        **{key: {} for key in runner.EXPECTED_CHECKPOINT_KEYS if key != "config"},
    }

    def fake_load(*args: object, **kwargs: object) -> object:
        seen_during_load.extend(f"{item.__module__}.{item.__qualname__}" for item in active)  # type: ignore[attr-defined]
        return payload

    class TinyModel:
        def __init__(self, *_: object, **__: object) -> None:
            self.weight = torch.nn.Parameter(torch.ones(1))

        def load_state_dict(self, *_: object, strict: bool) -> None:
            assert strict is True

        def parameters(self):
            yield self.weight

    monkeypatch.setattr(runner.torch.serialization, "get_safe_globals", lambda: list(active))
    monkeypatch.setattr(runner.torch.serialization, "clear_safe_globals", clear)
    monkeypatch.setattr(runner.torch.serialization, "add_safe_globals", add)
    monkeypatch.setattr(runner.torch.serialization, "safe_globals", scoped)
    monkeypatch.setattr(runner.torch, "load", fake_load)
    monkeypatch.setattr(runner, "DirectFluxUNet", TinyModel)
    monkeypatch.setattr(runner, "EXPECTED_PARAMETER_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "_validate_checkpoint_payload",
        lambda value: (_tiny_config(sample_rejection_factor=4, sample_selection_metric="composite"), {"weight": torch.ones(1)}),
    )
    runner.safe_extract_legacy_checkpoint(
        checkpoint, expected_bytes=checkpoint.stat().st_size, expected_sha256=digest
    )
    assert seen_during_load == [
        "numpy._core.multiarray._reconstruct",
        "numpy.ndarray",
        "numpy.dtype",
        "numpy.dtypes.Int64DType",
        "numpy.dtypes.Float64DType",
    ]
    assert active == ambient

    def failing_load(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic deserialization failure")

    monkeypatch.setattr(runner.torch, "load", failing_load)
    with pytest.raises(RuntimeError, match="synthetic deserialization failure"):
        runner.safe_extract_legacy_checkpoint(
            checkpoint,
            expected_bytes=checkpoint.stat().st_size,
            expected_sha256=digest,
        )
    assert active == ambient


def test_real_torch_safe_global_set_restores_tuple_ambient_after_success_and_throw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"real-safe-global-regression")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    original = list(torch.serialization.get_safe_globals())

    def ambient_callable() -> None:
        return None

    ambient_tuple = (ambient_callable, "synthetic.ambient_alias")
    approved = (
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType,
        np.dtypes.Float64DType,
    )

    def authority(entry: object) -> tuple[int, str]:
        if isinstance(entry, tuple):
            assert len(entry) == 2 and callable(entry[0]) and isinstance(entry[1], str)
            value, name = entry
        else:
            value = entry
            name = f"{value.__module__}.{value.__qualname__}"  # type: ignore[attr-defined]
        return id(value), name

    def authority_set(values: list[object]) -> frozenset[tuple[int, str]]:
        return frozenset(authority(value) for value in values)

    payload = {
        "config": {"sample_rejection_factor": 4, "sample_selection_metric": "composite"},
        **{key: {} for key in runner.EXPECTED_CHECKPOINT_KEYS if key != "config"},
    }

    class TinyModel:
        def __init__(self, *_: object, **__: object) -> None:
            self.weight = torch.nn.Parameter(torch.ones(1))

        def load_state_dict(self, *_: object, strict: bool) -> None:
            assert strict is True

        def parameters(self):
            yield self.weight

    monkeypatch.setattr(runner, "DirectFluxUNet", TinyModel)
    monkeypatch.setattr(runner, "EXPECTED_PARAMETER_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "_validate_checkpoint_payload",
        lambda value: (
            _tiny_config(
                sample_rejection_factor=4,
                sample_selection_metric="composite",
            ),
            {"weight": torch.ones(1)},
        ),
    )
    active_at_load: list[frozenset[tuple[int, str]]] = []

    def successful_load(*args: object, **kwargs: object) -> object:
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        active_at_load.append(authority_set(torch.serialization.get_safe_globals()))
        return payload

    try:
        torch.serialization.add_safe_globals([ambient_tuple])
        ambient = authority_set(torch.serialization.get_safe_globals())
        approved_set = authority_set(list(approved))
        assert authority(ambient_tuple) in ambient

        monkeypatch.setattr(runner.torch, "load", successful_load)
        runner.safe_extract_legacy_checkpoint(
            checkpoint,
            expected_bytes=checkpoint.stat().st_size,
            expected_sha256=digest,
        )
        assert active_at_load == [approved_set]
        after_success = authority_set(torch.serialization.get_safe_globals())
        assert after_success == ambient
        assert (after_success - ambient).isdisjoint(approved_set)

        def throwing_load(*args: object, **kwargs: object) -> object:
            assert kwargs == {"map_location": "cpu", "weights_only": True}
            active_at_load.append(authority_set(torch.serialization.get_safe_globals()))
            raise RuntimeError("synthetic deserialization failure")

        monkeypatch.setattr(runner.torch, "load", throwing_load)
        with pytest.raises(RuntimeError, match="synthetic deserialization failure"):
            runner.safe_extract_legacy_checkpoint(
                checkpoint,
                expected_bytes=checkpoint.stat().st_size,
                expected_sha256=digest,
            )
        assert active_at_load == [approved_set, approved_set]
        after_throw = authority_set(torch.serialization.get_safe_globals())
        assert after_throw == ambient
        assert (after_throw - ambient).isdisjoint(approved_set)
    finally:
        torch.serialization.clear_safe_globals()
        if original:
            torch.serialization.add_safe_globals(original)
    assert authority_set(torch.serialization.get_safe_globals()) == authority_set(original)


def test_atomic_replace_retries_windows_sharing_violation_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "status.json"
    runner._write_json(path, {"state": "old"})
    real_replace = runner.os.replace
    attempts: list[tuple[object, object]] = []
    sleeps: list[float] = []

    def flaky(source: object, destination: object) -> None:
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError(5, "simulated sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", flaky)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    runner._write_json(path, {"state": "new"})
    assert runner._read_json(path) == {"state": "new"}
    assert len(attempts) == 3 and sleeps == [0.05, 0.05]

    old_bytes = path.read_bytes()
    attempts.clear()
    sleeps.clear()

    def blocked(source: object, destination: object) -> None:
        attempts.append((source, destination))
        raise PermissionError(5, "simulated sharing violation")

    monkeypatch.setattr(runner.os, "replace", blocked)
    with pytest.raises(runner.IntegrityFailure, match="atomic replace"):
        runner._write_json(path, {"state": "must-not-appear"})
    assert path.read_bytes() == old_bytes
    assert 2 <= len(attempts) <= 20 and len(sleeps) == len(attempts) - 1
    assert sum(sleeps) <= 5.0
    assert not list(tmp_path.glob(".status.json.*.tmp")) and not list(tmp_path.glob("status.json.tmp"))


def test_payload_authority_is_exact_and_does_not_mutate_legacy_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_payload = dataclasses.asdict(
        DirectFluxMNISTConfig(sample_rejection_factor=4, sample_selection_metric="composite")
    )
    config_hash = hashlib.sha256(
        (json.dumps(config_payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    model = runner.DirectFluxUNet(
        DirectFluxMNISTConfig(**config_payload), base_channels=48, num_classes=10
    )
    state = model.state_dict()
    payload = {key: {} for key in runner.EXPECTED_CHECKPOINT_KEYS}
    payload["config"] = config_payload
    payload["model_state_dict"] = state
    before = json.dumps(config_payload, sort_keys=True)
    monkeypatch.setattr(runner, "LEGACY_CONFIG_SHA256", config_hash)
    monkeypatch.setattr(runner, "EXPECTED_STATE_TENSORS", len(state))
    config, returned_state = runner._validate_checkpoint_payload(payload)
    assert config.sample_rejection_factor == 4 and config.sample_selection_metric == "composite"
    assert returned_state is state and json.dumps(config_payload, sort_keys=True) == before

    with pytest.raises(runner.IntegrityFailure, match="key set"):
        runner._validate_checkpoint_payload({**payload, "unexpected": 1})
    bad = dict(payload)
    bad["args"] = object()
    with pytest.raises(runner.IntegrityFailure, match="forbidden type"):
        runner._validate_checkpoint_payload(bad)

    first_name = next(iter(state))
    missing_tensor = dict(payload)
    missing_tensor["model_state_dict"] = {
        name: tensor for name, tensor in state.items() if name != first_name
    }
    with pytest.raises(runner.IntegrityFailure, match="tensor count"):
        runner._validate_checkpoint_payload(missing_tensor)

    renamed_state = dict(state)
    first_tensor = renamed_state.pop(first_name)
    renamed_state["unexpected.weight"] = first_tensor
    renamed_tensor = dict(payload)
    renamed_tensor["model_state_dict"] = renamed_state
    with pytest.raises(runner.IntegrityFailure, match="name/order"):
        runner._validate_checkpoint_payload(renamed_tensor)

    for replacement, message in (
        (state[first_name].to(torch.float64), "float32"),
        (state[first_name].reshape(-1), "shape"),
    ):
        changed_state = dict(state)
        changed_state[first_name] = replacement
        changed = dict(payload)
        changed["model_state_dict"] = changed_state
        with pytest.raises(runner.IntegrityFailure, match=message):
            runner._validate_checkpoint_payload(changed)

    nonfinite_state = dict(state)
    nonfinite = state[first_name].clone()
    nonfinite.reshape(-1)[0] = torch.nan
    nonfinite_state[first_name] = nonfinite
    nonfinite_payload = dict(payload)
    nonfinite_payload["model_state_dict"] = nonfinite_state
    with pytest.raises(runner.IntegrityFailure, match="nonfinite"):
        runner._validate_checkpoint_payload(nonfinite_payload)


def test_arff_hash_failure_stops_before_content_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arff = tmp_path / "mnist.arff"
    arff.write_text("@RELATION tiny\n@DATA\n", encoding="utf-8")
    monkeypatch.setattr(runner, "MNIST_ARFF_SHA256", "0" * 64)
    monkeypatch.setattr(
        runner,
        "_parse_mnist_arff_prefix",
        lambda *args, **kwargs: pytest.fail("ARFF content parse must remain unreachable"),
    )
    with pytest.raises(runner.IntegrityFailure, match="ARFF SHA-256"):
        runner.read_mnist_development_prefix(arff)


def test_arff_prefix_does_not_fetch_the_first_terminal_row() -> None:
    row = ",".join(["0"] * 784 + ["3"])

    class StopBeforeTerminal:
        def __init__(self) -> None:
            self.values = iter(["@RELATION tiny\n", "@DATA\n", row + "\n", row + "\n"])
            self.calls = 0

        def __iter__(self) -> "StopBeforeTerminal":
            return self

        def __next__(self) -> str:
            self.calls += 1
            if self.calls > 4:
                raise AssertionError("terminal row was fetched")
            return next(self.values)

    source = StopBeforeTerminal()
    images, labels, audit = runner._parse_mnist_arff_prefix(source, stop=2)
    assert images.shape == (2, 28, 28) and labels.tolist() == [3, 3]
    assert audit["content_rows_parsed"] == 2 and source.calls == 4


def test_path_inventory_is_exact_balanced_factor_one_policy() -> None:
    inventory = runner.build_path_inventory()
    assert inventory["path_ids"].tolist() == [f"efr-v1-{i:03d}" for i in range(160)]
    assert inventory["requested_labels"].tolist() == np.repeat(np.arange(10), 16).tolist()
    assert inventory["within_class_indices"].tolist() == np.tile(np.arange(16), 10).tolist()
    assert inventory["source_seeds"].tolist() == [0xE14F1000 + i for i in range(160)]
    assert runner.ROW_ROOT_SEEDS == {
        "null": 0xE14F2001,
        "teacher": 0xE14F3001,
        "learned": 0xE14F4001,
    }


def test_start_bank_uses_one_isolated_cpu_call_per_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, int, str, torch.dtype]] = []

    def fake_source(batch_size: int, config: DirectFluxMNISTConfig, *, device: torch.device,
                    dtype: torch.dtype, **_: object) -> SimpleNamespace:
        seed = int(torch.initial_seed())
        seen.append((batch_size, seed, str(device), dtype))
        value = np.float32((seed - runner.SOURCE_SEED_BASE + 1) / 10_000)
        masses = torch.full((1, config.grid_size**2), float(value), dtype=dtype, device=device)
        masses /= masses.sum(dim=1, keepdim=True)
        masses[0, 0] += torch.tensor(value / 100, dtype=dtype, device=device)
        masses /= masses.sum(dim=1, keepdim=True)
        return SimpleNamespace(masses=masses)

    monkeypatch.setattr(runner, "_sample_source_batch_torch", fake_source)
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state().clone()
    bank = runner.build_start_bank(_tiny_config())
    assert bank.shape == (160, 16) and bank.dtype == np.float32
    assert [seed for _, seed, _, _ in seen] == [runner.SOURCE_SEED_BASE + i for i in range(160)]
    assert all(batch == 1 and device == "cpu" and dtype == torch.float32 for batch, _, device, dtype in seen)
    assert torch.equal(torch.random.get_rng_state(), state_before)
    assert len(np.unique(bank, axis=0)) == 160


def test_teacher_targets_are_first_sixteen_validation_occurrences() -> None:
    labels = np.resize(np.arange(10, dtype=np.int64), 5_000)
    images = np.zeros((5_000, 28, 28), dtype=np.uint8)
    for index in range(5_000):
        images[index].reshape(-1)[index % 784] = np.uint8(index % 254 + 1)
    target = runner.build_teacher_target_bank(images, labels)
    expected_local = np.concatenate([np.flatnonzero(labels == digit)[:16] for digit in range(10)])
    assert target["arff_global_row_ids"].tolist() == (55_000 + expected_local).tolist()
    assert target["requested_labels"].tolist() == np.repeat(np.arange(10), 16).tolist()
    assert target["masses"].shape == (160, 784)
    assert target["masses"].dtype == np.float32
    assert np.allclose(target["masses"].sum(axis=1), 1.0, atol=2e-6)
    assert np.array_equal(target["images_uint8"], images[expected_local])
    # Original ARFF pixels and the fixed global display of normalized masses are
    # separate authorities unless an image happens to have exactly 25471 ink.
    rendered_masses = runner.mass_to_uint8(target["masses"])
    assert not np.array_equal(target["images_uint8"], rendered_masses)


def test_sealed_teacher_target_bank_keeps_source_and_rendered_images_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    for relative in (
        "config.json",
        "source_bindings.json",
        "input_bindings/predecessors.json",
        "input_bindings/legacy_checkpoint_receipt.json",
        "input_bindings/clean_model_state_receipt.json",
        "input_bindings/ddpm_evaluator_binding.json",
    ):
        runner._write_json(run_dir / relative, {"synthetic": 1})
    (run_dir / "input_bindings" / "clean_model_state.pt").write_bytes(b"synthetic")

    images = np.zeros((60_000, 28, 28), dtype=np.uint8)
    images[:27_500] = _mass_image(25_470)
    images[27_500:55_000] = _mass_image(25_472)
    labels = np.zeros(60_000, dtype=np.int64)
    labels[55_000:] = np.resize(np.arange(10, dtype=np.int64), 5_000)
    for index in range(5_000):
        images[55_000 + index].reshape(-1)[index % 784] = np.uint8(index % 250 + 1)

    starts = np.full((160, 784), np.float32(1 / 784), dtype=np.float32)
    for index in range(160):
        starts[index, index] += np.float32(1e-7)
        starts[index, (index + 1) % 784] -= np.float32(1e-7)
    monkeypatch.setattr(runner, "build_start_bank", lambda *args, **kwargs: starts.copy())
    runner._write_inventory_authorities(
        run_dir,
        config=DirectFluxMNISTConfig(),
        development_images=images,
        development_labels=labels,
        data_audit={"content_rows_parsed": 60_000},
    )
    with np.load(run_dir / "inventory" / "teacher_target_bank.npz", allow_pickle=False) as bank:
        assert set(bank.files) == {
            "masses",
            "source_images_uint8",
            "rendered_images_uint8",
            "requested_labels",
            "validation_local_ids",
            "arff_global_row_ids",
            "path_ids",
        }
        assert not np.array_equal(bank["source_images_uint8"], bank["rendered_images_uint8"])
        assert np.array_equal(bank["rendered_images_uint8"], runner.mass_to_uint8(bank["masses"]))

    audit = {"content_rows_parsed": 60_000}
    monkeypatch.setattr(
        runner,
        "read_mnist_development_prefix",
        lambda _: (images.copy(), labels.copy(), dict(audit)),
    )
    verifier_config = {"input_paths": {"arff": str(tmp_path / "synthetic.arff")}}
    runner._verify_data_and_inventory(run_dir, verifier_config, DirectFluxMNISTConfig())

    target_path = run_dir / "inventory" / "teacher_target_bank.npz"
    with np.load(target_path, allow_pickle=False) as bank:
        tampered = {key: bank[key].copy() for key in bank.files}
    donor = int(np.argmax(tampered["masses"][0]))
    recipient = (donor + 1) % 784
    tampered["masses"][0, donor] -= np.float32(1e-5)
    tampered["masses"][0, recipient] += np.float32(1e-5)
    tampered["rendered_images_uint8"] = runner.mass_to_uint8(tampered["masses"])
    runner._write_npz(target_path, **tampered)
    start_seal_path = run_dir / "inventory" / "START_BANK_SEALED.json"
    start_seal = runner._read_json(start_seal_path)
    start_seal["teacher_target_bank_sha256"] = runner.sha256_file(target_path)
    start_seal["teacher_target_mass_sha256"] = runner._hash_array(tampered["masses"])
    runner._write_json(start_seal_path, start_seal)
    with pytest.raises(runner.IntegrityFailure, match="teacher target masses changed"):
        runner._verify_data_and_inventory(run_dir, verifier_config, DirectFluxMNISTConfig())


def test_target_firewalls_are_visible_in_public_signatures() -> None:
    null_parameters = inspect.signature(runner.run_null_row).parameters
    learned_parameters = inspect.signature(runner.run_learned_row).parameters
    teacher_parameters = inspect.signature(runner.run_teacher_row).parameters
    assert "targets" not in null_parameters and "model" not in null_parameters
    assert "targets" not in learned_parameters
    assert "model" not in teacher_parameters


def test_all_three_rows_dispatch_through_the_same_integrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts = np.full((1, 16), np.float32(1 / 16), dtype=np.float32)
    labels = np.array([3], dtype=np.int64)
    targets = starts.copy()
    config = _tiny_config()
    model = SimpleNamespace(config=config)
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_run_row(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, **kwargs})
        return sentinel

    monkeypatch.setattr(runner, "run_row", fake_run_row)
    assert runner.run_null_row(starts, labels, config, root_seed=11) is sentinel
    assert runner.run_teacher_row(starts, labels, targets, config, root_seed=12) is sentinel
    assert runner.run_learned_row(starts, labels, model, config, root_seed=13) is sentinel
    assert [call["row"] for call in calls] == ["null", "teacher", "learned"]
    assert [call["root_seed"] for call in calls] == [11, 12, 13]
    assert "model" not in calls[0] and "targets" not in calls[0]
    assert calls[1]["targets"] is targets and "model" not in calls[1]
    assert calls[2]["model"] is model and "targets" not in calls[2]


class _RecordingModel:
    def __init__(self, config: DirectFluxMNISTConfig) -> None:
        self.config = config
        self.sources: list[torch.Tensor] = []
        self.eval_called = False

    def to(self, _: object) -> "_RecordingModel":
        return self

    def eval(self) -> "_RecordingModel":
        self.eval_called = True
        return self

    def predict_flux(self, tau: torch.Tensor, masses: torch.Tensor, labels: torch.Tensor,
                     source_masses: torch.Tensor | None = None) -> torch.Tensor:
        assert source_masses is not None
        self.sources.append(source_masses.detach().cpu().clone())
        n = self.config.grid_size
        return torch.zeros((len(masses), 2, n, n), dtype=masses.dtype, device=masses.device)


def test_learned_provider_uses_persistent_source_masses_and_anchors() -> None:
    config = _tiny_config(adaptive_sampling=False)
    starts = np.arange(32, dtype=np.float32).reshape(2, 16) + 1
    starts /= starts.sum(axis=1, keepdims=True)
    labels = np.array([2, 8], dtype=np.int64)
    model = _RecordingModel(config)
    result = runner.run_learned_row(
        starts, labels, model, config, root_seed=17, num_steps=4, anchors=(0, 1, 2, 3, 4)
    )
    assert model.eval_called and len(model.sources) == 4
    assert all(np.array_equal(source.numpy(), starts) for source in model.sources)
    assert result.anchors.shape == (5, 2, 16) and result.anchors.dtype == np.float32
    assert result.labels.tolist() == [2, 8]
    assert [row["completed_step"] for row in result.telemetry] == [1, 2, 3, 4]


def test_learned_common_integrator_matches_the_current_generation_api() -> None:
    config = _tiny_config(adaptive_sampling=False, noise_weight=0.0)
    starts = np.full((2, 16), np.float32(1 / 16), dtype=np.float32)
    starts[1, 0] += np.float32(1 / 64)
    starts[1, 1] -= np.float32(1 / 64)
    labels = np.array([2, 8], dtype=np.int64)
    row_model = _RecordingModel(config)
    api_model = _RecordingModel(config)
    row = runner.run_learned_row(
        starts,
        labels,
        row_model,
        config,
        root_seed=19,
        num_steps=4,
        anchors=(0, 4),
    )
    api = simulate_direct_flux_generation(
        api_model,  # type: ignore[arg-type]
        labels,
        config=config,
        num_steps=4,
        deterministic=False,
        device="cpu",
        seed=19,
        use_amp=False,
        show_progress=False,
        initial_states=starts,
    )
    assert np.array_equal(row.endpoints, api.samples.astype(np.float32))
    assert len(row_model.sources) == len(api_model.sources) == 4


def test_adaptive_retry_resets_state_consumes_new_draws_and_records_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(adaptive_sampling=True, noise_weight=0.0)
    starts = np.arange(1, 17, dtype=np.float32)[None]
    starts /= starts.sum(axis=1, keepdims=True)
    labels = np.array([1], dtype=np.int64)
    state_inputs: list[np.ndarray] = []
    random_draws: list[float] = []

    def fake_step(states: torch.Tensor, *args: object, **kwargs: object):
        state_inputs.append(states.detach().cpu().numpy().copy())
        random_draws.append(float(torch.rand(())))
        call = len(state_inputs) - 1
        clipped = 1 if call < 3 else 0
        return torch.roll(states, shifts=1, dims=1), clipped, 1

    monkeypatch.setattr(runner, "eulerian_flux_step_torch", fake_step)
    monkeypatch.setattr(
        runner,
        "step_component_rms_torch",
        lambda *args, **kwargs: {
            "learned_step_rms": 0.0,
            "free_step_rms": 0.0,
            "noise_step_rms": 0.0,
        },
    )
    result = runner.run_null_row(starts, labels, config, root_seed=71, num_steps=1, anchors=(0, 1))
    assert len(state_inputs) == 7 and len(set(random_draws)) == 7
    assert np.array_equal(state_inputs[0], starts)
    assert np.array_equal(state_inputs[1], starts)  # retry with two substeps reset
    assert np.array_equal(state_inputs[3], starts)  # retry with four substeps reset
    assert np.array_equal(result.endpoints, np.roll(starts, 4, axis=1))
    record = result.telemetry[0]
    assert record["accepted_substeps"] == 4 and record["rejected_attempt_count"] == 2
    assert [attempt["substeps"] for attempt in record["attempts"]] == [1, 2, 4]


def test_nonadaptive_row_accepts_its_single_attempt_even_when_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(adaptive_sampling=False)
    starts = np.full((1, 16), 1 / 16, dtype=np.float32)
    seen_flux: list[torch.Tensor] = []

    def clipped_step(states: torch.Tensor, flux: torch.Tensor, *args: object, **kwargs: object):
        seen_flux.append(flux.detach().clone())
        return states.clone(), 1, 1

    monkeypatch.setattr(runner, "eulerian_flux_step_torch", clipped_step)
    result = runner.run_null_row(
        starts, np.array([0], dtype=np.int64), config, num_steps=1, anchors=(0, 1)
    )
    assert len(seen_flux) == 1 and torch.count_nonzero(seen_flux[0]) == 0
    assert result.telemetry[0]["accepted_substeps"] == 1
    assert result.telemetry[0]["accepted_clipping_fraction"] == 1.0


def test_teacher_flux_is_free_aware_and_has_requested_centered_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(adaptive_sampling=False, noise_weight=0.0)
    starts = np.arange(1, 33, dtype=np.float32).reshape(2, 16)
    starts /= starts.sum(axis=1, keepdims=True)
    targets = np.flip(starts, axis=1).copy()
    labels = np.array([2, 7], dtype=np.int64)
    captured: list[torch.Tensor] = []

    def capture_step(states: torch.Tensor, flux: torch.Tensor, *args: object, **kwargs: object):
        captured.append(flux.detach().clone())
        return states.clone(), 0, int(flux.numel())

    monkeypatch.setattr(runner, "eulerian_flux_step_torch", capture_step)
    runner.run_teacher_row(starts, labels, targets, config, num_steps=1, anchors=(0, 1))
    state_t = torch.from_numpy(starts)
    horizon = runner.natural_horizon(config)
    velocity = (torch.from_numpy(targets) - state_t) / horizon
    velocity -= velocity.mean(dim=1, keepdim=True)
    total_flux = captured[0] + config.free_weight * runner.free_drift_flux_torch(state_t, config)
    divergence = flux_divergence_torch(total_flux).reshape_as(velocity)
    assert torch.allclose(divergence, velocity, atol=2e-4, rtol=2e-5)


def test_deterministic_row_restart_reproduces_scientific_hashes_not_resource_telemetry() -> None:
    config = _tiny_config(adaptive_sampling=False, noise_weight=0.002)
    starts = np.arange(1, 33, dtype=np.float32).reshape(2, 16)
    starts /= starts.sum(axis=1, keepdims=True)
    labels = np.array([0, 9], dtype=np.int64)
    first = runner.run_null_row(starts, labels, config, root_seed=88, num_steps=2, anchors=(0, 1, 2))
    second = runner.run_null_row(starts, labels, config, root_seed=88, num_steps=2, anchors=(0, 1, 2))
    other = runner.run_null_row(starts, labels, config, root_seed=89, num_steps=2, anchors=(0, 1, 2))
    assert np.array_equal(first.anchors, second.anchors)
    assert first.scientific_digest == second.scientific_digest
    assert runner._hash_array(first.anchors) == runner._hash_array(second.anchors)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.path_ids, second.path_ids)
    assert first.root_seed == second.root_seed == 88
    assert not np.array_equal(first.endpoints, other.endpoints)
    telemetry_without_resource_identity = [dict(record) for record in first.telemetry]
    telemetry_without_resource_identity[0]["elapsed_seconds"] += 123.0
    telemetry_without_resource_identity[0]["cuda_allocated_bytes"] += 99
    telemetry_without_resource_identity[0]["cuda_peak_allocated_bytes"] += 101
    assert (
        runner._scientific_row_digest(first.anchors, telemetry_without_resource_identity)
        == first.scientific_digest
    )
    telemetry_without_resource_identity[0]["accepted_clipped"] += 1
    assert (
        runner._scientific_row_digest(first.anchors, telemetry_without_resource_identity)
        != first.scientific_digest
    )


def test_last_valid_anchor_is_attached_to_callback_failure() -> None:
    config = _tiny_config(adaptive_sampling=False)
    starts = np.full((1, 16), 1 / 16, dtype=np.float32)
    labels = np.array([4], dtype=np.int64)

    def stop_after_first(_: object) -> None:
        raise runner.ResourceStop("synthetic stop")

    with pytest.raises(runner.ResourceStop) as raised:
        runner.run_null_row(
            starts,
            labels,
            config,
            num_steps=2,
            anchors=(0, 1, 2),
            outer_step_callback=stop_after_first,
        )
    partial = getattr(raised.value, "partial_row_result")
    assert partial.anchors.shape == (2, 1, 16)
    assert len(partial.telemetry) == 1
    assert np.all(np.isfinite(partial.anchors)) and float(partial.anchors.min()) >= 0.0


def test_numerical_health_failure_retains_the_last_valid_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config(adaptive_sampling=False)
    starts = np.full((1, 16), np.float32(1 / 16), dtype=np.float32)

    def nonfinite_step(states: torch.Tensor, *args: object, **kwargs: object):
        invalid = states.clone()
        invalid[0, 0] = torch.nan
        return invalid, 0, 1

    monkeypatch.setattr(runner, "eulerian_flux_step_torch", nonfinite_step)
    with pytest.raises(runner.IntegrityFailure, match="numerical health") as raised:
        runner.run_null_row(
            starts,
            np.array([4], dtype=np.int64),
            config,
            num_steps=1,
            anchors=(0, 1),
        )
    partial = getattr(raised.value, "partial_row_result")
    assert partial.anchor_steps.tolist() == [0]
    assert np.array_equal(partial.anchors, starts[None])
    assert partial.telemetry == []


def test_post_completion_row_stop_keeps_full_population_and_renders_latest_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    with np.load(run_dir / "populations" / "null.npz", allow_pickle=False) as archive:
        anchors = archive["anchors"].copy()
        labels = archive["labels"].copy()
        path_ids = archive["path_ids"].copy()
        telemetry = [json.loads(str(value)) for value in archive["telemetry_json"].tolist()]
    result = runner.RowResult(
        row="null",
        anchors=anchors,
        labels=labels,
        path_ids=path_ids,
        telemetry=telemetry,
        root_seed=runner.ROW_ROOT_SEEDS["null"],
        scientific_digest=runner._scientific_row_digest(anchors, telemetry),
        anchor_steps=np.asarray(runner.ANCHORS, dtype=np.int64),
    )
    (run_dir / "populations" / "null.npz").unlink()
    (run_dir / "telemetry" / "null_steps.csv").unlink()

    def completed_row(*args: object, **kwargs: object) -> runner.RowResult:
        callback = kwargs["outer_step_callback"]
        callback(
            {
                "row": "null",
                "completed_step": runner.OUTER_STEPS,
                "saved_anchors": [torch.from_numpy(anchor.copy()) for anchor in anchors],
                "saved_steps": list(runner.ANCHORS),
                "state": torch.from_numpy(anchors[-1].copy()),
                "telemetry": telemetry,
            }
        )
        return result

    class StopAfterDurableSave:
        def admit(self, *args: object, **kwargs: object) -> None:
            return None

        def complete(self, *args: object, **kwargs: object) -> None:
            raise runner.ResourceStop("synthetic post-completion stop")

    monkeypatch.setattr(runner, "run_null_row", completed_row)
    with pytest.raises(runner.ResourceStop, match="post-completion") as raised:
        runner._execute_full_row(
            run_dir,
            governor=StopAfterDurableSave(),
            row="null",
            starts=anchors[0],
            labels=labels,
            path_ids=path_ids,
            targets=anchors[-1],
            config=_tiny_config(grid_size=28),
            model=SimpleNamespace(),
            device="cpu",
            predicted_eight_step_seconds=0.01,
        )

    durable = getattr(raised.value, "durable_full_row_result")
    assert durable is result
    assert (run_dir / "populations" / "null.npz").is_file()
    assert not (run_dir / "populations" / "partial_null.npz").exists()
    runner._persist_durable_full_failure_image(run_dir, durable)
    latest = run_dir / "images" / "partial_null_latest.png"
    runner._verify_sheet_pixels(
        latest,
        runner.mass_to_uint8(
            anchors[-1],
            runner._read_json(run_dir / "input_bindings" / "mass_to_uint8.json"),
        ),
        columns=16,
        scale=2,
        captions=None,
    )


def test_resource_projection_formula_and_caps() -> None:
    projection = runner.resource_projection(
        charged_active_seconds=4.0,
        teacher8_seconds=1.0,
        null8_seconds=2.0,
        learned8_seconds=3.0,
        projected_persisted_bytes=10 * 1024 * 1024,
        peak_cuda_fraction=0.5,
    )
    assert projection["projected_rows_seconds"] == pytest.approx(1.25 * 32 * 6)
    assert projection["projected_total_seconds"] == pytest.approx(4 + 1.25 * 32 * 6 + 30)
    assert projection["passed"] == 0  # 274 seconds exceeds the frozen 240-second cap.
    assert "active" in projection["stop_reason"]


def test_resource_ledger_rehydrates_cumulative_time_and_rejects_unresolved_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    budget = runner.ResourceBudget(reserve_seconds=0.0)
    ledger = {
        "schema": runner.VERSION + "-resource-ledger",
        "budget": dataclasses.asdict(budget),
        "active_seconds": 7.25,
        "events": [{"event": "complete", "kind": "preflight"}],
        "failed_admission": None,
        "open_events": [],
    }
    runner._write_json(run_dir / "resource_ledger.json", ledger)
    clock = iter([100.0, 101.5])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    governor = runner.ResourceGovernor.rehydrate(run_dir, device="cpu")
    governor.admit(
        "human_review_terminalization",
        predicted_seconds=2.0,
        predicted_next_bytes=1024,
        reserve_remaining_seconds=0.0,
    )
    governor.complete("human_review_terminalization")
    persisted = runner._read_json(run_dir / "resource_ledger.json")
    assert persisted["active_seconds"] == pytest.approx(8.75)
    assert [event["event"] for event in persisted["events"][-2:]] == ["admit", "complete"]

    for field, value, message in (
        ("open_events", ["learned_row"], "unresolved open event"),
        ("failed_admission", {"kind": "scoring"}, "failed admission"),
    ):
        invalid = {**ledger, field: value}
        runner._write_json(run_dir / "resource_ledger.json", invalid)
        with pytest.raises(runner.IntegrityFailure, match=message):
            runner.ResourceGovernor.rehydrate(run_dir, device="cpu")


def test_resource_recovery_conservatively_charges_one_well_formed_interruption(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    budget = runner.ResourceBudget()
    runner._write_json(
        run_dir / "resource_ledger.json",
        {
            "schema": runner.VERSION + "-resource-ledger",
            "budget": dataclasses.asdict(budget),
            "active_seconds": 0.0,
            "events": [
                {
                    "event": "admit",
                    "kind": "learned_row",
                    "predicted_seconds": 2.5,
                    "predicted_next_bytes": 1024,
                    "active_seconds_before": 0.0,
                    "reserve_remaining_seconds": 30.0,
                    "storage_bytes_before": 0,
                    "cuda_allocated_bytes": 0,
                    "cuda_total_bytes": 0,
                    "cuda_fraction": 0.0,
                    "checks": {"active": True, "storage": True, "cuda": True, "quantum": True},
                    "passed": 1,
                    "recorded_at": "synthetic",
                }
            ],
            "failed_admission": None,
            "open_events": ["learned_row"],
        },
    )
    recovered = runner.ResourceGovernor.rehydrate(
        run_dir, device="cpu", recover_interrupted=True
    )
    assert recovered.active_seconds == pytest.approx(2.5)
    persisted = runner._read_json(run_dir / "resource_ledger.json")
    assert persisted["open_events"] == []
    assert persisted["events"][-1]["event"] == "interrupted-close"
    assert persisted["events"][-1]["charged_predicted_seconds"] == pytest.approx(2.5)
    runner._verify_resource_ledger(run_dir, {"state": "awaiting_human_review"})

    malformed = runner._read_json(run_dir / "resource_ledger.json")
    malformed["open_events"] = ["teacher_row"]
    runner._write_json(run_dir / "resource_ledger.json", malformed)
    with pytest.raises(runner.IntegrityFailure, match="malformed"):
        runner.ResourceGovernor.rehydrate(run_dir, device="cpu", recover_interrupted=True)


def test_resource_verifier_accepts_an_authenticated_interrupted_terminal_closure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    governor = runner.ResourceGovernor(
        run_dir,
        runner.ResourceBudget(reserve_seconds=0.0),
        device="cpu",
    )
    governor.admit(
        "machine_terminalization",
        predicted_seconds=2.5,
        predicted_next_bytes=1,
        reserve_remaining_seconds=0.0,
    )
    recovered = runner.ResourceGovernor.rehydrate(
        run_dir,
        device="cpu",
        recover_interrupted=True,
    )
    assert recovered.events[-1]["event"] == "interrupted-close"
    verified = runner._verify_resource_ledger(
        run_dir,
        {"state": "awaiting_human_review", "route": "awaiting_human_review"},
    )
    assert verified["events"][-1]["kind"] == "machine_terminalization"


def test_real_population_and_review_receipts_preserve_terminal_reserve_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    budget = runner.ResourceBudget()
    governor = runner.ResourceGovernor(run_dir, budget, device="cpu")
    monkeypatch.setattr(runner, "_storage_bytes", lambda _: 0)
    clock = iter([10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))

    governor.admit(
        "population_seal_and_scoring",
        predicted_seconds=30.0,
        predicted_next_bytes=25 * 1024 * 1024,
    )
    governor.complete("population_seal_and_scoring")
    governor.admit(
        "review_prepare",
        predicted_seconds=10.0,
        predicted_next_bytes=15 * 1024 * 1024,
    )
    governor.complete("review_prepare")
    governor.admit(
        "machine_terminalization",
        predicted_seconds=5.0,
        predicted_next_bytes=1024,
        reserve_remaining_seconds=0.0,
    )
    governor.complete("machine_terminalization")

    ledger = runner._read_json(run_dir / "resource_ledger.json")
    admissions = [event for event in ledger["events"] if event["event"] == "admit"]
    assert [
        (event["kind"], event["reserve_remaining_seconds"])
        for event in admissions
    ] == [
        ("population_seal_and_scoring", 30.0),
        ("review_prepare", 30.0),
        ("machine_terminalization", 0.0),
    ]
    assert runner._verify_resource_ledger(
        run_dir,
        {"state": "awaiting_human_review", "route": "awaiting_human_review"},
    ) == ledger


@pytest.mark.parametrize(
    ("kind", "predicted_seconds", "predicted_next_bytes"),
    [
        ("population_seal_and_scoring", 30.0, 25 * 1024 * 1024),
        ("review_prepare", 10.0, 15 * 1024 * 1024),
    ],
)
def test_real_population_and_review_failed_admissions_replay_default_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    predicted_seconds: float,
    predicted_next_bytes: int,
) -> None:
    run_dir = tmp_path / kind
    governor = runner.ResourceGovernor(
        run_dir,
        runner.ResourceBudget(max_storage_bytes=1024),
        device="cpu",
    )
    monkeypatch.setattr(runner, "_storage_bytes", lambda _: 0)

    with pytest.raises(runner.ResourceStop, match=kind):
        governor.admit(
            kind,
            predicted_seconds=predicted_seconds,
            predicted_next_bytes=predicted_next_bytes,
        )

    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["failed_admission"]["reserve_remaining_seconds"] == 30.0
    assert runner._verify_resource_ledger(run_dir, {"state": "resource_stopped"}) == ledger


@pytest.mark.parametrize("overrun", ["storage", "cuda", "quantum"])
def test_resource_completion_stops_and_verifies_observed_cap_overrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrun: str
) -> None:
    budget = runner.ResourceBudget(
        max_storage_bytes=1024,
        max_cuda_fraction=0.5,
        reserve_seconds=0.0,
        maximum_quantum_seconds=0.5 if overrun == "quantum" else 60.0,
    )
    governor = runner.ResourceGovernor(tmp_path / "run", budget, device="cpu")
    clock = iter([10.0, 11.0])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    if overrun == "storage":
        storage = iter([0, 2048])
        monkeypatch.setattr(runner, "_storage_bytes", lambda _: next(storage))
        monkeypatch.setattr(governor, "_cuda_receipt", lambda: (0, 0, 0.0))
    elif overrun == "cuda":
        monkeypatch.setattr(runner, "_storage_bytes", lambda _: 0)
        cuda = iter([(0, 100, 0.0), (75, 100, 0.75)])
        monkeypatch.setattr(governor, "_cuda_receipt", lambda: next(cuda))
    else:
        monkeypatch.setattr(runner, "_storage_bytes", lambda _: 0)
        monkeypatch.setattr(governor, "_cuda_receipt", lambda: (0, 0, 0.0))
    governor.admit(
        "scoring",
        predicted_seconds=0.25 if overrun == "quantum" else 2.0,
        predicted_next_bytes=512,
        reserve_remaining_seconds=0.0,
    )
    with pytest.raises(runner.ResourceStop, match=overrun):
        governor.complete("scoring")
    ledger = runner._read_json(tmp_path / "run" / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert ledger["events"][-1]["event"] == "complete"
    runner._verify_resource_ledger(tmp_path / "run", {"state": "resource_stopped"})


def test_population_seal_recomputes_fixed_raster_and_review_is_exactly_blind(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    seal = runner.seal_populations(run_dir)
    assert seal["generated_candidates_per_path"] == 1
    assert seal["selector"] is None and seal["all_endpoints_retained"] == 1
    for row in ("teacher", "null", "learned"):
        with np.load(run_dir / "populations" / f"{row}.npz", allow_pickle=False) as raw, np.load(
            run_dir / "populations" / f"{row}_uint8.npz", allow_pickle=False
        ) as rendered:
            assert np.array_equal(rendered["anchors"], runner.mass_to_uint8(raw["anchors"]))

    ready = runner.prepare_blind_review(run_dir)
    assert (ready["sample_count"], ready["learned_count"], ready["null_count"]) == (80, 40, 40)
    indices = np.load(run_dir / "review" / "review_indices.npy", allow_pickle=False)
    expected = [digit * 16 + offset for digit in range(10) for offset in (0, 5, 10, 15)]
    assert indices.tolist() == expected
    membership = json.loads((run_dir / "review" / "private_membership.json").read_text())["entries"]
    assert [entry["row"] for entry in membership] == ["learned"] * 40 + ["null"] * 40
    assert [entry["path_index"] for entry in membership[:40]] == expected
    public = (run_dir / "review" / "human_review_template.csv").read_text(encoding="utf-8")
    assert "learned" not in public and "null" not in public and "efr-v1" not in public
    assert public.splitlines()[0] == "review_order,sample_id,assigned_label,notes"
    assert not (run_dir / "outcome.json").exists()
    runner._verify_review_bundle(run_dir, seal)

    membership_path = run_dir / "review" / "private_membership.json"
    coordinated_membership = runner._read_json(membership_path)
    coordinated_membership["entries"][0]["row"] = "null"
    runner._write_json(membership_path, coordinated_membership)
    ready_path = run_dir / "review" / "READY.json"
    coordinated_ready = runner._read_json(ready_path)
    coordinated_ready["membership_sha256"] = runner.sha256_file(membership_path)
    runner._write_json(ready_path, coordinated_ready)
    with pytest.raises(runner.IntegrityFailure, match="private review membership changed"):
        runner._verify_review_bundle(run_dir, seal)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("start", "anchor zero"),
        ("identity", "labels do not match"),
        ("telemetry", "telemetry must contain 256"),
    ],
)
def test_population_admission_rejects_start_identity_or_telemetry_drift(
    tmp_path: Path, tamper: str, message: str
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    path = run_dir / "populations" / "learned.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if tamper == "start":
        arrays["anchors"][0, 0, 0] += np.float32(0.0001)
        arrays["anchors"][0, 0, 1] -= np.float32(0.0001)
    elif tamper == "identity":
        arrays["labels"][0] = 9
    else:
        arrays["telemetry_json"] = arrays["telemetry_json"][:-1]
    telemetry = [json.loads(str(value)) for value in arrays["telemetry_json"].tolist()]
    arrays["scientific_digest"] = np.asarray(
        [runner._scientific_row_digest(arrays["anchors"], telemetry)], dtype=np.str_
    )
    runner._write_npz(path, **arrays)
    with pytest.raises(runner.IntegrityFailure, match=message):
        runner.seal_populations(run_dir)


def test_population_semantic_verifier_rejects_coordinated_raw_hash_update(
    tmp_path: Path,
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    raw_path = run_dir / "populations" / "learned.npz"
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["anchors"][-1, 0, 0] += np.float32(0.0005)
    arrays["anchors"][-1, 0, 1] -= np.float32(0.0005)
    telemetry = [json.loads(str(value)) for value in arrays["telemetry_json"].tolist()]
    coordinated_digest = runner._scientific_row_digest(arrays["anchors"], telemetry)
    arrays["scientific_digest"] = np.asarray([coordinated_digest], dtype=np.str_)
    runner._write_npz(raw_path, **arrays)
    seal_path = run_dir / "populations" / "POPULATIONS_SEALED.json"
    seal = runner._read_json(seal_path)
    seal["rows"]["learned"]["raw_file_sha256"] = runner.sha256_file(raw_path)
    seal["rows"]["learned"]["raw_anchor_array_sha256"] = runner._hash_array(arrays["anchors"])
    seal["rows"]["learned"]["scientific_digest"] = coordinated_digest
    runner._write_json(seal_path, seal)
    with pytest.raises(runner.IntegrityFailure, match="uint8 rendering"):
        runner._verify_population_seal(run_dir)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("start_bank", "start anchor"),
        ("row_seed", "root seed"),
        ("telemetry", "telemetry step inventory|telemetry identity"),
    ],
)
def test_population_semantic_verifier_rejects_coordinated_authority_tamper(
    tmp_path: Path, tamper: str, message: str
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    seal_path = run_dir / "populations" / "POPULATIONS_SEALED.json"
    seal = runner._read_json(seal_path)
    if tamper == "start_bank":
        path = run_dir / "inventory" / "start_bank.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["starts"][0, 0] += np.float32(1e-5)
        arrays["starts"][0, 1] -= np.float32(1e-5)
        runner._write_npz(path, **arrays)
        start_seal_path = run_dir / "inventory" / "START_BANK_SEALED.json"
        start_seal = runner._read_json(start_seal_path)
        start_seal.update(
            {
                "start_bank_sha256": runner.sha256_file(path),
                "starts_sha256": runner._hash_array(arrays["starts"]),
            }
        )
        runner._write_json(start_seal_path, start_seal)
        seal["start_bank_sha256"] = runner.sha256_file(path)
        seal["start_bank_seal_sha256"] = runner.sha256_file(start_seal_path)
    else:
        path = run_dir / "populations" / "learned.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        telemetry = [json.loads(str(value)) for value in arrays["telemetry_json"].tolist()]
        if tamper == "row_seed":
            arrays["root_seed"] = np.asarray([runner.ROW_ROOT_SEEDS["learned"] + 1], dtype=np.uint64)
        else:
            telemetry[0]["completed_step"] = 2
            arrays["telemetry_json"] = np.asarray(
                [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in telemetry],
                dtype=np.str_,
            )
            arrays["scientific_digest"] = np.asarray(
                [runner._scientific_row_digest(arrays["anchors"], telemetry)], dtype=np.str_
            )
            csv_rows = []
            for record in telemetry:
                csv_row = dict(record)
                csv_row["attempts"] = json.dumps(
                    csv_row["attempts"], sort_keys=True, separators=(",", ":")
                )
                csv_rows.append(csv_row)
            runner._write_csv(run_dir / "telemetry" / "learned_steps.csv", csv_rows)
            seal["rows"]["learned"]["telemetry_file_sha256"] = runner.sha256_file(
                run_dir / "telemetry" / "learned_steps.csv"
            )
        runner._write_npz(path, **arrays)
        seal["rows"]["learned"]["raw_file_sha256"] = runner.sha256_file(path)
        seal["rows"]["learned"]["scientific_digest"] = str(arrays["scientific_digest"][0])
    runner._write_json(seal_path, seal)
    with pytest.raises(runner.IntegrityFailure, match=message):
        runner._verify_population_seal(run_dir)


def test_public_verifier_is_read_only_and_preserves_every_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    manifest = runner._seal_manifest(run_dir)
    _patch_verifier_to_population_only(monkeypatch)
    before = {
        path.relative_to(run_dir).as_posix(): (
            path.stat().st_mtime_ns,
            path.stat().st_size,
            runner.sha256_file(path),
        )
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    receipt = runner.verify_run(run_dir)
    assert receipt == {
        "schema": runner.VERSION + "-verification-receipt",
        "passed": 1,
        "state": "resource_stopped",
        "route": "resource_stopped",
        "artifact_count": manifest["artifact_count"],
        "artifact_bytes": manifest["artifact_bytes"],
        "tree_digest": manifest["tree_digest"],
        "read_only": 1,
    }
    after = {
        path.relative_to(run_dir).as_posix(): (
            path.stat().st_mtime_ns,
            path.stat().st_size,
            runner.sha256_file(path),
        )
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_public_verifier_rejects_coordinated_manifest_and_start_seal_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    start_path = run_dir / "inventory" / "start_bank.npz"
    with np.load(start_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["starts"][0, 0] += np.float32(1e-5)
    arrays["starts"][0, 1] -= np.float32(1e-5)
    runner._write_npz(start_path, **arrays)
    start_seal_path = run_dir / "inventory" / "START_BANK_SEALED.json"
    start_seal = runner._read_json(start_seal_path)
    start_seal["start_bank_sha256"] = runner.sha256_file(start_path)
    start_seal["starts_sha256"] = runner._hash_array(arrays["starts"])
    runner._write_json(start_seal_path, start_seal)
    population_seal_path = run_dir / "populations" / "POPULATIONS_SEALED.json"
    population_seal = runner._read_json(population_seal_path)
    population_seal["start_bank_sha256"] = runner.sha256_file(start_path)
    population_seal["start_bank_seal_sha256"] = runner.sha256_file(start_seal_path)
    runner._write_json(population_seal_path, population_seal)
    # The attacker also refreshes both top-level hash authorities. Semantic
    # replay must still catch that row anchor zero no longer matches the start.
    runner._seal_manifest(run_dir)
    _patch_verifier_to_population_only(monkeypatch)
    with pytest.raises(runner.IntegrityFailure, match="start anchor"):
        runner.verify_run(run_dir)


@pytest.mark.parametrize("row_evidence", ["partial", "durable_full"])
@pytest.mark.parametrize("tamper", ["missing", "pixels"])
def test_public_verifier_requires_exact_latest_image_for_failed_row_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row_evidence: str,
    tamper: str,
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    with np.load(run_dir / "populations" / "null.npz", allow_pickle=False) as archive:
        anchors = archive["anchors"].copy()
        labels = archive["labels"].copy()
        path_ids = archive["path_ids"].copy()
        telemetry = [json.loads(str(value)) for value in archive["telemetry_json"].tolist()]
    for relative in ("populations/learned.npz", "telemetry/learned_steps.csv"):
        (run_dir / relative).unlink()
    if row_evidence == "partial":
        (run_dir / "populations" / "null.npz").unlink()
        (run_dir / "telemetry" / "null_steps.csv").unlink()
        partial_anchors = np.stack((anchors[0], anchors[1])).astype(np.float32, copy=False)
        partial_telemetry = telemetry[:8]
        result = runner.RowResult(
            row="null",
            anchors=partial_anchors,
            labels=labels,
            path_ids=path_ids,
            telemetry=partial_telemetry,
            root_seed=runner.ROW_ROOT_SEEDS["null"],
            scientific_digest=runner._scientific_row_digest(
                partial_anchors, partial_telemetry
            ),
            anchor_steps=np.asarray([0, 8], dtype=np.int64),
        )
        runner._persist_partial_failure(run_dir, result)
    else:
        result = runner.RowResult(
            row="null",
            anchors=anchors,
            labels=labels,
            path_ids=path_ids,
            telemetry=telemetry,
            root_seed=runner.ROW_ROOT_SEEDS["null"],
            scientific_digest=runner._scientific_row_digest(anchors, telemetry),
            anchor_steps=np.asarray(runner.ANCHORS, dtype=np.int64),
        )
        runner._persist_durable_full_failure_image(run_dir, result)

    status = {"state": "resource_stopped", "route": "resource_stopped"}
    monkeypatch.setattr(
        runner,
        "_verify_stage_and_status",
        lambda root: (status, ["teacher_row"], "null_row"),
    )
    monkeypatch.setattr(runner, "_verify_lifecycle_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_resource_ledger", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_verify_config_and_sources", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_verify_report_contract", lambda *args, **kwargs: None)
    runner._seal_manifest(run_dir)
    assert runner.verify_run(run_dir)["passed"] == 1

    latest = run_dir / "images" / "partial_null_latest.png"
    if tamper == "missing":
        latest.unlink()
    else:
        with runner.Image.open(latest) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        pixels[0, 0] ^= np.uint8(1)
        runner.Image.fromarray(pixels, mode="L").save(latest)
    # Refreshing both hash authorities must not hide a missing or edited task image.
    runner._seal_manifest(run_dir)
    with pytest.raises(runner.IntegrityFailure, match="contact.sheet"):
        runner.verify_run(run_dir)


def test_evaluator_and_terminal_test_are_unreachable_before_population_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal_called = False
    load_called = False

    def forbidden_terminal(*args: object, **kwargs: object):
        nonlocal terminal_called
        terminal_called = True
        raise AssertionError("terminal evidence was opened before population seal")

    def forbidden_load(*args: object, **kwargs: object):
        nonlocal load_called
        load_called = True
        raise AssertionError("evaluator was loaded before population seal")

    monkeypatch.setattr(runner, "read_mnist_arff_slice", forbidden_terminal)
    monkeypatch.setattr(runner.torch, "load", forbidden_load)
    with pytest.raises((runner.IntegrityFailure, FileNotFoundError)):
        runner.evaluate_sealed_populations(
            tmp_path / "unsealed", arff_path=tmp_path / "missing.arff", device="cpu"
        )
    with pytest.raises((runner.IntegrityFailure, FileNotFoundError)):
        runner._load_evaluator_after_population_seal(tmp_path / "unsealed", device="cpu")
    assert not terminal_called and not load_called


def test_evaluator_extracts_state_dict_with_strict_weights_only_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    checkpoint = run_dir / "input_bindings" / "selected_checkpoint.pt"
    checkpoint.write_bytes(b"synthetic-evaluator")
    binding_path = run_dir / "input_bindings" / "ddpm_evaluator_binding.json"
    runner._write_json(binding_path, {"copied_checkpoint": "input_bindings/selected_checkpoint.pt"})
    runner._write_json(
        run_dir / "evaluation" / "EVALUATOR_OPEN_EVENT.json",
        {
            "population_seal_sha256": runner.sha256_file(
                run_dir / "populations" / "POPULATIONS_SEALED.json"
            ),
            "evaluator_binding_sha256": runner.sha256_file(binding_path),
        },
    )
    calls: list[dict[str, object]] = []
    state = runner.SmallMnistCNN().state_dict()

    def fake_load(path: Path, **kwargs: object) -> object:
        calls.append({"path": path, **kwargs})
        return {"state_dict": state, "selected_epoch": 3}

    monkeypatch.setattr(runner, "EVALUATOR_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(runner, "EVALUATOR_SHA256", runner.sha256_file(checkpoint))
    monkeypatch.setattr(runner.torch, "load", fake_load)
    model = runner._load_evaluator_after_population_seal(run_dir, device="cpu")
    assert calls == [{"path": checkpoint, "map_location": "cpu", "weights_only": True}]
    assert model.training is False

    monkeypatch.setattr(
        runner.torch,
        "load",
        lambda *args, **kwargs: {"model": state, "selected_epoch": 3},
    )
    with pytest.raises(runner.IntegrityFailure, match="payload schema"):
        runner._load_evaluator_after_population_seal(run_dir, device="cpu")


def test_evaluation_verifier_uses_bound_device_and_bitwise_logit_replay() -> None:
    assert runner._bound_evaluator_replay_device(
        {"execution_authority": {"device": "cuda:0"}}
    ) == "cuda:0"
    for invalid in ({}, {"execution_authority": {}}, {"execution_authority": {"device": ""}}):
        with pytest.raises(runner.IntegrityFailure, match="evaluator replay"):
            runner._bound_evaluator_replay_device(invalid)

    sample_ids = np.asarray(["efr-v1-000", "efr-v1-001"], dtype=np.str_)
    labels = np.asarray([0, 1], dtype=np.int64)
    predictions = labels.copy()
    recorded = np.zeros((2, 10), dtype=np.float64)
    replay_arrays = {
        "sample_ids": sample_ids.copy(),
        "requested_labels": labels.copy(),
        "predictions": predictions.copy(),
        "logits": recorded.copy(),
    }
    runner._verify_evaluator_replay_arrays(
        "learned",
        replay_arrays,
        sample_ids=sample_ids,
        requested_labels=labels,
        predictions=predictions,
        logits=recorded,
    )
    one_coordinate_tamper = recorded.copy()
    one_coordinate_tamper[1, 4] = np.nextafter(np.float64(0.0), np.float64(1.0))
    assert np.allclose(one_coordinate_tamper, recorded, rtol=2e-5, atol=2e-5)
    with pytest.raises(runner.IntegrityFailure, match="replay logits changed"):
        runner._verify_evaluator_replay_arrays(
            "learned",
            {**replay_arrays, "logits": one_coordinate_tamper},
            sample_ids=sample_ids,
            requested_labels=labels,
            predictions=predictions,
            logits=recorded,
        )

    tree = ast.parse(inspect.getsource(runner._verify_evaluation))
    replay_device_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "replay_device"
            for target in node.targets
        )
    ]
    assert len(replay_device_assignments) == 1
    assert ast.unparse(replay_device_assignments[0].value) == (
        "_bound_evaluator_replay_device(run_config)"
    )

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    evaluator_loads = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_load_evaluator_after_population_seal"
    ]
    metric_replays = [
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "compute_generation_metrics"
    ]
    assert len(evaluator_loads) == len(metric_replays) == 1
    for call in evaluator_loads + metric_replays:
        device = next(keyword.value for keyword in call.keywords if keyword.arg == "device")
        assert isinstance(device, ast.Name) and device.id == "replay_device"

    array_verifications = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_verify_evaluator_replay_arrays"
    ]
    assert len(array_verifications) == 1


def test_test_open_event_precedes_terminal_fetch_and_metrics_json_stays_compact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    arff = tmp_path / "authenticated-synthetic.arff"
    arff.write_bytes(b"synthetic ARFF authority; content fetch is mocked")
    monkeypatch.setattr(runner, "MNIST_ARFF_SHA256", runner.sha256_file(arff))
    development = np.broadcast_to(np.zeros((1, 28, 28), dtype=np.uint8), (60_000, 28, 28))
    development_labels = np.resize(np.arange(10, dtype=np.int64), 60_000)
    terminal = np.zeros((160, 28, 28), dtype=np.uint8)
    terminal_labels = np.repeat(np.arange(10, dtype=np.int64), 16)
    opened = False

    def terminal_loader(path: Path, start: int, stop: int):
        nonlocal opened
        event_path = run_dir / "data" / "test_open_event.json"
        assert event_path.is_file()
        event = runner._read_json(event_path)
        assert event["population_seal_sha256"] == runner.sha256_file(
            run_dir / "populations" / "POPULATIONS_SEALED.json"
        )
        assert (start, stop) == (60_000, 70_000)
        opened = True
        return terminal, terminal_labels

    def fake_metrics(model: object, images: np.ndarray, labels: np.ndarray, ids: np.ndarray, **kwargs: object):
        return {
            "classifier": {
                "requested_label_accuracy": 0.1,
                "predictions": labels.copy(),
                "logits": np.zeros((len(labels), 10), dtype=np.float64),
            },
            "duplicates": {"duplicate_pair_count": 0},
            "diversity": {"aggregate_median_ratio": 0.5},
            "exact_reference_match_count": {},
        }

    monkeypatch.setattr(runner, "read_mnist_arff_slice", terminal_loader)
    monkeypatch.setattr(runner, "_load_evaluator_after_population_seal", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "compute_generation_metrics", fake_metrics)

    def fake_teacher_control(root: Path, **_: object) -> dict[str, object]:
        report: dict[str, object] = {"passed": 1, "synthetic": 1}
        runner._write_json(root / "controls" / "teacher_gate.json", report)
        return report

    monkeypatch.setattr(runner, "_teacher_positive_control", fake_teacher_control)
    runner._write_json(run_dir / "input_bindings" / "ddpm_evaluator_binding.json", {"bound": 1})
    runner.evaluate_sealed_populations(
        run_dir,
        arff_path=arff,
        device="cpu",
        development_images=development,
        development_labels=development_labels,
    )
    assert opened
    learned = runner._read_json(run_dir / "evaluation" / "learned_metrics.json")
    assert "predictions" not in learned["classifier"] and "logits" not in learned["classifier"]
    with np.load(run_dir / "evaluation" / "predictions.npz", allow_pickle=False) as predictions:
        assert predictions["learned_predictions"].shape == (160,)
        assert predictions["learned_logits"].shape == (160, 10)


def test_record_review_stratifies_hidden_rows_and_never_auto_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    runner.prepare_blind_review(run_dir)
    _write_synthetic_review_authorities(run_dir)
    key = runner._read_json(run_dir / "review" / "review_key.json")["entries"]
    membership = {
        entry["member_id"]: entry
        for entry in runner._read_json(run_dir / "review" / "private_membership.json")["entries"]
    }
    answers = tmp_path / "answers.csv"
    with answers.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["review_order", "sample_id", "assigned_label", "notes"]
        )
        writer.writeheader()
        for entry in key:
            member = membership[entry["source_sample_id"]]
            assignment = str(member["requested_label"]) if member["row"] == "learned" else "noise"
            writer.writerow(
                {
                    "review_order": entry["review_order"],
                    "sample_id": entry["sample_id"],
                    "assigned_label": assignment,
                    "notes": "manual",
                }
            )
    verify_states: list[str] = []

    def fake_verify(path: Path) -> dict[str, object]:
        ledger = runner._read_json(Path(path) / "resource_ledger.json")
        assert ledger["open_events"] == []
        verify_states.append(str(runner._read_json(Path(path) / "status.json")["state"]))
        return {"passed": 1, "tree_digest": "synthetic"}

    monkeypatch.setattr(runner, "verify_run", fake_verify)
    outcome = runner.record_review(
        run_dir, answers, reviewer="manual-reviewer", confirm_manual_review=True
    )
    assert outcome["route"] == "factor_one_feasible"
    assert outcome["human_marker"]["learned"]["requested_label_agreement_count"] == 40
    assert outcome["human_marker"]["null"]["recognizable_count"] == 0
    assert all(
        values["count"] == 4
        for values in outcome["human_marker"]["learned"]["by_class"].values()
    )
    assert outcome["full_scale_auto_launched"] == 0
    assert verify_states == ["awaiting_human_review", "complete"]
    assert "separately" not in outcome["next_action"].lower() or "approval" in outcome["next_action"].lower()
    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["active_seconds"] > 7.25
    assert [event["event"] for event in ledger["events"][-2:]] == ["admit", "complete"]
    assert all(event["kind"] == "human_review_terminalization" for event in ledger["events"][-2:])
    gate_e = runner._read_json(run_dir / "gates.json")["gate_e"]
    assert gate_e == {
        "gate_type": "diagnostic threshold",
        "state": "complete",
        "passed": 1,
        "conditions": {
            "gates_a_to_d_passed": 1,
            "learned_human_recognizability_at_least_0_90": 1,
            "learned_human_requested_label_agreement_at_least_0_75": 1,
            "learned_classifier_accuracy_at_least_0_80": 1,
            "learned_zero_duplicate_pairs": 1,
            "learned_diversity_ratio_at_least_0_25": 1,
            "learned_human_agreement_exceeds_null": 1,
            "learned_classifier_accuracy_exceeds_null": 1,
        },
        "values": {
            "learned_human_recognizability": 1.0,
            "learned_human_requested_label_agreement": 1.0,
            "null_human_requested_label_agreement": 0.0,
            "learned_classifier_accuracy": 0.9,
            "null_classifier_accuracy": 0.1,
            "learned_duplicate_pair_count": 0,
            "learned_diversity_ratio": 0.5,
        },
    }
    positive_claim = " ".join(
        (
            "On 160 prespecified fresh low-frequency source measures, the hash-pinned historical "
            "global edge-flux checkpoint under the bound current sampler produced a factor-one, "
            "unselected exploratory population that met the frozen recognizability, requested-label, "
            "uniqueness, and diversity markers and exceeded the separately randomized zero-conditioning "
            "population in aggregate human and classifier label agreement."
        ).split()
    )
    evidence_paths = (
        "controls/teacher_gate.json",
        "controls/teacher_gate_arrays.npz",
        "evaluation/learned_metrics.json",
        "evaluation/predictions.npz",
        "review/human_review_by_row.json",
        "review/human_review_answers.csv",
        "populations/POPULATIONS_SEALED.json",
        "gates.json",
        "outcome.json",
    )
    raw_scalars = {
        "median_relative_squared_l2_anchor64": 0.5,
        "median_relative_squared_l2_endpoint": 0.1,
        "endpoint_improved_path_count": 160,
        "teacher_requested_label_accuracy": 0.95,
        **gate_e["values"],
    }
    for name in ("REPORT.md", "HANDOFF.md"):
        document = (run_dir / name).read_text(encoding="utf-8")
        normalized = " ".join(document.split())
        assert "factor_one_feasible" in document
        assert "Gate E" in document and "complete" in document and "passed" in document
        assert positive_claim in normalized
        for key, value in raw_scalars.items():
            assert key in document and f"`{value}`" in document
        for relative in evidence_paths:
            assert relative in document
        assert "No larger population" in document and "automatic DSM" in document

    handoff = (run_dir / "HANDOFF.md").read_text(encoding="utf-8")
    required_handoff_sections = (
        "## 1. Program objective",
        "## 2. Current milestone and distance to goal",
        "## 3. Strategy review",
        "## 4. Research mode and evidence roles",
        "## 5. Exact result of the latest run",
        "## 6. Confirmed facts, current inferences, and open hypotheses",
        "## 7. Decision the next patch must resolve",
        "## 8. Candidate actions and value of information",
        "## 9. Recommended next patch",
        "## 10. Gates and claim boundaries",
        "## 11. Outcome-to-action table",
        "## 12. Constraints",
        "## 13. Resource budget and stop rule",
        "## 14. Alternative and pivot plan",
        "## 15. Evidence map",
        "## 16. Deliberate omissions",
        "## 17. Reproduction commands",
        "## 18. Bundle-integrity audit",
        "## 19. Exact deliverable for the receiving agent",
    )
    assert all(section in handoff for section in required_handoff_sections)

    gates = runner._read_json(run_dir / "gates.json")
    evaluation = {
        "rows": {
            row: runner._read_json(run_dir / "evaluation" / f"{row}_metrics.json")
            for row in ("learned", "null")
        }
    }
    sealed_outcome = runner._read_json(run_dir / "outcome.json")
    human = {
        "rows": {
            "learned": sealed_outcome["human_marker"]["learned"],
            "null": sealed_outcome["human_marker"]["null"],
        }
    }
    status = runner._read_json(run_dir / "status.json")
    assert (
        runner._verify_complete_outcome(run_dir, status, gates, human, evaluation)
        == sealed_outcome
    )
    coordinated_outcome = dict(sealed_outcome)
    coordinated_outcome["next_action"] = "silently launch another experiment"
    runner._write_json(run_dir / "outcome.json", coordinated_outcome)
    with pytest.raises(runner.IntegrityFailure, match="prespecified outcome replay changed"):
        runner._verify_complete_outcome(run_dir, status, gates, human, evaluation)
    runner._write_json(run_dir / "outcome.json", sealed_outcome)

    runner._verify_report_contract(run_dir, status)
    report_path = run_dir / "REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    scalar_line = "  - learned_classifier_accuracy: `0.9`"
    assert scalar_line in report
    report_path.write_text(
        report.replace(scalar_line, "  - learned_classifier_accuracy: `0.8`", 1),
        encoding="utf-8",
    )
    with pytest.raises(runner.IntegrityFailure, match="REPORT.md.*[Gg]ate E|[Rr]eport.*scalar"):
        runner._verify_report_contract(run_dir, status)


@pytest.mark.parametrize("tamper", ["scoring", "review"])
def test_record_review_verifies_both_readiness_seals_before_copying_answers(
    tmp_path: Path, tamper: str
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    runner.prepare_blind_review(run_dir)
    _write_synthetic_review_authorities(run_dir)
    if tamper == "scoring":
        path = run_dir / "evaluation" / "SCORING_READY.json"
        authority = runner._read_json(path)
        authority["population_seal_sha256"] = "0" * 64
    else:
        path = run_dir / "review" / "READY.json"
        authority = runner._read_json(path)
        authority["template_sha256"] = "0" * 64
    runner._write_json(path, authority)
    answers = tmp_path / "answers.csv"
    answers.write_text("review_order,sample_id,assigned_label,notes\n", encoding="utf-8")
    with pytest.raises(runner.IntegrityFailure, match="[Ss]coring|[Rr]eview|template|population"):
        runner.record_review(
            run_dir,
            answers,
            reviewer="manual-reviewer",
            confirm_manual_review=True,
        )
    assert not (run_dir / "review" / "human_review_answers.csv").exists()
    assert not (run_dir / "outcome.json").exists()


def test_invalid_review_answers_after_admission_end_in_a_sealed_integrity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    runner.prepare_blind_review(run_dir)
    _write_synthetic_review_authorities(run_dir)
    answers = tmp_path / "invalid-answers.csv"
    answers.write_text(
        "review_order,sample_id,assigned_label,notes\n0,blind-000,cat,invalid\n",
        encoding="utf-8",
    )
    verify_states: list[str] = []

    def fake_verify(path: Path) -> dict[str, object]:
        root = Path(path)
        assert runner._read_json(root / "resource_ledger.json")["open_events"] == []
        state = str(runner._read_json(root / "status.json")["state"])
        verify_states.append(state)
        return {"passed": 1, "state": state, "tree_digest": "synthetic"}

    monkeypatch.setattr(runner, "verify_run", fake_verify)
    result = runner.record_review(
        run_dir,
        answers,
        reviewer="manual-reviewer",
        confirm_manual_review=True,
    )

    assert result["state"] == result["route"] == "integrity_failed"
    assert result["failed_stage"] == "human_review_terminalization"
    assert result["error_type"] == "IntegrityFailure"
    assert result["verification"]["passed"] == 1
    assert verify_states == ["awaiting_human_review", "integrity_failed"]
    assert (run_dir / "review" / "human_review_answers.csv").is_file()
    assert not (run_dir / "outcome.json").exists()
    assert runner._read_json(run_dir / "gates.json")["gate_e"]["state"] == "pending"
    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert [
        event["event"]
        for event in ledger["events"]
        if event["kind"] == "human_review_terminalization"
    ] == ["admit", "failed-complete"]
    runner._verify_resource_ledger(run_dir, runner._read_json(run_dir / "status.json"))


def test_record_review_reentry_closes_one_interrupted_human_quantum_then_retries_whole(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    runner.prepare_blind_review(run_dir)
    _write_synthetic_review_authorities(run_dir)
    answers = tmp_path / "answers.csv"
    _write_valid_review_answers(run_dir, answers)
    interrupted = runner.ResourceGovernor.rehydrate(run_dir, device="cpu")
    interrupted.admit(
        "human_review_terminalization",
        predicted_seconds=10.0,
        predicted_next_bytes=2 * 1024 * 1024,
        reserve_remaining_seconds=0.0,
    )
    verify_states: list[str] = []

    def fake_verify(path: Path) -> dict[str, object]:
        root = Path(path)
        assert runner._read_json(root / "resource_ledger.json")["open_events"] == []
        state = str(runner._read_json(root / "status.json")["state"])
        verify_states.append(state)
        return {"passed": 1, "state": state, "tree_digest": "synthetic"}

    monkeypatch.setattr(runner, "verify_run", fake_verify)
    result = runner.record_review(
        run_dir,
        answers,
        reviewer="manual-reviewer",
        confirm_manual_review=True,
    )

    assert result["route"] == "factor_one_feasible"
    assert verify_states == ["awaiting_human_review", "complete"]
    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert [
        event["event"]
        for event in ledger["events"]
        if event["kind"] == "human_review_terminalization"
    ] == ["admit", "interrupted-close", "admit", "complete"]
    runner._verify_resource_ledger(run_dir, runner._read_json(run_dir / "status.json"))


def test_human_review_post_completion_cap_stop_retains_review_evidence_without_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    runner.prepare_blind_review(run_dir)
    _write_synthetic_review_authorities(run_dir)
    answers = tmp_path / "answers.csv"
    _write_valid_review_answers(run_dir, answers)
    cuda_receipts = iter(
        [
            (0, 100, 0.0),
            (80, 100, 0.8),
            (0, 100, 0.0),
            (0, 100, 0.0),
        ]
    )
    monkeypatch.setattr(
        runner.ResourceGovernor,
        "_cuda_receipt",
        lambda self: next(cuda_receipts),
    )
    verify_states: list[str] = []

    def fake_verify(path: Path) -> dict[str, object]:
        root = Path(path)
        assert runner._read_json(root / "resource_ledger.json")["open_events"] == []
        state = str(runner._read_json(root / "status.json")["state"])
        verify_states.append(state)
        return {"passed": 1, "state": state, "tree_digest": "synthetic"}

    monkeypatch.setattr(runner, "verify_run", fake_verify)
    result = runner.record_review(
        run_dir,
        answers,
        reviewer="manual-reviewer",
        confirm_manual_review=True,
    )

    assert result["state"] == result["route"] == "resource_stopped"
    assert result["failed_stage"] == "human_review_terminalization"
    assert result["error_type"] == "ResourceStop"
    assert result["verification"]["passed"] == 1
    assert verify_states == ["awaiting_human_review", "resource_stopped"]
    for relative in (
        "review/human_review_answers.csv",
        "review/human_review.json",
        "review/human_review_by_row.json",
    ):
        assert (run_dir / relative).is_file()
    assert not (run_dir / "outcome.json").exists()
    assert runner._read_json(run_dir / "gates.json")["gate_e"] == {
        "gate_type": "diagnostic threshold",
        "state": "pending",
        "passed": None,
        "conditions": {},
    }
    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert ledger["failed_admission"]["kind"] == "human_review_terminalization"
    assert ledger["failed_admission"]["phase"] == "post-completion"
    runner._verify_resource_ledger(run_dir, runner._read_json(run_dir / "status.json"))


@pytest.mark.parametrize(
    ("state", "expected"),
    [("complete", 0), ("resource_stopped", 3), ("integrity_failed", 4)],
)
def test_record_review_cli_returns_the_terminal_route_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: str,
    expected: int,
) -> None:
    monkeypatch.setattr(
        runner,
        "record_review",
        lambda *args, **kwargs: {"passed": int(state == "complete"), "state": state},
    )
    code = runner.main(
        [
            "record-review",
            "--run-dir",
            "synthetic-run",
            "--answers",
            "synthetic-answers.csv",
            "--reviewer",
            "manual-reviewer",
            "--confirm-manual-review",
        ]
    )
    assert code == expected
    assert json.loads(capsys.readouterr().out)["state"] == state


@pytest.mark.parametrize(
    ("gates", "human", "classifier", "noncollapse", "exceeds", "expected"),
    [
        (False, True, True, True, True, "invalid_repair_same_experiment"),
        (True, True, True, True, True, "factor_one_feasible"),
        (True, True, False, True, True, "human_positive_evaluator_disagreement"),
        (True, False, True, True, True, "factor_one_negative_stop_checkpoint_line"),
        (True, True, True, True, False, "factor_one_negative_stop_checkpoint_line"),
        (True, True, True, False, True, "factor_one_negative_stop_checkpoint_line"),
    ],
)
def test_outcome_truth_table(
    gates: bool,
    human: bool,
    classifier: bool,
    noncollapse: bool,
    exceeds: bool,
    expected: str,
) -> None:
    assert runner.route_outcome(
        gates_a_to_d_passed=gates,
        learned_human_positive=human,
        learned_classifier_positive=classifier,
        learned_noncollapse_positive=noncollapse,
        learned_exceeds_null=exceeds,
    ) == expected


@pytest.mark.parametrize("state", ["awaiting_human_review", "complete"])
def test_terminal_run_reentry_is_verify_only_and_does_not_mutate_the_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    run_dir = tmp_path / "terminal"
    runner._write_json(run_dir / "status.json", {"state": state})
    marker = run_dir / "populations" / "POPULATIONS_SEALED.json"
    runner._write_json(marker, {"synthetic": 1})
    before = {
        path.relative_to(run_dir).as_posix(): (
            path.stat().st_mtime_ns,
            runner.sha256_file(path),
        )
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    verified: list[Path] = []
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        runner,
        "verify_run",
        lambda path: verified.append(Path(path))
        or {"passed": 1, "state": state, "tree_digest": "synthetic"},
    )
    for name in ("run_null_row", "run_teacher_row", "run_learned_row", "seal_populations"):
        monkeypatch.setattr(
            runner,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"{_name} must be unreachable for a terminal run"
            ),
        )
    args = SimpleNamespace(
        run_dir=str(run_dir),
        legacy_checkpoint="unused.pt",
        ddpm_run_dir="unused-ddpm",
        k128_run_dir="unused-k128",
        arff="unused.arff",
        device="cuda:0",
        approval_id="synthetic-approved-reentry",
        max_active_seconds=runner.MAX_ACTIVE_SECONDS,
        max_storage_mib=runner.MAX_STORAGE_MIB,
        max_cuda_fraction=runner.MAX_CUDA_FRACTION,
    )
    assert runner.run_production(args) == 0
    assert verified and set(verified) == {run_dir.resolve()}
    after = {
        path.relative_to(run_dir).as_posix(): (
            path.stat().st_mtime_ns,
            runner.sha256_file(path),
        )
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_reentry_classification_distinguishes_terminal_sealed_and_unsealed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified: list[Path] = []
    monkeypatch.setattr(
        runner,
        "verify_run",
        lambda path: verified.append(Path(path)) or {"passed": 1},
    )
    for state in (
        "awaiting_human_review",
        "complete",
        "resource_stopped",
        "integrity_failed",
    ):
        terminal = tmp_path / state
        runner._write_json(terminal / "status.json", {"state": state})
        assert runner._classify_reentry(terminal) == "verify_only"
    assert verified == [tmp_path / "awaiting_human_review", tmp_path / "complete"]

    sealed = tmp_path / "sealed"
    runner._write_json(sealed / "status.json", {"state": "running"})
    runner._write_json(sealed / "populations" / "POPULATIONS_SEALED.json", {"sealed": 1})
    assert runner._classify_reentry(sealed) == "continue_sealed"

    unsealed = tmp_path / "unsealed"
    runner._write_json(unsealed / "status.json", {"state": "running"})
    runner._write_json(unsealed / "inventory" / "START_BANK_SEALED.json", {"sealed": 1})
    assert runner._classify_reentry(unsealed) == "rerun_all_rows"

    missing_authority = tmp_path / "missing-authority"
    runner._write_json(missing_authority / "status.json", {"state": "running"})
    with pytest.raises(runner.IntegrityFailure, match="sealed start authority"):
        runner._classify_reentry(missing_authority)


@pytest.mark.parametrize(
    ("state", "terminal_kind", "completed"),
    [
        ("awaiting_human_review", "machine_terminalization", False),
        ("awaiting_human_review", "machine_terminalization", True),
        ("complete", "human_review_terminalization", False),
        ("complete", "human_review_terminalization", True),
    ],
)
def test_terminal_crash_window_continues_one_authenticated_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    terminal_kind: str,
    completed: bool,
) -> None:
    run_dir = tmp_path / state
    runner._write_json(run_dir / "status.json", {"state": state})
    runner._write_json(run_dir / "populations" / "POPULATIONS_SEALED.json", {"sealed": 1})
    governor = runner.ResourceGovernor(
        run_dir,
        runner.ResourceBudget(reserve_seconds=0.0),
        device="cpu",
    )
    governor.admit(
        terminal_kind,
        predicted_seconds=1.0,
        predicted_next_bytes=1,
        reserve_remaining_seconds=0.0,
    )
    if completed:
        governor.complete(terminal_kind)
    monkeypatch.setattr(
        runner,
        "verify_run",
        lambda _: (_ for _ in ()).throw(runner.IntegrityFailure("stale terminal manifest")),
    )
    monkeypatch.setattr(
        runner,
        "_verify_terminal_recovery_evidence",
        lambda *args, **kwargs: None,
    )

    assert runner._classify_reentry(run_dir) == "continue_sealed"

    ledger = runner._read_json(run_dir / "resource_ledger.json")
    ledger["events"].append({"event": "admit", "kind": "unexpected"})
    runner._write_json(run_dir / "resource_ledger.json", ledger)
    with pytest.raises(runner.IntegrityFailure, match="not a recoverable terminalization crash"):
        runner._classify_reentry(run_dir)


def test_terminal_recovery_semantic_tamper_is_rejected_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _synthetic_population_run(tmp_path / "run")
    runner.seal_populations(run_dir)
    runner._write_json(
        run_dir / "status.json",
        {"state": "awaiting_human_review", "route": "awaiting_human_review"},
    )
    governor = runner.ResourceGovernor(
        run_dir,
        runner.ResourceBudget(reserve_seconds=0.0),
        device="cpu",
    )
    governor.admit(
        "machine_terminalization",
        predicted_seconds=1.0,
        predicted_next_bytes=1,
        reserve_remaining_seconds=0.0,
    )
    row_path = run_dir / "populations" / "learned.npz"
    with np.load(row_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["root_seed"] = np.asarray(
        [runner.ROW_ROOT_SEEDS["learned"] + 1], dtype=np.uint64
    )
    runner._write_npz(row_path, **arrays)
    seal_path = run_dir / "populations" / "POPULATIONS_SEALED.json"
    seal = runner._read_json(seal_path)
    seal["rows"]["learned"]["raw_file_sha256"] = runner.sha256_file(row_path)
    runner._write_json(seal_path, seal)
    monkeypatch.setattr(
        runner,
        "verify_run",
        lambda _: (_ for _ in ()).throw(runner.IntegrityFailure("stale manifest")),
    )
    before = {
        path.relative_to(run_dir).as_posix(): (
            path.stat().st_mtime_ns,
            path.stat().st_size,
            runner.sha256_file(path),
        )
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    with pytest.raises(runner.IntegrityFailure, match="root seed"):
        runner._classify_reentry(run_dir)

    after = {
        path.relative_to(run_dir).as_posix(): (
            path.stat().st_mtime_ns,
            path.stat().st_size,
            runner.sha256_file(path),
        )
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_unsealed_restart_clears_every_derived_row_but_preserves_frozen_authorities(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    authority_paths = (
        "config.json",
        "input_bindings/checkpoint.json",
        "inventory/start_bank.npz",
        "inventory/START_BANK_SEALED.json",
        "inventory/teacher_target_bank.npz",
        "data/development_roles.json",
    )
    for index, relative in enumerate(authority_paths):
        path = run_dir / relative
        if path.suffix == ".npz":
            runner._write_npz(path, values=np.asarray([index], dtype=np.int64))
        else:
            runner._write_json(path, {"authority": index})
    frozen = {relative: runner.sha256_file(run_dir / relative) for relative in authority_paths}

    stage_events = [
        {"stage": stage, "state": "completed", "recorded_at": f"t{index}"}
        for index, stage in enumerate(
            (
                "initialize_and_bind",
                "checkpoint_extract",
                "data_and_inventory",
                "preflight",
                "teacher_row",
            )
        )
    ]
    runner._write_json(
        run_dir / "stage_ledger.json",
        {"schema": runner.VERSION + "-stage-ledger", "events": stage_events},
    )
    for relative in (
        "populations/teacher.npz",
        "populations/partial_null.npz",
        "preflight/deterministic_replay.json",
        "preflight/resource_projection.json",
        "telemetry/teacher.jsonl",
        "images/teacher.png",
        "controls/teacher_gate.json",
        "evaluation/learned_metrics.json",
        "review/READY.json",
        "data/test_open_event.json",
        "gates.json",
        "outcome.json",
        "failure.json",
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "REPORT.md",
        "HANDOFF.md",
    ):
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"derived")

    runner._clear_unsealed_outputs(run_dir)

    assert {
        relative: runner.sha256_file(run_dir / relative) for relative in authority_paths
    } == frozen
    assert runner._stage_events(run_dir) == stage_events[:3]
    for directory in (
        "preflight",
        "populations",
        "telemetry",
        "images",
        "controls",
        "evaluation",
        "review",
    ):
        assert list((run_dir / directory).iterdir()) == []
    assert not (run_dir / "data" / "test_open_event.json").exists()
    for relative in (
        "gates.json",
        "outcome.json",
        "failure.json",
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "REPORT.md",
        "HANDOFF.md",
    ):
        assert not (run_dir / relative).exists()


def test_restart_authority_is_append_only_and_never_rewrites_frozen_authorities(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    runner._write_json(run_dir / "config.json", {"frozen": 1})
    runner._write_json(
        run_dir / "inventory" / "START_BANK_SEALED.json",
        {"config_sha256": runner.sha256_file(run_dir / "config.json"), "frozen": 1},
    )
    start_bank = run_dir / "inventory" / "start_bank.npz"
    runner._write_npz(start_bank, starts=np.asarray([[1.0]], dtype=np.float32))
    frozen = {
        relative: runner.sha256_file(run_dir / relative)
        for relative in (
            "config.json",
            "inventory/START_BANK_SEALED.json",
            "inventory/start_bank.npz",
        )
    }

    first = runner._record_restart_authority(run_dir, mode="rerun_all_rows")
    second = runner._record_restart_authority(run_dir, mode="rerun_all_rows")

    assert [first["restart_index"], second["restart_index"]] == [1, 2]
    assert first["partial_resume_used"] == second["partial_resume_used"] == 0
    assert {
        relative: runner.sha256_file(run_dir / relative)
        for relative in frozen
    } == frozen
    history = runner._read_json(run_dir / "restart_history.json")
    assert [event["mode"] for event in history["events"]] == [
        "rerun_all_rows",
        "rerun_all_rows",
    ]


def test_sealed_restart_authority_never_mutates_config_or_start_seal(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    runner._write_json(run_dir / "config.json", {"frozen": 1})
    runner._write_json(run_dir / "inventory" / "START_BANK_SEALED.json", {"frozen": 1})
    runner._write_json(run_dir / "populations" / "POPULATIONS_SEALED.json", {"sealed": 1})
    before = {
        relative: runner.sha256_file(run_dir / relative)
        for relative in ("config.json", "inventory/START_BANK_SEALED.json")
    }

    event = runner._record_restart_authority(run_dir, mode="continue_sealed")

    assert event["population_sealed_before_restart"] == 1
    assert event["partial_resume_used"] == 0
    assert {
        relative: runner.sha256_file(run_dir / relative)
        for relative in ("config.json", "inventory/START_BANK_SEALED.json")
    } == before


def _production_args(run_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_dir=str(run_dir),
        legacy_checkpoint="frozen-checkpoint.pt",
        ddpm_run_dir="frozen-ddpm",
        k128_run_dir="frozen-k128",
        arff="frozen.arff",
        device="cuda:0",
        approval_id="synthetic-approved-reentry",
        max_active_seconds=runner.MAX_ACTIVE_SECONDS,
        max_storage_mib=runner.MAX_STORAGE_MIB,
        max_cuda_fraction=runner.MAX_CUDA_FRACTION,
    )


def _synthetic_reentry_base() -> tuple[object, ...]:
    inventory = runner.build_path_inventory()
    starts = np.full((runner.PATH_COUNT, 784), np.float32(1 / 784), dtype=np.float32)
    targets = {"masses": starts.copy()}
    governor = SimpleNamespace(admit=lambda *args, **kwargs: None, complete=lambda *args: None)
    return (
        governor,
        {"frozen": 1},
        _tiny_config(grid_size=28),
        np.empty((0, 784), dtype=np.uint8),
        np.empty(0, dtype=np.int64),
        inventory,
        starts,
        targets,
    )


def test_run_production_unsealed_reentry_skips_bound_authorities_and_starts_at_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    runner._write_json(run_dir / "status.json", {"state": "running"})
    calls: list[str] = []
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner, "_classify_reentry", lambda *args: "rerun_all_rows")
    monkeypatch.setattr(runner, "_validate_reentry_base", lambda *args, **kwargs: _synthetic_reentry_base())
    monkeypatch.setattr(runner, "_record_restart_authority", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_clear_unsealed_outputs", lambda *args: calls.append("clear"))
    monkeypatch.setattr(runner, "_reset_running_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_synthetic_teacher_preflight",
        lambda *args, **kwargs: calls.append("preflight")
        or (_ for _ in ()).throw(runner.IntegrityFailure("synthetic preflight stop")),
    )

    def forbidden(name: str):
        def fail(*_: object, **__: object) -> None:
            calls.append(name)
            raise runner.IntegrityFailure(f"forbidden reentry call: {name}")

        return fail

    monkeypatch.setattr(runner, "_bind_external_authorities", forbidden("bind"))
    monkeypatch.setattr(runner, "safe_extract_legacy_checkpoint", forbidden("extract"))
    monkeypatch.setattr(runner, "read_mnist_development_prefix", forbidden("read_development"))
    monkeypatch.setattr(runner, "_write_inventory_authorities", forbidden("write_inventory"))
    monkeypatch.setattr(runner, "_finalize_failure", lambda *args, **kwargs: {"synthetic": 1})

    assert runner.run_production(_production_args(run_dir)) == 4
    assert calls == ["clear", "preflight"]


def test_run_production_population_sealed_reentry_never_repeats_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    runner._write_json(run_dir / "status.json", {"state": "running"})
    runner._write_json(run_dir / "populations" / "POPULATIONS_SEALED.json", {"sealed": 1})
    runner._write_json(
        run_dir / "stage_ledger.json",
        {
            "schema": runner.VERSION + "-stage-ledger",
            "events": [
                {"stage": stage, "state": "completed", "recorded_at": "t"}
                for stage in runner.STAGE_ORDER[: runner.STAGE_ORDER.index("scoring")]
            ],
        },
    )
    calls: list[str] = []
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner, "_classify_reentry", lambda *args: "continue_sealed")
    monkeypatch.setattr(runner, "_validate_reentry_base", lambda *args, **kwargs: _synthetic_reentry_base())
    monkeypatch.setattr(runner, "_verify_population_seal", lambda *args: {"sealed": 1})
    monkeypatch.setattr(runner, "_verify_one_row_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_telemetry_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_record_restart_authority", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_reset_running_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "evaluate_sealed_populations",
        lambda *args, **kwargs: calls.append("scoring")
        or (_ for _ in ()).throw(runner.IntegrityFailure("synthetic scoring stop")),
    )

    def forbidden(name: str):
        def fail(*_: object, **__: object) -> None:
            calls.append(name)
            raise runner.IntegrityFailure(f"forbidden sealed-reentry call: {name}")

        return fail

    for name, attribute in (
        ("bind", "_bind_external_authorities"),
        ("extract", "safe_extract_legacy_checkpoint"),
        ("read_development", "read_mnist_development_prefix"),
        ("write_inventory", "_write_inventory_authorities"),
        ("preflight", "_synthetic_teacher_preflight"),
        ("device_preflight", "_run_device_preflight"),
        ("row", "_execute_full_row"),
        ("population_seal", "seal_populations"),
    ):
        monkeypatch.setattr(runner, attribute, forbidden(name))
    monkeypatch.setattr(runner, "_finalize_failure", lambda *args, **kwargs: {"synthetic": 1})

    assert runner.run_production(_production_args(run_dir)) == 4
    assert calls == ["scoring"]


@pytest.mark.parametrize(
    ("terminal_state", "terminal_kind"),
    [
        ("awaiting_human_review", "machine_terminalization"),
        ("complete", "human_review_terminalization"),
    ],
)
def test_run_production_commits_missing_terminal_stage_once_after_closed_quantum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_state: str,
    terminal_kind: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    terminal_index = runner.STAGE_ORDER.index(terminal_kind)
    initial_stages = list(runner.STAGE_ORDER[:terminal_index])
    runner._write_json(
        run_dir / "stage_ledger.json",
        {
            "schema": runner.VERSION + "-stage-ledger",
            "events": [
                {"stage": stage, "state": "completed", "recorded_at": f"t{index}"}
                for index, stage in enumerate(initial_stages)
            ],
        },
    )
    route = "awaiting_human_review" if terminal_state == "awaiting_human_review" else "factor_one_feasible"
    runner._write_json(
        run_dir / "status.json",
        {"state": terminal_state, "route": route, "error": None},
    )
    if terminal_state == "complete":
        runner._write_json(run_dir / "outcome.json", {"route": route})

    events = [
        {"event": "admit", "kind": terminal_kind},
        {"event": "complete", "kind": terminal_kind},
    ]
    runner._write_json(
        run_dir / "resource_ledger.json",
        {"events": events, "open_events": [], "failed_admission": None},
    )

    def forbidden_resource(*args: object, **kwargs: object) -> None:
        raise AssertionError("a closed terminal quantum must not be charged twice")

    governor = SimpleNamespace(
        events=events,
        admit=forbidden_resource,
        complete=forbidden_resource,
    )
    base = list(_synthetic_reentry_base())
    base[0] = governor
    verified: list[str] = []

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner, "_classify_reentry", lambda *args: "continue_sealed")
    monkeypatch.setattr(runner, "_validate_reentry_base", lambda *args, **kwargs: tuple(base))
    monkeypatch.setattr(runner, "_record_restart_authority", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_reset_running_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_population_seal", lambda *args: {"sealed": 1})
    monkeypatch.setattr(runner, "_verify_one_row_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_telemetry_summary", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_verify_evaluation",
        lambda *args: {"teacher_control": {}},
    )
    monkeypatch.setattr(runner, "_verify_review_bundle", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "_machine_gates",
        lambda *args, **kwargs: {"gate_d": {"passed": 1}},
    )
    monkeypatch.setattr(
        runner,
        "_verify_machine_gates",
        lambda *args, **kwargs: {"gate_d": {"passed": 1}},
    )
    monkeypatch.setattr(runner, "_replay_human_review", lambda *args: {})
    monkeypatch.setattr(runner, "_verify_complete_outcome", lambda *args: None)
    monkeypatch.setattr(runner, "_write_reports", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_seal_manifest",
        lambda *args: {"artifact_count": 1, "tree_digest": "synthetic-tree"},
    )

    def verify_terminal(path: Path) -> dict[str, object]:
        stages = [event["stage"] for event in runner._stage_events(Path(path))]
        assert stages == initial_stages + [terminal_kind]
        assert stages.count(terminal_kind) == 1
        assert runner._read_json(Path(path) / "status.json")["state"] == terminal_state
        verified.append(terminal_state)
        return {
            "passed": 1,
            "state": terminal_state,
            "route": route,
            "tree_digest": "synthetic-tree",
        }

    monkeypatch.setattr(runner, "verify_run", verify_terminal)

    def forbidden_generation(*args: object, **kwargs: object) -> None:
        raise AssertionError("terminal crash recovery reached generation or scoring")

    for attribute in (
        "_synthetic_teacher_preflight",
        "_run_device_preflight",
        "_execute_full_row",
        "evaluate_sealed_populations",
    ):
        monkeypatch.setattr(runner, attribute, forbidden_generation)

    assert runner.run_production(_production_args(run_dir)) == 0
    assert verified == [terminal_state]


def test_run_production_recovers_open_human_quantum_then_recommits_complete_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    human_stage = "human_review_terminalization"
    human_index = runner.STAGE_ORDER.index(human_stage)
    initial_stages = list(runner.STAGE_ORDER[:human_index])
    runner._write_json(
        run_dir / "stage_ledger.json",
        {
            "schema": runner.VERSION + "-stage-ledger",
            "events": [
                {"stage": stage, "state": "completed", "recorded_at": f"t{index}"}
                for index, stage in enumerate(initial_stages)
            ],
        },
    )
    runner._write_json(
        run_dir / "status.json",
        {"state": "complete", "route": "factor_one_feasible", "error": None},
    )
    runner._write_json(run_dir / "outcome.json", {"route": "factor_one_feasible"})
    monkeypatch.setattr(runner, "_storage_bytes", lambda *args: 0)
    interrupted = runner.ResourceGovernor(
        run_dir,
        runner.ResourceBudget(reserve_seconds=0.0),
        device="cpu",
    )
    interrupted.admit(
        human_stage,
        predicted_seconds=10.0,
        predicted_next_bytes=2 * 1024 * 1024,
        reserve_remaining_seconds=0.0,
    )

    base = list(_synthetic_reentry_base())

    def recover_base(*args: object, **kwargs: object) -> tuple[object, ...]:
        base[0] = runner.ResourceGovernor.rehydrate(
            run_dir,
            device="cpu",
            recover_interrupted=True,
        )
        return tuple(base)

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner, "_classify_reentry", lambda *args: "continue_sealed")
    monkeypatch.setattr(runner, "_validate_reentry_base", recover_base)
    monkeypatch.setattr(runner, "_record_restart_authority", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_reset_running_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_population_seal", lambda *args: {"sealed": 1})
    monkeypatch.setattr(runner, "_verify_one_row_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_telemetry_summary", lambda *args: None)
    monkeypatch.setattr(runner, "_verify_evaluation", lambda *args: {"teacher_control": {}})
    monkeypatch.setattr(runner, "_verify_review_bundle", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "_verify_machine_gates",
        lambda *args, **kwargs: {"gate_d": {"passed": 1}},
    )
    monkeypatch.setattr(runner, "_replay_human_review", lambda *args: {})
    monkeypatch.setattr(runner, "_verify_complete_outcome", lambda *args: None)
    monkeypatch.setattr(runner, "_write_reports", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_seal_manifest",
        lambda *args: {"artifact_count": 1, "tree_digest": "synthetic-tree"},
    )

    def verify_recovery(path: Path) -> dict[str, object]:
        ledger = runner._read_json(Path(path) / "resource_ledger.json")
        assert ledger["open_events"] == []
        assert [
            event["event"]
            for event in ledger["events"]
            if event["kind"] == human_stage
        ] == ["admit", "interrupted-close", "admit", "complete"]
        stages = [event["stage"] for event in runner._stage_events(Path(path))]
        assert stages == initial_stages + [human_stage]
        return {
            "passed": 1,
            "state": "complete",
            "route": "factor_one_feasible",
            "tree_digest": "synthetic-tree",
        }

    monkeypatch.setattr(runner, "verify_run", verify_recovery)

    def forbidden_generation(*args: object, **kwargs: object) -> None:
        raise AssertionError("human terminal recovery reached generation or scoring")

    for attribute in (
        "_synthetic_teacher_preflight",
        "_run_device_preflight",
        "_execute_full_row",
        "evaluate_sealed_populations",
        "_resume_existing_production",
    ):
        monkeypatch.setattr(runner, attribute, forbidden_generation)

    assert runner.run_production(_production_args(run_dir)) == 0


def test_human_terminal_retry_admission_stop_is_terminalized_without_open_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    human_stage = "human_review_terminalization"
    human_index = runner.STAGE_ORDER.index(human_stage)
    runner._write_json(
        run_dir / "stage_ledger.json",
        {
            "schema": runner.VERSION + "-stage-ledger",
            "events": [
                {"stage": stage, "state": "completed", "recorded_at": f"t{index}"}
                for index, stage in enumerate(runner.STAGE_ORDER[:human_index])
            ],
        },
    )
    runner._write_json(
        run_dir / "status.json",
        {"state": "complete", "route": "factor_one_feasible", "error": None},
    )
    runner._write_json(run_dir / "outcome.json", {"route": "factor_one_feasible"})
    runner._write_json(
        run_dir / "gates.json",
        {"gate_e": {"state": "complete", "passed": 1, "conditions": {"synthetic": 1}}},
    )
    monkeypatch.setattr(runner, "_storage_bytes", lambda *args: 0)
    interrupted = runner.ResourceGovernor(
        run_dir,
        runner.ResourceBudget(
            max_storage_bytes=10 * 1024 * 1024,
            reserve_seconds=0.0,
        ),
        device="cpu",
    )
    interrupted.admit(
        human_stage,
        predicted_seconds=10.0,
        predicted_next_bytes=2 * 1024 * 1024,
        reserve_remaining_seconds=0.0,
    )
    # The retried 2 MiB human quantum no longer fits, while the 1 MiB failure
    # terminalization reserve still does.
    monkeypatch.setattr(runner, "_storage_bytes", lambda *args: 9 * 1024 * 1024)
    base = list(_synthetic_reentry_base())

    def recover_base(*args: object, **kwargs: object) -> tuple[object, ...]:
        base[0] = runner.ResourceGovernor.rehydrate(
            run_dir,
            device="cpu",
            recover_interrupted=True,
        )
        return tuple(base)

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner, "_classify_reentry", lambda *args: "continue_sealed")
    monkeypatch.setattr(runner, "_validate_reentry_base", recover_base)
    monkeypatch.setattr(runner, "_record_restart_authority", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_reset_running_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_population_seal", lambda *args: {"sealed": 1})
    monkeypatch.setattr(runner, "_verify_one_row_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_verify_telemetry_summary", lambda *args: None)
    monkeypatch.setattr(runner, "_verify_evaluation", lambda *args: {"teacher_control": {}})
    monkeypatch.setattr(runner, "_verify_review_bundle", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "_verify_machine_gates",
        lambda *args, **kwargs: {"gate_d": {"passed": 1}},
    )
    monkeypatch.setattr(runner, "_replay_human_review", lambda *args: {})
    monkeypatch.setattr(runner, "_verify_complete_outcome", lambda *args: None)
    monkeypatch.setattr(runner, "_write_reports", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_seal_manifest",
        lambda *args: {"artifact_count": 1, "tree_digest": "synthetic-tree"},
    )

    def verify_failure(path: Path) -> dict[str, object]:
        root = Path(path)
        status = runner._read_json(root / "status.json")
        ledger = runner._read_json(root / "resource_ledger.json")
        assert status["state"] == status["route"] == "resource_stopped"
        assert ledger["open_events"] == []
        assert ledger["failed_admission"]["kind"] == human_stage
        assert [
            event["event"]
            for event in ledger["events"]
            if event["kind"] == human_stage
        ] == ["admit", "interrupted-close"]
        assert [
            event["event"]
            for event in ledger["events"]
            if event["kind"] == "failure_terminalization"
        ] == ["admit", "complete"]
        return {
            "passed": 1,
            "state": "resource_stopped",
            "route": "resource_stopped",
            "tree_digest": "synthetic-tree",
        }

    monkeypatch.setattr(runner, "verify_run", verify_failure)
    assert runner.run_production(_production_args(run_dir)) == 3
    assert not (run_dir / "outcome.json").exists()
    assert runner._read_json(run_dir / "gates.json")["gate_e"] == {
        "gate_type": "diagnostic threshold",
        "state": "pending",
        "passed": None,
        "conditions": {},
    }
    failure = runner._read_json(run_dir / "failure.json")
    assert failure["failed_stage"] == human_stage
    assert failure["error_type"] == "ResourceStop"
    assert failure["original_failed_admission"]["kind"] == human_stage


def test_cpu_smoke_cli_is_bounded_deterministic_and_has_no_production_authority(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*args: object, **kwargs: object) -> None:
            forbidden_calls.append(name)
            raise AssertionError(f"CPU smoke accessed forbidden production authority: {name}")

        return fail

    for attribute in (
        "run_production",
        "safe_extract_legacy_checkpoint",
        "read_mnist_development_prefix",
        "read_mnist_arff_slice",
        "_bind_external_authorities",
        "_repository_root",
        "_atomic_bytes",
        "_write_json",
        "_write_npz",
        "_write_csv",
    ):
        monkeypatch.setattr(runner, attribute, forbidden(attribute))
    for attribute in (
        "is_available",
        "device_count",
        "memory_allocated",
        "max_memory_allocated",
        "get_device_properties",
        "synchronize",
    ):
        monkeypatch.setattr(runner.torch.cuda, attribute, forbidden("torch.cuda." + attribute))

    parser = runner.build_parser()
    parsed = parser.parse_args(["smoke"])
    assert vars(parsed) == {"command": "smoke"}
    assert runner.SMOKE_SEED == 0xE14FF001

    assert runner.main(["smoke"]) == 0
    first_text = capsys.readouterr().out
    assert runner.main(["smoke"]) == 0
    second_text = capsys.readouterr().out
    first = json.loads(first_text)
    second = json.loads(second_text)

    assert first_text == second_text
    assert first == second
    assert first == {
        **first,
        "schema": runner.VERSION + "-cpu-smoke",
        "passed": 1,
        "test_only": 1,
        "production_launched": 0,
        "persisted_artifact_count": 0,
        "device": "cpu",
        "seed": 0xE14FF001,
        "grid_size": 8,
        "path_count": 2,
        "outer_steps": 4,
        "schedule_steps": 4,
        "null_noop_exact": 1,
        "teacher_replay_exact": 1,
        "teacher_endpoint_changed": 1,
        "target_firewall_exact": 1,
        "anchors_complete": 1,
        "telemetry_complete": 1,
    }
    assert len(first["scientific_digest"]) == 64
    int(first["scientific_digest"], 16)
    assert "run_dir" not in first
    assert all(
        token not in key.lower()
        for key in first
        for token in ("elapsed", "timing", "cuda", "memory", "checkpoint", "arff")
    )
    assert forbidden_calls == []


def test_cli_exposes_only_execution_review_read_only_verify_and_cpu_smoke() -> None:
    parser = runner.build_parser()
    choices = next(action.choices for action in parser._actions if getattr(action, "choices", None))
    assert set(choices) == {"run", "record-review", "verify", "smoke"}
    tree = ast.parse(inspect.getsource(runner.run_production))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any(name.startswith("train_") for name in called_names)
    assert "backward" not in called_names and "select_generation_result_by_classifier" not in called_names
    assert "next experiment" not in inspect.getsource(runner.run_production).lower()
