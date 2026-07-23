from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import mnist.d0_one_image_gate as gate_helpers
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    LegacyArtifactError,
    OneImageGateThresholds,
    atomic_write_csv,
    atomic_write_json,
    compute_validation_metrics,
    deterministic_path_split,
    evaluate_cache_gate,
    evaluate_optimization_gate,
    evaluate_overfit_gates,
    evaluate_reconstruction_gate,
    freeze_training_target_scale,
    infer_training_target_scale,
    load_cache_bundle,
    load_training_checkpoint,
    restore_training_checkpoint,
    save_cache_bundle,
    save_training_checkpoint,
    select_best_ema_checkpoint,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig
from mnist.experiment12_d0 import D0TrainingCache, Experiment12D0Config


def _cache(*, paths: int = 4, slices_per_path: int = 2) -> D0TrainingCache:
    n = 8
    size = paths * slices_per_path
    base = torch.arange(1, n * n + 1, dtype=torch.float32)
    base = base / base.sum()
    states = torch.stack([torch.roll(base, index % (n * n)) for index in range(size)])
    earlier = states.clone()
    starts = torch.arange(size, dtype=torch.long) % 2
    physical = torch.zeros((size, 2, n, n), dtype=torch.float32)
    physical[:, 0, 0, 0] = torch.linspace(0.01, 0.08, size)
    start_images = base.repeat(size, 1)
    terminal = np.stack(
        [np.roll(base.numpy()[::-1].copy(), index).reshape(n, n) for index in range(paths)]
    )
    return D0TrainingCache(
        states=states,
        tau=torch.linspace(0.0, 1.0, size),
        labels=torch.full((size,), 3, dtype=torch.long),
        innovations=torch.zeros((size, 2, n, n), dtype=torch.float32),
        masks=torch.ones((size, 2, n, n), dtype=torch.bool),
        starts=starts,
        path_indices=torch.arange(paths, dtype=torch.long).repeat_interleave(slices_per_path),
        start_images=start_images,
        earlier_states=earlier,
        physical_transfers=physical,
        physical_target_scale=1.0,
        terminal_states=terminal,
        source_indices=np.zeros(paths, dtype=np.int64),
        requested_labels=np.full(paths, 3, dtype=np.int64),
        rate_schedule=np.asarray([0.1, 0.2], dtype=np.float64),
        horizon=1.0,
        dt_sub=0.5,
        stride_substeps=1,
        sample_steps=2,
        reference_substeps=1,
        lambda_mix=0.35,
        raw_limited_fraction=0.001,
        mobility_weighted_limited_fraction=0.0001,
        noise_energy_weighted_limited_fraction=0.0001,
        valid_innovation_fraction=1.0,
        valid_innovation_mobility_fraction=1.0,
        valid_innovation_noise_energy_fraction=1.0,
        floor_correction_l1=0.0,
        renorm_correction_l1=0.0,
        teacher_mode="d0-forward",
        cache_build_mode="substep",
        requested_stride_substeps=1,
        floor_touched_pixels=0,
        floor_proposed_pixels=paths * 2 * slices_per_path,
        floor_touched_fraction=0.0,
    )


def _configs() -> tuple[DirectFluxMNISTConfig, Experiment12D0Config]:
    dynamics = DirectFluxMNISTConfig(
        grid_size=8,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        num_steps=2,
        condition_on_source=False,
        mass_floor=1e-7,
        limiter_fraction=1.0,
    )
    d0 = Experiment12D0Config(
        cache_build_mode="substep",
        d0_target_space="doob-physical-residual",
        teacher_stride_substeps=1,
        physical_target_scale=0.0,
        physical_target_scale_floor=1e-8,
        physical_target_normalization="global-rms",
        physical_loss_mask="all",
        sample_steps=2,
        reference_substeps=1,
    )
    return dynamics, d0


def test_whole_path_split_is_deterministic_and_has_no_slice_leakage() -> None:
    cache = _cache(paths=6, slices_per_path=3)
    first = deterministic_path_split(cache, validation_paths=2, seed=17)
    second = deterministic_path_split(cache, validation_paths=2, seed=17)
    assert first.fingerprint == second.fingerprint
    assert np.array_equal(first.validation_path_ids, second.validation_path_ids)
    assert np.intersect1d(first.train_slice_indices, first.validation_slice_indices).size == 0
    train_paths = np.unique(cache.path_indices[first.train_slice_indices].numpy())
    validation_paths = np.unique(cache.path_indices[first.validation_slice_indices].numpy())
    assert np.array_equal(train_paths, first.train_path_ids)
    assert np.array_equal(validation_paths, first.validation_path_ids)


def test_target_scale_uses_training_slices_only() -> None:
    cache = _cache(paths=4, slices_per_path=2)
    dynamics, d0 = _configs()
    split = deterministic_path_split(cache, validation_paths=1, seed=3)
    initial = infer_training_target_scale(cache, dynamics, d0, split.train_slice_indices)
    cache.physical_transfers[split.validation_slice_indices] *= 1_000_000.0
    changed = infer_training_target_scale(cache, dynamics, d0, split.train_slice_indices)
    assert changed == pytest.approx(initial)
    frozen = freeze_training_target_scale(cache, dynamics, d0, split)
    assert frozen.physical_target_scale == pytest.approx(initial)
    assert cache.physical_target_scale == pytest.approx(initial)


def test_cache_bundle_round_trip_is_complete_and_fingerprinted(tmp_path: Path) -> None:
    cache = _cache()
    cache.floor_touched_fraction = float("nan")
    cache.trajectory_window_states = torch.stack(
        [cache.states.clone(), cache.earlier_states.clone()], dim=1
    )
    cache.trajectory_window_valid = torch.ones((cache.size, 2), dtype=torch.bool)
    cache.trajectory_window_depths = torch.tensor([1, 2], dtype=torch.long)
    path = tmp_path / "cache.npz"
    saved = save_cache_bundle(path, cache, fingerprints={"kernel": "abc"})
    loaded = load_cache_bundle(path, expected_fingerprints={"kernel": "abc"})
    assert loaded.cache_fingerprint == saved.cache_fingerprint
    assert torch.equal(loaded.cache.states, cache.states)
    assert torch.equal(loaded.cache.path_indices, cache.path_indices)
    assert torch.equal(loaded.cache.trajectory_window_states, cache.trajectory_window_states)
    assert torch.equal(loaded.cache.trajectory_window_valid, cache.trajectory_window_valid)
    assert torch.equal(loaded.cache.trajectory_window_depths, cache.trajectory_window_depths)
    assert np.array_equal(loaded.cache.terminal_states, cache.terminal_states)
    assert math.isnan(loaded.cache.floor_touched_fraction)
    with pytest.raises(ArtifactCompatibilityError, match="fingerprint mismatch"):
        load_cache_bundle(path, expected_fingerprints={"kernel": "changed"})


def test_cache_loader_rejects_legacy_npz(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(path, states=np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(LegacyArtifactError, match="report-only"):
        load_cache_bundle(path)


class _ZeroFlux(nn.Module):
    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.grid_size = int(grid_size)

    def forward(
        self,
        tau: torch.Tensor,
        states: torch.Tensor,
        labels: torch.Tensor,
        source: torch.Tensor | None,
    ) -> torch.Tensor:
        del tau, labels, source
        return self.bias * torch.ones(
            (states.shape[0], 2, self.grid_size, self.grid_size),
            dtype=states.dtype,
            device=states.device,
        )


def test_validation_metrics_have_fixed_bins_and_zero_model_gain_zero() -> None:
    cache = _cache(paths=5, slices_per_path=2)
    dynamics, d0 = _configs()
    d0 = replace(d0, physical_target_scale=1.0)
    overall, bins = compute_validation_metrics(
        _ZeroFlux(8),
        cache,
        np.arange(cache.size),
        dynamics,
        d0,
        device="cpu",
        batch_size=3,
        weights="raw",
    )
    assert len(bins) == 5
    assert sum(int(row["slice_count"]) for row in bins) == cache.size
    assert overall["prediction_gain"] == pytest.approx(0.0, abs=1e-7)
    assert overall["target_finite_fraction"] == 1.0
    assert math.isfinite(overall["residual_covariance_trace"])


def test_ema_selection_uses_lowest_finite_mse_then_earliest_step() -> None:
    rows = [
        {"step": 1000, "primary_mse": 0.2},
        {"step": 500, "primary_mse": 0.1},
        {"step": 100, "primary_mse": float("nan")},
        {"step": 1500, "primary_mse": 0.1},
    ]
    selected = select_best_ema_checkpoint(rows)
    assert selected is not None
    assert selected["step"] == 500
    assert select_best_ema_checkpoint([{"step": 1, "primary_mse": float("inf")}]) is None
    assert select_best_ema_checkpoint([{"step": 1, "primary_mse": -1.0}]) is None


def _optimizer_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(3, 2)
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()


def test_training_checkpoint_restores_model_optimizer_and_rng_exactly(tmp_path: Path) -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    generator = np.random.default_rng(11)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _optimizer_step(model, optimizer)
    ema = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    path = tmp_path / "step-1.pt"
    saved_payload = save_training_checkpoint(
        path,
        model=model,
        ema_state=ema,
        optimizer=optimizer,
        scaler=None,
        step=1,
        history=[{"step": 1, "loss": 2.0}],
        best_validation={"step": 0, "primary_mse": 1.0},
        fingerprints={"cache": "one", "runtime": "two"},
        numpy_rng=generator,
    )
    immutable_weight = next(iter(saved_payload["model_state_dict"].values())).clone()
    with torch.no_grad():
        next(iter(model.parameters())).add_(7.0)
    assert torch.equal(next(iter(saved_payload["model_state_dict"].values())), immutable_weight)
    model.load_state_dict(saved_payload["model_state_dict"])
    expected_python = random.random()
    expected_numpy_global = float(np.random.random())
    expected_numpy_generator = float(generator.random())
    expected_torch = torch.rand(4)

    restored_model = nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-2)
    restored_generator = np.random.default_rng(999)
    with pytest.raises(ArtifactCompatibilityError, match="fingerprint mismatch"):
        load_training_checkpoint(path, expected_fingerprints={"cache": "changed"})
    with pytest.raises(ArtifactCompatibilityError, match="extra=runtime"):
        load_training_checkpoint(path, expected_fingerprints={"cache": "one"})
    load_training_checkpoint(
        path,
        expected_fingerprints={"cache": "one"},
        exact_fingerprints=False,
    )
    payload = load_training_checkpoint(
        path, expected_fingerprints={"cache": "one", "runtime": "two"}
    )
    result = restore_training_checkpoint(
        payload,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=None,
        numpy_rng=restored_generator,
    )
    assert result["step"] == 1
    assert result["history"] == [{"step": 1, "loss": 2.0}]
    assert all(
        torch.equal(result["ema_state"][name], value)
        for name, value in ema.items()
    )
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), restored_model.state_dict().values()))
    original_optimizer_state = optimizer.state_dict()["state"]
    restored_optimizer_state = restored_optimizer.state_dict()["state"]
    assert original_optimizer_state.keys() == restored_optimizer_state.keys()
    for parameter_id in original_optimizer_state:
        for key, value in original_optimizer_state[parameter_id].items():
            restored_value = restored_optimizer_state[parameter_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, restored_value)
            else:
                assert value == restored_value
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy_global
    assert float(restored_generator.random()) == expected_numpy_generator
    assert torch.equal(torch.rand(4), expected_torch)


def test_legacy_checkpoint_is_report_only(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(LegacyArtifactError):
        load_training_checkpoint(path)
    with pytest.warns(RuntimeWarning, match="report-only"):
        payload = load_training_checkpoint(path, allow_legacy_report_only=True)
    assert payload["_legacy_report_only"] is True


def _passing_cache_metrics() -> dict[str, float | int | str]:
    return {
        "cache_paths": 8,
        "cache_build_mode": "substep",
        "cache_stride_substeps": 1,
        "physical_target_scale": 0.5,
        "target_finite_fraction": 1.0,
        "oracle_direct_l1": 0.01,
        "oracle_positive_free_only_l1": 0.02,
        "terminal_target_abs_corr_mean": 0.10,
        "nonfinite_edges": 0,
        "floor_touched_pixels": 0,
        "max_simplex_mass_error": 2e-6,
        "floor_correction_l1_per_path_substep": 1e-8,
        "renorm_correction_l1_per_path_substep": 1e-6,
        "raw_limited_fraction": 0.005,
        "mobility_weighted_limited_fraction": 0.0005,
        "noise_energy_weighted_limited_fraction": 0.0005,
    }


def _healthy_arm() -> dict[str, float | int]:
    return {
        "nonfinite_edges": 0,
        "floor_touched_pixels": 0,
        "max_simplex_mass_error": 2e-6,
        "floor_correction_l1_per_path_substep": 1e-8,
        "renorm_correction_l1_per_path_substep": 1e-6,
        "raw_limited_fraction": 0.005,
        "mobility_weighted_limited_fraction": 0.0005,
        "noise_energy_weighted_limited_fraction": 0.0005,
    }


def _passing_reconstruction_metrics() -> dict[str, object]:
    return {
        "complete": 1,
        "sample_count": 16,
        "strength_1_mean_corr": 0.90,
        "strength_1_mean_l1": 0.20,
        "strength_1_good_corr_fraction": 0.80,
        "paired_mean_corr_improvement": 0.20,
        "relative_l1_reduction": 0.25,
        "strength_0": _healthy_arm(),
        "strength_1": _healthy_arm(),
    }


def test_gate_boundaries_pass_and_nonfinite_values_fail_closed() -> None:
    assert evaluate_cache_gate(_passing_cache_metrics())["passed"] == 1
    assert evaluate_optimization_gate(
        {"selected_ema_validation_gain": 1e-12, "selected_ema_data_end_gain": 1e-12, "selected_ema_data_end_count": 1}
    )["passed"] == 1
    assert evaluate_reconstruction_gate(_passing_reconstruction_metrics())["passed"] == 1
    failed = _passing_reconstruction_metrics()
    failed["strength_1_mean_corr"] = float("nan")
    assert evaluate_reconstruction_gate(failed)["passed"] == 0
    failed_count = _passing_reconstruction_metrics()
    failed_count["sample_count"] = 15
    assert evaluate_reconstruction_gate(failed_count)["passed"] == 0


def test_gates_reject_incomplete_or_impossible_negative_metrics() -> None:
    incomplete = _passing_reconstruction_metrics()
    incomplete["complete"] = 0
    assert evaluate_reconstruction_gate(incomplete)["passed"] == 0

    negative_cache = _passing_cache_metrics()
    negative_cache["max_simplex_mass_error"] = -1e-9
    assert evaluate_cache_gate(negative_cache)["passed"] == 0
    negative_oracle = _passing_cache_metrics()
    negative_oracle["oracle_direct_l1"] = -1.0
    assert evaluate_cache_gate(negative_oracle)["passed"] == 0

    negative_arm = copy.deepcopy(_passing_reconstruction_metrics())
    negative_arm["strength_0"]["raw_limited_fraction"] = -1e-9  # type: ignore[index]
    assert evaluate_reconstruction_gate(negative_arm)["passed"] == 0
    assert evaluate_optimization_gate(
        {
            "selected_ema_validation_gain": 1.01,
            "selected_ema_data_end_gain": 0.1,
            "selected_ema_data_end_count": 1,
        }
    )["passed"] == 0


def test_cumulative_required_gate_needs_all_prior_stages() -> None:
    result = evaluate_overfit_gates(
        cache_metrics=_passing_cache_metrics(),
        optimization_metrics={
            "selected_ema_validation_gain": 0.1,
            "selected_ema_data_end_gain": 0.1,
            "selected_ema_data_end_count": 2,
        },
        reconstruction_metrics=_passing_reconstruction_metrics(),
        require_gate="reconstruction",
    )
    assert result["required_gate_pass"] == 1
    broken = _passing_cache_metrics()
    broken["cache_stride_substeps"] = 2
    failed = evaluate_overfit_gates(
        cache_metrics=broken,
        optimization_metrics={
            "selected_ema_validation_gain": 0.1,
            "selected_ema_data_end_gain": 0.1,
            "selected_ema_data_end_count": 2,
        },
        reconstruction_metrics=_passing_reconstruction_metrics(),
        require_gate="reconstruction",
    )
    assert failed["reconstruction"]["passed"] == 1
    assert failed["required_gate_pass"] == 0


def test_atomic_json_and_csv_helpers_write_readable_artifacts(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "status.json"
    csv_path = tmp_path / "nested" / "metrics.csv"
    atomic_write_json(json_path, {"pass": True, "value": np.float32(2.0)})
    atomic_write_csv(csv_path, [{"step": 0, "gain": 0.0}, {"step": 1, "gain": 0.5, "extra": "x"}])
    assert json.loads(json_path.read_text(encoding="utf-8"))["pass"] is True
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["0", "1"]
    assert rows[1]["extra"] == "x"


def test_source_fingerprint_reads_sorted_real_paths_and_tracks_content(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    original = gate_helpers.source_fingerprint([second, first])
    assert original == gate_helpers.source_fingerprint([first, second])
    first.write_text("value = 3\n", encoding="utf-8")
    assert gate_helpers.source_fingerprint([first, second]) != original


def test_atomic_json_preserves_previous_artifact_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "status.json"
    atomic_write_json(path, {"old": True})

    def fail_replace(source: object, destination: object) -> None:
        del source, destination
        raise OSError("injected replace failure")

    monkeypatch.setattr(gate_helpers.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_write_json(path, {"old": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}


def test_threshold_mapping_is_fingerprintable_json_data() -> None:
    thresholds = OneImageGateThresholds()
    assert thresholds.reconstruction_sample_count == 16
