from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_coarse_residual import (
    ALL_CONTRAST_NAMES,
    DEFAULT_MAX_T_SEED,
    GLOBAL_SHRINKAGE,
    PRIMARY_CONTRAST_NAMES,
    WITNESS_AVERAGED_TABLE_NOISE,
    WITNESS_BASELINE_ENERGY,
    WITNESS_PANEL_MEAN_NOISE,
    WITNESS_SIGNAL_ENERGY,
    WITNESS_VALUES_SHA256,
    CoarseResidualContractError,
    CoarseResidualPredictor,
    FrozenCoarseBaseline,
    PathContrastTable,
    combined_exact_mse,
    derive_frozen_witness_baseline,
    exact_combined_target_scale,
    load_frozen_coarse_baseline,
    load_frozen_witness_baseline,
    one_sided_studentized_max_t,
    path_loss_contrasts,
    residualized_mse_algebra_error,
    reverse_time_quartile_numpy,
    reverse_time_quartile_tensor,
    save_frozen_coarse_baseline,
    zero_initialize_residual,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    SELECTED_OUTER_STEPS,
    STATE_SIZE,
    JacobiRBPhasePredictor,
    ModelInputs,
    selected_reverse_time,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _panel_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (64, 4, PHASE_COUNT, EDGES_PER_PHASE)
    coordinate = np.linspace(-0.2, 0.2, np.prod(shape[1:]), dtype=np.float64)
    coordinate = coordinate.reshape(shape[1:])
    path_offset = np.linspace(-0.1, 0.1, 64, dtype=np.float64)[:, None, None, None]
    left = np.ascontiguousarray(1.0 + coordinate[None, ...] + path_offset)
    right = np.ascontiguousarray(1.5 + coordinate[None, ...] - path_offset)
    return (
        left,
        right,
        np.arange(0x10000, 0x10040, dtype=np.int64),
        np.arange(0x10100, 0x10140, dtype=np.int64),
    )


def _baseline_fixture() -> FrozenCoarseBaseline:
    left, right, left_paths, right_paths = _panel_fixture()
    return derive_frozen_witness_baseline(
        left,
        right,
        left_paths,
        right_paths,
        left_cell_means_file_sha256="a" * 64,
        right_cell_means_file_sha256="b" * 64,
        witness_registry_sha256="c" * 64,
    )


def _model_inputs(
    steps: list[int], phases: list[int], *, dtype: torch.dtype = torch.float32
) -> ModelInputs:
    batch = len(steps)
    phase = torch.tensor(phases, dtype=torch.long)
    return ModelInputs(
        later_full_state=torch.full(
            (batch, STATE_SIZE), 1.0 / STATE_SIZE, dtype=dtype
        ),
        reverse_time=torch.tensor(
            [selected_reverse_time(step, item) for step, item in zip(steps, phases)],
            dtype=dtype,
        ),
        phase=phase,
        color=torch.tensor([PHASE_MATCHINGS[item] for item in phases], dtype=torch.long),
        duration=torch.tensor(
            [PHASE_DURATIONS[item] for item in phases], dtype=dtype
        ),
        label=torch.full((batch,), 3, dtype=torch.long),
    )


def test_frozen_production_scalar_derivation_is_self_consistent() -> None:
    assert GLOBAL_SHRINKAGE == WITNESS_SIGNAL_ENERGY / (
        WITNESS_SIGNAL_ENERGY + WITNESS_AVERAGED_TABLE_NOISE
    )
    assert WITNESS_AVERAGED_TABLE_NOISE == 0.5 * WITNESS_PANEL_MEAN_NOISE
    assert GLOBAL_SHRINKAGE == pytest.approx(0.2910413880506186, abs=0.0)
    assert WITNESS_BASELINE_ENERGY == pytest.approx(
        0.00018871847424106853, abs=0.0
    )
    assert (
        WITNESS_VALUES_SHA256
        == "5d4e73153c36a59e26403439befd4e13b7f4fe096f7cbf9af6b77ac26565a9df"
    )
    assert DEFAULT_MAX_T_SEED == 261255
    assert PRIMARY_CONTRAST_NAMES == (
        "overall.baseline_vs_zero",
        "overall.combined_vs_baseline",
    )


def test_derive_baseline_uses_cross_panel_signal_and_averaged_noise() -> None:
    left, right, left_paths, right_paths = _panel_fixture()
    baseline = derive_frozen_witness_baseline(
        left,
        right,
        left_paths,
        right_paths,
        left_cell_means_file_sha256="a" * 64,
        right_cell_means_file_sha256="b" * 64,
        witness_registry_sha256="c" * 64,
    )
    left_mean = left.mean(axis=0)
    right_mean = right.mean(axis=0)
    signal = np.mean(left_mean * right_mean)
    panel_noise = 0.5 * np.mean((left_mean - right_mean) ** 2)
    shrinkage = signal / (signal + 0.5 * panel_noise)
    np.testing.assert_array_equal(
        baseline.raw_values, 0.5 * (left_mean + right_mean)
    )
    np.testing.assert_array_equal(
        baseline.values, shrinkage * baseline.raw_values
    )
    assert baseline.signal_energy == pytest.approx(signal, rel=0.0, abs=2e-16)
    assert baseline.panel_mean_noise == pytest.approx(
        panel_noise, rel=0.0, abs=2e-16
    )
    assert baseline.shrinkage == pytest.approx(shrinkage, rel=0.0, abs=2e-16)
    assert baseline.to_record()["target_modified"] == 0
    assert baseline.to_record()["values_sha256"] == hashlib.sha256(
        baseline.values.tobytes(order="C")
    ).hexdigest()
    assert baseline.to_record()["baseline_energy"] == baseline.baseline_energy


@pytest.mark.parametrize("defect", ["shape", "nonfinite", "overlap", "hash"])
def test_baseline_derivation_and_contract_fail_closed(defect: str) -> None:
    left, right, left_paths, right_paths = _panel_fixture()
    left_hash = "a" * 64
    if defect == "shape":
        left = left[:-1]
    elif defect == "nonfinite":
        left = left.copy()
        left[0, 0, 0, 0] = math.nan
    elif defect == "overlap":
        right_paths = left_paths.copy()
    elif defect == "hash":
        left_hash = "not-a-sha"
    with pytest.raises(CoarseResidualContractError):
        derive_frozen_witness_baseline(
            left,
            right,
            left_paths,
            right_paths,
            left_cell_means_file_sha256=left_hash,
            right_cell_means_file_sha256="b" * 64,
            witness_registry_sha256="c" * 64,
        )


def test_baseline_file_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    baseline = _baseline_fixture()
    path = tmp_path / "baseline.npz"
    artifact = save_frozen_coarse_baseline(path, baseline)
    replay = load_frozen_coarse_baseline(path, expected_sha256=artifact["sha256"])
    assert replay.to_record() == baseline.to_record()
    assert replay.fingerprint == artifact["baseline_semantic_sha256"]

    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(CoarseResidualContractError):
        load_frozen_coarse_baseline(path, expected_sha256=artifact["sha256"])


def _write_fake_witness(tmp_path: Path) -> tuple[Path, FrozenCoarseBaseline]:
    run_dir = tmp_path / "witness"
    (run_dir / "panels" / "a").mkdir(parents=True)
    (run_dir / "panels" / "b").mkdir(parents=True)
    left, right, left_paths, right_paths = _panel_fixture()
    records: list[dict[str, object]] = []

    def register(relative: str) -> None:
        path = run_dir / relative
        records.append(
            {"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)}
        )

    gate = {"evaluation_status": "evaluated", "passed": 1}
    (run_dir / "coarse_signal_witness_gate.json").write_text(
        json.dumps(gate), encoding="utf-8"
    )
    register("coarse_signal_witness_gate.json")
    file_hashes: list[str] = []
    for role, values, paths in (
        ("a", left, left_paths),
        ("b", right, right_paths),
    ):
        panel_path = run_dir / "panels" / role / "cell_means.npz"
        np.savez_compressed(panel_path, cell_means=values, path_ids=paths)
        file_hash = _sha256(panel_path)
        file_hashes.append(file_hash)
        register(f"panels/{role}/cell_means.npz")
        seal = {
            "cell_means_file": f"panels/{role}/cell_means.npz",
            "cell_means_file_sha256": file_hash,
            "cell_means_array_sha256": _array_sha256(values),
            "path_ids": paths.tolist(),
            "path_plan_sha256": (
                "76f44f7c83f5f294ebda5d55f610c0942e7f16f1ab2ea2940ae70a5f2b059a65"
            ),
            "statistic_plan_sha256": (
                "91cee2ce9eb5a1688dcfc72ba2a27c02c53a73b88e29807cc556617a10f7343"
            ),
        }
        # Correct the intentionally readable literal above to the production
        # statistic-plan fingerprint before writing.
        seal["statistic_plan_sha256"] = (
            "91cee2ce9eb5a1688dcfc72ba2a27c02c53a73b88e29807cc556aff75c403ec0"
        )
        seal_path = run_dir / f"panel_{role}_seal.json"
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
        register(f"panel_{role}_seal.json")

    registry = {"record_count": len(records), "records": records}
    registry_path = run_dir / "artifact_registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    registry_semantic = "d" * 64
    status = {
        "state": "completed",
        "decision": "exact_physical_coarse_signal_detected",
        "artifact_registry_file_sha256": _sha256(registry_path),
        "artifact_registry_record_count": len(records),
        "artifact_registry_sha256": registry_semantic,
    }
    (run_dir / "run_status.json").write_text(json.dumps(status), encoding="utf-8")
    expected = derive_frozen_witness_baseline(
        left,
        right,
        left_paths,
        right_paths,
        left_cell_means_file_sha256=file_hashes[0],
        right_cell_means_file_sha256=file_hashes[1],
        witness_registry_sha256=registry_semantic,
    )
    return run_dir, expected


def test_witness_loader_checks_registry_seals_and_panel_hashes(tmp_path: Path) -> None:
    run_dir, expected = _write_fake_witness(tmp_path)
    loaded = load_frozen_witness_baseline(
        run_dir, expected_registry_sha256=None
    )
    assert loaded.to_record() == expected.to_record()

    panel = run_dir / "panels" / "a" / "cell_means.npz"
    panel.write_bytes(panel.read_bytes() + b"tamper")
    with pytest.raises(
        CoarseResidualContractError, match="hash/size mismatch"
    ):
        load_frozen_witness_baseline(run_dir, expected_registry_sha256=None)


def test_reverse_time_quartile_recovers_every_selected_phase_without_outer_step() -> None:
    times = np.array(
        [
            selected_reverse_time(step, phase)
            for step in SELECTED_OUTER_STEPS
            for phase in range(PHASE_COUNT)
        ],
        dtype=np.float64,
    )
    phases = np.tile(np.arange(PHASE_COUNT, dtype=np.int64), len(SELECTED_OUTER_STEPS))
    expected = np.repeat(
        np.asarray(SELECTED_OUTER_STEPS, dtype=np.int64) // 128, PHASE_COUNT
    )
    np.testing.assert_array_equal(
        reverse_time_quartile_numpy(times, phases), expected
    )
    torch.testing.assert_close(
        reverse_time_quartile_tensor(
            torch.tensor(times, dtype=torch.float64), torch.tensor(phases)
        ),
        torch.tensor(expected),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        reverse_time_quartile_tensor(
            torch.tensor(times, dtype=torch.float32), torch.tensor(phases)
        ),
        torch.tensor(expected),
        rtol=0,
        atol=0,
    )
    tampered = times.copy()
    tampered[0] += 1.0e-4
    with pytest.raises(CoarseResidualContractError):
        reverse_time_quartile_numpy(tampered, phases)


def test_zero_initialized_predictor_is_exactly_the_baseline() -> None:
    baseline = _baseline_fixture()
    residual = JacobiRBPhasePredictor(width=32)
    conv_before = residual.conv1.weight.detach().clone()
    predictor = CoarseResidualPredictor(baseline, residual)
    inputs = _model_inputs([15, 159, 303, 511], [0, 1, 3, 6])
    output = predictor(inputs)
    assert output.dtype == torch.float64
    expected = torch.tensor(
        baseline.predict_numpy(
            inputs.reverse_time.numpy(), inputs.phase.numpy()
        ),
        dtype=output.dtype,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(residual.conv1.weight, conv_before, rtol=0, atol=0)
    assert torch.count_nonzero(residual.spatial_output.weight) == 0
    assert torch.count_nonzero(residual.local_affine.weight) == 0

    with pytest.raises(CoarseResidualContractError):
        predictor({"later_full_state": inputs.later_full_state})  # type: ignore[arg-type]


def test_zero_initializer_rejects_an_altered_model_class() -> None:
    with pytest.raises(CoarseResidualContractError):
        zero_initialize_residual(torch.nn.Linear(2, 1))


def test_exact_target_scale_and_combined_mse_keep_target_unchanged() -> None:
    target = torch.linspace(
        -2.0, 2.0, 3 * EDGES_PER_PHASE, dtype=torch.float64
    ).reshape(3, EDGES_PER_PHASE)
    baseline = torch.full_like(target, 0.25)
    residual = torch.full_like(target, -0.1)
    combined = baseline + residual
    scale = exact_combined_target_scale(target)
    optimizer_loss, raw = combined_exact_mse(combined, target, scale)
    expected = torch.mean((combined - target).square())
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer_loss, expected / (scale * scale), rtol=0, atol=0
    )
    assert residualized_mse_algebra_error(baseline, residual, target) == 0.0
    with pytest.raises(CoarseResidualContractError):
        exact_combined_target_scale(target.float())


def _contrast_fixture(path_count: int = 12) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rows_target: list[np.ndarray] = []
    rows_baseline: list[np.ndarray] = []
    rows_combined: list[np.ndarray] = []
    paths: list[int] = []
    times: list[float] = []
    phases: list[int] = []
    for path in range(path_count):
        for step, phase in ((15, 0), (511, 6)):
            target_value = 1.0 + 0.01 * path + 0.1 * (step == 511)
            baseline_value = 0.20
            combined_value = 0.35
            rows_target.append(
                np.full(EDGES_PER_PHASE, target_value, dtype=np.float64)
            )
            rows_baseline.append(
                np.full(EDGES_PER_PHASE, baseline_value, dtype=np.float64)
            )
            rows_combined.append(
                np.full(EDGES_PER_PHASE, combined_value, dtype=np.float64)
            )
            paths.append(0x20000 + path)
            times.append(selected_reverse_time(step, phase))
            phases.append(phase)
    return (
        np.stack(rows_target),
        np.stack(rows_baseline),
        np.stack(rows_combined),
        np.asarray(paths, dtype=np.int64),
        np.asarray(times, dtype=np.float64),
        np.asarray(phases, dtype=np.int64),
    )


def test_path_contrasts_are_whole_path_paired_and_algebraically_complete() -> None:
    table = path_loss_contrasts(*_contrast_fixture())
    assert table.path_count == 12
    assert table.names == ALL_CONTRAST_NAMES
    np.testing.assert_allclose(
        table.column("overall.baseline_vs_zero")
        + table.column("overall.combined_vs_baseline"),
        table.column("overall.combined_vs_zero"),
        rtol=0,
        atol=4e-16,
    )
    np.testing.assert_allclose(
        table.column("data_end.baseline_vs_zero")
        + table.column("data_end.combined_vs_baseline"),
        table.column("data_end.combined_vs_zero"),
        rtol=0,
        atol=4e-16,
    )
    assert np.all(table.column("overall.baseline_vs_zero") > 0)
    assert np.all(table.column("overall.combined_vs_baseline") > 0)


def _max_t_table() -> PathContrastTable:
    path_count = 64
    index = np.arange(path_count, dtype=np.float64)
    wave = np.sin(index * 0.37)
    columns = np.empty((path_count, len(ALL_CONTRAST_NAMES)), dtype=np.float64)
    columns[:, 0] = 0.020 + 0.0020 * wave
    columns[:, 1] = 0.015 + 0.0015 * np.cos(index * 0.41)
    columns[:, 2] = columns[:, 0] + columns[:, 1]
    columns[:, 3] = 0.012 + 0.0018 * np.sin(index * 0.29)
    columns[:, 4] = 0.010 + 0.0012 * np.cos(index * 0.31)
    columns[:, 5] = columns[:, 3] + columns[:, 4]
    return PathContrastTable(
        np.arange(0x30000, 0x30000 + path_count, dtype=np.int64), columns
    )


def test_studentized_max_t_is_deterministic_joint_and_chunk_invariant() -> None:
    table = _max_t_table()
    first = one_sided_studentized_max_t(
        table, replicates=2_000, seed=123, namespace=7, chunk_size=73
    )
    replay = one_sided_studentized_max_t(
        table, replicates=2_000, seed=123, namespace=7, chunk_size=512
    )
    assert first.to_record() == replay.to_record()
    assert first.family_names == PRIMARY_CONTRAST_NAMES
    assert first.passed
    assert np.all(first.lower_bounds > 0.0)
    assert first.to_record()["bootstrap_unit"] == "whole_path_jointly_across_family"
    changed = one_sided_studentized_max_t(
        table, replicates=2_000, seed=124, namespace=7
    )
    assert changed.critical_value != first.critical_value


def test_studentized_max_t_fails_closed_on_degenerate_or_invalid_family() -> None:
    table = _max_t_table()
    values = np.array(table.values, copy=True)
    values[:, 0] = 1.0
    degenerate = PathContrastTable(table.path_ids, values)
    with pytest.raises(CoarseResidualContractError, match="degenerate"):
        one_sided_studentized_max_t(degenerate, replicates=100)
    with pytest.raises(CoarseResidualContractError):
        one_sided_studentized_max_t(
            table, family_names=("not-a-contrast",), replicates=100
        )
