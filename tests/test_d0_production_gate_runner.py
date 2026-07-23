"""Production-runner and forecast contracts for the D0 zero-residual gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mnist.diag_d0_zero_residual as gate
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig


def _dynamics(*, limiter_fraction: float = 1.0) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=4,
        num_steps=2,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-7,
        limiter_fraction=limiter_fraction,
    )


def _summary(raw: float, weighted: float) -> dict[str, float | int]:
    return {
        "limiter_fraction": raw,
        "mobility_weighted_limiter_fraction": weighted,
        "noise_energy_weighted_limiter_fraction": weighted,
        "limited_edges": 1,
        "proposed_edges": 10,
        "nonfinite_edges": 0,
        "floor_touched_pixels": 0,
        "floor_proposed_pixels": 64,
        "floor_correction_l1_per_path_substep": 0.0,
        "renorm_correction_l1_per_path_substep": 0.0,
        "max_simplex_mass_error": 0.0,
        "kernel_substeps_executed": 1,
    }


def test_preflight_reuses_inputs_auto_doubles_and_never_sets_science_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamics = _dynamics()
    config = gate.ZeroResidualDiagnosticConfig(
        num_paths=8,
        sample_steps=2,
        substep_levels=(1, 2),
        horizon=0.08,
        tau_eff=0.01,
        calibration_reps=1,
        preflight_only=True,
        preflight_paths=8,
        preflight_reps=2,
        preflight_limiter_fractions=(0.25, 1.0),
        preflight_max_substeps=8,
    )
    direct_inputs: dict[int, list[tuple[np.ndarray, torch.Tensor]]] = {0: [], 1: []}
    forward_inputs: dict[int, list[np.ndarray]] = {0: [], 1: []}
    first_values: dict[int, float] = {}

    def identify_rep(initial: np.ndarray) -> int:
        first = float(initial[0, 0])
        for rep, value in first_values.items():
            if math.isclose(first, value, rel_tol=0.0, abs_tol=0.0):
                return rep
        rep = len(first_values)
        first_values[rep] = first
        return rep

    def fake_direct(initial, normal, *, dynamics_config, rate, dt, device):
        del rate, device
        rep = identify_rep(initial)
        direct_inputs[rep].append((initial, normal))
        level = round(config.horizon / (config.sample_steps * dt))
        passes = dynamics_config.limiter_fraction == 1.0 and level >= 4
        return _summary(0.004 if passes else 0.02, 0.0004 if passes else 0.01)

    def fake_forward(initial, *, dynamics_config, rate, dt, seed, device):
        del rate, seed, device
        rep = identify_rep(initial)
        forward_inputs[rep].append(initial)
        level = round(config.horizon / (config.sample_steps * dt))
        passes = dynamics_config.limiter_fraction == 1.0 and level >= 4
        return _summary(0.004 if passes else 0.02, 0.0004 if passes else 0.01)

    monkeypatch.setattr(gate, "_run_direct_preflight_step", fake_direct)
    monkeypatch.setattr(gate, "_run_forward_preflight_step", fake_forward)
    result = gate.run_zero_residual_preflight(
        dynamics_config=dynamics,
        diagnostic_config=config,
        device=torch.device("cpu"),
        rate_schedule=np.asarray([0.25, 0.5]),
    )

    assert result.levels_evaluated == (1, 2, 4)
    assert result.summary["auto_doubling_used"] == 1
    assert result.summary["first_joint_threshold_substeps"] == 4
    assert result.summary["eligible_limiter_fractions_at_stop"] == [1.0]
    serialized = result.to_dict()
    assert serialized["mode"] == "forecast-only-intervention-preflight"
    assert "gate" not in serialized and "stationarity" not in serialized
    for rows in direct_inputs.values():
        if not rows:
            continue
        assert all(row[0] is rows[0][0] for row in rows)
        assert all(row[1] is rows[0][1] for row in rows)
    for rows in forward_inputs.values():
        if rows:
            assert all(row is rows[0] for row in rows)


def test_full_budget_has_no_more_intervention_on_boundary_stress_fixture() -> None:
    rng = np.random.default_rng(15)
    initial = rng.dirichlet(np.ones(16), size=32)
    normal = torch.Generator().manual_seed(28)
    standard_normal = torch.randn((32, 2, 4, 4), generator=normal)
    summaries: dict[float, dict[str, float | int]] = {}
    for limiter in (0.25, 1.0):
        summaries[limiter] = gate._run_direct_preflight_step(
            initial,
            standard_normal,
            dynamics_config=_dynamics(limiter_fraction=limiter),
            rate=1.0,
            dt=2e-3,
            device=torch.device("cpu"),
        )
    for key in (
        "limiter_fraction",
        "mobility_weighted_limiter_fraction",
        "noise_energy_weighted_limiter_fraction",
    ):
        assert float(summaries[1.0][key]) <= float(summaries[0.25][key])

    forward: dict[float, dict[str, float | int]] = {}
    for limiter in (0.25, 1.0):
        forward[limiter] = gate._run_forward_preflight_step(
            initial,
            dynamics_config=_dynamics(limiter_fraction=limiter),
            rate=1.0,
            dt=2e-3,
            seed=28,
            device=torch.device("cpu"),
        )
    for key in (
        "limiter_fraction",
        "mobility_weighted_limiter_fraction",
        "noise_energy_weighted_limiter_fraction",
    ):
        assert float(forward[1.0][key]) <= float(forward[0.25][key])


def test_preflight_cap_contract_and_nonfinite_metrics_fail_closed() -> None:
    assert gate._preflight_candidate_levels((2, 4), max_substeps=16) == (2, 4, 8, 16)
    with pytest.raises(ValueError, match="power of two"):
        gate._preflight_candidate_levels((2, 4), max_substeps=12)
    row = _summary(float("nan"), 0.0)
    assert not gate._intervention_thresholds_met(
        row,
        raw_threshold=0.005,
        weighted_threshold=0.0005,
    )


def _forward_rows() -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for level, factor in ((2, 1.0), (4, 0.8), (8, 0.6)):
        rows.append(
            {
                "substeps": level,
                "stationarity_quantile_distance": 0.01 * factor,
                "stationarity_quantile_threshold": 0.02,
                "stationarity_quantile_ratio": 0.5 * factor,
                "stationarity_feature_mmd": 5e-4 * factor,
                "stationarity_feature_mmd_threshold": 1e-3,
                "stationarity_feature_mmd_ratio": 0.5 * factor,
                "entropy_standard_error_units": factor,
                "entropy_analytic_standard_error_units": factor,
                "entropy_paired_drift_standard_error_units": factor,
                "second_moment_standard_error_units": factor,
                "second_moment_analytic_standard_error_units": factor,
                "second_moment_paired_drift_standard_error_units": factor,
                "limiter_fraction": 0.004 * factor,
                "mobility_weighted_limiter_fraction": 4e-4 * factor,
                "noise_energy_weighted_limiter_fraction": 4e-4 * factor,
                "nonfinite_edges": 0,
                "max_simplex_mass_error": 1e-7,
                "floor_correction_l1_per_path_substep": 0.0,
                "renorm_correction_l1_per_path_substep": 1e-8,
            }
        )
    return rows


def test_forward_control_gate_is_fail_closed_and_does_not_require_coupled_rms() -> None:
    passed = gate.evaluate_forward_reference_control(_forward_rows())
    assert passed["forward_reference_control_pass"] == 1
    assert passed["forward_control_strict_intervention_thresholds_met"] == 1

    failed_rows = _forward_rows()
    failed_rows[-1]["nonfinite_edges"] = 1
    failed = gate.evaluate_forward_reference_control(failed_rows)
    assert failed["forward_control_pass_numerical_health"] == 0
    assert failed["forward_reference_control_pass"] == 0
    assert gate.evaluate_forward_reference_control([])["forward_reference_control_pass"] == 0


def test_forward_control_requires_strict_intervention_thresholds() -> None:
    rows = _forward_rows()
    for row, raw in zip(rows, (0.010, 0.008, 0.006)):
        row["limiter_fraction"] = raw
    result = gate.evaluate_forward_reference_control(rows)
    assert result["forward_control_pass_nonincreasing_interventions"] == 1
    assert result["forward_control_strict_intervention_thresholds_met"] == 0
    assert result["forward_reference_control_pass"] == 0


def test_forward_control_uses_forward_schedule_order_and_monotone_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rates: list[float] = []
    completed: list[int] = []

    def fake_step(states, _dt, _config, **kwargs):
        rates.append(float(kwargs["free_weight"]))
        return SimpleNamespace(
            states=states,
            device_diagnostics=None,
            limited_edges=0,
            proposed_edges=1,
            mobility_weight_sum=1.0,
            limited_mobility_weight_sum=0.0,
            noise_energy_sum=1.0,
            limited_noise_energy_sum=0.0,
            nonfinite_edges=0,
            floor_touched_pixels=0,
            floor_correction_l1=0.0,
            renorm_correction_l1=0.0,
        )

    monkeypatch.setattr(gate, "masked_reference_free_step_torch", fake_step)
    initial = np.full((2, 16), 1.0 / 16.0, dtype=np.float64)
    gate._run_forward_control_level(
        initial,
        dynamics_config=_dynamics(),
        rate_schedule=np.asarray([1.0, 2.0, 3.0]),
        horizon=0.3,
        substeps=1,
        seed=7,
        device=torch.device("cpu"),
        progress_callback=lambda event: completed.append(int(event["outer_completed"])),
    )
    assert rates == [1.0, 2.0, 3.0]
    assert completed == [1, 2, 3]


def _tiny_cli(root: Path, run_name: str, *extra: str) -> list[str]:
    return [
        "--runs-root",
        str(root),
        "--run-name",
        run_name,
        "--device",
        "cpu",
        "--num-paths",
        "8",
        "--grid-size",
        "4",
        "--sample-steps",
        "1",
        "--substeps",
        "1,2,4",
        "--tau-eff",
        "1e-5",
        "--calibration-reps",
        "1",
        "--progress-every",
        "0",
        *extra,
    ]


def test_resume_skips_complete_seed_and_rejects_scientific_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate.main(_tiny_cli(tmp_path, "resume-smoke"))
    run_dir = next(tmp_path.glob("*_resume-smoke"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["completed_seeds"] == [260715]

    monkeypatch.setattr(
        gate,
        "run_zero_residual_diagnostic",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("completed seed reran")),
    )
    gate.main([*_tiny_cli(tmp_path, "ignored"), "--resume-run-dir", str(run_dir)])
    resumed = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert resumed["attempt_count"] == 2
    aggregate = json.loads((run_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert aggregate["replicates"][0]["resumed"] == 1

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        gate.main(
            [
                *_tiny_cli(tmp_path, "ignored", "--limiter-fraction", "1"),
                "--resume-run-dir",
                str(run_dir),
            ]
        )


def test_resume_recovers_incomplete_seed_by_recomputing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_runner = gate.run_zero_residual_diagnostic

    def interrupted_runner(**_kwargs):
        raise RuntimeError("synthetic incomplete seed")

    monkeypatch.setattr(gate, "run_zero_residual_diagnostic", interrupted_runner)
    with pytest.raises(RuntimeError, match="incomplete seed"):
        gate.main(_tiny_cli(tmp_path, "resume-incomplete"))
    run_dir = next(tmp_path.glob("*_resume-incomplete"))
    failed = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["completed_seeds"] == []

    monkeypatch.setattr(gate, "run_zero_residual_diagnostic", real_runner)
    gate.main([*_tiny_cli(tmp_path, "ignored"), "--resume-run-dir", str(run_dir)])
    recovered = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert recovered["status"] == "complete"
    assert recovered["attempt_count"] == 2
    assert recovered["completed_seeds"] == [260715]
    assert (run_dir / "aggregate_summary.json").exists()


def test_required_gate_failure_is_reported_after_artifacts(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        gate.main(
            _tiny_cli(
                tmp_path,
                "fail-closed",
                "--require-gate",
                "training-ready",
            )
        )
    assert exc.value.code == 1
    run_dir = next(tmp_path.glob("*_fail-closed"))
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "aggregate_summary.json").exists()
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["outcome"] == "gate_failed"
    assert status["required_gate_pass"] == 0


def test_preflight_status_never_claims_a_scientific_gate(tmp_path: Path) -> None:
    gate.main(
        _tiny_cli(
            tmp_path,
            "preflight-status",
            "--preflight-only",
            "--preflight-paths",
            "8",
            "--preflight-reps",
            "1",
            "--preflight-max-substeps",
            "4",
        )
    )
    run_dir = next(tmp_path.glob("*_preflight-status"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["outcome"] == "preflight_complete"
    assert status["scientific_gate_evaluated"] == 0
    assert "required_gate_pass" not in status


def test_atomic_json_keeps_previous_file_if_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "status.json"
    path.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("replacement interrupted")

    monkeypatch.setattr(gate.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        gate._atomic_write_json(path, {"old": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}


def test_forward_seed_resolution_and_fingerprint_are_order_stable() -> None:
    assert gate._resolve_forward_control_seeds(
        [1, 2, 3], include_all=False, explicit="3,1"
    ) == [3, 1]
    assert gate._resolve_forward_control_seeds(
        [1, 2], include_all=True, explicit=""
    ) == [1, 2]
    with pytest.raises(ValueError, match="subset"):
        gate._resolve_forward_control_seeds([1, 2], include_all=False, explicit="3")
    with pytest.raises(ValueError, match="unique"):
        gate._resolve_forward_control_seeds([1, 2], include_all=False, explicit="1,1")

    first = gate._config_fingerprint({"b": [2, 1], "a": {"z": 3}})
    second = gate._config_fingerprint({"a": {"z": 3}, "b": [2, 1]})
    changed = gate._config_fingerprint({"a": {"z": 4}, "b": [2, 1]})
    assert first == second
    assert changed != first


def test_manifest_fingerprints_algorithm_sources_and_runtime() -> None:
    config = gate.ZeroResidualDiagnosticConfig(
        num_paths=8,
        sample_steps=2,
        substep_levels=(1, 2, 4),
        calibration_reps=1,
    )
    manifest = gate._scientific_manifest(
        dynamics_config=_dynamics(),
        diagnostic_config=config,
        seeds=[2, 1],
        forward_control_seeds=[1],
        rate_schedule=np.asarray([0.5, 1.0]),
        mode="diagnostic",
        device=torch.device("cpu"),
    )
    assert manifest["algorithm_version"] == gate._RUN_ALGORITHM_VERSION
    assert set(manifest["implementation_sha256"]) == set(
        gate._IMPLEMENTATION_SOURCE_FILES
    )
    assert manifest["runtime"]["device_type"] == "cpu"
    assert manifest["runtime"]["numpy_version"] == np.__version__

    changed_manifest = {
        **manifest,
        "implementation_sha256": {
            **manifest["implementation_sha256"],
            "diag_d0_zero_residual.py": "changed",
        },
    }
    assert gate._config_fingerprint(changed_manifest) != gate._config_fingerprint(manifest)
