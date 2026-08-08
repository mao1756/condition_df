from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_haar_gate import ANTITHETIC_HAAR_PROFILE
from mnist.d0_jacobi_rb_haar_power import FORBIDDEN_COUNTS
from mnist.d0_jacobi_rb_haar_power_recovery_gate import (
    decide_recovery_workflow,
)
import mnist.diag_d0_jacobi_rb_haar_power_recovery_confirmation as cli


def _candidate(*, width: float) -> dict[str, Any]:
    return {
        "main_paths": 16,
        "reference_paths": 16,
        "predicted_main_half_width": width,
        "predicted_generator_reference_half_width": width,
        "predicted_reference_stability_half_width": width,
        "projected_hours": 20.0,
        "conservative_rate": 2_000.0,
        "independent_pool_covariance": 0.0,
        "panel_complete_pass": 1,
        "panel_finite_pass": 1,
        "panel_certification_pass": 1,
        "panel_numerical_health_pass": 1,
        "mass_conservation_pass": 1,
        "shard_chain_pass": 1,
        "pilot_production_isolation_pass": 1,
        "pilot_means_excluded_pass": 1,
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
    }


def _execution() -> dict[str, Any]:
    return {
        "transition_count": 1_000,
        "certified_count": 1_000,
        "certificate_fraction": 1.0,
        "fallback_count": 0,
        "fallback_fraction": 0.0,
        "fallback_elapsed_seconds": 0.0,
        "fallback_cost_fraction": 0.0,
        "elapsed_seconds": 0.5,
        "conservative_rate": 2_000.0,
        "peak_memory_fraction": 0.01,
        "mass_error": 1.0e-15,
        "shard_chain_pass": 1,
        "state_updates_device_resident_pass": 1,
        **{name: 0 for name in FORBIDDEN_COUNTS},
    }


def _panel(role: str, *, width: float) -> dict[str, Any]:
    return {
        "evaluation_status": "evaluated",
        "profile": ANTITHETIC_HAAR_PROFILE,
        "panel": role,
        "root_seed": cli.ROOT_SEED,
        "path_id_plan_sha256": "path-plan",
        "source_npz_sha256": "source",
        "cluster_count": 8,
        "path_id_pools": {f"pool-{role}": list(range(8))},
        "execution": _execution(),
        "candidates": [_candidate(width=width)],
        "complete": 1,
        "finite": 1,
        "production_authorizing_pass": 1,
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
        "independent_pool_variance_pass": 1,
        "richardson_formula_pass": 1,
        "pilot_production_isolation_pass": 1,
    }


def _confirmation(*, width: float) -> dict[str, Any]:
    return {
        "evaluation_status": "evaluated",
        "profile": ANTITHETIC_HAAR_PROFILE,
        "main_paths": 16,
        "reference_paths": 16,
        "complete_pass": 1,
        "finite_pass": 1,
        "certification_pass": 1,
        "numerical_health_pass": 1,
        "mass_conservation_pass": 1,
        "shard_chain_pass": 1,
        "production_authorizing_pass": 1,
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
        "independent_pool_variance_pass": 1,
        "richardson_formula_pass": 1,
        "pilot_production_isolation_pass": 1,
        "main_half_width": width,
        "generator_reference_half_width": width,
        "reference_stability_half_width": width,
        "projected_hours": 20.0,
        "minimum_rate": 2_000.0,
        "certificate_fraction": 1.0,
        "fallback_fraction": 0.0,
        "fallback_cost_fraction": 0.0,
        "peak_memory_fraction": 0.01,
        "mass_error": 1.0e-15,
        **{name: 0 for name in FORBIDDEN_COUNTS},
    }


def _sealed_registry() -> dict[str, Any]:
    return {
        "panels_frozen_before_device_execution": 1,
        "panel_regeneration_permitted": 0,
        "profile_order": [
            "nested_haar_single_arm",
            "pairwise_haar_antithetic",
        ],
    }


def _args() -> Any:
    return type("Args", (), {"device": "cuda"})()


def test_panel_a_no_nominee_never_opens_panel_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def run_panel(**kwargs: Any) -> dict[str, Any]:
        calls.append(str(kwargs["panel"]))
        return _panel(str(kwargs["panel"]), width=0.01)

    monkeypatch.setattr(cli, "run_recovery_antithetic_panel", run_panel)
    gate = cli._run_pilot_stage(
        tmp_path,
        _args(),
        parent_dir=tmp_path / "parent",
        sealed_registry=_sealed_registry(),
    )
    assert calls == ["a"]
    assert gate["passed"] == 0
    assert gate["numerically_valid"] == 1
    assert gate["resource_valid"] == 1
    assert gate["panel_a_nominated"] == 0
    selection = json.loads(
        (tmp_path / "sealed_antithetic_selection.json").read_text()
    )
    assert selection["selection_status"] == (
        "antithetic_panel_a_no_eligible_design"
    )
    decision = decide_recovery_workflow(
        provenance=True,
        preflight_gate={"evaluation_status": "evaluated", "passed": 1},
        replay_gate={"evaluation_status": "evaluated", "passed": 1},
        pilot_gate=gate,
    )
    assert decision["decision"] == "hierarchical_power_infeasible"


def test_nominee_opens_b_once_and_freezes_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def run_panel(**kwargs: Any) -> dict[str, Any]:
        role = str(kwargs["panel"])
        calls.append(role)
        return _panel(role, width=0.001)

    monkeypatch.setattr(cli, "run_recovery_antithetic_panel", run_panel)
    monkeypatch.setattr(
        cli,
        "panel_confirmation_record",
        lambda panel, selected: _confirmation(width=0.001),
    )
    monkeypatch.setattr(
        cli,
        "combine_certified_haar_power_panels",
        lambda **kwargs: _confirmation(width=0.001),
    )
    gate = cli._run_pilot_stage(
        tmp_path,
        _args(),
        parent_dir=tmp_path / "parent",
        sealed_registry=_sealed_registry(),
    )
    assert calls == ["a", "b"]
    assert gate["passed"] == 1
    assert gate["panels_agree"] == 1

    calls.clear()
    assert (
        cli._run_pilot_stage(
            tmp_path,
            _args(),
            parent_dir=tmp_path / "parent",
            sealed_registry=_sealed_registry(),
        )
        == gate
    )
    assert calls == []


def test_confirmation_rejects_any_forbidden_event() -> None:
    record = _confirmation(width=0.001)
    assert cli._confirmation_pass(record)
    record["projection_count"] = 1
    assert not cli._confirmation_pass(record)


def test_old_haar_run_is_not_a_recovery_resume() -> None:
    parent = Path(
        "runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation/"
        "20260725-212650_production-certified-haar-strang-power-adapter-fix-v2"
    )
    if not parent.is_dir():
        pytest.skip("immutable production fixture is unavailable")
    with pytest.raises(ArtifactCompatibilityError):
        cli._verify_terminal_registry(parent)


def test_no_forbidden_pipeline_imports() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8").lower()
    assert "trainer" not in source
    assert "reverse_sampler" not in source
