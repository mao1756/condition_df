from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from mnist.d0_score_boundary_controls import (
    bounded_teacher_log_relative_potential,
)
from mnist.d0_score_density_ratio import (
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
)
from mnist.d0_score_density_ratio_matched_flux import (
    MATCHED_FLUX_BOOTSTRAP_VERSION,
    evaluate_matched_teacher_flux_path_energies,
    evaluate_matched_teacher_flux_reduction,
    joint_matched_flux_family_bootstrap,
    joint_whole_path_relative_flux_reduction_bootstrap,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig


torch.set_num_threads(1)


def _config() -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=4,
        num_steps=8,
        source_lowfreq_size=2,
        ot_lowres_size=2,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-12,
        limiter_fraction=1.0,
        condition_on_source=False,
        flux_parameterization="edge",
    )


def _panel(path_count: int = 4):
    plan = build_density_ratio_stream_plan(
        root_seed=261041, grid_size=4, horizon=0.125
    )
    return build_density_ratio_panel(
        plan,
        phase="matched-pilot",
        role="b",
        task="bounded_teacher",
        path_count=path_count,
        dtype=torch.float64,
    )


class _ScaledAnalyticTeacher(nn.Module):
    def __init__(self, horizon: float, scale: float) -> None:
        super().__init__()
        self.horizon = float(horizon)
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float64))

    def forward(
        self, tau: torch.Tensor, states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del labels
        return self.scale * bounded_teacher_log_relative_potential(
            states, tau / self.horizon, epsilon=0.5
        )


def test_matched_evaluation_recomputes_exact_per_path_flux_energies() -> None:
    panel = _panel(path_count=4)
    selected = _ScaledAnalyticTeacher(0.125, 0.5).train()
    baseline = _ScaledAnalyticTeacher(0.125, 0.0).eval()
    report = evaluate_matched_teacher_flux_reduction(
        selected,
        baseline,
        panel,
        _config(),
        seed=91,
        reps=512,
        confidence=0.95,
        batch_size=16,
        selected_role="rho-1",
        baseline_role="rho-0",
        evaluation_role="sealed-b",
    )

    # Scaling the analytic log potential by 1/2 scales its physical flux by
    # 1/2.  Relative error is therefore exactly 1/2 versus the zero model's 1.
    evaluation = report["evaluation"]
    for scope, expected_states in (("overall", 4 * 32), ("data_end", 4 * 16)):
        record = evaluation["scopes"][scope]
        assert record["path_count"] == 4
        assert record["state_count"] == expected_states
        assert len(record["path_energies"]) == 4
        assert record["selected_relative_flux_l2"] == pytest.approx(0.5, abs=2e-12)
        assert record["baseline_relative_flux_l2"] == pytest.approx(1.0, abs=2e-12)
        assert record["point_relative_flux_l2_reduction"] == pytest.approx(
            0.5, abs=2e-12
        )
        bound = report["simultaneous_bootstrap"]["scopes"][scope]
        assert bound["simultaneous_lower_bound"] == pytest.approx(0.5, abs=2e-12)
        assert bound["positive_simultaneous_lower_bound"] == 1

    assert selected.training is True
    assert baseline.training is False
    assert evaluation["panel_fingerprint"] == panel.fingerprint
    assert evaluation["panel_role"] == "b"
    assert evaluation["evaluation_role"] == "sealed-b"
    assert evaluation["selected_role"] == "rho-1"
    assert evaluation["baseline_role"] == "rho-0"
    assert evaluation["selected_model_sha256"] != evaluation["baseline_model_sha256"]
    assert report["physical_training_performed"] == 0
    assert report["sampling_performed"] == 0
    json.dumps(report, sort_keys=True, allow_nan=False)


def _manual_evaluation() -> dict[str, object]:
    path_ids = [11, 17, 23, 29, 31]

    def scope(name: str, selected: list[float], baseline: list[float]):
        records = [
            {
                "path_id": path_id,
                "state_count": 2,
                "edge_value_count": 4,
                "selected_error_energy": selected[index],
                "baseline_error_energy": baseline[index],
                "target_flux_energy": 1.0 + 0.1 * index,
            }
            for index, path_id in enumerate(path_ids)
        ]
        return {"scope": name, "path_energies": records}

    return {
        "record_sha256": "e" * 64,
        "panel_fingerprint": "p" * 64,
        "panel_role": "b",
        "evaluation_role": "pilot-b",
        "selected_role": "rho-03",
        "baseline_role": "rho-0",
        "path_ids": path_ids,
        "scopes": {
            "overall": scope(
                "overall", [1.0, 3.0, 2.0, 5.0, 2.5], [4.0, 5.0, 5.0, 7.0, 6.0]
            ),
            "data_end": scope(
                "data_end", [0.5, 2.0, 1.0, 3.0, 1.5], [3.0, 4.0, 3.0, 5.0, 4.0]
            ),
        },
    }


def test_joint_bootstrap_is_deterministic_joint_and_path_order_invariant() -> None:
    evaluation = _manual_evaluation()
    first = joint_whole_path_relative_flux_reduction_bootstrap(
        evaluation, reps=2_500, confidence=0.95, seed=123, chunk_size=37
    )
    replay = joint_whole_path_relative_flux_reduction_bootstrap(
        evaluation, reps=2_500, confidence=0.95, seed=123, chunk_size=2_500
    )
    assert replay == first

    reordered = copy.deepcopy(evaluation)
    reordered["path_ids"] = list(reversed(reordered["path_ids"]))
    for value in reordered["scopes"].values():
        value["path_energies"] = list(reversed(value["path_energies"]))
    reordered_result = joint_whole_path_relative_flux_reduction_bootstrap(
        reordered, reps=2_500, confidence=0.95, seed=123, chunk_size=91
    )
    assert reordered_result == first

    assert first["bootstrap_version"] == MATCHED_FLUX_BOOTSTRAP_VERSION
    assert first["cluster_unit"] == "whole_path_id"
    assert first["scope_coupling"] == "same_resampled_path_indices"
    assert first["replicates"] == 2_500
    assert first["path_count"] == 5
    assert first["selected_role"] == "rho-03"
    assert first["baseline_role"] == "rho-0"
    for scope in ("overall", "data_end"):
        value = first["scopes"][scope]
        assert math.isfinite(value["simultaneous_lower_bound"])
        assert value["simultaneous_lower_bound"] <= (
            value["point_relative_flux_l2_reduction"] + 1e-15
        )
    json.dumps(first, sort_keys=True, allow_nan=False)


def test_zero_matched_effect_has_exact_zero_simultaneous_bound() -> None:
    evaluation = _manual_evaluation()
    for scope in evaluation["scopes"].values():
        for record in scope["path_energies"]:
            record["selected_error_energy"] = record["baseline_error_energy"]
    result = joint_whole_path_relative_flux_reduction_bootstrap(
        evaluation, reps=128, confidence=0.95, seed=7
    )
    assert result["simultaneous_critical_shortfall"] == 0.0
    assert result["all_simultaneous_lower_bounds_positive"] == 0
    for scope in ("overall", "data_end"):
        assert result["scopes"][scope]["point_relative_flux_l2_reduction"] == 0.0
        assert result["scopes"][scope]["simultaneous_lower_bound"] == 0.0


def test_confirmation_family_bootstrap_has_exact_18_members_and_role_coupling() -> None:
    evaluations = []
    for model_seed in (261061, 261062, 261063):
        for role in ("b", "c", "d"):
            value = copy.deepcopy(_manual_evaluation())
            value["model_seed"] = model_seed
            value["seed"] = model_seed
            value["panel_role"] = role
            value["evaluation_role"] = role
            value["record_sha256"] = f"{model_seed}-{role}"
            evaluations.append(value)
    first = joint_matched_flux_family_bootstrap(
        evaluations, seed=91, reps=1_000, confidence=0.95, chunk_size=37
    )
    replay = joint_matched_flux_family_bootstrap(
        list(reversed(evaluations)), seed=91, reps=1_000,
        confidence=0.95, chunk_size=1_000,
    )
    assert replay == first
    assert first["passed"] == 1
    assert first["family_size"] == 18
    assert first["path_coupling"] == "joint-within-role-independent-across-roles"
    assert {
        (row["model_seed"], row["panel_role"], row["scope"])
        for row in first["members"]
    } == {
        (seed, role, scope)
        for seed in (261061, 261062, 261063)
        for role in ("b", "c", "d")
        for scope in ("overall", "data_end")
    }
    assert all(
        row["simultaneous_lower_bound"] <= row["point_relative_flux_l2_reduction"]
        for row in first["members"]
    )


def test_confirmation_family_rejects_misaligned_paths_within_role() -> None:
    first = copy.deepcopy(_manual_evaluation())
    first.update({"model_seed": 1, "panel_role": "b"})
    second = copy.deepcopy(_manual_evaluation())
    second.update({"model_seed": 2, "panel_role": "b"})
    second["path_ids"][0] = 999
    for scope in second["scopes"].values():
        scope["path_energies"][0]["path_id"] = 999
    with pytest.raises(ValueError, match="not aligned across model seeds"):
        joint_matched_flux_family_bootstrap(
            [first, second], seed=1, reps=10, confidence=0.95
        )


def test_matched_bootstrap_fails_closed_on_path_or_energy_corruption() -> None:
    missing = _manual_evaluation()
    missing["scopes"]["data_end"]["path_energies"].pop()
    with pytest.raises(ValueError, match="aligned path family"):
        joint_whole_path_relative_flux_reduction_bootstrap(
            missing, reps=16, confidence=0.95, seed=1
        )

    duplicate = _manual_evaluation()
    duplicate["scopes"]["overall"]["path_energies"][1]["path_id"] = 11
    with pytest.raises(ValueError, match="duplicate path ID"):
        joint_whole_path_relative_flux_reduction_bootstrap(
            duplicate, reps=16, confidence=0.95, seed=1
        )

    zero_baseline = _manual_evaluation()
    for scope in zero_baseline["scopes"].values():
        for record in scope["path_energies"]:
            record["baseline_error_energy"] = 0.0
    with pytest.raises(ValueError, match="baseline error energy"):
        joint_whole_path_relative_flux_reduction_bootstrap(
            zero_baseline, reps=16, confidence=0.95, seed=1
        )


def test_matched_evaluation_rejects_non_teacher_panel() -> None:
    plan = build_density_ratio_stream_plan(
        root_seed=261042, grid_size=4, horizon=0.125
    )
    panel = build_density_ratio_panel(
        plan,
        phase="matched-pilot",
        role="b",
        task="dirichlet_null",
        path_count=2,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="bounded-teacher"):
        evaluate_matched_teacher_flux_path_energies(
            _ScaledAnalyticTeacher(0.125, 0.5),
            _ScaledAnalyticTeacher(0.125, 0.0),
            panel,
            _config(),
        )
