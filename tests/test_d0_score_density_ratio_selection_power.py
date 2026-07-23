from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import mnist.d0_score_density_ratio_selection_power as power
from mnist.d0_score_boundary_controls import bounded_teacher_log_relative_potential
from mnist.d0_score_density_ratio import (
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
)


torch.set_num_threads(1)


def _panel(
    *,
    root_seed: int = 5,
    path_count: int = 4,
    role: str = "calibration",
    start_step: int = 0,
):
    plan = build_density_ratio_stream_plan(
        root_seed=root_seed,
        grid_size=4,
        horizon=0.125,
    )
    return build_density_ratio_panel(
        plan,
        phase="selection-power-test",
        role=role,
        task="bounded_teacher",
        path_count=path_count,
        start_step=start_step,
        dtype=torch.float64,
    )


def test_exact_oracle_logits_and_balanced_bce_improvement() -> None:
    panel = _panel(path_count=3)
    actual = power.exact_bounded_teacher_oracle_logits(panel, batch_size=37)
    expected = bounded_teacher_log_relative_potential(
        panel.states.double(), panel.tau_fraction.double(), epsilon=0.5
    )
    assert actual.dtype == torch.float64
    assert torch.equal(actual, expected)

    improvement = power.balanced_bce_improvement(
        actual, panel.class_targets
    )
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        expected, panel.class_targets.double(), reduction="none"
    )
    assert torch.allclose(improvement, math.log(2.0) - losses, atol=1e-15)

    record = power.evaluate_exact_teacher_oracle_panel(
        panel,
        reps=500,
        confidence=0.90,
        seed=91,
    )
    assert record["finite"] == 1
    assert record["whole_path_structure_valid"] == 1
    assert record["overall"]["state_count"] == 3 * 64
    assert record["data_end"]["state_count"] == 3 * 32
    assert record["overall"]["path_count"] == 3
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0


def test_whole_path_bootstrap_is_one_sided_deterministic_and_path_level() -> None:
    # Unequal row counts make a row bootstrap differ from the required mean of
    # four whole-path means.
    path_ids = np.asarray([10, 20, 20, 30, 30, 30, 40, 40], dtype=np.int64)
    values = np.asarray([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0])
    record = power.whole_path_one_sided_lower_bound(
        values,
        path_ids,
        reps=2500,
        confidence=0.90,
        seed=123,
    )
    means = np.asarray([1.0, 2.0, 3.0, 4.0])
    generator = np.random.default_rng(123)
    bootstrap = means[
        generator.integers(0, 4, size=(2500, 4))
    ].mean(axis=1)
    assert record["point_estimate"] == pytest.approx(2.5)
    assert record["tail_probability"] == pytest.approx(0.10)
    assert record["lower_bound"] == pytest.approx(
        np.quantile(bootstrap, 0.10)
    )

    np.random.seed(999)
    torch.manual_seed(999)
    replay = power.whole_path_one_sided_lower_bound(
        values,
        path_ids,
        reps=2500,
        confidence=0.90,
        seed=123,
    )
    assert replay == record


def test_subset_identity_is_ordered_immutable_and_rejects_foreign_paths() -> None:
    panel = _panel(path_count=4)
    ordered = []
    for value in panel.path_ids.tolist():
        if value not in ordered:
            ordered.append(value)
    first = power.oracle_panel_subset_identity(
        panel, ordered[:2], name="first-half"
    )
    replay = power.oracle_panel_subset_identity(
        panel, ordered[:2], name="first-half"
    )
    reversed_order = power.oracle_panel_subset_identity(
        panel, ordered[:2][::-1], name="first-half"
    )
    assert first == replay
    assert first["path_count"] == 2
    assert first["row_count"] == 128
    assert first["fingerprint"] != reversed_order["fingerprint"]
    with pytest.raises(ValueError, match="not a subset"):
        power.oracle_panel_subset_identity(panel, [-1], name="foreign")


def test_calibration_requires_full_panel_and_predetermined_disjoint_halves() -> None:
    # This deterministic four-path fixture has positive exact-oracle bounds in
    # both fixed halves.  Production calls freeze these counts at 256/128.
    panel = _panel(root_seed=2, path_count=4)
    record = power.evaluate_oracle_power_calibration(
        panel,
        reps=2000,
        seed=10,
        expected_paths=4,
        expected_half_paths=2,
    )
    assert record["passed"] == 1
    assert record["partition_valid"] == 1
    assert record["full"]["passed"] == 1
    assert record["full"]["evidence"]["confidence"] == pytest.approx(0.99)
    assert all(
        value["evidence"]["confidence"] == pytest.approx(0.90)
        for value in record["halves"]
    )
    identities = [
        value["evidence"]["subset"]
        for value in record["halves"]
    ]
    assert set(identities[0]["path_ids"]).isdisjoint(
        identities[1]["path_ids"]
    )
    assert len(set(identities[0]["path_ids"]) | set(identities[1]["path_ids"])) == 4

    wrong_size = power.evaluate_oracle_power_calibration(
        panel,
        reps=10,
        seed=10,
        expected_paths=256,
        expected_half_paths=128,
    )
    assert wrong_size["passed"] == 0
    assert "frozen full/half path counts" in wrong_size["reason"]


def test_panel_feasibility_fails_closed_when_a_scope_lower_bound_is_nonpositive() -> None:
    panel = _panel(root_seed=1, path_count=4)
    record = power.evaluate_oracle_panel_feasibility(
        panel,
        reps=1000,
        confidence=0.99,
        seed=7,
    )
    assert record["passed"] == 0
    assert any(float(value) <= 0.0 for value in record["lower_bounds"].values())


def test_saved_forensic_reproduction_checks_points_bounds_and_power_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel_a = _panel(root_seed=32, path_count=16, role="a", start_step=0)
    panel_b = _panel(root_seed=32, path_count=16, role="b", start_step=100)
    seed = 260931
    # Build a fixture-specific immutable reference using the same published
    # estimator; the production constants bind the real 16-path parent.
    prelim = {
        role: power.evaluate_exact_teacher_oracle_panel(
            panel,
            reps=1000,
            confidence=0.90,
            seed=power.derive_selection_power_seed(
                seed, "saved-16-path", role
            ),
            subset_name=f"saved-parent-{role}",
            scope_seeds=power.SAVED_16_PATH_BOOTSTRAP_SEEDS[role],
        )
        for role, panel in (("a", panel_a), ("b", panel_b))
    }
    reference = {
        role: {
            scope: {
                "point_estimate": prelim[role][scope]["point_estimate"],
                "lower_bound": prelim[role][scope]["lower_bound"],
            }
            for scope in ("overall", "data_end")
        }
        for role in ("a", "b")
    }
    monkeypatch.setattr(
        power,
        "SAVED_16_PATH_PANEL_FINGERPRINTS",
        {"a": panel_a.fingerprint, "b": panel_b.fingerprint},
    )
    monkeypatch.setattr(power, "SAVED_16_PATH_ORACLE_REFERENCE", reference)
    result = power.reproduce_saved_16_path_oracle_forensic(
        panel_a,
        panel_b,
        reps=1000,
        seed=seed,
        point_atol=1e-15,
        lower_bound_atol=1e-15,
    )
    assert result["passed"] == 1
    assert result["panels"]["a"]["overall"]["lower_bound"] > 0.0
    assert result["panels"]["a"]["data_end"]["lower_bound"] > 0.0
    assert result["panels"]["b"]["overall"]["lower_bound"] < 0.0
    assert result["panels"]["b"]["data_end"]["lower_bound"] < 0.0


def test_oracle_helpers_reject_null_panels_and_invalid_inputs() -> None:
    plan = build_density_ratio_stream_plan(
        root_seed=1, grid_size=4, horizon=0.125
    )
    null = build_density_ratio_panel(
        plan,
        phase="test",
        role="null",
        task="dirichlet_null",
        path_count=1,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="teacher panel"):
        power.exact_bounded_teacher_oracle_logits(null)
    with pytest.raises(ValueError, match="confidence"):
        power.whole_path_one_sided_lower_bound(
            np.ones(2), np.arange(2), reps=10, confidence=1.0, seed=1
        )
    with pytest.raises(ValueError, match="matching vectors"):
        power.balanced_bce_improvement(torch.ones(2), torch.ones(3))
