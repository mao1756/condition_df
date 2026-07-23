from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from mnist.d0_score_boundary_controls import (
    bounded_teacher_density_ratio,
    bounded_teacher_log_relative_potential,
)
from mnist.d0_score_density_ratio import (
    DENSITY_RATIO_OBJECTIVE_VERSION,
    FROZEN_CLASS_BIN_COUNTS,
    analytic_teacher_metrics,
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
    calibrate_density_ratio_loss_scale,
    class_posterior_from_log_ratio,
    classification_loss,
    correct_logit_for_class_prior,
    density_ratio_replay_record,
    equal_prior_bayes_logit,
    evaluate_classification_panel,
    evaluate_classification_risk,
    generate_density_ratio_batch,
    load_density_ratio_panel,
    panel_disjointness_record,
    panel_identity,
    save_density_ratio_panel,
    scaled_classification_loss,
    stream_plan_record,
    verify_density_ratio_replay,
    verify_panel_identity,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig


torch.set_num_threads(1)


def _plan(root_seed: int = 260841):
    return build_density_ratio_stream_plan(
        root_seed=root_seed, grid_size=4, horizon=0.125
    )


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


def test_teacher_and_null_streams_are_exactly_balanced_and_replayable() -> None:
    plan = _plan()
    record = stream_plan_record(plan)
    assert record["batch_size"] == 64
    assert record["examples_per_class"] == 32
    assert record["objective_version"] == DENSITY_RATIO_OBJECTIVE_VERSION

    torch.manual_seed(1)
    teacher = generate_density_ratio_batch(
        plan, phase="pilot", task="bounded_teacher", step=17, dtype=torch.float64
    )
    torch.manual_seed(999)
    replay = generate_density_ratio_batch(
        plan, phase="pilot", task="bounded_teacher", step=17, dtype=torch.float64
    )
    next_step = generate_density_ratio_batch(
        plan, phase="pilot", task="bounded_teacher", step=18, dtype=torch.float64
    )
    null = generate_density_ratio_batch(
        plan, phase="pilot", task="dirichlet_null", step=17, dtype=torch.float64
    )
    assert torch.equal(teacher.states, replay.states)
    assert teacher.fingerprint == replay.fingerprint
    assert teacher.fingerprint != next_step.fingerprint
    assert teacher.fingerprint != null.fingerprint
    assert int(teacher.class_targets.sum()) == 32
    assert torch.allclose(teacher.states.sum(1), torch.ones(64, dtype=torch.float64))

    targets = teacher.class_targets.cpu().numpy().astype(np.int64)
    for class_value in (0, 1):
        assert tuple(
            int(((targets == class_value) & (teacher.strata == index)).sum())
            for index in range(5)
        ) == FROZEN_CLASS_BIN_COUNTS
    for anchor in range(32):
        ids = np.flatnonzero(teacher.anchor_ids == anchor)
        assert len(ids) == 2
        assert set(targets[ids]) == {0, 1}
        assert teacher.tau_fraction[ids[0]] == teacher.tau_fraction[ids[1]]

    # The null uses one pooled sampler namespace and a stateless paired swap,
    # rather than separate class-conditioned state RNG namespaces.
    assert null.seeds["null-pool"] != null.seeds["null-swaps"]
    assert int(null.class_targets.sum()) == 32


def test_stream_replay_certificate_fails_closed_on_tampering() -> None:
    plan = _plan()
    record = density_ratio_replay_record(
        plan, phase="pilot", task="bounded_teacher", step=3
    )
    assert verify_density_ratio_replay(plan, record)["passed"] == 1
    changed = copy.deepcopy(record)
    changed["batch"]["state_sha256"] = "0" * 64
    verification = verify_density_ratio_replay(plan, changed)
    assert verification["passed"] == 0
    assert verification["reason"]


def test_equal_prior_bayes_logit_and_prior_correction_are_exact() -> None:
    batch = generate_density_ratio_batch(
        _plan(), phase="preflight", task="bounded_teacher", step=5, dtype=torch.float64
    )
    expected = torch.log(
        bounded_teacher_density_ratio(batch.states, batch.tau_fraction, epsilon=0.5)
    )
    actual = equal_prior_bayes_logit(
        batch.states, batch.tau_fraction, task="bounded_teacher", epsilon=0.5
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert torch.equal(
        equal_prior_bayes_logit(
            batch.states, batch.tau_fraction, task="dirichlet_null"
        ),
        torch.zeros_like(actual),
    )

    prior = 0.2
    prior_log_odds = math.log(prior / (1.0 - prior))
    shifted = actual + prior_log_odds
    torch.testing.assert_close(
        correct_logit_for_class_prior(shifted, positive_prior=prior), actual
    )
    torch.testing.assert_close(
        class_posterior_from_log_ratio(actual, positive_prior=prior),
        torch.sigmoid(shifted),
    )


def test_classification_loss_uses_raw_logits_and_requires_equal_priors() -> None:
    logits = torch.tensor([-2.0, -0.5, 0.5, 2.0], dtype=torch.float64)
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    expected = F.binary_cross_entropy_with_logits(logits, targets)
    torch.testing.assert_close(classification_loss(logits, targets), expected)
    per_sample = classification_loss(logits, targets, reduction="none")
    assert per_sample.shape == (4,)
    unscaled, scaled = scaled_classification_loss(
        logits, targets, loss_scale=0.125
    )
    torch.testing.assert_close(unscaled, expected)
    torch.testing.assert_close(scaled, expected * 0.125)
    with pytest.raises(ValueError, match="balanced"):
        classification_loss(logits, torch.tensor([0.0, 1.0, 1.0, 1.0]))


class _AnalyticTeacherLogit(nn.Module):
    def __init__(self, horizon: float) -> None:
        super().__init__()
        self.horizon = float(horizon)

    def forward(
        self, tau: torch.Tensor, states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del labels
        return bounded_teacher_log_relative_potential(
            states, tau / self.horizon, epsilon=0.5
        )


class _ZeroLogit(nn.Module):
    def forward(
        self, tau: torch.Tensor, states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del tau, labels
        return states[:, 0] * 0.0


class _TimeBinOffsetAnalyticTeacherLogit(nn.Module):
    def __init__(self, horizon: float) -> None:
        super().__init__()
        self.horizon = float(horizon)

    def forward(
        self, tau: torch.Tensor, states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del labels
        fraction = tau / self.horizon
        time_bin = torch.bucketize(
            fraction,
            fraction.new_tensor((0.2, 0.4, 0.6, 0.8)),
            right=False,
        )
        return bounded_teacher_log_relative_potential(
            states, fraction, epsilon=0.5
        ) + 0.25 * time_bin.to(dtype=states.dtype)


def test_exact_bayes_classifier_beats_zero_risk_and_null_zero_is_exact() -> None:
    plan = _plan()
    teacher_panel = build_density_ratio_panel(
        plan,
        phase="preflight",
        role="teacher-risk",
        task="bounded_teacher",
        path_count=32,
        dtype=torch.float64,
    )
    teacher = evaluate_classification_panel(
        _AnalyticTeacherLogit(plan.horizon), teacher_panel, batch_size=128
    )
    assert teacher["finite"] == 1
    assert teacher["objective_improvement"] > 0.0
    assert teacher["data_end"]["objective_improvement"] > 0.0
    assert len(teacher["per_path"]) == 32

    null_panel = build_density_ratio_panel(
        plan,
        phase="preflight",
        role="null-risk",
        task="dirichlet_null",
        path_count=3,
        dtype=torch.float64,
    )
    null = evaluate_classification_panel(_ZeroLogit(), null_panel)
    assert null["risk"] == pytest.approx(math.log(2.0), abs=1e-15)
    assert null["objective_improvement"] == pytest.approx(0.0, abs=1e-15)
    assert all(row["objective_improvement_vs_zero"] == 0.0 for row in null["per_path"])


def test_classification_risk_reports_balanced_time_bins_and_paths() -> None:
    batch = generate_density_ratio_batch(
        _plan(), phase="report", task="bounded_teacher", step=4
    )
    logits = equal_prior_bayes_logit(
        batch.states, batch.tau_fraction, task="bounded_teacher"
    )
    report = evaluate_classification_risk(
        logits,
        batch.class_targets,
        strata=batch.strata,
        path_ids=batch.path_ids,
    )
    assert len(report["time_bins"]) == 5
    assert report["data_end"]["positive_count"] == 16
    assert report["data_end"]["negative_count"] == 16
    assert len(report["per_path"]) == 1


def test_analytic_raw_logit_has_exact_score_and_physical_flux() -> None:
    plan = _plan()
    panel = build_density_ratio_panel(
        plan,
        phase="audit",
        role="analytic-teacher",
        task="bounded_teacher",
        path_count=3,
        dtype=torch.float64,
    )
    metrics = analytic_teacher_metrics(
        _AnalyticTeacherLogit(plan.horizon),
        panel,
        _config(),
        batch_size=16,
    )
    assert metrics["finite"] == 1
    assert metrics["audit_overall_score_gain"] == pytest.approx(1.0, abs=2e-12)
    assert metrics["audit_data_end_score_gain"] == pytest.approx(1.0, abs=2e-12)
    assert metrics["overall_flux_cosine"] == pytest.approx(1.0, abs=2e-12)
    assert metrics["overall_relative_flux_l2"] == pytest.approx(0.0, abs=2e-12)
    assert all(value == pytest.approx(1.0, abs=2e-12) for value in metrics["time_bin_flux_cosines"])
    assert all(value == pytest.approx(0.0, abs=2e-12) for value in metrics["time_bin_relative_flux_l2"])
    assert metrics["overall"]["logit_mse"] == pytest.approx(0.0, abs=2e-24)
    assert metrics["raw_logit_correlation"] == pytest.approx(1.0, abs=2e-12)
    assert metrics["time_bin_centered_logit_mse"] == pytest.approx(0.0, abs=2e-24)
    assert metrics["time_bin_centered_logit_correlation"] == pytest.approx(
        1.0, abs=2e-12
    )
    for key in (
        "predicted_cell_gradient_abs_quantiles",
        "predicted_edge_score_abs_quantiles",
        "predicted_physical_flux_abs_quantiles",
    ):
        quantiles = metrics[key]
        values = [
            quantiles[name]
            for name in ("q00", "q10", "q50", "q90", "q99", "q100")
        ]
        assert all(math.isfinite(value) and value >= 0.0 for value in values)
        assert values == sorted(values)
        assert quantiles == metrics["overall"][key]


def test_time_bin_centered_logit_metrics_remove_only_binwise_offsets() -> None:
    plan = _plan()
    panel = build_density_ratio_panel(
        plan,
        phase="audit",
        role="analytic-teacher-offset",
        task="bounded_teacher",
        path_count=3,
        dtype=torch.float64,
    )
    metrics = analytic_teacher_metrics(
        _TimeBinOffsetAnalyticTeacherLogit(plan.horizon),
        panel,
        _config(),
        batch_size=16,
    )
    assert metrics["overall"]["logit_mse"] > 0.0
    assert metrics["raw_logit_correlation"] < 1.0
    assert metrics["time_bin_centered_logit_mse"] == pytest.approx(
        0.0, abs=2e-24
    )
    assert metrics["time_bin_centered_logit_correlation"] == pytest.approx(
        1.0, abs=2e-12
    )
    assert metrics["audit_overall_score_gain"] == pytest.approx(1.0, abs=2e-12)
    assert metrics["overall_flux_cosine"] == pytest.approx(1.0, abs=2e-12)


def test_panels_replay_round_trip_and_are_role_disjoint(tmp_path) -> None:
    plan = _plan()
    first = build_density_ratio_panel(
        plan,
        phase="pilot",
        role="selection-a",
        task="bounded_teacher",
        path_count=2,
    )
    replay = build_density_ratio_panel(
        plan,
        phase="pilot",
        role="selection-a",
        task="bounded_teacher",
        path_count=2,
    )
    second = build_density_ratio_panel(
        plan,
        phase="pilot",
        role="selection-b",
        task="bounded_teacher",
        path_count=2,
    )
    audit = build_density_ratio_panel(
        plan,
        phase="confirmation",
        role="audit-a",
        task="bounded_teacher",
        path_count=2,
    )
    assert first.fingerprint == replay.fingerprint
    assert torch.equal(first.states, replay.states)
    disjoint = panel_disjointness_record([first, second, audit])
    assert disjoint["passed"] == 1
    assert not disjoint["overlaps"]

    binding = {"scientific_fingerprint": "abc", "parent": "stability"}
    path = save_density_ratio_panel(tmp_path / "selection-a.pt", first, binding)
    loaded = load_density_ratio_panel(
        path,
        binding,
        expected_plan_fingerprint=plan.fingerprint,
        expected_role="selection-a",
        expected_task="bounded_teacher",
    )
    assert loaded.fingerprint == first.fingerprint
    assert torch.equal(loaded.states, first.states)
    assert verify_panel_identity(loaded, panel_identity(first))["passed"] == 1
    with pytest.raises(ValueError, match="binding"):
        load_density_ratio_panel(path, {"scientific_fingerprint": "changed"})


class _ScaledLinearLogit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, tau: torch.Tensor, states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del labels
        return self.weight * 1_000_000.0 * (states[:, 0] + 0.1 * tau)


def test_density_ratio_calibration_is_training_only_and_scales_before_backward() -> None:
    plan = _plan()
    panel = build_density_ratio_panel(
        plan,
        phase="calibration",
        role="train-calibration",
        task="bounded_teacher",
        path_count=4,
        dtype=torch.float64,
    )
    model = _ScaledLinearLogit().double()
    calibration = calibrate_density_ratio_loss_scale(
        model,
        panel,
        batch_size=64,
        target_initial_gradient_norm=0.1,
        binding={"initialization_seed": 260842},
    )
    assert calibration.objective_kind == "density_ratio_balanced_raw_logit_bce"
    assert calibration.calibration_state_count == 256
    assert calibration.training_only == 1
    assert calibration.loss_scale < 1.0
    assert calibration.scaled_initial_gradient_norm == pytest.approx(0.1, rel=1e-12)
    assert calibration.binding["panel_fingerprint"] == panel.fingerprint
    assert model.weight.grad is None

    selection = build_density_ratio_panel(
        plan,
        phase="confirmation",
        role="selection-a",
        task="bounded_teacher",
        path_count=1,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="training-only"):
        calibrate_density_ratio_loss_scale(model, selection)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_density_ratio_stream_is_device_independent() -> None:
    plan = _plan()
    cpu = generate_density_ratio_batch(
        plan, phase="pilot", task="bounded_teacher", step=9, device="cpu"
    )
    cuda = generate_density_ratio_batch(
        plan, phase="pilot", task="bounded_teacher", step=9, device="cuda"
    )
    assert cpu.fingerprint == cuda.fingerprint
    assert torch.equal(cpu.states, cuda.states.cpu())
    assert torch.equal(cpu.class_targets, cuda.class_targets.cpu())
