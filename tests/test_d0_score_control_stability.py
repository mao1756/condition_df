from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from mnist.d0_dirichlet_score import dirichlet_score_objective
from mnist.d0_score_boundary_controls import (
    bounded_teacher_log_relative_potential,
    orthogonal_hadamard_edge_probes,
    sample_bounded_teacher_mixture,
)
from mnist.d0_score_control_stability import (
    FROZEN_BATCH_BIN_COUNTS,
    STREAM_DERIVATION_VERSION,
    STREAM_SCHEMA,
    analytic_bounded_score_objective_terms,
    build_stream_plan,
    evaluate_pilot_profile,
    generate_stream_batch,
    run_stein_identity_preflight,
    select_stability_profile,
    stateless_probe_banks,
    stream_plan_record,
    stream_replay_record,
    verify_stream_replay,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig


torch.set_num_threads(1)


def _config(grid_size: int = 4) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
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


def _plan():
    return build_stream_plan(root_seed=260801, grid_size=4, horizon=0.125)


def test_stream_plan_and_batches_freeze_exact_strata_and_stateless_freshness() -> None:
    plan = _plan()
    record = stream_plan_record(plan)
    assert record["schema"] == STREAM_SCHEMA
    assert record["derivation_version"] == STREAM_DERIVATION_VERSION
    assert record["batch_size"] == 64
    assert tuple(record["batch_bin_counts"]) == FROZEN_BATCH_BIN_COUNTS
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0

    first = generate_stream_batch(
        plan, phase="pilot", law="bounded_teacher", step=17
    )
    replay = generate_stream_batch(
        plan, phase="pilot", law="bounded_teacher", step=17
    )
    next_step = generate_stream_batch(
        plan, phase="pilot", law="bounded_teacher", step=18
    )
    confirmation = generate_stream_batch(
        plan, phase="confirmation", law="bounded_teacher", step=17
    )
    null = generate_stream_batch(
        plan, phase="pilot", law="dirichlet_null", step=17
    )

    assert torch.equal(first.states, replay.states)
    assert first.fingerprint == replay.fingerprint
    assert first.fingerprint != next_step.fingerprint
    assert first.fingerprint != confirmation.fingerprint
    assert first.fingerprint != null.fingerprint
    assert first.seed != next_step.seed != confirmation.seed
    assert not torch.equal(first.states, next_step.states)
    assert first.states.shape == (64, 16)
    assert torch.allclose(first.states.sum(1), torch.ones(64), atol=2e-6)
    assert tuple((first.strata == index).sum() for index in range(5)) == (
        8,
        8,
        8,
        8,
        32,
    )
    assert tuple((first.strata[first.cluster_ids == cluster] == index).sum()
                 for cluster in (34, 35) for index in range(5)) == (
        4,
        4,
        4,
        4,
        16,
        4,
        4,
        4,
        4,
        16,
    )


def test_stateless_two_bank_probes_are_replayable_independent_and_orthogonal() -> None:
    plan = _plan()
    first = stateless_probe_banks(
        plan, phase="pilot", law="bounded_teacher", step=9
    )
    replay = stateless_probe_banks(
        plan, phase="pilot", law="bounded_teacher", step=9
    )
    reordered = stateless_probe_banks(
        plan, phase="pilot", law="dirichlet_null", step=9
    )
    assert torch.equal(first.a, replay.a)
    assert torch.equal(first.b, replay.b)
    assert first.fingerprint == replay.fingerprint
    assert first.fingerprint != reordered.fingerprint
    assert first.seeds["a"] != first.seeds["b"]
    assert not torch.equal(first.a, first.b)
    assert first.a.shape == (4, 64, 2, 4, 4)
    expected = 32.0 * torch.eye(4)
    for bank in (first.a, first.b):
        flat = bank.flatten(2)
        for batch_index in (0, 17, 63):
            assert torch.equal(flat[:, batch_index] @ flat[:, batch_index].T, expected)


def test_stream_replay_certificate_detects_tampering() -> None:
    plan = _plan()
    record = stream_replay_record(
        plan, phase="pilot", law="bounded_teacher", step=3
    )
    assert verify_stream_replay(plan, record)["passed"] == 1
    changed = copy.deepcopy(record)
    changed["batch"]["state_sha256"] = "0" * 64
    assert verify_stream_replay(plan, changed)["passed"] == 0


class _AnalyticBoundedPotential(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(
        self, reverse_fraction: torch.Tensor, states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del labels
        return self.scale * bounded_teacher_log_relative_potential(
            states, reverse_fraction
        )


@pytest.mark.parametrize("scale", [0.25, 1.0, 2.0])
def test_rank_one_analytic_objective_matches_exact_complete_probe_objective(
    scale: float,
) -> None:
    config = _config()
    fractions = torch.tensor([0.15, 0.55, 0.95], dtype=torch.float64)
    states = sample_bounded_teacher_mixture(
        fractions, 4, seed=260802, dtype=torch.float64
    )
    probes = orthogonal_hadamard_edge_probes(
        32,
        3,
        4,
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(260803),
    )
    exact = dirichlet_score_objective(
        _AnalyticBoundedPotential(scale),
        fractions,
        states,
        torch.full((3,), 3, dtype=torch.long),
        config,
        probes,
        create_graph=False,
    )
    analytic = analytic_bounded_score_objective_terms(
        states, fractions, config, scale=scale
    )
    assert torch.allclose(analytic["energy"], exact.energy, atol=2e-12, rtol=2e-12)
    assert torch.allclose(analytic["trace"], exact.trace, atol=2e-12, rtol=2e-12)
    assert torch.allclose(analytic["drift"], exact.drift, atol=2e-12, rtol=2e-12)
    assert torch.allclose(
        analytic["objective"], exact.per_sample, atol=2e-12, rtol=2e-12
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_streamed_cpu_cuda_diagnostics_are_equivalent() -> None:
    plan = _plan()
    cpu_batch = generate_stream_batch(
        plan,
        phase="confirm",
        law="bounded_teacher",
        step=23,
        device="cpu",
        dtype=torch.float32,
    )
    cuda_batch = generate_stream_batch(
        plan,
        phase="confirm",
        law="bounded_teacher",
        step=23,
        device="cuda",
        dtype=torch.float32,
    )
    assert cpu_batch.fingerprint == cuda_batch.fingerprint
    assert torch.equal(cpu_batch.states, cuda_batch.states.cpu())
    assert torch.equal(cpu_batch.tau_fraction, cuda_batch.tau_fraction.cpu())

    cpu_probes = stateless_probe_banks(
        plan,
        phase="confirm",
        law="bounded_teacher",
        step=23,
        device="cpu",
        dtype=torch.float32,
    )
    cuda_probes = stateless_probe_banks(
        plan,
        phase="confirm",
        law="bounded_teacher",
        step=23,
        device="cuda",
        dtype=torch.float32,
    )
    assert cpu_probes.fingerprint == cuda_probes.fingerprint
    assert torch.equal(cpu_probes.a, cuda_probes.a.cpu())
    assert torch.equal(cpu_probes.b, cuda_probes.b.cpu())

    cpu_metrics = analytic_bounded_score_objective_terms(
        cpu_batch.states,
        cpu_batch.tau_fraction,
        _config(),
        scale=0.5,
    )
    cuda_metrics = analytic_bounded_score_objective_terms(
        cuda_batch.states,
        cuda_batch.tau_fraction,
        _config(),
        scale=0.5,
    )
    for name in ("objective", "energy", "generator", "trace", "drift", "edge_score"):
        torch.testing.assert_close(
            cpu_metrics[name],
            cuda_metrics[name].cpu(),
            atol=2e-6,
            rtol=2e-6,
        )


def test_exact_stein_identity_preflight_passes_on_fresh_whole_path_panels() -> None:
    report = run_stein_identity_preflight(
        _config(),
        root_seed=260801,
        path_count=128,
        bootstrap_reps=2_000,
        confidence=0.99,
    )
    assert report["passed"] == 1
    assert report["finite"] == 1
    assert report["path_count_per_law"] == 128
    assert report["anchors_per_path"] == 32
    assert report["null_identity"]["passed"] == 1
    assert [value["scale"] for value in report["teacher_identities"]] == [
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
    ]
    assert all(value["passed"] == 1 for value in report["teacher_identities"])
    assert report["physical_training_performed"] == 0
    assert report["sampling_performed"] == 0


def _banks(*, lower: float, risk: float) -> dict[str, object]:
    return {
        bank: {
            scope: {"lower_bound": lower, "model_score_risk": risk}
            for scope in ("overall", "data_end")
        }
        for bank in ("a", "b")
    }


def _teacher_result(*, risk: float = -0.4, clip: float = 0.02) -> dict[str, object]:
    return {
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "clip_fraction_steps_101_1000": clip,
        "final_200_clip_fraction": clip,
        "selected_step": 250,
        "selection_objective_banks": _banks(lower=0.1, risk=risk),
        "selection_overall_score_gain": 0.2,
        "selection_data_end_score_gain": 0.1,
        "overall_flux_cosine": 0.3,
        "data_end_flux_cosine": 0.2,
        "overall_relative_flux_l2": 0.8,
        "data_end_relative_flux_l2": 0.9,
    }


def _null_result(*, clip: float = 0.01) -> dict[str, object]:
    return {
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "clip_fraction_steps_101_1000": clip,
        "final_200_clip_fraction": clip,
        "selected_step": 0,
        "selection_objective_banks": _banks(lower=0.0, risk=0.0),
    }


def test_pilot_profile_gate_and_frozen_ranking_are_fail_closed() -> None:
    low_rate = evaluate_pilot_profile(
        _teacher_result(risk=-0.5, clip=0.02),
        _null_result(clip=0.03),
        learning_rate=1e-5,
    )
    high_rate = evaluate_pilot_profile(
        _teacher_result(risk=-0.4, clip=0.01),
        _null_result(clip=0.01),
        learning_rate=3e-5,
    )
    assert low_rate["eligible"] == 1
    assert high_rate["eligible"] == 1
    selected = select_stability_profile([high_rate, low_rate])
    assert selected["passed"] == 1
    assert selected["selected"]["learning_rate"] == pytest.approx(1e-5)

    clipping_failure = evaluate_pilot_profile(
        _teacher_result(clip=0.1000001), _null_result(), learning_rate=1e-4
    )
    null_failure = _null_result()
    null_failure["selected_step"] = 25
    null_signal_failure = evaluate_pilot_profile(
        _teacher_result(), null_failure, learning_rate=3e-6
    )
    assert clipping_failure["eligible"] == 0
    assert null_signal_failure["eligible"] == 0
    failed = select_stability_profile([clipping_failure, null_signal_failure])
    assert failed["passed"] == 0
    assert failed["selected"] is None


def test_pilot_ranking_breaks_risk_ties_by_clip_then_smaller_rate() -> None:
    first = evaluate_pilot_profile(
        _teacher_result(risk=-0.4, clip=0.03),
        _null_result(clip=0.03),
        learning_rate=3e-5,
    )
    second = evaluate_pilot_profile(
        _teacher_result(risk=-0.4, clip=0.02),
        _null_result(clip=0.02),
        learning_rate=1e-5,
    )
    third = evaluate_pilot_profile(
        _teacher_result(risk=-0.4, clip=0.02),
        _null_result(clip=0.02),
        learning_rate=3e-6,
    )
    selected = select_stability_profile([first, second, third])
    assert selected["selected"]["learning_rate"] == pytest.approx(3e-6)
