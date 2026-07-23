from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from mnist.d0_dirichlet_score import (
    edge_difference_channels,
    harmonic_mobility_exact,
    physical_flux_from_edge_score,
)
from mnist.d0_score_density_ratio_h1_trust import (
    H1_TRUST_BIN_COUNTS,
    H1_TRUST_OPERATOR_VERSION,
    H1_TRUST_SCALE_FLOOR,
    H1TrustPlan,
    build_h1_trust_plan,
    calibrate_h1_trust,
    derive_reference_trust_seed,
    generate_reference_trust_batch,
    h1_increment_components,
    h1_trust_plan_record,
)
from mnist.d0_score_density_ratio_head import D0BoundarySmoothMeanHeadPotentialUNet
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig


class LinearPotential(nn.Module):
    def __init__(self, coefficients: torch.Tensor, bias: float = 0.0) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(coefficients.detach().clone())
        self.bias = nn.Parameter(coefficients.new_tensor(float(bias)))

    def forward(
        self, tau: torch.Tensor, states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        del labels
        return (states * self.coefficients[None, :]).sum(dim=1) + self.bias + 0.0 * tau


def _dynamics(grid_size: int = 4) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
        alpha_eff=1.0,
        edge_alpha_mode="alpha_eff",
        source_lowfreq_size=min(7, grid_size),
        ot_lowres_size=min(7, grid_size),
    )


def _plan(grid_size: int = 4) -> H1TrustPlan:
    return build_h1_trust_plan(
        grid_size=grid_size, horizon=0.125, root_seed=261001, label=3
    )


def test_h1_trust_plan_record_is_stable_and_fail_closed() -> None:
    first = _plan()
    second = H1TrustPlan(grid_size=4, horizon=0.125)
    assert first.fingerprint == second.fingerprint
    record = h1_trust_plan_record(first)
    assert record["fingerprint"] == first.fingerprint
    assert record["candidate_fields_in_seed"] == []
    assert record["banks_per_update"] == 2
    assert record["states_per_bank"] == 32
    assert record["bin_counts"] == [4, 4, 4, 4, 16]
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0
    json.dumps(record, sort_keys=True)

    with pytest.raises(ValueError, match="two banks"):
        H1TrustPlan(grid_size=4, horizon=0.125, banks_per_update=1)
    with pytest.raises(ValueError, match="32 states"):
        H1TrustPlan(grid_size=4, horizon=0.125, states_per_bank=16)
    with pytest.raises(ValueError, match="time strata"):
        H1TrustPlan(grid_size=4, horizon=0.125, bin_counts=(8, 8, 8, 4, 4))


def test_reference_trust_stream_replays_and_is_order_invariant() -> None:
    plan = _plan(grid_size=3)
    bank_one_first = generate_reference_trust_batch(
        plan,
        phase="pilot",
        task="bounded_teacher",
        optimizer_step=17,
        bank=1,
        dtype=torch.float64,
    )
    bank_zero = generate_reference_trust_batch(
        plan,
        phase="pilot",
        task="bounded_teacher",
        optimizer_step=17,
        bank=0,
        dtype=torch.float64,
    )
    bank_one_replay = generate_reference_trust_batch(
        plan,
        phase="pilot",
        task="bounded_teacher",
        optimizer_step=17,
        bank=1,
        dtype=torch.float64,
    )

    assert bank_one_first.fingerprint == bank_one_replay.fingerprint
    assert torch.equal(bank_one_first.states, bank_one_replay.states)
    assert torch.equal(bank_one_first.tau_fraction, bank_one_replay.tau_fraction)
    assert np.array_equal(bank_one_first.strata, bank_one_replay.strata)
    assert bank_zero.fingerprint != bank_one_first.fingerprint
    assert torch.allclose(bank_zero.states.sum(dim=1), torch.ones(32, dtype=torch.float64))
    assert bool((bank_zero.states > 0.0).all())
    assert tuple(
        int(np.count_nonzero(bank_zero.strata == index)) for index in range(5)
    ) == H1_TRUST_BIN_COUNTS
    assert bank_zero.record()["candidate_independent"] == 1

    # Candidate identity is deliberately absent from the derivation API.  The
    # same scientific stream cursor therefore always produces the same seed.
    seed = derive_reference_trust_seed(
        plan, "pilot", "bounded_teacher", 17, 0, "reference-gamma"
    )
    assert seed == bank_zero.gamma_seed
    assert seed == derive_reference_trust_seed(
        plan, "pilot", "bounded_teacher", 17, 0, "reference-gamma"
    )
    null = generate_reference_trust_batch(
        plan,
        phase="pilot",
        task="dirichlet_null",
        optimizer_step=17,
        bank=0,
        dtype=torch.float64,
    )
    assert null.fingerprint != bank_zero.fingerprint


def test_h1_increment_matches_full_l2_gamma_and_physical_flux() -> None:
    dtype = torch.float64
    plan = _plan()
    batch = generate_reference_trust_batch(
        plan,
        phase="unit",
        task="bounded_teacher",
        optimizer_step=1,
        bank=0,
        dtype=dtype,
    )
    coefficients = torch.linspace(-0.75, 0.75, 16, dtype=dtype)
    model = LinearPotential(coefficients, bias=0.2)
    ema = LinearPotential(torch.zeros_like(coefficients), bias=-0.1)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    dynamics = _dynamics()

    components = h1_increment_components(
        model,
        ema,
        batch.tau,
        batch.states,
        batch.labels,
        dynamics,
        value_scale=2.0,
        energy_scale=4.0,
        create_graph=True,
    )
    expected_logits = batch.states @ coefficients + 0.3
    expected_gradient = coefficients[None, :].expand(batch.size, -1)
    expected_edge = edge_difference_channels(expected_gradient, 4)
    theta = harmonic_mobility_exact(batch.states, dynamics)
    expected_natural = (theta * expected_edge.square()).flatten(1).mean(dim=1)
    expected_flux = physical_flux_from_edge_score(
        expected_edge, batch.states, dynamics, time_change=1.0
    )
    expected_physical = expected_flux.square().flatten(1).mean(dim=1)

    assert torch.allclose(components.per_state_value, expected_logits.square())
    assert torch.allclose(components.per_state_natural_energy, expected_natural)
    assert torch.allclose(components.per_state_physical_flux_energy, expected_physical)
    assert torch.allclose(
        components.normalized_value, expected_logits.square().mean() / 4.0
    )
    assert torch.allclose(
        components.normalized_natural_energy, expected_natural.mean() / 16.0
    )
    assert torch.allclose(
        components.objective,
        0.5
        * (
            expected_logits.square().mean() / 4.0
            + expected_natural.mean() / 16.0
        ),
    )
    components.objective.backward()
    assert model.coefficients.grad is not None
    assert bool(torch.isfinite(model.coefficients.grad).all())
    assert all(parameter.grad is None for parameter in ema.parameters())
    record = components.detached_record()
    assert record["operator_version"] == H1_TRUST_OPERATOR_VERSION
    assert math.isclose(
        record["physical_flux_step_rms"],
        math.sqrt(float(expected_physical.mean())),
        rel_tol=1e-12,
    )


def test_h1_calibration_uses_shadow_increment_and_balances_gradients() -> None:
    dtype = torch.float64
    plan = _plan()
    batches = [
        generate_reference_trust_batch(
            plan,
            phase="calibration",
            task="bounded_teacher",
            optimizer_step=1,
            bank=bank,
            dtype=dtype,
        )
        for bank in (0, 1)
    ]
    shadow = LinearPotential(
        torch.linspace(-0.4, 0.5, 16, dtype=dtype), bias=0.05
    )
    prestep_ema = LinearPotential(torch.zeros(16, dtype=dtype), bias=0.0)
    for parameter in prestep_ema.parameters():
        parameter.requires_grad_(False)
    scaled_bce_norm = 0.125
    calibration = calibrate_h1_trust(
        shadow,
        prestep_ema,
        batches,
        _dynamics(),
        scaled_bce_gradient_norm=scaled_bce_norm,
        binding={"shadow_step": 1, "split": "train"},
    )

    assert calibration.passed == 1
    assert calibration.calibration_state_count == 64
    assert len(calibration.trust_batch_fingerprints) == 2
    assert calibration.value_scale > H1_TRUST_SCALE_FLOOR
    assert calibration.energy_scale > H1_TRUST_SCALE_FLOOR
    assert calibration.normalized_h1_gradient_norm > H1_TRUST_SCALE_FLOOR
    assert calibration.lambda_base is not None
    assert math.isclose(
        calibration.lambda_base * calibration.normalized_h1_gradient_norm,
        scaled_bce_norm,
        rel_tol=1e-12,
    )
    assert calibration.value_scale_floor_hit == 0
    assert calibration.energy_scale_floor_hit == 0
    assert calibration.h1_gradient_floor_hit == 0
    assert calibration.training_only == 1
    assert calibration.shadow_steps == 1
    json.dumps(calibration.to_record(), sort_keys=True)
    assert all(parameter.grad is None for parameter in prestep_ema.parameters())


def test_h1_calibration_reports_every_floor_hit_fail_closed() -> None:
    plan = _plan()
    batches = [
        generate_reference_trust_batch(
            plan,
            phase="calibration",
            task="dirichlet_null",
            optimizer_step=1,
            bank=bank,
            dtype=torch.float64,
        )
        for bank in (0, 1)
    ]
    shadow = LinearPotential(torch.zeros(16, dtype=torch.float64), bias=0.0)
    prestep_ema = LinearPotential(torch.zeros(16, dtype=torch.float64), bias=0.0)
    calibration = calibrate_h1_trust(
        shadow,
        prestep_ema,
        batches,
        _dynamics(),
        scaled_bce_gradient_norm=0.0,
    )
    assert calibration.passed == 0
    assert calibration.lambda_base is None
    assert calibration.value_scale == H1_TRUST_SCALE_FLOOR
    assert calibration.energy_scale == H1_TRUST_SCALE_FLOOR
    assert calibration.value_scale_floor_hit == 1
    assert calibration.energy_scale_floor_hit == 1
    assert calibration.bce_gradient_floor_hit == 1
    assert calibration.h1_gradient_floor_hit == 1


def test_h1_increment_rejects_invalid_scales() -> None:
    batch = generate_reference_trust_batch(
        _plan(),
        phase="unit",
        task="bounded_teacher",
        optimizer_step=1,
        bank=0,
        dtype=torch.float64,
    )
    model = LinearPotential(torch.ones(16, dtype=torch.float64))
    ema = LinearPotential(torch.zeros(16, dtype=torch.float64))
    with pytest.raises(ValueError, match="value_scale"):
        h1_increment_components(
            model,
            ema,
            batch.tau,
            batch.states,
            batch.labels,
            _dynamics(),
            value_scale=0.0,
            energy_scale=1.0,
            create_graph=False,
        )


def test_h1_increment_actual_boundary_model_has_finite_mixed_backward() -> None:
    dynamics = _dynamics()
    model = D0BoundarySmoothMeanHeadPotentialUNet(dynamics, base_channels=8)
    ema = copy.deepcopy(model)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        model.out.weight.fill_(0.01)
        model.out.bias.fill_(0.005)
    batch = generate_reference_trust_batch(
        _plan(),
        phase="unit",
        task="bounded_teacher",
        optimizer_step=2,
        bank=0,
    )
    components = h1_increment_components(
        model,
        ema,
        batch.tau,
        batch.states,
        batch.labels,
        dynamics,
        value_scale=1.0,
        energy_scale=1.0,
        create_graph=True,
    )
    assert float(components.value_mean.detach()) > 0.0
    assert float(components.natural_energy_mean.detach()) > 0.0
    components.objective.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(parameter.grad is None for parameter in ema.parameters())
