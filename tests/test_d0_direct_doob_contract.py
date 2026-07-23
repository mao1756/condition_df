"""Focused contracts for the direct-Doob D0 reverse parameterization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mnist.experiment12_d0 as d0_module
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, free_drift_flux_torch


def _toy_dynamics(*, grid_size: int = 4) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
        num_steps=1,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        source_uniform_mix=0.10,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        ot_com_weight=0.10,
        condition_on_source=False,
        flux_parameterization="edge",
        limiter_fraction=1.0,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-10,
    )


def _valid_doob_config(**overrides: object) -> d0_module.Experiment12D0Config:
    values: dict[str, object] = {
        "cache_build_mode": "substep",
        "teacher_stride_substeps": 1,
        "d0_target_space": "doob-physical-residual",
        "physical_target_normalization": "none",
        "physical_sampler_noise_mode": "reference",
        "eta_l2_weight": 0.0,
        "state_delta_loss_weight": 0.0,
        "rollout_loss_weight": 0.0,
        "trajectory_rollout_loss_weight": 0.0,
        "invalid_output_l2_weight": 0.0,
        "curl_loss_weight": 0.0,
        "edge_laplacian_loss_weight": 0.0,
        "sample_steps": 1,
        "reference_substeps": 1,
        "tau_eff": 0.1,
    }
    values.update(overrides)
    return d0_module.Experiment12D0Config(**values)


@pytest.mark.parametrize(
    "alias",
    (
        "doob-physical-residual",
        "doob_physical_residual",
        "direct-doob",
        "doob-residual",
        "direct-physical-residual",
    ),
)
def test_doob_target_space_aliases_have_one_canonical_name(alias: str) -> None:
    assert d0_module._normalize_d0_target_space(alias) == "doob-physical-residual"
    assert d0_module._d0_is_physical_space(alias)


def test_direct_doob_and_legacy_residual_baselines_have_opposite_signs() -> None:
    dynamics = _toy_dynamics()
    states = torch.tensor(
        [
            [0.18, 0.04, 0.08, 0.02, 0.03, 0.12, 0.02, 0.07, 0.05, 0.03, 0.09, 0.04, 0.02, 0.06, 0.08, 0.07],
            [0.03, 0.11, 0.02, 0.05, 0.14, 0.03, 0.04, 0.06, 0.02, 0.08, 0.12, 0.04, 0.07, 0.03, 0.09, 0.07],
        ],
        dtype=torch.float32,
    )
    states = states / states.sum(dim=1, keepdim=True)
    batch = {
        "states": states,
        "starts": torch.tensor([0, 1], dtype=torch.long),
        "stride_substeps": torch.ones(2, dtype=torch.long),
        "reference_substeps": torch.ones(2, dtype=torch.long),
        "dt_sub": torch.full((2,), 0.01),
        "rate_schedule": torch.tensor([2.0, 5.0]),
    }

    free = free_drift_flux_torch(states, dynamics)
    rates = torch.tensor([2.0, 5.0]).view(2, 1, 1, 1)
    expected_direct = rates * free * 0.01
    direct = d0_module._direct_reverse_free_block_baseline_from_batch(batch, dynamics)
    legacy = d0_module._reverse_free_block_baseline_from_batch(batch, dynamics)

    assert float(expected_direct.abs().max()) > 0.0
    assert torch.allclose(direct, expected_direct)
    assert torch.allclose(legacy, -expected_direct)
    assert torch.allclose(direct, -legacy)

    zero = torch.zeros_like(direct)
    direct_total = d0_module._physical_total_prediction_from_raw(
        zero,
        batch,
        dynamics,
        _valid_doob_config(),
    )
    legacy_total = d0_module._physical_total_prediction_from_raw(
        zero,
        batch,
        dynamics,
        d0_module.Experiment12D0Config(d0_target_space="physical-residual"),
    )
    assert torch.allclose(direct_total, direct)
    assert torch.allclose(legacy_total, legacy)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"control_output_clip": 0.5}, "control_output_clip must be 0"),
        ({"edge_innovation_loss_weight": 0.0}, "edge_innovation_loss_weight must be positive"),
        ({"physical_target_scale": float("nan")}, "physical_target_scale must be finite and non-negative"),
        ({"physical_target_scale_floor": 0.0}, "physical_target_scale_floor must be finite and positive"),
    ),
)
def test_direct_doob_rejects_settings_that_change_or_remove_the_conditional_mean_objective(
    overrides: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        d0_module._validate_direct_doob_config(_valid_doob_config(**overrides))


def test_direct_doob_baseline_requires_interval_metadata() -> None:
    states = torch.full((1, 16), 1.0 / 16.0)
    with pytest.raises(ValueError, match="direct Doob reference baseline requires interval metadata"):
        d0_module._direct_reverse_free_block_baseline_from_batch(
            {"states": states},
            _toy_dynamics(),
        )


def test_physical_scale_inference_ignores_nonfinite_entries() -> None:
    target = torch.tensor([3.0, float("nan"), 4.0])
    scale = d0_module._physical_target_scale_from_tensor(
        target,
        _valid_doob_config(physical_target_normalization="global-rms"),
    )
    assert torch.isfinite(scale)
    assert scale.item() == pytest.approx(np.sqrt(12.5))


def test_masked_mse_averages_slices_instead_of_valid_edges_globally() -> None:
    dynamics = _toy_dynamics()
    pred = torch.zeros((2, 2, 4, 4))
    target = torch.zeros_like(pred)
    mask = torch.zeros_like(pred)

    target[0, 0, 0, 0] = 1.0
    mask[0, 0, 0, 0] = 1.0
    target[1, 0, :2, :2] = 3.0
    mask[1, 0, :2, :2] = 1.0

    loss, zero_loss, residual_rms, pred_rms, target_rms = d0_module._d0_masked_edge_mse(
        pred,
        target,
        mask,
        dynamics,
        d0_module.Experiment12D0Config(d0_target_space="raw"),
    )

    # Slice losses are 1 and 9, hence their equally weighted mean is 5.
    # A global valid-edge mean would instead be (1 + 4 * 9) / 5 = 7.4.
    assert loss.item() == pytest.approx(5.0)
    assert zero_loss.item() == pytest.approx(5.0)
    assert residual_rms.item() == pytest.approx(np.sqrt(7.4))
    assert pred_rms.item() == pytest.approx(0.0)
    assert target_rms.item() == pytest.approx(np.sqrt(7.4))


class _ZeroEdgeModel(torch.nn.Module):
    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.grid_size = int(grid_size)

    def forward(
        self,
        tau: torch.Tensor,
        states: torch.Tensor,
        labels: torch.Tensor,
        source: torch.Tensor | None,
    ) -> torch.Tensor:
        del tau, labels, source
        return states.new_zeros((states.shape[0], 2, self.grid_size, self.grid_size))


class _FixedEdgeModel(torch.nn.Module):
    def __init__(self, edge: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("edge", edge)

    def forward(
        self,
        tau: torch.Tensor,
        states: torch.Tensor,
        labels: torch.Tensor,
        source: torch.Tensor | None,
    ) -> torch.Tensor:
        del tau, labels, source
        return self.edge.expand(states.shape[0], -1, -1, -1)


def test_direct_doob_loss_regresses_projected_realized_minus_positive_free() -> None:
    dynamics = _toy_dynamics()
    config = _valid_doob_config()
    states = torch.tensor(
        [[0.18, 0.04, 0.08, 0.02, 0.03, 0.12, 0.02, 0.07,
          0.05, 0.03, 0.09, 0.04, 0.02, 0.06, 0.08, 0.07]],
        dtype=torch.float32,
    )
    states = states / states.sum(dim=1, keepdim=True)
    realized_reverse = torch.linspace(-0.02, 0.03, 32).reshape(1, 2, 4, 4)
    baseline_batch = {
        "states": states,
        "starts": torch.zeros(1, dtype=torch.long),
        "stride_substeps": torch.ones(1, dtype=torch.long),
        "reference_substeps": torch.ones(1, dtype=torch.long),
        "dt_sub": torch.full((1,), 0.01),
        "rate_schedule": torch.tensor([2.0]),
    }
    positive_free = d0_module._direct_reverse_free_block_baseline_from_batch(
        baseline_batch,
        dynamics,
    )
    expected_residual = d0_module.project_edge_flux_torch(
        realized_reverse - positive_free,
        grid_size=4,
    )
    batch = {
        **baseline_batch,
        "tau": torch.zeros(1),
        "labels": torch.zeros(1, dtype=torch.long),
        "innovations": torch.zeros_like(realized_reverse),
        "masks": torch.ones_like(realized_reverse, dtype=torch.bool),
        "earlier_states": states.clone(),
        "physical_transfers": realized_reverse,
    }

    loss, diagnostics = d0_module.d0_unweighted_innovation_loss(
        _FixedEdgeModel(expected_residual),
        batch,
        dynamics,
        config,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["loss_main"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["d0_target_space_normalized"] == "doob-physical-residual"
    assert diagnostics["physical_target_source"] == "realized-doob-residual"


def test_direct_cache_summary_uses_scaled_physical_residual_not_innovation_mask() -> None:
    dynamics = _toy_dynamics()
    config = _valid_doob_config(physical_target_normalization="global-rms", physical_target_scale=2.0)
    states = torch.tensor(
        [
            [0.18, 0.04, 0.08, 0.02, 0.03, 0.12, 0.02, 0.07, 0.05, 0.03, 0.09, 0.04, 0.02, 0.06, 0.08, 0.07],
            [0.03, 0.11, 0.02, 0.05, 0.14, 0.03, 0.04, 0.06, 0.02, 0.08, 0.12, 0.04, 0.07, 0.03, 0.09, 0.07],
        ],
        dtype=torch.float32,
    )
    states = states / states.sum(dim=1, keepdim=True)
    metadata = {
        "states": states,
        "starts": torch.tensor([0, 1], dtype=torch.long),
        "stride_substeps": torch.ones(2, dtype=torch.long),
        "reference_substeps": torch.ones(2, dtype=torch.long),
        "dt_sub": torch.full((2,), 0.01),
        "rate_schedule": torch.tensor([2.0, 5.0]),
    }
    baseline = d0_module._direct_reverse_free_block_baseline_from_batch(metadata, dynamics)
    residual_seed = torch.zeros_like(baseline)
    residual_seed[0, 0, 0, 0] = 0.03
    residual_seed[1, 1, 1, 1] = -0.07
    projected = d0_module.project_edge_flux_torch(residual_seed, grid_size=4)
    physical = baseline + residual_seed
    expected_scaled = projected / 2.0
    cache = SimpleNamespace(
        size=2,
        states=states,
        tau=torch.tensor([0.1, 0.9]),
        labels=torch.zeros(2, dtype=torch.long),
        innovations=torch.full_like(physical, 11.0),
        masks=torch.zeros_like(physical, dtype=torch.bool),
        starts=metadata["starts"],
        path_indices=torch.arange(2),
        start_images=states.clone(),
        earlier_states=states.clone(),
        physical_transfers=physical,
        physical_target_scale=2.0,
        terminal_states=np.zeros((1, 4, 4), dtype=np.float32),
        source_indices=np.zeros(1, dtype=np.int64),
        requested_labels=np.zeros(1, dtype=np.int64),
        rate_schedule=np.asarray([2.0, 5.0]),
        horizon=1.0,
        dt_sub=0.01,
        stride_substeps=1,
        requested_stride_substeps=1,
        sample_steps=2,
        reference_substeps=1,
        lambda_mix=0.1,
        raw_limited_fraction=1.0,
        mobility_weighted_limited_fraction=float("nan"),
        noise_energy_weighted_limited_fraction=float("nan"),
        valid_innovation_fraction=0.0,
        valid_innovation_mobility_fraction=float("nan"),
        valid_innovation_noise_energy_fraction=float("nan"),
        floor_correction_l1=0.0,
        renorm_correction_l1=0.0,
        floor_touched_pixels=0,
        floor_proposed_pixels=32,
        floor_touched_fraction=0.0,
        cache_build_mode="substep",
        trajectory_window_states=None,
        trajectory_window_valid=None,
        trajectory_window_depths=None,
    )

    summary = d0_module.cache_summary(cache, dynamics, config)

    assert summary["cache_target_source"] == "projected-realized-minus-positive-free-scaled"
    assert summary["cache_raw_innovation_rms"] == pytest.approx(11.0)
    assert summary["cache_target_rms"] == pytest.approx(float(torch.sqrt(expected_scaled.square().mean())))
    assert summary["cache_target_finite_fraction"] == pytest.approx(1.0)
    assert summary["cache_valid_mask_fraction"] == pytest.approx(0.0)
    assert np.isfinite(summary["cache_tau_bin0_target_rms"])
    assert np.isnan(summary["cache_tau_bin0_raw_innovation_rms"])


def test_tiny_exact_direct_cache_reports_theory_gate_diagnostics_and_direct_oracle() -> None:
    dynamics = _toy_dynamics()
    image = np.linspace(1.0, 16.0, 16, dtype=np.float32)
    image /= image.sum()
    config = _valid_doob_config(
        physical_target_normalization="global-rms",
        cache_paths=2,
        cache_batch_size=2,
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=1,
        single_image_overfit=True,
        cache_time_sampling="uniform",
    )
    cache = d0_module.build_d0_training_cache(
        dataset_images=image.reshape(1, 4, 4),
        dataset_labels=np.asarray([3], dtype=np.int64),
        dynamics_config=dynamics,
        d0_config=config,
        device=torch.device("cpu"),
        rng=np.random.default_rng(4),
        show_progress=False,
    )
    effective = d0_module._with_cache_physical_scale(config, cache, dynamics)
    summary = d0_module.cache_summary(cache, dynamics, effective)
    oracle = d0_module.d0_direct_doob_oracle_diagnostic(
        cache,
        dynamics,
        max_slices=2,
        device=torch.device("cpu"),
    )

    assert np.isfinite(cache.mobility_weighted_limited_fraction)
    assert np.isfinite(cache.noise_energy_weighted_limited_fraction)
    assert np.isfinite(cache.floor_correction_l1)
    assert np.isfinite(cache.renorm_correction_l1)
    assert cache.floor_proposed_pixels == 2 * 2 * 16
    assert np.isfinite(cache.floor_touched_fraction)
    assert summary["cache_target_source"] == "projected-realized-minus-positive-free-scaled"
    assert summary["cache_target_rms"] == pytest.approx(1.0, rel=1e-5)
    assert oracle["oracle_semantics"] == "direct-doob"
    assert oracle["oracle_mode"] == "direct_realized_minus_positive_free"
    assert np.isfinite(oracle["oracle_direct_rms"])


@pytest.mark.parametrize("deterministic", (True, False))
def test_direct_doob_sampler_adds_free_drift_and_optional_reference_noise(
    deterministic: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamics = _toy_dynamics()
    d0_config = _valid_doob_config()
    initial = np.full((1, 16), 1.0 / 16.0, dtype=np.float32)
    bank = {
        "terminal_states": initial,
        "labels": np.asarray([3], dtype=np.int64),
        "rate_schedule": np.asarray([2.0], dtype=np.float64),
        "sample_steps": 1,
        "substeps": 1,
        "start_substep": 1,
        "horizon": 0.1,
        "stride_substeps": 1,
        "physical_target_scale": 1.0,
    }
    applied: list[torch.Tensor] = []

    monkeypatch.setattr(d0_module, "_load_prior_bank", lambda *args, **kwargs: bank)
    monkeypatch.setattr(d0_module, "_validate_prior_bank_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        d0_module,
        "free_drift_flux_torch",
        lambda states, config: states.new_full((states.shape[0], 2, config.grid_size, config.grid_size), 3.0),
    )
    monkeypatch.setattr(
        d0_module,
        "edge_noise_std_channels",
        lambda states, dt, config: states.new_full((states.shape[0], 2, config.grid_size, config.grid_size), 5.0),
    )
    monkeypatch.setattr(d0_module.torch, "randn_like", lambda tensor: torch.ones_like(tensor))

    def capture_transfer(
        states: torch.Tensor,
        delta: torch.Tensor,
        config: DirectFluxMNISTConfig,
        **kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del config, kwargs
        applied.append(delta.detach().clone())
        return states, {
            "limited_edges": 0.0,
            "proposed_edges": float(delta.numel()),
            "mobility_weight_sum": 1.0,
            "limited_mobility_weight_sum": 0.0,
        }

    monkeypatch.setattr(d0_module, "_apply_oriented_edge_transfer", capture_transfer)
    result = d0_module.simulate_d0_reverse_generation(
        _ZeroEdgeModel(4),
        [3],
        dynamics_config=dynamics,
        d0_config=d0_config,
        prior_bank_path=Path("unused-doob-prior.npz"),
        device=torch.device("cpu"),
        seed=1,
        deterministic=deterministic,
        control_strength=1.0,
        show_progress=False,
    )

    assert len(applied) == 1
    expected_free = 2.0 * 3.0 * 0.1
    expected_noise = 0.0 if deterministic else np.sqrt(2.0) * 5.0
    assert torch.allclose(applied[0], torch.full_like(applied[0], expected_free + expected_noise))
    assert result.physical_sampler_mode == "doob-physical-residual"
    assert result.physical_sampler_noise_mode == "reference"


def test_rollout_undoes_each_cached_block_from_its_later_edge_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For start k and stride r, reverse q=k+r-1,...,k before earlier blocks."""

    dynamics = _toy_dynamics()
    state = torch.tensor(
        [[0.18, 0.04, 0.08, 0.02, 0.03, 0.12, 0.02, 0.07, 0.05, 0.03, 0.09, 0.04, 0.02, 0.06, 0.08, 0.07]],
        dtype=torch.float32,
    )
    state = state / state.sum(dim=1, keepdim=True)
    cache = SimpleNamespace(
        size=1,
        states=state,
        labels=torch.tensor([0], dtype=torch.long),
        start_images=state.clone(),
        tau=torch.zeros(1),
        starts=torch.tensor([2], dtype=torch.long),
        stride_substeps=2,
        reference_substeps=2,
        dt_sub=0.01,
        rate_schedule=np.asarray([10.0, 100.0]),
        horizon=1.0,
        physical_target_scale=1.0,
    )
    applied_means: list[float] = []

    monkeypatch.setattr(
        d0_module,
        "free_drift_flux_torch",
        lambda states, config: states.new_ones((states.shape[0], 2, config.grid_size, config.grid_size)),
    )
    monkeypatch.setattr(
        d0_module,
        "edge_noise_std_channels",
        lambda states, dt, config: states.new_zeros((states.shape[0], 2, config.grid_size, config.grid_size)),
    )

    def capture_transfer(
        states: torch.Tensor,
        delta: torch.Tensor,
        config: DirectFluxMNISTConfig,
        **kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del config, kwargs
        applied_means.append(float(delta.mean()))
        return states, {
            "limited_edges": 0.0,
            "proposed_edges": float(delta.numel()),
            "mobility_weight_sum": 1.0,
            "limited_mobility_weight_sum": 0.0,
        }

    monkeypatch.setattr(d0_module, "_apply_oriented_edge_transfer", capture_transfer)
    d0_module.d0_learned_rollout_diagnostic(
        _ZeroEdgeModel(4),
        cache,
        dynamics,
        d0_module.Experiment12D0Config(d0_target_space="raw", sample_project_learned_mean=False),
        max_slices=1,
        block_counts=[1, 2],
        device=torch.device("cpu"),
    )

    # start=2 and stride=2 represents the block q=2,3. Reverse it as 3,2,
    # then continue with the preceding block 1,0. Raw mode subtracts free drift.
    assert applied_means == pytest.approx([-1.0, -1.0, -0.1, -0.1])
