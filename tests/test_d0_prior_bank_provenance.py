"""Contracts for D0 terminal-bank reference-law provenance."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import mnist.experiment12_d0 as d0_module
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, edge_alpha_value, natural_horizon


def _configs() -> tuple[DirectFluxMNISTConfig, d0_module.Experiment12D0Config]:
    dynamics = DirectFluxMNISTConfig(
        grid_size=4,
        num_steps=2,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        limiter_fraction=0.375,
        edge_alpha_mode="alpha_eff",
        alpha_eff=0.75,
        mass_floor=2e-8,
    )
    config = d0_module.Experiment12D0Config(
        cache_build_mode="substep",
        teacher_stride_substeps=1,
        d0_target_space="doob-physical-residual",
        sample_steps=2,
        reference_substeps=1,
        tau_eff=1e-4,
    )
    return dynamics, config


def _cache(
    dynamics: DirectFluxMNISTConfig,
    config: d0_module.Experiment12D0Config,
) -> d0_module.D0TrainingCache:
    count = 2
    dim = int(dynamics.grid_size) ** 2
    states = torch.full((count, dim), 1.0 / float(dim), dtype=torch.float32)
    edge = torch.zeros((count, 2, dynamics.grid_size, dynamics.grid_size), dtype=torch.float32)
    horizon = natural_horizon(dynamics)
    rates = d0_module._expected_rate_schedule(dynamics, config)
    return d0_module.D0TrainingCache(
        states=states.clone(),
        tau=torch.zeros(count),
        labels=torch.tensor([1, 2], dtype=torch.long),
        innovations=edge.clone(),
        masks=torch.ones_like(edge, dtype=torch.bool),
        starts=torch.zeros(count, dtype=torch.long),
        path_indices=torch.arange(count, dtype=torch.long),
        start_images=states.clone(),
        earlier_states=states.clone(),
        physical_transfers=edge.clone(),
        physical_target_scale=1.0,
        terminal_states=states.numpy().copy(),
        source_indices=np.arange(count, dtype=np.int64),
        requested_labels=np.asarray([1, 2], dtype=np.int64),
        rate_schedule=rates,
        horizon=horizon,
        dt_sub=horizon / float(config.sample_steps * config.reference_substeps),
        stride_substeps=1,
        sample_steps=config.sample_steps,
        reference_substeps=config.reference_substeps,
        lambda_mix=config.lambda_mix,
        raw_limited_fraction=0.0,
        mobility_weighted_limited_fraction=0.0,
        noise_energy_weighted_limited_fraction=0.0,
        valid_innovation_fraction=1.0,
        valid_innovation_mobility_fraction=1.0,
        valid_innovation_noise_energy_fraction=1.0,
        cache_build_mode="substep",
        requested_stride_substeps=1,
    )


def test_training_terminal_bank_saves_complete_reference_provenance(tmp_path) -> None:
    dynamics, config = _configs()
    cache = _cache(dynamics, config)
    path = tmp_path / "training_terminal_bank.npz"

    d0_module._save_training_prior_bank(
        path,
        cache,
        dynamics,
        config,
        path_indices=np.asarray([0, 1]),
        subset_name="all",
    )

    with np.load(path) as saved:
        assert int(saved["grid_size"][0]) == dynamics.grid_size
        assert float(saved["mass_floor"][0]) == pytest.approx(dynamics.mass_floor)
        assert float(saved["limiter_fraction"][0]) == pytest.approx(dynamics.limiter_fraction)
        assert str(saved["edge_alpha_mode"][0]) == dynamics.edge_alpha_mode
        assert float(saved["edge_alpha_value"][0]) == pytest.approx(edge_alpha_value(dynamics))
        assert str(saved["reference_integrator"][0]) == "masked_reference_free_step_torch"
        assert int(saved["reference_integrator_version"][0]) == 1
        assert int(saved["sample_steps"][0]) == config.sample_steps
        assert int(saved["substeps"][0]) == config.reference_substeps
        assert np.array_equal(saved["rate_schedule"], cache.rate_schedule)

    loaded = d0_module._load_prior_bank(path)
    d0_module._validate_prior_bank_compatibility(
        loaded,
        d0_config=config,
        dynamics_config=dynamics,
        prior_bank_path=path,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("mass_floor", 9e-7), ("limiter_fraction", 0.125)),
)
def test_prior_bank_rejects_reference_geometry_mismatch(
    tmp_path,
    field: str,
    replacement: float,
) -> None:
    dynamics, config = _configs()
    cache = _cache(dynamics, config)
    path = tmp_path / "training_terminal_bank.npz"
    d0_module._save_training_prior_bank(
        path,
        cache,
        dynamics,
        config,
        path_indices=np.asarray([0, 1]),
        subset_name="all",
    )
    loaded = d0_module._load_prior_bank(path)
    loaded[field] = replacement

    with pytest.raises(ValueError, match=field):
        d0_module._validate_prior_bank_compatibility(
            loaded,
            d0_config=config,
            dynamics_config=dynamics,
            prior_bank_path=path,
        )


def test_prior_bank_rejects_spatial_grid_and_terminal_width_mismatch(tmp_path) -> None:
    dynamics, config = _configs()
    cache = _cache(dynamics, config)
    path = tmp_path / "training_terminal_bank.npz"
    d0_module._save_training_prior_bank(
        path,
        cache,
        dynamics,
        config,
        path_indices=np.asarray([0, 1]),
        subset_name="all",
    )
    loaded = d0_module._load_prior_bank(path)

    wrong_grid = dict(loaded)
    wrong_grid["grid_size"] = dynamics.grid_size + 1
    with pytest.raises(ValueError, match="grid_size"):
        d0_module._validate_prior_bank_compatibility(
            wrong_grid,
            d0_config=config,
            dynamics_config=dynamics,
            prior_bank_path=path,
        )

    wrong_width = dict(loaded)
    wrong_width["terminal_states"] = np.zeros((2, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="terminal_states shape"):
        d0_module._validate_prior_bank_compatibility(
            wrong_width,
            d0_config=config,
            dynamics_config=dynamics,
            prior_bank_path=path,
        )


def test_legacy_bank_warns_but_only_strict_direct_mode_refuses(tmp_path) -> None:
    dynamics, strict_config = _configs()
    cache = _cache(dynamics, strict_config)
    legacy_path = tmp_path / "legacy_terminal_bank.npz"
    np.savez_compressed(
        legacy_path,
        terminal_states=cache.terminal_states,
        labels=cache.requested_labels,
        rate_schedule=cache.rate_schedule,
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        stride_substeps=np.asarray([cache.stride_substeps], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
        lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
        edge_alpha_mode=np.asarray([dynamics.edge_alpha_mode]),
        alpha_eff=np.asarray([dynamics.alpha_eff], dtype=np.float64),
        cache_build_mode=np.asarray([cache.cache_build_mode]),
    )
    legacy = d0_module._load_prior_bank(legacy_path)

    legacy_config = replace(strict_config, d0_target_space="projected-edge")
    with pytest.warns(RuntimeWarning, match="missing provenance metadata"):
        d0_module._validate_prior_bank_compatibility(
            legacy,
            d0_config=legacy_config,
            dynamics_config=dynamics,
            prior_bank_path=legacy_path,
        )

    with pytest.warns(RuntimeWarning, match="missing provenance metadata"):
        with pytest.raises(ValueError, match="fully self-describing D0 prior bank"):
            d0_module._validate_prior_bank_compatibility(
                legacy,
                d0_config=strict_config,
                dynamics_config=dynamics,
                prior_bank_path=legacy_path,
            )

    override = replace(strict_config, allow_prior_bank_mismatch=True)
    with pytest.warns(RuntimeWarning, match="missing provenance metadata"):
        d0_module._validate_prior_bank_compatibility(
            legacy,
            d0_config=override,
            dynamics_config=dynamics,
            prior_bank_path=legacy_path,
        )
