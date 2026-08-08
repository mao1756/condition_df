from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mnist.d0_dirichlet_score import (
    edge_difference_channels,
    physical_flux_from_edge_score,
)
from mnist.d0_v0_density_ratio import (
    D0V0Config,
    build_smoke_cache,
    cache_indices_for_paths,
    density_ratio_flux_delta,
    dynamics_config,
    run_paired_sampling,
    sample_balanced_batch,
    split_paths,
    train,
)
from mnist.diag_d0_v0_one_image import main


def test_whole_path_split_is_deterministic_and_disjoint() -> None:
    config = D0V0Config.smoke()
    first = split_paths(config)
    second = split_paths(config)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert not np.intersect1d(*first).size
    assert np.union1d(*first).size == config.cache_paths

    cache = build_smoke_cache(config)
    train_indices = cache_indices_for_paths(cache, first[0])
    validation_indices = cache_indices_for_paths(cache, first[1])
    assert not np.intersect1d(train_indices, validation_indices).size


def test_balanced_batch_matches_time_and_labels() -> None:
    config = D0V0Config.smoke()
    cache = build_smoke_cache(config)
    indices = np.arange(len(cache.states))
    states, tau, labels, targets = sample_balanced_batch(
        cache,
        indices,
        3,
        alpha_eff=config.alpha_eff,
        rng=np.random.default_rng(config.seed),
        device=torch.device("cpu"),
    )
    assert states.shape == (6, config.grid_size**2)
    torch.testing.assert_close(tau[:3], tau[3:])
    torch.testing.assert_close(labels[:3], labels[3:])
    torch.testing.assert_close(states.sum(dim=1), torch.ones(6))
    torch.testing.assert_close(targets, torch.tensor([1, 1, 1, 0, 0, 0.0]))


def test_checkpoint_ema_and_resume(tmp_path: Path) -> None:
    base = D0V0Config.smoke()
    cache = build_smoke_cache(base)
    one_step = replace(base, train_steps=1)
    result = train(
        cache,
        one_step,
        tmp_path,
        device=torch.device("cpu"),
        show_progress=False,
    )
    assert np.isfinite(result["best_validation_bce"])
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best_ema.pt").is_file()

    train(
        cache,
        replace(base, train_steps=2),
        tmp_path,
        device=torch.device("cpu"),
        show_progress=False,
    )
    payload = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert payload["step"] == 2


def test_density_ratio_flux_has_edge_orientation_and_dt() -> None:
    config = D0V0Config.smoke()
    cells = config.grid_size**2
    weights = torch.arange(cells, dtype=torch.float32)

    class LinearPotential(torch.nn.Module):
        def forward(self, tau, states, labels):
            return (states * weights.to(states)).sum(dim=1)

    states = torch.full((1, cells), 1.0 / cells)
    tau = torch.ones(1)
    labels = torch.full((1,), config.label, dtype=torch.long)
    rate, dt = 0.25, 0.125
    actual = density_ratio_flux_delta(
        LinearPotential(), tau, states, labels, config, rate=rate, dt=dt
    )
    edge_score = edge_difference_channels(weights[None, :], config.grid_size)
    expected = physical_flux_from_edge_score(
        edge_score, states, dynamics_config(config), time_change=rate
    ) * dt
    torch.testing.assert_close(actual, expected)


def test_paired_sampler_shares_noise_and_zero_model_bypasses() -> None:
    config = D0V0Config.smoke()
    cache = build_smoke_cache(config)
    cache = replace(
        cache,
        terminal_states=cache.terminal_states.reshape(
            config.cache_paths, config.grid_size, config.grid_size
        ),
    )
    from mnist.d0_score_density_ratio_head import D0BoundarySmoothMeanHeadPotentialUNet

    zero_model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics_config(config), base_channels=config.base_channels
    )
    samples, _, summary = run_paired_sampling(
        zero_model, cache, config, device=torch.device("cpu")
    )
    np.testing.assert_array_equal(
        samples["samples_strength0"], samples["samples_strength1"]
    )
    assert summary["nonfinite_edges"] == 0


def test_smoke_cli_writes_minimal_artifacts(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--smoke",
                "--stage",
                "all",
                "--device",
                "cpu",
                "--no-progress",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    run_dir = next(tmp_path.iterdir())
    expected = {
        "config.json",
        "cache.npz",
        "latest.pt",
        "best_ema.pt",
        "training.csv",
        "validation.json",
        "paired_samples.npz",
        "paired_metrics.csv",
        "paired_contact_sheet.png",
        "summary.json",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}
