from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mnist.d0_score_density_ratio_head import D0BoundarySmoothMeanHeadPotentialUNet
from mnist.d0_v0_density_ratio import (
    build_smoke_cache,
    cache_indices_for_paths,
    dynamics_config,
    split_paths,
)
from mnist.d0_v1_potential_gradient import (
    D0V1Config,
    evaluate,
    mixed_target,
    normalized_increment_loss,
    potential_gradient_loss,
    reverse_rates,
    sample_stratified_batch,
    time_quartiles,
)
from mnist.diag_d0_v1_one_image import main


def test_mixed_target_and_reverse_rate_endpoints() -> None:
    config = D0V1Config.smoke()
    cache = build_smoke_cache(config)
    expected = (
        (1.0 - config.lambda_mix) * cache.target
        + config.lambda_mix / cache.target.size
    )
    np.testing.assert_allclose(mixed_target(cache, config), expected)

    tau = torch.tensor([0.0, cache.horizon], dtype=torch.float32)
    rates = reverse_rates(tau, cache, config)
    torch.testing.assert_close(
        rates,
        torch.tensor(
            [cache.rate_schedule[-1], cache.rate_schedule[0]], dtype=torch.float32
        ),
    )


def test_stratified_batch_covers_all_time_quartiles() -> None:
    config = D0V1Config.smoke()
    cache = build_smoke_cache(config)
    train_paths, _ = split_paths(config)
    indices = cache_indices_for_paths(cache, train_paths)
    states, tau, labels = sample_stratified_batch(
        cache,
        indices,
        config.batch_size,
        rng=np.random.default_rng(config.seed),
        device=torch.device("cpu"),
    )
    assert states.shape == (4, config.grid_size**2)
    assert set(time_quartiles(tau.numpy(), cache.horizon)) == set(range(4))
    assert labels.tolist() == [config.label] * config.batch_size


def test_normalized_increment_loss_prefers_exact_direction() -> None:
    target = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    exact = normalized_increment_loss(target, target)
    opposite = normalized_increment_loss(-target, target)
    torch.testing.assert_close(exact, torch.zeros_like(exact))
    assert float(opposite) > float(exact)


def test_potential_gradient_loss_reaches_model_parameters() -> None:
    config = D0V1Config.smoke()
    cache = build_smoke_cache(config)
    train_paths, _ = split_paths(config)
    indices = cache_indices_for_paths(cache, train_paths)
    states, tau, labels = sample_stratified_batch(
        cache,
        indices,
        config.batch_size,
        rng=np.random.default_rng(config.seed),
        device=torch.device("cpu"),
    )
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics_config(config), base_channels=config.base_channels
    )
    loss = potential_gradient_loss(
        model, states, tau, labels, cache, config, create_graph=True
    )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert torch.isfinite(loss)
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and bool(torch.count_nonzero(gradient))
        for gradient in gradients
    )


def test_full_validation_is_deterministic() -> None:
    config = D0V1Config.smoke()
    cache = build_smoke_cache(config)
    _, validation_paths = split_paths(config)
    indices = cache_indices_for_paths(cache, validation_paths)
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics_config(config), base_channels=config.base_channels
    )
    first = evaluate(model, cache, indices, config, device=torch.device("cpu"))
    second = evaluate(model, cache, indices, config, device=torch.device("cpu"))
    assert first == second


def test_smoke_cli_writes_v1_artifacts(tmp_path: Path) -> None:
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
    with np.load(run_dir / "paired_samples.npz") as samples:
        assert "mixed_target" in samples
