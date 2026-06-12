"""Smoke checks for Experiment 12/D0 Phase 0 forward noising diagnostics."""

from __future__ import annotations

import json

import numpy as np
import torch

from mnist.diag_forward_noising import (
    _gate_summary,
    _synthetic_digit_measures,
    run_forward_noising_single,
    save_phase0_result,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def test_d0_phase0_forward_noising_smoke(tmp_path) -> None:
    images, labels = _synthetic_digit_measures(examples_per_class=2, grid_size=8, seed=123)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        num_steps=4,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        ot_com_weight=0.10,
        limiter_fraction=0.25,
        mass_floor=1e-12,
    )
    result = run_forward_noising_single(
        images=images,
        labels=labels,
        config=config,
        lambda_mix=0.05,
        free_weight=0.0,
        noise_weight=0.002,
        sample_steps=4,
        num_paths=6,
        batch_size=3,
        theta_mask_min=1e-12,
        preview_fractions=[0.0, 0.5, 1.0],
        metric_bins=2,
        device=torch.device("cpu"),
        seed=11,
        show_progress=False,
    )
    assert result.initial_states.shape == (6, 64)
    assert result.terminal_states.shape == (6, 64)
    assert np.allclose(result.initial_states.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(result.terminal_states.sum(axis=1), 1.0, atol=1e-5)
    assert {0, 2, 4}.issubset(set(result.checkpoint_states))
    assert len(result.metrics) >= 3
    assert result.summary["cumulative_clip_fraction"] >= 0.0

    gated = _gate_summary(
        result,
        max_final_corr=1.1,
        max_clip_fraction=1.0,
        min_entropy_fraction=0.0,
        max_frozen_edge_fraction=1.0,
    )
    assert gated["gate_pass"] == 1

    result.summary = gated
    paths = save_phase0_result(result, tmp_path, preview_images=4, save_previews=True)
    assert (tmp_path / result.run_id / "metrics.csv").exists()
    assert (tmp_path / result.run_id / "prior_bank.npz").exists()
    assert (tmp_path / result.run_id / "forward_noising_panel.png").exists()
    assert "prior_bank_path" in paths
    summary = json.loads((tmp_path / result.run_id / "summary.json").read_text())
    assert summary["gate_pass"] == 1
