from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_rb_candidate_training_cache import (
    CandidatePrefixCacheSpec,
    build_candidate_prefix_cache,
)
from mnist.d0_jacobi_rb_learnability import PHASE_MATCHINGS, matching_indices
from mnist.d0_jacobi_rb_path_weighted_training import (
    CapacityTrainingConfig,
    compute_cache_loss_scales,
    train_capacity_model,
)


def _target(state: torch.Tensor, phase: int) -> torch.Tensor:
    tails, heads = matching_indices(device=state.device)
    color = int(PHASE_MATCHINGS[phase])
    tail = state[:, tails[color]]
    head = state[:, heads[color]]
    pair = tail + head
    fraction = head / pair
    mobility = fraction * (1.0 - fraction)
    return mobility * (head - tail) * 784.0


def test_small_fake_cache_build_and_training_smoke(tmp_path, monkeypatch) -> None:
    from mnist import d0_jacobi_rb_candidate_training_cache as cache_module

    def prefixes(
        state: torch.Tensor,
        path_ids: Sequence[int],
        *,
        outer_step: int,
        phase: int,
        root_seed: int,
        sample_steps: int,
        prefix_fractions: Sequence[float],
        runtime: object,
    ):
        del path_ids, outer_step, root_seed, sample_steps, runtime
        return tuple(
            (state.clone(), _target(state, phase) * float(value), {})
            for value in prefix_fractions
        )

    def full_phase(
        state: torch.Tensor,
        path_ids: Sequence[int],
        *,
        outer_step: int,
        phase: int,
        root_seed: int,
        sample_steps: int,
        runtime: object,
    ):
        del path_ids, outer_step, root_seed, sample_steps, runtime
        return state.clone(), _target(state, phase), {}

    def finish(
        parts: Sequence[Mapping[str, Any]],
        *,
        runtime: object,
        direction: str,
        outer_step: int,
        elapsed_started: float,
    ):
        del parts, runtime, direction, outer_step, elapsed_started
        return {
            "maximum_mass_error": 0.0,
            "maximum_pair_total_error": 0.0,
            "candidate_maximum_bracket_width": 0.0,
            "outer_step_seconds": 0.0,
        }

    monkeypatch.setattr(cache_module, "candidate_forward_phase_prefixes", prefixes)
    monkeypatch.setattr(cache_module, "candidate_forward_phase", full_phase)
    monkeypatch.setattr(cache_module, "finish_candidate_outer_step", finish)
    runtime = SimpleNamespace(device=torch.device("cpu"), candidate_binary_sha256="a" * 64)
    source = np.arange(1, 785, dtype=np.float64)
    source /= source.sum()
    spec = CandidatePrefixCacheSpec(
        sample_steps=128,
        record_outer_steps=(0, 127),
        prefix_fractions=(0.25, 0.75),
    )
    cache = build_candidate_prefix_cache(
        tmp_path / "cache",
        source,
        label=5,
        path_ids=(1, 2),
        root_seed=123,
        runtime=runtime,
        spec=spec,
    )
    assert len(cache) == 2 * 2 * 7 * 2
    assert cache.verify_hashes()["passed"] == 1
    inputs, target = cache.batch([0, 1, 2], device="cpu")
    assert inputs.later_full_state.shape == (3, 784)
    assert target.shape == (3, 392)

    scales = compute_cache_loss_scales(cache, rows_per_chunk=16)
    report = train_capacity_model(
        cache,
        cache,
        config=CapacityTrainingConfig(
            architecture="small",
            loss_name="path-weighted",
            updates=1,
            batch_size=4,
            learning_rate=1e-3,
            validation_interval=1,
            validation_batch_size=32,
            seed=9,
        ),
        device="cpu",
        scales=scales,
        output_dir=tmp_path / "training",
    )
    assert report["completed_updates"] == 1
    assert report["selected_update"] == 1
