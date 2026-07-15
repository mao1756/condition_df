from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mnist.d0_multiscale_cache import (
    D0MultiscaleCache,
    D0MultiscaleCompatibilityError,
    aggregate_aligned_block_quantities,
    block_arithmetic_metrics,
    block_residual_targets,
    build_multiscale_cache_shard,
    deterministic_three_way_path_split,
    evaluate_multiscale_cache_preflight,
    exact_reverse_reference_step_transfer,
    infer_training_block_scales,
    load_multiscale_cache_index,
    load_multiscale_cache_shard,
    make_multiscale_cache_index,
    make_stratified_anchor_plan,
    multiscale_cache_fingerprint,
    save_multiscale_cache_index,
    save_multiscale_cache_shard,
    slice_multiscale_cache_paths,
    validate_anchor_plan,
    validate_multiscale_cache,
    validate_three_way_path_split,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    flux_divergence_torch,
    project_edge_flux_torch,
)
from mnist.experiment12_d0 import Experiment12D0Config


def _dynamics(grid_size: int = 4) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
        num_steps=2,
        max_substeps=2,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        limiter_fraction=1.0,
        mass_floor=1e-7,
        condition_on_source=False,
        flux_parameterization="edge",
        source_lowfreq_size=4,
        ot_lowres_size=4,
    )


def _synthetic_cache(path_count: int = 6) -> D0MultiscaleCache:
    torch.manual_seed(123)
    dynamics = _dynamics()
    n = int(dynamics.grid_size)
    anchors = 2
    strides = torch.tensor([1, 2], dtype=torch.long)
    path_ids = torch.arange(path_count, dtype=torch.long)
    later = torch.full((path_count, anchors, n * n), 1.0 / float(n * n), dtype=torch.float32)
    ends = torch.tensor([[4, 2]], dtype=torch.long).repeat(path_count, 1)
    tau = 1.0 - ends.float() / 4.0
    strata = torch.tensor([[0, 1]], dtype=torch.long).repeat(path_count, 1)
    reverse = 1e-4 * torch.randn(2, path_count, anchors, 2, n, n)
    reverse = project_edge_flux_torch(
        reverse.reshape(-1, 2, n, n), grid_size=n
    ).reshape_as(reverse)
    reference = torch.zeros_like(reverse)
    later_flat = later.reshape(-1, n * n)
    q = (ends - 1).reshape(-1)
    # The anchors end on opposite sides of an outer-step boundary.  A
    # nonconstant schedule makes q=end-1 versus q=end (or start/end reversal)
    # observable in the r=1 compatibility identity.
    rate_schedule = np.asarray([0.5, 2.0], dtype=np.float64)
    reference[0] = exact_reverse_reference_step_transfer(
        later_flat,
        q,
        rate_schedule=rate_schedule,
        reference_substeps=2,
        dt_sub=0.25,
        dynamics_config=dynamics,
    ).reshape(path_count, anchors, 2, n, n)
    earlier = later.unsqueeze(0) + flux_divergence_torch(
        reverse.reshape(-1, 2, n, n)
    ).reshape(2, path_count, anchors, n * n)
    cache = D0MultiscaleCache(
        strides=strides,
        path_ids=path_ids,
        later_states=later,
        tau=tau,
        labels=torch.full((path_count,), 3, dtype=torch.long),
        end_substeps=ends,
        anchor_strata=strata,
        tau_fraction_edges=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        start_images=torch.full((path_count, n * n), 1.0 / float(n * n), dtype=torch.float32),
        earlier_states=earlier,
        reverse_transfers=reverse,
        reference_transfers=reference,
        innovations=torch.zeros_like(reverse),
        masks=torch.ones_like(reverse, dtype=torch.bool),
        terminal_states=np.full((path_count, n, n), 1.0 / float(n * n), dtype=np.float32),
        source_indices=np.arange(path_count, dtype=np.int64),
        requested_labels=np.full(path_count, 3, dtype=np.int64),
        rate_schedule=rate_schedule,
        horizon=1.0,
        dt_sub=0.25,
        sample_steps=2,
        reference_substeps=2,
        lambda_mix=0.35,
        anchor_plan_fingerprint="a" * 64,
        diagnostics={
            "raw_limited_fraction": 0.0,
            "mobility_weighted_limited_fraction": 0.0,
            "noise_energy_weighted_limited_fraction": 0.0,
            "masked_edges": 0,
            "proposed_edges": 100,
            "mobility_weight_sum": 10.0,
            "limited_mobility_weight_sum": 0.0,
            "noise_energy_sum": 20.0,
            "limited_noise_energy_sum": 0.0,
            "nonfinite_edges": 0,
            "floor_touched_pixels": 0,
            "floor_correction_l1": 0.0,
            "renorm_correction_l1": 0.0,
            "path_substep_count": path_count * 4,
        },
    )
    validate_multiscale_cache(cache)
    return cache


def test_stratified_anchor_plan_is_deterministic_balanced_and_valid() -> None:
    kwargs = dict(
        num_paths=64,
        anchors_per_path=16,
        total_substeps=131_072,
        max_stride=1024,
        seed=260718,
    )
    first = make_stratified_anchor_plan(**kwargs)
    second = make_stratified_anchor_plan(**kwargs)
    changed = make_stratified_anchor_plan(**{**kwargs, "seed": 260719})
    validate_anchor_plan(first)
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(first.end_substeps, second.end_substeps)
    assert first.fingerprint != changed.fingerprint
    assert int(first.end_substeps.min()) >= 1024
    assert int(first.end_substeps.max()) <= 131_072
    for row in first.stratum_indices:
        counts = np.bincount(row, minlength=5)
        assert int(counts.max() - counts.min()) <= 1


def test_stratified_anchor_plan_honors_prescribed_per_path_bin_counts() -> None:
    plan = make_stratified_anchor_plan(
        num_paths=64,
        anchors_per_path=32,
        total_substeps=131_072,
        max_stride=1024,
        seed=260718,
        bin_counts=(4, 4, 4, 4, 16),
    )
    validate_anchor_plan(plan)
    np.testing.assert_array_equal(plan.bin_counts, np.asarray([4, 4, 4, 4, 16]))
    for row in plan.stratum_indices:
        np.testing.assert_array_equal(
            np.bincount(row, minlength=5), np.asarray([4, 4, 4, 4, 16])
        )
    assert plan.to_dict()["bin_counts"] == [4, 4, 4, 4, 16]
    with pytest.raises(ValueError, match="sum"):
        make_stratified_anchor_plan(
            num_paths=2,
            anchors_per_path=32,
            total_substeps=131_072,
            max_stride=1024,
            seed=1,
            bin_counts=(4, 4, 4, 4, 15),
        )


def test_three_way_split_is_exact_deterministic_and_path_isolated() -> None:
    paths = np.arange(64, dtype=np.int64)
    split = deterministic_three_way_path_split(paths, seed=17)
    repeat = deterministic_three_way_path_split(paths, seed=17)
    validate_three_way_path_split(split, paths)
    assert split.fingerprint == repeat.fingerprint
    assert split.train_path_ids.size == 40
    assert split.validation_path_ids.size == 12
    assert split.confirmation_path_ids.size == 12
    assert not np.intersect1d(split.train_path_ids, split.validation_path_ids).size
    assert not np.intersect1d(split.train_path_ids, split.confirmation_path_ids).size
    assert not np.intersect1d(split.validation_path_ids, split.confirmation_path_ids).size
    with pytest.raises(ValueError, match="do not cover"):
        deterministic_three_way_path_split(paths, seed=17, train_paths=39)


def test_aligned_block_aggregation_has_exact_sum_sqrt_and_mask_semantics() -> None:
    reverse = torch.arange(4 * 2 * 2 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2, 2, 2)
    reference = 0.25 * reverse
    innovations = -reverse
    masks = torch.ones_like(reverse, dtype=torch.bool)
    masks[-2, 0, 0, 0, 0] = False
    result = aggregate_aligned_block_quantities(
        reverse, reference, innovations, masks, [1, 2, 4]
    )
    torch.testing.assert_close(result["reverse_transfers"][1], reverse[-2:].sum(0))
    torch.testing.assert_close(result["reference_transfers"][2], reference.sum(0))
    torch.testing.assert_close(result["innovations"][1], innovations[-2:].sum(0) / np.sqrt(2.0))
    assert bool(result["masks"][0].all())
    assert not bool(result["masks"][1, 0, 0, 0, 0])
    assert not bool(result["masks"][2, 0, 0, 0, 0])


def test_targets_scales_and_arithmetic_use_exact_stored_reference() -> None:
    cache = _synthetic_cache()
    dynamics = _dynamics()
    target = block_residual_targets(cache, dynamics, stride=1, path_ids=[0, 1])
    expected = project_edge_flux_torch(
        (
            cache.reverse_transfers[0, :2] - cache.reference_transfers[0, :2]
        ).reshape(-1, 2, cache.grid_size, cache.grid_size),
        grid_size=cache.grid_size,
    )
    torch.testing.assert_close(target, expected)
    scales = infer_training_block_scales(cache, dynamics, [0, 1, 2, 3], floor=1e-12)
    expected_scale = float(torch.sqrt(
        block_residual_targets(cache, dynamics, stride=1, path_ids=[0, 1, 2, 3]).double().square().mean()
    ))
    assert scales[1] == pytest.approx(expected_scale, rel=1e-12)
    metrics = block_arithmetic_metrics(cache, dynamics)
    assert metrics["all_finite"] == 1
    assert metrics["r1_reference_max_abs_error"] <= 1e-8
    assert metrics["r1_existing_direct_target_max_abs_error"] <= 1e-8
    assert max(row["replay_l1_max"] for row in metrics["by_stride"]) <= 1e-7
    preflight = evaluate_multiscale_cache_preflight(
        cache,
        dynamics,
        train_path_ids=[0, 1, 2, 3],
        scale_floor=1e-12,
        max_replay_l1=1e-6,
    )
    assert preflight["passed"] == 1
    assert preflight["checks"]["r1_existing_direct_target_identity"]["passed"]
    broken = replace(cache, earlier_states=cache.earlier_states + 1e-2)
    failed = evaluate_multiscale_cache_preflight(
        broken,
        dynamics,
        train_path_ids=[0, 1, 2, 3],
        scale_floor=1e-12,
        max_replay_l1=1e-6,
        max_simplex_mass_error=1.0,
    )
    assert failed["passed"] == 0
    assert not failed["checks"]["block_state_replay"]["passed"]
    unhealthy = replace(
        cache,
        diagnostics={**cache.diagnostics, "raw_limited_fraction": 0.006},
    )
    health_gate = evaluate_multiscale_cache_preflight(
        unhealthy,
        dynamics,
        train_path_ids=[0, 1, 2, 3],
        scale_floor=1e-12,
        max_replay_l1=1e-6,
    )
    assert health_gate["passed"] == 0
    assert not health_gate["checks"]["raw_intervention"]["passed"]


def test_scale_inference_uses_training_paths_only() -> None:
    cache = _synthetic_cache()
    dynamics = _dynamics()
    baseline = infer_training_block_scales(cache, dynamics, [0, 1, 2, 3], floor=1e-12)
    changed_reverse = cache.reverse_transfers.clone()
    changed_reverse[:, 4:] *= 1000.0
    changed = replace(cache, reverse_transfers=changed_reverse)
    same = infer_training_block_scales(changed, dynamics, [0, 1, 2, 3], floor=1e-12)
    assert same == pytest.approx(baseline, rel=0.0, abs=0.0)
    validation_scale = infer_training_block_scales(changed, dynamics, [4, 5], floor=1e-12)
    assert validation_scale[1] > 100.0 * baseline[1]


def test_atomic_shard_roundtrip_index_and_corruption_detection(tmp_path: Path) -> None:
    cache = _synthetic_cache()
    shard0 = slice_multiscale_cache_paths(cache, [0, 1, 2])
    shard1 = slice_multiscale_cache_paths(cache, [3, 4, 5])
    path0 = tmp_path / "shard-00000.npz"
    path1 = tmp_path / "shard-00001.npz"
    record0 = save_multiscale_cache_shard(path0, shard0, shard_id=0)
    record1 = save_multiscale_cache_shard(path1, shard1, shard_id=1)
    loaded0 = load_multiscale_cache_shard(path0, expected_record=record0)
    assert multiscale_cache_fingerprint(loaded0) == record0.cache_fingerprint
    assert loaded0.diagnostics["masked_edges"] == 0
    assert loaded0.diagnostics["proposed_edges"] == 100
    assert loaded0.diagnostics["mobility_weight_sum"] == pytest.approx(10.0)
    assert loaded0.diagnostics["limited_mobility_weight_sum"] == pytest.approx(0.0)
    assert loaded0.diagnostics["noise_energy_sum"] == pytest.approx(20.0)
    assert loaded0.diagnostics["limited_noise_energy_sum"] == pytest.approx(0.0)
    index = make_multiscale_cache_index(
        [record1, record0],
        expected_path_ids=np.arange(6),
        scientific_fingerprint="b" * 64,
        anchor_plan_fingerprint="a" * 64,
        metadata={"scope": "unit fixture"},
    )
    index_path = tmp_path / "index.json"
    save_multiscale_cache_index(index_path, index)
    loaded_index = load_multiscale_cache_index(index_path, verify_shards=True)
    assert [record.shard_id for record in loaded_index.records] == [0, 1]

    with path1.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(D0MultiscaleCompatibilityError, match="file hash"):
        load_multiscale_cache_index(index_path, verify_shards=True)


def test_index_fingerprint_and_path_coverage_fail_closed(tmp_path: Path) -> None:
    cache = _synthetic_cache()
    record = save_multiscale_cache_shard(
        tmp_path / "shard.npz", cache, shard_id=0
    )
    index = make_multiscale_cache_index(
        [record],
        expected_path_ids=np.arange(6),
        scientific_fingerprint="c" * 64,
        anchor_plan_fingerprint="a" * 64,
    )
    path = tmp_path / "index.json"
    save_multiscale_cache_index(path, index)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scientific_fingerprint"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(D0MultiscaleCompatibilityError, match="fingerprint"):
        load_multiscale_cache_index(path, verify_shards=False)
    with pytest.raises(D0MultiscaleCompatibilityError, match="cover"):
        make_multiscale_cache_index(
            [record],
            expected_path_ids=np.arange(7),
            scientific_fingerprint="c" * 64,
            anchor_plan_fingerprint="a" * 64,
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_tiny_prefix_rollout_builder_is_deterministic_and_replays(
    device: str,
) -> None:
    image = np.full((4, 4), 0.3 / 15.0, dtype=np.float32)
    image[1, 1] = 0.7
    images = np.stack([image], axis=0)
    labels = np.asarray([3], dtype=np.int64)
    plan = make_stratified_anchor_plan(
        num_paths=2,
        anchors_per_path=2,
        total_substeps=8,
        max_stride=4,
        seed=91,
        tau_fraction_edges=(0.0, 0.5, 1.0),
    )
    # end=6 with r=4 spans q=2,...,5 and therefore crosses the outer-rate
    # boundary between q=3 and q=4.  This catches both boundary ownership and
    # reference-schedule orientation errors in the vectorized capture path.
    assert 6 in plan.end_substeps
    dynamics = replace(_dynamics(), max_substeps=4)
    d0 = Experiment12D0Config(
        cache_build_mode="substep",
        cache_paths=2,
        time_slices_per_path=2,
        sample_steps=2,
        reference_substeps=4,
        teacher_stride_substeps=1,
        tau_eff=5e-5,
        reference_rate_min=0.5,
        reference_rate_max=2.0,
        lambda_mix=0.35,
        single_image_overfit=True,
        single_image_index=0,
        single_image_label=3,
        d0_target_space="doob-physical-residual",
        physical_sampler_noise_mode="reference",
        eta_l2_weight=0.0,
        invalid_output_l2_weight=0.0,
        curl_loss_weight=0.0,
        edge_laplacian_loss_weight=0.0,
        state_delta_loss_weight=0.0,
        rollout_loss_weight=0.0,
        trajectory_rollout_loss_weight=0.0,
    )
    first = build_multiscale_cache_shard(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=dynamics,
        d0_config=d0,
        anchor_plan=plan,
        strides=[1, 2, 4],
        device=device,
        seed=1234,
        show_progress=False,
        verify_slow_sums=True,
    )
    second = build_multiscale_cache_shard(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=dynamics,
        d0_config=d0,
        anchor_plan=plan,
        strides=[1, 2, 4],
        device=device,
        seed=1234,
        show_progress=False,
        verify_slow_sums=True,
    )
    assert multiscale_cache_fingerprint(first) == multiscale_cache_fingerprint(second)
    metrics = block_arithmetic_metrics(first, dynamics)
    assert metrics["r1_reference_max_abs_error"] <= 1e-8
    assert metrics["r1_existing_direct_target_max_abs_error"] <= 1e-8
    assert max(row["replay_l1_max"] for row in metrics["by_stride"]) <= 1e-5
    assert first.diagnostics["prefix_aggregation"] == 1
    assert first.diagnostics["slow_sum_verified"] == 1
    # Prefix subtraction and the explicit active-window sum contain the same
    # elementary tensors.  Float32 accumulation order can differ by a few ULPs,
    # especially after the innovation's sqrt(stride) normalization.
    assert first.diagnostics["slow_reverse_max_abs_error"] <= 2e-6
    assert first.diagnostics["slow_reference_max_abs_error"] <= 2e-6
    assert first.diagnostics["slow_innovation_max_abs_error"] <= 2e-6
    assert first.diagnostics["slow_target_max_abs_error"] <= 1e-10
    assert first.diagnostics["slow_replay_l1_max"] <= 1e-5
    assert first.diagnostics["slow_mask_mismatch_count"] == 0
    assert first.diagnostics["prefix_scan_mode"] == "outer-cumsum-float64"
    assert first.diagnostics["diagnostic_accumulation"] == "device-float64"

    diagnostics = first.diagnostics
    assert diagnostics["proposed_edges"] == 8 * 2 * 2 * 4 * 4
    assert diagnostics["raw_limited_fraction"] == pytest.approx(
        diagnostics["masked_edges"] / diagnostics["proposed_edges"]
    )
    assert diagnostics["mobility_weighted_limited_fraction"] == pytest.approx(
        diagnostics["limited_mobility_weight_sum"]
        / diagnostics["mobility_weight_sum"]
    )
    assert diagnostics["noise_energy_weighted_limited_fraction"] == pytest.approx(
        diagnostics["limited_noise_energy_sum"] / diagnostics["noise_energy_sum"]
    )
