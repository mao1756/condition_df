from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mnist.d0_multiscale_cache import (
    D0MultiscaleCache,
    deterministic_three_way_path_split,
    make_multiscale_cache_index,
    make_stratified_anchor_plan,
    save_multiscale_cache_index,
    save_multiscale_cache_shard,
)
from mnist.d0_one_image_gate import atomic_write_json, file_fingerprint
from mnist.d0_score_state_cache import (
    D0ScoreStateCompatibilityError,
    FRESH_ORIGIN,
    PARENT_ORIGIN,
    build_fresh_score_state_cache_shard,
    build_fresh_score_state_shards,
    derive_score_state_shard_seed,
    load_score_state_cache_index,
    load_score_state_cache_shard,
    load_score_state_cache_shards,
    make_score_state_anchor_plan,
    make_score_state_cache_index,
    materialize_parent_score_state_shards,
    recover_score_state_shard,
    save_score_state_cache_index,
    save_score_state_cache_shard,
    score_state_cache_fingerprint,
    score_state_cache_from_multiscale,
    slice_score_state_cache_paths,
    validate_score_state_cache,
    verified_score_state_shard_or_none,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig
from mnist.experiment12_d0 import Experiment12D0Config


SCIENCE = "5" * 64


def _dynamics(grid_size: int = 4, *, sample_steps: int = 5, substeps: int = 8) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
        num_steps=sample_steps,
        max_substeps=substeps,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        limiter_fraction=1.0,
        mass_floor=1e-7,
        condition_on_source=False,
        flux_parameterization="edge",
        source_lowfreq_size=grid_size,
        ot_lowres_size=grid_size,
    )


def _d0(*, sample_steps: int = 5, substeps: int = 8) -> Experiment12D0Config:
    return Experiment12D0Config(
        cache_build_mode="substep",
        sample_steps=sample_steps,
        reference_substeps=substeps,
        tau_eff=5e-5,
        lambda_mix=0.35,
        single_image_overfit=True,
        single_image_index=0,
        single_image_label=3,
    )


def _images() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((4, 4), 0.25 / 15.0, dtype=np.float32)
    image[1, 1] = 0.75
    image /= image.sum()
    return image[None, ...], np.asarray([3], dtype=np.int64)


def _parent_kernel() -> dict[str, object]:
    return {
        "grid_size": 4,
        "mass_floor": 1e-7,
        "limiter_fraction": 1.0,
        "edge_alpha_mode": "alpha_eff",
        "edge_alpha_value": 1.0,
        "integrator": "masked_reference_free_step_torch",
        "sample_steps": 2,
        "reference_substeps": 2,
        "lambda_mix": 0.35,
    }


def _multiscale_cache(path_count: int = 6) -> D0MultiscaleCache:
    n = 4
    anchors = 2
    paths = torch.arange(path_count, dtype=torch.long)
    states = torch.full((path_count, anchors, n * n), 1.0 / float(n * n))
    ends = torch.tensor([[4, 2]], dtype=torch.long).repeat(path_count, 1)
    edges = torch.zeros((2, path_count, anchors, 2, n, n), dtype=torch.float32)
    return D0MultiscaleCache(
        strides=torch.tensor([1, 2], dtype=torch.long),
        path_ids=paths,
        later_states=states,
        tau=1.0 - ends.float() / 4.0,
        labels=torch.full((path_count,), 3, dtype=torch.long),
        end_substeps=ends,
        anchor_strata=torch.tensor([[0, 1]], dtype=torch.long).repeat(path_count, 1),
        tau_fraction_edges=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        start_images=torch.full((path_count, n * n), 1.0 / float(n * n)),
        earlier_states=states.unsqueeze(0).repeat(2, 1, 1, 1),
        reverse_transfers=edges.clone(),
        reference_transfers=edges.clone(),
        innovations=edges.clone(),
        masks=torch.ones_like(edges, dtype=torch.bool),
        terminal_states=np.full((path_count, n, n), 1.0 / float(n * n), dtype=np.float32),
        source_indices=np.arange(path_count, dtype=np.int64),
        requested_labels=np.full(path_count, 3, dtype=np.int64),
        rate_schedule=np.asarray([0.5, 1.5], dtype=np.float64),
        horizon=1.0,
        dt_sub=0.25,
        sample_steps=2,
        reference_substeps=2,
        lambda_mix=0.35,
        anchor_plan_fingerprint="a" * 64,
        diagnostics={
            "masked_edges": 2,
            "proposed_edges": 100,
            "mobility_weight_sum": 10.0,
            "limited_mobility_weight_sum": 0.01,
            "noise_energy_sum": 20.0,
            "limited_noise_energy_sum": 0.02,
            "floor_correction_l1": 0.0,
            "renorm_correction_l1": 0.0,
            "floor_touched_pixels": 0,
            "floor_proposed_pixels": path_count * 4 * n * n,
            "nonfinite_edges": 0,
            "path_substep_count": path_count * 4,
        },
    )


def test_score_anchor_plan_has_frozen_counts_and_minimum() -> None:
    plan = make_score_state_anchor_plan(
        path_ids=np.arange(3),
        total_substeps=131_072,
        seed=260752,
    )
    assert plan.anchors_per_path == 32
    assert plan.max_stride == 1024
    assert int(plan.end_substeps.min()) >= 1024
    for row in plan.stratum_indices:
        np.testing.assert_array_equal(np.bincount(row, minlength=5), [4, 4, 4, 4, 16])
    assert derive_score_state_shard_seed(260751, 2, scope="fresh-audit") == derive_score_state_shard_seed(
        260751, 2, scope="fresh-audit"
    )
    assert derive_score_state_shard_seed(260751, 2, scope="fresh-audit") != derive_score_state_shard_seed(
        260751, 3, scope="fresh-audit"
    )


def test_parent_conversion_is_state_only_and_parent_audit_is_forbidden() -> None:
    parent = _multiscale_cache()
    converted = score_state_cache_from_multiscale(
        parent,
        [1, 3],
        role="train",
        scientific_fingerprint=SCIENCE,
        kernel_metadata=_parent_kernel(),
    )
    validate_score_state_cache(converted)
    assert converted.origin == PARENT_ORIGIN
    assert converted.role == "train"
    torch.testing.assert_close(converted.states, parent.later_states[[1, 3]])
    np.testing.assert_array_equal(converted.origin_path_ids, [1, 3])
    assert not hasattr(converted, "reverse_transfers")
    with pytest.raises(D0ScoreStateCompatibilityError, match="forbids audit"):
        score_state_cache_from_multiscale(
            parent,
            [0],
            role="audit",
            scientific_fingerprint=SCIENCE,
            kernel_metadata=_parent_kernel(),
        )


def test_deterministic_shard_roundtrip_index_and_corruption_recovery(tmp_path: Path) -> None:
    cache = score_state_cache_from_multiscale(
        _multiscale_cache(),
        [0, 1, 2, 3],
        role="selection",
        scientific_fingerprint=SCIENCE,
        kernel_metadata=_parent_kernel(),
    )
    shard0 = slice_score_state_cache_paths(cache, [0, 1])
    shard1 = slice_score_state_cache_paths(cache, [2, 3])
    path0 = tmp_path / "parent-selection-shard-00000.npz"
    path1 = tmp_path / "parent-selection-shard-00001.npz"
    record0 = save_score_state_cache_shard(path0, shard0, shard_id=0)
    first_bytes = path0.read_bytes()
    repeat = save_score_state_cache_shard(path0, shard0, shard_id=0)
    assert path0.read_bytes() == first_bytes
    assert repeat.file_sha256 == record0.file_sha256
    record1 = save_score_state_cache_shard(path1, shard1, shard_id=1)
    index = make_score_state_cache_index(
        [record1, record0],
        expected_path_ids=[0, 1, 2, 3],
        scientific_fingerprint=SCIENCE,
    )
    index_path = tmp_path / "cache_index.json"
    save_score_state_cache_index(index_path, index)
    loaded = load_score_state_cache_index(index_path, verify_shards=True)
    assert [record.shard_id for record in loaded.records] == [0, 1]
    assert score_state_cache_fingerprint(load_score_state_cache_shard(path0)) == record0.cache_fingerprint

    path1.write_bytes(b"incomplete")
    assert verified_score_state_shard_or_none(path1, expected_record=record1) is None
    repaired, rebuilt = recover_score_state_shard(
        path1, shard1, shard_id=1, expected_record=record1
    )
    assert rebuilt
    assert load_score_state_cache_shard(path1, expected_record=repaired).path_count == 2

    changed = replace(shard0, scientific_fingerprint="6" * 64)
    with pytest.raises(D0ScoreStateCompatibilityError, match="different experiment"):
        recover_score_state_shard(path0, changed, shard_id=0)


def test_parent_materialization_verifies_roles_and_excludes_audit(tmp_path: Path) -> None:
    parent_root = tmp_path / "parent"
    cache_root = parent_root / "cache"
    cache_root.mkdir(parents=True)
    parent = _multiscale_cache()
    record0 = save_multiscale_cache_shard(
        cache_root / "shard-00000.npz", replace(parent, path_ids=parent.path_ids[:3], later_states=parent.later_states[:3], tau=parent.tau[:3], labels=parent.labels[:3], end_substeps=parent.end_substeps[:3], anchor_strata=parent.anchor_strata[:3], start_images=parent.start_images[:3], earlier_states=parent.earlier_states[:, :3], reverse_transfers=parent.reverse_transfers[:, :3], reference_transfers=parent.reference_transfers[:, :3], innovations=parent.innovations[:, :3], masks=parent.masks[:, :3], terminal_states=parent.terminal_states[:3], source_indices=parent.source_indices[:3], requested_labels=parent.requested_labels[:3]), shard_id=0
    )
    second = replace(
        parent,
        path_ids=parent.path_ids[3:],
        later_states=parent.later_states[3:],
        tau=parent.tau[3:],
        labels=parent.labels[3:],
        end_substeps=parent.end_substeps[3:],
        anchor_strata=parent.anchor_strata[3:],
        start_images=parent.start_images[3:],
        earlier_states=parent.earlier_states[:, 3:],
        reverse_transfers=parent.reverse_transfers[:, 3:],
        reference_transfers=parent.reference_transfers[:, 3:],
        innovations=parent.innovations[:, 3:],
        masks=parent.masks[:, 3:],
        terminal_states=parent.terminal_states[3:],
        source_indices=parent.source_indices[3:],
        requested_labels=parent.requested_labels[3:],
    )
    record1 = save_multiscale_cache_shard(cache_root / "shard-00001.npz", second, shard_id=1)
    parent_index = make_multiscale_cache_index(
        [record0, record1],
        expected_path_ids=np.arange(6),
        scientific_fingerprint="7" * 64,
        anchor_plan_fingerprint="a" * 64,
    )
    parent_index_path = cache_root / "cache_index.json"
    save_multiscale_cache_index(parent_index_path, parent_index)
    split = deterministic_three_way_path_split(
        np.arange(6), seed=91, train_paths=2, validation_paths=2, confirmation_paths=2
    )
    split_payload = {
        **split.to_dict(),
        "selection_path_ids": split.validation_path_ids.tolist(),
        "audit_path_ids": split.confirmation_path_ids.tolist(),
    }
    split_path = parent_root / "path_split.json"
    atomic_write_json(split_path, split_payload)

    output = tmp_path / "score-parent"
    index = materialize_parent_score_state_shards(
        parent_index_path,
        split_path,
        output,
        scientific_fingerprint=SCIENCE,
        shard_paths=2,
        kernel_metadata=_parent_kernel(),
        enforce_frozen_kernel=False,
    )
    expected = set(split.train_path_ids.tolist() + split.validation_path_ids.tolist())
    assert set(index.expected_path_ids) == expected
    assert not expected.intersection(set(split.confirmation_path_ids.tolist()))
    assert index.metadata["parent_audit_paths_excluded"] == 1
    _, shards = load_score_state_cache_shards(output / "cache_index.json")
    assert {cache.role for cache in shards} == {"train", "selection"}
    assert {cache.origin for cache in shards} == {PARENT_ORIGIN}
    with pytest.raises(D0ScoreStateCompatibilityError, match="audit roles"):
        materialize_parent_score_state_shards(
            parent_index_path,
            split_path,
            tmp_path / "bad",
            scientific_fingerprint=SCIENCE,
            roles=("audit",),
            kernel_metadata=_parent_kernel(),
            enforce_frozen_kernel=False,
        )


def test_fresh_builder_captures_common_anchors_and_is_seed_deterministic() -> None:
    images, labels = _images()
    dynamics = _dynamics(sample_steps=2, substeps=4)
    d0 = _d0(sample_steps=2, substeps=4)
    plan = make_stratified_anchor_plan(
        path_ids=[100, 101],
        anchors_per_path=2,
        total_substeps=8,
        max_stride=2,
        seed=13,
        tau_fraction_edges=(0.0, 0.5, 1.0),
        bin_counts=(1, 1),
    )
    kwargs = dict(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=dynamics,
        d0_config=d0,
        anchor_plan=plan,
        path_ids=[100, 101],
        role="audit",
        device="cpu",
        seed=12345,
        scientific_fingerprint=SCIENCE,
        enforce_frozen_kernel=False,
        show_progress=False,
    )
    first = build_fresh_score_state_cache_shard(**kwargs)
    second = build_fresh_score_state_cache_shard(**kwargs)
    validate_score_state_cache(first)
    assert first.origin == FRESH_ORIGIN
    assert first.minimum_forward_substep == 2
    assert score_state_cache_fingerprint(first) == score_state_cache_fingerprint(second)
    torch.testing.assert_close(first.states, second.states, rtol=0.0, atol=0.0)
    assert first.diagnostics["nonfinite_edges"] == 0
    assert first.diagnostics["state_min"] > 0.0
    assert first.diagnostics["max_simplex_mass_error"] <= 2e-6


def test_fresh_shard_resume_skips_good_and_recovers_corrupt(tmp_path: Path) -> None:
    images, labels = _images()
    dynamics = _dynamics()
    d0 = _d0()
    output = tmp_path / "fresh"
    kwargs = dict(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=dynamics,
        d0_config=d0,
        output_dir=output,
        path_ids=[200, 201],
        role="audit",
        device="cpu",
        seed=260751,
        anchor_seed=260752,
        scientific_fingerprint=SCIENCE,
        anchors_per_path=5,
        bin_counts=(1, 1, 1, 1, 1),
        minimum_forward_substep=2,
        shard_paths=1,
        enforce_frozen_kernel=False,
        show_progress=False,
    )
    first = build_fresh_score_state_shards(**kwargs)
    good_path = output / first.records[0].filename
    bad_path = output / first.records[1].filename
    good_hash = file_fingerprint(good_path)
    expected_bad_hash = first.records[1].file_sha256
    bad_path.write_bytes(b"partial shard")
    resumed = build_fresh_score_state_shards(**kwargs)
    assert file_fingerprint(good_path) == good_hash
    assert file_fingerprint(bad_path) == expected_bad_hash
    assert resumed.fingerprint == first.fingerprint
    load_score_state_cache_index(output / "cache_index.json", verify_shards=True)

    changed = {**kwargs, "scientific_fingerprint": "9" * 64}
    with pytest.raises(D0ScoreStateCompatibilityError, match="scientific fingerprint"):
        build_fresh_score_state_shards(**changed)

