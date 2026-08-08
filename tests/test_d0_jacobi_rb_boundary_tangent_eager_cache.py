from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_boundary_tangent_schedule import (
    PROJECTED_BASE_TRANSITIONS,
    PROJECTED_MIDPOINT_TRANSITIONS,
    PROJECTED_TOTAL_TRANSITIONS,
)
import mnist.d0_jacobi_rb_boundary_tangent_eager_cache as cache


def _source() -> np.ndarray:
    return np.full(784, 1.0 / 784.0, dtype=np.float64)


def _generate_mixed_cohort(run_dir: Path, *, shard_runner=None):
    return cache.generate_eager_cache(
        run_dir,
        _source(),
        device="cpu",
        outer_steps=16,
        cohort_indices=(6,),
        shard_runner=shard_runner or cache.deterministic_test_shard_runner,
        branch_runner=cache.deterministic_test_branch_runner,
    )


def test_frozen_plans_and_confirmation_stream_share_eager_execution() -> None:
    assert cache.frozen_eager_cache_plan()["semantic_sha256"] == (
        "84236a075f93f20342a3427dc72f6f4b757b4aba2ef8f81d809b2ffebe20250c"
    )
    assert cache.eager_execution_contract()["semantic_sha256"] == (
        "9de28564b3926712282522faa765dcdccfd3016c3644e3003358b03cc17cce83"
    )
    train_validation = cache.frozen_cache_cohorts("train_validation")
    confirmation = cache.frozen_cache_cohorts("confirmation")
    assert [len(value.path_ids) for value in train_validation] == [10] * 9 + [6]
    assert [len(value.path_ids) for value in confirmation] == [10] * 6 + [4]
    assert train_validation[6].path_roles == ("train",) * 4 + ("validation",) * 6
    assert set(confirmation[-1].path_roles) == {"confirmation"}

    injected = []

    def shard_runner(states, **kwargs):
        injected.append(("base", kwargs["sampler"], kwargs["profile"]))
        return cache.deterministic_test_shard_runner(states, **kwargs)

    def branch_runner(states, **kwargs):
        injected.append(("branch", kwargs["sampler"], kwargs["profile"]))
        return cache.deterministic_test_branch_runner(states, **kwargs)

    executions = list(
        cache.iter_eager_shards(
            _source(),
            cohort_kind="confirmation",
            device="cpu",
            outer_steps=16,
            cohort_indices=(6,),
            shard_runner=shard_runner,
            branch_runner=branch_runner,
        )
    )
    assert [value.identity.start_step for value in executions] == [0, 8]
    assert [len(value.branches) for value in executions] == [0, 7]
    assert all(value.path_roles == ("confirmation",) * 4 for value in executions)
    assert len([value for value in injected if value[0] == "base"]) == 2
    assert len([value for value in injected if value[0] == "branch"]) == 7
    assert len({value[1] for value in injected}) == 1
    assert all(value[2].certificate_effort == "strengthened" for value in injected)

    aggregate = cache.EagerDiagnosticsAccumulator(
        "confirmation", outer_steps=16, cohort_indices=(6,)
    )
    for execution in executions:
        aggregate.add(execution)
    record = aggregate.to_record()
    assert record["base_transition_count"] == 4 * 16 * 7 * 392
    assert record["midpoint_transition_count"] == 4 * 7 * 8 * 392
    assert record["transition_count"] == 263_424
    assert record["branch_row_count"] == 4 * 7 * 8
    assert record["raw_label_persistence"] == 0


def test_explicit_cohort_entry_points_are_hash_bound_and_role_split(
    tmp_path: Path,
) -> None:
    cohorts = (
        cache.EagerCohort(
            kind="train_validation",
            index=0,
            path_ids=(0xF1000, 0xF1001, 0xF1100),
            path_roles=("train", "train", "validation"),
        ),
    )
    plan = cache.explicit_eager_cache_plan(cohorts)
    contract = cache.eager_execution_contract_for_cohorts(
        cohorts=cohorts,
        cohort_plan_sha256=plan["semantic_sha256"],
        outer_steps=16,
        shard_runner=cache.deterministic_test_shard_runner,
        branch_runner=cache.deterministic_test_branch_runner,
    )
    assert contract["cohort_plan_sha256"] == plan["semantic_sha256"]
    with pytest.raises(cache.EagerCacheError, match="fingerprint"):
        cache.eager_execution_contract_for_cohorts(
            cohorts=cohorts,
            cohort_plan_sha256="0" * 64,
            outer_steps=16,
        )

    result = cache.generate_eager_cache_for_cohorts(
        tmp_path,
        _source(),
        cohorts=cohorts,
        cohort_plan_sha256=plan["semantic_sha256"],
        device="cpu",
        outer_steps=16,
        shard_runner=cache.deterministic_test_shard_runner,
        branch_runner=cache.deterministic_test_branch_runner,
    )
    assert result["metrics"]["path_ids"] == [0xF1000, 0xF1001, 0xF1100]
    assert result["role_indexes"]["train"]["path_ids"] == [0xF1000, 0xF1001]
    assert result["role_indexes"]["validation"]["path_ids"] == [0xF1100]
    assert result["metrics"]["role_branch_row_counts"] == {
        "train": 2 * 7 * 8,
        "validation": 7 * 8,
    }


def test_mixed_cohort_is_physically_split_and_loaders_keep_label_firewall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _generate_mixed_cohort(tmp_path)
    assert result["recomputed_shard_count"] == 2
    assert set(result["role_indexes"]) == {"train", "validation"}
    assert result["metrics"]["transition_count"] == 10 * (16 + 8) * 7 * 392

    metadata_path = (
        tmp_path
        / "eager_cache"
        / "train_validation"
        / "cohort-006"
        / "shard-000008"
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["committed"] == 1
    assert metadata["cross_role_artifact_commit"] == 0
    assert set(metadata["role_artifacts"]) == {"train", "validation"}
    for role, expected_count in (("train", 4), ("validation", 6)):
        artifacts = metadata["role_artifacts"][role]
        assert artifacts["path_count"] == expected_count
        assert all(
            f"/{role}/" in "/" + artifacts[name]["path"]
            for name in ("continuation_state", "branch_inputs", "branch_labels")
        )

    opened: list[str] = []
    original_load = cache._load_npz

    def tracked_load(path: Path):
        opened.append(path.name)
        if path.name == "branch_labels.npz":
            raise AssertionError("input loader opened raw labels")
        return original_load(path)

    monkeypatch.setattr(cache, "_load_npz", tracked_load)
    train_inputs, train_index = cache.load_eager_role_inputs(tmp_path, "train")
    validation_inputs, validation_index = cache.load_eager_role_inputs(
        tmp_path, "validation"
    )
    assert "branch_labels.npz" not in opened
    assert train_inputs["later_full_state"].shape == (4 * 7 * 8, 784)
    assert validation_inputs["later_full_state"].shape == (6 * 7 * 8, 784)
    assert train_inputs["later_full_state"].dtype == np.float32
    assert train_index["input_row_count"] == train_index["label_row_count"] == 224
    assert validation_index["input_row_count"] == validation_index["label_row_count"] == 336

    monkeypatch.setattr(cache, "_load_npz", original_load)
    train_labels, _ = cache.load_eager_role_labels(tmp_path, "train")
    validation_labels, _ = cache.load_eager_role_labels(tmp_path, "validation")
    assert np.array_equal(train_inputs["sample_key"], train_labels["sample_key"])
    assert np.array_equal(
        validation_inputs["sample_key"], validation_labels["sample_key"]
    )
    assert train_labels["denoising_target"].shape == (4 * 7 * 8, 392)
    assert validation_labels["certificate_codes"].shape == (6 * 7 * 8, 392)

    train_paths, train_states = cache.load_eager_role_final_states(tmp_path, "train")
    validation_paths, validation_states = cache.load_eager_role_final_states(
        tmp_path, "validation"
    )
    assert train_paths.shape == (4,)
    assert validation_paths.shape == (6,)
    assert train_states.shape == (4, 784)
    assert validation_states.shape == (6, 784)
    assert not set(train_paths.tolist()).intersection(validation_paths.tolist())


def test_invalid_atomic_shard_recomputes_that_shard_and_its_cohort_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts: list[int] = []

    def counted_runner(states, **kwargs):
        starts.append(int(kwargs["start_step"]))
        return cache.deterministic_test_shard_runner(states, **kwargs)

    first = _generate_mixed_cohort(tmp_path, shard_runner=counted_runner)
    assert first["recomputed_shard_count"] == 2
    assert starts == [0, 8]

    metadata = json.loads(
        (
            tmp_path
            / "eager_cache/train_validation/cohort-006/shard-000000/metadata.json"
        ).read_text(encoding="utf-8")
    )
    state_path = tmp_path / metadata["role_artifacts"]["train"][
        "continuation_state"
    ]["path"]
    state_path.write_bytes(state_path.read_bytes() + b"corrupt")

    second = _generate_mixed_cohort(tmp_path, shard_runner=counted_runner)
    assert second["recomputed_shard_count"] == 2
    assert second["reused_shard_count"] == 0
    assert starts == [0, 8, 0, 8]

    original_load = cache._load_npz

    def resume_load(path: Path):
        if path.name == "branch_labels.npz":
            raise AssertionError("resume decoded raw labels")
        return original_load(path)

    monkeypatch.setattr(cache, "_load_npz", resume_load)
    third = _generate_mixed_cohort(tmp_path, shard_runner=counted_runner)
    assert third["recomputed_shard_count"] == 0
    assert third["reused_shard_count"] == 2
    assert starts == [0, 8, 0, 8]


def test_executor_rejects_divergent_device_and_committed_continuation_states() -> None:
    source = np.arange(1, 785, dtype=np.float64)
    source /= source.sum()
    cohort = cache.frozen_cache_cohorts("train_validation")[0]
    states = torch.as_tensor(source).reshape(1, -1).repeat(10, 1).contiguous()

    def divergent_runner(states, **kwargs):
        result = cache.deterministic_test_shard_runner(states, **kwargs)
        divergent = np.roll(result.committed_final_states, shift=1, axis=1).copy(
            order="C"
        )
        return replace(result, committed_final_states=divergent)

    with pytest.raises(
        cache.EagerCacheError,
        match="device and committed continuation states differ",
    ):
        cache.execute_eager_shard(
            states,
            cohort=cohort,
            start_step=0,
            shard_runner=divergent_runner,
            branch_runner=cache.deterministic_test_branch_runner,
        )


def test_resume_rehydrates_the_same_nonuniform_committed_chain(tmp_path: Path) -> None:
    source = np.arange(1, 785, dtype=np.float64)
    source /= source.sum()
    kwargs = {
        "device": "cpu",
        "outer_steps": 16,
        "cohort_indices": (6,),
        "shard_runner": cache.deterministic_test_shard_runner,
        "branch_runner": cache.deterministic_test_branch_runner,
    }

    first = cache.generate_eager_cache(tmp_path, source, **kwargs)
    first_states = {
        role: cache.load_eager_role_final_states(tmp_path, role)
        for role in ("train", "validation")
    }
    second = cache.generate_eager_cache(tmp_path, source, **kwargs)
    second_states = {
        role: cache.load_eager_role_final_states(tmp_path, role)
        for role in ("train", "validation")
    }

    assert first["recomputed_shard_count"] == 2
    assert first["reused_shard_count"] == 0
    assert second["recomputed_shard_count"] == 0
    assert second["reused_shard_count"] == 2
    for role in ("train", "validation"):
        first_paths, first_final = first_states[role]
        second_paths, second_final = second_states[role]
        assert np.array_equal(first_paths, second_paths)
        assert np.array_equal(first_final, second_final)


def test_confirmation_execution_cannot_be_persisted(tmp_path: Path) -> None:
    execution = list(
        cache.iter_eager_shards(
            _source(),
            cohort_kind="confirmation",
            device="cpu",
            outer_steps=16,
            cohort_indices=(6,),
            shard_runner=cache.deterministic_test_shard_runner,
            branch_runner=cache.deterministic_test_branch_runner,
        )
    )[-1]
    with pytest.raises(cache.EagerCacheError, match="streaming-only"):
        cache.persist_eager_shard(
            tmp_path, execution, execution_contract_sha256="0" * 64
        )
    assert not (tmp_path / "eager_cache").exists()


def test_executor_integrates_real_base_and_fused_schedulers_on_cpu() -> None:
    class ExactSampler:
        def __call__(self, head, exposure, **kwargs):
            del exposure
            count = int(head.numel())
            device = head.device
            zeros_i64 = torch.zeros((), dtype=torch.int64, device=device)
            zeros_f64 = torch.zeros((), dtype=torch.float64, device=device)
            return SimpleNamespace(
                later_head_fraction=head.clone(),
                denoising_target=torch.zeros_like(head),
                certificate_codes=torch.full(
                    (count,), 15, dtype=torch.uint8, device=device
                ),
                fallback_mask=torch.zeros(count, dtype=torch.bool, device=device),
                strengthened_mask=torch.ones(count, dtype=torch.bool, device=device),
                arb_fallback_reason_codes=torch.zeros(
                    count, dtype=torch.uint8, device=device
                ),
                mode_counts=torch.full(
                    (count,), 128, dtype=torch.int32, device=device
                ),
                prefix_bits=torch.full(
                    (count,), 128, dtype=torch.int32, device=device
                ),
                diagnostics={
                    "arb_fallback_elapsed_seconds": zeros_f64,
                    "fused_authorizer_elapsed_seconds": zeros_f64,
                    "candidate_elapsed_seconds": zeros_f64,
                    "maximum_cuda_launch_lanes": torch.as_tensor(
                        count, dtype=torch.int64, device=device
                    ),
                    "fused_authorizer_launch_count": torch.ones(
                        (), dtype=torch.int64, device=device
                    ),
                    **{
                        name: zeros_i64
                        for name in (
                            "resource_cap_count",
                            "invalid_density_count",
                            "approximation_count",
                            "correction_count",
                            "floor_count",
                            "limiter_count",
                            "renormalization_count",
                            "nonfinite_count",
                        )
                    },
                },
            )

    cohort = cache.frozen_cache_cohorts("confirmation")[-1]
    states = torch.as_tensor(_source()).reshape(1, -1).repeat(4, 1).contiguous()
    execution = cache.execute_eager_shard(
        states,
        cohort=cohort,
        start_step=8,
        sampler=ExactSampler(),
    )
    assert execution.selected_step == 15
    assert len(execution.branches) == 7
    assert execution.diagnostics["base_transition_count"] == 4 * 8 * 7 * 392
    assert execution.diagnostics["midpoint_transition_count"] == 4 * 7 * 8 * 392
    assert execution.diagnostics["certified_count"] == execution.diagnostics[
        "transition_count"
    ]
    assert execution.diagnostics["maximum_mass_error"] <= 2.0e-12


def test_combined_full_plan_counts_match_frozen_projection() -> None:
    def aggregate(kind: str, paths: int) -> dict[str, object]:
        base = paths * 512 * 7 * 392
        midpoint = paths * 32 * 7 * 8 * 392
        return {
            "cohort_kind": kind,
            "complete_frozen_cohort_plan": 1,
            "outer_steps": 512,
            "selected_outer_steps": list(range(15, 512, 16)),
            "base_transition_count": base,
            "midpoint_transition_count": midpoint,
            "certified_count": base + midpoint,
            "fallback_count": 0,
            "fallback_elapsed_seconds": 0.0,
            "complete_pipeline_elapsed_seconds": 1.0,
            "maximum_mass_error": 0.0,
            "maximum_launch_lanes": 4096,
            "maximum_peak_memory_fraction": 0.5,
            "persisted_bytes": 0,
            "forbidden_counts": {},
        }

    combined = cache.combine_eager_metrics(
        aggregate("train_validation", 96), aggregate("confirmation", 64)
    )
    assert combined["base_transition_count"] == PROJECTED_BASE_TRANSITIONS
    assert combined["midpoint_transition_count"] == PROJECTED_MIDPOINT_TRANSITIONS
    assert combined["transition_count"] == PROJECTED_TOTAL_TRANSITIONS
    assert combined["full_production_projection"] == 1
    assert combined["exact_projected_counts_passed"] == 1
