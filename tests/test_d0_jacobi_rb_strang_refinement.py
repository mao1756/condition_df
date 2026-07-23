from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_multipath import run_exact_multipath_shard
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    FINEST_SAMPLE_STEPS,
    MAX_REFINEMENT_PATHS_PER_GROUP,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    REFINEMENT_SHARD_STEPS,
    SUPPORTED_SAMPLE_STEPS,
    canonical_refinement_transition_ids,
    evaluate_refinement_observables,
    exact_dirichlet_observable_moments,
    finest_tick_for_step,
    legacy_k512_transition_ids,
    refinement_observable_spec,
    refinement_phase_exposure,
    run_refinement_shard,
)


class _RecordingSampler:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        x: torch.Tensor,
        exposure: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        ids = kwargs["transition_ids"]
        assert isinstance(ids, torch.Tensor)
        self.calls.append(
            {
                "x": x.detach().clone(),
                "exposure": exposure.detach().clone(),
                "ids": ids.detach().clone(),
                "rng_key": kwargs["rng_key"],
            }
        )
        # This stand-in is lane-local and depends on both production inputs.
        # It therefore catches ID, exposure, ordering, and resume changes.
        jitter = (
            torch.remainder(ids.to(torch.int64), 31).to(torch.float64) * 2.0**-46
            + exposure * 2.0**-48
        )
        later = torch.clamp(x + jitter, 0.0, 1.0)
        count = int(x.numel())
        zero_i64 = torch.zeros((), dtype=torch.int64, device=x.device)
        zero_f64 = torch.zeros((), dtype=torch.float64, device=x.device)
        return SimpleNamespace(
            later_head_fraction=later,
            denoising_target=later - x,
            certificate_codes=torch.full(
                (count,), 15, dtype=torch.uint8, device=x.device
            ),
            fallback_mask=torch.zeros(count, dtype=torch.bool, device=x.device),
            strengthened_mask=torch.zeros(
                count, dtype=torch.bool, device=x.device
            ),
            mode_counts=torch.full(
                (count,), 128, dtype=torch.int32, device=x.device
            ),
            prefix_bits=torch.full(
                (count,), 64, dtype=torch.int32, device=x.device
            ),
            arb_fallback_reason_codes=torch.zeros(
                count, dtype=torch.uint8, device=x.device
            ),
            diagnostics={
                "maximum_cuda_launch_lanes": torch.as_tensor(
                    count, dtype=torch.int64, device=x.device
                ),
                "fused_authorizer_launch_count": torch.ones(
                    (), dtype=torch.int64, device=x.device
                ),
                "arb_fallback_elapsed_seconds": zero_f64,
                "fused_authorizer_elapsed_seconds": zero_f64,
                "candidate_elapsed_seconds": zero_f64,
                "resource_cap_count": zero_i64,
                "invalid_density_count": zero_i64,
                "approximation_count": zero_i64,
                "correction_count": zero_i64,
                "floor_count": zero_i64,
                "limiter_count": zero_i64,
                "renormalization_count": zero_i64,
                "nonfinite_count": zero_i64,
            },
        )


def _states(path_count: int) -> torch.Tensor:
    rng = np.random.Generator(np.random.Philox(261151))
    values = rng.dirichlet(np.ones(28 * 28), size=path_count)
    return torch.as_tensor(values, dtype=torch.float64).contiguous()


def test_frozen_levels_phase_plan_and_shard_contract() -> None:
    assert SUPPORTED_SAMPLE_STEPS == (128, 256, 512, 1024, 2048)
    assert FINEST_SAMPLE_STEPS == 2048
    assert REFINEMENT_SHARD_STEPS == 8
    assert MAX_REFINEMENT_PATHS_PER_GROUP == 8
    assert PHASE_MATCHINGS == (0, 1, 2, 3, 2, 1, 0)
    assert PHASE_DURATIONS == (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)


@pytest.mark.parametrize("sample_steps", SUPPORTED_SAMPLE_STEPS)
def test_exact_state_dependent_exposure_formula(sample_steps: int) -> None:
    pair_total = torch.as_tensor([0.25, 2.0 / 784.0, 0.0], dtype=torch.float64)
    observed = refinement_phase_exposure(
        pair_total, sample_steps=sample_steps, duration_fraction=0.5
    )
    numerator = 3.0 * (5.0e-5 / sample_steps) * 0.5 / (1.0 / 28.0) ** 2
    expected = torch.as_tensor(
        [numerator / 0.25, numerator / (2.0 / 784.0), 0.0],
        dtype=torch.float64,
    )
    assert torch.equal(observed, expected)


def test_finest_tick_ids_have_only_prescribed_cross_level_aliasing() -> None:
    device = torch.device("cpu")
    assert finest_tick_for_step(128, 0) == 15
    assert finest_tick_for_step(2048, 2047) == 2047
    aligned = [
        canonical_refinement_transition_ids(
            (7,),
            sample_steps=level,
            outer_step=level // 128 - 1,
            phase=3,
            device=device,
        )
        for level in SUPPORTED_SAMPLE_STEPS
    ]
    assert all(torch.equal(aligned[0], item) for item in aligned[1:])
    unaligned = canonical_refinement_transition_ids(
        (7,),
        sample_steps=2048,
        outer_step=0,
        phase=3,
        device=device,
    )
    assert not torch.equal(aligned[0], unaligned)

    ids = canonical_refinement_transition_ids(
        (9, 2, 17),
        sample_steps=2048,
        outer_step=2047,
        phase=6,
        device=device,
    )
    assert ids.dtype == torch.uint64
    assert ids.is_contiguous()
    assert ids.numel() == 3 * EDGES_PER_PHASE
    assert torch.unique(ids).numel() == ids.numel()
    permuted = canonical_refinement_transition_ids(
        (17, 9, 2),
        sample_steps=2048,
        outer_step=2047,
        phase=6,
        device=device,
    ).reshape(3, EDGES_PER_PHASE)
    original = ids.reshape(3, EDGES_PER_PHASE)
    assert torch.equal(permuted[0], original[2])
    assert torch.equal(permuted[1], original[0])
    assert torch.equal(permuted[2], original[1])


def test_legacy_k512_id_hook_is_exact_and_rejects_other_levels() -> None:
    device = torch.device("cpu")
    observed = legacy_k512_transition_ids(
        (3, 8),
        sample_steps=512,
        outer_step=17,
        phase=4,
        device=device,
    )
    from mnist.d0_jacobi_rb_cuda_multipath import (
        canonical_same_phase_transition_ids,
    )

    expected = canonical_same_phase_transition_ids(
        (3, 8), outer_step=17, phase=4, device=device
    )
    assert torch.equal(observed, expected)
    with pytest.raises(ValueError, match="only for K=512"):
        legacy_k512_transition_ids(
            (3,),
            sample_steps=1024,
            outer_step=17,
            phase=4,
            device=device,
        )


def test_observable_spec_has_exact_dirichlet_moments_and_unit_standardization() -> None:
    moments = exact_dirichlet_observable_moments()
    assert len(moments) == 10
    assert [item.family for item in moments].count("linear") == 8
    assert moments[0].mean == 0.0
    assert moments[0].variance == pytest.approx(1.0 / (2.0 * 785.0))
    assert moments[8].mean == pytest.approx(2.0 / 785.0)
    assert moments[9].mean == pytest.approx(6.0 / (785.0 * 786.0))
    assert all(item.variance > 0.0 for item in moments)

    spec = refinement_observable_spec()
    assert spec.names[-2:] == ("quadratic_mass", "cubic_mass")
    assert spec.families == (
        "linear",
        "linear",
        "linear",
        "linear",
        "linear",
        "linear",
        "linear",
        "linear",
        "quadratic",
        "cubic",
    )
    assert np.max(np.abs(spec.fourier_weights.sum(axis=1))) <= 1.0e-13
    assert np.allclose(
        np.sum(spec.fourier_weights**2, axis=1), 784.0 / 2.0, rtol=2e-14
    )

    uniform = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    raw = evaluate_refinement_observables(
        uniform, spec=spec, standardized=False
    )
    standardized = evaluate_refinement_observables(uniform, spec=spec)
    assert isinstance(raw, np.ndarray)
    assert isinstance(standardized, np.ndarray)
    assert np.max(np.abs(raw[0, :8])) <= 2.0e-16
    assert raw[0, 8] == pytest.approx(1.0 / 784.0)
    assert raw[0, 9] == pytest.approx(1.0 / 784.0**2)
    np.testing.assert_allclose(
        standardized,
        (raw - spec.means) / spec.standard_deviations,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("sample_steps", SUPPORTED_SAMPLE_STEPS)
def test_variable_k_shard_is_certified_conservative_and_records_observables(
    sample_steps: int,
) -> None:
    sampler = _RecordingSampler()
    result = run_refinement_shard(
        _states(2),
        path_ids=(4, 19),
        sample_steps=sample_steps,
        start_step=0,
        root_seed=261151,
        panel_namespace=f"unit-k{sample_steps}",
        profile=JacobiRBCudaProfile(),
        sampler=sampler,
        checkpoint_steps=(8, sample_steps),
        capture_phase_state_trace=True,
    )
    diagnostics = result.diagnostics
    expected_count = 2 * 8 * 7 * 392
    assert len(sampler.calls) == 8 * 7
    assert diagnostics["transition_count"] == expected_count
    assert diagnostics["certified_count"] == expected_count
    assert diagnostics["uncertified_count"] == 0
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["certificate_code_counts"] == {"15": expected_count}
    assert diagnostics["mode_count_counts"] == {"128": expected_count}
    assert diagnostics["prefix_bit_counts"] == {"64": expected_count}
    assert diagnostics["arb_fallback_reason_code_counts"] == {
        "0": expected_count
    }
    assert diagnostics["maximum_backend_call_size"] == 2 * 392
    assert diagnostics["maximum_global_simplex_error"] <= 2.0e-12
    assert diagnostics["in_shard_host_roundtrip_count"] == 0
    assert diagnostics["diagnostics_device_resident_until_commit"] == 1
    assert len(result.phase_state_records) == 8 * 7
    assert len(result.path_records) == 2
    assert result.path_records[0].certificate_code_counts == {
        "15": 8 * 7 * 392
    }
    assert result.observable_checkpoints[0].completed_step == 8
    # The horizon endpoint is outside this first shard at every supported K.
    assert len(result.observable_checkpoints) == 1
    assert result.observable_checkpoints[0].values.shape == (2, 10)
    assert not result.committed_final_states.flags.writeable
    assert result.to_record()["diagnostics"]["sample_steps"] == sample_steps
    assert sampler.calls[0]["rng_key"] == (
        261151,
        "jacobi-rb-strang-common-quantile-v1",
        f"unit-k{sample_steps}",
    )


def test_k512_legacy_injection_replays_parent_phase_states_and_outputs() -> None:
    states = _states(2)
    paths = (5, 12)
    parent = run_exact_multipath_shard(
        states,
        path_ids=paths,
        start_step=16,
        root_seed=261151,
        profile=JacobiRBCudaProfile(),
        group_sizes=(2,),
        sampler=_RecordingSampler(),
        capture_phase_state_trace=True,
    )
    replay = run_refinement_shard(
        states,
        path_ids=paths,
        sample_steps=512,
        start_step=16,
        root_seed=261151,
        panel_namespace="legacy-parent-replay",
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
        transition_id_provider=legacy_k512_transition_ids,
        rng_key_override=(261151, "full-path-v2"),
        capture_phase_state_trace=True,
    )
    assert torch.equal(parent.final_states, replay.final_states)
    assert parent.batch_output_sha256 == replay.batch_output_sha256
    assert parent.batch_final_state_sha256 == replay.batch_final_state_sha256
    assert [item.to_dict() for item in parent.phase_state_records] == [
        item.to_dict() for item in replay.phase_state_records
    ]
    parent_by_path = {item.path_id: item for item in parent.path_records}
    replay_by_path = {item.path_id: item for item in replay.path_records}
    assert {
        path: (
            item.input_state_sha256,
            item.output_sha256,
            item.final_state_sha256,
        )
        for path, item in parent_by_path.items()
    } == {
        path: (
            item.input_state_sha256,
            item.output_sha256,
            item.final_state_sha256,
        )
        for path, item in replay_by_path.items()
    }


def test_path_order_chunk_resume_and_panel_namespaces_are_stateless() -> None:
    states = _states(3)
    paths = (7, 2, 11)
    whole_a_sampler = _RecordingSampler()
    first = run_refinement_shard(
        states,
        path_ids=paths,
        sample_steps=128,
        start_step=0,
        root_seed=261151,
        panel_namespace="resume-a",
        profile=JacobiRBCudaProfile(),
        sampler=whole_a_sampler,
    )
    second = run_refinement_shard(
        first.final_states,
        path_ids=paths,
        sample_steps=128,
        start_step=8,
        root_seed=261151,
        panel_namespace="resume-a",
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
    )
    repeated_first = run_refinement_shard(
        states,
        path_ids=paths,
        sample_steps=128,
        start_step=0,
        root_seed=261151,
        panel_namespace="resume-a",
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
    )
    repeated_second = run_refinement_shard(
        repeated_first.final_states,
        path_ids=paths,
        sample_steps=128,
        start_step=8,
        root_seed=261151,
        panel_namespace="resume-a",
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
    )
    assert torch.equal(second.final_states, repeated_second.final_states)
    assert second.batch_output_sha256 == repeated_second.batch_output_sha256

    permutation = torch.as_tensor((2, 0, 1), dtype=torch.int64)
    permuted = run_refinement_shard(
        states.index_select(0, permutation).contiguous(),
        path_ids=tuple(paths[index] for index in permutation.tolist()),
        sample_steps=128,
        start_step=0,
        root_seed=261151,
        panel_namespace="resume-a",
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
    )
    assert first.batch_output_sha256 == permuted.batch_output_sha256
    assert first.batch_final_state_sha256 == permuted.batch_final_state_sha256

    other_panel_sampler = _RecordingSampler()
    run_refinement_shard(
        states,
        path_ids=paths,
        sample_steps=128,
        start_step=0,
        root_seed=261151,
        panel_namespace="resume-b",
        profile=JacobiRBCudaProfile(),
        sampler=other_panel_sampler,
    )
    assert torch.equal(
        whole_a_sampler.calls[0]["ids"], other_panel_sampler.calls[0]["ids"]
    )
    assert whole_a_sampler.calls[0]["rng_key"] != other_panel_sampler.calls[0]["rng_key"]


def test_invalid_level_shard_and_path_contracts_fail_closed() -> None:
    common = {
        "root_seed": 261151,
        "panel_namespace": "invalid",
        "profile": JacobiRBCudaProfile(),
        "sampler": _RecordingSampler(),
    }
    with pytest.raises(ValueError, match="sample_steps"):
        run_refinement_shard(
            _states(1), path_ids=(0,), sample_steps=64, start_step=0, **common
        )
    with pytest.raises(ValueError, match="eight-step"):
        run_refinement_shard(
            _states(1), path_ids=(0,), sample_steps=128, start_step=1, **common
        )
    with pytest.raises(ValueError, match="at most eight"):
        run_refinement_shard(
            _states(9),
            path_ids=tuple(range(9)),
            sample_steps=128,
            start_step=0,
            **common,
        )
    bad_mass = _states(1) * 0.5
    with pytest.raises(ValueError, match="unit simplex"):
        run_refinement_shard(
            bad_mass, path_ids=(0,), sample_steps=128, start_step=0, **common
        )
