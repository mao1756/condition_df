from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest
import torch

import mnist.d0_jacobi_rb_haar_scheduler as haar_scheduler
import mnist.d0_jacobi_rb_haar_controls as haar_controls
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_dynkin import DynkinAccumulatorState
from mnist.d0_jacobi_rb_haar import (
    HaarCouplingProfile,
    unpack_haar_transition_id,
)
from mnist.d0_jacobi_rb_haar_scheduler import (
    HaarSchedulerError,
    HaarShardIdentity,
    NestedHaarSchedule,
    PairwiseHaarAntitheticSchedule,
    canonical_haar_scheduler_transition_ids,
    commit_haar_shard,
    exact_pairwise_fine_observable_mean,
    initialize_antithetic_branch_states,
    initialize_nested_branch_states,
    inspect_haar_backend_contract,
    load_committed_haar_shard,
    require_production_haar_backend,
    run_nested_haar_shard,
    run_pairwise_haar_antithetic_shard,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    evaluate_refinement_observables,
    refinement_observable_spec,
)


def _hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _states(paths: int = 1) -> torch.Tensor:
    return torch.full((paths, 784), 1.0 / 784.0, dtype=torch.float64)


def _parity_fixture():
    initial = _states(2)
    final = initial.clone()
    final[0, 0] = torch.nextafter(
        final[0, 0], torch.tensor(float("inf"), dtype=torch.float64)
    )
    final[0, 1] -= final[0, 0] - initial[0, 0]
    raw_before = evaluate_refinement_observables(
        initial, standardized=False
    )
    raw_after = evaluate_refinement_observables(final, standardized=False)
    assert isinstance(raw_before, torch.Tensor)
    assert isinstance(raw_after, torch.Tensor)
    drift = torch.zeros((2, 10), dtype=torch.float64)
    drift_radius = torch.full((2, 10), 1.0e-16, dtype=torch.float64)
    scales = torch.as_tensor(
        refinement_observable_spec(28).standard_deviations,
        dtype=torch.float64,
    )
    residual = (raw_after - raw_before - drift) / scales

    def result(
        selected: slice,
        *,
        before_delta: float = 0.0,
        after_delta: float = 0.0,
        drift_delta: float = 0.0,
        residual_delta: float = 0.0,
    ):
        before = raw_before[selected].clone()
        after = raw_after[selected].clone()
        selected_drift = drift[selected].clone()
        selected_residual = residual[selected].clone()
        before[0, 0] += before_delta
        after[0, 0] += after_delta
        selected_drift[0, 0] += drift_delta
        selected_residual[0, 0] += residual_delta
        return SimpleNamespace(
            final_states=final[selected].clone(),
            raw_before_values=before,
            raw_after_values=after,
            drift_center=selected_drift,
            drift_error_radius=drift_radius[selected].clone(),
            standardized_residual=selected_residual,
        )

    combined = result(
        slice(0, 2),
        before_delta=1.0e-16,
        after_delta=5.0e-16,
        drift_delta=1.0e-19,
        residual_delta=2.0e-14,
    )
    singletons = [result(slice(0, 1)), result(slice(1, 2))]
    return initial, combined, singletons


def _profile() -> JacobiRBCudaProfile:
    return JacobiRBCudaProfile()


def _fast_runner(
    states: torch.Tensor,
    *,
    path_ids,
    sample_steps,
    start_step,
    sampler,
    checkpoint_steps,
    accumulator_state,
    **kwargs,
):
    del kwargs
    # The scheduler separately tests sampler integration below.  This double
    # isolates level/subshard orchestration and exact restart bookkeeping.
    sampler.call_count = 8 * 7
    delta = (
        float(sample_steps) * 1.0e-10
        + float(start_step + 8) * 1.0e-12
        + float(sampler.detail_sign) * 1.0e-13
    )
    final = states.detach().clone()
    final[:, 0] += delta
    final[:, 1] -= delta
    raw = final[:, :10].detach().cpu().numpy()
    dynkin = raw + float(sample_steps) * 1.0e-9
    radius = np.full_like(raw, 1.0e-14)
    if accumulator_state is None:
        center = torch.zeros(
            (len(path_ids), 10), dtype=torch.float64, device=states.device
        )
        compensation = torch.zeros_like(center)
        error_radius = torch.zeros_like(center)
    else:
        center = accumulator_state.center.detach().clone()
        compensation = accumulator_state.compensation.detach().clone()
        error_radius = accumulator_state.error_radius.detach().clone()
    center = center + torch.as_tensor(
        dynkin, dtype=torch.float64, device=states.device
    )
    error_radius = error_radius + 1.0e-14
    accumulator = DynkinAccumulatorState(
        center=center.contiguous(),
        compensation=compensation.contiguous(),
        error_radius=error_radius.contiguous(),
    )
    committed_center = center.detach().cpu().numpy()
    committed_compensation = compensation.detach().cpu().numpy()
    committed_radius = error_radius.detach().cpu().numpy()
    final_host = final.detach().cpu().numpy()
    checkpoint = SimpleNamespace(
        completed_step=checkpoint_steps[-1],
        raw_values=raw,
        dynkin_values=dynkin,
        dynkin_error_radius=radius,
    )
    token = json.dumps(
        {
            "steps": sample_steps,
            "start": start_step,
            "sign": sampler.detail_sign,
            "state": _hash(final_host),
        },
        sort_keys=True,
    ).encode()
    output_hash = hashlib.sha256(b"out" + token).hexdigest()
    state_hash = hashlib.sha256(b"state" + token).hexdigest()
    return SimpleNamespace(
        final_states=final.contiguous(),
        accumulator_state=accumulator,
        committed_final_states=final_host,
        committed_accumulator_center=committed_center,
        committed_accumulator_compensation=committed_compensation,
        committed_accumulator_error_radius=committed_radius,
        observable_checkpoints=(checkpoint,),
        batch_output_sha256=output_hash,
        batch_final_state_sha256=state_hash,
        diagnostics={"transition_count": len(path_ids) * 8 * 7 * 392},
    )


@dataclass(frozen=True)
class _FakeUniformBatch:
    uniform_lower: torch.Tensor
    uniform_upper: torch.Tensor
    shape: tuple[int, ...]
    diagnostics: dict
    runtime_report: dict


def _fake_uniform_builder(**kwargs):
    paths = len(tuple(kwargs["path_ids"]))
    edges = len(tuple(kwargs["edge_ids"]))
    device = kwargs["device"]
    lower = torch.full(
        (paths, edges), 0.49, dtype=torch.float64, device=device
    )
    upper = torch.full(
        (paths, edges), 0.51, dtype=torch.float64, device=device
    )
    return _FakeUniformBatch(
        uniform_lower=lower,
        uniform_upper=upper,
        shape=(paths, edges),
        diagnostics={"arb_fallback_count": 0},
        runtime_report={
            "fused_cuda_authorizer_available": True,
            "arb_fallback_fraction": 0.0,
            "arb_fallback_time_fraction": 0.0,
        },
    )


_fake_uniform_builder.haar_backend_contract = {
    "fused_cuda_normal_authorizer": True,
    "normal_cuda_authorizing": True,
    "normal_fallback_fraction_upper_bound": 0.0,
    "normal_fallback_time_fraction_upper_bound": 0.0,
    "source": "test-double",
}


def _fake_interval_authorizer(
    x,
    exposure,
    lower,
    upper,
    *,
    transition_ids,
    refinement_callback,
    profile,
):
    del lower, upper, refinement_callback, profile
    shape = x.shape
    zeros_bool = torch.zeros(shape, dtype=torch.bool, device=x.device)
    zeros_i32 = torch.zeros(shape, dtype=torch.int32, device=x.device)
    zeros_u8 = torch.zeros(shape, dtype=torch.uint8, device=x.device)
    return SimpleNamespace(
        later_head_fraction=x.detach().clone(),
        denoising_target=torch.zeros_like(x),
        certificate_codes=torch.full(
            shape, 15, dtype=torch.uint8, device=x.device
        ),
        fallback_mask=zeros_bool,
        strengthened_mask=zeros_bool,
        mode_counts=zeros_i32,
        prefix_bits=torch.full(
            shape, 64, dtype=torch.int32, device=x.device
        ),
        arb_fallback_reason_codes=zeros_u8,
        transition_ids=transition_ids,
        diagnostics={},
        runtime_report={"arbitrary_uniform_interval_authorizing": True},
    )


_fake_interval_authorizer.haar_interval_authorizer_contract = {
    "arbitrary_uniform_jacobi_authorizer": True
}


def _exercising_runner(*args, sampler, transition_id_provider, **kwargs):
    states = args[0]
    path_ids = kwargs["path_ids"]
    sample_steps = kwargs["sample_steps"]
    start_step = kwargs["start_step"]
    profile = kwargs["profile"]
    for local_step in range(8):
        for phase in range(7):
            x = torch.full(
                (len(path_ids) * 392,),
                0.5,
                dtype=torch.float64,
                device=states.device,
            )
            exposure = torch.full_like(x, 0.01)
            ids = transition_id_provider(
                path_ids,
                sample_steps=sample_steps,
                outer_step=start_step + local_step,
                phase=phase,
                device=states.device,
            )
            sampler(
                x,
                exposure,
                rng_key=("ignored",),
                transition_ids=ids,
                profile=profile,
            )
    return _fast_runner(*args, sampler=sampler, **kwargs)


def test_installed_fused_backend_advertises_production_contract():
    report = inspect_haar_backend_contract()
    assert report.production_ready
    assert report.source == "d0_jacobi_rb_haar_fused"
    assert report.normal_fallback_fraction_upper_bound <= 1.0e-4
    assert require_production_haar_backend().production_ready


def test_custom_backend_contract_can_be_verified():
    contract = inspect_haar_backend_contract(
        uniform_builder=_fake_uniform_builder,
        interval_authorizer=_fake_interval_authorizer,
    )
    assert contract.production_ready
    assert contract.arbitrary_uniform_jacobi_authorizer


def test_one_phase_batching_parity_distinguishes_state_from_reductions():
    initial, combined, singletons = _parity_fixture()
    report = haar_controls._compare_one_phase_batching_results(
        combined=combined,
        singletons=singletons,
        initial_states=initial,
    )
    assert report["passed"] == 1
    assert report["exact_state_equality_pass"] == 1
    assert report["exact_drift_error_radius_equality_pass"] == 1
    assert report["derived_observable_agreement_pass"] == 1
    assert (
        report["derived_observables"]["standardized_residual"][
            "maximum_absolute_difference"
        ]
        > 0.0
    )
    assert all(
        record["maximum_bound_fraction"] <= 1.0
        for record in report["derived_observables"].values()
    )


@pytest.mark.parametrize(
    "field,perturbation",
    [
        ("raw_before_values", 1.0e-8),
        ("raw_after_values", 1.0e-8),
        ("drift_center", 1.0e-8),
        ("standardized_residual", 1.0e-8),
    ],
)
def test_one_phase_batching_parity_fails_closed_beyond_numerical_envelope(
    field,
    perturbation,
):
    initial, combined, singletons = _parity_fixture()
    changed = getattr(combined, field).clone()
    changed[0, 0] += perturbation
    combined = SimpleNamespace(**{**vars(combined), field: changed})
    report = haar_controls._compare_one_phase_batching_results(
        combined=combined,
        singletons=singletons,
        initial_states=initial,
    )
    assert report["passed"] == 0
    assert report["derived_observables"][field]["passed"] == 0


def test_one_phase_batching_parity_requires_exact_state_and_radius():
    initial, combined, singletons = _parity_fixture()
    changed_state = combined.final_states.clone()
    changed_state[0, 0] = torch.nextafter(
        changed_state[0, 0],
        torch.tensor(float("inf"), dtype=torch.float64),
    )
    state_report = haar_controls._compare_one_phase_batching_results(
        combined=SimpleNamespace(
            **{**vars(combined), "final_states": changed_state}
        ),
        singletons=singletons,
        initial_states=initial,
    )
    assert state_report["passed"] == 0
    assert state_report["exact_state_equality_pass"] == 0

    changed_radius = combined.drift_error_radius.clone()
    changed_radius[0, 0] = torch.nextafter(
        changed_radius[0, 0],
        torch.tensor(float("inf"), dtype=torch.float64),
    )
    radius_report = haar_controls._compare_one_phase_batching_results(
        combined=SimpleNamespace(
            **{**vars(combined), "drift_error_radius": changed_radius}
        ),
        singletons=singletons,
        initial_states=initial,
    )
    assert radius_report["passed"] == 0
    assert radius_report["exact_drift_error_radius_equality_pass"] == 0


def test_nested_schedule_executes_aligned_subshards_and_records_observables():
    schedule = NestedHaarSchedule(pool="reference", role="nested_a")
    identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=(0xA0000,),
        coarsest_start_step=0,
        root_seed=261181,
        panel_namespace="test-reference",
    )
    initial = _states()
    result = run_nested_haar_shard(
        initialize_nested_branch_states(initial, schedule),
        identity=identity,
        jacobi_profile=_profile(),
        level_runner=_fast_runner,
        production_authorizing=False,
        nonauthorizing_test_only=True,
    )
    assert tuple(result.branches) == ("k1024", "k2048", "k512")
    assert len(result.branches["k512"].completed_steps) == 1
    assert len(result.branches["k1024"].completed_steps) == 2
    assert len(result.branches["k2048"].completed_steps) == 4
    for branch in result.branches.values():
        assert branch.raw_observables.shape[1:] == (1, 10)
        assert branch.dynkin_observables.shape == branch.raw_observables.shape
    assert result.diagnostics["raw_observables_recorded"] == 1
    assert result.diagnostics["dynkin_observables_recorded"] == 1


@pytest.mark.parametrize(
    ("coarse", "fine"),
    ((128, 256), (256, 512), (512, 1024), (1024, 2048)),
)
def test_antithetic_uses_pair_local_tree_and_one_coarse(coarse, fine):
    schedule = PairwiseHaarAntitheticSchedule(
        coarse_steps=coarse, fine_steps=fine, role="antithetic_a"
    )
    identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=(0xB0000,),
        coarsest_start_step=0,
        root_seed=261181,
        panel_namespace=f"pair-{coarse}",
    )
    states = initialize_antithetic_branch_states(_states())
    result = run_pairwise_haar_antithetic_shard(
        coarse_state=states["coarse"],
        fine_plus_state=states["fine_plus"],
        fine_minus_state=states["fine_minus"],
        identity=identity,
        jacobi_profile=_profile(),
        level_runner=_fast_runner,
        production_authorizing=False,
        nonauthorizing_test_only=True,
    )
    assert result.diagnostics["pair_local_tree"] == 1
    assert result.diagnostics["coarse_executed_once"] == 1
    assert len(result.branches["coarse"].shard_results) == 1
    assert len(result.branches["fine_plus"].shard_results) == 2
    assert len(result.branches["fine_minus"].shard_results) == 2
    expected = 0.5 * (
        result.branches["fine_plus"].raw_observables
        + result.branches["fine_minus"].raw_observables
    )
    np.testing.assert_array_equal(
        exact_pairwise_fine_observable_mean(result), expected
    )


@pytest.mark.parametrize(
    ("coarse", "fine"),
    ((128, 256), (256, 512), (512, 1024), (1024, 2048)),
)
def test_all_pairwise_structural_ids_bind_the_pair_local_root(coarse, fine):
    path_id = 0xB0000
    for level, arm in ((coarse, 1), (fine, 1), (fine, -1)):
        ids = canonical_haar_scheduler_transition_ids(
            role="antithetic_a",
            path_ids=(path_id,),
            sample_steps=level,
            outer_step=level - 1,
            phase=6,
            detail_sign=arm,
            tree_root_steps=coarse,
            device="cpu",
        )
        assert ids.shape == (392,)
        assert torch.unique(ids).numel() == 392
        first = unpack_haar_transition_id(int(ids[0].item()))
        last = unpack_haar_transition_id(int(ids[-1].item()))
        assert first["tree_root_steps"] == coarse
        assert first["sample_steps"] == level
        assert first["arm"] == arm
        assert first["edge_id"] == 0
        assert last["edge_id"] == 391


def test_sampler_hook_consumes_hierarchical_uniforms_and_jacobi_authorizer():
    schedule = PairwiseHaarAntitheticSchedule(
        coarse_steps=128, fine_steps=256, role="antithetic_a"
    )
    identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=(0xB0000,),
        coarsest_start_step=0,
        root_seed=261181,
        panel_namespace="sampler-hook",
    )
    states = initialize_antithetic_branch_states(_states())
    result = run_pairwise_haar_antithetic_shard(
        coarse_state=states["coarse"],
        fine_plus_state=states["fine_plus"],
        fine_minus_state=states["fine_minus"],
        identity=identity,
        jacobi_profile=_profile(),
        level_runner=_exercising_runner,
        uniform_builder=_fake_uniform_builder,
        interval_authorizer=_fake_interval_authorizer,
        production_authorizing=False,
        nonauthorizing_test_only=True,
    )
    assert result.diagnostics["normal_sample_count"] > 0
    assert result.diagnostics["normal_fallback_count"] == 0
    assert result.diagnostics["normal_transform_seconds"] >= 0.0
    assert result.diagnostics["jacobi_authorizer_seconds"] >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_interval_adapter_flattens_certificate_tensors_path_major():
    device = torch.device("cuda")
    target = torch.empty((8 * 392,), dtype=torch.float64, device=device)
    logical = torch.arange(
        target.numel(), dtype=torch.float64, device=device
    ).reshape(392, 8).transpose(0, 1)
    assert logical.shape == (8, 392)
    assert not logical.is_contiguous()

    for name in (
        "uniform_center_hi",
        "uniform_center_lo",
        "uniform_radius",
    ):
        normalized = haar_scheduler._normalize_interval_adapter_tensor(
            logical,
            name=name,
            target=target,
        )
        assert normalized is not None
        assert normalized.shape == target.shape
        assert normalized.dtype == logical.dtype
        assert normalized.device == target.device
        assert normalized.is_contiguous()
        torch.testing.assert_close(
            normalized, logical.reshape(-1), rtol=0.0, atol=0.0
        )

    prefix = torch.full(
        (8, 392), 128, dtype=torch.int64, device=device
    ).transpose(0, 1).contiguous().transpose(0, 1)
    assert not prefix.is_contiguous()
    normalized_prefix = haar_scheduler._normalize_interval_adapter_tensor(
        prefix,
        name="prefix_bits",
        target=target,
        prefix_bits=True,
    )
    assert normalized_prefix is not None
    assert normalized_prefix.shape == target.shape
    assert normalized_prefix.dtype == torch.int64
    assert normalized_prefix.is_contiguous()
    torch.testing.assert_close(
        normalized_prefix, prefix.reshape(-1), rtol=0.0, atol=0.0
    )

    already_flat = logical.reshape(-1).contiguous()
    normalized_flat = haar_scheduler._normalize_interval_adapter_tensor(
        already_flat,
        name="uniform_center_hi",
        target=target,
    )
    torch.testing.assert_close(
        normalized_flat, already_flat, rtol=0.0, atol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("kind", "expected_fragment"),
    (
        ("count", "element count"),
        ("device", "common CUDA device"),
        ("dtype", "int32 or int64"),
        ("low", "[1,1024]"),
        ("high", "[1,1024]"),
    ),
)
def test_interval_adapter_rejects_invalid_certificate_contract(
    kind, expected_fragment
):
    device = torch.device("cuda")
    target = torch.empty((16,), dtype=torch.float64, device=device)
    if kind == "count":
        value = torch.ones((15,), dtype=torch.int32, device=device)
    elif kind == "device":
        value = torch.ones((16,), dtype=torch.int32)
    elif kind == "dtype":
        value = torch.ones((16,), dtype=torch.float64, device=device)
    elif kind == "low":
        value = torch.zeros((16,), dtype=torch.int32, device=device)
    else:
        value = torch.full(
            (16,), 1025, dtype=torch.int64, device=device
        )

    with pytest.raises(HaarSchedulerError) as error:
        haar_scheduler._normalize_interval_adapter_tensor(
            value,
            name="prefix_bits",
            target=target,
            prefix_bits=True,
        )
    assert (
        error.value.failure_code
        == "hierarchical_interval_adapter_shape_invalid"
    )
    assert error.value.failure_domain == "scheduler_execution"
    assert expected_fragment in str(error.value)


def _certificate_uniform_builder(*, flattened: bool):
    def builder(**kwargs):
        paths = len(tuple(kwargs["path_ids"]))
        edges = len(tuple(kwargs["edge_ids"]))
        device = kwargs["device"]
        shape = (paths * edges,) if flattened else (paths, edges)
        lower = torch.full(
            (paths, edges), 0.49, dtype=torch.float64, device=device
        )
        upper = torch.full(
            (paths, edges), 0.51, dtype=torch.float64, device=device
        )
        center_hi = torch.full(
            shape, 0.5, dtype=torch.float64, device=device
        )
        center_lo = torch.zeros(
            shape, dtype=torch.float64, device=device
        )
        radius = torch.full(
            shape, 0.01, dtype=torch.float64, device=device
        )
        prefix = torch.full(
            shape, 128, dtype=torch.int32, device=device
        )
        return SimpleNamespace(
            uniform_lower=lower,
            uniform_upper=upper,
            uniform_center_hi=center_hi,
            uniform_center_lo=center_lo,
            uniform_radius=radius,
            prefix_bits=prefix,
            shape=(paths, edges),
            diagnostics={"arb_fallback_count": 0},
            runtime_report={
                "fused_cuda_authorizer_available": True,
                "device_resident_certified_output": 1,
                "arb_fallback_fraction": 0.0,
                "arb_fallback_time_fraction": 0.0,
            },
        )

    return builder


def _single_call_sampler(*, builder, authorizer, device):
    paths = tuple(range(0xA0000, 0xA0008))
    sampler = haar_scheduler._CertifiedHaarSampler(
        root_seed=261181,
        role="nested_a",
        path_ids=paths,
        sample_steps=128,
        start_step=0,
        detail_sign=1,
        pair_coarse_steps=None,
        haar_profile=HaarCouplingProfile(),
        jacobi_profile=_profile(),
        uniform_builder=builder,
        interval_authorizer=authorizer,
        enforce_runtime_contract=False,
    )
    x = torch.full(
        (len(paths) * 392,), 0.5, dtype=torch.float64, device=device
    )
    exposure = torch.full_like(x, 0.05)
    transition_ids = sampler._ids(
        sample_steps=128,
        outer_step=0,
        phase=0,
        device=torch.device(device),
    )
    return sampler(
        x,
        exposure,
        rng_key=("ignored",),
        transition_ids=transition_ids,
        profile=_profile(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sampler_adapter_passes_flat_contiguous_certificates_to_authorizer(
    monkeypatch,
):
    captured: list[tuple[torch.Tensor, ...]] = []

    def authorizer(
        x,
        exposure,
        lower,
        upper,
        *,
        transition_ids,
        refinement_callback,
        profile,
        uniform_center_hi,
        uniform_center_lo,
        uniform_radius,
        source_prefix_bits,
    ):
        del (
            lower,
            upper,
            refinement_callback,
            profile,
            transition_ids,
        )
        captured.append(
            (
                uniform_center_hi.clone(),
                uniform_center_lo.clone(),
                uniform_radius.clone(),
                source_prefix_bits.clone(),
            )
        )
        for value in captured[-1]:
            assert value.shape == x.shape
            assert value.device == x.device
            assert value.is_contiguous()
        return SimpleNamespace(
            later_head_fraction=x.clone(),
            denoising_target=torch.zeros_like(x),
            certificate_codes=torch.full_like(x, 15, dtype=torch.uint8),
            runtime_report={},
        )

    monkeypatch.setattr(
        haar_scheduler,
        "sample_alpha1_rb_transition_batch_cuda_from_uniform_cells",
        authorizer,
    )
    shaped = _single_call_sampler(
        builder=_certificate_uniform_builder(flattened=False),
        authorizer=authorizer,
        device="cuda",
    )
    flat = _single_call_sampler(
        builder=_certificate_uniform_builder(flattened=True),
        authorizer=authorizer,
        device="cuda",
    )
    torch.testing.assert_close(
        shaped.later_head_fraction,
        flat.later_head_fraction,
        rtol=0.0,
        atol=0.0,
    )
    assert len(captured) == 2
    for shaped_value, flat_value in zip(
        captured[0], captured[1], strict=True
    ):
        torch.testing.assert_close(
            shaped_value, flat_value, rtol=0.0, atol=0.0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("exception_type", (TypeError, ValueError))
def test_installed_authorizer_input_errors_become_scheduler_contract_failures(
    monkeypatch, exception_type
):
    def authorizer(*args, **kwargs):
        del args, kwargs
        raise exception_type("invalid authorizer input fixture")

    monkeypatch.setattr(
        haar_scheduler,
        "sample_alpha1_rb_transition_batch_cuda_from_uniform_cells",
        authorizer,
    )
    with pytest.raises(HaarSchedulerError) as error:
        _single_call_sampler(
            builder=_certificate_uniform_builder(flattened=False),
            authorizer=authorizer,
            device="cuda",
        )
    assert (
        error.value.failure_code
        == "hierarchical_interval_adapter_shape_invalid"
    )
    assert error.value.failure_domain == "scheduler_execution"
    assert error.value.__cause__.__class__ is exception_type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_installed_authorizer_preserves_typed_scheduler_failures(monkeypatch):
    sentinel = HaarSchedulerError(
        "typed numerical fixture",
        failure_code="typed_fixture",
        failure_domain="jacobi_authorizer",
    )

    def authorizer(*args, **kwargs):
        del args, kwargs
        raise sentinel

    monkeypatch.setattr(
        haar_scheduler,
        "sample_alpha1_rb_transition_batch_cuda_from_uniform_cells",
        authorizer,
    )
    with pytest.raises(HaarSchedulerError) as error:
        _single_call_sampler(
            builder=_certificate_uniform_builder(flattened=False),
            authorizer=authorizer,
            device="cuda",
        )
    assert error.value is sentinel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_cuda_adapter_completes_eight_path_eight_step_shard():
    """Regression for the production ``[8,392] -> [3136]`` boundary."""

    paths = tuple(range(0xA0000, 0xA0008))
    schedule = NestedHaarSchedule(pool="main", role="nested_a")
    identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=paths,
        coarsest_start_step=0,
        root_seed=261181,
        panel_namespace="real-adapter-regression",
    )
    result = haar_scheduler._run_branch(
        branch="k128",
        states=_states(len(paths)).to(device="cuda").contiguous(),
        accumulator_state=None,
        sample_steps=128,
        coarsest_steps=128,
        coarsest_start_step=0,
        detail_sign=1,
        pair_coarse_steps=None,
        identity=identity,
        haar_profile=HaarCouplingProfile(),
        jacobi_profile=_profile(),
        level_runner=_exercising_runner,
        uniform_builder=haar_scheduler.build_certified_haar_uniform_batch,
        interval_authorizer=(
            haar_scheduler
            .sample_alpha1_rb_transition_batch_cuda_from_uniform_cells
        ),
        enforce_runtime_contract=False,
    )
    assert result.completed_steps == (8,)
    assert result.raw_observables.shape == (1, len(paths), 10)
    assert result.diagnostics["phase_count"] == 8 * 7
    assert (
        result.diagnostics["sampler_reports"][0]["haar_sampler_call_count"]
        == 8 * 7
    )
    assert result.diagnostics["normal_sample_count"] == (
        len(paths) * 392 * 8 * 7
    )


def test_atomic_commit_load_and_exact_continuation(tmp_path: Path):
    schedule = PairwiseHaarAntitheticSchedule(
        coarse_steps=128, fine_steps=256, role="antithetic_a"
    )
    first_identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=(0xB0000,),
        coarsest_start_step=0,
        root_seed=261181,
        panel_namespace="resume",
    )
    initial = initialize_antithetic_branch_states(_states())
    first = run_pairwise_haar_antithetic_shard(
        coarse_state=initial["coarse"],
        fine_plus_state=initial["fine_plus"],
        fine_minus_state=initial["fine_minus"],
        identity=first_identity,
        jacobi_profile=_profile(),
        level_runner=_fast_runner,
        production_authorizing=False,
        nonauthorizing_test_only=True,
    )
    metadata = commit_haar_shard(first, tmp_path)
    assert metadata["timing"]["state_shard_io_seconds"] >= 0.0
    assert (
        metadata["timing"][
            "complete_pipeline_including_state_shard_io_seconds"
        ]
        >= metadata["timing"]["complete_pipeline_before_file_commit_seconds"]
    )
    loaded = load_committed_haar_shard(
        tmp_path, expected_identity=first_identity, device="cpu"
    )

    second_identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=(0xB0000,),
        coarsest_start_step=8,
        root_seed=261181,
        panel_namespace="resume",
    )
    resumed = run_pairwise_haar_antithetic_shard(
        coarse_state=loaded.states["coarse"],
        fine_plus_state=loaded.states["fine_plus"],
        fine_minus_state=loaded.states["fine_minus"],
        coarse_accumulator=loaded.accumulators["coarse"],
        fine_plus_accumulator=loaded.accumulators["fine_plus"],
        fine_minus_accumulator=loaded.accumulators["fine_minus"],
        identity=second_identity,
        jacobi_profile=_profile(),
        level_runner=_fast_runner,
        production_authorizing=False,
        nonauthorizing_test_only=True,
    )
    direct = run_pairwise_haar_antithetic_shard(
        coarse_state=first.branches["coarse"].final_states,
        fine_plus_state=first.branches["fine_plus"].final_states,
        fine_minus_state=first.branches["fine_minus"].final_states,
        coarse_accumulator=first.branches["coarse"].accumulator_state,
        fine_plus_accumulator=first.branches["fine_plus"].accumulator_state,
        fine_minus_accumulator=first.branches["fine_minus"].accumulator_state,
        identity=second_identity,
        jacobi_profile=_profile(),
        level_runner=_fast_runner,
        production_authorizing=False,
        nonauthorizing_test_only=True,
    )
    assert resumed.output_sha256 == direct.output_sha256
    for branch in resumed.branches:
        np.testing.assert_array_equal(
            resumed.branches[branch].committed_final_states,
            direct.branches[branch].committed_final_states,
        )


def test_corrupt_archive_and_identity_mismatch_fail_closed(tmp_path: Path):
    schedule = PairwiseHaarAntitheticSchedule(
        coarse_steps=128, fine_steps=256, role="antithetic_b"
    )
    identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=(0xB1000,),
        coarsest_start_step=0,
        root_seed=261181,
        panel_namespace="tamper",
    )
    initial = initialize_antithetic_branch_states(_states())
    result = run_pairwise_haar_antithetic_shard(
        coarse_state=initial["coarse"],
        fine_plus_state=initial["fine_plus"],
        fine_minus_state=initial["fine_minus"],
        identity=identity,
        jacobi_profile=_profile(),
        level_runner=_fast_runner,
        production_authorizing=False,
        nonauthorizing_test_only=True,
    )
    metadata = commit_haar_shard(result, tmp_path)
    archive = tmp_path / metadata["state_archive"]["path"]
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(HaarSchedulerError) as error:
        load_committed_haar_shard(
            tmp_path, expected_identity=identity, device="cpu"
        )
    assert error.value.failure_code == "hierarchical_shard_archive_corrupt"


def test_production_run_requires_cuda_state_before_runner():
    called = False

    def runner(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError

    schedule = NestedHaarSchedule(pool="reference", role="nested_a")
    identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=(0xA0000,),
        coarsest_start_step=0,
        root_seed=261181,
        panel_namespace="production-prereq",
    )
    with pytest.raises(HaarSchedulerError) as error:
        run_nested_haar_shard(
            initialize_nested_branch_states(_states(), schedule),
            identity=identity,
            jacobi_profile=_profile(),
            level_runner=runner,
        )
    assert error.value.failure_code == "hierarchical_cuda_state_required"
    assert not called


def test_disabling_production_checks_requires_explicit_test_flag():
    schedule = NestedHaarSchedule(pool="reference", role="nested_a")
    identity = HaarShardIdentity(
        schedule=schedule,
        path_ids=(0xA0000,),
        coarsest_start_step=0,
        root_seed=261181,
        panel_namespace="bad-mode",
    )
    with pytest.raises(HaarSchedulerError) as error:
        run_nested_haar_shard(
            initialize_nested_branch_states(_states(), schedule),
            identity=identity,
            jacobi_profile=_profile(),
            level_runner=_fast_runner,
            production_authorizing=False,
        )
    assert error.value.failure_code == "nonauthorizing_mode_not_explicit"
