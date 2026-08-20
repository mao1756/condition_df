from __future__ import annotations

import inspect

import pytest
import torch

import mnist.d0_jacobi_rb_cuda_deferred as rb_cuda


def _frozen_cuda_available() -> bool:
    if not torch.cuda.is_available() or rb_cuda._reference._arb is None:
        return False
    properties = torch.cuda.get_device_properties(0)
    return (
        str(torch.__version__) == "2.11.0+cu128"
        and str(torch.version.cuda) == "12.8"
        and (properties.major, properties.minor) == (12, 0)
    )


def _fake_prepared() -> rb_cuda.PreparedDeferredRBCudaBackend:
    return rb_cuda.PreparedDeferredRBCudaBackend(
        device=torch.device("cuda:0"),
        profile=rb_cuda.JacobiRBCudaProfile(),
        candidate_kernel=object(),
        candidate_binary_sha256="0" * 64,
        fused_bundle=object(),
        fused_report={"fused_cuda_authorizer_available": True},
    )


def test_deferred_api_is_additive_and_separates_synchronous_preparation() -> None:
    signature = inspect.signature(
        rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_no_fallback
    )
    assert list(signature.parameters) == [
        "head_fraction",
        "exposure",
        "rng_key",
        "transition_ids",
        "prepared",
        "prepared_rng_seed",
    ]
    assert signature.parameters["rng_key"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["transition_ids"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["prepared"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["prepared_rng_seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert callable(rb_cuda.sample_alpha1_rb_transition_batch_cuda)
    assert "probe_fused_cuda_authorizer" in inspect.getsource(
        rb_cuda.prepare_alpha1_rb_transition_batch_cuda_deferred
    )

    candidate_signature = inspect.signature(
        rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_candidate
    )
    assert list(candidate_signature.parameters) == list(signature.parameters)
    candidate_fields = set(rb_cuda.CandidateRBCudaBatch.__dataclass_fields__)
    assert {
        "active_mask",
        "structural_noop_mask",
        "approximation_mask",
        "valid_mask",
        "candidate_lower",
        "candidate_upper",
    } <= candidate_fields
    assert not any(
        "certif" in name or "authoriz" in name for name in candidate_fields
    )
    candidate_source = inspect.getsource(
        rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_candidate
    )
    for counter in (
        "resource_cap_count",
        "invalid_density_count",
        "correction_count",
        "clipping_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
    ):
        assert f'"{counter}"' in candidate_source


def test_deferred_enqueue_and_static_validator_contain_no_host_barrier() -> None:
    sources = "\n".join(
        [
            inspect.getsource(rb_cuda._require_deferred_cuda_inputs),
            inspect.getsource(
                rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_no_fallback
            ),
        ]
    )
    for forbidden in (
        ".item(",
        ".cpu(",
        ".numpy(",
        "cuda.synchronize(",
        "perf_counter(",
        "probe_fused_cuda_authorizer(",
        "launch_fused_cuda_authorizer_with_neighbors(",
        "torch.tensor(",
    ):
        assert forbidden not in sources

    candidate_source = inspect.getsource(
        rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_candidate
    )
    for forbidden in (
        ".item(",
        ".cpu(",
        ".numpy(",
        "cuda.synchronize(",
        "launch_fused_cuda_authorizer(",
        "sample_alpha1_rb_transition_batch_cuda(",
        "torch.tensor(",
    ):
        assert forbidden not in candidate_source


def test_deferred_static_validation_rejects_host_tensors_before_launch() -> None:
    x = torch.tensor([0.25, 0.75], dtype=torch.float64)
    u = torch.full_like(x, 0.001)
    ids = torch.tensor([17, 17], dtype=torch.uint64)
    with pytest.raises(ValueError, match="CUDA tensor"):
        rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_no_fallback(
            x,
            u,
            rng_key=(261402, "deferred-static"),
            transition_ids=ids,
            prepared=_fake_prepared(),
            prepared_rng_seed=None,
        )

    with pytest.raises(ValueError, match="CUDA tensor"):
        rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_candidate(
            x,
            u,
            rng_key=(261402, "candidate-static"),
            transition_ids=ids,
            prepared=_fake_prepared(),
            prepared_rng_seed=None,
        )


@pytest.mark.skipif(
    not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable"
)
def test_candidate_cuda_ids_pairing_permutation_chunk_and_no_false_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    profile = rb_cuda.JacobiRBCudaProfile()
    prepared = rb_cuda.prepare_alpha1_rb_transition_batch_cuda_deferred(
        device=device, profile=profile
    )
    key = (261402, "candidate-duplicate-id-control")
    prepared_seed = rb_cuda.prepare_alpha1_rb_transition_cuda_rng_seed(
        rng_key=key, prepared=prepared
    )
    x = torch.tensor(
        [0.25, 0.25, 0.40, 0.60, 0.75, 0.75],
        dtype=torch.float64,
        device=device,
    )
    u = torch.tensor(
        [0.001, 0.001, 0.002, 0.003, 0.0, 0.0],
        dtype=torch.float64,
        device=device,
    )
    ids = torch.tensor([17, 17, 23, 29, 31, 31], dtype=torch.uint64, device=device)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate path invoked an exact authorizer")

    with monkeypatch.context() as patch:
        patch.setattr(rb_cuda, "launch_fused_cuda_authorizer", forbidden)
        full = rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_candidate(
            x,
            u,
            rng_key=key,
            transition_ids=ids,
            prepared=prepared,
            prepared_rng_seed=prepared_seed,
        )
    torch.cuda.synchronize(device)
    assert full.transition_ids.tolist() == ids.tolist()
    assert full.later_head_fraction[0].item() == full.later_head_fraction[1].item()
    assert full.denoising_target[0].item() == full.denoising_target[1].item()
    assert full.structural_noop_mask.tolist() == [False, False, False, False, True, True]
    assert full.approximation_mask.tolist() == [True, True, True, True, False, False]
    assert full.valid_mask.tolist() == [True] * 6
    assert torch.equal(full.later_head_fraction[4:], x[4:])
    assert torch.equal(full.denoising_target[4:], torch.zeros_like(x[4:]))
    assert not any(
        "certif" in name or "authoriz" in name
        for name in full.device_diagnostics
    )
    for counter in (
        "resource_cap_count",
        "invalid_density_count",
        "correction_count",
        "clipping_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
    ):
        value = full.device_diagnostics[counter]
        assert value.ndim == 0
        assert value.dtype == torch.int64
        assert value.device == device
        assert int(value.item()) == 0
    exact = rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_no_fallback(
        x,
        u,
        rng_key=key,
        transition_ids=ids,
        prepared=prepared,
        prepared_rng_seed=prepared_seed,
    )
    assert torch.equal(full.transition_ids, exact.transition_ids)

    pieces = []
    targets = []
    lowers = []
    uppers = []
    for start, stop in ((0, 2), (2, 5), (5, 6)):
        part = rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_candidate(
            x[start:stop].contiguous(),
            u[start:stop].contiguous(),
            rng_key=key,
            transition_ids=ids[start:stop].contiguous(),
            prepared=prepared,
            prepared_rng_seed=prepared_seed,
        )
        pieces.append(part.later_head_fraction)
        targets.append(part.denoising_target)
        lowers.append(part.candidate_lower)
        uppers.append(part.candidate_upper)
    assert torch.equal(torch.cat(pieces), full.later_head_fraction)
    assert torch.equal(torch.cat(targets), full.denoising_target)
    assert torch.equal(torch.cat(lowers), full.candidate_lower)
    assert torch.equal(torch.cat(uppers), full.candidate_upper)

    permutation = torch.tensor([3, 0, 5, 2, 1, 4], device=device)
    permuted = rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_candidate(
        x[permutation].contiguous(),
        u[permutation].contiguous(),
        rng_key=key,
        transition_ids=ids.to(torch.int64)[permutation].to(torch.uint64).contiguous(),
        prepared=prepared,
        prepared_rng_seed=prepared_seed,
    )
    inverse = torch.argsort(permutation)
    assert torch.equal(permuted.later_head_fraction[inverse], full.later_head_fraction)
    assert torch.equal(permuted.denoising_target[inverse], full.denoising_target)
    assert torch.equal(permuted.candidate_lower[inverse], full.candidate_lower)
    assert torch.equal(permuted.candidate_upper[inverse], full.candidate_upper)


@pytest.mark.skipif(
    not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable"
)
def test_deferred_duplicate_ids_match_synchronous_exact_without_host_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    profile = rb_cuda.JacobiRBCudaProfile()
    prepared = rb_cuda.prepare_alpha1_rb_transition_batch_cuda_deferred(
        device=device, profile=profile
    )
    x = torch.tensor(
        [0.25, 0.25, 0.40, 0.60], dtype=torch.float64, device=device
    )
    u = torch.full_like(x, 0.001)
    ids = torch.tensor([17, 17, 23, 23], dtype=torch.uint64, device=device)
    key = (261402, "deferred-duplicate-id-control")
    prepared_seed = rb_cuda.prepare_alpha1_rb_transition_cuda_rng_seed(
        rng_key=key, prepared=prepared
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("deferred enqueue crossed a host synchronization boundary")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "synchronize", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch, "tensor", forbidden)
        deferred = rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_no_fallback(
            x,
            u,
            rng_key=key,
            transition_ids=ids,
            prepared=prepared,
            prepared_rng_seed=prepared_seed,
        )

    torch.cuda.synchronize(device)
    assert bool(deferred.valid_mask.all())
    assert bool(deferred.certified_mask.all())
    assert not bool(deferred.fallback_mask.any())
    assert not bool(deferred.device_diagnostics["replay_required"])
    assert deferred.transition_ids.tolist() == [17, 17, 23, 23]
    assert deferred.later_head_fraction[0].item() == deferred.later_head_fraction[1].item()
    assert deferred.denoising_target[0].item() == deferred.denoising_target[1].item()

    synchronous = rb_cuda.sample_alpha1_rb_transition_batch_cuda(
        x,
        u,
        rng_key=key,
        transition_ids=ids,
        profile=profile,
    )
    torch.testing.assert_close(
        deferred.later_head_fraction,
        synchronous.later_head_fraction,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        deferred.denoising_target,
        synchronous.denoising_target,
        rtol=0,
        atol=0,
    )
    assert torch.equal(deferred.certificate_codes, synchronous.certificate_codes)


@pytest.mark.skipif(
    not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable"
)
def test_deferred_invalid_values_are_device_marked_for_whole_shard_replay() -> None:
    device = torch.device("cuda:0")
    prepared = rb_cuda.prepare_alpha1_rb_transition_batch_cuda_deferred(
        device=device, profile=rb_cuda.JacobiRBCudaProfile()
    )
    x = torch.tensor([0.25, float("nan"), 0.75], dtype=torch.float64, device=device)
    u = torch.tensor([0.0, 0.001, -0.5], dtype=torch.float64, device=device)
    ids = torch.tensor([31, 32, 33], dtype=torch.uint64, device=device)
    key = (261402, "deferred-invalid-control")
    prepared_seed = rb_cuda.prepare_alpha1_rb_transition_cuda_rng_seed(
        rng_key=key, prepared=prepared
    )
    result = rb_cuda.enqueue_alpha1_rb_transition_batch_cuda_no_fallback(
        x,
        u,
        rng_key=key,
        transition_ids=ids,
        prepared=prepared,
        prepared_rng_seed=prepared_seed,
    )
    torch.cuda.synchronize(device)
    assert result.valid_mask.tolist() == [True, False, False]
    assert result.structural_noop_mask.tolist() == [True, False, False]
    assert bool(result.device_diagnostics["replay_required"])
    assert result.device_diagnostics["invalid_input_count"].item() == 2
    assert result.later_head_fraction[0].item() == x[0].item()
    assert result.denoising_target[0].item() == 0.0
