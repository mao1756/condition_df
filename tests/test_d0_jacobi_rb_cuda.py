from __future__ import annotations

import inspect
import math
import random

import pytest
import torch

import mnist.d0_jacobi_rb_cuda as rb_cuda
import mnist.d0_jacobi_rb_cuda_fused as rb_fused
from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    BOUNDARY_TANGENT_CACHE_VERSION,
)
from mnist.d0_jacobi_rb_boundary_tangent_prefix_schedule import (
    eager_prefix_profile,
)
from mnist.d0_jacobi_rb_boundary_tangent_prefix_fallback import (
    sample_alpha1_rb_transition_batch_cuda_eager,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule import ROOT_SEED
from mnist.d0_jacobi_rb_cuda import (
    CertifiedRBCudaBatch,
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)


def test_public_contract_and_frozen_profile_are_explicit() -> None:
    assert rb_cuda.__all__ == [
        "JacobiRBCudaProfile",
        "CertifiedRBCudaBatch",
        "sample_alpha1_rb_transition_batch_cuda",
    ]
    signature = inspect.signature(sample_alpha1_rb_transition_batch_cuda)
    assert list(signature.parameters) == [
        "head_fraction",
        "exposure",
        "rng_key",
        "transition_ids",
        "profile",
    ]
    assert signature.parameters["rng_key"].kind is inspect.Parameter.KEYWORD_ONLY
    profile = JacobiRBCudaProfile()
    payload = profile.to_dict()
    assert payload["frozen_torch_version"] == "2.11.0+cu128"
    assert payload["frozen_cuda_version"] == "12.8"
    assert payload["frozen_compute_capability"] == "12.0"
    assert payload["compile_flags"] == (
        "--std=c++17",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    assert payload["cuda_candidate_authorizing"] is False
    assert payload["fused_cuda_authorizer_available"] is None
    assert payload["fused_cuda_authorizer_requires_runtime_selftest"] is True
    assert profile.strengthened().certificate_effort == "strengthened"
    with pytest.raises(ValueError, match="immutable"):
        JacobiRBCudaProfile(frozen_cuda_version="12.9")
    with pytest.raises(ValueError, match="require the Arb fallback"):
        JacobiRBCudaProfile(allow_certified_cpu_fallback=False)


def test_pure_philox_reference_and_canonical_ids_are_stable() -> None:
    assert rb_cuda._philox4x32_10((0, 0, 0, 0), (0, 0)) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )
    assert rb_cuda._canonical_seed(0) == 0x3ACAF20C33B5472B
    assert rb_cuda._philox_u64(0, 0) == 0x277C3B9BAA2CA9E2
    assert rb_cuda._philox_u64(0, 0, 1) == 0x2F337C00F41855F1
    canonical = rb_cuda._canonical_seed(261121)
    prefix = rb_cuda._StatelessPhiloxPrefix(
        canonical, 42, 1024, seed_is_canonical=True
    )
    assert prefix.numerator == rb_cuda._philox_u64(261121, 42)
    assert rb_cuda.canonical_v2_transition_id("p", 2, 3, "e") == (
        rb_cuda.canonical_v2_transition_id("p", 2, 3, "e")
    )
    assert rb_cuda.canonical_v2_transition_id("p", 2, 3, "e") != (
        rb_cuda.canonical_v2_transition_id("p", 2, 3, "f")
    )
    assert rb_cuda._canonical_transition_id(17) == 17
    assert rb_cuda._canonical_transition_id(("image", 4, "phase", 2)) == (
        rb_cuda._canonical_transition_id(("image", 4, "phase", 2))
    )
    assert rb_cuda._canonical_transition_id(("image", 4, "phase", 2)) != (
        rb_cuda._canonical_transition_id(("image", 4, "phase", 3))
    )
    uniform = rb_cuda._philox_uniform_midpoint(261121, 42)
    assert 0.0 < uniform < 1.0
    assert uniform == rb_cuda._philox_uniform_midpoint(261121, 42)


def test_strict_rounding_cell_helper_rejects_boundaries_and_nonfinite_values() -> None:
    value = 0.375
    previous = math.nextafter(value, -math.inf)
    assert rb_cuda._strict_rounding_cell_contains(value, value, value)
    assert not rb_cuda._strict_rounding_cell_contains(previous, value, value)
    assert not rb_cuda._strict_rounding_cell_contains(value, math.inf, value)
    assert not rb_cuda._strict_rounding_cell_contains(1.0, 0.0, value)


def test_fused_device_constants_have_exact_host_proofs() -> None:
    report = rb_fused.verify_fused_device_constants()
    assert report["passed"] == 1
    assert report["coefficient_count"] == 25
    assert report["source_sha256"] == rb_fused.SOURCE_SHA256
    assert report["legendre_recurrence_certificate_pass"] == 1
    assert report["legendre_recurrence_theorem"] == (
        "Johansson-Mezzarobba-2018-Proposition-5"
    )
    assert report["legendre_recurrence_error_factor"] == "(N+1)*(N+2)/4"
    assert all(report["legendre_recurrence_hypotheses"].values())


def test_strict_input_validation_rejects_host_and_wrong_dtypes() -> None:
    profile = JacobiRBCudaProfile()
    with pytest.raises(ValueError, match="CUDA tensor"):
        sample_alpha1_rb_transition_batch_cuda(
            torch.tensor([0.2], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            rng_key=1,
            transition_ids=torch.tensor([2], dtype=torch.uint64),
            profile=profile,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rb_fused.launch_fused_cuda_authorizer_with_neighbors(
            None,
            torch.tensor([0.2], dtype=torch.float64),
            torch.tensor([-0.5], dtype=torch.float64),
            torch.tensor([2], dtype=torch.uint64),
            torch.tensor([0.3], dtype=torch.float64),
            seed=1,
            threads_per_block=32,
            max_prefix_bits=64,
        )
    oversized = torch.zeros(4097, dtype=torch.float64)
    with pytest.raises(ValueError, match="4096-lane cap"):
        rb_fused.launch_fused_cuda_authorizer(
            None,
            oversized,
            oversized,
            torch.zeros(4097, dtype=torch.uint64),
            oversized,
            seed=1,
            threads_per_block=32,
            max_prefix_bits=64,
        )


def _frozen_cuda_available() -> bool:
    if not torch.cuda.is_available() or rb_cuda._reference._arb is None:
        return False
    properties = torch.cuda.get_device_properties(0)
    return (
        str(torch.__version__) == "2.11.0+cu128"
        and str(torch.version.cuda) == "12.8"
        and (properties.major, properties.minor) == (12, 0)
    )


@pytest.mark.skipif(not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable")
def test_prop5_legendre_enclosures_cover_arb_and_inflate_on_fault() -> None:
    bundle, report = rb_fused.probe_fused_cuda_authorizer(
        torch.device("cuda"),
        compile_flags=tuple(JacobiRBCudaProfile().compile_flags),
        cpu_preflight=rb_cuda._certificate_arithmetic_preflight(),
    )
    assert bundle is not None
    assert report["legendre_recurrence_certificate_pass"] == 1
    generator = random.Random(261121)
    random_z = [generator.uniform(-0.999, 0.999) for _ in range(8)]
    z_values = [-1.0, 1.0, *random_z]
    degree_values = [511, 512, 2, 3, 7, 31, 64, 127, 257, 513]
    z = torch.tensor(z_values, dtype=torch.float64, device="cuda")
    degrees = torch.tensor(degree_values, dtype=torch.int32, device="cuda")
    result = rb_fused.probe_fused_legendre_enclosures(bundle, z, degrees)
    torch.cuda.synchronize()
    assert bool(result["valid"].all())
    lower = result["lower"].cpu().tolist()
    upper = result["upper"].cpu().tolist()

    reference = rb_cuda._reference
    with reference._ARB_CONTEXT_LOCK:
        previous_precision = int(reference._flint_ctx.prec)
        try:
            reference._flint_ctx.prec = 256
            for index, (z_value, degree) in enumerate(
                zip(z_values, degree_values, strict=True)
            ):
                p_previous = reference._arb(1)
                p_current = reference._arb_exact(z_value)
                if degree == 0:
                    p_current = p_previous
                for n in range(1, degree):
                    p_next = (
                        (2 * n + 1) * reference._arb_exact(z_value) * p_current
                        - n * p_previous
                    ) / (n + 1)
                    p_previous, p_current = p_current, p_next
                oracle = float(p_current.mid())
                assert lower[index] <= oracle <= upper[index]
        finally:
            reference._flint_ctx.prec = previous_precision

    assert lower[0] == upper[0] == -1.0
    assert lower[1] == upper[1] == 1.0
    fault_z = torch.tensor([0.125], dtype=torch.float64, device="cuda")
    fault_degree = torch.tensor([257], dtype=torch.int32, device="cuda")
    baseline = rb_fused.probe_fused_legendre_enclosures(
        bundle, fault_z, fault_degree
    )
    injected_error = math.ldexp(1.0, -70)
    inflated = rb_fused.probe_fused_legendre_enclosures(
        bundle,
        fault_z,
        fault_degree,
        injected_maximum_local_error=torch.tensor(
            [injected_error], dtype=torch.float64, device="cuda"
        ),
    )
    torch.cuda.synchronize()
    assert inflated["radius"].item() > baseline["radius"].item()
    theorem_floor = ((258 * 259) / 4) * injected_error
    assert inflated["radius"].item() >= theorem_floor


@pytest.mark.skipif(not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable")
def test_zero_duration_is_device_noop_without_loading_nvrtc(monkeypatch) -> None:
    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("zero-duration rows must not load NVRTC")

    monkeypatch.setattr(rb_cuda, "_load_cuda_kernel", forbidden_loader)
    x = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64, device="cuda")
    exposure = torch.zeros_like(x)
    transition_ids = torch.tensor([5, 7, 11], dtype=torch.uint64, device="cuda")
    result = sample_alpha1_rb_transition_batch_cuda(
        x,
        exposure,
        rng_key=(261121, "noops"),
        transition_ids=transition_ids,
        profile=JacobiRBCudaProfile(),
    )
    assert isinstance(result, CertifiedRBCudaBatch)
    torch.testing.assert_close(result.later_head_fraction, x, rtol=0, atol=0)
    assert not bool(result.active_mask.any())
    assert not bool(result.certified_mask.any())
    assert not bool(result.fallback_mask.any())
    assert bool((result.denoising_target == 0).all())
    assert bool((result.certificate_codes == 0).all())
    assert bool((result.prefix_bits == 0).all())
    assert result.diagnostics["zero_duration_count"].device == x.device
    assert result.diagnostics["zero_duration_count"].item() == 3


@pytest.mark.skipif(not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable")
def test_active_row_is_device_authorized_and_bit_matches_arb() -> None:
    x = torch.tensor([0.37, 0.2], dtype=torch.float64, device="cuda")
    exposure = torch.tensor([0.5, 0.0], dtype=torch.float64, device="cuda")
    transition_ids = torch.tensor([7, 8], dtype=torch.uint64, device="cuda")
    result = sample_alpha1_rb_transition_batch_cuda(
        x,
        exposure,
        rng_key=261121,
        transition_ids=transition_ids,
        profile=JacobiRBCudaProfile(candidate_modes=32),
    )
    assert result.later_head_fraction.is_cuda
    assert result.diagnostics["active_count"].is_cuda
    assert result.certificate_codes.tolist() == [15, 0]
    assert result.certified_mask.tolist() == [True, False]
    assert result.cuda_certified_mask.tolist() == [True, False]
    assert result.fallback_mask.tolist() == [False, False]
    assert result.arb_fallback_reason_codes.tolist() == [0, 0]
    assert result.arb_fallback_mode_counts[0].item() == 0
    assert result.mode_counts[0].item() > 0
    assert result.mode_counts[1].item() == 0
    assert result.runtime_report["fused_cuda_authorizer_available"] is True
    assert result.runtime_report["directed_rounding_intrinsics_pass"] is True
    assert result.runtime_report["arithmetic_selftest_mask"] == (
        rb_fused.REQUIRED_SELFTEST_MASK
    )
    assert result.diagnostics["maximum_cuda_launch_lanes"].item() <= 4096
    assert result.diagnostics["candidate_kernel_launch_count"].item() == 1
    assert result.diagnostics["fused_authorizer_launch_count"].item() >= 1
    oracle = rb_cuda._arb_candidate_cell_worker({
        "x": 0.37,
        "exposure": 0.5,
        "candidate": float(result.later_head_fraction[0]),
        "transition_id": 7,
        "profile": rb_cuda._reference_profile(JacobiRBCudaProfile()),
        "max_prefix_bits": 1024,
        "prefix_kind": "philox-v2",
        "seed": rb_cuda._canonical_seed(261121),
    })
    assert float(result.later_head_fraction[0]) == oracle["later"]
    assert float(result.denoising_target[0]) == oracle["target"]
    assert len(result.runtime_report["kernel_sha256"]) == 64
    assert len(result.runtime_report["compile_options_sha256"]) == 64
    assert len(result.runtime_report["binary_sha256"]) == 64
    assert result.quantile_lower[0] <= result.later_head_fraction[0]
    assert result.later_head_fraction[0] <= result.quantile_upper[0]
    torch.testing.assert_close(result.later_head_fraction[1], x[1], rtol=0, atol=0)


@pytest.mark.skipif(not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable")
def test_device_cdf_miss_codes_prove_the_only_permitted_neighbor_direction() -> None:
    profile = JacobiRBCudaProfile()
    bundle, report = rb_fused.probe_fused_cuda_authorizer(
        torch.device("cuda"),
        compile_flags=tuple(profile.compile_flags),
        cpu_preflight=rb_cuda._certificate_arithmetic_preflight(),
    )
    assert bundle is not None, report
    x = torch.tensor([0.37], dtype=torch.float64, device="cuda")
    exposure = torch.tensor([0.5], dtype=torch.float64, device="cuda")
    transition_id = torch.tensor([7], dtype=torch.uint64, device="cuda")
    certified = sample_alpha1_rb_transition_batch_cuda(
        x,
        exposure,
        rng_key=261121,
        transition_ids=transition_id,
        profile=profile,
    )
    exact_cell = float(certified.later_head_fraction.item())
    candidates = torch.tensor(
        [(exact_cell + 1.0) * 0.5, exact_cell * 0.5],
        dtype=torch.float64,
        device="cuda",
    )
    attempt = rb_fused.launch_fused_cuda_authorizer(
        bundle,
        x.repeat(2),
        exposure.repeat(2),
        transition_id.repeat(2),
        candidates,
        seed=rb_cuda._canonical_seed(261121),
        threads_per_block=profile.threads_per_block,
        max_prefix_bits=profile.max_prefix_bits,
    )
    torch.cuda.synchronize()
    assert attempt.authorized_mask.tolist() == [False, False]
    assert attempt.fallback_reason_codes.tolist() == [
        rb_fused._CDF_CANDIDATE_TOO_HIGH,
        rb_fused._CDF_CANDIDATE_TOO_LOW,
    ]


@pytest.mark.skipif(not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable")
def test_authorizer_selftest_fault_fails_before_arb(monkeypatch) -> None:
    monkeypatch.setattr(
        rb_cuda,
        "probe_fused_cuda_authorizer",
        lambda *_args, **_kwargs: (
            None,
            {
                "fused_cuda_authorizer_available": False,
                "fused_cuda_authorizer_unavailable_reason": "injected selftest fault",
            },
        ),
    )
    monkeypatch.setattr(
        rb_cuda, "_arb_pool",
        lambda: (_ for _ in ()).throw(AssertionError("Arb must not mask a backend fault")),
    )
    with pytest.raises(RuntimeError, match="injected selftest fault"):
        sample_alpha1_rb_transition_batch_cuda(
            torch.tensor([0.37], dtype=torch.float64, device="cuda"),
            torch.tensor([0.5], dtype=torch.float64, device="cuda"),
            rng_key=1,
            transition_ids=torch.tensor([2], dtype=torch.uint64, device="cuda"),
            profile=JacobiRBCudaProfile(),
        )


@pytest.mark.skipif(not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable")
def test_forced_strengthened_profile_uses_full_prefix_without_changing_bits() -> None:
    x = torch.tensor([0.37], dtype=torch.float64, device="cuda")
    exposure = torch.tensor([0.5], dtype=torch.float64, device="cuda")
    ids = torch.tensor([7], dtype=torch.uint64, device="cuda")
    primary = sample_alpha1_rb_transition_batch_cuda(
        x, exposure, rng_key=261121, transition_ids=ids,
        profile=JacobiRBCudaProfile(),
    )
    strengthened = sample_alpha1_rb_transition_batch_cuda(
        x, exposure, rng_key=261121, transition_ids=ids,
        profile=JacobiRBCudaProfile().strengthened(),
    )
    torch.testing.assert_close(
        strengthened.later_head_fraction, primary.later_head_fraction, rtol=0, atol=0
    )
    torch.testing.assert_close(
        strengthened.denoising_target, primary.denoising_target, rtol=0, atol=0
    )
    assert strengthened.strengthened_mask.tolist() == [True]
    assert strengthened.prefix_bits.tolist() == [128]


@pytest.mark.skipif(not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable")
def test_recorded_v1_prefix_replay_is_explicit_and_matches_same_prefix() -> None:
    seed, transition_id = 261121, 91
    x = torch.tensor([0.37, 0.2], dtype=torch.float64, device="cuda")
    exposure = torch.tensor([0.5, 0.0], dtype=torch.float64, device="cuda")
    ids = torch.tensor([transition_id, 92], dtype=torch.uint64, device="cuda")
    public = sample_alpha1_rb_transition_batch_cuda(
        x,
        exposure,
        rng_key=seed,
        transition_ids=ids,
        profile=JacobiRBCudaProfile(candidate_modes=16),
    )
    numerators = torch.tensor(
        [rb_cuda._philox_u64(seed, transition_id), 0],
        dtype=torch.uint64,
        device="cuda",
    )
    bits = torch.tensor([64, 0], dtype=torch.int32, device="cuda")
    replay = rb_cuda.certify_alpha1_rb_transition_batch_cuda_with_dyadic_prefixes(
        x,
        exposure,
        numerators,
        bits,
        transition_ids=ids,
        profile=JacobiRBCudaProfile(candidate_modes=16),
    )
    torch.testing.assert_close(
        replay.later_head_fraction, public.later_head_fraction, rtol=0, atol=0
    )
    torch.testing.assert_close(replay.denoising_target, public.denoising_target, rtol=0, atol=0)
    assert replay.runtime_report["rng_contract"] == "parent-v1-recorded-dyadic-prefix"
    assert replay.runtime_report["rng_contract"] != public.runtime_report["rng_contract"]


def test_parent_v1_keyed_continuation_refines_the_recorded_word(monkeypatch) -> None:
    monkeypatch.setattr(
        rb_cuda._reference,
        "_target_interval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate-local fallback must use its adaptive target stop")
        ),
    )
    numerator, bits, _ = rb_cuda._reference.philox_uniform_prefix(
        (261121, "support-prefix", 688), bits=64
    )
    row = rb_cuda._arb_candidate_cell_worker({
        "x": 0.0,
        "exposure": 0.00011484375000000002,
        "candidate": 3.0878081795091672e-09,
        "transition_id": 0,
        "profile": rb_cuda._reference_profile(JacobiRBCudaProfile()),
        "max_prefix_bits": 1024,
        "prefix_kind": "parent-v1-verified-continuation",
        "prefix_numerator": numerator,
        "prefix_bits": bits,
        "v1_key_candidate": 688,
    })
    assert row["later"] == 3.087808179509169e-09
    assert row["target"] == -2.6886007904383883e-05
    assert row["prefix_bits"] == 128
    assert row["certificate_code"] == 15


@pytest.mark.skipif(not _frozen_cuda_available(), reason="frozen CUDA/Arb runtime unavailable")
def test_eager_arb_escalation_certifies_small_exposure_branch_lane() -> None:
    result = sample_alpha1_rb_transition_batch_cuda_eager(
        torch.tensor([0.5], dtype=torch.float64, device="cuda"),
        torch.tensor(
            [7.1777343750000015e-06], dtype=torch.float64, device="cuda"
        ),
        rng_key=(
            ROOT_SEED,
            BOUNDARY_TANGENT_CACHE_VERSION,
            "partial-phase-target-prefix",
        ),
        transition_ids=torch.tensor(
            [7086446741318270976], dtype=torch.uint64, device="cuda"
        ),
        profile=eager_prefix_profile(),
    )

    assert result.fallback_mask.tolist() == [True]
    assert result.strengthened_mask.tolist() == [True]
    assert result.certificate_codes.tolist() == [15]
    assert result.later_head_fraction.item() == 0.4996324195179097
    assert result.denoising_target.item() == 25.60541796782855
    assert result.prefix_bits.item() >= 128
    assert result.quantile_lower.item() <= result.later_head_fraction.item()
    assert result.later_head_fraction.item() <= result.quantile_upper.item()
    assert result.target_lower.item() <= result.denoising_target.item()
    assert result.denoising_target.item() <= result.target_upper.item()
