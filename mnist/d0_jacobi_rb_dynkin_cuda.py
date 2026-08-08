"""Additive certified CUDA exponential enclosure for Dynkin observables.

The immutable Jacobi authorizer already contains the required directed
double-double ball algebra and certified degree-24 exponential.  This module
compiles that byte-identical source together with one additional read-only
kernel that exposes ``expm1(-2u)`` and ``expm1(-6u)`` balls.  It does not
modify or replace the transition backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
from typing import Any

import torch
from torch import Tensor

from mnist.d0_jacobi_rb_cuda_fused import (
    REQUIRED_SELFTEST_MASK,
    SELFTEST_KERNEL_NAME,
    _CUDA_SOURCE as _PARENT_CUDA_SOURCE,
)


DYNKIN_CUDA_VERSION = "jacobi-rb-dynkin-dd-exp24-v1"
DYNKIN_EXP_KERNEL_NAME = "jacobi_rb_dynkin_exp24_v1"

_DYNKIN_KERNEL_SOURCE = r"""

extern "C" __global__ void jacobi_rb_dynkin_exp24_v1(
    const double* exposure,
    double* expm1_2_center,
    double* expm1_2_radius,
    double* expm1_6_center,
    double* expm1_6_radius,
    u8* valid,
    int count
) {
    int index=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (index>=count) return;
    double u=exposure[index];
    if (!finite_d(u) || u<0.0) {
        expm1_2_center[index]=0.0;
        expm1_2_radius[index]=0.0;
        expm1_6_center[index]=0.0;
        expm1_6_radius[index]=0.0;
        valid[index]=0;
        return;
    }
    if (u==0.0) {
        expm1_2_center[index]=0.0;
        expm1_2_radius[index]=0.0;
        expm1_6_center[index]=0.0;
        expm1_6_radius[index]=0.0;
        valid[index]=1;
        return;
    }
    Ball ub=exact_double(u);
    Ball one=exact_double(1.0);
    Ball e2=ball_sub(exp24(ball_scale(ub,-2.0)),one);
    Ball e6=ball_sub(exp24(ball_scale(ub,-6.0)),one);
    if (!e2.ok || !e6.ok) {
        expm1_2_center[index]=0.0;
        expm1_2_radius[index]=0.0;
        expm1_6_center[index]=0.0;
        expm1_6_radius[index]=0.0;
        valid[index]=0;
        return;
    }
    double c2=dd_rn(e2.c);
    double c6=dd_rn(e6.c);
    double lo2=ball_lower(e2);
    double hi2=ball_upper(e2);
    double lo6=ball_lower(e6);
    double hi6=ball_upper(e6);
    double r2=dmax(__dsub_ru(c2,lo2),__dsub_ru(hi2,c2));
    double r6=dmax(__dsub_ru(c6,lo6),__dsub_ru(hi6,c6));
    int ok=finite_d(c2) && finite_d(c6) && finite_d(r2) && finite_d(r6)
        && r2>=0.0 && r6>=0.0 && lo2<=c2 && c2<=hi2
        && lo6<=c6 && c6<=hi6;
    expm1_2_center[index]=c2;
    expm1_2_radius[index]=r2;
    expm1_6_center[index]=c6;
    expm1_6_radius[index]=r6;
    valid[index]=(u8)(ok ? 1 : 0);
}
"""

CUDA_SOURCE = _PARENT_CUDA_SOURCE + _DYNKIN_KERNEL_SOURCE
CUDA_SOURCE_SHA256 = hashlib.sha256(CUDA_SOURCE.encode("utf-8")).hexdigest()


class DynkinCudaCertificateError(RuntimeError):
    """Raised when the additive CUDA certificate cannot authorize output."""


@dataclass(frozen=True)
class _DynkinCudaBundle:
    kernel: Any
    selftest_mask: int
    binary_sha256: str
    source_sha256: str = CUDA_SOURCE_SHA256


@dataclass(frozen=True)
class CertifiedDynkinDecayBatch:
    expm1_2_center: Tensor
    expm1_2_radius: Tensor
    expm1_6_center: Tensor
    expm1_6_radius: Tensor
    valid_mask: Tensor
    diagnostics: dict[str, Any]


_CACHE: dict[tuple[int, int, int, tuple[str, ...]], _DynkinCudaBundle] = {}
_LOCK = threading.Lock()


def _compile(device: torch.device, compile_flags: tuple[str, ...]) -> _DynkinCudaBundle:
    if not torch.cuda.is_available():
        raise DynkinCudaCertificateError("CUDA is unavailable")
    index = device.index
    if index is None:
        index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    key = (
        int(index),
        int(properties.major),
        int(properties.minor),
        tuple(compile_flags),
    )
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        from torch.utils import cpp_extension
        from torch.cuda._utils import _cuda_load_module, _nvrtc_compile

        previous_cuda_home = cpp_extension.CUDA_HOME
        if previous_cuda_home is None:
            cpp_extension.CUDA_HOME = str(Path(torch.__file__).resolve().parent)
        try:
            with torch.cuda.device(index):
                binary, lowered = _nvrtc_compile(
                    CUDA_SOURCE,
                    DYNKIN_EXP_KERNEL_NAME,
                    compute_capability=f"{properties.major}{properties.minor}",
                    nvcc_options=list(compile_flags),
                )
                loaded = _cuda_load_module(
                    binary, [lowered, SELFTEST_KERNEL_NAME]
                )
                mask = torch.zeros(1, dtype=torch.uint64, device=device)
                loaded[SELFTEST_KERNEL_NAME](
                    grid=(1, 1, 1),
                    block=(1, 1, 1),
                    args=[mask],
                    stream=torch.cuda.current_stream(device),
                )
                torch.cuda.synchronize(device)
                observed = int(mask.item())
                if observed != REQUIRED_SELFTEST_MASK:
                    raise DynkinCudaCertificateError(
                        "parent directed-arithmetic self-test failed: "
                        f"observed=0x{observed:x}, required=0x{REQUIRED_SELFTEST_MASK:x}"
                    )
                bundle = _DynkinCudaBundle(
                    kernel=loaded[lowered],
                    selftest_mask=observed,
                    binary_sha256=hashlib.sha256(binary).hexdigest(),
                )
        except DynkinCudaCertificateError:
            raise
        except Exception as exc:
            raise DynkinCudaCertificateError(
                f"additive Dynkin NVRTC backend failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            cpp_extension.CUDA_HOME = previous_cuda_home
        _CACHE[key] = bundle
        return bundle


def certified_dynkin_decay_batch_cuda(
    exposure: Tensor,
    *,
    compile_flags: tuple[str, ...],
    threads_per_block: int = 128,
) -> CertifiedDynkinDecayBatch:
    """Return directed DD/exp24 balls for the two Dynkin eigen-decays."""

    if (
        not isinstance(exposure, Tensor)
        or exposure.dtype != torch.float64
        or exposure.ndim != 2
        or not exposure.is_cuda
        or not exposure.is_contiguous()
    ):
        raise TypeError("exposure must be contiguous rank-two CUDA float64")
    if int(threads_per_block) not in {32, 64, 128, 256, 512}:
        raise ValueError("threads_per_block must be a supported whole-warp size")
    count = int(exposure.numel())
    if count <= 0 or count > 4096:
        raise ValueError("Dynkin exp24 launch must contain 1..4096 lanes")
    bundle = _compile(exposure.device, tuple(compile_flags))
    e2 = torch.empty_like(exposure)
    r2 = torch.empty_like(exposure)
    e6 = torch.empty_like(exposure)
    r6 = torch.empty_like(exposure)
    valid = torch.empty(exposure.shape, dtype=torch.uint8, device=exposure.device)
    blocks = (count + int(threads_per_block) - 1) // int(threads_per_block)
    bundle.kernel(
        grid=(blocks, 1, 1),
        block=(int(threads_per_block), 1, 1),
        args=[exposure, e2, r2, e6, r6, valid, count],
        stream=torch.cuda.current_stream(exposure.device),
    )
    valid_bool = valid != 0
    return CertifiedDynkinDecayBatch(
        expm1_2_center=e2,
        expm1_2_radius=r2,
        expm1_6_center=e6,
        expm1_6_radius=r6,
        valid_mask=valid_bool,
        diagnostics={
            "version": DYNKIN_CUDA_VERSION,
            "source_sha256": bundle.source_sha256,
            "binary_sha256": bundle.binary_sha256,
            "selftest_mask": int(bundle.selftest_mask),
            "lane_count": count,
            "invalid_count_deferred_to_certificate_mask": 1,
            "authorization_checked_at_shard_commit": 1,
            "authorizing_directed_dd_exp24": 1,
            "libdevice_transcendental_authorization": 0,
            "launch_count": 1,
        },
    )


__all__ = [
    "DYNKIN_CUDA_VERSION",
    "DYNKIN_EXP_KERNEL_NAME",
    "CUDA_SOURCE_SHA256",
    "DynkinCudaCertificateError",
    "CertifiedDynkinDecayBatch",
    "certified_dynkin_decay_batch_cuda",
]
