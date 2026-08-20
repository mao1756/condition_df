"""PyTorch-version-compatible NVRTC loader for Jacobi CUDA kernels.

PyTorch 2.8's private ``_nvrtc_compile`` returns PTX bytes, while the
2.11 implementation used by the original frozen experiment returns
``(binary, lowered_name)``.  The Jacobi kernels are ``extern "C"`` and
therefore have stable unmangled names, so both forms can be normalized into a
single loader without changing the CUDA source or compile flags.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch


def compile_cuda_kernels(
    source: str,
    *,
    primary_name: str,
    kernel_names: Sequence[str],
    device_index: int,
    compute_capability: str,
    compile_flags: Sequence[str],
) -> tuple[dict[str, Any], str, str]:
    """Compile ``source`` and return named kernels plus a binary fingerprint.

    The returned primary name is the lowered name on PyTorch versions that
    expose it and the declared ``extern "C"`` name otherwise.
    """

    from torch.cuda._utils import _cuda_load_module, _nvrtc_compile
    from torch.utils import cpp_extension

    previous_cuda_home = cpp_extension.CUDA_HOME
    if previous_cuda_home is None:
        cpp_extension.CUDA_HOME = str(Path(torch.__file__).resolve().parent)
    try:
        with torch.cuda.device(int(device_index)):
            compiled = _nvrtc_compile(
                source,
                primary_name,
                compute_capability=str(compute_capability),
                nvcc_options=list(compile_flags),
            )
            if isinstance(compiled, tuple):
                if len(compiled) != 2:
                    raise RuntimeError(
                        "unexpected PyTorch NVRTC tuple result; expected "
                        "(binary, lowered_name)"
                    )
                binary, lowered_name = compiled
                lowered = str(lowered_name)
            else:
                binary = compiled
                lowered = str(primary_name)

            if isinstance(binary, str):
                binary_bytes = binary.encode("utf-8")
            elif isinstance(binary, (bytes, bytearray, memoryview)):
                binary_bytes = bytes(binary)
            else:
                raise RuntimeError(
                    "unexpected PyTorch NVRTC binary type: "
                    f"{type(binary).__name__}"
                )

            requested = [lowered]
            for name in kernel_names:
                value = str(name)
                if value not in requested:
                    requested.append(value)
            loaded = _cuda_load_module(binary_bytes, requested)
            if not isinstance(loaded, dict):
                loaded = {name: getattr(loaded, name) for name in requested}
            return loaded, hashlib.sha256(binary_bytes).hexdigest(), lowered
    finally:
        cpp_extension.CUDA_HOME = previous_cuda_home
