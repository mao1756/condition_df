"""Cross-platform protected-source integrity checks for RunPod preflight.

The historical candidate pilot stores exact byte hashes for 27 protected source
files.  One of those files uses mixed LF/CRLF newlines, so a normal Git/ZIP
checkout can preserve the Python source while changing its byte hash.  This
module keeps a second, RunPod-only inventory whose hashes are computed after
canonical newline normalization.  It therefore accepts newline-only checkout
conversion while continuing to reject every other source edit.

The original byte-level inventory and its historical tests remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Mapping


PROTECTED_SOURCE_CANONICAL_LF_HASHES: dict[str, str] = {
    "mnist/__init__.py": "1afbf919b879fc8c499db24009ce92e92ee03b198cfb427830a18df37df86ce4",
    "mnist/conditioned_diffusion.py": "a52b1dd829ce0d77b2207bd1a2081fc976d7dd73033d75083bd108c7fc2317bd",
    "mnist/d0_jacobi_artifacts.py": "75bac4e947349993f6f7bdc3cf6df31e0861f67d8fbe7688d3761ee7d6325e21",
    "mnist/d0_jacobi_denoising.py": "a434b27daba832ae81de5e677b8c8482d30680e9cbd079f427fb7ceec1a47b39",
    "mnist/d0_jacobi_rb_absolute_coordinate.py": "5fbd05880e584bfe5fcfb5090955b1cf15db8f1dc3eb0395cc2c915d7c5a7183",
    "mnist/d0_jacobi_rb_boundary_tangent.py": "be4ab9ad8007e567bb518b98c04b37e4900669c53607ef6612f951d15d3a17ce",
    "mnist/d0_jacobi_rb_boundary_tangent_frequency1_coordinate.py": "bb67d8f44136e82647b881e8badd4f6b72382432874aff9fb30d65da21edbed4",
    "mnist/d0_jacobi_rb_boundary_tangent_v3_provenance.py": "c591d4047c6b3763247d56e7eedcff97d4c8e7d82fa28d5ec844ae142b59e4f6",
    "mnist/d0_jacobi_rb_boundary_tangent_zero_baseline.py": "5aa6fdfe7f6e23a92ef37fa86deeca317760c1ca90abdffee059bcf06b09235c",
    "mnist/d0_jacobi_rb_coarse_residual.py": "b3157a81cad5dcb257cb5deb09054b515e659023e44a4546edd61d58659826b3",
    "mnist/d0_jacobi_rb_controls.py": "3186c3321a4f48bda6b7a2a28a600812b7686d0b68aa499ca9fe6735bc7a7d17",
    "mnist/d0_jacobi_rb_cuda.py": "94b95db6c93510c97c36b7cd67b2dec3b1f13a62b3077299e6edd6b97f0ba97a",
    "mnist/d0_jacobi_rb_cuda_certificate.py": "f43bd0459a3200bbead706cf7def1cca17e344bbccd7ae4de5cfb26b1eb9aced",
    "mnist/d0_jacobi_rb_cuda_controls.py": "a834445afa5f4003931254a13fbe1e0838904bf9e47726abeaf3faa5955f01ff",
    "mnist/d0_jacobi_rb_cuda_fused.py": "184a3e9e8e476b835e808de4f1b5b7d641d33997448968539ab240a54f91204d",
    "mnist/d0_jacobi_rb_cuda_multipath.py": "5949dc794085cde340b42133a4a2102815ac85dece4e6799b23762de62507f77",
    "mnist/d0_jacobi_rb_global_dilated.py": "2ea368bc0d001803ce8e8c5f9862feefe01aa88ada395f0279636e8ce6e4135a",
    "mnist/d0_jacobi_rb_learnability.py": "081c9dfa7414c3c9fda80b262162eb3ad6c84ddaff905a058896580c1f1d50b2",
    "mnist/d0_jacobi_rb_reverse_controller.py": "adac975b5d64e23f7d0861dce0ac1b497fa054eed6db2a76358e462d94c8ee5f",
    "mnist/d0_jacobi_rb_spectral.py": "f16851db6f9b5f91cec5fc7ab1121461a4b915a63003dba88530b1a8a4f1b635",
    "mnist/d0_jacobi_rb_strang_refinement.py": "9ba9a12032fb9e4babc72568d5494380c5cc06a74c5495c81369b658fd048975",
    "mnist/d0_jacobi_source_compat.py": "f90ac705d105e03ca258f8507fa74e77e9cc2ef3cea2bf8615594cc5dc5c07ed",
    "mnist/d0_jacobi_v3_source_compat.py": "17e9f47c573944a1affcae43ee63bd057fc18fc54ea917fdbdd6fceecc0c6b8c",
    "mnist/diag_eulerian_jacobi_ddpm_mnist.py": "3f5a0f963bc4b2042a10e71f9290478b2ce27c4520913d219c98c03a807a2c9e",
    "mnist/eulerian_jacobi_ddpm.py": "5875373c34fa6fd4620749c5763ce91903728f7f0d2f70c6e7a65a1f8023ab98",
    "mnist/mnist_generation_benchmark.py": "2ebf13e37e03646222b8decde91034f10dff564fdc0d3f7a967af789ef3cbfd6",
    "mnist/weighted_point_cloud.py": "b70db19c8adbaf7cd89818a61a7dc8b167ec83e013911682702161c7e28fca7d",
}


def canonicalize_newlines(raw: bytes) -> bytes:
    """Return bytes with CRLF and CR newlines canonicalized to LF."""

    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_source_sha256(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(canonicalize_newlines(raw)).hexdigest()


def verify_protected_sources(
    repository_root: str | Path,
    expected: Mapping[str, str] = PROTECTED_SOURCE_CANONICAL_LF_HASHES,
) -> None:
    root = Path(repository_root).resolve()
    failures: list[str] = []
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        observed = canonical_source_sha256(path)
        if observed != digest:
            failures.append(
                f"drifted: {relative}: expected {digest}, observed {observed}"
            )
    if failures:
        raise RuntimeError("protected source integrity failed:\n" + "\n".join(failures))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=".")
    return parser


def main() -> None:
    args = _parser().parse_args()
    verify_protected_sources(args.repository_root)
    print(
        f"Protected source canonical integrity passed "
        f"({len(PROTECTED_SOURCE_CANONICAL_LF_HASHES)} files)."
    )


if __name__ == "__main__":
    main()
