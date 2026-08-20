from __future__ import annotations

from pathlib import Path


def test_nvrtc_compat_loader_source_handles_both_private_api_shapes() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "mnist"
        / "d0_jacobi_rb_nvrtc_compat.py"
    ).read_text()
    assert "if isinstance(compiled, tuple)" in source
    assert "binary = compiled" in source
    assert "_cuda_load_module(binary_bytes, requested)" in source
    assert "hashlib.sha256(binary_bytes).hexdigest()" in source
