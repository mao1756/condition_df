from __future__ import annotations

from pathlib import Path

import pytest

import mnist.d0_jacobi_rb_bayes_power_provenance as provenance
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError


def test_frozen_parent_constants() -> None:
    assert provenance.PARENT_REGISTRY_RECORD_COUNT == 544
    assert (
        provenance.PARENT_REGISTRY_SEMANTIC_SHA256
        == "5e0b46328b6783614bdb7d394587b32e63d2d33b76f0279abdab6ecdf7d4e18a"
    )
    assert (
        provenance.PARENT_REGISTRY_FILE_SHA256
        == "26370722f9f7ce5a6675bc3b626710373b407f4a4134a3425c852af8b17259a5"
    )
    assert (
        provenance.PARENT_SOURCE_FINGERPRINT
        == "f651d7322384275f269de3442f8e7a03cf062994b6bd894db735541d9f2a699d"
    )
    assert (
        provenance.PARENT_SCIENTIFIC_CONFIG_SHA256
        == "58ccdfc5df2c4b30c28da5a143aa2570e390b007e7825b89d164762d7d23b01c"
    )
    assert provenance.PARENT_DECISION == "no_detectable_one_image_conditional_signal"


@pytest.mark.parametrize(
    "path",
    (
        "cache/train_labels_audit.npz",
        r"cache\validation_labels_audit.npz",
        "/tmp/confirmation_labels_audit.npz",
    ),
)
def test_physical_label_firewall_rejects_every_separator(path: str) -> None:
    assert provenance.is_parent_physical_label_path(path)
    with pytest.raises(ArtifactCompatibilityError, match="label access"):
        provenance.assert_parent_label_firewall([path])


def test_template_access_is_allowlisted_and_disjoint(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    for name in provenance.PARENT_TEMPLATE_ALLOWLIST:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    accessed = provenance.validate_parent_template_access(
        root,
        [
            root / "cache/train_inputs.npz",
            "scientific_config.json",
        ],
    )
    assert accessed == ("cache/train_inputs.npz", "scientific_config.json")
    with pytest.raises(ArtifactCompatibilityError, match="allowlisted"):
        provenance.validate_parent_template_access(root, ["selected_model.pt"])
    with pytest.raises(ArtifactCompatibilityError, match="label access"):
        provenance.validate_parent_template_access(
            root, ["cache/train_labels_audit.npz"]
        )
    with pytest.raises(ArtifactCompatibilityError, match="duplicates"):
        provenance.validate_parent_template_access(
            root, ["scientific_config.json", "scientific_config.json"]
        )


def test_real_parent_verifies_without_opening_label_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    root = (
        repository
        / "runs"
        / "experiment12_d0_jacobi_rb_one_image_learnability"
        / provenance.PARENT_RUN_BASENAME
    )
    if not root.is_dir():
        pytest.skip("immutable production parent is not present")
    original = provenance.file_fingerprint
    hashed: list[str] = []

    def audited_fingerprint(path: str | Path, **kwargs: object) -> str:
        relative = Path(path).resolve().relative_to(root).as_posix()
        assert not provenance.is_parent_physical_label_path(relative)
        hashed.append(relative)
        return original(path, **kwargs)

    monkeypatch.setattr(provenance, "file_fingerprint", audited_fingerprint)
    record = provenance.verify_no_signal_parent(
        root,
        accessed_parent_paths=(
            "cache/train_inputs.npz",
            "cache/validation_inputs.npz",
            "scientific_config.json",
        ),
        verify_nonlabel_hashes=False,
    )
    assert record["passed"] == 1
    assert record["parent_only_aggregate_zero_failure_pass"] == 1
    assert record["parent_label_firewall_pass"] == 1
    assert record["forbidden_label_bytes_opened"] == 0
    assert hashed == ["artifact_registry.json"]


def test_scope_tree_rejects_sampling_or_reconstruction() -> None:
    provenance._zero_scope_tree(
        {"nested": {"sampling_performed": 0, "sampling_authorized": False}},
        "fixture",
    )
    with pytest.raises(ArtifactCompatibilityError, match="sampling_performed"):
        provenance._zero_scope_tree(
            {"nested": {"sampling_performed": 1}},
            "fixture",
        )
    with pytest.raises(ArtifactCompatibilityError, match="reconstruction"):
        provenance._zero_scope_tree(
            {"reconstruction_claim_authorized": True},
            "fixture",
        )
