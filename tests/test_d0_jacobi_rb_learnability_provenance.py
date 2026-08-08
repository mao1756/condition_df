from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnist.d0_jacobi_rb_learnability_provenance as provenance
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError


def test_frozen_parent_and_source_constants() -> None:
    assert provenance.MULTIPATH_REGISTRY_RECORD_COUNT == 891
    assert (
        provenance.MULTIPATH_REGISTRY_SHA256
        == "b1724cb1222baf315b3aff24858ac6d979a2ed36e0331995245220a5861545f5"
    )
    assert provenance.STRANG_REGISTRY_RECORD_COUNT == 1308
    assert (
        provenance.STRANG_REGISTRY_SHA256
        == "734c93e1e7d0be29041e1d567b36cbd8ea7aac50df7996d5f8c41fbddef8e632"
    )
    assert provenance.HAAR_REGISTRY_RECORD_COUNT == 511
    assert (
        provenance.HAAR_REGISTRY_SHA256
        == "8281cc9254fd91e824baea9f0a0e19386a045a21aee5ba377295dfcb734acfde"
    )
    assert (
        provenance.SOURCE_IMAGE_NPZ_SHA256
        == "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
    )
    assert provenance.ALLOWED_MODEL_INPUTS == (
        "later_full_state",
        "reverse_time",
        "phase",
        "color",
        "duration",
        "label",
    )


def test_safe_artifact_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ArtifactCompatibilityError, match="unsafe"):
        provenance._safe_artifact_path(tmp_path.resolve(), "../escape.json")
    with pytest.raises(ArtifactCompatibilityError, match="unsafe"):
        provenance._safe_artifact_path(tmp_path.resolve(), str(tmp_path.resolve()))


def test_no_work_check_fails_closed() -> None:
    provenance._assert_no_work({}, "fixture")
    with pytest.raises(ArtifactCompatibilityError, match="physical_training"):
        provenance._assert_no_work(
            {"physical_training_performed": 1}, "fixture"
        )
    with pytest.raises(ArtifactCompatibilityError, match="reverse_sampling"):
        provenance._assert_no_work(
            {"reverse_sampling_performed": True}, "fixture"
        )


def test_source_image_npz_is_hashed_from_bytes(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_root = (
        repository
        / "runs"
        / "experiment12_d0_jacobi_rb_strang_refinement"
        / provenance.STRANG_RUN_BASENAME
    )
    if not (source_root / "source_image.npz").is_file():
        pytest.skip("production source-image artifact is not present")
    target = tmp_path / "source"
    target.mkdir()
    (target / "source_image.json").write_bytes(
        (source_root / "source_image.json").read_bytes()
    )
    (target / "source_image.npz").write_bytes(
        (source_root / "source_image.npz").read_bytes()
    )
    record = provenance._verify_source_image(target)
    assert record["source_image_npz_hash_pass"] == 1
    assert record["source_image_npz_sha256"] == provenance.SOURCE_IMAGE_NPZ_SHA256

    with (target / "source_image.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ArtifactCompatibilityError, match="SHA-256"):
        provenance._verify_source_image(target)


def test_future_contract_rejects_an_extra_model_input(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = (
        repository
        / "runs"
        / "experiment12_d0_jacobi_rb_haar_power_recovery_confirmation"
        / provenance.HAAR_RUN_BASENAME
        / "future_model_input_contract.json"
    )
    if not source.is_file():
        pytest.skip("production future-model contract is not present")
    target = tmp_path / "haar"
    target.mkdir()
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["allowed_inputs"].append("earlier_state")
    (target / "future_model_input_contract.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactCompatibilityError, match="file SHA-256"):
        provenance._verify_future_model_contract(target)


def _parent_record(kind: str) -> dict[str, object]:
    return {
        "schema": f"fixture-{kind}",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


def test_combined_verifier_exposes_preflight_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provenance,
        "verify_successful_multipath_parent",
        lambda path: _parent_record("multipath"),
    )
    monkeypatch.setattr(
        provenance,
        "verify_failed_strang_parent",
        lambda path: _parent_record("strang"),
    )
    monkeypatch.setattr(
        provenance,
        "verify_power_only_haar_parent",
        lambda path: _parent_record("haar"),
    )
    record = provenance.verify_learnability_parents(
        parent_multipath_run_dir="multipath",
        parent_strang_run_dir="strang",
        parent_haar_run_dir="haar",
    )
    assert record["passed"] == 1
    assert record["multipath_kernel_gate_pass"] == 1
    assert record["multipath_target_gate_pass"] == 1
    assert record["strang_power_failure_preserved_pass"] == 1
    assert record["haar_power_only_failure_pass"] == 1
    assert record["parents_no_training_pass"] == 1
    assert record["state_dependent_strang_refinement_established"] == 0
    assert record["larger_exact_discrete_chain_training_planning_authorized"] == 0
    compatibility = provenance.verify_learnability_parents(
        multipath_run_dir="multipath",
        strang_run_dir="strang",
        haar_run_dir="haar",
    )
    assert compatibility == record


def test_combined_verifier_fails_if_a_parent_does_not_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provenance,
        "verify_successful_multipath_parent",
        lambda path: {**_parent_record("multipath"), "passed": 0},
    )
    monkeypatch.setattr(
        provenance,
        "verify_failed_strang_parent",
        lambda path: _parent_record("strang"),
    )
    monkeypatch.setattr(
        provenance,
        "verify_power_only_haar_parent",
        lambda path: _parent_record("haar"),
    )
    with pytest.raises(ArtifactCompatibilityError, match="failed verification"):
        provenance.verify_learnability_parents(
            parent_multipath_run_dir="multipath",
            parent_strang_run_dir="strang",
            parent_haar_run_dir="haar",
        )
