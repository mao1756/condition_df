from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint, file_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_provenance import (
    CHECKPOINT_COUNT,
    CHECKPOINT_INDEX_FILE_SHA256,
    CHECKPOINT_INDEX_SEMANTIC_SHA256,
    GAIN_CALIBRATION_SEAL_FILE_SHA256,
    GAIN_TABLE_FILE_SHA256,
    NONZERO_CHECKPOINT_COUNT,
    PARENT_ARTIFACT_COUNT,
    PARENT_BASENAME,
    PARENT_REGISTRY_FILE_SHA256,
    PARENT_REGISTRY_SEMANTIC_SHA256,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_FINGERPRINT,
    PARENT_TERMINAL_DECISION,
    RANK_LABEL_OPEN_FILE_SHA256,
    TRAINING_RANK_PATH_TABLES_FILE_SHA256,
    DirectionAdjudicationProvenanceError,
    compare_parent_snapshots,
    load_already_open_role,
    snapshot_parent_run,
    source_fingerprint,
    source_paths,
    validate_semantic_config,
    verify_parent,
    verify_resume_compatibility,
)


def _parent() -> Path:
    return (
        Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist")
        / PARENT_BASENAME
    ).resolve()


@pytest.fixture(scope="module")
def verified_parent() -> tuple[Path, dict[str, object], dict[str, object]]:
    parent = _parent()
    snapshot = snapshot_parent_run(parent)
    result = verify_parent(parent, snapshot=snapshot)
    return parent, snapshot, result


def test_exact_parent_constants_and_full_parent_verification(
    verified_parent: tuple[Path, dict[str, object], dict[str, object]],
) -> None:
    parent, snapshot, result = verified_parent
    assert parent.name == "20260807-132351_production-exact-quartile-specialist"
    assert PARENT_ARTIFACT_COUNT == 4_120
    assert PARENT_REGISTRY_SEMANTIC_SHA256 == (
        "e5f6b3ee257b3d4f86ec3ac54f4223540cf76caa24489d22e9c138a54e19c7bb"
    )
    assert PARENT_REGISTRY_FILE_SHA256 == (
        "e24c7db28081dbceb8f0acf690d779f835379c82a89a2b263976c3e0b631f798"
    )
    assert PARENT_SOURCE_FINGERPRINT == (
        "61a8c2fd6a317c05b9eed73e208d40b0cc6c01d6fdd227ae33d71d0be7c6027d"
    )
    assert PARENT_SCIENTIFIC_CONFIG_SHA256 == (
        "05263b7b01c2586e9a771bd71fe28fbb74d8e5d9da873ce4746019c5167c08c1"
    )
    assert GAIN_TABLE_FILE_SHA256 == (
        "48ec1f17be4869f9a816c0338e8b23cddbdf44dd7000ca60fe317fd787925815"
    )
    assert TRAINING_RANK_PATH_TABLES_FILE_SHA256 == (
        "93f5c4ea39bc658cc5f46b7d31930a0c8c02b2c7ccb106cf314229f0eec32d9b"
    )
    assert CHECKPOINT_INDEX_FILE_SHA256 == (
        "6446cec12529f5634870c43eb349c3a43b9e1b64f0850c04c680b13c1c749d2b"
    )
    assert CHECKPOINT_INDEX_SEMANTIC_SHA256 == (
        "c4112fa6c971bac1ca3b0da471c8915a955a1ee760529cd948530963e38e77c7"
    )
    assert GAIN_CALIBRATION_SEAL_FILE_SHA256 == (
        "a165b1d3c601625ebd058cf67ee564ede751ec5356496c6fdb0c7c8e4094e189"
    )
    assert RANK_LABEL_OPEN_FILE_SHA256 == (
        "9eac05c28339202fafbcf5abdf00e4040679f9087ad929729dbd085736e6e1b6"
    )

    assert result["passed"] == 1
    assert result["decision"] == PARENT_TERMINAL_DECISION
    assert result["valid_scientific_negative"] == 1
    assert result["artifact_count"] == PARENT_ARTIFACT_COUNT
    assert result["all_registered_artifact_hashes_verified"] == 1
    assert result["all_checkpoint_hashes_verified"] == 1
    assert result["all_checkpoint_state_hashes_verified"] == 1
    assert result["checkpoint_count"] == CHECKPOINT_COUNT == 492
    assert result["nonzero_checkpoint_count"] == NONZERO_CHECKPOINT_COUNT == 480
    assert result["cache_bindings_valid"] == 1
    assert result["cache_row_identities_verified"] == 1
    assert result["opened_roles"] == [
        "physical_fit",
        "gain_calibration",
        "training_rank",
    ]
    assert result["selection_confirmation_absent"] == 1
    assert result["historical_design_evidence_authorizing"] == 0
    assert result["parent_files_modified"] == 0
    assert result["parent_tree_sha256"] == snapshot["tree_sha256"]


def test_changed_registry_and_new_selection_evidence_are_rejected(tmp_path: Path) -> None:
    changed = tmp_path / "changed" / PARENT_BASENAME
    changed.mkdir(parents=True)
    (changed / "artifact_registry.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(DirectionAdjudicationProvenanceError, match="registry file hash"):
        verify_parent(
            changed,
            verify_checkpoint_states=False,
            verify_cache_rows=False,
        )

    appeared = tmp_path / "appeared" / PARENT_BASENAME
    appeared.mkdir(parents=True)
    (appeared / "selection_open.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        DirectionAdjudicationProvenanceError,
        match="selection or confirmation evidence appeared",
    ):
        verify_parent(
            appeared,
            verify_checkpoint_states=False,
            verify_cache_rows=False,
        )


def test_strict_loader_reads_open_role_as_nonwriteable_without_parent_mutation(
    verified_parent: tuple[Path, dict[str, object], dict[str, object]],
) -> None:
    parent, _, _ = verified_parent
    protected = {
        path: file_fingerprint(path)
        for path in (
            parent / "gain_label_open.json",
            parent / "rank_label_open.json",
            parent / "gain_calibration_cache_binding.json",
        )
    }
    loaded = load_already_open_role(parent, "gain_calibration")
    assert loaded.role == "gain_calibration"
    assert len(loaded.inputs["sample_key"]) == 57_344
    assert len(loaded.labels["sample_key"]) == 57_344
    assert loaded.input_index["semantic_sha256"] == (
        "14a2d18444667a56b4a1d29a03782d5d9bcfc5b760762ed739835e8c232bd9ac"
    )
    assert loaded.binding["semantic_sha256"] == (
        "dd92addc270df65ff6dfc32e364e434bc3017fad3d71cb5c4ef318828a99e399"
    )
    assert loaded.role_open["semantic_sha256"] == (
        "61080a841b8d5c31bb755d6a92ae6f461617b9a857b4e7e2b06d57027349d2cb"
    )
    assert len(loaded.row_identity_sha256) == 64
    assert all(not value.flags.writeable for value in loaded.inputs.values())
    assert all(not value.flags.writeable for value in loaded.labels.values())
    with pytest.raises(ValueError):
        loaded.labels["denoising_target"].flat[0] = 0.0
    with pytest.raises(TypeError):
        loaded.inputs["new"] = loaded.inputs["sample_key"]  # type: ignore[index]
    assert protected == {path: file_fingerprint(path) for path in protected}
    del loaded
    gc.collect()


def test_strict_loader_never_creates_a_missing_role_open_record(tmp_path: Path) -> None:
    parent = tmp_path / PARENT_BASENAME
    parent.mkdir()
    open_path = parent / "gain_label_open.json"
    with pytest.raises(DirectionAdjudicationProvenanceError, match="not already opened"):
        load_already_open_role(parent, "gain_calibration")
    assert not open_path.exists()
    with pytest.raises(DirectionAdjudicationProvenanceError, match="forbidden historical role"):
        load_already_open_role(parent, "fresh_selection")
    assert not (parent / "selection_open.json").exists()


def test_complete_before_after_snapshots_match_and_detect_changes(
    verified_parent: tuple[Path, dict[str, object], dict[str, object]],
) -> None:
    parent, before, _ = verified_parent
    after = snapshot_parent_run(parent)
    result = compare_parent_snapshots(before, after)
    assert result["passed"] == 1
    assert result["parent_files_modified"] == 0
    assert result["file_count"] == 4_124

    changed = deepcopy(after)
    changed["files"][0]["size"] += 1  # type: ignore[index]
    changed["total_bytes"] += 1  # type: ignore[operator]
    changed["tree_sha256"] = config_fingerprint(changed["files"])
    changed.pop("semantic_sha256")
    changed["semantic_sha256"] = config_fingerprint(changed)
    with pytest.raises(DirectionAdjudicationProvenanceError, match="snapshot changed"):
        compare_parent_snapshots(before, changed)


def test_source_config_and_resume_bindings_are_fail_closed(tmp_path: Path) -> None:
    entries = (
        Path(
            "mnist/d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_provenance.py"
        ),
        Path("mnist/d0_jacobi_rb_boundary_tangent_quartile_gate.py"),
    )
    paths = source_paths(entries)
    assert all(path.is_file() for path in paths)
    assert source_fingerprint(entries) == source_fingerprint(entries)

    config = {
        "schema": "direction-test-config",
        "schema_version": 1,
        "historical_design_evidence_only": 1,
        "authorizing": 0,
        "new_role_count": 0,
        "new_path_count": 0,
        "new_seed_count": 0,
        "cache_generation_authorized": 0,
        "training_authorized": 0,
        "selection_authorized": 0,
        "confirmation_authorized": 0,
        "controller_execution_authorized": 0,
        "sampling_authorized": 0,
    }
    config["semantic_sha256"] = config_fingerprint(config)
    assert validate_semantic_config(
        config,
        expected_schema="direction-test-config",
        expected_sha256=config["semantic_sha256"],
    )["passed"] == 1
    authorizing = dict(config)
    authorizing["training_authorized"] = 1
    authorizing["semantic_sha256"] = config_fingerprint(
        {key: value for key, value in authorizing.items() if key != "semantic_sha256"}
    )
    with pytest.raises(DirectionAdjudicationProvenanceError, match="authorizes"):
        validate_semantic_config(authorizing)

    run = tmp_path / "resume"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"source_fingerprint": "abc", "scientific_config_sha256": "def"}),
        encoding="utf-8",
    )
    artifact = {"schema": "resume-artifact", "schema_version": 1}
    artifact["semantic_sha256"] = config_fingerprint(artifact)
    (run / "preflight_gate.json").write_text(json.dumps(artifact), encoding="utf-8")
    binary = run / "shard.npz"
    binary.write_bytes(b"sealed shard")
    result = verify_resume_compatibility(
        run,
        expected_bindings={
            "source_fingerprint": "abc",
            "scientific_config_sha256": "def",
        },
        artifact_bindings={
            "preflight_gate.json": artifact["semantic_sha256"],
            "shard.npz": file_fingerprint(binary),
        },
    )
    assert result["passed"] == 1
    with pytest.raises(DirectionAdjudicationProvenanceError, match="manifest"):
        verify_resume_compatibility(
            run, expected_bindings={"source_fingerprint": "changed"}
        )

