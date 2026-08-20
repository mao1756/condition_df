from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint, file_fingerprint
from mnist.d0_jacobi_rb_quartile_directional_adjudication_provenance import (
    PHYSICAL_FIT_CACHE_BINDING_FILE_SHA256,
    PHYSICAL_FIT_PATH_COUNT,
    PHYSICAL_FIT_ROLE_INDEX_FILE_SHA256,
    PHYSICAL_FIT_ROW_COUNT,
    SPECIALIST_PARENT_BASENAME,
    SPECIALIST_PARENT_CONFIG_SHA256,
    SPECIALIST_PARENT_DECISION,
    SPECIALIST_PARENT_REGISTRY_COUNT,
    SPECIALIST_PARENT_REGISTRY_FILE_SHA256,
    SPECIALIST_PARENT_REGISTRY_SEMANTIC_SHA256,
    SPECIALIST_PARENT_SOURCE_FINGERPRINT,
    TIME_LOCAL_PARENT_BASENAME,
    TIME_LOCAL_PARENT_CONFIG_SHA256,
    TIME_LOCAL_PARENT_DECISION,
    TIME_LOCAL_PARENT_REGISTRY_COUNT,
    TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256,
    TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256,
    TIME_LOCAL_PARENT_SOURCE_FINGERPRINT,
    QuartileDirectionalProvenanceError,
    load_already_open_inputs,
    load_already_open_role,
    snapshot_parent_runs,
    source_fingerprint,
    source_paths,
    validate_semantic_config,
    verify_parent_immutability,
    verify_parents,
    verify_resume_compatibility,
)


def _parents() -> tuple[Path, Path]:
    specialist = (
        Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist")
        / SPECIALIST_PARENT_BASENAME
    ).resolve()
    time_local = (
        Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication"
        )
        / TIME_LOCAL_PARENT_BASENAME
    ).resolve()
    return specialist, time_local


@pytest.fixture(scope="module")
def parent_snapshots() -> tuple[Path, Path, dict[str, object]]:
    specialist, time_local = _parents()
    snapshots = snapshot_parent_runs(
        specialist_run_dir=specialist,
        time_local_run_dir=time_local,
    )
    return specialist, time_local, snapshots


def test_exact_parent_constants_and_two_parent_chain(
    parent_snapshots: tuple[Path, Path, dict[str, object]],
) -> None:
    specialist, time_local, snapshots = parent_snapshots
    assert SPECIALIST_PARENT_DECISION == "no_training_only_quartile_system"
    assert SPECIALIST_PARENT_REGISTRY_COUNT == 4_120
    assert SPECIALIST_PARENT_REGISTRY_SEMANTIC_SHA256 == (
        "e5f6b3ee257b3d4f86ec3ac54f4223540cf76caa24489d22e9c138a54e19c7bb"
    )
    assert SPECIALIST_PARENT_REGISTRY_FILE_SHA256 == (
        "e24c7db28081dbceb8f0acf690d779f835379c82a89a2b263976c3e0b631f798"
    )
    assert SPECIALIST_PARENT_SOURCE_FINGERPRINT == (
        "61a8c2fd6a317c05b9eed73e208d40b0cc6c01d6fdd227ae33d71d0be7c6027d"
    )
    assert SPECIALIST_PARENT_CONFIG_SHA256 == (
        "05263b7b01c2586e9a771bd71fe28fbb74d8e5d9da873ce4746019c5167c08c1"
    )
    assert TIME_LOCAL_PARENT_DECISION == "exact_rb_high_reverse_time_only_signal"
    assert TIME_LOCAL_PARENT_REGISTRY_COUNT == 29
    assert TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256 == (
        "b25256d606f1fea2c9ef78ab5f14a7b8ccd67bc6f5c234bd2ed2a1a0086fd9f5"
    )
    assert TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256 == (
        "15220d3f4ee3e7a4740fd5fae2695e1da1d0b1ea91ee05c70357c3a152569a64"
    )
    assert TIME_LOCAL_PARENT_SOURCE_FINGERPRINT == (
        "55f259f30ecb1eb47915a44d3ba67a353bac87abd87743f917c55bbcb06a0123"
    )
    assert TIME_LOCAL_PARENT_CONFIG_SHA256 == (
        "faf395317449a842e63de0807d39102f68d7afa49c7700e9cd6c94e0d381b009"
    )

    result = verify_parents(
        specialist_run_dir=specialist,
        time_local_run_dir=time_local,
        snapshots=snapshots,
        verify_checkpoint_states=False,
        verify_cache_rows=False,
        verify_external_cache=False,
    )
    assert result["passed"] == 1
    assert result["all_explicit_parent_registries_verified"] == 1
    assert result["all_transitive_parent_registries_verified"] == 1
    assert result["all_registered_artifact_hashes_verified"] == 1
    assert result["historical_design_evidence_authorizing"] == 0
    assert result["selection_paths_opened"] == 0
    assert result["confirmation_paths_opened"] == 0
    assert result["parents_mutated"] == 0


def test_parent_snapshots_are_complete_and_detect_change(
    parent_snapshots: tuple[Path, Path, dict[str, object]],
) -> None:
    specialist, time_local, snapshots = parent_snapshots
    result = verify_parent_immutability(
        specialist_run_dir=specialist,
        time_local_run_dir=time_local,
        snapshots=snapshots,
    )
    assert result["passed"] == 1
    assert result["parents_mutated"] == 0

    changed = deepcopy(snapshots)
    changed["specialist"]["files"][0]["size"] += 1  # type: ignore[index]
    changed["specialist"]["total_bytes"] += 1  # type: ignore[index,operator]
    changed["specialist"]["tree_sha256"] = config_fingerprint(  # type: ignore[index]
        changed["specialist"]["files"]  # type: ignore[index]
    )
    changed["specialist"].pop("semantic_sha256")  # type: ignore[union-attr]
    changed["specialist"]["semantic_sha256"] = config_fingerprint(  # type: ignore[index]
        changed["specialist"]  # type: ignore[index]
    )
    changed.pop("semantic_sha256")
    changed["semantic_sha256"] = config_fingerprint(changed)
    with pytest.raises(QuartileDirectionalProvenanceError, match="snapshot changed"):
        verify_parent_immutability(
            specialist_run_dir=specialist,
            time_local_run_dir=time_local,
            snapshots=changed,
        )


def test_input_only_loader_never_deserializes_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specialist, _ = _parents()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("label loader was called")

    monkeypatch.setattr(
        "mnist.d0_jacobi_rb_quartile_directional_adjudication_provenance."
        "_load_eager_role_labels",
        forbidden,
    )
    loaded = load_already_open_inputs(specialist, "gain_calibration")
    assert loaded.role == "gain_calibration"
    assert len(loaded.inputs["sample_key"]) == 57_344
    assert all(not array.flags.writeable for array in loaded.inputs.values())
    with pytest.raises(TypeError):
        loaded.inputs["new"] = loaded.inputs["sample_key"]  # type: ignore[index]
    del loaded
    gc.collect()


def test_physical_fit_role_loader_and_exact_cache_bindings() -> None:
    specialist, _ = _parents()
    protected = {
        path: file_fingerprint(path)
        for path in (
            specialist / "physical_fit_cache_binding.json",
            specialist / "fit_label_open.json",
        )
    }
    loaded = load_already_open_role(specialist, "physical_fit")
    assert PHYSICAL_FIT_ROW_COUNT == 114_688
    assert PHYSICAL_FIT_PATH_COUNT == 64
    assert PHYSICAL_FIT_CACHE_BINDING_FILE_SHA256 == (
        "bb299e98009d4e5000162f8dd826416b41fe168cc84b6c87c6c612919c375c5d"
    )
    assert PHYSICAL_FIT_ROLE_INDEX_FILE_SHA256 == (
        "81bc267c715894a211059e111002cfb8133e40667cbae2e62fcb62b6cf47f57d"
    )
    assert loaded.role == "physical_fit"
    assert len(loaded.inputs["sample_key"]) == PHYSICAL_FIT_ROW_COUNT
    assert len(loaded.labels["sample_key"]) == PHYSICAL_FIT_ROW_COUNT
    assert len(loaded.row_identity_sha256) == 64
    assert all(not array.flags.writeable for array in loaded.inputs.values())
    assert all(not array.flags.writeable for array in loaded.labels.values())
    with pytest.raises(ValueError):
        loaded.labels["denoising_target"].flat[0] = 0.0
    assert protected == {path: file_fingerprint(path) for path in protected}
    del loaded
    gc.collect()


def test_role_loader_rejects_unopened_or_forbidden_role(tmp_path: Path) -> None:
    fake = tmp_path / SPECIALIST_PARENT_BASENAME
    fake.mkdir()
    with pytest.raises(QuartileDirectionalProvenanceError, match="forbidden historical role"):
        load_already_open_inputs(fake, "fresh_selection")
    assert not (fake / "selection_open.json").exists()
    with pytest.raises(QuartileDirectionalProvenanceError, match="missing parent path-id plan"):
        load_already_open_role(fake, "training_rank")
    assert not (fake / "rank_label_open.json").exists()


def test_source_config_and_resume_bindings_fail_closed(tmp_path: Path) -> None:
    entries = (
        Path("mnist/d0_jacobi_rb_quartile_directional_adjudication_provenance.py"),
        Path("mnist/d0_jacobi_rb_boundary_tangent_quartile_gate.py"),
    )
    assert all(path.is_file() for path in source_paths(entries))
    assert source_fingerprint(entries) == source_fingerprint(entries)

    config = {
        "schema": "directional-test-config",
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
        expected_schema="directional-test-config",
        expected_sha256=config["semantic_sha256"],
    )["passed"] == 1
    changed = dict(config)
    changed["training_authorized"] = 1
    changed["semantic_sha256"] = config_fingerprint(
        {key: value for key, value in changed.items() if key != "semantic_sha256"}
    )
    with pytest.raises(QuartileDirectionalProvenanceError, match="authorizes"):
        validate_semantic_config(changed)

    run = tmp_path / "resume"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"source_fingerprint": "abc", "scientific_config_sha256": "def"}),
        encoding="utf-8",
    )
    binary = run / "shard.npz"
    binary.write_bytes(b"sealed")
    result = verify_resume_compatibility(
        run,
        expected_bindings={"source_fingerprint": "abc"},
        artifact_bindings={"shard.npz": file_fingerprint(binary)},
    )
    assert result["passed"] == 1
    with pytest.raises(QuartileDirectionalProvenanceError, match="manifest"):
        verify_resume_compatibility(
            run, expected_bindings={"source_fingerprint": "changed"}
        )
