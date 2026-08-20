from __future__ import annotations

import hashlib
import json
import shutil
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import mnist.d0_jacobi_rb_absolute_coordinate_provenance as provenance
from mnist.d0_jacobi_artifacts import config_fingerprint, file_fingerprint
from mnist.d0_jacobi_rb_absolute_coordinate_provenance import (
    AbsoluteCoordinateProvenanceError,
    COARSE_WITNESS_BASENAME,
    COARSE_WITNESS_REGISTRY_FILE_SHA256,
    COARSE_WITNESS_REGISTRY_SEMANTIC_SHA256,
    PORTABLE_RESULT_ARCHIVE_SHA256,
    PORTABLE_RESULT_BASENAME,
    PORTABLE_RESULT_CONFIG_SHA256,
    PORTABLE_RESULT_DECISION,
    PORTABLE_RESULT_REGISTRY_FILE_SHA256,
    PORTABLE_RESULT_REGISTRY_SEMANTIC_SHA256,
    CoarsePanelSpec,
    compare_coarse_witness_snapshots,
    compare_portable_result_snapshots,
    load_verified_coarse_witness_panels,
    snapshot_absolute_coordinate_parents,
    snapshot_coarse_witness_run,
    snapshot_portable_result_archive,
    verify_absolute_coordinate_parent_immutability,
    verify_absolute_coordinate_parents,
    verify_coarse_witness_run,
    verify_portable_result_archive,
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _semantic(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


ArchivePayloadMutator = Callable[[dict[str, Any]], None]
ArchiveRegistryMutator = Callable[[dict[str, Any]], None]


def _make_archive(
    tmp_path: Path,
    *,
    payload_mutator: ArchivePayloadMutator | None = None,
    registry_mutator: ArchiveRegistryMutator | None = None,
) -> tuple[Path, provenance.PortableResultSpec]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = provenance.PORTABLE_RESULT_SPEC
    root_name = "portable-root"
    archive_path = tmp_path / "portable-result.zip"
    config = _semantic(
        {
            "schema": base.config_schema,
            "schema_version": 1,
            "authorizing": 0,
            "training_authorized": 0,
        }
    )
    manifest = _semantic(
        {
            "schema": base.manifest_schema,
            "schema_version": 1,
            "scientific_config_sha256": config["semantic_sha256"],
            "training_authorized": 0,
        }
    )
    status = {
        "schema": base.status_schema,
        "schema_version": 1,
        "state": "valid_scientific_stop",
        "stage": "report",
        "decision": base.terminal_decision,
        "failure_domain": "scientific_gate",
        "failure_code": base.terminal_decision,
        "scientific_evidence_complete": 1,
        "training_authorized": 0,
    }
    decision = {
        "schema": base.decision_schema,
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "decision": base.terminal_decision,
        "terminal": 1,
        "scientific_evidence_complete": 1,
        "invalid_evidence": 0,
        "valid_scientific_stop": 1,
        "unique_representation_identified": 0,
        "training_authorized": 0,
    }
    payload_objects: dict[str, Any] = {
        "run_manifest.json": manifest,
        "scientific_config.json": config,
        "run_status.json": status,
        "quartile_directional_adjudication_decision.json": decision,
    }
    if payload_mutator is not None:
        payload_mutator(payload_objects)
    payloads: dict[str, bytes] = {
        name: _json_bytes(value) for name, value in payload_objects.items()
    }
    payloads["extra.bin"] = b"sealed-extra-payload"
    artifacts = [
        {"path": name, "sha256": _sha_bytes(payload), "size": len(payload)}
        for name, payload in sorted(payloads.items())
    ]
    registry = _semantic(
        {
            "schema": base.registry_schema,
            "schema_version": 1,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "training_authorized": 0,
        }
    )
    if registry_mutator is not None:
        registry_mutator(registry)
    registry_bytes = _json_bytes(registry)
    all_payloads = {**payloads, "artifact_registry.json": registry_bytes}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{root_name}/", b"")
        for name, payload in sorted(all_payloads.items()):
            archive.writestr(f"{root_name}/{name}", payload)
    manifest_value = payload_objects["run_manifest.json"]
    config_value = payload_objects["scientific_config.json"]
    spec = replace(
        base,
        basename=archive_path.name,
        archive_sha256=file_fingerprint(archive_path),
        archive_size=archive_path.stat().st_size,
        root_name=root_name,
        entry_count=len(all_payloads) + 1,
        file_count=len(all_payloads),
        directory_paths=("",),
        total_uncompressed_bytes=sum(len(value) for value in all_payloads.values()),
        registry_artifact_count=len(artifacts),
        registry_semantic_sha256=str(registry["semantic_sha256"]),
        registry_file_sha256=_sha_bytes(registry_bytes),
        manifest_semantic_sha256=str(manifest_value["semantic_sha256"]),
        config_semantic_sha256=str(config_value["semantic_sha256"]),
    )
    return archive_path, spec


def _append_archive_member(
    path: Path,
    spec: provenance.PortableResultSpec,
    member_name: str,
    payload: bytes,
) -> provenance.PortableResultSpec:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(member_name, payload)
    return replace(
        spec,
        archive_sha256=file_fingerprint(path),
        archive_size=path.stat().st_size,
        entry_count=spec.entry_count + 1,
        file_count=spec.file_count + 1,
        total_uncompressed_bytes=spec.total_uncompressed_bytes + len(payload),
    )


def _corrupt_stored_member(path: Path, member_name: str) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        info = archive.getinfo(member_name)
        offset = info.header_offset
    payload = bytearray(path.read_bytes())
    name_length = int.from_bytes(payload[offset + 26 : offset + 28], "little")
    extra_length = int.from_bytes(payload[offset + 28 : offset + 30], "little")
    data_offset = offset + 30 + name_length + extra_length
    payload[data_offset] ^= 0x01
    path.write_bytes(payload)


def _resign_registry(registry: dict[str, Any]) -> None:
    registry.pop("semantic_sha256", None)
    registry["semantic_sha256"] = config_fingerprint(registry)


def _make_coarse_witness(
    tmp_path: Path,
) -> tuple[Path, provenance.CoarseWitnessSpec, dict[str, np.ndarray]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = provenance.COARSE_WITNESS_SPEC
    run = tmp_path / "coarse-fixture"
    run.mkdir()
    run_schema = "fixture-physical-coarse-witness"
    path_plan = "1" * 64
    statistic_plan = "2" * 64
    source_fingerprint = "3" * 64
    raw_panels: dict[str, np.ndarray] = {}
    panel_specs: list[CoarsePanelSpec] = []
    seals: dict[str, dict[str, Any]] = {}
    for panel_index, panel in enumerate(("a", "b")):
        path_ids = np.asarray(
            [100 + 10 * panel_index, 101 + 10 * panel_index], dtype=np.int64
        )
        means = (
            np.arange(6, dtype=np.float64).reshape(2, 1, 1, 3)
            + 0.25
            + panel_index
        )
        raw_panels[panel] = means.copy()
        relative = f"panels/{panel}/cell_means.npz"
        data_path = run / relative
        data_path.parent.mkdir(parents=True)
        np.savez_compressed(data_path, path_ids=path_ids, cell_means=means)
        provisional = CoarsePanelSpec(
            panel=panel,
            relative_path=relative,
            file_size=data_path.stat().st_size,
            file_sha256=file_fingerprint(data_path),
            array_sha256=provenance._array_sha256(means),
            panel_fingerprint="",
            seal_file_sha256="",
            path_ids=tuple(int(value) for value in path_ids),
            shape=means.shape,
        )
        provisional = replace(
            provisional,
            panel_fingerprint=provenance._panel_fingerprint(
                provisional, path_ids, means
            ),
        )
        metrics_path = run / f"panels/{panel}/metrics.json"
        audit_path = run / f"panel_{panel}_cell_mean_persistence_audit.json"
        resource_path = run / f"panel_{panel}_resource_summary.json"
        _write_json(metrics_path, {"panel": panel, "complete": 1})
        _write_json(audit_path, {"panel": panel, "passed": 1})
        _write_json(resource_path, {"panel": panel, "passed": 1})
        seal = {
            "schema": f"{run_schema}-panel-seal",
            "schema_version": 1,
            "panel": panel,
            "path_ids": path_ids.tolist(),
            "cell_means_file": relative,
            "cell_means_file_sha256": provisional.file_sha256,
            "cell_means_array_sha256": provisional.array_sha256,
            "panel_fingerprint": provisional.panel_fingerprint,
            "path_plan_sha256": path_plan,
            "statistic_plan_sha256": statistic_plan,
            "execution_metrics_file_sha256": file_fingerprint(metrics_path),
            "persistence_audit_file_sha256": file_fingerprint(audit_path),
            "resource_summary_file_sha256": file_fingerprint(resource_path),
            "analysis_opened": 0,
            "physical_training_performed": 0,
        }
        seal_path = run / f"panel_{panel}_seal.json"
        _write_json(seal_path, seal)
        panel_specs.append(
            replace(provisional, seal_file_sha256=file_fingerprint(seal_path))
        )
        seals[panel] = seal

    joint = {
        "schema": f"{run_schema}-joint-analysis-seal",
        "schema_version": 1,
        "panel_a_seal_sha256": config_fingerprint(seals["a"]),
        "panel_b_seal_sha256": config_fingerprint(seals["b"]),
        "panel_a_file_sha256": panel_specs[0].file_sha256,
        "panel_b_file_sha256": panel_specs[1].file_sha256,
        "statistic_plan_sha256": statistic_plan,
        "analysis_definition_frozen_before_open": 1,
        "analysis_open_count": 0,
        "physical_training_performed": 0,
    }
    joint_path = run / "joint_analysis_seal.json"
    _write_json(joint_path, joint)
    config = _semantic(
        {
            "schema": f"{run_schema}-scientific-config",
            "schema_version": 1,
            "path_plan_sha256": path_plan,
            "statistic_plan_sha256": statistic_plan,
            "physical_training_performed": 0,
        }
    )
    _write_json(run / "scientific_config.json", config)
    _write_json(
        run / "run_manifest.json",
        {
            "schema": run_schema,
            "schema_version": 1,
            "source_fingerprint": source_fingerprint,
            "scientific_config_sha256": config["semantic_sha256"],
            "path_plan_sha256": path_plan,
            "statistic_plan_sha256": statistic_plan,
            "physical_training_performed": 0,
        },
    )
    _write_json(
        run / "physical_coarse_signal_decision.json",
        {
            "schema": base.decision_schema,
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "decision": base.terminal_decision,
            "scientific_outcome": base.terminal_decision,
            "full_state_conditional_mean_zero_proven": 0,
            "physical_training_performed": 0,
        },
    )
    for filename, gate_name in (
        ("coarse_signal_preflight_gate.json", "preflight"),
        ("coarse_signal_panel_a_gate.json", "panel-a"),
        ("coarse_signal_panel_b_gate.json", "panel-b"),
        ("coarse_signal_witness_gate.json", "witness"),
    ):
        _write_json(
            run / filename,
            {
                "schema": base.gate_schema,
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "gate": gate_name,
                "passed": 1,
                "physical_training_performed": 0,
            },
        )
    _write_json(
        run / "parent_provenance.json",
        {
            "schema": "fixture-parent-provenance",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": 1,
            "physical_training_performed": 0,
        },
    )
    spec = replace(
        base,
        basename=run.name,
        run_schema=run_schema,
        config_schema=f"{run_schema}-scientific-config",
        config_semantic_sha256=str(config["semantic_sha256"]),
        source_fingerprint=source_fingerprint,
        path_plan_sha256=path_plan,
        statistic_plan_sha256=statistic_plan,
        panels=tuple(panel_specs),
        joint_seal_file_sha256=file_fingerprint(joint_path),
    )
    spec = _rebind_coarse_registry(run, spec)
    return run, spec, raw_panels


def _rebind_coarse_registry(
    run: Path, spec: provenance.CoarseWitnessSpec
) -> provenance.CoarseWitnessSpec:
    artifacts = sorted(
        path
        for path in run.rglob("*")
        if path.is_file() and path.name not in {"artifact_registry.json", "run_status.json"}
    )
    rows = [
        {
            "path": path.relative_to(run).as_posix(),
            "sha256": file_fingerprint(path),
            "size": path.stat().st_size,
        }
        for path in artifacts
    ]
    registry = {
        "schema": spec.registry_schema,
        "schema_version": 1,
        "record_count": len(rows),
        "records": rows,
        "registry_sha256": config_fingerprint(rows),
        "physical_training_performed": 0,
    }
    registry_path = run / "artifact_registry.json"
    _write_json(registry_path, registry)
    registry_file_sha = file_fingerprint(registry_path)
    status = {
        "schema": f"{spec.run_schema}-status",
        "schema_version": 1,
        "state": "completed",
        "stage": "analyze",
        "decision": spec.terminal_decision,
        "artifact_registry_record_count": len(rows),
        "artifact_registry_sha256": registry["registry_sha256"],
        "artifact_registry_file_sha256": registry_file_sha,
        "artifact_registry_file_size": registry_path.stat().st_size,
        "physical_training_performed": 0,
    }
    status_path = run / "run_status.json"
    _write_json(status_path, status)
    files = [path for path in run.rglob("*") if path.is_file()]
    return replace(
        spec,
        registry_record_count=len(rows),
        registry_semantic_sha256=str(registry["registry_sha256"]),
        registry_file_sha256=registry_file_sha,
        registry_file_size=registry_path.stat().st_size,
        status_file_sha256=file_fingerprint(status_path),
        expected_file_count=len(files),
        expected_total_bytes=sum(path.stat().st_size for path in files),
    )


def test_frozen_production_bindings_are_exact() -> None:
    assert PORTABLE_RESULT_BASENAME.endswith("directional-continuation.zip")
    assert PORTABLE_RESULT_ARCHIVE_SHA256 == (
        "0f9914b79011a1182bac8fd9645e7ac0e222618d5be92047c03268e8b9ab3f7d"
    )
    assert PORTABLE_RESULT_REGISTRY_SEMANTIC_SHA256 == (
        "cf206b49a094ede6196fd794f945c8ecf616e3caf48ef12b32c31afc8cafea64"
    )
    assert PORTABLE_RESULT_REGISTRY_FILE_SHA256 == (
        "3fe04ff3d4a8a5231f6588b8383610e8d283492774f0e7bbe2146824550f50b6"
    )
    assert PORTABLE_RESULT_CONFIG_SHA256 == (
        "00f2464129b4c4dcfbd727aed97173abcc59e0e29697b77bafa76d1d28c0d39e"
    )
    assert PORTABLE_RESULT_DECISION == "representation_cancellation_nonidentifying_stop"
    assert COARSE_WITNESS_BASENAME.endswith("physical-coarse-signal-jsonfix")
    assert COARSE_WITNESS_REGISTRY_SEMANTIC_SHA256 == (
        "ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3"
    )
    assert COARSE_WITNESS_REGISTRY_FILE_SHA256 == (
        "866984822ef14dbb13f0644a0f23697f6fb42ecec40b07341249d74043319747"
    )


def test_portable_archive_is_verified_without_root_dependence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, spec = _make_archive(tmp_path / "first")
    monkeypatch.setattr(provenance, "PORTABLE_RESULT_SPEC", spec)
    first = verify_portable_result_archive(archive)
    relocated = tmp_path / "other-root" / archive.name
    relocated.parent.mkdir()
    shutil.copyfile(archive, relocated)
    second = verify_portable_result_archive(relocated)
    assert first == second
    assert first["passed"] == 1
    assert first["safe_relative_paths"] == 1
    assert first["casefold_unique_paths"] == 1
    assert first["all_archive_crcs_verified"] == 1
    assert first["all_registered_artifact_hashes_verified"] == 1
    assert first["terminal"]["decision"] == PORTABLE_RESULT_DECISION


@pytest.mark.parametrize("kind", ["traversal", "duplicate", "case_collision"])
def test_portable_archive_rejects_unsafe_or_colliding_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    archive, spec = _make_archive(tmp_path)
    root = spec.root_name
    member = {
        "traversal": f"{root}/../escape.bin",
        "duplicate": f"{root}/extra.bin",
        "case_collision": f"{root}/EXTRA.bin",
    }[kind]
    spec = _append_archive_member(archive, spec, member, b"x")
    monkeypatch.setattr(provenance, "PORTABLE_RESULT_SPEC", spec)
    match = {
        "traversal": "unsafe archive member path",
        "duplicate": "duplicated",
        "case_collision": "case collision",
    }[kind]
    with pytest.raises(AbsoluteCoordinateProvenanceError, match=match):
        verify_portable_result_archive(archive)


def test_portable_archive_reads_every_member_and_rejects_bad_crc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, spec = _make_archive(tmp_path)
    _corrupt_stored_member(archive, f"{spec.root_name}/extra.bin")
    monkeypatch.setattr(provenance, "PORTABLE_RESULT_SPEC", spec)
    with pytest.raises(AbsoluteCoordinateProvenanceError, match="CRC or payload"):
        verify_portable_result_archive(archive)


@pytest.mark.parametrize("field", ["size", "sha256"])
def test_portable_registry_row_size_and_sha_are_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    def mutate(registry: dict[str, Any]) -> None:
        if field == "size":
            registry["artifacts"][0]["size"] += 1
        else:
            registry["artifacts"][0]["sha256"] = "0" * 64
        _resign_registry(registry)

    archive, spec = _make_archive(tmp_path, registry_mutator=mutate)
    monkeypatch.setattr(provenance, "PORTABLE_RESULT_SPEC", spec)
    with pytest.raises(
        AbsoluteCoordinateProvenanceError,
        match=f"artifact {field.replace('sha256', 'hash')} changed",
    ):
        verify_portable_result_archive(archive)


def test_portable_registry_semantic_commitment_is_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(registry: dict[str, Any]) -> None:
        registry["semantic_sha256"] = "0" * 64

    archive, spec = _make_archive(tmp_path, registry_mutator=mutate)
    monkeypatch.setattr(provenance, "PORTABLE_RESULT_SPEC", spec)
    with pytest.raises(AbsoluteCoordinateProvenanceError, match="semantic hash"):
        verify_portable_result_archive(archive)


@pytest.mark.parametrize("kind", ["manifest", "config", "terminal"])
def test_portable_manifest_config_and_terminal_bindings_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    def mutate(payloads: dict[str, Any]) -> None:
        if kind == "manifest":
            manifest = payloads["run_manifest.json"]
            manifest["scientific_config_sha256"] = "0" * 64
            payloads["run_manifest.json"] = _semantic(manifest)
        elif kind == "config":
            payloads["scientific_config.json"]["changed"] = 1
        else:
            payloads["run_status.json"]["decision"] = "changed-terminal"
            payloads["quartile_directional_adjudication_decision.json"][
                "decision"
            ] = "changed-terminal"

    archive, spec = _make_archive(tmp_path, payload_mutator=mutate)
    monkeypatch.setattr(provenance, "PORTABLE_RESULT_SPEC", spec)
    match = {
        "manifest": "manifest/config binding",
        "config": "config semantic hash",
        "terminal": "terminal status",
    }[kind]
    with pytest.raises(AbsoluteCoordinateProvenanceError, match=match):
        verify_portable_result_archive(archive)


def test_coarse_witness_registry_seals_and_panels_are_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, spec, raw = _make_coarse_witness(tmp_path)
    monkeypatch.setattr(provenance, "COARSE_WITNESS_SPEC", spec)
    result = verify_coarse_witness_run(run)
    assert result["passed"] == 1
    assert result["all_registered_artifact_hashes_verified"] == 1
    assert result["all_panel_hashes_verified"] == 1
    assert result["terminal"]["decision"] == spec.terminal_decision
    assert set(result["gates"]) == {"preflight", "panel-a", "panel-b", "witness"}
    panels = load_verified_coarse_witness_panels(run)
    np.testing.assert_array_equal(panels.panel_a, raw["a"])
    np.testing.assert_array_equal(panels.panel_b, raw["b"])
    assert not panels.panel_a.flags.writeable
    assert not panels.panel_b.flags.writeable
    with pytest.raises(ValueError):
        panels.panel_a.flat[0] = 0.0


def test_coarse_witness_registered_artifact_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, spec, _ = _make_coarse_witness(tmp_path)
    monkeypatch.setattr(provenance, "COARSE_WITNESS_SPEC", spec)
    target = run / "panels/a/metrics.json"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 1
    target.write_bytes(payload)
    with pytest.raises(AbsoluteCoordinateProvenanceError, match="artifact hash changed"):
        verify_coarse_witness_run(run)


def test_coarse_witness_joint_seal_cross_binding_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, spec, _ = _make_coarse_witness(tmp_path)
    joint_path = run / "joint_analysis_seal.json"
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    joint["panel_a_seal_sha256"] = "0" * 64
    _write_json(joint_path, joint)
    spec = replace(spec, joint_seal_file_sha256=file_fingerprint(joint_path))
    spec = _rebind_coarse_registry(run, spec)
    monkeypatch.setattr(provenance, "COARSE_WITNESS_SPEC", spec)
    with pytest.raises(AbsoluteCoordinateProvenanceError, match="joint analysis seal"):
        verify_coarse_witness_run(run)


def test_root_independent_snapshots_compare_and_detect_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, archive_spec = _make_archive(tmp_path / "archive-a")
    run, coarse_spec, _ = _make_coarse_witness(tmp_path / "coarse-a")
    monkeypatch.setattr(provenance, "PORTABLE_RESULT_SPEC", archive_spec)
    monkeypatch.setattr(provenance, "COARSE_WITNESS_SPEC", coarse_spec)
    archive_copy = tmp_path / "archive-b" / archive.name
    archive_copy.parent.mkdir()
    shutil.copyfile(archive, archive_copy)
    run_copy = tmp_path / "coarse-b" / run.name
    shutil.copytree(run, run_copy)
    archive_before = snapshot_portable_result_archive(archive)
    archive_after = snapshot_portable_result_archive(archive_copy)
    coarse_before = snapshot_coarse_witness_run(run)
    coarse_after = snapshot_coarse_witness_run(run_copy)
    assert archive_before == archive_after
    assert coarse_before == coarse_after
    assert compare_portable_result_snapshots(archive_before, archive_after)["passed"] == 1
    assert compare_coarse_witness_snapshots(coarse_before, coarse_after)["passed"] == 1
    target = run_copy / "panels/a/metrics.json"
    changed = bytearray(target.read_bytes())
    changed[-2] ^= 1
    target.write_bytes(changed)
    changed_snapshot = snapshot_coarse_witness_run(run_copy)
    with pytest.raises(AbsoluteCoordinateProvenanceError, match="snapshot changed"):
        compare_coarse_witness_snapshots(coarse_before, changed_snapshot)


def test_combined_parent_api_and_before_after_immutability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, archive_spec = _make_archive(tmp_path / "archive")
    run, coarse_spec, _ = _make_coarse_witness(tmp_path / "coarse")
    monkeypatch.setattr(provenance, "PORTABLE_RESULT_SPEC", archive_spec)
    monkeypatch.setattr(provenance, "COARSE_WITNESS_SPEC", coarse_spec)
    snapshots = snapshot_absolute_coordinate_parents(
        portable_zip_path=archive, coarse_witness_run_dir=run
    )
    result = verify_absolute_coordinate_parents(
        portable_zip_path=archive,
        coarse_witness_run_dir=run,
        snapshots=snapshots,
    )
    assert result["passed"] == result["provenance_valid"] == 1
    assert result["portable_directional_parent_valid"] == 1
    assert result["coarse_witness_parent_valid"] == 1
    assert result["parent_files_modified"] == 0
    immutability = verify_absolute_coordinate_parent_immutability(
        portable_zip_path=archive,
        coarse_witness_run_dir=run,
        snapshots=snapshots,
    )
    assert immutability["passed"] == 1
    assert immutability["parent_files_modified"] == 0


def test_real_exact_parents_when_available() -> None:
    archive = Path.home() / "Downloads" / PORTABLE_RESULT_BASENAME
    coarse = (
        Path("runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness")
        / COARSE_WITNESS_BASENAME
    )
    if not archive.is_file() or not coarse.is_dir():
        pytest.skip("exact local parent artifacts are not installed")
    portable = verify_portable_result_archive(archive)
    witness = verify_coarse_witness_run(coarse)
    assert portable["registry"]["artifact_count"] == 2_041
    assert portable["terminal"]["decision"] == PORTABLE_RESULT_DECISION
    assert witness["registry"]["record_count"] == 2_616
    assert witness["panels"]["a"]["shape"] == [64, 4, 7, 392]
    assert witness["panels"]["b"]["shape"] == [64, 4, 7, 392]
