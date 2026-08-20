from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from mnist import d0_jacobi_rb_quartile_directional_portable as portable
from mnist.d0_jacobi_artifacts import config_fingerprint
from mnist.d0_jacobi_rb_quartile_directional_adjudication_gate import (
    ZERO_AUTHORIZATION_FIELDS,
    ZERO_WORK_FIELDS,
)


READY_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_"
    "quartile_directional_adjudication/"
    "20260808-203454_production-read-only-quartile-directional-"
    "adjudication-bootstrap-fix"
)
SPECIALIST_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist/"
    "20260807-132351_production-exact-quartile-specialist"
)


def _semantic(body: dict[str, object]) -> dict[str, object]:
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _copy_ready_run(destination: Path) -> Path:
    if not READY_RUN.is_dir():
        pytest.skip("the immutable ready-for-fittrace run is not available")
    target = destination / portable.PREDECESSOR_BASENAME
    shutil.copytree(READY_RUN, target)
    return target


def _copy_legacy_sources(predecessor: Path, destination: Path) -> Path:
    closure = json.loads((predecessor / "source_closure.json").read_text("utf-8"))
    root = destination / "relocated-repository"
    for row in closure["sources"]:
        normalized = str(row["path"]).replace("\\", "/")
        relative = "mnist/" + normalized.rsplit("/mnist/", 1)[1]
        source = Path(relative)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


@pytest.mark.skipif(not READY_RUN.is_dir(), reason="production evidence unavailable")
def test_exact_ready_predecessor_registry_and_inventory_are_bound() -> None:
    result = portable.verify_ready_predecessor(READY_RUN)
    assert result["passed"] == 1
    assert result["artifact_count"] == portable.PREDECESSOR_ARTIFACT_COUNT == 26
    assert result["registry_semantic_sha256"] == (
        portable.PREDECESSOR_REGISTRY_SEMANTIC_SHA256
    )
    assert result["registry_file_sha256"] == (
        portable.PREDECESSOR_REGISTRY_FILE_SHA256
    )
    assert result["decision"] == "ready_for_fittrace"
    assert result["later_stage_evidence_opened"] == 0
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert result[field] == 0


def test_predecessor_copy_rejects_extra_and_registered_file_tampering(
    tmp_path: Path,
) -> None:
    copied = _copy_ready_run(tmp_path)
    assert portable.verify_ready_predecessor(copied)["passed"] == 1

    extra = copied / "unregistered.txt"
    extra.write_text("not sealed", encoding="utf-8")
    with pytest.raises(portable.PortableContinuationError, match="inventory"):
        portable.verify_ready_predecessor(copied)
    extra.unlink()

    status = copied / "run_status.json"
    status.write_bytes(status.read_bytes() + b" ")
    with pytest.raises(portable.PortableContinuationError, match="registered"):
        portable.verify_ready_predecessor(copied)


def test_legacy_source_closure_is_root_independent_but_byte_exact(
    tmp_path: Path,
) -> None:
    predecessor = _copy_ready_run(tmp_path / "evidence")
    relocated_repo = _copy_legacy_sources(predecessor, tmp_path)

    result = portable.verify_legacy_source_closure(
        predecessor, repo_root=relocated_repo
    )
    assert result["passed"] == 1
    assert result["source_count"] == portable.LEGACY_SOURCE_COUNT == 37
    assert result["legacy_source_fingerprint"] == (
        portable.PREDECESSOR_SOURCE_FINGERPRINT
    )
    assert len(result["content_fingerprint"]) == 64
    assert all(not str(row["path"]).startswith(("C:", "/")) for row in result["sources"])
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert result[field] == 0

    victim = relocated_repo / str(result["sources"][0]["path"])
    victim.write_bytes(victim.read_bytes() + b"# tampered\n")
    with pytest.raises(portable.PortableContinuationError, match="legacy source"):
        portable.verify_legacy_source_closure(predecessor, repo_root=relocated_repo)


def _snapshot(root: str, *, changed: bool = False) -> dict[str, object]:
    rows = [
        {
            "path": "artifact_registry.json",
            "size": 10,
            "sha256": ("b" if changed else "a") * 64,
        }
    ]
    body: dict[str, object] = {
        "schema": "fixture-parent-tree-snapshot",
        "schema_version": 1,
        "run_dir": root,
        "parent_basename": Path(root).name,
        "file_count": 1,
        "total_bytes": 10,
        "files": rows,
        "tree_sha256": config_fingerprint(rows),
    }
    return _semantic(body)


def test_relocated_parent_comparison_ignores_only_run_dir_and_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = tmp_path / portable.PREDECESSOR_BASENAME
    predecessor.mkdir()
    specialist = tmp_path / "linux-mount" / portable.SPECIALIST_PARENT_BASENAME
    time_local = tmp_path / "other-mount" / portable.TIME_LOCAL_PARENT_BASENAME
    specialist.mkdir(parents=True)
    time_local.mkdir(parents=True)
    saved_specialist = _snapshot(
        "C:/old/runs/" + portable.SPECIALIST_PARENT_BASENAME
    )
    saved_time_local = _snapshot(
        "C:/old/runs/" + portable.TIME_LOCAL_PARENT_BASENAME
    )
    parent_record = _semantic(
        {
            "schema": "fixture-parent-snapshots",
            "schema_version": 1,
            "quartile_specialist": saved_specialist,
            "time_local": saved_time_local,
            **{field: 0 for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS},
        }
    )
    (predecessor / "parent_immutability_before.json").write_text(
        json.dumps(parent_record), encoding="utf-8"
    )

    changed = {"specialist": False}

    def observed(root: Path) -> dict[str, object]:
        is_specialist = Path(root).name == portable.SPECIALIST_PARENT_BASENAME
        return _snapshot(
            str(Path(root).resolve()),
            changed=is_specialist and changed["specialist"],
        )

    monkeypatch.setattr(portable, "snapshot_parent_run", observed)
    result = portable.verify_relocated_parent_snapshots(
        predecessor,
        specialist_run_dir=specialist,
        time_local_run_dir=time_local,
    )
    assert result["passed"] == 1
    assert result["ignored_fields"] == ["run_dir", "semantic_sha256"]
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert result[field] == 0

    changed["specialist"] = True
    with pytest.raises(portable.PortableContinuationError, match="specialist tree"):
        portable.verify_relocated_parent_snapshots(
            predecessor,
            specialist_run_dir=specialist,
            time_local_run_dir=time_local,
        )


@pytest.mark.skipif(not SPECIALIST_RUN.is_dir(), reason="specialist evidence unavailable")
def test_portable_cache_lookup_ignores_historical_cache_root_but_not_payload_metadata(
    tmp_path: Path,
) -> None:
    relocated = tmp_path / portable.SPECIALIST_PARENT_BASENAME
    role = "gain_calibration"
    (relocated / "role_caches" / role / "eager_cache").mkdir(parents=True)
    shutil.copyfile(
        SPECIALIST_RUN / f"{role}_cache_binding.json",
        relocated / f"{role}_cache_binding.json",
    )
    for name in ("train_index.json", "train_validation_metrics.json"):
        shutil.copyfile(
            SPECIALIST_RUN / "role_caches" / role / "eager_cache" / name,
            relocated / "role_caches" / role / "eager_cache" / name,
        )

    binding, index = portable._portable_cache_binding(relocated, role)  # noqa: SLF001
    assert binding["role"] == role
    assert index["path_count"] == 32
    assert Path(str(binding["cache_root"])) != (
        relocated / "role_caches" / role
    )

    path = relocated / f"{role}_cache_binding.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(portable.PortableContinuationError, match="file hash"):
        portable._portable_cache_binding(relocated, role)  # noqa: SLF001


def test_portable_identity_contains_no_operational_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def record(kind: str) -> dict[str, object]:
        return _semantic(
            {
                "schema": f"fixture-{kind}",
                "schema_version": 1,
                "passed": 1,
                **{field: 0 for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS},
            }
        )

    monkeypatch.setattr(portable, "verify_ready_predecessor", lambda _root: record("run"))
    monkeypatch.setattr(
        portable,
        "verify_legacy_source_closure",
        lambda _root, repo_root=None: record("sources"),
    )
    monkeypatch.setattr(
        portable,
        "verify_relocated_parent_snapshots",
        lambda _root, **_kwargs: record("parents"),
    )
    result = portable.verify_portable_continuation(
        "/new/child",
        specialist_run_dir="/new/specialist",
        time_local_run_dir="/new/time-local",
        repo_root="/new/repo",
    )
    assert result["passed"] == 1
    assert result["root_paths_authorizing"] == 0
    assert result["ignored_historical_field_count"] == 2
    assert not any("/new/" in str(value) for value in result.values())
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert result[field] == 0
