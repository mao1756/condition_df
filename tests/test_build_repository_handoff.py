from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from tools import build_repository_handoff as handoff


def _write(path: Path, payload: bytes = b"payload") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _fixture_repo(root: Path) -> Path:
    repo = root / "repo"
    _write(repo / "AGENTS.md", b"instructions")
    _write(repo / ".gitignore", b"runs/\n")
    _write(repo / "mnist/model.py", b"model = 1\n")
    _write(repo / "docs/runs/note.md", b"nested source stays")
    _write(repo / "untracked_source.py", b"included")
    for relative in (
        ".git/config",
        ".venv/pyvenv.cfg",
        ".venv-runpod/pyvenv.cfg",
        "mnist_data/train.bin",
        "runs/experiment/model.pt",
        "artifacts/result.bin",
        "outputs/image.png",
        "logs/train.log",
        "test-output/result.bin",
        "handoff/old.zip",
        "mnist/__pycache__/model.pyc",
        "tests/.pytest_cache/state",
        "docs/.ipynb_checkpoints/note.md",
        "docs/.pytest-working/state",
        "docs/.tmp-work/state",
        "scratch.tmp",
        "docs.zip",
        "mnist_cp_samples.png",
    ):
        _write(repo / relative)
    return repo


def test_selected_files_include_working_tree_and_exclude_generated_data(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)

    selected = [relative for relative, _ in handoff.selected_files(repo)]

    assert selected == [
        ".gitignore",
        "AGENTS.md",
        "docs/runs/note.md",
        "mnist/model.py",
        "untracked_source.py",
    ]


def test_build_and_verify_self_describing_archive(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    output = repo / "transfer/repository.zip"

    result = handoff.build_handoff(repo, output)
    verified = handoff.verify_handoff(output)

    assert result["file_count"] == 5
    assert verified["verified"] == 1
    assert verified["archive_sha256"] == result["archive_sha256"]
    assert output.with_suffix(".zip.sha256").is_file()
    with ZipFile(output, "r") as archive:
        names = archive.namelist()
        assert f"{handoff.ARCHIVE_ROOT}/transfer/repository.zip" not in names
        assert f"{handoff.ARCHIVE_ROOT}/transfer/repository.zip.sha256" not in names
        manifest = json.loads(
            archive.read(f"{handoff.ARCHIVE_ROOT}/{handoff.MANIFEST_NAME}")
        )
    assert manifest["schema"] == handoff.SCHEMA
    assert manifest["file_count"] == 5
    assert [row["path"] for row in manifest["files"]] == [
        ".gitignore",
        "AGENTS.md",
        "docs/runs/note.md",
        "mnist/model.py",
        "untracked_source.py",
    ]


def test_list_is_read_only_and_existing_output_requires_force(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    output = tmp_path / "repository.zip"

    listed = handoff.list_handoff(repo, output)

    assert listed["file_count"] == 5
    assert not output.exists()
    handoff.build_handoff(repo, output)
    original = output.read_bytes()
    with pytest.raises(handoff.HandoffError, match="output already exists"):
        handoff.build_handoff(repo, output)
    assert output.read_bytes() == original
    handoff.build_handoff(repo, output, force=True)
    handoff.verify_handoff(output)


def test_verify_rejects_changed_payload(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    output = tmp_path / "repository.zip"
    handoff.build_handoff(repo, output)
    changed = tmp_path / "changed.zip"

    with ZipFile(output, "r") as source, ZipFile(
        changed, "w", compression=ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            payload = source.read(info)
            if info.filename.endswith("/AGENTS.md"):
                payload = b"changed"
            replacement = ZipInfo(info.filename, date_time=info.date_time)
            replacement.compress_type = ZIP_DEFLATED
            replacement.external_attr = info.external_attr
            replacement.create_system = info.create_system
            target.writestr(replacement, payload)

    with pytest.raises(handoff.HandoffError, match="payload size changed|payload checksum changed"):
        handoff.verify_handoff(changed)


def test_sidecar_commits_to_whole_archive(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    output = tmp_path / "repository.zip"
    result = handoff.build_handoff(repo, output)
    sidecar = output.with_suffix(".zip.sha256")

    digest, filename = sidecar.read_text(encoding="utf-8").strip().split("  ")

    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert digest == result["archive_sha256"]
    assert filename == output.name
    sidecar.write_text(f"{'0' * 64}  {output.name}\n", encoding="utf-8")
    with pytest.raises(handoff.HandoffError, match="sidecar"):
        handoff.verify_handoff(output)
