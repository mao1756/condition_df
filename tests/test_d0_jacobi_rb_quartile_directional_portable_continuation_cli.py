from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mnist import (
    diag_d0_jacobi_rb_quartile_directional_portable_continuation as cli,
)
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError, config_fingerprint
from mnist.d0_jacobi_rb_quartile_directional_adjudication_gate import (
    ZERO_AUTHORIZATION_FIELDS,
    ZERO_WORK_FIELDS,
)


def _args(tmp_path: Path, **changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "stage": "relocate",
        "require_gate": "relocate",
        "source_adjudication_run_dir": (tmp_path / "source").resolve(),
        "parent_quartile_specialist_run_dir": (tmp_path / "specialist").resolve(),
        "parent_time_local_run_dir": (tmp_path / "time-local").resolve(),
        "resume_run_dir": None,
        "runs_root": (tmp_path / "runs").resolve(),
        "run_name": "test-portable",
        "device": "cpu",
        "test_only": True,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _semantic(body: dict[str, object]) -> dict[str, object]:
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _safe_record(schema: str, **extra: object) -> dict[str, object]:
    return _semantic(
        {
            "schema": schema,
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": 1,
            **extra,
            **{field: 0 for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS},
        }
    )


def test_parse_requires_resume_after_relocation_and_locks_required_gate(
    tmp_path: Path,
) -> None:
    base = [
        "--source-adjudication-run-dir",
        str(tmp_path / "source"),
        "--parent-quartile-specialist-run-dir",
        str(tmp_path / "specialist"),
        "--parent-time-local-run-dir",
        str(tmp_path / "time-local"),
        "--device",
        "cpu",
        "--test-only",
    ]
    args = cli.parse_args(
        ["--stage", "relocate", "--require-gate", "relocate", *base]
    )
    assert args.stage == "relocate"
    assert args.require_gate == "relocate"
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "fittrace", *base])
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--stage",
                "fittrace",
                "--require-gate",
                "nominate",
                "--resume-run-dir",
                str(tmp_path / "child"),
                *base,
            ]
        )
    assert cli._stage_sequence("all") == (
        "relocate",
        "fittrace",
        "nominate",
        "adjudicate",
        "report",
    )


def test_portable_config_changes_only_operational_binding_and_keeps_zero_scope(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    legacy = _semantic(
        {
            "schema": "legacy-scientific-config",
            "schema_version": 1,
            "parent_quartile_specialist_run_dir": "C:/old/specialist",
            "parent_time_local_run_dir": "C:/old/time-local",
            "source_fingerprint": "a" * 64,
            "grid_size": 28,
            "candidate_count": 480,
            "nominee_stream_count": 36,
            "max_t_family_size": 72,
            **{field: 0 for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS},
        }
    )
    closure = {"portable_content_sha256": "b" * 64}
    result = cli._scientific_config(
        args,
        legacy=legacy,
        identity_sha256="c" * 64,
        source_closure=closure,
    )
    assert result["legacy_scientific_config_sha256"] == legacy["semantic_sha256"]
    assert result["scientific_contract_sha256"] == config_fingerprint(
        cli._science_core(legacy)
    )
    assert result["relocation_changes_scientific_contract"] == 0
    assert result["candidate_count"] == 480
    assert result["nominee_stream_count"] == 36
    assert result["max_t_family_size"] == 72
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert result[field] == 0


def test_runtime_contract_fails_closed_on_production_cpu_but_allows_test_fixture() -> None:
    production = cli._runtime_contract(torch.device("cpu"), test_only=False)
    fixture = cli._runtime_contract(torch.device("cpu"), test_only=True)
    assert production["passed"] == 0
    assert production["test_only"] == 0
    assert fixture["passed"] == 1
    assert fixture["test_only"] == 1
    for record in (production, fixture):
        for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
            assert record[field] == 0


def test_only_uncommitted_relocation_may_restart_in_place(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    assert cli._relocation_resume_is_uncommitted(run_dir)

    cli.atomic_write_json(
        run_dir / "run_status.json",
        {
            "state": "interrupted",
            "decision": "interrupted_relocate",
        },
    )
    assert cli._relocation_resume_is_uncommitted(run_dir)

    cli.atomic_write_json(
        run_dir / "run_status.json",
        {
            "state": "execution_failed",
            "decision": "portable_continuation_invalid",
        },
    )
    assert not cli._relocation_resume_is_uncommitted(run_dir)

    (run_dir / "run_status.json").unlink()
    (run_dir / "relocation_artifact_seal.json").write_text("{}", encoding="utf-8")
    assert not cli._relocation_resume_is_uncommitted(run_dir)


def _write_import_source(source: Path) -> dict[str, object]:
    source.mkdir(parents=True)
    for name in cli._IMPORTED_EVIDENCE:  # noqa: SLF001
        (source / name).write_bytes(f"sealed:{name}\n".encode())
    (source / "resource_projection.json").write_text("{}\n", encoding="utf-8")
    legacy = _semantic(
        {
            "schema": "legacy-config",
            "schema_version": 1,
            "source_fingerprint": "d" * 64,
            "grid_size": 28,
            "candidate_count": 480,
            "nominee_stream_count": 36,
            "max_t_family_size": 72,
            **{field: 0 for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS},
        }
    )
    (source / "scientific_config.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    return legacy


def test_relocation_imports_ready_for_fittrace_without_invoking_old_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    legacy = _write_import_source(args.source_adjudication_run_dir)
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    predecessor = _safe_record(
        "fixture-predecessor", registry_file_sha256="e" * 64
    )
    sources = _safe_record(
        "fixture-sources",
        legacy_source_fingerprint=legacy["source_fingerprint"],
        source_fingerprint=legacy["source_fingerprint"],
        sources=[],
    )
    parents = _safe_record("fixture-parents")
    identity = _safe_record(
        "fixture-identity",
        predecessor=predecessor,
        legacy_sources=sources,
        parents=parents,
    )
    monkeypatch.setattr(cli, "verify_ready_predecessor", lambda *_a, **_k: predecessor)
    monkeypatch.setattr(
        cli, "verify_legacy_source_closure", lambda *_a, **_k: sources
    )
    monkeypatch.setattr(
        cli, "verify_relocated_parent_snapshots", lambda *_a, **_k: parents
    )
    monkeypatch.setattr(cli, "verify_portable_continuation", lambda *_a, **_k: identity)
    monkeypatch.setattr(
        cli,
        "_portable_source_closure",
        lambda *_a, **_k: _safe_record(
            "fixture-portable-sources",
            portable_content_sha256="f" * 64,
            source_fingerprint=legacy["source_fingerprint"],
            sources=[],
        ),
    )
    monkeypatch.setattr(
        cli,
        "_runtime_contract",
        lambda *_a, **_k: _safe_record("fixture-runtime"),
    )

    @contextmanager
    def loading(_root: Path):
        yield

    monkeypatch.setattr(cli, "portable_role_loading", loading)
    monkeypatch.setattr(
        cli._base,
        "_resource_pilot",
        lambda _args: {"within_limits": 1, **cli.safety_record()},
    )
    monkeypatch.setattr(
        cli,
        "_write_native_parent_snapshot",
        lambda target, _args: cli.atomic_write_json(
            target / "parent_immutability_before.json",
            _safe_record("fixture-native-parents"),
        ),
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a completed predecessor stage was rerun")

    monkeypatch.setattr(cli._base, "_preflight_stage", forbidden)
    monkeypatch.setattr(cli._base, "_replay_stage", forbidden)
    monkeypatch.setattr(cli._base, "_controls_stage", forbidden)
    cli._relocate_stage(run_dir, args)

    gate = cli._load_json(run_dir / "relocation_gate.json")
    status = cli._load_json(run_dir / "run_status.json")
    assert gate["passed"] == 1
    assert gate["scientific_contract_unchanged"] == 1
    assert status["decision"] == "ready_for_fittrace"
    assert not (run_dir / "fit_label_open.json").exists()
    for name in cli._IMPORTED_EVIDENCE:  # noqa: SLF001
        assert (run_dir / name).read_bytes() == (
            args.source_adjudication_run_dir / name
        ).read_bytes()
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert gate[field] == 0


def test_resume_rejects_mount_map_change_before_mutating_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    original = _args(tmp_path)
    changed = _args(
        tmp_path,
        parent_time_local_run_dir=(tmp_path / "different-time-local").resolve(),
    )
    cli.atomic_write_json(
        run_dir / "relocation_gate.json",
        _safe_record("fixture-relocation-gate"),
    )
    cli.atomic_write_json(run_dir / "relocation_mount_map.json", cli._mount_map(original))
    sentinel = run_dir / "sentinel.bin"
    sentinel.write_bytes(b"unchanged")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    monkeypatch.setattr(cli, "_verify_seal", lambda *_a, **_k: {})
    monkeypatch.setattr(
        cli,
        "verify_portable_continuation",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("evidence must not open after mount-map failure")
        ),
    )
    with pytest.raises(ArtifactCompatibilityError, match="mount map"):
        cli._verify_resume(run_dir, changed)
    after = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    assert after == before


def test_required_relocation_failure_commits_readable_evidence_before_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    args = _args(tmp_path)
    monkeypatch.setattr(cli, "_make_run_dir", lambda _args: (run_dir, False))

    def fail_runtime(target: Path, _args: SimpleNamespace) -> None:
        cli.atomic_write_json(
            target / "portable_runtime_contract.json",
            _safe_record("fixture-runtime", passed=0),
        )
        raise cli.PortableWorkflowError(
            "runtime mismatch",
            failure_domain="portable_runtime",
            failure_code="portable_runtime_invalid",
        )

    monkeypatch.setattr(cli, "_relocate_stage", fail_runtime)
    assert cli._run(args) == 1
    failure = cli._load_json(run_dir / "relocate_execution_failure.json")
    gate = cli._load_json(run_dir / "relocation_gate.json")
    status = cli._load_json(run_dir / "run_status.json")
    registry = cli._load_json(run_dir / "artifact_registry.json")
    assert failure["failure_code"] == "portable_runtime_invalid"
    assert gate["evaluation_status"] == "execution_failed"
    assert gate["passed"] == 0
    assert status["state"] == "execution_failed"
    assert registry["artifact_count"] >= 4
    for record in (failure, gate, status, registry):
        for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
            assert record[field] == 0


@pytest.mark.parametrize(
    ("stage", "required_gate"),
    [
        ("fittrace", "controls_gate.json"),
        ("nominate", "fittrace_gate.json"),
        ("adjudicate", "nominate_gate.json"),
        ("report", "adjudicate_gate.json"),
    ],
)
def test_stage_delegation_is_exact_and_uses_portable_role_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    required_gate: str,
) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    cli.atomic_write_json(
        run_dir / required_gate,
        {"evaluation_status": "evaluated", "passed": 1},
    )
    args = _args(tmp_path)
    events: list[str] = []

    @contextmanager
    def loading(root: Path):
        events.append(f"enter:{Path(root).name}")
        yield
        events.append("exit")

    monkeypatch.setattr(cli, "portable_role_loading", loading)
    delegated = {
        "fittrace": "_fittrace_stage",
        "nominate": "_nominate_stage",
        "adjudicate": "_adjudicate_stage",
        "report": "_report_stage",
    }[stage]
    monkeypatch.setattr(
        cli._base,
        delegated,
        lambda actual_run, actual_args: events.append(
            f"call:{stage}:{actual_run == run_dir}:{actual_args is args}"
        ),
    )
    cli._delegate_stage(run_dir, args, stage)
    assert f"call:{stage}:True:True" in events
    if stage == "report":
        assert events == ["call:report:True:True"]
    else:
        assert events[0].startswith("enter:")
        assert events[-1] == "exit"
