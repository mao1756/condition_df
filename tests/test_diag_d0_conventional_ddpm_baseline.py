from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from mnist import diag_d0_conventional_ddpm_baseline as runner
from mnist import mnist_generation_benchmark as benchmark
from mnist.conditioned_diffusion import SmallMnistCNN


@pytest.fixture(autouse=True)
def _bind_synthetic_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_git_revision", lambda _: "a" * 40)


def _arff(path: Path, rows: list[str]) -> Path:
    path.write_text("@RELATION tiny\n@DATA\n% comment\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _row(pixel: str = "0", label: str = "0", count: int = 784) -> str:
    return ",".join([pixel] * count + [label])


def _dataset(per_class: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(10, dtype=np.int64), per_class)
    images = np.zeros((len(labels), 28, 28), dtype=np.uint8)
    for index, label in enumerate(labels):
        within = index % per_class
        images[index, 2 + label:4 + label, 3 + within:8 + within] = 80 + 10 * label
        images[index, 20 - within, 5 + label:8 + label] = 180 + within
    return images, labels


def test_atomic_replace_retries_transient_permission_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "resource_ledger.json"
    runner._write_json(path, {"state": "old"})
    real_replace, attempts, sleeps = runner.os.replace, [], []

    def flaky_replace(source: Path, destination: Path) -> None:
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError(5, "simulated sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", flaky_replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    runner._write_json(path, {"state": "new"})
    assert runner._read_json(path) == {"state": "new"}
    assert len(attempts) == 3 and sleeps == [0.05, 0.1]
    assert not list(tmp_path.glob(".resource_ledger.json.*.tmp"))


def test_atomic_replace_exhaustion_preserves_old_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "resource_ledger.json"
    runner._write_json(path, {"state": "old"})
    old_bytes, attempts, sleeps = path.read_bytes(), [], []

    def blocked_replace(source: Path, destination: Path) -> None:
        attempts.append((source, destination))
        raise PermissionError(5, "simulated sharing violation")

    monkeypatch.setattr(runner.os, "replace", blocked_replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        runner._write_json(path, {"state": "new"})
    assert path.read_bytes() == old_bytes and runner._read_json(path) == {"state": "old"}
    assert len(attempts) == 6 and len(sleeps) == 5 and sum(sleeps) == pytest.approx(0.75)
    assert not list(tmp_path.glob(".resource_ledger.json.*.tmp"))


class _TinyEpsilon(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Conv2d(1, 1, 1)
        self.time = nn.Embedding(4, 1)
        self.label = nn.Embedding(10, 1)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        condition = self.time(timesteps) + self.label(labels)
        return self.image(images) + condition[:, :, None, None]


def _tiny_config() -> dict:
    config = copy.deepcopy(runner.FROZEN_CONFIG)
    config["model"]["parameter_count"] = sum(p.numel() for p in _TinyEpsilon().parameters())
    config["schedule"] = {"steps": 4, "beta_start": 1e-4, "beta_end": 2e-2}
    config["training"].update({"epochs": 2, "batch_size": 20, "validation_per_class": 1})
    config["evaluator"].update({"epochs": 1, "batch_size": 20})
    config["reconstruction"].update({"per_class": 1, "start_timesteps": [0, 1, 3]})
    config["sampling"] = {
        "batches": 1,
        "per_class_per_batch": 4,
        "anchors": [0, 1, 2, 3, 4],
        "review_within_class": [0, 1, 2, 3],
    }
    config["resource_defaults"]["terminal_reserve_seconds"] = 0.0
    return config


def _write_answers(path: Path, entries: list[dict], assignment: str = "noise") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_order", "sample_id", "assigned_label", "notes"])
        writer.writeheader()
        for entry in entries:
            writer.writerow({"review_order": entry["review_order"], "sample_id": entry["sample_id"], "assigned_label": assignment, "notes": "manual"})


def test_arff_slice_is_exact_and_fixed_splits_are_disjoint(tmp_path: Path) -> None:
    path = _arff(tmp_path / "tiny.arff", [_row("0", "1"), _row("7", "2"), _row("255", "3")])
    images, labels = benchmark.read_mnist_arff_slice(path, 1, 3)
    assert images.shape == (2, 28, 28) and images.dtype == np.uint8
    assert labels.dtype == np.int64 and labels.tolist() == [2, 3]
    assert int(images[0, 0, 0]) == 7 and int(images[1, -1, -1]) == 255
    split_sets = {name: set(range(*bounds)) for name, bounds in benchmark.FIXED_SPLITS.items()}
    assert {name: len(values) for name, values in split_sets.items()} == {"train": 55_000, "validation": 5_000, "test": 10_000}
    assert not (split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"])


@pytest.mark.parametrize("bad_row", [_row(count=783), _row("word"), _row("0.5"), _row("256"), _row("0", "10")])
def test_arff_slice_rejects_malformed_selected_rows(tmp_path: Path, bad_row: str) -> None:
    path = _arff(tmp_path / "bad.arff", [bad_row])
    with pytest.raises(ValueError):
        benchmark.read_mnist_arff_slice(path, 0, 1)


def test_hash_gate_fails_before_data_or_model_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    arff = _arff(tmp_path / "wrong.arff", [_row()])
    called = False

    def forbidden(_: Path) -> tuple:
        nonlocal called
        called = True
        raise AssertionError("data construction must remain unreachable")

    monkeypatch.setattr(runner, "_source_hashes", lambda _: {"source.py": "bound"})
    monkeypatch.setattr(runner, "load_train_validation_mnist", forbidden)
    monkeypatch.setattr(runner, "EXPECTED_ARFF_SHA256", "0" * 64)
    with pytest.raises(runner.DDPMRunError, match="hash mismatch"):
        runner.initialize_run(repository, arff, tmp_path / "run", device="cpu", maximum_active_seconds=1000, maximum_cuda_fraction=.5, maximum_storage_mib=1, approval_reference="approval-123")
    assert not called and not (tmp_path / "run").exists()


def test_numeric_transforms_duplicate_and_diversity_formulas() -> None:
    image = np.zeros((1, 28, 28), np.uint8)
    image[0, 0, :6] = [0, 1, 127, 128, 254, 255]
    model = benchmark.uint8_to_model_space(image)
    assert model.shape == (1, 1, 28, 28) and model.dtype == np.float32
    assert np.array_equal(benchmark.model_space_to_uint8(model), image)
    clipped = np.zeros((3, 1, 28, 28), np.float32)
    clipped[:, 0, 0, 0] = [-2, 0, 2]
    assert benchmark.model_space_to_uint8(clipped)[:, 0, 0].tolist() == [0, 128, 255]

    images = np.zeros((4, 28, 28), np.uint8)
    images[3] = 1
    duplicate = benchmark.exact_duplicate_metrics(images, np.zeros(4, np.int64))
    assert (duplicate["unique_count"], duplicate["duplicate_count"], duplicate["duplicate_group_count"], duplicate["duplicate_pair_count"]) == (2, 2, 1, 3)

    generated = np.stack([np.zeros((28, 28), np.uint8), np.full((28, 28), 10, np.uint8)])
    real = np.stack([np.zeros((28, 28), np.uint8), np.full((28, 28), 20, np.uint8)])
    diversity = benchmark.within_class_nn_diversity(generated, np.zeros(2, np.int64), real, np.zeros(2, np.int64))
    assert diversity["aggregate_median_ratio"] == pytest.approx(.25)
    assert diversity["by_class"]["0"]["generated_median_nn_mse"] == pytest.approx((10 / 255) ** 2)


def test_contact_sheet_and_blinded_review_are_ordered_and_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = np.stack([np.full((28, 28), value, np.uint8) for value in (10, 20, 30, 40)])
    labels = np.asarray([1, 2, 3, 4], np.int64)
    ids = np.asarray([f"opaque-{index}" for index in range(4)])
    sheet_path = benchmark.write_contact_sheet(tmp_path / "sheet.png", images, columns=2, scale=2)
    with Image.open(sheet_path) as sheet:
        assert sheet.size == (112, 112)
        assert [sheet.getpixel(point) for point in ((0, 0), (56, 0), (0, 56), (56, 56))] == [10, 20, 30, 40]

    captions: list[str] = []
    original_write_sheet = benchmark.write_contact_sheet

    def capture_sheet(*args, **kwargs):
        captions.extend(kwargs.get("captions") or [])
        return original_write_sheet(*args, **kwargs)

    monkeypatch.setattr(benchmark, "write_contact_sheet", capture_sheet)
    first = benchmark.write_blinded_review_bundle(tmp_path / "first", images, labels, ids, seed=99, columns=2, scale=1)
    second = benchmark.write_blinded_review_bundle(tmp_path / "second", images, labels, ids, seed=99, columns=2, scale=1)
    assert first["ordered_sample_ids"].tolist() == second["ordered_sample_ids"].tolist()
    public = (tmp_path / "first/human_review_template.csv").read_text(encoding="utf-8")
    header = public.splitlines()[0]
    assert header == "review_order,sample_id,assigned_label,notes" and "requested" not in header
    assert captions[:4] == [f"blind-{index:03d}" for index in range(4)]
    assert all(source not in public and source not in captions[:4] for source in ids)
    key_entries = json.loads(Path(first["key"]).read_text(encoding="utf-8"))["entries"]
    assert {entry["source_sample_id"] for entry in key_entries} == set(ids.tolist())
    answers = tmp_path / "answers.csv"
    _write_answers(answers, key_entries)
    assert benchmark.score_human_review(answers, first["key"], reviewer="human", confirm_manual_review=True)["sample_count"] == 4


@pytest.mark.parametrize("case", ["missing", "duplicate", "invalid", "unconfirmed"])
def test_review_scoring_rejects_incomplete_or_nonmanual_answers(tmp_path: Path, case: str) -> None:
    images = np.zeros((3, 28, 28), np.uint8)
    bundle = benchmark.write_blinded_review_bundle(tmp_path / "review", images, np.asarray([1, 2, 3]), np.asarray(["a", "b", "c"]), seed=7)
    entries = json.loads(Path(bundle["key"]).read_text(encoding="utf-8"))["entries"]
    answers = tmp_path / "answers.csv"
    _write_answers(answers, entries)
    if case == "missing":
        _write_answers(answers, entries[:-1])
    elif case == "duplicate":
        _write_answers(answers, [entries[0], entries[0], entries[2]])
    elif case == "invalid":
        _write_answers(answers, entries, "machine-label")
    with pytest.raises(ValueError):
        benchmark.score_human_review(answers, bundle["key"], reviewer="human", confirm_manual_review=case != "unconfirmed")


@pytest.mark.parametrize("approval", ["", " ", "<fresh-approval-reference>", "fresh approval", "TODO", "placeholder"])
def test_placeholder_approval_references_are_rejected(approval: str) -> None:
    with pytest.raises(runner.DDPMRunError, match="approval"):
        runner._approval(approval)


@pytest.mark.parametrize("mismatch", ["source", "config", "data", "environment"])
def test_resume_rejects_source_config_or_data_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    arff = _arff(tmp_path / "data.arff", [_row()])
    expected = hashlib.sha256(arff.read_bytes()).hexdigest()
    source = {"source.py": "first"}
    environment = {"device": "cpu", "runtime": "first"}
    monkeypatch.setattr(runner, "EXPECTED_ARFF_SHA256", expected)
    monkeypatch.setattr(runner, "_source_hashes", lambda _: dict(source))
    monkeypatch.setattr(runner, "_environment", lambda _: dict(environment))
    empty = (np.empty((55_000, 0, 0), np.uint8), np.zeros(55_000, np.int64), np.empty((5_000, 0, 0), np.uint8), np.zeros(5_000, np.int64))
    monkeypatch.setattr(runner, "load_train_validation_mnist", lambda _: empty)
    monkeypatch.setattr(runner, "__name__", "__main__")
    config = _tiny_config()
    run_dir = runner.initialize_run(repository, arff, tmp_path / "run", device="cpu", maximum_active_seconds=2000, maximum_cuda_fraction=.5, maximum_storage_mib=5, approval_reference="approval-123", config=config)
    assert "-m mnist.diag_d0_conventional_ddpm_baseline" in (run_dir / "command.txt").read_text(encoding="utf-8")
    if mismatch == "source":
        runner.initialize_run(repository, arff, run_dir, device="cpu", maximum_active_seconds=2000, maximum_cuda_fraction=.5, maximum_storage_mib=5, approval_reference="approval-123", resume=True, config=config)
        assert runner._ledger(run_dir)["restart_count"] == 1
    resume_config = copy.deepcopy(config)
    if mismatch == "source":
        source["source.py"] = "second"
    elif mismatch == "config":
        resume_config["diagnostic"]["classifier_accuracy"] = .1
    else:
        if mismatch == "data":
            arff.write_text(arff.read_text(encoding="utf-8") + "% changed\n", encoding="utf-8")
            monkeypatch.setattr(runner, "EXPECTED_ARFF_SHA256", hashlib.sha256(arff.read_bytes()).hexdigest())
        else:
            environment["runtime"] = "second"
    with pytest.raises(runner.DDPMRunError, match="mismatch|environment changed"):
        runner.initialize_run(repository, arff, run_dir, device="cpu", maximum_active_seconds=2000, maximum_cuda_fraction=.5, maximum_storage_mib=5, approval_reference="approval-123", resume=True, config=resume_config)


def _generator_run_dir(path: Path) -> Path:
    (path / "training").mkdir(parents=True)
    (path / "controls").mkdir()
    runner._write_json(path / "source_bindings.json", {"data_sha256": "synthetic"})
    runner._write_json(path / "controls/oracle_preflight.json", {"passed": 1})
    runner._write_json(path / "resource_ledger.json", {"active_seconds": 0.0, "events": [], "maximum_active_seconds": 1000.0, "maximum_storage_bytes": 10_000_000, "maximum_cuda_fraction": .5, "peak_cuda_fraction": 0.0, "peak_storage_bytes": 0})
    return path


def test_epoch_resume_is_exact_tie_selects_earliest_and_pause_keeps_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_config()
    train_x, train_y = _dataset(2)
    validation_x, validation_y = _dataset(1)
    monkeypatch.setattr(runner, "ClassConditionalUNet28", _TinyEpsilon)
    monkeypatch.setattr(runner, "_validation_mse", lambda *args, **kwargs: .5)
    calls = 0

    def pause_second(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise runner.ResourcePause("synthetic cap")

    resumed_dir = _generator_run_dir(tmp_path / "resumed")
    monkeypatch.setattr(runner, "_resource_check", pause_second)
    with pytest.raises(runner.ResourcePause, match="synthetic cap"):
        runner.train_or_resume_generator(resumed_dir, train_x, train_y, validation_x, validation_y, torch.device("cpu"), config)
    paused = torch.load(resumed_dir / "training/latest.pt", map_location="cpu", weights_only=True)
    assert paused["completed_epoch"] == 1 and config["training"]["epochs"] == 2
    monkeypatch.setattr(runner, "_resource_check", lambda *args, **kwargs: None)
    selection = runner.train_or_resume_generator(resumed_dir, train_x, train_y, validation_x, validation_y, torch.device("cpu"), config)
    assert selection["selected_epoch"] == 1 and selection["completed_epochs"] == 2

    clean_dir = _generator_run_dir(tmp_path / "clean")
    runner.train_or_resume_generator(clean_dir, train_x, train_y, validation_x, validation_y, torch.device("cpu"), config)
    resumed = torch.load(resumed_dir / "training/latest.pt", map_location="cpu", weights_only=True)
    clean = torch.load(clean_dir / "training/latest.pt", map_location="cpu", weights_only=True)
    assert resumed["history"] == clean["history"]
    assert all(torch.equal(resumed["model_state"][key], clean["model_state"][key]) for key in clean["model_state"])


def _fake_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(device="cuda:0", resume=False, repository_root=str(tmp_path), arff=str(tmp_path / "data.arff"), runs_root=str(tmp_path), run_name="run", run_dir=None, maximum_active_seconds=2000, maximum_cuda_fraction=.5, maximum_storage_mib=5, approval_reference="approval-123")


def test_oracle_failure_stops_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "initialize_run", lambda *args, **kwargs: run_dir)
    runner._status(run_dir, "initialized", resumable=True)
    monkeypatch.setattr(runner, "load_train_validation_mnist", lambda _: _dataset(1) * 2)
    def failed_oracle(*args, **kwargs):
        runner._write_json(run_dir / "controls/oracle_preflight.json", {"passed": 0})
        raise runner.DDPMRunError("Gate B failed")

    monkeypatch.setattr(runner, "run_oracle_preflight", failed_oracle)
    monkeypatch.setattr(runner, "train_or_load_evaluator", lambda *args, **kwargs: pytest.fail("training was reached"))
    with pytest.raises(runner.DDPMRunError, match="Gate B"):
        runner._run(_fake_args(tmp_path))
    status = runner._read_json(run_dir / "status.json")
    assert status["state"] == "failed" and status["resumable"] == 0
    arff = tmp_path / "data.arff"
    arff.write_text("synthetic", encoding="utf-8")
    runner._write_json(run_dir / "source_bindings.json", {"repository_root": str(tmp_path), "arff": str(arff)})
    resume_args = _fake_args(tmp_path)
    resume_args.resume, resume_args.run_dir = True, str(run_dir)
    with pytest.raises(runner.DDPMRunError, match="Gate B"):
        runner._run(resume_args)


@pytest.mark.parametrize(
    ("classifier", "recognizable", "agreement", "route"),
    [(.2, .9, .9, "audit_evaluator_numeric_transform"), (.9, .2, .0, "metric_misalignment")],
)
def test_human_classifier_disagreement_has_an_explicit_route(tmp_path: Path, classifier: float, recognizable: float, agreement: float, route: str) -> None:
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluator").mkdir()
    runner._write_json(tmp_path / "evaluation/metrics.json", {"classifier": {"requested_label_accuracy": classifier}, "duplicates": {"duplicate_pair_count": 0}, "diversity": {"aggregate_median_ratio": 1.0}})
    runner._write_json(tmp_path / "evaluator/real_validation_metrics.json", {"gate_c_validation_passed": 1})
    runner._write_json(tmp_path / "evaluator/real_test_metrics.json", {"gate_c_test_passed": 1})
    runner._write_json(tmp_path / "resource_ledger.json", {"active_seconds": 1, "maximum_active_seconds": 10, "peak_cuda_fraction": 0, "maximum_cuda_fraction": .5, "peak_storage_bytes": 1, "maximum_storage_bytes": 10})
    human = {"human_requested_label_agreement": agreement, "human_recognizability": recognizable}
    assert runner._outcome(tmp_path, human, _tiny_config())["route"] == route


def test_machine_lifecycle_preserves_adverse_artifacts_and_opens_test_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_config()
    repository = tmp_path / "repository"
    repository.mkdir()
    arff = _arff(tmp_path / "data.arff", [_row()])
    digest = hashlib.sha256(arff.read_bytes()).hexdigest()
    train_x, train_y = _dataset(2)
    validation_x, validation_y = _dataset(2)
    test_x, test_y = _dataset(4)
    source = {"synthetic-source.py": "fixed"}
    monkeypatch.setattr(runner, "EXPECTED_ARFF_SHA256", digest)
    monkeypatch.setattr(runner, "_source_hashes", lambda _: source)
    empty = (np.empty((55_000, 0, 0), np.uint8), np.zeros(55_000, np.int64), np.empty((5_000, 0, 0), np.uint8), np.zeros(5_000, np.int64))
    monkeypatch.setattr(runner, "load_train_validation_mnist", lambda _: empty)
    monkeypatch.setattr(runner, "ClassConditionalUNet28", _TinyEpsilon)
    monkeypatch.setattr(runner, "_validation_mse", lambda *args, **kwargs: .5)
    monkeypatch.setattr(runner, "_resource_check", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "FROZEN_CONFIG", config)

    def fake_evaluator(*args, **kwargs):
        history = {"train_loss": [1.0], "train_accuracy": [.1], "val_loss": [1.0], "val_accuracy": [.1]}
        return SmallMnistCNN(), {"history": history, "selected_epoch": 1, "selected_validation_accuracy": .1, "selected_validation_loss": 1.0}

    monkeypatch.setattr(runner, "train_frozen_image_evaluator", fake_evaluator)
    test_calls = 0

    def terminal_loader(_: Path):
        nonlocal test_calls
        test_calls += 1
        required = ["training/selected_checkpoint.pt", "evaluator/selected_checkpoint.pt", "evaluation/prior_starts.npz", "evaluation/samples_uint8.npz", "evaluation/sampling_manifest.csv"]
        assert all((run_dir / relative).is_file() for relative in required)
        event = runner._read_json(run_dir / "data/test_open_event.json")
        assert all(runner._file_sha256(run_dir / relative) == event["frozen_hashes"][relative] for relative in required)
        return test_x, test_y

    monkeypatch.setattr(runner, "load_test_mnist_terminal", terminal_loader)
    run_dir = runner.initialize_run(repository, arff, tmp_path / "run", device="cpu", maximum_active_seconds=2000, maximum_cuda_fraction=.5, maximum_storage_mib=20, approval_reference="approval-123", config=config)
    monkeypatch.setattr(runner, "load_train_validation_mnist", lambda _: (train_x, train_y, validation_x, validation_y))
    assert test_calls == 0
    oracle = runner.run_oracle_preflight(run_dir, validation_x, validation_y, torch.device("cpu"), config)
    assert oracle["passed"] == 1 and test_calls == 0
    evaluator, evaluator_selection = runner.train_or_load_evaluator(run_dir, train_x, train_y, validation_x, validation_y, torch.device("cpu"), config)
    assert evaluator_selection["gate_c_validation_passed"] == 0 and test_calls == 0
    runner.train_or_resume_generator(run_dir, train_x, train_y, validation_x, validation_y, torch.device("cpu"), config)
    runner.run_reconstruction_panel(run_dir, evaluator, torch.device("cpu"), config)
    original_sample_reverse = runner.sample_reverse
    sample_calls = 0

    def committed_sample(*args, **kwargs):
        nonlocal sample_calls
        sample_calls += 1
        assert (run_dir / "evaluation/prior_starts.npz").is_file()
        assert (run_dir / "evaluation/sampling_manifest.csv").is_file()
        return original_sample_reverse(*args, **kwargs)

    monkeypatch.setattr(runner, "sample_reverse", committed_sample)
    gallery = runner.generate_prior_gallery(run_dir, torch.device("cpu"), config)
    assert gallery["images"].shape == (40, 28, 28) and sample_calls == 1 and test_calls == 0
    runner.open_test_and_score(run_dir, arff, train_x, evaluator, torch.device("cpu"), config)
    assert test_calls == 1
    review = runner.prepare_human_review(run_dir, config)
    assert review["review_count"] == 40
    public_review = (run_dir / "review/human_review_template.csv").read_text(encoding="utf-8")
    assert all(source_id not in public_review for source_id in gallery["sample_ids"])
    assert all(row["sample_id"] == f"blind-{index:03d}" for index, row in enumerate(csv.DictReader(public_review.splitlines())))
    assert runner.finalize_run(run_dir, config) is None
    assert runner._read_json(run_dir / "status.json")["state"] == "awaiting_human_review"
    assert not (run_dir / "outcome.json").exists()
    for relative in ("evaluation/samples_uint8.npz", "evaluation/images/balanced-final.png", "evaluation/metrics.json", "review/human_review_template.csv", "REPORT.md"):
        assert (run_dir / relative).is_file()
    assert len(set(gallery["sample_ids"].tolist())) == 40
    assert all(str(value).startswith("id-") and len(str(value)) == 23 and str(value)[3:].isalnum() and str(value)[3:].islower() for value in gallery["sample_ids"])

    before = runner._tree_digest(run_dir)
    verified = runner.verify_run(run_dir)
    assert verified["tree_digest"] == before == runner._tree_digest(run_dir)
    metric_paths = tuple(run_dir / path for path in ("evaluator/real_validation_metrics.json", "evaluator/real_test_metrics.json", "evaluator/selection.json"))
    original_metric_rows = [runner._read_json(path) for path in metric_paths]

    def reseal_pending() -> None:
        runner._replace(run_dir / "REPORT.md", lambda path: path.write_text(runner._report(run_dir, None), encoding="utf-8")); runner._refresh_manifest(run_dir)

    drifted = copy.deepcopy(original_metric_rows)
    drifted[0]["loss"] *= 1.00015; drifted[1]["loss"] *= 1.00015; drifted[2]["restored_validation_loss"] *= 1.00015
    for path, row in zip(metric_paths, drifted): runner._write_json(path, row)
    reseal_pending()
    assert runner.verify_run(run_dir)["passed"] == 1
    excessive = copy.deepcopy(drifted[0])
    excessive["loss"] = original_metric_rows[0]["loss"] * 1.0005
    runner._write_json(metric_paths[0], excessive); reseal_pending()
    with pytest.raises(runner.DDPMRunError, match="saved validation metrics changed"):
        runner.verify_run(run_dir)
    exact_tamper = copy.deepcopy(original_metric_rows[0])
    exact_tamper["accuracy"] += 1e-12; exact_tamper["gate_c_validation_passed"] = 1 - exact_tamper["gate_c_validation_passed"]
    runner._write_json(metric_paths[0], exact_tamper); reseal_pending()
    with pytest.raises(runner.DDPMRunError, match="saved validation metrics changed"):
        runner.verify_run(run_dir)
    for path, row in zip(metric_paths, original_metric_rows): runner._write_json(path, row)
    reseal_pending()
    key = json.loads((run_dir / "review/review_key.json").read_text(encoding="utf-8"))["entries"]
    answers = run_dir / "review/manual-answers.csv"
    _write_answers(answers, key)
    frozen = {relative: runner._file_sha256(run_dir / relative) for relative in ("training/selected_checkpoint.pt", "evaluation/prior_starts.npz", "evaluation/samples_uint8.npz")}
    outcome = runner.record_human_review(run_dir, answers, "Manual Reviewer", True, config)
    assert outcome["route"] == "repair_evaluator" and runner._read_json(run_dir / "status.json")["state"] == "complete"
    assert frozen == {relative: runner._file_sha256(run_dir / relative) for relative in frozen}
    assert runner.verify_run(run_dir)["passed"] == 1
    sealed = runner._tree_digest(run_dir)
    assert runner.record_human_review(run_dir, run_dir / "review/human_review_answers.csv", "ignored", True, config) == outcome
    assert runner._tree_digest(run_dir) == sealed

    for relative in ("training/selected_checkpoint.pt", "evaluation/samples_uint8.npz", "evaluation/images/balanced-final.png", "evaluation/metrics.json", "data/train_indices.npy"):
        path, original = run_dir / relative, (run_dir / relative).read_bytes()
        path.write_bytes(original + b"tamper")
        with pytest.raises(runner.DDPMRunError, match="artifact changed"):
            runner.verify_run(run_dir)
        path.write_bytes(original)
    missing = run_dir / "controls/oracle_preflight.json"
    original = missing.read_bytes()
    missing.unlink()
    with pytest.raises(runner.DDPMRunError, match="inventory changed"):
        runner.verify_run(run_dir)
    missing.write_bytes(original)

    metrics_path = run_dir / "evaluation/metrics.json"
    original_metrics = runner._read_json(metrics_path)
    scoring_path = run_dir / "evaluation/SCORING_READY.json"
    original_scoring = runner._read_json(scoring_path)
    changed_metrics = copy.deepcopy(original_metrics)
    changed_metrics["classifier"]["requested_label_accuracy"] = .987654321
    runner._write_json(metrics_path, changed_metrics)
    changed_scoring = copy.deepcopy(original_scoring)
    changed_scoring["metrics_sha256"] = runner._file_sha256(metrics_path)
    runner._write_json(scoring_path, changed_scoring)
    runner._refresh_manifest(run_dir)
    with pytest.raises(runner.DDPMRunError, match="saved sample metrics changed"):
        runner.verify_run(run_dir)
    runner._write_json(metrics_path, original_metrics)
    runner._write_json(scoring_path, original_scoring)
    runner._refresh_manifest(run_dir)
    assert runner.verify_run(run_dir)["passed"] == 1
