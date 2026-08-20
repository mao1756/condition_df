from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
import torch

import mnist.d0_jacobi_rb_cuda_deferred as rb_cuda
import mnist.eulerian_jacobi_ddpm as core
from mnist import diag_eulerian_jacobi_ddpm_candidate_pilot as runner
from mnist import eulerian_jacobi_ddpm_candidate as candidate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROTECTED_SOURCE_HASHES = {
    "mnist/__init__.py": "1afbf919b879fc8c499db24009ce92e92ee03b198cfb427830a18df37df86ce4",
    "mnist/conditioned_diffusion.py": "96906c6c1cf7fed4de191e56d6861621446e65b6952171cbf6fa556303450892",
    "mnist/d0_jacobi_artifacts.py": "75bac4e947349993f6f7bdc3cf6df31e0861f67d8fbe7688d3761ee7d6325e21",
    "mnist/d0_jacobi_denoising.py": "a434b27daba832ae81de5e677b8c8482d30680e9cbd079f427fb7ceec1a47b39",
    "mnist/d0_jacobi_rb_absolute_coordinate.py": "5fbd05880e584bfe5fcfb5090955b1cf15db8f1dc3eb0395cc2c915d7c5a7183",
    "mnist/d0_jacobi_rb_boundary_tangent.py": "be4ab9ad8007e567bb518b98c04b37e4900669c53607ef6612f951d15d3a17ce",
    "mnist/d0_jacobi_rb_boundary_tangent_frequency1_coordinate.py": "bb67d8f44136e82647b881e8badd4f6b72382432874aff9fb30d65da21edbed4",
    "mnist/d0_jacobi_rb_boundary_tangent_v3_provenance.py": "c591d4047c6b3763247d56e7eedcff97d4c8e7d82fa28d5ec844ae142b59e4f6",
    "mnist/d0_jacobi_rb_boundary_tangent_zero_baseline.py": "5aa6fdfe7f6e23a92ef37fa86deeca317760c1ca90abdffee059bcf06b09235c",
    "mnist/d0_jacobi_rb_coarse_residual.py": "b3157a81cad5dcb257cb5deb09054b515e659023e44a4546edd61d58659826b3",
    "mnist/d0_jacobi_rb_controls.py": "3186c3321a4f48bda6b7a2a28a600812b7686d0b68aa499ca9fe6735bc7a7d17",
    "mnist/d0_jacobi_rb_cuda.py": "94b95db6c93510c97c36b7cd67b2dec3b1f13a62b3077299e6edd6b97f0ba97a",
    "mnist/d0_jacobi_rb_cuda_certificate.py": "f43bd0459a3200bbead706cf7def1cca17e344bbccd7ae4de5cfb26b1eb9aced",
    "mnist/d0_jacobi_rb_cuda_controls.py": "a834445afa5f4003931254a13fbe1e0838904bf9e47726abeaf3faa5955f01ff",
    "mnist/d0_jacobi_rb_cuda_fused.py": "184a3e9e8e476b835e808de4f1b5b7d641d33997448968539ab240a54f91204d",
    "mnist/d0_jacobi_rb_cuda_multipath.py": "5949dc794085cde340b42133a4a2102815ac85dece4e6799b23762de62507f77",
    "mnist/d0_jacobi_rb_global_dilated.py": "2ea368bc0d001803ce8e8c5f9862feefe01aa88ada395f0279636e8ce6e4135a",
    "mnist/d0_jacobi_rb_learnability.py": "081c9dfa7414c3c9fda80b262162eb3ad6c84ddaff905a058896580c1f1d50b2",
    "mnist/d0_jacobi_rb_reverse_controller.py": "adac975b5d64e23f7d0861dce0ac1b497fa054eed6db2a76358e462d94c8ee5f",
    "mnist/d0_jacobi_rb_spectral.py": "f16851db6f9b5f91cec5fc7ab1121461a4b915a63003dba88530b1a8a4f1b635",
    "mnist/d0_jacobi_rb_strang_refinement.py": "9ba9a12032fb9e4babc72568d5494380c5cc06a74c5495c81369b658fd048975",
    "mnist/d0_jacobi_source_compat.py": "f90ac705d105e03ca258f8507fa74e77e9cc2ef3cea2bf8615594cc5dc5c07ed",
    "mnist/d0_jacobi_v3_source_compat.py": "17e9f47c573944a1affcae43ee63bd057fc18fc54ea917fdbdd6fceecc0c6b8c",
    "mnist/diag_eulerian_jacobi_ddpm_mnist.py": "3f5a0f963bc4b2042a10e71f9290478b2ce27c4520913d219c98c03a807a2c9e",
    "mnist/eulerian_jacobi_ddpm.py": "5875373c34fa6fd4620749c5763ce91903728f7f0d2f70c6e7a65a1f8023ab98",
    "mnist/mnist_generation_benchmark.py": "2ebf13e37e03646222b8decde91034f10dff564fdc0d3f7a967af789ef3cbfd6",
    "mnist/weighted_point_cloud.py": "b70db19c8adbaf7cd89818a61a7dc8b167ec83e013911682702161c7e28fca7d",
}
EXPECTED_PATH_RANGES = {
    "numerical_audit": (0xB6000, 0xB6010),
    "resource_smoke": (0xB6100, 0xB6108),
    "training": (0xB6200, 0xB62FA),
    "validation": (0xB6400, 0xB6464),
    "prior": (0xB6500, 0xB6514),
    "forward_terminal": (0xB6520, 0xB6534),
    "oracle": (0xB6540, 0xB654A),
}
_PATH_AUDIT_CACHE: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config() -> dict[str, Any]:
    value = runner.candidate_config()
    assert isinstance(value, dict)
    return value


def _path_audit_fixture() -> dict[str, Any]:
    global _PATH_AUDIT_CACHE
    if _PATH_AUDIT_CACHE is None:
        _PATH_AUDIT_CACHE = runner.path_id_audit(REPOSITORY_ROOT)
    return json.loads(json.dumps(_PATH_AUDIT_CACHE))


def _parser_for(command: str):
    parser = runner.build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    return parser, subparsers.choices[command]


def _simplex_rows(count: int) -> np.ndarray:
    rows = np.arange(1, count * core.STATE_SIZE + 1, dtype=np.float64).reshape(
        count, core.STATE_SIZE
    )
    rows /= rows.sum(axis=1, keepdims=True)
    return rows


def _diagnostics(
    active: torch.Tensor,
    *,
    invalid_output_count: int = 0,
) -> dict[str, torch.Tensor]:
    device = active.device
    zero = lambda: torch.zeros((), dtype=torch.int64, device=device)
    result = {
        "sample_count": torch.tensor(active.numel(), dtype=torch.int64, device=device),
        "active_count": active.sum(dtype=torch.int64),
        "structural_noop_count": (~active).sum(dtype=torch.int64),
        "approximation_count": active.sum(dtype=torch.int64),
        "invalid_input_count": zero(),
        "invalid_output_count": torch.tensor(
            invalid_output_count, dtype=torch.int64, device=device
        ),
        "nonfinite_count": zero(),
        "negative_bracket_width_count": zero(),
        "bracket_order_invalid_count": zero(),
        "maximum_candidate_bracket_width": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "candidate_kernel_launch_count": torch.ones(
            (), dtype=torch.int64, device=device
        ),
    }
    for name in (
        "resource_cap_count",
        "invalid_density_count",
        "correction_count",
        "clipping_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
    ):
        result[name] = zero()
    return result


class FakeCandidateEnqueue:
    """Stateless, ID-bound CPU stand-in for the proposal-only CUDA enqueue."""

    def __init__(self, *, invalid_lane: int | None = None) -> None:
        self.invalid_lane = invalid_lane
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        head_fraction: torch.Tensor,
        exposure: torch.Tensor,
        *,
        rng_key: Any,
        transition_ids: torch.Tensor,
        prepared: object,
        prepared_rng_seed: object,
    ) -> rb_cuda.CandidateRBCudaBatch:
        del prepared
        active = exposure > 0
        id_term = (transition_ids.to(torch.int64) % 17).to(head_fraction.dtype)
        displacement = (id_term - 8.0) * 1e-5 * head_fraction * (1.0 - head_fraction)
        later = torch.where(active, head_fraction + displacement, head_fraction)
        target = torch.where(active, displacement, torch.zeros_like(displacement))
        valid = torch.ones_like(active)
        if self.invalid_lane is not None:
            valid.reshape(-1)[self.invalid_lane] = False
        diagnostics = _diagnostics(
            active,
            invalid_output_count=int(self.invalid_lane is not None),
        )
        self.calls.append(
            {
                "rng_key": rng_key,
                "transition_ids": transition_ids.detach().clone(),
                "head_fraction": head_fraction.detach().clone(),
                "exposure": exposure.detach().clone(),
                "prepared_rng_seed": prepared_rng_seed,
            }
        )
        return rb_cuda.CandidateRBCudaBatch(
            earlier_head_fraction=head_fraction,
            later_head_fraction=later,
            denoising_target=target,
            exposure=exposure,
            transition_ids=transition_ids,
            active_mask=active,
            structural_noop_mask=~active,
            approximation_mask=active,
            valid_mask=valid,
            candidate_lower=later,
            candidate_upper=later,
            device_diagnostics=diagnostics,
        )


def _fake_runtime(*root_seeds: int) -> candidate.CandidateRuntime:
    keys: list[tuple[Any, ...]] = []
    for root_seed in root_seeds:
        keys.append((root_seed, "forward"))
        keys.extend(
            (root_seed, "reverse", micro, side)
            for micro in range(2)
            for side in ("pre", "post")
        )
    profile = rb_cuda.JacobiRBCudaProfile()
    return candidate.CandidateRuntime(
        device=torch.device("cpu"),
        profile=profile,
        prepared=SimpleNamespace(device=torch.device("cpu"), profile=profile),
        prepared_seeds={key: SimpleNamespace(key=key) for key in keys},
        candidate_binary_sha256="a" * 64,
    )


class ZeroRecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[core.ModelInputs] = []
        self.predictor = SimpleNamespace(
            score_prediction_prevalidated=self.score_prediction_prevalidated
        )

    def score_prediction_prevalidated(self, inputs: core.ModelInputs) -> torch.Tensor:
        self.inputs.append(inputs)
        return torch.zeros(
            (inputs.later_full_state.shape[0], core.EDGES_PER_PHASE),
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )

    def score_prediction(self, _inputs: core.ModelInputs) -> torch.Tensor:
        raise AssertionError("candidate adapter must use the prevalidated predictor seam")


class NonfiniteRecordingModel(ZeroRecordingModel):
    def score_prediction_prevalidated(self, inputs: core.ModelInputs) -> torch.Tensor:
        self.inputs.append(inputs)
        return torch.full(
            (inputs.later_full_state.shape[0], core.EDGES_PER_PHASE),
            float("nan"),
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )


def test_protected_source_hashes_are_unchanged() -> None:
    assert runner.PROTECTED_SOURCE_HASHES == EXPECTED_PROTECTED_SOURCE_HASHES
    assert len(EXPECTED_PROTECTED_SOURCE_HASHES) == 27
    for relative, expected in EXPECTED_PROTECTED_SOURCE_HASHES.items():
        path = REPOSITORY_ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected, relative


def test_candidate_config_has_no_science_cli_overrides() -> None:
    parser, run_parser = _parser_for("run")
    commands = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    assert set(commands.choices) == {"run", "record-review", "verify"}
    options = {
        option
        for action in run_parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    assert options == {
        "--run-dir",
        "--arff",
        "--ddpm-run-dir",
        "--device",
        "--approval-id",
        "--max-active-seconds",
        "--max-storage-mib",
        "--max-cuda-fraction",
        "--help",
    }
    assert not any(
        token in option
        for option in options
        for token in ("seed", "threshold", "gain", "paths", "updates", "modes")
    )


def test_fresh_path_ranges_are_exact_disjoint_and_20_bit(tmp_path: Path) -> None:
    configured = _config()["path_ids"]
    all_values: list[int] = []
    for role, (start, stop) in EXPECTED_PATH_RANGES.items():
        row = configured[role]
        assert (row["start"], row["stop_exclusive"]) == (start, stop)
        assert row["count"] == stop - start
        values = list(range(start, stop))
        assert row["sha256"] == hashlib.sha256(
            np.asarray(values, dtype="<i8").tobytes()
        ).hexdigest()
        assert all(0 <= value < 1 << 20 for value in values)
        all_values.extend(values)
    assert len(all_values) == len(set(all_values))
    audit = runner.path_id_audit(REPOSITORY_ROOT)
    assert audit["passed"] == 1
    assert audit["pairwise_overlaps"] == []
    assert audit["historical_collisions"] == []
    legacy_dir = tmp_path / "mnist"
    legacy_dir.mkdir()
    legacy = legacy_dir / "legacy.py"
    legacy.write_text("PATH_START = 0xB6000\n", encoding="utf-8")
    with pytest.raises(runner.IntegrityFailure, match="collides"):
        runner.path_id_audit(tmp_path)
    legacy.write_text("PATH_START = 0xB2000\n", encoding="utf-8")
    assert runner.path_id_audit(tmp_path)["passed"] == 1


def test_data_roles_are_balanced_and_below_terminal_slice() -> None:
    assert tuple(runner.AUDIT_TRAIN_POSITIONS) == (
        0,
        25,
        50,
        75,
        100,
        125,
        150,
        175,
        200,
        225,
        1,
        26,
        51,
        76,
        101,
        126,
    )
    assert tuple(runner.ORACLE_VALIDATION_POSITIONS) == tuple(range(0, 100, 10))
    assert tuple(runner.FORWARD_VALIDATION_POSITIONS) == tuple(
        value for digit in range(10) for value in (10 * digit, 10 * digit + 1)
    )
    train_labels = np.repeat(np.arange(10), 25)
    validation_labels = np.repeat(np.arange(10), 10)
    np.testing.assert_array_equal(
        train_labels[np.asarray(runner.AUDIT_TRAIN_POSITIONS)][:10], np.arange(10)
    )
    np.testing.assert_array_equal(
        validation_labels[np.asarray(runner.ORACLE_VALIDATION_POSITIONS)],
        np.arange(10),
    )
    np.testing.assert_array_equal(
        np.bincount(
            validation_labels[np.asarray(runner.FORWARD_VALIDATION_POSITIONS)],
            minlength=10,
        ),
        np.full(10, 2),
    )
    roles = _config()["data"]
    assert roles["train_slice"] == [0, 55_000]
    assert roles["validation_slice"] == [55_000, 60_000]
    assert max(roles["validation_slice"]) <= 60_000
    assert roles["terminal_test_content_rows_parsed"] == 0
    assert roles["whole_file_sha256_read"] == 1


def test_arff_prefix_reader_stops_before_fetching_terminal_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = ["0"] * 784 + ["7"]
    parsed = np.zeros(785, dtype=np.float64)
    parsed[-1] = 7.0
    real_asarray = np.asarray

    def quick_asarray(value: object, *args: Any, **kwargs: Any) -> np.ndarray:
        if value is fields:
            return parsed.copy()
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(runner.np, "asarray", quick_asarray)

    class DataLine:
        def strip(self) -> DataLine:
            return self

        def __bool__(self) -> bool:
            return True

        def startswith(self, _prefix: str) -> bool:
            return False

        def split(self, separator: str) -> list[str]:
            assert separator == ","
            return fields

    row = DataLine()

    def lines():
        yield "@DATA\n"
        for _ in range(60_000):
            yield row
        raise AssertionError("terminal ARFF content was fetched")

    images, labels, access = runner._read_mnist_arff_prefix(lines())  # noqa: SLF001
    assert images.shape == (60_000, 28, 28)
    assert labels.shape == (60_000,)
    assert labels[0] == labels[-1] == 7
    assert access == {
        "content_rows_read": 60_000,
        "last_content_row_index": 59_999,
        "terminal_content_rows_read": 0,
        "last_text_line_number_read": 60_001,
        "full_file_read_purpose": "sha256-only",
    }
    production_source = inspect.getsource(runner.run_production)
    assert "_load_train_validation_mnist_strict(" in production_source
    assert "load_train_validation_mnist(" not in production_source


def test_model_input_firewall_is_unchanged() -> None:
    assert {
        "target_image",
        "source_image",
        "forward_uniforms",
        "uniform_bits",
        "path_id",
    } <= core.FORBIDDEN_MODEL_INPUT_FIELDS
    assert tuple(core.ModelInputs.__dataclass_fields__) == (
        "later_full_state",
        "reverse_time",
        "phase",
        "color",
        "duration",
        "label",
    )
    source = inspect.getsource(candidate)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "ModelInputs")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "ModelInputs")
        )
    ]
    assert calls
    for call in calls:
        keyword_names = {keyword.arg for keyword in call.keywords}
        assert keyword_names.isdisjoint(core.FORBIDDEN_MODEL_INPUT_FIELDS)


def test_raster_rule_is_exact_global_rule() -> None:
    masses = _simplex_rows(3)
    demixed, rendered = runner.rasterize_population(masses)
    expected_demixed = np.maximum(
        (masses - 0.35 / core.STATE_SIZE) / (1.0 - 0.35), 0.0
    )
    expected_demixed /= expected_demixed.sum(axis=1, keepdims=True)
    expected_uint8 = np.rint(
        255.0 * np.clip(expected_demixed * (25_471 / 255), 0.0, 1.0)
    ).astype(np.uint8)
    np.testing.assert_array_equal(demixed, expected_demixed)
    np.testing.assert_array_equal(rendered.reshape(3, -1), expected_uint8)


def test_candidate_dispatch_uses_only_candidate_prepare_and_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_prepare(*, device: object, profile: object) -> SimpleNamespace:
        calls.append(("prepare", (torch.device(device), profile)))
        return SimpleNamespace(
            device=torch.device("cpu"),
            profile=profile,
            candidate_binary_sha256="b" * 64,
        )

    def fake_seed(*, rng_key: Any, prepared: object) -> SimpleNamespace:
        calls.append(("seed", (rng_key, prepared)))
        return SimpleNamespace(key=rng_key)

    enqueue = FakeCandidateEnqueue()
    monkeypatch.setattr(
        candidate, "prepare_alpha1_rb_transition_batch_cuda_candidate", fake_prepare
    )
    monkeypatch.setattr(
        candidate, "prepare_alpha1_rb_transition_cuda_rng_seed", fake_seed
    )
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    runtime = candidate.prepare_candidate_runtime(
        device="cuda:0", rng_keys=[(71, "forward")]
    )
    state = torch.as_tensor(_simplex_rows(1), dtype=torch.float64)
    candidate.candidate_forward_phase(
        state,
        [0xB6200],
        outer_step=0,
        phase=0,
        root_seed=71,
        sample_steps=128,
        runtime=runtime,
    )
    assert [name for name, _ in calls] == ["prepare", "seed"]
    assert len(enqueue.calls) == 1
    sources = "\n".join(
        (
            inspect.getsource(candidate.prepare_candidate_runtime),
            inspect.getsource(candidate.candidate_forward_phase),
            inspect.getsource(candidate._reverse_candidate_half),  # noqa: SLF001
        )
    )
    assert "prepare_alpha1_rb_transition_batch_cuda_candidate" in sources
    assert "enqueue_alpha1_rb_transition_batch_cuda_candidate" in sources
    for forbidden in (
        "prepare_alpha1_rb_transition_batch_cuda_deferred",
        "enqueue_alpha1_rb_transition_batch_cuda_no_fallback",
        "sample_alpha1_rb_transition_batch_cuda(",
        "arb_fallback",
    ):
        assert forbidden not in sources


def test_candidate_profile_is_exactly_128_modes_56_bisections() -> None:
    runtime = _fake_runtime(72)
    assert candidate.CANDIDATE_BACKEND_NAME == "cuda-approximate-candidate-128m-56b"
    assert (
        runtime.profile.candidate_modes,
        runtime.profile.candidate_bisection_steps,
        runtime.profile.threads_per_block,
    ) == (128, 56, 128)
    assert (
        candidate.CANDIDATE_TARGET_SEMANTICS
        == "approximate-candidate Rao--Blackwell target"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_candidate_prepare_and_one_phase_health() -> None:
    root_seed = 0xE14E71
    runtime = candidate.prepare_candidate_runtime(
        device="cuda:0", rng_keys=[(root_seed, "forward")]
    )
    state = torch.full(
        (1, core.STATE_SIZE),
        1.0 / core.STATE_SIZE,
        dtype=torch.float64,
        device=runtime.device,
    )
    output, target, health = candidate.candidate_forward_phase(
        state,
        [0xB6000],
        outer_step=0,
        phase=0,
        root_seed=root_seed,
        sample_steps=128,
        runtime=runtime,
    )
    torch.cuda.synchronize(runtime.device)
    assert runtime.profile.candidate_modes == 128
    assert runtime.profile.candidate_bisection_steps == 56
    assert output.shape == (1, core.STATE_SIZE)
    assert target.shape == (1, core.EDGES_PER_PHASE)
    assert bool(torch.isfinite(output).all().item())
    assert bool(torch.isfinite(target).all().item())
    assert float(torch.max(torch.abs(output.sum(1) - 1.0)).item()) <= 2e-12
    for name in (
        "candidate_invalid_input_count",
        "candidate_invalid_output_count",
        "candidate_nonfinite_count",
        "candidate_negative_bracket_width_count",
        "candidate_bracket_order_invalid_count",
        "candidate_correction_count",
        "candidate_projection_count",
        "candidate_renormalization_count",
    ):
        assert int(health[name].item()) == 0
    assert int(health["candidate_approximation_mismatch_count"].item()) == 0
    assert float(health["maximum_pair_total_error"].item()) <= 2e-12


def test_candidate_phase_preserves_orientation_ids_pair_totals_and_simplex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_seed = 73
    runtime = _fake_runtime(root_seed)
    enqueue = FakeCandidateEnqueue()
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    state = torch.as_tensor(_simplex_rows(2), dtype=torch.float64)
    before = state.clone()
    output, target, health = candidate.candidate_forward_phase(
        state,
        [0xB6200, 0xB6201],
        outer_step=19,
        phase=3,
        root_seed=root_seed,
        sample_steps=128,
        runtime=runtime,
    )
    tails_all, heads_all = candidate.matching_indices(device=state.device)
    color = int(candidate.PHASE_MATCHINGS[3])
    tails, heads = tails_all[color], heads_all[color]
    pair_before = before[:, tails] + before[:, heads]
    pair_after = output[:, tails] + output[:, heads]
    torch.testing.assert_close(pair_after, pair_before, rtol=0.0, atol=2e-12)
    torch.testing.assert_close(
        output.sum(1), torch.ones(2, dtype=torch.float64), rtol=0.0, atol=2e-12
    )
    expected_ids = candidate.canonical_refinement_transition_ids(
        (0xB6200, 0xB6201),
        sample_steps=128,
        outer_step=19,
        phase=3,
        device=state.device,
    ).reshape(2, core.EDGES_PER_PHASE)
    assert torch.equal(enqueue.calls[0]["transition_ids"], expected_ids)
    expected_fraction = before[:, heads] / pair_before
    torch.testing.assert_close(
        enqueue.calls[0]["head_fraction"], expected_fraction, rtol=0.0, atol=0.0
    )
    assert target.shape == (2, core.EDGES_PER_PHASE)
    assert health["candidate_target_semantics"] == candidate.CANDIDATE_TARGET_SEMANTICS


def test_candidate_structural_noop_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    root_seed = 74
    runtime = _fake_runtime(root_seed)
    enqueue = FakeCandidateEnqueue()
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    state = torch.full((1, core.STATE_SIZE), 1.0 / core.STATE_SIZE, dtype=torch.float64)
    tails_all, heads_all = candidate.matching_indices(device=state.device)
    color = int(candidate.PHASE_MATCHINGS[0])
    tail, head = int(tails_all[color][0]), int(heads_all[color][0])
    moved = state[0, tail] + state[0, head]
    state[0, tail] = 0.0
    state[0, head] = 0.0
    state[0, int(heads_all[color][1])] += moved
    output, target, _ = candidate.candidate_forward_phase(
        state,
        [0xB6200],
        outer_step=0,
        phase=0,
        root_seed=root_seed,
        sample_steps=128,
        runtime=runtime,
    )
    assert enqueue.calls[0]["exposure"][0, 0] == 0.0
    assert output[0, tail] == output[0, head] == 0.0
    assert target[0, 0] == 0.0


def test_candidate_invalid_lane_fails_without_projection_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_seed = 75
    runtime = _fake_runtime(root_seed)
    enqueue = FakeCandidateEnqueue(invalid_lane=0)
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    forbidden: list[str] = []
    for name in (
        "enqueue_alpha1_rb_transition_batch_cuda_no_fallback",
        "sample_alpha1_rb_transition_batch_cuda",
    ):
        if hasattr(candidate, name):
            monkeypatch.setattr(
                candidate,
                name,
                lambda *_args, _name=name, **_kwargs: forbidden.append(_name),
            )
    with pytest.raises(candidate.CandidatePilotError, match="failed closed"):
        candidate.forward_terminal_states_candidate(
            _simplex_rows(1),
            [0xB6200],
            root_seed=root_seed,
            runtime=runtime,
        )
    assert forbidden == []


def test_nonfinite_controller_fails_closed_even_when_flow_does_not_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_seed = 0xE14E72
    runtime = _fake_runtime(root_seed)
    monkeypatch.setattr(
        candidate,
        "enqueue_alpha1_rb_transition_batch_cuda_candidate",
        FakeCandidateEnqueue(),
    )
    with pytest.raises(candidate.CandidatePilotError, match="failed closed"):
        candidate.candidate_reverse_outer_step(
            torch.as_tensor(_simplex_rows(1), dtype=torch.float64),
            torch.tensor([3], dtype=torch.long),
            [0xB6500],
            outer_step=127,
            controller="learned",
            root_seed=root_seed,
            runtime=runtime,
            model=NonfiniteRecordingModel(),
        )


def test_candidate_batch_never_exceeds_4096_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_seed = 76
    runtime = _fake_runtime(root_seed)
    enqueue = FakeCandidateEnqueue()
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    with pytest.raises(candidate.CandidatePilotError, match="1..8"):
        candidate.candidate_forward_phase(
            torch.as_tensor(_simplex_rows(9), dtype=torch.float64),
            list(range(0xB6200, 0xB6209)),
            outer_step=0,
            phase=0,
            root_seed=root_seed,
            sample_steps=128,
            runtime=runtime,
        )
    assert enqueue.calls == []
    state = torch.as_tensor(_simplex_rows(8), dtype=torch.float64)
    candidate.candidate_forward_phase(
        state,
        list(range(0xB6200, 0xB6208)),
        outer_step=0,
        phase=0,
        root_seed=root_seed,
        sample_steps=128,
        runtime=runtime,
    )
    assert enqueue.calls[-1]["transition_ids"].numel() == 3136


def test_forward_records_match_core_selection_and_are_labeled_approximate_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_seed = 77
    runtime = _fake_runtime(root_seed)
    enqueue = FakeCandidateEnqueue()
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    callbacks: list[Mapping[str, Any]] = []
    datasets = list(
        candidate.iter_forward_record_batches_candidate(
            _simplex_rows(2),
            np.asarray([2, 7], dtype=np.int64),
            [0xB6200, 0xB6201],
            root_seed=root_seed,
            runtime=runtime,
            outer_step_callback=callbacks.append,
        )
    )
    assert len(datasets) == 1
    dataset = datasets[0]
    assert len(dataset) == 8
    np.testing.assert_array_equal(dataset.outer_steps, np.repeat([15, 47, 79, 111], 2))
    for quartile, outer_step in enumerate((15, 47, 79, 111)):
        rows = np.flatnonzero(dataset.outer_steps == outer_step)
        np.testing.assert_array_equal(
            dataset.phase[rows],
            np.asarray([(row + quartile) % 7 for row in range(2)]),
        )
    assert len(callbacks) == 128
    assert all(
        row["candidate_target_semantics"] == candidate.CANDIDATE_TARGET_SEMANTICS
        for row in callbacks
    )
    assert np.isfinite(dataset.targets).all()


def test_reverse_order_times_microsteps_and_logistic_flow_match_core_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_seed = 78
    runtime = _fake_runtime(root_seed)
    enqueue = FakeCandidateEnqueue()
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    model = ZeroRecordingModel()
    state = torch.as_tensor(_simplex_rows(1), dtype=torch.float64)
    output, telemetry = candidate.candidate_reverse_outer_step(
        state,
        torch.tensor([4], dtype=torch.long),
        [0xB6500],
        outer_step=127,
        controller="learned",
        root_seed=root_seed,
        runtime=runtime,
        model=model,
    )
    expected_key_block = [
        (root_seed, "reverse", 0, "pre"),
        (root_seed, "reverse", 0, "post"),
        (root_seed, "reverse", 1, "pre"),
        (root_seed, "reverse", 1, "post"),
    ]
    assert [row["rng_key"] for row in enqueue.calls] == expected_key_block * 7
    assert len(model.inputs) == 14
    assert [int(inputs.phase[0]) for inputs in model.inputs] == [
        phase for phase in range(6, -1, -1) for _ in range(2)
    ]
    expected_times = [
        core.reverse_midpoint_time(127, phase, micro, sample_steps=128)
        for phase in range(6, -1, -1)
        for micro in range(2)
    ]
    assert [float(inputs.reverse_time[0]) for inputs in model.inputs] == pytest.approx(
        expected_times, abs=2e-8
    )
    assert telemetry["score_count"] == 14 * core.EDGES_PER_PHASE
    torch.testing.assert_close(output.sum(1), torch.ones(1, dtype=torch.float64))


def test_prevalidated_logistic_flow_has_no_tensor_truth_or_host_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = torch.as_tensor(_simplex_rows(2), dtype=torch.float64)
    tails_all, heads_all = candidate.matching_indices(device=state.device)
    tails, heads = tails_all[0], heads_all[0]
    score = torch.linspace(
        -2.0,
        2.0,
        2 * core.EDGES_PER_PHASE,
        dtype=torch.float64,
    ).reshape(2, core.EDGES_PER_PHASE)
    expected = candidate.frozen_score_logistic_flow(
        state, (tails, heads), score, 0.125
    )
    scalar_calls: list[str] = []
    synchronize_calls: list[object] = []

    def scalar_bomb(_tensor: torch.Tensor, *_args: Any, **_kwargs: Any) -> Any:
        scalar_calls.append("host-scalar")
        raise AssertionError("device tensor was materialized in the prevalidated flow")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", scalar_bomb)
        context.setattr(torch.Tensor, "__bool__", scalar_bomb)
        context.setattr(
            torch.cuda,
            "synchronize",
            lambda *args, **kwargs: synchronize_calls.append((args, kwargs)),
        )
        zero = candidate._score_logistic_flow_prevalidated(  # noqa: SLF001
            state, tails, heads, torch.zeros_like(score), 0.125
        )
        actual = candidate._score_logistic_flow_prevalidated(  # noqa: SLF001
            state, tails, heads, score, 0.125
        )
    assert scalar_calls == []
    assert synchronize_calls == []
    assert torch.equal(zero, state)
    pair_before = state[:, tails] + state[:, heads]
    pair_after = actual[:, tails] + actual[:, heads]
    torch.testing.assert_close(pair_after, pair_before, rtol=0.0, atol=2e-12)
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)


def test_null_learned_common_seed_and_transition_id_roles_are_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_seed = 79
    runtime = _fake_runtime(root_seed)
    enqueue = FakeCandidateEnqueue()
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    state = torch.as_tensor(_simplex_rows(2), dtype=torch.float64)
    labels = torch.tensor([1, 8], dtype=torch.long)
    path_ids = [0xB6500, 0xB6501]
    null, _ = candidate.candidate_reverse_outer_step(
        state.clone(),
        labels,
        path_ids,
        outer_step=64,
        controller="null",
        root_seed=root_seed,
        runtime=runtime,
    )
    null_calls = list(enqueue.calls)
    enqueue.calls.clear()
    learned, _ = candidate.candidate_reverse_outer_step(
        state.clone(),
        labels,
        path_ids,
        outer_step=64,
        controller="learned",
        root_seed=root_seed,
        runtime=runtime,
        model=ZeroRecordingModel(),
    )
    assert [row["rng_key"] for row in enqueue.calls] == [
        row["rng_key"] for row in null_calls
    ]
    for actual, expected in zip(enqueue.calls, null_calls, strict=True):
        assert torch.equal(actual["transition_ids"], expected["transition_ids"])
    torch.testing.assert_close(learned, null, rtol=0.0, atol=0.0)


def test_batch_partition_does_not_change_stateless_candidate_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_seed = 80
    runtime = _fake_runtime(root_seed)
    enqueue = FakeCandidateEnqueue()
    monkeypatch.setattr(
        candidate, "enqueue_alpha1_rb_transition_batch_cuda_candidate", enqueue
    )
    states = _simplex_rows(3)
    ids = [0xB6200, 0xB6201, 0xB6202]
    together, together_target, _ = candidate.candidate_forward_phase(
        torch.as_tensor(states, dtype=torch.float64),
        ids,
        outer_step=21,
        phase=4,
        root_seed=root_seed,
        sample_steps=128,
        runtime=runtime,
    )
    pieces = [
        candidate.candidate_forward_phase(
            torch.as_tensor(states[index : index + 1], dtype=torch.float64),
            [ids[index]],
            outer_step=21,
            phase=4,
            root_seed=root_seed,
            sample_steps=128,
            runtime=runtime,
        )
        for index in range(3)
    ]
    torch.testing.assert_close(
        torch.cat([row[0] for row in pieces]), together, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        torch.cat([row[1] for row in pieces]), together_target, rtol=0.0, atol=0.0
    )


def _audit_bank_sha256(bank: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(bank):
        array = np.ascontiguousarray(bank[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _audit_pairing_fixture(
    bank: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    active = np.asarray(bank["active_mask"], dtype=bool)
    randomness = runner._v2_audit_randomness(  # noqa: SLF001
        (runner.SEEDS["candidate_audit"], "k128-candidate-audit"),
        np.asarray(bank["transition_ids"], dtype=np.uint64),
        active,
    )
    adaptive_modes = runner._candidate_adaptive_mode_counts(  # noqa: SLF001
        np.asarray(bank["exposure"], dtype=np.float64)
    )
    outputs = {
        "rng_v2_initial_prefix_numerators": np.asarray(
            randomness["initial_prefix_numerators"], dtype=np.uint64
        ),
        "rng_v2_initial_prefix_bits": np.asarray(
            randomness["initial_prefix_bits"], dtype=np.int32
        ),
        "rng_v2_uniform_midpoints": np.asarray(
            randomness["uniform_midpoints"], dtype=np.float64
        ),
        "candidate_adaptive_modes": adaptive_modes,
        "certified_mask": active.copy(),
        "certified_cuda_mask": active.copy(),
        "certified_fallback_mask": np.zeros_like(active),
    }
    diagnostics = {
        "candidate_modes": 128,
        "candidate_minimum_modes": 128,
        "candidate_maximum_adaptive_modes": int(np.max(adaptive_modes)),
        "candidate_adaptive_above_minimum_count": int(
            np.count_nonzero(adaptive_modes > 128)
        ),
        "audit_rng_contract": randomness["rng_contract"],
        "audit_pairing_contract": randomness["pairing_contract"],
        "audit_canonical_seed": int(randomness["canonical_seed"]),
        "audit_reference_certified_calls": 1,
        "audit_reference_authorizer_calls": 1,
        "audit_reference_fallback_calls": 0,
        "audit_reference_fallback_lane_count": 0,
        "audit_reference_runtime_contract_pass": 1,
    }
    return outputs, diagnostics


def _candidate_backend_fixture_record(
    *, binary_sha256: str = "a" * 64
) -> dict[str, Any]:
    return {
        "schema": runner.VERSION + "-candidate-backend",
        "backend": candidate.CANDIDATE_BACKEND_NAME,
        "target_semantics": candidate.CANDIDATE_TARGET_SEMANTICS,
        "candidate_modes": 128,
        "candidate_modes_semantics": "adaptive minimum",
        "candidate_adaptive_maximum_modes": 1024,
        "candidate_bisection_steps": 56,
        "threads_per_block": 128,
        "dispatch": [
            "prepare_alpha1_rb_transition_batch_cuda_candidate",
            "prepare_alpha1_rb_transition_cuda_rng_seed",
            "enqueue_alpha1_rb_transition_batch_cuda_candidate",
        ],
        "authorizer_calls": 0,
        "certified_cuda_calls": 0,
        "arb_fallback_calls": 0,
        "scope": "production_candidate_runtime_excludes_audit_references",
        "prepared_rng_keys_sha256": runner._semantic_sha256(  # noqa: SLF001
            runner.candidate_rng_keys()
        ),
        "runtime_type": "CandidateRuntime",
        "candidate_binary_sha256": binary_sha256,
    }


def test_512_lane_bank_has_448_phase_and_64_analytic_cases() -> None:
    bank = runner.build_candidate_audit_bank(_simplex_rows(250))
    assert {len(value) for value in bank.values()} == {512}
    assert np.count_nonzero(bank["section"] == 0) == 448
    assert np.count_nonzero(bank["section"] == 1) == 64
    for phase in range(7):
        mask = (bank["section"] == 0) & (bank["phase"] == phase)
        assert np.count_nonzero(mask) == 64
        np.testing.assert_array_equal(
            np.unique(bank["edge_slot"][mask]), np.asarray([0, 97, 196, 391])
        )
    assert np.unique(bank["transition_ids"]).size == 512
    np.testing.assert_array_equal(
        bank["structural_noop_mask"], bank["exposure"] == 0.0
    )
    np.testing.assert_array_equal(bank["active_mask"], bank["exposure"] > 0.0)


def test_audit_bank_hash_and_case_table_are_deterministic() -> None:
    first = runner.build_candidate_audit_bank(_simplex_rows(250))
    second = runner.build_candidate_audit_bank(_simplex_rows(250))
    assert _audit_bank_sha256(first) == _audit_bank_sha256(second)
    analytic = first["section"] == 1
    np.testing.assert_array_equal(
        first["head_fraction"][analytic],
        np.repeat(
            np.asarray([0.0, 2.0**-20, 1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-3, 1.0]),
            8,
        ),
    )
    np.testing.assert_array_equal(
        first["pair_total"][analytic],
        np.tile(np.asarray([0.0, 2.0**-20, 1e-6, 1e-4, 1e-3, 1e-2, 0.25, 1.0]), 8),
    )
    np.testing.assert_array_equal(
        first["analytic_duration_index"][analytic],
        np.tile(np.asarray([0, 1, 2, 3, 4, 5, 6, 3]), 8),
    )
    np.testing.assert_array_equal(
        first["analytic_head_index"][analytic], np.repeat(np.arange(8), 8)
    )
    np.testing.assert_array_equal(
        first["analytic_total_index"][analytic], np.tile(np.arange(8), 8)
    )


def test_v2_64_bit_midpoint_and_full_bank_witness_are_frozen() -> None:
    bank = runner.build_candidate_audit_bank(_simplex_rows(250))
    witness = runner._v2_audit_randomness(  # noqa: SLF001
        (runner.SEEDS["candidate_audit"], "k128-candidate-audit"),
        bank["transition_ids"],
        bank["active_mask"],
    )
    numerators = witness["initial_prefix_numerators"]
    bits = witness["initial_prefix_bits"]
    uniforms = witness["uniform_midpoints"]

    assert witness["canonical_seed"] == 18_189_725_709_840_662_943
    assert runner._array_sha256(numerators) == (  # noqa: SLF001
        "c4b1e0637a80f128466db927f033058a02da8befa5270c19855d2ad5a08d245f"
    )
    assert runner._array_sha256(bits) == (  # noqa: SLF001
        "bdd749d89e0d4a87647c2a5e95b7a783465d3cef1ff3e1f96d22509f246a2e76"
    )
    assert runner._array_sha256(uniforms) == (  # noqa: SLF001
        "353800c9353d2a2ede1ca606af71df294d405b1ca7fd14eea1e50ed2ed352226"
    )
    active = bank["active_mask"]
    np.testing.assert_array_equal(bits[active], np.full(np.count_nonzero(active), 64))
    np.testing.assert_array_equal(numerators[~active], np.zeros(8, dtype=np.uint64))
    np.testing.assert_array_equal(bits[~active], np.zeros(8, dtype=np.int32))
    np.testing.assert_array_equal(uniforms[~active], np.zeros(8, dtype=np.float64))
    assert int(numerators[0]) == 13_781_389_959_716_147_645
    assert float(uniforms[0]).hex() == "0x1.7e82aa4d88082p-1"


def test_candidate_mode_telemetry_reports_minimum_and_actual_adaptation() -> None:
    bank = runner.build_candidate_audit_bank(_simplex_rows(250))
    modes = runner._candidate_adaptive_mode_counts(bank["exposure"])  # noqa: SLF001
    active = bank["active_mask"]

    assert modes.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(modes[~active], np.zeros(8, dtype=np.int32))
    values, counts = np.unique(modes[active], return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist(), strict=True)) == {128: 488, 256: 16}
    assert int(np.max(modes)) == 256


def test_aligned_568_reference_uses_the_frozen_v2_midpoints() -> None:
    transition_ids = np.asarray(
        [0x5B0003FF000, 0x5B0003FF061, 0x5B007BFF187, 0x5B007FFFD87],
        dtype=np.uint64,
    )
    head = np.asarray([0.2, 0.5, 0.8, 0.3], dtype=np.float64)
    exposure = np.asarray([0.0, 0.25, 0.5, 2.0], dtype=np.float64)
    witness = runner._v2_audit_randomness(  # noqa: SLF001
        (runner.SEEDS["candidate_audit"], "k128-candidate-audit"),
        transition_ids,
        exposure > 0.0,
    )
    later, target = runner._fixed_uniform_568_reference(  # noqa: SLF001
        torch.as_tensor(head),
        torch.as_tensor(exposure),
        torch.as_tensor(witness["uniform_midpoints"]),
    )

    assert runner._array_sha256(later.numpy()) == (  # noqa: SLF001
        "5a10ce6c5c5018c7ec70968d7efba9cb57e09fa5ea6ee9f4edcd1123aa9ff632"
    )
    assert runner._array_sha256(target.numpy()) == (  # noqa: SLF001
        "84acf493d24cd001996534814693de2b6e8beac4118fb2da8c59322e42eefc30"
    )
    assert later[0].item() == head[0]
    assert target[0].item() == 0.0


def test_candidate_audit_does_not_call_legacy_random_stream_samplers() -> None:
    source = inspect.getsource(runner.run_candidate_audit)
    assert "propose_alpha1_rb_transition_batch_torch(" not in source
    assert "sample_alpha1_rb_transition_batch(" not in source
    assert "_v2_audit_randomness(" in source
    assert "_fixed_uniform_568_reference(" in source
    assert "sample_alpha1_rb_transition_batch_cuda(" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_candidate_audit_uses_shared_v2_stream_and_aligned_568(
    tmp_path: Path,
) -> None:
    key = (runner.SEEDS["candidate_audit"], "k128-candidate-audit")
    runtime = candidate.prepare_candidate_runtime(device="cuda:0", rng_keys=[key])
    runner._write_json(  # noqa: SLF001
        tmp_path / "stage_ledger.json",
        {"schema": runner.VERSION + "-stage-ledger", "events": []},
    )
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class Governor:
        def admit(self, kind: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(("admit", kind, kwargs))
            return {"passed": 1}

        def complete(self, kind: str, **kwargs: Any) -> None:
            calls.append(("complete", kind, kwargs))

    report = runner.run_candidate_audit(
        tmp_path,
        _simplex_rows(250),
        runtime,
        Governor(),  # type: ignore[arg-type]
    )

    assert runner.gate_b_passed(report)
    assert [row[:2] for row in calls] == [
        ("admit", "candidate_audit"),
        ("complete", "candidate_audit"),
    ]
    assert calls[0][2]["predicted_bytes"] > 0
    assert calls[1][2]["transitions"] == 512 * 3
    assert report["candidate_vs_568"]["maximum_later_fraction_error"] <= (
        runner.FROZEN_CONFIG["numerical_gate"]["maximum_later_error"]
    )
    assert report["candidate_vs_568"]["maximum_target_error"] <= (
        runner.FROZEN_CONFIG["numerical_gate"]["maximum_target_error"]
    )
    assert report["rng_alignment"]["rng_arrays_exact"] == 1
    assert report["rng_alignment"]["candidate_internal_later_maximum_error"] == 0.0
    assert report["rng_alignment"]["candidate_internal_target_maximum_error"] == 0.0
    assert report["candidate_mode_telemetry"] == {
        "minimum_modes": 128,
        "maximum_adaptive_modes": 256,
        "adaptive_above_minimum_count": 16,
        "adaptive_modes_sha256": (
            "e5a756825e13ff8b20b80736047ce29603754a99849a14ecf310d43973485ba8"
        ),
    }
    with np.load(tmp_path / "candidate_audit/outputs.npz", allow_pickle=False) as saved:
        assert runner._array_sha256(  # noqa: SLF001
            saved["rng_v2_initial_prefix_numerators"]
        ) == "c4b1e0637a80f128466db927f033058a02da8befa5270c19855d2ad5a08d245f"
        assert runner._array_sha256(  # noqa: SLF001
            saved["rng_v2_initial_prefix_bits"]
        ) == "bdd749d89e0d4a87647c2a5e95b7a783465d3cef1ff3e1f96d22509f246a2e76"
        assert runner._array_sha256(  # noqa: SLF001
            saved["rng_v2_uniform_midpoints"]
        ) == "353800c9353d2a2ede1ca606af71df294d405b1ca7fd14eea1e50ed2ed352226"


def test_verifier_rebuilds_candidate_bank_against_coordinated_tamper(
    tmp_path: Path,
) -> None:
    train_states = _simplex_rows(250)
    bank = runner.build_candidate_audit_bank(train_states)
    active = np.asarray(bank["active_mask"], dtype=bool)
    pairing_outputs, pairing_diagnostics = _audit_pairing_fixture(bank)
    outputs = {
        "candidate_later": np.asarray(bank["head_fraction"], dtype=np.float64).copy(),
        "candidate_target": np.zeros(512, dtype=np.float64),
        "candidate_lower": np.asarray(bank["head_fraction"], dtype=np.float64).copy(),
        "candidate_upper": np.asarray(bank["head_fraction"], dtype=np.float64).copy(),
        "candidate_approximation_mask": active.copy(),
        "candidate_valid_mask": np.ones(512, dtype=bool),
        "fast_later": np.asarray(bank["head_fraction"], dtype=np.float64).copy(),
        "fast_target": np.zeros(512, dtype=np.float64),
        "certified_later": np.asarray(bank["head_fraction"], dtype=np.float64).copy(),
        "certified_target": np.zeros(512, dtype=np.float64),
        "certified_candidate_later": np.asarray(
            bank["head_fraction"], dtype=np.float64
        ).copy(),
        "certified_candidate_target": np.zeros(512, dtype=np.float64),
        "certified_quantile_lower": np.asarray(
            bank["head_fraction"], dtype=np.float64
        ).copy(),
        "certified_quantile_upper": np.asarray(
            bank["head_fraction"], dtype=np.float64
        ).copy(),
        "certified_target_lower": np.zeros(512, dtype=np.float64),
        "certified_target_upper": np.zeros(512, dtype=np.float64),
        "certified_certificate_codes": np.zeros(512, dtype=np.uint8),
        "certified_prefix_bits": np.zeros(512, dtype=np.int32),
        "transition_ids": np.asarray(bank["transition_ids"], dtype=np.uint64).copy(),
        "earlier_head_fraction": np.asarray(
            bank["head_fraction"], dtype=np.float64
        ).copy(),
        "exposure": np.asarray(bank["exposure"], dtype=np.float64).copy(),
        **pairing_outputs,
    }
    diagnostics = {
        **pairing_diagnostics,
        "candidate_bisection_steps": 56,
        "candidate_binary_sha256": "a" * 64,
        "candidate_batch_type": "CandidateRBCudaBatch",
        "authorizer_calls": 0,
        "fallback_calls": 0,
        "sample_count": 512,
        "candidate_kernel_launch_count": 1,
        "maximum_candidate_bracket_width": 0.0,
        "active_count": int(np.sum(active)),
        "structural_noop_count": int(np.sum(~active)),
        "approximation_count": int(np.sum(active)),
        **{name: 0 for name in runner._CANDIDATE_ZERO_DIAGNOSTICS},
    }
    report = runner.recompute_candidate_audit_metrics(bank, outputs, diagnostics)
    report["certified_diagnostics"] = {
        "sample_count": 512,
        "active_count": int(np.sum(active)),
        "certified_count": int(np.sum(active)),
        "cuda_authorized_count": int(np.sum(active)),
        "fallback_count": 0,
    }
    report["certified_runtime"] = {
        "rng_contract": runner.AUDIT_RNG_CONTRACT,
        "profile": rb_cuda.JacobiRBCudaProfile().to_dict(),
        "runtime_contract_pass": True,
    }
    assert runner.gate_b_passed(report)
    (tmp_path / "candidate_audit").mkdir()
    runner._write_npz(tmp_path / "data_roles.npz", train_mixed_masses=train_states)  # noqa: SLF001
    runner._write_npz(tmp_path / "candidate_audit/bank.npz", **bank)  # noqa: SLF001
    runner._write_npz(tmp_path / "candidate_audit/outputs.npz", **outputs)  # noqa: SLF001
    runner._write_json(tmp_path / "candidate_audit/report.json", report)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_backend.json",
        _candidate_backend_fixture_record(),
    )
    assert runner._verify_candidate_audit(tmp_path)["passed"] == 1  # noqa: SLF001

    original_numerators = outputs["rng_v2_initial_prefix_numerators"].copy()
    outputs["rng_v2_initial_prefix_numerators"][int(np.flatnonzero(active)[0])] ^= np.uint64(1)
    forged_rng_report = runner.recompute_candidate_audit_metrics(
        bank, outputs, diagnostics
    )
    forged_rng_report["certified_diagnostics"] = report["certified_diagnostics"]
    forged_rng_report["certified_runtime"] = report["certified_runtime"]
    assert forged_rng_report["rng_alignment"]["rng_arrays_exact"] == 0
    runner._write_npz(tmp_path / "candidate_audit/outputs.npz", **outputs)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_audit/report.json", forged_rng_report
    )
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="RNG|rng|pair"):
        runner._verify_candidate_audit(tmp_path)  # noqa: SLF001
    outputs["rng_v2_initial_prefix_numerators"] = original_numerators
    runner._write_npz(tmp_path / "candidate_audit/outputs.npz", **outputs)  # noqa: SLF001
    runner._write_json(tmp_path / "candidate_audit/report.json", report)  # noqa: SLF001
    runner._seal_manifest(tmp_path)  # noqa: SLF001

    forged_report = json.loads(json.dumps(report))
    forged_report["diagnostics"]["candidate_batch_type"] = "CertifiedRBCudaBatch"
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_audit/report.json", forged_report
    )
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="candidate.*(?:batch|audit)"):
        runner._verify_candidate_audit(tmp_path)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_audit/report.json", report
    )
    runner._seal_manifest(tmp_path)  # noqa: SLF001

    tampered_bank = {name: np.asarray(value).copy() for name, value in bank.items()}
    lane = int(np.flatnonzero(active)[0])
    tampered_bank["head_fraction"][lane] += 1e-3
    for name in (
        "candidate_later",
        "fast_later",
        "certified_later",
        "certified_candidate_later",
    ):
        outputs[name][lane] += 1e-3
    tampered_report = runner.recompute_candidate_audit_metrics(
        tampered_bank, outputs, diagnostics
    )
    tampered_report["certified_diagnostics"] = report["certified_diagnostics"]
    tampered_report["certified_runtime"] = report["certified_runtime"]
    assert runner.gate_b_passed(tampered_report)
    runner._write_npz(tmp_path / "candidate_audit/bank.npz", **tampered_bank)  # noqa: SLF001
    runner._write_npz(tmp_path / "candidate_audit/outputs.npz", **outputs)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_audit/report.json", tampered_report
    )
    with pytest.raises(runner.IntegrityFailure, match="bank"):
        runner._verify_candidate_audit(tmp_path)  # noqa: SLF001


def test_candidate_backend_semantics_survive_coordinated_manifest_tamper(
    tmp_path: Path,
) -> None:
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_backend.json", _candidate_backend_fixture_record()
    )
    verified = runner._verify_candidate_backend(tmp_path)  # noqa: SLF001
    assert verified["candidate_modes"] == 128
    assert verified["runtime_type"] == "CandidateRuntime"
    forged = _candidate_backend_fixture_record()
    forged["backend"] = "forged-k128-backend"
    runner._write_json(tmp_path / "candidate_backend.json", forged)  # noqa: SLF001
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="candidate backend"):
        runner._verify_candidate_backend(tmp_path)  # noqa: SLF001


def _boundary_audit_inputs(
    *, later_error: float, target_error: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    active = np.ones(512, dtype=bool)
    bank = {
        "head_fraction": np.zeros(512, dtype=np.float64),
        "pair_total": np.ones(512, dtype=np.float64),
        "exposure": np.full(512, 0.5, dtype=np.float64),
        "transition_ids": np.arange(512, dtype=np.uint64),
        "active_mask": active,
        "structural_noop_mask": ~active,
        "section": np.r_[np.zeros(448, dtype=np.int8), np.ones(64, dtype=np.int8)],
        "phase": np.arange(512, dtype=np.int64) % 7,
    }
    pairing_outputs, pairing_diagnostics = _audit_pairing_fixture(bank)
    outputs = {
        "candidate_later": np.zeros(512, dtype=np.float64),
        "candidate_target": np.zeros(512, dtype=np.float64),
        "candidate_approximation_mask": active.copy(),
        "candidate_valid_mask": active.copy(),
        "fast_later": np.full(512, later_error, dtype=np.float64),
        "fast_target": np.full(512, target_error, dtype=np.float64),
        "certified_later": np.full(512, later_error, dtype=np.float64),
        "certified_target": np.full(512, target_error, dtype=np.float64),
        "certified_candidate_later": np.zeros(512, dtype=np.float64),
        "certified_candidate_target": np.zeros(512, dtype=np.float64),
        **pairing_outputs,
    }
    diagnostics = {
        **pairing_diagnostics,
        "candidate_bisection_steps": 56,
        "candidate_binary_sha256": "a" * 64,
        "candidate_batch_type": "CandidateRBCudaBatch",
        "authorizer_calls": 0,
        "fallback_calls": 0,
        "sample_count": 512,
        "candidate_kernel_launch_count": 1,
        "maximum_candidate_bracket_width": 0.0,
        "active_count": 512,
        "structural_noop_count": 0,
        "approximation_count": 512,
    }
    diagnostics.update({name: 0 for name in runner._CANDIDATE_ZERO_DIAGNOSTICS})
    return bank, outputs, diagnostics


def test_gate_b_metric_recomputation_and_boundary_values() -> None:
    bank, outputs, diagnostics = _boundary_audit_inputs(
        later_error=2e-10, target_error=2e-8
    )
    boundary = runner.recompute_candidate_audit_metrics(bank, outputs, diagnostics)
    assert runner.gate_b_passed(boundary)
    assert "scientific_negative" not in boundary
    assert boundary["candidate_vs_568"]["maximum_later_fraction_error"] == 2e-10
    assert boundary["candidate_vs_certified"]["maximum_target_error"] == 2e-8

    outputs["fast_later"][0] = np.nextafter(2e-10, np.inf)
    failed = runner.recompute_candidate_audit_metrics(bank, outputs, diagnostics)
    assert not runner.gate_b_passed(failed)
    outputs["fast_later"][0] = 2e-10
    outputs["certified_target"][0] = np.nextafter(2e-8, np.inf)
    failed = runner.recompute_candidate_audit_metrics(bank, outputs, diagnostics)
    assert not runner.gate_b_passed(failed)
    outputs["certified_target"][0] = 2e-8
    assert runner.gate_b_passed(
        runner.recompute_candidate_audit_metrics(bank, outputs, diagnostics)
    )
    for required in runner._CANDIDATE_REQUIRED_DIAGNOSTICS:
        incomplete = dict(diagnostics)
        incomplete.pop(required)
        report = runner.recompute_candidate_audit_metrics(bank, outputs, incomplete)
        assert not runner.gate_b_passed(report), required


def test_oracle_stage_must_complete_before_any_training_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "training").mkdir()
    (tmp_path / "training/checkpoint_0250.pt").write_bytes(b"too early")
    (tmp_path / "oracle_control").mkdir()
    called: list[str] = []
    monkeypatch.setattr(
        candidate,
        "forward_terminal_states_candidate",
        lambda *_args, **_kwargs: called.append("forward"),
    )
    monkeypatch.setattr(
        runner, "_admit_major_stage", lambda *_args, **_kwargs: {}  # noqa: SLF001
    )
    with pytest.raises(runner.IntegrityFailure, match="before oracle control"):
        runner.run_oracle_control(
            tmp_path,
            _simplex_rows(100),
            np.repeat(np.arange(10), 10),
            np.arange(55_000, 55_100),
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )
    assert called == []
    assert not (tmp_path / "oracle_control/COMPLETE.json").exists()


def _sampling_result(
    starts: np.ndarray,
    *,
    controller: str,
    final_states: np.ndarray | None = None,
) -> core.SamplingResult:
    values = np.ascontiguousarray(starts, dtype=np.float64)
    final = values if final_states is None else np.ascontiguousarray(final_states, dtype=np.float64)
    return core.SamplingResult(
        starts=values.copy(),
        final_states=final.copy(),
        anchors={
            anchor: (values if anchor == 0 else final).copy()
            for anchor in (0, 32, 64, 96, 128)
        },
        telemetry={
            "controller": controller,
            "controller_rms": 0.1 if controller == "learned" else 0.0,
            "finite": 1,
            "nonnegative": 1,
            "candidate_modes": 128,
            "candidate_bisection_steps": 56,
            "backend": candidate.CANDIDATE_BACKEND_NAME,
            "candidate_target_semantics": candidate.CANDIDATE_TARGET_SEMANTICS,
            "candidate_binary_sha256": "a" * 64,
            "candidate_maximum_bracket_width": 0.0,
            "maximum_mass_error": 0.0,
            "maximum_pair_total_error": 0.0,
            "maximum_absolute_q": 0.0,
            "exact_facet_count": 0,
            "by_time_quarter": {},
            "outer_step_seconds": {},
            **{name: 0 for name in runner._SAMPLING_ZERO_COUNTERS},
        },
    )


def _exercise_objective_populations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for relative in (
        "oracle_control",
        "populations/stages/null_prior",
        "populations/stages/learned_prior",
        "populations/stages/forward_terminal_starts",
        "populations/stages/null_forward_terminal",
        "populations/stages/learned_forward_terminal",
        "populations/contact_sheets",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    runner._write_json(  # noqa: SLF001
        tmp_path / "stage_ledger.json", {"schema": "test", "events": []}
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_backend.json", _candidate_backend_fixture_record()
    )
    oracle_targets = _simplex_rows(10)
    oracle_labels = np.arange(10, dtype=np.int64)
    oracle_path_ids = np.arange(0xB6540, 0xB654A, dtype=np.int64)
    oracle_sample_ids = np.asarray([f"oracle-{value:05x}" for value in oracle_path_ids])
    runner._write_npz(  # noqa: SLF001
        tmp_path / "oracle_control/authority.npz",
        source_targets=oracle_targets,
        requested_labels=oracle_labels,
        path_ids=oracle_path_ids,
        sample_ids=oracle_sample_ids,
    )
    oracle_null = _sampling_result(oracle_targets, controller="null")
    oracle = _sampling_result(oracle_targets, controller="oracle")
    runner._save_sampling_npz(  # noqa: SLF001
        tmp_path / "oracle_control/null.npz",
        oracle_null,
        oracle_labels,
        oracle_path_ids,
        oracle_sample_ids,
        source_targets=oracle_targets,
        terminal_starts=oracle_null.starts,
    )
    runner._save_sampling_npz(  # noqa: SLF001
        tmp_path / "oracle_control/oracle.npz",
        oracle,
        oracle_labels,
        oracle_path_ids,
        oracle_sample_ids,
        source_targets=oracle_targets,
        terminal_starts=oracle.starts,
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "oracle_control/COMPLETE.json",
        {
            "authority_sha256": _sha256(tmp_path / "oracle_control/authority.npz"),
            "null_sha256": _sha256(tmp_path / "oracle_control/null.npz"),
            "oracle_sha256": _sha256(tmp_path / "oracle_control/oracle.npz"),
            "passed": 1,
        },
    )
    calls: list[dict[str, Any]] = []

    def fake_reverse_cohorts(
        _run_dir: Path,
        stage_directory: Path,
        starts: np.ndarray,
        labels: np.ndarray,
        path_ids: np.ndarray,
        sample_ids: np.ndarray,
        *,
        controller: str,
        image_directory: Path | None = None,
        **_kwargs: Any,
    ) -> core.SamplingResult:
        authority = tmp_path / "populations/prior_start_authority.npz"
        calls.append(
            {
                "controller": controller,
                "stage": stage_directory.name,
                "authority_committed": authority.is_file(),
                "starts": np.asarray(starts).copy(),
                "labels": np.asarray(labels).copy(),
                "path_ids": np.asarray(path_ids).copy(),
            }
        )
        final = np.roll(np.asarray(starts), 1, axis=1)
        result = _sampling_result(starts, controller=controller, final_states=final)
        for cohort_index, start in enumerate(range(0, len(path_ids), 8)):
            stop = min(start + 8, len(path_ids))
            part = _sampling_result(
                starts[start:stop],
                controller=controller,
                final_states=final[start:stop],
            )
            runner._save_sampling_npz(  # noqa: SLF001
                stage_directory / f"cohort_{cohort_index:03d}.npz",
                part,
                labels[start:stop],
                path_ids[start:stop],
                sample_ids[start:stop],
            )
        if image_directory is not None:
            _, images = runner.rasterize_population(result.final_states)
            runner._save_individual_pngs(image_directory, images, sample_ids)  # noqa: SLF001
        return result

    def fake_forward_starts(
        _run_dir: Path,
        targets: np.ndarray,
        labels: np.ndarray,
        path_ids: np.ndarray,
        sample_ids: np.ndarray,
        **_kwargs: Any,
    ) -> np.ndarray:
        for cohort_index, start in enumerate(range(0, len(path_ids), 8)):
            stop = min(start + 8, len(path_ids))
            runner._write_npz(  # noqa: SLF001
                tmp_path
                / f"populations/stages/forward_terminal_starts/cohort_{cohort_index:03d}.npz",
                source_targets=targets[start:stop],
                terminal_starts=targets[start:stop],
                requested_labels=labels[start:stop],
                path_ids=path_ids[start:stop],
                sample_ids=sample_ids[start:stop],
            )
            runner._write_json(  # noqa: SLF001
                tmp_path
                / f"populations/stages/forward_terminal_starts/cohort_{cohort_index:03d}.telemetry.json",
                _healthy_forward_telemetry(),
            )
        return np.asarray(targets, dtype=np.float64)

    monkeypatch.setattr(runner, "_run_reverse_cohorts", fake_reverse_cohorts)
    monkeypatch.setattr(runner, "_forward_terminal_starts_cohorts", fake_forward_starts)
    validation_states = _simplex_rows(100)
    validation_labels = np.repeat(np.arange(10), 10)
    validation_arff_indices = np.arange(55_000, 55_100, dtype=np.int64)
    runner._write_npz(  # noqa: SLF001
        tmp_path / "data_roles.npz",
        validation_mixed_masses=validation_states,
        validation_labels=validation_labels,
        validation_arff_indices=validation_arff_indices,
    )
    populations = runner.sample_objective_populations(
        tmp_path,
        validation_states,
        validation_labels,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        oracle_null,
        oracle,
    )
    return populations, calls


def test_oracle_results_are_reused_not_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    populations, calls = _exercise_objective_populations(tmp_path, monkeypatch)
    assert populations["raw"]["oracle"].shape == (10, core.STATE_SIZE)
    assert {row["controller"] for row in calls} == {"null", "learned"}
    stage_index = json.loads(
        (tmp_path / "populations/stage_index.json").read_text(encoding="utf-8")
    )
    assert stage_index["oracle_recomputed_after_training"] == 0
    assert set(stage_index["stages"]) == {
        "null_prior",
        "learned_prior",
        "null_forward_terminal",
        "learned_forward_terminal",
        "null_oracle",
        "oracle",
    }
    assert stage_index["oracle_complete_sha256"] == _sha256(
        tmp_path / "oracle_control/COMPLETE.json"
    )
    complete = json.loads(
        (tmp_path / "oracle_control/COMPLETE.json").read_text(encoding="utf-8")
    )
    for name, expected in (("null_oracle", "null"), ("oracle", "oracle")):
        assembled = tmp_path / f"populations/{name}.npz"
        source = tmp_path / f"oracle_control/{expected}.npz"
        assert assembled.is_file()
        with np.load(assembled, allow_pickle=False) as stage:
            assert stage["final_states"].shape == (10, core.STATE_SIZE)
            assert {
                "starts",
                "anchors_000",
                "anchors_032",
                "anchors_064",
                "anchors_096",
                "anchors_128",
                "final_states",
                "requested_labels",
                "path_ids",
                "sample_ids",
            } <= set(stage.files)
        assert complete[f"{'null' if name == 'null_oracle' else 'oracle'}_sha256"] == _sha256(
            source
        )
        row = stage_index["stages"][name]
        assert row["row_count"] == 10
        assert row["assembled_sha256"] == _sha256(assembled)
        assert row["reused_source_sha256"] == _sha256(source)


def _healthy_oracle_telemetry() -> dict[str, Any]:
    return {
        "controller": "oracle",
        "controller_rms": 0.0,
        "finite": 1,
        "nonnegative": 1,
        "candidate_modes": 128,
        "candidate_bisection_steps": 56,
        "backend": candidate.CANDIDATE_BACKEND_NAME,
        "candidate_target_semantics": candidate.CANDIDATE_TARGET_SEMANTICS,
        "candidate_binary_sha256": "a" * 64,
        "candidate_maximum_bracket_width": 0.0,
        "maximum_mass_error": 0.0,
        "maximum_pair_total_error": 0.0,
        "maximum_absolute_q": 0.0,
        "exact_facet_count": 0,
        "by_time_quarter": {},
        "outer_step_seconds": {},
        **{name: 0 for name in runner._SAMPLING_ZERO_COUNTERS},
    }


def _healthy_forward_telemetry() -> dict[str, Any]:
    return {
        "backend": candidate.CANDIDATE_BACKEND_NAME,
        "candidate_target_semantics": candidate.CANDIDATE_TARGET_SEMANTICS,
        "candidate_modes": 128,
        "candidate_bisection_steps": 56,
        "candidate_binary_sha256": "a" * 64,
        "outer_steps": 128,
        "outer_step_seconds": [0.0] * 128,
        "candidate_maximum_bracket_width": 0.0,
        "maximum_mass_error": 0.0,
        "maximum_pair_total_error": 0.0,
        **{name: 0 for name in runner._SAMPLING_ZERO_COUNTERS},
    }


def test_gate_c_uses_whole_path_final_raw_mass_l1() -> None:
    targets = _simplex_rows(10)
    null = np.roll(targets, 1, axis=1)
    oracle = targets.copy()
    report = runner.oracle_control_metrics(
        targets,
        null,
        oracle,
        _healthy_oracle_telemetry(),
        _healthy_oracle_telemetry(),
    )
    expected_null = np.sum(np.abs(null - targets), axis=1)
    np.testing.assert_allclose(report["null_final_raw_mass_l1"], expected_null)
    np.testing.assert_array_equal(report["oracle_final_raw_mass_l1"], np.zeros(10))
    assert report["oracle_improved_path_count"] == 10
    assert runner.gate_c_passed(report)
    assert "scientific_negative" not in report

    oracle[-1] = null[-1]
    nine_wins = runner.oracle_control_metrics(
        targets,
        null,
        oracle,
        _healthy_oracle_telemetry(),
        _healthy_oracle_telemetry(),
    )
    assert nine_wins["oracle_improved_path_count"] == 9
    assert runner.gate_c_passed(nine_wins)
    oracle[-2] = null[-2]
    eight_wins = runner.oracle_control_metrics(
        targets,
        null,
        oracle,
        _healthy_oracle_telemetry(),
        _healthy_oracle_telemetry(),
    )
    assert not runner.gate_c_passed(eight_wins)


def _write_oracle_verifier_fixture(run_dir: Path) -> None:
    for relative in (
        "oracle_control/stages/forward_terminal",
        "oracle_control/stages/null",
        "oracle_control/stages/oracle",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    runner._write_json(  # noqa: SLF001
        run_dir / "candidate_backend.json", _candidate_backend_fixture_record()
    )
    targets = _simplex_rows(10)
    terminal = np.roll(targets, 1, axis=1)
    labels = np.arange(10, dtype=np.int64)
    arff_indices = np.arange(55_000, 55_010, dtype=np.int64)
    path_ids = np.arange(0xB6540, 0xB654A, dtype=np.int64)
    sample_ids = np.asarray([f"oracle-{value:05x}" for value in path_ids])
    validation_states = _simplex_rows(100)
    validation_labels = np.repeat(np.arange(10), 10)
    validation_arff_indices = np.arange(55_000, 55_100, dtype=np.int64)
    validation_states[runner.ORACLE_VALIDATION_POSITIONS] = targets
    validation_labels[runner.ORACLE_VALIDATION_POSITIONS] = labels
    validation_arff_indices[runner.ORACLE_VALIDATION_POSITIONS] = arff_indices
    runner._write_npz(  # noqa: SLF001
        run_dir / "data_roles.npz",
        validation_mixed_masses=validation_states,
        validation_labels=validation_labels,
        validation_arff_indices=validation_arff_indices,
    )
    runner._write_npz(  # noqa: SLF001
        run_dir / "oracle_control/authority.npz",
        source_targets=targets,
        requested_labels=labels,
        arff_indices=arff_indices,
        path_ids=path_ids,
        sample_ids=sample_ids,
    )
    telemetry = _healthy_oracle_telemetry()
    null = core.SamplingResult(
        starts=terminal.copy(),
        final_states=terminal.copy(),
        anchors={anchor: terminal.copy() for anchor in (0, 32, 64, 96, 128)},
        telemetry={**telemetry, "controller": "null"},
    )
    oracle = core.SamplingResult(
        starts=terminal.copy(),
        final_states=targets.copy(),
        anchors={
            anchor: (terminal if anchor == 0 else targets).copy()
            for anchor in (0, 32, 64, 96, 128)
        },
        telemetry={**telemetry, "controller": "oracle"},
    )
    for cohort_index, start in enumerate((0, 8)):
        stop = min(start + 8, 10)
        runner._write_npz(  # noqa: SLF001
            run_dir
            / f"oracle_control/stages/forward_terminal/cohort_{cohort_index:03d}.npz",
            source_targets=targets[start:stop],
            terminal_starts=terminal[start:stop],
            requested_labels=labels[start:stop],
            arff_indices=arff_indices[start:stop],
            path_ids=path_ids[start:stop],
            sample_ids=sample_ids[start:stop],
        )
        runner._write_json(  # noqa: SLF001
            run_dir
            / f"oracle_control/stages/forward_terminal/cohort_{cohort_index:03d}.telemetry.json",
            _healthy_forward_telemetry(),
        )
        for name, result in (("null", null), ("oracle", oracle)):
            part = core.SamplingResult(
                starts=result.starts[start:stop],
                final_states=result.final_states[start:stop],
                anchors={key: value[start:stop] for key, value in result.anchors.items()},
                telemetry=result.telemetry,
            )
            runner._save_sampling_npz(  # noqa: SLF001
                run_dir / f"oracle_control/stages/{name}/cohort_{cohort_index:03d}.npz",
                part,
                labels[start:stop],
                path_ids[start:stop],
                sample_ids[start:stop],
            )
    for name, result in (("null", null), ("oracle", oracle)):
        runner._save_sampling_npz(  # noqa: SLF001
            run_dir / f"oracle_control/{name}.npz",
            result,
            labels,
            path_ids,
            sample_ids,
            source_targets=targets,
            terminal_starts=terminal,
        )
    for name, rows in (
        ("targets", targets),
        ("null", null.final_states),
        ("oracle", oracle.final_states),
    ):
        _, images = runner.rasterize_population(rows)
        runner._save_individual_pngs(  # noqa: SLF001
            run_dir / f"oracle_control/images/{name}", images, sample_ids
        )
    target_images = runner.rasterize_population(targets)[1]
    null_images = runner.rasterize_population(null.final_states)[1]
    oracle_images = runner.rasterize_population(oracle.final_states)[1]
    runner.write_contact_sheet(
        run_dir / "oracle_control/contact_sheet.png",
        np.stack(
            [
                row
                for triple in zip(
                    target_images, null_images, oracle_images, strict=True
                )
                for row in triple
            ]
        ),
        columns=3,
        captions=[
            f"{int(label)}:{name}"
            for label in labels
            for name in ("target", "null", "oracle")
        ],
    )
    metrics = runner.oracle_control_metrics(
        targets, null.final_states, oracle.final_states, null.telemetry, oracle.telemetry
    )
    metrics["anchor_raw_mass_l1"] = {
        str(anchor): {
            "null": np.sum(np.abs(null.anchors[anchor] - targets), axis=1),
            "oracle": np.sum(np.abs(oracle.anchors[anchor] - targets), axis=1),
        }
        for anchor in (0, 32, 64, 96, 128)
    }
    runner._write_json(run_dir / "oracle_control/metrics.json", metrics)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        run_dir / "oracle_control/COMPLETE.json",
        {
            "passed": 1,
            "authority_sha256": _sha256(run_dir / "oracle_control/authority.npz"),
            "null_sha256": _sha256(run_dir / "oracle_control/null.npz"),
            "oracle_sha256": _sha256(run_dir / "oracle_control/oracle.npz"),
            "metrics_sha256": _sha256(run_dir / "oracle_control/metrics.json"),
        },
    )


def test_oracle_verifier_replays_identity_and_terminal_start_alignment(
    tmp_path: Path,
) -> None:
    _write_oracle_verifier_fixture(tmp_path)
    assert runner._verify_oracle_control(tmp_path)["passed"] == 1  # noqa: SLF001
    oracle_path = tmp_path / "oracle_control/oracle.npz"
    arrays = runner._npz_arrays(oracle_path)  # noqa: SLF001
    arrays["path_ids"] = np.roll(arrays["path_ids"], 1)
    runner._write_npz(oracle_path, **arrays)  # noqa: SLF001
    complete = json.loads(
        (tmp_path / "oracle_control/COMPLETE.json").read_text(encoding="utf-8")
    )
    complete["oracle_sha256"] = _sha256(oracle_path)
    runner._write_json(  # noqa: SLF001
        tmp_path / "oracle_control/COMPLETE.json", complete
    )
    with pytest.raises(runner.IntegrityFailure, match="alignment|identity|path"):
        runner._verify_oracle_control(tmp_path)  # noqa: SLF001


def test_resource_stopped_partial_oracle_replays_saved_cohort_authority(
    tmp_path: Path,
) -> None:
    _write_oracle_verifier_fixture(tmp_path)
    for relative in (
        "oracle_control/null.npz",
        "oracle_control/null.telemetry.json",
        "oracle_control/oracle.npz",
        "oracle_control/oracle.telemetry.json",
        "oracle_control/metrics.json",
        "oracle_control/COMPLETE.json",
    ):
        (tmp_path / relative).unlink()
    for stage in ("forward_terminal", "null", "oracle"):
        for suffix in ("npz", "telemetry.json"):
            (tmp_path / f"oracle_control/stages/{stage}/cohort_001.{suffix}").unlink()
    runner._verify_partial_oracle_control(tmp_path)  # noqa: SLF001

    cohort_path = tmp_path / "oracle_control/stages/null/cohort_000.npz"
    arrays = runner._npz_arrays(cohort_path)  # noqa: SLF001
    arrays["starts"][0] = np.roll(arrays["starts"][0], 1)
    arrays["anchors_000"][0] = arrays["starts"][0]
    arrays["final_states"][0] = arrays["starts"][0]
    arrays["anchors_128"][0] = arrays["starts"][0]
    runner._write_npz(cohort_path, **arrays)  # noqa: SLF001
    _, rendered = runner.rasterize_population(arrays["final_states"])
    runner._save_individual_pngs(  # noqa: SLF001
        tmp_path / "oracle_control/images/null",
        rendered,
        arrays["sample_ids"],
    )
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="starts|partial oracle"):
        runner._verify_partial_oracle_control(tmp_path)  # noqa: SLF001


def test_checkpoint_selection_is_earliest_minimum_at_250_500_750() -> None:
    history = [
        {"update": 250, "eligible": 1, "validation_normalized_mse": 0.4},
        {"update": 500, "eligible": 1, "validation_normalized_mse": 0.4},
        {"update": 750, "eligible": 1, "validation_normalized_mse": 0.6},
    ]
    selection = runner.select_earliest_checkpoint(history)
    assert selection["selected_update"] == 250
    assert selection["selected_validation_normalized_mse"] == 0.4


def test_update_zero_is_not_eligible() -> None:
    history = [
        {"update": 0, "eligible": 1, "validation_normalized_mse": 0.0},
        {"update": 250, "eligible": 1, "validation_normalized_mse": 0.5},
        {"update": 500, "eligible": 1, "validation_normalized_mse": 0.6},
        {"update": 750, "eligible": 1, "validation_normalized_mse": 0.7},
    ]
    selection = runner.select_earliest_checkpoint(history)
    assert selection["selected_update"] == 250
    assert selection["update_zero_eligible"] == 0


def test_resource_stopped_partial_checkpoint_replays_payload_without_selection(
    tmp_path: Path,
) -> None:
    (tmp_path / "training").mkdir()
    model_state = core.make_model().state_dict()
    payload = {
        "schema": runner.VERSION + "-training-checkpoint",
        "target_semantics": candidate.CANDIDATE_TARGET_SEMANTICS,
        "completed_update": 250,
        "model_state_dict": model_state,
        "ema_state_dict": {name: value.clone() for name, value in model_state.items()},
        "optimizer_state_dict": {},
        "history": [{"update": 250, "training_loss": 1.0}],
    }
    checkpoint = tmp_path / "training/checkpoint_0250.pt"
    torch.save(payload, checkpoint)
    assert not (tmp_path / "training/selection.json").exists()
    runner._verify_partial_training(tmp_path)  # noqa: SLF001
    payload["history"][-1]["update"] = 249
    torch.save(payload, checkpoint)
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="partial training checkpoint"):
        runner._verify_partial_training(tmp_path)  # noqa: SLF001


def test_prior_starts_are_label_independent_and_bound_before_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, calls = _exercise_objective_populations(tmp_path, monkeypatch)
    prior_calls = [row for row in calls if row["stage"] in {"null_prior", "learned_prior"}]
    assert len(prior_calls) == 2
    assert all(row["authority_committed"] for row in prior_calls)
    np.testing.assert_array_equal(prior_calls[0]["starts"], prior_calls[1]["starts"])
    np.testing.assert_array_equal(prior_calls[0]["path_ids"], prior_calls[1]["path_ids"])
    authority = json.loads(
        (tmp_path / "populations/prior_start_authority.json").read_text(encoding="utf-8")
    )
    assert authority["committed_before_sampling"] == 1
    assert authority["label_independent_start_law"] == "Dirichlet(1,...,1)"
    with np.load(tmp_path / "populations/prior_start_authority.npz", allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank["prior_starts"], prior_calls[0]["starts"])
        np.testing.assert_array_equal(bank["requested_labels"], np.repeat(np.arange(10), 2))
        repeated = core.sample_dirichlet_starts(
            bank["path_ids"], root_seed=runner.SEEDS["prior_start"]
        )
        np.testing.assert_array_equal(repeated, bank["prior_starts"])


def test_null_and_learned_populations_keep_every_declared_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_objective_populations(tmp_path, monkeypatch)
    for left_name, right_name in (
        ("null_prior", "learned_prior"),
        ("null_forward_terminal", "learned_forward_terminal"),
    ):
        with (
            np.load(tmp_path / f"populations/{left_name}.npz", allow_pickle=False) as left,
            np.load(tmp_path / f"populations/{right_name}.npz", allow_pickle=False) as right,
        ):
            assert left["final_states"].shape == right["final_states"].shape == (
                20,
                core.STATE_SIZE,
            )
            np.testing.assert_array_equal(left["path_ids"], right["path_ids"])
            np.testing.assert_array_equal(left["requested_labels"], right["requested_labels"])
            np.testing.assert_array_equal(left["starts"], right["starts"])
            assert np.unique(left["path_ids"]).size == 20
    with np.load(tmp_path / "populations/raw_populations.npz", allow_pickle=False) as raw:
        assert raw["null_prior"].shape[0] == raw["learned_prior"].shape[0] == 20
        assert (
            raw["null_forward_terminal"].shape[0]
            == raw["learned_forward_terminal"].shape[0]
            == 20
        )
        assert raw["null_oracle"].shape[0] == raw["oracle"].shape[0] == 10


def test_all_required_anchors_and_individual_pngs_are_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_objective_populations(tmp_path, monkeypatch)
    required = {
        "anchors_000",
        "anchors_032",
        "anchors_064",
        "anchors_096",
        "anchors_128",
    }
    for name in (
        "null_prior",
        "learned_prior",
        "null_forward_terminal",
        "learned_forward_terminal",
    ):
        with np.load(tmp_path / f"populations/{name}.npz", allow_pickle=False) as stage:
            assert required <= set(stage.files)
            assert all(stage[key].shape[0] == 20 for key in required)
    expected_counts = {
        "prior/null": 20,
        "prior/learned": 20,
        "forward/null": 20,
        "forward/learned": 20,
        "forward/targets": 20,
        "oracle/null": 10,
        "oracle/oracle": 10,
        "oracle/targets": 10,
    }
    for relative, expected in expected_counts.items():
        images = list((tmp_path / "populations/images" / relative).glob("*.png"))
        assert len(images) == expected, relative


def _reseal_population_fixture(run_dir: Path) -> None:
    seal_path = run_dir / "POPULATIONS_SEALED.json"
    seal_path.unlink(missing_ok=True)
    paths = sorted(
        path
        for root_name in ("populations", "oracle_control")
        for path in (run_dir / root_name).rglob("*")
        if path.is_file()
    )
    rows = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    runner._write_json(  # noqa: SLF001
        seal_path,
        {
            "schema": runner.VERSION + "-population-seal",
            "sealed_before_evaluator_or_review": 1,
            "artifact_count": len(rows),
            "tree_digest": runner._tree_digest(rows),  # noqa: SLF001
            "artifacts": rows,
        },
    )


def _roll_stage_identities(run_dir: Path, stage_name: str) -> None:
    stage_path = run_dir / f"populations/{stage_name}.npz"
    arrays = runner._npz_arrays(stage_path)  # noqa: SLF001
    permutation = np.roll(np.arange(len(arrays["path_ids"])), 1)
    identity_names = ("requested_labels", "path_ids", "sample_ids")
    for name in identity_names:
        arrays[name] = arrays[name][permutation]
    runner._write_npz(stage_path, **arrays)  # noqa: SLF001
    offset = 0
    for cohort_path in sorted(
        (run_dir / f"populations/stages/{stage_name}").glob("cohort_*.npz")
    ):
        cohort = runner._npz_arrays(cohort_path)  # noqa: SLF001
        stop = offset + len(cohort["path_ids"])
        for name in identity_names:
            cohort[name] = arrays[name][offset:stop]
        runner._write_npz(cohort_path, **cohort)  # noqa: SLF001
        offset = stop
    assert offset == len(permutation)
    index_path = run_dir / "populations/stage_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["stages"][stage_name]["path_ids_sha256"] = runner._array_sha256(  # noqa: SLF001
        arrays["path_ids"]
    )
    index["stages"][stage_name]["assembled_sha256"] = _sha256(stage_path)
    runner._write_json(index_path, index)  # noqa: SLF001


def test_population_verifier_binds_learned_identity_to_null_and_frozen_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_objective_populations(tmp_path, monkeypatch)
    _reseal_population_fixture(tmp_path)
    runner._verify_populations(tmp_path)  # noqa: SLF001
    _roll_stage_identities(tmp_path, "learned_prior")
    _reseal_population_fixture(tmp_path)
    with pytest.raises(runner.IntegrityFailure, match="identity|paired|prior"):
        runner._verify_populations(tmp_path)  # noqa: SLF001


def test_population_verifier_replays_prior_start_authority_after_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_objective_populations(tmp_path, monkeypatch)
    _reseal_population_fixture(tmp_path)
    runner._verify_populations(tmp_path)  # noqa: SLF001
    authority_path = tmp_path / "populations/prior_start_authority.npz"
    authority = runner._npz_arrays(authority_path)  # noqa: SLF001
    permutation = np.roll(np.arange(len(authority["path_ids"])), 1)
    for name in authority:
        authority[name] = authority[name][permutation]
    runner._write_npz(authority_path, **authority)  # noqa: SLF001
    authority_json_path = tmp_path / "populations/prior_start_authority.json"
    authority_json = json.loads(authority_json_path.read_text(encoding="utf-8"))
    authority_json["npz_sha256"] = _sha256(authority_path)
    authority_json["path_ids_sha256"] = runner._array_sha256(  # noqa: SLF001
        authority["path_ids"]
    )
    runner._write_json(authority_json_path, authority_json)  # noqa: SLF001
    _reseal_population_fixture(tmp_path)
    with pytest.raises(runner.IntegrityFailure, match="prior.*authority|path"):
        runner._verify_populations(tmp_path)  # noqa: SLF001


def test_population_verifier_binds_forward_identities_to_validation_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_objective_populations(tmp_path, monkeypatch)
    _reseal_population_fixture(tmp_path)
    runner._verify_populations(tmp_path)  # noqa: SLF001
    _roll_stage_identities(tmp_path, "null_forward_terminal")
    _roll_stage_identities(tmp_path, "learned_forward_terminal")

    identities_path = tmp_path / "populations/identities.npz"
    identities = runner._npz_arrays(identities_path)  # noqa: SLF001
    permutation = np.roll(np.arange(20), 1)
    for name in (
        "forward_requested_labels",
        "forward_path_ids",
        "forward_sample_ids",
    ):
        identities[name] = identities[name][permutation]
    runner._write_npz(identities_path, **identities)  # noqa: SLF001
    archives: dict[str, dict[str, np.ndarray]] = {}
    for archive_name in (
        "raw_populations",
        "demixed_populations",
        "uint8_populations",
    ):
        path = tmp_path / f"populations/{archive_name}.npz"
        archive = runner._npz_arrays(path)  # noqa: SLF001
        for name in (
            "forward_requested_labels",
            "forward_path_ids",
            "forward_sample_ids",
        ):
            archive[name] = identities[name]
        runner._write_npz(path, **archive)  # noqa: SLF001
        archives[archive_name] = archive
    rendered = archives["uint8_populations"]
    for directory, name in (
        ("forward/null", "null_forward_terminal"),
        ("forward/learned", "learned_forward_terminal"),
        ("forward/targets", "forward_targets"),
    ):
        runner._save_individual_pngs(  # noqa: SLF001
            tmp_path / f"populations/images/{directory}",
            rendered[name],
            identities["forward_sample_ids"],
        )
    runner.write_contact_sheet(
        tmp_path / "populations/contact_sheets/forward_target_null_learned.png",
        np.stack(
            [
                row
                for triple in zip(
                    rendered["forward_targets"],
                    rendered["null_forward_terminal"],
                    rendered["learned_forward_terminal"],
                    strict=True,
                )
                for row in triple
            ]
        ),
        columns=6,
        captions=[
            f"{int(label)}:{kind}"
            for label in identities["forward_requested_labels"]
            for kind in ("target", "null", "learned")
        ],
    )
    _reseal_population_fixture(tmp_path)
    with pytest.raises(runner.IntegrityFailure, match="forward.*identity|validation"):
        runner._verify_populations(tmp_path)  # noqa: SLF001


def test_population_verifier_rejects_coordinated_assembled_raw_index_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_objective_populations(tmp_path, monkeypatch)
    _reseal_population_fixture(tmp_path)
    runner._verify_populations(tmp_path)  # noqa: SLF001

    stage_path = tmp_path / "populations/null_prior.npz"
    stage = runner._npz_arrays(stage_path)  # noqa: SLF001
    stage["final_states"][0] = np.roll(stage["final_states"][0], 1)
    stage["anchors_128"][0] = stage["final_states"][0]
    runner._write_npz(stage_path, **stage)  # noqa: SLF001

    raw_path = tmp_path / "populations/raw_populations.npz"
    raw = runner._npz_arrays(raw_path)  # noqa: SLF001
    raw["null_prior"][0] = stage["final_states"][0]
    runner._write_npz(raw_path, **raw)  # noqa: SLF001
    demixed_path = tmp_path / "populations/demixed_populations.npz"
    demixed = runner._npz_arrays(demixed_path)  # noqa: SLF001
    uint8_path = tmp_path / "populations/uint8_populations.npz"
    rendered = runner._npz_arrays(uint8_path)  # noqa: SLF001
    demixed["null_prior"], rendered["null_prior"] = runner.rasterize_population(
        raw["null_prior"]
    )
    runner._write_npz(demixed_path, **demixed)  # noqa: SLF001
    runner._write_npz(uint8_path, **rendered)  # noqa: SLF001
    identities = runner._npz_arrays(  # noqa: SLF001
        tmp_path / "populations/identities.npz"
    )
    runner._save_individual_pngs(  # noqa: SLF001
        tmp_path / "populations/images/prior/null",
        rendered["null_prior"],
        identities["prior_sample_ids"],
    )
    stage_index_path = tmp_path / "populations/stage_index.json"
    stage_index = json.loads(stage_index_path.read_text(encoding="utf-8"))
    stage_index["stages"]["null_prior"]["assembled_sha256"] = _sha256(stage_path)
    runner._write_json(stage_index_path, stage_index)  # noqa: SLF001
    _reseal_population_fixture(tmp_path)
    with pytest.raises(runner.IntegrityFailure, match="cohort|assembled"):
        runner._verify_populations(tmp_path)  # noqa: SLF001


def test_evaluator_torch_load_is_blocked_before_population_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[object] = []
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: loads.append((args, kwargs)))
    with pytest.raises(runner.IntegrityFailure, match="seal is missing"):
        runner.evaluate_sealed_populations(
            tmp_path,
            {},
            runner.EvaluatorFirewall(),
            None,  # type: ignore[arg-type]
        )
    assert loads == []
    assert not (tmp_path / "evaluation/OPEN_EVENT.json").exists()


def test_direct_evaluator_loader_requires_the_current_population_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "evaluator.pt"
    checkpoint.write_bytes(b"test evaluator checkpoint")
    checkpoint_sha256 = _sha256(checkpoint)
    runner._write_json(  # noqa: SLF001
        tmp_path / "source_bindings.json",
        {"evaluator_files": {"checkpoint": str(checkpoint)}},
    )
    monkeypatch.setattr(
        runner, "ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256", checkpoint_sha256
    )
    loads: list[Path] = []

    def fake_load(path: Path, **_kwargs: Any) -> dict[str, Any]:
        loads.append(Path(path))
        return {"state_dict": {}}

    class FakeEvaluator:
        def load_state_dict(self, state: Mapping[str, Any]) -> None:
            assert state == {}

        def eval(self) -> "FakeEvaluator":
            return self

    monkeypatch.setattr(torch, "load", fake_load)
    monkeypatch.setattr(runner, "SmallMnistCNN", FakeEvaluator)
    with pytest.raises(runner.IntegrityFailure, match="population seal"):
        runner._load_accepted_evaluator(  # noqa: SLF001
            tmp_path, runner.EvaluatorFirewall(), "a" * 64
        )
    assert loads == []

    forged_firewall = runner.EvaluatorFirewall()
    forged_firewall.mark_populations_sealed("a" * 64)
    with pytest.raises(runner.IntegrityFailure, match="population seal"):
        runner._load_accepted_evaluator(  # noqa: SLF001
            tmp_path, forged_firewall, "a" * 64
        )
    assert loads == []

    runner._write_json(  # noqa: SLF001
        tmp_path / "POPULATIONS_SEALED.json",
        {
            "schema": "test-population-seal",
            "artifact_count": 0,
            "tree_digest": runner._tree_digest([]),  # noqa: SLF001
            "artifacts": [],
        },
    )
    seal_sha256 = _sha256(tmp_path / "POPULATIONS_SEALED.json")
    firewall = runner.EvaluatorFirewall()
    firewall.mark_populations_sealed(seal_sha256)
    evaluator = runner._load_accepted_evaluator(  # noqa: SLF001
        tmp_path, firewall, seal_sha256
    )
    assert isinstance(evaluator, FakeEvaluator)
    assert firewall.state == runner.EvaluatorFirewall.EVALUATOR_OPENED
    assert loads == [checkpoint]


def test_terminal_test_loader_is_never_called() -> None:
    data = _config()["data"]
    assert data["terminal_test_content_rows_parsed"] == 0
    assert data["whole_file_sha256_read"] == 1
    source = inspect.getsource(runner)
    assert "_load_train_validation_mnist_strict" in source
    for forbidden in (
        "load_test_mnist",
        "load_terminal_test",
        "terminal_test_loader",
        "test_images",
        "test_labels",
    ):
        assert forbidden not in source


def test_review_key_is_created_only_after_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative in ("evaluation", "review"):
        (tmp_path / relative).mkdir(parents=True)
    runner._write_json(  # noqa: SLF001
        tmp_path / "stage_ledger.json", {"schema": "test", "events": []}
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "source_bindings.json",
        {"evaluator_hashes": {"checkpoint": "a" * 64, "selection": "b" * 64}},
    )
    monkeypatch.setattr(runner, "_admit_major_stage", lambda *_args: {"passed": 1})
    firewall = runner.EvaluatorFirewall()
    with pytest.raises(runner.IntegrityFailure, match="seal is missing"):
        runner.evaluate_sealed_populations(
            tmp_path, {}, firewall, None  # type: ignore[arg-type]
        )
    assert not (tmp_path / "review/review_key.json").exists()

    empty_rows: list[dict[str, Any]] = []
    runner._write_json(  # noqa: SLF001
        tmp_path / "POPULATIONS_SEALED.json",
        {
            "schema": "test-population-seal",
            "artifact_count": 0,
            "tree_digest": runner._tree_digest(empty_rows),  # noqa: SLF001
            "artifacts": empty_rows,
        },
    )
    monkeypatch.setattr(runner, "_load_accepted_evaluator", lambda *_args: object())

    def fake_evaluation(
        _evaluator: object,
        images: np.ndarray,
        labels: np.ndarray,
        sample_ids: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        count = len(images)
        return (
            {
                "logits": np.zeros((count, 10)),
                "probabilities": np.full((count, 10), 0.1),
                "predictions": np.zeros(count, dtype=np.int64),
                "requested_labels": labels,
                "sample_ids": sample_ids,
                "requested_log_probabilities": np.zeros(count),
            },
            {"loss": 0.0, "requested_label_accuracy": 0.0, "per_class": {}},
        )

    def fake_contact_sheet(path: Path, *_args: Any, **_kwargs: Any) -> None:
        Path(path).write_bytes(b"test contact sheet")

    class Governor:
        def admit(self, _kind: str, **_kwargs: Any) -> None:
            return None

        def complete(self, _kind: str, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(runner, "_evaluation_record", fake_evaluation)
    monkeypatch.setattr(runner, "write_contact_sheet", fake_contact_sheet)
    labels = np.repeat(np.arange(10), 2)
    ids = np.asarray([f"prior-{index:03d}" for index in range(20)])
    populations = {
        "rendered": {
            name: np.zeros((20, 28, 28), dtype=np.uint8)
            for name in (
                "null_prior",
                "learned_prior",
                "null_forward_terminal",
                "learned_forward_terminal",
            )
        },
        "identities": {
            "prior_requested_labels": labels,
            "prior_path_ids": np.arange(0xB6500, 0xB6514),
            "prior_sample_ids": ids,
            "forward_requested_labels": labels,
            "forward_sample_ids": np.asarray(
                [f"forward-{index:03d}" for index in range(20)]
            ),
        },
        "forward_marker": {"passed": 0},
    }
    runner.evaluate_sealed_populations(
        tmp_path,
        populations,
        firewall,
        Governor(),  # type: ignore[arg-type]
    )
    assert (tmp_path / "POPULATIONS_SEALED.json").is_file()
    key = json.loads((tmp_path / "review/review_key.json").read_text(encoding="utf-8"))
    assert key["population_seal_sha256"] == _sha256(
        tmp_path / "POPULATIONS_SEALED.json"
    )
    assert (tmp_path / "evaluation/OPEN_EVENT.json").is_file()


def _write_evaluation_verifier_fixture(run_dir: Path) -> None:
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (run_dir / "populations").mkdir(parents=True, exist_ok=True)
    labels = np.tile(np.arange(10, dtype=np.int64), 2)
    prior_sample_ids = np.asarray([f"prior-{index:03d}" for index in range(20)])
    forward_sample_ids = np.asarray([f"forward-{index:03d}" for index in range(20)])
    runner._write_npz(  # noqa: SLF001
        run_dir / "populations/identities.npz",
        prior_requested_labels=labels,
        prior_path_ids=np.arange(0xB6500, 0xB6514, dtype=np.int64),
        prior_sample_ids=prior_sample_ids,
        forward_requested_labels=labels,
        forward_path_ids=np.arange(0xB6520, 0xB6534, dtype=np.int64),
        forward_sample_ids=forward_sample_ids,
        oracle_requested_labels=np.arange(10, dtype=np.int64),
        oracle_path_ids=np.arange(0xB6540, 0xB654A, dtype=np.int64),
        oracle_sample_ids=np.asarray([f"oracle-{index:03d}" for index in range(10)]),
    )
    identity_path = run_dir / "populations/identities.npz"
    seal_rows = [
        {
            "path": "populations/identities.npz",
            "size": identity_path.stat().st_size,
            "sha256": _sha256(identity_path),
        }
    ]
    runner._write_json(  # noqa: SLF001
        run_dir / "POPULATIONS_SEALED.json",
        {
            "schema": runner.VERSION + "-population-seal",
            "sealed_before_evaluator_or_review": 1,
            "artifact_count": len(seal_rows),
            "tree_digest": runner._tree_digest(seal_rows),  # noqa: SLF001
            "artifacts": seal_rows,
        },
    )
    seal_sha256 = _sha256(run_dir / "POPULATIONS_SEALED.json")
    evaluator_hashes = {
        "checkpoint": "b" * 64,
        "selection": "c" * 64,
    }
    runner._write_json(  # noqa: SLF001
        run_dir / "source_bindings.json",
        {"evaluator_hashes": evaluator_hashes},
    )
    runner._write_json(  # noqa: SLF001
        run_dir / "evaluation/OPEN_EVENT.json",
        {
            "population_seal_sha256": seal_sha256,
            "opened_after_population_seal": 1,
            "terminal_test_rows_opened": 0,
        },
    )
    runner._write_json(  # noqa: SLF001
        run_dir / "evaluation/evaluator_binding.json",
        {
            "checkpoint_sha256": evaluator_hashes["checkpoint"],
            "selection_sha256": evaluator_hashes["selection"],
            "population_seal_sha256": seal_sha256,
            "device": "cpu",
        },
    )
    output_arrays: dict[str, np.ndarray] = {}
    population_metrics: dict[str, Any] = {}
    for role in (
        "null_prior",
        "learned_prior",
        "null_forward_terminal",
        "learned_forward_terminal",
    ):
        logits = np.zeros((20, 10), dtype=np.float64)
        if role.startswith("learned"):
            logits[np.arange(20), labels] = 1.0
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        requested_log_probabilities = log_probabilities[np.arange(20), labels]
        predictions = np.argmax(probabilities, axis=1)
        sample_ids = (
            prior_sample_ids if role.endswith("prior") else forward_sample_ids
        )
        output_arrays.update(
            {
                f"{role}_logits": logits,
                f"{role}_probabilities": probabilities,
                f"{role}_predictions": predictions,
                f"{role}_requested_labels": labels,
                f"{role}_sample_ids": sample_ids,
                f"{role}_requested_log_probabilities": requested_log_probabilities,
            }
        )
        population_metrics[role] = {
            "loss": float(-np.mean(requested_log_probabilities)),
            "requested_label_accuracy": float(np.mean(predictions == labels)),
            "per_class": {
                str(digit): {
                    "count": 2,
                    "accuracy": float(np.mean(predictions[labels == digit] == digit)),
                }
                for digit in range(10)
            },
        }
    runner._write_npz(  # noqa: SLF001
        run_dir / "evaluation/outputs.npz", **output_arrays
    )
    prior_delta = (
        output_arrays["learned_prior_requested_log_probabilities"]
        - output_arrays["null_prior_requested_log_probabilities"]
    )
    null_accuracy = population_metrics["null_prior"]["requested_label_accuracy"]
    learned_accuracy = population_metrics["learned_prior"]["requested_label_accuracy"]
    evaluator_marker = {
        "gate_type": "diagnostic threshold",
        "null_prior_requested_label_accuracy": null_accuracy,
        "learned_prior_requested_label_accuracy": learned_accuracy,
        "paired_requested_log_probability_win_count": int(np.sum(prior_delta > 0.0)),
        "mean_paired_requested_log_probability_improvement": float(np.mean(prior_delta)),
    }
    evaluator_marker["passed"] = int(
        learned_accuracy >= 0.20
        and learned_accuracy > null_accuracy
        and evaluator_marker["paired_requested_log_probability_win_count"] >= 12
        and evaluator_marker["mean_paired_requested_log_probability_improvement"] > 0.0
    )
    forward_marker = {"gate_type": "diagnostic threshold", "passed": 1}
    runner._write_json(  # noqa: SLF001
        run_dir / "populations/forward_direct_metrics.json", forward_marker
    )
    runner._write_json(  # noqa: SLF001
        run_dir / "evaluation/metrics.json",
        {
            "schema": runner.VERSION + "-evaluation-metrics",
            "terminal_test_rows_used": 0,
            "populations": population_metrics,
            "prior_paired_requested_log_probability_effect": prior_delta,
            "evaluator_marker": evaluator_marker,
            "forward_direct_marker": forward_marker,
        },
    )


def test_evaluator_paired_effect_and_marker_are_replayed(
    tmp_path: Path,
) -> None:
    _write_evaluation_verifier_fixture(tmp_path)
    saved = runner._verify_evaluation(tmp_path)  # noqa: SLF001
    assert saved["evaluator_marker"]["passed"] == 1
    saved["prior_paired_requested_log_probability_effect"] = (
        np.asarray(saved["prior_paired_requested_log_probability_effect"]) + 1.0
    )
    saved["evaluator_marker"]["paired_requested_log_probability_win_count"] = 20
    saved["evaluator_marker"]["mean_paired_requested_log_probability_improvement"] += 1.0
    runner._write_json(tmp_path / "evaluation/metrics.json", saved)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="paired effect|marker"):
        runner._verify_evaluation(tmp_path)  # noqa: SLF001


def test_evaluator_outputs_bind_to_sealed_population_identities(
    tmp_path: Path,
) -> None:
    _write_evaluation_verifier_fixture(tmp_path)
    runner._verify_evaluation(tmp_path)  # noqa: SLF001
    output_path = tmp_path / "evaluation/outputs.npz"
    outputs = runner._npz_arrays(output_path)  # noqa: SLF001
    role = "learned_prior"
    permutation = np.roll(np.arange(20), 1)
    outputs[f"{role}_requested_labels"] = outputs[
        f"{role}_requested_labels"
    ][permutation]
    outputs[f"{role}_sample_ids"] = outputs[f"{role}_sample_ids"][permutation]
    logits = np.asarray(outputs[f"{role}_logits"], dtype=np.float64)
    labels = np.asarray(outputs[f"{role}_requested_labels"], dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    outputs[f"{role}_requested_log_probabilities"] = log_probabilities[
        np.arange(20), labels
    ]
    runner._write_npz(output_path, **outputs)  # noqa: SLF001

    metrics_path = tmp_path / "evaluation/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = np.asarray(outputs[f"{role}_predictions"], dtype=np.int64)
    requested = outputs[f"{role}_requested_log_probabilities"]
    metrics["populations"][role] = {
        "loss": float(-np.mean(requested)),
        "requested_label_accuracy": float(np.mean(predictions == labels)),
        "per_class": {
            str(digit): {
                "count": int(np.sum(labels == digit)),
                "accuracy": float(np.mean(predictions[labels == digit] == digit)),
            }
            for digit in range(10)
        },
    }
    delta = (
        requested
        - np.asarray(
            outputs["null_prior_requested_log_probabilities"], dtype=np.float64
        )
    )
    metrics["prior_paired_requested_log_probability_effect"] = delta
    null_accuracy = float(
        metrics["populations"]["null_prior"]["requested_label_accuracy"]
    )
    learned_accuracy = float(
        metrics["populations"]["learned_prior"]["requested_label_accuracy"]
    )
    marker = {
        "gate_type": "diagnostic threshold",
        "null_prior_requested_label_accuracy": null_accuracy,
        "learned_prior_requested_label_accuracy": learned_accuracy,
        "paired_requested_log_probability_win_count": int(np.sum(delta > 0.0)),
        "mean_paired_requested_log_probability_improvement": float(np.mean(delta)),
    }
    marker["passed"] = int(
        learned_accuracy >= 0.20
        and learned_accuracy > null_accuracy
        and marker["paired_requested_log_probability_win_count"] >= 12
        and marker["mean_paired_requested_log_probability_improvement"] > 0.0
    )
    metrics["evaluator_marker"] = marker
    runner._write_json(metrics_path, metrics)  # noqa: SLF001
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="identity|sample|label"):
        runner._verify_evaluation(tmp_path)  # noqa: SLF001


def test_human_metrics_and_outcome_route_recompute_from_answers(tmp_path: Path) -> None:
    entries: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    for index in range(40):
        controller = "learned" if index < 20 else "null"
        requested = index % 10
        if controller == "learned" and index < 15:
            assignment = str(requested)
        elif controller == "null" and index < 25:
            assignment = str(requested)
        else:
            assignment = "noise"
        sample_id = f"blind-{index:03d}"
        entries.append(
            {
                "review_order": index,
                "sample_id": sample_id,
                "source_sample_id": f"{controller}:prior-{index % 20:03d}",
                "controller": controller,
                "requested_label": requested,
                "path_id": 0xB6500 + index % 20,
            }
        )
        answer_rows.append(
            {
                "review_order": index,
                "sample_id": sample_id,
                "assigned_label": assignment,
                "notes": "",
            }
        )
    answers = tmp_path / "answers.csv"
    with answers.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("review_order", "sample_id", "assigned_label", "notes"),
        )
        writer.writeheader()
        writer.writerows(answer_rows)
    metrics = runner.compute_human_review_metrics(answers, {"entries": entries})
    assert metrics["learned"]["recognizable_count"] == 15
    assert metrics["learned"]["requested_label_count"] == 15
    assert metrics["null"]["recognizable_count"] == 5
    assert metrics["null"]["requested_label_count"] == 5
    assert metrics["learned_counts_exceed_null"] == 1
    assert len(metrics["paired_paths"]) == 20
    assert metrics["aggregate_paired_recognizable_difference"] == 10
    assert metrics["aggregate_paired_requested_label_difference"] == 10
    assert metrics["paired_recognizable_win_loss_tie"] == {
        "wins": 10,
        "losses": 0,
        "ties": 10,
    }
    assert metrics["paired_requested_label_win_loss_tie"] == {
        "wins": 10,
        "losses": 0,
        "ties": 10,
    }
    assert all(
        set(row)
        == {
            "path_id",
            "requested_label",
            "learned_sample_id",
            "null_sample_id",
            "learned_recognizable",
            "null_recognizable",
            "learned_requested_label",
            "null_requested_label",
            "recognizable_difference",
            "requested_label_difference",
        }
        for row in metrics["paired_paths"]
    )
    learned_images = np.zeros((20, 28, 28), dtype=np.uint8)
    for index in range(20):
        learned_images[index].flat[index] = 255
    (tmp_path / "populations").mkdir()
    runner._write_npz(  # noqa: SLF001
        tmp_path / "populations/uint8_populations.npz",
        learned_prior=learned_images,
        prior_requested_labels=np.asarray(
            [row["requested_label"] for row in entries[:20]], dtype=np.int64
        ),
    )
    human_marker = runner._human_marker_from_metrics(tmp_path, metrics)  # noqa: SLF001
    assert human_marker["passed"] == 1
    forged_marker = dict(human_marker)
    forged_marker["learned_requested_label_count"] -= 1
    assert runner._semantic_sha256(forged_marker) != runner._semantic_sha256(  # noqa: SLF001
        runner._human_marker_from_metrics(tmp_path, metrics)  # noqa: SLF001
    )
    assert "_human_marker_from_metrics" in inspect.getsource(runner.verify_run)


@pytest.mark.parametrize(
    ("forward", "human", "evaluator", "expected"),
    (
        (0, 0, 0, "v0_negative_pivot_experiment10"),
        (0, 0, 1, "human_negative_evaluator_positive_audit"),
        (0, 1, 0, "suspicious_prior_forward_disagreement"),
        (0, 1, 1, "suspicious_prior_forward_disagreement"),
        (1, 0, 0, "prior_terminal_mismatch_or_on_policy_shift"),
        (1, 0, 1, "evaluator_render_mismatch_task_negative"),
        (1, 1, 0, "human_direct_positive_evaluator_disagreement"),
        (1, 1, 1, "approximate_candidate_feasibility_reference_audit_next"),
    ),
)
def test_outcome_route_truth_table_matches_prespecified_actions(
    forward: int,
    human: int,
    evaluator: int,
    expected: str,
) -> None:
    assert (
        runner.route_outcome(
            {"passed": forward}, {"passed": human}, {"passed": evaluator}
        )
        == expected
    )


def test_resource_check_prices_next_outer_step_before_entry() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class Governor:
        def complete(self, kind: str, **kwargs: Any) -> None:
            calls.append(("complete", kind, kwargs))

        def admit(self, kind: str, **kwargs: Any) -> None:
            calls.append(("admit", kind, kwargs))

    callback = runner._governed_outer_callback(  # noqa: SLF001
        Governor(), "forward-cache-outer", paths=8, predicted_seconds=11.0
    )
    callback({"direction": "forward", "outer_step": 0})
    assert [row[:2] for row in calls] == [
        ("complete", "forward-cache-outer"),
        ("admit", "forward-cache-outer"),
    ]
    assert calls[0][2]["transitions"] == 8 * 7 * core.EDGES_PER_PHASE
    assert calls[1][2]["predicted_seconds"] == 11.0
    calls.clear()
    callback({"direction": "forward", "outer_step": 127})
    assert [row[0] for row in calls] == ["complete"]


def test_resource_check_includes_current_elapsed_time_and_900_second_reserve() -> None:
    budget = runner.ResourceBudget(
        max_active_seconds=1_013.0,
        max_storage_bytes=1_000_000,
        max_cuda_fraction=0.75,
    )
    decision = runner.resource_admission(
        budget,
        active_seconds=100.0,
        predicted_next_quantum_seconds=10.0,
    )
    assert decision["passed"] == 1
    assert decision["reserve_remaining_seconds"] == 900.0
    assert decision["quantum_inequality_lhs"] == 100.0 + 1.25 * 10.0 + 900.0
    boundary = runner.resource_admission(
        budget,
        active_seconds=100.5,
        predicted_next_quantum_seconds=10.0,
    )
    assert boundary["passed"] == 0
    assert boundary["reason"] == "next_quantum_active_projection"
    post_seal = runner.resource_admission(
        budget,
        active_seconds=100.5,
        predicted_next_quantum_seconds=10.0,
        populations_sealed=True,
    )
    assert post_seal["passed"] == 1
    assert post_seal["reserve_remaining_seconds"] == 0.0
    storage_boundary = runner.resource_admission(
        runner.ResourceBudget(
            max_active_seconds=10_800.0,
            max_storage_bytes=1_000,
            max_cuda_fraction=0.75,
        ),
        active_seconds=0.0,
        storage_bytes=800,
        predicted_next_bytes=200,
        populations_sealed=True,
    )
    assert storage_boundary["passed"] == 0
    assert storage_boundary["reason"] == "storage_projection"
    assert storage_boundary["storage_after_next_bytes"] == 1_000


def test_slow_quantum_that_would_repeat_23_minute_overshoot_is_rejected() -> None:
    decision = runner.resource_admission(
        runner.ResourceBudget(
            max_active_seconds=10_800.0,
            max_storage_bytes=2 * 1024**3,
            max_cuda_fraction=0.75,
        ),
        active_seconds=8_000.0,
        predicted_next_quantum_seconds=23.0 * 60.0,
    )
    assert decision["passed"] == 0
    assert "next_quantum_duration" in decision["reasons"]
    assert decision["quantum_inequality_lhs"] == 8_000.0 + 1.25 * 1_380.0 + 900.0


def test_major_stage_projection_receipts_cover_every_post_smoke_boundary(
    tmp_path: Path,
) -> None:
    (tmp_path / "resource_smoke").mkdir()
    runner._write_json(  # noqa: SLF001
        tmp_path / "resource_smoke/timings.json",
        {
            "candidate_transition_seconds": 1.0,
            "candidate_transition_count": 1_000_000_000,
            "training_seconds": 1.0,
            "training_updates": 1_000,
            "persistence_seconds": 0.01,
            "storage_bytes_per_record": 100.0,
        },
    )
    budget = runner.ResourceBudget(
        max_active_seconds=1_000_000.0,
        max_storage_bytes=1_000_000_000,
        max_cuda_fraction=0.75,
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "resource_ledger.json",
        {
            "schema": "test-resource-ledger",
            "budget": {
                "max_active_seconds": budget.max_active_seconds,
                "max_storage_bytes": budget.max_storage_bytes,
                "max_cuda_fraction": budget.max_cuda_fraction,
                "reserve_seconds": budget.reserve_seconds,
                "maximum_quantum_seconds": budget.maximum_quantum_seconds,
                "projection_multiplier": budget.projection_multiplier,
            },
            "active_seconds": 0.0,
            "peak_storage_bytes": 0,
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_reserved_bytes": 0,
            "peak_cuda_fraction": 0.0,
            "events": [],
            "admissions": [],
            "last_admission": None,
        },
    )
    governor = runner.ResourceGovernor(tmp_path, torch.device("cpu"), budget)
    kinds = (
        "oracle_stage",
        "forward_record_caches_stage",
        "training_stage",
        "objective_sampling_stage",
        "population_seal_stage",
        "sealed_evaluation_stage",
    )
    for kind in kinds:
        receipt = runner._admit_major_stage(tmp_path, governor, kind)  # noqa: SLF001
        assert receipt["passed"] == 1
        assert receipt["major_stage"] == 1
        assert receipt["projected_remaining_seconds"] > 0.0
        assert receipt["predicted_next_bytes"] > 0
    ledger = json.loads((tmp_path / "resource_ledger.json").read_text(encoding="utf-8"))
    major = [row for row in ledger["admissions"] if row["major_stage"] == 1]
    assert [row["kind"] for row in major] == list(kinds)
    remaining = [float(row["projected_remaining_seconds"]) for row in major]
    assert all(left > right for left, right in zip(remaining, remaining[1:]))
    for function, kind in (
        (runner.run_oracle_control, "oracle_stage"),
        (runner.build_forward_record_caches, "forward_record_caches_stage"),
        (runner.train_candidate_model, "training_stage"),
        (runner.sample_objective_populations, "objective_sampling_stage"),
        (runner.seal_populations, "population_seal_stage"),
        (runner.evaluate_sealed_populations, "sealed_evaluation_stage"),
    ):
        source = inspect.getsource(function)
        assert f'_admit_major_stage(run_dir, governor, "{kind}")' in source


def test_resource_stop_retains_completed_cohorts_and_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_directory = tmp_path / "populations/stages/null_prior"
    image_directory = tmp_path / "populations/images/prior/null"
    stage_directory.mkdir(parents=True)
    path_ids = np.arange(0xB6500, 0xB6514, dtype=np.int64)
    labels = np.repeat(np.arange(10), 2)
    sample_ids = np.asarray([f"prior-{value:05x}" for value in path_ids])
    starts = core.sample_dirichlet_starts(
        path_ids, root_seed=runner.SEEDS["prior_start"]
    )
    runner._write_npz(  # noqa: SLF001
        tmp_path / "populations/prior_start_authority.npz",
        prior_starts=starts,
        requested_labels=labels,
        path_ids=path_ids,
        sample_ids=sample_ids,
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "populations/prior_start_authority.json",
        {
            "committed_before_sampling": 1,
            "npz_sha256": _sha256(
                tmp_path / "populations/prior_start_authority.npz"
            ),
            "path_ids_sha256": runner._array_sha256(path_ids),  # noqa: SLF001
        },
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_backend.json", _candidate_backend_fixture_record()
    )
    (tmp_path / "resource_smoke").mkdir()
    runner._write_json(  # noqa: SLF001
        tmp_path / "resource_smoke/timings.json",
        {
            "persistence_seconds": 0.01,
            "outer_step_maxima": {
                "forward": 1.0,
                "null_reverse": 1.0,
                "zero_model_reverse": 1.0,
            },
        },
    )
    admission_rows: list[dict[str, Any]] = []

    class Governor:
        def admit(self, _kind: str, **_kwargs: Any) -> None:
            admission_rows.append({"kind": _kind, **_kwargs})
            if len(admission_rows) == 1:
                assert int(_kwargs["predicted_bytes"]) > 0
            if len(admission_rows) == 2:
                assert (stage_directory / "cohort_000.npz").is_file()
                assert (stage_directory / "cohort_000.telemetry.json").is_file()
                assert len(list(image_directory.glob("*.png"))) == 8
                raise runner.ResourceStop("next cohort does not fit")

        def complete(self, _kind: str, **_kwargs: Any) -> None:
            return None

    def fake_reverse(
        starts: np.ndarray,
        *_args: Any,
        **_kwargs: Any,
    ) -> core.SamplingResult:
        values = np.asarray(starts, dtype=np.float64)
        return _sampling_result(values, controller="null")

    monkeypatch.setattr(candidate, "reverse_sample_candidate", fake_reverse)
    with pytest.raises(runner.ResourceStop, match="next cohort"):
        runner._run_reverse_cohorts(  # noqa: SLF001
            tmp_path,
            stage_directory,
            starts,
            labels,
            path_ids,
            sample_ids,
            controller="null",
            root_seed=runner.SEEDS["prior_reverse"],
            runtime=SimpleNamespace(),
            governor=Governor(),  # type: ignore[arg-type]
            image_directory=image_directory,
        )
    saved = stage_directory / "cohort_000.npz"
    assert saved.is_file()
    assert (stage_directory / "cohort_000.telemetry.json").is_file()
    assert not (stage_directory / "cohort_001.npz").exists()
    with np.load(saved, allow_pickle=False) as cohort:
        np.testing.assert_array_equal(cohort["path_ids"], path_ids[:8])
        assert set(cohort.files) >= {
            "starts",
            "anchors_000",
            "anchors_032",
            "anchors_064",
            "anchors_096",
            "anchors_128",
            "final_states",
            "requested_labels",
            "path_ids",
            "sample_ids",
        }
    assert len(list(image_directory.glob("*.png"))) == 8
    assert len(admission_rows) == 2
    assert all(int(row["predicted_bytes"]) > 0 for row in admission_rows)
    runner._verify_partial_objective_populations(tmp_path)  # noqa: SLF001

    arrays = runner._npz_arrays(saved)  # noqa: SLF001
    arrays["starts"][0] = np.roll(arrays["starts"][0], 1)
    for anchor in (0, 32, 64, 96, 128):
        arrays[f"anchors_{anchor:03d}"][0] = arrays["starts"][0]
    arrays["final_states"][0] = arrays["starts"][0]
    runner._write_npz(saved, **arrays)  # noqa: SLF001
    _, rendered = runner.rasterize_population(arrays["final_states"])
    runner._save_individual_pngs(  # noqa: SLF001
        image_directory, rendered, arrays["sample_ids"]
    )
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="partial null_prior authority"):
        runner._verify_partial_objective_populations(tmp_path)  # noqa: SLF001


def _minimal_verifier_tree(run_dir: Path, route: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    defaults = runner.FROZEN_CONFIG["resource_defaults"]
    approval = "user-approved-candidate-verifier-fixture-20260817"
    arff = run_dir.parent / "mnist-authority.arff"
    ddpm = run_dir.parent / "ddpm-authority"
    maximum_active_seconds = float(defaults["maximum_active_seconds"])
    maximum_storage_mib = float(defaults["maximum_storage_mib"])
    maximum_cuda_fraction = float(defaults["maximum_cuda_fraction"])
    canonical_argv = runner._canonical_run_argv(  # noqa: SLF001
        run_dir,
        arff,
        ddpm,
        device="cuda:0",
        approval_id=approval,
        maximum_active_seconds=maximum_active_seconds,
        maximum_storage_mib=maximum_storage_mib,
        maximum_cuda_fraction=maximum_cuda_fraction,
    )
    command_text = runner._canonical_command_text(canonical_argv)  # noqa: SLF001
    config = json.loads(json.dumps(runner.FROZEN_CONFIG))
    config["execution_authority"] = {
        "approval_id": approval,
        "device": "cuda:0",
        "maximum_active_seconds": maximum_active_seconds,
        "maximum_storage_mib": maximum_storage_mib,
        "maximum_cuda_fraction": maximum_cuda_fraction,
        "terminal_reserve_seconds": 900.0,
        "exact_cli_subcommand": "run",
        "canonical_argv": canonical_argv,
        "command_sha256": hashlib.sha256(command_text.encode("utf-8")).hexdigest(),
        "whole_run_restart_only": 1,
        "automatic_full_scale_launches": 0,
    }
    runner._write_json(run_dir / "config.json", config)  # noqa: SLF001
    runner._write_text(run_dir / "command.txt", command_text)  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        run_dir / "source_bindings.json",
        {
            "repository_root": str(REPOSITORY_ROOT),
            "arff": str(arff.resolve()),
            "ddpm_run_dir": str(ddpm.resolve()),
            "config_sha256": runner._semantic_sha256(config),  # noqa: SLF001
            "git": {"revision": "fixture-revision"},
            "historical_source_inventory": EXPECTED_PROTECTED_SOURCE_HASHES,
        },
    )
    runner._write_json(run_dir / "environment.json", {"device": "cuda:0"})  # noqa: SLF001
    runner._write_json(  # noqa: SLF001
        run_dir / "path_id_audit.json", _path_audit_fixture()
    )

    if route == "review_pending":
        completed = runner._STAGE_ORDER[:-1]  # noqa: SLF001
    elif route == "complete":
        completed = runner._STAGE_ORDER  # noqa: SLF001
    else:
        completed = runner._STAGE_ORDER[:1]  # noqa: SLF001
    stage_events = [
        {"stage": stage, "state": "complete", "at": "fixture"}
        for stage in completed
    ]
    failure_routes = {
        "integrity_failed",
        "candidate_health_failed",
        "resource_projection_failed",
        "oracle_control_failed",
        "resource_stopped",
    }
    if route in failure_routes:
        stage_events.append(
            {
                "stage": "terminal_failure",
                "state": "complete",
                "route": route,
                "at": "fixture",
            }
        )
        runner._write_json(  # noqa: SLF001
            run_dir / "failure.json", {"route": route}
        )
    runner._write_json(  # noqa: SLF001
        run_dir / "stage_ledger.json",
        {"schema": runner.VERSION + "-stage-ledger", "events": stage_events},
    )

    budget = runner.ResourceBudget(
        max_active_seconds=maximum_active_seconds,
        max_storage_bytes=int(maximum_storage_mib * 1024 * 1024),
        max_cuda_fraction=maximum_cuda_fraction,
    )
    events: list[dict[str, Any]] = []
    admissions: list[dict[str, Any]] = []
    active_seconds = 0.0
    if route in {"resource_projection_failed", "resource_stopped"}:
        active_seconds = 10_000.0
        events.append(
            {
                "kind": "fixture_work",
                "seconds": active_seconds,
                "candidate_transitions": 0,
                "model_evaluations": 0,
                "tree_bytes": 0,
                "completed": 1,
                "at": "fixture",
            }
        )
        failed = runner.resource_admission(
            budget,
            active_seconds=active_seconds,
            predicted_next_quantum_seconds=30.0,
        )
        admissions.append(
            {
                "kind": "data_roles_write",
                **failed,
                "declared_predicted_next_quantum_seconds": 30.0,
                "recent_completed_same_kind_seconds": [],
                "event_count_before_admission": 1,
                "kind_admission_ordinal": 1,
                "post_completion_check": 0,
                "at": "fixture",
            }
        )
    if route in failure_routes:
        terminal_decision = runner.resource_admission(
            budget,
            active_seconds=active_seconds,
            predicted_next_quantum_seconds=runner.FAILURE_TERMINALIZATION_SECONDS,
            predicted_next_bytes=runner.FAILURE_TERMINALIZATION_BYTES,
            terminalization=True,
        )
        assert terminal_decision["passed"] == 1
        admissions.append(
            {
                "kind": "failure_terminalization",
                **terminal_decision,
                "declared_predicted_next_quantum_seconds": (
                    runner.FAILURE_TERMINALIZATION_SECONDS
                ),
                "recent_completed_same_kind_seconds": [],
                "event_count_before_admission": len(events),
                "kind_admission_ordinal": 1,
                "post_completion_check": 0,
                "at": "fixture",
            }
        )
        events.append(
            {
                "kind": "failure_terminalization",
                "seconds": 0.0,
                "candidate_transitions": 0,
                "model_evaluations": 0,
                "tree_bytes": 0,
                "completed": 1,
                "at": "fixture",
            }
        )
    ledger = {
        "schema": runner.VERSION + "-resource-ledger",
        "budget": vars(budget),
        "active_seconds": active_seconds,
        "peak_storage_bytes": (1 << 20) if route in failure_routes else 0,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
        "peak_cuda_fraction": 0.0,
        "events": events,
        "admissions": admissions,
        "last_admission": admissions[-1] if admissions else None,
    }
    runner._write_json(run_dir / "resource_ledger.json", ledger)  # noqa: SLF001
    error = "fixture terminal failure"
    if route in failure_routes:
        failure_propositions = {
            "candidate_health_failed": (
                "the fixed 512-lane candidate numerical execution/integrity criterion did not pass"
            ),
            "oracle_control_failed": (
                "the fixed ten-path oracle/null positive-control execution/integrity criterion did not pass"
            ),
            "resource_projection_failed": (
                "a required major-stage resource projection did not fit"
            ),
            "resource_stopped": (
                "a priced resource quantum did not fit or exceeded its cap after completion"
            ),
            "integrity_failed": "an execution or artifact-integrity requirement failed",
        }
        failure = {
            "schema": runner.VERSION + "-failure",
            "route": route,
            "error_type": "FixtureFailure",
            "error": error,
            "traceback": "fixture traceback",
            "scientific_objective_result_available": 0,
            "exact_failure_proposition": failure_propositions[route],
            "claim_boundary": (
                "no learned-controller or generator scientific conclusion is available from this route"
            ),
            "full_scale_auto_launched": 0,
            "failure_terminalization_admitted": 1,
            "failure_terminalization_reserve_seconds": 0.0,
            "at": "fixture",
        }
        runner._write_json(run_dir / "failure.json", failure)  # noqa: SLF001
        if route in {"resource_projection_failed", "resource_stopped"}:
            failed_admission = next(
                row for row in admissions if int(row["passed"]) == 0
            )
            runner._write_json(  # noqa: SLF001
                run_dir / "resource_stop.json",
                {
                    "schema": runner.VERSION + "-resource-stop",
                    "route": route,
                    "error": error,
                    "failed_admission": failed_admission,
                    "active_seconds": active_seconds,
                    "tree_bytes": runner._directory_bytes(run_dir),  # noqa: SLF001
                    "completed_artifacts_retained": 1,
                },
            )
        runner._status(run_dir, route, error=error)  # noqa: SLF001
        runner._write_text(  # noqa: SLF001
            run_dir / "REPORT.md", runner._failure_report(run_dir, failure)  # noqa: SLF001
        )
        runner._write_text(  # noqa: SLF001
            run_dir / "experiment_note.md",
            runner._failure_experiment_note(run_dir, failure),  # noqa: SLF001
        )
    else:
        runner._status(run_dir, route)  # noqa: SLF001
    if route in failure_routes:
        runner._seal_manifest(run_dir)  # noqa: SLF001


def test_execution_authority_caps_command_and_cuda_peak_are_replayed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _minimal_verifier_tree(run_dir, "integrity_failed")
    assert runner._verify_config_and_resources(  # noqa: SLF001
        run_dir, "integrity_failed"
    )["active_seconds"] == 0.0

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config["execution_authority"]["canonical_argv"].append("--forged")
    runner._write_json(run_dir / "config.json", config)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="canonical run command"):
        runner._verify_config_and_resources(run_dir, "integrity_failed")  # noqa: SLF001

    _minimal_verifier_tree(run_dir, "integrity_failed")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    authority = config["execution_authority"]
    authority["maximum_active_seconds"] = (
        float(runner.FROZEN_CONFIG["resource_defaults"]["maximum_active_seconds"])
        + 1.0
    )
    argv = runner._canonical_run_argv(  # noqa: SLF001
        run_dir,
        Path(json.loads((run_dir / "source_bindings.json").read_text())["arff"]),
        Path(
            json.loads((run_dir / "source_bindings.json").read_text())["ddpm_run_dir"]
        ),
        device=authority["device"],
        approval_id=authority["approval_id"],
        maximum_active_seconds=authority["maximum_active_seconds"],
        maximum_storage_mib=authority["maximum_storage_mib"],
        maximum_cuda_fraction=authority["maximum_cuda_fraction"],
    )
    command = runner._canonical_command_text(argv)  # noqa: SLF001
    authority["canonical_argv"] = argv
    authority["command_sha256"] = hashlib.sha256(command.encode("utf-8")).hexdigest()
    runner._write_json(run_dir / "config.json", config)  # noqa: SLF001
    runner._write_text(run_dir / "command.txt", command)  # noqa: SLF001
    bindings = json.loads(
        (run_dir / "source_bindings.json").read_text(encoding="utf-8")
    )
    bindings["config_sha256"] = runner._semantic_sha256(config)  # noqa: SLF001
    runner._write_json(run_dir / "source_bindings.json", bindings)  # noqa: SLF001
    ledger = json.loads(
        (run_dir / "resource_ledger.json").read_text(encoding="utf-8")
    )
    ledger["budget"]["max_active_seconds"] = authority["maximum_active_seconds"]
    runner._write_json(run_dir / "resource_ledger.json", ledger)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="approval envelope"):
        runner._verify_config_and_resources(run_dir, "integrity_failed")  # noqa: SLF001

    _minimal_verifier_tree(run_dir, "integrity_failed")
    ledger = json.loads(
        (run_dir / "resource_ledger.json").read_text(encoding="utf-8")
    )
    budget = runner.ResourceBudget(**ledger["budget"])
    receipt = {
        "kind": "data_roles_write",
        **runner.resource_admission(
            budget,
            active_seconds=0.0,
            predicted_next_quantum_seconds=30.0,
            cuda_fraction=0.5,
        ),
        "declared_predicted_next_quantum_seconds": 30.0,
        "recent_completed_same_kind_seconds": [],
        "event_count_before_admission": 0,
        "kind_admission_ordinal": 1,
        "post_completion_check": 0,
        "at": "fixture",
    }
    ledger["admissions"] = [receipt]
    ledger["last_admission"] = receipt
    ledger["peak_cuda_fraction"] = 0.4
    runner._write_json(run_dir / "resource_ledger.json", ledger)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="CUDA peak"):
        runner._verify_config_and_resources(run_dir, "integrity_failed")  # noqa: SLF001


def test_config_hash_resource_stop_and_path_audit_reject_coordinated_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_run = tmp_path / "config"
    _minimal_verifier_tree(config_run, "integrity_failed")
    config = json.loads((config_run / "config.json").read_text(encoding="utf-8"))
    config["execution_authority"]["approval_id"] += "-forged"
    runner._write_json(config_run / "config.json", config)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="config hash"):
        runner._verify_source_bindings(config_run)  # noqa: SLF001

    stopped = tmp_path / "stopped"
    _minimal_verifier_tree(stopped, "resource_stopped")
    runner._verify_config_and_resources(stopped, "resource_stopped")  # noqa: SLF001
    ledger = json.loads(
        (stopped / "resource_ledger.json").read_text(encoding="utf-8")
    )
    ledger["last_admission"]["passed"] = 1
    ledger["admissions"][-1] = ledger["last_admission"]
    runner._write_json(stopped / "resource_ledger.json", ledger)  # noqa: SLF001
    stop = json.loads((stopped / "resource_stop.json").read_text(encoding="utf-8"))
    stop["failed_admission"] = ledger["last_admission"]
    runner._write_json(stopped / "resource_stop.json", stop)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="admission replay|failed admission"):
        runner._verify_config_and_resources(stopped, "resource_stopped")  # noqa: SLF001

    path_run = tmp_path / "path"
    _minimal_verifier_tree(path_run, "integrity_failed")
    monkeypatch.setattr(
        runner,
        "path_id_audit",
        lambda _root: _path_audit_fixture(),
    )
    runner._verify_path_id_authority(path_run)  # noqa: SLF001
    saved = json.loads(
        (path_run / "path_id_audit.json").read_text(encoding="utf-8")
    )
    saved["roles"]["oracle"]["start"] -= 1
    runner._write_json(path_run / "path_id_audit.json", saved)  # noqa: SLF001
    runner._seal_manifest(path_run)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="path-ID audit"):
        runner._verify_path_id_authority(path_run)  # noqa: SLF001


@pytest.mark.parametrize("route", ["candidate_health_failed", "oracle_control_failed"])
def test_gate_failures_are_not_labeled_scientific_negatives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    run_dir = tmp_path / route
    run_dir.mkdir()
    runner._write_json(  # noqa: SLF001
        run_dir / "stage_ledger.json", {"schema": "fixture", "events": []}
    )
    runner._write_json(run_dir / "resource_ledger.json", {})  # noqa: SLF001
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class Governor:
        def admit(self, kind: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(("admit", kind, kwargs))
            return {
                "passed": 1,
                "reserve_remaining_seconds": 0.0,
            }

        def complete(self, kind: str, **kwargs: Any) -> None:
            calls.append(("complete", kind, kwargs))

    def seal(_run_dir: Path) -> dict[str, Any]:
        calls.append(("seal", "manifest", {}))
        return {}

    monkeypatch.setattr(runner, "_seal_manifest", seal)
    monkeypatch.setattr(
        runner,
        "verify_run",
        lambda _run_dir: (
            calls.append(("verify", route, {}))
            or {"passed": 1, "route": route}
        ),
    )
    monkeypatch.setattr(
        runner, "_failure_report", lambda _run_dir, _failure: "fixture report"  # noqa: SLF001
    )
    monkeypatch.setattr(
        runner,
        "_failure_experiment_note",
        lambda _run_dir, _failure: "fixture note",  # noqa: SLF001
    )
    error: BaseException
    if route == "candidate_health_failed":
        error = runner.CandidateHealthFailure("Gate B did not pass")
    else:
        error = runner.OracleControlFailed("Gate C did not pass")
    runner._finalize_failure(run_dir, error, Governor())  # type: ignore[arg-type]  # noqa: SLF001
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert "scientific_negative" not in failure
    assert failure["failure_terminalization_admitted"] == 1
    assert failure["failure_terminalization_reserve_seconds"] == 0.0
    assert calls[0] == (
        "admit",
        "failure_terminalization",
        {
            "predicted_seconds": runner.FAILURE_TERMINALIZATION_SECONDS,
            "predicted_bytes": runner.FAILURE_TERMINALIZATION_BYTES,
            "terminalization": True,
        },
    )
    assert [row[0] for row in calls] == [
        "admit",
        "seal",
        "complete",
        "seal",
        "verify",
    ]
    assert calls[2] == (
        "complete",
        "failure_terminalization",
        {"synchronize": False, "terminalization": True},
    )


def test_candidate_health_failure_report_binds_shared_stream_and_claim_scope(
    tmp_path: Path,
) -> None:
    _minimal_verifier_tree(tmp_path, "integrity_failed")
    (tmp_path / "candidate_audit").mkdir()
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_audit/report.json",
        {
            "candidate_vs_568": {
                "maximum_later_fraction_error": 2.220446049250313e-16,
                "maximum_target_error": 4.5075054799781356e-14,
            },
            "candidate_vs_certified": {
                "maximum_later_fraction_error": 0.0,
                "maximum_target_error": 0.0,
            },
            "rng_alignment": {
                "rng_arrays_exact": 1,
                "maximum_568_cdf_residual_at_candidate_later": 5.94e-14,
            },
        },
    )
    failure = {
        "route": "candidate_health_failed",
        "error_type": "CandidateHealthFailure",
        "error": "fixed Gate B did not pass",
        "scientific_objective_result_available": 0,
        "exact_failure_proposition": (
            "the fixed 512-lane shared-v2-Philox candidate numerical "
            "execution/integrity criterion did not pass"
        ),
        "claim_boundary": (
            "no learned-controller or generator scientific conclusion is available "
            "from this route"
        ),
    }
    report = runner._failure_report(tmp_path, failure)  # noqa: SLF001
    assert "philox4x32-10-canonical-transition-v2" in report
    assert "2.2204460492503131e-16" in report
    assert "4.5075054799781356e-14" in report
    assert "candidate_audit/outputs.npz" in report
    assert "alignment replay flag is `1`" in report
    assert "not a scientific\nnegative" in report
    assert "authorizes no\nautomatic full-scale run" in report


def test_record_review_exception_uses_governed_failure_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "review-run"
    _minimal_verifier_tree(run_dir, "review_pending")
    answers = tmp_path / "answers.csv"
    answers.write_text("sample_id,assigned_label\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "record_review",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.IntegrityFailure("review fixture failure")
        ),
    )
    monkeypatch.setattr(runner, "_failure_report", lambda *_args: "failure report")
    monkeypatch.setattr(
        runner, "_failure_experiment_note", lambda *_args: "failure note"
    )
    monkeypatch.setattr(runner, "_seal_manifest", lambda _run_dir: {})
    monkeypatch.setattr(
        runner,
        "verify_run",
        lambda _run_dir: {"passed": 1, "route": "integrity_failed"},
    )

    assert runner.main(
        [
            "record-review",
            "--run-dir",
            str(run_dir),
            "--answers",
            str(answers),
            "--reviewer",
            "fixture-reviewer",
            "--confirm-manual-review",
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {"passed": 1, "route": "integrity_failed"}
    ledger = json.loads((run_dir / "resource_ledger.json").read_text(encoding="utf-8"))
    admissions = [
        row for row in ledger["admissions"] if row["kind"] == "failure_terminalization"
    ]
    events = [
        row for row in ledger["events"] if row["kind"] == "failure_terminalization"
    ]
    assert len(admissions) == 1
    assert admissions[0]["passed"] == 1
    assert admissions[0]["terminalization"] == 1
    assert len(events) == 1
    assert events[0]["completed"] == 1
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["route"] == "integrity_failed"


@pytest.mark.parametrize("route", ["review_pending", "complete"])
def test_success_routes_reject_an_empty_objective_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    run_dir = tmp_path / route
    _minimal_verifier_tree(run_dir, route)
    runner._seal_manifest(run_dir)  # noqa: SLF001
    monkeypatch.setattr(runner, "_verify_source_bindings", lambda _run_dir: None)
    monkeypatch.setattr(
        runner, "_verify_config_and_resources", lambda _run_dir, _route: {}
    )
    with pytest.raises(
        runner.IntegrityFailure,
        match="required|missing|review.pending|complete|passing candidate audit|later stage",
    ):
        runner.verify_run(run_dir)


def test_status_and_stage_ledger_incoherence_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _minimal_verifier_tree(tmp_path, "integrity_failed")
    runner._status(tmp_path, "complete")  # noqa: SLF001
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    monkeypatch.setattr(runner, "_verify_source_bindings", lambda _run_dir: None)
    monkeypatch.setattr(
        runner, "_verify_config_and_resources", lambda _run_dir, _route: {}
    )
    with pytest.raises(runner.IntegrityFailure, match="status|stage ledger|complete"):
        runner.verify_run(tmp_path)


def test_resource_stopped_tree_rejects_out_of_order_later_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _minimal_verifier_tree(tmp_path, "resource_stopped")
    (tmp_path / "review").mkdir()
    runner._write_json(  # noqa: SLF001
        tmp_path / "review/READY.json", {"forged_after_stop": 1}
    )
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    monkeypatch.setattr(runner, "_verify_source_bindings", lambda _run_dir: None)
    with pytest.raises(
        runner.IntegrityFailure, match="artifact|later|stage|resource.stop"
    ):
        runner.verify_run(tmp_path)


def test_terminal_resource_event_is_sealed_before_the_final_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"completed": 0, "seal_generation": 0}
    calls: list[str] = []

    def seal(run_dir: Path) -> dict[str, Any]:
        state["seal_generation"] += 1
        runner._write_json(  # noqa: SLF001
            run_dir / "fixture_manifest.json",
            {
                "generation": state["seal_generation"],
                "resource_event_completed": state["completed"],
            },
        )
        calls.append("seal")
        return {}

    def verify(run_dir: Path) -> dict[str, Any]:
        receipt = json.loads(
            (run_dir / "fixture_manifest.json").read_text(encoding="utf-8")
        )
        assert receipt["resource_event_completed"] == state["completed"]
        calls.append("verify")
        return {"passed": 1, "tree_digest": str(receipt["generation"])}

    class Governor:
        def complete(self, kind: str, **_kwargs: Any) -> None:
            assert kind == "post_evaluator_finalization"
            state["completed"] = 1
            calls.append("complete")

    monkeypatch.setattr(runner, "_seal_manifest", seal)
    monkeypatch.setattr(runner, "verify_run", verify)
    receipt = runner._finish_priced_terminalization(  # noqa: SLF001
        tmp_path,
        Governor(),  # type: ignore[arg-type]
        "post_evaluator_finalization",
    )
    assert calls == ["seal", "verify", "complete", "seal", "verify"]
    assert receipt["tree_digest"] == "2"
    source = (
        REPOSITORY_ROOT / "mnist/diag_eulerian_jacobi_ddpm_candidate_pilot.py"
    ).read_text(encoding="utf-8")
    assert '_admit_major_stage(run_dir, governor, "post_evaluator_finalization")' in source
    assert '_admit_major_stage(run_dir, governor, "record_review_terminalization")' in source
    assert "_finish_priced_terminalization" in source


def test_report_contains_checkpoint_gates_resources_commands_and_action(
    tmp_path: Path,
) -> None:
    _minimal_verifier_tree(tmp_path, "complete")
    for relative in (
        "training",
        "candidate_audit",
        "oracle_control",
        "evaluation",
        "review",
        "resource_smoke",
    ):
        (tmp_path / relative).mkdir(exist_ok=True)
    bindings = json.loads(
        (tmp_path / "source_bindings.json").read_text(encoding="utf-8")
    )
    bindings["git"] = {"revision": "fixture-revision"}
    bindings["historical_source_inventory"] = EXPECTED_PROTECTED_SOURCE_HASHES
    runner._write_json(tmp_path / "source_bindings.json", bindings)  # noqa: SLF001
    (tmp_path / "training/selected_checkpoint.pt").write_bytes(b"fixture checkpoint")
    runner._write_json(  # noqa: SLF001
        tmp_path / "training/selection.json",
        {
            "selected_update": 500,
            "selected_validation_normalized_mse": 0.25,
            "selection": runner.FROZEN_CONFIG["training"]["selection"],
            "checkpoint_sha256": _sha256(
                tmp_path / "training/selected_checkpoint.pt"
            ),
        },
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_audit/report.json",
        {
            "gate_type": "execution/integrity",
            "passed": 1,
            "candidate_vs_certified": {
                "maximum_later_fraction_error": 0.0,
                "maximum_target_error": 0.0,
            },
            "candidate_mode_telemetry": {"maximum_adaptive_modes": 256},
        },
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "oracle_control/metrics.json",
        {
            "gate_type": "execution/integrity",
            "passed": 1,
            "oracle_improved_path_count": 10,
        },
    )
    forward_marker = {
        "gate_type": "diagnostic threshold",
        "passed": 1,
        "learned_l1_win_count": 15,
        "aggregate_relative_l1_improvement": 0.1,
        "learned_controller_rms": 0.01,
    }
    evaluator_marker = {
        "gate_type": "diagnostic threshold",
        "passed": 1,
        "learned_prior_requested_label_accuracy": 0.5,
        "null_prior_requested_label_accuracy": 0.1,
        "paired_requested_log_probability_win_count": 15,
        "mean_paired_requested_log_probability_improvement": 0.2,
    }
    runner._write_json(  # noqa: SLF001
        tmp_path / "evaluation/metrics.json",
        {
            "evaluator_marker": evaluator_marker,
            "forward_direct_marker": forward_marker,
        },
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "review/metrics.json",
        {
            "aggregate_paired_recognizable_difference": 10,
            "aggregate_paired_requested_label_difference": 10,
        },
    )
    runner._write_text(  # noqa: SLF001
        tmp_path / "review/record_command.txt", "record-review fixture\n"
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "resource_smoke/timings.json",
        {"projection": {"projected_remaining_seconds": 100.0}},
    )
    outcome = {
        "route": "approximate_candidate_feasibility_reference_audit_next",
        "human_marker": {
            "gate_type": "diagnostic threshold",
            "passed": 1,
            "learned_recognizable_count": 15,
            "learned_requested_label_count": 12,
        },
    }
    report = runner._production_report(tmp_path, outcome).lower()  # noqa: SLF001
    for required in (
        "selected checkpoint",
        "execution/integrity",
        "diagnostic-threshold",
        "resource accounting",
        "exact command",
        "deliberate omissions",
        "outcome-to-action",
        "training/selected_checkpoint.pt",
        "candidate_audit/report.json",
        "oracle_control/metrics.json",
        "evaluation/metrics.json",
        "command.txt",
    ):
        assert required in report

    forward_marker["passed"] = 0
    runner._write_json(  # noqa: SLF001
        tmp_path / "evaluation/metrics.json",
        {
            "evaluator_marker": evaluator_marker,
            "forward_direct_marker": forward_marker,
        },
    )
    mismatch_outcome = {
        "route": "human_negative_evaluator_positive_audit",
        "human_marker": {
            "gate_type": "diagnostic threshold",
            "passed": 0,
            "learned_recognizable_count": 0,
            "learned_requested_label_count": 0,
        },
    }
    mismatch_report = runner._production_report(  # noqa: SLF001
        tmp_path, mismatch_outcome
    )
    assert (
        "| Human negative, evaluator positive | Treat task result as negative; "
        "audit evaluator/render/proxy mismatch only; do not select samples |"
        in mismatch_report
    )
    next_action = next(
        line for line in mismatch_report.splitlines() if line.startswith("Next action:")
    )
    assert next_action == (
        "Next action: treat the task result as negative; audit "
        "evaluator/render/proxy mismatch only; do not select samples. "
        "Any reference audit requires a separate approval."
    )


def test_failure_routes_forbid_later_stage_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _minimal_verifier_tree(tmp_path, "integrity_failed")
    (tmp_path / "training").mkdir()
    (tmp_path / "training/selected_checkpoint.pt").write_bytes(b"must not exist")
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    monkeypatch.setattr(runner, "_verify_source_bindings", lambda _run_dir: None)
    with pytest.raises(runner.IntegrityFailure, match="beyond the completed stage"):
        runner.verify_run(tmp_path)


def test_verify_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _minimal_verifier_tree(tmp_path, "integrity_failed")
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    monkeypatch.setattr(runner, "_verify_source_bindings", lambda _run_dir: None)
    before = runner._tree_snapshot(tmp_path)  # noqa: SLF001
    receipt = runner.verify_run(tmp_path)
    after = runner._tree_snapshot(tmp_path)  # noqa: SLF001
    assert before == after
    assert receipt["passed"] == 1
    assert receipt["route"] == "integrity_failed"

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["populations"]["automatic_full_scale_launches"] = 1
    runner._write_json(tmp_path / "config.json", config)  # noqa: SLF001
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    with pytest.raises(runner.IntegrityFailure, match="scientific configuration"):
        runner.verify_run(tmp_path)


def test_complete_route_verification_is_read_only_after_semantic_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _minimal_verifier_tree(tmp_path, "complete")
    for relative in (
        "candidate_audit",
        "resource_smoke",
        "oracle_control",
        "populations",
        "evaluation",
        "review",
    ):
        (tmp_path / relative).mkdir(exist_ok=True)
    runner._write_json(  # noqa: SLF001
        tmp_path / "candidate_audit/report.json", {"passed": 1}
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "resource_smoke/timings.json", {"fixture": 1}
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "oracle_control/metrics.json", {"passed": 1}
    )
    runner._write_npz(  # noqa: SLF001
        tmp_path / "populations/raw_populations.npz", fixture=np.zeros(1)
    )
    runner._write_json(  # noqa: SLF001
        tmp_path / "POPULATIONS_SEALED.json",
        {"artifact_count": 0, "tree_digest": runner._tree_digest([]), "artifacts": []},  # noqa: SLF001
    )
    for relative in (
        "evaluation/OPEN_EVENT.json",
        "evaluation/metrics.json",
        "evaluation/evaluator_binding.json",
        "review/READY.json",
        "review/review_key.json",
    ):
        runner._write_json(tmp_path / relative, {})  # noqa: SLF001
    runner._write_npz(  # noqa: SLF001
        tmp_path / "evaluation/outputs.npz",
        learned_prior_sample_ids=np.asarray([], dtype=np.str_),
        learned_prior_predictions=np.asarray([], dtype=np.int64),
        null_prior_sample_ids=np.asarray([], dtype=np.str_),
        null_prior_predictions=np.asarray([], dtype=np.int64),
    )
    for relative in (
        "review/review_template.csv",
        "review/submitted_answers.csv",
    ):
        (tmp_path / relative).write_text("fixture\n", encoding="utf-8")
    (tmp_path / "review/blind_contact_sheet.png").write_bytes(b"fixture")
    forward = {"passed": 1}
    evaluator = {"passed": 1}
    human = {"passed": 1}
    replayed_review = {
        "learned": {"recognizable_count": 15, "requested_label_count": 12},
        "null": {"recognizable_count": 0, "requested_label_count": 0},
        "learned_counts_exceed_null": 1,
        "rows": [],
        "paired_paths": [],
        "aggregate_paired_recognizable_difference": 15,
        "aggregate_paired_requested_label_difference": 12,
        "paired_recognizable_win_loss_tie": {"wins": 15, "losses": 0, "ties": 5},
        "paired_requested_label_win_loss_tie": {"wins": 12, "losses": 0, "ties": 8},
    }
    saved_review = {
        **replayed_review,
        "human_marker": human,
        "human_machine_disagreement_count": 0,
        "answers_source_path": str(tmp_path / "answers-source.csv"),
        "reviewer": "fixture-reviewer",
    }
    runner._write_json(  # noqa: SLF001
        tmp_path / "review/metrics.json", saved_review
    )
    review_command = runner._canonical_command_text(  # noqa: SLF001
        [
            str(Path(runner.sys.executable).resolve()),
            "-B",
            "-m",
            "mnist.diag_eulerian_jacobi_ddpm_candidate_pilot",
            "record-review",
            "--run-dir",
            str(tmp_path),
            "--answers",
            saved_review["answers_source_path"],
            "--reviewer",
            saved_review["reviewer"],
            "--confirm-manual-review",
        ]
    )
    runner._write_text(  # noqa: SLF001
        tmp_path / "review/record_command.txt", review_command
    )
    outcome = {
        "route": runner.route_outcome(forward, human, evaluator),
        "human_marker": human,
    }
    runner._write_json(tmp_path / "outcome.json", outcome)  # noqa: SLF001
    (tmp_path / "REPORT.md").write_text("fixture report", encoding="utf-8")
    (tmp_path / "experiment_note.md").write_text("fixture note", encoding="utf-8")
    monkeypatch.setattr(runner, "_verify_source_bindings", lambda _run_dir: None)
    monkeypatch.setattr(
        runner, "_verify_config_and_resources", lambda _run_dir, _route: {}
    )
    monkeypatch.setattr(runner, "_verify_candidate_audit", lambda _run_dir: {"passed": 1})
    monkeypatch.setattr(runner, "_verify_oracle_control", lambda _run_dir: {"passed": 1})
    monkeypatch.setattr(runner, "_verify_populations", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_verify_review_bundle", lambda _run_dir: None)
    monkeypatch.setattr(
        runner,
        "_verify_evaluation",
        lambda _run_dir: {
            "forward_direct_marker": forward,
            "evaluator_marker": evaluator,
        },
    )
    monkeypatch.setattr(
        runner,
        "compute_human_review_metrics",
        lambda _answers, _key: replayed_review,
    )
    monkeypatch.setattr(
        runner, "_human_marker_from_metrics", lambda _run_dir, _metrics: human
    )
    monkeypatch.setattr(runner, "_verify_reports", lambda _run_dir, _outcome: None)
    runner._seal_manifest(tmp_path)  # noqa: SLF001
    before = runner._tree_snapshot(tmp_path)  # noqa: SLF001
    receipt = runner.verify_run(tmp_path)
    after = runner._tree_snapshot(tmp_path)  # noqa: SLF001
    assert before == after
    assert receipt["passed"] == 1
    assert receipt["route"] == "complete"


def test_verify_accepts_valid_resource_stopped_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_verify_source_bindings", lambda _run_dir: None)
    partial = tmp_path / "partial"
    _minimal_verifier_tree(partial, "resource_stopped")
    runner._seal_manifest(partial)  # noqa: SLF001
    partial_receipt = runner.verify_run(partial)
    assert partial_receipt["passed"] == 1
    assert partial_receipt["route"] == "resource_stopped"


def test_cli_rejects_cpu_production_placeholder_approval_and_nonempty_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(runner.IntegrityFailure, match="requires CUDA"):
        runner.initialize_run(
            tmp_path,
            tmp_path / "missing.arff",
            tmp_path / "missing-ddpm",
            tmp_path / "cpu-run",
            device="cpu",
            approval_id="real-approval",
            maximum_active_seconds=10_800,
            maximum_storage_mib=2_048,
            maximum_cuda_fraction=0.75,
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(runner.IntegrityFailure, match="fresh real approval"):
        runner.initialize_run(
            tmp_path,
            tmp_path / "missing.arff",
            tmp_path / "missing-ddpm",
            tmp_path / "placeholder-run",
            device="cuda:0",
            approval_id="<fresh-approval>",
            maximum_active_seconds=10_800,
            maximum_storage_mib=2_048,
            maximum_cuda_fraction=0.75,
        )

    arff = tmp_path / "mnist.arff"
    arff.write_bytes(b"authority stand-in")
    ddpm = tmp_path / "ddpm"
    ddpm.mkdir()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(runner, "_file_sha256", lambda _path: runner.MNIST_ARFF_SHA256)
    with pytest.raises(runner.IntegrityFailure, match="fresh and absent"):
        runner.initialize_run(
            tmp_path,
            arff,
            ddpm,
            occupied,
            device="cuda:0",
            approval_id="user-approved-candidate-k128-20260817",
            maximum_active_seconds=10_800,
            maximum_storage_mib=2_048,
            maximum_cuda_fraction=0.75,
        )
    assert (occupied / "existing.txt").read_text(encoding="utf-8") == "preserve"


def test_run_never_auto_launches_full_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    observed_initialize: dict[str, Any] = {}
    roles = {
        "train_mixed_masses": np.empty((0, core.STATE_SIZE)),
        "train_labels": np.empty(0, dtype=np.int64),
        "validation_mixed_masses": np.empty((0, core.STATE_SIZE)),
        "validation_labels": np.empty(0, dtype=np.int64),
        "validation_arff_indices": np.empty(0, dtype=np.int64),
    }

    class Governor:
        def admit(self, kind: str, **kwargs: Any) -> None:
            assert kind == "data_roles_write"
            assert kwargs["predicted_bytes"] > 0
            calls.append("admit_data_roles")

        def complete(self, kind: str, **_kwargs: Any) -> None:
            assert kind == "data_roles_write"
            calls.append("complete_data_roles")

    def initialize(*_args: Any, **kwargs: Any) -> tuple[Path, object, object]:
        calls.append("initialize")
        observed_initialize.update(kwargs)
        return tmp_path, Governor(), object()

    monkeypatch.setattr(runner, "initialize_run", initialize)
    arff_access = {
        "content_rows_read": 60_000,
        "last_content_row_index": 59_999,
        "terminal_content_rows_read": 0,
        "last_text_line_number_read": 60_001,
        "full_file_read_purpose": "sha256-only",
        "full_file_sha256": runner.MNIST_ARFF_SHA256,
    }
    monkeypatch.setattr(
        runner,
        "_load_train_validation_mnist_strict",  # noqa: SLF001
        lambda _path: (
            calls.append("load_data")
            or (None, None, None, None, arff_access)
        ),
    )

    def prepare_roles(*_args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        assert kwargs == {"arff_access": arff_access}
        calls.append("data_roles")
        return roles

    monkeypatch.setattr(runner, "prepare_data_roles", prepare_roles)
    monkeypatch.setattr(runner, "_append_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "prepare_runtime_stage",
        lambda *_args: (calls.append("prepare_candidate") or object()),
    )
    monkeypatch.setattr(
        runner,
        "run_candidate_audit",
        lambda *_args: calls.append("candidate_audit"),
    )
    monkeypatch.setattr(
        runner,
        "run_resource_smoke",
        lambda *_args: calls.append("resource_smoke"),
    )
    monkeypatch.setattr(
        runner,
        "run_oracle_control",
        lambda *_args: (calls.append("oracle") or (object(), object(), {})),
    )
    monkeypatch.setattr(
        runner,
        "build_forward_record_caches",
        lambda *_args: (calls.append("records") or (object(), object())),
    )
    monkeypatch.setattr(
        runner,
        "train_candidate_model",
        lambda *_args: (calls.append("training") or (object(), object())),
    )
    monkeypatch.setattr(
        runner,
        "sample_objective_populations",
        lambda *_args: (calls.append("objective_populations") or {}),
    )
    monkeypatch.setattr(
        runner,
        "seal_populations",
        lambda *_args: calls.append("population_seal"),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_sealed_populations",
        lambda *_args: calls.append("evaluation_and_review_bundle"),
    )
    monkeypatch.setattr(
        runner,
        "_admit_major_stage",
        lambda *_args: (calls.append("post_evaluator_admission") or {"passed": 1}),
    )
    monkeypatch.setattr(
        runner,
        "_finish_priced_terminalization",
        lambda *_args: (
            calls.append("priced_terminalization")
            or {"passed": 1, "route": "review_pending"}
        ),
    )
    args = SimpleNamespace(
        run_dir=str(tmp_path),
        arff="mnist.arff",
        ddpm_run_dir="ddpm",
        device="cuda:0",
        approval_id="user-approved-candidate-k128-20260817",
        max_active_seconds=10_800.0,
        max_storage_mib=2_048.0,
        max_cuda_fraction=0.75,
    )
    assert runner.run_production(args) == 0
    assert calls == [
        "initialize",
        "admit_data_roles",
        "load_data",
        "data_roles",
        "complete_data_roles",
        "prepare_candidate",
        "candidate_audit",
        "resource_smoke",
        "oracle",
        "records",
        "training",
        "objective_populations",
        "population_seal",
        "evaluation_and_review_bundle",
        "post_evaluator_admission",
        "priced_terminalization",
    ]
    assert observed_initialize == {
        "device": "cuda:0",
        "approval_id": "user-approved-candidate-k128-20260817",
        "maximum_active_seconds": 10_800.0,
        "maximum_storage_mib": 2_048.0,
        "maximum_cuda_fraction": 0.75,
    }
    assert _config()["populations"]["automatic_full_scale_launches"] == 0
