from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_certificate_semantics import (
    CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
    CERTIFICATE_SEMANTICS_PAYLOAD_ADAPTER_VERSION,
    CertificateSemanticsError,
    compare_certificate_semantics,
    comparator_payload_from_certified_cuda_batch,
    comparator_payload_from_eager_midpoint_branch,
    comparator_payload_from_multipath_capture,
    eager_base_record_payload_availability,
)


def _batch(count: int = 6) -> dict[str, np.ndarray]:
    earlier = np.linspace(0.1, 0.6, count, dtype=np.float64)
    exposure = np.ones(count, dtype=np.float64)
    exposure[-1] = 0.0
    active = exposure > 0.0
    later = earlier + np.where(active, 0.01, 0.0)
    target = np.where(active, np.linspace(-0.2, 0.2, count), 0.0).astype(
        np.float64
    )
    return {
        "transition_ids": np.arange(100, 100 + count, dtype=np.uint64),
        "earlier_head_fraction": earlier,
        "exposure": exposure,
        "later_head_fraction": later,
        "denoising_target": target,
        "active_mask": active,
        "certified_mask": active.copy(),
        "certificate_codes": np.where(active, 15, 0).astype(np.uint8),
        "prefix_bits": np.where(active, 64, 0).astype(np.int32),
        "mode_counts": np.where(active, 256, 0).astype(np.int32),
        "strengthened_mask": np.zeros(count, dtype=bool),
        "fallback_mask": np.zeros(count, dtype=bool),
        "quantile_lower": later - np.where(active, 1.0e-16, 0.0),
        "quantile_upper": later + np.where(active, 1.0e-16, 0.0),
        "target_lower": target - np.where(active, 1.0e-16, 0.0),
        "target_upper": target + np.where(active, 1.0e-16, 0.0),
    }


def _copy(value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.array(array, copy=True) for name, array in value.items()}


def test_metadata_only_differences_are_advisory() -> None:
    adaptive = _batch()
    eager = _copy(adaptive)
    active = eager["active_mask"]
    eager["certificate_codes"][active] = 9
    eager["prefix_bits"][active] = 128
    eager["mode_counts"][active] = 128
    eager["strengthened_mask"][active] = True
    eager["quantile_lower"][active] -= 1.0e-15
    eager["quantile_upper"][active] += 1.0e-15

    result = compare_certificate_semantics(adaptive, eager)
    assert result["schema"] == CERTIFICATE_SEMANTICS_COMPARATOR_VERSION
    assert result["comparator_valid"] == 1
    assert result["certificate_semantics_comparator_valid"] == 1
    assert result["scientific_payload_equal"] == 1
    assert result["certificate_semantics_equal"] == 1
    assert result["scheduler_seam_valid"] == 1
    assert result["proof_metadata_equal"] == 0
    assert result["proof_metadata_equality_required"] == 0
    assert {
        "certificate_codes",
        "prefix_bits",
        "mode_counts",
        "strengthened_mask",
        "quantile_lower",
        "quantile_upper",
    }.issubset(result["differing_proof_metadata_fields"])
    assert result["mismatch_count"] == 0


def test_fallback_route_and_reason_differences_are_advisory() -> None:
    adaptive = _batch()
    eager = _copy(adaptive)
    adaptive["fallback_mask"] = np.zeros(6, dtype=bool)
    eager["fallback_mask"] = np.asarray(
        [True, False, False, False, False, False], dtype=bool
    )
    adaptive["fallback_reason_codes"] = np.zeros(6, dtype=np.int32)
    eager["fallback_reason_codes"] = np.asarray(
        [7, 0, 0, 0, 0, 0], dtype=np.int32
    )

    result = compare_certificate_semantics(adaptive, eager)

    assert result["comparator_valid"] == 1
    assert result["scheduler_seam_valid"] == 1
    assert result["proof_metadata_equal"] == 0
    assert {"fallback_mask", "fallback_reason_codes"}.issubset(
        result["differing_proof_metadata_fields"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("transition_ids", 999),
        ("earlier_head_fraction", 0.125),
        ("exposure", 0.5),
        ("later_head_fraction", 0.75),
        ("denoising_target", -0.75),
    ],
)
def test_scientific_identity_state_and_target_mismatches_fail(
    field: str, replacement: float | int
) -> None:
    left = _batch()
    right = _copy(left)
    right[field][0] = replacement
    result = compare_certificate_semantics(left, right)
    assert result["comparator_valid"] == 0
    assert result["scientific_payload_equal"] == 0
    assert result["scheduler_seam_valid"] == 0
    assert result["mismatch_counts"][field] == 1
    assert result["mismatch_records"][0]["field"] == field


def test_active_mask_and_canonical_order_mismatches_fail() -> None:
    left = _batch()
    right = _copy(left)
    right["active_mask"][0] = False
    result = compare_certificate_semantics(left, right)
    assert result["scientific_payload_equal"] == 0
    assert result["mismatch_counts"]["active_mask"] == 1

    right = _copy(left)
    right["transition_ids"][[0, 1]] = right["transition_ids"][[1, 0]]
    result = compare_certificate_semantics(left, right)
    assert result["scientific_payload_equal"] == 0
    assert result["mismatch_counts"]["transition_ids"] == 2
    assert result["scheduler_seam_valid"] == 0


def test_authorization_mismatch_and_invalid_authorization_fail() -> None:
    left = _batch()
    right = _copy(left)
    right["certified_mask"][0] = False
    result = compare_certificate_semantics(left, right)
    assert result["scientific_payload_equal"] == 1
    assert result["certificate_semantics_equal"] == 0
    assert result["right_authorization_valid"] == 0
    assert result["right_authorization_counts"]["active_uncertified_count"] == 1
    assert result["mismatch_counts"]["normalized_authorization"] == 1
    assert result["scheduler_seam_valid"] == 0


def test_duplicate_active_transition_identity_fails_closed() -> None:
    left = _batch()
    left["transition_ids"][1] = left["transition_ids"][0]
    right = _copy(left)

    result = compare_certificate_semantics(left, right)

    assert result["scientific_payload_equal"] == 1
    assert result["left_authorization_valid"] == 0
    assert result["right_authorization_valid"] == 0
    assert (
        result["left_authorization_counts"][
            "active_duplicate_transition_id_count"
        ]
        == 1
    )
    assert (
        result["right_authorization_counts"][
            "active_duplicate_transition_id_count"
        ]
        == 1
    )
    assert result["certificate_semantics_equal"] == 0
    assert result["comparator_valid"] == 0
    assert result["scheduler_seam_valid"] == 0


def test_identically_invalid_structural_noop_cannot_pass() -> None:
    left = _batch()
    left["later_head_fraction"][-1] += 0.1
    right = _copy(left)
    result = compare_certificate_semantics(left, right)
    assert result["scientific_payload_equal"] == 1
    assert result["left_authorization_counts"]["inactive_state_change_count"] == 1
    assert result["right_authorization_counts"]["inactive_state_change_count"] == 1
    assert result["certificate_semantics_equal"] == 0
    assert result["comparator_valid"] == 0


def test_final_state_is_part_of_scientific_payload() -> None:
    left = _batch()
    right = _copy(left)
    left_state = np.arange(12, dtype=np.float64).reshape(3, 4)
    right_state = left_state.copy()
    right_state[1, 2] += 1.0
    result = compare_certificate_semantics(
        left,
        right,
        left_final_state=left_state,
        right_final_state=right_state,
    )
    assert result["final_state_compared"] == 1
    assert result["final_state_equal"] == 0
    assert result["scientific_payload_equal"] == 0
    assert result["scheduler_seam_valid"] == 0


def test_mismatch_records_have_one_global_bound() -> None:
    left = _batch(100)
    right = _copy(left)
    right["later_head_fraction"][:50] += 0.1
    right["denoising_target"][50:99] += 0.2
    result = compare_certificate_semantics(
        left, right, maximum_mismatch_records=3
    )
    assert result["mismatch_count"] >= 99
    assert result["mismatch_record_count"] == 3
    assert len(result["mismatch_records"]) == 3
    assert result["mismatch_records_truncated"] == 1
    with pytest.raises(CertificateSemanticsError, match="must lie"):
        compare_certificate_semantics(left, right, maximum_mismatch_records=0)


def test_malformed_shapes_and_partial_enclosures_fail_closed() -> None:
    malformed = _batch()
    malformed["active_mask"] = malformed["active_mask"][:-1]
    with pytest.raises(CertificateSemanticsError, match="transition shape"):
        compare_certificate_semantics(malformed, _batch())

    partial = _batch()
    partial.pop("target_upper")
    with pytest.raises(CertificateSemanticsError, match="supplied together"):
        compare_certificate_semantics(partial, _batch())


def test_binary64_payload_and_normalized_masks_fail_closed() -> None:
    low_precision = _batch()
    low_precision["denoising_target"] = low_precision[
        "denoising_target"
    ].astype(np.float32)
    with pytest.raises(CertificateSemanticsError, match="float64 dtype"):
        compare_certificate_semantics(low_precision, _batch())

    malformed_mask = _batch()
    malformed_mask["certified_mask"] = np.where(
        malformed_mask["active_mask"], 2, 0
    ).astype(np.uint8)
    with pytest.raises(CertificateSemanticsError, match="integer 0/1 mask"):
        compare_certificate_semantics(malformed_mask, _batch())

    with pytest.raises(CertificateSemanticsError, match="final states"):
        compare_certificate_semantics(
            _batch(),
            _batch(),
            left_final_state=np.ones(4, dtype=np.float32),
            right_final_state=np.ones(4, dtype=np.float64),
        )


def test_certified_cuda_batch_adapter_retains_complete_payload() -> None:
    source = _batch()
    payload = comparator_payload_from_certified_cuda_batch(source)
    assert payload["payload_adapter"] == (
        CERTIFICATE_SEMANTICS_PAYLOAD_ADAPTER_VERSION
    )
    assert payload["payload_source"] == "certified_cuda_batch"
    assert payload["unavailable_proof_metadata_fields"]
    result = compare_certificate_semantics(payload, source)
    assert result["comparator_valid"] == 1


def test_eager_midpoint_adapter_reconstructs_exact_lane_inputs() -> None:
    from mnist.d0_jacobi_rb_boundary_tangent_cache import MIDPOINT_FRACTIONS
    from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE
    from mnist.d0_jacobi_rb_reverse_controller import controller_transition_ids

    path_ids = (17, 23)
    path_count = len(path_ids)
    midpoint_count = len(MIDPOINT_FRACTIONS)
    shape = (midpoint_count, path_count, EDGES_PER_PHASE)
    states = torch.full(
        (path_count, 28 * 28), 1.0 / (28 * 28), dtype=torch.float64
    ).contiguous()
    later = torch.full(shape, 0.5, dtype=torch.float64)
    scientific = SimpleNamespace(
        path_ids=path_ids,
        outer_step=15,
        phase=0,
        midpoint_fractions=MIDPOINT_FRACTIONS,
        later_full_state=states.unsqueeze(0).expand(
            midpoint_count, -1, -1
        ).clone(),
        later_head_fraction=later,
        denoising_target=torch.zeros(shape, dtype=torch.float64),
        certificate_codes=torch.full(shape, 15, dtype=torch.uint8),
        mode_counts=torch.full(shape, 128, dtype=torch.int32),
        prefix_bits=torch.full(shape, 64, dtype=torch.int32),
        fallback_mask=torch.zeros(shape, dtype=torch.bool),
        strengthened_mask=torch.zeros(shape, dtype=torch.bool),
    )
    fused = SimpleNamespace(
        batch=scientific,
        fallback_reason_codes=torch.zeros(shape, dtype=torch.uint8),
    )
    branch = SimpleNamespace(pre_phase_states=states, batch=fused)
    payload = comparator_payload_from_eager_midpoint_branch(branch)
    expected_ids = controller_transition_ids(
        path_ids,
        outer_step=15,
        phase=0,
        reverse_microstep=0,
        role="partial_phase_target_prefix",
        device="cpu",
    )
    assert payload["transition_ids"].shape == shape
    assert torch.equal(payload["transition_ids"][0], expected_ids)
    assert torch.equal(
        payload["earlier_head_fraction"], torch.full(shape, 0.5, dtype=torch.float64)
    )
    assert bool(torch.all(payload["active_mask"]))
    assert bool(torch.all(payload["certified_mask"]))
    assert payload["unavailable_proof_metadata_fields"] == [
        "candidate_denoising_target",
        "candidate_later_head_fraction",
        "candidate_match_mask",
        "cuda_certified_mask",
        "fallback_mode_counts",
        "quantile_lower",
        "quantile_upper",
        "target_lower",
        "target_upper",
    ]


def test_multipath_capture_adapter_reconstructs_base_inputs() -> None:
    from mnist.d0_jacobi_rb_learnability import matching_indices

    captured_ids = (3, 7)
    provided_ids = (7, 3)
    state_by_id = {
        3: np.full(28 * 28, 1.0 / (28 * 28), dtype=np.float64),
        7: np.full(28 * 28, 1.0 / (28 * 28), dtype=np.float64),
    }
    state_by_id[7][0] += state_by_id[7][1] * 0.25
    state_by_id[7][1] *= 0.75
    initial = np.stack([state_by_id[item] for item in provided_ids])
    canonical = np.stack([state_by_id[item] for item in captured_ids])
    tails, heads = matching_indices(device="cpu")
    tails0 = tails[0].numpy()
    heads0 = heads[0].numpy()
    pair = canonical[:, tails0] + canonical[:, heads0]
    later = canonical[:, heads0] / pair
    capture = SimpleNamespace(
        path_ids=captured_ids,
        outer_steps=(0,),
        phases=(0,),
        later_head_fractions=later[None, :, :].copy(),
        denoising_targets=np.zeros((1, 2, 392), dtype=np.float64),
        certificate_codes=np.full((1, 2, 392), 15, dtype=np.uint8),
        post_phase_states=canonical[None, :, :].copy(),
    )
    payload = comparator_payload_from_multipath_capture(
        capture,
        initial_states=initial,
        initial_state_path_ids=provided_ids,
    )
    assert payload["transition_ids"].shape == (1, 2, 392)
    assert np.array_equal(payload["earlier_head_fraction"][0], later)
    assert np.array_equal(payload["final_state"], canonical)
    assert np.all(payload["certified_mask"])


def test_summary_only_eager_base_record_reports_hash_limitations() -> None:
    execution = SimpleNamespace(
        base_record={
            "batch_output_sha256": "output",
            "batch_final_state_sha256": "state",
            "batch_certificate_sha256": "proof",
            "diagnostics": {
                "transition_count": 42,
                "certified_count": 42,
                "forbidden_counts": {"nonfinite_count": 0},
            },
        }
    )
    report = eager_base_record_payload_availability(execution)
    assert report["raw_capture_available"] == 0
    assert report["comparator_payload_available"] == 0
    assert "denoising_target" in report[
        "missing_scientific_fields_without_capture"
    ]
    assert report[
        "combined_output_hash_equal_is_sufficient_but_not_necessary"
    ] == 1
    assert report["advisory_batch_certificate_sha256"] == "proof"
