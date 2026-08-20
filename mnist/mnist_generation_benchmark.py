from __future__ import annotations

"""Array-level MNIST data, evaluation, rendering, and review boundary."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from mnist.conditioned_diffusion import SmallMnistCNN, evaluate_image_classifier, train_image_classifier

MNIST_ARFF_SHA256 = "418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b"
TRAIN_START, TRAIN_STOP, VALIDATION_START, VALIDATION_STOP = 0, 55_000, 55_000, 60_000
TEST_START, TEST_STOP = 60_000, 70_000
TRAIN_SLICE, VALIDATION_SLICE, TEST_SLICE = (0, 55_000), (55_000, 60_000), (60_000, 70_000)
FIXED_SPLITS = {"train": TRAIN_SLICE, "validation": VALIDATION_SLICE, "test": TEST_SLICE}
REVIEW_ASSIGNMENTS = frozenset([*(str(i) for i in range(10)), "noise", "ambiguous"])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mnist_arff_slice(path: str | Path, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(start, int) or not isinstance(stop, int) or start < 0 or stop < start:
        raise ValueError("require integer 0 <= start <= stop")
    images, labels, in_data, row = [], [], False, 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not in_data:
                in_data = text.upper() == "@DATA"
                continue
            if not text or text.startswith("%"):
                continue
            if row >= stop:
                break
            if row >= start:
                fields = text.split(",")
                if len(fields) != 785:
                    raise ValueError(f"ARFF row {row} (line {line_number}) must have 785 fields")
                try:
                    values = np.asarray(fields, dtype=np.float64)
                except ValueError as error:
                    raise ValueError(f"ARFF row {row} has a nonnumeric field") from error
                if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
                    raise ValueError(f"ARFF row {row} must contain finite integers")
                pixels, label = values[:784], int(values[-1])
                if np.any((pixels < 0) | (pixels > 255)) or not 0 <= label <= 9:
                    raise ValueError(f"ARFF row {row} has an out-of-range pixel or label")
                images.append(pixels.astype(np.uint8).reshape(28, 28))
                labels.append(label)
            row += 1
    if not in_data:
        raise ValueError("ARFF file has no @DATA marker")
    if len(images) != stop - start:
        raise ValueError(f"ARFF has only {row} rows; requested stop={stop}")
    if not images:
        return np.empty((0, 28, 28), np.uint8), np.empty(0, np.int64)
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def _authenticated(path: str | Path) -> Path:
    path, expected = Path(path), MNIST_ARFF_SHA256
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"MNIST ARFF hash mismatch: expected {expected}, got {actual}")
    return path


def load_train_validation_mnist(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images, labels = read_mnist_arff_slice(_authenticated(path), TRAIN_START, VALIDATION_STOP)
    return images[:TRAIN_STOP], labels[:TRAIN_STOP], images[TRAIN_STOP:], labels[TRAIN_STOP:]


def load_test_mnist_terminal(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    return read_mnist_arff_slice(_authenticated(path), TEST_START, TEST_STOP)


def _images(images: np.ndarray, name: str = "images") -> np.ndarray:
    array = np.asarray(images)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[1:] != (28, 28):
        raise ValueError(f"{name} must be uint8 (N,28,28)")
    return array


def _labels(labels: np.ndarray, count: int, name: str = "labels") -> np.ndarray:
    array = np.asarray(labels)
    if array.shape != (count,) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer ({count},)")
    array = array.astype(np.int64, copy=False)
    if np.any((array < 0) | (array > 9)):
        raise ValueError(f"{name} must contain only 0,...,9")
    return array


def _ids(sample_ids: Sequence[str] | np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(sample_ids)
    if values.shape != (count,) or not all(isinstance(x, (str, np.str_)) for x in values):
        raise ValueError(f"sample_ids must be strings ({count},)")
    values = np.asarray([str(x) for x in values], dtype=np.str_)
    if any(not x.strip() for x in values) or len(set(values.tolist())) != count:
        raise ValueError("sample_ids must be nonempty and unique")
    return values


def uint8_to_unit_interval(images: np.ndarray) -> np.ndarray:
    return (_images(images).astype(np.float32) / np.float32(255))[:, None]


def uint8_to_model_space(images: np.ndarray) -> np.ndarray:
    return np.float32(2) * uint8_to_unit_interval(images) - np.float32(1)


def model_space_to_uint8(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.dtype.kind != "f" or array.ndim != 4 or array.shape[1:] != (1, 28, 28):
        raise ValueError("images must be floating point (N,1,28,28)")
    if not np.all(np.isfinite(array)):
        raise ValueError("images must be finite")
    return np.rint((np.clip(array[:, 0], -1, 1) + 1) * 127.5).astype(np.uint8)


uint8_to_eval, uint8_to_model, model_to_uint8 = uint8_to_unit_interval, uint8_to_model_space, model_space_to_uint8


def validate_generated_batch(images: np.ndarray, requested_labels: np.ndarray, sample_ids: Sequence[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = _images(images)
    if not len(images):
        raise ValueError("generated batch is empty")
    return images, _labels(requested_labels, len(images)), _ids(sample_ids, len(images))


def _eval_images(images: np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(images)
    valid = array.dtype.kind == "f" and array.shape == (count, 1, 28, 28)
    if not valid or not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be finite float [0,1] ({count},1,28,28)")
    return array.astype(np.float32, copy=False)


def train_frozen_image_evaluator(train_images: np.ndarray, train_labels: np.ndarray, validation_images: np.ndarray, validation_labels: np.ndarray, *, epochs: int = 8, batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 0xDD1005, device: str | torch.device | None = None, verbose: bool = True) -> tuple[SmallMnistCNN, dict[str, Any]]:
    train_y, val_y = _labels(train_labels, len(train_images)), _labels(validation_labels, len(validation_images))
    train_x = _eval_images(train_images, len(train_y), "train_images")
    val_x = _eval_images(validation_images, len(val_y), "validation_images")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = SmallMnistCNN()
    history = train_image_classifier(
        model, train_x, train_y, val_images=val_x, val_labels=val_y, epochs=epochs,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, device=device, verbose=verbose,
    )
    accuracy = np.asarray(history["val_accuracy"], dtype=np.float64)
    if accuracy.shape != (epochs,) or not np.all(np.isfinite(accuracy)):
        raise RuntimeError("evaluator validation history is incomplete or nonfinite")
    selected = int(np.argmax(accuracy))
    return model, {"history": history, "selected_epoch": selected + 1, "selected_validation_accuracy": float(accuracy[selected]), "selected_validation_loss": float(history["val_loss"][selected])}


def evaluate_generated_labels(model: SmallMnistCNN, images: np.ndarray, requested_labels: np.ndarray, sample_ids: Sequence[str] | np.ndarray, *, batch_size: int = 256, device: str | torch.device | None = None) -> dict[str, Any]:
    images, labels, ids = validate_generated_batch(images, requested_labels, sample_ids)
    result = evaluate_image_classifier(model, uint8_to_unit_interval(images), labels,
                                       batch_size=batch_size, device=device)
    predictions = np.asarray(result["predictions"], dtype=np.int64)
    per_class = {}
    for digit in range(10):
        mask = labels == digit
        per_class[str(digit)] = {"count": int(mask.sum()),
                                 "accuracy": float(np.mean(predictions[mask] == digit)) if mask.any() else None}
    return {"loss": float(result["loss"]), "accuracy": float(result["accuracy"]), "requested_label_accuracy": float(result["accuracy"]), "sample_ids": ids, "requested_labels": labels, "predictions": predictions, "logits": np.asarray(result["logits"], dtype=np.float64), "per_class": per_class}


def _duplicate_summary(images: np.ndarray) -> dict[str, int]:
    _, counts = np.unique(images.reshape(len(images), -1), axis=0, return_counts=True)
    repeats = counts[counts > 1]
    return {"sample_count": len(images), "unique_count": len(counts), "duplicate_count": int(np.sum(repeats - 1)), "duplicate_group_count": len(repeats), "duplicate_pair_count": int(np.sum(repeats * (repeats - 1) // 2))}


def exact_duplicate_metrics(images: np.ndarray, requested_labels: np.ndarray, sample_ids: Sequence[str] | np.ndarray | None = None) -> dict[str, Any]:
    images, labels = _images(images), _labels(requested_labels, len(images))
    if sample_ids is not None:
        _ids(sample_ids, len(images))
    result = _duplicate_summary(images)
    result["total_duplicate_count"] = result["duplicate_count"]
    result["by_class"] = {str(d): _duplicate_summary(images[labels == d])
                          for d in range(10) if np.any(labels == d)}
    return result


def _median_nn(images: np.ndarray) -> float:
    flat = images.reshape(len(images), -1).astype(np.float64) / 255
    distance = np.mean((flat[:, None] - flat[None]) ** 2, axis=2)
    np.fill_diagonal(distance, np.inf)
    return float(np.median(np.min(distance, axis=1)))


def within_class_nn_diversity(generated_images: np.ndarray, generated_labels: np.ndarray, real_images: np.ndarray, real_labels: np.ndarray) -> dict[str, Any]:
    generated, real = _images(generated_images), _images(real_images)
    generated_y, real_y = _labels(generated_labels, len(generated)), _labels(real_labels, len(real))
    per_class, ratios = {}, []
    for digit in sorted(np.unique(generated_y).tolist()):
        gen, ref = generated[generated_y == digit], real[real_y == digit]
        if len(gen) < 2 or len(ref) < len(gen):
            raise ValueError(f"class {digit} lacks a matched diversity reference")
        gen_nn, real_nn = _median_nn(gen), _median_nn(ref[: len(gen)])
        if real_nn <= 0:
            raise ValueError(f"class {digit} has zero real-reference diversity")
        ratio = gen_nn / real_nn
        ratios.append(ratio)
        per_class[str(digit)] = {"count": len(gen), "generated_median_nn_mse": gen_nn,
                                 "real_median_nn_mse": real_nn, "ratio": ratio}
    if not ratios:
        raise ValueError("generated batch is empty")
    return {"aggregate_median_ratio": float(np.median(ratios)), "by_class": per_class}


def _match_count(images: np.ndarray, reference: np.ndarray) -> int:
    keys = {row.tobytes() for row in np.ascontiguousarray(reference).reshape(len(reference), -1)}
    return sum(row.tobytes() in keys for row in np.ascontiguousarray(images).reshape(len(images), -1))


def compute_generation_metrics(model: SmallMnistCNN, images: np.ndarray, requested_labels: np.ndarray, sample_ids: Sequence[str] | np.ndarray, *, real_reference_images: np.ndarray, real_reference_labels: np.ndarray, train_images: np.ndarray | None = None, test_images: np.ndarray | None = None, batch_size: int = 256, device: str | torch.device | None = None) -> dict[str, Any]:
    images, labels, ids = validate_generated_batch(images, requested_labels, sample_ids)
    matches = {}
    for name, reference in (("train", train_images), ("test", test_images)):
        if reference is not None:
            matches[name] = _match_count(images, _images(reference, f"{name}_images"))
    return {"classifier": evaluate_generated_labels(model, images, labels, ids, batch_size=batch_size, device=device), "duplicates": exact_duplicate_metrics(images, labels, ids), "diversity": within_class_nn_diversity(images, labels, real_reference_images, real_reference_labels), "exact_reference_match_count": matches}


def write_contact_sheet(path: str | Path, images: np.ndarray, *, columns: int, scale: int = 4, captions: Sequence[str] | None = None) -> Path:
    images = _images(images)
    if not len(images) or columns <= 0 or scale <= 0:
        raise ValueError("contact sheet, columns, and scale must be nonempty/positive")
    if captions is not None and len(captions) != len(images):
        raise ValueError("captions must match images")
    width, height, caption_height = 28 * scale, 28 * scale, 12 if captions is not None else 0
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("L", (columns * width, rows * (height + caption_height)), 255)
    draw = ImageDraw.Draw(sheet)
    for index, array in enumerate(images):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + caption_height)
        sheet.paste(Image.fromarray(array).resize((width, height), Image.Resampling.NEAREST), (x, y))
        if captions is not None:
            draw.text((x + 1, y + height + 1), str(captions[index]), fill=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")
    return path


render_contact_sheet = write_contact_sheet


def write_blinded_review_bundle(output_dir: str | Path, images: np.ndarray, requested_labels: np.ndarray, sample_ids: Sequence[str] | np.ndarray, *, seed: int = 0xDD4000, columns: int = 8, scale: int = 4) -> dict[str, Any]:
    images, labels, ids = validate_generated_batch(images, requested_labels, sample_ids)
    order = np.random.default_rng(seed).permutation(len(images))
    images, labels, ids = images[order], labels[order], ids[order]
    blind_ids = np.asarray([f"blind-{index:03d}" for index in range(len(images))], dtype=np.str_)
    output_dir, sample_dir = Path(output_dir), Path(output_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        Image.fromarray(image).save(sample_dir / f"sample-{index:03d}.png", format="PNG")
    sheet = write_contact_sheet(output_dir / "blinded-contact-sheet.png", images, columns=columns,
                                scale=scale, captions=blind_ids.tolist())
    template, fields = output_dir / "human_review_template.csv", ["review_order", "sample_id", "assigned_label", "notes"]
    with template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample_id in enumerate(blind_ids):
            writer.writerow({"review_order": index, "sample_id": sample_id, "assigned_label": "", "notes": ""})
    entries = [{"review_order": i, "sample_id": str(blind), "source_sample_id": str(source), "requested_label": int(y)}
               for i, (blind, source, y) in enumerate(zip(blind_ids, ids, labels, strict=True))]
    key = output_dir / "review_key.json"
    key.write_text(json.dumps({"schema": "mnist-blinded-review-v1", "seed": seed,
                               "entries": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "Complete the blinded CSV before opening review_key.json or machine metrics.\n", encoding="utf-8")
    return {"review_order": order, "ordered_sample_ids": ids, "blind_ids": blind_ids, "contact_sheet": sheet, "template": template, "key": key, "sample_directory": sample_dir}


def _read_review_key(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("review key has no entries")
    key = {str(entry.get("sample_id", "")): entry for entry in entries}
    orders = {entry.get("review_order") for entry in entries}
    valid = len(key) == len(entries) == len(orders) and all(
        sample_id and isinstance(entry.get("review_order"), int)
        and isinstance(entry.get("requested_label"), int)
        and 0 <= entry["requested_label"] <= 9 for sample_id, entry in key.items()
    )
    if not valid:
        raise ValueError("review key has invalid or duplicate entries")
    return key


def score_human_review(answers_path: str | Path, review_key_path: str | Path, *, reviewer: str, confirm_manual_review: bool, timestamp: str | None = None) -> dict[str, Any]:
    if confirm_manual_review is not True or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("manual confirmation and a reviewer identity are required")
    key, fields = _read_review_key(review_key_path), ["review_order", "sample_id", "assigned_label", "notes"]
    with Path(answers_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"review columns must be exactly {fields}")
        rows = list(reader)
    if len(rows) != len(key):
        raise ValueError("review must contain every sample exactly once")
    seen_ids, seen_orders, scored = set(), set(), []
    for row in rows:
        sample_id, assignment = row["sample_id"].strip(), row["assigned_label"].strip().lower()
        try:
            order = int(row["review_order"])
        except ValueError as error:
            raise ValueError("review_order must be an integer") from error
        if sample_id in seen_ids or order in seen_orders or sample_id not in key:
            raise ValueError("review contains unknown or duplicate membership")
        entry = key[sample_id]
        if entry["review_order"] != order or assignment not in REVIEW_ASSIGNMENTS:
            raise ValueError("review order or assigned_label is invalid")
        seen_ids.add(sample_id)
        seen_orders.add(order)
        scored.append((entry["requested_label"], assignment))
    if seen_ids != set(key):
        raise ValueError("review is missing samples")
    recognizable = np.array([a.isdigit() for _, a in scored])
    agreement = np.array([a == str(y) for y, a in scored])
    per_class = {}
    for digit in range(10):
        mask = np.array([y == digit for y, _ in scored])
        per_class[str(digit)] = {"count": int(mask.sum()), "recognizable_count": int(recognizable[mask].sum()), "recognizability": float(recognizable[mask].mean()) if mask.any() else None, "agreement_count": int(agreement[mask].sum()), "requested_label_agreement": float(agreement[mask].mean()) if mask.any() else None}
    recorded_at = timestamp or datetime.now(timezone.utc).isoformat()
    if not isinstance(recorded_at, str) or not recorded_at.strip():
        raise ValueError("timestamp must be a nonempty string")
    return {"schema": "mnist-human-review-v1", "reviewer": reviewer.strip(), "recorded_at": recorded_at, "sample_count": len(scored), "recognizable_count": int(recognizable.sum()), "human_recognizability": float(recognizable.mean()), "requested_label_agreement_count": int(agreement.sum()), "human_requested_label_agreement": float(agreement.mean()), "assignment_counts": {a: sum(v == a for _, v in scored) for a in sorted(REVIEW_ASSIGNMENTS)}, "per_class": per_class}
