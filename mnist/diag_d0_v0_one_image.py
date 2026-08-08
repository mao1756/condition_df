"""Run the deliberately small D0-v0 one-image milestone."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from .d0_v0_density_ratio import (
    D0V0Config,
    acceptance_summary,
    build_cache,
    build_smoke_cache,
    load_best_model,
    load_cache,
    run_paired_sampling,
    save_cache,
    save_contact_sheet,
    train,
    write_metrics_csv,
)
from .eulerian_flux_mnist import load_mnist_measure_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache", "train", "sample", "all"), default="all")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/d0_v0"))
    parser.add_argument("--run-name", default="one-image-density-ratio")
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _run_dir(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        path = args.resume_run_dir
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.runs_root / f"{stamp}_{args.run_name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _load_one_image(args: argparse.Namespace, config: D0V0Config) -> np.ndarray:
    dataset = load_mnist_measure_dataset(
        args.data_root,
        examples_per_class=1,
        download=args.download,
        seed=config.seed,
    )
    indices = np.flatnonzero(dataset.train_labels == config.label)
    if not len(indices):
        raise ValueError(f"MNIST has no label {config.label}")
    return dataset.train_images[int(indices[0])]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = D0V0Config.smoke() if args.smoke else D0V0Config()
    device = torch.device(args.device)
    run_dir = _run_dir(args)
    config_path = run_dir / "config.json"
    if config_path.is_file():
        stored = D0V0Config(**json.loads(config_path.read_text(encoding="utf-8")))
        if stored != config:
            raise ValueError("resume configuration does not match this D0-v0 mode")
    else:
        config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    stages = ("cache", "train", "sample") if args.stage == "all" else (args.stage,)
    cache_path = run_dir / "cache.npz"
    if "cache" in stages:
        cache = (
            build_smoke_cache(config)
            if args.smoke
            else build_cache(
                _load_one_image(args, config),
                config,
                device=device,
                show_progress=not args.no_progress,
            )
        )
        save_cache(cache_path, cache)
    else:
        cache = load_cache(cache_path)

    training_result: dict[str, object] | None = None
    if "train" in stages:
        training_result = train(
            cache,
            config,
            run_dir,
            device=device,
            show_progress=not args.no_progress,
        )

    if "sample" in stages:
        if training_result is None:
            validations = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
            best = min(validations, key=lambda row: (row["validation_bce"], row["step"]))
            training_result = {
                "best_step": best["step"],
                "best_validation_bce": best["validation_bce"],
                "zero_baseline_bce": best["zero_baseline_bce"],
            }
        model = load_best_model(run_dir, config, device)
        samples, rows, sampling_result = run_paired_sampling(
            model, cache, config, device=device
        )
        np.savez_compressed(run_dir / "paired_samples.npz", **samples)
        write_metrics_csv(run_dir / "paired_metrics.csv", rows)
        save_contact_sheet(run_dir / "paired_contact_sheet.png", samples, config.grid_size)
        summary = acceptance_summary(training_result, sampling_result)
        if args.smoke:
            summary["checks"]["eight_samples"] = sampling_result["num_samples"] == 1
            summary["status"] = "smoke_complete"
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    else:
        print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
