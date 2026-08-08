"""Run the fixed-cache D0-v1 one-image potential-gradient milestone."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .d0_v0_density_ratio import (
    build_smoke_cache,
    load_cache,
    save_cache,
    write_metrics_csv,
)
from .d0_v1_potential_gradient import (
    D0V1Config,
    acceptance_summary,
    load_best_model,
    run_v1_paired_sampling,
    save_contact_sheet,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("cache", "train", "sample", "all"), default="all"
    )
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/d0_v1"))
    parser.add_argument("--run-name", default="one-image-potential-gradient")
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def _run_dir(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        if not args.resume_run_dir.is_dir():
            raise FileNotFoundError(args.resume_run_dir)
        return args.resume_run_dir
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.runs_root / f"{stamp}_{args.run_name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = D0V1Config.smoke() if args.smoke else D0V1Config()
    device = torch.device(args.device)
    run_dir = _run_dir(args)
    config_path = run_dir / "config.json"
    if config_path.is_file():
        stored = D0V1Config(**json.loads(config_path.read_text(encoding="utf-8")))
        if stored != config:
            raise ValueError("resume configuration does not match this D0-v1 mode")
    else:
        config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    stages = ("cache", "train", "sample") if args.stage == "all" else (args.stage,)
    local_cache_path = run_dir / "cache.npz"
    if "cache" in stages:
        if args.smoke:
            cache = build_smoke_cache(config)
        else:
            if args.cache_path is None:
                raise ValueError("--cache-path is required outside smoke mode")
            cache = load_cache(args.cache_path)
            (run_dir / "source_cache.json").write_text(
                json.dumps({"path": str(args.cache_path.resolve())}, indent=2),
                encoding="utf-8",
            )
        save_cache(local_cache_path, cache)
    else:
        cache = load_cache(local_cache_path)

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
            validations = json.loads(
                (run_dir / "validation.json").read_text(encoding="utf-8")
            )
            best = min(
                validations,
                key=lambda row: (row["validation_gradient_loss"], row["step"]),
            )
            training_result = {
                "best_step": best["step"],
                "best_validation_gradient_loss": best["validation_gradient_loss"],
            }
        model = load_best_model(run_dir, config, device)
        samples, rows, sampling_result = run_v1_paired_sampling(
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
