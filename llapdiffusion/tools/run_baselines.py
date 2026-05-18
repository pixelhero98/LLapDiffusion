from __future__ import annotations

import argparse
from pathlib import Path

from llapdiffusion.baselines.registry import BASELINES, DATASET_KEYS, EXTRAPOLATION_BASELINES, IMPUTATION_BASELINES, selected
from llapdiffusion.baselines.runner import (
    SmokeConfig,
    TrainConfig,
    export_notes,
    run_baseline_matrix,
    run_practical_matrix,
    write_isambard_jobs,
    write_rows,
)
from llapdiffusion.baselines.sources import SourceManager


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-source-root",
        dest="source_root",
        required=False,
        help="Parent directory containing pinned official baseline checkouts; defaults to LLAPDIFF_BASELINE_SOURCE_ROOT.",
    )
    parser.add_argument("--output-dir", default="baseline_results")
    parser.add_argument("--work-cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-entities", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--allow-cache-copy", action="store_true")


def _smoke_config(args: argparse.Namespace) -> SmokeConfig:
    return SmokeConfig(
        source_root=args.source_root,
        output_dir=args.output_dir,
        work_cache_dir=args.work_cache_dir,
        device=args.device,
        max_entities=args.max_entities,
        max_batches=args.max_batches,
        seed=args.seed,
        num_samples=args.num_samples,
        allow_cache_copy=args.allow_cache_copy,
        keep_going=not args.fail_fast,
    )


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        source_root=args.source_root,
        output_dir=args.output_dir,
        work_cache_dir=args.work_cache_dir,
        device=args.device,
        max_entities=args.max_entities,
        max_batches=args.max_batches,
        seed=args.seed,
        num_samples=args.num_samples,
        allow_cache_copy=args.allow_cache_copy,
        keep_going=True,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )


def _validate_sources(args: argparse.Namespace) -> None:
    manager = SourceManager(args.source_root)
    for key in sorted(BASELINES):
        manager.validate(BASELINES[key])
        print(f"OK source {key}", flush=True)


def _run_smoke(args: argparse.Namespace) -> None:
    if args.validate_sources_only:
        _validate_sources(args)
        return
    baselines = selected(tuple(BASELINES), args.baseline)
    datasets = selected(DATASET_KEYS, args.dataset)
    rows = run_baseline_matrix(baselines, datasets, _smoke_config(args))
    write_rows(rows, args.output_dir, prefix="baseline_pool_smoke")
    failures = [r for r in rows if r.get("status") != "ok"]
    if failures:
        raise SystemExit(f"{len(failures)} baseline smoke checks failed")


def _run_practical_extrapolation(args: argparse.Namespace) -> None:
    baselines = selected(EXTRAPOLATION_BASELINES, args.baseline)
    datasets = selected(DATASET_KEYS, args.dataset)
    run_practical_matrix(baselines, datasets, _train_config(args), args.output_dir)


def _run_csdi(args: argparse.Namespace) -> None:
    datasets = ["physionet", "crypto", "noaa_uk"] if args.dataset == "all" else [args.dataset]
    run_practical_matrix(IMPUTATION_BASELINES, datasets, _train_config(args), args.output_dir)


def _run_export_notes(args: argparse.Namespace) -> None:
    path = export_notes(args.output_dir)
    print(path, flush=True)


def _run_write_jobs(args: argparse.Namespace) -> None:
    baselines = selected(EXTRAPOLATION_BASELINES, args.baseline)
    datasets = selected(DATASET_KEYS, args.dataset)
    scripts = write_isambard_jobs(
        Path(args.output_dir),
        source_root=args.source_root,
        datasets=datasets,
        baselines=baselines,
        time_limit=args.time_limit,
    )
    for script in scripts:
        print(script, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLapDiffusion external baseline adapters.")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Run one-batch source/import/forward/loss/backward checks.")
    _add_common(smoke)
    smoke.add_argument("--baseline", choices=tuple(BASELINES) + ("all",), default="all")
    smoke.add_argument("--dataset", choices=DATASET_KEYS + ("all",), default="all")
    smoke.add_argument("--fail-fast", action="store_true")
    smoke.add_argument("--validate-sources-only", action="store_true")
    smoke.set_defaults(func=_run_smoke)

    practical = sub.add_parser("practical-extrapolation", help="Run bounded early-stop extrapolation baselines.")
    _add_common(practical)
    practical.add_argument("--baseline", choices=EXTRAPOLATION_BASELINES + ("all",), default="all")
    practical.add_argument("--dataset", choices=DATASET_KEYS + ("all",), default="all")
    practical.add_argument("--epochs", type=int, default=50)
    practical.add_argument("--patience", type=int, default=8)
    practical.add_argument("--lr", type=float, default=1e-3)
    practical.add_argument("--max-train-batches", type=int, default=256)
    practical.add_argument("--max-eval-batches", type=int, default=256)
    practical.set_defaults(func=_run_practical_extrapolation)

    csdi = sub.add_parser("csdi-imputation", help="Run bounded CSDI context-imputation baselines.")
    _add_common(csdi)
    csdi.add_argument("--dataset", choices=("physionet", "crypto", "noaa_uk", "all"), default="all")
    csdi.add_argument("--epochs", type=int, default=50)
    csdi.add_argument("--patience", type=int, default=8)
    csdi.add_argument("--lr", type=float, default=1e-3)
    csdi.add_argument("--max-train-batches", type=int, default=256)
    csdi.add_argument("--max-eval-batches", type=int, default=256)
    csdi.set_defaults(func=_run_csdi)

    notes = sub.add_parser("export-notes", help="Write baseline metadata and caveats.")
    notes.add_argument("--output-dir", default="baseline_results")
    notes.set_defaults(func=_run_export_notes)

    jobs = sub.add_parser("write-isambard-jobs", help="Write Slurm scripts for longest-horizon extrapolation baselines.")
    jobs.add_argument("--baseline-source-root", dest="source_root", required=True)
    jobs.add_argument("--output-dir", default="isambard_jobs")
    jobs.add_argument("--baseline", choices=EXTRAPOLATION_BASELINES + ("all",), default="all")
    jobs.add_argument("--dataset", choices=DATASET_KEYS + ("all",), default="all")
    jobs.add_argument("--time-limit", default="08:00:00")
    jobs.set_defaults(func=_run_write_jobs)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
