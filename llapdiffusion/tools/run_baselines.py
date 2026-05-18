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


FULL_NUM_SAMPLES = 25
FULL_EPOCHS = 600
FULL_PATIENCE = 50
QUICK_MAX_ENTITIES = 4
QUICK_MAX_BATCHES = 30
QUICK_MAX_TRAIN_BATCHES = 256
QUICK_MAX_EVAL_BATCHES = 256
QUICK_EPOCHS = 50
QUICK_PATIENCE = 8
QUICK_NUM_SAMPLES = 4


def _add_common(
    parser: argparse.ArgumentParser,
    *,
    max_entities_default: int | None = 4,
    max_batches_default: int | None = 30,
    num_samples_default: int | None = QUICK_NUM_SAMPLES,
) -> None:
    parser.add_argument(
        "--baseline-source-root",
        dest="source_root",
        required=False,
        help="Parent directory containing pinned official baseline checkouts; defaults to LLAPDIFF_BASELINE_SOURCE_ROOT.",
    )
    parser.add_argument("--output-dir", default="baseline_results")
    parser.add_argument("--work-cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-entities", type=int, default=max_entities_default, help="Entity cap; use 0 for full panel.")
    parser.add_argument("--max-batches", type=int, default=max_batches_default, help="Valid-batch scan cap; use 0 for no cap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=num_samples_default)
    parser.add_argument(
        "--imputation-random-mask-ratio",
        type=float,
        default=0.30,
        help="Fraction of observed entries to hide for random-mask imputation comparisons.",
    )
    parser.add_argument("--allow-cache-copy", action="store_true")


def _parse_horizons(values: list[str] | None) -> tuple[int, ...] | str | None:
    if values is None:
        return None
    if len(values) == 1 and values[0].lower() == "all":
        return "all"
    if any(value.lower() == "all" for value in values):
        raise SystemExit("--horizons accepts either 'all' or explicit integer horizons, not both.")
    try:
        return tuple(int(value) for value in values)
    except ValueError as exc:
        raise SystemExit("--horizons values must be integers or 'all'.") from exc


def _normalize_cap(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise SystemExit(f"--{name.replace('_', '-')} must be non-negative; use 0 for no cap.")
    return None if value == 0 else value


def _positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    return value


def _quick_default(value, *, quick: bool, full, quick_value):
    if value is not None:
        return value
    return quick_value if quick else full


def _smoke_config(args: argparse.Namespace) -> SmokeConfig:
    return SmokeConfig(
        source_root=args.source_root,
        output_dir=args.output_dir,
        work_cache_dir=args.work_cache_dir,
        device=args.device,
        max_entities=_normalize_cap(args.max_entities, "max_entities"),
        max_batches=_normalize_cap(args.max_batches, "max_batches"),
        seed=args.seed,
        num_samples=_positive_int(args.num_samples, "num_samples") or QUICK_NUM_SAMPLES,
        imputation_random_mask_ratio=args.imputation_random_mask_ratio,
        csdi_imputation_target=getattr(args, "csdi_imputation_target", "target"),
        allow_cache_copy=args.allow_cache_copy,
        keep_going=not args.fail_fast,
    )


def _train_config(args: argparse.Namespace) -> TrainConfig:
    quick = bool(getattr(args, "quick", False))
    return TrainConfig(
        source_root=args.source_root,
        output_dir=args.output_dir,
        work_cache_dir=args.work_cache_dir,
        device=args.device,
        max_entities=_normalize_cap(
            _quick_default(args.max_entities, quick=quick, full=None, quick_value=QUICK_MAX_ENTITIES),
            "max_entities",
        ),
        max_batches=_normalize_cap(
            _quick_default(args.max_batches, quick=quick, full=None, quick_value=QUICK_MAX_BATCHES),
            "max_batches",
        ),
        seed=args.seed,
        num_samples=_positive_int(
            _quick_default(args.num_samples, quick=quick, full=FULL_NUM_SAMPLES, quick_value=QUICK_NUM_SAMPLES),
            "num_samples",
        )
        or FULL_NUM_SAMPLES,
        imputation_random_mask_ratio=args.imputation_random_mask_ratio,
        allow_cache_copy=args.allow_cache_copy,
        keep_going=True,
        epochs=_positive_int(_quick_default(args.epochs, quick=quick, full=FULL_EPOCHS, quick_value=QUICK_EPOCHS), "epochs")
        or FULL_EPOCHS,
        patience=_positive_int(
            _quick_default(args.patience, quick=quick, full=FULL_PATIENCE, quick_value=QUICK_PATIENCE),
            "patience",
        )
        or FULL_PATIENCE,
        lr=args.lr,
        max_train_batches=_normalize_cap(
            _quick_default(args.max_train_batches, quick=quick, full=None, quick_value=QUICK_MAX_TRAIN_BATCHES),
            "max_train_batches",
        ),
        max_eval_batches=_normalize_cap(
            _quick_default(args.max_eval_batches, quick=quick, full=None, quick_value=QUICK_MAX_EVAL_BATCHES),
            "max_eval_batches",
        ),
        horizons=_parse_horizons(getattr(args, "horizons", None)) or "all",
        csdi_imputation_target=getattr(args, "csdi_imputation_target", "target"),
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
    datasets = selected(DATASET_KEYS, args.dataset)
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
        horizons=_parse_horizons(args.horizons),
        quick=args.quick,
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

    practical = sub.add_parser("practical-extrapolation", help="Run full-data early-stop extrapolation baselines.")
    _add_common(practical, max_entities_default=None, max_batches_default=None, num_samples_default=None)
    practical.add_argument("--baseline", choices=EXTRAPOLATION_BASELINES + ("all",), default="all")
    practical.add_argument("--dataset", choices=DATASET_KEYS + ("all",), default="all")
    practical.add_argument("--quick", action="store_true", help="Use legacy bounded debug caps and short training budget.")
    practical.add_argument(
        "--horizons",
        nargs="+",
        default=None,
        help="Horizon selection: omit or use 'all' for every supported horizon, or list explicit supported horizons.",
    )
    practical.add_argument("--epochs", type=int, default=None)
    practical.add_argument("--patience", type=int, default=None)
    practical.add_argument("--lr", type=float, default=1e-3)
    practical.add_argument("--max-train-batches", type=int, default=None)
    practical.add_argument("--max-eval-batches", type=int, default=None)
    practical.set_defaults(func=_run_practical_extrapolation)

    csdi = sub.add_parser("csdi-imputation", help="Run full-data CSDI target-horizon imputation baselines.")
    _add_common(csdi, max_entities_default=None, max_batches_default=None, num_samples_default=None)
    csdi.add_argument("--dataset", choices=DATASET_KEYS + ("all",), default="all")
    csdi.add_argument("--quick", action="store_true", help="Use legacy bounded debug caps and short training budget.")
    csdi.add_argument("--csdi-imputation-target", choices=("target", "context"), default="target")
    csdi.add_argument(
        "--horizons",
        nargs="+",
        default=None,
        help="Horizon selection: omit or use 'all' for every supported horizon, or list explicit supported horizons.",
    )
    csdi.add_argument("--epochs", type=int, default=None)
    csdi.add_argument("--patience", type=int, default=None)
    csdi.add_argument("--lr", type=float, default=1e-3)
    csdi.add_argument("--max-train-batches", type=int, default=None)
    csdi.add_argument("--max-eval-batches", type=int, default=None)
    csdi.set_defaults(func=_run_csdi)

    notes = sub.add_parser("export-notes", help="Write baseline metadata and caveats.")
    notes.add_argument("--output-dir", default="baseline_results")
    notes.set_defaults(func=_run_export_notes)

    jobs = sub.add_parser("write-isambard-jobs", help="Write Slurm scripts for practical extrapolation baselines.")
    jobs.add_argument("--baseline-source-root", dest="source_root", required=False)
    jobs.add_argument("--output-dir", default="isambard_jobs")
    jobs.add_argument("--baseline", choices=EXTRAPOLATION_BASELINES + ("all",), default="all")
    jobs.add_argument("--dataset", choices=DATASET_KEYS + ("all",), default="all")
    jobs.add_argument(
        "--horizons",
        nargs="+",
        default=None,
        help="Horizon selection: omit or use 'all' for every supported horizon, or list explicit supported horizons.",
    )
    jobs.add_argument("--quick", action="store_true", help="Write bounded debug job scripts instead of full-data scripts.")
    jobs.add_argument("--time-limit", default="08:00:00")
    jobs.set_defaults(func=_run_write_jobs)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
