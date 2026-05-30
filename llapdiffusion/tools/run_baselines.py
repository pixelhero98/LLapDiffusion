from __future__ import annotations

import argparse

from llapdiffusion.benchmark_protocol import DETERMINISTIC_BASELINE_SEEDS, PROBABILISTIC_BASELINE_NUM_SAMPLES
from llapdiffusion.baselines.registry import DATASET_KEYS, EXTRAPOLATION_BASELINES, IMPUTATION_BASELINES, selected
from llapdiffusion.baselines.runner import TrainConfig, export_notes, run_practical_matrix


FULL_NUM_SAMPLES = PROBABILISTIC_BASELINE_NUM_SAMPLES
FULL_EPOCHS = 600
FULL_PATIENCE = 20
COVERAGE_HELP = "fraction of observed context entries to hide; 0 disables induced missingness"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-source-root",
        dest="source_root",
        required=False,
        help="Parent directory containing pinned official baseline checkouts; defaults to LLAPDIFF_BASELINE_SOURCE_ROOT.",
    )
    parser.add_argument("--output-dir", default="baseline_results")
    parser.add_argument("--work-cache-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation/training seed for probabilistic baselines; deterministic baselines use --deterministic-seeds.",
    )
    parser.add_argument(
        "--deterministic-seeds",
        nargs="+",
        type=int,
        default=list(DETERMINISTIC_BASELINE_SEEDS),
        help=(
            "Seeds to train/evaluate and mean for deterministic practical baselines. "
            "The default is 42 through 51."
        ),
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Optional scalar target feature column. Defaults to the dataset cache target_col/target_cols.",
    )
    parser.add_argument(
        "--target-cols",
        nargs="+",
        default=None,
        help="Optional target feature columns for multi-target DLinear/PatchTST runs.",
    )
    parser.add_argument(
        "--input-policy",
        choices=("target_only", "all_features"),
        default="target_only",
        help=(
            "Feature policy for baseline adapters. target_only is the primary comparison; "
            "all_features currently affects only DLinear and PatchTST, while other adapters remain target-only."
        ),
    )
    parser.add_argument(
        "--imputation-random-mask-ratio",
        type=float,
        default=0.30,
        help="Fraction of observed target-horizon entries to hide for CSDI imputation comparisons.",
    )
    parser.add_argument("--coverage", type=float, default=0.0, help=COVERAGE_HELP)
    parser.add_argument("--allow-cache-copy", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch baseline training progress.")


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


def _train_config(args: argparse.Namespace) -> TrainConfig:
    if getattr(args, "target_col", None) and getattr(args, "target_cols", None):
        raise SystemExit("Use either --target-col or --target-cols, not both.")
    coverage = float(getattr(args, "coverage", 0.0))
    if not 0.0 <= coverage < 1.0:
        raise SystemExit("--coverage must satisfy 0 <= coverage < 1.")
    target_cols = tuple(args.target_cols) if getattr(args, "target_cols", None) else None
    deterministic_seeds = tuple(int(seed) for seed in getattr(args, "deterministic_seeds", ()))
    if not deterministic_seeds:
        raise SystemExit("--deterministic-seeds must contain at least one seed.")
    return TrainConfig(
        source_root=args.source_root,
        output_dir=args.output_dir,
        work_cache_dir=args.work_cache_dir,
        device=args.device,
        seed=args.seed,
        num_samples=FULL_NUM_SAMPLES,
        deterministic_seeds=deterministic_seeds,
        imputation_random_mask_ratio=args.imputation_random_mask_ratio,
        allow_cache_copy=args.allow_cache_copy,
        epochs=FULL_EPOCHS,
        patience=FULL_PATIENCE,
        lr=args.lr,
        horizons=_parse_horizons(getattr(args, "horizons", None)) or "all",
        input_policy=args.input_policy,
        target_col=args.target_col,
        target_cols=target_cols,
        coverage=coverage,
        verbose=bool(getattr(args, "verbose", False)),
    )


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


def _add_horizon_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=None,
        help="Horizon selection: omit or use 'all' for every supported horizon, or list explicit supported horizons.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLapDiffusion baseline adapters on full-data comparison scopes.")
    sub = parser.add_subparsers(dest="command", required=True)

    practical = sub.add_parser("practical-extrapolation", help="Run full-data early-stop extrapolation baselines.")
    _add_common(practical)
    practical.add_argument("--baseline", choices=EXTRAPOLATION_BASELINES + ("all",), default="all")
    practical.add_argument("--dataset", choices=DATASET_KEYS + ("all",), default="all")
    _add_horizon_arg(practical)
    practical.add_argument("--lr", type=float, default=1e-3)
    practical.set_defaults(func=_run_practical_extrapolation)

    csdi = sub.add_parser("csdi-imputation", help="Run full-data CSDI target-horizon imputation baselines.")
    _add_common(csdi)
    csdi.add_argument("--dataset", choices=DATASET_KEYS + ("all",), default="all")
    _add_horizon_arg(csdi)
    csdi.add_argument("--lr", type=float, default=1e-3)
    csdi.set_defaults(func=_run_csdi)

    notes = sub.add_parser("export-notes", help="Write baseline metadata and caveats.")
    notes.add_argument("--output-dir", default="baseline_results")
    notes.set_defaults(func=_run_export_notes)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "target_col", None) and getattr(args, "target_cols", None):
        raise SystemExit("Use either --target-col or --target-cols, not both.")
    args.func(args)


if __name__ == "__main__":
    main()
