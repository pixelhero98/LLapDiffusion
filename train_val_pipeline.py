"""Train latent VAE, summarizer, and LLapDiff in one pipeline."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from configs import config
from configs.config_utils import make_jsonable
from configs.dataset_archives import configure_dataset_archive
from configs.dataset_defaults import apply_dataset_preset, dataset_keys, default_horizons, infer_dataset_key
from configs.dataset_registry import resolve_run_experiment
from trainers import train_val_latent, train_val_summarizer, train_val_llapdiff


def _import_trainers():
    """Return the trainer modules exposed through the public package layout."""
    return train_val_latent, train_val_summarizer, train_val_llapdiff


def prepare_dataloaders(
    config=config,
) -> Tuple[Any, Any, Any, Tuple[int, int, int]]:
    """Build train/val/test loaders using the shared configuration."""

    run_experiment = resolve_run_experiment(config.DATA_DIR)
    return run_experiment(
        data_dir=config.DATA_DIR,
        date_batching=config.date_batching,
        dates_per_batch=int(config.DATES_PER_BATCH),
        K=config.WINDOW,
        H=config.PRED,
        coverage=config.COVERAGE,
        ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
    )


def _fmt_optional(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    if value is None:
        return "None"
    return str(value)


def _summarizer_ckpt_path(config=config) -> Path:
    ckpt = config.SUM_CKPT
    if ckpt:
        return Path(ckpt)
    return (
        Path(config.SUM_DIR)
        / f"{config.PRED}-{config.VAE_LATENT_CHANNELS}-summarizer.pt"
    )


def _select_vae_checkpoint(stats: Dict[str, object], fallback: Path) -> Path:
    for key in ("loaded_checkpoint", "best_elbo_path", "checkpoint", "best_recon_path"):
        value = stats.get(key)
        if value not in (None, ""):
            return Path(str(value))
    return fallback


def _resolve_sum_context_len(pred: int, *, config=config) -> int:
    fixed = getattr(config, "SUM_CONTEXT_LEN_FIXED", None)
    if fixed not in {None, "", False}:
        return int(fixed)
    return int(pred)


def _resolve_dataset_key(config=config) -> str:
    dataset_key = str(getattr(config, "DATASET_KEY", "") or "").strip().lower()
    if dataset_key:
        return dataset_key
    data_dir = str(getattr(config, "DATA_DIR", "") or "").strip()
    if data_dir:
        return infer_dataset_key(data_dir)
    raise ValueError("DATASET_KEY is required for preset-driven pipeline runs.")


def _update_config_for_pred(pred: int, config=config) -> None:
    apply_dataset_preset(config, _resolve_dataset_key(config=config), pred=int(pred))
    config.SUM_CONTEXT_LEN = _resolve_sum_context_len(pred, config=config)
    config.SUM_CKPT = str(
        Path(config.SUM_DIR) / f"{pred}-{config.VAE_LATENT_CHANNELS}-summarizer.pt"
    )


def _apply_pred_output_dirs(
    pred: int, *, base_out_dir: Path, base_ckpt_dir: Path, config=config
) -> None:
    """Route per-horizon outputs to separate directories to avoid checkpoint overwrites."""
    pred_out = base_out_dir / f"pred-{pred}"
    pred_ckpt = base_ckpt_dir / f"pred-{pred}"
    config.OUT_DIR = str(pred_out)
    if hasattr(config, "CKPT_DIR"):
        config.CKPT_DIR = str(pred_ckpt)
    if hasattr(config, "POLE_PLOT_DIR"):
        config.POLE_PLOT_DIR = str(pred_out / "pole_plots")


def _extract_llapdiff_reporting_fields(
    llapdiff_stats: Dict[str, object],
) -> Tuple[Dict[str, object], object, object]:
    """Normalize return payloads from different train_val_llapdiff.run versions."""
    eval_stats = llapdiff_stats.get("eval_stats")
    if not isinstance(eval_stats, dict):
        alt_eval = llapdiff_stats.get("test_metrics")
        eval_stats = alt_eval if isinstance(alt_eval, dict) else {}

    best_val = llapdiff_stats.get("best_val")
    if best_val is None:
        val_history = llapdiff_stats.get("val_history")
        if isinstance(val_history, list) and val_history:
            crps_vals = [
                row.get("crps")
                for row in val_history
                if isinstance(row, dict) and row.get("crps") is not None
            ]
            if crps_vals:
                best_val = min(crps_vals)

    loaded_checkpoint = llapdiff_stats.get("loaded_checkpoint")
    if loaded_checkpoint is None:
        loaded_checkpoint = llapdiff_stats.get("best_checkpoint") or llapdiff_stats.get(
            "last_checkpoint"
        )

    return eval_stats, best_val, loaded_checkpoint


def run_single_pred(
    pred: int,
    *,
    recompute_vae: bool = False,
    recompute_summarizer: bool = False,
    latent_plot_only: bool = False,
    use_shared_loaders: bool = True,
    base_out_dir: Path | None = None,
    base_ckpt_dir: Path | None = None,
    config=config,
) -> Dict[str, object]:
    """
    Train/evaluate the full LLapDiff pipeline for one prediction horizon.

    Returns a dictionary with the combined stats from each stage.
    """
    _update_config_for_pred(pred, config=config)
    if base_out_dir is not None and base_ckpt_dir is not None:
        _apply_pred_output_dirs(
            pred,
            base_out_dir=base_out_dir,
            base_ckpt_dir=base_ckpt_dir,
            config=config,
        )

    train_val_latent, train_val_summarizer, train_val_llapdiff = _import_trainers()

    train_dl = val_dl = test_dl = None
    sizes = None
    if use_shared_loaders:
        train_dl, val_dl, test_dl, sizes = prepare_dataloaders(config=config)

    vae_ckpt_path = Path(config.VAE_CKPT)
    if recompute_vae or not vae_ckpt_path.exists():
        latent_stats = train_val_latent.run(
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            sizes=sizes,
            plot_only=latent_plot_only,
            config=config,
        )
    else:
        latent_stats = {
            "status": "skipped",
            "reason": "checkpoint_exists",
            "checkpoint": str(vae_ckpt_path),
        }
    config.VAE_CKPT = str(_select_vae_checkpoint(latent_stats, vae_ckpt_path))

    summ_ckpt_path = _summarizer_ckpt_path(config=config)
    if recompute_summarizer or not summ_ckpt_path.exists():
        summarizer_stats = train_val_summarizer.run(
            train_loader=train_dl,
            val_loader=val_dl,
            test_loader=test_dl,
            sizes=sizes,
            config=config,
        )
    else:
        summarizer_stats = {
            "status": "skipped",
            "reason": "checkpoint_exists",
            "checkpoint": str(summ_ckpt_path),
        }

    llapdiff_stats = train_val_llapdiff.run(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        sizes=sizes,
        config=config,
    )

    eval_stats, best_val, loaded_checkpoint = _extract_llapdiff_reporting_fields(
        llapdiff_stats
    )
    balanced_evaluation = None
    eval_ckpt = llapdiff_stats.get("best_checkpoint_raw") or llapdiff_stats.get("best_checkpoint")
    if eval_ckpt:
        try:
            from tools.llapdiff_checkpoint_eval import evaluate_checkpoint

            balanced_evaluation = evaluate_checkpoint(
                config,
                eval_ckpt,
                label=f"{_resolve_dataset_key(config=config)}_pred{pred}",
            )
        except Exception as exc:
            balanced_evaluation = {
                "label": f"{_resolve_dataset_key(config=config)}_pred{pred}",
                "checkpoint": str(eval_ckpt),
                "status": "fail",
                "error": str(exc),
            }

    return {
        "pred": pred,
        "vae": latent_stats,
        "summarizer": summarizer_stats,
        "llapdiff": llapdiff_stats,
        "eval_stats": eval_stats,
        "balanced_evaluation": balanced_evaluation,
        "best_val": best_val,
        "loaded_checkpoint": loaded_checkpoint,
    }


def run_preds(
    preds: Iterable[int],
    *,
    recompute_vae: bool = False,
    recompute_summarizer: bool = False,
    latent_plot_only: bool = False,
    use_shared_loaders: bool = True,
    config=config,
) -> Dict[int, Dict[str, object]]:
    """
    Run the pipeline for multiple prediction horizons and collect stats.
    """
    base_out_dir = Path(getattr(config, "OUT_DIR", "./outputs"))
    base_ckpt_dir = Path(getattr(config, "CKPT_DIR", str(base_out_dir / "checkpoints")))

    results: Dict[int, Dict[str, object]] = {}
    for pred in preds:
        print(f"\n=== Running pipeline for pred={pred} ===")
        results[int(pred)] = run_single_pred(
            int(pred),
            recompute_vae=recompute_vae,
            recompute_summarizer=recompute_summarizer,
            latent_plot_only=latent_plot_only,
            use_shared_loaders=use_shared_loaders,
            base_out_dir=base_out_dir,
            base_ckpt_dir=base_ckpt_dir,
            config=config,
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train latent VAE, summarizer, and LLapDiff in one pipeline."
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        choices=dataset_keys(),
        required=True,
        help="Dataset preset key to run.",
    )
    parser.add_argument(
        "--preds",
        type=int,
        nargs="+",
        default=None,
        help="Prediction horizons to run. Defaults to config.PIPELINE_PREDS or [config.PRED].",
    )
    parser.add_argument(
        "--recompute-vae",
        action="store_true",
        help="Force retraining the latent VAE even if a checkpoint already exists.",
    )
    parser.add_argument(
        "--recompute-summarizer",
        action="store_true",
        help="Force retraining the summarizer even if a checkpoint already exists.",
    )
    parser.add_argument(
        "--latent-plot-only",
        action="store_true",
        help="Pass plot_only=True to the latent VAE trainer.",
    )
    parser.add_argument(
        "--no-shared-loaders",
        action="store_true",
        help="Let each stage build its own dataloaders instead of sharing them.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path to save a compact JSON summary of pipeline results.",
    )
    parser.add_argument(
        "--dataset-zip",
        type=str,
        default=None,
        help="Optional zipped dataset cache. Required when the preset cache directory is absent.",
    )
    parser.add_argument(
        "--dataset-extract-dir",
        type=str,
        default=None,
        help="Optional directory for extracting --dataset-zip. Defaults to the repository Dataset directory.",
    )
    return parser.parse_args()


def _pred_list_from_config(config=config) -> Tuple[int, ...]:
    preds = getattr(config, "PIPELINE_PREDS", None)
    if preds is None:
        dataset_key = _resolve_dataset_key(config=config)
        preds_tuple = tuple(int(p) for p in default_horizons(dataset_key))
    elif isinstance(preds, (list, tuple, set)):
        preds_tuple = tuple(int(p) for p in preds)
    else:
        preds_tuple = (int(preds),)

    # Preserve order while removing duplicates and invalid values.
    seen = set()
    cleaned = []
    for pred in preds_tuple:
        if pred <= 0:
            raise ValueError(f"Prediction horizons must be positive integers, got {pred}.")
        if pred not in seen:
            seen.add(pred)
            cleaned.append(pred)

    if not cleaned:
        raise ValueError("No valid prediction horizons found in configuration.")

    return tuple(cleaned)


def _json_safe(value: Any) -> Any:
    return make_jsonable(value)


def _save_summary_json(path: str, results: Dict[int, Dict[str, object]]) -> None:
    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_key": _resolve_dataset_key(config=config),
        "results": {str(pred): _json_safe(stats) for pred, stats in results.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved pipeline summary to {output_path}")


def _print_summary_table(results: Dict[int, Dict[str, object]]) -> None:
    header = (
        "pred",
        "vae",
        "summarizer",
        "llapdiff_ckpt",
        "best_val",
        "crps",
        "mae",
        "mse",
    )
    rows = [header]
    for pred in sorted(results):
        result = results[pred]
        vae_stats = result.get("vae", {})
        sum_stats = result.get("summarizer", {})
        eval_stats = result.get("eval_stats", {})
        balanced_eval = result.get("balanced_evaluation", {})
        forecast_stats = (
            balanced_eval.get("forecast_test", {})
            if isinstance(balanced_eval, dict)
            else {}
        )
        llapdiff_ckpt = result.get("loaded_checkpoint")

        def stage_desc(stage_stats: Any) -> str:
            if not isinstance(stage_stats, dict):
                return str(stage_stats)
            if stage_stats.get("status") == "skipped":
                return f"skipped ({stage_stats.get('reason', 'unknown')})"
            if "checkpoint" in stage_stats:
                return "trained"
            if "status" in stage_stats:
                return str(stage_stats["status"])
            return "done"

        rows.append(
            (
                str(pred),
                stage_desc(vae_stats),
                stage_desc(sum_stats),
                str(llapdiff_ckpt) if llapdiff_ckpt is not None else "None",
                _fmt_optional(result.get("best_val")),
                _fmt_optional(
                    forecast_stats.get("crps")
                    if isinstance(forecast_stats, dict) and forecast_stats
                    else (eval_stats.get("crps") if isinstance(eval_stats, dict) else None)
                ),
                _fmt_optional(
                    forecast_stats.get("mae")
                    if isinstance(forecast_stats, dict) and forecast_stats
                    else (eval_stats.get("mae") if isinstance(eval_stats, dict) else None)
                ),
                _fmt_optional(
                    forecast_stats.get("mse")
                    if isinstance(forecast_stats, dict) and forecast_stats
                    else (eval_stats.get("mse") if isinstance(eval_stats, dict) else None)
                ),
            )
        )

    widths = [max(len(row[idx]) for row in rows) for idx in range(len(header))]
    print("\nPipeline summary:")
    for row_idx, row in enumerate(rows):
        line = " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        print(line)
        if row_idx == 0:
            print("-+-".join("-" * w for w in widths))


def main() -> Dict[int, Dict[str, object]]:
    args = _parse_args()
    configure_dataset_archive(args.dataset_zip, args.dataset_extract_dir)
    apply_dataset_preset(config, args.dataset_key)
    preds = tuple(args.preds) if args.preds else _pred_list_from_config(config=config)
    if not preds:
        raise ValueError("No prediction horizons provided to the pipeline.")

    base_out_dir = Path(getattr(config, "OUT_DIR", "./outputs"))
    base_ckpt_dir = Path(getattr(config, "CKPT_DIR", str(base_out_dir / "checkpoints")))

    results: Dict[int, Dict[str, object]] = {}
    for pred in preds:
        results[int(pred)] = run_single_pred(
            int(pred),
            recompute_vae=args.recompute_vae,
            recompute_summarizer=args.recompute_summarizer,
            latent_plot_only=args.latent_plot_only,
            use_shared_loaders=not args.no_shared_loaders,
            base_out_dir=base_out_dir,
            base_ckpt_dir=base_ckpt_dir,
            config=config,
        )

    _print_summary_table(results)
    if args.summary_json:
        _save_summary_json(args.summary_json, results)
    return results


if __name__ == "__main__":  # pragma: no cover
    main()
