"""Train latent VAE, summarizer, and LLapDiff in one pipeline."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from llapdiffusion.configs import config
from llapdiffusion.benchmark_protocol import llapdiff_protocol_metadata, split_protocol_metadata
from llapdiffusion.configs.config_utils import (
    DEFAULT_PREDICT_TYPE,
    PREDICT_TYPES,
    make_jsonable,
    normalize_predict_type,
)
from llapdiffusion.configs.dataset_archives import configure_dataset_archive
from llapdiffusion.configs.dataset_defaults import apply_dataset_preset, dataset_keys, default_horizons, infer_dataset_key
from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.datasets.target_selection import resolve_target_selection
from llapdiffusion.logging_utils import apply_verbosity
from llapdiffusion.target_artifacts import (
    loader_target_request_from_config,
    sync_target_artifact_config,
)


COVERAGE_HELP = "fraction of observed context entries to hide; 0 disables induced missingness"
PREDICT_TYPE_DIR_PREFIX = "predict"


def _import_trainers():
    """Return the trainer modules exposed through the public package layout."""
    from llapdiffusion.trainers import train_val_latent, train_val_summarizer, train_val_llapdiff

    return train_val_latent, train_val_summarizer, train_val_llapdiff


def prepare_dataloaders(
    config=config,
) -> Tuple[Any, Any, Any, Tuple[int, int, int]]:
    """Build train/val/test loaders using the shared configuration."""

    run_experiment = resolve_run_experiment(config.DATA_DIR)
    target_col, target_cols = loader_target_request_from_config(config)
    batch_size = _effective_batch_size(config)
    return run_experiment(
        data_dir=config.DATA_DIR,
        date_batching=config.date_batching,
        dates_per_batch=batch_size,
        K=config.WINDOW,
        H=config.PRED,
        coverage=config.COVERAGE,
        batch_size=batch_size,
        ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
        split_policy=getattr(config, "split_policy", "global_purged_horizon"),
        exact_timestamp_batches=bool(getattr(config, "exact_timestamp_batches", True)),
        target_col=target_col,
        target_cols=target_cols,
    )


def _fmt_optional(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    if value is None:
        return "None"
    return str(value)


def _validate_coverage(value: object) -> float:
    coverage = float(value)
    if not 0.0 <= coverage < 1.0:
        raise ValueError("--coverage must satisfy 0 <= coverage < 1.")
    return coverage


def _validate_batch_size(value: object) -> int:
    batch_size = int(value)
    if batch_size < 1:
        raise ValueError("--batch-size must be a positive integer.")
    return batch_size


def _parse_predict_type(value: object) -> str:
    try:
        return normalize_predict_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _effective_batch_size(config=config) -> int:
    fallback = getattr(config, "DATES_PER_BATCH", 1)
    return _validate_batch_size(getattr(config, "BATCH_SIZE", fallback))


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


def _requested_predict_type_arg(config=config) -> str | None:
    requested = getattr(config, "REQUESTED_PREDICT_TYPE_ARG", None)
    if requested in (None, ""):
        return None
    return normalize_predict_type(requested)


def _active_predict_type(config=config) -> str:
    requested = _requested_predict_type_arg(config=config)
    if requested is not None:
        return requested
    return normalize_predict_type(getattr(config, "PREDICT_TYPE", DEFAULT_PREDICT_TYPE))


def _predict_type_dir_name(predict_type: str) -> str:
    return f"{PREDICT_TYPE_DIR_PREFIX}-{predict_type}"


def _path_contains_predict_type_dir(path: Path, dirname: str) -> bool:
    return any(part == dirname or part.startswith(f"{dirname}_") for part in path.parts)


def _apply_predict_type_output_routing(*, config=config) -> str:
    """
    Put explicit non-default prediction parameterizations in their own artifact dirs.

    The default v-pred run keeps the historical paths. Target-specific suffixing is
    still applied later by _sync_target_shape_config, so checkpoint filename tags keep
    their existing pred/target composition.
    """
    predict_type = _active_predict_type(config=config)
    config.PREDICT_TYPE = predict_type
    if predict_type == DEFAULT_PREDICT_TYPE:
        return predict_type

    dirname = _predict_type_dir_name(predict_type)

    def route(value: object) -> Path:
        path = Path(str(value))
        if _path_contains_predict_type_dir(path, dirname):
            return path
        return path / dirname

    if hasattr(config, "OUT_DIR"):
        config.OUT_DIR = str(route(getattr(config, "OUT_DIR")))
    if hasattr(config, "CKPT_DIR"):
        config.CKPT_DIR = str(route(getattr(config, "CKPT_DIR")))
    if hasattr(config, "POLE_PLOT_DIR") and hasattr(config, "OUT_DIR"):
        config.POLE_PLOT_DIR = str(Path(str(config.OUT_DIR)) / "pole_plots")
    return predict_type


def _target_policy(config=config) -> Dict[str, object]:
    requested = getattr(config, "TARGET_COL", None)
    requested_cols = getattr(config, "TARGET_COLS", None)
    meta_path = Path(str(getattr(config, "DATA_DIR", ""))) / "cache_ratio_index" / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        selected = resolve_target_selection(
            meta,
            None if requested_cols else requested,
            requested_target_cols=requested_cols,
        )
        return {
            "target_col": selected.target_col,
            "target_cols": list(selected.target_cols),
            "target_indices": list(selected.target_indices),
            "target_dim": selected.target_dim,
            "target_source": selected.target_source,
            "requested_target_col": selected.requested_target_col,
            "requested_target_cols": list(selected.requested_target_cols or []),
            "calendar_feature_cols": list(selected.calendar_feature_cols),
        }
    except Exception as exc:
        if requested not in (None, "") or requested_cols not in (None, "", []):
            raise ValueError(f"Could not resolve requested target columns.") from exc
        return {
            "target_col": requested,
            "target_cols": (
                list(requested_cols)
                if requested_cols
                else ([requested] if requested else [])
            ),
            "target_indices": [],
            "target_dim": 1,
            "target_source": "unresolved",
            "requested_target_col": requested,
            "requested_target_cols": list(requested_cols) if requested_cols else [],
            "calendar_feature_cols": [],
        }


def _sync_target_shape_config(config=config) -> Dict[str, object]:
    policy = _target_policy(config=config)
    sync_target_artifact_config(config, policy, update_output_dirs=True)
    return policy


def _update_config_for_pred(pred: int, config=config) -> None:
    split_policy = getattr(config, "split_policy", "global_purged_horizon")
    split_scope = getattr(config, "split_scope", "global_target_time")
    exact_timestamp_batches = bool(getattr(config, "exact_timestamp_batches", True))
    requested_batch_size = getattr(config, "REQUESTED_BATCH_SIZE_ARG", None)
    requested_target_col = getattr(
        config,
        "REQUESTED_TARGET_COL_ARG",
        getattr(config, "TARGET_COL", None),
    )
    requested_target_cols = getattr(
        config,
        "REQUESTED_TARGET_COLS_ARG",
        getattr(config, "TARGET_COLS", None),
    )
    requested_predict_type = _requested_predict_type_arg(config=config)
    active_predict_type = _active_predict_type(config=config)
    apply_dataset_preset(config, _resolve_dataset_key(config=config), pred=int(pred))
    config.split_policy = split_policy
    config.split_scope = split_scope
    config.exact_timestamp_batches = exact_timestamp_batches
    config.REQUESTED_BATCH_SIZE_ARG = requested_batch_size
    if requested_batch_size is not None:
        batch_size = _validate_batch_size(requested_batch_size)
        config.BATCH_SIZE = batch_size
        config.DATES_PER_BATCH = batch_size
    config.REQUESTED_TARGET_COL_ARG = requested_target_col
    config.REQUESTED_TARGET_COLS_ARG = requested_target_cols
    config.TARGET_COL = requested_target_col
    config.TARGET_COLS = requested_target_cols
    config.REQUESTED_PREDICT_TYPE_ARG = requested_predict_type
    config.PREDICT_TYPE = active_predict_type
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


def run_single_pred(
    pred: int,
    *,
    recompute_vae: bool = False,
    recompute_summarizer: bool = False,
    latent_plot_only: bool = False,
    use_shared_loaders: bool = True,
    run_checkpoint_eval: bool = False,
    allow_balanced_eval_failure: bool = False,
    checkpoint_eval_num_samples: int | None = None,
    checkpoint_eval_forecast_num_samples: int | None = None,
    checkpoint_eval_imputation_num_samples: int | None = None,
    checkpoint_eval_max_eval_batches: int | None = None,
    checkpoint_eval_random_mask_ratio: float | None = None,
    base_out_dir: Path | None = None,
    base_ckpt_dir: Path | None = None,
    training_overrides: Dict[str, object] | None = None,
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
    else:
        _apply_predict_type_output_routing(config=config)
    _apply_training_overrides(training_overrides, config=config)
    data_policy = _sync_target_shape_config(config=config)

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

    eval_stats = llapdiff_stats.get("eval_stats")
    if not isinstance(eval_stats, dict):
        raise ValueError("train_val_llapdiff.run must return an eval_stats dictionary.")
    best_val = llapdiff_stats.get("best_val")
    loaded_checkpoint = llapdiff_stats.get("loaded_checkpoint")
    balanced_evaluation = None
    eval_ckpt = loaded_checkpoint
    if run_checkpoint_eval and eval_ckpt:
        try:
            from llapdiffusion.tools.llapdiff_checkpoint_eval import evaluate_checkpoint

            eval_kwargs = {
                "label": f"{_resolve_dataset_key(config=config)}_pred{pred}",
                "random_mask_ratio": checkpoint_eval_random_mask_ratio,
                "num_samples": checkpoint_eval_num_samples,
                "forecast_num_samples": checkpoint_eval_forecast_num_samples,
                "imputation_num_samples": checkpoint_eval_imputation_num_samples,
                "max_eval_batches": checkpoint_eval_max_eval_batches,
            }
            if bool(getattr(config, "VERBOSE", False) or getattr(config, "DEBUG", False)):
                eval_kwargs["verbose"] = True
            balanced_evaluation = evaluate_checkpoint(config, eval_ckpt, **eval_kwargs)
        except Exception as exc:
            if not allow_balanced_eval_failure:
                raise
            balanced_evaluation = {
                "label": f"{_resolve_dataset_key(config=config)}_pred{pred}",
                "checkpoint": str(eval_ckpt),
                "status": "fail",
                "error": str(exc),
            }

    return {
        "pred": pred,
        "benchmark_protocol": llapdiff_protocol_metadata(),
        "vae": latent_stats,
        "summarizer": summarizer_stats,
        "llapdiff": llapdiff_stats,
        "eval_stats": eval_stats,
        "data_policy": {
            **data_policy,
            "split_policy": getattr(config, "split_policy", "global_purged_horizon"),
            "split_scope": getattr(config, "split_scope", "global_target_time"),
            "batching_policy": (
                "exact_context_end_timestamp"
                if bool(getattr(config, "exact_timestamp_batches", True))
                else "calendar_day"
            ),
            **split_protocol_metadata(
                _resolve_dataset_key(config=config),
                split_policy=getattr(config, "split_policy", "global_purged_horizon"),
                split_scope=getattr(config, "split_scope", "global_target_time"),
            ),
        },
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
    run_checkpoint_eval: bool = False,
    allow_balanced_eval_failure: bool = False,
    checkpoint_eval_num_samples: int | None = None,
    checkpoint_eval_forecast_num_samples: int | None = None,
    checkpoint_eval_imputation_num_samples: int | None = None,
    checkpoint_eval_max_eval_batches: int | None = None,
    checkpoint_eval_random_mask_ratio: float | None = None,
    training_overrides: Dict[str, object] | None = None,
    config=config,
) -> Dict[int, Dict[str, object]]:
    """
    Run the pipeline for multiple prediction horizons and collect stats.
    """
    _apply_predict_type_output_routing(config=config)
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
            run_checkpoint_eval=run_checkpoint_eval,
            allow_balanced_eval_failure=allow_balanced_eval_failure,
            checkpoint_eval_num_samples=checkpoint_eval_num_samples,
            checkpoint_eval_forecast_num_samples=checkpoint_eval_forecast_num_samples,
            checkpoint_eval_imputation_num_samples=checkpoint_eval_imputation_num_samples,
            checkpoint_eval_max_eval_batches=checkpoint_eval_max_eval_batches,
            checkpoint_eval_random_mask_ratio=checkpoint_eval_random_mask_ratio,
            base_out_dir=base_out_dir,
            base_ckpt_dir=base_ckpt_dir,
            training_overrides=training_overrides,
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
        "--target-col",
        type=str,
        default=None,
        help="Optional scalar target feature column. Defaults to the dataset cache target_col.",
    )
    parser.add_argument(
        "--target-cols",
        type=str,
        nargs="+",
        default=None,
        help="Optional target feature columns for multi-target forecasting.",
    )
    parser.add_argument(
        "--preds",
        type=int,
        nargs="+",
        default=None,
        help="Prediction horizons to run. Defaults to config.PIPELINE_PREDS or [config.PRED].",
    )
    parser.add_argument("--coverage", type=float, default=0.0, help=COVERAGE_HELP)
    parser.add_argument(
        "--batch-size",
        type=_validate_batch_size,
        default=None,
        help="Effective loader batch size. Defaults to the dataset preset table value.",
    )
    parser.add_argument(
        "--predict-type",
        type=_parse_predict_type,
        choices=PREDICT_TYPES,
        default=None,
        help="Diffusion prediction parameterization. Defaults to v.",
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
        "--allow-balanced-eval-failure",
        action="store_true",
        help="Record balanced checkpoint evaluation errors instead of failing the run.",
    )
    parser.add_argument(
        "--run-checkpoint-eval",
        action="store_true",
        help="After training, run the optional checkpoint forecast and target-imputation evaluation.",
    )
    parser.add_argument(
        "--checkpoint-eval-num-samples",
        type=int,
        default=None,
        help="Shared sample count for optional checkpoint forecast and imputation evaluation.",
    )
    parser.add_argument(
        "--checkpoint-eval-forecast-num-samples",
        type=int,
        default=None,
        help="Forecast sample count for optional checkpoint evaluation.",
    )
    parser.add_argument(
        "--checkpoint-eval-imputation-num-samples",
        type=int,
        default=None,
        help="Imputation sample count for optional checkpoint evaluation.",
    )
    parser.add_argument(
        "--checkpoint-eval-max-eval-batches",
        type=int,
        default=None,
        help="Optional batch cap for optional checkpoint evaluation; 0 means no cap.",
    )
    parser.add_argument(
        "--checkpoint-eval-random-mask-ratio",
        type=float,
        default=None,
        help="Fraction of observed target entries hidden in optional random-mask imputation evaluation.",
    )
    parser.add_argument(
        "--target-mask-aux-p",
        type=float,
        default=None,
        help="Probability of mixing target-mask reconstruction batches into LLapDiff training.",
    )
    parser.add_argument(
        "--target-mask-aux-keep-mode",
        choices=("random", "regular", "prefix", "mixed"),
        default=None,
        help="Target-mask auxiliary keep-mask mode.",
    )
    parser.add_argument(
        "--target-mask-aux-keep-prob",
        type=float,
        default=None,
        help="Observed target keep probability for random target-mask auxiliary batches.",
    )
    parser.add_argument(
        "--target-mask-aux-keep-stride",
        type=int,
        default=None,
        help="Observed target keep stride for regular target-mask auxiliary batches.",
    )
    parser.add_argument(
        "--target-mask-aux-start-epoch",
        type=int,
        default=None,
        help="Epoch at which target-mask auxiliary batches may begin.",
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
        help="Optional directory for extracting --dataset-zip. Defaults to the user cache directory.",
    )
    parser.add_argument(
        "--split-policy",
        choices=("global_purged_horizon", "per_asset_purged_horizon", "contiguous"),
        default=None,
        help="Loader split policy. Defaults to the corrected global_purged_horizon policy.",
    )
    parser.add_argument(
        "--calendar-day-batches",
        action="store_true",
        help="Use legacy calendar-day batch grouping instead of exact context-end timestamp grouping.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print trainer diagnostics.")
    parser.add_argument("--debug", action="store_true", help="Print verbose trainer diagnostics.")
    return parser.parse_args()


def _training_overrides_from_args(args: argparse.Namespace) -> Dict[str, object]:
    names = (
        "target_mask_aux_p",
        "target_mask_aux_keep_mode",
        "target_mask_aux_keep_prob",
        "target_mask_aux_keep_stride",
        "target_mask_aux_start_epoch",
    )
    return {name: value for name in names if (value := getattr(args, name, None)) is not None}


def _apply_training_overrides(overrides: Dict[str, object] | None, *, config=config) -> None:
    if not overrides:
        return

    if "target_mask_aux_p" in overrides:
        value = float(overrides["target_mask_aux_p"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("--target-mask-aux-p must be between 0 and 1.")
        config.TARGET_MASK_AUX_P = value
        if value > 0.0:
            config.IMPUTATION_TRAINING = True

    if "target_mask_aux_keep_mode" in overrides:
        config.TARGET_MASK_AUX_KEEP_MODE = str(overrides["target_mask_aux_keep_mode"])

    if "target_mask_aux_keep_prob" in overrides:
        value = float(overrides["target_mask_aux_keep_prob"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("--target-mask-aux-keep-prob must be between 0 and 1.")
        config.TARGET_MASK_AUX_KEEP_PROB = value

    if "target_mask_aux_keep_stride" in overrides:
        value = int(overrides["target_mask_aux_keep_stride"])
        if value < 1:
            raise ValueError("--target-mask-aux-keep-stride must be at least 1.")
        config.TARGET_MASK_AUX_KEEP_STRIDE = value

    if "target_mask_aux_start_epoch" in overrides:
        value = int(overrides["target_mask_aux_start_epoch"])
        if value < 0:
            raise ValueError("--target-mask-aux-start-epoch must be non-negative.")
        config.TARGET_MASK_AUX_START_EPOCH = value


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
    initial_pred = int(args.preds[0]) if args.preds else int(default_horizons(args.dataset_key)[0])
    apply_dataset_preset(config, args.dataset_key, pred=initial_pred)
    apply_verbosity(config, verbose=args.verbose, debug=args.debug)
    config.REQUESTED_BATCH_SIZE_ARG = args.batch_size
    if args.batch_size is not None:
        batch_size = _validate_batch_size(args.batch_size)
        config.BATCH_SIZE = batch_size
        config.DATES_PER_BATCH = batch_size
    if args.target_col and args.target_cols:
        raise ValueError("Use either --target-col or --target-cols, not both.")
    config.REQUESTED_TARGET_COL_ARG = args.target_col
    config.REQUESTED_TARGET_COLS_ARG = list(args.target_cols) if args.target_cols else None
    config.TARGET_COL = args.target_col
    config.TARGET_COLS = args.target_cols
    if args.split_policy is not None:
        config.split_policy = args.split_policy
    if args.calendar_day_batches:
        config.exact_timestamp_batches = False
    config.COVERAGE = _validate_coverage(args.coverage)
    config.REQUESTED_PREDICT_TYPE_ARG = args.predict_type
    if args.predict_type is not None:
        config.PREDICT_TYPE = args.predict_type
    else:
        config.PREDICT_TYPE = normalize_predict_type(getattr(config, "PREDICT_TYPE", DEFAULT_PREDICT_TYPE))
    training_overrides = _training_overrides_from_args(args)
    preds = tuple(args.preds) if args.preds else _pred_list_from_config(config=config)
    if not preds:
        raise ValueError("No prediction horizons provided to the pipeline.")

    _apply_predict_type_output_routing(config=config)
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
            run_checkpoint_eval=args.run_checkpoint_eval,
            allow_balanced_eval_failure=args.allow_balanced_eval_failure,
            checkpoint_eval_num_samples=args.checkpoint_eval_num_samples,
            checkpoint_eval_forecast_num_samples=args.checkpoint_eval_forecast_num_samples,
            checkpoint_eval_imputation_num_samples=args.checkpoint_eval_imputation_num_samples,
            checkpoint_eval_max_eval_batches=args.checkpoint_eval_max_eval_batches,
            checkpoint_eval_random_mask_ratio=args.checkpoint_eval_random_mask_ratio,
            base_out_dir=base_out_dir,
            base_ckpt_dir=base_ckpt_dir,
            training_overrides=training_overrides,
            config=config,
        )

    _print_summary_table(results)
    if args.summary_json:
        _save_summary_json(args.summary_json, results)
    return results


def cli_main() -> None:
    main()


if __name__ == "__main__":  # pragma: no cover
    cli_main()
