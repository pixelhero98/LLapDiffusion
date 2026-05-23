"""Run LLapDiff on controlled synthetic regime-shift tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from llapdiffusion.configs.config_utils import clone_config, make_jsonable
from llapdiffusion.logging_utils import apply_verbosity
from llapdiffusion.datasets.synthetic_regime_dataset import (
    SyntheticRegimeCacheConfig,
    build_context_end_eval_loader,
    build_regime_eval_loader,
    prepare_synthetic_regime_cache,
    run_experiment as synthetic_run_experiment,
)
from llapdiffusion.tools.llapdiff_checkpoint_eval import _load_stack
from llapdiffusion.trainers import train_val_latent, train_val_llapdiff, train_val_summarizer
from llapdiffusion.trainers import train_val_llapdiff as tv

from llapdiffusion.models.llapdiff_utils import set_torch


DEFAULT_WORK_ROOT = Path.cwd()
COVERAGE_HELP = "fraction of observed context entries to hide; 0 disables induced missingness"


TASK_SPECS: Mapping[str, Mapping[str, object]] = {
    "synthetic_freq_shift": {
        "data_seed": 20260327,
        "shift_kind": "frequency",
        "default_multiplier": 2.0,
    },
    "synthetic_decay_shift": {
        "data_seed": 20260328,
        "shift_kind": "decay",
        "default_multiplier": 2.5,
    },
}

DEFAULT_SEEDS = (3407, 3408, 3409)
DEFAULT_FREQ_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0)
DEFAULT_DECAY_MULTIPLIERS = (1.25, 1.5, 2.0, 2.5)
CONDITIONING_MODES = ("conditioned", "unconditioned")


@dataclass(frozen=True)
class RunSpec:
    task: str
    seed: int
    shift_multiplier: float
    protocol_name: str

    @property
    def shift_kind(self) -> str:
        return str(TASK_SPECS[self.task]["shift_kind"])

    @property
    def shift_tag(self) -> str:
        prefix = "freqx" if self.task == "synthetic_freq_shift" else "decayx"
        return f"{prefix}{_tag_float(self.shift_multiplier)}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run public LLapDiff synthetic frequency/decay regime-shift tests."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_SPECS.keys()),
        default=tuple(TASK_SPECS.keys()),
        help="Synthetic tasks to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--protocol-name",
        choices=("boundary_crossing", "strict_unseen_regime"),
        default="boundary_crossing",
    )
    parser.add_argument(
        "--freq-multipliers",
        nargs="+",
        type=float,
        default=DEFAULT_FREQ_MULTIPLIERS,
        help="Frequency-shift severities.",
    )
    parser.add_argument(
        "--decay-multipliers",
        nargs="+",
        type=float,
        default=DEFAULT_DECAY_MULTIPLIERS,
        help="Decay-shift severities.",
    )
    parser.add_argument(
        "--conditioning-modes",
        nargs="+",
        choices=CONDITIONING_MODES,
        default=CONDITIONING_MODES,
        help="Evaluate checkpoints with conditioning enabled, disabled, or both.",
    )
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--series-length", type=int, default=None)
    parser.add_argument("--change-point", type=int, default=None)
    parser.add_argument("--num-entities", type=int, default=64)
    parser.add_argument(
        "--lookback-steps",
        type=int,
        default=12,
        help="Context-end lookback width for the boundary-crossing robustness slice.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_WORK_ROOT / "ldt" / "synthetic_data" / "synthetic_regime"),
        help="Root directory for generated synthetic caches.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_WORK_ROOT / "ldt" / "results" / "synthetic_regime"),
        help="Root directory for synthetic metric tables.",
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=str(DEFAULT_WORK_ROOT / "ldt" / "synthetic_artifacts"),
        help="Root directory for synthetic VAE, summarizer, and diffusion checkpoints.",
    )
    parser.add_argument("--validate-split-only", action="store_true")
    parser.add_argument("--recompute-artifacts", action="store_true")
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--print-json", action="store_true", help="Print the full overall JSON payload to stdout.")
    parser.add_argument("--target-col", type=str, default=None, help="Optional scalar target feature column.")
    parser.add_argument("--target-cols", nargs="+", default=None, help="Optional target feature columns.")
    parser.add_argument("--coverage", type=float, default=0.0, help=COVERAGE_HELP)
    parser.add_argument("--verbose", action="store_true", help="Print trainer diagnostics.")
    parser.add_argument("--debug", action="store_true", help="Print verbose trainer diagnostics.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a short training/evaluation schedule for end-to-end smoke testing.",
    )
    return parser.parse_args()


def _tag_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _validate_coverage(value: object) -> float:
    coverage = float(value)
    if not 0.0 <= coverage < 1.0:
        raise ValueError("--coverage must satisfy 0 <= coverage < 1.")
    return coverage


def _protocol_defaults(protocol_name: str) -> Tuple[int, int]:
    if protocol_name == "strict_unseen_regime":
        return 432, 373
    return 288, 216


def _resolve_series_length(args: argparse.Namespace) -> int:
    default_length, _ = _protocol_defaults(str(args.protocol_name))
    return int(args.series_length if args.series_length is not None else default_length)


def _resolve_change_point(args: argparse.Namespace) -> int:
    _, default_change = _protocol_defaults(str(args.protocol_name))
    return int(args.change_point if args.change_point is not None else default_change)


def _multipliers_for_task(task: str, args: argparse.Namespace) -> Sequence[float]:
    if task == "synthetic_freq_shift":
        return tuple(float(v) for v in args.freq_multipliers)
    return tuple(float(v) for v in args.decay_multipliers)


def _specs(args: argparse.Namespace) -> List[RunSpec]:
    return [
        RunSpec(str(task), int(seed), float(multiplier), str(args.protocol_name))
        for task in args.tasks
        for multiplier in _multipliers_for_task(str(task), args)
        for seed in args.seeds
    ]


def _cache_dir(spec: RunSpec, args: argparse.Namespace) -> Path:
    return (
        Path(args.data_root)
        / spec.task
        / spec.protocol_name
        / spec.shift_tag
        / f"len-{_resolve_series_length(args)}_cp-{_resolve_change_point(args)}_entities-{int(args.num_entities)}"
    ).resolve()


def _artifact_dir(spec: RunSpec, args: argparse.Namespace) -> Path:
    return (
        Path(args.artifact_root)
        / spec.task
        / spec.protocol_name
        / spec.shift_tag
        / f"seed-{int(spec.seed)}"
    ).resolve()


def _result_dir(args: argparse.Namespace) -> Path:
    return (Path(args.output_root) / str(args.protocol_name)).resolve()


def _prepare_cache(spec: RunSpec, args: argparse.Namespace) -> Mapping[str, object]:
    cfg = SyntheticRegimeCacheConfig(
        task=spec.task,
        window=int(args.window),
        horizon=int(args.horizon),
        data_dir=str(_cache_dir(spec, args)),
        num_entities=int(args.num_entities),
        series_length=_resolve_series_length(args),
        change_point=_resolve_change_point(args),
        seed=int(TASK_SPECS[spec.task]["data_seed"]),
        freq_multiplier=(
            float(spec.shift_multiplier)
            if spec.task == "synthetic_freq_shift"
            else float(TASK_SPECS["synthetic_freq_shift"]["default_multiplier"])
        ),
        decay_multiplier=(
            float(spec.shift_multiplier)
            if spec.task == "synthetic_decay_shift"
            else float(TASK_SPECS["synthetic_decay_shift"]["default_multiplier"])
        ),
        overwrite=bool(args.overwrite_data),
    )
    return prepare_synthetic_regime_cache(cfg)


def _configure(spec: RunSpec, args: argparse.Namespace) -> SimpleNamespace:
    cfg = clone_config()
    base = _artifact_dir(spec, args)
    cfg.DATASET_KEY = spec.task
    cfg.MKT = spec.task
    cfg.SEED = int(spec.seed)
    cfg.DATA_DIR = str(_cache_dir(spec, args))
    cfg.WINDOW = int(args.window)
    cfg.PRED = int(args.horizon)
    cfg.SUM_CONTEXT_LEN_FIXED = int(args.window)
    cfg.SUM_CONTEXT_LEN = int(args.window)
    cfg.COVERAGE = _validate_coverage(getattr(args, "coverage", 0.0))
    cfg.date_batching = True
    cfg.DATES_PER_BATCH = 16
    cfg.BATCH_SIZE = 16
    cfg.VAE_LATENT_CHANNELS = 24
    cfg.VAE_ENTITY_CONDITION = True
    cfg.VAE_NUM_ENTITIES = None
    cfg.PREDICT_TYPE = "v"
    cfg.LOSS_WEIGHT_SCHEME = "weighted_min_snr"
    cfg.MINSNR_GAMMA = 5.0
    cfg.BASE_LR = 1.5e-4
    cfg.LR_SCHEDULE = "warmup_constant"
    cfg.PRIMARY_EVAL_METRIC = "val_diag_mse_raw"
    cfg.TARGET_MASK_AUX_P = 0.0
    cfg.TARGET_MASK_AUX_START_EPOCH = 0
    cfg.TIMESTEPS = 1000
    cfg.SCHEDULE = "cosine"
    cfg.MODEL_WIDTH = 256
    cfg.NUM_LAYERS = 5
    cfg.NUM_HEADS = 4
    cfg.LAPLACE_K = 256
    cfg.EPOCHS = 80
    cfg.EARLY_STOP = 8
    cfg.EARLY_STOP_MIN_EPOCHS = 20
    cfg.EVAL_EVERY = 1
    cfg.VAL_DIAG_EVERY = 1
    cfg.DOWNSTREAM_EVAL_EVERY = 10
    cfg.IRREG_CHECK_EVERY = 0
    cfg.EMA_COMPARE_EVERY = 0
    cfg.NUM_EVAL_SAMPLES = 10
    cfg.EVAL_STEPS = 32
    cfg.TEST_STEPS = 64
    cfg.GEN_STEPS = 64
    cfg.GEN_ETA = 0.0
    cfg.DIFF_AMP = False

    if args.smoke:
        cfg.EPOCHS = 1
        cfg.EARLY_STOP = 1
        cfg.EARLY_STOP_MIN_EPOCHS = 0
        cfg.SUM_EPOCHS = 1
        cfg.SUM_PATIENCE = 1
        cfg.VAE_WARMUP_EPOCHS = 0
        cfg.VAE_KL_ANNEAL_EPOCHS = 1
        cfg.VAE_MIN_EPOCHS = 1
        cfg.VAE_MAX_PATIENCE = 1
        cfg.NUM_EVAL_SAMPLES = 2
        cfg.EVAL_STEPS = 4
        cfg.TEST_STEPS = 4
        cfg.GEN_STEPS = 4
        cfg.DOWNSTREAM_EVAL_EVERY = 1

    cfg.VAE_DIR = str(base / "vae")
    cfg.SUM_DIR = str(base / "summarizer")
    cfg.CKPT_DIR = str(base / "checkpoints")
    cfg.OUT_DIR = str(base / "output")
    cfg.POLE_PLOT_DIR = str(base / "output" / "pole_plots")
    cfg.VAE_CKPT = str(
        Path(cfg.VAE_DIR) / f"pred-{cfg.PRED}_ch-{cfg.VAE_LATENT_CHANNELS}_entity_elbo.pt"
    )
    cfg.SUM_CKPT = str(Path(cfg.SUM_DIR) / f"{cfg.PRED}-{cfg.VAE_LATENT_CHANNELS}-summarizer.pt")
    cfg.SYNTHETIC_SERIES_LENGTH = _resolve_series_length(args)
    cfg.SYNTHETIC_CHANGE_POINT = _resolve_change_point(args)
    cfg.SYNTHETIC_NUM_ENTITIES = int(args.num_entities)
    if getattr(args, "target_col", None) and getattr(args, "target_cols", None):
        raise ValueError("Use either --target-col or --target-cols, not both.")
    cfg.TARGET_COL = getattr(args, "target_col", None)
    cfg.TARGET_COLS = list(args.target_cols) if getattr(args, "target_cols", None) else None
    apply_verbosity(cfg, verbose=bool(getattr(args, "verbose", False)), debug=bool(getattr(args, "debug", False)))
    return cfg


def _validate_forecast_only_config(cfg: SimpleNamespace) -> None:
    target_mask_aux_p = float(getattr(cfg, "TARGET_MASK_AUX_P", 0.0))
    if abs(target_mask_aux_p) > 1e-12:
        raise RuntimeError(
            "Synthetic regime-shift runs are forecast-only; "
            f"TARGET_MASK_AUX_P must be 0.0, got {target_mask_aux_p}."
        )


def _build_loaders(cfg: SimpleNamespace):
    batch_size = int(getattr(cfg, "BATCH_SIZE", getattr(cfg, "DATES_PER_BATCH", 1)))
    return synthetic_run_experiment(
        data_dir=cfg.DATA_DIR,
        date_batching=bool(getattr(cfg, "date_batching", True)),
        dates_per_batch=batch_size,
        K=int(cfg.WINDOW),
        H=int(cfg.PRED),
        coverage=float(cfg.COVERAGE),
        ratios=(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio),
        batch_size=batch_size,
        norm="train_only",
        per_asset=True,
        split_policy=getattr(cfg, "split_policy", "global_purged_horizon"),
        exact_timestamp_batches=bool(getattr(cfg, "exact_timestamp_batches", True)),
        target_col=None if getattr(cfg, "TARGET_COLS", None) else getattr(cfg, "TARGET_COL", None),
        target_cols=getattr(cfg, "TARGET_COLS", None),
        shuffle_train=False,
    )


def _resolve_pairs_and_window(dataset) -> Tuple[np.ndarray, int]:
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        pairs, window = _resolve_pairs_and_window(dataset.dataset)
        return np.asarray(pairs)[np.asarray(dataset.indices, dtype=np.int64)], int(window)
    if not hasattr(dataset, "pairs") or not hasattr(dataset, "window"):
        raise TypeError("dataset must expose pairs and window")
    return np.asarray(dataset.pairs), int(dataset.window)


def _per_asset(values: np.ndarray) -> Dict[str, object]:
    if values.size == 0:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": int(values.min()), "max": int(values.max()), "mean": float(values.mean())}


def _loader_geometry(loader, *, change_point: int, horizon: int) -> Dict[str, object]:
    pairs, window = _resolve_pairs_and_window(loader.dataset)
    if pairs.size == 0:
        return {"count": 0}
    asset_ids = pairs[:, 0].astype(np.int64)
    starts = pairs[:, 1].astype(np.int64)
    context_end = starts + int(window) - 1
    forecast_start = context_end + 1
    forecast_end = context_end + int(horizon)
    forecast_only_violations = forecast_start != (context_end + 1)
    boundary_crossing = (
        (context_end < int(change_point))
        & (forecast_start <= int(change_point))
        & (forecast_end >= int(change_point))
    )
    post_shift_context = context_end >= int(change_point)
    pre_change_forecast = forecast_end < int(change_point)
    shifted_target = forecast_end >= int(change_point)

    def counts_per_asset(mask: np.ndarray) -> Dict[str, object]:
        counts = [int(mask[asset_ids == aid].sum()) for aid in np.unique(asset_ids)]
        return _per_asset(np.asarray(counts, dtype=np.int64))

    return {
        "count": int(pairs.shape[0]),
        "window": int(window),
        "context_end_min": int(context_end.min()),
        "context_end_max": int(context_end.max()),
        "forecast_start_min": int(forecast_start.min()),
        "forecast_start_max": int(forecast_start.max()),
        "forecast_end_min": int(forecast_end.min()),
        "forecast_end_max": int(forecast_end.max()),
        "forecast_only_violations_total": int(forecast_only_violations.sum()),
        "forecast_only_violations_per_asset": counts_per_asset(forecast_only_violations),
        "boundary_crossing_windows_total": int(boundary_crossing.sum()),
        "boundary_crossing_windows_per_asset": counts_per_asset(boundary_crossing),
        "post_shift_context_windows_total": int(post_shift_context.sum()),
        "post_shift_context_windows_per_asset": counts_per_asset(post_shift_context),
        "pre_change_forecast_windows_total": int(pre_change_forecast.sum()),
        "pre_change_forecast_windows_per_asset": counts_per_asset(pre_change_forecast),
        "shifted_target_windows_total": int(shifted_target.sum()),
        "shifted_target_windows_per_asset": counts_per_asset(shifted_target),
    }


def _validate_strict_geometry(split_geometry: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    train = split_geometry["train"]
    val = split_geometry["val"]
    test = split_geometry["test"]
    checks = {
        "train_forecast_only": int(train["forecast_only_violations_total"]) == 0,
        "val_forecast_only": int(val["forecast_only_violations_total"]) == 0,
        "test_forecast_only": int(test["forecast_only_violations_total"]) == 0,
        "train_fully_pre_change": int(train["shifted_target_windows_total"]) == 0,
        "val_fully_pre_change": int(val["shifted_target_windows_total"]) == 0,
        "test_has_boundary_crossing": int(test["boundary_crossing_windows_total"]) > 0,
    }
    checks["valid"] = all(bool(v) for v in checks.values())
    return checks


def _split_geometry(cfg: SimpleNamespace, loaders) -> Dict[str, object]:
    train_dl, val_dl, test_dl, _ = loaders
    meta = json.loads((Path(cfg.DATA_DIR) / "cache_ratio_index" / "meta.json").read_text())
    change_point = int(meta["change_point"])
    geometry = {
        "train": _loader_geometry(train_dl, change_point=change_point, horizon=int(cfg.PRED)),
        "val": _loader_geometry(val_dl, change_point=change_point, horizon=int(cfg.PRED)),
        "test": _loader_geometry(test_dl, change_point=change_point, horizon=int(cfg.PRED)),
    }
    return {"change_point": change_point, "splits": geometry, "strict": _validate_strict_geometry(geometry)}


def _metric(payload: Optional[Mapping[str, object]], metric: str) -> Optional[float]:
    if not isinstance(payload, Mapping) or payload.get(metric) is None:
        return None
    value = float(payload[metric])
    return value if math.isfinite(value) else None


def _write_rows(rows: Sequence[Mapping[str, object]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(make_jsonable(list(rows)), indent=2, sort_keys=True))


def _geometry_row(spec: RunSpec, cfg: SimpleNamespace, split_payload: Mapping[str, object]) -> Dict[str, object]:
    strict = split_payload["strict"]
    train = split_payload["splits"]["train"]
    val = split_payload["splits"]["val"]
    test = split_payload["splits"]["test"]
    return {
        "task": spec.task,
        "shift_kind": spec.shift_kind,
        "shift_multiplier": float(spec.shift_multiplier),
        "shift_tag": spec.shift_tag,
        "seed": int(spec.seed),
        "protocol_name": spec.protocol_name,
        "series_length": int(cfg.SYNTHETIC_SERIES_LENGTH),
        "change_point": int(split_payload["change_point"]),
        "window": int(cfg.WINDOW),
        "horizon": int(cfg.PRED),
        "num_entities": int(cfg.SYNTHETIC_NUM_ENTITIES),
        "strict_geometry_valid": bool(strict["valid"]),
        "train_forecast_end_max": train["forecast_end_max"],
        "val_forecast_end_max": val["forecast_end_max"],
        "test_context_end_min": test["context_end_min"],
        "test_context_end_max": test["context_end_max"],
        "test_boundary_crossing_windows_per_asset": test["boundary_crossing_windows_per_asset"]["max"],
        "test_post_shift_context_windows_per_asset": test["post_shift_context_windows_per_asset"]["max"],
    }


def _stats(values: Iterable[object]) -> Dict[str, object]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _summary_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            row.get("task"),
            row.get("shift_kind"),
            row.get("shift_multiplier"),
            row.get("protocol_name"),
            row.get("conditioning_mode"),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row: Dict[str, object] = {
            "task": key[0],
            "shift_kind": key[1],
            "shift_multiplier": key[2],
            "protocol_name": key[3],
            "conditioning_mode": key[4],
            "runs": len(group_rows),
        }
        for metric in (
            "forecast_crps",
            "forecast_mae",
            "forecast_mse",
            "boundary_crossing_crps",
            "boundary_crossing_mae",
            "boundary_crossing_mse",
            "post_shift_context_crps",
            "post_shift_context_mae",
            "post_shift_context_mse",
        ):
            stat = _stats(r.get(metric) for r in group_rows)
            row[f"{metric}_mean"] = stat["mean"]
            row[f"{metric}_std"] = stat["std"]
        out.append(row)
    return out


def _conditioned_vs_unconditioned_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], Dict[str, Mapping[str, object]]] = {}
    for row in rows:
        key = (
            row.get("task"),
            row.get("shift_kind"),
            row.get("shift_multiplier"),
            row.get("protocol_name"),
            row.get("seed"),
        )
        groups.setdefault(key, {})[str(row.get("conditioning_mode"))] = row

    pair_rows = []
    for key, modes in groups.items():
        if "conditioned" not in modes or "unconditioned" not in modes:
            continue
        cond = modes["conditioned"]
        uncond = modes["unconditioned"]
        out = {
            "task": key[0],
            "shift_kind": key[1],
            "shift_multiplier": key[2],
            "protocol_name": key[3],
            "seed": key[4],
        }
        for metric in (
            "boundary_crossing_crps",
            "boundary_crossing_mae",
            "boundary_crossing_mse",
            "post_shift_context_crps",
            "post_shift_context_mae",
            "post_shift_context_mse",
        ):
            cval = _metric(cond, metric)
            uval = _metric(uncond, metric)
            out[f"conditioned_{metric}"] = cval
            out[f"unconditioned_{metric}"] = uval
            out[f"unconditioned_minus_conditioned_{metric}"] = (
                None if cval is None or uval is None else float(uval) - float(cval)
            )
        pair_rows.append(out)

    summary_groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in pair_rows:
        key = (row["task"], row["shift_kind"], row["shift_multiplier"], row["protocol_name"])
        summary_groups.setdefault(key, []).append(row)
    summary = []
    for key, group_rows in sorted(summary_groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        out = {
            "task": key[0],
            "shift_kind": key[1],
            "shift_multiplier": key[2],
            "protocol_name": key[3],
            "paired_runs": len(group_rows),
        }
        for metric in (
            "boundary_crossing_crps",
            "boundary_crossing_mae",
            "boundary_crossing_mse",
            "post_shift_context_crps",
            "post_shift_context_mae",
            "post_shift_context_mse",
        ):
            for prefix in ("conditioned", "unconditioned", "unconditioned_minus_conditioned"):
                stat = _stats(r.get(f"{prefix}_{metric}") for r in group_rows)
                out[f"{prefix}_{metric}_mean"] = stat["mean"]
                out[f"{prefix}_{metric}_std"] = stat["std"]
        summary.append(out)
    return summary


def _train_or_reuse_stack(cfg: SimpleNamespace, loaders, args: argparse.Namespace) -> Mapping[str, object]:
    train_dl, val_dl, test_dl, sizes = loaders
    if args.recompute_artifacts or not Path(cfg.VAE_CKPT).exists():
        vae_stats = train_val_latent.run(
            train_dl=train_dl,
            val_dl=val_dl,
            test_dl=test_dl,
            sizes=sizes,
            config=cfg,
        )
        if vae_stats.get("best_elbo_path"):
            cfg.VAE_CKPT = str(vae_stats["best_elbo_path"])
    else:
        vae_stats = {"status": "skipped", "reason": "checkpoint_exists", "checkpoint": cfg.VAE_CKPT}

    if args.recompute_artifacts or not Path(cfg.SUM_CKPT).exists():
        sum_stats = train_val_summarizer.run(
            train_loader=train_dl,
            val_loader=val_dl,
            test_loader=test_dl,
            sizes=sizes,
            config=cfg,
        )
        if sum_stats.get("checkpoint"):
            cfg.SUM_CKPT = str(sum_stats["checkpoint"])
    else:
        sum_stats = {"status": "skipped", "reason": "checkpoint_exists", "checkpoint": cfg.SUM_CKPT}

    llap_stats = train_val_llapdiff.run(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        sizes=sizes,
        config=cfg,
    )
    checkpoint = (
        llap_stats.get("best_checkpoint_raw")
        or llap_stats.get("best_checkpoint")
        or llap_stats.get("loaded_checkpoint")
        or llap_stats.get("last_checkpoint")
    )
    if not checkpoint:
        raise RuntimeError("LLapDiff training did not produce an evaluation checkpoint.")
    return {
        "vae": vae_stats,
        "summarizer": sum_stats,
        "llapdiff": llap_stats,
        "paths": {"vae": cfg.VAE_CKPT, "summarizer": cfg.SUM_CKPT, "llapdiff": str(checkpoint)},
    }


def _row_from_eval(
    *,
    spec: RunSpec,
    cfg: SimpleNamespace,
    conditioning_mode: str,
    split_payload: Mapping[str, object],
    forecast: Mapping[str, object],
    crossing: Mapping[str, object],
    post_shift: Optional[Mapping[str, object]],
    crossing_subset_size: int,
    post_shift_subset_size: Optional[int],
    checkpoint: str,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "task": spec.task,
        "shift_kind": spec.shift_kind,
        "shift_multiplier": float(spec.shift_multiplier),
        "shift_tag": spec.shift_tag,
        "seed": int(spec.seed),
        "protocol_name": spec.protocol_name,
        "conditioning_mode": conditioning_mode,
        "series_length": int(cfg.SYNTHETIC_SERIES_LENGTH),
        "change_point": int(split_payload["change_point"]),
        "window": int(cfg.WINDOW),
        "horizon": int(cfg.PRED),
        "num_entities": int(cfg.SYNTHETIC_NUM_ENTITIES),
        "target_col": getattr(cfg, "TARGET_COL", None),
        "target_cols": json.dumps(list(getattr(cfg, "TARGET_COLS", None) or [])),
        "target_dim": int(getattr(cfg, "TARGET_DIM", 1)),
        "checkpoint": checkpoint,
        "strict_geometry_valid": bool(split_payload["strict"]["valid"]),
        "boundary_crossing_subset_size": int(crossing_subset_size),
        "post_shift_context_subset_size": post_shift_subset_size,
    }
    for prefix, payload in (
        ("forecast", forecast),
        ("boundary_crossing", crossing),
        ("post_shift_context", post_shift),
    ):
        for metric in ("crps", "mae", "mse"):
            row[f"{prefix}_{metric}"] = _metric(payload, metric)
    return row


def _evaluate_checkpoint(
    spec: RunSpec,
    cfg: SimpleNamespace,
    loaders,
    split_payload: Mapping[str, object],
    checkpoint: str,
    conditioning_mode: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    train_dl, _, test_dl, _ = loaders
    device = set_torch(seed=int(getattr(cfg, "SEED", 42)), deterministic=bool(getattr(cfg, "DETERMINISTIC", False)))
    diff_model, vae, summarizer, mu_mean, mu_std = _load_stack(cfg, Path(checkpoint), device, train_dl)
    sampling = tv._sampling_kwargs(cfg, prefix="TEST")
    disable_conditioning = conditioning_mode == "unconditioned"
    common = dict(
        device=device,
        mu_mean=mu_mean,
        mu_std=mu_std,
        config=cfg,
        ema=None,
        self_cond=bool(getattr(cfg, "SELF_COND", False)),
        disable_conditioning=disable_conditioning,
        verbose=bool(args.verbose or args.debug),
    )
    forecast = tv.evaluate_regression(diff_model, vae, summarizer, test_dl, **common, **sampling)
    change_point = int(split_payload["change_point"])
    if spec.protocol_name == "boundary_crossing":
        crossing_loader = build_regime_eval_loader(
            test_dl,
            change_point=change_point,
            lookback_steps=int(args.lookback_steps),
        )
        post_loader = None
    else:
        crossing_loader = build_context_end_eval_loader(
            test_dl,
            min_context_end=change_point - int(cfg.PRED),
            max_context_end=change_point - 1,
        )
        post_loader = build_context_end_eval_loader(test_dl, min_context_end=change_point, max_context_end=None)
    crossing = tv.evaluate_regression(diff_model, vae, summarizer, crossing_loader, **common, **sampling)
    post_shift = None
    post_size = None
    if post_loader is not None:
        post_size = int(len(post_loader.dataset))
        post_shift = tv.evaluate_regression(diff_model, vae, summarizer, post_loader, **common, **sampling)
    return _row_from_eval(
        spec=spec,
        cfg=cfg,
        conditioning_mode=conditioning_mode,
        split_payload=split_payload,
        forecast=forecast,
        crossing=crossing,
        post_shift=post_shift,
        crossing_subset_size=int(len(crossing_loader.dataset)),
        post_shift_subset_size=post_size,
        checkpoint=str(checkpoint),
    )


def main() -> None:
    args = _parse_args()
    result_root = _result_dir(args)
    raw_rows: List[Dict[str, object]] = []
    geometry_rows: List[Dict[str, object]] = []

    for spec in _specs(args):
        _prepare_cache(spec, args)
        cfg = _configure(spec, args)
        _validate_forecast_only_config(cfg)
        loaders = _build_loaders(cfg)
        split_payload = _split_geometry(cfg, loaders)
        if spec.protocol_name == "strict_unseen_regime" and not bool(split_payload["strict"]["valid"]):
            raise RuntimeError(
                "Strict unseen-regime geometry is invalid: "
                + json.dumps(make_jsonable(split_payload["strict"]), sort_keys=True)
            )
        geometry_rows.append(_geometry_row(spec, cfg, split_payload))
        if args.validate_split_only:
            continue

        stage_payload = _train_or_reuse_stack(cfg, loaders, args)
        checkpoint = str(stage_payload["paths"]["llapdiff"])
        stage_path = _artifact_dir(spec, args) / "stage_summary.json"
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path.write_text(json.dumps(make_jsonable(stage_payload), indent=2, sort_keys=True))
        for conditioning_mode in args.conditioning_modes:
            raw_rows.append(
                _evaluate_checkpoint(spec, cfg, loaders, split_payload, checkpoint, str(conditioning_mode), args)
            )

    _write_rows(
        geometry_rows,
        result_root / "synthetic_regime_geometry.csv",
        result_root / "synthetic_regime_geometry.json",
    )
    if raw_rows:
        _write_rows(
            raw_rows,
            result_root / "synthetic_regime_raw.csv",
            result_root / "synthetic_regime_raw.json",
        )
        _write_rows(
            _summary_rows(raw_rows),
            result_root / "synthetic_regime_summary.csv",
            result_root / "synthetic_regime_summary.json",
        )
        cond_rows = _conditioned_vs_unconditioned_rows(raw_rows)
        if cond_rows:
            _write_rows(
                cond_rows,
                result_root / "conditioned_vs_unconditioned_summary.csv",
                result_root / "conditioned_vs_unconditioned_summary.json",
            )

    overall = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_name": str(args.protocol_name),
        "status": "validated" if args.validate_split_only else "completed",
        "num_geometry_rows": len(geometry_rows),
        "num_raw_rows": len(raw_rows),
        "result_root": str(result_root),
    }
    overall_path = result_root / "synthetic_regime_overall.json"
    overall_path.write_text(json.dumps(make_jsonable(overall), indent=2, sort_keys=True))
    if args.print_json:
        print(json.dumps(make_jsonable(overall), indent=2, sort_keys=True))
    else:
        print(
            f"{overall['status']}: geometry_rows={overall['num_geometry_rows']} "
            f"raw_rows={overall['num_raw_rows']} result_root={result_root}"
        )


if __name__ == "__main__":
    main()
