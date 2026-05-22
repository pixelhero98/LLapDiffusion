"""Synthetic regime-shift dataset utilities for LLapDiff robustness checks."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from llapdiffusion.datasets._normalization import NormalizationStatsAccumulator
from llapdiffusion.datasets._types import PathLike
from llapdiffusion.datasets.fin_dataset import (
    CachePaths,
    _validate_context_missingness_rate,
    rebuild_window_index_only,
    load_dataloaders_with_ratio_split as _load_fin_ratio_split,
)


DATASET_NAME = "synthetic_regime"
TARGET_COLUMN = "signal"
DEFAULT_FREQ = "1h"
MAX_WINDOW = 96
MAX_HORIZON = 48


@dataclass
class SyntheticRegimeCacheConfig:
    task: str = "synthetic_freq_shift"
    window: int = MAX_WINDOW
    horizon: int = MAX_HORIZON
    data_dir: PathLike = "./synthetic_regime_cache"
    num_entities: int = 64
    series_length: int = 288
    change_point: int = 216
    freq: str = DEFAULT_FREQ
    seed: int = 20260327
    normalize_per_entity: bool = True
    overwrite: bool = False
    amplitude_min: float = 0.8
    amplitude_max: float = 1.2
    baseline_min: float = -0.1
    baseline_max: float = 0.1
    frequency_min: float = 1.0 / 48.0
    frequency_max: float = 1.0 / 24.0
    decay_min: float = 0.002
    decay_max: float = 0.01
    freq_multiplier: float = 2.0
    decay_multiplier: float = 2.5
    noise_std: float = 0.05
    phase_min: float = 0.0
    phase_max: float = 2.0 * np.pi
    keep_time_meta: str = "end"


def _validate_task(task: str) -> str:
    value = str(task).strip().lower()
    if value not in {"synthetic_freq_shift", "synthetic_decay_shift"}:
        raise ValueError(
            f"Unsupported synthetic task '{task}'. "
            "Expected one of {'synthetic_freq_shift', 'synthetic_decay_shift'}."
        )
    return value


def _generate_signal(cfg: SyntheticRegimeCacheConfig, rng: np.random.Generator) -> np.ndarray:
    amplitude = float(rng.uniform(cfg.amplitude_min, cfg.amplitude_max))
    baseline = float(rng.uniform(cfg.baseline_min, cfg.baseline_max))
    phase0 = float(rng.uniform(cfg.phase_min, cfg.phase_max))
    base_frequency = float(rng.uniform(cfg.frequency_min, cfg.frequency_max))
    base_decay = float(rng.uniform(cfg.decay_min, cfg.decay_max))

    frequency = np.full((cfg.series_length,), base_frequency, dtype=np.float32)
    decay = np.full((cfg.series_length,), base_decay, dtype=np.float32)
    if cfg.task == "synthetic_freq_shift":
        frequency[cfg.change_point :] = np.float32(float(cfg.freq_multiplier) * base_frequency)
    else:
        decay[cfg.change_point :] = np.float32(float(cfg.decay_multiplier) * base_decay)

    phase_path = phase0 + np.cumsum((2.0 * np.pi * frequency).astype(np.float64))
    envelope = np.exp(-np.cumsum(decay.astype(np.float64)))
    noise = rng.normal(0.0, cfg.noise_std, size=(cfg.series_length,)).astype(np.float64)
    signal = baseline + amplitude * envelope * np.sin(phase_path) + noise
    return signal.astype(np.float32, copy=False)


def prepare_synthetic_regime_cache(cfg: SyntheticRegimeCacheConfig) -> Mapping[str, object]:
    cfg.task = _validate_task(cfg.task)
    if cfg.window > MAX_WINDOW or cfg.horizon > MAX_HORIZON:
        raise ValueError(
            f"Requested window/horizon ({cfg.window}, {cfg.horizon}) exceed "
            f"the supported synthetic maxima ({MAX_WINDOW}, {MAX_HORIZON})."
        )
    if cfg.series_length < (cfg.window + cfg.horizon):
        raise ValueError(
            "series_length must be at least window + horizon "
            f"({cfg.window + cfg.horizon})."
        )
    if not (0 < cfg.change_point < cfg.series_length):
        raise ValueError("change_point must lie strictly inside the generated series.")

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    paths = CachePaths.from_dir(data_dir)
    if cfg.overwrite and paths.cache_root.exists():
        shutil.rmtree(paths.cache_root)
    paths.ensure()

    rng = np.random.default_rng(int(cfg.seed))
    assets = [f"entity_{idx:03d}" for idx in range(int(cfg.num_entities))]
    asset_to_id = {asset: idx for idx, asset in enumerate(assets)}
    feature_cols = [TARGET_COLUMN]

    norm_acc = NormalizationStatsAccumulator(
        num_assets=len(assets),
        feature_dim=len(feature_cols),
        per_asset=bool(cfg.normalize_per_entity),
    )

    start_time = np.datetime64("2000-01-01T00:00:00")
    pairs: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    generation_rows: List[Dict[str, object]] = []

    for asset in assets:
        aid = asset_to_id[asset]
        signal = _generate_signal(cfg, rng)
        features = signal.reshape(-1, 1).astype(np.float32, copy=False)
        targets = signal.astype(np.float32, copy=False)
        times = start_time + np.arange(cfg.series_length).astype("timedelta64[h]")
        obs_mask = np.ones_like(features, dtype=bool)
        fill_mask = np.ones_like(features, dtype=bool)

        np.save(paths.features / f"{aid}.npy", features.astype(np.float16, copy=False))
        np.save(paths.targets / f"{aid}.npy", targets.astype(np.float16, copy=False))
        np.save(paths.times / f"{aid}.npy", times.astype("datetime64[ns]"))
        np.save(paths.obs_masks / f"{aid}.npy", obs_mask)
        np.save(paths.fill_masks / f"{aid}.npy", fill_mask)
        norm_acc.update(aid, features, targets)

        total_rows = int(features.shape[0])
        max_start = total_rows - (cfg.window + cfg.horizon) + 1
        starts = np.arange(0, max_start, dtype=np.int32)
        end_times = times[starts + cfg.window - 1]
        pairs.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        ends.append(end_times.astype("datetime64[ns]"))
        generation_rows.append(
            {
                "asset": asset,
                "asset_id": aid,
                "series_length": total_rows,
                "change_point": int(cfg.change_point),
            }
        )

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)
    end_times = np.concatenate(ends, axis=0).astype("datetime64[ns]")
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", end_times)

    with paths.norm_stats.open("w") as f:
        json.dump(norm_acc.finalize(assets), f)

    meta = {
        "dataset": DATASET_NAME,
        "task": cfg.task,
        "assets": assets,
        "asset2id": asset_to_id,
        "feature_cols": feature_cols,
        "target_col": TARGET_COLUMN,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "max_window": int(MAX_WINDOW),
        "max_horizon": int(MAX_HORIZON),
        "freq": str(cfg.freq),
        "keep_time_meta": str(cfg.keep_time_meta),
        "normalize_per_entity": bool(cfg.normalize_per_entity),
        "series_length": int(cfg.series_length),
        "change_point": int(cfg.change_point),
        "seed": int(cfg.seed),
        "generation_rows": generation_rows,
        "generation_config": {
            "amplitude_min": float(cfg.amplitude_min),
            "amplitude_max": float(cfg.amplitude_max),
            "baseline_min": float(cfg.baseline_min),
            "baseline_max": float(cfg.baseline_max),
            "frequency_min": float(cfg.frequency_min),
            "frequency_max": float(cfg.frequency_max),
            "decay_min": float(cfg.decay_min),
            "decay_max": float(cfg.decay_max),
            "freq_multiplier": float(cfg.freq_multiplier),
            "decay_multiplier": float(cfg.decay_multiplier),
            "noise_std": float(cfg.noise_std),
        },
    }
    with paths.meta.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "data_dir": str(data_dir),
        "task": cfg.task,
        "num_entities": int(cfg.num_entities),
        "series_length": int(cfg.series_length),
        "window_count": int(global_pairs.shape[0]),
        "change_point": int(cfg.change_point),
    }


def _validate_cache(paths: CachePaths) -> Dict[str, object]:
    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at '{paths.meta}'. Did you run prepare_synthetic_regime_cache()?"
        )
    with paths.meta.open("r") as f:
        meta = json.load(f)
    if meta.get("dataset") != DATASET_NAME:
        raise ValueError(
            f"The cache at '{paths.cache_root}' does not correspond to the synthetic regime dataset."
        )
    return meta


def run_experiment(
    data_dir: PathLike,
    K: Optional[int] = None,
    H: Optional[int] = None,
    *,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    per_asset: bool = True,
    date_batching: bool = True,
    coverage: float = 0.0,
    panel_coverage: float = 0.0,
    dates_per_batch: int = 16,
    batch_size: int = 16,
    norm: str = "train_only",
    reindex: bool = True,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    split_policy: str = "global_purged_horizon",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
):
    coverage = _validate_context_missingness_rate(coverage, name="coverage")
    paths = CachePaths.from_dir(data_dir)
    meta = _validate_cache(paths)
    cached_window = int(meta.get("window", MAX_WINDOW))
    cached_horizon = int(meta.get("horizon", MAX_HORIZON))
    max_window = int(meta.get("max_window", MAX_WINDOW))
    max_horizon = int(meta.get("max_horizon", MAX_HORIZON))

    if K is None:
        K = cached_window
    if H is None:
        H = cached_horizon
    K = int(K)
    H = int(H)

    if K > max_window or H > max_horizon:
        raise ValueError(
            "Requested (window, horizon) exceed the cached configuration. "
            "Re-run prepare_synthetic_regime_cache with larger values first."
        )

    needs_reindex = K != cached_window or H != cached_horizon
    needs_target_reindex = target_col is not None or target_cols is not None

    if reindex and (needs_reindex or needs_target_reindex):
        rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=needs_reindex,
            backup_old=False,
            target_col=target_col,
            target_cols=target_cols,
        )

    train_dl, val_dl, test_dl, lengths = _load_fin_ratio_split(
        data_dir=str(data_dir),
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        batch_size=batch_size,
        regression=True,
        per_asset=per_asset,
        norm_scope=norm,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        coverage_per_window=panel_coverage,
        date_batching=date_batching,
        dates_per_batch=dates_per_batch,
        window=K,
        horizon=H,
        split_policy=split_policy,
        exact_timestamp_batches=exact_timestamp_batches,
        target_col=target_col,
        target_cols=target_cols,
        coverage=coverage,
    )
    return train_dl, val_dl, test_dl, lengths


def _resolve_pairs_and_window(dataset) -> Tuple[np.ndarray, int]:
    if isinstance(dataset, Subset):
        base_pairs, window = _resolve_pairs_and_window(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return np.asarray(base_pairs)[indices], int(window)
    if not hasattr(dataset, "pairs") or not hasattr(dataset, "window"):
        raise TypeError("dataset must expose 'pairs' and 'window'.")
    return np.asarray(dataset.pairs), int(dataset.window)


def build_context_end_eval_loader(
    test_loader: DataLoader,
    *,
    min_context_end: Optional[int] = None,
    max_context_end: Optional[int] = None,
) -> DataLoader:
    dataset = getattr(test_loader, "dataset", None)
    if dataset is None:
        raise TypeError("test_loader must expose a dataset.")

    pairs, window = _resolve_pairs_and_window(dataset)
    end_indices = pairs[:, 1].astype(np.int64) + int(window) - 1
    keep_mask = np.ones(end_indices.shape[0], dtype=bool)
    if min_context_end is not None:
        keep_mask &= end_indices >= int(min_context_end)
    if max_context_end is not None:
        keep_mask &= end_indices <= int(max_context_end)
    keep = np.nonzero(keep_mask)[0]
    if keep.size == 0:
        raise RuntimeError(
            "No evaluation windows found for the requested context-end slice "
            f"[{min_context_end}, {max_context_end}]."
        )

    subset = Subset(dataset, keep.tolist())
    return DataLoader(
        subset,
        batch_size=getattr(test_loader, "batch_size", None) or 1,
        shuffle=False,
        num_workers=int(getattr(test_loader, "num_workers", 0)),
        pin_memory=bool(getattr(test_loader, "pin_memory", False)),
        collate_fn=getattr(test_loader, "collate_fn", None),
        drop_last=False,
    )


def build_regime_eval_loader(
    test_loader: DataLoader,
    *,
    change_point: int,
    lookback_steps: int = 12,
) -> DataLoader:
    lo = int(change_point) - int(lookback_steps)
    hi = int(change_point) - 1
    return build_context_end_eval_loader(
        test_loader,
        min_context_end=lo,
        max_context_end=hi,
    )


__all__ = [
    "DATASET_NAME",
    "DEFAULT_FREQ",
    "MAX_HORIZON",
    "MAX_WINDOW",
    "SyntheticRegimeCacheConfig",
    "build_context_end_eval_loader",
    "build_regime_eval_loader",
    "prepare_synthetic_regime_cache",
    "run_experiment",
]
