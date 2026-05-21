from __future__ import annotations

import inspect
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from llapdiffusion.configs.dataset_defaults import get_dataset_preset
from llapdiffusion.configs.dataset_registry import resolve_run_experiment


_COPIED_CACHES: dict[tuple[str, str, str, int], Path] = {}


def load_dataset_loaders(
    dataset_key: str,
    *,
    allow_cache_copy: bool,
    work_cache_dir: Path | None,
    horizon: int | None = None,
):
    preset = get_dataset_preset(dataset_key)
    data_dir = Path(preset.data_dir)
    requested_horizon = max(preset.horizons) if horizon is None else int(horizon)
    if requested_horizon not in preset.horizons:
        raise ValueError(f"{dataset_key}: horizon={requested_horizon} not in supported horizons {preset.horizons}")
    horizon = requested_horizon
    window = preset.context_length
    split_policy = str(getattr(preset, "split_policy", "global_purged_horizon"))
    split_scope = str(getattr(preset, "split_scope", "global_target_time"))
    exact_timestamp_batches = bool(getattr(preset, "exact_timestamp_batches", True))
    copied_cache = False
    reindex = False

    if dataset_key == "noaa_us":
        meta_path = data_dir / "cache_ratio_index" / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("horizon", 0)) < horizon:
            if not allow_cache_copy:
                raise RuntimeError(f"noaa_us H={horizon} requires --allow-cache-copy and --work-cache-dir")
            if work_cache_dir is None:
                raise RuntimeError("noaa_us copied cache reindex requires --work-cache-dir")
            root = Path(work_cache_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            cache_key = (dataset_key, str(data_dir.resolve()), str(root), horizon)
            copy_dir = _COPIED_CACHES.get(cache_key)
            if copy_dir is None:
                copy_dir = root / f"noaa_us_h{horizon}_{os.getpid()}_{time.time_ns()}"
                shutil.copytree(data_dir, copy_dir)
                _COPIED_CACHES[cache_key] = copy_dir
            data_dir = copy_dir
            copied_cache = True
            reindex = True

    if dataset_key in {"us_equity", "crypto"}:
        from llapdiffusion.datasets.fin_dataset import run_experiment
    else:
        run_experiment = resolve_run_experiment(data_dir)

    kwargs = {
        "data_dir": str(data_dir),
        "K": window,
        "H": horizon,
        "ratios": (0.7, 0.1, 0.2),
        "per_asset": True,
        "date_batching": True,
        "coverage": 0.0,
        "dates_per_batch": preset.table_batch_size,
        "batch_size": preset.table_batch_size,
        "norm": "train_only",
        "reindex": reindex,
        "split_policy": split_policy,
        "exact_timestamp_batches": exact_timestamp_batches,
    }
    sig = inspect.signature(run_experiment)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    train_dl, val_dl, test_dl, lengths = run_experiment(**filtered)
    meta = json.loads((data_dir / "cache_ratio_index" / "meta.json").read_text(encoding="utf-8"))
    info = {
        "dataset": dataset_key,
        "data_dir": str(data_dir),
        "copied_cache": copied_cache,
        "lengths": [int(x) for x in lengths],
        "window": window,
        "horizon": horizon,
        "assets": len(meta.get("assets", [])),
        "feature_cols": meta.get("feature_cols", []),
        "target_col": meta.get("target_col", ""),
        "split_policy": split_policy,
        "split_scope": split_scope,
        "batching_policy": "exact_context_end_timestamp" if exact_timestamp_batches else "calendar_day",
    }
    return (train_dl, val_dl, test_dl), info


def batch_to_device(batch, device: torch.device):
    (V, T), y, meta = batch
    meta_out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in meta.items()}
    return (V.to(device), T.to(device)), y.to(device), meta_out


def canonical_x_obs(meta: dict[str, Any], V: torch.Tensor) -> torch.Tensor:
    mask = meta.get("x_obs_mask")
    if mask is None:
        return torch.isfinite(V)
    mask = mask.to(device=V.device, dtype=torch.bool)
    if mask.shape == V.shape:
        return mask
    if mask.shape == V.shape[:-1]:
        return mask.unsqueeze(-1).expand_as(V)
    raise ValueError(f"x_obs_mask shape {tuple(mask.shape)} incompatible with V {tuple(V.shape)}")


def target_mask(meta: dict[str, Any], y: torch.Tensor) -> torch.Tensor:
    entity = meta["entity_mask"].to(device=y.device, dtype=torch.bool)
    observed = meta.get("y_obs_mask")
    if observed is None:
        observed = torch.isfinite(y)
    else:
        observed = observed.to(device=y.device, dtype=torch.bool)
    return entity.unsqueeze(-1) & observed & torch.isfinite(y)


def target_index(dataset_info: dict[str, Any]) -> int:
    cols = list(dataset_info.get("feature_cols") or [])
    target = str(dataset_info.get("target_col") or "")
    if target not in cols:
        raise ValueError(f"{dataset_info.get('dataset')}: target_col {target!r} not found in feature_cols")
    return cols.index(target)


def regular_feature_target_index(dataset_info: dict[str, Any]) -> int:
    if str(dataset_info.get("input_policy", "target_only")).lower() == "target_only":
        return 0
    return target_index(dataset_info)


def context_target_mask(meta: dict[str, Any], V: torch.Tensor, dataset_info: dict[str, Any]) -> torch.Tensor:
    idx = target_index(dataset_info)
    x_obs = canonical_x_obs(meta, V)
    if idx >= x_obs.shape[-1]:
        raise ValueError(f"{dataset_info.get('dataset')}: target index {idx} outside context feature mask")
    entity = meta["entity_mask"].to(device=V.device, dtype=torch.bool)
    return entity & x_obs[..., idx].any(dim=-1)


def find_batch(
    loader,
    dataset_info: dict[str, Any],
    device: torch.device,
):
    skipped = 0
    for raw in loader:
        batch = batch_to_device(raw, device)
        valid = target_mask(batch[2], batch[1])
        if valid.any():
            return batch, skipped
        skipped += 1
    raise RuntimeError(f"{dataset_info['dataset']}: no valid batch found")
