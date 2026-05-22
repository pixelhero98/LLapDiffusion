from __future__ import annotations

import inspect
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from llapdiffusion.benchmark_protocol import split_protocol_metadata
from llapdiffusion.configs.dataset_defaults import get_dataset_preset
from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.datasets.target_selection import resolve_target_selection


_COPIED_CACHES: dict[tuple[str, str, str, int, tuple[str, ...]], Path] = {}


def load_dataset_loaders(
    dataset_key: str,
    *,
    allow_cache_copy: bool,
    work_cache_dir: Path | None,
    horizon: int | None = None,
    target_col: str | None = None,
    target_cols: tuple[str, ...] | list[str] | None = None,
    coverage: float = 0.0,
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
    coverage = float(coverage)
    if not 0.0 <= coverage < 1.0:
        raise ValueError("coverage must be in the half-open interval [0, 1)")
    requested_target_cols = tuple(str(col).strip() for col in (target_cols or ()) if str(col).strip())
    if target_col and requested_target_cols:
        raise ValueError("Use either target_col or target_cols, not both.")

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
            cache_key = (dataset_key, str(data_dir.resolve()), str(root), horizon, requested_target_cols)
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

    sig = inspect.signature(run_experiment)
    if requested_target_cols and len(requested_target_cols) > 1 and "target_cols" not in sig.parameters:
        raise RuntimeError(
            f"{dataset_key}: multi-target baseline runs require dataset loader support for target_cols."
        )
    loader_target_col = target_col
    if requested_target_cols and len(requested_target_cols) == 1 and "target_cols" not in sig.parameters:
        loader_target_col = requested_target_cols[0]
    if requested_target_cols and "target_cols" in sig.parameters:
        reindex = True
        if not copied_cache and allow_cache_copy and work_cache_dir is not None:
            root = Path(work_cache_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            cache_key = (dataset_key, str(data_dir.resolve()), str(root), horizon, requested_target_cols)
            copy_dir = _COPIED_CACHES.get(cache_key)
            if copy_dir is None:
                tag = "_".join(col.lower().replace("/", "_") for col in requested_target_cols[:3])
                copy_dir = root / f"{dataset_key}_h{horizon}_{tag}_{os.getpid()}_{time.time_ns()}"
                shutil.copytree(data_dir, copy_dir)
                _COPIED_CACHES[cache_key] = copy_dir
            data_dir = copy_dir
            copied_cache = True

    kwargs = {
        "data_dir": str(data_dir),
        "K": window,
        "H": horizon,
        "ratios": (0.7, 0.1, 0.2),
        "per_asset": True,
        "date_batching": True,
        "coverage": coverage,
        "dates_per_batch": preset.table_batch_size,
        "batch_size": preset.table_batch_size,
        "norm": "train_only",
        "reindex": reindex,
        "split_policy": split_policy,
        "exact_timestamp_batches": exact_timestamp_batches,
        "target_col": loader_target_col,
        "target_cols": requested_target_cols or None,
    }
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    train_dl, val_dl, test_dl, lengths = run_experiment(**filtered)
    meta = json.loads((data_dir / "cache_ratio_index" / "meta.json").read_text(encoding="utf-8"))
    target_selection = resolve_target_selection(meta, target_col, requested_target_cols=requested_target_cols or None)
    info = {
        "dataset": dataset_key,
        "data_dir": str(data_dir),
        "copied_cache": copied_cache,
        "lengths": [int(x) for x in lengths],
        "window": window,
        "horizon": horizon,
        "assets": len(meta.get("assets", [])),
        "feature_cols": meta.get("feature_cols", []),
        "target_col": target_selection.target_col,
        "target_cols": list(target_selection.target_cols),
        "target_index": target_selection.target_index,
        "target_indices": list(target_selection.target_indices),
        "target_dim": target_selection.target_dim,
        "target_source": target_selection.target_source,
        "requested_target_col": target_selection.requested_target_col,
        "requested_target_cols": list(target_selection.requested_target_cols or []),
        "calendar_feature_cols": list(target_selection.calendar_feature_cols),
        "split_policy": split_policy,
        "split_scope": split_scope,
        "batching_policy": "exact_context_end_timestamp" if exact_timestamp_batches else "calendar_day",
        "coverage": coverage,
        **split_protocol_metadata(
            dataset_key,
            split_policy=split_policy,
            split_scope=split_scope,
        ),
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
    while observed.ndim < y.ndim:
        observed = observed.unsqueeze(-1)
    for _ in range(max(0, y.ndim - entity.ndim)):
        entity = entity.unsqueeze(-1)
    return entity & observed & torch.isfinite(y)


def target_cols(dataset_info: dict[str, Any]) -> tuple[str, ...]:
    cols = tuple(str(col) for col in (dataset_info.get("target_cols") or []) if str(col))
    if cols:
        return cols
    target = str(dataset_info.get("target_col") or "")
    return (target,) if target else ()


def target_indices(dataset_info: dict[str, Any]) -> tuple[int, ...]:
    feature_cols = list(dataset_info.get("feature_cols") or [])
    cols = target_cols(dataset_info)
    if not cols:
        raise ValueError(f"{dataset_info.get('dataset')}: target_cols is empty")
    missing = [col for col in cols if col not in feature_cols]
    if missing:
        raise ValueError(f"{dataset_info.get('dataset')}: target columns {missing!r} not found in feature_cols")
    resolved = dataset_info.get("target_indices")
    if resolved is not None:
        indices = tuple(int(idx) for idx in resolved)
        if len(indices) == len(cols) and all(0 <= idx < len(feature_cols) and feature_cols[idx] == col for idx, col in zip(indices, cols)):
            return indices
    if len(cols) == 1:
        resolved_one = dataset_info.get("target_index")
        if resolved_one is not None:
            idx = int(resolved_one)
            if 0 <= idx < len(feature_cols) and feature_cols[idx] == cols[0]:
                return (idx,)
    return tuple(feature_cols.index(col) for col in cols)


def target_index(dataset_info: dict[str, Any]) -> int:
    indices = target_indices(dataset_info)
    if len(indices) != 1:
        raise ValueError(f"{dataset_info.get('dataset')}: target_index is scalar-only; use target_indices for multi-target runs")
    return indices[0]


def regular_feature_target_indices(dataset_info: dict[str, Any]) -> tuple[int, ...]:
    if str(dataset_info.get("input_policy", "target_only")).lower() == "target_only":
        return tuple(range(len(target_indices(dataset_info))))
    return target_indices(dataset_info)


def regular_feature_target_index(dataset_info: dict[str, Any]) -> int:
    indices = regular_feature_target_indices(dataset_info)
    if len(indices) != 1:
        raise ValueError(f"{dataset_info.get('dataset')}: regular_feature_target_index is scalar-only")
    return indices[0]


def context_target_mask(meta: dict[str, Any], V: torch.Tensor, dataset_info: dict[str, Any]) -> torch.Tensor:
    indices = target_indices(dataset_info)
    x_obs = canonical_x_obs(meta, V)
    if any(idx >= x_obs.shape[-1] for idx in indices):
        raise ValueError(f"{dataset_info.get('dataset')}: target indices {indices} outside context feature mask")
    entity = meta["entity_mask"].to(device=V.device, dtype=torch.bool)
    selected = x_obs.index_select(-1, torch.as_tensor(indices, device=V.device))
    return entity & selected.any(dim=-1).any(dim=-1)


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
