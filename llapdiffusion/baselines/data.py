from __future__ import annotations

import inspect
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from llapdiffusion.configs.dataset_defaults import get_dataset_preset
from llapdiffusion.configs.dataset_registry import resolve_run_experiment


_COPIED_CACHES: dict[tuple[str, str, str, int], Path] = {}


def load_dataset_loaders(
    dataset_key: str,
    *,
    allow_cache_copy: bool,
    work_cache_dir: Path | None,
    split: str = "all",
    horizon: int | None = None,
):
    preset = get_dataset_preset(dataset_key)
    data_dir = Path(preset.data_dir)
    requested_horizon = max(preset.horizons) if horizon is None else int(horizon)
    if requested_horizon not in preset.horizons:
        raise ValueError(f"{dataset_key}: horizon={requested_horizon} not in supported horizons {preset.horizons}")
    horizon = requested_horizon
    window = preset.context_length
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
        "dates_per_batch": 1,
        "batch_size": preset.table_batch_size,
        "norm": "train_only",
        "reindex": reindex,
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
    }
    if split == "train":
        return train_dl, info
    if split == "val":
        return val_dl, info
    if split == "test":
        return test_dl, info
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


def context_target_mask(meta: dict[str, Any], V: torch.Tensor, dataset_info: dict[str, Any]) -> torch.Tensor:
    idx = target_index(dataset_info)
    x_obs = canonical_x_obs(meta, V)
    if idx >= x_obs.shape[-1]:
        raise ValueError(f"{dataset_info.get('dataset')}: target index {idx} outside context feature mask")
    entity = meta["entity_mask"].to(device=V.device, dtype=torch.bool)
    return entity & x_obs[..., idx].any(dim=-1)


def slice_entities(batch, indices: Sequence[int]):
    (V, T), y, meta = batch
    B, N = V.shape[:2]
    idx = torch.as_tensor(list(indices), dtype=torch.long, device=V.device)
    if idx.numel() == 0:
        raise ValueError("At least one entity index is required")
    if bool((idx < 0).any()) or bool((idx >= N).any()):
        raise IndexError(f"Entity indices {indices} outside batch entity dimension {N}")

    def maybe_slice(value):
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == B and value.shape[1] == N:
            return value.index_select(1, idx.to(value.device))
        return value

    sliced_meta = {k: maybe_slice(v) for k, v in meta.items()}
    return (
        (V.index_select(1, idx), T.index_select(1, idx.to(T.device))),
        y.index_select(1, idx.to(y.device)),
        sliced_meta,
    )


def normalize_cap(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    cap = int(value)
    if cap < 0:
        raise ValueError(f"{name} must be non-negative; use 0 for no cap")
    return None if cap == 0 else cap


def select_entities(batch, max_entities: int | None):
    max_entities = normalize_cap(max_entities, "max_entities")
    if max_entities is None:
        return batch, None
    (V, T), y, meta = batch
    _, N = V.shape[:2]
    cap = min(int(max_entities), N)
    if cap == N:
        return batch, list(range(N))
    x_obs = canonical_x_obs(meta, V)
    entity = meta["entity_mask"].to(dtype=torch.bool, device=V.device)
    counts = x_obs.sum(dim=tuple(i for i in range(x_obs.dim()) if i != 1))
    counts = counts * entity.to(dtype=counts.dtype).sum(dim=0).clamp(max=1)
    idx = torch.topk(counts, k=cap, largest=True).indices.sort().values
    selected = [int(i) for i in idx.detach().cpu().tolist()]
    return slice_entities(batch, selected), selected


def select_stable_entities(
    loaders,
    dataset_info: dict[str, Any],
    device: torch.device,
    max_entities: int | None,
    max_batches: int | None,
) -> list[int] | None:
    max_entities = normalize_cap(max_entities, "max_entities")
    max_batches = normalize_cap(max_batches, "max_batches")
    if max_entities is None:
        return None
    split_counts = []
    for loader in loaders:
        counts = None
        for i, raw in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch_to_device(raw, device)
            (V, _), _, meta = batch
            valid_context = context_target_mask(meta, V, dataset_info)
            current = valid_context.sum(dim=0).to(dtype=torch.float32)
            counts = current if counts is None else counts + current
        if counts is not None:
            split_counts.append(counts)

    if not split_counts:
        raise RuntimeError(f"{dataset_info['dataset']}: no batches available for stable entity selection")

    cap = min(int(max_entities), split_counts[0].numel())
    common_scores = torch.stack(split_counts).amin(dim=0)
    candidate = torch.nonzero(common_scores > 0, as_tuple=False).flatten()
    if candidate.numel() >= cap:
        local = torch.topk(common_scores.index_select(0, candidate), k=cap, largest=True).indices
        idx = candidate.index_select(0, local).sort().values
    else:
        aggregate = torch.stack(split_counts).sum(dim=0)
        idx = torch.topk(aggregate, k=cap, largest=True).indices.sort().values
    return [int(i) for i in idx.detach().cpu().tolist()]


def find_batch(
    loader,
    dataset_info: dict[str, Any],
    device: torch.device,
    max_entities: int | None,
    max_batches: int | None,
    *,
    require_future_target: bool = True,
    selected_entities: Sequence[int] | None = None,
):
    max_entities = normalize_cap(max_entities, "max_entities")
    max_batches = normalize_cap(max_batches, "max_batches")
    skipped = 0
    for i, raw in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch_to_device(raw, device)
        if selected_entities is None and max_entities is not None:
            batch, selected = select_entities(batch, max_entities)
        else:
            if selected_entities is None:
                selected = None
            else:
                batch = slice_entities(batch, selected_entities)
                selected = list(selected_entities)
        valid = target_mask(batch[2], batch[1]) if require_future_target else context_target_mask(batch[2], batch[0][0], dataset_info)
        if valid.any():
            return batch, selected, skipped
        skipped += 1
    scanned = "all" if max_batches is None else str(max_batches)
    raise RuntimeError(f"{dataset_info['dataset']}: no valid batch found in first {scanned} batches")
