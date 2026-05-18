from __future__ import annotations

import math
from typing import Any

import torch

from llapdiffusion.baselines.data import canonical_x_obs, target_index, target_mask


def time_features(meta: dict[str, Any], V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = meta["delta_t"].to(device=V.device, dtype=V.dtype)
    denom = dt.amax(dim=-1, keepdim=True).clamp_min(1.0)
    t = (dt / denom).clamp(0.0, 1.0)
    gap = torch.zeros_like(dt)
    gap[..., 1:] = (dt[..., 1:] - dt[..., :-1]).clamp_min(0.0)
    gap = gap / gap.amax(dim=-1, keepdim=True).clamp_min(1.0)
    dt_y = meta["delta_t_y"].to(device=V.device, dtype=V.dtype)
    yden = dt_y.amax(dim=-1, keepdim=True).clamp_min(1.0)
    ty = (dt_y / yden).clamp(0.0, 1.0)
    return t, gap, ty


def progressing_context_time(meta: dict[str, Any], V: torch.Tensor) -> torch.Tensor:
    t, _, _ = time_features(meta, V)
    K = V.shape[-2]
    index_time = torch.linspace(0.0, 1.0, K, device=V.device, dtype=V.dtype).reshape(1, 1, K)
    has_progress = (t[..., 1:] - t[..., :-1]).abs().sum(dim=-1, keepdim=True) > 0
    return torch.where(has_progress, t, index_time.expand_as(t))


def regular_features(batch, dataset_info: dict[str, Any]) -> torch.Tensor:
    (V, T), _, meta = batch
    V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    x_obs = canonical_x_obs(meta, V).to(dtype=V.dtype)
    obs_frac = x_obs.mean(dim=-1, keepdim=True)
    t, gap, _ = time_features(meta, V)
    return torch.cat(
        [
            V,
            T,
            t.unsqueeze(-1),
            torch.sin(2 * math.pi * t).unsqueeze(-1),
            torch.cos(2 * math.pi * t).unsqueeze(-1),
            gap.unsqueeze(-1),
            obs_frac,
        ],
        dim=-1,
    )


def target_context(batch, dataset_info: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (V, _), y, meta = batch
    idx = min(target_index(dataset_info), V.shape[-1] - 1)
    x = V[..., idx]
    x_obs = canonical_x_obs(meta, V)
    mask = x_obs[..., idx] if x_obs.shape == V.shape else x_obs[..., 0]
    mask = mask & meta["entity_mask"].to(device=V.device, dtype=torch.bool).unsqueeze(-1)
    valid = target_mask(meta, y)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), mask, torch.nan_to_num(y, nan=0.0), valid
