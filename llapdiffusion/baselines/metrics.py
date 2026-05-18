from __future__ import annotations

import math
from typing import Any

import torch


def masked_mse(pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if pred.shape != y.shape:
        raise ValueError(f"prediction shape {tuple(pred.shape)} != target {tuple(y.shape)}")
    w = valid.to(dtype=pred.dtype)
    return (((pred - y) ** 2) * w).sum() / w.sum().clamp_min(1.0)


def masked_mae(pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if pred.shape != y.shape:
        raise ValueError(f"prediction shape {tuple(pred.shape)} != target {tuple(y.shape)}")
    w = valid.to(dtype=pred.dtype)
    return ((pred - y).abs() * w).sum() / w.sum().clamp_min(1.0)


def sample_crps(samples: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    point = samples.mean(dim=0)
    mse = masked_mse(point, y, valid)
    w = valid.to(dtype=samples.dtype)
    term1 = (samples - y.unsqueeze(0)).abs().mean(dim=0)
    if samples.shape[0] > 1:
        diffs = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean(dim=(0, 1))
    else:
        diffs = torch.zeros_like(term1)
    crps = ((term1 - 0.5 * diffs) * w).sum() / w.sum().clamp_min(1.0)
    return crps, mse


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
