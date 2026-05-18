from __future__ import annotations

import math
from typing import Any

import torch


def masked_error_sums(pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    if pred.shape != y.shape:
        raise ValueError(f"prediction shape {tuple(pred.shape)} != target {tuple(y.shape)}")
    w = valid.to(dtype=pred.dtype)
    diff = pred - y
    return {
        "abs_sum": (diff.abs() * w).sum(),
        "sq_sum": (diff.square() * w).sum(),
        "count": w.sum(),
    }


def masked_mse(pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    sums = masked_error_sums(pred, y, valid)
    return sums["sq_sum"] / sums["count"].clamp_min(1.0)


def masked_mae(pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    sums = masked_error_sums(pred, y, valid)
    return sums["abs_sum"] / sums["count"].clamp_min(1.0)


def sample_crps_sums(
    samples: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    *,
    pair_samples: int = 200,
) -> dict[str, torch.Tensor]:
    point = samples.mean(dim=0)
    point_sums = masked_error_sums(point, y, valid)
    w = valid.to(dtype=samples.dtype)
    term1 = (samples - y.unsqueeze(0)).abs().mean(dim=0)
    sample_count = samples.shape[0]
    if sample_count <= 1:
        term2 = torch.zeros_like(term1)
    else:
        pair_count = sample_count * (sample_count - 1) // 2
        if pair_count <= max(1, int(pair_samples)):
            pairs = torch.triu_indices(sample_count, sample_count, offset=1, device=samples.device)
            diffs = (samples[pairs[0]] - samples[pairs[1]]).abs()
        else:
            draws = int(min(max(1, int(pair_samples)), pair_count))
            i = torch.randint(0, sample_count, (draws,), device=samples.device)
            j = torch.randint(0, sample_count - 1, (draws,), device=samples.device)
            j = j + (j >= i).to(j.dtype)
            diffs = (samples[i] - samples[j]).abs()
        term2 = diffs.mean(dim=0)
    return {
        **point_sums,
        "crps_sum": ((term1 - 0.5 * term2) * w).sum(),
    }


def sample_crps(samples: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sums = sample_crps_sums(samples, y, valid)
    crps = sums["crps_sum"] / sums["count"].clamp_min(1.0)
    mse = sums["sq_sum"] / sums["count"].clamp_min(1.0)
    return crps, mse


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None
