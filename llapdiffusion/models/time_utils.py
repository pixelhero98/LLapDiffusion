from __future__ import annotations

import torch


def relative_time_offsets(
    dt: torch.Tensor,
    *,
    time_dim: int = 1,
    tol: float = 1e-6,
    recenter: bool = True,
) -> torch.Tensor:
    """Validate monotone time metadata and optionally recenter to the first timestamp."""

    if dt.dim() < 2:
        raise ValueError(f"dt must have at least 2 dims with a time dimension, got {tuple(dt.shape)}")
    if not torch.isfinite(dt).all():
        raise ValueError("dt must contain only finite values")

    time_dim = int(time_dim)
    if time_dim < 0:
        time_dim += dt.dim()
    if not 0 <= time_dim < dt.dim():
        raise ValueError(f"time_dim={time_dim} outside dt rank {dt.dim()}")

    moved = dt.movedim(time_dim, 1) if time_dim != 1 else dt
    if moved.size(1) == 0:
        return dt

    if moved.size(1) > 1:
        if not (moved[:, 1:] - moved[:, :-1] >= -float(tol)).all():
            raise ValueError("dt must be nondecreasing along the time dimension")

    out = moved - moved[:, :1] if recenter else moved
    return out.movedim(1, time_dim) if time_dim != 1 else out
