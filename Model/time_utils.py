from __future__ import annotations

import torch


def relative_time_offsets(dt: torch.Tensor, *, time_dim: int = 1, tol: float = 1e-6) -> torch.Tensor:
    """Convert time metadata to window-local relative offsets.

    LLapDiff dataset metadata stores ``delta_t`` and ``delta_t_y`` as offsets
    from the first timestamp in the corresponding window. Older callers may
    still pass increments, so preserve offset-style monotone series that start
    at zero and otherwise fall back to cumulative increments.
    """

    if dt.dim() < 2:
        raise ValueError(f"dt must have at least 2 dims with a time dimension, got {tuple(dt.shape)}")

    time_dim = int(time_dim)
    if time_dim < 0:
        time_dim += dt.dim()
    if not 0 <= time_dim < dt.dim():
        raise ValueError(f"time_dim={time_dim} outside dt rank {dt.dim()}")

    moved = dt.movedim(time_dim, 1) if time_dim != 1 else dt
    if moved.size(1) == 0:
        return dt

    starts_at_zero = moved[:, :1].abs() <= float(tol)
    if moved.size(1) > 1:
        nondecreasing = (moved[:, 1:] - moved[:, :-1] >= -float(tol)).all(dim=1, keepdim=True)
    else:
        nondecreasing = torch.ones_like(starts_at_zero, dtype=torch.bool)
    is_offset = starts_at_zero & nondecreasing

    rel_offsets = moved - moved[:, :1]
    rel_increments = moved.cumsum(dim=1)
    rel_increments = rel_increments - rel_increments[:, :1]
    out = torch.where(is_offset, rel_offsets, rel_increments)
    return out.movedim(1, time_dim) if time_dim != 1 else out
