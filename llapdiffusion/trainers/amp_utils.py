"""PyTorch AMP helpers shared by training loops."""

from __future__ import annotations

import torch


def create_grad_scaler(*, enabled: bool, device: torch.device):
    """Create a gradient scaler for the active device."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover


def autocast_for_device(*, enabled: bool, device: torch.device):
    """Return an autocast context for the active device."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)  # pragma: no cover
