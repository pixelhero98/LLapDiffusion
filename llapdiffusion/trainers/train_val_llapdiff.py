"""Train/eval entrypoint for LLapDiff with set-VAE targets."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from llapdiffusion.configs import config
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from llapdiffusion.benchmark_protocol import llapdiff_protocol_metadata, split_protocol_metadata
from llapdiffusion.configs.dataset_registry import resolve_run_experiment
from llapdiffusion.logging_utils import is_debug, is_verbose, progress_iter, progress_task
from llapdiffusion.diffusion_cache import (
    DiffusionSplitCache,
    build_or_load_diffusion_input_cache,
)
from llapdiffusion.latent_space.latent_vae import LatentVAE
from llapdiffusion.models.summarizer import LaplaceAE
from llapdiffusion.models.llapdiff import LLapDiff
from llapdiffusion.models.llapdiff_utils import (
    EMA,
    set_torch,
    encode_mu_norm,
    make_lr_scheduler,
    calculate_target_variance,
    compute_latent_stats,
    diffusion_loss,
    build_context,
    normalize_cond_per_batch,
    pack_targets_tokens,
    infer_target_dim_from_loader,
    sample_training_timesteps,
    decode_latents_with_vae,
    target_time_observed,
    targets_to_bhnc,
    vae_io_dims_for_target_dim,
)
from llapdiffusion.target_artifacts import (
    loader_target_request_from_config,
    target_metadata_from_config,
    unwrap_checkpoint_model,
    validate_checkpoint_target_metadata,
)
from llapdiffusion.models.time_utils import relative_time_offsets


LoaderTuple = Tuple[DataLoader, DataLoader, DataLoader]


def _cfg_value(config_obj: object, *names: str, default):
    for name in names:
        if hasattr(config_obj, name):
            return getattr(config_obj, name)
    return default


def _make_grad_scaler(*, enabled: bool, device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover


def _autocast_context(*, enabled: bool, device: torch.device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)  # pragma: no cover


def _nan_to_num(x: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _is_finite_tensor(x: Optional[torch.Tensor]) -> bool:
    return x is None or bool(torch.isfinite(x).all().item())


def _release_cuda_allocator(device: torch.device) -> None:
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


def _grads_are_finite_params(params) -> bool:
    for param in params:
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            return False
    return True


def _should_skip_nonfinite_summary_ft_gradients(
    *,
    diffusion_params,
    summary_ft_params,
    amp_enabled: bool,
    summary_ft_active: bool,
    skipped_nonfinite_grad_steps: int,
    max_nonfinite_grad_steps: int,
) -> bool:
    if not (amp_enabled and summary_ft_active):
        return False
    max_nonfinite_grad_steps = max(0, int(max_nonfinite_grad_steps))
    if int(skipped_nonfinite_grad_steps) >= max_nonfinite_grad_steps:
        return False
    diffusion_params = list(diffusion_params)
    summary_ft_params = list(summary_ft_params)
    if not summary_ft_params:
        return False
    return _grads_are_finite_params(diffusion_params) and not _grads_are_finite_params(summary_ft_params)


def _entity_finite_mask(x: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(x)
    for _ in range(x.dim() - 2):
        finite = finite.all(dim=-1)
    return finite


def _canon_obs_mask_like(
    obs_mask: Optional[torch.Tensor],
    ref: torch.Tensor,
    *,
    device: torch.device,
    name: str,
) -> Optional[torch.Tensor]:
    if obs_mask is None:
        return None
    mask = torch.as_tensor(obs_mask, device=device, dtype=torch.bool)
    B, N, K, Fdim = ref.shape
    if mask.shape == (B, N, K, Fdim):
        return mask
    if mask.shape == (B, K, N, Fdim):
        return mask.permute(0, 2, 1, 3).contiguous()
    if mask.shape == (B, N, K, 1):
        return mask.expand(B, N, K, Fdim)
    if mask.shape == (B, K, N, 1):
        return mask.permute(0, 2, 1, 3).contiguous().expand(B, N, K, Fdim)
    if mask.shape == (B, N, K):
        return mask.unsqueeze(-1).expand(B, N, K, Fdim)
    if mask.shape == (B, K, N):
        return mask.permute(0, 2, 1).contiguous().unsqueeze(-1).expand(B, N, K, Fdim)
    raise ValueError(f"{name} shape {tuple(mask.shape)} is incompatible with tensor shape {tuple(ref.shape)}")


def _canon_target_obs_mask_like(
    obs_mask: Optional[torch.Tensor],
    yb: torch.Tensor,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if obs_mask is None:
        return None
    mask = torch.as_tensor(obs_mask, device=device, dtype=torch.bool)
    if yb.dim() == 3:
        B, N, H = yb.shape
        if mask.shape == (B, N, H):
            return mask
        if mask.shape == (B, H, N):
            return mask.permute(0, 2, 1).contiguous()
        if mask.shape == (B, N, H, 1):
            return mask.squeeze(-1)
        if mask.shape == (B, H, N, 1):
            return mask.squeeze(-1).permute(0, 2, 1).contiguous()
        raise ValueError(f"y_obs_mask shape {tuple(mask.shape)} is incompatible with target shape {tuple(yb.shape)}")

    if yb.dim() != 4:
        raise ValueError(f"target shape must be [B,N,H] or [B,N,H,C], got {tuple(yb.shape)}")
    B, N, H, C = yb.shape
    if mask.shape == (B, N, H, C):
        return mask
    if mask.shape == (B, H, N, C):
        return mask.permute(0, 2, 1, 3).contiguous()
    if mask.shape == (B, N, H, 1):
        return mask.expand(B, N, H, C)
    if mask.shape == (B, H, N, 1):
        return mask.permute(0, 2, 1, 3).contiguous().expand(B, N, H, C)
    if mask.shape == (B, N, H):
        return mask.unsqueeze(-1).expand(B, N, H, C)
    if mask.shape == (B, H, N):
        return mask.permute(0, 2, 1).contiguous().unsqueeze(-1).expand(B, N, H, C)
    raise ValueError(f"y_obs_mask shape {tuple(mask.shape)} is incompatible with target shape {tuple(yb.shape)}")


def _raise_if_observed_nonfinite(name: str, tensor: torch.Tensor, observed: torch.Tensor) -> None:
    bad = observed.to(device=tensor.device, dtype=torch.bool) & ~torch.isfinite(tensor)
    if bad.any():
        raise ValueError(f"{name} contains non-finite values marked as observed")


def _sanitize_batch(
    xb: Tuple[torch.Tensor, torch.Tensor],
    yb: torch.Tensor,
    meta: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Validate observed values, fill masked missing values, and build the entity mask."""

    V_raw, T_raw = xb
    V = V_raw.to(device)
    T = T_raw.to(device)
    yb = yb.to(device)
    mask = meta["entity_mask"].to(device=device, dtype=torch.bool)

    x_obs = _canon_obs_mask_like(meta.get("x_obs_mask"), V, device=device, name="x_obs_mask")
    entity_obs = mask[:, :, None, None].expand_as(V)
    observed = x_obs & entity_obs if x_obs is not None else entity_obs
    _raise_if_observed_nonfinite("V", V, observed)
    _raise_if_observed_nonfinite("T", T, observed)

    y_obs = _canon_target_obs_mask_like(meta.get("y_obs_mask"), yb, device=device)
    if y_obs is not None:
        entity_target_mask = mask[:, :, None]
        if yb.dim() == 4:
            entity_target_mask = entity_target_mask.unsqueeze(-1)
        _raise_if_observed_nonfinite("yb", yb, y_obs & entity_target_mask)

    if x_obs is None:
        mask = mask & _entity_finite_mask(V) & _entity_finite_mask(T)
    else:
        has_observed = x_obs.any(dim=(2, 3))
        mask = mask & has_observed

    return (_nan_to_num(V), _nan_to_num(T)), _nan_to_num(yb), mask



def _flatten_dt(
    meta: Dict[str, object],
    mask_bn: torch.Tensor,
    device: torch.device,
    *,
    key: str,
) -> Optional[torch.Tensor]:
    """
    Reduce per-entity time offsets to a per-batch query grid.

    The dataloader/collate provides context ``delta_t`` relative to the first context timestamp and
    target/query ``delta_t_y`` relative to the last context timestamp, both as [B,N,L]. The set-VAE
    + diffusion operate at the batch level (B), so we collapse the entity dimension via a masked mean
    over entities present in ``mask_bn`` without re-centering. Callers pass a context/entity-valid mask;
    this function must not inspect future target values or target masks.
    """
    delta_t = meta.get(key)
    if delta_t is None:
        return None

    dt = torch.as_tensor(delta_t, dtype=torch.float32, device=device)
    if dt.dim() == 4:
        if dt.size(-1) != 1:
            raise ValueError(f"{key} with 4 dims must have a singleton final dimension, got {tuple(dt.shape)}")
        dt = dt.squeeze(-1)
    if dt.dim() != 3:
        raise ValueError(f"{key} must have shape [B, N, L] or [B, N, L, 1], got {tuple(dt.shape)}")

    B, N, _ = dt.shape
    m = mask_bn.to(device=device, dtype=torch.bool)
    if m.shape != (B, N):
        raise ValueError(f"{key} batch/entity shape {tuple(dt.shape[:2])} does not match mask shape {tuple(m.shape)}")

    valid_dt = dt[m]
    if valid_dt.numel() and not torch.isfinite(valid_dt).all():
        raise ValueError(f"{key} contains non-finite values for valid entities")
    if key == "delta_t_y":
        for batch_idx in range(B):
            valid = m[batch_idx]
            if int(valid.sum().item()) <= 1:
                continue
            grids = dt[batch_idx, valid]
            if not torch.allclose(grids, grids[:1], rtol=1e-5, atol=1e-6):
                raise ValueError(
                    "delta_t_y must use the same query grid for every valid entity in a batch; "
                    "per-entity future query grids are not supported by the shared latent diffusion model"
                )
    w = m.to(dtype=dt.dtype).unsqueeze(-1)
    dt = torch.where(m.unsqueeze(-1), dt, torch.zeros_like(dt))
    denom = w.sum(dim=1).clamp(min=1.0)
    return (dt * w).sum(dim=1) / denom  # [B,L]


def _match_dt_to_horizon(
    dt_flat: Optional[torch.Tensor], horizon: int
) -> Optional[torch.Tensor]:
    """Return validated per-batch time metadata for a latent horizon."""
    if dt_flat is None:
        return None
    if dt_flat.dim() != 2:
        raise ValueError(f"dt metadata must have shape [B, H], got {tuple(dt_flat.shape)}")
    if dt_flat.size(1) != int(horizon):
        raise ValueError(f"dt horizon mismatch: expected H={int(horizon)}, got {dt_flat.size(1)}")
    return dt_flat



class ContextStatsAdapter(nn.Module):
    """Small residual adapter that lets diffusion training reshape frozen context tokens."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        stat_dim: int = 3,
        adapter_dim: int = 128,
        dropout: float = 0.0,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.cond_norm = nn.LayerNorm(hidden_dim)
        self.stat_proj = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, hidden_dim),
        )
        nn.init.normal_(self.stat_proj[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.stat_proj[-1].bias)
        nn.init.normal_(self.fuse[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, cond_summary: torch.Tensor, stat_tokens: torch.Tensor) -> torch.Tensor:
        if cond_summary.dim() != 3:
            raise ValueError(f"cond_summary must be [B,S,H], got {tuple(cond_summary.shape)}")
        if stat_tokens.dim() != 3:
            raise ValueError(f"stat_tokens must be [B,K,C], got {tuple(stat_tokens.shape)}")

        pooled_stats = F.adaptive_avg_pool1d(
            stat_tokens.transpose(1, 2), cond_summary.size(1)
        ).transpose(1, 2)
        stat_emb = self.stat_proj(pooled_stats)
        fused = torch.cat([self.cond_norm(cond_summary), stat_emb], dim=-1)
        delta = self.fuse(fused)
        return cond_summary + self.residual_scale * delta


def _history_stat_tokens(
    V: torch.Tensor,
    T: torch.Tensor,
    mask_bn: torch.Tensor,
    device: torch.device,
    *,
    dt: Optional[torch.Tensor] = None,
    x_obs_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build lightweight per-step history statistics for the condition adapter."""

    B, N, K, Fdim = V.shape
    mask = mask_bn.to(device=device, dtype=torch.bool)
    entity_feat_mask = mask[:, :, None, None].to(dtype=V.dtype)

    denom_feat = (mask.to(dtype=V.dtype).sum(dim=1, keepdim=True) * float(Fdim)).clamp_min(1.0)
    abs_t = (T.abs() * entity_feat_mask).sum(dim=(1, 3)) / denom_feat

    if x_obs_mask is None:
        obs_frac = torch.ones((B, K), device=device, dtype=V.dtype)
    else:
        obs = torch.as_tensor(x_obs_mask, device=device)
        if obs.dim() == 4:
            if obs.size(1) == N and obs.size(2) == K:
                obs = obs.permute(0, 2, 1, 3).contiguous()
            elif obs.size(1) == K and obs.size(2) == N:
                obs = obs.contiguous()
            else:
                raise ValueError(f"Unrecognized x_obs_mask shape: {tuple(obs.shape)}")
            obs = obs.to(dtype=V.dtype)
            valid = mask[:, None, :, None].to(dtype=obs.dtype).expand(B, K, N, obs.size(-1))
            obs = obs * valid
            denom = valid.sum(dim=(2, 3)).clamp_min(1.0)
            obs_frac = obs.sum(dim=(2, 3)) / denom
        elif obs.dim() == 3:
            if obs.size(1) == N and obs.size(2) == K:
                obs = obs.permute(0, 2, 1).contiguous()
            elif obs.size(1) == K and obs.size(2) == N:
                obs = obs.contiguous()
            else:
                raise ValueError(f"Unrecognized x_obs_mask shape: {tuple(obs.shape)}")
            obs = obs.to(dtype=V.dtype)
            valid = mask[:, None, :].to(dtype=obs.dtype).expand(B, K, N)
            obs = obs * valid
            denom = valid.sum(dim=2).clamp_min(1.0)
            obs_frac = obs.sum(dim=2) / denom
        else:
            raise ValueError(f"x_obs_mask must have 3 or 4 dims, got {tuple(obs.shape)}")

    if dt is None:
        rel_t_unit = torch.linspace(0.0, 1.0, steps=K, device=device, dtype=V.dtype).unsqueeze(0).expand(B, -1)
    else:
        dt_raw = torch.as_tensor(dt, dtype=V.dtype, device=device)
        if dt_raw.dim() == 4 and dt_raw.size(-1) == 1:
            dt_raw = dt_raw.squeeze(-1)
        if dt_raw.dim() == 2:
            dt_flat = dt_raw
        elif dt_raw.dim() == 3:
            if dt_raw.size(1) == N and dt_raw.size(2) == K:
                dt_bnK = dt_raw
            elif dt_raw.size(1) == K and dt_raw.size(2) == N:
                dt_bnK = dt_raw.permute(0, 2, 1).contiguous()
            else:
                raise ValueError(f"Unrecognized delta_t shape: {tuple(dt_raw.shape)}")
            w = mask.to(dtype=dt_bnK.dtype).unsqueeze(-1)
            denom = w.sum(dim=1).clamp_min(1.0)
            dt_flat = (dt_bnK * w).sum(dim=1) / denom
        else:
            raise ValueError(f"delta_t must have 2 or 3 dims, got {tuple(dt_raw.shape)}")

        if not torch.isfinite(dt_flat).all():
            raise ValueError("delta_t contains non-finite values")
        rel_t = relative_time_offsets(dt_flat, time_dim=1)
        rel_t_unit = rel_t / rel_t.amax(dim=1, keepdim=True).clamp_min(1.0)

    stats = torch.stack(
        [
            torch.nan_to_num(abs_t, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(obs_frac, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(rel_t_unit, nan=0.0, posinf=0.0, neginf=0.0),
        ],
        dim=-1,
    )
    return stats


def _build_cond_summary(
    summarizer: nn.Module,
    diff_model: Optional[nn.Module],
    V: torch.Tensor,
    T: torch.Tensor,
    mask_bn: torch.Tensor,
    device: torch.device,
    *,
    dt: Optional[torch.Tensor] = None,
    x_obs_mask: Optional[torch.Tensor] = None,
    norm: bool = True,
    requires_grad: bool = False,
    summary_base_raw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    adapter = getattr(diff_model, "cond_adapter", None) if diff_model is not None else None
    if summary_base_raw is None:
        cond_summary_raw = build_context(
            summarizer,
            V,
            T,
            mask_bn,
            device,
            dt=dt,
            x_obs_mask=x_obs_mask,
            norm=False,
            requires_grad=requires_grad,
        )
    else:
        if requires_grad:
            raise RuntimeError("cached summarizer outputs cannot be used while summarizer gradients are active")
        cond_summary_raw = summary_base_raw.to(device=device, dtype=V.dtype)
    if adapter is not None:
        stats = _history_stat_tokens(V, T, mask_bn, device, dt=dt, x_obs_mask=x_obs_mask)
        cond_summary_raw = adapter(cond_summary_raw, stats)
    cond_summary = normalize_cond_per_batch(cond_summary_raw) if norm else cond_summary_raw
    return cond_summary


def _build_cond_summary_pair(
    summarizer: nn.Module,
    diff_model: Optional[nn.Module],
    V: torch.Tensor,
    T: torch.Tensor,
    mask_bn: torch.Tensor,
    device: torch.device,
    *,
    dt: Optional[torch.Tensor] = None,
    x_obs_mask: Optional[torch.Tensor] = None,
    norm: bool = True,
    requires_grad: bool = False,
    summary_base_raw: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    adapter = getattr(diff_model, "cond_adapter", None) if diff_model is not None else None
    if summary_base_raw is None:
        cond_summary_raw = build_context(
            summarizer,
            V,
            T,
            mask_bn,
            device,
            dt=dt,
            x_obs_mask=x_obs_mask,
            norm=False,
            requires_grad=requires_grad,
        )
    else:
        if requires_grad:
            raise RuntimeError("cached summarizer outputs cannot be used while summarizer gradients are active")
        cond_summary_raw = summary_base_raw.to(device=device, dtype=V.dtype)
    if adapter is not None:
        stats = _history_stat_tokens(V, T, mask_bn, device, dt=dt, x_obs_mask=x_obs_mask)
        cond_summary_raw = adapter(cond_summary_raw, stats)
    cond_summary = normalize_cond_per_batch(cond_summary_raw) if norm else cond_summary_raw
    return cond_summary, cond_summary_raw


def _build_pole_cond_vec(
    diff_model: nn.Module,
    t_vec: torch.Tensor,
    *,
    cond_summary: Optional[torch.Tensor] = None,
    cond_summary_raw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    backbone = getattr(diff_model, "model", None)
    if backbone is not None and hasattr(backbone, "make_pole_cond"):
        return backbone.make_pole_cond(
            t_vec,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
        )

    if cond_summary is not None:
        summary_pool = cond_summary.mean(dim=1)
    elif cond_summary_raw is not None:
        summary_pool = cond_summary_raw.mean(dim=1)
    else:
        summary_pool = torch.zeros_like(t_vec)
    return torch.cat([t_vec, summary_pool], dim=-1)


def _ensure_loaders(
    train_dl: Optional[DataLoader],
    val_dl: Optional[DataLoader],
    test_dl: Optional[DataLoader],
    sizes: Optional[Sequence[int]],
    config=config,
) -> Tuple[LoaderTuple, Optional[Tuple[int, int, int]]]:
    if any(loader is None for loader in (train_dl, val_dl, test_dl)):
        run_experiment = resolve_run_experiment(config.DATA_DIR)
        target_col, target_cols = loader_target_request_from_config(config)
        batch_size = int(getattr(config, "BATCH_SIZE", getattr(config, "DATES_PER_BATCH", 1)))
        train_dl, val_dl, test_dl, sizes = run_experiment(
            data_dir=config.DATA_DIR,
            date_batching=config.date_batching,
            dates_per_batch=batch_size,
            K=config.WINDOW,
            H=config.PRED,
            coverage=config.COVERAGE,
            batch_size=batch_size,
            ratios=(config.train_ratio, config.val_ratio, config.test_ratio),
            split_policy=getattr(config, "split_policy", "global_purged_horizon"),
            exact_timestamp_batches=bool(getattr(config, "exact_timestamp_batches", True)),
            target_col=target_col,
            target_cols=target_cols,
        )
    elif sizes is None:
        try:
            sizes = tuple(len(dl.dataset) for dl in (train_dl, val_dl, test_dl))
        except Exception:
            sizes = None

    if train_dl is None or val_dl is None or test_dl is None:
        raise RuntimeError("Failed to obtain train/val/test dataloaders.")

    return (train_dl, val_dl, test_dl), sizes


def _load_module_state(module: nn.Module, state_dict: Dict[str, torch.Tensor], *, strict: bool = True) -> None:
    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    missing = list(missing)
    unexpected = list(unexpected)
    if missing:
        print(f"[load] missing keys for {module.__class__.__name__}: {missing}")
    if unexpected:
        print(f"[load] unexpected keys for {module.__class__.__name__}: {unexpected}")


def _init_pole_probe(
    diff_model: nn.Module,
    summarizer: nn.Module,
    train_dl: DataLoader,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Capture a fixed conditioning batch so we can track whether learned poles move."""
    analysis = getattr(getattr(diff_model, "model", None), "analysis", None)
    if analysis is None or not hasattr(analysis, "effective_poles"):
        return None

    for xb, yb, meta in train_dl:
        (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue

        cond_summary, cond_summary_raw = _build_cond_summary_pair(
            summarizer,
            diff_model,
            V,
            T,
            mask_bn,
            device,
            dt=meta.get("delta_t"),
            x_obs_mask=meta.get("x_obs_mask"),
        )
        if not _is_finite_tensor(cond_summary):
            continue

        B = cond_summary.size(0)
        probe_t = torch.full(
            (B,),
            max(1, int(diff_model.scheduler.timesteps // 2)),
            device=device,
            dtype=torch.long,
        )
        dtype = cond_summary.dtype
        t_vec = diff_model._time_embed(probe_t).to(dtype)
        cond_vec = _build_pole_cond_vec(
            diff_model,
            t_vec,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
        )
        base_rho0, base_omega0 = analysis._base_poles(dtype, device)
        eff_rho0, eff_omega0 = analysis.effective_poles(B, dtype, device, cond=cond_vec)
        return {
            "cond_summary": cond_summary.detach(),
            "cond_summary_raw": cond_summary_raw.detach(),
            "probe_t": probe_t.detach(),
            "init_base_rho": base_rho0.detach().clone(),
            "init_base_omega": base_omega0.detach().clone(),
            "init_eff_rho": eff_rho0.detach().clone(),
            "init_eff_omega": eff_omega0.detach().clone(),
        }
    return None


@torch.no_grad()
def _collect_pole_probe(
    diff_model: nn.Module,
    probe_state: Optional[Dict[str, torch.Tensor]],
) -> Optional[Dict[str, float]]:
    if not probe_state:
        return None

    analysis = getattr(getattr(diff_model, "model", None), "analysis", None)
    if analysis is None:
        return None

    cond_summary = probe_state["cond_summary"]
    cond_summary_raw = probe_state.get("cond_summary_raw")
    probe_t = probe_state["probe_t"]
    dtype = cond_summary.dtype
    device = cond_summary.device

    base_rho, base_omega = analysis._base_poles(dtype, device)
    t_vec = diff_model._time_embed(probe_t).to(dtype)
    cond_vec = _build_pole_cond_vec(
        diff_model,
        t_vec,
        cond_summary=cond_summary,
        cond_summary_raw=cond_summary_raw,
    )
    eff_rho, eff_omega = analysis.effective_poles(probe_t.numel(), dtype, device, cond=cond_vec)

    return {
        "base_rho_mean": float(base_rho.mean().item()),
        "base_omega_mean": float(base_omega.mean().item()),
        "eff_rho_mean": float(eff_rho.mean().item()),
        "eff_omega_mean": float(eff_omega.mean().item()),
        "base_rho_delta_mean": float((base_rho - probe_state["init_base_rho"]).abs().mean().item()),
        "base_omega_delta_mean": float((base_omega - probe_state["init_base_omega"]).abs().mean().item()),
        "eff_rho_delta_mean": float((eff_rho - probe_state["init_eff_rho"]).abs().mean().item()),
        "eff_omega_delta_mean": float((eff_omega - probe_state["init_eff_omega"]).abs().mean().item()),
    }


def _latent_targets_for_batch(
    vae: nn.Module,
    yb: torch.Tensor,
    mask_bn: torch.Tensor,
    meta: Dict[str, torch.Tensor],
    device: torch.device,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
    *,
    cached_batch=None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if cached_batch is not None and cached_batch.mu_norm is not None and cached_batch.obs_any is not None:
        return cached_batch.mu_norm, cached_batch.obs_any

    x_tok, entity_pad, obs = pack_targets_tokens(
        yb, mask_bn, device, y_obs_mask=meta.get("y_obs_mask")
    )
    if x_tok is None or obs is None or not obs.any():
        return None, None

    obs_any = target_time_observed(obs)
    mu_norm = encode_mu_norm(
        vae, x_tok, entity_pad=entity_pad, mu_mean=mu_mean, mu_std=mu_std
    )
    mu_norm = mu_norm * obs_any.unsqueeze(-1).to(dtype=mu_norm.dtype)
    return mu_norm, obs_any


@torch.no_grad()
def _collect_latent_probe(
    vae: nn.Module,
    train_dl: DataLoader,
    device: torch.device,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
    *,
    max_batches: int = 4,
    input_cache: Optional[DiffusionSplitCache] = None,
) -> Optional[Dict[str, float]]:
    """Estimate whether normalized latent targets are actually centered/scaled as expected."""
    vals = []
    batches = 0
    if input_cache is not None:
        input_cache.reset()
    for xb, yb, meta in train_dl:
        (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue
        cached_batch = (
            input_cache.next_batch(
                meta,
                device=device,
                mu_mean=mu_mean,
                mu_std=mu_std,
                load_latents=True,
                load_summary=False,
            )
            if input_cache is not None
            else None
        )
        mu_norm, obs_any = _latent_targets_for_batch(
            vae,
            yb,
            mask_bn,
            meta,
            device,
            mu_mean,
            mu_std,
            cached_batch=cached_batch,
        )
        if mu_norm is None or obs_any is None or not obs_any.any():
            continue
        mu_obs = mu_norm[obs_any]
        if mu_obs.numel() == 0:
            continue
        vals.append(mu_obs.detach().cpu())
        batches += 1
        if batches >= max_batches:
            break

    if not vals:
        return None

    flat = torch.cat(vals, dim=0).float()
    feat_std = flat.std(dim=0, unbiased=False)
    abs_flat = flat.abs()
    return {
        "num_tokens": int(flat.shape[0]),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "abs_mean": float(abs_flat.mean().item()),
        "abs_p95": float(torch.quantile(abs_flat.reshape(-1), 0.95).item()),
        "feat_std_mean": float(feat_std.mean().item()),
        "feat_std_min": float(feat_std.min().item()),
        "feat_std_max": float(feat_std.max().item()),
    }


def _init_timestep_loss_meter(num_bins: int, total_timesteps: int) -> Dict[str, object]:
    bins = max(1, int(num_bins))
    return {
        "num_bins": bins,
        "total_timesteps": max(1, int(total_timesteps)),
        "raw_sum": torch.zeros(bins, dtype=torch.float64),
        "weighted_sum": torch.zeros(bins, dtype=torch.float64),
        "count": torch.zeros(bins, dtype=torch.float64),
    }


def _update_timestep_loss_meter(
    meter: Dict[str, object],
    t: torch.Tensor,
    raw_per_sample: torch.Tensor,
    weighted_per_sample: torch.Tensor,
) -> None:
    num_bins = int(meter["num_bins"])
    total_timesteps = int(meter["total_timesteps"])
    t_cpu = t.detach().long().cpu()
    raw_cpu = raw_per_sample.detach().to(dtype=torch.float64).cpu()
    weighted_cpu = weighted_per_sample.detach().to(dtype=torch.float64).cpu()
    bin_ids = torch.div(t_cpu * num_bins, max(1, total_timesteps), rounding_mode="floor")
    bin_ids = bin_ids.clamp(min=0, max=num_bins - 1)
    meter["count"] += torch.bincount(bin_ids, minlength=num_bins).to(dtype=torch.float64)
    meter["raw_sum"] += torch.bincount(bin_ids, weights=raw_cpu, minlength=num_bins).to(dtype=torch.float64)
    meter["weighted_sum"] += torch.bincount(bin_ids, weights=weighted_cpu, minlength=num_bins).to(dtype=torch.float64)


def _finalize_timestep_loss_meter(meter: Dict[str, object]) -> Dict[str, object]:
    count = meter["count"].clone()
    denom = count.clamp_min(1.0)
    raw = (meter["raw_sum"] / denom).tolist()
    weighted = (meter["weighted_sum"] / denom).tolist()
    return {
        "counts": [int(x) for x in count.tolist()],
        "raw": [float(x) for x in raw],
        "weighted": [float(x) for x in weighted],
    }


def _init_snr_bin_meter(
    num_bins: int,
    *,
    min_log_snr: float = -12.0,
    max_log_snr: float = 12.0,
) -> Dict[str, object]:
    bins = max(1, int(num_bins))
    return {
        "num_bins": bins,
        "min_log_snr": float(min_log_snr),
        "max_log_snr": float(max_log_snr),
        "raw_sum": torch.zeros(bins, dtype=torch.float64),
        "count": torch.zeros(bins, dtype=torch.float64),
    }


def _update_snr_bin_meter(
    meter: Dict[str, object],
    scheduler,
    t: torch.Tensor,
    raw_per_sample: torch.Tensor,
) -> None:
    num_bins = int(meter["num_bins"])
    min_log_snr = float(meter["min_log_snr"])
    max_log_snr = float(meter["max_log_snr"])

    log_snr = torch.log(scheduler.snr_at(t).clamp_min(1e-12)).detach().cpu().to(torch.float64)
    raw_cpu = raw_per_sample.detach().cpu().to(torch.float64)
    span = max(max_log_snr - min_log_snr, 1e-8)
    pos = (log_snr - min_log_snr) / span
    bin_ids = torch.floor(pos * num_bins).long().clamp(min=0, max=num_bins - 1)

    meter["count"] += torch.bincount(bin_ids, minlength=num_bins).to(torch.float64)
    meter["raw_sum"] += torch.bincount(bin_ids, weights=raw_cpu, minlength=num_bins).to(torch.float64)


def _finalize_snr_bin_meter(meter: Dict[str, object]) -> Dict[str, object]:
    count = meter["count"].clone()
    denom = count.clamp_min(1.0)
    raw = (meter["raw_sum"] / denom).tolist()

    num_bins = int(meter["num_bins"])
    min_log_snr = float(meter["min_log_snr"])
    max_log_snr = float(meter["max_log_snr"])
    edges = torch.linspace(min_log_snr, max_log_snr, steps=num_bins + 1).tolist()

    return {
        "counts": [int(x) for x in count.tolist()],
        "raw": [float(x) for x in raw],
        "log_snr_edges": [float(x) for x in edges],
    }


@torch.no_grad()
def evaluate_val_diagnostics(
    diff_model: nn.Module,
    vae: nn.Module,
    summarizer: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
    config,
    *,
    ema=None,
    num_snr_bins: int = 8,
    min_log_snr: float = -12.0,
    max_log_snr: float = 12.0,
    input_cache: Optional[DiffusionSplitCache] = None,
    progress_enabled: bool = False,
    progress_label: Optional[str] = None,
) -> Dict[str, object]:
    """Return validation MSE diagnostics for the active prediction parameterization."""
    predict_type_name = str(getattr(config, "PREDICT_TYPE", "v")).strip().lower() or "v"
    diff_model.eval()
    scheduler = diff_model.scheduler

    raw_sum = 0.0
    num_samples = 0
    snr_meter = _init_snr_bin_meter(
        num_snr_bins,
        min_log_snr=min_log_snr,
        max_log_snr=max_log_snr,
    )

    use_ema = ema is not None
    if use_ema:
        ema.store(diff_model)
        ema.copy_to(diff_model)

    if input_cache is not None:
        input_cache.reset()
    batches = progress_iter(
        dataloader,
        desc=progress_label or "llapdiff val-diag",
        enabled=progress_enabled,
        unit="batch",
    )
    for xb, yb, meta in batches:
        (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue
        cached_batch = (
            input_cache.next_batch(
                meta,
                device=device,
                mu_mean=mu_mean,
                mu_std=mu_std,
                load_latents=True,
                load_summary=True,
            )
            if input_cache is not None
            else None
        )

        cond_summary, cond_summary_raw = _build_cond_summary_pair(
            summarizer,
            diff_model,
            V,
            T,
            mask_bn,
            device,
            dt=meta.get("delta_t"),
            x_obs_mask=meta.get("x_obs_mask"),
            summary_base_raw=(cached_batch.summary_raw if cached_batch is not None else None),
        )
        dt_b = _flatten_dt(
            meta,
            mask_bn,
            device,
            key="delta_t_y",
        )

        mu_norm, obs_any = _latent_targets_for_batch(
            vae,
            yb,
            mask_bn,
            meta,
            device,
            mu_mean,
            mu_std,
            cached_batch=cached_batch,
        )
        if mu_norm is None or obs_any is None or not obs_any.any():
            continue
        if not _is_finite_tensor(mu_norm):
            raise FloatingPointError("non-finite latent targets detected during validation")

        dt_model = _match_dt_to_horizon(dt_b, mu_norm.size(1))
        Beff = mu_norm.size(0)
        t = sample_training_timesteps(
            scheduler,
            Beff,
            device,
            sampler=str(getattr(config, "TRAIN_T_SAMPLER", "uniform")),
            karras_rho=float(getattr(config, "TRAIN_T_KARRAS_RHO", getattr(config, "KARRAS_RHO", 7.5))),
        )
        noise = torch.randn_like(mu_norm)
        x_t, eps_true = scheduler.q_sample(mu_norm, t, noise)

        _, stats = diffusion_loss(
            diff_model,
            scheduler,
            mu_norm,
            t,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            predict_type=predict_type_name,
            weight_scheme="none",
            minsnr_gamma=float(getattr(config, "MINSNR_GAMMA", 5.0)),
            minsnr_normalize="none",
            dt=dt_model,
            reuse_xt_eps=(x_t, eps_true),
            target_mask=obs_any,
            return_stats=True,
        )
        raw_sum += float(stats["raw_loss"].item()) * Beff
        num_samples += Beff
        _update_snr_bin_meter(snr_meter, scheduler, t, stats["per_sample_raw"])

    if use_ema:
        ema.restore(diff_model)

    if num_samples <= 0:
        raise RuntimeError("Validation diagnostic found no valid diffusion samples")
    metric_value = raw_sum / num_samples
    snr_bins = _finalize_snr_bin_meter(snr_meter)
    out = {
        "val_diag_mse_raw": metric_value,
        "val_diag_mse_by_snr_bin": snr_bins,
    }
    metric_key = f"val_{predict_type_name}_mse_raw"
    metric_bins_key = f"val_{predict_type_name}_mse_by_snr_bin"
    out[metric_key] = metric_value
    out[metric_bins_key] = snr_bins
    return out


@torch.no_grad()
def evaluate_irregular_time_checks(
    summarizer: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    diff_model: Optional[nn.Module] = None,
    max_batches: int = 4,
) -> Dict[str, float]:
    ctx_delta_no_dt = 0.0
    ctx_delta_no_xmask = 0.0
    ctx_delta_zero_tsig = 0.0
    batches = 0

    for xb, yb, meta in dataloader:
        (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta, device)
        if not mask_bn.any():
            continue

        cs_full = _build_cond_summary(
            summarizer,
            diff_model,
            V,
            T,
            mask_bn,
            device,
            dt=meta.get("delta_t"),
            x_obs_mask=meta.get("x_obs_mask"),
        )
        cs_no_dt = _build_cond_summary(
            summarizer,
            diff_model,
            V,
            T,
            mask_bn,
            device,
            dt=None,
            x_obs_mask=meta.get("x_obs_mask"),
        )
        cs_no_xmask = _build_cond_summary(
            summarizer,
            diff_model,
            V,
            T,
            mask_bn,
            device,
            dt=meta.get("delta_t"),
            x_obs_mask=None,
        )
        cs_zero_tsig = _build_cond_summary(
            summarizer,
            diff_model,
            V,
            torch.zeros_like(T),
            mask_bn,
            device,
            dt=meta.get("delta_t"),
            x_obs_mask=meta.get("x_obs_mask"),
        )

        denom = cs_full.abs().mean().clamp_min(1e-8).item()
        ctx_delta_no_dt += float((cs_full - cs_no_dt).abs().mean().item() / denom)
        ctx_delta_no_xmask += float((cs_full - cs_no_xmask).abs().mean().item() / denom)
        ctx_delta_zero_tsig += float((cs_full - cs_zero_tsig).abs().mean().item() / denom)
        batches += 1
        if batches >= max_batches:
            break

    if batches == 0:
        raise RuntimeError("Irregular-time diagnostics processed no valid batches")

    return {
        "ctx_delta_no_dt": ctx_delta_no_dt / batches,
        "ctx_delta_no_xmask": ctx_delta_no_xmask / batches,
        "ctx_delta_zero_tsig": ctx_delta_zero_tsig / batches,
    }


def _sampling_kwargs(config_obj: object, *, prefix: str = "EVAL") -> Dict[str, object]:
    def _read(name: str, *, default, aliases: Tuple[str, ...] = ()):
        names = [f"{prefix}_{name}"]
        names.extend(aliases)
        return _cfg_value(config_obj, *names, default=default)

    guidance_strength = _read(
        "GUIDANCE",
        default=2.0,
        aliases=("GUIDANCE_STRENGTH",),
    )

    return {
        "steps": int(_read("STEPS", default=36, aliases=("GEN_STEPS",))),
        "guidance_strength": guidance_strength,
        "guidance_power": float(_read("GUIDANCE_POWER", default=0.3)),
        "eta": float(_read("ETA", default=0.0, aliases=("GEN_ETA",))),
        "aggregation_method": str(_read("AGGREGATION", default="mean")),
        "quantiles": tuple(
            _cfg_value(config_obj, "EVAL_QUANTILES", default=(0.1, 0.5, 0.9))
        ),
        "dynamic_thresh_p": float(
            _cfg_value(config_obj, "DYN_THRESH_P", "DYNAMIC_THRESH_P", default=0.0)
        ),
        "dynamic_thresh_max": float(
            _cfg_value(config_obj, "DYN_THRESH_MAX", "DYNAMIC_THRESH_MAX", default=1.0)
        ),
        "rho": float(_cfg_value(config_obj, "KARRAS_RHO", default=7.5)),
    }


def _summarize_dataset(
    train_dl: DataLoader,
    sizes: Optional[Sequence[int]],
    *,
    verbose: bool = True,
) -> Tuple[int, int, int, int]:
    if verbose:
        if sizes is not None:
            print("sizes:", tuple(sizes))
        else:
            print("sizes: (unknown)")

    try:
        xb0, yb0, _ = next(iter(train_dl))
    except StopIteration as exc:  # pragma: no cover
        raise RuntimeError("Training dataloader produced no batches.") from exc

    V0, T0 = xb0
    B0, N0, K0, Fv = V0.shape
    Ft = T0.shape[-1]
    assert Fv == Ft, f"Expected Fv == Ft, got {Fv} vs {Ft}"
    if verbose:
        print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)
    return B0, N0, K0, Fv


@torch.inference_mode()
def evaluate_regression(
    diff_model,
    vae,
    summarizer,
    dataloader,
    device,
    mu_mean,
    mu_std,
    config,
    ema=None,
    self_cond: bool = False,
    disable_conditioning: bool = False,
    steps: int = 36,
    guidance_strength: Union[float, Tuple[float, float]] = 2.0,
    guidance_power: float = 0.3,
    eta: float = 1.0,
    aggregation_method: str = "mean",
    quantiles: tuple = (0.1, 0.5, 0.9),
    dynamic_thresh_p: float = 0.995,
    dynamic_thresh_max: float = 1.0,
    rho: float = 7.5,
    generator_seed: Optional[int] = None,
    crps_pair_samples: int = 200,
    verbose: bool = False,
    progress_enabled: bool = False,
    progress_label: Optional[str] = None,
):
    """
    Evaluate probabilistic forecasts in observation space (set-VAE pipeline).

    Decoding produces y_hat: [B,H,N,1]. Metrics are computed only where:
      entity_mask[b,n] == True AND y_obs_mask[b,n,h] == True (if present) AND y is finite.
    """
    if aggregation_method not in ["mean", "median"]:
        raise ValueError("aggregation_method must be either 'mean' or 'median'")
    if not all(0.0 < float(q) < 1.0 for q in quantiles):
        raise ValueError("All quantiles must be in the open interval (0, 1).")

    diff_model.eval()

    abs_sum, sq_sum, elts = 0.0, 0.0, 0.0
    crps_sum, crps_elts = 0.0, 0.0
    pinball_sums = {float(q): 0.0 for q in quantiles}
    per_target_abs_sum = None
    per_target_sq_sum = None
    per_target_elts = None

    num_samples = int(getattr(config, "NUM_EVAL_SAMPLES", 16))
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(generator_seed))
    use_ema = ema is not None
    if use_ema:
        ema.store(diff_model)
        ema.copy_to(diff_model)

    try:
        progress_total = len(dataloader) * num_samples * max(1, int(steps))
    except TypeError:
        progress_total = None

    with progress_task(
        desc=progress_label or "llapdiff eval",
        enabled=progress_enabled,
        total=progress_total,
        unit="step",
    ) as progress:
        for xb, yb, meta in dataloader:
            (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta, device)
            if not mask_bn.any():
                continue

            cond_summary, cond_summary_raw = _build_cond_summary_pair(
                summarizer,
                diff_model,
                V,
                T,
                mask_bn,
                device,
                dt=meta.get("delta_t"),
                x_obs_mask=meta.get("x_obs_mask"),
            )
            if disable_conditioning:
                cond_summary = None
                cond_summary_raw = None
            dt_b = _flatten_dt(
                meta,
                mask_bn,
                device,
                key="delta_t_y",
            )

            y_obs_mask = meta.get("y_obs_mask")
            x_tok, entity_pad, obs = pack_targets_tokens(
                yb, mask_bn, device, y_obs_mask=y_obs_mask
            )
            if x_tok is None or not obs.any():
                continue

            B, Hcur, Z = x_tok.size(0), x_tok.size(1), int(mu_mean.shape[-1])
            dt_model = _match_dt_to_horizon(dt_b, Hcur)

            y_true = torch.nan_to_num(
                targets_to_bhnc(yb, mask_bn, device=device),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if obs.dim() == 3:
                obs_entries = obs.unsqueeze(-1)
            elif obs.dim() == 4:
                obs_entries = obs
            else:
                raise ValueError(f"target observation mask must be [B,H,N] or [B,H,N,C], got {tuple(obs.shape)}")
            if obs_entries.shape != y_true.shape:
                try:
                    obs_entries = obs_entries.expand_as(y_true)
                except RuntimeError as exc:
                    raise ValueError(
                        f"target observation mask shape {tuple(obs_entries.shape)} does not match target shape {tuple(y_true.shape)}"
                    ) from exc
            valid = obs_entries.to(dtype=y_true.dtype)

            all_y_hats = []
            for _ in range(num_samples):
                x0_norm = diff_model.generate(
                    shape=(B, Hcur, Z),
                    steps=steps,
                    guidance_strength=guidance_strength,
                    guidance_power=guidance_power,
                    eta=eta,
                    cond_summary=cond_summary,
                    cond_summary_raw=cond_summary_raw,
                    self_cond=self_cond,
                    cfg_rescale=True,
                    dt=dt_model,
                    dynamic_thresh_p=dynamic_thresh_p,
                    dynamic_thresh_max=dynamic_thresh_max,
                    rho=rho,
                    generator=generator,
                )
                y_hat_sample = decode_latents_with_vae(
                    vae, x0_norm, entity_pad=entity_pad, mu_mean=mu_mean, mu_std=mu_std
                )
                all_y_hats.append(y_hat_sample)
                progress.update(max(1, int(steps)))

            all_samples = torch.stack(all_y_hats, dim=0)  # [S,B,H,N,C]

            if aggregation_method == "mean":
                point_forecast = all_samples.mean(dim=0)
            else:
                point_forecast = all_samples.median(dim=0).values

            res = (point_forecast - y_true) * valid
            abs_sum += res.abs().sum().item()
            sq_sum += (res**2).sum().item()
            elts += valid.sum().item()
            target_reduce_dims = (0, 1, 2)
            batch_abs = res.abs().sum(dim=target_reduce_dims).detach().cpu()
            batch_sq = (res**2).sum(dim=target_reduce_dims).detach().cpu()
            batch_elts = valid.sum(dim=target_reduce_dims).detach().cpu()
            if per_target_abs_sum is None:
                per_target_abs_sum = batch_abs
                per_target_sq_sum = batch_sq
                per_target_elts = batch_elts
            else:
                per_target_abs_sum += batch_abs
                per_target_sq_sum += batch_sq
                per_target_elts += batch_elts

            M = all_samples.shape[0]
            term1 = (all_samples - y_true.unsqueeze(0)).abs().mean(dim=0)
            if M <= 1:
                term2 = torch.zeros_like(term1)
            else:
                P_full = M * (M - 1) // 2
                P = int(min(max(1, crps_pair_samples), P_full))
                i = torch.randint(0, M, (P,), device=all_samples.device)
                j = torch.randint(0, M - 1, (P,), device=all_samples.device)
                j = j + (j >= i).to(j.dtype)
                diffs = (all_samples[i] - all_samples[j]).abs()
                term2 = diffs.mean(dim=0)

            crps_elem = term1 - 0.5 * term2
            crps_sum += (crps_elem * valid).sum().item()
            crps_elts += valid.sum().item()

            for q in quantiles:
                q = float(q)
                y_q = torch.quantile(all_samples, q, dim=0, interpolation="linear")
                diff = y_true - y_q
                loss_q = torch.maximum(q * diff, (q - 1.0) * diff) * valid
                pinball_sums[q] += loss_q.sum().item()

    if use_ema:
        ema.restore(diff_model)

    if elts <= 0 or crps_elts <= 0:
        raise RuntimeError("Evaluation found no valid target observations")
    mae = abs_sum / elts
    mse = sq_sum / elts
    crps = crps_sum / crps_elts
    pinball = {q: (pinball_sums[q] / elts) for q in pinball_sums.keys()}
    qs_fmt = ", ".join(f"{q:.2f}:{pinball[q]:.6f}" for q in sorted(pinball.keys()))
    if verbose:
        print(f"[eval ({num_samples} samples, aggregation: {aggregation_method})]")
        print(f" CRPS: {crps:.6f} | MAE: {mae:.6f} | MSE: {mse:.6f} | Pinball[{qs_fmt}]")
    metrics = {
        "crps": crps,
        "mae": mae,
        "mse": mse,
        "pinball": pinball,
        "num_samples": num_samples,
        "aggregation": aggregation_method,
    }
    if per_target_abs_sum is not None and int(per_target_abs_sum.numel()) > 1:
        denom = per_target_elts.clamp_min(1.0)
        target_cols = getattr(config, "TARGET_COLS", None) or getattr(config, "target_cols", None)
        if target_cols is None:
            target_cols = [f"target_{idx}" for idx in range(int(per_target_abs_sum.numel()))]
        elif isinstance(target_cols, str):
            target_cols = [part.strip() for part in target_cols.split(",") if part.strip()]
        target_cols = list(target_cols)
        while len(target_cols) < int(per_target_abs_sum.numel()):
            target_cols.append(f"target_{len(target_cols)}")
        metrics["per_target"] = {
            str(name): {
                "mae": float((per_target_abs_sum[idx] / denom[idx]).item()),
                "mse": float((per_target_sq_sum[idx] / denom[idx]).item()),
                "observed_entries": int(per_target_elts[idx].item()),
            }
            for idx, name in enumerate(target_cols[: int(per_target_abs_sum.numel())])
        }
    return metrics


def _save_checkpoint(out_path: Path, payload: Dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def _llapdiff_model_kwargs(config_obj: object) -> Dict[str, object]:
    return {
        "data_dim": int(getattr(config_obj, "VAE_LATENT_CHANNELS")),
        "hidden_dim": int(getattr(config_obj, "MODEL_WIDTH")),
        "num_layers": int(getattr(config_obj, "NUM_LAYERS")),
        "num_heads": int(getattr(config_obj, "NUM_HEADS")),
        "predict_type": str(getattr(config_obj, "PREDICT_TYPE")),
        "laplace_k": int(getattr(config_obj, "LAPLACE_K")),
        "timesteps": int(getattr(config_obj, "TIMESTEPS")),
        "schedule": str(getattr(config_obj, "SCHEDULE")),
        "dropout": float(getattr(config_obj, "DROPOUT")),
        "attn_dropout": float(getattr(config_obj, "ATTN_DROPOUT")),
        "self_conditioning": bool(getattr(config_obj, "SELF_COND")),
        "summary_pool_mode": str(getattr(config_obj, "COND_POOL_MODE", "mean")),
        "pole_pool_use_raw_summary": bool(getattr(config_obj, "COND_POOL_USE_RAW", False)),
        "block_summary_adaln": bool(getattr(config_obj, "BLOCK_SUMMARY_ADALN", False)),
        "analysis_summary_qk": bool(getattr(config_obj, "ANALYSIS_SUMMARY_QK", False)),
        "analysis_qk_use_raw_summary": bool(getattr(config_obj, "ANALYSIS_QK_USE_RAW", False)),
        "rho_conditioning_mode": str(getattr(config_obj, "RHO_CONDITIONING_MODE", "raw")),
    }


def _cond_adapter_config(config_obj: object) -> Dict[str, object]:
    mode = str(getattr(config_obj, "COND_ADAPTER_MODE", "none")).strip().lower()
    if mode not in {"none", "stats"}:
        raise ValueError(f"Unknown COND_ADAPTER_MODE '{mode}'. Use 'none' or 'stats'.")
    return {
        "mode": mode,
        "hidden_dim": int(getattr(config_obj, "SUM_CONTEXT_DIM")),
        "stat_dim": 3,
        "adapter_dim": int(getattr(config_obj, "COND_ADAPTER_HIDDEN", 128)),
        "dropout": float(getattr(config_obj, "COND_ADAPTER_DROPOUT", 0.0)),
        "residual_scale": float(getattr(config_obj, "COND_ADAPTER_SCALE", 0.1)),
    }


def _llapdiff_model_config(config_obj: object) -> Dict[str, object]:
    return {
        "llapdiff": _llapdiff_model_kwargs(config_obj),
        "cond_adapter": _cond_adapter_config(config_obj),
    }


def _llapdiff_config_from_checkpoint(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return {"rho_conditioning_mode": "legacy_effective"}
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        return {"rho_conditioning_mode": "legacy_effective"}
    llapdiff_config = model_config.get("llapdiff")
    if isinstance(llapdiff_config, dict):
        config = dict(llapdiff_config)
    else:
        config = {
            key: value
            for key, value in model_config.items()
            if key != "cond_adapter"
        }
    config.setdefault("rho_conditioning_mode", "legacy_effective")
    return config


def _llapdiff_model_kwargs_from_checkpoint(config_obj: object, payload: object) -> Dict[str, object]:
    kwargs = _llapdiff_model_kwargs(config_obj)
    kwargs.update(_llapdiff_config_from_checkpoint(payload))
    return kwargs


def build_llapdiff_model(
    config_obj: object,
    device: torch.device,
    *,
    checkpoint_payload: object | None = None,
) -> LLapDiff:
    model_kwargs = (
        _llapdiff_model_kwargs(config_obj)
        if checkpoint_payload is None
        else _llapdiff_model_kwargs_from_checkpoint(config_obj, checkpoint_payload)
    )
    model = LLapDiff(**model_kwargs).to(device)
    adapter_cfg = _cond_adapter_config(config_obj)
    if adapter_cfg["mode"] == "stats":
        model.cond_adapter = ContextStatsAdapter(
            hidden_dim=int(adapter_cfg["hidden_dim"]),
            stat_dim=int(adapter_cfg["stat_dim"]),
            adapter_dim=int(adapter_cfg["adapter_dim"]),
            dropout=float(adapter_cfg["dropout"]),
            residual_scale=float(adapter_cfg["residual_scale"]),
        ).to(device)
    return model


def _load_eval_checkpoint(
    checkpoint_path: Optional[Path],
    *,
    diff_model: nn.Module,
    ema: Optional[EMA],
    device: torch.device,
    config_obj: object,
    verbose: bool = False,
) -> Optional[str]:
    """Load a checkpoint for evaluation and return the loaded path.

    Preference order is: best checkpoint (if available) then final checkpoint.
    """
    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    payload = torch.load(checkpoint_path, map_location=device)
    validate_checkpoint_target_metadata(payload, config_obj, context="LLapDiff")
    state_dict = payload.get("model") if isinstance(payload, dict) else None
    if state_dict is None:
        print(f"[warn] checkpoint missing model weights: {checkpoint_path}")
        return None

    diff_model.load_state_dict(state_dict)
    if ema is not None and isinstance(payload, dict) and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])

    if verbose:
        print(f"[loaded for eval] {checkpoint_path}")
    return str(checkpoint_path)


def _select_eval_checkpoint_path(
    *,
    test_metric_source: str,
    val_metric_source: str,
    best_ckpt_path: Path,
    best_ckpt_path_raw: Path,
    best_ckpt_path_ema: Path,
    last_ckpt_path: Path,
) -> Optional[Path]:
    if test_metric_source == "raw":
        preferred = [best_ckpt_path_raw]
        if val_metric_source == "raw":
            preferred.append(best_ckpt_path)
    elif test_metric_source == "ema":
        preferred = [best_ckpt_path_ema]
        if val_metric_source == "ema":
            preferred.append(best_ckpt_path)
    else:
        preferred = []

    preferred.extend([best_ckpt_path, last_ckpt_path])
    for path in preferred:
        if path.exists():
            return path
    return None


def _resolve_metric_source(
    source: object,
    *,
    default: str,
    use_ema: bool,
    label: str,
) -> str:
    source_name = str(source).strip().lower()
    if source_name in {"", "auto", "default", "same"}:
        source_name = str(default).strip().lower()
    if source_name not in {"raw", "ema"}:
        raise ValueError(f"{label} must be 'raw', 'ema', 'auto', or 'same', got '{source}'.")
    if source_name == "ema" and not use_ema:
        print(f"[warn] {label} requested ema metrics but EMA is unavailable; falling back to raw")
        source_name = "raw"
    return source_name


def _maybe_metric_ema(source: str, ema: Optional[EMA]) -> Optional[EMA]:
    return ema if source == "ema" else None


def _resolve_final_test_eval_mode(value: object) -> str:
    if isinstance(value, bool):
        return "run" if value else "skip"
    mode = str(value).strip().lower()
    if mode in {"", "true", "yes", "on", "1", "run", "evaluate", "eval"}:
        return "run"
    if mode in {"false", "no", "off", "0", "skip", "skipped", "none"}:
        return "skip"
    if mode in {"defer", "deferred"}:
        return "defer"
    raise ValueError(
        "FINAL_TEST_EVAL must be one of 'run', 'skip', or 'defer' "
        f"(or a boolean), got {value!r}."
    )


def _finite_or_none(value: float) -> Optional[float]:
    return None if not math.isfinite(float(value)) else float(value)


def _sample_target_keep_mask(
    obs_any: torch.Tensor,
    *,
    mode: str,
    keep_prob: float,
    keep_stride: int,
) -> torch.Tensor:
    """Sample target-time observations for auxiliary inpainting training."""
    obs_any = torch.as_tensor(obs_any, dtype=torch.bool)
    if obs_any.dim() != 2:
        raise ValueError(f"obs_any must be [B,H], got {tuple(obs_any.shape)}")

    mode_name = str(mode).strip().lower()
    keep = torch.zeros_like(obs_any, dtype=torch.bool)

    if mode_name == "random":
        keep = (torch.rand(obs_any.shape, device=obs_any.device) < float(keep_prob)) & obs_any
    elif mode_name == "regular":
        stride = max(1, int(keep_stride))
        keep[:, ::stride] = True
        keep &= obs_any
    elif mode_name == "prefix":
        keep_prob = float(min(max(keep_prob, 0.0), 1.0))
        for b in range(obs_any.size(0)):
            idx = torch.where(obs_any[b])[0]
            if idx.numel() == 0:
                continue
            count = int(math.ceil(float(idx.numel()) * keep_prob))
            if idx.numel() >= 2:
                count = min(max(1, count), int(idx.numel()) - 1)
            else:
                count = min(int(idx.numel()), max(0, count))
            if count > 0:
                keep[b, idx[:count]] = True
    elif mode_name == "mixed":
        mode_weights = torch.tensor([0.5, 0.3, 0.2], device=obs_any.device)
        mode_choices = ("prefix", "regular", "random")
        sampled = torch.multinomial(
            mode_weights,
            num_samples=int(obs_any.size(0)),
            replacement=True,
        )
        for b, mode_idx in enumerate(sampled.tolist()):
            keep[b : b + 1] = _sample_target_keep_mask(
                obs_any[b : b + 1],
                mode=mode_choices[int(mode_idx)],
                keep_prob=keep_prob,
                keep_stride=keep_stride,
            )
    else:
        raise ValueError(
            f"Unknown TARGET_MASK_AUX_KEEP_MODE '{mode}'. Use 'random', 'regular', 'prefix', or 'mixed'."
        )

    for b in range(obs_any.size(0)):
        idx = torch.where(obs_any[b])[0]
        if idx.numel() < 2:
            keep[b].zero_()
            continue
        if keep[b].sum().item() == 0:
            keep[b, idx[0]] = True
        hidden = obs_any[b] & (~keep[b])
        if hidden.sum().item() == 0:
            keep[b, idx[-1]] = False
            if keep[b].sum().item() == 0:
                keep[b, idx[0]] = True

    return keep & obs_any


def _resolve_target_mask_aux_start_epoch(
    config_obj: object,
    *,
    aux_prob: float,
) -> int:
    raw_start = getattr(config_obj, "TARGET_MASK_AUX_START_EPOCH", 10)
    if aux_prob <= 0.0:
        return 0
    if raw_start is None:
        return max(
            1,
            int(
                math.ceil(
                    float(getattr(config_obj, "WARMUP_FRAC", 0.0))
                    * max(1, int(getattr(config_obj, "EPOCHS", 1)))
                )
            ),
        )
    start_epoch = int(raw_start or 0)
    return 1 if start_epoch <= 0 else start_epoch


def _effective_target_mask_aux_probability(config_obj: object) -> float:
    aux_prob = float(getattr(config_obj, "TARGET_MASK_AUX_P", 0.0) or 0.0)
    if aux_prob <= 0.0:
        return 0.0
    if bool(getattr(config_obj, "IMPUTATION_TRAINING", True)):
        return aux_prob
    raise ValueError(
        "TARGET_MASK_AUX_P > 0 requires IMPUTATION_TRAINING=True; "
        "set TARGET_MASK_AUX_P=0.0 for forecast-only training."
    )


def _maybe_apply_target_mask_aux(
    scheduler,
    x_t: torch.Tensor,
    mu_norm: torch.Tensor,
    obs_any: torch.Tensor,
    t: torch.Tensor,
    *,
    enabled: bool,
    keep_mode: str,
    keep_prob: float,
    keep_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Optionally convert a conditional training batch into a target-time completion batch.

    Returns:
        x_t_masked: possibly re-noised batch with observed target-time anchors injected
        target_mask: supervision mask for loss reduction (full obs_any or hidden-only)
        stats: bookkeeping for epoch logging
    """
    target_mask = obs_any
    stats = {
        "applied": False,
        "keep_frac": 1.0,
        "hidden_frac": 0.0,
    }
    if not enabled:
        return x_t, target_mask, stats

    keep_mask = _sample_target_keep_mask(
        obs_any,
        mode=keep_mode,
        keep_prob=keep_prob,
        keep_stride=keep_stride,
    )
    hidden_mask = obs_any & (~keep_mask)
    obs_count = float(obs_any.sum().item())
    keep_count = float(keep_mask.sum().item())
    hidden_count = float(hidden_mask.sum().item())

    if keep_count <= 0.0 or hidden_count <= 0.0 or obs_count <= 0.0:
        return x_t, target_mask, stats

    y_obs = mu_norm * keep_mask.unsqueeze(-1).to(dtype=mu_norm.dtype)
    x_obs_t, _ = scheduler.q_sample(y_obs, t)
    x_t_masked = torch.where(keep_mask.unsqueeze(-1), x_obs_t, x_t)
    stats["applied"] = True
    stats["keep_frac"] = keep_count / obs_count
    stats["hidden_frac"] = hidden_count / obs_count
    return x_t_masked, hidden_mask, stats


def _matches_param_prefix(name: str, prefixes: Tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _select_summarizer_finetune_named_params(
    summarizer: nn.Module,
    mode: object,
) -> List[Tuple[str, nn.Parameter]]:
    mode_name = str(mode).strip().lower()
    named_params = list(summarizer.named_parameters())
    if mode_name in {"", "none", "off", "false"}:
        return []
    if mode_name == "all":
        return named_params
    if mode_name in {"top", "upper"}:
        prefixes = (
            "history_encoder",
            "token_proj",
            "queries",
            "norm",
            "pos_embedding",
            "input_pad",
        )
        return [(name, param) for name, param in named_params if _matches_param_prefix(name, prefixes)]
    if mode_name in {"pool", "proj"}:
        prefixes = ("token_proj", "queries", "norm", "pos_embedding", "input_pad")
        return [(name, param) for name, param in named_params if _matches_param_prefix(name, prefixes)]
    raise ValueError(
        f"Unknown SUM_FT_MODE '{mode}'. Use 'none', 'pool', 'top', or 'all'."
    )


def _set_named_params_trainable(
    named_params: Sequence[Tuple[str, nn.Parameter]],
    enabled: bool,
) -> None:
    for _, param in named_params:
        param.requires_grad_(enabled)


def run(
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
    sizes: Optional[Sequence[int]] = None,
    config=config,
) -> Dict[str, object]:
    (train_dl, val_dl, test_dl), sizes = _ensure_loaders(
        train_dl, val_dl, test_dl, sizes, config
    )
    verbose = is_verbose(config)
    debug = is_debug(config)
    _, N0, K0, Fv = _summarize_dataset(train_dl, sizes, verbose=verbose)
    target_dim = int(getattr(config, "TARGET_DIM", 0) or 0)
    if target_dim <= 0:
        target_dim = infer_target_dim_from_loader(train_dl)
        setattr(config, "TARGET_DIM", target_dim)
    vae_input_dim, vae_output_dim = vae_io_dims_for_target_dim(config, target_dim)
    setattr(config, "VAE_INPUT_DIM", vae_input_dim)
    setattr(config, "VAE_OUTPUT_DIM", vae_output_dim)
    if verbose:
        print(f"LLapDiff target_dim: {target_dim}")

    device = set_torch(
        seed=int(getattr(config, "SEED", 42)),
        deterministic=bool(getattr(config, "DETERMINISTIC", False)),
    )
    if verbose:
        print(f"Using device: {device}")

    # ---------------- VAE (frozen) ----------------
    vae = LatentVAE(
        seq_len=config.PRED,
        latent_dim=config.VAE_LATENT_DIM,
        latent_channel=config.VAE_LATENT_CHANNELS,
        enc_layers=config.VAE_LAYERS,
        enc_heads=config.VAE_HEADS,
        enc_ff=config.VAE_FF,
        dec_layers=config.VAE_LAYERS,
        dec_heads=config.VAE_HEADS,
        dec_ff=config.VAE_FF,
        input_dim=vae_input_dim,
        output_dim=vae_output_dim,
        num_entities=N0,
        entity_conditioned=bool(getattr(config, "VAE_ENTITY_CONDITION", False)),
    ).to(device)
    vae_ckpt = Path(config.VAE_CKPT)
    if not vae_ckpt.exists():
        raise FileNotFoundError(f"Missing VAE checkpoint: {vae_ckpt}")
    vae_payload = torch.load(vae_ckpt, map_location=device)
    validate_checkpoint_target_metadata(vae_payload, config, context="VAE")
    _load_module_state(vae, unwrap_checkpoint_model(vae_payload), strict=True)
    vae.eval()

    # ---------------- Summarizer (frozen) ----------------
    laplace_summarizer = LaplaceAE(
        num_entities=N0,
        feat_dim=Fv,
        window_size=K0,
        mix_dim=int(getattr(config, "SUM_MIX_DIM", 64)),
        tv_hidden=config.SUM_TV_HIDDEN,
        out_len=config.SUM_CONTEXT_LEN,
        context_dim=config.SUM_CONTEXT_DIM,
        n_heads=config.NUM_HEADS,
        dropout=config.SUM_DROPOUT,
        time2vec_dim=int(getattr(config, "SUM_TIME2VEC_DIM", 9)),
        irreg_pooling=str(getattr(config, "SUM_IRREG_POOLING", "none")),
        irreg_hidden=int(getattr(config, "SUM_IRREG_HIDDEN", 32)),
        irreg_residual_scale=float(getattr(config, "SUM_IRREG_RES_SCALE", 0.1)),
        t_token_mode=str(getattr(config, "SUM_T_TOKEN_MODE", "none")),
        t_token_scale=float(getattr(config, "SUM_T_TOKEN_SCALE", 0.1)),
        pos_encoding=str(getattr(config, "SUM_POS_ENCODING", "learned_abs")),
        rope_base=float(getattr(config, "SUM_ROPE_BASE", 10000.0)),
        channel_balanced_x_loss=bool(getattr(config, "SUM_CHANNEL_BALANCED_X_LOSS", False)),
    ).to(device)
    sum_ckpt = Path(config.SUM_CKPT)
    if not sum_ckpt.exists():
        raise FileNotFoundError(f"Missing summarizer checkpoint: {sum_ckpt}")
    sum_state = torch.load(sum_ckpt, map_location=device)
    _load_module_state(
        laplace_summarizer,
        sum_state["model"]
        if isinstance(sum_state, dict) and "model" in sum_state
        else sum_state,
        strict=True,
    )
    laplace_summarizer.eval()

    sum_ft_mode = str(getattr(config, "SUM_FT_MODE", "none"))
    sum_ft_named_params = _select_summarizer_finetune_named_params(laplace_summarizer, sum_ft_mode)
    for param in laplace_summarizer.parameters():
        param.requires_grad_(False)
    _set_named_params_trainable(sum_ft_named_params, False)
    sum_ft_lr_mult = float(getattr(config, "SUM_FT_LR_MULT", 0.1))
    sum_ft_weight_decay = float(getattr(config, "SUM_FT_WEIGHT_DECAY", getattr(config, "WEIGHT_DECAY", 0.0)))
    sum_ft_start_epoch = int(getattr(config, "SUM_FT_START_EPOCH", 0) or 0)
    if sum_ft_named_params and sum_ft_start_epoch <= 0:
        sum_ft_start_epoch = int(
            math.ceil(float(getattr(config, "WARMUP_FRAC", 0.0)) * max(1, int(getattr(config, "EPOCHS", 1))))
        )
    sum_ft_start_epoch = max(1, sum_ft_start_epoch) if sum_ft_named_params else 0
    sum_ft_param_count = sum(param.numel() for _, param in sum_ft_named_params)

    # ---------------- Diffusion model ----------------
    model_kwargs = _llapdiff_model_kwargs(config)
    cond_pool_mode = str(model_kwargs["summary_pool_mode"])
    cond_pool_use_raw = bool(model_kwargs["pole_pool_use_raw_summary"])
    block_summary_adaln = bool(model_kwargs["block_summary_adaln"])
    analysis_summary_qk = bool(model_kwargs["analysis_summary_qk"])
    analysis_qk_use_raw = bool(model_kwargs["analysis_qk_use_raw_summary"])
    adapter_cfg = _cond_adapter_config(config)
    cond_adapter_mode = str(adapter_cfg["mode"])
    diff_model = build_llapdiff_model(config, device)

    init_ckpt_raw = getattr(config, "DIFF_INIT_CKPT", None)
    init_ckpt_path = None
    init_ema_state = None
    if init_ckpt_raw not in {None, "", False}:
        init_ckpt_path = Path(str(init_ckpt_raw))
        if not init_ckpt_path.exists():
            raise FileNotFoundError(f"Missing DIFF_INIT_CKPT: {init_ckpt_path}")
        init_payload = torch.load(init_ckpt_path, map_location=device)
        init_state = init_payload.get("model") if isinstance(init_payload, dict) else init_payload
        if init_state is None:
            raise ValueError(f"DIFF_INIT_CKPT does not contain model weights: {init_ckpt_path}")
        _load_module_state(diff_model, init_state, strict=True)
        if isinstance(init_payload, dict):
            init_ema_state = init_payload.get("ema")
        if verbose:
            print(f"[init] loaded diffusion weights from {init_ckpt_path} strict=True")

    scheduler = diff_model.scheduler
    optimizer_param_groups = [
        {
            "params": [param for param in diff_model.parameters() if param.requires_grad],
            "lr": float(config.BASE_LR),
            "weight_decay": float(config.WEIGHT_DECAY),
        }
    ]
    if sum_ft_named_params:
        optimizer_param_groups.append(
            {
                "params": [param for _, param in sum_ft_named_params],
                "lr": float(config.BASE_LR) * sum_ft_lr_mult,
                "weight_decay": sum_ft_weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_param_groups, lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)
    amp_enabled = bool(getattr(config, "DIFF_AMP", False) and device.type == "cuda")
    scaler = _make_grad_scaler(enabled=amp_enabled, device=device)
    total_steps = max(1, int(config.EPOCHS) * max(1, len(train_dl)))
    lr_schedule_name = str(getattr(config, "LR_SCHEDULE", "warmup_cosine"))
    lr_sched = make_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        schedule=lr_schedule_name,
        warmup_frac=config.WARMUP_FRAC,
        base_lr=config.BASE_LR,
        min_lr=config.MIN_LR,
    )
    ema = (
        EMA(diff_model, decay=config.EMA_DECAY)
        if getattr(config, "USE_EMA_EVAL", False)
        else None
    )
    if ema is not None and init_ema_state is not None:
        ema.load_state_dict(init_ema_state)
        if verbose:
            print(f"[init] loaded EMA weights from {init_ckpt_path}")
    use_ema_eval = ema is not None and bool(getattr(config, "USE_EMA_EVAL", False))
    val_metric_source = _resolve_metric_source(
        getattr(config, "VAL_METRIC_SOURCE", "ema"),
        default="ema" if use_ema_eval else "raw",
        use_ema=use_ema_eval,
        label="VAL_METRIC_SOURCE",
    )
    test_metric_source_cfg = getattr(config, "TEST_METRIC_SOURCE", "same")
    test_metric_source = _resolve_metric_source(
        test_metric_source_cfg,
        default=val_metric_source,
        use_ema=use_ema_eval,
        label="TEST_METRIC_SOURCE",
    )
    ema_compare = use_ema_eval and bool(getattr(config, "EMA_COMPARE", True))
    trainable_params = sum(p.numel() for p in diff_model.parameters() if p.requires_grad)
    if debug:
        print(
            "[capacity/reg] "
            f"params={trainable_params:,} "
            f"width={config.MODEL_WIDTH} "
            f"layers={config.NUM_LAYERS} "
            f"heads={config.NUM_HEADS} "
            f"dropout={float(config.DROPOUT):.3f}/{float(config.ATTN_DROPOUT):.3f} "
            f"weight_decay={float(config.WEIGHT_DECAY):.2e} "
            f"ema={bool(getattr(config, 'USE_EMA_EVAL', False))}/{float(getattr(config, 'EMA_DECAY', 0.0)):.3f} "
            f"drop_cond_p={float(getattr(config, 'DROP_COND_P', 0.0)):.2f}"
        )
        print(
            "[conditioning] "
            f"pool_mode={cond_pool_mode} "
            f"pool_use_raw={cond_pool_use_raw} "
            f"block_summary_adaln={block_summary_adaln} "
            f"analysis_summary_qk={analysis_summary_qk} "
            f"analysis_qk_use_raw={analysis_qk_use_raw}"
        )
    if sum_ft_named_params and verbose:
        print(
            "[summ ft] "
            f"mode={sum_ft_mode} "
            f"params={sum_ft_param_count:,} "
            f"lr_mult={sum_ft_lr_mult:.3f} "
            f"weight_decay={sum_ft_weight_decay:.2e} "
            f"start_epoch={sum_ft_start_epoch}"
        )
    if cond_adapter_mode != "none" and debug:
        print(
            "[cond adapter] "
            f"mode={cond_adapter_mode} "
            f"hidden={int(getattr(config, 'COND_ADAPTER_HIDDEN', 128))} "
            f"scale={float(getattr(config, 'COND_ADAPTER_SCALE', 0.1)):.3f} "
            f"dropout={float(getattr(config, 'COND_ADAPTER_DROPOUT', 0.0)):.3f}"
        )

    # ---------------- Precomputed frozen diffusion inputs ----------------
    diffusion_input_cache = build_or_load_diffusion_input_cache(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        vae=vae,
        summarizer=laplace_summarizer,
        device=device,
        config_obj=config,
        summary_ft_mode=sum_ft_mode,
        verbose=verbose,
    )
    if diffusion_input_cache is not None:
        _release_cuda_allocator(device)

    # ---------------- Latent stats / calibration ----------------
    latent_norm_mode = str(getattr(config, "LATENT_NORM_MODE", "global"))
    if diffusion_input_cache is not None:
        mu_mean, mu_std = diffusion_input_cache.mu_mean, diffusion_input_cache.mu_std
        baseline_target_variance = diffusion_input_cache.train.calculate_target_variance(
            predict_type=config.PREDICT_TYPE,
            scheduler=scheduler,
            device=device,
            mu_mean=mu_mean,
            mu_std=mu_std,
        )
    else:
        mu_mean, mu_std = compute_latent_stats(vae, train_dl, device, mode=latent_norm_mode)
        baseline_target_variance = calculate_target_variance(
            predict_type=config.PREDICT_TYPE,
            dataloader=train_dl,
            device=device,
            scheduler=scheduler,
            vae=vae,
            latent_stats=(mu_mean, mu_std),
        )
    if diffusion_input_cache is not None:
        _release_cuda_allocator(device)
    if debug:
        print(
            f"Baseline target variance ({config.PREDICT_TYPE}): {baseline_target_variance:.6f}"
        )
        print(f"[latent stats] mode={latent_norm_mode}")
    latent_probe_batches = max(0, int(getattr(config, "LATENT_PROBE_BATCHES", 4)))
    latent_probe = (
        _collect_latent_probe(
            vae,
            train_dl,
            device,
            mu_mean,
            mu_std,
            max_batches=latent_probe_batches,
            input_cache=(diffusion_input_cache.train if diffusion_input_cache is not None else None),
        )
        if debug and latent_probe_batches > 0
        else None
    )
    if latent_probe is not None and debug:
        print(
            "[latent probe] "
            f"mean={latent_probe['mean']:.4f} std={latent_probe['std']:.4f} "
            f"abs_mean={latent_probe['abs_mean']:.4f} abs_p95={latent_probe['abs_p95']:.4f} "
            f"feat_std(mean/min/max)="
            f"{latent_probe['feat_std_mean']:.4f}/"
            f"{latent_probe['feat_std_min']:.4f}/"
            f"{latent_probe['feat_std_max']:.4f}"
        )
    if diffusion_input_cache is not None:
        _release_cuda_allocator(device)

    pole_probe_state = _init_pole_probe(diff_model, laplace_summarizer, train_dl, device) if bool(getattr(config, "POLE_PROBE", False)) else None
    pole_probe_every = max(1, int(getattr(config, "POLE_PROBE_EVERY", 1)))
    train_t_sampler = str(getattr(config, "TRAIN_T_SAMPLER", "uniform"))
    train_t_karras_rho = float(getattr(config, "TRAIN_T_KARRAS_RHO", getattr(config, "KARRAS_RHO", 7.5)))
    train_loss_t_bins = max(0, int(getattr(config, "TRAIN_LOSS_T_BINS", 8)))
    minsnr_normalize = str(getattr(config, "MINSNR_NORMALIZE", "auto"))
    cond_train_mode = str(getattr(config, "COND_TRAIN_MODE", "auto")).strip().lower()
    if cond_train_mode == "auto":
        cond_train_mode = "stochastic"
    if cond_train_mode not in {"stochastic", "dual"}:
        raise ValueError(
            f"Unknown COND_TRAIN_MODE '{cond_train_mode}'. Use 'auto', 'stochastic' or 'dual'."
        )
    target_mask_aux_p = _effective_target_mask_aux_probability(config)
    target_mask_aux_keep_prob = float(getattr(config, "TARGET_MASK_AUX_KEEP_PROB", 0.5))
    target_mask_aux_keep_mode = str(getattr(config, "TARGET_MASK_AUX_KEEP_MODE", "random"))
    target_mask_aux_keep_stride = max(1, int(getattr(config, "TARGET_MASK_AUX_KEEP_STRIDE", 4)))
    target_mask_aux_start_epoch = _resolve_target_mask_aux_start_epoch(
        config,
        aux_prob=target_mask_aux_p,
    )
    if debug:
        print(
            "[diffusion train] "
            f"lr_schedule={lr_schedule_name} "
            f"t_sampler={train_t_sampler} "
            f"t_karras_rho={train_t_karras_rho:.2f} "
            f"minsnr_norm={minsnr_normalize} "
            f"cond_mode={cond_train_mode} "
            f"drop_cond_p={float(getattr(config, 'DROP_COND_P', 0.0)):.2f}"
        )
    if target_mask_aux_p > 0.0 and verbose:
        print(
            "[target-mask aux] "
            f"p={target_mask_aux_p:.2f} "
            f"start_epoch={target_mask_aux_start_epoch} "
            f"keep_mode={target_mask_aux_keep_mode} "
            f"keep_prob={target_mask_aux_keep_prob:.2f} "
            f"keep_stride={target_mask_aux_keep_stride}"
        )
    if debug:
        print(
            "[eval selection] "
            f"val_source={val_metric_source} "
            f"test_source={test_metric_source} "
            f"ema_compare={ema_compare}"
        )

    max_summary_nonfinite_grad_steps = max(
        0,
        int(getattr(config, "SUM_MAX_NONFINITE_GRAD_STEPS", 0) or 0),
    )
    skipped_summary_ft_nonfinite_grad_steps = 0

    # ---------------- Training loop ----------------
    def train_one_epoch(epoch: int) -> Dict[str, object]:
        nonlocal skipped_summary_ft_nonfinite_grad_steps
        diff_model.train()
        summary_ft_active = bool(sum_ft_named_params) and (epoch + 1) >= sum_ft_start_epoch
        _set_named_params_trainable(sum_ft_named_params, summary_ft_active)
        # Keep summarizer deterministic while allowing gradients once fine-tuning starts.
        laplace_summarizer.eval()
        running_loss = 0.0
        running_raw_loss = 0.0
        running_raw_loss_cond = 0.0
        running_raw_loss_uncond = 0.0
        num_samples = 0
        num_samples_cond = 0
        num_samples_uncond = 0
        cond_weight_sum = 0.0
        target_mask_aux_batches = 0.0
        target_mask_aux_keep_frac_sum = 0.0
        target_mask_aux_hidden_frac_sum = 0.0
        t_loss_meter = (
            _init_timestep_loss_meter(train_loss_t_bins, scheduler.timesteps)
            if train_loss_t_bins > 0
            else None
        )

        train_input_cache = diffusion_input_cache.train if diffusion_input_cache is not None else None
        if train_input_cache is not None:
            train_input_cache.reset()
        epoch_skipped_summary_ft_nonfinite_grad_steps = 0
        train_batches = progress_iter(
            train_dl,
            desc=f"llapdiff train e{epoch + 1:03d}/{epochs:03d}",
            enabled=verbose,
            unit="batch",
        )
        for xb, yb, meta in train_batches:
            (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta, device)
            if not mask_bn.any():
                continue
            cached_batch = (
                train_input_cache.next_batch(
                    meta,
                    device=device,
                    mu_mean=mu_mean,
                    mu_std=mu_std,
                    load_latents=True,
                    load_summary=True,
                )
                if train_input_cache is not None
                else None
            )

            cond_summary, cond_summary_raw = _build_cond_summary_pair(
                laplace_summarizer,
                diff_model,
                V,
                T,
                mask_bn,
                device,
                dt=meta.get("delta_t"),
                x_obs_mask=meta.get("x_obs_mask"),
                requires_grad=summary_ft_active,
                summary_base_raw=(cached_batch.summary_raw if cached_batch is not None else None),
            )
            if not _is_finite_tensor(cond_summary):
                raise FloatingPointError("non-finite cond_summary detected")
            if cond_summary_raw is not None and not _is_finite_tensor(cond_summary_raw):
                raise FloatingPointError("non-finite raw conditioning summary detected")

            dt_flat = _flatten_dt(
                meta,
                mask_bn,
                device,
                key="delta_t_y",
            )
            if not _is_finite_tensor(dt_flat):
                raise FloatingPointError("non-finite delta_t_y detected")

            mu_norm, obs_any = _latent_targets_for_batch(
                vae,
                yb,
                mask_bn,
                meta,
                device,
                mu_mean,
                mu_std,
                cached_batch=cached_batch,
            )
            if mu_norm is None or obs_any is None or not obs_any.any():
                continue
            if not _is_finite_tensor(mu_norm):
                raise FloatingPointError("non-finite latent targets detected")

            dt_model = _match_dt_to_horizon(dt_flat, mu_norm.size(1))
            Beff = mu_norm.size(0)

            p_drop = float(getattr(config, "DROP_COND_P", 0.0))
            if cond_train_mode == "dual":
                idx_c = torch.arange(Beff, device=device)
                idx_u = torch.arange(Beff, device=device)
                w_c = 1.0 - p_drop
                w_u = p_drop
            else:
                m_cond = torch.rand(Beff, device=device) >= p_drop
                idx_c = m_cond.nonzero(as_tuple=False).squeeze(1)
                idx_u = (~m_cond).nonzero(as_tuple=False).squeeze(1)
                w_c = idx_c.numel() / max(1, Beff)
                w_u = idx_u.numel() / max(1, Beff)

            t = sample_training_timesteps(
                scheduler,
                Beff,
                device,
                sampler=train_t_sampler,
                karras_rho=train_t_karras_rho,
            )
            noise = torch.randn_like(mu_norm)
            x_t, eps_true = scheduler.q_sample(mu_norm, t, noise)
            if (not _is_finite_tensor(x_t)) or (not _is_finite_tensor(eps_true)):
                raise FloatingPointError("non-finite noisy latent sample detected")

            x_t_c = x_t[idx_c] if idx_c.numel() > 0 else None
            eps_true_c = eps_true[idx_c] if idx_c.numel() > 0 else None
            target_mask_c = obs_any[idx_c] if idx_c.numel() > 0 else None
            cond_summary_raw_c = (
                cond_summary_raw[idx_c]
                if (cond_summary_raw is not None and idx_c.numel() > 0)
                else None
            )
            target_mask_aux_stats = {
                "applied": False,
                "keep_frac": 1.0,
                "hidden_frac": 0.0,
            }
            aux_enabled = (
                idx_c.numel() > 0
                and target_mask_aux_p > 0.0
                and (epoch + 1) >= target_mask_aux_start_epoch
                and float(torch.rand(()).item()) < target_mask_aux_p
            )
            if aux_enabled:
                x_t_c, target_mask_c, target_mask_aux_stats = _maybe_apply_target_mask_aux(
                    scheduler,
                    x_t_c,
                    mu_norm[idx_c],
                    obs_any[idx_c],
                    t[idx_c],
                    enabled=True,
                    keep_mode=target_mask_aux_keep_mode,
                    keep_prob=target_mask_aux_keep_prob,
                    keep_stride=target_mask_aux_keep_stride,
                )

            sc_feat_c = sc_feat_u = None
            use_sc = (
                bool(getattr(config, "SELF_COND", False))
                and epoch >= int(getattr(config, "SELF_COND_START_EPOCH", 0))
                and float(torch.rand(()).item())
                < float(getattr(config, "SELF_COND_P", 0.0))
            )
            if use_sc:
                with torch.no_grad():
                    if idx_c.numel() > 0:
                        pred_ng_c = diff_model(
                            x_t_c,
                            t[idx_c],
                            cond_summary=cond_summary[idx_c],
                            cond_summary_raw=cond_summary_raw_c,
                            sc_feat=None,
                            dt=(dt_model[idx_c] if dt_model is not None else None),
                        )
                        sc_feat_c = scheduler.to_x0(
                            x_t_c, t[idx_c], pred_ng_c, config.PREDICT_TYPE
                        ).detach()

                    if idx_u.numel() > 0:
                        pred_ng_u = diff_model(
                            x_t[idx_u],
                            t[idx_u],
                            cond_summary=None,
                            sc_feat=None,
                            dt=(dt_model[idx_u] if dt_model is not None else None),
                        )
                        sc_feat_u = scheduler.to_x0(
                            x_t[idx_u], t[idx_u], pred_ng_u, config.PREDICT_TYPE
                        ).detach()

            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            raw_loss = torch.zeros((), device=device)

            with _autocast_context(enabled=amp_enabled, device=device):
                if idx_c.numel() > 0:
                    loss_c, loss_c_stats = diffusion_loss(
                        diff_model,
                        scheduler,
                        mu_norm[idx_c],
                        t[idx_c],
                        cond_summary=cond_summary[idx_c],
                        cond_summary_raw=cond_summary_raw_c,
                        predict_type=config.PREDICT_TYPE,
                        dt=(dt_model[idx_c] if dt_model is not None else None),
                        weight_scheme=getattr(config, "LOSS_WEIGHT_SCHEME", "none"),
                        minsnr_gamma=float(getattr(config, "MINSNR_GAMMA", 5.0)),
                        minsnr_normalize=minsnr_normalize,
                        sc_feat=sc_feat_c,
                        reuse_xt_eps=(x_t_c, eps_true_c),
                        target_mask=target_mask_c,
                        return_stats=True,
                    )
                    loss = loss + loss_c * w_c
                    raw_loss = raw_loss + loss_c_stats["raw_loss"] * w_c
                    running_raw_loss_cond += float(loss_c_stats["raw_loss"].item()) * int(idx_c.numel())
                    num_samples_cond += int(idx_c.numel())
                    if t_loss_meter is not None:
                        _update_timestep_loss_meter(
                            t_loss_meter,
                            t[idx_c],
                            loss_c_stats["per_sample_raw"],
                            loss_c_stats["per_sample_weighted"],
                        )

                if idx_u.numel() > 0:
                    loss_u, loss_u_stats = diffusion_loss(
                        diff_model,
                        scheduler,
                        mu_norm[idx_u],
                        t[idx_u],
                        cond_summary=None,
                        predict_type=config.PREDICT_TYPE,
                        dt=(dt_model[idx_u] if dt_model is not None else None),
                        weight_scheme=getattr(config, "LOSS_WEIGHT_SCHEME", "none"),
                        minsnr_gamma=float(getattr(config, "MINSNR_GAMMA", 5.0)),
                        minsnr_normalize=minsnr_normalize,
                        sc_feat=sc_feat_u,
                        reuse_xt_eps=(x_t[idx_u], eps_true[idx_u]),
                        target_mask=obs_any[idx_u],
                        return_stats=True,
                    )
                    loss = loss + loss_u * w_u
                    raw_loss = raw_loss + loss_u_stats["raw_loss"] * w_u
                    running_raw_loss_uncond += float(loss_u_stats["raw_loss"].item()) * int(idx_u.numel())
                    num_samples_uncond += int(idx_u.numel())
                    if t_loss_meter is not None:
                        _update_timestep_loss_meter(
                            t_loss_meter,
                            t[idx_u],
                            loss_u_stats["per_sample_raw"],
                            loss_u_stats["per_sample_weighted"],
                        )

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("non-finite loss detected")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            diffusion_grad_params = [param for param in diff_model.parameters() if param.requires_grad]
            summary_grad_params = (
                [param for _, param in sum_ft_named_params if param.requires_grad]
                if summary_ft_active
                else []
            )
            grad_params = [*diffusion_grad_params, *summary_grad_params]
            diffusion_grads_finite = _grads_are_finite_params(diffusion_grad_params)
            summary_grads_finite = _grads_are_finite_params(summary_grad_params)
            if not (diffusion_grads_finite and summary_grads_finite):
                skip_summary_ft = _should_skip_nonfinite_summary_ft_gradients(
                    diffusion_params=diffusion_grad_params,
                    summary_ft_params=summary_grad_params,
                    amp_enabled=amp_enabled,
                    summary_ft_active=summary_ft_active,
                    skipped_nonfinite_grad_steps=skipped_summary_ft_nonfinite_grad_steps,
                    max_nonfinite_grad_steps=max_summary_nonfinite_grad_steps,
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if skip_summary_ft:
                    skipped_summary_ft_nonfinite_grad_steps += 1
                    epoch_skipped_summary_ft_nonfinite_grad_steps += 1
                    continue
                if not diffusion_grads_finite:
                    raise FloatingPointError("non-finite LLapDiff gradients detected")
                if summary_ft_active and not summary_grads_finite:
                    raise FloatingPointError(
                        "non-finite summarizer fine-tuning gradients detected "
                        f"after {skipped_summary_ft_nonfinite_grad_steps} skipped optimizer step(s)"
                    )
                raise FloatingPointError("non-finite gradients detected")

            grad_clip = float(getattr(config, "GRAD_CLIP", 0.0) or 0.0)
            if grad_clip > 0:
                clip_params = [param for param in grad_params if param.requires_grad]
                grad_norm = nn.utils.clip_grad_norm_(clip_params, grad_clip)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    raise FloatingPointError("non-finite gradient norm detected")

            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(diff_model)
            lr_sched.step()

            running_loss += float(loss.item()) * Beff
            running_raw_loss += float(raw_loss.item()) * Beff
            cond_weight_sum += float(w_c) * Beff
            if target_mask_aux_stats["applied"]:
                target_mask_aux_batches += 1.0
                target_mask_aux_keep_frac_sum += float(target_mask_aux_stats["keep_frac"])
                target_mask_aux_hidden_frac_sum += float(target_mask_aux_stats["hidden_frac"])
            num_samples += Beff

        if num_samples <= 0:
            if epoch_skipped_summary_ft_nonfinite_grad_steps > 0:
                raise FloatingPointError(
                    "all LLapDiff optimizer steps were skipped because summarizer "
                    "fine-tuning gradients were non-finite"
                )
            raise RuntimeError("No valid diffusion training samples were processed in this epoch")
        epoch_loss = running_loss / num_samples
        epoch_raw_loss = running_raw_loss / num_samples
        epoch_stats = {
            "loss": epoch_loss,
            "raw_loss": epoch_raw_loss,
            "cond_fraction": cond_weight_sum / num_samples,
            "summ_ft_active": float(summary_ft_active),
            "summ_ft_skipped_nonfinite_grad_steps": int(epoch_skipped_summary_ft_nonfinite_grad_steps),
            "summ_ft_skipped_nonfinite_grad_steps_total": int(skipped_summary_ft_nonfinite_grad_steps),
            "summ_max_nonfinite_grad_steps": int(max_summary_nonfinite_grad_steps),
            "target_mask_aux_batches": int(target_mask_aux_batches),
            "target_mask_aux_batch_frac": target_mask_aux_batches / max(1, len(train_dl)),
            "target_mask_keep_frac": target_mask_aux_keep_frac_sum / max(1.0, target_mask_aux_batches),
            "target_mask_hidden_frac": target_mask_aux_hidden_frac_sum / max(1.0, target_mask_aux_batches),
        }
        predict_type_name = str(getattr(config, "PREDICT_TYPE", "")).strip().lower()
        if predict_type_name == "x0":
            epoch_stats["train_x0_mse_raw"] = epoch_raw_loss
            if num_samples_cond > 0:
                epoch_stats["train_x0_mse_raw_cond"] = running_raw_loss_cond / num_samples_cond
            if num_samples_uncond > 0:
                epoch_stats["train_x0_mse_raw_uncond"] = running_raw_loss_uncond / num_samples_uncond
        elif predict_type_name:
            key_base = f"train_{predict_type_name}_mse_raw"
            epoch_stats[key_base] = epoch_raw_loss
            if num_samples_cond > 0:
                epoch_stats[f"{key_base}_cond"] = running_raw_loss_cond / num_samples_cond
            if num_samples_uncond > 0:
                epoch_stats[f"{key_base}_uncond"] = running_raw_loss_uncond / num_samples_uncond
        if t_loss_meter is not None:
            epoch_stats["timestep_loss_bins"] = _finalize_timestep_loss_meter(t_loss_meter)
        return epoch_stats

    train_losses = []
    train_history = []
    val_history = []
    raw_val_history = []
    ema_val_history = []
    pole_probe_history = []
    best_val_crps = float("inf")
    best_val_crps_by_source = {"raw": float("inf"), "ema": float("inf")}
    last_epoch = 0
    out_dir = Path(getattr(config, "OUT_DIR", "./outputs"))
    target_suffix = str(getattr(config, "TARGET_ARTIFACT_SUFFIX", "") or "")
    pred_tag = f"pred-{int(getattr(config, 'PRED', 0))}{target_suffix}"
    best_ckpt_path = out_dir / f"llapdiff_{pred_tag}_best.pt"
    best_ckpt_path_raw = out_dir / f"llapdiff_{pred_tag}_best_raw.pt"
    best_ckpt_path_ema = out_dir / f"llapdiff_{pred_tag}_best_ema.pt"
    last_ckpt_path = out_dir / f"llapdiff_{pred_tag}_last.pt"
    save_best = bool(getattr(config, "SAVE_BEST", True))

    def _checkpoint_payload(**extra) -> Dict[str, object]:
        metadata = target_metadata_from_config(config)
        payload: Dict[str, object] = {
            "model": diff_model.state_dict(),
            "model_config": _llapdiff_model_config(config),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "target_metadata": metadata,
            "target_dim": int(target_dim),
            "target_cols": list(metadata.get("target_cols", [])),
            "target_indices": list(metadata.get("target_indices", [])),
            "target_source": str(metadata.get("target_source", "")),
            "target_signature": str(metadata.get("target_signature", "")),
            "vae_input_dim": int(vae_input_dim),
            "vae_output_dim": int(vae_output_dim),
            "mu_mean": mu_mean.detach().cpu(),
            "mu_std": mu_std.detach().cpu(),
            "summ_ft_skipped_nonfinite_grad_steps": int(skipped_summary_ft_nonfinite_grad_steps),
            "summ_max_nonfinite_grad_steps": int(max_summary_nonfinite_grad_steps),
        }
        payload.update(extra)
        return payload

    epochs = int(getattr(config, "EPOCHS", 1))
    eval_every = max(1, int(getattr(config, "EVAL_EVERY", 1)))

    def _eval_interval(name: str, default: int, *, allow_disable: bool) -> int:
        value = int(getattr(config, name, default))
        if allow_disable and value <= 0:
            return 0
        return max(1, value)

    downstream_eval_every = _eval_interval(
        "DOWNSTREAM_EVAL_EVERY", eval_every, allow_disable=True
    )
    val_diag_every = _eval_interval("VAL_DIAG_EVERY", eval_every, allow_disable=False)
    disabled_default = downstream_eval_every if downstream_eval_every > 0 else 0
    irreg_check_every = _eval_interval(
        "IRREG_CHECK_EVERY", disabled_default, allow_disable=True
    )
    ema_compare_every = _eval_interval(
        "EMA_COMPARE_EVERY", disabled_default, allow_disable=True
    )
    early_stop_patience = int(getattr(config, "EARLY_STOP", 0) or 0)
    early_stop_min_epochs = int(getattr(config, "EARLY_STOP_MIN_EPOCHS", 0) or 0)
    if early_stop_patience > 0 and early_stop_min_epochs <= 0:
        early_stop_min_epochs = int(math.ceil(float(getattr(config, "WARMUP_FRAC", 0.0)) * max(1, epochs)))
    early_stop_min_epochs = max(0, early_stop_min_epochs)
    patience_ctr = 0
    primary_eval_metric = str(getattr(config, "PRIMARY_EVAL_METRIC", "crps")).strip().lower()
    if primary_eval_metric not in {"crps", "val_diag_mse_raw"}:
        raise ValueError(f"Unsupported PRIMARY_EVAL_METRIC: {primary_eval_metric}")
    best_primary_metric = float("inf")

    for epoch in range(epochs):
        epoch_stats = train_one_epoch(epoch)
        train_loss = float(epoch_stats["loss"])
        train_losses.append(train_loss)
        train_history.append({"epoch": epoch + 1, **epoch_stats})
        last_epoch = epoch + 1
        lr_now = optimizer.param_groups[0]["lr"]
        if debug and (epoch + 1) in {1, 71}:
            print(f"[lr probe] epoch={epoch + 1} lr={lr_now:.8e}")
        if sum_ft_named_params and verbose and (epoch + 1) == sum_ft_start_epoch:
            print(f"[summ ft] activated at epoch {epoch + 1}")
        if target_mask_aux_p > 0.0 and verbose and (epoch + 1) == target_mask_aux_start_epoch:
            print(f"[target-mask aux] activated at epoch {epoch + 1}")
        if verbose:
            print(
                f"[epoch {epoch + 1}/{epochs}] "
                f"train_loss={train_loss:.6f} "
                f"raw_loss={float(epoch_stats['raw_loss']):.6f} "
                f"cond_frac={float(epoch_stats['cond_fraction']):.3f} "
                f"lr={lr_now:.3e}"
            )
            if int(epoch_stats.get("summ_ft_skipped_nonfinite_grad_steps", 0)) > 0:
                print(
                    " skipped "
                    f"{int(epoch_stats['summ_ft_skipped_nonfinite_grad_steps'])} "
                    "summarizer fine-tuning optimizer step(s) with non-finite AMP gradients"
                )
        predict_type_name = str(getattr(config, "PREDICT_TYPE", "")).strip().lower()
        train_metric_key = (
            "train_x0_mse_raw"
            if predict_type_name == "x0"
            else f"train_{predict_type_name}_mse_raw"
        )
        if train_metric_key in epoch_stats and debug:
            metric_label = "x0" if predict_type_name == "x0" else predict_type_name
            print(f" Train {metric_label}-MSE (unweighted): {float(epoch_stats[train_metric_key]):.6f}")
            cond_key = f"{train_metric_key}_cond"
            uncond_key = f"{train_metric_key}_uncond"
            if cond_key in epoch_stats or uncond_key in epoch_stats:
                cond_str = (
                    f"{float(epoch_stats[cond_key]):.6f}"
                    if cond_key in epoch_stats
                    else "n/a"
                )
                uncond_str = (
                    f"{float(epoch_stats[uncond_key]):.6f}"
                    if uncond_key in epoch_stats
                    else "n/a"
                )
                print(f" Train {metric_label}-MSE split: cond={cond_str} uncond={uncond_str}")
        if target_mask_aux_p > 0.0 and debug:
            print(
                "[target-mask aux epoch] "
                f"batches={int(epoch_stats['target_mask_aux_batches'])} "
                f"batch_frac={float(epoch_stats['target_mask_aux_batch_frac']):.3f} "
                f"keep_frac={float(epoch_stats['target_mask_keep_frac']):.3f} "
                f"hidden_frac={float(epoch_stats['target_mask_hidden_frac']):.3f}"
            )
        t_loss_bins = epoch_stats.get("timestep_loss_bins")
        if t_loss_bins is not None and debug:
            raw_bins = ", ".join(f"{v:.4f}" for v in t_loss_bins["raw"])
            weighted_bins = ", ".join(f"{v:.4f}" for v in t_loss_bins["weighted"])
            print(f"[train t-bins raw] {raw_bins}")
            print(f"[train t-bins weighted] {weighted_bins}")

        if pole_probe_state is not None and (epoch + 1) % pole_probe_every == 0:
            pole_metrics = _collect_pole_probe(diff_model, pole_probe_state)
            if pole_metrics is not None:
                pole_probe_history.append({"epoch": epoch + 1, **pole_metrics})
                if debug:
                    print(
                        "[pole probe] "
                        f"base_drho={pole_metrics['base_rho_delta_mean']:.6e} "
                        f"base_domega={pole_metrics['base_omega_delta_mean']:.6e} "
                        f"eff_drho={pole_metrics['eff_rho_delta_mean']:.6e} "
                        f"eff_domega={pole_metrics['eff_omega_delta_mean']:.6e}"
                    )

        eval_due = len(val_dl) > 0
        run_downstream_eval = (
            eval_due and downstream_eval_every > 0 and ((epoch + 1) % downstream_eval_every == 0)
        )
        run_val_diag = eval_due and ((epoch + 1) % val_diag_every == 0)
        run_irreg_check = (
            eval_due and irreg_check_every > 0 and ((epoch + 1) % irreg_check_every == 0)
        )
        run_ema_compare = (
            ema_compare
            and run_downstream_eval
            and ema_compare_every > 0
            and ((epoch + 1) % ema_compare_every == 0)
        )

        val_metrics = None
        raw_val_metrics = None
        ema_val_metrics = None
        val_diag = None
        irreg = None

        if run_downstream_eval:
            val_metrics = evaluate_regression(
                diff_model,
                vae,
                laplace_summarizer,
                val_dl,
                device,
                mu_mean,
                mu_std,
                config,
                ema=_maybe_metric_ema(val_metric_source, ema),
                self_cond=bool(getattr(config, "SELF_COND", False)),
                verbose=debug,
                progress_enabled=verbose,
                progress_label=f"llapdiff val e{epoch + 1:03d}/{epochs:03d}",
                **_sampling_kwargs(config, prefix="EVAL"),
            )

        if run_val_diag:
            val_diag = evaluate_val_diagnostics(
                diff_model,
                vae,
                laplace_summarizer,
                val_dl,
                device,
                mu_mean,
                mu_std,
                config,
                ema=_maybe_metric_ema(val_metric_source, ema),
                num_snr_bins=int(getattr(config, "VAL_DIAG_SNR_BINS", 8)),
                min_log_snr=float(getattr(config, "VAL_DIAG_LOGSNR_MIN", -12.0)),
                max_log_snr=float(getattr(config, "VAL_DIAG_LOGSNR_MAX", 12.0)),
                input_cache=(diffusion_input_cache.val if diffusion_input_cache is not None else None),
                progress_enabled=verbose,
                progress_label=f"llapdiff val-diag e{epoch + 1:03d}/{epochs:03d}",
            )
            predict_type_name = str(getattr(config, "PREDICT_TYPE", "")).strip().lower()
            metric_label = "x0" if predict_type_name in {"", "x0"} else predict_type_name
            if debug:
                print(f" Val {metric_label}-MSE (unweighted): {val_diag['val_diag_mse_raw']:.6f}")
            snr_bins = val_diag["val_diag_mse_by_snr_bin"]
            edges = snr_bins["log_snr_edges"]
            vals = snr_bins["raw"]
            counts = snr_bins["counts"]
            snr_fmt = ", ".join(
                f"[{edges[i]:.1f},{edges[i + 1]:.1f}]:{vals[i]:.6f} (n={counts[i]})"
                for i in range(len(vals))
            )
            if debug:
                print(f" Val {metric_label}-MSE by log-SNR bin: {snr_fmt}")
            if not run_downstream_eval:
                val_history.append({"epoch": epoch + 1, "source": val_metric_source, **val_diag})

        if run_irreg_check:
            irreg = evaluate_irregular_time_checks(
                laplace_summarizer,
                val_dl,
                device,
                diff_model=diff_model,
                max_batches=int(getattr(config, "IRREG_CHECK_BATCHES", 4)),
            )
            if debug:
                print(
                    "[irreg check] "
                    f"ctx_delta_no_dt={irreg['ctx_delta_no_dt']:.6f} "
                    f"ctx_delta_no_xmask={irreg['ctx_delta_no_xmask']:.6f} "
                    f"ctx_delta_zero_tsig={irreg['ctx_delta_zero_tsig']:.6f}"
                )

        if run_downstream_eval:
            if val_diag is not None:
                val_metrics.update(val_diag)
            if irreg is not None:
                val_metrics.update(irreg)
            raw_val_metrics = val_metrics if val_metric_source == "raw" else None
            ema_val_metrics = val_metrics if val_metric_source == "ema" else None
            if run_ema_compare:
                compare_source = "raw" if val_metric_source == "ema" else "ema"
                compare_val_metrics = evaluate_regression(
                    diff_model,
                    vae,
                    laplace_summarizer,
                    val_dl,
                    device,
                    mu_mean,
                    mu_std,
                    config,
                    ema=_maybe_metric_ema(compare_source, ema),
                    self_cond=bool(getattr(config, "SELF_COND", False)),
                    verbose=debug,
                    progress_enabled=verbose,
                    progress_label=f"llapdiff val-compare e{epoch + 1:03d}/{epochs:03d}",
                    **_sampling_kwargs(config, prefix="EVAL"),
                )
                if compare_source == "raw":
                    raw_val_metrics = compare_val_metrics
                else:
                    ema_val_metrics = compare_val_metrics
                if debug:
                    print(
                        "[ema compare] "
                        f"raw_crps={float(raw_val_metrics['crps']):.6f} "
                        f"ema_crps={float(ema_val_metrics['crps']):.6f}"
                    )
            val_history.append({"epoch": epoch + 1, "source": val_metric_source, **val_metrics})
            if raw_val_metrics is not None:
                raw_val_history.append({"epoch": epoch + 1, **raw_val_metrics})
            if ema_val_metrics is not None:
                ema_val_history.append({"epoch": epoch + 1, **ema_val_metrics})

            if raw_val_metrics is not None and float(raw_val_metrics["crps"]) < best_val_crps_by_source["raw"]:
                best_val_crps_by_source["raw"] = float(raw_val_metrics["crps"])
                if save_best:
                    _save_checkpoint(
                        best_ckpt_path_raw,
                        _checkpoint_payload(**{
                            "epoch": epoch + 1,
                            "val_metrics": raw_val_metrics,
                            "val_metric_source": "raw",
                            "best_val_crps": best_val_crps_by_source["raw"],
                            "best_val_crps_by_source": {
                                "raw": _finite_or_none(best_val_crps_by_source["raw"]),
                                "ema": _finite_or_none(best_val_crps_by_source["ema"]),
                            },
                        }),
                    )

            if ema_val_metrics is not None and float(ema_val_metrics["crps"]) < best_val_crps_by_source["ema"]:
                best_val_crps_by_source["ema"] = float(ema_val_metrics["crps"])
                if save_best:
                    _save_checkpoint(
                        best_ckpt_path_ema,
                        _checkpoint_payload(**{
                            "epoch": epoch + 1,
                            "val_metrics": ema_val_metrics,
                            "val_metric_source": "ema",
                            "best_val_crps": best_val_crps_by_source["ema"],
                            "best_val_crps_by_source": {
                                "raw": _finite_or_none(best_val_crps_by_source["raw"]),
                                "ema": _finite_or_none(best_val_crps_by_source["ema"]),
                            },
                        }),
                    )

        primary_metrics = None
        if run_downstream_eval and val_metrics is not None:
            primary_metrics = dict(val_metrics)
        if val_diag is not None:
            if primary_metrics is None:
                primary_metrics = {}
            primary_metrics.update(val_diag)
        if irreg is not None:
            if primary_metrics is None:
                primary_metrics = {}
            primary_metrics.update(irreg)

        current_primary_metric = None
        if primary_eval_metric == "crps":
            if val_metrics is not None and "crps" in val_metrics:
                current_primary_metric = float(val_metrics["crps"])
        elif val_diag is not None and "val_diag_mse_raw" in val_diag:
            current_primary_metric = float(val_diag["val_diag_mse_raw"])

        if current_primary_metric is not None:
            if current_primary_metric < best_primary_metric:
                best_primary_metric = current_primary_metric
                if primary_eval_metric == "crps":
                    best_val_crps = current_primary_metric
                patience_ctr = 0
                if save_best:
                    _save_checkpoint(
                        best_ckpt_path,
                        _checkpoint_payload(**{
                            "epoch": epoch + 1,
                            "val_metrics": primary_metrics,
                            "val_metric_source": val_metric_source,
                            "best_val_crps": _finite_or_none(best_val_crps),
                            "best_val_crps_by_source": {
                                "raw": _finite_or_none(best_val_crps_by_source["raw"]),
                                "ema": _finite_or_none(best_val_crps_by_source["ema"]),
                            },
                            "best_primary_metric": best_primary_metric,
                            "best_primary_metric_name": primary_eval_metric,
                        }),
                    )
            elif (epoch + 1) >= early_stop_min_epochs:
                patience_ctr += 1

            if early_stop_patience > 0 and (epoch + 1) >= early_stop_min_epochs and patience_ctr >= early_stop_patience:
                if verbose:
                    print(
                        f"[early stop] validation {primary_eval_metric} did not improve for {early_stop_patience} evals "
                        f"after epoch {early_stop_min_epochs}"
                    )
                break

    # Always save final checkpoint
    _save_checkpoint(
        last_ckpt_path,
        _checkpoint_payload(**{
            "epoch": last_epoch,
            "train_losses": train_losses,
            "train_history": train_history,
            "val_history": val_history,
            "raw_val_history": raw_val_history,
            "ema_val_history": ema_val_history,
            "pole_probe_history": pole_probe_history,
            "latent_probe": latent_probe,
            "best_val_crps": (best_val_crps if best_val_crps != float("inf") else None),
            "best_val_crps_by_source": {
                "raw": _finite_or_none(best_val_crps_by_source["raw"]),
                "ema": _finite_or_none(best_val_crps_by_source["ema"]),
            },
            "best_primary_metric": _finite_or_none(best_primary_metric),
            "best_primary_metric_name": primary_eval_metric,
            "val_metric_source": val_metric_source,
            "test_metric_source": test_metric_source,
        }),
    )

    best_checkpoint = str(best_ckpt_path) if best_ckpt_path.exists() else None
    best_checkpoint_raw = str(best_ckpt_path_raw) if best_ckpt_path_raw.exists() else None
    best_checkpoint_ema = str(best_ckpt_path_ema) if best_ckpt_path_ema.exists() else None
    last_checkpoint = str(last_ckpt_path) if last_ckpt_path.exists() else None
    best_val = best_val_crps if best_val_crps != float("inf") else None

    eval_checkpoint_path = _select_eval_checkpoint_path(
        test_metric_source=test_metric_source,
        val_metric_source=val_metric_source,
        best_ckpt_path=best_ckpt_path,
        best_ckpt_path_raw=best_ckpt_path_raw,
        best_ckpt_path_ema=best_ckpt_path_ema,
        last_ckpt_path=last_ckpt_path,
    )
    final_test_eval_mode = _resolve_final_test_eval_mode(getattr(config, "FINAL_TEST_EVAL", "run"))
    loaded_checkpoint = str(eval_checkpoint_path) if eval_checkpoint_path is not None else None
    test_metrics: Dict[str, object]
    if final_test_eval_mode == "run":
        loaded_checkpoint = _load_eval_checkpoint(
            eval_checkpoint_path,
            diff_model=diff_model,
            ema=ema,
            device=device,
            config_obj=config,
            verbose=verbose,
        )

        test_metrics = evaluate_regression(
            diff_model,
            vae,
            laplace_summarizer,
            test_dl,
            device,
            mu_mean,
            mu_std,
            config,
            ema=_maybe_metric_ema(test_metric_source, ema),
            self_cond=bool(getattr(config, "SELF_COND", False)),
            verbose=debug,
            progress_enabled=verbose,
            progress_label="llapdiff test",
            **_sampling_kwargs(config, prefix="TEST"),
        )
        final_test_eval = {
            "status": "completed",
            "mode": final_test_eval_mode,
            "metric_source": test_metric_source,
            "checkpoint": loaded_checkpoint,
        }
    else:
        final_test_eval = {
            "status": "deferred" if final_test_eval_mode == "defer" else "skipped",
            "mode": final_test_eval_mode,
            "reason": f"FINAL_TEST_EVAL={final_test_eval_mode}",
            "metric_source": test_metric_source,
            "checkpoint": loaded_checkpoint,
        }
        test_metrics = dict(final_test_eval)
        if verbose:
            print(f"[test eval] {final_test_eval['status']} ({final_test_eval['reason']})")

    return {
        "benchmark_protocol": llapdiff_protocol_metadata(),
        "baseline_target_variance": baseline_target_variance,
        "latent_probe": latent_probe,
        "train_losses": train_losses,
        "train_history": train_history,
        "val_history": val_history,
        "raw_val_history": raw_val_history,
        "ema_val_history": ema_val_history,
        "pole_probe_history": pole_probe_history,
        "eval_stats": test_metrics,
        "final_test_eval": final_test_eval,
        "best_val": best_val,
        "best_val_by_source": {
            "raw": _finite_or_none(best_val_crps_by_source["raw"]),
            "ema": _finite_or_none(best_val_crps_by_source["ema"]),
        },
        "best_primary_metric": _finite_or_none(best_primary_metric),
        "best_primary_metric_name": primary_eval_metric,
        "selected_val_metric_source": val_metric_source,
        "selected_test_metric_source": test_metric_source,
        "data_policy": {
            "target_dim": int(target_dim),
            "vae_input_dim": int(vae_input_dim),
            "vae_output_dim": int(vae_output_dim),
            "split_policy": getattr(config, "split_policy", "global_purged_horizon"),
            "split_scope": getattr(config, "split_scope", "global_target_time"),
            "batching_policy": (
                "exact_context_end_timestamp"
                if bool(getattr(config, "exact_timestamp_batches", True))
                else "calendar_day"
            ),
            **split_protocol_metadata(
                getattr(config, "DATASET_KEY", ""),
                split_policy=getattr(config, "split_policy", "global_purged_horizon"),
                split_scope=getattr(config, "split_scope", "global_target_time"),
            ),
        },
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_raw": best_checkpoint_raw,
        "best_checkpoint_ema": best_checkpoint_ema,
        "last_checkpoint": last_checkpoint,
        "loaded_checkpoint": loaded_checkpoint,
        "checkpoint_dir": str(out_dir),
        "summ_ft_skipped_nonfinite_grad_steps": int(skipped_summary_ft_nonfinite_grad_steps),
        "summ_max_nonfinite_grad_steps": int(max_summary_nonfinite_grad_steps),
    }
