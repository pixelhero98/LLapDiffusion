
"""Utilities for LLapDiff training, latent handling, and diffusion scheduling."""

from __future__ import annotations

import inspect
import math
import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ============================
# Runtime helpers
# ============================

def set_torch(seed: int = 42, *, deterministic: bool = False) -> torch.device:
    """Set process seeds and return the active PyTorch device."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        return torch.device("cuda")
    return torch.device("cpu")


def sample_t_uniform(
    scheduler: "NoiseScheduler",
    n: int,
    device: torch.device,
    *,
    exclude_t0: bool = True,
) -> torch.Tensor:
    """Uniformly sample discrete diffusion steps."""
    t_min = 1 if exclude_t0 else 0
    t_max = int(scheduler.timesteps)
    if t_min >= t_max:
        raise ValueError(f"Invalid timestep range: [{t_min}, {t_max})")
    return torch.randint(t_min, t_max, (int(n),), device=device, dtype=torch.long)


@torch.no_grad()
def sample_t_uniform_karras(
    scheduler: "NoiseScheduler",
    n: int,
    device: torch.device,
    *,
    rho: float = 7.5,
    exclude_t0: bool = True,
) -> torch.Tensor:
    """
    Sample timesteps approximately uniformly in Karras sigma-space, then snap to nearest index.
    """
    ab = scheduler.alpha_bars.to(device=device, dtype=torch.float32)  # [T]
    sigmas = torch.sqrt((1.0 - ab) / (ab + 1e-12))  # [T], increasing with t

    t_min = 1 if exclude_t0 else 0
    sigma_min = sigmas[t_min].item()
    sigma_max = sigmas[-1].item()

    u = torch.rand(int(n), device=device, dtype=torch.float32)
    inv_rho = 1.0 / float(rho)
    target = (sigma_max ** inv_rho + u * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** float(rho)

    idx = torch.searchsorted(sigmas, target).clamp(min=t_min, max=sigmas.numel() - 1)
    idxm = (idx - 1).clamp(min=t_min)
    pick_lower = torch.abs(sigmas[idxm] - target) <= torch.abs(sigmas[idx] - target)
    t = torch.where(pick_lower, idxm, idx)
    return t.long()


def sample_training_timesteps(
    scheduler: "NoiseScheduler",
    n: int,
    device: torch.device,
    *,
    sampler: str = "uniform",
    karras_rho: float = 7.5,
    exclude_t0: bool = True,
) -> torch.Tensor:
    """Dispatch training-time timestep sampling strategies from config."""
    sampler_name = str(sampler).strip().lower()
    if sampler_name in {"uniform", "rand", "random"}:
        return sample_t_uniform(scheduler, n, device, exclude_t0=exclude_t0)
    if sampler_name in {"karras", "sigma", "sigma_uniform"}:
        return sample_t_uniform_karras(
            scheduler,
            n,
            device,
            rho=float(karras_rho),
            exclude_t0=exclude_t0,
        )
    raise ValueError(
        f"Unknown training timestep sampler '{sampler}'. Use 'uniform' or 'karras'."
    )


def make_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float = 0.05,
    base_lr: float = 5e-4,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a cosine scheduler with linear warmup."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if not (0.0 <= warmup_frac <= 1.0):
        raise ValueError(f"warmup_frac must be in [0, 1], got {warmup_frac}")

    warmup_steps = int(total_steps * warmup_frac)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        floor = min_lr / max(base_lr, 1e-12)
        if warmup_steps > 0 and step < warmup_steps:
            return max(floor, (step + 1) / warmup_steps)

        # Clamp progress so stepping beyond total_steps keeps the terminal LR.
        cosine_span = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / cosine_span
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(floor, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_warmup_constant(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float = 0.05,
    base_lr: float = 5e-4,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a constant scheduler with linear warmup."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if not (0.0 <= warmup_frac <= 1.0):
        raise ValueError(f"warmup_frac must be in [0, 1], got {warmup_frac}")

    warmup_steps = int(total_steps * warmup_frac)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        floor = min_lr / max(base_lr, 1e-12)
        if warmup_steps > 0 and step < warmup_steps:
            return max(floor, (step + 1) / warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_constant_lr(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a flat learning-rate schedule."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    *,
    schedule: str = "warmup_cosine",
    warmup_frac: float = 0.05,
    base_lr: float = 5e-4,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Factory for the small set of LR schedules we want to compare cleanly."""
    schedule_name = str(schedule).strip().lower()
    if schedule_name == "warmup_cosine":
        return make_warmup_cosine(
            optimizer,
            total_steps=total_steps,
            warmup_frac=warmup_frac,
            base_lr=base_lr,
            min_lr=min_lr,
        )
    if schedule_name == "warmup_constant":
        return make_warmup_constant(
            optimizer,
            total_steps=total_steps,
            warmup_frac=warmup_frac,
            base_lr=base_lr,
            min_lr=min_lr,
        )
    if schedule_name == "constant":
        return make_constant_lr(optimizer)
    raise ValueError(
        f"Unknown LR schedule '{schedule}'. Use 'warmup_cosine', 'warmup_constant', or 'constant'."
    )


def _cosine_alpha_bar(ts: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine cumulative noise schedule from Nichol & Dhariwal (2021)."""
    return torch.cos(((ts + s) / (1.0 + s)) * math.pi * 0.5).pow(2)


class NoiseScheduler(nn.Module):
    """
    Diffusion utilities with precomputed buffers and a DDIM sampler.

    Supports 'linear' or 'cosine' schedules and epsilon-/v-/x0-parameterization.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        if schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.schedule = schedule

        # ---- build betas ----
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
            betas = betas.clamp(min=1e-8, max=0.999)
            betas[0] = 0.0  # ensure ᾱ(0)=1 and no noise at t=0
        else:
            ts = torch.linspace(0.0, 1.0, self.timesteps, dtype=torch.float32)
            abar = _cosine_alpha_bar(ts)
            abar = abar / abar[0].clamp_min(1e-12)
            alphas = torch.ones(self.timesteps, dtype=torch.float32)
            if self.timesteps > 1:
                alphas[1:] = (abar[1:] / abar[:-1]).clamp(1e-8, 0.999999)
            betas = (1.0 - alphas)
            betas[0] = 0.0
            if self.timesteps > 1:
                betas[1:] = betas[1:].clamp(min=1e-8, max=0.999)

        self.register_buffer("betas", betas)
        alphas = (1.0 - betas).clamp(1e-12, 1.0)
        self.register_buffer("alphas", alphas)
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)

        ab = alpha_bars.clamp(0.0, 1.0)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(ab))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt((1.0 - ab).clamp(0.0, 1.0)))

    def _gather(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_idx = t.clamp(min=0, max=self.timesteps - 1).to(device=buf.device, dtype=torch.long)
        return buf.gather(0, t_idx)

    def _expand_like(self, buf: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self._gather(buf, t).view(-1, *([1] * (ref.dim() - 1)))

    @torch.no_grad()
    def alpha_bar_at(self, t: torch.Tensor) -> torch.Tensor:
        """
        ᾱ(t) for possibly non-integer t in [0, T-1] via linear interpolation.
        Matches self.alpha_bars[t] exactly when t is integer.
        """
        t = t.to(self.alpha_bars.device, dtype=torch.float32)
        t0 = t.floor().clamp(0, self.timesteps - 1)
        t1 = (t0 + 1).clamp(0, self.timesteps - 1)
        w = (t - t0).clamp(0.0, 1.0)
        ab0 = self.alpha_bars.index_select(0, t0.long())
        ab1 = self.alpha_bars.index_select(0, t1.long())
        return (1.0 - w) * ab0 + w * ab1

    @torch.no_grad()
    def snr_at(self, t: torch.Tensor) -> torch.Tensor:
        abar = self.alpha_bar_at(t).clamp(1e-6, 1.0 - 1e-6)
        return abar / (1.0 - abar)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from the forward process and return the noise used."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._expand_like(self.sqrt_alpha_bars, t, x0)
        sqrt_1_ab = self._expand_like(self.sqrt_one_minus_alpha_bars, t, x0)
        x_t = sqrt_ab * x0 + sqrt_1_ab * noise
        return x_t, noise  # ε_true

    def pred_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha = self._expand_like(self.sqrt_alpha_bars, t, x_t)
        sigma = self._expand_like(self.sqrt_one_minus_alpha_bars, t, x_t)
        return (x_t - sigma * eps) / (alpha + 1e-12)

    def pred_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        alpha = self._expand_like(self.sqrt_alpha_bars, t, x_t)
        sigma = self._expand_like(self.sqrt_one_minus_alpha_bars, t, x_t)
        return (x_t - alpha * x0) / (sigma + 1e-12)

    def pred_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = self._expand_like(self.sqrt_alpha_bars, t, x_t)
        sigma = self._expand_like(self.sqrt_one_minus_alpha_bars, t, x_t)
        return alpha * x_t - sigma * v

    def pred_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = self._expand_like(self.sqrt_alpha_bars, t, x_t)
        sigma = self._expand_like(self.sqrt_one_minus_alpha_bars, t, x_t)
        return sigma * x_t + alpha * v

    def v_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha = self._expand_like(self.sqrt_alpha_bars, t, x_t)
        sigma = self._expand_like(self.sqrt_one_minus_alpha_bars, t, x_t)
        return (eps - sigma * x_t) / (alpha + 1e-12)

    def to_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor, param_type: str) -> torch.Tensor:
        if param_type == "eps":
            return self.pred_x0_from_eps(x_t, t, pred)
        if param_type == "v":
            return self.pred_x0_from_v(x_t, t, pred)
        if param_type == "x0":
            return pred
        raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    def to_eps(self, x_t: torch.Tensor, t: torch.Tensor, pred: torch.Tensor, param_type: str) -> torch.Tensor:
        if param_type == "eps":
            return pred
        if param_type == "v":
            return self.pred_eps_from_v(x_t, t, pred)
        if param_type == "x0":
            return self.pred_eps_from_x0(x_t, t, pred)
        raise ValueError("param_type must be 'eps', 'v', or 'x0'")

    @torch.no_grad()
    def ddim_sigma(self, t: torch.Tensor, t_prev: torch.Tensor, eta: float) -> torch.Tensor:
        ab_t = self._gather(self.alpha_bars, t).clamp(1e-12, 1.0)
        ab_prev = self._gather(self.alpha_bars, t_prev).clamp(1e-12, 1.0)
        sigma = (
            eta
            * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t))
            * torch.sqrt((1.0 - (ab_t / (ab_prev + 1e-12))).clamp_min(0.0))
        )
        return sigma.view(-1)

    @torch.no_grad()
    def ddim_step_from(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        pred: torch.Tensor,
        param_type: str,
        eta: float = 0.0,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_t)
        ab_prev = self._expand_like(self.alpha_bars, t_prev, x_t).clamp(1e-12, 1.0)
        sigma = self.ddim_sigma(t, t_prev, eta).view(-1, *([1] * (x_t.dim() - 1)))
        x0_pred = self.to_x0(x_t, t, pred, param_type)
        eps_pred = self.to_eps(x_t, t, pred, param_type)
        dir_coeff = ((1.0 - ab_prev) - sigma**2).clamp_min(0.0)
        x_prev = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(dir_coeff) * eps_pred + sigma * noise
        return x_prev


# ============================
# VAE latent stats helpers
# ============================

def flatten_targets(
    yb: torch.Tensor,
    mask_bn: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """yb: [B,N,H] or [B,N,H,C] -> y_in: [Beff,H,C], batch_ids: [Beff]."""
    y_bhnc = targets_to_bhnc(yb, mask_bn, device=device)
    if y_bhnc is None:
        return None, None
    y_bnhc = y_bhnc.permute(0, 2, 1, 3).contiguous()
    finite_mask = torch.isfinite(y_bnhc).all(dim=(2, 3))

    y = torch.nan_to_num(y_bnhc, nan=0.0, posinf=0.0, neginf=0.0)
    B, N, Hcur, C = y.shape
    y_flat = y.reshape(B * N, Hcur, C)

    mask = mask_bn.to(device=device, dtype=torch.bool).reshape(B * N)
    m_flat = mask & finite_mask.reshape(B * N)
    if not m_flat.any():
        return None, None

    y_in = y_flat[m_flat]
    batch_ids = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand(B, N)
        .reshape(B * N)[m_flat]
    )
    return y_in, batch_ids


@torch.no_grad()
def _flatten_for_mask(yb, mask_bn, device):
    return flatten_targets(yb, mask_bn, device)


# ============================
# EMA
# ============================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self._backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    def store(self, model: nn.Module):
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def copy_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n].data)

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data.copy_(self._backup[n].data)

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()


# ============================
# Latent helpers
# ============================

def simple_norm(
    mu: torch.Tensor,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
    clip_val: Optional[float] = None,
) -> torch.Tensor:
    """Dataset-level per-dim z-score on latent means."""
    mu_mean = _broadcast_norm_stats(mu_mean, mu)
    mu_std = _broadcast_norm_stats(mu_std, mu).clamp_min(1e-6)
    x = (mu - mu_mean) / mu_std
    if clip_val is not None:
        x = x.clamp(-clip_val, clip_val)
    return x


def invert_simple_norm(x: torch.Tensor, mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    mu_mean = _broadcast_norm_stats(mu_mean, x)
    mu_std = _broadcast_norm_stats(mu_std, x)
    return x * mu_std + mu_mean


def _broadcast_norm_stats(stats: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast latent normalization stats of shape [Z] or [H,Z] to match [B,H,Z]."""
    stats = torch.as_tensor(stats, device=ref.device, dtype=ref.dtype)
    if stats.dim() == 1:
        if stats.numel() != ref.size(-1):
            raise ValueError(
                f"Latent stat width mismatch: stats has {stats.numel()} dims, ref expects {ref.size(-1)}."
            )
        return stats.view(1, 1, -1)
    if stats.dim() == 2:
        if stats.shape != ref.shape[1:]:
            raise ValueError(
                f"Latent stat shape mismatch: got {tuple(stats.shape)}, expected {(ref.size(1), ref.size(2))}."
            )
        return stats.unsqueeze(0)
    raise ValueError(
        f"Latent stats must have shape [Z] or [H,Z], got {tuple(stats.shape)}."
    )


def normalize_cond_per_batch(cs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """z-score over (B,S) for each feature dim; keeps gradients."""
    m = cs.mean(dim=(0, 1), keepdim=True)
    v = cs.var(dim=(0, 1), keepdim=True, unbiased=False)
    return (cs - m) / (v.sqrt() + eps)


def infer_target_dim(yb: torch.Tensor, mask_bn: Optional[torch.Tensor] = None) -> int:
    """Return the target channel count for scalar or multi-target batches."""
    y = torch.as_tensor(yb)
    if y.dim() == 3:
        return 1
    if y.dim() != 4:
        raise ValueError(f"target tensor must be [B,N,H] or [B,N,H,C], got {tuple(y.shape)}")
    if mask_bn is not None:
        mask = torch.as_tensor(mask_bn)
        if mask.dim() != 2 or mask.shape[0] != y.shape[0]:
            raise ValueError(f"entity mask shape {tuple(mask.shape)} is incompatible with target shape {tuple(y.shape)}")
        if y.shape[1] != mask.shape[1] and y.shape[2] != mask.shape[1]:
            raise ValueError(f"target shape {tuple(y.shape)} does not contain entity axis N={mask.shape[1]}")
    return int(y.shape[-1])


def infer_target_dim_from_loader(loader) -> int:
    """Infer target channel count from the first batch of a dataloader."""
    try:
        _, yb, meta = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("Cannot infer target_dim from an empty dataloader.") from exc
    return infer_target_dim(yb, meta.get("entity_mask") if isinstance(meta, dict) else None)


def vae_io_dims_for_target_dim(config_obj: object, target_dim: int) -> Tuple[int, int]:
    """Resolve VAE token input/output dimensions for a target channel count."""
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    expected_input_dim = 2 * target_dim
    configured_input_dim = int(getattr(config_obj, "VAE_INPUT_DIM", expected_input_dim))
    configured_output_dim = int(getattr(config_obj, "VAE_OUTPUT_DIM", target_dim))

    if target_dim == 1:
        if configured_input_dim != expected_input_dim:
            raise ValueError(f"Scalar target VAE expects VAE_INPUT_DIM=2, got {configured_input_dim}")
        if configured_output_dim != 1:
            raise ValueError(f"Scalar target VAE expects VAE_OUTPUT_DIM=1, got {configured_output_dim}")
        return configured_input_dim, configured_output_dim

    if configured_output_dim == 1 and configured_input_dim in {2, expected_input_dim}:
        return expected_input_dim, target_dim
    if configured_input_dim != expected_input_dim or configured_output_dim != target_dim:
        raise ValueError(
            "Multi-target VAE dims must match selected targets: "
            f"target_dim={target_dim}, expected VAE_INPUT_DIM={expected_input_dim}, "
            f"VAE_OUTPUT_DIM={target_dim}; got {configured_input_dim}/{configured_output_dim}."
        )
    return configured_input_dim, configured_output_dim


def targets_to_bhnc(
    yb: torch.Tensor,
    mask_bn: torch.Tensor,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Canonicalize target values to [B,H,N,C]."""
    y = torch.as_tensor(yb, device=device)
    mask = torch.as_tensor(mask_bn, device=device, dtype=torch.bool)
    if mask.dim() != 2:
        return None
    B, N = mask.shape
    if y.dim() == 3:
        if y.shape[:2] != (B, N):
            return None
        return y.permute(0, 2, 1).contiguous().unsqueeze(-1)
    if y.dim() == 4:
        if y.shape[0] != B:
            return None
        if y.shape[1] == N:
            return y.permute(0, 2, 1, 3).contiguous()
        if y.shape[2] == N:
            return y.contiguous()
    return None


def target_obs_mask_to_bhnc(
    y_obs_mask: Optional[torch.Tensor],
    y_bhnc: torch.Tensor,
    mask_bn: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Canonicalize optional target observation masks to [B,H,N,C]."""
    B, H, N, C = y_bhnc.shape
    mask_bn = torch.as_tensor(mask_bn, device=device, dtype=torch.bool)
    if y_obs_mask is None:
        if torch.is_floating_point(y_bhnc):
            obs = torch.isfinite(y_bhnc)
        else:
            obs = torch.ones_like(y_bhnc, dtype=torch.bool)
        return obs & mask_bn[:, None, :, None]

    obs = torch.as_tensor(y_obs_mask, device=device, dtype=torch.bool)
    if obs.shape == (B, N, H):
        obs = obs.permute(0, 2, 1).contiguous().unsqueeze(-1).expand(B, H, N, C)
    elif obs.shape == (B, H, N):
        obs = obs.unsqueeze(-1).expand(B, H, N, C)
    elif obs.shape == (B, N, H, C):
        obs = obs.permute(0, 2, 1, 3).contiguous()
    elif obs.shape == (B, H, N, C):
        obs = obs.contiguous()
    elif obs.shape == (B, N, H, 1):
        obs = obs.permute(0, 2, 1, 3).contiguous().expand(B, H, N, C)
    elif obs.shape == (B, H, N, 1):
        obs = obs.expand(B, H, N, C)
    else:
        raise ValueError(
            f"y_obs_mask shape {tuple(obs.shape)} is incompatible with target shape [B,H,N,C]={(B, H, N, C)}"
        )

    bad_observed = obs & mask_bn[:, None, :, None] & ~torch.isfinite(y_bhnc)
    if bad_observed.any():
        raise ValueError("y_obs_mask marks non-finite target values as observed")
    return obs & torch.isfinite(y_bhnc) & mask_bn[:, None, :, None]


def target_time_observed(obs: torch.Tensor) -> torch.Tensor:
    """Reduce target observations to [B,H] latent-supervision availability."""
    obs = torch.as_tensor(obs, dtype=torch.bool)
    if obs.dim() == 3:
        return obs.any(dim=2)
    if obs.dim() == 4:
        return obs.any(dim=(2, 3))
    raise ValueError(f"target observation mask must be [B,H,N] or [B,H,N,C], got {tuple(obs.shape)}")


@torch.no_grad()
def pack_targets_tokens(
    yb: torch.Tensor,
    mask_bn: torch.Tensor,
    device: torch.device,
    *,
    y_obs_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepare tokenized targets for the set-attention VAE.

    Returns:
        x_tok: [B,H,N,2*C] = [values*obs, obs]
        entity_pad: [B,N] bool (True for padded entities)
        obs: [B,H,N,C] bool
    """
    mask_bn = torch.as_tensor(mask_bn, device=device, dtype=torch.bool)
    y = targets_to_bhnc(yb, mask_bn, device=device)
    if y is None:
        return None, None, None

    obs = target_obs_mask_to_bhnc(y_obs_mask, y, mask_bn, device=device)
    y_clean = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    x_val = y_clean * obs.to(dtype=y_clean.dtype)
    x_tok = torch.cat([x_val, obs.to(dtype=y_clean.dtype)], dim=-1)  # [B,H,N,2*C]
    entity_pad = ~mask_bn
    return x_tok, entity_pad, obs


def compute_latent_stats(
    vae: nn.Module,
    dataloader,
    device: torch.device,
    *,
    mode: str = "global",
):
    """
    Compute dataset-level latent mean/std on raw μ for the set-attention VAE.
    """
    mode_name = str(mode).strip().lower()
    if mode_name not in {"global", "per_horizon", "horizon"}:
        raise ValueError(
            f"Unknown latent stats mode '{mode}'. Use 'global' or 'per_horizon'."
        )

    mu_sum = mu_sumsq = mu_count = None
    for _, yb, meta in dataloader:
        mask_bn = meta["entity_mask"].to(device=device, dtype=torch.bool)
        yb = yb.to(device)
        y_obs_mask = meta.get("y_obs_mask", None)
        if y_obs_mask is not None:
            y_obs_mask = torch.as_tensor(y_obs_mask, device=device, dtype=torch.bool)

        x_tok, entity_pad, obs = pack_targets_tokens(yb, mask_bn, device=device, y_obs_mask=y_obs_mask)
        if x_tok is None:
            continue

        _, mu, _ = vae(x_tok, entity_pad)  # [B,H,Z]
        if not torch.isfinite(mu).all():
            raise FloatingPointError("VAE encoder produced non-finite latent means")
        obs_any = target_time_observed(obs)  # [B,H]
        if not obs_any.any():
            continue
        mu = mu.detach().float()

        if mode_name == "global":
            mu_obs = mu[obs_any]
            if mu_obs.numel() == 0:
                continue
            batch_sum = mu_obs.sum(dim=0).cpu().to(dtype=torch.float64)
            batch_sumsq = mu_obs.square().sum(dim=0).cpu().to(dtype=torch.float64)
            batch_count = torch.tensor(float(mu_obs.shape[0]), dtype=torch.float64)
        else:
            obs_f = obs_any.unsqueeze(-1).to(dtype=mu.dtype)
            batch_sum = (mu * obs_f).sum(dim=0).cpu().to(dtype=torch.float64)
            batch_sumsq = (mu.square() * obs_f).sum(dim=0).cpu().to(dtype=torch.float64)
            batch_count = obs_f.sum(dim=0).cpu().to(dtype=torch.float64)

        if mu_sum is None:
            mu_sum = batch_sum
            mu_sumsq = batch_sumsq
            mu_count = batch_count
        else:
            mu_sum += batch_sum
            mu_sumsq += batch_sumsq
            mu_count += batch_count

    if mu_sum is None or mu_count is None:
        raise RuntimeError("No valid latent samples found after filtering masks/non-finite values.")

    denom = mu_count.clamp_min(1.0)
    mu_mean = (mu_sum / denom).to(device=device, dtype=torch.float32)
    mu_var = (mu_sumsq / denom).to(device=device, dtype=torch.float32) - mu_mean.square()
    mu_std = mu_var.clamp_min(0.0).sqrt().clamp_min(1e-6)
    return mu_mean, mu_std


def decode_latents_with_vae(
    vae,
    x0_norm: torch.Tensor,
    *,
    entity_pad: torch.Tensor,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
) -> torch.Tensor:
    """
    Invert μ-normalization and decode set-attention VAE.
    Returns x_hat: [B,H,N,C]
    """
    mu_est = invert_simple_norm(x0_norm, mu_mean, mu_std)
    if mu_est.dim() != 3:
        raise ValueError(f"mu_est must be [B,H,Z], got {tuple(mu_est.shape)}")

    B, H, Z = mu_est.shape
    entity_pad = entity_pad.to(device=mu_est.device, dtype=torch.bool).contiguous()
    if entity_pad.dim() != 2 or entity_pad.shape[0] != B:
        raise ValueError(f"entity_pad must be [B,N], got {tuple(entity_pad.shape)}")
    N = entity_pad.shape[1]

    expected_z = getattr(vae, "latent_channel", None)
    if expected_z is not None and int(expected_z) != Z:
        raise ValueError(
            f"Latent width mismatch: x0_norm decodes to Z={Z}, but VAE expects latent_channel={int(expected_z)}. "
            "LLapDiff should predict normalized VAE μ latents (shape [B,H,C]) where C == VAE_LATENT_CHANNELS."
        )

    if hasattr(vae, "decode_mu"):
        return vae.decode_mu(mu_est, entity_pad)

    z = mu_est.reshape(B * H, Z)
    pad_bt = entity_pad.unsqueeze(1).expand(B, H, N).reshape(B * H, N)

    dec = vae.z_proj(z).unsqueeze(1).expand(-1, N, -1)
    dec = vae.decoder(dec, key_padding_mask=pad_bt)
    x_hat_bt = vae.out_proj(dec)
    return x_hat_bt.reshape(B, H, N, 1)

def build_context(
    context_module: nn.Module,
    V: torch.Tensor,
    T: torch.Tensor,
    mask_bn: torch.Tensor,
    device: torch.device,
    *,
    dt: Optional[torch.Tensor] = None,
    x_obs_mask: Optional[torch.Tensor] = None,
    norm: bool = True,
    requires_grad: bool = False,
):
    """
    Build the history summary embedding E_ti.

    Returns:
        cond_summary: [B,S,Hm]
    """
    series_diff = T.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
    series = V.permute(0, 2, 1, 3).to(device)       # [B,K,N,F]
    mask_bn = mask_bn.to(device=device, dtype=torch.bool)

    if mask_bn.dtype != torch.bool:
        raise TypeError(f"entity_mask must be bool, got {mask_bn.dtype}")

    mask_val = mask_bn[:, None, :, None].to(dtype=series.dtype, device=device)
    series = series * mask_val
    series_diff = series_diff * mask_val

    if dt is not None:
        dt = torch.as_tensor(dt, dtype=torch.float32, device=device)
        if dt.dim() == 4 and dt.size(-1) == 1:
            dt = dt.squeeze(-1)

    obs_mask = None
    if x_obs_mask is not None:
        obs_mask = torch.as_tensor(x_obs_mask, device=device)
        if obs_mask.dim() == 4:
            if obs_mask.size(1) == series.size(2) and obs_mask.size(2) == series.size(1):
                obs_mask = obs_mask.permute(0, 2, 1, 3).contiguous()
        elif obs_mask.dim() == 3:
            if obs_mask.size(1) == series.size(2) and obs_mask.size(2) == series.size(1):
                obs_mask = obs_mask.permute(0, 2, 1).contiguous()

        if obs_mask.dim() == 4:
            obs_mask = obs_mask.bool() & mask_bn[:, None, :, None]
        elif obs_mask.dim() == 3:
            obs_mask = obs_mask.bool() & mask_bn[:, None, :]
        else:
            raise ValueError(f"x_obs_mask must have 3 or 4 dims, got {tuple(obs_mask.shape)}")

    if context_module is None:
        raise AttributeError("context_module must be provided to build_context.")

    frozen = not any(p.requires_grad for p in context_module.parameters())
    grad_guard = torch.enable_grad if (requires_grad or not frozen) else torch.no_grad

    forward_params = inspect.signature(context_module.forward).parameters
    supports_obs_mask = "obs_mask" in forward_params
    supports_pad_mask = "pad_mask" in forward_params

    with grad_guard():
        kwargs = {"x": series, "ctx_diff": series_diff, "dt": dt}
        if supports_pad_mask:
            kwargs["pad_mask"] = mask_bn
        if supports_obs_mask:
            kwargs["obs_mask"] = obs_mask
        cond_summary, _ = context_module(**kwargs)

    if norm:
        cond_summary = normalize_cond_per_batch(cond_summary)
    if not requires_grad:
        return cond_summary.detach()
    return cond_summary


def encode_mu_norm(
    vae,
    x_tok: torch.Tensor,
    *,
    entity_pad: Optional[torch.Tensor] = None,
    mu_mean: torch.Tensor,
    mu_std: torch.Tensor,
) -> torch.Tensor:
    """Encode with the set-VAE and z-score the resulting μ."""
    with torch.no_grad():
        _, mu, _ = vae(x_tok, entity_pad)
    if not torch.isfinite(mu).all():
        raise FloatingPointError("VAE encoder produced non-finite latent means")
    mu_norm = simple_norm(mu, mu_mean, mu_std, clip_val=None)
    if not torch.isfinite(mu_norm).all():
        raise FloatingPointError("Normalized VAE latent means are non-finite")
    return mu_norm


def diffusion_loss(
    model,
    scheduler: NoiseScheduler,
    x0_lat_norm: torch.Tensor,
    t: torch.Tensor,
    *,
    cond_summary: Optional[torch.Tensor],
    cond_summary_raw: Optional[torch.Tensor] = None,
    predict_type: str = "v",
    weight_scheme: str = "none",
    minsnr_gamma: float = 5.0,
    sc_feat: Optional[torch.Tensor] = None,
    dt: Optional[torch.Tensor] = None,
    reuse_xt_eps: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    target_mask: Optional[torch.Tensor] = None,
    minsnr_normalize: str = "auto",
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    MSE on x0/v/eps with optional horizon masking and MinSNR weighting.

    target_mask:
        Optional boolean mask aligned to [B, H] (or broadcastable to that),
        used to ignore timesteps/horizons that have no observed supervision.
    """
    if reuse_xt_eps is None:
        noise = torch.randn_like(x0_lat_norm)
        x_t, eps_true = scheduler.q_sample(x0_lat_norm, t, noise)
    else:
        x_t, eps_true = reuse_xt_eps

    pred = model(
        x_t,
        t,
        cond_summary=cond_summary,
        cond_summary_raw=cond_summary_raw,
        sc_feat=sc_feat,
        dt=dt,
    )

    if predict_type == "eps":
        target = eps_true
    elif predict_type == "v":
        target = scheduler.v_from_eps(x_t, t, eps_true)
    elif predict_type == "x0":
        target = x0_lat_norm
    else:
        raise ValueError(
            f"Unknown predict_type '{predict_type}'. Use 'x0', 'v', or 'eps'."
        )

    err = (pred - target).pow(2)

    per_sample = _reduce_loss_per_sample(err, target_mask=target_mask)

    if weight_scheme == "none":
        weights_raw = torch.ones_like(per_sample)
    elif weight_scheme == "weighted_min_snr":
        weights_raw = _minsnr_weights(
            scheduler,
            t,
            gamma=minsnr_gamma,
            predict_type=predict_type,
        ).to(device=per_sample.device, dtype=per_sample.dtype)
    else:
        raise ValueError(
            f"Unknown weight_scheme '{weight_scheme}'. Use 'none' or 'weighted_min_snr'."
        )

    weights = _normalize_loss_weights(
        weights_raw,
        scheduler=scheduler,
        gamma=minsnr_gamma,
        predict_type=predict_type,
        normalize=minsnr_normalize,
    ).to(device=per_sample.device, dtype=per_sample.dtype).detach()
    weighted_per_sample = weights * per_sample
    loss = weighted_per_sample.mean()
    if not return_stats:
        return loss

    stats = {
        "raw_loss": per_sample.mean().detach(),
        "weighted_loss": loss.detach(),
        "per_sample_raw": per_sample.detach(),
        "per_sample_weighted": weighted_per_sample.detach(),
        "weights": weights.detach(),
        "weights_raw": weights_raw.detach(),
    }
    return loss, stats


def _reduce_loss_per_sample(
    err: torch.Tensor,
    *,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce elementwise prediction errors to per-sample losses."""
    if target_mask is not None:
        mask = torch.as_tensor(target_mask, device=err.device, dtype=torch.bool)

        # Accept [B,H] or [B,H,1...] and broadcast over feature dim(s)
        if mask.dim() == err.dim() - 1:
            mask = mask.unsqueeze(-1)
        while mask.dim() < err.dim():
            mask = mask.unsqueeze(-1)

        if mask.shape[0] != err.shape[0] or mask.shape[1] != err.shape[1]:
            raise ValueError(
                f"target_mask must align with [B,H,...]; "
                f"got {tuple(mask.shape)} for err {tuple(err.shape)}"
            )

        mask_f = mask.to(dtype=err.dtype)
        err = err * mask_f

        reduce_dims = tuple(range(1, err.ndim))
        denom = mask_f.expand_as(err).sum(dim=reduce_dims).clamp_min(1.0)
        return err.sum(dim=reduce_dims) / denom

    reduce_dims = tuple(range(1, err.ndim))
    return err.mean(dim=reduce_dims)


def _minsnr_weights(
    scheduler: NoiseScheduler,
    t: torch.Tensor,
    *,
    gamma: float,
    predict_type: str,
) -> torch.Tensor:
    """Per-sample Min-SNR loss weights for eps/x0/v targets."""
    snr = scheduler.snr_at(t).to(dtype=torch.float32)
    clipped = snr.clamp(max=max(float(gamma), 0.0))

    if predict_type == "eps":
        return clipped / snr.clamp_min(1e-8)
    if predict_type == "x0":
        return clipped
    if predict_type == "v":
        return clipped / (snr + 1.0)
    raise ValueError(f"Unknown predict_type: {predict_type}")


def _normalize_loss_weights(
    weights_raw: torch.Tensor,
    *,
    scheduler: NoiseScheduler,
    gamma: float,
    predict_type: str,
    normalize: str = "auto",
    exclude_t0: bool = True,
) -> torch.Tensor:
    """Normalize Min-SNR weights without collapsing date-batched B=1 training to all ones."""
    mode = str(normalize).strip().lower()
    if mode == "none":
        denom = torch.ones((), device=weights_raw.device, dtype=weights_raw.dtype)
    elif mode == "batch":
        denom = weights_raw.mean().clamp_min(1e-8)
    elif mode in {"global", "auto"}:
        if mode == "auto" and weights_raw.numel() > 1:
            denom = weights_raw.mean().clamp_min(1e-8)
        else:
            denom = _minsnr_weight_reference(
                scheduler,
                gamma=gamma,
                predict_type=predict_type,
                exclude_t0=exclude_t0,
            ).to(device=weights_raw.device, dtype=weights_raw.dtype)
    else:
        raise ValueError(
            f"Unknown minsnr normalization '{normalize}'. Use 'auto', 'batch', 'global', or 'none'."
        )
    return weights_raw / denom.clamp_min(1e-8)


def _minsnr_weight_reference(
    scheduler: NoiseScheduler,
    *,
    gamma: float,
    predict_type: str,
    exclude_t0: bool = True,
) -> torch.Tensor:
    """Return a batch-size-independent reference weight over the scheduler support."""
    cache = getattr(scheduler, "_minsnr_weight_reference_cache", None)
    if cache is None:
        cache = {}
        setattr(scheduler, "_minsnr_weight_reference_cache", cache)

    key = (bool(exclude_t0), round(float(gamma), 8), str(predict_type))
    if key not in cache:
        t_min = 1 if exclude_t0 else 0
        t_ref = torch.arange(
            t_min,
            int(scheduler.timesteps),
            device=scheduler.alpha_bars.device,
            dtype=torch.long,
        )
        ref = _minsnr_weights(
            scheduler,
            t_ref,
            gamma=gamma,
            predict_type=predict_type,
        ).mean().clamp_min(1e-8)
        cache[key] = ref.detach()
    return cache[key]


@torch.no_grad()
def calculate_v_variance(
    dataloader,
    device: torch.device,
    scheduler: NoiseScheduler,
    vae=None,
    mu_mean=None,
    mu_std=None,
):
    """Estimate Var[v] in latent space (set-VAE path when available)."""
    all_v_targets = []
    for xb, yb, meta in dataloader:
        mask_bn = meta["entity_mask"].to(device=device, dtype=torch.bool)
        if not mask_bn.any():
            continue

        if vae is not None and (mu_mean is not None) and (mu_std is not None):
            y_obs_mask = meta.get("y_obs_mask")
            x_tok, entity_pad, obs = pack_targets_tokens(yb, mask_bn, device, y_obs_mask=y_obs_mask)
            if x_tok is None or not obs.any():
                continue
            x0 = encode_mu_norm(vae, x_tok, entity_pad=entity_pad, mu_mean=mu_mean, mu_std=mu_std)
            obs_any = target_time_observed(obs)
            if not obs_any.any():
                continue
            x0 = x0[obs_any]
        else:
            y_in, _ = flatten_targets(yb, mask_bn, device)
            if y_in is None:
                continue
            x0 = y_in

        B = x0.size(0)
        t = sample_t_uniform(scheduler, B, device)
        eps_true = torch.randn_like(x0)
        x_t, _ = scheduler.q_sample(x0, t, eps_true)
        v_target = scheduler.v_from_eps(x_t, t, eps_true)
        all_v_targets.append(v_target.detach().cpu())

    if not all_v_targets:
        raise RuntimeError("Cannot estimate v-target variance from an empty dataloader/mask set")
    all_v = torch.cat(all_v_targets, dim=0)
    return all_v.var(unbiased=False).item()


@torch.no_grad()
def calculate_x0_variance(
    dataloader,
    device: torch.device,
    latent_encoder=None,
):
    """
    Estimate Var[x0] from flattened targets or an optional latent encoder.
    """
    vals = []
    for _, yb, meta in dataloader:
        mask_bn = meta["entity_mask"].to(device=device, dtype=torch.bool)
        if not mask_bn.any():
            continue
        y_in, _ = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue
        x0 = y_in
        if latent_encoder is not None:
            try:
                x0 = latent_encoder(x0)
            except TypeError:
                x0 = latent_encoder(y_in)
        vals.append(x0.detach().cpu())
    if not vals:
        raise RuntimeError("Cannot estimate x0 variance from an empty dataloader/mask set")
    x0_cat = torch.cat(vals, dim=0)
    return x0_cat.var(unbiased=False).item()


@torch.no_grad()
def calculate_x0_latent_variance_setvae(dataloader, vae, device, latent_stats):
    """
    Estimate Var[x0] in normalized set-VAE latent space where x0 := mu_norm.
    """
    if latent_stats is None:
        raise ValueError("latent_stats must be provided as (mu_mean, mu_std).")
    mu_mean, mu_std = latent_stats

    all_x0 = []
    for _, yb, meta in dataloader:
        mask_bn = meta["entity_mask"].to(device=device, dtype=torch.bool)
        if not mask_bn.any():
            continue
        y_obs_mask = meta.get("y_obs_mask")
        x_tok, entity_pad, obs = pack_targets_tokens(yb, mask_bn, device, y_obs_mask=y_obs_mask)
        if x_tok is None or obs is None or not obs.any():
            continue
        x0 = encode_mu_norm(vae, x_tok, entity_pad=entity_pad, mu_mean=mu_mean, mu_std=mu_std)
        obs_any = target_time_observed(obs)
        if obs_any.any():
            all_x0.append(x0[obs_any].detach().cpu())

    if not all_x0:
        raise RuntimeError("Cannot estimate latent x0 variance from an empty dataloader/mask set")
    x0_cat = torch.cat(all_x0, dim=0)
    return x0_cat.var(unbiased=False).item()


def calculate_target_variance(
    *,
    predict_type: str,
    dataloader,
    device,
    scheduler: Optional[NoiseScheduler] = None,
    latent_encoder=None,
    vae=None,
    latent_stats=None,
):
    """Dispatch variance calculation based on prediction parameterization."""
    if predict_type == "v":
        if scheduler is None or dataloader is None or vae is None or latent_stats is None:
            raise ValueError("scheduler, dataloader, vae, and latent_stats are required for v variance.")
        mu_mean, mu_std = latent_stats
        return calculate_v_variance(dataloader, device, scheduler, vae=vae, mu_mean=mu_mean, mu_std=mu_std)

    if predict_type == "eps":
        return 1.0

    if predict_type == "x0":
        if dataloader is None:
            raise ValueError("dataloader is required for x0 variance calculation.")
        if vae is not None and latent_stats is not None:
            return calculate_x0_latent_variance_setvae(dataloader, vae, device, latent_stats)
        return calculate_x0_variance(dataloader, device, latent_encoder=latent_encoder)

    raise ValueError("predict_type must be one of 'x0', 'v', or 'eps'.")
