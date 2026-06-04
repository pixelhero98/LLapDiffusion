
"""Latent Laplace diffusion model."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from llapdiffusion.models.lapformer import LapFormer
from llapdiffusion.models.llapdiff_utils import NoiseScheduler

class LLapDiff(nn.Module):
    """
    Latent Laplace Diffusion for multivariate time series with global conditioning.

    The backbone always uses LapFormer's attention-based Laplace analysis.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        *,
        predict_type: str = "v",
        laplace_k: int = 32,
        timesteps: int = 1000,
        schedule: str = "cosine",
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        self_conditioning: bool = False,
        summary_pool_mode: str = "mean",
        pole_pool_use_raw_summary: bool = False,
        block_summary_adaln: bool = False,
        analysis_summary_qk: bool = False,
        analysis_qk_use_raw_summary: bool = False,
        rho_conditioning_mode: str = "raw",
    ) -> None:
        super().__init__()
        if predict_type not in {"eps", "v", "x0"}:
            raise ValueError("predict_type must be either 'eps', 'v', or 'x0'")

        self.predict_type = predict_type
        self.self_conditioning = bool(self_conditioning)
        self.data_dim = int(data_dim)

        # Diffusion scheduler utilities
        self.scheduler = NoiseScheduler(timesteps=timesteps, schedule=schedule)

        # Learned timestep embedding
        self.time_embed = nn.Embedding(timesteps, hidden_dim)
        nn.init.normal_(self.time_embed.weight, std=0.02)

        # Main LapFormer backbone
        self.model = LapFormer(
            input_dim=self.data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            laplace_k=laplace_k,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_conditioning=self_conditioning,
            summary_pool_mode=summary_pool_mode,
            pole_pool_use_raw_summary=pole_pool_use_raw_summary,
            block_summary_adaln=block_summary_adaln,
            analysis_summary_qk=analysis_summary_qk,
            analysis_qk_use_raw_summary=analysis_qk_use_raw_summary,
            rho_conditioning_mode=rho_conditioning_mode,
        )
        self.time_dim = hidden_dim

    # -------------------------------
    # Embeddings & conditioning
    # -------------------------------
    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        return F.silu(self.time_embed(t.long()))

    # -------------------------------
    # Forward call
    # -------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        cond_summary: Optional[torch.Tensor] = None,
        cond_summary_raw: Optional[torch.Tensor] = None,
        sc_feat: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self._time_embed(t).to(x_t.dtype)
        out_tokens = self.model(
            x_t,
            t_emb,
            cond_summary=cond_summary,
            cond_summary_raw=cond_summary_raw,
            sc_feat=sc_feat,
            dt=dt,
        )
        return out_tokens

    # -------------------------------
    # Sampling
    # -------------------------------
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, int, int],
        steps: int = 36,
        guidance_strength: Union[float, Tuple[float, float]] = 2.0,
        guidance_power: float = 2.0,
        eta: float = 0.0,
        *,
        cond_summary: Optional[torch.Tensor] = None,
        cond_summary_raw: Optional[torch.Tensor] = None,
        y_obs: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        cfg_rescale: bool = True,
        self_cond: bool = False,
        rho: float = 7.5,
        generator: Optional[torch.Generator] = None,
        dynamic_thresh_p: float = 0.0,
        dynamic_thresh_max: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample trajectories with DDIM and optional CFG.

        Args:
            shape: Tuple ``(B, L, D)`` describing the desired sample batch.
            steps: Number of denoising steps to take (``<= timesteps``).
            guidance_strength: CFG strength (scalar or ``(min, max)`` schedule).
            guidance_power: Exponent for scheduled CFG strength.
            eta: DDIM stochasticity parameter.
            cond_summary: Optional context tokens used for the conditional pass.
            y_obs / obs_mask: Optional observations for inpainting.
            dt: Optional step-size metadata forwarded to the Laplace backbone.
            cfg_rescale: Whether to apply the CFG rescaling heuristic.
            self_cond: Enable self-conditioning during sampling.
            rho: Karras sigma schedule exponent.
            generator: Optional RNG for reproducible initial/DDIM noise.
            dynamic_thresh_p / dynamic_thresh_max: Parameters for dynamic thresholding of ``x0``.

        Returns:
            The final ``x0`` prediction corresponding to the denoised samples.
        """
        if not (0.0 <= float(dynamic_thresh_p) <= 1.0):
            raise ValueError(f"dynamic_thresh_p must be in [0,1], got {dynamic_thresh_p}")
        device = next(self.parameters()).device
        if len(shape) != 3:
            raise ValueError(f"shape must be a tuple of positive ints (B, L, D), got {shape}")
        B, L, D = shape
        T = int(self.scheduler.timesteps)
        if min(int(B), int(L), int(D)) <= 0:
            raise ValueError(f"shape must be a tuple of positive ints (B, L, D), got {shape}")
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")
        if cond_summary is not None and cond_summary.shape[0] != B:
            raise ValueError(
                f"cond_summary batch mismatch: expected B={B}, got {cond_summary.shape[0]}"
            )
        if cond_summary_raw is not None and cond_summary_raw.shape[0] != B:
            raise ValueError(
                f"cond_summary_raw batch mismatch: expected B={B}, got {cond_summary_raw.shape[0]}"
            )
        if dt is not None:
            dt = torch.as_tensor(dt, device=device)
            if not torch.is_floating_point(dt):
                dt = dt.to(dtype=torch.float32)
            if dt.dim() == 2:
                dt_check = dt
            elif dt.dim() == 3 and dt.size(-1) == 1:
                dt_check = dt.squeeze(-1)
            else:
                raise ValueError(f"dt must have shape [B, L] or [B, L, 1], got {tuple(dt.shape)}")
            if tuple(dt_check.shape) != (int(B), int(L)):
                raise ValueError(
                    f"dt shape must match generation shape (B={int(B)}, L={int(L)}), got {tuple(dt_check.shape)}"
                )
            if not torch.isfinite(dt_check).all():
                raise ValueError("dt must contain only finite values")
            if dt_check.size(1) > 1:
                if not (dt_check[:, 1:] - dt_check[:, :-1] >= -1e-6).all():
                    raise ValueError("dt must be nondecreasing along the time dimension")

        n = int(max(1, min(int(steps), T)))
        if n >= T:
            step_indices = torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)
        else:
            t_min = 1
            alpha_bars = self.scheduler.alpha_bars.to(device=device, dtype=torch.float32).clamp_min(1e-12)
            sigmas = torch.sqrt((1.0 - alpha_bars) / alpha_bars)
            smin, smax = sigmas[t_min].item(), sigmas[-1].item()
            i = torch.linspace(0, 1, n, device=device)
            target = (
                smax ** (1 / float(rho))
                + i * (smin ** (1 / float(rho)) - smax ** (1 / float(rho)))
            ) ** float(rho)
            idx = torch.searchsorted(sigmas, target).clamp(min=t_min, max=T - 1)
            idxm = (idx - 1).clamp(min=t_min)
            pick_lower = torch.abs(sigmas[idxm] - target) <= torch.abs(sigmas[idx] - target)
            idx = torch.where(pick_lower, idxm, idx).long()
            step_indices = torch.flip(torch.unique(idx, sorted=True), dims=[0]).long()

        ts_prev = torch.cat([step_indices[1:], step_indices.new_tensor([-1])])

        def _randn_like(ref: torch.Tensor) -> torch.Tensor:
            if generator is None:
                return torch.randn_like(ref)
            return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=generator)

        def _alpha_bar_batched(t_b: torch.Tensor) -> torch.Tensor:
            return self.scheduler._gather(self.scheduler.alpha_bars, t_b).view(B, 1, 1)

        def _cfg(
            pred_u: torch.Tensor,
            pred_c: torch.Tensor,
            g_scalar: torch.Tensor,
            mask_scale: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if mask_scale is not None:
                g_eff = (1.0 + (g_scalar - 1.0) * mask_scale).to(pred_u.dtype)
            else:
                g_eff = g_scalar.to(pred_u.dtype)

            guided = pred_u + g_eff * (pred_c - pred_u)
            if not cfg_rescale:
                return guided

            reduce_dims = (1, 2)
            mu_c = pred_c.mean(dim=reduce_dims, keepdim=True)
            std_c = pred_c.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            mu_g = guided.mean(dim=reduce_dims, keepdim=True)
            std_g = guided.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            return (guided - mu_g) / std_g * std_c + mu_c

        def _dynamic_threshold(x0: torch.Tensor, p: float, max_val: float) -> torch.Tensor:
            if p <= 0.0:
                return x0
            x = x0.float()
            s = torch.quantile(x.reshape(B, -1).abs(), q=p, dim=1).clamp_min(1.0).view(B, 1, 1)
            x = (x / s).clamp(-max_val, max_val)
            return x.to(dtype=x0.dtype)

        # ---- init state (+ optional inpainting obs at first sampled time) ----
        if generator is None:
            x_t = torch.randn(B, L, D, device=device)
        else:
            x_t = torch.randn(B, L, D, device=device, generator=generator)
        sc_feat_next = torch.zeros_like(x_t) if self_cond else None
        obs_u = None
        tar_scale = None

        if obs_mask is not None:
            obs = obs_mask.to(device=device, dtype=x_t.dtype)
            obs_u = obs.unsqueeze(-1)
            tar_scale = (1.0 - obs).unsqueeze(-1)

        if (y_obs is not None) and (obs_u is not None):
            t0_b = step_indices[0].expand(B)
            y_obs_typed = y_obs.to(device=device, dtype=x_t.dtype)
            x_T_obs, _ = self.scheduler.q_sample(y_obs_typed, t0_b, noise=_randn_like(y_obs_typed))
            x_t = obs_u * x_T_obs + (1.0 - obs_u) * x_t

        # ---- main loop ----
        last_x0 = None
        for t_i, t_prev_i in zip(step_indices, ts_prev):
            t_b = torch.full((B,), int(t_i.item()), device=device, dtype=torch.long)

            # classifier-free guidance (optionally scheduled)
            if isinstance(guidance_strength, (tuple, list)):
                g_min, g_max = guidance_strength
                ab_b = _alpha_bar_batched(t_b)  # [B,1,1]
                g_min_t = torch.as_tensor(g_min, device=device, dtype=ab_b.dtype)
                g_max_t = torch.as_tensor(g_max, device=device, dtype=ab_b.dtype)
                g_scalar = g_min_t + (g_max_t - g_min_t) * (ab_b ** guidance_power)
            else:
                g_scalar = (
                    torch.as_tensor(float(guidance_strength), device=device)
                    .view(1, 1, 1)
                    .expand(B, 1, 1)
                )

            # Avoid an unnecessary unconditional pass when CFG is inactive.
            cond_present = (cond_summary is not None) or (cond_summary_raw is not None)
            cfg_active = cond_present and torch.any(torch.abs(g_scalar - 1.0) > 1e-12).item()
            pred_c = self.forward(
                x_t,
                t_b,
                cond_summary=cond_summary,
                cond_summary_raw=cond_summary_raw,
                sc_feat=sc_feat_next if self_cond else None,
                dt=dt,
            )
            if cfg_active:
                pred_u = self.forward(
                    x_t,
                    t_b,
                    cond_summary=None,
                    sc_feat=sc_feat_next if self_cond else None,
                    dt=dt,
                )
                pred = _cfg(pred_u, pred_c, g_scalar, mask_scale=tar_scale)
            else:
                pred = pred_c

            # predict x0
            x0_hat = self.scheduler.to_x0(x_t, t_b, pred, param_type=self.predict_type)
            x0_hat = _dynamic_threshold(x0_hat, dynamic_thresh_p, dynamic_thresh_max)
            last_x0 = x0_hat

            if self_cond:
                sc_feat_next = x0_hat.detach()

            # time update
            if int(t_prev_i) >= 0:
                tprev_b = torch.full((B,), int(t_prev_i.item()), device=device, dtype=torch.long)
                x_t = self.scheduler.ddim_step_from(
                    x_t,
                    t_b,
                    tprev_b,
                    pred,
                    param_type=self.predict_type,
                    eta=eta,
                    noise=_randn_like(x_t),
                )
            else:
                x_t = x0_hat

            # keep observed values consistent across steps (inpainting)
            if (y_obs is not None) and (obs_u is not None):
                if int(t_prev_i) >= 0:
                    y_obs_typed = y_obs.to(device=device, dtype=x_t.dtype)
                    x_obs_t, _ = self.scheduler.q_sample(y_obs_typed, tprev_b, noise=_randn_like(y_obs_typed))
                    x_t = obs_u * x_obs_t + (1.0 - obs_u) * x_t
                else:
                    x_t = obs_u * y_obs.to(device=device, dtype=x_t.dtype) + (1.0 - obs_u) * x_t
                    last_x0 = x_t

        final = last_x0 if last_x0 is not None else x_t
        return final
