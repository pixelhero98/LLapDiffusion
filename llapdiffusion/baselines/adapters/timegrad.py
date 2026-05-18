from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn

from llapdiffusion.baselines.features import target_context, time_features
from llapdiffusion.baselines.sources import SourceManager


class TimeGradAdapter(nn.Module):
    def __init__(
        self,
        dataset_info: dict[str, object],
        sample_batch,
        source_manager: SourceManager,
        *,
        num_samples: int = 4,
    ):
        super().__init__()
        with source_manager.prepend(source_manager.path("timegrad"), module_prefixes=("utils", "module", "epsilon_theta", "time_grad_network")):
            eps_module = importlib.import_module("epsilon_theta")
            diff_module = importlib.import_module("module")
        self.H = int(dataset_info["horizon"])
        self.num_samples = int(num_samples)
        self.context_encoder = nn.GRU(input_size=7, hidden_size=8, num_layers=1, batch_first=True)
        self.future_encoder = nn.GRU(input_size=3, hidden_size=8, num_layers=1, batch_first=True)
        denoise = eps_module.EpsilonTheta(
            target_dim=self.H,
            cond_length=8,
            residual_layers=1,
            residual_channels=8,
            dilation_cycle_length=1,
        )
        self.diffusion = diff_module.GaussianDiffusion(
            denoise,
            input_size=self.H,
            diff_steps=4,
            loss_type="l2",
            beta_end=0.1,
            beta_schedule="linear",
        )

    def _inputs(self, batch, dataset_info):
        (V, _), _, meta = batch
        x, mask, y_clean, valid = target_context(batch, dataset_info)
        B, N, K = x.shape
        t, gap, ty = time_features(meta, V)
        H = y_clean.shape[-1]
        past_time = torch.stack([t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t), gap, mask.to(dtype=V.dtype)], dim=-1)
        future_time = torch.stack([ty, torch.sin(2 * math.pi * ty), torch.cos(2 * math.pi * ty)], dim=-1)
        context = torch.cat(
            [
                x.reshape(B * N, K, 1),
                mask.to(dtype=V.dtype).reshape(B * N, K, 1),
                past_time.reshape(B * N, K, 5),
            ],
            dim=-1,
        )
        future_context = future_time.reshape(B * N, H, 3)
        target = y_clean.reshape(B * N, 1, H)
        target_valid = valid.reshape(B * N, 1, H).to(dtype=target.dtype)
        return context, future_context, target, target_valid, (B, N, H)

    def _cond(self, context: torch.Tensor, future_context: torch.Tensor) -> torch.Tensor:
        _, hidden = self.context_encoder(context)
        _, future_hidden = self.future_encoder(future_context)
        return (hidden[-1] + future_hidden[-1]).unsqueeze(1)

    def loss(self, batch, dataset_info):
        context, future_context, target, target_valid, (B, N, _) = self._inputs(batch, dataset_info)
        cond = self._cond(context, future_context)
        t = torch.randint(0, self.diffusion.num_timesteps, (B * N,), device=target.device)
        noise = torch.randn_like(target)
        x_noisy = self.diffusion.q_sample(x_start=target, t=t, noise=noise)
        x_recon = self.diffusion.denoise_fn(x_noisy, t, cond=cond)
        return (((x_recon - noise) ** 2) * target_valid).sum() / target_valid.sum().clamp_min(1.0)

    def loss_and_samples(self, batch, dataset_info):
        context, future_context, target, target_valid, (B, N, H) = self._inputs(batch, dataset_info)
        cond = self._cond(context, future_context)
        t = torch.randint(0, self.diffusion.num_timesteps, (B * N,), device=target.device)
        noise = torch.randn_like(target)
        x_noisy = self.diffusion.q_sample(x_start=target, t=t, noise=noise)
        x_recon = self.diffusion.denoise_fn(x_noisy, t, cond=cond)
        loss = (((x_recon - noise) ** 2) * target_valid).sum() / target_valid.sum().clamp_min(1.0)
        samples = [self.diffusion.sample(cond=cond).reshape(B, N, H) for _ in range(self.num_samples)]
        return loss, torch.stack(samples, dim=0)
