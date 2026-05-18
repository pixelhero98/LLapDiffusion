from __future__ import annotations

import math

import torch
import torch.nn as nn

from llapdiffusion.baselines.features import target_context, time_features
from llapdiffusion.baselines.sources import SourceManager


class MTANAdapter(nn.Module):
    def __init__(
        self,
        dataset_info: dict[str, object],
        sample_batch,
        source_manager: SourceManager,
        device: torch.device,
        *,
        num_samples: int = 4,
    ):
        super().__init__()
        mtan_models = source_manager.load_module("llap_baseline_mtan_models", source_manager.path("mTAN") / "src" / "models.py")
        K, H = int(dataset_info["window"]), int(dataset_info["horizon"])
        query = torch.linspace(0.0, 1.0, K)
        self.enc = mtan_models.enc_mtan_rnn(1, query=query, latent_dim=4, nhidden=8, embed_time=8, num_heads=1, learn_emb=False, device=str(device))
        self.dec = mtan_models.dec_mtan_rnn(1, query=query, latent_dim=4, nhidden=8, embed_time=8, num_heads=1, learn_emb=False, device=str(device))
        self.log_std = nn.Parameter(torch.tensor(-0.5))
        self.H = H
        self.num_samples = int(num_samples)

    def forward_dist(self, batch, dataset_info):
        (V, _), _, meta = batch
        x, mask, _, _ = target_context(batch, dataset_info)
        t, _, ty = time_features(meta, V)
        B, N, K = x.shape
        flat_x = x.reshape(B * N, K, 1)
        flat_m = mask.to(dtype=x.dtype).reshape(B * N, K, 1)
        flat_t = t.reshape(B * N, K)
        qz = self.enc(torch.cat([flat_x, flat_m], dim=-1), flat_t)
        mean, logvar = qz[..., :4], qz[..., 4:]
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        future_t = ty.mean(dim=1).repeat_interleave(N, dim=0)
        pred = self.dec(z, future_t).reshape(B, N, self.H)
        std = torch.exp(self.log_std).clamp_min(1e-4)
        samples = pred.unsqueeze(0) + std * torch.randn(self.num_samples, *pred.shape, device=pred.device)
        return pred, std, samples

    def loss(self, batch, dataset_info):
        _, _, y, valid = target_context(batch, dataset_info)
        pred, std, _ = self.forward_dist(batch, dataset_info)
        nll = 0.5 * (((pred - y) / std) ** 2 + 2 * torch.log(std) + math.log(2 * math.pi))
        return (nll * valid.to(dtype=nll.dtype)).sum() / valid.sum().clamp_min(1)
