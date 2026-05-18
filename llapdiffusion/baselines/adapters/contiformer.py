from __future__ import annotations

import torch
import torch.nn as nn

from llapdiffusion.baselines.features import progressing_context_time, target_context
from llapdiffusion.baselines.sources import SourceManager


class ContiFormerAdapter(nn.Module):
    def __init__(self, dataset_info: dict[str, object], sample_batch, source_manager: SourceManager):
        super().__init__()
        with source_manager.prepend(source_manager.path("physiopro"), module_prefixes=("physiopro",)):
            from physiopro.network.contiformer import ContiFormer

        self.model = ContiFormer(
            input_size=2,
            d_model=16,
            d_inner=32,
            n_layers=1,
            n_head=2,
            d_k=8,
            d_v=8,
            dropout=0.0,
            actfn_ode="tanh",
            layer_type_ode="concat",
            zero_init_ode=True,
            atol_ode=0.1,
            rtol_ode=0.1,
            method_ode="rk4",
            linear_type_ode="before",
            approximate_method="bilinear",
            nlinspace=1,
            interpolate_ode="linear",
            itol_ode=1e-2,
            add_pe=False,
            normalize_before=False,
            max_length=int(dataset_info["window"]),
        )
        self.head = nn.Linear(16, int(dataset_info["horizon"]))

    def forward(self, batch, dataset_info):
        (V, _), _, meta = batch
        x, mask, _, _ = target_context(batch, dataset_info)
        t = progressing_context_time(meta, V)
        B, N, K = x.shape
        seq = torch.stack([x, mask.to(dtype=x.dtype)], dim=-1).reshape(B * N, K, 2)
        seq_mask = mask.reshape(B * N, K)
        h, _ = self.model(seq, t.reshape(B * N, K), mask=(~seq_mask).unsqueeze(-1))
        pooled = (h * seq_mask.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=1) / seq_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(pooled).reshape(B, N, int(dataset_info["horizon"]))
