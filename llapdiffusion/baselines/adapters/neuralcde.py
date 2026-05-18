from __future__ import annotations

import torch
import torch.nn as nn

from llapdiffusion.baselines.features import target_context, time_features
from llapdiffusion.baselines.sources import SourceManager


class NeuralCDEAdapter(nn.Module):
    def __init__(self, dataset_info: dict[str, object], sample_batch, source_manager: SourceManager):
        super().__init__()
        with source_manager.prepend(source_manager.path("NeuralCDE"), module_prefixes=("controldiffeq", "experiments")):
            import controldiffeq
            from experiments.models import metamodel, vector_fields

        self.controldiffeq = controldiffeq
        input_channels = 4
        func = vector_fields.FinalTanh(
            input_channels,
            hidden_channels=8,
            hidden_hidden_channels=16,
            num_hidden_layers=1,
        )
        self.model = metamodel.NeuralCDE(
            func,
            input_channels,
            hidden_channels=8,
            output_channels=dataset_info["horizon"],
            initial=True,
        )
        self.register_buffer("times", torch.linspace(0.0, 1.0, int(dataset_info["window"])), persistent=False)

    def forward(self, batch, dataset_info):
        (V, _), _, meta = batch
        x, mask, _, _ = target_context(batch, dataset_info)
        t, gap, _ = time_features(meta, V)
        x_path = x.masked_fill(~mask, float("nan"))
        feat = torch.stack([t, x_path, mask.to(dtype=V.dtype), gap], dim=-1)
        times = self.times.to(device=V.device, dtype=V.dtype)
        coeffs = self.controldiffeq.natural_cubic_spline_coeffs(times, feat)
        final_index = torch.full(feat.shape[:-2], int(dataset_info["window"]) - 1, dtype=torch.long, device=V.device)
        return self.model(times, coeffs, final_index)
