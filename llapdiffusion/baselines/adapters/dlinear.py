from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from llapdiffusion.baselines.data import regular_feature_target_index
from llapdiffusion.baselines.features import regular_features
from llapdiffusion.baselines.sources import SourceManager


class DLinearAdapter(nn.Module):
    def __init__(self, dataset_info: dict[str, object], sample_batch, source_manager: SourceManager):
        super().__init__()
        module = source_manager.load_module(
            "llap_baseline_dlinear",
            source_manager.path("LTSF-Linear") / "models" / "DLinear.py",
        )
        feat = regular_features(sample_batch, dataset_info)
        cfg = SimpleNamespace(
            seq_len=dataset_info["window"],
            pred_len=dataset_info["horizon"],
            enc_in=feat.shape[-1],
            individual=False,
        )
        self.model = module.Model(cfg)
        self.target_idx = regular_feature_target_index(dataset_info)

    def forward(self, batch, dataset_info):
        feat = regular_features(batch, dataset_info)
        B, N, K, C = feat.shape
        out = self.model(feat.reshape(B * N, K, C)).reshape(B, N, dataset_info["horizon"], C)
        return out[..., min(self.target_idx, C - 1)]
