from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch
import torch.nn as nn

from llapdiffusion.baselines.data import regular_feature_target_indices
from llapdiffusion.baselines.features import regular_features
from llapdiffusion.baselines.sources import SourceManager


class PatchTSTAdapter(nn.Module):
    def __init__(self, dataset_info: dict[str, object], sample_batch, source_manager: SourceManager):
        super().__init__()
        source = source_manager.path("PatchTST") / "PatchTST_supervised"
        with source_manager.prepend(source, module_prefixes=("layers", "models")):
            module = importlib.import_module("models.PatchTST")
        feat = regular_features(sample_batch, dataset_info)
        patch_len = min(16, max(4, int(dataset_info["window"]) // 4))
        stride = max(1, patch_len // 2)
        cfg = SimpleNamespace(
            enc_in=feat.shape[-1],
            seq_len=dataset_info["window"],
            pred_len=dataset_info["horizon"],
            e_layers=1,
            n_heads=2,
            d_model=16,
            d_ff=32,
            dropout=0.0,
            fc_dropout=0.0,
            head_dropout=0.0,
            individual=False,
            patch_len=patch_len,
            stride=stride,
            padding_patch="end",
            revin=False,
            affine=False,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )
        self.model = module.Model(cfg)
        self.target_indices = regular_feature_target_indices(dataset_info)

    def forward(self, batch, dataset_info):
        feat = regular_features(batch, dataset_info)
        B, N, K, C = feat.shape
        out = self.model(feat.reshape(B * N, K, C)).reshape(B, N, dataset_info["horizon"], C)
        indices = torch.as_tensor([min(idx, C - 1) for idx in self.target_indices], device=out.device)
        selected = out.index_select(-1, indices)
        return selected.squeeze(-1) if len(self.target_indices) == 1 else selected
