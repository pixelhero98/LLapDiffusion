from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from llapdiffusion.baselines.features import target_context, time_features
from llapdiffusion.baselines.sources import SourceManager


class TPatchGNNAdapter(nn.Module):
    def __init__(self, dataset_info: dict[str, object], sample_batch, source_manager: SourceManager, device: torch.device):
        super().__init__()
        with source_manager.prepend(
            source_manager.path("t-PatchGNN") / "tPatchGNN",
            source_manager.path("t-PatchGNN"),
            module_prefixes=("model", "lib"),
        ):
            module = importlib.import_module("model.tPatchGNN")
        N = sample_batch[0][0].shape[1]
        self.npatch = min(4, int(dataset_info["window"]))
        args = SimpleNamespace(
            device=str(device),
            hid_dim=8,
            ndim=N,
            nhead=1,
            tf_layer=1,
            nlayer=1,
            node_dim=4,
            hop=1,
            outlayer="Linear",
            te_dim=4,
            patch_size=1,
            stride=1,
            npatch=self.npatch,
        )
        self.model = module.tPatchGNN(args, supports=None, dropout=0.0).to(device)

    def _patches(self, batch, dataset_info):
        (V, _), _, meta = batch
        x, mask, _, _ = target_context(batch, dataset_info)
        t, _, ty = time_features(meta, V)
        B, N, K = x.shape
        L = math.ceil(K / self.npatch)
        total = L * self.npatch
        pad = total - K
        x_pad = F.pad(x, (0, pad))
        m_pad = F.pad(mask.to(dtype=x.dtype), (0, pad))
        t_pad = F.pad(t, (0, pad))
        X = x_pad.reshape(B, N, self.npatch, L).permute(0, 2, 3, 1)
        M = m_pad.reshape(B, N, self.npatch, L).permute(0, 2, 3, 1)
        TT = t_pad.reshape(B, N, self.npatch, L).permute(0, 2, 3, 1)
        pred_t = ty.mean(dim=1)
        return X, TT, M, pred_t

    def forward(self, batch, dataset_info):
        X, TT, M, pred_t = self._patches(batch, dataset_info)
        out = self.model.forecasting(pred_t, X, TT, M)
        return out.squeeze(0).permute(0, 2, 1)
