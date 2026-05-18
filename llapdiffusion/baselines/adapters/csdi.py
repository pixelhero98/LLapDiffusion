from __future__ import annotations

import importlib

import torch
import torch.nn as nn

from llapdiffusion.baselines.features import target_context, time_features
from llapdiffusion.baselines.sources import SourceManager


class CSDIAdapter(nn.Module):
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
        with source_manager.prepend(source_manager.path("CSDI"), module_prefixes=("diff_models", "main_model")):
            module = importlib.import_module("main_model")
        N = sample_batch[0][0].shape[1]
        config = {
            "model": {"timeemb": 16, "featureemb": 8, "is_unconditional": 0, "target_strategy": "random"},
            "diffusion": {
                "layers": 1,
                "channels": 8,
                "nheads": 1,
                "diffusion_embedding_dim": 16,
                "beta_start": 0.0001,
                "beta_end": 0.1,
                "num_steps": 4,
                "schedule": "linear",
                "is_linear": False,
            },
        }
        self.model = module.CSDI_Physio(config, device, target_dim=N)
        self.num_samples = int(num_samples)

    @staticmethod
    def _random_gt_mask(observed_mask: torch.Tensor) -> torch.Tensor:
        gt_mask = observed_mask.clone().contiguous()
        for b in range(gt_mask.shape[0]):
            observed = torch.nonzero(observed_mask[b].reshape(-1) > 0, as_tuple=False).flatten()
            if observed.numel() == 0:
                continue
            holdout = max(1, observed.numel() // 5)
            chosen = observed[torch.randperm(observed.numel(), device=observed.device)[:holdout]]
            flat = gt_mask[b].view(-1)
            flat[chosen] = 0.0
        return gt_mask

    def _batch(self, batch, dataset_info):
        (V, _), _, meta = batch
        x, mask, _, _ = target_context(batch, dataset_info)
        t, _, _ = time_features(meta, V)
        observed_mask = mask.to(dtype=x.dtype).permute(0, 2, 1)
        return {
            "observed_data": x.permute(0, 2, 1),
            "observed_mask": observed_mask,
            "timepoints": t.mean(dim=1),
            "gt_mask": self._random_gt_mask(observed_mask),
        }

    def loss(self, batch, dataset_info):
        return self.model(self._batch(batch, dataset_info), is_train=1)

    def loss_and_samples(self, batch, dataset_info):
        csdi_batch = self._batch(batch, dataset_info)
        loss = self.model(csdi_batch, is_train=1)
        samples, observed_data, target_mask, _, _ = self.model.evaluate(csdi_batch, n_samples=self.num_samples)
        return loss, samples.permute(1, 0, 2, 3), observed_data, target_mask
