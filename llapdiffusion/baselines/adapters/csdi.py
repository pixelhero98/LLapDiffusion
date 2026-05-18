from __future__ import annotations

import importlib

import torch
import torch.nn as nn

from llapdiffusion.baselines.features import target_context
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
        imputation_random_mask_ratio: float = 0.30,
    ):
        super().__init__()
        if not 0.0 < float(imputation_random_mask_ratio) < 1.0:
            raise ValueError("imputation_random_mask_ratio must be in the open interval (0, 1)")
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
        self.imputation_random_mask_ratio = float(imputation_random_mask_ratio)
        self.metric_target_type = "target_horizon_imputation"

    def _target_horizon_gt_mask(self, observed_mask: torch.Tensor, context_length: int) -> torch.Tensor:
        gt_mask = observed_mask.clone().contiguous()
        for b in range(gt_mask.shape[0]):
            future = observed_mask[b, context_length:, :]
            observed = torch.nonzero(future.reshape(-1) > 0, as_tuple=False).flatten()
            if observed.numel() == 0:
                continue
            holdout = int(round(observed.numel() * self.imputation_random_mask_ratio))
            holdout = min(max(1, holdout), observed.numel())
            chosen = observed[torch.randperm(observed.numel(), device=observed.device)[:holdout]]
            future_flat = gt_mask[b, context_length:, :].view(-1)
            future_flat[chosen] = 0.0
        return gt_mask

    def _target_timepoints(self, meta, V: torch.Tensor, context_length: int) -> torch.Tensor:
        def _masked_mean(values: torch.Tensor) -> torch.Tensor:
            entity_mask = meta.get("entity_mask")
            if entity_mask is None:
                return values.mean(dim=1)
            mask = entity_mask.to(device=values.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(values.shape[:2]):
                raise ValueError(
                    f"entity_mask shape {tuple(mask.shape)} does not match time metadata shape {tuple(values.shape[:2])}"
                )
            weights = mask.to(dtype=values.dtype).unsqueeze(-1)
            return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        dt = _masked_mean(meta["delta_t"].to(device=V.device, dtype=V.dtype))
        dt_y = _masked_mean(meta["delta_t_y"].to(device=V.device, dtype=V.dtype))
        future = dt[:, -1:].clamp_min(0.0) + dt_y.clamp_min(0.0)
        combined = torch.cat([dt, future], dim=1)
        denom = combined.amax(dim=1, keepdim=True).clamp_min(1.0)
        return (combined / denom).clamp(0.0, 1.0)

    def _batch(self, batch, dataset_info):
        (V, _), _, meta = batch
        x, mask, _, _ = target_context(batch, dataset_info)
        _, _, y_clean, valid = target_context(batch, dataset_info)
        observed_data = torch.cat([x, y_clean], dim=-1).permute(0, 2, 1)
        observed_mask = torch.cat([mask, valid], dim=-1).to(dtype=x.dtype).permute(0, 2, 1)
        context_length = x.shape[-1]
        return {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "timepoints": self._target_timepoints(meta, V, context_length),
            "gt_mask": self._target_horizon_gt_mask(observed_mask, context_length),
        }

    def _loss_with_gt_mask(self, csdi_batch):
        observed_data, observed_mask, observed_tp, gt_mask = self.model.process_data(csdi_batch)[:4]
        side_info = self.model.get_side_info(observed_tp, gt_mask)
        return self.model.calc_loss(observed_data, gt_mask, observed_mask, side_info, is_train=1)

    def loss(self, batch, dataset_info):
        return self._loss_with_gt_mask(self._batch(batch, dataset_info))

    def loss_and_samples(self, batch, dataset_info):
        csdi_batch = self._batch(batch, dataset_info)
        loss = self._loss_with_gt_mask(csdi_batch)
        samples, observed_data, target_mask, _, _ = self.model.evaluate(csdi_batch, n_samples=self.num_samples)
        return loss, samples.permute(1, 0, 2, 3), observed_data, target_mask
