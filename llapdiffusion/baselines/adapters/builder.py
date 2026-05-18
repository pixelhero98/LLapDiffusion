from __future__ import annotations

import torch
import torch.nn as nn

from llapdiffusion.baselines.adapters.contiformer import ContiFormerAdapter
from llapdiffusion.baselines.adapters.csdi import CSDIAdapter
from llapdiffusion.baselines.adapters.dlinear import DLinearAdapter
from llapdiffusion.baselines.adapters.mtan import MTANAdapter
from llapdiffusion.baselines.adapters.mr_diff import MRDiffAdapter
from llapdiffusion.baselines.adapters.neuralcde import NeuralCDEAdapter
from llapdiffusion.baselines.adapters.patchtst import PatchTSTAdapter
from llapdiffusion.baselines.adapters.t_patchgnn import TPatchGNNAdapter
from llapdiffusion.baselines.adapters.timegrad import TimeGradAdapter
from llapdiffusion.baselines.sources import SourceManager


def build_adapter(
    key: str,
    dataset_info: dict[str, object],
    sample_batch,
    source_manager: SourceManager,
    device: torch.device,
    *,
    num_samples: int = 4,
    imputation_random_mask_ratio: float = 0.30,
) -> nn.Module:
    if key == "dlinear":
        return DLinearAdapter(dataset_info, sample_batch, source_manager)
    if key == "patchtst":
        return PatchTSTAdapter(dataset_info, sample_batch, source_manager)
    if key == "neuralcde":
        return NeuralCDEAdapter(dataset_info, sample_batch, source_manager)
    if key == "mtan":
        return MTANAdapter(dataset_info, sample_batch, source_manager, device, num_samples=num_samples)
    if key == "timegrad":
        return TimeGradAdapter(dataset_info, sample_batch, source_manager, num_samples=num_samples)
    if key == "mr-diff":
        return MRDiffAdapter(dataset_info, sample_batch, num_samples=num_samples)
    if key == "t_patchgnn":
        return TPatchGNNAdapter(dataset_info, sample_batch, source_manager, device)
    if key == "contiformer":
        return ContiFormerAdapter(dataset_info, sample_batch, source_manager)
    if key == "csdi":
        return CSDIAdapter(
            dataset_info,
            sample_batch,
            source_manager,
            device,
            num_samples=num_samples,
            imputation_random_mask_ratio=imputation_random_mask_ratio,
        )
    raise KeyError(key)
