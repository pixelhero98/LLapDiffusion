from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llapdiffusion.baselines.features import target_context, time_features


MRDIFF_STAGES = 5
MRDIFF_KERNELS = (5, 25, 51, 201)
MRDIFF_WIDTH = 256
MRDIFF_STEP_EMBED = 128
MRDIFF_DIFFUSION_STEPS = 100
MRDIFF_BETA_START = 1e-4
MRDIFF_BETA_END = 1e-1
MRDIFF_SAMPLE_CLIP = 10.0


def _timestep_embedding(step: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=step.device, dtype=torch.float32) / max(half, 1)
    )
    args = step.to(dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _conv_stack(channels: int, blocks: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(blocks):
        layers.extend(
            [
                nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(channels),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(0.1),
            ]
        )
    return nn.Sequential(*layers)


def _same_length_pool(value: torch.Tensor, mask: torch.Tensor | None, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    if kernel_size <= 1:
        return value, mask
    left = kernel_size // 2
    right = kernel_size - 1 - left
    if mask is None:
        padded = F.pad(value, (left, right), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1), None

    weight = mask.to(dtype=value.dtype)
    padded_value = F.pad(value * weight, (left, right), mode="replicate")
    padded_weight = F.pad(weight, (left, right), mode="replicate")
    value_sum = F.avg_pool1d(padded_value, kernel_size=kernel_size, stride=1)
    weight_sum = F.avg_pool1d(padded_weight, kernel_size=kernel_size, stride=1)
    pooled = value_sum / weight_sum.clamp_min(1e-6)
    pooled_mask = weight_sum > 0
    return torch.where(pooled_mask, pooled, torch.zeros_like(pooled)), pooled_mask


def _trend_stack(value: torch.Tensor, mask: torch.Tensor | None, kernels: tuple[int, ...]) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    values = [value]
    masks = [mask]
    current_value = value
    current_mask = mask
    for kernel in kernels:
        current_value, current_mask = _same_length_pool(current_value, current_mask, kernel)
        values.append(current_value)
        masks.append(current_mask)
    return values, masks


class _ObservedRevIN:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def from_context(cls, x: torch.Tensor, mask: torch.Tensor) -> _ObservedRevIN:
        weight = mask.to(dtype=x.dtype)
        count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (x * weight).sum(dim=-1, keepdim=True) / count
        centered = (x - mean) * weight
        var = (centered.square().sum(dim=-1, keepdim=True) / count).clamp_min(0.0)
        std = torch.sqrt(var + 1e-5)
        return cls(mean, std)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean


class _HistoryEncoder(nn.Module):
    def __init__(self, num_entities: int, width: int, horizon: int):
        super().__init__()
        self.horizon = int(horizon)
        self.project = nn.Conv1d(num_entities * 6, width, kernel_size=1)
        self.blocks = _conv_stack(width)
        self.out = nn.Conv1d(width, num_entities, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bsz, entities, feat, length = features.shape
        encoded = self.project(features.reshape(bsz, entities * feat, length))
        encoded = self.blocks(encoded)
        out = self.out(encoded)
        if out.shape[-1] != self.horizon:
            out = F.interpolate(out, size=self.horizon, mode="linear", align_corners=False)
        return out


class _StageDenoiser(nn.Module):
    def __init__(
        self,
        num_entities: int,
        width: int,
        step_embed_dim: int,
        *,
        has_coarse_condition: bool,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(num_entities, width, kernel_size=1)
        cond_groups = 5 if has_coarse_condition else 4
        self.condition_projection = nn.Conv1d(num_entities * cond_groups, width, kernel_size=1)
        self.condition_blocks = _conv_stack(width)
        self.step_projection = nn.Sequential(
            nn.Linear(step_embed_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.encoder = _conv_stack(width)
        self.decoder_in = nn.Conv1d(width * 2, width, kernel_size=1)
        self.decoder = _conv_stack(width)
        self.out = nn.Conv1d(width, num_entities, kernel_size=1)

    def condition(self, z_mix: torch.Tensor, coarse: torch.Tensor | None, future_time: torch.Tensor) -> torch.Tensor:
        pieces = [z_mix]
        if coarse is not None:
            pieces.append(coarse)
        pieces.extend(
            [
                future_time,
                torch.sin(2 * math.pi * future_time),
                torch.cos(2 * math.pi * future_time),
            ]
        )
        cond = torch.cat(pieces, dim=1)
        return self.condition_blocks(self.condition_projection(cond))

    def forward(self, noisy: torch.Tensor, step: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(noisy)
        emb = _timestep_embedding(step, MRDIFF_STEP_EMBED).to(dtype=hidden.dtype)
        hidden = hidden + self.step_projection(emb).unsqueeze(-1)
        hidden = self.encoder(hidden)
        hidden = self.decoder_in(torch.cat([hidden, condition], dim=1))
        return self.out(self.decoder(hidden))


class _MRDiffStage(nn.Module):
    def __init__(
        self,
        num_entities: int,
        horizon: int,
        width: int,
        step_embed_dim: int,
        *,
        has_coarse_condition: bool,
    ):
        super().__init__()
        self.history = _HistoryEncoder(num_entities, width, horizon)
        self.denoiser = _StageDenoiser(
            num_entities,
            width,
            step_embed_dim,
            has_coarse_condition=has_coarse_condition,
        )


class MRDiffAdapter(nn.Module):
    def __init__(
        self,
        dataset_info: dict[str, object],
        sample_batch,
        *,
        num_samples: int = 4,
        stages: int = MRDIFF_STAGES,
        kernels: tuple[int, ...] = MRDIFF_KERNELS,
        width: int = MRDIFF_WIDTH,
        diffusion_steps: int = MRDIFF_DIFFUSION_STEPS,
    ):
        super().__init__()
        if stages != len(kernels) + 1:
            raise ValueError("MR-Diff stages must equal len(kernels) + 1")
        (V, _), _, _ = sample_batch
        self.num_entities = int(V.shape[1])
        self.horizon = int(dataset_info["horizon"])
        self.num_samples = int(num_samples)
        self.kernels = tuple(int(k) for k in kernels)
        self.diffusion_steps = int(diffusion_steps)
        self.stages = nn.ModuleList(
            [
                _MRDiffStage(
                    self.num_entities,
                    self.horizon,
                    int(width),
                    MRDIFF_STEP_EMBED,
                    has_coarse_condition=s < stages - 1,
                )
                for s in range(stages)
            ]
        )
        betas = torch.linspace(MRDIFF_BETA_START, MRDIFF_BETA_END, self.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1e-12)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def _validate_entities(self, x: torch.Tensor) -> None:
        if x.shape[1] != self.num_entities:
            raise ValueError(
                f"MR-Diff was initialized for {self.num_entities} entities, "
                f"but the batch has {x.shape[1]} entities"
            )

    def _stage_features(
        self,
        x_trends: list[torch.Tensor],
        mask_trends: list[torch.Tensor | None],
        t_trends: list[torch.Tensor],
        gap_trends: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        features = []
        for x, mask, t, gap in zip(x_trends, mask_trends, t_trends, gap_trends, strict=True):
            if mask is None:
                mask_float = torch.ones_like(x)
            else:
                mask_float = mask.to(dtype=x.dtype)
            features.append(
                torch.stack(
                    [
                        x,
                        mask_float,
                        t,
                        torch.sin(2 * math.pi * t),
                        torch.cos(2 * math.pi * t),
                        gap,
                    ],
                    dim=2,
                )
            )
        return features

    def _inputs(self, batch, dataset_info):
        (V, _), y, meta = batch
        x, mask, y_clean, valid = target_context(batch, dataset_info)
        self._validate_entities(x)
        revin = _ObservedRevIN.from_context(x, mask)
        x_norm = torch.where(mask, revin.normalize(x), torch.zeros_like(x))
        y_norm = revin.normalize(y_clean)
        y_norm = torch.where(valid, y_norm, torch.zeros_like(y_norm))

        t, gap, ty = time_features(meta, V)
        x_trends, mask_trends = _trend_stack(x_norm, mask, self.kernels)
        y_trends, y_masks = _trend_stack(y_norm, valid, self.kernels)
        t_trends, _ = _trend_stack(t, None, self.kernels)
        gap_trends, _ = _trend_stack(gap, None, self.kernels)
        features = self._stage_features(x_trends, mask_trends, t_trends, gap_trends)
        return features, y_trends, [m.to(dtype=torch.bool) for m in y_masks], ty, revin

    def _extract(self, schedule: torch.Tensor, step: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return schedule.gather(0, step).to(dtype=target.dtype).reshape(target.shape[0], 1, 1)

    def _q_sample(self, clean: torch.Tensor, step: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, step, clean)
        return torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise

    def _posterior_step(self, noisy: torch.Tensor, clean_pred: torch.Tensor, step: int) -> torch.Tensor:
        t = torch.full((noisy.shape[0],), int(step), dtype=torch.long, device=noisy.device)
        beta = self._extract(self.betas, t, noisy)
        alpha = self._extract(self.alphas, t, noisy)
        alpha_bar = self._extract(self.alpha_bars, t, noisy)
        alpha_bar_prev = self._extract(self.alpha_bars_prev, t, noisy)
        coef_clean = beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar).clamp_min(1e-12)
        coef_noisy = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp_min(1e-12)
        mean = coef_clean * clean_pred + coef_noisy * noisy
        if step == 0:
            return mean
        var = self._extract(self.posterior_variance, t, noisy)
        return mean + torch.sqrt(var) * torch.randn_like(noisy)

    def _training_loss(self, features, y_trends, y_masks, future_time) -> torch.Tensor:
        losses = []
        history_trends = [stage.history(features[idx]) for idx, stage in enumerate(self.stages)]
        for stage_idx in range(len(self.stages) - 1, -1, -1):
            stage = self.stages[stage_idx]
            target = y_trends[stage_idx]
            target_mask = y_masks[stage_idx]
            z_history = history_trends[stage_idx]
            coarse = history_trends[stage_idx + 1].detach() if stage_idx < len(self.stages) - 1 else None
            condition = stage.denoiser.condition(z_history, coarse, future_time)
            step = torch.randint(0, self.diffusion_steps, (target.shape[0],), device=target.device)
            noise = torch.randn_like(target)
            noisy = self._q_sample(target, step, noise)
            pred = stage.denoiser(noisy, step, condition)
            weight = target_mask.to(dtype=pred.dtype)
            losses.append(((pred - target).square() * weight).sum() / weight.sum().clamp_min(1.0))
        return torch.stack(losses).mean()

    def loss(self, batch, dataset_info):
        features, y_trends, y_masks, future_time, _ = self._inputs(batch, dataset_info)
        return self._training_loss(features, y_trends, y_masks, future_time)

    def _sample_normalized(self, features, future_time) -> torch.Tensor:
        generated: dict[int, torch.Tensor] = {}
        for stage_idx in range(len(self.stages) - 1, -1, -1):
            stage = self.stages[stage_idx]
            z_history = stage.history(features[stage_idx])
            coarse = generated.get(stage_idx + 1)
            condition = stage.denoiser.condition(z_history, coarse, future_time)
            current = torch.randn(
                z_history.shape[0],
                self.num_entities,
                self.horizon,
                device=z_history.device,
                dtype=z_history.dtype,
            )
            for step in range(self.diffusion_steps - 1, -1, -1):
                step_tensor = torch.full((current.shape[0],), step, dtype=torch.long, device=current.device)
                clean_pred = stage.denoiser(current, step_tensor, condition).clamp(-MRDIFF_SAMPLE_CLIP, MRDIFF_SAMPLE_CLIP)
                current = self._posterior_step(current, clean_pred, step)
            generated[stage_idx] = current
        return generated[0]

    def loss_and_samples(self, batch, dataset_info):
        features, y_trends, y_masks, future_time, revin = self._inputs(batch, dataset_info)
        loss = self._training_loss(features, y_trends, y_masks, future_time)
        with torch.no_grad():
            samples = [revin.denormalize(self._sample_normalized(features, future_time)) for _ in range(self.num_samples)]
        return loss, torch.stack(samples, dim=0)

    def forward(self, batch, dataset_info):
        features, _, _, future_time, revin = self._inputs(batch, dataset_info)
        return revin.denormalize(self._sample_normalized(features, future_time))
