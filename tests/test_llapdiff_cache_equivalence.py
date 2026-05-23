from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llapdiffusion.diffusion_cache import build_or_load_diffusion_input_cache, cache_allowed
from llapdiffusion.models.llapdiff_utils import (
    build_context,
    normalize_cond_per_batch,
    pack_targets_tokens,
    simple_norm,
    target_time_observed,
)


class TinyVAE(nn.Module):
    def forward(self, x_tok, entity_pad=None):
        values = x_tok[..., 0]
        obs = x_tok[..., 1]
        if entity_pad is None:
            present = torch.ones_like(values[:, :1])
        else:
            present = (~entity_pad).to(dtype=values.dtype).unsqueeze(1)
        denom = present.sum(dim=2).clamp_min(1.0)
        pooled = (values * present).sum(dim=2) / denom
        obs_frac = (obs * present).sum(dim=2) / denom
        mu = torch.stack(
            [
                pooled,
                0.5 * pooled + obs_frac,
                torch.sin(pooled),
            ],
            dim=-1,
        )
        return None, mu, None


class TinySummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, ctx_diff, dt=None, pad_mask=None, obs_mask=None):
        if pad_mask is None:
            present = torch.ones_like(x[..., :1])
        else:
            present = pad_mask[:, None, :, None].to(dtype=x.dtype)
        denom = present.sum(dim=2).clamp_min(1.0)
        level = (x * present).sum(dim=(2, 3)) / denom.squeeze(-1)
        diff = (ctx_diff * present).sum(dim=(2, 3)) / denom.squeeze(-1)
        out = torch.stack([level, diff, level - diff], dim=-1) * self.scale
        return out, None


def _batch(seed: int = 0):
    torch.manual_seed(seed)
    b, n, k, f, h = 2, 3, 4, 2, 3
    v = torch.randn(b, n, k, f)
    t = torch.randn(b, n, k, f) * 0.1
    y = torch.randn(b, n, h)
    entity_mask = torch.tensor([[True, True, False], [True, True, True]])
    y_obs_mask = torch.ones(b, n, h, dtype=torch.bool)
    y_obs_mask[0, 1, 2] = False
    y_obs_mask[0, 2, :] = False
    y_obs_mask[~entity_mask] = False
    meta = {
        "entity_mask": entity_mask,
        "delta_t": torch.arange(k, dtype=torch.float32).view(1, 1, k).expand(b, n, k).clone(),
        "delta_t_y": torch.arange(1, h + 1, dtype=torch.float32).view(1, 1, h).expand(b, n, h).clone(),
        "x_obs_mask": torch.ones(b, n, k, f, dtype=torch.bool),
        "y_obs_mask": y_obs_mask,
        "context_end_time_keys": torch.tensor([1000, 2000], dtype=torch.int64),
        "date_keys": torch.tensor([0, 0], dtype=torch.int64),
        "cache_asset_ids": torch.tensor([[0, 1, -1], [0, 1, 2]], dtype=torch.int64),
        "cache_window_starts": torch.tensor([[10, 10, -1], [20, 20, 20]], dtype=torch.int64),
    }
    return (v, t), y, meta


def _config(tmp_path):
    vae_ckpt = tmp_path / "vae.pt"
    sum_ckpt = tmp_path / "sum.pt"
    vae_ckpt.write_bytes(b"vae")
    sum_ckpt.write_bytes(b"sum")
    return SimpleNamespace(
        DATASET_KEY="toy",
        DATA_DIR=str(tmp_path / "data"),
        ARTIFACT_ROOT=str(tmp_path / "artifacts"),
        PRED=3,
        WINDOW=4,
        BATCH_SIZE=2,
        DATES_PER_BATCH=2,
        VAE_LATENT_CHANNELS=3,
        LATENT_NORM_MODE="global",
        SUM_CONTEXT_LEN=4,
        SUM_CONTEXT_DIM=3,
        TARGET_COL="pm25",
        TARGET_COLS=None,
        TARGET_SOURCE="cache_target",
        split_policy="global_purged_horizon",
        split_scope="global_target_time",
        COVERAGE=0.0,
        VAE_CKPT=str(vae_ckpt),
        SUM_CKPT=str(sum_ckpt),
        SUM_FT_MODE="none",
        date_batching=True,
        exact_timestamp_batches=True,
        DIFF_PRECOMPUTE_INPUTS=True,
        DIFF_PRECOMPUTE_LATENT_DTYPE="float32",
        DIFF_PRECOMPUTE_SUMMARY_DTYPE="float16",
        DIFF_PRECOMPUTE_DIR=str(tmp_path / "cache"),
    )


def _build_cache(tmp_path):
    batch = _batch()
    cfg = _config(tmp_path)
    loader = [batch]
    cache = build_or_load_diffusion_input_cache(
        train_dl=loader,
        val_dl=loader,
        test_dl=loader,
        vae=TinyVAE(),
        summarizer=TinySummarizer(),
        device=torch.device("cpu"),
        config_obj=cfg,
        summary_ft_mode="none",
        verbose=False,
    )
    return cache, batch


def test_precomputed_latents_match_live_encoder_for_observed_horizons(tmp_path):
    cache, ((v, t), y, meta) = _build_cache(tmp_path)
    cache.train.reset()
    cached = cache.train.next_batch(
        meta,
        device=torch.device("cpu"),
        mu_mean=cache.mu_mean,
        mu_std=cache.mu_std,
        load_latents=True,
        load_summary=False,
    )

    x_tok, entity_pad, obs = pack_targets_tokens(
        y,
        meta["entity_mask"],
        device=torch.device("cpu"),
        y_obs_mask=meta["y_obs_mask"],
    )
    _, mu_raw, _ = TinyVAE()(x_tok, entity_pad)
    obs_any = target_time_observed(obs)
    direct = simple_norm(mu_raw, cache.mu_mean, cache.mu_std) * obs_any.unsqueeze(-1)

    torch.testing.assert_close(cached.mu_norm, direct, atol=1e-6, rtol=1e-6)
    assert torch.equal(cached.obs_any, obs_any)
    assert torch.equal(cached.mu_norm[~obs_any], torch.zeros_like(cached.mu_norm[~obs_any]))


def test_precomputed_raw_summary_matches_live_summarizer_and_batch_norm(tmp_path):
    cache, ((v, t), y, meta) = _build_cache(tmp_path)
    cache.train.reset()
    cached = cache.train.next_batch(
        meta,
        device=torch.device("cpu"),
        mu_mean=cache.mu_mean,
        mu_std=cache.mu_std,
        load_latents=False,
        load_summary=True,
    )

    direct = build_context(
        TinySummarizer(),
        v,
        t,
        meta["entity_mask"],
        torch.device("cpu"),
        dt=meta["delta_t"],
        x_obs_mask=meta["x_obs_mask"],
        norm=False,
        requires_grad=False,
    )
    torch.testing.assert_close(cached.summary_raw, direct, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        normalize_cond_per_batch(cached.summary_raw),
        normalize_cond_per_batch(direct),
        atol=2e-3,
        rtol=2e-3,
    )


def test_cache_metadata_rejects_stale_batch_identity(tmp_path):
    cache, ((v, t), y, meta) = _build_cache(tmp_path)
    bad_meta = dict(meta)
    bad_meta["context_end_time_keys"] = meta["context_end_time_keys"] + 1

    cache.train.reset()
    with pytest.raises(RuntimeError, match="cache order mismatch"):
        cache.train.next_batch(
            bad_meta,
            device=torch.device("cpu"),
            mu_mean=cache.mu_mean,
            mu_std=cache.mu_std,
            load_latents=True,
            load_summary=False,
        )


def test_summary_cache_disabled_when_summarizer_is_trainable(tmp_path):
    cfg = _config(tmp_path)
    allowed, reason = cache_allowed(cfg, summary_ft_mode="top")
    assert not allowed
    assert "summarizer fine-tuning" in reason
