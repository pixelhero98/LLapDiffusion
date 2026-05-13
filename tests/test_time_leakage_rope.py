import inspect
import io
import zipfile
from types import SimpleNamespace

import pytest
import torch

from Model.laptrans import LaplaceTransformEncoder
from Model.summarizer import LaplaceAE
from Model.time_utils import relative_time_offsets


def test_relative_offset_dt_is_not_accumulated_twice():
    dt = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    rel_t = relative_time_offsets(dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_increment_dt_falls_back_to_cumulative_offsets():
    dt = torch.tensor([[[1.0], [1.0], [1.0], [1.0]]])
    rel_t = relative_time_offsets(dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_irregular_relative_offset_dt_is_preserved():
    dt = torch.tensor([[[0.0], [1.0], [4.0], [5.0]]])
    rel_t = relative_time_offsets(dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 4.0, 5.0]]))


def test_laplace_ae_uses_shared_time_offsets():
    dt = torch.tensor([[[0.0], [1.0], [4.0], [5.0]]])
    rel_t = LaplaceAE._relative_time_from_dt(dt)
    assert torch.allclose(rel_t, relative_time_offsets(dt))



def test_summarizer_position_defaults_and_learned_abs_override():
    from configs import config as cfg

    assert cfg.SUM_POS_ENCODING == "continuous_rope"
    assert float(cfg.SUM_ROPE_BASE) == 10000.0

    kwargs = dict(
        num_entities=2,
        feat_dim=1,
        window_size=4,
        mix_dim=8,
        tv_hidden=8,
        out_len=2,
        context_dim=16,
        enc_layers=1,
        n_heads=2,
        dropout=0.0,
        time2vec_dim=3,
    )
    default_model = LaplaceAE(**kwargs)
    assert default_model.pos_encoding == "continuous_rope"
    assert default_model.use_rope is True
    assert default_model.use_learned_pos is False

    learned_model = LaplaceAE(**kwargs, pos_encoding="learned_abs")
    assert learned_model.pos_encoding == "learned_abs"
    assert learned_model.use_rope is False
    assert learned_model.use_learned_pos is True


def test_summarizer_builder_passes_rope_base():
    from trainers import train_val_summarizer as tvs

    cfg = SimpleNamespace(
        WINDOW=5,
        SUM_MIX_DIM=8,
        SUM_TV_HIDDEN=8,
        SUM_CONTEXT_LEN=2,
        SUM_CONTEXT_DIM=16,
        SUM_DROPOUT=0.0,
        SUM_TIME2VEC_DIM=3,
        SUM_IRREG_POOLING="none",
        SUM_IRREG_HIDDEN=8,
        SUM_IRREG_RES_SCALE=0.1,
        SUM_T_TOKEN_MODE="none",
        SUM_T_TOKEN_SCALE=0.1,
        SUM_POS_ENCODING="continuous_rope",
        SUM_ROPE_BASE=256.0,
    )
    xb = (torch.zeros(1, 2, 5, 1), torch.zeros(1, 2, 5, 1))
    yb = torch.zeros(1, 2, 2)
    model = tvs._build_model([(xb, yb, {})], None, torch.device("cpu"), config=cfg, verbose=False)

    attn = model.history_encoder.layers[0].self_attn
    expected = 1.0 / (256.0 ** (torch.arange(0, attn.rope_dim, 2, dtype=torch.float32) / attn.rope_dim))
    assert torch.allclose(attn.inv_freq.cpu(), expected)


def test_vae_checkpoint_path_preserves_entity_suffix(tmp_path):
    from trainers import train_val_latent as tvl

    cfg = SimpleNamespace(
        VAE_DIR=str(tmp_path),
        PRED=20,
        VAE_LATENT_CHANNELS=12,
        VAE_ENTITY_CONDITION=True,
    )

    assert tvl._vae_checkpoint_path("elbo", config=cfg).name == "pred-20_ch-12_entity_elbo.pt"


def test_run_single_pred_applies_output_dirs_after_pred_update(monkeypatch, tmp_path):
    import train_val_pipeline as pipeline

    vae_ckpt = tmp_path / "pred-20_ch-12_entity_elbo.pt"
    sum_ckpt = tmp_path / "20-12-summarizer.pt"
    vae_ckpt.write_text("vae")
    sum_ckpt.write_text("sum")
    cfg = SimpleNamespace(DATASET_KEY="crypto")

    def fake_update(pred, config):
        config.PRED = pred
        config.VAE_LATENT_CHANNELS = 12
        config.VAE_CKPT = str(vae_ckpt)
        config.SUM_CKPT = str(sum_ckpt)
        config.OUT_DIR = "preset-output"
        config.CKPT_DIR = "preset-checkpoints"
        config.POLE_PLOT_DIR = "preset-poles"

    fake_latent = SimpleNamespace(run=lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent should be skipped")))
    fake_summarizer = SimpleNamespace(run=lambda **kwargs: (_ for _ in ()).throw(AssertionError("summarizer should be skipped")))
    fake_llapdiff = SimpleNamespace(run=lambda **kwargs: {"eval_stats": {}, "loaded_checkpoint": "ok.pt"})

    monkeypatch.setattr(pipeline, "_update_config_for_pred", fake_update)
    monkeypatch.setattr(pipeline, "_import_trainers", lambda: (fake_latent, fake_summarizer, fake_llapdiff))
    monkeypatch.setattr(pipeline, "prepare_dataloaders", lambda config: (None, None, None, (0, 0, 0)))

    pipeline.run_single_pred(20, base_out_dir=tmp_path / "out", base_ckpt_dir=tmp_path / "ckpt", config=cfg)

    assert cfg.VAE_CKPT == str(vae_ckpt)
    assert cfg.OUT_DIR == str(tmp_path / "out" / "pred-20")
    assert cfg.CKPT_DIR == str(tmp_path / "ckpt" / "pred-20")
    assert cfg.POLE_PLOT_DIR == str(tmp_path / "out" / "pred-20" / "pole_plots")


def test_missing_dataset_archive_fails_early(tmp_path, monkeypatch):
    from configs import dataset_archives

    monkeypatch.delenv(dataset_archives.DATASET_ZIP_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="Provide a dataset cache zip"):
        dataset_archives.resolve_dataset_dir(tmp_path / "Dataset" / "crypto", repo_root=tmp_path)


def test_safe_zip_extraction_rejects_path_traversal(tmp_path):
    from configs.dataset_archives import extract_zip_safely

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    payload.seek(0)

    with zipfile.ZipFile(payload) as archive:
        with pytest.raises(ValueError, match="Unsafe path"):
            extract_zip_safely(archive, tmp_path / "extract")

def test_laplace_relative_time_preserves_regular_offsets():
    dt = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_laplace_relative_time_preserves_irregular_offsets():
    dt = torch.tensor([[0.0, 1.0, 4.0, 5.0]])
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 4.0, 5.0]]))


def test_laplace_relative_time_converts_increments():
    dt = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_laplace_relative_time_prefers_explicit_t():
    dt = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    t = torch.tensor([[10.0, 11.0, 14.0, 15.0]])
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt, t=t)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 4.0, 5.0]]))


def test_target_dt_flatten_then_laplace_preserves_offsets():
    from trainers import train_val_llapdiff as tv

    meta = {"delta_t_y": torch.tensor([[[0.0, 1.0, 4.0, 5.0], [0.0, 1.0, 4.0, 5.0]]])}
    mask = torch.tensor([[True, True]])
    dt_b = tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt_b)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 4.0, 5.0]]))


def test_target_dt_flatten_does_not_depend_on_target_values():
    from trainers import train_val_llapdiff as tv

    meta = {
        "delta_t_y": torch.tensor(
            [[[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]]]
        )
    }
    mask = torch.tensor([[True, True]])

    dt_b = tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")

    assert torch.allclose(dt_b, torch.tensor([[0.0, 1.5, 3.0, 4.5]]))


def test_vae_target_mask_excludes_zero_filled_missing_targets():
    from trainers import train_val_latent as tvl

    y = torch.tensor([[[1.0, 0.0, 3.0]]])
    entity_mask = torch.tensor([[True]])
    y_obs_mask = torch.tensor([[[True, False, True]]])

    prepared = tvl._prepare_latent_batch(
        y,
        entity_mask,
        y_obs_mask=y_obs_mask,
        p_drop=0.0,
        noise_std=0.0,
    )

    assert prepared is not None
    x_tok, y_clean, obs, entity_pad = prepared
    assert obs.tolist() == [[[True, False, True]]]
    assert entity_pad.tolist() == [[False]]
    assert x_tok[0, 1, 0, 0].item() == 0.0
    assert x_tok[0, 1, 0, 1].item() == 0.0

    y_hat = torch.tensor([[[1.0, 100.0, 3.0]]])
    loss, count = tvl._masked_mse(y_hat, y_clean, obs)
    assert count == 2
    assert loss.item() == 0.0


def test_default_config_allows_imputation_but_keeps_aux_inactive():
    from configs import config as cfg

    assert bool(getattr(cfg, "IMPUTATION_TRAINING")) is True
    assert float(getattr(cfg, "TARGET_MASK_AUX_P")) == 0.0


def test_target_mask_aux_guard_requires_imputation_training_for_positive_probability():
    from trainers import train_val_llapdiff as tv

    forecast_cfg = SimpleNamespace(TARGET_MASK_AUX_P=0.0, IMPUTATION_TRAINING=False)
    impute_cfg = SimpleNamespace(TARGET_MASK_AUX_P=0.2, IMPUTATION_TRAINING=True)
    invalid_cfg = SimpleNamespace(TARGET_MASK_AUX_P=0.2, IMPUTATION_TRAINING=False)

    assert tv._effective_target_mask_aux_probability(forecast_cfg) == 0.0
    assert tv._effective_target_mask_aux_probability(impute_cfg) == 0.2
    try:
        tv._effective_target_mask_aux_probability(invalid_cfg)
    except ValueError as exc:
        assert "IMPUTATION_TRAINING=True" in str(exc)
    else:
        raise AssertionError("positive TARGET_MASK_AUX_P should require IMPUTATION_TRAINING=True")


def test_history_stat_tokens_preserve_context_offsets():
    from trainers import train_val_llapdiff as tv

    V = torch.ones(1, 2, 4, 1)
    T = torch.zeros(1, 2, 4, 1)
    mask = torch.tensor([[True, True]])
    dt = torch.tensor([[[0.0, 1.0, 4.0, 5.0], [0.0, 1.0, 4.0, 5.0]]])

    stats = tv._history_stat_tokens(V, T, mask, torch.device("cpu"), dt=dt)

    assert torch.allclose(stats[0, :, 2], torch.tensor([0.0, 0.2, 0.8, 1.0]))


def test_forecast_generation_does_not_condition_on_target_values_or_masks():
    from trainers import train_val_llapdiff as tv

    class FakeDiffModel:
        def __init__(self):
            self.calls = []

        def eval(self):
            return None

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.zeros(kwargs["shape"])

    old_build = tv._build_cond_summary_pair
    old_pack = tv.pack_targets_tokens
    old_encode = tv.encode_mu_norm
    old_decode = tv.decode_latents_with_vae
    diff_model = FakeDiffModel()

    def fake_build(*args, **kwargs):
        return torch.zeros(1, 2, 4), torch.zeros(1, 2, 4)

    def fake_pack(yb, mask_bn, device, y_obs_mask=None):
        return torch.zeros(1, 4, 2, 2), torch.zeros(1, 2, dtype=torch.bool), torch.ones(1, 4, 2, dtype=torch.bool)

    def fake_encode(*args, **kwargs):
        return torch.zeros(1, 4, 3)

    def fake_decode(*args, **kwargs):
        return torch.zeros(1, 4, 2, 1)

    tv._build_cond_summary_pair = fake_build
    tv.pack_targets_tokens = fake_pack
    tv.encode_mu_norm = fake_encode
    tv.decode_latents_with_vae = fake_decode
    try:
        xb = (torch.ones(1, 2, 3, 1), torch.zeros(1, 2, 3, 1))
        yb = torch.ones(1, 2, 4)
        meta = {
            "entity_mask": torch.tensor([[True, True]]),
            "delta_t": torch.zeros(1, 2, 3),
            "delta_t_y": torch.tensor([[[0.0, 1.0, 4.0, 5.0], [0.0, 1.0, 4.0, 5.0]]]),
            "x_obs_mask": torch.ones(1, 2, 3, 1, dtype=torch.bool),
            "y_obs_mask": torch.ones(1, 2, 4, dtype=torch.bool),
        }
        tv.evaluate_regression(
            diff_model,
            vae=object(),
            summarizer=object(),
            dataloader=[(xb, yb, meta)],
            device=torch.device("cpu"),
            mu_mean=torch.zeros(3),
            mu_std=torch.ones(3),
            config=SimpleNamespace(NUM_EVAL_SAMPLES=1),
            steps=2,
            crps_pair_samples=1,
        )
    finally:
        tv._build_cond_summary_pair = old_build
        tv.pack_targets_tokens = old_pack
        tv.encode_mu_norm = old_encode
        tv.decode_latents_with_vae = old_decode

    assert len(diff_model.calls) == 1
    call = diff_model.calls[0]
    assert torch.allclose(call["dt"], torch.tensor([[0.0, 1.0, 4.0, 5.0]]))
    assert "y_obs" not in call
    assert "obs_mask" not in call


def test_imputation_generation_only_uses_intentionally_observed_target_tokens():
    from tools import llapdiff_checkpoint_eval as ce

    source = inspect.getsource(ce._evaluate_impute_case)
    generate_call = source[source.index("x0_norm = diff_model.generate("): source.index("all_samples.append")]

    assert "y_obs = mu_norm * keep_mask.unsqueeze(-1).to(dtype=mu_norm.dtype)" in source
    assert "hidden_valid = (obs & (~keep_mask.unsqueeze(-1)))" in source
    assert "y_obs=y_obs" in generate_call
    assert "obs_mask=keep_mask" in generate_call
    assert "hidden_valid" not in generate_call


def test_random_imputation_keep_mask_generator_advances_between_batches():
    from tools import llapdiff_checkpoint_eval as ce

    obs_any = torch.ones(4, 20, dtype=torch.bool)
    generator = torch.Generator(device=obs_any.device)
    generator.manual_seed(1234)

    first = ce._make_random_keep(obs_any, frac=0.70, generator=generator)
    second = ce._make_random_keep(obs_any, frac=0.70, generator=generator)

    assert not torch.equal(first, second)


def test_continuous_rope_summarizer_forward_shape_and_finiteness():
    torch.manual_seed(7)
    model = LaplaceAE(
        num_entities=3,
        feat_dim=2,
        window_size=5,
        mix_dim=8,
        tv_hidden=8,
        out_len=2,
        context_dim=16,
        enc_layers=2,
        n_heads=2,
        dropout=0.0,
        time2vec_dim=3,
        pos_encoding="continuous_rope",
    )
    x = torch.randn(2, 5, 3, 2)
    ctx_diff = torch.randn(2, 5, 3, 2)
    dt = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 3.0, 4.0], [5.0, 5.0, 8.0], [6.0, 7.0, 9.0]],
            [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [4.0, 3.0, 2.0], [6.0, 6.0, 5.0], [8.0, 7.0, 8.0]],
        ]
    )
    entity_mask = torch.tensor([[True, True, False], [True, True, True]])
    obs_mask = torch.ones(2, 5, 3, 2, dtype=torch.bool)

    context, aux = model(x, pad_mask=entity_mask, ctx_diff=ctx_diff, dt=dt, obs_mask=obs_mask)

    assert context.shape == (2, 2, 16)
    assert aux["rel_t"].shape == (2, 5, 3)
    assert torch.isfinite(context).all()
    assert torch.isfinite(aux["rel_t_unit"]).all()
