import io
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llapdiffusion.models.laptrans import LaplaceTransformEncoder
from llapdiffusion.models.summarizer import LaplaceAE
from llapdiffusion.models.time_utils import relative_time_offsets


def test_relative_offset_dt_is_not_accumulated_twice():
    dt = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    rel_t = relative_time_offsets(dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_offset_dt_is_shifted_to_first_timestamp():
    dt = torch.tensor([[[5.0], [6.0], [9.0], [10.0]]])
    rel_t = relative_time_offsets(dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 4.0, 5.0]]))


def test_relative_time_offsets_rejects_decreasing_clocks():
    dt = torch.tensor([[[0.0], [2.0], [1.0], [3.0]]])
    with pytest.raises(ValueError, match="nondecreasing"):
        relative_time_offsets(dt)


def test_irregular_relative_offset_dt_is_preserved():
    dt = torch.tensor([[[0.0], [1.0], [4.0], [5.0]]])
    rel_t = relative_time_offsets(dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 4.0, 5.0]]))


def test_relative_time_offsets_can_preserve_supplied_query_offsets():
    dt = torch.tensor([[[1.0], [4.0], [5.0]]])
    rel_t = relative_time_offsets(dt, recenter=False)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[1.0, 4.0, 5.0]]))


def test_laplace_ae_uses_shared_time_offsets():
    dt = torch.tensor([[[0.0], [1.0], [4.0], [5.0]]])
    rel_t = LaplaceAE._relative_time_from_dt(dt)
    assert torch.allclose(rel_t, relative_time_offsets(dt))


def test_index_backed_dataset_target_offsets_are_context_end_relative():
    from llapdiffusion.datasets.fin_dataset import _IndexBackedDataset

    ds = _IndexBackedDataset.__new__(_IndexBackedDataset)
    ds.pairs = np.asarray([[0, 0]], dtype=np.int64)
    ds.assets = ["asset"]
    ds.window = 3
    ds.horizon = 3
    ds.regression = True
    ds.keep_time_meta = "full"
    ds.clamp_sigma = 10.0
    ds.per_ticker = False
    ds.mean_x = np.zeros((1, 1, 1), dtype=np.float32)
    ds.std_x = np.ones((1, 1, 1), dtype=np.float32)
    ds.mean_y = 0.0
    ds.std_y = 1.0
    ds.native_time_scale_seconds = 86400.0
    ds.native_time_scale_name = "1D"

    features = np.arange(6, dtype=np.float32).reshape(6, 1)
    targets = np.arange(6, dtype=np.float32)
    times = np.array(
        ["2026-01-10", "2026-01-11", "2026-01-14", "2026-01-15", "2026-01-18", "2026-01-19"],
        dtype="datetime64[D]",
    )
    ds._get_arrays = lambda aid: (features, targets, times, None, None)

    _, _, meta = ds[0]

    assert np.allclose(meta["delta_t"], np.array([0.0, 1.0, 4.0], dtype=np.float32))
    assert np.allclose(meta["delta_t_y"], np.array([1.0, 4.0, 5.0], dtype=np.float32))


def test_collate_target_time_fallback_anchors_at_context_end():
    from llapdiffusion.datasets.fin_dataset import make_collate_level_and_firstdiff

    collate = make_collate_level_and_firstdiff(n_entities=1)
    _, _, meta = collate(
        [
            {
                "asset_id": 0,
                "V": np.zeros((3, 1), dtype=np.float32),
                "T": np.zeros((3, 1), dtype=np.float32),
                "y": np.zeros(3, dtype=np.float32),
                "ctx_times": np.array([10, 11, 14]),
                "y_times": np.array([15, 18, 19]),
            }
        ]
    )

    assert torch.allclose(meta["delta_t"][0, 0], torch.tensor([0.0, 1.0, 4.0]))
    assert torch.allclose(meta["delta_t_y"][0, 0], torch.tensor([1.0, 4.0, 5.0]))


def test_collate_datetime_time_fallback_anchors_at_context_end_in_seconds():
    from llapdiffusion.datasets.fin_dataset import make_collate_level_and_firstdiff

    collate = make_collate_level_and_firstdiff(n_entities=1)
    _, _, meta = collate(
        [
            {
                "asset_id": 0,
                "V": np.zeros((3, 1), dtype=np.float32),
                "T": np.zeros((3, 1), dtype=np.float32),
                "y": np.zeros(2, dtype=np.float32),
                "ctx_times": np.array(["2026-01-10", "2026-01-11", "2026-01-14"], dtype="datetime64[D]"),
                "y_times": np.array(["2026-01-15", "2026-01-18"], dtype="datetime64[D]"),
            }
        ]
    )

    assert torch.allclose(meta["delta_t"][0, 0], torch.tensor([0.0, 86400.0, 345600.0]))
    assert torch.allclose(meta["delta_t_y"][0, 0], torch.tensor([86400.0, 345600.0]))


def test_collate_target_time_fallback_requires_context_anchor():
    from llapdiffusion.datasets.fin_dataset import make_collate_level_and_firstdiff

    collate = make_collate_level_and_firstdiff(n_entities=1)
    with pytest.raises(ValueError, match="Cannot infer delta_t_y"):
        collate(
            [
                {
                    "asset_id": 0,
                    "V": np.zeros((3, 1), dtype=np.float32),
                    "T": np.zeros((3, 1), dtype=np.float32),
                    "y": np.zeros(2, dtype=np.float32),
                    "y_times": np.array([15, 18]),
                }
            ]
        )



def test_summarizer_position_defaults_and_learned_abs_override():
    from llapdiffusion.configs import config as cfg

    assert cfg.SUM_POS_ENCODING == "learned_abs"
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
    assert default_model.pos_encoding == "learned_abs"
    assert default_model.use_rope is False
    assert default_model.use_learned_pos is True

    learned_model = LaplaceAE(**kwargs, pos_encoding="learned_abs")
    assert learned_model.pos_encoding == "learned_abs"
    assert learned_model.use_rope is False
    assert learned_model.use_learned_pos is True

    rope_model = LaplaceAE(**kwargs, pos_encoding="continuous_rope")
    assert rope_model.pos_encoding == "continuous_rope"
    assert rope_model.use_rope is True
    assert rope_model.use_learned_pos is False


def test_summarizer_builder_passes_rope_base():
    from llapdiffusion.trainers import train_val_summarizer as tvs

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


def test_public_presets_apply_only_selected_overrides(monkeypatch, tmp_path):
    from llapdiffusion.configs import config as base_cfg
    from llapdiffusion.configs import dataset_defaults as dd

    monkeypatch.setattr(dd, "resolve_dataset_dir", lambda expected, **kwargs: expected)

    def make_cfg():
        names = [
            "VAE_ENTITY_CONDITION",
            "VAE_INPUT_DROPOUT",
            "VAE_NOISE_STD",
            "VAE_CONSIST_LAMBDA",
            "VAE_RECON_BALANCE",
            "VAE_MAX_PATIENCE",
            "SUM_POS_ENCODING",
            "SUM_LOSS_W_DT",
            "SUM_LOSS_W_OBS",
            "SUM_CHANNEL_BALANCED_X_LOSS",
            "SUM_IRREG_POOLING",
            "SUM_T_TOKEN_MODE",
            "SUM_T_TOKEN_SCALE",
            "SUM_PATIENCE",
            "SUM_LR",
            "SUM_AMP",
            "SUM_TIME2VEC_DIM",
            "PRIMARY_EVAL_METRIC",
        ]
        cfg = SimpleNamespace(**{name: getattr(base_cfg, name) for name in names})
        cfg.ARTIFACT_ROOT = str(tmp_path / "ldt")
        return cfg

    improved = {"physionet", "crypto"}
    for key in dd.dataset_keys():
        cfg = dd.apply_dataset_preset(make_cfg(), key, pred=dd.default_horizons(key)[-1])
        if key in improved:
            assert cfg.SUM_POS_ENCODING == "continuous_rope"
            assert cfg.VAE_INPUT_DROPOUT == 0.35
            assert cfg.VAE_NOISE_STD == 0.02
            assert cfg.VAE_CONSIST_LAMBDA == 0.05
            assert cfg.VAE_RECON_BALANCE == "coverage"
            assert cfg.SUM_LOSS_W_DT == 0.05
            assert cfg.SUM_LOSS_W_OBS == 0.05
            assert cfg.SUM_CHANNEL_BALANCED_X_LOSS is True
            assert cfg.SUM_IRREG_POOLING == "repair"
            assert cfg.SUM_T_TOKEN_MODE == "both"
        else:
            assert cfg.SUM_POS_ENCODING == "learned_abs"
            assert cfg.VAE_INPUT_DROPOUT == 0.20
            assert cfg.VAE_NOISE_STD == 0.01
            assert cfg.VAE_CONSIST_LAMBDA == 0.0
            assert cfg.VAE_RECON_BALANCE == "none"
            assert cfg.SUM_LOSS_W_DT == 0.0
            assert cfg.SUM_LOSS_W_OBS == 0.0
            assert cfg.SUM_CHANNEL_BALANCED_X_LOSS is False
            assert cfg.SUM_IRREG_POOLING == "none"
            assert cfg.SUM_T_TOKEN_MODE == "none"
        if key == "bms_air":
            assert cfg.SUM_LR == 1e-4
            assert cfg.SUM_AMP is False
        assert cfg.TARGET_MASK_AUX_P == 0.0

    reused = make_cfg()
    dd.apply_dataset_preset(reused, "physionet", pred=12)
    dd.apply_dataset_preset(reused, "noaa_uk", pred=168)
    assert reused.SUM_POS_ENCODING == "learned_abs"
    assert reused.VAE_INPUT_DROPOUT == 0.20
    assert reused.VAE_RECON_BALANCE == "none"
    assert reused.SUM_CHANNEL_BALANCED_X_LOSS is False


def test_vae_checkpoint_path_preserves_entity_suffix(tmp_path):
    from llapdiffusion.trainers import train_val_latent as tvl

    cfg = SimpleNamespace(
        VAE_DIR=str(tmp_path),
        PRED=20,
        VAE_LATENT_CHANNELS=12,
        VAE_ENTITY_CONDITION=True,
    )

    assert tvl._vae_checkpoint_path("elbo", config=cfg).name == "pred-20_ch-12_entity_elbo.pt"


def test_vae_amp_flag_controls_cuda_autocast():
    from llapdiffusion.trainers import train_val_latent as tvl

    assert tvl._vae_amp_enabled(torch.device("cpu"), config=SimpleNamespace(VAE_AMP=True)) is False
    assert tvl._vae_amp_enabled(torch.device("cuda"), config=SimpleNamespace(VAE_AMP=False)) is False
    assert tvl._vae_amp_enabled(torch.device("cuda"), config=SimpleNamespace(VAE_AMP=True)) is True
    assert tvl._vae_amp_enabled(torch.device("cuda"), config=SimpleNamespace()) is False


def test_run_single_pred_applies_output_dirs_after_pred_update(monkeypatch, tmp_path):
    from llapdiffusion import pipeline as pipeline

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
    fake_llapdiff = SimpleNamespace(run=lambda **kwargs: {"eval_stats": {}, "loaded_checkpoint": None})

    monkeypatch.setattr(pipeline, "_update_config_for_pred", fake_update)
    monkeypatch.setattr(pipeline, "_import_trainers", lambda: (fake_latent, fake_summarizer, fake_llapdiff))
    monkeypatch.setattr(pipeline, "prepare_dataloaders", lambda config: (None, None, None, (0, 0, 0)))

    pipeline.run_single_pred(20, base_out_dir=tmp_path / "out", base_ckpt_dir=tmp_path / "ckpt", config=cfg)

    assert cfg.VAE_CKPT == str(vae_ckpt)
    assert cfg.OUT_DIR == str(tmp_path / "out" / "pred-20")
    assert cfg.CKPT_DIR == str(tmp_path / "ckpt" / "pred-20")
    assert cfg.POLE_PLOT_DIR == str(tmp_path / "out" / "pred-20" / "pole_plots")


def test_run_single_pred_checkpoint_eval_is_opt_in(monkeypatch, tmp_path):
    from llapdiffusion import pipeline

    vae_ckpt = tmp_path / "pred-20_ch-12_entity_elbo.pt"
    sum_ckpt = tmp_path / "20-12-summarizer.pt"
    diff_ckpt = tmp_path / "llapdiff.pt"
    for path in (vae_ckpt, sum_ckpt, diff_ckpt):
        path.write_text("checkpoint")
    cfg = SimpleNamespace(DATASET_KEY="crypto")
    calls = []

    def fake_update(pred, config):
        config.PRED = pred
        config.VAE_LATENT_CHANNELS = 12
        config.VAE_CKPT = str(vae_ckpt)
        config.SUM_CKPT = str(sum_ckpt)
        config.OUT_DIR = "preset-output"
        config.CKPT_DIR = "preset-checkpoints"

    fake_latent = SimpleNamespace(run=lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent should be skipped")))
    fake_summarizer = SimpleNamespace(run=lambda **kwargs: (_ for _ in ()).throw(AssertionError("summarizer should be skipped")))
    fake_llapdiff = SimpleNamespace(run=lambda **kwargs: {"eval_stats": {"mse": 1.0}, "loaded_checkpoint": str(diff_ckpt)})

    def fake_evaluate_checkpoint(*args, **kwargs):
        calls.append((args, kwargs))
        return {"forecast_test": {"mse": 2.0}}

    monkeypatch.setattr(pipeline, "_update_config_for_pred", fake_update)
    monkeypatch.setattr(pipeline, "_import_trainers", lambda: (fake_latent, fake_summarizer, fake_llapdiff))
    monkeypatch.setattr(pipeline, "prepare_dataloaders", lambda config: (None, None, None, (0, 0, 0)))
    monkeypatch.setitem(
        sys.modules,
        "llapdiffusion.tools.llapdiff_checkpoint_eval",
        SimpleNamespace(evaluate_checkpoint=fake_evaluate_checkpoint),
    )

    default_result = pipeline.run_single_pred(20, base_out_dir=tmp_path / "out", base_ckpt_dir=tmp_path / "ckpt", config=cfg)
    assert calls == []
    assert default_result["balanced_evaluation"] is None

    explicit_result = pipeline.run_single_pred(
        20,
        run_checkpoint_eval=True,
        checkpoint_eval_num_samples=3,
        checkpoint_eval_forecast_num_samples=4,
        checkpoint_eval_imputation_num_samples=5,
        checkpoint_eval_max_eval_batches=6,
        checkpoint_eval_random_mask_ratio=0.25,
        base_out_dir=tmp_path / "out",
        base_ckpt_dir=tmp_path / "ckpt",
        config=cfg,
    )
    assert explicit_result["balanced_evaluation"] == {"forecast_test": {"mse": 2.0}}
    assert calls[0][1] == {
        "label": "crypto_pred20",
        "random_mask_ratio": 0.25,
        "num_samples": 3,
        "forecast_num_samples": 4,
        "imputation_num_samples": 5,
        "max_eval_batches": 6,
    }


def test_llapdiff_train_target_mask_aux_cli_overrides(monkeypatch):
    from llapdiffusion import pipeline

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llapdiff-train",
            "--dataset-key",
            "crypto",
            "--target-mask-aux-p",
            "0.35",
            "--target-mask-aux-keep-mode",
            "regular",
            "--target-mask-aux-keep-prob",
            "0.65",
            "--target-mask-aux-keep-stride",
            "3",
            "--target-mask-aux-start-epoch",
            "7",
        ],
    )

    args = pipeline._parse_args()

    assert args.target_mask_aux_p == 0.35
    assert args.target_mask_aux_keep_mode == "regular"
    assert args.target_mask_aux_keep_prob == 0.65
    assert args.target_mask_aux_keep_stride == 3
    assert args.target_mask_aux_start_epoch == 7

    overrides = pipeline._training_overrides_from_args(args)
    cfg = SimpleNamespace(
        IMPUTATION_TRAINING=False,
        TARGET_MASK_AUX_P=0.0,
        TARGET_MASK_AUX_KEEP_MODE="prefix",
        TARGET_MASK_AUX_KEEP_PROB=0.5,
        TARGET_MASK_AUX_KEEP_STRIDE=4,
        TARGET_MASK_AUX_START_EPOCH=10,
    )
    pipeline._apply_training_overrides(overrides, config=cfg)

    assert cfg.IMPUTATION_TRAINING is True
    assert cfg.TARGET_MASK_AUX_P == 0.35
    assert cfg.TARGET_MASK_AUX_KEEP_MODE == "regular"
    assert cfg.TARGET_MASK_AUX_KEEP_PROB == 0.65
    assert cfg.TARGET_MASK_AUX_KEEP_STRIDE == 3
    assert cfg.TARGET_MASK_AUX_START_EPOCH == 7


def test_llapdiff_train_main_applies_dataset_preset_with_requested_pred(monkeypatch):
    from llapdiffusion import pipeline

    calls = {"preset": [], "preds": []}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llapdiff-train",
            "--dataset-key",
            "crypto",
            "--preds",
            "100",
        ],
    )
    monkeypatch.setattr(pipeline, "configure_dataset_archive", lambda *args, **kwargs: None)

    def fake_apply(config, dataset_key, pred=None):
        calls["preset"].append((dataset_key, pred))
        config.OUT_DIR = "out"
        config.CKPT_DIR = "ckpt"

    def fake_run_single_pred(pred, **kwargs):
        calls["preds"].append(pred)
        return {"eval_stats": {}}

    monkeypatch.setattr(pipeline, "apply_dataset_preset", fake_apply)
    monkeypatch.setattr(pipeline, "run_single_pred", fake_run_single_pred)
    monkeypatch.setattr(pipeline, "_print_summary_table", lambda results: None)

    result = pipeline.main()

    assert calls["preset"] == [("crypto", 100)]
    assert calls["preds"] == [100]
    assert result == {100: {"eval_stats": {}}}


def test_missing_dataset_archive_fails_early(tmp_path, monkeypatch):
    from llapdiffusion.configs import dataset_archives

    monkeypatch.delenv(dataset_archives.DATASET_ZIP_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="Provide a dataset cache zip"):
        dataset_archives.resolve_dataset_dir(tmp_path / "llapdiffusion" / "datasets" / "crypto", package_root=tmp_path / "llapdiffusion")


def test_safe_zip_extraction_rejects_path_traversal(tmp_path):
    from llapdiffusion.configs.dataset_archives import extract_zip_safely

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    payload.seek(0)

    with zipfile.ZipFile(payload) as archive:
        with pytest.raises(ValueError, match="Unsafe path"):
            extract_zip_safely(archive, tmp_path / "extract")


def test_bundled_dataset_archive_is_used_when_env_is_absent(tmp_path, monkeypatch):
    from llapdiffusion.configs import dataset_archives

    package_root = tmp_path / "llapdiffusion"
    extract_root = tmp_path / "cache"
    monkeypatch.delenv(dataset_archives.DATASET_ZIP_ENV, raising=False)
    monkeypatch.setenv(dataset_archives.DATASET_EXTRACT_ENV, str(extract_root))
    archive_path = package_root / "datasets" / dataset_archives.DEFAULT_ARCHIVE_NAME
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("crypto/sample.txt", "ok")

    resolved = dataset_archives.resolve_dataset_dir(
        package_root / "datasets" / "crypto",
        package_root=package_root,
    )

    assert resolved == (extract_root / "crypto").resolve()
    assert (resolved / "sample.txt").read_text() == "ok"


def test_finance_cache_metadata_resolves_fin_dataset_loader(tmp_path, monkeypatch):
    from llapdiffusion.configs import dataset_registry

    data_dir = tmp_path / "fin_dataset" / "crypto"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text('{"dataset": "fin_dataset"}', encoding="utf-8")

    marker = object()
    monkeypatch.setattr(dataset_registry, "_import_fin_run_experiment", lambda: marker)

    assert dataset_registry.resolve_run_experiment(data_dir) is marker


def test_laplace_relative_time_preserves_regular_offsets():
    dt = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_laplace_relative_time_preserves_irregular_offsets():
    dt = torch.tensor([[0.0, 1.0, 4.0, 5.0]])
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 4.0, 5.0]]))


def test_laplace_relative_time_preserves_context_end_query_offsets():
    dt = torch.tensor([[1.0, 4.0, 5.0, 9.0]])
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[1.0, 4.0, 5.0, 9.0]]))


def test_laplace_relative_time_prefers_explicit_t():
    dt = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    t = torch.tensor([[10.0, 11.0, 14.0, 15.0]])
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt, t=t)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[0.0, 1.0, 4.0, 5.0]]))


def test_target_dt_flatten_then_laplace_preserves_offsets():
    from llapdiffusion.trainers import train_val_llapdiff as tv

    meta = {"delta_t_y": torch.tensor([[[1.0, 4.0, 5.0, 9.0], [1.0, 4.0, 5.0, 9.0]]])}
    mask = torch.tensor([[True, True]])
    dt_b = tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")
    rel_t = LaplaceTransformEncoder.relative_time(1, 4, torch.float32, torch.device("cpu"), dt=dt_b)
    assert torch.allclose(rel_t.squeeze(-1), torch.tensor([[1.0, 4.0, 5.0, 9.0]]))


def test_target_dt_flatten_rejects_mismatched_valid_entity_grids():
    from llapdiffusion.trainers import train_val_llapdiff as tv

    meta = {
        "delta_t_y": torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 5.0, 7.0]]]
        )
    }
    mask = torch.tensor([[True, True]])

    with pytest.raises(ValueError, match="same query grid"):
        tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")


def test_target_dt_flatten_ignores_padded_entity_grid():
    from llapdiffusion.trainers import train_val_llapdiff as tv

    meta = {
        "delta_t_y": torch.tensor(
            [[[1.0, 4.0, 5.0, 9.0], [100.0, 300.0, 500.0, 700.0]]]
        )
    }
    mask = torch.tensor([[True, False]])

    dt_b = tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")

    assert torch.allclose(dt_b, torch.tensor([[1.0, 4.0, 5.0, 9.0]]))


def test_target_dt_flatten_ignores_padded_entity_nonfinite_grid():
    from llapdiffusion.trainers import train_val_llapdiff as tv

    meta = {
        "delta_t_y": torch.tensor(
            [[[1.0, 4.0, 5.0, 9.0], [float("nan"), float("inf"), 5.0, 7.0]]]
        )
    }
    mask = torch.tensor([[True, False]])

    dt_b = tv._flatten_dt(meta, mask, torch.device("cpu"), key="delta_t_y")

    assert torch.allclose(dt_b, torch.tensor([[1.0, 4.0, 5.0, 9.0]]))


def test_llapdiff_generate_forwards_context_end_query_offsets_unchanged():
    from llapdiffusion.models.llapdiff import LLapDiff

    model = LLapDiff(data_dim=1, hidden_dim=4, num_layers=1, num_heads=1, laplace_k=2, timesteps=4)
    seen = []

    def fake_forward(x_t, t, **kwargs):
        seen.append(kwargs["dt"].detach().cpu().clone())
        return torch.zeros_like(x_t)

    model.forward = fake_forward
    out = model.generate(
        (1, 3, 1),
        steps=1,
        guidance_strength=1.0,
        dt=torch.tensor([[1.0, 4.0, 5.0]]),
    )

    assert out.shape == (1, 3, 1)
    assert seen
    assert torch.allclose(seen[0], torch.tensor([[1.0, 4.0, 5.0]]))


def test_llapdiff_generate_rejects_invalid_query_dt_shape():
    from llapdiffusion.models.llapdiff import LLapDiff

    model = LLapDiff(data_dim=1, hidden_dim=4, num_layers=1, num_heads=1, laplace_k=2, timesteps=4)
    with pytest.raises(ValueError, match="dt shape"):
        model.generate((1, 3, 1), steps=1, dt=torch.tensor([[1.0, 2.0]]))


def test_llapdiff_generate_rejects_nonfinite_query_dt():
    from llapdiffusion.models.llapdiff import LLapDiff

    model = LLapDiff(data_dim=1, hidden_dim=4, num_layers=1, num_heads=1, laplace_k=2, timesteps=4)
    with pytest.raises(ValueError, match="finite"):
        model.generate((1, 3, 1), steps=1, dt=torch.tensor([[1.0, float("nan"), 3.0]]))


def test_llapdiff_generate_rejects_decreasing_query_dt():
    from llapdiffusion.models.llapdiff import LLapDiff

    model = LLapDiff(data_dim=1, hidden_dim=4, num_layers=1, num_heads=1, laplace_k=2, timesteps=4)
    with pytest.raises(ValueError, match="nondecreasing"):
        model.generate((1, 3, 1), steps=1, dt=torch.tensor([[1.0, 3.0, 2.0]]))


def test_vae_target_mask_excludes_zero_filled_missing_targets():
    from llapdiffusion.trainers import train_val_latent as tvl

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
    assert obs.tolist() == [[[[True], [False], [True]]]]
    assert entity_pad.tolist() == [[False]]
    assert x_tok[0, 1, 0, 0].item() == 0.0
    assert x_tok[0, 1, 0, 1].item() == 0.0

    y_hat = torch.tensor([[[1.0, 100.0, 3.0]]])
    loss, count = tvl._masked_mse(y_hat, y_clean, obs)
    assert count == 2
    assert loss.item() == 0.0


def test_pack_targets_tokens_rejects_bad_y_obs_mask_shape():
    from llapdiffusion.models.llapdiff_utils import pack_targets_tokens

    y = torch.zeros(1, 2, 3)
    entity_mask = torch.tensor([[True, True]])
    bad_mask = torch.ones(1, 4, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="y_obs_mask shape"):
        pack_targets_tokens(y, entity_mask, torch.device("cpu"), y_obs_mask=bad_mask)


def test_collect_latent_means_filters_unobserved_horizons():
    from llapdiffusion.trainers import train_val_latent as tvl

    class FakeVAE(torch.nn.Module):
        def forward(self, x_tok, entity_pad=None):
            B, T, _, _ = x_tok.shape
            mu = torch.arange(B * T * 2, dtype=torch.float32).view(B, T, 2)
            return torch.zeros(B, T, 1, 1), mu, torch.zeros_like(mu)

    y = torch.tensor([[[1.0, 0.0, 3.0]]])
    meta = {
        "entity_mask": torch.tensor([[True]]),
        "y_obs_mask": torch.tensor([[[True, False, True]]]),
    }

    latents = tvl.collect_latent_means(
        [(None, y, meta)],
        FakeVAE(),
        torch.device("cpu"),
    )

    assert latents.shape == (2, 2)
    assert torch.allclose(latents, torch.tensor([[0.0, 1.0], [4.0, 5.0]]))


def test_vae_coverage_balanced_loss_still_ignores_unobserved_targets():
    from llapdiffusion.trainers import train_val_latent as tvl

    y_true = torch.tensor([[[1.0, 0.0, 3.0]]])
    y_hat = torch.tensor([[[1.0, 100.0, 3.0]]])
    obs = torch.tensor([[[True, False, True]]])

    loss, count = tvl._masked_mse(y_hat, y_true, obs, balance_mode="coverage")

    assert count == 2
    assert loss.item() == 0.0


def test_summarizer_prepare_batch_accepts_3d_context_observation_mask():
    from llapdiffusion.trainers import train_val_summarizer as tvs

    V = torch.ones(1, 2, 3, 1)
    T = torch.zeros(1, 2, 3, 1)
    y = torch.zeros(1, 2, 1)
    meta = {
        "entity_mask": torch.tensor([[True, False]]),
        "x_obs_mask": torch.tensor([[[True, False, True], [True, True, True]]]),
    }

    Vp, Tp, mask, elems, dt, obs_mask = tvs._prepare_batch(((V, T), y, meta), torch.device("cpu"))

    assert Vp.shape == (1, 3, 2, 1)
    assert Tp.shape == (1, 3, 2, 1)
    assert mask.tolist() == [[True, False]]
    assert elems == 3.0
    assert dt is None
    assert obs_mask.shape == (1, 3, 2, 1)
    assert obs_mask[:, :, 1, :].sum().item() == 0
    assert obs_mask[0, :, 0, 0].tolist() == [True, False, True]


def test_summarizer_loss_x_respects_context_observation_mask():
    model = LaplaceAE(
        num_entities=1,
        feat_dim=1,
        window_size=2,
        mix_dim=8,
        tv_hidden=8,
        out_len=1,
        context_dim=16,
        enc_layers=1,
        n_heads=2,
        dropout=0.0,
        time2vec_dim=3,
    )
    aux = {
        "x": torch.zeros(1, 2, 1, 1),
        "x_hat": torch.tensor([[[[0.0]], [[100.0]]]]),
        "obs_mask": torch.tensor([[[[True]], [[False]]]]),
        "v_sig": torch.zeros(1, 2, 1),
        "v_hat": torch.zeros(1, 2, 1),
        "t_sig": torch.zeros(1, 2, 1),
        "t_hat": torch.zeros(1, 2, 1),
        "rel_t_unit": torch.zeros(1, 2, 1),
        "dt_hat": torch.zeros(1, 2, 1),
        "obs_frac": torch.zeros(1, 2, 1),
        "obs_hat": torch.zeros(1, 2, 1),
    }

    loss = model.recon_loss(aux, torch.tensor([[True]]), weights=(1.0, 0.0, 0.0, 0.0, 0.0))

    assert loss.item() == 0.0


def test_sanitize_batch_entity_mask_does_not_inspect_future_targets():
    from llapdiffusion.trainers import train_val_llapdiff as tv

    xb = (torch.ones(1, 1, 2, 1), torch.zeros(1, 1, 2, 1))
    yb = torch.tensor([[[float("nan"), float("nan")]]])
    mask = torch.tensor([[True]])

    (_, _), y_clean, clean_mask = tv._sanitize_batch(xb, yb, {"entity_mask": mask}, torch.device("cpu"))

    assert clean_mask.tolist() == [[True]]
    assert torch.isfinite(y_clean).all()


def test_default_config_allows_imputation_but_keeps_aux_inactive():
    from llapdiffusion.configs import config as cfg

    assert bool(getattr(cfg, "IMPUTATION_TRAINING")) is True
    assert float(getattr(cfg, "TARGET_MASK_AUX_P")) == 0.0


def test_target_mask_aux_guard_requires_imputation_training_for_positive_probability():
    from llapdiffusion.trainers import train_val_llapdiff as tv

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
    from llapdiffusion.trainers import train_val_llapdiff as tv

    V = torch.ones(1, 2, 4, 1)
    T = torch.zeros(1, 2, 4, 1)
    mask = torch.tensor([[True, True]])
    dt = torch.tensor([[[0.0, 1.0, 4.0, 5.0], [0.0, 1.0, 4.0, 5.0]]])

    stats = tv._history_stat_tokens(V, T, mask, torch.device("cpu"), dt=dt)

    assert torch.allclose(stats[0, :, 2], torch.tensor([0.0, 0.2, 0.8, 1.0]))


def test_history_stat_tokens_use_valid_entity_denominators():
    from llapdiffusion.trainers import train_val_llapdiff as tv

    V = torch.ones(1, 2, 2, 2)
    T = torch.ones(1, 2, 2, 2)
    mask = torch.tensor([[True, False]])
    obs_mask = torch.tensor(
        [[[[True, False], [True, True]], [[True, True], [True, True]]]]
    )

    stats = tv._history_stat_tokens(V, T, mask, torch.device("cpu"), x_obs_mask=obs_mask)

    assert torch.allclose(stats[0, :, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(stats[0, :, 1], torch.tensor([0.5, 1.0]))


def test_llapdiff_builder_constructs_adapter_for_strict_checkpoint_load():
    from llapdiffusion.trainers import train_val_llapdiff as tv

    cfg = SimpleNamespace(
        VAE_LATENT_CHANNELS=3,
        MODEL_WIDTH=16,
        NUM_LAYERS=1,
        NUM_HEADS=2,
        PREDICT_TYPE="v",
        LAPLACE_K=4,
        TIMESTEPS=8,
        SCHEDULE="cosine",
        DROPOUT=0.0,
        ATTN_DROPOUT=0.0,
        SELF_COND=False,
        COND_POOL_MODE="mean",
        COND_POOL_USE_RAW=False,
        BLOCK_SUMMARY_ADALN=False,
        ANALYSIS_SUMMARY_QK=False,
        ANALYSIS_QK_USE_RAW=False,
        SUM_CONTEXT_DIM=16,
        COND_ADAPTER_MODE="stats",
        COND_ADAPTER_HIDDEN=8,
        COND_ADAPTER_DROPOUT=0.0,
        COND_ADAPTER_SCALE=0.1,
    )

    model = tv.build_llapdiff_model(cfg, torch.device("cpu"))
    loaded = tv.build_llapdiff_model(cfg, torch.device("cpu"))
    tv._load_module_state(loaded, model.state_dict(), strict=True)

    assert hasattr(loaded, "cond_adapter")


def test_forecast_generation_does_not_condition_on_target_values_or_masks():
    from llapdiffusion.trainers import train_val_llapdiff as tv

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
            "delta_t_y": torch.tensor([[[1.0, 4.0, 5.0, 9.0], [1.0, 4.0, 5.0, 9.0]]]),
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
    assert torch.allclose(call["dt"], torch.tensor([[1.0, 4.0, 5.0, 9.0]]))
    assert "y_obs" not in call
    assert "obs_mask" not in call


def test_imputation_generation_only_uses_intentionally_observed_target_tokens(monkeypatch):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    class FakeDiffModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.zeros(kwargs["shape"])

    diff_model = FakeDiffModel()
    mu_norm = torch.arange(12, dtype=torch.float32).view(1, 4, 3)

    monkeypatch.setattr(ce.tv, "_sanitize_batch", lambda xb, yb, meta, device: (xb, yb.to(device), meta["entity_mask"].to(device)))
    monkeypatch.setattr(
        ce.tv,
        "_build_cond_summary_pair",
        lambda *args, **kwargs: (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4)),
    )
    monkeypatch.setattr(ce.tv, "_flatten_dt", lambda *args, **kwargs: torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    monkeypatch.setattr(ce, "pack_targets_tokens", lambda *args, **kwargs: (
        torch.zeros(1, 4, 1, 2),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.ones(1, 4, 1, dtype=torch.bool),
    ))
    monkeypatch.setattr(ce, "encode_mu_norm", lambda *args, **kwargs: mu_norm.clone())
    monkeypatch.setattr(ce, "decode_latents_with_vae", lambda *args, **kwargs: torch.zeros(1, 4, 1, 1))

    xb = (torch.ones(1, 1, 3, 1), torch.zeros(1, 1, 3, 1))
    yb = torch.ones(1, 1, 4)
    meta = {
        "entity_mask": torch.tensor([[True]]),
        "delta_t": torch.zeros(1, 1, 3),
        "delta_t_y": torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        "x_obs_mask": torch.ones(1, 1, 3, 1, dtype=torch.bool),
        "y_obs_mask": torch.ones(1, 1, 4, dtype=torch.bool),
    }

    ce._evaluate_impute_case(
        [(xb, yb, meta)],
        diff_model=diff_model,
        vae=object(),
        summarizer=object(),
        device=torch.device("cpu"),
        mu_mean=torch.zeros(3),
        mu_std=torch.ones(3),
        keep_fn=lambda obs_any: torch.tensor([[True, True, False, False]]),
        num_samples=1,
        steps=2,
    )

    call = diff_model.calls[0]
    expected_keep = torch.tensor([[True, True, False, False]])
    expected_y_obs = mu_norm * expected_keep.unsqueeze(-1).to(dtype=mu_norm.dtype)
    assert torch.equal(call["obs_mask"], expected_keep)
    assert torch.allclose(call["y_obs"], expected_y_obs)
    assert torch.all(call["y_obs"][:, 2:] == 0)


def test_random_imputation_keep_mask_generator_advances_between_batches():
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    obs_any = torch.ones(4, 20, dtype=torch.bool)
    generator = torch.Generator(device=obs_any.device)
    generator.manual_seed(1234)

    first = ce._make_random_keep(obs_any, frac=0.70, generator=generator)
    second = ce._make_random_keep(obs_any, frac=0.70, generator=generator)

    assert not torch.equal(first, second)


def _checkpoint_eval_cfg(num_eval_samples=25):
    return SimpleNamespace(
        DATA_DIR="demo",
        date_batching=True,
        DATES_PER_BATCH=1,
        WINDOW=4,
        PRED=10,
        COVERAGE=0.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        SEED=42,
        DETERMINISTIC=False,
        NUM_EVAL_SAMPLES=num_eval_samples,
        SELF_COND=False,
    )


def _checkpoint_eval_impute_metrics():
    return {
        "hidden_mae": 0.0,
        "hidden_mse": 0.0,
        "hidden_crps": 1.0,
        "observed_mae": 0.0,
        "observed_token_frac": 0.7,
        "hidden_token_frac": 0.3,
    }


def _patch_checkpoint_eval_dependencies(
    monkeypatch,
    ce,
    *,
    test_dl=None,
    forecast_fn=None,
    impute_fn=None,
):
    if test_dl is None:
        test_dl = ["test"]
    monkeypatch.setattr(ce, "set_torch", lambda **kwargs: torch.device("cpu"))
    monkeypatch.setattr(
        ce,
        "resolve_run_experiment",
        lambda data_dir: (
            lambda **kwargs: (["train"], ["val"], test_dl, (1, 1, len(test_dl)))
        ),
    )
    monkeypatch.setattr(
        ce,
        "_load_stack",
        lambda *args, **kwargs: (
            object(),
            object(),
            object(),
            torch.zeros(1),
            torch.ones(1),
        ),
    )
    monkeypatch.setattr(ce.tv, "_sampling_kwargs", lambda cfg, prefix: {
        "steps": 2,
        "guidance_strength": (1.0, 2.0),
        "guidance_power": 1.0,
        "eta": 0.0,
        "dynamic_thresh_p": 0.0,
        "dynamic_thresh_max": 1.0,
        "rho": 7.5,
    })
    monkeypatch.setattr(
        ce.tv,
        "evaluate_regression",
        forecast_fn or (lambda *args, **kwargs: {"crps": 1.0}),
    )
    monkeypatch.setattr(
        ce,
        "_evaluate_impute_case",
        impute_fn or (lambda *args, **kwargs: _checkpoint_eval_impute_metrics()),
    )


def test_checkpoint_eval_cli_parses_sample_and_batch_knobs(monkeypatch):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llapdiff-checkpoint-eval",
            "--dataset-key",
            "crypto",
            "--checkpoint",
            "model.pt",
            "--num-samples",
            "6",
            "--forecast-num-samples",
            "7",
            "--imputation-num-samples",
            "9",
            "--max-eval-batches",
            "0",
        ],
    )

    args = ce._parse_args()

    assert args.num_samples == 6
    assert args.forecast_num_samples == 7
    assert args.imputation_num_samples == 9
    assert args.max_eval_batches == 0


def test_checkpoint_eval_main_prints_compact_summary_by_default(monkeypatch, capsys):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    captured = {}
    cfg = SimpleNamespace()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llapdiff-checkpoint-eval",
            "--dataset-key",
            "crypto",
            "--checkpoint",
            "model.pt",
        ],
    )
    monkeypatch.setattr(ce, "configure_dataset_archive", lambda *args, **kwargs: None)
    monkeypatch.setattr(ce, "default_horizons", lambda dataset_key: (20,))
    monkeypatch.setattr(ce, "build_eval_config", lambda *args, **kwargs: cfg)

    def fake_evaluate_checkpoint(*args, **kwargs):
        captured["verbose"] = kwargs["verbose"]
        return {
            "forecast_test": {"crps": 1.25},
            "balanced_summary": {"avg_hidden_crps": 2.5},
        }

    monkeypatch.setattr(ce, "evaluate_checkpoint", fake_evaluate_checkpoint)

    ce.main()

    out = capsys.readouterr().out
    assert captured["verbose"] is False
    assert cfg.VERBOSE is False
    assert cfg.DEBUG is False
    assert not out.lstrip().startswith("{")
    assert "crypto_pred20: forecast_crps=1.25 avg_hidden_crps=2.5" in out


def test_checkpoint_eval_main_print_json_is_opt_in(monkeypatch, capsys):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    cfg = SimpleNamespace()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llapdiff-checkpoint-eval",
            "--dataset-key",
            "crypto",
            "--checkpoint",
            "model.pt",
            "--print-json",
            "--debug",
        ],
    )
    monkeypatch.setattr(ce, "configure_dataset_archive", lambda *args, **kwargs: None)
    monkeypatch.setattr(ce, "default_horizons", lambda dataset_key: (20,))
    monkeypatch.setattr(ce, "build_eval_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(
        ce,
        "evaluate_checkpoint",
        lambda *args, **kwargs: {
            "forecast_test": {"crps": 1.25},
            "balanced_summary": {"avg_hidden_crps": 2.5},
        },
    )

    ce.main()

    out = capsys.readouterr().out
    assert cfg.VERBOSE is True
    assert cfg.DEBUG is True
    assert out.lstrip().startswith("{")
    assert '"forecast_test"' in out


def test_checkpoint_eval_routes_sample_counts(monkeypatch, tmp_path):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    captured = {"forecast": [], "impute": []}

    def fake_forecast(*args, **kwargs):
        captured["forecast"].append(args[7].NUM_EVAL_SAMPLES)
        return {"crps": 1.0}

    def fake_impute(*args, **kwargs):
        captured["impute"].append(kwargs["num_samples"])
        return _checkpoint_eval_impute_metrics()

    _patch_checkpoint_eval_dependencies(
        monkeypatch,
        ce,
        forecast_fn=fake_forecast,
        impute_fn=fake_impute,
    )
    cfg = _checkpoint_eval_cfg(num_eval_samples=25)

    ce.evaluate_checkpoint(cfg, tmp_path / "model.pt", label="default")
    assert captured == {"forecast": [25], "impute": [25, 25]}

    captured["forecast"].clear()
    captured["impute"].clear()
    ce.evaluate_checkpoint(cfg, tmp_path / "model.pt", label="shared", num_samples=6)
    assert captured == {"forecast": [6], "impute": [6, 6]}

    captured["forecast"].clear()
    captured["impute"].clear()
    ce.evaluate_checkpoint(
        cfg,
        tmp_path / "model.pt",
        label="specific",
        num_samples=6,
        forecast_num_samples=7,
        imputation_num_samples=9,
    )
    assert captured == {"forecast": [7], "impute": [9, 9]}


def test_checkpoint_eval_routes_max_eval_batches(monkeypatch, tmp_path):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    test_batches = ["batch0", "batch1", "batch2", "batch3"]
    captured = {"forecast": [], "impute": []}

    def fake_forecast(*args, **kwargs):
        captured["forecast"].append(list(args[3]))
        return {"crps": 1.0}

    def fake_impute(test_dl, *args, **kwargs):
        captured["impute"].append(list(test_dl))
        return _checkpoint_eval_impute_metrics()

    _patch_checkpoint_eval_dependencies(
        monkeypatch,
        ce,
        test_dl=test_batches,
        forecast_fn=fake_forecast,
        impute_fn=fake_impute,
    )
    cfg = _checkpoint_eval_cfg()

    ce.evaluate_checkpoint(cfg, tmp_path / "model.pt", label="capped", max_eval_batches=2)
    assert captured["forecast"] == [["batch0", "batch1"]]
    assert captured["impute"] == [["batch0", "batch1"], ["batch0", "batch1"]]

    captured["forecast"].clear()
    captured["impute"].clear()
    ce.evaluate_checkpoint(cfg, tmp_path / "model.pt", label="uncapped", max_eval_batches=0)
    assert captured["forecast"] == [test_batches]
    assert captured["impute"] == [test_batches, test_batches]


def test_checkpoint_eval_random_mask30_uses_seventy_percent_keep(monkeypatch, tmp_path):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    captured = {}

    monkeypatch.setattr(ce, "set_torch", lambda **kwargs: torch.device("cpu"))
    monkeypatch.setattr(
        ce,
        "resolve_run_experiment",
        lambda data_dir: (
            lambda **kwargs: (["train"], ["val"], ["test"], (1, 1, 1))
        ),
    )
    monkeypatch.setattr(
        ce,
        "_load_stack",
        lambda *args, **kwargs: (
            object(),
            object(),
            object(),
            torch.zeros(1),
            torch.ones(1),
        ),
    )
    monkeypatch.setattr(ce.tv, "_sampling_kwargs", lambda cfg, prefix: {
        "steps": 2,
        "guidance_strength": (1.0, 2.0),
        "guidance_power": 1.0,
        "eta": 0.0,
        "dynamic_thresh_p": 0.0,
        "dynamic_thresh_max": 1.0,
        "rho": 7.5,
    })
    monkeypatch.setattr(ce.tv, "evaluate_regression", lambda *args, **kwargs: {"crps": 1.0})

    def fake_impute_case(*args, **kwargs):
        keep_fn = kwargs["keep_fn"]
        obs_any = torch.ones(100, 100, dtype=torch.bool)
        keep = keep_fn(obs_any)
        key = "random" if "regular" in captured else "regular"
        captured[key] = keep
        return {
            "hidden_mae": 0.0,
            "hidden_mse": 0.0,
            "hidden_crps": 1.0,
            "observed_mae": 0.0,
            "observed_token_frac": float(keep.sum().item() / obs_any.sum().item()),
            "hidden_token_frac": float((obs_any & (~keep)).sum().item() / obs_any.sum().item()),
        }

    monkeypatch.setattr(ce, "_evaluate_impute_case", fake_impute_case)
    cfg = SimpleNamespace(
        DATA_DIR="demo",
        date_batching=True,
        DATES_PER_BATCH=1,
        WINDOW=4,
        PRED=10,
        COVERAGE=0.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        SEED=42,
        DETERMINISTIC=False,
        NUM_EVAL_SAMPLES=1,
        SELF_COND=False,
    )

    result = ce.evaluate_checkpoint(cfg, tmp_path / "model.pt", label="demo")

    assert "random" in captured
    assert result["random_mask_ratio"] == pytest.approx(0.30)
    assert result["random_mask"]["observed_token_frac"] == pytest.approx(0.70, abs=0.10)
    assert result["random_mask"]["hidden_token_frac"] == pytest.approx(0.30, abs=0.10)
    assert result["random_mask30"]["observed_token_frac"] == pytest.approx(0.70, abs=0.10)
    assert result["random_mask30"]["hidden_token_frac"] == pytest.approx(0.30, abs=0.10)
    assert result["regular_keep25"]["metric_target_type"] == "target_horizon_imputation"
    assert result["random_mask"]["metric_target_type"] == "target_horizon_imputation"
    assert result["random_mask30"]["metric_target_type"] == "target_horizon_imputation"


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
