from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from llapdiffusion.configs.config_utils import normalize_predict_type


def test_normalize_predict_type_canonicalizes_supported_names():
    assert normalize_predict_type(None) == "v"
    assert normalize_predict_type("V") == "v"
    assert normalize_predict_type("x_0") == "x0"
    assert normalize_predict_type("epsilon") == "eps"

    with pytest.raises(ValueError, match="Unknown predict_type"):
        normalize_predict_type("score")


def test_project_script_uses_success_returning_train_wrapper():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["llapdiff-train"] == "llapdiffusion.pipeline:cli_main"


def test_direct_predict_type_defaults_match_documented_v():
    pytest.importorskip("torch")
    from llapdiffusion.models.llapdiff import LLapDiff
    from llapdiffusion.models.llapdiff_utils import diffusion_loss

    model = LLapDiff(
        data_dim=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        laplace_k=4,
        timesteps=4,
    )

    assert model.predict_type == "v"
    assert diffusion_loss.__kwdefaults__["predict_type"] == "v"


def test_diffusion_loss_x0_targets_latent_tensor():
    torch = pytest.importorskip("torch")
    from llapdiffusion.models.llapdiff_utils import NoiseScheduler, diffusion_loss

    x0_lat_norm = torch.tensor(
        [[[0.25, -0.5], [1.0, 0.75]], [[-1.0, 0.5], [0.125, -0.25]]],
        dtype=torch.float32,
    )
    t = torch.tensor([1, 2], dtype=torch.long)
    x_t = torch.zeros_like(x0_lat_norm)
    eps_true = torch.ones_like(x0_lat_norm) * 3.0
    scheduler = NoiseScheduler(timesteps=4, schedule="linear")

    class X0Model(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return x0_lat_norm

    loss, stats = diffusion_loss(
        X0Model(),
        scheduler,
        x0_lat_norm,
        t,
        cond_summary=None,
        predict_type="x0",
        reuse_xt_eps=(x_t, eps_true),
        return_stats=True,
    )

    assert loss.item() == pytest.approx(0.0)
    assert stats["raw_loss"].item() == pytest.approx(0.0)


def test_update_config_for_pred_preserves_explicit_predict_type_reset(monkeypatch, tmp_path):
    from llapdiffusion import pipeline

    calls = []

    def fake_apply(config, dataset_key, pred):
        calls.append((dataset_key, pred))
        config.DATASET_KEY = dataset_key
        config.PRED = pred
        config.PREDICT_TYPE = "v"
        config.SUM_DIR = str(tmp_path / "summarizer")
        config.VAE_LATENT_CHANNELS = 12

    cfg = SimpleNamespace(
        DATASET_KEY="crypto",
        PREDICT_TYPE="v",
        REQUESTED_PREDICT_TYPE_ARG="x0",
        split_policy="global_purged_horizon",
        split_scope="global_target_time",
        exact_timestamp_batches=True,
        TARGET_COL=None,
        TARGET_COLS=None,
    )

    monkeypatch.setattr(pipeline, "apply_dataset_preset", fake_apply)

    pipeline._update_config_for_pred(20, config=cfg)

    assert calls == [("crypto", 20)]
    assert cfg.PREDICT_TYPE == "x0"
    assert cfg.REQUESTED_PREDICT_TYPE_ARG == "x0"
    assert cfg.SUM_CKPT == str(tmp_path / "summarizer" / "20-12-summarizer.pt")


def test_run_preds_routes_nondefault_mode_before_base_dirs(monkeypatch, tmp_path):
    from llapdiffusion import pipeline

    calls = []
    cfg = SimpleNamespace(
        PREDICT_TYPE="x0",
        OUT_DIR=str(tmp_path / "out"),
        CKPT_DIR=str(tmp_path / "ckpt"),
        POLE_PLOT_DIR=str(tmp_path / "out" / "pole_plots"),
    )

    def fake_run_single_pred(pred, **kwargs):
        calls.append(
            {
                "pred": pred,
                "base_out_dir": kwargs["base_out_dir"],
                "base_ckpt_dir": kwargs["base_ckpt_dir"],
                "predict_type": kwargs["config"].PREDICT_TYPE,
            }
        )
        return {"eval_stats": {}}

    monkeypatch.setattr(pipeline, "run_single_pred", fake_run_single_pred)

    result = pipeline.run_preds([5, 20], config=cfg)

    assert result == {5: {"eval_stats": {}}, 20: {"eval_stats": {}}}
    assert [call["pred"] for call in calls] == [5, 20]
    assert {call["base_out_dir"] for call in calls} == {tmp_path / "out" / "predict-x0"}
    assert {call["base_ckpt_dir"] for call in calls} == {tmp_path / "ckpt" / "predict-x0"}
    assert {call["predict_type"] for call in calls} == {"x0"}
    assert cfg.OUT_DIR == str(tmp_path / "out" / "predict-x0")
    assert cfg.CKPT_DIR == str(tmp_path / "ckpt" / "predict-x0")
    assert cfg.POLE_PLOT_DIR == str(tmp_path / "out" / "predict-x0" / "pole_plots")


def test_run_preds_keeps_default_v_pred_base_dirs(monkeypatch, tmp_path):
    from llapdiffusion import pipeline

    calls = []
    cfg = SimpleNamespace(
        PREDICT_TYPE="v",
        OUT_DIR=str(tmp_path / "out"),
        CKPT_DIR=str(tmp_path / "ckpt"),
        POLE_PLOT_DIR=str(tmp_path / "out" / "pole_plots"),
    )

    def fake_run_single_pred(pred, **kwargs):
        calls.append((kwargs["base_out_dir"], kwargs["base_ckpt_dir"]))
        return {"eval_stats": {}}

    monkeypatch.setattr(pipeline, "run_single_pred", fake_run_single_pred)

    assert pipeline.run_preds([5], config=cfg) == {5: {"eval_stats": {}}}
    assert calls == [(tmp_path / "out", tmp_path / "ckpt")]
    assert cfg.OUT_DIR == str(tmp_path / "out")
    assert cfg.CKPT_DIR == str(tmp_path / "ckpt")
    assert cfg.POLE_PLOT_DIR == str(tmp_path / "out" / "pole_plots")


def test_llapdiff_train_predict_type_routes_cli_base_dirs(monkeypatch, tmp_path):
    from llapdiffusion import pipeline

    cfg = SimpleNamespace()
    calls = {}

    monkeypatch.setattr(pipeline, "config", cfg)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llapdiff-train",
            "--dataset-key",
            "crypto",
            "--preds",
            "100",
            "--predict-type",
            "eps",
        ],
    )
    monkeypatch.setattr(pipeline, "configure_dataset_archive", lambda *args, **kwargs: None)

    def fake_apply(config, dataset_key, pred=None):
        config.DATASET_KEY = dataset_key
        config.PIPELINE_PREDS = (100,)
        config.PRED = pred
        config.OUT_DIR = str(tmp_path / "out")
        config.CKPT_DIR = str(tmp_path / "ckpt")
        config.POLE_PLOT_DIR = str(tmp_path / "out" / "pole_plots")
        config.PREDICT_TYPE = "v"

    def fake_run_single_pred(pred, **kwargs):
        calls["pred"] = pred
        calls["predict_type"] = kwargs["config"].PREDICT_TYPE
        calls["requested_predict_type"] = kwargs["config"].REQUESTED_PREDICT_TYPE_ARG
        calls["base_out_dir"] = kwargs["base_out_dir"]
        calls["base_ckpt_dir"] = kwargs["base_ckpt_dir"]
        return {"eval_stats": {}}

    monkeypatch.setattr(pipeline, "apply_dataset_preset", fake_apply)
    monkeypatch.setattr(pipeline, "run_single_pred", fake_run_single_pred)
    monkeypatch.setattr(pipeline, "_print_summary_table", lambda results: None)

    assert pipeline.main() == {100: {"eval_stats": {}}}
    assert calls == {
        "pred": 100,
        "predict_type": "eps",
        "requested_predict_type": "eps",
        "base_out_dir": tmp_path / "out" / "predict-eps",
        "base_ckpt_dir": tmp_path / "ckpt" / "predict-eps",
    }


def test_llapdiff_train_console_entrypoint_returns_success(monkeypatch):
    from llapdiffusion import pipeline

    called = {}

    def fake_main():
        called["main"] = True
        return {100: {"status": "ok"}}

    monkeypatch.setattr(pipeline, "main", fake_main)

    assert pipeline.cli_main() is None
    assert called == {"main": True}


def test_predict_type_dir_and_target_suffix_compose(monkeypatch, tmp_path):
    from llapdiffusion import pipeline

    policy = {
        "target_col": "RET_CLOSE",
        "target_cols": ["RET_CLOSE", "RVOL20_CLOSE"],
        "target_indices": [0, 1],
        "target_dim": 2,
        "target_source": "feature_columns",
        "requested_target_col": None,
        "requested_target_cols": ["RET_CLOSE", "RVOL20_CLOSE"],
    }
    cfg = SimpleNamespace(
        VAE_DIR=str(tmp_path / "vae"),
        PRED=20,
        VAE_LATENT_CHANNELS=12,
        VAE_ENTITY_CONDITION=True,
        VAE_CKPT=str(tmp_path / "vae" / "pred-20_ch-12_entity_elbo.pt"),
        OUT_DIR=str(tmp_path / "out" / "predict-x0" / "pred-20"),
        CKPT_DIR=str(tmp_path / "ckpt" / "predict-x0" / "pred-20"),
        POLE_PLOT_DIR=str(tmp_path / "out" / "predict-x0" / "pred-20" / "pole_plots"),
    )

    monkeypatch.setattr(pipeline, "_target_policy", lambda config: policy)

    pipeline._sync_target_shape_config(config=cfg)

    out_dir = Path(cfg.OUT_DIR)
    ckpt_dir = Path(cfg.CKPT_DIR)
    assert out_dir.parent.name == "predict-x0"
    assert ckpt_dir.parent.name == "predict-x0"
    assert out_dir.name.startswith("pred-20_tdim-2_targets-ret-close-rvol20-close-")
    assert ckpt_dir.name.startswith("pred-20_tdim-2_targets-ret-close-rvol20-close-")
    assert Path(cfg.POLE_PLOT_DIR) == out_dir / "pole_plots"
    assert cfg.TARGET_ARTIFACT_SUFFIX in Path(cfg.VAE_CKPT).name
