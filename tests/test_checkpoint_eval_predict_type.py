from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("torch") is None:
    torch = None
    pytestmark = pytest.mark.skip(reason="torch is not installed")
else:
    import torch


def _base_eval_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        DATA_DIR="demo",
        date_batching=True,
        BATCH_SIZE=1,
        DATES_PER_BATCH=1,
        WINDOW=4,
        PRED=10,
        COVERAGE=0.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        split_policy="global_purged_horizon",
        split_scope="global_target_time",
        exact_timestamp_batches=True,
        SEED=42,
        DETERMINISTIC=False,
        NUM_EVAL_SAMPLES=1,
        SELF_COND=False,
        VAE_DIR="vae",
        VAE_LATENT_CHANNELS=4,
        VAE_ENTITY_CONDITION=False,
        VAE_CKPT="vae/pred-10_ch-4_elbo.pt",
    )


def test_checkpoint_metadata_predict_type_wins():
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    payload = {"model_config": {"llapdiff": {"predict_type": "V"}}}

    assert ce._resolve_checkpoint_predict_type(payload, explicit_predict_type=None) == (
        "v",
        "checkpoint_metadata",
    )


def test_legacy_checkpoint_requires_explicit_predict_type():
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    with pytest.raises(ValueError, match="legacy checkpoint"):
        ce._resolve_checkpoint_predict_type({"model": {}}, explicit_predict_type=None)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        ce._resolve_checkpoint_predict_type({"model": {}}, explicit_predict_type="")


def test_legacy_checkpoint_uses_explicit_predict_type():
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    assert ce._resolve_checkpoint_predict_type({"model": {}}, explicit_predict_type="eps") == (
        "eps",
        "cli",
    )


def test_explicit_predict_type_must_match_checkpoint_metadata():
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    payload = {"model_config": {"llapdiff": {"predict_type": "x0"}}}
    with pytest.raises(ValueError, match="does not match checkpoint metadata"):
        ce._resolve_checkpoint_predict_type(payload, explicit_predict_type="v")


def test_evaluation_result_records_resolved_predict_type(monkeypatch):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    cfg = _base_eval_cfg()
    payload = {"model_config": {"llapdiff": {"predict_type": "eps"}}}

    def fake_load_stack(cfg, ckpt_path, device, train_dl, *, predict_type=None, **kwargs):
        assert kwargs["checkpoint_payload"] is payload
        resolved, source = ce._resolve_checkpoint_predict_type(
            payload,
            explicit_predict_type=predict_type,
        )
        cfg.PREDICT_TYPE = resolved
        cfg.PREDICT_TYPE_SOURCE = source
        return object(), object(), object(), torch.zeros(1), torch.ones(1)

    monkeypatch.setattr(ce, "set_torch", lambda **kwargs: torch.device("cpu"))
    monkeypatch.setattr(
        ce,
        "resolve_run_experiment",
        lambda data_dir: (lambda **kwargs: (["train"], ["val"], ["test"], (1, 1, 1))),
    )
    monkeypatch.setattr(ce.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(ce, "_load_stack", fake_load_stack)
    monkeypatch.setattr(
        ce.tv,
        "evaluate_regression",
        lambda *args, **kwargs: {"crps": 1.0, "mae": 0.0, "mse": 0.0},
    )
    monkeypatch.setattr(
        ce,
        "_evaluate_impute_case",
        lambda *args, **kwargs: {
            "hidden_mae": 0.0,
            "hidden_mse": 0.0,
            "hidden_crps": 1.0,
            "observed_mae": 0.0,
            "observed_token_frac": 0.7,
            "hidden_token_frac": 0.3,
        },
    )

    result = ce.evaluate_checkpoint(cfg, "checkpoint.pt", label="demo")

    assert result["predict_type"] == "eps"
    assert result["predict_type_source"] == "checkpoint_metadata"
    assert result["checkpoint_target_metadata_applied"] is False


def test_checkpoint_target_metadata_feeds_loader_when_target_unrequested(monkeypatch):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    cfg = _base_eval_cfg()
    payload = {
        "model_config": {"llapdiff": {"predict_type": "x0"}},
        "target_metadata": {
            "target_col": "RET_CLOSE",
            "target_cols": ["RET_CLOSE", "RVOL20_CLOSE"],
            "target_indices": [0, 2],
            "target_dim": 2,
            "target_source": "feature_columns",
            "requested_target_col": None,
            "requested_target_cols": ["RET_CLOSE", "RVOL20_CLOSE"],
            "calendar_feature_cols": ["DAY_OF_WEEK"],
        },
    }
    loader_kwargs = {}

    def fake_run_experiment(**kwargs):
        loader_kwargs.update(kwargs)
        return ["train"], ["val"], ["test"], (1, 1, 1)

    monkeypatch.setattr(ce.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(ce, "set_torch", lambda **kwargs: torch.device("cpu"))
    monkeypatch.setattr(ce, "resolve_run_experiment", lambda data_dir: fake_run_experiment)
    monkeypatch.setattr(ce, "_target_policy", lambda cfg: payload["target_metadata"])
    monkeypatch.setattr(
        ce,
        "_load_stack",
        lambda *args, **kwargs: (object(), object(), object(), torch.zeros(1), torch.ones(1)),
    )
    monkeypatch.setattr(
        ce.tv,
        "evaluate_regression",
        lambda *args, **kwargs: {"crps": 1.0, "mae": 0.0, "mse": 0.0},
    )
    monkeypatch.setattr(
        ce,
        "_evaluate_impute_case",
        lambda *args, **kwargs: {
            "hidden_mae": 0.0,
            "hidden_mse": 0.0,
            "hidden_crps": 1.0,
            "observed_mae": 0.0,
            "observed_token_frac": 0.7,
            "hidden_token_frac": 0.3,
        },
    )

    result = ce.evaluate_checkpoint(cfg, "checkpoint.pt", label="demo")

    assert loader_kwargs["target_col"] is None
    assert loader_kwargs["target_cols"] == ["RET_CLOSE", "RVOL20_CLOSE"]
    vae_name = Path(cfg.VAE_CKPT).name
    assert vae_name.startswith("pred-10_ch-4_tdim-2_targets-ret-close-rvol20-close-")
    assert vae_name.endswith("_elbo.pt")
    assert result["predict_type"] == "x0"
    assert result["checkpoint_target_metadata_applied"] is True
