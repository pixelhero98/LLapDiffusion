from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llapdiffusion.models.laptrans import LaplaceTransformEncoder
from llapdiffusion.trainers import train_val_llapdiff as tv
from llapdiffusion.viz import plot_llapdiff_poles


def _encoder(mode: str) -> LaplaceTransformEncoder:
    return LaplaceTransformEncoder(
        k=4,
        feat_dim=2,
        hidden_dim=8,
        num_heads=2,
        cond_dim=6,
        rho_conditioning_mode=mode,
    )


def _cfg(mode: str = "raw") -> SimpleNamespace:
    return SimpleNamespace(
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
        RHO_CONDITIONING_MODE=mode,
        SUM_CONTEXT_DIM=16,
        COND_ADAPTER_MODE="none",
        COND_ADAPTER_HIDDEN=8,
        COND_ADAPTER_DROPOUT=0.0,
        COND_ADAPTER_SCALE=0.1,
    )


def test_raw_rho_zero_conditioning_preserves_base_poles():
    encoder = _encoder("raw")
    cond = torch.zeros(3, 6)

    base_rho, base_omega = encoder._base_poles(torch.float32, torch.device("cpu"))
    rho, omega = encoder.effective_poles(3, torch.float32, torch.device("cpu"), cond=cond)

    torch.testing.assert_close(rho, base_rho.unsqueeze(0).expand_as(rho))
    torch.testing.assert_close(omega, base_omega.unsqueeze(0).expand_as(omega))


def test_legacy_effective_zero_conditioning_reproduces_old_rho_formula():
    encoder = _encoder("legacy_effective")
    cond = torch.zeros(2, 6)

    base_rho, _ = encoder._base_poles(torch.float32, torch.device("cpu"))
    rho, _ = encoder.effective_poles(2, torch.float32, torch.device("cpu"), cond=cond)

    expected = F.softplus(base_rho.unsqueeze(0).expand_as(rho)) + encoder.alpha_min
    torch.testing.assert_close(rho, expected)
    assert not torch.allclose(rho, base_rho.unsqueeze(0).expand_as(rho))


def test_invalid_rho_conditioning_mode_raises():
    with pytest.raises(ValueError, match="rho_conditioning_mode"):
        _encoder("effective")


def test_model_builder_propagates_raw_rho_conditioning_mode():
    cfg = _cfg("raw")
    kwargs = tv._llapdiff_model_kwargs(cfg)
    model = tv.build_llapdiff_model(cfg, torch.device("cpu"))

    assert kwargs["rho_conditioning_mode"] == "raw"
    assert model.model.analysis.rho_conditioning_mode == "raw"
    assert tv._llapdiff_model_config(cfg)["llapdiff"]["rho_conditioning_mode"] == "raw"


def test_checkpoint_missing_rho_conditioning_mode_defaults_to_legacy():
    cfg = _cfg("raw")
    payload = {
        "model_config": {
            "llapdiff": {
                "data_dim": 3,
                "hidden_dim": 16,
            }
        }
    }

    kwargs = tv._llapdiff_model_kwargs_from_checkpoint(cfg, payload)

    assert kwargs["data_dim"] == 3
    assert kwargs["hidden_dim"] == 16
    assert kwargs["rho_conditioning_mode"] == "legacy_effective"


def test_checkpoint_nested_model_config_preserves_raw_mode():
    cfg = _cfg("legacy_effective")
    payload = {
        "model_config": {
            "llapdiff": {
                "data_dim": 3,
                "rho_conditioning_mode": "raw",
            },
            "cond_adapter": {"mode": "none"},
        }
    }

    kwargs = tv._llapdiff_model_kwargs_from_checkpoint(cfg, payload)
    plot_kwargs = plot_llapdiff_poles._read_model_config(payload)

    assert kwargs["rho_conditioning_mode"] == "raw"
    assert plot_kwargs["rho_conditioning_mode"] == "raw"
    assert "llapdiff" not in plot_kwargs
    assert "cond_adapter" not in plot_kwargs


def test_plot_checkpoint_config_without_mode_defaults_to_legacy():
    payload = {"model_config": {"llapdiff": {"data_dim": 3}}}

    kwargs = plot_llapdiff_poles._read_model_config(payload)

    assert kwargs["data_dim"] == 3
    assert kwargs["rho_conditioning_mode"] == "legacy_effective"
