import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_progress_iter_disabled_emits_nothing():
    from llapdiffusion.logging_utils import progress_iter

    stream = io.StringIO()
    assert list(progress_iter([1, 2], desc="demo", enabled=False, stream=stream)) == [1, 2]
    assert stream.getvalue() == ""


def test_progress_iter_noninteractive_uses_stderr_lines():
    from llapdiffusion.logging_utils import progress_iter

    stream = io.StringIO()
    assert list(progress_iter([1, 2], desc="demo", enabled=True, stream=stream)) == [1, 2]
    text = stream.getvalue()
    assert "[progress] demo:" in text
    assert "2/2 batch" in text


def test_progress_task_noninteractive_updates_on_stderr():
    from llapdiffusion.logging_utils import progress_task

    stream = io.StringIO()
    with progress_task(desc="manual", enabled=True, total=4, unit="step", stream=stream) as progress:
        progress.update(2)
        progress.update(2)

    text = stream.getvalue()
    assert "[progress] manual:" in text
    assert "4/4 step" in text


def test_progress_iter_interactive_uses_tqdm_stream():
    from llapdiffusion.logging_utils import progress_iter

    class TtyStream(io.StringIO):
        def isatty(self):
            return True

    stream = TtyStream()
    assert list(progress_iter([1], desc="tty-demo", enabled=True, stream=stream)) == [1]
    assert "tty-demo" in stream.getvalue()


def _checkpoint_eval_cfg(num_eval_samples=25):
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
    forecast_fn=None,
    impute_fn=None,
    checkpoint_payload=None,
):
    if checkpoint_payload is None:
        checkpoint_payload = {"model_config": {"llapdiff": {"predict_type": "x0"}}}
    monkeypatch.setattr(ce.torch, "load", lambda *args, **kwargs: checkpoint_payload)
    monkeypatch.setattr(ce, "set_torch", lambda **kwargs: torch.device("cpu"))
    monkeypatch.setattr(
        ce,
        "resolve_run_experiment",
        lambda data_dir: (lambda **kwargs: (["train"], ["val"], ["test"], (1, 1, 1))),
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
    monkeypatch.setattr(
        ce.tv,
        "evaluate_regression",
        forecast_fn or (lambda *args, **kwargs: {"crps": 1.0, "mae": 0.0, "mse": 0.0}),
    )
    monkeypatch.setattr(
        ce,
        "_evaluate_impute_case",
        impute_fn or (lambda *args, **kwargs: _checkpoint_eval_impute_metrics()),
    )


def test_checkpoint_eval_routes_verbose_progress_labels(monkeypatch):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    calls = []

    def forecast_fn(*args, **kwargs):
        calls.append(("forecast", kwargs["progress_enabled"], kwargs["progress_label"]))
        return {"crps": 1.0, "mae": 0.0, "mse": 0.0}

    def impute_fn(*args, **kwargs):
        calls.append(("impute", kwargs["progress_enabled"], kwargs["progress_label"]))
        return _checkpoint_eval_impute_metrics()

    _patch_checkpoint_eval_dependencies(
        monkeypatch,
        ce,
        forecast_fn=forecast_fn,
        impute_fn=impute_fn,
    )

    ce.evaluate_checkpoint(_checkpoint_eval_cfg(), "checkpoint.pt", label="demo", verbose=True)

    assert calls == [
        ("forecast", True, "checkpoint-eval forecast_test"),
        ("impute", True, "checkpoint-eval regular_keep25"),
        ("impute", True, "checkpoint-eval random_mask"),
    ]


def test_checkpoint_eval_print_json_verbose_keeps_progress_on_stderr(monkeypatch, capsys):
    from llapdiffusion.tools import llapdiff_checkpoint_eval as ce

    def fake_evaluate_checkpoint(*args, **kwargs):
        print("[progress] checkpoint-eval forecast_test: 1/1 step", file=sys.stderr)
        return {
            "label": "demo",
            "forecast_test": {"crps": 1.0},
            "balanced_summary": {"avg_hidden_crps": 2.0},
        }

    monkeypatch.setattr(ce, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(ce, "configure_dataset_archive", lambda *args, **kwargs: None)
    monkeypatch.setattr(ce, "build_eval_config", lambda *args, **kwargs: _checkpoint_eval_cfg())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llapdiff-checkpoint-eval",
            "--dataset-key",
            "bms_air",
            "--pred",
            "10",
            "--checkpoint",
            "checkpoint.pt",
            "--label",
            "demo",
            "--verbose",
            "--print-json",
        ],
    )

    ce.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["label"] == "demo"
    assert "[progress] checkpoint-eval forecast_test" in captured.err


def test_artifact_trainers_wrap_verbose_loader_progress(monkeypatch):
    from llapdiffusion.trainers import train_val_latent as tvl
    from llapdiffusion.trainers import train_val_summarizer as tvs

    labels = []

    def fake_progress_iter(iterable, *, desc, enabled=False, **kwargs):
        labels.append((desc, enabled))
        return iter(iterable)

    monkeypatch.setattr(tvl, "progress_iter", fake_progress_iter)
    monkeypatch.setattr(tvs, "progress_iter", fake_progress_iter)

    tvl._epoch_pass(
        [],
        model=object(),
        device=torch.device("cpu"),
        beta=0.0,
        progress_enabled=True,
        progress_label="vae train e001/001",
    )
    with pytest.raises(RuntimeError, match="processed no valid elements"):
        tvs._run_epoch(
            [],
            model=object(),
            device=torch.device("cpu"),
            loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
            progress_enabled=True,
            progress_label="summ train e001/001",
        )

    assert labels == [
        ("vae train e001/001", True),
        ("summ train e001/001", True),
    ]


def test_llapdiff_val_diag_and_cache_wrap_verbose_progress(monkeypatch, tmp_path):
    from llapdiffusion import diffusion_cache as dc
    from llapdiffusion.trainers import train_val_llapdiff as tv

    labels = []

    def fake_progress_iter(iterable, *, desc, enabled=False, **kwargs):
        labels.append((desc, enabled))
        return iter(iterable)

    monkeypatch.setattr(tv, "progress_iter", fake_progress_iter)
    monkeypatch.setattr(dc, "progress_iter", fake_progress_iter)

    class FakeDiffModel:
        scheduler = object()

        def eval(self):
            return None

    with pytest.raises(RuntimeError, match="Validation diagnostic found no valid diffusion samples"):
        tv.evaluate_val_diagnostics(
            FakeDiffModel(),
            vae=object(),
            summarizer=object(),
            dataloader=[],
            device=torch.device("cpu"),
            mu_mean=torch.zeros(1),
            mu_std=torch.ones(1),
            config=SimpleNamespace(PREDICT_TYPE="x0"),
            progress_enabled=True,
            progress_label="llapdiff val-diag e001/001",
        )

    class FakeModule:
        def eval(self):
            return None

    with pytest.raises(RuntimeError, match="produced no arrays"):
        dc._write_split(
            name="train",
            dataloader=[],
            plan=dc._SplitPlan(name="train", batch_rows=[], batch_digests=[]),
            root=tmp_path,
            vae=FakeModule(),
            summarizer=FakeModule(),
            device=torch.device("cpu"),
            latent_dtype=np.dtype("float32"),
            summary_dtype=np.dtype("float16"),
            summary_enabled=False,
            stats_mode="global",
            collect_stats=False,
            progress_enabled=True,
        )

    assert labels == [
        ("llapdiff val-diag e001/001", True),
        ("diffusion-cache train", True),
    ]
