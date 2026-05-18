from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llapdiffusion.baselines.data import select_stable_entities, target_index
from llapdiffusion.baselines.metrics import masked_mae, masked_mse, sample_crps
from llapdiffusion.baselines.registry import BASELINES, DATASET_KEYS, EXTRAPOLATION_BASELINES, IMPUTATION_BASELINES
from llapdiffusion.baselines.sources import SourceManager, prepend_paths


def test_baseline_registry_records_public_contracts():
    assert set(EXTRAPOLATION_BASELINES) == {
        "dlinear",
        "neuralcde",
        "patchtst",
        "timegrad",
        "mtan",
        "mr-diff",
        "t_patchgnn",
        "contiformer",
    }
    assert IMPUTATION_BASELINES == ("csdi",)
    assert len(DATASET_KEYS) == 7
    assert BASELINES["dlinear"].source_sha == "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
    assert BASELINES["csdi"].placement == "imputation/csdi"
    assert "context-window imputation" in BASELINES["csdi"].dependency_caveat
    assert BASELINES["mtan"].probabilistic is True
    assert BASELINES["contiformer"].dependency_sources == (("physiopro", "5486d1ccaff8f33d635753e3debd7465234b09f1"),)


def _require_mr_diff_adapter():
    module = pytest.importorskip("llapdiffusion.baselines.adapters.mr_diff")
    return module.MRDiffAdapter


def _sample_baseline_batch(batch_size: int = 2, entities: int = 3, context: int = 5, horizon: int = 4):
    values = torch.randn(batch_size, entities, context, 1)
    times = torch.zeros_like(values)
    target = torch.randn(batch_size, entities, horizon)
    meta = {
        "x_obs_mask": torch.ones(batch_size, entities, context, 1, dtype=torch.bool),
        "y_obs_mask": torch.ones(batch_size, entities, horizon, dtype=torch.bool),
        "entity_mask": torch.ones(batch_size, entities, dtype=torch.bool),
        "delta_t": torch.arange(context, dtype=torch.float32).view(1, 1, context).expand(batch_size, entities, -1),
        "delta_t_y": torch.arange(1, horizon + 1, dtype=torch.float32).view(1, 1, horizon).expand(batch_size, entities, -1),
    }
    return (values, times), target, meta


def _sample_dataset_info(context: int = 5, horizon: int = 4):
    return {
        "dataset": "demo",
        "window": context,
        "horizon": horizon,
        "feature_cols": ["target"],
        "target_col": "target",
    }


def test_mr_diff_registry_declares_first_party_contract():
    _require_mr_diff_adapter()
    assert "mr-diff" in EXTRAPOLATION_BASELINES
    assert "mr-diff" in BASELINES
    spec = BASELINES["mr-diff"]
    assert spec.placement == "extrapolation/mr-diff"
    assert spec.metric_type == "probabilistic_crps_mse"
    assert spec.source_name == "LLapDiffusion"
    assert spec.source_sha == "first-party-paper-derived"
    assert "ICLR 2024" in spec.official_reference
    assert spec.probabilistic is True
    assert spec.first_party is True
    assert spec.dependency_sources == ()
    assert "first-party" in spec.dependency_caveat


def test_source_manager_allows_first_party_without_external_root():
    source = SourceManager(None).validate(BASELINES["mr-diff"])
    assert source["source_name"] == "LLapDiffusion"
    assert source["source_sha"] == "first-party-paper-derived"
    assert source["source_clean"] is True
    with pytest.raises(ValueError, match="baseline-source-root"):
        SourceManager(None).validate(BASELINES["dlinear"])


def test_source_manager_defers_external_root_resolution_for_first_party(monkeypatch, tmp_path):
    missing_root = tmp_path / "missing"
    monkeypatch.setenv("LLAPDIFF_BASELINE_SOURCE_ROOT", str(missing_root))

    source = SourceManager(None).validate(BASELINES["mr-diff"])

    assert source["source_name"] == "LLapDiffusion"
    with pytest.raises(FileNotFoundError, match="Baseline source root does not exist"):
        SourceManager(None).validate(BASELINES["dlinear"])


def test_build_adapter_dispatches_mr_diff(monkeypatch):
    _require_mr_diff_adapter()
    from llapdiffusion.baselines.adapters import builder

    seen = {}

    class FakeMRDiffAdapter(torch.nn.Module):
        def __init__(self, dataset_info, sample_batch, *, num_samples=4):
            super().__init__()
            seen["dataset_info"] = dataset_info
            seen["sample_batch"] = sample_batch
            seen["num_samples"] = num_samples

    monkeypatch.setattr(builder, "MRDiffAdapter", FakeMRDiffAdapter)
    dataset_info = _sample_dataset_info()
    sample_batch = _sample_baseline_batch()
    adapter = builder.build_adapter(
        "mr-diff",
        dataset_info,
        sample_batch,
        SourceManager(None),
        torch.device("cpu"),
        num_samples=7,
    )
    assert isinstance(adapter, FakeMRDiffAdapter)
    assert seen == {
        "dataset_info": dataset_info,
        "sample_batch": sample_batch,
        "num_samples": 7,
    }


def test_build_adapter_passes_csdi_imputation_mask_ratio(monkeypatch):
    from llapdiffusion.baselines.adapters import builder

    seen = {}

    class FakeCSDIAdapter(torch.nn.Module):
        def __init__(
            self,
            dataset_info,
            sample_batch,
            source_manager,
            device,
            *,
            num_samples=4,
            imputation_random_mask_ratio=0.30,
        ):
            super().__init__()
            seen["num_samples"] = num_samples
            seen["imputation_random_mask_ratio"] = imputation_random_mask_ratio

    monkeypatch.setattr(builder, "CSDIAdapter", FakeCSDIAdapter)
    adapter = builder.build_adapter(
        "csdi",
        _sample_dataset_info(),
        _sample_baseline_batch(),
        SourceManager(None),
        torch.device("cpu"),
        num_samples=5,
        imputation_random_mask_ratio=0.45,
    )

    assert isinstance(adapter, FakeCSDIAdapter)
    assert seen == {"num_samples": 5, "imputation_random_mask_ratio": 0.45}


def test_mr_diff_adapter_loss_and_samples_shape():
    MRDiffAdapter = _require_mr_diff_adapter()
    dataset_info = _sample_dataset_info()
    sample_batch = _sample_baseline_batch()
    adapter = MRDiffAdapter(dataset_info, sample_batch, num_samples=3, stages=2, kernels=(3,), width=8, diffusion_steps=2)
    loss, samples = adapter.loss_and_samples(sample_batch, dataset_info)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert samples.shape == (3, 2, 3, 4)
    assert torch.isfinite(samples).all()


def test_baseline_loader_uses_longest_supported_horizon(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    data_dir = tmp_path / "demo"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        '{"assets": ["a"], "feature_cols": ["target"], "target_col": "target"}',
        encoding="utf-8",
    )
    preset = SimpleNamespace(
        data_dir=data_dir,
        horizons=(4, 8, 12),
        context_length=24,
        table_batch_size=3,
    )
    seen = {}

    def fake_run_experiment(data_dir, K, H, ratios, per_asset, date_batching, coverage, dates_per_batch, batch_size, norm, reindex):
        kwargs = {
            "data_dir": data_dir,
            "K": K,
            "H": H,
            "ratios": ratios,
            "per_asset": per_asset,
            "date_batching": date_batching,
            "coverage": coverage,
            "dates_per_batch": dates_per_batch,
            "batch_size": batch_size,
            "norm": norm,
            "reindex": reindex,
        }
        seen.update(kwargs)
        return ["train"], ["val"], ["test"], (1, 2, 3)

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)
    monkeypatch.setattr(baseline_data, "resolve_run_experiment", lambda path: fake_run_experiment)

    loaders, info = baseline_data.load_dataset_loaders(
        "demo",
        allow_cache_copy=False,
        work_cache_dir=None,
    )

    assert loaders == (["train"], ["val"], ["test"])
    assert seen["K"] == 24
    assert seen["H"] == 12
    assert info["horizon"] == 12
    assert info["window"] == 24


def test_baseline_loader_accepts_supported_explicit_horizon(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    data_dir = tmp_path / "demo"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        '{"assets": ["a"], "feature_cols": ["target"], "target_col": "target"}',
        encoding="utf-8",
    )
    preset = SimpleNamespace(
        data_dir=data_dir,
        horizons=(4, 8, 12),
        context_length=24,
        table_batch_size=3,
    )
    seen = {}

    def fake_run_experiment(data_dir, K, H, ratios, per_asset, date_batching, coverage, dates_per_batch, batch_size, norm, reindex):
        kwargs = {
            "data_dir": data_dir,
            "K": K,
            "H": H,
            "ratios": ratios,
            "per_asset": per_asset,
            "date_batching": date_batching,
            "coverage": coverage,
            "dates_per_batch": dates_per_batch,
            "batch_size": batch_size,
            "norm": norm,
            "reindex": reindex,
        }
        seen.update(kwargs)
        return ["train"], ["val"], ["test"], (1, 2, 3)

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)
    monkeypatch.setattr(baseline_data, "resolve_run_experiment", lambda path: fake_run_experiment)

    loaders, info = baseline_data.load_dataset_loaders(
        "demo",
        horizon=8,
        allow_cache_copy=False,
        work_cache_dir=None,
    )

    assert loaders == (["train"], ["val"], ["test"])
    assert seen["H"] == 8
    assert info["horizon"] == 8


def test_baseline_loader_rejects_unsupported_horizon(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    preset = SimpleNamespace(
        data_dir=tmp_path / "demo",
        horizons=(4, 8, 12),
        context_length=24,
        table_batch_size=3,
    )
    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)

    with pytest.raises(ValueError, match="horizon=5"):
        baseline_data.load_dataset_loaders("demo", horizon=5, allow_cache_copy=False, work_cache_dir=None)


def test_baseline_loader_validates_noaa_us_long_horizon_cache(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    data_dir = tmp_path / "noaa_us"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text('{"horizon": 24}', encoding="utf-8")
    preset = SimpleNamespace(
        data_dir=data_dir,
        horizons=(24, 48, 96, 168),
        context_length=336,
        table_batch_size=2,
    )

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)

    with pytest.raises(RuntimeError, match="allow-cache-copy"):
        baseline_data.load_dataset_loaders(
            "noaa_us",
            allow_cache_copy=False,
            work_cache_dir=None,
        )


def test_target_index_fails_if_target_column_is_missing():
    with pytest.raises(ValueError, match="target_col"):
        target_index({"dataset": "demo", "feature_cols": ["x"], "target_col": "y"})


def test_csdi_random_gt_mask_holds_out_configured_hidden_fraction():
    from llapdiffusion.baselines.adapters.csdi import CSDIAdapter

    observed = torch.ones(2, 5, 10)
    adapter = CSDIAdapter.__new__(CSDIAdapter)
    adapter.imputation_random_mask_ratio = 0.30

    torch.manual_seed(7)
    gt_mask = adapter._random_gt_mask(observed)

    assert gt_mask.shape == observed.shape
    assert torch.equal(gt_mask <= observed, torch.ones_like(observed, dtype=torch.bool))
    assert gt_mask.sum(dim=(1, 2)).tolist() == [35.0, 35.0]
    assert (observed - gt_mask).sum(dim=(1, 2)).tolist() == [15.0, 15.0]


def test_practical_matrix_expands_all_supported_horizons(monkeypatch, tmp_path):
    from llapdiffusion.baselines import runner

    calls = []

    def fake_run_one(baseline, dataset, config, run_root, *, horizon=None):
        calls.append((baseline, dataset, horizon))
        return {"status": "ok", "baseline": baseline, "dataset": dataset, "horizon": horizon}

    monkeypatch.setattr(runner, "default_horizons", lambda dataset: (4, 8))
    monkeypatch.setattr(runner, "run_practical_one", fake_run_one)
    monkeypatch.setattr(runner, "write_rows", lambda *args, **kwargs: None)

    config = runner.TrainConfig(source_root=None, horizons="all")
    rows = runner.run_practical_matrix(("mr-diff",), ("crypto",), config, tmp_path)

    assert calls == [("mr-diff", "crypto", 4), ("mr-diff", "crypto", 8)]
    assert [row["horizon"] for row in rows] == [4, 8]


def test_write_jobs_omits_source_root_for_first_party_mr_diff(tmp_path):
    from llapdiffusion.baselines.runner import write_isambard_jobs

    scripts = write_isambard_jobs(
        tmp_path,
        source_root="/unused/external/root",
        datasets=("crypto",),
        baselines=("mr-diff",),
        horizons=(100,),
    )

    assert len(scripts) == 1
    script = scripts[0].read_text(encoding="utf-8")
    assert "--baseline mr-diff --dataset crypto --horizons 100" in script
    assert "--baseline-source-root" not in script


def test_write_jobs_requires_source_root_for_external_baselines(tmp_path):
    from llapdiffusion.baselines.runner import write_isambard_jobs

    with pytest.raises(ValueError, match="External baseline jobs require"):
        write_isambard_jobs(tmp_path, source_root=None, datasets=("crypto",), baselines=("dlinear",))


def test_masked_metrics_and_point_crps_respect_valid_mask():
    pred = torch.tensor([[[1.0, 10.0], [3.0, 4.0]]])
    y = torch.tensor([[[2.0, 20.0], [1.0, 4.0]]])
    valid = torch.tensor([[[True, False], [True, True]]])
    assert torch.allclose(masked_mse(pred, y, valid), torch.tensor((1.0 + 4.0 + 0.0) / 3.0))
    assert torch.allclose(masked_mae(pred, y, valid), torch.tensor((1.0 + 2.0 + 0.0) / 3.0))
    samples = pred.unsqueeze(0)
    crps, mse = sample_crps(samples, y, valid)
    assert torch.allclose(crps, masked_mae(pred, y, valid))
    assert torch.allclose(mse, masked_mse(pred, y, valid))


def test_source_manager_loads_modules_without_leaking_sys_path(tmp_path):
    module_path = tmp_path / "fake_upstream.py"
    module_path.write_text("VALUE = 7\n", encoding="utf-8")
    manager = SourceManager(tmp_path)
    before = list(sys.path)
    with prepend_paths(tmp_path):
        assert sys.path[0] == str(tmp_path)
    assert sys.path == before
    module = manager.load_module("llap_fake_upstream", module_path)
    assert module.VALUE == 7


def test_prepend_paths_cleans_selected_imported_modules(tmp_path):
    module_path = tmp_path / "fake_upstream.py"
    module_path.write_text("VALUE = 11\n", encoding="utf-8")
    before = list(sys.path)
    with prepend_paths(tmp_path, module_prefixes=("fake_upstream",)):
        module = importlib.import_module("fake_upstream")
        assert module.VALUE == 11
    assert sys.path == before
    assert "fake_upstream" not in sys.modules


def test_prepend_paths_restores_preexisting_modules(tmp_path):
    module_path = tmp_path / "fake_upstream.py"
    module_path.write_text("VALUE = 11\n", encoding="utf-8")
    sentinel = ModuleType("fake_upstream")
    sentinel.VALUE = 5
    child = ModuleType("fake_upstream.child")
    sentinel.child = child
    sys.modules["fake_upstream"] = sentinel
    sys.modules["fake_upstream.child"] = child
    try:
        with prepend_paths(tmp_path, module_prefixes=("fake_upstream",)):
            module = importlib.import_module("fake_upstream")
            assert module.VALUE == 11
        assert sys.modules["fake_upstream"] is sentinel
        assert sys.modules["fake_upstream.child"] is child
    finally:
        sys.modules.pop("fake_upstream", None)
        sys.modules.pop("fake_upstream.child", None)


def test_select_stable_entities_uses_context_coverage_across_splits():
    def batch(mask):
        V = torch.zeros(1, 3, 2, 1)
        T = torch.zeros_like(V)
        y = torch.zeros(1, 3, 1)
        meta = {
            "x_obs_mask": torch.tensor(mask, dtype=torch.bool).reshape(1, 3, 2, 1),
            "entity_mask": torch.ones(1, 3, dtype=torch.bool),
        }
        return (V, T), y, meta

    loaders = [
        [batch([1, 1, 1, 1, 0, 0])],
        [batch([0, 0, 1, 1, 1, 1])],
        [batch([0, 0, 1, 1, 0, 0])],
    ]
    selected = select_stable_entities(
        loaders,
        {"dataset": "demo", "feature_cols": ["target"], "target_col": "target"},
        torch.device("cpu"),
        max_entities=1,
        max_batches=2,
    )
    assert selected == [1]


def test_mr_diff_adapter_handles_masked_irregular_inputs_and_backward():
    MRDiffAdapter = _require_mr_diff_adapter()
    torch.manual_seed(11)
    dataset_info = _sample_dataset_info(context=6, horizon=5)
    sample_batch = _sample_baseline_batch(batch_size=2, entities=2, context=6, horizon=5)
    (V, T), y, meta = sample_batch
    V = V.clone()
    y = y.clone()
    meta = {k: v.clone() if torch.is_tensor(v) else v for k, v in meta.items()}
    meta["x_obs_mask"][0, 0, 1, 0] = False
    meta["x_obs_mask"][1, 1, 3, 0] = False
    meta["y_obs_mask"][0, 1, 2] = False
    meta["entity_mask"][1, 1] = False
    meta["delta_t"] = torch.tensor(
        [
            [[0.0, 1.0, 1.5, 4.0, 4.5, 9.0], [0.0, 2.0, 3.0, 7.0, 8.0, 12.0]],
            [[0.0, 0.5, 2.0, 2.5, 6.0, 10.0], [0.0, 3.0, 3.5, 4.0, 8.0, 11.0]],
        ]
    )
    meta["delta_t_y"] = torch.tensor(
        [
            [[1.0, 2.0, 4.0, 7.0, 11.0], [1.0, 3.0, 4.0, 8.0, 13.0]],
            [[2.0, 3.0, 5.0, 8.0, 12.0], [1.0, 2.0, 6.0, 9.0, 14.0]],
        ]
    )
    batch = (V.requires_grad_(True), T), y, meta
    adapter = MRDiffAdapter(dataset_info, batch, num_samples=2, stages=2, kernels=(3,), width=8, diffusion_steps=2)

    loss = adapter.loss(batch, dataset_info)
    loss.backward()
    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in adapter.parameters()
        if p.requires_grad
    )
    adapter.zero_grad(set_to_none=True)
    pred = adapter(batch, dataset_info)
    pred.sum().backward()
    forward_grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in adapter.parameters()
        if p.requires_grad
    )

    assert torch.isfinite(loss)
    assert grad_ok
    assert pred.shape == (2, 2, 5)
    assert torch.isfinite(pred).all()
    assert pred.requires_grad is True
    assert forward_grad_ok


def test_mr_diff_conditioning_does_not_depend_on_clean_future_targets():
    MRDiffAdapter = _require_mr_diff_adapter()
    torch.manual_seed(19)
    dataset_info = _sample_dataset_info(context=6, horizon=5)
    batch = _sample_baseline_batch(batch_size=2, entities=2, context=6, horizon=5)
    (V, T), y, meta = batch
    altered = (V, T), y.mul(100.0).add(17.0), meta
    adapter = MRDiffAdapter(dataset_info, batch, num_samples=1, stages=2, kernels=(3,), width=8, diffusion_steps=2)
    adapter.eval()

    features_a, _, _, future_time_a, _ = adapter._inputs(batch, dataset_info)
    features_b, _, _, future_time_b, _ = adapter._inputs(altered, dataset_info)

    assert torch.allclose(future_time_a, future_time_b)
    for feat_a, feat_b in zip(features_a, features_b, strict=True):
        assert torch.allclose(feat_a, feat_b)

    with torch.no_grad():
        history_a = [stage.history(features_a[idx]) for idx, stage in enumerate(adapter.stages)]
        history_b = [stage.history(features_b[idx]) for idx, stage in enumerate(adapter.stages)]
        for stage_idx, stage in enumerate(adapter.stages):
            coarse_a = history_a[stage_idx + 1] if stage_idx < len(adapter.stages) - 1 else None
            coarse_b = history_b[stage_idx + 1] if stage_idx < len(adapter.stages) - 1 else None
            cond_a = stage.denoiser.condition(history_a[stage_idx], coarse_a, future_time_a)
            cond_b = stage.denoiser.condition(history_b[stage_idx], coarse_b, future_time_b)
            assert torch.allclose(cond_a, cond_b)
