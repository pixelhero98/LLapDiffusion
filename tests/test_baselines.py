from __future__ import annotations

import importlib
import sys
from types import ModuleType

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


def test_target_index_fails_if_target_column_is_missing():
    with pytest.raises(ValueError, match="target_col"):
        target_index({"dataset": "demo", "feature_cols": ["x"], "target_col": "y"})


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
