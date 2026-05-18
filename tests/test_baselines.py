from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest
import torch

from llapdiffusion.baselines.data import target_index
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
