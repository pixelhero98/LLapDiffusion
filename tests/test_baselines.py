from __future__ import annotations

import csv
import importlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llapdiffusion.baselines.data import find_batch, target_index
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
    assert "target-horizon imputation" in BASELINES["csdi"].dependency_caveat
    assert BASELINES["mtan"].probabilistic is True
    assert BASELINES["contiformer"].dependency_sources == (("physiopro", "5486d1ccaff8f33d635753e3debd7465234b09f1"),)


def test_benchmark_protocol_metadata_declares_comparison_scopes():
    from llapdiffusion.benchmark_protocol import baseline_protocol_metadata, llapdiff_protocol_metadata
    from llapdiffusion.baselines.runner import notes_payload

    dlinear = baseline_protocol_metadata("dlinear", requested_input_policy="all_features")
    timegrad = baseline_protocol_metadata("timegrad", requested_input_policy="all_features")
    csdi = baseline_protocol_metadata("csdi")
    llapdiff = llapdiff_protocol_metadata()
    notes = notes_payload()

    assert dlinear["comparison_type"] == "extrapolation"
    assert dlinear["input_scope"] == "all_features"
    assert dlinear["input_policy_effective"] == "all_features"
    assert timegrad["input_scope"] == "target_only"
    assert timegrad["input_policy_effective"] == "target_only"
    assert timegrad["missingness_scope"] == "target_mask"
    assert csdi["comparison_type"] == "imputation"
    assert "target_horizon" in csdi["missingness_scope"]
    assert llapdiff["modeling_scope"] == "joint global"
    assert llapdiff["missingness_scope"] == "per_feature_covariate_mask"
    assert notes["csdi"]["comparison_type"] == "imputation"
    assert notes["dlinear"]["modeling_scope"] == "uni-average/shared-weight"
    assert timegrad["num_eval_samples"] == 25
    assert timegrad["seed_aggregation"] == "single_seed"
    assert dlinear["seed_aggregation"] == "mean"
    assert dlinear["deterministic_seed_count"] == 10


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
        "lengths": [1, 1, 1],
        "copied_cache": False,
        "feature_cols": ["target"],
        "target_col": "target",
        "target_cols": ["target"],
        "target_indices": [0],
        "target_dim": 1,
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

    def fake_run_experiment(
        data_dir,
        K,
        H,
        ratios,
        per_asset,
        date_batching,
        coverage,
        dates_per_batch,
        batch_size,
        norm,
        reindex,
        split_policy,
        exact_timestamp_batches,
    ):
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
            "split_policy": split_policy,
            "exact_timestamp_batches": exact_timestamp_batches,
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
    assert seen["dates_per_batch"] == 3
    assert seen["split_policy"] == "global_purged_horizon"
    assert seen["exact_timestamp_batches"] is True
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

    def fake_run_experiment(
        data_dir,
        K,
        H,
        ratios,
        per_asset,
        date_batching,
        coverage,
        dates_per_batch,
        batch_size,
        norm,
        reindex,
        split_policy,
        exact_timestamp_batches,
    ):
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
            "split_policy": split_policy,
            "exact_timestamp_batches": exact_timestamp_batches,
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
    assert seen["split_policy"] == "global_purged_horizon"
    assert seen["exact_timestamp_batches"] is True
    assert info["horizon"] == 8


def test_baseline_loader_forwards_induced_context_missingness(monkeypatch, tmp_path):
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
        horizons=(4,),
        context_length=16,
        table_batch_size=2,
    )
    seen = {}

    def fake_run_experiment(
        data_dir,
        K,
        H,
        ratios,
        per_asset,
        date_batching,
        coverage,
        dates_per_batch,
        batch_size,
        norm,
        reindex,
        split_policy,
        exact_timestamp_batches,
    ):
        seen["coverage"] = coverage
        return ["train"], ["val"], ["test"], (1, 2, 3)

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)
    monkeypatch.setattr(baseline_data, "resolve_run_experiment", lambda path: fake_run_experiment)

    _, info = baseline_data.load_dataset_loaders(
        "demo",
        horizon=4,
        allow_cache_copy=False,
        work_cache_dir=None,
        coverage=0.35,
    )

    assert seen["coverage"] == 0.35
    assert info["coverage"] == 0.35


def test_baseline_loader_forwards_target_col_and_reports_effective_target(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    data_dir = tmp_path / "demo"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        '{"assets": ["a"], "feature_cols": ["target", "alt"], "target_col": "target"}',
        encoding="utf-8",
    )
    preset = SimpleNamespace(
        data_dir=data_dir,
        horizons=(4, 8),
        context_length=16,
        table_batch_size=2,
    )
    seen = {}

    def fake_run_experiment(
        data_dir,
        K,
        H,
        ratios,
        per_asset,
        date_batching,
        coverage,
        dates_per_batch,
        batch_size,
        norm,
        reindex,
        split_policy,
        exact_timestamp_batches,
        target_col,
    ):
        seen["target_col"] = target_col
        return ["train"], ["val"], ["test"], (1, 2, 3)

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)
    monkeypatch.setattr(baseline_data, "resolve_run_experiment", lambda path: fake_run_experiment)

    _, info = baseline_data.load_dataset_loaders(
        "demo",
        horizon=4,
        target_col="alt",
        allow_cache_copy=False,
        work_cache_dir=None,
    )

    assert seen["target_col"] == "alt"
    assert info["target_col"] == "alt"
    assert info["target_source"] == "feature_column"
    assert info["target_index"] == 1


def test_baseline_loader_forwards_target_cols_and_reports_metadata(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    data_dir = tmp_path / "demo"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        '{"assets": ["a"], "feature_cols": ["open", "close", "DOW_SIN"], "target_col": "close"}',
        encoding="utf-8",
    )
    preset = SimpleNamespace(
        data_dir=data_dir,
        horizons=(4,),
        context_length=16,
        table_batch_size=2,
    )
    seen = {}

    def fake_run_experiment(
        data_dir,
        K,
        H,
        ratios,
        per_asset,
        date_batching,
        coverage,
        dates_per_batch,
        batch_size,
        norm,
        reindex,
        split_policy,
        exact_timestamp_batches,
        target_cols,
    ):
        seen["target_cols"] = target_cols
        seen["reindex"] = reindex
        return ["train"], ["val"], ["test"], (1, 2, 3)

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)
    monkeypatch.setattr(baseline_data, "resolve_run_experiment", lambda path: fake_run_experiment)

    _, info = baseline_data.load_dataset_loaders(
        "demo",
        horizon=4,
        target_cols=("open", "close"),
        allow_cache_copy=False,
        work_cache_dir=None,
    )

    assert seen["target_cols"] == ("open", "close")
    assert seen["reindex"] is True
    assert info["target_col"] == "open"
    assert info["target_cols"] == ["open", "close"]
    assert info["target_indices"] == [0, 1]
    assert info["target_dim"] == 2
    assert info["target_source"] == "feature_columns"
    assert info["requested_target_cols"] == ["open", "close"]


@pytest.mark.parametrize(
    ("module_name", "dataset_name", "rebuild_attr", "loader_attr"),
    [
        (
            "llapdiffusion.datasets.noaa_isd_dataset",
            "noaa_isd",
            "_rebuild_window_index_only",
            "load_isd_dataloaders_with_ratio_split",
        ),
        (
            "llapdiffusion.datasets.physionet_cinc_dataset",
            "physionet_cinc",
            "_rebuild_window_index_only",
            "load_physionet_dataloaders_with_ratio_split",
        ),
        (
            "llapdiffusion.datasets.synthetic_regime_dataset",
            "synthetic_regime",
            "rebuild_window_index_only",
            "_load_fin_ratio_split",
        ),
    ],
)
def test_dataset_wrappers_reindex_requested_target_cols_when_horizon_matches(
    monkeypatch,
    tmp_path,
    module_name,
    dataset_name,
    rebuild_attr,
    loader_attr,
):
    module = importlib.import_module(module_name)
    data_dir = tmp_path / dataset_name
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "assets": ["a"],
                "feature_cols": ["open", "close"],
                "target_col": "close",
                "target_cols": ["close"],
                "window": 12,
                "horizon": 4,
                "max_window": 12,
                "max_horizon": 4,
            }
        ),
        encoding="utf-8",
    )

    rebuild_calls = []

    def fake_rebuild(data_dir, **kwargs):
        rebuild_calls.append(kwargs)
        return 1

    monkeypatch.setattr(module, rebuild_attr, fake_rebuild)
    monkeypatch.setattr(
        module,
        loader_attr,
        lambda **kwargs: (["train"], ["val"], ["test"], (1, 2, 3)),
    )

    loaders = module.run_experiment(
        data_dir,
        K=12,
        H=4,
        target_cols=("open", "close"),
        reindex=True,
    )

    assert loaders == (["train"], ["val"], ["test"], (1, 2, 3))
    assert len(rebuild_calls) == 1
    assert rebuild_calls[0]["window"] == 12
    assert rebuild_calls[0]["horizon"] == 4
    assert rebuild_calls[0]["update_meta"] is False
    assert rebuild_calls[0]["target_cols"] == ("open", "close")


def test_baseline_loader_rejects_multi_target_without_loader_support(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    data_dir = tmp_path / "demo"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        '{"assets": ["a"], "feature_cols": ["open", "close"], "target_col": "close"}',
        encoding="utf-8",
    )
    preset = SimpleNamespace(
        data_dir=data_dir,
        horizons=(4,),
        context_length=16,
        table_batch_size=2,
    )

    def fake_run_experiment(data_dir, K, H, ratios, per_asset, date_batching, coverage, dates_per_batch, batch_size, norm, reindex, split_policy, exact_timestamp_batches):
        return ["train"], ["val"], ["test"], (1, 2, 3)

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)
    monkeypatch.setattr(baseline_data, "resolve_run_experiment", lambda path: fake_run_experiment)

    with pytest.raises(RuntimeError, match="target_cols"):
        baseline_data.load_dataset_loaders(
            "demo",
            horizon=4,
            target_cols=("open", "close"),
            allow_cache_copy=False,
            work_cache_dir=None,
        )


def test_baseline_loader_maps_single_target_cols_to_legacy_target_col(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    data_dir = tmp_path / "demo"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        '{"assets": ["a"], "feature_cols": ["open", "close"], "target_col": "close"}',
        encoding="utf-8",
    )
    preset = SimpleNamespace(data_dir=data_dir, horizons=(4,), context_length=16, table_batch_size=2)
    seen = {}

    def fake_run_experiment(data_dir, K, H, ratios, per_asset, date_batching, coverage, dates_per_batch, batch_size, norm, reindex, split_policy, exact_timestamp_batches, target_col):
        seen["target_col"] = target_col
        return ["train"], ["val"], ["test"], (1, 2, 3)

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)
    monkeypatch.setattr(baseline_data, "resolve_run_experiment", lambda path: fake_run_experiment)

    _, info = baseline_data.load_dataset_loaders(
        "demo",
        horizon=4,
        target_cols=("open",),
        allow_cache_copy=False,
        work_cache_dir=None,
    )

    assert seen["target_col"] == "open"
    assert info["target_cols"] == ["open"]
    assert info["target_dim"] == 1


def test_baseline_loader_uses_physionet_preset_split_policy(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    data_dir = tmp_path / "physionet"
    meta_dir = data_dir / "cache_ratio_index"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(
        '{"assets": ["p1"], "feature_cols": ["HR"], "target_col": "HR"}',
        encoding="utf-8",
    )
    preset = SimpleNamespace(
        data_dir=data_dir,
        horizons=(4, 8, 12),
        context_length=24,
        table_batch_size=5,
        split_policy="contiguous",
        split_scope="physionet_patient_relative_time",
        exact_timestamp_batches=True,
    )
    seen = {}

    def fake_run_experiment(
        data_dir,
        K,
        H,
        ratios,
        per_asset,
        date_batching,
        coverage,
        dates_per_batch,
        batch_size,
        norm,
        reindex,
        split_policy,
        exact_timestamp_batches,
    ):
        seen.update(
            {
                "split_policy": split_policy,
                "exact_timestamp_batches": exact_timestamp_batches,
                "per_asset": per_asset,
                "H": H,
            }
        )
        return ["train"], ["val"], ["test"], (10, 2, 3)

    monkeypatch.setattr(baseline_data, "get_dataset_preset", lambda key: preset)
    monkeypatch.setattr(baseline_data, "resolve_run_experiment", lambda path: fake_run_experiment)

    _, info = baseline_data.load_dataset_loaders(
        "physionet",
        horizon=12,
        allow_cache_copy=False,
        work_cache_dir=None,
    )

    assert seen == {
        "split_policy": "contiguous",
        "exact_timestamp_batches": True,
        "per_asset": True,
        "H": 12,
    }
    assert info["split_policy"] == "contiguous"
    assert info["split_scope"] == "physionet_patient_relative_time"
    assert info["split_note"] == "patient_relative_contiguous_split"
    assert info["split_caveat"] == "special_case_insufficient_horizon_windows_for_purged_split"
    assert info["batching_policy"] == "exact_context_end_timestamp"


def test_baseline_loader_rejects_split_argument(monkeypatch, tmp_path):
    from llapdiffusion.baselines import data as baseline_data

    with pytest.raises(TypeError, match="split"):
        baseline_data.load_dataset_loaders("demo", **{"spl" + "it": "train"}, allow_cache_copy=False, work_cache_dir=None)


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
    with pytest.raises(ValueError, match="target"):
        target_index({"dataset": "demo", "feature_cols": ["x"], "target_col": "y"})


def test_target_context_and_regular_features_support_multi_target():
    from llapdiffusion.baselines.features import regular_features, target_context

    batch = _sample_baseline_batch(batch_size=1, entities=2, context=3, horizon=2)
    (V, T), _, meta = batch
    V = torch.cat([V, V + 10.0], dim=-1)
    T = torch.cat([T, T], dim=-1)
    y = torch.stack(
        [
            torch.full((1, 2, 2), 1.0),
            torch.full((1, 2, 2), 2.0),
        ],
        dim=-1,
    )
    meta = {k: v.clone() if torch.is_tensor(v) else v for k, v in meta.items()}
    meta["x_obs_mask"] = torch.ones_like(V, dtype=torch.bool)
    meta["y_obs_mask"] = torch.ones_like(y, dtype=torch.bool)
    info = {
        **_sample_dataset_info(context=3, horizon=2),
        "feature_cols": ["open", "close"],
        "target_col": "open",
        "target_cols": ["open", "close"],
        "target_indices": [0, 1],
        "target_dim": 2,
    }

    x, mask, y_clean, valid = target_context(((V, T), y, meta), info)
    feat = regular_features(((V, T), y, meta), info)

    assert x.shape == (1, 2, 3, 2)
    assert mask.shape == (1, 2, 3, 2)
    assert y_clean.shape == (1, 2, 2, 2)
    assert valid.shape == (1, 2, 2, 2)
    assert feat.shape[-1] == 10


def test_dlinear_adapter_returns_multi_target_channels(monkeypatch):
    from llapdiffusion.baselines.adapters.dlinear import DLinearAdapter

    class FakeModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

        def forward(self, x):
            B, _, C = x.shape
            return torch.arange(C, dtype=x.dtype, device=x.device).view(1, 1, C).expand(B, self.cfg.pred_len, C)

    source_manager = SimpleNamespace(
        load_module=lambda *args, **kwargs: SimpleNamespace(Model=FakeModel),
        path=lambda name: Path(name),
    )
    batch = _sample_baseline_batch(batch_size=1, entities=2, context=3, horizon=2)
    (V, T), _, meta = batch
    V = torch.cat([V, V + 10.0], dim=-1)
    T = torch.cat([T, T], dim=-1)
    y = torch.zeros(1, 2, 2, 2)
    meta = {k: v.clone() if torch.is_tensor(v) else v for k, v in meta.items()}
    meta["x_obs_mask"] = torch.ones_like(V, dtype=torch.bool)
    meta["y_obs_mask"] = torch.ones_like(y, dtype=torch.bool)
    info = {
        **_sample_dataset_info(context=3, horizon=2),
        "feature_cols": ["open", "close"],
        "target_cols": ["open", "close"],
        "target_indices": [0, 1],
        "target_dim": 2,
    }

    adapter = DLinearAdapter(info, ((V, T), y, meta), source_manager)
    pred = adapter(((V, T), y, meta), info)

    assert pred.shape == (1, 2, 2, 2)
    assert pred[..., 0].eq(0).all()
    assert pred[..., 1].eq(1).all()


def test_csdi_target_horizon_mask_hides_only_future_tokens():
    from llapdiffusion.baselines.adapters.csdi import CSDIAdapter

    adapter = CSDIAdapter.__new__(CSDIAdapter)
    adapter.imputation_random_mask_ratio = 0.50
    batch = _sample_baseline_batch(batch_size=1, entities=2, context=3, horizon=4)
    dataset_info = _sample_dataset_info(context=3, horizon=4)

    torch.manual_seed(13)
    csdi_batch = adapter._batch(batch, dataset_info)

    observed = csdi_batch["observed_mask"]
    kept = csdi_batch["gt_mask"]
    assert observed.shape == (1, 7, 2)
    assert torch.equal(kept[:, :3, :], observed[:, :3, :])
    hidden_future = observed[:, 3:, :] - kept[:, 3:, :]
    assert hidden_future.sum().item() == 4.0
    assert kept[:, 3:, :].sum().item() == 4.0


def test_csdi_timepoints_use_context_end_relative_future_offsets():
    from llapdiffusion.baselines.adapters.csdi import CSDIAdapter

    adapter = CSDIAdapter.__new__(CSDIAdapter)
    adapter.imputation_random_mask_ratio = 0.50
    batch = _sample_baseline_batch(batch_size=1, entities=1, context=3, horizon=3)
    (V, T), y, meta = batch
    meta = {k: v.clone() if torch.is_tensor(v) else v for k, v in meta.items()}
    meta["delta_t"] = torch.tensor([[[0.0, 1.0, 4.0]]])
    meta["delta_t_y"] = torch.tensor([[[1.0, 4.0, 5.0]]])
    batch = (V, T), y, meta

    csdi_batch = adapter._batch(batch, _sample_dataset_info(context=3, horizon=3))

    expected = torch.tensor([[0.0, 1.0, 4.0, 5.0, 8.0, 9.0]]) / 9.0
    assert torch.allclose(csdi_batch["timepoints"], expected)


def test_csdi_timepoints_ignore_padded_entity_offsets():
    from llapdiffusion.baselines.adapters.csdi import CSDIAdapter

    adapter = CSDIAdapter.__new__(CSDIAdapter)
    adapter.imputation_random_mask_ratio = 0.50
    batch = _sample_baseline_batch(batch_size=1, entities=2, context=3, horizon=2)
    (V, T), y, meta = batch
    meta = {k: v.clone() if torch.is_tensor(v) else v for k, v in meta.items()}
    meta["entity_mask"] = torch.tensor([[True, False]])
    meta["delta_t"] = torch.tensor([[[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]]])
    meta["delta_t_y"] = torch.tensor([[[1.0, 3.0], [0.0, 0.0]]])
    batch = (V, T), y, meta

    csdi_batch = adapter._batch(batch, _sample_dataset_info(context=3, horizon=2))

    expected = torch.tensor([[0.0, 2.0, 4.0, 5.0, 7.0]]) / 7.0
    assert torch.allclose(csdi_batch["timepoints"], expected)


def test_baseline_future_time_features_preserve_nonuniform_query_grid():
    from llapdiffusion.baselines.features import time_features

    batch = _sample_baseline_batch(batch_size=1, entities=1, context=3, horizon=3)
    (V, _), _, meta = batch
    meta = {k: v.clone() if torch.is_tensor(v) else v for k, v in meta.items()}
    meta["delta_t_y"] = torch.tensor([[[1.0, 4.0, 5.0]]])

    _, _, ty = time_features(meta, V)

    assert torch.allclose(ty[0, 0], torch.tensor([0.2, 0.8, 1.0]))


def test_csdi_loss_and_samples_returns_target_horizon_mask():
    from llapdiffusion.baselines.adapters.csdi import CSDIAdapter

    class FakeCSDI(torch.nn.Module):
        def process_data(self, batch):
            observed_data = batch["observed_data"].float().permute(0, 2, 1)
            observed_mask = batch["observed_mask"].float().permute(0, 2, 1)
            observed_tp = batch["timepoints"].float()
            gt_mask = batch["gt_mask"].float().permute(0, 2, 1)
            return observed_data, observed_mask, observed_tp, gt_mask, None

        def get_side_info(self, observed_tp, gt_mask):
            return torch.zeros_like(gt_mask)

        def calc_loss(self, observed_data, gt_mask, observed_mask, side_info, is_train=1):
            return (observed_mask - gt_mask).sum()

        def evaluate(self, batch, n_samples=1):
            observed_data, observed_mask, observed_tp, gt_mask = self.process_data(batch)[:4]
            samples = observed_data.unsqueeze(0).expand(n_samples, -1, -1, -1).contiguous()
            target_mask = observed_mask - gt_mask
            return samples, observed_data, target_mask, observed_mask, observed_tp

    adapter = CSDIAdapter.__new__(CSDIAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.model = FakeCSDI()
    adapter.num_samples = 2
    adapter.imputation_random_mask_ratio = 0.50
    batch = _sample_baseline_batch(batch_size=1, entities=2, context=3, horizon=4)

    torch.manual_seed(13)
    loss, samples, observed, target_mask = adapter.loss_and_samples(batch, _sample_dataset_info(context=3, horizon=4))

    assert torch.isfinite(loss)
    assert samples.shape == (1, 2, 2, 7)
    assert observed.shape == (1, 2, 7)
    assert target_mask[:, :, :3].sum().item() == 0.0
    assert target_mask[:, :, 3:].sum().item() == 4.0


def test_csdi_target_horizon_mask_hides_sparse_future_token():
    from llapdiffusion.baselines.adapters.csdi import CSDIAdapter

    adapter = CSDIAdapter.__new__(CSDIAdapter)
    adapter.imputation_random_mask_ratio = 0.30
    observed = torch.zeros(1, 5, 2)
    observed[:, :3, :] = 1.0
    observed[:, 4, 1] = 1.0

    kept = adapter._target_horizon_gt_mask(observed, context_length=3)

    assert torch.equal(kept[:, :3, :], observed[:, :3, :])
    assert kept[:, 3:, :].sum().item() == 0.0
    assert (observed[:, 3:, :] - kept[:, 3:, :]).sum().item() == 1.0


def test_csdi_loss_uses_target_horizon_gt_mask_for_training():
    from llapdiffusion.baselines.adapters.csdi import CSDIAdapter

    class FakeCSDI(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.forward_called = False
            self.cond_mask = None
            self.received_gt = None
            self.target_mask = None

        def forward(self, batch, is_train=1):
            self.forward_called = True
            raise AssertionError("adapter should not use upstream random-mask forward for training")

        def process_data(self, batch):
            observed_data = batch["observed_data"].float().permute(0, 2, 1)
            observed_mask = batch["observed_mask"].float().permute(0, 2, 1)
            observed_tp = batch["timepoints"].float()
            gt_mask = batch["gt_mask"].float().permute(0, 2, 1)
            self.received_gt = gt_mask.detach().clone()
            return observed_data, observed_mask, observed_tp, gt_mask, observed_mask, torch.zeros(observed_data.shape[0])

        def get_side_info(self, observed_tp, cond_mask):
            self.cond_mask = cond_mask.detach().clone()
            return cond_mask.unsqueeze(1)

        def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
            self.target_mask = observed_mask - cond_mask
            return self.target_mask.sum() * self.weight

    adapter = CSDIAdapter.__new__(CSDIAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.imputation_random_mask_ratio = 0.50
    adapter.model = FakeCSDI()
    batch = _sample_baseline_batch(batch_size=1, entities=2, context=3, horizon=4)
    dataset_info = _sample_dataset_info(context=3, horizon=4)

    torch.manual_seed(17)
    loss = adapter.loss(batch, dataset_info)
    loss.backward()

    assert adapter.model.forward_called is False
    assert torch.equal(adapter.model.cond_mask, adapter.model.received_gt)
    assert adapter.model.target_mask[..., :3].sum().item() == 0.0
    assert adapter.model.target_mask[..., 3:].sum().item() == 4.0
    assert adapter.model.weight.grad.item() == pytest.approx(4.0)


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


def test_practical_matrix_averages_deterministic_seed_rows(monkeypatch, tmp_path):
    from llapdiffusion.baselines import runner

    calls = []
    written = []

    def fake_run_one(baseline, dataset, config, run_root, *, horizon=None):
        calls.append((baseline, dataset, horizon, config.seed, config.run_suffix))
        return {
            "status": "ok",
            "baseline": baseline,
            "dataset": dataset,
            "horizon": horizon,
            "seed": config.seed,
            "seeds": [config.seed],
            "seed_count": 1,
            "seed_aggregation": "single_seed",
            "num_samples": None,
            "completion_mode": "full_train_loop",
            "best_epoch": config.seed - 100,
            "best_val_mse": float(config.seed),
            "checkpoint": f"seed-{config.seed}/best.pt",
            "train_config": {"seed": config.seed, "deterministic_seeds": list(config.deterministic_seeds)},
            "test": {
                "loss": float(config.seed),
                "mse": float(config.seed),
                "mae": float(config.seed) / 10.0,
                "crps": None,
                "valid_observations": 12,
            },
            "runtime_seconds": 1.0,
        }

    monkeypatch.setattr(runner, "default_horizons", lambda dataset: (4,))
    monkeypatch.setattr(runner, "run_practical_one", fake_run_one)
    monkeypatch.setattr(runner, "write_rows", lambda rows, output, *, prefix: written.append((prefix, list(rows))))

    config = runner.TrainConfig(source_root=None, deterministic_seeds=(101, 102))
    rows = runner.run_practical_matrix(("dlinear",), ("crypto",), config, tmp_path)

    assert calls == [
        ("dlinear", "crypto", 4, 101, "seed101"),
        ("dlinear", "crypto", 4, 102, "seed102"),
    ]
    assert len(rows) == 1
    assert rows[0]["seed_aggregation"] == "mean"
    assert rows[0]["seed_count"] == 2
    assert rows[0]["seeds"] == [101, 102]
    assert rows[0]["test"]["mse"] == pytest.approx(101.5)
    assert rows[0]["test"]["mae"] == pytest.approx(10.15)
    assert [item[0] for item in written] == ["baseline_practical", "baseline_practical_seed_rows"]


def test_run_baselines_practical_defaults_are_full_comparison(monkeypatch):
    from llapdiffusion.tools import run_baselines

    monkeypatch.setattr(sys, "argv", ["llapdiff-baselines", "practical-extrapolation"])
    args = run_baselines.parse_args()
    config = run_baselines._train_config(args)

    assert config.horizons == "all"
    assert config.epochs == 600
    assert config.patience == 20
    assert config.num_samples == 25
    assert config.deterministic_seeds == tuple(range(42, 52))
    assert config.device == "auto"
    assert config.input_policy == "target_only"
    assert config.target_cols is None
    assert config.coverage == 0.0
    assert config.verbose is False


def test_run_baselines_accepts_target_cols(monkeypatch):
    from llapdiffusion.tools import run_baselines

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llapdiff-baselines",
            "practical-extrapolation",
            "--target-cols",
            "RET_OPEN",
            "RET_CLOSE",
            "--coverage",
            "0.25",
            "--verbose",
        ],
    )
    args = run_baselines.parse_args()
    config = run_baselines._train_config(args)

    assert config.target_col is None
    assert config.target_cols == ("RET_OPEN", "RET_CLOSE")
    assert config.coverage == 0.25
    assert config.verbose is True


def test_run_baselines_rejects_target_col_conflict(monkeypatch):
    from llapdiffusion.tools import run_baselines

    monkeypatch.setattr(
        sys,
        "argv",
        ["llapdiff-baselines", "practical-extrapolation", "--target-col", "RET_OPEN", "--target-cols", "RET_CLOSE"],
    )
    args = run_baselines.parse_args()
    with pytest.raises(SystemExit, match="either --target-col or --target-cols"):
        run_baselines._train_config(args)


def test_multi_target_rejected_for_scalar_only_baseline(tmp_path):
    from llapdiffusion.baselines import runner

    config = runner.TrainConfig(source_root=None, target_cols=("x", "y"))
    with pytest.raises(ValueError, match="scalar targets only"):
        runner.run_practical_one("timegrad", "crypto", config, tmp_path, horizon=4)


def test_run_baselines_csdi_defaults_to_target_horizon_all_horizons(monkeypatch):
    from llapdiffusion.tools import run_baselines

    monkeypatch.setattr(sys, "argv", ["llapdiff-baselines", "csdi-imputation"])
    args = run_baselines.parse_args()
    config = run_baselines._train_config(args)

    assert config.horizons == "all"
    assert config.epochs == 600
    assert config.patience == 20
    assert config.num_samples == 25
    assert config.device == "auto"


def test_run_baselines_public_help_excludes_removed_surfaces(monkeypatch, capsys):
    from llapdiffusion.tools import run_baselines

    removed = (
        "smo" + "ke",
        "--" + "qui" + "ck",
        "--ma" + "x-entities",
        "--ma" + "x-batches",
        "--ma" + "x-train-batches",
        "--ma" + "x-eval-batches",
        "--csdi-imputation-" + "target",
        "--fail-" + "fast",
        "--validate-" + "sources-only",
        "--num-" + "samples",
    )
    help_text = []
    for argv in (
        ["llapdiff-baselines", "--help"],
        ["llapdiff-baselines", "practical-extrapolation", "--help"],
        ["llapdiff-baselines", "csdi-imputation", "--help"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            run_baselines.parse_args()
        help_text.append(capsys.readouterr().out)
    combined = "\n".join(help_text)
    for token in removed:
        assert token not in combined


def test_public_docs_and_requirements_are_clone_ready():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").strip()
    readme = (root / "README.md").read_text(encoding="utf-8")
    main_commands = readme.split("## Main Commands", 1)[1].split("## Target Selection", 1)[0]
    baselines = readme.split("## Baselines", 1)[1].split("## Citation", 1)[0]
    tracked_text = "\n".join(
        path.read_text(encoding="utf-8")
        for base in ("README.md", "llapdiffusion", "tests")
        for path in ([root / base] if (root / base).is_file() else sorted((root / base).rglob("*.py")))
    )

    assert requirements == "-e ."
    assert "## Quick start" in readme
    assert "## Target Selection" in readme
    assert "--target-col" in readme
    assert "--target-cols" in readme
    assert "--coverage" in readme
    assert "fraction of observed context entries to hide" in readme
    assert "Dense-date panel filtering" in readme
    assert "panel_coverage" in readme
    assert "llapdiff-train" in main_commands
    assert "llapdiff-checkpoint-eval" in main_commands
    assert "--target-mask-aux-p" in main_commands
    assert "--imputation-random-mask-ratio" in main_commands
    assert "llapdiff-baselines csdi-imputation" not in main_commands
    assert "llapdiff-baselines csdi-imputation" in baselines
    assert "--target-mask-aux-p 0.0" in readme
    assert "--target-mask-aux-p > 0" in readme
    assert "hides 30% of observed target-horizon entries" in readme
    assert "https://arxiv.org/abs/2605.19805" in readme
    private_patterns = (
        r"hf_[A-Za-z0-9]{20,}",
        r"(?i)c:\\users\\[a-z0-9_.-]+",
        r"(?i)/home/[a-z0-9_.-]+",
    )
    for pattern in private_patterns:
        assert re.search(pattern, tracked_text) is None


def test_baseline_resolve_device_auto_uses_available_backend(monkeypatch):
    from llapdiffusion.baselines import runner

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    assert runner.resolve_device("auto").type == "cpu"
    assert runner.resolve_device("cpu").type == "cpu"

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    assert runner.resolve_device("auto").type == "cuda"


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


def test_sample_crps_uses_distinct_pair_estimator():
    samples = torch.tensor([[[[0.0]]], [[[4.0]]]])
    y = torch.tensor([[[1.0]]])
    valid = torch.tensor([[[True]]])

    crps, mse = sample_crps(samples, y, valid)

    assert torch.allclose(crps, torch.tensor(0.0))
    assert torch.allclose(mse, torch.tensor(1.0))


class _ZeroPointModel(torch.nn.Module):
    def forward(self, batch, dataset_info):
        return torch.zeros_like(batch[1])


def _metric_batch(values: torch.Tensor, valid: torch.Tensor):
    batch_size, entities, horizon = values.shape
    V = torch.zeros(batch_size, entities, 1, 1)
    T = torch.zeros_like(V)
    meta = {
        "x_obs_mask": torch.ones(batch_size, entities, 1, 1, dtype=torch.bool),
        "y_obs_mask": valid,
        "entity_mask": torch.ones(batch_size, entities, dtype=torch.bool),
    }
    return (V, T), values, meta


def test_evaluate_loader_weights_metrics_by_valid_observations():
    from llapdiffusion.baselines.runner import _evaluate_loader

    loader = [
        _metric_batch(torch.tensor([[[10.0, 0.0, 0.0]]]), torch.tensor([[[True, False, False]]])),
        _metric_batch(torch.tensor([[[1.0, 1.0, 1.0]]]), torch.tensor([[[True, True, True]]])),
    ]
    result = _evaluate_loader(
        _ZeroPointModel(),
        "dlinear",
        loader,
        _sample_dataset_info(context=1, horizon=3),
        torch.device("cpu"),
    )

    assert result["batches"] == 2
    assert result["valid_observations"] == 4
    assert result["raw_batches_scanned"] == 2
    assert result["metric_aggregation"] == "valid_observation_weighted"
    assert result["loss_aggregation"] == "batch_mean"
    assert result["mse"] == pytest.approx((100.0 + 1.0 + 1.0 + 1.0) / 4.0)
    assert result["mae"] == pytest.approx((10.0 + 1.0 + 1.0 + 1.0) / 4.0)
    assert result["crps"] is None


class _ZeroCSDIModel(torch.nn.Module):
    metric_target_type = "target_horizon_imputation"

    def loss_and_samples(self, batch, dataset_info):
        observed = batch[1].permute(0, 2, 1).contiguous()
        mask = batch[2]["csdi_target_mask"].permute(0, 2, 1).contiguous()
        return torch.tensor(0.0), torch.zeros(1, *observed.shape), observed, mask


def test_evaluate_loader_uses_csdi_holdout_denominator():
    from llapdiffusion.baselines.runner import _evaluate_loader

    first = _metric_batch(torch.tensor([[[10.0, 0.0, 0.0]]]), torch.tensor([[[True, False, False]]]))
    first[2]["csdi_target_mask"] = torch.tensor([[[True, False, False]]])
    second = _metric_batch(torch.tensor([[[1.0, 1.0, 1.0]]]), torch.tensor([[[True, True, True]]]))
    second[2]["csdi_target_mask"] = torch.tensor([[[True, True, True]]])
    result = _evaluate_loader(
        _ZeroCSDIModel(),
        "csdi",
        [first, second],
        _sample_dataset_info(context=1, horizon=3),
        torch.device("cpu"),
    )

    assert result["valid_observations"] == 4
    assert result["metric_target_type"] == "target_horizon_imputation"
    assert result["mse"] == pytest.approx((100.0 + 1.0 + 1.0 + 1.0) / 4.0)
    assert result["crps"] == pytest.approx((10.0 + 1.0 + 1.0 + 1.0) / 4.0)


def test_write_rows_flattens_practical_metrics(tmp_path):
    from llapdiffusion.baselines.runner import write_rows

    write_rows(
        [
            {
                "status": "ok",
                "baseline": "mr-diff",
                "dataset": "crypto",
                "comparison_type": "extrapolation",
                "horizon": 100,
                "entity_selection_mode": "full_panel",
                "input_policy": "target_only",
                "input_policy_effective": "target_only",
                "input_scope": "target_only",
                "missingness_scope": "target_mask",
                "modeling_scope": "multi-series",
                "split_note": "global_purged_horizon_split",
                "split_caveat": "none",
                "time_feature_protocol": "context features use delta_t only",
                "best_epoch": 3,
                "best_val_mse": 1.25,
                "loader_batches": {"train": 10, "val": 2, "test": 4},
                "test": {
                    "mse": 2.5,
                    "crps": 1.5,
                    "valid_observations": 99,
                    "metric_aggregation": "valid_observation_weighted",
                },
            }
        ],
        tmp_path,
        prefix="baseline_practical",
    )

    with (tmp_path / "baseline_practical.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["test_mse"] == "2.5"
    assert row["test_crps"] == "1.5"
    assert row["test_valid_observations"] == "99"
    assert row["test_metric_aggregation"] == "valid_observation_weighted"
    assert row["best_epoch"] == "3"
    assert row["comparison_type"] == "extrapolation"
    assert row["input_scope"] == "target_only"
    assert row["missingness_scope"] == "target_mask"
    assert row["modeling_scope"] == "multi-series"
    assert row["split_note"] == "global_purged_horizon_split"
    assert row["train_loader_batches"] == "10"
    assert row["val_loader_batches"] == "2"
    assert row["test_loader_batches"] == "4"
    assert "entity_cap" not in row
    assert "train_batch_cap" not in row


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


def test_find_batch_returns_full_panel_batch():
    batch = _sample_baseline_batch(batch_size=1, entities=5, context=3, horizon=2)
    found, skipped = find_batch(
        [batch],
        _sample_dataset_info(context=3, horizon=2),
        torch.device("cpu"),
    )

    assert skipped == 0
    assert found[0][0].shape[1] == 5


def test_practical_runner_reports_full_panel_scope(monkeypatch, tmp_path):
    from llapdiffusion.baselines import runner

    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, batch, dataset_info):
            return torch.zeros_like(batch[1]) + self.weight

    batch = _sample_baseline_batch(batch_size=1, entities=2, context=3, horizon=2)
    monkeypatch.setattr(runner, "resolve_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(runner, "SourceManager", lambda source_root: SimpleNamespace(validate=lambda spec: {
        "source_name": "LLapDiffusion",
        "source_sha": "first-party-paper-derived",
        "source_clean": True,
        "official_reference": "",
        "dependency_caveat": "",
        "dependency_sources": {},
    }))
    def fake_load_dataset_loaders(*args, **kwargs):
        seen["loader_coverage"] = kwargs["coverage"]
        return ([batch], [batch], [batch]), _sample_dataset_info(context=3, horizon=2)

    monkeypatch.setattr(runner, "load_dataset_loaders", fake_load_dataset_loaders)

    def fake_build_adapter(baseline, dataset_info, sample_batch, *args, **kwargs):
        seen["sample_entities"] = sample_batch[0][0].shape[1]
        return FakeModel()

    monkeypatch.setattr(runner, "build_adapter", fake_build_adapter)

    config = runner.TrainConfig(source_root=None, epochs=1, patience=1, coverage=0.4)
    result = runner.run_practical_one("dlinear", "crypto", config, tmp_path, horizon=2)

    assert seen["sample_entities"] == 2
    assert seen["loader_coverage"] == 0.4
    assert result["entity_selection_mode"] == "full_panel"
    assert result["num_entities_used"] == 2
    assert result["comparison_type"] == "extrapolation"
    assert result["input_scope"] == "target_only"
    assert result["missingness_scope"] == "target_mask"
    assert result["modeling_scope"] == "uni-average/shared-weight"
    assert "selected" + "_entities" not in result
    assert "entity_cap" not in result
    assert "train_batch_cap" not in result


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
