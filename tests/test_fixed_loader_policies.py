from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from llapdiffusion.baselines.data import regular_feature_target_index, target_mask
from llapdiffusion.baselines.features import regular_features
from llapdiffusion.datasets.dataset_summary import _apply_split
from llapdiffusion.datasets.fin_dataset import (
    CachePaths,
    _assign_ratio_splits,
    load_dataloaders_with_ratio_split,
    make_collate_level_and_firstdiff,
)


def _write_tiny_cache(
    root: Path,
    *,
    num_assets: int = 2,
    length: int = 40,
    window: int = 2,
    horizon: int = 3,
) -> tuple[Path, np.ndarray, np.ndarray]:
    paths = CachePaths.from_dir(root)
    paths.ensure()
    assets = [f"a{idx}" for idx in range(num_assets)]
    feature_cols = ["noise", "target"]
    start_time = np.datetime64("2020-01-01T00:00:00", "ns")
    pairs = []
    end_times = []

    for aid in range(num_assets):
        values = np.arange(length, dtype=np.float32)
        features = np.stack([values + 1000 * aid, values + 10 * aid], axis=1)
        targets = values + 10 * aid
        times = start_time + np.arange(length).astype("timedelta64[h]")
        obs = np.ones_like(features, dtype=bool)

        np.save(paths.features / f"{aid}.npy", features.astype(np.float16))
        np.save(paths.targets / f"{aid}.npy", targets.astype(np.float16))
        np.save(paths.times / f"{aid}.npy", times.astype("datetime64[ns]"))
        np.save(paths.obs_masks / f"{aid}.npy", obs)
        np.save(paths.fill_masks / f"{aid}.npy", obs)

        max_start = length - window - horizon + 1
        starts = np.arange(max_start, dtype=np.int32)
        pairs.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        end_times.append(times[starts + window - 1])

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)
    global_end_times = np.concatenate(end_times, axis=0).astype("datetime64[ns]")
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", global_end_times)
    paths.meta.write_text(
        json.dumps(
            {
                "dataset": "tiny",
                "assets": assets,
                "asset2id": {asset: idx for idx, asset in enumerate(assets)},
                "feature_cols": feature_cols,
                "target_col": "target",
                "window": window,
                "horizon": horizon,
                "max_window": window,
                "max_horizon": horizon,
                "keep_time_meta": "end",
                "clamp_sigma": 5.0,
                "freq": "1h",
                "normalize_per_ticker": False,
            }
        ),
        encoding="utf-8",
    )
    paths.norm_stats.write_text(
        json.dumps(
            {
                "per_ticker": False,
                "mean_x": [[[0.0, 0.0]]],
                "std_x": [[[1.0, 1.0]]],
                "mean_y": 0.0,
                "std_y": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return root, global_pairs, global_end_times


def _target_time_indices(pairs: np.ndarray, assign: np.ndarray, split: int, *, window: int, horizon: int) -> set[int]:
    out: set[int] = set()
    for aid, start in pairs[assign == split]:
        del aid
        for offset in range(horizon):
            out.add(int(start) + window + offset)
    return out


def test_global_purged_split_has_no_target_timestamp_overlap(tmp_path):
    _, pairs, end_times = _write_tiny_cache(tmp_path, length=40, window=2, horizon=3)
    order = np.argsort(end_times.astype("datetime64[ns]").astype(np.int64), kind="mergesort")
    pairs = pairs[order]
    end_times = end_times[order]

    assign = _assign_ratio_splits(
        pairs,
        end_times,
        0.7,
        0.1,
        0.2,
        per_asset=True,
        split_policy="global_purged_horizon",
        horizon=3,
    )

    train_targets = _target_time_indices(pairs, assign, 0, window=2, horizon=3)
    val_targets = _target_time_indices(pairs, assign, 1, window=2, horizon=3)
    test_targets = _target_time_indices(pairs, assign, 2, window=2, horizon=3)
    assert train_targets
    assert val_targets
    assert test_targets
    assert train_targets.isdisjoint(val_targets)
    assert train_targets.isdisjoint(test_targets)
    assert val_targets.isdisjoint(test_targets)


def test_dataset_summary_split_counts_match_loader(tmp_path):
    data_dir, pairs, end_times = _write_tiny_cache(tmp_path, length=40, window=2, horizon=3)
    loaders = load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        batch_size=4,
        norm_scope="train_only",
        date_batching=False,
        window=2,
        horizon=3,
        split_policy="global_purged_horizon",
        exact_timestamp_batches=True,
    )
    summary_counts = _apply_split(
        pairs,
        end_times,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        per_asset=True,
        split_policy="global_purged_horizon",
        horizon=3,
    )
    assert loaders[3] == summary_counts


def test_exact_timestamp_collate_preserves_hourly_rows_on_same_day():
    collate = make_collate_level_and_firstdiff(n_entities=1, return_entity_mask=True)
    first = (
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([3.0]),
        {
            "asset_id": 0,
            "ctx_times": np.array([np.datetime64("2020-01-01T01:00:00", "ns")]),
            "y_times": np.array([np.datetime64("2020-01-01T02:00:00", "ns")]),
            "delta_t": np.array([0.0, 1.0], dtype=np.float32),
            "delta_t_y": np.array([1.0], dtype=np.float32),
            "x_obs_mask": np.ones((2, 1), dtype=bool),
            "y_obs_mask": np.ones((1,), dtype=bool),
        },
    )
    second = (
        torch.tensor([[4.0], [5.0]]),
        torch.tensor([6.0]),
        {
            "asset_id": 0,
            "ctx_times": np.array([np.datetime64("2020-01-01T02:00:00", "ns")]),
            "y_times": np.array([np.datetime64("2020-01-01T03:00:00", "ns")]),
            "delta_t": np.array([0.0, 1.0], dtype=np.float32),
            "delta_t_y": np.array([1.0], dtype=np.float32),
            "x_obs_mask": np.ones((2, 1), dtype=bool),
            "y_obs_mask": np.ones((1,), dtype=bool),
        },
    )

    (V, _), y, meta = collate([first, second])

    assert V.shape[:2] == (2, 1)
    assert y.shape == (2, 1, 1)
    assert meta["entity_mask"].sum().item() == 2
    assert meta["context_end_time_keys"].shape == (2,)
    assert torch.unique(meta["date_keys"]).numel() == 1


def test_target_only_regular_features_and_output_channel_use_transformed_target():
    V = torch.tensor([[[[10.0, 1.0], [20.0, 2.0], [30.0, 4.0]]]])
    T = torch.zeros_like(V)
    y = torch.tensor([[[5.0, 6.0]]])
    meta = {
        "x_obs_mask": torch.ones_like(V, dtype=torch.bool),
        "y_obs_mask": torch.ones_like(y, dtype=torch.bool),
        "entity_mask": torch.ones(1, 1, dtype=torch.bool),
        "delta_t": torch.arange(3, dtype=torch.float32).view(1, 1, 3),
        "delta_t_y": torch.arange(1, 3, dtype=torch.float32).view(1, 1, 2),
    }
    info = {
        "dataset": "demo",
        "feature_cols": ["noise", "target"],
        "target_col": "target",
        "input_policy": "target_only",
    }

    feat = regular_features(((V, T), y, meta), info)

    assert regular_feature_target_index(info) == 0
    assert torch.equal(feat[..., 0], V[..., 1])
    assert feat.shape[-1] == 7
    info["input_policy"] = "all_features"
    assert regular_feature_target_index(info) == 1


def test_public_dataset_wrappers_expose_split_and_batching_policy():
    modules = [
        "llapdiffusion.datasets.fin_dataset",
        "llapdiffusion.datasets.noaa_isd_dataset",
        "llapdiffusion.datasets.bms_air_dataset",
        "llapdiffusion.datasets.uci_air_quality_dataset",
        "llapdiffusion.datasets.physionet_cinc_dataset",
        "llapdiffusion.datasets.synthetic_regime_dataset",
    ]
    for name in modules:
        module = __import__(name, fromlist=["run_experiment"])
        signature = inspect.signature(module.run_experiment)
        assert "split_policy" in signature.parameters
        assert "exact_timestamp_batches" in signature.parameters


def test_public_ratio_loader_helpers_expose_split_and_batching_policy():
    helpers = [
        ("llapdiffusion.datasets.bms_air_dataset", "load_bms_dataloaders_with_ratio_split"),
        ("llapdiffusion.datasets.uci_air_quality_dataset", "load_uci_dataloaders_with_ratio_split"),
        ("llapdiffusion.datasets.physionet_cinc_dataset", "load_physionet_dataloaders_with_ratio_split"),
        ("llapdiffusion.datasets.noaa_isd_dataset", "load_isd_dataloaders_with_ratio_split"),
    ]
    for module_name, helper_name in helpers:
        module = __import__(module_name, fromlist=[helper_name])
        signature = inspect.signature(getattr(module, helper_name))
        assert "split_policy" in signature.parameters
        assert "exact_timestamp_batches" in signature.parameters


def test_synthetic_public_path_defaults_to_exact_timestamp_batching():
    from llapdiffusion.datasets.synthetic_regime_dataset import run_experiment
    from llapdiffusion.tools import run_synthetic_regime_shift

    run_signature = inspect.signature(run_experiment)
    assert run_signature.parameters["date_batching"].default is True
    assert run_signature.parameters["exact_timestamp_batches"].default is True

    cfg = run_synthetic_regime_shift._configure(
        run_synthetic_regime_shift.RunSpec(
            task="synthetic_freq_shift",
            seed=1,
            shift_multiplier=2.0,
            protocol_name="test",
        ),
        SimpleNamespace(
            artifact_root=".",
            data_root=".",
            output_root=".",
            protocol_name="test",
            window=4,
            horizon=2,
            series_length=16,
            change_point=8,
            num_entities=2,
            epochs=1,
            samples=1,
            overwrite_data=False,
            smoke=False,
            skip_existing=False,
            force_rebuild=False,
        ),
    )
    assert cfg.date_batching is True


def test_pipeline_forwards_loader_policy(monkeypatch):
    from llapdiffusion import pipeline

    seen = {}

    def fake_run_experiment(**kwargs):
        seen.update(kwargs)
        return "train", "val", "test", (1, 2, 3)

    monkeypatch.setattr(pipeline, "resolve_run_experiment", lambda data_dir: fake_run_experiment)
    cfg = SimpleNamespace(
        DATA_DIR="demo",
        date_batching=True,
        DATES_PER_BATCH=4,
        WINDOW=2,
        PRED=3,
        COVERAGE=0.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        split_policy="global_purged_horizon",
        exact_timestamp_batches=True,
    )

    assert pipeline.prepare_dataloaders(config=cfg)[3] == (1, 2, 3)
    assert seen["split_policy"] == "global_purged_horizon"
    assert seen["exact_timestamp_batches"] is True


def test_target_mask_excludes_missing_targets_even_when_entity_present():
    y = torch.tensor([[[1.0, 2.0, 0.0]]])
    meta = {
        "entity_mask": torch.ones(1, 1, dtype=torch.bool),
        "y_obs_mask": torch.tensor([[[True, False, True]]]),
    }

    mask = target_mask(meta, y)

    assert mask.tolist() == [[[True, False, True]]]
