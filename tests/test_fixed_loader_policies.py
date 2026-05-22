from __future__ import annotations

import inspect
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llapdiffusion.baselines.data import regular_feature_target_index, target_index, target_mask
from llapdiffusion.baselines.features import regular_features, target_context
from llapdiffusion.datasets.dataset_summary import _apply_split
from llapdiffusion.datasets.target_selection import resolve_target_selection, valid_scalar_target_cols
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


def _write_query_grid_cache(
    root: Path,
    *,
    future_grids: list[tuple[float, ...]],
    window: int = 2,
) -> Path:
    """Write one window per asset with a shared context end and configurable query grids."""
    paths = CachePaths.from_dir(root)
    paths.ensure()
    horizon = len(future_grids[0])
    assets = [f"a{idx}" for idx in range(len(future_grids))]
    feature_cols = ["noise", "target"]
    base = np.datetime64("2020-01-01T00:00:00", "ns")
    context_hours = np.arange(window, dtype=np.int64)
    context_end = base + np.timedelta64(int(context_hours[-1]), "h")
    pairs = []
    end_times = []

    for aid, grid in enumerate(future_grids):
        assert len(grid) == horizon
        future_times = np.array(
            [context_end + np.timedelta64(int(offset * 3600), "s") for offset in grid],
            dtype="datetime64[ns]",
        )
        times = np.concatenate(
            [
                base + context_hours.astype("timedelta64[h]"),
                future_times,
            ]
        ).astype("datetime64[ns]")
        values = np.arange(times.shape[0], dtype=np.float32) + 10 * aid
        features = np.stack([values, values], axis=1)
        obs = np.ones_like(features, dtype=bool)

        np.save(paths.features / f"{aid}.npy", features.astype(np.float16))
        np.save(paths.targets / f"{aid}.npy", values.astype(np.float16))
        np.save(paths.times / f"{aid}.npy", times)
        np.save(paths.obs_masks / f"{aid}.npy", obs)
        np.save(paths.fill_masks / f"{aid}.npy", obs)
        pairs.append([aid, 0])
        end_times.append(times[window - 1])

    np.save(paths.windows / "global_pairs.npy", np.asarray(pairs, dtype=np.int32))
    np.save(paths.windows / "end_times.npy", np.asarray(end_times, dtype="datetime64[ns]"))
    paths.meta.write_text(
        json.dumps(
            {
                "dataset": "query_grid_tiny",
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
                "freq": "irregular",
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
    return root


def _query_grid_train_loader(
    data_dir: Path,
    *,
    coverage_per_window: float = 0.0,
    target_cols: tuple[str, ...] | None = None,
):
    train_dl, _, _, _ = load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        batch_size=8,
        n_entities=len(
            json.loads((CachePaths.from_dir(data_dir).meta).read_text(encoding="utf-8"))["assets"]
        ),
        norm_scope="cache",
        shuffle_train=False,
        date_batching=True,
        dates_per_batch=1,
        window=2,
        horizon=3,
        split_policy="contiguous",
        coverage_per_window=coverage_per_window,
        exact_timestamp_batches=True,
        target_cols=target_cols,
    )
    return train_dl


def _valid_query_grids(meta: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    grids = []
    for row_idx in range(meta["entity_mask"].shape[0]):
        valid = meta["entity_mask"][row_idx]
        if valid.any():
            grids.append(meta["delta_t_y"][row_idx, valid])
    return grids


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


def test_physionet_relative_time_split_uses_legacy_per_patient_policy(tmp_path):
    _, pairs, end_times = _write_tiny_cache(tmp_path, num_assets=4, length=49, window=24, horizon=12)
    order = np.lexsort((end_times.astype("datetime64[ns]").astype(np.int64), pairs[:, 0].astype(np.int64)))
    pairs = pairs[order]
    end_times = end_times[order]

    with pytest.raises(ValueError, match="Not enough unique context-end timestamps"):
        _assign_ratio_splits(
            pairs,
            end_times,
            0.7,
            0.1,
            0.2,
            per_asset=True,
            split_policy="global_purged_horizon",
            horizon=12,
        )

    assign = _assign_ratio_splits(
        pairs,
        end_times,
        0.7,
        0.1,
        0.2,
        per_asset=True,
        split_policy="contiguous",
        horizon=12,
    )

    assert (assign == 0).sum() > 0
    assert (assign == 1).sum() > 0
    assert (assign == 2).sum() > 0


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


def test_loader_coverage_zero_preserves_context_observations(tmp_path):
    data_dir, _, _ = _write_tiny_cache(tmp_path, length=40, window=2, horizon=3)
    train_dl, _, _, _ = load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        batch_size=1,
        n_entities=2,
        norm_scope="cache",
        shuffle_train=False,
        date_batching=False,
        window=2,
        horizon=3,
        split_policy="contiguous",
        exact_timestamp_batches=True,
        coverage=0.0,
    )

    (V, _), _, meta = next(iter(train_dl))
    present = meta["entity_mask"][0]

    assert meta["x_obs_mask"][0, present].all()
    assert torch.count_nonzero(V[0, present]).item() > 0


def test_loader_coverage_hides_context_only_and_is_deterministic(tmp_path):
    data_dir, _, _ = _write_tiny_cache(tmp_path, length=40, window=3, horizon=3)

    def first_batch():
        train_dl, _, _, _ = load_dataloaders_with_ratio_split(
            data_dir=str(data_dir),
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            batch_size=1,
            n_entities=2,
            norm_scope="cache",
            shuffle_train=False,
            date_batching=False,
            window=3,
            horizon=3,
            split_policy="contiguous",
            exact_timestamp_batches=True,
            coverage=0.5,
            seed=123,
        )
        return next(iter(train_dl))

    (V1, _), _, meta1 = first_batch()
    (V2, _), _, meta2 = first_batch()
    present = meta1["entity_mask"][0]
    x_obs = meta1["x_obs_mask"][0, present]
    hidden = ~x_obs

    assert torch.equal(meta1["x_obs_mask"], meta2["x_obs_mask"])
    assert torch.equal(V1, V2)
    assert int(x_obs.sum().item()) == 3
    assert int(hidden.sum().item()) == 3
    assert torch.all(V1[0, present][hidden] == 0.0)
    assert meta1["y_obs_mask"][0, present].all()


def test_loader_coverage_preserves_sparse_missingness_and_target_metadata(tmp_path):
    data_dir, _, _ = _write_tiny_cache(tmp_path, num_assets=1, length=12, window=4, horizon=3)
    paths = CachePaths.from_dir(data_dir)
    obs = np.load(paths.obs_masks / "0.npy")
    obs[0, 0] = False
    obs[2, 1] = False
    np.save(paths.obs_masks / "0.npy", obs)
    np.save(paths.fill_masks / "0.npy", obs)

    def first_batch(coverage: float):
        train_dl, _, _, _ = load_dataloaders_with_ratio_split(
            data_dir=str(data_dir),
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
            batch_size=1,
            n_entities=1,
            norm_scope="cache",
            shuffle_train=False,
            date_batching=False,
            window=4,
            horizon=3,
            split_policy="contiguous",
            exact_timestamp_batches=True,
            coverage=coverage,
            seed=99,
        )
        return next(iter(train_dl))

    (V0, _), y0, meta0 = first_batch(0.0)
    (Vh, _), yh, metah = first_batch(0.5)

    assert torch.equal(meta0["x_obs_mask"][0, 0], torch.tensor([[False, True], [True, True], [True, False], [True, True]]))
    assert torch.equal(metah["x_obs_mask"] & ~meta0["x_obs_mask"], torch.zeros_like(metah["x_obs_mask"]))
    assert torch.count_nonzero(meta0["x_obs_mask"] & ~metah["x_obs_mask"]).item() > 0
    hidden = meta0["x_obs_mask"] & ~metah["x_obs_mask"]
    assert torch.all(Vh[hidden] == 0.0)
    assert torch.equal(y0, yh)
    assert torch.equal(meta0["y_obs_mask"], metah["y_obs_mask"])
    assert torch.equal(meta0["delta_t"], metah["delta_t"])
    assert torch.equal(meta0["delta_t_y"], metah["delta_t_y"])


@pytest.mark.parametrize("coverage", [-0.1, 1.0])
def test_loader_coverage_rejects_invalid_rates(tmp_path, coverage):
    data_dir, _, _ = _write_tiny_cache(tmp_path, length=40, window=2, horizon=3)

    with pytest.raises(ValueError, match="coverage"):
        load_dataloaders_with_ratio_split(
            data_dir=str(data_dir),
            date_batching=False,
            window=2,
            horizon=3,
            split_policy="contiguous",
            coverage=coverage,
        )


def test_keep_time_meta_none_preserves_distinct_date_rows(tmp_path):
    data_dir, _, _ = _write_tiny_cache(tmp_path, num_assets=1, length=10, window=2, horizon=2)
    paths = CachePaths.from_dir(data_dir)
    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    meta["keep_time_meta"] = "none"
    paths.meta.write_text(json.dumps(meta), encoding="utf-8")

    train_dl, _, _, _ = load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        batch_size=8,
        n_entities=1,
        norm_scope="cache",
        shuffle_train=False,
        date_batching=True,
        dates_per_batch=2,
        window=2,
        horizon=2,
        split_policy="contiguous",
        exact_timestamp_batches=True,
    )

    _, y, meta_batch = next(iter(train_dl))

    assert y.shape == (2, 1, 2)
    assert meta_batch["entity_mask"].tolist() == [[True], [True]]
    assert meta_batch["context_end_time_keys"].shape == (2,)
    assert torch.unique(meta_batch["context_end_time_keys"]).numel() == 2


def test_loader_construction_does_not_mutate_meta_json(tmp_path):
    data_dir, _, _ = _write_tiny_cache(tmp_path, num_assets=1, length=10, window=2, horizon=2)
    paths = CachePaths.from_dir(data_dir)
    meta = json.loads(paths.meta.read_text(encoding="utf-8"))
    meta.pop("native_time_scale", None)
    meta.pop("native_time_scale_seconds", None)
    paths.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    before = paths.meta.read_bytes()

    load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        batch_size=1,
        n_entities=1,
        norm_scope="cache",
        shuffle_train=False,
        date_batching=False,
        window=2,
        horizon=2,
        split_policy="contiguous",
        exact_timestamp_batches=True,
    )

    assert paths.meta.read_bytes() == before


def test_panel_coverage_preserves_dense_date_filter(tmp_path):
    data_dir, pairs, end_times = _write_tiny_cache(tmp_path, num_assets=3, length=50, window=2, horizon=3)
    days = end_times.astype("datetime64[D]")
    partial_day = np.unique(days)[-1]
    keep = ~((pairs[:, 0] == 2) & (days == partial_day))
    paths = CachePaths.from_dir(data_dir)
    np.save(paths.windows / "global_pairs.npy", pairs[keep])
    np.save(paths.windows / "end_times.npy", end_times[keep])

    low = load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        batch_size=1,
        n_entities=3,
        date_batching=True,
        dates_per_batch=1,
        window=2,
        horizon=3,
        split_policy="contiguous",
        exact_timestamp_batches=False,
        coverage_per_window=2 / 3,
    )
    high = load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        batch_size=1,
        n_entities=3,
        date_batching=True,
        dates_per_batch=1,
        window=2,
        horizon=3,
        split_policy="contiguous",
        exact_timestamp_batches=False,
        coverage_per_window=1.0,
    )

    assert sum(low[3]) > sum(high[3])


def test_query_grid_incompatible_assets_do_not_share_joint_row(tmp_path):
    data_dir = _write_query_grid_cache(
        tmp_path,
        future_grids=[(1.0, 2.0, 3.0), (1.0, 4.0, 5.0)],
    )
    train_dl = _query_grid_train_loader(data_dir)

    rows = []
    for _, _, meta in train_dl:
        rows.extend(_valid_query_grids(meta))

    assert len(rows) == 2
    assert all(row.shape[0] == 1 for row in rows)
    assert {tuple(row[0].tolist()) for row in rows} == {
        (1.0, 2.0, 3.0),
        (1.0, 4.0, 5.0),
    }


def test_query_grid_aligned_assets_still_co_batch(tmp_path):
    data_dir = _write_query_grid_cache(
        tmp_path,
        future_grids=[(1.0, 4.0, 5.0), (1.0, 4.0, 5.0)],
    )
    train_dl = _query_grid_train_loader(data_dir)

    rows = []
    for _, _, meta in train_dl:
        rows.extend(_valid_query_grids(meta))

    assert len(rows) == 1
    assert rows[0].shape == (2, 3)
    assert torch.allclose(rows[0], torch.tensor([[1.0, 4.0, 5.0], [1.0, 4.0, 5.0]]))


def test_query_grid_batching_preserves_multi_target_channels(tmp_path):
    data_dir = _write_query_grid_cache(
        tmp_path,
        future_grids=[(1.0, 2.0, 3.0), (1.0, 4.0, 5.0)],
    )
    train_dl = _query_grid_train_loader(data_dir, target_cols=("noise", "target"))

    rows = []
    targets = []
    masks = []
    entity_masks = []
    for _, y, meta in train_dl:
        rows.extend(_valid_query_grids(meta))
        targets.append(y)
        masks.append(meta["y_obs_mask"])
        entity_masks.append(meta["entity_mask"])

    y_all = torch.cat(targets, dim=0)
    mask_all = torch.cat(masks, dim=0)
    entity_all = torch.cat(entity_masks, dim=0)
    assert len(rows) == 2
    assert all(row.shape[0] == 1 for row in rows)
    assert y_all.shape == (2, 2, 3, 2)
    assert mask_all.shape == y_all.shape
    assert torch.equal(mask_all, entity_all[:, :, None, None].expand_as(mask_all))
    for row in y_all:
        valid = row.abs().sum(dim=(1, 2)) > 0
        active = row[valid]
        assert active.shape == (1, 3, 2)
        assert torch.allclose(active[..., 0], active[..., 1])


def test_coverage_per_window_is_applied_to_query_grid_compatible_groups(tmp_path):
    data_dir = _write_query_grid_cache(
        tmp_path,
        future_grids=[(1.0, 4.0, 5.0), (1.0, 4.0, 5.0), (1.0, 2.0, 3.0)],
    )
    train_dl = _query_grid_train_loader(data_dir, coverage_per_window=2 / 3)

    rows = []
    for _, _, meta in train_dl:
        rows.extend(_valid_query_grids(meta))

    assert len(rows) == 1
    assert rows[0].shape == (2, 3)
    assert torch.allclose(rows[0], torch.tensor([[1.0, 4.0, 5.0], [1.0, 4.0, 5.0]]))


def test_manual_incompatible_query_grid_metadata_raises_clear_guard():
    from llapdiffusion.trainers import train_val_llapdiff as tv

    meta = {
        "delta_t_y": torch.tensor(
            [[[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]]],
            dtype=torch.float32,
        )
    }
    entity_mask = torch.tensor([[True, True]])

    with pytest.raises(ValueError, match="delta_t_y.*same query grid"):
        tv._flatten_dt(meta, entity_mask, torch.device("cpu"), key="delta_t_y")


def test_target_override_selects_non_default_feature_and_keeps_scalar_shape(tmp_path):
    data_dir, _, _ = _write_tiny_cache(tmp_path, length=40, window=2, horizon=3)
    train_dl, _, _, _ = load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        batch_size=1,
        n_entities=2,
        norm_scope="cache",
        shuffle_train=False,
        date_batching=False,
        window=2,
        horizon=3,
        split_policy="contiguous",
        exact_timestamp_batches=True,
        target_col="noise",
    )

    (_, _), y, meta = next(iter(train_dl))

    assert y.shape == (1, 2, 3)
    assert meta["y_obs_mask"].shape == y.shape
    assert torch.equal(y[0, 0], torch.tensor([2.0, 3.0, 4.0]))


def test_finance_calendar_feature_cannot_be_selected_as_target():
    meta = {
        "dataset": "fin_dataset",
        "feature_cols": ["RET_CLOSE", "DOW_SIN", "DOW_COS"],
        "target_col": "RET_CLOSE",
        "calendar_feature_cols": ["DOW_SIN", "DOW_COS"],
    }

    with pytest.raises(ValueError, match="calendar feature"):
        resolve_target_selection(meta, "DOW_SIN")
    assert valid_scalar_target_cols(meta) == ("RET_CLOSE",)


def test_calendar_feature_rejection_is_metadata_driven():
    meta = {
        "dataset": "custom_cache",
        "feature_cols": ["RET_CLOSE", "RET_OPEN", "DOW_SIN"],
        "target_cols": ["RET_CLOSE"],
        "calendar_feature_cols": ["DOW_SIN"],
    }

    with pytest.raises(ValueError, match="calendar feature"):
        resolve_target_selection(meta, requested_target_cols=["RET_OPEN", "DOW_SIN"])


def test_multi_target_selection_reports_indices_and_dimension():
    meta = {
        "dataset": "demo",
        "feature_cols": ["x", "y", "z"],
        "target_col": "x",
    }

    selected = resolve_target_selection(meta, requested_target_cols=["z", "y"])

    assert selected.target_col == "z"
    assert selected.target_cols == ("z", "y")
    assert selected.target_indices == (2, 1)
    assert selected.target_dim == 2
    assert selected.target_source == "feature_columns"


def test_scalar_target_col_rejects_comma_list():
    meta = {"dataset": "demo", "feature_cols": ["x", "y"], "target_col": "x"}

    with pytest.raises(ValueError, match="target_col accepts exactly one"):
        resolve_target_selection(meta, "x,y")


def test_target_index_accepts_single_target_cols_metadata():
    info = {"dataset": "demo", "feature_cols": ["x", "y"], "target_cols": ["y"]}

    assert target_index(info) == 1

    selected = resolve_target_selection({"dataset": "demo", "feature_cols": ["x", "y"], "target_col": "y"})
    assert selected.target_col == "y"
    assert selected.target_source == "cache_target"


def test_pack_targets_tokens_supports_multi_target_shape():
    from llapdiffusion.models.llapdiff_utils import pack_targets_tokens, target_time_observed

    y = torch.tensor([[[[1.0, 2.0], [3.0, float("nan")]]]])
    entity_mask = torch.tensor([[True]])
    y_obs_mask = torch.tensor([[[[True, True], [True, False]]]])

    x_tok, entity_pad, obs = pack_targets_tokens(y, entity_mask, torch.device("cpu"), y_obs_mask=y_obs_mask)

    assert x_tok.shape == (1, 2, 1, 4)
    assert entity_pad.shape == (1, 1)
    assert obs.shape == (1, 2, 1, 2)
    assert target_time_observed(obs).tolist() == [[True, True]]
    assert torch.equal(x_tok[..., :2], torch.tensor([[[[1.0, 2.0]], [[3.0, 0.0]]]]))


def test_latent_vae_decodes_multi_target_channels():
    from llapdiffusion.latent_space.latent_vae import LatentVAE

    model = LatentVAE(
        seq_len=2,
        latent_dim=8,
        latent_channel=4,
        enc_layers=1,
        enc_heads=2,
        enc_ff=16,
        dec_layers=1,
        dec_heads=2,
        dec_ff=16,
        input_dim=4,
        output_dim=2,
        dropout=0.0,
    )
    x_tok = torch.randn(1, 2, 3, 4)
    x_tok[..., 2:] = 1.0
    entity_pad = torch.tensor([[False, False, True]])

    x_hat, mu, logvar = model(x_tok, entity_pad)

    assert x_hat.shape == (1, 2, 3, 2)
    assert mu.shape == (1, 2, 4)
    assert logvar.shape == (1, 2, 4)


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


def test_target_context_rejects_feature_width_mismatch():
    V = torch.ones(1, 1, 2, 1)
    T = torch.zeros_like(V)
    y = torch.ones(1, 1, 1)
    meta = {
        "x_obs_mask": torch.ones_like(V, dtype=torch.bool),
        "y_obs_mask": torch.ones_like(y, dtype=torch.bool),
        "entity_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    info = {"dataset": "demo", "feature_cols": ["x", "target"], "target_col": "target"}

    with pytest.raises(ValueError, match="outside the input feature width"):
        target_context(((V, T), y, meta), info)


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
        assert "target_col" in signature.parameters


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
        assert "target_col" in signature.parameters


def test_physionet_public_wrapper_defaults_to_legacy_split():
    from llapdiffusion.configs.dataset_defaults import get_dataset_preset
    from llapdiffusion.datasets.physionet_cinc_dataset import (
        load_physionet_dataloaders_with_ratio_split,
        run_experiment,
    )

    preset = get_dataset_preset("physionet")

    assert preset.split_policy == "contiguous"
    assert preset.split_scope == "physionet_patient_relative_time"
    assert inspect.signature(run_experiment).parameters["split_policy"].default == "contiguous"
    assert (
        inspect.signature(load_physionet_dataloaders_with_ratio_split)
        .parameters["split_policy"]
        .default
        == "contiguous"
    )


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


def test_synthetic_public_path_accepts_induced_context_missingness():
    from llapdiffusion.tools import run_synthetic_regime_shift

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
            coverage=0.2,
        ),
    )

    assert cfg.COVERAGE == 0.2


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
        COVERAGE=0.25,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        split_policy="global_purged_horizon",
        exact_timestamp_batches=True,
        TARGET_COL="ozone",
    )

    assert pipeline.prepare_dataloaders(config=cfg)[3] == (1, 2, 3)
    assert seen["split_policy"] == "global_purged_horizon"
    assert seen["exact_timestamp_batches"] is True
    assert seen["target_col"] == "ozone"
    assert seen["coverage"] == 0.25


@pytest.mark.parametrize(
    "module_name",
    [
        "llapdiffusion.trainers.train_val_latent",
        "llapdiffusion.trainers.train_val_summarizer",
        "llapdiffusion.trainers.train_val_llapdiff",
    ],
)
def test_trainer_fallback_loaders_forward_target_col(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    seen = {}

    def fake_resolve_run_experiment(data_dir):
        seen["data_dir"] = data_dir

        def fake_run_experiment(**kwargs):
            seen.update(kwargs)
            return object(), object(), object(), (1, 2, 3)

        return fake_run_experiment

    monkeypatch.setattr(module, "resolve_run_experiment", fake_resolve_run_experiment)
    cfg = SimpleNamespace(
        DATA_DIR="demo-cache",
        date_batching=True,
        DATES_PER_BATCH=2,
        WINDOW=4,
        PRED=2,
        COVERAGE=0.0,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        split_policy="global_purged_horizon",
        exact_timestamp_batches=True,
        TARGET_COL="alt",
    )

    module._ensure_loaders(None, None, None, None, config=cfg)

    assert seen["data_dir"] == "demo-cache"
    assert seen["target_col"] == "alt"


def test_target_mask_excludes_missing_targets_even_when_entity_present():
    y = torch.tensor([[[1.0, 2.0, 0.0]]])
    meta = {
        "entity_mask": torch.ones(1, 1, dtype=torch.bool),
        "y_obs_mask": torch.tensor([[[True, False, True]]]),
    }

    mask = target_mask(meta, y)

    assert mask.tolist() == [[[True, False, True]]]
