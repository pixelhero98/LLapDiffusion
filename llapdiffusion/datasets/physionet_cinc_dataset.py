"""PhysioNet 2012 (CinC Challenge) dataset utilities.

This module mirrors the public API provided by :mod:`llapdiffusion.datasets.fin_dataset` and
the auxiliary air-quality datasets so that the existing training pipeline can
consume intensive-care unit (ICU) time-series data without additional glue
code.  The helpers create the same compact cache layout used throughout the
repository: per-entity feature matrices stored as ``float16`` together with a
global window index.  Each entity corresponds to a single patient admission and
the target series ``Y`` represents future values of a chosen vital sign stored
under the unified column name ``future_vital``.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple
from numpy.lib.stride_tricks import sliding_window_view

import numpy as np
import pandas as pd
import requests

from llapdiffusion.configs.dataset_archives import extract_zip_safely

from llapdiffusion.datasets._types import PathLike
from llapdiffusion.datasets._normalization import NormalizationStatsAccumulator
from llapdiffusion.datasets.fin_dataset import (
    CachePaths,
    _validate_context_missingness_rate,
    load_dataloaders_with_ratio_split as _load_fin_ratio_split,
    rebuild_window_index_only as _rebuild_window_index_only,
)
# ---------------------------------------------------------------------------
# Public constants & configuration helpers

DATA_URLS = {
    "set-a": "https://physionet.org/static/published-projects/challenge-2012/"
    "physionet-challenge-2012-set-a-1.0.0.zip",
    "set-b": "https://physionet.org/static/published-projects/challenge-2012/"
    "physionet-challenge-2012-set-b-1.0.0.zip",
    "set-c": "https://physionet.org/static/published-projects/challenge-2012/"
    "physionet-challenge-2012-set-c-1.0.0.zip",
}

OUTCOME_FILENAMES = {
    "set-a": "Outcomes-a.txt",
    "set-b": "Outcomes-b.txt",
    "set-c": "Outcomes-c.txt",
}

TARGET_COLUMN = "HR"

# PhysioNet measurements are recorded every hour for the challenge; align to that by default.
DEFAULT_FREQ = "1h"

# Keep the cache bounded so ``run_experiment`` can rebuild smaller indices cheaply.
MAX_WINDOW = 24
MAX_HORIZON = 12


@dataclass
class PhysioNetCacheConfig:
    """Configuration used when building the compact PhysioNet cache."""

    window: int = MAX_WINDOW
    horizon: int = MAX_HORIZON
    data_dir: PathLike = "./physionet_cinc_cache"
    raw_data_dir: Optional[PathLike] = "./physionet_cinc_raw"
    subset: str = "set-a"
    target_parameter: str = "HR"
    parameters: Optional[Sequence[str]] = None
    normalize_per_patient: bool = True
    clamp_sigma: float = 5.0
    keep_time_meta: str = "end"  # "full" | "end" | "none"
    freq: str = DEFAULT_FREQ
    min_minutes: int = 60
    min_coverage: float = 0.6
    max_patients: Optional[int] = None
    include_outcomes: bool = False
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Downloading & ingestion utilities

def download_physionet_cinc_dataset(dest_path="./physionet_cinc_raw", subset="set-a") -> Path:
    """Return the local directory containing the requested PhysioNet split.

    This helper is intentionally conservative about network access: if the
    files already exist locally (either as a standalone split folder or as part
    of the full project archive), we will reuse them instead of re-downloading.
    """

    subset_key = subset.lower()
    if subset_key not in DATA_URLS:
        raise ValueError(f"Unknown subset '{subset}'. Expected one of {sorted(DATA_URLS)}.")

    dest = Path(dest_path).expanduser().resolve()

    def _has_patient_txt(folder: Path) -> bool:
        return any(
            p for p in folder.glob("*.txt")
            if not p.name.lower().startswith("outcomes")
        )

    # --- Case 1: files are already in dest/<subset>/ ---
    direct_subset = dest / subset_key
    if direct_subset.exists() and _has_patient_txt(direct_subset):
        outcomes = direct_subset / OUTCOME_FILENAMES[subset_key]
        if not outcomes.exists():
            for root in (direct_subset.parent, dest):
                src = root / OUTCOME_FILENAMES[subset_key]
                if src.exists():
                    shutil.copy2(src, outcomes)
                    break
        return direct_subset

    # --- Case 2: user extracted the *full project* archive under dest/ ---
    # e.g., dest/<project_name>/<subset>/
    for candidate in sorted(dest.rglob(subset_key)):
        if not candidate.is_dir():
            continue
        if _has_patient_txt(candidate):
            outcomes = candidate / OUTCOME_FILENAMES[subset_key]
            if not outcomes.exists():
                for root in (candidate.parent, dest):
                    src = root / OUTCOME_FILENAMES[subset_key]
                    if src.exists():
                        shutil.copy2(src, outcomes)
                        break
            return candidate

    # --- Fall back: download the split zip and extract into dest/ ---
    dest.mkdir(parents=True, exist_ok=True)
    url = DATA_URLS[subset_key]
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk: tmp_file.write(chunk)
        tmp_path = tmp_file.name
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            extract_zip_safely(zf, dest)
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

    subset_dir = dest / subset_key
    outcomes = subset_dir / OUTCOME_FILENAMES[subset_key]
    if not outcomes.exists():
        # The split zip may not include the outcomes file; try common roots.
        for root in (subset_dir.parent, dest):
            src = root / OUTCOME_FILENAMES[subset_key]
            if src.exists():
                shutil.copy2(src, outcomes)
                break

    if not subset_dir.exists() or not _has_patient_txt(subset_dir):
        raise FileNotFoundError(
            f"Patient time-series files for '{subset_key}' not found after extraction into '{dest}'."
        )

    if not outcomes.exists():
        raise FileNotFoundError(
            f"Outcome file '{OUTCOME_FILENAMES[subset_key]}' not found under '{subset_dir}' (or '{dest}')."
        )

    return subset_dir



def _read_patient_file(path: Path) -> Tuple[str, pd.DataFrame]:
    df = pd.read_csv(path)

    # PhysioNet official format: Time,Parameter,Value
    if set(df.columns) == {"Time", "Parameter", "Value"}:
        rid_row = df.loc[df["Parameter"] == "RecordID", "Value"]
        record_id = str(rid_row.iloc[0]) if not rid_row.empty else path.stem
        df = df[df["Parameter"] != "RecordID"].copy()

    # Alternate format (some preprocessed dumps): RecordID,Time,Parameter,Value
    elif {"RecordID", "Time", "Parameter", "Value"}.issubset(df.columns):
        record_id = str(df["RecordID"].iloc[0])

    else:
        raise ValueError(f"Unexpected columns in '{path}': {list(df.columns)}")

    # -1 means missing in this dataset → treat as NaN
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df.loc[df["Value"] < 0, "Value"] = np.nan  # important

    time_delta = pd.to_timedelta(df["Time"].astype(str) + ":00")
    df["minutes"] = (time_delta / pd.Timedelta(minutes=1)).astype(int)

    pivot = df.pivot_table(index="minutes", columns="Parameter", values="Value", aggfunc="last")
    pivot = pivot.sort_index()
    observed_mask = pivot.notna()

    origin = pd.Timestamp("2000-01-01")
    pivot.index = origin + pd.to_timedelta(pivot.index, unit="m")
    observed_mask.index = pivot.index
    pivot = pivot.ffill().dropna(how="all")
    observed_mask = observed_mask.reindex(pivot.index).fillna(False)
    pivot.attrs["observed_mask"] = observed_mask.astype(bool)

    return record_id, pivot



def load_physionet_patient_panels(
    data_dir: PathLike,
    *,
    parameters: Optional[Sequence[str]] = None,
    freq: str = DEFAULT_FREQ,
    min_minutes: int = 60,
    min_coverage: float = 0.6,
    max_patients: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Load and align per-patient panels from raw PhysioNet text files."""

    root = Path(data_dir)
    txt_files = sorted(p for p in root.glob("*.txt") if not p.name.lower().startswith("outcomes"))
    if max_patients is not None:
        txt_files = txt_files[: int(max_patients)]

    panels: Dict[str, pd.DataFrame] = {}
    keep_params = set(str(p) for p in parameters) if parameters is not None else None

    for file_path in txt_files:
        record_id, pivot = _read_patient_file(file_path)

        if keep_params is not None:
            cols = [c for c in pivot.columns if c in keep_params]
            pivot = pivot[cols]

        if pivot.empty or pivot.shape[0] < 2:
            continue

        # Regularize to the requested sampling frequency.
        if freq:
            start, end = pivot.index.min(), pivot.index.max()
            range_minutes = (end - start) / pd.Timedelta(minutes=1)
            if range_minutes < max(min_minutes, 1):
                continue
            full_index = pd.date_range(start, end, freq=freq)
            observed_mask = pivot.attrs.get("observed_mask")
            if isinstance(observed_mask, pd.DataFrame):
                observed_mask = observed_mask.reindex(full_index).fillna(False)
            else:
                observed_mask = pivot.reindex(full_index).notna()
            pivot = pivot.reindex(full_index)
        else:
            observed_mask = pivot.attrs.get("observed_mask")
            if isinstance(observed_mask, pd.DataFrame):
                observed_mask = observed_mask.reindex(pivot.index).fillna(False)
            else:
                observed_mask = pivot.notna()
        pivot = pivot.ffill()
        keep_rows = pivot.notna().sum(axis=1) >= int(np.ceil(min_coverage * pivot.shape[1]))
        pivot = pivot.loc[keep_rows]
        observed_mask = observed_mask.reindex(pivot.index).fillna(False)

        if pivot.empty:
            continue

        out = pivot.astype(np.float32)
        out.attrs["observed_mask"] = observed_mask[pivot.columns].astype(bool)
        panels[record_id] = out

    if not panels:
        raise RuntimeError(
            "No valid patient panels were produced. Check the frequency/min_coverage configuration."
        )

    return panels


def load_physionet_outcomes(path: PathLike) -> pd.DataFrame:
    """Load static outcome information associated with each patient."""

    df = pd.read_csv(path)
    if "RecordID" not in df.columns:
        raise ValueError(f"Outcome file '{path}' is missing the 'RecordID' column.")
    df["RecordID"] = df["RecordID"].astype(str)
    df = df.set_index("RecordID")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Cache preparation

def _validate_window_and_horizon(window: int, horizon: int) -> Tuple[int, int]:
    """Clamp obvious configuration errors early and return normalized values."""

    if window <= 0:
        raise ValueError("window must be a positive integer")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if window > MAX_WINDOW:
        raise ValueError(f"window={window} exceeds the supported maximum of {MAX_WINDOW}.")
    if horizon > MAX_HORIZON:
        raise ValueError(f"horizon={horizon} exceeds the supported maximum of {MAX_HORIZON}.")
    return int(window), int(horizon)


def _ensure_future_vital_target(panel: pd.DataFrame, target_parameter: str) -> pd.DataFrame:
    observed_mask = panel.attrs.get("observed_mask")
    if isinstance(observed_mask, pd.DataFrame):
        observed_mask = observed_mask.reindex(panel.index).fillna(False)
    if target_parameter not in panel.columns:
        panel[target_parameter] = np.nan
        if isinstance(observed_mask, pd.DataFrame):
            observed_mask[target_parameter] = False

    target = panel[target_parameter].astype(np.float32)
    target_observed = (
        observed_mask[target_parameter].astype(bool)
        if isinstance(observed_mask, pd.DataFrame) and target_parameter in observed_mask.columns
        else target.notna()
    )
    out = panel.drop(columns=[target_parameter], errors="ignore")
    out[TARGET_COLUMN] = target  # keep NaNs for masked training
    if isinstance(observed_mask, pd.DataFrame):
        obs_out = observed_mask.drop(columns=[target_parameter], errors="ignore")
        obs_out[TARGET_COLUMN] = target_observed
        out.attrs["observed_mask"] = obs_out.reindex(index=out.index, columns=out.columns).fillna(False).astype(bool)
    return out




def prepare_physionet_cinc_cache(cfg: PhysioNetCacheConfig) -> Mapping[str, object]:
    """Prepare the compact cache using the shared cache layout.

    Notes
    -----
    PhysioNet panels are irregular and may contain missing values even after
    alignment. Since the downstream pipeline supports masked training, we keep
    NaNs in both features and targets, and only require that each patient has
    at least one observed target value.
    """

    window, horizon = _validate_window_and_horizon(cfg.window, cfg.horizon)

    keep_time_meta = cfg.keep_time_meta.lower()
    if keep_time_meta not in {"full", "end", "none"}:
        raise ValueError("keep_time_meta must be one of {'full', 'end', 'none'}")

    subset_dir = download_physionet_cinc_dataset(
        cfg.raw_data_dir or "./physionet_cinc_raw",
        cfg.subset,
    )

    # Ensure the target parameter is not accidentally dropped when parameter
    # filtering is enabled.
    load_params = None
    if cfg.parameters is not None:
        load_params = list(dict.fromkeys([*cfg.parameters, cfg.target_parameter]))

    panels = load_physionet_patient_panels(
        subset_dir,
        parameters=load_params,
        freq=cfg.freq,
        min_minutes=cfg.min_minutes,
        min_coverage=cfg.min_coverage,
        max_patients=cfg.max_patients,
    )

    outcomes: Optional[pd.DataFrame] = None
    if cfg.include_outcomes:
        outcomes = load_physionet_outcomes(subset_dir / OUTCOME_FILENAMES[cfg.subset.lower()])

    min_required = window + horizon

    # ------------------------------------------------------------------
    # Build per-patient frames and determine a shared feature set.

    raw_assets = sorted(panels.keys())
    feature_frames: Dict[str, pd.DataFrame] = {}
    common_feature_cols: Optional[Set[str]] = None

    for asset in raw_assets:
        panel = _ensure_future_vital_target(panels[asset].copy(), cfg.target_parameter)

        if outcomes is not None and asset in outcomes.index:
            observed_mask = panel.attrs.get("observed_mask")
            if not isinstance(observed_mask, pd.DataFrame):
                observed_mask = panel.notna()
            static_values = outcomes.loc[asset]
            for col, value in static_values.items():
                panel[f"OUTCOME_{col.upper()}"] = np.float32(value) if pd.notna(value) else np.nan
                observed_mask[f"OUTCOME_{col.upper()}"] = bool(pd.notna(value))
            # Forward-fill only to keep broadcast static covariates constant.
            panel = panel.ffill()
            panel.attrs["observed_mask"] = (
                observed_mask.reindex(index=panel.index, columns=panel.columns).fillna(False).astype(bool)
            )

        panel = panel.dropna(how="all")
        observed_mask = panel.attrs.get("observed_mask")
        if isinstance(observed_mask, pd.DataFrame):
            panel.attrs["observed_mask"] = (
                observed_mask.reindex(index=panel.index, columns=panel.columns).fillna(False).astype(bool)
            )
        if panel.shape[0] < min_required:
            continue

        # Skip patients with no observed target value at all.
        target_vals = panel.get(TARGET_COLUMN)
        if target_vals is None:
            continue
        observed_mask = panel.attrs.get("observed_mask")
        if isinstance(observed_mask, pd.DataFrame) and TARGET_COLUMN in observed_mask.columns:
            has_observed_target = bool(observed_mask[TARGET_COLUMN].any())
        else:
            has_observed_target = bool(np.isfinite(target_vals.to_numpy(dtype=np.float32, copy=False)).any())
        if not has_observed_target:
            continue

        non_empty_cols = {c for c in panel.columns if not panel[c].isna().all()}
        feature_frames[asset] = panel.astype(np.float32)
        common_feature_cols = (
            non_empty_cols if common_feature_cols is None else common_feature_cols & non_empty_cols
        )

    if not feature_frames:
        raise RuntimeError(
            "No patient panels remaining after applying target/outcome processing. "
            "Try lowering min_coverage or disabling parameter filtering."
        )

    if not common_feature_cols:
        raise RuntimeError("No common feature columns remain across patient panels.")

    feature_cols = [TARGET_COLUMN] + sorted(c for c in common_feature_cols if c != TARGET_COLUMN)
    if TARGET_COLUMN not in feature_cols:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from feature columns.")

    # Recompute assets after filtering so ids/files are consistent.
    assets = sorted(feature_frames.keys())
    asset_to_id = {asset: idx for idx, asset in enumerate(assets)}

    # ------------------------------------------------------------------
    # Persist arrays and build the global window index.

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    paths = CachePaths.from_dir(data_dir)
    if cfg.overwrite and paths.cache_root.exists():
        shutil.rmtree(paths.cache_root)
    paths.ensure()

    pairs: List[np.ndarray] = []
    end_times: List[np.ndarray] = []
    start_times: List[np.datetime64] = []
    stop_times: List[np.datetime64] = []

    norm_acc = NormalizationStatsAccumulator(
        num_assets=len(assets),
        feature_dim=len(feature_cols),
        per_asset=cfg.normalize_per_patient,
    )

    for asset in assets:
        aid = asset_to_id[asset]
        panel = feature_frames[asset][feature_cols]

        if panel.shape[0] < min_required:
            # Should not happen due to earlier filtering, but keep robust.
            continue

        features = panel.to_numpy(dtype=np.float32, copy=True)
        targets = panel[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
        times = panel.index.to_numpy(dtype="datetime64[ns]")
        observed_mask = panel.attrs.get("observed_mask")
        if isinstance(observed_mask, pd.DataFrame):
            obs_mask = observed_mask.reindex(panel.index)[feature_cols].fillna(False).to_numpy(dtype=bool, copy=False)
        else:
            obs_mask = np.isfinite(features)
        fill_mask = np.isfinite(features)

        np.save(paths.features / f"{aid}.npy", features.astype(np.float16, copy=False))
        np.save(paths.targets / f"{aid}.npy", targets.astype(np.float16, copy=False))
        np.save(paths.times / f"{aid}.npy", times)
        np.save(paths.obs_masks / f"{aid}.npy", obs_mask)
        np.save(paths.fill_masks / f"{aid}.npy", fill_mask)

        # Normalization must ignore missing values downstream (masked training).
        norm_acc.update(aid, features, targets)

        max_start = panel.shape[0] - min_required + 1
        if max_start <= 0:
            continue
        starts = np.arange(0, max_start, dtype=np.int32)

        # Keep only windows with at least one observed target value in the forecast horizon.
        if horizon > 0:
            obs = obs_mask[:, feature_cols.index(TARGET_COLUMN)]
            future_obs = sliding_window_view(obs, window_shape=horizon)
            valid = future_obs[window : window + max_start].any(axis=1)
            starts = starts[valid]

        if starts.size == 0:
            continue

        window_end_times = times[starts + window - 1].astype("datetime64[ns]")
        pairs.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        end_times.append(window_end_times)
        start_times.append(times[0])
        stop_times.append(times[-1])

    if not pairs:
        raise RuntimeError("No valid context windows available across patients (target too sparse?).")

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)
    global_end_times = np.concatenate(end_times, axis=0).astype("datetime64[ns]")
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", global_end_times)

    norm_stats = norm_acc.finalize(assets)

    dataset_start = min(start_times).astype("datetime64[s]") if start_times else None
    dataset_end = max(stop_times).astype("datetime64[s]") if stop_times else None

    meta = {
        "dataset": "physionet_cinc",
        "format": "indexcache_v1",
        "subset": cfg.subset,
        "window": window,
        "horizon": horizon,
        "max_window": int(MAX_WINDOW),
        "max_horizon": int(MAX_HORIZON),
        "assets": assets,
        "asset2id": asset_to_id,
        "feature_cols": feature_cols,
        "target_col": TARGET_COLUMN,
        "target_column": TARGET_COLUMN,
        "target_parameter": cfg.target_parameter,
        "normalize_per_patient": bool(cfg.normalize_per_patient),
        "normalize_per_ticker": bool(cfg.normalize_per_patient),
        "clamp_sigma": float(cfg.clamp_sigma),
        "keep_time_meta": keep_time_meta,
        "freq": cfg.freq or DEFAULT_FREQ,
        "native_time_scale": cfg.freq or DEFAULT_FREQ,
        "native_time_scale_seconds": 3600.0,
        "start": str(dataset_start) if dataset_start is not None else None,
        "end": str(dataset_end) if dataset_end is not None else None,
    }

    with paths.meta.open("w") as f:
        json.dump(meta, f, indent=2)

    with paths.norm_stats.open("w") as f:
        json.dump(norm_stats, f, indent=2)

    return {"assets": assets, "feature_cols": feature_cols}


# ---------------------------------------------------------------------------
# Loader wrappers

def load_physionet_dataloaders_with_ratio_split(
    data_dir: PathLike,
    *,
    split_policy: str = "contiguous",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
    **loader_kwargs,
):
    """Wrapper around the financial ratio-split loader for PhysioNet datasets."""

    return _load_fin_ratio_split(
        data_dir=data_dir,
        split_policy=split_policy,
        exact_timestamp_batches=exact_timestamp_batches,
        target_col=target_col,
        target_cols=target_cols,
        **loader_kwargs,
    )


def _validate_physionet_cache(paths: CachePaths) -> Dict[str, object]:
    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at '{paths.meta}'. Did you run prepare_physionet_cinc_cache()?"
        )

    with paths.meta.open("r") as f:
        meta: Dict[str, object] = json.load(f)

    dataset = meta.get("dataset")
    if dataset not in {"physionet_cinc", "physionet_cinc_dataset"}:
        raise ValueError(
            f"The cache at '{paths.cache_root}' does not correspond to the PhysioNet CinC dataset."
        )

    return meta


def run_experiment(
    data_dir: PathLike,
    K: int,
    H: int,
    *,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    per_asset: bool = True,
    date_batching: bool = True,
    coverage: float = 0.0,
    panel_coverage: float = 0.0,
    dates_per_batch: int = 14,
    batch_size: int = 64,
    norm: str = "train_only",
    reindex: bool = True,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    split_policy: str = "contiguous",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
):
    """Mirror :func:`llapdiffusion.datasets.fin_dataset.run_experiment` for PhysioNet caches."""

    coverage = _validate_context_missingness_rate(coverage, name="coverage")
    paths = CachePaths.from_dir(data_dir)
    meta = _validate_physionet_cache(paths)
    base_window = min(int(meta.get("window", K)), MAX_WINDOW)
    base_horizon = min(int(meta.get("horizon", H)), MAX_HORIZON)
    max_window = min(int(meta.get("max_window", MAX_WINDOW)), MAX_WINDOW)
    max_horizon = min(int(meta.get("max_horizon", MAX_HORIZON)), MAX_HORIZON)

    # Sanity check: old caches might predate the tighter maxima.
    base_window, base_horizon = _validate_window_and_horizon(base_window, base_horizon)

    if K > max_window or H > max_horizon:
        raise ValueError(
            "Requested (window, horizon) exceed the cached configuration. "
            "Re-run prepare_physionet_cinc_cache with larger values first."
        )

    needs_reindex = K != base_window or H != base_horizon
    needs_target_reindex = target_col is not None or target_cols is not None

    if reindex and (needs_reindex or needs_target_reindex):
        _validate_window_and_horizon(K, H)
        _rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=needs_reindex,
            backup_old=False,
            target_col=target_col,
            target_cols=target_cols,
        )
        base_window, base_horizon = K, H

    elif not reindex and (K > base_window or H > base_horizon):
        raise ValueError(
            "Requested (window, horizon) exceed the currently indexed configuration. "
            "Enable reindex=True to rebuild within the cached maximums."
        )

    train_dl, val_dl, test_dl, lengths = load_physionet_dataloaders_with_ratio_split(
        data_dir=data_dir,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        batch_size=batch_size,
        regression=True,
        per_asset=per_asset,
        norm_scope=norm,
        date_batching=date_batching,
        coverage_per_window=panel_coverage,
        dates_per_batch=dates_per_batch,
        window=K,
        horizon=H,
        split_policy=split_policy,
        exact_timestamp_batches=exact_timestamp_batches,
        target_col=target_col,
        target_cols=target_cols,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        coverage=coverage,
    )

    return train_dl, val_dl, test_dl, lengths


__all__ = [
    "PhysioNetCacheConfig",
    "download_physionet_cinc_dataset",
    "load_physionet_patient_panels",
    "load_physionet_outcomes",
    "prepare_physionet_cinc_cache",
    "load_physionet_dataloaders_with_ratio_split",
    "run_experiment",
]
