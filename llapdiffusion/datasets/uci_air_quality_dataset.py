"""UCI Air Quality dataset utilities.

This module mirrors the public API provided by :mod:`llapdiffusion.datasets.fin_dataset` and
the other dataset wrappers living in llapdiffusion.datasets.  It prepares
the UCI Air Quality dataset using the compact "ratio index" cache format so the
existing training and evaluation pipelines can consume the data without any
additional glue code.

The dataset contains hourly air quality measurements collected by a single
monitoring station in an Italian city.  We model the future concentration of
``NO2(GT)`` (nitrogen dioxide) as the regression target while the context
features include the remaining pollutant and meteorological readings provided
by the original dataset.  Missing values are denoted with ``-200`` in the raw
files; the cleaning pipeline replaces those entries with ``NaN`` and preserves
missingness during resampling.  Values are forward-filled when possible and
rows with insufficient feature coverage are dropped, allowing masked training
to exploit the remaining ``NaN`` structure.  The processed panel is resampled
at one-hour intervals by default.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from numpy.lib.stride_tricks import sliding_window_view

from llapdiffusion.datasets._normalization import NormalizationStatsAccumulator
from llapdiffusion.datasets._types import PathLike
from llapdiffusion.datasets.fin_dataset import (
    CachePaths,
    _validate_context_missingness_rate,
    load_dataloaders_with_ratio_split as _load_fin_ratio_split,
    rebuild_window_index_only as _rebuild_window_index_only,
)
# ---------------------------------------------------------------------------
# Public constants & configuration helpers

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
ARCHIVE_CSV_NAME = "AirQualityUCI.csv"
TARGET_COLUMN = "NO2(GT)"
DEFAULT_FREQ = "1h"
MAX_LOOKBACK = 336
MAX_HORIZON = 168
DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
)
ASSET_NAME = "uci_air_monitor"


@dataclass
class UCIAirCacheConfig:
    """Configuration used when preparing the compact UCI cache."""

    window: int
    horizon: int
    data_dir: PathLike = "./uci_air_cache"
    raw_data_dir: Optional[PathLike] = "./uci_air_quality_raw"
    freq: str = DEFAULT_FREQ
    target_column: str = TARGET_COLUMN
    feature_columns: Optional[Sequence[str]] = None
    normalize_per_asset: bool = False
    min_coverage: float = 0.6
    clamp_sigma: float = 5.0
    keep_time_meta: str = "end"  # "full" | "end" | "none"
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Helpers


def _resolve_feature_columns(
    feature_columns: Optional[Sequence[str]],
    target_column: str,
) -> Tuple[str, ...]:
    """Return a clean feature column ordering with the target in front."""

    if feature_columns is None:
        cols = list(DEFAULT_FEATURE_COLUMNS)
    else:
        cols = [str(c) for c in feature_columns]

    if target_column not in cols:
        cols.insert(0, target_column)
    else:
        cols = [target_column] + [c for c in cols if c != target_column]

    return tuple(cols)


def _find_existing_member(root: Path, filenames: Sequence[str]) -> Optional[Path]:
    """Return the first existing file under ``root`` matching ``filenames``."""

    for name in filenames:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _normalize_column_name(name: object) -> str:
    """Return a canonical representation for raw Air Quality headers."""

    text = str(name).replace("\xa0", " ").replace("\u200b", "")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return text.strip()


def _parse_air_quality_timestamp(date: pd.Series, time: pd.Series) -> pd.Series:
    """Parse the raw ``Date``/``Time`` columns into timestamps.

    The UCI Air Quality files primarily represent timestamps using the
    ``"%d/%m/%Y %H.%M.%S"`` pattern, but some community re-publications swap the
    dot separators for colons.  ``pandas.to_datetime`` raises a warning when the
    format has to be inferred for every element, so we first try the canonical
    dotted format, then the colon-separated variant, and finally fall back to
    ``dateutil`` for any remaining edge-cases.  This preserves backwards
    stable parsing while avoiding noisy warnings.
    """

    date_str = date.astype(str).str.strip()
    time_str = time.astype(str).str.strip()
    combined = date_str.str.cat(time_str, sep=" ")

    timestamp = pd.to_datetime(combined, format="%d/%m/%Y %H.%M.%S", errors="coerce")

    missing_mask = timestamp.isna()
    if missing_mask.any():
        alt = pd.to_datetime(
            combined[missing_mask],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        timestamp.loc[missing_mask] = alt

    missing_mask = timestamp.isna()
    if missing_mask.any():
        timestamp.loc[missing_mask] = pd.to_datetime(
            combined[missing_mask],
            dayfirst=True,
            errors="coerce",
        )

    return timestamp


def download_uci_air_quality_dataset(
    dest_path: PathLike = "./uci_air_quality_raw",
    url: str = DATA_URL,
) -> Path:
    """Download the Air Quality dataset archive and extract the CSV file.

    Parameters
    ----------
    dest_path:
        Directory where the extracted CSV will be stored.  If the file already
        exists, it is reused as-is.
    url:
        Download URL for the UCI archive.
    """

    dest = Path(dest_path).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_member(dest, (ARCHIVE_CSV_NAME, "AirQualityUCI.xlsx"))
    if existing is not None:
        return existing

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            members = [m for m in zf.namelist() if m.lower().endswith(('.xlsx', '.xls'))]
        if not members:
            raise RuntimeError("Unable to locate a CSV/XLSX file inside the UCI archive.")

        member = members[0]
        target_path = dest / Path(member).name
        with zf.open(member) as source, target_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    return target_path


def load_raw_uci_air_quality(data_path: PathLike) -> pd.DataFrame:
    """Load the raw Air Quality dataset into a :class:`pandas.DataFrame`."""

    path = Path(data_path)
    if path.is_dir():
        existing = _find_existing_member(path, (ARCHIVE_CSV_NAME, "AirQualityUCI.xlsx"))
        if existing is None:
            raise FileNotFoundError(
                f"No AirQualityUCI CSV/XLSX file found in '{path}'. Did you download the dataset?"
            )
        path = existing

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            na_values=[-200, "-200"],
            parse_dates=False,
            engine="python",
        )
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, na_values=[-200])
    else:
        raise ValueError(f"Unsupported Air Quality file format: '{path.suffix}'.")

    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Some distributions of the UCI Air Quality CSV/XLSX files include trailing
    # whitespace in the column names (for example ``"NO2(GT) "``).  Downstream
    # processing expects the canonical feature names without any padding, so we
    # normalize the headers here before selecting the feature columns.  This
    # mirrors the behaviour of other public loaders and prevents confusing
    # ``KeyError`` exceptions when ``clean_uci_air_quality`` runs.
    df.columns = [_normalize_column_name(c) for c in df.columns]
    return df


def clean_uci_air_quality(
    df: pd.DataFrame,
    *,
    freq: str = DEFAULT_FREQ,
    feature_columns: Optional[Sequence[str]] = None,
    target_column: str = TARGET_COLUMN,
    min_coverage: float = 0.6,
) -> pd.DataFrame:
    """Clean the raw Air Quality dataframe and return a panel of features.

    The resulting dataframe is indexed by timestamps sampled at ``freq`` and the
    columns contain the numeric features used for training.  The target column
    is included in the returned frame to simplify downstream cache creation.
    """

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Raw UCI Air Quality dataframe must contain 'Date' and 'Time' columns.")

    frame = df.copy()
    timestamp = _parse_air_quality_timestamp(frame["Date"], frame["Time"])
    frame["datetime"] = timestamp
    frame = frame.dropna(subset=["datetime"])

    frame = frame.set_index("datetime").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    feature_cols = _resolve_feature_columns(feature_columns, target_column)
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns in UCI Air Quality data: {missing}")

    cols = list(feature_cols)  # coerce any sequence to list for pandas
    numeric = frame[cols].apply(pd.to_numeric, errors="coerce")

    if freq:
        resample_freq = freq.lower() if isinstance(freq, str) else freq
        numeric = numeric.resample(resample_freq).mean()

    observed_mask = numeric.notna()
    numeric = numeric.ffill()

    required = max(1, int(np.ceil(min_coverage * len(feature_cols))))
    keep_rows = numeric.notna().sum(axis=1) >= required
    numeric = numeric.loc[keep_rows]
    observed_mask = observed_mask.reindex(numeric.index).fillna(False)
    if target_column not in observed_mask.columns or not bool(observed_mask[target_column].any()):
        raise RuntimeError("No observed target values remain after cleaning the UCI Air Quality data.")
    if numeric.empty:
        raise RuntimeError("No usable rows remain after applying coverage filtering.")

    out = numeric.astype(np.float32)
    out.attrs["observed_mask"] = observed_mask[cols].astype(bool)
    return out




# ---------------------------------------------------------------------------
# Cache builder


def prepare_uci_air_cache(cfg: UCIAirCacheConfig) -> Mapping[str, List[str]]:
    """Prepare the compact cache with context features and nitrogen-dioxide targets."""

    if cfg.window <= 0:
        raise ValueError("window must be a positive integer")
    if cfg.horizon < 0:
        raise ValueError("horizon must be non-negative")
    if cfg.window > MAX_LOOKBACK:
        raise ValueError(
            f"window must be less than or equal to {MAX_LOOKBACK} for the UCI Air Quality dataset"
        )
    if cfg.horizon > MAX_HORIZON:
        raise ValueError(
            f"horizon must be less than or equal to {MAX_HORIZON} for the UCI Air Quality dataset"
        )

    keep_time_meta = cfg.keep_time_meta.lower()
    if keep_time_meta not in {"full", "end", "none"}:
        raise ValueError("keep_time_meta must be one of {'full', 'end', 'none'}")

    feature_cols = _resolve_feature_columns(cfg.feature_columns, cfg.target_column)

    raw_root = cfg.raw_data_dir or "./uci_air_quality_raw"
    raw_path = download_uci_air_quality_dataset(raw_root)
    raw_df = load_raw_uci_air_quality(raw_path)
    clean_df = clean_uci_air_quality(
        raw_df,
        freq=cfg.freq,
        feature_columns=feature_cols,
        target_column=cfg.target_column,
        min_coverage=cfg.min_coverage,
    )

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    paths = CachePaths.from_dir(data_dir)
    if cfg.overwrite and paths.cache_root.exists():
        shutil.rmtree(paths.cache_root)
    paths.ensure()

    assets = [ASSET_NAME]
    asset_to_id = {ASSET_NAME: 0}

    feature_cols = [cfg.target_column] + [c for c in clean_df.columns if c != cfg.target_column]

    panel = clean_df[feature_cols]
    total_rows = panel.shape[0]
    min_required = cfg.window + cfg.horizon
    if total_rows < min_required:
        raise ValueError("Insufficient history for the requested window/horizon configuration.")

    features = panel.to_numpy(dtype=np.float32, copy=True)
    targets = panel[cfg.target_column].to_numpy(dtype=np.float32, copy=True)
    times = panel.index.to_numpy(dtype="datetime64[ns]")
    observed_mask = clean_df.attrs.get("observed_mask")
    if isinstance(observed_mask, pd.DataFrame):
        obs_mask = observed_mask.reindex(panel.index)[feature_cols].fillna(False).to_numpy(dtype=bool, copy=False)
    else:
        obs_mask = np.isfinite(features)
    fill_mask = np.isfinite(features)

    np.save(paths.features / "0.npy", features.astype(np.float16, copy=False))
    np.save(paths.targets / "0.npy", targets.astype(np.float16, copy=False))
    np.save(paths.times / "0.npy", times)
    np.save(paths.obs_masks / "0.npy", obs_mask)
    np.save(paths.fill_masks / "0.npy", fill_mask)

    norm_acc = NormalizationStatsAccumulator(
        num_assets=len(assets),
        feature_dim=len(feature_cols),
        per_asset=cfg.normalize_per_asset,
    )
    norm_acc.update(0, features, targets)

    max_start = total_rows - min_required + 1
    if max_start <= 0:
        raise RuntimeError("No valid context windows available for the requested configuration.")

    starts = np.arange(0, max_start, dtype=np.int32)
    if cfg.horizon > 0:
        target_idx = feature_cols.index(cfg.target_column)
        obs = obs_mask[:, target_idx]
        future_obs = sliding_window_view(obs, window_shape=cfg.horizon)
        valid = future_obs[cfg.window : cfg.window + max_start].any(axis=1)
        starts = starts[valid]
    if starts.size == 0:
        raise RuntimeError("No valid context windows with observed targets in the forecast horizon.")
    window_end_times = times[starts + cfg.window - 1].astype("datetime64[ns]")
    global_pairs = np.stack([np.zeros_like(starts), starts], axis=1).astype(np.int32)
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", window_end_times)

    norm_stats = norm_acc.finalize(assets)

    dataset_start = times[0].astype("datetime64[s]") if times.size else None
    dataset_end = times[-1].astype("datetime64[s]") if times.size else None

    meta = {
        "dataset": "uci_air_quality",
        "format": "indexcache_v1",
        "assets": assets,
        "asset2id": asset_to_id,
        "feature_cols": feature_cols,
        "target_col": cfg.target_column,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "max_window": MAX_LOOKBACK,
        "max_horizon": MAX_HORIZON,
        "keep_time_meta": keep_time_meta,
        "normalize_per_ticker": cfg.normalize_per_asset,
        "clamp_sigma": cfg.clamp_sigma,
        "freq": cfg.freq,
        "native_time_scale": cfg.freq,
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


def load_uci_dataloaders_with_ratio_split(
    data_dir: PathLike,
    *,
    split_policy: str = "global_purged_horizon",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
    **loader_kwargs,
):
    """Wrapper around the financial ratio-split loader for the UCI dataset."""

    return _load_fin_ratio_split(
        data_dir=data_dir,
        split_policy=split_policy,
        exact_timestamp_batches=exact_timestamp_batches,
        target_col=target_col,
        target_cols=target_cols,
        **loader_kwargs,
    )


def _validate_uci_cache(paths: CachePaths) -> Dict[str, object]:
    """Read and validate the cache metadata, returning the parsed JSON."""

    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache meta file not found at '{paths.meta}'. Did you run prepare_uci_air_cache()?"
        )

    with paths.meta.open("r") as f:
        meta: Dict[str, object] = json.load(f)

    dataset = meta.get("dataset")
    if dataset not in {"uci_air_quality", "uci_air_dataset"}:
        raise ValueError(
            f"The cache located at '{paths.cache_root}' does not correspond to the UCI Air Quality dataset."
        )

    return meta


def run_experiment(
    data_dir: PathLike,
    K: int,
    H: int,
    *,
    ratios=(0.7, 0.1, 0.2),
    per_asset: bool = True,
    date_batching: bool = True,
    coverage: float = 0.0,
    panel_coverage: float = 0.0,
    dates_per_batch: int = 30,
    batch_size: int = 64,
    norm: str = "train_only",
    reindex: bool = True,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    split_policy: str = "global_purged_horizon",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
):
    """Build train/val/test loaders for the prepared UCI Air Quality cache."""

    coverage = _validate_context_missingness_rate(coverage, name="coverage")
    if K <= 0:
        raise ValueError("K (lookback window) must be a positive integer")
    if H < 0:
        raise ValueError("H (forecast horizon) must be non-negative")
    if K > MAX_LOOKBACK or H > MAX_HORIZON:
        raise ValueError(
            f"Requested (window, horizon) must not exceed ({MAX_LOOKBACK}, {MAX_HORIZON})."
        )

    paths = CachePaths.from_dir(data_dir)
    meta = _validate_uci_cache(paths)
    max_window = int(meta.get("max_window", MAX_LOOKBACK))
    max_horizon = int(meta.get("max_horizon", MAX_HORIZON))
    if K > max_window or H > max_horizon:
        raise ValueError(
            "Requested (window, horizon) exceed the cached configuration. "
            "Re-run prepare_uci_air_cache with larger values first."
        )

    base_window = int(meta.get("window", K))
    base_horizon = int(meta.get("horizon", H))

    needs_update = K != base_window or H != base_horizon
    if needs_update:
        _rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=True,
            backup_old=False,
            target_col=target_col,
            target_cols=target_cols,
        )
    elif reindex:
        _rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=False,
            backup_old=False,
            target_col=target_col,
            target_cols=target_cols,
        )

    train_dl, val_dl, test_dl, lengths = load_uci_dataloaders_with_ratio_split(
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
    "UCIAirCacheConfig",
    "download_uci_air_quality_dataset",
    "load_raw_uci_air_quality",
    "clean_uci_air_quality",
    "prepare_uci_air_cache",
    "load_uci_dataloaders_with_ratio_split",
    "run_experiment",
]
