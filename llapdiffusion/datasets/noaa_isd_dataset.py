"""NOAA ISD (Integrated Surface Database) dataset utilities.

This module mirrors the public API provided by :mod:`llapdiffusion.datasets.fin_dataset` and
:mod:`llapdiffusion.datasets.bms_air_dataset` so that downstream training code can reuse the
same dataloading pipeline across heterogeneous domains.  The helper functions
construct the compact cache layout used by :func:`fin_dataset.prepare_features_and_index_cache`
consisting of per-station feature/target matrices and a global window index.

The target series ``Y`` corresponds to future values of the ambient air
``temperature`` (degrees Celsius).  Context features ``X`` include the current
and historical readings for several meteorological variables such as dew point,
wind speed and sea level pressure.  Data are sampled hourly by default
(``1H``), the maximum supported lookback is 336 hours and the maximum horizon is
168 hours.  All arrays are written in float16 for storage efficiency;
normalization statistics are computed either globally or per station depending
on the configuration.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _initialise_isd_backend() -> Tuple[Optional[object], Optional[str]]:
    """Attempt to import an ISD client from supported third-party libraries."""

    candidates: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("isd_fetch", ("ISDFetch", "ISDClient", "Client", "client", "fetch")),
        ("isd_fetch.client", ("ISDFetch", "ISDClient", "Client")),
        ("pyisd", ("IsdLite", "ISDLite")),
    )

    for module_name, attr_names in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - optional dependency
            continue

        if not attr_names:
            return module, module_name

        for attr_name in attr_names:
            attr = getattr(module, attr_name, None)
            if attr is None:
                continue

            # Some libraries expose a factory/class that needs instantiating,
            # others expose a ready-to-use module-like object.
            if callable(attr):
                try:
                    instance = attr()  # type: ignore[misc]
                except TypeError:
                    try:
                        instance = attr(verbose=1)  # type: ignore[misc]
                    except TypeError:
                        # Treat as module-level namespace when we cannot
                        # determine the appropriate constructor signature.
                        return module, module_name
                    except Exception:  # pragma: no cover - best effort import
                        continue
                except Exception:  # pragma: no cover - best effort import
                    continue
                else:
                    return instance, module_name

            return attr, module_name

        return module, module_name

    return None, None


isd, _isd_backend_name = _initialise_isd_backend()

from llapdiffusion.datasets._normalization import NormalizationStatsAccumulator
from llapdiffusion.datasets._types import PathLike
from llapdiffusion.datasets.fin_dataset import (
    CachePaths,
    _validate_context_missingness_rate,
    load_dataloaders_with_ratio_split as _load_fin_ratio_split,
    rebuild_window_index_only as _rebuild_window_index_only,
)
TARGET_COLUMN = "temperature"
DEFAULT_FREQ = "h"  # Hourly sampling interval
MAX_LOOKBACK = 336
MAX_HORIZON = 168

DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "temperature",
    "dew_point",
    "sea_level_pressure",
    "wind_speed",
    "precipitation",
)


@dataclass
class ISDCacheConfig:
    """Configuration used when preparing the compact ISD cache."""

    window: int = MAX_LOOKBACK
    horizon: int = MAX_HORIZON
    years: Sequence[int] = field(default_factory=list)
    data_dir: PathLike = "./noaa_isd_cache"
    raw_data_dir: Optional[PathLike] = "./noaa_isd_raw"
    country: Optional[str] = "UK"
    stations: Optional[Sequence[str]] = None
    feature_columns: Optional[Sequence[str]] = None
    normalize_per_station: bool = True
    clamp_sigma: float = 5.0
    keep_time_meta: str = "end"  # "full" | "end" | "none"
    freq: str = DEFAULT_FREQ
    min_coverage: float = 0.85
    max_stations: Optional[int] = None
    overwrite: bool = False


def _require_isd() -> None:
    if isd is None:
        raise ImportError(
            "An ISD client library is required to download NOAA ISD data. "
            "Install via 'pip install isd-fetch'."
        )


def _get_station_inventory() -> pd.DataFrame:
    """Return the raw station inventory from the installed ISD client library.

    The upstream package occasionally changes its public API.  Historical
    versions exposed a ``get_stations`` helper, while newer releases provide a
    ``stations`` function.  We attempt a number of known entry points and
    normalise the returned object to a :class:`pandas.DataFrame`.
    """

    _require_isd()

    candidates = (
        "get_stations",
        "stations",
        "station_inventory",
        "list_stations",
        "inventory",
        "inventory.get_stations",
        "inventory.stations",
        "fetch_stations",
        "raw_metadata",
        "inventory.raw_metadata",
    )
    last_error: Optional[Exception] = None
    for name in candidates:
        attr: Optional[object] = isd
        for part in name.split("."):
            if attr is None:
                break
            attr = getattr(attr, part, None)
        if attr is None:
            continue

        try:
            result = attr() if callable(attr) else attr
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue

        if isinstance(result, pd.DataFrame):
            return result

        # Some variants return iterables / mappings instead of a DataFrame.
        try:
            frame = pd.DataFrame(result)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue

        if not frame.empty:
            return frame

    tried = ", ".join(candidates)
    if last_error is not None:
        raise AttributeError(
            f"Unable to locate a usable station inventory in the installed ISD client. "
            f"Tried attributes: {tried}. Last error: {last_error}"
        )
    raise AttributeError(
        "The installed ISD client does not expose a recognised station inventory accessor. "
        f"Tried attributes: {tried}."
    )


def _get_station_year_data(station_id: str, year: int) -> Tuple[pd.DataFrame, Optional[Mapping[str, object]]]:
    """Retrieve data for a single station/year combination from the ISD client.

    Similar to :func:`_get_station_inventory`, this helper tolerates API
    differences between releases by attempting several call conventions.
    """

    _require_isd()

    if _isd_backend_name == "pyisd":
        from pyisd import IsdLite
        usaf, wban = station_id[:6], station_id[6:]     # '03772099999' -> '037720','99999'
        sid = f"{usaf}-{wban}"                          # pyisd expects 'USAF-WBAN'

        isdl = IsdLite(verbose=0)
        data = isdl.get_data(
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            station_id=sid,
            organize_by="location",
        )
        if not data:
            return pd.DataFrame(), None

        # There should be exactly one entry keyed by the station id.
        df = next(iter(data.values())).reset_index()

        # Standardize time + feature names for downstream steps
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        if "index" in df.columns:
            df = df.rename(columns={"index": "datetime"})
        df = df.rename(columns={
            "temp": "temperature",
            "dewtemp": "dew_point",
            "pressure": "sea_level_pressure",
            "windspeed": "wind_speed",
            "precipitation-1h": "precipitation",
            "precipitation_1h": "precipitation",   # some versions use underscore
        })
        df["station"] = station_id                   # required later by _clean_raw_isd_frame
        return df, {"backend": "pyisd"}

    candidates = (
        "get_data",
        "data",
        "get_station_data",
        "fetch",
        "fetch_data",
        "station_data",
        "download",
        "inventory.fetch",
    )
    usaf = station_id[:6]
    wban = station_id[6:]
    call_variants = (
        ((station_id, year), {}),
        ((), {"station": station_id, "year": year}),
        ((), {"station_id": station_id, "year": year}),
        ((), {"usaf": usaf, "wban": wban, "year": year}),
        ((), {"usaf_wban": station_id, "year": year}),
    )

    last_error: Optional[Exception] = None
    for name in candidates:
        func: Optional[object] = isd
        for part in name.split("."):
            if func is None:
                break
            func = getattr(func, part, None)
        if func is None or not callable(func):
            continue

        for args, kwargs in call_variants:
            try:
                result = func(*args, **kwargs)
            except TypeError:
                # Try the next calling convention.
                continue
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                continue

            metadata: Optional[Mapping[str, object]] = None
            if isinstance(result, tuple):
                if not result:
                    continue
                data_frame = result[0]
                if len(result) > 1:
                    meta_candidate = result[1]
                    if isinstance(meta_candidate, Mapping):
                        metadata = meta_candidate
            else:
                data_frame = result

            if isinstance(data_frame, pd.DataFrame):
                return data_frame, metadata

            try:
                frame = pd.DataFrame(data_frame)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                continue

            return frame, metadata

    tried = ", ".join(candidates)
    if last_error is not None:
        raise AttributeError(
            f"Unable to retrieve ISD measurements via the installed ISD client. "
            f"Tried callables: {tried}. Last error: {last_error}"
        )
    raise AttributeError(
        "The installed ISD client does not expose a recognised data retrieval function. "
        f"Tried callables: {tried}."
    )


def list_isd_stations(
    *,
    country: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    stations: Optional[Sequence[str]] = None,
    max_stations: Optional[int] = None,
) -> pd.DataFrame:
    """Retrieve and filter the ISD station inventory.

    Parameters
    ----------
    country:
        Restrict the inventory to a specific ISO country code (e.g. ``"UK"``).
    start_year, end_year:
        Only keep stations that were active for *all* requested years.
    stations:
        Optional explicit station identifiers in ``USAFWBAN`` format.  When
        provided the inventory is filtered to this set.
    max_stations:
        Limit the number of stations returned after filtering.  Useful for quick
        experiments.
    """

    inv = _get_station_inventory()
    inv = inv.copy()
    inv.columns = [str(c).lower() for c in inv.columns]
    if "ctry" in inv.columns and "country" not in inv.columns:
        inv = inv.rename(columns={"ctry": "country"})

    if "usaf" not in inv.columns or "wban" not in inv.columns:
        raise RuntimeError("Unexpected ISD inventory format; expected 'usaf' and 'wban' columns.")

    inv["station_id"] = inv["usaf"].astype(str).str.zfill(6) + inv["wban"].astype(str).str.zfill(5)

    if country is not None and "country" in inv.columns:
        country_mask = inv["country"].str.upper() == country.upper()
        country_mask = country_mask.fillna(False)
        inv = inv.loc[country_mask]

    if stations is not None:
        wanted = {str(s) for s in stations}
        station_mask = inv["station_id"].isin(wanted)
        inv = inv.loc[station_mask]

    if start_year is not None or end_year is not None:
        begin = pd.to_datetime(inv.get("begin")) if "begin" in inv.columns else None
        end = pd.to_datetime(inv.get("end")) if "end" in inv.columns else None
        if begin is not None:
            begin_year = begin.dt.year
            valid_begin = begin_year.dropna()
            if not valid_begin.empty:
                cutoff = start_year if start_year is not None else int(valid_begin.min())
                begin_mask = begin_year <= cutoff
                inv = inv.loc[begin_mask.fillna(False)]
        if end is not None:
            end_years = end.dt.year
            valid_end = end_years.dropna()
            if not valid_end.empty:
                cutoff = end_year if end_year is not None else int(valid_end.max())
                end_mask = end_years >= cutoff
                inv = inv.loc[end_mask.fillna(False)]

    inv = inv.sort_values("station_id")

    if max_stations is not None:
        inv = inv.head(int(max_stations))

    return inv.reset_index(drop=True)


def _load_or_download_station_year(
    station_id: str,
    year: int,
    raw_data_dir: Optional[Path],
) -> pd.DataFrame:
    """Fetch a single station-year dataframe, caching to disk when requested."""

    if raw_data_dir is not None:
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = raw_data_dir / f"{station_id}_{year}.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)

    df_year, _meta = _get_station_year_data(station_id, year)
    if not isinstance(df_year, pd.DataFrame) or df_year.empty:
        return pd.DataFrame()

    df_year = df_year.copy()
    df_year.columns = [str(c).lower() for c in df_year.columns]

    if raw_data_dir is not None:
        parquet_path = raw_data_dir / f"{station_id}_{year}.parquet"
        df_year.to_parquet(parquet_path)

    return df_year


def download_isd_dataset(
    station_ids: Sequence[str],
    years: Sequence[int],
    raw_data_dir: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Download the NOAA ISD subset for the requested stations and years."""

    if not station_ids:
        raise ValueError("At least one station identifier must be provided.")

    years_sorted = sorted(set(int(y) for y in years))
    if not years_sorted:
        raise ValueError("No years specified for download.")

    raw_dir = Path(raw_data_dir).expanduser() if raw_data_dir is not None else None

    frames: List[pd.DataFrame] = []
    for station_id in station_ids:
        for year in years_sorted:
            frame = _load_or_download_station_year(str(station_id), int(year), raw_dir)
            if not frame.empty:
                frames.append(frame)

    if not frames:
        raise RuntimeError("No ISD data could be retrieved for the requested configuration.")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def _clean_raw_isd_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and enforce datetime ordering."""

    cleaned = df.copy()
    cleaned.columns = [str(c).lower() for c in cleaned.columns]

    if "date" in cleaned.columns:
        cleaned["datetime"] = pd.to_datetime(cleaned.pop("date"), errors="coerce")
    elif "datetime" in cleaned.columns:
        cleaned["datetime"] = pd.to_datetime(cleaned["datetime"], errors="coerce")
    else:
        raise ValueError("Raw ISD dataframe must contain a 'date' or 'datetime' column.")

    cleaned = cleaned.dropna(subset=["datetime"])

    if "station" not in cleaned.columns:
        if "station_id" in cleaned.columns:
            station_values = cleaned.pop("station_id")
            valid_mask = station_values.notna()
            cleaned = cleaned.loc[valid_mask].copy()
            station_values = station_values.loc[valid_mask]
        elif {"usaf", "wban"}.issubset(cleaned.columns):
            usaf_values = cleaned.pop("usaf")
            wban_values = cleaned.pop("wban")
            valid_mask = usaf_values.notna() & wban_values.notna()
            cleaned = cleaned.loc[valid_mask].copy()
            station_values = (
                usaf_values.loc[valid_mask].astype(str).str.zfill(6)
                + wban_values.loc[valid_mask].astype(str).str.zfill(5)
            )
        else:
            raise ValueError(
                "Raw ISD dataframe must contain a 'station' column or enough metadata to construct one."
            )
        cleaned["station"] = station_values

    station_values = cleaned["station"]
    if station_values.isna().any():
        cleaned = cleaned.loc[station_values.notna()].copy()
        station_values = cleaned["station"]

    cleaned["station"] = station_values.astype(str)
    cleaned = cleaned.sort_values(["station", "datetime"]).reset_index(drop=True)

    return cleaned


def _build_station_panels(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    freq: str,
    min_coverage: float,
) -> Dict[str, pd.DataFrame]:
    """Create a dictionary mapping station identifiers to aligned feature panels."""

    df = _clean_raw_isd_frame(df)

    feature_cols = [str(c).lower() for c in feature_columns]
    if TARGET_COLUMN not in feature_cols:
        raise ValueError(f"Target column '{TARGET_COLUMN}' must be included in feature_columns.")

    available_features = [c for c in feature_cols if c in df.columns]
    if len(available_features) != len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        raise ValueError(f"Missing expected ISD feature columns: {sorted(missing)}")

    panels: Dict[str, pd.DataFrame] = {}
    for station_id, group in df.groupby("station", sort=False):
        station_df = group.set_index("datetime")[available_features]
        station_df = station_df.apply(pd.to_numeric, errors="coerce")
        station_df = station_df.resample(freq).mean()
        observed_mask = station_df.notna()
        station_df = station_df.ffill()
        if station_df.empty:
            continue
        row_coverage = 1.0 - station_df.isna().mean(axis=1)
        coverage = float(row_coverage.mean()) if len(row_coverage) else 0.0
        if coverage < min_coverage:
            continue
        required = max(1, int(np.ceil(min_coverage * len(feature_cols))))
        keep_rows = station_df.notna().sum(axis=1) >= required
        station_df = station_df.loc[keep_rows]
        observed_mask = observed_mask.reindex(station_df.index).fillna(False)
        if station_df.empty:
            continue
        if TARGET_COLUMN not in observed_mask.columns or not bool(observed_mask[TARGET_COLUMN].any()):
            continue
        out = station_df.astype(np.float32)
        out.attrs["observed_mask"] = observed_mask[available_features].astype(bool)
        panels[str(station_id)] = out

    if not panels:
        raise RuntimeError("All stations were filtered out during panel construction.")

    return panels




def prepare_isd_cache(cfg: ISDCacheConfig) -> Mapping[str, List[str]]:
    """Prepare the compact cache for the NOAA ISD dataset."""

    if cfg.window <= 0:
        raise ValueError("window must be positive")
    if cfg.horizon < 0:
        raise ValueError("horizon must be non-negative")
    if cfg.window > MAX_LOOKBACK:
        raise ValueError(
            f"window ({cfg.window}) exceeds supported maximum ({MAX_LOOKBACK})."
        )
    if cfg.horizon > MAX_HORIZON:
        raise ValueError(
            f"horizon ({cfg.horizon}) exceeds supported maximum ({MAX_HORIZON})."
        )
    if not cfg.years:
        raise ValueError("At least one year must be specified in cfg.years")

    keep_time_meta = cfg.keep_time_meta.lower()
    if keep_time_meta not in {"full", "end", "none"}:
        raise ValueError("keep_time_meta must be one of {'full', 'end', 'none'}")

    feature_columns = list(dict.fromkeys((cfg.feature_columns or DEFAULT_FEATURE_COLUMNS)))
    feature_columns = [c.lower() for c in feature_columns]

    station_inventory = list_isd_stations(
        country=cfg.country,
        start_year=min(cfg.years),
        end_year=max(cfg.years),
        stations=cfg.stations,
        max_stations=cfg.max_stations,
    )

    if station_inventory.empty:
        raise RuntimeError("No ISD stations matched the requested filters.")

    station_ids = station_inventory["station_id"].astype(str).tolist()

    raw_dir = Path(cfg.raw_data_dir).expanduser().resolve() if cfg.raw_data_dir else None
    raw_df = download_isd_dataset(station_ids=station_ids, years=cfg.years, raw_data_dir=raw_dir)

    panels = _build_station_panels(
        raw_df,
        feature_columns=feature_columns,
        freq=cfg.freq,
        min_coverage=cfg.min_coverage,
    )

    assets = sorted(panels.keys())
    asset_to_id = {asset: idx for idx, asset in enumerate(assets)}

    feature_cols = list(feature_columns)
    target_col = TARGET_COLUMN
    if target_col in feature_cols:
        feature_cols = [target_col] + [c for c in feature_cols if c != target_col]

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    paths = CachePaths.from_dir(data_dir)
    if cfg.overwrite and paths.cache_root.exists():
        import shutil

        shutil.rmtree(paths.cache_root)
    paths.ensure()

    pairs: List[np.ndarray] = []
    end_times: List[np.ndarray] = []
    start_times: List[np.datetime64] = []
    stop_times: List[np.datetime64] = []

    feature_dim = len(feature_cols)
    norm_acc = NormalizationStatsAccumulator(
        num_assets=len(assets),
        feature_dim=feature_dim,
        per_asset=cfg.normalize_per_station,
    )

    for asset in assets:
        aid = asset_to_id[asset]
        panel = panels[asset][feature_cols]
        total_rows = panel.shape[0]
        min_required = cfg.window + cfg.horizon
        if total_rows < min_required:
            raise ValueError(
                f"Station '{asset}' has insufficient history for the requested window/horizon."
            )

        features = panel.to_numpy(dtype=np.float32, copy=True)
        targets = panel[target_col].to_numpy(dtype=np.float32, copy=True)
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

        norm_acc.update(aid, features, targets)

        max_start = total_rows - min_required + 1
        if max_start <= 0:
            continue
        starts = np.arange(0, max_start, dtype=np.int32)
        if cfg.horizon > 0:
            obs = obs_mask[:, feature_cols.index(target_col)]
            future_obs = sliding_window_view(obs, window_shape=cfg.horizon)
            valid = future_obs[cfg.window : cfg.window + max_start].any(axis=1)
            starts = starts[valid]
        if starts.size == 0:
            continue
        window_end_times = times[starts + cfg.window - 1].astype("datetime64[ns]")
        pairs.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        end_times.append(window_end_times)
        start_times.append(times[0])
        stop_times.append(times[-1])

    if not pairs:
        raise RuntimeError("No valid context windows available across stations.")

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)
    global_end_times = np.concatenate(end_times, axis=0).astype("datetime64[ns]")
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", global_end_times)

    norm_stats = norm_acc.finalize(assets)

    dataset_start = min(start_times).astype("datetime64[s]") if start_times else None
    dataset_end = max(stop_times).astype("datetime64[s]") if stop_times else None

    meta = {
        "dataset": "noaa_isd",
        "format": "indexcache_v1",
        "assets": assets,
        "asset2id": asset_to_id,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "max_window": MAX_LOOKBACK,
        "max_horizon": MAX_HORIZON,
        "keep_time_meta": keep_time_meta,
        "normalize_per_ticker": cfg.normalize_per_station,
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


def load_isd_dataloaders_with_ratio_split(
    data_dir: PathLike,
    *,
    split_policy: str = "global_purged_horizon",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
    **loader_kwargs,
):
    """Wrapper around the financial ratio-split loader for ISD datasets."""

    return _load_fin_ratio_split(
        data_dir=data_dir,
        split_policy=split_policy,
        exact_timestamp_batches=exact_timestamp_batches,
        target_col=target_col,
        target_cols=target_cols,
        **loader_kwargs,
    )


def _validate_isd_cache(paths: CachePaths) -> Dict[str, object]:
    """Read and validate the cache metadata, returning the parsed JSON."""

    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache meta file not found at '{paths.meta}'. Did you run prepare_isd_cache()?"
        )

    with paths.meta.open("r") as f:
        meta: Dict[str, object] = json.load(f)

    dataset = meta.get("dataset")
    if dataset != "noaa_isd":
        raise ValueError(
            f"The cache located at '{paths.cache_root}' does not correspond to the NOAA ISD dataset."
        )

    # Clamp the recorded maxima to the supported limits so callers cannot request
    # windows that exceed the hard-coded capabilities of this dataset module.
    try:
        meta["max_window"] = min(int(meta.get("max_window", MAX_LOOKBACK)), MAX_LOOKBACK)
    except (TypeError, ValueError):
        meta["max_window"] = MAX_LOOKBACK
    try:
        meta["max_horizon"] = min(int(meta.get("max_horizon", MAX_HORIZON)), MAX_HORIZON)
    except (TypeError, ValueError):
        meta["max_horizon"] = MAX_HORIZON

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
    """Build train/val/test loaders for a prepared ISD cache.

    Mirrors :func:`llapdiffusion.datasets.fin_dataset.run_experiment` so downstream training
    code can switch datasets with minimal friction.
    """

    coverage = _validate_context_missingness_rate(coverage, name="coverage")
    paths = CachePaths.from_dir(data_dir)
    meta = _validate_isd_cache(paths)
    max_window = int(meta.get("max_window", MAX_LOOKBACK))
    max_horizon = int(meta.get("max_horizon", MAX_HORIZON))

    def _coerce_within(value: object, default: int, upper: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return min(parsed, upper)

    base_window = _coerce_within(meta.get("window"), max_window, max_window)
    base_horizon = _coerce_within(meta.get("horizon"), max_horizon, max_horizon)

    if K <= 0:
        raise ValueError("K (window) must be positive")
    if H < 0:
        raise ValueError("H (horizon) must be non-negative")

    if K > max_window or H > max_horizon:
        raise ValueError(
            f"Requested (window={K}, horizon={H}) exceed the supported maximums "
            f"({max_window}, {max_horizon})."
        )

    needs_reindex = (int(K), int(H)) != (base_window, base_horizon)
    needs_target_reindex = target_col is not None or target_cols is not None

    if needs_reindex and not reindex:
        raise ValueError(
            "reindex=False but (K, H) differ from the cached configuration. "
            "Enable reindex to rebuild the window index."
        )

    if reindex and (needs_reindex or needs_target_reindex):
        _rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=needs_reindex,
            backup_old=False,
            target_col=target_col,
            target_cols=target_cols,
        )

    train_dl, val_dl, test_dl, lengths = load_isd_dataloaders_with_ratio_split(
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
    "ISDCacheConfig",
    "list_isd_stations",
    "download_isd_dataset",
    "prepare_isd_cache",
    "load_isd_dataloaders_with_ratio_split",
    "run_experiment",
]
