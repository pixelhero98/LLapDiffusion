"""Utilities for preparing and loading financial time-series datasets.

The compact cache produced by :func:`prepare_features_and_index_cache` stores data
under ``<data_dir>/cache_ratio_index`` using the following layout::

    features_fp16/<asset_id>.npy   # [T, F] float16 feature matrix
    targets_fp16/<asset_id>.npy    # [T] float16 target series
    times/<asset_id>.npy           # [T] datetime64[ns] index
    windows/global_pairs.npy       # [M, 2] int32 (asset_id, start_idx)
    windows/end_times.npy          # [M] datetime64[ns] context end timestamps
    meta.json                      # schema + feature configuration
    norm_stats.json                # per-ticker or global normalization stats
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass, field
from math import ceil as _ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler as _Sampler

from llapdiffusion.datasets._types import PathLike
from llapdiffusion.datasets._normalization import NormalizationStatsAccumulator
from llapdiffusion.datasets.target_selection import (
    finance_calendar_feature_cols_from_names,
    resolve_target_selection,
)

# --------------------- Public cache configs ---------------------

@dataclass
class CalendarConfig:
    include_dow: bool = True
    include_dom: bool = True
    include_moy: bool = True
    dow_period: int = 7
    moy_period: int = 12
    dow_sin_name: str = "DOW_SIN"
    dow_cos_name: str = "DOW_COS"
    dom_sin_name: str = "DOM_SIN"
    dom_cos_name: str = "DOM_COS"
    moy_sin_name: str = "MOY_SIN"
    moy_cos_name: str = "MOY_COS"

@dataclass
class FeatureConfig:
    price_fields: List[str] = field(default_factory=lambda: ["Close"])  # which to convert to returns
    returns_mode: str = "log"       # 'log' or 'pct'
    include_rvol: bool = True
    rvol_span: int = 20
    rvol_on: str = "Close"
    include_dlv: bool = True
    market_proxy: Optional[str] = "SPY"
    include_oc: bool = False
    include_gap: bool = False
    include_hl_range: bool = False
    target_field: str = "Close"     # used for Y
    target_col: Optional[str] = None
    if_calendar: bool = True
    calendar: Optional[CalendarConfig] = None
    include_entity_id_feature: bool = False

    def __post_init__(self) -> None:
        """Ensure calendar configuration respects the calendar toggle."""
        if self.if_calendar:
            if self.calendar is None:
                self.calendar = CalendarConfig()
        else:
            self.calendar = None


@dataclass(frozen=True)
class CachePaths:
    """Helper for managing paths in the compact on-disk cache."""

    data_dir: Path

    @classmethod
    def from_dir(cls, data_dir: PathLike) -> "CachePaths":
        return cls(Path(data_dir).expanduser())

    @property
    def cache_root(self) -> Path:
        return self.data_dir / "cache_ratio_index"

    @property
    def features(self) -> Path:
        return self.cache_root / "features_fp16"

    @property
    def targets(self) -> Path:
        return self.cache_root / "targets_fp16"

    @property
    def times(self) -> Path:
        return self.cache_root / "times"

    @property
    def windows(self) -> Path:
        return self.cache_root / "windows"

    @property
    def obs_masks(self) -> Path:
        return self.cache_root / "obs_masks_bool"

    @property
    def fill_masks(self) -> Path:
        return self.cache_root / "fill_masks_bool"

    @property
    def meta(self) -> Path:
        return self.cache_root / "meta.json"

    @property
    def norm_stats(self) -> Path:
        return self.cache_root / "norm_stats.json"

    def ensure(self) -> None:
        """Create all cache directories if they do not already exist."""
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.features.mkdir(parents=True, exist_ok=True)
        self.targets.mkdir(parents=True, exist_ok=True)
        self.times.mkdir(parents=True, exist_ok=True)
        self.windows.mkdir(parents=True, exist_ok=True)
        self.obs_masks.mkdir(parents=True, exist_ok=True)
        self.fill_masks.mkdir(parents=True, exist_ok=True)
            
# --------------------- Lightweight stores ---------------------

def _indexcache_dir(data_dir: PathLike) -> Path:
    return CachePaths.from_dir(data_dir).cache_root


def _features_dir(data_dir: PathLike) -> Path:
    return CachePaths.from_dir(data_dir).features


def _targets_dir(data_dir: PathLike) -> Path:
    return CachePaths.from_dir(data_dir).targets


def _times_dir(data_dir: PathLike) -> Path:
    return CachePaths.from_dir(data_dir).times


def _windows_dir(data_dir: PathLike) -> Path:
    return CachePaths.from_dir(data_dir).windows


def _meta_path(data_dir: PathLike) -> Path:
    return CachePaths.from_dir(data_dir).meta


def _norm_path(data_dir: PathLike) -> Path:
    return CachePaths.from_dir(data_dir).norm_stats

# --------------------- Feature engineering (same semantics) ---------------------

_EPS = 1e-6

def _mask_nonpos(s: pd.Series) -> pd.Series:
    return s.where((s > 0) & np.isfinite(s))

def _safe_log_series(s: pd.Series) -> pd.Series:
    return np.log(_mask_nonpos(s))

def _safe_log1p_series(s: pd.Series) -> pd.Series:
    return np.log1p(s.clip(lower=-1 + _EPS))

def _safe_pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change().replace([np.inf, -np.inf], np.nan)

def _log_return(s: pd.Series) -> pd.Series:
    return _safe_log_series(s).diff()

def _ewma_vol(ret: pd.Series, span: int = 20) -> pd.Series:
    return ret.pow(2).ewm(span=span, adjust=False).mean().pow(0.5)

def _delta_log_volume(vol: pd.Series) -> pd.Series:
    v = vol.replace([0, np.inf, -np.inf], np.nan)
    return _safe_log_series(v).diff()


def _cyclical_from_int(values: np.ndarray, period: int):
    ang = 2.0 * np.pi * (values.astype(np.float32) / float(period))
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)


def _finite_mean_std(
    arr: np.ndarray, axis=None, keepdims: bool = True, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute finite-only mean/std with stability safeguards."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    count = finite.sum(axis=axis, keepdims=keepdims)
    safe = np.where(finite, arr, 0.0)

    denom = np.maximum(count, 1)
    mean = safe.sum(axis=axis, dtype=np.float64, keepdims=keepdims) / denom
    sq_sum = np.square(safe, dtype=np.float64).sum(axis=axis, keepdims=keepdims)
    var = sq_sum / denom - np.square(mean, dtype=np.float64)
    var = np.maximum(var, eps)
    std = np.sqrt(var, dtype=np.float64)
    std = np.where((std == 0.0) | ~np.isfinite(std), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), count.astype(np.int64)


def _compute_time_deltas(times: np.ndarray) -> np.ndarray:
    """Return first differences between consecutive timestamps in native units."""
    if times.size == 0:
        return np.array([], dtype=np.float32)
    deltas = np.diff(times.astype("datetime64[s]").astype(np.int64)).astype(np.float32)
    deltas = np.concatenate([np.zeros((1,), dtype=np.float32), deltas])
    return deltas


def _frequency_to_seconds(freq: Optional[str]) -> Optional[float]:
    """Convert a pandas-compatible frequency string into seconds."""
    if not freq:
        return None
    try:
        offset = pd.tseries.frequencies.to_offset(str(freq))
    except (TypeError, ValueError):
        return None

    # Tick-like offsets expose a deterministic nanosecond length.
    if hasattr(offset, "nanos"):
        nanos = int(offset.nanos)
        if nanos > 0:
            return float(nanos) / 1e9

    return None


def _infer_native_time_scale(paths: CachePaths, assets: Sequence[str], meta: dict) -> Tuple[str, float]:
    """Infer the natural delta_t unit for the cached dataset.

    Returns
    -------
    Tuple[str, float]
        A human-readable unit name and the corresponding number of seconds.
    """
    freq_seconds = _frequency_to_seconds(meta.get("freq"))
    if freq_seconds is not None and np.isfinite(freq_seconds) and freq_seconds > 0:
        return str(meta.get("freq")), float(freq_seconds)

    if meta.get("dataset") == "fin_dataset":
        # Financial bars are effectively daily in this pipeline.
        return "1D", 86400.0

    for aid in range(len(assets)):
        t_path = paths.times / f"{aid}.npy"
        if not t_path.exists():
            continue
        times = np.load(t_path, mmap_mode="r")
        if times.size < 2:
            continue
        deltas = np.diff(times.astype("datetime64[s]").astype(np.int64))
        deltas = deltas[deltas > 0]
        if deltas.size == 0:
            continue
        native_seconds = float(np.median(deltas))
        if np.isfinite(native_seconds) and native_seconds > 0:
            return "median_step", native_seconds

    # Safe fallback: leave values in seconds.
    return "1s", 1.0


def _compute_time_offsets_from_anchor(
    times: np.ndarray,
    anchor_time,
    native_scale_seconds: float,
) -> np.ndarray:
    """
    Return offsets from ``anchor_time`` measured in native time units.

    For daily data, ``times=[t1, t4]`` and ``anchor_time=t0`` returns ``[1, 4]``.
    """
    if times is None:
        return np.array([], dtype=np.float32)

    arr = np.asarray(times)
    if arr.size == 0:
        return np.array([], dtype=np.float32)

    if np.issubdtype(arr.dtype, np.number):
        values = arr.astype(np.float32, copy=False)
        anchor = float(np.asarray(anchor_time, dtype=np.float32).reshape(-1)[0])
        return (values - anchor).astype(np.float32, copy=False)

    if np.issubdtype(arr.dtype, np.datetime64):
        t_sec = arr.astype("datetime64[s]").astype(np.int64)
        anchor_sec = np.asarray(anchor_time).astype("datetime64[s]").astype(np.int64)
        rel_sec = (t_sec - anchor_sec).astype(np.float32)

        scale = float(native_scale_seconds)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        return (rel_sec / scale).astype(np.float32, copy=False)

    try:
        values = arr.astype(np.float32, copy=False)
        anchor = float(np.asarray(anchor_time, dtype=np.float32).reshape(-1)[0])
        return (values - anchor).astype(np.float32, copy=False)
    except (TypeError, ValueError):
        t_sec = arr.astype("datetime64[s]").astype(np.int64)
        anchor_sec = np.asarray(anchor_time).astype("datetime64[s]").astype(np.int64)
        rel_sec = (t_sec - anchor_sec).astype(np.float32)
        scale = float(native_scale_seconds)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return (rel_sec / scale).astype(np.float32, copy=False)


def _compute_relative_time_deltas(times: np.ndarray, native_scale_seconds: float) -> np.ndarray:
    """
    Return context-window-relative time offsets, measured in native time units.

    Example (daily data):
      times = [t0, t1, t2] -> [0, 1, 2]
    """
    if times is None:
        return np.array([], dtype=np.float32)

    arr = np.asarray(times)
    if arr.size == 0:
        return np.array([], dtype=np.float32)

    return _compute_time_offsets_from_anchor(arr, arr.reshape(-1)[0], native_scale_seconds)


def _validate_context_missingness_rate(rate: float, *, name: str = "coverage") -> float:
    """Validate the induced context-missingness rate."""
    value = float(rate)
    if not np.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1.")
    return value


def _sample_identity_seed(seed: int, aid: int, start: int) -> int:
    key = f"{int(seed)}:{int(aid)}:{int(start)}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def _apply_context_missingness(
    obs_mask: np.ndarray,
    rate: float,
    seed: int,
    aid: int,
    start: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Hide floor(rate * observed_entries) observed context entries deterministically."""
    obs = np.asarray(obs_mask, dtype=bool).copy()
    observed = np.flatnonzero(obs.reshape(-1))
    n_hide = int(np.floor(float(rate) * observed.size))
    if n_hide <= 0:
        return obs, None

    rng = np.random.default_rng(_sample_identity_seed(seed, aid, start))
    hidden_idx = rng.choice(observed, size=n_hide, replace=False)
    hidden = np.zeros(obs.size, dtype=bool)
    hidden[hidden_idx] = True
    hidden = hidden.reshape(obs.shape)
    obs[hidden] = False
    return obs, hidden


def _zero_hidden_context_values(x: np.ndarray, hidden_mask: Optional[np.ndarray]) -> None:
    if hidden_mask is None:
        return
    hidden = np.asarray(hidden_mask, dtype=bool)
    if hidden.shape == x.shape:
        x[hidden] = 0.0
        return
    if hidden.ndim == 1 and x.ndim == 2 and hidden.shape[0] == x.shape[0]:
        x[hidden, :] = 0.0
        return
    x[np.broadcast_to(hidden, x.shape)] = 0.0


def build_calendar_frame(idx: pd.DatetimeIndex, cfg: CalendarConfig) -> pd.DataFrame:
    import pandas as pd
    cols = {}
    if cfg.include_dow:
        dow = idx.dayofweek.values
        s, c = _cyclical_from_int(dow, cfg.dow_period)
        cols[cfg.dow_sin_name] = s
        cols[cfg.dow_cos_name] = c
    if cfg.include_dom:
        dom = idx.day.values
        dim = idx.days_in_month.values
        ang = 2.0 * np.pi * ((dom - 1).astype(np.float32) / dim.astype(np.float32).clip(min=1))
        cols[cfg.dom_sin_name] = np.sin(ang).astype(np.float32)
        cols[cfg.dom_cos_name] = np.cos(ang).astype(np.float32)
    if cfg.include_moy:
        moy = (idx.month.values - 1)
        s, c = _cyclical_from_int(moy, cfg.moy_period)
        cols[cfg.moy_sin_name] = s
        cols[cfg.moy_cos_name] = c
    out = pd.DataFrame(cols, index=idx)
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    return out

# --------------------- Compact cache builder ---------------------

def prepare_features_and_index_cache(
    tickers: List[str],
    start: str,
    end: str,
    window: int,
    horizon: int,
    data_dir: str = "./data",
    feature_cfg: Optional[FeatureConfig] = None,
    normalize_per_ticker: bool = True,
    clamp_sigma: float = 5.0,
    min_obs_buffer: int = 50,
    min_train_coverage: Optional[float] = None,
    min_row_coverage: Optional[float] = 0.0,
    liquidity_rank_window: Optional[Tuple[str,str]] = None,
    top_n_by_dollar_vol: Optional[int] = None,
    max_windows_per_ticker: Optional[int] = None,  # applies at *index* build time
    regression: bool = True,
    seed: int = 1337,
    keep_time_meta: str = "end",  # "full" | "end" | "none"
):
    """Builds a *compact* cache: per-ticker feature matrices + global window index.
    - No split-by-date; splitting is performed later (ratio-based) in the loader.
    - Stores float16 features/targets to halve disk usage again.
    """
    try:
        import pandas as pd
        import yfinance as yf
    except Exception as e:
        raise ImportError("pandas + yfinance required to prepare cache. pip install pandas yfinance") from e

    feature_cfg = feature_cfg or FeatureConfig()

    paths = CachePaths.from_dir(data_dir)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.ensure()
    target_col = feature_cfg.target_col or f"RET_{feature_cfg.target_field.upper()}"

    # ---- download / load features per ticker (reuses your feature logic) ----
    need_price_cols = set(feature_cfg.price_fields) | {feature_cfg.target_field}
    if feature_cfg.include_oc or feature_cfg.include_gap:
        need_price_cols |= {"Open", "Close"}
    if feature_cfg.include_hl_range:
        need_price_cols |= {"High", "Low"}
    wanted_cols = sorted(list(need_price_cols | ({'Volume'} if feature_cfg.include_dlv else set())))

    tickers_dl = tickers[:]
    if feature_cfg.market_proxy and feature_cfg.market_proxy not in tickers_dl:
        tickers_dl.append(feature_cfg.market_proxy)

    raw = yf.download(tickers_dl, start=start, end=end, auto_adjust=True, group_by="column", progress=False)
    if hasattr(raw, "columns") and getattr(raw.columns, "nlevels", 1) > 1:
        # normalize to (field, ticker)
        raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

    def get_ticker_df(t: str, raw_df: pd.DataFrame = raw) -> pd.DataFrame:
        if getattr(raw_df, "columns", None) is not None and getattr(raw_df.columns, "nlevels", 1) > 1:
            try:
                df = raw_df.xs(t, axis=1, level=1)
            except Exception:
                df = raw_df.xs(t, axis=1, level=0)
        else:
            df = raw_df
        keep = [c for c in wanted_cols if c in df.columns]
        return df[keep].sort_index()

    proxy_ret = None
    if feature_cfg.market_proxy:
        try:
            proxy_df = get_ticker_df(feature_cfg.market_proxy)
            if 'Close' in proxy_df:
                proxy_ret = (_log_return(proxy_df['Close']) if feature_cfg.returns_mode == 'log' else _safe_pct_change(proxy_df['Close'])).rename('MKT')
        except Exception:
            proxy_ret = None

    def median_dollar_volume(df: pd.DataFrame, a: str, b: str) -> float:
        sub = df.loc[a:b]
        if 'Close' not in sub or 'Volume' not in sub:
            return 0.0
        dv = (sub['Close'] * sub['Volume']).replace([np.inf, -np.inf], np.nan).dropna()
        return float(dv.median()) if len(dv) else 0.0

    if liquidity_rank_window and top_n_by_dollar_vol:
        a, b = liquidity_rank_window
        ranks = []
        for t in tickers:
            df_t = get_ticker_df(t)
            ranks.append((t, median_dollar_volume(df_t, a, b)))
        ranks.sort(key=lambda x: x[1], reverse=True)
        tickers = [t for t, _ in ranks[:top_n_by_dollar_vol]]

    def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
        feat = {}
        for price_field in feature_cfg.price_fields:
            if price_field in df:
                if feature_cfg.returns_mode == 'log':
                    feat[f'RET_{price_field.upper()}'] = _log_return(df[price_field])
                else:
                    feat[f'RET_{price_field.upper()}'] = _safe_pct_change(df[price_field])
        if feature_cfg.include_oc and 'Open' in df and 'Close' in df:
            oc = (_safe_log_series(df['Close']) - _safe_log_series(df['Open'])
                  if feature_cfg.returns_mode == 'log' else (df['Close'] / df['Open'] - 1.0))
            feat['OC_RET'] = oc
        if feature_cfg.include_gap and 'Open' in df and 'Close' in df:
            gap = (_safe_log_series(df['Open']) - _safe_log_series(df['Close'].shift(1))
                   if feature_cfg.returns_mode == 'log' else (df['Open'] / df['Close'].shift(1) - 1.0))
            feat['GAP_RET'] = gap
        if feature_cfg.include_hl_range and 'High' in df and 'Low' in df:
            hlr = (_safe_log_series(df['High']) - _safe_log_series(df['Low'])
                   if feature_cfg.returns_mode == 'log' else (df['High'] / df['Low'] - 1.0))
            feat['HL_RANGE'] = hlr
        if feature_cfg.include_dlv and 'Volume' in df:
            feat['DLV'] = _delta_log_volume(df['Volume'])
        if feature_cfg.include_rvol:
            base_col = f"RET_{feature_cfg.rvol_on.upper()}"
            if base_col in feat:
                feat[f'RVOL{feature_cfg.rvol_span}_{feature_cfg.rvol_on.upper()}'] = _ewma_vol(feat[base_col], span=feature_cfg.rvol_span)
        if proxy_ret is not None:
            feat['MKT'] = proxy_ret.reindex(df.index)
        out = pd.DataFrame(feat)
        if feature_cfg.if_calendar and feature_cfg.calendar is not None:
            cal = build_calendar_frame(out.index, feature_cfg.calendar)
            out = pd.concat([out, cal], axis=1)
        out = out.dropna(how="all")
        # enforce float32 now, will save to float16 later
        for c in out.columns:
            if out[c].dtype == np.float64:
                out[c] = out[c].astype(np.float32)
        return out

    # ---- Build per-ticker features ----
    per_ticker: Dict[str, pd.DataFrame] = {}
    min_obs = window + horizon + min_obs_buffer
    for t in tickers:
        if t == feature_cfg.market_proxy:
            continue
        df_raw = get_ticker_df(t)
        if df_raw.shape[0] < min_obs:
            continue
        feat_df = build_feature_frame(df_raw)
        if feat_df.shape[0] < min_obs:
            continue
        if target_col not in feat_df.columns:
            continue
        obs_mask_full = feat_df.notna()
        # coverage check using first 80% of sample as proxy for train adequacy
        if min_train_coverage is not None:
            train_like = obs_mask_full.iloc[: max(1, int(0.8 * len(feat_df)))]
            row_coverage = train_like.mean(axis=1).to_numpy(dtype=np.float32) if len(train_like) else []
            coverage = float(np.mean(row_coverage)) if len(row_coverage) else 0.0
            if coverage < min_train_coverage:
                continue
        if min_row_coverage is not None:
            min_non_missing = max(1, int(np.ceil(min_row_coverage * feat_df.shape[1])))
        else:
            min_non_missing = 1
        keep_rows = obs_mask_full.sum(axis=1) >= min_non_missing
        feat_df = feat_df.loc[keep_rows]
        if not np.isfinite(feat_df[target_col].to_numpy(dtype=np.float32, copy=False)).any():
            continue
        per_ticker[t] = feat_df

    if not per_ticker:
        raise RuntimeError("No tickers passed the cleaning criteria. Relax thresholds or check dates.")

    # ---- Align feature columns across tickers ----
    assets = sorted(per_ticker.keys())
    asset2id = {a: i for i, a in enumerate(assets)}

    if getattr(feature_cfg, 'include_entity_id_feature', False):
        denom = max(1, len(assets) - 1)
        for a in assets:
            aid = asset2id[a]
            val = np.float32(aid / denom) if denom > 0 else np.float32(0.0)
            per_ticker[a]['ENTITY_ID'] = np.full((len(per_ticker[a])), val, dtype=np.float32)

    col_sets = [set(df.columns) for df in per_ticker.values()]
    feature_cols = sorted(list(set.intersection(*col_sets)))
    target_col = feature_cfg.target_col or f"RET_{feature_cfg.target_field.upper()}"
    if target_col not in feature_cols:
        raise ValueError(f"Target column '{target_col}' not in common features.")
    calendar_feature_cols = finance_calendar_feature_cols_from_names(feature_cols)
    if target_col in set(calendar_feature_cols):
        raise ValueError(f"Target column '{target_col}' is a calendar feature and cannot be used as a financial target.")

    # ---- Save compact arrays ----
    # Per-ticker matrices: X[t] shape [T,F] float16, Y[t] shape [T] float16, times[t] datetime64
    norm_acc = NormalizationStatsAccumulator(
        num_assets=len(assets),
        feature_dim=len(feature_cols),
        per_asset=normalize_per_ticker,
    )
    for a in assets:
        df = per_ticker[a]
        feature_view = df[feature_cols]
        X = feature_view.to_numpy(dtype=np.float32, copy=False)
        Y = df[target_col].to_numpy(dtype=np.float32, copy=False)
        times = df.index.to_numpy()
        obs_mask = feature_view.notna().to_numpy(dtype=bool, copy=False)
        fill_mask = None  # no in-place filling performed here
        # Write as float16 (2× smaller) – okay post-normalization/standardization
        aid = asset2id[a]
        np.save(paths.features / f"{aid}.npy", X.astype(np.float16))
        np.save(paths.targets / f"{aid}.npy", Y.astype(np.float16))
        np.save(paths.times / f"{aid}.npy", times.astype("datetime64[ns]"))
        np.save(paths.obs_masks / f"{aid}.npy", obs_mask)
        if fill_mask is not None:
            np.save(paths.fill_masks / f"{aid}.npy", fill_mask)

        norm_acc.update(aid, X, Y)

    # ---- Precompute a *global* window index (small) ----
    # This is optional but speeds up loader start; also lets us cap max_windows_per_ticker deterministically
    pairs: List[np.ndarray] = []   # (aid, start_idx)
    ends:  List[np.ndarray] = []   # end-of-context times
    for a in assets:
        aid = asset2id[a]
        times = np.load(paths.times / f"{aid}.npy")
        Y = np.load(paths.targets / f"{aid}.npy")
        times_arr = times.astype("datetime64[ns]")
        T = times_arr.shape[0]
        min_required = window + horizon
        if T < min_required:
            continue
        start_idxs = np.arange(0, T - min_required + 1, dtype=np.int32)
        if max_windows_per_ticker is not None and start_idxs.size > max_windows_per_ticker:
            start_idxs = start_idxs[:max_windows_per_ticker]
        if horizon > 0:
            obs = np.isfinite(Y)
            future_obs = sliding_window_view(obs, window_shape=horizon)
            valid = future_obs[window : window + start_idxs.size].any(axis=1)
            start_idxs = start_idxs[valid]
        if start_idxs.size == 0:
            continue
        end_times = times_arr[start_idxs + window - 1]
        pairs.append(np.stack([np.full_like(start_idxs, aid), start_idxs], axis=1))
        ends.append(end_times.astype('datetime64[ns]'))

    if not pairs:
        raise RuntimeError("No valid windows across assets after indexing.")

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)    # [M,2]
    end_times    = np.concatenate(ends,  axis=0).astype('datetime64[ns]')  # [M]

    # Persist tiny index
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", end_times)

    # ---- Norm stats (per-ticker, scalar Y) ----
    norm_stats = norm_acc.finalize(assets)
    with paths.norm_stats.open("w") as f:
        json.dump(norm_stats, f)

    # ---- Meta
    meta = {
        'dataset': 'fin_dataset',
        'format': 'indexcache_v1',
        'assets': assets,
        'asset2id': {a:i for i,a in enumerate(assets)},
        'start': start, 'end': end,
        'window': int(window), 'horizon': int(horizon),
        'feature_cols': feature_cols,
        'target_col': target_col,
        'target_cols': [target_col],
        'target_source': 'cache_target',
        'calendar_feature_cols': list(calendar_feature_cols),
        'feature_cfg': {
            'price_fields': feature_cfg.price_fields,
            'returns_mode': feature_cfg.returns_mode,
            'include_rvol': feature_cfg.include_rvol,
            'rvol_span': feature_cfg.rvol_span,
            'rvol_on': feature_cfg.rvol_on,
            'include_dlv': feature_cfg.include_dlv,
            'market_proxy': feature_cfg.market_proxy,
            'include_oc': feature_cfg.include_oc,
            'include_gap': feature_cfg.include_gap,
            'include_hl_range': feature_cfg.include_hl_range,
            'target_field': feature_cfg.target_field,
            'target_col': feature_cfg.target_col,
        },
        'normalize_per_ticker': bool(normalize_per_ticker),
        'clamp_sigma': float(clamp_sigma),
        'min_obs_buffer': int(min_obs_buffer),
        'min_train_coverage': (float(min_train_coverage) if min_train_coverage is not None else None),
        'min_row_coverage': (float(min_row_coverage) if min_row_coverage is not None else None),
        'liquidity_rank_window': liquidity_rank_window,
        'top_n_by_dollar_vol': top_n_by_dollar_vol,
        'max_windows_per_ticker': max_windows_per_ticker,
        'regression': bool(regression),
        'seed': int(seed),
        'keep_time_meta': keep_time_meta,
        'freq': '1D',
        'native_time_scale': '1D',
        'native_time_scale_seconds': 86400.0,
    }
    with paths.meta.open("w") as f:
        json.dump(meta, f, indent=2)

    # Free big refs
    del raw, per_ticker
    gc.collect()

    return True

# --------------------- Reindex-only ---------------------------------------------------------
def rebuild_window_index_only(
    data_dir: str,
    window: int,
    horizon: int,
    max_windows_per_ticker: Optional[int] = None,
    update_meta: bool = True,
    backup_old: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
) -> int:
    """
    Rebuilds windows/global_pairs.npy and windows/end_times.npy for a NEW (K,H)
    using existing per-ticker times. Fast: does NOT touch features/targets.

    Returns the total number of indexed windows.
    """
    import shutil
    paths = CachePaths.from_dir(data_dir)
    times_dir = paths.times
    windows_dir = paths.windows
    meta_path = paths.meta

    with meta_path.open("r") as f:
        meta = json.load(f)
    assets = meta["assets"]
    target_selection = resolve_target_selection(meta, target_col, requested_target_cols=target_cols)
    target_indices = list(target_selection.target_indices)
    feature_target_source = target_selection.target_source != "cache_target"

    pairs_list, ends_list = [], []
    for aid in range(len(assets)):
        tp = times_dir / f"{aid}.npy"
        if not tp.exists():
            continue
        times = np.load(tp)  # datetime64[ns]
        if feature_target_source:
            features = np.load(paths.features / f"{aid}.npy", allow_pickle=False)
            obs_path = paths.obs_masks / f"{aid}.npy"
            if obs_path.exists():
                obs = np.load(obs_path, allow_pickle=False)[:, target_indices].astype(bool, copy=False)
            else:
                obs = np.isfinite(features[:, target_indices])
        else:
            targets = np.load(paths.targets / f"{aid}.npy", allow_pickle=False).astype(np.float32, copy=False)
            if targets.ndim == 1:
                targets = targets[:, None]
            obs_path = paths.obs_masks / f"{aid}.npy"
            if obs_path.exists() and target_indices:
                feature_obs = np.load(obs_path, allow_pickle=False)
                if feature_obs.ndim == 2 and max(target_indices, default=-1) < feature_obs.shape[1]:
                    obs = feature_obs[:, target_indices].astype(bool, copy=False)
                else:
                    obs = np.isfinite(targets)
            else:
                obs = np.isfinite(targets)
        T = int(times.shape[0])
        min_required = window + horizon
        if T < min_required:
            continue
        starts = np.arange(0, T - min_required + 1, dtype=np.int32)
        if max_windows_per_ticker is not None and starts.size > max_windows_per_ticker:
            starts = starts[:max_windows_per_ticker]
        if horizon > 0:
            if obs.ndim == 1:
                obs = obs[:, None]
            obs = obs.any(axis=1)
            future_obs = sliding_window_view(obs, window_shape=horizon)
            valid = future_obs[window : window + starts.size].any(axis=1)
            starts = starts[valid]
        if starts.size == 0:
            continue
        end_times = times[starts + window - 1]
        pairs_list.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        ends_list.append(end_times.astype("datetime64[ns]"))

    if not pairs_list:
        raise RuntimeError("No windows with the requested (window,horizon). Try smaller values.")

    global_pairs = np.concatenate(pairs_list, axis=0).astype(np.int32)
    end_times = np.concatenate(ends_list, axis=0).astype("datetime64[ns]")

    windows_dir.mkdir(parents=True, exist_ok=True)
    gp_path = windows_dir / "global_pairs.npy"
    et_path = windows_dir / "end_times.npy"
    if backup_old:
        for p in (gp_path, et_path):
            if p.exists():
                shutil.move(str(p), str(p.with_suffix(p.suffix + ".bak")))

    np.save(gp_path, global_pairs)
    np.save(et_path, end_times)

    if update_meta:
        meta["window"] = int(window)
        meta["horizon"] = int(horizon)
        if "target_cols" not in meta and meta.get("target_col"):
            meta["target_cols"] = [str(meta["target_col"])]
        meta["selected_target_cols"] = list(target_selection.target_cols)
        meta["selected_target_indices"] = list(target_selection.target_indices)
        meta["selected_target_dim"] = int(target_selection.target_dim)
        meta["selected_target_source"] = target_selection.target_source
        meta["requested_target_cols"] = (
            list(target_selection.requested_target_cols)
            if target_selection.requested_target_cols is not None
            else None
        )
        meta["calendar_feature_cols"] = list(target_selection.calendar_feature_cols)
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

    return int(global_pairs.shape[0])

# --------------------- Utility samplers & collates (grouping preserved) ---------------------

class _ListBatchSampler(_Sampler):
    def __init__(self, batches: Sequence[Sequence[int]]):
        self.batches = [list(b) for b in batches if len(b)]
    def __iter__(self):
        for b in self.batches:
            yield b
    def __len__(self):
        return len(self.batches)

def make_collate_level_and_firstdiff(
    n_entities: int,
    return_entity_mask: bool = True,
):
    """
    Replacement collate_fn that keeps metadata used by LLapDiff training/eval:
      - entity_mask: [B,N] bool
      - x_obs_mask:  [B,N,K] or [B,N,K,F] (if present)
      - y_obs_mask:  [B,N,H] or [B,N,H,C] bool (if present)
      - delta_t:     [B,N,K] float
      - delta_t_y:   [B,N,H] float

    This function is intentionally tolerant of slightly different sample dict schemas.
    """

    def _as_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _pick(sample, *keys, default=None):
        for k in keys:
            if isinstance(sample, dict) and k in sample:
                return sample[k]
        return default

    def _safe_stack(arrs, dtype=None):
        if len(arrs) == 0:
            return None
        out = np.stack(arrs, axis=0)
        return out.astype(dtype, copy=False) if dtype is not None else out

    def collate(batch):
        # ---------- Normalize each item to a dict ----------
        items = []
        for s in batch:
            if isinstance(s, dict):
                items.append(s)
                continue
            # fallback common tuple formats
            if isinstance(s, (tuple, list)):
                if len(s) == 3 and isinstance(s[2], dict):
                    (x_payload, y, meta) = s
                    if isinstance(x_payload, (tuple, list)) and len(x_payload) == 2:
                        V, T = x_payload
                    else:
                        V = x_payload
                        if torch.is_tensor(V):
                            T = torch.zeros_like(V)
                            if V.dim() >= 2 and V.size(0) > 1:
                                T[1:] = V[1:] - V[:-1]
                        else:
                            V = np.asarray(V, dtype=np.float32)
                            T = np.zeros_like(V)
                            if V.ndim >= 2 and V.shape[0] > 1:
                                T[1:] = V[1:] - V[:-1]
                    d = dict(meta)
                    d.setdefault("V", V)
                    d.setdefault("T", T)
                    d.setdefault("y", y)
                    items.append(d)
                    continue
            raise TypeError(f"Unsupported sample type for collate: {type(s)}")

        # ---------- Required tensors ----------
        V_list, T_list, y_list = [], [], []
        asset_ids = []
        window_starts = []
        ctx_times_list, y_times_list = [], []
        delta_t_list, delta_t_y_list = [], []
        x_obs_list, y_obs_list = [], []

        for s in items:
            V = _as_np(_pick(s, "V", "x_level", "x_levels"))
            T = _as_np(_pick(s, "T", "x_diff", "x_firstdiff"))
            y = _as_np(_pick(s, "y", "target", "targets"))
            if V is None or T is None or y is None:
                raise KeyError("Sample must contain V/T/y (or aliases).")

            # expected shapes per entity sample: [K,F], [K,F], [H] or [H,C]
            if y.ndim == 2 and y.shape[-1] == 1:
                y = y[..., 0]

            V_list.append(V.astype(np.float32, copy=False))
            T_list.append(T.astype(np.float32, copy=False))
            y_list.append(y.astype(np.float32, copy=False))

            asset_ids.append(int(_pick(s, "asset_id", "entity_id", default=-1)))
            window_starts.append(int(_pick(s, "start_idx", "window_start", default=-1)))

            ctx_times = _pick(s, "ctx_times", "times_x", "x_times")
            y_times = _pick(s, "y_times", "times_y", "target_times")
            ctx_times_arr = _as_np(ctx_times) if ctx_times is not None else None
            context_end_key = _pick(s, "context_end_time_key", "context_end_key")
            if ctx_times_arr is None and context_end_key is not None:
                key_arr = np.asarray(context_end_key).reshape(-1)
                if key_arr.size:
                    try:
                        ctx_times_arr = np.asarray([int(key_arr[0])], dtype="datetime64[ns]")
                    except Exception:
                        ctx_times_arr = key_arr[:1]
            ctx_times_list.append(ctx_times_arr)
            y_times_list.append(_as_np(y_times) if y_times is not None else None)

            # Preserve explicit relative deltas when provided by the dataset.
            dt_ctx = _pick(s, "delta_t", "dt_x", "x_delta_t")
            dt_y = _pick(s, "delta_t_y", "dt_y", "y_delta_t")
            delta_t_list.append(_as_np(dt_ctx) if dt_ctx is not None else None)
            delta_t_y_list.append(_as_np(dt_y) if dt_y is not None else None)

            x_obs = _pick(s, "x_obs_mask", "obs_mask_x")
            y_obs = _pick(s, "y_obs_mask", "obs_mask_y")
            x_obs_list.append(_as_np(x_obs) if x_obs is not None else None)
            y_obs_list.append(_as_np(y_obs) if y_obs is not None else None)

        # Infer dimensions
        K, F = V_list[0].shape
        H = y_list[0].shape[0]
        y_tail_shape = tuple(y_list[0].shape[1:])

        def _infer_batch_key(ctx_times, y_times, delta_t_y):
            time_key = None
            for raw_times in (ctx_times, y_times):
                if raw_times is None:
                    continue
                arr = np.asarray(raw_times)
                if arr.size == 0:
                    continue
                try:
                    time_key = int(np.asarray(arr.reshape(-1)[-1]).astype("datetime64[ns]").astype(np.int64))
                    break
                except Exception:
                    continue
            sig = None
            if delta_t_y is not None:
                dt_y_arr = np.asarray(delta_t_y, dtype=np.float32).reshape(-1)
                if dt_y_arr.size:
                    sig = _query_grid_signature(dt_y_arr)
            if time_key is None and sig is None:
                return None
            return (time_key, sig)

        batch_keys = [
            _infer_batch_key(ctx_times_list[j], y_times_list[j], delta_t_y_list[j])
            for j in range(len(items))
        ]
        batch_rows = []
        batch_time_keys = []
        if batch_keys and all(batch_key is not None for batch_key in batch_keys):
            row_by_time = {}
            for batch_key in batch_keys:
                if batch_key not in row_by_time:
                    row_by_time[batch_key] = len(batch_time_keys)
                    batch_time_keys.append(batch_key[0])
                batch_rows.append(row_by_time[batch_key])
            B = len(batch_time_keys)
        else:
            B = 1
            batch_rows = [0] * len(items)
            batch_time_keys = None

        V = np.stack(V_list, axis=0)   # [M,K,F]
        T = np.stack(T_list, axis=0)   # [M,K,F]
        y = np.stack(y_list, axis=0)   # [M,H] or [M,H,C]

        # entity mask: mark seen asset_ids
        entity_mask = np.zeros((B, int(n_entities)), dtype=bool)
        cache_asset_ids = np.full((B, int(n_entities)), -1, dtype=np.int64)
        cache_window_starts = np.full((B, int(n_entities)), -1, dtype=np.int64)
        V_full = np.zeros((B, int(n_entities), K, F), dtype=np.float32)
        T_full = np.zeros((B, int(n_entities), K, F), dtype=np.float32)
        y_full_shape = (B, int(n_entities), H, *y_tail_shape)
        y_full = np.zeros(y_full_shape, dtype=np.float32)

        x_obs_full = None
        y_obs_full = None
        if any(m is not None for m in x_obs_list):
            # preserve feature-wise masks if provided
            xm = x_obs_list[0]
            if xm is None:
                x_obs_full = np.zeros((B, int(n_entities), K), dtype=bool)
            else:
                x_obs_full = np.zeros((B, int(n_entities), *np.asarray(xm).shape), dtype=bool)
        if any(m is not None for m in y_obs_list):
            y_obs_full = np.zeros(y_full_shape, dtype=bool)

        delta_t = np.zeros((B, int(n_entities), K), dtype=np.float32)
        delta_t_y = np.zeros((B, int(n_entities), H), dtype=np.float32)

        for j, aid in enumerate(asset_ids):
            row = batch_rows[j]
            if not (0 <= aid < int(n_entities)):
                # fallback: place sequentially if asset ids are unavailable
                aid = j if j < int(n_entities) else None
            if aid is None:
                continue

            entity_mask[row, aid] = True
            cache_asset_ids[row, aid] = int(asset_ids[j])
            cache_window_starts[row, aid] = int(window_starts[j])
            V_full[row, aid] = V[j]
            T_full[row, aid] = T[j]
            y_full[row, aid] = y[j]

            if x_obs_full is not None:
                xo = x_obs_list[j]
                if xo is None:
                    # infer from finiteness
                    if x_obs_full.ndim == 3:
                        x_obs_full[row, aid] = np.isfinite(V[j]).all(axis=-1)
                    else:
                        x_obs_full[row, aid] = np.isfinite(V[j])
                else:
                    x_obs_full[row, aid] = np.asarray(xo, dtype=bool)

            if y_obs_full is not None:
                yo = y_obs_list[j]
                if yo is None:
                    y_obs_full[row, aid] = np.isfinite(y[j])
                else:
                    y_obs_full[row, aid] = np.asarray(yo, dtype=bool)

            # relative time offsets (prefer explicit deltas if provided)
            dt_ctx = delta_t_list[j]
            if dt_ctx is not None:
                dt_ctx_arr = np.asarray(dt_ctx, dtype=np.float32).reshape(-1)
                if dt_ctx_arr.shape[0] == K:
                    delta_t[row, aid] = dt_ctx_arr
            else:
                ctx_t = ctx_times_list[j]
                if ctx_t is not None and len(ctx_t) == K:
                    # NOTE: native scale is expected to be already handled upstream if you store relative units.
                    delta_t[row, aid] = _compute_relative_time_deltas(ctx_t, native_scale_seconds=1.0)

            dt_y = delta_t_y_list[j]
            if dt_y is not None:
                dt_y_arr = np.asarray(dt_y, dtype=np.float32).reshape(-1)
                if dt_y_arr.shape[0] == H:
                    delta_t_y[row, aid] = dt_y_arr
            else:
                yt = y_times_list[j]
                if yt is not None and len(yt) == H:
                    ctx_t = ctx_times_list[j]
                    anchor = None
                    if ctx_t is not None:
                        ctx_arr = np.asarray(ctx_t)
                        if ctx_arr.size > 0:
                            anchor = ctx_arr.reshape(-1)[-1]
                    if anchor is None:
                        raise ValueError("Cannot infer delta_t_y from y_times without ctx_times or explicit delta_t_y")
                    delta_t_y[row, aid] = _compute_time_offsets_from_anchor(yt, anchor, native_scale_seconds=1.0)

        for row in range(B):
            valid = entity_mask[row]
            if int(valid.sum()) <= 1:
                continue
            grids = delta_t_y[row, valid]
            if not np.allclose(grids, grids[:1], rtol=1e-5, atol=1e-6):
                raise ValueError(
                    "collate produced incompatible delta_t_y query grids for one joint row; "
                    "date batching must group entities by context end and future query grid"
                )

        xb = (
            torch.from_numpy(V_full),  # [B,N,K,F]
            torch.from_numpy(T_full),  # [B,N,K,F]
        )
        yb = torch.from_numpy(y_full)    # [B,N,H]

        meta = {
            "entity_mask": torch.from_numpy(entity_mask),
            "delta_t": torch.from_numpy(delta_t),
            "delta_t_y": torch.from_numpy(delta_t_y),
            "cache_asset_ids": torch.from_numpy(cache_asset_ids),
            "cache_window_starts": torch.from_numpy(cache_window_starts),
        }
        if batch_time_keys is not None and all(key is not None for key in batch_time_keys):
            time_keys = np.asarray(batch_time_keys, dtype=np.int64)
            meta["context_end_time_keys"] = torch.from_numpy(time_keys)
            meta["date_keys"] = torch.from_numpy(time_keys // np.int64(24 * 60 * 60 * 1_000_000_000))
        if x_obs_full is not None:
            meta["x_obs_mask"] = torch.from_numpy(x_obs_full)
        if y_obs_full is not None:
            meta["y_obs_mask"] = torch.from_numpy(y_obs_full)

        if return_entity_mask:
            return xb, yb, meta
        return xb, yb

    return collate

# --------------------- Datasets (index-backed on-the-fly windows) ---------------------

class _IndexBackedDataset(Dataset):
    """On-the-fly slicer from compact per-ticker arrays using stored (asset_id, start_idx).
    Applies (optional) per-ticker normalization + clamp.
    """
    def __init__(self,
                 pairs: np.ndarray,               # [N,2] int32 (aid, start)
                 assets: List[str],
                 data_dir: str,
                 window: int,
                 horizon: int,
                 regression: bool,
                 keep_time_meta: str,
                 norm_stats: dict,
                 clamp_sigma: float,
                 native_time_scale_seconds: float = 1.0,
                 native_time_scale_name: str = "1s",
                 target_index: int = 0,
                 target_indices: Optional[Sequence[int]] = None,
                 target_dim: int = 1,
                 target_source: str = "cache_target",
                 context_missingness_rate: float = 0.0,
                 seed: int = 1337,
                 ): 
        self.pairs = pairs
        self.assets = assets
        self.data_dir = data_dir
        self.paths = CachePaths.from_dir(data_dir)
        self.window = int(window)
        self.horizon = int(horizon)
        self.regression = bool(regression)
        self.keep_time_meta = keep_time_meta
        self.clamp_sigma = float(clamp_sigma)
        nts = float(native_time_scale_seconds)
        if (not np.isfinite(nts)) or nts <= 0:
            nts = 1.0
        self.native_time_scale_seconds = nts
        self.native_time_scale_name = str(native_time_scale_name)
        if target_indices is None:
            target_indices = (int(target_index),)
        self.target_indices = tuple(int(idx) for idx in target_indices)
        if not self.target_indices:
            raise ValueError("target_indices must be non-empty.")
        self.target_index = int(self.target_indices[0])
        self.target_dim = len(self.target_indices)
        self.target_source = str(target_source)
        self.context_missingness_rate = _validate_context_missingness_rate(
            context_missingness_rate,
            name="coverage",
        )
        self.seed = int(seed)

        # Cache per-asset arrays lazily. We avoid mmap here because keeping
        # train/val/test datasets alive at once can exhaust the default open-file
        # limit when many assets are present.
        self._X: Dict[int, np.ndarray] = {}
        self._Y: Dict[int, np.ndarray] = {}
        self._T: Dict[int, np.ndarray] = {}
        self._OBS: Dict[int, Optional[np.ndarray]] = {}
        self._FILL: Dict[int, Optional[np.ndarray]] = {}

        self.per_ticker = bool(norm_stats.get('per_ticker', True))
        self.mean_x = norm_stats['mean_x']
        self.std_x  = norm_stats['std_x']
        self.mean_y = norm_stats['mean_y']
        self.std_y  = norm_stats['std_y']

    def __len__(self):
        return self.pairs.shape[0]

    def _get_arrays(self, aid: int):
        if aid not in self._X:
            self._X[aid] = np.load(self.paths.features / f"{aid}.npy", allow_pickle=False)
            self._Y[aid] = np.load(self.paths.targets / f"{aid}.npy", allow_pickle=False)
            self._T[aid] = np.load(self.paths.times / f"{aid}.npy", allow_pickle=False)
            obs_path = self.paths.obs_masks / f"{aid}.npy"
            fill_path = self.paths.fill_masks / f"{aid}.npy"
            self._OBS[aid] = np.load(obs_path, allow_pickle=False) if obs_path.exists() else None
            self._FILL[aid] = np.load(fill_path, allow_pickle=False) if fill_path.exists() else None
        return (
            self._X[aid],
            self._Y[aid],
            self._T[aid],
            self._OBS.get(aid),
            self._FILL.get(aid),
        )

    def __getitem__(self, i: int):
        aid, start = self.pairs[i]
        Xf, Yf, Tf, Obs, Fill = self._get_arrays(int(aid))
        s = int(start)
        e = s + self.window
        x = Xf[s:e, :].astype(np.float32)  # [K,F]
        obs_slice = Obs[s:e] if Obs is not None else None
        if obs_slice is None:
            obs_slice = np.isfinite(x)
        fill_slice = Fill[s:e] if Fill is not None else obs_slice
        hidden_context_mask = None
        context_missingness_rate = float(getattr(self, "context_missingness_rate", 0.0))
        if context_missingness_rate > 0.0:
            obs_slice, hidden_context_mask = _apply_context_missingness(
                obs_slice,
                context_missingness_rate,
                int(getattr(self, "seed", 1337)),
                int(aid),
                s,
            )
        target_source = getattr(self, "target_source", "cache_target")
        target_indices = tuple(int(idx) for idx in getattr(self, "target_indices", (getattr(self, "target_index", 0),)))
        target_dim = int(getattr(self, "target_dim", len(target_indices) or 1))
        
        if target_source in {"feature_column", "feature_columns"}:
            y_raw = Xf[e:e+self.horizon, list(target_indices)].astype(np.float32)
            if Obs is not None and Obs.ndim == 2 and max(target_indices, default=-1) < Obs.shape[1]:
                y_obs_mask = Obs[e:e+self.horizon, list(target_indices)].astype(bool)
            else:
                y_obs_mask = np.isfinite(y_raw)
            y_vec = y_raw
        else:
            y_vec = Yf[e:e+self.horizon].astype(np.float32)  # [H]
            if Obs is not None and Obs.ndim == 2 and max(target_indices, default=-1) < Obs.shape[1]:
                y_obs_mask = Obs[e:e+self.horizon, list(target_indices)].astype(bool)
            else:
                y_obs_mask = np.isfinite(y_vec)
        if target_dim == 1 and np.ndim(y_vec) == 2:
            y_vec = y_vec[:, 0]
        if target_dim == 1 and np.ndim(y_obs_mask) == 2:
            y_obs_mask = y_obs_mask[:, 0]
        y_for_label = np.asarray(y_vec)
        raw_last_for_label = (
            float(y_for_label.reshape(y_for_label.shape[0], -1)[-1, 0]) if y_for_label.size else 0.0
        )


        # Normalize + clamp (train-style). Use per-ticker or global stats
        if self.per_ticker:
            mx = np.array(self.mean_x[aid], dtype=np.float32)   # [1,1,F]
            sx = np.array(self.std_x[aid],  dtype=np.float32)
            if target_source in {"feature_column", "feature_columns"}:
                my = mx.reshape(-1)[list(target_indices)].astype(np.float32)
                sy = sx.reshape(-1)[list(target_indices)].astype(np.float32)
            else:
                my = float(self.mean_y[aid]) if isinstance(self.mean_y, list) else float(self.mean_y)
                sy = float(self.std_y[aid])  if isinstance(self.std_y, list)  else float(self.std_y)
        else:
            mx = np.array(self.mean_x, dtype=np.float32)        # [1,1,F]
            sx = np.array(self.std_x,  dtype=np.float32)
            if target_source in {"feature_column", "feature_columns"}:
                my = mx.reshape(-1)[list(target_indices)].astype(np.float32)
                sy = sx.reshape(-1)[list(target_indices)].astype(np.float32)
            else:
                my = float(self.mean_y)
                sy = float(self.std_y)
        my = np.asarray(my, dtype=np.float32)
        sy = np.asarray(sy, dtype=np.float32)
        sy = np.where(np.isfinite(sy) & (sy != 0.0), sy, 1.0).astype(np.float32)
        if target_dim == 1:
            my = float(my.reshape(-1)[0])
            sy = float(sy.reshape(-1)[0])
        elif y_vec.ndim == 1:
            y_vec = y_vec[:, None]
            y_obs_mask = y_obs_mask[:, None]
        if np.isscalar(sy) and (not np.isfinite(sy) or sy == 0.0):
            sy = 1.0
        lo, hi = mx - self.clamp_sigma * sx, mx + self.clamp_sigma * sx
        x = np.clip(x, lo[0,0], hi[0,0], out=x)
        x = (x - mx[0,0]) / sx[0,0]
        lo_y, hi_y = my - self.clamp_sigma * sy, my + self.clamp_sigma * sy
        y_vec = np.clip(y_vec, lo_y, hi_y, out=y_vec)
        y_vec = (y_vec - my) / sy
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        _zero_hidden_context_values(x, hidden_context_mask)
        y_vec = np.nan_to_num(y_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Torch tensors expected by collate: X->float32, Y->float32 or int
        x_t = torch.tensor(x, dtype=torch.float32)
        if self.regression:
            y_t = torch.tensor(y_vec, dtype=torch.float32)
        else:
            y_t = torch.tensor(int(raw_last_for_label > 0.0), dtype=torch.int64)

        meta = {
            'asset_id': int(aid),
            'asset': self.assets[int(aid)],
            'start_idx': int(s),
            'window_start': int(s),
        }
        # Relative time deltas for context (K) and forecast horizon (H).
        # Always store these because diffusion/Laplace evaluation needs dt for irregular sampling.
        meta['delta_t'] = _compute_relative_time_deltas(Tf[s:e], self.native_time_scale_seconds)
        meta['delta_t_y'] = _compute_time_offsets_from_anchor(
            Tf[e:e+self.horizon],
            Tf[e-1],
            self.native_time_scale_seconds,
        )
        try:
            meta['context_end_time_key'] = int(np.asarray(Tf[e-1]).astype("datetime64[ns]").astype(np.int64))
        except Exception:
            meta['context_end_time_key'] = Tf[e-1]
        if self.keep_time_meta != 'none':
            if self.keep_time_meta == 'full':
                meta['ctx_times'] = Tf[s:e]
                meta['y_times'] = Tf[e:e+self.horizon]
            else:
                meta['ctx_times'] = Tf[e-1]
                meta['y_times']  = Tf[e+self.horizon-1]
        if obs_slice is not None:
            meta['x_obs_mask'] = obs_slice
        if fill_slice is not None:
            meta['x_mask'] = fill_slice
        meta['y_obs_mask'] = y_obs_mask
        meta['delta_t_unit'] = self.native_time_scale_name
        return x_t, y_t, meta

# --------------------- Date grouping helpers for ratio-split ---------------------


def _normalize_to_day(ts: np.ndarray) -> np.ndarray:
    """Convert a datetime64 array to int-day keys safely."""
    return ts.astype('datetime64[D]').astype(np.int64)


def _query_grid_signature(delta_t_y: np.ndarray) -> bytes:
    """Return a stable signature for a target query grid."""
    grid = np.asarray(delta_t_y, dtype=np.float64).reshape(-1)
    if grid.size and not np.isfinite(grid).all():
        raise ValueError("delta_t_y contains non-finite offsets")
    rounded = np.round(grid, decimals=6).astype(np.float32, copy=False)
    return hashlib.blake2b(rounded.tobytes(), digest_size=16).digest()


def _future_query_grid_signature(
    times: np.ndarray,
    start: int,
    window: int,
    horizon: int,
    native_time_scale_seconds: float,
) -> bytes:
    """Compute the query-grid signature used by _IndexBackedDataset for one row."""
    if int(horizon) <= 0:
        return _query_grid_signature(np.empty((0,), dtype=np.float32))
    s = int(start)
    e = s + int(window)
    future = np.asarray(times[e:e + int(horizon)])
    if future.shape[0] != int(horizon):
        raise ValueError(
            f"Cannot build future query grid for start={s}: "
            f"expected horizon={int(horizon)}, got {future.shape[0]}"
        )
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, np.datetime64):
        future_ns = future.astype("datetime64[ns]").astype(np.int64)
        anchor_ns = np.asarray(times[e - 1]).astype("datetime64[ns]").astype(np.int64)
        deltas = (future_ns - anchor_ns).astype(np.int64, copy=False)
        return hashlib.blake2b(deltas.tobytes(), digest_size=16).digest()
    if np.issubdtype(arr.dtype, np.number):
        values = future.astype(np.float64, copy=False)
        anchor = float(np.asarray(times[e - 1], dtype=np.float64).reshape(-1)[0])
        return _query_grid_signature(values - anchor)
    offsets = _compute_time_offsets_from_anchor(
        future,
        np.asarray(times[e - 1]),
        native_time_scale_seconds,
    )
    return _query_grid_signature(offsets)


def _future_query_grid_signatures_for_pairs(
    data_dir: str,
    pairs: np.ndarray,
    window: int,
    horizon: int,
    native_time_scale_seconds: float,
) -> np.ndarray:
    """Compute per-row future query-grid signatures for joint date batching."""
    paths = CachePaths.from_dir(data_dir)
    out: List[bytes] = []
    times_cache: Dict[int, np.ndarray] = {}
    for aid_raw, start_raw in np.asarray(pairs, dtype=np.int64):
        aid = int(aid_raw)
        if aid not in times_cache:
            times_cache[aid] = np.load(paths.times / f"{aid}.npy", allow_pickle=False)
        out.append(
            _future_query_grid_signature(
                times_cache[aid],
                int(start_raw),
                int(window),
                int(horizon),
                float(native_time_scale_seconds),
            )
        )
    return np.asarray(out, dtype=object)


def _build_date_batches_from_pairs(order_pairs: np.ndarray,
                                   end_times: np.ndarray,
                                   dates_per_batch: int,
                                   min_real_entities: int,
                                   exact_timestamp: bool = False,
                                   query_grid_signatures: Optional[Sequence[object]] = None) -> List[np.ndarray]:
    # order_pairs: [M,2] (aid, start) sorted by end_times.
    keys = (
        end_times.astype("datetime64[ns]").astype(np.int64)
        if exact_timestamp
        else _normalize_to_day(end_times)
    )
    if exact_timestamp and query_grid_signatures is not None:
        sigs = np.asarray(query_grid_signatures, dtype=object)
        if sigs.shape[0] != end_times.shape[0]:
            raise ValueError(
                "query_grid_signatures length must match end_times length: "
                f"{sigs.shape[0]} != {end_times.shape[0]}"
            )
        context_keys = end_times.astype("datetime64[ns]").astype(np.int64)
        sort_order = sorted(
            range(len(keys)),
            key=lambda idx: (int(keys[idx]), int(context_keys[idx]), sigs[idx]),
        )
        groups_with_base: List[Tuple[int, np.ndarray]] = []
        start_pos = 0
        while start_pos < len(sort_order):
            first_idx = sort_order[start_pos]
            group_key = (int(context_keys[first_idx]), sigs[first_idx])
            base_key = int(keys[first_idx])
            end_pos = start_pos + 1
            while end_pos < len(sort_order):
                idx = sort_order[end_pos]
                if (int(context_keys[idx]), sigs[idx]) != group_key:
                    break
                end_pos += 1
            group = np.asarray(sort_order[start_pos:end_pos], dtype=np.int64)
            if group.size >= int(min_real_entities):
                groups_with_base.append((base_key, group))
            start_pos = end_pos

        batches: List[np.ndarray] = []
        base_order: List[int] = []
        by_base: Dict[int, List[np.ndarray]] = {}
        for base_key, group in groups_with_base:
            if base_key not in by_base:
                by_base[base_key] = []
                base_order.append(base_key)
            by_base[base_key].append(group)
        for k in range(0, len(base_order), dates_per_batch):
            chunk_keys = base_order[k:k + dates_per_batch]
            chunk = [group for base_key in chunk_keys for group in by_base[base_key]]
            if chunk:
                batches.append(np.concatenate(chunk, axis=0))
        return batches

    order = np.argsort(keys, kind='mergesort')
    keys_sorted = keys[order]
    _, starts = np.unique(keys_sorted, return_index=True)
    groups = np.split(order, starts[1:])
    dense_groups = [g for g in groups if g.size >= int(min_real_entities)]
    batches = []
    for k in range(0, len(dense_groups), dates_per_batch):
        chunk = dense_groups[k:k+dates_per_batch]
        if chunk:
            batches.append(np.concatenate(chunk, axis=0))
    return batches


def _split_counts(n: int, tr: float, vr: float, te: float) -> Tuple[int, int, int]:
    s = float(tr + vr + te)
    if s <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    trn = int(np.floor(n * (tr / s)))
    van = int(np.floor(n * (vr / s)))
    ten = n - trn - van
    if n >= 3:
        if trn == 0:
            trn, ten = 1, ten - 1
        if van == 0 and n - trn >= 2:
            van, ten = 1, ten - 1
        if ten == 0:
            ten = 1
    return trn, van, ten


def _canonical_split_policy(split_policy: str) -> str:
    value = str(split_policy or "global_purged_horizon").strip().lower().replace("-", "_")
    aliases = {
        "purged_horizon": "global_purged_horizon",
        "global_purged": "global_purged_horizon",
        "global_purge": "global_purged_horizon",
        "global": "global_purged_horizon",
        "per_asset_purged": "per_asset_purged_horizon",
        "per_asset_purge": "per_asset_purged_horizon",
        "asset_purged_horizon": "per_asset_purged_horizon",
        "asset_purge": "per_asset_purged_horizon",
        "ratio": "contiguous",
        "legacy": "contiguous",
    }
    value = aliases.get(value, value)
    if value not in {"global_purged_horizon", "per_asset_purged_horizon", "contiguous"}:
        raise ValueError(
            "split_policy must be one of "
            "{'global_purged_horizon', 'per_asset_purged_horizon', 'contiguous'}"
        )
    return value


def _target_interval_times_for_pairs(
    data_dir: PathLike,
    pairs: np.ndarray,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return absolute target interval start/end times for indexed windows."""

    paths = CachePaths.from_dir(data_dir)
    pairs = np.asarray(pairs)
    starts = np.empty(pairs.shape[0], dtype="datetime64[ns]")
    ends = np.empty(pairs.shape[0], dtype="datetime64[ns]")
    times_cache: Dict[int, np.ndarray] = {}
    window = int(window)
    horizon = int(horizon)
    for row, (aid_raw, start_raw) in enumerate(pairs):
        aid = int(aid_raw)
        start = int(start_raw)
        if aid not in times_cache:
            times_cache[aid] = np.load(paths.times / f"{aid}.npy", allow_pickle=False).astype("datetime64[ns]")
        times = times_cache[aid]
        if horizon > 0:
            first = start + window
            last = first + horizon - 1
        else:
            first = last = start + window - 1
        if first < 0 or last >= times.shape[0]:
            raise ValueError(
                f"Window (asset={aid}, start={start}, window={window}, horizon={horizon}) "
                f"exceeds cached time axis length {times.shape[0]}."
            )
        starts[row] = times[first]
        ends[row] = times[last]
    return starts, ends


def _coerce_target_interval_times(
    target_start_times: Optional[np.ndarray],
    target_end_times: Optional[np.ndarray],
    n_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_start_times is None or target_end_times is None:
        raise ValueError("target interval times are required for purged split policies")
    starts = np.asarray(target_start_times).astype("datetime64[ns]").astype(np.int64)
    ends = np.asarray(target_end_times).astype("datetime64[ns]").astype(np.int64)
    if starts.shape != (n_rows,) or ends.shape != (n_rows,):
        raise ValueError(
            "target interval time arrays must have shape "
            f"({n_rows},), got {starts.shape} and {ends.shape}"
        )
    if np.any(ends < starts):
        raise ValueError("target interval end times must be greater than or equal to start times")
    return starts, ends


def _target_interval_boundaries(
    target_starts: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int]:
    unique_starts = np.unique(target_starts)
    if unique_starts.size < 3:
        raise ValueError(f"Not enough unique target timestamps for purged split: {unique_starts.size}")
    trn, van, _ = _split_counts(int(unique_starts.size), train_ratio, val_ratio, test_ratio)
    test_idx = trn + van
    if trn <= 0 or van <= 0 or test_idx >= unique_starts.size:
        raise ValueError(f"Could not form non-empty purged splits from {unique_starts.size} target timestamps")
    return int(unique_starts[trn]), int(unique_starts[test_idx])


def _assign_by_target_interval_boundaries(
    assign: np.ndarray,
    rows: np.ndarray,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    val_start, test_start = _target_interval_boundaries(
        target_starts[rows],
        train_ratio,
        val_ratio,
        test_ratio,
    )
    row_starts = target_starts[rows]
    row_ends = target_ends[rows]
    assign[rows[row_ends < val_start]] = 0
    assign[rows[(row_starts >= val_start) & (row_ends < test_start)]] = 1
    assign[rows[row_starts >= test_start]] = 2


def _assign_ratio_splits(
    pairs: np.ndarray,
    end_times: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    per_asset: bool,
    split_policy: str,
    horizon: int,
    target_start_times: Optional[np.ndarray] = None,
    target_end_times: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assign rows to train/val/test using the configured chronological policy."""

    assign = np.full(pairs.shape[0], 255, dtype=np.uint8)
    policy = _canonical_split_policy(split_policy)

    if pairs.size == 0:
        return assign

    if policy == "global_purged_horizon":
        target_starts, target_ends = _coerce_target_interval_times(
            target_start_times,
            target_end_times,
            pairs.shape[0],
        )
        _assign_by_target_interval_boundaries(
            assign,
            np.arange(pairs.shape[0], dtype=np.int64),
            target_starts,
            target_ends,
            train_ratio,
            val_ratio,
            test_ratio,
        )
        if any(int((assign == split).sum()) == 0 for split in (0, 1, 2)):
            raise ValueError("target-interval purged split produced an empty train/val/test split")
        return assign

    aids = pairs[:, 0].astype(np.int64)
    if policy == "per_asset_purged_horizon":
        target_starts, target_ends = _coerce_target_interval_times(
            target_start_times,
            target_end_times,
            pairs.shape[0],
        )
        for aid in np.unique(aids):
            idx = np.nonzero(aids == aid)[0]
            if np.unique(target_starts[idx]).size < 3:
                continue
            _assign_by_target_interval_boundaries(
                assign,
                idx,
                target_starts,
                target_ends,
                train_ratio,
                val_ratio,
                test_ratio,
            )
        return assign

    if per_asset:
        for aid in np.unique(aids):
            idx = np.nonzero(aids == aid)[0]
            trn, van, _ = _split_counts(int(idx.size), train_ratio, val_ratio, test_ratio)
            assign[idx[:trn]] = 0
            assign[idx[trn:trn + van]] = 1
            assign[idx[trn + van:]] = 2
    else:
        trn, van, _ = _split_counts(int(pairs.shape[0]), train_ratio, val_ratio, test_ratio)
        assign[:trn] = 0
        assign[trn:trn + van] = 1
        assign[trn + van:] = 2
    return assign


def split_policy_name(split_policy: str) -> str:
    """Return the canonical public name for a split policy string."""

    return _canonical_split_policy(split_policy)


def _compute_train_only_norm_stats(
    data_dir: str,
    assets: List[str],
    tr_pairs: np.ndarray,    # [Nt,2] (aid, start)
    window: int,
    horizon: int,
    per_ticker: bool,
    feature_dim: int,
) -> dict | None:
    """
    Compute mean/std for X and Y using ONLY rows that can appear in TRAIN contexts.
    For asset a: use feature prefixes up to max(start + window - 1) and targets up to
    max(start + window + horizon - 1) across train windows.
    Returns a dict like norm_stats.json or None if no train rows exist.
    """
    import numpy as _np

    paths = CachePaths.from_dir(data_dir)

    last_ctx_end = _np.full(len(assets), -1, dtype=_np.int64)
    last_label_end = _np.full(len(assets), -1, dtype=_np.int64)
    eff_horizon = int(max(horizon, 0))
    
    if tr_pairs.size > 0:
        aids = tr_pairs[:, 0].astype(_np.int64)
        ends = tr_pairs[:, 1].astype(_np.int64) + (window - 1)
        for a, ctx_end in zip(aids, ends):
            if ctx_end > last_ctx_end[a]:
                last_ctx_end[a] = ctx_end
            label_end = ctx_end if eff_horizon == 0 else ctx_end + eff_horizon
            if label_end > last_label_end[a]:
                last_label_end[a] = label_end

    has_train = last_ctx_end >= 0

    if per_ticker:
        mean_x, std_x, mean_y, std_y = [], [], [], []

        g_count = _np.zeros((feature_dim,), dtype=_np.int64)
        g_sum = _np.zeros((feature_dim,), dtype=_np.float64)
        g_sumsq = _np.zeros((feature_dim,), dtype=_np.float64)
        g_y_count = 0
        g_y_sum = 0.0
        g_y_sumsq = 0.0

        for aid in range(len(assets)):
            fp = paths.features / f"{aid}.npy"
            yp = paths.targets / f"{aid}.npy"
            if not (fp.exists() and yp.exists()):
                mean_x.append(_np.zeros((1,1,feature_dim), dtype=_np.float32).tolist())
                std_x.append(_np.ones((1,1,feature_dim), dtype=_np.float32).tolist())
                mean_y.append(0.0)
                std_y.append(1.0)
                continue

            if has_train[aid]:
                Xf = _np.load(fp, allow_pickle=False).astype(_np.float32)
                Yf = _np.load(yp, allow_pickle=False).astype(_np.float32)
                upto_x = int(last_ctx_end[aid]) + 1
                upto_y = int(last_label_end[aid]) + 1 if last_label_end[aid] >= 0 else 0
                upto_y = min(upto_y, Yf.shape[0])
                Xp = Xf[:upto_x, :]
                Yp = Yf[:upto_y]

                mx, sx, count_x = _finite_mean_std(Xp, axis=0, keepdims=True)
                my_arr, sy_arr, count_y_arr = _finite_mean_std(Yp, axis=0, keepdims=False)
                if int(count_x.max()) == 0:
                    mx = _np.zeros((1, 1, feature_dim), dtype=_np.float32)
                    sx = _np.ones((1, 1, feature_dim), dtype=_np.float32)
                else:
                    mx = mx[None, ...]
                    sx = sx[None, ...]
                if int(count_y_arr) == 0:
                    my = 0.0
                    sy = 1.0
                else:
                    my = float(my_arr)
                    sy = float(sy_arr)

                mean_x.append(mx.tolist())
                std_x.append(sx.tolist())
                mean_y.append(my)
                std_y.append(sy)

                finite_x = _np.isfinite(Xp)
                g_count += finite_x.sum(axis=0, dtype=_np.int64)
                safe_x = _np.where(finite_x, Xp, 0.0)
                g_sum   += safe_x.sum(axis=0, dtype=_np.float64)
                g_sumsq += (safe_x.astype(_np.float64) ** 2).sum(axis=0)

                finite_y = _np.isfinite(Yp)
                g_y_count += int(finite_y.sum())
                safe_y = _np.where(finite_y, Yp, 0.0)
                g_y_sum   += float(safe_y.sum())
                g_y_sumsq += float((safe_y.astype(_np.float64) ** 2).sum())
            else:
                mean_x.append(None)
                std_x.append(None)
                mean_y.append(None)
                std_y.append(None)

        total_g_count = int(g_count.sum())
        if total_g_count > 0:
            safe_g_count = _np.maximum(g_count, 1)
            g_mx = (g_sum / safe_g_count).astype(_np.float32)
            g_vx = (g_sumsq / safe_g_count) - (g_mx.astype(_np.float64) ** 2)
            g_vx = _np.maximum(g_vx, 1e-12)
            g_sx = _np.sqrt(g_vx).astype(_np.float32)
            g_mx = _np.where(g_count > 0, g_mx, 0.0).astype(_np.float32)
            g_sx = _np.where(g_count > 0, g_sx, 1.0).astype(_np.float32)
            g_sx[g_sx == 0] = 1.0
            if g_y_count > 0:
                g_my = float(g_y_sum / g_y_count)
                g_vy = max((g_y_sumsq / g_y_count) - (g_my ** 2), 1e-12)
                g_sy = float(_np.sqrt(g_vy))
                g_sy = (1.0 if g_sy == 0 else g_sy)
            else:
                g_my = 0.0
                g_sy = 1.0

            for aid in range(len(assets)):
                if mean_x[aid] is None:
                    mean_x[aid] = g_mx.reshape(1,1,-1).tolist()
                    std_x[aid]  = g_sx.reshape(1,1,-1).tolist()
                    mean_y[aid] = g_my
                    std_y[aid] = g_sy
        else:
            return None

        return {
            'per_ticker': True,
            'mean_x': mean_x, 'std_x': std_x,
            'mean_y': mean_y, 'std_y': std_y,
        }

    # global stats
    g_count = _np.zeros((feature_dim,), dtype=_np.int64)
    g_sum = _np.zeros((feature_dim,), dtype=_np.float64)
    g_sumsq = _np.zeros((feature_dim,), dtype=_np.float64)
    g_y_count = 0
    g_y_sum = 0.0
    g_y_sumsq = 0.0

    for aid in range(len(assets)):
        if not has_train[aid]:
            continue
        Xf = _np.load(paths.features / f"{aid}.npy", allow_pickle=False).astype(_np.float32)
        Yf = _np.load(paths.targets / f"{aid}.npy", allow_pickle=False).astype(_np.float32)
        upto_x = int(last_ctx_end[aid]) + 1
        upto_y = int(last_label_end[aid]) + 1 if last_label_end[aid] >= 0 else 0
        upto_y = min(upto_y, Yf.shape[0])
        Xp = Xf[:upto_x, :]
        Yp = Yf[:upto_y]
        finite_x = _np.isfinite(Xp)
        g_count += finite_x.sum(axis=0, dtype=_np.int64)
        safe_x = _np.where(finite_x, Xp, 0.0)
        g_sum   += safe_x.sum(axis=0, dtype=_np.float64)
        g_sumsq += (safe_x.astype(_np.float64) ** 2).sum(axis=0)
        finite_y = _np.isfinite(Yp)
        g_y_count += int(finite_y.sum())
        safe_y = _np.where(finite_y, Yp, 0.0)
        g_y_sum   += float(safe_y.sum())
        g_y_sumsq += float((safe_y.astype(_np.float64) ** 2).sum())

    total_g_count = int(g_count.sum())
    if total_g_count == 0:
        return None

    safe_g_count = _np.maximum(g_count, 1)
    g_mx = (g_sum / safe_g_count).astype(_np.float32)
    g_vx = (g_sumsq / safe_g_count) - (g_mx.astype(_np.float64) ** 2)
    g_vx = _np.maximum(g_vx, 1e-12)
    g_sx = _np.sqrt(g_vx).astype(_np.float32)
    g_mx = _np.where(g_count > 0, g_mx, 0.0).astype(_np.float32)
    g_sx = _np.where(g_count > 0, g_sx, 1.0).astype(_np.float32)
    g_sx[g_sx == 0] = 1.0
    if g_y_count > 0:
        g_my = float(g_y_sum / g_y_count)
        g_vy = max((g_y_sumsq / g_y_count) - (g_my ** 2), 1e-12)
        g_sy = float(_np.sqrt(g_vy))
        g_sy = (1.0 if g_sy == 0 else g_sy)
    else:
        g_my = 0.0
        g_sy = 1.0

    return {
        'per_ticker': False,
        'mean_x': g_mx.reshape(1,1,-1).tolist(),
        'std_x':  g_sx.reshape(1,1,-1).tolist(),
        'mean_y': g_my,
        'std_y':  g_sy,
    }

# --------------------- Ratio-split loader (ONLY) ---------------------

def load_dataloaders_with_ratio_split(
    data_dir: str = './data',
    train_ratio: float = 0.55,
    val_ratio: float = 0.05,
    test_ratio: float = 0.40,
    batch_size: int = 64,
    regression: bool = True,
    per_asset: bool = True,
    norm_scope: str = "train_only",
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    seed: int = 1337,
    n_entities: int = 8,
    pad_incomplete: str = 'zeros',
    collate_fn=None,
    coverage_per_window: float = 0.0,
    date_batching: Optional[bool] = None,
    dates_per_batch: int = 4,
    window: Optional[int] = None,
    horizon: Optional[int] = None,
    split_policy: str = "global_purged_horizon",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
    coverage: float = 0.0,
):
    coverage = _validate_context_missingness_rate(coverage, name="coverage")
    paths = CachePaths.from_dir(data_dir)

    # Read meta + norm
    with paths.meta.open('r') as f:
        meta = json.load(f)
    assets = meta['assets']
    target_selection = resolve_target_selection(meta, target_col, requested_target_cols=target_cols)
    base_window = int(meta['window'])
    base_horizon = int(meta['horizon'])
    window = int(window if window is not None else base_window)
    horizon = int(horizon if horizon is not None else base_horizon)
    if window > base_window or horizon > base_horizon:
        raise ValueError(f"Requested (window={window}, horizon={horizon}) exceed cached meta "
             f"({base_window}, {base_horizon}). Call rebuild_window_index_only(...) first."
            )

    keep_time_meta = meta.get('keep_time_meta', 'end')
    with paths.norm_stats.open('r') as f:
        norm_stats = json.load(f)

    native_time_scale_name, native_time_scale_seconds = _infer_native_time_scale(paths, assets, meta)

    # Collate (levels + first-diff), grouped-by-end if requested later
    if collate_fn is None:
        collate_fn = make_collate_level_and_firstdiff(
            n_entities=len(assets),
            return_entity_mask = True
        )

    
    # Load small global index
    pairs = np.load(paths.windows / 'global_pairs.npy')  # [M,2]
    end_times = np.load(paths.windows / 'end_times.npy')  # [M]
    
    # --- Coverage pre-filter (unconditional) so ratios apply to the filtered set ---
    if coverage_per_window > 0.0:
        days = _normalize_to_day(end_times)
        uniq_days, inv = np.unique(days, return_inverse=True)
    
        # Count unique assets per day via unique rows of (day_idx, asset_id)
        da = np.stack([inv.astype(np.int64), pairs[:, 0].astype(np.int64)], axis=1)
        dau = np.unique(da, axis=0)
        counts = np.bincount(dau[:, 0], minlength=len(uniq_days))
    
        min_real = int(np.ceil(coverage_per_window * len(assets)))  # use full panel width
        keep_days = counts >= max(1, min_real)
        keep_mask = keep_days[inv]
    
        pairs = pairs[keep_mask]
        end_times = end_times[keep_mask]
    
        # Ensure chronological ordering within asset (pairs might already be grouped, but do it anyway)
        # We'll sort by (asset_id, end_time)
        
    aid = pairs[:, 0].astype(np.int32)
    policy = _canonical_split_policy(split_policy)
    if policy == "global_purged_horizon":
        order = np.argsort(end_times.astype('datetime64[ns]').astype(np.int64), kind='mergesort')
    elif per_asset:
        order = np.lexsort((end_times.astype('datetime64[ns]').astype(np.int64), aid))
    else:
        order = np.argsort(end_times.astype('datetime64[ns]').astype(np.int64), kind='mergesort')
    pairs = pairs[order]
    end_times = end_times[order]
    aid = pairs[:, 0].astype(np.int32)
    target_start_times = target_end_times = None
    if policy in {"global_purged_horizon", "per_asset_purged_horizon"}:
        target_start_times, target_end_times = _target_interval_times_for_pairs(
            data_dir,
            pairs,
            window,
            horizon,
        )
    
    # Ratio assignment
    assign = _assign_ratio_splits(
        pairs,
        end_times,
        train_ratio,
        val_ratio,
        test_ratio,
        per_asset=per_asset,
        split_policy=policy,
        horizon=horizon,
        target_start_times=target_start_times,
        target_end_times=target_end_times,
    )
    
    tr_pairs = pairs[assign == 0]
    va_pairs = pairs[assign == 1]
    te_pairs = pairs[assign == 2]

    # ---- Train-only normalization (optional) ----
    if isinstance(norm_scope, str) and norm_scope.lower() == "train_only":
        per_ticker_flag = bool(norm_stats.get('per_ticker', meta.get('normalize_per_ticker', True)))
        feature_dim = len(meta.get('feature_cols', []))
        tr_norm = _compute_train_only_norm_stats(
            data_dir=data_dir,
            assets=assets,
            tr_pairs=tr_pairs,
            window=window,
            horizon=horizon,
            per_ticker=per_ticker_flag,
            feature_dim=feature_dim,
        )
        if tr_norm is not None:
            norm_stats = tr_norm
    
    ds_tr = _IndexBackedDataset(tr_pairs, assets, data_dir, window, horizon, regression,
                                keep_time_meta, norm_stats, clamp_sigma=float(meta.get('clamp_sigma', 5.0)),
                                native_time_scale_seconds=float(native_time_scale_seconds),
                                native_time_scale_name=str(native_time_scale_name),
                                target_index=target_selection.target_index,
                                target_indices=target_selection.target_indices,
                                target_dim=target_selection.target_dim,
                                target_source=target_selection.target_source,
                                context_missingness_rate=coverage,
                                seed=seed)
    ds_va = _IndexBackedDataset(va_pairs, assets, data_dir, window, horizon, regression,
                                keep_time_meta, norm_stats, clamp_sigma=float(meta.get('clamp_sigma', 5.0)),
                                native_time_scale_seconds=float(native_time_scale_seconds),
                                native_time_scale_name=str(native_time_scale_name),
                                target_index=target_selection.target_index,
                                target_indices=target_selection.target_indices,
                                target_dim=target_selection.target_dim,
                                target_source=target_selection.target_source,
                                context_missingness_rate=coverage,
                                seed=seed)
    ds_te = _IndexBackedDataset(te_pairs, assets, data_dir, window, horizon, regression,
                                keep_time_meta, norm_stats, clamp_sigma=float(meta.get('clamp_sigma', 5.0)),
                                native_time_scale_seconds=float(native_time_scale_seconds),
                                native_time_scale_name=str(native_time_scale_name),
                                target_index=target_selection.target_index,
                                target_indices=target_selection.target_indices,
                                target_dim=target_selection.target_dim,
                                target_source=target_selection.target_source,
                                context_missingness_rate=coverage,
                                seed=seed)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Optional date-aware batching (uses end_times filtered by split)
    if date_batching is None:
        date_batching = (coverage_per_window > 0.0)
    if date_batching:
        min_real = max(1, int(_ceil(coverage_per_window * len(assets)))) if coverage_per_window > 0 else 1
        query_grid_signatures = (
            _future_query_grid_signatures_for_pairs(
                data_dir,
                pairs,
                window,
                horizon,
                float(native_time_scale_seconds),
            )
            if exact_timestamp_batches
            else None
        )
        # recover split-specific end_times by masking
        tr_mask = (assign == 0)
        va_mask = (assign == 1)
        te_mask = (assign == 2)
        batches_tr = _build_date_batches_from_pairs(
            tr_pairs,
            end_times[tr_mask],
            dates_per_batch,
            min_real,
            exact_timestamp=exact_timestamp_batches,
            query_grid_signatures=(
                query_grid_signatures[tr_mask] if query_grid_signatures is not None else None
            ),
        )
        batches_va = _build_date_batches_from_pairs(
            va_pairs,
            end_times[va_mask],
            dates_per_batch,
            min_real,
            exact_timestamp=exact_timestamp_batches,
            query_grid_signatures=(
                query_grid_signatures[va_mask] if query_grid_signatures is not None else None
            ),
        )
        batches_te = _build_date_batches_from_pairs(
            te_pairs,
            end_times[te_mask],
            dates_per_batch,
            min_real,
            exact_timestamp=exact_timestamp_batches,
            query_grid_signatures=(
                query_grid_signatures[te_mask] if query_grid_signatures is not None else None
            ),
        )
        train_dl = DataLoader(ds_tr, batch_sampler=_ListBatchSampler(batches_tr), pin_memory=pin_memory,
                              num_workers=num_workers, persistent_workers=False, generator=gen,
                              collate_fn=collate_fn)
        val_dl   = DataLoader(ds_va, batch_sampler=_ListBatchSampler(batches_va), pin_memory=pin_memory,
                              num_workers=num_workers, persistent_workers=False, generator=gen,
                              collate_fn=collate_fn)
        test_dl  = DataLoader(ds_te, batch_sampler=_ListBatchSampler(batches_te), pin_memory=pin_memory,
                              num_workers=num_workers, persistent_workers=False, generator=gen,
                              collate_fn=collate_fn)
    else:
        def _mk(ds, split):
            return DataLoader(
                ds, batch_size=batch_size, shuffle=(split == 'train' and shuffle_train),
                pin_memory=pin_memory, num_workers=num_workers, persistent_workers=False,
                generator=gen, collate_fn=collate_fn,
            )
        train_dl = _mk(ds_tr, 'train')
        val_dl   = _mk(ds_va, 'val')
        test_dl  = _mk(ds_te, 'test')

    return train_dl, val_dl, test_dl, (len(ds_tr), len(ds_va), len(ds_te))


def distinct_end_dates(dl, max_batches=None):
    """Return a sorted list of unique end-dates (YYYY-MM-DD) seen in a DataLoader."""
    seen = set()
    b = 0
    for (_, _), _, meta in dl:
        # Try common meta keys; fall back to computing day keys
        if "date_keys" in meta:  # int days since epoch
            days = pd.to_datetime(meta["date_keys"].cpu().numpy(), unit="D").date
        elif "dates" in meta:    # string/np.datetime64 list
            days = pd.to_datetime(np.array(meta["dates"])).date
        elif "ctx_times" in meta:  # np.datetime64 ns per sample
            days = pd.to_datetime(np.array(meta["ctx_times"])).normalize().date
        elif "end_times" in meta:
            days = pd.to_datetime(np.array(meta["end_times"])).normalize().date
        else:
            # Robust fallback: if no date info is exposed, skip
            days = []

        for d in np.unique(days):
            seen.add(str(d))
        b += 1
        if max_batches and b >= max_batches:
            break
    return sorted(seen)

def run_experiment(
    data_dir: str,
    K: int,
    H: int,
    *,
    ratios=(0.7, 0.1, 0.2),
    per_asset=True,
    date_batching=True,
    coverage=0.0,
    panel_coverage=0.0,
    dates_per_batch=30,
    batch_size=64,
    norm="train_only",    # default is "train_only" in the patched loader; set "cache" if you want fixed μ/σ
    reindex=True,         # set False if NOT using date batching and you don’t need to rebuild end_times
    split_policy: str = "global_purged_horizon",
    exact_timestamp_batches: bool = True,
    target_col: Optional[str] = None,
    target_cols: Optional[Sequence[str]] = None,
):
    """
    Builds loaders for a given (K, H) using the already-downloaded cache.
    - If date_batching=True, reindex first so context end_times align with K/H.
    - coverage induces context missingness; panel_coverage applies the old per-day panel filter.
    """
    coverage = _validate_context_missingness_rate(coverage, name="coverage")
    if reindex:
        rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=True,
            backup_old=False,
            target_col=target_col,
            target_cols=target_cols,
        )

    train_dl, val_dl, test_dl, lengths = load_dataloaders_with_ratio_split(
        data_dir=data_dir,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        batch_size=batch_size,
        regression=True,         # set False for classification
        per_asset=per_asset,     # freely tunable each run
        norm_scope=norm,         # "train_only" (recommended) or "cache"
        date_batching=date_batching,
        coverage_per_window=panel_coverage,
        dates_per_batch=dates_per_batch,
        window=K,
        horizon=H,
        split_policy=split_policy,
        exact_timestamp_batches=exact_timestamp_batches,
        target_col=target_col,
        target_cols=target_cols,
        coverage=coverage,
    )

    return train_dl, val_dl, test_dl, lengths
