"""Summarize prepared dataset caches.

This script inspects the compact cache produced by the dataset utilities in
this repository and reports:

- Number of input channels (feature columns)
- Number of entities
- Train/validation/test step counts (split using the *same ratio logic* as the loader)
- Coverage statistics computed *per day* across cached windows (matches loader's coverage filter)

Example:
    python -m llapdiffusion.datasets.dataset_summary --data-dir ./data --coverage 0.85
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

from llapdiffusion.datasets.fin_dataset import (
    CachePaths,
    _assign_ratio_splits,
    _canonical_split_policy,
    _normalize_to_day,
    _target_interval_times_for_pairs,
    split_policy_name,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class CoverageSummary:
    steps: int
    min: float
    mean: float
    median: float
    max: float


def _load_meta(paths: CachePaths) -> Dict[str, object]:
    """Load cache metadata, trying a couple of common locations."""
    candidates = [
        paths.meta,
        paths.cache_root / "meta.json",              # redundant but explicit
        paths.data_dir / "meta.json",                # older layout (rare)
        paths.data_dir / "cache_ratio_index" / "meta.json",
    ]
    for mp in candidates:
        if mp.exists():
            with mp.open("r") as f:
                return json.load(f)
    raise FileNotFoundError(
        "Cache metadata not found. Tried:\n  - " + "\n  - ".join(str(p) for p in candidates)
    )


def _summarize_coverage(per_step: np.ndarray) -> CoverageSummary:
    if per_step.size == 0:
        return CoverageSummary(steps=0, min=0.0, mean=0.0, median=0.0, max=0.0)
    return CoverageSummary(
        steps=int(per_step.size),
        min=float(per_step.min()),
        mean=float(per_step.mean()),
        median=float(np.median(per_step)),
        max=float(per_step.max()),
    )


def _compute_day_coverage(
    pairs: np.ndarray, end_times: np.ndarray, num_assets: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return coverage per *day* plus mapping info.

    Matches fin_dataset.load_dataloaders_with_ratio_split coverage filter:
    - group by day = normalize_to_day(end_times)
    - count unique (day, asset_id)
    - coverage(day) = count / num_assets

    Returns:
        coverage: [D] float
        inv:      [M] int inverse index mapping each row -> day index
        unique_days: [D] datetime64[D]
        counts:   [D] int unique asset counts per day
    """
    if pairs.size == 0 or end_times.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
            np.array([], dtype="datetime64[D]"),
            np.array([], dtype=np.int64),
        )

    day_keys = _normalize_to_day(end_times.astype("datetime64[ns]"))
    unique_days, inv = np.unique(day_keys, return_inverse=True)

    # unique rows of (day_idx, asset_id)
    day_asset = np.stack([inv.astype(np.int64), pairs[:, 0].astype(np.int64)], axis=1)
    day_asset_unique = np.unique(day_asset, axis=0)
    counts = np.bincount(day_asset_unique[:, 0], minlength=len(unique_days)).astype(np.int64)

    denom = float(max(1, int(num_assets)))
    coverage = counts.astype(np.float64) / denom

    # convert keys back to datetime64[D] for readability
    unique_days_dt = unique_days.astype("datetime64[D]")
    return coverage.astype(np.float64), inv.astype(np.int64), unique_days_dt, counts


def _split_counts(n: int, tr: float, vr: float, te: float) -> Tuple[int, int, int]:
    """Exactly the same split-count logic used in fin_dataset.load_dataloaders_with_ratio_split."""
    s = float(tr + vr + te)
    trn = int(np.floor(n * (tr / s))) if s > 0 else 0
    van = int(np.floor(n * (vr / s))) if s > 0 else 0
    ten = n - trn - van
    if n >= 3:
        if trn == 0:
            trn, ten = 1, ten - 1
        if van == 0 and n - trn >= 2:
            van, ten = 1, ten - 1
        if ten == 0:
            ten = 1
    return trn, van, ten


def _apply_split(
    pairs: np.ndarray,
    end_times: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    per_asset: bool,
    split_policy: str,
    horizon: int,
    data_dir: Path | None = None,
    window: int | None = None,
) -> Tuple[int, int, int]:
    """Replicate the ratio split logic from fin_dataset.load_dataloaders_with_ratio_split."""
    if pairs.size == 0:
        return 0, 0, 0

    aids = pairs[:, 0].astype(np.int32)
    t_int = end_times.astype("datetime64[ns]").astype(np.int64)
    if per_asset:
        order = np.lexsort((t_int, aids))
    else:
        order = np.argsort(t_int)
    pairs = pairs[order]
    end_times = end_times[order]
    policy = _canonical_split_policy(split_policy)
    target_start_times = target_end_times = None
    if policy in {"global_purged_horizon", "per_asset_purged_horizon"}:
        if data_dir is None or window is None:
            raise ValueError("data_dir and window are required to summarize purged target-interval splits")
        target_start_times, target_end_times = _target_interval_times_for_pairs(
            data_dir,
            pairs,
            int(window),
            int(horizon),
        )
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

    return int((assign == 0).sum()), int((assign == 1).sum()), int((assign == 2).sum())


def summarize_dataset(
    data_dir: Path,
    coverage_threshold: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    per_asset: bool,
    split_policy: str,
) -> None:
    paths = CachePaths.from_dir(data_dir)
    meta = _load_meta(paths)
    assets = meta.get("assets", [])
    feature_cols = meta.get("feature_cols", [])
    window = int(meta.get("window", meta.get("max_window", 0)))
    horizon = int(meta.get("horizon", meta.get("max_horizon", 0)))

    pairs_path = paths.windows / "global_pairs.npy"
    end_times_path = paths.windows / "end_times.npy"
    if not pairs_path.exists() or not end_times_path.exists():
        raise FileNotFoundError(
            f"Window index not found. Expected:\n  - {pairs_path}\n  - {end_times_path}\n"
            "Did you prepare the dataset cache?"
        )

    pairs = np.load(pairs_path)
    end_times = np.load(end_times_path)

    # Coverage per *day* (matches loader)
    base_cov, inv_day, unique_days, counts = _compute_day_coverage(pairs, end_times, len(assets))
    base_cov_summary = _summarize_coverage(base_cov)

    kept_pairs = pairs
    kept_end_times = end_times
    filtered_cov_summary = base_cov_summary
    min_real_req: Optional[int] = None
    kept_days: Optional[int] = None

    if coverage_threshold > 0.0 and base_cov.size > 0:
        # loader uses: min_real = ceil(coverage_per_window * num_assets)
        min_real_req = max(1, int(ceil(float(coverage_threshold) * max(1, len(assets)))))
        min_real = int(min_real_req)
        keep_days_mask = counts >= min_real
        keep_rows_mask = keep_days_mask[inv_day]

        kept_pairs = pairs[keep_rows_mask]
        kept_end_times = end_times[keep_rows_mask]

        filtered_cov, *_rest = _compute_day_coverage(kept_pairs, kept_end_times, len(assets))
        filtered_cov_summary = _summarize_coverage(filtered_cov)
        kept_days = int(filtered_cov_summary.steps)

    tr_steps, va_steps, te_steps = _apply_split(
        kept_pairs,
        kept_end_times,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        per_asset=per_asset,
        split_policy=split_policy,
        horizon=horizon,
        data_dir=data_dir,
        window=window,
    )

    print(f"Package root   : {PACKAGE_ROOT}")
    print(f"Data directory : {paths.cache_root}")
    print(f"Dataset        : {meta.get('dataset', 'unknown')}")
    print(f"Entities       : {len(assets)}")
    print(f"Input channels : {len(feature_cols)}")
    print(f"Window/Horizon : {window}/{horizon}")
    print(f"Total windows  : {int(pairs.shape[0])}")
    print()
    print("Coverage per day (window end_times grouped to calendar day):")
    print(
        f"  days={base_cov_summary.steps} "
        f"min={base_cov_summary.min:.3f} "
        f"mean={base_cov_summary.mean:.3f} "
        f"median={base_cov_summary.median:.3f} "
        f"max={base_cov_summary.max:.3f}"
    )
    if coverage_threshold > 0.0:
        print(
            f"Applied coverage >= {coverage_threshold:.2f} "
            f"(min_real={min_real_req} entities/day): "
            f"kept {kept_days} / {base_cov_summary.steps} days"
        )
        print(
            f"  min={filtered_cov_summary.min:.3f} "
            f"mean={filtered_cov_summary.mean:.3f} "
            f"median={filtered_cov_summary.median:.3f} "
            f"max={filtered_cov_summary.max:.3f}"
        )
    print()
    print(f"Windows by split (policy={split_policy_name(split_policy)}):")
    print(f"  train={tr_steps}  val={va_steps}  test={te_steps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prepared dataset caches.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory containing cache_ratio_index/.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.0,
        help="Minimum per-day coverage required (0.0 disables filtering).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio used when computing step counts.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio used when computing step counts.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test ratio used when computing step counts.",
    )
    parser.add_argument(
        "--per-asset",
        action="store_true",
        help="Split chronologically within each asset (matches loader default).",
    )
    parser.add_argument(
        "--global-order",
        dest="per_asset",
        action="store_false",
        help="Split by global chronological order instead of per-asset.",
    )
    parser.add_argument(
        "--split-policy",
        default="global_purged_horizon",
        choices=("global_purged_horizon", "per_asset_purged_horizon", "contiguous"),
        help="Split policy used when computing step counts.",
    )
    parser.set_defaults(per_asset=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize_dataset(
        data_dir=args.data_dir,
        coverage_threshold=args.coverage,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        per_asset=args.per_asset,
        split_policy=args.split_policy,
    )
