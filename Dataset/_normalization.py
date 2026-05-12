"""Utilities for accumulating normalization statistics across datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class _PerAssetState:
    """Running sufficient statistics for one asset."""

    count_x: np.ndarray
    sum_x: np.ndarray
    sumsq_x: np.ndarray
    count_y: int = 0
    sum_y: float = 0.0
    sumsq_y: float = 0.0


class NormalizationStatsAccumulator:
    """
    Accumulate normalization statistics for features/targets.

    Supports:
      - per-asset statistics (one set per asset)
      - global statistics (single shared set)

    Output is JSON-friendly and matches the expected ``norm_stats.json`` schema.
    """

    def __init__(self, num_assets: int, feature_dim: int, *, per_asset: bool) -> None:
        if num_assets <= 0:
            raise ValueError("num_assets must be a positive integer")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")

        self._per_asset = bool(per_asset)
        self._feature_dim = int(feature_dim)

        if self._per_asset:
            self._state: List[_PerAssetState] = [
                _PerAssetState(
                    count_x=np.zeros(self._feature_dim, dtype=np.int64),
                    sum_x=np.zeros(self._feature_dim, dtype=np.float64),
                    sumsq_x=np.zeros(self._feature_dim, dtype=np.float64),
                )
                for _ in range(int(num_assets))
            ]
        else:
            self._count_x = np.zeros(self._feature_dim, dtype=np.int64)
            self._sum_x = np.zeros(self._feature_dim, dtype=np.float64)
            self._sumsq_x = np.zeros(self._feature_dim, dtype=np.float64)

            self._count_y = 0
            self._sum_y = 0.0
            self._sumsq_y = 0.0

    # ------------------------------------------------------------------
    # Update helpers

    def _validate_inputs(self, features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = np.asarray(features, dtype=np.float32)
        t = np.asarray(targets, dtype=np.float32)

        if f.ndim != 2 or f.shape[1] != self._feature_dim:
            raise ValueError(
                "features must be a 2D array with shape (N, feature_dim); "
                f"received {f.shape}"
            )

        if t.ndim != 1:
            t = t.reshape(-1)

        if t.shape[0] != f.shape[0]:
            raise ValueError("targets must align with features along axis 0")

        return f, t

    @staticmethod
    def _finite_batch_stats(features: np.ndarray, targets: np.ndarray):
        """Return finite-aware sufficient statistics for one mini-batch."""
        f64 = features.astype(np.float64, copy=False)
        t64 = targets.astype(np.float64, copy=False)

        finite_x = np.isfinite(f64)
        count_x = finite_x.sum(axis=0, dtype=np.int64)
        safe_x = np.where(finite_x, f64, 0.0)
        sum_x = safe_x.sum(axis=0, dtype=np.float64)
        sumsq_x = np.square(safe_x).sum(axis=0, dtype=np.float64)

        finite_y = np.isfinite(t64)
        count_y = int(finite_y.sum())
        safe_y = np.where(finite_y, t64, 0.0)
        sum_y = float(safe_y.sum())
        sumsq_y = float(np.square(safe_y).sum())

        return count_x, sum_x, sumsq_x, count_y, sum_y, sumsq_y

    def update(self, asset_id: int, features: np.ndarray, targets: np.ndarray) -> None:
        """Update statistics with a new mini-batch for ``asset_id``."""
        features, targets = self._validate_inputs(features, targets)
        count_x, sum_x, sumsq_x, count_y, sum_y, sumsq_y = self._finite_batch_stats(features, targets)

        if self._per_asset:
            if not (0 <= asset_id < len(self._state)):
                raise IndexError("asset_id is out of range for the accumulator state")

            state = self._state[asset_id]
            state.count_x += count_x
            state.sum_x += sum_x
            state.sumsq_x += sumsq_x
            state.count_y += count_y
            state.sum_y += sum_y
            state.sumsq_y += sumsq_y
            return

        # Global statistics mode
        self._count_x += count_x
        self._sum_x += sum_x
        self._sumsq_x += sumsq_x
        self._count_y += count_y
        self._sum_y += sum_y
        self._sumsq_y += sumsq_y

    # ------------------------------------------------------------------
    # Finalization helpers

    @staticmethod
    def _finalize_feature_stats(count_x: np.ndarray, sum_x: np.ndarray, sumsq_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        denom_x = np.maximum(count_x.astype(np.float64), 1.0)
        mean_x = sum_x / denom_x
        var_x = (sumsq_x / denom_x) - np.square(mean_x)
        var_x = np.maximum(var_x, 1e-12)
        std_x = np.sqrt(var_x)
        std_x = np.where((std_x == 0.0) | ~np.isfinite(std_x), 1.0, std_x)

        return mean_x.astype(np.float32), std_x.astype(np.float32)

    @staticmethod
    def _finalize_target_stats(count_y: int, sum_y: float, sumsq_y: float) -> tuple[float, float]:
        if count_y <= 0:
            return 0.0, 1.0

        denom_y = max(int(count_y), 1)
        mean_y = float(sum_y / denom_y)
        var_y = max(float(sumsq_y / denom_y) - (mean_y ** 2), 1e-12)
        std_y = float(np.sqrt(var_y))
        if std_y == 0.0 or not np.isfinite(std_y):
            std_y = 1.0
        return mean_y, std_y

    def finalize(self, assets: Sequence[str]) -> dict:
        """Return the JSON-serialisable statistics dictionary."""
        assets_list = list(assets)
        if len(assets_list) == 0:
            raise ValueError("assets must contain at least one entry")

        if self._per_asset:
            if len(assets_list) != len(self._state):
                raise ValueError("Number of assets does not match the accumulator state")

            # Keep strict behavior: all assets should have been seen.
            missing_assets = [
                i for i, s in enumerate(self._state)
                if int(s.count_x.sum()) == 0 and s.count_y == 0
            ]
            if missing_assets:
                raise RuntimeError(
                    "Normalization statistics missing for some assets: "
                    + ", ".join(map(str, missing_assets))
                )

            mean_x_list = []
            std_x_list = []
            mean_y_list = []
            std_y_list = []

            for s in self._state:
                mean_x, std_x = self._finalize_feature_stats(s.count_x, s.sum_x, s.sumsq_x)
                mean_y, std_y = self._finalize_target_stats(s.count_y, s.sum_y, s.sumsq_y)

                mean_x_list.append(mean_x.reshape(1, 1, -1).tolist())
                std_x_list.append(std_x.reshape(1, 1, -1).tolist())
                mean_y_list.append(mean_y)
                std_y_list.append(std_y)

            return {
                "per_ticker": True,
                "assets": assets_list,
                "mean_x": mean_x_list,
                "std_x": std_x_list,
                "mean_y": mean_y_list,
                "std_y": std_y_list,
            }

        if int(self._count_x.sum()) == 0 and self._count_y == 0:
            raise RuntimeError("Unable to compute normalization statistics (no samples).")

        mean_x, std_x = self._finalize_feature_stats(self._count_x, self._sum_x, self._sumsq_x)
        mean_y, std_y = self._finalize_target_stats(self._count_y, self._sum_y, self._sumsq_y)

        return {
            "per_ticker": False,
            "assets": assets_list,
            "mean_x": mean_x.reshape(1, 1, -1).tolist(),
            "std_x": std_x.reshape(1, 1, -1).tolist(),
            "mean_y": mean_y,
            "std_y": std_y,
        }


__all__ = ["NormalizationStatsAccumulator"]
