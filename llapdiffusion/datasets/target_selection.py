"""Helpers for resolving forecast targets from dataset metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


_CALENDAR_PREFIXES = ("DOW_", "DOM_", "MOY_")


@dataclass(frozen=True)
class TargetSelection:
    target_cols: tuple[str, ...]
    target_indices: tuple[int, ...]
    target_dim: int
    target_col: str
    target_index: int
    target_source: str
    requested_target_cols: tuple[str, ...] | None
    requested_target_col: str | None
    calendar_feature_cols: tuple[str, ...] = ()


def infer_calendar_feature_cols(meta: Mapping[str, object], feature_cols: Sequence[str]) -> tuple[str, ...]:
    configured = meta.get("calendar_feature_cols")
    if isinstance(configured, str):
        configured_cols = [configured]
    elif isinstance(configured, Sequence):
        configured_cols = [str(col) for col in configured]
    else:
        configured_cols = []
    inferred = [
        str(col)
        for col in feature_cols
        if str(col).upper().startswith(_CALENDAR_PREFIXES)
    ]
    return tuple(dict.fromkeys([*configured_cols, *inferred]))


def finance_calendar_feature_cols_from_names(feature_cols: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        str(col)
        for col in feature_cols
        if str(col).upper().startswith(_CALENDAR_PREFIXES)
    )


def valid_target_cols(meta: Mapping[str, object]) -> tuple[str, ...]:
    feature_cols = [str(col) for col in meta.get("feature_cols", [])]
    calendar_cols = {col.upper() for col in infer_calendar_feature_cols(meta, feature_cols)}
    return tuple(col for col in feature_cols if col.upper() not in calendar_cols)


def valid_scalar_target_cols(meta: Mapping[str, object]) -> tuple[str, ...]:
    """Backward-compatible alias for callers that only need the valid column list."""

    return valid_target_cols(meta)


def _coerce_target_cols(value: object) -> tuple[str, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence):
        raw = [str(part).strip() for part in value]
    else:
        raw = [str(value).strip()]
    cols = tuple(part for part in raw if part)
    return cols or None


def _default_target_cols(
    meta: Mapping[str, object],
    default_target_col: str | None,
    default_target_cols: Sequence[str] | str | None = None,
) -> tuple[str, ...] | None:
    target_cols = _coerce_target_cols(meta.get("target_cols"))
    if target_cols is not None:
        return target_cols
    target_cols = _coerce_target_cols(default_target_cols)
    if target_cols is not None:
        return target_cols
    target_col = str(meta.get("target_col") or meta.get("target_column") or default_target_col or "").strip()
    return (target_col,) if target_col else None


def resolve_target_selection(
    meta: Mapping[str, object],
    requested_target_col: str | None = None,
    *,
    requested_target_cols: Sequence[str] | str | None = None,
    default_target_col: str | None = None,
    default_target_cols: Sequence[str] | str | None = None,
) -> TargetSelection:
    feature_cols = [str(col) for col in meta.get("feature_cols", [])]
    if not feature_cols:
        raise ValueError("Dataset metadata must include non-empty feature_cols for target selection.")

    scalar_requested = _coerce_target_cols(requested_target_col)
    multi_requested = _coerce_target_cols(requested_target_cols)
    if scalar_requested is not None and multi_requested is not None:
        raise ValueError("Use either target_col or target_cols, not both.")
    if scalar_requested is not None and len(scalar_requested) != 1:
        raise ValueError("target_col accepts exactly one column; use target_cols for multi-target forecasting.")

    requested = multi_requested if multi_requested is not None else scalar_requested
    default_cols = _default_target_cols(meta, default_target_col, default_target_cols)
    target_cols = requested or default_cols
    if not target_cols:
        raise ValueError("No target columns were provided and metadata does not define target_col/target_cols.")
    if len(set(target_cols)) != len(target_cols):
        raise ValueError(f"Duplicate target columns are not allowed: {', '.join(target_cols)}")

    valid_targets = valid_target_cols(meta)
    missing = [col for col in target_cols if col not in feature_cols]
    if missing:
        raise ValueError(
            f"target columns {missing!r} are not present in feature_cols. "
            f"Valid target columns are: {', '.join(valid_targets)}"
        )

    calendar_cols = infer_calendar_feature_cols(meta, feature_cols)
    calendar_lookup = {col.upper() for col in calendar_cols}
    calendar_targets = [col for col in target_cols if col.upper() in calendar_lookup]
    if calendar_targets:
        raise ValueError(
            f"Target columns {calendar_targets!r} are calendar features and cannot be used as targets. "
            f"Valid target columns are: {', '.join(valid_targets)}"
        )

    if default_cols and len(target_cols) == 1 and target_cols == default_cols:
        source = "cache_target"
    else:
        source = "feature_column" if len(target_cols) == 1 else "feature_columns"

    indices = tuple(feature_cols.index(col) for col in target_cols)
    first_col = target_cols[0]
    return TargetSelection(
        target_cols=tuple(target_cols),
        target_indices=indices,
        target_dim=len(target_cols),
        target_col=first_col,
        target_index=indices[0],
        target_source=source,
        requested_target_cols=requested,
        requested_target_col=requested[0] if requested else None,
        calendar_feature_cols=calendar_cols,
    )
