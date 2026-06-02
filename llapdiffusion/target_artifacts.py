"""Target-selection metadata helpers for artifact naming and validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _clean_list(value: object) -> list[str]:
    if value is None or (isinstance(value, str) and value == ""):
        return []
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, Sequence):
        parts = [str(part) for part in value]
    else:
        parts = [str(value)]
    return [part.strip() for part in parts if part and part.strip()]


def _int_list(value: object) -> list[int]:
    if value is None or (isinstance(value, str) and value == ""):
        return []
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [int(v) for v in value]
    return [int(value)]


def _slug(text: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    return slug or "target"


def normalize_target_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    payload = dict(metadata or {})
    cols = _clean_list(payload.get("target_cols"))
    requested_cols = _clean_list(payload.get("requested_target_cols"))
    target_col = str(payload.get("target_col") or (cols[0] if cols else "")).strip()
    requested_target_col = str(payload.get("requested_target_col") or "").strip() or None
    target_dim = int(payload.get("target_dim") or max(1, len(cols)))
    source = str(payload.get("target_source") or "").strip()
    indices = _int_list(payload.get("target_indices"))
    if not cols and target_col:
        cols = [target_col]
    legacy = bool(
        target_dim == 1
        and source in {"", "cache_target", "unresolved"}
    )
    canonical = {
        "target_col": target_col,
        "target_cols": cols,
        "target_indices": indices,
        "target_dim": target_dim,
        "target_source": source,
        "requested_target_col": requested_target_col,
        "requested_target_cols": requested_cols,
        "legacy_scalar_default": legacy,
    }
    canonical["target_signature"] = target_signature(canonical)
    return canonical


def target_signature(metadata: Mapping[str, object]) -> str:
    target_dim = int(metadata.get("target_dim") or 1)
    cols = _clean_list(metadata.get("target_cols")) or ["target"]
    label = "-".join(_slug(col) for col in cols)[:64].strip("-") or "target"
    payload = {
        "target_cols": cols,
        "target_indices": _int_list(metadata.get("target_indices")),
        "target_dim": target_dim,
        "target_source": str(metadata.get("target_source") or ""),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:8]
    return f"tdim-{target_dim}_targets-{label}-{digest}"


def target_metadata_from_policy(policy: Mapping[str, object]) -> dict[str, object]:
    return normalize_target_metadata(policy)


def target_metadata_from_config(config_obj: object) -> dict[str, object]:
    configured = getattr(config_obj, "TARGET_METADATA", None)
    if isinstance(configured, Mapping):
        return normalize_target_metadata(configured)
    return normalize_target_metadata(
        {
            "target_col": getattr(config_obj, "TARGET_COL", None),
            "target_cols": getattr(config_obj, "TARGET_COLS", None),
            "target_indices": getattr(config_obj, "TARGET_INDICES", None),
            "target_dim": getattr(config_obj, "TARGET_DIM", 1),
            "target_source": getattr(config_obj, "TARGET_SOURCE", ""),
            "requested_target_col": getattr(config_obj, "REQUESTED_TARGET_COL", None),
            "requested_target_cols": getattr(config_obj, "REQUESTED_TARGET_COLS", None),
        }
    )


def target_artifact_suffix(metadata: Mapping[str, object]) -> str:
    normalized = normalize_target_metadata(metadata)
    if bool(normalized.get("legacy_scalar_default")):
        return ""
    return "_" + str(normalized["target_signature"])


def apply_target_metadata_to_config(config_obj: object, policy: Mapping[str, object]) -> dict[str, object]:
    metadata = target_metadata_from_policy(policy)
    suffix = target_artifact_suffix(metadata)
    setattr(config_obj, "TARGET_METADATA", metadata)
    setattr(config_obj, "TARGET_COL", metadata["target_col"])
    setattr(config_obj, "TARGET_COLS", list(metadata["target_cols"]))
    setattr(config_obj, "TARGET_INDICES", list(metadata["target_indices"]))
    setattr(config_obj, "TARGET_DIM", int(metadata["target_dim"]))
    setattr(config_obj, "TARGET_SOURCE", metadata["target_source"])
    setattr(config_obj, "REQUESTED_TARGET_COL", metadata["requested_target_col"])
    setattr(config_obj, "REQUESTED_TARGET_COLS", list(metadata["requested_target_cols"]))
    setattr(config_obj, "TARGET_SIGNATURE", metadata["target_signature"])
    setattr(config_obj, "TARGET_ARTIFACT_SUFFIX", suffix)
    return metadata


def sync_target_artifact_config(
    config_obj: object,
    policy: Mapping[str, object] | None = None,
    *,
    update_output_dirs: bool = True,
) -> dict[str, object]:
    metadata = (
        apply_target_metadata_to_config(config_obj, policy)
        if policy is not None
        else apply_target_metadata_to_config(config_obj, target_metadata_from_config(config_obj))
    )
    target_dim = int(metadata.get("target_dim") or 1)
    suffix = str(getattr(config_obj, "TARGET_ARTIFACT_SUFFIX", "") or "")
    setattr(config_obj, "VAE_OUTPUT_DIM", target_dim)
    setattr(config_obj, "VAE_INPUT_DIM", 2 * target_dim)
    if not suffix:
        return metadata

    required = ("VAE_DIR", "PRED", "VAE_LATENT_CHANNELS")
    missing = [name for name in required if not hasattr(config_obj, name)]
    if missing:
        raise AttributeError(
            "Cannot build target-specific VAE checkpoint path; "
            f"missing config attributes: {', '.join(missing)}"
        )

    entity_suffix = "_entity" if bool(getattr(config_obj, "VAE_ENTITY_CONDITION", False)) else ""
    setattr(
        config_obj,
        "VAE_CKPT",
        str(
            Path(str(getattr(config_obj, "VAE_DIR")))
            / f"pred-{getattr(config_obj, 'PRED')}_ch-{getattr(config_obj, 'VAE_LATENT_CHANNELS')}"
            f"{entity_suffix}{suffix}_elbo.pt"
        ),
    )
    if update_output_dirs:
        if hasattr(config_obj, "OUT_DIR"):
            out_dir = Path(str(getattr(config_obj, "OUT_DIR")))
            if not out_dir.name.endswith(suffix):
                setattr(config_obj, "OUT_DIR", str(out_dir.with_name(out_dir.name + suffix)))
        if hasattr(config_obj, "CKPT_DIR"):
            ckpt_dir = Path(str(getattr(config_obj, "CKPT_DIR")))
            if not ckpt_dir.name.endswith(suffix):
                setattr(config_obj, "CKPT_DIR", str(ckpt_dir.with_name(ckpt_dir.name + suffix)))
        if hasattr(config_obj, "POLE_PLOT_DIR") and hasattr(config_obj, "OUT_DIR"):
            setattr(config_obj, "POLE_PLOT_DIR", str(Path(str(getattr(config_obj, "OUT_DIR"))) / "pole_plots"))
    return metadata


def loader_target_request_from_config(config_obj: object) -> tuple[str | None, list[str] | None]:
    requested_col = getattr(config_obj, "REQUESTED_TARGET_COL_ARG", None)
    if requested_col in (None, ""):
        requested_col = getattr(config_obj, "REQUESTED_TARGET_COL", None)

    cols = _clean_list(getattr(config_obj, "REQUESTED_TARGET_COLS_ARG", None))
    if not cols:
        cols = _clean_list(getattr(config_obj, "REQUESTED_TARGET_COLS", None))

    if not cols and requested_col in (None, "") and not hasattr(config_obj, "TARGET_METADATA"):
        cols = _clean_list(getattr(config_obj, "TARGET_COLS", None))
        requested_col = None if cols else getattr(config_obj, "TARGET_COL", None)

    return (None if cols else requested_col, cols or None)


def checkpoint_target_metadata(payload: Any) -> dict[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    if isinstance(payload.get("target_metadata"), Mapping):
        return normalize_target_metadata(payload["target_metadata"])
    if "target_dim" in payload or "target_cols" in payload:
        return normalize_target_metadata(
            {
                "target_col": payload.get("target_col"),
                "target_cols": payload.get("target_cols"),
                "target_indices": payload.get("target_indices"),
                "target_dim": payload.get("target_dim"),
                "target_source": payload.get("target_source"),
                "requested_target_col": payload.get("requested_target_col"),
                "requested_target_cols": payload.get("requested_target_cols"),
            }
        )
    return None


def unwrap_checkpoint_model(payload: Any) -> Any:
    if isinstance(payload, Mapping) and "model" in payload:
        return payload["model"]
    return payload


def validate_checkpoint_target_metadata(
    payload: Any,
    config_obj: object,
    *,
    context: str,
) -> None:
    expected = target_metadata_from_config(config_obj)
    actual = checkpoint_target_metadata(payload)
    if actual is None:
        if bool(expected.get("legacy_scalar_default")):
            return
        raise ValueError(
            f"{context} checkpoint does not contain target metadata; "
            f"expected {expected['target_signature']} for target_cols={expected['target_cols']}."
        )
    if (
        bool(actual.get("legacy_scalar_default"))
        and bool(expected.get("legacy_scalar_default"))
        and not actual.get("target_cols")
    ):
        return

    mismatches = []
    for key in ("target_dim", "target_cols", "target_signature"):
        if actual.get(key) != expected.get(key):
            mismatches.append(f"{key}: checkpoint={actual.get(key)!r} expected={expected.get(key)!r}")
    if mismatches:
        raise ValueError(f"{context} checkpoint target metadata mismatch: " + "; ".join(mismatches))
