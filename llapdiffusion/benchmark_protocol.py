from __future__ import annotations

from typing import Any


ALL_FEATURE_POLICY_BASELINES = frozenset({"dlinear", "patchtst"})
MULTI_SERIES_BASELINES = frozenset({"mr-diff", "t_patchgnn", "csdi"})

BASELINE_TIME_FEATURE_PROTOCOL = (
    "context features use delta_t only; forecast features use delta_t_y as known "
    "query-grid metadata, not target values"
)


def baseline_protocol_metadata(
    baseline: str,
    *,
    requested_input_policy: str = "target_only",
) -> dict[str, Any]:
    """Return explicit benchmark-scope metadata for a baseline result row."""

    key = str(baseline).strip().lower()
    requested = str(requested_input_policy or "target_only").strip().lower()
    supports_all_features = key in ALL_FEATURE_POLICY_BASELINES
    effective_input_policy = "all_features" if requested == "all_features" and supports_all_features else "target_only"

    if key == "csdi":
        return {
            "comparison_type": "imputation",
            "input_scope": "target_only_with_retained_target_horizon_tokens",
            "missingness_scope": "target_context_mask_and_target_horizon_holdout_mask",
            "modeling_scope": "multi-series",
            "input_policy_effective": "target_only",
            "all_features_policy_supported": False,
            "time_feature_protocol": (
                "imputation timepoints combine context delta_t and delta_t_y query-grid metadata; "
                "scoring is target-horizon imputation"
            ),
        }

    if effective_input_policy == "all_features":
        input_scope = "all_features"
        missingness_scope = "target_mask_and_context_observation_fraction"
    else:
        input_scope = "target_only"
        missingness_scope = "target_mask"

    return {
        "comparison_type": "extrapolation",
        "input_scope": input_scope,
        "missingness_scope": missingness_scope,
        "modeling_scope": "multi-series" if key in MULTI_SERIES_BASELINES else "uni-average/shared-weight",
        "input_policy_effective": effective_input_policy,
        "all_features_policy_supported": supports_all_features,
        "time_feature_protocol": BASELINE_TIME_FEATURE_PROTOCOL,
    }


def llapdiff_protocol_metadata() -> dict[str, str]:
    """Return benchmark-scope metadata for LLapDiff result summaries."""

    return {
        "comparison_type": "extrapolation",
        "input_scope": "all_features",
        "missingness_scope": "per_feature_covariate_mask",
        "modeling_scope": "joint global",
        "time_feature_protocol": (
            "joint cross-entity context with globally shared query grids; delta_t_y is "
            "known query-grid metadata, not target values"
        ),
    }


def split_protocol_metadata(
    dataset_key: str,
    *,
    split_policy: str,
    split_scope: str,
) -> dict[str, str]:
    """Return concise split notes for result metadata."""

    dataset = str(dataset_key or "").strip().lower()
    policy = str(split_policy or "").strip().lower()
    scope = str(split_scope or "").strip().lower()
    if dataset == "physionet" or scope == "physionet_patient_relative_time":
        return {
            "split_note": "patient_relative_contiguous_split",
            "split_caveat": "special_case_insufficient_horizon_windows_for_purged_split",
        }
    if policy == "global_purged_horizon":
        return {
            "split_note": "global_purged_horizon_split",
            "split_caveat": "none",
        }
    return {
        "split_note": policy or "unspecified_split",
        "split_caveat": "none",
    }
