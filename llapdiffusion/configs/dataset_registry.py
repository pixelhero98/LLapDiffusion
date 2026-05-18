"""Resolve dataset-specific dataloader entrypoints from a cache directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict


RunExperiment = Callable[..., object]


def _import_fin_run_experiment() -> RunExperiment:
    from llapdiffusion.datasets.fin_dataset import run_experiment
    return run_experiment


def _import_bms_run_experiment() -> RunExperiment:
    from llapdiffusion.datasets.bms_air_dataset import run_experiment
    return run_experiment


def _import_noaa_run_experiment() -> RunExperiment:
    from llapdiffusion.datasets.noaa_isd_dataset import run_experiment
    return run_experiment


def _import_uci_run_experiment() -> RunExperiment:
    from llapdiffusion.datasets.uci_air_quality_dataset import run_experiment
    return run_experiment


def _import_physionet_run_experiment() -> RunExperiment:
    from llapdiffusion.datasets.physionet_cinc_dataset import run_experiment
    return run_experiment


def _import_synthetic_run_experiment() -> RunExperiment:
    from llapdiffusion.datasets.synthetic_regime_dataset import run_experiment
    return run_experiment


_IMPORTERS: Dict[str, Callable[[], RunExperiment]] = {
    "bms_air_quality": _import_bms_run_experiment,
    "bms_air_dataset": _import_bms_run_experiment,
    "noaa_isd": _import_noaa_run_experiment,
    "uci_air_quality": _import_uci_run_experiment,
    "physionet_cinc": _import_physionet_run_experiment,
    "synthetic_regime": _import_synthetic_run_experiment,
}


def _meta_path_for_data_dir(data_dir: object) -> Path:
    root = Path(str(data_dir))
    return root / "cache_ratio_index" / "meta.json"


def dataset_name_from_data_dir(data_dir: object) -> str:
    meta_path = _meta_path_for_data_dir(data_dir)
    if not meta_path.exists():
        return ""
    try:
        payload = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid dataset metadata JSON: {meta_path}") from exc
    return str(payload.get("dataset", "")).strip().lower()


def _dataset_name_from_public_path(data_dir: object) -> str:
    parts = {part.lower() for part in Path(str(data_dir)).parts}
    if {"crypto", "us_equity", "fin_dataset"} & parts:
        return "finance"
    if "bms_air" in parts:
        return "bms_air_quality"
    if "uci_air" in parts:
        return "uci_air_quality"
    if "physionet" in parts:
        return "physionet_cinc"
    if "noaa_us" in parts or "noaa_uk" in parts:
        return "noaa_isd"
    if "synthetic_regime" in parts:
        return "synthetic_regime"
    return ""


def resolve_run_experiment(data_dir: object) -> RunExperiment:
    dataset_name = dataset_name_from_data_dir(data_dir) or _dataset_name_from_public_path(data_dir)
    if dataset_name in {"finance", "fin_dataset"}:
        return _import_fin_run_experiment()
    importer = _IMPORTERS.get(dataset_name)
    if importer is None:
        raise ValueError(
            f"Cannot determine dataset loader for {data_dir}. "
            "Ensure cache_ratio_index/meta.json contains a supported dataset name."
        )
    return importer()
