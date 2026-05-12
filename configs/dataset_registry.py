"""Resolve dataset-specific dataloader entrypoints from a cache directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict


RunExperiment = Callable[..., object]


def _import_fin_run_experiment() -> RunExperiment:
    try:  # pragma: no cover
        from Dataset.fin_dataset import run_experiment
    except Exception:  # pragma: no cover
        from fin_dataset import run_experiment
    return run_experiment


def _import_bms_run_experiment() -> RunExperiment:
    try:  # pragma: no cover
        from Dataset.bms_air_dataset import run_experiment
    except Exception:  # pragma: no cover
        from bms_air_dataset import run_experiment
    return run_experiment


def _import_noaa_run_experiment() -> RunExperiment:
    try:  # pragma: no cover
        from Dataset.noaa_isd_dataset import run_experiment
    except Exception:  # pragma: no cover
        from noaa_isd_dataset import run_experiment
    return run_experiment


def _import_uci_run_experiment() -> RunExperiment:
    try:  # pragma: no cover
        from Dataset.uci_air_quality_dataset import run_experiment
    except Exception:  # pragma: no cover
        from uci_air_quality_dataset import run_experiment
    return run_experiment


def _import_physionet_run_experiment() -> RunExperiment:
    try:  # pragma: no cover
        from Dataset.physionet_cinc_dataset import run_experiment
    except Exception:  # pragma: no cover
        from physionet_cinc_dataset import run_experiment
    return run_experiment


def _import_synthetic_run_experiment() -> RunExperiment:
    try:  # pragma: no cover
        from Dataset.synthetic_regime_dataset import run_experiment
    except Exception:  # pragma: no cover
        from synthetic_regime_dataset import run_experiment
    return run_experiment


_IMPORTERS: Dict[str, Callable[[], RunExperiment]] = {
    "bms_air_quality": _import_bms_run_experiment,
    "bms_air_dataset": _import_bms_run_experiment,
    "nnoa_isd": _import_noaa_run_experiment,
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
    except Exception:
        return ""
    return str(payload.get("dataset", "")).strip().lower()


def resolve_run_experiment(data_dir: object) -> RunExperiment:
    dataset_name = dataset_name_from_data_dir(data_dir)
    importer = _IMPORTERS.get(dataset_name)
    if importer is None:
        return _import_fin_run_experiment()
    return importer()
