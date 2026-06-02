from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llapdiffusion.configs import config as base_config


PREDICT_TYPES = ("v", "x0", "eps")
DEFAULT_PREDICT_TYPE = "v"


def normalize_predict_type(value: object) -> str:
    """Return the canonical diffusion prediction parameterization name."""
    raw = (
        DEFAULT_PREDICT_TYPE
        if value is None or (isinstance(value, str) and not value.strip())
        else value
    )
    name = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "epsilon": "eps",
        "noise": "eps",
        "x_0": "x0",
        "xstart": "x0",
        "x_start": "x0",
        "velocity": "v",
        "v_prediction": "v",
    }
    normalized = aliases.get(name, name)
    if normalized not in PREDICT_TYPES:
        choices = ", ".join(PREDICT_TYPES)
        raise ValueError(f"Unknown predict_type {value!r}; expected one of: {choices}.")
    return normalized


def clone_config(source: object = base_config) -> SimpleNamespace:
    data = {}
    for name in dir(source):
        if name.startswith("_"):
            continue
        value = getattr(source, name)
        if callable(value):
            continue
        data[name] = value
    return SimpleNamespace(**data)


def make_jsonable(obj: Any):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
