from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llapdiffusion.configs import config as base_config


PREDICT_TYPES = ("v", "x0", "eps")
PREDICT_TYPE_VELOCITY = "v"


def resolve_predict_type(value: object) -> str:
    """Validate and return a supported diffusion prediction parameterization."""

    raw = (
        PREDICT_TYPE_VELOCITY
        if value is None or (isinstance(value, str) and not value.strip())
        else value
    )
    predict_type = str(raw).strip().lower()
    if predict_type not in PREDICT_TYPES:
        choices = ", ".join(PREDICT_TYPES)
        raise ValueError(f"Unsupported predict_type {value!r}; expected one of: {choices}.")
    return predict_type


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
