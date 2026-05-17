"""Shared typing helpers for dataset modules."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover - typing helper only
    from os import PathLike as _PathLike

    PathLike = Union[str, _PathLike[str]]
else:  # Runtime path-like union used across dataset helpers
    PathLike = Union[str, os.PathLike]


__all__ = ["PathLike"]

