"""Small console-output helpers shared by CLIs and trainers."""

from __future__ import annotations


def is_debug(config_obj: object) -> bool:
    return bool(getattr(config_obj, "DEBUG", False))


def is_verbose(config_obj: object) -> bool:
    return bool(getattr(config_obj, "VERBOSE", False) or is_debug(config_obj))


def apply_verbosity(config_obj: object, *, verbose: bool = False, debug: bool = False) -> None:
    setattr(config_obj, "VERBOSE", bool(verbose or debug))
    setattr(config_obj, "DEBUG", bool(debug))
