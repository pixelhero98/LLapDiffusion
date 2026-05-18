from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterable

from llapdiffusion.baselines.registry import BaselineSpec


def resolve_source_root(source_root: str | os.PathLike[str] | None) -> Path:
    raw = source_root or os.environ.get("LLAPDIFF_BASELINE_SOURCE_ROOT")
    if not raw:
        raise ValueError("Provide --baseline-source-root or LLAPDIFF_BASELINE_SOURCE_ROOT.")
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Baseline source root does not exist: {root}")
    return root


def git_sha(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def git_status_clean(path: Path) -> bool:
    status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
    return not status.strip()


def _matches_module(name: str, prefixes: tuple[str, ...]) -> bool:
    return name in prefixes or name.startswith(tuple(prefix + "." for prefix in prefixes))


@contextmanager
def prepend_paths(*paths: Path, module_prefixes: Iterable[str] = ()):
    old_path = list(sys.path)
    prefixes = tuple(module_prefixes)
    old_modules = {name: module for name, module in sys.modules.items() if _matches_module(name, prefixes)}
    if prefixes:
        for name in list(old_modules):
            del sys.modules[name]
    for path in reversed([str(p) for p in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = old_path
        if prefixes:
            for name in list(sys.modules):
                if _matches_module(name, prefixes):
                    del sys.modules[name]
            sys.modules.update(old_modules)


def load_module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceManager:
    def __init__(self, source_root: str | os.PathLike[str] | None):
        self._source_root = source_root
        self.root: Path | None = None

    def path(self, name: str) -> Path:
        if self.root is None:
            self.root = resolve_source_root(self._source_root)
        return self.root / name

    def validate(self, spec: BaselineSpec) -> dict[str, object]:
        if spec.first_party:
            return {
                "source_name": spec.source_name,
                "source_sha": spec.source_sha,
                "source_clean": True,
                "official_reference": spec.official_reference,
                "dependency_caveat": spec.dependency_caveat,
                "dependency_sources": {},
            }
        source_path = self.path(spec.source_name)
        actual = git_sha(source_path)
        clean = git_status_clean(source_path)
        if actual != spec.source_sha:
            raise RuntimeError(f"{spec.key}: source SHA mismatch {actual} != {spec.source_sha}")
        if not clean:
            raise RuntimeError(f"{spec.key}: source checkout has tracked/unexpected changes")

        dependencies: dict[str, dict[str, object]] = {}
        for name, expected_sha in spec.dependency_sources:
            dep_path = self.path(name)
            dep_sha = git_sha(dep_path)
            dep_clean = git_status_clean(dep_path)
            if dep_sha != expected_sha:
                raise RuntimeError(f"{spec.key}: dependency {name} SHA mismatch {dep_sha} != {expected_sha}")
            if not dep_clean:
                raise RuntimeError(f"{spec.key}: dependency {name} checkout has tracked/unexpected changes")
            dependencies[name] = {
                "source_sha": dep_sha,
                "source_clean": dep_clean,
            }

        return {
            "source_name": spec.source_name,
            "source_sha": actual,
            "source_clean": clean,
            "official_reference": spec.official_reference,
            "dependency_caveat": spec.dependency_caveat,
            "dependency_sources": dependencies,
        }

    def load_module(self, module_name: str, path: Path) -> ModuleType:
        return load_module_from_file(module_name, path)

    def prepend(self, *paths: Path, module_prefixes: Iterable[str] = ()):
        return prepend_paths(*paths, module_prefixes=module_prefixes)
