"""Resolve preset dataset caches from an optional zip archive."""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Sequence


DATASET_ZIP_ENV = "LLAPDIFF_DATASET_ZIP"
DATASET_EXTRACT_ENV = "LLAPDIFF_DATASET_EXTRACT_DIR"
DEFAULT_ARCHIVE_NAME = "LLapDiff-evaluation-datasets.zip"


def configure_dataset_archive(
    archive_path: object | None = None,
    extract_dir: object | None = None,
) -> None:
    """Configure archive resolution for the current process."""

    if archive_path not in (None, ""):
        os.environ[DATASET_ZIP_ENV] = str(Path(str(archive_path)).expanduser().resolve())
    if extract_dir not in (None, ""):
        os.environ[DATASET_EXTRACT_ENV] = str(Path(str(extract_dir)).expanduser().resolve())


def find_dataset_archive(package_root: Path) -> Path | None:
    """Return the configured or bundled dataset archive without extracting it."""

    configured = os.environ.get(DATASET_ZIP_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{DATASET_ZIP_ENV} points to a missing file: {path}")
        return path

    bundled = (package_root / "datasets" / DEFAULT_ARCHIVE_NAME).resolve()
    if bundled.exists():
        return bundled

    return None


def resolve_dataset_dir(expected_dir: Path, *, package_root: Path) -> Path:
    """
    Return a usable dataset cache directory.

    If the expected package data directory is absent and a dataset archive is
    available, the archive is safely extracted to a user cache directory and the
    matching cache path is returned.
    """

    expected_dir = expected_dir.resolve()
    if expected_dir.exists():
        return expected_dir

    archive_path = find_dataset_archive(package_root)
    if archive_path is None:
        raise FileNotFoundError(
            f"Dataset cache directory is missing: {expected_dir}. "
            f"Provide a dataset cache zip with --dataset-zip or set {DATASET_ZIP_ENV}."
        )

    extract_root = _extract_root()
    prefixes = tuple(_candidate_prefixes(expected_dir, package_root=package_root))
    _extract_archive_once(archive_path, extract_root, prefixes=prefixes)

    for candidate in _candidate_dirs(expected_dir, package_root=package_root, extract_root=extract_root):
        if candidate.exists():
            return candidate.resolve()

    _extract_archive_once(archive_path, extract_root, prefixes=prefixes, force=True)
    for candidate in _candidate_dirs(expected_dir, package_root=package_root, extract_root=extract_root):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Dataset archive {archive_path} did not contain the expected cache directory for {expected_dir}."
    )


def _extract_root() -> Path:
    configured = os.environ.get(DATASET_EXTRACT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return (cache_home / "llapdiffusion" / "datasets").resolve()


def _candidate_dirs(expected_dir: Path, *, package_root: Path, extract_root: Path) -> Iterable[Path]:
    yield expected_dir

    relative = _relative_dataset_path(expected_dir, package_root=package_root)

    yield extract_root / relative


def _candidate_prefixes(expected_dir: Path, *, package_root: Path) -> Iterable[str]:
    relative = _relative_dataset_path(expected_dir, package_root=package_root).as_posix().strip("/")
    if relative:
        yield f"{relative}/"


def _relative_dataset_path(expected_dir: Path, *, package_root: Path) -> Path:
    dataset_root = (package_root / "datasets").resolve()
    try:
        return expected_dir.relative_to(dataset_root)
    except ValueError:
        return Path(expected_dir.name)


def _archive_stamp_path(
    archive_path: Path,
    extract_root: Path,
    *,
    prefixes: Sequence[str] | None,
) -> Path:
    stat = archive_path.stat()
    prefix_payload = ",".join(sorted(prefixes or ("*",)))
    payload = f"{archive_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{prefix_payload}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return extract_root / f".llapdiff_dataset_archive_{digest}.stamp"


def _extract_archive_once(
    archive_path: Path,
    extract_root: Path,
    *,
    prefixes: Sequence[str] | None = None,
    force: bool = False,
) -> None:
    stamp_path = _archive_stamp_path(archive_path, extract_root, prefixes=prefixes)
    if stamp_path.exists() and not force:
        return

    with zipfile.ZipFile(archive_path) as archive:
        extract_zip_safely(archive, extract_root, prefixes=prefixes)

    stamp_path.write_text(str(archive_path.resolve()))


def extract_zip_safely(
    archive: zipfile.ZipFile,
    extract_root: Path,
    *,
    prefixes: Sequence[str] | None = None,
) -> None:
    """Extract a ZIP archive while rejecting paths outside ``extract_root``."""

    extract_root.mkdir(parents=True, exist_ok=True)
    for member in archive.infolist():
        if prefixes is not None and not _matches_prefix(member.filename, prefixes):
            continue
        destination = _safe_destination(extract_root, member.filename)
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


def _matches_prefix(member_name: str, prefixes: Sequence[str]) -> bool:
    normalized = member_name.replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _safe_destination(extract_root: Path, member_name: str) -> Path:
    raw_name = member_name.replace("\\", "/")
    parts = [part for part in raw_name.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts) or ":" in parts[0]:
        raise ValueError(f"Unsafe path in dataset archive: {member_name!r}")

    root = extract_root.resolve()
    destination = root.joinpath(*parts).resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"Unsafe path in dataset archive: {member_name!r}")
    return destination
