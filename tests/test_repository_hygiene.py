"""Repository-level safeguards that do not require the ML runtime."""

from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".gitignore", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
EXCLUDED_DIRECTORIES = {".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
LOCAL_INFORMATION_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)" + r"c:" + r"\\users\\[a-z0-9_.-]+"),
    re.compile(r"(?i)" + r"/" + r"home/[a-z0-9_.-]+"),
    re.compile(r"(?i)" + r"/" + r"users/[a-z0-9_.-]+"),
)


def _public_text_files() -> list[Path]:
    files = []
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE"}:
            files.append(path)
    return sorted(files)


def test_public_text_has_no_machine_paths_or_credentials():
    for path in _public_text_files():
        content = path.read_text(encoding="utf-8")
        for pattern in LOCAL_INFORMATION_PATTERNS:
            assert pattern.search(content) is None, (
                f"{path.relative_to(REPOSITORY_ROOT)} contains a local path or credential-like value."
            )
