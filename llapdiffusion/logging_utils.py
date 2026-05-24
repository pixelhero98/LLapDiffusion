"""Small console-output helpers shared by CLIs and trainers."""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from typing import Iterable, Iterator, Optional, TextIO, TypeVar


_T = TypeVar("_T")


def is_debug(config_obj: object) -> bool:
    return bool(getattr(config_obj, "DEBUG", False))


def is_verbose(config_obj: object) -> bool:
    return bool(getattr(config_obj, "VERBOSE", False) or is_debug(config_obj))


def apply_verbosity(config_obj: object, *, verbose: bool = False, debug: bool = False) -> None:
    setattr(config_obj, "VERBOSE", bool(verbose or debug))
    setattr(config_obj, "DEBUG", bool(debug))


def _safe_len(iterable: Iterable[object]) -> Optional[int]:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _use_tqdm(stream: TextIO) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _progress_interval(total: Optional[int], log_every: Optional[int]) -> int:
    if log_every is not None:
        return max(1, int(log_every))
    if total is None or total <= 0:
        return 50
    return max(1, total // 20)


def _emit_progress(
    stream: TextIO,
    desc: str,
    count: int,
    total: Optional[int],
    unit: str,
    *,
    done: bool = False,
) -> None:
    if total is None:
        detail = f"{count} {unit}"
    else:
        detail = f"{count}/{total} {unit}"
    suffix = " done" if done else ""
    print(f"[progress] {desc}: {detail}{suffix}", file=stream, flush=True)


def progress_iter(
    iterable: Iterable[_T],
    *,
    desc: str,
    enabled: bool = False,
    total: Optional[int] = None,
    unit: str = "batch",
    log_every: Optional[int] = None,
    stream: Optional[TextIO] = None,
) -> Iterator[_T]:
    """Yield ``iterable`` with verbose-only progress on stderr.

    Interactive terminals get a tqdm bar. Non-interactive logs get sparse plain
    lines so tmux/tee output stays readable and stdout remains machine-parseable.
    """

    if not enabled:
        yield from iterable
        return

    stream = sys.stderr if stream is None else stream
    resolved_total = _safe_len(iterable) if total is None else int(total)
    if _use_tqdm(stream):
        from tqdm.auto import tqdm

        yield from tqdm(
            iterable,
            total=resolved_total,
            desc=desc,
            unit=unit,
            file=stream,
            dynamic_ncols=True,
            leave=False,
        )
        return

    interval = _progress_interval(resolved_total, log_every)
    count = 0
    last_reported = -1
    _emit_progress(stream, desc, 0, resolved_total, unit)
    for item in iterable:
        yield item
        count += 1
        should_report = count == 1 or count % interval == 0
        if resolved_total is not None:
            should_report = should_report or count >= resolved_total
        if should_report:
            _emit_progress(stream, desc, count, resolved_total, unit)
            last_reported = count
    if last_reported != count:
        _emit_progress(stream, desc, count, resolved_total, unit, done=True)


class _ProgressTask(AbstractContextManager["_ProgressTask"]):
    def __init__(
        self,
        *,
        desc: str,
        enabled: bool,
        total: Optional[int],
        unit: str,
        log_every: Optional[int],
        stream: Optional[TextIO],
    ) -> None:
        self.desc = desc
        self.enabled = bool(enabled)
        self.total = None if total is None else int(total)
        self.unit = unit
        self.log_every = log_every
        self.stream = sys.stderr if stream is None else stream
        self.count = 0
        self._last_reported = -1
        self._bar = None

    def __enter__(self) -> "_ProgressTask":
        if not self.enabled:
            return self
        if _use_tqdm(self.stream):
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                file=self.stream,
                dynamic_ncols=True,
                leave=False,
            )
        else:
            _emit_progress(self.stream, self.desc, 0, self.total, self.unit)
        return self

    def update(self, n: int = 1) -> None:
        if not self.enabled:
            return
        step = int(n)
        if step <= 0:
            return
        self.count += step
        if self._bar is not None:
            self._bar.update(step)
            return
        interval = _progress_interval(self.total, self.log_every)
        should_report = self.count == step or self.count % interval == 0
        if self.total is not None:
            should_report = should_report or self.count >= self.total
        if should_report:
            _emit_progress(self.stream, self.desc, self.count, self.total, self.unit)
            self._last_reported = self.count

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self.enabled:
            return False
        if self._bar is not None:
            self._bar.close()
            return False
        if self._last_reported != self.count:
            _emit_progress(self.stream, self.desc, self.count, self.total, self.unit, done=True)
        return False


def progress_task(
    *,
    desc: str,
    enabled: bool = False,
    total: Optional[int] = None,
    unit: str = "step",
    log_every: Optional[int] = None,
    stream: Optional[TextIO] = None,
) -> _ProgressTask:
    """Return a manual progress reporter for work that cannot be wrapped."""

    return _ProgressTask(
        desc=desc,
        enabled=enabled,
        total=total,
        unit=unit,
        log_every=log_every,
        stream=stream,
    )
