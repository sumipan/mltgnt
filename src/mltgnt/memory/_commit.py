"""mltgnt.memory._commit — debounced git commits of memory files (nexus #3833).

Each written path is committed through the ``memory`` sink of
``mltgnt.bridges.files_adapter.commit`` after ``debounce_sec`` of quiet.
Commit failures are logged and never reach the memory writer.
"""
from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DEFAULT_COMMIT_DEBOUNCE_SEC", "flush_memory_commits", "schedule_commit"]

_log = logging.getLogger(__name__)

DEFAULT_COMMIT_DEBOUNCE_SEC: float = 300.0


@dataclass
class _Pending:
    path: Path
    message: str
    timer: threading.Timer


_lock = threading.Lock()
_pending: dict[str, _Pending] = {}
_atexit_registered = False


def _commit_now(path: Path, message: str) -> None:
    from mltgnt.bridges.files_adapter import commit

    try:
        result = commit([path], message)
    except Exception as exc:  # noqa: BLE001 — memory writes must not fail on commit errors
        _log.warning("memory commit failed for %s: %s", path, exc)
        return
    if result.committed and not result.pushed and (result.reason or "").startswith("push failed"):
        _log.warning("memory commit push failed for %s: %s", path, result.reason)


def _fire(key: str, entry: _Pending) -> None:
    with _lock:
        if _pending.get(key) is not entry:
            return
        del _pending[key]
    _commit_now(entry.path, entry.message)


def schedule_commit(
    path: Path,
    persona: str,
    kind: str,
    *,
    debounce_sec: float = DEFAULT_COMMIT_DEBOUNCE_SEC,
) -> None:
    """Commit ``path`` after ``debounce_sec`` (the last call per path wins)."""
    global _atexit_registered
    message = f"mltgnt(memory): {persona} {kind}"
    key = str(Path(path).resolve())
    with _lock:
        if not _atexit_registered:
            atexit.register(flush_memory_commits)
            _atexit_registered = True
        old = _pending.pop(key, None)
        if old is not None:
            old.timer.cancel()
        if debounce_sec > 0:
            timer = threading.Timer(debounce_sec, lambda: _fire(key, entry))
            timer.daemon = True
            entry = _Pending(Path(path), message, timer)
            _pending[key] = entry
            timer.start()
            return
    _commit_now(Path(path), message)


def flush_memory_commits() -> int:
    """Commit every pending path now; return the number of commits attempted."""
    with _lock:
        entries = list(_pending.values())
        _pending.clear()
        for entry in entries:
            entry.timer.cancel()
    for entry in entries:
        _commit_now(entry.path, entry.message)
    return len(entries)
