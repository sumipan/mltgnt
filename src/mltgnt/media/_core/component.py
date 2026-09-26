"""Supervise a media bridge: run it, report each exit, restart with backoff.

Auditing and alerting stay in the host: every exit is handed to ``on_exit(code)``.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from collections.abc import Callable, Sequence
from typing import Literal

__all__ = ["MediaBridgeComponent"]

_log = logging.getLogger(__name__)


def _exit_code(start: Callable[[], None]) -> int:
    """Run ``start`` in the calling thread and turn its outcome into an exit code."""
    try:
        start()
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    except Exception:
        _log.exception("[MediaBridgeComponent] bridge raised")
        return 1
    return 0


class MediaBridgeComponent:
    """Run ``start`` and restart it after ``backoff_sec[0]``, ``backoff_sec[1]``, ... seconds.

    ``mode="process"`` runs ``start`` in a child process (exit code from the
    process), ``mode="thread"`` runs it in the supervisor thread (return is 0,
    ``SystemExit`` gives its code, any other exception is 1). After
    ``len(backoff_sec)`` restarts the component gives up.
    """

    def __init__(
        self,
        start: Callable[[], None],
        *,
        mode: Literal["process", "thread"],
        backoff_sec: Sequence[float],
        on_exit: Callable[[int], None] | None = None,
    ) -> None:
        if mode not in ("process", "thread"):
            raise ValueError(f"unknown mode: {mode!r}")
        self._start = start
        self._mode = mode
        self._backoff_sec = tuple(backoff_sec)
        self._on_exit = on_exit
        self._stop_event = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._proc: multiprocessing.process.BaseProcess | None = None

    @property
    def name(self) -> str:
        return "media_bridge"

    def start(self) -> None:
        """Begin supervising on a daemon thread."""
        self._stop_event.clear()
        self._supervisor = threading.Thread(target=self.run, daemon=True, name="MediaBridgeComponent")
        self._supervisor.start()

    def stop(self, timeout: float | None = None) -> None:
        """Stop restarting, terminate a child process and wait for the supervisor."""
        self._stop_event.set()
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.terminate()
        if self._supervisor is not None:
            self._supervisor.join(timeout)

    def join(self, timeout: float | None = None) -> None:
        if self._supervisor is not None:
            self._supervisor.join(timeout)

    def run(self) -> None:
        """Supervise in the calling thread until stopped or out of restarts."""
        attempt = 0
        while not self._stop_event.is_set():
            code = self._run_once()
            if self._on_exit is not None:
                try:
                    self._on_exit(code)
                except Exception:
                    _log.exception("[MediaBridgeComponent] on_exit failed")
            if self._stop_event.is_set():
                return
            if attempt >= len(self._backoff_sec):
                _log.warning("[MediaBridgeComponent] giving up after %d restarts", attempt)
                return
            delay = self._backoff_sec[attempt]
            attempt += 1
            _log.info("[MediaBridgeComponent] exit code=%d; restart in %.1fs", code, delay)
            if self._stop_event.wait(delay):
                return

    def _run_once(self) -> int:
        if self._mode == "thread":
            return _exit_code(self._start)
        proc = multiprocessing.get_context().Process(target=self._start, name="media-bridge")
        self._proc = proc
        proc.start()
        proc.join()
        self._proc = None
        return proc.exitcode if proc.exitcode is not None else 1
