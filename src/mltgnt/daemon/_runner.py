import logging
import signal
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import DaemonComponent

from mltgnt.exceptions import DependencyError

logger = logging.getLogger("mltgnt.daemon")


class DaemonRunner:
    """Manage component start/stop and signal handling."""

    def __init__(
        self,
        *,
        pid_file: Path,
        components: list["DaemonComponent"],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        from ._pidlock import PidLock

        self._pid_lock = PidLock(pid_file)
        self._components = list(components)
        self._logger = logger or logging.getLogger("mltgnt.daemon")
        self._stop_event = threading.Event()

        # Register signal handlers on the main thread at construction time so
        # that tests can instantiate DaemonRunner from the main thread and then
        # call run() from a worker thread (signal.signal() requires main thread).
        if threading.current_thread() is threading.main_thread():
            def _signal_handler(signum, frame):
                self._logger.info("Signal %s received, shutting down.", signum)
                self._stop_event.set()

            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)

    def run(self) -> None:
        """
        Main entry point.
        1. Acquire PID lock (DependencyError on failure)
        2. Register SIGTERM/SIGINT handlers (when on the main thread)
        3. Call start() on all components (registration order)
        4. Wait until a signal is received
        5. Call stop() on all components (reverse order)
        6. Release PID lock
        """
        if not self._pid_lock.acquire():
            self._logger.error("Another instance is already running.")
            raise DependencyError("Another instance is already running.")

        if threading.current_thread() is threading.main_thread():
            def _signal_handler(signum, frame):
                self._logger.info("Signal %s received, shutting down.", signum)
                self._stop_event.set()

            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)

        started: list["DaemonComponent"] = []
        try:
            for component in self._components:
                self._logger.info("Starting component: %s", component.name)
                component.start()
                started.append(component)
        except Exception:
            self._logger.exception("Component start failed, rolling back.")
            for comp in reversed(started):
                try:
                    comp.stop()
                except Exception:
                    self._logger.exception("Error stopping component %s during rollback.", comp.name)
            self._pid_lock.release()
            raise

        try:
            self._stop_event.wait()
        finally:
            self._logger.info("Stopping components.")
            for comp in reversed(self._components):
                try:
                    comp.stop()
                except Exception:
                    self._logger.exception("Error stopping component %s.", comp.name)
            self._pid_lock.release()

    def stop(self) -> None:
        """Request stop from outside (for tests)."""
        self._stop_event.set()
