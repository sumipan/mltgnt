import os
from pathlib import Path


class PidLock:
    """Single-instance control via a PID file."""

    def __init__(self, pid_file: Path) -> None:
        self._pid_file = pid_file

    def acquire(self) -> bool:
        """Acquire the lock. True on success; False if another process holds it."""
        if self._pid_file.exists():
            try:
                pid = int(self._pid_file.read_text().strip())
                os.kill(pid, 0)
                # Process is alive
                return False
            except PermissionError:
                # Process exists but owned by another user -- treat as alive
                return False
            except (ValueError, ProcessLookupError):
                # stale PID or unreadable -- fall through to overwrite
                pass
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(os.getpid()))
        return True

    def release(self) -> None:
        """Remove the PID file."""
        try:
            self._pid_file.unlink()
        except FileNotFoundError:
            pass
