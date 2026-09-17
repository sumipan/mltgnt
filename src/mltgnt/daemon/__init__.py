from typing import Protocol, runtime_checkable

from ._pidlock import PidLock
from ._runner import DaemonRunner
from ._skill_watcher import SkillWatcherComponent


@runtime_checkable
class DaemonComponent(Protocol):
    """Type contract for components registerable with the daemon."""

    @property
    def name(self) -> str: ...

    def start(self) -> None:
        """Start the component. Must not block."""
        ...

    def stop(self) -> None:
        """Stop the component, including resource cleanup."""
        ...


__all__ = ["DaemonComponent", "DaemonRunner", "PidLock", "SkillWatcherComponent"]
