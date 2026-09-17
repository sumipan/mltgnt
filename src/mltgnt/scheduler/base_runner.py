"""mltgnt.scheduler.base_runner — shared ABC for the tick loop."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class BaseRunner(ABC):
    @abstractmethod
    def tick(self, now: datetime | None = None) -> Any:
        ...
