"""Settings shared by media implementations. Implementations subclass MediaConfig."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mltgnt.config.language import JA, LanguagePack

__all__ = ["MediaConfig"]


@dataclass(frozen=True)
class MediaConfig:
    """Paths are required (no defaults); the host decides where state lives."""

    state_dir: Path
    pending_dir: Path
    events_dir: Path
    language: LanguagePack = field(default=JA)
    progress_min_interval_sec: float = 5.0
    approval_ttl_sec: float = 600.0
