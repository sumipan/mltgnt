"""Pending-request metadata store (one JSON file per request id)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mltgnt.media._core.config import MediaConfig

__all__ = ["PendingStore"]


class PendingStore:
    """Save / load pending metadata as ``<store_dir>/<prefix><uid>.json``."""

    def __init__(self, store_dir: Path, *, prefix: str = "pending-") -> None:
        self._store_dir = store_dir
        self._prefix = prefix

    @classmethod
    def from_config(cls, config: MediaConfig, *, prefix: str = "pending-") -> PendingStore:
        """Store under ``config.pending_dir``."""
        return cls(config.pending_dir, prefix=prefix)

    def _path(self, uid: str) -> Path:
        return self._store_dir / f"{self._prefix}{uid}.json"

    def save(self, uid: str, metadata: dict[str, Any]) -> None:
        """Write atomically (temp file + rename)."""
        self._store_dir.mkdir(parents=True, exist_ok=True)
        target = self._path(uid)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(self._store_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False)
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def load(self, uid: str) -> dict[str, Any] | None:
        """Return the saved metadata, or None when missing or unreadable."""
        try:
            data = json.loads(self._path(uid).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def delete(self, uid: str) -> bool:
        try:
            self._path(uid).unlink()
            return True
        except FileNotFoundError:
            return False

    def consume(self, uid: str) -> dict[str, Any] | None:
        """Load and delete atomically: only one concurrent caller gets the dict."""
        src = self._path(uid)
        dst = src.with_suffix(".consumed")
        try:
            src.rename(dst)
        except FileNotFoundError:
            return None
        try:
            data = json.loads(dst.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = None
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
        return data if isinstance(data, dict) else None
