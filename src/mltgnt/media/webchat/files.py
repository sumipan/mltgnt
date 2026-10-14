"""Image files attached to WebChat messages.

Layout: ``<files_dir>/<YYYY-MM-DD>/<32 hex>.<ext>``. The file id is the path relative
to ``files_dir`` (``<YYYY-MM-DD>/<32 hex>.<ext>``) and is also the URL tail of
``GET /files/{day}/{name}``. Only the image types in ``CONTENT_TYPES`` are accepted.
"""

from __future__ import annotations

import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["CONTENT_TYPES", "WebChatFiles", "content_type_for"]

CONTENT_TYPES: Mapping[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}
_TYPE_BY_EXT = {ext: ctype for ctype, ext in CONTENT_TYPES.items()}
_TYPE_BY_EXT["jpeg"] = "image/jpeg"
_ID_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})/([0-9a-f]{32})\.(png|jpg|gif|webp)$")
_NAME_MAX = 120
_UNSAFE_NAME_RE = re.compile(r"[\x00-\x1f/\\]")


def _now() -> datetime:
    return datetime.now().astimezone()


def content_type_for(name: str) -> str | None:
    """Image content type from the file extension of ``name``; None when not supported."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _TYPE_BY_EXT.get(ext)


def _display_name(name: str | None, fallback: str) -> str:
    text = _UNSAFE_NAME_RE.sub("_", (name or "").strip())
    return text[:_NAME_MAX] or fallback


class WebChatFiles:
    """Save and look up message images under ``files_dir``."""

    def __init__(
        self,
        files_dir: Path,
        *,
        max_bytes: int = 20 * 1024 * 1024,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self._dir = Path(files_dir)
        self._max_bytes = max_bytes
        self._clock = clock

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def save(self, data: bytes, *, content_type: str, name: str | None = None) -> dict[str, Any]:
        """Store ``data`` and return its file entry. ValueError on a bad type or size; OSError propagates."""
        ctype = (content_type or "").split(";", 1)[0].strip().lower()
        ext = CONTENT_TYPES.get(ctype)
        if ext is None:
            raise ValueError(f"unsupported content type: {content_type!r}")
        if not data:
            raise ValueError("empty file")
        if len(data) > self._max_bytes:
            raise ValueError(f"file too large: {len(data)} > {self._max_bytes} bytes")
        file_id = f"{self._clock().date().isoformat()}/{uuid.uuid4().hex}.{ext}"
        path = self._dir / file_id
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return {"id": file_id, "name": _display_name(name, path.name), "content_type": ctype, "size": len(data)}

    def save_path(self, source: Path, *, name: str | None = None) -> dict[str, Any]:
        """Copy an existing image file in. ValueError on a bad type or size; OSError propagates."""
        source = Path(source)
        ctype = content_type_for(source.name)
        if ctype is None:
            raise ValueError(f"unsupported file type: {source.name}")
        size = source.stat().st_size
        if size == 0 or size > self._max_bytes:
            raise ValueError(f"bad file size: {size}")
        file_id = f"{self._clock().date().isoformat()}/{uuid.uuid4().hex}.{CONTENT_TYPES[ctype]}"
        path = self._dir / file_id
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        return {"id": file_id, "name": _display_name(name or source.name, path.name), "content_type": ctype, "size": size}

    def path(self, file_id: str) -> Path | None:
        """Path of an existing file for a well-formed id; None otherwise."""
        if not isinstance(file_id, str) or not _ID_RE.match(file_id):
            return None
        candidate = self._dir / file_id
        return candidate if candidate.is_file() else None

    def entry(self, file_id: str, name: str | None = None) -> dict[str, Any] | None:
        """File entry for a message row (``id`` / ``name`` / ``content_type`` / ``size``); None when unknown."""
        path = self.path(file_id)
        if path is None:
            return None
        return {
            "id": file_id,
            "name": _display_name(name, path.name),
            "content_type": content_type_for(path.name),
            "size": path.stat().st_size,
        }
