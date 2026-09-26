"""Download image files attached to a Slack event and store them locally.

Layout: ``<dest_root>/<channel>/<YYYY-MM-DD>/<file id>_<name><ext>`` (date in ``tz``).
"""

from __future__ import annotations

import logging
import re
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any

__all__ = ["SavedMedia", "ext_from_mimetype", "safe_filename", "save_images"]

_log = logging.getLogger(__name__)

_STEM_MAX = 80
_UNSAFE_CHARS_RE = re.compile(r"[/\\:*?\"<>|\x00-\x1f]")
_EXT_BY_MIMETYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}

Fetch = Callable[[str, Mapping[str, str]], bytes]


@dataclass(frozen=True)
class SavedMedia:
    path: Path
    file_id: str
    mimetype: str
    original_name: str | None


def _urllib_fetch(url: str, headers: Mapping[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - URL comes from Slack file metadata
        data: bytes = resp.read()
    return data


def save_images(
    event: Mapping[str, Any],
    *,
    dest_root: Path,
    tz: tzinfo,
    token: str | None = None,
    fetch: Fetch | None = None,
    logger: logging.Logger | None = None,
    max_bytes: int = 20_971_520,
) -> list[SavedMedia]:
    """Save the ``image/*`` entries of ``event["files"]``; return the ones saved.

    Non-images, oversized files, and files without a URL are skipped; download or
    write failures are logged and skipped. ``fetch(url, headers)`` defaults to urllib.
    """
    log = logger or _log
    files = event.get("files")
    if not files:
        return []
    get = fetch or _urllib_fetch
    dest_dir = _dest_dir_for_event(event, dest_root, tz)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    results: list[SavedMedia] = []
    for f in files:
        file_id = str(f.get("id") or "UNKNOWN")
        mimetype = str(f.get("mimetype") or "")
        if not mimetype.startswith("image/"):
            log.info("media_store: skip non-image file id=%s mimetype=%s", file_id, mimetype)
            continue
        size = f.get("size") or 0
        if size > max_bytes:
            log.warning("media_store: skip oversized file id=%s size=%d (max=%d)", file_id, size, max_bytes)
            continue
        url = f.get("url_private_download") or f.get("url_private")
        if not url:
            log.warning("media_store: no download URL for file id=%s", file_id)
            continue
        try:
            data = get(url, headers)
        except Exception as exc:
            log.error("media_store: download failed id=%s: %s", file_id, exc)
            continue

        original_name = f.get("name") or None
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            path = dest_dir / safe_filename(file_id, original_name, ext_from_mimetype(mimetype), dest_dir)
            path.write_bytes(data)
        except OSError as exc:
            log.error("media_store: write failed id=%s dir=%s: %s", file_id, dest_dir, exc)
            continue
        results.append(SavedMedia(path=path, file_id=file_id, mimetype=mimetype, original_name=original_name))
        log.info("media_store: saved id=%s -> %s", file_id, path)
    return results


def _dest_dir_for_event(event: Mapping[str, Any], dest_root: Path, tz: tzinfo) -> Path:
    channel = str(event.get("channel") or "UNKNOWN")
    try:
        epoch = float(str(event.get("ts", "0")).split(".")[0])
    except ValueError:
        epoch = 0.0
    return dest_root / channel / datetime.fromtimestamp(epoch, tz=tz).strftime("%Y-%m-%d")


def safe_filename(file_id: str, name: str | None, ext: str, dest_dir: Path) -> str:
    """OS-safe ``<file id>_<stem><ext>``; ``_<n>`` is appended on collision."""
    if name:
        stem = _UNSAFE_CHARS_RE.sub("", Path(name).stem).replace("..", "").lstrip(".")
        base = f"{file_id}_{stem[:_STEM_MAX] or file_id}"
    else:
        base = file_id
    candidate = f"{base}{ext}"
    n = 1
    while (dest_dir / candidate).exists():
        candidate = f"{base}_{n}{ext}"
        n += 1
    return candidate


def ext_from_mimetype(mimetype: str) -> str:
    """``image/png`` -> ``.png``; unknown subtypes map to ``.<subtype>`` (``.bin`` if empty)."""
    ext = _EXT_BY_MIMETYPE.get(mimetype)
    if ext:
        return ext
    sub = mimetype.split("/", 1)[-1].split(";")[0].strip()
    return f".{sub}" if sub else ".bin"
