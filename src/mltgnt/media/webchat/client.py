"""WebChatClient: MediaClient implementation over the WebChat store.

``post`` appends a new message, ``update`` / ``set_status`` append a new snapshot
of an existing one, ``upload`` copies an image into the files directory and posts it. Failures are logged and reported as None / False.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mltgnt.interfaces.hub import HubDisplay
from mltgnt.interfaces.media import Status
from mltgnt.media.webchat.config import WebChatMediaConfig
from mltgnt.media.webchat.files import WebChatFiles
from mltgnt.media.webchat.store import WebChatStore

__all__ = ["WebChatClient"]

_log = logging.getLogger(__name__)


def _author_from_extra(extra: Mapping[str, Any] | None, default: str) -> str:
    if extra is None or "author" not in extra:
        return default
    raw = extra["author"]
    if isinstance(raw, str) and raw.strip():
        return raw
    _log.warning("webchat client: ignoring invalid extra author=%r", raw)
    return default


def _display_from_extra(extra: Mapping[str, Any] | None) -> dict[str, str] | None:
    if extra is None or "display" not in extra:
        return None
    raw = extra["display"]
    if not isinstance(raw, Mapping):
        _log.warning("webchat client: ignoring invalid extra display=%r", raw)
        return None
    try:
        display = HubDisplay.from_mapping({k: raw[k] for k in ("name", "icon_url") if k in raw})
    except ValueError:
        _log.warning("webchat client: ignoring invalid extra display=%r", raw)
        return None
    if display is None:
        _log.warning("webchat client: ignoring empty extra display=%r", raw)
        return None
    mapping = display.to_mapping()
    if not mapping:
        _log.warning("webchat client: ignoring empty extra display=%r", raw)
        return None
    return mapping


class WebChatClient:
    """Single-channel client: ``space`` is accepted for the contract but not stored."""

    def __init__(
        self,
        config: WebChatMediaConfig,
        *,
        store: WebChatStore | None = None,
        files: WebChatFiles | None = None,
        author: str = "assistant",
    ) -> None:
        self._config = config
        self._store = store if store is not None else WebChatStore(config.store_dir)
        self._files = (
            files
            if files is not None
            else WebChatFiles(config.resolved_files_dir, max_bytes=config.max_upload_bytes)
        )
        self._author = author

    @property
    def store(self) -> WebChatStore:
        return self._store

    def post(
        self,
        text: str,
        space: str,
        thread: str | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> str | None:
        message_id = uuid.uuid4().hex
        author = _author_from_extra(extra, self._author)
        append_extra: dict[str, Any] = {}
        display = _display_from_extra(extra)
        if display is not None:
            append_extra["display"] = display
        try:
            self._store.append(
                message_id=message_id,
                author=author,
                text=text,
                thread_ts=thread or None,
                extra=append_extra or None,
            )
        except OSError as exc:
            _log.warning("webchat client: post failed space=%s: %s", space, exc)
            return None
        return message_id

    def update(self, message_id: str, text: str) -> bool:
        return self._revise(message_id, "update", text=text)

    def set_status(self, message_id: str, status: Status) -> bool:
        return self._revise(message_id, "status", status=Status(status).value)

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        """Post the image at ``path`` as a message with no text. False for non-images or on failure."""
        try:
            entry = self._files.save_path(Path(path))
            self._store.append(
                message_id=uuid.uuid4().hex,
                author=self._author,
                text="",
                thread_ts=thread or None,
                files=[entry],
            )
        except (OSError, ValueError) as exc:
            _log.warning("webchat client: upload failed space=%s path=%s: %s", space, path, exc)
            return False
        return True

    def react(self, message_id: str, name: str) -> bool:
        try:
            row = self._store.add_reaction(message_id, name)
        except (OSError, ValueError) as exc:
            _log.warning("webchat client: reaction failed message_id=%s: %s", message_id, exc)
            return False
        if row is None:
            _log.warning("webchat client: unknown message_id=%s for reaction", message_id)
            return False
        return True

    def _revise(self, message_id: str, kind: str, **changes: str) -> bool:
        try:
            row = self._store.revise(message_id, kind, **changes)
        except (OSError, ValueError) as exc:
            _log.warning("webchat client: %s failed message_id=%s: %s", kind, message_id, exc)
            return False
        if row is None:
            _log.warning("webchat client: unknown message_id=%s", message_id)
            return False
        return True
