"""WebChatClient: MediaClient implementation over the WebChat store.

``post`` appends a new message, ``update`` / ``set_status`` append a new snapshot
of an existing one. Failures are logged and reported as None / False.
"""

from __future__ import annotations

import logging
import uuid

from mltgnt.interfaces.media import Status
from mltgnt.media.webchat.config import WebChatMediaConfig
from mltgnt.media.webchat.store import WebChatStore

__all__ = ["WebChatClient"]

_log = logging.getLogger(__name__)


class WebChatClient:
    """Single-channel client: ``space`` is accepted for the contract but not stored."""

    def __init__(
        self,
        config: WebChatMediaConfig,
        *,
        store: WebChatStore | None = None,
        author: str = "assistant",
    ) -> None:
        self._config = config
        self._store = store if store is not None else WebChatStore(config.store_dir)
        self._author = author

    @property
    def store(self) -> WebChatStore:
        return self._store

    def post(self, text: str, space: str, thread: str | None = None) -> str | None:
        message_id = uuid.uuid4().hex
        try:
            self._store.append(message_id=message_id, author=self._author, text=text, thread_ts=thread or None)
        except OSError as exc:
            _log.warning("webchat client: post failed space=%s: %s", space, exc)
            return None
        return message_id

    def update(self, message_id: str, text: str) -> bool:
        return self._revise(message_id, "update", text=text)

    def set_status(self, message_id: str, status: Status) -> bool:
        return self._revise(message_id, "status", status=Status(status).value)

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        """Unsupported: file posting is not implemented for WebChat."""
        return False

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
