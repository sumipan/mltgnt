"""Media-agnostic client contract (MediaClient / Status) and the legacy adapter.

Lives in the lowest layer so that core packages (e.g. ``mltgnt.scheduler``) can
reference the type without importing ``mltgnt.media``.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = ["Status", "MediaClient", "adapt_client"]


class Status(str, Enum):
    """Processing status shown on a posted message."""

    RECEIVED = "received"
    WORKING = "working"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@runtime_checkable
class MediaClient(Protocol):
    """Post / update messages on a medium. Failures return None / False (never raise)."""

    def post(self, text: str, space: str, thread: str | None = None) -> str | None:
        """Post ``text`` to ``space`` (optionally in ``thread``). Return the message id, or None on failure."""
        ...

    def update(self, message_id: str, text: str) -> bool:
        """Replace the text of ``message_id``. Return False on failure."""
        ...

    def set_status(self, message_id: str, status: Status) -> bool:
        """Show ``status`` on ``message_id``. Return False on failure."""
        ...

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        """Upload the file at ``path``. Default: unsupported (False)."""
        return False


class _PostMessageAdapter:
    """Expose a legacy ``post_message`` client as a MediaClient."""

    def __init__(self, legacy: Any) -> None:
        self._legacy = legacy

    def post(self, text: str, space: str, thread: str | None = None, **extra: Any) -> str | None:
        post_message_ts = getattr(self._legacy, "post_message_ts", None)
        if callable(post_message_ts) and not extra:
            return post_message_ts(text, channel=space, thread_ts=thread)  # type: ignore[no-any-return]
        ok = self._legacy.post_message(text, channel=space, thread_ts=thread, **extra)
        return "" if ok else None

    def update(self, message_id: str, text: str) -> bool:
        return False

    def set_status(self, message_id: str, status: Status) -> bool:
        return False

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        return False


def adapt_client(obj: Any) -> MediaClient:
    """Return ``obj`` as a MediaClient.

    Objects with ``post`` are returned unchanged. Objects with only the deprecated
    ``post_message`` (SlackClientProtocol) are wrapped, with one DeprecationWarning.
    """
    if callable(getattr(obj, "post", None)):
        return obj  # type: ignore[no-any-return]
    if callable(getattr(obj, "post_message", None)):
        warnings.warn(
            "SlackClientProtocol (post_message) is deprecated; use MediaClient (post / update / set_status)",
            DeprecationWarning,
            stacklevel=2,
        )
        return _PostMessageAdapter(obj)
    raise TypeError(f"{type(obj).__name__} has neither post nor post_message")
