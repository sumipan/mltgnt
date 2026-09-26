"""WebChat ``POST /messages`` body -> MediaEvent.

The text is kept as typed: an ``@name`` call stays in ``text`` and in ``raw`` and is
left to the host (hooks) to interpret.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mltgnt.media._core import id_map
from mltgnt.media._core.types import MediaEvent

__all__ = ["DEFAULT_AUTHOR", "to_media_event"]

DEFAULT_AUTHOR = "user"


def _optional_str(body: Mapping[str, Any], key: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip() or None


def to_media_event(body: Any, *, space_id: str, message_id: str) -> MediaEvent:
    """Build a MediaEvent for a new message ``message_id``. Raise ValueError on an invalid body.

    ``text`` must be a non-blank string. ``author`` defaults to ``DEFAULT_AUTHOR``.
    A message without ``thread_ts`` starts its own thread, so ``conversation_id`` is
    ``<space_id>:<thread_ts or message_id>``.
    """
    if not isinstance(body, Mapping):
        raise ValueError("body must be a JSON object")
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    thread_ts = _optional_str(body, "thread_ts")
    author = _optional_str(body, "author") or DEFAULT_AUTHOR
    return MediaEvent(
        space_id=space_id,
        conversation_id=id_map.to_conversation_id(space_id, thread_ts or message_id),
        message_id=message_id,
        author=author,
        text=text.strip(),
        raw=dict(body),
    )
