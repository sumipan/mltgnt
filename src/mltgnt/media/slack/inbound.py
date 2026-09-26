"""Slack event dict -> MediaEvent."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from mltgnt.interfaces.turn import Attachment
from mltgnt.media._core import id_map
from mltgnt.media._core.types import MediaEvent

__all__ = ["files_to_attachments", "strip_mentions", "thread_ts_from_event", "to_media_event"]

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")


def strip_mentions(text: str) -> str:
    """Remove ``<@U...>`` mention tokens."""
    return _MENTION_RE.sub("", text or "").strip()


def thread_ts_from_event(event: Mapping[str, Any]) -> str:
    """Thread to reply in: ``thread_ts``, else the message's own ``ts``."""
    return str(event.get("thread_ts") or event.get("ts") or "").strip()


def files_to_attachments(files: object) -> tuple[Attachment, ...]:
    """Convert a Slack ``files`` list (file_share) into attachment references."""
    if not isinstance(files, list):
        return ()
    out: list[Attachment] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        out.append(
            Attachment(
                name=str(f.get("name") or f.get("id") or "file"),
                content_type=f.get("mimetype") or None,
                uri=f.get("url_private_download") or f.get("url_private") or None,
            )
        )
    return tuple(out)


def to_media_event(
    event: Mapping[str, Any],
    *,
    strip_bot_mentions: bool | None = None,
    text_override: str | None = None,
) -> MediaEvent:
    """Build a MediaEvent from a Bolt event dict.

    Mentions are stripped for ``app_mention`` events unless ``strip_bot_mentions``
    says otherwise. ``files`` (file_share) become ``attachments``.
    ``message_id`` is the event ``ts``; ``conversation_id`` is ``<channel>:<thread ts>``.
    """
    channel = str(event.get("channel") or "")
    thread_ts = thread_ts_from_event(event)
    if strip_bot_mentions is None:
        strip_bot_mentions = event.get("type") == "app_mention"
    raw_text = text_override if text_override is not None else str(event.get("text") or "")
    text = strip_mentions(raw_text) if strip_bot_mentions else raw_text.strip()
    return MediaEvent(
        space_id=channel,
        conversation_id=id_map.to_conversation_id(channel, thread_ts) if channel and thread_ts else "",
        message_id=str(event.get("ts") or ""),
        author=str(event.get("user") or ""),
        text=text,
        attachments=files_to_attachments(event.get("files")),
        raw=event,
    )
