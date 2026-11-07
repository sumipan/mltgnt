"""Matrix client event dict -> MediaEvent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from mltgnt.interfaces.turn import Attachment
from mltgnt.media._core import id_map
from mltgnt.media._core.types import MediaEvent
from mltgnt.media.matrix.config import MatrixMediaConfig

__all__ = ["to_media_event"]


def _sender(event: Mapping[str, Any]) -> str | None:
    raw = event.get("sender")
    return raw if isinstance(raw, str) else None


def _strip_quote_fallback(body: str, relates: Mapping[str, Any]) -> str:
    in_reply = relates.get("m.in_reply_to")
    if not isinstance(in_reply, Mapping):
        return body
    if relates.get("is_falling_back") is True:
        return body
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("> "):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        out.extend(lines[i:])
        break
    return "\n".join(out).strip() if out else body.strip()


def to_media_event(
    event: Mapping[str, Any],
    *,
    room_id: str,
    config: MatrixMediaConfig,
    files: Callable[[str, Mapping[str, Any]], Attachment | None],
) -> MediaEvent | None:
    """Map one Matrix timeline event to a MediaEvent, or None when ignored."""
    if event.get("type") != "m.room.message":
        return None
    sender = _sender(event)
    if sender is None or sender == config.user_id:
        return None
    if sender not in config.allowed_senders:
        return None
    space_id = config.space_for_room(room_id)
    if space_id is None:
        return None
    content = event.get("content")
    if not isinstance(content, Mapping):
        return None
    msgtype = content.get("msgtype")
    if msgtype == "m.notice":
        return None
    if msgtype not in ("m.text", "m.image", "m.file"):
        return None
    relates = content.get("m.relates_to")
    if isinstance(relates, Mapping) and relates.get("rel_type") == "m.replace":
        return None

    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        return None

    thread_root = event_id
    if isinstance(relates, Mapping) and relates.get("rel_type") == "m.thread":
        root = relates.get("event_id")
        if isinstance(root, str) and root:
            thread_root = root

    conversation_id = id_map.to_conversation_id(space_id, thread_root)
    attachments: tuple[Attachment, ...] = ()

    if msgtype == "m.text":
        body = content.get("body")
        if not isinstance(body, str):
            return None
        text = _strip_quote_fallback(body, relates) if isinstance(relates, Mapping) else body
    else:
        body = content.get("body")
        if not isinstance(body, str):
            body = ""
        text = body
        url = content.get("url")
        info = content.get("info")
        if isinstance(url, str) and isinstance(info, Mapping):
            att = files(url, info)
            if att is not None:
                attachments = (att,)

    return MediaEvent(
        space_id=space_id,
        conversation_id=conversation_id,
        message_id=event_id,
        author=sender,
        text=text,
        attachments=attachments,
        raw=dict(event),
    )
