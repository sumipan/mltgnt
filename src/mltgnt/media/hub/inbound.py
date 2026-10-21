"""Hub envelope rows -> MediaEvent (no FastAPI)."""

from __future__ import annotations

from mltgnt.interfaces.hub import HubMessage
from mltgnt.media._core import id_map
from mltgnt.media._core.types import MediaEvent

__all__ = ["to_media_event"]


def _author_from_sender(message: HubMessage) -> str:
    host = message.sender.host
    persona = message.sender.persona
    if persona is None:
        return f"{host}/user"
    return f"{host}/{persona}"


def to_media_event(message: HubMessage, *, space_prefix: str) -> MediaEvent:
    """Map one hub message to a MediaEvent for the host bridge."""
    space_id = f"{space_prefix}{message.room}"
    thread_key = message.thread or message.id
    author = _author_from_sender(message)
    return MediaEvent(
        space_id=space_id,
        conversation_id=id_map.to_conversation_id(space_id, thread_key),
        message_id=message.id,
        author=author,
        text=f"[{author}] {message.text}",
        attachments=message.attachments,
        raw=message,
    )
