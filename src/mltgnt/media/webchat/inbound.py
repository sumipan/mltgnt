"""WebChat ``POST /messages`` body -> MediaEvent.

The text is kept as typed: an ``@name`` call stays in ``text`` and in ``raw`` and is
left to the host (hooks) to interpret. Images uploaded with ``POST /files`` are
referenced by ``files: [{"id": ..., "name": ...}]`` and become ``attachments`` whose
``uri`` is the local file path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mltgnt.interfaces.turn import Attachment
from mltgnt.media._core import id_map
from mltgnt.media._core.types import MediaEvent
from mltgnt.media.webchat.files import WebChatFiles

__all__ = ["DEFAULT_AUTHOR", "MAX_FILES", "file_entries", "to_media_event"]

DEFAULT_AUTHOR = "user"
MAX_FILES = 10


def _optional_str(body: Mapping[str, Any], key: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip() or None


def file_entries(body: Mapping[str, Any], files: WebChatFiles | None) -> list[dict[str, Any]]:
    """Resolve ``body["files"]`` to stored file entries. Raise ValueError on a bad or unknown file."""
    raw = body.get("files")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("files must be a list")
    if raw and files is None:
        raise ValueError("files are not supported")
    if len(raw) > MAX_FILES:
        raise ValueError(f"at most {MAX_FILES} files")
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError("each file must be an object with a string id")
        name = item.get("name")
        entry = files.entry(item["id"], name if isinstance(name, str) else None) if files is not None else None
        if entry is None:
            raise ValueError(f"unknown file: {item['id']}")
        out.append(entry)
    return out


def to_media_event(
    body: Any,
    *,
    space_id: str,
    message_id: str,
    files: WebChatFiles | None = None,
) -> MediaEvent:
    """Build a MediaEvent for a new message ``message_id``. Raise ValueError on an invalid body.

    ``text`` must be a non-blank string unless the body carries ``files`` (then it may
    be empty). ``author`` defaults to ``DEFAULT_AUTHOR``.
    A message without ``thread_ts`` starts its own thread, so ``conversation_id`` is
    ``<space_id>:<thread_ts or message_id>``.
    """
    if not isinstance(body, Mapping):
        raise ValueError("body must be a JSON object")
    entries = file_entries(body, files)
    text = body.get("text", "")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not text.strip() and not entries:
        raise ValueError("text must be a non-empty string")
    thread_ts = _optional_str(body, "thread_ts")
    author = _optional_str(body, "author") or DEFAULT_AUTHOR
    return MediaEvent(
        space_id=space_id,
        conversation_id=id_map.to_conversation_id(space_id, thread_ts or message_id),
        message_id=message_id,
        author=author,
        text=text.strip(),
        attachments=tuple(
            Attachment(name=e["name"], content_type=e["content_type"], uri=str(files.path(e["id"])))
            for e in entries
            if files is not None
        ),
        raw=dict(body),
    )
