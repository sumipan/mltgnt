"""Append-only message store: one ``YYYY-MM-DD.jsonl`` file per day, one JSON row per line.

Every row is a full snapshot of one message (``ts`` / ``message_id`` / ``author`` /
``text`` / ``thread_ts`` / ``kind`` / ``status`` / ``task_uuid``, plus ``files`` when
the message carries images). ``kind`` says why
the row was written: ``message`` (new), ``update`` (text changed), ``status``,
``bookmark`` or ``reaction``. Rows are only appended under ``fcntl.flock``; the latest
row of a message_id wins for display kinds. Bookmark and reaction rows are append-only
events that derive ``bookmarked`` and ``reactions`` on the target message.
"""

from __future__ import annotations

from typing import Any

from mltgnt.media._core.jsonl_store import KINDS, MESSAGE_KINDS, JsonlMessageStore

__all__ = ["KINDS", "MESSAGE_KINDS", "WebChatStore"]


class WebChatStore(JsonlMessageStore):
    """Day-file JSONL store under ``store_dir``. ``clock`` decides ``ts`` and the day file."""

    def append(  # type: ignore[override]  # legacy API intentionally excludes JsonlMessageStore.extra
        self,
        *,
        message_id: str,
        author: str,
        text: str,
        thread_ts: str | None = None,
        kind: str = "message",
        status: str | None = None,
        task_uuid: str | None = None,
        bookmarked: bool | None = None,
        reaction: str | None = None,
        files: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Append a webchat row while preserving the legacy public signature."""
        return super().append(
            message_id=message_id,
            author=author,
            text=text,
            thread_ts=thread_ts,
            kind=kind,
            status=status,
            task_uuid=task_uuid,
            bookmarked=bookmarked,
            reaction=reaction,
            files=files,
        )
