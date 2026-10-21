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

from mltgnt.media._core.jsonl_store import KINDS, MESSAGE_KINDS, JsonlMessageStore

__all__ = ["KINDS", "MESSAGE_KINDS", "WebChatStore"]


class WebChatStore(JsonlMessageStore):
    """Day-file JSONL store under ``store_dir``. ``clock`` decides ``ts`` and the day file."""
