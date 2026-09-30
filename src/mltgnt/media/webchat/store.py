"""Append-only message store: one ``YYYY-MM-DD.jsonl`` file per day, one JSON row per line.

Every row is a full snapshot of one message (``ts`` / ``message_id`` / ``author`` /
``text`` / ``thread_ts`` / ``kind`` / ``status`` / ``task_uuid``). ``kind`` says why
the row was written: ``message`` (new), ``update`` (text changed), ``status``,
``bookmark`` or ``reaction``. Rows are only appended under ``fcntl.flock``; the latest
row of a message_id wins for display kinds. Bookmark and reaction rows are append-only
events that derive ``bookmarked`` and ``reactions`` on the target message.
"""

from __future__ import annotations

import fcntl
import json
import logging
import threading
from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

__all__ = ["KINDS", "MESSAGE_KINDS", "WebChatStore"]

_log = logging.getLogger(__name__)

KINDS = ("message", "update", "status", "bookmark", "reaction")
MESSAGE_KINDS = ("message", "update", "status")


def _now() -> datetime:
    return datetime.now().astimezone()


def _latest_per_id(rows: Iterable[dict[str, Any]], *, kinds: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Last row of each message_id, ordered by the message's first appearance."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if kinds is not None and row.get("kind") not in kinds:
            continue
        message_id = row.get("message_id")
        if isinstance(message_id, str) and message_id:
            latest[message_id] = row
    return list(latest.values())


class WebChatStore:
    """Day-file JSONL store under ``store_dir``. ``clock`` decides ``ts`` and the day file."""

    def __init__(self, store_dir: Path, *, clock: Callable[[], datetime] = _now) -> None:
        self._dir = Path(store_dir)
        self._clock = clock
        self._lock = threading.Lock()

    def today(self) -> date:
        return self._clock().date()

    def path_for(self, day: date) -> Path:
        return self._dir / f"{day.isoformat()}.jsonl"

    def append(
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
    ) -> dict[str, Any]:
        """Append one row to today's file and return it. OSError propagates to the caller."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {kind!r}")
        now = self._clock()
        row: dict[str, Any] = {
            "ts": now.isoformat(timespec="microseconds"),
            "message_id": message_id,
            "author": author,
            "text": text,
            "thread_ts": thread_ts,
            "kind": kind,
            "status": status,
            "task_uuid": task_uuid,
        }
        if bookmarked is not None:
            row["bookmarked"] = bookmarked
        if reaction is not None:
            row["reaction"] = reaction
        line = json.dumps(row, ensure_ascii=False) + "\n"
        path = self.path_for(now.date())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return row

    def revise(self, message_id: str, kind: str, **changes: Any) -> dict[str, Any] | None:
        """Append a new snapshot of ``message_id`` with ``changes``; None when the id is unknown."""
        with self._lock:
            current = self.latest(message_id)
            if current is None:
                return None
            fields: dict[str, Any] = {k: current.get(k) for k in ("author", "text", "thread_ts", "status", "task_uuid")}
            fields.update(changes)
            return self.append(
                message_id=message_id,
                kind=kind,
                author=str(fields["author"] or ""),
                text=str(fields["text"] or ""),
                thread_ts=fields["thread_ts"],
                status=fields["status"],
                task_uuid=fields["task_uuid"],
            )

    def read_from(self, day: date, offset: int = 0) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        """Rows of ``day`` after byte ``offset`` as ``(end offset, row)``, and the new offset.

        A trailing line without a newline (still being written) is left for the next call.
        """
        path = self.path_for(day)
        if not path.is_file():
            return [], offset
        out: list[tuple[int, dict[str, Any]]] = []
        with path.open("rb") as f:
            f.seek(offset)
            while True:
                raw = f.readline()
                if not raw.endswith(b"\n"):
                    break
                offset += len(raw)
                try:
                    row = json.loads(raw)
                except ValueError:
                    _log.warning("[webchat store] skip broken line in %s", path)
                    continue
                if isinstance(row, dict):
                    out.append((offset, row))
        return out, offset

    def _rows(self, day: date) -> list[dict[str, Any]]:
        return [row for _, row in self.read_from(day)[0]]

    def _all_rows(self) -> list[dict[str, Any]]:
        return [row for day in self.days() for row in self._rows(day)]

    def days(self) -> list[date]:
        """Days that have a file, oldest first."""
        out: list[date] = []
        if not self._dir.is_dir():
            return out
        for path in self._dir.glob("*.jsonl"):
            try:
                out.append(date.fromisoformat(path.stem))
            except ValueError:
                continue
        return sorted(out)

    def _bookmark_state(self) -> dict[str, bool]:
        state: dict[str, bool] = {}
        for row in self._all_rows():
            if row.get("kind") == "bookmark" and isinstance(row.get("bookmarked"), bool):
                message_id = row.get("message_id")
                if isinstance(message_id, str) and message_id:
                    state[message_id] = row["bookmarked"]
        return state

    def _reaction_state(self) -> dict[str, list[str]]:
        state: dict[str, list[str]] = {}
        for row in self._all_rows():
            if row.get("kind") != "reaction":
                continue
            message_id = row.get("message_id")
            reaction = row.get("reaction")
            if isinstance(message_id, str) and message_id and isinstance(reaction, str) and reaction:
                state.setdefault(message_id, []).append(reaction)
        return state

    def _thread_summaries(self) -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        seen_reply_message: set[str] = set()
        for row in self._all_rows():
            if row.get("kind") != "message":
                continue
            message_id = row.get("message_id")
            thread_ts = row.get("thread_ts")
            if not isinstance(message_id, str) or not message_id:
                continue
            if not isinstance(thread_ts, str) or not thread_ts:
                continue
            if message_id in seen_reply_message:
                continue
            seen_reply_message.add(message_id)
            summary = summaries.setdefault(
                thread_ts,
                {"reply_count": 0, "last_reply_ts": None, "participants": []},
            )
            summary["reply_count"] += 1
            ts = row.get("ts")
            if isinstance(ts, str):
                summary["last_reply_ts"] = ts
            author = row.get("author")
            if isinstance(author, str) and author and author not in summary["participants"]:
                summary["participants"].append(author)
        return summaries

    def enrich(self, row: dict[str, Any]) -> dict[str, Any]:
        """Add derived ``reply_count``, ``last_reply_ts``, ``participants``, ``bookmarked`` and ``reactions``."""
        message_id = row.get("message_id")
        if not isinstance(message_id, str):
            return row
        bookmarks = self._bookmark_state()
        reactions = self._reaction_state()
        summaries = self._thread_summaries()
        out = dict(row)
        out["bookmarked"] = bookmarks.get(message_id, False)
        out["reactions"] = list(reactions.get(message_id, []))
        if row.get("thread_ts"):
            out["reply_count"] = 0
            out["last_reply_ts"] = None
            out["participants"] = []
        else:
            summary = summaries.get(message_id, {})
            out["reply_count"] = int(summary.get("reply_count", 0))
            out["last_reply_ts"] = summary.get("last_reply_ts")
            out["participants"] = list(summary.get("participants", []))
        return out

    def read(self, day: date) -> list[dict[str, Any]]:
        """Messages of ``day``: the last row of each message_id, with derived fields."""
        rows = _latest_per_id(self._rows(day), kinds=MESSAGE_KINDS)
        return [self.enrich(row) for row in rows]

    def read_all(self) -> list[dict[str, Any]]:
        """Messages of every day: the last row of each message_id, with derived fields."""
        rows = _latest_per_id(self._all_rows(), kinds=MESSAGE_KINDS)
        return [self.enrich(row) for row in rows]

    def latest(self, message_id: str) -> dict[str, Any] | None:
        """Last display row of ``message_id`` searching the newest day first."""
        for day in reversed(self.days()):
            found = None
            for row in self._rows(day):
                if row.get("message_id") == message_id and row.get("kind") in MESSAGE_KINDS:
                    found = row
            if found is not None:
                return self.enrich(found)
        return None

    def thread(self, thread_id: str) -> list[dict[str, Any]]:
        """The thread's root message and its replies."""
        rows = [
            row
            for row in self.read_all()
            if row.get("message_id") == thread_id or row.get("thread_ts") == thread_id
        ]
        return [self.enrich(row) for row in rows]

    def set_bookmark(self, message_id: str, bookmarked: bool) -> dict[str, Any] | None:
        """Append a bookmark row for a root message. None when the id is unknown or a reply."""
        current = self.latest(message_id)
        if current is None or current.get("thread_ts"):
            return None
        return self.append(
            message_id=message_id,
            author=str(current.get("author") or ""),
            text=str(current.get("text") or ""),
            thread_ts=None,
            kind="bookmark",
            bookmarked=bookmarked,
        )

    def add_reaction(self, message_id: str, name: str) -> dict[str, Any] | None:
        """Append a reaction row. None when the id is unknown."""
        current = self.latest(message_id)
        if current is None:
            return None
        return self.append(
            message_id=message_id,
            author=str(current.get("author") or ""),
            text=str(current.get("text") or ""),
            thread_ts=current.get("thread_ts"),
            kind="reaction",
            reaction=name,
        )

    def bookmarks(self) -> list[dict[str, Any]]:
        """Bookmarked root messages, newest bookmark first."""
        state = self._bookmark_state()
        bookmarked_ids = [mid for mid, on in state.items() if on]
        if not bookmarked_ids:
            return []
        rows = [row for row in self.read_all() if row.get("message_id") in bookmarked_ids and not row.get("thread_ts")]
        rows.sort(key=lambda row: row.get("ts") or "", reverse=True)
        return rows
