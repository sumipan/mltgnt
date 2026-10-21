"""Append-only JSONL message store (one file per day). Shared by webchat and hub."""

from __future__ import annotations

import fcntl
import json
import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

__all__ = ["KINDS", "MESSAGE_KINDS", "JsonlMessageStore"]

_log = logging.getLogger(__name__)

KINDS = ("message", "update", "status", "bookmark", "reaction")
MESSAGE_KINDS = ("message", "update", "status")

_ROW_FIELD_KEYS = frozenset(
    {
        "ts",
        "message_id",
        "author",
        "text",
        "thread_ts",
        "kind",
        "status",
        "task_uuid",
        "bookmarked",
        "reaction",
        "files",
    }
)
_DERIVED_ROW_KEYS = frozenset(
    {
        "bookmarked",
        "reactions",
        "reply_count",
        "last_reply_ts",
        "participants",
    }
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _latest_per_id(rows: Iterable[dict[str, Any]], *, kinds: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if kinds is not None and row.get("kind") not in kinds:
            continue
        message_id = row.get("message_id")
        if isinstance(message_id, str) and message_id:
            latest[message_id] = row
    return list(latest.values())


def _posted_before(row: dict[str, Any], before: datetime) -> bool:
    ts = row.get("ts")
    if not isinstance(ts, str):
        return False
    try:
        posted = datetime.fromisoformat(ts)
    except ValueError:
        return False
    if (posted.tzinfo is None) != (before.tzinfo is None):
        posted = posted.replace(tzinfo=None)
        before = before.replace(tzinfo=None)
    return posted < before


def _enrich_row(
    row: dict[str, Any],
    bookmarks: dict[str, bool],
    reactions: dict[str, list[str]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    message_id = row.get("message_id")
    if not isinstance(message_id, str):
        return row
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


def _extra_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    skip = _ROW_FIELD_KEYS | _DERIVED_ROW_KEYS
    return {k: v for k, v in row.items() if k not in skip}


class JsonlMessageStore:
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
        files: list[dict[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
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
        if files:
            row["files"] = [dict(f) for f in files]
        if extra:
            overlap = extra.keys() & (_ROW_FIELD_KEYS | _DERIVED_ROW_KEYS)
            if overlap:
                raise ValueError(f"extra contains reserved keys: {sorted(overlap)!r}")
            row.update(extra)
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
            fields: dict[str, Any] = {
                k: current.get(k) for k in ("author", "text", "thread_ts", "status", "task_uuid")
            }
            fields.update(changes)
            return JsonlMessageStore.append(
                self,
                message_id=message_id,
                kind=kind,
                author=str(fields["author"] or ""),
                text=str(fields["text"] or ""),
                thread_ts=fields["thread_ts"],
                status=fields["status"],
                task_uuid=fields["task_uuid"],
                files=current.get("files"),
                extra=_extra_from_row(current),
            )

    def read_from(self, day: date, offset: int = 0) -> tuple[list[tuple[int, dict[str, Any]]], int]:
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
                    _log.warning("[jsonl store] skip broken line in %s", path)
                    continue
                if isinstance(row, dict):
                    out.append((offset, row))
        return out, offset

    def _rows(self, day: date) -> list[dict[str, Any]]:
        return [row for _, row in self.read_from(day)[0]]

    def _all_rows(self) -> list[dict[str, Any]]:
        return [row for day in self.days() for row in self._rows(day)]

    def days(self) -> list[date]:
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
        return self.enrich_many([row])[0]

    def enrich_many(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        bookmarks = self._bookmark_state()
        reactions = self._reaction_state()
        summaries = self._thread_summaries()
        return [_enrich_row(row, bookmarks, reactions, summaries) for row in rows]

    def read(self, day: date) -> list[dict[str, Any]]:
        return self.enrich_many(_latest_per_id(self._rows(day), kinds=MESSAGE_KINDS))

    def read_all(self) -> list[dict[str, Any]]:
        return self.enrich_many(_latest_per_id(self._all_rows(), kinds=MESSAGE_KINDS))

    def recent(self, limit: int, *, before: datetime | None = None) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        rows = [row for row in _latest_per_id(self._all_rows(), kinds=MESSAGE_KINDS) if not row.get("thread_ts")]
        if before is not None:
            rows = [row for row in rows if _posted_before(row, before)]
        return self.enrich_many(rows[-limit:])

    def latest(self, message_id: str) -> dict[str, Any] | None:
        for day in reversed(self.days()):
            found = None
            for row in self._rows(day):
                if row.get("message_id") == message_id and row.get("kind") in MESSAGE_KINDS:
                    found = row
            if found is not None:
                return self.enrich(found)
        return None

    def thread(self, thread_id: str) -> list[dict[str, Any]]:
        return [
            row
            for row in self.read_all()
            if row.get("message_id") == thread_id or row.get("thread_ts") == thread_id
        ]

    def set_bookmark(self, message_id: str, bookmarked: bool) -> dict[str, Any] | None:
        current = self.latest(message_id)
        if current is None or current.get("thread_ts"):
            return None
        return JsonlMessageStore.append(
            self,
            message_id=message_id,
            author=str(current.get("author") or ""),
            text=str(current.get("text") or ""),
            thread_ts=None,
            kind="bookmark",
            bookmarked=bookmarked,
            extra=_extra_from_row(current),
        )

    def add_reaction(self, message_id: str, name: str) -> dict[str, Any] | None:
        current = self.latest(message_id)
        if current is None:
            return None
        return JsonlMessageStore.append(
            self,
            message_id=message_id,
            author=str(current.get("author") or ""),
            text=str(current.get("text") or ""),
            thread_ts=current.get("thread_ts"),
            kind="reaction",
            reaction=name,
            extra=_extra_from_row(current),
        )

    def bookmarks(self) -> list[dict[str, Any]]:
        state = self._bookmark_state()
        bookmarked_ids = [mid for mid, on in state.items() if on]
        if not bookmarked_ids:
            return []
        rows = [row for row in self.read_all() if row.get("message_id") in bookmarked_ids and not row.get("thread_ts")]
        rows.sort(key=lambda row: row.get("ts") or "", reverse=True)
        return rows
