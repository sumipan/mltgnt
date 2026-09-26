"""Append-only message store: one ``YYYY-MM-DD.jsonl`` file per day, one JSON row per line.

Every row is a full snapshot of one message (``ts`` / ``message_id`` / ``author`` /
``text`` / ``thread_ts`` / ``kind`` / ``status`` / ``task_uuid``). ``kind`` says why
the row was written: ``message`` (new), ``update`` (text changed) or ``status``.
Rows are only appended under ``fcntl.flock``; the latest row of a message_id wins.
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

__all__ = ["KINDS", "WebChatStore"]

_log = logging.getLogger(__name__)

KINDS = ("message", "update", "status")


def _now() -> datetime:
    return datetime.now().astimezone()


def _latest_per_id(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last row of each message_id, ordered by the message's first appearance."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        message_id = row.get("message_id")
        if isinstance(message_id, str) and message_id:
            latest[message_id] = row  # re-assigning an existing key keeps its position
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

    def read(self, day: date) -> list[dict[str, Any]]:
        """Messages of ``day``: the last row of each message_id."""
        return _latest_per_id(self._rows(day))

    def read_all(self) -> list[dict[str, Any]]:
        """Messages of every day: the last row of each message_id."""
        return _latest_per_id(row for day in self.days() for row in self._rows(day))

    def latest(self, message_id: str) -> dict[str, Any] | None:
        """Last row of ``message_id`` searching the newest day first."""
        for day in reversed(self.days()):
            found = None
            for row in self._rows(day):
                if row.get("message_id") == message_id:
                    found = row
            if found is not None:
                return found
        return None

    def thread(self, thread_id: str) -> list[dict[str, Any]]:
        """The thread's root message and its replies."""
        return [
            row for row in self.read_all() if row.get("message_id") == thread_id or row.get("thread_ts") == thread_id
        ]
