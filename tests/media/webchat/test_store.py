"""mltgnt.media.webchat.store (#4032): append-only day files, latest row wins."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from mltgnt.media.webchat.store import WebChatStore

DAY = date(2026, 1, 5)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _store(tmp_path: Path) -> tuple[WebChatStore, FakeClock]:
    clock = FakeClock(datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc))
    return WebChatStore(tmp_path / "store", clock=clock), clock


def test_append_writes_one_json_line_to_the_day_file(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    row = store.append(message_id="m1", author="u1", text="hello")
    path = tmp_path / "store" / "2026-01-05.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == row
    assert set(row) == {"ts", "message_id", "author", "text", "thread_ts", "kind", "status", "task_uuid"}
    assert row["kind"] == "message"
    assert row["ts"].startswith("2026-01-05T10:00:00")


def test_post_then_update_reads_back_the_latest_row_only(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    store.append(message_id="m1", author="bot", text="v1")
    store.revise("m1", "update", text="v2")
    rows = store.read(DAY)
    assert [(r["message_id"], r["text"], r["kind"]) for r in rows] == [("m1", "v2", "update")]
    assert len((tmp_path / "store" / "2026-01-05.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_revise_keeps_other_fields(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    store.append(message_id="m1", author="bot", text="v1", thread_ts="t1")
    store.revise("m1", "status", status="working")
    store.revise("m1", "update", text="v2")
    row = store.latest("m1")
    assert row is not None
    assert (row["author"], row["text"], row["thread_ts"], row["status"]) == ("bot", "v2", "t1", "working")


def test_revise_unknown_id_is_none(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    assert store.revise("missing", "update", text="x") is None
    assert store.days() == []


def test_read_keeps_first_appearance_order(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    store.append(message_id="a", author="u", text="1")
    store.append(message_id="b", author="u", text="2")
    store.revise("a", "update", text="1b")
    assert [(r["message_id"], r["text"]) for r in store.read(DAY)] == [("a", "1b"), ("b", "2")]


def test_unknown_kind_is_rejected(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    with pytest.raises(ValueError):
        store.append(message_id="m1", author="u", text="x", kind="other")


def test_concurrent_appends_keep_every_line(tmp_path: Path) -> None:
    store = WebChatStore(tmp_path / "store")

    def writer(prefix: str) -> None:
        for i in range(100):
            store.append(message_id=f"{prefix}{i}", author=prefix, text="x" * 200)

    threads = [threading.Thread(target=writer, args=(p,)) for p in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = store.path_for(store.today()).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200
    assert {json.loads(line)["message_id"] for line in lines} == {f"{p}{i}" for p in "ab" for i in range(100)}


def test_read_from_skips_partial_and_broken_lines(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    store.append(message_id="m1", author="u", text="x")
    path = store.path_for(DAY)
    with path.open("a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write('{"message_id": "m2"')
    rows, offset = store.read_from(DAY)
    assert [r["message_id"] for _, r in rows] == ["m1"]
    assert offset == len(path.read_bytes()) - len('{"message_id": "m2"')
    assert store.read_from(DAY, offset) == ([], offset)


def test_read_of_missing_day_is_empty(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    assert store.read(DAY) == []
    assert store.read_from(DAY, 7) == ([], 7)


def test_latest_and_thread_span_days(tmp_path: Path) -> None:
    store, clock = _store(tmp_path)
    store.append(message_id="root", author="u", text="q")
    clock.now += timedelta(days=1)
    store.append(message_id="r1", author="bot", text="a", thread_ts="root")
    store.append(message_id="other", author="u", text="z")
    store.revise("root", "status", status="done")
    (tmp_path / "store" / "notes.jsonl").write_text("", encoding="utf-8")
    assert store.days() == [DAY, DAY + timedelta(days=1)]
    root = store.latest("root")
    assert root is not None and root["status"] == "done"
    assert [r["message_id"] for r in store.thread("root")] == ["root", "r1"]
    assert [r["message_id"] for r in store.read(DAY)] == ["root"]
    assert store.read(DAY)[0]["status"] is None
    assert store.latest("missing") is None
