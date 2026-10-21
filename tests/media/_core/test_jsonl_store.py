"""JsonlMessageStore: extra fields, day rollover, flock append."""

from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from mltgnt.media._core.jsonl_store import JsonlMessageStore

DAY = date(2026, 1, 5)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _store(tmp_path: Path) -> tuple[JsonlMessageStore, FakeClock]:
    clock = FakeClock(datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc))
    return JsonlMessageStore(tmp_path / "store", clock=clock), clock


def test_append_extra_merged_into_row(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    row = store.append(
        message_id="m1",
        author="host-a/persona-x",
        text="hi",
        extra={"room": "r1", "hop": 2},
    )
    assert row["room"] == "r1"
    assert row["hop"] == 2
    line = json.loads((tmp_path / "store" / "2026-01-05.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["room"] == "r1"
    assert line["hop"] == 2


def test_revise_carries_extra_fields(tmp_path: Path) -> None:
    store, _clock = _store(tmp_path)
    store.append(
        message_id="m1",
        author="host-a/persona-x",
        text="v1",
        extra={"room": "r1", "hop": 2},
    )
    revised = store.revise("m1", "status", status="working")
    assert revised is not None
    assert revised["room"] == "r1"
    assert revised["hop"] == 2
    assert revised["kind"] == "status"
    assert revised["status"] == "working"


def test_append_rolls_to_next_day_file(tmp_path: Path) -> None:
    store, clock = _store(tmp_path)
    store.append(message_id="d1", author="u", text="day1")
    clock.now += timedelta(days=1)
    store.append(message_id="d2", author="u", text="day2")
    assert store.days() == [DAY, DAY + timedelta(days=1)]
    assert (tmp_path / "store" / "2026-01-05.jsonl").is_file()
    assert (tmp_path / "store" / "2026-01-06.jsonl").is_file()
    assert [r["message_id"] for r in store.read(DAY)] == ["d1"]
    assert [r["message_id"] for r in store.read(DAY + timedelta(days=1))] == ["d2"]


def test_concurrent_appends_under_flock(tmp_path: Path) -> None:
    store = JsonlMessageStore(tmp_path / "store")

    def writer(prefix: str) -> None:
        for i in range(50):
            store.append(message_id=f"{prefix}{i}", author=prefix, text="x")

    threads = [threading.Thread(target=writer, args=(p,)) for p in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = store.path_for(store.today()).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
