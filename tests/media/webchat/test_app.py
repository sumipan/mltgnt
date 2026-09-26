"""mltgnt.media.webchat.app (#4032): endpoints and SSE over FastAPI's TestClient."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mltgnt.media._core.types import MediaEvent  # noqa: E402
from mltgnt.media.webchat.app import _tail, create_app, parse_event_id, serve  # noqa: E402
from mltgnt.media.webchat.config import WebChatMediaConfig  # noqa: E402
from mltgnt.media.webchat.store import WebChatStore  # noqa: E402
from mltgnt.media.webchat.ui import INDEX_HTML  # noqa: E402

DAY = date(2026, 1, 5)


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


class RecordingBridge:
    def __init__(self, store: WebChatStore | None = None, error: BaseException | None = None) -> None:
        self.events: list[MediaEvent] = []
        self.store = store
        self.error = error

    def handle_event(self, event: MediaEvent) -> None:
        self.events.append(event)
        if self.error is not None:
            raise self.error
        if self.store is not None:
            thread = event.conversation_id.partition(":")[2]
            self.store.append(message_id="r-" + event.message_id, author="bot", text="pong", thread_ts=thread)


def _config(tmp_path: Path) -> WebChatMediaConfig:
    return WebChatMediaConfig(
        state_dir=tmp_path / "s", pending_dir=tmp_path / "p", events_dir=tmp_path / "e", store_dir=tmp_path / "w"
    )


def _app(tmp_path: Path, bridge: Any = None, **kwargs: Any) -> tuple[TestClient, WebChatStore, Any]:
    clock = FakeClock()
    store = WebChatStore(tmp_path / "w", clock=clock)
    bridge = bridge if bridge is not None else RecordingBridge(store)
    app = create_app(_config(tmp_path), bridge, store=store, **kwargs)
    return TestClient(app), store, bridge


def _frames(text: str) -> list[dict[str, str]]:
    out = []
    for block in text.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(": ")
            fields[key] = value
        if "event" in fields:
            out.append(fields)
    return out


def test_index_serves_ascii_html(tmp_path: Path) -> None:
    client, _store, _bridge = _app(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.text == INDEX_HTML
    assert INDEX_HTML.isascii()


def test_post_message_stores_and_runs_the_turn(tmp_path: Path) -> None:
    client, store, bridge = _app(tmp_path)
    resp = client.post("/messages", json={"text": "ping", "author": "u1"})
    assert resp.status_code == 202
    message_id = resp.json()["message_id"]
    assert resp.json() == {"message_id": message_id, "thread_ts": None, "conversation_id": f"webchat:{message_id}"}
    assert [e.text for e in bridge.events] == ["ping"]
    assert bridge.events[0].message_id == message_id
    rows = client.get("/messages").json()
    assert [(r["author"], r["text"], r["thread_ts"]) for r in rows] == [
        ("u1", "ping", None),
        ("bot", "pong", message_id),
    ]
    assert client.get("/messages", params={"day": "2026-01-05"}).json() == rows
    assert client.get("/messages", params={"day": "2026-01-04"}).json() == []


def test_reply_in_thread_and_thread_endpoint(tmp_path: Path) -> None:
    client, _store, bridge = _app(tmp_path)
    root = client.post("/messages", json={"text": "first"}).json()["message_id"]
    resp = client.post("/messages", json={"text": "second", "thread_ts": root})
    assert resp.json()["thread_ts"] == root
    assert bridge.events[1].conversation_id == f"webchat:{root}"
    client.post("/messages", json={"text": "elsewhere"})
    texts = [r["text"] for r in client.get(f"/threads/{root}").json()]
    assert texts == ["first", "pong", "second", "pong"]


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "  "}, {}, ["x"]])
def test_empty_text_is_422_and_store_unchanged(tmp_path: Path, body: Any) -> None:
    client, store, bridge = _app(tmp_path)
    resp = client.post("/messages", json=body)
    assert resp.status_code == 422
    assert bridge.events == []
    assert store.days() == []


def test_non_json_body_is_422(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path)
    resp = client.post("/messages", content=b"not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert store.days() == []


def test_store_failure_is_503(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    bridge = RecordingBridge()
    app = create_app(_config(tmp_path), bridge, store=WebChatStore(blocker / "w"))
    resp = TestClient(app).post("/messages", json={"text": "ping"})
    assert resp.status_code == 503
    assert bridge.events == []


def test_bridge_failure_does_not_break_the_request(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path, bridge=RecordingBridge(error=RuntimeError("boom")))
    resp = client.post("/messages", json={"text": "ping"})
    assert resp.status_code == 202
    assert [r["text"] for r in store.read(DAY)] == ["ping"]


def test_stream_resumes_from_last_event_id(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path, poll_interval_sec=0.01, stream_timeout_sec=0.05)
    store.append(message_id="m1", author="bot", text="v1")
    store.revise("m1", "update", text="v2")
    store.revise("m1", "status", status="done")
    resp = client.get("/stream", headers={"Last-Event-ID": "2026-01-05:0"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _frames(resp.text)
    assert [f["event"] for f in frames] == ["message", "update", "status"]
    assert [json.loads(f["data"])["text"] for f in frames] == ["v1", "v2", "v2"]
    assert all(json.loads(f["data"])["message_id"] == "m1" for f in frames)
    size = store.path_for(DAY).stat().st_size
    assert frames[-1]["id"] == f"2026-01-05:{size}"
    resumed = client.get("/stream", headers={"Last-Event-ID": frames[0]["id"]})
    assert [f["event"] for f in _frames(resumed.text)] == ["update", "status"]


def test_stream_without_last_event_id_starts_at_the_end(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path, poll_interval_sec=0.01, stream_timeout_sec=0.05)
    assert _frames(client.get("/stream").text) == []
    store.append(message_id="m1", author="bot", text="old")
    resp = client.get("/stream", headers={"Last-Event-ID": "garbage"})
    assert resp.text.startswith("retry:")
    assert _frames(resp.text) == []


class _Connected:
    async def is_disconnected(self) -> bool:
        return False


class _Disconnected:
    async def is_disconnected(self) -> bool:
        return True


async def test_tail_follows_the_next_day(tmp_path: Path) -> None:
    clock = FakeClock()
    store = WebChatStore(tmp_path / "w", clock=clock)
    store.append(message_id="m1", author="u", text="day one")
    clock.now += timedelta(days=1)
    store.append(message_id="m2", author="u", text="day two")
    frames = [f async for f in _tail(_Connected(), store, (DAY, 0), 0.0, 0.0)]
    events = _frames("".join(frames))
    assert [json.loads(f["data"])["text"] for f in events] == ["day one", "day two"]
    assert events[1]["id"].startswith("2026-01-06:")


async def test_tail_stops_when_the_client_leaves(tmp_path: Path) -> None:
    store = WebChatStore(tmp_path / "w", clock=FakeClock())
    frames = [f async for f in _tail(_Disconnected(), store, (DAY, 0), 0.0, None)]
    assert frames == ["retry: 1000\n\n"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-01-05:12", (DAY, 12)),
        (None, None),
        ("", None),
        ("2026-01-05", None),
        ("day:1", None),
        ("2026-01-05:x", None),
        ("2026-01-05:-1", None),
    ],
)
def test_parse_event_id(value: str | None, expected: tuple[date, int] | None) -> None:
    assert parse_event_id(value) == expected


def test_serve_binds_config_address(tmp_path: Path) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    app = object()
    serve(app, _config(tmp_path), runner=lambda a, **kw: calls.append((a, kw)))
    assert calls == [(app, {"host": "127.0.0.1", "port": 8765})]
