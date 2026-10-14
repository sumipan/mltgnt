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
from mltgnt.media.webchat.app import _tail, create_app, parse_event_id, safe_asset_path, serve  # noqa: E402
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


def test_config_endpoint_returns_avatars(tmp_path: Path) -> None:
    config = WebChatMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        store_dir=tmp_path / "w",
        avatars={"u1": "avatar.png", "bot": "https://example.com/bot.png"},
        display_names={"u1": "User One"},
        channel_name="audit",
        max_upload_bytes=1024,
    )
    client = TestClient(create_app(config, RecordingBridge()))
    resp = client.get("/config")
    assert resp.status_code == 200
    assert resp.json() == {
        "avatars": {"u1": "avatar.png", "bot": "https://example.com/bot.png"},
        "display_names": {"u1": "User One"},
        "channel_name": "audit",
        "max_upload_bytes": 1024,
    }


def test_assets_serves_direct_file(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "avatar.png").write_bytes(b"PNG")
    config = WebChatMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        store_dir=tmp_path / "w",
        assets_dir=assets,
    )
    client = TestClient(create_app(config, RecordingBridge()))
    resp = client.get("/assets/avatar.png")
    assert resp.status_code == 200
    assert resp.content == b"PNG"


@pytest.mark.parametrize("name", ["../secret", "sub/file.png"])
def test_assets_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    config = WebChatMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        store_dir=tmp_path / "w",
        assets_dir=assets,
    )
    client = TestClient(create_app(config, RecordingBridge()))
    assert client.get(f"/assets/{name}").status_code == 404
    assert safe_asset_path(assets, name) is None


def test_safe_asset_path_rejects_dotdot(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    assert safe_asset_path(assets, "..") is None


def test_bookmark_toggle_and_list(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path)
    root = client.post("/messages", json={"text": "keep"}).json()["message_id"]
    on = client.post(f"/messages/{root}/bookmark", json={"bookmarked": True})
    assert on.status_code == 200
    assert on.json()["bookmarked"] is True
    assert client.get("/bookmarks").json()[0]["message_id"] == root
    off = client.post(f"/messages/{root}/bookmark", json={"bookmarked": False})
    assert off.status_code == 200
    assert off.json()["bookmarked"] is False
    assert client.get("/bookmarks").json() == []


@pytest.mark.parametrize("body", [{"bookmarked": "yes"}, {"flag": True}, {}])
def test_bookmark_rejects_non_bool(tmp_path: Path, body: Any) -> None:
    client, store, _bridge = _app(tmp_path)
    root = store.append(message_id="root", author="u", text="x")["message_id"]
    assert client.post(f"/messages/{root}/bookmark", json=body).status_code == 422


def test_bookmark_unknown_or_reply_is_404(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path)
    root = store.append(message_id="root", author="u", text="x")["message_id"]
    store.append(message_id="reply", author="u", text="r", thread_ts=root)
    assert client.post("/messages/missing/bookmark", json={"bookmarked": True}).status_code == 404
    assert client.post("/messages/reply/bookmark", json={"bookmarked": True}).status_code == 404


def test_messages_include_reply_count(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path)
    root = store.append(message_id="root", author="u", text="q")["message_id"]
    store.append(message_id="r1", author="bot", text="a", thread_ts=root)
    rows = client.get("/messages").json()
    root_row = next(r for r in rows if r["message_id"] == "root")
    assert root_row["reply_count"] == 1
    assert root_row["reactions"] == []
    assert root_row["bookmarked"] is False


def test_stream_emits_bookmark_and_reaction(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path, poll_interval_sec=0.01, stream_timeout_sec=0.05)
    root = store.append(message_id="root", author="u", text="x")["message_id"]
    store.set_bookmark(root, True)
    store.add_reaction(root, "bulb")
    resp = client.get("/stream", headers={"Last-Event-ID": "2026-01-05:0"})
    events = [f["event"] for f in _frames(resp.text)]
    assert "bookmark" in events
    assert "reaction" in events


def test_ui_uses_container_scoped_message_ids_and_safe_links() -> None:
    assert 'var id = container.id + "-m-" + row.message_id' in INDEX_HTML
    assert 'nodes[i].id === "log-m-" + threadTs' in INDEX_HTML
    assert "panel-body-m-" in INDEX_HTML
    assert "noopener noreferrer" in INDEX_HTML
    assert "mailto:" in INDEX_HTML


def test_stream_status_event_includes_enriched_thread_fields(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path, poll_interval_sec=0.01, stream_timeout_sec=0.05)
    store.append(message_id="root", author="u", text="q")
    store.append(message_id="r1", author="a", text="a", thread_ts="root")
    store.revise("root", "status", status="done")
    resp = client.get("/stream", headers={"Last-Event-ID": "2026-01-05:0"})
    status_frames = [f for f in _frames(resp.text) if f["event"] == "status"]
    assert len(status_frames) == 1
    data = json.loads(status_frames[0]["data"])
    assert data["message_id"] == "root"
    assert data["reply_count"] == 1
    assert data["last_reply_ts"] == store.latest("r1")["ts"]
    assert data["participants"] == ["a"]
    assert data["reactions"] == []
    assert data["bookmarked"] is False


def test_stream_reaction_event_includes_enriched_reactions(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path, poll_interval_sec=0.01, stream_timeout_sec=0.05)
    root = store.append(message_id="root", author="u", text="x")["message_id"]
    store.add_reaction(root, "bulb")
    resp = client.get("/stream", headers={"Last-Event-ID": "2026-01-05:0"})
    reaction_frames = [f for f in _frames(resp.text) if f["event"] == "reaction"]
    assert len(reaction_frames) == 1
    data = json.loads(reaction_frames[0]["data"])
    assert data["kind"] == "reaction"
    assert data["reactions"] == ["bulb"]


def test_index_html_slack_ui_elements() -> None:
    for token in (
        'id="panel-form"',
        'id="panel-text"',
        'id="saved-link"',
        ".thread-summary",
        ".chip",
        ".save",
        ".state",
        ".divider",
    ):
        assert token in INDEX_HTML


def test_index_html_removed_legacy_elements() -> None:
    for token in (
        'id="tabs"',
        'id="tab-chat"',
        'id="tab-bookmarks"',
        '"unbookmark"',
        "renderToolbar",
        ".status.done",
        "base.reactions.push",
    ):
        assert token not in INDEX_HTML


PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def test_upload_then_post_message_with_files(tmp_path: Path) -> None:
    client, store, bridge = _app(tmp_path)
    resp = client.post("/files?name=shot.png", content=PNG, headers={"Content-Type": "image/png"})
    assert resp.status_code == 201
    entry = resp.json()
    assert entry["name"] == "shot.png"
    assert entry["content_type"] == "image/png"
    assert entry["size"] == len(PNG)
    got = client.get("/files/" + entry["id"])
    assert got.status_code == 200
    assert got.content == PNG
    assert got.headers["content-type"] == "image/png"

    posted = client.post("/messages", json={"text": "", "files": [{"id": entry["id"], "name": "shot.png"}]})
    assert posted.status_code == 202
    message_id = posted.json()["message_id"]
    row = store.latest(message_id)
    assert row is not None
    assert row["files"] == [{"id": entry["id"], "name": "shot.png", "content_type": "image/png", "size": len(PNG)}]
    (event,) = bridge.events
    assert event.text == ""
    (attachment,) = event.attachments
    assert attachment.name == "shot.png"
    assert attachment.content_type == "image/png"
    assert attachment.uri == str(tmp_path / "files" / entry["id"])


def test_files_survive_status_revision(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path)
    entry = client.post("/files", content=PNG, headers={"Content-Type": "image/png"}).json()
    message_id = client.post("/messages", json={"text": "see", "files": [{"id": entry["id"]}]}).json()["message_id"]
    store.revise(message_id, "status", status="done")
    row = store.latest(message_id)
    assert row is not None and row["status"] == "done"
    assert [f["id"] for f in row["files"]] == [entry["id"]]


@pytest.mark.parametrize(
    ("content", "ctype", "status"),
    [(PNG, "text/plain", 415), (b"", "image/png", 422), (b"0" * 2048, "image/png", 413)],
)
def test_upload_rejects_bad_files(tmp_path: Path, content: bytes, ctype: str, status: int) -> None:
    config = WebChatMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        store_dir=tmp_path / "w",
        max_upload_bytes=1024,
    )
    client = TestClient(create_app(config, RecordingBridge()))
    resp = client.post("/files", content=content, headers={"Content-Type": ctype})
    assert resp.status_code == status
    assert not (tmp_path / "files").exists() or not any((tmp_path / "files").rglob("*.png"))


@pytest.mark.parametrize(
    "files",
    [[{"id": "2026-01-05/" + "0" * 32 + ".png"}], [{"id": "../w/2026-01-05.jsonl"}], "x", [1]],
)
def test_post_message_rejects_unknown_files(tmp_path: Path, files: Any) -> None:
    client, store, bridge = _app(tmp_path)
    resp = client.post("/messages", json={"text": "hi", "files": files})
    assert resp.status_code == 422
    assert store.read(DAY) == []
    assert bridge.events == []


@pytest.mark.parametrize("path", ["/files/2026-01-05/nothing.png", "/files/..%2F..%2Fetc/passwd", "/files/x/y"])
def test_get_file_404_for_unknown_or_unsafe(tmp_path: Path, path: str) -> None:
    client, _store, _bridge = _app(tmp_path)
    assert client.get(path).status_code == 404


def test_messages_limit_returns_recent_roots_across_days(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path)
    clock = FakeClock()
    store._clock = clock
    store.append(message_id="d1", author="u", text="day one")
    clock.now = clock.now + timedelta(days=1)
    store.append(message_id="d2a", author="u", text="day two a")
    store.append(message_id="d2r", author="a", text="reply", thread_ts="d2a")
    store.append(message_id="d2b", author="u", text="day two b")
    rows = client.get("/messages?limit=2").json()
    assert [r["message_id"] for r in rows] == ["d2a", "d2b"]
    assert rows[0]["reply_count"] == 1
    assert [r["message_id"] for r in client.get("/messages?limit=10").json()] == ["d1", "d2a", "d2b"]


def test_ui_supports_avatar_urls_and_image_upload() -> None:
    for token in ("https?:\\/\\/|data:image\\/", 'fetch("files?name="', 'id="file-input"', "addEventListener(\"paste\"", "e.isComposing"):
        assert token in INDEX_HTML


def test_messages_before_pages_back_through_days(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path)
    clock = FakeClock()
    store._clock = clock
    for i in range(5):
        store.append(message_id=f"m{i}", author="u", text=str(i))
        clock.now = clock.now + timedelta(hours=12)
    page = client.get("/messages?limit=2").json()
    assert [r["message_id"] for r in page] == ["m3", "m4"]
    older = client.get("/messages", params={"limit": 2, "before": page[0]["ts"]}).json()
    assert [r["message_id"] for r in older] == ["m1", "m2"]
    oldest = client.get("/messages", params={"limit": 2, "before": older[0]["ts"]}).json()
    assert [r["message_id"] for r in oldest] == ["m0"]


def test_messages_before_accepts_plus_decoded_as_space(tmp_path: Path) -> None:
    client, store, _bridge = _app(tmp_path)
    store.append(message_id="m0", author="u", text="x")
    resp = client.get("/messages?limit=5&before=2099-01-01T00:00:00 09:00")
    assert [r["message_id"] for r in resp.json()] == ["m0"]


def test_messages_before_rejects_bad_timestamp(tmp_path: Path) -> None:
    client, _store, _bridge = _app(tmp_path)
    assert client.get("/messages?limit=5&before=yesterday").status_code == 422
