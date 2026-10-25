"""Hub FastAPI server (TestClient)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mltgnt.exceptions import ConfigError  # noqa: E402
from mltgnt.media._core.jsonl_store import JsonlMessageStore  # noqa: E402
from mltgnt.media.hub.guard import GuardConfig  # noqa: E402
from mltgnt.media.hub.server import (  # noqa: E402
    HubServerConfig,
    create_app,
    load_server_config,
    parse_cursor,
    purge_expired_daily_files,
)

DAY = date(2026, 1, 5)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _config(tmp_path: Path, *, rooms: tuple[str, ...] = ("lobby",)) -> HubServerConfig:
    return HubServerConfig(
        store_dir=tmp_path / "store",
        rooms=rooms,
        token_env={"host-a": "TOKEN_A", "host-b": "TOKEN_B"},
        guard=GuardConfig(hop_limit=3, max_agent_run=4, min_interval_sec=5.0, room_daily_limit=200),
        ttl_days=7,
    )


def _client(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    rooms: tuple[str, ...] = ("lobby",),
) -> TestClient:
    if clock is None:
        clock = FakeClock(datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc))
    app = create_app(
        _config(tmp_path, rooms=rooms),
        environ={"TOKEN_A": "secret-a", "TOKEN_B": "secret-b"},
        clock=clock,
        poll_interval_sec=0.01,
        stream_timeout_sec=0.05,
    )
    return TestClient(app)


def _auth(token: str = "secret-a") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _post_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "msg-1",
        "room": "lobby",
        "sender": {"host": "host-a"},
        "text": "hello",
        "hop": 0,
        "kind": "message",
    }
    body.update(overrides)
    return body


def test_health_without_auth(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["rooms"] == ["lobby"]
    assert "version" in resp.json()


def test_missing_token_is_401(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/rooms/lobby/messages", json=_post_body()).status_code == 401


def test_invalid_token_is_401(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/rooms/lobby/messages", json=_post_body(), headers=_auth("nope")).status_code == 401


def test_compare_digest_used(tmp_path: Path) -> None:
    client = _client(tmp_path)
    with patch("secrets.compare_digest", return_value=False) as digest:
        resp = client.post("/rooms/lobby/messages", json=_post_body(), headers=_auth())
    assert resp.status_code == 401
    digest.assert_called()


def test_sender_host_mismatch_is_403(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = _post_body(sender={"host": "host-b"})
    assert client.post("/rooms/lobby/messages", json=body, headers=_auth("secret-a")).status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        _post_body(sender={"host": "host-a", "extra": True}),
        _post_body(attachments=[{"name": "a.txt", "extra": True}]),
    ],
)
def test_post_rejects_extra_nested_fields(tmp_path: Path, body: dict[str, object]) -> None:
    client = _client(tmp_path)
    assert client.post("/rooms/lobby/messages", json=body, headers=_auth()).status_code == 422


def test_unknown_room_is_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/rooms/missing/messages", headers=_auth()).status_code == 404


def test_store_read_error_is_503(tmp_path: Path) -> None:
    client = _client(tmp_path)
    with patch.object(JsonlMessageStore, "read_from", side_effect=OSError("disk unavailable")):
        resp = client.get("/rooms/lobby/messages", headers=_auth())
    assert resp.status_code == 503
    assert resp.json()["detail"] == "store unavailable"


def test_post_creates_message(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.post("/rooms/lobby/messages", json=_post_body(), headers=_auth())
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == "msg-1"
    assert data["ts"]


def test_post_stores_display_and_rejects_invalid(tmp_path: Path) -> None:
    client = _client(tmp_path)
    display = {"name": "Staff A", "icon_url": "https://example.com/a.png"}
    resp = client.post("/rooms/lobby/messages", json=_post_body(display=display), headers=_auth())
    assert resp.status_code == 201
    rows = client.get("/rooms/lobby/messages", headers=_auth()).json()
    rows = rows["messages"] if isinstance(rows, dict) else rows
    assert rows[-1]["display"] == display
    again = client.post("/rooms/lobby/messages", json=_post_body(display=display), headers=_auth())
    assert again.status_code == 200
    changed = client.post(
        "/rooms/lobby/messages",
        json=_post_body(display={"name": "Other"}),
        headers=_auth(),
    )
    assert changed.status_code == 409
    bad = client.post(
        "/rooms/lobby/messages",
        json=_post_body(id="msg-2", display={"icon_url": "javascript:x"}),
        headers=_auth(),
    )
    assert bad.status_code == 422


def test_idempotent_resend_returns_200(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = _post_body()
    assert client.post("/rooms/lobby/messages", json=body, headers=_auth()).status_code == 201
    again = client.post("/rooms/lobby/messages", json=body, headers=_auth())
    assert again.status_code == 200
    lines = (tmp_path / "store" / "lobby" / "2026-01-05.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_id_conflict_returns_409(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/rooms/lobby/messages", json=_post_body(), headers=_auth()).status_code == 201
    conflict = client.post(
        "/rooms/lobby/messages",
        json=_post_body(text="other"),
        headers=_auth(),
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "id_conflict"


def test_guard_rejection_returns_409(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = _post_body(hop=9)
    resp = client.post("/rooms/lobby/messages", json=body, headers=_auth())
    assert resp.status_code == 409
    assert resp.json()["detail"]["reason"] == "hop_exceeded"


def test_root_post_after_persona_exchange_is_accepted(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc))
    client = _client(tmp_path, clock=clock)
    for i in range(4):
        clock.now += timedelta(seconds=10)
        token, host = ("secret-a", "host-a") if i % 2 == 0 else ("secret-b", "host-b")
        body = _post_body(id=f"m{i}" if i else "root", sender={"host": host, "persona": "p"}, thread="root" if i else None)
        assert client.post("/rooms/lobby/messages", json=body, headers=_auth(token)).status_code == 201
    clock.now += timedelta(seconds=10)
    new_root = _post_body(id="m-new", sender={"host": "host-b", "persona": "p"})
    assert client.post("/rooms/lobby/messages", json=new_root, headers=_auth("secret-b")).status_code == 201
    clock.now += timedelta(seconds=10)
    in_thread = _post_body(id="m-5", sender={"host": "host-a", "persona": "p"}, thread="root")
    resp = client.post("/rooms/lobby/messages", json=in_thread, headers=_auth())
    assert resp.status_code == 409
    assert resp.json()["detail"]["reason"] == "agent_run_exceeded"


def test_patch_update_and_status(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/rooms/lobby/messages", json=_post_body(), headers=_auth()).status_code == 201
    upd = client.patch(
        "/rooms/lobby/messages/msg-1",
        json={"kind": "update", "text": "revised"},
        headers=_auth(),
    )
    assert upd.status_code == 200
    st = client.patch(
        "/rooms/lobby/messages/msg-1",
        json={"kind": "status", "status": "done"},
        headers=_auth(),
    )
    assert st.status_code == 200
    wrong = client.patch(
        "/rooms/lobby/messages/msg-1",
        json={"kind": "update", "text": "nope"},
        headers=_auth("secret-b"),
    )
    assert wrong.status_code == 403


def test_patch_unknown_id_is_404(tmp_path: Path) -> None:
    client = _client(tmp_path)
    resp = client.patch(
        "/rooms/lobby/messages/missing",
        json={"kind": "update", "text": "x"},
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_patch_rejects_unknown_status(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/rooms/lobby/messages", json=_post_body(), headers=_auth()).status_code == 201
    resp = client.patch(
        "/rooms/lobby/messages/msg-1",
        json={"kind": "status", "status": "not-a-status"},
        headers=_auth(),
    )
    assert resp.status_code == 422


def test_list_messages_cursor_and_limits(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc))
    client = _client(tmp_path, clock=clock)
    for idx in range(3):
        body = _post_body(id=f"m{idx}", text=f"t{idx}")
        client.post("/rooms/lobby/messages", json=body, headers=_auth())
        clock.now += timedelta(seconds=6)
    first = client.get("/rooms/lobby/messages", params={"limit": 2}, headers=_auth())
    assert first.status_code == 200
    payload = first.json()
    assert len(payload["messages"]) == 2
    cursor = payload["cursor"]
    second = client.get("/rooms/lobby/messages", params={"after": cursor, "limit": 10}, headers=_auth())
    assert len(second.json()["messages"]) == 1
    assert client.get("/rooms/lobby/messages", params={"limit": 0}, headers=_auth()).status_code == 422
    assert client.get("/rooms/lobby/messages", params={"limit": 1001}, headers=_auth()).status_code == 422
    assert client.get("/rooms/lobby/messages", params={"after": "bad"}, headers=_auth()).status_code == 422


def test_guard_context_and_thread(tmp_path: Path) -> None:
    client = _client(tmp_path)
    root = client.get("/rooms/lobby/guard-context", headers=_auth())
    assert root.status_code == 200
    assert root.json()["thread_rows"] == []
    assert root.json()["room_today_count"] == 0
    client.post("/rooms/lobby/messages", json=_post_body(id="root"), headers=_auth())
    thread = client.get("/rooms/lobby/guard-context", params={"thread": "root"}, headers=_auth())
    assert thread.json()["room_today_count"] == 1
    rows = client.get("/rooms/lobby/threads/root", headers=_auth()).json()
    assert rows[0]["message_id"] == "root"


def test_stream_sse_and_invalid_cursor(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/rooms/lobby/messages", json=_post_body(), headers=_auth())
    resp = client.get("/rooms/lobby/stream", headers={**_auth(), "Last-Event-ID": "2026-01-05:0"})
    assert resp.status_code == 200
    assert "event: message" in resp.text
    assert client.get("/rooms/lobby/stream", headers={**_auth(), "Last-Event-ID": "bad"}).status_code == 422


def test_ttl_purge_removes_only_expired_daily_files(tmp_path: Path) -> None:
    room_dir = tmp_path / "store" / "lobby"
    room_dir.mkdir(parents=True)
    old = room_dir / "2020-01-01.jsonl"
    old.write_text('{"x":1}\n', encoding="utf-8")
    keep = room_dir / "2026-01-04.jsonl"
    keep.write_text('{"x":2}\n', encoding="utf-8")
    other = room_dir / "notes.txt"
    other.write_text("stay", encoding="utf-8")
    sub = room_dir / "nested"
    sub.mkdir()
    purge_expired_daily_files(room_dir, ttl_days=7, today=DAY)
    assert not old.exists()
    assert keep.exists()
    assert other.exists()
    assert sub.is_dir()


def test_load_server_config_rejects_bad_room_and_ttl(tmp_path: Path) -> None:
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
store_dir: /tmp/hub
rooms: ["../escape"]
ttl_days: 0
token_env:
  host-a: TOKEN_A
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_server_config(cfg, environ={"TOKEN_A": "x"})
    cfg.write_text(
        """
store_dir: /tmp/hub
rooms: ["lobby", "lobby"]
token_env:
  host-a: TOKEN_A
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_server_config(cfg, environ={"TOKEN_A": "x"})


def test_load_server_config_requires_token_env(tmp_path: Path) -> None:
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        """
store_dir: /tmp/hub
rooms: ["lobby"]
token_env:
  host-a: TOKEN_A
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="TOKEN_A"):
        load_server_config(cfg, environ={})


@pytest.mark.parametrize(
    "guard",
    [
        "hop_limit: -1",
        "max_agent_run: 0",
        "min_interval_sec: -0.1",
        "room_daily_limit: 0",
    ],
)
def test_load_server_config_rejects_invalid_guard_values(tmp_path: Path, guard: str) -> None:
    cfg = tmp_path / "hub.yaml"
    cfg.write_text(
        f"""
store_dir: /tmp/hub
rooms: ["lobby"]
token_env:
  host-a: TOKEN_A
guard:
  {guard}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="guard"):
        load_server_config(cfg, environ={"TOKEN_A": "x"})


def test_parse_cursor() -> None:
    assert parse_cursor("2026-01-05:0") == (DAY, 0)
    assert parse_cursor("garbage") is None


def test_messages_follow_next_day(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 1, 5, 23, 59, tzinfo=timezone.utc))
    client = _client(tmp_path, clock=clock)
    client.post("/rooms/lobby/messages", json=_post_body(id="d1"), headers=_auth())
    clock.now = datetime(2026, 1, 6, 0, 1, tzinfo=timezone.utc)
    client.post("/rooms/lobby/messages", json=_post_body(id="d2"), headers=_auth())
    listed = client.get(
        "/rooms/lobby/messages",
        params={"after": "2026-01-05:0", "limit": 10},
        headers=_auth(),
    ).json()
    assert {row["message_id"] for row in listed["messages"]} == {"d1", "d2"}


def test_messages_do_not_skip_intermediate_days(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 1, 5, 23, 59, tzinfo=timezone.utc))
    client = _client(tmp_path, clock=clock)
    client.post("/rooms/lobby/messages", json=_post_body(id="d1"), headers=_auth())
    clock.now = datetime(2026, 1, 6, 12, 0, tzinfo=timezone.utc)
    client.post("/rooms/lobby/messages", json=_post_body(id="d2"), headers=_auth())
    clock.now = datetime(2026, 1, 7, 0, 1, tzinfo=timezone.utc)
    client.post("/rooms/lobby/messages", json=_post_body(id="d3"), headers=_auth())

    listed = client.get(
        "/rooms/lobby/messages",
        params={"after": "2026-01-05:0", "limit": 10},
        headers=_auth(),
    ).json()

    assert [row["message_id"] for row in listed["messages"]] == ["d1", "d2", "d3"]
