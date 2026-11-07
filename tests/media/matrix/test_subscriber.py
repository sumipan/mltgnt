"""MatrixSubscriber (#6101)."""

from __future__ import annotations

import json
from pathlib import Path

from mltgnt.media.matrix.client import MatrixClient, MatrixResponse
from mltgnt.media.matrix.config import MatrixMediaConfig
from mltgnt.media.matrix.subscriber import MatrixSubscriber
from tests.media.matrix.fakes import FakeHomeserver


def _config(tmp_path: Path) -> MatrixMediaConfig:
    return MatrixMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        homeserver_url="http://127.0.0.1:8008",
        user_id="@staff:x",
        rooms={"staff": "!a:x"},
        allowed_senders=("@u:x",),
    )


class RecordingHandler:
    def __init__(self) -> None:
        self.events: list[object] = []

    def handle_event(self, event: object) -> None:
        self.events.append(event)


def test_initial_sync_skips_timeline_but_saves_batch(tmp_path: Path) -> None:
    fake = FakeHomeserver()
    sync_body = {
        "next_batch": "s1",
        "rooms": {
            "join": {
                "!a:x": {
                    "timeline": {
                        "events": [
                            {
                                "type": "m.room.message",
                                "event_id": "$m1",
                                "sender": "@u:x",
                                "content": {"msgtype": "m.text", "body": "hi"},
                            }
                        ]
                    }
                }
            }
        },
    }
    fake.queue("GET", "/sync", MatrixResponse(200, sync_body))
    handler = RecordingHandler()
    client = MatrixClient(_config(tmp_path), transport=fake)
    sub = MatrixSubscriber(_config(tmp_path), handler, client=client, transport=fake)
    assert sub.sync_once() == 0
    assert handler.events == []
    state = json.loads((tmp_path / "s" / "matrix_sync.json").read_text(encoding="utf-8"))
    assert state["next_batch"] == "s1"
    first = [c for c in fake.calls if c.path == "/_matrix/client/v3/sync"][0]
    assert first.query is not None
    assert "filter" in first.query


def test_incremental_sync_delivers_one_event(tmp_path: Path) -> None:
    state_dir = tmp_path / "s"
    state_dir.mkdir(parents=True)
    (state_dir / "matrix_sync.json").write_text(json.dumps({"next_batch": "s1", "encrypted_notified": []}))
    fake = FakeHomeserver()
    sync_body = {
        "next_batch": "s2",
        "rooms": {
            "join": {
                "!a:x": {
                    "timeline": {
                        "events": [
                            {
                                "type": "m.room.message",
                                "event_id": "$m2",
                                "sender": "@u:x",
                                "content": {"msgtype": "m.text", "body": "hello"},
                            }
                        ]
                    }
                }
            }
        },
    }
    fake.queue("GET", "/sync", MatrixResponse(200, sync_body))
    handler = RecordingHandler()
    client = MatrixClient(_config(tmp_path), transport=fake)
    sub = MatrixSubscriber(_config(tmp_path), handler, client=client, transport=fake)
    assert sub.sync_once() == 1
    assert len(handler.events) == 1
    call = [c for c in fake.calls if c.path == "/_matrix/client/v3/sync"][-1]
    assert call.query is not None and call.query.get("since") == "s1"


def test_encrypted_room_notice_once(tmp_path: Path) -> None:
    state_dir = tmp_path / "s"
    state_dir.mkdir(parents=True)
    (state_dir / "matrix_sync.json").write_text(json.dumps({"next_batch": "s1", "encrypted_notified": []}))
    fake = FakeHomeserver()
    ev = {"type": "m.room.encrypted", "event_id": "$enc"}
    sync1 = {"next_batch": "s2", "rooms": {"join": {"!a:x": {"timeline": {"events": [ev]}}}}}
    sync2 = {"next_batch": "s3", "rooms": {"join": {"!a:x": {"timeline": {"events": [ev]}}}}}
    fake.queue("GET", "/sync", MatrixResponse(200, sync1))
    fake.queue("GET", "/sync", MatrixResponse(200, sync2))
    handler = RecordingHandler()
    client = MatrixClient(_config(tmp_path), transport=fake)
    sub = MatrixSubscriber(_config(tmp_path), handler, client=client, transport=fake)
    assert sub.sync_once() == 0
    assert handler.events == []
    sends = fake.send_bodies()
    assert len(sends) == 1
    sub.sync_once()
    assert len(fake.send_bodies()) == 1


def test_invite_join_only_when_allowed(tmp_path: Path) -> None:
    state_dir = tmp_path / "s"
    state_dir.mkdir(parents=True)
    (state_dir / "matrix_sync.json").write_text(json.dumps({"next_batch": "s0", "encrypted_notified": []}))
    fake = FakeHomeserver()
    sync_ok = {
        "next_batch": "s1",
        "rooms": {
            "invite": {
                "!a:x": {
                    "invite_state": {
                        "events": [
                            {
                                "type": "m.room.member",
                                "state_key": "@staff:x",
                                "sender": "@u:x",
                                "content": {"membership": "invite"},
                            }
                        ]
                    }
                }
            }
        },
    }
    sync_skip = {
        "next_batch": "s2",
        "rooms": {
            "invite": {
                "!b:x": {
                    "invite_state": {
                        "events": [
                            {
                                "type": "m.room.member",
                                "state_key": "@staff:x",
                                "sender": "@stranger:x",
                                "content": {"membership": "invite"},
                            }
                        ]
                    }
                }
            }
        },
    }
    fake.queue("GET", "/sync", MatrixResponse(200, sync_ok))
    fake.queue("GET", "/sync", MatrixResponse(200, sync_skip))
    handler = RecordingHandler()
    client = MatrixClient(_config(tmp_path), transport=fake)
    sub = MatrixSubscriber(_config(tmp_path), handler, client=client, transport=fake)
    sub.sync_once()
    joins = [c for c in fake.calls if c.method == "POST" and c.path.endswith("/join")]
    assert len(joins) == 1
    fake.calls.clear()
    sub.sync_once()
    assert not [c for c in fake.calls if c.method == "POST" and "/join" in c.path]
